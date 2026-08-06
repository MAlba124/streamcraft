//! Depayload elements: whole RTP packets (from `rtpsession`) in, codec access
//! units out. Thin wrappers — the payload-format logic is
//! [`crate::depay::h264`] / [`crate::depay::opus`]; these own pads, pts, and
//! loss signalling.
//!
//! pts: every packet of an access unit shares its RTP timestamp (RFC 6184
//! §5.1), and the session derives pts from that timestamp — so the completing
//! packet's pts *is* the AU's pts, no bookkeeping needed.
//!
//! Loss: the session emits in sequence order and releases gaps after the
//! jitter latency; a non-consecutive sequence number here therefore means a
//! real loss — the depacketizer drops its partial AU and resynchronizes
//! (RFC 6184 §5.8: an incomplete FU-A run must not emit).

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use crate::depay::h264::H264Depay;
use crate::packet::RtpPacket;

static RTP_OFFERS: [OfferDesc; 1] = [OfferDesc::any("rtp")];
static H264_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static OPUS_OFFERS: [OfferDesc; 1] = [OfferDesc::any("opus")];

const SRC: PadId = PadId(1);

static H264_PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &RTP_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &H264_OFFERS, dynamic: false, validate: None },
];
static H264_DESC: ElementDesc = ElementDesc {
    name: "rtph264depay",
    pads: &H264_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

static OPUS_PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &RTP_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &OPUS_OFFERS, dynamic: false, validate: None },
];
static OPUS_DESC: ElementDesc = ElementDesc {
    name: "rtpopusdepay",
    pads: &OPUS_PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// Track packet-to-packet sequence continuity (the session already reordered;
/// a gap here is a declared loss).
#[derive(Default)]
struct SeqWatch {
    last: Option<u16>,
}

impl SeqWatch {
    /// Feed the next seq; `true` = contiguous with the previous packet.
    fn contiguous(&mut self, seq: u16) -> bool {
        let ok = match self.last {
            None => true,
            Some(last) => seq == last.wrapping_add(1),
        };
        self.last = Some(seq);
        ok
    }
}

/// RFC 6184 depayload element: RTP packets → Annex-B access units.
pub struct RtpH264Depay {
    depay: H264Depay,
    seq: SeqWatch,
    /// Pool-dry carry: completed AUs awaiting output slots, with their pts.
    pending: std::collections::VecDeque<(Vec<u8>, Timestamp)>,
    /// Depacketize errors since the last warning (loss-driven resyncs are
    /// routine on a lossy path — count, warn sparsely, never spam per packet).
    errors: u64,
}

impl RtpH264Depay {
    /// `sprop`: out-of-band parameter sets from the SDP fmtp
    /// (`sprop-parameter-sets`, RFC 6184 §8.1), already base64-decoded.
    pub fn new(sprop: Vec<Vec<u8>>) -> RtpH264Depay {
        RtpH264Depay {
            depay: H264Depay::new(sprop),
            seq: SeqWatch::default(),
            pending: std::collections::VecDeque::new(),
            errors: 0,
        }
    }

    /// Emit carried AUs while slots free; `false` = stalled (carry holds).
    fn flush_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some((au, pts)) = self.pending.pop_front() {
            let Some(mut buf) = ctx.try_alloc(SRC) else {
                self.pending.push_front((au, pts));
                return false;
            };
            let n = au.len().min(buf.memory.capacity());
            buf.memory.as_mut_full()[..n].copy_from_slice(&au[..n]);
            buf.memory.set_len(n);
            buf.pts = pts;
            ctx.out(SRC).push(buf);
        }
        true
    }
}

impl Element for RtpH264Depay {
    fn desc(&self) -> &'static ElementDesc {
        &H264_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.flush_pending(ctx) {
            return Ok(Flow::Ok); // pool dry — leave input in the ring
        }
        while let Some(buf) = inputs.pop() {
            let Ok(p) = RtpPacket::parse(buf.memory.data()) else { continue };
            if !self.seq.contiguous(p.seq()) {
                self.depay.discontinuity();
            }
            match self.depay.push(p.payload(), p.marker(), p.timestamp()) {
                Ok(aus) => {
                    for au in aus {
                        self.pending.push_back((au, buf.pts));
                    }
                    if !self.flush_pending(ctx) {
                        return Ok(Flow::Ok);
                    }
                }
                // The depacketizer resynchronizes itself; an interleaved-mode
                // sender (Unsupported) is a session-level mismatch — warn once.
                Err(e) => {
                    self.errors += 1;
                    if self.errors == 1 {
                        let element = ctx.element();
                        ctx.post(profluens_core::bus::BusMessage::Warning {
                            element,
                            error: Error::Element {
                                element,
                                message: format!("rtph264depay: {e:?} — resynchronizing"),
                            },
                        });
                    }
                }
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::Eos) {
            // Drain the final (marker-less tail) access unit, exact-size: no
            // slot pressure matters at end of stream.
            if let Some(au) = self.depay.flush() {
                self.pending.push_back((au, Timestamp::NONE));
                let _ = self.flush_pending(ctx);
            }
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// RFC 7587 depayload element: one RTP packet = one Opus packet.
pub struct RtpOpusDepay {
    seq: SeqWatch,
    /// Pool-dry carry (a popped packet must never be dropped for backpressure).
    pending: Option<(Vec<u8>, Timestamp)>,
}

impl RtpOpusDepay {
    pub fn new() -> RtpOpusDepay {
        RtpOpusDepay { seq: SeqWatch::default(), pending: None }
    }

    /// Emit the carried packet if a slot frees; `false` = still stalled.
    fn flush_pending(&mut self, ctx: &mut Ctx) -> bool {
        let Some((pkt, pts)) = self.pending.take() else { return true };
        let Some(mut out) = ctx.try_alloc(SRC) else {
            self.pending = Some((pkt, pts));
            return false;
        };
        let n = pkt.len().min(out.memory.capacity());
        out.memory.as_mut_full()[..n].copy_from_slice(&pkt[..n]);
        out.memory.set_len(n);
        out.pts = pts;
        ctx.out(SRC).push(out);
        true
    }
}

impl Default for RtpOpusDepay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpOpusDepay {
    fn desc(&self) -> &'static ElementDesc {
        &OPUS_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.flush_pending(ctx) {
            return Ok(Flow::Ok); // pool dry — leave input in the ring
        }
        while let Some(buf) = inputs.pop() {
            let Ok(p) = RtpPacket::parse(buf.memory.data()) else { continue };
            // A gap is fine here — each packet is independently one Opus
            // packet (RFC 7587 §4.2); the decoder's PLC owns concealment.
            let _ = self.seq.contiguous(p.seq());
            self.pending = Some((crate::depay::opus::depay(p.payload()), buf.pts));
            if !self.flush_pending(ctx) {
                return Ok(Flow::Ok);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
