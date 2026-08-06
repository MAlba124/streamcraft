//! `rtpsession` — the receive-side RTP session (RFC 3550 §6/§8, receiver
//! half): demux arriving datagrams by payload type, reorder/dejitter each
//! stream, and emit **whole RTP packets** in sequence order on one dynamic src
//! pad per stream, pts-mapped onto the pipeline's running time.
//!
//! Division of labour (see the crate docs): this element owns *time* and
//! *topology* — arrival stamps in, `pts` out, one pad per SDP media stream —
//! while the packet bytes stay opaque here beyond the fixed header. The
//! depayloaders downstream re-parse the packet (via [`crate::packet`], cheap)
//! and own the payload-format logic.
//!
//! Streams are declared **out-of-band** (from the SDP: payload type + clock
//! rate — RFC 8866 `rtpmap`), handed to the constructor; pads exist at preroll
//! (spec: dynamic pads — the topology settles before data, no data-driven pad
//! surprises on a live source).
//!
//! ## Timing model (v1)
//! - Arrival running time rides in on each datagram's `pts` (stamped by
//!   `udpsrc`); the jitter buffer holds packets up to [`SESSION_LATENCY`]
//!   before declaring gaps lost (RFC 3550 §6.4.1's model; A.8 estimator).
//! - Output `pts`: per-stream anchor — the first popped packet maps its RTP
//!   timestamp to `arrival + latency`, later packets follow the RTP timestamp
//!   delta at the stream's clock rate (unwrapped 32-bit ts, §5.1). Inter-stream
//!   (A/V) alignment from RTCP SR NTP mappings is parsed and stashed but not
//!   yet applied — a named follow-up; per-stream pacing is exact without it.
//! - The declared latency is the static [`SESSION_LATENCY`] (an `ElementDesc`
//!   is `'static`; per-instance declared latency is a framework follow-up).
//!
//! ## RTCP (v1)
//! A second sink pad accepts the paired RTCP datagrams (link a second
//! `udpsrc` on the odd port, RFC 3550 §11); SRs are parsed and stashed for the
//! sync follow-up. Receiver reports are not yet sent (needs a send path on the
//! socket — follow-up with the send elements).

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

use crate::jitter::{Held, JitterBuffer};
use crate::packet::RtpPacket;
use crate::rtcp::{self, SenderInfo};

/// The reorder/dejitter hold. 100 ms covers LAN/localhost and modest WAN
/// jitter; it is also the declared path latency sinks schedule around.
pub const SESSION_LATENCY: Timestamp = Timestamp::from_millis(100);

static DATAGRAM_OFFERS: [OfferDesc; 1] = [OfferDesc::any("datagram")];
/// Src pads carry whole RTP packets; the depayloaders offer this family.
static RTP_OFFERS: [OfferDesc; 1] = [OfferDesc::any("rtp")];

const RTP_SINK: PadId = PadId(0);
const RTCP_SINK: PadId = PadId(1);

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &DATAGRAM_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "rtcp",
        direction: Direction::Sink,
        offers: &DATAGRAM_OFFERS,
        dynamic: false,
        validate: None,
    },
];

static DESC: ElementDesc = ElementDesc {
    name: "rtpsession",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    // Consume whichever sink has data — RTCP is sparse and optional.
    inputs: InputPolicy::Any,
    latency: LatencyDesc {
        min: SESSION_LATENCY,
        max: SESSION_LATENCY,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// One declared stream (from the SDP media section's `rtpmap`).
#[derive(Clone, Copy, Debug)]
pub struct RtpStreamDesc {
    /// The (usually dynamic, 96–127) payload type this stream uses.
    pub payload_type: u8,
    /// The RTP timestamp clock rate (RFC 8866 `rtpmap` rate; e.g. 90000 for
    /// H.264 per RFC 6184 §8.1, 48000 for Opus per RFC 7587 §4.1).
    pub clock_rate: u32,
}

/// Per-stream receive state.
struct Stream {
    desc: RtpStreamDesc,
    pad: PadId,
    jitter: JitterBuffer,
    /// Current SSRC — a change is a stream restart (RFC 3550 §8.2): reset.
    ssrc: Option<u32>,
    /// Unwrapped RTP timestamp state: (last raw ts, extended ts).
    ts_unwrap: Option<(u32, u64)>,
    /// `(extended ts, pts)` of the anchor packet — later packets follow the
    /// timestamp delta at `clock_rate`.
    anchor: Option<(u64, Timestamp)>,
    /// Pool-dry carry: a popped packet awaiting an output slot.
    pending: Option<Held>,
    /// Latest sender report for this SSRC (sync follow-up).
    last_sr: Option<SenderInfo>,
}

/// The receive-side session element. See the module docs.
pub struct RtpSession {
    descs: Vec<RtpStreamDesc>,
    streams: Vec<Stream>,
    /// Datagrams that matched no declared payload type (counted, dropped).
    unknown_pt: u64,
}

impl RtpSession {
    /// A session expecting the given streams (one src pad each, added at
    /// preroll, named `src_pt<N>`).
    // COLD: one-time constructor; `streams` is populated at preroll, not per packet.
    #[allow(clippy::disallowed_methods)]
    pub fn new(descs: Vec<RtpStreamDesc>) -> RtpSession {
        RtpSession { descs, streams: Vec::new(), unknown_pt: 0 }
    }

    /// Unwrap the 32-bit RTP timestamp against the stream's last (§5.1 —
    /// timestamps wrap; deltas are signed and small between adjacent packets).
    fn extend_ts(stream: &mut Stream, ts: u32) -> u64 {
        let ext = match stream.ts_unwrap {
            None => ts as u64,
            Some((last, last_ext)) => {
                let delta = ts.wrapping_sub(last) as i32 as i64;
                last_ext.wrapping_add_signed(delta)
            }
        };
        stream.ts_unwrap = Some((ts, ext));
        ext
    }

    /// Route one RTP datagram into its stream's jitter buffer.
    fn ingest_rtp(&mut self, datagram: &[u8], arrival: Timestamp) {
        let Ok(p) = RtpPacket::parse(datagram) else {
            self.unknown_pt += 1; // not RTP — count with the strays
            return;
        };
        let Some(stream) = self.streams.iter_mut().find(|s| s.desc.payload_type == p.payload_type())
        else {
            self.unknown_pt += 1;
            return;
        };
        // SSRC change = restart (§8.2): drop reorder state, keep the pad.
        if stream.ssrc != Some(p.ssrc()) {
            if stream.ssrc.is_some() {
                stream.jitter = JitterBuffer::new(
                    SESSION_LATENCY.nanos().unwrap_or(0),
                    stream.desc.clock_rate,
                );
                stream.ts_unwrap = None;
                stream.anchor = None;
                stream.pending = None;
            }
            stream.ssrc = Some(p.ssrc());
        }
        let arrival_ns = arrival.nanos().unwrap_or(0);
        stream.jitter.insert(p.seq(), p.timestamp(), datagram.to_vec(), arrival_ns);
    }

    /// Emit everything ready across streams; `false` = a pool-dry stall (the
    /// carry holds the packet; resume next crank).
    fn emit_ready(&mut self, ctx: &mut Ctx) -> bool {
        let now_ns = ctx.now().nanos().unwrap_or(0);
        for stream in &mut self.streams {
            loop {
                let held = match stream.pending.take() {
                    Some(h) => h,
                    None => match stream.jitter.pop_ready(now_ns) {
                        Some(h) => h,
                        None => break,
                    },
                };
                let Some(mut buf) = ctx.try_alloc(stream.pad) else {
                    stream.pending = Some(held); // backpressure carry
                    return false;
                };
                // pts: anchor-relative RTP timestamp at the stream's rate.
                let raw_ts = RtpPacket::parse(&held.datagram)
                    .map(|p| p.timestamp())
                    .unwrap_or(0);
                let ext_ts = Self::extend_ts(stream, raw_ts);
                let (anchor_ts, anchor_pts) = *stream.anchor.get_or_insert_with(|| {
                    (ext_ts, Timestamp::from_nanos(held.arrival_ns).saturating_add(SESSION_LATENCY))
                });
                let delta = ext_ts.wrapping_sub(anchor_ts) as i64;
                let delta_ns = delta.saturating_mul(1_000_000_000) / stream.desc.clock_rate.max(1) as i64;
                let base = anchor_pts.nanos().unwrap_or(0) as i64;
                buf.pts = Timestamp::from_nanos(base.saturating_add(delta_ns).max(0) as u64);

                let n = held.datagram.len().min(buf.memory.capacity());
                buf.memory.as_mut_full()[..n].copy_from_slice(&held.datagram[..n]);
                buf.memory.set_len(n);
                ctx.out(stream.pad).push(buf);
            }
        }
        true
    }

    /// Book the next crank on the clock (spec: Scheduling — `Ctx::wake_at`).
    ///
    /// A held packet is released by the *passage of time*: the head of a gap waits
    /// out [`SESSION_LATENCY`] before the missing sequence numbers are declared lost
    /// (RFC 3550 §6.4.1). Nothing else brings the element back — with its input
    /// drained the group parks blocking on the upstream ring, which only a datagram
    /// wakes, and a live RTP source is *routinely* quiet: the tail of a stream, the
    /// end of a talkspurt, a burst loss that takes the rest of the burst with it. The
    /// packets behind the gap then sit here until the next datagram, however many
    /// seconds away that is (or forever, if the stream is over) instead of the 100 ms
    /// the buffer promised. Book the wake-up and the deadline fires on its own.
    fn arm_wakeup(&self, ctx: &mut Ctx) {
        for stream in &self.streams {
            if stream.pending.is_some() {
                // Not waiting on the clock but on an output slot: come back as soon
                // as the scheduler will have us (it paces the retry on its idle tick).
                let now = ctx.now();
                ctx.wake_at(now);
            } else if let Some(at) = stream.jitter.next_deadline_ns() {
                ctx.wake_at(Timestamp::from_nanos(at));
            }
        }
    }
}

impl Element for RtpSession {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Topology settles at preroll (spec: dynamic pads): one src pad per
        // declared stream — no data needed, the SDP already named them.
        for d in &self.descs {
            let pad = ctx.add_pad(Direction::Src, &format!("src_pt{}", d.payload_type), &RTP_OFFERS);
            self.streams.push(Stream {
                desc: *d,
                pad,
                jitter: JitterBuffer::new(SESSION_LATENCY.nanos().unwrap_or(0), d.clock_rate),
                ssrc: None,
                ts_unwrap: None,
                anchor: None,
                pending: None,
                last_sr: None,
            });
        }
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Feed. With one linked sink the scheduler delivers batches through
        // `inputs`; as a fan-in head each pad arrives via `take_input_on`.
        // Handle both, like MkvMuxN — each path is a no-op under the other.
        // (`inputs` can only be the RTP sink: the RTCP pad is secondary and
        // never the single linked pad of a working session.)
        while let Some(buf) = inputs.pop() {
            self.ingest_rtp(buf.memory.data(), buf.pts);
        }
        // RTP datagrams → jitter buffers.
        let mut batch = ctx.take_input_on(RTP_SINK);
        while let Some(buf) = batch.pop_front() {
            self.ingest_rtp(buf.memory.data(), buf.pts);
        }
        ctx.recycle_input(batch);

        // RTCP: stash sender reports (sync follow-up reads them).
        let mut batch = ctx.take_input_on(RTCP_SINK);
        while let Some(buf) = batch.pop_front() {
            if let Ok(items) = rtcp::parse_compound(buf.memory.data()) {
                for item in items {
                    if let rtcp::RtcpItem::SenderReport(sr) = item {
                        if let Some(s) = self.streams.iter_mut().find(|s| s.ssrc == Some(sr.ssrc)) {
                            s.last_sr = Some(sr);
                        }
                    }
                }
            }
        }
        ctx.recycle_input(batch);

        self.emit_ready(ctx);
        self.arm_wakeup(ctx);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}
