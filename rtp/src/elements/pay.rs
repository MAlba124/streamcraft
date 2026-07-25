//! Payload elements: codec access units in, RTP packets out. Thin wrappers —
//! the packetization logic is [`crate::pay`]; these own the RTP header state
//! (RFC 3550 §5.1: sequence numbers, the timestamp mapping, the SSRC).
//!
//! Header state per §5.1: the initial sequence number and the timestamp
//! offset are random ("SHOULD be random ... to make known-plaintext attacks
//! more difficult"), the SSRC is chosen randomly per stream (§8). Output
//! buffers keep the access unit's `pts` — the paced network sink downstream
//! waits on it, which is what turns a fast file read into a realtime stream.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::packet::RtpPacketBuilder;

/// RTP payload budget per packet: a common 1500-byte Ethernet MTU minus
/// IP/UDP/RTP headers with margin (RFC 6184 §5.8's reason for FU-A).
const MTU: usize = 1400;

/// H.264's fixed RTP clock (RFC 6184 §8.1: "clock rate MUST be 90000").
const H264_CLOCK: u64 = 90_000;

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("rtp")];

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

static PADS: [PadDesc; 2] = [
    PadDesc { name: "sink", direction: Direction::Sink, offers: &SINK_OFFERS, dynamic: false, validate: None },
    PadDesc { name: "src", direction: Direction::Src, offers: &SRC_OFFERS, dynamic: false, validate: None },
];

static DESC: ElementDesc = ElementDesc {
    name: "rtph264pay",
    pads: &PADS,
    props: &[],
    // Active: a demuxer branches only from its group's tail — an inlined
    // passive payloader behind mkvdemux breaks the branch ("inter-group
    // branch from a non-tail element"); as an Active head it forms its own
    // group, exactly like the decoders in the playback pipelines.
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// A weak-but-sufficient seed for the §5.1 random header fields — this is
/// collision avoidance and plaintext offsetting, not cryptography.
fn seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // SplitMix64 finalizer over time ⊕ pid.
    let mut z = t ^ ((std::process::id() as u64) << 32);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// RFC 6184 payload element: Annex-B access units → RTP packets.
pub struct RtpH264Pay {
    payload_type: u8,
    ssrc: u32,
    seq: u16,
    ts_offset: u32,
    /// Packets awaiting output slots (multiple packets per AU).
    pending: std::collections::VecDeque<(Vec<u8>, Timestamp)>,
}

impl RtpH264Pay {
    /// A payloader announcing `payload_type` (the dynamic PT the SDP maps,
    /// conventionally 96).
    pub fn new(payload_type: u8) -> RtpH264Pay {
        let s = seed();
        RtpH264Pay {
            payload_type,
            ssrc: (s >> 32) as u32,
            seq: s as u16,
            ts_offset: (s >> 16) as u32,
            pending: std::collections::VecDeque::new(),
        }
    }

    /// The stream's SSRC (the SDP/RTSP layer may advertise it).
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    fn flush_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some((pkt, pts)) = self.pending.pop_front() {
            let Some(mut buf) = ctx.try_alloc(SRC) else {
                self.pending.push_front((pkt, pts));
                return false;
            };
            let n = pkt.len().min(buf.memory.capacity());
            buf.memory.as_mut_full()[..n].copy_from_slice(&pkt[..n]);
            buf.memory.set_len(n);
            buf.pts = pts;
            ctx.out(SRC).push(buf);
        }
        true
    }
}

impl Element for RtpH264Pay {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.flush_pending(ctx) {
            return Ok(Flow::Ok);
        }
        while let Some(buf) = inputs.pop() {
            // The RTP timestamp: the AU's pts at the 90 kHz media clock (§5.1),
            // plus the random session offset; 32-bit wrap is the format.
            let ts_media = buf
                .pts
                .nanos()
                .map(|ns| (ns as u128 * H264_CLOCK as u128 / 1_000_000_000) as u64)
                .unwrap_or(0);
            let timestamp = (ts_media as u32).wrapping_add(self.ts_offset);
            for (payload, marker) in crate::pay::h264::pay(buf.memory.data(), MTU) {
                let b = RtpPacketBuilder {
                    marker,
                    payload_type: self.payload_type,
                    seq: self.seq,
                    timestamp,
                    ssrc: self.ssrc,
                };
                self.seq = self.seq.wrapping_add(1);
                self.pending.push_back((b.build(&payload), buf.pts));
            }
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

// The sink-pad id is unused by name but documents the pad table.
#[allow(dead_code)]
const _: PadId = SINK;
