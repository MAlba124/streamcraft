//! The jitter buffer's playout deadline must fire on its own.
//!
//! `rtpsession` holds packets behind a gap for [`SESSION_LATENCY`] before
//! declaring the missing sequence numbers lost (RFC 3550 §6.4.1). That release
//! is driven by *time*, not by arriving data — so the element has to be cranked
//! when the deadline expires even if the source has gone quiet. A live RTP
//! source going quiet is normal (end of a talk spurt, a camera between
//! GOPs, the tail of a stream), and the scheduler's head parks *blocking* on
//! the upstream ring when the element's inputs are drained: nothing but a new
//! datagram wakes it.
//!
//! The source here sends a burst with one sequence number missing and then
//! stops for good, which is exactly that shape.

// Test-side scripting and collection, sanctioned: these allocate on the test
// thread and in test elements' setup, not on an element's hot path
// (clippy.toml: disallowed-methods).
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pf_rtp::elements::{RtpSession, RtpStreamDesc};
use pf_rtp::packet::RtpPacket;
use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

const PT: u8 = 96;
const CLOCK_RATE: u32 = 90_000;

/// One 12-byte-header RTP packet with a 4-byte payload (RFC 3550 §5.1).
fn packet(seq: u16, ts: u32) -> Vec<u8> {
    let mut d = vec![0x80, PT, 0, 0, 0, 0, 0, 0, 0xDE, 0xAD, 0xBE, 0xEF];
    d[2..4].copy_from_slice(&seq.to_be_bytes());
    d[4..8].copy_from_slice(&ts.to_be_bytes());
    d.extend_from_slice(&seq.to_be_bytes());
    d.extend_from_slice(&seq.to_be_bytes());
    d
}

static DGRAM_OFFERS: [OfferDesc; 1] = [OfferDesc::any("datagram")];
static RTP_OFFERS: [OfferDesc; 1] = [OfferDesc::any("rtp")];

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &DGRAM_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "pktsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    // A live source, like `udpsrc`: no EOS, it simply stops having data.
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Emits a scripted burst of datagrams (one per pass, each held back until its
/// scripted running time), then goes quiet forever without ever signalling EOS
/// — a live source between bursts.
struct PktSrc {
    /// `(datagram, not before this running time)`.
    datagrams: Vec<(Vec<u8>, Timestamp)>,
    at: usize,
    /// Set once the whole burst is on the wire, so the test can time the gap
    /// from "the last packet was sent", not from "the pipeline started".
    sent_all: Arc<AtomicBool>,
}

impl Element for PktSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.at >= self.datagrams.len() {
            self.sent_all.store(true, Ordering::Release);
            return Ok(Flow::Ok); // quiet, but still live: no EOS
        }
        let (d, due) = &self.datagrams[self.at];
        if ctx.now() < *due {
            return Ok(Flow::Ok); // not yet: the group's idle tick retries
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let n = d.len().min(buf.memory.capacity());
        buf.memory.as_mut_full()[..n].copy_from_slice(&d[..n]);
        buf.memory.set_len(n);
        buf.pts = ctx.now();
        ctx.out(PadId(0)).push(buf);
        self.at += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static COLLECT_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &RTP_OFFERS,
    dynamic: false,
    validate: None,
}];
static COLLECT_DESC: ElementDesc = ElementDesc {
    name: "pktcollect",
    pads: &COLLECT_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Records `(sequence number, arrival wall time)` per forwarded packet.
struct PktCollect {
    got: Arc<Mutex<Vec<(u16, Instant)>>>,
}

impl Element for PktCollect {
    fn desc(&self) -> &'static ElementDesc {
        &COLLECT_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            let seq = RtpPacket::parse(buf.memory.data()).expect("forwarded packet is RTP").seq();
            got.push((seq, Instant::now()));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A scripted burst of `(sequence number, send at this running time)` through
/// `pktsrc ! rtpsession ! pktcollect`, waiting up to `budget` for `want`
/// packets to come out. Returns what arrived plus the moment the source ran
/// out of script.
fn run_burst(
    script: &[(u16, Timestamp)],
    want: usize,
    budget: Duration,
) -> (Vec<(u16, Instant)>, Instant) {
    let datagrams: Vec<(Vec<u8>, Timestamp)> =
        script.iter().map(|&(s, due)| (packet(s, s as u32 * 3000), due)).collect();
    let got = Arc::new(Mutex::new(Vec::new()));
    let sent_all = Arc::new(AtomicBool::new(false));

    let mut p = Pipeline::new();
    let src = p.add(PktSrc {
        datagrams,
        at: 0,
        sent_all: Arc::clone(&sent_all),
    });
    let session =
        p.add(RtpSession::new(vec![RtpStreamDesc { payload_type: PT, clock_rate: CLOCK_RATE }]));
    p.link((src, "src"), (session, "sink")).expect("src -> session");
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one stream pad from the session");
    let collect = p.add(PktCollect { got: Arc::clone(&got) });
    p.link((added[0].element, &added[0].name), (collect, "sink")).expect("session -> collect");

    let stop = p.stop_handle();
    let runner = std::thread::spawn(move || p.run());

    // Wait for the burst to leave the source, then time the release from there.
    let armed = Instant::now();
    while !sent_all.load(Ordering::Acquire) && armed.elapsed() < budget {
        std::thread::sleep(Duration::from_millis(1));
    }
    let quiet_at = Instant::now();
    while got.lock().unwrap().len() < want && quiet_at.elapsed() < budget {
        std::thread::sleep(Duration::from_millis(1));
    }
    stop.stop();
    runner.join().unwrap().expect("run");
    let out = got.lock().unwrap().clone();
    (out, quiet_at)
}

/// The regression: packets stranded behind a lost sequence number must be
/// released on the jitter buffer's own playout deadline, not on the arrival of
/// some later datagram that a quiet source will never send.
#[test]
fn a_gap_at_the_head_releases_on_its_deadline_after_the_source_goes_quiet() {
    // Seq 3 never arrives: 4 and 5 sit behind the gap. Everything goes out at
    // once, so the source is quiet from the first pass onwards.
    let now = Timestamp::ZERO;
    let (got, quiet_at) =
        run_burst(&[(1, now), (2, now), (4, now), (5, now)], 4, Duration::from_secs(5));

    let seqs: Vec<u16> = got.iter().map(|(s, _)| *s).collect();
    let held: Vec<String> = got
        .iter()
        .map(|(s, t)| format!("seq {s} @ {:?}", t.saturating_duration_since(quiet_at)))
        .collect();
    assert_eq!(
        seqs,
        vec![1, 2, 4, 5],
        "every arrived packet is forwarded in sequence order; got {held:?}"
    );

    // The two stranded packets must appear within the latency plus scheduler
    // slack — not "whenever the next datagram happens to arrive".
    let release = got[2].1.saturating_duration_since(quiet_at);
    assert!(
        release < Duration::from_millis(600),
        "seq 4 released {release:?} after the source went quiet \
         (SESSION_LATENCY is 100 ms); timeline: {held:?}"
    );
    // ...and not *before* it either: the wait is the reorder window, and
    // collapsing it would trade this stall for dropped late packets.
    assert!(
        release > Duration::from_millis(50),
        "seq 4 released after only {release:?} — the reorder window was skipped, \
         not waited out; timeline: {held:?}"
    );
}

/// The same deadline must not delay an in-order stream: a packet whose
/// predecessor is present is ready on arrival (RFC 3550 §6.4.1 — the wait
/// exists only to cover a gap).
#[test]
fn an_in_order_burst_is_forwarded_without_waiting_out_the_latency() {
    let now = Timestamp::ZERO;
    let (got, quiet_at) =
        run_burst(&[(10, now), (11, now), (12, now), (13, now)], 4, Duration::from_secs(5));
    let seqs: Vec<u16> = got.iter().map(|(s, _)| *s).collect();
    assert_eq!(seqs, vec![10, 11, 12, 13], "in-order burst passes straight through");
    let last = got[3].1.saturating_duration_since(quiet_at);
    assert!(
        last < Duration::from_millis(100),
        "the last in-order packet waited {last:?} — the gap timer must not pace a gapless stream"
    );
}

/// Waking on the deadline must not be bought by *blocking* on it: the element
/// has to keep ingesting throughout the wait, or the reorder window it exists
/// for is dead. A packet that fills the gap 40 ms into a 100 ms window still
/// releases the whole run in sequence order (RFC 3550 §6.4.1).
#[test]
fn a_packet_arriving_inside_the_window_still_fills_the_gap_in_order() {
    let now = Timestamp::ZERO;
    let late = Timestamp::from_millis(40);
    // 1, 2, 4, 5 arrive at once; 3 is reordered behind them but well inside the
    // 100 ms hold, so nothing is ever declared lost.
    let (got, quiet_at) =
        run_burst(&[(1, now), (2, now), (4, now), (5, now), (3, late)], 5, Duration::from_secs(5));

    let held: Vec<String> = got
        .iter()
        .map(|(s, t)| format!("seq {s} @ {:?}", t.saturating_duration_since(quiet_at)))
        .collect();
    let seqs: Vec<u16> = got.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        seqs,
        vec![1, 2, 3, 4, 5],
        "the reordered packet is placed, not delivered late and out of band; got {held:?}"
    );
    // And it releases on *arrival*, not at the end of the window.
    let release = got[2].1.saturating_duration_since(quiet_at);
    assert!(
        release < Duration::from_millis(60),
        "the gap filler took {release:?} to release the run — the element sat out \
         the deadline instead of consuming input; timeline: {held:?}"
    );
}
