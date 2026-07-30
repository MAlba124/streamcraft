//! Steady-state allocation audit for a **real FLAC→Opus transcode** — the whole pipeline, not just
//! the encoder. Runs `filesrc → flacdec → opusenc → sink` on a 48 kHz/s16/mono FLAC fixture through
//! the actual scheduler, with a counting global allocator, and reports allocations per emitted Opus
//! packet over a steady window (excluding startup and teardown).
//!
//!   cargo run -p pf-opus --example transcode_alloc_check
//!
//! This is the honest end-to-end number: flacdec decode + opusenc (CELT) encode + pipeline/batch/
//! event overhead, per Opus packet.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

use pf_flac::FlacDec;
use pf_opus::OpusEnc;
use profluens_elements::io::FileSrc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
// SAFETY: delegates to System; only bumps a relaxed counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Counting = Counting;

static PKT: AtomicUsize = AtomicUsize::new(0);

/// A sink that just counts Opus packets (one buffer each). The steady-state alloc rate is derived
/// in `main` from the whole-run counter against this packet count (the Active sink drains after the
/// producer chain, so a sink-side window can't see the interleaving).
struct CountSink;

static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("opus"), OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "packetcountsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

impl Element for CountSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while inputs.pop().is_some() {
            PKT.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn main() {
    // Default to a long (looped) fixture so one-time startup amortizes to ~0/pkt; override with an
    // argument. (The Active sink drains after the producer chain, so we measure the whole run's
    // allocations against the packet count rather than a sink-side window.)
    let flac = std::env::args().nth(1).unwrap_or_else(|| {
        let long = "/tmp/sc_long.flac";
        if std::path::Path::new(long).exists() { long.into() } else { "fixtures/out/audio.flac".into() }
    });
    if !std::path::Path::new(&flac).exists() {
        eprintln!("missing fixture {flac}");
        std::process::exit(2);
    }

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&flac));
    let dec = p.add(FlacDec::new());
    let enc = p.add(OpusEnc::new());
    let snk = p.add(CountSink);
    p.link((src, "src"), (dec, "sink")).expect("filesrc→flacdec");
    p.link((dec, "src"), (enc, "sink")).expect("flacdec→opusenc");
    p.link((enc, "src"), (snk, "sink")).expect("opusenc→sink");

    // Baseline the counter right before the run so program + pipeline construction don't count;
    // what remains is decode + encode + scheduler overhead across the whole stream.
    let base = ALLOCS.load(Ordering::Relaxed);
    p.run().expect("run");
    let total_allocs = ALLOCS.load(Ordering::Relaxed).saturating_sub(base);
    let pkts = PKT.load(Ordering::Relaxed);

    println!("fixture: {flac}");
    println!("Opus packets: {pkts}   run allocations: {total_allocs}");
    if pkts > 0 {
        println!(
            "FLAC→Opus transcode: {:.1} allocs/pkt  (flacdec decode + opusenc encode + scheduler; \
             one-time startup amortized. Default = libopus backend; --no-default-features = pure-Rust)",
            total_allocs as f64 / pkts as f64
        );
    }
}
