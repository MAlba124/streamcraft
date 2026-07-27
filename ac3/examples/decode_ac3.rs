//! Decode a raw AC-3 or E-AC-3 elementary stream through the real element to
//! interleaved s16le — the direct-library probe for the two target files (the
//! `sc-aac` `decode_adts` / `sc-h265` `decode_annexb` precedent: adoption gates
//! must include a real file, not just fixtures).
//!
//! Feeds the raw byte stream in chunks so the self-syncing framer
//! ([`sc_ac3::parse`]) is exercised exactly as an AVI chunk / `filesrc` split
//! would exercise it, then dumps the decoded PCM (channel order L,R,C,LFE,Ls,Rs).
//!
//! ```text
//! ffmpeg -i movie.mkv -map 0:a:0 -c copy audio.eac3
//! cargo run --release -p sc-ac3 --example decode_ac3 -- eac3 audio.eac3 out.s16le [max_frames]
//! ffmpeg -i audio.eac3 -f s16le -ac 6 ref.s16le   # then compare (see snr.rs test)
//! ```

use std::io::Write;
use std::sync::{Arc, Mutex};

use sc_ac3::{Ac3Dec, Eac3Dec};
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

static BYTES_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &BYTES_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "chunksrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

/// Emits the file as a series of fixed-size byte chunks, so the decoder's framer
/// resyncs across arbitrary boundaries (the AVI-chunk case).
struct ChunkSrc {
    data: Vec<u8>,
    pos: usize,
    chunk: usize,
}

impl Element for ChunkSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.pos >= self.data.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let cap = buf.memory.capacity().min(self.chunk);
        let end = (self.pos + cap).min(self.data.len());
        let n = end - self.pos;
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.data[self.pos..end]);
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        self.pos = end;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &BYTES_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "pcmrecordsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PcmRecordSink {
    got: Arc<Mutex<Vec<u8>>>,
    max_bytes: usize,
}

impl Element for PcmRecordSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            if got.len() < self.max_bytes {
                got.extend_from_slice(buf.memory.data());
            }
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(kind), Some(inp), Some(outp)) = (args.first(), args.get(1), args.get(2)) else {
        eprintln!("usage: decode_ac3 <ac3|eac3> IN OUT.s16le [max_frames]");
        std::process::exit(2);
    };
    let max_frames: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    // Cap decoded bytes to keep the run bounded when max_frames is large.
    // 6 ch * 2 B * 1536 samples/frame.
    let max_bytes = max_frames.saturating_mul(6 * 2 * 1536).min(1 << 30);

    #[allow(clippy::disallowed_methods)] // example harness, not element code
    let data = std::fs::read(inp).expect("read input");
    // Only feed enough bytes to produce ~max_frames; a DD+ frame is ≤ ~2 KB.
    let feed_len = max_frames.saturating_mul(2560).min(data.len());
    let data: Vec<u8> = data[..feed_len].to_vec();
    eprintln!("feeding {} bytes ({kind})", data.len());

    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(ChunkSrc { data, pos: 0, chunk: 4096 });
    let dec = if kind == "eac3" {
        p.add(Eac3Dec::new())
    } else {
        p.add(Ac3Dec::new())
    };
    let sink = p.add(PcmRecordSink { got: Arc::clone(&got), max_bytes });
    p.link((src, "src"), (dec, "sink")).expect("src ! dec");
    p.link((dec, "src"), (sink, "sink")).expect("dec ! sink");
    p.run().expect("run");

    let mut warns = 0;
    while let Some(msg) = p.bus().try_recv() {
        if let streamcraft_core::bus::BusMessage::Warning { error, .. } = msg {
            if warns < 8 {
                eprintln!("warning: {error:?}");
            }
            warns += 1;
        }
    }
    if warns > 8 {
        eprintln!("... and {} more warnings", warns - 8);
    }

    let bytes = got.lock().unwrap();
    #[allow(clippy::disallowed_methods)] // example harness
    std::fs::File::create(outp)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("write output");
    eprintln!("wrote {} bytes of s16le ({} warnings)", bytes.len(), warns);
}
