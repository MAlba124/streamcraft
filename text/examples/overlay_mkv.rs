//! End-to-end proof of the subtitle path (spec: subtitle support; RFC 9559 §12.7).
//!
//! Builds `bytesrc(mkv) ! mkvdemux`, prerolls, then routes the **video** track through
//! `h264dec` into `suboverlay.video` and the **subtitle** track through `subparse` into
//! `suboverlay.text`, and captures the composited frames at a recording sink:
//!
//! ```text
//!   bytesrc ! mkvdemux ─(video: h264/annexb)→ h264dec ─(video/raw i420)→ suboverlay.video
//!                     └─(subtitle: subtitle/srt)→ subparse ─(subtitle/events)→ suboverlay.text
//!   suboverlay.src ─(video/raw)→ planerecordsink
//! ```
//!
//! The source mkv is generated with ffmpeg (a `testsrc2` video + a hand-written `/tmp/subs.srt`
//! with two timed cues), muxed as an `S_TEXT/UTF8` soft-subtitle track. The proof: the
//! caption region's luma **changes while a cue is active** (frames at ~1.5 s and ~4.5 s) versus
//! a frame with no active cue (~2.5 s / ~5.75 s). Headless — no display.
//!
//! Run: `nix develop --command cargo run --release -p sc-text --example overlay_mkv`

use std::process::Command;
use std::sync::{Arc, Mutex};

use sc_h264::H264Dec;
use sc_mkv::ebml::id;
use sc_mkv::{MatroskaReader, MkvDemux};
use sc_text::{SubParse, SubtitleOverlay};

use streamcraft_elements::flow::Queue;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

const MKV_PATH: &str = "/tmp/subbed.mkv";
const SRT_PATH: &str = "/tmp/subs.srt";

fn main() {
    if !generate_fixture() {
        eprintln!("overlay_mkv: ffmpeg not available or failed — skipping the e2e proof");
        return;
    }
    let stream = read_file(MKV_PATH);
    let recorded = run_overlay(stream);
    let rec = recorded.lock().unwrap();

    println!("overlay_mkv: recorded {} composited frames", rec.frames.len());
    assert!(!rec.frames.is_empty(), "the overlay produced frames end-to-end");
    let (w, h) = rec.dims.expect("video format announced downstream of the overlay");
    println!("overlay_mkv: frame geometry {w}x{h} i420");

    // The bottom third is the caption band. The burned-in white glyphs (luma≈235) stand out
    // against testsrc2's mid-tones, so *count the bright pixels* in the band — a far stronger,
    // less-diluted signal than the band mean (a small caption barely moves the mean of a whole
    // third of the frame, but adds hundreds of near-white pixels that are otherwise absent).
    let bright_pixels = |f: &Frame| -> usize {
        let stride = w as usize;
        let start = (2 * h as usize / 3) * stride;
        let band = &f.planes[start..(h as usize) * stride];
        band.iter().filter(|&&p| p > 220).count()
    };

    // Classify each frame by whether a cue should be on screen (cue windows [1,3) and [4,5.5)),
    // sampling "off" frames well clear of the cue edges to avoid off-by-a-frame ambiguity.
    let mut on_bright: Vec<usize> = Vec::new();
    let mut off_bright: Vec<usize> = Vec::new();
    for f in &rec.frames {
        let t = f.pts.nanos().unwrap_or(0) as f64 / 1e9;
        let active = (1.0..3.0).contains(&t) || (4.0..5.5).contains(&t);
        if active {
            on_bright.push(bright_pixels(f));
        } else if !(0.9..3.1).contains(&t) && !(3.9..5.6).contains(&t) {
            off_bright.push(bright_pixels(f));
        }
    }
    let avg = |v: &[usize]| if v.is_empty() { 0.0 } else { v.iter().sum::<usize>() as f64 / v.len() as f64 };
    let (on, off) = (avg(&on_bright), avg(&off_bright));
    println!(
        "overlay_mkv: caption-band bright pixels (luma>220)  with-cue={on:.0}  without-cue={off:.0}  (n_on={}, n_off={})",
        on_bright.len(),
        off_bright.len()
    );

    assert!(!on_bright.is_empty(), "some frames fall inside a cue window");
    assert!(!off_bright.is_empty(), "some frames fall outside every cue window");
    // White glyphs light up the caption band while a cue is active, and are absent otherwise.
    assert!(
        on > off + 50.0,
        "the caption band has far more bright (white-text) pixels while a cue is active \
         ({on:.0} vs {off:.0}) — the overlay burned in the subtitle"
    );
    println!("overlay_mkv: PASS — the caption region's luma changes with the active cue");
}

// -------------------------------------------------------------------------------------------
// fixture generation (app/test setup — the reactor-IO rule exempts the launcher; ffmpeg is an
// external process, and reading the produced file is app setup, not element code)
// -------------------------------------------------------------------------------------------

/// Write `/tmp/subs.srt` and mux `/tmp/subbed.mkv` with ffmpeg. Returns false if ffmpeg is
/// missing or fails (the proof then skips rather than failing a box without ffmpeg).
fn generate_fixture() -> bool {
    #[allow(clippy::disallowed_methods)] // app setup: writing a throwaway test fixture
    if std::fs::write(
        SRT_PATH,
        "1\n00:00:01,000 --> 00:00:03,000\nHello, subtitles!\n\n\
         2\n00:00:04,000 --> 00:00:05,500\nSecond caption line\n",
    )
    .is_err()
    {
        return false;
    }
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=6",
            "-i", SRT_PATH,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:s", "srt",
            "-shortest", MKV_PATH,
        ])
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Read the whole file (app setup — outside any element).
fn read_file(path: &str) -> Vec<u8> {
    #[allow(clippy::disallowed_methods)] // app setup: loading the fixture before the pipeline
    std::fs::read(path).expect("read fixture mkv")
}

/// The header prefix `MkvDemux::new` needs — everything up to the first Cluster.
fn header_prefix(stream: &[u8]) -> Vec<u8> {
    let cluster = stream
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("stream has a Cluster");
    stream[..cluster].to_vec()
}

// -------------------------------------------------------------------------------------------
// the pipeline
// -------------------------------------------------------------------------------------------

fn run_overlay(stream: Vec<u8>) -> Arc<Mutex<Recorded>> {
    let header = header_prefix(&stream);

    // Probe the header once (a throwaway reader) to learn each track's codec, so we can classify
    // the demuxer's `src_track{N}` pads by family without reaching into the live element.
    let mut probe = MatroskaReader::new();
    probe.push(&header).expect("probe header parse");
    let family_of = |n: u64| -> &'static str {
        probe
            .tracks()
            .iter()
            .find(|t| t.track_number == n)
            .map(|t| sc_mkv::codec::family_for(&t.codec_id))
            .unwrap_or("bytes")
    };

    let mut p = Pipeline::new();
    let src = p.add(ByteSrc { data: stream, chunk: 4096, pos: 0 });
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");

    let added = p.preroll().expect("preroll");
    // Match each discovered pad to its role by its track's family.
    let mut video_pad: Option<streamcraft_core::pipeline::AddedPadInfo> = None;
    let mut subtitle_pad: Option<streamcraft_core::pipeline::AddedPadInfo> = None;
    for ap in &added {
        let n: u64 = ap.name.trim_start_matches("src_track").parse().unwrap_or(0);
        let family = family_of(n);
        println!("overlay_mkv: track {n} family={family}");
        if family == "h264/annexb" {
            video_pad = Some(ap.clone());
        } else if family.starts_with("subtitle/") {
            subtitle_pad = Some(ap.clone());
        }
    }
    let video_pad = video_pad.expect("an h264 video track");
    let subtitle_pad = subtitle_pad.expect("a subtitle track");

    let dec = p.add(H264Dec::new());
    // A `queue` heads the subtitle branch: the demuxer (passive) fans out to two consumers, and
    // the scheduler requires each branch to be its own active group head (a passive fan-out is
    // unsupported). h264dec is already Active; the subtitle branch gets an explicit `queue`
    // (Active passthrough) so both branches are legal group heads. subparse (Passive) then
    // inlines into the queue's group.
    let subq = p.add(Queue::new());
    let subparse = p.add(SubParse::new());
    let overlay = p.add(SubtitleOverlay::new());
    let (sink, recorded) = PlaneRecordSink::new();
    let snk = p.add(sink);

    // video track → h264dec → overlay.video
    p.link((video_pad.element, &video_pad.name), (dec, "sink")).expect("demux video -> h264dec");
    p.link((dec, "src"), (overlay, "video")).expect("h264dec -> overlay.video");
    // subtitle track → queue → subparse → overlay.text
    p.link((subtitle_pad.element, &subtitle_pad.name), (subq, "sink")).expect("demux subs -> queue");
    p.link((subq, "src"), (subparse, "sink")).expect("queue -> subparse");
    p.link((subparse, "src"), (overlay, "text")).expect("subparse -> overlay.text");
    // overlay → sink
    p.link((overlay, "src"), (snk, "sink")).expect("overlay -> sink");

    p.run().expect("run");
    recorded
}

// -------------------------------------------------------------------------------------------
// test elements: a chunked byte source and a plane-recording sink (copied from the mkv tests)
// -------------------------------------------------------------------------------------------

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "bytesrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct ByteSrc {
    data: Vec<u8>,
    chunk: usize,
    pos: usize,
}

impl Element for ByteSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.pos = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.pos >= self.data.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let cap = buf.memory.capacity();
        let n = self.chunk.min(cap).min(self.data.len() - self.pos);
        buf.memory.as_mut_full()[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        self.pos += n;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static PIXFMTS: [ValueDesc; 2] = [ValueDesc::Id("i420"), ValueDesc::Id("nv12")];
static RAW_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "pixfmt", allowed: ConstraintDesc::Set(&PIXFMTS), preferred: None },
];
static RAW_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "video/raw", fields: &RAW_FIELDS }];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &RAW_OFFERS,
    dynamic: false,
    validate: None,
}];
static RECORD_DESC: ElementDesc = ElementDesc {
    name: "planerecordsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct Frame {
    pts: Timestamp,
    planes: Vec<u8>,
}

#[derive(Default)]
struct Recorded {
    frames: Vec<Frame>,
    dims: Option<(u32, u32)>,
}

struct PlaneRecordSink {
    shared: Arc<Mutex<Recorded>>,
}

impl PlaneRecordSink {
    fn new() -> (Self, Arc<Mutex<Recorded>>) {
        let shared = Arc::new(Mutex::new(Recorded::default()));
        (Self { shared: Arc::clone(&shared) }, shared)
    }
}

impl Element for PlaneRecordSink {
    fn desc(&self) -> &'static ElementDesc {
        &RECORD_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.shared.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.frames.push(Frame { pts: buf.pts, planes: buf.memory.data().to_vec() });
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(_) = event {
            let f = ctx.negotiated(PadId(0)).expect("format installed").clone();
            let get = |name: &str| ctx.field_id(name).and_then(|id| f.get(id));
            if let (Some(Value::Int(w)), Some(Value::Int(h))) = (get("width"), get("height")) {
                self.shared.lock().unwrap().dims = Some((w as u32, h as u32));
            }
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}
