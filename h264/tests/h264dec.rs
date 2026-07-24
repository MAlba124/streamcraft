//! `H264Dec` element integration (spec: Milestone applications §5 — the decoder
//! half). H.264 Annex B access units are encoded in-test with the same crate's
//! encoder (an IDR + a P frame, so both the intra and inter decode paths run),
//! decoded once directly through `H264CodecDecoder` as the reference, then pushed
//! through a pipeline `AuSrc ! H264Dec ! PlaneRecordSink` — output planes must be
//! byte-identical to the reference decode, pts must pass through, and the
//! announced `video/raw` format must carry the real (MB-aligned) dimensions.
//! A corrupt / garbage access unit must degrade per-buffer (bus warning, drop),
//! never panic or kill the pipeline.
//!
//! An ffmpeg-oracle test (`ffmpeg_oracle_decode_matches`) additionally cross-checks
//! our decode against the system ffmpeg when it is present, and cleanly skips
//! otherwise (mirrors oxideav-vp8's blackbox_oracle pattern) — CI needs no ffmpeg.

use std::sync::{Arc, Mutex};

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
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

use sc_h264::H264Dec;

/// 128x96 = 8x6 MBs exactly, so the decoder's MB-aligned output equals the coded
/// dimensions (no crop padding) — stride == width, the packed-I420 case the
/// element wires.
const W: u32 = 128;
const H: u32 = 96;
/// One frame per 33 ms — a stand-in container timestamp grid.
const FRAME_NS: u64 = 33_000_000;

// ---- test elements -----------------------------------------------------------

static AU_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &AU_OFFERS,
    dynamic: false,
    validate: None,
}];

static AUSRC_DESC: ElementDesc = ElementDesc {
    name: "ausrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Emits pre-encoded H.264 Annex B access units, one per buffer (the demuxer
/// contract), pts on the 33 ms grid, then EOS.
struct AuSrc {
    aus: Vec<Vec<u8>>,
    next: usize,
}

impl AuSrc {
    fn new(aus: Vec<Vec<u8>>) -> Self {
        Self { aus, next: 0 }
    }
}

impl Element for AuSrc {
    fn desc(&self) -> &'static ElementDesc {
        &AUSRC_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.aus.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let bytes = &self.aus[self.next];
        assert!(buf.memory.capacity() >= bytes.len(), "access unit fits a pool slot");
        buf.memory.as_mut_full()[..bytes.len()].copy_from_slice(bytes);
        buf.memory.set_len(bytes.len());
        buf.pts = Timestamp::from_nanos(self.next as u64 * FRAME_NS);
        ctx.out(PadId(0)).push(buf);
        self.next += 1;
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static PIXFMTS: [ValueDesc; 1] = [ValueDesc::Id("i420")];
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
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

#[derive(Default)]
struct Recorded {
    frames: Vec<(Timestamp, Vec<u8>)>,
    /// (width, height, pixfmt name) read from the announced format at FormatChange.
    format: Option<(i64, i64, String)>,
}

/// Records every received frame's pts + packed planes, and the announced format.
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
            rec.frames.push((buf.pts, buf.memory.data().to_vec()));
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if let Event::FormatChange(_) = event {
            // Read the installed format by name — the dynamic-caps consumer path.
            let f = ctx.negotiated(PadId(0)).expect("format installed").clone();
            let get = |name: &str| ctx.field_id(name).and_then(|id| f.get(id));
            let (Some(Value::Int(w)), Some(Value::Int(h)), Some(Value::Id(pf))) =
                (get("width"), get("height"), get("pixfmt"))
            else {
                panic!("announced format missing width/height/pixfmt");
            };
            let name = ctx.value_name(pf).expect("pixfmt name interned").to_string();
            self.shared.lock().unwrap().format = Some((w, h, name));
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---- fixtures ----------------------------------------------------------------

/// A deterministic I420 source picture; each frame index gets distinct content
/// (a smooth diagonal luma gradient plus a per-frame shift, chroma gradients) so
/// the intra + inter paths reconstruct something non-trivial.
fn make_planes(idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let s = idx as usize;
    let y = (0..w * h)
        .map(|i| {
            let (x, yy) = (i % w, i / w);
            (16 + ((x + yy + s) * (240 - 16)) / (w + h)) as u8
        })
        .collect();
    let u = (0..cw * ch)
        .map(|i| (64 + ((i % cw) * 128) / cw + s) as u8)
        .collect();
    let v = (0..cw * ch)
        .map(|i| (64 + ((i / cw) * 128) / ch + s * 2) as u8)
        .collect();
    (y, u, v)
}

/// Encode an IDR followed by `p_count` P frames — the intra path plus the inter
/// path. Returns one Annex B access unit per element (the demuxer contract).
fn encode_i_then_p(p_count: u32) -> Vec<Vec<u8>> {
    let cfg = EncoderConfig::new(W, H);
    let enc = Encoder::new(cfg);
    let (y0, u0, v0) = make_planes(0);
    let f0 = YuvFrame { width: W, height: H, y: &y0, u: &u0, v: &v0 };
    let idr = enc.encode_idr(&f0);
    let mut aus = vec![idr.annex_b.clone()];
    let mut prev_ref = EncodedFrameRef::from(&idr);
    let mut owned: Vec<_> = Vec::new();
    for i in 1..=p_count {
        let (y, u, v) = make_planes(i);
        let f = YuvFrame { width: W, height: H, y: &y, u: &u, v: &v };
        let p = enc.encode_p(&f, &prev_ref, i, i * 2);
        aus.push(p.annex_b.clone());
        // Keep the reconstructed P as the next frame's reference.
        owned.push(p);
        prev_ref = EncodedFrameRef::from(owned.last().unwrap());
    }
    aus
}

/// Reference decode: the same access units through the library directly, packed
/// Y|U|V, in the decoder's own emission (display) order.
fn reference_decode(aus: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    for au in aus {
        let pkt = Packet::new(0, TimeBase::new(1, 1), au.clone());
        dec.send_packet(&pkt).expect("reference send_packet");
    }
    dec.flush().expect("reference flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let p = vf.image_planes();
                let mut packed =
                    Vec::with_capacity(p[0].data.len() + p[1].data.len() + p[2].data.len());
                packed.extend_from_slice(&p[0].data);
                packed.extend_from_slice(&p[1].data);
                packed.extend_from_slice(&p[2].data);
                out.push(packed);
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

fn run_pipeline(aus: Vec<Vec<u8>>) -> (Pipeline, Arc<Mutex<Recorded>>) {
    let mut p = Pipeline::new();
    let (sink, recorded) = PlaneRecordSink::new();
    let src = p.add(AuSrc::new(aus));
    let dec = p.add(H264Dec::new());
    let snk = p.add(sink);
    p.link((src, "src"), (dec, "sink")).expect("link src!dec");
    p.link((dec, "src"), (snk, "sink")).expect("link dec!sink");
    p.run().expect("run");
    (p, recorded)
}

// ---- tests -------------------------------------------------------------------

#[test]
fn decodes_bit_exactly_with_pts_passthrough_and_announced_format() {
    // IDR + 2 P frames: exercises both the intra and inter decode paths.
    let aus = encode_i_then_p(2);
    let reference = reference_decode(&aus);
    assert_eq!(reference.len(), 3, "encoder+decoder produce all three pictures");
    let (_p, recorded) = run_pipeline(aus);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 3, "every picture emitted through the element");
    for (i, ((_pts, planes), want)) in rec.frames.iter().zip(&reference).enumerate() {
        assert_eq!(planes, want, "frame {i} planes byte-identical to reference decode");
    }
    // This IDR+P sequence has no reordering (POC monotonically increases), so the
    // element's decode-order pts equals the display pts on the 33 ms grid.
    for (i, (pts, _planes)) in rec.frames.iter().enumerate() {
        assert_eq!(*pts, Timestamp::from_nanos(i as u64 * FRAME_NS), "frame {i} pts");
    }
    let (w, h, pf) = rec.format.clone().expect("format announced");
    assert_eq!((w, h), (W as i64, H as i64));
    assert_eq!(pf, "i420");
}

#[test]
fn corrupt_access_unit_warns_and_recovers_at_next_idr() {
    // A coherent IDR + P stream with a garbage access unit spliced between them
    // (a start-code-delimited NAL of nonsense). The garbage unit must fail
    // per-buffer (bus warning, drop) without corrupting decoder state, and the
    // IDR + P must both still decode — the P reconstructs against the intact IDR
    // reference. Mirrors sc-vp8's corrupt-frame recovery test.
    let good = encode_i_then_p(1); // [IDR, P]
    let reference = reference_decode(&good);
    assert_eq!(reference.len(), 2, "clean stream decodes to two pictures");

    let mut aus = good;
    aus.insert(1, vec![0x00, 0x00, 0x00, 0x01, 0x67, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE]);
    let (p, recorded) = run_pipeline(aus);

    let rec = recorded.lock().unwrap();
    assert_eq!(rec.frames.len(), 2, "good IDR + P decoded, corrupt unit dropped");
    assert_eq!(rec.frames[0].1, reference[0], "IDR bit-exact after recovery");
    assert_eq!(rec.frames[1].1, reference[1], "P bit-exact after recovery");

    let mut warnings = 0;
    while let Some(msg) = p.bus().try_recv() {
        if matches!(msg, BusMessage::Warning { .. }) {
            warnings += 1;
        }
    }
    assert!(warnings >= 1, "the corrupt access unit surfaced as a bus warning");
}

#[test]
fn garbage_only_stream_emits_nothing_and_never_panics() {
    let garbage: Vec<Vec<u8>> = (0..4)
        .map(|i| {
            let mut v = vec![0x00, 0x00, 0x00, 0x01];
            v.extend((0..(48 + i * 17)).map(|k| (k as u8).wrapping_mul(37).wrapping_add(i as u8)));
            v
        })
        .collect();
    let (_p, recorded) = run_pipeline(garbage);
    assert!(recorded.lock().unwrap().frames.is_empty());
}

// ---- ffmpeg oracle (skips cleanly when ffmpeg is absent) ---------------------

/// Cross-check our decode of a self-encoded IDR against the system ffmpeg's
/// decode of the *same* Annex B bytes. Skips (returns Ok, prints a note) when
/// ffmpeg is not installed — CI needs neither ffmpeg nor network.
///
/// The comparison is intentionally a coarse PSNR floor, not bit-exactness: two
/// independent conformant H.264 decoders may differ by a rounding ULP here and
/// there, but a correctly-decoded picture is within a fraction of a dB of the
/// oracle. A broken decode (wrong prediction, mis-parsed residual) diverges by
/// tens of dB.
#[test]
fn ffmpeg_oracle_decode_matches() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    if !ffmpeg_available() {
        eprintln!("skip ffmpeg_oracle_decode_matches: ffmpeg not on PATH");
        return;
    }

    // One IDR — the most portable thing to hand a stock ffmpeg build.
    let au = encode_i_then_p(0).remove(0);

    // Our decode → packed I420.
    let ours = reference_decode(std::slice::from_ref(&au));
    assert_eq!(ours.len(), 1, "our decode produced the IDR");
    let ours = &ours[0];

    // ffmpeg: Annex B on stdin → rawvideo yuv420p on stdout.
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error",
            "-f", "h264", "-i", "pipe:0",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn ffmpeg");
    child.stdin.take().unwrap().write_all(&au).expect("write annexb to ffmpeg");
    let out = child.wait_with_output().expect("ffmpeg output");
    assert!(out.status.success(), "ffmpeg decode failed");
    let theirs = out.stdout;

    assert_eq!(theirs.len(), ours.len(), "same I420 byte count (both {W}x{H})");

    // Mean-squared error over all planes → PSNR.
    let n = ours.len() as f64;
    let mse: f64 = ours
        .iter()
        .zip(&theirs)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / n;
    let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0 * 255.0 / mse).log10() };
    assert!(
        psnr >= 40.0,
        "our decode diverges from the ffmpeg oracle: PSNR {psnr:.1} dB (mse {mse:.3})"
    );
}
