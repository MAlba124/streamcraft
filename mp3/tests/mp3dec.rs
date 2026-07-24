//! `PacketSrc(byte chunks) ! Mp3Dec ! PcmRecordSink` — the incremental MP3 decode path
//! (spec: Milestone applications §3; Formats — dynamic caps). Drives tiny committed
//! MPEG-1 Layer III fixtures through the element at *arbitrary* byte-chunk boundaries
//! (the `filesrc` shape), records the interleaved PCM, and scores it against a committed
//! ffmpeg reference decode by SNR — the same conformance bar as the adoption gate.
//!
//! MP3 decode is spec-exact enough that two conforming decoders agree in the
//! float-rounding regime, so a healthy SNR against ffmpeg's own decode is the
//! compliance criterion (ISO 11172-4 full-scale accuracy). The reference is
//! gapless-trimmed and `mp3dec`'s output is not (gapless is out of v1), so the scorer
//! aligns the two by a best-shift search and skips the first frame's transient before
//! measuring — it grades decode fidelity, not the trim.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use sc_mp3::Mp3Dec;

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

// ---------- fixtures ----------

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixtures_dir().join(name))
        .unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

fn read_ref_s16le(name: &str) -> Vec<i16> {
    read_fixture(name)
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------- test elements (PacketSrc / PcmRecordSink shapes) ----------

static BYTES_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &BYTES_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "packetsrc",
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

/// Emits pre-chopped byte chunks in order — one per `process` pass — then EOS. The
/// chunk boundaries are deliberately not frame-aligned (that is the point of the test).
struct PacketSrc {
    packets: Vec<Vec<u8>>,
    next: usize,
}

impl Element for PacketSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.next = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.packets.len() {
            return Ok(Flow::Eos);
        }
        let pkt = &self.packets[self.next];
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        buf.memory.as_mut_full()[..pkt.len()].copy_from_slice(pkt);
        buf.memory.set_len(pkt.len());
        ctx.out(PadId(0)).push(buf);
        self.next += 1;
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
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Concatenates every received PCM byte, in order (the decoded interleaved i16 stream).
struct PcmRecordSink {
    got: Arc<Mutex<Vec<u8>>>,
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
            got.extend_from_slice(buf.memory.data());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// ---------- pipeline driver ----------

/// Decode `mp3` bytes chopped into `chunk`-sized packets through
/// `PacketSrc ! Mp3Dec ! PcmRecordSink`, returning the recorded interleaved i16 PCM.
fn decode_via_pipeline(mp3: &[u8], chunk: usize) -> Vec<i16> {
    let packets: Vec<Vec<u8>> = mp3.chunks(chunk).map(<[u8]>::to_vec).collect();
    let got = Arc::new(Mutex::new(Vec::new()));

    let mut p = Pipeline::new();
    let src = p.add(PacketSrc { packets, next: 0 });
    let dec = p.add(Mp3Dec::new());
    let sink = p.add(PcmRecordSink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).expect("src ! mp3dec");
    p.link((dec, "src"), (sink, "sink")).expect("mp3dec ! sink");
    p.run().expect("pipeline run");

    let bytes = got.lock().unwrap().clone();
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

// ---------- SNR scorer (best-shift aligned; first frame skipped) ----------

/// SNR in dB of `got` against `reference`, both interleaved i16 with `channels`
/// channels. MP3 decoders differ by a whole-frame encoder/decoder-delay offset and the
/// reference is gapless-trimmed, so we search a small delay window for the shift that
/// maximizes agreement, skip the first frame's transient, and score the aligned
/// overlap. Returns `(best_snr_db, overlap_samples)`.
fn best_snr_db(got: &[i16], reference: &[i16], channels: usize) -> (f64, usize) {
    let frame = 1152 * channels; // MPEG-1: two granules × 576 PCM samples/ch
    let max_shift = (6 * frame) as i64;
    let step = channels as i64; // keep channel alignment
    let skip = frame; // drop the first-frame transient

    let mut best = (f64::NEG_INFINITY, 0usize);
    let mut s = -max_shift;
    while s <= max_shift {
        let (start_g, start_r) = if s < 0 { ((-s) as usize, 0) } else { (0, s as usize) };
        let overlap = got
            .len()
            .saturating_sub(start_g)
            .min(reference.len().saturating_sub(start_r));
        let overlap = overlap.saturating_sub(skip);
        if overlap > 2000 {
            let (mut sig, mut err) = (0f64, 0f64);
            for i in skip..skip + overlap {
                let g = f64::from(got[start_g + i]);
                let r = f64::from(reference[start_r + i]);
                sig += r * r;
                err += (g - r) * (g - r);
            }
            let snr = if err <= 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
            if snr > best.0 {
                best = (snr, overlap);
            }
        }
        s += step;
    }
    best
}

/// The compliance threshold: two conforming MP3 decoders should agree to well within
/// 16-bit PCM's ~96 dB dynamic-range floor. 60 dB (≈ 10 LSB RMS) is a conservative
/// pass that the observed 84–102 dB clears by a wide margin, while still failing loudly
/// on a genuinely broken decode.
const SNR_THRESHOLD_DB: f64 = 60.0;

// ---------- tests ----------

#[test]
fn mono_128_decodes_and_matches_ffmpeg_reference() {
    let mp3 = read_fixture("mono_128.mp3");
    let reference = read_ref_s16le("mono_128.ref.s16le");
    // A frame-unaligned chunk size so the element must buffer + resync across boundaries.
    let got = decode_via_pipeline(&mp3, 100);
    assert!(!got.is_empty(), "mono decode produced no PCM");
    let (snr, n) = best_snr_db(&got, &reference, 1);
    assert!(
        snr >= SNR_THRESHOLD_DB,
        "mono SNR {snr:.2} dB < {SNR_THRESHOLD_DB} dB threshold (overlap {n})"
    );
}

#[test]
fn joint_stereo_128_decodes_and_matches_ffmpeg_reference() {
    let mp3 = read_fixture("js_128.mp3");
    let reference = read_ref_s16le("js_128.ref.s16le");
    let got = decode_via_pipeline(&mp3, 137);
    assert!(!got.is_empty(), "joint-stereo decode produced no PCM");
    // Even sample count → whole interleaved (L, R) pairs.
    assert_eq!(got.len() % 2, 0, "stereo output is not an even sample count");
    let (snr, n) = best_snr_db(&got, &reference, 2);
    assert!(
        snr >= SNR_THRESHOLD_DB,
        "joint-stereo SNR {snr:.2} dB < {SNR_THRESHOLD_DB} dB threshold (overlap {n})"
    );
}

#[test]
fn ragged_chunk_boundaries_are_stable_across_sizes() {
    // The same bytes chopped at several awkward sizes (including 1-byte-at-a-time and a
    // size larger than the whole stream) must all decode to the *same* PCM — the
    // element's internal buffering/resync cannot depend on where a read happened to
    // split. (Byte-exact across chunkings, not just SNR-close.)
    let mp3 = read_fixture("mono_128.mp3");
    let baseline = decode_via_pipeline(&mp3, 1000);
    assert!(!baseline.is_empty());
    for &chunk in &[1usize, 3, 7, 64, 417, 511, 4096, mp3.len() + 10] {
        let got = decode_via_pipeline(&mp3, chunk);
        assert_eq!(
            got, baseline,
            "chunk size {chunk} produced a different decode than the 1000-byte chunking"
        );
    }
}

#[test]
fn id3v2_prefixed_stream_decodes_like_the_bare_stream() {
    // The ID3v2-tagged mono fixture carries the same audio as the bare mono fixture; the
    // element's synchsafe ID3v2 skip (and, as a backstop, the framer's syncword resync)
    // must reach the first audio frame and decode identically to the untagged stream.
    let bare = decode_via_pipeline(&read_fixture("mono_128.mp3"), 256);
    let tagged = decode_via_pipeline(&read_fixture("mono_128_id3.mp3"), 256);
    assert!(!tagged.is_empty(), "ID3-prefixed decode produced no PCM");
    // The audio payload is identical; only the leading ID3 tag differs, which the
    // element strips — so the decoded PCM is byte-exact with the bare stream.
    assert_eq!(tagged, bare, "ID3v2-prefixed decode differs from the bare decode");
}

#[test]
fn garbage_and_truncation_never_panic() {
    // Malformed / adversarial byte streams must be tolerated (dropped frames post bus
    // Warnings, the pipeline still reaches EOS) — never a panic. (spec: Supervision.)
    let cases: Vec<Vec<u8>> = vec![
        vec![],                              // empty
        vec![0xFF; 64],                      // all-sync-bytes, no valid header
        vec![0x00; 4096],                    // no syncword at all
        (0u8..=255).cycle().take(8192).collect(), // structured junk
        {
            // A real header truncated one byte before the frame body — the framer must
            // resync-skip it, not read past the buffer.
            let mut v = read_fixture("mono_128.mp3");
            v.truncate(200);
            v
        },
        {
            // Valid stream with a garbage island spliced into the middle.
            let good = read_fixture("mono_128.mp3");
            let mut v = good[..good.len() / 2].to_vec();
            v.extend_from_slice(&[0xAB; 300]);
            v.extend_from_slice(&good[good.len() / 2..]);
            v
        },
    ];
    for (i, bytes) in cases.into_iter().enumerate() {
        // Must not panic; produced PCM (if any) is not asserted on.
        let _ = decode_via_pipeline(&bytes, 91);
        // A second run at a different chunking exercises more boundary states.
        let _ = decode_via_pipeline(&bytes, 5);
        let _ = i;
    }
}

// ---------- ffmpeg oracle (skip-if-absent) ----------

fn have_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Decode `mp3` with ffmpeg to interleaved s16le PCM (the oracle), or `None` when
/// ffmpeg is absent or the decode fails.
fn ffmpeg_decode(mp3_path: &std::path::Path, channels: u32) -> Option<Vec<i16>> {
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(mp3_path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-ac", &channels.to_string(), "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
}

#[test]
fn oracle_cross_check_against_live_ffmpeg() {
    // Re-derive the oracle at test time from the *committed* .mp3 fixtures and confirm
    // the element still agrees — a live tripwire that the committed .ref.s16le files
    // (and the decoder) have not drifted. Skips cleanly when ffmpeg is unavailable.
    if !have_ffmpeg() {
        eprintln!("skipping oracle cross-check: ffmpeg not found");
        return;
    }
    for (name, channels) in [("mono_128.mp3", 1u32), ("js_128.mp3", 2u32)] {
        let path = fixtures_dir().join(name);
        let Some(oracle) = ffmpeg_decode(&path, channels) else {
            eprintln!("skipping {name}: ffmpeg decode failed");
            continue;
        };
        let got = decode_via_pipeline(&read_fixture(name), 333);
        let (snr, n) = best_snr_db(&got, &oracle, channels as usize);
        assert!(
            snr >= SNR_THRESHOLD_DB,
            "{name}: live-ffmpeg SNR {snr:.2} dB < {SNR_THRESHOLD_DB} dB (overlap {n})"
        );
    }
}
