//! `PacketSrc(ASC + raw AUs) ! AacDec ! PcmRecordSink` — the container-shaped AAC
//! decode path (spec: Formats — dynamic caps; the adoption gate). The committed
//! fixture is an ffmpeg-encoded AAC-LC ADTS stream; the test strips the ADTS
//! framing (ISO/IEC 14496-3 §1.A.2.2) into the raw `raw_data_block()` AUs our
//! demuxers deliver, synthesizes the matching 2-byte AudioSpecificConfig
//! (§1.6.2.1) as the in-band head, and scores the decode against ffmpeg's own
//! decode of the same stream by best-shift SNR (the `pf-mp3` conformance bar:
//! decoders differ by output delay, so alignment is searched, fidelity is
//! scored).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pf_aac::AacDec;

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

// ---------- ADTS → (ASC, raw AUs) ----------

/// Strip an ADTS stream (ISO/IEC 14496-3 §1.A.2.2) into its raw AUs plus the
/// synthesized AudioSpecificConfig its fixed header implies. The header is 7
/// octets (9 with a CRC — `protection_absent == 0`); `aac_frame_length` counts
/// header + payload.
fn adts_to_aus(data: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut aus = Vec::new();
    let mut asc = Vec::new();
    let mut at = 0usize;
    while at + 7 <= data.len() {
        assert_eq!(data[at], 0xFF, "ADTS syncword");
        assert_eq!(data[at + 1] & 0xF0, 0xF0, "ADTS syncword");
        let protection_absent = data[at + 1] & 1;
        let profile = data[at + 2] >> 6; // `profile` = AOT - 1
        let sf_index = (data[at + 2] >> 2) & 0xF;
        let chan_cfg = ((data[at + 2] & 1) << 2) | (data[at + 3] >> 6);
        let frame_len = ((data[at + 3] as usize & 0x3) << 11)
            | ((data[at + 4] as usize) << 3)
            | (data[at + 5] as usize >> 5);
        let hdr = if protection_absent == 1 { 7 } else { 9 };
        if asc.is_empty() {
            // §1.6.2.1: AOT(5) | samplingFrequencyIndex(4) | channelConfiguration(4)
            // | GASpecificConfig{frameLengthFlag, dependsOnCoreCoder, extensionFlag}(3).
            let aot = profile + 1;
            let bits: u16 = (u16::from(aot) << 11)
                | (u16::from(sf_index) << 7)
                | (u16::from(chan_cfg) << 3);
            asc = bits.to_be_bytes().to_vec();
        }
        aus.push(data[at + hdr..at + frame_len].to_vec());
        at += frame_len;
    }
    (asc, aus)
}

// ---------- test elements (the pf-mp3 harness shapes) ----------

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

/// Emits pre-built packets in order — one per `process` pass — then EOS. For AAC
/// that is the ASC head first, then one raw AU per buffer (the demuxer contract).
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

/// Concatenates every received PCM byte, in order.
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

fn decode_via_pipeline(packets: Vec<Vec<u8>>) -> Vec<i16> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(PacketSrc { packets, next: 0 });
    let dec = p.add(AacDec::new());
    let sink = p.add(PcmRecordSink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).expect("src ! aacdec");
    p.link((dec, "src"), (sink, "sink")).expect("aacdec ! sink");
    p.run().expect("pipeline run");
    let bytes = got.lock().unwrap().clone();
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

// ---------- SNR scorer (best-shift aligned; first frame skipped) ----------

/// SNR in dB of `got` against `reference` (both interleaved i16). AAC decoders
/// differ by a whole-frame output-delay offset (1024-sample frames plus encoder
/// delay), so a small shift window is searched and the first frame's transient
/// skipped — this grades decode fidelity, not the trim.
fn best_snr_db(got: &[i16], reference: &[i16], channels: usize) -> (f64, usize) {
    let frame = 1024 * channels;
    let max_shift = (8 * frame) as i64;
    let step = channels as i64;
    let skip = frame;

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

/// Two conforming AAC decoders inter-agree far above this; 40 dB still fails
/// loudly on a genuinely broken decode (ISO/IEC 14496-26 accuracy is stricter,
/// but that grades against the reference decoder, not between implementations).
const SNR_THRESHOLD_DB: f64 = 40.0;

// ---------- tests ----------

#[test]
fn stereo_lc_decodes_and_matches_ffmpeg_reference() {
    let adts = read_fixture("tone.adts");
    let reference = read_ref_s16le("tone.ref.s16le");
    let (asc, aus) = adts_to_aus(&adts);
    assert!(!aus.is_empty(), "fixture stripped to no AUs");
    let mut packets = vec![asc];
    packets.extend(aus);
    let got = decode_via_pipeline(packets);
    assert!(!got.is_empty(), "decode produced no PCM");
    let (snr, n) = best_snr_db(&got, &reference, 2);
    eprintln!("aac stereo LC vs ffmpeg: SNR {snr:.2} dB over {n} samples");
    assert!(
        snr >= SNR_THRESHOLD_DB,
        "stereo LC SNR {snr:.2} dB < {SNR_THRESHOLD_DB} dB threshold (overlap {n})"
    );
}

/// Garbage AUs after a valid config are warned-and-dropped, never fatal, and
/// decode resumes on the next valid AU (spec: Supervision — per-buffer scope).
#[test]
fn garbage_aus_are_dropped_not_fatal() {
    let adts = read_fixture("tone.adts");
    let (asc, aus) = adts_to_aus(&adts);
    let n_aus = aus.len();
    let mut packets = vec![asc];
    packets.extend(aus.into_iter().take(4));
    packets.push((0u8..=255).cycle().take(512).collect()); // structured junk
    packets.push(vec![0xFF; 64]);
    let got = decode_via_pipeline(packets);
    assert!(
        got.len() >= 3 * 1024 * 2,
        "valid AUs before the junk still decoded (got {} samples, {n_aus} AUs total)",
        got.len()
    );
}
