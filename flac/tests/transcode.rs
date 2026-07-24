//! `flacdec ! flacenc` — the transcode link (spec: Formats — dynamic caps, consumer
//! side). Regression for the parse-launch failure `no common format between
//! flacdec.src and flacenc.sink (offers ["audio/raw"] vs ["bytes"])`: `flacenc`'s
//! sink pad must speak `audio/raw` so a decoder links directly, and the encoder must
//! adopt rate/channels/sample from the *announced* input format — the constructor
//! values here are deliberately wrong, so a lossless, correctly-labelled output
//! proves the runtime `FormatChange` learning path end to end.

use std::sync::{Arc, Mutex};

use sc_flac::{FlacDec, FlacDecoder, FlacEnc, FlacEncoder, SampleFormat};
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

const RATE: u32 = 22_050;
const CHANNELS: u32 = 2;

/// A couple of detuned sines per channel — non-trivial, compressible content
/// (mirrors `element.rs`/`oggflac_roundtrip.rs`).
fn gen_s16(n: usize) -> (Vec<u8>, Vec<i64>) {
    let mut bytes = Vec::with_capacity(n * CHANNELS as usize * 2);
    let mut expected = Vec::with_capacity(n * CHANNELS as usize);
    for i in 0..n {
        for c in 0..CHANNELS {
            let phase = 2.0 * std::f64::consts::PI * (3.0 + c as f64) * (i as f64) / 384.0;
            let s = (11000.0 * phase.sin()).round() as i64;
            expected.push(s);
            bytes.extend_from_slice(&(s as i16).to_le_bytes());
        }
    }
    (bytes, expected)
}

/// A streamable native FLAC byte stream for `pcm` — the same forward-only form the
/// `flacenc` element emits, which `flacdec` decodes incrementally.
fn encode_stream(pcm: &[u8]) -> Vec<u8> {
    let (mut enc, header) =
        FlacEncoder::new_streaming(RATE, CHANNELS, SampleFormat::S16, 4096).expect("enc");
    let mut out = header;
    enc.encode_interleaved(pcm, &mut out).expect("encode");
    out
}

// --- test elements (the PacketSrc/collect-sink shapes from oggflac_roundtrip.rs) ---

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

/// Emits chunks in order — up to `burst` per `process` call, as fast as the pool
/// allows (an IO source's shape: several completions per pass) — then EOS.
struct PacketSrc {
    packets: Vec<Vec<u8>>,
    next: usize,
    burst: usize,
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
        for _ in 0..self.burst {
            if self.next >= self.packets.len() {
                return Ok(Flow::Eos);
            }
            let pkt = &self.packets[self.next];
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            buf.memory.as_mut_full()[..pkt.len()].copy_from_slice(pkt);
            buf.memory.set_len(pkt.len());
            ctx.out(PadId(0)).push(buf);
            self.next += 1;
        }
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
    name: "bytesink",
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

/// Concatenates every received byte, in order.
struct ByteSink {
    got: Arc<Mutex<Vec<u8>>>,
}

impl Element for ByteSink {
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

// --- tests ---

#[test]
fn flacdec_links_to_flacenc_over_audio_raw() {
    // The exact link the bug report failed on: flacdec.src (audio/raw) into
    // flacenc.sink — must negotiate, and must pick the typed family.
    let mut p = Pipeline::new();
    let dec = p.add(FlacDec::new());
    let enc = p.add(FlacEnc::new(44_100, 2, SampleFormat::S16));
    let link = p.link((dec, "src"), (enc, "sink")).expect("flacdec ! flacenc links");
    let fam = p.negotiated(link).expect("format fixed").family;
    assert_eq!(p.family_name(fam), Some("audio/raw"), "typed family preferred over bytes");
}

#[test]
fn transcode_is_lossless_and_learns_the_announced_format() {
    // 10_000 interchannel samples: several full blocks plus a short final frame.
    let (pcm, expected) = gen_s16(10_000);
    let stream = encode_stream(&pcm);
    let packets: Vec<Vec<u8>> = stream.chunks(1500).map(<[u8]>::to_vec).collect();

    let mut p = Pipeline::new();
    let src = p.add(PacketSrc { packets, next: 0, burst: 1 });
    let dec = p.add(FlacDec::new());
    // Deliberately wrong constructor parameters: only the FormatChange announced by
    // flacdec (22050 Hz stereo s16, from STREAMINFO) can make the output correct.
    let enc = p.add(FlacEnc::new(8_000, 1, SampleFormat::S32));
    let got = Arc::new(Mutex::new(Vec::new()));
    let sink = p.add(ByteSink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).expect("src!dec");
    p.link((dec, "src"), (enc, "sink")).expect("dec!enc");
    p.link((enc, "src"), (sink, "sink")).expect("enc!sink");
    p.run().expect("run");

    let flac = got.lock().unwrap().clone();
    assert!(flac.starts_with(b"fLaC"), "re-encoded output is a FLAC stream");
    let out = FlacDecoder::decode(&flac).expect("re-encoded stream decodes");
    assert_eq!(out.info.sample_rate, RATE, "rate learned from the announcement");
    assert_eq!(out.info.channels, CHANNELS, "channels learned from the announcement");
    assert_eq!(out.info.bits_per_sample, 16, "sample format learned from the announcement");
    assert_eq!(out.samples, expected, "decode → re-encode → decode is lossless");
}

#[test]
fn bursty_source_with_starved_pool_does_not_deadlock() {
    // Regression: `filesrc ! flacdec ! flacenc ! filesink` livelocked — the source,
    // inlined ahead of the latency-bounded decoder, free-ran until every pool slot
    // sat in the decoder's unconsumed input; the decoder then couldn't allocate its
    // own output, never consumed, and no slot ever came back. The scheduler's inline
    // backpressure gate (an element is not run while its successor holds a backlog)
    // must keep this flowing. Reproduced deliberately small: a 16-slot / 4 KiB pool
    // and a source bursting 4 packets per pass, with a run-thread timeout as the
    // deadlock detector.
    let (pcm, expected) = gen_s16(60_000);
    let stream = encode_stream(&pcm);
    // Small chunks so the packet count dwarfs the pool regardless of how well the
    // sines compress.
    let packets: Vec<Vec<u8>> = stream.chunks(512).map(<[u8]>::to_vec).collect();
    assert!(packets.len() > 64, "enough packets to overwhelm a 16-slot pool");

    let mut p = Pipeline::new();
    p.set_pool(4096, 16);
    let src = p.add(PacketSrc { packets, next: 0, burst: 4 });
    let dec = p.add(FlacDec::new());
    let enc = p.add(FlacEnc::new(8_000, 1, SampleFormat::S32));
    let got = Arc::new(Mutex::new(Vec::new()));
    let sink = p.add(ByteSink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).expect("src!dec");
    p.link((dec, "src"), (enc, "sink")).expect("dec!enc");
    p.link((enc, "src"), (sink, "sink")).expect("enc!sink");

    let run = std::thread::spawn(move || p.run());
    for _ in 0..600 {
        if run.is_finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(run.is_finished(), "transcode deadlocked under a starved pool");
    run.join().expect("joined").expect("run ok");

    let flac = got.lock().unwrap().clone();
    let out = FlacDecoder::decode(&flac).expect("output decodes");
    assert_eq!(out.samples, expected, "still lossless under pool pressure");
}
