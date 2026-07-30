//! FLAC-in-Ogg de-framing, bit-exact end to end (spec: `flac/spec/ogg-flac-mapping.md`;
//! xiph "Ogg Mapping for FLAC").
//!
//! The load-bearing property: **PCM → FLAC-encode → Ogg-FLAC mux → de-frame → decode** is
//! bit-exact. We build a real Ogg-FLAC byte stream (mapping header packet + one metadata
//! block per packet + one FLAC frame per packet, muxed into pages by the library
//! [`OggMux`]), then run `oggdemux ! oggflacdeframe ! flacdec` in a pipeline and assert the
//! decoded PCM equals the input sample-for-sample. `oggflacdeframe` is the element under
//! test; the mux side is the tested `pf-ogg` writer, so this exercises exactly the
//! milestone pipeline minus the sink: `filesrc ! oggdemux ! oggflacdeframe ! flacdec`.

use std::sync::{Arc, Mutex};

use pf_flac::{FlacDec, FlacDecoder, FlacEncoder, OggFlacDeframe, SampleFormat};
use pf_ogg::{OggDemux, OggMux};
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

// =====================================================================================
// Building an Ogg-FLAC stream from raw PCM
// =====================================================================================

/// A couple of detuned sines per channel — non-trivial, compressible content (mirrors
/// `streaming.rs`/`element.rs`).
fn signal(frames: usize, channels: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(frames * channels as usize);
    for i in 0..frames {
        for c in 0..channels {
            let phase = 2.0 * std::f64::consts::PI * (3.0 + c as f64) * i as f64 / 200.0;
            v.push((9000.0 * phase.sin()) as i16);
        }
    }
    v
}

/// Interleaved S16 LE bytes for `samples` (channel-major within each interchannel sample).
fn interleave_le(samples: &[i16]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

/// Encode `samples` to the list of native-FLAC **packets** the FLAC-to-Ogg mapping wants:
/// packet 0 = mapping header (`0x7F "FLAC" 1 0 <count:be16>` + native `fLaC` + STREAMINFO),
/// then any further metadata blocks (this encoder emits none), then **one packet per FLAC
/// frame** (spec: the mapping doc §1–§3). Frames are produced one at a time by feeding the
/// encoder one block of samples per call, so each `encode_interleaved` appends exactly one
/// frame whose bytes become one packet.
fn encode_ogg_flac_packets(samples: &[i16], channels: u32, rate: u32) -> Vec<Vec<u8>> {
    let pcm = interleave_le(samples);
    let bytes_per_interchannel = 2 * channels as usize;

    let (mut enc, mut header) = FlacEncoder::new(rate, channels, SampleFormat::S16).expect("enc");
    let block = enc.max_block_size() as usize; // interchannel samples per frame

    // Encode frame by frame, capturing each frame's exact byte range.
    let mut frame_packets: Vec<Vec<u8>> = Vec::new();
    let n_interchannel = samples.len() / channels as usize;
    let mut off = 0usize;
    while off < n_interchannel {
        let take = block.min(n_interchannel - off);
        let start_byte = off * bytes_per_interchannel;
        let end_byte = (off + take) * bytes_per_interchannel;
        let mut one = Vec::new();
        enc.encode_interleaved(&pcm[start_byte..end_byte], &mut one)
            .expect("encode one block");
        // Exactly one block in → exactly one frame out (block <= max_block_size).
        frame_packets.push(one);
        off += take;
    }
    let body = enc.finish();
    // Patch the finalised STREAMINFO body over the placeholder in the native header.
    header[pf_flac::streaminfo_offset()..pf_flac::streaminfo_offset() + body.len()]
        .copy_from_slice(&body);

    // Split the native header into its metadata blocks: `fLaC` (4) + a chain of
    // [4-byte block header][body]. This encoder emits STREAMINFO only (last-block flag set),
    // so there is exactly one block, but parse the chain generally.
    assert_eq!(&header[..4], b"fLaC", "native FLAC signature");
    let mut blocks: Vec<&[u8]> = Vec::new();
    let mut p = 4usize;
    loop {
        let bh = &header[p..p + 4];
        let last = bh[0] & 0x80 != 0;
        let len = ((bh[1] as usize) << 16) | ((bh[2] as usize) << 8) | bh[3] as usize;
        blocks.push(&header[p..p + 4 + len]);
        p += 4 + len;
        if last {
            break;
        }
    }
    assert_eq!(p, header.len(), "metadata chain consumes the whole native header");

    // Packet 0: the mapping header. `0x7F "FLAC"` + major.minor version + big-endian count
    // of the *further* header packets (metadata blocks after STREAMINFO), then native `fLaC`
    // + the STREAMINFO block (blocks[0]).
    let further_headers = (blocks.len() - 1) as u16;
    let mut mapping = vec![0x7F];
    mapping.extend_from_slice(b"FLAC");
    mapping.extend_from_slice(&[0x01, 0x00]); // mapping version 1.0
    mapping.extend_from_slice(&further_headers.to_be_bytes()); // header packet count, big-endian
    mapping.extend_from_slice(b"fLaC");
    mapping.extend_from_slice(blocks[0]); // STREAMINFO (block header + body)

    let mut packets: Vec<Vec<u8>> = Vec::new();
    packets.push(mapping);
    // Any further metadata blocks, one per packet (none for this encoder, but general).
    for b in &blocks[1..] {
        packets.push(b.to_vec());
    }
    // One packet per FLAC frame.
    packets.extend(frame_packets);
    packets
}

// =====================================================================================
// A source that emits a fixed list of byte buffers (one per packet), then EOS. Same shape
// as `ogg/tests/element_roundtrip.rs`'s PacketSrc.
// =====================================================================================

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "packetsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active, // a real source: its own group, drives the pipeline
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Emits each buffer in `packets` in order (one `process` call each), then `Flow::Eos`.
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
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok), // pool full → try again next turn
        };
        assert!(
            pkt.len() <= buf.memory.capacity(),
            "test packet {} ({} bytes) exceeds pool slot ({})",
            self.next,
            pkt.len(),
            buf.memory.capacity()
        );
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

// =====================================================================================
// A sink that concatenates decoded PCM bytes in order.
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("audio/raw")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "pcmsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active, // its own group, so buffers cross a real ring
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Concatenates every decoded-PCM byte it receives, in order.
struct PcmSink {
    got: Arc<Mutex<Vec<u8>>>,
}

impl Element for PcmSink {
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

// =====================================================================================
// Tests
// =====================================================================================

/// Decode the decoded-PCM bytes back into interleaved i64 samples (S16 LE).
fn pcm_to_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i64)
        .collect()
}

/// Run the milestone pipeline (minus the audio sink): a source emitting the Ogg-FLAC
/// packets → `oggmux` (bytes) → `oggdemux` → `oggflacdeframe` → `flacdec` → a PCM sink.
/// The `oggmux`+`oggdemux` pair reproduces a real container round-trip on the byte stream,
/// and `oggflacdeframe` is the element under test.
fn run_pipeline(packets: Vec<Vec<u8>>) -> Vec<u8> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(PacketSrc { packets, next: 0 });
    let mux = p.add(OggMux::new());
    let demux = p.add(OggDemux::new());
    let deframe = p.add(OggFlacDeframe::new());
    let dec = p.add(FlacDec::new());
    let sink = p.add(PcmSink { got: Arc::clone(&got) });
    p.link((src, "src"), (mux, "sink")).expect("src->mux");
    p.link((mux, "src"), (demux, "sink")).expect("mux->demux");
    p.link((demux, "src"), (deframe, "sink")).expect("demux->deframe");
    p.link((deframe, "src"), (dec, "sink")).expect("deframe->flacdec");
    p.link((dec, "src"), (sink, "sink")).expect("flacdec->sink");
    p.run().expect("run");
    let out = got.lock().unwrap().clone();
    out
}

#[test]
fn ogg_flac_deframe_roundtrip_is_bit_exact() {
    // A spread of sizes/channel counts: sub-block, multi-frame, mono, and stereo — so the
    // per-frame packetisation and the reconstructed metadata/frame boundary both run.
    for &(frames, ch, rate) in &[
        (1usize, 2u32, 44100u32),   // one interchannel sample, single short frame
        (600, 1, 48000),            // sub-block mono
        (10_000, 2, 44100),         // several frames (> 4608 block), stereo
        (13_337, 1, 44100),         // several frames, mono, non-block-multiple
    ] {
        let samples = signal(frames, ch);
        let expected: Vec<i64> = samples.iter().map(|&s| s as i64).collect();

        // Sanity: the native encode itself is lossless (isolates any Ogg-mapping bug from a
        // codec bug), and the mapping packets reconstruct exactly the native stream.
        let packets = encode_ogg_flac_packets(&samples, ch, rate);
        assert!(packets.len() >= 2, "at least the mapping header + one frame packet");

        let native = reconstruct_native(&packets);
        let ref_dec = FlacDecoder::decode(&native).expect("native decode");
        assert_eq!(ref_dec.samples, expected, "native FLAC decode lossless (frames={frames} ch={ch})");
        assert_eq!(ref_dec.info.channels, ch);
        assert_eq!(ref_dec.info.sample_rate, rate);

        // The pipeline: oggmux ! oggdemux ! oggflacdeframe ! flacdec must be bit-exact.
        let pcm = run_pipeline(packets);
        let decoded = pcm_to_i64(&pcm);
        assert_eq!(
            decoded, expected,
            "Ogg-FLAC de-frame round trip must be bit-exact (frames={frames} ch={ch} rate={rate})"
        );
    }
}

/// Reconstruct the native FLAC byte stream from the Ogg-FLAC packet list exactly as
/// `OggFlacDeframe` does — strip the 9-byte mapping prefix from packet 0, concatenate the
/// rest — so the test's expectation of "native stream" matches the element's contract.
fn reconstruct_native(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut native = Vec::new();
    native.extend_from_slice(&packets[0][9..]); // drop 0x7F "FLAC" v.v count
    for pkt in &packets[1..] {
        native.extend_from_slice(pkt);
    }
    native
}

#[test]
fn mapping_header_packet_has_the_expected_shape() {
    // Guard the exact bytes the mapping header packet carries (spec: the mapping doc §1):
    // 0x7F "FLAC" 0x01 0x00 <count:be16=0> "fLaC" <STREAMINFO 38 bytes>.
    let samples = signal(100, 2);
    let packets = encode_ogg_flac_packets(&samples, 2, 44100);
    let h = &packets[0];
    assert_eq!(h[0], 0x7F, "mapping packet type");
    assert_eq!(&h[1..5], b"FLAC", "mapping signature");
    assert_eq!(&h[5..7], &[0x01, 0x00], "mapping version 1.0");
    // This encoder emits STREAMINFO only, so zero further header packets.
    assert_eq!(&h[7..9], &0u16.to_be_bytes(), "header packet count 0");
    assert_eq!(&h[9..13], b"fLaC", "native FLAC signature after the mapping prefix");
    // STREAMINFO: 4-byte block header (last-block flag set, type 0, length 34) + 34 body.
    assert_eq!(h[13] & 0x80, 0x80, "STREAMINFO is the last metadata block");
    assert_eq!(h[13] & 0x7F, 0, "block type 0 == STREAMINFO");
    assert_eq!(h.len(), 9 + 4 + 4 + 34, "mapping packet is 9 + fLaC + STREAMINFO");
}
