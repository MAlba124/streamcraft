//! `flacdec` through the element [`Harness`] — the real dynamic-caps proof (spec: Formats
//! — dynamic caps; Testing — Element harness). This is the case the harness was built for:
//! a decoder that learns its output format from the FLAC header and **announces** it at
//! runtime, carries partially-decoded frames as bytes arrive, and flushes its tail at EOS.
//! The hand-rolled equivalent (`flac/tests/transcode.rs`) stands up a whole `Pipeline` with
//! a bespoke `PacketSrc` and a `Mutex`-guarded collect-sink to observe the same three
//! properties (announce + carry + EOS); here they are assertions on an inline caller.
//!
//! The FLAC bytes are encoded in-test with [`FlacEncoder`] (the same streamable form the
//! `flacenc` element emits and `flacdec` decodes incrementally), so there is no external
//! tool and no fixture file. The decoded PCM is cross-checked against the standalone
//! [`FlacDecoder`], so the harness path is validated against the codec's own oracle.

use pf_flac::{FlacDecoder, FlacEncoder, SampleFormat};
use profluens_core::format::Value;
use profluens_core::harness::Harness;

const RATE: u32 = 22_050;
const CHANNELS: u32 = 2;

/// A couple of detuned sines per channel — non-trivial, compressible S16 content (the
/// generator shared with `transcode.rs`/`element.rs`). Returns the interleaved LE bytes
/// and the expected per-sample `i64` values.
fn gen_s16(frames: usize) -> (Vec<u8>, Vec<i64>) {
    let mut bytes = Vec::with_capacity(frames * CHANNELS as usize * 2);
    let mut expected = Vec::with_capacity(frames * CHANNELS as usize);
    for i in 0..frames {
        for c in 0..CHANNELS {
            let phase = 2.0 * std::f64::consts::PI * (3.0 + c as f64) * (i as f64) / 384.0;
            let s = (11_000.0 * phase.sin()).round() as i64;
            expected.push(s);
            bytes.extend_from_slice(&(s as i16).to_le_bytes());
        }
    }
    (bytes, expected)
}

/// A streamable native FLAC byte stream for `pcm` — the forward-only form `flacenc` emits
/// and `flacdec` decodes incrementally.
fn encode_stream(pcm: &[u8]) -> Vec<u8> {
    let (mut enc, header) =
        FlacEncoder::new_streaming(RATE, CHANNELS, SampleFormat::S16, 4096).expect("enc");
    let mut out = header;
    enc.encode_interleaved(pcm, &mut out).expect("encode");
    out
}

/// Like [`encode_stream`], but splices a `VORBIS_COMMENT` (and optionally a `PICTURE`) metadata
/// block after STREAMINFO — a valid FLAC tag chain (`fLaC` + STREAMINFO(not-last) +
/// VORBIS_COMMENT(not-last if a picture follows) + [PICTURE(last)] + frames).
fn encode_stream_with_tags(pcm: &[u8], comments: &[&str], picture: Option<(&str, &[u8])>) -> Vec<u8> {
    let (mut enc, mut header) =
        FlacEncoder::new_streaming(RATE, CHANNELS, SampleFormat::S16, 4096).expect("enc");
    header[4] &= 0x7f; // clear STREAMINFO's last-metadata-block flag

    // VORBIS_COMMENT — last unless a PICTURE block follows it.
    let mut vc = Vec::new();
    vc.extend_from_slice(&3u32.to_le_bytes());
    vc.extend_from_slice(b"ref"); // vendor string
    vc.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in comments {
        vc.extend_from_slice(&(c.len() as u32).to_le_bytes());
        vc.extend_from_slice(c.as_bytes());
    }
    header.push(if picture.is_some() { 4 } else { 0x80 | 4 });
    header.extend_from_slice(&(vc.len() as u32).to_be_bytes()[1..]);
    header.extend_from_slice(&vc);

    if let Some((mime, data)) = picture {
        let mut pic = Vec::new();
        pic.extend_from_slice(&3u32.to_be_bytes()); // picture type: front cover
        pic.extend_from_slice(&(mime.len() as u32).to_be_bytes());
        pic.extend_from_slice(mime.as_bytes());
        pic.extend_from_slice(&0u32.to_be_bytes()); // description length
        pic.extend_from_slice(&[0u8; 16]); // width, height, colour depth, indexed colours
        pic.extend_from_slice(&(data.len() as u32).to_be_bytes());
        pic.extend_from_slice(data);
        header.push(0x80 | 6); // last-block | PICTURE
        header.extend_from_slice(&(pic.len() as u32).to_be_bytes()[1..]);
        header.extend_from_slice(&pic);
    }

    let mut out = header;
    enc.encode_interleaved(pcm, &mut out).expect("encode");
    out
}

/// Reassemble a run of little-endian s16 PCM bytes into a flat `i64` sample vector.
fn samples_from_bytes(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i64)
        .collect()
}

#[test]
fn flacdec_announces_format_and_decodes_pcm() {
    let frames = 4_000; // several FLAC frames (block size 4096 → a couple of frames)
    let (pcm, expected) = gen_s16(frames);
    let flac = encode_stream(&pcm);

    let mut h = Harness::new(pf_flac::FlacDec::new());

    // Feed the FLAC byte stream in small chunks — the incremental, arrives-a-bit-at-a-time
    // shape a real byte source produces — running `process()` after each, collecting every
    // decoded PCM buffer as it is emitted.
    let mut decoded_bytes: Vec<u8> = Vec::new();
    for chunk in flac.chunks(700) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("flacdec process");
        while let Some(out) = h.pull("src") {
            decoded_bytes.extend_from_slice(out.memory.data());
        }
    }

    // The announcement fired the moment the first frame's header decoded (spec: dynamic
    // caps), resolved through the harness vocabulary exactly as the scheduler resolves it.
    let ann = h.announced().expect("flacdec announced its audio/raw format");
    let rate_id = h.vocabulary().field_id("rate").expect("rate interned from offers");
    let channels_id = h.vocabulary().field_id("channels").expect("channels interned");
    let sample_id = h.vocabulary().field_id("sample").expect("sample interned");
    assert_eq!(h.vocabulary().family_name(ann.family), Some("audio/raw"));
    assert_eq!(ann.get(rate_id), Some(Value::Int(RATE as i64)), "announced rate");
    assert_eq!(ann.get(channels_id), Some(Value::Int(CHANNELS as i64)), "announced channels");
    // The categorical sample format resolves back to s16 (16-bit input), proving the Id
    // (categorical value) lowering path works through the harness.
    match ann.get(sample_id) {
        Some(Value::Id(v)) => assert_eq!(h.vocabulary().value_name(v), Some("s16")),
        other => panic!("expected a categorical sample format, got {other:?}"),
    }

    // EOS flushes any remaining whole frames through the unbounded pool (the tail-drain).
    for buf in h.eos().expect("eos flush") {
        decoded_bytes.extend_from_slice(buf.memory.data());
    }

    // --- decoded PCM: lossless, complete, in order ---
    let got = samples_from_bytes(&decoded_bytes);
    assert_eq!(got.len(), expected.len(), "decoded sample count matches the input");
    assert_eq!(got, expected, "decoded PCM is lossless and correctly ordered");

    // --- cross-check against the codec's own decoder (the oracle) ---
    let oracle = FlacDecoder::decode(&flac).expect("standalone decode");
    assert_eq!(oracle.info.sample_rate, RATE);
    assert_eq!(oracle.info.channels, CHANNELS);
    assert_eq!(got, oracle.samples, "harness path matches the standalone decoder");
}

/// The parser element surfaces the FLAC's Vorbis comments to the application as a `BusMessage::Tags`
/// — the out-of-band half of the GStreamer-style tag flow (the in-band `Event::Tags` travels
/// downstream to a muxer). Emitted once, when the header is decoded.
#[test]
fn flacdec_posts_tags_to_the_bus() {
    use profluens_core::bus::BusMessage;

    let (pcm, _) = gen_s16(4_000);
    let art = b"\xFF\xD8\xFF\xE0jpeg-ish cover-art bytes";
    let flac = encode_stream_with_tags(
        &pcm,
        &["TITLE=Set Theory", "ARTIST=Carbon Based Lifeforms"],
        Some(("image/jpeg", art)),
    );

    let mut h = Harness::new(pf_flac::FlacDec::new());
    for chunk in flac.chunks(700) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("flacdec process");
        while h.pull("src").is_some() {} // drain decoded PCM
    }
    for _ in h.eos().expect("eos flush") {}

    let tag_lists: Vec<_> = h
        .bus_messages()
        .into_iter()
        .filter_map(|m| match m {
            BusMessage::Tags { tags, .. } => Some(tags),
            _ => None,
        })
        .collect();
    assert_eq!(tag_lists.len(), 1, "tags posted exactly once");
    let tags = &tag_lists[0];
    assert_eq!(tags.get("TITLE"), Some("Set Theory"));
    assert_eq!(tags.get("ARTIST"), Some("Carbon Based Lifeforms"));
    assert_eq!(tags.get("artist"), Some("Carbon Based Lifeforms"), "lookup is case-insensitive");

    // The cover art is forwarded too — MIME + bytes intact, held zero-copy.
    assert_eq!(tags.pictures().len(), 1, "one attached picture");
    assert_eq!(&*tags.pictures()[0].mime, "image/jpeg");
    assert_eq!(tags.pictures()[0].data.data(), art, "image bytes forwarded intact");
}
