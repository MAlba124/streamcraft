//! Lacing-mode coverage for `MatroskaReader` (spec: `mkv/spec/MATROSKA.md`; RFC 9559 §10.3).
//!
//! The `MatroskaWriter` only emits **no-lacing** SimpleBlocks (one frame per block), so the
//! reader's Xiph / EBML / fixed-size lacing paths are exercised here against **hand-built**
//! laced blocks — the same worked examples the RFC gives (800/500/1000-octet frames), plus a
//! FLAC track head so a full, valid stream is parsed end to end. A laced block unpacks into
//! several frames sharing the block timestamp (RFC 9559 §10.3.5), and each mode's size coding
//! is verified frame-for-frame.

use pf_mkv::ebml::{self, id};
use pf_mkv::MatroskaReader;

/// Build a minimal but valid MKV stream: EBML Header + open Segment + Info(TimestampScale) +
/// Tracks(one A_FLAC entry) + one open Cluster(Timestamp=0) + the given raw `block_bytes`
/// appended as-is (each already a full `SimpleBlock` element). Lets a test craft laced blocks
/// the writer never produces while still driving the whole reader.
fn build_stream(block_bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();

    // EBML Header (minimal — the reader skips it as opaque, but it must parse).
    ebml::write_id(&mut out, id::EBML);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_uint(&mut out, id::EBML_VERSION, 1);
    ebml::write_string(&mut out, id::DOC_TYPE, "matroska");
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    // Open Segment (unknown size).
    ebml::write_id(&mut out, id::SEGMENT);
    ebml::write_unknown_size(&mut out);

    // Info with the default TimestampScale (1 ms/tick).
    ebml::write_id(&mut out, id::INFO);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_uint(&mut out, id::TIMESTAMP_SCALE, 1_000_000);
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    // Tracks with one A_FLAC audio entry (track 1).
    ebml::write_id(&mut out, id::TRACKS);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_id(&mut out, id::TRACK_ENTRY);
    let te_at = ebml::reserve_size(&mut out);
    let te_body = out.len();
    ebml::write_uint(&mut out, id::TRACK_NUMBER, 1);
    ebml::write_uint(&mut out, id::TRACK_TYPE, 2); // audio
    ebml::write_string(&mut out, id::CODEC_ID, "A_FLAC");
    let te_len = (out.len() - te_body) as u64;
    ebml::patch_size(&mut out, te_at, te_len);
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    // Open Cluster at base timestamp 0, then the caller's raw block bytes.
    ebml::write_id(&mut out, id::CLUSTER);
    ebml::write_unknown_size(&mut out);
    ebml::write_uint(&mut out, id::TIMESTAMP, 0);
    out.extend_from_slice(block_bytes);

    out
}

/// Wrap a block **body** (track VINT + rel-ts + flags + lace) as a full `SimpleBlock` element.
fn simple_block(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    ebml::write_id(&mut out, id::SIMPLE_BLOCK);
    ebml::write_size(&mut out, body.len() as u64);
    out.extend_from_slice(body);
    out
}

/// The common block-body prefix: track 1 (VINT 0x81), rel-ts 0, then the given flags byte.
fn block_prefix(flags: u8) -> Vec<u8> {
    let mut b = vec![0x81]; // track number 1 as a VINT
    b.extend_from_slice(&0i16.to_be_bytes()); // relative timestamp 0
    b.push(flags);
    b
}

/// Decode `stream` fully and collect every frame's bytes for track 1, in order.
fn decode_frames(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut r = MatroskaReader::new();
    r.push(stream).expect("valid stream parses");
    let mut got = Vec::new();
    while let Some(f) = r.next_frame() {
        assert_eq!(f.track_number, 1);
        got.push(f.data);
    }
    got
}

/// Three distinct frames (their byte value marks which frame) to make routing/splitting
/// failures obvious: 800 × 0xA1, 500 × 0xB2, 1000 × 0xC3 — the RFC's worked-example sizes.
fn example_frames() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (vec![0xA1u8; 800], vec![0xB2u8; 500], vec![0xC3u8; 1000])
}

#[test]
fn xiph_lacing_splits_three_frames() {
    let (f1, f2, f3) = example_frames();
    // Flags: LACING = 01b (Xiph) → bits 0x02.
    let mut body = block_prefix(0x02);
    // Xiph lace head (RFC 9559 §10.3.2): count-1 = 2, then sizes of all but the last frame as
    // 0xFF runs + a final < 0xFF octet: 800 = 255,255,255,35; 500 = 255,245.
    body.push(2); // number of frames minus 1
    body.extend_from_slice(&[0xFF, 0xFF, 0xFF, 35]); // 800
    body.extend_from_slice(&[0xFF, 245]); // 500
    body.extend_from_slice(&f1);
    body.extend_from_slice(&f2);
    body.extend_from_slice(&f3);

    let stream = build_stream(&simple_block(&body));
    let got = decode_frames(&stream);
    assert_eq!(got, vec![f1, f2, f3], "Xiph lace splits into the three frames, in order");
}

#[test]
fn ebml_lacing_splits_three_frames() {
    let (f1, f2, f3) = example_frames();
    // Flags: LACING = 11b (EBML) → bits 0x06.
    let mut body = block_prefix(0x06);
    // EBML lace head (RFC 9559 §10.3.3): count-1 = 2; first size 800 as a VINT (0x4320); then
    // the delta 500-800 = -300 as a signed VINT (0x5ED3 per the RFC worked example).
    body.push(2); // number of frames minus 1
    body.extend_from_slice(&[0x43, 0x20]); // 800 = 0x320 | 0x4000 (2-octet VINT)
    body.extend_from_slice(&[0x5E, 0xD3]); // -300 signed: 0x1FFF - 0x12C = 0x1ED3 | 0x4000
    body.extend_from_slice(&f1);
    body.extend_from_slice(&f2);
    body.extend_from_slice(&f3);

    let stream = build_stream(&simple_block(&body));
    let got = decode_frames(&stream);
    assert_eq!(got, vec![f1, f2, f3], "EBML lace splits into the three frames, in order");
}

#[test]
fn fixed_lacing_splits_equal_frames() {
    // Fixed-size lacing (RFC 9559 §10.3.4): no sizes stored; three equal 800-octet frames.
    let f1 = vec![0xA1u8; 800];
    let f2 = vec![0xB2u8; 800];
    let f3 = vec![0xC3u8; 800];
    // Flags: LACING = 10b (fixed) → bits 0x04.
    let mut body = block_prefix(0x04);
    body.push(2); // number of frames minus 1 (3 frames)
    body.extend_from_slice(&f1);
    body.extend_from_slice(&f2);
    body.extend_from_slice(&f3);

    let stream = build_stream(&simple_block(&body));
    let got = decode_frames(&stream);
    assert_eq!(got, vec![f1, f2, f3], "fixed lace splits into equal-size frames");
}

#[test]
fn no_lacing_is_one_frame() {
    // The writer's normal path: LACING = 00b, one frame is the whole payload.
    let frame = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
    let mut body = block_prefix(0x00);
    body.extend_from_slice(&frame);
    let stream = build_stream(&simple_block(&body));
    let got = decode_frames(&stream);
    assert_eq!(got, vec![frame], "no-lacing block is exactly one frame");
}

#[test]
fn laced_frames_share_the_block_timestamp() {
    // All frames in a lace carry the block's timestamp (RFC 9559 §10.3.5). Cluster base is 0,
    // rel-ts 0, scale 1 ms → every frame pts is 0 ns.
    let mut body = block_prefix(0x04); // fixed lacing
    body.push(2);
    // Equal-size frames for fixed lacing.
    let (f1, f2, f3) = (vec![1u8; 300], vec![2u8; 300], vec![3u8; 300]);
    body.extend_from_slice(&f1);
    body.extend_from_slice(&f2);
    body.extend_from_slice(&f3);

    let stream = build_stream(&simple_block(&body));
    let mut r = MatroskaReader::new();
    r.push(&stream).unwrap();
    let mut ptss = Vec::new();
    while let Some(f) = r.next_frame() {
        ptss.push(f.pts_ns);
    }
    assert_eq!(ptss, vec![0, 0, 0], "all laced frames share the block pts");
}
