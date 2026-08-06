//! Malformed / truncated input never panics (spec: "a crash on bad input is a P0";
//! decoders parse untrusted input). Table-driven: each case is fed to a fresh
//! [`MatroskaReader`], and the reader must either buffer it (incomplete) or return an
//! `Err` — but MUST NOT panic, slice out of range, loop forever, or over-allocate.
//!
//! Two feed strategies are exercised per case: the whole blob at once, and byte-by-byte
//! (which also covers the cross-boundary buffering path). A valid stream head is prepended to
//! the block-level cases so the reader is actually *in* the streaming state when it hits the
//! corruption, rather than rejecting it at the top level.

use pf_mkv::ebml::{self, id};
use pf_mkv::MatroskaReader;

/// A valid stream head (EBML Header + Segment + Info + Tracks + open Cluster) so appended bad
/// block bytes are parsed in the streaming phase. Mirrors `lacing.rs`'s builder.
fn valid_head() -> Vec<u8> {
    let mut out = Vec::new();
    ebml::write_id(&mut out, id::EBML);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_uint(&mut out, id::EBML_VERSION, 1);
    ebml::write_string(&mut out, id::DOC_TYPE, "matroska");
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    ebml::write_id(&mut out, id::SEGMENT);
    ebml::write_unknown_size(&mut out);

    ebml::write_id(&mut out, id::INFO);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_uint(&mut out, id::TIMESTAMP_SCALE, 1_000_000);
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    ebml::write_id(&mut out, id::TRACKS);
    let at = ebml::reserve_size(&mut out);
    let body = out.len();
    ebml::write_id(&mut out, id::TRACK_ENTRY);
    let te_at = ebml::reserve_size(&mut out);
    let te_body = out.len();
    ebml::write_uint(&mut out, id::TRACK_NUMBER, 1);
    ebml::write_uint(&mut out, id::TRACK_TYPE, 2);
    ebml::write_string(&mut out, id::CODEC_ID, "A_FLAC");
    let te_len = (out.len() - te_body) as u64;
    ebml::patch_size(&mut out, te_at, te_len);
    let len = (out.len() - body) as u64;
    ebml::patch_size(&mut out, at, len);

    ebml::write_id(&mut out, id::CLUSTER);
    ebml::write_unknown_size(&mut out);
    ebml::write_uint(&mut out, id::TIMESTAMP, 0);
    out
}

/// Feed `data` to a fresh reader whole, then again byte-by-byte, draining frames each time.
/// The only contract asserted here is **no panic** — the harness catches a panic as a test
/// failure. Whether a given case errors or merely buffers is case-specific and not asserted.
fn feed_no_panic(data: &[u8]) {
    // Whole blob at once.
    let mut r = MatroskaReader::new();
    let _ = r.push(data);
    while r.next_frame().is_some() {}

    // Byte by byte (cross-boundary buffering).
    let mut r = MatroskaReader::new();
    for &b in data {
        // A push may return Err on a structural violation; that is fine — just don't panic.
        if r.push(&[b]).is_err() {
            break;
        }
        while r.next_frame().is_some() {}
    }
}

/// A block-level bad case: a corrupt `SimpleBlock` (or raw bytes) appended after a valid head.
fn with_head(block: &[u8]) -> Vec<u8> {
    let mut s = valid_head();
    s.extend_from_slice(block);
    s
}

/// A raw `SimpleBlock` element wrapping `body` (which may itself be malformed/truncated).
fn simple_block(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    ebml::write_id(&mut out, id::SIMPLE_BLOCK);
    ebml::write_size(&mut out, body.len() as u64);
    out.extend_from_slice(body);
    out
}

#[test]
fn malformed_inputs_never_panic() {
    // Each entry: (name, bytes). All must be handled without a panic.
    let cases: Vec<(&str, Vec<u8>)> = vec![
        // --- top-level garbage ---
        ("empty", vec![]),
        ("single zero byte (invalid VINT first octet)", vec![0x00]),
        ("all 0xFF", vec![0xFF; 32]),
        ("random-ish bytes", (0u8..64).map(|b| b.wrapping_mul(37)).collect()),
        ("a bare EBML id with no size", id::EBML.to_vec()),
        (
            "an element claiming a huge size but no data",
            {
                let mut v = id::TRACKS.to_vec();
                // 8-octet size claiming ~2^40 bytes that never arrive → must stay Incomplete,
                // not allocate 1 TiB.
                v.push(0x01);
                v.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]);
                v
            },
        ),
        // --- Info / Tracks corruption ---
        (
            "TrackEntry with a zero TrackNumber",
            {
                let mut out = Vec::new();
                ebml::write_id(&mut out, id::EBML);
                let at = ebml::reserve_size(&mut out);
                let b = out.len();
                ebml::write_uint(&mut out, id::EBML_VERSION, 1);
                let l = (out.len() - b) as u64;
                ebml::patch_size(&mut out, at, l);
                ebml::write_id(&mut out, id::SEGMENT);
                ebml::write_unknown_size(&mut out);
                ebml::write_id(&mut out, id::TRACKS);
                let at = ebml::reserve_size(&mut out);
                let b = out.len();
                ebml::write_id(&mut out, id::TRACK_ENTRY);
                let te = ebml::reserve_size(&mut out);
                let teb = out.len();
                ebml::write_uint(&mut out, id::TRACK_NUMBER, 0); // illegal (RFC 9559 §5.1.4.1.1)
                let tel = (out.len() - teb) as u64;
                ebml::patch_size(&mut out, te, tel);
                let l = (out.len() - b) as u64;
                ebml::patch_size(&mut out, at, l);
                out
            },
        ),
        (
            "a child claiming to run past its master",
            {
                let mut out = valid_head();
                // A SimpleBlock whose declared size is far larger than the data present — the
                // reader must wait (Incomplete), never read out of range.
                ebml::write_id(&mut out, id::SIMPLE_BLOCK);
                out.push(0x01); // 8-octet size...
                out.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00]); // huge
                out.extend_from_slice(&[0x81, 0, 0, 0x80]); // a few real bytes then nothing
                out
            },
        ),
        // --- block header / lacing corruption ---
        ("SimpleBlock body empty", with_head(&simple_block(&[]))),
        ("SimpleBlock body one byte (track VINT only)", with_head(&simple_block(&[0x81]))),
        (
            "SimpleBlock header truncated (no flags byte)",
            with_head(&simple_block(&[0x81, 0x00])),
        ),
        (
            "Xiph lace: frame count but no sizes",
            with_head(&simple_block(&[0x81, 0, 0, 0x02, 0x03])), // flags 0x02 (Xiph), count-1=3
        ),
        (
            "Xiph lace: size runs off the end",
            with_head(&simple_block(&[0x81, 0, 0, 0x02, 0x01, 0xFF, 0xFF, 0xFF])),
        ),
        (
            "EBML lace: first size VINT truncated",
            with_head(&simple_block(&[0x81, 0, 0, 0x06, 0x01])), // flags 0x06 (EBML), count-1=1
        ),
        (
            "EBML lace: negative frame size from a bad delta",
            {
                // count-1 = 1; first size small (0x81 = 1), delta hugely negative → prev < 0.
                with_head(&simple_block(&[0x81, 0, 0, 0x06, 0x01, 0x81, 0x80]))
            },
        ),
        (
            "fixed lace: block size not divisible by frame count",
            // flags 0x04 (fixed), count-1=2 (3 frames), then 4 body bytes → 4 % 3 != 0.
            with_head(&simple_block(&[0x81, 0, 0, 0x04, 0x02, 0xAA, 0xBB, 0xCC, 0xDD])),
        ),
        (
            "laced frame size overflow claim",
            // Xiph, count-1=1, first size = 255*many → huge, but no data → runs past block.
            with_head(&simple_block(&[0x81, 0, 0, 0x02, 0x01, 0xFF, 0xFF, 0x01])),
        ),
        (
            "BlockGroup with a truncated Block",
            {
                let mut bg = Vec::new();
                ebml::write_id(&mut bg, id::BLOCK_GROUP);
                let block_body = vec![0x81u8, 0, 0]; // track + rel-ts, no flags → truncated
                let mut block = Vec::new();
                ebml::write_id(&mut block, id::BLOCK);
                ebml::write_size(&mut block, block_body.len() as u64);
                block.extend_from_slice(&block_body);
                ebml::write_size(&mut bg, block.len() as u64);
                bg.extend_from_slice(&block);
                with_head(&bg)
            },
        ),
    ];

    for (name, data) in &cases {
        // A panic in `feed_no_panic` fails the test and names the case via the message below.
        feed_no_panic(data);
        // Also fuzz every prefix of the case — truncation at any offset must be safe.
        for n in 0..data.len() {
            feed_no_panic(&data[..n]);
        }
        eprintln!("ok: {name}");
    }
}

/// A specific structural violation returns `Err` (not silently accepted): a TrackEntry with a
/// zero TrackNumber cannot route blocks, so discovery on it fails loudly.
#[test]
fn zero_track_number_is_a_hard_error() {
    let mut out = Vec::new();
    ebml::write_id(&mut out, id::SEGMENT);
    ebml::write_unknown_size(&mut out);
    ebml::write_id(&mut out, id::TRACKS);
    let at = ebml::reserve_size(&mut out);
    let b = out.len();
    ebml::write_id(&mut out, id::TRACK_ENTRY);
    let te = ebml::reserve_size(&mut out);
    let teb = out.len();
    ebml::write_uint(&mut out, id::TRACK_NUMBER, 0);
    ebml::write_string(&mut out, id::CODEC_ID, "A_FLAC");
    let tel = (out.len() - teb) as u64;
    ebml::patch_size(&mut out, te, tel);
    let l = (out.len() - b) as u64;
    ebml::patch_size(&mut out, at, l);

    let mut r = MatroskaReader::new();
    assert!(r.push(&out).is_err(), "a zero TrackNumber must be a hard error");
}
