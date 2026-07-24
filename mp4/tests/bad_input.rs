//! Malformed / truncated input never panics (spec: "a crash on bad input is a P0";
//! demuxers parse untrusted input). Mirrors `mkv/tests/bad_input.rs`. Two layers are
//! exercised:
//! 1. **Resolution** ([`Mp4Reader::new`]) over hand-built and truncated `moov` blobs — must
//!    return an `Err`, never panic / over-allocate / loop forever.
//! 2. **Streaming** ([`Mp4Reader::push`] / `next_sample`) over a valid head but a truncated /
//!    garbage `mdat` body — must yield whatever samples are fully buffered and stop, never
//!    read out of range.
//!
//! Every case is also fed as **each of its prefixes** (truncation at every offset) so a cut
//! at any byte is safe.

use sc_mp4::Mp4Reader;

/// Assemble a box: 8-byte header (size, type) + body. `size` is the total incl. header.
fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A minimal-but-plausible `ftyp` box.
fn ftyp() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"isom"); // major_brand
    body.extend_from_slice(&0u32.to_be_bytes()); // minor_version
    body.extend_from_slice(b"isom"); // compatible_brands[0]
    boxed(b"ftyp", &body)
}

/// Resolve `head` and assert it does not panic — whether it errors or (rarely) resolves is
/// case-specific and not asserted here; the only contract is **no panic / no hang / no OOM**.
fn resolve_no_panic(head: &[u8]) {
    let _ = Mp4Reader::new(head);
}

/// Feed a valid head, then a (possibly garbage/truncated) `mdat` region, streaming it in
/// 1-byte chunks and draining samples — the streaming slicer must never read out of range.
fn stream_no_panic(head: &[u8], tail: &[u8]) {
    let Ok(mut reader) = Mp4Reader::new(head) else {
        return; // head did not resolve — nothing to stream
    };
    reader.reset_stream();
    // The reader is fed the WHOLE file from byte 0: head then tail.
    let mut file = head.to_vec();
    file.extend_from_slice(tail);
    for b in &file {
        reader.push(std::slice::from_ref(b));
        while reader.next_sample().is_some() {}
    }
}

#[test]
fn malformed_heads_never_panic() {
    // Each entry: (name, head bytes). All must be handled without a panic.
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("single byte", vec![0x00]),
        ("all zeros", vec![0u8; 64]),
        ("all 0xFF", vec![0xFF; 64]),
        ("random-ish bytes", (0u8..128).map(|b| b.wrapping_mul(37)).collect()),
        ("ftyp only (no moov)", ftyp()),
        (
            "a box claiming a huge size but no data",
            {
                // size = 0x7FFF_FFFF, type moov, but nothing follows → must not allocate/loop.
                let mut v = 0x7FFF_FFFFu32.to_be_bytes().to_vec();
                v.extend_from_slice(b"moov");
                v
            },
        ),
        (
            "largesize box claiming ~2^40 bytes, none present",
            {
                let mut v = 1u32.to_be_bytes().to_vec(); // size == 1 → largesize follows
                v.extend_from_slice(b"moov");
                v.extend_from_slice(&(1u64 << 40).to_be_bytes());
                v
            },
        ),
        (
            "box size smaller than its own header",
            {
                let mut v = 4u32.to_be_bytes().to_vec(); // size 4 < 8-byte header
                v.extend_from_slice(b"moov");
                v
            },
        ),
        ("empty moov (no trak)", boxed(b"moov", &[])),
        (
            "moov with a truncated trak",
            {
                let mut moov = Vec::new();
                // a trak whose declared size runs past the moov body
                let mut trak = 200u32.to_be_bytes().to_vec();
                trak.extend_from_slice(b"trak");
                trak.extend_from_slice(&[0u8; 4]); // only 4 body bytes, not 192
                moov.extend_from_slice(&trak);
                boxed(b"moov", &moov)
            },
        ),
        (
            "moov/trak/mdia/minf/stbl with a truncated stsz claiming a huge table",
            {
                // stsz: sample_size 0 (per-sample table) + sample_count u32::MAX, no data.
                let mut stsz = vec![0u8; 4]; // version+flags
                stsz.extend_from_slice(&0u32.to_be_bytes());
                stsz.extend_from_slice(&u32::MAX.to_be_bytes());
                let stbl = boxed(b"stbl", &boxed(b"stsz", &stsz));
                let minf = boxed(b"minf", &stbl);
                let mut mdia = boxed(b"mdhd", &{
                    let mut m = vec![0u8; 4];
                    m.extend_from_slice(&0u64.to_be_bytes()); // times
                    m.extend_from_slice(&1000u32.to_be_bytes()); // timescale
                    m.extend_from_slice(&0u32.to_be_bytes()); // duration
                    m.extend_from_slice(&[0u8; 4]); // language + pre_defined
                    m
                });
                mdia.extend_from_slice(&minf);
                let mdia = boxed(b"mdia", &mdia);
                let trak = boxed(b"trak", &mdia);
                boxed(b"moov", &trak)
            },
        ),
        (
            "fragmented movie (moov with mvex)",
            {
                let mvex = boxed(b"mvex", &[]);
                boxed(b"moov", &mvex)
            },
        ),
    ];

    for (name, head) in &cases {
        resolve_no_panic(head);
        // Truncation at every offset must be safe too.
        for n in 0..head.len() {
            resolve_no_panic(&head[..n]);
        }
        eprintln!("ok (resolve): {name}");
    }
}

/// The fragmented case is rejected loudly (not silently treated as an empty movie).
#[test]
fn fragmented_movie_is_a_hard_error() {
    let mvex = boxed(b"mvex", &[]);
    let moov = boxed(b"moov", &mvex);
    let mut head = ftyp();
    head.extend_from_slice(&moov);
    match Mp4Reader::new(&head) {
        Err(sc_mp4::Mp4Error::Fragmented) => {}
        Err(other) => panic!("moov/mvex should be Fragmented, got {other:?}"),
        Ok(_) => panic!("moov/mvex must not resolve as a progressive movie"),
    }
}

/// A valid head but a garbage / truncated `mdat` body: streaming must not panic or read out
/// of range, regardless of how the sample offsets point into the (missing) data.
#[test]
fn streaming_over_bad_mdat_never_panics() {
    // Build a real valid head by taking the committed fixture's head, then feed it various
    // broken tails. The head resolves (the sample table is valid); the tail is where the
    // samples' bytes should live but do not / are truncated.
    let file = fixture_bytes("tiny_h264.mp4");
    let head = head_through_moov(&file);

    // Sanity: the head resolves on its own.
    assert!(Mp4Reader::new(&head).is_ok(), "the fixture head resolves");

    let tails: Vec<(&str, Vec<u8>)> = vec![
        ("no mdat at all", Vec::new()),
        ("a few garbage bytes", vec![0xAB; 16]),
        ("half the real mdat", file[head.len()..head.len() + (file.len() - head.len()) / 2].to_vec()),
        ("the real mdat (full)", file[head.len()..].to_vec()),
        ("all 0xFF where samples should be", vec![0xFF; 4096]),
    ];

    for (name, tail) in &tails {
        stream_no_panic(&head, tail);
        // Truncate the tail at every offset too.
        for n in 0..tail.len().min(300) {
            stream_no_panic(&head, &tail[..n]);
        }
        eprintln!("ok (stream): {name}");
    }
}

// --- fixture helpers (shared shape with the other test files) ---

fn fixture_bytes(name: &str) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

fn head_through_moov(file: &[u8]) -> Vec<u8> {
    let mut at = 0usize;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]]) as usize;
        if &file[at + 4..at + 8] == b"mdat" {
            return file[..at].to_vec();
        }
        let advance = if size == 0 { file.len() - at } else if size == 1 {
            u64::from_be_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if advance == 0 {
            break;
        }
        at += advance;
    }
    file.to_vec()
}
