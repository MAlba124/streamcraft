//! Cross-validation against the reference implementation (libogg / libVorbis via the
//! `oggenc` CLI), a dev-only out-of-process oracle, never linked in. If `oggenc` is on
//! `PATH`, encode a tiny raw-PCM tone into a **real** Ogg Vorbis file and confirm our
//! reader (a) parses every page, (b) recomputes each page's CRC to the exact value
//! libogg stamped, and (c) reassembles a sensible number of packets. When `oggenc` is
//! absent the test no-ops with a printed note — the in-crate round-trip and the baked-in
//! real-page known-answer test ([`sc_ogg::page`]) remain the gate; this is the extra
//! net (spec: First-party codecs — cross-checked against a reference).

use std::io::Write;
use std::process::Command;

use sc_ogg::{page_crc, OggReader, PageHeader};

fn have_oggenc() -> bool {
    Command::new("oggenc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_ogg_xv_{}_{}.{}", tag, std::process::id(), ext));
    p
}

/// Write ~0.3 s of a stereo 16-bit sine as raw little-endian PCM.
fn write_raw_pcm(path: &std::path::Path) {
    let sr = 44100usize;
    let n = sr * 3 / 10;
    let mut bytes = Vec::with_capacity(n * 2 * 2);
    for i in 0..n {
        for c in 0..2 {
            let v = (20000.0 * (2.0 * std::f64::consts::PI * (440.0 + 30.0 * c as f64) * i as f64 / sr as f64).sin()) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(path).unwrap().write_all(&bytes).unwrap();
}

#[test]
fn our_reader_verifies_libogg_crc_and_reassembles_packets() {
    if !have_oggenc() {
        eprintln!("cross_validate: `oggenc` not on PATH — skipping (baked-in real-page KAT is the gate)");
        return;
    }

    let raw = temp_path("tone", "raw");
    let ogg = temp_path("tone", "ogg");
    write_raw_pcm(&raw);

    // Encode raw PCM → real Ogg Vorbis. Flags: quiet, raw input, 44100 Hz, 16-bit,
    // 2 channels, little-endian.
    let status = Command::new("oggenc")
        .args(["-Q", "-r", "-R", "44100", "-B", "16", "-C", "2", "--raw-endianness", "0", "-o"])
        .arg(&ogg)
        .arg(&raw)
        .status()
        .expect("run oggenc");
    assert!(status.success(), "oggenc failed");

    let data = std::fs::read(&ogg).expect("read produced ogg");
    assert!(!data.is_empty());

    // Walk every page: it must parse, and its stored CRC must equal our recomputation —
    // proving our CRC-32 matches libogg's on real, varied page contents.
    let mut off = 0;
    let mut pages = 0;
    let mut saw_bos = false;
    let mut saw_eos = false;
    while off < data.len() {
        let page = PageHeader::parse(&data[off..]).expect("libogg page parses");
        assert_eq!(page.crc(), page_crc(page.raw()), "page {pages} CRC vs libogg");
        saw_bos |= page.is_bos();
        saw_eos |= page.is_eos();
        off += page.len();
        pages += 1;
    }
    assert_eq!(off, data.len(), "stream fully consumed, no trailing garbage");
    assert!(pages >= 2, "expected at least a header page and an audio page");
    assert!(saw_bos, "a bos page must be present");
    assert!(saw_eos, "an eos page must be present");

    // Reassemble packets. Vorbis emits three header packets (id/comment/setup) plus
    // audio packets, so we expect at least four, with no resyncs on clean input.
    let mut reader = OggReader::new();
    reader.push(&data);
    let leftover = reader.finish();
    assert_eq!(leftover, 0, "clean stream ends on a page boundary");
    let mut packets = 0;
    let mut first = None;
    while let Some(p) = reader.next_packet() {
        if first.is_none() {
            first = Some(p.data.clone());
        }
        packets += 1;
    }
    assert_eq!(reader.resync_count(), 0, "no resyncs on clean libogg output");
    assert!(packets >= 4, "expected ≥4 packets (3 Vorbis headers + audio), got {packets}");
    // The first packet is the Vorbis identification header: 0x01 "vorbis" (§4).
    let first = first.unwrap();
    assert!(first.starts_with(&[0x01, b'v', b'o', b'r', b'b', b'i', b's']), "first packet is the Vorbis id header");

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ogg);
}
