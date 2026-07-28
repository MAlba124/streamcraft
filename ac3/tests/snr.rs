//! SNR conformance gate (ATSC A/52): decode a real AC-3 elementary stream through
//! the `Frame` core and compare per-channel PCM against an ffmpeg s16le reference.
//!
//! A/52 is lossy but *deterministic*: a correct decoder matches ffmpeg's decode to
//! a high SNR (tens-to-hundreds of dB), so this gate is the objective bit-accuracy
//! check for the §7.2 bit-allocation + §7.3 mantissa reconstruction.
//!
//! The reference fixtures are produced from a licensed movie the developer owns and
//! are **not** committed; the test is `#[ignore]`d by default and only runs when the
//! two files exist under `/tmp` (produced by the commands in the module docs). Run
//! with `cargo test -p sc-ac3 -- --ignored` after generating them:
//!
//! ```text
//! ffmpeg -i "$HOME/Videos/Nord.2009.720p.BRRip.XviD.AC3-ViSiON.avi" -map 0:a:0 \
//!        -c copy /tmp/nord.ac3
//! # extract a byte-aligned segment with real content (skip the quiet intro):
//! ffmpeg -i /tmp/nord.ac3 -f s16le -ac 6 /tmp/nord_ref.s16
//! ```

// Integration test harness (not element code): plain std file IO is fine here.
#![allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)]

use sc_ac3::frame::Frame;
use sc_ac3::parse::{next_frame, Framed};

/// Decode `data` (a raw AC-3/E-AC-3 elementary stream) through the framer + core,
/// returning interleaved S16 PCM (ITU order L,R,C,LFE,Ls,Rs).
fn decode_all(data: &[u8], max_frames: usize) -> (Vec<i16>, usize) {
    let mut frame = Frame::new();
    let mut out: Vec<i16> = Vec::new();
    let mut nch = 0usize;
    let mut pos = 0usize;
    let mut n = 0usize;
    while pos < data.len() && n < max_frames {
        match next_frame(&data[pos..]) {
            Framed::Frame { offset, len, .. } => {
                let start = pos + offset;
                if let Ok(dec) = frame.decode(&data[start..start + len]) {
                    nch = dec.info.channels;
                    out.extend_from_slice(dec.pcm);
                    n += 1;
                }
                pos = start + len;
            }
            Framed::NeedMore { offset } => {
                pos += offset + 1;
            }
            Framed::NoSync => break,
        }
    }
    (out, nch)
}

/// Per-channel SNR (dB) of `out` vs `reference`, both interleaved with `nch`
/// channels. Silent reference channels (no energy) are skipped.
fn per_channel_snr(out: &[i16], reference: &[i16], nch: usize) -> Vec<Option<f64>> {
    let frames = (out.len() / nch).min(reference.len() / nch);
    (0..nch)
        .map(|ch| {
            let mut sig = 0f64;
            let mut noise = 0f64;
            for i in 0..frames {
                let a = f64::from(out[i * nch + ch]);
                let b = f64::from(reference[i * nch + ch]);
                sig += b * b;
                noise += (a - b) * (a - b);
            }
            if sig < 1.0 {
                None // silent reference channel — no meaningful SNR
            } else if noise == 0.0 {
                Some(f64::INFINITY)
            } else {
                Some(10.0 * (sig / noise).log10())
            }
        })
        .collect()
}

#[test]
#[ignore = "requires /tmp/nord.ac3 + /tmp/nord_ref.s16 fixtures (see module docs)"]
fn nord_ac3_snr_gate() {
    let Ok(data) = std::fs::read("/tmp/nord.ac3") else {
        eprintln!("skip: /tmp/nord.ac3 not present");
        return;
    };
    let Ok(refpcm) = std::fs::read("/tmp/nord_ref.s16") else {
        eprintln!("skip: /tmp/nord_ref.s16 not present");
        return;
    };
    let refi16: Vec<i16> = refpcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let (out, nch) = decode_all(&data, 64);
    assert!(nch > 0, "decoded no channels");
    let snrs = per_channel_snr(&out, &refi16, nch);
    for (ch, snr) in snrs.iter().enumerate() {
        eprintln!("ch {ch}: SNR = {snr:?} dB");
    }
    // Target: AC-3 is deterministic → a correct decoder matches ffmpeg to ≥ 60 dB
    // on every non-silent channel.
    for (ch, snr) in snrs.iter().enumerate() {
        if let Some(v) = snr {
            assert!(*v >= 60.0, "channel {ch} SNR {v:.2} dB below 60 dB gate");
        }
    }
}
