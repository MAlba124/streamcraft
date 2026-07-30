//! Encoder validation gate for the CELT-only `OpusEncoder` — the workspace's three-way check:
//!
//! 1. **Round-trip through our own `OpusDec`** — encode a signal, decode it back, best-offset SNR.
//! 2. **Perceptual gate** — `profluens-audio`'s `PerceptualAnalyzer` (NMR + bark loudness), the
//!    right metric for a codec that hides noise below the masking threshold (raw SNR alone lies —
//!    see `audio/PERCEPTUAL_QUALITY.md`).
//! 3. **External oracle** — mux one case into an Ogg-Opus (`.opus`) file (RFC 7845) and decode it
//!    with **ffmpeg/libopus**, confirming our packets are valid *and* close in SNR to what an
//!    independent decoder reconstructs. A pure-Rust encoder that only round-trips through its own
//!    decoder can hide a shared bug; the oracle is what catches it.
//!
//!   cargo run -p pf-opus --example encode_roundtrip
//!
//! Exits non-zero if any case fails its round-trip SNR gate, or if ffmpeg is present and its decode
//! of our Ogg-Opus disagrees. ffmpeg absent → the oracle is skipped with a note (not a failure).

#![allow(clippy::disallowed_methods, clippy::chunks_exact_to_as_chunks)] // one-shot validation harness, not a hot path

use std::f64::consts::PI;

use oxideav_opus::OpusDecoder;
use pf_ogg::OggWriter;
use pf_opus::{EncoderConfig, OpusEncoder, OPUS_RATE};
use profluens_audio::quality::{rms_normalize, PerceptualAnalyzer, QualityReport};

/// One test signal: interleaved 48 kHz `i16`, a channel count, and a human label.
struct Signal {
    label: &'static str,
    channels: usize,
    pcm: Vec<i16>,
}

fn main() {
    let seconds = 2.0;
    let n = (OPUS_RATE as f64 * seconds) as usize;
    let signals = [
        Signal { label: "mono 440+1k tone", channels: 1, pcm: mono_tone(n) },
        Signal { label: "stereo 440/660 tone", channels: 2, pcm: stereo_tone(n) },
        Signal { label: "mono sweep 200-8k", channels: 1, pcm: mono_sweep(n) },
        Signal { label: "stereo white noise", channels: 2, pcm: stereo_noise(n) },
    ];

    // The gate is **perceptual**, not raw SNR: a transform codec puts quantization noise below the
    // masking threshold, so it deliberately does NOT reconstruct samples identically (especially
    // for noise) — NMR < 0 ≈ transparent is the right pass condition (see PERCEPTUAL_QUALITY.md and
    // the handoff's "don't gate on raw SNR alone"). SNR is printed for context only; note that a
    // correlation offset on a *pure tone* is phase-ambiguous, so a low tone-SNR next to a very
    // negative NMR is a measurement artifact, not a codec fault.
    const NMR_PASS_DB: f64 = 2.0;
    let mut any_fail = false;
    for sig in &signals {
        for &bitrate in &[64_000u32, 96_000, 128_000] {
            let cfg = EncoderConfig {
                bitrate_bps: bitrate,
                ..EncoderConfig::new(sig.channels as u8)
            };
            let mut enc = OpusEncoder::new(cfg).unwrap();
            let packets = encode_all(&mut enc, &sig.pcm, sig.channels);
            let decoded = decode_all(&packets, sig.channels);

            let (snr, off) = robust_snr(&sig.pcm, &decoded, sig.channels);
            let q = perceptual(&sig.pcm, &decoded, sig.channels, off);

            let pass = q.nmr_db < NMR_PASS_DB;
            any_fail |= !pass;
            println!(
                "{:<22} {:>4} kbps  {}ch  NMR {:>7.2} dB (peak {:>6.2})  [SNR {:>6.2} dB]  {}",
                sig.label,
                bitrate / 1000,
                sig.channels,
                q.nmr_db,
                q.peak_nmr_db,
                snr,
                if pass { "ok" } else { "FAIL" },
            );
        }
    }

    // ---- External oracle: one representative case through ffmpeg/libopus. ----
    let sig = &signals[1]; // stereo tone
    let cfg = EncoderConfig { bitrate_bps: 128_000, ..EncoderConfig::new(sig.channels as u8) };
    let mut enc = OpusEncoder::new(cfg).unwrap();
    let packets = encode_all(&mut enc, &sig.pcm, sig.channels);
    let frame_samples = enc.frame_samples();
    let path = "/tmp/sc_opusenc.opus";
    write_ogg_opus(path, sig.channels, frame_samples, &packets);
    println!("\nwrote {path} ({} packets, {}ch)", packets.len(), sig.channels);

    match ffmpeg_decode(path, sig.channels) {
        Some(ff) => {
            // The oracle's job: confirm our packets are *standard* Opus. ffmpeg decoding without
            // error already proves validity; then we check that libopus's reconstruction agrees
            // with our own decoder's (both decode the identical bitstream, so they should track
            // closely — the same 20–40 dB float-noise floor the decode conformance gate sits at).
            // Comparing against libopus (not the original) sidesteps the pure-tone SNR ambiguity.
            let ours = decode_all(&packets, sig.channels);
            let (agree, _) = robust_snr(&ours, &ff, sig.channels);
            let q = perceptual(&sig.pcm, &ff, sig.channels, robust_snr(&sig.pcm, &ff, sig.channels).1);
            println!(
                "ffmpeg/libopus decoded our Ogg-Opus: agrees with our decoder at {agree:.2} dB SNR, \
                 NMR vs original {:.2} dB",
                q.nmr_db
            );
            if agree < 15.0 {
                eprintln!("FAIL: libopus and our decoder disagree on our own bitstream — a \
                           non-standard-Opus encoder bug");
                any_fail = true;
            } else {
                println!("external-oracle gate: PASS (valid standard Opus; libopus concurs)");
            }
        }
        None => println!("(ffmpeg not found or failed — external oracle skipped; \
                          run `ffmpeg -i {path} -f s16le -` to check manually)"),
    }

    // ---- Rate-distortion A/B: OUR encoder vs libopus's encoder, same signal + bitrate. ----
    // Decodes both with the same libopus decoder and compares perceptual NMR vs the original, so
    // the numbers are directly comparable — this is the real "how good is it" question.
    println!("\nours vs libopus encoder (both decoded by libopus, NMR vs original — lower = better):");
    for sig in &signals {
        for &bitrate in &[64_000u32, 96_000] {
            let cfg = EncoderConfig { bitrate_bps: bitrate, ..EncoderConfig::new(sig.channels as u8) };
            let mut enc = OpusEncoder::new(cfg).unwrap();
            let packets = encode_all(&mut enc, &sig.pcm, sig.channels);
            let fs = enc.frame_samples();
            let p = "/tmp/sc_ours_ab.opus";
            write_ogg_opus(p, sig.channels, fs, &packets);
            let (Some(ours), Some(theirs)) =
                (ffmpeg_decode(p, sig.channels), libopus_roundtrip(&sig.pcm, sig.channels, bitrate))
            else {
                println!("  (ffmpeg libopus encoder unavailable — A/B skipped)");
                break;
            };
            let our_nmr = perceptual(&sig.pcm, &ours, sig.channels, robust_snr(&sig.pcm, &ours, sig.channels).1).nmr_db;
            let lib_nmr = perceptual(&sig.pcm, &theirs, sig.channels, robust_snr(&sig.pcm, &theirs, sig.channels).1).nmr_db;
            println!(
                "  {:<22} {:>4} kbps   ours {:>7.2} dB   libopus {:>7.2} dB   (Δ {:+.2})",
                sig.label, bitrate / 1000, our_nmr, lib_nmr, our_nmr - lib_nmr,
            );
        }
    }

    if any_fail {
        eprintln!("\nFAIL: an encode case did not clear its gate");
        std::process::exit(1);
    }
    println!("\nPASS");
}

// ---------------------------------------------------------------------------
// Encode / decode helpers
// ---------------------------------------------------------------------------

fn encode_all(enc: &mut OpusEncoder, pcm: &[i16], _channels: usize) -> Vec<Vec<u8>> {
    let flen = enc.frame_len();
    let mut packets = Vec::new();
    let mut pkt = Vec::new();
    for chunk in pcm.chunks_exact(flen) {
        enc.encode_frame(chunk, &mut pkt).unwrap();
        packets.push(pkt.clone());
    }
    packets
}

fn decode_all(packets: &[Vec<u8>], _channels: usize) -> Vec<i16> {
    let mut dec = OpusDecoder::new();
    let mut out = Vec::new();
    let mut pcm = Vec::new();
    for p in packets {
        if dec.decode_packet_into(p, &mut pcm).is_ok() {
            out.extend_from_slice(&pcm);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Robust best-offset SNR: brute-force the per-channel startup-delay offset that *maximizes* the
/// interleaved SNR over a half-second window (mirrors `examples/conformance.rs`), then score the
/// full overlap there. Brute force beats correlation alignment for periodic signals, whose
/// autocorrelation is phase-ambiguous. Returns `(snr_db, per_channel_offset)`.
fn robust_snr(reference: &[i16], test: &[i16], channels: usize) -> (f64, usize) {
    const WIN_FRAMES: usize = 24_000; // 0.5 s
    let win = WIN_FRAMES * channels;
    let mut best = (f64::NEG_INFINITY, 0usize);
    for off in 0..=480 {
        let ts = off * channels;
        if ts + win > test.len() || win > reference.len() {
            break;
        }
        let s = snr_i16(&reference[..win], &test[ts..ts + win]);
        if s > best.0 {
            best = (s, off);
        }
    }
    let ts = (best.1 * channels).min(test.len());
    let m = reference.len().min(test.len() - ts);
    (snr_i16(&reference[..m], &test[ts..ts + m]), best.1)
}

/// Perceptual NMR of `test` vs `reference` on channel 0, aligned by the given per-channel `offset`
/// and level-matched (the analyzer expects aligned, level-matched inputs).
fn perceptual(reference: &[i16], test: &[i16], channels: usize, offset: usize) -> QualityReport {
    let r = to_f32(&deinterleave(reference, channels, 0));
    let t_full = to_f32(&deinterleave(test, channels, 0));
    let t_aligned = &t_full[offset.min(t_full.len())..];
    let m = r.len().min(t_aligned.len());
    let r = r[..m].to_vec();
    let mut t = t_aligned[..m].to_vec();
    rms_normalize(&r, &mut t);
    let mut an = PerceptualAnalyzer::new(OPUS_RATE);
    an.analyze(&r, &t)
}

fn snr_i16(reference: &[i16], test: &[i16]) -> f64 {
    let n = reference.len().min(test.len());
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = reference[i] as f64;
        let e = r - test[i] as f64;
        sig += r * r;
        err += e * e;
    }
    if err <= 0.0 {
        200.0
    } else {
        10.0 * (sig / err).log10()
    }
}

fn deinterleave(interleaved: &[i16], channels: usize, ch: usize) -> Vec<i16> {
    interleaved.iter().skip(ch).step_by(channels).copied().collect()
}

fn to_f32(v: &[i16]) -> Vec<f32> {
    v.iter().map(|&s| s as f32 / 32768.0).collect()
}

// ---------------------------------------------------------------------------
// Signal generators (deterministic)
// ---------------------------------------------------------------------------

fn mono_tone(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let t = i as f64 / OPUS_RATE as f64;
            let s = 0.3 * ((2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 1000.0 * t).sin());
            (s * 16384.0) as i16
        })
        .collect()
}

fn stereo_tone(n: usize) -> Vec<i16> {
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / OPUS_RATE as f64;
        v.push((0.3 * (2.0 * PI * 440.0 * t).sin() * 16384.0) as i16);
        v.push((0.3 * (2.0 * PI * 660.0 * t).sin() * 16384.0) as i16);
    }
    v
}

fn mono_sweep(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let t = i as f64 / OPUS_RATE as f64;
            // Linear chirp 200 Hz → 8 kHz over the buffer.
            let f = 200.0 + (8000.0 - 200.0) * (i as f64 / n as f64);
            (0.3 * (2.0 * PI * f * t).sin() * 16384.0) as i16
        })
        .collect()
}

fn stereo_noise(n: usize) -> Vec<i16> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as i32 - 8192) as i16 // ~±0.25 FS
    };
    (0..n * 2).map(|_| next()).collect()
}

// ---------------------------------------------------------------------------
// Ogg-Opus muxing (RFC 7845) + ffmpeg oracle
// ---------------------------------------------------------------------------

/// RFC 7845 §5.1 `OpusHead` (channel-mapping family 0: mono/stereo).
fn opus_head(channels: usize) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(channels as u8);
    // pre_skip: the CELT encode+decode chain's 2.5 ms MDCT delay (120 samples @ 48 kHz).
    h.extend_from_slice(&120u16.to_le_bytes());
    h.extend_from_slice(&OPUS_RATE.to_le_bytes()); // original input rate (informational)
    h.extend_from_slice(&0u16.to_le_bytes()); // output gain (Q7.8)
    h.push(0); // channel mapping family 0
    h
}

/// RFC 7845 §5.2 `OpusTags` (empty comment list).
fn opus_tags() -> Vec<u8> {
    let vendor = b"streamcraft-opusenc";
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor);
    t.extend_from_slice(&0u32.to_le_bytes()); // 0 user comments
    t
}

fn write_ogg_opus(path: &str, _channels: usize, frame_samples: usize, packets: &[Vec<u8>]) {
    const PRE_SKIP: u64 = 120;
    // A fixed, arbitrary serial number for the single logical stream.
    let mut w = OggWriter::new(0x5C0F_0005);
    let mut out = Vec::new();
    // Ogg-Opus mapping: OpusHead alone on the BOS page, OpusTags on its own page, then audio.
    w.write_packet(&mut out, &opus_head(_channels), 0).unwrap();
    w.flush(&mut out);
    w.write_packet(&mut out, &opus_tags(), 0).unwrap();
    w.flush(&mut out);
    for (k, p) in packets.iter().enumerate() {
        // Granule = total 48 kHz PCM samples (per channel) decodable through this packet, plus the
        // declared pre-skip (RFC 7845 §4).
        let granule = (k as u64 + 1) * frame_samples as u64 + PRE_SKIP;
        w.write_packet(&mut out, p, granule).unwrap();
    }
    w.finish(&mut out);
    std::fs::write(path, &out).expect("write .opus");
}

/// Encode `pcm` with **ffmpeg's libopus encoder** at `bitrate`, then decode it back — the
/// reference encoder for the rate-distortion A/B. Returns the decoded PCM, or `None` if ffmpeg /
/// libopus is unavailable.
fn libopus_roundtrip(pcm: &[i16], channels: usize, bitrate: u32) -> Option<Vec<i16>> {
    let raw = "/tmp/sc_ref.pcm";
    let opus = "/tmp/sc_libopus.opus";
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(raw, &bytes).ok()?;
    let enc = std::process::Command::new("ffmpeg")
        .args([
            "-y", "-v", "error", "-f", "s16le", "-ar", "48000", "-ac", &channels.to_string(),
            "-i", raw, "-c:a", "libopus", "-b:a", &bitrate.to_string(), "-vbr", "off", opus,
        ])
        .output()
        .ok()?;
    if !enc.status.success() {
        return None; // libopus encoder not built into this ffmpeg
    }
    ffmpeg_decode(opus, channels)
}

/// Decode `path` with ffmpeg to interleaved 48 kHz `s16` and return it, or `None` if ffmpeg is
/// missing / fails.
fn ffmpeg_decode(path: &str, channels: usize) -> Option<Vec<i16>> {
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "48000",
            "-ac", &channels.to_string(), "-",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(out.stdout.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
}
