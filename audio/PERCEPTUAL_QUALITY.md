# Perceptual audio-quality rating for transcode QA — design notes

**Feature:** after transcoding audio (decode X → encode Y, e.g. → Opus), automatically rate the
perceptual quality of the output vs the original reference, flag bad output, and use the score to
drive encoder tuning. Full-reference (intrusive) objective quality.

Constraint (as always here): pure-Rust / minimal-deps / clean-room preferred over C bindings.

## The metrics landscape (verified July 2026)

| Metric | What / scale | Impl reality for us |
| ------ | ------------ | ------------------- |
| **PEAQ** (ITU-R BS.1387) | The *standard* for grading music codecs. **ODG** −4…0 (0 = transparent, −1 = audible-but-fine, −4 = badly broken); Basic = FFT ear model + 11 MOVs → fixed-weight MLP; Advanced adds a filterbank model (5 MOVs). Full-reference, needs time-aligned ref+test @ 48 kHz. | **No freely redistributable conformant reference** (patent/FRAND). GstPEAQ (C, GPL) is the best open impl but not fully conformant + drags all of GStreamer. PQevalAudio (Kabal, MATLAB) is educational-only. **No Rust crate exists.** Clean-room Rust is a multi-week effort and — like our AC-3/MPEG-4p2 wall — the normative weights resist exact reproduction. |
| **ViSQOL v3** (Google) | Spectro-temporal similarity → **MOS-LQO** 1…5. Audio mode = 48 kHz (music); speech mode = 16 kHz. | Reference is C++/Bazel/Armadillo/libsvm (heavy, un-Rusty). **But `visqol-rs` exists** — a **pure-Rust** reimplementation of ViSQOL v3.1, **Apache-2.0** (GPL-3-compatible), v0.3.1 (2025), gammatone filterbank in Rust, no C++/Bazel. Community reimpl (partly a learning exercise) → **validate its scores against Google's reference before trusting as a gate.** |
| **PESQ** (P.862) / **POLQA** (P.863) | Speech/telephony only. | Wrong content model + bandwidth for music. PESQ withdrawn 2024; POLQA proprietary. **Skip.** |
| **PEMO-Q / 2f-model** | Research psychoacoustic FR metrics. | No redistributable impl, no Rust. Reading material only. |
| **CDPAM/DPAM, Audiobox-Aesthetics** | ML perceptual distance / no-reference aesthetics. | Need PyTorch + big weights (contradicts pure-Rust). Audiobox is *no-reference* — wrong tool for ref-vs-test. Offline-research only. |

**Cheap objective measures** (pure Rust, a few hundred lines over our existing FFT), by how well
they track perception:
- **SNR / segmental SNR — do NOT gate on this.** It's a waveform error; perceptual codecs
  deliberately put quantization noise *below the masking threshold*, so a transparent transcode
  can have terrible SNR. Also dominated by tiny time/level misalignment. Useful only as an
  *alignment sanity check* (a sudden SNR cliff usually means alignment broke, not the audio).
- **Log-spectral distance / MCD / spectral convergence** — phase-blind, cheap, *moderate*
  correlation. Good for **relative** ranking (regression detection, param sweeps), weak at
  absolute "is it transparent?".
- **Bark-band / gammatone loudness difference** — group magnitude into ~24 Bark bands →
  specific-loudness → difference the loudness patterns. Essentially the *front half* of
  PEAQ/ViSQOL without the trained back-end; correlates with perception far better than SNR/LSD
  because errors are auditory-sensitivity weighted.

## Recommended path for this project

1. **Alignment + normalization preprocessor FIRST — nothing works without it.** Every
   full-reference metric assumes sample-accurate time alignment and matched level.
   - **Opus delay compensation (critical):** CELT-only ≈ 2.5 ms look-ahead, SILK ≈ 6.5 ms
     (5 + 1.5 ms resampling). The Ogg **pre-skip** (RFC 7845) says how many leading samples to
     drop. Since *our* codecs are hand-written, expose the encoder's look-ahead directly instead
     of parsing it back out. Trim pre-skip from the decoded test; equalize trailing padding.
   - **Fine align:** cross-correlate ±a few ms to remove residual sub-frame offset.
   - **Level align:** normalize test to the reference RMS/loudness (single global gain).
   - Resample both to **48 kHz mono** (common denominator for PEAQ + ViSQOL audio mode).
2. **Ship a pure-Rust Bark/gammatone loudness-difference gate** as the always-on inline metric
   (tunable threshold), plus LSD + spectral-convergence as secondary regression signals for
   encoder-parameter sweeps. Zero new deps, in-ethos, immediately useful.
3. **Add `visqol-rs`** (pure-Rust, Apache-2.0) behind a feature flag for a real **MOS-LQO** score
   in audio mode @ 48 kHz; calibrate the cheap gate's threshold against it. First confirm in the
   repo (`github.com/dstrub18/visqol-rs`) that audio-mode/48 kHz is supported and whether it needs
   a bundled SVR model file.
4. **CI validation harness** (like our external-player-validation rule): run reference Google
   ViSQOL and/or Octave-PEAQ over a fixture corpus to keep the Rust metrics honest — offline only,
   never a runtime dep.
5. **Clean-room PEAQ Basic in Rust — stretch goal**, only if a consumer specifically needs ITU
   ODG numbers; go in expecting non-conformance (the codec-correctness-wall pattern).

**Gate thresholds (starting points):** ODG ≥ −1 (or −0.5) = acceptable, < −2 = flag/re-encode;
for ViSQOL MOS-LQO, calibrate against known-good clips (audio-mode max ≈ 4.75).

## Sources
- ITU-R BS.1387-2; Kabal, *An Examination and Interpretation of ITU-R BS.1387* (McGill 2002).
- GstPEAQ (HSU-ANT, GPL); ViSQOL (google/visqol, Apache-2.0); `visqol-rs` (crates.io, Apache-2.0).
- RFC 7845 (Ogg/Opus pre-skip), RFC 6716 (Opus); Valin et al. Opus paper (delay figures).
