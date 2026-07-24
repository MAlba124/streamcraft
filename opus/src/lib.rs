//! sc-opus — the Opus (RFC 6716) codec plugin. **Not yet wired.**
//!
//! Like the five OxideAV video decoders, the plan was to adopt
//! [`oxideav-opus`](https://github.com/OxideAV/oxideav-opus) — a pure-Rust Opus
//! decoder — rather than hand-write SILK + CELT (the spec's largest codec). The
//! adoption rubric is `vp8/src/lib.rs`: pure Rust, zero `unsafe`, no `build.rs`,
//! MIT, RFC-annotated, and — the deciding gate — *it must actually decode*.
//!
//! # Adoption verdict: REJECT (2026-07-24) — the published crate does not decode Opus
//!
//! The evaluation judged the **published `0.0.13`** — the version this workspace's
//! `Cargo.lock` pins (`checksum 2943bf31…`) — not the crate's git HEAD. This
//! distinction is the whole story, and it is the vp9 lesson taken to the extreme:
//!
//! - **git HEAD** (README + `CHANGELOG.md`, ~67 k lines) advertises a nearly
//!   complete codec: bit-exact SILK vs the RFC 6716 §A reference listing (100 dB
//!   fixture gates), CELT at the float-noise floor (~80–111 dB), hybrid end-to-end
//!   (~71–98 dB), plus PLC, FEC, an encoder, and RFC 7845 multistream. Its shipped
//!   `.expected.wav` fixtures and `silk_reference_waveform.rs` waveform-gate tests
//!   (which do not exist in 0.0.13) corroborate a genuinely working decoder.
//! - **0.0.13, which we actually depend on**, is far behind that HEAD. Its
//!   `FrameDecodeStatus` has no `CeltDecoded`, `HybridDecoded`, or `Concealed`
//!   variants at all — CELT-only and Hybrid frames run their range-coded *prefix*
//!   (coarse energy / allocation) and then emit **silence** (`LayerNotWired`);
//!   there is no MDCT band decode, no hybrid sum, no PLC. Only SILK routes to a
//!   sample-producing path, and that path resamples to 48 kHz with **linear
//!   interpolation** (git HEAD uses the reference fixed-point `SilkUpsampler`),
//!   and predates dozens of decode-correctness fixes the HEAD CHANGELOG records
//!   (the §4.2.7.5.6 P/Q mirror-boundary, the cross-Opus-frame §4.2.7.4 /
//!   §4.2.7.5.5 reconstruction carries, split-table transcription errors, …).
//!
//! ## What was measured (the gate)
//!
//! The official **RFC 6716 test vectors** (`opus_testvectors.tar.gz` from
//! opus-codec.org — 12 vectors, each an `<n>.bit` self-delimited packet stream +
//! an `<n>.dec` reference decode, always interleaved s16le/48 kHz/stereo) were
//! decoded through `oxideav_opus::OpusDecoder::decode_packet` (the crates.io dep in
//! this workspace) and scored as best-offset SNR against the `.dec` reference
//! (scanning a wide pre-skip / group-delay window so a constant delay cannot mask a
//! good decode). Every vector, best case:
//!
//! ```text
//! vector   best_SNR   modes / notes
//! vec01      0.00 dB  CELT prefix only + malformed-packet rejections
//! vec02     -2.27 dB  all-SILK-routed  (should be tens of dB)
//! vec03     -5.92 dB  all-SILK-routed
//! vec04    -11.31 dB  all-SILK-routed
//! vec05      0.00 dB  LayerNotWired  (silence)
//! vec06      0.00 dB  LayerNotWired  (silence)
//! vec07      0.00 dB  CELT prefix only + malformed rejections
//! vec08..11  0.00 dB  CELT prefix / silence / malformed rejections
//! vec12     -0.78 dB  mixed SILK + LayerNotWired
//! ```
//!
//! No configuration clears even 0 dB. Cross-checked against an **ffmpeg/libopus
//! oracle** on the crate's *own* bundled Ogg-Opus SILK fixtures: `silk-nb-mono`
//! decoded at **0.42 dB** and `silk-wb-stereo` at **0.93 dB** versus ffmpeg —
//! i.e. noise. (For calibration, ffmpeg vs the crate's git-HEAD `.expected.wav`
//! reference scored 19.8 dB / 44.8 dB, confirming the *HEAD* decoder is real and
//! the references are sound; it is specifically **0.0.13** that does not decode.)
//!
//! ## Conclusion
//!
//! Wiring 0.0.13 would ship an `OpusDec` that turns every real Opus stream into
//! silence or noise — worse than no element. Per the adoption rubric ("it must
//! actually decode"), the crate is **not adopted at this version** and the
//! `oxideav-opus` dependency is removed so this stub stays honest and buildable.
//!
//! ## Re-evaluate when a real version publishes (the next agent's turnkey path)
//!
//! The git HEAD is the codec we want; it simply has never been published. When
//! oxideav-opus publishes a version whose `FrameDecodeStatus` gains `CeltDecoded`
//! / `HybridDecoded` (the tell that CELT + hybrid landed) — re-run the exact gate:
//!
//! 1. `curl -LO https://opus-codec.org/testvectors/opus_testvectors.tar.gz`
//!    (37 MB; 12 vectors, provenance: the canonical RFC 6716 conformance suite).
//! 2. Split each `.bit` (4-byte BE length, 4-byte BE final-range, then that many
//!    packet bytes; the range is ignored on decode), feed packets in order to one
//!    stateful `OpusDecoder`, and best-offset-SNR against the `.dec` (upmix mono to
//!    stereo to match the always-stereo reference).
//! 3. Gate at a real threshold (the RFC 6716 §6 idea; SILK/CELT/hybrid mainstream
//!    configs at 48 kHz should reach tens of dB). Cross-check with the ffmpeg
//!    oracle (`ffmpeg -i x.opus -f s16le -ar 48000 -`), skipping cleanly if absent.
//!
//! Then wire `OpusDec` per the `flacdec` template (`flac/src/flacdec.rs`): sink pad
//! family `"opus"` (one packet per buffer); an `OpusHead` (RFC 7845 §5.1) first
//! packet configures channels / pre-skip / output-gain (the crate exposes
//! `OpusHead::parse`, `PreSkip`, and `apply_output_gain`); an `OpusTags` packet is
//! consumed silently; a broad dynamic `audio/raw` src pad announcing rate=48000,
//! channels, sample="s16" at first configuration; trim the first `pre_skip` samples
//! (RFC 7845 §4.2); per-buffer warn-drop error scope; `FlushStart` resets decoder
//! state only (no pre-skip re-arm mid-stream); bounded pending carry with
//! `ctx.try_alloc` backpressure. The public API is decode-ready today:
//! `OpusDecoder::{new, reset, decode_packet}` → `DecodedAudio { pcm: Vec<i16>,
//! channels, sample_rate_hz: 48000, frame_outcomes }`.
