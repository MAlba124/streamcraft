# sc-flac spec notes

Interpretation decisions and errata for the FLAC implementation, per the codec
conventions (spec: First-party codecs — "Spec errata and interpretation decisions get
documented in `spec/NOTES.md` per crate"). The normative text is `rfc9639.txt`
(RFC 9639, "Free Lossless Audio Codec", December 2024). Code cross-references sections
as `§9.2.7` etc.

## Reference file

`rfc9639.txt` is the authoritative IETF text (`https://www.rfc-editor.org/rfc/rfc9639.txt`).
The two worked examples in Appendix D are used verbatim as known-answer tests:

- **Example 1 (§D.1)** — stereo, 16-bit, 44100 Hz, 1 interchannel sample, two VERBATIM
  subframes with wasted bits. Drives `decoder::tests::decode_example_1_from_spec`
  (whole-file decode) and the CRC-8 / CRC-16 / STREAMINFO-layout / frame-header
  known-answer tests.
- **Example 2 (§D.2)** — a FIXED order-1 subframe with a partitioned Rice residual and
  side-right stereo. Used to cross-check the residual and CRC logic by inspection; its
  header CRC-8 (0x99) is a known-answer test.

## Encoder scope (what we emit)

- **Metadata**: one STREAMINFO block only (§8.2), marked as the last block. No
  seektable / Vorbis comment / picture blocks yet — tags will arrive via the framework
  `Tag` event, not by hand-rolling a Vorbis comment here.
- **Subframes** (§9.2): `CONSTANT`, `VERBATIM`, and `FIXED` orders 0–4. The best of the
  three is chosen per subframe by an estimated bit-cost heuristic (integer, exact for
  the residual portion). **LPC (§9.2.6) is not emitted** — FIXED predictors alone make
  a valid, well-compressing stream; LPC is a later compression win.
- **Residual** (§9.2.7): partitioned Rice, method 0b00 (4-bit) or 0b01 (5-bit)
  parameters, with the partition order searched under the streamable-subset cap of 8
  (§7). The escape code (§9.2.7.1) is never emitted: the §9.2.7.3 range check
  guarantees a finite Rice cost exists, and VERBATIM is the fallback when no FIXED
  predictor keeps residuals in range.
- **Stereo**: channels are coded **independently** (channel assignment = channels − 1).
  Left/side, side/right, and mid/side decorrelation (§4.2) are **not** applied by the
  encoder (a future compression win). The decoder *does* implement all three, so it can
  read real-world and reference-encoder output.
- **Streamable subset (§7)**: output stays in-subset — block size ≤ 4608 for ≤ 48 kHz,
  partition order ≤ 8, no streaminfo-relative frame-header fields, standard channel
  order. So `flac`/libFLAC and hardware decoders accept it.

## Decisions

- **MD5 = all zeros ("unknown")**. §8.2 explicitly allows a zero MD5 to mean "not
  known". We do not compute the MD5 of the unencoded audio (a hand-written MD5 is a
  nice-to-have, not required for a valid stream). `flac -t` therefore verifies sync and
  CRCs but reports no MD5 check; losslessness is proven instead by the in-crate
  round-trip and by `flac -d` diffing against the original PCM.

- **Two STREAMINFO fill modes.**
  - *Two-pass* (`FlacEncoder::new` + `finish`): the placeholder header is patched after
    encoding with exact min/max block size, min/max frame size, and total samples. Used
    by the round-trip / cross-validation tests and the benchmark, where the output
    buffer is in memory and can be back-patched.
  - *Streaming, forward-only* (`FlacEncoder::new_streaming`, used by the `FlacEnc`
    element): min/max block size are set to the fixed target block; **frame sizes and
    total samples are written as 0 == "unknown"** (§8.2, §9.1 both allow 0). A
    sequential sink (`filesink`) cannot be back-patched, and 0 is spec-legal, so this
    keeps the element single-pass without buffering the whole file.

- **Fixed block size stream** (blocking strategy bit = 0, §9.1). The coded number
  (§9.1.5) is therefore a frame number. Arbitrary block sizes use the 8-/16-bit
  "uncommon block size" escape (§9.1.6); the last (short) frame uses the 8-bit escape.

- **Coded-number encoding** (§9.1.5 Table 18): the extended UTF-8-like scheme with a
  leading run of one bits equal to the total byte length (a 2-byte code leads with
  `0b110`, a 7-byte code with `0b11111110`). Verified against the §9.1.5 worked example
  (51 billion) and by round-tripping through the decoder.

- **Arithmetic width**: all sample/residual arithmetic uses `i64`. Per Appendix A, an
  order-4 FIXED residual of ≤ 32-bit input needs at most bps+4 bits, so `i64` cannot
  overflow; the only range check performed is the normative §9.2.7.3 limit
  (|residual| < 2^31), after which the encoder falls back to VERBATIM.

## Not yet / follow-ups

- LPC subframes (encode side) for better compression on complex material.
- Stereo decorrelation on the encode side (independent channels today).
- Hand-written MD5 of the unencoded audio for the STREAMINFO checksum.
- Encoder speed: the per-subframe search evaluates all five FIXED orders and a full
  partition-order sweep; a prefix-sum / bit-count reuse across orders and partition
  orders would cut the constant factor without changing output.
