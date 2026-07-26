# streamcraft patches to oxideav-vp8 0.2.6

Vendored from crates.io `oxideav-vp8 0.2.6` (wired via `[patch.crates-io]`,
same arrangement as `vendor/oxideav-h264` / `vendor/oxideav-aac`). One
behavioral patch, to be offered upstream:

## Bool decoder: implicit zero bytes past the partition end

`src/bool_decoder.rs` surfaced `BoolDecoderError::EndOfStream` the moment
renormalisation needed a byte beyond the partition slice. Real VP8 encoders do
not flush the arithmetic coder's tail: the final symbols of a partition are
recovered by renormalising against **implicit zero bytes**, which is exactly
how libvpx's reference `bool_decoder_fill` behaves (it pads the value register
with `VP8_LOTS_OF_BITS` worth of zeros at EOF; RFC 6386's §7.3 listing reads
`bool_decoder->input` unboundedly for the same reason).

In practice this rejected ~30% of frames produced by Intel iHD's hardware VP8
encoder — `Macroblock(BoolDecoder(EndOfStream))` in the mode/mv partition and
`MbCoeffs { index: <last MB> … EndOfStream }` in the DCT partition, always at
the very last macroblock — and the same failure class applies to libvpx
streams whose tail happens to need the padding.

The patch (all sites marked `STREAMCRAFT PATCH`):

- `BoolDecoder` gains a `virtual_zeros` counter; both renormalisation byte
  pulls (`renormalize()` and the batched `read_literal` fast path) feed a zero
  byte when the input is exhausted, up to `MAX_VIRTUAL_ZERO_BYTES` (8 — beyond
  any legitimate tail recovery, so truly truncated streams still error fast).
- The `end_of_stream_surfaced` unit test became
  `end_of_stream_pads_virtual_zeros_then_surfaces`, asserting the padded reads
  succeed and the cap still fires.

Verified by `sc-vaapi`'s `vp8_hw_encode_sw_decode_round_trip` (24/24 hardware
frames decode, mean PSNR gate) and the unchanged sc-vp8 suite.
