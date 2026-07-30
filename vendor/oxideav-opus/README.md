# oxideav-opus

[![CI](https://github.com/OxideAV/oxideav-opus/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-opus/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-opus.svg)](https://crates.io/crates/oxideav-opus) [![docs.rs](https://docs.rs/oxideav-opus/badge.svg)](https://docs.rs/oxideav-opus) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust Opus audio codec (SILK + CELT) for the
[oxideav](https://github.com/OxideAV) framework.

## Status

**Clean-room rebuild in progress (orphan scaffold).** The prior
implementation was retired under the workspace clean-room policy; the
crate is being re-implemented from scratch against the published RFCs
using only material under `docs/` and black-box validator binaries.

A top-level `OpusDecoder::decode_packet` packet → PCM orchestration is
now in place: it parses the §3.1 TOC, splits the §3.2 frame packing
(all four frame-count codes), runs the §4.5 multi-frame loop, routes
each Opus frame by mode, and lays out the interleaved 48 kHz output
buffer (RFC 7845 §5.1) with correct per-frame sample counts. Both **mono
and stereo SILK-only** packets now decode **end-to-end to real PCM**: the
§4.2 bitstream decode (the §4.2.3 header bits, the §4.2.5 LBRR / §4.2.6
regular SILK frame loop, each frame decoded in Table-5 order through
gains / LSF chain / LTP / excitation with inter-frame state threaded),
then the §4.2.7.9 LTP / LPC synthesis filters in the **exact
fixed-point arithmetic of the RFC 6716 §A embedded reference listing**
(`silk_decode_core` — Q14 excitation, Q13/Q15 LTP with output-history
re-whitening and gain-change state rescaling, Q14 LPC, i16 output; the
per-subframe LPC selection and all cross-frame histories included),
then the §4.2.8 mono one-sample delay and the §4.2.9 resample to
48 kHz (`SilkUpsampler` — the reference decoder's fixed-point
resampler: per-rate delay compensation, 2× allpass upsampling,
fractional-phase 8-tap FIR interpolation, with the RFC 8251 §5
correction). For **stereo**, the §4.2.2 mid/side interleave (mid
frame then side frame per 20 ms interval, the §4.2.7.2 mid-only flag
skipping the side frame) is decoded into two independent per-channel
synthesis states and converted from mid/side to left/right by the
integer §4.2.8 unmixer (`stereo_ms_to_lr_i16`), run **per SILK
interval** with that interval's §4.2.7.1 weights and the cross-packet
unmix history. The
§4.5.2 SILK state reset (CELT→SILK transition) and the §4.2.7.1
mono→stereo weight reset are applied across packets. **SILK decode is
bit-exact against the reference listing's decoder** (RFC 8251
corrections applied): every pure-SILK fixture and a 100+-stream
oracle corpus (NB/MB/WB × 10/20/40/60 ms × mono + mid/side stereo ×
6–40 kb/s, transient-heavy content) reproduces the reference decode
sample-for-sample at 48 kHz; the gates in
`tests/silk_reference_waveform.rs` sit at a 100 dB floor.
**CELT-only packets now decode
end-to-end to real PCM** (`FrameDecodeStatus::CeltDecoded`): the whole
§4.3 Table-56 entropy layer runs with the normative per-symbol budget
gates (`celt_frame_decode` — silence with the exhausted-budget rule,
the §4.3.7.1 post-filter parameters, transient + intra, §4.3.2.1
coarse energy with its three low-budget fallbacks, §4.3.1 TF flags,
spread, the shrinking-budget dynalloc boosts, trim, the §4.3.3
*implicit* allocation ported in exact 1/8-bit integer arithmetic
(`celt_rate_alloc` — quality-row search, 6-step interpolation
bisection, backward skip decode, intensity / dual-stereo, fine-energy
split), §4.3.2.2 fine energy, the §4.3.4 recursive band decode
(`celt_band_decode` — PVQ leaves with the exact two-stride spreading
rotation, split angles on the triangular / uniform / step PDFs with
bit-exact mid/side weighting, stereo merge + intensity + dual-stereo,
Haar/Hadamard time-frequency reorganization, spectral folding with the
RFC 8251 §9 update, collapse masks), the §4.3.5 anti-collapse, and the
§4.3.2.3 final fine bits), then the signal half (`celt_mdct_synthesis`
— denormalisation in the log2-amplitude energy domain with the
RFC 8251 §8 cap, unit-scale inverse MDCT for long and short blocks
under the low-overlap window, overlap-add, the recursive §4.3.7.1 comb
filter with crossfaded parameter transitions, §4.3.7.2 de-emphasis),
with all cross-frame state carried and the §4.5.2 resets applied.
Validated against the reference decodes of the fixture corpus:
`celt-fb-stereo-128kbps` (20 ms FB stereo) and `celt-2.5ms-low-latency`
reconstruct at **~88–108 dB SNR** — i16-quantization-level waveform
agreement — and a 60+-stream black-box low-bitrate corpus (6–48 kb/s,
2.5–20 ms, mono/stereo, transient-heavy content) decodes at the
float-arithmetic noise floor against the reference listing's decoder
(~80–111 dB; every packet ≥ 55 dB — the formerly-reported transient
seam does not reproduce against a reference-lineage decode). **Hybrid packets decode end-to-end**
(`FrameDecodeStatus::HybridDecoded`): the SILK layer (WB internal) and
the CELT layer (bands 17–21) share one range coder with the §4.5.1
redundancy side information decoded between them (the main coder's
buffer reduced per §4.5.1.3 so its raw bits read from the reduced
end), and the 48 kHz outputs sum per §4.4 — the bit-exact SILK band
lands on the reference timeline through the reference §4.2.9
resampler, and `hybrid-fb-mono-28kbps` decodes at **~71 dB**
whole-stream against the reference-listing decode (float-noise floor;
gated at 60 dB), with hybrid SWB oracle streams at ~93–98 dB. The **§4.5
transition machinery is in place**: the 5 ms redundant CELT frame is
decoded like a CELT-only frame (own coder, no TOC, carrier channels /
bandwidth with the MB→WB override) through the stream's single CELT
state whose geometry adapts without dropping state, the §4.5.2 resets
land where Figure 18 puts them (an end-position redundant frame takes
the reset and warms the following CELT frames; a beginning-position
one continues the previous state ahead of the deferred main-layer
reset), and the §4.5.1.4 output stitching (first-2.5 ms-as-is +
power-complementary cross-lap) is applied on both placements — the
`mode-switching` fixture decodes at ~103 dB whole-stream against the
reference-listing decode (hybrid segment, transition window and
CELT-only segment all at the float-noise floor). **Packet-loss concealment
(§4.4)** is implemented per the RFC's per-mode guidance
(`OpusDecoder::conceal_loss`): LPC extrapolation (Burg fit +
pitch-cyclic residual) after SILK-bearing frames, pitch-periodic
waveform repetition after CELT-only frames, an energy-decay envelope
across consecutive losses down to the silence floor, and a 2.5 ms
extrapolation tail cross-lapped into the first packet decoded after
the loss run; in-band FEC (`decode_packet_fec`) remains the preferred
recovery when the next packet is available.

The crate now also carries the start of the **encode side**: the
bit-exact §5.1 range *encoder* (`RangeEncoder` — the §5.1.1 symbol
update, §5.1.1.2 carry propagation, the §5.1.2 division-free variants
sharing the decoder's `icdf[]` tables, §5.1.3 raw bits, §5.1.4
uniform integers, §5.1.5 finalization, §5.1.6 `tell`/`tell_frac`
matching the decoder bit-for-bit), write-side mirrors of **every**
SILK §4.2.7 decode stage (header / gains with a deterministic
quantizer / LSF stage-1 + stage-2 / interpolation index / LTP / seed
/ excitation, each returning the value the decoder will
reconstruct), the whole-frame Table-5 composition
(`encode_silk_frame`), and SILK-only **packet encoders for both mono
and stereo** (`encode_silk_only_packet_mono` /
`encode_silk_only_packet_stereo`: TOC byte + §4.2.3/§4.2.4 header
bits + 1–3 SILK frames at 10/20/40/60 ms — the stereo entry writing
the §4.2.2 mid/side interleave with the §4.2.7.1 weight quintuple and
gated §4.2.7.2 mid-only flag on each mid frame, and two independent
per-channel carried states) whose packets decode end-to-end through a
fresh `OpusDecoder::decode_packet` to real SILK PCM, with every
per-frame parameter verified equal to the encoder's prediction. LBRR
(in-band FEC, §4.2.5) emission is included for both channel layouts
and closes the FEC loop: `decode_packet_fec` recovers real (mono or
two-channel) audio from the encoder's own redundancy. On top of the
packet writers sit the **stereo analysis front half** — the exact
§4.2.8 algebraic-inverse downmix `stereo_lr_to_ms` (L/R → mid/side
with the decoder's weight-interpolation ramp; roundtrips to the
input at the §4.2.8 one-sample delay), the least-squares §4.2.7.1
weight estimator `estimate_stereo_weights`, and the exhaustive
codebook quantizer `StereoWeightSymbols::quantize` — plus the **§3.2
/ Appendix-B framing writers** (`compose_packet`,
`compose_packet_code3`, `compose_self_delimited`; all four codes,
CBR/VBR, §3.2.5 padding chains, parser-validated R2/R3/R5/R6) and
the **RFC 7845 write side** (`OpusHead::compose`, byte-identical on
reparse, and `assemble_multistream_packet`, roundtripped against the
splitter and decoded sample-identically through
`MultistreamDecoder`). On top of it all now sits the **§5.2.3 SILK
signal analysis** — `encode(pcm)` is real: `SilkEncoderMono` /
`SilkEncoderStereo` derive every Table-5 symbol from internal-rate
PCM across the full SILK packet matrix — 10 / 20 / 40 / 60 ms
packets (one 2-subframe frame, or one to three 20 ms frames with the
intra-packet delta-gain / §4.2.7.6.1 relative-lag / §4.2.7.6.3
scaling-presence threading of the decoder's regular walk), per-frame
§4.2.3 VAD flags derived from the signal (silent intervals code
frame type 0 and skip the pitch search), §4.2.5 **LBRR in-band FEC**
from PCM (`set_fec(true)`: each packet re-encodes the previous
packet's active intervals at a reduced rate from a pre-packet
analyzer snapshot with a fresh closed-loop state, recovered
end-to-end through `decode_packet_fec`), and §3.2.5 **CBR transport
shaping** (`encode_packet_cbr` / `pad_packet_to`: exact-byte-size
code-3 re-framing, every target size reachable, decode-identical).
The chain is Burg's-method LPC
(§5.2.3.4.2.1) → analysis-direction LPC→NLSF conversion (deflated
line-spectral root search, verified as the exact inverse of the
§4.2.7.5.6 fixed-point reconstruction) → exhaustive stage-1
analysis-by-synthesis NLSF quantisation scored on the real decode
chain → whitened-domain §5.2.3.2 pitch analysis with joint
(primary-lag × Table 33-36 contour) quantisation → §5.2.3.6
exact-distortion LTP codebook search → per-subframe residual-energy
gain selection through the §4.2.7.4 quantizer (cross-packet
clamp-safe) → a closed-loop excitation quantiser (the §5.2.3.8 role)
that rounds each pulse against the prediction the decoder will
actually form, LCG sign inversion included, and updates the carried
state through the real §4.2.7.9 synthesis chain. Sine, pulse-train
(voiced), and amplitude-panned stereo inputs all decode back through
the real streaming `OpusDecoder` at >10 dB tone-projection SNR on
the 48 kHz output, with stereo panning preserved.

Round 418 completed the encoder arc beyond SILK: **CELT-mode packet
encode is real, end to end** (`CeltEncoder`). The full §5.3 stage
sequence mirrors the §4.3 decoder symbol for symbol — silence flag,
post-filter (off), transient analysis + short blocks, two-pass
intra/inter §5.3.2 coarse energy with the decoder-lockstep quantized
`oldBandE` carry, budget-gated tf flags, the §5.3.4 spreading
decision, the dynalloc boost loop, trim analysis, the §4.3.3
allocation with coded skip / intensity / dual-stereo decisions
(encode→decode roundtrips to identical allocations), fine energy,
the recursive §4.3.4 band encode (split angles measured from the
band energies on the step/uniform/triangular PDFs, intensity
collapse, Haar/Hadamard time reorganisation, PVQ pyramid search +
exact §4.3.4.2 index construction at the leaves, the decode side's
exact 1/8-bit budget bookkeeping), the anti-collapse bit, the final
fine backfill, and the fixed-size §5.1.5 finalization where range
bytes and raw bits share exactly the frame's bytes. The whole
configuration matrix encodes — NB/WB/SWB/FB × 2.5/5/10/20 ms × mono
+ stereo at any constant payload 2..=1275 bytes — and every stream
was validated through BOTH decoders: the crate's own `OpusDecoder`
(13–46 dB multitone SNR by rate with a monotone rate ladder) and the
RFC 6716 §A reference-listing decoder (RFC 8251-patched,
hash-verified extraction), which reconstructs our streams
identically to ours at 88–108 dB (float-noise floor, max 1 LSB).
At matched CBR rates on the same content our encoder lands within
2.7–3.3 dB of the reference listing's own encoder (32→128 kb/s
sweep). **Hybrid encode works too** (`HybridEncoderMono`, configs
12–15: SWB/FB × 10/20 ms): the WB SILK layer and the CELT bands
17.. share one range coder with the §4.5.1.1 redundancy flag coded
off under the decoder's 37-bit gate, and the two layers sit on one
timeline (a 165-tap linear-phase 48→16 kHz decimator's 82-sample
delay + the §4.2.9 resampler's 35 + the §4.2.8 mono delay's 3
exactly equal the CELT path's 120-sample MDCT-overlap delay; an
empirical best-lag search returns 120). Hybrid streams decode
through both decoders as well (the listing decoder agrees with ours
at 105–108 dB). The SILK layer has no rate control yet, so a payload
it alone would overflow is rejected cleanly.

Round 431 adds **Opus-level VBR** (RFC 6716 §2.1.8 / §3.2.1):
`vbr::VbrRateControl` elects every code-0 packet's size against a
target bitrate — unconstrained mode corrects by the accumulated drift
(clamped to ±one frame's target, so silence cannot bank an unbounded
spree), constrained mode adds the §2.1.8 bit-reservoir simulation
(spend above target only what below-target packets banked; bank
capped at a documented 100 ms default, giving the provable
`n·target + cap` bound on every n-packet window). `CeltVbrEncoder`
covers the full CELT matrix with 3-byte digital-silence collapse and
a transient pre-detect boost the drift repays; `HybridVbrEncoderMono`
rides the new `encode_packet_elected` (SILK floor raises feed the
drift); the SILK-only arm's natural quality-driven emission is
already VBR per §2.1.8 (a 21% transport saving over CBR padding at
bit-identical decode) but electing it against a target needs
SILK-layer rate control, which remains open. Realized averages land
within 2–5% of target on every arm and frame size; at matched
average rate VBR ≥ CBR on steady content and beats CBR by ~3.3 dB on
mixed tone/silence content at equal total bytes. A 15-stream VBR
corpus (CELT NB/WB/SWB/FB × 2.5–20 ms × mono/stereo ×
constrained/unconstrained + all four Hybrid configs) decodes through
the §A reference-listing decoder with exact packet and sample counts,
agreeing with our decoder at 90–107 dB (max 1 LSB). Stereo hybrid,
tf analysis, and the encoder-side pitch pre-filter remain open.

Differential encoder/decoder testing and a restored cargo-fuzz suite
(6 coverage-guided targets, incl. an encoder↔decoder range-coder
roundtrip and the CELT / VBR encode→decode harnesses) have also
hardened the decoder: five mis-transcribed rows
in the §4.2.7.8.3 split tables (now verified cell-by-cell against the
RFC across all 64 rows), a `dec_bits(32)` shift overflow, a
§4.2.7.5.8 recurrence i64 overflow on adversarial input, and the
§4.2.7.8 10 ms-MB 128-vs-120-sample special case (previously every
10 ms MB SILK packet failed to synthesize) are all fixed with
regression tests. The round-388 encoder work exposed one more
long-standing decode bug: the §4.2.7.5.6 P/Q recurrence dropped the
"p_Q16[k][k+2] = p_Q16[k][k]" symmetric-mirror boundary condition at
the j = k+1 read (substituting 0), producing badly wrong LPC filters
that burned up to 12 prediction-gain-limiter rounds on perfectly
stable codebook vectors — now fixed and pinned by an analytic
closed-form regression over all 64 NB/WB stage-1 codebook entries.
Round 391 closed two more reconstruction-level streaming gaps: the
§4.2.7.4 gain-clamp base (`previous_log_gain`) and the §4.2.7.5.5
NLSF interpolation base `n0` now carry ACROSS Opus frames in the
streaming `OpusDecoder` (both were previously re-armed per packet,
so the first frame of every packet skipped the independent-gain
clamp and ignored its coded `w_Q2`), cleared exactly on the RFC's
reset events (§4.5.2 SILK reset, bandwidth change, uncoded side
frame) and seeded from the LBRR reconstruction after FEC recovery
under the §4.2.7.4 packet-loss latitude.

The crate ships a large, individually unit-tested set of SILK and
CELT building blocks plus a complete RFC 7845 multistream /
multichannel decode subsystem (1440+ lib tests + SILK-fixture,
multistream (incl. the 5.1 reference-listing gate), FEC, CELT
synthesis-backend, CELT-encode, Hybrid-encode, VBR, and
registry-resolution integration suites). Per-stage progress lives in
`CHANGELOG.md`.

## What works

**Packet → PCM orchestration (RFC 6716 §3 / §4):**

- `OpusDecoder::decode_packet` — the top-level packet → interleaved
  48 kHz PCM path: TOC parse, §3.2 frame split, §4.5 multi-frame loop,
  per-mode routing, the §4.5.2 cross-packet SILK state reset, the
  cross-packet §4.2.7.4 / §4.2.7.5.5 reconstruction carry, and the
  RFC 7845 §5.1 output sample-count layout. Mono SILK-only packets decode
  end-to-end to real PCM (bitstream → §4.2.7.9 synthesis → §4.2.9
  resample); other modes emit correct-length silence flagged via
  `FrameDecodeStatus`.
- `silk_decode::decode_silk_frame` — the §4.2.6 / §4.2.7 in-order SILK
  frame decode that composes the per-stage decoders in exact Table-5
  symbol order and runs the LSF → stable-Q12-LPC chain.
- `silk_synthesis::synthesize_silk_frame` — the §4.2.7.9 synthesis
  composition: §4.2.7.9.1 LTP + §4.2.7.9.2 LPC filters with the §4.2.7.9
  per-subframe LPC selection and cross-frame `SilkSynthState` histories,
  producing internal-rate (8/12/16 kHz) time-domain samples.
- `OpusDecoder::decode_silk_only_stereo` — the §4.2.2 stereo SILK decode:
  the §4.2.3 two-channel header bits, the §4.2.5 / §4.2.6 interleaved
  mid/side SILK frames (the §4.2.7.1 weights + §4.2.7.2 mid-only flag on
  the mid frame; an uncoded side frame clears its §4.2.7.9 LTP buffer per
  §4.5.2), two independent per-channel synthesis states, and the §4.2.8
  `silk_stereo::stereo_ms_to_lr` mid/side → left/right unmix run per SILK
  interval into interleaved L/R PCM.

**Packet & framing (RFC 6716 §3 / §4.2):**

- `OpusTocByte` — the §3.1 TOC parser (config × stereo flag × frame-count
  code).
- `OpusPacket` — the §3.2 frame-packing parser for all four frame-count
  codes (single, two-equal, two-unequal, signalled with optional VBR
  lengths + padding); returned frame slices borrow from the input.
- `parse_self_delimited` — RFC 6716 Appendix B self-delimiting framing
  (for chaining inside a multistream demuxer).
- `OpusFrameRouting` — §3.1 / §4.2 mode dispatch (SILK-only / Hybrid /
  CELT-only, SILK-frame count, per-frame LBRR-flag gating, channel
  multiplier).
- A §3.4 R1–R7 malformed-input rejection audit
  (`tests/malformed_input.rs`).
- An **end-to-end SILK fixture-decode suite** (`tests/silk_fixture_decode.rs`)
  that decodes the in-project NB-mono / WB-stereo / MB-60 ms-mono Opus
  streams packet-by-packet through `decode_packet` and validates §3.1 TOC
  routing, whole-stream error-free SILK decode (mono + stereo, NB/MB/WB,
  20/60 ms), §3 sample-count accounting, and 440 Hz dominance on the NB
  sine fixture.
- A **SILK waveform regression-gate suite**
  (`tests/silk_reference_waveform.rs`) that compares each SILK-bearing
  fixture's pre-skip-trimmed 48 kHz decode against its shipped
  reference decode (produced by the §A reference listing's decoder
  with the RFC 8251 corrections) at a **100 dB floor** — the SILK
  fixtures decode bit-exactly, pinning the fixed-point §4.2.7.9 core,
  the integer §4.2.8 unmix + mono delay, and the reference §4.2.9
  resampler.

**Multistream / multichannel (RFC 7845 §3 / §5.1 / §5.1.1):**

- `OpusHead` — the §5.1 identification-header parser: version (with the
  major-nibble compatibility bound), output channel count, pre-skip,
  input sample rate, output gain, mapping family, and the §5.1.1
  channel-mapping table (stream count N, coupled count M, per-output
  mapping indices). Enforces every MUST in §5.1 / §5.1.1 (non-zero
  channel/stream counts, per-family channel ranges, `M ≤ N`,
  `M + N ≤ 255`, and the `< M+N` / 255 mapping-index bound). Family 0
  synthesizes the table from the RFC-pinned defaults.
- `split_multistream_packet` — the §3 N-packet split: the first `N − 1`
  streams via Appendix-B self-delimited framing, the final stream as the
  undelimited remainder.
- `MultistreamDecoder` — the multichannel decode: one stateful
  sub-decoder per coded stream, decoding each split packet and
  assembling the `C` output channels by the §5.1.1 index rule
  (coupled-stream L/R by parity, mono streams, index-255 silence, a
  decoded channel routed to multiple outputs), with the §3 equal-duration
  constraint enforced. Validated end-to-end against the real SILK
  fixtures: an `N = 1` family-0 decode is byte-identical to a plain
  `OpusDecoder`, a coupled-stream L/R split reproduces a plain stereo
  decode exactly, and mono-pair / swapped / silence / duplicate maps all
  route correctly.
- `apply_output_gain` / `PreSkip` — the §5.1 post-decode output-gain
  application (Q7.8 dB, i16-saturating) and the cross-packet pre-skip
  accumulator.
- `register(ctx)` — the framework registration declares the `opus`
  codec id with its RFC 7845 §5.1 payload magic (`OpusHead`), so
  container layers without a codec tag resolve an Opus logical stream
  from its first payload bytes
  (`CodecRegistry::resolve_payload_magic_ref`); `OpusTags` and every
  truncation of the magic are refused by construction (pinned in
  `tests/registry_resolution.rs`).

**Range coder (RFC 6716 §4.1 / §5.1):** `RangeDecoder` — the shared
entropy primitive consumed by both layers, including the §4.1.2
two-step `ec_decode` / `ec_dec_update` path and the Laplace / iCDF
helpers — and `RangeEncoder`, its bit-exact §5.1 write-side mirror
(validated by per-primitive roundtrips, `tell`/`tell_frac` lockstep,
a 5000-seed mixed-symbol fuzz roundtrip, and a coverage-guided
libfuzzer differential target).

**SILK encode side (RFC 6716 §5.2 bitstream back end):** write-side
mirrors of every §4.2.7 stage sharing the decode tables
(`SilkFrameHeader::encode_pre_gains` / `encode_lsf_stage1`,
`SubframeGains::encode`/`quantize`, `LsfStage2::encode`,
`LsfInterpolated::encode_index`, `encode_lcg_seed`,
`LtpParameters::encode`, `Excitation::encode`), the Table-5
whole-frame composition `encode_silk_frame`, the §4.2.3/§4.2.4
header-bit writer `SilkHeaderBits::encode` (mono + two-channel), the
§3.1 TOC composer `OpusTocByte::compose_byte`, and the packet-level
`encode_silk_only_packet_mono` / `encode_silk_only_packet_stereo`
(each with a `_with_lbrr` variant for §4.2.5 in-band-FEC emission;
the stereo entry writes the §4.2.2 mid/side interleave with the
§4.2.7.1 weights and gated §4.2.7.2 mid-only flag per interval and
threads two independent per-channel carried states, exactly
mirroring the decoder's stereo walk) — every layer
roundtrip-verified against the decoder, up to whole packets decoding
end-to-end through `OpusDecoder::decode_packet` (mono and stereo)
and FEC recovery through `decode_packet_fec`.

**Stereo encode analysis (§4.2.7.1 / §4.2.8 write half):**
`stereo_lr_to_ms` — the exact algebraic inverse of the §4.2.8
unmixer (frame-aligned L/R → mid/side with the decoder's
weight-interpolation ramp, one-sample lookahead for the final `p0`,
`StereoDownmixState` history; a multi-frame roundtrip through
`stereo_ms_to_lr` reproduces the input at the §4.2.8 one-sample
delay) — `estimate_stereo_weights` (least-squares fit of the raw
side onto the `p0` / mid predictor pair, f64 normal equations) and
`StereoWeightSymbols::quantize` (exhaustive deterministic argmin
over the 5625-quintuple §4.2.7.1 codebook; representable targets
roundtrip value-exactly).

**Packet-framing / RFC 7845 write side:** `compose_packet` /
`compose_packet_code3` / `compose_self_delimited` / `encode_length` —
the §3.2 + Appendix-B framing writers (all four codes, CBR/VBR,
§3.2.5 padding chains, every parser-enforced requirement validated
before writing; roundtripped against `OpusPacket::parse` /
`parse_self_delimited`, including chained self-delimited buffers and
multi-frame SILK packets decoding end-to-end) — plus
`OpusHead::compose` (byte-identical reparse, full §5.1/§5.1.1 MUST
validation) and `assemble_multistream_packet` (§3 stream packing via
the Appendix-B writer, equal-duration constraint enforced,
sample-identical decode through `MultistreamDecoder`).

**SILK (RFC 6716 §4.2):** frame-header decode (§4.2.7.1–§4.2.7.5.1),
subframe gains (§4.2.7.4), the full LSF chain (stage-2 residual → NLSF
reconstruction → stabilization → interpolation → NLSF→LPC →
bandwidth-expansion → prediction-gain limiting, §4.2.7.5.2–§4.2.7.5.8),
LTP parameters (§4.2.7.6), LCG seed (§4.2.7.7), excitation
(§4.2.7.8), LTP + LPC synthesis filters (§4.2.7.9), stereo unmixing
(§4.2.8, including the mono one-sample delay), the §4.2.9 resampler
(`SilkUpsampler` — the reference decoder's fixed-point resampler over
the Table 54 budget machinery), and **in-band FEC
recovery** (§2.1.7 / §4.2.5): `OpusDecoder::decode_packet_fec`
reconstructs a lost frame's audio from the Low Bit-Rate Redundancy
(LBRR) frames carried in the next received packet — decoding the §4.2.5
LBRR frame(s) (mono, or interleaved mid/side for stereo), running the
full §4.2.7.9 synthesis from a fresh state, unmixing a stereo recovery
via §4.2.8, and resampling to 48 kHz, reported through `FecDecodeStatus`.

**CELT (RFC 6716 §4.3 / §4.5):** the §4.3 band layout (Table 55), the
pre-band header symbols (silence / post-filter / transient / intra),
the §4.3.4.5 *time-frequency change decode* (`celt_tf_decode` — the
per-band `tf_change` flag loop, first band absolute and subsequent
bands difference-coded relative to the previous band's choice, plus the
§4.3.1-gated `tf_select` flag and the resulting per-band TF adjustment
vector) layered on the §4.3.4.5 TF-resolution adjustment tables, the
coarse-energy Laplace
parameter tables (§4.3.2.1), the allocation parameter surfaces
(log2-frac / alloc-trim / cache-caps / static-allocation), the
§4.3.4.1 *Bits-to-Pulses* pulse-cost cache (the run-packed
`cache_bits50` / `cache_index50` lookup plus the budget-to-pulse-count
inversion), the §4.3.6 band denormalisation (unit-norm PVQ shape ×
`sqrt(2**log2_energy)`, laid out across the coded bands into the
inverse-MDCT input buffer), the §4.3.7 inverse MDCT transform core (the
`N` frequency-domain bins → `2N` time-domain samples mapping, scaled by
`1/2`, with the §4.3.7 overlap-add window already landed at
`celt_mdct_window`), the §4.3.7 *weighted overlap-add* (`celt_overlap_add`
— the stateful per-channel adder that windows each `2N` inverse-MDCT
block with the low-overlap synthesis window and overlap-adds the leading
half with the previous block's windowed trailing half at hop `N`,
carrying the overlap history across frames and reconstructing the
aliasing-free time-domain signal), the §4.3.4.5 *time-frequency Hadamard
transform* (`celt_tf_hadamard` — the across-block / sequency-order
orthonormal Walsh–Hadamard reshaping that consumes the per-band
`TfDirection`, preserving the unit-norm shape energy), the §4.3.4
*per-band shape decode orchestrator* (`celt_band_shape` — composing
§4.3.4.2 PVQ decode → §4.3.4.3 spreading → §4.3.4.5 TF transform into
one `decode_band_shape` call given a band's `(N, K, spread, tf_adjust,
nb_blocks)`), and the §4.5 redundancy / mode-transition state-reset
machinery.

(The §4.3.3 allocation orchestration and the §4.3.5 anti-collapse —
once listed here as structural blockers — have long been implemented
in exact integer arithmetic; the historical note is kept only in the
changelog. With the RFC 6716 §A embedded reference listing ratified as
staged spec material, no CELT or SILK decode stage remains blocked on
external documentation.)

**CELT / Hybrid encode side (RFC 6716 §5.3):** `CeltEncoder`
(`celt_packet_encode`) — CELT-only code-0 packets over the full
configuration matrix at any constant payload size, via
`celt_analysis` (pre-emphasis, forward MDCT, band energies, transient
detector), `celt_energy_encode` (two-pass coarse + fine + finalise),
`celt_alloc_encode` (the §4.3.3 allocation, encode side),
`celt_band_encode` (the recursive §4.3.4 band coder, encode side),
`celt_pvq_encode` (PVQ search + §4.3.4.2 index construction), the
§4.3.2.1 Laplace encoder, and `RangeEncoder::finish_fixed` (the
fixed-size §5.1.5 finalization) — plus `HybridEncoderMono`
(`hybrid_packet_encode`): the WB SILK layer and CELT bands 17.. on
one range coder with delay-matched layer alignment, now including
`encode_packet_elected` (elected payload with SILK-floor raise).
Validated through the crate's own decoder and the §A
reference-listing decoder (88–108 dB agreement between the two
decoders on our streams).

**Opus-level VBR (RFC 6716 §2.1.8 / §3.2.1):** `vbr::VbrRateControl`
— the per-frame size election under a target-bitrate drift
controller, with the constrained-VBR bit-reservoir discipline
(`elect_packet_bytes` / `commit` / `constrained_ceiling_bits`) — and
its mode arms `vbr::CeltVbrEncoder` (silence collapse, transient
boost) and `vbr::HybridVbrEncoderMono` (floor-raise feedback).
Gated by `tests/vbr_encode_roundtrip.rs` (rate tracking, parity vs
CBR, the constrained window bound under adversarial bias, exact
frame accounting) and the `vbr_encode_roundtrip` fuzz target.

## Clean-room sources

The rebuild consults only:

- RFC 6716 — Definition of the Opus Audio Codec, **including its
  Appendix A embedded reference listing** (extracted from the staged
  RFC text itself and hash-verified against the RFC-pinned digest;
  ratified as staged spec material). An instrumented build of the
  listing serves as the decode-exactness oracle.
- RFC 8251 — Updates to the Opus Audio Codec, including the correction
  patches embedded in its text (applied to the oracle and reflected in
  the decoder).
- RFC 7587 — RTP Payload Format for Opus.
- RFC 7845 — Ogg Encapsulation for Opus.
- Black-box invocations of the `opusdec` / `opusenc` binaries (not
  their source) as opaque validators.

No external library source is permitted as a reference under the
workspace clean-room policy.

## License

MIT. See `LICENSE`.
