# Changelog

All notable changes to `oxideav-opus` are recorded here.

## [Unreleased]

- **VBR-election fuzz target** (`fuzz/vbr_encode_roundtrip`, round
  431): coverage-guided config × target-bitrate × constrained-flag ×
  PCM exploration asserting every elected packet decodes cleanly
  through the streaming decoder with the exact sample count, sizes
  stay inside the §3.2.1 limits, and constrained-mode packets never
  outrun the controller's pre-encode reservoir ceiling. Pre-hardened
  locally (600 s, ~230k runs across the config/bitrate surface,
  clean). The 15-stream VBR validation
  corpus (CELT NB/WB/SWB/FB × 2.5–20 ms × mono/stereo ×
  constrained/unconstrained + all four Hybrid configs) also decodes
  through the §A reference-listing decoder (RFC 8251-patched,
  hash-verified extraction) with exact packet/sample counts at
  90–107 dB agreement with our decoder (max 1 LSB).

- **Opus-level VBR** (round 431): `vbr::VbrRateControl` — per-frame
  packet-size election under a target-bitrate drift controller
  (RFC 6716 §2.1.8 / §3.2.1). Unconstrained mode elects
  `target − drift (+ content bias)` with the drift clamped to ±one
  frame's target (a long silence cannot bank an unbounded spending
  spree); constrained mode adds the §2.1.8 "simulates a bit
  reservoir" discipline — spend above target only what earlier
  below-target packets actually banked, bank capped at a documented
  default of 100 ms of target bitrate (the RFC specifies the
  reservoir concept, not a bound), giving the provable window bound
  `n·target + cap` for every window of n packets. On top:
  `vbr::CeltVbrEncoder` (full config matrix — NB/WB/SWB/FB ×
  2.5/5/10/20 ms × mono/stereo; digital-silence frames collapse to
  3-byte packets, a PCM-domain transient pre-detector grants a +25%
  boost the drift later repays) and `vbr::HybridVbrEncoderMono`
  (configs 12–15; the election rides `encode_packet_elected`, SILK
  floor raises feed back into the drift). The SILK-only arm's VBR is
  its natural quality-driven emission (§2.1.8: the LP layer is
  inherently VBR); election around a target there needs SILK-layer
  rate control, which remains an open frontier. Integration gates
  (`tests/vbr_encode_roundtrip.rs`): realized average within 2–5% of
  target on every arm/LM, exact packet/frame/sample accounting,
  silence collapse + bounded post-silence spend + reconvergence,
  transient boosts firing and repaid, the constrained per-window
  bound holding under adversarial bias, VBR ≥ CBR − 0.7 dB at
  matched average rate on steady content, and VBR > CBR + 1 dB on
  mixed tone/silence content at equal total bytes (measured +3.3 dB).

- **Hybrid elected-payload encode** (round 431):
  `HybridEncoderMono::encode_packet_elected(pcm, elected)` — the VBR
  entry point. The §4.2 SILK layer (quality-driven, no rate control)
  is coded first on a fresh range coder; the election is then
  honoured when it leaves room and otherwise raised to the SILK floor
  (`ceil(silk_bits/8) + HYBRID_MIN_CELT_TAIL_BYTES`, capped at the
  §3.2.1 1275-byte frame limit) before the §4.5.1.1 redundancy flag
  and the §4.3 CELT tail land on the same coder. `encode_packet`
  (CBR) is re-expressed over the same two phases with identical
  behaviour. Floor raises are pinned across all four Hybrid configs
  (SWB/FB × 10/20 ms): a 2-byte election still yields packets that
  decode as `HybridDecoded` with exact sample counts.

- **`OpusHead` payload-magic registration** (round 431): `register(ctx)`
  now declares the `opus` codec id via
  `CodecInfo::payload_magic(b"OpusHead")` (core 0.1.33), so container
  layers with no codec tag resolve an Opus logical stream from the
  fixed 8-octet RFC 7845 §5.1 identification-header signature.
  Capability metadata (lossy audio, 48 kHz, up to 255 mapped output
  channels per §5.1.1) rides along. `tests/registry_resolution.rs`
  pins the positive resolutions (full §5.1 header, exact-length magic,
  the `CodecResolver` dyn surface) and the refusals: the RFC 7845 §5.2
  `OpusTags` comment header, every proper truncation of the magic
  (7 bytes down to empty), a corrupted final octet, and unrelated
  payloads.

- **CELT-encoder fuzz target** (`fuzz/celt_encode_roundtrip`):
  coverage-guided config × payload × PCM exploration asserting every
  produced packet decodes cleanly through the streaming decoder with
  the exact sample count. Pre-hardened locally with a 640-packet
  full-scale adversarial sweep (white noise, Nyquist square, DC rails,
  step discontinuities × every bandwidth/frame-size/channel config ×
  payloads 2..1275) plus a 60-packet mixed-content stereo stress whose
  streams the reference-listing decoder reproduces at 94.7 dB / max
  1 LSB.

- **§5.3.4 spreading decision** (round 418): the CELT encoder now runs
  the listing's tonality analysis (per-band magnitude-CDF measure,
  recursive averaging, hysteresis against the previous decision)
  instead of hardcoding SPREAD_NORMAL; transients and tiny budgets
  keep the fixed decision exactly as the listing gates them. Oracle
  re-validated: the reference-listing decoder still reconstructs our
  streams identically to our decoder (~100 dB).

- **multistream-5.1 fixture re-anchored to the reference listing**
  (round 418). The 5.1 fixture's reference decode was left on the old
  third-party lineage in the round-415 re-anchor; the staged
  `docs/audio/opus/fixtures/multistream-5.1/expected.wav` is now the
  decode of the §A listing's own multistream decoder (RFC 8251
  patches applied), pre-skip trimmed, end-trimmed to the granule
  length, in the RFC 7845 §5.1.1.2 Vorbis channel order (the old file
  was WAV-order and not byte-anchored; its sha256 predated its own
  bytes). The fixture pair is now embedded in `tests/fixtures/` and a
  new gate in `tests/multistream_decode.rs` decodes the whole corpus
  through `MultistreamDecoder`: whole-stream ≥ 90 dB (measured
  ~100.8 dB) and every one of the six Vorbis-order channels ≥ 80 dB
  against the listing decode.

- **Hybrid (SILK + CELT) packet encode** (round 418).
  `HybridEncoderMono` (`hybrid_packet_encode`) produces code-0 Hybrid
  packets for configs 12-15 (SWB/FB x 10/20 ms): the WB-internal §4.2
  SILK layer (one frame per packet, §4.2.3 header bits + Table-5 frame
  through the existing analyzer/writers) and the §4.3 CELT layer
  (bands 17.., through the new CELT frame encoder) share one range
  coder, with the §4.5.1.1 Hybrid redundancy flag coded off under the
  decoder's exact 37-bit gate. The two layers are placed on ONE
  timeline: a 165-tap linear-phase 48→16 kHz decimator (82-sample
  group delay) plus the decoder's §4.2.9 WB resampler (35) and §4.2.8
  mono delay (3) exactly equal the CELT path's 120-sample MDCT-overlap
  delay — verified empirically (best-lag search returns 120).
  Validated through the crate's own decoder (HybridDecoded on every
  frame; 15-19 dB broadband-multitone SNR at 64-112 kb/s) and through
  the RFC 6716 §A reference-listing decoder, which reconstructs our
  hybrid streams identically to ours at **105-108 dB** (float-noise
  floor) across FB/SWB x 10/20 ms. The SILK layer has no rate control
  yet; a payload the SILK layer alone would overflow is rejected
  cleanly rather than emitted corrupt.

- **CELT-mode packet encode, end to end** (round 418). `CeltEncoder`
  (`celt_packet_encode`) produces code-0 CELT-only packets for the
  full configuration matrix — NB/WB/SWB/FB × 2.5/5/10/20 ms × mono +
  stereo — at any caller-chosen constant payload size, through the
  complete §5.3 stage sequence mirroring the §4.3 decoder symbol for
  symbol: silence flag, post-filter (signalled off), transient
  analysis + flag, two-pass intra/inter §5.3.2 coarse energy with the
  decoder-lockstep quantized `oldBandE` carry, budget-gated tf flags,
  spread, the dynalloc band-boost loop with the listing's
  band-peak offsets, trim analysis (stereo correlation + spectral
  tilt), the §4.3.3 allocation with coded skip / intensity /
  dual-stereo decisions (`celt_alloc_encode` — encode/decode
  roundtripped to identical `Allocation`s across the parameter
  surface), fine energy, the recursive §4.3.4 band encode
  (`celt_band_encode` — split angles from measured band energies via
  the step/uniform/triangular PDFs, intensity collapse, mono
  time-splits with the Hadamard/Haar reorganisation, PVQ leaves, the
  exact decode-side budget bookkeeping), the anti-collapse bit, and
  the §4.3.2.3 final fine backfill. New analysis front end
  (`celt_analysis`): §5.3 pre-emphasis (exact §4.3.7.2 inverse),
  forward MDCT (long + short interleaved blocks; > 120 dB analysis →
  synthesis round-trip at every LM), §4.3.2 band energies /
  log-energies / unit-norm shapes, and the listing's transient
  detector. Validated two ways on a 9-configuration matrix
  (mono/stereo, all four frame sizes, NB/SWB/FB, click-train
  transients, 16–256 kb/s): every packet decodes through the crate's
  own `OpusDecoder` (tone roundtrips 13–46 dB SNR by rate, monotone
  rate ladder), and the same packets decoded by the **RFC 6716 §A
  reference-listing decoder** (RFC 8251-patched, hash-verified
  extraction) agree with our decoder at **88–101 dB / max 1 LSB** —
  the two decoders reconstruct our streams identically to the
  float-noise floor.

- **CELT encoder primitives** (round 418): the §4.3.2.1 Laplace
  *encoder* (`ec_laplace_encode`, the exact write-side mirror of the
  decoder including the flat-tail value clamp), the §4.3.4.2 PVQ
  codeword-index construction (`celt_pvq_encode::encode_pvq_vector`,
  the exact inverse of the five-step index recovery — exhaustively
  roundtripped over small codebooks and sampled over the largest legal
  `V(N, K)` surfaces), the §5.3.4 PVQ pulse search (pyramid projection
  + greedy `Rxy²/Ryy` refinement) with the spreading-rotation
  `alg_quant` wrapper, and `RangeEncoder::finish_fixed` — the §5.1.5
  finalization into a **fixed-size** buffer where range bytes and
  §5.1.3 raw bits share exactly the frame's bytes (the CELT frame
  layout; the final partial raw window is OR-merged into the last
  byte, sharing the range value's free trailing bits when they meet).

- **SILK decode is now reference-exact** (round 415). The whole SILK
  reconstruction chain was moved from the RFC's real-number narrative
  formulas to the exact fixed-point arithmetic of the RFC 6716 §A
  embedded reference listing (extracted from the staged RFC text and
  hash-verified; RFC 8251 corrections applied where they touch the
  decoder), and validated as an instrumented black-box oracle:
  - **`silk_decode_core`** — the §4.2.7.9 core reconstruction in
    integer Q-domains: Q14 excitation (our §4.2.7.8.6 `e_Q23[]` is the
    reference `exc_Q14[] × 2^6`, verified sample-exact), §4.2.7.9.1
    LTP synthesis in Q13/Q15 with the output-history re-whitening and
    the per-subframe gain-change state rescaling
    (`silk_INVERSE32_varQ` / `silk_DIV32_varQ` transcriptions), and
    §4.2.7.9.2 LPC synthesis in Q14 with `SAT16(round(⋅×Gain_Q10≫8))`
    output. `synthesize_silk_frame` keeps its seam (its `f32` output
    is now an exact `xq/32768` image; `synthesize_silk_frame_i16`
    exposes the native samples), and the decoder's SILK domain is i16
    end-to-end. The side-channel reset after a mid-only run keeps the
    previous-subframe gain (only the full §4.5.2 reset re-arms it),
    matching the reference decoder.
  - **§4.2.9 resampler** — `SilkUpsampler` re-implemented as the
    reference decoder's fixed-point resampler (per-rate 1 ms delay
    compensation, 2× three-section allpass upsampling in Q10,
    fractional-phase 8-tap FIR interpolation in 10 ms batches, with
    the RFC 8251 §5 buffer-type correction), replacing the calibrated
    polyphase windowed-sinc stopgap. Verified sample-identical to the
    listing's resampler on synthetic vectors at all three rates. This
    also puts the Hybrid SILK band on the reference timeline with no
    calibration.
  - **§4.2.8 stereo unmix** — `stereo_ms_to_lr_i16`, the integer
    mid/side → left/right conversion (Q8/Q11/Q13 forms, 8 ms
    interpolation ramp, two-sample buffering with the built-in
    one-sample delay, carried side history applied even for mid-only
    intervals), replacing the `f32` formulation on the decode path.
  - Output conversion: the final float → i16 quantization now matches
    the §A listing's (`×32768`, saturate, ties-to-even) in both the
    SILK and CELT paths.

  Measured against an instrumented oracle built from the staged
  listing (plus the staged RFC 8251 patches): **every pure-SILK test
  stream decodes bit-exactly** — NB/MB/WB × 10/20/40/60 ms × mono and
  true mid/side stereo (including mid-only intervals) × 6–40 kb/s,
  101-packet streams of transient-heavy content — and the docs fixture
  corpus's SILK streams (`silk-nb`, `silk-mb-60ms`, `silk-wb-stereo`,
  `fec-on`, `silence-low-bitrate`, `code-1`, `code-3`) are bit-exact
  end-to-end at 48 kHz. Hybrid and CELT-bearing streams sit at the
  float-arithmetic noise floor (~71–111 dB). The former "NB ~19 dB /
  MB ~28 dB non-bit-exact-by-design reconstruction drift" ceilings
  were an artifact of the float realization and are gone.

- **Fixture references re-anchored to the reference listing.** The
  embedded `tests/fixtures/*.expected.wav` decodes were regenerated
  with the §A reference listing decoder (RFC 8251 patches applied) —
  the previous references came from a third-party black-box decoder
  whose non-normative §4.2.9 resampler (and pre-8251 hybrid folding)
  put them on a different timeline, capping the old gates at ~19–72 dB.
  Every waveform gate was raised accordingly: SILK fixtures to a
  100 dB floor (bit-exact in practice), CELT/packing fixtures to
  90–100 dB, the Hybrid fixture to 60 dB, and the mode-switching
  segments to 80–90 dB.

- Marked all internal SILK/CELT stage-plumbing modules (range coder,
  band shapes, PVQ, resampler internals, transition machinery) and
  their crate-root re-exports `#[doc(hidden)]`; the stable API surface
  is `Error`, `register`, and the packet/container-level modules
  (`decoder`, `toc`, `frames`, `framing`, `framing_self_delim`,
  `packet_compose`, `multistream`, `opus_head`). Attributes and
  comments only — no semantic changes.

- **§4.2.9 real resampler: `SilkUpsampler`**, a stateful streaming
  polyphase windowed-sinc upsampler (Kaiser-windowed, per-phase
  DC-normalized, per-(bandwidth × path) group delay, input history
  carried across Opus frames), replacing the stateless zero-delay
  linear interpolator on every SILK→48 kHz path (SILK-only mono +
  stereo, Hybrid, FEC recovery). The §4.2.9 filter is non-normative
  but its **delay** is what places SILK audio on the 48 kHz timeline:
  the design delays are calibrated black-box against the reference
  decodes of the staged fixture corpus (totals of 36/40/39 48 kHz
  samples for mono NB/MB/WB and 30/36/36 for stereo, the kernel
  half-width sitting exactly on the §4.2.9 causality cap
  `floor((1+d₄₈)/U)`). The measured mono−stereo asymmetry is exactly
  one input sample; the NB/MB stereo rows extrapolate the offset
  pinned at WB. Carried per-channel in `OpusDecoder` (mono state +
  stereo pair), history-cleared on the §4.5.2 SILK reset and on a
  channel-count change, rebuilt on a SILK bandwidth change.
- **FIX (§4.2.8): the mono one-sample delay was missing.** "In order
  to allow seamless switching between stereo and mono, mono streams
  must also impose the same one-sample delay" — the stereo unmixing
  formulas read `mid[i-1]` / `side[i-1]`, so stereo output leaves
  §4.2.8 one internal-rate sample late, and the mono path must match.
  The mono SILK output (SILK-only, Hybrid, and FEC recovery) is now
  delayed by one internal-rate sample, the trailing sample carried
  across Opus frames and zeroed on the §4.5.2 reset / bandwidth
  change ("zeros are used instead"). Found black-box: every mono
  SILK fixture sat one input sample early against its reference
  decode (and the Hybrid SILK band sat early against the bit-aligned
  CELT band).
- **FIX (§4.1.2.1): a Hybrid frame whose SILK layer legally overreads
  its bit budget was discarded whole.** Reading past the end of the
  frame is defined behaviour ("if no more input bytes remain, it uses
  zero bits instead"), and a real low-budget Hybrid frame can spend
  its entire budget on the SILK layer — the CELT layer must then take
  the §4.3 exhausted-budget silence path and the frame's audio is the
  SILK band alone. The decoder treated `tell() > budget` after the
  CELT layer as a whole-frame `CeltDecodeError` and zeroed the PCM,
  discarding perfectly good SILK audio (observed on the code-1
  fixture's 72-byte frames, whose reference decode keeps the SILK
  band). Only a real range-coder error zeroes the frame now.
- **§3.2 packing + CELT-pair waveform gates**
  (`tests/packing_fixture_decode.rs`), staging six more fixture pairs:
  `code-0-single-frame` (Hybrid/CELT mix, ~72.5 dB, gated > 55),
  `code-2-two-different-frames` (§3.2.4, measured bit-exact, > 60),
  `code-3-arbitrary-frames-with-padding` (§3.2.5 VBR + padding,
  ~37 dB, > 25), `pair-mono-48k-64kbps` (the corpus's only mono
  CELT-only stream, ~104.6 dB, > 60), `pair-stereo-48k-64kbps`
  (~111.4 dB, > 60), and the degenerate `code-1-two-equal-frames`
  (repacked mid-stream frames that overread into §4.1.2.1 zero-fill;
  black-box validators disagree with *each other* on it at the ~7 dB
  level, so its gate is structural + a loose floor).
- The `mode-switching` suite's Hybrid-segment energy-parity check is
  upgraded to real waveform gates: Hybrid segment > 50 dB (measured
  ~80+ dB through the delay-calibrated upsampler) and a whole-stream
  gate > 60 dB (measured **~82.2 dB** — up from a segment that
  previously could not be waveform-compared at all). `SilkUpsampler` /
  `SilkChannelPath` are re-exported at the crate root.
- **SILK waveform regression gates** (`tests/silk_reference_waveform.rs`):
  every SILK-bearing fixture stream is decoded end-to-end and compared
  pre-skip-trimmed against its shipped reference decode as an SNR
  floor — WB stereo > 55 dB, WB mono (fec-on main path) > 55 dB, MB
  60 ms > 22 dB, NB mono > 14 dB, silence-low-bitrate > 14 dB — with
  the WB gates sharp enough that one output sample of misalignment
  fails them (pinning the §4.2.9 delay calibration and the §4.2.8
  mono delay). The hybrid fixture's loose energy-parity check is
  replaced by a real 30 dB whole-stream SNR gate (measured ~37.5 dB)
  that pins the Hybrid SILK↔CELT alignment. New fixture pairs staged
  from the corpus: the three SILK `expected.wav` references,
  `fec-on.expected.wav`, and the `silence-low-bitrate` stream + its
  reference (near-DTX 6-byte packets, LCG comfort-noise excitation).
- Fixture-corpus waveform validation (pre-skip-trimmed SNR vs the
  shipped reference decodes): `silk-wb-stereo-20kbps` −4.7 → 68.7 dB,
  `fec-on` (WB mono main path) −5.1 → 71.9 dB, `hybrid-fb-mono-28kbps`
  1.4 → 37.5 dB (the SILK band now lands on the CELT layer's
  timeline), `silk-mb-60ms-mono-20kbps` −3.1 → 27.7 dB,
  `silk-nb-mono-16kbps` −4.6 → 18.5 dB, `silence-low-bitrate` −4.6 →
  18.7 dB. The NB/MB ceilings are reconstruction-arithmetic drift, not
  alignment: §4.2.7.9 explicitly frees the reconstruction from
  bit-exactness ("small errors should only introduce proportionally
  small distortions") and the fixture references were produced by a
  fixed-point reconstruction whose rounding recirculates through the
  LTP feedback on strongly periodic content.

- §4.5 mode-switching integration suite (`tests/mode_switching_decode.rs`)
  on the embedded `mode-switching` fixture (Hybrid→CELT-only switch
  with encoder-emitted §4.5.1 redundancy): whole-stream decode with
  per-mode status assertions, Hybrid-segment energy parity, and
  waveform-level thresholds on the transition window (~109 dB measured)
  and the CELT-only segment (~107 dB) — a wrong §4.5.2 reset placement
  or a skipped redundant frame desynchronizes the CELT segment's
  inter-coded energy and fails the suite
- FIX: `CeltEnergyState::new` initialised the §4.3.5 anti-collapse
  energy references (`old_log_e` / `old_log_e2`) to zero; a fresh
  stream (or a §4.5.2 reset) has no previous frame energy, so they
  must start at the −28 log2 energy floor — with a zero init the
  first transient frame after stream start / reset computed its
  anti-collapse thresholds as if every band had held unit energy and
  injected the wrong noise, and the error rang through the overlap /
  post-filter / de-emphasis state for several frames. Found black-box:
  low-rate CELT-only streams whose first frame is transient decoded at
  ~30 dB startup SNR against the reference; with the floor init the
  same streams reach >99 dB (a 16 kb/s 20 ms tone stream improves
  53.4 → 106.7 dB whole-stream, a noise stream 45.4 → 99.1 dB), and
  the staged fixture corpus is unchanged
- **FIX (wire-format): the §3.2.5 code-3 frame-count byte was parsed
  and written bit-reversed.** RFC 6716 Figure 5 is MSB-first — `v`
  (VBR) is 0x80, `p` (padding) is 0x40, `M` is the low six bits — but
  both §3.2 parsers (`OpusPacket::parse`, `parse_self_delimited`) and
  all code-3 writers (`compose_packet_code3`, the code-3 arm of
  `compose_self_delimited`, `pad_packet_to`) used `v` = 0x01, `p` =
  0x02, `M` = the HIGH six bits. Because writer and parser agreed with
  each other, every in-crate roundtrip test passed while **every
  code-3 packet from a real encoder was rejected as malformed** (a
  black-box CBR VoIP stream lost 80 of 101 packets; a CBR-padded
  packet's byte 0x41 read as "VBR, M=16") and every packet we emitted
  was undecodable elsewhere. Found by a black-box stress corpus;
  wire-layout bytes are now pinned literally on both the parse side
  and the write side, precisely because roundtrip testing is blind to
  a consistently reversed layout
- **§4.5 transition machinery: redundant-frame synthesis + cross-lap
  + reset placement.** The 5 ms redundant CELT frame (§4.5.1) is now
  *decoded and mixed*, not just skipped over:
  - SILK-only frames run the §4.5.1.1 implicit check (≥ 17 bits
    remaining after the SILK layer) that was previously absent, then
    the Table-65 position and the whole-bytes-remaining size;
  - Hybrid frames now **shrink the main coder's buffer**
    (`RangeDecoder::shrink_buffer`, new) by the §4.5.1.3 size so the
    MDCT layer's raw bits read from the end of the reduced buffer as
    the spec requires (previously only the bit budget shrank — any
    redundancy-bearing hybrid frame with raw bits misread them);
  - the redundant frame decodes "like any other CELT-only frame"
    (§4.5.1.4: own coder, no TOC, 5 ms, carrier's channels, carrier
    bandwidth with MB→WB) through the stream's one CELT state, whose
    geometry now *adapts without dropping state*
    (`CeltSynthesis::set_geometry`, new — §4.5 keeps state across
    frame-size/channel changes; resets happen only where §4.5.2 puts
    them);
  - §4.5.2 placement per Figure 18: an end-position redundant frame
    takes the CELT reset itself (`!R`) and its warmed state carries
    into the following CELT-only/Hybrid frames (the packet-boundary
    reset is suppressed via the carried §4.5.1 decision — rule 3, now
    also applied to SILK→Hybrid per the `;H` row); a
    beginning-position one decodes on the un-reset state (rule 4)
    before the deferred main-layer reset (`|H`);
  - §4.5.1.4 output stitching: beginning-position — first 2.5 ms of
    the redundant output as-is (main output discarded), second 2.5 ms
    cross-lapped; end-position — the redundant frame's second half
    cross-lapped with the end of the main signal; both under the
    power-complementary CELT MDCT window.
  On the staged mode-switching fixture the Hybrid→CELT transition
  window improves from 105.6 dB to 108.8 dB against the reference
  decode (the CELT-only segment stays reference-exact at ~107 dB);
  black-box A/B across a switching stress corpus shows no regressions
- **Packet-loss concealment (RFC 6716 §4.4)** — `OpusDecoder::conceal_loss`
  fills a lost packet with real extrapolated audio instead of silence,
  following the §4.4 per-mode guidance: after a SILK-only / Hybrid
  frame, an LPC extrapolation of the previous output (Burg fit over
  the trailing history, bandwidth-expanded for guaranteed decay,
  driven by the pitch-cyclic LPC residual — the long-term + short-term
  predictor continuation); after a CELT-only frame, a pitch-periodic
  waveform repetition (normalized-autocorrelation pitch search over
  the §4.3.7.1 period range 15..=1022, cyclic repeat of the final
  period). Consecutive losses decay in energy to the silence floor,
  and each concealment carries a 2.5 ms extrapolation tail that is
  cross-lapped (power-complementary window pair, the §4.5.1.4 seam
  shape) into the head of the first packet decoded after the loss run
  — both joins are smooth. FEC recovery
  (`decode_packet_fec`) now also feeds the concealment history, so a
  loss immediately after a recovered frame extrapolates from the
  recovered signal. New `plc` module (13 unit tests) + a
  `FrameDecodeStatus::Concealed` outcome + a 5-test `plc_decode`
  integration suite driving dropped-packet patterns on the real SILK
  and CELT fixtures (single-loss continuity at both joins bounded by
  the neighbourhood's natural sample-to-sample step, non-silent
  single-loss energy parity, 30-loss burst monotone decay to the
  floor with clean post-burst recovery, duration tracking, and 440 Hz
  tonal-continuation dominance on the sine fixture)
- FIX: `celt_frame_prefix` decoded the §4.3.7.1 post-filter octave as
  `ec_dec_uint(7)` (values 0..=6); Table 56 codes it `uniform (6)`, i.e.
  `ec_dec_uint(6)` over 0..=5 — the §4.3.7.1 period bound ("between 15
  and 1022, inclusively") only holds for octave <= 5. Any caller
  decoding a post-filter-enabled frame through this helper consumed the
  wrong interval and desynchronized every following symbol (the live
  frame-decode path already read the correct 6-value symbol; the helper
  is the exported API surface). The octave sweep test now pins the
  0..=5 range, the reachability of octave 5, and the §4.3.7.1 period
  bound across the whole sweep
- FIX: `celt_band_boost::decode_band_boosts` gated the §4.3.3 boost
  loop against a constant raw-frame budget (`total_bits + total_boost`,
  which the section's own updates hold constant) and computed the boost
  quanta from per-channel bin counts. The gate is the **shrinking**
  budget — the frame in 1/8 bits minus every boost committed so far —
  per the §4.3.3 intent clause ("enough room to obey the boost and …
  to code the boost symbol"), the §4.3.3 trim gate immediately after
  ("total frame size in 8th bits minus total_boost"), and a black-box
  discrimination experiment: across a 48-stream low-bitrate CELT-only
  stress corpus, three streams decode boost bits inside the divergence
  window and only the shrinking gate stays aligned with the reference
  decode (10–16 dB vs 35–39 dB SNR). The quanta `N` counts the band's
  MDCT bins across all coded channels (`C * (band_width << LM)`). The
  live CELT frame decode now **calls** the module (previously it carried
  its own inline copy of the loop; the fixed helper is byte-identical
  on the whole stress corpus + fixtures), and a range-encoder-crafted
  regression pins the shrinking-gate discriminator
- FIX: `celt_pulse_cache` indexed the §4.3.4.1 cost cache band-major
  (`band * 5 + LM`); the 105-entry index is LM-major
  (`(LM + 1) * 21 + band`, five rows of 21 bands — pulse-cache format
  trace §2.1, corrected). Every non-trivial `(band, LM)` lookup
  resolved to the wrong run. Accessors now use the LM-major mapping and
  a data-internal test pins the trace-§2.3 property that each of the 23
  runs serves exactly one effective band size `N`
- `celt_rate_alloc`: the §4.3.3 implicit allocation, decode side —
  the interpolated bits-to-pulses computation over the Table 57 static
  matrix (coarse quality search, 6-step fine bisection, backward
  band-skip decode, intensity / dual-stereo symbol decode, per-band
  fine-energy split with priority flags and balance carry), plus the
  §4.3.4.1 `bits2pulses` / `pulses2bits` / `get_pulses` cost-cache
  accessors and the `logN400` table; exact-integer 1/8-bit arithmetic
  throughout
- **Hybrid frames decode end-to-end to real PCM**
  (`FrameDecodeStatus::HybridDecoded`): the §4.2 SILK layer (WB
  internal) and the §4.3 CELT layer (bands 17–21) share one range
  coder, with the §4.5.1 redundancy side information decoded between
  them (a present redundant frame shrinks the CELT budget; its own
  5 ms synthesis/cross-lap is a pending refinement) and the two 48 kHz
  outputs summed per §4.4; validated on the hybrid-fb-mono-28kbps
  fixture (energy parity with the reference decode; the non-normative
  SILK resampler bounds waveform alignment)
- **CELT-only frames decode end-to-end to real PCM**
  (`FrameDecodeStatus::CeltDecoded`): `celt_frame_decode` sequences the
  whole Table-56 entropy layer (silence / post-filter / transient /
  intra flags with the exact budget gates, coarse energy with its
  budget fallbacks, TF flags, spread, dynalloc boosts, trim, the
  implicit allocation, fine energy, PVQ band shapes, the anti-collapse
  bit, and the §4.3.2.3 final-bit backfill) and rolls the cross-frame
  energy history; `celt_mdct_synthesis` runs the signal half
  (denormalisation, long/short-block inverse MDCT + overlap-add, the
  §4.3.7.1 pitch post-filter with crossfaded parameter transitions,
  §4.3.7.2 de-emphasis) with all carried state. Validated against the
  reference decodes of the fixture corpus: `celt-fb-stereo-128kbps`
  and `celt-2.5ms-low-latency` both reconstruct at ~100 dB SNR
  (arithmetic-precision agreement); §4.5.2 CELT resets are applied on
  mode transitions. The narrower `CeltCoarseEnergyDecoded` /
  `CeltAllocationDecoded` statuses are superseded and removed
- `celt_band_decode`: the §4.3.4 recursive band decode — PVQ leaf
  decode with the exact spreading rotation (two-stride lattice), band
  splitting with entropy-coded split angles (triangular / uniform /
  step PDFs, bit-exact `bitexact_cos` / `bitexact_log2tan` mid-side
  weighting), stereo mid/side merge + N=2 orthogonal special case +
  intensity/inversion, dual-stereo routing, time/frequency Haar
  recombination with Hadamard block (de)interleaving, spectral folding
  with the RFC 8251 §9 hybrid-folding update, per-short-block collapse
  masks, and the §4.3.5 anti-collapse noise injection
- `RangeDecoder::range_size()`: read-only accessor for the §4.1 range
  state `rng`, the carried folding-noise LCG seed of the CELT layer
- analysed 10 ms packets: `with_packet_duration(bw, 100)` on both encoders
  emits one 2-subframe SILK frame per packet (§4.2.7.5.5 factor not
  stored), completing the encoder's SILK frame-size matrix (10/20/40/60 ms
  x NB/MB/WB x mono/stereo, FEC and CBR shaping included)
- CBR / padding on encode: `pad_packet_to` re-frames a code-0 packet as a
  §3.2.5 code-3 packet padded to an exact byte size (every target size
  reachable, non-minimal padding chains at the 254-byte boundaries) and
  `SilkEncoderMono/Stereo::encode_packet_cbr` emit constant-size packets
  that decode identically to the unpadded stream
- LBRR (in-band FEC) from PCM: `SilkEncoderMono::set_fec` /
  `SilkEncoderStereo::set_fec` — each packet carries a reduced-rate §4.2.5
  re-encode of the PREVIOUS packet's active intervals, analysed from a
  pre-packet analyzer snapshot with a fresh closed-loop state (mirroring
  the FEC decoder's fresh synthesis) and coded unvoiced (LTP has no history
  in the fresh-state LBRR sequence); inactive intervals leave §4.2.4 gaps;
  stereo LBRR reuses the regular pass's downmix + §4.2.7.1 weights;
  recovery verified through `decode_packet_fec` on dropped packets
- packet writers now enforce §3.2 R2 (compressed frame <= 1275 bytes)
- analysed stereo SILK encoder: 40 / 60 ms multi-frame stereo packets
  (`SilkEncoderStereo::with_packet_duration`) — per 20 ms interval the
  §4.2.7.1 weights are re-estimated, the §4.2.8 exact downmix re-run, and
  the §4.2.7.2 mid-only decision re-taken (coded-side, inactive-side
  `Some(false)` flag, and mid-only intervals can mix inside one packet),
  matching the decoder's per-interval unmix walk
- analysed SILK encoder: 40 / 60 ms multi-frame packets from PCM
  (`SilkEncoderMono::with_packet_duration`) — one analysed 20 ms SILK frame
  per §4.2.2 interval with the intra-packet carried state threaded like the
  decoder's regular walk (delta-coded first gains, §4.2.7.6.1 relative lags
  after a voiced frame, §4.2.7.6.3 scaling on the first frame only), plus
  signal-derived per-frame §4.2.3 VAD flags (silent intervals code frame
  type 0 / Inactive and skip the pitch search)
- streaming decoder: carry the §4.2.7.4 gain-clamp base (`previous_log_gain`)
  and the §4.2.7.5.5 NLSF interpolation base `n0` across Opus frames — the
  independent first-subframe gain of a packet is now clamped against the
  previous packet's last gain, and a coded `w_Q2 < 4` on a packet's first
  20 ms frame interpolates against the previous packet's decoded NLSFs
  (both were previously reset per packet); FEC recovery seeds the bases
  from the LBRR reconstruction under the §4.2.7.4 packet-loss latitude

## [0.0.13](https://github.com/OxideAV/oxideav-opus/compare/v0.0.12...v0.0.13) - 2026-07-03

### Other

- document the stereo SILK packet-encode subsystem + framing / RFC 7845 write side
- §4.2.7.1 stereo-weight estimator (estimate_stereo_weights)
- RFC 7845 write side — OpusHead::compose + assemble_multistream_packet
- §3.2 + Appendix-B packet-framing writers (packet_compose)
- encode-side stereo downmix stereo_lr_to_ms — exact §4.2.8 algebraic inverse
- §4.2.7.1 stereo-weight quantizer (StereoWeightSymbols::quantize)
- stereo SILK-only packet encoder with §4.2.2 mid/side interleave + stereo LBRR emission
- LBRR emission landed; update encoder-scope tail
- LBRR (in-band FEC) emission in the packet encoder, closing the FEC loop
- document the §5.1 range encoder + SILK encode-side subsystem and the round's decoder hardening
- SILK-only mono packet encoder, decoder-verified end-to-end + 10ms-MB synthesis fix
- encode_silk_frame — whole-frame Table-5 write-side composition
- encode-side symbol writers for the full §4.2.7 frame stack + Table 47-50 corrections
- fix two fuzz-found crashes (dec_bits(32) shift overflow, §4.2.7.5.8 i64 overflow)
- restore cargo-fuzz harness suite (4 targets) for the scheduled Fuzz workflow
- RFC 6716 §5.1 range encoder, roundtrip-exact against the §4.1 decoder
- truncation-safety test across the full CELT-only decode path
- decode §4.3.1 tf_change/tf_select + §4.3.4.3 spread before §4.3.3 allocation (Table-56 order)
- wire §4.3.3 allocation header (boost/trim/reservations) into CELT-only decode
- record §4.3.4.5 + §4.3.4 modules; sharpen the §4.3.3 docs gap
- §4.3.4 per-band shape decode orchestrator (celt_band_shape)
- §4.3.4.5 Hadamard time-frequency transform (celt_tf_hadamard)
- drop generic enumerated-denial sentences from module docs
- README + CHANGELOG — document the RFC 7845 multistream subsystem
- multistream coupled-stream (stereo) decode tests on real PCM
- enforce §3 equal-duration constraint across multistream streams
- §5.1 output-gain application + pre-skip accumulator helpers
- MultistreamDecoder — multichannel decode + §5.1.1 channel map
- factor a shared decode body + self-delimited packet entry point
- multistream packet split (RFC 7845 §3)
- OpusHead identification-header parser (RFC 7845 §5.1/§5.1.1)
- *(opus)* neutralize black-box-validator product naming in FEC fixture prose
- cover stereo in-band FEC routing path (fec_decode.rs)
- SILK in-band FEC (LBRR) recovery — decode_packet_fec (§2.1.7/§4.2.5)
- neutralize residual line-wrapped validator name in SILK test module-doc
- neutralize black-box-validator product naming in r362 SILK-fixture prose
- embed SILK fixtures in-crate so the decode suite runs in standalone CI
- first end-to-end SILK fixture-decode validation suite
- neutralise reference-source filename in celt_e_prob_model doc comments
- neutralise reference-source naming in coarse-energy clamp doc-gap note
- README — coarse-energy front half now decodes for non-silent CELT
- wire CELT non-silent coarse-energy decode into the decoder
- §4.3.2.1 CELT coarse-energy reconstruction recurrence
- fix CELT post-filter octave to ec_dec_uint(7) (7 values 0..=6)
- README + CHANGELOG — CELT silence-frame end-to-end milestone
- wire CELT-only silence frame end-to-end through synthesis backend
- CELT frame-prefix symbol decode (Table 56 head)
- CELT §4.3.2.1 Laplace symbol decode (ec_laplace_decode)
- CELT synthesis-backend integration suite + README/CHANGELOG
- frame-level interleaved-i16 CELT synthesis output (celt_synthesis)
- §4.3.6→§4.3.7.2 CELT synthesis backend composition (celt_synthesis)
- README — stereo SILK-only now decodes end-to-end to interleaved PCM
- run §4.2.8 stereo unmix per SILK interval, not per Opus frame
- wire stereo SILK-only decode to interleaved PCM (§4.2.2/§4.2.7.2/§4.2.8)
- opus §4.2.7.1/§4.2.7.2: stereo mid-channel header decode in decode_silk_frame
- README — mono SILK-only now decodes end-to-end to real PCM
- opus §4.5.2: cross-packet SILK state reset on CELT->SILK transition
- opus §4: wire SILK synthesis into decoder — mono SILK-only → real PCM
- opus §4.2.7.9: SILK frame synthesis composition (silk_synthesis)
- README — document packet→PCM orchestration + mono SILK-only decode path
- §4.2.6/§4.2.7 in-order SILK frame decode + wire mono SILK-only packet path
- §3/§4 top-level packet→PCM orchestration (OpusDecoder::decode_packet)
- §4.3.4.5 time-frequency change decode (tf_change loop + gated tf_select)
- §4.3.7 weighted overlap-add (celt_overlap_add)
- CELT §4.3.7 inverse MDCT transform core (celt_imdct)
- CELT §4.3.6 band denormalisation (celt_denormalise)
- CELT §4.3.4.1 Bits-to-Pulses pulse-cost cache (round 49)
- refresh to current status, drop per-round changelog cruft

### Added

- *§4.2.7.1 stereo-weight **estimator** —
  `silk_stereo::estimate_stereo_weights`.* The encoder-analysis
  companion to the quantizer and downmix: a least-squares fit of the
  raw side signal `(L-R)/2` onto the two §4.2.8 predictor terms (the
  low-passed mid `p0` and the mid itself), solving the 2×2 normal
  equations in f64 over the frame with the same `p0` boundary terms
  the downmix uses, returning zero weights for a (near-)singular
  system (silent mid). RFC 6716 leaves the encoder's weight choice
  free — only the decode of the coded quintuple is normative — so
  the estimator's output is a *target* pair fed through
  `StereoWeightSymbols::quantize`. Verified: a side channel built as
  exactly `w0*p0 + w1*mid` estimates back to the planted Q13 pair
  within ±1, and the full analysis chain (raw L/R → estimate →
  quantize → `stereo_lr_to_ms`) codes a side channel with markedly
  less energy than the unpredicted `(L-R)/2` on correlated stereo
  while the §4.2.8 roundtrip identity still holds.

- *RFC 7845 write side — `OpusHead::compose` (§5.1 / §5.1.1) and
  `multistream::assemble_multistream_packet` (§3).* `compose` is the
  exact write-side mirror of `OpusHead::parse`: it validates every
  MUST the parser enforces (version nibble, non-zero channel count,
  per-family channel ranges, non-zero stream count, `M ≤ N`,
  `M + N ≤ 255`, mapping-table length and index bounds; a family-0
  header must hold the RFC-pinned synthesized default table since the
  wire format omits it) and re-emits parsed headers byte-identically
  — including the real fixture's family-0 header. The multistream
  assembler packs `N` regular per-stream packets per §3: the first
  `N − 1` re-framed through the new Appendix-B writer (code-3
  padding preserved, CBR/VBR chosen from the parsed frame lengths),
  the last appended verbatim, with the §3 equal-duration constraint
  (same TOC config + frame count) enforced across inputs. Verified:
  assemble→split roundtrips over mixed code-2/code-3 streams,
  byte-identity with the integration suite's hand-built self-delimited
  construction, and a whole-fixture write→read sweep where every
  assembled 2-stream packet decodes through `MultistreamDecoder`
  sample-identically to a plain stateful decode.

- *§3.2 / Appendix-B packet-framing **writers** —
  `packet_compose::{compose_packet, compose_packet_code3,
  compose_self_delimited, encode_length}`.* The write-side mirror of
  the §3.2 parser and the Appendix-B self-delimited parser: the
  §3.2.1 one/two-byte length writer (roundtripped against the shared
  decoder for every legal length 0..=1275), code 0/1/2 framing,
  code-3 CBR/VBR framing with the §3.2.5 frame-count byte and the
  255-chained padding-length header (CBR auto-selected for uniform
  lengths, VBR/padding forceable), and the Appendix-B Figures 25-29
  self-delimited variants for chaining streams inside a multistream
  payload. Every parser-enforced requirement is validated before
  writing (R2 frame-length cap, R3 code-1 equality, R5 120 ms
  duration bound, M ∈ 1..=48, R6 CBR uniformity). Verified: 200
  random compose→parse roundtrips across all four codes, VBR/padding
  roundtrips including chained padding lengths, self-delimited
  roundtrips with exact byte consumption plus a three-packet
  back-to-back chain, a shape-violation rejection audit, and an
  end-to-end §4.5 check where two independently encoded 20 ms SILK
  frame bodies packed as code-1 / code-2 / code-3-VBR packets decode
  through `OpusDecoder::decode_packet` as two real SILK frames. The
  stereo packet-encode subsystem's public surface (stereo packet
  encoders, downmix, framing writers) is now re-exported at the crate
  root.

- *Encode-side stereo downmix — `silk_stereo::stereo_lr_to_ms`, the
  exact algebraic inverse of the §4.2.8 unmixer.* Solving the §4.2.8
  reconstruction for `mid` and `side` gives the frame-aligned inverse
  `mid[k] = (left[k]+right[k])/2`, `side[k] = (left[k]-right[k])/2 -
  w1(k+1)*mid[k] - w0(k+1)*p0(k+1)` with the same interpolation ramp
  the decoder applies (the last side sample is consumed by the *next*
  frame's first reconstruction, whose ramp start equals this frame's
  final weights — legal because every SILK frame is longer than the
  8 ms interpolation phase). One sample of L/R lookahead
  (`next_lr`) feeds the final `p0`; a stream-end hold is provided.
  Cross-frame history lives in the new `StereoDownmixState` (trailing
  mid sample + previous weights), mirroring `StereoUnmixState`'s
  reset semantics. Verified: a 3-frame NB/MB/WB roundtrip through
  `stereo_ms_to_lr` with per-frame weight changes reproduces the
  input delayed by exactly one sample (the §4.2.8 delay) to 1e-4,
  mid is exactly `(L+R)/2`, a single-frame roundtrip is lookahead-
  independent, and validation mirrors the unmixer.

- *§4.2.7.1 stereo-weight quantizer —
  `StereoWeightSymbols::quantize`.* The deterministic write-side
  inverse of the shared `weights()` reconstruction: an exhaustive
  argmin of the squared Q13 error over all 5625 codebook quintuples
  (25 stage-1 × 3×5 × 3×5 stage-2/3 combinations), with lexicographic
  first-wins tie-breaking. Completes the stereo encode-side symbol
  story: an encoder now maps a target `(w0_Q13, w1_Q13)` pair to the
  quintuple the mid frame carries. Verified: representable targets
  roundtrip value-exactly across the codebook, random targets are
  true argmins against a full codebook scan, and out-of-range targets
  saturate to the extreme reachable weights (derived from the
  codebook, not hard-coded).

- *Stereo SILK-only Opus **packet** encoder —
  `silk_packet_encode::encode_silk_only_packet_stereo` (+
  `_with_lbrr`), the §4.2.2 mid/side interleave writer.* Per 20 ms
  interval the mid SILK frame is written (carrying the §4.2.7.1
  stereo-prediction-weight quintuple and, when the interval's side
  channel is not active, the §4.2.7.2 mid-only flag), then the side
  SILK frame when coded, with two independent per-channel carried
  states (previous gain / lag / NLSF) threaded exactly the way the
  packet decoder's stereo walk threads them. All three side-coding
  shapes are supported and validated: an active side frame (side VAD
  set, no mid-only flag), an inactive coded side frame (mid-only flag
  present and cleared), and a skipped side frame (flag set; the side
  carried state is left untouched, mirroring the decoder). §4.2.5
  stereo LBRR emission interleaves optional mid / side redundancy
  frames ahead of the regular frames (mid LBRR carries weights; the
  mid-only flag on a mid LBRR frame is present iff the interval has
  no side LBRR frame and must then be set; side-only LBRR intervals
  are legal), closing the stereo FEC loop. Validated end-to-end: 120
  random stereo packets across every bandwidth × duration ×
  side-pattern mix decode through a fresh `OpusDecoder::decode_packet`
  to real stereo SILK PCM (`SilkStereoDecoded`, exact §3 interleaved
  sample counts) with a parallel mid/side Table-5 walk reconstructing
  both channels' predictions field-for-field, and 60 LBRR-bearing
  packets keep their regular decode aligned while `decode_packet_fec`
  recovers real two-channel audio (`Recovered`; an all-side-only
  redundancy set reports `NoLbrr` per the decoder's mid-anchored FEC
  policy). Inconsistent scripts (missing weights, mid-only flag
  mismatches, weights on a side frame, inactive LBRR) are rejected.

### Fixed

- *Two decoder crashes found by the restored fuzz suite's first CI
  run (both reproduced as in-tree regression tests).* (1) §4.1.4 raw
  bits: `RangeDecoder::dec_bits(32)` overflowed its u32 window's
  shifts (both the refill concatenation and the consume shift); the
  working window is now u64, making the documented full-32-bit read
  well-defined (`roundtrip_raw_bits_32_wide`). (2) §4.2.7.5.8
  prediction-gain recurrence: adversarial LSF coefficients whose Q24
  widening escapes the spec's "all intermediate results fit in 32
  bits" envelope overflowed the recurrence's i64 products; the
  recurrence now classifies any value escaping the 32-bit envelope as
  unstable (triggering the §4.2.7.5.8 bandwidth-expansion round)
  instead of overflowing — checked at row initialization, at
  `num_Q24`, and at the next-row store. The verbatim 18-byte crash
  artifact is pinned in `tests/malformed_input.rs`.

- *Five mis-transcribed rows in the §4.2.7.8.3 pulse-count split
  tables, surfaced by differential encoder/decoder testing:* Table 47
  rows 9 and 11, Table 48 row 9, and Table 49 rows 7 and 16 all had
  wrong tail cells (Table 49 row 16's final cell read 0 instead of 1,
  making a fully-left-concentrated 16-pulse split undecodable and
  hanging the new encoder). All 64 rows of Tables 47-50 are now
  verified cell-by-cell against the RFC 6716 PDF text by a single
  exhaustive test, replacing the previous spot-checks that had missed
  the bad rows.

### Fixed

- *10 ms MB SILK frames failed to synthesize.* §4.2.7.8 codes a 10 ms
  MB frame as 8 shell blocks (128 excitation samples) of which only
  the first 120 are used; `synthesize_silk_frame` rejected the longer
  parsed excitation outright, so every 10 ms MB SILK packet decoded to
  `SilkDecodeError` silence. The synthesis now uses the frame-length
  prefix and discards the parsed tail, per the §4.2.7.8 preamble.
  Surfaced by the new packet-encoder end-to-end sweep.

### Added

- *LBRR (in-band FEC) emission —
  `encode_silk_only_packet_mono_with_lbrr` (§2.1.7 / §4.2.5).* The
  packet encoder can now carry per-interval redundancy scripts: the
  §4.2.3 / §4.2.4 LBRR flags are derived from which intervals carry
  one, and the LBRR frames are written ahead of the regular frames
  with their own independent carried state, exactly mirroring the
  decode-side LBRR walk (active-coded per §4.2.7.3, first-coded-frame
  independence and LTP-scaling rules enforced). This closes the FEC
  loop end-to-end: 60 random LBRR-bearing packets across every
  bandwidth × duration still decode their regular frames to real SILK
  PCM (pinning the range-coder alignment across the LBRR bits), and
  `decode_packet_fec` recovers real audio from the emitted redundancy
  (`FecDecodeStatus::Recovered`, exact §3 sample counts), while a
  redundancy-free packet reports `NoLbrr`.

- *SILK-only mono Opus **packet** encoder —
  `silk_packet_encode::encode_silk_only_packet_mono` (§3.1 / §4.2.2-
  §4.2.6).* Produces complete decoder-ready packets: the §3.1 TOC
  byte (via the new `OpusTocByte::compose_byte`, the Table-2 inverse
  of `from_byte`), the §4.2.3 / §4.2.4 header bits (via the new
  `SilkHeaderBits::encode` write-side mirror: per-frame VAD bits
  derived from each frame type, LBRR off, Table-4 per-frame LBRR
  validation), and 1-3 regular SILK frames (10/20/40/60 ms) written
  in Table-5 order with the carried state (previous gain / lag /
  NLSF) threaded exactly the way the packet decoder threads it.
  Validated end-to-end: 120 random packets across every bandwidth ×
  duration decode through a fresh `OpusDecoder::decode_packet` to
  real SILK PCM (`SilkParamsDecoded`, exact §3 sample counts), and a
  parallel `decode_silk_frame` walk reconstructs the encoder's
  per-frame `SilkFrameDecoded` predictions field-for-field.

- *`encode_silk_frame` — the whole-frame Table-5 composition
  (§4.2.6 / §4.2.7), the write-side mirror of `decode_silk_frame`.*
  Consumes a `SilkFrameSymbols` script (header / gains / LSF stage-1
  + stage-2 / interpolation index / LTP / seed / excitation), writes
  every symbol in exact Table-5 order through the per-stage encoders,
  runs the same non-bitstream §4.2.7.5.3-§4.2.7.5.8 LSF → LPC chain
  as the decoder (the §4.2.7.5.5 interpolation tail factored into
  `LsfInterpolated::from_decoded_index` and shared by both
  directions), and returns the `SilkFrameDecoded` the decoder will
  reconstruct — so an encoder can carry cross-frame state
  (`last_log_gain`, `nlsf_q15`, `primary_lag`) exactly as a decoder
  would. Validated by the capstone roundtrip: 250 random full-frame
  scripts across every bandwidth / frame size / signal type /
  stereo-mid context / carried-state combination encode and decode
  back to a field-for-field identical `SilkFrameDecoded`, including
  the derived stable Q12 LPC filters and the LCG-reconstructed Q23
  excitation. LTP-presence/frame-type mismatches are rejected.

- *SILK encode-side symbol writers for the complete §4.2.7 regular
  frame stack — the write-side mirrors of every SILK decode stage.*
  Each encoder shares the decode module's PDF tables and normative
  reconstruction formulas, and returns the value the decoder will
  reconstruct so the caller can carry quantization feedback:
  - `SilkFrameHeader::encode_pre_gains` / `encode_lsf_stage1`
    (§4.2.7.1 stereo-weight quintuple via `StereoWeightSymbols`, the
    §4.2.7.2 mid-only flag, the §4.2.7.3 frame type from the
    kind-selected Table 9 PDF, and the §4.2.7.5.1 Table 14 index),
    with the §4.2.7.1 weight reconstruction extracted into
    `StereoWeightSymbols::weights` and shared by both directions.
  - `SubframeGains::encode` (§4.2.7.4: independent MSB+LSB pair or
    Table 13 delta per `GainSymbol`) plus `SubframeGains::quantize`,
    a deterministic locally-optimal gain-plan quantizer (argmin over
    the 41 delta symbols with the decode-side clamp reproduced).
  - `LsfStage2::encode` (§4.2.7.5.2: per-coefficient base symbol from
    the I1-selected Table 15/16 codebook, the forced Table 19
    extension for `|I2| >= 4`), with the backwards-prediction inverse
    shared via `compute_res_q10`.
  - `LsfInterpolated::encode_index` (§4.2.7.5.5) and
    `encode_lcg_seed` (§4.2.7.7).
  - `LtpParameters::encode` (§4.2.7.6: absolute / relative /
    relative-fallback primary lag via `LagSymbols`, Table 32 contour,
    periodicity + per-subframe filter indices, Table 42 LTP scaling)
    consuming an `LtpSymbols` script.
  - `Excitation::encode` (§4.2.7.8: Table 45 rate level, the
    17-escape LSB-depth chain over Table 46, the §4.2.7.8.3 preorder
    split tree, §4.2.7.8.4 MSB-first LSB refinement bits, §4.2.7.8.5
    signs) consuming an `ExcitationSymbols` script (the quantized
    signed `e_raw[]` plus per-block LSB depths), with the §4.2.7.8.6
    LCG reconstruction shared via `reconstruct_e_q23`.
  Every stage is validated by seeded randomized encode → decode
  roundtrips against the real range coder (500-800 scripts per stage
  covering all bandwidths / frame sizes / signal types) plus
  bad-input rejection tests; the range encoder itself gained a
  zero-probability-cell debug assertion.


- *Cargo-fuzz harness suite restored (`fuzz/`).* The scheduled Fuzz
  workflow had been failing since the orphan rebuild dropped the old
  `fuzz/` directory while the workflow survived. Four coverage-guided
  libfuzzer targets now back it: `decode_packet` (the top-level
  packet → PCM path fed multiple consecutive carved packets on one
  stateful decoder, plus the Appendix-B self-delimited entry),
  `decode_packet_fec` (the §4.2.5 LBRR recovery entry interleaved with
  regular decodes), `multistream_decode` (RFC 7845 §5.1 OpusHead parse
  + §3 packet split + multichannel assembly), and `range_roundtrip`
  (differential testing of the new §5.1 range encoder: the fuzz input
  is a symbol script encoded through `RangeEncoder` and decoded back
  through `RangeDecoder`, every symbol asserted equal). The stale
  workflow comment describing a dropped oracle harness was rewritten
  to match the new target inventory.

- *Range encoder (RFC 6716 §5.1) — `range_encoder::RangeEncoder`.* The
  first encode-side subsystem: a bit-exact clean-room transcription of
  the §5.1 range coder, the shared entropy back end both the SILK and
  CELT layers of an Opus encoder write every coded symbol through. The
  §5.1 state four-tuple `(val, rng, rem, ext)` is carried directly, with
  the §5.1.1 `ec_encode` symbol update, §5.1.1.1 renormalization,
  §5.1.1.2 carry propagation / output buffering (the deferred 255-run
  `ext` scheme), the §5.1.2.1–§5.1.2.3 division-free variants
  (`encode_bin`, `enc_bit_logp`, and `enc_icdf` — the latter consuming
  the decoder's `icdf[]` tables verbatim), §5.1.3 raw bits packed
  back-to-front at the buffer tail, §5.1.4 `enc_uint` (range-coded top-8
  bits + raw remainder), §5.1.5 stream finalization (the
  maximal-trailing-zeros terminating value `end`, the carry-buffer
  flush, and the raw-bit tail layout), and §5.1.6 `tell` / `tell_frac`
  bit-usage accounting that matches the decoder's value bit-for-bit
  after the same symbols. Every write path is validated by decoding
  back through the crate's §4.1 `RangeDecoder`: per-primitive roundtrip
  tests (icdf / bit_logp / raw bits / uint / generic `ec_encode`),
  §5.1.6 `tell` / `tell_frac` lockstep symbol-for-symbol, and a
  5000-seed mixed-symbol fuzz roundtrip interleaving all four symbol
  kinds. When raw bits are present, `finish` isolates them from the
  range data with a zero pad one range-window deep so the decoder's
  forward lookahead never consumes a raw byte (a fixed-frame-size
  `finish` that byte-shares per §5.1.5 can layer on once the CELT
  encode-side bit allocation exists to guarantee the §5.1.6 budget
  invariant). 8 new unit tests.

- *CELT-only decode-path truncation-safety test (`malformed_input`).*
  The CELT-only frame decode now consumes a long run of range-coded
  symbols past the §4.3.7.1 prefix (coarse energy, tf_change / tf_select,
  spread, and the §4.3.3 signalled allocation header), so a new test
  decodes a config-19 (20 ms CELT-only mono) packet at every truncation
  length from 1 byte to the full body. Every length must decode without
  panicking, emit exactly the 960-sample 20 ms output, and report a real
  CELT outcome — verifying the §4.1 range coder's sticky-error fallback
  to the §4.6 silence floor across the whole new decode path.

- *CELT §4.3.1 TF + §4.3.4.3 spread + §4.3.3 allocation-header decode
  wired into the CELT-only frame path
  (`decoder::OpusDecoder::decode_celt_tf_spread_allocation`).* A
  non-silent CELT-only frame now consumes the full run of Table-56
  symbols between the §4.3.2.1 coarse energy and the §4.3.4 residual,
  from the live range coder and in exact Table-56 order: the §4.3.1
  per-band `tf_change` flags and gated `tf_select` bit
  (`celt_tf_decode::decode_tf`), the §4.3.4.3 `spread` symbol
  (`celt_spreading::decode_spread`), then the entire *signalled* part of
  the §4.3.3 bit allocation — band boosts
  (`celt_band_boost::decode_band_boosts`, walking the `start..end` coding
  window with the per-band `cap[]` from `celt_cache_caps50::cap_for_band_bits`
  and the per-channel MDCT-bin counts), the allocation trim
  (`celt_alloc_trim::decode_alloc_trim`, gated on the running
  `ec_tell_frac`), and the anti-collapse / skip / intensity-stereo /
  dual-stereo reservations (`celt_reservations::reserve_block`).
  Consuming `tf_change` / `tf_select` / `spread` *before* the allocation
  is required by Table 56: every subsequent symbol would otherwise read
  from the wrong bitstream position. This advances the entropy decoder
  through everything the bitstream explicitly carries before the
  (reference-code-only, docs-gapped) implicit `interp_bits2pulses`
  interpolation, leaving the coder positioned exactly where the §4.3.4
  PVQ residual decode will resume. A frame that reaches this point
  reports the new `FrameDecodeStatus::CeltAllocationDecoded`; the §4.3.3
  implicit allocation, §4.3.4 PVQ band shapes, and §4.3.2.2 fine energy
  remain pending, so the synthesis backend is still advanced with
  all-zero bands and correct-length silence is emitted. 2 new decode-path
  tests (the TF+spread+allocation header reachable end-to-end; it
  strictly advances the range-coder tell past coarse energy and yields a
  `tf_change` per coded band plus an in-range `spread`).

- *CELT §4.3.4.5 time-frequency Hadamard transform (`celt_tf_hadamard`).*
  Consumes a band's `TfDirection` (from `celt_tf_decode` /
  `celt_tf_adjust`) and reshapes its interleaved short-MDCT shape
  vector: `IncreaseFrequency` applies the across-block orthonormal
  Walsh–Hadamard transform in natural order; `IncreaseTime` applies the
  same butterfly in sequency (Walsh) order ("input vector is sorted in
  time"), via the bit-reverse + inverse-Gray permutation; `Unchanged`
  is identity. The orthonormal (1/√2) butterfly preserves the band's
  unit-norm shape energy exactly (the invariant the §4.3.6
  denormalisation relies on) and is its own inverse. Partial-level
  transforms split each `B`-block group into consecutive `2**levels`
  sub-runs. 14 unit tests.
- *CELT §4.3.4 per-band shape decode orchestrator (`celt_band_shape`).*
  `decode_band_shape{,_into}` composes the three fully-specified §4.3.4
  steps — §4.3.4.2 PVQ decode → §4.3.4.3 spreading → §4.3.4.5 TF
  Hadamard — given a band's `(N, K, spread, tf_adjust, nb_blocks)`,
  producing the normalized time-frequency shape ready for §4.3.6
  denormalisation. Each step is norm-preserving; `K = 0` yields an
  all-zero band. The §4.3.4.4 recursive split path (codebooks > 32
  bits, gain precision "derived from the current allocation") is
  deliberately not composed — its derivation is reference-only and
  absent from the RFC narrative. 9 unit tests.

- *Multistream / multichannel decode subsystem (RFC 7845 §3 / §5.1 /
  §5.1.1).* A complete new path for multichannel Opus:
  - `OpusHead` parses and fully validates the §5.1 identification header
    (version, output channel count, pre-skip, input sample rate, output
    gain, mapping family) and the §5.1.1 channel-mapping table (stream
    count N, coupled count M, per-output mapping indices), enforcing every
    MUST in the two sections (non-zero counts, per-family channel ranges,
    `M ≤ N`, `M + N ≤ 255`, `< M+N`/255 index bound; family 0 synthesizes
    the table from RFC-pinned defaults).
  - `split_multistream_packet` recovers the N per-stream Opus packets from
    one multistream packet (§3): the first `N − 1` via RFC 6716 Appendix-B
    self-delimited framing, the last as the undelimited remainder.
  - `MultistreamDecoder` wraps one stateful `OpusDecoder` per coded stream
    and assembles the `C` output channels by the §5.1.1 index rule
    (coupled-stream L/R by parity, mono streams, index-255 silence,
    decoded channels routed to multiple outputs), enforcing the §3
    equal-duration constraint. `decode_packet` /
    `decode_self_delimited_packet` on `OpusDecoder` share one decode body
    so both framing variants thread identical layer state.
  - `apply_output_gain` / `OpusHead::apply_gain` apply the §5.1 Q7.8 dB
    output gain (i16-saturating); `PreSkip` accumulates the §5.1 pre-skip
    across packets.
  - End-to-end validated on the real SILK fixtures: an `N = 1` family-0
    decode is byte-identical to a plain `OpusDecoder`; a coupled-stream
    L/R split reproduces a plain stereo decode exactly; mono-pair,
    swapped, silence, and duplicate channel maps all route correctly; a
    mismatched-duration packet is rejected.
- *In-band FEC (§4.2.5 LBRR) recovery — `OpusDecoder::decode_packet_fec`*
  (RFC 6716 §2.1.7 / §4.2.5): a new public entry point that reconstructs a
  **lost** frame's audio from the Low Bit-Rate Redundancy carried in the
  *next* received packet. In-band FEC re-encodes the signal immediately
  prior to a packet at a lower bitrate as one or more §4.2.5 LBRR frames;
  when the application detects a loss and holds the following packet, it
  calls `decode_packet_fec` on that packet to recover the missing frame
  rather than emitting pure silence. The method decodes the §4.2.4 LBRR
  flags + the §4.2.5 LBRR frame(s) (mono, or interleaved mid/side for
  stereo) in Table-5 order, runs the full §4.2.7.9 LTP / LPC synthesis from
  a fresh state (the lost frame's true history is, by definition,
  unavailable), unmixes a stereo recovery via §4.2.8, and resamples to
  48 kHz. Until now the LBRR frames were parsed only to keep the range
  coder aligned and then discarded; they now produce **real recovered
  PCM**. The outcome is reported through the new `FecDecodeStatus`
  (`Recovered` / `NoLbrr` / `NotSilk` / `DecodeError`) on a `FecRecovered`
  result, and on success the carried SILK synthesis (and stereo unmix)
  state advances to the recovered frame so a subsequent `decode_packet`
  continues smoothly. Per §4.2.5 all LBRR frames are active-coded; a
  CELT-only carrier reports `NotSilk` (no LBRR exists). The new
  `tests/fec_decode.rs` drives the path against the `fec-on` fixture (a
  mono WB SILK stream encoded with `-fec 1`), asserting (1) at least one
  frame is recovered from LBRR, (2) the recovered 440 Hz audio is
  non-silent, (3) a recovered frame yields the carrier's per-frame 48 kHz
  sample count, (4) a no-FEC stream cleanly reports `NoLbrr` + silence, and
  (5) a regular decode continues correctly after a FEC recovery. The
  `fec-on.opus` fixture is embedded in `tests/fixtures/`.
- *First end-to-end SILK fixture-decode validation*
  (`tests/silk_fixture_decode.rs`): decodes three SILK Opus streams
  (`silk-nb-mono-16kbps`, `silk-wb-stereo-20kbps`,
  `silk-mb-60ms-mono-20kbps`) packet-by-packet through the top-level
  `OpusDecoder::decode_packet` path and asserts (1) §3.1 TOC
  routing (mode / bandwidth / channels) per packet, (2) whole-stream
  decode with no error status — every audio packet takes a real SILK
  path across mono / stereo and NB / MB / WB and 20 ms / 60 ms frames,
  (3) §3 sample-count accounting on the 48 kHz interleaved output, and
  (4) a Goertzel probe confirming the 440 Hz NB sine fixture is decoded
  to a 440 Hz-dominant signal. A minimal **test-only** Ogg page de-laker
  recovers the raw Opus packets from the `.opus` fixtures (the codec
  crate itself owns no container parsing); validation is signal- /
  structure-based rather than bit-exact because the §4.2.9 SILK→48 kHz
  resampler is non-normative. This is the first whole-stream exercise of
  the SILK decode pipeline on real reference-encoder-produced data. The `.opus`
  streams are embedded in `tests/fixtures/` (copied from the project's
  `docs/audio/opus/fixtures/` corpus) so the suite runs in the crate's
  standalone CI without the umbrella `docs/` submodule.
- §4.3.2.1 *CELT coarse-energy reconstruction recurrence*
  (`celt_coarse_energy`): turns the per-band Laplace prediction-error
  symbols into the reconstructed base-2-log band energies by running the
  §4.3.2.1 2-D prediction filter `A(z_l, z_b)` in reverse. The
  recurrence is derived algebraically from the RFC's z-transform: the
  in-frame frequency accumulator updates as `pred_freq[b+1] =
  pred_freq[b] + (1-beta)*R[b]` and the reconstruction is `E[b][l] =
  alpha*E[b][l-1] + pred_freq[b] + R[b]`, with the cross-frame history
  `E[b][l-1]` threaded on a `CoarseEnergyState` (reset on a SILK→CELT
  transition / intra frame). The §4.3 `e_means` Q4 baseline is added
  back only for the reported energy that feeds denormalise. Inter / intra
  `(alpha, beta)` come from `celt_e_prob_model`; the `e_means` data is the
  25-value numeric table from `docs/audio/celt/tables/e_means.csv`. The
  RFC's "clamped internally" bound is not in the normative body (it lives
  only in reference code), so the clamp is left as a documented identity
  seam — exact for every in-range bitstream — pending a clean-room docs
  trace of the bound.
- §4.3.2.1 *CELT coarse-energy Laplace symbol decode*
  (`celt_laplace::ec_laplace_decode`): the 15-bit Laplace path the
  coarse-energy decoder uses — draw a 15-bit cumulative position, walk
  the decaying body until the per-magnitude mass reaches the
  `LAPLACE_MINP` floor, cover the flat tail with one division, resolve
  the sign, and report the consumed `[fl, min(fl+fs, 32768))` interval
  back to the range coder. Constants + the `prob<<7` / `decay<<6`
  Q-scalings + the `get_freq1` first-magnitude helper from
  `docs/audio/celt/spec/celt-laplace-decode.md`.
- §4.3.7.1 *CELT frame-prefix symbol decode* (`celt_frame_prefix`):
  decodes the fixed range-coded prefix every CELT frame opens with in
  RFC 6716 Table-56 order — silence (`{32767,1}/32768`), the post-filter
  group (on/off bit + `uniform(6)` octave + `4+octave` raw pitch bits +
  3 raw gain bits + `{2,1,1}/4` tapset, with `T=(16<<octave)+fine-1` and
  `G=3*(int_gain+1)/32`), then transient and intra (`{7,1}/8` each).
- *CELT-only silence frame wired end-to-end* (`decoder`): a CELT frame
  whose §4.3.7.1 silence flag is set now decodes through the real range
  coder and drives the §4.3.6→§4.3.7.2 synthesis backend with all-zero
  bands, emitting silence PCM while carrying the MDCT overlap-add +
  de-emphasis state forward (`FrameDecodeStatus::CeltSilence`). The
  `CeltSynthState` is carried on `OpusDecoder` and rebuilt on a frame
  size / channel count change.
- *CELT-only non-silent coarse-energy decode wired into the decoder*
  (`decoder`): a CELT-only frame whose §4.3.7.1 silence flag is clear now
  decodes its §4.3.2.1 coarse energy from the real range coder via
  `celt_coarse_energy::CoarseEnergyState`, threaded on `OpusDecoder`
  (`celt_coarse`), reset on an intra frame / SILK→CELT transition. The
  reconstructed per-band log-energy envelope is consumed and the
  cross-frame predictor state advances; the synthesis backend is driven
  with all-zero bands (the §4.3.3 allocation / §4.3.4 PVQ band shapes /
  §4.3.2.2 fine energy are still pending), so the frame still emits the
  §4.6 floor but reports the new `FrameDecodeStatus::CeltCoarseEnergyDecoded`
  — the coarse-energy *front half* of the CELT entropy decode is now real.
- §4.3.6 / §4.3.7 / §4.3.7.2 *CELT synthesis backend composition*
  (`celt_synthesis`): a new `CeltSynthState` composes the
  individually-tested §4.3.6 denormalise → §4.3.7 inverse MDCT → §4.3.7
  weighted overlap-add → §4.3.7.2 de-emphasis primitives into one
  per-channel call (`synthesize_channel_into`) that turns the
  frequency-domain output of the CELT entropy front half (per-band
  unit-L2 shapes + per-band `log2` energies) into time-domain PCM,
  threading the cross-frame overlap-add history and de-emphasis one-pole
  memory the spec's continuous reconstruction requires; `reset` zeroes
  both for the §4.5.2 CELT reset. The frequency buffer is sized to the
  **full** MDCT size `N = frame_samples` (120/240/480/960) and the
  denormaliser fills only the lower `coded_bins` (Table-55 column sum,
  100/200/400/800 CELT-only), leaving the high bins above the 20 kHz band
  edge as the exact zero-pad the inverse MDCT consumes.
  `synthesize_frame_interleaved_i16` runs every channel and returns
  interleaved 48 kHz signed-16-bit PCM using the decoder's SILK amplitude
  convention (clamp ±1.0, ×32767, round). 14 module tests + a 5-test
  `celt_synthesis_roundtrip` integration suite (long silent stream → exact
  silence, an independent per-stage re-derivation of the backend with the
  de-emphasis undone, the §4.3.7 TDAC constant-spectrum steady state,
  frame-boundary continuity, and finite/bounded output across all four
  frame sizes × mono/stereo). This is the CELT analogue of
  `silk_synthesis`; the gapped §4.3.2 entropy front half
  (`ec_laplace_decode`) is the remaining CELT decode milestone that will
  feed it.
- §4.3 *`celt_total_bins_per_channel` doc correction* (`celt_band_layout`):
  the doc comment claimed 120/960/640 band-bin sums where the function
  (and its own `column_sum_pinned_values` test) returns 100/800/480, and
  now documents that these band-bin sums are strictly below the full MDCT
  size `frame_samples`.

- §4.2 / §4.2.8 *Stereo SILK-only packets now decode to real interleaved
  PCM* (`decoder`): `OpusDecoder` wires the full stereo SILK path it
  previously stubbed as `LayerNotWired`. A new `decode_silk_only_stereo`
  decodes the §4.2.3 two-channel header bits, then the §4.2.5 LBRR and
  §4.2.6 regular SILK frames in §4.2.2 interleaved order — per 20 ms
  interval the mid SILK frame (carrying the §4.2.7.1 stereo prediction
  weights and, when signalled, the §4.2.7.2 mid-only flag) followed by the
  side SILK frame, with the side frame skipped (and its §4.2.7.9 LTP
  buffer cleared per §4.5.2) when the mid-only flag is set or the side VAD
  path leaves it uncoded. Each channel is synthesized through its own
  carried `SilkSynthState`, then converted from mid/side to left/right via
  the existing §4.2.8 `silk_stereo::stereo_ms_to_lr` (threading the
  cross-packet `StereoUnmixState`), resampled to 48 kHz, and written
  interleaved (`[L0, R0, …]`). A new `FrameDecodeStatus::SilkStereoDecoded`
  outcome flags the real audio. The §4.2.8 unmixing runs **per SILK frame**
  (per 20 ms interval), matching the spec's `j <= i < (j + n2)` definition
  where `j` is the SILK frame start and `n2` is the SILK frame length: each
  interval applies its own §4.2.7.1 weights and restarts the 8 ms
  interpolation phase, with the previous interval's weights and trailing
  mid/side samples threaded through the carried `StereoUnmixState`; the
  per-interval L/R outputs are concatenated. The §4.2.7.1 "previous weights
  reset to zeros on any mono→stereo transition" rule and the §4.5.2 SILK
  state reset now clear the stereo synthesis + unmix state alongside the
  mono state. A new per-channel `ChannelDecodeState` helper threads the
  §4.2.7.4 / §4.2.7.6.1 / §4.2.7.5.5 inter-frame prediction state for the
  mid and side channels independently. Six decoder tests cover the
  happy-path interleaved decode, a fully-decoded long body, the 40 ms
  two-interval and 60 ms three-interval per-interval-unmix cases,
  cross-packet state threading, and the mono→stereo transition; the former
  `stereo_silk_only_routes_to_not_wired` test was rewritten.

### Changed

- §4.5.2 *Cross-packet SILK state reset* (`decoder`): `OpusDecoder` now
  records the previous Opus frame's operating mode and applies the §4.5.2
  rule "the SILK state is reset before every SILK-only or Hybrid frame
  where the previous frame was CELT-only" via the existing
  `mode_transition_reset::decide_state_resets`, clearing the carried
  §4.2.7.9 synthesis history on a CELT→SILK transition. Two tests pin the
  reset (CELT interlude makes a repeated SILK packet match a fresh decode)
  and its complement (consecutive SILK packets thread state).
- §4 *Mono SILK-only packets now decode to real PCM* (`decoder`):
  `OpusDecoder` wires the new §4.2.7.9 synthesis into the mono SILK-only
  path. After the §4.2 bitstream decode, the decoded SILK frames are run
  through `silk_synthesis::synthesize_silk_frame` (threading a carried
  per-decoder `SilkSynthState` across the packet's Opus frames, cleared on
  `reset`/§4.5.2) and the internal-rate output is resampled to 48 kHz by a
  non-normative linear-interpolation resampler (§4.2.9 explicitly leaves
  the kernel to the decoder) and converted to signed 16-bit PCM. A
  `FrameDecodeStatus::SilkParamsDecoded` outcome now carries real audio
  instead of forced silence; the still-unwired layers (stereo SILK,
  CELT-only, Hybrid) remain correct-length silence. The all-silence sweep
  test was rewritten to a length-invariant + per-status silence check, and
  a non-silent mono SILK decode test added.

### Added

- §4.2.7.1 / §4.2.7.2 *Stereo mid-channel header decode in
  `decode_silk_frame`* (`silk_decode`): `SilkFrameConfig` gained an
  optional `stereo: Option<StereoHeaderContext>` field; when `Some`, the
  front-half decode reads the §4.2.7.1 stereo prediction weights (and,
  when signalled, the §4.2.7.2 mid-only flag) in Table-5 order ahead of
  the frame type, surfacing them in `SilkFrameDecoded::stereo_pred` /
  `mid_only_flag`. This makes the SILK bitstream front half stereo-ready
  (a prerequisite for the §4.2.8 mid/side unmixing); the mono path passes
  `None` and is unchanged. 3 tests (mono has no stereo fields, mid-channel
  reads the §4.2.7.1 weights and advances the bitstream, mid-only flag
  decoded when present).
- §4.2.7.9 *SILK frame synthesis composition* (`silk_synthesis`): the new
  `synthesize_silk_frame` composes the individually-tested §4.2.7.9.1 LTP
  synthesis (`silk_ltp_synth`) and §4.2.7.9.2 LPC synthesis
  (`silk_lpc_synth`) filters into one call that turns a decoded
  `SilkFrameDecoded` parameter set + excitation into the SILK frame's
  internal-rate (8/12/16 kHz) time-domain `out[]` samples, subframe by
  subframe. It owns the §4.2.7.9 per-subframe LPC selection (subframes
  0/1 of an interpolation-split 20 ms frame use the interpolated `n1`
  filter, every other subframe the uninterpolated `n2` filter) and drives
  the §4.2.7.9.1 LSF-interpolation-split rewhitening branch for subframes
  2/3. The cross-frame `SilkSynthState` carries the §4.2.7.9.1 `out[]` /
  `lpc[]` histories and the §4.2.7.9.2 LPC history across SILK frames and
  resets per §4.5.2. `synthesize_silk_frames` chains a 2/3-SILK-frame
  Opus frame; `silk_frame_internal_samples` reports the per-frame output
  geometry. 8 unit tests.
- §4.2.6 / §4.2.7 *In-order SILK frame decode* (`silk_decode`):
  `decode_silk_frame` composes the individually-tested per-stage SILK
  decoders into one call that reads a regular SILK frame's bitstream in
  the exact Table-5 symbol order — frame type (§4.2.7.3), subframe gains
  (§4.2.7.4), LSF stage-1 (§4.2.7.5.1), LSF stage-2 (§4.2.7.5.2), LSF
  interpolation weight (§4.2.7.5.5, 20 ms), LTP lags/gains/scaling
  (§4.2.7.6, voiced), LCG seed (§4.2.7.7), and quantized excitation
  (§4.2.7.8) — with the critical property that the §4.2.7.4 gains are
  read *between* the frame type and the LSF stage-1 index, as Table 5
  requires. After the bitstream is consumed it runs the non-bitstream
  §4.2.7.5.3–§4.2.7.5.8 LSF → LPC chain (codebook lookup →
  stabilization → interpolation → NLSF→LPC → bandwidth-expansion →
  prediction-gain limiting) so the `SilkFrameDecoded` result carries the
  final stable Q12 LPC coefficients (both halves for a 20 ms frame), the
  LTP parameters, and the Q23 excitation. To support this, `silk_frame`
  gained composable `SilkFrameHeader::decode_pre_gains` (stereo weights +
  mid-only + frame type) and `decode_lsf_stage1` entries plus a
  `SilkHeaderPreGains` carrier — the old `SilkFrameHeader::decode`
  (which read the LSF index back-to-back after the frame type, with no
  gains slot) is now documented as a header-field utility unsuitable for
  full-frame decode. 4 unit tests (zero-buffer totality, bit-consumption
  + d_LPC per bandwidth, the 10 ms no-interpolation-split case, SWB/FB
  rejection). `decoder::decode_silk_only_frame` now runs the real §4.2.3
  header bits + §4.2.5 LBRR / §4.2.6 regular SILK frame loop for **mono**
  SILK-only packets (1 / 2 / 3 SILK frames per §4.2.2), threading the
  inter-frame previous-gain / previous-lag / previous-NLSF state, and
  reports `FrameDecodeStatus::SilkParamsDecoded` / `SilkDecodeError`. The
  §4.2.7.9 synthesis + §4.2.9 resample to 48 kHz samples (and the stereo
  mid/side interleave) remain follow-ups; mono SILK PCM is still silence.
- §3 / §4 *Top-level packet → PCM orchestration* (`decoder`): the
  keystone `OpusDecoder` that turns a raw Opus packet into interleaved
  48 kHz PCM. `decode_packet` performs the §3.1 TOC parse, the §3.2
  frame split (via `OpusPacket::parse`, covering all four frame-count
  codes), and the §4.5 multi-frame loop — every Opus frame in a
  code-1 / code-2 / code-3 packet is decoded in order and its PCM
  appended to one contiguous output buffer. Each frame is routed
  through its `OpusFrameRouting` to a per-mode decode seam
  (`decode_silk_only_frame` / `decode_celt_only_frame` /
  `decode_hybrid_frame`). The §3.2.1 zero-length DTX / lost-frame
  marker contributes one Opus-frame of silence (the §4.6 PLC floor),
  flagged `FrameDecodeStatus::DtxOrLost`; a frame whose layer decode is
  not yet composed emits silence of the correct length flagged
  `FrameDecodeStatus::LayerNotWired(mode)`, so the multi-frame loop and
  the RFC 7845 §5.1 48 kHz sample-count accounting
  (`output_samples_per_channel`: tenths-ms × 48 / 10, exact for all six
  Table-2 durations) are exercised end-to-end regardless of which
  layer's range-coded decode has landed. `DecodedAudio` carries the
  interleaved `pcm`, the channel count, the 48 kHz rate, and the
  per-frame `FrameOutcome` vector. `OpusDecoder::reset` implements the
  §4.5.2 decoder reset. 8 unit tests: the six-duration sample-count
  table, empty-packet rejection (§3.1 R1), the SILK-NB-mono single-frame
  PCM length + status, the CELT-only stereo interleaved length, the
  code-1 two-frame PCM concatenation, the §3.2.1 DTX silence + status,
  the reset clearing carried channel state, and an all-32-config silence
  sweep verifying every routing produces the routed PCM length.
- §4.3.4.5 *Time-frequency change decode* (`celt_tf_decode`): the
  range-coded read path that complements the round-20 `celt_tf_adjust`
  tables. `decode_tf` walks the coded band range (`0..21` CELT-only /
  `17..21` Hybrid) reading one `tf_change[b]` flag per band — the first
  coded band as an *absolute* choice (`{3,1}/4` transient / `{15,1}/16`
  otherwise) and every subsequent band as a *difference* bit coded
  relative to the previous band's choice (`{15,1}/16` transient /
  `{31,1}/32` otherwise), the running choice toggled by each diff. After
  the band loop it reads the §4.3.1 `tf_select` flag (`{1,1}/2`) *only*
  when `celt_tf_select_can_affect` reports it can change at least one
  band's adjustment, otherwise leaving it implicitly `0` with no bit
  consumed, then maps each `(frame_size, transient, tf_select,
  tf_change[b])` tuple through `celt_tf_adjustment` to the per-band
  Table-60..63 adjustment. The "relative = difference/toggle" reading of
  the §4.3.4.5 subsequent-band prose is documented in the module. The
  `TfDecode` result carries the per-band `tf_change`, the `tf_select`
  flag, and the parallel adjustment vector. 11 unit tests built on a
  bit-exact §5.1 range-*encoder* test fixture (transcribed from the
  normative RFC 6716 §5.1.1 / §5.1.1.2 / §5.1.5 encoder, the inverse of
  the §4.1 decoder): a fixture self-round-trip through the real decoder,
  empty-range no-op, absolute first-band choice, relative-toggle
  subsequent bands, the tf_select skip/read gate at 2.5 ms (never) and
  10 ms / 20 ms (conditional), full 21-band CELT-only decode, and the
  Table-63 tf_select=1 route.
- §4.3.7 *Weighted overlap-add* (`celt_overlap_add`): the CELT stage
  between the inverse MDCT (§4.3.7, `celt_imdct`) and the post-filter
  (§4.3.7.1). `WeightedOverlapAdd` is the stateful, per-channel adder
  that turns the stream of `2N`-sample inverse-MDCT blocks into the
  aliasing-free time-domain signal: each frame it applies the §4.3.7
  low-overlap synthesis window (rising ramp on the leading `overlap`
  samples, falling ramp on the trailing `overlap`, unity in the middle),
  overlap-adds the windowed leading half with the previous block's
  windowed trailing half (hop = `N`), emits `N` samples, and carries the
  new trailing half forward as the overlap history. `new` / `reset`
  give the all-zero stream-start / §4.5.2-reset state; `process` /
  `process_into` run one frame; `apply_synthesis_window` is the reusable
  free-function windower. `OverlapAddError` covers zero `N`, non-even /
  mismatched block lengths, out-of-range or odd overlap, and a too-small
  output buffer. 14 unit tests + 1 doctest: silence-stays-silent, the
  zero-history first-frame leading-half emission, the windowed
  trailing-half history carry, reset, the rising/falling/unity window
  layout, a multi-frame arithmetic cross-check against an independent
  hand computation with the §4.3.7 ramp, an end-to-end TDAC
  reconstruction (windowed forward MDCT -> inverse MDCT -> hop-`N`
  overlap-add reconstructs the input at the documented `1/2` scale,
  err < 1e-9), and every error path.
- §4.3.7 *Inverse MDCT* transform core (`celt_imdct`): the CELT stage
  between band denormalisation (§4.3.6) and the weighted overlap-add /
  post-filter (§4.3.7.1). `imdct_into` / `imdct` map the `N`
  denormalised MDCT bins to `2N` time-domain samples via the textbook
  inverse type-IV/MDCT cosine kernel
  `y(n) = (1/N)·Σ_k X(k)·cos[(pi/N)(n+1/2+N/2)(k+1/2)]`, the `1/N`
  constant realising the RFC's stated "scaling by 1/2" relative to the
  unnormalised forward/inverse pair. A matched-kernel `mdct_forward`
  (test-only partner; the decoder never runs a forward MDCT) lets the
  round-trip / TDAC behaviour be pinned. `ImdctError` covers empty
  spectra and output-length mismatch. 13 unit tests: direct-formula
  recompute, linearity, the canonical MDCT time-domain aliasing folds
  (`y(n) = -y(N-1-n)` lower / mirror upper), the exact aliased
  round-trip block, and a windowed forward/inverse overlap-add of two
  adjacent frames reconstructing an arbitrary signal at the documented
  `1/2` gain (TDAC, err < 1e-9) using a symmetric Princen-Bradley
  window, plus every error path.
- §4.3.6 *Denormalisation* (`celt_denormalise`): the last CELT step
  before the inverse MDCT. `denormalise_gain` converts a per-band
  base-2-log energy `L` to the linear shape gain `sqrt(2**L) =
  2**(L/2)`; `denormalise_band` multiplies one band's unit-L2-norm
  PVQ shape by that gain in place; `denormalise_bands` walks the coded
  bands (`0..21` CELT-only, `17..21` Hybrid) via the Table-55 layout
  and lays each band's denormalised coefficients end-to-end into the
  frequency-domain buffer the inverse MDCT consumes, returning the
  coefficient count written. `DenormaliseError` covers band-range,
  output-too-small, and shape/output length-mismatch. 11 unit tests:
  gain vs independent `sqrt(2**L)`, energy-preservation
  (`||g·shape||² = 2**L`), zero-shape passthrough, attenuating
  negative `L`, full CELT-only and Hybrid buffer fills, and every
  error path.
- §4.3.4.1 *Bits-to-Pulses* pulse-cost cache (`celt_pulse_cache`): the
  105-entry band-major `(band, LM)` → offset `CACHE_INDEX50`, the
  392-byte run-packed `CACHE_BITS50` cost curves (1/8-bit / Q3 units),
  the `bits_to_pulses` budget-inversion scan, the run accessors
  (`cache_run_offset` / `cache_max_pulses` / `cache_pulse_cost`), and
  the eight-tuple sentinel closed-form signal (`PulseCacheError`). 17
  unit tests: index/bits table lengths, 23-distinct-run packing closes
  exactly 392 bytes, per-run monotone cost curves, sentinel routing,
  exact-threshold and saturating inversion, budget monotonicity.

## [0.0.12](https://github.com/OxideAV/oxideav-opus/compare/v0.0.11...v0.0.12) - 2026-06-15

### Other

- §4.3.7.1 pitch post-filter response (celt_post_filter)
- §4.3.4.2 PVQ shape read path (index read + unit-L2 normalization)
- fix clippy assertions-on-constants in celt_deemphasis test
- §4.3.7.2 de-emphasis filter (round 48)
- §4.3.7 inverse-MDCT overlap window (round 47)
- §4.1.2 public two-step range-coder symbol API (ec_decode + ec_dec_update)
- §4.3.2.1 per-LM inter-mode (alpha, beta) prediction coefficients (round 45)
- §4.3.4.3 spreading rotation (round 44)
- §4.3.4.2 PVQ index-to-vector decode (round 43)
- clean-room round 42 — §4.3.2.2 fine-energy quantization
- §4.3.4.2 PVQ codebook-size function V(N, K) (round 41)
- §4.3.3 1/64-step interpolated allocation search (round 40)
- §4.3.3 static allocation table (Table 57) (round 39)
- §4.5.3 normative + recommended-non-normative transition table (round 38)
- drop release-plz.toml — use release-plz defaults across the workspace
- RFC 6716 Appendix B self-delimiting framing (round 37)
- §4.3.3 per-band allocation-trim offsets (round 36)
- §4.3.3 per-band minimum-allocation vector (round 35)
- §4.3.3 reservation block (round 34)
- §4.3.3 band-boost decoder (round 33)
- §4.3.3 allocation-trim Table-58 PDF + signalling gate (round 32)
- §4.3.3 CACHE_CAPS50 per-band maximum-allocation parameter surface (round 31)
- §4.3.3 LOG2_FRAC_TABLE intensity-stereo reservation parameter surface (round 30)
- §4.3.2.1 CELT coarse-energy Laplace-model parameter surface (round 29)
- §4.5.1.4 redundant-CELT-frame decode parameters + cross-lap placement (round 28)
- §4.5.2 SILK + CELT state-reset policy across mode transitions (round 27)
- §4.5.1 CELT redundancy / mode-transition side information (round 26)

### Added

* **Clean-room round 50 (2026-06-15):** §4.3.7.1 pitch *post-filter
  response* (`celt_post_filter`). The pitch comb filter the CELT decoder
  applies to the inverse-MDCT / overlap-add output (§4.3.7) just before
  the round-48 §4.3.7.2 de-emphasis; the post-filter *parameters* landed
  in `celt_header` at round 20, this round owns their *application*. RFC
  6716 §4.3.7.1 (p. 121) states the five-tap symmetric comb response
  `y(n) = x(n) + G*(g0*y(n-T) + g1*(y(n-T+1)+y(n-T-1)) +
  g2*(y(n-T+2)+y(n-T-2)))` — recursive over past *output* samples (state
  = the trailing `T + 2` outputs, carried across frames, reset only on a
  §4.5.2 CELT state reset). The ASCII equation prints each pair term with
  a repeated index; the symmetric `y(n-T±1)` / `y(n-T±2)` reading is the
  only one consistent with the "g0, g1, g2" symmetric-tap-set prose, and
  the module documents the slip. New surface: `POST_FILTER_TAPS` (the
  three §4.3.7.1 `(g0, g1, g2)` tapsets) + `tapset_coefficients` +
  `post_filter_gain` (`G = 3*(gain_index+1)/32`) + the formula constants;
  `PostFilterCoeffs { period, a0, a1, a2 }` (gain folded into each tap)
  via `new` / `from_header`; `PostFilter` carrying the output history
  (`new` / `reset` / `step` / `process_in_place` / `process` /
  `process_gain_transition` — the §4.3.7.1 squared-MDCT-window gain
  crossfade pushing the mixed value into the shared feedback history per
  "interpolated one at a time"); the standalone `crossfade_transition`;
  and `PostFilterError::{TapsetOutOfRange, PeriodOutOfRange,
  GainIndexOutOfRange, OutputBufferTooSmall, TransitionLengthMismatch,
  Window}` with `From<MdctWindowError>`. Twenty-six new unit tests (1064
  lib tests, up from 1038; 20 integration unchanged): the tapset decimals
  + `g2 = 0` for tapsets 1/2, the gain formula + monotonicity, the
  gain-fold, period bounds, header round-trip, fresh-history
  pass-through, a hand-expanded impulse response placing each tap at its
  exact lag, impulse-response symmetry about `T`, history carry across
  split blocks, the buffer paths, `reset`, the gain-transition endpoints
  + old==new identity + length checks, the standalone crossfade
  convexity, and every error `Display`. Provenance: §4.3.7.1 comb
  response, tapsets, gain map, and squared-window transition rule in the
  staged `docs/audio/opus/rfc6716-opus.txt`; the squared window reuses
  the round-47 `celt_mdct_window`; no external library source consulted.
* **Clean-room round 49 (2026-06-15):** §4.3.4.2 PVQ *shape* read path
  (`celt_pvq_decode`). Composes the three steps RFC 6716 §4.3.4.2
  (p. 116–117) states in sequence: the up-front
  `i = ec_dec_uint(V(N, K))` codeword-index read
  (`RangeDecoder::dec_uint`), the round-43 five-step index-to-vector
  walk, and the final normalization — "The decoded vector X is then
  normalized such that its L2-norm equals one." New surface:
  `pvq_unit_normalize(&[i32], &mut [f64])` (the unit-L2 scaling, with
  the `K = 0` no-direction case left all-zeros); `decode_pvq_shape(rd,
  n, k) -> Vec<f64>` and `decode_pvq_shape_into(rd, n, k, &mut [f64])`
  (the full read path returning the unit-norm `f64` shape the §4.3.4.3
  spreading rotation operates on and §4.3.6 denormalization later
  scales by the band energy); and `PvqShapeError::{CodebookSize,
  RangeDecoder, PulseVector, OutputBufferTooSmall}` with `From<PvqVError>`
  / `From<PvqDecodeError>`. Fourteen new unit tests (1038 lib tests, up
  from 1024; 20 integration unchanged): the unit-L2 norm over every
  codeword of `(N, K) ∈ 1..=6 × 1..=6`, direction preservation,
  single-pulse ±1, the zero-vector carve-out, the buffer paths;
  `decode_pvq_shape` ↔ `decode_pvq_vector` + `pvq_unit_normalize`
  consistency from fixed buffers, the `_into` parity, the `K = 0`
  no-bit-consumption edge, `N = 1` signed-unit, codebook-size error
  propagation, and the error conversions/`Display`.
* **Clean-room round 48 (2026-06-14):** §4.3.7.2 de-emphasis filter
  (`celt_deemphasis`). The final stage of the CELT decode pipeline,
  applied after the inverse MDCT + overlap-add (§4.3.7) and the
  §4.3.7.1 pitch post-filter. RFC 6716 §4.3.7.2 (p. 122) states it as
  the inverse of the encoder's pre-emphasis filter,
  `1/A(z) = 1/(1 - alpha_p*z^-1)` with `alpha_p = 0.8500061035`;
  inverting the one-pole transfer function gives the time-domain
  recurrence `y(n) = x(n) + alpha_p*y(n-1)`. The single state element
  carries across frame boundaries (the recurrence is continuous over
  the whole stream, reset only on a §4.5.2 CELT state reset). New
  surface: `DEEMPHASIS_ALPHA_P = 0.8500061035` constant;
  `DeemphasisFilter` with `new()` / `with_memory(mem)` / `memory()` /
  `reset()` / `step(x)` (one-sample recurrence) /
  `process_in_place(&mut [f64])` / `process(input, output)`; and
  `DeemphasisError::OutputBufferTooSmall`. Fifteen new unit tests
  (1024 lib tests, up from 1009; 20 integration unchanged): the
  `alpha_p` constant + stability bound, fresh-filter zero memory,
  first-sample pass-through, hand-computed recurrence, constant-input
  convergence to the DC gain `1/(1 - alpha_p)`, the
  pre-emphasis-round-trip inverse property, memory carry across split
  blocks, `with_memory` seeding, `reset`, the `process_in_place` ↔
  `step` parity, output-buffer write + over-long acceptance +
  short-buffer rejection (state unchanged on error), empty-input
  no-op, and the error `Display`. Provenance: §4.3.7.2 transfer
  function + `alpha_p` constant in the staged
  `docs/audio/opus/rfc6716-opus.txt`; the recurrence is the textbook
  inverse of the stated one-pole filter; no reference source
  consulted.

* **Clean-room round 47 (2026-06-14):** §4.3.7 inverse-MDCT overlap
  window (`celt_mdct_window`). RFC 6716 §4.3.7 (p. 121) states the
  basic full-overlap 240-sample Vorbis-derived window directly:
  `W(n) = sin( (pi/2) * sin( (pi/2) * (n + 1/2) / L )^2 )` (the `2`
  superscript squares the *inner* sine — the only nesting that
  satisfies the §4.3.7 power-complementarity / Princen-Bradley
  requirement `W(n)^2 + W(L-1-n)^2 = 1`, which the module proves
  algebraically and pins in tests). New surface: `window_tap(n, len)`
  (the amplitude tap for window length `L = len`), `basic_window()`
  (the 240-tap full-overlap window), `mdct_window(overlap)` (the
  §4.3.7 "low-overlap" ramp for an arbitrary even overlap, built by
  evaluating the same shape with `L = overlap` — the "zero-pad +
  insert ones in the middle" construction expressed as a per-overlap
  rising ramp), `celt_overlap_window()` (the fixed CELT 120-sample
  overlap at 48 kHz — the 2.5 ms look-ahead "fixed by the decoder",
  RFC 6716 §1), constants `BASIC_WINDOW_LEN = 240` /
  `CELT_OVERLAP_48K = 120`, and `MdctWindowError::{PositionOutOfRange,
  ZeroLength, OddOverlap}`. Eighteen new unit tests (1009 lib tests,
  up from 991; 20 integration unchanged): the formula spot-check,
  unit-interval bound, monotone-increasing ramp, power complementarity
  on both the basic window and arbitrary overlaps, the half-power
  centre pair, endpoint shape, `mdct_window` ↔ `window_tap` parity,
  the `celt_overlap_window` ↔ `mdct_window(120)` parity, and every
  error path. The inverse MDCT itself and the weighted overlap-add
  that consumes this ramp run at the §4.3.7 consumer site.
  Provenance: §4.3.7 window equation + low-overlap narrative + §1
  fixed-overlap statement, all in the staged
  `docs/audio/opus/rfc6716-opus.txt`; no reference source consulted.

* **Clean-room round 46 (2026-06-13):** public two-step range-coder
  symbol API `RangeDecoder::ec_decode(ft) -> fs` and
  `RangeDecoder::ec_dec_update(fl, fh, ft)` per RFC 6716 §4.1.2,
  promoting the previously-private `decode` / `dec_update` helpers to a
  validated public surface. `ec_decode` computes the §4.1.2 symbol
  proxy `fs = ft - min(val/(rng/ft) + 1, ft)` and rejects `ft == 0`
  (division by zero) by latching the sticky error flag; `ec_dec_update`
  narrows the range to the chosen symbol's `[fl, fh)` sub-interval of
  `[0, ft)` and rejects malformed tuples (`ft == 0`, `fh > ft`,
  `fl >= fh`) with the same error-latch-and-no-op guard, so a corrupt
  search result cannot underflow `val` or zero `rng`. This is the
  generic symbol path required when the frequency model is computed at
  run time rather than baked into a fixed `icdf[]` table — the building
  block for the deferred §4.3.2.1 CELT coarse-energy Laplace decoder and
  the §4.3.3 allocation interpolation search. Five new unit tests (991
  lib tests, up from 986; 20 integration unchanged): split-path ↔
  fused-`dec_icdf` lockstep over a uniform 8-way PDF, split-path ↔
  `dec_uint` lockstep in the small (`ftb <= 8`) regime, `fs ∈ [0, ft)`
  for `ft` up to `2**16`, `ec_decode(0)` error latch, and the three
  malformed-tuple rejections for `ec_dec_update`. Provenance: §4.1.2
  prose (the `ec_decode` / `ec_dec_update` equations) is fully normative
  in the staged `docs/audio/opus/rfc6716-opus.txt`; no reference source
  was consulted. Note: the §4.3.2.1 `ec_laplace_decode` *control flow*
  remains a docs gap — the RFC defers its algorithm to `laplace.c` in
  the Appendix A tarball (reference source, off-limits to the
  Implementer); this round lands the documented primitive that decoder
  will consume once a clean-room §4.3.2.1 Laplace trace is staged.

* **Clean-room round 45 (2026-06-12):** RFC 6716 §4.3.2.1 per-LM
  *inter*-mode `(alpha, beta)` coarse-energy prediction coefficients,
  closing the round-29 deferral. `celt_e_prob_model` gains
  `INTER_PRED_ALPHA_Q15 = [29440, 26112, 21248, 16384]` and
  `INTER_PRED_BETA_Q15 = [30147, 22282, 12124, 6554]` (Q15 numerators
  against `Q15_ONE = 32768`, indexed by `LM = log2(frame_size/120) ∈
  0..=3`), the `EnergyPredCoef { alpha_q15, beta_q15 }` pair type with
  exact `alpha()` / `beta()` binary-fraction float views, and the
  range-checked `energy_pred_coef(lm, mode)` accessor unifying the
  §4.3.2.1 p. 108 intra carve-out (`(0, 4915)`, LM-independent) with
  the per-LM inter pairs. Provenance: the RFC prose states the inter
  coefficients "depend on the frame size in use" and defers the
  numbers to the normative Appendix A reference code; the values are
  numeric facts read from the `pred_coef[4]` / `beta_coef[4]`
  declarations in `quant_bands.c` of the Appendix A source embedded in
  the staged `docs/audio/opus/rfc6716-opus.txt` (extracted via the
  RFC's own §A.1 procedure, tarball SHA-1 verified against §A.1;
  §A.2 keeps the in-document code normative; `beta_intra = 4915` in
  the same file confirms the p. 108 intra constant). Ten new unit
  tests (986 lib tests, up from 976; 20 integration unchanged): value
  pins, the exact-half `LM = 3` alpha, strict monotone decrease of
  both arrays in LM, inter-beta > intra-beta, accessor↔table
  agreement, intra LM-independence, `lm` range rejection in both
  modes, exact float views, and the doc-comment approximations.
  Follow-up note: the same Appendix A grounding stages the normative
  pulse-cache construction (`rate.c`), making the §4.3.4.1
  `cache-bits50` / `cache-index50` run-layout gap recorded in round 44
  reachable for a future round.

* **Clean-room round 44 (2026-06-11):** RFC 6716 §4.3.4.3 *spreading
  (rotation)* — a new `celt_spreading` module applying the §4.3.4.3
  anti-tonal rotation to the §4.3.4.2-decoded shape vector. Implements
  the RFC prose verbatim: the Table 56 per-frame "spread" symbol
  (`decode_spread`, PDF `{7, 2, 21, 2}/32` as `SPREAD_PDF` /
  `SPREAD_ICDF` / `SPREAD_FTB = 5`); the Table 59 `spread → f_r` map
  (`spread_f_r` / `SPREAD_F_R`: 0 → infinite/no rotation, 1 → 15,
  2 → 10, 3 → 5); the rotation gain `g_r = N/(N + f_r*K)`
  (`rotation_gain`) and angle `theta = pi*g_r^2/4` (`rotation_angle`,
  composed as `spread_theta`); the back-and-forth adjacent-pair 2-D
  rotation series (`rotate_in_place`, orthogonal /
  L2-norm-preserving); the multi-block interleave stride
  `round(sqrt(N/nb_blocks))` (`spreading_stride`, round-half-up
  documented since the RFC leaves the tie rule open); the per-set
  strided variant (`rotate_strided`, sets `S_k = {stride*n + k}`);
  and the composed `apply_spreading` (per-block `theta` rotation +
  the `(pi/2 − theta)` strided pre-rotation when `nb_blocks > 1` and
  blocks span ≥ 8 samples — `SPREAD_PRE_ROTATION_MIN_BLOCK_LEN`; the
  module doc records the documented reading of the §4.3.4.3
  multi-block paragraph, whose prose reuses `N` for the full vector
  and a block). `SpreadingError::{SpreadOutOfRange, ZeroDimensions,
  ZeroBlocks, ZeroStride, BlocksDoNotDivideLength}` covers
  caller-side bookkeeping. Twenty-eight new unit tests (976 lib
  tests, up from 948; 20 integration tests unchanged): Table 56
  PDF/iCDF consistency + exhaustive first-byte decode sweep, the
  Table 59 map, worked `g_r`/`theta` points and monotonicities, the
  2-D step against the RFC definition, the `N = 3` series against an
  explicit-matrix composition, norm-preservation / zero-angle /
  sign-linearity properties, stride worked points including the
  2.5-tie, strided-rotation gather-scatter equivalence + set
  independence, and every composed path + error path. Docs-gap note:
  §4.3.4.1 Bits-to-Pulses stays blocked — the staged
  `cache-bits50.csv` / `cache-index50.csv` carry the pulse-cache
  *values* but no staged trace describes the run layout /
  permitted-`K` mapping, and RFC 6716 §4.3.4.1 prose (p. 116) does
  not pin it. Source: RFC 6716 §4.3.4.3 (pp. 117–118) + Tables 56/59.

* **Clean-room round 43 (2026-06-11):** RFC 6716 §4.3.4.2 *PVQ
  index-to-vector decoding* — a new `celt_pvq_decode` module that
  turns a decoded codeword index `i ∈ 0..V(N, K)` into the
  integer-magnitude pulse vector `X` with `sum |X[j]| == K`, the
  consumer of the round-41 `celt_pvq_v::pvq_codebook_size` primitive.
  Implements the §4.3.4.2 five-step recovery verbatim: per coordinate
  `j`, `p = (V(N-j-1,k) + V(N-j,k)) / 2`, sign selection on `i < p`,
  then the `p -= V(N-j-1,k)` magnitude-walk loop, yielding
  `X[j] = sgn*(k0 - k)`. New public surface:
  `decode_pvq_vector(n, k, index) -> Result<Vec<i32>, PvqDecodeError>`
  (allocating) and `decode_pvq_vector_into(n, k, index, &mut [i32]) ->
  Result<usize, PvqDecodeError>` (in-place); `pvq_l1_norm(&[i32]) ->
  u64` and `pvq_l2_norm_squared(&[i32]) -> u64` invariant helpers (the
  §4.3.4.2 final "L2-norm equals one" normalization is a
  floating-point step left to the §4.3.4 consumer site); constants
  `PVQ_DECODE_N_MAX` / `PVQ_DECODE_K_MAX` mirrored from `celt_pvq_v`;
  and `PvqDecodeError::{CodebookSize(PvqVError), IndexOutOfRange,
  OutputBufferTooSmall}`. Twenty-seven new unit tests including a
  full-index-range bijection proof (`L1 == K` for every codeword over
  `(N, K) ∈ 1..=6 × 0..=6`, plus injectivity giving surjectivity onto
  the K-pulse lattice by a counting argument), hand-enumerated
  codebooks at `(1,1)/(1,3)/(2,1)/(2,2)/(3,2)`, the index-0 leading-
  positive-pulse property, error-path and buffer-handling coverage,
  and the §4.1.5 overflow propagation at `V(176,176)`. The up-front
  `ec_dec_uint(V(N, K))` index read and the §4.3.4.1 Bits-to-Pulses
  search that supplies `K` remain deferred to the consumer site.

* **Clean-room round 42 (2026-06-10):** RFC 6716 §4.3.2.2 *fine-energy
  quantization* — a new `celt_fine_energy` module that converts an
  already-read fine-energy refinement `f ∈ [0, 2**B_i - 1]` into the
  §4.3.2.2 correction `(f + 1/2) / 2**B_i - 1/2 = (2f + 1 - 2**B_i) /
  2**(B_i + 1)`, a zero-mean fraction of a 6 dB coarse-energy step,
  and plans the §4.3.2.2 *final* fine-energy bit allocation (one extra
  bit per band per channel, priority-0 bands band-0-upward then
  priority-1 bands band-0-upward, leftover bits unused). New public
  surface: `fine_correction_ratio(bits, f) -> Result<(i32, i32), …>`
  (exact reduced `(numerator, denominator)`); `fine_correction_q15(bits,
  f) -> Result<i32, …>` (correction in Q15, exact for every reachable
  `B_i`, strictly within `(-16384, +16384)`); `fine_correction_q(bits,
  f, shift) -> Result<i64, …>` (arbitrary Q-format, e.g. CELT's
  `DB_SHIFT = 10`); `fine_energy_levels(bits) -> Result<u32, …>`
  (`2**B_i`); `plan_final_fine_bits(priorities, channels, leftover_bits)
  -> FinalFineBitPlan { granted, bits_used, bits_unused }`; the
  `FineEnergyChannels::{Mono, Stereo}` / `FinalBitPriority::{Priority0,
  Priority1}` enums; constants `FINE_ENERGY_MAX_BITS = 14`,
  `FINE_ENERGY_Q15_ONE = 32768`, `FINE_ENERGY_HALF_Q15 = 16384`; and
  `FineEnergyError::{BitsOutOfRange, RefinementOutOfRange}` for
  caller-side bookkeeping bugs. The range-decoder `dec_bits(B_i)` /
  `dec_bits(channels)` reads and the addition of the correction onto
  the §4.3.2.1 reconstructed log-energy run at the consumer site.
  Thirty-three new unit tests pin the correction at worked `(B_i, f)`
  points, the odd-numerator-lowest-terms / zero-mean-symmetry /
  uniform-step / strictly-within-`±1/2` invariants, the Q15 exactness
  against the ratio form, the `DB_SHIFT = 10` parity, and the
  final-bit priority sweep (priority ordering, band order within a
  priority, stereo two-bit cost, leftover-unused, None-band exclusion,
  and the `bits_used + bits_unused == leftover` conservation law).
  Source: RFC 6716 §4.3.2.2 (p. 109).

* **Clean-room round 41 (2026-06-08):** RFC 6716 §4.3.4.2 *PVQ
  codebook-size function `V(N, K)`* — a new `celt_pvq_v` module
  evaluating the bivariate recurrence the RFC states directly:
  `V(N, K) = V(N - 1, K) + V(N, K - 1) + V(N - 1, K - 1)` with base
  cases `V(N, 0) = 1` and `V(0, K) = 0 (K != 0)`. `V(N, K)` counts
  the integer-magnitude lattice points `{ x ∈ Z^N : |x_0| + |x_1|
  + ... + |x_{N-1}| = K }` — the size of the §4.3.4 PVQ codebook for
  `N` MDCT bins and `K` pulses. The §4.3.4.2 PVQ index is decoded
  with `ec_dec_uint(V(N, K))` (§4.1.5 caps `ec_dec_uint`'s `ft`
  parameter at `2**32 − 1`), and §4.3.4.1 *Bits to Pulses* picks
  `K` by searching the codebook size at the §4.3.3 per-band
  allocation; both consume this primitive. New public surface:
  `pvq_codebook_size(n, k) -> Result<u32, PvqVError>` evaluating
  the recurrence in `u64` over two rolling rows of length `K + 1`
  and short-circuiting when any intermediate cell crosses
  `2**32 − 1`; `PVQ_V_N_MAX = 352` (caller-side bookkeeping bound
  covering joint-stereo bands at 20 ms: `2 × CELT_MAX_BINS_PER_BAND
  = 2 × 176 = 352`); `PVQ_V_K_MAX = 4096` (conservative caller-side
  upper bound on `K` so fuzz callers can sweep wide envelopes);
  `PVQ_V_MAX = 2**32 − 1` (the §4.1.5 `ec_dec_uint` ceiling
  inherited as the overflow guard); and `PvqVError::{NOutOfRange,
  KOutOfRange, OverflowsDecUintRange}` for caller-side bookkeeping
  errors and stream-impossibility reports. Twenty-three new unit
  tests (888 lib tests total, up from 865 at round-40 close; 20
  integration tests unchanged) pin the four §4.3.4.2 base cases
  (`V(0, 0) = 1`, `V(N, 0) = 1`, `V(0, K) = 0`, `V(1, K) = 2`,
  `V(N, 1) = 2N`), cross-check the bivariate recurrence over the
  N, K ∈ 1..=12 sweep, pin a 7×7 hand-computed table of `V(N, K)`
  values, pin two specific worked points (`V(3, 3) = 38`,
  `V(4, 2) = 32`) showing the `V(N, K) ≠ V(K, N)` asymmetry,
  validate the monotone-non-decreasing-in-`N` invariant (for every
  fixed `K`), validate the monotone-non-decreasing-in-`K`
  invariant for `N ≥ 2`, exercise the §4.1.5 overflow guard on
  `V(176, 176)` (well above `2**32`), confirm the guard does *not*
  trip on values just under the ceiling (`V(2, K) = 4K` for the
  full `K ∈ 0..=100` window), exercise the boundary `PVQ_V_N_MAX`
  and `PVQ_V_K_MAX` rejection paths, validate the three module
  constants (`PVQ_V_N_MAX = 352`, `PVQ_V_K_MAX = 4096`, `PVQ_V_MAX
  = 4_294_967_295`), and pin every error-Display message at the
  failing input.
* **Clean-room round 40 (2026-06-08):** RFC 6716 §4.3.3 *1/64-step
  interpolated static-allocation search* — a new `celt_alloc_search`
  module that closes the interpolation + search gap round 39 noted
  as the natural next step on top of `celt_static_alloc`. RFC 6716
  §4.3.3 (p. 111, lines 6223–6230) is explicit: "The allocation is
  obtained by linearly interpolating between two values of q (in
  steps of 1/64) to find the highest allocation that does not
  exceed the number of bits remaining." This module owns the
  interpolation + search half; the orchestrated §4.3.3 allocator
  that folds in the round-31 per-band cap, the round-33 boosts,
  the round-34 reservations, the round-35 per-band minimum, the
  round-36 trim offsets, and the skip / dual-stereo / intensity-
  stereo flag reads runs at the consumer site. New public surface:
  `Q_FP_MAX = 640` (the fixed-point quality bound packing `q'_fp =
  q_lo * 64 + frac` with `q_lo ∈ 0..=9`, `frac ∈ 0..=63`, plus the
  saturation endpoint `(q_lo = 9, frac = 64)` representing
  `q' = 10.0`); `STATIC_ALLOC_INTERP_RIGHT_SHIFT = 8` (the
  combined `>> 2` Q5→Q3 fold plus `>> 6` Q6 step-weight reduction);
  the typed decomposition `QFpComponents { q_lo, frac }` and the
  invertible `q_fp_to_components(q_fp) -> QFpComponents` /
  `q_fp_from_components(q_lo, frac) -> u32` accessors;
  `per_band_eighth_bits_at_q_fp(band, q_fp, channels, n_bins, lm) ->
  u64` returning the per-band Q3 allocation under the linear
  interpolation `cell_q11 = alloc[b][q_lo] * (64 - frac) +
  alloc[b][q_lo + 1] * frac` followed by the
  `(channels * N * cell_q11) << LM >> 8` unit conversion;
  `total_eighth_bits_at_q_fp(q_fp, channels, frame_size, is_hybrid)
  -> u64` summing across coded bands while respecting the §4.3
  first-coded-band rule (`0` for CELT-only / `17` for Hybrid);
  `search_q_fp(budget, channels, frame_size, is_hybrid) ->
  AllocSearchOutcome` running the §4.3.3 linear scan from
  `q_fp = Q_FP_MAX` downwards and returning the highest
  `q_fp ∈ 0..=640` whose summed allocation fits the budget;
  `AllocSearchOutcome { q_fp, total_eighth_bits }` carrying the
  chosen quality plus its evaluated total; and `AllocSearchError::
  {ChannelsOutOfRange, QFpOutOfRange, BandOutOfRange}` for
  caller-side bookkeeping errors. Twenty-seven new unit tests pin
  the `Q_FP_MAX = 640` constant + `STATIC_ALLOC_INTERP_RIGHT_SHIFT
  = 8` derived constant; the `(q_lo, frac)` decomposition at
  `q_fp = 0` (`q_lo = 0, frac = 0`), at every integer column
  (`q_fp = q * 64 ⇒ frac = 0`), at the saturation endpoint
  (`q_fp = 640 ⇒ q_lo = 9, frac = 64`), and at a mid-step
  (`q_fp = 352 ⇒ q_lo = 5, frac = 32`); the round-trip identity
  `q_fp_from_components(q_fp_to_components(q_fp)) == q_fp` for
  every `q_fp ∈ 0..=640`; the four invalid `(q_lo, frac)` shapes
  the recomposer rejects (q_lo = 10 with non-zero frac, frac = 64
  at q_lo ≠ 9, frac > 64, q_lo > 10); the per-band parity check
  that `per_band_eighth_bits_at_q_fp(band, q * 64, ...)` exactly
  reproduces the round-39 `static_alloc_eighth_bits(band, q, ...)`
  for every `(band, q, channels, n_bins, lm)` in a representative
  sweep; the saturation parity check that
  `per_band_eighth_bits_at_q_fp(band, Q_FP_MAX, ...)` matches the
  pure column-10 `static_alloc_eighth_bits(band, 10, ...)`; the
  column-zero pin (`per_band_eighth_bits_at_q_fp(_, 0, ...) = 0`);
  the §4.3.3 invariant that per-band allocations are monotone
  non-decreasing in `q_fp` across the full `0..=640` sweep; every
  caller-bookkeeping error path (`BandOutOfRange`,
  `ChannelsOutOfRange`, `QFpOutOfRange`); the total-across-coded-
  bands properties (`total(q_fp = 0) = 0` across every
  `(frame_size, channels, is_hybrid)` combination; monotone-non-
  decreasing-in-`q_fp` total; CELT-only total exceeds Hybrid at
  saturation; stereo total is bounded by `2 * mono` and
  `2 * mono + 21` to capture the per-band `>> 8` rounding slack);
  and the search behaviour (`budget = 0` ⇒ `q_fp = 0`,
  `budget = u64::MAX` ⇒ `q_fp = Q_FP_MAX`, exact-target probes
  return at least the target quality; one-less-than-target probes
  fall strictly below; the self-consistency invariant that the
  returned total recomputes to the same value AND if
  `q_fp < Q_FP_MAX` the next step's total strictly exceeds the
  budget). Bridges round 39's `celt_static_alloc` parameter
  surface with the orchestrated §4.3.3 allocator the next round
  will land. Source: RFC 6716 §4.3.3 (pp. 111–112) — held in-repo
  at `docs/audio/opus/rfc6716-opus.txt`.

* **Clean-room round 39 (2026-06-08):** RFC 6716 §4.3.3 *static
  allocation table* (Table 57) — a new `celt_static_alloc` module
  landing the 21-band × 11-quality-column Q5 grid `alloc[band][q]`
  the §4.3.3 *Bit Allocation* procedure interpolates over to derive
  each band's static shape allocation. RFC 6716 §4.3.3 (p. 111)
  describes the conversion as
  `channels * N * alloc[band][q] << LM >> 2`, where the result is
  in 1/8 bits (the same units as every other §4.3.3 budget
  quantity). New public surface: `STATIC_ALLOC: [[u8; 11]; 21]`
  reproducing the 231 cells of Table 57, the layout / conversion
  constants `STATIC_ALLOC_Q_COUNT = 11`, `STATIC_ALLOC_Q_MIN = 0`,
  `STATIC_ALLOC_Q_MAX = 10`, `STATIC_ALLOC_TOTAL_CELLS = 231`,
  `STATIC_ALLOC_RIGHT_SHIFT = 2`, `STATIC_ALLOC_INTERP_STEPS = 64`,
  the typed accessors `static_alloc_cell(band, q) -> u8` and
  `static_alloc_row(band) -> &[u8; 11]` for raw Q5 lookups, the
  `static_alloc_eighth_bits(band, q, channels, n_bins, lm) -> u32`
  conversion folding the per-band scale `(channels * N) << LM >> 2`
  into the Q5-to-Q3 unit fold, and `StaticAllocError::{BandOutOfRange,
  QualityOutOfRange, ChannelsOutOfRange, LmOutOfRange}` for caller-side
  bookkeeping errors. Twenty-eight new unit tests pin the table shape
  (21 × 11 = 231 cells; column 0 uniformly zero; column 10 at 200 for
  bands 0..=7 then declining to 104 at band 20), the
  monotone-non-decreasing-in-`q` invariant the §4.3.3 search depends
  on, hand-picked corner cells (band 0 / q 1 / q 10; band 13 / q 1 / q
  2; band 20 / q 5..=9), worked-example traces of the `<< LM >> 2`
  unit conversion at LM = 0 and LM = 3, the `<< LM` doubling property
  across the four CELT frame sizes, a cross-check against the round-24
  Table 55 band-width lookup, and every out-of-range guard. Bridges
  the round-31 cap surface (`celt_cache_caps50`), the round-33
  band-boost decoder, the round-34 reservation block, the round-35
  per-band minimum threshold, and the round-36 trim offsets with the
  §4.3.3 1/64-step interpolated search the next round will land.
  Source: RFC 6716 §4.3.3 Table 57 (pp. 111–112) — held in-repo at
  `docs/audio/opus/rfc6716-opus.txt`.

* **Clean-room round 38 (2026-06-07):** RFC 6716 §4.5.3 *Summary of
  Transitions* (Figure 18 + Figure 19) — a new `celt_transitions`
  module that closes the §4.5 chain after the round-26 §4.5.1
  redundancy side information, the round-28 §4.5.1.4 cross-lap
  placement, and the round-27 §4.5.2 state-reset policy. §4.5.3
  enumerates the nine *normative* transition shapes (Figure 18)
  the encoder is allowed to use, plus seven *recommended
  non-normative* shapes (Figure 19) for transitions without
  redundancy where PLC is allowed. New public surface:
  `NormativeTransition` with one variant per Figure-18 row,
  `RecommendedNonNormativeTransition` with one variant per
  Figure-19 row, `BoundaryOp` lifting the figure-key markers (`;`,
  `|`, `!`, `&`, `+`, `c`, `P`, `>`) to a typed list,
  `classify_normative_transition(prev_mode, prev_silk_bw,
  next_mode, next_silk_bw, redundancy_present) ->
  Option<NormativeTransition>` for Figure-18 lookup, and
  `recommended_non_normative(prev_mode, prev_silk_bw, next_mode,
  next_silk_bw) -> Option<RecommendedNonNormativeTransition>` for
  Figure-19 lookup. Each enum exposes a `seam_operations() ->
  &'static [BoundaryOp]` accessor returning the ordered §4.5.3
  marker list at the transition seam. The classifier bakes in the
  SILK-bandwidth split between Figure-18 rows 2 ("NB or MB SILK to
  Hybrid with Redundancy") and 3 ("WB SILK to Hybrid"), the
  symmetric Hybrid → SILK split (rows 5 and 6), and the §4.5
  first-paragraph "audio-bandwidth change is the glitch source"
  reading that rules out same-bandwidth SILK→SILK from row 1.
  Forty-two new unit tests pin every Figure-18 and Figure-19 row,
  the SILK-bandwidth splits, the §4.5 first-paragraph "no special
  treatment" exemption for same-configuration CELT→CELT and
  Hybrid→Hybrid pairs, the seam-op ordering per row, and a
  cross-check that the §4.5.3 figure-reset markers agree with the
  §4.5.2 state-reset policy already encoded in
  `mode_transition_reset`. Source: RFC 6716 §4.5.3 (pp. 128–130) —
  held in-repo at `docs/audio/opus/rfc6716-opus.txt`.

* **Clean-room round 37 (2026-06-07):** RFC 6716 *Appendix B
  self-delimiting framing* — a new `framing_self_delim` module wires
  up the alternate Opus framing that prefixes one or two extra
  §3.2.1 length fields so a transport can pack multiple Opus streams
  back to back (the appendix's worked example is the multi-channel
  case formed from several one- or two-channel Opus streams). The
  parser handles all five Appendix-B shapes — Figure 25 (code 0,
  one Opus frame, one length), Figure 26 (code 1, two equal-size
  frames, one length), Figure 27 (code 2, two length fields: the
  §3.2.4 inline length for the first frame and the Appendix-B
  extra length for the second), Figure 28 (CBR code 3, the
  Appendix-B length applies to every frame; padding chain handled
  per §3.2.5), and Figure 29 (VBR code 3, `M − 1` §3.2.1 inline
  lengths plus the Appendix-B trailing length for frame M). New
  public surface: `parse_self_delimited(buffer) -> Result<
  SelfDelimitedParse<'_>, Error>` returns the parsed
  `OpusPacket<'_>` plus a `consumed` byte count so a multistream
  demuxer can advance exactly one packet at a time (`buffer[
  consumed..]` is the next stream's TOC byte). Frame slices borrow
  the buffer just like [`OpusPacket::parse`] — the two construction
  paths are interchangeable downstream. All Appendix-B malformation
  conditions are bundled into [`Error::MalformedPacket`] / [`Error::
  EmptyPacket`]: zero-byte buffer, truncated length, truncated
  padding chain, frame larger than [`MAX_FRAME_BYTES`] (= 1275, the
  §3.2.1 cap), `M = 0` or `M > MAX_FRAMES_PER_PACKET` (= 48), or
  any frame / padding payload running past the buffer end. Unlike
  [`OpusPacket::parse`] (which consumes the entire input as one
  packet), this entry point consumes only the bytes its lengths
  describe — that's the whole point of the Appendix-B variant.
  Reuses the §3.2.1 length decoder via a crate-private re-export
  (`frames::decode_length`) so the two framing modes share one
  length-encoding implementation. 17 new tests cover each of the
  five figures, multistream chaining (parse, advance, parse), and
  the seven malformation paths. Source: RFC 6716 Appendix B
  (September 2012), pp. 321–325 — held in-repo at
  `docs/audio/opus/rfc6716-opus.txt`.

* **Clean-room round 36 (2026-06-07):** §4.3.3 *per-band allocation-trim
  offsets* — a new `celt_trim_offsets` module delivering the §4.3.3
  `trim_offsets[]` per-band tilt vector that biases the §4.3.3 Table 57
  static-allocation search after the round-35
  [`band_min_thresh`] floor is applied. RFC 6716 §4.3.3 (p. 115)
  specifies the formula: for each coded band `b`, with `channels ∈ {1, 2}`,
  `LM ∈ {0, 1, 2, 3}` (the §4.3 frame-size scale),
  `n_shortest = celt_band_bins_per_channel(b, Ms2_5)` (Table 55 column 0
  — the shortest §4.3 frame size for the standard CELT mode),
  `n_per_channel = celt_band_bins_per_channel(b, frame_size)`, and
  `remaining_bands` the band-position-dependent factor,
  `base = (alloc_trim - 5 - LM) * channels * n_shortest * remaining_bands
  * (1 << LM) * 8 / 64`, then `trim_offsets[b] = base - 8 * channels`
  when `n_per_channel == 1` (width-1 bands receive greater benefit from
  the coarse-energy coding, so the §4.3.3 narrative backs the trim off by
  one whole bit per channel for them). All arithmetic is signed (the
  `(alloc_trim - 5 - LM)` factor reaches `-8` at the lowest trim with the
  largest frame size); the output is in 1/8 bits, the same units the
  §4.3.3 budget loop works in. The "number of remaining bands" choice is
  deferred to the consumer site (the round that lands the §4.3.3 Table 57
  static-allocation search): the RFC narrative phrases it as a
  per-band-iteration quantity, and this module accepts `remaining_bands`
  as an explicit caller-supplied parameter. New public surface:
  `band_trim_offset(alloc_trim, lm, is_stereo, n_shortest, n_per_channel,
  remaining_bands) -> Result<i32, TrimOffsetError>` per-band primitive
  (validates `alloc_trim ≤ ALLOC_TRIM_MAX`); `band_trim_offset_for_band(
  band, alloc_trim, frame_size, is_stereo, remaining_bands) ->
  Result<i32, TrimOffsetError>` convenience that derives `n_shortest`
  and `n_per_channel` from the round-24 Table 55 layout; `band_n_shortest
  (band) -> Option<u16>` Table-55 column-0 lookup helper;
  `shortest_frame_size() -> CeltFrameSize` returning `Ms2_5` for the
  standard §4.3 mode; formula constants `TRIM_OFFSETS_BIAS = 5`
  (§4.3.3 "subtract 5"), `TRIM_OFFSETS_NUMERATOR_SCALE = 8` (§4.3.3
  "multiply by 8"), `TRIM_OFFSETS_DIVISOR = 64` (§4.3.3 "divide by 64"),
  `TRIM_OFFSETS_WIDTH_ONE_BINS_PER_CHANNEL = 1` (§4.3.3 width-1 trigger),
  `TRIM_OFFSETS_WIDTH_ONE_PER_CHANNEL_EIGHTH_BITS = 8` (§4.3.3
  per-channel subtraction = one whole bit), and
  `TRIM_OFFSETS_MONO_CHANNELS = 1` / `TRIM_OFFSETS_STEREO_CHANNELS = 2`
  channel multipliers matching the round-35
  `BAND_THRESH_{MONO,STEREO}_CHANNELS` pins; error variants
  `TrimOffsetError::{AllocTrimOutOfRange{provided, max}, BandOutOfRange
  {band}}` for caller-side bookkeeping bugs. Forty-two new unit tests
  (751 lib tests total, up from 709 at round-35 close; 20 integration
  tests unchanged, grand total 771) cover: the seven §4.3.3 formula
  constants pinned to their narrative sources; the constant cross-check
  `TRIM_OFFSETS_BIAS == ALLOC_TRIM_DEFAULT == 5` (the §4.3.3 trim
  default cancels the multiplicative kernel at `LM = 0`); the
  net-scale-keeps-Q3-units invariant (`8 / 64 = 1/8`); the
  `band_n_shortest` Table-55-column-0 path at three pin cells
  (band 0 ⇒ 1, band 20 ⇒ 22, band 21 ⇒ `None`); the
  `band_n_shortest`-matches-`celt_band_bins_per_channel(_, Ms2_5)`
  cross-check over the full 21 bands; the §4.3.3 single-band formula at
  six worked points (default-trim / LM 0 / no-width-1 ⇒ 0; default-trim
  / LM 0 / width-1 mono ⇒ -8; default-trim / LM 0 / width-1 stereo ⇒
  -16; max-trim / LM 0 / large factors ⇒ +577; min-trim / LM 3 / large
  factors mono ⇒ -3 696; min-trim / LM 3 / width-1 stereo ⇒ -352); the
  LM-factor-doubles cross-check at four LMs (40 → 64 → 96 → 128 with
  `trim_term` adjusted per LM); the channel-factor-scales-linearly
  invariant; the n_shortest-scales-linearly invariant; the
  remaining_bands-scales-linearly invariant; the
  `remaining_bands = 0` case reducing to the width-1 correction only;
  the `trim_term = 0` kernel-cancel path over many factor combinations
  (with the width-1 correction still firing where applicable); the
  `alloc_trim - 5 == LM` kernel-cancel pin at `(alloc_trim=8, LM=3)`;
  the width-1 subtraction is purely additive (the difference between
  `n_per_channel = 1` and `n_per_channel ≥ 2` is exactly `8 *
  channels`); the width-1 trigger fires only at `n_per_channel == 1`
  (verified at `{0, 2, 3, 22, 176}` exclusion edges); the
  truncating-toward-zero integer-division behaviour at three numerator
  cells (`-512 / 64 = -8` exact, `-8 / 64 = 0` truncating-positive-zero,
  `-80 / 64 = -1` truncating-toward-zero); the
  `alloc_trim > ALLOC_TRIM_MAX` error path at the boundary (11) and
  far above (255); the `alloc_trim ∈ {0, ALLOC_TRIM_MAX}` accepted-edge
  cases; the `band_trim_offset_for_band` Table-55 wrapper rejecting
  `band ≥ CELT_NUM_BANDS`; the wrapper matching the primitive's output
  over the full 21 × 4 × 2 (band × frame-size × stereo) matrix; the
  wrapper propagating `AllocTrimOutOfRange`; the wrapper width-1
  trigger at Table-55 band 0 / 2.5 ms (N = 1); the wrapper width-1
  inactive at band 20 / 20 ms (N = 176 ⇒ result = -1 386); a
  determinism sweep over five `alloc_trim` values × four frame sizes ×
  21 bands × 2 channels × 4 `remaining_bands` values; the
  output-fits-well-within-`i32` guarantee at the worst-case input
  edges; and `Debug` rendering for both error variants. The §4.3.3
  Table 57 static-allocation search that consumes `trim_offsets[]`
  (against the round-31 `cap[]` per-band maximum, the round-33
  `boosts[]`, and the round-35 `thresh[]` floor) is the responsibility
  of the §4.3.3 allocator and runs in a downstream round.

* **Clean-room round 35 (2026-06-06):** §4.3.3 *per-band minimum-allocation
  vector* — a new `celt_band_thresh` module delivering the §4.3.3
  `thresh[band]` lower bound used by the §4.3.3 Table 57 static-allocation
  search to drop low-rate bands rather than code them sparsely. RFC 6716
  §4.3.3 (p. 115) specifies the formula: for each coded band `b`, with
  `N = celt_band_bins_per_channel(b, frame_size)` (Table 55) and
  `channels ∈ {1, 2}`, `thresh[b] = max((24 * N) / 16, 8 * channels)` in
  1/8 bits — one whole bit per channel or 48 128th-bits per MDCT bin,
  whichever is greater. The §4.3.3 narrative is explicit that the
  band-size dependent term `(24 * N) / 16` is *not* scaled by the
  channel count: at the very low rates where this floor binds, the
  §4.3.3 allocator concentrates the budget on the mid channel. New
  public surface: `band_min_thresh(band, frame_size, is_stereo) ->
  Option<u32>` per-band lookup (`None` when `band ≥ 21`);
  `compute_band_min_thresh(start, end, frame_size, is_stereo, &mut
  thresh)` in-place vector fill over the §4.3 coding window `start..end`
  (`0..21` CELT-only, `17..21` Hybrid); `band_min_thresh_vec(start,
  end, frame_size, is_stereo) -> Result<Vec<u32>, BandThreshError>`
  allocating convenience; `standard_band_window(is_hybrid) -> (usize,
  usize)` helper producing the §4.3 full-frame window from
  [`crate::celt_band_layout::celt_first_coded_band`] +
  [`crate::celt_band_layout::celt_end_coded_band`]; formula constants
  `BAND_THRESH_BINS_MULTIPLIER = 24` (§4.3.3 "24 times the number of MDCT
  bins"), `BAND_THRESH_BINS_DIVISOR = 16` (§4.3.3 "divide by 16"),
  `BAND_THRESH_PER_CHANNEL_EIGHTH_BITS = 8` (§4.3.3 "8 times the number
  of channels"), `BAND_THRESH_MONO_CHANNELS = 1` /
  `BAND_THRESH_STEREO_CHANNELS = 2` channel multipliers; error variants
  `BandThreshError::{InvertedBandWindow{start, end},
  BandWindowOutOfRange{end}, OutputBufferTooSmall{expected, provided}}`
  for caller-side bookkeeping bugs (the band layout itself is total over
  `band < 21`; an out-of-range window cannot come from a corrupt
  bitstream). Thirty-eight new unit tests (709 lib tests total, up from
  671 at round-34 close; 20 integration tests unchanged, grand total
  729) cover: the five §4.3.3 constants pinned to their narrative
  sources; the channel-multiplier sanity (mono = 1, stereo = 2); the
  §4.3.3 per-band formula at three key cells (band 0 / 2.5 ms / mono ⇒
  channel-term wins ⇒ `thresh = 8`; band 0 / 20 ms / mono ⇒ bin-term
  wins ⇒ `thresh = 12`; band 20 / 20 ms ⇒ bin-term dominates ⇒ `thresh
  = 264` independent of channel count); the `band ≥ 21` `None` path
  (Custom mode out of scope); a full cross-check that the function
  equals `max((24 * N) / 16, 8 * channels)` for every (band, frame_size,
  channels) triple over the standard 21-band × 4-frame-size × 2-channel
  matrix; the §4.3.3 "not scaled by channel count" invariant at the
  bin-term-dominated cell (band 20 / 20 ms: mono = stereo = 264); the
  channel-term-doubles-with-stereo behaviour at the channel-term-
  dominated cell (band 0 / 2.5 ms: stereo = 2 × mono); the full
  CELT-only window (`0..21`) and the §4.3 Hybrid window (`17..21`)
  driver paths; a partial CELT-only NB 2.5 ms window (`0..13`) where
  every band hits the channel-term floor (= 8 mono); the
  `band_min_thresh_vec` allocator agreeing with the slice form; the
  three `BandThreshError` paths (inverted window, end past 21, output
  buffer mismatched length); the empty-window success case; the §4.3.3
  "at least one whole bit per channel" lower-bound invariant across
  every (band, frame_size) cell for both channel counts; the
  thresh-monotonic-in-frame-size invariant at fixed band (driven by N
  doubling across each Table 55 column); the
  `stereo ≥ mono` invariant; a units cross-check pinning the §4.3.3
  "48 128th bits per MDCT bin" wording to `(24 * N) / 16` 1/8 bits;
  two Table 55 cell pins (band 8 / 20 ms / stereo: N = 16, bin_term
  = 24, channel_term = 16 ⇒ thresh = 24; band 20 / 2.5 ms: N looked
  up via [`celt_band_bins_per_channel`] and formula reconciled); the
  `standard_band_window` helper at CELT-only and Hybrid; an
  integration test threading `standard_band_window(true)` ⇒
  `band_min_thresh_vec` ⇒ stereo-floor invariant; determinism across
  repeats; and `Debug` rendering for the error type. The §4.3.3
  Table 57 static-allocation search that consumes `thresh[]` (against
  the round-31 `cap[]` per-band maximum and the upcoming
  `trim_offsets[]` per-band tilt) is the responsibility of the
  §4.3.3 allocator and runs in a downstream round.

* **Clean-room round 34 (2026-06-04):** §4.3.3 *reservation block* —
  a new `celt_reservations` module delivering the §4.3.3 fixed-cost
  preamble that runs after the §4.3.3 band-boost loop (round 33) and
  the §4.3.3 allocation-trim decode (round 32) but before the §4.3.3
  Table 57 static-allocation search. RFC 6716 §4.3.3 (p. 114)
  specifies four reservations skimmed off the working `total`
  budget: `anti_collapse_rsv` (8 1/8 bits iff transient && `LM > 1`
  && `total ≥ (LM + 2) * 8`), `skip_rsv` (8 1/8 bits iff `total > 8`
  after anti-collapse), `intensity_rsv` (stereo only; equal to
  `LOG2_FRAC_TABLE[end − start]` from round 30 except reset to 0 if
  greater than `total`), and `dual_stereo_rsv` (stereo only, 8 1/8
  bits iff `total > 8` after the intensity-stereo deduction; only
  considered when intensity_rsv was successfully reserved). The
  §4.3.3 working `total` starts at `frame_size_bytes * 64 −
  ec_tell_frac − 1` (the trailing `-1` is the §4.3.3 conservative
  deduction). New public surface: `reserve_block(frame_size_bytes,
  ec_tell_frac, total_boost, lm, is_transient, is_stereo,
  coded_bands) -> Result<ReservationOutcome, ReservationError>`
  pure-function evaluator (`lm` typed `CeltFrameSize`,
  `coded_bands` = `end − start` for the §4.3 band-coding window);
  `ReservationOutcome { anti_collapse_rsv, skip_rsv, intensity_rsv,
  dual_stereo_rsv, total_remaining_eighth_bits }` typed outcome with
  `reserved_total_eighth_bits()` summing helper; reservation-cost
  constants `ONE_BIT_EIGHTH_BITS = 8` (anti-collapse / skip /
  dual-stereo) and `CONSERVATIVE_DEDUCTION_EIGHTH_BITS = 1`;
  anti-collapse gating constants `ANTI_COLLAPSE_LM_MIN_EXCLUSIVE =
  1` (strict `LM > 1` floor) and
  `ANTI_COLLAPSE_HEADROOM_MULT_EIGHTH_BITS = 8` /
  `ANTI_COLLAPSE_HEADROOM_LM_OFFSET = 2` (the `(LM + 2) * 8`
  1/8-bit headroom test); the module-local `EIGHTH_BITS_PER_BYTE =
  64` mirror of `celt_alloc_trim::EIGHTH_BITS_PER_BYTE`; error
  variants `ReservationError::{FrameSizeOverflows, TellExceedsFrame
  { frame_eighth_bits, ec_tell_frac }, TotalBoostExceedsFrame {
  frame_eighth_bits, ec_tell_frac, total_boost },
  LogFracLookupFailed(Log2FracError) }` for caller-side bookkeeping
  bugs (the range coder's sticky error flag is the right channel
  for a corrupt bitstream signal); `From<Log2FracError>` round-trip
  helper. Forty-one new unit tests (671 lib tests total, up from
  630 at round-33 close; 20 integration tests unchanged, grand total
  691) cover: the five RFC constants pinned to their narrative
  sources; the `EIGHTH_BITS_PER_BYTE` agreement with
  `celt_alloc_trim`; the `CeltFrameSize::column_index() → LM`
  cross-check at every frame size; the four anti-collapse predicate
  paths (non-transient ⇒ no rsv; LM ∈ {0, 1} ⇒ no rsv even with
  transient; LM = 2 / LM = 3 with budget ⇒ rsv = 8); the §4.3.3
  anti-collapse threshold inequality at exact match and one short;
  the §4.3.3 skip gate at `total = 8` (rsv = 0) and `total = 9`
  (rsv = 8); a strict-ordering check that the anti-collapse
  deduction precedes the skip gate; the mono branch skipping all
  stereo reservations even with budget; the stereo
  intensity-reset-on-overflow path with `dual_stereo_rsv = 0`
  follow-on; the stereo intensity-just-fits path with
  `dual_stereo_rsv ∈ {0, 8}` depending on the remaining budget vs
  the `total > 8` gate; the §4.3 Hybrid 4-band window producing
  `intensity_rsv = 19` from `LOG2_FRAC_TABLE[4]`; the `coded_bands
  ∈ {0, 1}` boundary cells; the §4.3.3 invariant `total_remaining +
  reserved = frame_eighth − ec_tell − 1` across mono / stereo /
  transient / non-transient / nonzero-tell permutations; the four
  `ReservationError` paths; the mono short-circuit on out-of-range
  `coded_bands` (intensity-stereo lookup is *not* attempted for
  mono frames, so the input is harmless); the zero-byte and
  one-byte frame edge cases; the §3.4 R5 1275-byte max-frame
  headroom assertion with every reservation at its maximum (8 + 8 +
  37 + 8 = 61, total_remaining = 81538); the
  `ReservationOutcome::default()` all-zero pattern; determinism
  across repeats; debug formatting; and the `From<Log2FracError>`
  round-trip. The §4.3.3 *use* of the reservations — the actual
  `dec_bit_logp(1)` reads of the anti-collapse / skip /
  dual-stereo flags and the `ec_dec_uint(end − start)` read of the
  intensity-stereo band — runs at the §4.3.3 allocator's consumer
  site once the Table 57 search produces the per-band shape
  allocation; the per-band `trim_offsets[]` derivation that biases
  the Table 57 search is the responsibility of the §4.3.3 allocator
  and runs in a downstream round.

* **Clean-room round 33 (2026-06-04):** §4.3.3 *band-boost* decoder —
  a new `celt_band_boost` module delivering the §4.3.3 band-boost
  decode loop (RFC 6716 §4.3.3, pp. 113–114) — the third §4.3.3
  fragment after rounds 30 and 31's `LOG2_FRAC_TABLE` /
  `CACHE_CAPS50` parameter surfaces, and the structural bridge that
  takes round 31's `cap[]` per-band upper bound and round 32's
  `decode_alloc_trim` gate's `total_boost` input and ties them
  together at the §4.3.3 control-flow site. The §4.3.3 narrative is:
  *"To decode the band boosts: First, set 'dynalloc_logp' to 6 […]
  'total_bits' to the size of the frame in 8th bits, 'total_boost'
  to zero […]. For each band […] the boost quanta in units of 1/8
  bit is calculated as `quanta = min(8*N, max(48, N))`. […] Set
  'boost' to zero and 'dynalloc_loop_logp' to dynalloc_logp. While
  dynalloc_loop_logp […] in 8th bits plus tell is less than
  total_bits plus total_boost and boost is less than `cap[]` for
  this band: Decode a bit […] update tell […] If the decoded value
  is zero break the loop. Otherwise, add quanta to boost and
  total_boost, subtract quanta from total_bits, and set
  dynalloc_loop_log to 1. […] If boost is non-zero and dynalloc_logp
  is greater than 2, decrease dynalloc_logp."* New public surface:
  shape constants `DYNALLOC_LOGP_INIT = 6` (§4.3.3 initial
  first-boost cost in whole bits), `DYNALLOC_LOGP_MIN = 2` (§4.3.3
  first-boost floor), `DYNALLOC_LOOP_LOGP_AFTER_FIRST = 1` (§4.3.3
  second-and-subsequent-bits cost within a single band's loop),
  `BAND_BOOST_QUANTA_FLOOR_EIGHTH_BITS = 48` (§4.3.3 `max(48, N)`
  floor), and `BAND_BOOST_QUANTA_CEIL_MULT = 8` (§4.3.3 `8*N`
  1-bit/sample ceiling multiplier); the `band_boost_quanta(n)` pure
  helper computing `min(8*n, max(48, n))` over `u32`; the
  `decode_band_boosts(rd, start, end, caps, n_bins,
  frame_size_bytes) -> Result<BandBoostOutcome, BandBoostError>`
  driver walking the §4.3 `start..end` coding window (`0..end`
  normally, `17..end` in Hybrid mode) and running the §4.3.3
  per-band inner loop against caller-supplied `caps[band - start]`
  and `n_bins[band - start]` slices; the
  `BandBoost { boost_eighth_bits, bits_read }` per-band outcome
  struct; the `BandBoostOutcome { per_band, total_boost_eighth_bits,
  total_bits_remaining_eighth_bits, dynalloc_logp_final }` full
  driver outcome bundling the §4.3.3 `total_boost` accumulator
  consumed by `celt_alloc_trim::decode_alloc_trim` downstream; and
  error variants `BandBoostError::{CapsLengthMismatch{expected,
  provided}, NBinsLengthMismatch{expected, provided},
  EmptyBandWindow{start, end}, InvertedBandWindow{start, end}}`
  for caller-side bookkeeping bugs that don't belong to the range
  coder's sticky error flag. Thirty-seven new unit tests (630 lib
  tests total, up from 593 at round-32 close; 20 integration tests
  unchanged, grand total 650) cover: the five §4.3.3 RFC constants
  pinned to their narrative sources; the §4.3.3
  `quanta = min(8*N, max(48, N))` rule sampled at `N = 48`
  (boundary), `N > 48` (linear regime), `6 ≤ N < 48` (floor regime),
  `N < 6` (ceiling regime), `N = 0` (degenerate), and as a total
  function over every `u16`; the four `BandBoostError` paths with
  the range coder's `tell_frac()` unchanged across error returns;
  the no-room-for-any-boost path (`frame_size_bytes = 0`) returning
  all-zero boosts and the §4.3.3 invariant
  `total_bits + total_boost = frame_eighth_bits` holding at zero;
  the stop-bit-biased payload (`[0x00; 64]` whose §4.1.1 init
  `val = 127 - (b0 >> 1) = 127` biases `dec_bit_logp` toward the
  §4.3.3 stop branch) decoding zero boosts with `bits_read = 1`
  per band and the §4.3.3 `dynalloc_logp_final =
  DYNALLOC_LOGP_INIT` no-decrement rule; the boost-bit-biased
  payload (`[0xFF; 64]` ⇒ `val = 0`) actually boosting at least one
  band and decrementing `dynalloc_logp` below its initial value;
  the `per_band` vector alignment with the `start..end` window
  (including the §4.3 Hybrid `17..21` four-band window); the §4.3.3
  invariant `total_bits + total_boost = frame_size_bytes * 64`
  conserved across both the stop and boost paths; the §4.3.3
  `dynalloc_logp` cross-band floor at `DYNALLOC_LOGP_MIN`; the
  `boost = 0` short-circuit on a `cap = 0` band (no range-coder
  bits read); the `BandBoostOutcome` debug / equality / determinism
  cross-check on identical runs; and the §3.4 R5 `1275 * 64`
  max-frame headroom assertion. The §4.3.3 *use* of the per-band
  `boost` values — the per-band shape-allocation adjustment that
  feeds into the §4.3.3 Table 57 static-allocation search and the
  §4.3.3 anti-collapse / skip / dual-stereo reservations — is the
  responsibility of the §4.3.3 allocator and runs at the call site
  of `decode_band_boosts`.

* **Clean-room round 32 (2026-06-03):** §4.3.3 *allocation trim*
  parameter surface — a new `celt_alloc_trim` module delivering the
  Table-58 PDF of RFC 6716 §4.3.3 (p. 115) plus the §4.3.3
  signalling-gate predicate (RFC 6716 §4.3.3 p. 114) and the typed
  decode wrapper that fuses the two. The §4.3.3 narrative reads:
  *"To decode the trim, first set the trim value to 5, then if and
  only if the count of decoded 8th bits so far (ec_tell_frac) plus
  48 (6 bits) is less than or equal to the total frame size in 8th
  bits minus total_boost […], decode the trim value using the PDF
  in Table 58."* New public surface: `ALLOC_TRIM_PDF: [u8; 11]`
  (the 11-cell Table-58 PDF `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2,
  2}/128`) and the derived `ALLOC_TRIM_ICDF: [u8; 11]` (`[126, 124,
  119, 109, 87, 41, 19, 9, 4, 2, 0]`) for direct
  `RangeDecoder::dec_icdf` consumption; shape constants
  `ALLOC_TRIM_PDF_LEN = 11`, `ALLOC_TRIM_FTB = 7`,
  `ALLOC_TRIM_PDF_DENOMINATOR = 128`; trim-integer range constants
  `ALLOC_TRIM_DEFAULT = 5`, `ALLOC_TRIM_MIN = 0`,
  `ALLOC_TRIM_MAX = 10` per the RFC's "an integer value from 0-10"
  and "the default value of 5 indicates no trim" wording;
  signalling-cost constants `ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS =
  48` (6 whole bits at 1/8-bit precision) and
  `EIGHTH_BITS_PER_BYTE = 64`; the §4.3.3 signalling-gate
  predicate `alloc_trim_is_signalled(ec_tell_frac, frame_eighth_bits,
  total_boost) -> bool` evaluating
  `(ec_tell_frac + 48) ≤ (frame_eighth_bits − total_boost)` with
  saturating arithmetic for the malformed-input edge cases; the
  byte-to-1/8-bit conversion helper
  `frame_eighth_bits(frame_size_bytes) -> Result<u32,
  AllocTrimError>` with `u32` overflow rejection; the composite
  decode wrapper `decode_alloc_trim(rd, ec_tell_frac,
  frame_size_bytes, total_boost) -> Result<u8, AllocTrimError>`
  fusing the gate evaluation, the gate-fail-returns-`5` rule, and
  the `RangeDecoder::dec_icdf(&ALLOC_TRIM_ICDF, 7)` read into one
  typed call; full-table borrows `alloc_trim_pdf()` and
  `alloc_trim_icdf()`; error variants
  `AllocTrimError::{FrameSizeOverflows, TotalBoostExceedsFrame {
  frame_eighth_bits, total_boost }}` for caller-side bookkeeping
  bugs. Thirty-three new unit tests (593 lib tests total, up from
  560 at round-31 close; 20 integration tests unchanged, grand
  total 613) pin the shape constants, the Table 58 PDF cells
  against the RFC body, the PDF sum-to-denominator and
  symmetric-around-default invariants, the heaviest-mass-at-default
  cell pin, the iCDF strict-monotone-decreasing invariant, the
  iCDF-from-PDF derivation, spot iCDF cells, the
  `frame_eighth_bits` scaling and overflow rejection, the §4.3.3
  signalling gate at the six-bit boundary (`ec_tell_frac + 48 =
  budget` passes, one over fails) and across the total_boost / no
  room / underflow / `u32` overflow edge cases, the
  `decode_alloc_trim` gate-fail returns `5` and consumes no
  range-coder bits, the gate-pass returns an in-range value and
  advances `tell_frac`, the error paths leave the range coder
  untouched, and the worst-case-symbol-cost-matches-gate-budget
  math (`log2(128/2) = 6` whole bits = 48 1/8 bits). The §4.3.3
  *use* of the trim — the per-band `trim_offsets[]` derivation
  that biases the Table 57 static allocation search — runs at the
  call site of `decode_alloc_trim` and is out of scope for this
  parameter surface.

* **Clean-room round 31 (2026-06-03):** §4.3.3 per-band
  maximum-allocation parameter surface — a new `celt_cache_caps50`
  module delivering the `CACHE_CAPS50` lookup piece of RFC 6716 §4.3.3
  (pp. 113–114): the 168-byte bits/sample table the §4.3.3 per-band
  bit cap `cap[band] = ((cache_caps50[i] + 64) * channels * N) / 4`
  consumes (named `init_caps()` in RFC 6716 §4.3.3 p. 114). Round 24
  noted the §4.3.3 allocator as blocked on `cache_caps50` +
  `LOG2_FRAC_TABLE`; round 30 landed `LOG2_FRAC_TABLE`, and this round
  closes that pair by landing `CACHE_CAPS50` plus the `init_caps()`
  convert-rule helpers. New public surface: `CACHE_CAPS50: [u8; 168]`
  (168 Q0 bytes; layout `[LM ∈ 0..4][stereo ∈ {0,1}][band ∈ 0..21]`
  flattened by the §4.3.3 `i = nbBands * (2*LM + stereo) + band` rule
  with `nbBands = 21`, matching
  `docs/audio/celt/tables/cache_caps50.csv` row-for-row); shape
  constants `CACHE_CAPS50_LM_COUNT = 4`,
  `CACHE_CAPS50_STEREO_COUNT = 2`, `CACHE_CAPS50_TOTAL_BYTES = 168`;
  stereo-axis index constants `CACHE_CAPS50_STEREO_MONO = 0`,
  `CACHE_CAPS50_STEREO_STEREO = 1`; convert-rule constants
  `INIT_CAPS_BIAS = 64`, `INIT_CAPS_DIVISOR = 4`,
  `INIT_CAPS_MAX_CHANNELS = 2`; typed stereo-axis selector
  `CacheCapsStereo::{Mono, Stereo}` with `axis_index()` /
  `channels()` / `from_is_stereo(bool)` helpers (the `channels()`
  helper turns `Mono → 1` / `Stereo → 2` to feed the `init_caps()`
  multiplier independently of the axis index); typed accessors
  `cache_caps_value(lm, stereo, band) -> Result<u8, CacheCaps50Error>`
  and `cache_caps_row(lm, stereo) -> Result<&'static [u8],
  CacheCaps50Error>` over the lookup; flat-offset helper
  `cache_caps_offset(lm, stereo, band) -> usize` covering the §4.3.3
  row-stride rule; `init_caps(caps_value, channels, n_bins) -> u32`
  computing the §4.3.3 `(value + 64) * channels * N / 4` convert
  rule on a single byte; composite
  `cap_for_band_bits(lm, stereo, band, channels, n_bins) ->
  Result<u32, CacheCaps50Error>` performing lookup-plus-convert in
  one typed call; error variants
  `CacheCaps50Error::{LmOutOfRange, BandOutOfRange,
  ChannelsOutOfRange}`. Twenty-nine new unit tests (560 lib tests
  total, up from 531 at round-30 close; 20 integration tests
  unchanged, grand total 580) pin the table shape, the
  `INIT_CAPS_BIAS = 64` / `INIT_CAPS_DIVISOR = 4` /
  `INIT_CAPS_MAX_CHANNELS = 2` convert-rule constants, the
  `CACHE_CAPS50_STEREO_MONO = 0` / `CACHE_CAPS50_STEREO_STEREO = 1`
  axis indices, the `CacheCapsStereo::channels()` `Mono → 1` /
  `Stereo → 2` helper mapping, the `from_is_stereo(bool)` round-trip,
  eight CSV-cell spot-checks at `(row 0, band 0)` / `(row 1, band 20)`
  / `(row 2, band 0)` / `(row 3, band 8)` / `(row 4, band 12)` /
  `(row 5, band 17)` / `(row 6, band 20)` / `(row 7, band 0)`, the
  §4.3.3 `cache_caps_offset()` rule against every `(LM, stereo, band)`
  triple (168 cells) plus its two endpoints, the `cache_caps_value()`
  total-function sweep, the `cache_caps_row()` per-cell mirror, the
  `LmOutOfRange` / `BandOutOfRange` / `ChannelsOutOfRange` error
  paths on both accessors, four §4.3.3 `init_caps()` formula pins
  (including the `(caps=255, channels=2, N=192) → 30624` upper-bound
  cell and the floor-division corner at `caps ∈ {1,2,3}`), a
  `cap_for_band_bits()` composite cross-check against the manual
  lookup-plus-`init_caps()` sequence, the §4.3.3 narrative invariant
  that 20 ms stereo caps fit in `i16` but at least one exceeds
  `i8::MAX`, and two §4.3.3-reachable-cell pins (CELT-only 20 ms
  stereo band 0 → `caps = 204` → `cap = 134 * n_bins`; Hybrid 20 ms
  mono band 17 → `caps = 173` → `cap = (237 * n_bins) / 4`). The
  §4.3.3 bit allocation orchestration that consumes the `cap[]`
  vector (boost / trim / anti-collapse / skip / dual-stereo
  reservations, the Table 57 static allocation search, the
  reallocation / fine-vs-shape split / band-priority computation) is
  out of scope for this round.

* **Clean-room round 30 (2026-06-02):** §4.3.3 intensity-stereo
  reservation parameter surface — a new `celt_log2_frac_table` module
  delivering the `LOG2_FRAC_TABLE` lookup piece of RFC 6716 §4.3.3
  (p. 113): the 24-byte conservative `log2` table (in Q3 / 1/8-bit
  units) the §4.3.3 `intensity_rsv = LOG2_FRAC_TABLE[end − start]`
  reservation consumes. Round 24 noted the §4.3.3 allocator as blocked
  on `cache_caps50` + `LOG2_FRAC_TABLE`; this round delivers the
  smaller of the two table dependencies. New public surface:
  `LOG2_FRAC_TABLE: [u8; 24]` (24 Q3 bytes; layout
  `LOG2_FRAC_TABLE[coded_bands] = conservative_log2(coded_bands)` in
  1/8-bit units, matching `docs/audio/celt/tables/log2_frac_table.csv`
  row-for-row); shape constant `LOG2_FRAC_TABLE_LEN = 24`;
  unit-denominator constant `Q3_BITS_PER_WHOLE_BIT = 8`; typed
  accessor `log2_frac(coded_bands) -> Result<u8, Log2FracError>`
  with the §4.3.3 `coded_bands = end − start` indexing rule and a
  bounds check covering the `coded_bands ≥ 24` case; full-row
  borrow `log2_frac_row() -> &'static [u8; 24]`; error variant
  `Log2FracError::CodedBandsOutOfRange { coded_bands }`. Seventeen
  new unit tests (531 lib tests total, up from 514 at round-29
  close) pin the table shape, the `Q3_BITS_PER_WHOLE_BIT = 8` unit
  constant, seven CSV-row spot-checks at indices 0 / 1 / 2 / 4 / 14
  / 15 / 21 / 23 (covering the §4.3.3 base case, the 1-bit floor,
  the upward-rounded conservative entry, the Hybrid reachable
  index, the 32-byte plateau pair, the CELT-only reachable index,
  and the final entry), a monotone-non-decreasing property across
  every adjacent pair, a conservative-bound property
  `LOG2_FRAC_TABLE[n] ≥ 8 × floor(log2(n))` for every `n ∈ 1..24`
  (leading-zero-count formulation, no floats), a total-function
  sweep over every in-range index, the `CodedBandsOutOfRange` error
  paths, a row-vs-pair cross-check, and two §4.3.3-reachable-index
  sanity pins (CELT-only `end − start = 21` → `36` Q3; Hybrid
  `end − start = 4` → `19` Q3). The rest of the §4.3.3 allocation
  algorithm (anti-collapse / skip / dual-stereo reservations, the
  Table 57 static-allocation search, boost / trim decoding, and the
  `cache_caps50` per-band maximum vector) is out of scope.

* **Clean-room round 29 (2026-06-01):** §4.3.2.1 CELT coarse-energy
  Laplace-model parameter surface — a new `celt_e_prob_model` module
  delivering the parameter-surface piece of RFC 6716 §4.3.2.1
  (pp. 108–109): the per-`(LM, mode, band)` Q8 `{prob, decay}` table
  the §4.3.2.1 `ec_laplace_decode` routine consumes. Round 20's CELT
  pre-band header noted the §4.3.2.1 coarse energy as blocked on this
  table; this round delivers it plus the surrounding selector /
  accessor surface so the Laplace decoder and 2-D `(time, frequency)`
  predictor can be wired up against it next. New public surface:
  `E_PROB_MODEL: [[[u8; 42]; 2]; 4]` (336 Q8 bytes; layout
  `[LM ∈ 0..4][mode ∈ {inter, intra}][band × 2 + {prob, decay}]`,
  matching `docs/audio/celt/tables/e_prob_model.csv` row-for-row);
  shape constants `E_PROB_MODEL_LM_COUNT = 4`,
  `E_PROB_MODEL_MODE_COUNT = 2`, `E_PROB_MODEL_BYTES_PER_BAND = 2`,
  `E_PROB_MODEL_BYTES_PER_ROW = 42`,
  `E_PROB_MODEL_TOTAL_BYTES = 336`; inner-axis index constants
  `E_PROB_MODEL_MODE_INTER = 0`, `E_PROB_MODEL_MODE_INTRA = 1`;
  typed selector `EnergyPredictionMode::{Inter, Intra}` with
  `from_intra_flag(bool)` decode helper and a `table_index()`
  accessor; `EProbPair { prob: u8, decay: u8 }`; typed accessors
  `e_prob_pair(lm, mode, band) -> Result<EProbPair, EProbModelError>`
  and `e_prob_row(lm, mode) -> Result<&'static [u8; 42],
  EProbModelError>`; intra-case prediction-coefficient constants
  `INTRA_PRED_ALPHA_Q15 = 0` and `INTRA_PRED_BETA_Q15 = 4915` against
  `Q15_ONE = 32768` per RFC 6716 §4.3.2.1 p. 108
  (`4915 / 32768 ≈ 0.15`). Twenty-two new unit tests (514 lib tests
  total, up from 492 at round-28 close) pin the table shape, the Q8
  byte values at seven CSV-row spot-checks, the `EnergyPredictionMode`
  mapping, the `LmOutOfRange` / `BandOutOfRange` error paths, a
  total-function sweep over every `(LM, mode, band)` triple
  (4 × 2 × 21 = 168 cells), a pair-vs-row cross-check on every cell,
  and the §4.3.2.1 prediction-effectiveness sanity property
  (`intra_band0_prob < inter_band0_prob` for every LM). The
  §4.3.2.1 Laplace decoder itself, the 2-D `(time, frequency)`
  predictor application, and the §4.3.2.2 fine-energy follow-up are
  out of scope for this module. The per-LM *inter*-mode
  `(alpha, beta)` pair is a §4.3.2.1 docs gap (the RFC names them as
  "depend on the frame size in use" without giving numeric values);
  deferred until the docs side delivers the gap fill.

* **Clean-room round 28 (2026-06-01):** §4.5.1.4 redundant-CELT-frame
  decode parameters and cross-lap placement — a new
  `redundancy_decode_params` module encoding the two normative halves
  of RFC 6716 §4.5.1.4 (pp. 126–127). Half 1: the parameter-derivation
  rule (no TOC byte, 5 ms fixed duration via
  `REDUNDANT_FRAME_TENTHS_MS = 50`, inherited channel count,
  inherited bandwidth with the §4.5.1.4 "MB SILK → WB" exception via
  `apply_mb_to_wb_override`) bundled into
  `RedundantFrameParams { duration_tenths_ms, channels, bandwidth,
  position, size_bytes, cross_lap }`. Half 2: the cross-lap placement
  rule (`CrossLapPlacement::FirstHalfAsIs` for
  `RedundancyPosition::Beginning` — CELT → SILK/Hybrid carriers,
  where the redundant CELT frame's first 2.5 ms replace the carrier's
  leading 2.5 ms and the second 2.5 ms cross-lap;
  `CrossLapPlacement::SecondHalfAsIs` for `RedundancyPosition::End`
  — SILK/Hybrid → CELT carriers, where only the second 2.5 ms is
  used and it cross-laps with the SILK/Hybrid trailing edge).
  `redundant_frame_params(routing, decision)` is the pure-function
  driver entry; `RedundancyDecision::Invalid` (the §4.5.1.3 overflow
  outcome) and `RedundancyDecision::NotPresent` both route to `None`.
  Twenty-five new unit tests (492 lib tests total, up from 467 at
  round-27 close) pin every rule, cross-check four §4.5.3 Figure 18
  transition rows, and sweep the total-function output for the
  MB-bandwidth invariant.

* **Clean-room round 27 (2026-05-31):** §4.5.2 SILK + CELT decoder
  state-reset policy across mode transitions — a new
  `mode_transition_reset` module encoding the four normative rules
  of RFC 6716 §4.5.2 (p. 127) as a pure decision function
  `decide_state_resets(prev_mode, next_mode, redundancy) ->
  StateReset { silk, celt: CeltResetPlacement }`. Rule 1 resets
  SILK on every CELT-only → SILK-only/Hybrid transition; rule 2
  resets CELT on every mode change into Hybrid or CELT-only except
  when redundancy is used; rule 3 places the CELT reset *before
  the redundant CELT frame* on SILK/Hybrid → CELT-only with
  redundancy and skips it before the following CELT-only frame;
  rule 4 suppresses the CELT reset on CELT-only → SILK/Hybrid
  with redundancy. `RedundancyDecision::Invalid` (the §4.5.1.3
  overflow outcome) is treated as no usable redundancy.
  `CeltResetPlacement::{None, BeforeFrame, BeforeRedundantOnly}`
  plus `StateReset::{celt_resets, is_noop}` accessors round out
  the public surface. Twenty-seven new unit tests (467 lib tests
  total) pin every cell of the 3×3 mode-pair × redundancy
  cross-product and cross-check four §4.5.3 Figure 18 transition
  rows.

## [0.0.11](https://github.com/OxideAV/oxideav-opus/compare/v0.0.10...v0.0.11) - 2026-05-30

### Other

- §4.3.4.5 CELT TF-resolution adjustment lookup (round 25)

### Added

* **Clean-room round 26 (2026-05-30):** §4.5.1 CELT redundancy /
  mode-transition side-information decoder — a new
  `celt_redundancy` module owning RFC 6716 §4.5.1.1–§4.5.1.3
  (Tables 64 and 65). Exposes
  `decode_redundancy(rd, mode, opus_frame_bytes) -> RedundancyDecision`
  routing CELT-only Opus frames to a no-op bypass, SILK-only Opus
  frames through the §4.5.1.1 implicit 17-bit remaining-budget
  gate + §4.5.1.2 Table 65 `{1,1}/2` position symbol + §4.5.1.3
  "remaining whole bytes" size, and Hybrid Opus frames through
  the §4.5.1.1 explicit 37-bit gate + Table 64 `{4095,1}/4096`
  flag + Table 65 position symbol + §4.5.1.3 `2 + dec_uint(256)`
  size with the "claimed > whole bytes remaining" overflow
  routed to `RedundancyDecision::Invalid`. Also lands
  `RedundancyPosition::{End, Beginning}` for the §4.5.1.2 placement
  decision, `RedundancyDecision::{NotPresent, Present {position,
  size_bytes}, Invalid}` for the three legal outcomes, named
  constants `SILK_ONLY_REDUNDANCY_MIN_REMAINING_BITS = 17` /
  `HYBRID_REDUNDANCY_MIN_REMAINING_BITS = 37` /
  `REDUNDANCY_FLAG_ICDF = [1, 0]` / `REDUNDANCY_FLAG_ICDF_FTB = 12`
  / `REDUNDANCY_POSITION_ICDF = [1, 0]` /
  `REDUNDANCY_POSITION_ICDF_FTB = 1` /
  `HYBRID_REDUNDANCY_SIZE_BASELINE_BYTES = 2` /
  `HYBRID_REDUNDANCY_SIZE_DEC_UINT_FT = 256` /
  `REDUNDANCY_MIN_SIZE_BYTES = 2`, and helper accounting
  functions `remaining_bits(rd, opus_frame_bytes)` /
  `whole_bytes_remaining(rd, opus_frame_bytes)` per §4.1.6 +
  §4.5.1.3.

  Round 26 stops at the §4.5.1 boundary metadata (where the
  redundant CELT bytes start, how many of them there are); the
  decode of the redundant CELT frame itself needs §4.3.2.1
  coarse energy (#936, Laplace decoder + `e_prob_model`) and
  §4.3.3 bit allocation (#943, `cache_caps50` + `LOG2_FRAC_TABLE`).

  Twelve new unit tests cover the SILK-only implicit-flag boundary
  (below 17 bits → not present; full buffer → present with
  remaining-whole-bytes size; size-equals-whole-bytes-remaining
  invariant), the Hybrid 37-bit gate (below → not present; full
  buffer → Table 64 read advances `tell()`), the CELT-only bypass
  invariant, the Table 64 / Table 65 ICDF derivations from the
  RFC PDFs, the `RedundancyPosition::from_symbol` Table 65 mapping
  (including the defensive non-binary fall-through), the
  `RedundancyDecision` accessor helpers, the §4.1.6 + §4.5.1.3
  `remaining_bits` / `whole_bytes_remaining` helper accounting,
  and a sanity check on every named constant against the RFC text.
  Decoder runs to 440 unit tests + 20 integration tests, all passing.

* **Clean-room round 25 (2026-05-30):** §4.3.4.5 CELT TF-resolution
  adjustment lookup — a new `celt_tf_adjust` module owning RFC 6716
  Tables 60–63 (the four `(frame_size, choice) -> i8` adjustment
  tables keyed by `(transient, tf_select)`). Exposes
  `celt_tf_adjustment(frame_size, transient, tf_select, tf_change)
  -> i8` (the routed lookup, return value `∈ [-3, 3]`),
  `celt_tf_select_can_affect(frame_size, transient, &[bool]) -> bool`
  (the §4.3.1 "tf_select is only decoded if it can have an impact on
  the result knowing the value of all per-band tf_change flags"
  redundancy gate that the §4.3.4.5 band loop calls AFTER decoding
  every per-band `tf_change[b]`), `TfDirection::{Unchanged,
  IncreaseTime(N), IncreaseFrequency(N)}` carrying the §4.3.4.5
  Hadamard-transform branch and level count, `TfAdjustment` (= `i8`)
  storage type, and named constants `TF_ADJUSTMENT_MAX = 3`,
  `TF_ADJUSTMENT_ABS_MAX = 3`, plus the four tables themselves
  (`TF_ADJ_NONTRANSIENT_SELECT0`, `TF_ADJ_NONTRANSIENT_SELECT1`,
  `TF_ADJ_TRANSIENT_SELECT0`, `TF_ADJ_TRANSIENT_SELECT1`) for direct
  inspection.

  This is the lookup the §4.3.4.5 band loop downstream consumes once
  it's wired up (gated on §4.3.2.1 coarse energy and §4.3.3 bit
  allocation, both still deferred); it sits between Table 55's
  per-band MDCT-bin count and the §4.3.4.2 PVQ shape decoder.

  Twenty-seven new module tests (428 lib tests total, up from 401 at
  round-24 close; 20 integration tests unchanged, grand total 448).
  Coverage: every table's `4 × 2` shape; every cell `∈ [-3, 3]`; all
  32 cells across Tables 60 / 61 / 62 / 63 hand-pinned to RFC 6716
  §4.3.4.5; the "non-transient `choice = 0` is always 0" invariant;
  the "non-transient `choice = 1` is always ≤ 0" invariant; the
  "positive adjustments only on transient frames" §4.3.4.5
  asymmetry pinned at both the table and `TfDirection` layer; the
  Table 62 `choice = 0` monotone `0, 1, 2, 3` scale across frame
  sizes; the universal 2.5 ms `[0, -1]` row across all four tables;
  `TF_ADJUSTMENT_MAX` and `TF_ADJUSTMENT_ABS_MAX` matching the
  observed max over every cell; `celt_tf_adjustment` entry-point
  routing each `(transient, tf_select)` corner; `celt_tf_select_can_affect`
  returning `false` on empty band sets and on the redundant 2.5 ms
  rows; returning `false` on 10 ms non-transient with all-`choice 0`
  bands and `true` as soon as any band picks `choice = 1`;
  returning `true` on 20 ms transient for any non-empty band set;
  `TfDirection::from_adjustment` classification + `levels()` value
  matching `adj.unsigned_abs()` over `[-3, 3]`;
  `IncreaseFrequency` never reachable on non-transient frames;
  `IncreaseTime` always reached for non-transient `choice = 1`.

## [0.0.10](https://github.com/OxideAV/oxideav-opus/compare/v0.0.9...v0.0.10) - 2026-05-29

### Other

- §4.3 Table 55 CELT MDCT-band layout (round 24)
- §4.2.7.4 SILK gain dequant tail (silk_log2lin) — round 23
- §3.4 R1..R7 malformed-input rejection audit (round 22)
- §3.1 / §4.2 framing dispatch (round 21)
- land §4.3 Table 56 CELT pre-band header symbols
- Round 19: §4.2.9 SILK resampler delay budget + sample-rate accounting
- round 18: §4.2.3 SILK header bits + §4.2.4 per-frame LBRR flags
- §4.2.8 stereo unmixing — silk_stereo_MS_to_LR (round 17)
- fix the round-16 test count phrasing
- §4.2.7.9.1 LTP synthesis filter (round 16)
- round 15 — RFC 6716 §4.2.7.9.2 LPC synthesis filter
- §4.2.7.7 LCG seed + §4.2.7.8 excitation reconstruction
- §4.2.7.6 LTP parameters (pitch lags + LTP filter + LTP scaling)
- §4.2.7.5.8 LPC prediction-gain stability limiting
- §4.2.7.5.7 LPC range-limiting bandwidth expansion
- round 10: RFC 6716 §4.2.7.5.6 SILK NLSF→LPC core conversion
- clean-room round 9 — SILK §4.2.7.5.5 NLSF interpolation
- round 8: RFC 6716 §4.2.7.5.4 SILK NLSF stabilization
- round 7: RFC 6716 §4.2.7.5.3 SILK NLSF reconstruction
- round 6: RFC 6716 §4.2.7.5.2 SILK LSF Stage-2
- round 5 fix: compare iCDF slices by value, not pointer identity
- round 5: RFC 6716 §4.2.7.4 SILK subframe gains
- round 4: RFC 6716 §4.2.7.1–§4.2.7.5.1 SILK frame header
- round 3: RFC 6716 §4.1 range decoder
- round 2: RFC 6716 §3.2 frame-packing parser
- round 1: RFC 6716 §3.1 packet TOC byte parser
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Clean-room round 24 (2026-05-29):** §4.3 CELT MDCT-band layout —
  a new `celt_band_layout` module owning RFC 6716 Table 55 (the
  21-band partition with per-band MDCT bin counts at 2.5 / 5 / 10 /
  20 ms and the `0..=20000 Hz` band-edge frequencies) and the §4.3
  "first 17 bands (up to 8 kHz) are not coded in Hybrid mode" rule.
  Exposes `CeltFrameSize` (the four CELT frame-size variants whose
  `repr(u8)` discriminants double as Table 55 column index `0..=3`,
  plus `from_frame_tenths_ms` / `to_frame_tenths_ms`),
  `celt_band_bins_per_channel(band, fs) -> Option<u16>`,
  `celt_band_start_hz(b)` / `celt_band_stop_hz(b)` band edges,
  `celt_band_at_hz(hz) -> Option<usize>` reverse lookup,
  `celt_first_coded_band(is_hybrid)` / `celt_end_coded_band()`
  iterator bounds, `celt_total_bins_per_channel(fs, is_hybrid)`
  column-sum helper, and named constants `CELT_NUM_BANDS = 21`,
  `HYBRID_FIRST_CODED_BAND = 17`, `CELT_MAX_BINS_PER_BAND = 176`.
  The "Custom" mode of §6.2 is explicitly out of scope.

  This is the layout the deferred §4.3.2 coarse-energy decoder,
  §4.3.3 bit allocator, §4.3.4 PVQ shape decoder, §4.3.6
  denormalisation, and §4.3.7 inverse MDCT all need before any band
  loop can iterate.

  Twenty new module tests (401 lib tests total, up from 381 at
  round-23 close; 20 integration tests unchanged) cover: full-band
  start / stop pin (band 0 starts at 0 Hz, band 20 stops at 20 kHz);
  adjacent bands tile without gaps (`stop(b) == start(b + 1)` for
  every `b ∈ 0..=19`); positive band widths everywhere; the
  power-of-two column-scaling invariant (`column(c) == 1 << c *
  column(0)` per band); every cell `∈ [1, 176]` per the §4.3 prose;
  hand-pinned Table 55 cells at bands 0 / 8 / 12 / 15 / 17 / 20;
  band-edge spot pins at start, the §4.3 Hybrid boundary
  (`stop(16) = 8000` = `start(17)`), and tail (`stop(20) = 20000`);
  out-of-range band index returns `None`;
  `CeltFrameSize::from_frame_tenths_ms` round-trip with explicit
  SILK-only rejection (400 / 600 ms); discriminant-vs-column-index
  agreement; the Hybrid-vs-CELT-only first-coded-band split with the
  8 kHz boundary pin; `celt_total_bins_per_channel` column-sum
  agreement against an independent iterator sum for each mode; strict
  `hybrid_total < celt_only_total` invariant; pinned CELT-only column
  sums (100 / 200 / 400 / 800) and Hybrid column sums (60 / 120 /
  240 / 480); `celt_band_at_hz` round-trip against the band-edge
  triple (start, midpoint, `stop - 1` all land on the same band);
  `>= 20 kHz` rejection of `celt_band_at_hz`;
  `celt_band_at_hz(8000) == 17` lining up with
  `HYBRID_FIRST_CODED_BAND`; multiple-of-200-Hz band-width
  invariant with three pinned widths (200 Hz for band 0, 400 Hz for
  band 8, 4400 Hz for band 20); `CELT_MAX_BINS_PER_BAND == max(every
  cell)`.

* **Clean-room round 23 (2026-05-29):** §4.2.7.4 SILK gain
  dequantization tail — a new `silk_log2lin` module exposing
  `silk_log2lin` (the spec's piecewise-linear approximation of
  `2^(inLog_Q7/128)`) and `silk_gains_dequant` (the composed
  `log_gain ∈ 0..=63 → gain_Q16` pipeline
  `silk_log2lin((0x1D1C71*log_gain >> 16) + 2090)`), plus the named
  constants `SILK_LOG_GAIN_MULTIPLIER`, `SILK_LOG_GAIN_BIAS`,
  `SILK_GAIN_Q16_MIN = 81_920`, `SILK_GAIN_Q16_MAX = 1_686_110_208`.
  Also adds a `SubframeGains::dequant_q16` convenience that maps an
  entire decoded frame's `log_gain[]` into the fixed-size `[u32;
  SILK_MAX_SUBFRAMES]` array consumed by the §4.2.7.9.1 LTP and
  §4.2.7.9.2 LPC synthesis filters (with trailing unused slots left
  at zero for two-subframe frames).

  This closes the §4.2.7.4 tail-end conversion that was previously
  noted as deferred since round 5; the §4.2.7.9 synthesis filters
  already accept a `gain_q16` input but were missing the official
  RFC-spec mapping from the decoded `log_gain` integer to the linear
  Q16 gain.

  Nineteen new module tests (381 lib tests total, up from 362 at
  round-22 close, plus the 20 integration tests unchanged) cover:

  * The §4.2.7.4 documented endpoints — `log_gain = 0` returns
    `81920` (= 1.25 in linear scale) and `log_gain = 63` returns
    `1_686_110_208` (≈ 25 728 in linear).
  * Strict monotonicity: `silk_gains_dequant(g+1) > silk_gains_dequant(g)`
    for every `g ∈ 0..=62`.
  * Spec-range invariant across the full `0..=63` sweep.
  * Pure-power-of-two collapse: `silk_log2lin(128*i) == 1<<i` for
    `i ∈ 0..=30`.
  * `silk_log2lin(0) == 1` and `silk_log2lin(1) == 1` (the
    approximation can't resolve sub-128 Q7 below `i = 7`).
  * Pinned `silk_log2lin(7*128 | 64) = 181` — exact match of the
    §4.2.7.4 approximation against the true `2^7.5 ≈ 181.019…`
    halfway between `2^7` and `2^8`, exercising both the `bowed`
    correction and the linear term.
  * Independent i64 oracle of the §4.2.7.4 formula matched bit-for-bit
    by the production i32 implementation across (a) every reachable
    `inLog_Q7` from the `log_gain` dequant pipeline and (b) the full
    `i ∈ 0..=30 × f ∈ 0..=127` Q7 domain.
  * Endpoint algebra pinned independently of `silk_gains_dequant`:
    `log_gain = 0 → in_log_q7 = 2090`; `log_gain = 63 → in_log_q7 =
    3923` (= `30*128 + 83`), `silk_log2lin(2090) = 81_920`,
    `silk_log2lin(3923) = 1_686_110_208`.
  * `SubframeGains::dequant_q16` leaves trailing slots at zero for a
    two-subframe frame and matches per-subframe `silk_gains_dequant`
    calls across the four-subframe frame.

* **Clean-room round 22 (2026-05-27):** §3.4 R1..R7 malformed-input
  rejection audit — a dedicated integration-level test file
  (`tests/malformed_input.rs`, 20 tests) that pins every concrete
  failure mode RFC 6716 §3.4 enumerates for a malformed packet, plus
  property-style sweeps proving the §4.2.3 / §4.2.4 SILK header
  decoder is panic-free on any truncation of a previously-valid
  bitstream (§4.1.4 zero-extension contract). Covers R1 (empty
  packet), R2 (frame > 1275 B for codes 0 and 1, plus the §3.2.1
  boundary at 1275 B), R3 (code-1 odd body length), R4 (code-2
  length-byte truncations + length > remaining + DTX boundary), R5
  (`M = 0` + `M > 48` rejection), R6 (CBR `R % M != 0`), R7 (VBR
  declared length overrun), §3.2.5 padding-chain pathologies, TOC
  total-function self-consistency, §4.2.3 / §4.2.4 truncation
  panic-freeness across `(num_silk_frames, stereo)` × prefix-length
  1..=32, §4.2.4 LBRR-bitmap-never-zero invariant for 40 / 60 ms,
  mono channel never emits side state, parsed-frame slice
  lifetimes, and a code × body-len short-packet panic sweep. Test
  totals: 362 lib + 20 integration = 382 (was 362 lib + 0
  integration after round 21).

* **Clean-room round 21 (2026-05-27):** §3.1 / §4.2 framing dispatch —
  a new `framing` module exposing `OpusFrameRouting`, `OperatingMode`,
  and `SilkBandwidth`. `OpusFrameRouting::from_toc` turns the parsed
  `OpusTocByte` into the per-Opus-frame dispatch decision a §4 decoder
  needs *before* it touches the range coder:

  * `operating_mode` — SilkOnly / Hybrid / CeltOnly from §3.1 Table 2.
  * `silk_layer` / `celt_layer` — which layers are present.
  * `silk_bandwidth` — internal SILK bandwidth (NB / MB / WB), pinned
    to WB for every Hybrid frame regardless of the TOC's SWB / FB per
    RFC 6716 §4.2 first paragraph ("the LP layer itself still only runs
    in WB").
  * `silk_frames_per_channel` — §4.2.2 LP-layer organisation (1 for
    10 / 20 ms Opus frames, 2 for 40 ms, 3 for 60 ms; `None` for
    CELT-only).
  * `total_silk_frames` — channel-count × per-channel SILK frames.
  * `has_per_frame_lbrr_bits` — §4.2.4 duration gate (true for SILK-
    bearing frames longer than 20 ms).

  Thirteen new unit tests cover the SILK-only Table 2 row-by-row
  expectations (12 cells), the Hybrid WB-pin, the CELT-only frames
  sweep across mono/stereo, the §4.2.4 per-frame LBRR gate against
  every Table 2 cell, the `total_silk_frames` formula across all 32
  configs × {mono, stereo}, a 60 ms stereo SILK-only worked example,
  the `c`-bit independence of the routing, the channel-mapping
  pass-through, and the `silk_layer ⇔ silk_bandwidth.is_some() ⇔
  silk_frames_per_channel.is_some()` invariants across the entire
  Table 2 grid.

* **Clean-room round 20 (2026-05-26):** first CELT-layer fragment —
  the RFC 6716 §4.3 / Table 56 pre-band header symbols behind a new
  `celt_header` module (`CeltHeaderPrefix` / `CeltPostFilter`).

  * `silence` — `dec_icdf` against the 2-entry `{32767, 1}/32768`
    iCDF `[1, 0]` at ftb=15. When the flag fires the rest of the
    CELT prefix is force-defaulted per the §4.3 shortcut (no
    post-filter, no transient, no intra).
  * §4.3.7.1 pitch post-filter parameter group: one `dec_bit_logp(1)`
    enable bit, then on the enabled branch `octave` via
    `dec_uint(6)` (uniform on `0..=5`), the `4 + octave` raw-bit
    `fine_pitch` field through `dec_bits`, the §4.3.7.1 pitch-period
    reconstruction `T = (16 << octave) + fine_pitch - 1` (bounded
    `15..=1022`), the 3-bit `gain_index` raw field whose downstream
    gain is `G = 3 * (gain_index + 1) / 32`, and the §4.3.7.1
    `tapset` `{2, 1, 1}/4` iCDF `[2, 1, 0]` at ftb=2.
  * §4.3.1 `transient` and §4.3.2.1 `intra` flags via
    `dec_bit_logp(3)` (PDF `{7, 1}/8`).
  * This is the only Table-56 segment that fits between the SILK
    pipeline already wired up and the §4.3.2.1 coarse-energy
    (Laplace decoder + `e_prob_model` table, gated on a docs gap)
    / §4.3.3 bit allocation (`cache_caps50` + `LOG2_FRAC_TABLE`,
    also gated on a docs gap) sub-pieces; the per-band `tf_change`
    symbols (§4.3.1) live in the band loop and are decoded after
    `coarse energy` per Table 56, so they're deferred as well.
  * Ten new unit tests cover the iCDF transcription self-checks
    (silence PDF sums to 32768, tapset PDF sums to 4), the pitch
    period formula at the global minimum (15), maximum (1022), and
    per-octave boundaries, an all-zero buffer (most-likely symbol
    on every branch ⇒ no silence / no post-filter / no transient /
    no intra), an all-ones buffer (every produced field stays in
    its declared range), a `tell()`-advance proof, a 256-buffer
    fuzz-style range sweep over the post-filter fields, and the
    silence-shortcut post-condition.

* **Clean-room round 19 (2026-05-26):** RFC 6716 §4.2.9 SILK resampler
  delay budget and the internal-vs-output sample-rate accounting
  behind a new `silk_resampler` module
  (`silk_resampler_delay_ms` / `silk_resampler_delay_samples_at` /
  `silk_internal_rate_hz` / `silk_frame_samples_internal` /
  `silk_frame_samples_at_output` / `is_supported_output_rate` /
  `SUPPORTED_OUTPUT_RATES_HZ` / `REFERENCE_RATE_HZ` plus the
  `SILK_RESAMPLER_DELAY_MS_{NB,MB,WB}` constants).

  * Table 54 normative delay allocation: NB = 0.538 ms, MB = 0.692 ms,
    WB = 0.706 ms. The §4.2.9 resampler itself is non-normative ("a
    decoder can use any method it wants to perform the resampling"),
    but the delay budget is normative so the encoder can apply a
    matching pre-delay and keep SILK/CELT aligned across a §4.5 mode
    switch. SWB and FB never reach the §4.2.9 SILK stage and return
    `None`.
  * Internal SILK sample rates per bandwidth (NB = 8 kHz, MB = 12 kHz,
    WB = 16 kHz) and per-frame sample-count accounting at both the
    internal rate and any output rate. NB 20 ms = 160 internal samples
    or 960 output samples at 48 kHz; MB 20 ms = 240 / 960; WB 20 ms =
    320 / 960.
  * The five §4.2.9 supported output rates (8 / 12 / 16 / 24 / 48 kHz),
    the rates "the reference implementation is able to resample to …
    within or near this delay constraint".
  * Delay-samples helper rounds half away from zero per the §4.2.9
    caveat that exact whole-sample delays may be unachievable at
    arbitrary output rates.

  18 new module tests (339 lib tests total, up from 321): Table 54
  transcription + SWB/FB exclusion + strict NB < MB < WB monotonicity;
  Table 54 expansion to 48 kHz samples (26 / 33 / 34) and the
  internal-rate / 24 kHz intermediate-rate expansions; SWB / FB /
  zero-rate rejections; the §4.2.9 supported-output-rate set plus a
  sweep of unsupported rates; the SILK internal rate per bandwidth
  and its membership in the supported-output set; canonical per-frame
  sample counts at internal + output rates plus rejection of
  non-SILK durations; and a cross-check that the Table 54 delay is
  strictly less than one 10 ms SILK frame at every supported output
  rate × every SILK bandwidth.

* **Clean-room round 18 (2026-05-26):** RFC 6716 §4.2.3 SILK
  packet-level header bits and §4.2.4 per-frame LBRR flags behind a
  new `SilkHeaderBits` / `SilkChannelHeader` / `PerFrameLbrr` /
  `silk_frame_count` API (`silk_header` module). The decoder reads
  the §4.2.2 Figures 15/16 prefix:

  * Per channel (mono: 1; stereo: 2): `N` uniform-binary VAD bits
    plus one global LBRR flag via `RangeDecoder::dec_bit_logp(1)`,
    where `N` is the SILK-frame count from §4.2.2 (1 for 10/20 ms
    Opus frames, 2 for 40 ms, 3 for 60 ms).
  * For Opus frames > 20 ms with the channel's global LBRR flag set,
    one §4.2.4 per-frame LBRR symbol from Table 4
    (`{0, 53, 53, 150}/256` for 40 ms or
    `{0, 41, 20, 29, 41, 15, 28, 82}/256` for 60 ms). Both PDFs have
    a leading zero entry per §4.1.3.3, so the iCDF tables
    (`PER_FRAME_LBRR_{40MS,60MS}_ICDF`) drop the leading zero and the
    helper adds offset 1, producing a 2- or 3-bit LBRR bitmap packed
    LSB-to-MSB (bit `i` ↔ SILK frame `i`).
  * For Opus frames ≤ 20 ms the per-frame LBRR bitmap mirrors the
    global LBRR flag without consuming any extra bits, per §4.2.4.

  Output is a `SilkHeaderBits` with per-channel VAD bitmaps, global
  LBRR flags, and a fully-expanded `PerFrameLbrr` bitmap for the
  downstream §4.2.5 / §4.2.6 LBRR + regular SILK loop.

  14 new module tests (321 lib tests total, up from 307): Table 4
  PDF/iCDF transcription self-checks (40 ms + 60 ms, with
  strictly-decreasing + terminator-zero invariants); `per_frame_lbrr_pdf`
  dispatch fallback; `silk_frame_count` §4.2.2 dispatch including the
  2.5/5 ms CELT-only `None` arm; mono 10 ms decode consumes exactly
  2 bits; stereo 60 ms decode populates 3-bit bitmaps within range;
  rejection of `num_silk_frames ∉ {1, 2, 3}`; the §4.2.3-implied
  per-frame mirror on 10 ms with the global flag set (verifying no
  extra symbol consumed); the §4.2.4 skip path on 60 ms with both
  global flags unset (verifying exactly 8 bits consumed); VAD / LBRR
  accessors for present-side and missing-side cases; exhaustive 40 ms
  and 60 ms `decode_per_frame_lbrr` symbol-range sweeps plus a 60 ms
  full-coverage sweep over `{1..=7}`.

* **Clean-room round 17 (2026-05-25):** RFC 6716 §4.2.8 SILK stereo
  unmixing (`silk_stereo_MS_to_LR`) behind a new `stereo_ms_to_lr` /
  `StereoUnmixState` / `StereoWeightsQ13` / `StereoFrame` API
  (`silk_stereo` module). Converts the decoded mid/side `out[]` signals
  into left/right:

  * `p0 = (mid[i-2] + 2*mid[i-1] + mid[i]) / 4.0` is the low-passed,
    one-sample-delayed mid term; the unfiltered mid is also delayed one
    sample (`mid[i-1]`).
  * `left[i] = clamp(-1.0, (1+w1)*mid[i-1] + side[i-1] + w0*p0, 1.0)`,
    `right[i] = clamp(-1.0, (1-w1)*mid[i-1] - side[i-1] - w0*p0, 1.0)`.
  * Phase 1 (first `n1` = 64 NB / 96 MB / 128 WB samples) interpolates
    the §4.2.7.1 Q13 weights from the previous frame's
    `(prev_w0_Q13, prev_w1_Q13)` to the current frame's; phase 2 uses
    the current weights.
  * An uncoded side channel (§4.2.7.2 mid-only flag) is treated as
    all-zero.
  * `StereoUnmixState` carries the two trailing mid samples, one
    trailing side sample, and the previous-frame weights across the
    frame boundary, cleared to zero on a decoder reset per §4.2.8.

  9 module tests: the `interp_phase_samples` table, fresh/reset state,
  empty/length validation, zero-weight delayed-mono collapse, a
  hand-computed constant-weight mid/side reconstruction, phase-1 ramp
  endpoints, mid- and side-history carry across frame boundaries, and
  output clamping.

* **Clean-room round 16 (2026-05-25):** RFC 6716 §4.2.7.9.1 SILK LTP
  synthesis filter behind a new `ltp_synthesis_subframe` /
  `ltp_synth_commit_subframe` / `LtpSynthState` / `LtpSynthSubframe`
  API (`silk_ltp_synth` module). Two regimes:

  * **Unvoiced** (`signal_type != Voiced`): `res[i] = e_Q23[i] / 2^23`
    (a normalised copy of the §4.2.7.8 excitation).
  * **Voiced**: 5-tap Q7 LTP convolution
    `res[i] = e_Q23[i]/2^23 + Σ_{k=0..4} res[i - pitch_lag + 2 - k] *
    b_Q7[k]/128`, where the "prior res[]" values come from rewhitening
    the previous subframes' outputs through the current subframe's LPC
    coefficients. Two rewhitening regions:
    * Region A (out[] rewhiten, `(j - pitch_lag - 2) <= i < out_end`):
      `res[i] = 4*LTP_scale_Q14/gain_Q16 * clamp(out[i] - Σ
      out[i-k-1] * a_Q12[k]/4096, -1, 1)`.
    * Region B (lpc[] rewhiten, `out_end <= i < j`):
      `res[i] = 65536/gain_Q16 * (lpc[i] - Σ lpc[i-k-1] *
      a_Q12[k]/4096)`.

  `out_end` and the effective `LTP_scale_Q14` follow the §4.2.7.9.1
  LSF-interpolation-split branch: third/fourth subframe of a 20 ms
  SILK frame with `w_Q2 < 4` ⇒ `out_end = j - (s-2)*n` and
  `LTP_scale_Q14 = 16384`; otherwise `out_end = j - s*n` and the
  §4.2.7.6.3 decoded scaling factor is used.

  `LtpSynthState` carries 306 samples of `out[]` history (`lag_max
  288 + d_LPC 16 + 2`) and 256 samples of `lpc[]` history (`3 prior
  WB subframes 240 + d_LPC 16`) — the buffer sizes called out in the
  §4.2.7.9.1 paragraphs. `reset()` clears both for the §4.5.2
  decoder-reset / uncoded-side-channel-frame paths;
  `ltp_synth_commit_subframe` pushes the §4.2.7.9.2 outputs back into
  the state for the next subframe's rewhitening.

  Twenty-one new unit tests (298 lib tests total) cover the
  spec-stated buffer-size constants, `LtpSynthState` d_LPC routing +
  zero-init + reset + start_frame + push_subframe ordering, the
  unvoiced normalised-excitation identity (NB / Wb sweeps with
  Inactive and Unvoiced both routed to the unvoiced path), four
  input-validation rejections (mismatched lengths, bandwidth, subframe
  index, non-positive pitch lag), the voiced zero-history /
  zero-excitation / zero-b identity, the voiced `b == 0` pass-through
  identity, the voiced `b_Q7[0]` region-A pitch-lookback algebra
  (`0.5 * 4*LTP_scale_Q14/gain_Q16 * out[j-14]`), the voiced `b_Q7[2]`
  region-B (lpc[]) rewhiten algebra, the LSF-interpolation-split
  override (effective scale becomes `4*16384/65536 = 1.0` exactly),
  voiced-decode determinism, and a no-panic finite-output sweep across
  3 buffers × {NB, MB, WB} × {10 ms, 20 ms} × 4 subframes with state
  carried via `ltp_synth_commit_subframe`.

* **Clean-room round 15 (2026-05-25):** RFC 6716 §4.2.7.9.2 SILK LPC
  synthesis filter behind a new `lpc_synthesis_subframe` /
  `lpc_synthesis_frame` / `LpcSynthState` API (`silk_lpc_synth` module).
  Given the §4.2.7.9.1 LPC residual `res[]` for the current subframe, the
  §4.2.7.4 Q16 quantization gain `gain_Q16[s]`, and the §4.2.7.5.8
  stabilised Q12 short-term predictor `a_Q12[k]`, the filter runs:

  ```
                                  d_LPC-1
                 gain_Q16[s]         __              a_Q12[k]
        lpc[i] = ----------- * res[i] + \  lpc[i-k-1] * --------
                   65536.0              /_               4096.0
                                        k=0

        out[i] = clamp(-1.0, lpc[i], 1.0)
  ```

  Each subframe carries d_LPC unclamped `lpc[i]` history samples forward
  into the next subframe through `LpcSynthState`, which is cleared to
  zero on a decoder reset (RFC 6716 §4.5.2) or after an uncoded regular
  SILK frame for the channel. The §4.2.7.9 preamble explicitly licenses
  a floating-point implementation here ("the remainder of the
  reconstruction process for the frame does not need to be bit-exact"),
  so the filter accumulates in `f32` with the spec's left-to-right
  formula. The §4.2.7.9.2 wording that "the decoder saves the unclamped
  values lpc[i] to feed into the LPC filter for the next subframe, but
  saves the clamped values out[i] for rewhitening in voiced frames" is
  implemented exactly: state holds unclamped values; the rendered output
  is the clamped vector. d_LPC routing follows §4.2.7.5: 10 for NB / MB
  and 16 for WB; SWB / FB rejected at the SILK layer.

  Eighteen new unit tests (277 lib tests total, up from 259 at round-14
  close) cover `subframe_samples` (40 / 60 / 80 for NB / MB / WB + SWB /
  FB rejection); `LpcSynthState` d_LPC routing + zero initialisation +
  reset; the three input-validation rejections (mismatched `res` /
  `out_clamped` / `a_q12` lengths); the algebraic identities (a_Q12 = 0
  gives `lpc = gain_Q16/65536 * res`; res = 0 with zero history gives
  identically zero output regardless of a_Q12 / gain); a hand-pinned
  single-tap unity-gain NB filter (impulse response is the constant
  `1.0`); a hand-pinned single-tap half-gain WB filter (impulse response
  is the geometric series `0.5^(i+1)` and the history holds the final 16
  unclamped samples to 1e-9 precision); a hand-traced two-tap NB filter
  with non-trivial `res[]` `[1, 2, 3, 0, ...]` yielding the exact
  sequence `[1.0, 2.5, 4.5, 2.875, 2.5625, ...]` plus the per-sample
  clamp; the cross-subframe history carry-over (an impulse decays into a
  unit-feedback subframe and the next subframe of zero residual still
  emits `1.0` everywhere); the decoder-reset path zeroes history; the
  `out[]` ∈ `[-1.0, 1.0]` clamp post-condition under deliberately
  over-driven input; the spec wording that `state.history` stores the
  unclamped `lpc[i]` and not the saturated `out[i]`; the
  `lpc_synthesis_frame` wrapper matches an explicit per-subframe loop
  bit-for-bit (state included) and rejects bad input lengths; and a
  sweep over {NB, MB, WB} × {10, 20 ms} that asserts no panics, the
  correct output length, the clamp post-condition, and the correct
  history length. The §4.2.7.9.1 LTP synthesis filter that produces the
  voiced-frame `res[]` is deferred to a later round; this stage can
  already be driven directly off `e_Q23[i] / 2^23` for unvoiced
  subframes per the §4.2.7.9.1 wording.

* **Clean-room round 14 (2026-05-25):** RFC 6716 §4.2.7.7 SILK LCG seed
  (`silk_lcg_seed` module) and §4.2.7.8 SILK excitation decoder behind a
  new `Excitation` / `ExcitationConfig` API (`silk_excitation` module).

  The §4.2.7.7 LCG seed is a single uniform 4-entry symbol from Table
  43 (`{64, 64, 64, 64}/256`) producing a value in `0..=3` that
  initialises the LCG used by §4.2.7.8.6 reconstruction.

  The §4.2.7.8 excitation runs in six substeps: §4.2.7.8.1 rate level
  (one symbol per SILK frame from one of two Table 45 PDFs chosen by
  signal type, producing `0..=8`); §4.2.7.8.2 per-shell-block pulse
  count (Table 46 PDFs at the chosen rate level, with the special
  value 17 chaining into rate level 9, then to rate level 10 on the
  10th occurrence — capping extra LSBs at 10 per block per the
  §4.2.7.8.2 spec note); §4.2.7.8.3 recursive pulse-location decoding
  (partition halves 16 → 8 → 4 → 2 → 1 with Tables 47/48/49/50 split
  PDFs selected by partition size + remaining pulse count); §4.2.7.8.4
  per-coefficient LSB decoding (Table 51 `{136, 120}/256`, doubling
  the magnitude and adding each bit MSB-first); §4.2.7.8.5 sign
  decoding (Table 52, picked by `(signal_type, qoff_type,
  min(pulses_in_block, 6))` — 42 PDFs in all); and §4.2.7.8.6
  reconstruction with `e_Q23[i] = (e_raw << 8) - sign(e_raw)*20 +
  offset_Q23` (Table 53 offsets `{25, 60, 25, 60, 8, 25}`) plus the
  32-bit LCG step `seed = (196314165*seed + 907633515) & 0xFFFFFFFF`
  whose MSB drives a per-sample sign flip, followed by
  `seed = (seed + e_raw[i]) & 0xFFFFFFFF` for the next iteration.

  Table 44 routes `(bandwidth, frame_size)` to the shell-block count
  (5/8/10/10/15/20 for the six NB/MB/WB × 10ms/20ms cells; SWB/FB
  rejected as not paired with the SILK layer). The 10 ms MB special
  case decodes 8 shell blocks (128 samples) of which the trailing 8
  are discarded by the caller per the §4.2.7.8 preamble.

  Thirty new unit tests (259 lib tests total, up from 229 at round-13
  close) cover the Table 43 transcription + the 0..=3 + 2-bits-per-
  symbol invariants; Table 44 (all six cells + SWB/FB rejection); both
  Table 45 PDFs; all eleven Table 46 PDFs including the L10 cell-17 =
  0 boundary; spot-checks on Tables 47/48/49/50 (1- and ≥7-pulse
  cells); Table 51; six Table 52 spot-checks across each
  `(signal_type, qoff_type)` quadrant + the "6 or more" saturation;
  all six Table 53 offsets; the LCG recurrence pinned algebraically
  for the first two steps from seed=0; `Excitation::decode` rejections
  (invalid LCG seed, SWB/FB bandwidth); per-cell sample count; the
  §4.2.7.8 "fits in 24 bits including sign" invariant across three
  buffers × all (NB/MB/WB × 10/20 ms) cells; per-block pulse-count ≤
  16 and LSB-count ≤ 10 invariants; a hand-pinned reconstruction of
  an isolated mag=5 sign=-1 sample producing ±1235; the
  zero-magnitude `|e_Q23[i]| == offset_Q23` identity; cross-pass
  reproducibility; LCG-seed divergence; and a no-panic sweep over
  three buffers × {NB, MB, WB} × {10, 20 ms} × 3 signal types × 2
  qoff types × 4 seeds. The §4.2.7.9 LTP / LPC synthesis filters that
  consume `e_Q23[]` are deferred to a later round.

* **Clean-room round 13 (2026-05-24):** RFC 6716 §4.2.7.6 SILK Long-Term
  Prediction parameters behind a new `LtpParameters` / `LtpConfig` API
  (`silk_ltp` module). Decodes the §4.2.7.6.1 primary pitch lag (either
  absolute, via Table 29 high-part + Table 30 bandwidth-conditioned
  low-part / scale / lag-range, or relative, via the Table 31 21-entry
  delta PDF with a zero-delta fallback to absolute coding), the
  pitch-contour VQ index (Table 32 PDF; Tables 33–36 codebooks) that
  refines the primary lag into per-subframe pitch lags clamped to the
  bandwidth's `[lag_min, lag_max]`, the §4.2.7.6.2 periodicity index
  (Table 37) and per-subframe 5-tap Q7 LTP filter taps (Table 38 PDFs;
  Tables 39–41 codebooks of sizes 8 / 16 / 32), and the optional
  §4.2.7.6.3 Q14 LTP scaling factor (Table 42 → `{15565, 12288, 8192}`;
  default `15565` ≈ 0.95 when not coded or for non-voiced frames).
  Non-voiced frames consume no LTP bits. The §4.2.7.9 LTP synthesis
  filter that consumes these parameters is deferred to a later round.

  Nineteen new unit tests (229 lib tests total in the crate, up from
  210 at round-12 close) cover the eleven PDF → iCDF transcriptions
  (Tables 29 / 30 NB-MB-WB / 31 / 32 four PDFs / 37 / 38 three PDFs /
  42), Table 30 scale + lag-range values, contour codebook
  size-matches-PDF self-checks + index-0 all-zero rows + four
  interior-row spot-checks against the spec, LTP filter codebook sizes
  (8 / 16 / 32) + four boundary-row spot-checks against Tables 39–41,
  the non-voiced no-bits-consumed property (both Inactive and Unvoiced),
  rejection of non-2-non-4 `num_subframes` and SWB / FB bandwidths,
  in-range absolute-coding lags + production / independent formula
  agreement across {NB, MB, WB} × {2, 4} subframes, relative-coding
  non-zero-delta + zero-delta-fallback paths, LTP-scaling-present output
  ∈ `{15565, 12288, 8192}` and absent-uses-default-without-reading bit
  positioning, and a sweep over {NB, MB, WB} × {2, 4} × {absent,
  present} × {Absolute, Relative} × three buffers asserting no panics,
  the `[lag_min, lag_max]` post-condition, and periodicity ≤ 2.

* **Clean-room round 12 (2026-05-24):** RFC 6716 §4.2.7.5.8 SILK LPC
  prediction-gain limiting behind a new `LpcQ17::prediction_gain_limited`
  method returning a new `LpcQ12` type (`silk_lsf_to_lpc` module).
  Consumes the (range-limited) §4.2.7.5.7 `a32_Q17[]` and produces the
  final stable Q12 filter `a_Q12[k]` for the §4.2.7.9.2 LPC synthesis.

  - **Up to 16 rounds of stability-driven bandwidth expansion.** Each
    round converts to the real Q12 coefficients
    `a32_Q12[n] = (a32_Q17[n] + 16) >> 5` and runs the
    `silk_LPC_inverse_pred_gain_QA()` stability test. If the filter is
    stable the Q12 coefficients are returned; otherwise a chirp round with
    `sc_Q16[0] = 65536 - (2<<i)` is applied via the same
    `silk_bwexpander_32` as §4.2.7.5.7. On round 15 `sc_Q16[0] = 0`,
    zeroing every coefficient so an all-zero (trivially stable) filter is
    the worst-case outcome.
  - **`silk_LPC_inverse_pred_gain_QA()` stability test (`is_lpc_stable`).**
    The DC-response check (`DC_resp = Σ a32_Q12[n] > 4096` ⇒ unstable)
    followed by the fixed-point Levinson recurrence on the Q24-widened
    coefficients: initialize `inv_gain_Q30[d_LPC] = 1<<30` and
    `a32_Q24[d_LPC-1][n] = a32_Q12[n] << 12`, then for each `k` from
    `d_LPC-1` down to `0` reject on `abs(a32_Q24[k][k]) > 16773022`
    (≈ 0.99975 in Q24) or `inv_gain_Q30[k] < 107374` (≈ 1/10000 in Q30)
    via `rc_Q31 = -a32_Q24[k][k] << 7`,
    `div_Q30 = (1<<30) - (rc_Q31*rc_Q31 >> 32)`,
    `inv_gain_Q30[k] = (inv_gain_Q30[k+1]*div_Q30 >> 32) << 2`. Each
    surviving step (for `k > 0`) computes row `k-1` via the spec's
    `b1 = ilog(div_Q30)`, `inv_Qb2`, `err_Q29`, `gain_Qb1`, `num_Q24[n]`,
    `a32_Q24[k-1][n]` formulas. Every multiply the spec marks as needing
    more than 32 bits is performed in `i64`.

  `LpcQ12` exposes `a_q12()`, `len()`, `is_empty()`, and `rounds()` (the
  number of chirp rounds that ran before the filter was deemed stable).

  9 new unit tests (210 lib tests total in the crate; up from 201 in the
  round-11 close) covering:

  - `is_lpc_stable` agrees with an independent 2D-matrix spec
    transcription oracle on hand-built filters (all-zero, gentle decay,
    near-unit single tap at the DC=4096 boundary, DC over the ceiling,
    mixed-sign moderate).
  - The all-zero filter is stable for both NB/MB and WB widths.
  - DC response `> 4096` is rejected before the Levinson recurrence; the
    DC=4096 boundary is not rejected by the DC check alone.
  - A real §4.2.7.5.7 → §4.2.7.5.8 conversion of a typical decoded NLSF
    vector returns on round 0 with `a_Q12 == (a32_Q17 + 16) >> 5`.
  - Deliberately unstable Q17 inputs (near-unit tap, high-gain resonant
    alternating taps, DC over the ceiling) always converge to a stable
    Q12 filter within ≤ 16 rounds.
  - A persistently-unstable input zeroes every coefficient if it reaches
    the forced round-15 (`sc_Q16[0] = 0`) step.
  - The emitted Q12 filter fits a signed 16-bit value.
  - A real §4.2.7.5.2 → … → §4.2.7.5.7 → §4.2.7.5.8 pipeline sweep across
    all 32 `I1` values × {NB, MB, WB} on three buffers: the emitted Q12
    filter is always stable (cross-checked vs the oracle) and the round
    count is ≤ 16.
  - `ilog64` (the i64 variant used by §4.2.7.5.8) matches the §1.1.10
    definition for the spec examples plus the `2^30` / `2^30 - 1`
    `div_Q30`-domain boundaries.

* **Clean-room round 11 (2026-05-24):** RFC 6716 §4.2.7.5.7 SILK LPC
  range-limiting bandwidth expansion behind a new
  `LpcQ17::range_limited` method (`silk_lsf_to_lpc` module). Consumes the
  raw §4.2.7.5.6 `a32_Q17[]` and reduces it so it fits a signed 16-bit
  Q12 value:

  - **Up to 10 rounds of `silk_bwexpander_32` chirping.** Each round
    finds the index `k` of the largest `abs(a32_Q17[k])` (ties to the
    lowest `k`), computes `maxabs_Q12 = min((maxabs_Q17 + 16) >> 5,
    163838)`, and stops once `maxabs_Q12 <= 32767`. Otherwise it derives
    the chirp factor `sc_Q16[0] = 65470 - ((maxabs_Q12 - 32767) << 14) /
    ((maxabs_Q12 * (k+1)) >> 2)` (integer division) and runs the
    `silk_bwexpander_32` recurrence `a32_Q17[k] = (a32_Q17[k]*sc_Q16[k])
    >> 16`, `sc_Q16[k+1] = (sc_Q16[0]*sc_Q16[k] + 32768) >> 16`. The
    first multiply runs in i64 ("up to 48 bits of precision"); the second
    is performed unsigned per the spec to avoid 32-bit overflow.
  - **Post-loop Q12 saturation.** If `maxabs_Q12` is still greater than
    32767 after the 10th round, each coefficient is saturated in the Q12
    domain and converted back to Q17:
    `a32_Q17[k] = clamp(-32768, (a32_Q17[k] + 16) >> 5, 32767) << 5`.
    The result is returned in the Q17 domain (the §4.2.7.5.8
    prediction-gain limiting that follows consumes Q17 coefficients), so
    it shares the `LpcQ17` representation. The §4.2.7.5.8 stability check
    is deferred to a subsequent round.

  `maxabs_Q17` is taken via `i32::unsigned_abs()` so an `i32::MIN`
  coefficient from an adversarial §4.2.7.5.6 output does not panic.

  6 new unit tests (201 lib tests total in the crate; up from 195 in the
  round-10 close) covering:

  - Small coefficients already fitting Q12 pass through unchanged.
  - Production agrees bit-for-bit with an independent i128 transcription
    of the §4.2.7.5.7 loop on synthetic overflow vectors (single peak,
    peak at a non-zero index, mixed-sign large coefficients, a moderate
    overshoot) and on an extreme input pinned to the 163838 cap.
  - Every range-limited output fits a signed 16-bit Q12 value.
  - The `i32::MIN` coefficient no-panic edge.
  - The post-loop Q12 saturation formula pinned in isolation (the
    adaptive chirp converges every realistic input within 10 rounds, so
    the engaged branch is effectively unreachable; the formula is pinned
    directly to catch a transcription typo).
  - A real §4.2.7.5.2 → §4.2.7.5.3 → §4.2.7.5.4 → §4.2.7.5.6 →
    §4.2.7.5.7 pipeline sweep across all 32 `I1` values × {NB, MB, WB}
    asserting the Q12 fit and production/oracle agreement.

* **Clean-room round 10 (2026-05-24):** RFC 6716 §4.2.7.5.6 SILK
  Normalized LSF → LPC core conversion behind a new `LpcQ17` API
  (`silk_lsf_to_lpc` module). Consumes a stabilized / interpolated
  `nlsf_q15[]` (the §4.2.7.5.4 / §4.2.7.5.5 output) and runs the
  `silk_NLSF2A` procedure in three steps:

  - **Table 27 ordering + Table 28 cosine table (`silk_NLSF2A_cos`).**
    The 129-entry Q12 cosine table (`cos_Q12[0]=4096`, `cos_Q12[64]=0`,
    `cos_Q12[128]=-4096`, anti-symmetric about i=64) is transcribed
    verbatim. For each coefficient `i = nlsf >> 8`, `f = nlsf & 255`
    and the §4.2.7.5.6 piecewise-linear interpolation
    `c_Q17[ordering[k]] = (cos_Q12[i]*256 + (cos_Q12[i+1]-cos_Q12[i])*f
    + 4) >> 3` lands the re-ordered Q17 cosine vector. The Table 27
    `ordering[]` vectors are NB/MB `[0,9,6,3,4,5,8,1,2,7]` and WB
    `[0,15,8,7,4,11,12,3,2,13,10,5,6,9,14,1]`.
  - **`silk_NLSF2A_find_poly` P/Q recurrence.** Two rolling-row passes
    on the even-indexed (P) and odd-indexed (Q) `c_Q17[]` cells run
    `p[k][j] = p[k-1][j] + p[k-1][j-2] - ((c*p[k-1][j-1] + 32768)>>16)`
    in i64 to absorb the spec's noted "up to 48 bits of intermediate
    precision" requirement, with the §4.2.7.5.6 boundary conditions
    `p[k][j<0] = 0` and `p[k][k+2] = p[k][k]`.
  - **`silk_NLSF2A` last-row assembly.** The final i64 rows are folded
    into the 32-bit Q17 LPC coefficients via the §4.2.7.5.6 sum/diff
    pair: `a32_Q17[k] = -((q_diff) + (p_sum))` and
    `a32_Q17[d_LPC-k-1] = (q_diff) - (p_sum)`, where
    `q_diff = q[d2-1][k+1] - q[d2-1][k]` and
    `p_sum  = p[d2-1][k+1] + p[d2-1][k]`.

  The §4.2.7.5.7 range-limiting bandwidth-expansion loop (up to 10
  rounds shrinking `a32_Q17[]` so it fits Q12) and the §4.2.7.5.8
  prediction-gain stability check (up to 16 chirp rounds + the
  `silk_LPC_inverse_pred_gain_QA` test) are deferred to subsequent
  rounds.

  22 new unit tests (195 lib tests total in the crate; up from 173 in
  the round-9 close) covering:

  - Table 27 row-widths, permutation-of-0..d_LPC self-checks, and
    bandwidth routing (`ordering()` rejects SWB / FB).
  - Table 28 length (129), the three anchors (0 → 4096; 64 → 0;
    128 → -4096), the strict-monotone-decreasing pairwise check, the
    anti-symmetric-about-64 invariant, the Q12-range bound, and four
    row spot-checks (rows 0, 16, 60, 64, 124).
  - `nlsf_to_c_q17` at the table anchor points (`f == 0` round-trip
    against `cos_Q12[8*k]`) and at the linear-interpolation midpoint
    (`f == 128` matching the `16*(a+b)` algebraic identity).
  - `nlsf_to_c_q17` rejects SWB / FB and `nlsf_q15.len() != d_LPC`.
  - `LpcQ17` length, SWB / FB and length-mismatch rejection.
  - Production `LpcQ17::from_nlsf` agrees bit-for-bit with an
    independent 2D-matrix spec-transcription oracle of the §4.2.7.5.6
    recurrence on synthetic ascending NLSF vectors for both NB and WB.
  - Production `LpcQ17::from_nlsf` agrees with the same oracle when
    driven by the full §4.2.7.5.2 → §4.2.7.5.3 → §4.2.7.5.4 decoder
    pipeline across all 32 `I1` values × {NB, MB, WB}.
  - A no-panic sweep over three buffers × all 32 `I1` × {NB, MB, WB}
    confirming the full §4.2.7.5.2..§4.2.7.5.6 path is panic-free.

* **Clean-room round 9 (2026-05-24):** RFC 6716 §4.2.7.5.5 SILK
  Normalized LSF interpolation behind a new `LsfInterpolated` /
  `LsfInterpContext` API (`silk_lsf_interp` module). For a 20 ms SILK
  frame the first half (the first two subframes) may use NLSF
  coefficients interpolated between the most recent coded frame's
  vector `n0_Q15[]` and the current §4.2.7.5.4-stabilized vector
  `n2_Q15[]`:

  - **`TwentyMs`** decodes the Q2 factor `w_Q2 ∈ 0..=4` from the
    Table 26 PDF (`{13, 22, 29, 11, 181}/256`; iCDF
    `[243, 221, 192, 181, 0]`) and computes
    `n1_Q15[k] = n0_Q15[k] + (w_Q2*(n2_Q15[k] - n0_Q15[k]) >> 2)`.
  - **`TwentyMsAfterResetOrUncoded`** still decodes the factor (to keep
    the range coder in sync) but discards it and uses `4` instead, so
    `n1_Q15[] == n2_Q15[]`. The same forced-4 behaviour applies when
    `n0_Q15[]` is `None` (no prior-frame history).
  - **`TenMs`** reads no factor (it is not present in the bitstream)
    and produces no first-half vector.

  The result exposes the decoded `w_q2()` (`None` for 10 ms) and the
  first-half `n1_q15()` (`None` for 10 ms). The second half of a 20 ms
  frame and the whole of a 10 ms frame always use `n2_Q15[]` directly.

  10 new unit tests (173 lib tests total in the crate; up from 163 in
  the round-8 close) covering:

  - Table 26 PDF→iCDF transcription (sum-to-256 and strict
    monotone-decreasing iCDF self-checks; exactly five factors).
  - 10 ms path reads nothing (`tell()` unchanged) and has no
    first-half vector.
  - End-to-end 20 ms interpolation matching an independent
    re-derivation of the §4.2.7.5.5 formula.
  - The `w_Q2 == 0 → n1 == n0` and `w_Q2 == 4 → n1 == n2` algebraic
    identities.
  - The reset/uncoded context decodes the factor then forces `4`
    (asserting `tell()` parity with the normal context).
  - The no-history `n0 = None` path forces `n1 == n2` even in the
    normal context while still decoding the factor.
  - `n0`-length-mismatch rejection (`Error::MalformedPacket`).
  - A sweep asserting every interpolated value stays in `[0, 32767]`
    across {NB, MB, WB} × all 32 `I1` × `w_Q2 ∈ 0..=4`.

* **Clean-room round 8 (2026-05-23):** RFC 6716 §4.2.7.5.4 SILK
  Normalized LSF stabilization behind a new `NlsfStabilized` API
  (`silk_lsf_stabilize` module). Consumes the §4.2.7.5.3
  `NlsfReconstructed` output and enforces the Table 25 minimum spacing
  between consecutive `NLSF_Q15[]` coefficients, with the boundary
  conventions `NLSF_Q15[-1] = 0` / `NLSF_Q15[d_LPC] = 32768` and a
  Table 25 column of `d_LPC + 1` entries.

  - **Up to 20 distortion-minimizing passes.** Each pass finds the
    smallest `NLSF_Q15[i] - NLSF_Q15[i-1] - NDeltaMin_Q15[i]` over
    `i ∈ 0..=d_LPC` (ties to lower `i`); stops if non-negative.
    Otherwise `i == 0` → `NLSF_Q15[0] = NDeltaMin_Q15[0]`,
    `i == d_LPC` → `NLSF_Q15[d_LPC-1] = 32768 - NDeltaMin_Q15[d_LPC]`,
    and any interior `i` re-centres the pair via the
    `min_center`/`max_center` running sums and
    `center_freq = clamp(min_center, (NLSF[i-1]+NLSF[i]+1)>>1,
    max_center)`, then `NLSF_Q15[i-1] = center_freq -
    (NDeltaMin_Q15[i]>>1)` and `NLSF_Q15[i] = NLSF_Q15[i-1] +
    NDeltaMin_Q15[i]`.
  - **Fallback (once after the 20th pass).** Sort ascending, then a
    forward `max(NLSF[k], NLSF[k-1] + NDeltaMin[k])` sweep and a
    backward `min(NLSF[k], NLSF[k+1] - NDeltaMin[k+1])` sweep.
  - **RFC 8251 §7 erratum.** The fallback forward sweep's addition
    uses 16-bit saturating addition (`silk_ADD_SAT16`) to avoid the
    integer wrap-around the erratum documents on adversarial inputs
    with extremely large high-LSF parameters.

  Table 25 is transcribed verbatim: NB/MB column
  `{250, 3, 6, 3, 3, 3, 4, 3, 3, 3, 461}`, WB column
  `{100, 3, 40, 3, 3, 3, 5, 14, 14, 10, 11, 3, 8, 9, 7, 3, 347}`.

  19 new unit tests (163 lib tests total in the crate; up from 144 in
  the round-7 close) covering:

  - Table 25 lengths (`d_LPC + 1` for NB/MB and WB) and spot-checks.
  - `ndelta_min_q15` rejects SWB / FB.
  - `add_sat16` saturates at both `i16` bounds.
  - An already-stable comfortably-spaced input is left bit-identical
    (NB and WB).
  - First-coefficient-too-low pushed up to `NDeltaMin[0]`;
    last-coefficient-too-high pulled down to `32768 - NDeltaMin[d_LPC]`.
  - Interior re-centring with hand-computed exact `NLSF_Q15[i-1]` /
    `NLSF_Q15[i]` values for an isolated tight pair.
  - The fallback sort + sweeps fix a fully-reversed input; all-zero
    and all-32767 inputs are spread to valid spacing.
  - The RFC 8251 §7 no-wrap guard: an all-`i16::MAX` input stays in
    `[0, 32767]` (a wrap-around would produce a negative value).
  - End-to-end sweep across all 32 `I1` values × {NB, MB, WB} wired
    through the §4.2.7.5.2 / §4.2.7.5.3 decoders, asserting the
    spacing post-condition, the `[0, 32767]` bound, and strict
    monotonicity of every stabilized vector.
  - `from_reconstructed` rejects SWB / FB and a bandwidth ↔ recon
    length mismatch.

* **Clean-room round 7 (2026-05-22):** RFC 6716 §4.2.7.5.3 SILK
  Normalized LSF reconstruction behind a new `NlsfReconstructed` API
  (`silk_lsf_recon` module). Lifts the stage-2 residual `res_Q10[]`
  (returned by round 6's `LsfStage2`) to the final `NLSF_Q15[]`
  coefficient vector in three steps:

  - **Tables 23 / 24 lookup.** The 32 × 10 NB/MB and 32 × 16 WB
    stage-1 codebook vectors `cb1_Q8[]` are transcribed verbatim from
    the RFC text. The `(bandwidth, I1)` lookup yields a slice of
    `d_LPC` Q8 cells.
  - **IHMW weights `w_Q9[k]`.** The low-complexity Inverse Harmonic
    Mean Weighting derivation
    `w2_Q18[k] = (1024/(cb1_Q8[k]-cb1_Q8[k-1])
                + 1024/(cb1_Q8[k+1]-cb1_Q8[k])) << 16`
    (with boundary `cb1_Q8[-1] = 0` and `cb1_Q8[d_LPC] = 256` and
    integer division) is reduced through the spec's square-root
    approximation: `i = ilog(w2_Q18[k])`,
    `f = (w2_Q18[k] >> (i-8)) & 127`,
    `y = ((i & 1) ? 32768 : 46214) >> ((32-i) >> 1)`,
    `w_Q9[k] = y + ((213 * f * y) >> 16)`. Every weight across the
    full 32 × {NB/MB d_LPC=10, WB d_LPC=16} sweep falls inside the
    spec's documented 13-bit `[1819, 5227]` range.
  - **Final NLSF.**
    `NLSF_Q15[k] = clamp(0, (cb1_Q8[k]<<7) + (res_Q10[k]<<14)/w_Q9[k], 32767)`,
    integer division throughout. The §4.2.7.5.4 stabilization and
    §4.2.7.5.5 interpolation passes that consume `NLSF_Q15[]` are
    deferred to a later round.

  26 new unit tests (144 lib tests total in the crate; up from 118 in
  the round-6 close) covering:

  - `ilog(n)` matches the RFC §1.1.10 examples for `n ∈ {-1, 0, 1,
    2, 3, 4, 7}`, plus 8 / 255 / 256 / 2^24.
  - Tables 23 and 24 rows are strictly monotone increasing (a
    pre-condition of the IHMW divisor being positive).
  - Tables 23 / 24 row widths equal `D_LPC_NB_MB` (10) and
    `D_LPC_WB` (16).
  - Table 23 row 0 (`12 35 60 83 108 132 157 180 206 228`),
    Table 23 row 31, Table 24 row 0
    (`7 23 38 54 69 85 100 116 131 147 162 178 193 208 223 239`),
    Table 24 row 31 spot-checks.
  - `cb1_q8()` routes Nb / Mb to Table 23 and Wb to Table 24, and
    rejects `I1 >= 32` and Swb / Fb (SILK never sees the latter
    after the §4.2.2 hybrid split).
  - All 32 × NB IHMW weights and all 32 × WB IHMW weights are in
    `[1819, 5227]` (the spec's own documented range for the 13-bit
    tabulated form).
  - Concrete hand-computed IHMW match: NB I1=0 k=0 → 2897; WB I1=0
    k=0 → 3657 — both derived from `1024/diff` integer arithmetic
    against the transcribed `cb1_Q8` cells.
  - With `res_Q10[k] == 0`, every reconstructed `NLSF_Q15[k]` equals
    `cb1_Q8[k] << 7` (bounded by `242 << 7 = 30976`, within the
    `32767` clamp).
  - Sweep across all 32 `I1` values × {NB, MB, WB} via a synthetic
    range-decoder buffer: every reconstructed `NLSF_Q15[k]` is in
    `[0, 32767]` and exactly reproduces the §4.2.7.5.3 formula
    re-applied to the decoded `res_Q10[k]` and `w_Q9[k]`.
  - `from_stage1_and_stage2` rejects mismatched bandwidth ↔ stage-2
    length (e.g. WB-decoded stage-2 with NB reconstruction),
    out-of-range `I1`, and Swb / Fb bandwidths.

* **Clean-room round 6 (2026-05-22):** RFC 6716 §4.2.7.5.2 Normalized
  LSF Stage-2 decoder behind a new `LsfStage2` API. The caller passes
  the SILK-layer bandwidth (`Nb` / `Mb` / `Wb`) and the stage-1 codebook
  index `I1 ∈ 0..32` (returned by the §4.2.7.5.1 decoder). The decoder:
  - Reads `d_LPC` stage-2 residual indices `I2[k]` (10 cells for
    NB / MB, 16 for WB) using one of 16 signal-shape codebook PDFs
    (Tables 15 a..h for NB/MB, Table 16 i..p for WB). The
    `(bandwidth, I1) → codebook` mapping comes from Table 17 (NB/MB)
    or Table 18 (WB). Each raw symbol is `0..=8`; after the `-4`
    subtraction the index is `[-4, 4]`. If `|idx| == 4`, the Table 19
    extension PDF (`{156, 60, 24, 9, 4, 2, 1}/256`) supplies an
    additional `0..=6` magnitude with the same sign, giving
    `I2[k] ∈ [-10, 10]`.
  - Undoes the backwards-prediction step with the §4.2.7.5.2 formula
    `res_Q10[k] = (k+1 < d_LPC ? (res_Q10[k+1]*pred_Q8[k])>>8 : 0)
    + ((((I2[k]<<10) - sign(I2[k])*102) * qstep) >> 16)` running
    `k = d_LPC-1` down to `0`. `qstep = 11796 (Q16)` for NB / MB and
    `9830` for WB. The Q8 prediction weight is chosen per-coefficient
    from Table 20 lists A / B (NB/MB) or C / D (WB) via Table 21
    (NB/MB) or Table 22 (WB).
  - Returns `i2[]` and `res_Q10[]`; the §4.2.7.5.3 reconstruction
    (stage-1 codebook lookup + IHMW weights + final `NLSF_Q15[k]`),
    §4.2.7.5.4 stabilization, and §4.2.7.5.5 interpolation are
    deferred to round 7+.

  The RFC 6716 Table 17 row at `I1 = 6` is mislabelled `g` in the
  source PDF; the row's cells (`a c c c c c c c c b`) are valid
  codebook letters and the table is transcribed with the I1 row-label
  restored, matching the table's documented intent.

  30 new unit tests (148 total in the crate) covering:
  - All 16 stage-2 PDFs sum to 256 and their transcribed iCDFs are
    monotone non-increasing with a trailing zero (Tables 15, 16).
  - The Table 19 extension PDF sums to 256 and the iCDF cells match
    `256 - fh[k]`.
  - Tables 17, 18, 21, 22 row widths match `d_LPC` (NB/MB) and
    `d_LPC` / `d_LPC - 1` (WB pred-weight); all entries fall in
    `0..=7` (codebook selection) or `0..=1` (prediction-weight
    selection).
  - Table 17 `I1 = 0` (all-`a`), `I1 = 2`, and the `I1 = 6` typo-row
    spot-checks; Table 18 `I1 = 0` (all-`i`), `I1 = 6` (all-`i`),
    `I1 = 9` (`k j i ...`) spot-checks.
  - Table 20 A[0] = 179, B[0] = 116, A[8] = 163, B[8] = 92,
    C[0] = 175, D[0] = 68, C[14] = 182, D[14] = 155 spot-checks;
    Table 21 / 22 `I1 = 0` rows.
  - `pred_weight` resolves the right A/B and C/D list cells per
    coefficient against the Table 21 / 22 selection rows.
  - End-to-end decode for `(Nb, I1=0)`, `(Mb, I1=5)`, `(Wb, I1=0)`,
    `(Wb, I1=9)` with `i2[k] ∈ [-10, 10]` for every populated
    coefficient.
  - Independent rejection of `I1 = 32` (out of range), `Swb`, and
    `Fb` (SILK never sees SWB / FB after the §4.2.2 hybrid split).
  - `res_Q10[]` from `LsfStage2::decode` exactly reproduces the
    §4.2.7.5.2 formula re-applied to the decoded `i2[]` for both
    NB/MB and WB.
  - Sweep across all 32 I1 values × {NB, MB, WB} confirming every
    decode succeeds and `i2[k] ∈ [-10, 10]` for every coefficient.
  - `RangeDecoder::tell()` is monotone non-decreasing across a
    stage-2 decode (the decoder consumes ≥ 1 bit).

* **Clean-room round 5 (2026-05-22):** RFC 6716 §4.2.7.4 SILK
  subframe quantization-gain decoder behind a new `SubframeGains` /
  `SubframeGainsConfig` API. Two coding paths:
  - Independent (Table 11 signal-type-conditioned 3-bit MSB +
    Table 12 uniform 3-bit LSB) joined into `gain_index ∈ 0..=63`
    and clamped with `log_gain = max(gain_index, previous_log_gain
    - 16)` per §4.2.7.4. The clamp is skipped when the caller
    passes no previous gain (decoder reset / side-channel
    previously uncoded / packet loss).
  - Delta (Table 13 41-symbol iCDF) folded into the previous gain
    via `log_gain = clamp(0, max(2*delta - 16, prev + delta - 4),
    63)`.
  The first subframe of a SILK frame uses the independent path
  iff the §4.2.7.4 enumeration triggers ("first SILK frame of its
  type for this channel in the current Opus frame, OR previous
  SILK frame of the same type was not coded"); every other
  subframe uses the delta path. Output is the integer `log_gain
  ∈ 0..=63` per subframe; the §4.2.7.4 tail-end conversion to
  `gain_Q16` via `silk_log2lin` is part of the excitation stage
  and not wired up here.
  20 new unit tests (88 total in the crate) covering PDF→iCDF
  transcription self-checks (Tables 11 / 12 / 13 each sum to
  256), all four `SignalType` → iCDF routings, the §4.2.7.4
  clamp behaviour across the four prev-value regimes (None, low,
  high, sub-16-saturate-to-zero), the delta path's dual-max +
  clamp formula reproduced against an independent range-decoder
  pass, end-to-end decode for mono-inactive 4-subframe,
  mono-unvoiced 2-subframe-with-prev, mono-voiced 4-subframe with
  high prev (asserting the floor clamp), the rejection of a
  pathological "first-subframe-delta without prev" config and
  num_subframes ∉ {2, 4}, and a four-subframe chain-consistency
  check that re-derives the gain chain from the raw PDF reads.

* **Clean-room round 4 (2026-05-21):** RFC 6716 §4.2.7.1 through
  §4.2.7.5.1 SILK frame-header decoder behind a new `SilkFrameHeader`
  type. The caller passes a `SilkFrameHeaderConfig` describing whether
  the current SILK frame is mid- or side-channel of a stereo Opus
  frame, whether the side channel is otherwise required (driving the
  presence of the mid-only flag), the frame kind (regular-inactive /
  regular-active / LBRR), and the SILK-layer audio bandwidth (NB / MB
  / WB). `SilkFrameHeader::decode` returns:
  - `stereo_pred: Option<StereoPredictionWeights>` per §4.2.7.1 with
    the three sub-symbols (Table 6 stage-1 25-cell, two stage-2
    3-cell, two stage-3 5-cell) composed into `(w0_Q13, w1_Q13)` per
    the spec formula and Table 7 weight table.
  - `mid_only_flag: Option<bool>` per §4.2.7.2 (Table 8 PDF
    `{192, 64}/256`).
  - `frame_type: u8` in `0..=5` per §4.2.7.3 (Table 9 inactive /
    active PDFs; active rows use a 4-entry tail-truncated iCDF with
    a +2 caller offset, since the §4.1.3.3 primitive cannot model
    leading-zero-mass cells).
  - `signal_type: SignalType`, `qoff_type: QuantizationOffsetType`
    decoded from `frame_type` via Table 10.
  - `lsf_stage1: u8` in `0..32` per §4.2.7.5.1 with the PDF chosen
    from Table 14 by `(bandwidth, signal_type)`.
  17 new unit tests (68 total in the crate) covering PDF→iCDF
  transcription self-checks (Tables 6 / 8 / 9 / 14 each sum to 256
  with monotone-decreasing iCDFs), the Table 7 weight-table symmetry
  (`w[15-k] == -w[k]`), the Table 10 frame-type → `(signal, qoff)`
  mapping, end-to-end decode against the range coder for mono
  inactive / mono active / stereo mid-channel / stereo side-channel
  / LBRR configurations, and a random-buffer sweep of
  `decode_stereo_pred` to confirm `wi0/wi1 ≤ 14` clamping keeps the
  Table 7 lookup in-bounds.
* **Clean-room round 3 (2026-05-21):** RFC 6716 §4.1 range decoder
  behind a new `RangeDecoder` API — the shared entropy primitive
  consumed by every SILK and CELT symbol. Implements §4.1.1
  initialization, §4.1.2 generic symbol decode, §4.1.2.1
  renormalization (with §4.1.4 zero-extension past EOF), §4.1.3.1
  `decode_bin` for power-of-two `ft`, §4.1.3.2 `dec_bit_logp` for
  `2^-logp` binaries, §4.1.3.3 `dec_icdf` for inverse-CDF tables,
  §4.1.4 `dec_bits` LSB-first raw bits from the END of the frame,
  §4.1.5 `dec_uint` (both small-ftb range-only and large-ftb
  range-plus-raw branches, with the corrupt-frame sticky error
  latch), §4.1.6.1 `tell()`, and §4.1.6.2 `tell_frac()` (with the
  `tell() == ceil(tell_frac() / 8.0)` identity holding across mixed
  operations). The sibling `oxideav-celt` crate carries an
  independent clean-room copy of the same primitive — both own
  their own copy until a shared low-level primitives crate exists.
  19 new unit tests (51 total in the crate).
* **Clean-room round 2 (2026-05-21):** RFC 6716 §3.2 frame-packing
  parser behind a new `OpusPacket::parse` entry point covering all
  four `c` codes:
  * Code 0 (§3.2.2) — one frame, remaining `N - 1` bytes.
  * Code 1 (§3.2.3) — two equal-size frames; R3 odd-payload rejection.
  * Code 2 (§3.2.4) — one- or two-byte §3.2.1 length sequence then
    `N1` + remainder; R4 length-exceeds-remaining rejection.
  * Code 3 (§3.2.5) — `M ∈ 1..=48` (R5) frame-count byte with the
    `v|p|M` bit layout; optional Opus padding with the §3.2.5
    255-byte-extension chain; CBR with R6 `R % M == 0` check; VBR
    with `M - 1` declared lengths and implicit final-frame size,
    R7 length-overrun rejection.
  * §3.2.1 length helper: `0` (DTX), `1..=251` single-byte,
    `252..=255` two-byte `(second * 4 + first)` up to 1275 (R2).
  Frame slices borrow from the input buffer via `OpusPacket::frames()
  -> &[&[u8]]`; padding length is exposed separately. Adds
  `Error::MalformedPacket` for §3.2 requirement violations. 27 new
  unit tests (32 total in the crate).
* **Clean-room round 1 (2026-05-20):** RFC 6716 §3.1 packet TOC byte
  parser. `OpusTocByte::parse` / `OpusTocByte::from_byte` decode the
  five-bit `config` against Table 2 (32 mode × bandwidth × frame-size
  tuples), the `s` stereo bit against the Table 3 prose, and the `c`
  frame-count code against the Table 4 prose (one frame /
  two-equal / two-unequal / arbitrary). `frame_count_range()` gives
  the implied `(min, max)` frame count without consulting further
  bytes (code 3 reports the legal `(1, 48)` range derived from
  §3.2.5's "no more than 120 ms total"). Five unit tests sweep the
  full enumeration plus the R1 empty-packet rejection.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Higher-level decode / encode paths still return
  `Error::NotImplemented`. A clean-room re-implementation of the
  SILK / CELT inner decoders, the §3.2 frame-packing layer, the §4
  range coder, and the §5 encoder pipeline against RFC 6716 +
  RFC 8251 + RFC 7587 + RFC 7845 is planned for subsequent rounds.
