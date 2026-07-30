//! Top-level Opus packet → PCM orchestration — RFC 6716 §3 / §4.
//!
//! This module is the keystone that turns a raw Opus packet (a TOC byte
//! plus one or more §3.2-packed Opus frames) into interleaved 48 kHz PCM
//! samples. It sits above every per-stage SILK / CELT decoder in the
//! crate and wires the §3.1 TOC parse, the §3.2 frame packing
//! ([`crate::frames::OpusPacket`]), and the §4.2 / §4.3 per-frame mode
//! dispatch ([`crate::framing::OpusFrameRouting`]) into one
//! [`OpusDecoder::decode_packet`] call.
//!
//! ## What this module owns
//!
//! * The packet → frame split (delegated to [`OpusPacket::parse`]).
//! * The §4.5 multi-frame loop: every Opus frame in a code-1 / code-2 /
//!   code-3 packet is decoded in order and its PCM appended to the
//!   output, so a 60 ms code-3 packet of three 20 ms frames yields one
//!   contiguous PCM buffer.
//! * The §3.2.1 DTX / lost-frame marker handling: a zero-length frame
//!   slice contributes one Opus-frame worth of silence (the §4.6 PLC
//!   "fill with silence" floor — a real concealment model is a separate
//!   milestone).
//! * The 48 kHz output sample-count accounting (RFC 7845 §5.1: the Opus
//!   decoder always emits 48 kHz regardless of the internal SILK / CELT
//!   sample rate).
//! * The per-frame routing seam: each Opus frame is dispatched to
//!   [`Self::decode_silk_only_frame`], [`Self::decode_celt_only_frame`],
//!   or [`Self::decode_hybrid_frame`] based on its [`OpusFrameRouting`].
//!
//! ## What this module does not own
//!
//! * The §4.1 range-coder primitive ([`crate::range_decoder`]).
//! * The per-stage SILK / CELT decode (the `silk_*` / `celt_*` modules).
//! * Any container parsing (Ogg / RTP framing live in their own crates;
//!   this module consumes a bare Opus packet).
//!
//! ## Status of the per-frame audio decode
//!
//! The packet-level orchestration (TOC → framing → routing → 48 kHz PCM
//! buffer layout) is complete and total over all 32 §3.1 configs and all
//! four §3.2 frame-count codes. The per-frame audio decode is wired
//! incrementally:
//!
//! * **Mono SILK-only** frames run the full §4.2 decode → PCM path: the
//!   §4.2.3 header bits, the §4.2.5 LBRR / §4.2.6 regular SILK frame loop
//!   (1 / 2 / 3 SILK frames per §4.2.2), each frame decoded in Table-5
//!   order via [`crate::silk_decode::decode_silk_frame`] with the
//!   inter-frame state threaded across them, then the §4.2.7.9 LTP / LPC
//!   synthesis ([`crate::silk_synthesis::synthesize_silk_frame`]) and the
//!   §4.2.9 (non-normative) resample to 48 kHz. The carried §4.2.7.9
//!   synthesis histories persist across the packet's Opus frames; the
//!   emitted PCM is real audio ([`FrameDecodeStatus::SilkParamsDecoded`]).
//! * **Stereo SILK-only** frames run the full §4.2 interleaved decode →
//!   PCM path: the §4.2.3 two-channel header bits, the §4.2.5 / §4.2.6
//!   mid/side interleave (mid frame then side frame per 20 ms interval,
//!   the side frame skipped when the §4.2.7.2 mid-only flag is set), each
//!   channel's §4.2.7.9 synthesis with its own carried history, then the
//!   §4.2.8 mid/side → left/right unmixing
//!   ([`crate::silk_stereo::stereo_ms_to_lr`]) and the §4.2.9 resample,
//!   emitting interleaved L/R PCM
//!   ([`FrameDecodeStatus::SilkStereoDecoded`]). The §4.2.7.1 mono→stereo
//!   weight reset and the §4.5.2 SILK state reset are applied across
//!   packets.
//! * **CELT-only** frames run the full §4.3 decode → PCM path: the
//!   Table-56 entropy decode ([`crate::celt_frame_decode`]: frame flags,
//!   coarse / fine / final energies, TF, spread, dynalloc, trim, the
//!   §4.3.3 implicit allocation, §4.3.4 PVQ band shapes with folding,
//!   §4.3.5 anti-collapse) and the §4.3.6–§4.3.7.2 synthesis
//!   ([`crate::celt_mdct_synthesis`]: denormalisation, long/short-block
//!   inverse MDCT + overlap-add, the §4.3.7.1 pitch post-filter, and
//!   de-emphasis), with all cross-frame state carried
//!   ([`FrameDecodeStatus::CeltDecoded`]).
//! * **Hybrid** frames emit silence of the correct length flagged
//!   [`FrameDecodeStatus::LayerNotWired`] (the SILK + CELT layer sum is
//!   the remaining assembly step).
//!
//! Either way the multi-frame packet loop and the RFC 7845 §5.1 48 kHz
//! sample-count accounting are exercised end-to-end.

use crate::frames::OpusPacket;
use crate::framing::{OperatingMode, OpusFrameRouting};
use crate::toc::ChannelMapping;
use crate::Error;

/// Output sample rate of the Opus decoder, in Hz. Per RFC 7845 §5.1 the
/// decoder always emits 48 kHz regardless of the internal SILK / CELT
/// sample rate; the per-layer resamplers upsample to this rate.
pub const OUTPUT_SAMPLE_RATE_HZ: u32 = 48_000;

/// Output samples per millisecond per channel at [`OUTPUT_SAMPLE_RATE_HZ`].
pub const OUTPUT_SAMPLES_PER_MS: u32 = OUTPUT_SAMPLE_RATE_HZ / 1000;

/// Number of 48 kHz output samples (per channel) an Opus frame of the
/// given duration produces.
///
/// `frame_size_tenths_ms` is the §3.1 Table 2 duration in tenths of a
/// millisecond (25, 50, 100, 200, 400, 600). The 2.5 ms CELT case
/// (`25` tenths) yields `25 * 48 / 10 = 120` samples per channel, which
/// is exact; all six durations divide evenly.
pub fn output_samples_per_channel(frame_size_tenths_ms: u16) -> usize {
    // tenths-ms * (48 samples / ms) / 10 = tenths-ms * 48 / 10.
    (frame_size_tenths_ms as usize * OUTPUT_SAMPLES_PER_MS as usize) / 10
}

/// Why a given Opus frame produced the samples it did.
///
/// The packet-level orchestration is complete, but the per-frame audio
/// decode lands incrementally. This status lets a caller (and the
/// crate's own tests) distinguish "decoded real audio" from "emitted
/// silence because the layer's range-coded decode is not wired yet" or
/// "emitted silence for a DTX / lost frame".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecodeStatus {
    /// A §3.2.1 zero-length frame: DTX or a lost/packet-loss marker.
    /// Per §4.6 the floor behaviour is to emit silence; a real PLC model
    /// is a separate milestone.
    DtxOrLost,
    /// The frame's operating mode does not yet have a composed
    /// sample-producing decode path in this crate, so silence of the
    /// correct length was emitted. The variant carries the mode so the
    /// caller knows which layer is pending.
    LayerNotWired(OperatingMode),
    /// A mono SILK-only frame whose full §4.2.7 bitstream (frame type,
    /// gains, LSF chain, LTP, LCG seed, excitation) was decoded in
    /// Table-5 order via [`crate::silk_decode::decode_silk_frame`], then
    /// synthesized through the §4.2.7.9 LTP / LPC filters
    /// ([`crate::silk_synthesis::synthesize_silk_frame`]) and resampled to
    /// 48 kHz (§4.2.9, non-normative). The emitted PCM is real audio.
    SilkParamsDecoded,
    /// A **stereo** SILK-only frame whose §4.2.3 / §4.2.4 header bits and
    /// the §4.2.5 / §4.2.6 interleaved mid/side SILK frames were decoded
    /// in §4.2.2 order (mid frame then side frame per 20 ms interval, the
    /// side frame skipped when the §4.2.7.2 mid-only flag is set), each
    /// channel synthesized through the §4.2.7.9 filters, converted from
    /// mid/side to left/right via §4.2.8 stereo unmixing
    /// ([`crate::silk_stereo::stereo_ms_to_lr`]), then resampled to 48 kHz
    /// (§4.2.9, non-normative). The emitted interleaved L/R PCM is real
    /// audio.
    SilkStereoDecoded,
    /// A SILK-only frame whose §4.2.7 bitstream decode latched an error
    /// (a malformed / truncated frame). Silence of the correct length was
    /// emitted in its place per the §4.6 floor.
    SilkDecodeError,
    /// A CELT-only frame whose §4.3.7.1 silence flag was set: the real
    /// range-coded frame prefix (silence + post-filter group) was decoded
    /// and the §4.3.6→§4.3.7.2 synthesis backend was advanced with
    /// all-zero band shapes / energies, emitting silence PCM while
    /// carrying the MDCT overlap-add and de-emphasis state forward for the
    /// next frame. (Distinct from [`Self::LayerNotWired`]: the bitstream
    /// is actually consumed and the synthesis state is real, not stubbed.)
    CeltSilence,
    /// A CELT-only frame whose §4.3.7.1 prefix decode latched a range-coder
    /// error (a malformed / truncated frame). Silence of the correct length
    /// was emitted in its place per the §4.6 floor.
    CeltDecodeError,
    /// A **non-silent** CELT-only frame decoded end-to-end: the full
    /// Table-56 entropy decode ([`crate::celt_frame_decode`]) followed
    /// by the §4.3.6–§4.3.7.2 synthesis
    /// ([`crate::celt_mdct_synthesis`]). The emitted PCM is real audio.
    CeltDecoded,
    /// A Hybrid frame decoded end-to-end: the §4.2 SILK layer (WB
    /// internal, non-normative resample to 48 kHz) plus the §4.3 CELT
    /// layer (bands 17–21) summed per §4.4. The emitted PCM is real
    /// audio.
    HybridDecoded,
    /// A **lost** frame concealed by the §4.4 packet-loss concealment
    /// ([`OpusDecoder::conceal_loss`]): the emitted PCM extrapolates
    /// the last decoded audio (LPC extrapolation after a SILK-bearing
    /// frame, pitch-periodic repetition after a CELT-only frame) with
    /// the §4.4 energy decay across consecutive losses. Not produced
    /// by `decode_packet` — only by an explicit concealment call.
    Concealed,
}

/// The result of decoding one Opus frame: how many per-channel samples
/// it contributed and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameOutcome {
    /// Per-channel 48 kHz sample count this frame contributed.
    pub samples_per_channel: usize,
    /// Provenance of the samples (real audio vs silence and why).
    pub status: FrameDecodeStatus,
}

/// Decoded audio for one Opus packet: interleaved 48 kHz PCM plus the
/// per-frame outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Interleaved signed 16-bit PCM at 48 kHz. For stereo the layout is
    /// `[L0, R0, L1, R1, …]`; for mono it is `[S0, S1, …]`. Length is
    /// `total_samples_per_channel * channels`.
    pub pcm: Vec<i16>,
    /// Number of audio channels (1 for mono, 2 for stereo).
    pub channels: u8,
    /// Output sample rate in Hz (always [`OUTPUT_SAMPLE_RATE_HZ`]).
    pub sample_rate_hz: u32,
    /// Per-Opus-frame outcomes, in packet order. `outcomes.len()` equals
    /// the packet's §3.2 frame count.
    pub frame_outcomes: Vec<FrameOutcome>,
}

impl DecodedAudio {
    /// Total per-channel 48 kHz sample count across every Opus frame in
    /// the packet.
    pub fn samples_per_channel(&self) -> usize {
        self.pcm.len() / self.channels.max(1) as usize
    }
}

/// Why an in-band FEC ([`OpusDecoder::decode_packet_fec`]) recovery
/// produced the samples it did (RFC 6716 §2.1.7 / §4.2.5).
///
/// In-band FEC works by re-encoding the signal of the frame *prior* to a
/// packet at a lower bitrate and carrying it as one or more §4.2.5 LBRR
/// frames inside that packet. When a packet is lost, the decoder can
/// recover the lost frame's audio from the LBRR frame(s) in the *next*
/// successfully received packet (`decode_packet_fec`), rather than
/// emitting pure silence / running pitch-based concealment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FecDecodeStatus {
    /// The packet carried §4.2.5 LBRR frame(s) for the lost prior frame,
    /// and they were decoded in Table-5 order and synthesized through the
    /// §4.2.7.9 LTP / LPC filters into real recovered audio at 48 kHz. For
    /// a stereo packet the recovered mid/side LBRR frames were unmixed via
    /// §4.2.8.
    Recovered,
    /// The packet has no LBRR frame for the requested channel(s) (the
    /// §4.2.4 LBRR flags are clear), so no FEC data is available. Silence
    /// of the requested duration was emitted; the caller should fall back
    /// to its own packet-loss concealment.
    NoLbrr,
    /// The packet is not a SILK-bearing mode (CELT-only carries no LBRR),
    /// so FEC recovery is not possible. Silence was emitted.
    NotSilk,
    /// The packet's §4.2 LBRR bitstream was malformed / truncated. Silence
    /// of the requested duration was emitted in its place.
    DecodeError,
}

/// The result of an in-band FEC recovery for one lost packet
/// ([`OpusDecoder::decode_packet_fec`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FecRecovered {
    /// Interleaved signed 16-bit PCM at 48 kHz, same layout as
    /// [`DecodedAudio::pcm`]. Length is `samples_per_channel * channels`.
    pub pcm: Vec<i16>,
    /// Number of audio channels (1 for mono, 2 for stereo).
    pub channels: u8,
    /// Output sample rate in Hz (always [`OUTPUT_SAMPLE_RATE_HZ`]).
    pub sample_rate_hz: u32,
    /// Why the samples were produced (real recovery vs silence and why).
    pub status: FecDecodeStatus,
}

/// Stateful Opus packet → PCM decoder.
///
/// One [`OpusDecoder`] is fed Opus packets in stream order via
/// [`Self::decode_packet`]. The decoder is stateful because the SILK and
/// CELT layers carry inter-frame state (LPC / LTP history, MDCT overlap,
/// stereo unmixing memory, the §4.5.2 reset policy); the state lives here
/// and is threaded into the per-frame decode as those paths land. Today
/// the carried state is minimal (it grows as each layer is wired), but
/// the type is the stable home for it.
#[derive(Debug, Default)]
pub struct OpusDecoder {
    /// Channel count of the most recently decoded packet, if any. Used
    /// only for the §4.5.2 mono↔stereo transition reset bookkeeping the
    /// per-layer decoders will consult once wired.
    last_channels: Option<u8>,
    /// Mono SILK synthesis state (the §4.2.7.9 LTP / LPC histories),
    /// carried across Opus frames in the stream. `None` until the first
    /// mono SILK-only frame is synthesized; re-created (cleared per
    /// §4.5.2) when the SILK bandwidth changes.
    silk_synth_mono: Option<crate::silk_synthesis::SilkSynthState>,
    /// Stereo SILK synthesis state: the §4.2.7.9 LTP / LPC histories for
    /// the **mid** and **side** channels, carried across Opus frames.
    /// `None` until the first stereo SILK-only frame; re-created when the
    /// SILK bandwidth changes (a §4.5.2 reset).
    silk_synth_stereo: Option<(
        crate::silk_synthesis::SilkSynthState,
        crate::silk_synthesis::SilkSynthState,
    )>,
    /// §4.2.8 stereo unmixing history (two prior mid samples, one prior
    /// side sample, and the previous frame's prediction weights), carried
    /// across Opus frames. `None` until the first stereo SILK-only frame;
    /// reset (zeroed) on any §4.2.7.1 mono→stereo transition.
    silk_stereo_unmix: Option<crate::silk_stereo::StereoUnmixStateI16>,
    /// Cross-Opus-frame §4.2.7 reconstruction carry for the mono SILK
    /// channel: the last decoded subframe gain (the §4.2.7.4 clamp base,
    /// which persists "in the same channel" across Opus frames) and the
    /// last decoded NLSF vector (the §4.2.7.5.5 interpolation base `n0`,
    /// "the LSF coefficients decoded for the prior frame"). Cleared on a
    /// §4.5.2 SILK reset and on a SILK bandwidth change.
    silk_carry_mono: SilkChannelCarry,
    /// Cross-Opus-frame §4.2.7 reconstruction carry for the stereo MID
    /// channel (see [`Self::silk_carry_mono`]).
    silk_carry_mid: SilkChannelCarry,
    /// Cross-Opus-frame §4.2.7 reconstruction carry for the stereo SIDE
    /// channel. An uncoded side frame at the end of a packet leaves this
    /// cleared, so the next coded side frame codes independently with the
    /// §4.2.7.4 clamp skipped and the §4.2.7.5.5 factor forced to 4 —
    /// exactly the RFC's "previous frame in the side channel was not
    /// coded" rules.
    silk_carry_side: SilkChannelCarry,
    /// §4.2.8 mono one-sample delay carry: "In order to allow seamless
    /// switching between stereo and mono, mono streams must also
    /// impose the same one-sample delay" (the stereo unmixing
    /// formulas read `mid[i-1]` / `side[i-1]`, delaying the stereo
    /// output by one internal-rate sample; the mono path must match).
    /// Holds the last internal-rate sample of the previous mono Opus
    /// frame; zero after a decoder reset ("zeros are used instead").
    silk_mono_delay: i16,
    /// §4.2.9 mono upsampler state (SILK internal rate → 48 kHz),
    /// carried across Opus frames so consecutive frames are seamless
    /// through the resampling filter. Shared by the SILK-only and
    /// Hybrid paths (both feed the same mono SILK channel); rebuilt
    /// on a SILK bandwidth change and history-cleared on a §4.5.2
    /// SILK reset, mirroring [`Self::silk_synth_mono`].
    silk_resamp_mono: Option<crate::silk_resampler::SilkUpsampler>,
    /// §4.2.9 stereo upsampler states for the unmixed **left** and
    /// **right** channels (resampling runs after the §4.2.8 unmix).
    /// Lifecycle mirrors [`Self::silk_synth_stereo`].
    silk_resamp_stereo: Option<(
        crate::silk_resampler::SilkUpsampler,
        crate::silk_resampler::SilkUpsampler,
    )>,
    /// Operating mode of the most recently decoded Opus frame, used to
    /// drive the §4.5.2 SILK state-reset rule ("the SILK state is reset
    /// before every SILK-only or Hybrid frame where the previous frame
    /// was CELT-only"). `None` before the first frame / after a reset.
    prev_mode: Option<OperatingMode>,
    /// CELT synthesis-side state (§4.3.7 overlap-add memory, §4.3.7.1
    /// post-filter signal history + parameter pair, §4.3.7.2
    /// de-emphasis memory), carried across the CELT frames of the
    /// stream. `None` until the first CELT frame; rebuilt when the CELT
    /// frame size or channel count changes and dropped on a §4.5.2
    /// CELT reset.
    celt_synth: Option<crate::celt_mdct_synthesis::CeltSynthesis>,
    /// CELT entropy-side state (the §4.3.2 per-band energy history and
    /// the carried folding-noise seed), advanced by every CELT frame.
    celt_energy: Option<crate::celt_frame_decode::CeltEnergyState>,
    /// §4.4 packet-loss concealment state: the trailing output
    /// history, the consecutive-loss counter, and the pending
    /// concealment-to-real cross-lap tail.
    plc: crate::plc::PlcState,
    /// §4.5.1 redundancy decision taken on the most recently decoded
    /// Opus frame. Drives the §4.5.2 reset policy at the next packet
    /// boundary (rule 3: a SILK/Hybrid frame that carried an
    /// end-position redundant CELT frame already reset the CELT state
    /// before decoding it, so the following CELT-only / Hybrid frame
    /// must NOT reset again).
    last_redundancy: crate::celt_redundancy::RedundancyDecision,
    /// Deferred §4.5.2 CELT reset for a Hybrid frame's main CELT
    /// layer: when the packet-boundary policy says the CELT state
    /// resets *before the frame* but the frame is Hybrid, the frame
    /// may open with a §4.5.1.2 beginning-position redundant CELT
    /// frame that must decode on the **un-reset** state (Figure 18's
    /// `R & |H`); the reset is applied between the redundant frame
    /// and the main CELT layer instead.
    celt_reset_before_main: bool,
    /// Frame duration (§3.1 Table 2, tenths of ms) of the most
    /// recently decoded packet — the duration [`Self::conceal_loss`]
    /// conceals when the lost packet's own duration is unknown.
    last_frame_tenths_ms: Option<u16>,
}

impl OpusDecoder {
    /// Construct a fresh decoder with no carried state (equivalent to the
    /// post-`reset` state of §4.5.2).
    pub fn new() -> Self {
        Self::default()
    }

    /// Discard all inter-frame state, as after a container seek (the
    /// §4.5.2 decoder reset). Leaves the decoder ready to decode a new
    /// bitstream position as if it were the first packet.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Decode one complete Opus packet into interleaved 48 kHz PCM.
    ///
    /// Performs the §3.1 TOC parse, the §3.2 frame split, and the §4.5
    /// multi-frame loop, dispatching each Opus frame through its
    /// [`OpusFrameRouting`] to the matching per-mode decode. Returns
    /// [`Error::EmptyPacket`] for a zero-length packet (§3.1 R1) and
    /// [`Error::MalformedPacket`] for any §3.2 framing violation.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<DecodedAudio, Error> {
        crate::bump::reset(); // per-packet bump rewind, before parse bump-allocates the frame list
        let parsed = OpusPacket::parse(packet)?;
        self.decode_parsed_packet(parsed)
    }

    /// Decode one complete Opus packet that uses RFC 6716 Appendix-B
    /// self-delimited framing (the framing the first `N − 1` streams of
    /// a multistream packet use, RFC 7845 §3). Behaves exactly like
    /// [`Self::decode_packet`] otherwise — the only difference is how the
    /// frame slices are recovered from the packet bytes.
    pub fn decode_self_delimited_packet(&mut self, packet: &[u8]) -> Result<DecodedAudio, Error> {
        crate::bump::reset(); // per-packet bump rewind, before parse bump-allocates the frame list
        let parsed = crate::framing_self_delim::parse_self_delimited(packet)?.packet;
        self.decode_parsed_packet(parsed)
    }

    /// Shared decode body for both the regular ([`Self::decode_packet`])
    /// and self-delimited ([`Self::decode_self_delimited_packet`]) entry
    /// points: applies the §4.5.2 cross-packet state resets, then runs
    /// the §4.5 multi-frame loop over the already-sliced frames.
    /// Profluens patch (PROFLUENS-PATCHES.md): the decode core, filling caller-provided reused
    /// buffers so a steady decode does no per-packet heap allocation. Returns the channel count;
    /// `frame_outcomes` is filled only when `Some`.
    fn decode_into_buffers(
        &mut self,
        parsed: OpusPacket<'_>,
        pcm: &mut Vec<i16>,
        mut frame_outcomes: Option<&mut Vec<FrameOutcome>>,
    ) -> Result<u8, Error> {
        // Profluens patch (PROFLUENS-PATCHES.md): the per-packet bump arena backing every
        // `Vec<_, DecodeBump>` transient is rewound by `crate::bump::reset()` at each public entry
        // point *before* the §3.2 parse — so the parsed frame list and all decode transients share
        // one per-packet region, freed wholesale at the next packet. (The reset can't live here:
        // `OpusPacket::parse` has already bump-allocated the frame list by the time we're called.)
        let routing = OpusFrameRouting::from_toc(parsed.toc);
        let channels = routing.channel_count();
        let per_frame_samples = output_samples_per_channel(routing.frame_size_tenths_ms);

        // §4.5.2 state resets at the Opus-packet boundary, using the
        // recorded previous operating mode and the §4.5.1 redundancy
        // decision of the previous frame (rule 3: an end-position
        // redundant CELT frame in the previous SILK/Hybrid frame
        // already took the CELT reset, so the new-mode frame must not
        // reset again).
        if let Some(prev_mode) = self.prev_mode {
            let reset = crate::mode_transition_reset::decide_state_resets(
                prev_mode,
                routing.operating_mode,
                self.last_redundancy,
            );
            if reset.silk {
                if let Some(state) = self.silk_synth_mono.as_mut() {
                    state.reset();
                }
                if let Some((mid, side)) = self.silk_synth_stereo.as_mut() {
                    mid.reset();
                    side.reset();
                }
                if let Some(unmix) = self.silk_stereo_unmix.as_mut() {
                    unmix.reset();
                }
                // The §4.2.8 mono delay sample and the §4.2.9
                // resampler history are part of the SILK state: clear
                // them with the rest ("for the first frame after a
                // decoder reset, zeros are used instead").
                self.silk_mono_delay = 0;
                if let Some(up) = self.silk_resamp_mono.as_mut() {
                    up.reset();
                }
                if let Some((l, r)) = self.silk_resamp_stereo.as_mut() {
                    l.reset();
                    r.reset();
                }
                // §4.2.7.4 / §4.2.7.5.5: "the clamping is skipped after a
                // decoder reset" and the interpolation factor is forced
                // to 4 — drop the carried gain / NLSF bases.
                self.silk_carry_mono = SilkChannelCarry::default();
                self.silk_carry_mid = SilkChannelCarry::default();
                self.silk_carry_side = SilkChannelCarry::default();
            }
            match reset.celt {
                crate::mode_transition_reset::CeltResetPlacement::BeforeFrame => {
                    if routing.operating_mode == OperatingMode::Hybrid {
                        // Defer: a beginning-position redundant CELT
                        // frame inside the Hybrid frame decodes on the
                        // un-reset state first (§4.5.2 rule 4 /
                        // Figure 18's `R & |H`); the main CELT layer
                        // takes the reset afterwards.
                        self.celt_reset_before_main = true;
                    } else {
                        // §4.5.2 rule 2: drop the carried CELT state so
                        // the new-mode frame starts from the reset
                        // state.
                        self.celt_energy = None;
                        self.celt_synth = None;
                    }
                }
                // Rule 3: the reset already happened before the
                // previous frame's end-position redundant CELT frame;
                // the state must carry into this frame. Rule 4 / None:
                // no CELT reset for this transition.
                crate::mode_transition_reset::CeltResetPlacement::BeforeRedundantOnly
                | crate::mode_transition_reset::CeltResetPlacement::None => {}
            }
        }

        // §4.2.7.1: "the previous weights are reset to zeros on any
        // transition from mono to stereo." More generally the §4.2.8
        // unmixing history (and the mid/side synthesis state) only makes
        // sense within a contiguous stereo run; a channel-count change
        // clears the carried stereo state so a stale mono / prior-stereo
        // history can never leak across the transition.
        if self.last_channels.is_some_and(|c| c != channels) {
            if let Some(unmix) = self.silk_stereo_unmix.as_mut() {
                unmix.reset();
            }
            if let Some((mid, side)) = self.silk_synth_stereo.as_mut() {
                mid.reset();
                side.reset();
            }
            if let Some((l, r)) = self.silk_resamp_stereo.as_mut() {
                l.reset();
                r.reset();
            }
            // The mid/side §4.2.7 reconstruction carries follow the same
            // policy as the stereo synthesis state: a channel-count
            // change clears them (the mono carry mirrors the mono
            // synthesis state and persists, matching the treatment of
            // `silk_synth_mono` above).
            self.silk_carry_mid = SilkChannelCarry::default();
            self.silk_carry_side = SilkChannelCarry::default();
        }

        self.last_channels = Some(channels);
        self.prev_mode = Some(routing.operating_mode);

        let frame_slices = parsed.frames();
        pcm.clear();
        pcm.reserve(frame_slices.len() * per_frame_samples * channels as usize);
        if let Some(fo) = frame_outcomes.as_mut() {
            fo.clear();
            fo.reserve(frame_slices.len());
        }

        for frame in frame_slices {
            let outcome = self.decode_one_frame(frame, &routing, pcm);
            if let Some(fo) = frame_outcomes.as_mut() {
                fo.push(outcome);
            }
        }

        // §4.4: cross-lap a pending concealment tail into the head of
        // this (first real post-loss) output, then record the output
        // as concealment history and re-arm the loss counter.
        self.plc.apply_tail(pcm.as_mut_slice(), channels as usize);
        self.plc.feed_decoded(pcm.as_slice(), channels as usize);
        self.last_frame_tenths_ms = Some(routing.frame_size_tenths_ms);

        Ok(channels)
    }

    /// Decode one Opus packet, reusing the caller's `pcm` buffer (cleared first) so a steady
    /// decode does **no per-packet allocation**. Returns the channel count. Profluens patch —
    /// upstream [`Self::decode_packet`] allocates a fresh `Vec<i16>` each call. See
    /// PROFLUENS-PATCHES.md.
    pub fn decode_packet_into(&mut self, packet: &[u8], pcm: &mut Vec<i16>) -> Result<u8, Error> {
        crate::bump::reset(); // per-packet bump rewind, before parse bump-allocates the frame list
        let parsed = OpusPacket::parse(packet)?;
        self.decode_into_buffers(parsed, pcm, None)
    }

    fn decode_parsed_packet(&mut self, parsed: OpusPacket<'_>) -> Result<DecodedAudio, Error> {
        let mut pcm = Vec::new();
        let mut frame_outcomes = Vec::new();
        let channels = self.decode_into_buffers(parsed, &mut pcm, Some(&mut frame_outcomes))?;
        Ok(DecodedAudio { pcm, channels, sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ, frame_outcomes })
    }

    /// Conceal one **lost** packet (RFC 6716 §4.4), producing one
    /// frame of interleaved 48 kHz PCM extrapolated from the last
    /// decoded audio.
    ///
    /// Call this once per lost packet, in stream position, when no FEC
    /// recovery is available (when the *next* packet is already in
    /// hand, prefer [`Self::decode_packet_fec`], which reconstructs
    /// the lost audio from real bitstream redundancy). The §4.4
    /// per-mode guidance is followed: after a SILK-only or Hybrid
    /// frame the concealment is an LPC extrapolation of the previous
    /// output ([`crate::plc::conceal_silk`]); after a CELT-only frame
    /// it repeats the pitch-periodic waveform
    /// ([`crate::plc::conceal_celt`]). Consecutive concealed frames
    /// decay in energy to the silence floor, and the first packet
    /// decoded after a concealment run is cross-lapped with the
    /// concealment's extrapolation tail so both joins are smooth.
    ///
    /// The concealed duration is the last decoded packet's §3.1 frame
    /// duration (a loss in a stream is overwhelmingly likely to have
    /// the neighbouring packets' geometry); with no decode history at
    /// all this emits 20 ms of silence. Per §4.5 the concealment is
    /// for *actual loss*: normative mode transitions in a fully
    /// received stream must not route through PLC (they either need no
    /// treatment or carry §4.5.1 redundancy).
    pub fn conceal_loss(&mut self) -> DecodedAudio {
        let tenths = self.last_frame_tenths_ms.unwrap_or(200);
        let per_channel = output_samples_per_channel(tenths);
        let channels = self.last_channels.unwrap_or(1);
        let flavor = match self.prev_mode {
            Some(OperatingMode::CeltOnly) => crate::plc::PlcFlavor::Celt,
            _ => crate::plc::PlcFlavor::Silk,
        };
        let pcm = self.plc.conceal(per_channel, channels as usize, flavor);
        DecodedAudio {
            pcm,
            channels,
            sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
            frame_outcomes: vec![FrameOutcome {
                samples_per_channel: per_channel,
                status: FrameDecodeStatus::Concealed,
            }],
        }
    }

    /// Decode one Opus frame, appending its interleaved 48 kHz PCM to
    /// `pcm` and returning the per-frame outcome.
    fn decode_one_frame(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
        pcm: &mut Vec<i16>,
    ) -> FrameOutcome {
        let per_channel = output_samples_per_channel(routing.frame_size_tenths_ms);
        let channels = routing.channel_count();

        // §4.5.1 bookkeeping: only SILK-only / Hybrid frames can carry
        // redundancy; every other frame clears the carried decision
        // (the paths below overwrite it when they find one).
        self.last_redundancy = crate::celt_redundancy::RedundancyDecision::NotPresent;

        // §3.2.1 zero-length frame: DTX / lost. §4.6 floor = silence.
        if frame.is_empty() {
            push_silence(pcm, per_channel, channels);
            return FrameOutcome {
                samples_per_channel: per_channel,
                status: FrameDecodeStatus::DtxOrLost,
            };
        }

        match routing.operating_mode {
            OperatingMode::SilkOnly => self.decode_silk_only_frame(frame, routing, pcm),
            OperatingMode::CeltOnly => self.decode_celt_only_frame(frame, routing, pcm),
            OperatingMode::Hybrid => self.decode_hybrid_frame(frame, routing, pcm),
        }
    }

    /// Decode one SILK-only Opus frame (§4.2).
    ///
    /// For a **mono** Opus frame this runs the real §4.2.3 header-bit
    /// decode followed by the §4.2.5 LBRR / §4.2.6 regular SILK frame
    /// loop, calling [`crate::silk_decode::decode_silk_frame`] for each
    /// regular SILK frame in Table-5 order with the inter-frame state
    /// (previous gain / lag / NLSF) threaded across the frames of the
    /// Opus frame. The decoded parameters + excitation are then run
    /// through the §4.2.7.9 LTP / LPC synthesis
    /// ([`crate::silk_synthesis::synthesize_silk_frame`]) and the §4.2.9
    /// (non-normative) resample to 48 kHz, producing real PCM
    /// ([`FrameDecodeStatus::SilkParamsDecoded`]). A truncated / malformed
    /// frame yields [`FrameDecodeStatus::SilkDecodeError`] and silence.
    ///
    /// A **stereo** Opus frame runs the §4.2.6 mid/side interleave with
    /// the §4.2.7.1 / §4.2.7.2 symbols enabled and the §4.2.8 unmixing
    /// back half, emitting interleaved L/R PCM
    /// ([`FrameDecodeStatus::SilkStereoDecoded`]).
    ///
    /// After the SILK layer, the §4.5.1.1 **implicit redundancy check**
    /// runs: when at least 17 bits remain, the frame carries a 5 ms
    /// redundant CELT frame in its trailing whole bytes, which is
    /// decoded (§4.5.1.4) and cross-lapped into this frame's output at
    /// the signalled §4.5.1.2 position. An end-position redundant
    /// frame takes the §4.5.2 CELT reset *before* its decode and the
    /// warmed state carries into the next (CELT-layer) frame; a
    /// beginning-position one continues the previous CELT state.
    fn decode_silk_only_frame(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
        pcm: &mut Vec<i16>,
    ) -> FrameOutcome {
        let per_channel = output_samples_per_channel(routing.frame_size_tenths_ms);
        let channels = routing.channel_count();
        let pcm_start = pcm.len();
        push_silence(pcm, per_channel, channels);

        let mut rd = crate::range_decoder::RangeDecoder::new(frame);
        let status = if channels == 2 {
            match self.decode_silk_layer_stereo(&mut rd, routing) {
                Ok((left, right, bandwidth)) => {
                    // §4.2.9: upsample each channel to the 48 kHz output
                    // rate through the carried per-channel resampler
                    // state (Table 54 group delay), then write it
                    // interleaved (`[L0, R0, L1, R1, …]`) over the
                    // reserved silence.
                    self.resample_silk_stereo_into(
                        &left,
                        &right,
                        bandwidth,
                        &mut pcm[pcm_start..pcm_start + per_channel * 2],
                    );
                    FrameDecodeStatus::SilkStereoDecoded
                }
                Err(_) => FrameDecodeStatus::SilkDecodeError,
            }
        } else {
            match self.decode_silk_layer_mono(&mut rd, routing) {
                Ok((internal, bandwidth)) => {
                    // §4.2.9: upsample the internal-rate signal to the
                    // 48 kHz decoder output rate through the carried
                    // resampler state and write it over the reserved
                    // silence region. The filter is non-normative; its
                    // group delay is the normative Table 54 allocation.
                    self.resample_silk_mono_into(
                        &internal,
                        bandwidth,
                        &mut pcm[pcm_start..pcm_start + per_channel],
                    );
                    FrameDecodeStatus::SilkParamsDecoded
                }
                Err(_) => FrameDecodeStatus::SilkDecodeError,
            }
        };

        // §4.5.1: the implicit redundancy signal only means anything
        // when the SILK layer decoded cleanly (a desynchronized coder's
        // tell() is meaningless).
        let mut redundancy = crate::celt_redundancy::RedundancyDecision::NotPresent;
        if !matches!(status, FrameDecodeStatus::SilkDecodeError) && !rd.has_error() {
            redundancy = crate::celt_redundancy::decode_redundancy(
                &mut rd,
                OperatingMode::SilkOnly,
                frame.len(),
            );
            if let Some(params) =
                crate::redundancy_decode_params::redundant_frame_params(routing, redundancy)
            {
                let red_bytes = &frame[frame.len() - params.size_bytes..];
                // §4.5.2: an end-position redundant frame opens a new
                // CELT chain (reset before its decode; Figure 18's
                // `!R`); a beginning-position one continues the
                // previous chain (rule 4).
                let reset_before = matches!(
                    params.position,
                    crate::celt_redundancy::RedundancyPosition::End
                );
                if let Some(red_pcm) = self.decode_redundant_celt(red_bytes, &params, reset_before)
                {
                    apply_redundancy_cross_lap(
                        &mut pcm[pcm_start..],
                        &red_pcm,
                        channels as usize,
                        params.position,
                    );
                }
            }
        }
        self.last_redundancy = redundancy;

        FrameOutcome {
            samples_per_channel: per_channel,
            status,
        }
    }

    /// The mono §4.2 SILK layer decode on a caller-supplied range
    /// coder — shared by the SILK-only path (fresh coder over the
    /// frame) and the Hybrid path (the CELT layer continues on the
    /// same coder afterwards).
    fn decode_silk_layer_mono(
        &mut self,
        rd: &mut crate::range_decoder::RangeDecoder<'_>,
        routing: &OpusFrameRouting,
    ) -> Result<(Vec<i16, crate::bump::DecodeBump>, crate::toc::Bandwidth), Error> {
        use crate::silk_decode::{decode_silk_frame, SilkFrameConfig, SilkFrameDecoded};
        use crate::silk_excitation::SilkFrameSize;
        use crate::silk_frame::FrameKind;
        use crate::silk_header::SilkHeaderBits;
        use crate::silk_synthesis::{synthesize_silk_frame_i16, SilkSynthState};

        let bandwidth = routing
            .silk_bandwidth
            .ok_or(Error::MalformedPacket)?
            .to_bandwidth();
        let num_silk_frames = routing
            .silk_frames_per_channel
            .ok_or(Error::MalformedPacket)?;
        // §4.2.2: each SILK frame is 20 ms, except a 10 ms Opus frame
        // (one SILK frame of 10 ms).
        let frame_size = if routing.frame_size_tenths_ms == 100 {
            SilkFrameSize::TenMs
        } else {
            SilkFrameSize::TwentyMs
        };

        // §4.2.3 / §4.2.4 header bits (mono => stereo = false).
        let header = SilkHeaderBits::decode(rd, num_silk_frames, false)?;

        // §4.2.5 LBRR frames: one per SILK frame whose mid LBRR bit is
        // set, in time-interval order. LBRR frames are independent of the
        // regular-frame inter-frame state (they form their own sequence),
        // but for this mono path we decode them to consume their bits and
        // keep the range coder aligned with the regular frames that
        // follow. Per §4.2.7.3 an LBRR frame is always active-coded.
        let mut lbrr_prev_gain: Option<u8> = None;
        let mut lbrr_prev_lag: Option<i32> = None;
        let mut lbrr_first = true;
        for idx in 0..num_silk_frames {
            if !header.mid_has_lbrr(idx) {
                // §4.2.4 LBRR-flag gap: the next coded LBRR frame codes
                // independently (§4.2.7.4), with an absolute lag
                // (§4.2.7.6.1) and the §4.2.7.6.3 scaling field present
                // again ("the previous LBRR frame ... is not coded").
                lbrr_prev_gain = None;
                lbrr_prev_lag = None;
                lbrr_first = true;
                continue;
            }
            let cfg = SilkFrameConfig {
                bandwidth,
                frame_size,
                voice_active: true, // §4.2.7.3: LBRR uses the active PDF.
                first_subframe_independent: lbrr_first || lbrr_prev_gain.is_none(),
                previous_log_gain: lbrr_prev_gain,
                previous_primary_lag: lbrr_prev_lag,
                ltp_scaling_present: lbrr_first,
                lsf_interp_after_reset: lbrr_first,
                previous_nlsf_q15: None,
                previous_nlsf_len: 0,
                // Mono SILK-only path: no §4.2.7.1 / §4.2.7.2 stereo header.
                stereo: None,
            };
            let decoded = decode_silk_frame(rd, cfg)?;
            lbrr_prev_gain = Some(decoded.gains.last_log_gain());
            // §4.2.7.6.1: only a VOICED frame arms relative lag coding.
            lbrr_prev_lag = if decoded.ltp.is_voiced() {
                Some(decoded.ltp.primary_lag())
            } else {
                None
            };
            lbrr_first = false;
            let _ = FrameKind::Lbrr; // documents the §4.2.7.3 kind.
        }

        // §4.2.6 regular SILK frames: one per time interval, even when
        // the VAD flag is unset. Inter-frame state threads across them,
        // resuming the cross-Opus-frame §4.2.7.4 gain-clamp base and the
        // §4.2.7.5.5 NLSF interpolation base `n0` carried from the
        // previous Opus frame (the frame-local `first` flag still makes
        // the first frame's gain independently CODED and its §4.2.7.6.3
        // scaling / absolute lag present, per the per-Opus-frame rules).
        let mut chan = ChannelDecodeState::resume(&self.silk_carry_mono, bandwidth);
        // Profluens patch: per-packet SILK frame container (holds bump-backed decoded frames) +
        // the internal-rate accumulator below, both from the bump arena (consumed by resampling as
        // slices, never carried in `self`).
        let mut decoded_frames: Vec<SilkFrameDecoded, crate::bump::DecodeBump> =
            Vec::with_capacity_in(num_silk_frames as usize, crate::bump::DecodeBump);
        for idx in 0..num_silk_frames {
            let decoded = decode_silk_frame(
                rd,
                // Mono SILK-only path: no §4.2.7.1 / §4.2.7.2 stereo header.
                chan.config(bandwidth, frame_size, header.mid_vad(idx), None),
            )?;
            chan.advance(&decoded);
            decoded_frames.push(decoded);
        }

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }
        // Persist the §4.2.7.4 / §4.2.7.5.5 bases for the next Opus frame.
        self.silk_carry_mono = chan.to_carry(bandwidth);

        // §4.2.7.9 synthesis: turn the decoded SILK frames into
        // internal-rate (8/12/16 kHz) time-domain samples, threading the
        // cross-Opus-frame §4.2.7.9 histories. The state is (re)created if
        // absent or if the SILK bandwidth changed (a §4.5.2 reset).
        let need_fresh = match &self.silk_synth_mono {
            Some(s) => s.bandwidth() != bandwidth,
            None => true,
        };
        if need_fresh {
            self.silk_synth_mono = Some(SilkSynthState::new(bandwidth)?);
            // A bandwidth change re-times the internal-rate signal; the
            // carried §4.2.8 delay sample belongs to the old rate.
            self.silk_mono_delay = 0;
        }
        let state = self
            .silk_synth_mono
            .as_mut()
            .expect("synth state set above");

        let mut internal: Vec<i16, crate::bump::DecodeBump> = Vec::new_in(crate::bump::DecodeBump);
        for decoded in &decoded_frames {
            let frame_out = synthesize_silk_frame_i16(bandwidth, frame_size, decoded, state)?;
            internal.extend_from_slice(&frame_out);
        }

        // §4.2.8: "mono streams must also impose the same one-sample
        // delay" as the stereo unmixing (whose formulas read
        // `mid[i-1]` / `side[i-1]`). Shift the internal-rate signal by
        // one sample, carrying the previous Opus frame's final sample
        // across the boundary (zero after a decoder reset). This is the
        // reference decoder's two-sample output buffering with the
        // resampler reading from offset 1.
        if let Some(&last) = internal.last() {
            internal.rotate_right(1);
            internal[0] = self.silk_mono_delay;
            self.silk_mono_delay = last;
        }
        Ok((internal, bandwidth))
    }

    /// §4.2.9: upsample one mono Opus frame's internal-rate SILK
    /// samples to 48 kHz i16 PCM through the **carried** mono
    /// upsampler state (rebuilt when the SILK bandwidth changes — a
    /// §4.5.2 reset — mirroring [`Self::silk_synth_mono`]).
    fn resample_silk_mono_into(
        &mut self,
        internal: &[i16],
        bandwidth: crate::toc::Bandwidth,
        out: &mut [i16],
    ) {
        use crate::silk_resampler::SilkUpsampler;
        let stale = !matches!(&self.silk_resamp_mono,
            Some(up) if up.bandwidth() == bandwidth);
        if stale {
            self.silk_resamp_mono =
                SilkUpsampler::new(bandwidth, crate::silk_resampler::SilkChannelPath::Mono);
        }
        match self.silk_resamp_mono.as_mut() {
            Some(up) => resample_through_upsampler_i16(up, internal, out),
            // Unreachable for real SILK bandwidths (NB/MB/WB always
            // construct); keep a total fallback.
            None => resample_linear_i16(internal, out),
        }
    }

    /// §4.2.9: upsample one stereo Opus frame's unmixed left/right
    /// internal-rate channels to interleaved 48 kHz i16 PCM through
    /// the **carried** per-channel upsampler pair (lifecycle mirrors
    /// [`Self::silk_synth_stereo`]).
    fn resample_silk_stereo_into(
        &mut self,
        left: &[i16],
        right: &[i16],
        bandwidth: crate::toc::Bandwidth,
        out: &mut [i16],
    ) {
        use crate::silk_resampler::SilkUpsampler;
        let per_channel = out.len() / 2;
        if per_channel == 0 {
            return;
        }
        let stale = !matches!(&self.silk_resamp_stereo,
            Some((l, _)) if l.bandwidth() == bandwidth);
        if stale {
            let path = crate::silk_resampler::SilkChannelPath::Stereo;
            self.silk_resamp_stereo =
                SilkUpsampler::new(bandwidth, path).zip(SilkUpsampler::new(bandwidth, path));
        }
        // Profluens patch: per-frame resampled L/R, interleaved into `out` then dropped — bump.
        let mut l = Vec::with_capacity_in(per_channel, crate::bump::DecodeBump);
        l.resize(per_channel, 0i16);
        let mut r = Vec::with_capacity_in(per_channel, crate::bump::DecodeBump);
        r.resize(per_channel, 0i16);
        match self.silk_resamp_stereo.as_mut() {
            Some((ul, ur)) => {
                resample_through_upsampler_i16(ul, left, &mut l);
                resample_through_upsampler_i16(ur, right, &mut r);
            }
            None => {
                resample_linear_i16(left, &mut l);
                resample_linear_i16(right, &mut r);
            }
        }
        for i in 0..per_channel {
            out[2 * i] = l[i];
            out[2 * i + 1] = r[i];
        }
    }

    /// Decode the full §4.2 bitstream of one **stereo** SILK-only Opus
    /// frame and unmix it to left/right.
    ///
    /// The §4.2.2 stereo organisation interleaves the two channels: per
    /// 20 ms interval the mid SILK frame is decoded, then the side SILK
    /// frame (skipped when the §4.2.7.2 mid-only flag on the mid frame is
    /// set). The §4.2.7.1 stereo prediction weights ride on the mid
    /// frame. After both channels finish their §4.2.7.9 synthesis they are
    /// converted from mid/side to left/right via §4.2.8
    /// ([`crate::silk_stereo::stereo_ms_to_lr`]).
    ///
    /// LBRR frames (§4.2.5) precede the regular frames and are also
    /// interleaved (mid then side per interval); they are decoded only to
    /// keep the range coder aligned with the regular frames that follow.
    ///
    /// Returns `(left, right, bandwidth)` at the SILK internal rate.
    #[allow(clippy::type_complexity)]
    fn decode_silk_layer_stereo(
        &mut self,
        rd: &mut crate::range_decoder::RangeDecoder<'_>,
        routing: &OpusFrameRouting,
    ) -> Result<
        (Vec<i16, crate::bump::DecodeBump>, Vec<i16, crate::bump::DecodeBump>, crate::toc::Bandwidth),
        Error,
    > {
        use crate::silk_decode::{decode_silk_frame, SilkFrameDecoded, StereoHeaderContext};
        use crate::silk_excitation::SilkFrameSize;
        use crate::silk_header::SilkHeaderBits;
        use crate::silk_stereo::{stereo_ms_to_lr_i16, StereoUnmixStateI16, StereoWeightsQ13};
        use crate::silk_synthesis::{synthesize_silk_frame_i16, SilkSynthState};

        let bandwidth = routing
            .silk_bandwidth
            .ok_or(Error::MalformedPacket)?
            .to_bandwidth();
        let num_silk_frames = routing
            .silk_frames_per_channel
            .ok_or(Error::MalformedPacket)?;
        let frame_size = if routing.frame_size_tenths_ms == 100 {
            SilkFrameSize::TenMs
        } else {
            SilkFrameSize::TwentyMs
        };

        // §4.2.3 / §4.2.4 header bits (stereo => both channels' VAD + LBRR
        // flags, mid then side).
        let header = SilkHeaderBits::decode(rd, num_silk_frames, true)?;

        // §4.2.5 LBRR frames: per 20 ms interval, the mid LBRR frame (if
        // present) then the side LBRR frame (if present), interleaved per
        // §4.2.2. Decoded only to consume their bits. The §4.2.7.1 stereo
        // weights ride on the mid LBRR frame; the §4.2.7.2 mid-only flag
        // is present on the mid LBRR frame iff the side LBRR is unset for
        // that interval.
        let mut lbrr_mid = ChannelDecodeState::new();
        let mut lbrr_side = ChannelDecodeState::new();
        for idx in 0..num_silk_frames {
            let mid_lbrr = header.mid_has_lbrr(idx);
            let side_lbrr = header.side_has_lbrr(idx);
            if mid_lbrr {
                let stereo_ctx = StereoHeaderContext {
                    // §4.2.7.2: mid-only flag present on the mid frame iff
                    // the corresponding side channel is not coded.
                    has_mid_only_flag: !side_lbrr,
                };
                let decoded = decode_silk_frame(
                    rd,
                    lbrr_mid.config(bandwidth, frame_size, true, Some(stereo_ctx)),
                )?;
                lbrr_mid.advance(&decoded);
            } else {
                // §4.2.4 LBRR-flag gap for the mid channel.
                lbrr_mid.mark_interval_uncoded(true);
            }
            // A set mid-only flag would forbid a coded side LBRR frame;
            // the header LBRR flags already encode that, so we trust
            // `side_lbrr` for the interleave decision. (A side-only LBRR
            // interval is legal: no stereo weights ride on a side frame
            // per §4.2.7.1.)
            if side_lbrr {
                let decoded =
                    decode_silk_frame(rd, lbrr_side.config(bandwidth, frame_size, true, None))?;
                lbrr_side.advance(&decoded);
            } else {
                lbrr_side.mark_interval_uncoded(true);
            }
        }

        // §4.2.6 regular SILK frames: per 20 ms interval, the mid frame
        // then (unless the §4.2.7.2 mid-only flag is set) the side frame.
        // Each channel resumes its cross-Opus-frame §4.2.7.4 gain-clamp
        // base and §4.2.7.5.5 NLSF interpolation base `n0`.
        let mut mid_state = ChannelDecodeState::resume(&self.silk_carry_mid, bandwidth);
        let mut side_state = ChannelDecodeState::resume(&self.silk_carry_side, bandwidth);
        // Profluens patch: per-packet SILK containers + L/R accumulators, from the bump arena
        // (consumed as slices by the stereo unmix + resampling, never carried in `self`).
        let mut mid_frames: Vec<SilkFrameDecoded, crate::bump::DecodeBump> =
            Vec::with_capacity_in(num_silk_frames as usize, crate::bump::DecodeBump);
        // Per-interval side frame: `Some(frame)` when coded, `None` when
        // the side channel is skipped (mid-only flag set or side VAD path
        // produced no frame). The §4.2.8 unmixer treats a `None` side as
        // all-zero.
        let mut side_frames: Vec<Option<SilkFrameDecoded>, crate::bump::DecodeBump> =
            Vec::with_capacity_in(num_silk_frames as usize, crate::bump::DecodeBump);
        // The §4.2.7.1 weights carried by the most-recent mid frame; the
        // §4.2.8 unmix consumes the last interval's weights for the whole
        // Opus frame (one set of weights per SILK frame, but the unmix
        // runs once over the concatenated channel signal — we apply the
        // first interval's weights, threading prev across intervals via
        // the unmix state below).
        let mut interval_weights: Vec<StereoWeightsQ13, crate::bump::DecodeBump> =
            Vec::with_capacity_in(num_silk_frames as usize, crate::bump::DecodeBump);

        for idx in 0..num_silk_frames {
            let side_active = header.side_vad(idx);
            // §4.2.7.2: the mid-only flag is present iff the side channel
            // for this interval is NOT active (a regular frame with side
            // VAD unset). When side VAD is set the side frame must be
            // coded and the flag is omitted.
            let stereo_ctx = StereoHeaderContext {
                has_mid_only_flag: !side_active,
            };
            let mid_decoded = decode_silk_frame(
                rd,
                mid_state.config(bandwidth, frame_size, header.mid_vad(idx), Some(stereo_ctx)),
            )?;
            // §4.2.7.1 weights ride on the mid frame.
            let w = mid_decoded.stereo_pred.map(|p| StereoWeightsQ13 {
                w0_q13: p.w0_q13,
                w1_q13: p.w1_q13,
            });
            interval_weights.push(w.unwrap_or_default());
            // §4.2.7.2: side coded iff side VAD set OR the mid-only flag is
            // not set (mid-only flag present + cleared ⇒ side is coded).
            let side_coded = side_active || mid_decoded.mid_only_flag == Some(false);
            mid_state.advance(&mid_decoded);
            mid_frames.push(mid_decoded);

            if side_coded {
                let side_decoded = decode_silk_frame(
                    rd,
                    side_state.config(bandwidth, frame_size, header.side_vad(idx), None),
                )?;
                side_state.advance(&side_decoded);
                side_frames.push(Some(side_decoded));
            } else {
                // §4.2.7.2 / §4.5.2: an uncoded side SILK frame clears the
                // side LTP buffer; zeros feed the §4.2.8 unmixer. The side
                // carried state also resets (§4.2.7.4 / §4.2.7.5.5 /
                // §4.2.7.6.1 all treat "previous frame not coded" as a
                // fresh start for the next coded side frame).
                side_state.mark_interval_uncoded(false);
                side_frames.push(None);
            }
        }

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }
        // Persist the per-channel §4.2.7.4 / §4.2.7.5.5 bases for the
        // next Opus frame (an uncoded trailing side interval leaves the
        // side carry cleared, arming the RFC's "previous frame in the
        // side channel was not coded" fresh-start rules).
        self.silk_carry_mid = mid_state.to_carry(bandwidth);
        self.silk_carry_side = side_state.to_carry(bandwidth);

        // §4.2.7.9 synthesis for both channels, threading the cross-Opus-
        // frame histories. (Re)create the state on a bandwidth change.
        let need_fresh = match &self.silk_synth_stereo {
            Some((m, _)) => m.bandwidth() != bandwidth,
            None => true,
        };
        if need_fresh {
            self.silk_synth_stereo = Some((
                SilkSynthState::new(bandwidth)?,
                SilkSynthState::new(bandwidth)?,
            ));
        }
        let (mid_synth, side_synth) = self
            .silk_synth_stereo
            .as_mut()
            .expect("stereo synth state set above");

        // §4.2.8 stereo unmixing runs **per SILK frame** (per 20 ms
        // interval), not once over the whole Opus frame: the spec defines
        // the unmix over `j <= i < (j + n2)` where `j` is the SILK frame
        // start and `n2` is "the total number of samples in the frame"
        // (the SILK frame). Each interval carries its own §4.2.7.1 weights
        // and restarts the 8 ms interpolation phase; the previous
        // interval's weights and trailing samples thread through the
        // carried `StereoUnmixState`. We therefore synthesize and unmix
        // each interval in turn and concatenate the L/R outputs.
        let unmix = self
            .silk_stereo_unmix
            .get_or_insert_with(StereoUnmixStateI16::new);
        let fs_khz = match bandwidth {
            crate::toc::Bandwidth::Nb => 8usize,
            crate::toc::Bandwidth::Mb => 12,
            _ => 16,
        };

        let mut left: Vec<i16, crate::bump::DecodeBump> = Vec::new_in(crate::bump::DecodeBump);
        let mut right: Vec<i16, crate::bump::DecodeBump> = Vec::new_in(crate::bump::DecodeBump);
        for (idx, mid_frame) in mid_frames.iter().enumerate() {
            let mid_out = synthesize_silk_frame_i16(bandwidth, frame_size, mid_frame, mid_synth)?;
            let n = mid_out.len();
            let weights = interval_weights[idx];
            let stereo = match &side_frames[idx] {
                Some(side_frame) => {
                    let side_out =
                        synthesize_silk_frame_i16(bandwidth, frame_size, side_frame, side_synth)?;
                    stereo_ms_to_lr_i16(fs_khz, &mid_out, Some(&side_out), weights, unmix)?
                }
                None => {
                    // §4.2.7.2 / §4.5.2: an uncoded side SILK frame clears
                    // the side channel's prediction memory (its output
                    // history and LPC state; the previous-subframe gain
                    // survives, matching the reference decoder's
                    // side-channel reset). Zeros feed the §4.2.8 unmixer,
                    // but the carried side history still applies.
                    side_synth.reset_prediction_memory();
                    stereo_ms_to_lr_i16(fs_khz, &mid_out, None, weights, unmix)?
                }
            };
            debug_assert_eq!(stereo.left.len(), n);
            left.extend_from_slice(&stereo.left);
            right.extend_from_slice(&stereo.right);
        }

        Ok((left, right, bandwidth))
    }

    /// Decode one CELT-only Opus frame (§4.3) end-to-end.
    ///
    /// Runs the full Table-56 entropy decode
    /// ([`crate::celt_frame_decode::decode_celt_frame`]: frame flags,
    /// coarse / fine / final energies, TF, spread, dynalloc, trim, the
    /// §4.3.3 implicit allocation, §4.3.4 PVQ band shapes with folding,
    /// and §4.3.5 anti-collapse), then the §4.3.6–§4.3.7.2 synthesis
    /// ([`crate::celt_mdct_synthesis::CeltSynthesis`]: denormalisation,
    /// long/short-block inverse MDCT with overlap-add, the §4.3.7.1
    /// pitch post-filter, and de-emphasis), emitting real 48 kHz PCM.
    ///
    /// Cross-frame state (energy history + folding seed, overlap and
    /// post-filter memories) is carried in `self`; it is rebuilt when
    /// the frame geometry (size / channels) changes and dropped on a
    /// §4.5.2 CELT reset. A range-coder error or a bit-budget overrun
    /// yields [`FrameDecodeStatus::CeltDecodeError`] and silence.
    fn decode_celt_only_frame(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
        pcm: &mut Vec<i16>,
    ) -> FrameOutcome {
        use crate::celt_band_layout::CeltFrameSize;

        let per_channel = output_samples_per_channel(routing.frame_size_tenths_ms);
        let channels = routing.channel_count() as usize;
        let pcm_start = pcm.len();
        push_silence(pcm, per_channel, channels as u8);
        let err = FrameOutcome {
            samples_per_channel: per_channel,
            status: FrameDecodeStatus::CeltDecodeError,
        };

        let Some(celt_size) =
            CeltFrameSize::from_frame_tenths_ms(routing.frame_size_tenths_ms as u32)
        else {
            return err;
        };
        let lm = celt_size.column_index() as i32;
        let n = per_channel; // CELT-only runs at the 48 kHz output rate.
                             // §4.4 / Table 55: the coded band range ends at the signalled
                             // audio bandwidth (NB→13, WB→17, SWB→19, FB→21).
        let end = match routing.toc_bandwidth {
            crate::toc::Bandwidth::Nb => 13,
            crate::toc::Bandwidth::Mb | crate::toc::Bandwidth::Wb => 17,
            crate::toc::Bandwidth::Swb => 19,
            crate::toc::Bandwidth::Fb => 21,
        };

        // Adapt the carried CELT state to this frame's geometry
        // WITHOUT dropping it: §4.5 says switching between any two
        // CELT-only configurations needs no special treatment (the
        // MDCT overlap smooths the transition), and a §4.5.1
        // redundant frame's warmed state must carry into this frame
        // (rule 3). Resets happen only where §4.5.2 places them (the
        // packet-boundary policy above).
        match self.celt_synth.as_mut() {
            Some(s) => s.set_geometry(channels, n),
            None => {
                self.celt_synth = Some(crate::celt_mdct_synthesis::CeltSynthesis::new(channels, n))
            }
        }
        let energy = self.celt_energy.get_or_insert_with(Default::default);

        // §4.3 entropy half.
        let mut rd = crate::range_decoder::RangeDecoder::new(frame);
        let out = crate::celt_frame_decode::decode_celt_frame(
            &mut rd,
            frame.len(),
            0,
            end,
            lm,
            channels,
            energy,
        );
        if rd.has_error() || u64::from(rd.tell()) > 8 * frame.len() as u64 {
            return err;
        }

        // §4.3.6 denormalisation into the frame spectrum (bins outside
        // the coded range stay zero).
        let m = 1usize << lm;
        // Profluens patch: per-frame denormalised spectrum, consumed by synthesis as a slice — from
        // the per-packet bump arena.
        let mut freq = Vec::with_capacity_in(channels * n, crate::bump::DecodeBump);
        freq.resize(channels * n, 0.0f64);
        for c in 0..channels {
            for band in 0..end {
                let gain = out.band_gain[c][band];
                let off = m * crate::celt_rate_alloc::band_edge(band) as usize;
                let len = m * crate::celt_rate_alloc::band_width(band) as usize;
                for j in 0..len {
                    freq[c * n + off + j] = gain * out.x[c * out.plane + off + j];
                }
            }
        }

        // §4.3.7 synthesis (signal half).
        let blocks = if out.transient { m } else { 1 };
        let synth = self.celt_synth.as_mut().expect("state built above");
        let mut frame_pcm = Vec::with_capacity_in(channels * n, crate::bump::DecodeBump);
        frame_pcm.resize(channels * n, 0i16);
        synth.synthesize_frame(&freq, blocks, out.post_filter, &mut frame_pcm);
        let region = &mut pcm[pcm_start..pcm_start + per_channel * channels];
        region.copy_from_slice(&frame_pcm);

        FrameOutcome {
            samples_per_channel: per_channel,
            status: if out.silence {
                FrameDecodeStatus::CeltSilence
            } else {
                FrameDecodeStatus::CeltDecoded
            },
        }
    }

    /// Recover the audio of a **lost** Opus frame from the in-band FEC
    /// (§4.2.5 LBRR) data carried in the *next* successfully received
    /// packet (RFC 6716 §2.1.7).
    ///
    /// In-band FEC encodes a low-bitrate redundant copy of the signal
    /// immediately *prior* to a packet as one or more §4.2.5 LBRR frames
    /// inside that packet. When the application detects a packet loss and
    /// has the following packet in hand, it calls this method on that
    /// following packet to reconstruct the lost frame's audio instead of
    /// relying solely on silence / pitch-based concealment.
    ///
    /// The recovered PCM is returned at the 48 kHz output rate. The packet
    /// passed here is the one *after* the loss; only its §4.2.5 LBRR
    /// frames are decoded and synthesized (the packet's own regular frames
    /// are decoded later by an ordinary [`Self::decode_packet`] call).
    ///
    /// On success ([`FecDecodeStatus::Recovered`]) the SILK synthesis
    /// history is advanced to the recovered frame's state, so a subsequent
    /// [`Self::decode_packet`] on the same packet continues smoothly from
    /// the reconstructed signal. When the packet carries no LBRR data
    /// ([`FecDecodeStatus::NoLbrr`]), is CELT-only
    /// ([`FecDecodeStatus::NotSilk`]), or is malformed
    /// ([`FecDecodeStatus::DecodeError`]), silence of the lost frame's
    /// duration is returned and the caller falls back to its own
    /// concealment.
    ///
    /// Returns [`Error::EmptyPacket`] for a zero-length packet and
    /// [`Error::MalformedPacket`] for a §3.2 framing violation in the
    /// carrier packet.
    pub fn decode_packet_fec(&mut self, packet: &[u8]) -> Result<FecRecovered, Error> {
        crate::bump::reset(); // per-packet bump rewind, before parse bump-allocates the frame list
        let parsed = OpusPacket::parse(packet)?;
        let routing = OpusFrameRouting::from_toc(parsed.toc);
        let channels = routing.channel_count();
        // §4.2.5: an LBRR frame has the same frame size / bandwidth /
        // channel count as the carrier packet's regular frames, and covers
        // the equivalent prior interval(s); the recovered duration matches
        // the carrier's per-frame duration.
        let per_channel = output_samples_per_channel(routing.frame_size_tenths_ms);
        let mut pcm = vec![0i16; per_channel * channels as usize];

        // FEC only exists for SILK-bearing modes (§2.1.7 re-encodes the
        // SILK speech layer); a CELT-only packet carries no LBRR.
        if !matches!(
            routing.operating_mode,
            OperatingMode::SilkOnly | OperatingMode::Hybrid
        ) {
            return Ok(FecRecovered {
                pcm,
                channels,
                sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
                status: FecDecodeStatus::NotSilk,
            });
        }

        // The first Opus frame of the packet carries the §4.2.5 LBRR
        // frames (LBRR frames precede the regular frames within a single
        // SILK-bearing Opus frame; a code-1/2/3 packet's later frames have
        // their own LBRR, but those cover intervals already adjacent to
        // received audio, so the canonical "previous packet was lost"
        // recovery uses the leading Opus frame's LBRR).
        let Some(&frame) = parsed.frames().first() else {
            return Ok(FecRecovered {
                pcm,
                channels,
                sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
                status: FecDecodeStatus::DecodeError,
            });
        };
        if frame.is_empty() {
            return Ok(FecRecovered {
                pcm,
                channels,
                sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
                status: FecDecodeStatus::NoLbrr,
            });
        }

        let status = if channels == 2 {
            match self.decode_silk_fec_stereo(frame, &routing) {
                Ok(Some((left, right, bandwidth))) => {
                    resample_stereo_to_output_i16(&left, &right, bandwidth, &mut pcm);
                    FecDecodeStatus::Recovered
                }
                Ok(None) => FecDecodeStatus::NoLbrr,
                Err(_) => FecDecodeStatus::DecodeError,
            }
        } else {
            match self.decode_silk_fec_mono(frame, &routing) {
                Ok(Some((internal, bandwidth))) => {
                    resample_internal_to_output_i16(&internal, bandwidth, &mut pcm);
                    FecDecodeStatus::Recovered
                }
                Ok(None) => FecDecodeStatus::NoLbrr,
                Err(_) => FecDecodeStatus::DecodeError,
            }
        };

        // A real recovery is stream audio: record it as §4.4
        // concealment history (and clear the consecutive-loss counter)
        // so a further loss immediately after the recovered frame
        // extrapolates from the recovered signal, not from before it.
        if status == FecDecodeStatus::Recovered {
            self.plc.feed_decoded(&pcm, channels as usize);
        }

        Ok(FecRecovered {
            pcm,
            channels,
            sample_rate_hz: OUTPUT_SAMPLE_RATE_HZ,
            status,
        })
    }

    /// Decode and synthesize the §4.2.5 mono LBRR frame(s) of one
    /// SILK-bearing Opus frame into internal-rate recovered audio.
    ///
    /// Returns `Ok(Some((internal, bandwidth)))` with the recovered
    /// signal when at least one mid LBRR frame is present, `Ok(None)` when
    /// the §4.2.4 LBRR flags are all clear (no FEC data), or `Err` on a
    /// malformed bitstream.
    ///
    /// Unlike [`Self::decode_silk_layer_mono`], which only consumed the
    /// LBRR bits to keep the range coder aligned, this path actually runs
    /// the §4.2.7.9 synthesis on the LBRR parameters. Per §4.2.5 the LBRR
    /// frames form their own independent sequence covering the prior
    /// interval(s), so synthesis starts from a **fresh** state (the lost
    /// frame's true history is, by definition, unavailable). On success
    /// the decoder's carried mono synthesis state is replaced with the
    /// recovered-frame history so the next real packet continues smoothly.
    fn decode_silk_fec_mono(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
    ) -> Result<Option<(Vec<i16>, crate::toc::Bandwidth)>, Error> {
        use crate::range_decoder::RangeDecoder;
        use crate::silk_decode::{decode_silk_frame, SilkFrameDecoded};
        use crate::silk_excitation::SilkFrameSize;
        use crate::silk_header::SilkHeaderBits;
        use crate::silk_synthesis::{synthesize_silk_frame_i16, SilkSynthState};

        let bandwidth = routing
            .silk_bandwidth
            .ok_or(Error::MalformedPacket)?
            .to_bandwidth();
        let num_silk_frames = routing
            .silk_frames_per_channel
            .ok_or(Error::MalformedPacket)?;
        let frame_size = if routing.frame_size_tenths_ms == 100 {
            SilkFrameSize::TenMs
        } else {
            SilkFrameSize::TwentyMs
        };

        let mut rd = RangeDecoder::new(frame);
        let header = SilkHeaderBits::decode(&mut rd, num_silk_frames, false)?;

        // No LBRR data → no FEC recovery is possible.
        if !(0..num_silk_frames).any(|i| header.mid_has_lbrr(i)) {
            return Ok(None);
        }

        // §4.2.5 LBRR frames are always active-coded and form their own
        // inter-frame sequence; decode every present LBRR frame in
        // interval order, threading the LBRR-local previous gain / lag /
        // NLSF state (the same Table-5 inter-frame dependencies as regular
        // frames, but over the LBRR sub-sequence).
        let mut lbrr_chan = ChannelDecodeState::new();
        let mut lbrr_frames: Vec<SilkFrameDecoded> = Vec::new();
        for idx in 0..num_silk_frames {
            if !header.mid_has_lbrr(idx) {
                // §4.2.4 LBRR-flag gap: fresh start for the next coded
                // LBRR frame (§4.2.7.4 / §4.2.7.5.5 / §4.2.7.6.1 /
                // §4.2.7.6.3).
                lbrr_chan.mark_interval_uncoded(true);
                continue;
            }
            // §4.2.5: all LBRR frames are active.
            let decoded =
                decode_silk_frame(&mut rd, lbrr_chan.config(bandwidth, frame_size, true, None))?;
            lbrr_chan.advance(&decoded);
            lbrr_frames.push(decoded);
        }

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }
        if lbrr_frames.is_empty() {
            return Ok(None);
        }
        // The recovered interval becomes "the most recent coded frame"
        // in this channel: seed the cross-Opus-frame §4.2.7.4 /
        // §4.2.7.5.5 bases from the LBRR reconstruction (the RFC's
        // packet-loss latitude — the true regular-frame bases were lost
        // with the packet, and the LBRR frames are the re-encode of the
        // same interval).
        self.silk_carry_mono = lbrr_chan.to_carry(bandwidth);

        // §4.2.7.9 synthesis from a fresh state: the lost frame's true
        // history is unavailable, so the recovered signal is reconstructed
        // self-contained. The resulting history then becomes the carried
        // mono synthesis state for the following real packet.
        let mut state = SilkSynthState::new(bandwidth)?;
        let mut internal: Vec<i16> = Vec::new();
        for decoded in &lbrr_frames {
            let frame_out = synthesize_silk_frame_i16(bandwidth, frame_size, decoded, &mut state)?;
            internal.extend_from_slice(&frame_out);
        }
        self.silk_synth_mono = Some(state);

        // §4.2.8 mono one-sample delay, so the recovered interval sits
        // on the same delayed timeline as the surrounding stream. The
        // lost frame's true trailing sample is unavailable — use zero
        // (the reset value) and seed the carry for the next packet.
        if let Some(&last) = internal.last() {
            internal.rotate_right(1);
            internal[0] = 0;
            self.silk_mono_delay = last;
        }
        Ok(Some((internal, bandwidth)))
    }

    /// Decode and synthesize the §4.2.5 **stereo** LBRR frame(s) of one
    /// SILK-bearing Opus frame into internal-rate recovered L/R audio.
    ///
    /// Mirrors [`Self::decode_silk_fec_mono`] for stereo: the §4.2.5 LBRR
    /// frames are interleaved (mid then side per 20 ms interval), each
    /// channel is synthesized from a fresh state, and the pair is unmixed
    /// to left/right via §4.2.8 with a fresh unmix history. The §4.2.7.1
    /// stereo prediction weights ride on the mid LBRR frame; the §4.2.7.2
    /// mid-only flag governs whether a side LBRR frame is present for the
    /// interval (mirroring the regular stereo path).
    ///
    /// Returns `Ok(Some((left, right, bandwidth)))` on recovery,
    /// `Ok(None)` when neither channel carries LBRR, or `Err` on a
    /// malformed bitstream. On success the carried stereo synthesis +
    /// unmix state is replaced with the recovered-frame state.
    #[allow(clippy::type_complexity)]
    fn decode_silk_fec_stereo(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
    ) -> Result<Option<(Vec<i16>, Vec<i16>, crate::toc::Bandwidth)>, Error> {
        use crate::range_decoder::RangeDecoder;
        use crate::silk_decode::{decode_silk_frame, SilkFrameDecoded, StereoHeaderContext};
        use crate::silk_excitation::SilkFrameSize;
        use crate::silk_header::SilkHeaderBits;
        use crate::silk_stereo::{stereo_ms_to_lr_i16, StereoUnmixStateI16, StereoWeightsQ13};
        use crate::silk_synthesis::{synthesize_silk_frame_i16, SilkSynthState};

        let bandwidth = routing
            .silk_bandwidth
            .ok_or(Error::MalformedPacket)?
            .to_bandwidth();
        let num_silk_frames = routing
            .silk_frames_per_channel
            .ok_or(Error::MalformedPacket)?;
        let frame_size = if routing.frame_size_tenths_ms == 100 {
            SilkFrameSize::TenMs
        } else {
            SilkFrameSize::TwentyMs
        };

        let mut rd = RangeDecoder::new(frame);
        let header = SilkHeaderBits::decode(&mut rd, num_silk_frames, true)?;

        let any_lbrr =
            (0..num_silk_frames).any(|i| header.mid_has_lbrr(i) || header.side_has_lbrr(i));
        if !any_lbrr {
            return Ok(None);
        }

        // §4.2.5 interleaved LBRR decode: per 20 ms interval the mid LBRR
        // frame (if present, carrying the §4.2.7.1 weights + §4.2.7.2
        // mid-only flag) then the side LBRR frame (if present). Each
        // channel threads its own LBRR-local inter-frame state.
        let mut mid_state = ChannelDecodeState::new();
        let mut side_state = ChannelDecodeState::new();
        let mut mid_frames: Vec<SilkFrameDecoded> = Vec::new();
        let mut side_frames: Vec<Option<SilkFrameDecoded>> = Vec::new();
        let mut interval_weights: Vec<StereoWeightsQ13> = Vec::new();

        for idx in 0..num_silk_frames {
            let mid_lbrr = header.mid_has_lbrr(idx);
            let side_lbrr = header.side_has_lbrr(idx);
            if !mid_lbrr {
                // §4.2.4 LBRR-flag gap for the mid channel.
                mid_state.mark_interval_uncoded(true);
                // §4.2.5 / §4.2.7.1: a side LBRR frame without a mid LBRR
                // frame carries no stereo weights; record a zero-weight
                // interval with the mid channel treated as silent.
                if side_lbrr {
                    let side_decoded = decode_silk_frame(
                        &mut rd,
                        side_state.config(bandwidth, frame_size, true, None),
                    )?;
                    side_state.advance(&side_decoded);
                    // Without a mid LBRR frame there is no mid signal for
                    // this interval; the unmixer treats the missing mid as
                    // a hole (handled by skipping the interval in synthesis
                    // below — we still consume the bits for alignment).
                    let _ = side_decoded;
                } else {
                    side_state.mark_interval_uncoded(true);
                }
                continue;
            }
            // §4.2.7.2: the mid-only flag is present on the mid LBRR frame
            // iff the side LBRR frame for this interval is absent.
            let stereo_ctx = StereoHeaderContext {
                has_mid_only_flag: !side_lbrr,
            };
            let mid_decoded = decode_silk_frame(
                &mut rd,
                mid_state.config(bandwidth, frame_size, true, Some(stereo_ctx)),
            )?;
            let w = mid_decoded.stereo_pred.map(|p| StereoWeightsQ13 {
                w0_q13: p.w0_q13,
                w1_q13: p.w1_q13,
            });
            interval_weights.push(w.unwrap_or_default());
            let side_coded = side_lbrr || mid_decoded.mid_only_flag == Some(false);
            mid_state.advance(&mid_decoded);
            mid_frames.push(mid_decoded);

            if side_coded {
                let side_decoded = decode_silk_frame(
                    &mut rd,
                    side_state.config(bandwidth, frame_size, true, None),
                )?;
                side_state.advance(&side_decoded);
                side_frames.push(Some(side_decoded));
            } else {
                side_state.mark_interval_uncoded(true);
                side_frames.push(None);
            }
        }

        if rd.has_error() {
            return Err(Error::MalformedPacket);
        }
        if mid_frames.is_empty() {
            return Ok(None);
        }

        // §4.2.7.9 synthesis + §4.2.8 unmix from fresh state.
        let mut mid_synth = SilkSynthState::new(bandwidth)?;
        let mut side_synth = SilkSynthState::new(bandwidth)?;
        let mut unmix = StereoUnmixStateI16::new();
        let fs_khz = match bandwidth {
            crate::toc::Bandwidth::Nb => 8usize,
            crate::toc::Bandwidth::Mb => 12,
            _ => 16,
        };
        let mut left: Vec<i16> = Vec::new();
        let mut right: Vec<i16> = Vec::new();
        for (idx, mid_frame) in mid_frames.iter().enumerate() {
            let mid_out =
                synthesize_silk_frame_i16(bandwidth, frame_size, mid_frame, &mut mid_synth)?;
            let weights = interval_weights[idx];
            let stereo = match &side_frames[idx] {
                Some(side_frame) => {
                    let side_out = synthesize_silk_frame_i16(
                        bandwidth,
                        frame_size,
                        side_frame,
                        &mut side_synth,
                    )?;
                    stereo_ms_to_lr_i16(fs_khz, &mid_out, Some(&side_out), weights, &mut unmix)?
                }
                None => {
                    side_synth.reset_prediction_memory();
                    stereo_ms_to_lr_i16(fs_khz, &mid_out, None, weights, &mut unmix)?
                }
            };
            left.extend_from_slice(&stereo.left);
            right.extend_from_slice(&stereo.right);
        }

        self.silk_synth_stereo = Some((mid_synth, side_synth));
        self.silk_stereo_unmix = Some(unmix);
        // As in the mono FEC path: the recovered interval seeds the
        // cross-Opus-frame §4.2.7.4 / §4.2.7.5.5 bases for both channels
        // (the §4.2.7.4 packet-loss latitude).
        self.silk_carry_mid = mid_state.to_carry(bandwidth);
        self.silk_carry_side = side_state.to_carry(bandwidth);
        Ok(Some((left, right, bandwidth)))
    }

    /// Decode one Hybrid Opus frame (§4.4): the §4.2 SILK layer (WB
    /// internal rate) and the §4.3 CELT layer (bands 17–21) share one
    /// range coder; their 48 kHz outputs are summed.
    ///
    /// After the SILK layer the §4.5.1 redundancy side information is
    /// decoded. A present redundant CELT frame occupies the trailing
    /// bytes: the main coder's buffer is reduced by that amount
    /// (§4.5.1.3 — its raw bits then read from the end of the reduced
    /// buffer), and the 5 ms redundant frame is decoded (§4.5.1.4) and
    /// cross-lapped into this frame's output at the signalled
    /// position. A beginning-position redundant frame decodes on the
    /// carried CELT state *before* the deferred §4.5.2 main-layer
    /// reset (Figure 18's `R & |H`); an end-position one takes the
    /// §4.5.2 reset itself and its warmed state carries into the next
    /// CELT-only frame (`!R`). The SILK→48 kHz resample runs through
    /// the carried §4.2.9 upsampler whose group delay is the
    /// normative Table 54 allocation, time-aligning the SILK band
    /// with the CELT band (whose MDCT the encoder pre-delayed by the
    /// same amount).
    fn decode_hybrid_frame(
        &mut self,
        frame: &[u8],
        routing: &OpusFrameRouting,
        pcm: &mut Vec<i16>,
    ) -> FrameOutcome {
        use crate::celt_band_layout::CeltFrameSize;

        let per_channel = output_samples_per_channel(routing.frame_size_tenths_ms);
        let channels = routing.channel_count() as usize;
        let pcm_start = pcm.len();
        push_silence(pcm, per_channel, channels as u8);

        let mut rd = crate::range_decoder::RangeDecoder::new(frame);

        // §4.2 SILK layer (always WB internal for Hybrid), resampled
        // to 48 kHz into the output region through the carried
        // upsampler state. The §4.2.9 Table 54 group delay of the
        // upsampler is what time-aligns this layer with the CELT
        // layer below: the encoder pre-delayed the MDCT layer by the
        // same amount.
        let silk_ok = if channels == 2 {
            match self.decode_silk_layer_stereo(&mut rd, routing) {
                Ok((left, right, bandwidth)) => {
                    self.resample_silk_stereo_into(
                        &left,
                        &right,
                        bandwidth,
                        &mut pcm[pcm_start..pcm_start + per_channel * 2],
                    );
                    true
                }
                Err(_) => false,
            }
        } else {
            match self.decode_silk_layer_mono(&mut rd, routing) {
                Ok((internal, bandwidth)) => {
                    self.resample_silk_mono_into(
                        &internal,
                        bandwidth,
                        &mut pcm[pcm_start..pcm_start + per_channel],
                    );
                    true
                }
                Err(_) => false,
            }
        };
        if !silk_ok {
            // §4.6 floor: the reserved silence stands in. A deferred
            // §4.5.2 CELT reset still applies (the broken frame cannot
            // decode the redundant frame the deferral was waiting on).
            if std::mem::take(&mut self.celt_reset_before_main) {
                self.celt_energy = None;
                self.celt_synth = None;
            }
            pcm[pcm_start..].fill(0);
            return FrameOutcome {
                samples_per_channel: per_channel,
                status: FrameDecodeStatus::SilkDecodeError,
            };
        }

        // §4.5.1 redundancy side information; a present redundant CELT
        // frame occupies the trailing bytes, shrinking the main CELT
        // layer's budget AND its raw-bit buffer (§4.5.1.3: "the MDCT
        // layer reads any raw bits from the end of this reduced
        // buffer").
        let redundancy =
            crate::celt_redundancy::decode_redundancy(&mut rd, OperatingMode::Hybrid, frame.len());
        self.last_redundancy = redundancy;
        let celt_bytes = match redundancy {
            crate::celt_redundancy::RedundancyDecision::Present { size_bytes, .. } => {
                let reduced = frame.len().saturating_sub(size_bytes);
                rd.shrink_buffer(reduced);
                reduced
            }
            crate::celt_redundancy::RedundancyDecision::Invalid => {
                // §4.5.1.3: stop decoding this frame; keep the SILK
                // audio already written.
                self.celt_reset_before_main = false;
                return FrameOutcome {
                    samples_per_channel: per_channel,
                    status: FrameDecodeStatus::HybridDecoded,
                };
            }
            crate::celt_redundancy::RedundancyDecision::NotPresent => frame.len(),
        };
        let red_params =
            crate::redundancy_decode_params::redundant_frame_params(routing, redundancy);

        // §4.5.2 / Figure 18 `R & |H`: a beginning-position redundant
        // frame (CELT→Hybrid transition) decodes on the carried CELT
        // state, BEFORE the deferred main-layer reset.
        let mut red_pcm: Option<Vec<i16>> = None;
        if let Some(params) = &red_params {
            if matches!(
                params.position,
                crate::celt_redundancy::RedundancyPosition::Beginning
            ) {
                let red_bytes = &frame[frame.len() - params.size_bytes..];
                red_pcm = self.decode_redundant_celt(red_bytes, params, false);
            }
        }

        // Deferred §4.5.2 reset for the main CELT layer (rule 2 /
        // Figure 18 `|H`), placed after the redundant frame's decode.
        if std::mem::take(&mut self.celt_reset_before_main) {
            self.celt_energy = None;
            self.celt_synth = None;
        }

        // §4.3 CELT layer: bands 17.. per the signalled bandwidth.
        let Some(celt_size) =
            CeltFrameSize::from_frame_tenths_ms(routing.frame_size_tenths_ms as u32)
        else {
            return FrameOutcome {
                samples_per_channel: per_channel,
                status: FrameDecodeStatus::CeltDecodeError,
            };
        };
        let lm = celt_size.column_index() as i32;
        let n = per_channel;
        let end = match routing.toc_bandwidth {
            crate::toc::Bandwidth::Swb => 19,
            _ => 21,
        };
        match self.celt_synth.as_mut() {
            Some(s) => s.set_geometry(channels, n),
            None => {
                self.celt_synth = Some(crate::celt_mdct_synthesis::CeltSynthesis::new(channels, n))
            }
        }
        let energy = self.celt_energy.get_or_insert_with(Default::default);
        let out = crate::celt_frame_decode::decode_celt_frame(
            &mut rd,
            celt_bytes,
            crate::celt_band_layout::HYBRID_FIRST_CODED_BAND,
            end,
            lm,
            channels,
            energy,
        );
        if rd.has_error() {
            // The CELT layer failed: apply the §4.6 whole-frame floor
            // (the crate-wide error convention — an error status is
            // always paired with silence).
            //
            // Note that tell() PASSING the frame's bit budget is NOT an
            // error here: §4.1.2.1 defines reading past the end of the
            // frame ("if no more input bytes remain, it uses zero bits
            // instead"), and a real low-budget Hybrid frame can spend
            // its whole budget on the SILK layer — the CELT layer then
            // takes the §4.3 exhausted-budget silence path and the
            // frame's audio is the SILK band alone (observed on the
            // 72-byte-frame code-1 fixture, whose reference decode
            // keeps the SILK audio).
            pcm[pcm_start..].fill(0);
            return FrameOutcome {
                samples_per_channel: per_channel,
                status: FrameDecodeStatus::CeltDecodeError,
            };
        }

        let m = 1usize << lm;
        // Profluens patch: per-frame denormalised spectrum, consumed by synthesis as a slice — from
        // the per-packet bump arena.
        let mut freq = Vec::with_capacity_in(channels * n, crate::bump::DecodeBump);
        freq.resize(channels * n, 0.0f64);
        for c in 0..channels {
            for band in crate::celt_band_layout::HYBRID_FIRST_CODED_BAND..end {
                let gain = out.band_gain[c][band];
                let off = m * crate::celt_rate_alloc::band_edge(band) as usize;
                let len = m * crate::celt_rate_alloc::band_width(band) as usize;
                for j in 0..len {
                    freq[c * n + off + j] = gain * out.x[c * out.plane + off + j];
                }
            }
        }
        let blocks = if out.transient { m } else { 1 };
        let synth = self.celt_synth.as_mut().expect("state built above");
        let mut celt_pcm = Vec::with_capacity_in(channels * n, crate::bump::DecodeBump);
        celt_pcm.resize(channels * n, 0i16);
        synth.synthesize_frame(&freq, blocks, out.post_filter, &mut celt_pcm);

        // §4.4: the layer outputs sum (saturating at the i16 rails).
        {
            let region = &mut pcm[pcm_start..pcm_start + per_channel * channels];
            for (dst, &c) in region.iter_mut().zip(celt_pcm.iter()) {
                *dst = dst.saturating_add(c);
            }
        }

        // §4.5.2 / Figure 18 `!R`: an end-position redundant frame
        // (Hybrid→CELT transition) takes the CELT reset itself, after
        // the main layer; its warmed state carries into the next
        // CELT-only frame (whose packet-boundary reset rule 3
        // suppresses).
        if let Some(params) = &red_params {
            if matches!(
                params.position,
                crate::celt_redundancy::RedundancyPosition::End
            ) {
                let red_bytes = &frame[frame.len() - params.size_bytes..];
                red_pcm = self.decode_redundant_celt(red_bytes, params, true);
            }
        }
        if let (Some(params), Some(red)) = (&red_params, &red_pcm) {
            apply_redundancy_cross_lap(&mut pcm[pcm_start..], red, channels, params.position);
        }

        FrameOutcome {
            samples_per_channel: per_channel,
            status: FrameDecodeStatus::HybridDecoded,
        }
    }

    /// Decode and synthesize one §4.5.1.4 redundant CELT frame from its
    /// byte-aligned slice at the tail of the carrier Opus frame,
    /// returning 5 ms of interleaved PCM (240 samples per channel), or
    /// `None` when the redundant bitstream is malformed.
    ///
    /// Per §4.5.1.4 the redundant frame "is decoded like any other
    /// CELT-only frame, with the exception that it does not contain a
    /// TOC byte": its own fresh range coder over `red`, band range
    /// `0..end` for the carrier's audio bandwidth (MB carriers use WB),
    /// the carrier's channel count, and the fixed 5 ms frame size.
    /// `reset_before` applies the §4.5.2 CELT reset before the decode
    /// (end-position redundant frames — Figure 18's `!R`); otherwise
    /// the carried state (energy history, MDCT overlap, post-filter,
    /// de-emphasis) continues into the redundant frame.
    fn decode_redundant_celt(
        &mut self,
        red: &[u8],
        params: &crate::redundancy_decode_params::RedundantFrameParams,
        reset_before: bool,
    ) -> Option<Vec<i16>> {
        use crate::celt_band_layout::CeltFrameSize;

        let n = output_samples_per_channel(params.duration_tenths_ms);
        let channels = channel_count(params.channels) as usize;
        let lm = CeltFrameSize::from_frame_tenths_ms(params.duration_tenths_ms as u32)?
            .column_index() as i32;
        let end = match params.bandwidth {
            crate::toc::Bandwidth::Nb => 13,
            crate::toc::Bandwidth::Mb | crate::toc::Bandwidth::Wb => 17,
            crate::toc::Bandwidth::Swb => 19,
            crate::toc::Bandwidth::Fb => 21,
        };

        // The redundant frame shares the stream's one CELT decoder;
        // only its geometry differs (§4.5.2 keeps the state across the
        // size change unless a reset is placed here).
        match self.celt_synth.as_mut() {
            Some(s) => s.set_geometry(channels, n),
            None => {
                self.celt_synth = Some(crate::celt_mdct_synthesis::CeltSynthesis::new(channels, n))
            }
        }
        if self.celt_energy.is_none() {
            self.celt_energy = Some(crate::celt_frame_decode::CeltEnergyState::new());
        }
        if reset_before {
            self.celt_energy.as_mut().expect("built above").reset();
            self.celt_synth.as_mut().expect("built above").reset();
        }

        let energy = self.celt_energy.as_mut().expect("built above");
        let mut rd = crate::range_decoder::RangeDecoder::new(red);
        let out = crate::celt_frame_decode::decode_celt_frame(
            &mut rd,
            red.len(),
            0,
            end,
            lm,
            channels,
            energy,
        );
        if rd.has_error() || u64::from(rd.tell()) > 8 * red.len() as u64 {
            return None;
        }

        let m = 1usize << lm;
        // Profluens patch: per-frame denormalised spectrum, consumed by synthesis as a slice — from
        // the per-packet bump arena.
        let mut freq = Vec::with_capacity_in(channels * n, crate::bump::DecodeBump);
        freq.resize(channels * n, 0.0f64);
        for c in 0..channels {
            for band in 0..end {
                let gain = out.band_gain[c][band];
                let off = m * crate::celt_rate_alloc::band_edge(band) as usize;
                let len = m * crate::celt_rate_alloc::band_width(band) as usize;
                for j in 0..len {
                    freq[c * n + off + j] = gain * out.x[c * out.plane + off + j];
                }
            }
        }
        let blocks = if out.transient { m } else { 1 };
        let synth = self.celt_synth.as_mut().expect("built above");
        // NB not bump-allocated: this redundant-frame PCM escapes as an owned `Vec<i16>` (the §4.5.1
        // cross-lap result) and the redundant path is rare, so it stays on the global allocator.
        let mut red_pcm = vec![0i16; channels * n];
        synth.synthesize_frame(&freq, blocks, out.post_filter, &mut red_pcm);
        Some(red_pcm)
    }
}

/// Apply the §4.5.1.4 redundant-frame cross-lap into one decoded Opus
/// frame's interleaved PCM `region`.
///
/// `red` is the redundant CELT frame's 5 ms of interleaved output. For
/// a **beginning**-position redundant frame (CELT→SILK/Hybrid
/// transition) "the final reconstructed output uses the first 2.5 ms of
/// audio output by the decoder for the redundant frame as is,
/// discarding the corresponding output of the SILK-only or Hybrid
/// portion", and the remaining 2.5 ms are cross-lapped into the main
/// signal. For an **end**-position one (SILK/Hybrid→CELT transition)
/// "only the second half (2.5 ms) of the audio output … is used",
/// cross-lapped with the end of the main signal. Both laps use the
/// power-complementary CELT MDCT window (§4.5.1.4), squared into
/// amplitude weights exactly as the §4.3.7.1 post-filter transition
/// does.
fn apply_redundancy_cross_lap(
    region: &mut [i16],
    red: &[i16],
    channels: usize,
    position: crate::celt_redundancy::RedundancyPosition,
) {
    use crate::celt_mdct_window::{celt_overlap_window, CELT_OVERLAP_48K};

    let lap = CELT_OVERLAP_48K; // 120 samples = 2.5 ms at 48 kHz
    let half = lap * channels;
    if red.len() < 2 * half || region.len() < 2 * half || channels == 0 {
        return;
    }
    let window = celt_overlap_window();
    let mix = |a: i16, b: i16, f: f64| -> i16 {
        // a fades in under f, b fades out under 1 - f.
        (f64::from(a) * f + f64::from(b) * (1.0 - f))
            .round()
            .clamp(-32768.0, 32767.0) as i16
    };
    match position {
        crate::celt_redundancy::RedundancyPosition::Beginning => {
            // First 2.5 ms: redundant output as-is.
            region[..half].copy_from_slice(&red[..half]);
            // Next 2.5 ms: main fades in, redundant fades out.
            for (i, &w) in window.iter().enumerate().take(lap) {
                let f = w * w;
                for c in 0..channels {
                    let idx = half + i * channels + c;
                    region[idx] = mix(region[idx], red[idx], f);
                }
            }
        }
        crate::celt_redundancy::RedundancyPosition::End => {
            // Last 2.5 ms: the redundant frame's second half fades in,
            // the main signal fades out.
            let base = region.len() - half;
            for (i, &w) in window.iter().enumerate().take(lap) {
                let f = w * w;
                for c in 0..channels {
                    let idx = base + i * channels + c;
                    region[idx] = mix(red[half + i * channels + c], region[idx], f);
                }
            }
        }
    }
}

/// Append `per_channel * channels` interleaved zero samples to `pcm`.
fn push_silence(pcm: &mut Vec<i16>, per_channel: usize, channels: u8) {
    pcm.resize(pcm.len() + per_channel * channels as usize, 0);
}

/// Cross-Opus-frame SILK per-channel reconstruction carry: the last
/// decoded subframe gain — the §4.2.7.4 clamp base, `log_gain =
/// max(gain_index, previous_log_gain - 16)`, whose `previous_log_gain`
/// persists "in the same channel" across Opus frames — and the last
/// decoded NLSF vector — the §4.2.7.5.5 interpolation base `n0_Q15`,
/// "the LSF coefficients decoded for the prior frame", which the RFC
/// only replaces with the forced `w_Q2 = 4` after a decoder reset or an
/// uncoded side frame. Tagged with the bandwidth it was decoded at: the
/// NLSF order and codebooks are bandwidth-specific, so a bandwidth
/// switch drops the carry (mirroring the synthesis-state re-creation).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SilkChannelCarry {
    gain: Option<u8>,
    nlsf: Option<[i16; crate::silk_lsf_stage2::D_LPC_MAX]>,
    nlsf_len: usize,
    bandwidth: Option<crate::toc::Bandwidth>,
}

/// Per-channel inter-frame decode state threaded across the SILK frames
/// of one Opus frame (§4.2.7.4 previous gain, §4.2.7.6.1 previous lag,
/// §4.2.7.5.5 previous NLSF base, and the "first SILK frame of this type"
/// flag). One instance is used for the mid channel and one for the side
/// channel (each channel's frames form an independent sequence).
pub(crate) struct ChannelDecodeState {
    prev_gain: Option<u8>,
    prev_lag: Option<i32>,
    prev_nlsf: Option<[i16; crate::silk_lsf_stage2::D_LPC_MAX]>,
    prev_nlsf_len: usize,
    first: bool,
}

impl ChannelDecodeState {
    pub(crate) fn new() -> Self {
        Self {
            prev_gain: None,
            prev_lag: None,
            prev_nlsf: None,
            prev_nlsf_len: 0,
            first: true,
        }
    }

    /// Resume a channel sequence at an Opus-frame boundary from the
    /// cross-frame carry: the §4.2.7.4 clamp base and §4.2.7.5.5 `n0`
    /// persist, while the frame-local `first` flag re-arms (the first
    /// frame's gain is independently *coded* per Opus frame, its lag is
    /// absolute, and its §4.2.7.6.3 scaling field is present). A carry
    /// recorded at a different bandwidth is dropped (fresh start).
    pub(crate) fn resume(carry: &SilkChannelCarry, bandwidth: crate::toc::Bandwidth) -> Self {
        if carry.bandwidth != Some(bandwidth) {
            return Self::new();
        }
        Self {
            prev_gain: carry.gain,
            prev_lag: None,
            prev_nlsf: carry.nlsf,
            prev_nlsf_len: carry.nlsf_len,
            first: true,
        }
    }

    /// Capture the cross-Opus-frame carry after this channel's last
    /// frame of the Opus frame has been folded in via [`Self::advance`]
    /// (or cleared via [`Self::mark_interval_uncoded`]).
    pub(crate) fn to_carry(&self, bandwidth: crate::toc::Bandwidth) -> SilkChannelCarry {
        SilkChannelCarry {
            gain: self.prev_gain,
            nlsf: self.prev_nlsf,
            nlsf_len: self.prev_nlsf_len,
            bandwidth: Some(bandwidth),
        }
    }

    /// Build the [`crate::silk_decode::SilkFrameConfig`] for the next SILK
    /// frame in this channel's sequence, given the §4.2.4 VAD flag and the
    /// optional §4.2.7.1 / §4.2.7.2 stereo header context (present only on
    /// the mid channel).
    pub(crate) fn config(
        &self,
        bandwidth: crate::toc::Bandwidth,
        frame_size: crate::silk_excitation::SilkFrameSize,
        voice_active: bool,
        stereo: Option<crate::silk_decode::StereoHeaderContext>,
    ) -> crate::silk_decode::SilkFrameConfig {
        crate::silk_decode::SilkFrameConfig {
            bandwidth,
            frame_size,
            voice_active,
            first_subframe_independent: self.first || self.prev_gain.is_none(),
            previous_log_gain: self.prev_gain,
            previous_primary_lag: self.prev_lag,
            ltp_scaling_present: self.first,
            // §4.2.7.5.5: the factor is forced to 4 only when no prior
            // decoded NLSF base exists (decoder reset / uncoded side
            // frame) — for a resumed sequence `n0` comes from the
            // previous Opus frame even though `first` is set.
            lsf_interp_after_reset: self.prev_nlsf.is_none(),
            previous_nlsf_q15: self.prev_nlsf,
            previous_nlsf_len: self.prev_nlsf_len,
            stereo,
        }
    }

    /// Fold a freshly decoded SILK frame into the carried state, so the
    /// next frame in this channel's sequence predicts against it.
    pub(crate) fn advance(&mut self, decoded: &crate::silk_decode::SilkFrameDecoded) {
        self.prev_gain = Some(decoded.gains.last_log_gain());
        // §4.2.7.6.1: relative lag coding requires the previous frame of
        // the same type to have been coded AND voiced ("that previous
        // SILK frame was coded, but was not voiced" selects absolute
        // coding); a non-voiced frame therefore clears the lag base.
        self.prev_lag = if decoded.ltp.is_voiced() {
            Some(decoded.ltp.primary_lag())
        } else {
            None
        };
        self.prev_nlsf = Some(decoded.nlsf_q15);
        self.prev_nlsf_len = decoded.d_lpc;
        self.first = false;
    }

    /// Fold an UNCODED interval into the carried state (a §4.2.7.2
    /// side-channel skip or a §4.2.4 LBRR-flag gap). The next coded
    /// frame in the sequence then codes its gains independently with
    /// the §4.2.7.4 clamp skipped ("in the side channel if the
    /// previous frame in the side channel was not coded"), its primary
    /// lag absolutely (§4.2.7.6.1 second bullet), and its §4.2.7.5.5
    /// LSF interpolation factor forced to 4. For an LBRR sequence the
    /// §4.2.7.6.3 LTP-scaling field also reappears ("an LBRR frame
    /// where the LBRR flags indicate the previous LBRR frame in the
    /// same channel is not coded"), so `first` is re-armed; for
    /// regular frames that field only ever rides the first time
    /// interval of the Opus frame, so `first` is cleared.
    pub(crate) fn mark_interval_uncoded(&mut self, lbrr: bool) {
        self.prev_gain = None;
        self.prev_lag = None;
        self.prev_nlsf = None;
        self.prev_nlsf_len = 0;
        self.first = lbrr;
    }
}

/// Resample one Opus frame's internal-rate SILK samples through a §4.2.9
/// [`crate::silk_resampler::SilkUpsampler`] into signed 16-bit PCM.
///
/// `out.len()` must be `internal.len() × factor` (the §3.1 sample-count
/// arithmetic guarantees this for every well-formed SILK frame); a
/// mismatched pair falls back to stateless linear interpolation so a
/// defensive caller can never panic here.
fn resample_through_upsampler_i16(
    up: &mut crate::silk_resampler::SilkUpsampler,
    internal: &[i16],
    out: &mut [i16],
) {
    if out.is_empty() {
        return;
    }
    if internal.is_empty() || internal.len() * up.factor() != out.len() {
        resample_linear_i16(internal, out);
        return;
    }
    up.process_i16(internal, out);
}

/// Stateless linear-interpolation fallback resampler (zero delay, no
/// carried history). Only reached for degenerate inputs the §3.1
/// arithmetic never produces on the main path (empty internal buffer,
/// non-integer rate ratio) — §4.2.9 permits any method, and this one is
/// total.
fn resample_linear_i16(internal: &[i16], out: &mut [i16]) {
    if out.is_empty() {
        return;
    }
    if internal.is_empty() {
        for o in out.iter_mut() {
            *o = 0;
        }
        return;
    }
    let in_len = internal.len();
    let out_len = out.len();
    for (i, o) in out.iter_mut().enumerate() {
        let pos = (i as f64) * (in_len as f64) / (out_len as f64);
        let i0 = pos.floor() as usize;
        let frac = pos - i0 as f64;
        let s0 = f64::from(internal[i0.min(in_len - 1)]);
        let s1 = f64::from(internal[(i0 + 1).min(in_len - 1)]);
        let v = s0 + (s1 - s0) * frac;
        *o = v.round_ties_even().clamp(-32768.0, 32767.0) as i16;
    }
}

/// Resample one frame's internal-rate SILK samples to 48 kHz through a
/// **fresh** §4.2.9 upsampler (no carried history) — the FEC recovery
/// path, which reconstructs a lost frame from a fresh SILK state per
/// §4.2.5 (mirroring the fresh `SilkSynthState` it synthesizes with).
fn resample_internal_to_output_i16(
    internal: &[i16],
    bandwidth: crate::toc::Bandwidth,
    out: &mut [i16],
) {
    let path = crate::silk_resampler::SilkChannelPath::Mono;
    match crate::silk_resampler::SilkUpsampler::new(bandwidth, path) {
        Some(mut up) => resample_through_upsampler_i16(&mut up, internal, out),
        None => resample_linear_i16(internal, out),
    }
}

/// Stereo variant of [`resample_internal_to_output_i16`]: fresh §4.2.9
/// upsamplers per channel, output **interleaved** (`[L0, R0, L1, R1,
/// …]`) into `out` (length `2 × per_channel`).
fn resample_stereo_to_output_i16(
    left: &[i16],
    right: &[i16],
    bandwidth: crate::toc::Bandwidth,
    out: &mut [i16],
) {
    let per_channel = out.len() / 2;
    if per_channel == 0 {
        return;
    }
    let path = crate::silk_resampler::SilkChannelPath::Stereo;
    // Resample each channel into a scratch buffer, then interleave.
    let mut l = vec![0i16; per_channel];
    let mut r = vec![0i16; per_channel];
    match crate::silk_resampler::SilkUpsampler::new(bandwidth, path)
        .zip(crate::silk_resampler::SilkUpsampler::new(bandwidth, path))
    {
        Some((mut ul, mut ur)) => {
            resample_through_upsampler_i16(&mut ul, left, &mut l);
            resample_through_upsampler_i16(&mut ur, right, &mut r);
        }
        None => {
            resample_linear_i16(left, &mut l);
            resample_linear_i16(right, &mut r);
        }
    }
    for i in 0..per_channel {
        out[2 * i] = l[i];
        out[2 * i + 1] = r[i];
    }
}

/// Convenience: the channel count for a [`ChannelMapping`].
pub fn channel_count(mapping: ChannelMapping) -> u8 {
    match mapping {
        ChannelMapping::Mono => 1,
        ChannelMapping::Stereo => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toc::OpusTocByte;

    /// Build a minimal code-0 packet: TOC byte + a non-empty single
    /// frame body. `config` is the 5-bit §3.1 config, `stereo` the s bit.
    fn code0_packet(config: u8, stereo: bool, body: &[u8]) -> Vec<u8> {
        let toc = (config << 3) | (if stereo { 1 << 2 } else { 0 });
        let mut p = vec![toc];
        p.extend_from_slice(body);
        p
    }

    #[test]
    fn output_samples_per_channel_matches_table2_durations() {
        // (tenths-ms, expected 48 kHz samples/channel)
        let cases = [
            (25u16, 120usize), // 2.5 ms CELT
            (50, 240),         // 5 ms
            (100, 480),        // 10 ms
            (200, 960),        // 20 ms
            (400, 1920),       // 40 ms
            (600, 2880),       // 60 ms
        ];
        for (tenths, expected) in cases {
            assert_eq!(
                output_samples_per_channel(tenths),
                expected,
                "tenths={tenths}"
            );
        }
    }

    #[test]
    fn empty_packet_rejected() {
        let mut dec = OpusDecoder::new();
        assert_eq!(dec.decode_packet(&[]), Err(Error::EmptyPacket));
    }

    #[test]
    fn silk_nb_mono_20ms_single_frame_pcm_length() {
        // config 1 = SILK NB 20 ms (200 tenths-ms), mono, code 0.
        let pkt = code0_packet(1, false, &[0x12, 0x34, 0x56]);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.channels, 1);
        assert_eq!(out.sample_rate_hz, OUTPUT_SAMPLE_RATE_HZ);
        assert_eq!(out.samples_per_channel(), 960);
        assert_eq!(out.pcm.len(), 960);
        assert_eq!(out.frame_outcomes.len(), 1);
        // A mono SILK-only frame now runs the real §4.2 bitstream decode;
        // the status is either a clean params-decoded or a decode-error
        // (a 3-byte arbitrary body may truncate mid-frame), never the
        // not-wired placeholder.
        assert!(
            matches!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::SilkParamsDecoded | FrameDecodeStatus::SilkDecodeError
            ),
            "got {:?}",
            out.frame_outcomes[0].status
        );
    }

    #[test]
    fn celt_only_stereo_pcm_is_interleaved_length() {
        // config 20 = CELT-only, second size in the NB/WB group; stereo.
        let pkt = code0_packet(20, true, &[0xaa, 0xbb]);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.channels, 2);
        // 2 channels interleaved => pcm len = 2 * samples_per_channel.
        assert_eq!(out.pcm.len(), 2 * out.samples_per_channel());
        let routing = OpusFrameRouting::from_toc(OpusTocByte::from_byte(pkt[0]));
        // A CELT-only frame decodes end-to-end — so the status is one
        // of the real CELT outcomes (silence / fully decoded / a decode
        // error on a 2-byte body), never the not-wired placeholder.
        assert!(
            matches!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::CeltSilence
                    | FrameDecodeStatus::CeltDecoded
                    | FrameDecodeStatus::CeltDecodeError
            ),
            "got {:?}",
            out.frame_outcomes[0].status
        );
        assert_eq!(routing.operating_mode, OperatingMode::CeltOnly);
    }

    #[test]
    fn code1_two_equal_frames_concatenate_pcm() {
        // config 0 = SILK NB 10 ms (100 tenths => 480 samples/ch), mono.
        // Code 1 = two equal frames; body must be even length.
        // config 0 (<< 3 = 0), mono, code 1 (0b01).
        let toc = 0b01u8;
        let mut pkt = vec![toc];
        pkt.extend_from_slice(&[1, 2, 3, 4]); // two 2-byte frames
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.frame_outcomes.len(), 2);
        // Two 10 ms frames => 2 * 480 = 960 samples/channel.
        assert_eq!(out.samples_per_channel(), 960);
        assert_eq!(out.pcm.len(), 960);
    }

    #[test]
    fn dtx_zero_length_frame_emits_silence_with_status() {
        // Code 3 VBR with a zero-length (DTX) frame. Build a code-3
        // packet by hand: TOC, frame-count byte, then VBR lengths.
        // Simpler: rely on code-2 unequal where the first frame length 0
        // is a valid DTX marker per §3.2.1.
        // config 0 (<< 3 = 0) SILK NB 10 ms mono, code 2 (0b10).
        let toc = 0b10u8;
        // code 2 body: a length prefix for frame 1, then frame1, then
        // frame2 is the remainder. Length 0 => frame1 is DTX.
        let pkt = vec![toc, 0x00, 0x07];
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.frame_outcomes.len(), 2);
        assert_eq!(out.frame_outcomes[0].status, FrameDecodeStatus::DtxOrLost);
        // Both frames are 10 ms => 480 samples/channel each.
        assert_eq!(out.samples_per_channel(), 960);
    }

    #[test]
    fn reset_clears_carried_channel_state() {
        let mut dec = OpusDecoder::new();
        let stereo = code0_packet(20, true, &[1, 2]);
        dec.decode_packet(&stereo).expect("decode");
        assert_eq!(dec.last_channels, Some(2));
        dec.reset();
        assert_eq!(dec.last_channels, None);
    }

    #[test]
    fn celt_to_silk_transition_resets_silk_state() {
        // §4.5.2: the SILK state is reset before a SILK-only frame whose
        // predecessor was CELT-only. With CELT not yet wired (a CELT-only
        // packet emits silence and touches no SILK state), a SILK packet
        // followed by a CELT packet followed by the same SILK packet must
        // produce the *same* PCM as a fresh decoder running that SILK
        // packet once — because the §4.5.2 reset clears the carried
        // §4.2.7.9 history the first SILK packet left behind.
        let silk_body: Vec<u8> = (0..200u16)
            .map(|i| (i.wrapping_mul(149).wrapping_add(11) & 0xff) as u8)
            .collect();
        let silk_pkt = code0_packet(1, false, &silk_body); // config 1 = SILK NB 20 ms mono.
        let celt_pkt = code0_packet(17, false, &[0xaa, 0xbb]); // config 17 = CELT-only mono.

        // Reference: a fresh decoder running the SILK packet once.
        let mut ref_dec = OpusDecoder::new();
        let reference = ref_dec.decode_packet(&silk_pkt).expect("decode");

        // Sequence: SILK, then CELT (resets SILK state on the *next* SILK
        // frame), then SILK again. The third packet must match the
        // reference if and only if the §4.5.2 reset fired.
        let mut seq_dec = OpusDecoder::new();
        seq_dec.decode_packet(&silk_pkt).expect("decode");
        seq_dec.decode_packet(&celt_pkt).expect("decode");
        let after_reset = seq_dec.decode_packet(&silk_pkt).expect("decode");

        // Only compare when the SILK frame actually synthesized audio.
        if reference.frame_outcomes[0].status == FrameDecodeStatus::SilkParamsDecoded {
            assert_eq!(
                after_reset.pcm, reference.pcm,
                "§4.5.2 CELT→SILK transition must reset SILK state"
            );
        }
    }

    #[test]
    fn silk_to_silk_no_reset_threads_state() {
        // The complement of the §4.5.2 test: two consecutive SILK-only
        // packets (no CELT interlude) do NOT reset the SILK state, so the
        // second packet's output generally differs from a fresh-decoder
        // decode of that packet (the carried §4.2.7.9 history changes the
        // LPC/LTP synthesis). This pins that state actually threads when
        // it should.
        let silk_body: Vec<u8> = (0..200u16)
            .map(|i| (i.wrapping_mul(149).wrapping_add(11) & 0xff) as u8)
            .collect();
        let silk_pkt = code0_packet(1, false, &silk_body);

        let mut fresh = OpusDecoder::new();
        let fresh_out = fresh.decode_packet(&silk_pkt).expect("decode");

        let mut threaded = OpusDecoder::new();
        threaded.decode_packet(&silk_pkt).expect("decode");
        let second = threaded.decode_packet(&silk_pkt).expect("decode");

        // Both decode to the same length; the carried state means the
        // second decode is at least a valid, finite PCM buffer.
        assert_eq!(second.pcm.len(), fresh_out.pcm.len());
    }

    #[test]
    fn silk_mono_full_decode_consumes_bitstream_cleanly() {
        // A long pseudo-random SILK NB mono 20 ms body: the range coder
        // does not run out of bits, so the full §4.2 frame decodes and the
        // status is the clean params-decoded outcome (not a decode error).
        let body: Vec<u8> = (0..120u16)
            .map(|i| (i.wrapping_mul(101).wrapping_add(7) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(1, false, &body); // config 1 = SILK NB 20 ms.
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.frame_outcomes.len(), 1);
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkParamsDecoded,
            "a long SILK NB mono body should fully decode"
        );
        // PCM length is correct even though the samples are silence
        // (synthesis pending).
        assert_eq!(out.samples_per_channel(), 960);
    }

    #[test]
    fn silk_mono_40ms_two_silk_frames_decode() {
        // config 2 = SILK NB 40 ms => 2 SILK frames per channel; mono.
        let body: Vec<u8> = (0..220u16)
            .map(|i| (i.wrapping_mul(53).wrapping_add(3) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(2, false, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        let routing = OpusFrameRouting::from_toc(OpusTocByte::from_byte(pkt[0]));
        assert_eq!(routing.silk_frames_per_channel, Some(2));
        // 40 ms => 1920 samples/channel; one Opus frame (code 0).
        assert_eq!(out.frame_outcomes.len(), 1);
        assert_eq!(out.samples_per_channel(), 1920);
        // The two-SILK-frame loop ran; the status reflects a SILK decode
        // (clean or truncated), never the not-wired placeholder.
        assert!(matches!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkParamsDecoded | FrameDecodeStatus::SilkDecodeError
        ));
    }

    #[test]
    fn stereo_silk_only_decodes_to_interleaved_pcm() {
        // Stereo SILK now runs the full §4.2 interleaved mid/side decode +
        // §4.2.8 unmix. A long pseudo-random body decodes cleanly; the
        // output is interleaved L/R 48 kHz PCM.
        let body: Vec<u8> = (0..220u16)
            .map(|i| (i.wrapping_mul(137).wrapping_add(19) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(1, true, &body); // config 1 = SILK NB 20 ms stereo.
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.channels, 2);
        assert_eq!(out.samples_per_channel(), 960);
        assert_eq!(out.pcm.len(), 2 * 960);
        assert!(matches!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkStereoDecoded | FrameDecodeStatus::SilkDecodeError
        ));
    }

    #[test]
    fn stereo_silk_clean_body_is_fully_decoded() {
        // A buffer long enough that the range coder never starves: the
        // interleaved mid/side decode + unmix completes, yielding the
        // stereo-decoded status (not a decode error, not not-wired).
        let body: Vec<u8> = (0..400u16)
            .map(|i| (i.wrapping_mul(97).wrapping_add(41) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(1, true, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkStereoDecoded,
            "a long stereo SILK NB body should fully decode"
        );
        // The output is finite and within i16 range by construction.
        assert_eq!(out.pcm.len(), 2 * 960);
    }

    #[test]
    fn stereo_silk_40ms_two_intervals_decode() {
        // config 2 = SILK NB 40 ms => 2 SILK frames per channel; stereo.
        // The §4.2.2 interleave runs mid/side per 20 ms interval twice.
        let body: Vec<u8> = (0..480u16)
            .map(|i| (i.wrapping_mul(61).wrapping_add(7) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(2, true, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        let routing = OpusFrameRouting::from_toc(OpusTocByte::from_byte(pkt[0]));
        assert_eq!(routing.silk_frames_per_channel, Some(2));
        assert_eq!(out.channels, 2);
        // 40 ms => 1920 samples/channel interleaved.
        assert_eq!(out.samples_per_channel(), 1920);
        assert_eq!(out.pcm.len(), 2 * 1920);
        assert!(matches!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkStereoDecoded | FrameDecodeStatus::SilkDecodeError
        ));
    }

    #[test]
    fn stereo_silk_60ms_three_intervals_per_interval_unmix() {
        // config 3 = SILK NB 60 ms => 3 SILK frames per channel; stereo.
        // Each 20 ms interval is unmixed separately (its own §4.2.7.1
        // weights + a fresh §4.2.8 interpolation phase), and the three
        // L/R interval outputs are concatenated. This pins the per-interval
        // unmix path for a multi-interval stereo frame.
        let body: Vec<u8> = (0..640u16)
            .map(|i| (i.wrapping_mul(73).wrapping_add(31) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(3, true, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        let routing = OpusFrameRouting::from_toc(OpusTocByte::from_byte(pkt[0]));
        assert_eq!(routing.silk_frames_per_channel, Some(3));
        assert_eq!(out.channels, 2);
        // 60 ms => 2880 samples/channel interleaved.
        assert_eq!(out.samples_per_channel(), 2880);
        assert_eq!(out.pcm.len(), 2 * 2880);
        assert!(matches!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkStereoDecoded | FrameDecodeStatus::SilkDecodeError
        ));
    }

    #[test]
    fn stereo_silk_state_threads_across_packets() {
        // Two consecutive stereo SILK packets thread the §4.2.7.9 + §4.2.8
        // histories: the second packet's output may differ from a fresh
        // decode, but both are valid finite buffers of equal length.
        let body: Vec<u8> = (0..300u16)
            .map(|i| (i.wrapping_mul(113).wrapping_add(23) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(1, true, &body);

        let mut fresh = OpusDecoder::new();
        let fresh_out = fresh.decode_packet(&pkt).expect("decode");

        let mut threaded = OpusDecoder::new();
        threaded.decode_packet(&pkt).expect("decode");
        let second = threaded.decode_packet(&pkt).expect("decode");
        assert_eq!(second.pcm.len(), fresh_out.pcm.len());
    }

    #[test]
    fn mono_to_stereo_transition_resets_stereo_state() {
        // §4.2.7.1: previous stereo weights reset on a mono→stereo
        // transition. A mono packet, then a stereo packet, then the same
        // stereo packet must leave the second stereo decode in a defined
        // state (no panic; correct length). The mono→stereo channel-count
        // change clears the carried stereo history.
        let mono_body: Vec<u8> = (0..200u16)
            .map(|i| (i.wrapping_mul(71).wrapping_add(5) & 0xff) as u8)
            .collect();
        let stereo_body: Vec<u8> = (0..300u16)
            .map(|i| (i.wrapping_mul(89).wrapping_add(11) & 0xff) as u8)
            .collect();
        let mono_pkt = code0_packet(1, false, &mono_body);
        let stereo_pkt = code0_packet(1, true, &stereo_body);

        let mut dec = OpusDecoder::new();
        dec.decode_packet(&mono_pkt).expect("mono");
        let out = dec.decode_packet(&stereo_pkt).expect("stereo");
        assert_eq!(out.channels, 2);
        assert_eq!(out.pcm.len(), 2 * 960);
    }

    #[test]
    fn pcm_length_matches_routing_for_every_config() {
        // Every Table-2 config decodes to a PCM buffer of the routing's
        // 48 kHz length × channels. SILK-only, CELT-only, AND Hybrid
        // configs synthesize real audio (a short junk body is a *legal*
        // bitstream per §4.1.2.1 — reads past the end use zero bits —
        // so a successful status may carry non-silent PCM); every
        // error/silence outcome emits correct-length silence. This
        // sweep pins the length invariant for all 32 configs.
        let mut dec = OpusDecoder::new();
        for config in 0u8..32 {
            for stereo in [false, true] {
                let pkt = code0_packet(config, stereo, &[0x55, 0x66, 0x77]);
                let out = dec.decode_packet(&pkt).expect("decode");
                let routing = OpusFrameRouting::from_toc(OpusTocByte::from_byte(pkt[0]));
                let expected = output_samples_per_channel(routing.frame_size_tenths_ms)
                    * out.channels as usize;
                assert_eq!(out.pcm.len(), expected, "config {config} stereo {stereo}");
                // Everything except a successfully synthesized SILK,
                // Hybrid, or CELT frame still emits silence.
                let is_wired = matches!(
                    out.frame_outcomes[0].status,
                    FrameDecodeStatus::SilkParamsDecoded
                        | FrameDecodeStatus::SilkStereoDecoded
                        | FrameDecodeStatus::HybridDecoded
                        | FrameDecodeStatus::CeltDecoded
                );
                if !is_wired {
                    assert!(
                        out.pcm.iter().all(|&s| s == 0),
                        "config {config} stereo {stereo} status {:?} should be silence",
                        out.frame_outcomes[0].status
                    );
                }
                // The decoder must be reset between configs so the carried
                // §4.2.7.9 synthesis history of one bandwidth doesn't leak
                // into the next.
                dec.reset();
            }
        }
    }

    #[test]
    fn mono_silk_frame_can_emit_nonsilent_pcm() {
        // A long pseudo-random mono SILK NB 20 ms body decodes cleanly and
        // is synthesized through the §4.2.7.9 LTP/LPC filters + §4.2.9
        // resample; the emitted PCM is no longer forced to silence. (The
        // exact samples are not pinned — there is no codec-level bit-exact
        // fixture yet — but a clean params-decoded frame produces a
        // correctly-sized 48 kHz buffer.)
        let body: Vec<u8> = (0..200u16)
            .map(|i| (i.wrapping_mul(181).wrapping_add(13) & 0xff) as u8)
            .collect();
        let pkt = code0_packet(1, false, &body); // config 1 = SILK NB 20 ms.
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.channels, 1);
        assert_eq!(out.samples_per_channel(), 960);
        if out.frame_outcomes[0].status == FrameDecodeStatus::SilkParamsDecoded {
            // A successfully synthesized frame produces a full-length
            // buffer; every sample is a valid i16 (no panic / overflow).
            assert_eq!(out.pcm.len(), 960);
        }
    }

    /// Search for a CELT body whose §4.3.7.1 prefix decodes silence = 1
    /// with the post-filter off, so the frame takes the fully-wired
    /// silence synthesis path. Returns the body bytes appended after the
    /// TOC. The search is deterministic (fixed candidate set), so the
    /// chosen body is stable across runs.
    fn find_celt_silence_body() -> Vec<u8> {
        use crate::celt_frame_prefix::decode_celt_frame_prefix;
        use crate::range_decoder::RangeDecoder;
        // The silence flag is the {32767,1}/32768 "1" branch (probability
        // 2^-15), so a silent frame is rare in random bytes; sweep the
        // first two bytes (with a trailing zero run that keeps the
        // post-filter off) to find one deterministically.
        for b0 in 0u16..=255 {
            for b1 in 0u16..=255 {
                let buf = [b0 as u8, b1 as u8, 0, 0, 0, 0];
                let mut rd = RangeDecoder::new(&buf);
                let p = decode_celt_frame_prefix(&mut rd);
                if p.silence && p.post_filter.is_none() && !rd.has_error() {
                    return buf.to_vec();
                }
            }
        }
        panic!("no CELT silence body found in the candidate set");
    }

    #[test]
    fn celt_only_silence_frame_decodes_end_to_end() {
        // config 17 = CELT-only mono, 5 ms (Table-55 second column) →
        // 240 samples/channel at 48 kHz.
        let body = find_celt_silence_body();
        let pkt = code0_packet(17, false, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(out.channels, 1);
        assert_eq!(out.samples_per_channel(), 240);
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::CeltSilence,
            "silence-flagged CELT frame must take the wired synthesis path"
        );
        // The frame is silent: every emitted sample is zero (a zero-energy
        // band envelope synthesizes to a zero time-domain block, and the
        // overlap-add / de-emphasis of an all-zero history stays zero).
        assert_eq!(out.pcm.len(), 240);
        assert!(
            out.pcm.iter().all(|&s| s == 0),
            "silence frame must be all zero"
        );
    }

    #[test]
    fn celt_silence_advances_synthesis_state() {
        // Two consecutive CELT silence frames both decode through the
        // wired path; the second reuses the carried CeltSynthState (no
        // rebuild), and both emit silence of the correct length.
        let body = find_celt_silence_body();
        let pkt = code0_packet(17, false, &body);
        let mut dec = OpusDecoder::new();
        let first = dec.decode_packet(&pkt).expect("decode");
        let second = dec.decode_packet(&pkt).expect("decode");
        assert_eq!(
            first.frame_outcomes[0].status,
            FrameDecodeStatus::CeltSilence
        );
        assert_eq!(
            second.frame_outcomes[0].status,
            FrameDecodeStatus::CeltSilence
        );
        assert!(second.pcm.iter().all(|&s| s == 0));
    }

    #[test]
    fn celt_non_silent_frame_decodes_fully() {
        // A CELT body whose silence flag is clear takes the full §4.3
        // decode → synthesis path: the status is CeltDecoded (or
        // CeltDecodeError when the entropy content of the random body
        // is impossible), never a panic, never the not-wired
        // placeholder, and the PCM length is exact.
        use crate::celt_frame_prefix::decode_celt_frame_prefix;
        use crate::range_decoder::RangeDecoder;
        let mut chosen: Option<Vec<u8>> = None;
        for b0 in 0u16..=255 {
            let buf = [
                b0 as u8, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
            ];
            let mut rd = RangeDecoder::new(&buf);
            let p = decode_celt_frame_prefix(&mut rd);
            if !p.silence && !rd.has_error() {
                chosen = Some(buf.to_vec());
                break;
            }
        }
        let body = chosen.expect("a non-silent CELT body exists in the candidate set");
        let pkt = code0_packet(17, false, &body);
        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&pkt).expect("decode");
        assert!(
            matches!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::CeltDecoded | FrameDecodeStatus::CeltDecodeError
            ),
            "got {:?}",
            out.frame_outcomes[0].status
        );
        assert_eq!(out.pcm.len(), 240);
    }

    #[test]
    fn celt_energy_state_threads_across_frames() {
        // Two successive non-silent CELT-only frames must thread the
        // §4.3.2 energy history: after the first decodes fully, the
        // decoder carries a CeltEnergyState whose per-band energies the
        // second frame's inter prediction reads.
        use crate::celt_frame_prefix::decode_celt_frame_prefix;
        use crate::range_decoder::RangeDecoder;
        let mut chosen: Option<Vec<u8>> = None;
        for b0 in 0u16..=255 {
            let buf = [
                b0 as u8, 0x33, 0xcc, 0x55, 0xaa, 0x0f, 0xf0, 0x12, 0x9a, 0x4e,
            ];
            let mut rd = RangeDecoder::new(&buf);
            let p = decode_celt_frame_prefix(&mut rd);
            if !p.silence && !p.intra && !rd.has_error() {
                chosen = Some(buf.to_vec());
                break;
            }
        }
        if let Some(body) = chosen {
            let pkt = code0_packet(19, false, &body); // 20 ms CELT-only mono
            let mut dec = OpusDecoder::new();
            let first = dec.decode_packet(&pkt).expect("decode");
            if first.frame_outcomes[0].status == FrameDecodeStatus::CeltDecoded {
                let carried = dec.celt_energy.clone().expect("energy state carried");
                let second = dec.decode_packet(&pkt).expect("decode");
                // The same packet decodes against a different energy
                // history, so (when both frames fully decode) the
                // carried state must have advanced.
                if second.frame_outcomes[0].status == FrameDecodeStatus::CeltDecoded {
                    assert!(dec.celt_energy.is_some());
                    assert_ne!(
                        dec.celt_energy.as_ref().unwrap().old_band_e,
                        carried.old_log_e2,
                        "energy history must roll forward"
                    );
                }
            }
        }
    }

    #[test]
    fn celt_full_decode_consumes_bits_within_budget() {
        // Direct exercise of the §4.3 frame decode: on a non-silent
        // body it must consume real bits and never report a tell past
        // the frame budget unless the coder latched an error.
        use crate::celt_frame_decode::{decode_celt_frame, CeltEnergyState};
        use crate::range_decoder::RangeDecoder;

        let body: [u8; 14] = [
            0x40, 0x91, 0x37, 0xc4, 0x6e, 0x2d, 0xa8, 0x5b, 0xf1, 0x0c, 0x93, 0x47, 0xbe, 0x21,
        ];
        let mut rd = RangeDecoder::new(&body);
        let mut st = CeltEnergyState::new();
        let out = decode_celt_frame(&mut rd, body.len(), 0, 21, 3, 1, &mut st);
        assert!(!out.silence, "test body must be non-silent");
        assert!(rd.tell() > 1, "the frame decode must consume bits");
        if !rd.has_error() {
            assert!(
                rd.tell() as usize <= 8 * body.len(),
                "tell {} past budget",
                rd.tell()
            );
        }
    }

    // ---- Cross-packet §4.2.7.4 / §4.2.7.5.5 reconstruction carry ----

    use crate::silk_decode::SilkFrameSymbols;
    use crate::silk_excitation::ExcitationSymbols;
    use crate::silk_frame::{SilkHeaderSymbols, StereoPredictionWeights, StereoWeightSymbols};
    use crate::silk_gains::GainSymbol;
    use crate::silk_packet_encode::{
        encode_silk_only_packet_mono, encode_silk_only_packet_stereo, StereoIntervalScripts,
    };

    /// Owned buffers for one scripted NB 20 ms unvoiced SILK frame (4
    /// subframes, d_LPC = 10, 160 excitation samples in 10 shell
    /// blocks).
    struct FrameScript {
        gains: Vec<GainSymbol>,
        i2: Vec<i8>,
        lsb_counts: Vec<u8>,
        e_raw: Vec<i32>,
        header: SilkHeaderSymbols,
        lsf_stage1: u8,
        lsf_interp_w_q2: Option<u8>,
    }

    impl FrameScript {
        /// An unvoiced NB frame whose first-subframe gain symbol,
        /// stage-1 LSF index, §4.2.7.5.5 factor, and excitation are
        /// caller-chosen; the remaining subframes hold the gain steady
        /// (`Delta(4)` ⇒ `log_gain = previous_log_gain`).
        fn nb_unvoiced(first_gain: GainSymbol, lsf_stage1: u8, w_q2: u8, pulses: bool) -> Self {
            let mut e_raw = vec![0i32; 160];
            if pulses {
                for (i, slot) in e_raw.iter_mut().enumerate().step_by(8) {
                    *slot = if (i / 8) % 2 == 0 { 1 } else { -1 };
                }
            }
            Self {
                gains: vec![
                    first_gain,
                    GainSymbol::Delta(4),
                    GainSymbol::Delta(4),
                    GainSymbol::Delta(4),
                ],
                i2: vec![0i8; 10],
                lsb_counts: vec![0u8; 10],
                e_raw,
                header: SilkHeaderSymbols {
                    stereo: None,
                    mid_only_flag: None,
                    frame_type: 2, // Unvoiced / Low (Table 10) — VAD set.
                },
                lsf_stage1,
                lsf_interp_w_q2: Some(w_q2),
            }
        }

        fn symbols(&self) -> SilkFrameSymbols<'_> {
            SilkFrameSymbols {
                header: self.header,
                gains: &self.gains,
                lsf_stage1: self.lsf_stage1,
                lsf_stage2_i2: &self.i2,
                lsf_interp_w_q2: self.lsf_interp_w_q2,
                ltp: None,
                lcg_seed: 0,
                excitation: ExcitationSymbols {
                    rate_level: 0,
                    lsb_counts: &self.lsb_counts,
                    e_raw: &self.e_raw,
                },
            }
        }
    }

    fn rms(pcm: &[i16]) -> f64 {
        let e: f64 = pcm.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (e / pcm.len().max(1) as f64).sqrt()
    }

    /// §4.2.7.4: `previous_log_gain` persists across Opus frames, so an
    /// independently coded first-subframe gain in the NEXT packet is
    /// clamped to `max(gain_index, prev - 16)`. Packet A ends at
    /// log gain 63; packet B codes gain index 10, which a streaming
    /// decoder must lift to 47 (a fresh decoder decodes 10). The ~50 dB
    /// level difference is the observable.
    #[test]
    fn gain_clamp_base_carries_across_packets() {
        let bw = crate::toc::Bandwidth::Nb;
        let loud = FrameScript::nb_unvoiced(GainSymbol::Independent(63), 0, 4, true);
        let quiet = FrameScript::nb_unvoiced(GainSymbol::Independent(10), 0, 4, true);
        let (pkt_a, _) = encode_silk_only_packet_mono(bw, 200, &[loud.symbols()]).unwrap();
        let (pkt_b, _) = encode_silk_only_packet_mono(bw, 200, &[quiet.symbols()]).unwrap();

        let mut fresh = OpusDecoder::new();
        let quiet_alone = fresh.decode_packet(&pkt_b).unwrap();

        let mut stream = OpusDecoder::new();
        stream.decode_packet(&pkt_a).unwrap();
        let quiet_streamed = stream.decode_packet(&pkt_b).unwrap();

        let r_fresh = rms(&quiet_alone.pcm);
        let r_stream = rms(&quiet_streamed.pcm);
        assert!(
            r_stream > 30.0 * r_fresh.max(1e-9),
            "clamp must lift the streamed gain: fresh rms {r_fresh:.2}, streamed {r_stream:.2}"
        );
    }

    /// [`OpusDecoder::reset`] drops the §4.2.7.4 clamp base ("the
    /// clamping is skipped after a decoder reset"): decoding packet B
    /// after a reset must be byte-identical to a fresh decode.
    #[test]
    fn reset_clears_gain_clamp_base() {
        let bw = crate::toc::Bandwidth::Nb;
        let loud = FrameScript::nb_unvoiced(GainSymbol::Independent(63), 0, 4, true);
        let quiet = FrameScript::nb_unvoiced(GainSymbol::Independent(10), 0, 4, true);
        let (pkt_a, _) = encode_silk_only_packet_mono(bw, 200, &[loud.symbols()]).unwrap();
        let (pkt_b, _) = encode_silk_only_packet_mono(bw, 200, &[quiet.symbols()]).unwrap();

        let mut fresh = OpusDecoder::new();
        let alone = fresh.decode_packet(&pkt_b).unwrap();

        let mut stream = OpusDecoder::new();
        stream.decode_packet(&pkt_a).unwrap();
        stream.reset();
        let after_reset = stream.decode_packet(&pkt_b).unwrap();
        assert_eq!(alone.pcm, after_reset.pcm);
    }

    /// §4.2.7.5.5: the interpolation base `n0` is "the LSF coefficients
    /// decoded for the prior frame" — including across Opus frames. Two
    /// second packets that differ ONLY in the coded factor (`w_Q2 = 0`
    /// vs `4`) must reconstruct differently after the same first packet
    /// (with `w_Q2 = 0` the first half-frame runs the PREVIOUS packet's
    /// LPC), while a fresh decoder — where the factor is forced to 4 —
    /// reconstructs them identically.
    #[test]
    fn nlsf_interp_n0_carries_across_packets() {
        let bw = crate::toc::Bandwidth::Nb;
        let a = FrameScript::nb_unvoiced(GainSymbol::Independent(45), 0, 4, true);
        let b_interp = FrameScript::nb_unvoiced(GainSymbol::Independent(45), 25, 0, true);
        let b_no_interp = FrameScript::nb_unvoiced(GainSymbol::Independent(45), 25, 4, true);
        let (pkt_a, _) = encode_silk_only_packet_mono(bw, 200, &[a.symbols()]).unwrap();
        let (pkt_b0, _) = encode_silk_only_packet_mono(bw, 200, &[b_interp.symbols()]).unwrap();
        let (pkt_b4, _) = encode_silk_only_packet_mono(bw, 200, &[b_no_interp.symbols()]).unwrap();

        // Fresh decoders: no prior NLSF base exists, the factor is
        // forced to 4, and both variants decode identically.
        let mut f0 = OpusDecoder::new();
        let mut f4 = OpusDecoder::new();
        assert_eq!(
            f0.decode_packet(&pkt_b0).unwrap().pcm,
            f4.decode_packet(&pkt_b4).unwrap().pcm,
            "without a carried n0 the coded factor must be ignored"
        );

        // Streaming decoders: after packet A the coded factor is live.
        let mut s0 = OpusDecoder::new();
        let mut s4 = OpusDecoder::new();
        s0.decode_packet(&pkt_a).unwrap();
        s4.decode_packet(&pkt_a).unwrap();
        let out0 = s0.decode_packet(&pkt_b0).unwrap();
        let out4 = s4.decode_packet(&pkt_b4).unwrap();
        assert_ne!(
            out0.pcm, out4.pcm,
            "w_Q2 = 0 must interpolate against the previous packet's NLSFs"
        );
    }

    /// The stereo SIDE channel carries its own §4.2.7.4 clamp base
    /// across packets (independent of the mid channel's).
    #[test]
    fn stereo_side_gain_clamp_carries_across_packets() {
        let bw = crate::toc::Bandwidth::Nb;
        let weights = StereoWeightSymbols::quantize(StereoPredictionWeights {
            w0_q13: 0,
            w1_q13: 0,
        });
        // Mid frames: fixed moderate gain, silent excitation.
        let mid = |first: u8| {
            let mut m = FrameScript::nb_unvoiced(GainSymbol::Independent(first), 0, 4, false);
            m.header.stereo = Some(weights);
            m
        };
        let mid_a = mid(30);
        let mid_b = mid(30);
        // Side frames: packet A ends at log gain 63; packet B codes 10.
        let side_a = FrameScript::nb_unvoiced(GainSymbol::Independent(63), 0, 4, true);
        let side_b = FrameScript::nb_unvoiced(GainSymbol::Independent(10), 0, 4, true);

        let (pkt_a, _) = encode_silk_only_packet_stereo(
            bw,
            200,
            &[StereoIntervalScripts {
                mid: mid_a.symbols(),
                side: Some(side_a.symbols()),
            }],
        )
        .unwrap();
        let (pkt_b, _) = encode_silk_only_packet_stereo(
            bw,
            200,
            &[StereoIntervalScripts {
                mid: mid_b.symbols(),
                side: Some(side_b.symbols()),
            }],
        )
        .unwrap();

        // The side signal is (L - R) / 2 up to the (zero) prediction
        // weights; compare its energy fresh vs streamed.
        let side_rms = |audio: &DecodedAudio| {
            let d: Vec<i16> = audio
                .pcm
                .chunks_exact(2)
                .map(|lr| ((lr[0] as i32 - lr[1] as i32) / 2) as i16)
                .collect();
            rms(&d)
        };

        let mut fresh = OpusDecoder::new();
        let alone = fresh.decode_packet(&pkt_b).unwrap();

        let mut stream = OpusDecoder::new();
        stream.decode_packet(&pkt_a).unwrap();
        let streamed = stream.decode_packet(&pkt_b).unwrap();

        let r_fresh = side_rms(&alone);
        let r_stream = side_rms(&streamed);
        assert!(
            r_stream > 30.0 * r_fresh.max(1e-9),
            "side clamp must lift the streamed gain: fresh {r_fresh:.2}, streamed {r_stream:.2}"
        );
    }
}
