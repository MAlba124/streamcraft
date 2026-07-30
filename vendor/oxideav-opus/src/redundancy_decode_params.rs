//! Redundant-CELT-frame decode parameters and cross-lap placement
//! (RFC 6716 §4.5.1.4 "Decoding the Redundancy", pp. 126–127).
//!
//! Round 26 (`celt_redundancy`) decoded the §4.5.1.1–§4.5.1.3
//! redundancy *side information* in the tail of every SILK-only or
//! Hybrid Opus frame: whether the §4.5.1 redundancy flag was on,
//! which end of the Opus frame the redundant CELT frame sits at, and
//! how many bytes it occupies. Round 27 (`mode_transition_reset`)
//! decided which sub-decoders to reset and where the CELT reset is
//! placed relative to the redundant frame. This module handles the
//! step in between: turning the §4.5.1 boundary metadata into the
//! input the §4.3 CELT decoder needs to actually decode the
//! redundant frame, plus the §4.5.1.4 *cross-lap placement* that
//! tells the caller how the redundant CELT output and the
//! SILK/Hybrid output are spliced together at the 5 ms boundary.
//!
//! The §4.5.1.4 prose has two halves.
//!
//! ## Half 1 — redundant-frame parameters
//!
//! *"The redundant frame is decoded like any other CELT-only frame,
//! with the exception that it does not contain a TOC byte. The frame
//! size is fixed at 5 ms, the channel count is set to that of the
//! current frame, and the audio bandwidth is also set to that of the
//! current frame, with the exception that for MB SILK frames, it is
//! set to WB."*
//!
//! Four normative facts:
//!
//! 1. **No TOC byte.** The §3.1 TOC parse is skipped; the §4.3 CELT
//!    decoder is started directly on the redundant bytes.
//! 2. **Frame size fixed at 5 ms.** That's 50 tenths of a millisecond
//!    in the crate's `frame_size_tenths_ms` convention.
//! 3. **Channel count inherited.** Mono carrier → mono redundant
//!    CELT frame; stereo carrier → stereo redundant CELT frame.
//! 4. **Bandwidth inherited, with the MB SILK → WB override.**
//!    Hybrid carriers (SWB / FB) pass their bandwidth through
//!    untouched; SILK-only carriers pass NB and WB through
//!    untouched but bump MB up to WB (because the §4.3 CELT layer
//!    does not support the MB bandwidth).
//!
//! Note that the §4.5.1 redundancy flag is never decoded for a
//! CELT-only Opus frame (§4.5.1 only attaches to SILK-only or
//! Hybrid frames per `celt_redundancy::decode_redundancy`), so the
//! "carrier" of a redundant frame is always SILK-only or Hybrid by
//! construction. The reverse direction — a redundant CELT frame
//! *next to* a CELT-only neighbour — happens via the §4.5.3 figure
//! transition `C -> R & ;S -> S` where the carrier of `R` is the
//! first `S`, not the preceding `C`.
//!
//! ## Half 2 — cross-lap placement
//!
//! *"If the redundancy belongs at the beginning (in a CELT-only to
//! SILK-only or Hybrid transition), the final reconstructed output
//! uses the first 2.5 ms of audio output by the decoder for the
//! redundant frame as is, discarding the corresponding output from
//! the SILK-only or Hybrid portion of the frame. The remaining
//! 2.5 ms is cross-lapped with the decoded SILK/Hybrid signal using
//! the CELT's power-complementary MDCT window to ensure a smooth
//! transition."*
//!
//! *"If the redundancy belongs at the end (in a SILK-only or Hybrid
//! to CELT-only transition), only the second half (2.5 ms) of the
//! audio output by the decoder for the redundant frame is used. In
//! that case, the second half of the redundant frame is cross-lapped
//! with the end of the SILK/Hybrid signal, again using CELT's
//! power-complementary MDCT window to ensure a smooth transition."*
//!
//! Two normative cases:
//!
//! * **`Beginning`** (CELT → SILK/Hybrid carrier). The carrier is
//!   the post-transition SILK/Hybrid Opus frame. The redundant CELT
//!   frame's first 2.5 ms replace the carrier's leading 2.5 ms; the
//!   redundant frame's second 2.5 ms cross-lap with the SILK/Hybrid
//!   signal across the 2.5–5.0 ms region of the Opus frame.
//! * **`End`** (SILK/Hybrid → CELT carrier). The carrier is the
//!   pre-transition SILK/Hybrid Opus frame. Only the redundant CELT
//!   frame's second 2.5 ms are used; that half cross-laps with the
//!   end of the SILK/Hybrid signal. The redundant frame's first
//!   2.5 ms are discarded.
//!
//! Both cases cross-lap exactly 2.5 ms of redundant output against
//! 2.5 ms of SILK/Hybrid output, using the §4.3.7 power-complementary
//! MDCT window. The actual windowed mix is the §4.3.7 inverse-MDCT
//! stage, which is gated on §4.3.2 / §4.3.3 / §4.3.4 (all still
//! deferred). What this module owns is the placement metadata —
//! WHERE the 2.5 ms cross-lap region sits inside the carrier's
//! sample buffer and WHICH 2.5 ms of the redundant CELT output
//! feeds it.
//!
//! ## Module surface
//!
//! * [`REDUNDANT_FRAME_TENTHS_MS`] — the §4.5.1.4 "frame size is
//!   fixed at 5 ms" constant, expressed as `50` in the crate-wide
//!   tenths-of-a-millisecond convention.
//! * [`REDUNDANT_CROSS_LAP_TENTHS_MS`] — half of the redundant
//!   frame's duration (`25` = 2.5 ms), the size of the cross-lap
//!   region in both cases.
//! * [`RedundantFrameParams`] — the §4.5.1.4 half-1 outcome:
//!   `(duration_tenths_ms, channels, bandwidth, position,
//!   size_bytes)`. The bandwidth field already has the MB → WB
//!   override applied.
//! * [`CrossLapPlacement`] — the §4.5.1.4 half-2 outcome:
//!   `{ FirstHalfAsIs, SecondHalfAsIs }`.
//! * [`redundant_frame_params`] — the driver entry point. Returns
//!   `None` when the §4.5.1 decision was `NotPresent` or `Invalid`,
//!   otherwise returns the populated parameters.
//! * [`apply_mb_to_wb_override`] — the §4.5.1.4 bandwidth-override
//!   helper exposed for cross-checking and re-use.
//!
//! ## Provenance
//!
//! Every constant, every conditional, the "fixed at 5 ms" duration,
//! the "channel count is set to that of the current frame" rule,
//! the MB → WB override, the `Beginning` / `End` placement
//! distinction, and the 2.5 ms cross-lap region size is transcribed
//! from RFC 6716 §4.5.1.4 in `docs/audio/opus/rfc6716-opus.txt`
//! (pp. 126–127). The non-normative §4.5.3 Figure 18 (p. 129) was
//! used solely as a cross-check that the four redundancy-bearing
//! transition rows ("SILK → SILK with Redundancy", "SILK → CELT
//! with Redundancy", "CELT → SILK with Redundancy", "CELT → Hybrid
//! with Redundancy", and the NB-or-MB-SILK / Hybrid variants of
//! these) reproduce the figure's `R` placement. No external library
//! source was consulted.

use crate::celt_redundancy::{RedundancyDecision, RedundancyPosition};
use crate::framing::OpusFrameRouting;
use crate::toc::{Bandwidth, ChannelMapping};

/// §4.5.1.4 normative redundant-CELT-frame duration: "the frame size
/// is fixed at 5 ms".
///
/// Expressed in the crate's `frame_size_tenths_ms` convention so it
/// fits alongside the §3.1 Table 2 frame sizes (25, 50, 100, 200,
/// 400, 600). `5 ms × 10 = 50`.
pub const REDUNDANT_FRAME_TENTHS_MS: u16 = 50;

/// Half of the redundant CELT frame's duration: `25` tenths-of-a-ms
/// (= 2.5 ms).
///
/// This is the size of the cross-lap region in both §4.5.1.4 cases —
/// the `Beginning` case uses 2.5 ms of redundant output as-is then
/// cross-laps 2.5 ms; the `End` case discards 2.5 ms of redundant
/// output then cross-laps the remaining 2.5 ms.
pub const REDUNDANT_CROSS_LAP_TENTHS_MS: u16 = REDUNDANT_FRAME_TENTHS_MS / 2;

/// §4.5.1.4 cross-lap placement for the redundant CELT frame.
///
/// Two normative cases. The §4.3.7 power-complementary MDCT window
/// is applied across the 2.5 ms cross-lap region; that operation
/// itself is part of the §4.3.7 inverse-MDCT stage and gated on
/// the §4.3.2 / §4.3.3 / §4.3.4 chain, all still deferred. This
/// enum carries only the placement metadata: WHICH 2.5 ms half of
/// the redundant CELT output feeds the splice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossLapPlacement {
    /// The redundant CELT frame's FIRST 2.5 ms are used as-is, then
    /// its SECOND 2.5 ms cross-lap with the SILK/Hybrid signal.
    ///
    /// This is the §4.5.1.4 case for [`RedundancyPosition::Beginning`]:
    /// a CELT-only → SILK-only / Hybrid transition. The redundant
    /// CELT frame's first 2.5 ms replace the leading 2.5 ms of the
    /// SILK/Hybrid output (which is discarded); the second 2.5 ms
    /// of the redundant CELT frame is windowed and overlap-added
    /// with the SILK/Hybrid output in the 2.5–5.0 ms region.
    FirstHalfAsIs,
    /// The redundant CELT frame's FIRST 2.5 ms are discarded; its
    /// SECOND 2.5 ms cross-lap with the SILK/Hybrid signal.
    ///
    /// This is the §4.5.1.4 case for [`RedundancyPosition::End`]:
    /// a SILK-only / Hybrid → CELT-only transition. Only the second
    /// 2.5 ms of redundant CELT output is used; it is windowed and
    /// overlap-added with the end of the SILK/Hybrid signal.
    SecondHalfAsIs,
}

impl CrossLapPlacement {
    /// Derive the §4.5.1.4 cross-lap placement from the §4.5.1.2
    /// position symbol.
    pub fn from_position(position: RedundancyPosition) -> Self {
        match position {
            // §4.5.1.4: "If the redundancy belongs at the beginning
            // (in a CELT-only to SILK-only or Hybrid transition),
            // the final reconstructed output uses the first 2.5 ms
            // … as is, … The remaining 2.5 ms is cross-lapped …".
            RedundancyPosition::Beginning => CrossLapPlacement::FirstHalfAsIs,
            // §4.5.1.4: "If the redundancy belongs at the end (in a
            // SILK-only or Hybrid to CELT-only transition), only the
            // second half (2.5 ms) … is used. … the second half …
            // is cross-lapped …".
            RedundancyPosition::End => CrossLapPlacement::SecondHalfAsIs,
        }
    }

    /// `true` iff this placement uses the redundant CELT frame's
    /// first 2.5 ms (as-is or cross-lapped).
    pub fn uses_first_half(self) -> bool {
        matches!(self, CrossLapPlacement::FirstHalfAsIs)
    }

    /// `true` iff this placement uses the redundant CELT frame's
    /// second 2.5 ms as-is (rather than cross-lapped).
    pub fn second_half_is_used_as_is(self) -> bool {
        // §4.5.1.4: the second-half-as-is wording is only reached
        // through the `End` case, where the second half cross-laps
        // with the END of the SILK/Hybrid signal — the second half
        // of the redundant CELT frame contributes via the cross-lap,
        // not "as is".
        //
        // Both §4.5.1.4 cases cross-lap exactly one 2.5 ms region:
        //   * FirstHalfAsIs — second 2.5 ms cross-laps;
        //   * SecondHalfAsIs — second 2.5 ms cross-laps.
        // No §4.5.1.4 case takes the second half "as is" without a
        // cross-lap. This accessor exists so callers can distinguish
        // the two as-is regions; it always returns `false` because
        // the second 2.5 ms is never used "as is".
        let _ = self;
        false
    }
}

/// Apply the §4.5.1.4 "MB SILK frames → WB" bandwidth override.
///
/// §4.5.1.4 normative text: *"the audio bandwidth is also set to
/// that of the current frame, with the exception that for MB SILK
/// frames, it is set to WB."*
///
/// The "MB SILK" carrier is precisely a SILK-only Opus frame whose
/// §3.1 Table 2 audio bandwidth is `MB`. Hybrid carriers always
/// reach the §4.3 CELT layer via the Hybrid bandwidth (SWB or FB),
/// so they bypass the override. NB / WB SILK carriers also bypass
/// (NB and WB are both bands the §4.3 CELT layer supports; only MB
/// is exclusive to the §4.2 SILK layer and needs the bump).
///
/// SWB / FB inputs are pass-through (they only arrive via a Hybrid
/// carrier, where the override does not apply).
///
/// Exposed as a free function so the rule is grep-able and so a
/// caller building parameters from a non-routing source (e.g. a
/// test harness) can apply the override directly.
pub fn apply_mb_to_wb_override(carrier_bandwidth: Bandwidth, is_silk_only: bool) -> Bandwidth {
    if is_silk_only && matches!(carrier_bandwidth, Bandwidth::Mb) {
        Bandwidth::Wb
    } else {
        carrier_bandwidth
    }
}

/// Parameters needed to decode one redundant CELT frame per
/// RFC 6716 §4.5.1.4.
///
/// Built from the carrier Opus frame's [`OpusFrameRouting`] and the
/// §4.5.1 decode decision via [`redundant_frame_params`]. The
/// `bandwidth` field already has the §4.5.1.4 MB → WB override
/// applied; the `duration_tenths_ms` field is always
/// [`REDUNDANT_FRAME_TENTHS_MS`] (= 50, "frame size is fixed at
/// 5 ms"). `position` and `size_bytes` are carried through from
/// the §4.5.1 decision so the caller has a single struct
/// representing "here's the redundant-CELT-frame call site".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RedundantFrameParams {
    /// Fixed redundant-frame duration: 5 ms = 50 tenths-of-a-ms.
    pub duration_tenths_ms: u16,
    /// Channel count inherited from the carrier Opus frame.
    pub channels: ChannelMapping,
    /// Audio bandwidth for the redundant CELT frame after the
    /// §4.5.1.4 MB → WB override is applied.
    pub bandwidth: Bandwidth,
    /// Position of the redundant CELT frame within the carrier
    /// Opus frame's byte buffer (Beginning vs. End).
    pub position: RedundancyPosition,
    /// Size in whole bytes of the redundant CELT frame in the
    /// carrier Opus frame's byte buffer.
    pub size_bytes: usize,
    /// §4.5.1.4 cross-lap placement decision derived from
    /// `position`. Bundled here so the caller does not re-derive
    /// it.
    pub cross_lap: CrossLapPlacement,
}

impl RedundantFrameParams {
    /// Cross-lap region duration in tenths-of-a-millisecond. Always
    /// [`REDUNDANT_CROSS_LAP_TENTHS_MS`] (= 25 = 2.5 ms) per the
    /// §4.5.1.4 "2.5 ms" wording.
    pub fn cross_lap_tenths_ms(&self) -> u16 {
        REDUNDANT_CROSS_LAP_TENTHS_MS
    }

    /// `true` iff the redundant CELT frame's first 2.5 ms are used
    /// directly (as-is, no cross-lap). Convenience over
    /// `self.cross_lap.uses_first_half()`.
    pub fn first_half_is_used_as_is(&self) -> bool {
        self.cross_lap.uses_first_half()
    }
}

/// Drive §4.5.1.4 parameter derivation.
///
/// Returns `Some(RedundantFrameParams)` iff the §4.5.1 decision was
/// [`RedundancyDecision::Present`]; otherwise `None`. The
/// [`RedundancyDecision::Invalid`] case (the §4.5.1.3 overflow) is
/// treated as "no redundant frame to decode" per the §4.5.1.3
/// "stop decoding and discard" recommendation.
///
/// The function never panics. It is a pure data lookup driven by
/// the §4.5.1 decision metadata plus the carrier routing's
/// `operating_mode`, `toc_bandwidth`, and `channels` fields. It
/// does NOT touch the range decoder.
pub fn redundant_frame_params(
    routing: &OpusFrameRouting,
    decision: RedundancyDecision,
) -> Option<RedundantFrameParams> {
    let (position, size_bytes) = match decision {
        RedundancyDecision::Present {
            position,
            size_bytes,
        } => (position, size_bytes),
        RedundancyDecision::NotPresent | RedundancyDecision::Invalid => return None,
    };

    // §4.5.1.4 channel count inheritance: "the channel count is set
    // to that of the current frame". `OpusFrameRouting::channels`
    // already encodes the carrier's `s` bit.
    let channels = routing.channels;

    // §4.5.1.4 bandwidth inheritance + MB → WB override. The
    // §4.5.1.4 prose qualifies the override to "MB SILK frames"
    // specifically — the SILK-only modes the MB bandwidth is
    // exclusively assigned to (Hybrid carriers run at SWB / FB and
    // never at MB, so the override never fires for them).
    let is_silk_only = matches!(
        routing.operating_mode,
        crate::framing::OperatingMode::SilkOnly
    );
    let bandwidth = apply_mb_to_wb_override(routing.toc_bandwidth, is_silk_only);

    let cross_lap = CrossLapPlacement::from_position(position);

    Some(RedundantFrameParams {
        duration_tenths_ms: REDUNDANT_FRAME_TENTHS_MS,
        channels,
        bandwidth,
        position,
        size_bytes,
        cross_lap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::OperatingMode;
    use crate::toc::{Bandwidth, ChannelMapping, FrameCountCode, Mode};

    /// Hand-build an `OpusFrameRouting` for testing without round-
    /// tripping through `OpusTocByte::from_byte` (which constrains
    /// the bandwidth × mode combinations to the §3.1 Table 2 grid).
    /// All four fields a §4.5.1.4 derivation reads are explicit.
    fn fake_routing(
        operating_mode: OperatingMode,
        toc_bandwidth: Bandwidth,
        channels: ChannelMapping,
    ) -> OpusFrameRouting {
        let silk_layer = matches!(
            operating_mode,
            OperatingMode::SilkOnly | OperatingMode::Hybrid
        );
        let celt_layer = matches!(
            operating_mode,
            OperatingMode::Hybrid | OperatingMode::CeltOnly
        );
        let silk_bandwidth = if silk_layer {
            Some(match operating_mode {
                OperatingMode::Hybrid => crate::framing::SilkBandwidth::Wb,
                OperatingMode::SilkOnly => match toc_bandwidth {
                    Bandwidth::Nb => crate::framing::SilkBandwidth::Nb,
                    Bandwidth::Mb => crate::framing::SilkBandwidth::Mb,
                    Bandwidth::Wb | Bandwidth::Swb | Bandwidth::Fb => {
                        crate::framing::SilkBandwidth::Wb
                    }
                },
                OperatingMode::CeltOnly => unreachable!(),
            })
        } else {
            None
        };
        OpusFrameRouting {
            operating_mode,
            toc_bandwidth,
            frame_size_tenths_ms: 200, // 20 ms carrier; irrelevant to §4.5.1.4
            channels,
            silk_layer,
            celt_layer,
            silk_bandwidth,
            silk_frames_per_channel: if silk_layer { Some(1) } else { None },
        }
    }

    /// §4.5.1.4 duration constant matches the crate's
    /// tenths-of-a-millisecond convention.
    #[test]
    fn redundant_frame_tenths_ms_is_50() {
        // 5 ms × 10 = 50 — the §4.5.1.4 "fixed at 5 ms" rule.
        assert_eq!(REDUNDANT_FRAME_TENTHS_MS, 50);
    }

    /// The cross-lap region is exactly half of the redundant frame.
    #[test]
    fn cross_lap_is_half_of_redundant_frame() {
        assert_eq!(REDUNDANT_CROSS_LAP_TENTHS_MS, 25);
        assert_eq!(REDUNDANT_CROSS_LAP_TENTHS_MS * 2, REDUNDANT_FRAME_TENTHS_MS);
    }

    /// `from_position` is the §4.5.1.4 position → placement map.
    #[test]
    fn cross_lap_from_position_total() {
        assert_eq!(
            CrossLapPlacement::from_position(RedundancyPosition::Beginning),
            CrossLapPlacement::FirstHalfAsIs
        );
        assert_eq!(
            CrossLapPlacement::from_position(RedundancyPosition::End),
            CrossLapPlacement::SecondHalfAsIs
        );
    }

    /// `uses_first_half` reflects which 2.5 ms region the cross-lap
    /// places "as is" (Beginning = first-half-as-is).
    #[test]
    fn uses_first_half_accessor() {
        assert!(CrossLapPlacement::FirstHalfAsIs.uses_first_half());
        assert!(!CrossLapPlacement::SecondHalfAsIs.uses_first_half());
    }

    /// Neither §4.5.1.4 case takes the second half "as is" — both
    /// place the second half into the 2.5 ms cross-lap region.
    #[test]
    fn second_half_is_never_used_as_is() {
        assert!(!CrossLapPlacement::FirstHalfAsIs.second_half_is_used_as_is());
        assert!(!CrossLapPlacement::SecondHalfAsIs.second_half_is_used_as_is());
    }

    /// MB → WB override fires for SILK-only / MB only.
    #[test]
    fn mb_to_wb_override_silk_only_mb() {
        assert_eq!(apply_mb_to_wb_override(Bandwidth::Mb, true), Bandwidth::Wb);
    }

    /// MB → WB override does NOT fire for Hybrid / MB (which never
    /// occurs in practice, but the override predicate is gated on
    /// is_silk_only either way).
    #[test]
    fn mb_to_wb_override_skipped_for_hybrid() {
        assert_eq!(apply_mb_to_wb_override(Bandwidth::Mb, false), Bandwidth::Mb);
    }

    /// Other bandwidths are pass-through under any carrier mode.
    #[test]
    fn mb_to_wb_override_passthrough_for_nb_wb_swb_fb() {
        for is_silk_only in [false, true] {
            for bw in [Bandwidth::Nb, Bandwidth::Wb, Bandwidth::Swb, Bandwidth::Fb] {
                assert_eq!(apply_mb_to_wb_override(bw, is_silk_only), bw);
            }
        }
    }

    /// `redundant_frame_params` returns `None` for `NotPresent`.
    #[test]
    fn params_none_for_not_present() {
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        assert!(redundant_frame_params(&routing, RedundancyDecision::NotPresent).is_none());
    }

    /// `redundant_frame_params` returns `None` for the §4.5.1.3
    /// overflow case — per §4.5.1.3 the decoder "stops and
    /// discards", so there's no redundant CELT frame to decode.
    #[test]
    fn params_none_for_invalid() {
        let routing = fake_routing(
            OperatingMode::Hybrid,
            Bandwidth::Swb,
            ChannelMapping::Stereo,
        );
        assert!(redundant_frame_params(&routing, RedundancyDecision::Invalid).is_none());
    }

    /// SILK-only / NB carrier passes NB through (no MB override).
    /// `Beginning` position routes to FirstHalfAsIs.
    #[test]
    fn silk_only_nb_beginning_passes_nb_through() {
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Nb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 8,
        };
        let params = redundant_frame_params(&routing, decision).expect("should be present");
        assert_eq!(params.duration_tenths_ms, REDUNDANT_FRAME_TENTHS_MS);
        assert_eq!(params.channels, ChannelMapping::Mono);
        assert_eq!(params.bandwidth, Bandwidth::Nb);
        assert_eq!(params.position, RedundancyPosition::Beginning);
        assert_eq!(params.size_bytes, 8);
        assert_eq!(params.cross_lap, CrossLapPlacement::FirstHalfAsIs);
        assert!(params.first_half_is_used_as_is());
        assert_eq!(params.cross_lap_tenths_ms(), 25);
    }

    /// SILK-only / MB carrier bumps to WB per §4.5.1.4 exception.
    #[test]
    fn silk_only_mb_bumps_to_wb() {
        let routing = fake_routing(
            OperatingMode::SilkOnly,
            Bandwidth::Mb,
            ChannelMapping::Stereo,
        );
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::End,
            size_bytes: 12,
        };
        let params = redundant_frame_params(&routing, decision).expect("should be present");
        assert_eq!(params.bandwidth, Bandwidth::Wb);
        assert_eq!(params.channels, ChannelMapping::Stereo);
        assert_eq!(params.position, RedundancyPosition::End);
        assert_eq!(params.size_bytes, 12);
        assert_eq!(params.cross_lap, CrossLapPlacement::SecondHalfAsIs);
        assert!(!params.first_half_is_used_as_is());
    }

    /// SILK-only / WB carrier passes WB through unchanged.
    #[test]
    fn silk_only_wb_passes_through() {
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 16,
        };
        let params = redundant_frame_params(&routing, decision).expect("should be present");
        assert_eq!(params.bandwidth, Bandwidth::Wb);
    }

    /// Hybrid / SWB carrier passes SWB through (override does not
    /// fire for Hybrid).
    #[test]
    fn hybrid_swb_passes_through() {
        let routing = fake_routing(
            OperatingMode::Hybrid,
            Bandwidth::Swb,
            ChannelMapping::Stereo,
        );
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 4,
        };
        let params = redundant_frame_params(&routing, decision).expect("should be present");
        assert_eq!(params.bandwidth, Bandwidth::Swb);
        assert_eq!(params.channels, ChannelMapping::Stereo);
    }

    /// Hybrid / FB carrier passes FB through.
    #[test]
    fn hybrid_fb_passes_through() {
        let routing = fake_routing(OperatingMode::Hybrid, Bandwidth::Fb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::End,
            size_bytes: 7,
        };
        let params = redundant_frame_params(&routing, decision).expect("should be present");
        assert_eq!(params.bandwidth, Bandwidth::Fb);
    }

    /// Channel count is faithfully inherited under all (mode,
    /// bandwidth, channels) combinations.
    #[test]
    fn channel_count_inherited_under_all_carriers() {
        let cases: &[(OperatingMode, Bandwidth)] = &[
            (OperatingMode::SilkOnly, Bandwidth::Nb),
            (OperatingMode::SilkOnly, Bandwidth::Mb),
            (OperatingMode::SilkOnly, Bandwidth::Wb),
            (OperatingMode::Hybrid, Bandwidth::Swb),
            (OperatingMode::Hybrid, Bandwidth::Fb),
        ];
        for &(mode, bw) in cases {
            for &chan in &[ChannelMapping::Mono, ChannelMapping::Stereo] {
                let routing = fake_routing(mode, bw, chan);
                let decision = RedundancyDecision::Present {
                    position: RedundancyPosition::Beginning,
                    size_bytes: 3,
                };
                let params =
                    redundant_frame_params(&routing, decision).expect("present should populate");
                assert_eq!(params.channels, chan, "{mode:?} {bw:?} {chan:?}");
            }
        }
    }

    /// Duration is always 50 tenths regardless of the carrier's
    /// frame size.
    #[test]
    fn duration_is_fixed_at_50_tenths() {
        let mut routing =
            fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        for carrier_dur in [100u16, 200, 400, 600] {
            routing.frame_size_tenths_ms = carrier_dur;
            let decision = RedundancyDecision::Present {
                position: RedundancyPosition::End,
                size_bytes: 5,
            };
            let params = redundant_frame_params(&routing, decision).expect("present");
            assert_eq!(
                params.duration_tenths_ms, REDUNDANT_FRAME_TENTHS_MS,
                "carrier {carrier_dur} should still produce 5 ms redundant"
            );
        }
    }

    /// §4.5.3 Figure 18 cross-check: "CELT to SILK with Redundancy"
    /// — carrier is the first SILK-only frame, position Beginning.
    /// The figure marks the redundant CELT frame's first 2.5 ms as
    /// the "as-is" insertion, with the second 2.5 ms windowed
    /// against the SILK leading edge. Our placement matches.
    #[test]
    fn figure18_celt_to_silk_with_redundancy() {
        // Carrier is the post-transition SILK-only frame, e.g. WB
        // SILK; position is Beginning.
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 10,
        };
        let params = redundant_frame_params(&routing, decision).expect("present");
        assert_eq!(params.cross_lap, CrossLapPlacement::FirstHalfAsIs);
        assert!(params.first_half_is_used_as_is());
        assert_eq!(params.bandwidth, Bandwidth::Wb);
    }

    /// §4.5.3 Figure 18 cross-check: "CELT to Hybrid with Redundancy"
    /// — same Beginning placement, Hybrid carrier preserves
    /// SWB / FB through the bandwidth override.
    #[test]
    fn figure18_celt_to_hybrid_with_redundancy() {
        let routing = fake_routing(OperatingMode::Hybrid, Bandwidth::Fb, ChannelMapping::Stereo);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 14,
        };
        let params = redundant_frame_params(&routing, decision).expect("present");
        assert_eq!(params.cross_lap, CrossLapPlacement::FirstHalfAsIs);
        assert_eq!(params.bandwidth, Bandwidth::Fb);
        assert_eq!(params.channels, ChannelMapping::Stereo);
    }

    /// §4.5.3 Figure 18 cross-check: "SILK to CELT with Redundancy"
    /// — carrier is the last SILK-only frame, position End.
    /// Redundant frame's second 2.5 ms cross-laps with the SILK
    /// trailing edge.
    #[test]
    fn figure18_silk_to_celt_with_redundancy() {
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::End,
            size_bytes: 6,
        };
        let params = redundant_frame_params(&routing, decision).expect("present");
        assert_eq!(params.cross_lap, CrossLapPlacement::SecondHalfAsIs);
        assert!(!params.first_half_is_used_as_is());
    }

    /// §4.5.3 Figure 18 cross-check: "Hybrid to CELT with Redundancy"
    /// — End placement, SWB carrier; bandwidth pass-through.
    #[test]
    fn figure18_hybrid_to_celt_with_redundancy() {
        let routing = fake_routing(OperatingMode::Hybrid, Bandwidth::Swb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::End,
            size_bytes: 9,
        };
        let params = redundant_frame_params(&routing, decision).expect("present");
        assert_eq!(params.cross_lap, CrossLapPlacement::SecondHalfAsIs);
        assert_eq!(params.bandwidth, Bandwidth::Swb);
    }

    /// §4.5.3 Figure 18: "SILK to SILK with Redundancy" — both
    /// halves of the transition happen via a single intermediate
    /// SILK frame that carries the redundant CELT frame. The
    /// carrier here is the SILK frame; either Beginning or End is
    /// figure-legal depending on which side of the transition is
    /// being illustrated. We exercise both with MB → WB applied.
    #[test]
    fn figure18_silk_to_silk_mb_carrier_bumps_to_wb() {
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Mb, ChannelMapping::Mono);
        for pos in [RedundancyPosition::Beginning, RedundancyPosition::End] {
            let decision = RedundancyDecision::Present {
                position: pos,
                size_bytes: 5,
            };
            let params = redundant_frame_params(&routing, decision).expect("present");
            assert_eq!(params.bandwidth, Bandwidth::Wb, "MB→WB override on {pos:?}");
        }
    }

    /// `frame_count_code` and other carrier-only routing fields must
    /// not influence the §4.5.1.4 derivation.
    #[test]
    fn carrier_frame_count_code_irrelevant() {
        let _ = FrameCountCode::One; // touch the import
        let _ = Mode::SilkOnly; // touch the import
        let routing = fake_routing(OperatingMode::SilkOnly, Bandwidth::Wb, ChannelMapping::Mono);
        let decision = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 11,
        };
        let params_a = redundant_frame_params(&routing, decision).expect("present");
        // Mutate an irrelevant carrier field; result must be identical.
        let mut routing2 = routing;
        routing2.silk_frames_per_channel = Some(3);
        let params_b = redundant_frame_params(&routing2, decision).expect("present");
        assert_eq!(params_a, params_b);
    }

    /// Size-bytes is faithfully forwarded from the §4.5.1 decision.
    #[test]
    fn size_bytes_forwarded() {
        let routing = fake_routing(OperatingMode::Hybrid, Bandwidth::Swb, ChannelMapping::Mono);
        for &sz in &[2usize, 3, 7, 16, 64, 257, 1000] {
            let decision = RedundancyDecision::Present {
                position: RedundancyPosition::End,
                size_bytes: sz,
            };
            let params = redundant_frame_params(&routing, decision).expect("present");
            assert_eq!(params.size_bytes, sz);
        }
    }

    /// Total function: every (mode × bandwidth × channels × position)
    /// combination yields a valid `RedundantFrameParams` whose
    /// bandwidth is one of NB/WB/SWB/FB after the MB→WB override
    /// (never MB).
    #[test]
    fn total_function_sweep_no_mb_in_output() {
        let modes = [
            OperatingMode::SilkOnly,
            OperatingMode::Hybrid,
            // CELT-only never carries §4.5.1 side info, so the
            // §4.5.1 decision is always NotPresent and params is None.
        ];
        let bws_silk = [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb];
        let bws_hyb = [Bandwidth::Swb, Bandwidth::Fb];
        let chans = [ChannelMapping::Mono, ChannelMapping::Stereo];
        let positions = [RedundancyPosition::Beginning, RedundancyPosition::End];

        for mode in modes {
            let bw_list: &[Bandwidth] = if matches!(mode, OperatingMode::SilkOnly) {
                &bws_silk
            } else {
                &bws_hyb
            };
            for &bw in bw_list {
                for &chan in &chans {
                    for &pos in &positions {
                        let routing = fake_routing(mode, bw, chan);
                        let decision = RedundancyDecision::Present {
                            position: pos,
                            size_bytes: 4,
                        };
                        let params = redundant_frame_params(&routing, decision)
                            .expect("Present should always yield params");
                        assert_ne!(
                            params.bandwidth,
                            Bandwidth::Mb,
                            "{mode:?}/{bw:?}/{chan:?}/{pos:?} must not emit MB"
                        );
                        assert_eq!(
                            params.duration_tenths_ms, REDUNDANT_FRAME_TENTHS_MS,
                            "{mode:?}/{bw:?}/{chan:?}/{pos:?}"
                        );
                    }
                }
            }
        }
    }
}
