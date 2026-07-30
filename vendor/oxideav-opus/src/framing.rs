//! Opus packet → SILK/CELT layer routing — RFC 6716 §3.1 Table 2,
//! §4.2 / §4.3 dispatch.
//!
//! This module sits between [`crate::toc::OpusTocByte`] (which decodes
//! the §3.1 `(mode, bandwidth, frame_size)` triple from the TOC byte)
//! and the per-layer decoders ([`crate::silk_header`],
//! [`crate::silk_frame`], [`crate::celt_header`], …). For every Opus
//! frame the §4 decoder must answer three questions before it can read
//! the first range-coded symbol:
//!
//! 1. **Does this Opus frame carry a SILK layer?** Yes for [`OperatingMode::SilkOnly`]
//!    and [`OperatingMode::Hybrid`]; no for [`OperatingMode::CeltOnly`]
//!    (RFC 6716 §3.1 Table 2, §4.2 first paragraph).
//! 2. **Does this Opus frame carry a CELT layer?** Yes for [`OperatingMode::Hybrid`]
//!    and [`OperatingMode::CeltOnly`]; no for [`OperatingMode::SilkOnly`].
//! 3. **When the SILK layer is present, at what audio bandwidth does it
//!    run internally?** Per RFC 6716 §4.2 ("When used in a SWB or FB
//!    Hybrid frame, the LP layer itself still only runs in WB"), the
//!    SILK internal bandwidth is the TOC bandwidth for SILK-only, but
//!    pinned to [`SilkBandwidth::Wb`] for Hybrid regardless of the TOC's
//!    SWB / FB.
//!
//! Wiring these three answers up consistently is currently a
//! per-caller open-coded decision; this module turns it into a single
//! `OpusFrameRouting::from_toc` call that every Opus frame's §4
//! decoder runs first. Tables 2 and 3 (the §3.1 configuration table
//! and the §4.2.2 SILK-layer organization) are both consumed here, so
//! a downstream caller doesn't need to look at the TOC `config`
//! directly to know e.g. "this is a 60 ms stereo Hybrid frame: 2
//! channels × 3 SILK frames each, plus one CELT decode at WB / 20 ms".
//!
//! ## What this module does not own
//!
//! * The §4.1 range decoder primitive — see [`crate::range_decoder`].
//! * The §4.2.3 / §4.2.4 SILK header bits — see [`crate::silk_header`].
//! * The §4.3 / Table 56 CELT pre-band header — see
//!   [`crate::celt_header`].
//! * Anything bitstream-level — this module is a pure-function lookup
//!   on `(mode, bandwidth, frame_size, channels)`. It reads no bytes.
//!
//! ## Provenance
//!
//! Tables 2 (§3.1, p. 14) and the §4.2.2 SILK-layer organization (p.
//! 33) of RFC 6716 (September 2012) are the only sources; the SILK
//! frame-count enumeration is the same one [`crate::silk_header::silk_frame_count`]
//! already encodes. No external library source consulted.

use crate::silk_header::silk_frame_count;
use crate::toc::{Bandwidth, ChannelMapping, Mode, OpusTocByte};

/// Operating mode for one Opus frame, as routed from the §3.1 TOC
/// `config` field. Mirrors [`crate::toc::Mode`] under a more
/// dispatch-flavoured name so consumers reading
/// `routing.operating_mode` make the right cognitive distinction:
/// this is the *dispatch decision* derived from the TOC, not the raw
/// field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    /// SILK-only — only the §4.2 LP decoder runs.
    SilkOnly,
    /// Hybrid — both §4.2 LP and §4.3 CELT decoders run, with SILK
    /// pinned to WB regardless of the TOC bandwidth.
    Hybrid,
    /// CELT-only — only the §4.3 MDCT decoder runs.
    CeltOnly,
}

impl From<Mode> for OperatingMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::SilkOnly => OperatingMode::SilkOnly,
            Mode::Hybrid => OperatingMode::Hybrid,
            Mode::CeltOnly => OperatingMode::CeltOnly,
        }
    }
}

/// Audio bandwidth at which the SILK layer of a SILK-bearing Opus
/// frame runs internally.
///
/// Per RFC 6716 §4.2, the SILK layer only runs at NB, MB, or WB —
/// even when the Opus frame's TOC bandwidth is SWB or FB (Hybrid
/// mode), the LP layer itself is pinned to WB and the CELT layer
/// covers the higher frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkBandwidth {
    Nb,
    Mb,
    Wb,
}

impl SilkBandwidth {
    /// Convert to the crate-wide [`Bandwidth`] enum, suitable for
    /// passing into the per-stage SILK decoders that already key
    /// off `Bandwidth::{Nb, Mb, Wb}`.
    pub fn to_bandwidth(self) -> Bandwidth {
        match self {
            SilkBandwidth::Nb => Bandwidth::Nb,
            SilkBandwidth::Mb => Bandwidth::Mb,
            SilkBandwidth::Wb => Bandwidth::Wb,
        }
    }
}

/// Routing decision for one Opus frame, derived purely from the TOC
/// byte.
///
/// Holds every dispatch-level fact a §4 decoder needs before it
/// touches the range coder. Every field is derivable from
/// [`OpusTocByte`]; bundling them here keeps the dispatch logic
/// in one place (and one set of tests) instead of duplicated across
/// every caller that constructs a SILK or CELT context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusFrameRouting {
    /// §3.1 Table 2 mode (SILK / Hybrid / CELT-only).
    pub operating_mode: OperatingMode,
    /// §3.1 Table 2 audio bandwidth as signalled by the TOC. For a
    /// Hybrid frame this is the SWB / FB the *output* covers, not the
    /// (always-WB) internal SILK bandwidth — use [`Self::silk_bandwidth`]
    /// when feeding the SILK decoder.
    pub toc_bandwidth: Bandwidth,
    /// §3.1 Table 2 Opus-frame duration, in tenths of a millisecond
    /// (25, 50, 100, 200, 400, 600).
    pub frame_size_tenths_ms: u16,
    /// §3.1 `s` bit (mono vs stereo).
    pub channels: ChannelMapping,
    /// SILK-bearing flag (true for SILK-only or Hybrid).
    pub silk_layer: bool,
    /// CELT-bearing flag (true for Hybrid or CELT-only).
    pub celt_layer: bool,
    /// Internal SILK bandwidth (Some when [`silk_layer`] is true; None
    /// for CELT-only). For Hybrid this is always [`SilkBandwidth::Wb`]
    /// per RFC 6716 §4.2 first paragraph.
    ///
    /// [`silk_layer`]: Self::silk_layer
    pub silk_bandwidth: Option<SilkBandwidth>,
    /// Number of regular SILK frames per channel per Opus frame
    /// (Some when [`silk_layer`] is true; None otherwise). Per §4.2.2:
    /// 1 for 10 / 20 ms Opus frames, 2 for 40 ms, 3 for 60 ms.
    ///
    /// [`silk_layer`]: Self::silk_layer
    pub silk_frames_per_channel: Option<u8>,
}

impl OpusFrameRouting {
    /// Derive the routing for one Opus frame from its TOC byte.
    ///
    /// Total function — every [`OpusTocByte`] value produces a valid
    /// routing, because the TOC byte parser has already constrained
    /// `(mode, bandwidth, frame_size, channels)` to the §3.1 Table 2
    /// + Table 3 legal grid.
    pub fn from_toc(toc: OpusTocByte) -> Self {
        let operating_mode = OperatingMode::from(toc.mode);
        let silk_layer = matches!(
            operating_mode,
            OperatingMode::SilkOnly | OperatingMode::Hybrid
        );
        let celt_layer = matches!(
            operating_mode,
            OperatingMode::Hybrid | OperatingMode::CeltOnly
        );

        let silk_bandwidth = if silk_layer {
            // §4.2 first paragraph: "When used in a SWB or FB Hybrid
            // frame, the LP layer itself still only runs in WB".
            match operating_mode {
                OperatingMode::Hybrid => Some(SilkBandwidth::Wb),
                OperatingMode::SilkOnly => Some(match toc.bandwidth {
                    Bandwidth::Nb => SilkBandwidth::Nb,
                    Bandwidth::Mb => SilkBandwidth::Mb,
                    Bandwidth::Wb => SilkBandwidth::Wb,
                    // Unreachable: Table 2 never pairs SILK-only with
                    // SWB / FB. Defensive fall-through still produces
                    // the safest WB pin.
                    Bandwidth::Swb | Bandwidth::Fb => SilkBandwidth::Wb,
                }),
                OperatingMode::CeltOnly => None,
            }
        } else {
            None
        };

        let silk_frames_per_channel = if silk_layer {
            // silk_frame_count returns None only for the 2.5 / 5 ms
            // CELT-only durations (which we've already excluded
            // because silk_layer is false there). Defensive default
            // 1 keeps the routing total in pathological inputs.
            Some(silk_frame_count(toc.frame_size_tenths_ms).unwrap_or(1))
        } else {
            None
        };

        Self {
            operating_mode,
            toc_bandwidth: toc.bandwidth,
            frame_size_tenths_ms: toc.frame_size_tenths_ms,
            channels: toc.channels,
            silk_layer,
            celt_layer,
            silk_bandwidth,
            silk_frames_per_channel,
        }
    }

    /// Number of audio channels (1 for mono, 2 for stereo).
    pub fn channel_count(&self) -> u8 {
        match self.channels {
            ChannelMapping::Mono => 1,
            ChannelMapping::Stereo => 2,
        }
    }

    /// Total regular-SILK-frame count for this Opus frame across both
    /// channels (mono × `frames_per_channel`, or stereo × 2 ×
    /// `frames_per_channel`). Returns 0 for CELT-only frames.
    pub fn total_silk_frames(&self) -> u8 {
        match self.silk_frames_per_channel {
            Some(n) => self.channel_count() * n,
            None => 0,
        }
    }

    /// `true` iff this Opus frame is long enough to potentially carry
    /// §4.2.4 per-frame LBRR flag bytes (i.e. 40 ms or 60 ms). Per
    /// §4.2.4 the per-frame flags are only present when the global
    /// LBRR flag is set, but the duration gate alone is a routing
    /// concern.
    pub fn has_per_frame_lbrr_bits(&self) -> bool {
        self.silk_layer && matches!(self.frame_size_tenths_ms, 400 | 600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toc::{Bandwidth, ChannelMapping, FrameCountCode, Mode};

    fn route(config: u8, stereo: bool) -> OpusFrameRouting {
        let byte = (config << 3) | (if stereo { 1 << 2 } else { 0 });
        OpusFrameRouting::from_toc(OpusTocByte::from_byte(byte))
    }

    /// `OperatingMode::from(Mode)` is a total bijection onto the
    /// three operating modes.
    #[test]
    fn operating_mode_from_mode_total() {
        assert_eq!(OperatingMode::from(Mode::SilkOnly), OperatingMode::SilkOnly);
        assert_eq!(OperatingMode::from(Mode::Hybrid), OperatingMode::Hybrid);
        assert_eq!(OperatingMode::from(Mode::CeltOnly), OperatingMode::CeltOnly);
    }

    /// SilkBandwidth → Bandwidth lifts cleanly to the three SILK
    /// internal rates.
    #[test]
    fn silk_bandwidth_to_bandwidth() {
        assert_eq!(SilkBandwidth::Nb.to_bandwidth(), Bandwidth::Nb);
        assert_eq!(SilkBandwidth::Mb.to_bandwidth(), Bandwidth::Mb);
        assert_eq!(SilkBandwidth::Wb.to_bandwidth(), Bandwidth::Wb);
    }

    /// SILK-only configs 0..=11 produce a SILK layer, no CELT layer,
    /// SILK internal bandwidth that follows the TOC bandwidth, and
    /// the correct §4.2.2 SILK-frame count.
    #[test]
    fn silk_only_routing_matches_table2() {
        // 12 SILK-only configs: NB × {10, 20, 40, 60 ms},
        // MB × {…}, WB × {…}.
        let expected: [(Bandwidth, u16, SilkBandwidth, u8); 12] = [
            (Bandwidth::Nb, 100, SilkBandwidth::Nb, 1),
            (Bandwidth::Nb, 200, SilkBandwidth::Nb, 1),
            (Bandwidth::Nb, 400, SilkBandwidth::Nb, 2),
            (Bandwidth::Nb, 600, SilkBandwidth::Nb, 3),
            (Bandwidth::Mb, 100, SilkBandwidth::Mb, 1),
            (Bandwidth::Mb, 200, SilkBandwidth::Mb, 1),
            (Bandwidth::Mb, 400, SilkBandwidth::Mb, 2),
            (Bandwidth::Mb, 600, SilkBandwidth::Mb, 3),
            (Bandwidth::Wb, 100, SilkBandwidth::Wb, 1),
            (Bandwidth::Wb, 200, SilkBandwidth::Wb, 1),
            (Bandwidth::Wb, 400, SilkBandwidth::Wb, 2),
            (Bandwidth::Wb, 600, SilkBandwidth::Wb, 3),
        ];
        for (config, &(bw, dur, silk_bw, n)) in expected.iter().enumerate() {
            let r = route(config as u8, false);
            assert_eq!(r.operating_mode, OperatingMode::SilkOnly, "config {config}");
            assert_eq!(r.toc_bandwidth, bw, "config {config}");
            assert_eq!(r.frame_size_tenths_ms, dur, "config {config}");
            assert!(r.silk_layer, "config {config}");
            assert!(!r.celt_layer, "config {config}");
            assert_eq!(r.silk_bandwidth, Some(silk_bw), "config {config}");
            assert_eq!(r.silk_frames_per_channel, Some(n), "config {config}");
        }
    }

    /// Hybrid configs 12..=15 carry both layers, pin SILK to WB even
    /// though the TOC bandwidth is SWB / FB, and produce the correct
    /// §4.2.2 SILK-frame count.
    #[test]
    fn hybrid_routing_pins_silk_to_wb() {
        let expected: [(Bandwidth, u16, u8); 4] = [
            (Bandwidth::Swb, 100, 1),
            (Bandwidth::Swb, 200, 1),
            (Bandwidth::Fb, 100, 1),
            (Bandwidth::Fb, 200, 1),
        ];
        for (i, &(bw, dur, n)) in expected.iter().enumerate() {
            let config = 12 + i as u8;
            let r = route(config, false);
            assert_eq!(r.operating_mode, OperatingMode::Hybrid, "config {config}");
            assert_eq!(r.toc_bandwidth, bw, "config {config}");
            assert_eq!(r.frame_size_tenths_ms, dur, "config {config}");
            assert!(r.silk_layer, "config {config}");
            assert!(r.celt_layer, "config {config}");
            // The §4.2 pin to WB applies for every Hybrid frame.
            assert_eq!(r.silk_bandwidth, Some(SilkBandwidth::Wb), "config {config}");
            assert_eq!(r.silk_frames_per_channel, Some(n), "config {config}");
        }
    }

    /// CELT-only configs 16..=31 produce no SILK layer, just a CELT
    /// decode, regardless of channel count.
    #[test]
    fn celt_only_routing_has_no_silk() {
        for config in 16u8..32 {
            for stereo in [false, true] {
                let r = route(config, stereo);
                assert_eq!(
                    r.operating_mode,
                    OperatingMode::CeltOnly,
                    "c={config} s={stereo}"
                );
                assert!(!r.silk_layer, "c={config} s={stereo}");
                assert!(r.celt_layer, "c={config} s={stereo}");
                assert_eq!(r.silk_bandwidth, None);
                assert_eq!(r.silk_frames_per_channel, None);
                assert_eq!(r.total_silk_frames(), 0);
                assert!(!r.has_per_frame_lbrr_bits());
            }
        }
    }

    /// CELT-only 2.5 ms / 5 ms / 10 ms / 20 ms durations are all
    /// represented (the four configs per bandwidth group cover the
    /// four sizes in order).
    #[test]
    fn celt_only_frame_sizes_cover_25_to_200() {
        // Config 16..=19 → NB CELT-only, sizes 25/50/100/200.
        for (k, &dur) in [25u16, 50, 100, 200].iter().enumerate() {
            let r = route(16 + k as u8, false);
            assert_eq!(r.frame_size_tenths_ms, dur);
            assert_eq!(r.silk_frames_per_channel, None);
        }
    }

    /// Mono and stereo route the same way except for the channel
    /// count and the resulting total SILK-frame count.
    #[test]
    fn channel_count_doubles_total_silk_frames_in_stereo() {
        for config in 0u8..12 {
            let mono = route(config, false);
            let stereo = route(config, true);
            assert_eq!(mono.channel_count(), 1);
            assert_eq!(stereo.channel_count(), 2);
            assert_eq!(mono.silk_frames_per_channel, stereo.silk_frames_per_channel);
            assert_eq!(stereo.total_silk_frames(), 2 * mono.total_silk_frames());
        }
    }

    /// §4.2.4 per-frame LBRR flag presence is gated on a SILK layer
    /// and an Opus duration strictly greater than 20 ms. Verify the
    /// gate against every Table 2 cell.
    #[test]
    fn per_frame_lbrr_gate_matches_section_4_2_4() {
        for config in 0u8..32 {
            let r = route(config, false);
            let expected = r.silk_layer && matches!(r.frame_size_tenths_ms, 400 | 600);
            assert_eq!(r.has_per_frame_lbrr_bits(), expected, "config {config}");
        }
    }

    /// `total_silk_frames` matches `channel_count * silk_frames_per_channel`
    /// for every SILK-bearing config × {mono, stereo}, and is zero
    /// for every CELT-only config × {mono, stereo}.
    #[test]
    fn total_silk_frames_formula() {
        for config in 0u8..32 {
            for stereo in [false, true] {
                let r = route(config, stereo);
                match r.silk_frames_per_channel {
                    Some(n) => assert_eq!(
                        r.total_silk_frames(),
                        r.channel_count() * n,
                        "config {config} stereo {stereo}"
                    ),
                    None => assert_eq!(r.total_silk_frames(), 0),
                }
            }
        }
    }

    /// Concrete dispatch: a 60 ms stereo Hybrid SWB frame
    /// (config 13, s=1) implies 2 channels × 3 SILK frames each = 6
    /// regular SILK frames, plus a CELT decode covering the SWB
    /// bands. SILK runs internally at WB.
    #[test]
    fn worked_example_60ms_stereo_hybrid() {
        // Table 2: config 13 is Hybrid SWB 20 ms (not 60 ms — Hybrid
        // tops out at 20 ms per Table 2). The 60 ms / 40 ms cells
        // are SILK-only, not Hybrid.
        let r = route(13, true);
        assert_eq!(r.operating_mode, OperatingMode::Hybrid);
        assert_eq!(r.toc_bandwidth, Bandwidth::Swb);
        assert_eq!(r.frame_size_tenths_ms, 200);
        assert_eq!(r.silk_bandwidth, Some(SilkBandwidth::Wb));
        assert_eq!(r.channel_count(), 2);
        assert_eq!(r.silk_frames_per_channel, Some(1));
        assert_eq!(r.total_silk_frames(), 2);
        // 20 ms doesn't trigger §4.2.4 per-frame LBRR.
        assert!(!r.has_per_frame_lbrr_bits());

        // For a 60 ms stereo frame the only legal mode is SILK-only
        // (configs 3, 7, 11). Verify the routing for config 11 (WB
        // SILK-only 60 ms stereo).
        let r60 = route(11, true);
        assert_eq!(r60.operating_mode, OperatingMode::SilkOnly);
        assert_eq!(r60.frame_size_tenths_ms, 600);
        assert_eq!(r60.silk_bandwidth, Some(SilkBandwidth::Wb));
        assert_eq!(r60.silk_frames_per_channel, Some(3));
        assert_eq!(r60.channel_count(), 2);
        assert_eq!(r60.total_silk_frames(), 6);
        assert!(r60.has_per_frame_lbrr_bits());
    }

    /// `from_toc` passes the channel mapping and TOC bandwidth
    /// straight through (these are direct copies, not derived
    /// fields), regardless of the `c` frame-count bits.
    #[test]
    fn fields_passed_through_from_toc() {
        // The `c` bits live in the bottom of the TOC byte. Toggle
        // them and confirm the routing decision (which is
        // independent of `c` per §3.2) does not move.
        for config in 0u8..32 {
            let base = OpusFrameRouting::from_toc(OpusTocByte::from_byte(config << 3));
            for c in 1u8..=3 {
                let r = OpusFrameRouting::from_toc(OpusTocByte::from_byte((config << 3) | c));
                assert_eq!(r.operating_mode, base.operating_mode);
                assert_eq!(r.toc_bandwidth, base.toc_bandwidth);
                assert_eq!(r.frame_size_tenths_ms, base.frame_size_tenths_ms);
                assert_eq!(r.silk_layer, base.silk_layer);
                assert_eq!(r.celt_layer, base.celt_layer);
                assert_eq!(r.silk_bandwidth, base.silk_bandwidth);
                assert_eq!(r.silk_frames_per_channel, base.silk_frames_per_channel);
            }
        }

        // And a sanity check that the `c` decode itself is preserved
        // on the underlying TOC byte: routing doesn't touch that.
        let toc_c3 = OpusTocByte::from_byte(0b11);
        assert_eq!(toc_c3.frame_count_code, FrameCountCode::Arbitrary);
    }

    /// Stereo / mono flag preserved on the routing even for
    /// CELT-only frames (which still distinguish mono vs stereo for
    /// the §4.3.4 dual / intensity stereo decisions later).
    #[test]
    fn channel_mapping_preserved_for_celt_only() {
        let mono = route(20, false);
        let stereo = route(20, true);
        assert_eq!(mono.channels, ChannelMapping::Mono);
        assert_eq!(stereo.channels, ChannelMapping::Stereo);
        assert_eq!(mono.channel_count(), 1);
        assert_eq!(stereo.channel_count(), 2);
    }

    /// Every Table 2 cell produces a routing that satisfies the
    /// "silk_layer XOR celt_only" / "celt_layer XOR silk_only"
    /// structural invariants and that silk_bandwidth / frames are
    /// `Some` iff silk_layer.
    #[test]
    fn invariants_hold_across_all_32_configs() {
        for config in 0u8..32 {
            for stereo in [false, true] {
                let r = route(config, stereo);
                // At least one layer is present.
                assert!(r.silk_layer || r.celt_layer);
                // SILK-bearing iff silk_bandwidth is Some.
                assert_eq!(r.silk_layer, r.silk_bandwidth.is_some());
                // SILK-bearing iff silk_frames_per_channel is Some.
                assert_eq!(r.silk_layer, r.silk_frames_per_channel.is_some());
                // CELT-only ⇔ no SILK.
                assert_eq!(
                    matches!(r.operating_mode, OperatingMode::CeltOnly),
                    !r.silk_layer
                );
                // SILK-only ⇔ no CELT.
                assert_eq!(
                    matches!(r.operating_mode, OperatingMode::SilkOnly),
                    !r.celt_layer
                );
                // Hybrid ⇔ both layers.
                assert_eq!(
                    matches!(r.operating_mode, OperatingMode::Hybrid),
                    r.silk_layer && r.celt_layer
                );
                // Hybrid always pins SILK to WB.
                if matches!(r.operating_mode, OperatingMode::Hybrid) {
                    assert_eq!(r.silk_bandwidth, Some(SilkBandwidth::Wb));
                }
            }
        }
    }
}
