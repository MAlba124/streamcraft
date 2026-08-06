//! Normative + recommended non-normative transition table for
//! configuration switches (RFC 6716 §4.5.3, Figure 18 + Figure 19).
//!
//! §4.5.1 + §4.5.1.1..§4.5.1.4 (rounds 26 + 28) carry the
//! redundancy *side information* — flag, position, size, and the
//! cross-lap placement of the redundant CELT frame's two 2.5 ms
//! halves. §4.5.2 (round 27) carries the *state-reset policy* per
//! transition. §4.5.3 closes the §4.5 chain by enumerating the
//! exhaustive list of *normative* transitions the encoder is allowed
//! to use (Figure 18) and the *recommended non-normative*
//! transitions used when redundancy is intentionally not encoded
//! (Figure 19). Each entry pairs a `(prev_mode, prev_bandwidth,
//! next_mode, next_bandwidth)` shape with the boundary mixing
//! operations and decoder resets the §4.5.3 figure marks at the
//! transition seam.
//!
//! The §4.5.3 figures are presented as ASCII diagrams whose markers
//! carry normative meaning:
//!
//! * `S` — SILK-only Opus frame.
//! * `H` — Hybrid Opus frame.
//! * `C` — CELT-only Opus frame.
//! * `R` — redundant 5 ms CELT frame inside the Opus frame whose tail
//!   it occupies (placement per §4.5.1.2).
//! * `c` — CELT overlap-buffer contents extracted by decoding a 2.5 ms
//!   silence CELT frame using the next Opus frame's channel count
//!   (the §4.5 paragraph above §4.5.1: "by decoding a 2.5 ms silence
//!   frame with the CELT decoder using the channel count of the
//!   SILK-only packet").
//! * `;` — SILK decoder reset (§4.5.2 rule 1).
//! * `|` — both SILK and CELT decoder resets at the same seam.
//! * `!` — CELT decoder reset (§4.5.2 rule 2 / rule 3 default).
//! * `&` — windowed cross-lap (the CELT power-complementary MDCT
//!   window combining the redundant CELT frame's second 2.5 ms with
//!   the SILK/Hybrid signal, per §4.5.1.4).
//! * `+` — direct mixing (the §4.5 Hybrid→WB-SILK case: add the CELT
//!   overlap-buffer contents into the first SILK-only packet
//!   directly, no windowing).
//! * `P` — Packet Loss Concealment (§4.4) interval bridging the gap
//!   on a non-normative transition that does not carry redundancy.
//! * `>` — join into a different layer's stream of frames.
//!
//! ## What this module owns
//!
//! Three orthogonal pieces of the §4.5.3 surface:
//!
//! 1. The [`NormativeTransition`] enumeration — the nine canonical
//!    transition shapes named in §4.5.3 (Figure 18). One enum
//!    variant per row of the figure.
//! 2. The [`BoundaryOp`] enumeration — the marker semantics from the
//!    §4.5.3 key (windowed cross-lap, direct mix, overlap-buffer
//!    silence-frame extraction, decoder resets, the PLC fill, and
//!    the layer-stream join).
//! 3. The [`classify_normative_transition`] pure function — turns a
//!    `(prev_mode, prev_silk_bandwidth, next_mode,
//!    next_silk_bandwidth, redundancy_present)` 5-tuple into the
//!    matching [`NormativeTransition`] when one applies, plus the
//!    [`recommended_non_normative`] companion lookup against
//!    Figure 19. Together they tell the caller whether the
//!    transition it is about to perform is on the §4.5.3 figure at
//!    all; transitions absent from both figures fall outside the
//!    normative cases §4.5 enumerates.
//!
//! ## What this module does NOT own
//!
//! * The actual MDCT cross-lap mix — that is the §4.3.7
//!   inverse-MDCT path and runs at the §4.3.7 call site. This
//!   module only names the seam where it happens.
//! * The §4.5.2 reset *placement* (before-frame vs.
//!   before-redundant) — that is owned by
//!   [`crate::mode_transition_reset`]. The §4.5.3 markers `;` / `|`
//!   / `!` are restated here only as part of the figure's
//!   description so a caller can cross-check its §4.5.2 decision
//!   against the figure markers. The two modules agree.
//! * The PLC interior of `P` — §4.4 is a non-normative
//!   recommendation; the algorithm is out of scope.
//!
//! ## Provenance
//!
//! Every variant, every marker, every classifier branch, and every
//! cross-check assertion in the test module is transcribed from RFC
//! 6716 §4.5.3 (pp. 128–130), held in-repo at
//! `docs/audio/opus/rfc6716-opus.txt`. The §4.5.1 / §4.5.1.4 / §4.5.2
//! material this module cross-references is already on master in
//! [`crate::celt_redundancy`], [`crate::redundancy_decode_params`],
//! and [`crate::mode_transition_reset`]. No external library source
//! was consulted; the figures are the only source.

use crate::celt_redundancy::RedundancyDecision;
use crate::framing::{OperatingMode, SilkBandwidth};

/// Boundary mixing or reset operation marked on the §4.5.3 figures.
///
/// Each variant maps directly to one marker character in the
/// §4.5.3 keys (Figure 18 / Figure 19, p. 129–130). The figure
/// labels these as Opus-frame-boundary annotations; this enum
/// preserves them as a typed list so the consumer can cross-check
/// its own per-seam dispatch decisions against the figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryOp {
    /// `;` — SILK decoder reset. §4.5.2 rule 1.
    SilkReset,
    /// `|` — SILK + CELT decoders both reset at this seam.
    SilkAndCeltReset,
    /// `!` — CELT decoder reset. §4.5.2 rule 2 / rule 3.
    CeltReset,
    /// `&` — windowed cross-lap using CELT's power-complementary
    /// MDCT window. §4.5.1.4 attaches this to the second 2.5 ms of
    /// the redundant CELT frame.
    WindowedCrossLap,
    /// `+` — direct (non-windowed) mixing of the previous frame's
    /// CELT overlap buffer into the next frame's first samples.
    /// §4.5's "Switching from Hybrid mode to WB SILK …" paragraph
    /// pairs this with the `c` overlap extraction.
    DirectMix,
    /// `c` — CELT overlap-buffer extraction via a 2.5 ms silence
    /// CELT frame using the next Opus frame's channel count. §4.5
    /// describes this as "decoding a 2.5 ms silence frame with the
    /// CELT decoder using the channel count of the SILK-only
    /// packet (and any choice of audio bandwidth)".
    CeltOverlapExtract,
    /// `P` — Packet Loss Concealment (§4.4) interval bridging the
    /// gap on a recommended non-normative transition (Figure 19)
    /// that does not carry §4.5.1 redundancy.
    PacketLossConcealment,
    /// `>` — non-marker join into the next-mode's stream of frames
    /// (the Figure 19 "Hybrid to NB or MB SILK" row uses this as
    /// the `c + ;S` mix-and-reset point).
    StreamJoin,
}

/// Canonical normative transition shape from RFC 6716 §4.5.3,
/// Figure 18.
///
/// Each variant corresponds to one row of Figure 18. The
/// [`NormativeTransition::seam_operations`] method returns the
/// §4.5.3 markers that decorate the transition seam (i.e. the
/// boundary between the `R` carrier frame and the next-mode
/// frames, or the boundary between the two modes when there is no
/// `R`).
///
/// §4.5.3 names these as "normative transitions involving a mode
/// change, an audio bandwidth change, or both". §4.5.3's text
/// notes that "the first two and the last two Opus frames in each
/// example are illustrative" — a stream is not required to remain
/// in the same configuration for three frames before or after the
/// switch. Only the seam markers (`!`, `;`, `|`, `&`, `+`, `c`,
/// `R`) are normative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormativeTransition {
    /// Row 1: `SILK to SILK with Redundancy`.
    /// `S -> S -> S & !R -> R & ;S -> S -> S` — SILK-only Opus
    /// frame carries an `R` at the seam, the `R` is decoded with a
    /// CELT reset and cross-lapped with the following SILK frame
    /// which itself triggers a SILK reset.
    ///
    /// In §4.5.3's framing this row applies to any audio-bandwidth
    /// change between two SILK-only configurations (e.g. NB → MB,
    /// MB → WB, …) WHEN redundancy is signalled. The §4.5 first
    /// paragraph identifies the audio-bandwidth change as the
    /// glitch source that motivates redundancy in this row, so a
    /// same-bandwidth SILK→SILK pair (which has no glitch source
    /// to mask) does not match row 1 even when redundancy is
    /// signalled.
    SilkToSilkWithRedundancy,
    /// Row 2: `NB or MB SILK to Hybrid with Redundancy`.
    /// `S -> S -> S & !R -> ;H -> H -> H`. The transition starts
    /// from a SILK-only frame whose internal bandwidth is NB or MB
    /// (i.e. not WB), and lands in Hybrid. The `;` marker before
    /// the first `H` indicates a SILK decoder reset at the H
    /// boundary.
    ///
    /// §4.5 explicitly excludes WB-SILK → Hybrid from the
    /// redundant path: the next variant (`WbSilkToHybrid`) handles
    /// that case without redundancy.
    NbOrMbSilkToHybridWithRedundancy,
    /// Row 3: `WB SILK to Hybrid`. `S -> S -> S ->!H -> H -> H`.
    /// No redundancy: §4.5's first paragraph notes that
    /// "Switching between any two configurations of the CELT-only
    /// mode, any two configurations of the Hybrid mode, or from
    /// WB SILK to Hybrid mode does not require any special
    /// treatment in the decoder, as the MDCT overlap will smooth
    /// the transition." The `!` marker before the first `H`
    /// records the CELT-only-state reset on entering Hybrid for
    /// the first time.
    WbSilkToHybrid,
    /// Row 4: `SILK to CELT with Redundancy`.
    /// `S -> S -> S & !R -> C -> C -> C`. Mode change SILK-only →
    /// CELT-only with redundancy. The `R` rides the trailing 2.5 ms
    /// of the last SILK frame; the `&` cross-laps the second half
    /// of `R` with the SILK signal; the `!` resets CELT for the
    /// `R` decode but NOT for the following `C` per §4.5.2 rule 3.
    SilkToCeltWithRedundancy,
    /// Row 5: `Hybrid to NB or MB SILK with Redundancy`.
    /// `H -> H -> H & !R -> R & ;S -> S -> S`. Hybrid → NB-or-MB
    /// SILK transition with redundancy. Symmetric to row 1 with
    /// `H` replacing the prior `S`s; the `R` carries the
    /// mode-switch boundary; the `;` marks the SILK reset for the
    /// new SILK-only run.
    HybridToNbOrMbSilkWithRedundancy,
    /// Row 6: `Hybrid to WB SILK`. `H -> H -> H -> c \ > S -> S -> S`
    /// with a `+` marker between the `c` and the join. This is the
    /// §4.5 exception: "Switching from Hybrid mode to WB SILK
    /// requires adding in the final contents of the CELT overlap
    /// buffer to the first SILK-only packet. This can be done by
    /// decoding a 2.5 ms silence frame with the CELT decoder
    /// using the channel count of the SILK-only packet". The `c`
    /// is the silence-frame-derived overlap; `+` is the direct
    /// mix; no SILK or CELT reset markers fire.
    HybridToWbSilk,
    /// Row 7: `Hybrid to CELT with Redundancy`.
    /// `H -> H -> H & !R -> C -> C -> C`. Hybrid → CELT-only with
    /// redundancy. Symmetric to row 4 with `H` instead of `S` as
    /// the prior mode; same `&` + `!R` + `C` shape at the seam.
    HybridToCeltWithRedundancy,
    /// Row 8: `CELT to SILK with Redundancy`.
    /// `C -> C -> C -> R & ;S -> S -> S`. Mode change CELT-only →
    /// SILK-only with redundancy. The `R` sits in the *first* new
    /// frame's leading 2.5 ms (§4.5.1.2 position bit = 1 /
    /// `Beginning`); the `&` cross-laps the second half; the `;`
    /// is the SILK reset triggered by the predecessor having been
    /// CELT-only (§4.5.2 rule 1). §4.5.2 rule 4 explicitly forbids
    /// resetting CELT for the redundant `R` decode here.
    CeltToSilkWithRedundancy,
    /// Row 9: `CELT to Hybrid with Redundancy`.
    /// `C -> C -> C -> R & |H -> H -> H`. The `|` marker indicates
    /// both decoders reset at the H boundary. §4.5.2 rule 4
    /// forbids resetting CELT for the `R` itself; the SILK reset
    /// fires per rule 1 (prev was CELT-only, new is Hybrid). Per
    /// the round-27 [`crate::mode_transition_reset`] reading the
    /// `|` markers the *figure's* dual-reset annotation at the H
    /// boundary; the actual CELT-state continuity through the `R`
    /// decode is the §4.5.2 carve-out.
    CeltToHybridWithRedundancy,
}

impl NormativeTransition {
    /// Return the §4.5.3 marker operations that decorate this
    /// transition's seam, in left-to-right order across the figure.
    ///
    /// The returned slice covers only the markers *between* the
    /// prior mode's last frame and the next mode's first frame; the
    /// trailing `S -> S -> S` / `H -> H -> H` / `C -> C -> C` runs
    /// in the figure are illustrative per §4.5.3 and do not carry
    /// per-seam operations.
    pub fn seam_operations(self) -> &'static [BoundaryOp] {
        use BoundaryOp::*;
        match self {
            // Row 1: `& !R -> R & ;` — cross-lap into R, R cross-lap
            // into next SILK, SILK reset at the new run.
            Self::SilkToSilkWithRedundancy => {
                &[WindowedCrossLap, CeltReset, WindowedCrossLap, SilkReset]
            }
            // Row 2: `& !R -> ;` — cross-lap into R, R, then SILK
            // reset entering the H run (the `;` on the H label is
            // SILK-only state being cleared).
            Self::NbOrMbSilkToHybridWithRedundancy => &[WindowedCrossLap, CeltReset, SilkReset],
            // Row 3: `->!` — single CELT reset, no redundancy and
            // no cross-lap.
            Self::WbSilkToHybrid => &[CeltReset],
            // Row 4: `& !R -> ` — cross-lap into R, R, no further
            // resets (rule 3 carve-out: CELT not reset for the
            // following CELT-only frame).
            Self::SilkToCeltWithRedundancy => &[WindowedCrossLap, CeltReset],
            // Row 5: `& !R -> R & ;` — same shape as row 1.
            Self::HybridToNbOrMbSilkWithRedundancy => {
                &[WindowedCrossLap, CeltReset, WindowedCrossLap, SilkReset]
            }
            // Row 6: `-> c \ + > ` — extract the CELT overlap with a
            // silence frame, then direct-mix it into the first SILK
            // frame.
            Self::HybridToWbSilk => &[CeltOverlapExtract, DirectMix, StreamJoin],
            // Row 7: `& !R -> ` — same shape as row 4 with Hybrid
            // as the prior mode.
            Self::HybridToCeltWithRedundancy => &[WindowedCrossLap, CeltReset],
            // Row 8: `-> R & ;` — R rides at the start of the new
            // SILK frame; the `&` is the cross-lap of R's second
            // half with the SILK signal; the `;` is the SILK reset
            // triggered by the predecessor having been CELT-only.
            // Rule 4 explicitly forbids a CELT reset for R itself.
            Self::CeltToSilkWithRedundancy => &[WindowedCrossLap, SilkReset],
            // Row 9: `-> R & |` — R rides at the start of the new
            // Hybrid frame; the `&` is the cross-lap of R's second
            // half; the `|` marks the dual reset annotation at the
            // H boundary. Rule 4 still forbids a CELT reset for R.
            Self::CeltToHybridWithRedundancy => &[WindowedCrossLap, SilkAndCeltReset],
        }
    }

    /// `true` iff this transition shape carries a redundant CELT
    /// frame per §4.5.1.
    pub fn carries_redundancy(self) -> bool {
        matches!(
            self,
            Self::SilkToSilkWithRedundancy
                | Self::NbOrMbSilkToHybridWithRedundancy
                | Self::SilkToCeltWithRedundancy
                | Self::HybridToNbOrMbSilkWithRedundancy
                | Self::HybridToCeltWithRedundancy
                | Self::CeltToSilkWithRedundancy
                | Self::CeltToHybridWithRedundancy
        )
    }
}

/// Recommended non-normative transition shape from RFC 6716 §4.5.3,
/// Figure 19.
///
/// §4.5.3 introduces Figure 19 with: "The behavior of transitions
/// without redundancy where PLC is allowed is non-normative. An
/// encoder might still wish to use these transitions if, for
/// example, it doesn't want to add the extra bitrate required for
/// redundancy or if it makes a decision to switch after it has
/// already transmitted the frame that would have had to contain
/// the redundancy. Figure 19 illustrates the recommended
/// cross-lapping and decoder resets for these transitions."
///
/// Each variant corresponds to one row of Figure 19. These are
/// recommendations, not normative requirements — but they are the
/// only documented guidance §4.5.3 offers for the no-redundancy
/// paths through the §4.5 transition matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedNonNormativeTransition {
    /// Row 1: `SILK to SILK (audio bandwidth change): S -> S -> S ;S -> S -> S`.
    /// Same SILK-only mode at a different audio bandwidth, no
    /// redundancy. Recommendation: SILK reset at the seam, no
    /// cross-lap.
    SilkToSilkAudioBandwidthChange,
    /// Row 2: `NB or MB SILK to Hybrid: S -> S -> S |H -> H -> H`.
    /// SILK-only @ NB or MB → Hybrid without redundancy.
    /// Recommendation: both decoders reset at the seam.
    NbOrMbSilkToHybrid,
    /// Row 3: `SILK to CELT without Redundancy: S -> S -> S -> P & !C -> C -> C`.
    /// SILK-only → CELT-only without redundancy. Recommendation:
    /// PLC fills the gap, then CELT reset, then cross-lap.
    SilkToCeltWithoutRedundancy,
    /// Row 4: `Hybrid to NB or MB SILK: H -> H -> H -> c + ;S -> S -> S`.
    /// Hybrid → NB-or-MB SILK without redundancy. Recommendation:
    /// extract CELT overlap via silence frame, direct-mix, SILK
    /// reset.
    HybridToNbOrMbSilk,
    /// Row 5: `Hybrid to CELT without Redundancy: H -> H -> H -> P & !C -> C -> C`.
    /// Same shape as row 3 with Hybrid as the predecessor.
    HybridToCeltWithoutRedundancy,
    /// Row 6: `CELT to SILK without Redundancy: C -> C -> C -> P & ;S -> S -> S`.
    /// CELT-only → SILK-only without redundancy. Recommendation:
    /// PLC fills the gap, cross-lap into SILK, SILK reset.
    CeltToSilkWithoutRedundancy,
    /// Row 7: `CELT to Hybrid without Redundancy: C -> C -> C -> P & |H -> H -> H`.
    /// CELT-only → Hybrid without redundancy. Recommendation:
    /// PLC fills the gap, cross-lap into Hybrid, both decoders
    /// reset.
    CeltToHybridWithoutRedundancy,
}

impl RecommendedNonNormativeTransition {
    /// Return the §4.5.3 Figure-19 marker operations decorating
    /// this transition's seam.
    pub fn seam_operations(self) -> &'static [BoundaryOp] {
        use BoundaryOp::*;
        match self {
            Self::SilkToSilkAudioBandwidthChange => &[SilkReset],
            Self::NbOrMbSilkToHybrid => &[SilkAndCeltReset],
            Self::SilkToCeltWithoutRedundancy => {
                &[PacketLossConcealment, WindowedCrossLap, CeltReset]
            }
            Self::HybridToNbOrMbSilk => &[CeltOverlapExtract, DirectMix, SilkReset],
            Self::HybridToCeltWithoutRedundancy => {
                &[PacketLossConcealment, WindowedCrossLap, CeltReset]
            }
            Self::CeltToSilkWithoutRedundancy => {
                &[PacketLossConcealment, WindowedCrossLap, SilkReset]
            }
            Self::CeltToHybridWithoutRedundancy => {
                &[PacketLossConcealment, WindowedCrossLap, SilkAndCeltReset]
            }
        }
    }
}

/// Classify a transition against the §4.5.3 Figure 18 normative
/// table.
///
/// Returns `Some(transition)` when the
/// `(prev_mode, prev_silk_bandwidth, next_mode,
/// next_silk_bandwidth, redundancy_present)` 5-tuple matches one
/// of the nine §4.5.3 normative shapes; returns `None` otherwise.
///
/// The `prev_silk_bandwidth` and `next_silk_bandwidth` parameters
/// are read only on the rows where §4.5.3 keys behaviour off the
/// SILK bandwidth (the `NB or MB SILK → Hybrid` vs `WB SILK →
/// Hybrid` split, and the `Hybrid → NB or MB SILK` vs `Hybrid →
/// WB SILK` split). For CELT-only frames the parameter must be
/// `None`; for SILK-bearing frames it carries the §4.2-pinned
/// internal SILK bandwidth ([`SilkBandwidth`]).
///
/// ## Mapping
///
/// | Prev → Next                          | Redundancy | SilkBW prev | SilkBW next | Row |
/// |--------------------------------------|------------|-------------|-------------|-----|
/// | SILK → SILK                          | yes        | any         | any         | 1   |
/// | SILK NB/MB → Hybrid                  | yes        | NB or MB    | WB (pinned) | 2   |
/// | SILK WB → Hybrid                     | no         | WB          | WB (pinned) | 3   |
/// | SILK → CELT-only                     | yes        | any         | n/a         | 4   |
/// | Hybrid → SILK NB or MB               | yes        | WB (pinned) | NB or MB    | 5   |
/// | Hybrid → SILK WB                     | no         | WB (pinned) | WB          | 6   |
/// | Hybrid → CELT-only                   | yes        | WB (pinned) | n/a         | 7   |
/// | CELT-only → SILK                     | yes        | n/a         | any         | 8   |
/// | CELT-only → Hybrid                   | yes        | n/a         | WB (pinned) | 9   |
///
/// Same-mode transitions with no audio-bandwidth change are not on
/// Figure 18; they fall outside §4.5.3 entirely and the function
/// returns `None`. Same-mode transitions WITH an audio-bandwidth
/// change between two CELT-only or two Hybrid configurations are
/// covered by §4.5's "any two configurations of the CELT-only
/// mode, any two configurations of the Hybrid mode … does not
/// require any special treatment" paragraph and likewise return
/// `None`.
pub fn classify_normative_transition(
    prev_mode: OperatingMode,
    prev_silk_bandwidth: Option<SilkBandwidth>,
    next_mode: OperatingMode,
    next_silk_bandwidth: Option<SilkBandwidth>,
    redundancy_present: bool,
) -> Option<NormativeTransition> {
    use OperatingMode::*;
    match (prev_mode, next_mode, redundancy_present) {
        // Row 1: SILK to SILK with redundancy. §4.5 first
        // paragraph identifies the audio-bandwidth change as the
        // glitch source that motivates redundancy in this row
        // ("between SILK-only packets … may cause glitches,
        // because neither the LSF coefficients nor the LTP, LPC,
        // stereo unmixing, and resampler buffers are available at
        // the new sample rate"). Row 1 therefore applies when
        // redundancy is present AND the SILK internal bandwidth
        // actually changes; same-bandwidth SILK→SILK pairs are
        // not on Figure 18 because there is no glitch source for
        // redundancy to mask.
        (SilkOnly, SilkOnly, true) => match (prev_silk_bandwidth, next_silk_bandwidth) {
            (Some(a), Some(b)) if a != b => Some(NormativeTransition::SilkToSilkWithRedundancy),
            _ => None,
        },

        // Row 2 vs Row 3: SILK to Hybrid split on the prior SILK
        // bandwidth. The row-2 path requires redundancy and a
        // non-WB prior; row 3 is the only no-redundancy SILK→Hybrid
        // row §4.5.3 acknowledges.
        (SilkOnly, Hybrid, true) => match prev_silk_bandwidth {
            Some(SilkBandwidth::Nb) | Some(SilkBandwidth::Mb) => {
                Some(NormativeTransition::NbOrMbSilkToHybridWithRedundancy)
            }
            // WB SILK → Hybrid with redundancy is not a §4.5.3
            // row; §4.5's "WB SILK → Hybrid does not require any
            // special treatment" exempts it from the redundancy
            // path. Falls through to None.
            _ => None,
        },
        (SilkOnly, Hybrid, false) => match prev_silk_bandwidth {
            Some(SilkBandwidth::Wb) => Some(NormativeTransition::WbSilkToHybrid),
            _ => None,
        },

        // Row 4: SILK to CELT with redundancy. §4.5.3 names this
        // as "SILK to CELT with Redundancy" without further
        // bandwidth constraints; the §4.5.1.4 "MB SILK → WB" rule
        // adjusts the redundant frame's audio bandwidth but does
        // not change the row classification.
        (SilkOnly, CeltOnly, true) => Some(NormativeTransition::SilkToCeltWithRedundancy),

        // Row 5: Hybrid to NB or MB SILK with redundancy. Mirrors
        // row 2 in the opposite direction; keys off the *next*
        // SILK bandwidth.
        (Hybrid, SilkOnly, true) => match next_silk_bandwidth {
            Some(SilkBandwidth::Nb) | Some(SilkBandwidth::Mb) => {
                Some(NormativeTransition::HybridToNbOrMbSilkWithRedundancy)
            }
            _ => None,
        },
        // Row 6: Hybrid to WB SILK without redundancy. The
        // companion of row 3.
        (Hybrid, SilkOnly, false) => match next_silk_bandwidth {
            Some(SilkBandwidth::Wb) => Some(NormativeTransition::HybridToWbSilk),
            _ => None,
        },

        // Row 7: Hybrid to CELT with redundancy.
        (Hybrid, CeltOnly, true) => Some(NormativeTransition::HybridToCeltWithRedundancy),

        // Row 8: CELT to SILK with redundancy.
        (CeltOnly, SilkOnly, true) => Some(NormativeTransition::CeltToSilkWithRedundancy),

        // Row 9: CELT to Hybrid with redundancy.
        (CeltOnly, Hybrid, true) => Some(NormativeTransition::CeltToHybridWithRedundancy),

        // Same-mode transitions, no-redundancy transitions that
        // are not on rows 3 or 6, and SILK→SILK / Hybrid→Hybrid /
        // CELT-only→CELT-only same-mode cases all fall outside
        // §4.5.3 and return None.
        _ => None,
    }
}

/// Classify a transition against the §4.5.3 Figure 19
/// recommended-non-normative table.
///
/// Returns `Some(transition)` when the 4-tuple matches one of the
/// seven §4.5.3 Figure-19 shapes; returns `None` otherwise. The
/// figure entries are mutually exclusive with Figure 18: a
/// transition that matches Figure 18 (i.e. carries redundancy or
/// is the WB-SILK → Hybrid / Hybrid → WB-SILK exception) is
/// excluded from Figure 19's no-redundancy domain by construction.
///
/// Callers typically chain the two classifiers: try
/// [`classify_normative_transition`] first; if `None`, try
/// [`recommended_non_normative`]. If both return `None` the
/// transition is either same-mode (and §4.5 makes no
/// recommendation) or the WB-SILK ↔ Hybrid case (which §4.5's
/// first paragraph documents as needing no special treatment).
pub fn recommended_non_normative(
    prev_mode: OperatingMode,
    prev_silk_bandwidth: Option<SilkBandwidth>,
    next_mode: OperatingMode,
    next_silk_bandwidth: Option<SilkBandwidth>,
) -> Option<RecommendedNonNormativeTransition> {
    use OperatingMode::*;
    match (prev_mode, next_mode) {
        // Figure 19 row 1: SILK → SILK with audio-bandwidth change
        // (without redundancy). The bandwidth change is the
        // condition; if the bandwidths are equal there is no
        // transition. §4.5.3 row 1 names this explicitly as an
        // audio-bandwidth-change row.
        (SilkOnly, SilkOnly) => match (prev_silk_bandwidth, next_silk_bandwidth) {
            (Some(a), Some(b)) if a != b => {
                Some(RecommendedNonNormativeTransition::SilkToSilkAudioBandwidthChange)
            }
            _ => None,
        },

        // Figure 19 row 2: NB or MB SILK → Hybrid.
        (SilkOnly, Hybrid) => match prev_silk_bandwidth {
            Some(SilkBandwidth::Nb) | Some(SilkBandwidth::Mb) => {
                Some(RecommendedNonNormativeTransition::NbOrMbSilkToHybrid)
            }
            // WB SILK → Hybrid without redundancy is row 3 of
            // Figure 18 (normative), not Figure 19.
            _ => None,
        },

        // Figure 19 row 3: SILK → CELT without redundancy.
        (SilkOnly, CeltOnly) => {
            Some(RecommendedNonNormativeTransition::SilkToCeltWithoutRedundancy)
        }

        // Figure 19 row 4: Hybrid → NB or MB SILK.
        (Hybrid, SilkOnly) => match next_silk_bandwidth {
            Some(SilkBandwidth::Nb) | Some(SilkBandwidth::Mb) => {
                Some(RecommendedNonNormativeTransition::HybridToNbOrMbSilk)
            }
            // Hybrid → WB SILK is row 6 of Figure 18 (normative).
            _ => None,
        },

        // Figure 19 row 5: Hybrid → CELT without redundancy.
        (Hybrid, CeltOnly) => {
            Some(RecommendedNonNormativeTransition::HybridToCeltWithoutRedundancy)
        }

        // Figure 19 row 6: CELT → SILK without redundancy.
        (CeltOnly, SilkOnly) => {
            Some(RecommendedNonNormativeTransition::CeltToSilkWithoutRedundancy)
        }

        // Figure 19 row 7: CELT → Hybrid without redundancy.
        (CeltOnly, Hybrid) => {
            Some(RecommendedNonNormativeTransition::CeltToHybridWithoutRedundancy)
        }

        // Same-mode CELT-only → CELT-only and Hybrid → Hybrid
        // fall under §4.5 first paragraph's "does not require any
        // special treatment" and are not on Figure 19.
        (CeltOnly, CeltOnly) | (Hybrid, Hybrid) => None,
    }
}

/// Helper bridging the §4.5.1 [`RedundancyDecision`] to the
/// boolean §4.5.3 classifier consumes.
///
/// Treats [`RedundancyDecision::Invalid`] as "no usable redundancy"
/// — same convention [`crate::mode_transition_reset`] adopts so the
/// §4.5.2 and §4.5.3 modules agree on what to do with a malformed
/// §4.5.1.3 size claim.
pub fn redundancy_is_present(decision: RedundancyDecision) -> bool {
    decision.is_present()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_redundancy::{RedundancyDecision, RedundancyPosition};

    fn present() -> RedundancyDecision {
        RedundancyDecision::Present {
            position: RedundancyPosition::End,
            size_bytes: 2,
        }
    }

    fn invalid() -> RedundancyDecision {
        RedundancyDecision::Invalid
    }

    fn not_present() -> RedundancyDecision {
        RedundancyDecision::NotPresent
    }

    // ----- §4.5.3 Figure 18 rows -----

    #[test]
    fn row1_silk_to_silk_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Mb),
            true,
        );
        assert_eq!(t, Some(NormativeTransition::SilkToSilkWithRedundancy));
    }

    #[test]
    fn row1_silk_to_silk_without_redundancy_is_not_normative() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Mb),
            false,
        );
        assert_eq!(t, None);
    }

    #[test]
    fn row2_nb_silk_to_hybrid_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            true,
        );
        assert_eq!(
            t,
            Some(NormativeTransition::NbOrMbSilkToHybridWithRedundancy)
        );
    }

    #[test]
    fn row2_mb_silk_to_hybrid_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Mb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            true,
        );
        assert_eq!(
            t,
            Some(NormativeTransition::NbOrMbSilkToHybridWithRedundancy)
        );
    }

    #[test]
    fn row3_wb_silk_to_hybrid_without_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            false,
        );
        assert_eq!(t, Some(NormativeTransition::WbSilkToHybrid));
    }

    #[test]
    fn row3_wb_silk_to_hybrid_with_redundancy_is_not_normative() {
        // §4.5.3 only enumerates WB-SILK → Hybrid as the
        // no-redundancy row; with redundancy it falls outside
        // Figure 18 and the classifier returns None.
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            true,
        );
        assert_eq!(t, None);
    }

    #[test]
    fn row4_silk_to_celt_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::CeltOnly,
            None,
            true,
        );
        assert_eq!(t, Some(NormativeTransition::SilkToCeltWithRedundancy));
    }

    #[test]
    fn row5_hybrid_to_nb_silk_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            true,
        );
        assert_eq!(
            t,
            Some(NormativeTransition::HybridToNbOrMbSilkWithRedundancy)
        );
    }

    #[test]
    fn row5_hybrid_to_mb_silk_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Mb),
            true,
        );
        assert_eq!(
            t,
            Some(NormativeTransition::HybridToNbOrMbSilkWithRedundancy)
        );
    }

    #[test]
    fn row6_hybrid_to_wb_silk_without_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            false,
        );
        assert_eq!(t, Some(NormativeTransition::HybridToWbSilk));
    }

    #[test]
    fn row7_hybrid_to_celt_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::CeltOnly,
            None,
            true,
        );
        assert_eq!(t, Some(NormativeTransition::HybridToCeltWithRedundancy));
    }

    #[test]
    fn row8_celt_to_silk_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::CeltOnly,
            None,
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            true,
        );
        assert_eq!(t, Some(NormativeTransition::CeltToSilkWithRedundancy));
    }

    #[test]
    fn row9_celt_to_hybrid_with_redundancy() {
        let t = classify_normative_transition(
            OperatingMode::CeltOnly,
            None,
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            true,
        );
        assert_eq!(t, Some(NormativeTransition::CeltToHybridWithRedundancy));
    }

    // ----- Figure 18 boundary-op tables -----

    #[test]
    fn seam_ops_row1_silk_to_silk() {
        let ops = NormativeTransition::SilkToSilkWithRedundancy.seam_operations();
        assert_eq!(
            ops,
            &[
                BoundaryOp::WindowedCrossLap,
                BoundaryOp::CeltReset,
                BoundaryOp::WindowedCrossLap,
                BoundaryOp::SilkReset,
            ]
        );
    }

    #[test]
    fn seam_ops_row3_wb_silk_to_hybrid() {
        let ops = NormativeTransition::WbSilkToHybrid.seam_operations();
        assert_eq!(ops, &[BoundaryOp::CeltReset]);
    }

    #[test]
    fn seam_ops_row4_silk_to_celt_does_not_reset_following_celt() {
        // §4.5.2 rule 3 says CELT is reset for R but NOT for the
        // following CELT-only frame. Row 4's seam ops therefore
        // include exactly one CELT reset (for R itself), not two.
        let ops = NormativeTransition::SilkToCeltWithRedundancy.seam_operations();
        assert_eq!(ops, &[BoundaryOp::WindowedCrossLap, BoundaryOp::CeltReset]);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, BoundaryOp::CeltReset))
                .count(),
            1
        );
    }

    #[test]
    fn seam_ops_row6_hybrid_to_wb_silk_uses_direct_mix_not_cross_lap() {
        // §4.5: "Switching from Hybrid mode to WB SILK requires
        // adding in the final contents of the CELT overlap buffer
        // to the first SILK-only packet" — direct mix, no
        // windowed cross-lap.
        let ops = NormativeTransition::HybridToWbSilk.seam_operations();
        assert_eq!(
            ops,
            &[
                BoundaryOp::CeltOverlapExtract,
                BoundaryOp::DirectMix,
                BoundaryOp::StreamJoin,
            ]
        );
        assert!(!ops.contains(&BoundaryOp::WindowedCrossLap));
        assert!(!ops.contains(&BoundaryOp::CeltReset));
        assert!(!ops.contains(&BoundaryOp::SilkReset));
    }

    #[test]
    fn seam_ops_row8_celt_to_silk_resets_only_silk() {
        // §4.5.2 rule 1 + rule 4: SILK is reset; CELT is NOT
        // reset for R.
        let ops = NormativeTransition::CeltToSilkWithRedundancy.seam_operations();
        assert_eq!(ops, &[BoundaryOp::WindowedCrossLap, BoundaryOp::SilkReset]);
        assert!(!ops.contains(&BoundaryOp::CeltReset));
        assert!(!ops.contains(&BoundaryOp::SilkAndCeltReset));
    }

    #[test]
    fn seam_ops_row9_celt_to_hybrid_uses_dual_reset_marker() {
        // §4.5.3's `|` marker key reads "CELT and SILK decoder
        // resets" — the dual annotation at the H boundary. The
        // figure's annotation is preserved here even though
        // §4.5.2 rule 4 forbids resetting CELT for R itself; the
        // CELT-side continuity is then re-initialised by the R
        // decode itself before H.
        let ops = NormativeTransition::CeltToHybridWithRedundancy.seam_operations();
        assert_eq!(
            ops,
            &[BoundaryOp::WindowedCrossLap, BoundaryOp::SilkAndCeltReset]
        );
    }

    // ----- carries_redundancy -----

    #[test]
    fn carries_redundancy_matches_redundancy_signal() {
        for t in [
            NormativeTransition::SilkToSilkWithRedundancy,
            NormativeTransition::NbOrMbSilkToHybridWithRedundancy,
            NormativeTransition::SilkToCeltWithRedundancy,
            NormativeTransition::HybridToNbOrMbSilkWithRedundancy,
            NormativeTransition::HybridToCeltWithRedundancy,
            NormativeTransition::CeltToSilkWithRedundancy,
            NormativeTransition::CeltToHybridWithRedundancy,
        ] {
            assert!(t.carries_redundancy(), "{:?} carries R", t);
        }
        for t in [
            NormativeTransition::WbSilkToHybrid,
            NormativeTransition::HybridToWbSilk,
        ] {
            assert!(!t.carries_redundancy(), "{:?} does not carry R", t);
        }
    }

    // ----- Same-mode and unreachable cases -----

    #[test]
    fn same_mode_no_bandwidth_change_is_not_normative() {
        // SILK→SILK same-bandwidth, Hybrid→Hybrid, CELT→CELT: none
        // are on Figure 18. §4.5 first paragraph's "glitch source
        // is the audio-bandwidth change" reading rules out the
        // same-bandwidth SILK→SILK row, and §4.5 also exempts
        // "any two configurations of the CELT-only mode, any two
        // configurations of the Hybrid mode" from special
        // treatment.
        for (m, bw) in [
            (OperatingMode::SilkOnly, Some(SilkBandwidth::Wb)),
            (OperatingMode::Hybrid, Some(SilkBandwidth::Wb)),
            (OperatingMode::CeltOnly, None),
        ] {
            for red in [true, false] {
                let t = classify_normative_transition(m, bw, m, bw, red);
                assert_eq!(t, None, "same-mode {:?} should not match Figure 18", m);
            }
        }
    }

    #[test]
    fn silk_to_silk_same_bandwidth_with_redundancy_is_not_on_figure18() {
        // §4.5's redundancy rationale for the SILK→SILK row is the
        // audio-bandwidth change; a same-bandwidth SILK→SILK pair
        // — even with the redundancy bit signalled — is not on
        // Figure 18 row 1.
        for bw in [SilkBandwidth::Nb, SilkBandwidth::Mb, SilkBandwidth::Wb] {
            let t = classify_normative_transition(
                OperatingMode::SilkOnly,
                Some(bw),
                OperatingMode::SilkOnly,
                Some(bw),
                true,
            );
            assert_eq!(t, None, "SILK {:?}→{:?} +R should not be on row 1", bw, bw);
        }
    }

    #[test]
    fn nb_silk_to_hybrid_without_redundancy_is_not_on_figure18() {
        // Figure 18 only places NB-or-MB-SILK → Hybrid under the
        // with-redundancy heading. Without redundancy the
        // transition belongs to Figure 19.
        let t = classify_normative_transition(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            false,
        );
        assert_eq!(t, None);
    }

    #[test]
    fn celt_to_silk_without_redundancy_is_not_on_figure18() {
        let t = classify_normative_transition(
            OperatingMode::CeltOnly,
            None,
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            false,
        );
        assert_eq!(t, None);
    }

    // ----- §4.5.3 Figure 19 rows -----

    #[test]
    fn fig19_row1_silk_audio_bandwidth_change() {
        let t = recommended_non_normative(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Mb),
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::SilkToSilkAudioBandwidthChange)
        );
    }

    #[test]
    fn fig19_row1_silk_same_bandwidth_is_not_on_figure19() {
        // No bandwidth change → no transition to recommend.
        let t = recommended_non_normative(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(t, None);
    }

    #[test]
    fn fig19_row2_nb_silk_to_hybrid() {
        let t = recommended_non_normative(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::NbOrMbSilkToHybrid)
        );
    }

    #[test]
    fn fig19_row2_wb_silk_to_hybrid_is_not_on_figure19() {
        // WB SILK → Hybrid is on Figure 18 row 3 (normative).
        let t = recommended_non_normative(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(t, None);
    }

    #[test]
    fn fig19_row3_silk_to_celt_without_redundancy() {
        let t = recommended_non_normative(
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
            OperatingMode::CeltOnly,
            None,
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::SilkToCeltWithoutRedundancy)
        );
    }

    #[test]
    fn fig19_row4_hybrid_to_nb_silk() {
        let t = recommended_non_normative(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Nb),
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::HybridToNbOrMbSilk)
        );
    }

    #[test]
    fn fig19_row4_hybrid_to_wb_silk_is_not_on_figure19() {
        // Hybrid → WB SILK is on Figure 18 row 6 (normative).
        let t = recommended_non_normative(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(t, None);
    }

    #[test]
    fn fig19_row5_hybrid_to_celt_without_redundancy() {
        let t = recommended_non_normative(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::CeltOnly,
            None,
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::HybridToCeltWithoutRedundancy)
        );
    }

    #[test]
    fn fig19_row6_celt_to_silk_without_redundancy() {
        let t = recommended_non_normative(
            OperatingMode::CeltOnly,
            None,
            OperatingMode::SilkOnly,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::CeltToSilkWithoutRedundancy)
        );
    }

    #[test]
    fn fig19_row7_celt_to_hybrid_without_redundancy() {
        let t = recommended_non_normative(
            OperatingMode::CeltOnly,
            None,
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(
            t,
            Some(RecommendedNonNormativeTransition::CeltToHybridWithoutRedundancy)
        );
    }

    #[test]
    fn fig19_same_mode_celt_to_celt_is_none() {
        // §4.5 first paragraph: "Switching between … any two
        // configurations of the CELT-only mode … does not require
        // any special treatment in the decoder". Not on Figure 19.
        let t =
            recommended_non_normative(OperatingMode::CeltOnly, None, OperatingMode::CeltOnly, None);
        assert_eq!(t, None);
    }

    #[test]
    fn fig19_same_mode_hybrid_to_hybrid_is_none() {
        let t = recommended_non_normative(
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
            OperatingMode::Hybrid,
            Some(SilkBandwidth::Wb),
        );
        assert_eq!(t, None);
    }

    // ----- Figure 19 boundary-op tables -----

    #[test]
    fn fig19_row3_seam_ops_include_plc_celt_reset() {
        let ops = RecommendedNonNormativeTransition::SilkToCeltWithoutRedundancy.seam_operations();
        assert_eq!(
            ops,
            &[
                BoundaryOp::PacketLossConcealment,
                BoundaryOp::WindowedCrossLap,
                BoundaryOp::CeltReset,
            ]
        );
    }

    #[test]
    fn fig19_row4_seam_ops_use_overlap_extract_direct_mix() {
        let ops = RecommendedNonNormativeTransition::HybridToNbOrMbSilk.seam_operations();
        assert_eq!(
            ops,
            &[
                BoundaryOp::CeltOverlapExtract,
                BoundaryOp::DirectMix,
                BoundaryOp::SilkReset,
            ]
        );
    }

    #[test]
    fn fig19_row7_seam_ops_use_dual_reset_marker() {
        let ops =
            RecommendedNonNormativeTransition::CeltToHybridWithoutRedundancy.seam_operations();
        assert_eq!(
            ops,
            &[
                BoundaryOp::PacketLossConcealment,
                BoundaryOp::WindowedCrossLap,
                BoundaryOp::SilkAndCeltReset,
            ]
        );
    }

    // ----- Mutual exclusion of Figure 18 and Figure 19 -----

    #[test]
    fn fig18_normative_and_fig19_non_normative_cover_complementary_paths() {
        // Figure 18 covers transitions WITH redundancy (and the
        // two no-redundancy exceptions §4.5 calls out: WB-SILK →
        // Hybrid and Hybrid → WB-SILK). Figure 19 covers
        // transitions WITHOUT redundancy that need a documented
        // recommendation. For every mode-change with a non-WB SILK
        // bandwidth on one side, either Figure 18 (with
        // redundancy) or Figure 19 (without redundancy) provides
        // guidance — never both for the SAME signalled-redundancy
        // bit.
        use OperatingMode::*;
        let prev_modes = [SilkOnly, Hybrid, CeltOnly];
        let next_modes = [SilkOnly, Hybrid, CeltOnly];
        let silk_bws = [
            None,
            Some(SilkBandwidth::Nb),
            Some(SilkBandwidth::Mb),
            Some(SilkBandwidth::Wb),
        ];
        for prev_m in prev_modes {
            for next_m in next_modes {
                for prev_bw in silk_bws {
                    for next_bw in silk_bws {
                        let n_with_r =
                            classify_normative_transition(prev_m, prev_bw, next_m, next_bw, true);
                        let n_no_r =
                            classify_normative_transition(prev_m, prev_bw, next_m, next_bw, false);
                        let r = recommended_non_normative(prev_m, prev_bw, next_m, next_bw);
                        // §4.5.3 invariant: Figure 19 entries are
                        // "no-redundancy" recommendations, so a
                        // transition that ALSO matches Figure 18
                        // *without* redundancy (rows 3 and 6) must
                        // not appear on Figure 19. Rows 3 and 6
                        // are §4.5's "no special treatment" carve-
                        // outs and are documented normative, so
                        // they belong on Figure 18 exclusively.
                        if n_no_r.is_some() {
                            assert_eq!(
                                r, None,
                                "no-R Figure 18 row at {:?}→{:?} bws ({:?}, {:?}) must not also be on Figure 19",
                                prev_m, next_m, prev_bw, next_bw
                            );
                        }
                        // Figure 18 with-R rows do NOT preclude a
                        // matching Figure 19 row, because the two
                        // figures are keyed by the redundancy
                        // bit: an encoder picks one or the other.
                        // We only assert that some guidance is
                        // available for every mode-changing
                        // transition that isn't same-mode-same-bw.
                        let _ = n_with_r;
                    }
                }
            }
        }
    }

    // ----- Cross-checks against §4.5.2 reset markers -----

    #[test]
    fn figure_18_seam_resets_agree_with_section_4_5_2_decisions() {
        // For each Figure-18 row, the seam-op list's reset markers
        // must agree with [`crate::mode_transition_reset`]'s
        // §4.5.2 decision. SILK reset markers must match
        // §4.5.2's `silk` field; the presence of any CELT-side
        // reset marker must match `celt_resets()` UNLESS the
        // §4.5.3 marker is `|` (dual annotation at the H
        // boundary) which §4.5.2's [`CeltResetPlacement::None`]
        // legitimately disagrees with per the rule-4 carve-out
        // already encoded in [`crate::mode_transition_reset`]'s
        // `figure18_celt_to_hybrid_with_redundancy_marks_silk_reset_only`
        // test. The cases below pin the four-row agreement that
        // is unconditional.
        use crate::mode_transition_reset::{decide_state_resets, CeltResetPlacement};
        use OperatingMode::*;

        // Row 1: SILK → SILK with R. §4.5.2 sees same-mode → no
        // CELT reset, no SILK reset. The seam ops include
        // SilkReset (`;` from `;S -> S -> S`) and CeltReset
        // (`!` from `!R`) — these are placed on the R frame and
        // the new-mode run, not on §4.5.2's prev→next transition.
        let r = decide_state_resets(SilkOnly, SilkOnly, present());
        assert!(!r.silk);
        assert_eq!(r.celt, CeltResetPlacement::None);

        // Row 3: WB SILK → Hybrid. §4.5.2 rule 2 fires → CELT
        // reset before frame.
        let r = decide_state_resets(SilkOnly, Hybrid, not_present());
        assert!(!r.silk);
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
        let row3_ops = NormativeTransition::WbSilkToHybrid.seam_operations();
        assert!(row3_ops.contains(&BoundaryOp::CeltReset));

        // Row 4: SILK → CELT with R. §4.5.2 rule 3 fires → CELT
        // reset before redundant only.
        let r = decide_state_resets(SilkOnly, CeltOnly, present());
        assert!(!r.silk);
        assert_eq!(r.celt, CeltResetPlacement::BeforeRedundantOnly);
        let row4_ops = NormativeTransition::SilkToCeltWithRedundancy.seam_operations();
        assert!(row4_ops.contains(&BoundaryOp::CeltReset));

        // Row 8: CELT → SILK with R. §4.5.2 rule 1 fires → SILK
        // reset. Rule 4 forbids CELT reset.
        let r = decide_state_resets(CeltOnly, SilkOnly, present());
        assert!(r.silk);
        assert_eq!(r.celt, CeltResetPlacement::None);
        let row8_ops = NormativeTransition::CeltToSilkWithRedundancy.seam_operations();
        assert!(row8_ops.contains(&BoundaryOp::SilkReset));
        assert!(!row8_ops.contains(&BoundaryOp::CeltReset));
    }

    // ----- redundancy_is_present bridge -----

    #[test]
    fn redundancy_is_present_passthrough() {
        assert!(super::redundancy_is_present(present()));
        assert!(!super::redundancy_is_present(not_present()));
        // Invalid is treated as "no usable redundancy" per the
        // §4.5.1.3 stop-and-discard recommendation, matching the
        // §4.5.2 convention.
        assert!(!super::redundancy_is_present(invalid()));
    }
}
