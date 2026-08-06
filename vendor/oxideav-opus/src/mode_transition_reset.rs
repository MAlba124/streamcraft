//! SILK + CELT decoder state-reset policy across mode transitions
//! (RFC 6716 §4.5.2 "State Reset", p. 127).
//!
//! Round 26 (`celt_redundancy`) decoded the §4.5.1 redundancy-flag
//! metadata at the tail of every SILK-only or Hybrid Opus frame.
//! §4.5.2 picks up at the next normative step: deciding, given the
//! previous frame's mode, the current frame's mode, and whether the
//! transition carries a redundant CELT frame, which of the two
//! sub-decoders (SILK, CELT) must be reset, and — when a redundant
//! CELT frame is present — whether that reset happens *before* the
//! redundant frame or *between* the redundant frame and the main
//! frame.
//!
//! ## Rule transcription
//!
//! The four normative sentences of §4.5.2 are:
//!
//! 1. *"The SILK state is reset before every SILK-only or Hybrid
//!    frame where the previous frame was CELT-only."*
//! 2. *"The CELT state is reset every time the operating mode changes
//!    and the new mode is either Hybrid or CELT-only, except when the
//!    transition uses redundancy as described above."*
//! 3. *"When switching from SILK-only or Hybrid to CELT-only with
//!    redundancy, the CELT state is reset before decoding the
//!    redundant CELT frame embedded in the SILK-only or Hybrid frame,
//!    but it is not reset before decoding the following CELT-only
//!    frame."*
//! 4. *"When switching from CELT-only mode to SILK-only or Hybrid
//!    mode with redundancy, the CELT decoder is not reset for
//!    decoding the redundant CELT frame."*
//!
//! ## The "operating mode changes" predicate
//!
//! Rule 2 keys off whether the operating mode actually changes. Two
//! consecutive Hybrid frames, or two consecutive CELT-only frames, do
//! NOT trigger a CELT reset under rule 2: §4.5.2 only resets CELT
//! "every time the operating mode changes and the new mode is either
//! Hybrid or CELT-only", and a mode that did not change is the
//! complement of "the operating mode changes". The same predicate
//! also gates rule 1 (the SILK-reset clause), where "where the
//! previous frame was CELT-only" implicitly requires the current
//! frame to NOT be CELT-only (a CELT-only → CELT-only sequence does
//! not match because the *current* frame would not be SILK-only or
//! Hybrid).
//!
//! ## The redundancy exception
//!
//! Rules 2 + 3 carve out three sub-cases that need to be tracked
//! separately because they place the reset boundary differently:
//!
//! * **SILK/Hybrid → CELT-only WITH redundancy.** Rule 3 places the
//!   CELT reset *before the redundant CELT frame*, then keeps state
//!   live across the redundant frame into the following CELT-only
//!   frame ("but it is not reset before decoding the following
//!   CELT-only frame"). [`CeltResetPlacement::BeforeRedundantOnly`]
//!   captures this.
//! * **CELT-only → SILK-only/Hybrid WITH redundancy.** Rule 4
//!   forbids resetting CELT before the redundant CELT frame. SILK
//!   resets per rule 1 (since the predecessor was CELT-only and the
//!   current frame is SILK-only or Hybrid). The CELT decoder is not
//!   reset in this transition at all from §4.5.2's standpoint.
//!   [`CeltResetPlacement::None`].
//! * **Hybrid → Hybrid, CELT-only → CELT-only.** Mode does not
//!   change; rule 2 does not fire even when redundancy happens to be
//!   present. [`CeltResetPlacement::None`].
//!
//! All other mode-changing transitions to Hybrid or CELT-only fall
//! under the rule-2 default: [`CeltResetPlacement::BeforeFrame`].
//!
//! ## Provenance
//!
//! Every clause, every transition outcome, the carve-out for
//! redundancy, and the "before-redundant vs. before-frame" placement
//! is transcribed from RFC 6716 §4.5.2 in
//! `docs/audio/opus/rfc6716-opus.txt` (pp. 127). No external library
//! source was consulted. The companion §4.5.3 transition figure
//! (p. 128) is a non-normative summary and was not used to seed any
//! new rules — only as a cross-check that the four rules above
//! reproduce the figure's reset markers (`;` SILK reset, `|` SILK +
//! CELT resets).

use crate::celt_redundancy::RedundancyDecision;
use crate::framing::OperatingMode;

/// Where the CELT decoder reset (if any) is placed during a
/// transition, per RFC 6716 §4.5.2.
///
/// A transition that triggers a CELT reset may need to apply it
/// *before* the main frame (the rule-2 default), or *before the
/// redundant CELT frame* but not before the following main CELT-only
/// frame (the rule-3 carve-out for SILK/Hybrid → CELT-only with
/// redundancy). When no CELT reset fires at all, this is
/// [`CeltResetPlacement::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeltResetPlacement {
    /// §4.5.2 does not require a CELT decoder reset for this
    /// transition. Either the operating mode did not change, or the
    /// transition is CELT-only → SILK/Hybrid with redundancy where
    /// rule 4 explicitly forbids resetting the CELT decoder for the
    /// redundant frame.
    None,
    /// Rule-2 default: reset CELT immediately before decoding the
    /// new-mode frame. Applies to every mode-changing transition
    /// whose new mode is Hybrid or CELT-only and that does NOT carry
    /// a §4.5.1 redundant CELT frame.
    BeforeFrame,
    /// Rule-3 carve-out: SILK-only/Hybrid → CELT-only with
    /// redundancy. The CELT decoder is reset before decoding the
    /// 5 ms redundant CELT frame embedded in the trailing bytes of
    /// the previous SILK-only/Hybrid frame, and is NOT reset before
    /// the following CELT-only frame.
    BeforeRedundantOnly,
}

/// The outcome of §4.5.2 for one frame transition.
///
/// A SILK reset is a single bit (rule 1 either fires or it does
/// not); the CELT reset carries placement information per
/// [`CeltResetPlacement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateReset {
    /// Whether the SILK decoder must be reset before decoding the
    /// current frame, per rule 1.
    pub silk: bool,
    /// Where the CELT decoder reset (if any) is placed, per rules
    /// 2–4.
    pub celt: CeltResetPlacement,
}

impl StateReset {
    /// Convenience: `true` iff [`Self::celt`] is anything other than
    /// [`CeltResetPlacement::None`].
    pub fn celt_resets(&self) -> bool {
        !matches!(self.celt, CeltResetPlacement::None)
    }

    /// Convenience: `true` iff neither sub-decoder needs to be reset
    /// for this transition.
    pub fn is_noop(&self) -> bool {
        !self.silk && !self.celt_resets()
    }
}

/// Whether the upcoming transition carries a §4.5.1 redundant CELT
/// frame, distilled from a [`RedundancyDecision`].
///
/// §4.5.2 only cares about the *presence* of redundancy, not its
/// position (Table 65 / §4.5.1.2) or its size (§4.5.1.3). An
/// `Invalid` decision means the §4.5.1.3 size claim overflowed the
/// remaining bytes; the §4.5.1.3 RECOMMENDATION is to stop decoding
/// the Opus frame entirely, so for the §4.5.2 reset-policy step the
/// transition is treated as having NO usable redundant frame.
fn redundancy_is_present(decision: RedundancyDecision) -> bool {
    decision.is_present()
}

/// Compute the §4.5.2 state-reset policy for a transition from
/// `prev_mode` to `next_mode`, given the §4.5.1 redundancy decision
/// already taken on the *previous* frame (i.e. the SILK-only or
/// Hybrid frame that may have appended a 5 ms redundant CELT frame
/// to its tail) OR on the *current* frame (when the current frame is
/// SILK-only or Hybrid and the previous frame was CELT-only with a
/// redundant trailer — §4.5.1 places that redundancy on the *frame
/// whose tail it lives in*, which is always the SILK-only or Hybrid
/// side of the transition).
///
/// The caller passes [`RedundancyDecision::NotPresent`] when no
/// redundant CELT frame is part of the transition; this is the safe
/// default on the very first frame of a stream and on any transition
/// the §4.5.1 dispatcher routes around.
///
/// ## Mapping of the four §4.5.2 rules
///
/// * **Rule 1** is `next ∈ {SilkOnly, Hybrid} && prev == CeltOnly`.
/// * **Rule 2** is `next ∈ {Hybrid, CeltOnly} && prev != next`
///   unless rule 3 or rule 4 overrides.
/// * **Rule 3** overrides rule 2 when `prev ∈ {SilkOnly, Hybrid}`
///   carried an end-position redundant CELT frame into a mode change
///   whose new mode is Hybrid or CELT-only: the CELT reset moves from
///   "before the new-mode frame" to "before the redundant CELT frame
///   only" ([`CeltResetPlacement::BeforeRedundantOnly`]) — §4.5.3
///   Figure 18's `!R` rows (SILK/Hybrid → CELT with redundancy, and
///   NB/MB SILK → Hybrid with redundancy, whose Hybrid frame takes
///   only the SILK reset `;H`).
/// * **Rule 4** applies when `prev == CeltOnly` and
///   `next ∈ {SilkOnly, Hybrid}` with redundancy: the CELT decoder is
///   NOT reset *for the embedded redundant frame* (it decodes on the
///   carried state). For `next == SilkOnly` no frame-level CELT reset
///   exists anyway ([`CeltResetPlacement::None`]); for
///   `next == Hybrid` the frame-level reset still applies to the main
///   CELT layer (Figure 18 `|H`), ordered *after* the redundant
///   frame's decode by the driver.
pub fn decide_state_resets(
    prev_mode: OperatingMode,
    next_mode: OperatingMode,
    redundancy: RedundancyDecision,
) -> StateReset {
    let silk = silk_reset_required(prev_mode, next_mode);
    let celt = celt_reset_placement(prev_mode, next_mode, redundancy_is_present(redundancy));
    StateReset { silk, celt }
}

/// Rule 1: SILK is reset before every SILK-only or Hybrid frame
/// whose predecessor was CELT-only.
fn silk_reset_required(prev_mode: OperatingMode, next_mode: OperatingMode) -> bool {
    matches!(next_mode, OperatingMode::SilkOnly | OperatingMode::Hybrid)
        && matches!(prev_mode, OperatingMode::CeltOnly)
}

/// Rules 2–4: CELT placement.
fn celt_reset_placement(
    prev_mode: OperatingMode,
    next_mode: OperatingMode,
    redundancy_present: bool,
) -> CeltResetPlacement {
    // §4.5.2 rule 2 keys off "the operating mode changes AND the new
    // mode is either Hybrid or CELT-only". Same-mode transitions are
    // out of scope for rule 2 entirely; rules 3 and 4 are also
    // mode-changing rules, so any same-mode transition resolves to
    // None regardless of redundancy.
    if prev_mode == next_mode {
        return CeltResetPlacement::None;
    }

    match (prev_mode, next_mode) {
        // Rule 4: CELT-only → SILK-only. Rule 2 does not fire because
        // the new mode is SILK-only (which is not "Hybrid or
        // CELT-only"). Rule 4's "not reset for the redundant frame"
        // clause is the same outcome either way for the CELT side.
        // SILK still resets per rule 1, handled separately.
        (OperatingMode::CeltOnly, OperatingMode::SilkOnly) => CeltResetPlacement::None,

        // CELT-only → Hybrid. Rule 2 fires either way — §4.5.3
        // Figure 18 marks the Hybrid frame `|H` (CELT and SILK
        // resets) even in the "CELT to Hybrid with Redundancy" row.
        // Rule 4's "the CELT decoder is not reset for decoding the
        // redundant CELT frame" governs only the *embedded redundant
        // frame*, which decodes on the carried state *before* the
        // reset is applied to the Hybrid frame's main CELT layer (the
        // driver orders redundant-decode → reset → main layer).
        (OperatingMode::CeltOnly, OperatingMode::Hybrid) => CeltResetPlacement::BeforeFrame,

        // Rule 3 carve-out vs. rule-2 default for SILK-only/Hybrid →
        // CELT-only.
        (OperatingMode::SilkOnly | OperatingMode::Hybrid, OperatingMode::CeltOnly) => {
            if redundancy_present {
                CeltResetPlacement::BeforeRedundantOnly
            } else {
                CeltResetPlacement::BeforeFrame
            }
        }

        // SILK-only → Hybrid. Rule 2: the new mode is Hybrid → reset —
        // "except when the transition uses redundancy": §4.5.3
        // Figure 18's "NB or MB SILK to Hybrid with Redundancy" row is
        // `!R → ;H` — the CELT reset (`!`) is placed before the
        // end-position redundant CELT frame carried by the last SILK
        // frame, and the Hybrid frame takes only the SILK reset (`;`),
        // its CELT layer continuing from the redundant frame's warmed
        // state.
        (OperatingMode::SilkOnly, OperatingMode::Hybrid) => {
            if redundancy_present {
                CeltResetPlacement::BeforeRedundantOnly
            } else {
                CeltResetPlacement::BeforeFrame
            }
        }
        // Hybrid → SILK-only. Rule 2 does not fire (SILK-only is not
        // "Hybrid or CELT-only"); redundancy plays no role.
        (OperatingMode::Hybrid, OperatingMode::SilkOnly) => CeltResetPlacement::None,

        // Same-mode pairs are filtered above; the match is otherwise
        // exhaustive over the 3×3 product.
        (OperatingMode::SilkOnly, OperatingMode::SilkOnly)
        | (OperatingMode::Hybrid, OperatingMode::Hybrid)
        | (OperatingMode::CeltOnly, OperatingMode::CeltOnly) => CeltResetPlacement::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn not_present() -> RedundancyDecision {
        RedundancyDecision::NotPresent
    }

    fn present() -> RedundancyDecision {
        RedundancyDecision::Present {
            position: crate::celt_redundancy::RedundancyPosition::End,
            size_bytes: 2,
        }
    }

    fn invalid() -> RedundancyDecision {
        RedundancyDecision::Invalid
    }

    // ----- StateReset helpers -----

    #[test]
    fn state_reset_celt_resets_is_true_when_celt_resets() {
        let r = StateReset {
            silk: false,
            celt: CeltResetPlacement::BeforeFrame,
        };
        assert!(r.celt_resets());
        let r2 = StateReset {
            silk: true,
            celt: CeltResetPlacement::BeforeRedundantOnly,
        };
        assert!(r2.celt_resets());
    }

    #[test]
    fn state_reset_celt_resets_is_false_when_celt_none() {
        let r = StateReset {
            silk: true,
            celt: CeltResetPlacement::None,
        };
        assert!(!r.celt_resets());
    }

    #[test]
    fn state_reset_is_noop_only_when_both_false() {
        let r = StateReset {
            silk: false,
            celt: CeltResetPlacement::None,
        };
        assert!(r.is_noop());

        assert!(!StateReset {
            silk: true,
            celt: CeltResetPlacement::None
        }
        .is_noop());
        assert!(!StateReset {
            silk: false,
            celt: CeltResetPlacement::BeforeFrame
        }
        .is_noop());
        assert!(!StateReset {
            silk: false,
            celt: CeltResetPlacement::BeforeRedundantOnly
        }
        .is_noop());
    }

    // ----- Rule 1 (SILK) -----

    #[test]
    fn rule1_silk_resets_celt_to_silk() {
        let r = decide_state_resets(
            OperatingMode::CeltOnly,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert!(r.silk, "SILK must reset on CELT-only → SILK-only");
    }

    #[test]
    fn rule1_silk_resets_celt_to_hybrid() {
        let r = decide_state_resets(
            OperatingMode::CeltOnly,
            OperatingMode::Hybrid,
            not_present(),
        );
        assert!(r.silk, "SILK must reset on CELT-only → Hybrid");
    }

    #[test]
    fn rule1_silk_does_not_reset_silk_to_silk() {
        let r = decide_state_resets(
            OperatingMode::SilkOnly,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert!(!r.silk);
    }

    #[test]
    fn rule1_silk_does_not_reset_hybrid_to_silk() {
        let r = decide_state_resets(
            OperatingMode::Hybrid,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert!(!r.silk);
    }

    #[test]
    fn rule1_silk_does_not_reset_when_next_is_celtonly() {
        // The predicate requires next ∈ {SilkOnly, Hybrid}; CELT-only
        // next never fires rule 1.
        for prev in [
            OperatingMode::SilkOnly,
            OperatingMode::Hybrid,
            OperatingMode::CeltOnly,
        ] {
            let r = decide_state_resets(prev, OperatingMode::CeltOnly, not_present());
            assert!(!r.silk, "rule 1 must not fire when next is CELT-only");
        }
    }

    #[test]
    fn rule1_silk_reset_is_independent_of_redundancy_presence() {
        // §4.5.2 rule 1 has no redundancy clause; the bit fires
        // identically regardless of whether the transition carries a
        // redundant CELT frame.
        for red in [not_present(), present(), invalid()] {
            let r = decide_state_resets(OperatingMode::CeltOnly, OperatingMode::SilkOnly, red);
            assert!(r.silk);
        }
    }

    // ----- Rule 2 (CELT default, no redundancy) -----

    #[test]
    fn rule2_celt_resets_silk_to_hybrid() {
        let r = decide_state_resets(
            OperatingMode::SilkOnly,
            OperatingMode::Hybrid,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
    }

    #[test]
    fn rule2_celt_resets_silk_to_celtonly_without_redundancy() {
        let r = decide_state_resets(
            OperatingMode::SilkOnly,
            OperatingMode::CeltOnly,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
    }

    #[test]
    fn rule2_celt_resets_hybrid_to_celtonly_without_redundancy() {
        let r = decide_state_resets(
            OperatingMode::Hybrid,
            OperatingMode::CeltOnly,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
    }

    #[test]
    fn rule2_celt_does_not_reset_hybrid_to_silk() {
        // New mode is SILK-only, which is neither Hybrid nor CELT-only.
        let r = decide_state_resets(
            OperatingMode::Hybrid,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::None);
    }

    #[test]
    fn rule2_celt_does_not_reset_when_mode_unchanged() {
        // "every time the operating mode changes" — same-mode is out
        // of scope.
        for mode in [
            OperatingMode::SilkOnly,
            OperatingMode::Hybrid,
            OperatingMode::CeltOnly,
        ] {
            for red in [not_present(), present(), invalid()] {
                let r = decide_state_resets(mode, mode, red);
                assert_eq!(
                    r.celt,
                    CeltResetPlacement::None,
                    "same-mode {:?} must not reset CELT under any redundancy state",
                    mode
                );
            }
        }
    }

    // ----- Rule 3 (SILK/Hybrid → CELT-only with redundancy) -----

    #[test]
    fn rule3_silk_to_celtonly_with_redundancy_places_reset_before_redundant() {
        let r = decide_state_resets(OperatingMode::SilkOnly, OperatingMode::CeltOnly, present());
        assert_eq!(r.celt, CeltResetPlacement::BeforeRedundantOnly);
    }

    #[test]
    fn rule3_hybrid_to_celtonly_with_redundancy_places_reset_before_redundant() {
        let r = decide_state_resets(OperatingMode::Hybrid, OperatingMode::CeltOnly, present());
        assert_eq!(r.celt, CeltResetPlacement::BeforeRedundantOnly);
    }

    #[test]
    fn rule3_invalid_redundancy_falls_back_to_rule2_default() {
        // §4.5.1.3 RECOMMENDS the decoder stop on an Invalid decision;
        // for the §4.5.2 step we treat it as no usable redundancy, so
        // the rule-2 default placement applies.
        let r = decide_state_resets(OperatingMode::SilkOnly, OperatingMode::CeltOnly, invalid());
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
    }

    // ----- Rule 4 (CELT-only → SILK/Hybrid with redundancy) -----

    #[test]
    fn rule4_celt_to_silk_with_redundancy_does_not_reset_celt() {
        let r = decide_state_resets(OperatingMode::CeltOnly, OperatingMode::SilkOnly, present());
        assert_eq!(r.celt, CeltResetPlacement::None);
        // SILK still resets per rule 1.
        assert!(r.silk);
    }

    #[test]
    fn rule4_celt_to_hybrid_with_redundancy_resets_celt_before_main_layer() {
        // §4.5.3 Figure 18 "CELT to Hybrid with Redundancy" marks the
        // Hybrid frame `|H` (CELT and SILK resets). Rule 4 only
        // exempts the *embedded redundant frame*, which the driver
        // decodes on the carried state before applying this reset to
        // the main CELT layer.
        let r = decide_state_resets(OperatingMode::CeltOnly, OperatingMode::Hybrid, present());
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
        // SILK still resets per rule 1.
        assert!(r.silk);
    }

    #[test]
    fn rule4_celt_to_silk_without_redundancy_also_does_not_reset_celt() {
        // Rule 2's "new mode is either Hybrid or CELT-only" gate
        // already prevents CELT reset when the new mode is SILK-only,
        // so the result matches with-redundancy and without-redundancy
        // alike for CELT-only → SILK-only.
        let r = decide_state_resets(
            OperatingMode::CeltOnly,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::None);
    }

    #[test]
    fn rule4_celt_to_hybrid_without_redundancy_still_resets_celt() {
        // Without redundancy, rule 2 fires: mode changes AND new mode
        // is Hybrid → CELT resets before the new-mode frame.
        let r = decide_state_resets(
            OperatingMode::CeltOnly,
            OperatingMode::Hybrid,
            not_present(),
        );
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
        assert!(r.silk);
    }

    // ----- Full 3×3 × 2-redundancy cross-product cross-check -----

    #[test]
    fn full_transition_matrix_is_consistent() {
        // The 9 mode pairs × {present, not_present} cover the §4.5.2
        // decision space; this test pins every outcome explicitly so
        // any future code change has to update the table.
        use CeltResetPlacement as P;
        use OperatingMode::{CeltOnly as C, Hybrid as H, SilkOnly as S};

        // (prev, next, redundancy_present) => (silk, celt)
        let cases: &[(OperatingMode, OperatingMode, bool, bool, P)] = &[
            // Same-mode rows: no reset, regardless of redundancy.
            (S, S, false, false, P::None),
            (S, S, true, false, P::None),
            (H, H, false, false, P::None),
            (H, H, true, false, P::None),
            (C, C, false, false, P::None),
            (C, C, true, false, P::None),
            // SILK-only ↔ Hybrid. With redundancy the CELT reset
            // belongs before the previous SILK frame's end-position
            // redundant CELT frame (Figure 18 `!R → ;H`).
            (S, H, false, false, P::BeforeFrame),
            (S, H, true, false, P::BeforeRedundantOnly),
            (H, S, false, false, P::None),
            (H, S, true, false, P::None),
            // SILK-only/Hybrid → CELT-only.
            (S, C, false, false, P::BeforeFrame),
            (S, C, true, false, P::BeforeRedundantOnly),
            (H, C, false, false, P::BeforeFrame),
            (H, C, true, false, P::BeforeRedundantOnly),
            // CELT-only → SILK-only/Hybrid. The `|H` row of Figure 18
            // keeps the main-layer CELT reset even with redundancy
            // (the embedded redundant frame decodes before it).
            (C, S, false, true, P::None),
            (C, S, true, true, P::None),
            (C, H, false, true, P::BeforeFrame),
            (C, H, true, true, P::BeforeFrame),
        ];

        for (prev, next, red_present, expected_silk, expected_celt) in cases {
            let red = if *red_present {
                present()
            } else {
                not_present()
            };
            let r = decide_state_resets(*prev, *next, red);
            assert_eq!(
                r.silk, *expected_silk,
                "silk mismatch on prev={:?}, next={:?}, red={}",
                prev, next, red_present
            );
            assert_eq!(
                r.celt, *expected_celt,
                "celt mismatch on prev={:?}, next={:?}, red={}",
                prev, next, red_present
            );
        }
    }

    // ----- §4.5.3 figure cross-checks -----
    //
    // The §4.5.3 transition figure (p. 128) is non-normative but
    // marks decoder resets with `;` (SILK reset only) and `|` (SILK
    // + CELT resets). The cases below pin the four §4.5.3 entries
    // that depend on the §4.5.2 carve-outs; matching them
    // independently validates that the rule transcription does not
    // contradict the figure's summary.

    #[test]
    fn figure18_silk_to_celt_with_redundancy_resets_celt_before_redundant() {
        // §4.5.3 row "SILK to CELT with Redundancy": the redundant R
        // sits inside the last SILK frame's tail, then a C → C → C
        // run follows. §4.5.2 rule 3 places CELT reset before the
        // redundant frame and NOT before the following C frame.
        let r = decide_state_resets(OperatingMode::SilkOnly, OperatingMode::CeltOnly, present());
        assert_eq!(r.celt, CeltResetPlacement::BeforeRedundantOnly);
        assert!(!r.silk, "SILK is not reset entering CELT-only");
    }

    #[test]
    fn figure18_celt_to_silk_with_redundancy_marks_silk_reset_only() {
        // §4.5.3 row "CELT to SILK with Redundancy": the `;` marker
        // before the S → S → S run indicates a SILK reset only —
        // rule 1 fires, rule 4 suppresses any CELT reset for the
        // redundant frame.
        let r = decide_state_resets(OperatingMode::CeltOnly, OperatingMode::SilkOnly, present());
        assert!(r.silk);
        assert_eq!(r.celt, CeltResetPlacement::None);
    }

    #[test]
    fn figure18_celt_to_hybrid_with_redundancy_marks_both_resets() {
        // §4.5.3 row "CELT to Hybrid with Redundancy": the `|` marker
        // is "CELT and SILK decoder resets" in the key — both resets
        // apply at the H frame. Rule 4's "the CELT decoder is not
        // reset for decoding the redundant CELT frame" governs only
        // the embedded redundant frame, which decodes on the carried
        // state *before* the frame-level reset (the driver orders
        // redundant-decode → reset → main CELT layer).
        let r = decide_state_resets(OperatingMode::CeltOnly, OperatingMode::Hybrid, present());
        assert!(r.silk);
        assert_eq!(r.celt, CeltResetPlacement::BeforeFrame);
    }

    #[test]
    fn figure18_hybrid_to_wb_silk_resets_silk_only_under_rule_layer() {
        // §4.5.3 row "Hybrid to WB SILK": no redundancy is used; the
        // figure shows `c` (overlap buffer) carried into a `>` join
        // with the SILK frames. From §4.5.2, neither rule 1 (prev
        // was Hybrid, not CELT-only) nor rule 2 (new mode is
        // SILK-only) fires. The reset table is empty.
        let r = decide_state_resets(
            OperatingMode::Hybrid,
            OperatingMode::SilkOnly,
            not_present(),
        );
        assert!(r.is_noop());
    }

    // ----- redundancy_is_present helper -----

    #[test]
    fn redundancy_is_present_treats_invalid_as_absent() {
        assert!(!super::redundancy_is_present(RedundancyDecision::Invalid));
        assert!(!super::redundancy_is_present(
            RedundancyDecision::NotPresent
        ));
        assert!(super::redundancy_is_present(RedundancyDecision::Present {
            position: crate::celt_redundancy::RedundancyPosition::End,
            size_bytes: 2,
        }));
    }
}
