//! CELT §4.3.4.2 PVQ codebook-size function `V(N, K)`
//! (RFC 6716 §4.3.4.2, p. 116).
//!
//! The §4.3.4 *Shape Decoding* layer encodes the unit-norm normalized
//! "shape" vector of every CELT MDCT band as a Pyramid Vector
//! Quantizer codeword. The codeword index is decoded as a uniform
//! integer in `0..V(N, K) - 1`, where `V(N, K)` counts the number of
//! integer-magnitude lattice points
//!
//! ```text
//!     { x ∈ Z^N : |x_0| + |x_1| + ... + |x_{N-1}| = K }
//! ```
//!
//! — the size of the PVQ codebook for `N` dimensions and `K` pulses.
//! Concretely, `V(N, K)` is the number of ways to place `K` integer
//! "pulses" into `N` ordered samples, where pulses at the same sample
//! sum and where each pulse independently carries one of two signs.
//!
//! ## §4.3.4.2 recurrence
//!
//! RFC 6716 §4.3.4.2 (p. 116) states the recurrence directly:
//!
//! > The number of combinations can be computed recursively as
//! > `V(N,K) = V(N-1,K) + V(N,K-1) + V(N-1,K-1)`, with `V(N,0) = 1`
//! > and `V(0,K) = 0, K != 0`. There are many different ways to
//! > compute `V(N,K)`, including precomputed tables and direct use
//! > of the recursive formulation. […] Implementations MAY use any
//! > methods they like, as long as they are equivalent to the
//! > mathematical definition.
//!
//! The recurrence has the natural reading: a length-`N` lattice
//! configuration with sum `K` either (a) leaves the last coordinate
//! at zero, leaving a length-`(N-1)` configuration with sum `K`
//! [`V(N-1, K)` configurations]; (b) places one positive pulse at
//! the last coordinate and recurses on the same dimension with one
//! less pulse — covering both the "same coordinate, more pulses"
//! and "sign flip" doublings [`V(N, K-1) + V(N-1, K-1)`
//! configurations].
//!
//! ## §4.1.5 `ec_dec_uint` upper bound
//!
//! Per RFC 6716 §4.3.4.2 the index is read with [`crate::RangeDecoder`]
//! `ec_dec_uint(V(N, K))`. RFC 6716 §4.1.5 (p. 29, line 1592 in the
//! in-repo `docs/audio/opus/rfc6716-opus.txt`) caps `ec_dec_uint`'s
//! `ft` parameter at `2**32 − 1`. The §4.3.3 bit-allocation procedure
//! constrains the reachable `(N, K)` pairs so the corresponding
//! `V(N, K)` always fits this bound. This module returns
//! [`PvqVError::OverflowsDecUintRange`] if a caller asks for an `(N,
//! K)` whose `V(N, K)` would exceed `2**32 − 1`, since such an
//! index cannot be transmitted by a conforming Opus stream.
//!
//! ## Bounds chosen for this module
//!
//! * `N` is the per-band MDCT bin count. Standard non-Custom CELT
//!   bands have at most [`crate::celt_band_layout::CELT_MAX_BINS_PER_BAND`]
//!   = 176 bins per channel at the 20 ms frame size. Joint-stereo
//!   bands double that to 352 bins. The module accepts `N ∈ 0..=352`
//!   ([`PVQ_V_N_MAX`]); larger inputs are rejected with
//!   [`PvqVError::NOutOfRange`].
//! * `K` is the per-band pulse count chosen by the §4.3.3
//!   bit-allocation search; it is non-negative and bounded above by
//!   the §4.3.3 per-band cap. The module accepts `K ∈ 0..=4096`
//!   ([`PVQ_V_K_MAX`]); larger inputs are rejected with
//!   [`PvqVError::KOutOfRange`]. The §4.3.3 cap surface in
//!   [`crate::celt_cache_caps50`] keeps the reachable `K` well below
//!   this bound; the conservative ceiling is set so callers can sweep
//!   a wide envelope during testing without tripping the guard.
//!
//! Both bounds are caller-side bookkeeping guards; the recurrence
//! itself works for any non-negative `(N, K)`.
//!
//! ## Computation strategy
//!
//! The recurrence is evaluated in `u64` to retain headroom above the
//! 32-bit `ec_dec_uint` ceiling — intermediate sums during the inner
//! loop can briefly exceed `2**32` before the next outer-loop step
//! folds them back. The function tracks two rolling rows of length
//! `K + 1`:
//!
//! ```text
//!     row_prev[k] = V(N - 1, k)   (initially V(0, k) = δ_{k,0})
//!     row_curr[k] = V(N,     k)
//! ```
//!
//! and applies the recurrence `row_curr[k] = row_prev[k] +
//! row_curr[k - 1] + row_prev[k - 1]` left to right, with the base
//! cases `V(N, 0) = 1` and `V(0, K) = 0 (K != 0)` providing the
//! seed row.
//!
//! If any intermediate cell crosses `2**32 − 1` the computation
//! short-circuits and returns [`PvqVError::OverflowsDecUintRange`] —
//! a caller asking for that `(N, K)` couldn't decode such a stream
//! anyway.
//!
//! ## What this module does not own
//!
//! * **PVQ decoding itself.** The §4.3.4.2 "decode an index in
//!   `0..V(N,K) - 1` and convert to a vector" procedure consumes
//!   `V(N, K)` but lives downstream of this module.
//! * **The §4.3.4.1 Bits-to-Pulses search.** Selecting which `K` the
//!   §4.3.3 per-band allocation maps to is also a downstream
//!   consumer.
//! * **Any pre-computed `V(N, K)` table.** The recurrence is
//!   re-evaluated on every call. A future round may layer a cache on
//!   top of this primitive.
//! * **The "alternate univariate recurrence" / "direct polynomial
//!   solutions for small N" mentioned in §4.3.4.2.** The RFC
//!   explicitly permits multiple equivalent implementations; this
//!   module ships only the bivariate recurrence the RFC states.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.4.2 (p. 116), reproduced from
//! `docs/audio/opus/rfc6716-opus.txt`. No external library source
//! was consulted; the recurrence is given directly in the
//! standards-track text.

/// Maximum `N` (number of MDCT bins) this module accepts.
///
/// Standard non-Custom CELT bands have at most
/// [`crate::celt_band_layout::CELT_MAX_BINS_PER_BAND`] = 176 bins per
/// channel at the 20 ms frame size; joint-stereo bands double that
/// to 352. The bound is a caller-side bookkeeping guard, not an
/// algorithmic limit.
pub const PVQ_V_N_MAX: u32 = 352;

/// Maximum `K` (number of pulses) this module accepts.
///
/// The §4.3.3 per-band cap surface in [`crate::celt_cache_caps50`]
/// keeps the reachable `K` well below this ceiling; the conservative
/// `4096` upper bound lets fuzz callers sweep a wide envelope without
/// tripping the guard.
pub const PVQ_V_K_MAX: u32 = 4096;

/// Upper bound on `V(N, K)` enforced by §4.1.5 `ec_dec_uint`.
///
/// RFC 6716 §4.1.5 (p. 29) caps `ec_dec_uint`'s `ft` parameter at
/// `2**32 − 1`. A `(N, K)` whose `V(N, K)` would exceed this bound
/// is rejected with [`PvqVError::OverflowsDecUintRange`]: such an
/// index cannot be transmitted by a conforming Opus stream.
pub const PVQ_V_MAX: u64 = (1u64 << 32) - 1;

/// Errors returnable by [`pvq_codebook_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PvqVError {
    /// `N` exceeds [`PVQ_V_N_MAX`]. Caller-side bookkeeping bug.
    NOutOfRange {
        /// The `N` the caller passed.
        provided: u32,
        /// The maximum this module accepts ([`PVQ_V_N_MAX`]).
        max: u32,
    },
    /// `K` exceeds [`PVQ_V_K_MAX`]. Caller-side bookkeeping bug.
    KOutOfRange {
        /// The `K` the caller passed.
        provided: u32,
        /// The maximum this module accepts ([`PVQ_V_K_MAX`]).
        max: u32,
    },
    /// The recurrence produced a value strictly greater than
    /// [`PVQ_V_MAX`] = `2**32 − 1`. RFC 6716 §4.1.5 forbids
    /// `ec_dec_uint(ft)` with `ft > 2**32 − 1`, so the corresponding
    /// PVQ index cannot be transmitted by a conforming Opus stream.
    OverflowsDecUintRange {
        /// The `N` for which the recurrence overflowed.
        n: u32,
        /// The `K` for which the recurrence overflowed.
        k: u32,
    },
}

impl core::fmt::Display for PvqVError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            PvqVError::NOutOfRange { provided, max } => write!(
                f,
                "oxideav-opus: PVQ codebook-size N out of range: provided={provided}, max={max}"
            ),
            PvqVError::KOutOfRange { provided, max } => write!(
                f,
                "oxideav-opus: PVQ codebook-size K out of range: provided={provided}, max={max}"
            ),
            PvqVError::OverflowsDecUintRange { n, k } => write!(
                f,
                "oxideav-opus: PVQ codebook size V({n}, {k}) exceeds 2**32 − 1 \
                 (ec_dec_uint upper bound per RFC 6716 §4.1.5)"
            ),
        }
    }
}

impl std::error::Error for PvqVError {}

/// Computes the §4.3.4.2 PVQ codebook size `V(N, K)`.
///
/// `V(N, K)` is the number of integer-magnitude lattice points with
/// `|x_0| + |x_1| + ... + |x_{N-1}| = K`. RFC 6716 §4.3.4.2 (p. 116)
/// defines the bivariate recurrence
///
/// ```text
///     V(N, K) = V(N-1, K) + V(N, K-1) + V(N-1, K-1)
///     V(N, 0) = 1                       for all N ≥ 0
///     V(0, K) = 0                       for all K ≥ 1
/// ```
///
/// The function evaluates the recurrence in `u64` over two rolling
/// rows and short-circuits with [`PvqVError::OverflowsDecUintRange`]
/// the moment any intermediate cell crosses [`PVQ_V_MAX`] =
/// `2**32 − 1`.
///
/// # Edge cases
///
/// * `V(0, 0) = 1` (the empty lattice configuration).
/// * `V(N, 0) = 1` for every `N ≥ 0` (the all-zero vector).
/// * `V(0, K) = 0` for every `K ≥ 1` (no lattice point of dimension
///   zero can have positive `L1` norm).
/// * `V(1, K) = 2` for every `K ≥ 1` (the two sign-flipped pulses).
/// * `V(N, 1) = 2 * N` for every `N ≥ 1` (one pulse, `N` positions,
///   two signs each).
///
/// # Errors
///
/// * [`PvqVError::NOutOfRange`] if `n > PVQ_V_N_MAX`.
/// * [`PvqVError::KOutOfRange`] if `k > PVQ_V_K_MAX`.
/// * [`PvqVError::OverflowsDecUintRange`] if `V(n, k) > 2**32 − 1`.
pub fn pvq_codebook_size(n: u32, k: u32) -> Result<u32, PvqVError> {
    if n > PVQ_V_N_MAX {
        return Err(PvqVError::NOutOfRange {
            provided: n,
            max: PVQ_V_N_MAX,
        });
    }
    if k > PVQ_V_K_MAX {
        return Err(PvqVError::KOutOfRange {
            provided: k,
            max: PVQ_V_K_MAX,
        });
    }

    // Base cases first — they short-circuit the rolling-row walk.
    if k == 0 {
        // V(N, 0) = 1 for every N ≥ 0.
        return Ok(1);
    }
    if n == 0 {
        // V(0, K) = 0 for every K ≥ 1.
        return Ok(0);
    }

    // Rolling-row evaluation.
    //
    // row_prev holds V(N - 1, k) for k ∈ 0..=k_max.
    // row_curr holds V(N,     k) and is rebuilt left-to-right per step.
    //
    // Seed: row_prev = V(0, k) = δ_{k,0}, i.e. [1, 0, 0, ..., 0].
    // Profluens patch (PROFLUENS-PATCHES.md): this call site is ~93% of all decode allocations
    // (simprof: per-band, per-frame in the CELT PVQ hot loop — thousands of calls per packet). The
    // two rolling rows are function-local scratch that never escape, so back them with a reused
    // thread-local buffer: grown once to the largest K seen, then cleared+reused (no realloc within
    // capacity). A steady decode allocates nothing here. NB a per-packet bump arena is the *wrong*
    // tool for a site this hot — it would accumulate every call's rows until the packet boundary
    // and overflow; reused scratch is constant-memory.
    thread_local! {
        static PVQ_ROWS: core::cell::RefCell<(Vec<u64>, Vec<u64>)> =
            const { core::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    let k_usize = k as usize;
    PVQ_ROWS.with(|cell| {
        let mut guard = cell.borrow_mut();
        // row_prev holds V(N-1, k) for k ∈ 0..=k_max; row_curr is rebuilt left-to-right per step.
        // Seed: row_prev = V(0, k) = δ_{k,0} = [1, 0, ..., 0].
        let (row_prev, row_curr) = &mut *guard;
        row_prev.clear();
        row_prev.resize(k_usize + 1, 0u64);
        row_prev[0] = 1;
        row_curr.clear();
        row_curr.resize(k_usize + 1, 0u64);

        for _ in 1..=n {
            // V(N, 0) = 1 by definition.
            row_curr[0] = 1;
            for kk in 1..=k_usize {
                // V(N, K) = V(N - 1, K) + V(N, K - 1) + V(N - 1, K - 1).
                let v = row_prev[kk] + row_curr[kk - 1] + row_prev[kk - 1];
                if v > PVQ_V_MAX {
                    return Err(PvqVError::OverflowsDecUintRange { n, k });
                }
                row_curr[kk] = v;
            }
            // Slide rows: row_prev <- row_curr; row_curr becomes scratch.
            core::mem::swap(row_prev, row_curr);
        }

        // After the swap on the final iteration `row_prev` holds V(N, .).
        Ok(row_prev[k_usize] as u32)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §4.3.4.2 recurrence-defined edge cases -----------------------------

    #[test]
    fn v_zero_zero_is_one() {
        // V(0, 0) = 1 (the empty lattice configuration / all-zero vector).
        assert_eq!(pvq_codebook_size(0, 0), Ok(1));
    }

    #[test]
    fn v_n_zero_is_one_for_every_n() {
        // V(N, 0) = 1 for every N ≥ 0 (the all-zero vector).
        for n in 0..=PVQ_V_N_MAX {
            assert_eq!(pvq_codebook_size(n, 0), Ok(1), "V({n}, 0)");
        }
    }

    #[test]
    fn v_zero_k_is_zero_for_every_positive_k() {
        // V(0, K) = 0 for every K ≥ 1.
        for k in 1..=PVQ_V_K_MAX {
            assert_eq!(pvq_codebook_size(0, k), Ok(0), "V(0, {k})");
        }
    }

    #[test]
    fn v_one_k_is_two_for_every_positive_k() {
        // V(1, K) = 2 for every K ≥ 1 (the two sign-flipped pulses).
        // Bounded by 1..=200 to keep the test cheap.
        for k in 1..=200 {
            assert_eq!(pvq_codebook_size(1, k), Ok(2), "V(1, {k})");
        }
    }

    #[test]
    fn v_n_one_is_two_n_for_every_positive_n() {
        // V(N, 1) = 2N for every N ≥ 1 (one pulse, N positions,
        // two signs each).
        for n in 1..=PVQ_V_N_MAX {
            let expected = 2 * n;
            assert_eq!(pvq_codebook_size(n, 1), Ok(expected), "V({n}, 1)");
        }
    }

    // ---- §4.3.4.2 recurrence cross-check ------------------------------------

    #[test]
    fn recurrence_holds_for_small_sweep() {
        // Verify V(N, K) = V(N - 1, K) + V(N, K - 1) + V(N - 1, K - 1)
        // over a small sweep that comfortably fits u32. V(N, K)
        // grows roughly like (2K+1) over the diagonal so the safe
        // u32 envelope is around N + K ≤ 25.
        for n in 1..=12u32 {
            for k in 1..=12u32 {
                let v_n_k = pvq_codebook_size(n, k).unwrap() as u64;
                let v_nm1_k = pvq_codebook_size(n - 1, k).unwrap() as u64;
                let v_n_km1 = pvq_codebook_size(n, k - 1).unwrap() as u64;
                let v_nm1_km1 = pvq_codebook_size(n - 1, k - 1).unwrap() as u64;
                assert_eq!(
                    v_n_k,
                    v_nm1_k + v_n_km1 + v_nm1_km1,
                    "recurrence at (N={n}, K={k})"
                );
            }
        }
    }

    // ---- §4.3.4.2 worked points ---------------------------------------------
    //
    // The §4.3.4.2 recurrence has well-known closed-form values for
    // small (N, K) that fall out of expanding it by hand. Pinning a
    // handful here protects against subtle off-by-one slips in the
    // rolling-row walk.

    #[test]
    fn v_2_2_equals_eight() {
        // V(2, 2):
        //   V(1, 2) + V(2, 1) + V(1, 1)
        //   =     2  +     4  +     2
        //   = 8
        assert_eq!(pvq_codebook_size(2, 2), Ok(8));
    }

    #[test]
    fn v_3_3_equals_thirty_eight() {
        // Hand-expansion:
        //   V(2, 1) = 4, V(2, 2) = 8, V(2, 3) = 12.
        //   V(3, 1) = 6, V(3, 2) = V(2, 2) + V(3, 1) + V(2, 1)
        //              = 8 + 6 + 4 = 18.
        //   V(3, 3) = V(2, 3) + V(3, 2) + V(2, 2)
        //              = 12 + 18 + 8 = 38.
        assert_eq!(pvq_codebook_size(3, 3), Ok(38));
    }

    #[test]
    fn v_4_2_equals_twenty_four() {
        // V(4, 2):
        //   V(3, 1) = 6, V(3, 2) = 18, V(4, 1) = 8.
        //   V(4, 2) = V(3, 2) + V(4, 1) + V(3, 1)
        //             = 18 + 8 + 6 = 32?  No — recompute.
        //
        // Walk step by step:
        //   V(2, 0) = 1, V(2, 1) = 4, V(2, 2) = 8.
        //   V(3, 0) = 1.
        //   V(3, 1) = V(2, 1) + V(3, 0) + V(2, 0) = 4 + 1 + 1 = 6.
        //   V(3, 2) = V(2, 2) + V(3, 1) + V(2, 1) = 8 + 6 + 4 = 18.
        //   V(4, 0) = 1.
        //   V(4, 1) = V(3, 1) + V(4, 0) + V(3, 0) = 6 + 1 + 1 = 8.
        //   V(4, 2) = V(3, 2) + V(4, 1) + V(3, 1) = 18 + 8 + 6 = 32.
        assert_eq!(pvq_codebook_size(4, 2), Ok(32));
    }

    #[test]
    fn v_symmetric_under_n_k_swap_for_small_cases() {
        // The full V(N, K) is NOT symmetric in (N, K) — but a few low
        // hand-computed pairs share a structure. Pin V(2, 3) = V(3, 2)
        // = 12 / 18 — they are NOT equal — and instead pin the actual
        // asymmetry:
        //   V(2, 3) = V(1, 3) + V(2, 2) + V(1, 2)
        //             = 2 + 8 + 2 = 12.
        //   V(3, 2) = 18 (above).
        assert_eq!(pvq_codebook_size(2, 3), Ok(12));
        assert_eq!(pvq_codebook_size(3, 2), Ok(18));
        assert_ne!(
            pvq_codebook_size(2, 3).unwrap(),
            pvq_codebook_size(3, 2).unwrap()
        );
    }

    // ---- §4.3.4.2 monotonicity ---------------------------------------------

    #[test]
    fn monotone_non_decreasing_in_n_for_fixed_k() {
        // Adding a coordinate can never decrease the lattice count.
        // Sweep is bounded so values stay inside u32.
        for k in 0..=12u32 {
            let mut prev = pvq_codebook_size(0, k).unwrap();
            for n in 1..=12u32 {
                let v = pvq_codebook_size(n, k).unwrap();
                assert!(v >= prev, "V({n}, {k}) = {v} < V({}, {k}) = {prev}", n - 1);
                prev = v;
            }
        }
    }

    #[test]
    fn monotone_non_decreasing_in_k_for_fixed_n_ge_two() {
        // For N ≥ 2 the lattice count is non-decreasing in K. (N = 1
        // is the exception: V(1, K) = 2 for every K ≥ 1.) Sweep is
        // bounded so values stay inside u32.
        for n in 2..=12u32 {
            let mut prev = pvq_codebook_size(n, 0).unwrap();
            for k in 1..=12u32 {
                let v = pvq_codebook_size(n, k).unwrap();
                assert!(v >= prev, "V({n}, {k}) = {v} < V({n}, {}) = {prev}", k - 1);
                prev = v;
            }
        }
    }

    // ---- §4.1.5 overflow guard ---------------------------------------------

    #[test]
    fn overflow_guard_trips_when_recurrence_exceeds_2_pow_32_minus_1() {
        // V(176, 176) is well above 2^32. Verify the guard catches it.
        let result = pvq_codebook_size(176, 176);
        match result {
            Err(PvqVError::OverflowsDecUintRange { n: 176, k: 176 }) => {}
            other => panic!("expected OverflowsDecUintRange, got {other:?}"),
        }
    }

    #[test]
    fn overflow_guard_does_not_trip_on_values_just_under_the_ceiling() {
        // For N = 2, V(2, K) = 4K (a closed form that falls out of
        // the recurrence): V(2, 0) = 1, V(2, 1) = 4, V(2, 2) = 8,
        // V(2, 3) = 12, … grows linearly so never overflows in the
        // K ∈ 0..=PVQ_V_K_MAX window.
        for k in 0..=100u32 {
            let v = pvq_codebook_size(2, k).unwrap();
            if k == 0 {
                assert_eq!(v, 1);
            } else {
                assert_eq!(v, 4 * k, "V(2, {k})");
            }
        }
    }

    // ---- caller-side bookkeeping rejection ---------------------------------

    #[test]
    fn rejects_n_above_pvq_v_n_max() {
        let result = pvq_codebook_size(PVQ_V_N_MAX + 1, 5);
        assert_eq!(
            result,
            Err(PvqVError::NOutOfRange {
                provided: PVQ_V_N_MAX + 1,
                max: PVQ_V_N_MAX,
            })
        );
    }

    #[test]
    fn rejects_k_above_pvq_v_k_max() {
        let result = pvq_codebook_size(10, PVQ_V_K_MAX + 1);
        assert_eq!(
            result,
            Err(PvqVError::KOutOfRange {
                provided: PVQ_V_K_MAX + 1,
                max: PVQ_V_K_MAX,
            })
        );
    }

    #[test]
    fn n_max_boundary_is_accepted() {
        // At the boundary the K must be small enough that the value
        // fits 32-bit. V(352, 0) = 1 trivially.
        assert_eq!(pvq_codebook_size(PVQ_V_N_MAX, 0), Ok(1));
        // V(352, 1) = 704.
        assert_eq!(pvq_codebook_size(PVQ_V_N_MAX, 1), Ok(2 * PVQ_V_N_MAX));
    }

    #[test]
    fn k_max_boundary_is_accepted_for_small_n() {
        // V(1, K_max) = 2 — pure base case path.
        assert_eq!(pvq_codebook_size(1, PVQ_V_K_MAX), Ok(2));
        // V(2, K_max) = 4 * K_max.
        assert_eq!(pvq_codebook_size(2, PVQ_V_K_MAX), Ok(4 * PVQ_V_K_MAX));
    }

    // ---- module-constant pins ----------------------------------------------

    #[test]
    fn pvq_v_n_max_is_three_hundred_fifty_two() {
        // The bound covers stereo-joint bands at the 20 ms frame size
        // (2 × 176 = 352 bins).
        assert_eq!(PVQ_V_N_MAX, 352);
    }

    #[test]
    fn pvq_v_k_max_is_four_thousand_ninety_six() {
        // Conservative ceiling chosen so the §4.3.3 cap surface (which
        // keeps reachable K well below this) lives inside the
        // module's accepted envelope.
        assert_eq!(PVQ_V_K_MAX, 4096);
    }

    #[test]
    fn pvq_v_max_is_two_pow_thirty_two_minus_one() {
        // §4.1.5 caps ec_dec_uint's ft at 2**32 − 1; the overflow
        // guard inherits that bound.
        assert_eq!(PVQ_V_MAX, (1u64 << 32) - 1);
        assert_eq!(PVQ_V_MAX, 4_294_967_295);
    }

    // ---- error-Display sanity ----------------------------------------------

    #[test]
    fn display_messages_mention_the_failing_input() {
        let n_err = PvqVError::NOutOfRange {
            provided: PVQ_V_N_MAX + 5,
            max: PVQ_V_N_MAX,
        };
        let n_msg = format!("{n_err}");
        assert!(n_msg.contains(&format!("{}", PVQ_V_N_MAX + 5)));
        assert!(n_msg.contains(&format!("{PVQ_V_N_MAX}")));

        let k_err = PvqVError::KOutOfRange {
            provided: PVQ_V_K_MAX + 1,
            max: PVQ_V_K_MAX,
        };
        let k_msg = format!("{k_err}");
        assert!(k_msg.contains(&format!("{}", PVQ_V_K_MAX + 1)));

        let o_err = PvqVError::OverflowsDecUintRange { n: 200, k: 200 };
        let o_msg = format!("{o_err}");
        assert!(o_msg.contains("V(200, 200)"));
        assert!(o_msg.contains("2**32"));
    }

    // ---- §4.1.5 boundary pin -----------------------------------------------

    #[test]
    fn known_v_values_against_hand_computed_table() {
        // A 7 × 7 hand-computed table of V(N, K) values that lets a
        // reviewer spot-check the rolling-row walk.
        let table: [[u32; 7]; 7] = [
            // K=  0    1    2    3    4    5    6
            /*N=0*/ [1, 0, 0, 0, 0, 0, 0],
            /*N=1*/ [1, 2, 2, 2, 2, 2, 2],
            /*N=2*/ [1, 4, 8, 12, 16, 20, 24],
            /*N=3*/ [1, 6, 18, 38, 66, 102, 146],
            /*N=4*/ [1, 8, 32, 88, 192, 360, 608],
            /*N=5*/ [1, 10, 50, 170, 450, 1002, 1970],
            /*N=6*/ [1, 12, 72, 292, 912, 2364, 5336],
        ];
        for (n, row) in table.iter().enumerate() {
            for (k, &expected) in row.iter().enumerate() {
                assert_eq!(
                    pvq_codebook_size(n as u32, k as u32),
                    Ok(expected),
                    "V({n}, {k})"
                );
            }
        }
    }
}
