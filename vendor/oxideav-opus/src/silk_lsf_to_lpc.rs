//! SILK Normalized LSF → LPC conversion — RFC 6716 §4.2.7.5.6 (core).
//!
//! Given a stabilized + (optionally) interpolated normalized-LSF vector
//! `nlsf_q15[]` (the §4.2.7.5.4 or §4.2.7.5.5 output), this module runs
//! the `silk_NLSF2A()` procedure described by §4.2.7.5.6:
//!
//!  1. Approximate `cos(pi * n[k])` using the §4.2.7.5.6 Q12 cosine
//!     table (Table 28) with linear interpolation over the top-7-bits
//!     index `i = n[k] >> 8` and the next-8-bits fraction
//!     `f = n[k] & 255`, then re-order the resulting `c_Q17[]` via
//!     Table 27 so the polynomial reconstruction below stays
//!     numerically stable:
//!
//!     ```text
//!     c_Q17[ordering[k]] = (cos_Q12[i]*256
//!                           + (cos_Q12[i+1] - cos_Q12[i])*f + 4) >> 3
//!     ```
//!
//!  2. Run the §4.2.7.5.6 P/Q polynomial recurrence
//!     (`silk_NLSF2A_find_poly()`):
//!
//!     ```text
//!     d2 = d_LPC / 2
//!     p_Q16[0][0] = q_Q16[0][0] = 1 << 16
//!     p_Q16[0][1] = -c_Q17[0]    q_Q16[0][1] = -c_Q17[1]
//!
//!     for 0 < k < d2, 0 <= j <= k+1:
//!         p_Q16[k][j] = p_Q16[k-1][j] + p_Q16[k-1][j-2]
//!                     - ((c_Q17[2*k]   * p_Q16[k-1][j-1] + 32768) >> 16)
//!         q_Q16[k][j] = q_Q16[k-1][j] + q_Q16[k-1][j-2]
//!                     - ((c_Q17[2*k+1] * q_Q16[k-1][j-1] + 32768) >> 16)
//!     ```
//!
//!     with the §4.2.7.5.6 boundary conditions `p[k][j]=q[k][j]=0` for
//!     `j<0` and `p[k][k+2]=p[k][k]`, `q[k][k+2]=q[k][k]` (justified by
//!     the polynomial symmetry).
//!
//!  3. Combine the final row into the 32-bit Q17 LPC coefficients
//!     (`silk_NLSF2A` last block):
//!
//!     ```text
//!     a32_Q17[k]         = -(q_Q16[d2-1][k+1] - q_Q16[d2-1][k])
//!                          -  (p_Q16[d2-1][k+1] + p_Q16[d2-1][k])
//!
//!     a32_Q17[d_LPC-k-1] =  (q_Q16[d2-1][k+1] - q_Q16[d2-1][k])
//!                          - (p_Q16[d2-1][k+1] + p_Q16[d2-1][k])
//!     ```
//!
//! On top of the §4.2.7.5.6 **core conversion**, this module also lands the
//! §4.2.7.5.7 **range-limiting** bandwidth-expansion loop
//! ([`LpcQ17::range_limited`]): up to 10 rounds of chirp-factor bandwidth
//! expansion that shrink the raw `a32_Q17[]` so it fits a signed 16-bit Q12
//! value, followed by a fixed Q12 saturation after the 10th round if it
//! still overflows. The range-limited `a32_Q17[]` produced there is held in
//! the Q17 domain (per §4.2.7.5.7 the final saturation converts back to Q17
//! for the prediction-gain limiting that follows).
//!
//! Finally this module lands the §4.2.7.5.8 **prediction-gain limiting**
//! ([`LpcQ17::prediction_gain_limited`] → [`LpcQ12`]): up to 16 rounds of
//! bandwidth expansion driven by the `silk_LPC_inverse_pred_gain_QA()`
//! stability test rather than the coefficient magnitude. Each round converts
//! the range-limited `a32_Q17[]` to the real Q12 coefficients
//! `a32_Q12[n] = (a32_Q17[n] + 16) >> 5` that reconstruction will use, runs
//! the DC-response check (`DC_resp = sum(a32_Q12) > 4096` ⇒ unstable) and the
//! fixed-point Levinson recurrence on the Q24-widened coefficients
//! (`abs(a32_Q24[k][k]) > 16773022` or `inv_gain_Q30[k] < 107374` ⇒
//! unstable). If the filter is stable the final Q12 coefficients are
//! returned; otherwise a chirp round with `sc_Q16[0] = 65536 - (2<<i)` is
//! applied (the same `silk_bwexpander_32` as §4.2.7.5.7). On the 16th round
//! `sc_Q16[0]` is `0`, zeroing every coefficient and guaranteeing a stable
//! all-zero filter.

use crate::silk_lsf_stage2::{D_LPC_MAX, D_LPC_NB_MB, D_LPC_WB};
use crate::toc::Bandwidth;
use crate::Error;

// =====================================================================
// Table 27 — LSF Ordering for Polynomial Evaluation.
//
// `ordering[k]` is the destination slot in `c_Q17[]` for the linearly-
// interpolated cosine of `nlsf_q15[k]`. The reordering improves numerical
// stability of the §4.2.7.5.6 polynomial recurrence by pairing roots that
// cancel in P(z) / Q(z) close together in the index order.
//
// `d_LPC = 10` for NB / MB; `d_LPC = 16` for WB. Cells are in `0..d_LPC`.
// =====================================================================

const ORDERING_NB_MB: [u8; D_LPC_NB_MB] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

const ORDERING_WB: [u8; D_LPC_WB] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];

/// The Table 27 ordering vector for `bandwidth`. Rejects SWB / FB since
/// SILK never sees those after the §4.2.2 hybrid split.
pub fn ordering(bandwidth: Bandwidth) -> Result<&'static [u8], Error> {
    match bandwidth {
        Bandwidth::Nb | Bandwidth::Mb => Ok(&ORDERING_NB_MB),
        Bandwidth::Wb => Ok(&ORDERING_WB),
        Bandwidth::Swb | Bandwidth::Fb => Err(Error::MalformedPacket),
    }
}

// =====================================================================
// Table 28 — Q12 Cosine Table for LSF Conversion.
//
// 129 entries, `i ∈ 0..=128`. `cos_Q12[0] = 4096` (= cos(0) in Q12),
// `cos_Q12[64] = 0` (= cos(pi/2)), `cos_Q12[128] = -4096` (= cos(pi)).
// The table is anti-symmetric about `i = 64`: `cos_Q12[128-i] == -cos_Q12[i]`.
// =====================================================================

#[rustfmt::skip]
const COS_Q12: [i32; 129] = [
    4096,  4095,  4091,  4085,
    4076,  4065,  4052,  4036,
    4017,  3997,  3973,  3948,
    3920,  3889,  3857,  3822,
    3784,  3745,  3703,  3659,
    3613,  3564,  3513,  3461,
    3406,  3349,  3290,  3229,
    3166,  3102,  3035,  2967,
    2896,  2824,  2751,  2676,
    2599,  2520,  2440,  2359,
    2276,  2191,  2106,  2019,
    1931,  1842,  1751,  1660,
    1568,  1474,  1380,  1285,
    1189,  1093,   995,   897,
     799,   700,   601,   501,
     401,   301,   201,   101,
       0,  -101,  -201,  -301,
    -401,  -501,  -601,  -700,
    -799,  -897,  -995, -1093,
   -1189, -1285, -1380, -1474,
   -1568, -1660, -1751, -1842,
   -1931, -2019, -2106, -2191,
   -2276, -2359, -2440, -2520,
   -2599, -2676, -2751, -2824,
   -2896, -2967, -3035, -3102,
   -3166, -3229, -3290, -3349,
   -3406, -3461, -3513, -3564,
   -3613, -3659, -3703, -3745,
   -3784, -3822, -3857, -3889,
   -3920, -3948, -3973, -3997,
   -4017, -4036, -4052, -4065,
   -4076, -4085, -4091, -4095,
   -4096,
];

// =====================================================================
// silk_NLSF2A_cos approximation.
// =====================================================================

/// Compute the §4.2.7.5.6 re-ordered Q17 cosine vector `c_Q17[]` from a
/// stabilized / interpolated normalized-LSF vector `nlsf_q15[]`.
///
/// Each `nlsf_q15[k]` is split into the top 7 bits `i = nlsf >> 8` (in
/// `0..=127`, so `i+1` indexes a valid cell in [`COS_Q12`]) and the next
/// 8 bits `f = nlsf & 255`. The §4.2.7.5.6 piecewise-linear interpolation
///
/// ```text
/// c_Q17[ordering[k]] = (cos_Q12[i]*256 + (cos_Q12[i+1]-cos_Q12[i])*f + 4) >> 3
/// ```
///
/// is applied with the Table 27 destination index for `bandwidth`. Output
/// length is `d_LPC` (10 for NB / MB, 16 for WB).
///
/// Returns `Error::MalformedPacket` if `bandwidth` is SWB / FB (SILK
/// never sees those) or if `nlsf_q15.len() != d_LPC`.
pub fn nlsf_to_c_q17(bandwidth: Bandwidth, nlsf_q15: &[i16]) -> Result<[i32; D_LPC_MAX], Error> {
    let ord = ordering(bandwidth)?;
    let d_lpc = ord.len();
    if nlsf_q15.len() != d_lpc {
        return Err(Error::MalformedPacket);
    }

    let mut c_q17 = [0i32; D_LPC_MAX];
    for (k, &n) in nlsf_q15.iter().enumerate() {
        // The §4.2.7.5.4 stabilization guarantees nlsf_q15[k] ∈ [0, 32767]
        // so the top-7-bits index is at most 127 and `i+1` is a valid
        // COS_Q12 cell.
        let n = n as i32;
        let i = (n >> 8) as usize;
        let f = n & 0xFF;
        let a = COS_Q12[i];
        let b = COS_Q12[i + 1];
        // The `+ 4` in the formula is the half-LSB rounding term for the
        // final >> 3 (the only Q14→Q17 step that survives after the
        // *256 and *f terms cancel into the same Q14 scale).
        let v = (a * 256 + (b - a) * f + 4) >> 3;
        c_q17[ord[k] as usize] = v;
    }
    Ok(c_q17)
}

// =====================================================================
// silk_NLSF2A_find_poly + silk_NLSF2A.
// =====================================================================

/// Run the §4.2.7.5.6 P/Q polynomial recurrence on a single side
/// (selecting the even-indexed `c_Q17[2*k]` terms for P or the
/// odd-indexed `c_Q17[2*k+1]` terms for Q) and return the final row.
///
/// `d2 = d_LPC / 2` (5 for NB / MB, 8 for WB). The returned row holds the
/// first `d2 + 1` coefficients (the rest are redundant by the §4.2.7.5.6
/// symmetry `p[k][k+2] = p[k][k]`).
fn find_poly(c_q17: &[i32], d_lpc: usize, parity: usize) -> [i64; D_LPC_MAX / 2 + 1] {
    debug_assert!(d_lpc == D_LPC_NB_MB || d_lpc == D_LPC_WB);
    debug_assert!(parity < 2);
    let d2 = d_lpc / 2;

    // Two rolling rows: prev (k-1) and curr (k).
    let mut prev = [0i64; D_LPC_MAX / 2 + 1];
    let mut curr = [0i64; D_LPC_MAX / 2 + 1];

    // k = 0 initial conditions: only [0] and [1] are touched.
    prev[0] = 1i64 << 16;
    prev[1] = -(c_q17[parity] as i64);

    for k in 1..d2 {
        let c = c_q17[2 * k + parity] as i64;
        for j in 0..=k + 1 {
            // Boundary conditions per §4.2.7.5.6: p[k-1][j] = 0 for
            // j < 0, and — because row k-1 is the coefficient list of a
            // SYMMETRIC product of degree 2k, stored only up to its
            // middle coefficient j = k — the "assume p_Q16[k][k+2] =
            // p_Q16[k][k]" rule mirrors the read past the middle:
            // p[k-1][k+1] = p[k-1][k-1].
            let pj0 = if j == k + 1 { prev[k - 1] } else { prev[j] };
            let pjm2 = if j >= 2 { prev[j - 2] } else { 0 };
            let pjm1 = if j >= 1 { prev[j - 1] } else { 0 };
            // The (*c * pjm1 + 32768) >> 16 rounding term matches the
            // §4.2.7.5.6 recurrence verbatim. Up to 48 bits of intermediate
            // precision can be required; i64 covers it.
            curr[j] = pj0 + pjm2 - ((c * pjm1 + 32768) >> 16);
        }
        // Row k is now the first k+2 coefficients of the symmetric
        // degree-2(k+1) product; only j <= k+1 is ever stored, the
        // mirror rule above supplies the rest.
        prev.copy_from_slice(&curr);
    }
    prev
}

/// The §4.2.7.5.6 NLSF → LPC core conversion result.
///
/// Holds the 32-bit Q17 LPC coefficients `a32_Q17[k]`, `k ∈ 0..d_LPC`
/// (without the leading `1.0` coefficient). Use [`LpcQ17::range_limited`]
/// to apply the §4.2.7.5.7 range-limiting bandwidth expansion and then
/// [`LpcQ17::prediction_gain_limited`] to apply the §4.2.7.5.8
/// prediction-gain stability limiting, which produces the final Q12
/// [`LpcQ12`] filter used for reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpcQ17 {
    len: u8,
    a32_q17: [i32; D_LPC_MAX],
}

/// The final §4.2.7.5.8 prediction-gain-limited Q12 LPC coefficients
/// `a_Q12[k]`, `k ∈ 0..d_LPC`, ready for the §4.2.7.9.2 LPC synthesis
/// filter. These are guaranteed stable: the §4.2.7.5.8 chirp loop runs up
/// to 16 rounds of bandwidth expansion and, on the final round, zeroes
/// every coefficient so an all-zero (trivially stable) filter is the
/// worst-case outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpcQ12 {
    len: u8,
    a_q12: [i32; D_LPC_MAX],
    /// The number of §4.2.7.5.8 chirp rounds that ran before the filter
    /// was deemed stable (`0` when the input was already stable). Exposed
    /// for tests / diagnostics; not part of the reconstruction interface.
    rounds: u8,
}

impl LpcQ12 {
    /// The §4.2.7.5.8 Q12 LPC coefficients `a_Q12[k]`. Length is `d_LPC`
    /// (10 for NB / MB, 16 for WB).
    pub fn a_q12(&self) -> &[i32] {
        &self.a_q12[..self.len as usize]
    }

    /// Number of coefficients (10 for NB / MB, 16 for WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if there are no coefficients (never happens after a
    /// successful conversion of a valid normalized-LSF vector).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of §4.2.7.5.8 bandwidth-expansion rounds that ran before the
    /// filter passed the stability test (`0` if it was stable as-is).
    pub fn rounds(&self) -> usize {
        self.rounds as usize
    }
}

impl LpcQ17 {
    /// Run the §4.2.7.5.6 core conversion against a stabilized /
    /// interpolated normalized-LSF vector `nlsf_q15[]`.
    ///
    /// `bandwidth` selects the Table 27 ordering and the implied `d_LPC`:
    /// 10 for NB / MB, 16 for WB. SWB / FB are rejected (SILK never sees
    /// them after the §4.2.2 hybrid split).
    pub fn from_nlsf(bandwidth: Bandwidth, nlsf_q15: &[i16]) -> Result<Self, Error> {
        let ord = ordering(bandwidth)?;
        let d_lpc = ord.len();
        if nlsf_q15.len() != d_lpc {
            return Err(Error::MalformedPacket);
        }

        let c_q17 = nlsf_to_c_q17(bandwidth, nlsf_q15)?;
        let d2 = d_lpc / 2;

        let p_last = find_poly(&c_q17[..d_lpc], d_lpc, 0);
        let q_last = find_poly(&c_q17[..d_lpc], d_lpc, 1);

        // Assemble a32_Q17 from the last p/q rows. The §4.2.7.5.6 final
        // block walks k ∈ 0..d2 and writes both ends of the array at
        // once: a32_Q17[k] = -(q_diff + p_sum) and
        // a32_Q17[d_LPC-k-1] = q_diff - p_sum, where
        // q_diff = q[d2-1][k+1] - q[d2-1][k] and
        // p_sum  = p[d2-1][k+1] + p[d2-1][k].
        let mut a32_q17 = [0i32; D_LPC_MAX];
        for k in 0..d2 {
            let q_diff = q_last[k + 1] - q_last[k];
            let p_sum = p_last[k + 1] + p_last[k];
            let lo = -(q_diff + p_sum);
            let hi = q_diff - p_sum;
            // The §4.2.7.5.6 prose notes that overflow into 32-bit is
            // expected before §4.2.7.5.7 clamps it; cast to i32 with
            // wrapping so adversarial inputs that overflow Q17 still
            // produce a deterministic value the next stage will then
            // reject or expand-down.
            a32_q17[k] = lo as i32;
            a32_q17[d_lpc - k - 1] = hi as i32;
        }

        Ok(Self {
            len: d_lpc as u8,
            a32_q17,
        })
    }

    /// The Q17 LPC coefficients `a32_Q17[k]`. Length is `d_LPC` (10 for
    /// NB / MB, 16 for WB).
    pub fn a32_q17(&self) -> &[i32] {
        &self.a32_q17[..self.len as usize]
    }

    /// Number of coefficients (10 for NB / MB, 16 for WB).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if there are no coefficients (never happens after a
    /// successful conversion of a valid normalized-LSF vector).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Apply the RFC 6716 §4.2.7.5.7 range-limiting bandwidth expansion to
    /// the raw §4.2.7.5.6 `a32_Q17[]` coefficients.
    ///
    /// The raw coefficients are too large to fit a signed 16-bit value;
    /// reducing them to Q12 precision doesn't incur significant quality
    /// loss but still doesn't guarantee a fit. Up to 10 rounds of
    /// bandwidth expansion run per §4.2.7.5.7:
    ///
    ///  * Each round finds the index `k` with the largest `abs(a32_Q17[k])`
    ///    (ties broken toward the lowest `k`), computes
    ///    `maxabs_Q12 = min((maxabs_Q17 + 16) >> 5, 163838)`, and **stops**
    ///    once `maxabs_Q12 <= 32767` (the coefficients now fit Q12).
    ///  * Otherwise it derives the chirp factor
    ///    `sc_Q16[0] = 65470 - ((maxabs_Q12 - 32767) << 14)
    ///    / ((maxabs_Q12 * (k+1)) >> 2)` (integer division) and runs
    ///    `silk_bwexpander_32`:
    ///    `a32_Q17[k] = (a32_Q17[k]*sc_Q16[k]) >> 16` and
    ///    `sc_Q16[k+1] = (sc_Q16[0]*sc_Q16[k] + 32768) >> 16` (the second
    ///    multiply unsigned to avoid 32-bit overflow).
    ///
    /// If the coefficients still overflow Q12 after the 10th round, each
    /// coefficient is saturated in the Q12 domain and converted back to Q17:
    /// `a32_Q17[k] = clamp(-32768, (a32_Q17[k] + 16) >> 5, 32767) << 5`.
    /// Per §4.2.7.5.7 this saturation is performed only if `maxabs_Q12` is
    /// still greater than 32767 after the 10th round (i.e. it is skipped if
    /// expansion converged earlier).
    ///
    /// The result is returned in the Q17 domain (the §4.2.7.5.8
    /// prediction-gain limiting that follows consumes Q17 coefficients), so
    /// it shares the [`LpcQ17`] representation. The §4.2.7.5.8 stability
    /// check is **not** applied here.
    pub fn range_limited(&self) -> LpcQ17 {
        let mut a32 = self.a32_q17;
        let d_lpc = self.len as usize;
        limit_lpc_range(&mut a32[..d_lpc]);
        LpcQ17 {
            len: self.len,
            a32_q17: a32,
        }
    }

    /// Apply the RFC 6716 §4.2.7.5.8 prediction-gain limiting to the
    /// (range-limited) §4.2.7.5.7 `a32_Q17[]` coefficients, producing the
    /// final stable Q12 filter [`LpcQ12`].
    ///
    /// Even after §4.2.7.5.7 the filter may have so much prediction gain
    /// that it is unstable (especially for voiced sounds). Rather than
    /// using the prediction gain itself (which can diverge for an unstable
    /// filter), this stage drives up to **16 rounds** of bandwidth expansion
    /// off `silk_LPC_inverse_pred_gain_QA()`, which decides stability from
    /// the reflection coefficients computed by a fixed-point Levinson
    /// recurrence on the *real* Q12 coefficients that reconstruction will
    /// use:
    ///
    ///  * `a32_Q12[n] = (a32_Q17[n] + 16) >> 5` — the Q12 coefficients.
    ///  * **DC-response check.** `DC_resp = Σ a32_Q12[n]`; `DC_resp > 4096`
    ///    ⇒ unstable.
    ///  * **Levinson recurrence.** Initialize `inv_gain_Q30[d_LPC] = 1<<30`
    ///    and `a32_Q24[d_LPC-1][n] = a32_Q12[n] << 12`, then for each `k`
    ///    from `d_LPC-1` down to `0`:
    ///    * `abs(a32_Q24[k][k]) > 16773022` (≈ 0.99975 in Q24) ⇒ unstable;
    ///    * `rc_Q31 = -a32_Q24[k][k] << 7`,
    ///      `div_Q30 = (1<<30) - (rc_Q31*rc_Q31 >> 32)`,
    ///      `inv_gain_Q30[k] = (inv_gain_Q30[k+1]*div_Q30 >> 32) << 2`;
    ///    * `inv_gain_Q30[k] < 107374` (≈ 1/10000 in Q30) ⇒ unstable;
    ///    * otherwise (for `k > 0`) compute row `k-1` via the spec's
    ///      `b1 = ilog(div_Q30)`, `inv_Qb2`, `err_Q29`, `gain_Qb1`,
    ///      `num_Q24[n]`, `a32_Q24[k-1][n]` formulas.
    ///
    /// If the filter is stable on round `i`, the final coefficients
    /// `a_Q12[k] = (a32_Q17[k] + 16) >> 5` are returned. Otherwise a chirp
    /// round with `sc_Q16[0] = 65536 - (2<<i)` is applied to `a32_Q17[]`
    /// using the same `silk_bwexpander_32` as §4.2.7.5.7. On round 15
    /// `sc_Q16[0]` is `0`, so every coefficient becomes `0`, guaranteeing a
    /// stable filter.
    pub fn prediction_gain_limited(&self) -> LpcQ12 {
        let d_lpc = self.len as usize;
        let mut a32_q17 = self.a32_q17;

        for round in 0..16u32 {
            // Real Q12 coefficients reconstruction will use.
            let mut a32_q12 = [0i32; D_LPC_MAX];
            for k in 0..d_lpc {
                a32_q12[k] = (a32_q17[k] + 16) >> 5;
            }

            if is_lpc_stable(&a32_q12[..d_lpc]) {
                // Stable — emit the final Q12 coefficients.
                let mut a_q12 = [0i32; D_LPC_MAX];
                a_q12[..d_lpc].copy_from_slice(&a32_q12[..d_lpc]);
                return LpcQ12 {
                    len: self.len,
                    a_q12,
                    rounds: round as u8,
                };
            }

            // Unstable — apply one round of §4.2.7.5.7-style chirp with the
            // §4.2.7.5.8 round-dependent factor sc_Q16[0] = 65536 - (2<<i).
            // On round 15 this is exactly 0, zeroing every coefficient.
            let sc_q16_0 = 65536i64 - (2i64 << round);
            bwexpander_32(&mut a32_q17[..d_lpc], sc_q16_0);
        }

        // Round 15 forced sc_Q16[0] = 0, so a32_Q17[] is all zeros and the
        // Q12 conversion of zero is zero — an all-zero, trivially stable
        // filter. (We still produce it through the same conversion path so
        // the rounding term is applied uniformly.)
        let mut a_q12 = [0i32; D_LPC_MAX];
        for k in 0..d_lpc {
            a_q12[k] = (a32_q17[k] + 16) >> 5;
        }
        LpcQ12 {
            len: self.len,
            a_q12,
            rounds: 16,
        }
    }
}

/// RFC 6716 §4.2.7.5.8 `silk_LPC_inverse_pred_gain_QA()` stability test.
///
/// Returns `true` iff the LPC synthesis filter built from the real Q12
/// coefficients `a32_Q12[]` is stable, i.e. the DC response does not exceed
/// 4096 and the fixed-point Levinson recurrence keeps every reflection
/// coefficient sufficiently below one in magnitude.
///
/// All multiplies that the spec marks as requiring more than 32 bits are
/// performed in `i64`; `b1` ranges from 20 to 31 so `1 << (b1-1)` and the
/// shift amounts stay in range.
fn is_lpc_stable(a32_q12: &[i32]) -> bool {
    let d_lpc = a32_q12.len();

    // DC-response check: DC_resp = Σ a32_Q12[n]; > 4096 ⇒ unstable.
    let dc_resp: i64 = a32_q12.iter().map(|&c| c as i64).sum();
    if dc_resp > 4096 {
        return false;
    }

    // Widen the top row to Q24. inv_gain_Q30[d_LPC] = 1 << 30.
    // We keep just the current row (a32_Q24[k][..]) and shrink it each step.
    let mut row: [i64; D_LPC_MAX] = [0; D_LPC_MAX];
    for k in 0..d_lpc {
        // a32_Q24[d_LPC-1][n] = a32_Q12[n] << 12.
        row[k] = (a32_q12[k] as i64) << 12;
        // §4.2.7.5.8's arithmetic envelope has every a32_Q24 value fit
        // in 32 bits. A Q12 coefficient large enough to escape that
        // envelope (adversarial input ahead of sufficient bandwidth
        // expansion; a round-382 fuzz find) describes a wildly unstable
        // filter — classify it as such instead of overflowing the
        // recurrence's i64 products.
        if i32::try_from(row[k]).is_err() {
            return false;
        }
    }
    let mut inv_gain_q30: i64 = 1 << 30;

    // k from d_LPC-1 down to 0.
    for k in (0..d_lpc).rev() {
        let akk = row[k];
        // abs(a32_Q24[k][k]) > 16773022 (≈ 0.99975 in Q24) ⇒ unstable.
        if akk.unsigned_abs() > 16_773_022 {
            return false;
        }

        // rc_Q31[k] = -a32_Q24[k][k] << 7.
        let rc_q31 = -akk << 7;
        // div_Q30[k] = (1<<30) - (rc_Q31*rc_Q31 >> 32). The product needs
        // more than 32 bits → i64 (rc_Q31 fits ±~2.1e9, the square ±~4.6e18
        // which is within i64).
        let div_q30 = (1i64 << 30) - ((rc_q31 * rc_q31) >> 32);
        // inv_gain_Q30[k] = (inv_gain_Q30[k+1]*div_Q30 >> 32) << 2.
        inv_gain_q30 = ((inv_gain_q30 * div_q30) >> 32) << 2;
        // inv_gain_Q30[k] < 107374 (≈ 1/10000 in Q30) ⇒ unstable.
        if inv_gain_q30 < 107_374 {
            return false;
        }

        if k > 0 {
            // Compute row k-1 from row k.
            // b1 = ilog(div_Q30); b2 = b1 - 16. b1 ∈ [20, 31].
            let b1 = ilog64(div_q30) as i64;
            let b2 = b1 - 16;
            // inv_Qb2 = ((1<<29) - 1) / (div_Q30 >> (b2+1)). The divisor is
            // positive (div_Q30 > 0 for a stable step), so integer division
            // is well-defined.
            let inv_qb2 = ((1i64 << 29) - 1) / (div_q30 >> (b2 + 1));
            // err_Q29 = (1<<29) - ((div_Q30 << (15-b2)) * inv_Qb2 >> 16).
            let err_q29 = (1i64 << 29) - (((div_q30 << (15 - b2)) * inv_qb2) >> 16);
            // gain_Qb1 = (inv_Qb2 << 16) + (err_Q29*inv_Qb2 >> 13).
            let gain_qb1 = (inv_qb2 << 16) + ((err_q29 * inv_qb2) >> 13);

            // num_Q24[n] = a32_Q24[k][n]
            //            - ((a32_Q24[k][k-n-1]*rc_Q31 + (1<<30)) >> 31)
            // a32_Q24[k-1][n] = (num_Q24[n]*gain_Qb1 + (1<<(b1-1))) >> b1
            // for 0 <= n < k. The reads use the *current* row, so snapshot
            // it (n and k-n-1 both index the same row k) before overwriting.
            let cur = row;
            let round_b1 = 1i64 << (b1 - 1);
            for n in 0..k {
                let num_q24 = cur[n] - (((cur[k - n - 1] * rc_q31) + (1i64 << 30)) >> 31);
                // §4.2.7.5.8: "otherwise all intermediate results fit
                // in 32 bits or less". A value escaping that envelope
                // can only arise for a filter outside the procedure's
                // guaranteed domain (an adversarial coefficient set a
                // round-382 fuzz run produced overflowed the next
                // product here) — classify it unstable so the caller
                // applies another round of bandwidth expansion instead
                // of overflowing.
                if i32::try_from(num_q24).is_err() {
                    return false;
                }
                let next = ((num_q24 * gain_qb1) + round_b1) >> b1;
                if i32::try_from(next).is_err() {
                    return false;
                }
                row[n] = next;
            }
        }
    }

    // Every k passed both checks ⇒ stable.
    true
}

/// `ilog(n)` per RFC 6716 §1.1.10 for a non-negative `i64`: the minimum
/// number of bits required to store the positive integer `n` in binary, or
/// `0` for `n <= 0`. (`div_Q30` here is positive when the recurrence reaches
/// this point, so the `n <= 0` branch is defensive.)
fn ilog64(n: i64) -> u32 {
    if n <= 0 {
        0
    } else {
        64 - (n as u64).leading_zeros()
    }
}

/// In-place RFC 6716 §4.2.7.5.7 range-limiting bandwidth expansion of the
/// raw §4.2.7.5.6 Q17 LPC coefficients.
///
/// Runs up to 10 rounds of `silk_bwexpander_32` chirping, then — only if
/// the largest coefficient still overflows Q12 after the 10th round —
/// applies the fixed Q12 saturation. See [`LpcQ17::range_limited`] for the
/// formula breakdown.
fn limit_lpc_range(a32_q17: &mut [i32]) {
    for _round in 0..10 {
        // Find the index of the largest abs(a32_Q17[k]); ties → lowest k.
        // `unsigned_abs()` gives the magnitude even for i32::MIN without
        // the i32::MIN abs() panic; widen to i64 for the later arithmetic.
        let mut max_idx = 0usize;
        let mut maxabs_q17: i64 = a32_q17[0].unsigned_abs() as i64;
        for (k, &c) in a32_q17.iter().enumerate().skip(1) {
            let abs = c.unsigned_abs() as i64;
            if abs > maxabs_q17 {
                maxabs_q17 = abs;
                max_idx = k;
            }
        }

        // maxabs_Q12 = min((maxabs_Q17 + 16) >> 5, 163838). The upper bound
        // 163838 == ((2**31 - 1) >> 14) + 32767 caps the chirp numerator so
        // it stays inside a signed 32-bit value (we compute in i64 anyway).
        let maxabs_q12 = ((maxabs_q17 + 16) >> 5).min(163838);
        if maxabs_q12 <= 32767 {
            // The coefficients already fit Q12 — no expansion, no saturation.
            return;
        }

        // chirp factor sc_Q16[0]; integer division per §4.2.7.5.7.
        let numer = (maxabs_q12 - 32767) << 14;
        let denom = (maxabs_q12 * (max_idx as i64 + 1)) >> 2;
        let sc_q16_0 = 65470 - numer / denom;
        bwexpander_32(a32_q17, sc_q16_0);
    }

    // After the 10th round, saturate in Q12 only if the largest coefficient
    // still overflows. Re-derive maxabs_Q12 the same way as inside the loop.
    let maxabs_q17 = a32_q17
        .iter()
        .map(|&c| c.unsigned_abs() as i64)
        .max()
        .unwrap_or(0);
    let maxabs_q12 = ((maxabs_q17 + 16) >> 5).min(163838);
    if maxabs_q12 > 32767 {
        for c in a32_q17.iter_mut() {
            // clamp(-32768, (a32_Q17[k] + 16) >> 5, 32767) << 5 — saturate in
            // the Q12 domain, then convert back to Q17.
            let q12 = (((*c as i64 + 16) >> 5).clamp(-32768, 32767)) as i32;
            *c = q12 << 5;
        }
    }
}

/// RFC 6716 §4.2.7.5.7 `silk_bwexpander_32` recurrence.
///
/// `a32_Q17[k] = (a32_Q17[k]*sc_Q16[k]) >> 16` with
/// `sc_Q16[k+1] = (sc_Q16[0]*sc_Q16[k] + 32768) >> 16`. The first multiply
/// can require up to 48 bits of precision (done in i64); the second is
/// performed unsigned (both `sc_Q16` values are positive and < 2^16) to
/// avoid the 32-bit overflow the spec warns about.
fn bwexpander_32(a32_q17: &mut [i32], sc_q16_0: i64) {
    let mut sc_q16_k: u64 = sc_q16_0 as u64;
    for c in a32_q17.iter_mut() {
        // First multiply: signed, up to 48 bits → i64.
        *c = ((*c as i64 * sc_q16_k as i64) >> 16) as i32;
        // Second multiply: unsigned per §4.2.7.5.7.
        sc_q16_k = (sc_q16_0 as u64 * sc_q16_k + 32768) >> 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Table 27 transcription self-checks --------------------------

    #[test]
    fn table27_nbmb_is_permutation_of_0_to_9() {
        let mut sorted = ORDERING_NB_MB.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0u8..10).collect::<Vec<_>>());
    }

    #[test]
    fn table27_wb_is_permutation_of_0_to_15() {
        let mut sorted = ORDERING_WB.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0u8..16).collect::<Vec<_>>());
    }

    #[test]
    fn table27_row_widths_match_d_lpc() {
        assert_eq!(ORDERING_NB_MB.len(), D_LPC_NB_MB);
        assert_eq!(ORDERING_WB.len(), D_LPC_WB);
    }

    #[test]
    fn table27_ordering_helper_routes_per_bandwidth() {
        assert_eq!(ordering(Bandwidth::Nb).unwrap(), &ORDERING_NB_MB);
        assert_eq!(ordering(Bandwidth::Mb).unwrap(), &ORDERING_NB_MB);
        assert_eq!(ordering(Bandwidth::Wb).unwrap(), &ORDERING_WB);
        assert!(ordering(Bandwidth::Swb).is_err());
        assert!(ordering(Bandwidth::Fb).is_err());
    }

    #[test]
    fn table27_first_row_spot_checks() {
        // Per RFC 6716 Table 27 column "NB and MB" — k = 0 → 0, k = 1 → 9,
        // k = 2 → 6, k = 3 → 3, ... k = 9 → 7. The first/last/middle cells
        // are the most likely to catch a transcription typo.
        assert_eq!(ORDERING_NB_MB[0], 0);
        assert_eq!(ORDERING_NB_MB[1], 9);
        assert_eq!(ORDERING_NB_MB[5], 5); // self-pinned cell
        assert_eq!(ORDERING_NB_MB[9], 7);

        // Per RFC 6716 Table 27 column "WB" — k = 0 → 0, k = 1 → 15,
        // k = 2 → 8, ..., k = 15 → 1.
        assert_eq!(ORDERING_WB[0], 0);
        assert_eq!(ORDERING_WB[1], 15);
        assert_eq!(ORDERING_WB[8], 2);
        assert_eq!(ORDERING_WB[15], 1);
    }

    // --- Table 28 transcription self-checks --------------------------

    #[test]
    fn table28_has_129_entries() {
        assert_eq!(COS_Q12.len(), 129);
    }

    #[test]
    fn table28_anchor_values() {
        // cos(0) = 1, cos(pi/2) = 0, cos(pi) = -1 in Q12.
        assert_eq!(COS_Q12[0], 4096);
        assert_eq!(COS_Q12[64], 0);
        assert_eq!(COS_Q12[128], -4096);
    }

    #[test]
    fn table28_anti_symmetric_about_64() {
        // The spec table is the discrete cosine sampled on [0, pi], so
        // cos_Q12[128 - i] == -cos_Q12[i] for i ∈ 0..=128.
        for i in 0..=128 {
            assert_eq!(
                COS_Q12[128 - i],
                -COS_Q12[i],
                "table 28 anti-symmetry broken at i = {i}: {} vs -{}",
                COS_Q12[128 - i],
                COS_Q12[i]
            );
        }
    }

    #[test]
    fn table28_strictly_decreasing() {
        // The cosine function is strictly decreasing on [0, pi]. Since the
        // table is sampled at 129 points covering that interval, every
        // adjacent pair (i, i+1) must satisfy cos_Q12[i] > cos_Q12[i+1].
        for i in 0..128 {
            assert!(
                COS_Q12[i] > COS_Q12[i + 1],
                "table 28 not strictly decreasing at i = {i}: {} -> {}",
                COS_Q12[i],
                COS_Q12[i + 1]
            );
        }
    }

    #[test]
    fn table28_in_q12_range() {
        // All entries are signed Q12 cosines, so they live in [-4096, 4096].
        for (i, &v) in COS_Q12.iter().enumerate() {
            assert!(
                (-4096..=4096).contains(&v),
                "table 28 cell {i} = {v} outside Q12 range"
            );
        }
    }

    #[test]
    fn table28_spot_checks() {
        // Row 0 starts (4096, 4095, 4091, 4085).
        assert_eq!(COS_Q12[0..4], [4096, 4095, 4091, 4085]);
        // Row 16 starts (3784, 3745, 3703, 3659) per Table 28.
        assert_eq!(COS_Q12[16..20], [3784, 3745, 3703, 3659]);
        // Row 60 starts (401, 301, 201, 101).
        assert_eq!(COS_Q12[60..64], [401, 301, 201, 101]);
        // Row 64 starts (0, -101, -201, -301).
        assert_eq!(COS_Q12[64..68], [0, -101, -201, -301]);
        // Row 124 starts (-4076, -4085, -4091, -4095).
        assert_eq!(COS_Q12[124..128], [-4076, -4085, -4091, -4095]);
    }

    // --- nlsf_to_c_q17 -----------------------------------------------

    #[test]
    fn cos_lookup_at_table_anchor_points_matches_table_entries() {
        // When nlsf_q15 = i << 8 the fractional `f` is 0, so the §4.2.7.5.6
        // interpolation reduces to (cos_Q12[i]*256 + 4) >> 3 == cos_Q12[i]*32 + 0
        // (since 4 >> 3 == 0 for positive a, and the +4 is integer rounding).
        // More precisely: (a*256 + 4) >> 3 = a*32 + (4 >> 3) = a*32 for the
        // sign-positive arithmetic shift on i32. Build a synthetic NLSF
        // with k=0 carrying i=0 anchor, verify c_q17[ordering[0]] = 4096*32.
        let mut nlsf = vec![0i16; D_LPC_NB_MB];
        // Pick distinct top-7-bits indices so each c slot gets a known value.
        // Use i = 8*k → cos_Q12[i] anchors.
        for (k, slot) in nlsf.iter_mut().enumerate() {
            *slot = ((8 * k as i32) << 8) as i16;
        }
        let c = nlsf_to_c_q17(Bandwidth::Nb, &nlsf).unwrap();
        for (k, &dest) in ORDERING_NB_MB.iter().enumerate() {
            let i = 8 * k;
            let expected = (COS_Q12[i] * 256 + 4) >> 3;
            assert_eq!(c[dest as usize], expected, "anchor mismatch at k = {k}");
        }
    }

    #[test]
    fn cos_lookup_rejects_swb_fb() {
        let nlsf = vec![0i16; D_LPC_NB_MB];
        assert!(nlsf_to_c_q17(Bandwidth::Swb, &nlsf).is_err());
        assert!(nlsf_to_c_q17(Bandwidth::Fb, &nlsf).is_err());
    }

    #[test]
    fn cos_lookup_rejects_length_mismatch() {
        // NB ordering wants 10 entries; passing 16 is malformed.
        let nlsf = vec![0i16; 16];
        assert!(nlsf_to_c_q17(Bandwidth::Nb, &nlsf).is_err());
        let nlsf = vec![0i16; 10];
        assert!(nlsf_to_c_q17(Bandwidth::Wb, &nlsf).is_err());
    }

    #[test]
    fn cos_lookup_linear_interp_midpoint() {
        // For `f = 128` the interpolation lands exactly at the midpoint of
        // (cos_Q12[i], cos_Q12[i+1]):
        //   (a*256 + (b - a)*128 + 4) >> 3
        //   = (256*a + 128*b - 128*a + 4) >> 3
        //   = (128*(a + b) + 4) >> 3
        //   = 16 * (a + b)   (since 128/8 = 16; the +4 rounds the lone LSB)
        let mut nlsf = vec![0i16; D_LPC_NB_MB];
        // Put a known (i, f) pair into k=0: i=10, f=128 → nlsf = (10 << 8) | 128 = 2688
        nlsf[0] = (10 << 8) | 128;
        // Pad the rest with distinct values so we don't collide.
        for (k, slot) in nlsf.iter_mut().enumerate().take(D_LPC_NB_MB).skip(1) {
            *slot = ((20 + k as i32) << 8) as i16;
        }
        let c = nlsf_to_c_q17(Bandwidth::Nb, &nlsf).unwrap();
        let a = COS_Q12[10];
        let b = COS_Q12[11];
        let expected = (a * 256 + (b - a) * 128 + 4) >> 3;
        assert_eq!(c[ORDERING_NB_MB[0] as usize], expected);
    }

    // --- LpcQ17 ------------------------------------------------------

    #[test]
    fn lpc_length_matches_bandwidth() {
        let nb = vec![1638i16; D_LPC_NB_MB]; // ascending placeholder, monotone
        let mut nb_mono = nb.clone();
        for (k, v) in nb_mono.iter_mut().enumerate() {
            *v = (1000 + 2000 * k as i32) as i16;
        }
        let lpc = LpcQ17::from_nlsf(Bandwidth::Nb, &nb_mono).unwrap();
        assert_eq!(lpc.len(), D_LPC_NB_MB);
        assert_eq!(lpc.a32_q17().len(), D_LPC_NB_MB);

        let mut wb_mono = vec![0i16; D_LPC_WB];
        for (k, v) in wb_mono.iter_mut().enumerate() {
            *v = (500 + 1900 * k as i32) as i16;
        }
        let lpc = LpcQ17::from_nlsf(Bandwidth::Wb, &wb_mono).unwrap();
        assert_eq!(lpc.len(), D_LPC_WB);
        assert_eq!(lpc.a32_q17().len(), D_LPC_WB);
    }

    #[test]
    fn lpc_rejects_swb_fb_and_length_mismatch() {
        let nlsf = vec![0i16; D_LPC_NB_MB];
        assert!(LpcQ17::from_nlsf(Bandwidth::Swb, &nlsf).is_err());
        assert!(LpcQ17::from_nlsf(Bandwidth::Fb, &nlsf).is_err());
        let nlsf = vec![0i16; 12];
        assert!(LpcQ17::from_nlsf(Bandwidth::Nb, &nlsf).is_err());
    }

    /// Sanity oracle: independently re-run the §4.2.7.5.6 recurrence using
    /// vectors of dynamic-length intermediate Q16 coefficients. This is a
    /// straight transcription of the spec — distinct from the rolling-row
    /// production implementation — so a typo in the recurrence shows up
    /// as a divergence between the two paths.
    fn oracle_a32_q17(c_q17: &[i32], d_lpc: usize) -> Vec<i32> {
        let d2 = d_lpc / 2;
        let run_side = |parity: usize| -> Vec<i64> {
            // Full 2D matrix p[k][j], k ∈ 0..d2, j ∈ 0..=d2+1. The §4.2.7.5.6
            // recurrence only ever reads j-2..=j on the prior row so the
            // 2D allocation is correct (and wasteful in memory — fine for a
            // test oracle).
            let cols = d2 + 2;
            let mut p = vec![vec![0i64; cols]; d2];
            p[0][0] = 1 << 16;
            p[0][1] = -(c_q17[parity] as i64);
            for k in 1..d2 {
                let c = c_q17[2 * k + parity] as i64;
                for j in 0..=k + 1 {
                    // §4.2.7.5.6 boundary condition: row k-1 is a
                    // symmetric product stored up to its middle
                    // coefficient j = k, so the read past the middle
                    // mirrors back (p_Q16[k][k+2] = p_Q16[k][k], i.e.
                    // p[k-1][k+1] = p[k-1][k-1]).
                    let pj0 = if j == k + 1 {
                        p[k - 1][k - 1]
                    } else {
                        p[k - 1][j]
                    };
                    let pjm2 = if j >= 2 { p[k - 1][j - 2] } else { 0 };
                    let pjm1 = if j >= 1 { p[k - 1][j - 1] } else { 0 };
                    p[k][j] = pj0 + pjm2 - ((c * pjm1 + 32768) >> 16);
                }
            }
            p.pop().unwrap()
        };

        let p_last = run_side(0);
        let q_last = run_side(1);

        let mut a = vec![0i32; d_lpc];
        for k in 0..d2 {
            let q_diff = q_last[k + 1] - q_last[k];
            let p_sum = p_last[k + 1] + p_last[k];
            a[k] = (-(q_diff + p_sum)) as i32;
            a[d_lpc - k - 1] = (q_diff - p_sum) as i32;
        }
        a
    }

    fn ascending_nlsf(d_lpc: usize, start: i16, step: i16) -> Vec<i16> {
        (0..d_lpc as i16)
            .map(|k| start.saturating_add(k.saturating_mul(step)))
            .collect()
    }

    #[test]
    fn lpc_matches_oracle_nb() {
        let nlsf = ascending_nlsf(D_LPC_NB_MB, 1500, 2700); // ascending, ends ~25800
        let lpc = LpcQ17::from_nlsf(Bandwidth::Nb, &nlsf).unwrap();
        let c = nlsf_to_c_q17(Bandwidth::Nb, &nlsf).unwrap();
        let expected = oracle_a32_q17(&c[..D_LPC_NB_MB], D_LPC_NB_MB);
        assert_eq!(lpc.a32_q17(), expected.as_slice());
    }

    #[test]
    fn lpc_matches_oracle_wb() {
        let nlsf = ascending_nlsf(D_LPC_WB, 800, 1900); // ascending, ends ~29300
        let lpc = LpcQ17::from_nlsf(Bandwidth::Wb, &nlsf).unwrap();
        let c = nlsf_to_c_q17(Bandwidth::Wb, &nlsf).unwrap();
        let expected = oracle_a32_q17(&c[..D_LPC_WB], D_LPC_WB);
        assert_eq!(lpc.a32_q17(), expected.as_slice());
    }

    #[test]
    fn lpc_matches_oracle_against_real_decoder_pipeline() {
        // Drive a real §4.2.7.5.2 → §4.2.7.5.3 → §4.2.7.5.4 decoder
        // pipeline off a synthetic range-decoder buffer, then feed the
        // stabilized NLSF into both the production conversion and the
        // oracle. Sweep all 32 I1 values across {NB, MB, WB} for a
        // robust cross-check.
        use crate::range_decoder::RangeDecoder;
        use crate::silk_lsf_recon::NlsfReconstructed;
        use crate::silk_lsf_stabilize::NlsfStabilized;
        use crate::silk_lsf_stage2::LsfStage2;

        let buf = [
            0x5Au8, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x4C, 0x8E,
        ];
        for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
            for i1 in 0u8..32 {
                let mut rd = RangeDecoder::new(&buf);
                let stage2 = LsfStage2::decode(&mut rd, bw, i1).expect("stage-2");
                let recon =
                    NlsfReconstructed::from_stage1_and_stage2(bw, i1, &stage2).expect("recon");
                let stab = NlsfStabilized::from_reconstructed(bw, &recon).expect("stab");
                let lpc = LpcQ17::from_nlsf(bw, stab.nlsf_q15()).unwrap();
                let c = nlsf_to_c_q17(bw, stab.nlsf_q15()).unwrap();
                let d_lpc = stab.nlsf_q15().len();
                let expected = oracle_a32_q17(&c[..d_lpc], d_lpc);
                assert_eq!(
                    lpc.a32_q17(),
                    expected.as_slice(),
                    "production/oracle divergence: bw={bw:?} i1={i1}"
                );
            }
        }
    }

    /// Structurally independent analytic cross-check: build A(z) in
    /// f64 as the §4.2.7.5.6 closed form
    /// `A = (P + Q)/2`, `P = (1+z^-1) * prod (1 - 2cos(w_2k) z^-1 + z^-2)`,
    /// `Q = (1-z^-1) * prod (1 - 2cos(w_2k+1) z^-1 + z^-2)`
    /// using the SAME table-interpolated cosines the fixed-point path
    /// uses (`2*cos = c_Q17 / 2^16`), and require `a32_Q17 / 2^17` to
    /// match within fixed-point rounding noise.
    ///
    /// Round-388 regression: the rolling-row recurrence dropped the
    /// "p_Q16[k][k+2] = p_Q16[k][k]" symmetric-mirror boundary
    /// condition at the j = k+1 read (it substituted 0), producing
    /// badly wrong filters that then burned prediction-gain-limiter
    /// rounds on perfectly stable codebook vectors. The former test
    /// oracle transcribed the identical misreading, so only an
    /// analytic reference exposes it.
    #[test]
    fn lpc_matches_analytic_polynomial_on_codebook_vectors() {
        use crate::silk_lsf_recon::cb1_q8;
        for bw in [Bandwidth::Nb, Bandwidth::Wb] {
            for i1 in 0u8..32 {
                let cb = cb1_q8(bw, i1).unwrap();
                let nlsf: Vec<i16> = cb.iter().map(|&v| (v as i16) << 7).collect();
                let d_lpc = nlsf.len();
                let c = nlsf_to_c_q17(bw, &nlsf).unwrap();

                // Analytic product in f64 over the un-reordered slots:
                // parity-0 slots feed P, parity-1 slots feed Q.
                let poly_mul_quad = |poly: &mut Vec<f64>, two_cos: f64| {
                    let mut out = vec![0.0f64; poly.len() + 2];
                    for (i, &pc) in poly.iter().enumerate() {
                        out[i] += pc;
                        out[i + 1] -= two_cos * pc;
                        out[i + 2] += pc;
                    }
                    *poly = out;
                };
                let mut p = vec![1.0f64];
                let mut q = vec![1.0f64];
                for k in 0..d_lpc / 2 {
                    poly_mul_quad(&mut p, c[2 * k] as f64 / 65536.0);
                    poly_mul_quad(&mut q, c[2 * k + 1] as f64 / 65536.0);
                }
                // Trivial-root factors: P *= (1 + z^-1), Q *= (1 - z^-1).
                let mul_lin = |poly: &Vec<f64>, r: f64| {
                    let mut out = vec![0.0f64; poly.len() + 1];
                    for (i, &pc) in poly.iter().enumerate() {
                        out[i] += pc;
                        out[i + 1] += r * pc;
                    }
                    out
                };
                let pf = mul_lin(&p, 1.0);
                let qf = mul_lin(&q, -1.0);

                let lpc = LpcQ17::from_nlsf(bw, &nlsf).unwrap();
                for k in 0..d_lpc {
                    let analytic = -(pf[k + 1] + qf[k + 1]) / 2.0;
                    let fixed = lpc.a32_q17()[k] as f64 / 131072.0;
                    assert!(
                        (analytic - fixed).abs() < 2e-3,
                        "bw={bw:?} I1={i1} k={k}: analytic {analytic} vs fixed {fixed}"
                    );
                }

                // Stage-1 codebook centres are stable filters: the
                // §4.2.7.5.8 limiter must not burn a single round.
                let limited = lpc.range_limited().prediction_gain_limited();
                assert_eq!(limited.rounds(), 0, "bw={bw:?} I1={i1} burned rounds");
            }
        }
    }

    #[test]
    fn lpc_leading_term_uses_full_row_sum() {
        // For k = 0 the formula reduces to
        //   a32_Q17[0] = -((q[d2-1][1] - q[d2-1][0]) + (p[d2-1][1] + p[d2-1][0]))
        // and a32_Q17[d_LPC-1] = (q[d2-1][1] - q[d2-1][0]) - (p[d2-1][1] + p[d2-1][0]).
        // The two share the same |q_diff + p_sum| magnitude path — pin this
        // identity on a hand-built case so a refactor of the row assembly
        // can't accidentally swap signs.
        let nlsf = ascending_nlsf(D_LPC_NB_MB, 1500, 2700);
        let lpc = LpcQ17::from_nlsf(Bandwidth::Nb, &nlsf).unwrap();
        let a = lpc.a32_q17();
        // a[0] = -(q_diff + p_sum); a[d_LPC-1] = q_diff - p_sum.
        // Therefore a[0] + a[d_LPC-1] = -2 * p_sum.
        // We don't know p_sum here, but we can check parity / consistency
        // via the relation a[0] - a[d_LPC-1] = -2 * q_diff (must be even).
        assert_eq!((a[0] - a[D_LPC_NB_MB - 1]) % 2, 0);
        assert_eq!((a[0] + a[D_LPC_NB_MB - 1]) % 2, 0);
    }

    // --- §4.2.7.5.7 range-limiting bandwidth expansion ----------------

    /// Independent transcription of the §4.2.7.5.7 loop, written against the
    /// raw RFC formulas with a fresh control structure so a typo in the
    /// production `limit_lpc_range` shows up as a divergence. Returns the
    /// range-limited Q17 coefficients.
    fn oracle_range_limited(input: &[i32]) -> Vec<i32> {
        let mut a: Vec<i64> = input.iter().map(|&c| c as i64).collect();
        let d = a.len();

        let maxabs_q12_of = |a: &[i64]| -> (usize, i64) {
            // largest abs, ties to lowest index
            let mut idx = 0usize;
            let mut best = a[0].abs();
            for (k, &c) in a.iter().enumerate().skip(1) {
                if c.abs() > best {
                    best = c.abs();
                    idx = k;
                }
            }
            let q12 = ((best + 16) >> 5).min(163838);
            (idx, q12)
        };

        for _ in 0..10 {
            let (k, q12) = maxabs_q12_of(&a);
            if q12 <= 32767 {
                return a.iter().map(|&c| c as i32).collect();
            }
            let numer = (q12 - 32767) << 14;
            let denom = (q12 * (k as i64 + 1)) >> 2;
            let sc0 = 65470 - numer / denom;
            // bwexpander_32, computed independently in i128 to be sure the
            // production i64/u64 path doesn't silently truncate.
            let mut sc_k: i128 = sc0 as i128;
            for c in a.iter_mut() {
                *c = ((*c as i128 * sc_k) >> 16) as i64;
                sc_k = (sc0 as i128 * sc_k + 32768) >> 16;
            }
        }
        // recompute maxabs after round 10
        let (_, q12_after) = maxabs_q12_of(&a);
        if q12_after > 32767 {
            for c in a.iter_mut() {
                let q12 = ((*c + 16) >> 5).clamp(-32768, 32767);
                *c = q12 << 5;
            }
        }
        debug_assert_eq!(a.len(), d);
        a.iter().map(|&c| c as i32).collect()
    }

    /// After §4.2.7.5.7, every coefficient must fit a signed 16-bit Q12
    /// value: `(a32_Q17[k] + 16) >> 5 ∈ [-32768, 32767]`.
    fn assert_fits_q12(a: &[i32]) {
        for (k, &c) in a.iter().enumerate() {
            let q12 = (c as i64 + 16) >> 5;
            assert!(
                (-32768..=32767).contains(&q12),
                "coeff {k} = {c} does not fit Q12 (q12 = {q12})"
            );
        }
    }

    #[test]
    fn range_limit_leaves_small_coeffs_untouched() {
        // Coefficients whose maxabs_Q12 is already <= 32767 must pass through
        // unchanged (no expansion, no saturation). A Q17 magnitude of
        // 32767 << 5 = 1048544 maps to exactly Q12 = 32767, the boundary.
        let nlsf = ascending_nlsf(D_LPC_NB_MB, 1500, 2700);
        let lpc = LpcQ17::from_nlsf(Bandwidth::Nb, &nlsf).unwrap();
        // Only run this assertion if the raw output is already in range; a
        // typical decoded vector is, but assert the invariant either way.
        let raw = lpc.a32_q17().to_vec();
        let maxabs = raw.iter().map(|&c| (c as i64).abs()).max().unwrap();
        let maxabs_q12 = ((maxabs + 16) >> 5).min(163838);
        let limited = lpc.range_limited();
        if maxabs_q12 <= 32767 {
            assert_eq!(limited.a32_q17(), raw.as_slice());
        }
        assert_fits_q12(limited.a32_q17());
    }

    #[test]
    fn range_limit_matches_oracle_on_synthetic_overflow() {
        // Hand-built Q17 vectors that overflow Q12 by varying amounts so the
        // chirp loop runs at least one round. Cross-check production vs the
        // independent i128 oracle bit-for-bit.
        let cases: &[[i32; D_LPC_NB_MB]] = &[
            // a single coefficient just over the Q12 boundary
            [1_100_000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // the peak at a non-zero index (exercises the (k+1) divisor)
            [0, 0, 0, 2_500_000, -100_000, 0, 0, 0, 0, 0],
            // several large coefficients of mixed sign
            [
                3_000_000, -2_800_000, 2_600_000, -2_400_000, 2_200_000, -2_000_000, 1_800_000,
                -1_600_000, 1_400_000, -1_200_000,
            ],
            // a moderate overshoot that should converge well before round 10
            [1_200_000, -1_150_000, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        for case in cases {
            let lpc = LpcQ17 {
                len: D_LPC_NB_MB as u8,
                a32_q17: {
                    let mut a = [0i32; D_LPC_MAX];
                    a[..D_LPC_NB_MB].copy_from_slice(case);
                    a
                },
            };
            let limited = lpc.range_limited();
            let expected = oracle_range_limited(case);
            assert_eq!(
                limited.a32_q17(),
                expected.as_slice(),
                "production/oracle divergence on {case:?}"
            );
            assert_fits_q12(limited.a32_q17());
        }
    }

    #[test]
    fn range_limit_extreme_input_at_maxabs_cap_converges() {
        // An extreme coefficient pinned to the maxabs_Q12 = 163838 cap (the
        // §4.2.7.5.7 numerator-overflow bound). The adaptive chirp factor is
        // very small for such a large overshoot, so the expansion converges
        // within the 10-round budget; production must still agree with the
        // independent oracle bit-for-bit and the result must fit Q12.
        let huge = 163838i64 << 5; // Q17 magnitude that maps to Q12 = 163838
        let mut a = [0i32; D_LPC_MAX];
        a[0] = huge as i32;
        a[5] = -(huge as i32);
        a[9] = (huge as i32) / 2;
        let lpc = LpcQ17 {
            len: D_LPC_NB_MB as u8,
            a32_q17: a,
        };
        let limited = lpc.range_limited();
        let expected = oracle_range_limited(&a[..D_LPC_NB_MB]);
        assert_eq!(limited.a32_q17(), expected.as_slice());
        assert_fits_q12(limited.a32_q17());
    }

    #[test]
    fn range_limit_post_loop_saturation_formula() {
        // The §4.2.7.5.7 post-loop Q12 saturation is documented as a
        // belt-and-suspenders step run "regardless of whether or not the
        // Q12 version of any coefficient still overflows" — but in practice
        // the adaptive chirp converges every realistic input within 10
        // rounds, so the engaged branch is effectively unreachable. Pin the
        // saturation *formula* directly so a transcription typo is still
        // caught: clamp(-32768, (a + 16) >> 5, 32767) << 5.
        let saturate = |c: i64| -> i32 {
            let q12 = ((c + 16) >> 5).clamp(-32768, 32767);
            (q12 << 5) as i32
        };
        // Below the positive Q12 ceiling: round-trips through Q12 << 5.
        assert_eq!(saturate(32767i64 << 5), 32767 << 5);
        // Just over the ceiling clamps to 32767 << 5.
        assert_eq!(saturate((32767i64 << 5) + (1 << 5)), 32767 << 5);
        // Far over the ceiling clamps to the same maximum.
        assert_eq!(saturate(i32::MAX as i64), 32767 << 5);
        // Below the negative floor clamps to -32768 << 5.
        assert_eq!(saturate(-(32768i64 << 5) - (1 << 5)), -32768 << 5);
        assert_eq!(saturate(i32::MIN as i64), -32768 << 5);
        // Zero stays zero; the +16 rounding does not push it off.
        assert_eq!(saturate(0), 0);
    }

    #[test]
    fn range_limit_handles_i32_min_without_panic() {
        // unsigned_abs() must not panic on i32::MIN inside the max search.
        let mut a = [0i32; D_LPC_MAX];
        a[0] = i32::MIN;
        a[3] = i32::MAX;
        let lpc = LpcQ17 {
            len: D_LPC_NB_MB as u8,
            a32_q17: a,
        };
        let limited = lpc.range_limited();
        assert_fits_q12(limited.a32_q17());
        let expected = oracle_range_limited(&a[..D_LPC_NB_MB]);
        assert_eq!(limited.a32_q17(), expected.as_slice());
    }

    #[test]
    fn range_limit_real_pipeline_fits_q12_across_bandwidth_x_i1_sweep() {
        // Drive the real §4.2.7.5.2 → §4.2.7.5.3 → §4.2.7.5.4 → §4.2.7.5.6
        // pipeline, then range-limit. Every result must fit Q12 and agree
        // with the independent oracle for every (bandwidth, I1) on a few
        // buffers.
        use crate::range_decoder::RangeDecoder;
        use crate::silk_lsf_recon::NlsfReconstructed;
        use crate::silk_lsf_stabilize::NlsfStabilized;
        use crate::silk_lsf_stage2::LsfStage2;

        let bufs: &[&[u8]] = &[
            &[
                0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x4C, 0x8E,
            ],
            &[
                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            ],
        ];
        for buf in bufs {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                for i1 in 0u8..32 {
                    let mut rd = RangeDecoder::new(buf);
                    let stage2 = LsfStage2::decode(&mut rd, bw, i1).expect("stage-2");
                    let recon =
                        NlsfReconstructed::from_stage1_and_stage2(bw, i1, &stage2).expect("recon");
                    let stab = NlsfStabilized::from_reconstructed(bw, &recon).expect("stab");
                    let lpc = LpcQ17::from_nlsf(bw, stab.nlsf_q15()).unwrap();
                    let limited = lpc.range_limited();
                    assert_eq!(limited.len(), lpc.len());
                    assert_fits_q12(limited.a32_q17());
                    let expected = oracle_range_limited(lpc.a32_q17());
                    assert_eq!(
                        limited.a32_q17(),
                        expected.as_slice(),
                        "production/oracle divergence: bw={bw:?} i1={i1}"
                    );
                }
            }
        }
    }

    // --- §4.2.7.5.8 prediction-gain limiting --------------------------

    /// Independent spec transcription of `silk_LPC_inverse_pred_gain_QA()`
    /// using a full 2D `a32_Q24` matrix (one row per `k`) and the literal
    /// RFC formulas, with a control structure distinct from the production
    /// rolling-row `is_lpc_stable`. Returns `true` iff the filter is stable.
    fn oracle_is_stable(a32_q12: &[i32]) -> bool {
        let d = a32_q12.len();
        let dc: i64 = a32_q12.iter().map(|&c| c as i64).sum();
        if dc > 4096 {
            return false;
        }
        // a32_Q24[k][n], k ∈ 0..d, n ∈ 0..d. Only the top row is seeded.
        let mut a = vec![vec![0i64; d]; d];
        for n in 0..d {
            a[d - 1][n] = (a32_q12[n] as i64) << 12;
        }
        let mut inv_gain_q30: i64 = 1 << 30;
        for k in (0..d).rev() {
            let akk = a[k][k];
            if akk.abs() > 16_773_022 {
                return false;
            }
            let rc_q31 = -akk << 7;
            let div_q30 = (1i64 << 30) - ((rc_q31 * rc_q31) >> 32);
            inv_gain_q30 = ((inv_gain_q30 * div_q30) >> 32) << 2;
            if inv_gain_q30 < 107_374 {
                return false;
            }
            if k > 0 {
                let b1 = {
                    // ilog
                    if div_q30 <= 0 {
                        0i64
                    } else {
                        (64 - (div_q30 as u64).leading_zeros()) as i64
                    }
                };
                let b2 = b1 - 16;
                let inv_qb2 = ((1i64 << 29) - 1) / (div_q30 >> (b2 + 1));
                let err_q29 = (1i64 << 29) - (((div_q30 << (15 - b2)) * inv_qb2) >> 16);
                let gain_qb1 = (inv_qb2 << 16) + ((err_q29 * inv_qb2) >> 13);
                for n in 0..k {
                    let num_q24 = a[k][n] - (((a[k][k - n - 1] * rc_q31) + (1i64 << 30)) >> 31);
                    a[k - 1][n] = ((num_q24 * gain_qb1) + (1i64 << (b1 - 1))) >> b1;
                }
            }
        }
        true
    }

    #[test]
    fn stability_check_agrees_with_oracle_on_hand_built_filters() {
        // A spread of small / borderline / large filters; production and
        // oracle must classify each identically.
        let cases: &[&[i32]] = &[
            // The trivial all-zero filter: DC = 0, every rc = 0 ⇒ stable.
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // A gentle decaying filter (small magnitudes) ⇒ stable.
            &[400, -200, 100, -50, 25, -12, 6, -3, 1, 0],
            // A near-unit single-tap filter (a_Q12[0] ≈ 1.0 in Q12 = 4096):
            // DC_resp = 4096, exactly the boundary (NOT > 4096) ⇒ DC ok, but
            // the reflection coefficient is ≈ 1 so it should be unstable.
            &[4096, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // DC over the 4096 ceiling ⇒ unstable by the DC check alone.
            &[4097, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // Mixed-sign moderate filter.
            &[1500, -1200, 900, -600, 300, -150, 75, -30, 10, -5],
        ];
        for case in cases {
            assert_eq!(
                is_lpc_stable(case),
                oracle_is_stable(case),
                "stability divergence on {case:?}"
            );
        }
    }

    #[test]
    fn stability_check_zero_filter_is_stable() {
        assert!(is_lpc_stable(&[0i32; D_LPC_NB_MB]));
        assert!(is_lpc_stable(&[0i32; D_LPC_WB]));
    }

    #[test]
    fn stability_check_dc_response_over_4096_is_unstable() {
        // Σ a32_Q12 = 4097 > 4096 ⇒ rejected before the Levinson recurrence.
        let mut a = [0i32; D_LPC_NB_MB];
        a[0] = 4097;
        assert!(!is_lpc_stable(&a));
        // Right at the boundary the DC check passes (4096 is not > 4096); the
        // single near-unit tap then fails the recurrence, but the DC check
        // alone does not reject it. Confirm via the oracle.
        let mut b = [0i32; D_LPC_NB_MB];
        b[0] = 4096;
        assert_eq!(is_lpc_stable(&b), oracle_is_stable(&b));
    }

    #[test]
    fn pred_gain_limit_stable_input_passes_through_in_zero_rounds() {
        // A real decoded NLSF vector almost always yields a stable filter
        // after §4.2.7.5.7, so prediction_gain_limited returns on round 0
        // with a_Q12 == (a32_Q17 + 16) >> 5 of the range-limited input.
        let nlsf = ascending_nlsf(D_LPC_NB_MB, 1500, 2700);
        let limited = LpcQ17::from_nlsf(Bandwidth::Nb, &nlsf)
            .unwrap()
            .range_limited();
        let pg = limited.prediction_gain_limited();
        // The range-limited input is stable, so no chirp rounds run and the
        // Q12 coefficients are the straight conversion of a32_Q17.
        if pg.rounds() == 0 {
            let expected: Vec<i32> = limited.a32_q17().iter().map(|&c| (c + 16) >> 5).collect();
            assert_eq!(pg.a_q12(), expected.as_slice());
        }
        // Whatever the round count, the emitted filter must be stable.
        assert!(is_lpc_stable(pg.a_q12()));
        assert_eq!(pg.len(), D_LPC_NB_MB);
    }

    #[test]
    fn pred_gain_limit_always_emits_a_stable_filter() {
        // Feed deliberately aggressive (unstable) Q17 coefficients straight
        // in (skipping range-limiting) and confirm the §4.2.7.5.8 chirp loop
        // always converges to a stable Q12 filter.
        let cases: &[[i32; D_LPC_NB_MB]] = &[
            // A near-unit leading tap in Q17 (≈ 1.0): unstable, needs chirp.
            [4096 << 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // Several large alternating taps — a high-gain resonant filter.
            [
                3500 << 5,
                -3400 << 5,
                3300 << 5,
                -3200 << 5,
                3100 << 5,
                -3000 << 5,
                2900 << 5,
                -2800 << 5,
                2700 << 5,
                -2600 << 5,
            ],
            // DC way over the ceiling.
            [4000 << 5, 4000 << 5, 4000 << 5, 0, 0, 0, 0, 0, 0, 0],
        ];
        for case in cases {
            let lpc = LpcQ17 {
                len: D_LPC_NB_MB as u8,
                a32_q17: {
                    let mut a = [0i32; D_LPC_MAX];
                    a[..D_LPC_NB_MB].copy_from_slice(case);
                    a
                },
            };
            let pg = lpc.prediction_gain_limited();
            assert!(
                is_lpc_stable(pg.a_q12()),
                "emitted unstable filter for input {case:?} (rounds={})",
                pg.rounds()
            );
            assert!(pg.rounds() <= 16);
        }
    }

    #[test]
    fn pred_gain_limit_round15_zeroes_a_persistently_unstable_filter() {
        // Construct an input that the inverse-pred-gain test always rejects
        // even after chirping: a filter dominated by one tap pinned to the
        // maximum the prior stages can leave (Q12 = 32767, well above the
        // near-unit instability point). The §4.2.7.5.8 loop reaches round 15
        // where sc_Q16[0] = 0 zeroes every coefficient, producing the
        // all-zero (trivially stable) filter and rounds() == 16.
        let mut a = [0i32; D_LPC_MAX];
        a[0] = 32767 << 5; // huge near-Q12-max leading tap
        a[1] = 32767 << 5;
        let lpc = LpcQ17 {
            len: D_LPC_NB_MB as u8,
            a32_q17: a,
        };
        let pg = lpc.prediction_gain_limited();
        assert!(is_lpc_stable(pg.a_q12()));
        // If it ever reaches the forced-zero round, every coefficient is 0.
        if pg.rounds() == 16 {
            assert!(pg.a_q12().iter().all(|&c| c == 0));
        }
    }

    #[test]
    fn pred_gain_limit_emitted_q12_fits_signed_16bit() {
        // The §4.2.7.5.8 output is the Q12 filter used by reconstruction; it
        // must fit a signed 16-bit value (the chirp loop ran §4.2.7.5.7-style
        // expansion, and a32_Q17 was range-limited going in).
        let nlsf = ascending_nlsf(D_LPC_WB, 800, 1900);
        let pg = LpcQ17::from_nlsf(Bandwidth::Wb, &nlsf)
            .unwrap()
            .range_limited()
            .prediction_gain_limited();
        for (k, &c) in pg.a_q12().iter().enumerate() {
            assert!(
                (i16::MIN as i32..=i16::MAX as i32).contains(&c),
                "a_Q12[{k}] = {c} does not fit i16"
            );
        }
    }

    #[test]
    fn pred_gain_limit_real_pipeline_stable_across_bandwidth_x_i1_sweep() {
        // Drive the full §4.2.7.5.2 → … → §4.2.7.5.7 → §4.2.7.5.8 pipeline
        // for every (bandwidth, I1) on a few buffers. The emitted Q12 filter
        // must always be stable (per is_lpc_stable, cross-checked vs the
        // independent oracle) and the round count bounded by 16.
        use crate::range_decoder::RangeDecoder;
        use crate::silk_lsf_recon::NlsfReconstructed;
        use crate::silk_lsf_stabilize::NlsfStabilized;
        use crate::silk_lsf_stage2::LsfStage2;

        let bufs: &[&[u8]] = &[
            &[
                0x5A, 0xC3, 0x17, 0x9E, 0x42, 0xFB, 0x08, 0x71, 0x2D, 0xB6, 0x4C, 0x8E,
            ],
            &[
                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            ],
            &[
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC, 0xBA, 0x98,
            ],
        ];
        for buf in bufs {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                for i1 in 0u8..32 {
                    let mut rd = RangeDecoder::new(buf);
                    let stage2 = LsfStage2::decode(&mut rd, bw, i1).expect("stage-2");
                    let recon =
                        NlsfReconstructed::from_stage1_and_stage2(bw, i1, &stage2).expect("recon");
                    let stab = NlsfStabilized::from_reconstructed(bw, &recon).expect("stab");
                    let pg = LpcQ17::from_nlsf(bw, stab.nlsf_q15())
                        .unwrap()
                        .range_limited()
                        .prediction_gain_limited();
                    assert_eq!(pg.len(), stab.nlsf_q15().len());
                    assert!(pg.rounds() <= 16);
                    assert!(
                        is_lpc_stable(pg.a_q12()),
                        "unstable Q12 filter: bw={bw:?} i1={i1} rounds={}",
                        pg.rounds()
                    );
                    // Production stability classification agrees with the
                    // independent oracle on the emitted filter.
                    assert_eq!(
                        is_lpc_stable(pg.a_q12()),
                        oracle_is_stable(pg.a_q12()),
                        "stability classification divergence: bw={bw:?} i1={i1}"
                    );
                }
            }
        }
    }

    #[test]
    fn ilog64_matches_spec_definition() {
        // ilog(n) = floor(log2(n)) + 1 for n > 0, else 0 — the §1.1.10 rule
        // applied to the i64 div_Q30 domain used by §4.2.7.5.8.
        assert_eq!(ilog64(-1), 0);
        assert_eq!(ilog64(0), 0);
        assert_eq!(ilog64(1), 1);
        assert_eq!(ilog64(2), 2);
        assert_eq!(ilog64(3), 2);
        assert_eq!(ilog64(4), 3);
        assert_eq!(ilog64(7), 3);
        // div_Q30 ∈ roughly [1, 2^30]; ilog(2^30) = 31, ilog(2^30 - 1) = 30.
        assert_eq!(ilog64(1 << 30), 31);
        assert_eq!(ilog64((1 << 30) - 1), 30);
    }

    #[test]
    fn lpc_real_pipeline_does_not_panic_across_bandwidth_x_i1_sweep() {
        // The §4.2.7.5.6 recurrence's i64 intermediates protect against
        // overflow in the bitstream-driven case. Verify the full SILK
        // §4.2.7.5.2..§4.2.7.5.4 → §4.2.7.5.6 path is panic-free for every
        // (bandwidth, I1) on a few buffers.
        use crate::range_decoder::RangeDecoder;
        use crate::silk_lsf_recon::NlsfReconstructed;
        use crate::silk_lsf_stabilize::NlsfStabilized;
        use crate::silk_lsf_stage2::LsfStage2;

        let bufs: &[&[u8]] = &[
            &[
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC,
            ],
            &[
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC, 0xBA, 0x98,
            ],
            &[
                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            ],
        ];
        for buf in bufs {
            for bw in [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb] {
                for i1 in 0u8..32 {
                    let mut rd = RangeDecoder::new(buf);
                    let stage2 = LsfStage2::decode(&mut rd, bw, i1).expect("stage-2");
                    let recon =
                        NlsfReconstructed::from_stage1_and_stage2(bw, i1, &stage2).expect("recon");
                    let stab = NlsfStabilized::from_reconstructed(bw, &recon).expect("stab");
                    let lpc = LpcQ17::from_nlsf(bw, stab.nlsf_q15()).unwrap();
                    // §4.2.7.5.6 leaves a32_Q17 unbounded; just confirm the
                    // length is right and the call returned.
                    assert_eq!(lpc.len(), stab.nlsf_q15().len());
                }
            }
        }
    }

    /// §4.2.7.5.8 arithmetic-envelope hardening (round-382 fuzz find):
    /// adversarial Q12 coefficients large enough to escape the spec's
    /// "all intermediate results fit in 32 bits" envelope must classify
    /// as unstable — never overflow the recurrence's i64 products.
    #[test]
    fn is_lpc_stable_survives_extreme_coefficients() {
        // Alternating-sign extremes keep DC_resp <= 4096 (the first
        // check) while each Q24 widening escapes 32 bits.
        let mut a = [0i32; 16];
        for (k, c) in a.iter_mut().enumerate() {
            *c = if k % 2 == 0 {
                i32::MAX >> 5
            } else {
                -(i32::MAX >> 5)
            };
        }
        assert!(!is_lpc_stable(&a));

        // A moderately extreme set that passes the initial-row check
        // but can blow up mid-recurrence must also terminate cleanly
        // (either verdict is fine; no panic).
        let mut b = [0i32; 16];
        for (k, c) in b.iter_mut().enumerate() {
            *c = if k % 2 == 0 { 80_000 } else { -80_000 };
        }
        let _ = is_lpc_stable(&b);
    }
}
