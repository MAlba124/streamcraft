//! CELT §4.3.4 per-band shape decode orchestration
//! (RFC 6716 §4.3.4, pp. 116–120).
//!
//! §4.3.4 decodes one CELT band's normalized "shape" by composing three
//! steps the RFC presents in sequence, given the band's pulse count `K`
//! (from the §4.3.4.1 bits-to-pulses conversion), its dimension count
//! `N`, the frame-global "spread" symbol, the band's time-frequency
//! adjustment, and the number of interleaved MDCT blocks the band
//! represents:
//!
//! 1. **§4.3.4.2 PVQ decode** ([`crate::celt_pvq_decode::decode_pvq_shape_into`]):
//!    read `i = ec_dec_uint(V(N, K))`, recover the integer pulse vector,
//!    normalize to unit L2 norm.
//! 2. **§4.3.4.3 spreading** ([`crate::celt_spreading::apply_spreading`]):
//!    rotate the unit-norm vector to avoid tonal artifacts, per-block on
//!    the transient (multi-block) path.
//! 3. **§4.3.4.5 time-frequency change**
//!    ([`crate::celt_tf_hadamard::apply_tf_hadamard`]): apply the
//!    band's TF Hadamard transform (increase time or frequency
//!    resolution), reshaping the interleaved short-MDCT blocks.
//!
//! The result is the band's normalized time-frequency shape vector,
//! ready for the §4.3.6 denormalisation (which multiplies by the square
//! root of the decoded band energy).
//!
//! ## What this module does NOT do
//!
//! This is the **non-split, single-call** §4.3.4 path. The §4.3.4.4
//! recursive *split decoding* (codebooks larger than 32 bits split into
//! two N/2 sub-vectors with an entropy-coded relative gain) is **not**
//! composed here: its gain-quantization precision is "derived from the
//! current allocation" and that derivation lives only in the reference
//! `rate.c` / `bands.c`, absent from the RFC narrative. A band whose
//! `V(N, K)` exceeds the 32-bit codebook limit therefore cannot be
//! decoded through this path yet; callers must gate on that.
//!
//! Likewise, the per-band `K` itself comes from the §4.3.3 allocation
//! orchestration (`interp_bits2pulses`), which is not specified in the
//! RFC narrative — so this orchestrator is exercised here with
//! caller-supplied `K` values and unit-tested for its composition
//! semantics, not yet wired into an end-to-end CELT decode against a
//! real bitstream.
//!
//! ## Block geometry
//!
//! `nb_blocks` is the number of interleaved short-MDCT blocks `B` the
//! band spans (a power of two; `1` for a non-transient band). The
//! coefficient vector is laid out bin-major / block-minor: coefficient
//! `m` of block `b` at flat index `m * B + b`, matching
//! [`crate::celt_tf_hadamard`]. The spreading step
//! ([`crate::celt_spreading::apply_spreading`]) operates per time block
//! under the same `nb_blocks` count.
//!
//! ## Provenance
//!
//! Composition order + step definitions: RFC 6716 §4.3.4.2–§4.3.4.5
//! (pp. 116–120), reproduced from `docs/audio/opus/rfc6716-opus.txt`.
//! No external library source was consulted.

use crate::celt_pvq_decode::{decode_pvq_shape_into, PvqShapeError};
use crate::celt_spreading::{apply_spreading, SpreadingError};
use crate::celt_tf_adjust::{TfAdjustment, TfDirection};
use crate::celt_tf_hadamard::{apply_tf_hadamard, TfHadamardError};

/// Maximum PVQ codebook size, in bits, that the non-split §4.3.4.2 path
/// can decode (RFC 6716 §4.3.4.4: "the maximum size allowed for
/// codebooks is 32 bits"). A band whose `V(N, K)` needs more than this
/// must use the §4.3.4.4 split path, which is not composed here.
pub const PVQ_MAX_CODEBOOK_BITS: u32 = 32;

/// Errors returnable by the §4.3.4 band-shape orchestrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandShapeError {
    /// The §4.3.4.2 PVQ decode failed. Wraps [`PvqShapeError`].
    Pvq(PvqShapeError),
    /// The §4.3.4.3 spreading step failed. Wraps [`SpreadingError`].
    Spreading(SpreadingError),
    /// The §4.3.4.5 TF Hadamard transform failed. Wraps
    /// [`TfHadamardError`].
    TfHadamard(TfHadamardError),
    /// The caller-supplied output buffer is shorter than `N`.
    OutputBufferTooSmall {
        /// Required length `N`.
        required: usize,
        /// Provided length.
        provided: usize,
    },
}

impl core::fmt::Display for BandShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BandShapeError::Pvq(e) => write!(f, "oxideav-opus: §4.3.4 band shape PVQ: {e}"),
            BandShapeError::Spreading(e) => {
                write!(f, "oxideav-opus: §4.3.4 band shape spreading: {e}")
            }
            BandShapeError::TfHadamard(e) => {
                write!(f, "oxideav-opus: §4.3.4 band shape TF transform: {e}")
            }
            BandShapeError::OutputBufferTooSmall { required, provided } => write!(
                f,
                "oxideav-opus: §4.3.4 band shape output buffer too small: \
                 required={required}, provided={provided}"
            ),
        }
    }
}

impl std::error::Error for BandShapeError {}

impl From<PvqShapeError> for BandShapeError {
    fn from(e: PvqShapeError) -> Self {
        BandShapeError::Pvq(e)
    }
}
impl From<SpreadingError> for BandShapeError {
    fn from(e: SpreadingError) -> Self {
        BandShapeError::Spreading(e)
    }
}
impl From<TfHadamardError> for BandShapeError {
    fn from(e: TfHadamardError) -> Self {
        BandShapeError::TfHadamard(e)
    }
}

/// Decode one §4.3.4 band shape into `out[0..N]`.
///
/// Composes §4.3.4.2 PVQ decode → §4.3.4.3 spreading → §4.3.4.5 TF
/// Hadamard transform. See the module documentation for the block
/// geometry and the deliberately-omitted §4.3.4.4 split path.
///
/// Inputs:
///
/// * `rd` — the frame's range decoder (the PVQ index is read from it).
/// * `n` — the band's dimension count `N` (one channel's MDCT bins for
///   the band, times `nb_blocks` when transient).
/// * `k` — the band's pulse count `K` from §4.3.4.1.
/// * `spread` — the frame-global Table 59 spread symbol (`0..=3`).
/// * `tf_adjust` — the band's TF adjustment (from
///   [`crate::celt_tf_adjust`]); classified internally into a
///   [`TfDirection`].
/// * `nb_blocks` — the number of interleaved short-MDCT blocks (a power
///   of two; `1` for non-transient).
/// * `out` — output buffer of length `>= N`; filled with the band's
///   normalized time-frequency shape.
///
/// Returns the number of coordinates written (`N`).
///
/// For `K = 0` the PVQ step yields an all-zero vector; spreading and
/// the TF transform leave it zero, so the band emits no shape — the
/// §4.3.4 "no pulses" case.
///
/// # Errors
///
/// Returns [`BandShapeError`] wrapping the failing step's error, or
/// [`BandShapeError::OutputBufferTooSmall`] when `out.len() < N`.
#[allow(clippy::too_many_arguments)]
pub fn decode_band_shape_into(
    rd: &mut crate::RangeDecoder<'_>,
    n: u32,
    k: u32,
    spread: u8,
    tf_adjust: TfAdjustment,
    nb_blocks: usize,
    out: &mut [f64],
) -> Result<usize, BandShapeError> {
    let n_usize = n as usize;
    if out.len() < n_usize {
        return Err(BandShapeError::OutputBufferTooSmall {
            required: n_usize,
            provided: out.len(),
        });
    }
    let band = &mut out[..n_usize];

    // §4.3.4.2 PVQ decode → unit-norm shape.
    decode_pvq_shape_into(rd, n, k, band)?;

    // §4.3.4.3 spreading (per time block on the transient path). The
    // spreading helper tolerates K = 0 and short blocks as no-ops.
    apply_spreading(band, k, spread, nb_blocks)?;

    // §4.3.4.5 time-frequency change. Unchanged (adj == 0) is identity.
    let direction = TfDirection::from_adjustment(tf_adjust);
    apply_tf_hadamard(band, nb_blocks, direction)?;

    Ok(n_usize)
}

/// Allocating convenience wrapper around [`decode_band_shape_into`].
#[allow(clippy::too_many_arguments)]
pub fn decode_band_shape(
    rd: &mut crate::RangeDecoder<'_>,
    n: u32,
    k: u32,
    spread: u8,
    tf_adjust: TfAdjustment,
    nb_blocks: usize,
) -> Result<Vec<f64>, BandShapeError> {
    let mut out = vec![0.0f64; n as usize];
    decode_band_shape_into(rd, n, k, spread, tf_adjust, nb_blocks, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RangeDecoder;

    fn l2(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn k_zero_yields_all_zero_shape() {
        // No pulses ⇒ no range read for the index (V(N,0)=1 ⇒ dec_uint
        // returns 0 without consuming), and all downstream steps are
        // no-ops on the zero vector.
        let buf = [0xAB, 0xCD, 0xEF, 0x12];
        let mut rd = RangeDecoder::new(&buf);
        let shape = decode_band_shape(&mut rd, 6, 0, 2, 0, 1).unwrap();
        assert_eq!(shape, vec![0.0; 6]);
    }

    #[test]
    fn nonzero_k_no_spread_no_tf_is_unit_norm() {
        // spread = 0 (no rotation), tf_adjust = 0 (identity): the result
        // is exactly the unit-norm PVQ shape, so its L2 norm is 1.
        let buf = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let mut rd = RangeDecoder::new(&buf);
        let shape = decode_band_shape(&mut rd, 8, 3, 0, 0, 1).unwrap();
        assert!((l2(&shape) - 1.0).abs() < 1e-9, "norm = {}", l2(&shape));
    }

    #[test]
    fn spreading_preserves_unit_norm() {
        // The orthogonal spreading rotation preserves the L2 norm.
        for spread in 1u8..=3 {
            let buf = [0x42, 0x18, 0x7F, 0x03, 0xC1, 0x5E];
            let mut rd = RangeDecoder::new(&buf);
            let shape = decode_band_shape(&mut rd, 8, 4, spread, 0, 1).unwrap();
            assert!(
                (l2(&shape) - 1.0).abs() < 1e-9,
                "spread {spread}: norm = {}",
                l2(&shape)
            );
        }
    }

    #[test]
    fn tf_transform_preserves_unit_norm() {
        // Transient band with a TF adjustment: the orthonormal Hadamard
        // transform also preserves the unit norm. nb_blocks = 2, N must
        // be a multiple of nb_blocks.
        let buf = [0x71, 0x2C, 0x90, 0xEE, 0x44, 0x08];
        let mut rd = RangeDecoder::new(&buf);
        // tf_adjust = -1 ⇒ IncreaseTime(1).
        let shape = decode_band_shape(&mut rd, 8, 3, 0, -1, 2).unwrap();
        assert!((l2(&shape) - 1.0).abs() < 1e-9, "norm = {}", l2(&shape));
    }

    #[test]
    fn output_buffer_too_small_errors() {
        let buf = [0x00, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&buf);
        let mut out = [0.0f64; 3];
        let r = decode_band_shape_into(&mut rd, 8, 2, 1, 0, 1, &mut out);
        assert_eq!(
            r,
            Err(BandShapeError::OutputBufferTooSmall {
                required: 8,
                provided: 3
            })
        );
    }

    #[test]
    fn bad_block_geometry_surfaces_error() {
        // N = 5 is not a multiple of nb_blocks = 2. Spreading runs
        // before the TF step and rejects the non-divisible geometry
        // first (both steps share the same divisibility precondition);
        // the band-shape orchestrator surfaces it as a Spreading error.
        let buf = [0x33, 0x77, 0xAA, 0xDD];
        let mut rd = RangeDecoder::new(&buf);
        let r = decode_band_shape(&mut rd, 5, 1, 0, -1, 2);
        assert!(matches!(r, Err(BandShapeError::Spreading(_))), "{r:?}");
    }

    #[test]
    fn tf_levels_exceed_blocks_surfaces_tf_error() {
        // nb_blocks = 1 but a non-zero TF adjustment requests a 2-point
        // transform: spreading is a no-op for a single block, so the
        // error originates in the TF step.
        let buf = [0x33, 0x77, 0xAA, 0xDD];
        let mut rd = RangeDecoder::new(&buf);
        let r = decode_band_shape(&mut rd, 6, 1, 0, -1, 1);
        assert!(matches!(r, Err(BandShapeError::TfHadamard(_))), "{r:?}");
    }

    #[test]
    fn into_and_allocating_agree() {
        let buf = [0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB];
        let mut rd_a = RangeDecoder::new(&buf);
        let mut rd_b = RangeDecoder::new(&buf);
        let alloc = decode_band_shape(&mut rd_a, 8, 5, 2, 0, 1).unwrap();
        let mut into = vec![0.0f64; 8];
        let written = decode_band_shape_into(&mut rd_b, 8, 5, 2, 0, 1, &mut into).unwrap();
        assert_eq!(written, 8);
        assert_eq!(alloc, into);
    }

    #[test]
    fn deterministic_across_calls() {
        // Same bitstream, same params ⇒ identical decode.
        let buf = [0x5A, 0xA5, 0x33, 0xCC, 0x0F, 0xF0];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);
        let a = decode_band_shape(&mut rd1, 6, 4, 1, 0, 1).unwrap();
        let b = decode_band_shape(&mut rd2, 6, 4, 1, 0, 1).unwrap();
        assert_eq!(a, b);
    }
}
