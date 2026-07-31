use ndarray::{ArrayBase, Data, Ix2};
use profluens_core::memory::Arena;
use serde::{Serialize, Serializer};

/// Per-frequency-band values for one patch, stored **inline**.
///
/// ViSQOL runs 21 (speech) or 32 (audio) bands, so a heap `Vec` per field is pure overhead — and
/// these are the one thing `measure_patch_similarity` produces that must outlive the per-patch arena
/// reset, which made three `Vec`s per patch (plus `std_axis`'s two `Array1`s) the largest remaining
/// allocator in a comparison: ~740 of 873 for a 30 s scoring. Inline storage removes all of them and
/// makes a result `Copy`. Derefs to `&[f64]`, so callers index and iterate it exactly as before.
#[derive(Debug, Clone, Copy)]
pub struct BandValues {
    values: [f64; Self::MAX_BANDS],
    len: usize,
}

impl BandValues {
    /// The largest band count ViSQOL configures — [`crate::constants::NUM_BANDS_AUDIO`].
    pub const MAX_BANDS: usize = 32;

    /// `len` bands, each computed by `f`. Panics above [`MAX_BANDS`](Self::MAX_BANDS) — a band count
    /// that large would be a new ViSQOL mode, not a runtime condition.
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> f64) -> Self {
        assert!(len <= Self::MAX_BANDS, "{} bands exceeds MAX_BANDS", len);
        let mut values = [0.0f64; Self::MAX_BANDS];
        for (band, value) in values[..len].iter_mut().enumerate() {
            *value = f(band);
        }
        Self { values, len }
    }

    /// `len` bands of zero.
    pub fn zeros(len: usize) -> Self {
        Self::from_fn(len, |_| 0.0)
    }
}

impl std::ops::Deref for BandValues {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.values[..self.len]
    }
}

impl Default for BandValues {
    fn default() -> Self {
        Self { values: [0.0; Self::MAX_BANDS], len: 0 }
    }
}

impl Serialize for BandValues {
    /// Serialises as the sequence of live bands — indistinguishable from the `Vec<f64>` this
    /// replaced.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
/// Bundles similarity information of a single patch.
/// The term `Patch` here refers to a single of spectrogram data produced by a PatchCreator)
pub struct PatchSimilarityResult {
    /// Means of the individual frequency bands
    pub freq_band_means: BandValues,
    /// Standard deviation of the individual frequency bands
    pub freq_band_stddevs: BandValues,
    /// Energy of the degraded file per frequency band
    pub freq_band_deg_energy: BandValues,
    /// Calculated patch similarity score
    pub similarity: f64,
    /// Reference start of patch in seconds
    pub ref_patch_start_time: f64,
    /// Reference end of patch in seconds
    pub ref_patch_end_time: f64,
    /// Degraded start of patch in seconds
    pub deg_patch_start_time: f64,
    /// Degraded end of patch in seconds
    pub deg_patch_end_time: f64,
}

impl PatchSimilarityResult {
    /// Creates a new similarity result, stores mean, std and energy of degraded signal and sets the time information to 0
    pub fn new(
        freq_band_means: BandValues,
        freq_band_stddevs: BandValues,
        freq_band_deg_energy: BandValues,
        similarity: f64,
    ) -> Self {
        Self {
            freq_band_means,
            freq_band_stddevs,
            freq_band_deg_energy,
            similarity,
            ref_patch_start_time: 0.0,
            ref_patch_end_time: 0.0,
            deg_patch_start_time: 0.0,
            deg_patch_end_time: 0.0,
        }
    }
}

impl Default for PatchSimilarityResult {
    fn default() -> Self {
        Self {
            freq_band_means: BandValues::default(),
            freq_band_stddevs: BandValues::default(),
            freq_band_deg_energy: BandValues::default(),
            similarity: 0.0,
            ref_patch_start_time: 0.0,
            ref_patch_end_time: 0.0,
            deg_patch_start_time: 0.0,
            deg_patch_end_time: 0.0,
        }
    }
}

/// If implemented, this trait allows for computing a similarity score of 2 patches.
///
/// Both patches are borrowed **shared and generically over their storage**: the measure only reads
/// them, and the callers hand it an owned `Array2`, a zero-copy `ArrayView2` window into a
/// spectrogram, or an arena-backed `ArrayViewMut2` — none of which should have to be copied into an
/// owned patch first. (Generic methods cost object safety; nothing uses this trait as a `dyn`.)
pub trait PatchSimilarityComparator {
    fn measure_patch_similarity<Sr, Sd>(
        &self,
        ref_patch: &ArrayBase<Sr, Ix2>,
        deg_patch: &ArrayBase<Sd, Ix2>,
        arena: &Arena,
    ) -> PatchSimilarityResult
    where
        Sr: Data<Elem = f64>,
        Sd: Data<Elem = f64>;
}
