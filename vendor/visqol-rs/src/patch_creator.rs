use ndarray::{Array2, ArrayView2};

use crate::{
    analysis_window::AnalysisWindow, audio_signal::AudioSignal, visqol_error::VisqolError,
};

/// This trait enables the creation of patches from a spectrogram.
/// The term `patch` here refers to a segment of 2-dimensional data. How the data is segmented is determined by the individual implementation of this trait.
pub trait PatchCreator {
    /// Given a spectrogram, this function returns 0-indexed indices of each patch.
    fn create_ref_patch_indices(
        &self,
        spectrogram: &Array2<f64>,
        ref_signal: &AudioSignal,
        window: &AnalysisWindow,
    ) -> Result<Vec<usize>, VisqolError>;

    /// Given a spectrogram and the corresponding indices, this function performs the segmentation and returns each patch as a zero-copy view into the spectrogram.
    ///
    /// Views, not owned copies: a patch is a column range of the reference spectrogram, and every
    /// consumer (the NSIM measure and the 2-D convolution) is generic over storage — so copying each
    /// one out cost a heap `Array2` per reference patch for nothing.
    fn create_patches_from_indices<'a>(
        &self,
        spectrogram: &'a Array2<f64>,
        patch_indices: &[usize],
    ) -> Vec<ArrayView2<'a, f64>>;
}
