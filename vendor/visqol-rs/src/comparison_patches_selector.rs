use std::error::Error;

use crate::alignment::align_and_truncate_into;
use crate::convolution_2d::arena_mat;
use crate::gammatone_spectrogram_builder::GammatoneSpectrogramBuilder;
use crate::{
    analysis_window::AnalysisWindow,
    audio_signal::AudioSignal,
    audio_utils,
    neurogram_similiarity_index_measure::NeurogramSimiliarityIndexMeasure,
    patch_similarity_comparator::{BandValues, PatchSimilarityComparator, PatchSimilarityResult},
    spectrogram::Spectrogram,
    visqol_error::VisqolError,
};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use profluens_core::memory::Arena;
pub struct ComparisonPatchesSelector {
    sim_comparator: NeurogramSimiliarityIndexMeasure,
}

impl ComparisonPatchesSelector {
    pub fn new(sim_comparator: NeurogramSimiliarityIndexMeasure) -> Self { Self { sim_comparator } }

    /// This function composes the most suitable patches in a degraded signal given a reference signal.
    pub fn find_most_optimal_deg_patches(
        &self,
        ref_patches: &[ArrayView2<f64>],
        ref_patch_indices: &mut [usize],
        spectrogram_data: &Array2<f64>,
        frame_duration: f64,
        search_window_radius: i32,
        arena: &mut Arena,
    ) -> Result<Vec<PatchSimilarityResult>, VisqolError> {
        let num_frames_per_patch = ref_patches[0].ncols();
        let num_frames_in_deg_spectro = spectrogram_data.ncols();
        let patch_duration = frame_duration * num_frames_per_patch as f64;
        let search_window = search_window_radius * num_frames_per_patch as i32;
        let num_patches = Self::calc_max_num_patches(
            ref_patch_indices,
            num_frames_in_deg_spectro,
            num_frames_per_patch,
        );

        if num_patches == 0 {
            return Err(VisqolError::SignalsTooDifferent);
        } else if num_patches < ref_patch_indices.len() {
            log::warn!(
                "Warning: Dropping {} (of {}) reference patches 
            due to the degraded file being misaligned or too short. If too many 
            patches are dropped, the score will be less meaningful.",
                ref_patch_indices.len() - num_patches,
                ref_patch_indices.len()
            );
        }

        // The vector to store the similarity results
        let mut best_deg_patches = Vec::<PatchSimilarityResult>::new();
        best_deg_patches.resize(num_patches, PatchSimilarityResult::default());

        // The two DP tables are `patches × frames`, stored flat row-major: as `Vec<Vec<_>>` they
        // were one heap allocation *per reference patch* (~150 per comparison) for no benefit —
        // every row has the same length.
        let dp_stride = spectrogram_data.ncols();
        let mut cumulative_similarity_dp = vec![0.0f64; ref_patch_indices.len() * dp_stride];
        let mut backtrace = vec![0usize; ref_patch_indices.len() * dp_stride];

        // Attempt to get a good alignment with backtracking. The degraded candidate patch at each
        // slide offset is a (usually zero-copy) window into `spectrogram_data`, built on demand
        // inside the slide loop — the old pre-built `Vec<Array2>` of every column `to_owned`ed the
        // whole spectrogram ~ncols times (the largest remaining allocator, ~210K/song).
        for (index, ref_patch) in ref_patches.iter().enumerate() {
            self.find_most_optimal_deg_patch(
                spectrogram_data,
                ref_patch,
                &mut cumulative_similarity_dp,
                &mut backtrace,
                dp_stride,
                ref_patch_indices,
                index,
                search_window,
                arena,
            );
        }
        let mut max_similarity_score = f64::MIN;
        // The patch index for the last reference patch.
        let last_index = num_patches - 1;

        // The last_offset stores the offset at which the last reference patch got the
        // maximal similarity score over all the reference patches.

        let mut last_offset = 0;

        let lower_limit = 0.max(ref_patch_indices[last_index] as i32 - search_window) as usize;

        // The for loop is used to find the offset which maximizes the similarity
        // score across all the patches.
        // +1 for including last
        for slide_offset in lower_limit..ref_patch_indices[last_index] + search_window as usize + 1
        {
            if slide_offset >= num_frames_in_deg_spectro {
                // The frame offset for degraded start patch cannot be more than the
                // number of frames in the degraded spectrogram.
                break;
            }

            if cumulative_similarity_dp[last_index * dp_stride + slide_offset] > max_similarity_score
            {
                max_similarity_score =
                    cumulative_similarity_dp[last_index * dp_stride + slide_offset];
                last_offset = slide_offset;
            }
        }

        let mut patch_index: i32 = (num_patches - 1) as i32;
        while patch_index >= 0 {
            // Per-patch reset: the previous iteration's NSIM scratch is dead (its result is the
            // owned `PatchSimilarityResult` now in `best_deg_patches`), so reclaim the arena.
            arena.reset();
            // This sets the reference and degraded patch start and end times.
            // The reference patch is read, never written, so it is borrowed in place — the
            // `.clone()` this used to make was a full patch copy per reference patch.
            let ref_patch = &ref_patches[patch_index as usize];

            let deg_patch = Self::build_degraded_patch(
                spectrogram_data,
                last_offset,
                last_offset + ref_patch.ncols(),
                &*arena,
            );

            best_deg_patches[patch_index as usize] =
                self.sim_comparator.measure_patch_similarity(ref_patch, &deg_patch, &*arena);

            // This condition is true only if no matching patch was found for the given
            // reference patch. In this case, the matched patch is essentially set to
            // NULL (which is different from a silent patch).

            if last_offset == backtrace[patch_index as usize * dp_stride + last_offset] {
                best_deg_patches[patch_index as usize].deg_patch_start_time = 0.0;
                best_deg_patches[patch_index as usize].deg_patch_end_time = 0.0;
                best_deg_patches[patch_index as usize].similarity = 0.0;
                let num_rows = best_deg_patches[patch_index as usize].freq_band_means.len();
                best_deg_patches[patch_index as usize].freq_band_means = BandValues::zeros(num_rows);
            } else {
                best_deg_patches[patch_index as usize].deg_patch_start_time =
                    last_offset as f64 * frame_duration;
                best_deg_patches[patch_index as usize].deg_patch_end_time =
                    best_deg_patches[patch_index as usize].deg_patch_start_time + patch_duration;
            }

            best_deg_patches[patch_index as usize].ref_patch_start_time =
                ref_patch_indices[patch_index as usize] as f64 * frame_duration;
            best_deg_patches[patch_index as usize].ref_patch_end_time =
                best_deg_patches[patch_index as usize].ref_patch_start_time + patch_duration;
            last_offset = backtrace[patch_index as usize * dp_stride + last_offset];

            patch_index -= 1;
        }
        Ok(best_deg_patches)
    }

    /// This function finds the most suitable patch in a degraded signal given a reference patch.
    pub fn find_most_optimal_deg_patch(
        &self,
        spectrogram_data: &Array2<f64>,
        ref_patch: &ArrayView2<f64>,
        cumulative_similarity_dp: &mut [f64],
        backtrace: &mut [usize],
        dp_stride: usize,
        ref_patch_indices: &[usize],
        patch_index: usize,
        search_window: i32,
        arena: &mut Arena,
    ) {
        let ref_frame_index = ref_patch_indices[patch_index];
        let patch_width = ref_patch.ncols();

        let mut sim_similarity: f64;

        let mut slide_offset = ref_frame_index as i32 - search_window;
        while slide_offset <= ref_frame_index as i32 + search_window {
            if slide_offset < 0 {
                // The degraded patch index cannot be less than 0.
                slide_offset = 0;
                continue;
            }

            if slide_offset == spectrogram_data.ncols() as i32 {
                // The start of the degraded is past the end of the spectrogram, so
                // nothing left to compare.

                break;
            }
            // Per-candidate reset: the previous slide's NSIM scratch is dead (only the scalar score
            // was carried into the DP table), so reclaim the arena — this is the hot
            // O(patches × window) loop, so it dominates the allocation win.
            arena.reset();
            // Scalar-only: this loop uses nothing but the score, so skip the full measure's per-band
            // result `Vec`s. The degraded candidate patch `[start, end)` is a zero-copy view into the
            // spectrogram in the common case; the last few offsets spill past the end and need
            // zero-padding, which is carved from the arena. With the arena scratch the loop is
            // allocation-free.
            let start = slide_offset as usize;
            let end = start + patch_width;
            sim_similarity = if end <= spectrogram_data.ncols() {
                let deg_view = spectrogram_data.slice(s![.., start..end]);
                self.sim_comparator
                    .measure_similarity_score(&*ref_patch, &deg_view, &*arena)
            } else {
                let deg_patch = Self::build_degraded_patch(spectrogram_data, start, end, &*arena);
                self.sim_comparator
                    .measure_similarity_score(&*ref_patch, &deg_patch, &*arena)
            };
            let mut past_slide_offset = -1;
            let mut highest_sim = f64::MIN;

            if patch_index > 0 {
                // The lower_limit parameter tells us how far we should go
                // back to look for a possible match for the previous patch index
                // (patch_index - 1). The current value of lower_limit is used because the
                // search space for the previous patch index  is
                // (ref_patch_indices[patch_index - 1] - search_window,
                // ref_patch_indices[patch_index - 1] + search_window).
                let mut lower_limit: i32 =
                    ref_patch_indices[patch_index - 1] as i32 - search_window;
                lower_limit = lower_limit.max(0);
                // The back_offset parameter determines all the offsets that should be
                // considered while calculating the highest cumulative similarity score
                // achieved till patch_index - 1. Since two reference patches should
                // not map to the exact same degraded patch, the initial value of
                // back_offset is set to slide_offset - 1.
                let mut back_offset = slide_offset - 1;

                // The current for loop is used to find out the highest cumulative score
                // achieved till the previous ref_patch_index.
                while back_offset >= lower_limit {
                    let back = cumulative_similarity_dp
                        [(patch_index - 1) * dp_stride + back_offset as usize];
                    if back > highest_sim {
                        highest_sim = back;
                        past_slide_offset = back_offset;
                    }
                    back_offset -= 1;
                }

                sim_similarity += highest_sim;

                // If the current reference patch experienced a packet loss, then the
                // cumulative similarity score till the previous patch might be more and
                // in that case no matching patch for the current reference patch is found
                // in the degraded window.

                let previous =
                    cumulative_similarity_dp[(patch_index - 1) * dp_stride + slide_offset as usize];
                if previous > sim_similarity {
                    sim_similarity = previous;
                    past_slide_offset = slide_offset;
                }
            }
            cumulative_similarity_dp[patch_index * dp_stride + slide_offset as usize] =
                sim_similarity;
            backtrace[patch_index * dp_stride + slide_offset as usize] = past_slide_offset as usize;
            slide_offset += 1;
        }
    }

    /// Calculate the maximum number of patches that the degraded spectrogram can support.
    pub fn calc_max_num_patches(
        ref_patch_indices: &[usize],
        num_frames_in_deg_spectro: usize,
        num_frames_per_patch: usize,
    ) -> usize {
        let mut num_patches = ref_patch_indices.len();

        if num_patches != 0 {
            while (ref_patch_indices[num_patches - 1] - (num_frames_per_patch / 2))
                > num_frames_in_deg_spectro
            {
                num_patches -= 1;
            }
        }
        num_patches
    }

    /// The segment of `in_signal` from `start_time` to `end_time` (seconds), zero-padded where it
    /// falls outside the signal, written into the caller-owned `out`.
    ///
    /// `out` is cleared and refilled, so the fine-realignment loop reuses one buffer per signal
    /// across every patch instead of allocating a patch (plus a zero-padding `concatenate`, plus an
    /// owned `AudioSignal`) each time. Same samples, same zero padding, same order.
    pub fn slice_into(
        in_signal: &AudioSignal,
        start_time: f64,
        end_time: f64,
        out: &mut Vec<f64>,
    ) {
        let start_index = ((start_time * in_signal.sample_rate as f64) as usize).max(0);
        let end_index =
            ((end_time * in_signal.sample_rate as f64) as usize).min(in_signal.data_matrix.len());

        out.clear();
        // Pre-silence for a negative start time, then the real samples, then post-silence for an
        // end time past the signal.
        if start_time < 0.0 {
            out.resize((-start_time * in_signal.sample_rate as f64) as usize, 0.0);
        }
        out.extend(in_signal.data_matrix.slice(s![start_index..end_index]));

        let end_time_diff =
            (end_time * in_signal.sample_rate as f64 - in_signal.data_matrix.len() as f64) as usize;
        if end_time_diff > 0 {
            out.resize(out.len() + end_time_diff, 0.0);
        }
    }

    /// The degraded candidate patch spanning `[window_beginning, window_end)`, carved into the bump
    /// `arena` and zero-padded where it runs past the end of the spectrogram.
    ///
    /// This replaces an owned `Array2` built as `slice(..).to_owned()` plus (for the tail offsets)
    /// an `Array2::zeros` and a `concatenate` — three heap allocations per call, in the
    /// O(patches × window) slide loop. The values are the identical spectrogram elements in the
    /// identical positions with the identical zero padding, so every downstream score is unchanged.
    pub fn build_degraded_patch<'a>(
        spectrogram_data: &Array2<f64>,
        window_beginning: usize,
        window_end: usize,
        arena: &'a Arena,
    ) -> ArrayViewMut2<'a, f64> {
        let last_real_frame = window_end.min(spectrogram_data.ncols());
        let real_cols = last_real_frame.saturating_sub(window_beginning);
        let rows = spectrogram_data.nrows();

        let mut deg_patch = arena_mat(arena, rows, window_end - window_beginning);
        for row in 0..rows {
            let src = spectrogram_data.row(row);
            let mut dst = deg_patch.row_mut(row);
            for col in 0..real_cols {
                dst[col] = src[window_beginning + col];
            }
            // Past the end of the spectrogram: the zero padding the `concatenate` used to append.
            dst.slice_mut(s![real_cols..]).fill(0.0);
        }
        deg_patch
    }

    /// Performs alignment on a per-patch level.
    pub fn finely_align_and_recreate_patches<const NUM_BANDS: usize>(
        &self,
        sim_results: &mut [PatchSimilarityResult],
        ref_signal: &AudioSignal,
        deg_signal: &AudioSignal,
        spect_builder: &mut GammatoneSpectrogramBuilder<NUM_BANDS>,
        analysis_window: &AnalysisWindow,
        arena: &mut Arena,
    ) -> Result<Vec<PatchSimilarityResult>, Box<dyn Error>> {
        // Case: The patches are already matched.  Iterate over each pair.
        let mut realigned_results = Vec::<PatchSimilarityResult>::with_capacity(sim_results.len());
        realigned_results.resize(sim_results.len(), PatchSimilarityResult::default());
        // Transform scratch for the per-patch alignment, separate from `arena` because
        // `globally_align` resets it between its phases (see there). Patch-sized (a few MiB) and
        // reused across every patch of this comparison.
        let mut align_scratch = Arena::default();
        // Per-patch working storage, reused across every patch rather than reallocated: the two
        // sliced patch signals, the aligned pair, and the two spectrograms. Patches are all but
        // identical in size, so these `Vec`s grow on the first patch or two and never again —
        // together with the arena scratch that makes the loop allocation-free in steady state.
        let (mut ref_patch_audio, mut deg_patch_audio) = (Vec::new(), Vec::new());
        let (mut ref_audio_aligned, mut deg_audio_aligned) = (Vec::new(), Vec::new());
        let mut ref_spectrogram = Spectrogram::empty();
        let mut deg_spectrogram = Spectrogram::empty();
        for (i, result) in sim_results.iter_mut().enumerate() {
            // Per-patch reset: the previous iteration's NSIM scratch is dead (its result is owned
            // in `realigned_results`), so reclaim the arena before this patch's comparison.
            arena.reset();
            if result.deg_patch_start_time == result.deg_patch_end_time
                && result.deg_patch_start_time == 0.0
            {
                realigned_results[i] = result.clone();
                continue;
            }
            // 1. The sim results keep track of the start and end points of each matched
            // pair.  Extract the audio for this segment.
            Self::slice_into(
                ref_signal,
                result.ref_patch_start_time,
                result.ref_patch_end_time,
                &mut ref_patch_audio,
            );
            Self::slice_into(
                deg_signal,
                result.deg_patch_start_time,
                result.deg_patch_end_time,
                &mut deg_patch_audio,
            );

            // 2. For any pair, we want to shift the degraded signal to be maximally
            // aligned.
            let lag = align_and_truncate_into(
                &ref_patch_audio,
                &deg_patch_audio,
                ref_signal.sample_rate,
                &mut ref_audio_aligned,
                &mut deg_audio_aligned,
                &*arena,
                &mut align_scratch,
            )
            .ok_or(VisqolError::FailedToAlignSignals)?;

            let new_ref_duration = ref_audio_aligned.len() as f64 / ref_signal.sample_rate as f64;
            let new_deg_duration = deg_audio_aligned.len() as f64 / deg_signal.sample_rate as f64;
            // 3. Compute a new spectrogram for the degraded audio.

            spect_builder.build_into(
                &ref_audio_aligned,
                ref_signal.sample_rate,
                analysis_window,
                &mut ref_spectrogram,
            )?;
            spect_builder.build_into(
                &deg_audio_aligned,
                deg_signal.sample_rate,
                analysis_window,
                &mut deg_spectrogram,
            )?;
            // 4. Recreate an aligned degraded patch from the new spectrogram.

            audio_utils::prepare_spectrograms_for_comparison(
                &mut ref_spectrogram,
                &mut deg_spectrogram,
            );
            // 5. Update the similarity result with the new patch.

            let mut new_sim_result = self.sim_comparator.measure_patch_similarity(
                &ref_spectrogram.data,
                &deg_spectrogram.data,
                &*arena,
            );
            // Compare to the old result and take the max.
            if new_sim_result.similarity < result.similarity {
                realigned_results[i] = result.clone();
            } else {
                if lag > 0.0 {
                    new_sim_result.ref_patch_start_time = result.ref_patch_start_time + lag;
                    new_sim_result.deg_patch_start_time = result.deg_patch_start_time;
                } else {
                    new_sim_result.ref_patch_start_time = result.ref_patch_start_time;
                    new_sim_result.deg_patch_start_time = result.deg_patch_start_time - lag;
                }
                new_sim_result.ref_patch_end_time =
                    new_sim_result.ref_patch_start_time + new_ref_duration;
                new_sim_result.deg_patch_end_time =
                    new_sim_result.deg_patch_start_time + new_deg_duration;
                realigned_results[i] = new_sim_result;
            }
        }
        Ok(realigned_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio_signal::AudioSignal, image_patch_creator::ImagePatchCreator,
        neurogram_similiarity_index_measure::NeurogramSimiliarityIndexMeasure,
        patch_creator::PatchCreator,
    };
    use ndarray::{arr2, Array1, Array2};

    #[test]
    fn num_patches_is_computed_correctly() {
        let patch_indices = vec![0, 15, 30, 45, 60];

        let slide_offset = 45;
        let accepted_num_patches =
            ComparisonPatchesSelector::calc_max_num_patches(&patch_indices, slide_offset, 30);

        assert_eq!(patch_indices.len(), accepted_num_patches);

        let slide_offset = 44;
        let accepted_num_patches =
            ComparisonPatchesSelector::calc_max_num_patches(&patch_indices, slide_offset, 30);

        assert_eq!(patch_indices.len() - 1, accepted_num_patches);
    }

    #[test]
    fn time_slicing_signal_is_sample_accurate() {
        let fs = 16000;
        let num_seconds = 3;

        let mut silence_matrix = Array1::zeros(fs * num_seconds);

        silence_matrix[16000] = 1.0;
        let three_seconds_silence = AudioSignal::new(silence_matrix.as_slice().unwrap(), fs as u32);

        let mut sliced = Vec::new();
        ComparisonPatchesSelector::slice_into(&three_seconds_silence, 0.5, 2.5, &mut sliced);
        let sliced_signal = AudioSignal::new(&sliced, fs as u32);

        assert_eq!(sliced_signal.get_duration(), 2.0);
        assert_eq!(sliced_signal[7999], 0.0);
        assert_eq!(sliced_signal[8000], 1.0);
        assert_eq!(sliced_signal[8001], 0.0);
    }

    #[test]
    fn optimal_patches_start_times_are_correct() {
        let ref_matrix = arr2(&[
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0],
        ]);

        // Create reference patches from given patch indices
        let patch_size = 1;
        let mut patch_indices: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        // Defining the degraded audio matrix
        let rows_concatenated: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 2.0,
            0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(rows_concatenated.len(), 3 * 30);
        let deg_matrix = Array2::from_shape_vec((3, 30), rows_concatenated).unwrap();

        let frame_duration = 1.0;
        let search_window = 8;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
                &mut profluens_core::memory::Arena::default(),
            )
            .unwrap();

        assert_eq!(res[3].deg_patch_start_time, 0.0);
        assert_eq!(res[4].deg_patch_start_time, 7.0);
        assert_eq!(res[5].deg_patch_start_time, 8.0);
    }

    #[test]
    fn matches_are_out_of_order() {
        let ref_matrix = arr2(&[[1.0, 100.0, 3.0, 4.0], [0.0; 4], [1.0, 100.0, 3.0, 4.0]]);

        let patch_size = 1;
        let mut patch_indices = vec![0, 1, 2, 3];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let deg_matrix = arr2(&[[100.0, 1.0, 3.0, 4.0], [0.0; 4], [100.0, 1.0, 3.0, 4.0]]);

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
                &mut profluens_core::memory::Arena::default(),
            )
            .unwrap();

        assert_eq!(res[0].deg_patch_start_time, 1.0);
        assert_eq!(res[1].deg_patch_start_time, 0.0);
        assert_eq!(res[2].deg_patch_start_time, 2.0);
        assert_eq!(res[3].deg_patch_start_time, 3.0);
    }

    #[test]
    fn results_are_different() {
        let ref_matrix = arr2(&[[1.0], [1.0], [0.0]]);

        let patch_size = 1;
        let mut patch_indices = vec![0];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let concatenated_deg_mat = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let deg_matrix = Array2::from_shape_vec((3, 17), concatenated_deg_mat).unwrap();

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
                &mut profluens_core::memory::Arena::default(),
            )
            .unwrap();
        assert_eq!(res[0].deg_patch_start_time, 6.0);
    }

    #[test]
    fn start_times_in_longer_file_are_correct() {
        let ref_vec = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let ref_matrix = Array2::from_shape_vec((3, 31), ref_vec).unwrap();

        let patch_size = 2;

        let mut patch_indices = vec![4, 6, 10, 12, 14, 22];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let deg_vec = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let deg_matrix = Array2::from_shape_vec((3, 31), deg_vec).unwrap();

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
                &mut profluens_core::memory::Arena::default(),
            )
            .unwrap();

        assert_eq!(res[0].deg_patch_start_time, 6.0);
        assert_eq!(res[1].deg_patch_start_time, 8.0);
        assert_eq!(res[2].deg_patch_start_time, 12.0);
        assert_eq!(res[3].deg_patch_start_time, 14.0);
        assert_eq!(res[4].deg_patch_start_time, 16.0);
        assert_eq!(res[5].deg_patch_start_time, 22.0);
    }
}
