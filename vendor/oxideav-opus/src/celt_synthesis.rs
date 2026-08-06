//! CELT §4.3.6 → §4.3.7 → §4.3.7.2 synthesis backend composition
//! (RFC 6716 §4.3.6 / §4.3.7 / §4.3.7.2, p. 121–122).
//!
//! This module is the CELT analogue of [`crate::silk_synthesis`]: it
//! composes the individually-tested per-stage CELT decode primitives into
//! one call that turns the *frequency-domain* output of the CELT
//! entropy-decode front half — per-band unit-L2-norm shapes plus per-band
//! `log2` energies — into time-domain PCM samples, threading the
//! cross-frame state (the §4.3.7 overlap-add history and the §4.3.7.2
//! de-emphasis memory) the way the spec's continuous reconstruction
//! requires.
//!
//! ## The backend pipeline (§4.3.6 → §4.3.7.2)
//!
//! By the time control reaches this module the upstream CELT decode has
//! produced, per coded band, a unit-L2-norm MDCT **shape**
//! (§4.3.4 PVQ + §4.3.5 anti-collapse/folding) and a reconstructed
//! base-2-log band **energy** (§4.3.2 coarse + fine). This module runs
//! the three deterministic backend stages that the RFC states with no
//! free parameters:
//!
//! 1. **§4.3.6 denormalisation** — each band's shape is scaled by
//!    `sqrt(2**log2_energy)`, laying the coded bands end-to-end into a
//!    single per-channel frequency buffer of length `N`
//!    ([`crate::celt_denormalise::denormalise_bands`]).
//! 2. **§4.3.7 inverse MDCT** — the `N`-bin frequency buffer is
//!    transformed to a `2N`-sample time-domain block, scaled by `1/2`
//!    ([`crate::celt_imdct::imdct_into`]).
//! 3. **§4.3.7 weighted overlap-add** — the `2N` block is windowed with
//!    the §4.3.7 low-overlap window and overlap-added with the previous
//!    frame's windowed trailing half, emitting `N` aliasing-free samples
//!    and carrying the new trailing half forward
//!    ([`crate::celt_overlap_add::WeightedOverlapAdd`]).
//! 4. **§4.3.7.2 de-emphasis** — the `N` samples are run through the
//!    one-pole de-emphasis filter, whose memory is continuous across the
//!    whole stream ([`crate::celt_deemphasis::DeemphasisFilter`]).
//!
//! The §4.3.7.1 pitch post-filter that nominally sits between stages 3
//! and 4 is **not** composed here: this backend models the post-filter as
//! disabled (the §4.3.7.1 post-filter gains decode to zero / the bands
//! covered by this composition), which is the identity on the time-domain
//! samples. A frame that signals a non-trivial post-filter is out of
//! scope for this composition and the caller must not route it here; the
//! post-filter primitive ([`crate::celt_post_filter`]) is wired at its own
//! site. Stages 1–4 are the full no-post-filter CELT synthesis backend.
//!
//! ## State threading (continuous reconstruction)
//!
//! [`CeltSynthState`] holds one [`WeightedOverlapAdd`] and one
//! [`DeemphasisFilter`] per channel. Both carry across frame boundaries:
//! the overlap-add history is the windowed trailing half of the previous
//! frame (§4.3.7 — the very mechanism that cancels the time-domain
//! aliasing), and the de-emphasis memory is the single one-pole state
//! `y(n-1)` that the §4.3.7.2 recurrence runs continuously over the whole
//! decoded stream. [`CeltSynthState::reset`] zeroes both, the state a
//! §4.5.2 CELT reset / stream start begins from.
//!
//! Because a CELT-only frame always runs at the full 48 kHz internal rate
//! (unlike SILK's 8/12/16 kHz internal rates), the time-domain samples
//! this module emits are already at the Opus output rate — no resample
//! stage is needed (contrast `silk_synthesis` + the §4.2.9 resampler).
//!
//! ## MDCT size vs coded-band bins
//!
//! The §4.3.7 inverse MDCT operates on the **full** per-channel MDCT size
//! `N` = `frame_samples` (120 / 240 / 480 / 960 for the 2.5 / 5 / 10 /
//! 20 ms frames at 48 kHz), producing `2N` time-domain samples. The
//! §4.3.6 denormalised band coefficients fill only the lower
//! [`celt_total_bins_per_channel`] bins of that `N`-bin buffer — the band
//! layout (Table 55) tops out at the 20 kHz edge, so a 20 ms frame's 21
//! bands sum to 800 bins, leaving bins `800..960` of the MDCT input zero
//! (the inaudible region above 20 kHz the encoder never codes). This
//! module zero-initialises the full `N`-bin frequency buffer and lets the
//! denormaliser write the coded prefix, so the high zero-pad is exact.
//!
//! ## Provenance
//!
//! The backend stage ordering (denormalise → inverse MDCT → windowed
//! overlap-add → de-emphasis) and the continuity of the overlap and
//! de-emphasis state: RFC 6716 §4.3.6 / §4.3.7 / §4.3.7.2 (p. 121–122),
//! reproduced from `docs/audio/opus/rfc6716-opus.txt`. Each stage's
//! numeric definition lives in (and is unit-tested by) its own module;
//! this module only sequences them and owns the cross-frame state. No
//! external library source was consulted — the composition is exactly the
//! pipeline the standards-track text lays out.

use crate::celt_band_layout::{celt_first_coded_band, celt_total_bins_per_channel, CeltFrameSize};
use crate::celt_deemphasis::DeemphasisFilter;
use crate::celt_denormalise::{denormalise_bands, DenormaliseError};
use crate::celt_imdct::{imdct_into, ImdctError};
use crate::celt_mdct_window::CELT_OVERLAP_48K;
use crate::celt_overlap_add::{OverlapAddError, WeightedOverlapAdd};

/// Errors returnable by the §4.3.6–§4.3.7.2 CELT synthesis backend.
#[derive(Debug, Clone, PartialEq)]
pub enum CeltSynthError {
    /// The number of supplied band shapes / energies did not match the
    /// number of coded bands for the configured `(frame_size, is_hybrid)`.
    BandCountMismatch {
        /// The coded-band count the layout expects.
        expected: usize,
        /// The number of shapes (and energies) the caller supplied.
        got_shapes: usize,
        /// The number of energies the caller supplied.
        got_energies: usize,
    },
    /// The number of channels in the call did not match the number this
    /// [`CeltSynthState`] was built for.
    ChannelCountMismatch {
        /// The channel count the state carries.
        expected: usize,
        /// The channel count the call supplied.
        got: usize,
    },
    /// A §4.3.6 denormalisation error from the underlying primitive.
    Denormalise(DenormaliseError),
    /// A §4.3.7 inverse-MDCT error from the underlying primitive.
    Imdct(ImdctError),
    /// A §4.3.7 overlap-add error from the underlying primitive.
    OverlapAdd(OverlapAddError),
}

impl core::fmt::Display for CeltSynthError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CeltSynthError::BandCountMismatch {
                expected,
                got_shapes,
                got_energies,
            } => write!(
                f,
                "oxideav-opus: CELT synthesis expected {expected} coded bands, \
                 got {got_shapes} shapes / {got_energies} energies"
            ),
            CeltSynthError::ChannelCountMismatch { expected, got } => write!(
                f,
                "oxideav-opus: CELT synthesis state carries {expected} channel(s), \
                 call supplied {got}"
            ),
            CeltSynthError::Denormalise(e) => write!(f, "oxideav-opus: CELT §4.3.6 {e}"),
            CeltSynthError::Imdct(e) => write!(f, "oxideav-opus: CELT §4.3.7 {e}"),
            CeltSynthError::OverlapAdd(e) => write!(f, "oxideav-opus: CELT §4.3.7 {e}"),
        }
    }
}

impl std::error::Error for CeltSynthError {}

/// One CELT channel's persistent backend state: the §4.3.7 weighted
/// overlap-add (carrying the windowed trailing half of the previous
/// frame) and the §4.3.7.2 de-emphasis filter (carrying the one-pole
/// memory). Both are continuous across frames within a stream.
#[derive(Debug, Clone, PartialEq)]
struct CeltChannelState {
    /// §4.3.7 weighted overlap-add (its `history` is the cross-frame
    /// overlap memory).
    ola: WeightedOverlapAdd,
    /// §4.3.7.2 de-emphasis (its `mem` is the cross-frame one-pole state).
    deemph: DeemphasisFilter,
}

/// The §4.3.6–§4.3.7.2 CELT synthesis backend state for a whole stream
/// (one or two channels), threaded across the frames of an Opus stream.
///
/// Construct once at stream start with the frame's `N` (transform
/// half-length = total per-channel bin count for the frame size) and the
/// channel count, then call [`Self::synthesize_frame`] per CELT frame.
/// The frame size `N` is fixed for the life of the state — a CELT stream
/// whose frame size changes mid-stream resets the CELT state (§4.5.2), so
/// a fresh [`CeltSynthState`] is built at the transition.
///
/// # Examples
///
/// ```
/// use oxideav_opus::celt_band_layout::CeltFrameSize;
/// use oxideav_opus::celt_synthesis::CeltSynthState;
///
/// // A mono 20 ms CELT-only stream: N = 960 bins per channel.
/// let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
/// assert_eq!(st.transform_half_len(), 960);
/// assert_eq!(st.channels(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CeltSynthState {
    /// The transform half-length `N` = the full per-channel MDCT size
    /// (`frame_samples`): each frame produces an `N`-bin frequency buffer
    /// (whose coded prefix the denormaliser fills) and emits `N` PCM
    /// samples.
    n: usize,
    /// The number of *coded* MDCT bins per channel — the length of the
    /// denormalised prefix of the `N`-bin frequency buffer (Table-55
    /// column sum); bins `coded_bins..n` are the zero high-pad.
    coded_bins: usize,
    /// The number of coded bands per channel for this frame size / mode.
    coded_bands: usize,
    /// First coded band index (0 for CELT-only, 17 for Hybrid).
    first_coded_band: usize,
    /// The frame size (Table-55 column) every frame in this stream uses.
    frame_size: CeltFrameSize,
    /// Whether the frames are Hybrid (CELT layer starts at band 17).
    is_hybrid: bool,
    /// Per-channel backend state (1 or 2 entries).
    channels: Vec<CeltChannelState>,
}

impl CeltSynthState {
    /// Build a fresh CELT synthesis backend for the given frame size,
    /// Hybrid flag, and channel count (1 = mono, 2 = stereo), with zeroed
    /// overlap and de-emphasis state — the stream-start / post-§4.5.2-reset
    /// state.
    ///
    /// `N` is the full per-channel MDCT size for `frame_size`
    /// (`frame_samples`: 120 / 240 / 480 / 960); the coded denormalised
    /// prefix length is [`celt_total_bins_per_channel`]. The §4.3.7
    /// overlap is the fixed 48 kHz value `CELT_OVERLAP_48K` (120 samples),
    /// clamped to `N` for the 2.5 ms frame where `N == 120`.
    ///
    /// # Errors
    ///
    /// Returns [`CeltSynthError::ChannelCountMismatch`] if `channels` is
    /// not 1 or 2, or [`CeltSynthError::OverlapAdd`] if the derived
    /// `(N, overlap)` pair is rejected by the overlap-add constructor
    /// (only possible for a degenerate zero-bin layout).
    pub fn new(
        frame_size: CeltFrameSize,
        is_hybrid: bool,
        channels: usize,
    ) -> Result<Self, CeltSynthError> {
        if channels != 1 && channels != 2 {
            return Err(CeltSynthError::ChannelCountMismatch {
                expected: 1,
                got: channels,
            });
        }
        // N is the full MDCT size = frame_samples at 48 kHz
        // (tenths_ms * 48 / 10): 120 / 240 / 480 / 960. The denormalised
        // bands fill only the lower `coded_bins`; the rest is the zero
        // high-pad above the 20 kHz band edge.
        let n = (frame_size.to_frame_tenths_ms() as usize * 48) / 10;
        let coded_bins = celt_total_bins_per_channel(frame_size, is_hybrid) as usize;
        // The CELT overlap is fixed at 120 (2.5 ms at 48 kHz); for the
        // 2.5 ms frame N == 120, so the overlap equals N (the maximum the
        // §4.3.7 low-overlap construction allows). For larger frames the
        // overlap stays 120 < N.
        let overlap = CELT_OVERLAP_48K.min(n);
        let first = celt_first_coded_band(is_hybrid);
        let coded = crate::celt_band_layout::celt_end_coded_band() - first;

        let mut chans = Vec::with_capacity(channels);
        for _ in 0..channels {
            let ola = WeightedOverlapAdd::new(n, overlap).map_err(CeltSynthError::OverlapAdd)?;
            chans.push(CeltChannelState {
                ola,
                deemph: DeemphasisFilter::new(),
            });
        }

        Ok(Self {
            n,
            coded_bins,
            coded_bands: coded,
            first_coded_band: first,
            frame_size,
            is_hybrid,
            channels: chans,
        })
    }

    /// The transform half-length `N`: each frame emits this many PCM
    /// samples per channel.
    #[must_use]
    pub fn transform_half_len(&self) -> usize {
        self.n
    }

    /// The number of coded bands per channel for this stream's frame size
    /// / mode.
    #[must_use]
    pub fn coded_bands(&self) -> usize {
        self.coded_bands
    }

    /// The number of *coded* MDCT bins per channel — the denormalised
    /// prefix length (Table-55 column sum); bins `coded_bins..N` of the
    /// frequency buffer are the zero high-pad above 20 kHz.
    #[must_use]
    pub fn coded_bins(&self) -> usize {
        self.coded_bins
    }

    /// The first coded band index (0 for CELT-only, 17 for Hybrid).
    #[must_use]
    pub fn first_coded_band(&self) -> usize {
        self.first_coded_band
    }

    /// The number of channels this state threads (1 or 2).
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Reset the §4.3.7 overlap-add history and §4.3.7.2 de-emphasis
    /// memory of every channel to zero, as on a §4.5.2 CELT state reset.
    pub fn reset(&mut self) {
        for ch in self.channels.iter_mut() {
            ch.ola.reset();
            ch.deemph.reset();
        }
    }

    /// Synthesize one CELT frame for one channel: run the §4.3.6 →
    /// §4.3.7 → §4.3.7.2 backend on that channel's per-band shapes and
    /// energies, advancing the channel's overlap-add and de-emphasis
    /// state, and write the `N` time-domain PCM samples into `out`.
    ///
    /// `shapes[k]` is the unit-L2-norm shape of the `k`-th coded band
    /// (length = that band's Table-55 bin count); `log2_energy[k]` is the
    /// matching band's reconstructed base-2-log energy. `channel` selects
    /// the per-channel state (0 = first/mono, 1 = second/stereo).
    ///
    /// # Errors
    ///
    /// * [`CeltSynthError::BandCountMismatch`] if the shape / energy count
    ///   does not match [`Self::coded_bands`].
    /// * [`CeltSynthError::ChannelCountMismatch`] if `channel` is out of
    ///   range for this state, or `out.len() != N`.
    /// * [`CeltSynthError::Denormalise`] / [`CeltSynthError::Imdct`] /
    ///   [`CeltSynthError::OverlapAdd`] propagated from the stages.
    pub fn synthesize_channel_into(
        &mut self,
        channel: usize,
        shapes: &[&[f64]],
        log2_energy: &[f64],
        out: &mut [f64],
    ) -> Result<(), CeltSynthError> {
        if channel >= self.channels.len() {
            return Err(CeltSynthError::ChannelCountMismatch {
                expected: self.channels.len(),
                got: channel + 1,
            });
        }
        if shapes.len() != self.coded_bands || log2_energy.len() != self.coded_bands {
            return Err(CeltSynthError::BandCountMismatch {
                expected: self.coded_bands,
                got_shapes: shapes.len(),
                got_energies: log2_energy.len(),
            });
        }
        if out.len() != self.n {
            return Err(CeltSynthError::ChannelCountMismatch {
                expected: self.n,
                got: out.len(),
            });
        }

        // Stage 1 — §4.3.6 denormalise the bands into the N-bin frequency
        // buffer.
        let mut freq = vec![0.0_f64; self.n];
        let written = denormalise_bands(
            shapes,
            log2_energy,
            self.frame_size,
            self.is_hybrid,
            &mut freq,
        )
        .map_err(CeltSynthError::Denormalise)?;
        debug_assert_eq!(written, self.coded_bins);

        // Stage 2 — §4.3.7 inverse MDCT into the 2N time-domain block.
        let mut block = vec![0.0_f64; 2 * self.n];
        imdct_into(&freq, &mut block).map_err(CeltSynthError::Imdct)?;

        // Stage 3 — §4.3.7 windowed overlap-add (carries the trailing-half
        // history forward), emitting N aliasing-free samples into `out`.
        let ch = &mut self.channels[channel];
        ch.ola
            .process_into(&block, out)
            .map_err(CeltSynthError::OverlapAdd)?;

        // Stage 4 — §4.3.7.2 de-emphasis in place (continuous one-pole
        // memory across frames).
        ch.deemph.process_in_place(out);

        Ok(())
    }

    /// Synthesize one CELT frame across every channel and return the
    /// interleaved 48 kHz signed-16-bit PCM (`[c0_s0, c1_s0, c0_s1, …]`),
    /// advancing each channel's overlap-add and de-emphasis state.
    ///
    /// `per_channel[c]` is channel `c`'s `(shapes, log2_energy)` pair (the
    /// same arguments [`Self::synthesize_channel_into`] takes). The slice
    /// length must equal [`Self::channels`]. Each channel's `N`
    /// time-domain samples are scaled by `32768`, saturated to the i16
    /// range, and rounded ties-to-even — the same conversion the decoder
    /// applies to the SILK path so CELT and SILK output share one
    /// amplitude convention. Returns `channels * N` interleaved samples.
    ///
    /// # Errors
    ///
    /// * [`CeltSynthError::ChannelCountMismatch`] if `per_channel.len()`
    ///   does not equal [`Self::channels`].
    /// * The per-channel errors from [`Self::synthesize_channel_into`].
    pub fn synthesize_frame_interleaved_i16(
        &mut self,
        per_channel: &[(&[&[f64]], &[f64])],
    ) -> Result<Vec<i16>, CeltSynthError> {
        if per_channel.len() != self.channels.len() {
            return Err(CeltSynthError::ChannelCountMismatch {
                expected: self.channels.len(),
                got: per_channel.len(),
            });
        }
        let n = self.n;
        let ch_count = self.channels.len();
        // Synthesize each channel into its own scratch buffer first, then
        // interleave (the per-channel state borrows are sequential).
        let mut planar: Vec<Vec<f64>> = Vec::with_capacity(ch_count);
        for (c, (shapes, energies)) in per_channel.iter().enumerate() {
            let mut buf = vec![0.0_f64; n];
            self.synthesize_channel_into(c, shapes, energies, &mut buf)?;
            planar.push(buf);
        }
        let mut out = vec![0_i16; ch_count * n];
        for (s, slot) in out.chunks_exact_mut(ch_count).enumerate() {
            for (c, dst) in slot.iter_mut().enumerate() {
                *dst = celt_sample_to_i16(planar[c][s]);
            }
        }
        Ok(out)
    }
}

/// Convert one §4.3.7.2-domain time sample to signed 16-bit PCM, matching
/// the decoder's SILK conversion and the RFC 6716 §A reference listing's
/// output quantization of the normative float signal: scale by `32768`,
/// saturate to `[-32768, 32767]`, and round to nearest with ties to even.
/// A free function so the conversion is shared and unit-testable.
#[inline]
#[must_use]
fn celt_sample_to_i16(v: f64) -> i16 {
    let scaled = (v * 32768.0).clamp(-32768.0, 32767.0);
    scaled.round_ties_even() as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_band_layout::celt_band_bins_per_channel;

    /// Build a vector of per-band all-zero shapes (one zero slice per
    /// coded band, each of its Table-55 bin length) plus a matching
    /// zero-energy vector, for a given frame size / mode.
    fn zero_frame(frame_size: CeltFrameSize, is_hybrid: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
        let first = celt_first_coded_band(is_hybrid);
        let end = crate::celt_band_layout::celt_end_coded_band();
        let mut shapes = Vec::new();
        let mut energies = Vec::new();
        for band in first..end {
            let bins = celt_band_bins_per_channel(band, frame_size).unwrap() as usize;
            shapes.push(vec![0.0_f64; bins]);
            energies.push(0.0_f64);
        }
        (shapes, energies)
    }

    fn shape_refs(shapes: &[Vec<f64>]) -> Vec<&[f64]> {
        shapes.iter().map(|s| s.as_slice()).collect()
    }

    #[test]
    fn new_rejects_bad_channel_count() {
        assert!(matches!(
            CeltSynthState::new(CeltFrameSize::Ms20, false, 0),
            Err(CeltSynthError::ChannelCountMismatch { .. })
        ));
        assert!(matches!(
            CeltSynthState::new(CeltFrameSize::Ms20, false, 3),
            Err(CeltSynthError::ChannelCountMismatch { .. })
        ));
        assert!(CeltSynthState::new(CeltFrameSize::Ms20, false, 1).is_ok());
        assert!(CeltSynthState::new(CeltFrameSize::Ms20, false, 2).is_ok());
    }

    #[test]
    fn transform_half_len_matches_table55_total() {
        // 20 ms CELT-only: MDCT size N = 960; coded bands fill 800 bins,
        // bins 800..960 are the zero high-pad above 20 kHz.
        let st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        assert_eq!(st.transform_half_len(), 960);
        assert_eq!(st.coded_bins(), 800);
        assert!(st.coded_bins() < st.transform_half_len());
        assert_eq!(st.coded_bands(), 21);
        assert_eq!(st.first_coded_band(), 0);
        // 2.5 ms CELT-only: N = 960 / 8 = 120; coded bands fill 100;
        // overlap == N here.
        let st2 = CeltSynthState::new(CeltFrameSize::Ms2_5, false, 1).unwrap();
        assert_eq!(st2.transform_half_len(), 120);
        assert_eq!(st2.coded_bins(), 100);
        // Hybrid 20 ms: only bands 17..21 are CELT-coded.
        let sth = CeltSynthState::new(CeltFrameSize::Ms20, true, 1).unwrap();
        assert_eq!(sth.first_coded_band(), 17);
        assert_eq!(sth.coded_bands(), 4);
    }

    #[test]
    fn silent_frame_decodes_to_silence() {
        // A zero-shape, zero-energy frame produces N silent samples. With
        // log2_energy = 0 the denormalise gain is 1, but a zero shape ×
        // any gain is zero, so the frequency buffer is all-zero, the
        // IMDCT of zeros is zero, and the de-emphasis of zeros stays zero.
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        let mut out = vec![1.0_f64; st.transform_half_len()];
        st.synthesize_channel_into(0, &refs, &energies, &mut out)
            .unwrap();
        assert!(out.iter().all(|&s| s == 0.0), "silent frame must be silent");
    }

    #[test]
    fn band_count_mismatch_is_rejected() {
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let mut refs = shape_refs(&shapes);
        refs.pop(); // one fewer band than coded_bands
        let mut out = vec![0.0_f64; st.transform_half_len()];
        let err = st
            .synthesize_channel_into(0, &refs, &energies, &mut out)
            .unwrap_err();
        assert!(matches!(err, CeltSynthError::BandCountMismatch { .. }));
    }

    #[test]
    fn out_len_mismatch_is_rejected() {
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        let mut out = vec![0.0_f64; st.transform_half_len() - 1];
        let err = st
            .synthesize_channel_into(0, &refs, &energies, &mut out)
            .unwrap_err();
        assert!(matches!(err, CeltSynthError::ChannelCountMismatch { .. }));
    }

    #[test]
    fn channel_out_of_range_is_rejected() {
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        let mut out = vec![0.0_f64; st.transform_half_len()];
        // Channel 1 does not exist on a mono state.
        let err = st
            .synthesize_channel_into(1, &refs, &energies, &mut out)
            .unwrap_err();
        assert!(matches!(err, CeltSynthError::ChannelCountMismatch { .. }));
    }

    #[test]
    fn reset_clears_overlap_and_deemphasis() {
        // Decode a non-silent frame, then reset, then a silent frame:
        // after a reset the silent frame must produce exact silence (no
        // leftover overlap history or de-emphasis memory bleeding in).
        let mut st = CeltSynthState::new(CeltFrameSize::Ms10, false, 1).unwrap();
        let (mut shapes, mut energies) = zero_frame(CeltFrameSize::Ms10, false);
        // Put a single nonzero coefficient in band 5's shape, with energy.
        shapes[5][0] = 1.0;
        energies[5] = 4.0;
        let refs = shape_refs(&shapes);
        let mut out = vec![0.0_f64; st.transform_half_len()];
        st.synthesize_channel_into(0, &refs, &energies, &mut out)
            .unwrap();
        assert!(
            out.iter().any(|&s| s != 0.0),
            "nonzero frame should be audible"
        );

        st.reset();
        let (zshapes, zenergies) = zero_frame(CeltFrameSize::Ms10, false);
        let zrefs = shape_refs(&zshapes);
        let mut out2 = vec![0.0_f64; st.transform_half_len()];
        st.synthesize_channel_into(0, &zrefs, &zenergies, &mut out2)
            .unwrap();
        assert!(
            out2.iter().all(|&s| s == 0.0),
            "after reset, a silent frame must be exactly silent"
        );
    }

    #[test]
    fn two_silent_frames_stay_silent_across_state() {
        // Continuity check: two consecutive silent frames decode to all
        // zeros (the overlap history of frame 1 is all zeros, so frame 2's
        // overlap-add adds nothing).
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        for _ in 0..2 {
            let mut out = vec![0.0_f64; st.transform_half_len()];
            st.synthesize_channel_into(0, &refs, &energies, &mut out)
                .unwrap();
            assert!(out.iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn stereo_channels_are_independent() {
        // A stereo state threads two independent overlap/de-emphasis
        // states: a nonzero frame on channel 0 must not perturb channel 1.
        let mut st = CeltSynthState::new(CeltFrameSize::Ms10, false, 2).unwrap();
        assert_eq!(st.channels(), 2);
        let (mut shapes, mut energies) = zero_frame(CeltFrameSize::Ms10, false);
        shapes[3][0] = 1.0;
        energies[3] = 2.0;
        let refs = shape_refs(&shapes);
        let mut out0 = vec![0.0_f64; st.transform_half_len()];
        st.synthesize_channel_into(0, &refs, &energies, &mut out0)
            .unwrap();

        // Channel 1 decodes a silent frame; it must be exactly silent,
        // unaffected by channel 0's nonzero overlap history.
        let (zshapes, zenergies) = zero_frame(CeltFrameSize::Ms10, false);
        let zrefs = shape_refs(&zshapes);
        let mut out1 = vec![0.0_f64; st.transform_half_len()];
        st.synthesize_channel_into(1, &zrefs, &zenergies, &mut out1)
            .unwrap();
        assert!(out1.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn energy_increases_with_log2_energy() {
        // A larger log2 band energy must scale the synthesized output
        // amplitude up: the §4.3.6 denormalise gain is 2**(L/2), so the
        // sum of squared output samples should grow with L. Two fresh
        // states decode the same shape at energies L and L+2; the higher
        // energy must yield strictly more output power.
        let mk = |l: f64| -> f64 {
            let mut st = CeltSynthState::new(CeltFrameSize::Ms10, false, 1).unwrap();
            let (mut shapes, mut energies) = zero_frame(CeltFrameSize::Ms10, false);
            shapes[2][0] = 1.0;
            energies[2] = l;
            let refs = shape_refs(&shapes);
            let mut out = vec![0.0_f64; st.transform_half_len()];
            st.synthesize_channel_into(0, &refs, &energies, &mut out)
                .unwrap();
            out.iter().map(|&s| s * s).sum::<f64>()
        };
        let low = mk(0.0);
        let high = mk(2.0);
        assert!(
            high > low,
            "higher band energy must yield more output power"
        );
        assert!(low > 0.0, "a nonzero shape at L=0 must be audible");
    }

    #[test]
    fn sample_to_i16_matches_decoder_convention() {
        // The shared conversion: ×32768, saturate, round ties-to-even.
        assert_eq!(celt_sample_to_i16(0.0), 0);
        // +1.0 scales to 32768, which saturates at the i16 ceiling.
        assert_eq!(celt_sample_to_i16(1.0), 32767);
        assert_eq!(celt_sample_to_i16(-1.0), -32768);
        // Saturation beyond range.
        assert_eq!(celt_sample_to_i16(2.5), 32767);
        assert_eq!(celt_sample_to_i16(-3.0), -32768);
        // Rounding to nearest.
        assert_eq!(celt_sample_to_i16(16384.4 / 32768.0), 16384);
        assert_eq!(celt_sample_to_i16(16384.6 / 32768.0), 16385);
        // Exact halves round to even.
        assert_eq!(celt_sample_to_i16(16384.5 / 32768.0), 16384);
        assert_eq!(celt_sample_to_i16(16383.5 / 32768.0), 16384);
        assert_eq!(celt_sample_to_i16(-0.5 / 32768.0), 0);
    }

    #[test]
    fn interleaved_silent_frame_is_all_zero_i16() {
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 2).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        let per_channel: Vec<(&[&[f64]], &[f64])> = vec![
            (refs.as_slice(), energies.as_slice()),
            (refs.as_slice(), energies.as_slice()),
        ];
        let pcm = st.synthesize_frame_interleaved_i16(&per_channel).unwrap();
        assert_eq!(pcm.len(), 2 * st.transform_half_len());
        assert!(pcm.iter().all(|&s| s == 0));
    }

    #[test]
    fn interleaved_wrong_channel_count_rejected() {
        let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 2).unwrap();
        let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
        let refs = shape_refs(&shapes);
        // Only one channel supplied to a stereo state.
        let per_channel: Vec<(&[&[f64]], &[f64])> = vec![(refs.as_slice(), energies.as_slice())];
        let err = st
            .synthesize_frame_interleaved_i16(&per_channel)
            .unwrap_err();
        assert!(matches!(err, CeltSynthError::ChannelCountMismatch { .. }));
    }

    #[test]
    fn interleaved_layout_places_channels_correctly() {
        // Channel 0 carries audible content, channel 1 is silent: the
        // interleaved output must have zeros at every odd index (channel 1)
        // and at least one nonzero at an even index (channel 0).
        let mut st = CeltSynthState::new(CeltFrameSize::Ms10, false, 2).unwrap();
        let (mut a_shapes, mut a_energies) = zero_frame(CeltFrameSize::Ms10, false);
        a_shapes[4][0] = 1.0;
        a_energies[4] = 6.0;
        let a_refs = shape_refs(&a_shapes);
        let (z_shapes, z_energies) = zero_frame(CeltFrameSize::Ms10, false);
        let z_refs = shape_refs(&z_shapes);
        let per_channel: Vec<(&[&[f64]], &[f64])> = vec![
            (a_refs.as_slice(), a_energies.as_slice()),
            (z_refs.as_slice(), z_energies.as_slice()),
        ];
        let pcm = st.synthesize_frame_interleaved_i16(&per_channel).unwrap();
        let n = st.transform_half_len();
        assert_eq!(pcm.len(), 2 * n);
        // Odd indices = channel 1 (silent) → all zero.
        assert!(pcm.iter().skip(1).step_by(2).all(|&s| s == 0));
        // Even indices = channel 0 → at least one audible sample.
        assert!(pcm.iter().step_by(2).any(|&s| s != 0));
    }
}
