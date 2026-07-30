//! CELT §4.3.6 → §4.3.7 → §4.3.7.2 synthesis-backend integration tests.
//!
//! These exercise `oxideav_opus::celt_synthesis::CeltSynthState` — the
//! composition of the §4.3.6 denormalise, §4.3.7 inverse MDCT, §4.3.7
//! weighted overlap-add, and §4.3.7.2 de-emphasis stages — through its
//! public API across *multiple frames*, the way the decoder will drive it
//! once the gapped CELT entropy front half lands. They pin the cross-frame
//! state-continuity properties the RFC's continuous reconstruction relies
//! on, and an independent re-derivation of the backend against the
//! per-stage primitives.
//!
//! Provenance: this file reads only RFC 6716 §4.3.6 / §4.3.7 / §4.3.7.2
//! (the denormalise / inverse-MDCT / overlap-add / de-emphasis stage
//! definitions, `docs/audio/opus/rfc6716-opus.txt`) and uses only the
//! crate's public API. No external library source consulted; the upstream
//! entropy decode (coarse energy / PVQ) is deliberately *not* exercised —
//! its bitstream stages have an open clean-room gap (`ec_laplace_decode`),
//! so these tests feed already-decoded band shapes + energies, exactly the
//! backend's documented input boundary.

use oxideav_opus::celt_band_layout::{
    celt_band_bins_per_channel, celt_end_coded_band, celt_first_coded_band, CeltFrameSize,
};
use oxideav_opus::celt_deemphasis::DeemphasisFilter;
use oxideav_opus::celt_denormalise::denormalise_bands;
use oxideav_opus::celt_imdct::imdct;
use oxideav_opus::celt_mdct_window::CELT_OVERLAP_48K;
use oxideav_opus::celt_overlap_add::WeightedOverlapAdd;
use oxideav_opus::celt_synthesis::CeltSynthState;

/// Build a frame's per-band zero shapes + zero energies for a frame size.
fn zero_frame(frame_size: CeltFrameSize, is_hybrid: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let first = celt_first_coded_band(is_hybrid);
    let end = celt_end_coded_band();
    let mut shapes = Vec::new();
    let mut energies = Vec::new();
    for band in first..end {
        let bins = celt_band_bins_per_channel(band, frame_size).unwrap() as usize;
        shapes.push(vec![0.0_f64; bins]);
        energies.push(0.0_f64);
    }
    (shapes, energies)
}

fn refs(shapes: &[Vec<f64>]) -> Vec<&[f64]> {
    shapes.iter().map(|s| s.as_slice()).collect()
}

/// A long all-silent CELT-only stream decodes to exact digital silence,
/// frame after frame — the overlap-add history of a silent frame is zero,
/// so nothing accumulates and the de-emphasis pole stays at rest.
#[test]
fn long_silent_celt_stream_is_exact_silence() {
    let mut st = CeltSynthState::new(CeltFrameSize::Ms20, false, 1).unwrap();
    let (shapes, energies) = zero_frame(CeltFrameSize::Ms20, false);
    let r = refs(&shapes);
    let n = st.transform_half_len();
    for frame in 0..50 {
        let mut out = vec![1.0_f64; n];
        st.synthesize_channel_into(0, &r, &energies, &mut out)
            .unwrap();
        assert!(
            out.iter().all(|&s| s == 0.0),
            "frame {frame} of a silent stream must be exactly silent"
        );
    }
}

/// The backend (sans de-emphasis) re-derives stage-for-stage against the
/// public per-stage primitives: a hand-built two-frame stream pushed
/// through `denormalise_bands` → `imdct` → `WeightedOverlapAdd` must match
/// what `CeltSynthState` produces once its de-emphasis is undone.
///
/// De-emphasis is a deterministic one-pole post-filter, so undoing it
/// (applying the inverse pre-emphasis recurrence `x(n) = y(n) -
/// alpha*y(n-1)` with the same continuous memory) recovers the pre-filter
/// samples; those must equal the manual denormalise→imdct→OLA pipeline
/// exactly.
#[test]
fn backend_matches_manual_per_stage_pipeline() {
    let fs = CeltFrameSize::Ms10;
    let mut st = CeltSynthState::new(fs, false, 1).unwrap();
    let n = st.transform_half_len();
    let overlap = CELT_OVERLAP_48K.min(n);

    // Manual reference pipeline state.
    let mut ola = WeightedOverlapAdd::new(n, overlap).unwrap();

    // Track the de-emphasis state to undo it on the backend output.
    let mut deemph = DeemphasisFilter::new();

    // Two frames with distinct band content.
    let mut frames: Vec<(Vec<Vec<f64>>, Vec<f64>)> = Vec::new();
    for seed in 0..2u32 {
        let (mut shapes, mut energies) = zero_frame(fs, false);
        // Put a deterministic unit-norm shape in a couple of bands.
        let b0 = 2 + seed as usize;
        shapes[b0][0] = 1.0; // already unit-norm (single coeff)
        energies[b0] = 3.0 + seed as f64;
        let b1 = 8;
        // A 2-coeff unit-norm shape: (0.6, 0.8).
        shapes[b1][0] = 0.6;
        shapes[b1][1] = 0.8;
        energies[b1] = 1.0;
        frames.push((shapes, energies));
    }

    for (shapes, energies) in &frames {
        let r = refs(shapes);

        // --- Manual reference: denormalise → imdct → windowed OLA.
        let mut freq = vec![0.0_f64; n];
        denormalise_bands(&r, energies, fs, false, &mut freq).unwrap();
        let block = imdct(&freq).unwrap();
        let manual_pre_deemph = ola.process(&block).unwrap();

        // --- Backend under test (includes de-emphasis).
        let mut backend = vec![0.0_f64; n];
        st.synthesize_channel_into(0, &r, energies, &mut backend)
            .unwrap();

        // Undo the backend's de-emphasis with the matching continuous
        // inverse pre-emphasis recurrence; the result must equal the
        // manual pre-de-emphasis samples bit-for-bit (same arithmetic).
        let alpha = oxideav_opus::celt_deemphasis::DEEMPHASIS_ALPHA_P;
        let mut recovered = vec![0.0_f64; n];
        for i in 0..n {
            // y(n) = backend[i]; x(n) = y(n) - alpha*y(n-1).
            let prev = deemph.memory();
            recovered[i] = backend[i] - alpha * prev;
            // advance deemph by feeding it the recovered x so its memory
            // matches the backend's internal memory after this sample.
            let _ = deemph.step(recovered[i]);
        }

        for (i, (&a, &b)) in manual_pre_deemph.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "sample {i}: manual {a} vs de-emphasis-undone backend {b}"
            );
        }
    }
}

/// A stream that repeats the *same* frequency content every frame reaches
/// a steady state: once the overlap history is primed (after the first
/// frame), each subsequent frame's pre-de-emphasis output is identical to
/// the one before it (the overlap-add of identical blocks is
/// shift-invariant). This is the §4.3.7 TDAC steady-state property.
#[test]
fn constant_spectrum_reaches_steady_state() {
    let fs = CeltFrameSize::Ms10;
    let n = (fs.to_frame_tenths_ms() as usize * 48) / 10;
    let overlap = CELT_OVERLAP_48K.min(n);
    let mut ola = WeightedOverlapAdd::new(n, overlap).unwrap();

    let (mut shapes, mut energies) = zero_frame(fs, false);
    shapes[4][0] = 1.0;
    energies[4] = 5.0;
    let r = refs(&shapes);

    let mut freq = vec![0.0_f64; n];
    denormalise_bands(&r, &energies, fs, false, &mut freq).unwrap();
    let block = imdct(&freq).unwrap();

    let f0 = ola.process(&block).unwrap();
    let f1 = ola.process(&block).unwrap();
    let f2 = ola.process(&block).unwrap();

    // The first frame is not yet steady (history was zero), but f1 and f2
    // must be identical — the system has reached its periodic steady state.
    for (i, (&a, &b)) in f1.iter().zip(f2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "steady-state sample {i}: f1 {a} vs f2 {b}"
        );
    }
    // And f0 should differ from f1 in the overlap region (the priming
    // transient), confirming the history actually does something.
    let differs = f0
        .iter()
        .zip(f1.iter())
        .take(overlap)
        .any(|(&a, &b)| (a - b).abs() > 1e-9);
    assert!(
        differs,
        "the first frame must show the overlap priming transient"
    );
}

/// Inter-frame continuity: feeding two frames with the same content, the
/// junction between consecutive output frames is smooth — the overlap-add
/// is exactly what makes the concatenated output free of block-boundary
/// discontinuities. We check that the last sample of frame k and the first
/// sample of frame k+1 do not jump more than the in-frame sample-to-sample
/// variation, after the steady state is reached.
#[test]
fn frame_boundary_is_continuous() {
    let fs = CeltFrameSize::Ms10;
    let mut st = CeltSynthState::new(fs, false, 1).unwrap();
    let n = st.transform_half_len();
    let (mut shapes, mut energies) = zero_frame(fs, false);
    shapes[3][0] = 0.6;
    shapes[3][1] = 0.8;
    energies[3] = 4.0;
    let r = refs(&shapes);

    // Prime three frames; compare the boundary between frame 1 and 2.
    let mut prev: Vec<f64> = Vec::new();
    let mut boundary_jump = f64::INFINITY;
    let mut max_in_frame = 0.0_f64;
    for k in 0..3 {
        let mut out = vec![0.0_f64; n];
        st.synthesize_channel_into(0, &r, &energies, &mut out)
            .unwrap();
        if k == 2 {
            // boundary jump = |first sample of this frame - last of prev|
            boundary_jump = (out[0] - *prev.last().unwrap()).abs();
            max_in_frame = out
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0, f64::max);
        }
        prev = out;
    }
    // The boundary discontinuity must be on the order of (not wildly larger
    // than) the largest in-frame step — overlap-add stitches the blocks.
    assert!(
        boundary_jump <= max_in_frame * 4.0 + 1e-6,
        "frame boundary jump {boundary_jump} dwarfs in-frame variation {max_in_frame}"
    );
}

/// Output is bounded and finite for a realistic multi-band frame at every
/// supported frame size and channel count.
#[test]
fn all_frame_sizes_produce_finite_bounded_output() {
    for &fs in &[
        CeltFrameSize::Ms2_5,
        CeltFrameSize::Ms5,
        CeltFrameSize::Ms10,
        CeltFrameSize::Ms20,
    ] {
        for &channels in &[1usize, 2] {
            let mut st = CeltSynthState::new(fs, false, channels).unwrap();
            let n = st.transform_half_len();
            let (mut shapes, mut energies) = zero_frame(fs, false);
            // Spread modest energy across a few bands.
            for b in [1usize, 5, 9, 14] {
                shapes[b][0] = 1.0;
                energies[b] = 2.0;
            }
            let r = refs(&shapes);
            for c in 0..channels {
                let mut out = vec![0.0_f64; n];
                st.synthesize_channel_into(c, &r, &energies, &mut out)
                    .unwrap();
                assert!(
                    out.iter().all(|s| s.is_finite()),
                    "fs={fs:?} ch={c}: output must be finite"
                );
                // Energy 2.0 → linear 4 → per-band amplitude ~2; a handful
                // of bands → bounded well below any clipping disaster.
                assert!(
                    out.iter().all(|&s| s.abs() < 100.0),
                    "fs={fs:?} ch={c}: output must be bounded"
                );
            }
        }
    }
}
