use ndarray::{Array2, ArrayBase, ArrayViewMut2, Data, Ix2, Zip};
use profluens_core::memory::Arena;
use std::mem::{align_of, size_of};

/// Carve an uninitialised `n`-element `T` slice from the bump `arena`. The backing bytes are reused
/// arena memory (contents unspecified), so **every element must be written before it is read**.
///
/// `T` is restricted to the plain-old-data numeric types this crate carves (`f64`, `Complex64`):
/// every bit pattern is a valid value, none has a `Drop`, so handing out a typed view over raw
/// arena bytes is sound. Zero heap traffic on the steady-state path — the arena reuses its chunks
/// across [`Arena::reset`], which the per-patch loops do.
pub(crate) trait ArenaPod: Copy {}
impl ArenaPod for f64 {}
impl ArenaPod for num::complex::Complex64 {}

pub(crate) fn arena_slice<T: ArenaPod>(arena: &Arena, n: usize) -> &mut [T] {
    let bytes = arena.alloc(n * size_of::<T>(), align_of::<T>());
    // SAFETY: `bytes` is exactly `n * size_of::<T>()` bytes, `align_of::<T>()`-aligned; `T: ArenaPod`
    // has no invalid bit patterns and no `Drop`, and callers write every element before use.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), n) }
}

/// Carve an uninitialised `r × c` row-major `f64` matrix from the bump `arena` as a mutable view.
/// See [`arena_slice`] — every element must be written (or the whole view `fill`ed) before it is
/// read.
pub(crate) fn arena_mat(arena: &Arena, r: usize, c: usize) -> ArrayViewMut2<'_, f64> {
    let data = arena_slice::<f64>(arena, r * c);
    ArrayViewMut2::from_shape((r, c), data).expect("arena_mat: slice length matches r*c")
}

/// `out[i] = op(a[i], b[i])` into a fresh arena matrix — the allocation-free analogue of ndarray's
/// `&a <op> &b` (which mints a heap `Array2` per call). `a` and `b` may be any storage (owned,
/// arena view, …) and must share `a`'s shape. Same per-element op in the same order as the
/// operator form, so the result is bit-identical.
pub(crate) fn binop_into<'a, S1, S2>(
    arena: &'a Arena,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    op: impl Fn(f64, f64) -> f64,
) -> ArrayViewMut2<'a, f64>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let (r, c) = a.dim();
    let mut out = arena_mat(arena, r, c);
    Zip::from(&mut out).and(a).and(b).for_each(|o, &x, &y| *o = op(x, y));
    out
}

/// `out[i] = op(a[i])` into a fresh arena matrix — the arena analogue of `a.mapv(op)` / `&a <op> s`.
pub(crate) fn map_into<'a, S>(
    arena: &'a Arena,
    a: &ArrayBase<S, Ix2>,
    op: impl Fn(f64) -> f64,
) -> ArrayViewMut2<'a, f64>
where
    S: Data<Elem = f64>,
{
    let (r, c) = a.dim();
    let mut out = arena_mat(arena, r, c);
    Zip::from(&mut out).and(a).for_each(|o, &x| *o = op(x));
    out
}

/// Computes the convolution of `input_matrix` with `fir_filter`, writing the result into a fresh
/// arena matrix (was: a heap `Array2` per call). `input_matrix` is generic over storage so a conv
/// result (an arena view) can feed straight into the next conv without copying to an owned array.
pub fn perform_valid_2d_conv_with_boundary<'a, S>(
    fir_filter: &Array2<f64>,
    input_matrix: &ArrayBase<S, Ix2>,
    arena: &'a Arena,
) -> ArrayViewMut2<'a, f64>
where
    S: Data<Elem = f64>,
{
    let padded_matrix = add_matrix_boundary(input_matrix, arena);
    // `padded_matrix` is an `arena_mat` — standard-layout row-major — and the convolution indexes
    // it as `row*ncols + col`, so its backing slice is exactly the flattening this wants
    // (borrowed zero-copy, not rebuilt).
    let padded_flattened_matrix = padded_matrix
        .as_slice()
        .expect("padded matrix is standard-layout (row-major) contiguous");

    let i_r_c = padded_matrix.nrows();
    let i_c_c = padded_matrix.ncols();
    let f_r_c = fir_filter.nrows();
    let f_c_c = fir_filter.ncols();
    let o_r_c = i_r_c - f_r_c + 1;
    let o_c_c = i_c_c - f_c_c + 1;

    let flattened_filter = flatten_filter(fir_filter, arena);

    // Every (o_row, o_col) is written below, so the uninitialised arena matrix needs no fill.
    let mut out_matrix = arena_mat(arena, o_r_c, o_c_c);
    let out_slice = out_matrix
        .as_slice_mut()
        .expect("arena_mat output is standard-layout contiguous");
    conv2d_valid(
        padded_flattened_matrix,
        i_c_c,
        flattened_filter,
        f_r_c,
        f_c_c,
        o_r_c,
        o_c_c,
        out_slice,
    );
    out_matrix
}

/// Valid 2-D convolution core: `out[o_row][o_col] = Σ_taps padded[(f_row+o_row)*i_c_c + f_col+o_col]
/// * filter[filter_size-1 - tap]`. Vectorised across output columns (4 per AVX2 instruction) with a
/// scalar fallback. The per-output accumulation is the identical tap sequence in the identical order
/// as the scalar form and uses no FMA contraction, so it is bit-for-bit identical.
#[allow(clippy::too_many_arguments)]
fn conv2d_valid(
    padded: &[f64],
    i_c_c: usize,
    filter: &[f64],
    f_r_c: usize,
    f_c_c: usize,
    o_r_c: usize,
    o_c_c: usize,
    out: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated on runtime AVX2 detection; the body is plain safe Rust (no intrinsics) —
            // `target_feature` only lets the autovectoriser use 256-bit lanes.
            unsafe {
                conv2d_valid_avx2(padded, i_c_c, filter, f_r_c, f_c_c, o_r_c, o_c_c, out);
            }
            return;
        }
    }
    conv2d_valid_kernel(padded, i_c_c, filter, f_r_c, f_c_c, o_r_c, o_c_c, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d_valid_avx2(
    padded: &[f64],
    i_c_c: usize,
    filter: &[f64],
    f_r_c: usize,
    f_c_c: usize,
    o_r_c: usize,
    o_c_c: usize,
    out: &mut [f64],
) {
    conv2d_valid_kernel(padded, i_c_c, filter, f_r_c, f_c_c, o_r_c, o_c_c, out);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv2d_valid_kernel(
    padded: &[f64],
    i_c_c: usize,
    filter: &[f64],
    f_r_c: usize,
    f_c_c: usize,
    o_r_c: usize,
    o_c_c: usize,
    out: &mut [f64],
) {
    let filter_size = f_r_c * f_c_c;
    for o_row in 0..o_r_c {
        let out_row = &mut out[o_row * o_c_c..o_row * o_c_c + o_c_c];
        let mut o_col = 0;
        // Four output columns at a time: `padded[idx..idx+4]` is four adjacent columns (row-major,
        // stride 1) and the filter tap is a shared scalar, so the four MACs vectorise.
        while o_col + 4 <= o_c_c {
            let mut sum4 = [0.0f64; 4];
            let mut filter_index = filter_size - 1;
            for f_col in 0..f_c_c {
                for f_row in 0..f_r_c {
                    let idx = (f_row + o_row) * i_c_c + f_col + o_col;
                    let fv = filter[filter_index];
                    let p = &padded[idx..idx + 4];
                    for k in 0..4 {
                        sum4[k] += p[k] * fv;
                    }
                    filter_index = filter_index.saturating_sub(1);
                }
            }
            out_row[o_col..o_col + 4].copy_from_slice(&sum4);
            o_col += 4;
        }
        // Remainder columns — identical scalar accumulation.
        while o_col < o_c_c {
            let mut sum = 0.0f64;
            let mut filter_index = filter_size - 1;
            for f_col in 0..f_c_c {
                for f_row in 0..f_r_c {
                    let idx = (f_row + o_row) * i_c_c + f_col + o_col;
                    sum += padded[idx] * filter[filter_index];
                    filter_index = filter_index.saturating_sub(1);
                }
            }
            out_row[o_col] = sum;
            o_col += 1;
        }
    }
}

/// Row-major flatten of the (Fortran-order) FIR filter into arena memory. Row-major via `[(i, j)]`
/// indexing — independent of the array's layout, matching the convolution's reverse indexing (so
/// it is deliberately *not* `as_slice`, which would give the filter's Fortran memory order).
fn flatten_filter<'a>(input_matrix: &Array2<f64>, arena: &'a Arena) -> &'a mut [f64] {
    let (r, c) = (input_matrix.nrows(), input_matrix.ncols());
    let bytes = arena.alloc(r * c * size_of::<f64>(), align_of::<f64>());
    // SAFETY: as in `arena_mat`; every element is written just below before any read.
    let out = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f64>(), r * c) };
    let mut k = 0;
    for i in 0..r {
        for j in 0..c {
            out[k] = input_matrix[(i, j)];
            k += 1;
        }
    }
    out
}

/// Compute zero-padded matrix (in the arena) and fill zero-padded boundaries with the adjacent
/// non-zero rows and columns.
pub fn add_matrix_boundary<'a, S>(
    input_matrix: &ArrayBase<S, Ix2>,
    arena: &'a Arena,
) -> ArrayViewMut2<'a, f64>
where
    S: Data<Elem = f64>,
{
    let mut output_matrix = copy_matrix_within_padding(input_matrix, 1, 1, 1, 1, arena);

    for i in 0..output_matrix.ncols() {
        output_matrix.row_mut(0)[i] = output_matrix.row(1)[i];
        output_matrix.row_mut(output_matrix.nrows() - 1)[i] =
            output_matrix.row(output_matrix.nrows() - 2)[i];
    }

    for i in 0..output_matrix.nrows() {
        output_matrix.column_mut(0)[i] = output_matrix.column_mut(1)[i];
        output_matrix.column_mut(output_matrix.ncols() - 1)[i] =
            output_matrix.column(output_matrix.ncols() - 2)[i];
    }
    output_matrix
}

/// Returns a copy of `input_matrix` (in the arena) which is zero-padded by the specified amounts.
pub fn copy_matrix_within_padding<'a, S>(
    input_matrix: &ArrayBase<S, Ix2>,
    row_prepad_amount: usize,
    row_postpad_amount: usize,
    col_prepad_amount: usize,
    col_postpad_amount: usize,
    arena: &'a Arena,
) -> ArrayViewMut2<'a, f64>
where
    S: Data<Elem = f64>,
{
    let mut output_matrix = arena_mat(
        arena,
        row_prepad_amount + input_matrix.nrows() + row_postpad_amount,
        col_prepad_amount + input_matrix.ncols() + col_postpad_amount,
    );
    // Arena memory is uninitialised; the original allocated `Array2::zeros`, and `add_matrix_boundary`
    // reads the padding ring (as 0) before overwriting it — so zero the whole matrix first to stay
    // bit-identical.
    output_matrix.fill(0.0);

    for row_i in 0..input_matrix.nrows() {
        for col_i in 0..input_matrix.ncols() {
            output_matrix[(row_i + row_prepad_amount, col_i + col_prepad_amount)] =
                input_matrix[(row_i, col_i)];
        }
    }
    output_matrix
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, Array2, ShapeBuilder};
    use profluens_core::memory::Arena;

    use super::*;

    #[test]
    fn convolve_with_window() {
        let arena = Arena::default();
        let w = vec![
            0.0113033910173052,
            0.0838251475442633,
            0.0113033910173052,
            0.0838251475442633,
            0.619485845753726,
            0.0838251475442633,
            0.0113033910173052,
            0.0838251475442633,
            0.0113033910173052,
        ];
        let window = Array::from_shape_vec((3, 3).f(), w).unwrap();

        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();

        let result = perform_valid_2d_conv_with_boundary(&window, &matrix, &arena);

        let r = vec![
            40.6634, 42.8407, 40.6395, 41.0129, 41.5407, 42.4677, 44.2760, 44.2031, 41.2263,
            42.9752, 41.3784, 41.2656, 42.1388, 43.0366, 42.8042, 42.7613, 42.1817, 42.4590,
            43.2709, 42.9377,
        ];
        let expected_result = Array2::<f64>::from_shape_vec((5, 4).f(), r).unwrap();

        use approx::assert_abs_diff_eq;
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert_abs_diff_eq!(result[(i, j)], expected_result[(i, j)], epsilon = 0.001);
            }
        }
    }

    #[test]
    fn perform_padding() {
        let arena = Arena::default();
        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();
        let result = add_matrix_boundary(&matrix, &arena);

        let mut r = Vec::new();
        for i in 0..result.dim().0 {
            for j in 0..result.dim().1 {
                r.push(result[(i, j)]);
            }
        }

        let expected_result = vec![
            40.0392, 40.0392, 42.6852, 41.0119, 43.0624, 43.0624, 40.0392, 40.0392, 42.6852,
            41.0119, 43.0624, 43.0624, 43.3409, 43.3409, 45.2083, 40.4244, 42.261, 42.261, 39.527,
            39.527, 45.7769, 41.5932, 42.4725, 42.4725, 41.1731, 41.1731, 39.9689, 43.6027,
            43.4258, 43.4258, 41.3591, 41.3591, 43.619, 42.6204, 42.9079, 42.9079, 41.3591,
            41.3591, 43.619, 42.6204, 42.9079, 42.9079,
        ];

        assert_eq!(r, expected_result);
    }

    #[test]
    fn copy_with_zeros() {
        let arena = Arena::default();
        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();
        let result = copy_matrix_within_padding(&matrix, 1, 1, 1, 1, &arena);

        let er = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0392, 42.6852, 41.0119, 43.0624, 0.0, 0.0,
            43.3409, 45.2083, 40.4244, 42.261, 0.0, 0.0, 39.527, 45.7769, 41.5932, 42.4725, 0.0,
            0.0, 41.1731, 39.9689, 43.6027, 43.4258, 0.0, 0.0, 41.3591, 43.619, 42.6204, 42.9079,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // Extracted from cpp, with armadillo using column memory layout.
        let erm = Array::from_shape_vec((7, 6), er).unwrap();

        for (r_elem, erm_elem) in result.iter().zip(&erm) {
            assert_eq!(r_elem, erm_elem);
        }
    }
}
