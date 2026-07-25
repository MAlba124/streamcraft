//! Color science for the owned GPU render pipeline — all of it in Rust, so the
//! shaders stay dumb (they receive pre-computed matrices and mode flags in a
//! uniform block). SDL's fixed-function `SDL_UpdateYUVTexture` path let SDL pick
//! the YCbCr matrix and ignore the real colorimetry; here every transform is ours
//! and cites its standard at the point of use.
//!
//! Clean-room rule (standing repo policy): no libplacebo / mpv source was
//! consulted. The math is derived directly from the ITU-R / SMPTE / H.273 texts.
//!
//! Pipeline (matching the fragment shader stages, `shaders/video.frag`):
//!   sample planes  →  YCbCr→R'G'B'  (this module's `ycbcr_to_rgb_matrix`)
//!                  →  EOTF / linearize   (shader, transfer-selected)
//!                  →  tone-map slot      (shader, v1 Reinhard for HDR input)
//!                  →  primaries matrix   (this module's `primaries_matrix`)
//!                  →  sRGB OETF for the swapchain (shader).
//!
//! References:
//! - ITU-R BT.601-7 — SDTV, Kr/Kb and the 16..235 / 16..240 quantisation.
//! - ITU-R BT.709-6 — HDTV, Kr/Kb, primaries, quantisation (§4).
//! - ITU-R BT.2020-2 — UHDTV, Kr/Kb, primaries.
//! - ITU-R BT.2100-2 — HDR (PQ / HLG), used in the shader EOTFs.
//! - ITU-R H.273 — the coded-value ↔ signal conventions and the MatrixCoefficients /
//!   ColourPrimaries / TransferCharacteristics registries; also the "unspecified ⇒
//!   default" convention this module applies when a field is absent.
//! - SMPTE RP 177-1993 — deriving the RGB↔XYZ matrix from chromaticity coordinates
//!   (`npm` normalised primary matrix).

/// YCbCr → R'G'B' matrix identity (which luma coefficients to use).
///
/// Enumerated by the H.273 MatrixCoefficients they correspond to; `Identity`
/// (H.273 MatrixCoefficients = 0, "GBR") means the planes are already R'G'B' and
/// no YCbCr rotation is applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Matrix {
    /// BT.709-6 (H.273 code 1): Kr=0.2126, Kb=0.0722.
    Bt709,
    /// BT.601-7 (H.273 code 5/6): Kr=0.299, Kb=0.114.
    Bt601,
    /// BT.2020-2 non-constant luminance (H.273 code 9): Kr=0.2627, Kb=0.0593.
    Bt2020,
    /// GBR / RGB passthrough (H.273 code 0): the planes are already R'G'B'.
    Identity,
}

/// Quantisation range of the coded samples (H.273 VideoFullRangeFlag).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Range {
    /// "Limited"/"studio": 8-bit Y in 16..235, C in 16..240 (BT.601 §2.5.3,
    /// BT.709 §4). Also called narrow range. VideoFullRangeFlag = 0.
    Limited,
    /// "Full"/"PC": Y and C use the whole 0..255 code space. VideoFullRangeFlag = 1.
    Full,
}

/// Opto-electronic / electro-optical transfer characteristic of the R'G'B' signal
/// (H.273 TransferCharacteristics). The shader selects the linearising EOTF from
/// this; the CPU side only needs the tag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transfer {
    /// BT.709 OETF (H.273 code 1). We linearise with the BT.1886 EOTF (pure 2.4
    /// power), the reference display transfer for BT.709/BT.601 content.
    Bt709,
    /// sRGB / IEC 61966-2-1 piecewise transfer (H.273 code 13).
    Srgb,
    /// SMPTE ST 2084 / BT.2100 PQ (H.273 code 16).
    Pq,
    /// ARIB STD-B67 / BT.2100 HLG (H.273 code 18).
    Hlg,
    /// Linear light already (H.273 code 8) — the EOTF is identity.
    Linear,
}

/// Colour primaries — the chromaticity triangle the R'G'B' components live in
/// (H.273 ColourPrimaries). Converting between two primary sets is an RGB→XYZ→RGB
/// change of basis (see [`primaries_matrix`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Primaries {
    /// BT.709-6 / sRGB primaries (H.273 code 1).
    Bt709,
    /// BT.601-7 625-line ("EBU 3213") primaries (H.273 code 5). We use the 625-line
    /// set as the canonical "bt601" primaries.
    Bt601,
    /// BT.2020-2 primaries (H.273 code 9).
    Bt2020,
    /// DCI-P3 (SMPTE RP 431-2) primaries (H.273 code 11/12 use the display P3 white
    /// D65; here we use the DCI theatrical white as tagged — see the table below).
    DciP3,
}

/// The four negotiated colorimetry tags, with the defaults rule applied
/// ([`Colorimetry::resolve`]). This is what the sink hands the renderer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Colorimetry {
    pub matrix: Matrix,
    pub range: Range,
    pub transfer: Transfer,
    pub primaries: Primaries,
}

impl Matrix {
    /// Parse the negotiated `matrix` value name. `None` for an unknown name (the
    /// caller then applies the default rule).
    pub fn from_name(s: &str) -> Option<Matrix> {
        Some(match s {
            "bt709" => Matrix::Bt709,
            "bt601" => Matrix::Bt601,
            "bt2020" => Matrix::Bt2020,
            "identity" => Matrix::Identity,
            _ => return None,
        })
    }

    /// The name this tag round-trips to (H.273-ish vocabulary).
    pub fn name(self) -> &'static str {
        match self {
            Matrix::Bt709 => "bt709",
            Matrix::Bt601 => "bt601",
            Matrix::Bt2020 => "bt2020",
            Matrix::Identity => "identity",
        }
    }

    /// Luma coefficients `(Kr, Kb)`; `Kg = 1 - Kr - Kb`. Identity has no luma axis
    /// (the planes are R'G'B'), so we return `None` for it.
    ///
    /// Cited values:
    ///   * BT.601-7 §2.5.1:  Kr = 0.299,  Kb = 0.114.
    ///   * BT.709-6 §3.2:    Kr = 0.2126, Kb = 0.0722.
    ///   * BT.2020-2 Table 4: Kr = 0.2627, Kb = 0.0593.
    fn luma_coeffs(self) -> Option<(f64, f64)> {
        Some(match self {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
            Matrix::Identity => return None,
        })
    }
}

impl Range {
    pub fn from_name(s: &str) -> Option<Range> {
        Some(match s {
            "limited" => Range::Limited,
            "full" => Range::Full,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Range::Limited => "limited",
            Range::Full => "full",
        }
    }
}

impl Transfer {
    pub fn from_name(s: &str) -> Option<Transfer> {
        Some(match s {
            "bt709" => Transfer::Bt709,
            "srgb" => Transfer::Srgb,
            "pq" => Transfer::Pq,
            "hlg" => Transfer::Hlg,
            "linear" => Transfer::Linear,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Transfer::Bt709 => "bt709",
            Transfer::Srgb => "srgb",
            Transfer::Pq => "pq",
            Transfer::Hlg => "hlg",
            Transfer::Linear => "linear",
        }
    }
    /// The integer the shader switches on for the EOTF stage (must match the
    /// `#define TR_*` block in `shaders/video.frag`).
    pub fn shader_code(self) -> u32 {
        match self {
            Transfer::Bt709 => 0,
            Transfer::Srgb => 1,
            Transfer::Pq => 2,
            Transfer::Hlg => 3,
            Transfer::Linear => 4,
        }
    }
    /// Is this an HDR transfer (PQ / HLG)? The tone-map slot only engages for these.
    pub fn is_hdr(self) -> bool {
        matches!(self, Transfer::Pq | Transfer::Hlg)
    }
}

impl Primaries {
    pub fn from_name(s: &str) -> Option<Primaries> {
        Some(match s {
            "bt709" => Primaries::Bt709,
            "bt601" => Primaries::Bt601,
            "bt2020" => Primaries::Bt2020,
            "dci-p3" => Primaries::DciP3,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Primaries::Bt709 => "bt709",
            Primaries::Bt601 => "bt601",
            Primaries::Bt2020 => "bt2020",
            Primaries::DciP3 => "dci-p3",
        }
    }

    /// CIE 1931 chromaticity coordinates `[[Rx,Ry],[Gx,Gy],[Bx,By],[Wx,Wy]]`.
    ///
    /// Cited tables:
    ///   * BT.709-6 §1 (primaries), D65 white.
    ///   * BT.2020-2 Table 3 (primaries), D65 white.
    ///   * BT.601-7 Table 2 (625-line / EBU 3213-E primaries), D65 white.
    ///   * SMPTE RP 431-2 (DCI-P3), theatrical white ~(0.314, 0.351).
    fn chromaticities(self) -> [[f64; 2]; 4] {
        match self {
            Primaries::Bt709 => [
                [0.640, 0.330], // R
                [0.300, 0.600], // G
                [0.150, 0.060], // B
                [0.3127, 0.3290], // D65
            ],
            Primaries::Bt601 => [
                [0.640, 0.330], // R (625-line R matches 709 R)
                [0.290, 0.600], // G
                [0.150, 0.060], // B
                [0.3127, 0.3290], // D65
            ],
            Primaries::Bt2020 => [
                [0.708, 0.292], // R
                [0.170, 0.797], // G
                [0.131, 0.046], // B
                [0.3127, 0.3290], // D65
            ],
            Primaries::DciP3 => [
                [0.680, 0.320], // R
                [0.265, 0.690], // G
                [0.150, 0.060], // B
                [0.314, 0.351], // DCI theatrical white
            ],
        }
    }
}

impl Colorimetry {
    /// Resolve the four (possibly-absent) negotiated names into a concrete
    /// colorimetry, applying the H.273 "unspecified ⇒ default" convention:
    ///
    /// - matrix: height ≥ 720 ⇒ bt709, else bt601 (the SD/HD split; H.273
    ///   MatrixCoefficients=2 "Unspecified" is resolved by the receiver, and the
    ///   near-universal receiver heuristic is resolution-based).
    /// - range: limited (studio range is the coded-video default).
    /// - transfer: bt709 (matches the default matrix's companion OETF).
    /// - primaries: follow the matrix (bt709⇒bt709, bt601⇒bt601, bt2020⇒bt2020,
    ///   identity⇒bt709).
    ///
    /// A present-but-unrecognised name falls back to the same default as absence.
    pub fn resolve(
        height: usize,
        matrix: Option<&str>,
        range: Option<&str>,
        transfer: Option<&str>,
        primaries: Option<&str>,
    ) -> Colorimetry {
        let matrix = matrix.and_then(Matrix::from_name).unwrap_or({
            if height >= 720 {
                Matrix::Bt709
            } else {
                Matrix::Bt601
            }
        });
        let range = range.and_then(Range::from_name).unwrap_or(Range::Limited);
        let transfer = transfer.and_then(Transfer::from_name).unwrap_or(Transfer::Bt709);
        let primaries = primaries.and_then(Primaries::from_name).unwrap_or(match matrix {
            Matrix::Bt709 | Matrix::Identity => Primaries::Bt709,
            Matrix::Bt601 => Primaries::Bt601,
            Matrix::Bt2020 => Primaries::Bt2020,
        });
        Colorimetry { matrix, range, transfer, primaries }
    }
}

/// The YCbCr → R'G'B' affine transform as a 3×4 matrix (last column is the offset),
/// so the shader computes `rgb = M * vec4(y, cb, cr, 1)`.
///
/// Construction (BT.601 §2.5 / BT.709 §4, generalised over `(Kr, Kb)`):
///   R' = Y' + 2(1-Kr)·Cr
///   B' = Y' + 2(1-Kb)·Cb
///   G' = Y' - (2(1-Kr)Kr/Kg)·Cr - (2(1-Kb)Kb/Kg)·Cb,  Kg = 1-Kr-Kb
/// where (Y', Cb, Cr) are the *normalised* signal: Y' ∈ [0,1], Cb,Cr ∈ [-0.5, 0.5].
///
/// For **limited** 8-bit range the coded samples must first be expanded to that
/// normalised signal (BT.601 §2.5.3 / BT.709 §4):
///   Y'   = (Y_code·255 - 16) / (235-16)     = (Y_code - 16/255) · (255/219)
///   Cb/Cr= (C_code·255 - 128)/(240-16)      = (C_code - 128/255) · (255/224)
/// (`_code` is the normalised 0..1 texture sample.) The famous BT.709 Cr→R
/// coefficient 1.5748 = 2(1-Kr) applies to the *full-range* signal; the limited-range
/// matrix folds the 255/224 chroma gain in, giving ≈1.7927.
///
/// The 255/219 and 255/224 gains are the 8-bit forms; higher bit depths share the
/// same 16·2^(n-8)..235·2^(n-8) structure but the sink presents 8-bit textures, so
/// the 8-bit expansion is exact here. Identity (GBR) returns the plain plane→RGB
/// permutation with the range expansion applied to all three as luma.
pub fn ycbcr_to_rgb_matrix(matrix: Matrix, range: Range) -> [[f32; 4]; 3] {
    // Range-expansion gains + offsets on the *coded* (0..1) sample.
    // full: identity. limited: the 8-bit studio expansion above.
    let (y_gain, y_off, c_gain) = match range {
        Range::Full => (1.0_f64, 0.0_f64, 1.0_f64),
        // 255/219 for luma, 255/224 for chroma (8-bit limited range).
        Range::Limited => (255.0 / 219.0, 16.0 / 255.0, 255.0 / 224.0),
    };
    // The 8-bit coded chroma NEUTRAL is code 128 (both full and limited range;
    // BT.601 §2.5.3 / BT.709 §4 quantise Cb=Cr=0 to code 128). In the 0..1 texture
    // sample that is 128/255, NOT 0.5 (=127.5/255) — using 128/255 keeps a mid-grey
    // exactly achromatic (R=G=B) instead of leaving a half-code chroma residual.
    let c_center = 128.0 / 255.0;

    let Some((kr, kb)) = matrix.luma_coeffs() else {
        // Identity / GBR: planes are R'G'B'. In our plane layout the "Y" plane is
        // G', the "Cb" plane is B', the "Cr" plane is R' (H.273 GBR ordering); but
        // the sink only ever tags identity for gray/experimental paths, so we map
        // Y→all channels with luma range expansion, matching a monochrome intent.
        let g = y_gain as f32;
        let o = -(y_gain * y_off) as f32;
        return [[g, 0.0, 0.0, o], [g, 0.0, 0.0, o], [g, 0.0, 0.0, o]];
    };
    let kg = 1.0 - kr - kb;

    // Coefficients on the *normalised* signal (Y'∈[0,1], Cb,Cr∈[-0.5,0.5]).
    let cr_r = 2.0 * (1.0 - kr); // Cr → R
    let cb_b = 2.0 * (1.0 - kb); // Cb → B
    let cr_g = -2.0 * (1.0 - kr) * kr / kg; // Cr → G
    let cb_g = -2.0 * (1.0 - kb) * kb / kg; // Cb → G

    // Fold the coded→normalised expansion into a single affine map on the *coded*
    // (0..1) sample. Let y = y_gain·(Y_code - y_off), c = c_gain·(C_code - c_center).
    //   R = y + cr_r·cr
    //     = y_gain·Y_code - y_gain·y_off + cr_r·c_gain·C_cr - cr_r·c_gain·c_center
    // and similarly for G, B. Collect the constant terms into the offset column.
    let yg = y_gain;
    let cg = c_gain;
    let y_const = -yg * y_off;
    let cc = c_center;
    // Row = [coeff on Y_code, on Cb_code, on Cr_code, offset]
    let r = [yg, 0.0, cr_r * cg, y_const + (cr_r * cg) * -cc];
    let g = [yg, cb_g * cg, cr_g * cg, y_const + (cb_g * cg + cr_g * cg) * -cc];
    let b = [yg, cb_b * cg, 0.0, y_const + (cb_b * cg) * -cc];

    [to_f32_4(r), to_f32_4(g), to_f32_4(b)]
}

fn to_f32_4(r: [f64; 4]) -> [f32; 4] {
    [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
}

/// 3×3 matrix multiply (`a·b`), row-major `f64` for construction precision.
fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for (i, oi) in o.iter_mut().enumerate() {
        for (j, oij) in oi.iter_mut().enumerate() {
            *oij = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// Inverse of a 3×3 matrix via the adjugate / determinant (closed form — the
/// matrices here are always well-conditioned RGB↔XYZ bases).
fn mat3_inv(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;
    [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

/// The RGB→XYZ "normalised primary matrix" (`npm`) for a set of primaries, per
/// SMPTE RP 177-1993 §3. Given primaries chromaticities (xr,yr),(xg,yg),(xb,yb) and
/// white (xw,yw):
///
/// - form `X_i = x_i/y_i`, `Y_i = 1`, `Z_i = (1-x_i-y_i)/y_i` for each primary i,
/// - stack them as columns into `P` (3×3),
/// - find the per-primary luminance scalars `S = P⁻¹ · W`, where `W` is the white
///   point's XYZ at Y=1,
/// - the npm is `P` with each column scaled by the corresponding `S`.
///
/// The result maps linear-light R,G,B (in that primary set) to CIE XYZ.
fn rgb_to_xyz(p: Primaries) -> [[f64; 3]; 3] {
    let ch = p.chromaticities();
    let prim = |xy: [f64; 2]| {
        let (x, y) = (xy[0], xy[1]);
        [x / y, 1.0, (1.0 - x - y) / y]
    };
    let xr = prim(ch[0]);
    let xg = prim(ch[1]);
    let xb = prim(ch[2]);
    // P with primaries as columns.
    let p_mat = [
        [xr[0], xg[0], xb[0]],
        [xr[1], xg[1], xb[1]],
        [xr[2], xg[2], xb[2]],
    ];
    // White point XYZ at Y=1.
    let (wx, wy) = (ch[3][0], ch[3][1]);
    let w = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    // S = P⁻¹ · W.
    let p_inv = mat3_inv(p_mat);
    let s = [
        p_inv[0][0] * w[0] + p_inv[0][1] * w[1] + p_inv[0][2] * w[2],
        p_inv[1][0] * w[0] + p_inv[1][1] * w[1] + p_inv[1][2] * w[2],
        p_inv[2][0] * w[0] + p_inv[2][1] * w[1] + p_inv[2][2] * w[2],
    ];
    // Scale each column of P by S to get the npm.
    [
        [xr[0] * s[0], xg[0] * s[1], xb[0] * s[2]],
        [xr[1] * s[0], xg[1] * s[1], xb[1] * s[2]],
        [xr[2] * s[0], xg[2] * s[1], xb[2] * s[2]],
    ]
}

/// The 3×3 linear-light primaries conversion `src → dst` (e.g. bt2020 → bt709):
/// `M = XYZ_to_RGB(dst) · RGB_to_XYZ(src)`. Both primary sets share the D65 white
/// (except DCI-P3 theatrical, which carries its own white — the change of basis
/// then also adapts the white implicitly through the two npm's, i.e. this is an
/// absolute-XYZ conversion, no separate chromatic-adaptation step; a Bradford CAT
/// is the named follow-up if a white-preserving relative conversion is wanted).
///
/// When `src == dst` this is the identity within float precision.
pub fn primaries_matrix(src: Primaries, dst: Primaries) -> [[f32; 3]; 3] {
    if src == dst {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let src_to_xyz = rgb_to_xyz(src);
    let xyz_to_dst = mat3_inv(rgb_to_xyz(dst));
    let m = mat3_mul(xyz_to_dst, src_to_xyz);
    [
        [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
        [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
        [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
    ]
}

// --- Uniform block ---------------------------------------------------------------
//
// The single fragment-uniform block shared with `shaders/video.frag`. std140 layout
// (SDL_GPU / SPIR-V uniform buffers follow std140), documented byte-for-byte so the
// GLSL `layout(std140)` block and this `#[repr(C)]` struct stay in lockstep.
//
// std140 rules used here:
//   * a `mat3` is laid out as 3 column vectors each aligned/padded to 16 bytes
//     (a `vec4` stride) — so a `mat3` occupies 48 bytes, NOT 36. We therefore store
//     each matrix as an explicit `[[f32; 4]; 3]` (three padded rows) to make the
//     padding visible and correct.
//   * the YCbCr matrix is 3×4 (a real 4th column: the offset) → also 3 rows × 16 B.
//   * scalars pack into a trailing 16-byte slot.
//
// Byte layout (offsets in bytes):
//   0   : ycbcr    [[f32;4];3]  (48)  YCbCr→R'G'B' affine, row-major, col 3 = offset
//   48  : prim     [[f32;4];3]  (48)  primaries src→dst 3×3, padded to 3× vec4
//   96  : transfer u32          (4)   EOTF selector (Transfer::shader_code)
//   100 : tonemap  u32          (4)   0 = identity (SDR), 1 = Reinhard (HDR)
//   104 : chroma_mode u32       (4)   0 = sample chroma planes, 1 = flat-grey (gray8)
//   108 : _pad     u32          (4)   → total 112 bytes, a multiple of 16.
//
// The struct is `#[repr(C)]` with the same field order; Rust's C layout gives the
// same offsets because every field is either a `[[f32;4];3]` (16-aligned, 48 B) or a
// `u32`, and the four trailing `u32`s pack into one 16-byte tail. Total = 112 B.

/// The fragment-shader uniform block. Built by [`Colorimetry`] +
/// [`RenderParams`]. `#[repr(C)]`, std140-compatible (see the module note).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FragUniforms {
    /// YCbCr→R'G'B' affine (3 rows, col 3 = offset), padded to std140 mat rows.
    pub ycbcr: [[f32; 4]; 3],
    /// Primaries src→dst 3×3, padded to std140 mat rows (col 3 unused = 0).
    pub prim: [[f32; 4]; 3],
    /// EOTF selector (`Transfer::shader_code`).
    pub transfer: u32,
    /// Tone-map selector: 0 = identity (SDR), 1 = Reinhard v1 (HDR input).
    pub tonemap: u32,
    /// Chroma sampling mode: 0 = sample U/V planes, 1 = flat achromatic (gray8).
    pub chroma_mode: u32,
    /// Padding to round the block to 112 bytes (multiple of 16 for std140).
    pub _pad: u32,
}

// Compile-time assertion that the block is exactly 112 bytes (std140 tail-aligned).
const _: () = assert!(core::mem::size_of::<FragUniforms>() == 112);

/// Chroma sampling mode — how the fragment shader obtains Cb/Cr.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChromaMode {
    /// i420: sample Cb from the U texture (`.r`) and Cr from the V texture (`.r`).
    Planar,
    /// nv12: sample Cb (`.r`) and Cr (`.g`) from one interleaved R8G8 texture (bound
    /// as the U slot; the V slot is unused).
    Interleaved,
    /// gray8: no chroma planes — force Cb=Cr=0.5 (achromatic axis) in-shader.
    Flat,
}

impl ChromaMode {
    /// The integer the shader switches on (must match `CHROMA_*` in video.frag).
    fn code(self) -> u32 {
        match self {
            ChromaMode::Planar => 0,
            ChromaMode::Interleaved => 2,
            ChromaMode::Flat => 1,
        }
    }
}

impl FragUniforms {
    /// Build the uniform block for a given colorimetry and chroma mode. The
    /// destination primaries are always BT.709 (the sRGB swapchain's gamut); the
    /// primaries matrix therefore converts `colorimetry.primaries → bt709`.
    pub fn build(c: Colorimetry, chroma: ChromaMode) -> FragUniforms {
        let ycbcr = ycbcr_to_rgb_matrix(c.matrix, c.range);
        let prim3 = primaries_matrix(c.primaries, Primaries::Bt709);
        // Pad the 3×3 primaries matrix to std140 mat-row layout (col 3 = 0).
        let prim = [
            [prim3[0][0], prim3[0][1], prim3[0][2], 0.0],
            [prim3[1][0], prim3[1][1], prim3[1][2], 0.0],
            [prim3[2][0], prim3[2][1], prim3[2][2], 0.0],
        ];
        FragUniforms {
            ycbcr,
            prim,
            transfer: c.transfer.shader_code(),
            // Tone-map only for HDR (PQ/HLG) inputs; SDR stays identity.
            tonemap: if c.transfer.is_hdr() { 1 } else { 0 },
            chroma_mode: chroma.code(),
            _pad: 0,
        }
    }

    /// The block as a byte slice for `SDL_PushGPUFragmentUniformData`.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `FragUniforms` is `#[repr(C)]` of `f32`/`u32` — plain old data with
        // no padding holes that carry meaning (the trailing `_pad` is initialised),
        // so reading its bytes is well-defined.
        unsafe {
            core::slice::from_raw_parts(
                (self as *const FragUniforms) as *const u8,
                core::mem::size_of::<FragUniforms>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    #[test]
    fn name_round_trips() {
        for m in [Matrix::Bt709, Matrix::Bt601, Matrix::Bt2020, Matrix::Identity] {
            assert_eq!(Matrix::from_name(m.name()), Some(m));
        }
        for r in [Range::Limited, Range::Full] {
            assert_eq!(Range::from_name(r.name()), Some(r));
        }
        for t in [Transfer::Bt709, Transfer::Srgb, Transfer::Pq, Transfer::Hlg, Transfer::Linear] {
            assert_eq!(Transfer::from_name(t.name()), Some(t));
        }
        for p in [Primaries::Bt709, Primaries::Bt601, Primaries::Bt2020, Primaries::DciP3] {
            assert_eq!(Primaries::from_name(p.name()), Some(p));
        }
        assert_eq!(Matrix::from_name("nonsense"), None);
    }

    #[test]
    fn default_selection_rule() {
        // Unannounced, HD → bt709 matrix, bt709 primaries, limited, bt709 transfer.
        let hd = Colorimetry::resolve(1080, None, None, None, None);
        assert_eq!(hd.matrix, Matrix::Bt709);
        assert_eq!(hd.primaries, Primaries::Bt709);
        assert_eq!(hd.range, Range::Limited);
        assert_eq!(hd.transfer, Transfer::Bt709);
        // Unannounced, SD (<720) → bt601 matrix, bt601 primaries.
        let sd = Colorimetry::resolve(480, None, None, None, None);
        assert_eq!(sd.matrix, Matrix::Bt601);
        assert_eq!(sd.primaries, Primaries::Bt601);
        // Exactly 720 counts as HD.
        assert_eq!(Colorimetry::resolve(720, None, None, None, None).matrix, Matrix::Bt709);
        // Explicit tags win; primaries follow bt2020 matrix when absent.
        let uhd = Colorimetry::resolve(2160, Some("bt2020"), Some("full"), Some("pq"), None);
        assert_eq!(uhd.matrix, Matrix::Bt2020);
        assert_eq!(uhd.primaries, Primaries::Bt2020);
        assert_eq!(uhd.range, Range::Full);
        assert_eq!(uhd.transfer, Transfer::Pq);
        // Present-but-unknown falls back to the default.
        assert_eq!(Colorimetry::resolve(1080, Some("bogus"), None, None, None).matrix, Matrix::Bt709);
    }

    #[test]
    fn bt709_limited_matrix_goldens() {
        // Derive expected values from the spec math independently of the code.
        let m = ycbcr_to_rgb_matrix(Matrix::Bt709, Range::Limited);
        // Luma coefficient (Y_code → any channel) = 255/219.
        approx(m[0][0], 255.0 / 219.0, 1e-3, "Y gain");
        approx(m[1][0], 255.0 / 219.0, 1e-3, "Y gain G");
        approx(m[2][0], 255.0 / 219.0, 1e-3, "Y gain B");
        // Cr → R on the *normalised* signal is 2(1-Kr)=1.5748; on the *coded* sample
        // it is 1.5748 · (255/224).
        let cr_r_full = 2.0 * (1.0 - 0.2126);
        approx(cr_r_full as f32, 1.5748, 1e-3, "Cr→R full-range coeff (spec 1.5748)");
        approx(m[0][2], (cr_r_full * 255.0 / 224.0) as f32, 1e-3, "Cr→R coded");
        // Cb → R is zero.
        approx(m[0][1], 0.0, 1e-6, "Cb→R");
        // Cr → B is zero.
        approx(m[2][2], 0.0, 1e-6, "Cr→B");
    }

    /// Apply the 3×4 affine to a coded (0..1) YCbCr sample.
    fn apply(m: &[[f32; 4]; 3], y: f32, cb: f32, cr: f32) -> [f32; 3] {
        [
            m[0][0] * y + m[0][1] * cb + m[0][2] * cr + m[0][3],
            m[1][0] * y + m[1][1] * cb + m[1][2] * cr + m[1][3],
            m[2][0] * y + m[2][1] * cb + m[2][2] * cr + m[2][3],
        ]
    }

    #[test]
    fn bt709_limited_known_points() {
        let m = ycbcr_to_rgb_matrix(Matrix::Bt709, Range::Limited);
        // Studio white: coded (235,128,128)/255 → R'G'B' ≈ (1,1,1).
        let white = apply(&m, 235.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
        for c in white {
            approx(c, 1.0, 2.0 / 255.0, "white");
        }
        // Studio black: (16,128,128) → (0,0,0).
        let black = apply(&m, 16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
        for c in black {
            approx(c, 0.0, 2.0 / 255.0, "black");
        }
        // Mid grey Y=126 (about middle of 16..235) → equal R'=G'=B', all in range.
        let grey = apply(&m, 126.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
        approx(grey[0], grey[1], 1e-4, "grey R=G");
        approx(grey[1], grey[2], 1e-4, "grey G=B");
        assert!(grey[0] > 0.4 && grey[0] < 0.6, "mid grey in range: {}", grey[0]);
    }

    #[test]
    fn full_vs_limited_differ() {
        let lim = ycbcr_to_rgb_matrix(Matrix::Bt709, Range::Limited);
        let full = ycbcr_to_rgb_matrix(Matrix::Bt709, Range::Full);
        // Full-range luma gain is exactly 1; limited is 255/219 > 1.
        approx(full[0][0], 1.0, 1e-6, "full Y gain");
        assert!(lim[0][0] > full[0][0], "limited expands luma");
        // Full-range Cr→R equals the spec 1.5748 with no chroma gain.
        approx(full[0][2], 1.5748, 1e-3, "full Cr→R");
    }

    #[test]
    fn bt601_coeffs_distinct_from_709() {
        let m601 = ycbcr_to_rgb_matrix(Matrix::Bt601, Range::Limited);
        // BT.601 Cr→R full-range coeff = 2(1-0.299) = 1.402.
        let cr_r = 2.0 * (1.0 - 0.299);
        approx(cr_r as f32, 1.402, 1e-3, "601 Cr→R full");
        approx(m601[0][2], (cr_r * 255.0 / 224.0) as f32, 1e-3, "601 Cr→R coded");
    }

    #[test]
    fn primaries_round_trip_identity() {
        // bt709 → bt2020 → bt709 must be ≈ identity.
        let a = primaries_matrix(Primaries::Bt709, Primaries::Bt2020);
        let b = primaries_matrix(Primaries::Bt2020, Primaries::Bt709);
        // Compose in f64 precision via the public 3×3 (cast up).
        let compose = |x: [[f32; 3]; 3], y: [[f32; 3]; 3]| {
            let mut o = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    o[i][j] = x[i][0] * y[0][j] + x[i][1] * y[1][j] + x[i][2] * y[2][j];
                }
            }
            o
        };
        let id = compose(b, a);
        #[allow(clippy::needless_range_loop)] // 2D matrix index reads clearer as (i,j)
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                approx(id[i][j], expect, 1e-3, "primaries round-trip");
            }
        }
    }

    #[test]
    fn primaries_same_is_identity() {
        let m = primaries_matrix(Primaries::Bt709, Primaries::Bt709);
        #[allow(clippy::needless_range_loop)] // 2D matrix index reads clearer as (i,j)
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                approx(m[i][j], expect, 1e-6, "same-primaries identity");
            }
        }
    }

    #[test]
    fn bt2020_to_709_known_sign_pattern() {
        // Widening bt2020 → narrower bt709: the diagonal is > 1 (bt2020 red is more
        // saturated, so mapping into bt709 amplifies R) and off-diagonals negative.
        let m = primaries_matrix(Primaries::Bt2020, Primaries::Bt709);
        assert!(m[0][0] > 1.0, "bt2020→709 R diag >1: {}", m[0][0]);
        assert!(m[0][1] < 0.0 && m[0][2] < 0.0, "off-diagonals negative");
        // Rows sum to ~1 (D65 white maps to D65 white → equal RGB preserved).
        for row in m.iter() {
            approx(row[0] + row[1] + row[2], 1.0, 1e-3, "row sums to 1 (white preserved)");
        }
    }

    #[test]
    fn uniform_block_is_std140_sized() {
        assert_eq!(core::mem::size_of::<FragUniforms>(), 112);
        assert_eq!(core::mem::size_of::<FragUniforms>() % 16, 0);
        let u = FragUniforms::build(
            Colorimetry::resolve(1080, None, None, None, None),
            ChromaMode::Planar,
        );
        assert_eq!(u.as_bytes().len(), 112);
        assert_eq!(u.chroma_mode, 0);
        assert_eq!(u.tonemap, 0); // SDR
        // Flat chroma flips the mode; PQ engages tone-map.
        let u2 = FragUniforms::build(
            Colorimetry::resolve(2160, Some("bt2020"), None, Some("pq"), None),
            ChromaMode::Flat,
        );
        assert_eq!(u2.chroma_mode, 1);
        assert_eq!(u2.tonemap, 1);
        assert_eq!(u2.transfer, Transfer::Pq.shader_code());
    }
}
