#version 450
// Owned video color pipeline — fragment shader. The shader is deliberately "dumb":
// all color-science *decisions* (which matrix, which primaries, which EOTF) are made
// on the CPU in `gpu/color.rs` and delivered as pre-computed matrices + mode flags
// in one std140 uniform block. This stage only *applies* them, with each transform
// citing its standard.
//
// Clean-room (standing repo rule): derived from the ITU-R / SMPTE / H.273 texts,
// no libplacebo / mpv source consulted.
//
// Resource bindings follow the SDL_GPU SPIR-V convention (see SDL_CreateGPUShader):
//   set 2 = fragment sampled textures (Y, Cb, Cr)
//   set 3 = fragment uniform buffers   (the color block)
//
// Stages, in order:
//   1. sample planes                → coded YCbCr (or Y + flat chroma for gray8)
//   2. YCbCr → R'G'B'               → uniform 3×4 affine (BT.601-7 / .709-6 / .2020-2)
//   3. EOTF / linearize             → BT.1886 | sRGB | PQ(BT.2100) | HLG(BT.2100)
//   4. tone-map slot                → SDR: identity; HDR: normalize 203 nits +
//                                     Reinhard (Reinhard et al. 2002), v1 placeholder
//   5. primaries matrix             → uniform 3×3, src primaries → BT.709 (sRGB gamut)
//   6. sRGB OETF                    → encode for the (UNORM) swapchain

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// set 2: the plane textures. Cb/Cr are obtained per chroma_mode:
//   CHROMA_PLANAR (i420):      Cb = tex_u.r,  Cr = tex_v.r  (two R8 textures)
//   CHROMA_INTERLEAVED (nv12): Cb = tex_u.r,  Cr = tex_u.g  (one R8G8 texture; tex_v
//                              is bound but unused)
//   CHROMA_FLAT (gray8):       Cb = Cr = 0.5 (achromatic); tex_u/tex_v unused.
layout(set = 2, binding = 0) uniform sampler2D tex_y;
layout(set = 2, binding = 1) uniform sampler2D tex_u;
layout(set = 2, binding = 2) uniform sampler2D tex_v;

// set 3: the color uniform block. std140 — must match `FragUniforms` byte layout in
// gpu/color.rs. We declare it field-by-field as explicit vec4 rows so the std140
// offsets are unambiguous: a `mat3`/`mat3x4` would pad to 3 vec4 rows anyway, and
// spelling the rows out keeps the CPU `[[f32;4];3]` and the GLSL side byte-identical.
layout(set = 3, binding = 0, std140) uniform ColorBlock {
    vec4 ycbcr0;   // row 0 of the 3×4 YCbCr→R'G'B' affine: [Y, Cb, Cr, offset]
    vec4 ycbcr1;   // row 1
    vec4 ycbcr2;   // row 2
    vec4 prim0;    // row 0 of the 3×3 primaries matrix (w unused)
    vec4 prim1;    // row 1
    vec4 prim2;    // row 2
    uint transfer;   // EOTF selector (see TR_* below)
    uint tonemap;    // 0 = identity (SDR), 1 = Reinhard v1 (HDR input)
    uint chroma_mode;// 0 = sample U/V planes, 1 = flat achromatic (gray8)
    uint _pad;
};

// EOTF selectors — must match Transfer::shader_code() in gpu/color.rs.
#define TR_BT709   0u
#define TR_SRGB    1u
#define TR_PQ      2u
#define TR_HLG     3u
#define TR_LINEAR  4u

// Chroma mode selectors — must match ChromaMode::code() in gpu/color.rs.
#define CHROMA_PLANAR      0u
#define CHROMA_FLAT        1u
#define CHROMA_INTERLEAVED 2u

// --- EOTFs: encoded R'G'B' (0..1) → linear light -------------------------------

// BT.1886: the reference EOTF for BT.709/BT.601 content on a display — a pure 2.4
// power law (ITU-R BT.1886, with Lb=0 so the affine terms vanish to L = V^2.4).
vec3 eotf_bt1886(vec3 v) {
    return pow(max(v, vec3(0.0)), vec3(2.4));
}

// sRGB / IEC 61966-2-1 piecewise EOTF.
vec3 eotf_srgb(vec3 v) {
    bvec3 lo = lessThanEqual(v, vec3(0.04045));
    vec3 a = v / 12.92;
    vec3 b = pow((v + 0.055) / 1.055, vec3(2.4));
    return mix(b, a, vec3(lo));
}

// SMPTE ST 2084 / BT.2100 PQ EOTF (BT.2100-2 Table 4). Maps the [0,1] PQ signal to
// normalized linear luminance where 1.0 == 10 000 cd/m². Constants per the standard.
vec3 eotf_pq(vec3 e) {
    const float m1 = 0.1593017578125;   // 2610/16384
    const float m2 = 78.84375;          // 2523/32
    const float c1 = 0.8359375;         // 3424/4096
    const float c2 = 18.8515625;        // 2413/128
    const float c3 = 18.6875;           // 2392/128
    vec3 ep = pow(max(e, vec3(0.0)), vec3(1.0 / m2));
    vec3 num = max(ep - c1, vec3(0.0));
    vec3 den = c2 - c3 * ep;
    return pow(num / den, vec3(1.0 / m1)); // 0..1 == 0..10000 cd/m²
}

// ARIB STD-B67 / BT.2100 HLG inverse-OETF (OETF⁻¹): the [0,1] HLG signal → scene
// linear (BT.2100-2 Table 5). Returns "scene" light 0..1 (the OOTF/system gamma is
// folded into the tone-map normalization for this v1 placeholder).
vec3 eotf_hlg(vec3 e) {
    const float a = 0.17883277;
    const float b = 0.28466892;         // 1 - 4a
    const float c = 0.55991073;         // 0.5 - a*ln(4a)
    bvec3 lo = lessThanEqual(e, vec3(0.5));
    vec3 low = (e * e) / 3.0;
    vec3 high = (exp((e - c) / a) + b) / 12.0;
    return mix(high, low, vec3(lo));
}

vec3 linearize(vec3 rgb, uint tr) {
    if (tr == TR_SRGB)   return eotf_srgb(rgb);
    if (tr == TR_PQ)     return eotf_pq(rgb);
    if (tr == TR_HLG)    return eotf_hlg(rgb);
    if (tr == TR_LINEAR) return max(rgb, vec3(0.0));
    return eotf_bt1886(rgb); // TR_BT709 default
}

// --- Tone-map slot (v1) --------------------------------------------------------
// SDR: identity. HDR (PQ/HLG) input arrives in absolute-ish luminance; we normalize
// by the 203 cd/m² reference white (ITU-R BT.2408 — the reference level for HDR
// graphics white / diffuse white), then apply a maxRGB Reinhard compressor
// (Reinhard, Stark, Shirley & Ferwerda 2002, "Photographic Tone Reproduction for
// Digital Images", the L/(1+L) global operator) to bring highlights into [0,1].
// This is explicitly a v1 placeholder; the named follow-up is the BT.2390 EETF for a
// hue/saturation-preserving, display-referred roll-off.
vec3 apply_tonemap(vec3 lin, uint mode, uint tr) {
    if (mode == 0u) return lin; // SDR identity

    // PQ EOTF output is normalized to 10000 nits at 1.0; scale to nits then to the
    // 203-nit reference. HLG's inverse-OETF gives scene 0..1; treat 1.0 as ~1000
    // nits peak (BT.2100 nominal) before the same 203-nit normalization.
    vec3 nits = (tr == TR_PQ) ? (lin * 10000.0) : (lin * 1000.0);
    vec3 v = nits / 203.0; // reference white → 1.0 (BT.2408)
    // maxRGB Reinhard: compress on the channel max to preserve hue better than a
    // per-channel operator (Reinhard et al. 2002 §3, adapted to maxRGB).
    float m = max(max(v.r, v.g), v.b);
    float scale = 1.0 / (1.0 + m);
    return v * scale;
}

void main() {
    // 1. sample planes → coded YCbCr. Linear samplers give bilinear chroma upsample
    //    (chroma siting treated as CENTER — MPEG-2 style left-siting is a noted
    //    follow-up; H.273 ChromaSampleLocType is not yet plumbed).
    float y = texture(tex_y, v_uv).r;
    float cb, cr;
    if (chroma_mode == CHROMA_FLAT) {
        cb = 0.5; cr = 0.5; // gray8: flat achromatic axis (Cb=Cr=0.5 coded)
    } else if (chroma_mode == CHROMA_INTERLEAVED) {
        vec2 uv = texture(tex_u, v_uv).rg; // nv12: Cb=.r, Cr=.g from one R8G8 texture
        cb = uv.r; cr = uv.g;
    } else {
        cb = texture(tex_u, v_uv).r; // i420
        cr = texture(tex_v, v_uv).r;
    }

    // 2. YCbCr → R'G'B' via the uniform 3×4 affine (offset in the .w column).
    vec4 ycbcr_in = vec4(y, cb, cr, 1.0);
    vec3 rp = vec3(dot(ycbcr0, ycbcr_in), dot(ycbcr1, ycbcr_in), dot(ycbcr2, ycbcr_in));

    // 3. EOTF: encoded R'G'B' → linear light in the source primaries.
    vec3 lin = linearize(rp, transfer);

    // 4. tone-map slot (SDR identity; HDR Reinhard v1).
    lin = apply_tonemap(lin, tonemap, transfer);

    // 5. primaries: source-gamut linear → BT.709 (sRGB) linear via the uniform 3×3.
    //    prim0/1/2 are the ROWS of the 3×3, so the product is row·lin per output.
    vec3 rgb709 = vec3(
        dot(prim0.xyz, lin),
        dot(prim1.xyz, lin),
        dot(prim2.xyz, lin)
    );
    rgb709 = clamp(rgb709, vec3(0.0), vec3(1.0));

    // 6. encode sRGB OETF for the UNORM swapchain (IEC 61966-2-1). If the swapchain
    //    were an _SRGB format the hardware would do this; we target UNORM and encode
    //    ourselves so the classic and GPU paths look identical.
    bvec3 lo = lessThanEqual(rgb709, vec3(0.0031308));
    vec3 enc_a = rgb709 * 12.92;
    vec3 enc_b = 1.055 * pow(rgb709, vec3(1.0 / 2.4)) - 0.055;
    vec3 enc = mix(enc_b, enc_a, vec3(lo));

    o_color = vec4(enc, 1.0);
}
