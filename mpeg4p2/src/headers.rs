//! MPEG-4 Visual bitstream headers (ISO/IEC 14496-2 §6.2 syntax, §6.3 semantics):
//! the Visual Object Sequence / Visual Object / Video Object Layer that carry the
//! stream's persistent parameters, and the Video Object Plane header that
//! introduces each coded picture. Only the Advanced Simple / Simple profile
//! subset that the target file (XviD ASP, 1280×544, 4:2:0, half-pel, B-VOPs,
//! packed bitstream, no qpel, no GMC, progressive) actually exercises is parsed;
//! tools outside that subset are detected and flagged (see [`VolHeader`]) so the
//! decoder can warn rather than mis-decode.

use crate::bits::BitReader;

/// Start-code values (ISO/IEC 14496-2 §6.2.1, Table 6-3), the byte following the
/// `00 00 01` prefix.
pub mod startcode {
    pub const VO_SEQ_START: u8 = 0xB0; // visual_object_sequence_start_code
    pub const VO_SEQ_END: u8 = 0xB1;
    pub const VO_START: u8 = 0xB5; // visual_object_start_code
    pub const USER_DATA: u8 = 0xB2;
    pub const GOV_START: u8 = 0xB3; // group_of_vop_start_code
    pub const VOP_START: u8 = 0xB6; // video_object_plane_start_code
    // video_object_start_code    : 0x00..=0x1F
    // video_object_layer_start_code: 0x20..=0x2F
    pub const VOL_MIN: u8 = 0x20;
    pub const VOL_MAX: u8 = 0x2F;
    pub const VO_MIN: u8 = 0x00;
    pub const VO_MAX: u8 = 0x1F;
}

/// VOP coding types (§6.3.5, `vop_coding_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VopType {
    I,
    P,
    B,
    /// Sprite/GMC VOP (§6.3.5.4) — S-VOP. Not decoded here.
    S,
}

impl VopType {
    fn from_bits(b: u32) -> Self {
        match b {
            0 => VopType::I,
            1 => VopType::P,
            2 => VopType::B,
            _ => VopType::S,
        }
    }
}

/// The persistent per-layer parameters from the Video Object Layer header
/// (§6.3.3). These are parsed once from the stream's in-band ES headers (AVI XviD
/// keeps them) and reused for every VOP in the layer.
#[derive(Clone, Debug)]
pub struct VolHeader {
    pub width: u32,
    pub height: u32,
    /// §6.3.3 `interlaced` — this decoder handles progressive only.
    pub interlaced: bool,
    /// §6.3.3 `quant_type`: 0 == H.263 quantisation (§7.4.4), 1 == MPEG (custom
    /// or default matrices, §7.4.4). XviD ASP files commonly use MPEG-quant.
    pub mpeg_quant: bool,
    /// §7.4.4 intra weighting matrix (natural order), when `mpeg_quant` and a
    /// custom matrix was signalled — else the default intra matrix.
    pub intra_matrix: [u8; 64],
    pub inter_matrix: [u8; 64],
    /// §6.3.3 `quarter_sample` — quarter-pel motion. The target file is half-pel
    /// (this is false); qpel is detected and flagged if seen.
    pub quarter_sample: bool,
    /// §6.3.3.1 sprite_enable — GMC / static sprite. Flagged if non-zero.
    pub sprite_enable: u8,
    /// §6.3.3 `complexity_estimation_disable`.
    pub complexity_estimation_disable: bool,
    /// §6.3.3 `resync_marker_disable` — when false, the VOP may carry video
    /// packet resync markers (error resilience).
    pub resync_marker_disable: bool,
    /// §6.3.3 `data_partitioned`.
    pub data_partitioned: bool,
    /// §6.3.3 `reversible_vlc`.
    pub reversible_vlc: bool,
    /// §6.3.3 `vop_time_increment_resolution` — the modulus for the VOP time base
    /// (needed to size `vop_time_increment` in the VOP header).
    pub time_increment_resolution: u32,
    /// Number of bits used to code `vop_time_increment` (ceil(log2(resolution))).
    pub time_increment_bits: u32,
    /// §6.3.3 `low_delay` (no B-VOPs when set — but XviD sets it inconsistently).
    pub low_delay: bool,
}

/// Default intra quantiser matrix (ISO/IEC 14496-2 §7.4.4, Table 7-6), natural
/// (raster) order.
#[rustfmt::skip]
pub const DEFAULT_INTRA_MATRIX: [u8; 64] = [
     8, 17, 18, 19, 21, 23, 25, 27,
    17, 18, 19, 21, 23, 25, 27, 28,
    20, 21, 22, 23, 24, 26, 28, 30,
    21, 22, 23, 24, 26, 28, 30, 32,
    22, 23, 24, 26, 28, 30, 32, 35,
    23, 24, 26, 28, 30, 32, 35, 38,
    25, 26, 28, 30, 32, 35, 38, 41,
    27, 28, 30, 32, 35, 38, 41, 45,
];

/// Default inter (non-intra) quantiser matrix (§7.4.4, Table 7-7), natural order.
#[rustfmt::skip]
pub const DEFAULT_INTER_MATRIX: [u8; 64] = [
    16, 17, 18, 19, 20, 21, 22, 23,
    17, 18, 19, 20, 21, 22, 23, 24,
    18, 19, 20, 21, 22, 23, 24, 25,
    19, 20, 21, 22, 23, 24, 26, 27,
    20, 21, 22, 23, 25, 26, 27, 28,
    21, 22, 23, 24, 26, 27, 28, 30,
    22, 23, 24, 26, 27, 28, 30, 31,
    23, 24, 25, 27, 28, 30, 31, 33,
];

/// The zig-zag scan (§7.4.2, Figure 7-2) mapping scan index → natural index. Used
/// to read a custom quant matrix (which is coded in zig-zag order).
#[rustfmt::skip]
pub const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

impl Default for VolHeader {
    fn default() -> Self {
        VolHeader {
            width: 0,
            height: 0,
            interlaced: false,
            mpeg_quant: false,
            intra_matrix: DEFAULT_INTRA_MATRIX,
            inter_matrix: DEFAULT_INTER_MATRIX,
            quarter_sample: false,
            sprite_enable: 0,
            complexity_estimation_disable: true,
            resync_marker_disable: true,
            data_partitioned: false,
            reversible_vlc: false,
            time_increment_resolution: 1,
            time_increment_bits: 1,
            low_delay: false,
        }
    }
}

/// The per-picture parameters from a Video Object Plane header (§6.3.5).
#[derive(Clone, Debug)]
pub struct VopHeader {
    pub coding_type: VopType,
    /// §6.3.5 `vop_coded` — 0 means "not coded", the frame repeats its reference.
    pub coded: bool,
    /// §6.3.5 `vop_rounding_type` (P-VOP half-pel rounding control).
    pub rounding_type: u32,
    /// §6.3.5 `vop_quant` — the initial quantiser scale for the VOP.
    pub quant: u32,
    /// §6.3.5 `vop_fcode_forward` (P/B-VOP motion range code, 1..=7).
    pub fcode_forward: u32,
    /// §6.3.5 `vop_fcode_backward` (B-VOP).
    pub fcode_backward: u32,
    /// The `modulo_time_base` + `vop_time_increment`, combined to a display time
    /// tick, used to derive B-VOP interpolation weights (`TRB`/`TRD`).
    pub time_base: u32,
    pub time_increment: u32,
    /// §6.3.5 `intra_dc_vlc_thr` — selects DC coefficient VLC vs the escape code
    /// as a function of quant.
    pub intra_dc_vlc_thr: u32,
}

/// Locate the next start code (`00 00 01 xx`) at or after byte offset `from`.
/// Returns `(offset_of_00, code_byte)` where `offset_of_00` points at the first
/// `0x00` of the prefix. `None` if no further start code exists. Scans
/// byte-aligned, as start codes are always byte-aligned (§6.2.1).
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, u8)> {
    let mut i = from;
    while i + 3 < data.len() + 1 && i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            if i + 3 < data.len() {
                return Some((i, data[i + 3]));
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Parse the Video Object Layer header starting just *after* its start code (the
/// reader is positioned at the first bit of the VOL body). Returns the parsed
/// header, leaving the reader positioned after the fixed portion. Robust: any
/// overrun aborts with `None` rather than reading garbage.
///
/// Follows §6.3.3 `VideoObjectLayer()` for the ASP/SP subset.
pub fn parse_vol(r: &mut BitReader, vol: &mut VolHeader) -> Option<()> {
    let _random_accessible = r.read_bit();
    let _vo_type_indication = r.read_bits(8);
    let is_object_layer_identifier = r.read_bit();
    // Object-layer version id (§6.3.3). v1 == 1; ASP/XviD uses v2+ (typically 5).
    // Several later fields (quarter_sample, sprite_enable width, newpred/reduced
    // resolution) are gated on `ver_id != 1`.
    let mut ver_id = 1u32;
    if is_object_layer_identifier == 1 {
        ver_id = r.read_bits(4);
        let _priority = r.read_bits(3);
    }
    let aspect_ratio_info = r.read_bits(4);
    if aspect_ratio_info == 0xF {
        // extended PAR
        let _par_width = r.read_bits(8);
        let _par_height = r.read_bits(8);
    }
    let vol_control_parameters = r.read_bit();
    if vol_control_parameters == 1 {
        let _chroma_format = r.read_bits(2);
        vol.low_delay = r.read_bit() == 1;
        let vbv_parameters = r.read_bit();
        if vbv_parameters == 1 {
            // first/latter half bitrate, vbv buffer size, occupancy — skipped
            r.skip(15); // first_half_bit_rate
            r.marker_bit();
            r.skip(15); // latter_half_bit_rate
            r.marker_bit();
            r.skip(15); // first_half_vbv_buffer_size
            r.marker_bit();
            r.skip(3); // latter_half_vbv_buffer_size
            r.skip(11); // first_half_vbv_occupancy
            r.marker_bit();
            r.skip(15); // latter_half_vbv_occupancy
            r.marker_bit();
        }
    }
    let shape = r.read_bits(2); // video_object_layer_shape
    if shape != 0 {
        // Only rectangular (shape==0) is supported. Binary/gray shape is exotic.
        return None;
    }
    r.marker_bit();
    let time_res = r.read_bits(16); // vop_time_increment_resolution
    vol.time_increment_resolution = time_res.max(1);
    // ceil(log2(resolution)), at least 1 bit (§6.3.5)
    let mut bits = 0u32;
    let mut v = vol.time_increment_resolution.saturating_sub(1);
    while v > 0 {
        v >>= 1;
        bits += 1;
    }
    vol.time_increment_bits = bits.max(1);
    r.marker_bit();
    let fixed_vop_rate = r.read_bit();
    if fixed_vop_rate == 1 {
        r.skip(vol.time_increment_bits as usize); // fixed_vop_time_increment
    }
    // rectangular shape width/height
    r.marker_bit();
    vol.width = r.read_bits(13);
    r.marker_bit();
    vol.height = r.read_bits(13);
    r.marker_bit();
    vol.interlaced = r.read_bit() == 1;
    let _obmc_disable = r.read_bit();
    // sprite_enable (§6.3.3): 1 bit for object-layer v1, 2 bits for v2+ (values
    // 0 none / 1 static / 2 GMC). The target has GMC off, so this reads 0 either
    // way, but width the read correctly by version.
    vol.sprite_enable = if ver_id == 1 {
        r.read_bit() as u8
    } else {
        r.read_bits(2) as u8
    };
    if vol.sprite_enable != 0 {
        // GMC/sprite parameters follow — not decoded; bail cleanly (the caller
        // warns). We do not attempt to skip them precisely.
        return Some(());
    }
    let _not_8_bit = r.read_bit();
    // (if not_8_bit: quant_precision(4) + bits_per_pixel(4) — assume 8-bit)
    if _not_8_bit == 1 {
        let _quant_precision = r.read_bits(4);
        let _bits_per_pixel = r.read_bits(4);
    }
    vol.mpeg_quant = r.read_bit() == 1; // quant_type
    if vol.mpeg_quant {
        let load_intra = r.read_bit();
        if load_intra == 1 {
            read_quant_matrix(r, &mut vol.intra_matrix, &DEFAULT_INTRA_MATRIX)?;
        } else {
            vol.intra_matrix = DEFAULT_INTRA_MATRIX;
        }
        let load_inter = r.read_bit();
        if load_inter == 1 {
            read_quant_matrix(r, &mut vol.inter_matrix, &DEFAULT_INTER_MATRIX)?;
        } else {
            vol.inter_matrix = DEFAULT_INTER_MATRIX;
        }
    }
    // quarter_sample (§6.3.3) is coded only for object-layer version != 1.
    vol.quarter_sample = if ver_id != 1 { r.read_bit() == 1 } else { false };
    vol.complexity_estimation_disable = r.read_bit() == 1;
    if !vol.complexity_estimation_disable {
        // read_complexity_estimation_header — variable; skip via a conservative
        // parse is hard. The target file has it disabled, so we only handle that.
        // If enabled, bail (caller warns) rather than mis-parse the resync flags.
        return Some(());
    }
    vol.resync_marker_disable = r.read_bit() == 1;
    vol.data_partitioned = r.read_bit() == 1;
    if vol.data_partitioned {
        vol.reversible_vlc = r.read_bit() == 1;
    }
    if r.overrun() {
        return None;
    }
    Some(())
}

/// Read a custom quantiser matrix (§6.3.3): up to 64 8-bit values in zig-zag
/// order, terminated early by a `0` value (the rest inherit the previous). A
/// leading `0` means "use default".
fn read_quant_matrix(r: &mut BitReader, out: &mut [u8; 64], default: &[u8; 64]) -> Option<()> {
    let mut last = 0u8;
    let mut count = 0usize;
    let mut tmp = [0u8; 64];
    for (i, slot) in tmp.iter_mut().enumerate() {
        let v = r.read_bits(8) as u8;
        if v == 0 {
            break;
        }
        *slot = v;
        last = v;
        count = i + 1;
        if r.overrun() {
            return None;
        }
    }
    if count == 0 {
        *out = *default;
        return Some(());
    }
    // Fill remaining entries with the last coded value (§6.3.3).
    for e in tmp.iter_mut().skip(count) {
        *e = last;
    }
    // Coded in zig-zag scan order → convert to natural order.
    for (scan, &nat) in ZIGZAG.iter().enumerate() {
        out[nat] = tmp[scan];
    }
    Some(())
}

/// Parse a Video Object Plane header (§6.3.5) starting just after its start code.
/// `vol` supplies `time_increment_bits`. Returns the header; the reader is left at
/// the first macroblock-layer bit. Robust against overrun (`None`).
pub fn parse_vop(r: &mut BitReader, vol: &VolHeader) -> Option<VopHeader> {
    let coding_type = VopType::from_bits(r.read_bits(2));
    // modulo_time_base: a run of `1`s terminated by `0`.
    let mut modulo = 0u32;
    while r.read_bit() == 1 {
        modulo = modulo.saturating_add(1);
        if r.overrun() || modulo > 1 << 20 {
            return None;
        }
    }
    r.marker_bit();
    let time_increment = r.read_bits(vol.time_increment_bits);
    r.marker_bit();
    let coded = r.read_bit() == 1;
    if !coded {
        return Some(VopHeader {
            coding_type,
            coded: false,
            rounding_type: 0,
            quant: 1,
            fcode_forward: 1,
            fcode_backward: 1,
            time_base: modulo,
            time_increment,
            intra_dc_vlc_thr: 0,
        });
    }
    // (rounding_type is present only for P-VOP)
    let rounding_type = if coding_type == VopType::P { r.read_bit() } else { 0 };
    // intra_dc_vlc_thr (§6.3.5) — for rectangular non-sprite shape.
    let intra_dc_vlc_thr = r.read_bits(3);
    if vol.interlaced {
        let _top_field_first = r.read_bit();
        let _alternate_vertical_scan = r.read_bit();
    }
    let quant = r.read_bits(5);
    let mut fcode_forward = 1;
    let mut fcode_backward = 1;
    if coding_type == VopType::P || coding_type == VopType::S {
        fcode_forward = r.read_bits(3);
    }
    if coding_type == VopType::B {
        fcode_forward = r.read_bits(3);
        fcode_backward = r.read_bits(3);
    }
    if r.overrun() {
        return None;
    }
    Some(VopHeader {
        coding_type,
        coded: true,
        rounding_type,
        quant: quant.max(1),
        fcode_forward: fcode_forward.max(1),
        fcode_backward: fcode_backward.max(1),
        time_base: modulo,
        time_increment,
        intra_dc_vlc_thr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_start_codes() {
        // prefix at offset 1 (code 0xB6), next prefix at offset 6 (code 0xB2).
        let data = [0xFF, 0x00, 0x00, 0x01, 0xB6, 0x12, 0x00, 0x00, 0x01, 0xB2];
        let (off, code) = find_start_code(&data, 0).unwrap();
        assert_eq!((off, code), (1, 0xB6));
        let (off2, code2) = find_start_code(&data, off + 4).unwrap();
        assert_eq!((off2, code2), (6, 0xB2));
    }

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &z in &ZIGZAG {
            assert!(!seen[z]);
            seen[z] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
