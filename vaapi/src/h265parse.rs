//! Hand-rolled H.265 / HEVC Annex-B bitstream parsing (ITU-T Rec. H.265 | ISO/IEC
//! 23008-2). VA-API's decode entrypoint is "slice-level": the driver runs CABAC and
//! reconstruction, but the host must parse the parameter sets and slice segment
//! *headers* and hand the driver a filled `VAPictureParameterBufferHEVC` /
//! `VASliceParameterBufferHEVC`. This module is that parser.
//!
//! Coverage cites the clause at its point of use: NAL framing (§7.3.1.1 / Annex B),
//! Exp-Golomb (§9.2), the profile_tier_level (§7.3.3), SPS (§7.3.2.2.1), PPS
//! (§7.3.2.3.1), short-term reference picture sets (§7.3.7), the slice segment header
//! (§7.3.6.1), and picture order count (§8.3.1).
//!
//! Scope (POC boundary — see [`crate::h265dec`] crate docs): **Main profile, 8-bit,
//! 4:2:0, progressive**. Scaling lists, PCM, tiles, and range/screen-content
//! extensions are recognized (their flags parsed so a picture that uses them is
//! *refused* upstream, not mis-decoded) but the two BluRay-class x265 movies this
//! targets use none of them. Weighted prediction and long-term references are parsed
//! and forwarded. The bit reader is [`crate::h264parse::BitReader`] (codec-agnostic
//! Exp-Golomb over a de-emulated RBSP); only the framing + syntax differ.

#![allow(clippy::needless_range_loop)]

use crate::h264parse::BitReader;

// NAL unit types (§7.4.2.2, Table 7-1) — the ones this parser routes on. HEVC's NAL
// header is 2 bytes: forbidden_zero(1) | nal_unit_type(6) | nuh_layer_id(6) |
// temporal_id_plus1(3).
pub const NAL_TRAIL_N: u8 = 0;
pub const NAL_TRAIL_R: u8 = 1;
pub const NAL_TSA_N: u8 = 2;
pub const NAL_TSA_R: u8 = 3;
pub const NAL_STSA_N: u8 = 4;
pub const NAL_STSA_R: u8 = 5;
pub const NAL_RADL_N: u8 = 6;
pub const NAL_RADL_R: u8 = 7;
pub const NAL_RASL_N: u8 = 8;
pub const NAL_RASL_R: u8 = 9;
pub const NAL_BLA_W_LP: u8 = 16;
pub const NAL_BLA_W_RADL: u8 = 17;
pub const NAL_BLA_N_LP: u8 = 18;
pub const NAL_IDR_W_RADL: u8 = 19;
pub const NAL_IDR_N_LP: u8 = 20;
pub const NAL_CRA_NUT: u8 = 21;
pub const NAL_VPS: u8 = 32;
pub const NAL_SPS: u8 = 33;
pub const NAL_PPS: u8 = 34;
pub const NAL_AUD: u8 = 35;
pub const NAL_EOS: u8 = 36;
pub const NAL_EOB: u8 = 37;
pub const NAL_FD: u8 = 38;
pub const NAL_PREFIX_SEI: u8 = 39;
pub const NAL_SUFFIX_SEI: u8 = 40;

/// Whether `nal_type` is a VCL (slice) NAL (§7.4.2.2: types 0..=31).
pub fn is_slice_nal(nal_type: u8) -> bool {
    nal_type <= 31
}

/// Whether `nal_type` is an IRAP picture (§3: types 16..=23; BLA/IDR/CRA).
pub fn is_irap(nal_type: u8) -> bool {
    (NAL_BLA_W_LP..=23).contains(&nal_type)
}

/// Whether `nal_type` is an IDR picture (§3: types 19/20).
pub fn is_idr(nal_type: u8) -> bool {
    nal_type == NAL_IDR_W_RADL || nal_type == NAL_IDR_N_LP
}

/// Whether `nal_type` is a BLA picture (§3: types 16/17/18).
pub fn is_bla(nal_type: u8) -> bool {
    (NAL_BLA_W_LP..=NAL_BLA_N_LP).contains(&nal_type)
}

/// Whether `nal_type` is a RASL picture (§3: types 8/9), skipped after a CRA/BLA.
pub fn is_rasl(nal_type: u8) -> bool {
    nal_type == NAL_RASL_N || nal_type == NAL_RASL_R
}

/// A parsed NAL: `nal_unit_type` / `temporal_id` (§7.3.1.2) and the raw NAL bytes
/// *including* the 2-byte header and any emulation-prevention bytes — exactly what
/// VA-API wants in the slice-data buffer.
#[derive(Clone)]
pub struct Nal<'a> {
    pub nal_type: u8,
    pub temporal_id: u8,
    /// Raw NAL, start code stripped, emulation bytes intact (fed to VA as slice data).
    pub raw: &'a [u8],
}

/// Split an Annex-B access unit into its NALs (Annex B §B.2.2: 3- or 4-byte start
/// codes `00 00 01` / `00 00 00 01`). A single trailing `0x00` that belongs to the
/// next 4-byte start code is trimmed so it is not fed to the driver as slice data.
pub fn split_nals(au: &[u8]) -> Vec<Nal<'_>> {
    let mut out = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 <= au.len() {
        if au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    for (k, &s) in starts.iter().enumerate() {
        let mut end = if k + 1 < starts.len() { starts[k + 1] - 3 } else { au.len() };
        if end > s && au[end - 1] == 0 && k + 1 < starts.len() {
            end -= 1;
        }
        // Need at least the 2-byte NAL header.
        if end < s + 2 {
            continue;
        }
        let raw = &au[s..end];
        // §7.3.1.2 nal_unit_header: forbidden_zero_bit(1) nal_unit_type(6)
        // nuh_layer_id(6) nuh_temporal_id_plus1(3).
        let nal_type = (raw[0] >> 1) & 0x3f;
        let temporal_id = (raw[1] & 0x07).wrapping_sub(1);
        out.push(Nal { nal_type, temporal_id, raw });
    }
    out
}

/// De-emulate an RBSP (§7.3.1.1): drop each `emulation_prevention_three_byte` (the
/// `0x03` in a `00 00 03` run), skipping the 2-byte NAL header first. The result is
/// what the bit reader consumes; the *raw* NAL (emulation bytes intact) still goes to
/// VA. Also returns the count of emulation bytes removed (VA wants it per slice).
fn rbsp_of(nal: &[u8]) -> (Vec<u8>, u16) {
    let body = &nal[2.min(nal.len())..];
    let mut out = Vec::with_capacity(body.len());
    let mut emu = 0u16;
    let mut zeros = 0;
    let mut idx = 0;
    while idx < body.len() {
        let b = body[idx];
        if zeros >= 2 && b == 0x03 && idx + 1 < body.len() && body[idx + 1] <= 0x03 {
            emu = emu.saturating_add(1);
            zeros = 0;
            idx += 1;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
        idx += 1;
    }
    (out, emu)
}

/// A short-term reference picture set (§7.4.8): the ±POC deltas of the pictures a
/// slice may reference. `delta_poc[i]` are signed offsets from the current POC;
/// `used[i]` marks whether entry `i` is "used by current picture".
#[derive(Clone, Debug, Default)]
pub struct ShortTermRps {
    /// Negative deltas (POC < current), ascending magnitude — §7.4.8 DeltaPocS0.
    pub delta_poc_s0: Vec<i32>,
    pub used_s0: Vec<bool>,
    /// Positive deltas (POC > current) — §7.4.8 DeltaPocS1.
    pub delta_poc_s1: Vec<i32>,
    pub used_s1: Vec<bool>,
}

impl ShortTermRps {
    fn num_negative(&self) -> usize {
        self.delta_poc_s0.len()
    }
    fn num_positive(&self) -> usize {
        self.delta_poc_s1.len()
    }
}

/// Parsed SPS (§7.3.2.2.1) — the fields the VA HEVC picture parameter buffer + POC
/// derivation need. Only Main-profile 8-bit 4:2:0 fields are retained; higher-profile
/// syntax is parsed (to advance the reader) but its presence is exposed via
/// [`Sps::supported`] so an unsupported stream is refused upstream.
#[derive(Clone, Debug)]
pub struct Sps {
    pub sps_id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    /// Conformance-window crop offsets (§7.4.3.2.1), in *chroma sample* units — the
    /// cropped display size is the coded size minus `SubWidthC/SubHeightC ×` these.
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub sps_max_dec_pic_buffering_minus1: u32,
    pub log2_min_luma_coding_block_size_minus3: u32,
    pub log2_diff_max_min_luma_coding_block_size: u32,
    pub log2_min_transform_block_size_minus2: u32,
    pub log2_diff_max_min_transform_block_size: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled_flag: bool,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: u32,
    pub pcm_sample_bit_depth_chroma_minus1: u32,
    pub log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u32,
    pub pcm_loop_filter_disabled_flag: bool,
    pub num_short_term_ref_pic_sets: u32,
    /// The parsed SPS short-term RPS list (§7.3.7). Slices reference these by index.
    pub st_rps: Vec<ShortTermRps>,
    pub long_term_ref_pics_present_flag: bool,
    pub num_long_term_ref_pics_sps: u32,
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    pub used_by_curr_pic_lt_sps_flag: Vec<bool>,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    /// Whether this SPS declares a range/screen-content extension we do not decode.
    pub has_extension: bool,
}

impl Sps {
    /// CtbLog2SizeY (§7.4.3.2.1) — luma CTB size, log2.
    pub fn ctb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_coding_block_size_minus3
            + 3
            + self.log2_diff_max_min_luma_coding_block_size
    }
    fn ctb_size_y(&self) -> u32 {
        1 << self.ctb_log2_size_y()
    }
    /// PicWidthInCtbsY (§7.4.3.2.1).
    pub fn pic_width_in_ctbs(&self) -> u32 {
        self.pic_width_in_luma_samples.div_ceil(self.ctb_size_y())
    }
    pub fn pic_height_in_ctbs(&self) -> u32 {
        self.pic_height_in_luma_samples.div_ceil(self.ctb_size_y())
    }
    /// MaxPicOrderCntLsb (§7.4.3.2.1).
    pub fn max_poc_lsb(&self) -> i32 {
        1 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
    /// SubWidthC / SubHeightC (§Table 6-1). 4:2:0 (chroma_format_idc 1) → 2/2; 4:2:2 →
    /// 2/1; 4:4:4 → 1/1. This decoder only accepts 4:2:0, but the crop math is stated
    /// generally.
    fn sub_wh_c(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }
    /// The conformance-cropped *display* width (§7.4.3.2.1 eq. — the size the renderer
    /// and any reference decoder present), luma samples.
    pub fn display_width(&self) -> u32 {
        let (sub_w, _) = self.sub_wh_c();
        let crop = sub_w * (self.conf_win_left_offset + self.conf_win_right_offset);
        self.pic_width_in_luma_samples.saturating_sub(crop)
    }
    /// The conformance-cropped *display* height (§7.4.3.2.1), luma samples.
    pub fn display_height(&self) -> u32 {
        let (_, sub_h) = self.sub_wh_c();
        let crop = sub_h * (self.conf_win_top_offset + self.conf_win_bottom_offset);
        self.pic_height_in_luma_samples.saturating_sub(crop)
    }
    /// Cropped-region top-left offset in luma samples (§7.4.3.2.1): where the display
    /// picture begins inside the coded surface.
    pub fn crop_origin(&self) -> (u32, u32) {
        let (sub_w, sub_h) = self.sub_wh_c();
        (sub_w * self.conf_win_left_offset, sub_h * self.conf_win_top_offset)
    }
    /// Cropped display width (alias — see [`display_width`]). Kept so callers reading
    /// `sps.width()` get the *presented* size, matching the software decoder's output.
    pub fn width(&self) -> u32 {
        self.display_width()
    }
    /// Cropped display height (alias — see [`display_height`]).
    pub fn height(&self) -> u32 {
        self.display_height()
    }
    /// The *coded* picture width (CTB-multiple storage size), luma samples — the
    /// surface allocation dimension, before conformance cropping.
    pub fn coded_width(&self) -> u32 {
        self.pic_width_in_luma_samples
    }
    /// The *coded* picture height (storage size), luma samples.
    pub fn coded_height(&self) -> u32 {
        self.pic_height_in_luma_samples
    }
    /// Whether this SPS is within this decoder's Main-8-bit-4:2:0 subset.
    pub fn supported(&self) -> bool {
        self.chroma_format_idc == 1
            && !self.separate_colour_plane_flag
            && self.bit_depth_luma_minus8 == 0
            && self.bit_depth_chroma_minus8 == 0
            && !self.has_extension
    }
}

/// Parsed PPS (§7.3.2.3.1).
#[derive(Clone, Debug)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub init_qp_minus26: i32,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub num_tile_columns_minus1: u32,
    pub num_tile_rows_minus1: u32,
    pub uniform_spacing_flag: bool,
    pub column_width_minus1: Vec<u32>,
    pub row_height_minus1: Vec<u32>,
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_beta_offset_div2: i32,
    pub pps_tc_offset_div2: i32,
    pub pps_scaling_list_data_present_flag: bool,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u32,
    pub slice_segment_header_extension_present_flag: bool,
}

/// Slice type (§7.4.7.1): B=0, P=1, I=2 (the values VA's LongSliceFlags.slice_type
/// carries directly).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SliceType {
    B,
    P,
    I,
}

impl SliceType {
    fn from_u32(v: u32) -> Option<SliceType> {
        match v {
            0 => Some(SliceType::B),
            1 => Some(SliceType::P),
            2 => Some(SliceType::I),
            _ => None,
        }
    }
    /// The value VA's `LongSliceFlags.slice_type` field expects (== the H.265 code).
    pub fn va_value(self) -> u32 {
        match self {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        }
    }
    pub fn is_inter(self) -> bool {
        matches!(self, SliceType::P | SliceType::B)
    }
}

/// One explicit weight entry (§7.3.6.3 pred_weight_table). `luma`/`chroma` are the
/// *deltas* HEVC codes (VA wants the deltas verbatim, unlike H.264's absolute weights).
#[derive(Clone, Copy, Default)]
pub struct WeightEntry {
    pub luma_flag: bool,
    pub delta_luma_weight: i32,
    pub luma_offset: i32,
    pub chroma_flag: bool,
    pub delta_chroma_weight: [i32; 2],
    pub chroma_offset: [i32; 2],
}

/// Parsed pred_weight_table (§7.3.6.3).
#[derive(Clone, Default)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    pub delta_chroma_log2_weight_denom: i32,
    pub l0: Vec<WeightEntry>,
    pub l1: Vec<WeightEntry>,
}

/// A resolved long-term reference the slice header selects (§7.3.6.1).
#[derive(Clone, Copy, Default)]
pub struct LongTermRef {
    pub poc_lsb: u32,
    pub used_by_curr_pic: bool,
    pub delta_poc_msb_present: bool,
    pub delta_poc_msb_cycle: u32,
}

/// A ref-list modification entry (§7.3.6.2): explicit index into the RPS-derived
/// candidate list. `l0`/`l1` hold the `list_entry_l{0,1}` values when present.
#[derive(Clone, Default)]
pub struct RefListMods {
    pub l0_present: bool,
    pub l0: Vec<u32>,
    pub l1_present: bool,
    pub l1: Vec<u32>,
}

/// Parsed slice segment header (§7.3.6.1) — the fields the VA HEVC slice parameter
/// buffer + reference-list construction + POC need.
#[derive(Clone)]
pub struct SliceHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub no_output_of_prior_pics_flag: bool,
    pub pps_id: u32,
    pub dependent_slice_segment_flag: bool,
    pub slice_segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    /// slice_pic_order_cnt_lsb (§7.4.7.1) — 0 for IDR (absent → 0).
    pub pic_order_cnt_lsb: u32,
    /// The short-term RPS in force for this slice (from SPS by index, or in-slice).
    pub st_rps: ShortTermRps,
    /// Number of bits the in-slice `short_term_ref_pic_set()` occupied (0 if it came
    /// from the SPS) — VA's `st_rps_bits`.
    pub st_rps_bits: u32,
    pub long_term_refs: Vec<LongTermRef>,
    pub slice_temporal_mvp_enabled_flag: bool,
    pub slice_sao_luma_flag: bool,
    pub slice_sao_chroma_flag: bool,
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub ref_list_mods: RefListMods,
    pub mvd_l1_zero_flag: bool,
    pub cabac_init_flag: bool,
    pub collocated_from_l0_flag: bool,
    pub collocated_ref_idx: u32,
    pub pred_weights: Option<PredWeightTable>,
    pub five_minus_max_num_merge_cand: u32,
    pub slice_qp_delta: i32,
    pub slice_cb_qp_offset: i32,
    pub slice_cr_qp_offset: i32,
    pub deblocking_filter_override_flag: bool,
    pub slice_deblocking_filter_disabled_flag: bool,
    pub slice_beta_offset_div2: i32,
    pub slice_tc_offset_div2: i32,
    pub slice_loop_filter_across_slices_enabled_flag: bool,
    pub num_entry_point_offsets: u32,
    /// Byte offset from the NAL header to the start of slice_data(), in the
    /// de-emulated domain, *including* the NAL header (VA `slice_data_byte_offset`).
    pub slice_data_byte_offset: u32,
    /// Emulation-prevention bytes removed from this NAL (VA
    /// `slice_data_num_emu_prevn_bytes`).
    pub num_emu_prev_bytes: u16,
}

// --- profile_tier_level (§7.3.3) ------------------------------------------------------

/// Skip a `profile_tier_level(profilePresentFlag=1, maxNumSubLayersMinus1)`
/// (§7.3.3). Returns `true` for a Main-family (general_profile_idc 1/2, or the
/// general_progressive+frame_only constraint) profile; `false` if the profile is
/// outside our subset. Advances the reader either way.
fn skip_profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> bool {
    // general_profile_space(2) general_tier_flag(1) general_profile_idc(5)
    let _profile_space = r.u(2);
    let _tier = r.flag();
    let general_profile_idc = r.u(5);
    // general_profile_compatibility_flag[32]
    let mut compat_main = false;
    for j in 0..32 {
        let f = r.flag();
        if j == 1 || j == 2 {
            compat_main |= f; // Main (1) or Main10 (2) compatibility
        }
    }
    // general_progressive_source_flag, interlaced, non_packed, frame_only (4 flags)
    let _progressive = r.flag();
    let _interlaced = r.flag();
    let _non_packed = r.flag();
    let _frame_only = r.flag();
    // 43 reserved constraint flags + 1 (general_inbld_flag or reserved) = 44 bits
    // (general_reserved_zero_43bits + general_reserved_zero_bit / inbld).
    r.u(32);
    r.u(12);
    // general_level_idc(8)
    let _level = r.u(8);

    // sub_layer_profile/level present flags
    let mut sub_profile = [false; 8];
    let mut sub_level = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        if i < 8 {
            sub_profile[i] = r.flag();
            sub_level[i] = r.flag();
        }
    }
    if max_sub_layers_minus1 > 0 {
        // reserved_zero_2bits, up to 8 sub-layers
        for _ in max_sub_layers_minus1..8 {
            r.u(2);
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if i < 8 && sub_profile[i] {
            r.u(2); // profile_space, tier
            r.u(5); // profile_idc  (total 8, but read as parts below)
            r.u(32); // compatibility flags
            r.u(32);
            r.u(12); // 4 source flags + 43+1 reserved
        }
        if i < 8 && sub_level[i] {
            r.u(8); // sub_layer_level_idc
        }
    }
    // Accept the Main / Main10 family (idc 1/2) or a stream that flags Main-compat.
    general_profile_idc == 1 || general_profile_idc == 2 || compat_main
}

// --- scaling_list_data (§7.3.4) -------------------------------------------------------

/// Skip a `scaling_list_data()` (§7.3.4) — advance the reader over custom scaling
/// lists. This decoder refuses scaling-list-enabled streams upstream, but must still
/// consume the syntax to keep the reader aligned for what follows.
fn skip_scaling_list_data(r: &mut BitReader) {
    for size_id in 0..4 {
        let mut matrix_id = 0;
        let step = if size_id == 3 { 3 } else { 1 };
        while matrix_id < 6 {
            let pred_mode = r.flag(); // scaling_list_pred_mode_flag
            if !pred_mode {
                r.ue(); // scaling_list_pred_matrix_id_delta
            } else {
                let coef_num = core::cmp::min(64u32, 1u32 << (4 + (size_id << 1)));
                if size_id > 1 {
                    r.se(); // scaling_list_dc_coef_minus8
                }
                for _ in 0..coef_num {
                    r.se(); // scaling_list_delta_coef
                }
            }
            matrix_id += step;
        }
    }
}

// --- short_term_ref_pic_set (§7.3.7) --------------------------------------------------

/// Parse one `short_term_ref_pic_set(stRpsIdx)` (§7.3.7), resolving the
/// inter-RPS-prediction form against previously-parsed sets. Returns the resolved
/// set. `rps_list` holds the sets parsed so far (index 0..stRpsIdx).
fn parse_short_term_rps(
    r: &mut BitReader,
    st_rps_idx: u32,
    num_st_rps: u32,
    rps_list: &[ShortTermRps],
) -> ShortTermRps {
    let mut inter_pred = false;
    if st_rps_idx != 0 {
        inter_pred = r.flag(); // inter_ref_pic_set_prediction_flag
    }
    if inter_pred {
        // §7.4.8 inter-RPS prediction: derive from the reference RPS.
        let delta_idx_minus1 = if st_rps_idx == num_st_rps { r.ue() } else { 0 };
        let ref_idx = st_rps_idx as i64 - (delta_idx_minus1 as i64 + 1);
        let delta_rps_sign = r.flag();
        let abs_delta_rps_minus1 = r.ue() as i32;
        let delta_rps =
            (1 - 2 * delta_rps_sign as i32) * (abs_delta_rps_minus1 + 1);

        let empty = ShortTermRps::default();
        let ref_rps = rps_list
            .get(ref_idx.clamp(0, i64::MAX) as usize)
            .unwrap_or(&empty);
        let num_ref = ref_rps.num_negative() + ref_rps.num_positive();

        // Reconstruct the reference set's full DeltaPoc list, S0 (neg) then S1 (pos),
        // then the terminating 0 (§7.4.8). We build used/refined per the spec loop.
        // num_ref ≤ 32 (16 neg + 16 pos, HEVC max), so a fixed stack array replaces the
        // per-slice heap Vec; `rd_len` tracks the filled length exactly as `.push` did.
        let mut ref_deltas = [0i32; 33];
        let mut ref_used = [false; 33];
        let mut rd_len = 0usize;
        for j in (0..ref_rps.num_negative()).rev() {
            if rd_len < ref_deltas.len() {
                ref_deltas[rd_len] = ref_rps.delta_poc_s0[j];
                ref_used[rd_len] = ref_rps.used_s0[j];
                rd_len += 1;
            }
        }
        for j in 0..ref_rps.num_positive() {
            if rd_len < ref_deltas.len() {
                ref_deltas[rd_len] = ref_rps.delta_poc_s1[j];
                ref_used[rd_len] = ref_rps.used_s1[j];
                rd_len += 1;
            }
        }

        // used_by_curr / use_delta are indexed by 0..=num_ref (num_ref ≤ 32 refs, HEVC
        // max), so a fixed stack array replaces the per-slice heap Vec. The same
        // num_ref+1 flags are still read (bit-for-bit); array writes clamp to the
        // capacity so a corrupt over-large count cannot index out of bounds.
        let mut used_by_curr = [false; 33];
        let mut use_delta = [true; 33];
        for j in 0..=num_ref {
            let u = r.flag();
            if j < used_by_curr.len() {
                used_by_curr[j] = u;
            }
            if !u {
                let d = r.flag();
                if j < use_delta.len() {
                    use_delta[j] = d;
                }
            }
        }

        let mut out = ShortTermRps::default();
        // Positive-delta side (S1): §7.4.8 eq. 7-59..7-60.
        {
            let mut i = 0i32;
            let mut j = ref_rps.num_positive() as i32 - 1;
            while j >= 0 {
                let d_poc = ref_rps.delta_poc_s1[j as usize] + delta_rps;
                let k = ref_rps.num_negative() + j as usize;
                if d_poc < 0 && use_delta[k] {
                    out.delta_poc_s0.push(d_poc);
                    out.used_s0.push(used_by_curr[k]);
                    i += 1;
                }
                j -= 1;
            }
            let _ = i;
            if delta_rps < 0 && use_delta[num_ref] {
                out.delta_poc_s0.push(delta_rps);
                out.used_s0.push(used_by_curr[num_ref]);
            }
            for j in 0..ref_rps.num_negative() {
                let d_poc = ref_rps.delta_poc_s0[j] + delta_rps;
                if d_poc < 0 && use_delta[j] {
                    out.delta_poc_s0.push(d_poc);
                    out.used_s0.push(used_by_curr[j]);
                }
            }
        }
        // Negative-delta side handled above populates S0; now S1 (positive).
        {
            let mut j = ref_rps.num_negative() as i32 - 1;
            while j >= 0 {
                let d_poc = ref_rps.delta_poc_s0[j as usize] + delta_rps;
                if d_poc > 0 && use_delta[j as usize] {
                    out.delta_poc_s1.push(d_poc);
                    out.used_s1.push(used_by_curr[j as usize]);
                }
                j -= 1;
            }
            if delta_rps > 0 && use_delta[num_ref] {
                out.delta_poc_s1.push(delta_rps);
                out.used_s1.push(used_by_curr[num_ref]);
            }
            for j in 0..ref_rps.num_positive() {
                let d_poc = ref_rps.delta_poc_s1[j] + delta_rps;
                let k = ref_rps.num_negative() + j;
                if d_poc > 0 && use_delta[k] {
                    out.delta_poc_s1.push(d_poc);
                    out.used_s1.push(used_by_curr[k]);
                }
            }
        }
        let _ = ref_deltas;
        let _ = ref_used;
        let _ = rd_len;
        out
    } else {
        // Explicit form (§7.3.7): num_negative_pics, num_positive_pics, then deltas.
        let num_negative = r.ue();
        let num_positive = r.ue();
        // Guard against absurd counts on corrupt input (16 refs is the HEVC max).
        let num_negative = num_negative.min(16);
        let num_positive = num_positive.min(16);
        let mut out = ShortTermRps::default();
        let mut prev = 0i32;
        for _ in 0..num_negative {
            let delta_poc_s0_minus1 = r.ue() as i32;
            let used = r.flag();
            prev -= delta_poc_s0_minus1 + 1;
            out.delta_poc_s0.push(prev);
            out.used_s0.push(used);
        }
        let mut prev = 0i32;
        for _ in 0..num_positive {
            let delta_poc_s1_minus1 = r.ue() as i32;
            let used = r.flag();
            prev += delta_poc_s1_minus1 + 1;
            out.delta_poc_s1.push(prev);
            out.used_s1.push(used);
        }
        out
    }
}

// --- SPS (§7.3.2.2.1) -----------------------------------------------------------------

/// Parse an SPS NAL. Returns `None` on a truncated/implausible header.
pub fn parse_sps(nal: &[u8]) -> Option<Sps> {
    if nal.len() < 3 {
        return None;
    }
    let (rbsp, _emu) = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);

    let _sps_video_parameter_set_id = r.u(4);
    let sps_max_sub_layers_minus1 = r.u(3);
    let _sps_temporal_id_nesting_flag = r.flag();
    let ok_profile = skip_profile_tier_level(&mut r, sps_max_sub_layers_minus1);

    let sps_id = r.ue();
    if sps_id >= 16 {
        return None;
    }
    let chroma_format_idc = r.ue();
    let separate_colour_plane_flag =
        if chroma_format_idc == 3 { r.flag() } else { false };
    let pic_width_in_luma_samples = r.ue();
    let pic_height_in_luma_samples = r.ue();
    // Bound dimensions (8K-ish ceiling) so corrupt input cannot request an
    // absurd surface allocation.
    if pic_width_in_luma_samples == 0
        || pic_height_in_luma_samples == 0
        || pic_width_in_luma_samples > 16384
        || pic_height_in_luma_samples > 16384
    {
        return None;
    }
    let conformance_window_flag = r.flag();
    let (mut conf_win_left_offset, mut conf_win_right_offset) = (0, 0);
    let (mut conf_win_top_offset, mut conf_win_bottom_offset) = (0, 0);
    if conformance_window_flag {
        conf_win_left_offset = r.ue();
        conf_win_right_offset = r.ue();
        conf_win_top_offset = r.ue();
        conf_win_bottom_offset = r.ue();
    }
    let bit_depth_luma_minus8 = r.ue();
    let bit_depth_chroma_minus8 = r.ue();
    let log2_max_pic_order_cnt_lsb_minus4 = r.ue();
    if log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return None;
    }

    let sps_sub_layer_ordering_info_present_flag = r.flag();
    let start = if sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        sps_max_sub_layers_minus1
    };
    let mut sps_max_dec_pic_buffering_minus1 = 0;
    for _ in start..=sps_max_sub_layers_minus1 {
        sps_max_dec_pic_buffering_minus1 = r.ue();
        r.ue(); // sps_max_num_reorder_pics
        r.ue(); // sps_max_latency_increase_plus1
    }

    let log2_min_luma_coding_block_size_minus3 = r.ue();
    let log2_diff_max_min_luma_coding_block_size = r.ue();
    let log2_min_transform_block_size_minus2 = r.ue();
    let log2_diff_max_min_transform_block_size = r.ue();
    let max_transform_hierarchy_depth_inter = r.ue();
    let max_transform_hierarchy_depth_intra = r.ue();

    let scaling_list_enabled_flag = r.flag();
    if scaling_list_enabled_flag {
        let sps_scaling_list_data_present_flag = r.flag();
        if sps_scaling_list_data_present_flag {
            skip_scaling_list_data(&mut r);
        }
    }
    let amp_enabled_flag = r.flag();
    let sample_adaptive_offset_enabled_flag = r.flag();
    let pcm_enabled_flag = r.flag();
    let mut pcm_sample_bit_depth_luma_minus1 = 0;
    let mut pcm_sample_bit_depth_chroma_minus1 = 0;
    let mut log2_min_pcm_luma_coding_block_size_minus3 = 0;
    let mut log2_diff_max_min_pcm_luma_coding_block_size = 0;
    let mut pcm_loop_filter_disabled_flag = false;
    if pcm_enabled_flag {
        pcm_sample_bit_depth_luma_minus1 = r.u(4);
        pcm_sample_bit_depth_chroma_minus1 = r.u(4);
        log2_min_pcm_luma_coding_block_size_minus3 = r.ue();
        log2_diff_max_min_pcm_luma_coding_block_size = r.ue();
        pcm_loop_filter_disabled_flag = r.flag();
    }

    let num_short_term_ref_pic_sets = r.ue();
    if num_short_term_ref_pic_sets > 64 {
        return None;
    }
    let mut st_rps: Vec<ShortTermRps> = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for i in 0..num_short_term_ref_pic_sets {
        let rps = parse_short_term_rps(&mut r, i, num_short_term_ref_pic_sets, &st_rps);
        st_rps.push(rps);
    }

    let long_term_ref_pics_present_flag = r.flag();
    let mut num_long_term_ref_pics_sps = 0;
    let mut lt_ref_pic_poc_lsb_sps = Vec::new();
    let mut used_by_curr_pic_lt_sps_flag = Vec::new();
    if long_term_ref_pics_present_flag {
        num_long_term_ref_pics_sps = r.ue().min(32);
        let lsb_bits = log2_max_pic_order_cnt_lsb_minus4 + 4;
        for _ in 0..num_long_term_ref_pics_sps {
            lt_ref_pic_poc_lsb_sps.push(r.u(lsb_bits));
            used_by_curr_pic_lt_sps_flag.push(r.flag());
        }
    }

    let sps_temporal_mvp_enabled_flag = r.flag();
    let strong_intra_smoothing_enabled_flag = r.flag();
    let vui_parameters_present_flag = r.flag();
    // VUI is not needed for decode; the only thing after it we care about is the
    // extension flag, and we conservatively skip VUI by NOT parsing it (we do not
    // read past here). But sps_extension flags follow VUI; a stream that sets them
    // uses range/screen-content tools we refuse. We detect an extension only when
    // there is no VUI (the common x265 case has VUI); to stay safe, treat a present
    // VUI as "no detectable extension" and rely on the coding-tool flags + profile.
    let has_extension = !ok_profile;
    let _ = vui_parameters_present_flag;

    Some(Sps {
        sps_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conf_win_left_offset,
        conf_win_right_offset,
        conf_win_top_offset,
        conf_win_bottom_offset,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_max_dec_pic_buffering_minus1,
        log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_transform_block_size_minus2,
        log2_diff_max_min_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size,
        pcm_loop_filter_disabled_flag,
        num_short_term_ref_pic_sets,
        st_rps,
        long_term_ref_pics_present_flag,
        num_long_term_ref_pics_sps,
        lt_ref_pic_poc_lsb_sps,
        used_by_curr_pic_lt_sps_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
        has_extension,
    })
}

// --- PPS (§7.3.2.3.1) -----------------------------------------------------------------

/// Parse a PPS NAL. Returns `None` on a truncated/implausible header.
pub fn parse_pps(nal: &[u8]) -> Option<Pps> {
    if nal.len() < 3 {
        return None;
    }
    let (rbsp, _emu) = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);

    let pps_id = r.ue();
    let sps_id = r.ue();
    if pps_id >= 64 || sps_id >= 16 {
        return None;
    }
    let dependent_slice_segments_enabled_flag = r.flag();
    let output_flag_present_flag = r.flag();
    let num_extra_slice_header_bits = r.u(3);
    let sign_data_hiding_enabled_flag = r.flag();
    let cabac_init_present_flag = r.flag();
    let num_ref_idx_l0_default_active_minus1 = r.ue();
    let num_ref_idx_l1_default_active_minus1 = r.ue();
    let init_qp_minus26 = r.se();
    let constrained_intra_pred_flag = r.flag();
    let transform_skip_enabled_flag = r.flag();
    let cu_qp_delta_enabled_flag = r.flag();
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag { r.ue() } else { 0 };
    let pps_cb_qp_offset = r.se();
    let pps_cr_qp_offset = r.se();
    let pps_slice_chroma_qp_offsets_present_flag = r.flag();
    let weighted_pred_flag = r.flag();
    let weighted_bipred_flag = r.flag();
    let transquant_bypass_enabled_flag = r.flag();
    let tiles_enabled_flag = r.flag();
    let entropy_coding_sync_enabled_flag = r.flag();

    let mut num_tile_columns_minus1 = 0;
    let mut num_tile_rows_minus1 = 0;
    let mut uniform_spacing_flag = true;
    let mut column_width_minus1 = Vec::new();
    let mut row_height_minus1 = Vec::new();
    let mut loop_filter_across_tiles_enabled_flag = true;
    if tiles_enabled_flag {
        num_tile_columns_minus1 = r.ue().min(18);
        num_tile_rows_minus1 = r.ue().min(20);
        uniform_spacing_flag = r.flag();
        if !uniform_spacing_flag {
            for _ in 0..num_tile_columns_minus1 {
                column_width_minus1.push(r.ue());
            }
            for _ in 0..num_tile_rows_minus1 {
                row_height_minus1.push(r.ue());
            }
        }
        loop_filter_across_tiles_enabled_flag = r.flag();
    }
    let pps_loop_filter_across_slices_enabled_flag = r.flag();
    let deblocking_filter_control_present_flag = r.flag();
    let mut deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let mut pps_beta_offset_div2 = 0;
    let mut pps_tc_offset_div2 = 0;
    if deblocking_filter_control_present_flag {
        deblocking_filter_override_enabled_flag = r.flag();
        pps_deblocking_filter_disabled_flag = r.flag();
        if !pps_deblocking_filter_disabled_flag {
            pps_beta_offset_div2 = r.se();
            pps_tc_offset_div2 = r.se();
        }
    }
    let pps_scaling_list_data_present_flag = r.flag();
    if pps_scaling_list_data_present_flag {
        skip_scaling_list_data(&mut r);
    }
    let lists_modification_present_flag = r.flag();
    let log2_parallel_merge_level_minus2 = r.ue();
    let slice_segment_header_extension_present_flag = r.flag();

    Some(Pps {
        pps_id,
        sps_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp_minus26,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
        num_tile_columns_minus1,
        num_tile_rows_minus1,
        uniform_spacing_flag,
        column_width_minus1,
        row_height_minus1,
        loop_filter_across_tiles_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2,
        pps_tc_offset_div2,
        pps_scaling_list_data_present_flag,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
    })
}

// --- pred_weight_table (§7.3.6.3) -----------------------------------------------------

fn parse_pred_weight_table(
    r: &mut BitReader,
    slice_type: SliceType,
    chroma_format_idc: u32,
    num_l0: usize,
    num_l1: usize,
) -> PredWeightTable {
    let mut t = PredWeightTable {
        luma_log2_weight_denom: r.ue(),
        ..Default::default()
    };
    if chroma_format_idc != 0 {
        t.delta_chroma_log2_weight_denom = r.se();
    }
    let read_list = |r: &mut BitReader, n: usize| -> Vec<WeightEntry> {
        let mut luma_flags = vec![false; n];
        let mut chroma_flags = vec![false; n];
        for i in 0..n {
            luma_flags[i] = r.flag();
        }
        if chroma_format_idc != 0 {
            for i in 0..n {
                chroma_flags[i] = r.flag();
            }
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut e = WeightEntry { luma_flag: luma_flags[i], chroma_flag: chroma_flags[i], ..Default::default() };
            if luma_flags[i] {
                e.delta_luma_weight = r.se();
                e.luma_offset = r.se();
            }
            if chroma_flags[i] {
                for c in 0..2 {
                    e.delta_chroma_weight[c] = r.se();
                    e.chroma_offset[c] = r.se();
                }
            }
            out.push(e);
        }
        out
    };
    t.l0 = read_list(r, num_l0);
    if slice_type == SliceType::B {
        t.l1 = read_list(r, num_l1);
    }
    t
}

// --- slice segment header (§7.3.6.1) --------------------------------------------------

/// Parse a slice segment header. `nal_type` is the VCL NAL type; `sps`/`pps` are the
/// active parameter sets. Returns `None` on a malformed/out-of-subset header.
///
/// The header's optional fields are pre-seeded with their spec defaults (PPS-inherited
/// deblocking, `SliceType::I`, active ref counts) and overwritten as the syntax
/// presents them; a dependent slice segment returns `None` before any is read.
#[allow(unused_assignments)]
pub fn parse_slice_header(
    nal: &[u8],
    nal_type: u8,
    sps: &Sps,
    pps: &Pps,
) -> Option<SliceHeader> {
    if nal.len() < 3 {
        return None;
    }
    let (rbsp, emu) = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);

    let first_slice_segment_in_pic_flag = r.flag();
    let mut no_output_of_prior_pics_flag = false;
    if is_irap(nal_type) {
        no_output_of_prior_pics_flag = r.flag();
    }
    let pps_id = r.ue();
    if pps_id != pps.pps_id {
        return None;
    }

    let mut dependent_slice_segment_flag = false;
    let mut slice_segment_address = 0;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            dependent_slice_segment_flag = r.flag();
        }
        // Ceil(Log2(PicSizeInCtbsY)) bits.
        let pic_size_in_ctbs = sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs();
        let bits = ceil_log2(pic_size_in_ctbs);
        slice_segment_address = r.u(bits);
    }

    // Fields that are only present in an independent slice segment.
    let mut slice_type = SliceType::I;
    let mut pic_output_flag = true;
    let mut pic_order_cnt_lsb = 0;
    let mut st_rps = ShortTermRps::default();
    let mut st_rps_bits = 0u32;
    let mut long_term_refs: Vec<LongTermRef> = Vec::new();
    let mut slice_temporal_mvp_enabled_flag = false;
    let mut slice_sao_luma_flag = false;
    let mut slice_sao_chroma_flag = false;
    let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    let mut ref_list_mods = RefListMods::default();
    let mut mvd_l1_zero_flag = false;
    let mut cabac_init_flag = false;
    let mut collocated_from_l0_flag = true;
    let mut collocated_ref_idx = 0;
    let mut pred_weights = None;
    let mut five_minus_max_num_merge_cand = 0;
    let mut slice_cb_qp_offset = 0;
    let mut slice_cr_qp_offset = 0;
    let mut deblocking_filter_override_flag = false;
    let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
    let mut slice_beta_offset_div2 = pps.pps_beta_offset_div2;
    let mut slice_tc_offset_div2 = pps.pps_tc_offset_div2;
    let mut slice_loop_filter_across_slices_enabled_flag =
        pps.pps_loop_filter_across_slices_enabled_flag;

    if !dependent_slice_segment_flag {
        // num_extra_slice_header_bits — skip.
        for _ in 0..pps.num_extra_slice_header_bits {
            r.flag();
        }
        let st = r.ue();
        slice_type = SliceType::from_u32(st)?;
        if pps.output_flag_present_flag {
            pic_output_flag = r.flag();
        }
        if sps.separate_colour_plane_flag {
            r.u(2); // colour_plane_id
        }

        if !is_idr(nal_type) {
            let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            pic_order_cnt_lsb = r.u(lsb_bits);
            let short_term_ref_pic_set_sps_flag = r.flag();
            if !short_term_ref_pic_set_sps_flag {
                let bits_before = r.bit_pos();
                st_rps = parse_short_term_rps(
                    &mut r,
                    sps.num_short_term_ref_pic_sets,
                    sps.num_short_term_ref_pic_sets,
                    &sps.st_rps,
                );
                st_rps_bits = (r.bit_pos() - bits_before) as u32;
            } else if sps.num_short_term_ref_pic_sets > 0 {
                let n_bits = ceil_log2(sps.num_short_term_ref_pic_sets);
                let idx = if n_bits > 0 { r.u(n_bits) } else { 0 };
                st_rps = sps.st_rps.get(idx as usize).cloned().unwrap_or_default();
            }

            if sps.long_term_ref_pics_present_flag {
                let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
                    r.ue()
                } else {
                    0
                };
                let num_long_term_pics = r.ue();
                let total = (num_long_term_sps + num_long_term_pics).min(32);
                let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
                for i in 0..total {
                    let mut lt = LongTermRef::default();
                    if i < num_long_term_sps {
                        // lt_idx_sps
                        let idx = if sps.num_long_term_ref_pics_sps > 1 {
                            r.u(ceil_log2(sps.num_long_term_ref_pics_sps))
                        } else {
                            0
                        } as usize;
                        lt.poc_lsb =
                            sps.lt_ref_pic_poc_lsb_sps.get(idx).copied().unwrap_or(0);
                        lt.used_by_curr_pic =
                            sps.used_by_curr_pic_lt_sps_flag.get(idx).copied().unwrap_or(false);
                    } else {
                        lt.poc_lsb = r.u(lsb_bits);
                        lt.used_by_curr_pic = r.flag();
                    }
                    lt.delta_poc_msb_present = r.flag();
                    if lt.delta_poc_msb_present {
                        lt.delta_poc_msb_cycle = r.ue();
                    }
                    long_term_refs.push(lt);
                }
            }

            if sps.sps_temporal_mvp_enabled_flag {
                slice_temporal_mvp_enabled_flag = r.flag();
            }
        }

        if sps.sample_adaptive_offset_enabled_flag {
            slice_sao_luma_flag = r.flag();
            if sps.chroma_format_idc != 0 {
                slice_sao_chroma_flag = r.flag();
            }
        }

        if slice_type.is_inter() {
            let num_ref_idx_active_override_flag = r.flag();
            if num_ref_idx_active_override_flag {
                num_ref_idx_l0_active_minus1 = r.ue().min(14);
                if slice_type == SliceType::B {
                    num_ref_idx_l1_active_minus1 = r.ue().min(14);
                }
            }

            // Number of pictures marked "used by current" — the candidate list size.
            let num_pic_total_curr = num_pic_total_curr(&st_rps, &long_term_refs, pps);

            if pps.lists_modification_present_flag && num_pic_total_curr > 1 {
                let n_bits = ceil_log2(num_pic_total_curr);
                let l0_flag = r.flag();
                ref_list_mods.l0_present = l0_flag;
                if l0_flag {
                    for _ in 0..=num_ref_idx_l0_active_minus1 {
                        ref_list_mods.l0.push(r.u(n_bits));
                    }
                }
                if slice_type == SliceType::B {
                    let l1_flag = r.flag();
                    ref_list_mods.l1_present = l1_flag;
                    if l1_flag {
                        for _ in 0..=num_ref_idx_l1_active_minus1 {
                            ref_list_mods.l1.push(r.u(n_bits));
                        }
                    }
                }
            }

            if slice_type == SliceType::B {
                mvd_l1_zero_flag = r.flag();
            }
            if pps.cabac_init_present_flag {
                cabac_init_flag = r.flag();
            }
            if slice_temporal_mvp_enabled_flag {
                if slice_type == SliceType::B {
                    collocated_from_l0_flag = r.flag();
                }
                let active = if collocated_from_l0_flag {
                    num_ref_idx_l0_active_minus1
                } else {
                    num_ref_idx_l1_active_minus1
                };
                if active > 0 {
                    collocated_ref_idx = r.ue();
                }
            }
            let weighted = (pps.weighted_pred_flag && slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && slice_type == SliceType::B);
            if weighted {
                pred_weights = Some(parse_pred_weight_table(
                    &mut r,
                    slice_type,
                    sps.chroma_format_idc,
                    num_ref_idx_l0_active_minus1 as usize + 1,
                    num_ref_idx_l1_active_minus1 as usize + 1,
                ));
            }
            five_minus_max_num_merge_cand = r.ue();
        }

        let slice_qp_delta = r.se();
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            slice_cb_qp_offset = r.se();
            slice_cr_qp_offset = r.se();
        }
        // (pps_slice_act_qp_offsets_present, chroma_qp_offset_list — SCC/RExt only;
        // out of subset, so absent for the streams we accept.)
        let deblocking_filter_override = pps.deblocking_filter_override_enabled_flag;
        if deblocking_filter_override {
            deblocking_filter_override_flag = r.flag();
        }
        if deblocking_filter_override_flag {
            slice_deblocking_filter_disabled_flag = r.flag();
            if !slice_deblocking_filter_disabled_flag {
                slice_beta_offset_div2 = r.se();
                slice_tc_offset_div2 = r.se();
            }
        }
        if pps.pps_loop_filter_across_slices_enabled_flag
            && (slice_sao_luma_flag
                || slice_sao_chroma_flag
                || !slice_deblocking_filter_disabled_flag)
        {
            slice_loop_filter_across_slices_enabled_flag = r.flag();
        }

        let slice_qp_delta_val = slice_qp_delta;

        // entry point offsets (tiles / WPP). We must consume them to reach
        // byte_alignment, then slice_data() begins on the next byte boundary.
        let mut num_entry_point_offsets = 0;
        if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
            num_entry_point_offsets = r.ue();
            if num_entry_point_offsets > 0 {
                let offset_len_minus1 = r.ue().min(31);
                for _ in 0..num_entry_point_offsets {
                    r.u(offset_len_minus1 + 1);
                }
            }
        }
        if pps.slice_segment_header_extension_present_flag {
            let ext_len = r.ue();
            for _ in 0..ext_len {
                r.u(8);
            }
        }

        // byte_alignment(): alignment_bit_equal_to_one, then zeros to a byte boundary.
        r.flag();
        while !r.bit_pos().is_multiple_of(8) {
            r.flag();
        }
        let slice_data_bits = r.bit_pos();

        return Some(SliceHeader {
            first_slice_segment_in_pic_flag,
            no_output_of_prior_pics_flag,
            pps_id,
            dependent_slice_segment_flag,
            slice_segment_address,
            slice_type,
            pic_output_flag,
            pic_order_cnt_lsb,
            st_rps,
            st_rps_bits,
            long_term_refs,
            slice_temporal_mvp_enabled_flag,
            slice_sao_luma_flag,
            slice_sao_chroma_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            ref_list_mods,
            mvd_l1_zero_flag,
            cabac_init_flag,
            collocated_from_l0_flag,
            collocated_ref_idx,
            pred_weights,
            five_minus_max_num_merge_cand,
            slice_qp_delta: slice_qp_delta_val,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            deblocking_filter_override_flag,
            slice_deblocking_filter_disabled_flag,
            slice_beta_offset_div2,
            slice_tc_offset_div2,
            slice_loop_filter_across_slices_enabled_flag,
            num_entry_point_offsets,
            // slice_data_byte_offset counts from the NAL header (2 bytes) + de-emulated
            // slice-header bytes. slice_data_bits is measured within the RBSP body
            // (after the 2-byte header), so add the header's 16 bits.
            slice_data_byte_offset: (slice_data_bits as u32 + 16) / 8,
            num_emu_prev_bytes: emu,
        });
    }

    // Dependent slice segment: inherits the independent segment's header. We do not
    // decode multi-segment tiled pictures in this subset; refuse rather than guess.
    None
}

/// PicOrderCntVal (§8.3.1): full POC from the slice's `pic_order_cnt_lsb` and the
/// running `prev_poc_tid0`. Returns the new POC. IDR pictures reset to 0.
pub struct PocState {
    prev_poc_lsb: i32,
    prev_poc_msb: i32,
}

impl PocState {
    pub fn new() -> Self {
        PocState { prev_poc_lsb: 0, prev_poc_msb: 0 }
    }
    pub fn reset(&mut self) {
        self.prev_poc_lsb = 0;
        self.prev_poc_msb = 0;
    }
    /// Compute PicOrderCntVal (§8.3.1). `is_irap_no_rasl` is true when this is an
    /// IRAP with NoRaslOutputFlag (a clean random-access point) — then POC MSB is 0.
    pub fn compute(
        &mut self,
        sps: &Sps,
        sh: &SliceHeader,
        is_idr: bool,
        is_irap_no_rasl: bool,
        temporal_id: u8,
    ) -> i32 {
        if is_idr {
            self.prev_poc_lsb = 0;
            self.prev_poc_msb = 0;
            return 0;
        }
        let max_poc_lsb = sps.max_poc_lsb();
        let poc_lsb = sh.pic_order_cnt_lsb as i32;
        let poc_msb = if is_irap_no_rasl {
            0
        } else {
            let prev_lsb = self.prev_poc_lsb;
            let prev_msb = self.prev_poc_msb;
            if poc_lsb < prev_lsb && (prev_lsb - poc_lsb) >= max_poc_lsb / 2 {
                prev_msb + max_poc_lsb
            } else if poc_lsb > prev_lsb && (poc_lsb - prev_lsb) > max_poc_lsb / 2 {
                prev_msb - max_poc_lsb
            } else {
                prev_msb
            }
        };
        let poc = poc_msb + poc_lsb;
        // Update prev_poc_{lsb,msb} only for TemporalId 0 non-RASL/RADL/sub-layer
        // pictures (§8.3.1). x265 GOP structure keeps refs at TId 0; approximate by
        // updating on TId 0.
        if temporal_id == 0 {
            self.prev_poc_lsb = poc_lsb;
            self.prev_poc_msb = poc_msb;
        }
        poc
    }
}

impl Default for PocState {
    fn default() -> Self {
        Self::new()
    }
}

/// NumPicTotalCurr (§7.4.7.2): count of reference pictures used by the current
/// picture (short-term used + long-term used [+ IBC, out of subset]).
fn num_pic_total_curr(st_rps: &ShortTermRps, lt: &[LongTermRef], _pps: &Pps) -> u32 {
    let mut n = 0u32;
    for &u in &st_rps.used_s0 {
        n += u as u32;
    }
    for &u in &st_rps.used_s1 {
        n += u as u32;
    }
    for l in lt {
        n += l.used_by_curr_pic as u32;
    }
    n
}

/// Ceil(Log2(n)) — the fixed-length field width HEVC uses for indices (§9.2).
pub fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// The short-term RPS accessor for reference-list building: the negative deltas
/// (POC < cur) used by the current picture, ascending magnitude → the S0 "before"
/// list; positive deltas → the S1 "after" list. Exposed for [`crate::h265dec`].
impl ShortTermRps {
    /// (delta, used) pairs for POC-before references (§8.3.2 RefPicSetStCurrBefore).
    pub fn curr_before(&self) -> Vec<i32> {
        self.delta_poc_s0
            .iter()
            .zip(self.used_s0.iter())
            .filter(|(_, u)| **u)
            .map(|(d, _)| *d)
            .collect()
    }
    /// (delta) for POC-after references (§8.3.2 RefPicSetStCurrAfter).
    pub fn curr_after(&self) -> Vec<i32> {
        self.delta_poc_s1
            .iter()
            .zip(self.used_s1.iter())
            .filter(|(_, u)| **u)
            .map(|(d, _)| *d)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_spec() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(33), 6);
    }

    #[test]
    fn nal_header_decodes_type_and_tid() {
        // 00 00 01 | 0x40 0x01 | ... → nal_type = 0x40>>1 & 0x3f = 32 (VPS), tid = 0.
        let au = [0, 0, 1, 0x40, 0x01, 0xAA];
        let nals = split_nals(&au);
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].nal_type, NAL_VPS);
        assert_eq!(nals[0].temporal_id, 0);
    }

    #[test]
    fn idr_poc_resets_to_zero() {
        let mut poc = PocState::new();
        // A minimal SPS stub for max_poc_lsb.
        let sps = stub_sps();
        let sh = stub_slice(0);
        let v = poc.compute(&sps, &sh, true, true, 0);
        assert_eq!(v, 0, "IDR POC must reset to 0");
    }

    #[test]
    fn poc_msb_wraps_forward() {
        let mut poc = PocState::new();
        let sps = stub_sps(); // log2_max_poc_lsb_minus4 = 0 → max_poc_lsb = 16
        // Anchor at an IDR (POC 0, prev = 0/0), then advance to lsb=14 cleanly
        // (14-0=14 > 8 would spuriously wrap, so step via lsb=8 first).
        let _ = poc.compute(&sps, &stub_slice(0), true, true, 0);
        let _ = poc.compute(&sps, &stub_slice(8), false, false, 0); // poc 8
        let _ = poc.compute(&sps, &stub_slice(14), false, false, 0); // poc 14
        // Next lsb wraps to 1: prev_lsb=14, (14-1)=13 >= 8 → msb += 16 → poc 17.
        let v = poc.compute(&sps, &stub_slice(1), false, false, 0);
        assert_eq!(v, 17, "POC MSB must advance on lsb wrap");
    }

    fn stub_sps() -> Sps {
        Sps {
            sps_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 64,
            pic_height_in_luma_samples: 64,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0, // max_poc_lsb = 16
            sps_max_dec_pic_buffering_minus1: 4,
            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 3,
            log2_min_transform_block_size_minus2: 0,
            log2_diff_max_min_transform_block_size: 3,
            max_transform_hierarchy_depth_inter: 0,
            max_transform_hierarchy_depth_intra: 0,
            scaling_list_enabled_flag: false,
            amp_enabled_flag: true,
            sample_adaptive_offset_enabled_flag: true,
            pcm_enabled_flag: false,
            pcm_sample_bit_depth_luma_minus1: 0,
            pcm_sample_bit_depth_chroma_minus1: 0,
            log2_min_pcm_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_pcm_luma_coding_block_size: 0,
            pcm_loop_filter_disabled_flag: false,
            num_short_term_ref_pic_sets: 0,
            st_rps: Vec::new(),
            long_term_ref_pics_present_flag: false,
            num_long_term_ref_pics_sps: 0,
            lt_ref_pic_poc_lsb_sps: Vec::new(),
            used_by_curr_pic_lt_sps_flag: Vec::new(),
            sps_temporal_mvp_enabled_flag: true,
            strong_intra_smoothing_enabled_flag: false,
            has_extension: false,
        }
    }

    fn stub_slice(lsb: u32) -> SliceHeader {
        SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            pps_id: 0,
            dependent_slice_segment_flag: false,
            slice_segment_address: 0,
            slice_type: SliceType::P,
            pic_output_flag: true,
            pic_order_cnt_lsb: lsb,
            st_rps: ShortTermRps::default(),
            st_rps_bits: 0,
            long_term_refs: Vec::new(),
            slice_temporal_mvp_enabled_flag: false,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_list_mods: RefListMods::default(),
            mvd_l1_zero_flag: false,
            cabac_init_flag: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            pred_weights: None,
            five_minus_max_num_merge_cand: 0,
            slice_qp_delta: 0,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            deblocking_filter_override_flag: false,
            slice_deblocking_filter_disabled_flag: false,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_loop_filter_across_slices_enabled_flag: true,
            num_entry_point_offsets: 0,
            slice_data_byte_offset: 0,
            num_emu_prev_bytes: 0,
        }
    }
}
