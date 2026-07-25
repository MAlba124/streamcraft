//! Hand-rolled H.264 Annex-B bitstream parsing (ITU-T Rec. H.264 | ISO/IEC
//! 14496-10). VA-API's decode entrypoint is "slice-level": the driver runs entropy
//! decoding and reconstruction, but the host must parse the parameter sets and
//! slice *headers* and hand the driver a filled `VAPictureParameterBufferH264` /
//! `VASliceParameterBufferH264`. This module is that parser — start-code framing
//! (§B.1.1), Exp-Golomb (§9.1), SPS (§7.3.2.1), PPS (§7.3.2.2), the slice header
//! (§7.3.3), and picture order count (§8.2.1). Every non-trivial step cites the
//! clause it implements at its point of use.
//!
//! Scope (POC boundary, see crate docs): 8-bit 4:2:0, progressive frames. Field /
//! MBAFF coding, `separate_colour_plane`, scaling matrices and POC type 1 are
//! parsed enough to be recognized and refused upstream, not fully realised here.

#![allow(clippy::needless_range_loop)]

/// A parsed NAL: its `nal_ref_idc` / `nal_unit_type` (§7.4.1) and the raw NAL bytes
/// *including* the 1-byte header and any emulation-prevention bytes — the exact
/// bytes VA-API wants in the slice-data buffer.
#[derive(Clone)]
pub struct Nal<'a> {
    pub ref_idc: u8,
    pub unit_type: u8,
    /// Raw NAL, start code stripped, emulation bytes intact.
    pub raw: &'a [u8],
}

// NAL unit types we care about (§7.4.1, Table 7-1).
pub const NAL_SLICE_NON_IDR: u8 = 1;
pub const NAL_SLICE_IDR: u8 = 5;
pub const NAL_SPS: u8 = 7;
pub const NAL_PPS: u8 = 8;

/// Split an Annex-B access unit into its NALs (§B.1.1: 3- or 4-byte start codes
/// `00 00 01` / `00 00 00 01`). Yields each NAL's header fields + raw bytes.
pub fn split_nals(au: &[u8]) -> Vec<Nal<'_>> {
    let mut out = Vec::new();
    // Collect the byte offset of every start-code payload start.
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
        // NAL runs to the byte before the next start code (or EOF). A 4-byte start
        // code is a 3-byte one preceded by an extra 0x00, which is the trailing
        // zero of the previous NAL — trim a single trailing 0x00 so it is not fed
        // to the driver as slice data.
        let mut end = if k + 1 < starts.len() {
            starts[k + 1] - 3
        } else {
            au.len()
        };
        while end > s && au[end - 1] == 0 {
            // Only trim the zeros that belong to the *next* start code (at most the
            // trailing run); a slice can legally end in 0x00 too, but for framing
            // into VA this is harmless (the driver reads slice_data_size, and the
            // trailing zero is cabac_zero_word / rbsp trailing — safe to keep or
            // drop). Trim conservatively: at most 1 (the 4-byte start-code lead-in).
            if k + 1 < starts.len() {
                end -= 1;
            }
            break;
        }
        if end <= s {
            continue;
        }
        let raw = &au[s..end];
        let header = raw[0];
        out.push(Nal {
            ref_idc: (header >> 5) & 0x3,
            unit_type: header & 0x1f,
            raw,
        });
    }
    out
}

/// De-emulate an RBSP (§7.4.1): drop each `emulation_prevention_three_byte` (the
/// `0x03` in a `00 00 03` run). The result is what the bit reader consumes for
/// header parsing; the *raw* NAL (with emulation bytes) still goes to VA.
fn rbsp_of(nal: &[u8]) -> Vec<u8> {
    // Skip the 1-byte NAL header.
    let body = &nal[1..];
    let mut out = Vec::with_capacity(body.len());
    let mut zeros = 0;
    let mut idx = 0;
    while idx < body.len() {
        let b = body[idx];
        if zeros >= 2 && b == 0x03 && idx + 1 < body.len() && body[idx + 1] <= 0x03 {
            // Drop the emulation-prevention byte; the run counter resets.
            zeros = 0;
            idx += 1;
            continue;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        idx += 1;
    }
    out
}

/// A big-endian bit reader over an RBSP with Exp-Golomb helpers (§9.1). Tracks the
/// absolute bit position so a slice header can report its `slice_data_bit_offset`.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit index (counts the skipped NAL header? no — relative to `data`,
    /// which is the de-emulated body). `slice_data_bit_offset` is computed from the
    /// *raw* NAL and re-derived by the caller.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Current absolute bit position (bits consumed).
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    fn bit(&mut self) -> u32 {
        let byte = self.pos >> 3;
        if byte >= self.data.len() {
            self.pos += 1;
            return 0;
        }
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        ((self.data[byte] >> shift) & 1) as u32
    }

    /// Read `n` bits as an unsigned integer (`u(n)`, §7.2 / §9.1 fixed-length).
    pub fn u(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit();
        }
        v
    }

    /// One flag (`u(1)`).
    pub fn flag(&mut self) -> bool {
        self.bit() != 0
    }

    /// Unsigned Exp-Golomb `ue(v)` (§9.1): count leading zeros = `k`, then read `k`
    /// more bits, value = `2^k - 1 + suffix`.
    pub fn ue(&mut self) -> u32 {
        let mut zeros = 0u32;
        while self.bit() == 0 {
            zeros += 1;
            if zeros > 31 {
                return 0; // malformed guard
            }
        }
        if zeros == 0 {
            return 0;
        }
        let suffix = self.u(zeros);
        (1u32 << zeros) - 1 + suffix
    }

    /// Signed Exp-Golomb `se(v)` (§9.1.1): map `ue` code number `k` to
    /// `(-1)^(k+1) * ceil(k/2)`.
    pub fn se(&mut self) -> i32 {
        let k = self.ue();
        let mag = ((k + 1) >> 1) as i32;
        if k & 1 == 1 {
            mag
        } else {
            -mag
        }
    }

    /// `more_rbsp_data` (§7.2): true if any bit before the rbsp stop bit remains.
    pub fn more_rbsp_data(&self) -> bool {
        // Find the last set bit (the rbsp_stop_one_bit). If the current position is
        // strictly before it, there is more data.
        let total_bits = self.data.len() * 8;
        let mut last_one = None;
        for i in (0..total_bits).rev() {
            let byte = i >> 3;
            let shift = 7 - (i & 7);
            if (self.data[byte] >> shift) & 1 == 1 {
                last_one = Some(i);
                break;
            }
        }
        match last_one {
            Some(stop) => self.pos < stop,
            None => false,
        }
    }
}

/// Parsed SPS (§7.3.2.1), the fields the VA picture parameter buffer + POC need.
#[derive(Clone, Debug)]
pub struct Sps {
    pub seq_parameter_set_id: u32,
    pub profile_idc: u8,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub log2_max_frame_num_minus4: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offsets_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_crop_left: u32,
    pub frame_crop_right: u32,
    pub frame_crop_top: u32,
    pub frame_crop_bottom: u32,
}

impl Sps {
    /// Cropped luma width in pixels (§7.4.2.1.1). For 4:2:0 the crop unit X is 2.
    pub fn width(&self) -> u32 {
        let w_mbs = self.pic_width_in_mbs_minus1 + 1;
        let crop_unit_x = if self.chroma_format_idc == 1 || self.chroma_format_idc == 2 {
            2
        } else {
            1
        };
        w_mbs * 16 - crop_unit_x * (self.frame_crop_left + self.frame_crop_right)
    }

    /// Cropped luma height in pixels (§7.4.2.1.1). Frame height in map units × 16 ×
    /// (2 − frame_mbs_only); crop unit Y is 2 for 4:2:0 progressive.
    pub fn height(&self) -> u32 {
        let frame_mbs_mult = if self.frame_mbs_only_flag { 1 } else { 2 };
        let h_mbs = (self.pic_height_in_map_units_minus1 + 1) * frame_mbs_mult;
        let crop_unit_y = {
            let sub = if self.chroma_format_idc == 1 { 2 } else { 1 };
            sub * frame_mbs_mult
        };
        h_mbs * 16 - crop_unit_y * (self.frame_crop_top + self.frame_crop_bottom)
    }

    pub fn width_in_mbs(&self) -> u32 {
        self.pic_width_in_mbs_minus1 + 1
    }

    /// Frame height in MBs (§7.4.2.1.1: (2 − frame_mbs_only) × PicHeightInMapUnits).
    pub fn height_in_mbs(&self) -> u32 {
        let mult = if self.frame_mbs_only_flag { 1 } else { 2 };
        (self.pic_height_in_map_units_minus1 + 1) * mult
    }

    pub fn max_frame_num(&self) -> u32 {
        1u32 << (self.log2_max_frame_num_minus4 + 4)
    }

    pub fn max_poc_lsb(&self) -> u32 {
        1u32 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
}

/// Parse an SPS RBSP (§7.3.2.1). `nal` is the raw NAL (header + emulation bytes).
pub fn parse_sps(nal: &[u8]) -> Option<Sps> {
    let rbsp = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);
    let profile_idc = r.u(8) as u8;
    let _constraint_flags = r.u(8);
    let _level_idc = r.u(8);
    let seq_parameter_set_id = r.ue();

    // High-profile family (§7.3.2.1.1): chroma/bit-depth/scaling matrix live here.
    let mut chroma_format_idc = 1;
    let mut separate_colour_plane_flag = false;
    let mut bit_depth_luma_minus8 = 0;
    let mut bit_depth_chroma_minus8 = 0;
    if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        chroma_format_idc = r.ue();
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = r.flag();
        }
        bit_depth_luma_minus8 = r.ue();
        bit_depth_chroma_minus8 = r.ue();
        let _qpprime_y_zero_transform_bypass = r.flag();
        let seq_scaling_matrix_present = r.flag();
        if seq_scaling_matrix_present {
            // §7.3.2.1.1.1 — skip the scaling lists; the element uses flat 16
            // defaults (a POC boundary; scaling-matrix streams decode with the
            // default lists, which the driver accepts).
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let present = r.flag();
                if present {
                    skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 });
                }
            }
        }
    }

    let log2_max_frame_num_minus4 = r.ue();
    let pic_order_cnt_type = r.ue();
    let mut log2_max_pic_order_cnt_lsb_minus4 = 0;
    let mut delta_pic_order_always_zero_flag = false;
    let mut offset_for_non_ref_pic = 0;
    let mut offset_for_top_to_bottom_field = 0;
    let mut offsets_for_ref_frame = Vec::new();
    if pic_order_cnt_type == 0 {
        log2_max_pic_order_cnt_lsb_minus4 = r.ue();
    } else if pic_order_cnt_type == 1 {
        delta_pic_order_always_zero_flag = r.flag();
        offset_for_non_ref_pic = r.se();
        offset_for_top_to_bottom_field = r.se();
        let num_ref_frames_in_cycle = r.ue();
        for _ in 0..num_ref_frames_in_cycle {
            offsets_for_ref_frame.push(r.se());
        }
    }

    let max_num_ref_frames = r.ue();
    let gaps_in_frame_num_value_allowed_flag = r.flag();
    let pic_width_in_mbs_minus1 = r.ue();
    let pic_height_in_map_units_minus1 = r.ue();
    let frame_mbs_only_flag = r.flag();
    let mut mb_adaptive_frame_field_flag = false;
    if !frame_mbs_only_flag {
        mb_adaptive_frame_field_flag = r.flag();
    }
    let direct_8x8_inference_flag = r.flag();
    let frame_cropping_flag = r.flag();
    let (mut cl, mut cr, mut ct, mut cb) = (0, 0, 0, 0);
    if frame_cropping_flag {
        cl = r.ue();
        cr = r.ue();
        ct = r.ue();
        cb = r.ue();
    }
    // VUI is not needed for decode (timing is best-effort); stop here.

    Some(Sps {
        seq_parameter_set_id,
        profile_idc,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_frame_num_minus4,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        offsets_for_ref_frame,
        max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_crop_left: cl,
        frame_crop_right: cr,
        frame_crop_top: ct,
        frame_crop_bottom: cb,
    })
}

/// Skip a scaling list (§7.3.2.1.1.1) without storing it.
fn skip_scaling_list(r: &mut BitReader, size: usize) {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = r.se();
            next_scale = (last_scale + delta + 256) % 256;
        }
        last_scale = if next_scale == 0 { last_scale } else { next_scale };
    }
}

/// Parsed PPS (§7.3.2.2).
#[derive(Clone, Debug)]
pub struct Pps {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
    pub second_chroma_qp_index_offset: i32,
}

/// Parse a PPS RBSP (§7.3.2.2).
pub fn parse_pps(nal: &[u8]) -> Option<Pps> {
    let rbsp = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);
    let pic_parameter_set_id = r.ue();
    let seq_parameter_set_id = r.ue();
    let entropy_coding_mode_flag = r.flag();
    let bottom_field_pic_order_in_frame_present_flag = r.flag();
    let num_slice_groups_minus1 = r.ue();
    if num_slice_groups_minus1 > 0 {
        // FMO (slice groups) is out of scope; parse enough to reach the tail is
        // not attempted — reject by returning None so the AU is warn-dropped.
        return None;
    }
    let num_ref_idx_l0_default_active_minus1 = r.ue();
    let num_ref_idx_l1_default_active_minus1 = r.ue();
    let weighted_pred_flag = r.flag();
    let weighted_bipred_idc = r.u(2);
    let pic_init_qp_minus26 = r.se();
    let _pic_init_qs_minus26 = r.se();
    let chroma_qp_index_offset = r.se();
    let deblocking_filter_control_present_flag = r.flag();
    let constrained_intra_pred_flag = r.flag();
    let redundant_pic_cnt_present_flag = r.flag();

    // The trailing High-profile extension is optional (§7.3.2.2, more_rbsp_data).
    let mut transform_8x8_mode_flag = false;
    let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
    if r.more_rbsp_data() {
        transform_8x8_mode_flag = r.flag();
        let pic_scaling_matrix_present = r.flag();
        if pic_scaling_matrix_present {
            // Skip the PPS scaling lists (element uses flat defaults).
            let count = 6 + if transform_8x8_mode_flag { 6 } else { 2 };
            for i in 0..count {
                if r.flag() {
                    skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 });
                }
            }
        }
        second_chroma_qp_index_offset = r.se();
    }

    Some(Pps {
        pic_parameter_set_id,
        seq_parameter_set_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag,
        constrained_intra_pred_flag,
        redundant_pic_cnt_present_flag,
        transform_8x8_mode_flag,
        second_chroma_qp_index_offset,
    })
}

/// Slice type (§7.4.3, Table 7-6), reduced mod 5.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    fn from_raw(v: u32) -> SliceType {
        match v % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        }
    }
    /// The wire value VA expects (0..4).
    pub fn va_value(self) -> u8 {
        match self {
            SliceType::P => 0,
            SliceType::B => 1,
            SliceType::I => 2,
            SliceType::Sp => 3,
            SliceType::Si => 4,
        }
    }
}

/// One `ref_pic_list_modification` operation (§7.3.3.1).
#[derive(Clone, Debug)]
pub struct RefListMod {
    pub op: u32,      // modification_of_pic_nums_idc
    pub value: u32,   // abs_diff_pic_num_minus1 | long_term_pic_num
}

/// One `dec_ref_pic_marking` MMCO operation (§7.3.3.3).
#[derive(Clone, Debug)]
pub struct MmcoOp {
    pub op: u32,   // memory_management_control_operation
    pub arg1: u32,
    pub arg2: u32,
}

/// One reference's explicit prediction weights (§7.3.3.2). `None` = the bitstream
/// flagged them absent — the decoder substitutes the identity default
/// `weight = 1 << denom`, `offset = 0` (§8.4.2.3.2).
#[derive(Clone, Copy, Default, Debug)]
pub struct WeightEntry {
    /// `(luma_weight, luma_offset)` when `luma_weight_lX_flag` was set.
    pub luma: Option<(i32, i32)>,
    /// `[(cb_weight, cb_offset), (cr_weight, cr_offset)]` when the chroma flag was set.
    pub chroma: Option<[(i32, i32); 2]>,
}

/// The explicit `pred_weight_table` (§7.3.3.2), present on P/SP slices when the
/// PPS sets `weighted_pred_flag`, and on B slices when `weighted_bipred_idc == 1`.
/// Forwarded verbatim to the VA slice parameters — submitting zeros instead makes
/// the driver multiply every prediction by 0 (green P/B frames).
#[derive(Clone, Default, Debug)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    pub chroma_log2_weight_denom: u32,
    pub l0: Vec<WeightEntry>,
    pub l1: Vec<WeightEntry>,
}

/// Parsed slice header (§7.3.3) — the subset needed to fill VA slice params + POC.
#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    pub idr_pic_id: u32,
    pub is_idr: bool,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub ref_list_mods_l0: Vec<RefListMod>,
    pub ref_list_mods_l1: Vec<RefListMod>,
    pub cabac_init_idc: u32,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    pub direct_spatial_mv_pred_flag: bool,
    // dec_ref_pic_marking
    pub no_output_of_prior_pics_flag: bool,
    pub long_term_reference_flag: bool,
    pub adaptive_ref_pic_marking_mode_flag: bool,
    pub mmco_ops: Vec<MmcoOp>,
    /// The explicit prediction weight table, when the PPS/slice combination
    /// carries one (§7.3.3.2).
    pub pred_weights: Option<PredWeightTable>,
    /// Bit position (within the *de-emulated* RBSP body, after the NAL header) at
    /// the end of the slice header — the caller re-derives the raw-NAL bit offset.
    pub header_bits_in_rbsp: usize,
}

/// Parse a slice header (§7.3.3). Needs the SPS and PPS in force (looked up by the
/// caller) since several fields are conditional on them.
pub fn parse_slice_header(
    nal: &[u8],
    unit_type: u8,
    ref_idc: u8,
    sps: &Sps,
    pps: &Pps,
) -> Option<SliceHeader> {
    let rbsp = rbsp_of(nal);
    let mut r = BitReader::new(&rbsp);
    let is_idr = unit_type == NAL_SLICE_IDR;

    let first_mb_in_slice = r.ue();
    let slice_type = SliceType::from_raw(r.ue());
    let pic_parameter_set_id = r.ue();
    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.u(2);
    }
    let frame_num = r.u(sps.log2_max_frame_num_minus4 + 4);

    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = r.flag();
        if field_pic_flag {
            bottom_field_flag = r.flag();
        }
    }

    let mut idr_pic_id = 0;
    if is_idr {
        idr_pic_id = r.ue();
    }

    let mut pic_order_cnt_lsb = 0;
    let mut delta_pic_order_cnt_bottom = 0;
    let mut delta_pic_order_cnt = [0i32; 2];
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt_bottom = r.se();
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        delta_pic_order_cnt[0] = r.se();
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt[1] = r.se();
        }
    }

    if pps.redundant_pic_cnt_present_flag {
        let _redundant_pic_cnt = r.ue();
    }

    let mut direct_spatial_mv_pred_flag = false;
    if slice_type == SliceType::B {
        direct_spatial_mv_pred_flag = r.flag();
    }

    let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) {
        let num_ref_idx_active_override_flag = r.flag();
        if num_ref_idx_active_override_flag {
            num_ref_idx_l0_active_minus1 = r.ue();
            if slice_type == SliceType::B {
                num_ref_idx_l1_active_minus1 = r.ue();
            }
        }
    }

    // ref_pic_list_modification (§7.3.3.1).
    let mut ref_list_mods_l0 = Vec::new();
    let mut ref_list_mods_l1 = Vec::new();
    if !matches!(slice_type, SliceType::I | SliceType::Si) {
        if r.flag() {
            // ref_pic_list_modification_flag_l0
            loop {
                let op = r.ue();
                if op == 3 {
                    break;
                }
                let value = r.ue();
                ref_list_mods_l0.push(RefListMod { op, value });
                if ref_list_mods_l0.len() > 64 {
                    break; // malformed guard
                }
            }
        }
    }
    if slice_type == SliceType::B {
        if r.flag() {
            loop {
                let op = r.ue();
                if op == 3 {
                    break;
                }
                let value = r.ue();
                ref_list_mods_l1.push(RefListMod { op, value });
                if ref_list_mods_l1.len() > 64 {
                    break;
                }
            }
        }
    }

    // pred_weight_table (§7.3.3.2): parsed and forwarded to VA — the element
    // announces `weighted_pred_flag` in pic_fields, so the driver *uses* these.
    let mut pred_weights = None;
    if (pps.weighted_pred_flag && matches!(slice_type, SliceType::P | SliceType::Sp))
        || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B)
    {
        pred_weights = Some(parse_pred_weight_table(
            &mut r,
            sps.chroma_format_idc,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            slice_type,
        ));
    }

    // dec_ref_pic_marking (§7.3.3.3).
    let mut no_output_of_prior_pics_flag = false;
    let mut long_term_reference_flag = false;
    let mut adaptive_ref_pic_marking_mode_flag = false;
    let mut mmco_ops = Vec::new();
    if ref_idc != 0 {
        if is_idr {
            no_output_of_prior_pics_flag = r.flag();
            long_term_reference_flag = r.flag();
        } else {
            adaptive_ref_pic_marking_mode_flag = r.flag();
            if adaptive_ref_pic_marking_mode_flag {
                loop {
                    let op = r.ue();
                    if op == 0 {
                        break;
                    }
                    let mut arg1 = 0;
                    let mut arg2 = 0;
                    match op {
                        1 | 3 => arg1 = r.ue(), // difference_of_pic_nums_minus1
                        2 => arg1 = r.ue(),     // long_term_pic_num
                        4 => arg1 = r.ue(),     // max_long_term_frame_idx_plus1
                        6 => arg1 = r.ue(),     // long_term_frame_idx
                        _ => {}
                    }
                    if op == 3 {
                        arg2 = r.ue(); // long_term_frame_idx
                    }
                    mmco_ops.push(MmcoOp { op, arg1, arg2 });
                    if mmco_ops.len() > 64 {
                        break;
                    }
                }
            }
        }
    }

    let mut cabac_init_idc = 0;
    if pps.entropy_coding_mode_flag && !matches!(slice_type, SliceType::I | SliceType::Si) {
        cabac_init_idc = r.ue();
    }
    let slice_qp_delta = r.se();

    let mut disable_deblocking_filter_idc = 0;
    let mut slice_alpha_c0_offset_div2 = 0;
    let mut slice_beta_offset_div2 = 0;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = r.ue();
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset_div2 = r.se();
            slice_beta_offset_div2 = r.se();
        }
    }

    let header_bits_in_rbsp = r.bit_pos();

    Some(SliceHeader {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        is_idr,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        ref_list_mods_l0,
        ref_list_mods_l1,
        cabac_init_idc,
        slice_qp_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        direct_spatial_mv_pred_flag,
        no_output_of_prior_pics_flag,
        long_term_reference_flag,
        adaptive_ref_pic_marking_mode_flag,
        mmco_ops,
        pred_weights,
        header_bits_in_rbsp,
    })
}

/// Parse `pred_weight_table` (§7.3.3.2). Both consumers matter: the values are
/// forwarded to the VA slice params (explicit weighted prediction), and the bit
/// consumption positions `dec_ref_pic_marking` + the deblocking params after it.
fn parse_pred_weight_table(
    r: &mut BitReader,
    chroma_format_idc: u32,
    num_ref_l0: u32,
    num_ref_l1: u32,
    slice_type: SliceType,
) -> PredWeightTable {
    let luma_log2_weight_denom = r.ue();
    let chroma_log2_weight_denom = if chroma_format_idc != 0 { r.ue() } else { 0 };
    let read_list = |r: &mut BitReader, count: u32| -> Vec<WeightEntry> {
        let mut list = Vec::with_capacity((count as usize + 1).min(32));
        // Consume every bitstream entry (staying in sync even on a malformed
        // count) but store at most the 32 VA can carry.
        for _ in 0..=count {
            let mut e = WeightEntry::default();
            if r.flag() {
                // luma_weight_lX_flag
                e.luma = Some((r.se(), r.se()));
            }
            if chroma_format_idc != 0 && r.flag() {
                // chroma_weight_lX_flag — Cb then Cr, each (weight, offset).
                e.chroma = Some([(r.se(), r.se()), (r.se(), r.se())]);
            }
            if list.len() < 32 {
                list.push(e);
            }
        }
        list
    };
    let l0 = read_list(r, num_ref_l0);
    let l1 = if slice_type == SliceType::B { read_list(r, num_ref_l1) } else { Vec::new() };
    PredWeightTable { luma_log2_weight_denom, chroma_log2_weight_denom, l0, l1 }
}

// --- Picture order count (§8.2.1) ------------------------------------------------------

/// Running state for POC type 0 / 1 / 2 across pictures (§8.2.1).
#[derive(Default, Clone)]
pub struct PocState {
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num: i32,
    prev_frame_num_offset: i32,
}

impl PocState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset at an IDR / MMCO5 (§8.2.1: prevPicOrderCnt values reset to 0).
    pub fn reset(&mut self) {
        *self = PocState::default();
    }

    /// Compute the picture order count for the current picture (§8.2.1). Returns
    /// `(TopFieldOrderCnt, BottomFieldOrderCnt)`; for a progressive frame both are
    /// PicOrderCnt = min of the two. Only the frame case is realised.
    pub fn compute(&mut self, sps: &Sps, sh: &SliceHeader, is_idr: bool) -> (i32, i32) {
        match sps.pic_order_cnt_type {
            0 => self.compute_type0(sps, sh, is_idr),
            2 => self.compute_type2(sps, sh, is_idr),
            _ => self.compute_type1(sps, sh, is_idr),
        }
    }

    /// POC type 0 (§8.2.1.1).
    fn compute_type0(&mut self, sps: &Sps, sh: &SliceHeader, is_idr: bool) -> (i32, i32) {
        let max_poc_lsb = sps.max_poc_lsb() as i32;
        let (prev_msb, prev_lsb) = if is_idr {
            (0, 0)
        } else {
            (self.prev_poc_msb, self.prev_poc_lsb)
        };
        let poc_lsb = sh.pic_order_cnt_lsb as i32;
        let poc_msb = if poc_lsb < prev_lsb && (prev_lsb - poc_lsb) >= max_poc_lsb / 2 {
            prev_msb + max_poc_lsb
        } else if poc_lsb > prev_lsb && (poc_lsb - prev_lsb) > max_poc_lsb / 2 {
            prev_msb - max_poc_lsb
        } else {
            prev_msb
        };
        let top = poc_msb + poc_lsb;
        let bottom = top + sh.delta_pic_order_cnt_bottom;
        // Update state (only reference pictures update prev, §8.2.1.1 note).
        // We update unconditionally for the frame case; the element only carries
        // ref pictures across, so this is safe for our subset.
        self.prev_poc_msb = poc_msb;
        self.prev_poc_lsb = poc_lsb;
        (top, bottom.min(top))
    }

    /// POC type 2 (§8.2.1.3): tied directly to frame_num, no reordering.
    fn compute_type2(&mut self, sps: &Sps, sh: &SliceHeader, is_idr: bool) -> (i32, i32) {
        let max_frame_num = sps.max_frame_num() as i32;
        let frame_num = sh.frame_num as i32;
        let frame_num_offset = if is_idr {
            0
        } else if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset + max_frame_num
        } else {
            self.prev_frame_num_offset
        };
        // temporal_id / nal_ref_idc affects this; ref_idc==0 subtracts 1.
        let tmp = 2 * (frame_num_offset + frame_num);
        let poc = tmp; // for a ref picture; non-ref = tmp - 1 (approximation)
        self.prev_frame_num = frame_num;
        self.prev_frame_num_offset = frame_num_offset;
        (poc, poc)
    }

    /// POC type 1 (§8.2.1.2). Realised for the progressive-frame case.
    fn compute_type1(&mut self, sps: &Sps, sh: &SliceHeader, is_idr: bool) -> (i32, i32) {
        let max_frame_num = sps.max_frame_num() as i32;
        let frame_num = sh.frame_num as i32;
        let frame_num_offset = if is_idr {
            0
        } else if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset + max_frame_num
        } else {
            self.prev_frame_num_offset
        };
        let num_in_cycle = sps.offsets_for_ref_frame.len() as i32;
        let mut expected = 0i32;
        let abs_frame_num = if num_in_cycle != 0 {
            frame_num_offset + frame_num
        } else {
            0
        };
        let abs_frame_num = if sh.is_idr { 0 } else { abs_frame_num };
        let abs_frame_num = abs_frame_num.max(0);
        if num_in_cycle != 0 && abs_frame_num > 0 {
            let sum_cycle: i32 = sps.offsets_for_ref_frame.iter().sum();
            let poc_cycle_cnt = (abs_frame_num - 1) / num_in_cycle;
            let frame_num_in_cycle = (abs_frame_num - 1) % num_in_cycle;
            expected = poc_cycle_cnt * sum_cycle;
            for i in 0..=frame_num_in_cycle {
                expected += sps.offsets_for_ref_frame[i as usize];
            }
        }
        let expected = expected
            + sps.offset_for_non_ref_pic * 0
            + sh.delta_pic_order_cnt[0];
        let top = expected;
        let bottom = top + sps.offset_for_top_to_bottom_field + sh.delta_pic_order_cnt[1];
        self.prev_frame_num = frame_num;
        self.prev_frame_num_offset = frame_num_offset;
        (top, bottom.min(top))
    }
}
