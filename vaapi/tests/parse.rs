//! Pure H.264 bitstream-parse tests — no hardware, always run. They lock the
//! Annex-B framing, Exp-Golomb codecs, SPS/PPS parsing, and POC math independent of
//! any VA-API device (ITU-T H.264 clause references in `src/h264parse.rs`).

use pf_vaapi::h264parse::{
    self, parse_pps, parse_sps, split_nals, BitReader, PocState, SliceType,
};

// A tiny writer to hand-build RBSPs for the parser tests.
struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u32,
}
impl BitWriter {
    fn new() -> Self {
        BitWriter { bytes: Vec::new(), cur: 0, nbits: 0 }
    }
    fn bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | (b as u8 & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }
    fn u(&mut self, val: u32, n: u32) {
        for i in (0..n).rev() {
            self.bit((val >> i) & 1);
        }
    }
    fn ue(&mut self, val: u32) {
        // code_num = val; write leading zeros then the (val+1) binary.
        let code = val + 1;
        let len = 32 - code.leading_zeros();
        for _ in 0..(len - 1) {
            self.bit(0);
        }
        for i in (0..len).rev() {
            self.bit((code >> i) & 1);
        }
    }
    fn se(&mut self, val: i32) {
        let code = if val <= 0 {
            (-2 * val) as u32
        } else {
            (2 * val - 1) as u32
        };
        self.ue(code);
    }
    fn flag(&mut self, b: bool) {
        self.bit(b as u32);
    }
    /// Finish with an rbsp_stop_one_bit + byte align, then prepend a NAL header.
    fn finish_nal(mut self, nal_ref_idc: u8, nal_unit_type: u8) -> Vec<u8> {
        self.bit(1); // rbsp_stop_one_bit
        while self.nbits != 0 {
            self.bit(0);
        }
        let header = ((nal_ref_idc & 3) << 5) | (nal_unit_type & 0x1f);
        let mut out = vec![header];
        out.extend_from_slice(&self.bytes);
        out
    }
}

#[test]
fn exp_golomb_ue_round_trips() {
    for v in [0u32, 1, 2, 3, 7, 8, 15, 16, 100, 255, 1000] {
        let mut w = BitWriter::new();
        w.ue(v);
        w.bit(1); // stop
        while w.nbits != 0 {
            w.bit(0);
        }
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.ue(), v, "ue round-trip for {v}");
    }
}

#[test]
fn exp_golomb_se_round_trips() {
    for v in [0i32, 1, -1, 2, -2, 3, -3, 10, -10, 128, -128] {
        let mut w = BitWriter::new();
        w.se(v);
        w.bit(1);
        while w.nbits != 0 {
            w.bit(0);
        }
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.se(), v, "se round-trip for {v}");
    }
}

#[test]
fn nal_split_three_and_four_byte_start_codes() {
    // Two NALs: a 4-byte start code then a 3-byte start code.
    let stream = [
        0x00, 0x00, 0x00, 0x01, 0x67, 0xAA, 0xBB, // SPS (type 7)
        0x00, 0x00, 0x01, 0x68, 0xCC, // PPS (type 8)
    ];
    let nals = split_nals(&stream);
    assert_eq!(nals.len(), 2);
    assert_eq!(nals[0].unit_type, 7);
    assert_eq!(nals[0].ref_idc, 3); // 0x67 >> 5 = 3
    assert_eq!(nals[1].unit_type, 8);
    // The SPS raw should not include the 4-byte start code's trailing 0x00.
    assert_eq!(nals[0].raw, &[0x67, 0xAA, 0xBB]);
    assert_eq!(nals[1].raw, &[0x68, 0xCC]);
}

#[test]
fn nal_split_preserves_emulation_prevention_bytes() {
    // A NAL body containing 00 00 03 00 (an emulation-prevented 00 00 00).
    let stream = [0x00, 0x00, 0x01, 0x41, 0x00, 0x00, 0x03, 0x00, 0x99];
    let nals = split_nals(&stream);
    assert_eq!(nals.len(), 1);
    // The RAW NAL keeps the 0x03 (VA wants the original bitstream).
    assert_eq!(nals[0].raw, &[0x41, 0x00, 0x00, 0x03, 0x00, 0x99]);
}

/// Build a minimal Baseline SPS: 176x144 (QCIF), poc type 0.
fn build_sps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(66, 8); // profile_idc = 66 (Baseline, not high-profile family)
    w.u(0, 8); // constraint flags
    w.u(30, 8); // level_idc
    w.ue(0); // seq_parameter_set_id
    // (Baseline: no chroma/bit-depth block.)
    w.ue(0); // log2_max_frame_num_minus4 → MaxFrameNum 16
    w.ue(0); // pic_order_cnt_type = 0
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4 → MaxPocLsb 256
    w.ue(1); // max_num_ref_frames
    w.flag(false); // gaps_in_frame_num_value_allowed
    w.ue(10); // pic_width_in_mbs_minus1 → 11 MBs = 176 px
    w.ue(8); // pic_height_in_map_units_minus1 → 9 MUs
    w.flag(true); // frame_mbs_only_flag → height 9*16 = 144
    w.flag(true); // direct_8x8_inference_flag
    w.flag(false); // frame_cropping_flag
    w.flag(false); // vui_parameters_present_flag
    w.finish_nal(3, 7)
}

#[test]
fn sps_parse_dimensions_and_poc_type() {
    let nal = build_sps();
    let sps = parse_sps(&nal).expect("SPS parses");
    assert_eq!(sps.seq_parameter_set_id, 0);
    assert_eq!(sps.pic_order_cnt_type, 0);
    assert_eq!(sps.log2_max_frame_num_minus4, 0);
    assert_eq!(sps.max_frame_num(), 16);
    assert_eq!(sps.max_poc_lsb(), 256);
    assert_eq!(sps.width(), 176);
    assert_eq!(sps.height(), 144);
    assert_eq!(sps.width_in_mbs(), 11);
    assert_eq!(sps.height_in_mbs(), 9);
    assert!(sps.frame_mbs_only_flag);
    assert_eq!(sps.chroma_format_idc, 1);
    assert_eq!(sps.max_num_ref_frames, 1);
}

/// Build a minimal PPS: CAVLC, one ref, qp init 0.
fn build_pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // pic_parameter_set_id
    w.ue(0); // seq_parameter_set_id
    w.flag(false); // entropy_coding_mode_flag (CAVLC)
    w.flag(false); // bottom_field_pic_order_in_frame_present_flag
    w.ue(0); // num_slice_groups_minus1
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.flag(false); // weighted_pred_flag
    w.u(0, 2); // weighted_bipred_idc
    w.se(0); // pic_init_qp_minus26
    w.se(0); // pic_init_qs_minus26
    w.se(0); // chroma_qp_index_offset
    w.flag(false); // deblocking_filter_control_present_flag
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // redundant_pic_cnt_present_flag
    w.finish_nal(3, 8)
}

#[test]
fn pps_parse_basic() {
    let nal = build_pps();
    let pps = parse_pps(&nal).expect("PPS parses");
    assert_eq!(pps.pic_parameter_set_id, 0);
    assert_eq!(pps.seq_parameter_set_id, 0);
    assert!(!pps.entropy_coding_mode_flag);
    assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
    assert_eq!(pps.pic_init_qp_minus26, 0);
}

#[test]
fn slice_header_parse_idr() {
    let sps = parse_sps(&build_sps()).unwrap();
    let pps = parse_pps(&build_pps()).unwrap();
    // Build an IDR I-slice header: first_mb=0, slice_type=7 (I, all slices I),
    // pps_id=0, frame_num=0 (4 bits), idr_pic_id=0, poc_lsb=0 (8 bits).
    let mut w = BitWriter::new();
    w.ue(0); // first_mb_in_slice
    w.ue(7); // slice_type (I)
    w.ue(0); // pps_id
    w.u(0, 4); // frame_num (log2_max_frame_num_minus4+4 = 4 bits)
    w.ue(0); // idr_pic_id
    w.u(0, 8); // pic_order_cnt_lsb (8 bits)
    // dec_ref_pic_marking (IDR): no_output_of_prior_pics, long_term_reference
    w.flag(false);
    w.flag(false);
    w.se(0); // slice_qp_delta
    let nal = w.finish_nal(3, 5); // nal_unit_type 5 = IDR
    let sh = h264parse::parse_slice_header(&nal, 5, 3, &sps, &pps).expect("slice header");
    assert_eq!(sh.slice_type, SliceType::I);
    assert!(sh.is_idr);
    assert_eq!(sh.frame_num, 0);
    assert_eq!(sh.pic_order_cnt_lsb, 0);
    assert_eq!(sh.first_mb_in_slice, 0);
}

#[test]
fn poc_type0_ascending_sequence() {
    let sps = parse_sps(&build_sps()).unwrap();
    let pps = parse_pps(&build_pps()).unwrap();
    let mut poc = PocState::new();

    // IDR: poc_lsb 0 → POC 0.
    let idr = make_slice(&sps, &pps, 5, 3, 0, 0, SliceType::I);
    let (t, b) = poc.compute(&sps, &idr, true);
    assert_eq!(t.min(b), 0);

    // Following P frames with increasing poc_lsb → increasing POC (no MSB wrap).
    for (frame_num, lsb, expect) in [(1u32, 2u32, 2i32), (2, 4, 4), (3, 6, 6)] {
        let sh = make_slice(&sps, &pps, 1, 2, frame_num, lsb, SliceType::P);
        let (t, b) = poc.compute(&sps, &sh, false);
        assert_eq!(t.min(b), expect, "POC for lsb {lsb}");
    }
}

#[test]
fn poc_type0_msb_wraps() {
    let sps = parse_sps(&build_sps()).unwrap(); // MaxPocLsb = 256
    let pps = parse_pps(&build_pps()).unwrap();
    let mut poc = PocState::new();
    // IDR at lsb 0 → POC 0 (state: prev_lsb 0, prev_msb 0). Then climb in steps
    // below MaxPocLsb/2 so no wrap is inferred: 0 → 100 → 200.
    let idr = make_slice(&sps, &pps, 5, 3, 0, 0, SliceType::I);
    assert_eq!(poc.compute(&sps, &idr, true).0, 0);
    let p1 = make_slice(&sps, &pps, 1, 2, 1, 100, SliceType::P);
    assert_eq!(poc.compute(&sps, &p1, false).0, 100);
    let p2 = make_slice(&sps, &pps, 2, 2, 2, 200, SliceType::P);
    assert_eq!(poc.compute(&sps, &p2, false).0, 200);
    // lsb drops to 10 while prev_lsb is 200: a decrease of 190 ≥ MaxPocLsb/2 → a
    // forward MSB wrap → poc_msb = 0 + 256, POC = 256 + 10 = 266.
    let p3 = make_slice(&sps, &pps, 3, 2, 3, 10, SliceType::P);
    let (t1, b1) = poc.compute(&sps, &p3, false);
    assert_eq!(t1.min(b1), 266);
}

/// Build a slice header struct directly for POC tests (bypassing bit parsing).
fn make_slice(
    sps: &h264parse::Sps,
    _pps: &h264parse::Pps,
    unit_type: u8,
    ref_idc: u8,
    frame_num: u32,
    poc_lsb: u32,
    st: SliceType,
) -> h264parse::SliceHeader {
    // Round-trip through the real parser so the fields are exactly what decode uses.
    let mut w = BitWriter::new();
    w.ue(0); // first_mb
    w.ue(slice_type_code(st));
    w.ue(0); // pps_id
    w.u(frame_num, sps.log2_max_frame_num_minus4 + 4);
    if unit_type == 5 {
        w.ue(0); // idr_pic_id
    }
    w.u(poc_lsb, sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
    if st == SliceType::P {
        w.flag(false); // num_ref_idx_active_override_flag
        w.flag(false); // ref_pic_list_modification_flag_l0
    }
    if ref_idc != 0 {
        if unit_type == 5 {
            w.flag(false);
            w.flag(false);
        } else {
            w.flag(false); // adaptive_ref_pic_marking_mode_flag
        }
    }
    w.se(0); // slice_qp_delta
    let nal = w.finish_nal(ref_idc, unit_type);
    h264parse::parse_slice_header(&nal, unit_type, ref_idc, sps, _pps).expect("slice header parses")
}

fn slice_type_code(st: SliceType) -> u32 {
    match st {
        SliceType::P => 0,
        SliceType::B => 1,
        SliceType::I => 2,
        SliceType::Sp => 3,
        SliceType::Si => 4,
    }
}
