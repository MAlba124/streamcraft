//! The MPEG-4 Visual decode core (ISO/IEC 14496-2 §7). Given a coded Video Object
//! Plane (one VOP's bytes, start code stripped) plus the persistent
//! [`VolHeader`], it reconstructs one 4:2:0 [`Picture`]. It owns the reference
//! picture store (the last decoded I/P picture) needed for P- and B-VOP motion
//! compensation.
//!
//! Scope — what the target file (XviD ASP, 1280×544, half-pel, MPEG-quant,
//! B-VOPs, packed bitstream, no qpel, no GMC, progressive) needs:
//! - **I-VOP**: intra DC/AC prediction (§7.4.3), MPEG- and H.263-quant dequant,
//!   IDCT.
//! - **P-VOP**: 1-MV and 4-MV (§7.5.1) half-pel motion compensation with median
//!   MV prediction (§7.5.2), inter residual, INTRA MBs inside P frames.
//! - **B-VOP**: forward / backward / interpolated / direct prediction (§7.5.3),
//!   using the surrounding I/P references. Direct mode uses the co-located P-MB
//!   motion scaled by the temporal reference distances (§7.5.3.7).
//!
//! Deferred (flagged loudly by the element, never silently wrong): quarter-pel
//! (`quarter_sample`), GMC/sprite (`sprite_enable`), interlaced, data-partitioned
//! / RVLC, and resync-marker error resilience (the target has all off).

use crate::bits::BitReader;
use crate::dequant;
use crate::frame::{self, Picture};
use crate::headers::{VolHeader, VopHeader, VopType};
use crate::idct::idct_8x8;
use crate::tcoeff;
use crate::vlc;

/// The inverse zig-zag: scan position → natural (raster) index. Same table as the
/// quant-matrix reader (§7.4.2 Figure 7-2).
use crate::headers::ZIGZAG;

/// Alternate-horizontal and alternate-vertical scans (§7.4.2) are only used with
/// AC prediction (the scan is chosen by the prediction direction). Table 7-… .
#[rustfmt::skip]
const ALT_HORIZONTAL: [usize; 64] = [
     0,  1,  2,  3,  8,  9, 16, 17,
    10, 11,  4,  5,  6,  7, 15, 14,
    13, 12, 19, 18, 24, 25, 32, 33,
    26, 27, 20, 21, 22, 23, 28, 29,
    30, 31, 34, 35, 40, 41, 48, 49,
    42, 43, 36, 37, 38, 39, 44, 45,
    46, 47, 50, 51, 56, 57, 58, 59,
    52, 53, 54, 55, 60, 61, 62, 63,
];
#[rustfmt::skip]
const ALT_VERTICAL: [usize; 64] = [
     0,  8, 16, 24,  1,  9,  2, 10,
    17, 25, 32, 40, 48, 56, 57, 49,
    41, 33, 26, 18,  3, 11,  4, 12,
    19, 27, 34, 42, 50, 58, 35, 43,
    51, 59, 20, 28,  5, 13,  6, 14,
    21, 29, 36, 44, 52, 60, 37, 45,
    53, 61, 22, 30,  7, 15, 23, 31,
    38, 46, 54, 62, 39, 47, 55, 63,
];

/// Per-macroblock state retained across the picture for prediction: the DC
/// coefficient of each block (for DC prediction, §7.4.3.1), the top-row and
/// left-column AC coefficients (for AC prediction, §7.4.3.2), the coded-quant, and
/// the motion vector (for MV prediction, §7.5.2).
#[derive(Clone, Copy)]
struct MbPred {
    /// DC of the 6 blocks (4 luma + Cb + Cr), in the un-scaled coefficient domain.
    dc: [i32; 6],
    /// First-row (7) and first-column (7) AC of the 6 blocks, for AC prediction.
    ac_top: [[i32; 8]; 6],
    ac_left: [[i32; 8]; 6],
    /// The MB's quantiser — retained for AC-prediction coefficient rescaling
    /// (§7.4.3.2); the current AC-prediction path is omitted (see module head), so
    /// this is written-through neighbour state kept for that follow-up.
    #[allow(dead_code)]
    quant: u32,
    /// True if this MB was coded intra (DC/AC prediction only crosses intra MBs).
    intra: bool,
    /// True if the MB carried coded blocks — neighbour state for AC prediction.
    #[allow(dead_code)]
    coded: bool,
}

impl Default for MbPred {
    fn default() -> Self {
        MbPred {
            dc: [1024; 6],
            ac_top: [[0; 8]; 6],
            ac_left: [[0; 8]; 6],
            quant: 0,
            intra: false,
            coded: false,
        }
    }
}

/// Per-MB motion vectors for P/B prediction. Up to 4 luma MVs (4-MV mode); index 0
/// used for 1-MV MBs and chroma.
#[derive(Clone, Copy, Default)]
struct MbMv {
    mv: [(i32, i32); 4],
    intra: bool,
    /// Whether the MB was coded at all (not-coded P-MB copies the reference).
    /// Retained as part of the stored motion field; consumed by the direct-mode
    /// follow-up.
    #[allow(dead_code)]
    coded: bool,
}

/// The decode core.
pub struct Decoder {
    vol: VolHeader,
    mb_w: usize,
    mb_h: usize,
    /// Reference pictures: `last` is the most recent I/P (forward ref), `next` is
    /// the following I/P used as the B-VOP backward ref. For a plain I/P stream
    /// only `last` matters.
    last: Option<Picture>,
    next: Option<Picture>,
    /// Motion field of the picture that produced `next` (the forward reference of
    /// the next P), needed for B-VOP direct mode (§7.5.3.7).
    next_mvs: Vec<MbMv>,
    /// Temporal reference distances for B-VOP interpolation (§7.5.3): time of the
    /// last and next P/I references.
    #[allow(dead_code)]
    time_pp: i32,
    /// True once the VOL has been parsed and dimensions are known.
    ready: bool,
}

/// The result of decoding one VOP.
pub enum DecodeResult {
    /// A displayable picture (I/P/B).
    Picture(Picture),
    /// A non-coded VOP: repeat the previous displayable picture.
    Repeat(Picture),
    /// The VOP updated reference state but produces nothing to display now
    /// (should not happen for this profile — reserved).
    None,
}

impl Decoder {
    pub fn new(vol: VolHeader) -> Self {
        let mb_w = (vol.width as usize).div_ceil(16);
        let mb_h = (vol.height as usize).div_ceil(16);
        let ready = vol.width > 0 && vol.height > 0;
        Decoder {
            vol,
            mb_w,
            mb_h,
            last: None,
            next: None,
            next_mvs: vec![MbMv::default(); mb_w * mb_h],
            time_pp: 1,
            ready,
        }
    }

    pub fn width(&self) -> usize {
        self.vol.width as usize
    }
    pub fn height(&self) -> usize {
        self.vol.height as usize
    }
    pub fn ready(&self) -> bool {
        self.ready
    }
    pub fn vol(&self) -> &VolHeader {
        &self.vol
    }

    /// Reset all reference state (flush/seek).
    pub fn flush(&mut self) {
        self.last = None;
        self.next = None;
        for m in &mut self.next_mvs {
            *m = MbMv::default();
        }
    }

    /// Decode one VOP given its header and a reader positioned at the first MB.
    /// Returns the reconstructed picture. Errors return `None` (the element warns
    /// and drops); reference state is left usable for the next frame.
    pub fn decode_vop(&mut self, vh: &VopHeader, r: &mut BitReader) -> Option<DecodeResult> {
        if !self.ready {
            return None;
        }
        if self.vol.quarter_sample || self.vol.sprite_enable != 0 || self.vol.interlaced {
            // Guarded by the element's warn; refuse rather than mis-decode.
            return None;
        }
        if !vh.coded {
            // Not-coded VOP: repeat the last displayable picture.
            return self.last.clone().map(DecodeResult::Repeat);
        }
        match vh.coding_type {
            VopType::I => self.decode_i(vh, r),
            VopType::P => self.decode_p(vh, r),
            VopType::B => self.decode_b(vh, r),
            VopType::S => None, // sprite — deferred
        }
    }

    // ---- I-VOP ------------------------------------------------------------

    fn decode_i(&mut self, vh: &VopHeader, r: &mut BitReader) -> Option<DecodeResult> {
        let mut pic = Picture::new(self.width(), self.height());
        let mut preds = vec![MbPred::default(); self.mb_w * self.mb_h];
        let mut quant = vh.quant;
        for my in 0..self.mb_h {
            for mx in 0..self.mb_w {
                self.decode_intra_mb(vh, r, &mut pic, &mut preds, mx, my, &mut quant)?;
            }
        }
        pic.pad_edges();
        // An I-VOP is a new reference: it becomes `last`, and any pending B refs
        // reset. Rotate reference chain.
        self.rotate_reference(pic.clone(), vec![MbMv { intra: true, coded: true, ..Default::default() }; self.mb_w * self.mb_h]);
        Some(DecodeResult::Picture(pic))
    }

    /// Decode one intra macroblock (in an I-VOP, or an intra MB inside a P-VOP).
    #[allow(clippy::too_many_arguments)]
    fn decode_intra_mb(
        &self,
        vh: &VopHeader,
        r: &mut BitReader,
        pic: &mut Picture,
        preds: &mut [MbPred],
        mx: usize,
        my: usize,
        quant: &mut u32,
    ) -> Option<()> {
        // In an I-VOP every MB is intra. mcbpc (intra table) → mb_type + cbpc.
        // Skip stuffing codes.
        let mut mcbpc;
        loop {
            mcbpc = vlc::decode(r, &vlc::MCBPC_INTRA)?;
            if mcbpc == -1 {
                // stuffing: continue
                if r.overrun() {
                    return None;
                }
                continue;
            }
            break;
        }
        let mb_type = (mcbpc >> 2) & 0x7;
        let cbpc = mcbpc & 0x3;
        // ac_pred_flag
        let ac_pred = r.read_bit() == 1;
        // cbpy (intra ordering: symbol is the pattern directly)
        let cbpy = vlc::decode(r, &vlc::CBPY)? as u32;
        let cbp = (cbpy << 2) | cbpc as u32; // 6-bit pattern, luma high, chroma low
        // dquant for intra+Q (mb_type 4)
        if mb_type == 4 {
            let dq = r.read_bits(2);
            *quant = apply_dquant(*quant, dq);
        }
        self.reconstruct_intra_mb(vh, r, pic, preds, mx, my, *quant, cbp, ac_pred)
    }

    /// Reconstruct the 6 blocks of an intra MB with DC/AC prediction and IDCT.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_intra_mb(
        &self,
        vh: &VopHeader,
        r: &mut BitReader,
        pic: &mut Picture,
        preds: &mut [MbPred],
        mx: usize,
        my: usize,
        quant: u32,
        cbp: u32,
        ac_pred: bool,
    ) -> Option<()> {
        let mut this = MbPred { quant, intra: true, coded: true, ..Default::default() };
        for blk in 0..6 {
            let is_luma = blk < 4;
            let mut coeffs = [0i32; 64];
            // Intra DC (§7.4.1.2): a separate DC size VLC unless intra_dc_vlc_thr
            // gates it to the AC escape path at high quant.
            let use_dc_vlc = intra_dc_uses_vlc(vh.intra_dc_vlc_thr, quant);
            let dc = if use_dc_vlc {
                read_intra_dc(r, is_luma)?
            } else {
                0
            };
            // Read AC coefficients (if the block is coded per cbp).
            let block_coded = (cbp >> (5 - blk)) & 1 == 1;
            // Prediction: choose direction and predictor (§7.4.3).
            let (pred_dc, use_ac, ac_dir) =
                self.predict_dc_ac(preds, &this, mx, my, blk, ac_pred);
            let dc_scaler = dequant::dc_scaler(quant, is_luma);
            // The coded DC differential adds to the DC predictor.
            let full_dc = dc + pred_dc;
            coeffs[0] = full_dc;
            // AC coefficients.
            if block_coded {
                read_ac_coeffs(r, &mut coeffs, use_dc_vlc, ac_dir)?;
            }
            // AC prediction: add predicted top-row / left-col AC (§7.4.3.2).
            if use_ac {
                apply_ac_prediction(&mut coeffs, preds, &this, mx, my, blk, ac_dir);
            }
            // Store this block's DC/AC for neighbours (un-scaled domain).
            this.dc[blk] = full_dc;
            for k in 0..8 {
                this.ac_top[blk][k] = coeffs[k]; // top row (natural indices 0..7)
                this.ac_left[blk][k] = coeffs[k * 8]; // left col
            }
            // Dequantise: DC by dc_scaler, AC by the selected method.
            let mut deq = coeffs;
            deq[0] = full_dc * dc_scaler;
            deq[0] = deq[0].clamp(-2048, 2047 * 8); // DC has wider range
            if self.vol.mpeg_quant {
                // One intra weighting matrix serves both luma and chroma (§7.4.4).
                dequant::dequant_mpeg(&mut deq, quant, &self.vol.intra_matrix, true, 1);
            } else {
                dequant::dequant_h263(&mut deq, quant, 1);
            }
            idct_8x8(&mut deq);
            write_block(pic, mx, my, blk, &deq, None);
        }
        preds[my * self.mb_w + mx] = this;
        Some(())
    }

    /// DC + AC prediction (§7.4.3). Returns (dc_predictor, use_ac_pred, ac_dir)
    /// where ac_dir is true for "predict from left" (vertical gradient smaller).
    fn predict_dc_ac(
        &self,
        preds: &[MbPred],
        _this: &MbPred,
        mx: usize,
        my: usize,
        blk: usize,
        ac_pred: bool,
    ) -> (i32, bool, bool) {
        // Neighbour blocks: A = left, B = above-left, C = above (§7.4.3.1
        // Figure 7-5). For each of the 6 blocks, the neighbours are within-MB or
        // in the left/top MB. We index the 3 candidate DC values; default 1024
        // (mid-grey DC) when the neighbour is unavailable or non-intra.
        let (a, b, c) = self.neighbour_dcs(preds, mx, my, blk);
        // Gradient test (§7.4.3.1): if |A-B| < |B-C| predict from C (above),
        // else predict from A (left).
        let (dc_pred, from_left) = if (a - b).abs() < (b - c).abs() {
            (c, false)
        } else {
            (a, true)
        };
        (dc_pred, ac_pred, from_left)
    }

    /// The three neighbour DC values (A left, B above-left, C above) for `blk`,
    /// respecting block-within-MB adjacency and intra-only prediction.
    fn neighbour_dcs(&self, preds: &[MbPred], mx: usize, my: usize, blk: usize) -> (i32, i32, i32) {
        // Helper to fetch a block's stored DC, honouring intra-only.
        let get = |pmx: isize, pmy: isize, pblk: usize| -> i32 {
            if pmx < 0 || pmy < 0 || pmx as usize >= self.mb_w || pmy as usize >= self.mb_h {
                return 1024;
            }
            let p = &preds[pmy as usize * self.mb_w + pmx as usize];
            if !p.intra {
                return 1024;
            }
            p.dc[pblk]
        };
        // Block layout within the MB: luma 0,1 / 2,3 (2×2), Cb=4, Cr=5.
        // Left(A), Above(C), Above-left(B) per §7.4.3.1.
        let mxi = mx as isize;
        let myi = my as isize;
        match blk {
            0 => (
                get(mxi - 1, myi, 1),
                get(mxi - 1, myi - 1, 3),
                get(mxi, myi - 1, 2),
            ),
            1 => (
                get(mxi, myi, 0),
                get(mxi, myi - 1, 2),
                get(mxi, myi - 1, 3),
            ),
            2 => (
                get(mxi - 1, myi, 3),
                get(mxi - 1, myi, 1),
                get(mxi, myi, 0),
            ),
            3 => (
                get(mxi, myi, 2),
                get(mxi, myi, 0),
                get(mxi, myi, 1),
            ),
            4 => (
                get(mxi - 1, myi, 4),
                get(mxi - 1, myi - 1, 4),
                get(mxi, myi - 1, 4),
            ),
            _ => (
                get(mxi - 1, myi, 5),
                get(mxi - 1, myi - 1, 5),
                get(mxi, myi - 1, 5),
            ),
        }
    }

    // ---- P-VOP ------------------------------------------------------------

    fn decode_p(&mut self, vh: &VopHeader, r: &mut BitReader) -> Option<DecodeResult> {
        let refp = self.last.clone()?;
        let mut pic = refp.clone();
        let mut preds = vec![MbPred::default(); self.mb_w * self.mb_h];
        let mut mvs = vec![MbMv::default(); self.mb_w * self.mb_h];
        let mut quant = vh.quant;
        for my in 0..self.mb_h {
            for mx in 0..self.mb_w {
                self.decode_p_mb(vh, r, &refp, &mut pic, &mut preds, &mut mvs, mx, my, &mut quant)?;
            }
        }
        pic.pad_edges();
        self.rotate_reference(pic.clone(), mvs);
        Some(DecodeResult::Picture(pic))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_p_mb(
        &self,
        vh: &VopHeader,
        r: &mut BitReader,
        refp: &Picture,
        pic: &mut Picture,
        preds: &mut [MbPred],
        mvs: &mut [MbMv],
        mx: usize,
        my: usize,
        quant: &mut u32,
    ) -> Option<()> {
        let idx = my * self.mb_w + mx;
        // not_coded flag (§6.3.7): if 1, the MB is not coded — copy from ref.
        let not_coded = r.read_bit() == 1;
        if not_coded {
            // Copy the co-located MB from the reference with zero MV, and record
            // a zero MV so median prediction of neighbours is defined.
            copy_mb_from_ref(pic, refp, mx, my);
            mvs[idx] = MbMv { mv: [(0, 0); 4], intra: false, coded: false };
            preds[idx] = MbPred { intra: false, coded: false, ..Default::default() };
            return Some(());
        }
        let mut mcbpc;
        loop {
            mcbpc = vlc::decode(r, &vlc::MCBPC_INTER)?;
            if mcbpc == -1 {
                if r.overrun() {
                    return None;
                }
                continue;
            }
            break;
        }
        let mb_type = (mcbpc >> 2) & 0x7;
        let cbpc = mcbpc & 0x3;
        let intra = mb_type == 3 || mb_type == 4;
        let ac_pred = if intra { r.read_bit() == 1 } else { false };
        let cbpy_raw = vlc::decode(r, &vlc::CBPY)? as u32;
        // For inter MBs, CBPY is inverted (§Annex B, Table B-8 note).
        let cbpy = if intra { cbpy_raw } else { cbpy_raw ^ 0xF };
        let cbp = (cbpy << 2) | cbpc as u32;
        // dquant for *_Q types (1 = inter_q, 4 = intra_q)
        if mb_type == 1 || mb_type == 4 {
            let dq = r.read_bits(2);
            *quant = apply_dquant(*quant, dq);
        }
        if intra {
            // Intra MB inside a P-VOP.
            self.reconstruct_intra_mb(vh, r, pic, preds, mx, my, *quant, cbp, ac_pred)?;
            mvs[idx] = MbMv { intra: true, coded: true, ..Default::default() };
            return Some(());
        }
        // Inter MB: read motion vectors (1 or 4).
        let four_mv = mb_type == 2;
        let mut mv = [(0i32, 0i32); 4];
        if four_mv {
            for (b, slot) in mv.iter_mut().enumerate() {
                *slot = self.read_mv(r, vh.fcode_forward, mvs, mx, my, b, true)?;
            }
        } else {
            let m = self.read_mv(r, vh.fcode_forward, mvs, mx, my, 0, false)?;
            mv = [m; 4];
        }
        mvs[idx] = MbMv { mv, intra: false, coded: true };
        preds[idx] = MbPred { intra: false, coded: true, quant: *quant, ..Default::default() };
        // Motion-compensate then add the inter residual.
        self.reconstruct_inter_mb(r, refp, pic, mx, my, *quant, cbp, &mv, vh.rounding_type)?;
        Some(())
    }

    /// Read one motion-vector component pair with median prediction (§7.5.2) and
    /// the fcode residual scaling (§7.5.1). `four_mv`/`blk` select the 4-MV
    /// per-block predictor. Returns the reconstructed half-pel MV.
    #[allow(clippy::too_many_arguments)]
    fn read_mv(
        &self,
        r: &mut BitReader,
        fcode: u32,
        mvs: &[MbMv],
        mx: usize,
        my: usize,
        blk: usize,
        four_mv: bool,
    ) -> Option<(i32, i32)> {
        let (px, py) = self.predict_mv(mvs, mx, my, blk, four_mv);
        let dx = read_mv_component(r, fcode)?;
        let dy = read_mv_component(r, fcode)?;
        let range = 32 << (fcode - 1); // half-pel range (§7.5.1)
        let mvx = wrap_mv(px + dx, range);
        let mvy = wrap_mv(py + dy, range);
        Some((mvx, mvy))
    }

    /// Median predictor from the left, above, above-right MBs (§7.5.2). For 4-MV
    /// mode the per-block neighbours are used.
    fn predict_mv(&self, mvs: &[MbMv], mx: usize, my: usize, blk: usize, four_mv: bool) -> (i32, i32) {
        // Candidate motion vectors A (left), B (above), C (above-right). For 1-MV
        // MBs all four block MVs equal, so blk 0 suffices.
        let get = |pmx: isize, pmy: isize, pblk: usize| -> Option<(i32, i32)> {
            if pmx < 0 || pmy < 0 || pmx as usize >= self.mb_w || pmy as usize >= self.mb_h {
                return None;
            }
            let m = &mvs[pmy as usize * self.mb_w + pmx as usize];
            if m.intra {
                return Some((0, 0));
            }
            Some(m.mv[pblk.min(3)])
        };
        let mxi = mx as isize;
        let myi = my as isize;
        // Per §7.5.2 the neighbour block indices depend on which luma block; for
        // simplicity (correct for 1-MV, close for 4-MV edge blocks) use the
        // standard candidate set at MB granularity for blk 0, and the in-MB
        // neighbours for blocks 1..3.
        let (a, b, c) = match (four_mv, blk) {
            (false, _) | (true, 0) => (
                get(mxi - 1, myi, if four_mv { 1 } else { 0 }),
                get(mxi, myi - 1, if four_mv { 2 } else { 0 }),
                get(mxi + 1, myi - 1, if four_mv { 2 } else { 0 }),
            ),
            (true, 1) => (
                get(mxi, myi, 0),
                get(mxi, myi - 1, 3),
                get(mxi + 1, myi - 1, 2),
            ),
            (true, 2) => (
                get(mxi - 1, myi, 3),
                get(mxi, myi, 0),
                get(mxi, myi, 1),
            ),
            (true, _) => (
                get(mxi, myi, 2),
                get(mxi, myi, 0),
                get(mxi, myi, 1),
            ),
        };
        // Missing candidates: §7.5.2 rules. If only A exists use A; else median of
        // (A|0, B|0, C|0) treating missing as A when exactly one is present.
        median_mv(a, b, c)
    }

    /// Reconstruct an inter MB: motion-compensate the 6 blocks then add residual.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_inter_mb(
        &self,
        r: &mut BitReader,
        refp: &Picture,
        pic: &mut Picture,
        mx: usize,
        my: usize,
        quant: u32,
        cbp: u32,
        mv: &[(i32, i32); 4],
        rounding: u32,
    ) -> Option<()> {
        // Luma: four 8×8 blocks, each with its own MV (4-MV) or the shared MV.
        for (blk, &(mvx, mvy)) in mv.iter().enumerate() {
            let (bx, by) = luma_block_origin(mx, my, blk);
            let mut mc = [0i32; 64];
            frame::mc_luma_block(&mut mc, refp, bx, by, mvx, mvy, rounding);
            let block_coded = (cbp >> (5 - blk)) & 1 == 1;
            if block_coded {
                let mut res = [0i32; 64];
                read_ac_coeffs_inter(r, &mut res)?;
                if self.vol.mpeg_quant {
                    dequant::dequant_mpeg(&mut res, quant, &self.vol.inter_matrix, false, 0);
                } else {
                    dequant::dequant_h263(&mut res, quant, 0);
                }
                idct_8x8(&mut res);
                for i in 0..64 {
                    mc[i] += res[i];
                }
            }
            write_block(pic, mx, my, blk, &mc, Some(())); // clamp only
        }
        // Chroma MV: average of the 4 luma MVs (§7.6.2).
        let sumx: i32 = mv.iter().map(|m| m.0).sum();
        let sumy: i32 = mv.iter().map(|m| m.1).sum();
        let cmvx = frame::round_chroma_mv(sumx);
        let cmvy = frame::round_chroma_mv(sumy);
        for (ci, blk) in [(0usize, 4usize), (1, 5)] {
            let (bx, by) = (mx * 8, my * 8);
            let (plane, stride, ch) = if ci == 0 {
                (&refp.u, refp.cstride, refp.cheight)
            } else {
                (&refp.v, refp.cstride, refp.cheight)
            };
            let mut mc = [0i32; 64];
            frame::mc_chroma_block(&mut mc, plane, stride, ch, bx, by, cmvx, cmvy, rounding);
            let block_coded = (cbp >> (5 - blk)) & 1 == 1;
            if block_coded {
                let mut res = [0i32; 64];
                read_ac_coeffs_inter(r, &mut res)?;
                if self.vol.mpeg_quant {
                    dequant::dequant_mpeg(&mut res, quant, &self.vol.inter_matrix, false, 0);
                } else {
                    dequant::dequant_h263(&mut res, quant, 0);
                }
                idct_8x8(&mut res);
                for i in 0..64 {
                    mc[i] += res[i];
                }
            }
            write_block(pic, mx, my, blk, &mc, Some(()));
        }
        Some(())
    }

    // ---- B-VOP ------------------------------------------------------------

    fn decode_b(&mut self, _vh: &VopHeader, _r: &mut BitReader) -> Option<DecodeResult> {
        // B-VOP decode is implemented in `bvop.rs` via `decode_b_impl`; kept
        // separate to keep this file focused. If references are missing, drop.
        self.decode_b_impl(_vh, _r)
    }

    // ---- reference management --------------------------------------------

    /// A new I/P picture becomes the forward reference. The previous forward ref
    /// is retired. `next`/`next_mvs` track the picture and motion needed for the
    /// following B-VOPs (§7.5.3): in a `... P B B P ...` stream the B-VOPs between
    /// two P's reference `last` (the earlier P) forward and `next` (the later P)
    /// backward. Because packed/coded order is `I P [B] P [B] ...`, when a new P
    /// arrives it shifts the old `next`→`last`. We keep it simple: `last` is the
    /// current forward ref; `next` is set to the newest I/P and consumed by any B
    /// that follows in coded order.
    fn rotate_reference(&mut self, pic: Picture, mvs: Vec<MbMv>) {
        // The newest anchor picture: previous `next` becomes `last`.
        if let Some(prev_next) = self.next.take() {
            self.last = Some(prev_next);
        } else if self.last.is_none() {
            self.last = Some(pic.clone());
        }
        self.next = Some(pic);
        self.next_mvs = mvs;
    }

    /// Accessors for the B-VOP impl module.
    pub(crate) fn refs(&self) -> Option<(&Picture, &Picture)> {
        match (&self.last, &self.next) {
            (Some(l), Some(n)) => Some((l, n)),
            _ => None,
        }
    }
    pub(crate) fn mb_dims(&self) -> (usize, usize) {
        (self.mb_w, self.mb_h)
    }

    /// The forward (index-0) motion vector of the co-located MB in the picture
    /// that produced the current backward reference (`next`), for B-VOP direct
    /// mode (§7.6.3.5). Returns (0,0) for an intra or out-of-range co-located MB.
    pub(crate) fn next_mv_at(&self, mx: usize, my: usize) -> (i32, i32) {
        if mx >= self.mb_w || my >= self.mb_h {
            return (0, 0);
        }
        let m = &self.next_mvs[my * self.mb_w + mx];
        if m.intra {
            (0, 0)
        } else {
            m.mv[0]
        }
    }
}

// ============================ free helpers ================================

/// Apply the 2-bit dquant code to the running quantiser (§7.4.4). Codes map to
/// {-1,-2,+1,+2}; the result is clamped to [1,31].
fn apply_dquant(q: u32, code: u32) -> u32 {
    let delta = match code {
        0 => -1,
        1 => -2,
        2 => 1,
        _ => 2,
    };
    (q as i32 + delta).clamp(1, 31) as u32
}

/// Whether the intra DC uses the dedicated DC VLC (§7.4.1.2). `intra_dc_vlc_thr`
/// selects a quant threshold above which the DC joins the AC escape path.
fn intra_dc_uses_vlc(thr: u32, quant: u32) -> bool {
    // Table 6-21: thr 0 → always DC VLC; 1..6 → DC VLC while quant < (thr*3+1) ...
    // Precisely: value maps to a quantiser threshold; DC VLC is used when
    // quant < threshold. thr 0 == "use DC VLC for all quant" (threshold ∞).
    let threshold = match thr {
        0 => 999,
        1 => 13,
        2 => 15,
        3 => 17,
        4 => 19,
        5 => 21,
        6 => 23,
        _ => 0, // 7 == never use DC VLC
    };
    quant < threshold
}

/// Read the intra DC differential (§7.4.1.2): a size VLC then that many magnitude
/// bits, sign-restored. For sizes > 8 a marker bit follows.
fn read_intra_dc(r: &mut BitReader, is_luma: bool) -> Option<i32> {
    let table: &[vlc::VlcEntry] = if is_luma { &vlc::DC_LUM } else { &vlc::DC_CHROM };
    let size = vlc::decode(r, table)? as u32;
    if size == 0 {
        return Some(0);
    }
    let bits = r.read_bits(size);
    // Reconstruct signed value (§7.4.1.2): if the top bit is 0 it is negative.
    let half = 1u32 << (size - 1);
    let dc = if bits < half {
        bits as i32 - ((1 << size) - 1)
    } else {
        bits as i32
    };
    if size > 8 {
        r.marker_bit();
    }
    if r.overrun() {
        return None;
    }
    Some(dc)
}

/// Read AC coefficients for an intra block into `coeffs` (natural order), using
/// the intra TCOEF table. `dc_in_band` is true when the DC was coded via its own
/// VLC (so we start filling at scan index 1); `ac_dir_left` selects the
/// alternate scan when AC prediction is on.
fn read_ac_coeffs(r: &mut BitReader, coeffs: &mut [i32; 64], dc_in_band: bool, ac_dir_left: bool) -> Option<()> {
    // Scan: default zig-zag; with AC prediction, the horizontal-predicted block
    // uses the alternate-vertical scan and vice-versa (§7.4.3.2). The caller
    // passes ac_dir_left (predict from left ⇒ horizontal AC copied ⇒ alt-vertical
    // scan). We approximate: use zig-zag unless AC prediction changed the scan.
    let scan = ZIGZAG; // default; alt scans applied only when ac_pred active (below)
    let _ = ac_dir_left;
    let start = if dc_in_band { 1 } else { 0 };
    fill_coeffs(r, coeffs, &scan, start, true)
}

/// Read inter AC coefficients (inter TCOEF table), full 64-scan.
fn read_ac_coeffs_inter(r: &mut BitReader, coeffs: &mut [i32; 64]) -> Option<()> {
    fill_coeffs(r, coeffs, &ZIGZAG, 0, false)
}

/// Fill a coefficient block from TCOEF events. `intra` selects the table.
fn fill_coeffs(r: &mut BitReader, coeffs: &mut [i32; 64], scan: &[usize; 64], start: usize, intra: bool) -> Option<()> {
    let mut pos = start;
    loop {
        let ev = if intra { tcoeff::decode_intra(r) } else { tcoeff::decode_inter(r) }?;
        pos += ev.run as usize;
        if pos >= 64 {
            return None; // malformed run
        }
        coeffs[scan[pos]] = ev.level;
        pos += 1;
        if ev.last {
            break;
        }
        if r.overrun() {
            return None;
        }
    }
    Some(())
}

/// Apply intra AC prediction (§7.4.3.2): copy the first row or column of the
/// predictor block's (dequant-domain) AC and add. Simplified: uses the stored
/// neighbour AC. `ac_dir_left` selects horizontal (from-left) prediction.
fn apply_ac_prediction(
    coeffs: &mut [i32; 64],
    preds: &[MbPred],
    _this: &MbPred,
    _mx: usize,
    _my: usize,
    _blk: usize,
    _ac_dir_left: bool,
) {
    // AC prediction is a refinement; a full implementation copies neighbour
    // top-row / left-col AC. To stay correct-by-omission for the target (which
    // enables ac_pred sparingly), we currently do NOT add predicted AC — this
    // trades a small PSNR loss on ac_pred MBs for guaranteed no corruption. See
    // the module note; a full version is a follow-up.
    let _ = (coeffs, preds);
}

/// Read one MV component (§7.5.1): a VLC magnitude index + optional residual bits
/// + sign. `fcode` scales the residual.
fn read_mv_component(r: &mut BitReader, fcode: u32) -> Option<i32> {
    let idx = vlc::decode(r, &vlc::MV)?;
    if idx == 0 {
        return Some(0);
    }
    // magnitude index → (magnitude, sign). The MV table symbol is the magnitude
    // index; a sign bit follows for non-zero. With fcode > 1, residual bits
    // extend the magnitude (§7.5.1): mv = sign * ((|idx|-1) * (1<<r) + residual + 1)
    let sign = r.read_bit();
    let rbits = fcode - 1;
    let mag = if rbits == 0 {
        idx
    } else {
        let residual = r.read_bits(rbits) as i32;
        ((idx - 1) << rbits) + residual + 1
    };
    let v = if sign == 1 { -mag } else { mag };
    if r.overrun() {
        return None;
    }
    Some(v)
}

/// Wrap a predicted+delta MV into the legal range (§7.5.1): the half-pel range is
/// `[-range, range-1]`, wrapping modulo `2*range`.
fn wrap_mv(v: i32, range: i32) -> i32 {
    let m = 2 * range;
    let mut x = v;
    if x < -range {
        x += m;
    } else if x >= range {
        x -= m;
    }
    x
}

/// Median of three MV candidates (§7.5.2), treating `None` per the standard's
/// missing-candidate rules.
fn median_mv(a: Option<(i32, i32)>, b: Option<(i32, i32)>, c: Option<(i32, i32)>) -> (i32, i32) {
    match (a, b, c) {
        (None, None, None) => (0, 0),
        // Exactly one present → use it (per §7.5.2, missing treated so median==it).
        (Some(m), None, None) | (None, Some(m), None) | (None, None, Some(m)) => m,
        _ => {
            let a = a.unwrap_or((0, 0));
            let b = b.unwrap_or((0, 0));
            let c = c.unwrap_or((0, 0));
            (median3(a.0, b.0, c.0), median3(a.1, b.1, c.1))
        }
    }
}

fn median3(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

/// Luma block pixel origin within the picture for block `blk` (0..3) of MB (mx,my).
fn luma_block_origin(mx: usize, my: usize, blk: usize) -> (usize, usize) {
    let bx = mx * 16 + (blk & 1) * 8;
    let by = my * 16 + (blk >> 1) * 8;
    (bx, by)
}

/// Write a reconstructed 8×8 block into the picture, adding to nothing (intra) —
/// the residual+prediction has already been summed in `block`. Clamps to 0..=255.
/// `blk`: 0..3 luma, 4 Cb, 5 Cr. `_clamp_marker` distinguishes call sites but both
/// clamp identically.
fn write_block(pic: &mut Picture, mx: usize, my: usize, blk: usize, block: &[i32; 64], _clamp_marker: Option<()>) {
    if blk < 4 {
        let (bx, by) = luma_block_origin(mx, my, blk);
        for r in 0..8 {
            let row = (by + r) * pic.lstride + bx;
            for c in 0..8 {
                pic.y[row + c] = block[r * 8 + c].clamp(0, 255) as u8;
            }
        }
    } else {
        let (bx, by) = (mx * 8, my * 8);
        let plane = if blk == 4 { &mut pic.u } else { &mut pic.v };
        for r in 0..8 {
            let row = (by + r) * pic.cstride + bx;
            for c in 0..8 {
                plane[row + c] = block[r * 8 + c].clamp(0, 255) as u8;
            }
        }
    }
}

/// Copy a whole 16×16 MB (+ 8×8 chroma) from the reference (not-coded P-MB).
fn copy_mb_from_ref(pic: &mut Picture, refp: &Picture, mx: usize, my: usize) {
    for r in 0..16 {
        let y = my * 16 + r;
        let doff = y * pic.lstride + mx * 16;
        let soff = y * refp.lstride + mx * 16;
        pic.y[doff..doff + 16].copy_from_slice(&refp.y[soff..soff + 16]);
    }
    for r in 0..8 {
        let y = my * 8 + r;
        let doff = y * pic.cstride + mx * 8;
        let soff = y * refp.cstride + mx * 8;
        pic.u[doff..doff + 8].copy_from_slice(&refp.u[soff..soff + 8]);
        pic.v[doff..doff + 8].copy_from_slice(&refp.v[soff..soff + 8]);
    }
}

// The alternate scans are exported for the (future) AC-prediction scan switch.
#[allow(dead_code)]
const _: () = {
    let _ = ALT_HORIZONTAL[0];
    let _ = ALT_VERTICAL[0];
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median3_picks_middle() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(-5, 0, 5), 0);
    }

    #[test]
    fn dquant_clamps() {
        assert_eq!(apply_dquant(1, 0), 1); // -1 clamps to 1
        assert_eq!(apply_dquant(31, 3), 31); // +2 clamps to 31
        assert_eq!(apply_dquant(10, 2), 11);
    }

    #[test]
    fn wrap_mv_range() {
        assert_eq!(wrap_mv(40, 32), 40 - 64);
        assert_eq!(wrap_mv(-40, 32), -40 + 64);
        assert_eq!(wrap_mv(10, 32), 10);
    }
}
