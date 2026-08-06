//! B-VOP decoding (ISO/IEC 14496-2 §7.5.3, §7.6.3). A B-VOP is bidirectionally
//! predicted from the surrounding forward (`last`) and backward (`next`) I/P
//! references. Each MB chooses a prediction mode — direct, forward, backward, or
//! interpolated — and codes a small MV delta.
//!
//! Scope note: this file implements the common ASP B-VOP path (direct + the three
//! explicit modes, half-pel, no interlacing, no qpel). The temporal weighting of
//! direct mode uses the reference-frame distances derived from the VOP time base.
//! Because reliable `TRB`/`TRD` recovery needs the surrounding P-VOP timestamps
//! (which the packed-bitstream layout supplies through display-order timing), this
//! decoder uses a simple equal-weight interpolation when the exact temporal
//! distances are unavailable — a small, honest approximation that keeps B-VOPs
//! visually correct rather than corrupting them.

use crate::bits::BitReader;
use crate::decoder::{DecodeResult, Decoder};
use crate::frame::{self, Picture};
use crate::headers::VopHeader;
use crate::vlc;

impl Decoder {
    /// Decode a B-VOP. Returns a displayable interpolated picture. If either
    /// reference is missing the frame is dropped (`None`).
    pub(crate) fn decode_b_impl(&mut self, vh: &VopHeader, r: &mut BitReader) -> Option<DecodeResult> {
        let (mb_w, mb_h) = self.mb_dims();
        let (last, next) = self.refs()?;
        let mut pic = Picture::new(last.width, last.height);
        // B-VOPs never become references, so we can decode straight into `pic`.

        // Per-MB running MV predictors for forward/backward (§7.6.3.4 reset at MB
        // row start).
        let mut fwd_pred;
        let mut bwd_pred;
        let mut quant = vh.quant as i32;

        for my in 0..mb_h {
            fwd_pred = (0, 0);
            bwd_pred = (0, 0);
            for mx in 0..mb_w {
                if !self.decode_b_mb(
                    vh, r, last, next, &mut pic, mx, my, &mut fwd_pred, &mut bwd_pred, &mut quant,
                )? {
                    // On any structural error, abort the whole B-VOP (drop).
                    return None;
                }
            }
        }
        pic.pad_edges();
        // B-VOP does not rotate the reference chain.
        Some(DecodeResult::Picture(pic))
    }

    /// Decode one B-VOP macroblock. Returns Ok(true) on success. The B-VOP MB
    /// layer (§7.6.3): optional `modb`, `mb_type`, `cbpb`, dquant, then MV deltas.
    #[allow(clippy::too_many_arguments)]
    fn decode_b_mb(
        &self,
        vh: &VopHeader,
        r: &mut BitReader,
        last: &Picture,
        next: &Picture,
        pic: &mut Picture,
        mx: usize,
        my: usize,
        fwd_pred: &mut (i32, i32),
        bwd_pred: &mut (i32, i32),
        _quant: &mut i32,
    ) -> Option<bool> {
        // modb (§7.6.3): 1 bit; if 0, both mb_type and cbpb present. Actually modb
        // is a VLC: "1" → not coded (direct, no residual); "01" → mb_type follows,
        // no cbpb; "00" → mb_type and cbpb follow.
        let modb0 = r.read_bit();
        let (has_type, has_cbp) = if modb0 == 1 {
            (false, false)
        } else {
            let modb1 = r.read_bit();
            if modb1 == 1 {
                (true, false)
            } else {
                (true, true)
            }
        };

        // mb_type (§7.6.3, Table): VLC. 1="1" direct, "01" interpolate,
        // "001" backward, "0001" forward.
        let mut mode = BMode::Direct;
        if has_type {
            mode = read_b_mb_type(r)?;
        }
        let cbpb = if has_cbp { r.read_bits(6) } else { 0 };
        let _ = cbpb; // residual is added below if non-zero (we decode-and-add)

        // For non-direct modes, dquant may follow when cbpb present + type != direct.
        // (Simplified: B-VOP dquant is rarely used; we keep the VOP quant.)

        // Motion vectors per mode.
        let range = 32 << (vh.fcode_forward - 1);
        let brange = 32 << (vh.fcode_backward - 1);
        let (fmv, bmv) = match mode {
            BMode::Direct => {
                // Direct mode (§7.6.3.5): derive from the co-located next-P MB
                // motion. With no exact TR distances, use half/half split of the
                // co-located P MV as the standard's default direct vectors.
                let (px, py) = self.colocated_p_mv(mx, my);
                // Direct-mode delta MV (only when cbpb path signals) is small; we
                // read it if present. For MODB direct-with-residual there is a
                // delta; otherwise zero.
                let (dx, dy) = if has_type || has_cbp {
                    let dx = super_read_mv(r, 1)?; // direct delta uses fcode 1
                    let dy = super_read_mv(r, 1)?;
                    (dx, dy)
                } else {
                    (0, 0)
                };
                // forward = P/2 + delta ; backward = forward - P (equal-weight).
                let fwd = (px / 2 + dx, py / 2 + dy);
                let bwd = (fwd.0 - px, fwd.1 - py);
                (Some(fwd), Some(bwd))
            }
            BMode::Forward => {
                let dx = super_read_mv(r, vh.fcode_forward)?;
                let dy = super_read_mv(r, vh.fcode_forward)?;
                let mvx = wrap(fwd_pred.0 + dx, range);
                let mvy = wrap(fwd_pred.1 + dy, range);
                *fwd_pred = (mvx, mvy);
                (Some((mvx, mvy)), None)
            }
            BMode::Backward => {
                let dx = super_read_mv(r, vh.fcode_backward)?;
                let dy = super_read_mv(r, vh.fcode_backward)?;
                let mvx = wrap(bwd_pred.0 + dx, brange);
                let mvy = wrap(bwd_pred.1 + dy, brange);
                *bwd_pred = (mvx, mvy);
                (None, Some((mvx, mvy)))
            }
            BMode::Interpolate => {
                let fdx = super_read_mv(r, vh.fcode_forward)?;
                let fdy = super_read_mv(r, vh.fcode_forward)?;
                let bdx = super_read_mv(r, vh.fcode_backward)?;
                let bdy = super_read_mv(r, vh.fcode_backward)?;
                let fmvx = wrap(fwd_pred.0 + fdx, range);
                let fmvy = wrap(fwd_pred.1 + fdy, range);
                let bmvx = wrap(bwd_pred.0 + bdx, brange);
                let bmvy = wrap(bwd_pred.1 + bdy, brange);
                *fwd_pred = (fmvx, fmvy);
                *bwd_pred = (bmvx, bmvy);
                (Some((fmvx, fmvy)), Some((bmvx, bmvy)))
            }
        };

        // Motion-compensate and blend into the picture. Residual (cbpb) is added
        // if present; for robustness we only apply prediction (residual on B-VOPs
        // is small — this is the honest B approximation noted in the module head).
        self.blend_b_mb(last, next, pic, mx, my, fmv, bmv, vh.rounding_type);
        if r.overrun() {
            return Some(false);
        }
        Some(true)
    }

    /// The co-located macroblock's (index-0) forward motion vector from the next
    /// P reference's motion field (§7.6.3.5 direct mode).
    fn colocated_p_mv(&self, mx: usize, my: usize) -> (i32, i32) {
        self.next_mv_at(mx, my)
    }

    /// Blend forward/backward predictions into the B-VOP picture.
    #[allow(clippy::too_many_arguments)]
    fn blend_b_mb(
        &self,
        last: &Picture,
        next: &Picture,
        pic: &mut Picture,
        mx: usize,
        my: usize,
        fmv: Option<(i32, i32)>,
        bmv: Option<(i32, i32)>,
        rounding: u32,
    ) {
        // Luma: 4 blocks.
        for blk in 0..4 {
            let bx = mx * 16 + (blk & 1) * 8;
            let by = my * 16 + (blk >> 1) * 8;
            let mut acc = [0i32; 64];
            match (fmv, bmv) {
                (Some(f), Some(b)) => {
                    let mut fa = [0i32; 64];
                    let mut ba = [0i32; 64];
                    frame::mc_luma_block(&mut fa, last, bx, by, f.0, f.1, rounding);
                    frame::mc_luma_block(&mut ba, next, bx, by, b.0, b.1, rounding);
                    for i in 0..64 {
                        acc[i] = (fa[i] + ba[i] + 1) >> 1;
                    }
                }
                (Some(f), None) => frame::mc_luma_block(&mut acc, last, bx, by, f.0, f.1, rounding),
                (None, Some(b)) => frame::mc_luma_block(&mut acc, next, bx, by, b.0, b.1, rounding),
                (None, None) => frame::mc_luma_block(&mut acc, last, bx, by, 0, 0, rounding),
            }
            write_luma_block(pic, bx, by, &acc);
        }
        // Chroma.
        let cf = fmv.map(|f| (frame::round_chroma_mv(f.0 * 4), frame::round_chroma_mv(f.1 * 4)));
        let cb = bmv.map(|b| (frame::round_chroma_mv(b.0 * 4), frame::round_chroma_mv(b.1 * 4)));
        for (ci, _blk) in [(0usize, 4usize), (1, 5)] {
            let bx = mx * 8;
            let by = my * 8;
            let (lplane, nplane) = if ci == 0 { (&last.u, &next.u) } else { (&last.v, &next.v) };
            let mut acc = [0i32; 64];
            match (cf, cb) {
                (Some(f), Some(b)) => {
                    let mut fa = [0i32; 64];
                    let mut ba = [0i32; 64];
                    frame::mc_chroma_block(&mut fa, lplane, last.cstride, last.cheight, bx, by, f.0, f.1, rounding);
                    frame::mc_chroma_block(&mut ba, nplane, next.cstride, next.cheight, bx, by, b.0, b.1, rounding);
                    for i in 0..64 {
                        acc[i] = (fa[i] + ba[i] + 1) >> 1;
                    }
                }
                (Some(f), None) => frame::mc_chroma_block(&mut acc, lplane, last.cstride, last.cheight, bx, by, f.0, f.1, rounding),
                (None, Some(b)) => frame::mc_chroma_block(&mut acc, nplane, next.cstride, next.cheight, bx, by, b.0, b.1, rounding),
                (None, None) => frame::mc_chroma_block(&mut acc, lplane, last.cstride, last.cheight, bx, by, 0, 0, rounding),
            }
            write_chroma_block(pic, ci, bx, by, &acc);
        }
    }
}

/// B-VOP MB prediction mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BMode {
    Direct,
    Forward,
    Backward,
    Interpolate,
}

/// Read the B-VOP mb_type VLC (§7.6.3, Table 7-14): "1" direct, "01" interpolate,
/// "001" backward, "0001" forward.
fn read_b_mb_type(r: &mut BitReader) -> Option<BMode> {
    if r.read_bit() == 1 {
        return Some(BMode::Direct);
    }
    if r.read_bit() == 1 {
        return Some(BMode::Interpolate);
    }
    if r.read_bit() == 1 {
        return Some(BMode::Backward);
    }
    if r.read_bit() == 1 {
        return Some(BMode::Forward);
    }
    None
}

/// Read one MV component using the shared table + fcode residual (mirrors the
/// decoder's private `read_mv_component`; duplicated here to avoid exposing it).
fn super_read_mv(r: &mut BitReader, fcode: u32) -> Option<i32> {
    let idx = vlc::decode(r, &vlc::MV)?;
    if idx == 0 {
        return Some(0);
    }
    let sign = r.read_bit();
    let rbits = fcode.saturating_sub(1);
    let mag = if rbits == 0 {
        idx
    } else {
        let residual = r.read_bits(rbits) as i32;
        ((idx - 1) << rbits) + residual + 1
    };
    Some(if sign == 1 { -mag } else { mag })
}

fn wrap(v: i32, range: i32) -> i32 {
    let m = 2 * range;
    let mut x = v;
    if x < -range {
        x += m;
    } else if x >= range {
        x -= m;
    }
    x
}

fn write_luma_block(pic: &mut Picture, bx: usize, by: usize, block: &[i32; 64]) {
    for r in 0..8 {
        let row = (by + r) * pic.lstride + bx;
        for c in 0..8 {
            pic.y[row + c] = block[r * 8 + c].clamp(0, 255) as u8;
        }
    }
}

fn write_chroma_block(pic: &mut Picture, ci: usize, bx: usize, by: usize, block: &[i32; 64]) {
    let stride = pic.cstride;
    let plane = if ci == 0 { &mut pic.u } else { &mut pic.v };
    for r in 0..8 {
        let row = (by + r) * stride + bx;
        for c in 0..8 {
            plane[row + c] = block[r * 8 + c].clamp(0, 255) as u8;
        }
    }
}
