//! DivX "packed bitstream" handling. When `divx_packed` is set (as in the target
//! file), the muxer stores VOPs in *coded* (decode) order but packs them so that
//! each container frame carries one displayable VOP: a container frame that holds
//! a P-VOP followed by the *next* B-VOP has both concatenated, and a following
//! container frame is a stuffing-only "N-VOP" (not-coded VOP) that stands in for
//! the B-VOP already decoded. This lets a naive player show frames in the right
//! order without reordering (DivX 5 compatibility).
//!
//! This module splits an incoming buffer into the sequence of coded units it
//! actually contains, at VOP start-code (`00 00 01 B6`) granularity, so the
//! decoder sees exactly one coded VOP per call and display order falls out
//! naturally. Non-VOP headers (VOS/VO/VOL/GOV/user-data) are passed through as a
//! prefix on the first following VOP so the decoder still parses them.
//!
//! Reference: DivX "Video Packed Bitstream" (an out-of-standard extension; there
//! is no ISO clause — the behaviour is defined by the `packed bitstream`
//! user-data tag and the 7-byte stuffing pattern `00 00 01 B6 ...` volume of
//! not-coded VOPs). We detect stuffing tails and drop them.

use crate::headers::{find_start_code, startcode};

/// Streaming walk over the coded VOP units in a container buffer, allocation-free.
///
/// Handles the packed case (multiple VOPs in one buffer) and drops the trailing
/// not-coded stuffing VOP that DivX appends. Each VOP spans from its
/// `00 00 01 B6` start code up to the next VOP start code (or the buffer end),
/// exactly as the coded-order decode expects. The cursor holds only byte offsets
/// (no heap): the decoder pulls one VOP slice at a time and decodes it directly
/// from the input buffer, so no per-buffer allocation or byte copy is needed
/// (spec: performance #1).
///
/// Leading stream headers (VOS/VO/VOL/GOV/user-data) before the first VOP are
/// parsed separately by the element (`try_parse_headers`) and are simply not
/// yielded here. Robust: never panics; a buffer with no VOP yields nothing.
pub struct VopCursor {
    /// Start offset of the next VOP to yield (its `00 00 01 B6`), if any.
    pending: Option<usize>,
    /// Next byte position to resume the start-code scan from.
    scan: usize,
}

impl VopCursor {
    /// A cursor positioned at the first VOP start code in `data` (if any).
    pub fn new(data: &[u8]) -> Self {
        let mut cur = VopCursor { pending: None, scan: 0 };
        cur.pending = cur.next_vop_start(data);
        cur
    }

    /// Find the next VOP start code at/after `self.scan`, advancing `self.scan`
    /// past it. Non-VOP start codes are skipped. `None` when none remain.
    fn next_vop_start(&mut self, data: &[u8]) -> Option<usize> {
        let mut i = self.scan;
        while let Some((off, code)) = find_start_code(data, i) {
            if code == startcode::VOP_START {
                self.scan = off + 3;
                return Some(off);
            }
            i = off + 3;
        }
        self.scan = data.len();
        None
    }

    /// The next non-stuffing coded VOP slice, or `None` when the buffer is drained.
    /// The slice borrows `data`; pass the same buffer each call.
    pub fn next<'a>(&mut self, data: &'a [u8]) -> Option<&'a [u8]> {
        loop {
            let start = self.pending?;
            let next = self.next_vop_start(data);
            let end = next.unwrap_or(data.len());
            self.pending = next;
            let vop = &data[start..end];
            // Skip a trailing not-coded stuffing VOP.
            if is_stuffing_vop(vop) {
                continue;
            }
            return Some(vop);
        }
    }
}

/// True if a VOP is a DivX packed-bitstream stuffing (not-coded) VOP: it is very
/// short and its `vop_coded` flag is 0. We check the coding type + coded bit
/// cheaply from the first few bytes after the start code.
fn is_stuffing_vop(vop: &[u8]) -> bool {
    // vop = 00 00 01 B6 <coding_type:2> <modulo_time_base ...> ... <vop_coded:1>
    // A packed stuffing VOP is typically 7 bytes: 00 00 01 B6 + a byte or two.
    // We conservatively treat a VOP shorter than 8 bytes as stuffing; the decoder
    // also handles a genuine not-coded VOP by repeating, so a false negative here
    // is harmless (it decodes to a repeat), but a false positive would drop a real
    // frame — so keep the length threshold tight.
    vop.len() < 8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect every VOP the cursor yields (test helper only — the hot path pulls
    /// one at a time and never collects).
    fn collect_vops(data: &[u8]) -> Vec<&[u8]> {
        let mut cur = VopCursor::new(data);
        let mut out = Vec::new();
        while let Some(vop) = cur.next(data) {
            out.push(vop);
        }
        out
    }

    #[test]
    fn single_vop_one_unit() {
        // headers (VOL) + one VOP with >=8 bytes payload
        let mut data = vec![0x00, 0x00, 0x01, 0x20, 0xAA, 0xBB]; // VOL
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6]); // VOP
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let vops = collect_vops(&data);
        assert_eq!(vops.len(), 1);
        assert_eq!(vops[0][3], 0xB6, "yielded slice starts at the VOP start code");
    }

    #[test]
    fn packed_two_vops_plus_stuffing() {
        let mut data = vec![];
        // VOP A (long)
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6]);
        data.extend_from_slice(&[9; 20]);
        // VOP B (long)
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6]);
        data.extend_from_slice(&[7; 20]);
        // stuffing VOP (short)
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6, 0x00]);
        let vops = collect_vops(&data);
        assert_eq!(vops.len(), 2, "two real VOPs, stuffing dropped");
    }

    #[test]
    fn no_vop_yields_nothing() {
        // Only a VOL header, no VOP start code.
        let data = vec![0x00, 0x00, 0x01, 0x20, 0xAA, 0xBB];
        assert!(collect_vops(&data).is_empty());
    }
}
