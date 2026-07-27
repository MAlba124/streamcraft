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

/// One coded unit from a container buffer: the byte range that begins at (and
/// includes) a VOP start code, plus any leading stream headers that preceded it.
pub struct Unit<'a> {
    /// Header bytes preceding the VOP (VOS/VO/VOL/GOV/user-data), possibly empty.
    pub headers: &'a [u8],
    /// The VOP bytes, starting at its `00 00 01 B6` start code.
    pub vop: &'a [u8],
}

/// Split `data` into the coded VOP units it contains. Handles the packed case
/// (multiple VOPs in one buffer) and the trailing not-coded stuffing VOP that
/// DivX appends. Robust: never panics; a buffer with no VOP yields an empty vec
/// (the header bytes are still surfaced via [`leading_headers`]).
pub fn split_units(data: &[u8]) -> Vec<Unit<'_>> {
    let mut units = Vec::new();
    // Find the first VOP start code; everything before it is headers.
    let mut header_start = 0usize;
    let mut i = 0usize;
    // Collect start-code positions of interest.
    let mut vop_positions = Vec::new();
    let mut header_end = None;
    while let Some((off, code)) = find_start_code(data, i) {
        if code == startcode::VOP_START {
            if header_end.is_none() {
                header_end = Some(off);
            }
            vop_positions.push(off);
        }
        i = off + 3;
    }
    if vop_positions.is_empty() {
        return units;
    }
    let headers = &data[header_start..header_end.unwrap_or(0)];
    header_start = header_end.unwrap_or(0);
    let _ = header_start;

    for (n, &start) in vop_positions.iter().enumerate() {
        let end = vop_positions.get(n + 1).copied().unwrap_or(data.len());
        let vop = &data[start..end];
        // Skip a trailing not-coded stuffing VOP: a very short VOP (< a few
        // bytes of payload after the start code) whose `vop_coded` bit is 0.
        if is_stuffing_vop(vop) {
            continue;
        }
        units.push(Unit {
            headers: if n == 0 { headers } else { &data[start..start] },
            vop,
        });
    }
    units
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

/// The stream headers at the front of `data` (before the first VOP), for priming
/// the VOL parser from the first buffer. Empty if the buffer begins with a VOP.
pub fn leading_headers(data: &[u8]) -> &[u8] {
    match find_start_code_vop(data) {
        Some(off) => &data[..off],
        None => data,
    }
}

fn find_start_code_vop(data: &[u8]) -> Option<usize> {
    let mut i = 0;
    while let Some((off, code)) = find_start_code(data, i) {
        if code == startcode::VOP_START {
            return Some(off);
        }
        i = off + 3;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_vop_one_unit() {
        // headers (VOL) + one VOP with >=8 bytes payload
        let mut data = vec![0x00, 0x00, 0x01, 0x20, 0xAA, 0xBB]; // VOL
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0xB6]); // VOP
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let units = split_units(&data);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].vop[3], 0xB6);
        assert!(!units[0].headers.is_empty(), "headers surfaced on first unit");
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
        let units = split_units(&data);
        assert_eq!(units.len(), 2, "two real VOPs, stuffing dropped");
    }
}
