//! DCT coefficient (TCOEF) VLC tables and event decoding (ISO/IEC 14496-2 §7.4.1.3
//! and Annex B, Tables B-16 intra / B-17 inter). Each event codes a
//! `(last, run, level)` triple: `run` zero coefficients precede a coefficient of
//! magnitude `level`, and `last` marks the final coefficient of the block. Escape
//! coding (§7.4.1.3, three escape modes) handles events outside the VLC tables.
//!
//! The tables here are transcribed clean-room from Annex B. They are packed as
//! `(code, len, last, run, level)` — the payload is `(last<<12)|(run<<6)|level`
//! in the `VlcEntry::sym`, so one matcher serves both this and the header VLCs.
//!
//! # KNOWN DEFECT — the tables below are NOT yet bit-exact (do not trust for decode)
//!
//! Diagnosed against the ffmpeg oracle on real streams (Nord ASP and a Simple-Profile
//! `testsrc2` fixture): the tables here are incomplete/mis-assigned and drop even a
//! plain I-frame. Concrete evidence gathered while iterating:
//!
//! - `INTRA_TABLE`'s ~50 short entries match the canonical B-16 *short* codes, but the
//!   full B-16 has 102 events + escape. The dedicated codes for the *extended* run-0
//!   high levels (`(0,0,9)…(0,0,27)`, and the `(0,r,>1)` / `(1,r,>1)` families —
//!   consistent with the `LMAX`/`RMAX` arrays below) are MISSING. In the Nord I-frame
//!   the very first coded AC event is a run-0 high level whose bits (`0000111…`) match
//!   none of the short codes and are not the escape prefix (`0000011`), so decode
//!   aborts on the first block.
//! - Some short entries collide in prefix-space with those missing extended codes:
//!   e.g. `(0,3,2) = 000100100` sits exactly where the run-0 extended code
//!   `(0,0,14) = 000100100…` must live, so `(0,3,2)`'s real B-16 code differs.
//! - `TCOEF_INTER` (B-17): `(0,0,10/11/12)` are the three entries most suspected wrong
//!   (`00000000111/110`, `00000100000` here vs the B-17 `00000010111/110/101`), but
//!   this could not be independently confirmed because the inter table only exercises
//!   on P/B frames, which need a correct intra table to build the I-frame reference.
//!
//! Recovering the *exact* extended bit patterns from pixels was not achievable: for
//! high-detail I-blocks multiple coefficient sets reconstruct the same 8-bit-clamped
//! block within the IDCT's ±1 tolerance, so the pixel oracle does not uniquely
//! determine the coded `(run,level)` events, and ffmpeg exposes no coefficient trace.
//! The fix is a verbatim transcription of ISO/IEC 14496-2 Annex B Tables B-16/B-17
//! (all 102 events each, run-0/run-r extended levels included) — a normative-constant
//! transcription, not an empirical derivation.
//!
//! Lint note: `sym(last, run, level)` builds `(last<<12)|(run<<6)|level`, so a
//! zero field reads as `| 0` (identity_op), and VLC codes are odd-length
//! (unusual_byte_groupings) — both allowed for this table module.
#![allow(clippy::identity_op, clippy::unusual_byte_groupings)]

use crate::bits::BitReader;
use crate::vlc::VlcEntry;

/// Pack `(last, run, level)` into a VlcEntry symbol.
const fn sym(last: i32, run: i32, level: i32) -> i32 {
    (last << 12) | (run << 6) | level
}

/// A decoded coefficient event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coeff {
    pub last: bool,
    pub run: u32,
    pub level: i32,
}

// -- Intra TCOEF (Table B-16) --------------------------------------------------
// The MPEG-4 Visual INTRA DC/AC coefficient VLC (ISO/IEC 14496-2 Table B-16).
// Distinct from the inter table B-17. The escape prefix `0000011` is handled
// before the table match. last=0 group then last=1 group; level positive, sign
// follows the code. Built from the authoritative `(last,run,level,bits)` list in
// `INTRA_EVENTS` (below) so the codes live in one place and are validated for
// prefix-freeness by a unit test.
static TCOEF_INTRA: &[VlcEntry] = INTRA_TABLE;

/// The escape code prefix is `0000_011` (7 bits) for both intra and inter TCOEF
/// tables (§7.4.1.3). We detect it separately from the VLC match.
const ESCAPE_PREFIX: u32 = 0b0000_011;
const ESCAPE_PREFIX_LEN: u32 = 7;

/// Decode one intra TCOEF event. Handles the three escape modes (§7.4.1.3).
pub fn decode_intra(r: &mut BitReader) -> Option<Coeff> {
    decode_event(r, TCOEF_INTRA, false)
}

/// Decode one inter TCOEF event.
pub fn decode_inter(r: &mut BitReader) -> Option<Coeff> {
    decode_event(r, TCOEF_INTER, true)
}

fn decode_event(r: &mut BitReader, table: &[VlcEntry], inter: bool) -> Option<Coeff> {
    // Escape? peek 7 bits.
    if r.peek_bits(ESCAPE_PREFIX_LEN) == ESCAPE_PREFIX {
        return decode_escape(r, table, inter);
    }
    let s = crate::vlc::decode(r, table)?;
    if s < 0 {
        // A negative sym in these tables is the (unused) escape marker slot; it
        // should be unreachable because escape is handled above. Treat as error.
        return None;
    }
    let last = (s >> 12) & 1;
    let run = ((s >> 6) & 0x3F) as u32;
    let mut level = s & 0x3F;
    // sign bit follows every non-escape event
    if r.read_bit() == 1 {
        level = -level;
    }
    Some(Coeff { last: last == 1, run, level })
}

/// Escape coding (§7.4.1.3). After the 7-bit escape prefix, one bit selects
/// between the two "type" families:
/// - `0`: escape type 1/2 — a further bit distinguishes fixed level-offset
///   (type 1) from run-offset (type 2), where the payload is a normal
///   `(last,run,level)` VLC whose level/run is offset by the table's max.
/// - `1`: escape type 3 — fixed length `last(1) run(6) level(12)` literal.
///
/// XviD/DivX ASP files use escape type 3 heavily; types 1/2 are the level/run
/// extension. All three are supported.
fn decode_escape(r: &mut BitReader, table: &[VlcEntry], inter: bool) -> Option<Coeff> {
    r.skip(ESCAPE_PREFIX_LEN as usize);
    // We are inside escape. The next bits select the mode.
    // Per §7.4.1.3: read `escape` differentiators.
    // type selection:
    //   bit == 0 -> escape mode 1 or 2 (level/run offset), then another bit
    //   bit == 1 -> escape mode 3 (fixed-length literal)
    let mode_bit = r.read_bit();
    if mode_bit == 1 {
        // Escape type 3: fixed-length literal.
        let last = r.read_bit();
        let run = r.read_bits(6);
        r.marker_bit();
        let level_raw = r.read_bits(12);
        r.marker_bit();
        // level is a signed 12-bit two's complement value.
        let level = sign_extend(level_raw, 12);
        if level == 0 {
            // level 0 is forbidden — treat as error rather than emit a bogus run.
            return None;
        }
        if r.overrun() {
            return None;
        }
        return Some(Coeff { last: last == 1, run, level });
    }
    // mode_bit == 0: escape type 1 (level offset) or type 2 (run offset).
    let sub = r.read_bit();
    // For types 1 and 2 the following code is a normal TCOEF VLC event (in the
    // same intra/inter table) whose level (type 1) or run (type 2) is extended by
    // the table's LMAX/RMAX (§7.4.1.3).
    let base = crate::vlc::decode(r, table)?;
    if base < 0 {
        return None;
    }
    let last = ((base >> 12) & 1) == 1;
    let run = ((base >> 6) & 0x3F) as u32;
    let mut level = base & 0x3F;
    let sign = r.read_bit();
    if sub == 0 {
        // Escape type 1: level = level + LMAX(last,run).
        level += lmax(inter, last, run);
    } else {
        // Escape type 2: run = run + RMAX(last,level) + 1. Level unchanged.
        let extra = rmax(inter, last, level) + 1;
        if sign == 1 {
            level = -level;
        }
        if r.overrun() {
            return None;
        }
        return Some(Coeff { last, run: run + extra as u32, level });
    }
    if sign == 1 {
        level = -level;
    }
    if r.overrun() {
        return None;
    }
    Some(Coeff { last, run, level })
}

/// Sign-extend an `n`-bit value.
fn sign_extend(v: u32, n: u32) -> i32 {
    let shift = 32 - n;
    ((v << shift) as i32) >> shift
}

// LMAX/RMAX (§Table B-16 intra / B-17 inter footnotes): the largest level for a
// given (last,run) in the base table, and the largest run for a given
// (last,level). Used only for the (rare in XviD, which prefers escape type 3)
// escape types 1 and 2. Indexed by (last, run) / (last, level).
fn lmax(inter: bool, last: bool, run: u32) -> i32 {
    // Intra LMAX (Table B-16).
    const IL0: [i32; 15] = [27, 10, 5, 4, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1];
    const IL1: [i32; 21] = [
        8, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    // Inter LMAX (Table B-17).
    const PL0: [i32; 27] = [
        12, 6, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    const PL1: [i32; 41] = [
        3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    let r = run as usize;
    match (inter, last) {
        (false, false) => *IL0.get(r).unwrap_or(&1),
        (false, true) => *IL1.get(r).unwrap_or(&1),
        (true, false) => *PL0.get(r).unwrap_or(&1),
        (true, true) => *PL1.get(r).unwrap_or(&1),
    }
}

fn rmax(inter: bool, last: bool, level: i32) -> i32 {
    let l = level.unsigned_abs() as usize;
    // Intra RMAX (Table B-16).
    const IR0: [i32; 28] = [
        -1, 14, 9, 7, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const IR1: [i32; 9] = [-1, 20, 6, 1, 0, 0, 0, 0, 0];
    // Inter RMAX (Table B-17).
    const PR0: [i32; 13] = [-1, 26, 10, 6, 2, 1, 1, 0, 0, 0, 0, 0, 0];
    const PR1: [i32; 4] = [-1, 40, 1, 0];
    match (inter, last) {
        (false, false) => *IR0.get(l).unwrap_or(&0),
        (false, true) => *IR1.get(l).unwrap_or(&0),
        (true, false) => *PR0.get(l).unwrap_or(&0),
        (true, true) => *PR1.get(l).unwrap_or(&0),
    }
}

// -- Inter TCOEF (Table B-17) --------------------------------------------------
// The MPEG-4 Visual INTER coefficient VLC (ISO/IEC 14496-2 Table B-17). Escape
// prefix `0000011` handled before the table. Built from `INTER_EVENTS` below to
// keep the transcription in one authoritative place.
#[rustfmt::skip]
static TCOEF_INTER: &[VlcEntry] = &[
    VlcEntry { code: 0b10,               len: 2,  sym: sym(0, 0, 1) },
    VlcEntry { code: 0b1111,             len: 4,  sym: sym(0, 0, 2) },
    VlcEntry { code: 0b0101_01,          len: 6,  sym: sym(0, 0, 3) },
    VlcEntry { code: 0b0010_111,         len: 7,  sym: sym(0, 0, 4) },
    VlcEntry { code: 0b0001_1111,        len: 8,  sym: sym(0, 0, 5) },
    VlcEntry { code: 0b0001_0010_1,      len: 9,  sym: sym(0, 0, 6) },
    VlcEntry { code: 0b0001_0010_0,      len: 9,  sym: sym(0, 0, 7) },
    VlcEntry { code: 0b0000_1000_01,     len: 10, sym: sym(0, 0, 8) },
    VlcEntry { code: 0b0000_1000_00,     len: 10, sym: sym(0, 0, 9) },
    VlcEntry { code: 0b0000_0000_111,    len: 11, sym: sym(0, 0, 10) },
    VlcEntry { code: 0b0000_0000_110,    len: 11, sym: sym(0, 0, 11) },
    VlcEntry { code: 0b0000_0100_000,    len: 11, sym: sym(0, 0, 12) },
    VlcEntry { code: 0b1110,             len: 4,  sym: sym(0, 1, 1) },
    VlcEntry { code: 0b0001_1110,        len: 8,  sym: sym(0, 1, 2) },
    VlcEntry { code: 0b0000_0011_11,     len: 10, sym: sym(0, 1, 3) },
    VlcEntry { code: 0b0000_0100_001,    len: 11, sym: sym(0, 1, 4) },
    VlcEntry { code: 0b0000_0101_0000,   len: 12, sym: sym(0, 1, 5) },
    VlcEntry { code: 0b0101_00,          len: 6,  sym: sym(0, 2, 1) },
    VlcEntry { code: 0b0001_1101,        len: 8,  sym: sym(0, 2, 2) },
    VlcEntry { code: 0b0000_0101_0001,   len: 12, sym: sym(0, 2, 3) },
    VlcEntry { code: 0b0011_1,           len: 5,  sym: sym(0, 3, 1) },
    VlcEntry { code: 0b0001_1100,        len: 8,  sym: sym(0, 3, 2) },
    VlcEntry { code: 0b0011_0,           len: 5,  sym: sym(0, 4, 1) },
    VlcEntry { code: 0b0001_0011_1,      len: 9,  sym: sym(0, 5, 1) },
    VlcEntry { code: 0b0010_110,         len: 7,  sym: sym(0, 6, 1) },
    VlcEntry { code: 0b0010_101,         len: 7,  sym: sym(0, 7, 1) },
    VlcEntry { code: 0b0001_0011_0,      len: 9,  sym: sym(0, 8, 1) },
    VlcEntry { code: 0b0000_1000_11,     len: 10, sym: sym(0, 9, 1) },
    VlcEntry { code: 0b0000_1000_10,     len: 10, sym: sym(0, 10, 1) },
    VlcEntry { code: 0b0000_0101_0010,   len: 12, sym: sym(0, 11, 1) },
    VlcEntry { code: 0b0000_0101_0011,   len: 12, sym: sym(0, 12, 1) },
    VlcEntry { code: 0b0111,             len: 4,  sym: sym(1, 0, 1) },
    VlcEntry { code: 0b0000_1100,        len: 8,  sym: sym(1, 0, 2) },
    VlcEntry { code: 0b0000_0000_101,    len: 11, sym: sym(1, 0, 3) },
    VlcEntry { code: 0b0110,             len: 4,  sym: sym(1, 1, 1) },
    VlcEntry { code: 0b0000_0000_100,    len: 11, sym: sym(1, 1, 2) },
    VlcEntry { code: 0b0101_1,           len: 5,  sym: sym(1, 2, 1) },
    VlcEntry { code: 0b0001_1011,        len: 8,  sym: sym(1, 3, 1) },
    VlcEntry { code: 0b0001_1010,        len: 8,  sym: sym(1, 4, 1) },
    VlcEntry { code: 0b0001_0001_1,      len: 9,  sym: sym(1, 5, 1) },
    VlcEntry { code: 0b0001_0001_0,      len: 9,  sym: sym(1, 6, 1) },
    VlcEntry { code: 0b0001_0000_1,      len: 9,  sym: sym(1, 7, 1) },
    VlcEntry { code: 0b0001_0000_0,      len: 9,  sym: sym(1, 8, 1) },
    VlcEntry { code: 0b0000_1010_11,     len: 10, sym: sym(1, 9, 1) },
    VlcEntry { code: 0b0000_1010_10,     len: 10, sym: sym(1, 10, 1) },
    VlcEntry { code: 0b0000_1010_01,     len: 10, sym: sym(1, 11, 1) },
    VlcEntry { code: 0b0000_1010_00,     len: 10, sym: sym(1, 12, 1) },
    VlcEntry { code: 0b0000_0001_0011,   len: 12, sym: sym(1, 13, 1) },
    VlcEntry { code: 0b0000_0001_0010,   len: 12, sym: sym(1, 14, 1) },
    VlcEntry { code: 0b0000_0001_0001,   len: 12, sym: sym(1, 15, 1) },
    VlcEntry { code: 0b0000_0001_0000,   len: 12, sym: sym(1, 16, 1) },
    VlcEntry { code: 0b0000_0000_0000,   len: 12, sym: -2 }, // ESCAPE (inter)
];

/// Authoritative MPEG-4 Visual INTRA TCOEF table (ISO/IEC 14496-2 Table B-16), as
/// `(last, run, level, code, len)` rows. Kept as a flat const array so the actual
/// bit-patterns are visible in one place; `INTRA_TABLE` derives the matcher's
/// `VlcEntry` slice from it at compile time. Prefix-freeness and escape-collision
/// are checked by unit tests.
#[rustfmt::skip]
const INTRA_TABLE: &[VlcEntry] = &[
    // LAST = 0
    VlcEntry { code: 0b10,               len: 2,  sym: sym(0,  0,  1) },
    VlcEntry { code: 0b110,              len: 3,  sym: sym(0,  0,  2) },
    VlcEntry { code: 0b1110,             len: 4,  sym: sym(0,  1,  1) },
    VlcEntry { code: 0b0101_01,          len: 6,  sym: sym(0,  0,  3) },
    VlcEntry { code: 0b1111_0,           len: 5,  sym: sym(0,  2,  1) },
    VlcEntry { code: 0b0101_00,          len: 6,  sym: sym(0,  4,  1) },
    VlcEntry { code: 0b0011_11,          len: 6,  sym: sym(0,  3,  1) },
    VlcEntry { code: 0b0011_10,          len: 6,  sym: sym(0,  0,  4) },
    VlcEntry { code: 0b0011_01,          len: 6,  sym: sym(0,  5,  1) },
    VlcEntry { code: 0b0011_00,          len: 6,  sym: sym(0,  1,  2) },
    VlcEntry { code: 0b0010_111,         len: 7,  sym: sym(0,  6,  1) },
    VlcEntry { code: 0b0010_110,         len: 7,  sym: sym(0,  7,  1) },
    VlcEntry { code: 0b0010_101,         len: 7,  sym: sym(0,  0,  5) },
    VlcEntry { code: 0b0010_100,         len: 7,  sym: sym(0,  8,  1) },
    VlcEntry { code: 0b0001_1111,        len: 8,  sym: sym(0,  2,  2) },
    VlcEntry { code: 0b0001_1110,        len: 8,  sym: sym(0,  9,  1) },
    VlcEntry { code: 0b0001_1101,        len: 8,  sym: sym(0, 10,  1) },
    VlcEntry { code: 0b0001_1100,        len: 8,  sym: sym(0,  1,  3) },
    VlcEntry { code: 0b0001_0011_1,      len: 9,  sym: sym(0, 11,  1) },
    VlcEntry { code: 0b0001_0011_0,      len: 9,  sym: sym(0, 12,  1) },
    VlcEntry { code: 0b0001_0010_1,      len: 9,  sym: sym(0,  0,  6) },
    VlcEntry { code: 0b0001_0010_0,      len: 9,  sym: sym(0,  3,  2) },
    VlcEntry { code: 0b0001_0001_1,      len: 9,  sym: sym(0,  4,  2) },
    VlcEntry { code: 0b0001_0001_0,      len: 9,  sym: sym(0, 13,  1) },
    VlcEntry { code: 0b0001_0000_1,      len: 9,  sym: sym(0,  0,  7) },
    VlcEntry { code: 0b0001_0000_0,      len: 9,  sym: sym(0,  1,  4) },
    VlcEntry { code: 0b0000_1000_01,     len: 10, sym: sym(0,  2,  3) },
    VlcEntry { code: 0b0000_1000_00,     len: 10, sym: sym(0, 14,  1) },
    VlcEntry { code: 0b0000_1000_11,     len: 10, sym: sym(0, 15,  1) },
    VlcEntry { code: 0b0000_1000_10,     len: 10, sym: sym(0, 16,  1) },
    VlcEntry { code: 0b0000_0000_111,    len: 11, sym: sym(0,  0,  8) },
    VlcEntry { code: 0b0000_0000_110,    len: 11, sym: sym(0,  5,  2) },
    VlcEntry { code: 0b0000_0000_101,    len: 11, sym: sym(0,  6,  2) },
    VlcEntry { code: 0b0000_0000_100,    len: 11, sym: sym(0, 17,  1) },
    // LAST = 1
    VlcEntry { code: 0b0111,             len: 4,  sym: sym(1,  0,  1) },
    VlcEntry { code: 0b0000_1100,        len: 8,  sym: sym(1,  0,  2) },
    VlcEntry { code: 0b0110,             len: 4,  sym: sym(1,  1,  1) },
    VlcEntry { code: 0b0101_1,           len: 5,  sym: sym(1,  2,  1) },
    VlcEntry { code: 0b0001_1011,        len: 8,  sym: sym(1,  3,  1) },
    VlcEntry { code: 0b0001_1010,        len: 8,  sym: sym(1,  4,  1) },
    VlcEntry { code: 0b0000_1011_1,      len: 10, sym: sym(1,  5,  1) },
    VlcEntry { code: 0b0000_1011_0,      len: 10, sym: sym(1,  6,  1) },
    VlcEntry { code: 0b0000_1010_1,      len: 10, sym: sym(1,  7,  1) },
    VlcEntry { code: 0b0000_1010_0,      len: 10, sym: sym(1,  8,  1) },
    VlcEntry { code: 0b0000_1001_1,      len: 10, sym: sym(1,  9,  1) },
    VlcEntry { code: 0b0000_1001_0,      len: 10, sym: sym(1, 10,  1) },
    VlcEntry { code: 0b0000_0000_0011,   len: 12, sym: sym(1, 11,  1) },
    VlcEntry { code: 0b0000_0000_0010,   len: 12, sym: sym(1, 12,  1) },
    VlcEntry { code: 0b0000_0000_0110,   len: 12, sym: sym(1, 13,  1) },
    VlcEntry { code: 0b0000_0000_0111,   len: 12, sym: sym(1, 14,  1) },
    VlcEntry { code: 0b0000_0000_0000,   len: 12, sym: -2 }, // ESCAPE (intra)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_first_event() {
        // "10" + sign 0 → (last 0, run 0, level +1)
        let mut r = BitReader::new(&[0b1000_0000]);
        let c = decode_intra(&mut r).unwrap();
        assert_eq!(c, Coeff { last: false, run: 0, level: 1 });
    }

    #[test]
    fn intra_first_event_negative() {
        // "10" + sign 1 → (last 0, run 0, level -1)
        let mut r = BitReader::new(&[0b1010_0000]);
        let c = decode_intra(&mut r).unwrap();
        assert_eq!(c.level, -1);
    }

    /// Verify a TCOEF table is a valid prefix code: no code is a prefix of
    /// another. A transcription slip that makes one code a prefix of a longer one
    /// silently mis-decodes, so this guards the hand-entered tables.
    fn assert_prefix_free(table: &[VlcEntry]) {
        for (i, a) in table.iter().enumerate() {
            for (j, b) in table.iter().enumerate() {
                if i == j || a.len > b.len {
                    continue;
                }
                // Is `a` (shorter/equal) a prefix of `b`?
                let shift = b.len - a.len;
                if a.len == b.len {
                    assert_ne!(a.code, b.code, "duplicate code len {} at {i}/{j}", a.len);
                } else if (b.code >> shift) == a.code {
                    panic!(
                        "code {i} (len {}) is a prefix of code {j} (len {})",
                        a.len, b.len
                    );
                }
            }
        }
    }

    /// Every code must fit within its declared length (catches the underscore
    /// mis-width transcription slip where `len` and the literal disagree).
    fn assert_code_widths(table: &[VlcEntry]) {
        for e in table {
            assert!(
                e.len <= 12 && (e.code as u64) < (1u64 << e.len),
                "code {:b} does not fit in len {}",
                e.code,
                e.len
            );
        }
    }

    #[test]
    fn intra_table_is_prefix_free() {
        assert_code_widths(TCOEF_INTRA);
        assert_prefix_free(TCOEF_INTRA);
    }

    #[test]
    fn inter_table_is_prefix_free() {
        assert_code_widths(TCOEF_INTER);
        assert_prefix_free(TCOEF_INTER);
    }

    #[test]
    fn escape_prefix_not_in_tables() {
        // The escape prefix 0000011 (7 bits) must not collide with a table code.
        for t in [TCOEF_INTRA, TCOEF_INTER] {
            for e in t {
                if e.len >= ESCAPE_PREFIX_LEN as u8 {
                    let top = e.code >> (e.len as u32 - ESCAPE_PREFIX_LEN);
                    assert_ne!(top, ESCAPE_PREFIX, "code collides with escape prefix");
                }
            }
        }
    }

    /// Documents the KNOWN DEFECT (see module header): the `LMAX` arrays declare, per
    /// `(last, run)`, the largest `level` that Table B-16/B-17 assigns a *dedicated*
    /// VLC code — yet the tables below only carry the low-level codes, so the extended
    /// events (e.g. intra `(0,0,9)…(0,0,27)`) have no matching entry. This test asserts
    /// that gap so the incompleteness is visible and self-documenting; it should be
    /// deleted (turned into "every LMAX-declared event has a code") once B-16/B-17 are
    /// transcribed in full. `intra_dc_vlc_thr`-off real streams hit these first.
    #[test]
    fn extended_level_codes_are_missing_known_gap() {
        // Count how many intra events the table actually carries vs. how many the
        // LMAX declaration implies (its per-run maximum level).
        const IL0: [i32; 15] = [27, 10, 5, 4, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1];
        let declared: i32 = IL0.iter().sum(); // last=0 events implied by LMAX
        let mut present_last0_max_level = 0i32;
        for e in TCOEF_INTRA {
            if e.sym < 0 {
                continue; // escape slot
            }
            let last = (e.sym >> 12) & 1;
            let level = e.sym & 0x3F;
            if last == 0 && ((e.sym >> 6) & 0x3F) == 0 {
                present_last0_max_level = present_last0_max_level.max(level);
            }
        }
        // LMAX says run-0 reaches level 27; the table stops far short — proof the
        // extended dedicated codes are absent.
        assert_eq!(IL0[0], 27, "LMAX invariant");
        assert!(
            present_last0_max_level < 27,
            "run-0 max level in table is {present_last0_max_level}; if this reaches 27 \
             the extended codes have been added — flip this assertion to a full check \
             (declared = {declared})"
        );
    }

    #[test]
    fn escape_type3_literal() {
        // escape prefix 0000011, mode 1, last 1, run 000010(=2), marker 1,
        // level 000000000011 (=3), marker 1
        // bits: 0000011 1 1 000010 1 000000000011 1
        let bits = "0000011" .to_string()
            + "1"  // mode 3
            + "1"  // last
            + "000010" // run 2
            + "1"  // marker
            + "000000000011" // level 3
            + "1"; // marker
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let c = decode_inter(&mut r).unwrap();
        assert_eq!(c, Coeff { last: true, run: 2, level: 3 });
    }

    fn bits_to_bytes(s: &str) -> Vec<u8> {
        let mut out = vec![];
        let mut cur = 0u8;
        let mut n = 0;
        for ch in s.chars() {
            cur = (cur << 1) | if ch == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(cur << (8 - n));
        }
        out
    }
}
