//! A minimal, in-crate **EBML reader** used by the structural round-trip tests. It is
//! deliberately tiny — just enough to walk the element tree `pf-mkv` produces and verify it
//! reads back byte-for-byte (spec: `mkv/spec/MATROSKA.md`; RFC 8794). It is *not* a general
//! Matroska demuxer (that is a future crate); it only needs to handle the subset the muxer
//! emits: shortest-length and 8-octet sizes, the `0xFF` unknown-size streamed masters, and
//! the leaf types the muxer writes.
//!
//! This lives in an integration-test file (not `src/`) because it is test scaffolding, not
//! part of the muxer's public surface — the muxer is write-only.

#![allow(dead_code)] // shared helpers; not every test uses every accessor

/// Read one EBML VINT starting at `data[*off]`, following RFC 8794 §4: the first set bit
/// (from the MSB of the first octet) is the VINT_MARKER; its position gives the octet
/// length; the remaining bits are big-endian VINT_DATA. Advances `*off`. Returns
/// `(value, len, all_ones)` where `all_ones` flags the reserved unknown-size pattern
/// (§6.2).
pub fn read_vint(data: &[u8], off: &mut usize) -> (u64, usize, bool) {
    let first = data[*off];
    assert_ne!(first, 0x00, "a VINT's first octet cannot be zero (length > 8)");
    let len = first.leading_zeros() as usize + 1;
    assert!(len <= 8, "VINT length out of range");
    // Data bits of the first octet: clear the marker. `0xFF >> len` in u16 avoids the u8
    // overflow at len == 8 (mask is then 0 — an 8-octet VINT has no data in octet 0).
    let mut value = (first as u64) & ((0xFFu16 >> len) as u64);
    let mut all_ones = value == ((0xFFu16 >> len) as u64);
    for i in 1..len {
        let b = data[*off + i];
        value = (value << 8) | b as u64;
        all_ones &= b == 0xFF;
    }
    *off += len;
    (value, len, all_ones)
}

/// An Element ID read verbatim as its `len` on-the-wire octets (spec `§ID-tree`): unlike a
/// size, a Matroska ID *includes* its length-descriptor bits, so we return the raw bytes to
/// compare against the `ebml::id` constants. Advances `*off`.
pub fn read_id(data: &[u8], off: &mut usize) -> Vec<u8> {
    let first = data[*off];
    let len = first.leading_zeros() as usize + 1;
    let id = data[*off..*off + len].to_vec();
    *off += len;
    id
}

/// One parsed element: its ID (raw bytes), and either a known byte range of leaf data, or
/// `Unknown` size (a streamed master — Segment/Cluster).
#[derive(Clone, Debug)]
pub struct Element {
    pub id: Vec<u8>,
    /// `Some(range)` for a definite-size element; `None` for an unknown-size master.
    pub data: Option<(usize, usize)>,
}

/// Read one element header (ID + size) at `data[*off]`, advancing `*off` **past the header
/// only** (not the data). Returns the element and, for a definite size, the data byte
/// range; for an unknown size, `data == None`. The caller decides whether to descend into a
/// master's children or skip a leaf's data.
pub fn read_element_header(data: &[u8], off: &mut usize) -> Element {
    let id = read_id(data, off);
    let (size, _len, all_ones) = read_vint(data, off);
    if all_ones {
        Element { id, data: None } // unknown size (streamed master)
    } else {
        let start = *off;
        Element { id, data: Some((start, start + size as usize)) }
    }
}

/// Recursively collect **every** element in `data[start..end]` as a flat list of
/// `(depth, Element)`, descending into masters. A definite-size master is descended within
/// its data range; an unknown-size master is descended until the end of `data` or until an
/// ID appears that is not one of `child_ids` (the implicit-termination rule, spec
/// `§sizing`). `masters` lists the IDs to treat as masters (descend into); everything else
/// is a leaf (its data skipped).
pub fn walk(data: &[u8], masters: &[&[u8]]) -> Vec<(usize, Element)> {
    let mut out = Vec::new();
    walk_range(data, 0, data.len(), 0, masters, &mut out);
    out
}

fn is_master(id: &[u8], masters: &[&[u8]]) -> bool {
    masters.iter().any(|m| *m == id)
}

fn walk_range(
    data: &[u8],
    start: usize,
    end: usize,
    depth: usize,
    masters: &[&[u8]],
    out: &mut Vec<(usize, Element)>,
) {
    let mut off = start;
    while off < end {
        let el = read_element_header(data, &mut off);
        let id = el.id.clone();
        match el.data {
            Some((ds, de)) => {
                out.push((depth, el));
                if is_master(&id, masters) {
                    walk_range(data, ds, de, depth + 1, masters, out);
                }
                off = de; // skip past the element's data
            }
            None => {
                // Unknown-size master: descend from here to `end`. Children run until an ID
                // that is not a valid child of this master — approximated as "until another
                // unknown-size master ID (Segment/Cluster) appears at this level, or `end`".
                out.push((depth, el));
                // Find where this streamed master's children end: the next Segment/Cluster
                // ID at this offset level, else `end`.
                let child_end = next_streamed_master(data, off, end, &id, masters);
                walk_range(data, off, child_end, depth + 1, masters, out);
                off = child_end;
            }
        }
    }
}

/// Find the end of an unknown-size master's children: scan forward parsing element headers
/// at this level until we hit an ID that terminates the open master (another Cluster, or
/// end). For our muxer's output the only unknown-size masters are Segment (contains
/// Info/Tracks/Cluster*) and Cluster (contains Timestamp/SimpleBlock*); a Cluster ends at
/// the next Cluster; a Segment ends at `end`.
fn next_streamed_master(data: &[u8], start: usize, end: usize, open_id: &[u8], masters: &[&[u8]]) -> usize {
    use pf_mkv::ebml::id;
    // A Cluster's children terminate at the next Cluster ID.
    if open_id == id::CLUSTER {
        let mut off = start;
        while off < end {
            let here = off;
            let child = read_element_header(data, &mut off);
            if child.id == id::CLUSTER {
                return here; // next cluster starts here → this cluster's children end here
            }
            match child.data {
                Some((_ds, de)) => off = de,
                None => return here, // another streamed master → terminate
            }
            let _ = masters;
        }
        end
    } else {
        // Segment (or any other): its children run to the end of the stream.
        end
    }
}

/// A decoded SimpleBlock body (spec `§simpleblock`): track number, signed relative
/// timestamp (ticks), keyframe flag, and the frame bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleBlock {
    pub track: u64,
    pub rel_ts: i16,
    pub keyframe: bool,
    pub frame: Vec<u8>,
}

/// Parse a SimpleBlock element's *data* (the bytes after its ID+size) into its fields.
pub fn parse_simple_block(data: &[u8]) -> SimpleBlock {
    let mut off = 0;
    let (track, _len, _ao) = read_vint(data, &mut off);
    let rel_ts = i16::from_be_bytes([data[off], data[off + 1]]);
    off += 2;
    let flags = data[off];
    off += 1;
    SimpleBlock {
        track,
        rel_ts,
        keyframe: flags & 0x80 != 0,
        frame: data[off..].to_vec(),
    }
}

// The reader is exercised by the tests in `roundtrip.rs`; a couple of self-checks here keep
// the scaffolding honest on its own.
#[test]
fn vint_reader_matches_writer() {
    use pf_mkv::ebml;
    for &v in &[0u64, 1, 126, 127, 128, 16_000, 1_000_000] {
        let mut buf = Vec::new();
        ebml::write_size(&mut buf, v);
        let mut off = 0;
        let (got, _len, _ao) = read_vint(&buf, &mut off);
        assert_eq!(got, v);
        assert_eq!(off, buf.len());
    }
    // The 0xFF unknown marker reads back as all-ones.
    let mut off = 0;
    let (_v, len, all_ones) = read_vint(&[0xFF], &mut off);
    assert_eq!(len, 1);
    assert!(all_ones);
}
