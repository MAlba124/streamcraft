//! Annex B superframe splitting (VP9 spec v0.7 §B.1–B.4).
//!
//! VP9 lets an encoder consolidate several coded frames into one chunk
//! ("superframe"). The enclosed frames are concatenated, then an index is appended in
//! the last up-to-34 bytes:
//!
//! ```text
//! superframe( sz ) {                                     §B.2
//!     for( i = 0; i < NumFrames; i++ )
//!         frame( frame_sizes[ i ] )
//!     superframe_index( )
//! }
//! superframe_index( ) {                                  §B.2.1
//!     superframe_header( )                               // trailing copy at the START
//!     for( i = 0; i < NumFrames; i++ )
//!         frame_sizes[ i ]                               f(SzBytes)  // little-endian
//!     superframe_header( )                               // and again at the END
//! }
//! superframe_header( ) {                                 §B.2.2
//!     superframe_marker            f(3)                  // 0b110
//!     bytes_per_framesize_minus_1  f(2)
//!     frames_in_superframe_minus_1 f(3)
//! }
//! ```
//!
//! Detection per §B.4:
//!
//! 1. The top three bits of the final byte must equal the `superframe_marker` `0b110`.
//! 2. `SzIndex = 2 + NumFrames * SzBytes`.
//! 3. The first byte of the index (`chunk[len - SzIndex]`) must equal the final byte.
//!
//! If both checks pass the chunk is a superframe and each enclosed frame is decoded in
//! turn; otherwise the whole chunk is a single frame (the §B.4 fallback). This is
//! VP9-intrinsic framing (Annex B of the bitstream spec), not a separate container:
//! every VP9 chunk a decoder receives may or may not carry the index, and the split
//! must happen before the per-frame header walk. The backing `oxideav-vp9` 0.0.12 does
//! not split superframes, so `vp9dec` does — this module.

/// The §B.2.2 `superframe_marker`: the top three bits of the trailing (and leading)
/// index byte (`0b110`).
const SUPERFRAME_MARKER: u8 = 0b110;

/// Split a chunk into the byte ranges of its enclosed coded frames, in decode order.
///
/// Ranges are `(start, end)` half-open offsets into `chunk`; the superframe index
/// bytes themselves are never returned (they are not part of any coded frame).
/// Returning index ranges rather than slices lets the caller keep `chunk` borrowed
/// while it decodes — the returned `Vec` does not borrow `chunk`.
///
/// When the chunk is not a valid superframe the whole chunk is returned as one range
/// (the §B.4 single-frame fallback). An empty chunk yields an empty list. A malformed
/// index (wrong marker, too short, mismatched leading byte, or declared sizes that
/// overrun the payload region) is treated as "not a superframe" rather than an error,
/// exactly as §B.4 specifies — so a corrupt chunk never panics here.
pub fn split_superframe(chunk: &[u8]) -> Vec<(usize, usize)> {
    match superframe_frame_sizes(chunk) {
        Some(sizes) => {
            let mut out = Vec::with_capacity(sizes.len());
            let mut off = 0usize;
            for sz in sizes {
                out.push((off, off + sz));
                off += sz;
            }
            out
        }
        None => {
            if chunk.is_empty() {
                Vec::new()
            } else {
                vec![(0, chunk.len())]
            }
        }
    }
}

/// Parse the §B.2.1 index and return the enclosed `frame_sizes[ ]` when the chunk is a
/// valid superframe, or `None` for the §B.4 single-frame fallback.
fn superframe_frame_sizes(chunk: &[u8]) -> Option<Vec<usize>> {
    let len = chunk.len();
    if len == 0 {
        return None;
    }

    // §B.4 step 1 — the final byte must carry the superframe_marker.
    let marker_byte = chunk[len - 1];
    if (marker_byte >> 5) != SUPERFRAME_MARKER {
        return None;
    }

    // §B.2.2 — decode the trailing superframe_header.
    let bytes_per_framesize = ((marker_byte >> 3) & 0b11) as usize + 1; // SzBytes
    let num_frames = (marker_byte & 0b111) as usize + 1; // NumFrames

    // §B.4 step 2 — SzIndex = 2 + NumFrames * SzBytes.
    let sz_index = 2 + num_frames * bytes_per_framesize;
    if len < sz_index {
        return None;
    }

    // §B.4 step 3 — the leading index byte must equal the trailing one.
    let index_start = len - sz_index;
    if chunk[index_start] != marker_byte {
        return None;
    }

    // §B.2.1 — the NumFrames frame_sizes, little-endian f(SzBytes) each, sit between
    // the two superframe_header bytes.
    let mut sizes = Vec::with_capacity(num_frames);
    let mut p = index_start + 1;
    let mut total: usize = 0;
    for _ in 0..num_frames {
        let mut sz = 0usize;
        for b in 0..bytes_per_framesize {
            sz |= (chunk[p + b] as usize) << (8 * b);
        }
        p += bytes_per_framesize;
        // Guard a declared size that overruns the payload region (everything before
        // the index); a conforming stream never does, but a corrupt one must not
        // panic the slice in the caller.
        total = total.checked_add(sz)?;
        if total > index_start {
            return None;
        }
        sizes.push(sz);
    }

    Some(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a superframe index byte (the value at both ends of the §B.2.1 index).
    fn marker(bytes_per_minus_1: u8, frames_minus_1: u8) -> u8 {
        (SUPERFRAME_MARKER << 5) | ((bytes_per_minus_1 & 0b11) << 3) | (frames_minus_1 & 0b111)
    }

    /// Assemble a superframe chunk from explicit frame payloads with a
    /// `bytes_per_framesize` of `szb` (1..=4).
    fn build(frames: &[&[u8]], szb: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for f in frames {
            out.extend_from_slice(f);
        }
        let m = marker(szb as u8 - 1, frames.len() as u8 - 1);
        out.push(m);
        for f in frames {
            let sz = f.len();
            for b in 0..szb {
                out.push(((sz >> (8 * b)) & 0xff) as u8);
            }
        }
        out.push(m);
        out
    }

    fn parts(chunk: &[u8]) -> Vec<&[u8]> {
        split_superframe(chunk)
            .into_iter()
            .map(|(s, e)| &chunk[s..e])
            .collect()
    }

    #[test]
    fn non_superframe_returns_whole_chunk() {
        // A normal coded frame whose final byte is not a superframe_marker.
        let chunk = [0x82u8, 0x49, 0x83, 0x42, 0x00];
        let p = parts(&chunk);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], &chunk);
    }

    #[test]
    fn empty_chunk_yields_no_frames() {
        assert!(split_superframe(&[]).is_empty());
    }

    #[test]
    fn two_frame_superframe_one_byte_sizes() {
        let f0: Vec<u8> = (0..10u8).collect();
        let f1: Vec<u8> = (100..130u8).collect();
        let chunk = build(&[&f0, &f1], 1);
        let p = parts(&chunk);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], f0.as_slice());
        assert_eq!(p[1], f1.as_slice());
    }

    #[test]
    fn single_frame_superframe_is_legal() {
        // §B.3 NOTE — NumFrames == 1 is legal.
        let f0: Vec<u8> = (0..40u8).collect();
        let chunk = build(&[&f0], 1);
        let p = parts(&chunk);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], f0.as_slice());
    }

    #[test]
    fn multibyte_framesize_round_trips() {
        // A frame larger than 255 bytes needs SzBytes >= 2.
        let f0: Vec<u8> = (0..300u32).map(|v| v as u8).collect();
        let f1: Vec<u8> = (0..17u8).collect();
        let chunk = build(&[&f0, &f1], 2);
        let p = parts(&chunk);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].len(), 300);
        assert_eq!(p[1].len(), 17);
        assert_eq!(p[0], f0.as_slice());
    }

    #[test]
    fn mismatched_leading_index_byte_falls_back() {
        // §B.4 step 3: leading index byte must equal the trailing one.
        let f0: Vec<u8> = (0..10u8).collect();
        let f1: Vec<u8> = (0..20u8).collect();
        let mut chunk = build(&[&f0, &f1], 1);
        // SzIndex = 2 + NumFrames(2) * SzBytes(1) = 4.
        let lead = chunk.len() - 4;
        chunk[lead] ^= 0x01;
        let p = parts(&chunk);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], chunk.as_slice());
    }

    #[test]
    fn declared_sizes_overrunning_chunk_falls_back() {
        // A marker claiming sizes that exceed the payload region must not panic;
        // §B.4 fallback to a single frame.
        let mut chunk = vec![1u8, 2, 3, 4, 5];
        let m = marker(0, 1); // SzBytes=1, NumFrames=2
        chunk.push(m);
        chunk.push(200);
        chunk.push(200);
        chunk.push(m);
        let p = parts(&chunk);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], chunk.as_slice());
    }

    #[test]
    fn chunk_too_short_for_index_falls_back() {
        // Final byte looks like a marker but the chunk is shorter than SzIndex.
        let chunk = [marker(3, 7)]; // claims SzBytes=4, NumFrames=8 -> SzIndex=34
        let p = parts(&chunk);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], &chunk);
    }

    #[test]
    fn three_frame_superframe_with_hidden_first() {
        // A hidden ARF followed by a visible frame, both inside one chunk.
        let arf: Vec<u8> = (0..50u8).collect();
        let vis: Vec<u8> = (0..120u8).collect();
        let chunk = build(&[&arf, &vis], 1);
        let p = parts(&chunk);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], arf.as_slice());
        assert_eq!(p[1], vis.as_slice());
    }
}
