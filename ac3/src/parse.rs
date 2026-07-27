//! Self-syncing syncframe framer (ATSC A/52 §5.1). AC-3 / E-AC-3 is a
//! byte-stream of syncframes each beginning with the 16-bit `0B77` sync word;
//! the header carries the frame's own length (§5.1.2 frmsizecod / §E.1.2.2
//! frmsiz), so a reader locates and delimits frames by scanning for the sync
//! word and trusting the length — exactly like the MP3 framer, adapted to A/52.
//!
//! This lets the sink accept either a raw byte stream at arbitrary chunk
//! boundaries (a `filesrc` reading a `.ac3`, or an AVI audio chunk that may split
//! a frame) *and* pre-framed demuxer packets (one MKV Block per frame). Both are
//! the same to the framer: accumulate bytes, resync on `0B77`, hand each whole
//! frame to the core.
//!
//! ## Robustness (P0)
//! A frame whose declared length overruns the buffer is held for more input, not
//! read past (no OOB). A run with no sync word is bounded so pure garbage cannot
//! grow the buffer without limit. A frame whose header is structurally invalid is
//! skipped one byte and the scan resyncs (never a panic).

use crate::frame::Frame;

/// The 2-byte big-endian sync word (A/52 §5.1.1).
const SYNC_HI: u8 = 0x0B;
const SYNC_LO: u8 = 0x77;

/// Cap on buffered-but-unframed bytes. The largest AC-3 frame is 1280 words ×
/// 2 = 2560 bytes (48 kHz) and E-AC-3 frames are ≤ 2048 × 2; this holds many
/// whole frames and only bites on non-AC-3 input, where we drop the oldest bytes
/// keeping a sync-word-sized tail so a frame straddling the cut survives.
const MAX_BUFFERED: usize = 1 << 16;

/// The outcome of trying to frame the next syncframe out of a byte buffer.
pub enum Framed {
    /// A whole frame occupies `buf[offset..offset + len]`; junk before `offset`
    /// is skippable. `sample_rate` and `eac3` come from its header.
    Frame { offset: usize, len: usize, sample_rate: u32, eac3: bool },
    /// A valid header sits at `offset` but its frame is not fully buffered yet —
    /// drop the junk before `offset` and wait for more input. Never scan past
    /// this header for a shorter frame deeper in (the classic partial-frame bug).
    NeedMore { offset: usize },
    /// No frame header found anywhere in the buffer — pure junk.
    NoSync,
}

/// Frame the next syncframe out of `buf`, boundary-safely.
///
/// Scans for the `0B77` sync word, parses just enough header to get the frame
/// length (via [`Frame::peek_len`]), and confirms the candidate by checking the
/// *next* frame's sync word lands exactly at its end (the two-frame lock that
/// rejects a chance `0B 77` inside coded data). At end-of-buffer a single whole
/// frame is accepted so the stream tail still decodes.
pub fn next_frame(buf: &[u8]) -> Framed {
    let mut i = 0usize;
    while i + 2 <= buf.len() {
        if buf[i] != SYNC_HI || buf[i + 1] != SYNC_LO {
            i += 1;
            continue;
        }
        // Need at least a header's worth of bytes to derive the length.
        if i + 6 > buf.len() {
            return Framed::NeedMore { offset: i };
        }
        let (len, rate, eac3) = match Frame::peek_len(&buf[i..]) {
            Ok(v) => v,
            Err(_) => {
                i += 1;
                continue;
            }
        };
        if len < 6 {
            i += 1;
            continue;
        }
        if i + len > buf.len() {
            // Header valid but frame overruns — may be the real next frame still
            // arriving. Wait; never scan deeper into its payload.
            return Framed::NeedMore { offset: i };
        }
        // Confirm: the next frame's sync word should land at i+len. If the next
        // two bytes aren't buffered we can't confirm — but the frame is whole, so
        // accept it (the decoder's own header/CRC parse is the backstop, and the
        // stream tail must decode).
        let next = i + len;
        let confirmed = next + 2 > buf.len() || (buf[next] == SYNC_HI && buf[next + 1] == SYNC_LO);
        if confirmed {
            return Framed::Frame { offset: i, len, sample_rate: rate, eac3 };
        }
        // False sync inside coded data — scan one byte on.
        i += 1;
    }
    Framed::NoSync
}

/// Bound the buffer if it has grown past [`MAX_BUFFERED`] with no framable sync,
/// keeping only a sync-word-sized tail. Returns how many bytes to drop from the
/// front. The caller applies the drain.
pub fn overflow_drop(buf_len: usize) -> usize {
    if buf_len > MAX_BUFFERED {
        buf_len - 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal AC-3 48 kHz header: sync 0B77, crc1 (any), fscod=0,
    /// frmsizecod=... bsid=8. We build one 128-word (256-byte) frame's header and
    /// pad to length so the framer's length logic can be exercised without a full
    /// valid payload.
    fn synth_ac3_header(frmsizecod: u8) -> Vec<u8> {
        // 16 sync + 16 crc + 2 fscod + 6 frmsizecod + 5 bsid + ...
        // Build bits MSB-first.
        let mut bits: Vec<u8> = Vec::new();
        let push = |bits: &mut Vec<u8>, v: u32, n: u32| {
            for k in (0..n).rev() {
                bits.push(((v >> k) & 1) as u8);
            }
        };
        push(&mut bits, 0x0B77, 16); // sync
        push(&mut bits, 0x0000, 16); // crc1
        push(&mut bits, 0, 2); // fscod = 48k
        push(&mut bits, frmsizecod as u32, 6);
        push(&mut bits, 8, 5); // bsid = 8 (AC-3)
        // pad to byte boundary
        while !bits.len().is_multiple_of(8) {
            bits.push(0);
        }
        let mut bytes = vec![0u8; bits.len() / 8];
        for (i, b) in bits.iter().enumerate() {
            bytes[i / 8] |= b << (7 - (i % 8));
        }
        bytes
    }

    #[test]
    fn finds_and_measures_ac3_frame() {
        // frmsizecod 0 → 64 words → 128 bytes at 48 kHz.
        let hdr = synth_ac3_header(0);
        let mut frame = hdr.clone();
        frame.resize(128, 0);
        // Two frames back to back so the two-frame lock confirms the first.
        let mut stream = frame.clone();
        stream.extend_from_slice(&frame);
        match next_frame(&stream) {
            Framed::Frame { offset, len, sample_rate, eac3 } => {
                assert_eq!(offset, 0);
                assert_eq!(len, 128);
                assert_eq!(sample_rate, 48_000);
                assert!(!eac3);
            }
            _ => panic!("expected a framed AC-3 frame"),
        }
    }

    #[test]
    fn skips_junk_before_sync() {
        let hdr = synth_ac3_header(0);
        let mut frame = hdr.clone();
        frame.resize(128, 0);
        let mut stream = vec![0xAA, 0xBB, 0xCC];
        stream.extend_from_slice(&frame);
        stream.extend_from_slice(&frame);
        match next_frame(&stream) {
            Framed::Frame { offset, len, .. } => {
                assert_eq!(offset, 3);
                assert_eq!(len, 128);
            }
            _ => panic!("expected a framed frame after junk"),
        }
    }

    #[test]
    fn partial_frame_waits() {
        let hdr = synth_ac3_header(0);
        let mut frame = hdr.clone();
        frame.resize(64, 0); // only half the 128-byte frame buffered
        match next_frame(&frame) {
            Framed::NeedMore { offset } => assert_eq!(offset, 0),
            _ => panic!("expected NeedMore for a partial frame"),
        }
    }

    #[test]
    fn pure_junk_no_sync() {
        assert!(matches!(next_frame(&[0x12, 0x34, 0x56, 0x78]), Framed::NoSync));
    }
}
