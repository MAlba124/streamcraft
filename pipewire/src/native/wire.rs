//! The PipeWire native-protocol message framing (see `REFERENCES.md` → Native Protocol; matches
//! `module-protocol-native/connection.c`). Every message is a **16-byte header** of four
//! native-endian `u32` words followed by a single [`pod::PodReader`]-parseable `Struct` payload
//! (the method/event args) and an **optional footer** POD the reader skips:
//!
//! ```text
//! ┌──────────┬──────────────────────┬──────────┬──────────┬── Struct(args) ──┬─ footer? ─┐
//! │ id  u32  │ (opcode<<24)|size u32 │ seq  u32 │ n_fds u32 │  the arg tuple   │ (skipped) │
//! └──────────┴──────────────────────┴──────────┴──────────┴──────────────────┴───────────┘
//! ```
//!
//! `size` (the low 24 bits of word 1) is the byte length of everything **after** the header —
//! payload Struct **plus** any footer — so `16 + size` is the offset of the next message. We
//! never emit a footer (it is optional for the client); we skip an incoming one by reading only
//! the leading `Struct`. `opcode` is the interface-relative method/event number.

/// Header size in bytes (four `u32` words).
pub const HEADER: usize = 16;

/// A decoded message header. The payload (`Struct` of args) begins at byte 16 of the message.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    /// Destination object id (0 = Core, 1 = Client, or a client-allocated proxy id).
    pub id: u32,
    /// Interface-relative method (client→server) or event (server→client) opcode.
    pub opcode: u8,
    /// Byte length of the payload + footer that follow the header.
    pub size: u32,
    /// Monotonic sequence number (echoed in `Core.Done`/`Core.Ping` round-trips).
    pub seq: u32,
    /// Number of file descriptors that rode this message's `SCM_RIGHTS`.
    pub n_fds: u32,
}

impl Header {
    /// Total bytes this message occupies (`16 + size`), i.e. the offset of the next one.
    #[inline]
    pub fn total(&self) -> usize {
        HEADER + self.size as usize
    }
}

/// Parse a whole message at the front of `data`, returning its header and total byte length, or
/// `None` if `data` holds only a partial message (the caller keeps the bytes and reads more).
pub fn parse(data: &[u8]) -> Option<(Header, usize)> {
    if data.len() < HEADER {
        return None;
    }
    let id = rd(data, 0);
    let word1 = rd(data, 4);
    let opcode = (word1 >> 24) as u8;
    let size = word1 & 0x00ff_ffff;
    let seq = rd(data, 8);
    let n_fds = rd(data, 12);
    let total = HEADER + size as usize;
    if data.len() < total {
        return None;
    }
    Some((Header { id, opcode, size, seq, n_fds }, total))
}

#[inline]
fn rd(data: &[u8], at: usize) -> u32 {
    u32::from_ne_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

/// Write a message header into `buf` with a placeholder word 1, returning the header's offset so
/// [`finish`] can back-patch `(opcode<<24)|size` once the payload length is known.
pub fn begin(buf: &mut Vec<u8>, id: u32, seq: u32) -> usize {
    let at = buf.len();
    buf.extend_from_slice(&id.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes()); // word1 placeholder: (opcode<<24)|size
    buf.extend_from_slice(&seq.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes()); // n_fds placeholder
    at
}

/// Back-patch a header begun at `header` (offset into `buf`): pack `opcode` + the payload byte
/// length (everything after the 16-byte header) into word 1, and write `n_fds` into word 3.
pub fn finish(buf: &mut [u8], header: usize, opcode: u8, n_fds: u32) {
    let size = (buf.len() - header - HEADER) as u32;
    debug_assert!(size < (1 << 24), "message payload exceeds the 24-bit size field");
    let word1 = ((opcode as u32) << 24) | size;
    buf[header + 4..header + 8].copy_from_slice(&word1.to_ne_bytes());
    buf[header + 12..header + 16].copy_from_slice(&n_fds.to_ne_bytes());
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests build throwaway message buffers
mod tests {
    use super::*;
    use crate::native::pod::{PodBuilder, PodReader};

    #[test]
    fn header_round_trips_opcode_and_size() {
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        let h = begin(&mut buf, 0, 5);
        {
            let mut b = PodBuilder::new(&mut buf, &mut fds);
            b.push_struct();
            b.int(3); // e.g. Core.Hello(version=3)
            b.pop();
        }
        finish(&mut buf, h, 1 /* Hello */, 0);

        let (hdr, total) = parse(&buf).expect("one whole message");
        assert_eq!(total, buf.len());
        assert_eq!(hdr.id, 0);
        assert_eq!(hdr.opcode, 1);
        assert_eq!(hdr.seq, 5);
        assert_eq!(hdr.n_fds, 0);
        assert_eq!(hdr.total(), buf.len());
        // Payload parses as Struct(Int(3)).
        let mut args = PodReader::new(&buf[HEADER..hdr.total()]).enter_struct().unwrap();
        assert_eq!(args.int(), Some(3));
    }

    #[test]
    fn partial_message_reports_none() {
        assert!(parse(&[0u8; 8]).is_none());
        // Header claims 16 payload bytes, but none are present.
        let mut buf = Vec::new();
        let h = begin(&mut buf, 7, 0);
        // Manually stamp a size of 16 without appending a body.
        let word1 = 16u32;
        buf[h + 4..h + 8].copy_from_slice(&word1.to_ne_bytes());
        assert!(parse(&buf).is_none());
    }

    #[test]
    fn skips_a_trailing_footer() {
        // Build header + Struct(Int(1)) + a bogus 16-byte "footer"; the reader must find the
        // next message right after `size`, and parse only the leading struct.
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        let h = begin(&mut buf, 0, 0);
        {
            let mut b = PodBuilder::new(&mut buf, &mut fds);
            b.push_struct();
            b.int(1);
            b.pop();
        }
        // Append a fake footer (as if the v3 server sent a generation footer).
        {
            let mut b = PodBuilder::new(&mut buf, &mut fds);
            b.push_struct();
            b.long(99);
            b.pop();
        }
        finish(&mut buf, h, 1, 0);

        let (hdr, total) = parse(&buf).unwrap();
        assert_eq!(total, buf.len(), "size spans struct + footer");
        // Reading only the leading struct yields the arg and stops before the footer.
        let mut args = PodReader::new(&buf[HEADER..hdr.total()]).enter_struct().unwrap();
        assert_eq!(args.int(), Some(1));
    }
}
