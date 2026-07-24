//! The Wayland wire format (spec: `wayland/spec/wayland.xml`; the marshalling rules are
//! from the protocol's preamble / the Wayland book's "Wire format"). Hand-rolled — no
//! `wayland-client`, no scanner-generated code.
//!
//! Every message is a request or event on some object:
//!
//! ```text
//!   +0  u32   object id (sender's target)
//!   +4  u16   opcode          } packed into one u32 as (size << 16) | opcode, LE,
//!   +6  u16   size (bytes)    } so the size occupies the high half-word.
//!   +8  args … (each 32-bit aligned, LE)
//! ```
//!
//! Argument encodings (`wayland.xml` arg `type`s):
//! * `int` / `uint` / `object` / `new_id` (unnamed) — one 32-bit word.
//! * `new_id` with a named interface — one 32-bit word (the id we allocate).
//! * `string` — `u32` length *including* the trailing NUL, then the bytes, then the
//!   NUL, then zero-padding up to a 32-bit boundary. A null string is length 0.
//! * `array` — `u32` byte length, then the bytes, then padding to 32 bits.
//! * `fd` — carried **out of band** in the ancillary `SCM_RIGHTS` data, *not* in the
//!   message body (so it contributes no bytes here); see [`crate::sys::send_with_fd`].
//!
//! `size` is the total message length in bytes, header included, and is a multiple of 4.
//! All integers are little-endian on the wire; every mainstream Wayland platform is LE and
//! the protocol is defined that way, so we hard-code LE and reject a big-endian build.

#[cfg(target_endian = "big")]
compile_error!("the Wayland wire protocol is little-endian; big-endian is unsupported");

/// Round `n` up to the next multiple of 4 (32-bit alignment), used for string/array
/// padding. Saturating so a corrupt huge length cannot overflow into a small value.
#[inline]
pub fn pad4(n: usize) -> usize {
    n.saturating_add(3) & !3
}

/// A request being built for the socket. Accumulates the header + argument words into a
/// byte vector; [`finish`](MessageBuilder::finish) back-patches the size field once the
/// length is known.
///
/// Fd arguments are recorded separately (they ride in the ancillary data), so at most one
/// fd per message — which is all any request we send needs (`wl_shm.create_pool`).
pub struct MessageBuilder {
    object: u32,
    opcode: u16,
    body: Vec<u8>,
    /// A file descriptor to send out-of-band with this message (`SCM_RIGHTS`), if any.
    fd: Option<i32>,
}

impl MessageBuilder {
    /// Start a request `opcode` targeting object `object`.
    pub fn new(object: u32, opcode: u16) -> Self {
        Self {
            object,
            opcode,
            body: Vec::with_capacity(32),
            fd: None,
        }
    }

    /// Append a 32-bit `uint`/`object`/`new_id` word (LE).
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.body.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Append a signed 32-bit `int` word (LE).
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.u32(v as u32)
    }

    /// Append a `string`: length (incl. NUL) + bytes + NUL + pad to 32 bits. `wayland.xml`
    /// strings are NUL-terminated and 32-bit aligned.
    pub fn string(&mut self, s: &str) -> &mut Self {
        let len = s.len() + 1; // include the trailing NUL
        self.u32(len as u32);
        self.body.extend_from_slice(s.as_bytes());
        self.body.push(0);
        while self.body.len() % 4 != 0 {
            self.body.push(0);
        }
        self
    }

    /// Record a file descriptor to send in this message's ancillary data (no body bytes).
    pub fn fd(&mut self, fd: i32) -> &mut Self {
        self.fd = Some(fd);
        self
    }

    /// The out-of-band fd, if one was attached.
    pub fn take_fd(&self) -> Option<i32> {
        self.fd
    }

    /// Serialise to the on-wire bytes with the size field back-patched. The result is a
    /// multiple of 4 bytes (header is 8, every arg word is 4, strings/arrays pad to 4).
    pub fn finish(&self) -> Vec<u8> {
        let size = 8 + self.body.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&self.object.to_le_bytes());
        // size in the high 16 bits, opcode in the low 16 bits, one LE word.
        let packed = ((size as u32) << 16) | self.opcode as u32;
        out.extend_from_slice(&packed.to_le_bytes());
        out.extend_from_slice(&self.body);
        out
    }
}

/// A parsed event header: which object, which opcode, and the total message size in bytes
/// (header included). The body is `size - 8` bytes following the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub object: u32,
    pub opcode: u16,
    pub size: u16,
}

impl Header {
    pub const LEN: usize = 8;

    /// Parse an 8-byte header, or `None` if `buf` is too short. Never panics on short
    /// input — malformed server bytes are a protocol error, not a crash.
    pub fn parse(buf: &[u8]) -> Option<Header> {
        if buf.len() < Self::LEN {
            return None;
        }
        let object = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let packed = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        Some(Header {
            object,
            opcode: (packed & 0xFFFF) as u16,
            size: (packed >> 16) as u16,
        })
    }
}

/// A cursor over one event's argument body, decoding the same POD types the builder
/// encodes. Every getter is total: it returns `None` past the end rather than panicking,
/// so a truncated or malformed event can be rejected as an [`Error::Resource`] upstream.
///
/// [`Error::Resource`]: streamcraft_core::error::Error::Resource
pub struct ArgReader<'a> {
    body: &'a [u8],
    pos: usize,
}

impl<'a> ArgReader<'a> {
    /// Read over an event body (the bytes *after* the 8-byte header).
    pub fn new(body: &'a [u8]) -> Self {
        Self { body, pos: 0 }
    }

    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.body.len().saturating_sub(self.pos)
    }

    /// Next 32-bit `uint`/`object`/`new_id` word (LE), or `None` if the body is exhausted.
    pub fn u32(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        let word = self.body.get(self.pos..end)?;
        self.pos = end;
        Some(u32::from_le_bytes(word.try_into().ok()?))
    }

    /// Next signed 32-bit `int` word.
    pub fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }

    /// Next `string`: a length word (incl. NUL) then that many bytes, NUL-terminated and
    /// padded to 32 bits. Returns the string without its NUL; a length of 0 is a null
    /// string (surfaced as an empty string). Rejects a length that runs past the body.
    pub fn string(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        if len == 0 {
            return Some(String::new()); // null string
        }
        // `len` includes the trailing NUL; the payload occupies `pad4(len)` bytes.
        let padded = pad4(len);
        let end = self.pos.checked_add(padded)?;
        let bytes = self.body.get(self.pos..self.pos + len)?;
        // Drop the trailing NUL, lossily decode (server strings are UTF-8/ASCII).
        let s = &bytes[..len - 1];
        self.pos = end;
        Some(String::from_utf8_lossy(s).into_owned())
    }

    /// Next `array`: a length word then that many bytes, padded to 32 bits. Returns the
    /// raw bytes. Used for `xdg_toplevel.configure`'s `states`.
    pub fn array(&mut self) -> Option<&'a [u8]> {
        let len = self.u32()? as usize;
        let padded = pad4(len);
        let end = self.pos.checked_add(padded)?;
        let bytes = self.body.get(self.pos..self.pos + len)?;
        self.pos = end;
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad4_rounds_up_to_word() {
        assert_eq!(pad4(0), 0);
        assert_eq!(pad4(1), 4);
        assert_eq!(pad4(4), 4);
        assert_eq!(pad4(5), 8);
        assert_eq!(pad4(7), 8);
        // A near-usize::MAX length must not wrap to a small value.
        assert_eq!(pad4(usize::MAX), usize::MAX & !3);
    }

    #[test]
    fn header_packs_size_and_opcode() {
        // wl_surface.commit (opcode 6) on object 5: no args, size 8.
        let m = MessageBuilder::new(5, 6).finish();
        assert_eq!(m.len(), 8);
        let h = Header::parse(&m).unwrap();
        assert_eq!(h.object, 5);
        assert_eq!(h.opcode, 6);
        assert_eq!(h.size, 8);
    }

    #[test]
    fn request_with_int_args_round_trips_through_header_and_argreader() {
        // wl_surface.damage(x=1, y=2, w=640, h=360) — four ints.
        let mut b = MessageBuilder::new(9, 2);
        b.i32(1).i32(2).i32(640).i32(360);
        let bytes = b.finish();
        assert_eq!(bytes.len(), 8 + 16);
        let h = Header::parse(&bytes).unwrap();
        assert_eq!((h.object, h.opcode, h.size), (9, 2, 24));
        let mut r = ArgReader::new(&bytes[Header::LEN..]);
        assert_eq!(r.i32(), Some(1));
        assert_eq!(r.i32(), Some(2));
        assert_eq!(r.i32(), Some(640));
        assert_eq!(r.i32(), Some(360));
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.i32(), None, "reading past the end yields None, never panics");
    }

    #[test]
    fn string_arg_is_nul_terminated_and_padded() {
        // "hi" -> len 3 (incl NUL), bytes "hi\0", pad one byte => 8 body bytes total.
        let mut b = MessageBuilder::new(1, 0);
        b.string("hi");
        let bytes = b.finish();
        // header(8) + len word(4) + "hi\0" + pad(1) => 8 + 4 + 4 = 16
        assert_eq!(bytes.len(), 16);
        let mut r = ArgReader::new(&bytes[Header::LEN..]);
        assert_eq!(r.string().as_deref(), Some("hi"));
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn empty_and_word_aligned_strings_pad_correctly() {
        // "" -> len 1 (just NUL), padded to 4 body bytes.
        let mut b = MessageBuilder::new(1, 0);
        b.string("");
        let bytes = b.finish();
        assert_eq!(bytes.len(), 8 + 4 + 4);
        // "abc" -> len 4 (incl NUL), already word-aligned, no extra pad.
        let mut b = MessageBuilder::new(1, 0);
        b.string("abc");
        let bytes = b.finish();
        assert_eq!(bytes.len(), 8 + 4 + 4);
        let mut r = ArgReader::new(&bytes[Header::LEN..]);
        assert_eq!(r.string().as_deref(), Some("abc"));
    }

    #[test]
    fn error_event_body_parses_object_code_message() {
        // Simulate a wl_display.error: object(uint), code(uint), message(string).
        let mut body = Vec::new();
        body.extend_from_slice(&7u32.to_le_bytes()); // offending object
        body.extend_from_slice(&12u32.to_le_bytes()); // code
        let msg = "boom";
        body.extend_from_slice(&((msg.len() + 1) as u32).to_le_bytes());
        body.extend_from_slice(msg.as_bytes());
        body.push(0);
        while body.len() % 4 != 0 {
            body.push(0);
        }
        let mut r = ArgReader::new(&body);
        assert_eq!(r.u32(), Some(7));
        assert_eq!(r.u32(), Some(12));
        assert_eq!(r.string().as_deref(), Some("boom"));
    }

    #[test]
    fn truncated_header_and_body_return_none_not_panic() {
        assert!(Header::parse(&[0u8; 4]).is_none());
        assert!(Header::parse(&[]).is_none());
        // A string length that claims more bytes than the body holds must be rejected.
        let mut body = Vec::new();
        body.extend_from_slice(&100u32.to_le_bytes()); // claims 100 bytes
        body.extend_from_slice(b"ab"); // but only 2 follow
        let mut r = ArgReader::new(&body);
        assert_eq!(r.string(), None, "over-long string length rejected, no panic");
    }

    #[test]
    fn array_arg_reads_bytes_and_pads() {
        // states array of two u32 enum values (8 bytes, already word-aligned).
        let mut body = Vec::new();
        body.extend_from_slice(&8u32.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&4u32.to_le_bytes());
        let mut r = ArgReader::new(&body);
        let arr = r.array().unwrap();
        assert_eq!(arr.len(), 8);
        assert_eq!(u32::from_le_bytes(arr[0..4].try_into().unwrap()), 1);
        assert_eq!(r.remaining(), 0);
    }
}
