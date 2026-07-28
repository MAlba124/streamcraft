//! The Wayland wire format (protocol spec §Wire Format): a stream of 32-bit-aligned,
//! native-endian messages, each an 8-byte header followed by typed arguments.
//!
//! ```text
//! ┌─────────────┬────────┬────────┬──── args (each padded to 4 bytes) ────┐
//! │ object id   │ size   │ opcode │ i32 | u32 | fixed | string | array... │
//! │  u32        │ u16    │ u16    │  (fd args ride SCM_RIGHTS, not here)   │
//! └─────────────┴────────┴────────┴───────────────────────────────────────┘
//! ```
//! (`size` is the whole message length in bytes, header included; on the wire the second
//! word is `(size << 16) | opcode`.)
//!
//! Everything here writes into a **caller-owned reused buffer** and reads out of a received
//! byte window — no per-message heap allocation, which is the entire point of talking the
//! wire ourselves instead of via libwayland's `wl_closure`-per-call marshaller.

use std::os::unix::io::RawFd;

/// A 24.8 signed fixed-point number (`wl_fixed`), used for surface-local coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fixed(pub i32);

impl Fixed {
    /// The integer `n` as `wl_fixed` (`n << 8`).
    pub fn from_int(n: i32) -> Self {
        Fixed(n << 8)
    }
    /// The integer part (truncating toward zero), for pointer coordinates.
    pub fn to_int(self) -> i32 {
        self.0 >> 8
    }
}

/// Appends one request into `buf` (and any fds into `fds`), backpatching the size on
/// [`finish`](Writer::finish). Borrows the connection's reused send buffers, so building a
/// request allocates nothing.
pub struct Writer<'a> {
    buf: &'a mut Vec<u8>,
    fds: &'a mut Vec<RawFd>,
    /// Offset of this message's header in `buf`, for the size backpatch.
    header: usize,
}

impl<'a> Writer<'a> {
    /// Begin a message from `object`, request `opcode`. The 8-byte header is written with a
    /// placeholder size that [`finish`](Self::finish) backpatches.
    pub fn new(buf: &'a mut Vec<u8>, fds: &'a mut Vec<RawFd>, object: u32, opcode: u16) -> Self {
        let header = buf.len();
        buf.extend_from_slice(&object.to_ne_bytes());
        // Word 2 is `(size << 16) | opcode`; stash the opcode now (low 16 bits) and let
        // `finish` OR the final size into the high 16 once the body length is known.
        buf.extend_from_slice(&u32::from(opcode).to_ne_bytes());
        Writer { buf, fds, header }
    }

    /// A `uint`/`new_id`(numeric)/`object` argument.
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// An `int` argument.
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// A `fixed` (24.8) argument.
    pub fn fixed(&mut self, v: Fixed) -> &mut Self {
        self.i32(v.0)
    }

    /// An `object` id argument (`0` = null).
    pub fn object(&mut self, id: u32) -> &mut Self {
        self.u32(id)
    }

    /// A `new_id` argument for a **known** interface (just the numeric id).
    pub fn new_id(&mut self, id: u32) -> &mut Self {
        self.u32(id)
    }

    /// A `string` argument: length (incl. trailing NUL) then the bytes + NUL, padded to 4.
    /// A Wayland string is not NUL-terminated in Rust terms; we add the wire NUL here.
    pub fn string(&mut self, s: &str) -> &mut Self {
        let len = s.len() + 1; // + NUL
        self.u32(len as u32);
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        self.pad();
        self
    }

    /// A `new_id` argument for `wl_registry.bind`, which carries the bound interface's
    /// name + version inline: `string(interface) uint(version) new_id(id)`.
    pub fn bind_new_id(&mut self, interface: &str, version: u32, id: u32) -> &mut Self {
        self.string(interface).u32(version).u32(id)
    }

    /// An `array` argument: length then raw bytes, padded to 4.
    pub fn array(&mut self, bytes: &[u8]) -> &mut Self {
        self.u32(bytes.len() as u32);
        self.buf.extend_from_slice(bytes);
        self.pad();
        self
    }

    /// An `fd` argument — sent out-of-band via `SCM_RIGHTS`, contributing nothing to the
    /// message body (the receiver pairs fds to fd-args positionally).
    pub fn fd(&mut self, fd: RawFd) -> &mut Self {
        self.fds.push(fd);
        self
    }

    /// Pad `buf` up to the next 4-byte boundary with zeros.
    fn pad(&mut self) {
        while self.buf.len() % 4 != 0 {
            self.buf.push(0);
        }
    }

    /// Backpatch the header's size field with the finished message length. Takes `&mut self`
    /// so it terminates a builder chain (`req.u32(..).finish()`); the `Writer` is dropped at
    /// the end of the statement.
    pub fn finish(&mut self) {
        let size = (self.buf.len() - self.header) as u32;
        debug_assert!(size >= 8 && size % 4 == 0, "wayland message must be a whole ≥8-byte word");
        let word = (size << 16) | u32::from(self.opcode());
        self.buf[self.header + 4..self.header + 8].copy_from_slice(&word.to_ne_bytes());
    }

    /// The opcode stashed in the placeholder header (low 16 bits of word 2).
    fn opcode(&self) -> u16 {
        let b = &self.buf[self.header + 4..self.header + 8];
        (u32::from_ne_bytes([b[0], b[1], b[2], b[3]]) & 0xffff) as u16
    }
}

/// One decoded incoming message (event), borrowing the receive buffer.
#[derive(Clone, Copy, Debug)]
pub struct Event<'a> {
    /// The object that emitted the event.
    pub object: u32,
    /// The event opcode (interface-relative).
    pub opcode: u16,
    /// The argument bytes (everything after the 8-byte header).
    pub args: &'a [u8],
}

/// The number of whole messages available at the front of `data`, and a way to walk them.
/// Returns `None`/stops at the first partial message (the caller keeps those bytes for the
/// next read).
pub fn parse_message(data: &[u8]) -> Option<(Event<'_>, usize)> {
    if data.len() < 8 {
        return None;
    }
    let object = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
    let word2 = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
    let size = (word2 >> 16) as usize;
    let opcode = (word2 & 0xffff) as u16;
    if size < 8 || size % 4 != 0 || data.len() < size {
        return None;
    }
    Some((Event { object, opcode, args: &data[8..size] }, size))
}

/// Pulls typed arguments out of an [`Event`]'s `args` in order.
pub struct ArgReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ArgReader<'a> {
    pub fn new(args: &'a [u8]) -> Self {
        ArgReader { data: args, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.data.len() {
            return None;
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Some(s)
    }

    pub fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i32(&mut self) -> Option<i32> {
        Some(self.u32()? as i32)
    }

    pub fn fixed(&mut self) -> Option<Fixed> {
        Some(Fixed(self.i32()?))
    }

    /// A wire string: the `len`-prefixed bytes minus the trailing NUL, then 4-byte pad skipped.
    pub fn string(&mut self) -> Option<&'a str> {
        let len = self.u32()? as usize;
        if len == 0 {
            return Some("");
        }
        let bytes = self.take(len)?;
        // Consume padding to the next 4-byte boundary.
        let pad = (4 - (len % 4)) % 4;
        self.take(pad)?;
        // Drop the trailing NUL; tolerate malformed non-UTF8 as an empty match.
        std::str::from_utf8(&bytes[..len.saturating_sub(1)]).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_bind_request() {
        // wl_registry@2.bind(name=7, "wl_compositor", version=4, new_id=9)
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        {
            let mut w = Writer::new(&mut buf, &mut fds, 2, 0);
            w.u32(7).bind_new_id("wl_compositor", 4, 9);
            w.finish();
        }
        assert!(fds.is_empty());
        // Header: object=2, then (size<<16 | opcode=0).
        let (ev, consumed) = parse_message(&buf).expect("one whole message");
        assert_eq!(consumed, buf.len());
        assert_eq!(ev.object, 2);
        assert_eq!(ev.opcode, 0);
        let mut r = ArgReader::new(ev.args);
        assert_eq!(r.u32(), Some(7));
        assert_eq!(r.string(), Some("wl_compositor"));
        assert_eq!(r.u32(), Some(4));
        assert_eq!(r.u32(), Some(9));
    }

    #[test]
    fn message_is_4_byte_aligned_and_sized() {
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        {
            // A string of length 3 ("abc") → 4 (len) + 4 ("abc\0") = 8 arg bytes, no extra pad.
            let mut w = Writer::new(&mut buf, &mut fds, 1, 5);
            w.string("abc");
            w.finish();
        }
        assert_eq!(buf.len() % 4, 0);
        let word2 = u32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!((word2 >> 16) as usize, buf.len(), "size field == message length");
        assert_eq!((word2 & 0xffff) as u16, 5, "opcode preserved");
    }

    #[test]
    fn fd_args_go_to_the_side_channel_not_the_body() {
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        {
            let mut w = Writer::new(&mut buf, &mut fds, 3, 1);
            w.u32(100).fd(42).u32(200);
            w.finish();
        }
        // Body carries the two u32s only; the fd rode `fds`.
        assert_eq!(fds, vec![42]);
        let (ev, _) = parse_message(&buf).unwrap();
        let mut r = ArgReader::new(ev.args);
        assert_eq!(r.u32(), Some(100));
        assert_eq!(r.u32(), Some(200));
        assert_eq!(ev.args.len(), 8);
    }

    #[test]
    fn partial_message_reports_none() {
        assert!(parse_message(&[0u8; 4]).is_none());
        // Claims size 16 but only 8 bytes present.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_ne_bytes());
        buf.extend_from_slice(&((16u32 << 16) | 2).to_ne_bytes());
        assert!(parse_message(&buf).is_none());
    }
}
