//! The SPA POD ("Plain Old Data") binary serialization format — every value on the PipeWire
//! wire (method args, event args, format/buffer params) is a POD. This is a from-scratch,
//! zero-dependency builder + parser; it is the single most reused primitive in the native
//! client, so it writes into a **caller-owned reused buffer** (no per-message heap traffic —
//! the whole reason we speak the wire ourselves instead of via libpipewire's SPA builders).
//!
//! Wire layout (see `REFERENCES.md` → SPA POD; matches `spa/pod/pod.h` in PipeWire):
//!
//! ```text
//! ┌──────────┬──────────┬──── body (`size` bytes) ────┐
//! │ size u32 │ type u32 │  value, padded up to 8 bytes │
//! └──────────┴──────────┴──────────────────────────────┘
//! ```
//!
//! `size` is the body length **excluding** the 8-byte header and **excluding** trailing
//! padding; on the wire every POD occupies `round_up_8(8 + size)` bytes. All words are
//! native-endian (both peers share the machine). A POD is thus self-describing and walkable.

#![allow(unsafe_code)] // none here; kept off — see conn.rs for the syscalls.
// One-time / setup-path allocations only (handshake + param serialization, not an
// `Element::process()` frame loop). The eventual RT audio path fills pre-sized buffers, not
// these `Vec`s. (spec: performance #1 — allocation discipline; clippy.toml.)
#![allow(clippy::disallowed_methods)]

use std::os::unix::io::RawFd;

// --- POD type ids (`enum spa_type`, the "basic" range; REFERENCES.md → SPA POD) ------------

pub const TYPE_NONE: u32 = 1;
pub const TYPE_BOOL: u32 = 2;
pub const TYPE_ID: u32 = 3;
pub const TYPE_INT: u32 = 4;
pub const TYPE_LONG: u32 = 5;
pub const TYPE_FLOAT: u32 = 6;
pub const TYPE_DOUBLE: u32 = 7;
pub const TYPE_STRING: u32 = 8;
pub const TYPE_BYTES: u32 = 9;
pub const TYPE_RECTANGLE: u32 = 10;
pub const TYPE_FRACTION: u32 = 11;
pub const TYPE_BITMAP: u32 = 12;
pub const TYPE_ARRAY: u32 = 13;
pub const TYPE_STRUCT: u32 = 14;
pub const TYPE_OBJECT: u32 = 15;
pub const TYPE_SEQUENCE: u32 = 16;
pub const TYPE_POINTER: u32 = 17;
pub const TYPE_FD: u32 = 18;
pub const TYPE_CHOICE: u32 = 19;
pub const TYPE_POD: u32 = 20;

/// Round `n` up to the next multiple of 8 (POD bodies and whole PODs are 8-aligned).
#[inline]
pub fn round8(n: usize) -> usize {
    (n + 7) & !7
}

// --- builder -------------------------------------------------------------------------------

/// Appends PODs into a reused byte buffer, back-patching container sizes on [`pop`](Self::pop).
/// `fd` args are staged into a side vector (they ride `SCM_RIGHTS`, not the body — the POD only
/// stores their *index*), exactly like the message's fd handling.
pub struct PodBuilder<'a> {
    buf: &'a mut Vec<u8>,
    /// Message-level fd side channel; an [`fd`](Self::fd) arg pushes here and stores its index.
    fds: &'a mut Vec<RawFd>,
    /// `fds.len()` when this builder started — fd indices are relative to *this* message.
    fd_base: usize,
    /// Offsets of the `size` word of each open container (struct/object/array/choice), for
    /// the size back-patch in [`pop`](Self::pop). Setup-path; never in a frame loop.
    frames: Vec<usize>,
}

impl<'a> PodBuilder<'a> {
    /// Build into `buf`, staging fd args into `fds`. `fds` may be non-empty (a message can carry
    /// several args); indices are taken relative to its current length.
    pub fn new(buf: &'a mut Vec<u8>, fds: &'a mut Vec<RawFd>) -> Self {
        let fd_base = fds.len();
        PodBuilder { buf, fds, fd_base, frames: Vec::new() }
    }

    #[inline]
    fn w32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }

    /// Write the 8-byte POD header with a placeholder size, returning the header offset (so a
    /// container can back-patch `size` later). For fixed-size PODs pass the exact `size`.
    #[inline]
    fn header(&mut self, size: u32, ty: u32) -> usize {
        let at = self.buf.len();
        self.w32(size);
        self.w32(ty);
        at
    }

    /// Pad the buffer up to the next 8-byte boundary (POD bodies are 8-aligned on the wire).
    #[inline]
    fn pad8(&mut self) {
        while !self.buf.len().is_multiple_of(8) {
            self.buf.push(0);
        }
    }

    // -- scalars (each a whole 8-aligned POD) --

    pub fn none(&mut self) -> &mut Self {
        self.header(0, TYPE_NONE);
        self
    }
    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.header(4, TYPE_BOOL);
        self.w32(v as u32);
        self.pad8();
        self
    }
    /// An `Id` (an enum value from a SPA type namespace — e.g. a `SPA_AUDIO_FORMAT_*`).
    pub fn id(&mut self, v: u32) -> &mut Self {
        self.header(4, TYPE_ID);
        self.w32(v);
        self.pad8();
        self
    }
    pub fn int(&mut self, v: i32) -> &mut Self {
        self.header(4, TYPE_INT);
        self.w32(v as u32);
        self.pad8();
        self
    }
    pub fn long(&mut self, v: i64) -> &mut Self {
        self.header(8, TYPE_LONG);
        self.buf.extend_from_slice(&v.to_ne_bytes()); // already 8-aligned
        self
    }
    pub fn float(&mut self, v: f32) -> &mut Self {
        self.header(4, TYPE_FLOAT);
        self.w32(v.to_bits());
        self.pad8();
        self
    }
    pub fn double(&mut self, v: f64) -> &mut Self {
        self.header(8, TYPE_DOUBLE);
        self.buf.extend_from_slice(&v.to_bits().to_ne_bytes());
        self
    }
    /// A `Rectangle` (`width`,`height`) — image sizes in video formats.
    pub fn rectangle(&mut self, width: u32, height: u32) -> &mut Self {
        self.header(8, TYPE_RECTANGLE);
        self.w32(width);
        self.w32(height);
        self
    }
    /// A `Fraction` (`num`/`denom`) — frame rates.
    pub fn fraction(&mut self, num: u32, denom: u32) -> &mut Self {
        self.header(8, TYPE_FRACTION);
        self.w32(num);
        self.w32(denom);
        self
    }
    /// A `String` — length **includes** the trailing NUL; body padded to 8.
    pub fn string(&mut self, s: &str) -> &mut Self {
        let size = s.len() + 1;
        self.header(size as u32, TYPE_STRING);
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        self.pad8();
        self
    }
    /// A `Bytes` blob (body padded to 8).
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.header(b.len() as u32, TYPE_BYTES);
        self.buf.extend_from_slice(b);
        self.pad8();
        self
    }
    /// An `Fd` arg: the raw fd rides the message's `SCM_RIGHTS`; the POD stores only its
    /// **index** among this message's fds (`i64`). Used by the buffer/activation hand-off.
    pub fn fd(&mut self, fd: RawFd) -> &mut Self {
        let index = (self.fds.len() - self.fd_base) as i64;
        self.fds.push(fd);
        self.header(8, TYPE_FD);
        self.buf.extend_from_slice(&index.to_ne_bytes());
        self
    }

    // -- containers (open → write children → `pop`) --

    /// Open a `Struct` — a positional tuple of child PODs (every method/event arg list is one).
    pub fn push_struct(&mut self) -> &mut Self {
        let at = self.header(0, TYPE_STRUCT);
        self.frames.push(at);
        self
    }
    /// Open an `Object` of SPA type `object_type` with param id `object_id` (e.g.
    /// `Format`/`EnumFormat`). Fill it with [`property`](Self::property) + a value POD each.
    pub fn push_object(&mut self, object_type: u32, object_id: u32) -> &mut Self {
        let at = self.header(0, TYPE_OBJECT);
        self.w32(object_type);
        self.w32(object_id);
        self.frames.push(at);
        self
    }
    /// Begin one object property: `key` + `flags`, immediately followed by exactly one value
    /// POD (call a scalar/container method next). Only valid inside [`push_object`].
    pub fn property(&mut self, key: u32, flags: u32) -> &mut Self {
        self.w32(key);
        self.w32(flags);
        self
    }

    /// A PipeWire dict, marshalled as a **nested** `Struct{ Int(n_items), (String key, String
    /// value)* }` (matches libpipewire's `push_dict`). Used for `Client.UpdateProperties` and
    /// wherever the protocol carries a property list.
    pub fn dict(&mut self, items: &[(&str, &str)]) -> &mut Self {
        self.push_struct();
        self.int(items.len() as i32);
        for (k, v) in items {
            self.string(k).string(v);
        }
        self.pop()
    }
    /// Close the innermost open container, back-patching its `size` word.
    pub fn pop(&mut self) -> &mut Self {
        self.pad8();
        let at = self.frames.pop().expect("pop() without a matching push");
        let size = (self.buf.len() - at - 8) as u32;
        self.buf[at..at + 4].copy_from_slice(&size.to_ne_bytes());
        self
    }

    // -- homogeneous collections (children have NO per-element header) --

    /// An `Array` of `Id`s (`child_size`=4). Used for enumerated format choices etc.
    pub fn array_id(&mut self, values: &[u32]) -> &mut Self {
        self.array_raw(4, TYPE_ID, values.len(), |b| {
            for &v in values {
                b.extend_from_slice(&v.to_ne_bytes());
            }
        })
    }
    /// An `Array` of `Int`s.
    pub fn array_int(&mut self, values: &[i32]) -> &mut Self {
        self.array_raw(4, TYPE_INT, values.len(), |b| {
            for &v in values {
                b.extend_from_slice(&v.to_ne_bytes());
            }
        })
    }
    fn array_raw(
        &mut self,
        child_size: u32,
        child_type: u32,
        n: usize,
        fill: impl FnOnce(&mut Vec<u8>),
    ) -> &mut Self {
        // body = child_size(u32) child_type(u32) then n children of child_size bytes, packed.
        let at = self.header(0, TYPE_ARRAY);
        self.w32(child_size);
        self.w32(child_type);
        fill(self.buf);
        let size = (self.buf.len() - at - 8) as u32;
        debug_assert_eq!(size, 8 + child_size * n as u32);
        self.buf[at..at + 4].copy_from_slice(&size.to_ne_bytes());
        self.pad8();
        self
    }

    /// A `Choice` of `Id`s: `choice_type` (0=None, 1=Range, 2=Step, 3=Enum, 4=Flags),
    /// `values[0]` is the default/preferred, the rest are alternatives. Format negotiation
    /// offers a broad Enum choice the server narrows.
    pub fn choice_id(&mut self, choice_type: u32, values: &[u32]) -> &mut Self {
        self.choice_raw(choice_type, 4, TYPE_ID, values.len(), |b| {
            for &v in values {
                b.extend_from_slice(&v.to_ne_bytes());
            }
        })
    }
    /// A `Choice` of `Int`s.
    pub fn choice_int(&mut self, choice_type: u32, values: &[i32]) -> &mut Self {
        self.choice_raw(choice_type, 4, TYPE_INT, values.len(), |b| {
            for &v in values {
                b.extend_from_slice(&v.to_ne_bytes());
            }
        })
    }
    fn choice_raw(
        &mut self,
        choice_type: u32,
        child_size: u32,
        child_type: u32,
        _n: usize,
        fill: impl FnOnce(&mut Vec<u8>),
    ) -> &mut Self {
        // body = choice_type(u32) flags(u32) child_size(u32) child_type(u32) then packed values.
        let at = self.header(0, TYPE_CHOICE);
        self.w32(choice_type);
        self.w32(0); // flags
        self.w32(child_size);
        self.w32(child_type);
        fill(self.buf);
        let size = (self.buf.len() - at - 8) as u32;
        self.buf[at..at + 4].copy_from_slice(&size.to_ne_bytes());
        self.pad8();
        self
    }
}

// --- parser --------------------------------------------------------------------------------

/// A cursor over a sequence of PODs (a message payload, or a container's body). Typed getters
/// read the *next* POD and interpret it; positional reads mirror how args are laid out.
#[derive(Clone, Copy)]
pub struct PodReader<'a> {
    data: &'a [u8],
    pos: usize,
}

/// One raw POD peeled off the cursor: its `type` and its `size` bytes of body (unpadded).
#[derive(Clone, Copy)]
pub struct RawPod<'a> {
    pub ty: u32,
    pub body: &'a [u8],
}

impl<'a> PodReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        PodReader { data, pos: 0 }
    }

    /// An empty cursor (yields `None` for everything) — the graceful-failure sentinel.
    pub fn empty() -> Self {
        PodReader { data: &[], pos: 0 }
    }

    fn rd_u32(buf: &[u8], at: usize) -> Option<u32> {
        let b = buf.get(at..at + 4)?;
        Some(u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Peel the next raw POD, advancing the cursor past its 8-aligned footprint. `None` at end
    /// or on a truncated / buffer-overrunning POD.
    pub fn next_pod(&mut self) -> Option<RawPod<'a>> {
        let size = Self::rd_u32(self.data, self.pos)? as usize;
        let ty = Self::rd_u32(self.data, self.pos + 4)?;
        let body_start = self.pos + 8;
        let body = self.data.get(body_start..body_start + size)?;
        self.pos = body_start + round8(size);
        Some(RawPod { ty, body })
    }

    /// The type of the next POD without consuming it (`None` at end).
    pub fn peek_type(&self) -> Option<u32> {
        Self::rd_u32(self.data, self.pos + 4)
    }

    fn expect(&mut self, want: u32) -> Option<RawPod<'a>> {
        let p = self.next_pod()?;
        (p.ty == want).then_some(p)
    }

    pub fn int(&mut self) -> Option<i32> {
        let p = self.expect(TYPE_INT)?;
        Some(Self::rd_u32(p.body, 0)? as i32)
    }
    pub fn id(&mut self) -> Option<u32> {
        Self::rd_u32(self.expect(TYPE_ID)?.body, 0)
    }
    pub fn long(&mut self) -> Option<i64> {
        let b = self.expect(TYPE_LONG)?.body;
        let a = b.get(0..8)?;
        Some(i64::from_ne_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]))
    }
    pub fn bool(&mut self) -> Option<bool> {
        Some(Self::rd_u32(self.expect(TYPE_BOOL)?.body, 0)? != 0)
    }
    /// The next `String`, minus its trailing NUL. `None` if the next POD is not a String or is
    /// not valid UTF-8.
    pub fn string(&mut self) -> Option<&'a str> {
        let body = self.expect(TYPE_STRING)?.body;
        let s = body.strip_suffix(&[0]).unwrap_or(body);
        std::str::from_utf8(s).ok()
    }

    /// Enter the next `Struct` (or `Object`/`Choice`), returning a cursor over its body. For an
    /// Object the returned cursor is positioned *after* the `object_type`/`object_id` words; use
    /// [`enter_object`](Self::enter_object) when you need those.
    pub fn enter_struct(&mut self) -> Option<PodReader<'a>> {
        Some(PodReader::new(self.expect(TYPE_STRUCT)?.body))
    }

    /// Enter the next `Object`, returning `(object_type, object_id, body_cursor)` where the
    /// cursor is positioned at the first property (`key`,`flags`,value…).
    pub fn enter_object(&mut self) -> Option<(u32, u32, ObjectReader<'a>)> {
        let p = self.expect(TYPE_OBJECT)?;
        let object_type = Self::rd_u32(p.body, 0)?;
        let object_id = Self::rd_u32(p.body, 4)?;
        Some((object_type, object_id, ObjectReader { r: PodReader::new(&p.body[8..]) }))
    }

    /// Read a PipeWire dict — a **nested** `Struct{ Int(n), (String key, String value)* }`
    /// (matches libpipewire's `parse_dict_struct`) — from the next POD, invoking `f(key, value)`
    /// for each entry.
    pub fn dict(&mut self, mut f: impl FnMut(&'a str, &'a str)) -> Option<()> {
        let mut s = self.enter_struct()?;
        let n = s.int()?;
        for _ in 0..n.max(0) {
            let k = s.string()?;
            let v = s.string()?;
            f(k, v);
        }
        Some(())
    }

    /// Skip the next POD regardless of type.
    pub fn skip(&mut self) -> Option<()> {
        self.next_pod().map(|_| ())
    }
}

/// A cursor over an `Object`'s properties. Each `next()` yields `(key, flags, value_cursor)`
/// where the value cursor's first POD is the property value.
pub struct ObjectReader<'a> {
    r: PodReader<'a>,
}

impl<'a> ObjectReader<'a> {
    /// The next property as `(key, flags, value_cursor)`, or `None` past the last one.
    pub fn next_prop(&mut self) -> Option<(u32, u32, PodReader<'a>)> {
        let key = PodReader::rd_u32(self.r.data, self.r.pos)?;
        let flags = PodReader::rd_u32(self.r.data, self.r.pos + 4)?;
        self.r.pos += 8;
        let value_at = self.r.pos;
        // Advance past the value POD to position for the next property.
        self.r.next_pod()?;
        Some((key, flags, PodReader::new(&self.r.data[value_at..self.r.pos])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(f: impl FnOnce(&mut PodBuilder)) -> (Vec<u8>, Vec<RawFd>) {
        let mut buf = Vec::new();
        let mut fds = Vec::new();
        {
            let mut b = PodBuilder::new(&mut buf, &mut fds);
            f(&mut b);
        }
        (buf, fds)
    }

    #[test]
    fn scalars_are_8_aligned_and_sized() {
        let (buf, _) = build(|b| {
            b.int(-7);
        });
        assert_eq!(buf.len(), 16); // 8 header + 4 value + 4 pad
        assert_eq!(PodReader::rd_u32(&buf, 0), Some(4)); // size = 4
        assert_eq!(PodReader::rd_u32(&buf, 4), Some(TYPE_INT));
        assert_eq!(PodReader::new(&buf).int(), Some(-7));

        let (lbuf, _) = build(|b| {
            b.long(0x0102_0304_0506_0708);
        });
        assert_eq!(lbuf.len(), 16); // 8 header + 8 value, no pad
        assert_eq!(PodReader::new(&lbuf).long(), Some(0x0102_0304_0506_0708));
    }

    #[test]
    fn string_round_trips_with_nul_and_padding() {
        let (buf, _) = build(|b| {
            b.string("abc");
        });
        // header(8) + "abc\0"(4) → size 4, total padded to 16.
        assert_eq!(PodReader::rd_u32(&buf, 0), Some(4));
        assert_eq!(buf.len() % 8, 0);
        assert_eq!(PodReader::new(&buf).string(), Some("abc"));

        // A 7-char string: "hello!!\0" = 8 body bytes, size 8, no extra pad.
        let (buf2, _) = build(|b| {
            b.string("hello!!");
        });
        assert_eq!(PodReader::new(&buf2).string(), Some("hello!!"));
    }

    #[test]
    fn struct_of_mixed_fields_reads_back_positionally() {
        let (buf, _) = build(|b| {
            b.push_struct();
            b.int(3).string("pipewire").id(42).long(-1);
            b.pop();
        });
        let mut top = PodReader::new(&buf);
        let mut s = top.enter_struct().unwrap();
        assert_eq!(s.int(), Some(3));
        assert_eq!(s.string(), Some("pipewire"));
        assert_eq!(s.id(), Some(42));
        assert_eq!(s.long(), Some(-1));
        assert_eq!(s.int(), None); // exhausted
    }

    #[test]
    fn dict_round_trips() {
        // A dict is a nested Struct; here it is the single field of an outer message struct.
        let (buf, _) = build(|b| {
            b.push_struct();
            b.dict(&[("a", "1"), ("b", "22")]);
            b.pop();
        });
        let mut top = PodReader::new(&buf);
        let mut s = top.enter_struct().unwrap();
        let mut got = Vec::new();
        s.dict(|k, v| got.push((k.to_owned(), v.to_owned()))).unwrap();
        assert_eq!(got, vec![("a".into(), "1".into()), ("b".into(), "22".into())]);
    }

    #[test]
    fn object_with_properties_reads_type_id_and_values() {
        // A tiny Format-shaped object: type=0x30000 id=3, two Id properties.
        let (buf, _) = build(|b| {
            b.push_object(0x0003_0000, 3);
            b.property(1, 0).id(10);
            b.property(2, 0).id(20);
            b.pop();
        });
        let mut top = PodReader::new(&buf);
        let (ty, id, mut obj) = top.enter_object().unwrap();
        assert_eq!((ty, id), (0x0003_0000, 3));
        let (k1, _f1, mut v1) = obj.next_prop().unwrap();
        assert_eq!((k1, v1.id()), (1, Some(10)));
        let (k2, _f2, mut v2) = obj.next_prop().unwrap();
        assert_eq!((k2, v2.id()), (2, Some(20)));
        assert!(obj.next_prop().is_none());
    }

    #[test]
    fn array_and_choice_of_ids() {
        let (abuf, _) = build(|b| {
            b.array_id(&[1, 2, 3]);
        });
        let mut r = PodReader::new(&abuf);
        let p = r.next_pod().unwrap();
        assert_eq!(p.ty, TYPE_ARRAY);
        assert_eq!(PodReader::rd_u32(p.body, 0), Some(4)); // child_size
        assert_eq!(PodReader::rd_u32(p.body, 4), Some(TYPE_ID)); // child_type
        assert_eq!(PodReader::rd_u32(p.body, 8), Some(1));

        let (cbuf, _) = build(|b| {
            b.choice_id(3, &[7, 8, 9]); // Enum choice, default 7
        });
        let mut rc = PodReader::new(&cbuf);
        let pc = rc.next_pod().unwrap();
        assert_eq!(pc.ty, TYPE_CHOICE);
        assert_eq!(PodReader::rd_u32(pc.body, 0), Some(3)); // choice_type = Enum
        assert_eq!(PodReader::rd_u32(pc.body, 8), Some(4)); // child_size
        assert_eq!(PodReader::rd_u32(pc.body, 12), Some(TYPE_ID)); // child_type
    }

    #[test]
    fn fd_arg_rides_side_channel_and_stores_index() {
        let (buf, fds) = build(|b| {
            b.push_struct();
            b.int(1).fd(77).fd(88);
            b.pop();
        });
        assert_eq!(fds, vec![77, 88]);
        let mut top = PodReader::new(&buf);
        let mut s = top.enter_struct().unwrap();
        assert_eq!(s.int(), Some(1));
        // Each Fd POD stores its *index* (0, 1) as an i64, not the raw fd.
        let fd0 = s.next_pod().unwrap();
        assert_eq!(fd0.ty, TYPE_FD);
        assert_eq!(i64::from_ne_bytes(fd0.body[..8].try_into().unwrap()), 0);
        let fd1 = s.next_pod().unwrap();
        assert_eq!(i64::from_ne_bytes(fd1.body[..8].try_into().unwrap()), 1);
    }
}
