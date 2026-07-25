//! The pinned introspection wire format (spec: Introspection protocol and
//! scraft-scope). PUBLIC API under the `introspect` feature: the scope/ GUI client is
//! written against these exact bytes, so the layout here is a contract — every offset,
//! row size, and tag is fixed and pinned by golden tests in `core/tests/introspect.rs`.
//!
//! ## Conventions
//! - All integers little-endian; core is `#![deny(unsafe_code)]`, so every field is
//!   encoded/decoded by hand with `to_le_bytes`/`from_le_bytes` — no transmutes.
//! - Rows are naturally aligned and padded to a multiple of 8; a reply that carries a
//!   table prefixes it with a [`TableHeader`] whose `row_size` is the server's stride,
//!   so a client walks rows by that stride and stays forward-compatible when a minor
//!   version appends fields (additive-only; major = breaking, never intended).
//! - Max frame payload is [`MAX_FRAME_LEN`] (16 MiB); a longer frame is a protocol
//!   error and disconnects.
//!
//! ## Versioning
//! `ver_major` = 1: a bump is breaking and is not intended to happen. `ver_minor`
//! grows additively (new frame kinds; new fields appended to a row, lengthening
//! `row_size`). Clients gate on `ver_major` and walk tables by the server's stride.

use std::io::{self, Read};

/// Wire protocol major version (breaking; never intended to change).
pub const VER_MAJOR: u16 = 1;
/// Wire protocol minor version (additive kinds + appended row fields).
pub const VER_MINOR: u16 = 1;

/// The 4-byte magic in the `Hello` frame — "SCIP" (StreamCraft Introspection Protocol).
pub const MAGIC: [u8; 4] = *b"SCIP";

/// Frame header size in bytes (`len`, `kind`, `seq`).
pub const HEADER_LEN: usize = 8;

/// Maximum frame *payload* length (excludes the header). 16 MiB — a `Dot` dump or a
/// huge `Props`/`Topology` reply fits; anything beyond is a bug or an attack, so the
/// server errors and disconnects.
pub const MAX_FRAME_LEN: u32 = 16 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Frame kinds (spec: Frame catalog v1)
// ---------------------------------------------------------------------------

/// The `kind` field of a frame header. Direction in the doc comment: S→C server to
/// client, C→S client to server. Values are pinned; new kinds are appended (minor bump).
pub mod kind {
    pub const HELLO: u16 = 0x0001; // S→C
    pub const CLIENT_HELLO: u16 = 0x0002; // C→S
    pub const ERROR: u16 = 0x0003; // S→C
    pub const PING: u16 = 0x0004; // C→S (empty)
    pub const PONG: u16 = 0x0005; // S→C (empty)

    pub const GET_TOPOLOGY: u16 = 0x0010; // C→S
    pub const TOPOLOGY: u16 = 0x0011; // S→C
    pub const GET_DOT: u16 = 0x0012; // C→S
    pub const DOT: u16 = 0x0013; // S→C
    pub const GET_COUNTERS: u16 = 0x0014; // C→S
    pub const COUNTERS: u16 = 0x0015; // S→C
    pub const GET_LATENCY: u16 = 0x0016; // C→S
    pub const LATENCY: u16 = 0x0017; // S→C
    pub const GET_LATENCY_REPORT: u16 = 0x0018; // C→S
    pub const LATENCY_REPORT: u16 = 0x0019; // S→C
    pub const GET_PROPS: u16 = 0x001A; // C→S
    pub const PROPS: u16 = 0x001B; // S→C
    pub const SET_PROP: u16 = 0x001C; // C→S
    pub const ACK: u16 = 0x001D; // S→C (empty)
    pub const PAUSE: u16 = 0x001E; // C→S
    pub const RESUME: u16 = 0x001F; // C→S
    pub const STEP: u16 = 0x0020; // C→S (RESERVED → Error(Unsupported))
    pub const SET_LOG_LEVEL: u16 = 0x0021; // C→S
    pub const SET_TRACING: u16 = 0x0022; // C→S
    pub const SUBSCRIBE: u16 = 0x0023; // C→S
    pub const UNSUBSCRIBE: u16 = 0x0024; // C→S
    pub const GET_INFO: u16 = 0x0025; // C→S (v1.1, empty)
    pub const INFO: u16 = 0x0026; // S→C (v1.1)

    pub const STR_DEF: u16 = 0x0030; // S→C
    pub const BUS_MSG: u16 = 0x0031; // S→C (pushed)
    pub const LOG_REC: u16 = 0x0032; // S→C (pushed)
    pub const DROPPED: u16 = 0x0033; // S→C (pushed)
    pub const BYE: u16 = 0x0034; // S→C (empty, at shutdown)
}

/// Error codes carried in an [`Error`](kind::ERROR) frame. Pinned.
pub mod errcode {
    pub const UNKNOWN_KIND: u32 = 1;
    pub const BAD_FRAME: u32 = 2;
    pub const UNKNOWN_ELEMENT: u32 = 3;
    pub const UNKNOWN_PROP: u32 = 4;
    pub const REJECTED: u32 = 5;
    pub const UNSUPPORTED: u32 = 6;
    pub const VERSION_MISMATCH: u32 = 7;
}

/// The `stream` id in a [`Dropped`](kind::DROPPED) frame.
pub mod stream {
    pub const BUS: u32 = 0;
    pub const LOGS: u32 = 1;
}

/// Subscribe/Unsubscribe mask bits.
pub mod submask {
    pub const BUS: u32 = 1 << 0;
    pub const LOGS: u32 = 1 << 1;
}

/// Hello `flags` bits.
pub mod helloflags {
    /// Latency tracing was on when the server started (histograms are populated).
    pub const TRACING_ON: u32 = 1 << 0;
    /// Log channels were wired this run, so `Subscribe(logs)` will deliver records.
    pub const LOG_CHANNELS: u32 = 1 << 1;
}

// ---------------------------------------------------------------------------
// Frame header
// ---------------------------------------------------------------------------

/// A frame header: 8 bytes. `len` counts payload bytes only (excludes the header).
/// `seq` echoes a client request id on replies (0 on server-pushed frames).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub len: u32,
    pub kind: u16,
    pub seq: u16,
}

impl FrameHeader {
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(&self.len.to_le_bytes());
        b[4..6].copy_from_slice(&self.kind.to_le_bytes());
        b[6..8].copy_from_slice(&self.seq.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8; HEADER_LEN]) -> FrameHeader {
        FrameHeader {
            len: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            kind: u16::from_le_bytes([b[4], b[5]]),
            seq: u16::from_le_bytes([b[6], b[7]]),
        }
    }
}

/// A decoded frame: its header plus the payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub kind: u16,
    pub seq: u16,
    pub payload: Vec<u8>,
}

/// Read exactly one frame off a blocking reader (spec: server per-client loop). Reads
/// the 8-byte header, then `len` payload bytes. Errors with `InvalidData` if `len`
/// exceeds [`MAX_FRAME_LEN`] (the disconnect trigger). The client GUI uses this.
pub fn read_frame(r: &mut impl Read) -> io::Result<Frame> {
    let mut hb = [0u8; HEADER_LEN];
    r.read_exact(&mut hb)?;
    let h = FrameHeader::decode(&hb);
    if h.len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "introspect frame exceeds MAX_FRAME_LEN",
        ));
    }
    let mut payload = vec![0u8; h.len as usize];
    r.read_exact(&mut payload)?;
    Ok(Frame { kind: h.kind, seq: h.seq, payload })
}

/// Encode a full frame (header + payload) into a fresh `Vec`. The server writes these
/// into its `BufWriter`; a client can build request frames the same way.
pub fn encode_frame(kind: u16, seq: u16, payload: &[u8]) -> Vec<u8> {
    let h = FrameHeader { len: payload.len() as u32, kind, seq };
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&h.encode());
    out.extend_from_slice(payload);
    out
}

// ---------------------------------------------------------------------------
// A tiny cursor-free little-endian writer, so encoders read at "documented offsets"
// by construction (append in order). No dependency; std Vec only.
// ---------------------------------------------------------------------------

/// A little-endian byte builder. Every `put_*` appends; the resulting layout is the
/// wire layout, so "write at offset N" is enforced by call order plus the fixed sizes.
#[derive(Default)]
pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
    pub fn with_capacity(n: usize) -> Self {
        Self { buf: Vec::with_capacity(n) }
    }
    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn put_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// `n` zero bytes (padding to a natural boundary).
    pub fn pad(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }
    pub fn put_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// A little-endian byte reader over a payload slice. Mirrors [`Writer`]; the client
/// walks a reply with it. Returns `None` on underrun rather than panicking, so a
/// truncated/hostile frame can't crash a decoder.
pub struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.pos)
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.remaining() < n {
            return None;
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }
    pub fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }
    pub fn get_u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }
    pub fn get_u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }
    pub fn get_u32(&mut self) -> Option<u32> {
        self.take(4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    pub fn get_u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    pub fn get_i64(&mut self) -> Option<i64> {
        self.get_u64().map(|v| v as i64)
    }
    pub fn get_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        self.take(n)
    }
}

// ---------------------------------------------------------------------------
// WireValue — a POD encoding of `format::Value` for the wire (spec: shared primitives)
// ---------------------------------------------------------------------------

/// A `Value` on the wire: 16 bytes. `tag` selects the interpretation of `bits`.
/// These tags are the WIRE tags, independent of `props.rs`'s private packing tags.
///
/// - `0 = Unset` — only meaningful in a "current value" slot (a prop never set).
/// - `1 = Int` — `bits` is the two's-complement `i64`.
/// - `2 = Rat` — `bits = (num as u32 as u64) << 32 | (den as u32 as u64)`.
/// - `3 = Id` — `bits` is the CONNECTION string id (not the pipeline `ValueId`); the
///   server records the mapping so a `SetProp` with an `Id` value maps back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireValue {
    pub tag: u32,
    pub bits: u64,
}

impl WireValue {
    pub const TAG_UNSET: u32 = 0;
    pub const TAG_INT: u32 = 1;
    pub const TAG_RAT: u32 = 2;
    pub const TAG_ID: u32 = 3;

    /// Encoded size in bytes.
    pub const SIZE: usize = 16;

    pub const UNSET: WireValue = WireValue { tag: Self::TAG_UNSET, bits: 0 };

    pub fn int(v: i64) -> WireValue {
        WireValue { tag: Self::TAG_INT, bits: v as u64 }
    }
    pub fn rat(num: i32, den: i32) -> WireValue {
        WireValue {
            tag: Self::TAG_RAT,
            bits: ((num as u32 as u64) << 32) | (den as u32 as u64),
        }
    }
    /// An `Id` whose `bits` is the **connection** string id.
    pub fn id(conn_str_id: u32) -> WireValue {
        WireValue { tag: Self::TAG_ID, bits: conn_str_id as u64 }
    }

    /// The rational parts, if `tag == Rat`.
    pub fn as_rat(&self) -> Option<(i32, i32)> {
        if self.tag == Self::TAG_RAT {
            Some(((self.bits >> 32) as u32 as i32, self.bits as u32 as i32))
        } else {
            None
        }
    }

    pub fn write(&self, w: &mut Writer) {
        w.put_u32(self.tag);
        w.put_u32(0); // _pad
        w.put_u64(self.bits);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<WireValue> {
        let tag = r.get_u32()?;
        r.skip(4)?; // _pad
        let bits = r.get_u64()?;
        Some(WireValue { tag, bits })
    }
}

// ---------------------------------------------------------------------------
// TableHeader — the 8-byte prefix of a table reply (spec: shared primitives)
// ---------------------------------------------------------------------------

/// The 8-byte prefix of a table reply: `row_size` (the server's stride), `row_count`,
/// then 4 pad bytes. A client walks `row_count` rows by `row_size`, ignoring any bytes
/// past the fields it knows (forward compatibility).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableHeader {
    pub row_size: u16,
    pub row_count: u16,
}

impl TableHeader {
    pub const SIZE: usize = 8;

    pub fn write(&self, w: &mut Writer) {
        w.put_u16(self.row_size);
        w.put_u16(self.row_count);
        w.put_u32(0); // _pad
    }

    pub fn read(r: &mut Reader<'_>) -> Option<TableHeader> {
        let row_size = r.get_u16()?;
        let row_count = r.get_u16()?;
        r.skip(4)?; // _pad
        Some(TableHeader { row_size, row_count })
    }
}

// ---------------------------------------------------------------------------
// Fixed row sizes (spec: Rows). Pinned by tests.
// ---------------------------------------------------------------------------

pub const ROW_SIZE_ELEMENT: u16 = 40;
pub const ROW_SIZE_PAD: u16 = 16;
pub const ROW_SIZE_EDGE: u16 = 280;
pub const ROW_SIZE_COUNTER: u16 = 72;
pub const ROW_SIZE_LATENCY: u16 = 848;

/// The `Timestamp::NONE` sentinel, carried on the wire as-is (u64::MAX).
pub const TS_NONE: u64 = u64::MAX;

// --- Hello / ClientHello / Error / Ping-Pong -------------------------------

/// Server → client handshake (spec). `nelements` lets a client pre-size its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hello {
    pub ver_major: u16,
    pub ver_minor: u16,
    pub flags: u32,
    pub pid: u32,
    pub nelements: u32,
}

impl Hello {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.put_bytes(&MAGIC);
        w.put_u16(self.ver_major);
        w.put_u16(self.ver_minor);
        w.put_u32(self.flags);
        w.put_u32(self.pid);
        w.put_u32(self.nelements);
        w.into_vec()
    }

    pub fn decode(payload: &[u8]) -> Option<Hello> {
        let mut r = Reader::new(payload);
        let magic = r.get_bytes(4)?;
        if magic != MAGIC {
            return None;
        }
        Some(Hello {
            ver_major: r.get_u16()?,
            ver_minor: r.get_u16()?,
            flags: r.get_u32()?,
            pid: r.get_u32()?,
            nelements: r.get_u32()?,
        })
    }
}

/// Client → server: the first frame the client must send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientHello {
    pub ver_major: u16,
    pub ver_minor: u16,
}

impl ClientHello {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.put_u16(self.ver_major);
        w.put_u16(self.ver_minor);
        w.put_u32(0); // _pad
        w.into_vec()
    }
    pub fn decode(payload: &[u8]) -> Option<ClientHello> {
        let mut r = Reader::new(payload);
        Some(ClientHello { ver_major: r.get_u16()?, ver_minor: r.get_u16()? })
    }
}

/// An error reply (spec). `code` is one of [`errcode`]; `msg` is human text.
pub fn encode_error(code: u32, msg: &str) -> Vec<u8> {
    let mut w = Writer::new();
    let bytes = msg.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    w.put_u32(code);
    w.put_u16(len as u16);
    w.put_u16(0); // _pad
    w.put_bytes(&bytes[..len]);
    w.into_vec()
}

/// Decode an `Error` payload into `(code, message)`.
pub fn decode_error(payload: &[u8]) -> Option<(u32, String)> {
    let mut r = Reader::new(payload);
    let code = r.get_u32()?;
    let len = r.get_u16()? as usize;
    r.skip(2)?; // _pad
    let bytes = r.get_bytes(len)?;
    Some((code, String::from_utf8_lossy(bytes).into_owned()))
}

// --- StrDef (spec: Strings) ------------------------------------------------

/// A string-table definition: `id`, `len`, pad, then `len` UTF-8 bytes. Sent before
/// the first reference to `id`. Ids are dense from 1; 0 means "none".
pub fn encode_str_def(id: u32, s: &str) -> Vec<u8> {
    let mut w = Writer::new();
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    w.put_u32(id);
    w.put_u16(len as u16);
    w.put_u16(0); // _pad
    w.put_bytes(&bytes[..len]);
    w.into_vec()
}

/// Decode a `StrDef` payload into `(id, string)`.
pub fn decode_str_def(payload: &[u8]) -> Option<(u32, String)> {
    let mut r = Reader::new(payload);
    let id = r.get_u32()?;
    let len = r.get_u16()? as usize;
    r.skip(2)?; // _pad
    let bytes = r.get_bytes(len)?;
    Some((id, String::from_utf8_lossy(bytes).into_owned()))
}

// --- ElementRow (40 B) -----------------------------------------------------

/// Per-element topology row. `sched`: 0 passive, 1 active. `flags`: bit0 source,
/// bit1 sink, bit2 live. Latency fields carry [`TS_NONE`] as-is when unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementRow {
    pub element: u32,
    pub name_str: u32,
    pub group: u32,
    pub sched: u8,
    pub flags: u8,
    pub npads: u16,
    pub latency_min_ns: u64,
    pub latency_max_ns: u64,
    pub jitter_ns: u64,
}

impl ElementRow {
    pub const FLAG_SOURCE: u8 = 1 << 0;
    pub const FLAG_SINK: u8 = 1 << 1;
    pub const FLAG_LIVE: u8 = 1 << 2;

    pub fn write(&self, w: &mut Writer) {
        let start = w.len();
        w.put_u32(self.element);
        w.put_u32(self.name_str);
        w.put_u32(self.group);
        w.put_u8(self.sched);
        w.put_u8(self.flags);
        w.put_u16(self.npads);
        w.put_u64(self.latency_min_ns);
        w.put_u64(self.latency_max_ns);
        w.put_u64(self.jitter_ns);
        debug_assert_eq!(w.len() - start, ROW_SIZE_ELEMENT as usize);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<ElementRow> {
        let element = r.get_u32()?;
        let name_str = r.get_u32()?;
        let group = r.get_u32()?;
        let sched = r.get_u8()?;
        let flags = r.get_u8()?;
        let npads = r.get_u16()?;
        let latency_min_ns = r.get_u64()?;
        let latency_max_ns = r.get_u64()?;
        let jitter_ns = r.get_u64()?;
        Some(ElementRow {
            element,
            name_str,
            group,
            sched,
            flags,
            npads,
            latency_min_ns,
            latency_max_ns,
            jitter_ns,
        })
    }
}

// --- PadRow (16 B) ---------------------------------------------------------

/// Per-pad row. `direction`: 0 sink, 1 src (matches [`Direction`] ordinal).
/// `flags`: bit0 dynamic, bit1 linked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadRow {
    pub element: u32,
    pub pad: u32,
    pub name_str: u32,
    pub direction: u8,
    pub flags: u8,
}

impl PadRow {
    pub const DIR_SINK: u8 = 0;
    pub const DIR_SRC: u8 = 1;
    pub const FLAG_DYNAMIC: u8 = 1 << 0;
    pub const FLAG_LINKED: u8 = 1 << 1;

    pub fn write(&self, w: &mut Writer) {
        let start = w.len();
        w.put_u32(self.element);
        w.put_u32(self.pad);
        w.put_u32(self.name_str);
        w.put_u8(self.direction);
        w.put_u8(self.flags);
        w.put_u16(0); // _pad
        debug_assert_eq!(w.len() - start, ROW_SIZE_PAD as usize);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<PadRow> {
        let element = r.get_u32()?;
        let pad = r.get_u32()?;
        let name_str = r.get_u32()?;
        let direction = r.get_u8()?;
        let flags = r.get_u8()?;
        r.skip(2)?; // _pad
        Some(PadRow { element, pad, name_str, direction, flags })
    }
}

// --- EdgeRow (280 B) -------------------------------------------------------

/// One fixed `(field, value)` slot in an [`EdgeRow`]: 16 bytes. `tag` matches the
/// [`WireValue`] tags (1 Int, 2 Rat, 3 Id). Slots past `nfields` are zeroed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldSlot {
    pub field_str: u32,
    pub tag: u8,
    pub bits: u64,
}

impl FieldSlot {
    pub const SIZE: usize = 16;
    pub const ZERO: FieldSlot = FieldSlot { field_str: 0, tag: 0, bits: 0 };

    pub fn write(&self, w: &mut Writer) {
        w.put_u32(self.field_str);
        w.put_u8(self.tag);
        w.pad(3); // _pad[3]
        w.put_u64(self.bits);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<FieldSlot> {
        let field_str = r.get_u32()?;
        let tag = r.get_u8()?;
        r.skip(3)?;
        let bits = r.get_u64()?;
        Some(FieldSlot { field_str, tag, bits })
    }
}

/// The number of inline field slots on an [`EdgeRow`] (matches `FixedFormat`'s cap).
pub const EDGE_MAX_FIELDS: usize = 16;

/// A negotiated-edge row. Serializes a `FixedFormat`: the `family_str` and up to 16
/// `(field_str, tag, bits)` slots; slots `>= nfields` are zeroed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeRow {
    pub src: u32,
    pub src_pad: u32,
    pub sink: u32,
    pub sink_pad: u32,
    pub family_str: u32,
    pub nfields: u8,
    pub fields: [FieldSlot; EDGE_MAX_FIELDS],
}

impl EdgeRow {
    pub fn write(&self, w: &mut Writer) {
        let start = w.len();
        w.put_u32(self.src);
        w.put_u32(self.src_pad);
        w.put_u32(self.sink);
        w.put_u32(self.sink_pad);
        w.put_u32(self.family_str);
        w.put_u8(self.nfields);
        w.pad(3); // _pad[3]
        for slot in &self.fields {
            slot.write(w);
        }
        debug_assert_eq!(w.len() - start, ROW_SIZE_EDGE as usize);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<EdgeRow> {
        let src = r.get_u32()?;
        let src_pad = r.get_u32()?;
        let sink = r.get_u32()?;
        let sink_pad = r.get_u32()?;
        let family_str = r.get_u32()?;
        let nfields = r.get_u8()?;
        r.skip(3)?;
        let mut fields = [FieldSlot::ZERO; EDGE_MAX_FIELDS];
        for slot in fields.iter_mut() {
            *slot = FieldSlot::read(r)?;
        }
        Some(EdgeRow { src, src_pad, sink, sink_pad, family_str, nfields, fields })
    }
}

// --- CounterRow (72 B) -----------------------------------------------------

/// A direct image of a [`CounterSnapshot`](crate::counters::CounterSnapshot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CounterRow {
    pub element: u32,
    pub buffers_in: u64,
    pub buffers_out: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub batches_in: u64,
    pub batches_out: u64,
    pub queue_high_water: u32,
    pub drops: u64,
}

impl CounterRow {
    pub fn write(&self, w: &mut Writer) {
        let start = w.len();
        w.put_u32(self.element);
        w.put_u32(0); // _pad
        w.put_u64(self.buffers_in);
        w.put_u64(self.buffers_out);
        w.put_u64(self.bytes_in);
        w.put_u64(self.bytes_out);
        w.put_u64(self.batches_in);
        w.put_u64(self.batches_out);
        w.put_u32(self.queue_high_water);
        w.put_u32(0); // _pad
        w.put_u64(self.drops);
        debug_assert_eq!(w.len() - start, ROW_SIZE_COUNTER as usize);
    }

    pub fn read(r: &mut Reader<'_>) -> Option<CounterRow> {
        let element = r.get_u32()?;
        r.skip(4)?;
        let buffers_in = r.get_u64()?;
        let buffers_out = r.get_u64()?;
        let bytes_in = r.get_u64()?;
        let bytes_out = r.get_u64()?;
        let batches_in = r.get_u64()?;
        let batches_out = r.get_u64()?;
        let queue_high_water = r.get_u32()?;
        r.skip(4)?;
        let drops = r.get_u64()?;
        Some(CounterRow {
            element,
            buffers_in,
            buffers_out,
            bytes_in,
            bytes_out,
            batches_in,
            batches_out,
            queue_high_water,
            drops,
        })
    }
}

// --- LatencyRow (848 B) ----------------------------------------------------

/// The bucket count in a latency histogram (mirrors
/// [`LatencyHistogram::BUCKETS`](crate::counters::LatencyHistogram::BUCKETS)).
pub const LATENCY_BUCKETS: usize = 32;

/// One histogram on the wire: 8 + 8 + 8 + 32*8 = 280 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireHistogram {
    pub count: u64,
    pub sum_ns: u64,
    pub max_ns: u64,
    pub buckets: [u64; LATENCY_BUCKETS],
}

impl WireHistogram {
    pub const SIZE: usize = 8 + 8 + 8 + LATENCY_BUCKETS * 8;
    pub const ZERO: WireHistogram =
        WireHistogram { count: 0, sum_ns: 0, max_ns: 0, buckets: [0; LATENCY_BUCKETS] };

    pub fn write(&self, w: &mut Writer) {
        w.put_u64(self.count);
        w.put_u64(self.sum_ns);
        w.put_u64(self.max_ns);
        for &b in &self.buckets {
            w.put_u64(b);
        }
    }
    pub fn read(r: &mut Reader<'_>) -> Option<WireHistogram> {
        let count = r.get_u64()?;
        let sum_ns = r.get_u64()?;
        let max_ns = r.get_u64()?;
        let mut buckets = [0u64; LATENCY_BUCKETS];
        for b in buckets.iter_mut() {
            *b = r.get_u64()?;
        }
        Some(WireHistogram { count, sum_ns, max_ns, buckets })
    }
}

/// A per-element latency row: `element`, pad, then three histograms
/// (process, queue, wait_lateness). All-zero unless tracing was on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyRow {
    pub element: u32,
    pub process: WireHistogram,
    pub queue: WireHistogram,
    pub wait_lateness: WireHistogram,
}

impl LatencyRow {
    pub fn write(&self, w: &mut Writer) {
        let start = w.len();
        w.put_u32(self.element);
        w.put_u32(0); // _pad
        self.process.write(w);
        self.queue.write(w);
        self.wait_lateness.write(w);
        debug_assert_eq!(w.len() - start, ROW_SIZE_LATENCY as usize);
    }
    pub fn read(r: &mut Reader<'_>) -> Option<LatencyRow> {
        let element = r.get_u32()?;
        r.skip(4)?;
        let process = WireHistogram::read(r)?;
        let queue = WireHistogram::read(r)?;
        let wait_lateness = WireHistogram::read(r)?;
        Some(LatencyRow { element, process, queue, wait_lateness })
    }
}

// --- LatencyReport ---------------------------------------------------------

/// One element's contribution on a latency path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyPathElem {
    pub element: u32,
    pub min_ns: u64,
}

/// A single source→sink path in a latency report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatencyPath {
    pub sink: u32,
    pub is_live: u8,
    pub total_ns: u64,
    pub elems: Vec<LatencyPathElem>,
}

/// Encode a full latency report payload (spec: LatencyReport).
pub fn encode_latency_report(paths: &[LatencyPath]) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u32(paths.len() as u32);
    w.put_u32(0); // _pad
    for p in paths {
        w.put_u32(p.sink);
        w.put_u8(p.is_live);
        w.put_u8(0); // _pad
        w.put_u16(p.elems.len() as u16);
        w.put_u64(p.total_ns);
        for e in &p.elems {
            w.put_u32(e.element);
            w.put_u32(0); // _pad
            w.put_u64(e.min_ns);
        }
    }
    w.into_vec()
}

/// Decode a latency report payload.
pub fn decode_latency_report(payload: &[u8]) -> Option<Vec<LatencyPath>> {
    let mut r = Reader::new(payload);
    let n = r.get_u32()? as usize;
    r.skip(4)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let sink = r.get_u32()?;
        let is_live = r.get_u8()?;
        r.skip(1)?;
        let nelems = r.get_u16()? as usize;
        let total_ns = r.get_u64()?;
        let mut elems = Vec::with_capacity(nelems);
        for _ in 0..nelems {
            let element = r.get_u32()?;
            r.skip(4)?;
            let min_ns = r.get_u64()?;
            elems.push(LatencyPathElem { element, min_ns });
        }
        out.push(LatencyPath { sink, is_live, total_ns, elems });
    }
    Some(out)
}

// --- PropRow (variable, self-sizing) ---------------------------------------

/// A property row (spec). `ckind`: 0 Any, 1 Eq, 2 Range, 3 Set. `nvals` value slots
/// follow the header (Eq⇒1, Range⇒3 min/max/step, Set⇒len; Any⇒0). `current`'s tag is
/// `Unset` when the property was never set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropRow {
    pub element: u32,
    pub prop_index: u16,
    pub live: u8,
    pub ckind: u8,
    pub name_str: u32,
    pub current: WireValue,
    pub vals: Vec<WireValue>,
}

impl PropRow {
    pub const CKIND_ANY: u8 = 0;
    pub const CKIND_EQ: u8 = 1;
    pub const CKIND_RANGE: u8 = 2;
    pub const CKIND_SET: u8 = 3;

    pub fn write(&self, w: &mut Writer) {
        w.put_u32(self.element);
        w.put_u16(self.prop_index);
        w.put_u8(self.live);
        w.put_u8(self.ckind);
        w.put_u32(self.name_str);
        self.current.write(w);
        w.put_u16(self.vals.len() as u16);
        w.pad(6); // _pad[6]
        for v in &self.vals {
            v.write(w);
        }
    }

    pub fn read(r: &mut Reader<'_>) -> Option<PropRow> {
        let element = r.get_u32()?;
        let prop_index = r.get_u16()?;
        let live = r.get_u8()?;
        let ckind = r.get_u8()?;
        let name_str = r.get_u32()?;
        let current = WireValue::read(r)?;
        let nvals = r.get_u16()? as usize;
        r.skip(6)?;
        let mut vals = Vec::with_capacity(nvals);
        for _ in 0..nvals {
            vals.push(WireValue::read(r)?);
        }
        Some(PropRow { element, prop_index, live, ckind, name_str, current, vals })
    }
}

/// Encode a `Props` reply payload: `n_rows u32, _pad u32`, then the self-sizing rows.
pub fn encode_props(rows: &[PropRow]) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u32(rows.len() as u32);
    w.put_u32(0); // _pad
    for row in rows {
        row.write(&mut w);
    }
    w.into_vec()
}

/// Decode a `Props` reply payload.
pub fn decode_props(payload: &[u8]) -> Option<Vec<PropRow>> {
    let mut r = Reader::new(payload);
    let n = r.get_u32()? as usize;
    r.skip(4)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(PropRow::read(&mut r)?);
    }
    Some(out)
}

// --- SetProp request -------------------------------------------------------

/// A `SetProp` request payload (spec): `element u32, prop_index u16, _pad u16, value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetPropReq {
    pub element: u32,
    pub prop_index: u16,
    pub value: WireValue,
}

impl SetPropReq {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.put_u32(self.element);
        w.put_u16(self.prop_index);
        w.put_u16(0); // _pad
        self.value.write(&mut w);
        w.into_vec()
    }
    pub fn decode(payload: &[u8]) -> Option<SetPropReq> {
        let mut r = Reader::new(payload);
        let element = r.get_u32()?;
        let prop_index = r.get_u16()?;
        r.skip(2)?;
        let value = WireValue::read(&mut r)?;
        Some(SetPropReq { element, prop_index, value })
    }
}

// --- Small control requests ------------------------------------------------

/// A `GetLatency` / `GetProps` request: a single element id (u32::MAX = all).
pub const ELEM_ALL: u32 = u32::MAX;

pub fn encode_elem_req(element: u32) -> Vec<u8> {
    element.to_le_bytes().to_vec()
}
pub fn decode_elem_req(payload: &[u8]) -> Option<u32> {
    Reader::new(payload).get_u32()
}

/// A `SetLogLevel` request: `element u32 (MAX=global), rank u8 (0..5), _pad[3]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetLogLevelReq {
    pub element: u32,
    pub rank: u8,
}

impl SetLogLevelReq {
    pub const GLOBAL: u32 = u32::MAX;
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.put_u32(self.element);
        w.put_u8(self.rank);
        w.pad(3);
        w.into_vec()
    }
    pub fn decode(payload: &[u8]) -> Option<SetLogLevelReq> {
        let mut r = Reader::new(payload);
        let element = r.get_u32()?;
        let rank = r.get_u8()?;
        Some(SetLogLevelReq { element, rank })
    }
}

/// A `SetTracing` request: `on u8, _pad[3]`.
pub fn encode_set_tracing(on: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u8(on as u8);
    w.pad(3);
    w.into_vec()
}
pub fn decode_set_tracing(payload: &[u8]) -> Option<bool> {
    Reader::new(payload).get_u8().map(|v| v != 0)
}

/// A `Subscribe`/`Unsubscribe` request: `mask u32` (see [`submask`]).
pub fn encode_sub(mask: u32) -> Vec<u8> {
    mask.to_le_bytes().to_vec()
}
pub fn decode_sub(payload: &[u8]) -> Option<u32> {
    Reader::new(payload).get_u32()
}

// --- Counters reply prefix -------------------------------------------------

/// The `Counters` reply prefix: `now_ns u64`, then a [`TableHeader`] + rows. `now_ns`
/// is [`TS_NONE`] before the first run (mirrors `TapHandle::now()`).
pub fn write_counters_prefix(w: &mut Writer, now_ns: u64) {
    w.put_u64(now_ns);
}

// --- Dropped (pushed) ------------------------------------------------------

/// A `Dropped` push: `stream u32 (0 bus, 1 logs), _pad u32, count u64`.
pub fn encode_dropped(stream_id: u32, count: u64) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u32(stream_id);
    w.put_u32(0); // _pad
    w.put_u64(count);
    w.into_vec()
}
pub fn decode_dropped(payload: &[u8]) -> Option<(u32, u64)> {
    let mut r = Reader::new(payload);
    let stream_id = r.get_u32()?;
    r.skip(4)?;
    let count = r.get_u64()?;
    Some((stream_id, count))
}

// --- Info (v1.1) -------------------------------------------------------------

/// The `Info` reply (v1.1): sticky pipeline facts a late-attaching client would
/// otherwise have missed on the bus. Currently: `duration_ns u64` ([`TS_NONE`] =
/// unknown). Additive-versioned like table rows: new fields append, old clients
/// read the prefix they know, new clients treat a short payload as "absent".
pub fn encode_info(duration_ns: Option<u64>) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u64(duration_ns.unwrap_or(TS_NONE));
    w.into_vec()
}
pub fn decode_info(payload: &[u8]) -> Option<Option<u64>> {
    let mut r = Reader::new(payload);
    let ns = r.get_u64()?;
    Some(if ns == TS_NONE { None } else { Some(ns) })
}

// --- BusMsg (96 B) ---------------------------------------------------------
//
// The per-kind a/b/c/d mapping (spec: document the mapping in wire.rs). `kind` is the
// `BusMessage` variant ordinal in declaration order (bus.rs:26-41), pinned by a test:
//
//   0  Error          a=element                     msg=truncated error text
//   1  Warning        a=element                     msg=truncated error text
//   2  Eos            (none)
//   3  StateChanged   a=old (State ordinal) b=new
//   4  PadAdded       a=element  b=pad  c=family id   ← re-GetTopology cue
//   5  ElementAdded   a=element  b=group             ← re-GetTopology cue
//   6  ElementRemoved a=element                      ← re-GetTopology cue
//   7  LinkChanged    a=link                         ← re-GetTopology cue
//   8  Tags           a=element   (tag payload NOT carried — ids only)
//   9  SubgraphJoined a=group    c=added_latency ns
//   10 LatencyChanged c=old ns   d=new ns
//   11 Qos            a=sink     c=lateness_ns (i64 bits)
//   12 BranchSealed   a=group                        msg=truncated error text
//   13 DurationChanged a=element  c=duration ns       (v1.1: presentation duration known)
//
// `class`: 0 droppable, 1 critical (from `BusMessage::class`). The topology-mutation
// kinds (PadAdded/ElementAdded/ElementRemoved/LinkChanged) are the client's cue to
// re-issue GetTopology.

/// Length of the inline truncated message on a [`BusMsgRow`] (cut at a UTF-8 boundary).
pub const BUS_MSG_TEXT: usize = 64;

/// A `BusMsg` push row: 100 bytes.
///
/// NOTE: the pinned brief labels this "96 B", but its own field list —
/// `seq u64 (8), kind u8, class u8, msg_len u8, _pad u8 (4), a u32, b u32 (8),
/// c u64, d u64 (16), msg [u8;64] (64)` — sums to 100. The field list is the
/// authoritative layout a client decodes, so `SIZE` is 100; the "96" was an
/// arithmetic slip in the brief. Fields and their order are exactly as specified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusMsgRow {
    pub seq: u64,
    pub kind: u8,
    pub class: u8,
    pub msg_len: u8,
    pub a: u32,
    pub b: u32,
    pub c: u64,
    pub d: u64,
    pub msg: [u8; BUS_MSG_TEXT],
}

impl BusMsgRow {
    pub const SIZE: usize = 100;

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::with_capacity(Self::SIZE);
        w.put_u64(self.seq);
        w.put_u8(self.kind);
        w.put_u8(self.class);
        w.put_u8(self.msg_len);
        w.put_u8(0); // _pad
        w.put_u32(self.a);
        w.put_u32(self.b);
        w.put_u64(self.c);
        w.put_u64(self.d);
        w.put_bytes(&self.msg);
        debug_assert_eq!(w.len(), Self::SIZE);
        w.into_vec()
    }

    pub fn decode(payload: &[u8]) -> Option<BusMsgRow> {
        let mut r = Reader::new(payload);
        let seq = r.get_u64()?;
        let kind = r.get_u8()?;
        let class = r.get_u8()?;
        let msg_len = r.get_u8()?;
        r.skip(1)?;
        let a = r.get_u32()?;
        let b = r.get_u32()?;
        let c = r.get_u64()?;
        let d = r.get_u64()?;
        let mut msg = [0u8; BUS_MSG_TEXT];
        msg.copy_from_slice(r.get_bytes(BUS_MSG_TEXT)?);
        Some(BusMsgRow { seq, kind, class, msg_len, a, b, c, d, msg })
    }

    /// The inline message as a `&str` (up to `msg_len`, already a UTF-8 boundary).
    pub fn text(&self) -> &str {
        std::str::from_utf8(&self.msg[..self.msg_len as usize]).unwrap_or("")
    }
}

// --- LogRec (88 B) ---------------------------------------------------------

/// One structured log field on a [`LogRecRow`]: `key_str u32, tag u8, _pad[3], bits u64`.
/// `tag` mirrors [`FieldValue`](crate::log::FieldValue) (log.rs:218-225):
/// 0 Int, 1 Uint, 2 Bool, 3 Str (bits = string id), 4 Fourcc (low 4 bytes), 5 Time (ns).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogField {
    pub key_str: u32,
    pub tag: u8,
    pub bits: u64,
}

impl LogField {
    pub const SIZE: usize = 16;
    pub const ZERO: LogField = LogField { key_str: 0, tag: 0, bits: 0 };
    pub const TAG_INT: u8 = 0;
    pub const TAG_UINT: u8 = 1;
    pub const TAG_BOOL: u8 = 2;
    pub const TAG_STR: u8 = 3;
    pub const TAG_FOURCC: u8 = 4;
    pub const TAG_TIME: u8 = 5;

    pub fn write(&self, w: &mut Writer) {
        w.put_u32(self.key_str);
        w.put_u8(self.tag);
        w.pad(3);
        w.put_u64(self.bits);
    }
    pub fn read(r: &mut Reader<'_>) -> Option<LogField> {
        let key_str = r.get_u32()?;
        let tag = r.get_u8()?;
        r.skip(3)?;
        let bits = r.get_u64()?;
        Some(LogField { key_str, tag, bits })
    }
}

/// The inline field count on a [`LogRecRow`] (matches
/// [`log::MAX_FIELDS`](crate::log::MAX_FIELDS)).
pub const LOG_MAX_FIELDS: usize = 4;

/// A `LogRec` push row: 88 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogRecRow {
    pub ts: u64,
    pub element: u32,
    pub name_str: u32,
    pub event_str: u32,
    pub level: u8,
    pub nfields: u8,
    pub fields: [LogField; LOG_MAX_FIELDS],
}

impl LogRecRow {
    pub const SIZE: usize = 88;

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::with_capacity(Self::SIZE);
        w.put_u64(self.ts);
        w.put_u32(self.element);
        w.put_u32(self.name_str);
        w.put_u32(self.event_str);
        w.put_u8(self.level);
        w.put_u8(self.nfields);
        w.put_u16(0); // _pad
        for f in &self.fields {
            f.write(&mut w);
        }
        debug_assert_eq!(w.len(), Self::SIZE);
        w.into_vec()
    }

    pub fn decode(payload: &[u8]) -> Option<LogRecRow> {
        let mut r = Reader::new(payload);
        let ts = r.get_u64()?;
        let element = r.get_u32()?;
        let name_str = r.get_u32()?;
        let event_str = r.get_u32()?;
        let level = r.get_u8()?;
        let nfields = r.get_u8()?;
        r.skip(2)?;
        let mut fields = [LogField::ZERO; LOG_MAX_FIELDS];
        for f in fields.iter_mut() {
            *f = LogField::read(&mut r)?;
        }
        Some(LogRecRow { ts, element, name_str, event_str, level, nfields, fields })
    }
}
