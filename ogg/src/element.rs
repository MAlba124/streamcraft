//! `OggDemux` / `OggMux` — the streamcraft **elements** wrapping the tested
//! [`OggReader`](crate::OggReader) / [`OggWriter`](crate::OggWriter) library (spec:
//! Writing elements; RFC 3533). These are the **single logical bitstream** versions:
//! one sink pad and one src pad each, fitting today's static pad model (spec: Elements
//! and pads). The general multiplexer/demultiplexer wants one *dynamic* pad per logical
//! stream — deferred until the scheduler grows dynamic pad add/remove (see
//! `spec/NOTES.md`, "Not yet"). Everything here is a **passive transform**: bytes in,
//! bytes out, inlining into the upstream group like `flacdec` / `flacenc` (spec:
//! Scheduling — passive elements run inline, never block).
//!
//! Both pads carry raw [`bytes`](streamcraft_core::format::OfferDesc::any): an Ogg byte
//! stream on the container side, codec packets on the elementary side. There is no typed
//! format here — the container is codec-agnostic (§4: Ogg has "no concept of 'time'" and
//! no knowledge of the media it carries); a downstream depacketiser/decoder is what reads
//! the packet bytes.
//!
//! ## EOS flushing (spec: Events — Eos)
//! An Ogg stream is only well-formed when its **last page carries the eos flag** (§6, flag
//! 0x04). The muxer cannot know which packet is the last from `process()` alone, so it
//! flushes the terminating eos page at end-of-stream. It does this two ways, so whichever
//! the runtime provides wins:
//! 1. [`Element::event`] on [`Event::Eos`] — the pipeline delivers EOS in-band at
//!    end-of-stream and then pushes whatever output the handler produced before closing
//!    the ring. This is the forward-looking primary path.
//! 2. [`Element::stop`] — a belt-and-braces fallback: [`OggWriter::finish`] is idempotent,
//!    so flushing again here is a no-op if `event` already did it, and the safety net if
//!    it did not.
//!
//! Either way the final page is emitted exactly once, with the eos flag set.
//!
//! The muxer packs several packets per page (the efficient default): `process` flushes only
//! *full* pages, and the pending partial page is emitted by `finish_stream` on EOS, which
//! terminates the last real packet's page with the eos flag — so there is no trailing nil
//! packet. This relies on the pipeline delivering `Event::Eos` to `event()` in chain order
//! and pushing the handler's output before closing the ring, which it does (spec: Events —
//! EOS reaches elements). The `finish`-contract test in `tests/element_roundtrip.rs` covers
//! the eos-terminated bytes.

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::reader::OggReader;
use crate::writer::OggWriter;

/// The granule position stamped on every muxed packet. Ogg has "no concept of 'time'"
/// (§4); a real granule is a media-mapping detail this container-only element does not
/// synthesise. The −1 sentinel ([`GRANULE_NONE`](crate::page::GRANULE_NONE)) means "no
/// packet finishes here", which is *wrong* for a page a packet finishes on, so we stamp
/// 0 — a defined value a downstream mapping can ignore. Callers wanting real granules
/// drive the library [`OggWriter`] directly.
const DEFAULT_GRANULE: u64 = 0;

/// The default serial number for [`OggMux::new`]. An arbitrary fixed value; a real
/// multiplexing caller assigns a unique per-stream serial (§6, field 5) via
/// [`OggMux::with_serial`].
pub const DEFAULT_SERIAL: u32 = 0x5C_09_67_67; // "\x5C" + "Ogg"-ish, just a constant

// The src pad's local index (== position in the element's `pads` array). The sink pad
// (index 0) needs no id here: milestone-1 input is drained via `Inputs::pop`, which is
// pad-agnostic.
const SRC: PadId = PadId(1);

/// Raw `bytes` on both pads: an Ogg stream on one side, codec packets on the other.
/// Matches the `flacenc`/`flacdec` byte pads so an Ogg element links to a codec element
/// at link time without a typed vocabulary (the container is codec-agnostic).
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

// =====================================================================================
// OggDemux
// =====================================================================================

static DEMUX_PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        // A single static src pad: this element handles the *first* logical bitstream
        // only. One dynamic src pad per discovered serial is the multi-stream follow-up.
        dynamic: false,
        validate: None,
    },
];

static DEMUX_DESC: ElementDesc = ElementDesc {
    name: "oggdemux",
    pads: &DEMUX_PADS,
    props: &[],
    // Passive: a pure byte→packet transform, inlines into the upstream group.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Demultiplexes a single logical Ogg bitstream: an Ogg byte stream on the sink pad,
/// one reassembled codec packet **per output buffer** on the src pad.
///
/// Only the **first** logical bitstream (the first bos serial seen) is emitted; packets
/// on any other serial are dropped. Handling several concurrently-multiplexed streams
/// needs one dynamic src pad per serial (a documented follow-up — `spec/NOTES.md`). The
/// wrapped [`OggReader`] already demuxes every serial and resyncs past corruption without
/// panicking, so bad input is safe; this element just filters to one serial.
#[derive(Default)]
pub struct OggDemux {
    reader: OggReader,
    /// The serial we locked onto — the first one a packet arrived for. `None` until the
    /// first packet is seen, after which only this serial's packets are emitted.
    serial: Option<u32>,
}

impl OggDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one whole packet as a **single** output buffer on the src pad, preserving the
    /// packet boundary (buffer i out == packet i). A packet larger than the pool slot is
    /// split across buffers (it cannot fit one fixed slot); packets up to a slot stay 1:1,
    /// which covers everything short of a multi-page giant. A zero-length packet still
    /// emits one empty buffer, so nil packets survive the round-trip.
    fn emit_packet(ctx: &mut Ctx, data: &[u8]) {
        // Empty packet: emit exactly one zero-length buffer to preserve the boundary.
        if data.is_empty() {
            let mut buf = ctx.alloc(SRC);
            buf.memory.set_len(0);
            ctx.out(SRC).push(buf);
            return;
        }
        let mut off = 0;
        while off < data.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "oggdemux: zero-capacity pool slot");
            let n = cap.min(data.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&data[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
    }

    /// Drain every packet the reader has reassembled so far, emitting those on the locked
    /// serial. The first packet seen locks the serial (the first bos serial in a
    /// well-formed stream).
    fn drain(&mut self, ctx: &mut Ctx) {
        while let Some(packet) = self.reader.next_packet() {
            let serial = *self.serial.get_or_insert(packet.serial);
            if packet.serial == serial {
                Self::emit_packet(ctx, &packet.data);
            }
            // Packets on other serials are ignored — single-stream demux (follow-up:
            // dynamic pads for multi-stream).
        }
    }
}

impl Element for OggDemux {
    fn desc(&self) -> &'static ElementDesc {
        &DEMUX_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.reader = OggReader::new();
        self.serial = None;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Feed the bytes to the page engine; it parses whole pages and queues
            // reassembled packets. `buf` recycles on drop at the end of this iteration.
            self.reader.push(buf.memory.data());
            self.drain(ctx);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // On EOS, tell the reader no more bytes are coming and drain any final packet a
        // just-closed page completed. `finish` drops only unparsable trailing garbage; a
        // clean stream ends on a page boundary with nothing left.
        if matches!(event, Event::Eos) {
            self.reader.finish();
            self.drain(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: flush any final packet even if `event(Eos)` was not delivered.
        // `finish` + `drain` are idempotent once the queue is empty.
        self.reader.finish();
        self.drain(ctx);
        self.reader = OggReader::new();
        self.serial = None;
    }
}

// =====================================================================================
// OggMux
// =====================================================================================

static MUX_PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        // A single static sink pad: this element muxes one input stream. One dynamic sink
        // pad per input stream is the multi-stream follow-up.
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
];

static MUX_DESC: ElementDesc = ElementDesc {
    name: "oggmux",
    pads: &MUX_PADS,
    props: &[],
    // Passive: a pure packet→byte transform, inlines into the upstream group.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Multiplexes a single logical bitstream: one codec packet **per input buffer** on the
/// sink pad, an Ogg byte stream on the src pad. Each input buffer is wrapped as one Ogg
/// packet (its whole `data()`, including zero-length "nil" packets) through the wrapped
/// [`OggWriter`], and its page is emitted as bytes.
///
/// The stream is terminated with an eos page at end-of-stream — see the module docs on
/// EOS flushing. Handling several input streams into one multiplexed Ogg stream needs one
/// dynamic sink pad per input (a documented follow-up — `spec/NOTES.md`).
pub struct OggMux {
    serial: u32,
    /// The page writer. Present between `start` and the eos flush; `None` before start
    /// and after the stream has been finished (so a second flush is a no-op).
    writer: Option<OggWriter>,
    /// Reused byte buffer the writer appends pages into, so steady-state muxing does not
    /// reallocate a fresh `Vec` per `process`.
    scratch: Vec<u8>,
}

impl Default for OggMux {
    fn default() -> Self {
        Self::with_serial(DEFAULT_SERIAL)
    }
}

impl OggMux {
    /// A muxer with the [`DEFAULT_SERIAL`]. Use [`with_serial`](Self::with_serial) to set
    /// a specific bitstream serial number (§6, field 5) — required when several muxed
    /// streams share a physical Ogg stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// A muxer stamping `serial` as the bitstream serial number (§6, field 5).
    pub fn with_serial(serial: u32) -> Self {
        Self {
            serial,
            writer: None,
            scratch: Vec::new(),
        }
    }

    pub fn serial(&self) -> u32 {
        self.serial
    }

    /// Push accumulated Ogg bytes onto the src pad, chunked to the pool slot size so no
    /// single copy exceeds a buffer. Clears `bytes` afterwards, retaining its capacity.
    fn emit(ctx: &mut Ctx, bytes: &mut Vec<u8>) {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "oggmux: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
        bytes.clear();
    }

    /// Flush the terminating eos page (§6, flag 0x04) exactly once, pushing it downstream.
    /// Idempotent: the writer is taken on first call, so later calls (the `event` path
    /// then `stop`, or vice versa) do nothing.
    fn finish_stream(&mut self, ctx: &mut Ctx) {
        if let Some(mut writer) = self.writer.take() {
            // Guarantees a page with the eos flag — the pending page if any, else a nil
            // eos page (§6 allows "'nil' pages … with the eos flag set").
            writer.finish(&mut self.scratch);
            let mut out = std::mem::take(&mut self.scratch);
            Self::emit(ctx, &mut out);
            self.scratch = out; // reclaim the allocation
        }
    }
}

impl Element for OggMux {
    fn desc(&self) -> &'static ElementDesc {
        &MUX_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.writer = Some(OggWriter::new(self.serial));
        self.scratch.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            let writer = match self.writer.as_mut() {
                Some(w) => w,
                // Stream already finished (eos emitted): a late buffer after EOS is
                // dropped rather than reopening a closed stream.
                None => break,
            };
            // One input buffer == one Ogg packet (its whole payload, nil packets
            // included). `write_packet` segments it and flushes any *full* pages into
            // `scratch`, packing several packets per page; the pending partial page is
            // flushed on EOS by `finish_stream`, which terminates the last real packet's
            // page with the eos flag — no trailing nil packet. `AfterEos` cannot occur — we
            // hold the only writer and drop it only in `finish_stream`.
            let _ = writer.write_packet(&mut self.scratch, buf.memory.data(), DEFAULT_GRANULE);
            // `buf` recycles here on drop; move any completed page(s) downstream.
            let mut out = std::mem::take(&mut self.scratch);
            Self::emit(ctx, &mut out);
            self.scratch = out;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // Primary EOS path: the pipeline delivers EOS in-band and pushes whatever this
        // produces before closing the ring. Flush the terminating eos page now.
        if matches!(event, Event::Eos) {
            self.finish_stream(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: if `event(Eos)` did not run (or the runtime does not deliver
        // it yet), guarantee the eos page here. Idempotent via the `Option` take.
        self.finish_stream(ctx);
        self.scratch = Vec::new();
    }
}
