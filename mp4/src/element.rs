//! `Mp4Demux` — the ISO-BMFF container demultiplexer **element** (spec: `spec/NOTES.md`;
//! ISO/IEC 14496-12; dynamic pads). A progressive MP4 byte stream arrives on the sink pad;
//! each track's samples leave on their own runtime src pad, reframed so a downstream decoder
//! can start (avc1/hvc1: an Annex B parameter-set head from `avcC`/`hvcC`, then Annex B
//! access units; vp09/av01/Opus/AAC: the raw samples verbatim).
//!
//! This is the **exact structural sibling** of [`sc_mkv::MkvDemux`]: constructor-supplied
//! track discovery, one dynamic src pad per track, per-family reframing, runtime format
//! announcement, and — most importantly — the same **pool-bounded emission discipline**
//! (spec: the backpressure rule; the movie OOM). Read them side by side.
//!
//! ## Discovery is constructor-supplied (the chosen preroll strategy)
//! Track discovery — and therefore the src-pad set — must be settled in
//! [`preroll`](Element::preroll), because the scheduler freezes the topology after preroll.
//! But a *mid-pipeline* element gets **no input during preroll**. So `Mp4Demux` takes the
//! file's **head bytes at construction** ([`new`](Self::new)) — enough to cover `ftyp` +
//! the whole `moov` — and resolves the sample tables in `preroll` to learn the tracks and
//! add one src pad each. The full file (from byte 0) then arrives on the sink pad during
//! `process`, where the reader slices each resolved sample out by absolute offset. This is
//! the same accepted fallback `MkvDemux::new` uses (see its docs); MP4 *needs* it doubly,
//! because its sample tables live in `moov` and cannot be resolved incrementally at all.
//!
//! ## The backpressure discipline (spec: the pool bounds a demuxer, not the heap)
//! On the hot path samples are emitted as **refcounted slices of the retained input
//! chunks** (ZERO-COPY.md Stage 1) — no pool slot, no memcpy; the *source's* pool bounds
//! memory, because a downstream consumer holding slices keeps the read slots outstanding
//! and stalls the source's `try_alloc`. This element's own pool serves only the cold
//! byte paths — codec heads, chunk-straddle gathers, NAL→Annex B conversions — via
//! [`Ctx::try_alloc`] with a bounded **pending carry**: input is consumed **only while
//! emission keeps up** (un-popped input stays buffered, so upstream backs up into the
//! scheduler's gates, not this element's memory); the EOS/stop flush uses
//! [`Ctx::alloc_exact`] (right-sized, never a slot-sized over-allocation per small sample).
//! Unbounded `ctx.alloc` in a demuxer is **banned** — it OOM'd a real movie.

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::{Buffer, BufferFlags};
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{OfferDesc, ValueDesc};
use streamcraft_core::id::{FormatId, PadId};
use streamcraft_core::memory::Memory;
use streamcraft_core::time::Timestamp;

use crate::codec::{self, Reframer};
use crate::reader::{Mp4Error, Mp4Reader, SamplePayload};

/// The `bytes` offer on the sink pad: a raw MP4 byte stream (the container is codec-agnostic,
/// matching `MkvDemux`'s sink and any byte-producing upstream like `filesrc`).
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

/// The fallback announce family for an unknown/other codec: a raw `bytes` stream.
const FAMILY_BYTES: &str = "bytes";

/// Every announce family a demux src pad may carry, so a pad can announce the family it maps
/// to (a pad only announces a family it already offered — the vocabulary is interned from
/// these offers at link time). The families match the codec decoders' sink offers so
/// `mp4demux.src_track<N> ! h264dec.sink` (etc.) negotiates (spec: RFC 6381 four-CCs;
/// [`codec::family_for`]). All unconstrained byte families — concrete dims/rate ride the
/// runtime announcement, not the offer.
static DEMUX_SRC_OFFERS: [OfferDesc; 9] = [
    OfferDesc::any(FAMILY_BYTES),
    OfferDesc::any("h264/annexb"),
    OfferDesc::any("h265/annexb"),
    // The passthrough/remux framings (see [`Mp4Demux::passthrough`]): samples stay
    // length-prefixed as stored and the raw `avcC`/`hvcC` record leads in-band — exactly
    // Matroska's `V_MPEG4/ISO/AVC`//`V_MPEGH/ISO/HEVC` shape (RFC 9559 §12).
    OfferDesc::any("h264/avcc"),
    OfferDesc::any("h265/hvcc"),
    OfferDesc::any("vp9"),
    OfferDesc::any("av1"),
    OfferDesc::any("opus"),
    // `mp4a` with an esds-carried AudioSpecificConfig: raw AAC access units (exactly what
    // MP4 stores, and exactly what Matroska A_AAC wants — RFC 9559 §12: no ADTS framing),
    // led in-band by the ASC as the first buffer (the same first-buffer-is-CodecPrivate
    // contract the NAL passthrough families use). ASC-less `mp4a` stays `bytes`.
    OfferDesc::any("aac"),
];

static DEMUX_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];

static DEMUX_DESC: ElementDesc = ElementDesc {
    name: "mp4demux",
    pads: &DEMUX_PADS,
    props: &[],
    // Passive: a pure byte→frame transform, inlines into the upstream group like `mkvdemux`.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Not name-constructible: needs the file head at construction (see `register`).
    make_default: None,
};

/// A discovered track wired to its runtime src pad. Built during [`Mp4Demux::preroll`], read
/// during `process` to route samples and announce the pad format.
struct PadTrack {
    /// The runtime src pad this track's samples go out on ([`Ctx::add_pad`]).
    pad: PadId,
    /// The track's media timescale (ticks/s), cached so pts can be stamped while a
    /// sample still borrows the reader — the zero-copy drain never re-borrows it.
    timescale: u32,
    /// The announce family for the pad ([`codec::family_for`]): `h264/annexb`,
    /// `h265/annexb`, `vp9`, `av1`, `opus`, or `bytes`.
    family: &'static str,
    /// The Annex B parameter-set head to emit before the first frame (SPS/PPS from `avcC`,
    /// VPS/SPS/PPS from `hvcC`). Empty for raw / in-band-parameter-set / non-NAL tracks.
    codec_head: Vec<u8>,
    /// How each sample's payload is reframed: NAL length-prefix → Annex B for avc*/hvc*,
    /// passthrough otherwise.
    reframer: Reframer,
    /// Announce params captured at resolution: video dims (w,h) or audio (rate,channels).
    announce: Announce,
    /// False until the codec head + format announcement have been emitted on this pad.
    started: bool,
}

/// The runtime format params a src pad announces once, on its first sample.
#[derive(Clone, Copy)]
enum Announce {
    /// A video track: coded/presentation dims (0,0 → announce the bare family), plus the
    /// mdhd track duration in ns — announced **only in passthrough mode** (0 elsewhere):
    /// a remuxing consumer declares its offer field and writes `Info\Duration` from it,
    /// while decode pipelines never intern a "duration" field name, and an announcement
    /// with any un-interned name is dropped whole by `build_fixed`.
    Video { width: u32, height: u32, duration_ns: u64 },
    /// An audio track: sample rate + channels (0 → bare family), plus the mdhd track
    /// duration in ns under the same passthrough-only rule as `Video` (0 elsewhere) — a
    /// remuxing consumer takes the max across its pads for `Info\Duration`.
    Audio { rate: u32, channels: u32, duration_ns: u64 },
    /// No declared params — a bare family announcement.
    Bare,
}

/// A partially-emitted blob parked on pool exhaustion — or queued *behind* one, to keep
/// per-pad ordering: `payload[off..]` still needs emitting on `pad`, stamped `pts_ns`.
struct Carry {
    pad: PadId,
    pts_ns: Option<u64>,
    flags: BufferFlags,
    payload: CarryPayload,
    off: usize,
}

/// What a parked emission owns: copied bytes (a codec head, a reframed access unit, a
/// chunk-straddle gather — these need pool slots), or a zero-copy retained slice, which
/// needs no slot and parks only to stay ordered behind an owned carry that hit a dry pool.
enum CarryPayload {
    Owned(Vec<u8>),
    Slice(Memory),
}

/// Demultiplexes a progressive MP4 byte stream into one src pad per track (spec: dynamic
/// pads). See the module docs for the constructor-supplied discovery and the pool-bounded
/// emission discipline (both mirroring [`sc_mkv::MkvDemux`]).
pub struct Mp4Demux {
    /// The file head handed in at construction — resolved in `preroll` to discover tracks.
    head: Vec<u8>,
    /// Remux mode (see [`passthrough`](Self::passthrough)): NAL tracks keep their stored
    /// length-prefixed framing and lead with the raw config record instead of being
    /// reframed to Annex B for a decoder.
    passthrough: bool,
    /// The discovered tracks wired to their src pads, filled in `preroll`.
    pad_tracks: Vec<PadTrack>,
    /// The parse engine: resolved from the head in `preroll`, slices samples during
    /// `process`. `None` until `preroll` resolves the tables.
    reader: Option<Mp4Reader>,
    /// Emissions stalled on pool exhaustion (spec: the backpressure rule). While non-empty
    /// the demuxer consumes **no input**. Holds at most a codec head + one sample.
    pending: std::collections::VecDeque<Carry>,
    /// The `FormatId` a pool-allocated buffer on a src pad would carry (`Ctx` stamps its
    /// out-format; there is no public accessor), probed once at `start` so the zero-copy
    /// slice path can construct [`Buffer`]s identical to pooled ones. Not load-bearing on
    /// the milestone-1 transport — `Batch::push` drops the per-row format and `pop_front`
    /// restamps from the batch — but kept faithful for later transports.
    out_fmt: FormatId,
    /// Reused NAL→Annex B reframe buffer: each length-prefixed → Annex B conversion refills
    /// this in place ([`Reframer::reframe_into`]), so the per-sample reframe allocates nothing
    /// on the global heap (mirrors `sc_mkv::MkvDemux`). Passthrough leaves it untouched.
    reframe_buf: Vec<u8>,
}

impl Mp4Demux {
    /// A demuxer that resolves its tracks from `head` — the leading bytes of a progressive
    /// MP4, which MUST include `ftyp` + the whole `moov` (everything up to `mdat`). The
    /// caller reads these once up front; the full file (these bytes and all that follow) is
    /// then fed on the sink pad at run time.
    ///
    /// Resolution is deferred to [`preroll`](Element::preroll) so a construction-time parse
    /// failure surfaces as a preroll error (loud, at the right phase) rather than a panic in
    /// `new`. `new` therefore always succeeds; a malformed head errors in `preroll`.
    // COLD: one-time constructor — builds fixed element state, not a per-sample path.
    #[allow(clippy::disallowed_methods)]
    pub fn new(head: Vec<u8>) -> Self {
        Self {
            head,
            passthrough: false,
            pad_tracks: Vec::new(),
            reader: None, // resolved in `preroll`
            pending: std::collections::VecDeque::new(),
            out_fmt: FormatId(0), // probed at `start`
            reframe_buf: Vec::new(),
        }
    }

    /// A demuxer in **remux mode**: NAL tracks (`avc1`/`hvc1`) are *not* reframed to
    /// Annex B — samples go out length-prefixed exactly as stored, led by the raw
    /// `avcC`/`hvcC` record as the in-band codec head, under the `h264/avcc` /
    /// `h265/hvcc` families. That is precisely the shape Matroska wants
    /// (`V_MPEG4/ISO/AVC` CodecPrivate = the record verbatim, RFC 9559 §12), so
    /// `mp4demux(passthrough) ! mkvmux` remuxes without touching a single sample byte.
    /// Non-NAL tracks (vp9/av1/audio) are already passthrough and keep their families.
    /// In-band-parameter-set files (`avc3`/`hev1`, no config record) are a loud preroll
    /// error in this mode — they have no record to hand Matroska.
    pub fn passthrough(head: Vec<u8>) -> Self {
        let mut d = Self::new(head);
        d.passthrough = true;
        d
    }

    /// The tracks discovered during `preroll` (for tests / introspection). Empty before it.
    pub fn track_count(&self) -> usize {
        self.pad_tracks.len()
    }

    /// The src pad a track (0-based index) routes to, if discovered.
    pub fn pad_for(&self, track_index: usize) -> Option<PadId> {
        self.pad_tracks.get(track_index).map(|pt| pt.pad)
    }

    /// Push `bytes[off..]` on `pad` via **pooled** slots ([`Ctx::try_alloc`]), chunked to the
    /// slot size, returning the new offset — `< bytes.len()` means the pool ran dry and the
    /// caller must carry the remainder and stop consuming input (spec: the backpressure rule).
    fn emit_bounded(
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        flags: BufferFlags,
        bytes: &[u8],
        mut off: usize,
    ) -> usize {
        while off < bytes.len() {
            let Some(mut buf) = ctx.try_alloc(pad) else { return off };
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "mp4demux: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            if let Some(t) = pts_ns {
                buf.pts = Timestamp::from_nanos(t);
            }
            buf.flags = flags;
            ctx.out(pad).push(buf);
            off += n;
        }
        off
    }

    /// Emit `bytes[off..]` as one **exact-size** buffer ([`Ctx::alloc_exact`] — a right-sized
    /// heap allocation when the pool is dry, never a slot-sized over-allocation). Only for the
    /// bounded EOS/stop flush, where yielding for a slot is no longer possible.
    fn emit_exact(
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        flags: BufferFlags,
        bytes: &[u8],
        off: usize,
    ) {
        let rest = &bytes[off..];
        if rest.is_empty() {
            return;
        }
        let mut buf = ctx.alloc_exact(pad, rest.len());
        buf.memory.as_mut_full()[..rest.len()].copy_from_slice(rest);
        buf.memory.set_len(rest.len());
        if let Some(t) = pts_ns {
            buf.pts = Timestamp::from_nanos(t);
        }
        buf.flags = flags;
        ctx.out(pad).push(buf);
    }

    /// Emit a retained-input slice as **one buffer** — a refcount bump, no pool slot, no
    /// memcpy (ZERO-COPY.md Stage 1). Never blocks: the backing is already accounted for
    /// by the pool the input chunk came from, and downstream retention of the slice is
    /// exactly what keeps that slot outstanding (the backpressure loop).
    fn emit_slice(
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        flags: BufferFlags,
        memory: Memory,
        format: FormatId,
    ) {
        ctx.out(pad).push(Buffer {
            memory,
            pts: pts_ns.map_or(Timestamp::NONE, Timestamp::from_nanos),
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags,
            format,
            sync: None,
        });
    }

    /// Queue a byte blob for emission and try to emit it now; on pool exhaustion the
    /// remainder parks in `pending` (drained first on the next pass). Takes the queue
    /// directly so callers under the drain's split field borrows can still park.
    fn emit_or_park(
        pending: &mut std::collections::VecDeque<Carry>,
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        flags: BufferFlags,
        bytes: Vec<u8>,
    ) {
        let off = Self::emit_bounded(ctx, pad, pts_ns, flags, &bytes, 0);
        if off < bytes.len() {
            pending.push_back(Carry { pad, pts_ns, flags, payload: CarryPayload::Owned(bytes), off });
        }
    }

    /// Resume parked emissions. `true` when everything pending has drained. Owned carries
    /// go through pool slots (and re-park when the pool is still dry); slice carries never
    /// block — they were only parked to stay ordered behind an owned one.
    fn drain_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some(c) = self.pending.pop_front() {
            match c.payload {
                CarryPayload::Owned(bytes) => {
                    let off = Self::emit_bounded(ctx, c.pad, c.pts_ns, c.flags, &bytes, c.off);
                    if off < bytes.len() {
                        self.pending.push_front(Carry {
                            pad: c.pad,
                            pts_ns: c.pts_ns,
                            flags: c.flags,
                            payload: CarryPayload::Owned(bytes),
                            off,
                        });
                        return false;
                    }
                }
                CarryPayload::Slice(mem) => {
                    Self::emit_slice(ctx, c.pad, c.pts_ns, c.flags, mem, self.out_fmt);
                }
            }
        }
        true
    }

    /// Per-track pad wiring — announce family, in-band codec head, and reframer — for the
    /// current mode. Decode mode (default): Annex B head + NAL reframing, families the
    /// decoders link on. Passthrough mode: the raw config record verbatim as the head,
    /// samples untouched, the `h264/avcc`/`h265/hvcc` remux families (RFC 9559 §12 shape).
    // COLD: run once per track at preroll/start to seed the codec head — not per-sample.
    #[allow(clippy::disallowed_methods)]
    fn track_wiring(
        track: &crate::reader::Track,
        passthrough: bool,
    ) -> Result<(&'static str, Vec<u8>, Reframer), &'static str> {
        if passthrough {
            if let k @ (b"avc1" | b"avc3" | b"hvc1" | b"hev1") = &track.entry.kind {
                if track.entry.config_record.is_empty() {
                    return Err(
                        "mp4demux passthrough: no avcC/hvcC record (in-band avc3/hev1 \
                         parameter sets) — nothing to hand Matroska as CodecPrivate",
                    );
                }
                let is_hevc = matches!(k, b"hvc1" | b"hev1");
                let family = if is_hevc { "h265/hvcc" } else { "h264/avcc" };
                return Ok((family, track.entry.config_record.clone(), Reframer::Passthrough));
            }
        }
        Ok((track.family(), track.codec_head().to_vec(), track.reframer().clone()))
    }

    /// Drain samples the reader has resolved, emitting each on its track's src pad —
    /// **zero-copy on the hot path** (ZERO-COPY.md Stage 1): a passthrough sample goes out
    /// as the reader's retained-input slice itself — a refcount bump, no pool slot, no
    /// memcpy. The remaining per-sample byte traffic is exactly: a real NAL→Annex B
    /// conversion (decode mode), a chunk-straddle gather (cold, ~one per retained-chunk
    /// boundary), or a pool-dry carry — all owned blobs through pool slots under the
    /// carry discipline. Pool-bounded as before: returns `false` when slots ran out
    /// (remainder parked in `pending`; stop consuming input until it clears), and a slice
    /// arriving while an owned carry is parked queues *behind* it to keep per-pad order.
    /// On a track's first sample the codec head is emitted first and the pad's runtime
    /// format announced (spec: dynamic caps). A sample that fails to reframe is
    /// warned-and-dropped, not fatal (spec: untrusted input). The split field borrows
    /// (`let Self { reader, pad_tracks, pending, .. }`) keep the queue usable while the
    /// reader is borrowed.
    fn drain(&mut self, ctx: &mut Ctx) -> bool {
        if !self.drain_pending(ctx) {
            return false;
        }
        let fmt = self.out_fmt;
        loop {
            let Self { reader, pad_tracks, pending, reframe_buf, .. } = self;
            let Some(reader) = reader.as_mut() else { return true };
            let Some(s) = reader.next_sample() else { return true };
            let (idx, sync) = (s.track_index, s.sync);
            let pt = &mut pad_tracks[idx];
            let pts_ns = crate::reader::ticks_to_ns(s.pts, pt.timescale);
            // The sample table's sync bit rides the buffer (spec: Buffer flags), so a
            // remuxing consumer preserves seekability; DELTA is explicit — untagged would
            // read as "unknown", which a muxer defaults to keyframe.
            let flags = if sync { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            // First sample on this pad: announce the format, then emit the codec head.
            if !pt.started {
                pt.started = true;
                let (pad, family, ann) = (pt.pad, pt.family, pt.announce);
                let head = std::mem::take(&mut pt.codec_head);
                Self::announce(ctx, pad, family, ann);
                if !head.is_empty() {
                    Self::emit_or_park(pending, ctx, pad, Some(pts_ns), BufferFlags::empty(), head);
                }
            }
            let pad = pt.pad;
            // Route the payload: slice-through for passthrough (the zero-copy path),
            // otherwise an owned blob (straddle gather) or the reused reframe buffer.
            let owned: Vec<u8> = match (&pt.reframer, s.payload) {
                (Reframer::Passthrough, SamplePayload::Slice(mem)) => {
                    if pending.is_empty() {
                        Self::emit_slice(ctx, pad, Some(pts_ns), flags, mem, fmt);
                    } else {
                        // A parked head precedes this sample — keep order, park behind it.
                        pending.push_back(Carry {
                            pad,
                            pts_ns: Some(pts_ns),
                            flags,
                            payload: CarryPayload::Slice(mem),
                            off: 0,
                        });
                        return false;
                    }
                    continue;
                }
                (Reframer::Passthrough, SamplePayload::Copied(v)) => v,
                (reframer, payload) => {
                    // NAL→Annex B into the reused buffer — allocates nothing per sample on the
                    // common (pool-has-slots) path; only a parked remainder is owned (cold).
                    let bytes = match reframer.reframe_into(payload.data(), reframe_buf) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            let element = ctx.element();
                            ctx.post(BusMessage::Warning {
                                element,
                                error: Error::Element {
                                    element,
                                    message: format!(
                                        "mp4demux: dropping malformed sample on track {idx}: {e:?}"
                                    ),
                                },
                            });
                            continue; // drop this sample, try the next
                        }
                    };
                    let off = if pending.is_empty() {
                        Self::emit_bounded(ctx, pad, Some(pts_ns), flags, bytes, 0)
                    } else {
                        0 // a parked head precedes this sample — keep order, park whole
                    };
                    if off < bytes.len() {
                        // Pool dry (or head parked ahead): own only the un-emitted remainder.
                        pending.push_back(Carry {
                            pad,
                            pts_ns: Some(pts_ns),
                            flags,
                            payload: CarryPayload::Owned(own_remainder(bytes, off)),
                            off: 0,
                        });
                    }
                    if !pending.is_empty() {
                        return false; // pool dry — carry parked, stop pulling reader samples
                    }
                    continue;
                }
            };
            let off = if pending.is_empty() {
                Self::emit_bounded(ctx, pad, Some(pts_ns), flags, &owned, 0)
            } else {
                0 // a parked head precedes this sample — keep order, park whole
            };
            if off < owned.len() {
                pending.push_back(Carry {
                    pad,
                    pts_ns: Some(pts_ns),
                    flags,
                    payload: CarryPayload::Owned(owned),
                    off,
                });
            }
            if !pending.is_empty() {
                return false; // pool dry — carry parked, stop pulling reader samples
            }
        }
    }

    /// The EOS/stop flush: everything parked or still resolvable goes out — retained
    /// slices as-is (still zero-copy; no slot needed even here), owned blobs with
    /// **exact-size** allocations (no slot to wait for at end of stream, and no
    /// slot-sized over-allocation per small sample). Bounded in volume: the steady-state
    /// path only consumed input while it could emit, so at most one input batch's worth
    /// of samples sits here.
    fn flush_exact(&mut self, ctx: &mut Ctx) {
        let fmt = self.out_fmt;
        while let Some(c) = self.pending.pop_front() {
            match c.payload {
                CarryPayload::Owned(bytes) => {
                    Self::emit_exact(ctx, c.pad, c.pts_ns, c.flags, &bytes, c.off)
                }
                CarryPayload::Slice(mem) => {
                    Self::emit_slice(ctx, c.pad, c.pts_ns, c.flags, mem, fmt)
                }
            }
        }
        loop {
            let Self { reader, pad_tracks, reframe_buf, .. } = self;
            let Some(reader) = reader.as_mut() else { return };
            let Some(s) = reader.next_sample() else { return };
            let (idx, sync) = (s.track_index, s.sync);
            let pt = &mut pad_tracks[idx];
            let pts_ns = crate::reader::ticks_to_ns(s.pts, pt.timescale);
            let flags = if sync { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            if !pt.started {
                pt.started = true;
                let (pad, family, ann) = (pt.pad, pt.family, pt.announce);
                let head = std::mem::take(&mut pt.codec_head);
                Self::announce(ctx, pad, family, ann);
                if !head.is_empty() {
                    Self::emit_exact(ctx, pad, Some(pts_ns), BufferFlags::empty(), &head, 0);
                }
            }
            let pad = pt.pad;
            match (&pt.reframer, s.payload) {
                (Reframer::Passthrough, SamplePayload::Slice(mem)) => {
                    Self::emit_slice(ctx, pad, Some(pts_ns), flags, mem, fmt)
                }
                (Reframer::Passthrough, SamplePayload::Copied(v)) => {
                    Self::emit_exact(ctx, pad, Some(pts_ns), flags, &v, 0)
                }
                (reframer, payload) => match reframer.reframe_into(payload.data(), reframe_buf) {
                    Ok(bytes) => Self::emit_exact(ctx, pad, Some(pts_ns), flags, bytes, 0),
                    Err(_) => continue, // tail flush: drop silently rather than warn-spam
                },
            }
        }
    }

    /// Announce a src pad's runtime format (spec: dynamic caps; follows `MkvDemux`). Video
    /// families carry `width`/`height` when declared; audio carries `rate`/`channels`;
    /// otherwise a bare family announcement. Every family rides a byte bridge, so a
    /// caps-ignoring decoder still links on the family offer.
    // COLD: runs once per pad on its first sample to assemble the announce field list.
    #[allow(clippy::disallowed_methods)]
    fn announce(ctx: &mut Ctx, pad: PadId, family: &'static str, announce: Announce) {
        match announce {
            Announce::Video { width, height, duration_ns } if width != 0 && height != 0 => {
                let dims = [
                    ("width", ValueDesc::Int(width as i64)),
                    ("height", ValueDesc::Int(height as i64)),
                ];
                if duration_ns != 0 {
                    // Passthrough/remux only (see `Announce::Video`): the muxer writes
                    // `Info\Duration` from this, so the output is not a duration-less
                    // "live" stream.
                    let mut fields = dims.to_vec();
                    fields.push(("duration", ValueDesc::Int(duration_ns as i64)));
                    ctx.announce_format(pad, family, &fields);
                } else {
                    ctx.announce_format(pad, family, &dims);
                }
            }
            Announce::Audio { rate, channels, duration_ns } if rate != 0 || channels != 0 => {
                let params = [
                    ("rate", ValueDesc::Int(rate as i64)),
                    ("channels", ValueDesc::Int(channels as i64)),
                ];
                if duration_ns != 0 {
                    // Passthrough/remux only (see `Announce::Audio`): the muxer folds this
                    // into `Info\Duration` (max across its tracks).
                    let mut fields = params.to_vec();
                    fields.push(("duration", ValueDesc::Int(duration_ns as i64)));
                    ctx.announce_format(pad, family, &fields);
                } else {
                    ctx.announce_format(pad, family, &params);
                }
            }
            _ => ctx.announce_format(pad, family, &[]),
        }
    }
}

impl Element for Mp4Demux {
    fn desc(&self) -> &'static ElementDesc {
        &DEMUX_DESC
    }

    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Resolve the sample tables from the constructor-supplied head, then add one src pad
        // per track (spec: dynamic pads — topology settles at preroll). A resolution failure
        // is a loud preroll error.
        let reader = Mp4Reader::new(&self.head).map_err(map_resolve_err)?;
        for (i, track) in reader.tracks().iter().enumerate() {
            let (family, codec_head, reframer) =
                Self::track_wiring(track, self.passthrough).map_err(Error::Todo)?;
            let announce = if codec::is_video_family(track.family()) {
                Announce::Video {
                    width: track.width,
                    height: track.height,
                    duration_ns: if self.passthrough { track.duration_ns } else { 0 },
                }
            } else if family == "opus"
                || family == "aac"
                || track.entry.sample_rate != 0
                || track.entry.channels != 0
            {
                Announce::Audio {
                    rate: track.entry.sample_rate,
                    channels: track.entry.channels,
                    duration_ns: if self.passthrough { track.duration_ns } else { 0 },
                }
            } else {
                Announce::Bare
            };
            let name = format!("src_track{}", track.track_id);
            let pad = ctx.add_pad(Direction::Src, &name, &DEMUX_SRC_OFFERS);
            self.pad_tracks.push(PadTrack {
                pad,
                timescale: track.timescale,
                family,
                codec_head,
                reframer,
                announce,
                started: false,
            });
            let _ = i;
        }
        self.reader = Some(reader);
        Ok(())
    }

    // COLD: lifecycle start — the error-message string only builds on a misuse path, once.
    #[allow(clippy::disallowed_methods)]
    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Probe the out-format a pooled buffer would carry (see the `out_fmt` field): one
        // throwaway right-sized allocation, recycled on drop — never a steady-state cost.
        self.out_fmt = ctx.alloc_exact(PadId(0), 1).format;
        // Rewind the streaming cursor; the whole file (from byte 0) arrives on the sink pad.
        // The resolved tables are retained (resolution happened once, in `preroll`).
        let Some(reader) = self.reader.as_mut() else {
            return Err(Error::Resource(
                "mp4demux: started before preroll resolved the tables".to_string(),
            ));
        };
        reader.reset_stream();
        for pt in &mut self.pad_tracks {
            pt.started = false;
        }
        // Re-derive the codec heads (taken/consumed on a previous run) from the resolved
        // tracks, honouring the mode (wiring errors already surfaced at preroll).
        let heads: Vec<Vec<u8>> = reader
            .tracks()
            .iter()
            .map(|t| Self::track_wiring(t, self.passthrough).map(|(_, h, _)| h).unwrap_or_default())
            .collect();
        for (pt, head) in self.pad_tracks.iter_mut().zip(heads) {
            pt.codec_head = head;
        }
        self.pending.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Backpressure: resume any parked emission first, and consume input only while
        // emission keeps up (un-popped input stays buffered for the next pass, so the
        // scheduler's gates absorb a fast source racing a slow decoder — not this element).
        if !self.drain(ctx) {
            return Ok(Flow::Ok);
        }
        while let Some(buf) = inputs.pop() {
            // Retain the input chunk's `Memory` itself (a refcount move, no copy —
            // ZERO-COPY.md Stage 1); the reader keeps it alive exactly until every sample
            // sliced from it has been emitted. The chunk's pool slot stays outstanding
            // for as long as the reader — or any downstream holder of an emitted slice —
            // needs the bytes; that retention is the demuxer's backpressure.
            if let Some(reader) = self.reader.as_mut() {
                reader.push(buf.memory);
            }
            if !self.drain(ctx) {
                return Ok(Flow::Ok);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // On EOS, flush every sample whose bytes are buffered — exact-size allocations, since
        // waiting for pool slots is no longer an option.
        if matches!(event, Event::Eos) {
            self.flush_exact(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: flush any final sample even if `event(Eos)` was not delivered.
        self.flush_exact(ctx);
        if let Some(reader) = self.reader.as_mut() {
            reader.reset_stream();
        }
        self.pending.clear();
    }
}

/// Own the un-emitted tail of a reframed sample when the pool ran dry, so it can be parked.
// COLD: only on pool exhaustion (backpressure) — the steady-state path emits borrowed bytes.
#[allow(clippy::disallowed_methods)]
fn own_remainder(bytes: &[u8], off: usize) -> Vec<u8> {
    bytes[off..].to_vec()
}

/// Map a resolver error to a worded `streamcraft` error (loud, at preroll).
// COLD: builds an error string only on a preroll resolve failure — never per-sample.
#[allow(clippy::disallowed_methods)]
fn map_resolve_err(e: Mp4Error) -> Error {
    let msg = match e {
        Mp4Error::Fragmented => "mp4demux: fragmented MP4 (moof/mvex) is not supported — v1 \
            resolves progressive files only (see mp4/spec/NOTES.md)"
            .to_string(),
        Mp4Error::Missing(what) => format!("mp4demux: required box/table missing: {what}"),
        Mp4Error::Inconsistent(what) => format!("mp4demux: inconsistent sample table: {what}"),
        Mp4Error::Box(be) => format!("mp4demux: malformed box: {be:?}"),
    };
    Error::Resource(msg)
}
