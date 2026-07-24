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
//! Copied verbatim from `mkvdemux` after the movie-OOM fix: emission is via
//! [`Ctx::try_alloc`] with a bounded **pending carry** — input is consumed **only while
//! emission keeps up** (un-popped input stays buffered, so upstream backs up into the
//! scheduler's gates, not this element's memory); the EOS/stop flush uses
//! [`Ctx::alloc_exact`] (right-sized, never a slot-sized over-allocation per small sample).
//! Unbounded `ctx.alloc` in a demuxer is **banned** — it OOM'd a real movie.

use streamcraft_core::batch::Inputs;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::codec::{self, Reframer};
use crate::reader::{Mp4Error, Mp4Reader};

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
static DEMUX_SRC_OFFERS: [OfferDesc; 6] = [
    OfferDesc::any(FAMILY_BYTES),
    OfferDesc::any("h264/annexb"),
    OfferDesc::any("h265/annexb"),
    OfferDesc::any("vp9"),
    OfferDesc::any("av1"),
    OfferDesc::any("opus"),
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
    /// A video track: coded/presentation dims (0,0 → announce the bare family).
    Video { width: u32, height: u32 },
    /// An audio track: sample rate + channels (0 → bare family).
    Audio { rate: u32, channels: u32 },
    /// No declared params — a bare family announcement.
    Bare,
}

/// A partially-emitted blob: `bytes[off..]` still needs pool slots on `pad`, stamped `pts_ns`.
struct Carry {
    pad: PadId,
    pts_ns: Option<u64>,
    bytes: Vec<u8>,
    off: usize,
}

/// Demultiplexes a progressive MP4 byte stream into one src pad per track (spec: dynamic
/// pads). See the module docs for the constructor-supplied discovery and the pool-bounded
/// emission discipline (both mirroring [`sc_mkv::MkvDemux`]).
pub struct Mp4Demux {
    /// The file head handed in at construction — resolved in `preroll` to discover tracks.
    head: Vec<u8>,
    /// The discovered tracks wired to their src pads, filled in `preroll`.
    pad_tracks: Vec<PadTrack>,
    /// The parse engine: resolved from the head in `preroll`, slices samples during
    /// `process`. `None` until `preroll` resolves the tables.
    reader: Option<Mp4Reader>,
    /// Emissions stalled on pool exhaustion (spec: the backpressure rule). While non-empty
    /// the demuxer consumes **no input**. Holds at most a codec head + one sample.
    pending: std::collections::VecDeque<Carry>,
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
    pub fn new(head: Vec<u8>) -> Self {
        Self {
            head,
            pad_tracks: Vec::new(),
            reader: None, // resolved in `preroll`
            pending: std::collections::VecDeque::new(),
        }
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
    fn emit_bounded(ctx: &mut Ctx, pad: PadId, pts_ns: Option<u64>, bytes: &[u8], mut off: usize) -> usize {
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
            ctx.out(pad).push(buf);
            off += n;
        }
        off
    }

    /// Emit `bytes[off..]` as one **exact-size** buffer ([`Ctx::alloc_exact`] — a right-sized
    /// heap allocation when the pool is dry, never a slot-sized over-allocation). Only for the
    /// bounded EOS/stop flush, where yielding for a slot is no longer possible.
    fn emit_exact(ctx: &mut Ctx, pad: PadId, pts_ns: Option<u64>, bytes: &[u8], off: usize) {
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
        ctx.out(pad).push(buf);
    }

    /// Queue a blob for emission and try to emit it now; on pool exhaustion the remainder
    /// parks in `pending` (drained first on the next pass).
    fn emit_or_park(&mut self, ctx: &mut Ctx, pad: PadId, pts_ns: Option<u64>, bytes: Vec<u8>) {
        let off = Self::emit_bounded(ctx, pad, pts_ns, &bytes, 0);
        if off < bytes.len() {
            self.pending.push_back(Carry { pad, pts_ns, bytes, off });
        }
    }

    /// Resume parked emissions. `true` when everything pending has drained.
    fn drain_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some(mut c) = self.pending.pop_front() {
            c.off = Self::emit_bounded(ctx, c.pad, c.pts_ns, &c.bytes, c.off);
            if c.off < c.bytes.len() {
                self.pending.push_front(c);
                return false;
            }
        }
        true
    }

    /// Drain samples the reader has sliced, emitting each on its track's src pad — **pool
    /// bounded**: returns `false` when slots ran out (remainder parked in `pending`; stop
    /// consuming input until it clears). On a track's first sample the Annex B codec head is
    /// emitted first and the pad's runtime format is announced (spec: dynamic caps). A sample
    /// that fails to reframe is warned-and-dropped, not fatal (spec: untrusted input).
    fn drain(&mut self, ctx: &mut Ctx) -> bool {
        if !self.drain_pending(ctx) {
            return false;
        }
        // Slice the next available sample; borrow the reframed bytes, compute the pts, then
        // release the reader borrow before touching pool memory (the reader's window may be
        // mutated by the next slice). `take_next` hands back owned locals so the reader borrow
        // is dropped before emission (which may re-enter the reader).
        while let Some((idx, pts_ns, sync, reframed, head_to_emit, announce)) = self.take_next(ctx) {
            let _ = sync; // sync flag not surfaced downstream yet (kept for a future keyframe tag)
            let pad = self.pad_tracks[idx].pad;

            // First sample on this pad: announce the format, then emit the codec head.
            if let Some((family, ann)) = announce {
                Self::announce(ctx, pad, family, ann);
            }
            if let Some(head) = head_to_emit {
                self.emit_or_park(ctx, pad, Some(pts_ns), head);
            }

            let off = Self::emit_bounded(ctx, pad, Some(pts_ns), &reframed, 0);
            if off < reframed.len() {
                self.pending.push_back(Carry { pad, pts_ns: Some(pts_ns), bytes: reframed, off });
            }
            if !self.pending.is_empty() {
                return false; // pool dry — carry parked, stop pulling reader samples
            }
        }
        true
    }

    /// Slice + reframe the next available sample, returning its routing/bytes/pts and (on the
    /// track's first sample) the announce params + codec head. `None` when no sample's bytes
    /// are fully buffered yet. A sample that fails to reframe is warned-and-dropped (loops to
    /// the next one). Written as a helper so the reader borrow is scoped tightly.
    #[allow(clippy::type_complexity)]
    fn take_next(
        &mut self,
        ctx: &mut Ctx,
    ) -> Option<(usize, u64, bool, Vec<u8>, Option<Vec<u8>>, Option<(&'static str, Announce)>)> {
        loop {
            // Pull the next sample; copy its fields into owned locals so the `ResolvedSample`
            // borrow of the reader's window is dropped before we call `ticks_to_ns` (a second
            // `&self` borrow) or emit (which may re-enter the reader).
            let reader = self.reader.as_mut()?;
            let (idx, pts_ticks, sync, raw) = {
                let s = reader.next_sample()?;
                (s.track_index, s.pts, s.sync, s.bytes.to_vec())
            };
            let pts_ns = reader.ticks_to_ns(idx, pts_ticks);
            let pt = &self.pad_tracks[idx];
            let reframed = match pt.reframer.reframe_block(&raw) {
                Ok(bytes) => bytes.into_owned(),
                Err(e) => {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("mp4demux: dropping malformed sample on track {idx}: {e:?}"),
                        },
                    });
                    continue; // drop this sample, try the next
                }
            };
            // On the first sample: hand back the announce params + the (taken) codec head.
            let (head, announce) = if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                let head = (!head.is_empty()).then_some(head);
                let pt = &self.pad_tracks[idx];
                (head, Some((pt.family, pt.announce)))
            } else {
                (None, None)
            };
            return Some((idx, pts_ns, sync, reframed, head, announce));
        }
    }

    /// The EOS/stop flush: everything parked or still sliceable goes out with **exact-size**
    /// allocations (no slot to wait for at end of stream, and no slot-sized over-allocation
    /// per small sample). Bounded in volume: the steady-state path only consumed input while
    /// it could emit, so at most one input batch's worth of samples sits here.
    fn flush_exact(&mut self, ctx: &mut Ctx) {
        while let Some(c) = self.pending.pop_front() {
            Self::emit_exact(ctx, c.pad, c.pts_ns, &c.bytes, c.off);
        }
        while let Some((idx, pts_ns, _sync, reframed, head, announce)) = self.take_next(ctx) {
            let pad = self.pad_tracks[idx].pad;
            if let Some((family, ann)) = announce {
                Self::announce(ctx, pad, family, ann);
            }
            if let Some(head) = head {
                Self::emit_exact(ctx, pad, Some(pts_ns), &head, 0);
            }
            Self::emit_exact(ctx, pad, Some(pts_ns), &reframed, 0);
        }
    }

    /// Announce a src pad's runtime format (spec: dynamic caps; follows `MkvDemux`). Video
    /// families carry `width`/`height` when declared; audio carries `rate`/`channels`;
    /// otherwise a bare family announcement. Every family rides a byte bridge, so a
    /// caps-ignoring decoder still links on the family offer.
    fn announce(ctx: &mut Ctx, pad: PadId, family: &'static str, announce: Announce) {
        match announce {
            Announce::Video { width, height } if width != 0 && height != 0 => {
                ctx.announce_format(
                    pad,
                    family,
                    &[("width", ValueDesc::Int(width as i64)), ("height", ValueDesc::Int(height as i64))],
                );
            }
            Announce::Audio { rate, channels } if rate != 0 || channels != 0 => {
                ctx.announce_format(
                    pad,
                    family,
                    &[("rate", ValueDesc::Int(rate as i64)), ("channels", ValueDesc::Int(channels as i64))],
                );
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
            let family = track.family();
            let announce = if codec::is_video_family(family) {
                Announce::Video { width: track.width, height: track.height }
            } else if family == "opus" || track.entry.sample_rate != 0 || track.entry.channels != 0 {
                Announce::Audio { rate: track.entry.sample_rate, channels: track.entry.channels }
            } else {
                Announce::Bare
            };
            let name = format!("src_track{}", track.track_id);
            let pad = ctx.add_pad(Direction::Src, &name, &DEMUX_SRC_OFFERS);
            self.pad_tracks.push(PadTrack {
                pad,
                family,
                codec_head: track.codec_head().to_vec(),
                reframer: track.reframer().clone(),
                announce,
                started: false,
            });
            let _ = i;
        }
        self.reader = Some(reader);
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
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
        // Re-derive the codec heads (taken/consumed on a previous run) from the resolved tracks.
        let heads: Vec<Vec<u8>> = reader.tracks().iter().map(|t| t.codec_head().to_vec()).collect();
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
            // Feed the file bytes to the slicer; it emits samples as their bytes arrive.
            // `buf` recycles on drop at the end of this iteration.
            if let Some(reader) = self.reader.as_mut() {
                reader.push(buf.memory.data());
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

/// Map a resolver error to a worded `streamcraft` error (loud, at preroll).
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
