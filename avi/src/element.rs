//! `AviDemux` — the profluens **element** wrapping the hand-written [`crate::riff`]
//! parser: an AVI (RIFF) byte stream in on the `sink` pad, one **dynamic src pad per
//! stream** out (spec: dynamic pads). It is a **passive** byte→packet transform that inlines
//! into the upstream group exactly like `MkvDemux`/`OggDemux` — encoded frames leave stamped
//! with a PTS, backpressure gates input on the pool, and topology settles at `preroll`.
//!
//! ## Discovery is constructor-supplied (the same preroll strategy as `MkvDemux`)
//! A mid-pipeline element gets **no input during `preroll`**, but the scheduler freezes the
//! src-pad set after `preroll` (a pad added mid-`process` would never be wired into a
//! group/ring). So `AviDemux` takes the stream's **header prefix at construction**
//! ([`new`](AviDemux::new)) — enough of the file to cover the `RIFF/AVI` header + `LIST
//! 'hdrl'` (all the `strl` stream headers) — and parses it in `preroll` to learn the streams
//! and add one src pad each. The full stream (header included) then arrives on the sink pad
//! during `process`; the element skips the file bytes up to `movi`'s data and walks the media
//! chunks from there. This mirrors `MkvDemux::new` verbatim (AVI RIFF File Reference for the
//! structure).
//!
//! ## Timing (AVI RIFF Reference — a stream's time base is `dwScale/dwRate`)
//! - **Video** PTS is `frame_index × (dwScale/dwRate)` — the video `strh` time base, one tick
//!   per `movi` video chunk. XviD/DivX is constant-frame-rate, so counting chunks is exact.
//! - **Audio** is typically VBR (AC-3/MP3, `dwSampleSize == 0`): one `movi` chunk is one
//!   frame, and AVI carries no per-chunk timestamp. We stamp a **monotonically increasing
//!   PTS from the accumulated audio time** — for a CBR-per-block stream (`dwSampleSize != 0`)
//!   from the running sample count × `strh` time base; for VBR from the chunk ordinal ×
//!   time base (`dwScale/dwRate` per chunk). Interleave order in `movi` is preserved (the two
//!   tracks arrive alternating, matching the source's A/V interleave).
//!
//! ## Robustness (this brief's P0)
//! The [`crate::riff::MoviWalker`] bounds-checks every chunk and resyncs past corruption; the
//! element additionally drops a chunk whose stream number matches no discovered pad, and a
//! `bytes`-family (undecodable) track is still emitted for a byte peer but a one-shot warning
//! is posted so a missing decoder is diagnosable.

use profluens_core::batch::Inputs;
use profluens_core::buffer::BufferFlags;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;

use crate::codec;
use crate::riff::{self, MoviWalker, Stream, StreamKind};

/// Raw `bytes`: the demux **sink** side is an AVI byte stream.
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static DEMUX_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];

static DEMUX_DESC: ElementDesc = ElementDesc {
    name: "avidemux",
    pads: &DEMUX_PADS,
    props: &[],
    // Passive: a pure byte→packet transform that inlines into the upstream group like
    // `mkvdemux`/`oggdemux`.
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Not name-constructible: it needs the stream header bytes at construction (like
    // `mkvdemux`). Registration still exposes the descriptor for `--list`/help.
    make_default: None,
};

/// A discovered stream wired to its runtime src pad. Built in [`AviDemux::preroll`], read in
/// `process` to route chunks and stamp PTS.
struct PadStream {
    /// The 0-based stream number `movi` chunk ids route by (`NN` in `NNtt`).
    stream_index: usize,
    /// The runtime src pad this stream's chunks go out on ([`Ctx::add_pad`]).
    pad: PadId,
    /// The announce family ([`codec::family_for`]): `mpeg4/asp`, `h264/annexb`, `ac3`, `mp3`,
    /// `audio/raw`, or `bytes` for an unknown codec.
    family: &'static str,
    /// The full stream descriptor (timing + format) for the announcement and PTS math.
    stream: Stream,
    /// Nanoseconds per sample from the `strh` time base, precomputed (`None` → unknown timing,
    /// PTS stays `NONE`).
    sample_dur_ns: Option<u64>,
    /// Running sample/chunk ordinal, incremented per emitted chunk — the PTS multiplier.
    next_sample: u64,
    /// For a CBR audio stream (`dwSampleSize != 0`): accumulated sample count, so the PTS
    /// advances by `size/sample_size` samples per chunk rather than one tick per chunk.
    accum_samples: u64,
    /// False until this pad's format has been announced (once, before its first chunk).
    started: bool,
    /// True for a video stream — only video needs post-seek keyframe gating.
    is_video: bool,
    /// A one-shot "no decoder for this bytes track" warning has been posted.
    warned_undecodable: bool,
}

impl PadStream {
    /// The PTS (ns) for this stream's next chunk of `payload_size` bytes, advancing the
    /// stream's sample clock. AVI RIFF Reference — a stream's time is `sample_index ×
    /// scale/rate`:
    /// - `dwSampleSize == 0` (video, VBR audio): one chunk = one sample tick.
    /// - `dwSampleSize != 0` (CBR audio, e.g. PCM): a chunk holds `size/sample_size` samples,
    ///   so the clock advances by that many.
    fn next_pts_ns(&mut self, payload_size: usize) -> Option<u64> {
        let dur = self.sample_dur_ns?;
        let pts = if self.stream.sample_size != 0 {
            // CBR: sample-count based.
            let t = self.accum_samples.saturating_mul(dur);
            let n = (payload_size as u64) / self.stream.sample_size as u64;
            self.accum_samples = self.accum_samples.saturating_add(n.max(1));
            t
        } else {
            // One tick per chunk (video frame index / VBR audio-frame ordinal).
            let t = self.next_sample.saturating_mul(dur);
            self.next_sample = self.next_sample.saturating_add(1);
            t
        };
        Some(pts)
    }
}

/// A partially-emitted chunk: `bytes[off..]` still needs pool slots on `pad` (the backpressure
/// carry — spec: the demuxer rule, "a demuxer that can't push consumes no input"). Mirrors
/// `MkvDemux::Carry`.
struct Carry {
    pad: PadId,
    pts_ns: Option<u64>,
    flags: BufferFlags,
    bytes: Vec<u8>,
    off: usize,
}

/// Demultiplexes an AVI (RIFF) byte stream into one src pad per stream (spec: dynamic pads).
/// See the module docs for the discovery, timing, and robustness model.
pub struct AviDemux {
    /// The header prefix handed in at construction — parsed in `preroll` to discover streams.
    header: Vec<u8>,
    /// The discovered streams wired to their src pads, filled in `preroll`.
    pad_streams: Vec<PadStream>,
    /// The absolute file offset of `movi`'s data start, learned in `preroll`. The element
    /// skips the incoming byte stream up to here before feeding the walker.
    movi_data_start: u64,
    /// Bytes of the incoming stream skipped so far while seeking to `movi` (a header-prefix
    /// walk on the live stream — the element receives the file from byte 0).
    skipped: u64,
    /// True once the incoming stream position has reached `movi`'s data (or a seek resumed
    /// the source inside `movi`): subsequent bytes go straight to the walker, no skip math.
    in_movi: bool,
    /// The streaming `movi` chunk walker (fresh at `start`, fed from `movi` data onward).
    walker: MoviWalker,
    /// Emissions stalled on pool exhaustion (the backpressure rule — the pool, never the
    /// heap, bounds a demuxer racing a slow decoder). While non-empty the element consumes
    /// **no input**. Holds at most one chunk (split across pool slots).
    pending: std::collections::VecDeque<Carry>,
    /// `DurationChanged` has been posted this run (once, at stream time).
    posted_duration: bool,
    /// The presentation duration (ns) from the header, cached for the one-shot post.
    duration_ns: Option<u64>,
}

impl AviDemux {
    /// A demuxer that discovers its streams from `header` — the leading bytes of the AVI
    /// stream, which MUST include the whole `RIFF/AVI` header + `LIST 'hdrl'` (every `strl`
    /// stream header). The caller reads these once up front (e.g. the first read of a
    /// `filesrc`); the full stream — these bytes and all that follow — is then fed on the
    /// sink pad at run time. A few hundred KiB is ample (real files put `hdrl` first, under
    /// 64 KiB). See the type docs for why discovery is constructor-supplied.
    // COLD: one-time constructor state; `pad_streams` fills once at preroll, not per buffer.
    #[allow(clippy::disallowed_methods)]
    pub fn new(header: Vec<u8>) -> Self {
        Self {
            header,
            pad_streams: Vec::new(),
            movi_data_start: 0,
            skipped: 0,
            in_movi: false,
            walker: MoviWalker::new(),
            pending: std::collections::VecDeque::new(),
            posted_duration: false,
            duration_ns: None,
        }
    }

    /// The streams discovered during `preroll` (for tests / the probe example). Empty before
    /// `preroll` has run.
    pub fn streams(&self) -> Vec<&Stream> {
        self.pad_streams.iter().map(|ps| &ps.stream).collect()
    }

    /// The src pad a given stream number routes to, if that stream was discovered.
    pub fn pad_for(&self, stream_index: usize) -> Option<PadId> {
        self.pad_streams
            .iter()
            .find(|ps| ps.stream_index == stream_index)
            .map(|ps| ps.pad)
    }

    /// Push `bytes[off..]` on `pad` via **pooled** slots (`try_alloc`), chunked to the slot
    /// size, returning the new offset — `< bytes.len()` means the pool ran dry and the caller
    /// must carry the remainder and stop consuming input (spec: the backpressure rule). PTS
    /// and `flags` (the keyframe/delta tag) are stamped on each buffer. Mirrors
    /// `MkvDemux::emit_bounded`.
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
            debug_assert!(cap > 0, "avidemux: zero-capacity pool slot");
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

    /// Emit `bytes[off..]` as one **exact-size** buffer (`alloc_exact`) — only for the
    /// bounded EOS/stop flush, where yielding for a pool slot is no longer possible.
    fn emit_exact(ctx: &mut Ctx, pad: PadId, pts_ns: Option<u64>, flags: BufferFlags, bytes: &[u8], off: usize) {
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

    /// Resume parked emissions. `true` when everything pending has drained.
    fn drain_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some(mut c) = self.pending.pop_front() {
            c.off = Self::emit_bounded(ctx, c.pad, c.pts_ns, c.flags, &c.bytes, c.off);
            if c.off < c.bytes.len() {
                self.pending.push_front(c);
                return false;
            }
        }
        true
    }

    /// Drain media chunks the walker has parsed, routing each to its stream's src pad —
    /// **pool bounded**: returns `false` when slots ran out (remainder parked; stop consuming
    /// input until it clears). On a stream's first chunk the pad's runtime format is announced
    /// (spec: dynamic caps). A chunk for an undiscovered stream is dropped.
    fn drain(&mut self, ctx: &mut Ctx) -> bool {
        if !self.drain_pending(ctx) {
            return false;
        }
        while let Some(chunk) = self.walker.next_chunk() {
            let Some(idx) = self
                .pad_streams
                .iter()
                .position(|ps| ps.stream_index == chunk.stream_index)
            else {
                continue; // no pad for this stream number — drop
            };
            // First chunk on this pad: announce the runtime format.
            if !self.pad_streams[idx].started {
                self.pad_streams[idx].started = true;
                let (pad, family) = (self.pad_streams[idx].pad, self.pad_streams[idx].family);
                Self::announce(ctx, pad, family, &self.pad_streams[idx].stream);
                // Warn once if this track has no decoder family (still emitted for a byte
                // peer, but a downstream autoplug will drop it — this makes that diagnosable).
                if !codec::is_decodable(family) && !self.pad_streams[idx].warned_undecodable {
                    self.pad_streams[idx].warned_undecodable = true;
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!(
                                "avidemux: stream {} has no decoder family (fourcc/tag \
                                 unrecognised) — emitting raw bytes",
                                chunk.stream_index
                            ),
                        },
                    });
                }
            }
            let ps = &mut self.pad_streams[idx];
            let pts = ps.next_pts_ns(chunk.data.len());
            // AVI keyframe info is only in `idx1` (not in the chunk stream), so a streaming
            // walk has none per chunk. Tag video as DELTA-unknown by default? No: an
            // MPEG-4 ASP decoder resyncs at its own VOP headers, and a container-side
            // KEYFRAME hint is only load-bearing for a *remuxer*. We leave flags empty
            // (== "unknown"); a downstream muxer defaults unknown to keyframe, and the
            // decoder ignores it. Audio frames are all independently decodable anyway.
            let pad = ps.pad;
            let off = Self::emit_bounded(ctx, pad, pts, BufferFlags::empty(), &chunk.data, 0);
            if off < chunk.data.len() {
                self.pending.push_back(Carry {
                    pad,
                    pts_ns: pts,
                    flags: BufferFlags::empty(),
                    bytes: chunk.data,
                    off,
                });
                return false; // pool dry — stop pulling walker chunks
            }
        }
        true
    }

    /// The EOS/stop flush: everything parked or still queued in the walker goes out with
    /// exact-size allocations (no slot to wait for at end of stream). Bounded in volume — the
    /// steady state only consumed input while it could emit.
    fn flush_exact(&mut self, ctx: &mut Ctx) {
        while let Some(c) = self.pending.pop_front() {
            Self::emit_exact(ctx, c.pad, c.pts_ns, c.flags, &c.bytes, c.off);
        }
        while let Some(chunk) = self.walker.next_chunk() {
            let Some(idx) = self
                .pad_streams
                .iter()
                .position(|ps| ps.stream_index == chunk.stream_index)
            else {
                continue;
            };
            if !self.pad_streams[idx].started {
                self.pad_streams[idx].started = true;
                let (pad, family) = (self.pad_streams[idx].pad, self.pad_streams[idx].family);
                Self::announce(ctx, pad, family, &self.pad_streams[idx].stream);
            }
            let ps = &mut self.pad_streams[idx];
            let pts = ps.next_pts_ns(chunk.data.len());
            let pad = ps.pad;
            Self::emit_exact(ctx, pad, pts, BufferFlags::empty(), &chunk.data, 0);
        }
    }

    /// Announce a src pad's runtime format (spec: dynamic caps; follows `MkvDemux::announce`).
    /// The concrete params ride the announcement so a caps-aware consumer/decoder sees the
    /// format:
    /// - `ac3`/`mp3` → nothing extra required (the decoder reads its config from the
    ///   bitstream); we still carry container `rate`/`channels` to seed negotiation.
    /// - `audio/raw` → `rate`/`channels`/`sample` so a raw sink links directly.
    /// - a **video** family → `width`/`height` from the BITMAPINFOHEADER (the decoder
    ///   re-announces authoritative dims from the bitstream; this seeds negotiation).
    ///
    /// Every family rides a byte bridge, so a caps-ignoring decoder still links on the
    /// `bytes`/family offer.
    fn announce(ctx: &mut Ctx, pad: PadId, family: &'static str, stream: &Stream) {
        log!(
            &*ctx,
            Level::Debug,
            "announce",
            family = family,
            width = stream.width,
            height = stream.height,
        );
        match family {
            codec::FAMILY_AUDIO_RAW => {
                ctx.announce_format(
                    pad,
                    family,
                    &[
                        ("rate", ValueDesc::Int(stream.samples_per_sec as i64)),
                        ("channels", ValueDesc::Int(stream.channels as i64)),
                        ("sample", ValueDesc::Id(sample_name(stream.bits_per_sample))),
                    ],
                );
            }
            codec::FAMILY_AC3 | codec::FAMILY_MP3 if stream.samples_per_sec > 0 => {
                // Container-declared rate/channels seed negotiation (a muxer reads them); the
                // decoder reads its authoritative config from the frame headers.
                ctx.announce_format(
                    pad,
                    family,
                    &[
                        ("rate", ValueDesc::Int(stream.samples_per_sec as i64)),
                        ("channels", ValueDesc::Int(stream.channels as i64)),
                    ],
                );
            }
            _ if codec::is_video_family(family) && stream.width != 0 && stream.height != 0 => {
                ctx.announce_format(
                    pad,
                    family,
                    &[
                        ("width", ValueDesc::Int(stream.width as i64)),
                        ("height", ValueDesc::Int(stream.height as i64)),
                    ],
                );
            }
            _ => ctx.announce_format(pad, family, &[]),
        }
    }

    /// Feed incoming stream bytes to the walker, first skipping any that precede `movi`'s
    /// data (the file arrives from byte 0; the walker wants only `movi` chunks). Handles the
    /// skip straddling a buffer boundary. Once `in_movi` is set (all header bytes consumed,
    /// or a seek has resumed the source at a byte already inside `movi`), bytes pass straight
    /// through.
    fn feed(&mut self, data: &[u8]) {
        if self.in_movi {
            self.walker.push(data);
            return;
        }
        let end = self.skipped.saturating_add(data.len() as u64);
        if end <= self.movi_data_start {
            // Entirely before movi — skip it all.
            self.skipped = end;
            return;
        }
        // This buffer straddles (or starts past) the movi boundary — feed the tail.
        let cut = (self.movi_data_start.saturating_sub(self.skipped)) as usize;
        let cut = cut.min(data.len());
        self.walker.push(&data[cut..]);
        self.skipped = end;
        self.in_movi = true;
    }
}

/// Map a bit depth to the `sample` categorical name (spec: dynamic caps). An unusual/zero
/// depth falls back to `s16` — the most common PCM depth — so the announcement is always
/// well-formed (the byte-bridge peer ignores it anyway).
fn sample_name(bits: u16) -> &'static str {
    match bits {
        8 => "u8",
        24 => "s24",
        32 => "s32",
        _ => "s16",
    }
}

impl Element for AviDemux {
    fn desc(&self) -> &'static ElementDesc {
        &DEMUX_DESC
    }

    // COLD: preroll runs once per stream (topology freezes after it); the error strings are
    // built only on a malformed header, never per buffer.
    #[allow(clippy::disallowed_methods)]
    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Discover streams from the constructor-supplied header, then add one src pad per
        // stream (spec: dynamic pads — topology settles at preroll).
        let (header, movi) = riff::probe_header(&self.header)
            .map_err(|e| Error::Resource(format!("avidemux: header parse failed: {e:?}")))?;
        // The header prefix must at least reach `movi`, so we know where media begins.
        let movi = movi.ok_or_else(|| {
            Error::Resource(
                "avidemux: constructor header does not reach the 'movi' list — pass more of \
                 the stream head (through hdrl and the movi list header)"
                    .to_string(),
            )
        })?;
        self.movi_data_start = movi.movi_data_start;
        self.duration_ns = header.duration_ns();

        for stream in &header.streams {
            let family = codec::family_for(stream);
            let name = format!("src_stream{}", stream.index);
            // Per-stream offer menu (see `codec::offers_for`): the pad admits exactly its
            // codec's family (+ the bytes escape), so link-time negotiation *selects* the
            // right decoder instead of admitting them all.
            let pad = ctx.add_pad(Direction::Src, &name, codec::offers_for(stream));
            log!(
                &*ctx,
                Level::Debug,
                "stream",
                index = stream.index,
                family = family,
                width = stream.width,
                height = stream.height,
            );
            self.pad_streams.push(PadStream {
                stream_index: stream.index,
                pad,
                family,
                sample_dur_ns: stream.sample_duration_ns(),
                next_sample: 0,
                accum_samples: 0,
                started: false,
                is_video: matches!(stream.kind, StreamKind::Video),
                warned_undecodable: false,
                stream: stream.clone(),
            });
        }
        if self.pad_streams.is_empty() {
            return Err(Error::Resource("avidemux: no streams discovered in hdrl".to_string()));
        }
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // Fresh streaming walker; the whole file (header + movi) arrives on the sink pad from
        // byte 0, so reset the skip counter and per-stream clocks.
        self.walker = MoviWalker::new();
        self.skipped = 0;
        self.in_movi = false;
        self.pending.clear();
        self.posted_duration = false;
        for ps in &mut self.pad_streams {
            ps.started = false;
            ps.next_sample = 0;
            ps.accum_samples = 0;
        }
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Backpressure: resume any parked emission first, and consume input only while
        // emission keeps up (un-popped input stays buffered for the next pass, so the
        // scheduler's gates — not this element's memory — absorb a fast source racing a slow
        // decoder). Mirrors `MkvDemux::process`.
        if !self.drain(ctx) {
            return Ok(Flow::Ok);
        }
        // Post the presentation duration once (a transport UI acts on it; it can't ride
        // negotiated formats — decoders don't declare the `duration` field).
        if !self.posted_duration {
            if let Some(ns) = self.duration_ns {
                self.posted_duration = true;
                let element = ctx.element();
                ctx.post(BusMessage::DurationChanged { element, ns });
            }
        }
        while let Some(buf) = inputs.pop() {
            self.feed(buf.memory.data());
            if !self.drain(ctx) {
                return Ok(Flow::Ok);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // On EOS, flush everything still parked or queued — exact-size allocations, since
            // waiting for pool slots is no longer an option. AVI has no trailer to emit.
            Event::Eos => self.flush_exact(ctx),
            // Seek (spec: flush/seek). The byte source has already resumed reads at the seek
            // target byte (an `idx1`-resolved chunk offset or a proportional estimate); here
            // the demuxer resets its walk to match. We do NOT rebuild `pad_streams` (that
            // would forget the discovered streams). The skip counter is set so the walker is
            // fed from the resumed byte onward, and the walker resyncs by scanning for the
            // next chunk id (tolerating a mid-chunk landing). PTS recover from the per-stream
            // sample ordinal — approximate after a proportional seek, exact after an
            // idx1-indexed one (the app seeks to a keyframe chunk boundary).
            Event::FlushStart => {
                self.walker.resync();
                self.pending.clear();
                // The source resumes at the seek byte (already inside movi); pass the resumed
                // bytes straight to the walker, which resyncs to the next chunk id.
                self.in_movi = true;
                for ps in &mut self.pad_streams {
                    ps.started = false;
                    // Reset the sample clock so post-seek PTS re-establish from the resumed
                    // position. (Exact reconstruction of the pre-seek sample ordinal would
                    // need the idx1 chunk index; the app-side seek already rebased running
                    // time to the landed time, so a monotonic-from-zero restart is fine — the
                    // sink paces on the rebased clock.)
                    ps.next_sample = 0;
                    ps.accum_samples = 0;
                    let _ = ps.is_video; // gating is a decoder concern for MPEG-4 ASP
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: flush any final chunk even if `event(Eos)` was not delivered.
        self.flush_exact(ctx);
        self.walker = MoviWalker::new();
        self.pending.clear();
        self.posted_duration = false;
    }
}
