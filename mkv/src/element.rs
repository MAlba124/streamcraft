//! `MkvMux` — the streamcraft **element** wrapping the tested [`MatroskaWriter`]
//! (crate::MatroskaWriter) (spec: `spec/MATROSKA.md`; Writing elements). This is the
//! **single-track** version: one sink pad, one src pad, fitting today's static pad model
//! (spec: Elements and pads). The general multiplexer wants one *dynamic* sink pad per
//! input stream plus fan-in — deferred until the core grows dynamic sink pads (the writer
//! is already N-track ready; see the crate docs, "Not yet"). Everything here is a
//! **passive transform**: encoded frames in, MKV bytes out, inlining into the upstream
//! group like `sc-ogg`'s `OggMux`.
//!
//! Both pads carry raw [`bytes`](streamcraft_core::format::OfferDesc::any): encoded codec
//! frames on the sink side, a Matroska byte stream on the src side — matching `OggMux` so
//! an MKV muxer links to a codec/de-framer element at link time without a typed vocabulary.
//!
//! ## Codec-init data (out-of-band, for now)
//! The `CodecPrivate` (FLAC `fLaC` + STREAMINFO) and the audio params are supplied at
//! **construction** ([`MkvMux::flac`]), because format negotiation of codec-init blobs is
//! still being built. The clean future path is to carry the init data through the
//! **negotiated caps** — a `FixedFormat` config blob the upstream codec/parse element
//! announces (spec: Formats — dynamic caps; the same mechanism `flacdec` uses to announce
//! rate/channels). The element would then read `ctx.negotiated(sink)` in `start()` and
//! build the header from it, needing no constructor arguments. Until that lands, the
//! constructor arguments are the pragmatic bridge (exactly how `sc-flac`'s encoder took
//! audio params before negotiation existed).
//!
//! ## The header, and one frame per input buffer
//! `start()` (re)creates the writer; the **header is emitted lazily on the first frame**
//! (so a stream with no frames produces no partial file, and the writer's track config is
//! fixed before any block). Each input buffer is one encoded frame → one `SimpleBlock`
//! (spec `§simpleblock`); its bytes are pushed downstream chunked to the pool slot, exactly
//! like `OggMux`. Timestamps come from the buffer PTS (falling back to a synthesised cadence
//! when absent — a container needs *a* timeline, and a byte pipeline carries no PTS yet).
//! Every FLAC frame is independently decodable, so every block is flagged a keyframe unless
//! the buffer clears [`BufferFlags::KEYFRAME`].
//!
//! ## EOS / finalize (spec: Events — Eos)
//! The Segment and final Cluster are unknown-size streamed masters closed implicitly at end
//! of stream (spec `§sizing`), so there is no tail to flush — [`MatroskaWriter::finalize`]
//! is seek-free. The element still runs it on [`Event::Eos`] (primary) and in [`stop`]
//! (belt-and-braces, idempotent) for symmetry with `OggMux` and so a future trailing
//! element (Cues) has a hook. Any bytes `finalize` were to produce are pushed downstream.

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::BufferFlags;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{OfferDesc, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::reader::{MatroskaReader, Track};
use crate::writer::{MatroskaWriter, TrackConfig};

/// The one track this single-sink-pad element muxes (spec: single-track element). The
/// writer is N-track, but the element feeds it exactly this track.
const TRACK: u64 = 1;

// The src pad's local index (== position in the element's `pads` array). The sink pad
// (index 0) needs no id here: input is drained via `Inputs::pop`, which is pad-agnostic
// (matches `OggMux`).
const SRC: PadId = PadId(1);

/// Raw `bytes` on both pads: encoded frames on the sink side, an MKV byte stream on the
/// src side. Matches `OggMux`/`flacenc` byte pads so an MKV element links to a codec
/// element at link time without a typed vocabulary (the container is codec-agnostic).
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static MUX_PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        // A single static sink pad: this element muxes one track. One dynamic sink pad per
        // input track is the multi-track follow-up (the writer already supports N tracks).
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
    name: "mkvmux",
    pads: &MUX_PADS,
    props: &[],
    // Passive: a pure frame→byte transform, inlines into the upstream group.
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

/// Muxes a single track into Matroska: one encoded frame **per input buffer** on the sink
/// pad, an MKV byte stream on the src pad. The header is emitted before the first frame;
/// each frame becomes a `SimpleBlock` in a Cluster (spec `§simpleblock`).
///
/// The track's `CodecID`, `CodecPrivate` and audio params are given at construction (see
/// the module docs on codec-init data). Handling several input tracks into one MKV needs
/// one dynamic sink pad per input plus fan-in — a documented follow-up.
pub struct MkvMux {
    /// The track config the writer is built from. Cloned into a fresh writer at `start()`.
    track: TrackConfig,
    /// The writer. Present between `start` and finalize; `None` before start and after the
    /// stream is finished (so a second finalize is a no-op).
    writer: Option<MatroskaWriter>,
    /// Whether the header has been written (lazily, before the first frame).
    header_done: bool,
    /// Synthesised frame timeline: nanoseconds of the next frame when a buffer carries no
    /// PTS. A byte pipeline has no PTS yet; the container still needs a monotonic timeline,
    /// so successive frames are spaced by [`frame_duration_ns`](Self::frame_duration_ns).
    next_ts_ns: u64,
    /// The synthesised inter-frame spacing in ns, computed once from the track's sample
    /// rate (cached so the hot `process` path does not re-derive it, and to avoid a second
    /// borrow of `self` while the writer is borrowed).
    frame_dur_ns: u64,
    /// Reused byte buffer the writer appends into, so steady-state muxing does not
    /// reallocate a fresh `Vec` per `process` (mirrors `OggMux::scratch`).
    scratch: Vec<u8>,
}

impl MkvMux {
    /// An MKV muxer for one `A_FLAC` track from its native FLAC head (`fLaC` + STREAMINFO)
    /// and audio params (spec: A_FLAC mapping). The frames pushed on the sink pad must be
    /// native FLAC frames matching this STREAMINFO.
    pub fn flac(codec_private: Vec<u8>, sampling_frequency: f64, channels: u32, bit_depth: u32) -> Self {
        Self::with_track(TrackConfig::flac(TRACK, codec_private, sampling_frequency, channels, bit_depth))
    }

    /// An MKV muxer for an explicit single-track [`TrackConfig`]. The track number is
    /// forced to [`TRACK`] (this element muxes one track); other fields are used as given,
    /// so any frame-per-block codec works by setting `codec_id` / `codec_private`.
    pub fn with_track(mut track: TrackConfig) -> Self {
        track.track_number = TRACK;
        let frame_dur_ns = Self::frame_duration_ns(&track);
        Self {
            track,
            writer: None,
            header_done: false,
            next_ts_ns: 0,
            frame_dur_ns,
            scratch: Vec::new(),
        }
    }

    /// This element's track config (for tests / introspection).
    pub fn track(&self) -> &TrackConfig {
        &self.track
    }

    /// Nanoseconds between synthesised frame timestamps when buffers carry no PTS: one FLAC
    /// block (4096 interchannel samples — the encoder's block size) at the track's sample
    /// rate. A defined, monotonic cadence so the muxed timestamps advance sensibly; a real
    /// upstream that stamps PTS overrides this entirely.
    fn frame_duration_ns(track: &TrackConfig) -> u64 {
        const FLAC_BLOCK: u64 = 4096;
        let rate = track.audio.sampling_frequency;
        if rate > 0.0 {
            ((FLAC_BLOCK as f64) * 1_000_000_000.0 / rate) as u64
        } else {
            0
        }
    }

    /// Push accumulated MKV bytes onto the src pad, chunked to the pool slot size so no
    /// single copy exceeds a buffer. Clears `bytes` afterwards, retaining its capacity.
    /// Identical strategy to `OggMux::emit`.
    fn emit(ctx: &mut Ctx, bytes: &mut Vec<u8>) {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(SRC);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "mkvmux: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            ctx.out(SRC).push(buf);
            off += n;
        }
        bytes.clear();
    }

    /// Finalize the stream exactly once (spec `§sizing` — seek-free). Idempotent: the
    /// writer is taken on first call, so later calls (the `event` path then `stop`, or vice
    /// versa) do nothing. Any bytes finalize produces are pushed downstream.
    fn finish_stream(&mut self, ctx: &mut Ctx) {
        if let Some(mut writer) = self.writer.take() {
            let mut out = std::mem::take(&mut self.scratch);
            writer.finalize(&mut out);
            Self::emit(ctx, &mut out);
            self.scratch = out; // reclaim the allocation
        }
    }
}

impl Element for MkvMux {
    fn desc(&self) -> &'static ElementDesc {
        &MUX_DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.writer = Some(MatroskaWriter::new(vec![self.track.clone()]));
        self.header_done = false;
        self.next_ts_ns = 0;
        self.scratch.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Stop early if the stream is already finalized: a late buffer after EOS is
            // dropped rather than reopening a closed stream (matches `OggMux`).
            if self.writer.is_none() {
                break;
            }

            // Timestamp: prefer the buffer PTS; otherwise the synthesised cadence. Computed
            // before borrowing the writer (it mutates `self.next_ts_ns`). A frame is a
            // keyframe unless the buffer explicitly clears the flag — every FLAC frame is
            // independently decodable, so keyframe is the correct default.
            let ts_ns = match buf.pts.nanos() {
                None => {
                    let t = self.next_ts_ns;
                    self.next_ts_ns += self.frame_dur_ns;
                    t
                }
                Some(t) => {
                    // Keep the synth clock ahead of any real PTS, so a later PTS-less frame
                    // does not travel backwards.
                    self.next_ts_ns = t + self.frame_dur_ns;
                    t
                }
            };
            // Empty flags means "not tagged" → default to keyframe (FLAC frames all are).
            let keyframe = buf.flags.is_empty() || buf.flags.contains(BufferFlags::KEYFRAME);

            let mut out = std::mem::take(&mut self.scratch);
            let writer = self.writer.as_mut().expect("writer present");

            // Emit the header lazily, before the first frame — so an empty stream writes
            // nothing, and the track config is fixed before any block. The only error path
            // (BadTracks) cannot occur for our single fixed, validated track.
            if !self.header_done {
                let _ = writer.write_header(&mut out);
                self.header_done = true;
            }

            // One input buffer == one encoded frame == one SimpleBlock. The only error is
            // an unknown track, impossible for our single fixed track.
            let _ = writer.write_frame(&mut out, TRACK, ts_ns, buf.memory.data(), keyframe);

            // `buf` recycles here on drop; move the produced bytes downstream.
            Self::emit(ctx, &mut out);
            self.scratch = out;
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // Primary finalize path: the pipeline delivers EOS in-band and pushes whatever this
        // produces before closing the ring.
        if matches!(event, Event::Eos) {
            self.finish_stream(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: guarantee finalize even if `event(Eos)` was not delivered.
        // Idempotent via the `Option` take.
        self.finish_stream(ctx);
        self.scratch = Vec::new();
    }
}

// =====================================================================================
// MkvDemux — the container demultiplexer (spec: `spec/MATROSKA.md`; dynamic pads)
// =====================================================================================

/// The `A_FLAC` CodecID string (spec: A_FLAC mapping; RFC 9559 §12). Frames on such a track
/// are native FLAC frames; the CodecPrivate is the native FLAC head (`fLaC` + STREAMINFO), so
/// the reconstructed byte stream a downstream `flacdec` reads is CodecPrivate followed by
/// every frame — the same reconstruction `oggflacdeframe` does for the Ogg mapping.
const CODEC_A_FLAC: &str = "A_FLAC";

/// The `flac` announce family for an A_FLAC track's src pad (spec: dynamic caps). It rides a
/// `bytes` bridge to a downstream `flacdec` (whose sink offers `bytes`): the demux announces
/// the family so a caps-aware consumer or an auto-plugger can pick a FLAC decoder, while a
/// caps-ignoring byte peer just reads the reconstructed native FLAC stream (spec: Formats —
/// dynamic caps; the same tolerant install `flacdec`'s `audio/raw`→`bytes` uses).
const FAMILY_FLAC: &str = "flac";
/// The fallback announce family for an unknown/other CodecID: a raw `bytes` stream (spec:
/// dynamic caps — "bytes fallback for unknown codec ids").
const FAMILY_BYTES: &str = "bytes";

/// Both announce families a demux src pad may carry, so the pad's offer menu contains the
/// family it will announce (a pad can only announce a family it already offered — the
/// vocabulary is interned from these offers at link time). `flac` for A_FLAC, `bytes` for
/// everything else; both are unconstrained byte families.
static DEMUX_SRC_OFFERS: [OfferDesc; 2] = [OfferDesc::any(FAMILY_FLAC), OfferDesc::any(FAMILY_BYTES)];

static DEMUX_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DEMUX_DESC: ElementDesc = ElementDesc {
    name: "mkvdemux",
    pads: &DEMUX_PADS,
    props: &[],
    // Passive: a pure byte→frame transform, inlines into the upstream group like `oggdemux`.
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

/// A discovered track wired to its runtime src pad. Built during [`MkvDemux::preroll`], read
/// during `process` to route frames and announce the pad format.
struct PadTrack {
    /// The `TrackNumber` frames carry to route here.
    track_number: u64,
    /// The runtime src pad this track's frames go out on ([`Ctx::add_pad`]).
    pad: PadId,
    /// The announce family for the pad: `flac` for A_FLAC, else `bytes`.
    family: &'static str,
    /// The native codec head to emit *before* the first frame — for A_FLAC this is the
    /// CodecPrivate (`fLaC` + STREAMINFO), reconstructing a native FLAC byte stream a
    /// downstream `flacdec` decodes (spec: A_FLAC mapping). Empty for a raw `bytes` track.
    codec_head: Vec<u8>,
    /// The audio params, for a `flac`-family announcement (rate/channels/sample) mirroring
    /// what `flacdec` would announce from STREAMINFO.
    track: Track,
    /// False until the codec head + the format announcement have been emitted on this pad.
    started: bool,
}

/// Demultiplexes a Matroska/WebM byte stream into one src pad per track (spec: dynamic pads).
/// A Matroska byte stream arrives on the sink pad; each track's frames leave on their own
/// runtime src pad, reconstructed so a downstream decoder can start (for `A_FLAC`: the native
/// `fLaC` + STREAMINFO head from CodecPrivate, then the frames — like `oggflacdeframe`).
///
/// ## Discovery is constructor-supplied (the chosen preroll strategy)
/// Track discovery — and therefore the src-pad set — must be settled in
/// [`preroll`](Element::preroll), because the scheduler freezes the topology after preroll:
/// a pad added mid-`process` would never be wired into a group/ring. But a *mid-pipeline*
/// element gets **no input during preroll** (`Pipeline::preroll` instantiates pads without
/// streaming any buffers). So `MkvDemux` takes the stream's **header bytes at construction**
/// ([`new`](Self::new)) — enough of the file to cover the EBML Header + Info + Tracks — and
/// parses them in `preroll` to learn the tracks and add one src pad each. This is the
/// documented accepted fallback (the crate's `MkvMux::flac` likewise takes CodecPrivate at
/// construction while negotiation of codec-init blobs is built), and it composes with today's
/// scheduler exactly. The full stream (header included) then arrives on the sink pad during
/// `process`, where a fresh streaming reader parses it from byte 0 and routes the frames.
pub struct MkvDemux {
    /// The header bytes handed in at construction — parsed in `preroll` to discover tracks.
    header: Vec<u8>,
    /// The discovered tracks wired to their src pads, filled in `preroll`.
    pad_tracks: Vec<PadTrack>,
    /// The streaming reader that parses the full byte stream during `process` (fresh at
    /// `start`, fed from byte 0).
    reader: MatroskaReader,
    /// Reused byte buffer for chunked emission (mirrors `MkvMux::scratch`).
    scratch: Vec<u8>,
}

impl MkvDemux {
    /// A demuxer that discovers its tracks from `header` — the leading bytes of the Matroska
    /// stream, which MUST include the whole EBML Header + Segment Info + Tracks (everything up
    /// to the first Cluster). The caller reads these once up front (e.g. the first read of a
    /// `filesrc`); the full stream — these bytes and all that follow — is then fed on the
    /// sink pad at run time. See the type docs for why discovery is constructor-supplied.
    pub fn new(header: Vec<u8>) -> Self {
        Self {
            header,
            pad_tracks: Vec::new(),
            reader: MatroskaReader::new(),
            scratch: Vec::new(),
        }
    }

    /// The tracks discovered during `preroll` (for tests / introspection). Empty before
    /// `preroll` has run.
    pub fn tracks(&self) -> Vec<&Track> {
        self.pad_tracks.iter().map(|pt| &pt.track).collect()
    }

    /// The src pad a given track number routes to, if that track was discovered.
    pub fn pad_for(&self, track_number: u64) -> Option<PadId> {
        self.pad_tracks
            .iter()
            .find(|pt| pt.track_number == track_number)
            .map(|pt| pt.pad)
    }

    /// Reconstruct the codec head to emit before a track's first frame (spec: A_FLAC
    /// mapping). For A_FLAC the CodecPrivate *is* the native FLAC head (`fLaC` + STREAMINFO),
    /// so it is forwarded verbatim; a downstream `flacdec` then sees exactly a native `.flac`
    /// stream. Any other codec has no reconstruction — its frames are raw `bytes`.
    fn codec_head_for(track: &Track) -> Vec<u8> {
        if track.codec_id == CODEC_A_FLAC {
            track.codec_private.clone()
        } else {
            Vec::new()
        }
    }

    /// Push `bytes` on `pad`, chunked to the pool slot so no single copy exceeds a buffer, and
    /// stamp each buffer's PTS with `pts_ns` (identical chunking to `MkvMux::emit`, plus the
    /// timestamp a demuxer must carry). Empty `bytes` push nothing (a byte stream has no
    /// boundary to preserve — the native FLAC head/frames are just concatenated downstream).
    fn emit(ctx: &mut Ctx, pad: PadId, pts_ns: Option<u64>, bytes: &[u8]) {
        let mut off = 0;
        while off < bytes.len() {
            let mut buf = ctx.alloc(pad);
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "mkvdemux: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            if let Some(t) = pts_ns {
                buf.pts = Timestamp::from_nanos(t);
            }
            ctx.out(pad).push(buf);
            off += n;
        }
    }

    /// Drain every frame the reader has decoded so far, emitting each on its track's src pad.
    /// On a track's first frame the native codec head is emitted first and the pad's runtime
    /// format is announced (spec: dynamic caps). A frame for a track that was not discovered
    /// (absent from the header the demux was constructed with) is dropped — its pad does not
    /// exist, so there is nowhere to route it.
    fn drain(&mut self, ctx: &mut Ctx) {
        while let Some(frame) = self.reader.next_frame() {
            let Some(idx) = self.pad_tracks.iter().position(|pt| pt.track_number == frame.track_number) else {
                continue; // undiscovered track — no pad to route to
            };
            // First frame on this pad: emit the native codec head and announce the format.
            if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let (pad, family) = (self.pad_tracks[idx].pad, self.pad_tracks[idx].family);
                Self::announce(ctx, pad, family, &self.pad_tracks[idx].track);
                // The head carries the stream start; stamp it with the first frame's pts. Take
                // it (leaving empty) — it is emitted exactly once, before the first frame.
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                Self::emit(ctx, pad, Some(frame.pts_ns), &head);
            }
            let pad = self.pad_tracks[idx].pad;
            Self::emit(ctx, pad, Some(frame.pts_ns), &frame.data);
        }
    }

    /// Announce a src pad's runtime format (spec: dynamic caps; follows `OggDemux`/`flacdec`).
    /// For a `flac` track the audio params (rate/channels/sample) ride the announcement so a
    /// caps-aware consumer sees the concrete format; for a `bytes` track it is a bare family
    /// announcement. Either rides a `bytes` bridge to a byte-reading peer (`flacdec`).
    fn announce(ctx: &mut Ctx, pad: PadId, family: &'static str, track: &Track) {
        if family == FAMILY_FLAC {
            ctx.announce_format(
                pad,
                FAMILY_FLAC,
                &[
                    ("rate", ValueDesc::Int(track.sampling_frequency as i64)),
                    ("channels", ValueDesc::Int(track.channels as i64)),
                    ("sample", ValueDesc::Id(sample_name(track.bit_depth))),
                ],
            );
        } else {
            ctx.announce_format(pad, FAMILY_BYTES, &[]);
        }
    }
}

/// Map a bit depth to the `sample` categorical name `flacdec` uses (spec: dynamic caps). An
/// unusual/zero depth falls back to `s16` — the most common audio depth — so the announcement
/// is always well-formed (the byte-bridge peer ignores it anyway).
fn sample_name(bit_depth: u32) -> &'static str {
    match bit_depth {
        8 => "s8",
        24 => "s24",
        32 => "s32",
        _ => "s16",
    }
}

impl Element for MkvDemux {
    fn desc(&self) -> &'static ElementDesc {
        &DEMUX_DESC
    }

    fn preroll(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Discover tracks from the constructor-supplied header, then add one src pad per
        // track (spec: dynamic pads — topology settles at preroll). A throwaway reader: the
        // real streaming reader is created fresh in `start`, fed the whole stream from byte 0.
        let mut probe = MatroskaReader::new();
        probe
            .push(&self.header)
            .map_err(|e| Error::Resource(format!("mkvdemux: header parse failed: {e:?}")))?;
        if !probe.tracks_ready() {
            return Err(Error::Resource(
                "mkvdemux: constructor header does not contain a complete Tracks element — \
                 pass more of the stream head (through the first Cluster)"
                    .to_string(),
            ));
        }
        for track in probe.tracks() {
            let family = if track.codec_id == CODEC_A_FLAC { FAMILY_FLAC } else { FAMILY_BYTES };
            let name = format!("src_track{}", track.track_number);
            let pad = ctx.add_pad(Direction::Src, &name, &DEMUX_SRC_OFFERS);
            self.pad_tracks.push(PadTrack {
                track_number: track.track_number,
                pad,
                family,
                codec_head: Self::codec_head_for(track),
                track: track.clone(),
                started: false,
            });
        }
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // Fresh streaming reader; the whole stream (header + clusters) arrives on the sink pad.
        self.reader = MatroskaReader::new();
        for pt in &mut self.pad_tracks {
            pt.started = false;
            pt.codec_head = Self::codec_head_for(&pt.track);
        }
        self.scratch.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            // Feed the bytes to the structure reader; it parses whole elements and queues the
            // frames it decodes. A malformed stream errors here (never panics). `buf` recycles
            // on drop at the end of this iteration.
            self.reader
                .push(buf.memory.data())
                .map_err(|e| Error::Resource(format!("mkvdemux: parse error: {e:?}")))?;
            self.drain(ctx);
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // On EOS, drain any frame a just-completed element produced. The Matroska streamed
        // masters (Segment/Cluster) end implicitly, so there is no trailer to flush — a clean
        // stream ends on an element boundary with nothing buffered.
        if matches!(event, Event::Eos) {
            self.drain(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: drain any final frame even if `event(Eos)` was not delivered.
        self.drain(ctx);
        self.reader = MatroskaReader::new();
        self.scratch = Vec::new();
    }
}
