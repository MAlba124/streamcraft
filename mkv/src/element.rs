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
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

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
