//! `MkvMux` — the streamcraft **element** wrapping the tested [`MatroskaWriter`]
//! (crate::MatroskaWriter) (spec: `spec/MATROSKA.md`; Writing elements). This is the
//! **single-track** version: one sink pad, one src pad, fitting today's static pad model
//! (spec: Elements and pads). The general multiplexer — one sink pad per input stream
//! plus fan-in — is [`MkvMuxN`](crate::MkvMuxN) (`mux_multi`). Everything here is a
//! **passive transform**: encoded frames in, MKV bytes out, inlining into the upstream
//! group like `sc-ogg`'s `OggMux`.
//!
//! The **sink pad offers every family it can mux** (plus a `bytes` fallback for the
//! constructed path) with the field/value names those announcements carry — a consumer
//! that *reads* announced fields must declare them, since a pad's offers are what interns
//! the names the upstream's announcement resolves against (spec: Formats — dynamic caps;
//! a wildcard pad would admit everything but intern nothing). The src pad stays raw
//! [`bytes`](streamcraft_core::format::OfferDesc::any) — a Matroska byte stream.
//!
//! ## Track config: negotiated (remux) or constructed
//! Two ways to know the track:
//! - **[`MkvMux::from_caps`]** (the registry default): configured by the upstream runtime
//!   announcement (spec: dynamic caps) — `vp8`/`vp9` → a `V_VP8`/`V_VP9` video track with
//!   the announced `width`/`height`; `flac` → an `A_FLAC` track whose `CodecPrivate` *is*
//!   the native head (`fLaC` + metadata blocks, RFC 9559 A_FLAC mapping) absorbed from the
//!   leading in-band bytes — exactly what `MkvDemux`/`flacenc` emit first — with the audio
//!   params parsed out of STREAMINFO (RFC 9639 §8.2). This is what makes
//!   `mkvdemux ! mkvmux` a working remux. Families we cannot yet mux conformantly are a
//!   loud error at the announcement: `av1` (needs an `av1C` CodecPrivate extracted from
//!   the Sequence Header OBU) and `h264/h265` (arrive as Annex B, but Matroska wants
//!   length-prefixed NALs + a config record) — both documented follow-ups.
//! - **[`MkvMux::flac`] / [`MkvMux::with_track`]**: explicit construction, for producers
//!   that announce nothing (a bare byte pipeline).
//!
//! ## The header, and one frame per input buffer
//! The **header is emitted lazily on the first frame** (so a stream with no frames
//! produces no partial file, and the track config is fixed before any block). Each input
//! buffer is one encoded frame → one `SimpleBlock` (spec `§simpleblock`), staged
//! **scatter-style** (ZERO-COPY.md Stage 2): only the ~10-octet block header is
//! buffered; the frame's `Memory` rides the open Cluster as a refcount and leaves as a
//! whole payload buffer when the Cluster closes — frame bytes are never copied through
//! this element. Header byte runs still go through pool slots (`try_alloc` + carry —
//! the backpressure discipline). **A frame larger than the upstream's pool slot arrives
//! split and would mux as several blocks** — for remuxing, size the demuxer's pool to
//! the largest frame (`Pipeline::set_element_pool`). Timestamps come from the buffer
//! PTS (falling back to a synthesised cadence when absent). A block is a keyframe
//! unless the buffer is tagged [`BufferFlags::DELTA`] — untagged means "unknown", and
//! every FLAC frame is independently decodable, so keyframe is the right default; a
//! demuxed video track arrives explicitly tagged either way.
//!
//! ## EOS / finalize (spec: Events — Eos)
//! The Segment and final Cluster are unknown-size streamed masters closed implicitly at end
//! of stream (spec `§sizing`), so there is no tail to flush — [`MatroskaWriter::finalize`]
//! is seek-free. The element still runs it on [`Event::Eos`] (primary) and in [`stop`]
//! (belt-and-braces, idempotent) for symmetry with `OggMux` and so a future trailing
//! element (Cues) has a hook. Any bytes `finalize` were to produce are pushed downstream.

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::{Buffer, BufferFlags};
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::{FormatId, PadId};
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::memory::Memory;
use streamcraft_core::time::Timestamp;

use crate::codec::{self, Reframer};
use crate::reader::{MatroskaReader, Track};
use crate::writer::{MatroskaWriter, MuxOut, MuxPiece, TrackConfig};

/// The one track this single-sink-pad element muxes (spec: single-track element). The
/// writer is N-track, but the element feeds it exactly this track.
const TRACK: u64 = 1;

// The src pad's local index (== position in the element's `pads` array). The sink pad
// (index 0) needs no id here: input is drained via `Inputs::pop`, which is pad-agnostic
// (matches `OggMux`).
const SRC: PadId = PadId(1);

/// Raw `bytes`: the src side is a Matroska byte stream; also the demux sink side.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
/// The mux sink declares every family it can build a track from **and every field/value
/// name its announcements carry** — declaring them is what interns the names, so the
/// upstream's `build_fixed` can resolve its announcement in a pipeline where no decoder
/// ever mentioned them (spec: Formats — a `FixedFormat` resolves against the link-time
/// vocabulary; a wildcard pad would admit everything but intern nothing). A family a
/// demuxer announces that is *not* here (av1, the NAL codecs) still installs tolerantly
/// and reaches `event()`, where it is a loud unsupported-remux error.
static MUX_SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static MUX_AUDIO_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&MUX_SAMPLE_VALUES), preferred: None },
];
/// The colorimetry value names an upstream (demuxer) announcement may carry — `Set`,
/// not `Any`, for the same reason as [`MUX_SAMPLE_VALUES`]: declaring the *values* is
/// what interns them, and `build_fixed` drops an announcement whose names never
/// interned. The names mirror `streamcraft-video`'s color vocabulary (pinned by that
/// crate's tests); the muxer maps them back to H.273 code points in `color_map`.
static MUX_MATRIX_VALUES: [ValueDesc; 4] = [
    ValueDesc::Id("bt709"),
    ValueDesc::Id("bt601"),
    ValueDesc::Id("bt2020"),
    ValueDesc::Id("identity"),
];
static MUX_RANGE_VALUES: [ValueDesc; 2] = [ValueDesc::Id("limited"), ValueDesc::Id("full")];
static MUX_TRANSFER_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("bt709"),
    ValueDesc::Id("srgb"),
    ValueDesc::Id("pq"),
    ValueDesc::Id("hlg"),
    ValueDesc::Id("linear"),
];
static MUX_PRIMARIES_VALUES: [ValueDesc; 4] = [
    ValueDesc::Id("bt709"),
    ValueDesc::Id("bt601"),
    ValueDesc::Id("bt2020"),
    ValueDesc::Id("dci-p3"),
];
static MUX_VIDEO_FIELDS: [FieldDesc; 7] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    // Optional presentation duration (ns) — a remuxing demuxer knows it from the source's
    // tables and this muxer writes it as `Info\Duration`, so the output has a duration and
    // a seek bar instead of reading as a live stream.
    FieldDesc { field: "duration", allowed: ConstraintDesc::Any, preferred: None },
    // Optional colorimetry names (H.273 via the pipeline color vocab) — written
    // back as `Video\Colour` so a remux preserves color metadata. (Link-time
    // fixation may pin a phantom value here; harmless — the muxer configures
    // only from the runtime announcement, never the link format.)
    FieldDesc { field: "matrix", allowed: ConstraintDesc::Set(&MUX_MATRIX_VALUES), preferred: None },
    FieldDesc { field: "range", allowed: ConstraintDesc::Set(&MUX_RANGE_VALUES), preferred: None },
    FieldDesc { field: "transfer", allowed: ConstraintDesc::Set(&MUX_TRANSFER_VALUES), preferred: None },
    FieldDesc { field: "primaries", allowed: ConstraintDesc::Set(&MUX_PRIMARIES_VALUES), preferred: None },
];
static MUX_SINK_OFFERS: [OfferDesc; 6] = [
    OfferDesc { family: "flac", fields: &MUX_AUDIO_FIELDS },
    OfferDesc { family: "vp8", fields: &MUX_VIDEO_FIELDS },
    OfferDesc { family: "vp9", fields: &MUX_VIDEO_FIELDS },
    // The passthrough NAL framings (`sc-mp4`'s remux mode): length-prefixed samples led
    // by the raw config record in-band — Matroska's native shape (RFC 9559 §12).
    OfferDesc { family: "h264/avcc", fields: &MUX_VIDEO_FIELDS },
    OfferDesc { family: "h265/hvcc", fields: &MUX_VIDEO_FIELDS },
    OfferDesc::any("bytes"),
];

static MUX_PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &MUX_SINK_OFFERS,
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

// COLD: make_default boxes the element once at construction (descriptor factory).
#[allow(clippy::disallowed_methods)]
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
    // The parse/launch default is the caps-driven muxer: `... ! mkvmux ! filesink` learns
    // its track from the upstream announcement (spec: dynamic caps; the remux path).
    make_default: Some(|| Box::new(MkvMux::from_caps())),
};

/// How this muxer learns its [`TrackConfig`] (see the module docs).
enum Setup {
    /// Constructor-supplied ([`MkvMux::flac`]/[`MkvMux::with_track`]): `track` is present
    /// from birth and the writer is built at `start()`.
    Fixed,
    /// [`MkvMux::from_caps`]: waiting for the upstream `FormatChange`. Data before any
    /// announcement is a loud error — a caps-driven muxer cannot guess its codec.
    AwaitCaps,
    /// A `flac` family was announced: absorbing the leading in-band native head (`fLaC` +
    /// metadata blocks) into the `CodecPrivate` before the writer can be built. Holds the
    /// bytes absorbed so far.
    FlacHead(Vec<u8>),
    /// A passthrough NAL family (`h264/avcc`/`h265/hvcc`) was announced: the **first
    /// buffer** is the raw `avcC`/`hvcC` record (the demuxer emits it as one leading
    /// buffer — it is a few hundred octets, far below any pool slot), taken verbatim as
    /// `CodecPrivate` (RFC 9559 §12); every later buffer is one length-prefixed sample.
    NalHead {
        codec_id: &'static str,
        width: u32,
        height: u32,
        duration_ns: Option<u64>,
        /// Announced colorimetry → written back as `Video\Colour` (H.273 points).
        colour: Option<crate::writer::ColourConfig>,
    },
}

/// Muxes a single track into Matroska: one encoded frame **per input buffer** on the sink
/// pad, an MKV byte stream on the src pad. The header is emitted before the first frame;
/// each frame becomes a `SimpleBlock` in a Cluster (spec `§simpleblock`).
///
/// The track's `CodecID`/`CodecPrivate`/params come from the upstream announcement
/// ([`from_caps`](Self::from_caps) — the remux path) or from construction (see the module
/// docs). Handling several input tracks into one MKV needs one dynamic sink pad per input
/// plus fan-in — a documented follow-up.
pub struct MkvMux {
    /// The track config the writer is built from. `Some` from birth in `Fixed` setup;
    /// filled by the announcement/head in caps-driven setup.
    track: Option<TrackConfig>,
    /// How `track` is (or will be) obtained.
    setup: Setup,
    /// The writer. Present once the track is known (from `start()` in `Fixed` setup,
    /// from the announcement/head otherwise) until finalize.
    writer: Option<MatroskaWriter>,
    /// The stream has been finalized: late buffers are dropped, never remuxed into a
    /// reopened stream.
    done: bool,
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
    /// The scatter output the writer stages into (ZERO-COPY.md Stage 2): tiny header byte
    /// runs + refcounted payload slices, drained downstream by
    /// [`drain_out`](Self::drain_out). Doubles as the backpressure carry — while pieces
    /// are queued (pool dry mid-drain), `process` consumes no input.
    out: MuxOut,
    /// Byte offset already emitted of the *front* `Bytes` piece in `out` (a pool-dry
    /// partial emission resumes here). 0 whenever the front piece is fresh.
    out_off: usize,
    /// The `FormatId` a pool-allocated buffer on the src pad would carry (`Ctx` stamps
    /// its out-format; no public accessor), probed once at `start` so forwarded payload
    /// buffers match pooled ones. Not load-bearing on the milestone-1 transport
    /// (`Batch::push` drops the per-row format), but kept faithful.
    out_fmt: FormatId,
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
            track: Some(track),
            setup: Setup::Fixed,
            writer: None,
            done: false,
            header_done: false,
            next_ts_ns: 0,
            frame_dur_ns,
            out: MuxOut::new(),
            out_off: 0,
            out_fmt: FormatId(0), // probed at `start`
        }
    }

    /// A caps-driven MKV muxer (the registry default; the **remux** path): the track is
    /// built from the upstream runtime announcement — see the module docs for the
    /// supported families and the loud-error ones. `mkvdemux ! mkvmux` is the canonical
    /// use.
    pub fn from_caps() -> Self {
        Self {
            track: None,
            setup: Setup::AwaitCaps,
            writer: None,
            done: false,
            header_done: false,
            next_ts_ns: 0,
            frame_dur_ns: 0,
            out: MuxOut::new(),
            out_off: 0,
            out_fmt: FormatId(0), // probed at `start`
        }
    }

    /// This element's track config, once known (for tests / introspection). `None` for a
    /// caps-driven muxer that has not seen its announcement yet.
    pub fn track(&self) -> Option<&TrackConfig> {
        self.track.as_ref()
    }

    /// Install `track` and build the writer — the moment a caps-driven muxer becomes
    /// equivalent to a constructed one. `duration_ns` (when the announcement carried it)
    /// becomes `Info\Duration`, written with the lazy header.
    fn configure(&mut self, mut track: TrackConfig, duration_ns: Option<u64>) {
        track.track_number = TRACK;
        self.frame_dur_ns = Self::frame_duration_ns(&track);
        let mut writer = MatroskaWriter::new(vec![track.clone()]);
        if let Some(ns) = duration_ns {
            writer.set_duration_ns(ns);
        }
        self.writer = Some(writer);
        self.track = Some(track);
        self.setup = Setup::Fixed;
    }

    /// The total length of a complete native FLAC head — `fLaC` magic + every metadata
    /// block (RFC 9639 §8: a 4-octet block header of last-flag(1)+type(7) then a 24-bit
    /// big-endian length, repeated until the last-flag) — or `None` while `b` is still a
    /// proper prefix of one. A wrong magic is a hard error: the announced `flac` stream
    /// does not start with a FLAC head. (`pub(crate)`: shared with the multi-track
    /// [`MkvMuxN`](crate::MkvMuxN), whose per-pad flac setup absorbs the same head.)
    pub(crate) fn flac_head_len(b: &[u8]) -> Result<Option<usize>, Error> {
        if b.len() >= 4 && &b[..4] != b"fLaC" {
            return Err(Error::Todo("mkvmux: announced flac stream does not start with fLaC"));
        }
        if b.len() < 4 {
            return Ok(None);
        }
        let mut off = 4;
        loop {
            if b.len() < off + 4 {
                return Ok(None);
            }
            let last = b[off] & 0x80 != 0;
            let len = u32::from_be_bytes([0, b[off + 1], b[off + 2], b[off + 3]]) as usize;
            off += 4 + len;
            if last {
                // The head is complete once all of the last block's payload is present.
                return Ok(if b.len() >= off { Some(off) } else { None });
            }
        }
    }

    /// The audio params out of a native FLAC head's STREAMINFO (RFC 9639 §8.2: the first
    /// metadata block, type 0, 34 octets — sample rate is the 20 bits at bit offset 80,
    /// then 3 bits channels−1, then 5 bits bits-per-sample−1). (`pub(crate)`: shared with
    /// [`MkvMuxN`](crate::MkvMuxN).)
    pub(crate) fn flac_streaminfo_params(head: &[u8]) -> Result<(f64, u32, u32), Error> {
        // head[..4] == "fLaC"; the STREAMINFO body starts after its 4-octet block header.
        let si = head
            .get(8..8 + 34)
            .ok_or(Error::Todo("mkvmux: flac head too short for STREAMINFO"))?;
        if head[4] & 0x7f != 0 {
            return Err(Error::Todo("mkvmux: first flac metadata block is not STREAMINFO"));
        }
        let rate = ((si[10] as u32) << 12) | ((si[11] as u32) << 4) | ((si[12] as u32) >> 4);
        let channels = (((si[12] >> 1) & 0x7) as u32) + 1;
        let bits = ((((si[12] & 1) as u32) << 4) | ((si[13] as u32) >> 4)) + 1;
        if rate == 0 {
            return Err(Error::Todo("mkvmux: STREAMINFO sample rate is zero"));
        }
        Ok((rate as f64, channels, bits))
    }

    /// Nanoseconds between synthesised frame timestamps when buffers carry no PTS: one FLAC
    /// block (4096 interchannel samples — the encoder's block size) at the track's sample
    /// rate. A defined, monotonic cadence so the muxed timestamps advance sensibly; a real
    /// upstream that stamps PTS overrides this entirely. (`pub(crate)`: shared with
    /// [`MkvMuxN`](crate::MkvMuxN).)
    pub(crate) fn frame_duration_ns(track: &TrackConfig) -> u64 {
        const FLAC_BLOCK: u64 = 4096;
        let rate = track.audio.sampling_frequency;
        if rate > 0.0 {
            ((FLAC_BLOCK as f64) * 1_000_000_000.0 / rate) as u64
        } else {
            0
        }
    }

    /// Push a payload piece downstream as one whole buffer — a refcount move, no pool
    /// slot, no copy (ZERO-COPY.md Stage 2). The bytes are the retained input the
    /// upstream demuxer's pool already accounts for; a slow sink holding them keeps
    /// those slots outstanding — that *is* the backpressure. MKV byte-stream buffers
    /// carry no pts/flags (matching the chunked emission this replaces).
    fn emit_payload(ctx: &mut Ctx, memory: Memory, format: FormatId) {
        ctx.out(SRC).push(Buffer {
            memory,
            pts: Timestamp::NONE,
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags: BufferFlags::empty(),
            format,
            sync: None,
        });
    }

    /// Drain staged output pieces downstream, in order — **pool bounded** for the header
    /// byte runs ([`Ctx::try_alloc`] + the `out_off` cursor carry; returns `false` when
    /// the pool ran dry, the signal to stop consuming input), refcount-forwarding for
    /// payload pieces (never blocks). Reclaims the header arena once everything drained.
    fn drain_out(&mut self, ctx: &mut Ctx) -> bool {
        loop {
            match self.out.front() {
                None => break,
                Some(&MuxPiece::Bytes { start, end }) => {
                    let mut at = start + self.out_off;
                    while at < end {
                        let Some(mut buf) = ctx.try_alloc(SRC) else {
                            self.out_off = at - start;
                            return false; // pool dry — resume here next pass
                        };
                        let cap = buf.memory.capacity();
                        debug_assert!(cap > 0, "mkvmux: zero-capacity pool slot");
                        let n = cap.min(end - at);
                        buf.memory.as_mut_full()[..n]
                            .copy_from_slice(&self.out.header_bytes()[at..at + n]);
                        buf.memory.set_len(n);
                        ctx.out(SRC).push(buf);
                        at += n;
                    }
                    self.out_off = 0;
                    self.out.pop_front();
                }
                Some(MuxPiece::Payload(_)) => {
                    let Some(MuxPiece::Payload(mem)) = self.out.pop_front() else {
                        unreachable!("front was a payload")
                    };
                    Self::emit_payload(ctx, mem, self.out_fmt);
                }
            }
        }
        self.out.reclaim();
        true
    }

    /// The EOS drain: like [`drain_out`](Self::drain_out) but with **exact-size**
    /// allocations for the byte runs — waiting for a pool slot is no longer an option at
    /// end of stream (mirrors the demuxers' `flush_exact`).
    fn drain_out_exact(&mut self, ctx: &mut Ctx) {
        while let Some(piece) = self.out.pop_front() {
            match piece {
                MuxPiece::Bytes { start, end } => {
                    let at = start + self.out_off;
                    self.out_off = 0;
                    if at >= end {
                        continue;
                    }
                    let n = end - at;
                    let mut buf = ctx.alloc_exact(SRC, n);
                    buf.memory.as_mut_full()[..n]
                        .copy_from_slice(&self.out.header_bytes()[at..end]);
                    buf.memory.set_len(n);
                    ctx.out(SRC).push(buf);
                }
                MuxPiece::Payload(mem) => Self::emit_payload(ctx, mem, self.out_fmt),
            }
        }
        self.out.reclaim();
    }

    /// Finalize the stream exactly once (spec `§sizing` — seek-free). Idempotent: the
    /// writer is taken on first call, so later calls (the `event` path then `stop`, or vice
    /// versa) do nothing beyond draining any leftover pieces. The final Cluster's pieces
    /// are pushed downstream with exact-size byte runs (no slot to wait for at EOS).
    fn finish_stream(&mut self, ctx: &mut Ctx) {
        self.done = true;
        if let Some(mut writer) = self.writer.take() {
            // Cues are not enabled on the single-pad mux (its streaming users have no
            // seekable target), so there is never a patch to forward.
            let _ = writer.finalize_scatter(&mut self.out);
        }
        self.drain_out_exact(ctx);
    }
}

impl Element for MkvMux {
    fn desc(&self) -> &'static ElementDesc {
        &MUX_DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Fixed setup builds its writer here; caps-driven setup builds it when the
        // announcement (and, for flac, the in-band head) arrives.
        if let (Setup::Fixed, Some(track)) = (&self.setup, &self.track) {
            self.writer = Some(MatroskaWriter::new(vec![track.clone()]));
        }
        self.done = false;
        self.header_done = false;
        self.next_ts_ns = 0;
        self.out.clear();
        self.out_off = 0;
        // Probe the out-format a pooled buffer would carry (see the `out_fmt` field): one
        // throwaway right-sized allocation, recycled on drop — never a steady-state cost.
        self.out_fmt = ctx.alloc_exact(SRC, 1).format;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Backpressure: resume any parked output pieces first, and consume input only
        // while emission keeps up (un-popped input stays buffered for the next pass, so
        // the scheduler's gates absorb a fast demuxer racing a slow sink).
        if !self.drain_out(ctx) {
            return Ok(Flow::Ok);
        }
        while let Some(buf) = inputs.pop() {
            // Stop early if the stream is already finalized: a late buffer after EOS is
            // dropped rather than reopening a closed stream (matches `OggMux`).
            if self.done {
                break;
            }
            // A caps-driven muxer that has not yet built its writer is either absorbing
            // the flac in-band head or has been fed data before any announcement.
            if self.writer.is_none() {
                match &mut self.setup {
                    Setup::FlacHead(head) => {
                        head.extend_from_slice(buf.memory.data());
                        match Self::flac_head_len(head)? {
                            None => continue, // head still incomplete — keep absorbing
                            Some(len) => {
                                if len != head.len() {
                                    // The head must end on a buffer boundary — `MkvDemux`
                                    // and `flacenc` both emit it as its own emission; bytes
                                    // beyond it here would be frames we cannot re-split.
                                    return Err(Error::Todo(
                                        "mkvmux: flac head not on a buffer boundary",
                                    ));
                                }
                                let head = std::mem::take(head);
                                let (rate, channels, bits) = Self::flac_streaminfo_params(&head)?;
                                self.configure(
                                    TrackConfig::flac(TRACK, head, rate, channels, bits),
                                    None,
                                );
                                continue; // the head is CodecPrivate, never a block
                            }
                        }
                    }
                    Setup::NalHead { codec_id, width, height, duration_ns, colour } => {
                        // First buffer = the raw config record, verbatim (RFC 9559 §12:
                        // CodecPrivate *is* the AVC/HEVCDecoderConfigurationRecord).
                        // configurationVersion is 1 for both records — a cheap guard
                        // against being fed sample data first.
                        // COLD: once per stream — owns the CodecPrivate record before the writer exists.
                        #[allow(clippy::disallowed_methods)]
                        let head = buf.memory.data().to_vec();
                        if head.first() != Some(&1) {
                            return Err(Error::Todo(
                                "mkvmux: leading buffer is not an avcC/hvcC record \
                                 (configurationVersion != 1)",
                            ));
                        }
                        let (codec_id, w, h, dur, col) =
                            (*codec_id, *width, *height, *duration_ns, *colour);
                        let mut tc = TrackConfig::video(TRACK, codec_id, head, w, h);
                        if let Some(v) = tc.video.as_mut() {
                            v.colour = col;
                        }
                        self.configure(tc, dur);
                        continue; // the record is CodecPrivate, never a block
                    }
                    Setup::AwaitCaps => {
                        return Err(Error::Todo(
                            "mkvmux: data before any format announcement — a caps-driven \
                             muxer cannot guess its codec (use MkvMux::with_track for bare \
                             byte streams)",
                        ));
                    }
                    Setup::Fixed => break, // unreachable: Fixed builds the writer at start
                }
            }

            // Timestamp: prefer the buffer PTS; otherwise the synthesised cadence. Computed
            // before borrowing the writer (it mutates `self.next_ts_ns`).
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
            // A block is a keyframe unless explicitly tagged DELTA (spec: Buffer flags —
            // untagged means "unknown", and all-independent-frame codecs like FLAC are the
            // untagged norm; a demuxed video track arrives tagged either way).
            let keyframe = !buf.flags.contains(BufferFlags::DELTA);

            let writer = self.writer.as_mut().expect("writer present");

            // Emit the header lazily, before the first frame — so an empty stream writes
            // nothing, and the track config is fixed before any block. The only error path
            // (BadTracks) cannot occur for our single fixed, validated track.
            if !self.header_done {
                let _ = writer.write_header_scatter(&mut self.out);
                self.header_done = true;
            }

            // One input buffer == one encoded frame == one SimpleBlock (ZERO-COPY.md
            // Stage 2): the frame's `Memory` moves into the open Cluster as a refcount —
            // the staged Cluster holds refcounts on the retained input, not byte copies —
            // and comes back out of `drain_out` as a whole payload buffer when the
            // Cluster closes. The only error is an unknown track, impossible for our
            // single fixed track.
            let _ = writer.write_frame_scatter(&mut self.out, TRACK, ts_ns, buf.memory, keyframe);

            if !self.drain_out(ctx) {
                return Ok(Flow::Ok); // pool dry mid-drain — stop consuming input
            }
        }
        Ok(Flow::Ok)
    }

    // COLD: EOS finalize + once-per-stream caps-driven track-config assembly (no per-frame path).
    #[allow(clippy::disallowed_methods)]
    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Primary finalize path: the pipeline delivers EOS in-band and pushes whatever
            // this produces before closing the ring.
            Event::Eos => self.finish_stream(ctx),
            // The caps-driven track config (spec: dynamic caps; the remux path). Only the
            // first announcement configures; a mid-stream change cannot be honoured — a
            // Matroska track is fixed once the header is written — so it is a loud error
            // rather than a silently corrupt file.
            Event::FormatChange(f) => match &self.setup {
                Setup::AwaitCaps => {
                    let family = ctx.family_name(f.family).unwrap_or("?").to_owned();
                    let dim = |name: &str| {
                        ctx.field_id(name)
                            .and_then(|id| f.get(id))
                            .and_then(|v| match v {
                                Value::Int(n) if n > 0 => Some(n as u32),
                                _ => None,
                            })
                    };
                    // Optional announced presentation duration (ns) → `Info\Duration`.
                    let duration_ns = ctx
                        .field_id("duration")
                        .and_then(|id| f.get(id))
                        .and_then(|v| match v {
                            Value::Int(n) if n > 0 => Some(n as u64),
                            _ => None,
                        });
                    // Optional colorimetry → written back as `Video\Colour`.
                    let colour = crate::color_map::colour_from_format(ctx, f);
                    match family.as_str() {
                        // The CodecPrivate is the in-band native head; absorb it first.
                        "flac" => self.setup = Setup::FlacHead(Vec::new()),
                        "vp8" | "vp9" => {
                            let (Some(w), Some(h)) = (dim("width"), dim("height")) else {
                                return Err(Error::Todo(
                                    "mkvmux: video announcement without width/height",
                                ));
                            };
                            let id = if family == "vp8" { "V_VP8" } else { "V_VP9" };
                            let mut tc = TrackConfig::video(TRACK, id, Vec::new(), w, h);
                            if let Some(v) = tc.video.as_mut() {
                                v.colour = colour;
                            }
                            self.configure(tc, duration_ns);
                        }
                        // Passthrough NAL remux: the CodecPrivate arrives as the first
                        // buffer (the raw record); defer writer creation until then.
                        "h264/avcc" | "h265/hvcc" => {
                            let (Some(w), Some(h)) = (dim("width"), dim("height")) else {
                                return Err(Error::Todo(
                                    "mkvmux: video announcement without width/height",
                                ));
                            };
                            let codec_id = if family == "h264/avcc" {
                                "V_MPEG4/ISO/AVC"
                            } else {
                                "V_MPEGH/ISO/HEVC"
                            };
                            self.setup = Setup::NalHead {
                                codec_id,
                                width: w,
                                height: h,
                                duration_ns,
                                colour,
                            };
                        }
                        // Families we cannot mux *conformantly* yet — loud, with the reason
                        // (see the module docs): av1 needs an av1C CodecPrivate; the Annex B
                        // decode framings lack the config record Matroska wants (remux from
                        // MP4 with `Mp4Demux::passthrough` → `h264/avcc` instead).
                        other => {
                            return Err(Error::Resource(format!(
                                "mkvmux: cannot remux family '{other}' (av1 needs an av1C \
                                 CodecPrivate — follow-up; for h264/h265 use the demuxer's \
                                 passthrough mode, which announces h264/avcc / h265/hvcc)"
                            )));
                        }
                    }
                }
                Setup::Fixed | Setup::FlacHead(_) | Setup::NalHead { .. } => {
                    if self.header_done {
                        return Err(Error::Todo(
                            "mkvmux: mid-stream format change cannot be muxed (track is \
                             fixed once the header is written)",
                        ));
                    }
                    // Not yet writing: a repeated/refined announcement before data is
                    // harmless (e.g. a re-validation pass re-delivering the same format).
                }
            },
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: guarantee finalize even if `event(Eos)` was not delivered.
        // Idempotent via the `Option` take.
        self.finish_stream(ctx);
        self.out = MuxOut::new();
        self.out_off = 0;
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
    /// The announce family for the pad ([`codec::family_for`]): `flac`, `vp8`/`vp9`/`av1`,
    /// `h264/annexb`/`h265/annexb`, or `bytes` for an unknown codec id.
    family: &'static str,
    /// The native codec head to emit *before* the first frame. For A_FLAC this is the
    /// CodecPrivate (`fLaC` + STREAMINFO), reconstructing a native FLAC byte stream a
    /// downstream `flacdec` decodes (spec: A_FLAC mapping). For H.264/H.265 it is the
    /// parameter sets from the config record as an Annex B head (SPS/PPS, VPS/SPS/PPS). Empty
    /// for a raw `bytes` / WebM (VP8/VP9/AV1) track — those frames are self-contained.
    codec_head: Vec<u8>,
    /// How each Block's payload is reframed to the bytes emitted downstream: passthrough for
    /// FLAC / WebM / raw bytes; length-prefixed-NAL → Annex B for H.264/H.265.
    reframer: Reframer,
    /// The track params (audio rate/channels/sample, or video pixel dims) so the runtime
    /// announcement carries the concrete format a caps-aware consumer sees.
    track: Track,
    /// False until the codec head + the format announcement have been emitted on this pad.
    started: bool,
    /// True for a video track (a codec whose family [`is_video_family`]). Only video tracks
    /// need post-seek keyframe gating — an inter-frame decoder must start at a keyframe.
    is_video: bool,
    /// Post-seek keyframe gate (spec: flush/seek). Set on `FlushStart` for a video track; while
    /// set, delta frames are dropped until the first `keyframe` frame after the flush (a decoder
    /// that resumed mid-GOP would render garbage). Audio tracks never set it. Cleared once the
    /// first post-seek keyframe passes.
    awaiting_keyframe: bool,
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
    /// Reused NAL-reframe buffer: each length-prefixed → Annex B conversion refills this in
    /// place, so the per-frame hot path allocates nothing on the global heap (spec: Memory).
    reframe_buf: Vec<u8>,
    /// Emissions stalled on pool exhaustion (spec: the backpressure rule — the pool,
    /// never the heap, bounds a demuxer racing a slow decoder). While non-empty the
    /// demuxer consumes **no input**, so upstream backs up into the scheduler's gates
    /// instead of this element ballooning. Holds at most a codec head + one frame.
    pending: std::collections::VecDeque<Carry>,
    /// Recycled [`Carry`] payload buffers. A slow decoder keeps the pool full, so `drain` parks
    /// a remainder nearly every frame; returning each fully-emitted Carry's buffer here (instead
    /// of dropping it and `to_vec`-ing a fresh one next park) makes the backpressure path
    /// zero-alloc in steady state. Bounded (`MAX_PARK_FREE`).
    park_free: Vec<Vec<u8>>,
    /// `DurationChanged` has been posted this run (once, from `process` — posted at
    /// stream time, not preroll, so an introspection server started at `run()` sees it).
    posted_duration: bool,
}

/// Cap on retained park buffers — a head plus a few frames is the most that parks at once.
const MAX_PARK_FREE: usize = 8;

/// Return a fully-emitted [`Carry`]'s buffer to the free-list for the next park (bounded).
fn recycle_park(free: &mut Vec<Vec<u8>>, buf: Vec<u8>) {
    if free.len() < MAX_PARK_FREE {
        free.push(buf);
    }
}

/// A partially-emitted blob: `bytes[off..]` still needs pool slots on `pad`.
struct Carry {
    pad: PadId,
    pts_ns: Option<u64>,
    /// Presentation duration in ns (`BlockDuration`), stamped on every chunk of the frame so a
    /// split emission preserves it — load-bearing for duration-bearing sparse tracks
    /// (subtitles, where the cue span is `[pts, pts+dur)`). `None` when the block declared none.
    duration_ns: Option<u64>,
    flags: BufferFlags,
    bytes: Vec<u8>,
    off: usize,
}

/// Own a parked remainder's bytes for a [`Carry`], reusing a recycled buffer from `free` (a
/// slow decoder parks nearly every frame, so this is a hot path, not a rare stall — a recycled
/// buffer keeps it off the heap; `extend_from_slice` reuses the buffer's capacity). The parked
/// bytes must outlive the current `process()`, so they are copied out of the borrowed source.
// The `Vec::new()` fallback (empty, zero-heap) is hit only before the free-list warms.
#[allow(clippy::disallowed_methods)]
fn own_remainder(free: &mut Vec<Vec<u8>>, bytes: &[u8]) -> Vec<u8> {
    let mut buf = free.pop().unwrap_or_default();
    buf.clear();
    buf.extend_from_slice(bytes);
    buf
}

impl MkvDemux {
    /// A demuxer that discovers its tracks from `header` — the leading bytes of the Matroska
    /// stream, which MUST include the whole EBML Header + Segment Info + Tracks (everything up
    /// to the first Cluster). The caller reads these once up front (e.g. the first read of a
    /// `filesrc`); the full stream — these bytes and all that follow — is then fed on the
    /// sink pad at run time. See the type docs for why discovery is constructor-supplied.
    // COLD: element construction — empty reusable buffers/collections, filled at run time.
    #[allow(clippy::disallowed_methods)]
    pub fn new(header: Vec<u8>) -> Self {
        Self {
            header,
            pad_tracks: Vec::new(),
            reader: MatroskaReader::new(),
            scratch: Vec::new(),
            reframe_buf: Vec::new(),
            pending: std::collections::VecDeque::new(),
            park_free: Vec::new(),
            posted_duration: false,
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

    /// The stream-start head + the per-frame [`Reframer`] for a track (spec: A_FLAC mapping;
    /// RFC 9559 §12 codec mappings). Three shapes:
    /// - **A_FLAC**: the CodecPrivate *is* the native FLAC head (`fLaC` + STREAMINFO), forwarded
    ///   verbatim so a downstream `flacdec` sees exactly a native `.flac` stream; passthrough
    ///   frames.
    /// - **H.264 / H.265** (`V_MPEG4/ISO/AVC` / `V_MPEGH/ISO/HEVC`): the CodecPrivate is an
    ///   AVC/HEVC configuration record — the head is its parameter sets as an Annex B stream
    ///   (SPS/PPS, VPS/SPS/PPS), and each Block is reframed from length-prefixed NALs to Annex
    ///   B. A **malformed** record degrades to an empty head + passthrough (never panics): the
    ///   pad still links, the decoder simply gets no parameter sets until an in-band one arrives.
    /// - **Everything else** (WebM VP8/VP9/AV1, unknown `bytes`): no head, passthrough frames.
    // COLD: once per track (discovery/start/seek) — owns the codec head (CodecPrivate/param sets).
    #[allow(clippy::disallowed_methods)]
    fn head_and_reframer(track: &Track) -> (Vec<u8>, Reframer) {
        match track.codec_id.as_str() {
            CODEC_A_FLAC => (track.codec_private.clone(), Reframer::Passthrough),
            // A_AAC: CodecPrivate is the raw AudioSpecificConfig (RFC 9559 §12),
            // delivered as the in-band head; each Block is one raw AU as-is.
            "A_AAC" => (track.codec_private.clone(), Reframer::Passthrough),
            "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC" => {
                let is_hevc = track.codec_id == "V_MPEGH/ISO/HEVC";
                match codec::nal_head_from_config(&track.codec_private, is_hevc) {
                    Ok((head, length_size)) => (head, Reframer::Nal { length_size }),
                    // Untrusted CodecPrivate: a broken record must not kill the demuxer. Fall
                    // back to a bare pad and let the decoder resync at an in-band keyframe.
                    Err(_) => (Vec::new(), Reframer::Passthrough),
                }
            }
            _ => (Vec::new(), Reframer::Passthrough),
        }
    }

    /// Push `bytes[off..]` on `pad` via **pooled** slots (`try_alloc`), chunked to the slot
    /// size, returning the new offset — `< bytes.len()` means the pool ran dry and the caller
    /// must carry the remainder and stop consuming input (spec: the backpressure rule). PTS,
    /// duration (`BlockDuration`) and `flags` (the frame's keyframe/delta tag, on every chunk of
    /// the frame) are stamped on each buffer.
    fn emit_bounded(
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        duration_ns: Option<u64>,
        flags: BufferFlags,
        bytes: &[u8],
        mut off: usize,
    ) -> usize {
        while off < bytes.len() {
            let Some(mut buf) = ctx.try_alloc(pad) else { return off };
            let cap = buf.memory.capacity();
            debug_assert!(cap > 0, "mkvdemux: zero-capacity pool slot");
            let n = cap.min(bytes.len() - off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&bytes[off..off + n]);
            buf.memory.set_len(n);
            if let Some(t) = pts_ns {
                buf.pts = Timestamp::from_nanos(t);
            }
            if let Some(d) = duration_ns {
                buf.duration = Timestamp::from_nanos(d);
            }
            buf.flags = flags;
            ctx.out(pad).push(buf);
            off += n;
        }
        off
    }

    /// Emit `bytes[off..]` as one **exact-size** buffer (`alloc_exact` — a right-sized heap
    /// allocation when the pool is dry, never a slot-sized over-allocation). Only for the
    /// bounded EOS/stop flush, where yielding for a slot is no longer possible.
    fn emit_exact(
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        duration_ns: Option<u64>,
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
        if let Some(d) = duration_ns {
            buf.duration = Timestamp::from_nanos(d);
        }
        buf.flags = flags;
        ctx.out(pad).push(buf);
    }

    /// Queue a blob for emission and try to emit it now; on pool exhaustion the remainder
    /// parks in `pending` (drained first on the next pass).
    fn emit_or_park(
        &mut self,
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        duration_ns: Option<u64>,
        flags: BufferFlags,
        bytes: Vec<u8>,
    ) {
        let off = Self::emit_bounded(ctx, pad, pts_ns, duration_ns, flags, &bytes, 0);
        if off < bytes.len() {
            self.pending.push_back(Carry { pad, pts_ns, duration_ns, flags, bytes, off });
        }
    }

    /// Resume parked emissions. `true` when everything pending has drained.
    fn drain_pending(&mut self, ctx: &mut Ctx) -> bool {
        while let Some(mut c) = self.pending.pop_front() {
            c.off = Self::emit_bounded(ctx, c.pad, c.pts_ns, c.duration_ns, c.flags, &c.bytes, c.off);
            if c.off < c.bytes.len() {
                self.pending.push_front(c);
                return false;
            }
            // Fully emitted — return its buffer for the next park (zero-alloc backpressure).
            recycle_park(&mut self.park_free, c.bytes);
        }
        true
    }

    /// Drain frames the reader has decoded, emitting each on its track's src pad — **pool
    /// bounded**: returns `false` when slots ran out (remainder parked in `pending`; stop
    /// consuming input until it clears). On a track's first frame the native codec head is
    /// emitted first and the pad's runtime format is announced (spec: dynamic caps). A frame
    /// for an undiscovered track is dropped (no pad); a Block that fails to reframe is
    /// warned-and-dropped, not fatal (spec: untrusted input; per-buffer error scope).
    fn drain(&mut self, ctx: &mut Ctx) -> bool {
        if !self.drain_pending(ctx) {
            return false;
        }
        while let Some(frame) = self.reader.next_frame() {
            let Some(idx) = self.pad_tracks.iter().position(|pt| pt.track_number == frame.track_number) else {
                continue; // undiscovered track — no pad to route to
            };
            // Post-seek keyframe gate (spec: flush/seek): after a FlushStart a video track drops
            // delta frames until its first keyframe — a decoder resuming mid-GOP would render
            // garbage. Audio never gates. The first keyframe clears the gate and passes through.
            if self.pad_tracks[idx].awaiting_keyframe {
                if frame.keyframe {
                    self.pad_tracks[idx].awaiting_keyframe = false;
                } else {
                    continue; // delta before the first post-seek keyframe — drop
                }
            }
            // Reframe the Block payload to the downstream bytes (before touching pool memory)
            // so a malformed access unit is dropped without ever emitting the codec head/format
            // for a track that would then carry nothing (announce happens only on a good frame).
            // Reframe into the element's reusable buffer (taken out to free the `self` borrow for
            // the head emission below) — the NAL path allocates nothing per frame; passthrough
            // borrows `frame.data`, leaving the buffer untouched.
            let reframer = self.pad_tracks[idx].reframer;
            let mut reframe_buf = std::mem::take(&mut self.reframe_buf);
            let out_bytes: &[u8] = match reframer.reframe_into(&frame.data, &mut reframe_buf) {
                Ok(bytes) => bytes,
                Err(e) => {
                    self.reframe_buf = reframe_buf;
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!(
                                "mkvdemux: dropping malformed frame on track {}: {e:?}",
                                frame.track_number
                            ),
                        },
                    });
                    continue;
                }
            };
            // `out_bytes` borrows `frame.data` and/or `reframe_buf` (both locals now) — no `self`
            // borrow, so the head emission's `&mut self` call below is free.
            // First frame on this pad: emit the native codec head and announce the format.
            if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let (pad, family) = (self.pad_tracks[idx].pad, self.pad_tracks[idx].family);
                Self::announce(ctx, pad, family, &self.pad_tracks[idx].track);
                // The head carries the stream start; stamp it with the first frame's pts. Take
                // it (leaving empty) — it is emitted exactly once, before the first frame. The
                // head has no presentation duration of its own.
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                self.emit_or_park(ctx, pad, Some(frame.pts_ns), None, BufferFlags::empty(), head);
            }
            let pad = self.pad_tracks[idx].pad;
            // The container's keyframe bit rides the buffer (spec: Buffer flags) so a
            // remuxing consumer preserves seekability; `DELTA` is explicit — an untagged
            // buffer would read as "unknown", which a muxer defaults to keyframe.
            let flags = if frame.keyframe { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            // The block's presentation duration (BlockGroup\BlockDuration), when present:
            // rides the buffer's `duration` field so a duration-bearing sparse consumer (the
            // subtitle overlay) has the cue span `[pts, pts+dur)`. `None` for a SimpleBlock.
            let dur_ns = frame.duration_ns;
            // Emit from the (borrowed) reframed bytes; only a parked remainder pays the copy
            // into an owned carry.
            let out_len = out_bytes.len();
            let off = Self::emit_bounded(ctx, pad, Some(frame.pts_ns), dur_ns, flags, out_bytes, 0);
            // A parked remainder must own its bytes (it outlives this `process`); copy it out
            // before the `reframe_buf`/`frame.data` borrows end. Steady state (pool keeping up)
            // parks nothing, so no per-frame owned allocation.
            let remainder = (off < out_len).then(|| own_remainder(&mut self.park_free, &out_bytes[off..]));
            // `out_bytes` is now unused: its borrow of `reframe_buf`/`frame.data` has ended, so
            // the buffer can be restored (kept for the next frame) and the frame payload recycled.
            self.reframe_buf = reframe_buf;
            if let Some(bytes) = remainder {
                self.pending.push_back(Carry {
                    pad,
                    pts_ns: Some(frame.pts_ns),
                    duration_ns: dur_ns,
                    flags,
                    bytes,
                    off,
                });
            }
            // Return the frame's payload buffer to the reader for reuse (streamcraft patch): the
            // Matroska framing then allocates nothing in steady state. `out_bytes` is fully
            // consumed above, so `frame.data` is no longer borrowed. `next_frame` returned an
            // owned `Frame`, so `self.reader` is free to borrow here.
            self.reader.recycle(frame.data);
            if !self.pending.is_empty() {
                return false; // pool dry — carry parked, stop pulling reader frames
            }
        }
        true
    }

    /// The EOS/stop flush: everything parked or still queued in the reader goes out with
    /// exact-size allocations (no slot to wait for at end of stream, and no multi-MiB slot
    /// per small sample either). Bounded in volume: the steady-state path only consumed
    /// input while it could emit, so at most one input batch's worth of samples sits here.
    fn flush_exact(&mut self, ctx: &mut Ctx) {
        while let Some(c) = self.pending.pop_front() {
            Self::emit_exact(ctx, c.pad, c.pts_ns, c.duration_ns, c.flags, &c.bytes, c.off);
        }
        // Reuse drain()'s routing/announce logic by swapping the emitter is overkill for
        // the tail; inline the same walk with exact emission.
        while let Some(frame) = self.reader.next_frame() {
            let Some(idx) = self.pad_tracks.iter().position(|pt| pt.track_number == frame.track_number) else {
                continue;
            };
            // Reframe into the reusable buffer (taken out to free the `self` borrow), same as
            // `drain`; the tail is bounded so this stays off the per-frame heap too. `out_bytes`
            // borrows the local `reframe_buf`/`frame.data`, not `self`, so the head emission's
            // `&mut self` take below is free — and the head still goes out before the frame.
            let reframer = self.pad_tracks[idx].reframer;
            let mut reframe_buf = std::mem::take(&mut self.reframe_buf);
            let out_bytes: &[u8] = match reframer.reframe_into(&frame.data, &mut reframe_buf) {
                Ok(bytes) => bytes,
                Err(_) => {
                    self.reframe_buf = reframe_buf;
                    continue; // tail flush: drop silently rather than warn-spam
                }
            };
            if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let (pad, family) = (self.pad_tracks[idx].pad, self.pad_tracks[idx].family);
                Self::announce(ctx, pad, family, &self.pad_tracks[idx].track);
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                Self::emit_exact(ctx, pad, Some(frame.pts_ns), None, BufferFlags::empty(), &head, 0);
            }
            let pad = self.pad_tracks[idx].pad;
            let flags = if frame.keyframe { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            // Carry BlockDuration onto the tail buffer too (the subtitle cue span).
            Self::emit_exact(ctx, pad, Some(frame.pts_ns), frame.duration_ns, flags, out_bytes, 0);
            self.reframe_buf = reframe_buf; // borrow of the buffer ends here
        }
    }

    /// Announce a src pad's runtime format (spec: dynamic caps; follows `OggDemux`/`flacdec`).
    /// The concrete params ride the announcement so a caps-aware consumer sees the format:
    /// - `flac` → audio rate/channels/sample (mirroring what `flacdec` announces);
    /// - a **video** family (`vp8`/`vp9`/`av1`/`h264/annexb`/`h265/annexb`) → the container's
    ///   `width`/`height` from the Video element, when present (the decoder re-announces
    ///   authoritative dims from the bitstream — this just seeds negotiation);
    /// - `bytes` (or a video track with no declared dims) → a bare family announcement.
    ///
    /// Every family rides a byte bridge to a byte-reading peer, so a caps-ignoring decoder still
    /// links on the `bytes`/family offer.
    fn announce(ctx: &mut Ctx, pad: PadId, family: &'static str, track: &Track) {
        log!(
            &*ctx,
            Level::Debug,
            "announce",
            family = family,
            width = track.pixel_width,
            height = track.pixel_height,
        );
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
        } else if family == "aac" && track.sampling_frequency > 0.0 {
            // Container-declared rate/channels seed negotiation (a muxer reads
            // them); the decoder reads its authoritative config from the ASC head.
            ctx.announce_format(
                pad,
                "aac",
                &[
                    ("rate", ValueDesc::Int(track.sampling_frequency as i64)),
                    ("channels", ValueDesc::Int(track.channels as i64)),
                ],
            );
        } else if is_video_family(family) && track.pixel_width != 0 && track.pixel_height != 0 {
            // Colorimetry from the Colour element (RFC 9559 §5.1.4.1.31; values are
            // H.273 code points), mapped to the pipeline's categorical names. Only
            // announced when declared — absent fields stay unspecified and the
            // renderer defaults by resolution.
            let mut fields: Vec<(&str, ValueDesc)> = vec![
                ("width", ValueDesc::Int(track.pixel_width as i64)),
                ("height", ValueDesc::Int(track.pixel_height as i64)),
            ];
            use crate::color_map::*;
            if let Some(m) = h273_matrix_name(track.colour_matrix) {
                fields.push(("matrix", ValueDesc::Id(m)));
            }
            if let Some(r) = mkv_range_name(track.colour_range) {
                fields.push(("range", ValueDesc::Id(r)));
            }
            if let Some(t) = h273_transfer_name(track.colour_transfer) {
                fields.push(("transfer", ValueDesc::Id(t)));
            }
            if let Some(p) = h273_primaries_name(track.colour_primaries) {
                fields.push(("primaries", ValueDesc::Id(p)));
            }
            ctx.announce_format(pad, family, &fields);
        } else {
            ctx.announce_format(pad, family, &[]);
        }
    }
}

/// Whether a family names a video codec the demuxer routes (spec: RFC 9559 §12 codec
/// mappings). Used to decide whether a `width`/`height` announcement is meaningful.
fn is_video_family(family: &str) -> bool {
    matches!(family, "vp8" | "vp9" | "av1" | "h264/annexb" | "h265/annexb")
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

    // COLD: once-per-stream track discovery + pad creation (topology settles at preroll).
    #[allow(clippy::disallowed_methods)]
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
            let family = codec::family_for(&track.codec_id);
            let (codec_head, reframer) = Self::head_and_reframer(track);
            let name = format!("src_track{}", track.track_number);
            // Per-track offer menu (see `codec::offers_for`): the pad admits exactly
            // its codec's family (+ the bytes escape), so link-time negotiation
            // *selects* the right decoder instead of admitting them all.
            let pad = ctx.add_pad(Direction::Src, &name, codec::offers_for(&track.codec_id));
            log!(
                &*ctx,
                Level::Debug,
                "track",
                number = track.track_number,
                family = family,
                width = track.pixel_width,
                height = track.pixel_height,
            );
            self.pad_tracks.push(PadTrack {
                track_number: track.track_number,
                pad,
                family,
                codec_head,
                reframer,
                track: track.clone(),
                started: false,
                is_video: is_video_family(family),
                awaiting_keyframe: false,
            });
        }
        Ok(())
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // Fresh streaming reader; the whole stream (header + clusters) arrives on the sink pad.
        self.reader = MatroskaReader::new();
        for pt in &mut self.pad_tracks {
            pt.started = false;
            let (head, reframer) = Self::head_and_reframer(&pt.track);
            pt.codec_head = head;
            pt.reframer = reframer;
        }
        self.scratch.clear();
        self.reframe_buf.clear();
        self.pending.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Backpressure: resume any parked emission first, and consume input only while
        // emission keeps up. Un-popped input stays buffered for the next pass, which is
        // what makes the scheduler's gates (not this element's memory) absorb a fast
        // source racing a slow decoder.
        if !self.drain(ctx) {
            return Ok(Flow::Ok);
        }
        while let Some(buf) = inputs.pop() {
            // Feed the bytes to the structure reader; it parses whole elements and queues the
            // frames it decodes. A malformed stream errors here (never panics). `buf` recycles
            // on drop at the end of this iteration.
            self.reader
                .push(buf.memory.data())
                .map_err(|e| Error::Resource(format!("mkvdemux: parse error: {e:?}")))?;
            // Segment Info declared a presentation duration: post it once — a transport
            // UI acts on it, and it can't ride negotiated formats on a playback path
            // (decoders don't declare the `duration` field, so fixation drops it).
            if !self.posted_duration {
                if let Some(ns) = self.reader.duration_ns() {
                    self.posted_duration = true;
                    let element = ctx.element();
                    ctx.post(BusMessage::DurationChanged { element, ns });
                }
            }
            if !self.drain(ctx) {
                return Ok(Flow::Ok);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // On EOS, flush everything still parked or queued — exact-size allocations, since
            // waiting for pool slots is no longer an option. The Matroska streamed masters
            // (Segment/Cluster) end implicitly, so there is no trailer beyond this.
            Event::Eos => self.flush_exact(ctx),
            // Seek (spec: flush/seek). The byte source (filesrc) has already resumed reads at
            // the seek target byte; here the demuxer resets its *parse* state to match. We do
            // NOT rebuild the reader (`::new` would forget the discovered tracks/scale, which
            // the post-seek stream does not resend) — `resync_streaming` keeps that state and
            // scans forward for the next Cluster, tolerating a proportional seek that lands
            // mid-block. pts recover automatically: each post-seek Cluster's Timestamp
            // re-establishes the base, and a block's pts is always `cluster_base + rel_ts`.
            Event::FlushStart => {
                self.reader.resync_streaming();
                // Drop any staged, pre-seek output — those are buffers for content now behind us.
                self.pending.clear();
                for pt in &mut self.pad_tracks {
                    // Re-emit the codec head after the seek: downstream decoders reset on
                    // FlushStart and the NAL codecs (h264/h265) need their SPS/PPS again to
                    // decode the first post-seek keyframe. `started = false` makes `drain`
                    // re-announce + re-emit the head before the next frame; restore the head
                    // and reframer (`start()`/first-frame took the head via `mem::take`).
                    pt.started = false;
                    let (head, reframer) = Self::head_and_reframer(&pt.track);
                    pt.codec_head = head;
                    pt.reframer = reframer;
                    // Gate video tracks until their first post-seek keyframe (a proportional
                    // seek, and even a cue-indexed one, resumes at a Cluster whose first video
                    // frame is a keyframe — but a mid-block estimate may surface a delta first).
                    pt.awaiting_keyframe = pt.is_video;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: flush any final frame even if `event(Eos)` was not delivered.
        self.flush_exact(ctx);
        self.reader = MatroskaReader::new();
        self.scratch.clear();
        self.reframe_buf.clear();
        self.pending.clear();
        self.posted_duration = false;
    }
}

