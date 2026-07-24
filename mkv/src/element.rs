//! `MkvMux` — the streamcraft **element** wrapping the tested [`MatroskaWriter`]
//! (crate::MatroskaWriter) (spec: `spec/MATROSKA.md`; Writing elements). This is the
//! **single-track** version: one sink pad, one src pad, fitting today's static pad model
//! (spec: Elements and pads). The general multiplexer wants one *dynamic* sink pad per
//! input stream plus fan-in — deferred until the core grows dynamic sink pads (the writer
//! is already N-track ready; see the crate docs, "Not yet"). Everything here is a
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
//! buffer is one encoded frame → one `SimpleBlock` (spec `§simpleblock`); its bytes are
//! pushed downstream chunked to the pool slot, exactly like `OggMux`. **A frame larger
//! than the upstream's pool slot arrives split and would mux as several blocks** — for
//! remuxing, size the demuxer's pool to the largest frame
//! (`Pipeline::set_element_pool`); a frame-preserving emission mode is the documented
//! follow-up. Timestamps come from the buffer PTS (falling back to a synthesised cadence
//! when absent). A block is a keyframe unless the buffer is tagged
//! [`BufferFlags::DELTA`] — untagged means "unknown", and every FLAC frame is
//! independently decodable, so keyframe is the right default; a demuxed video track
//! arrives explicitly tagged either way.
//!
//! ## EOS / finalize (spec: Events — Eos)
//! The Segment and final Cluster are unknown-size streamed masters closed implicitly at end
//! of stream (spec `§sizing`), so there is no tail to flush — [`MatroskaWriter::finalize`]
//! is seek-free. The element still runs it on [`Event::Eos`] (primary) and in [`stop`]
//! (belt-and-braces, idempotent) for symmetry with `OggMux` and so a future trailing
//! element (Cues) has a hook. Any bytes `finalize` were to produce are pushed downstream.

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::BufferFlags;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

use crate::codec::{self, Reframer};
use crate::reader::{MatroskaReader, Track};
use crate::writer::{MatroskaWriter, TrackConfig};

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
static MUX_VIDEO_FIELDS: [FieldDesc; 2] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
];
static MUX_SINK_OFFERS: [OfferDesc; 4] = [
    OfferDesc { family: "flac", fields: &MUX_AUDIO_FIELDS },
    OfferDesc { family: "vp8", fields: &MUX_VIDEO_FIELDS },
    OfferDesc { family: "vp9", fields: &MUX_VIDEO_FIELDS },
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
            track: Some(track),
            setup: Setup::Fixed,
            writer: None,
            done: false,
            header_done: false,
            next_ts_ns: 0,
            frame_dur_ns,
            scratch: Vec::new(),
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
            scratch: Vec::new(),
        }
    }

    /// This element's track config, once known (for tests / introspection). `None` for a
    /// caps-driven muxer that has not seen its announcement yet.
    pub fn track(&self) -> Option<&TrackConfig> {
        self.track.as_ref()
    }

    /// Install `track` and build the writer — the moment a caps-driven muxer becomes
    /// equivalent to a constructed one.
    fn configure(&mut self, mut track: TrackConfig) {
        track.track_number = TRACK;
        self.frame_dur_ns = Self::frame_duration_ns(&track);
        self.writer = Some(MatroskaWriter::new(vec![track.clone()]));
        self.track = Some(track);
        self.setup = Setup::Fixed;
    }

    /// The total length of a complete native FLAC head — `fLaC` magic + every metadata
    /// block (RFC 9639 §8: a 4-octet block header of last-flag(1)+type(7) then a 24-bit
    /// big-endian length, repeated until the last-flag) — or `None` while `b` is still a
    /// proper prefix of one. A wrong magic is a hard error: the announced `flac` stream
    /// does not start with a FLAC head.
    fn flac_head_len(b: &[u8]) -> Result<Option<usize>, Error> {
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
    /// then 3 bits channels−1, then 5 bits bits-per-sample−1).
    fn flac_streaminfo_params(head: &[u8]) -> Result<(f64, u32, u32), Error> {
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
        self.done = true;
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
        // Fixed setup builds its writer here; caps-driven setup builds it when the
        // announcement (and, for flac, the in-band head) arrives.
        if let (Setup::Fixed, Some(track)) = (&self.setup, &self.track) {
            self.writer = Some(MatroskaWriter::new(vec![track.clone()]));
        }
        self.done = false;
        self.header_done = false;
        self.next_ts_ns = 0;
        self.scratch.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
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
                                self.configure(TrackConfig::flac(TRACK, head, rate, channels, bits));
                                continue; // the head is CodecPrivate, never a block
                            }
                        }
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
                    match family.as_str() {
                        // The CodecPrivate is the in-band native head; absorb it first.
                        "flac" => self.setup = Setup::FlacHead(Vec::new()),
                        "vp8" | "vp9" => {
                            let dim = |name: &str| {
                                ctx.field_id(name)
                                    .and_then(|id| f.get(id))
                                    .and_then(|v| match v {
                                        Value::Int(n) if n > 0 => Some(n as u32),
                                        _ => None,
                                    })
                            };
                            let (Some(w), Some(h)) = (dim("width"), dim("height")) else {
                                return Err(Error::Todo(
                                    "mkvmux: video announcement without width/height",
                                ));
                            };
                            let id = if family == "vp8" { "V_VP8" } else { "V_VP9" };
                            self.configure(TrackConfig::video(TRACK, id, Vec::new(), w, h));
                        }
                        // Families we cannot mux *conformantly* yet — loud, with the reason
                        // (see the module docs): av1 needs an av1C CodecPrivate; H.264/H.265
                        // arrive as Annex B but Matroska wants length-prefixed + a config
                        // record.
                        other => {
                            return Err(Error::Resource(format!(
                                "mkvmux: cannot remux family '{other}' yet (av1 needs an \
                                 av1C CodecPrivate; h264/h265 need Annex B → length-prefixed \
                                 reframing + a config record — documented follow-ups)"
                            )));
                        }
                    }
                }
                Setup::Fixed | Setup::FlacHead(_) => {
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

/// Every announce family a demux src pad may carry, so the pad's offer menu contains the
/// family it will announce (a pad can only announce a family it already offered — the
/// vocabulary is interned from these offers at link time). The video families match the
/// codec decoders' sink offers so `mkvdemux.src_track<N> ! vp8dec.sink` (etc.) negotiates
/// (spec: RFC 9559 §12 codec mappings; [`codec::family_for`]). All are unconstrained byte
/// families — the concrete `width`/`height` ride the runtime announcement, not the offer.
static DEMUX_SRC_OFFERS: [OfferDesc; 7] = [
    OfferDesc::any(FAMILY_FLAC),
    OfferDesc::any(FAMILY_BYTES),
    OfferDesc::any("vp8"),
    OfferDesc::any("vp9"),
    OfferDesc::any("av1"),
    OfferDesc::any("h264/annexb"),
    OfferDesc::any("h265/annexb"),
];

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
    /// Emissions stalled on pool exhaustion (spec: the backpressure rule — the pool,
    /// never the heap, bounds a demuxer racing a slow decoder). While non-empty the
    /// demuxer consumes **no input**, so upstream backs up into the scheduler's gates
    /// instead of this element ballooning. Holds at most a codec head + one frame.
    pending: std::collections::VecDeque<Carry>,
}

/// A partially-emitted blob: `bytes[off..]` still needs pool slots on `pad`.
struct Carry {
    pad: PadId,
    pts_ns: Option<u64>,
    flags: BufferFlags,
    bytes: Vec<u8>,
    off: usize,
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
            pending: std::collections::VecDeque::new(),
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
    fn head_and_reframer(track: &Track) -> (Vec<u8>, Reframer) {
        match track.codec_id.as_str() {
            CODEC_A_FLAC => (track.codec_private.clone(), Reframer::Passthrough),
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
    /// must carry the remainder and stop consuming input (spec: the backpressure rule). PTS
    /// and `flags` (the frame's keyframe/delta tag, on every chunk of the frame) are stamped
    /// on each buffer.
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
            debug_assert!(cap > 0, "mkvdemux: zero-capacity pool slot");
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

    /// Emit `bytes[off..]` as one **exact-size** buffer (`alloc_exact` — a right-sized heap
    /// allocation when the pool is dry, never a slot-sized over-allocation). Only for the
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

    /// Queue a blob for emission and try to emit it now; on pool exhaustion the remainder
    /// parks in `pending` (drained first on the next pass).
    fn emit_or_park(
        &mut self,
        ctx: &mut Ctx,
        pad: PadId,
        pts_ns: Option<u64>,
        flags: BufferFlags,
        bytes: Vec<u8>,
    ) {
        let off = Self::emit_bounded(ctx, pad, pts_ns, flags, &bytes, 0);
        if off < bytes.len() {
            self.pending.push_back(Carry { pad, pts_ns, flags, bytes, off });
        }
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
            // Reframe the Block payload to the downstream bytes (before touching pool memory)
            // so a malformed access unit is dropped without ever emitting the codec head/format
            // for a track that would then carry nothing (announce happens only on a good frame).
            let out_bytes = match self.pad_tracks[idx].reframer.reframe_block(&frame.data) {
                Ok(bytes) => bytes,
                Err(e) => {
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
            // First frame on this pad: emit the native codec head and announce the format.
            if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let (pad, family) = (self.pad_tracks[idx].pad, self.pad_tracks[idx].family);
                Self::announce(ctx, pad, family, &self.pad_tracks[idx].track);
                // The head carries the stream start; stamp it with the first frame's pts. Take
                // it (leaving empty) — it is emitted exactly once, before the first frame.
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                self.emit_or_park(ctx, pad, Some(frame.pts_ns), BufferFlags::empty(), head);
            }
            let pad = self.pad_tracks[idx].pad;
            // The container's keyframe bit rides the buffer (spec: Buffer flags) so a
            // remuxing consumer preserves seekability; `DELTA` is explicit — an untagged
            // buffer would read as "unknown", which a muxer defaults to keyframe.
            let flags = if frame.keyframe { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            // Emit from the (usually borrowed) reframed bytes; only a parked remainder
            // pays the copy into an owned carry.
            let off = Self::emit_bounded(ctx, pad, Some(frame.pts_ns), flags, &out_bytes, 0);
            if off < out_bytes.len() {
                self.pending.push_back(Carry {
                    pad,
                    pts_ns: Some(frame.pts_ns),
                    flags,
                    bytes: out_bytes.into_owned(),
                    off,
                });
            }
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
            Self::emit_exact(ctx, c.pad, c.pts_ns, c.flags, &c.bytes, c.off);
        }
        // Reuse drain()'s routing/announce logic by swapping the emitter is overkill for
        // the tail; inline the same walk with exact emission.
        while let Some(frame) = self.reader.next_frame() {
            let Some(idx) = self.pad_tracks.iter().position(|pt| pt.track_number == frame.track_number) else {
                continue;
            };
            let out_bytes = match self.pad_tracks[idx].reframer.reframe_block(&frame.data) {
                Ok(bytes) => bytes,
                Err(_) => continue, // tail flush: drop silently rather than warn-spam
            };
            if !self.pad_tracks[idx].started {
                self.pad_tracks[idx].started = true;
                let (pad, family) = (self.pad_tracks[idx].pad, self.pad_tracks[idx].family);
                Self::announce(ctx, pad, family, &self.pad_tracks[idx].track);
                let head = std::mem::take(&mut self.pad_tracks[idx].codec_head);
                Self::emit_exact(ctx, pad, Some(frame.pts_ns), BufferFlags::empty(), &head, 0);
            }
            let pad = self.pad_tracks[idx].pad;
            let flags = if frame.keyframe { BufferFlags::KEYFRAME } else { BufferFlags::DELTA };
            Self::emit_exact(ctx, pad, Some(frame.pts_ns), flags, &out_bytes, 0);
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
        } else if is_video_family(family) && track.pixel_width != 0 && track.pixel_height != 0 {
            ctx.announce_format(
                pad,
                family,
                &[
                    ("width", ValueDesc::Int(track.pixel_width as i64)),
                    ("height", ValueDesc::Int(track.pixel_height as i64)),
                ],
            );
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
            let pad = ctx.add_pad(Direction::Src, &name, &DEMUX_SRC_OFFERS);
            self.pad_tracks.push(PadTrack {
                track_number: track.track_number,
                pad,
                family,
                codec_head,
                reframer,
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
            let (head, reframer) = Self::head_and_reframer(&pt.track);
            pt.codec_head = head;
            pt.reframer = reframer;
        }
        self.scratch.clear();
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
            if !self.drain(ctx) {
                return Ok(Flow::Ok);
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // On EOS, flush everything still parked or queued — exact-size allocations, since
        // waiting for pool slots is no longer an option. The Matroska streamed masters
        // (Segment/Cluster) end implicitly, so there is no trailer beyond this.
        if matches!(event, Event::Eos) {
            self.flush_exact(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: flush any final frame even if `event(Eos)` was not delivered.
        self.flush_exact(ctx);
        self.reader = MatroskaReader::new();
        self.scratch = Vec::new();
        self.pending.clear();
    }
}
