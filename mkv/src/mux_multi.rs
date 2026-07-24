//! `MkvMuxN` — the **multi-track** Matroska muxer element: one sink pad per input track,
//! all tracks interleaved into one MKV byte stream (spec: `spec/MATROSKA.md`; RFC 9559).
//! The single-track [`MkvMux`](crate::MkvMux) wraps the same N-track-ready
//! [`MatroskaWriter`]; this element is the fan-in front the writer was built for — an
//! **Active** aggregator head whose sink pads the scheduler feeds from one upstream ring
//! each (spec: Aggregation; see `elements/tests/aggregation.rs` for the pattern).
//!
//! ## Static sink pads (the dynamic-pad fallback, documented)
//! The pads are a **static menu** `sink_0`..`sink_7` ([`MAX_TRACKS`]); the app links as
//! many as it has tracks and the element discovers the linked set at `start()` (a linked
//! pad has a negotiated format, an unlinked one has none). Dynamic sink pads
//! (`ctx.add_pad(Direction::Sink, ..)` in `preroll`) were the intended shape, but a
//! runtime `FormatChange` arriving on a dynamic sink pad panics in the scheduler today:
//! `deliver_events` re-validates against `elem.desc().pads[pad.0 as usize].offers`, and a
//! dynamic pad's id indexes past the static pad array. Until the core consults the
//! dynamic-pad offer table there, static pads are the safe equivalent (the linked subset
//! behaves identically). **Pad i → track i+1** in linked-pad order.
//!
//! ## Per-pad caps-driven track setup
//! Each linked pad runs the same state machine [`MkvMux::from_caps`] runs on its one pad:
//! the upstream `FormatChange` picks the family (`h264/avcc`/`h265/hvcc` → the first
//! buffer is the raw config record taken verbatim as `CodecPrivate`, RFC 9559 §12; `aac`
//! → the first buffer is the raw **AudioSpecificConfig**, the A_AAC `CodecPrivate` —
//! frames are raw AAC access units, exactly MP4's storage and exactly what Matroska
//! wants, no ADTS; `flac` → the in-band `fLaC` head is absorbed; `vp8`/`vp9` → built
//! from the announced fields). `event()` carries no pad id, so the pad an announcement
//! belongs to is recovered from the scheduler's contract: the announced format is
//! installed on the arrival pad (`set_negotiated_one`) *before* `event()` runs, so the
//! awaiting pad whose negotiated format equals the announced one is the target (ties —
//! byte-identical announcements on several awaiting pads — resolve to the lowest pad,
//! which is harmless: identical announcements build identical states, and each pad's
//! CodecPrivate still arrives as *its own* first buffer). The wished-for core API is an
//! explicit pad id on `event` — documented in the crate report.
//!
//! ## Header, interleave, and EOS
//! The header needs every track's `TrackConfig`, so it is written once **all** linked
//! pads are configured; frames arriving on already-configured pads queue until then,
//! bounded by [`PRECONFIG_MAX_BYTES`]/[`PRECONFIG_MAX_BUFS`] — a linked pad that never
//! announces while others stream is a loud error, never a silent hang. After the header,
//! frames are fed to the writer **globally pts-ordered across pads**: pop from the pad
//! whose head-of-queue timestamp is smallest (ties → lowest pad); an empty-but-open pad
//! blocks emission (it may still deliver an earlier pts — the fan-in head's natural
//! wait), a closed pad is skipped once drained. Order *within* a pad is never changed,
//! so a video track's decode-order (B-frame) sequence survives; only its head timestamp
//! competes in the merge. Queue growth while blocked is bounded by the upstream pools:
//! queued frames hold refcounted slices of the source's pool slots, so a lopsided
//! producer stalls its own source (the standing backpressure loop). The Cluster cadence
//! is the writer's: the **anchor track** — the first linked pad whose family is a video
//! family, falling back to the first linked pad when no track is video (tie-break:
//! lowest pad index) — opens a Cluster on each keyframe; audio frames (all keyframes by
//! the DELTA convention) never do. `Info\Duration` is the max over the pads' announced
//! `duration` fields. On EOS every queued frame drains in the same pts order, the final
//! Cluster is emitted sized, and the stream finalizes (spec `§sizing` — seek-free).
//!
//! Output follows `MkvMux` exactly: scatter-mode staging (ZERO-COPY.md Stage 2), header
//! byte runs through pooled slots (`try_alloc` + carry — the backpressure discipline),
//! frame payloads forwarded as refcounts, never copied.

use std::collections::VecDeque;

use streamcraft_core::batch::Inputs;
use streamcraft_core::buffer::{Buffer, BufferFlags};
use streamcraft_core::bus::BusMessage;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    AlignBy, Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{
    ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc,
};
use streamcraft_core::id::{FormatId, PadId};
use streamcraft_core::memory::Memory;
use streamcraft_core::time::Timestamp;

use crate::element::MkvMux;
use crate::writer::{MatroskaWriter, MuxOut, MuxPiece, TrackConfig};

/// The sink-pad menu size — the most input tracks one `MkvMuxN` can mux. A static bound
/// because the pads are static (see the module docs for why dynamic sink pads are not
/// usable yet); raise it if a real container ever carries more remuxable tracks.
pub const MAX_TRACKS: usize = 8;

/// The src pad's local index — right after the [`MAX_TRACKS`] sink pads.
const SRC: PadId = PadId(MAX_TRACKS as u32);

/// Pre-header queue bounds: frames buffered on configured pads while some linked pad has
/// not announced yet. Generous enough for any sane interleave (a whole GOP of 4K video
/// fits many times over), tight enough that a linked-but-silent pad turns into a loud
/// error long before memory pressure: exceeding either bound while a pad is still
/// unconfigured fails the pipeline with the offending pads named.
pub const PRECONFIG_MAX_BYTES: usize = 64 << 20;
pub const PRECONFIG_MAX_BUFS: usize = 4096;

/// Payload pieces at or below this size are **copied** into the open coalescing slot in
/// [`MkvMuxN::drain_out`] instead of forwarded as their own refcounted buffer. One
/// downstream buffer is one sink write submission, so a ~700-octet AAC frame as its own
/// buffer costs a positioned write *each* — the movie gate measured sys-time-dominated
/// exactly that way. A page-ish bound keeps every video frame on the zero-copy path
/// (ZERO-COPY.md Stage 2) while audio/tiny frames batch into full slots.
const COALESCE_MAX: usize = 4096;

/// Raw `bytes` on the src side: a Matroska byte stream.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

/// Sink offers: every family this muxer can build a track from, **with every field/value
/// name its announcements carry** — declaring them is what interns the names the
/// upstream's announcement resolves against (spec: Formats — dynamic caps; an
/// announcement with an un-interned name is dropped whole by `build_fixed`). Mirrors the
/// single-track menu plus `aac`. No `bytes` fallback: a byte stream with no announced
/// family could never become a Matroska track here.
static MUXN_SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static MUXN_AUDIO_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&MUXN_SAMPLE_VALUES), preferred: None },
];
/// AAC announcements: rate/channels plus the optional presentation `duration` (ns) a
/// remuxing demuxer knows from its tables — folded into `Info\Duration` (max over pads).
static MUXN_AAC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "duration", allowed: ConstraintDesc::Any, preferred: None },
];
static MUXN_VIDEO_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "width", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "height", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "duration", allowed: ConstraintDesc::Any, preferred: None },
];
static MUXN_SINK_OFFERS: [OfferDesc; 6] = [
    OfferDesc { family: "flac", fields: &MUXN_AUDIO_FIELDS },
    OfferDesc { family: "vp8", fields: &MUXN_VIDEO_FIELDS },
    OfferDesc { family: "vp9", fields: &MUXN_VIDEO_FIELDS },
    OfferDesc { family: "h264/avcc", fields: &MUXN_VIDEO_FIELDS },
    OfferDesc { family: "h265/hvcc", fields: &MUXN_VIDEO_FIELDS },
    OfferDesc { family: "aac", fields: &MUXN_AAC_FIELDS },
];

/// One sink `PadDesc` of the static menu (all identical but for the name).
const fn sink_pad(name: &'static str) -> PadDesc {
    PadDesc {
        name,
        direction: Direction::Sink,
        offers: &MUXN_SINK_OFFERS,
        dynamic: false,
        validate: None,
    }
}

static MUXN_PADS: [PadDesc; MAX_TRACKS + 1] = [
    sink_pad("sink_0"),
    sink_pad("sink_1"),
    sink_pad("sink_2"),
    sink_pad("sink_3"),
    sink_pad("sink_4"),
    sink_pad("sink_5"),
    sink_pad("sink_6"),
    sink_pad("sink_7"),
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
];

static MUXN_DESC: ElementDesc = ElementDesc {
    name: "mkvmuxn",
    pads: &MUXN_PADS,
    props: &[],
    // Active: a fan-in aggregator is its own group head, reading one upstream ring per
    // linked sink pad (spec: Aggregation).
    sched: SchedHint::Active,
    inputs: InputPolicy::All { by: AlignBy::RunningTime },
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(MkvMuxN::new())),
};

/// A linked sink pad's caps-driven setup — the per-pad twin of `MkvMux`'s `Setup`.
enum LaneState {
    /// Waiting for the upstream `FormatChange`. Data before any announcement is a loud
    /// error — a caps-driven muxer cannot guess its codec.
    AwaitCaps,
    /// `flac` announced: absorbing the leading in-band native head (`fLaC` + metadata
    /// blocks, RFC 9559 A_FLAC mapping) into the `CodecPrivate`.
    FlacHead(Vec<u8>),
    /// `h264/avcc`/`h265/hvcc` announced: the **first buffer** is the raw `avcC`/`hvcC`
    /// record, taken verbatim as `CodecPrivate` (RFC 9559 §12).
    NalHead { codec_id: &'static str, width: u32, height: u32 },
    /// `aac` announced: the **first buffer** is the raw AudioSpecificConfig (ISO/IEC
    /// 14496-3 §1.6.2.1) — the A_AAC `CodecPrivate` (RFC 9559 §12).
    AacHead { rate: u32, channels: u32 },
    /// Track config known; frames queue/mux from here on.
    Ready(TrackConfig),
}

/// One linked sink pad: its id, assigned track number, setup state, and the pts-ordered
/// frame queue the global merge pops from.
struct Lane {
    /// The static pad this lane reads (`sink_{i}`).
    pad: PadId,
    /// The Matroska `TrackNumber` (linked-pad position + 1 — "pad i → track i+1").
    track: u64,
    state: LaneState,
    /// Frames waiting for the merge, each with its effective timestamp (ns) — arrival
    /// order preserved (decode order for video; the merge only compares queue *heads*).
    queue: VecDeque<(u64, Buffer)>,
    /// Synthesised-timestamp cadence for PTS-less buffers (mirrors `MkvMux`): the next
    /// frame's ns, advanced by `frame_dur_ns` per frame and kept ahead of real PTS.
    next_ts_ns: u64,
    frame_dur_ns: u64,
}

impl Lane {
    /// The effective timestamp for an arriving buffer: its PTS, else the per-lane
    /// synthesised cadence (kept ahead of real PTS so a PTS-less frame never travels
    /// backwards) — computed at enqueue so the merge compares settled values.
    fn effective_ts(&mut self, buf: &Buffer) -> u64 {
        match buf.pts.nanos() {
            Some(t) => {
                self.next_ts_ns = t + self.frame_dur_ns;
                t
            }
            None => {
                let t = self.next_ts_ns;
                self.next_ts_ns += self.frame_dur_ns;
                t
            }
        }
    }
}

/// The multi-track Matroska muxer element (see the module docs). Construct with
/// [`MkvMuxN::new`] (linked-pad count discovered at `start()`) or [`MkvMux::multi`] (an
/// expected track count, validated loudly at `start()`), link `sink_0..sink_{n-1}` and
/// `src`, and feed each pad one encoded frame per buffer.
pub struct MkvMuxN {
    /// Expected linked-pad count ([`MkvMux::multi`]); `None` accepts whatever is linked.
    expect: Option<usize>,
    /// The linked pads in pad order, discovered at `start()`.
    lanes: Vec<Lane>,
    /// The writer — built once every lane is `Ready` (the header needs all tracks).
    writer: Option<MatroskaWriter>,
    header_done: bool,
    /// Finalized: late buffers are dropped, never remuxed into a reopened stream.
    done: bool,
    /// Max announced `duration` (ns) across pads → `Info\Duration`.
    duration_ns: u64,
    /// Pre-header queue accounting (see [`PRECONFIG_MAX_BYTES`]).
    prebuf_bytes: usize,
    prebuf_bufs: usize,
    /// Scatter output staging + partial-emission cursor + probed out-format — exactly
    /// `MkvMux`'s trio (see its field docs).
    out: MuxOut,
    out_off: usize,
    out_fmt: FormatId,
}

impl MkvMux {
    /// A multi-track muxer expecting exactly `n` linked sink pads (`sink_0..sink_{n-1}`,
    /// track numbers 1..=n). The count is validated at `start()` — a mislinked pipeline
    /// fails loudly instead of hanging on a never-fed pad. See [`MkvMuxN`].
    pub fn multi(n: usize) -> MkvMuxN {
        let mut m = MkvMuxN::new();
        m.expect = Some(n);
        m
    }
}

impl Default for MkvMuxN {
    fn default() -> Self {
        Self::new()
    }
}

impl MkvMuxN {
    /// A multi-track muxer for however many sink pads the app links (at least one).
    pub fn new() -> Self {
        Self {
            expect: None,
            lanes: Vec::new(),
            writer: None,
            header_done: false,
            done: false,
            duration_ns: 0,
            prebuf_bytes: 0,
            prebuf_bufs: 0,
            out: MuxOut::new(),
            out_off: 0,
            out_fmt: FormatId(0), // probed at `start`
        }
    }

    /// The track configs so far, in track-number order (tests / introspection): `None`
    /// for a lane still awaiting its caps/head.
    pub fn tracks(&self) -> Vec<Option<&TrackConfig>> {
        self.lanes
            .iter()
            .map(|l| match &l.state {
                LaneState::Ready(cfg) => Some(cfg),
                _ => None,
            })
            .collect()
    }

    // --- output drain (the `MkvMux` scatter discipline, verbatim but for `SRC`) ---

    /// Push a payload piece downstream as one whole buffer — a refcount move, no pool
    /// slot, no copy (ZERO-COPY.md Stage 2; see `MkvMux::emit_payload`).
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

    /// Drain staged output pieces downstream in order — pool-bounded
    /// ([`Ctx::try_alloc`] + the `out_off` carry; `false` = pool dry, stop consuming
    /// input). Two piece classes:
    ///
    /// - **Large payloads** (> [`COALESCE_MAX`]) forward as refcounts — the video hot
    ///   path, never copied (ZERO-COPY.md Stage 2).
    /// - **Byte runs and small payloads** *coalesce*: they pack into one open pool slot
    ///   until it fills or a large payload must interleave (order is preserved — the
    ///   open slot flushes first). This is what keeps a muxed audio track from becoming
    ///   one downstream buffer — and one sink write submission — **per ~700-octet AAC
    ///   frame**: the movie gate went syscall-bound (sys ≫ user) exactly that way.
    ///   Copying a sub-4-KiB frame costs far less than its own positioned write, and
    ///   releases its retained input slot immediately.
    fn drain_out(&mut self, ctx: &mut Ctx) -> bool {
        // The open coalescing slot, filled to `memory.len()` so far.
        let mut slot: Option<Buffer> = None;
        // Flush helper: push the open slot downstream if it holds anything.
        fn flush(ctx: &mut Ctx, slot: &mut Option<Buffer>) {
            if let Some(buf) = slot.take() {
                if !buf.memory.is_empty() {
                    ctx.out(SRC).push(buf);
                }
            }
        }
        loop {
            match self.out.front() {
                None => break,
                Some(&MuxPiece::Bytes { start, end }) => {
                    let mut at = start + self.out_off;
                    while at < end {
                        if slot.as_ref().is_some_and(|b| b.memory.len() == b.memory.capacity()) {
                            flush(ctx, &mut slot);
                        }
                        if slot.is_none() {
                            let Some(buf) = ctx.try_alloc(SRC) else {
                                self.out_off = at - start;
                                return false; // pool dry — resume here next pass
                            };
                            debug_assert!(buf.memory.capacity() > 0, "zero-capacity slot");
                            slot = Some(buf);
                        }
                        let buf = slot.as_mut().expect("slot open");
                        let filled = buf.memory.len();
                        let n = (buf.memory.capacity() - filled).min(end - at);
                        buf.memory.as_mut_full()[filled..filled + n]
                            .copy_from_slice(&self.out.header_bytes()[at..at + n]);
                        buf.memory.set_len(filled + n);
                        at += n;
                    }
                    self.out_off = 0;
                    self.out.pop_front();
                }
                Some(MuxPiece::Payload(mem)) => {
                    let len = mem.len();
                    // Small payload that fits a slot: coalesce by copy (whole-piece —
                    // never split, so the carry state stays the byte-run cursor only).
                    let coalesce = len <= COALESCE_MAX
                        && slot.as_ref().map_or(true, |b| b.memory.capacity() >= len);
                    if coalesce {
                        if slot.as_ref().is_some_and(|b| b.memory.capacity() - b.memory.len() < len)
                        {
                            flush(ctx, &mut slot);
                        }
                        if slot.is_none() {
                            let Some(buf) = ctx.try_alloc(SRC) else {
                                flush(ctx, &mut slot);
                                return false; // piece stays queued; resume next pass
                            };
                            slot = Some(buf);
                        }
                        let buf = slot.as_mut().expect("slot open");
                        if buf.memory.capacity() >= len {
                            let filled = buf.memory.len();
                            buf.memory.as_mut_full()[filled..filled + len]
                                .copy_from_slice(mem.data());
                            buf.memory.set_len(filled + len);
                            self.out.pop_front();
                            continue;
                        }
                        // A fresh slot smaller than even this small piece (odd pool
                        // config): fall through to the refcount path.
                    }
                    // Large payload: the open slot flushes first (order), then the
                    // `Memory` moves out as its own buffer — a refcount, no copy.
                    flush(ctx, &mut slot);
                    let Some(MuxPiece::Payload(mem)) = self.out.pop_front() else {
                        unreachable!("front was a payload")
                    };
                    Self::emit_payload(ctx, mem, self.out_fmt);
                }
            }
        }
        flush(ctx, &mut slot);
        self.out.reclaim();
        true
    }

    /// The EOS drain: exact-size allocations for the byte runs — waiting for a pool slot
    /// is no longer an option at end of stream (see `MkvMux::drain_out_exact`).
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

    // --- per-pad setup + queueing ---

    /// Feed one arriving buffer to its lane: pre-config buffers drive the setup state
    /// machine (heads become `CodecPrivate`, never blocks); configured lanes enqueue the
    /// frame with its effective timestamp for the merge, bounded pre-header.
    fn accept(&mut self, lane_idx: usize, buf: Buffer) -> Result<(), Error> {
        if self.done {
            return Ok(()); // late buffer after finalize: dropped (matches `MkvMux`)
        }
        let track = self.lanes[lane_idx].track;
        let lane = &mut self.lanes[lane_idx];
        match &mut lane.state {
            LaneState::Ready(_) => {
                if !self.header_done {
                    self.prebuf_bytes += buf.memory.len();
                    self.prebuf_bufs += 1;
                    if self.prebuf_bytes > PRECONFIG_MAX_BYTES
                        || self.prebuf_bufs > PRECONFIG_MAX_BUFS
                    {
                        let waiting: Vec<String> = self
                            .lanes
                            .iter()
                            .filter(|l| !matches!(l.state, LaneState::Ready(_)))
                            .map(|l| format!("sink_{}", l.pad.0))
                            .collect();
                        return Err(Error::Resource(format!(
                            "mkvmuxn: {} buffers / {} bytes queued but pad(s) {} never \
                             announced a format — the header needs every linked track \
                             (unlink dead pads or fix the upstream announcement)",
                            self.prebuf_bufs,
                            self.prebuf_bytes,
                            waiting.join(", "),
                        )));
                    }
                }
                let lane = &mut self.lanes[lane_idx];
                let ts = lane.effective_ts(&buf);
                lane.queue.push_back((ts, buf));
                Ok(())
            }
            LaneState::AwaitCaps => Err(Error::Todo(
                "mkvmuxn: data before any format announcement on a linked pad — a \
                 caps-driven muxer cannot guess its codec",
            )),
            LaneState::FlacHead(head) => {
                head.extend_from_slice(buf.memory.data());
                match MkvMux::flac_head_len(head)? {
                    None => Ok(()), // head still incomplete — keep absorbing
                    Some(len) => {
                        if len != head.len() {
                            // The head must end on a buffer boundary (see `MkvMux`).
                            return Err(Error::Todo(
                                "mkvmuxn: flac head not on a buffer boundary",
                            ));
                        }
                        let head = std::mem::take(head);
                        let (rate, ch, bits) = MkvMux::flac_streaminfo_params(&head)?;
                        Self::lane_ready(lane, TrackConfig::flac(track, head, rate, ch, bits));
                        Ok(())
                    }
                }
            }
            LaneState::NalHead { codec_id, width, height } => {
                // First buffer = the raw config record, verbatim (RFC 9559 §12).
                // configurationVersion is 1 for avcC and hvcC both — a cheap guard
                // against being fed sample data first.
                let head = buf.memory.data().to_vec();
                if head.first() != Some(&1) {
                    return Err(Error::Todo(
                        "mkvmuxn: leading buffer is not an avcC/hvcC record \
                         (configurationVersion != 1)",
                    ));
                }
                let (codec_id, w, h) = (*codec_id, *width, *height);
                Self::lane_ready(lane, TrackConfig::video(track, codec_id, head, w, h));
                Ok(())
            }
            LaneState::AacHead { rate, channels } => {
                // First buffer = the raw AudioSpecificConfig (ISO/IEC 14496-3 §1.6.2.1):
                // ≥ 2 octets (5-bit audioObjectType + 4-bit samplingFrequencyIndex +
                // 4-bit channelConfiguration = 13 bits minimum).
                let head = buf.memory.data().to_vec();
                if head.len() < 2 {
                    return Err(Error::Todo(
                        "mkvmuxn: leading aac buffer too short to be an \
                         AudioSpecificConfig",
                    ));
                }
                let (rate, channels) = (*rate, *channels);
                Self::lane_ready(
                    lane,
                    TrackConfig::audio(track, "A_AAC", head, rate as f64, channels, 0),
                );
                Ok(())
            }
        }
    }

    /// Install a lane's finished [`TrackConfig`] (and its synth cadence).
    fn lane_ready(lane: &mut Lane, cfg: TrackConfig) {
        lane.frame_dur_ns = MkvMux::frame_duration_ns(&cfg);
        lane.state = LaneState::Ready(cfg);
    }

    /// Build the writer + write the header once **every** lane is configured (the
    /// Tracks element needs them all). No-op until then — and after (the `header_done`
    /// latch also keeps a finalized stream from ever re-emitting a header).
    fn maybe_write_header(&mut self) {
        if self.header_done
            || !self.lanes.iter().all(|l| matches!(l.state, LaneState::Ready(_)))
        {
            return;
        }
        self.build_writer();
    }

    /// Build the writer from every `Ready` lane. Config order picks the **anchor**: the
    /// first video lane in pad order leads (its keyframes open Clusters — the typical
    /// Matroska cadence); with no video track the first lane leads. Tie-break is pad
    /// order throughout. Track *numbers* are untouched — `TrackEntry` order in the
    /// header is not track-number order, which RFC 9559 permits.
    fn build_writer(&mut self) {
        let mut configs: Vec<TrackConfig> = self
            .lanes
            .iter()
            .filter_map(|l| match &l.state {
                LaneState::Ready(cfg) => Some(cfg.clone()),
                _ => None,
            })
            .collect();
        if configs.is_empty() {
            return;
        }
        if let Some(anchor) = configs.iter().position(|c| c.video.is_some()) {
            let a = configs.remove(anchor);
            configs.insert(0, a);
        }
        let mut writer = MatroskaWriter::new(configs);
        if self.duration_ns > 0 {
            writer.set_duration_ns(self.duration_ns);
        }
        // The only error is BadTracks — impossible for ≥1 validated, unique-numbered
        // tracks built above.
        let _ = writer.write_header_scatter(&mut self.out);
        self.header_done = true;
        self.writer = Some(writer);
    }

    // --- the global merge ---

    /// Feed queued frames to the writer in **global timestamp order across pads**: pop
    /// the smallest head-of-queue timestamp (ties → the earliest lane in pad order);
    /// stop when an open pad's queue is empty (it may still deliver an earlier pts —
    /// the wait *is* the fan-in backpressure) or, mid-stream, when the pool runs dry
    /// mid-drain. `at_eos` drains everything regardless (no pad can deliver more, and
    /// byte runs flush exact-size afterwards).
    fn merge(&mut self, ctx: &mut Ctx, at_eos: bool) {
        if self.writer.is_none() {
            return; // header pending — frames stay queued (bounded; see `accept`)
        }
        loop {
            let mut best: Option<(usize, u64)> = None;
            for (i, lane) in self.lanes.iter().enumerate() {
                match lane.queue.front() {
                    Some(&(ts, _)) => {
                        if best.is_none_or(|(_, bts)| ts < bts) {
                            best = Some((i, ts));
                        }
                    }
                    None => {
                        if !at_eos && !ctx.is_pad_closed(lane.pad) {
                            return; // an open pad may still deliver an earlier pts
                        }
                    }
                }
            }
            let Some((i, ts)) = best else { return }; // every queue empty
            let (_, buf) = self.lanes[i].queue.pop_front().expect("head just seen");
            // A block is a keyframe unless explicitly tagged DELTA (the untagged norm
            // for all-independent-frame codecs like AAC/FLAC; demuxed video arrives
            // tagged either way).
            let keyframe = !buf.flags.contains(BufferFlags::DELTA);
            let track = self.lanes[i].track;
            let writer = self.writer.as_mut().expect("writer present");
            // One buffer == one frame == one SimpleBlock, staged scatter-style; the only
            // error is an unknown track, impossible for a lane-built config.
            let _ = writer.write_frame_scatter(&mut self.out, track, ts, buf.memory, keyframe);
            if !at_eos && !self.drain_out(ctx) {
                return; // pool dry mid-drain — resume next pass
            }
        }
    }

    /// Finalize exactly once (idempotent via the writer `Option`): drain every queued
    /// frame in timestamp order, emit the final sized Cluster, flush exact-size. If some
    /// lane never announced, its (necessarily empty — data pre-announce is an error)
    /// track is dropped with a Bus warning and the configured tracks still finalize.
    fn finish_stream(&mut self, ctx: &mut Ctx) {
        self.done = true;
        // `header_done` gates the build: a second call (stop after Eos) must not
        // rebuild the taken writer and emit a second header into a finalized stream.
        if self.writer.is_none() && !self.header_done {
            let waiting: Vec<String> = self
                .lanes
                .iter()
                .filter(|l| !matches!(l.state, LaneState::Ready(_)))
                .map(|l| format!("sink_{}", l.pad.0))
                .collect();
            if !waiting.is_empty() && self.lanes.len() > waiting.len() {
                let element = ctx.element();
                ctx.post(BusMessage::Warning {
                    element,
                    error: Error::Element {
                        element,
                        message: format!(
                            "mkvmuxn: pad(s) {} never announced a format — their tracks \
                             are dropped from the finalized file",
                            waiting.join(", "),
                        ),
                    },
                });
            }
            self.build_writer(); // from the Ready lanes only (no-op when none)
        }
        self.merge(ctx, true);
        if let Some(mut writer) = self.writer.take() {
            writer.finalize_scatter(&mut self.out);
        }
        self.drain_out_exact(ctx);
    }

    /// Handle a runtime format announcement: attribute it to a pad (see the module docs
    /// — the scheduler installed it on the arrival pad's negotiated slot before this
    /// call), then run the family dispatch `MkvMux::from_caps` runs, per pad.
    fn on_format_change(&mut self, ctx: &mut Ctx, f: &FixedFormat) -> Result<(), Error> {
        let matches = |l: &Lane| ctx.negotiated(l.pad).is_some_and(|g| fixed_eq(g, f));
        let Some(idx) = self
            .lanes
            .iter()
            .position(|l| matches!(l.state, LaneState::AwaitCaps) && matches(l))
        else {
            // No awaiting pad matches. A repeat of an already-installed announcement is
            // harmless before the header (a re-validation pass re-delivering the same
            // format) and fatal after (a Matroska track is fixed once written). Anything
            // else is unattributable — loud, because guessing a pad could weld the wrong
            // CodecPrivate to a track.
            if self.lanes.iter().any(|l| !matches!(l.state, LaneState::AwaitCaps) && matches(l)) {
                if self.header_done {
                    return Err(Error::Todo(
                        "mkvmuxn: mid-stream format change cannot be muxed (track is \
                         fixed once the header is written)",
                    ));
                }
                return Ok(());
            }
            return Err(Error::Todo(
                "mkvmuxn: cannot attribute a FormatChange to any linked sink pad \
                 (no awaiting pad's negotiated format matches the announcement)",
            ));
        };

        let family = ctx.family_name(f.family).unwrap_or("?").to_owned();
        let get = |name: &str| {
            ctx.field_id(name)
                .and_then(|id| f.get(id))
                .and_then(|v| match v {
                    Value::Int(n) if n > 0 => Some(n as u32),
                    _ => None,
                })
        };
        // Optional announced presentation duration (ns): `Info\Duration` is the max
        // across pads (the longest track bounds the presentation).
        let duration_ns = ctx
            .field_id("duration")
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Int(n) if n > 0 => Some(n as u64),
                _ => None,
            });
        if let Some(d) = duration_ns {
            self.duration_ns = self.duration_ns.max(d);
        }

        let track = self.lanes[idx].track;
        let lane = &mut self.lanes[idx];
        match family.as_str() {
            // The CodecPrivate is the in-band native head; absorb it first.
            "flac" => lane.state = LaneState::FlacHead(Vec::new()),
            "vp8" | "vp9" => {
                let (Some(w), Some(h)) = (get("width"), get("height")) else {
                    return Err(Error::Todo("mkvmuxn: video announcement without width/height"));
                };
                let id = if family == "vp8" { "V_VP8" } else { "V_VP9" };
                Self::lane_ready(lane, TrackConfig::video(track, id, Vec::new(), w, h));
            }
            // Passthrough NAL remux: the CodecPrivate arrives as the first buffer.
            "h264/avcc" | "h265/hvcc" => {
                let (Some(w), Some(h)) = (get("width"), get("height")) else {
                    return Err(Error::Todo("mkvmuxn: video announcement without width/height"));
                };
                let codec_id = if family == "h264/avcc" {
                    "V_MPEG4/ISO/AVC"
                } else {
                    "V_MPEGH/ISO/HEVC"
                };
                lane.state = LaneState::NalHead { codec_id, width: w, height: h };
            }
            // AAC: rate/channels ride the announcement (Matroska requires
            // SamplingFrequency); the ASC arrives as the first buffer.
            "aac" => {
                let (Some(rate), Some(channels)) = (get("rate"), get("channels")) else {
                    return Err(Error::Todo("mkvmuxn: aac announcement without rate/channels"));
                };
                lane.state = LaneState::AacHead { rate, channels };
            }
            other => {
                return Err(Error::Resource(format!(
                    "mkvmuxn: cannot remux family '{other}' (see mkvmux's family notes; \
                     for h264/h265 use the demuxer's passthrough mode)"
                )));
            }
        }
        Ok(())
    }
}

/// Structural [`FixedFormat`] equality (the type is deliberately not `PartialEq` in
/// core): same family, same field count, every fixed field equal. Order-insensitive.
fn fixed_eq(a: &FixedFormat, b: &FixedFormat) -> bool {
    a.family == b.family
        && a.len() == b.len()
        && a.fields().iter().all(|(id, v)| b.get(*id) == Some(*v))
}

impl Element for MkvMuxN {
    fn desc(&self) -> &'static ElementDesc {
        &MUXN_DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Discover the linked sink pads: a linked pad carries a link-time negotiated
        // format, an unlinked one carries none (see the module docs for why the pad set
        // is a static menu). Track numbers are linked-pad order, 1-based.
        self.lanes.clear();
        for i in 0..MAX_TRACKS {
            let pad = PadId(i as u32);
            if ctx.negotiated(pad).is_some() {
                self.lanes.push(Lane {
                    pad,
                    track: self.lanes.len() as u64 + 1,
                    state: LaneState::AwaitCaps,
                    queue: VecDeque::new(),
                    next_ts_ns: 0,
                    frame_dur_ns: 0,
                });
            }
        }
        if self.lanes.is_empty() {
            return Err(Error::Resource(
                "mkvmuxn: no sink pad linked (link sink_0..sink_{n-1})".to_string(),
            ));
        }
        if let Some(n) = self.expect {
            if self.lanes.len() != n {
                return Err(Error::Resource(format!(
                    "mkvmuxn: constructed for {n} track(s) but {} sink pad(s) are linked",
                    self.lanes.len(),
                )));
            }
        }
        self.writer = None;
        self.header_done = false;
        self.done = false;
        self.duration_ns = 0;
        self.prebuf_bytes = 0;
        self.prebuf_bufs = 0;
        self.out.clear();
        self.out_off = 0;
        // Probe the out-format a pooled buffer would carry (see `MkvMux::out_fmt`).
        self.out_fmt = ctx.alloc_exact(SRC, 1).format;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Backpressure: resume any parked output pieces first; consume input only while
        // emission keeps up (spec: the pool, never the heap, bounds a muxer).
        if !self.drain_out(ctx) {
            return Ok(Flow::Ok);
        }
        // Feed. With one linked pad the scheduler treats this element as a plain
        // single-input head and delivers batches through `inputs`; with several it is a
        // fan-in head — `inputs` stays empty and each pad's batches arrive via
        // `take_input_on`. Handle both (each path is a no-op under the other).
        if self.lanes.len() == 1 {
            while let Some(buf) = inputs.pop() {
                self.accept(0, buf)?;
            }
        }
        for i in 0..self.lanes.len() {
            let pad = self.lanes[i].pad;
            let mut batch = ctx.take_input_on(pad);
            while let Some(buf) = batch.pop_front() {
                self.accept(i, buf)?;
            }
        }
        self.maybe_write_header();
        self.merge(ctx, false);
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Primary finalize path: delivered once every upstream pad has closed and
            // all input is consumed, so the pts-ordered tail drain below is complete.
            Event::Eos => self.finish_stream(ctx),
            Event::FormatChange(f) => {
                if !self.done {
                    self.on_format_change(ctx, f)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, ctx: &mut Ctx) {
        // Belt-and-braces: guarantee finalize even if `event(Eos)` was not delivered.
        // Idempotent via the writer `Option` take.
        self.finish_stream(ctx);
        self.out = MuxOut::new();
        self.out_off = 0;
    }
}
