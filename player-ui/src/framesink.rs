//! [`FrameSlotSink`] — the clock-paced, GUI-bridging video sink (spec:
//! profluens.md's UI `<update>` §1209 — "use SDL3 with a custom UI library on
//! top … the UI must be the same philosophy as the rest of SC (very performant,
//! per-frame arenas)"). A media player cannot have two SDL windows fighting, so
//! this sink touches **no** SDL: the GUI thread owns the one window, and video
//! arrives here through a shared, lock-light frame slot the GUI reads at its own
//! vsync.
//!
//! # Why this exists, and how it preserves A/V sync
//!
//! [`Sdl3VideoSink`](pf_sdl3) is an *active* sink that paces each frame on the
//! pipeline clock (`base + pts + path_latency` via `ctx.wait_until`) and then
//! GPU-presents. Here we replicate the **pacing** verbatim — the same QoS drop,
//! the same `ctx.wait_until`, the same seek-gen bail — but instead of presenting we
//! **copy the released frame's bytes into a shared triple-buffered slot**
//! ([`FrameSlot`]). The audio device sink is the pipeline clock, so video releases
//! on the audio timeline exactly as before; the GUI just uploads and presents
//! whatever frame is *currently* released at its own refresh rate. That keeps the
//! classic audio-master A/V sync arrangement intact while moving presentation to
//! the app's single window.
//!
//! Per frame (mirroring `Sdl3VideoSink::render`):
//! * **QoS first** — a frame already more than one frame-duration late is dropped
//!   with [`BusMessage::Qos`] *before* any copy cost (spec: QoS);
//! * pace with `ctx.wait_until(pts)` — an `Interrupted` wait (flush/seek/shutdown)
//!   returns cleanly (spec: Clocking — only sinks wait; flush/seek — blocking sinks
//!   bail on `ctx.seek_gen()`);
//! * copy the decoded I420/NV12/Gray8 planes into the writer slot and publish it
//!   (never blocks the streaming thread — the GUI reads a *different* slot).
//!
//! On `Eos` the last published frame stays in the slot (the GUI keeps presenting
//! it). No display, no failure: this sink has no window to fail — it always copies.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::clock::WaitOutcome;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{
    ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc,
};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use pf_vaapi::gpuframe::{self, GpuFrame, GpuFrameChannel, GpuFrameHeader};

const SINK: PadId = PadId(0);

// `video/raw` vocabulary — literals, exactly as `Sdl3VideoSink` declares them, so the
// crate does not pin field-name identity to profluens-video (the pipeline interns by
// string, so ids line up with any video peer regardless of who declared the field).
const FAMILY: &str = "video/raw";
const F_WIDTH: &str = "width";
const F_HEIGHT: &str = "height";
const F_PIXFMT: &str = "pixfmt";
const F_FPS: &str = "fps";
const PIXFMT_I420: &str = "i420";
const PIXFMT_GRAY8: &str = "gray8";
const PIXFMT_NV12: &str = "nv12";

// The sink offers the same broad `video/raw` template `Sdl3VideoSink` does, constrained
// to the pixel formats the SDL classic YUV path can present (i420 natively; gray8 as
// flat-chroma IYUV; nv12 what hardware decoders emit). Dimensions/fps arrive at runtime
// via the decoder's FormatChange, so the pad is `dynamic`.
static PIXFMT_VALUES: [ValueDesc; 3] =
    [ValueDesc::Id(PIXFMT_I420), ValueDesc::Id(PIXFMT_GRAY8), ValueDesc::Id(PIXFMT_NV12)];
static SINK_FIELDS: [FieldDesc; 4] = [
    FieldDesc { field: F_WIDTH, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_HEIGHT, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_PIXFMT, allowed: ConstraintDesc::Set(&PIXFMT_VALUES), preferred: None },
    FieldDesc { field: F_FPS, allowed: ConstraintDesc::Any, preferred: None },
];
// Two sink offers: `video/raw` (the CPU path) and `video/gpu` (the zero-copy DMA-BUF path,
// the shared [`gpuframe::GPU_OFFER`]). A CPU sink links a `video/raw` decoder; a zero-copy
// sink links a `video/gpu` (DMA-BUF-exporting) decoder — the family intersection picks
// exactly one, so the two paths never cross-wire.
static SINK_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SINK_FIELDS }, gpuframe::GPU_OFFER];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: true, // dimensions/fps announced at runtime
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "frameslotsink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active, // its own thread; paces the graph on the clock like the SDL sink
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None, // built explicitly with a shared FrameSlot, not by parse-launch
};

/// The pixel format the slot carries. Mirrors [`profluens_video::PixelFormat`] but is
/// kept local so a consumer (the SDL video layer) does not have to depend on the video
/// crate just to read the enum — it is a tiny POD tag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotPix {
    /// Three planes: Y (`w*h`), Cb (`cw*ch`), Cr (`cw*ch`) — SDL `IYUV`.
    I420,
    /// Two planes: Y (`w*h`), interleaved CbCr (`2*cw*ch`) — SDL `NV12`.
    Nv12,
    /// One plane: `w*h` luma — presented as IYUV with flat 128 chroma.
    Gray8,
}

impl SlotPix {
    fn from_name(s: &str) -> Option<SlotPix> {
        match s {
            PIXFMT_I420 => Some(SlotPix::I420),
            PIXFMT_GRAY8 => Some(SlotPix::Gray8),
            PIXFMT_NV12 => Some(SlotPix::Nv12),
            _ => None,
        }
    }
}

/// One buffered frame: the geometry + a reused byte vector holding the tightly-packed
/// planes. The vector is grown once and reused (`clear()`+`extend`), so a steady-state
/// frame copy does no heap allocation — the one sanctioned per-frame-ish cost (a fixed
/// triple buffer, reused). `seq` monotonically increases so the GUI can tell a genuinely
/// new frame from a re-read of the same one (fps computation, upload-skip).
struct FrameBuf {
    width: usize,
    height: usize,
    pixfmt: SlotPix,
    pts: Timestamp,
    bytes: Vec<u8>,
    /// This buffer's publication sequence number (0 = never written).
    seq: u64,
}

impl FrameBuf {
    fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            pixfmt: SlotPix::I420,
            pts: Timestamp::NONE,
            bytes: Vec::new(),
            seq: 0,
        }
    }
}

/// The GUI's owned copy of the latest published frame. The reader side memcpys the slot's
/// freshest buffer into a reader-owned scratch under the lock, then reads/uploads from
/// this outside the lock — so the streaming thread is never blocked for a texture upload
/// (spec: never block the streaming thread on the GUI). Reused between frames, so a
/// steady-state read grows nothing.
pub struct FrameCopy {
    pub width: usize,
    pub height: usize,
    pub pixfmt: SlotPix,
    pub pts: Timestamp,
    pub bytes: Vec<u8>,
    pub seq: u64,
}

impl FrameCopy {
    /// A fresh, empty reader scratch. The GUI keeps one and refreshes it each frame.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            pixfmt: SlotPix::I420,
            pts: Timestamp::NONE,
            bytes: Vec::new(),
            seq: 0,
        }
    }
}

impl Default for FrameCopy {
    fn default() -> Self {
        Self::new()
    }
}

/// The shared frame slot: a three-buffer rotation the streaming thread writes and the GUI
/// thread reads.
///
/// # The three-slot rotation (why not a `Mutex<Frame>`)
///
/// A single mutexed frame would make the streaming thread wait on the GUI's whole read (a
/// stutter on either side stalling the other), and the spec is emphatic that the streaming
/// thread must never block on the GUI. Three buffers plus a "latest wins" discipline
/// decouple them: the writer fills the buffer that is neither the last it published
/// (`latest`) nor the one the reader is snapshotting (`reading`), so a `publish` never
/// waits on a `read` for anything but the tiny index bookkeeping. The reader takes the
/// lock, marks `latest` as `reading`, memcpys it into its own [`FrameCopy`], then unlocks
/// and uploads from the copy at leisure. Both sides hold the lock only for a bounded
/// memcpy (never an SDL texture upload). This is the classic triple-buffer,
/// single-producer/single-consumer, drop-old-frames arrangement: the GUI always sees the
/// freshest complete frame, never a torn one.
pub struct FrameSlot {
    /// The three frame buffers + index bookkeeping (the CPU path).
    inner: Mutex<Inner>,
    /// Published so the GUI can cheaply poll "is there a newer frame?" without taking the
    /// lock — one acquire load. Matches the `seq` of the buffer at `latest`.
    latest_seq: AtomicU64,
    /// The **zero-copy** slot: the latest paced [`GpuFrame`] the GUI hasn't yet imported
    /// (latest-wins, single frame). Distinct from the CPU triple buffer — a zero-copy run
    /// only ever fills this, a CPU run only the triple buffer. A GpuFrame superseded before
    /// the GUI took it is dropped here (its fds close), and its token released so the decoder
    /// reclaims the surface (see [`publish_gpu`](Self::publish_gpu)).
    gpu: Mutex<Option<GpuFrame>>,
    /// The presentation-release channel back to the zero-copy decoder (the GUI releases a
    /// token after presenting; the decoder reclaims the surface). `None` for a CPU-path slot.
    channel: Option<Arc<GpuFrameChannel>>,
    /// Bumped each time a GpuFrame is published, so the GUI can poll "new GPU frame?" without
    /// the lock.
    gpu_seq: AtomicU64,
    /// Set once the pipeline signals EOS, so the GUI can annotate "ended" without a bus.
    eos: std::sync::atomic::AtomicBool,
    /// Cumulative frames the sink dropped for QoS lateness — surfaced in the stats overlay
    /// alongside the tap counters (the tap counts buffers in/out; this is the *paced* drop).
    qos_drops: AtomicU64,
}

/// The lock-guarded bookkeeping: which of the three buffers is the freshest published
/// (`latest`), which the reader is currently copying out of (`reading`), and which is
/// free for the writer to fill next.
struct Inner {
    bufs: [FrameBuf; 3],
    /// The most recently published buffer index (what a reader will grab).
    latest: usize,
    /// The buffer a reader is snapshotting this instant (or `usize::MAX` when none). The
    /// writer never fills this one.
    reading: usize,
    /// Next sequence number to stamp on a publish.
    next_seq: u64,
}

impl FrameSlot {
    /// A fresh, empty slot for the **CPU** path. Share it (`Arc`) between the sink (writer)
    /// and the GUI (reader).
    pub fn new() -> Arc<Self> {
        Self::build(None)
    }

    /// A fresh slot for the **zero-copy** path, carrying the presentation-release `channel`
    /// back to the decoder (the GUI releases each token after presenting so the decoder
    /// reclaims its surface). Share it (`Arc`) between the zero-copy sink and the GUI.
    pub fn new_zerocopy(channel: Arc<GpuFrameChannel>) -> Arc<Self> {
        Self::build(Some(channel))
    }

    fn build(channel: Option<Arc<GpuFrameChannel>>) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                bufs: [FrameBuf::new(), FrameBuf::new(), FrameBuf::new()],
                latest: 0,
                reading: usize::MAX,
                next_seq: 0,
            }),
            latest_seq: AtomicU64::new(0),
            gpu: Mutex::new(None),
            channel,
            gpu_seq: AtomicU64::new(0),
            eos: std::sync::atomic::AtomicBool::new(false),
            qos_drops: AtomicU64::new(0),
        })
    }

    /// The seq of the latest published frame (0 = none yet). One acquire load — the GUI's
    /// per-frame "do I have a new frame to upload?" check.
    pub fn latest_seq(&self) -> u64 {
        self.latest_seq.load(Ordering::Acquire)
    }

    /// Has the pipeline reached EOS?
    pub fn is_eos(&self) -> bool {
        self.eos.load(Ordering::Acquire)
    }

    /// Cumulative QoS drops (paced frames dropped for lateness).
    pub fn qos_drops(&self) -> u64 {
        self.qos_drops.load(Ordering::Acquire)
    }

    /// The writer side: copy `bytes` (one tightly-packed frame) into a free buffer and
    /// publish it as the new latest. Fills the buffer that is neither `latest` nor the one
    /// the reader is copying out of, so it never overwrites a frame in use. `bytes` is the
    /// decoder's exact frame memory — copied once into the reused slot vector.
    fn publish(&self, width: usize, height: usize, pixfmt: SlotPix, pts: Timestamp, bytes: &[u8]) {
        let mut g = self.inner.lock().unwrap();
        let (latest, reading) = (g.latest, g.reading);
        // The one buffer that is neither the last published nor the one being read.
        let w = (0..3).find(|&i| i != latest && i != reading).expect("a free buffer of three");
        let seq = g.next_seq.wrapping_add(1);
        g.next_seq = seq;
        {
            let b = &mut g.bufs[w];
            b.width = width;
            b.height = height;
            b.pixfmt = pixfmt;
            b.pts = pts;
            b.seq = seq;
            // Reuse the vector's capacity: clear + extend never shrinks, so after the
            // first frame of a given size this is a bounded memcpy with no allocation.
            b.bytes.clear();
            b.bytes.extend_from_slice(bytes);
        }
        g.latest = w;
        drop(g);
        self.latest_seq.store(seq, Ordering::Release);
    }

    /// The reader side (GUI thread): copy the latest published frame into `dst` and return
    /// `true` if a *newer* frame was taken (a caller skips the texture upload when the seq
    /// is unchanged). Takes the lock only for the memcpy — the subsequent upload happens
    /// from `dst`, off the lock, so the writer is never blocked on the GUI.
    pub fn read_into(&self, dst: &mut FrameCopy) -> bool {
        let mut g = self.inner.lock().unwrap();
        let idx = g.latest;
        let seq = g.bufs[idx].seq;
        if seq == 0 || seq == dst.seq {
            return false; // nothing published yet, or the caller already has this frame
        }
        // Mark this buffer as being read so a concurrent publish avoids it, then copy.
        g.reading = idx;
        {
            let b = &g.bufs[idx];
            dst.width = b.width;
            dst.height = b.height;
            dst.pixfmt = b.pixfmt;
            dst.pts = b.pts;
            dst.seq = b.seq;
            dst.bytes.clear();
            dst.bytes.extend_from_slice(&b.bytes);
        }
        g.reading = usize::MAX;
        true
    }

    /// Upload the latest published frame **straight into the caller's GPU texture** via
    /// `upload`, with no intermediate [`FrameCopy`] — the copy-elimination path. `upload`
    /// receives the frame geometry + tightly-packed planes and returns whether it consumed
    /// them (a short/failed upload returns `false`, so the seq is not advanced). Returns
    /// `Some(seq)` of the newly-uploaded frame, or `None` when nothing newer than `last_seq`
    /// is published.
    ///
    /// The GUI thread calls this; the streaming thread's [`publish`](Self::publish) blocks on
    /// the same lock only for the upload's duration (a memcpy into GPU-mapped memory). That is
    /// a deliberate trade: one *fewer* full-frame CPU copy per displayed frame (the slot→copy
    /// memcpy is gone — the frame goes decoder→slot→texture, not decoder→slot→copy→texture) in
    /// exchange for a brief, rare producer stall. For a ~24–60 fps player that is a clear win
    /// (the copy was ~4% of playback CPU in profiling).
    pub fn upload_latest_new<F>(&self, last_seq: u64, upload: F) -> Option<u64>
    where
        F: FnOnce(usize, usize, SlotPix, &[u8]) -> bool,
    {
        let g = self.inner.lock().unwrap();
        let b = &g.bufs[g.latest];
        if b.seq == 0 || b.seq == last_seq {
            return None; // nothing published yet, or the caller already has this frame
        }
        let seq = b.seq;
        // The buffer stays valid for the whole call (we hold the lock), so `publish` — which
        // needs the same lock — cannot recycle it mid-upload.
        if upload(b.width, b.height, b.pixfmt, &b.bytes) {
            Some(seq)
        } else {
            None
        }
    }

    /// Writer (zero-copy sink): publish a paced [`GpuFrame`] for the GUI to import. Latest
    /// wins: a frame still sitting here (the GUI hasn't imported it) is superseded — it drops
    /// (closing its fds) and its token is released back to the decoder so the surface is not
    /// leaked. The GUI polls [`gpu_latest_seq`](Self::gpu_latest_seq) and takes with
    /// [`take_gpu`](Self::take_gpu).
    pub fn publish_gpu(&self, frame: GpuFrame) {
        let mut g = self.gpu.lock().unwrap();
        if let Some(stale) = g.take() {
            // The GUI never imported the previous frame; release its token so the decoder
            // reclaims the surface (the GpuFrame's own drop closes its fds).
            if let Some(ch) = &self.channel {
                ch.release(stale.token);
            }
            drop(stale);
        }
        *g = Some(frame);
        drop(g);
        self.gpu_seq.fetch_add(1, Ordering::Release);
    }

    /// The GUI's cheap "is there a new GPU frame?" poll (one acquire load).
    pub fn gpu_latest_seq(&self) -> u64 {
        self.gpu_seq.load(Ordering::Acquire)
    }

    /// Reader (GUI): take the pending [`GpuFrame`], transferring ownership (its fds) to the
    /// caller, which imports it and must later call
    /// [`release_gpu_token`](Self::release_gpu_token) once presented. `None` if nothing new.
    pub fn take_gpu(&self) -> Option<GpuFrame> {
        self.gpu.lock().unwrap().take()
    }

    /// Writer (zero-copy sink): on a **flush/seek**, drop the pending un-imported slot frame —
    /// release its token (the decoder reclaims the surface) and close its fds — so a pre-seek
    /// frame can never be published/shown after the seek. The GUI keeps displaying its last
    /// *imported* frame until the first post-seek frame arrives.
    pub fn flush_gpu(&self) {
        let mut g = self.gpu.lock().unwrap();
        if let Some(stale) = g.take() {
            if let Some(ch) = &self.channel {
                ch.release(stale.token);
            }
            // `stale` drops here → its fds close.
        }
    }

    /// Reader (GUI): after presenting `token`'s imported frame, release it so the decoder
    /// reclaims the surface (the anti-tear handshake's consumer half).
    pub fn release_gpu_token(&self, token: gpuframe::FrameToken) {
        if let Some(ch) = &self.channel {
            ch.release(token);
        }
    }

    /// Whether this slot carries the zero-copy release channel (a zero-copy run).
    pub fn is_zerocopy(&self) -> bool {
        self.channel.is_some()
    }

    /// Writer: mark end-of-stream (the last published frame stays latest).
    fn mark_eos(&self) {
        self.eos.store(true, Ordering::Release);
    }

    fn count_qos_drop(&self) {
        self.qos_drops.fetch_add(1, Ordering::Relaxed);
    }
}

/// The clock-paced frame-slot sink. Construct with a shared [`FrameSlot`]; add it to the
/// player's pipeline and link the player's `video_out()` tap to its `sink` pad. The default
/// [`new`](Self::new) accepts CPU `video/raw` frames (copied into the triple buffer); the
/// zero-copy [`new_zerocopy`](Self::new_zerocopy) accepts `video/gpu` descriptor buffers,
/// popping the paired [`GpuFrame`] from the channel and handing it to the GUI to import.
pub struct FrameSlotSink {
    slot: Arc<FrameSlot>,
    format: Option<FrameFormat>,
    /// Render the next frame immediately — no QoS, no clock wait — set on `FlushStart` so
    /// a paused seek publishes its new frame instantly (spec: flush/seek preroll), exactly
    /// as `Sdl3VideoSink` prerolls the first post-seek frame.
    preroll_next: bool,
    /// When `Some`, the sink is in **zero-copy** mode: incoming buffers are `video/gpu`
    /// descriptors, and the paired [`GpuFrame`] is popped from this channel (the decoder's
    /// push side) by token and published to the GUI. `None` = the CPU `video/raw` path.
    channel: Option<Arc<GpuFrameChannel>>,
}

/// The negotiated geometry the sink learned from a `FormatChange` (mirrors the SDL sink's
/// `FrameFormat`, minus the GPU colorimetry the classic SDL YUV path resolves itself).
#[derive(Clone, Copy)]
struct FrameFormat {
    width: usize,
    height: usize,
    pixfmt: SlotPix,
    /// ns/frame from fps, for the QoS "more than one frame late" test. `0` disables QoS.
    frame_dur_ns: u64,
}

impl FrameSlotSink {
    /// Build the CPU-path sink over a shared slot. The GUI holds the other `Arc` clone.
    pub fn new(slot: Arc<FrameSlot>) -> Self {
        Self { slot, format: None, preroll_next: false, channel: None }
    }

    /// Build the **zero-copy** sink: incoming `video/gpu` buffers are paced on the clock, then
    /// the paired [`GpuFrame`] is popped from `channel` and published to the slot for the GUI
    /// to import via EGL. Pair with a [`FrameSlot::new_zerocopy`] slot (same channel) and a
    /// zero-copy decoder (`pf_vaapi::video_decoder_zerocopy_for`, the channel's push side).
    pub fn new_zerocopy(slot: Arc<FrameSlot>, channel: Arc<GpuFrameChannel>) -> Self {
        Self { slot, format: None, preroll_next: false, channel: Some(channel) }
    }

    /// Read a `video/raw` `FixedFormat` into a [`FrameFormat`] — the same field reads
    /// `Sdl3VideoSink::read_format` does (width/height/pixfmt/fps), sans colorimetry.
    fn read_format(ctx: &Ctx, f: &FixedFormat) -> Option<FrameFormat> {
        let width = ctx.field_id(F_WIDTH).and_then(|id| f.get(id)).and_then(int_value)? as usize;
        let height = ctx.field_id(F_HEIGHT).and_then(|id| f.get(id)).and_then(int_value)? as usize;
        let pix_name = ctx
            .field_id(F_PIXFMT)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })?;
        let pixfmt = SlotPix::from_name(pix_name)?;
        // fps optional; when present compute ns/frame for QoS (den/num * 1e9), matching
        // the SDL sink's arithmetic exactly so both sinks drop at the same threshold.
        let frame_dur_ns = ctx
            .field_id(F_FPS)
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Rat(num, den) if num > 0 && den != 0 => {
                    Some((den.unsigned_abs() as u64).saturating_mul(1_000_000_000) / num as u64)
                }
                _ => None,
            })
            .unwrap_or(0);
        Some(FrameFormat { width, height, pixfmt, frame_dur_ns })
    }

    /// Learn geometry from a negotiated format (link-time or FormatChange). A live
    /// geometry/pixfmt change is fine: the slot's reused vectors just resize on the next
    /// copy, and the GUI recreates its texture when the view's dims change.
    fn configure(&mut self, ctx: &mut Ctx, f: &FixedFormat) {
        if let Some(fmt) = Self::read_format(ctx, f) {
            self.format = Some(fmt);
        }
    }

    /// Pace one frame on the clock (QoS + `wait_until`), then copy it into the slot.
    /// Mirrors [`Sdl3VideoSink::render`] beat for beat — the only difference is the tail:
    /// a slot copy instead of a texture present. Always returns `Ok` (a sink never fails
    /// the pipeline).
    fn accept(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(fmt) = self.format else { return Ok(()) };
        let preroll = std::mem::take(&mut self.preroll_next);

        // QoS: if the deadline is already behind us by more than one frame, drop it —
        // tested *before* the wait and the copy, so a late frame costs nothing (spec: QoS).
        if !preroll && fmt.frame_dur_ns > 0 && pts.is_some() {
            let now = ctx.now();
            if now.is_some() && now.0 > pts.0 {
                let lateness = now.0 - pts.0;
                if lateness > fmt.frame_dur_ns {
                    self.slot.count_qos_drop();
                    let sink = ctx.element();
                    ctx.post(BusMessage::Qos { sink, lateness_ns: lateness as i64 });
                    return Ok(());
                }
            }
        }

        // Pace on the clock (the audio DAC clock). An interrupt (flush/seek/stop) returns
        // cleanly — the buffer this wait paced is stale, so we do not publish it.
        if !preroll {
            match ctx.wait_until(pts) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => return Ok(()),
            }
        }

        // Publish: copy the released frame into the writer slot for the GUI to present.
        self.slot.publish(fmt.width, fmt.height, fmt.pixfmt, pts, data);
        Ok(())
    }

    /// The zero-copy accept: `data` is a [`GpuFrameHeader`] (the `video/gpu` buffer body).
    /// Pace on the clock exactly like [`accept`](Self::accept) (no fps-QoS — `video/gpu`
    /// carries no fps; the clock wait is the pacing), then pop the paired [`GpuFrame`] from
    /// the channel by token and publish it to the slot for the GUI to import. An interrupted
    /// wait (flush/seek/stop) or a stale frame releases the token so the decoder reclaims the
    /// surface — never a leak.
    fn accept_gpu(&mut self, ctx: &mut Ctx, data: &[u8], pts: Timestamp) -> Result<(), Error> {
        let Some(channel) = self.channel.clone() else { return Ok(()) };
        let Some(header) = GpuFrameHeader::from_bytes(data) else {
            // A malformed / non-video/gpu payload — drop it (loudly once would be nicer, but
            // this only happens on a wiring bug, which the family intersection prevents).
            return Ok(());
        };
        let preroll = std::mem::take(&mut self.preroll_next);

        // Pace on the clock. On an interrupt (flush/seek/stop) the frame is stale — pop it
        // and release the token so the decoder does not wait on a presentation that will
        // never come (bounded-latency: a dropped-here frame frees its surface immediately).
        if !preroll {
            match ctx.wait_until(pts) {
                WaitOutcome::Reached => {}
                WaitOutcome::Interrupted => {
                    if let Some(f) = channel.take(header.token) {
                        let tok = f.token;
                        drop(f); // closes fds
                        channel.release(tok);
                    }
                    return Ok(());
                }
            }
        }

        // Pop the out-of-band GpuFrame (fds + geometry) and hand it to the GUI via the slot.
        // If the decoder has not pushed it yet (a race the in-flight marking makes rare) or
        // it was already taken, skip — the previous frame stays displayed.
        if let Some(frame) = channel.take(header.token) {
            self.slot.publish_gpu(frame);
        }
        Ok(())
    }
}

fn int_value(v: Value) -> Option<i64> {
    match v {
        Value::Int(n) => Some(n),
        _ => None,
    }
}

impl Element for FrameSlotSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let zerocopy = self.channel.is_some();
        // CPU path: on first data, if no FormatChange arrived, try a fully-fixed link-time
        // format (the SDL sink's fallback). The zero-copy path reads geometry per-buffer from
        // the `video/gpu` header, so it needs no `self.format`.
        if !zerocopy && self.format.is_none() {
            if let Some(f) = ctx.negotiated(SINK).cloned() {
                self.configure(ctx, &f);
            }
        }
        // A seek that lands mid-batch makes the remaining frames stale: bail and let the
        // scheduler run the flush (spec: flush/seek — blocking sinks bail on seek_gen()).
        let entry_gen = ctx.seek_gen();
        while let Some(buf) = inputs.pop() {
            if ctx.seek_gen() != entry_gen {
                return Ok(Flow::Ok);
            }
            // `buf` is owned; borrowing its bytes does not clash with the `&mut ctx` the
            // pacing needs. The read is straight from the decoder's frame memory (CPU pixels)
            // or the tiny GPU-frame descriptor (zero-copy).
            if zerocopy {
                self.accept_gpu(ctx, buf.memory.data(), buf.pts)?;
            } else {
                self.accept(ctx, buf.memory.data(), buf.pts)?;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // CPU path learns geometry/fps from the FormatChange; the zero-copy path reads
            // geometry per-buffer from the `video/gpu` header, so it ignores it.
            Event::FormatChange(f) if self.channel.is_none() => self.configure(_ctx, f),
            // Preroll the first frame after a seek OR a resume: publish it immediately,
            // skipping the QoS/clock wait. On resume this stops the pause→resume "shift"
            // (frames the decoder buffered while paused would otherwise read as late against
            // the resumed clock and get QoS-dropped, skipping a few frames forward).
            Event::Resumed => self.preroll_next = true,
            Event::FlushStart => {
                self.preroll_next = true;
                // Zero-copy: the flush discards the in-flight `video/gpu` pipeline buffers, so
                // any GpuFrames the decoder already pushed but we hadn't paced are orphaned —
                // release their surfaces (else the decoder's pool starves and freezes), and
                // clear the pending un-imported slot frame (else a pre-seek frame flashes after
                // the seek). The CPU path (channel `None`) just prerolls.
                if let Some(ch) = &self.channel {
                    ch.flush_pending();
                    self.slot.flush_gpu();
                }
            }
            Event::Eos => self.slot.mark_eos(),
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pix_names_round_trip() {
        assert!(matches!(SlotPix::from_name("i420"), Some(SlotPix::I420)));
        assert!(matches!(SlotPix::from_name("nv12"), Some(SlotPix::Nv12)));
        assert!(matches!(SlotPix::from_name("gray8"), Some(SlotPix::Gray8)));
        assert!(SlotPix::from_name("rgb").is_none());
    }

    #[test]
    fn slot_publishes_and_reads_latest() {
        let slot = FrameSlot::new();
        let mut copy = FrameCopy::new();
        assert!(!slot.read_into(&mut copy), "empty slot yields no frame");
        assert_eq!(slot.latest_seq(), 0);

        // 4x4 I420 is 16 (Y) + 4 (Cb) + 4 (Cr) = 24 bytes.
        let frame_a = vec![0xAAu8; 24];
        slot.publish(4, 4, SlotPix::I420, Timestamp(1000), &frame_a);
        assert_eq!(slot.latest_seq(), 1);
        assert!(slot.read_into(&mut copy), "a frame after first publish");
        assert_eq!((copy.width, copy.height), (4, 4));
        assert_eq!(copy.pixfmt, SlotPix::I420);
        assert_eq!(copy.pts, Timestamp(1000));
        assert_eq!(copy.bytes, frame_a);
        assert_eq!(copy.seq, 1);

        // A re-read of the same frame is a no-op (seq unchanged).
        assert!(!slot.read_into(&mut copy), "same seq is not a new frame");

        // A second publish supersedes it (latest-wins).
        let frame_b = vec![0x55u8; 24];
        slot.publish(4, 4, SlotPix::I420, Timestamp(2000), &frame_b);
        assert_eq!(slot.latest_seq(), 2);
        assert!(slot.read_into(&mut copy));
        assert_eq!(copy.bytes, frame_b);
        assert_eq!(copy.pts, Timestamp(2000));
    }

    #[test]
    fn writer_always_finds_a_free_buffer() {
        // Rapid publishes with interleaved reads never panic on "a free buffer of three".
        let slot = FrameSlot::new();
        let mut copy = FrameCopy::new();
        for i in 0..50u64 {
            slot.publish(2, 2, SlotPix::Gray8, Timestamp(i), &[i as u8; 4]); // 2x2 gray8 = 4B
            let _ = slot.read_into(&mut copy);
        }
        assert_eq!(slot.latest_seq(), 50);
    }

    #[test]
    fn eos_and_qos_flags() {
        let slot = FrameSlot::new();
        assert!(!slot.is_eos());
        assert_eq!(slot.qos_drops(), 0);
        slot.mark_eos();
        slot.count_qos_drop();
        slot.count_qos_drop();
        assert!(slot.is_eos());
        assert_eq!(slot.qos_drops(), 2);
    }
}
