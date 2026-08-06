//! The PipeWire [`DeviceBackend`] — the production audio device.
//!
//! Everything PipeWire-specific lives here: the device thread, the stream, the `SPA` format
//! and buffer parameters, and the `pw_time` output-latency measurement. The audio logic
//! itself (pulling from the ring, the volume ramp, the frame counters) is *not* here — it is
//! [`Renderer`], shared with every other backend, and this module's real-time callback is
//! little more than "get a buffer, call `render`, hand it back".

use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::spa;
use pw::spa::pod::Pod;
use pw::spa::sys as spa_sys;

use profluens_core::error::Error;

use crate::out::{frames_to_ns, CanonicalFormat, DeviceBackend, Renderer, SampleFormat};

/// Requested device buffer in frames (via `SPA_PARAM_Buffers`, with `NODE_LATENCY` as the
/// matching hint). Bounds how soon the device pulls audio after a seek/flush — the perceptible
/// seek latency. ~1024 frames (~23 ms at 44.1 kHz) makes seeking feel instant; the byte ring
/// is the real jitter buffer, so a small device buffer does not risk underruns.
const QUANTUM: u32 = 1024;

/// How long [`PwBackend::start`] waits for the device thread to report that it connected.
/// Bounded: a slow or busy daemon must not stall the caller, so a timeout is treated as
/// "probably fine, carry on" while an explicit failure is reported.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

fn spa_format(fmt: SampleFormat) -> spa::param::audio::AudioFormat {
    use spa::param::audio::AudioFormat as A;
    match fmt {
        SampleFormat::U8 => A::U8,
        SampleFormat::S16 => A::S16LE,
        SampleFormat::S24 => A::S24LE,
        SampleFormat::S32 => A::S32LE,
        SampleFormat::F32 => A::F32LE,
    }
}

// --- output-latency measurement --------------------------------------------------------

/// The fields of PipeWire's `struct pw_time` this backend uses, lifted into plain Rust so the
/// unit conversion below is testable without a daemon.
///
/// Semantics are quoted from the `struct pw_time` documentation in `pipewire/stream.h`
/// (PipeWire 1.6.8, the header this crate's bindings are generated from):
///
/// * `rate` — "the rate of `ticks` and `delay`. This is usually expressed in
///   1/&lt;samplerate&gt;." So one tick is `rate.num / rate.denom` **seconds**.
/// * `delay` — "delay to device. This is the time it will take for the next output sample of
///   the stream to be presented by the playback device ... This delay includes the delay
///   introduced by all filters on the path between the stream and the device and extra delay
///   offsets. ... This delay does not include the delay caused by queued buffers. The delay
///   can be negative". It is expressed in `rate` ticks, and for a negative value the header
///   states "it is acceptable to clamp negative delays to 0" — which is what we do.
/// * `queued` — "the sum of all the pw_buffer.size fields of the buffers that are currently
///   queued in the stream but not yet processed. The application can choose the units of
///   this value". See [`refresh_out_delay`] for why it reads 0 here.
/// * `buffered` — "for audio/raw it contains the number of frames that are buffered inside
///   the resampler/converter", i.e. frames in the *stream's* own rate, not the graph's.
///
/// The header's own overview diagram places the three terms in series between our queue and
/// the speaker (`queued` → `buffered` → `delay`), which is exactly the write-side-to-audible
/// gap we want:
///
/// ```text
/// queue     +-+ +-+  +-----------+                 +--------+
/// ---->     | | | |->| converter | ->   graph  ->  | kernel | -> speaker
/// dequeue   buffers                \-------------------/\--------/
///                                     graph              internal
///                                    latency             latency
///         \--------/\-------------/\-----------------------------/
///           queued      buffered            delay
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PwTime {
    pub(crate) rate_num: u32,
    pub(crate) rate_denom: u32,
    pub(crate) delay: i64,
    pub(crate) queued: u64,
    pub(crate) buffered: u64,
}

/// How far the write-side position leads what the speaker is producing, in nanoseconds.
///
/// The `delay` term is a tick count in the graph's time domain, so it converts with the
/// `pw_time.rate` fraction (`ticks * num / denom` seconds); `queued` and `buffered` are
/// frame counts in the *stream's* domain, so they convert with `stream_rate`.
///
/// Deliberately **not** extrapolated by the header's `elapsed` correction. That correction
/// exists to age a snapshot when you are computing *when a specific sample will play*; we
/// want the steady-state gap between two positions that advance at the same rate, and the
/// snapshot is refreshed every graph cycle anyway.
pub(crate) fn out_delay_ns(t: PwTime, stream_rate: u32) -> u64 {
    let delay_ns = if t.rate_denom == 0 || t.delay <= 0 {
        0 // negative delay: the header says clamping to 0 is acceptable
    } else {
        let ns = t.delay as u128 * t.rate_num as u128 * 1_000_000_000 / t.rate_denom as u128;
        u64::try_from(ns).unwrap_or(u64::MAX)
    };
    delay_ns
        .saturating_add(frames_to_ns(t.queued, stream_rate))
        .saturating_add(frames_to_ns(t.buffered, stream_rate))
}

/// Refresh the published output delay from the stream's `pw_time` snapshot.
///
/// Called from the RT callback: `pw_stream_get_time_n` is documented "RT safe"
/// (`pipewire/stream.h`) — it copies out the stream's time area, taking no lock and
/// allocating nothing. It returns an error while the stream is not STREAMING ("This function
/// should only be called in the STREAMING state and will return an error when called in any
/// other state"), in which case we keep the previous estimate rather than publishing a bogus
/// zero. The safe `pipewire` 0.8 wrapper does not bind it (`TODO: pw_stream_get_time()` in
/// its `stream.rs`), so this goes through the re-exported `pipewire-sys` bindings.
///
/// The `pw_time.queued` term folded in here reads 0 in practice because we never set
/// `pw_buffer.size` (the safe `pipewire` 0.8 `Buffer` does not expose it), and that is very
/// nearly exact for this stream: it is a pull-driven `RT_PROCESS` stream, so the graph calls
/// us precisely because it has consumed the previous buffer — there is nothing queued at the
/// moment we sample. The term is kept in [`out_delay_ns`] so the math is the header's, and so
/// it becomes correct for free if that ever changes.
fn refresh_out_delay(stream: &pw::stream::StreamRef, stream_rate: u32, out_delay: &std::sync::atomic::AtomicU64) {
    // SAFETY: `stream` is a live `pw_stream` owned by this thread's PipeWire loop, and the
    // call only writes into `t`. `pw_time` is plain integers, so a zeroed value is valid;
    // passing its true size is how the call learns which fields this caller knows about
    // ("The size of this structure can grow as more fields are added in the future").
    let mut t: pw::sys::pw_time = unsafe { std::mem::zeroed() };
    let res = unsafe {
        pw::sys::pw_stream_get_time_n(
            stream.as_raw_ptr(),
            &mut t,
            std::mem::size_of::<pw::sys::pw_time>(),
        )
    };
    if res < 0 {
        return;
    }
    let snap = PwTime {
        rate_num: t.rate.num,
        rate_denom: t.rate.denom,
        delay: t.delay,
        queued: t.queued,
        buffered: t.buffered,
    };
    out_delay.store(out_delay_ns(snap, stream_rate), Ordering::Relaxed);
}

// --- the backend -----------------------------------------------------------------------

/// A message for the device thread. `pw_stream` is not thread-safe, so everything that has to
/// touch it — quitting, idle parking — is posted through PipeWire's own cross-thread channel
/// and executed on the loop thread.
enum PwCmd {
    Quit,
    SetActive(bool),
}

/// The PipeWire [`DeviceBackend`]: one device thread running a `pw_main_loop`, one playback
/// stream, and the RT process callback that drives [`Renderer::render`].
pub(crate) struct PwBackend {
    tx: Mutex<Option<pw::channel::Sender<PwCmd>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl PwBackend {
    pub(crate) fn new() -> Self {
        Self { tx: Mutex::new(None), thread: Mutex::new(None) }
    }

    fn send(&self, cmd: PwCmd) {
        if let Ok(guard) = self.tx.lock() {
            if let Some(tx) = guard.as_ref() {
                let _ = tx.send(cmd);
            }
        }
    }
}

impl DeviceBackend for PwBackend {
    fn start(&self, renderer: Renderer) -> Result<(), Error> {
        let fmt = renderer.format();
        let (tx, rx) = pw::channel::channel::<PwCmd>();
        // A one-shot handshake so `AudioOut::open` can report a device that refused to open,
        // instead of silently producing no sound. Sized 1 and never read twice.
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        let handle = std::thread::Builder::new()
            .name("pf-pipewire".into())
            .spawn(move || {
                // `renderer` is moved onto the PW thread and driven from its RT callback; on
                // any exit it drops here, closing the ring so a parked producer wakes.
                if let Err(e) = run_pw(renderer, rx, ready_tx) {
                    eprintln!("pipewireaudiosink: PipeWire error: {e}");
                }
            })
            .map_err(|e| Error::Resource(format!("AudioOut: device thread spawn: {e}")))?;

        *self.tx.lock().unwrap_or_else(|e| e.into_inner()) = Some(tx);
        *self.thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);

        match ready_rx.recv_timeout(CONNECT_TIMEOUT) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => Err(Error::Resource(format!("AudioOut: {msg}"))),
            // The sender vanished: the thread died before it could report. That is a genuine
            // failure and worth surfacing.
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::Resource(format!(
                "AudioOut: the PipeWire device thread exited during setup ({} Hz, {} ch)",
                fmt.rate, fmt.channels
            ))),
            // A daemon slow to answer is not a failure — the stream will come up on its own,
            // and until it does the ring simply fills.
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
        }
    }

    fn set_idle(&self, idle: bool) {
        self.send(PwCmd::SetActive(!idle));
    }

    fn stop(&self) {
        self.send(PwCmd::Quit);
        // Drop our end of the channel too, so a loop that is already gone cannot keep it
        // alive, then join. Idempotent: `take()` leaves `None` behind.
        let handle = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
        *self.tx.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Run the PipeWire device loop on its own thread. All PipeWire objects live here (they are
/// not `Send`); the shared state is the [`Renderer`] (driven from the RT callback) and the
/// command channel. The renderer is `Send` but the loop objects are not, so it is moved in
/// here and never leaves.
// One-time device setup: the two `Vec`s below are the SPA pod serialiser's output buffers,
// built once when the stream is created and never touched again — not a per-buffer allocation.
#[allow(clippy::disallowed_methods)]
fn run_pw(
    mut renderer: Renderer,
    cmd_rx: pw::channel::Receiver<PwCmd>,
    ready: mpsc::SyncSender<Result<(), String>>,
) -> Result<(), pw::Error> {
    let fmt: CanonicalFormat = renderer.format();
    let rate = fmt.rate;
    let stride = fmt.stride();
    let pb = Arc::clone(renderer.playback());

    // Every fallible setup step reports its failure to `PwBackend::start` before propagating,
    // so a device that refuses to open surfaces as an error from `AudioOut::open` rather than
    // as silence. Exactly one of these paths ever sends.
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).inspect_err(|e| {
        let _ = ready.try_send(Err(format!("main loop: {e}")));
    })?;
    let context = pw::context::Context::new(&mainloop).inspect_err(|e| {
        let _ = ready.try_send(Err(format!("context: {e}")));
    })?;
    let core = context.connect(None).inspect_err(|e| {
        let _ = ready.try_send(Err(format!("connect to the daemon: {e}")));
    })?;

    // Advertise our desired latency (~`QUANTUM` frames). The buffer size that actually bounds
    // the post-seek device latency is requested via `SPA_PARAM_Buffers` at `connect` below;
    // this is the matching hint.
    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::NODE_LATENCY => format!("{QUANTUM}/{rate}").as_str(),
    };
    // `Rc` because the command handler below also needs the stream (to activate/deactivate it)
    // and both closures live on this one thread.
    let stream = Rc::new(pw::stream::Stream::new(&core, "profluens", props).inspect_err(|e| {
        let _ = ready.try_send(Err(format!("create the stream: {e}")));
    })?);

    // The RT process callback: fill the device buffer from the shared renderer (ring pull +
    // volume ramp + counters). This runs on PipeWire's real-time thread, so everything it
    // touches is wait-free and allocation-free — no mutex is taken here.
    let _listener = stream
        .add_local_listener_with_user_data(())
        .process(move |stream, ()| {
            // Republish the output delay before handing over more frames, so the snapshot
            // pairs with the write-side count as it stands *now*: the buffer we are about to
            // fill is accounted for by not having bumped `frames` for it yet.
            refresh_out_delay(stream, rate, &pb.out_delay_ns);
            if pb.flush_stream.swap(false, Ordering::Relaxed) {
                // Seek: drop buffers already committed past our ring (the rate-converter's,
                // when the device rate differs from the stream's), so the new position is
                // heard now.
                let _ = stream.flush(false);
            }
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if let Some(data) = datas.get_mut(0) {
                    let size = match data.data() {
                        Some(slice) => renderer.render(slice),
                        None => 0,
                    };
                    let chunk = data.chunk_mut();
                    *chunk.offset_mut() = 0;
                    *chunk.stride_mut() = stride as _;
                    *chunk.size_mut() = size as _;
                }
            }
        })
        .register()
        .inspect_err(|e| {
            let _ = ready.try_send(Err(format!("register the stream listener: {e}")));
        })?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa_format(fmt.sample));
    audio_info.set_rate(rate);
    audio_info.set_channels(fmt.channels);

    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: spa_sys::SPA_TYPE_OBJECT_Format,
            id: spa_sys::SPA_PARAM_EnumFormat,
            properties: audio_info.into(),
        }),
    )
    .unwrap()
    .0
    .into_inner();

    // Request a small device buffer (spec: latency #2; responsive seeking). When the device
    // rate differs from the stream's, PipeWire otherwise sizes our buffer for its rate
    // converter — here ~278 ms — which is how long after a seek the device waits to pull the
    // new position's audio. Asking for `QUANTUM` frames bounds that; the ring is the real
    // jitter buffer. `size`/`stride` are bytes.
    let prop = |key: u32, v: i32| pw::spa::pod::Property {
        key,
        flags: pw::spa::pod::PropertyFlags::empty(),
        value: pw::spa::pod::Value::Int(v),
    };
    let buf_bytes = (QUANTUM as usize * stride) as i32;
    let buffers: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: spa_sys::SPA_TYPE_OBJECT_ParamBuffers,
            id: spa_sys::SPA_PARAM_Buffers,
            properties: vec![
                prop(spa_sys::SPA_PARAM_BUFFERS_buffers, 2),
                prop(spa_sys::SPA_PARAM_BUFFERS_blocks, 1),
                prop(spa_sys::SPA_PARAM_BUFFERS_size, buf_bytes),
                prop(spa_sys::SPA_PARAM_BUFFERS_stride, stride as i32),
            ],
        }),
    )
    .unwrap()
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).unwrap(), Pod::from_bytes(&buffers).unwrap()];

    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .inspect_err(|e| {
            let _ = ready.try_send(Err(format!("connect the stream: {e}")));
        })?;

    // Setup succeeded — release `AudioOut::open`.
    let _ = ready.try_send(Ok(()));

    // Cross-thread commands: quit, and idle park/unpark. `set_active` must run on the loop
    // thread, which is precisely what this channel gives us.
    let weak = mainloop.downgrade();
    let cmd_stream = Rc::clone(&stream);
    let _recv = cmd_rx.attach(mainloop.loop_(), move |cmd| match cmd {
        PwCmd::Quit => {
            if let Some(ml) = weak.upgrade() {
                ml.quit();
            }
        }
        PwCmd::SetActive(active) => {
            // Deactivating stops the node being scheduled at all — the RT callback ceases to
            // fire, so the ring is frozen and the counters stop. That freeze is the documented
            // idle policy (see `AudioOutHandle::set_idle`).
            let _ = cmd_stream.set_active(active);
        }
    });

    mainloop.run();
    Ok(())
}
