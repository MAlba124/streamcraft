//! `pipewireaudiosink` — plays interleaved PCM through PipeWire (spec: Milestone
//! applications — play an audio file).
//!
//! It is an **active** element on its own thread. Its `process()` pushes decoded PCM into
//! a bounded hand-off ring; a dedicated PipeWire thread runs the device loop and its
//! real-time callback pulls from that ring to fill device buffers. When the ring is full,
//! `process()` blocks — so the whole pipeline is paced by the audio device (backpressure
//! is the clock; full clock-slaving through PipeWire for A/V sync is a follow-up). The
//! PCM format is data-dependent, so the sink advertises a broad `dynamic` `audio/raw` sink
//! and configures itself from the runtime `FormatChange` a decoder announces (spec: dynamic
//! caps). On EOS the pipeline delivers `Event::Eos`, and the sink drains the ring (plays
//! out) before returning.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use pipewire as pw;
use pw::spa;
use pw::spa::pod::Pod;
use pw::spa::sys as spa_sys;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{ConstraintDesc, FieldDesc, FixedFormat, OfferDesc, Value, ValueDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

// --- audio/raw sink offer (broad; the concrete format arrives via dynamic caps) --------

const FAMILY: &str = "audio/raw";
static SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static SINK_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &SINK_FIELDS }];
static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: true, // format is announced at runtime
    validate: None,
}];
static DESC: ElementDesc = ElementDesc {
    name: "pipewireaudiosink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active, // its own thread; paces the graph by blocking on the device
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// A sample format we can hand to PipeWire (Copy/Send, so it crosses into the PW thread).
#[derive(Clone, Copy)]
enum Fmt {
    U8,
    S16,
    S24,
    S32,
    F32,
}

impl Fmt {
    fn from_name(name: &str) -> Option<Fmt> {
        Some(match name {
            "u8" => Fmt::U8,
            "s16" => Fmt::S16,
            "s24" => Fmt::S24,
            "s32" => Fmt::S32,
            "f32" => Fmt::F32,
            _ => return None,
        })
    }

    fn bytes(self) -> usize {
        match self {
            Fmt::U8 => 1,
            Fmt::S16 => 2,
            Fmt::S24 => 3,
            Fmt::S32 | Fmt::F32 => 4,
        }
    }

    fn spa(self) -> spa::param::audio::AudioFormat {
        use spa::param::audio::AudioFormat as A;
        match self {
            Fmt::U8 => A::U8,
            Fmt::S16 => A::S16LE,
            Fmt::S24 => A::S24LE,
            Fmt::S32 => A::S32LE,
            Fmt::F32 => A::F32LE,
        }
    }
}

// --- the PCM hand-off ring -------------------------------------------------------------

/// Bounded PCM hand-off between the streamcraft group thread (producer) and the PipeWire
/// real-time callback (consumer). A std mutex in the RT callback is not ideal — a lock-free
/// SPSC byte ring is the follow-up — but it is correct and simple for file playback.
struct Shared {
    ring: Mutex<Ring>,
    /// Signalled when the consumer frees space (a full-ring producer waits on it).
    space: Condvar,
    /// Signalled when the ring empties at EOS (the drain wait).
    drained: Condvar,
}

struct Ring {
    buf: VecDeque<u8>,
    cap: usize,
    eos: bool,
    closed: bool,
}

impl Shared {
    fn new(cap: usize) -> Self {
        Self {
            ring: Mutex::new(Ring {
                buf: VecDeque::with_capacity(cap),
                cap,
                eos: false,
                closed: false,
            }),
            space: Condvar::new(),
            drained: Condvar::new(),
        }
    }

    /// Producer: append PCM, blocking while the ring is full (backpressure). Returns early
    /// if the sink was closed (the PipeWire thread went away).
    fn push(&self, mut bytes: &[u8]) {
        let mut g = self.ring.lock().unwrap();
        while !bytes.is_empty() {
            while g.buf.len() >= g.cap && !g.closed {
                g = self.space.wait(g).unwrap();
            }
            if g.closed {
                return;
            }
            let room = g.cap - g.buf.len();
            let take = room.min(bytes.len());
            g.buf.extend(&bytes[..take]);
            bytes = &bytes[take..];
        }
    }

    /// Consumer (RT callback): fill `out` with available PCM, zero-padding the rest
    /// (underrun → silence, so the stream never glitches). `out.len()` is a frame multiple.
    fn pull(&self, out: &mut [u8]) {
        let mut g = self.ring.lock().unwrap();
        let n = out.len().min(g.buf.len());
        for (slot, b) in out[..n].iter_mut().zip(g.buf.drain(..n)) {
            *slot = b;
        }
        for slot in &mut out[n..] {
            *slot = 0;
        }
        if n > 0 {
            self.space.notify_one();
        }
        if g.buf.is_empty() && g.eos {
            self.drained.notify_all();
        }
    }

    /// Mark EOS and block until the ring drains (all audio played) or the sink closes.
    fn drain(&self) {
        let mut g = self.ring.lock().unwrap();
        g.eos = true;
        while !g.buf.is_empty() && !g.closed {
            g = self.drained.wait(g).unwrap();
        }
    }

    /// Close and wake anyone blocked (a producer waiting for space, a drainer waiting to
    /// empty), so they return promptly at shutdown.
    fn close(&self) {
        {
            let mut g = self.ring.lock().unwrap();
            g.closed = true;
        }
        self.space.notify_all();
        self.drained.notify_all();
    }
}

/// Run the PipeWire device loop on its own thread. All PipeWire objects live here (they are
/// not `Send`); the only shared state is the PCM ring and the quit channel.
fn run_pw(
    shared: Arc<Shared>,
    quit_rx: pw::channel::Receiver<()>,
    rate: u32,
    channels: u32,
    fmt: Fmt,
    stride: usize,
) -> Result<(), pw::Error> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&mainloop)?;
    let core = context.connect(None)?;

    let stream = pw::stream::Stream::new(
        &core,
        "streamcraft",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::MEDIA_CATEGORY => "Playback",
        },
    )?;

    let cb_shared = Arc::clone(&shared);
    let _listener = stream
        .add_local_listener_with_user_data(())
        .process(move |stream, ()| {
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if let Some(data) = datas.get_mut(0) {
                    let size = if let Some(slice) = data.data() {
                        let n = (slice.len() / stride) * stride; // whole frames only
                        cb_shared.pull(&mut slice[..n]);
                        n
                    } else {
                        0
                    };
                    let chunk = data.chunk_mut();
                    *chunk.offset_mut() = 0;
                    *chunk.stride_mut() = stride as _;
                    *chunk.size_mut() = size as _;
                }
            }
        })
        .register()?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(fmt.spa());
    audio_info.set_rate(rate);
    audio_info.set_channels(channels);

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
    let mut params = [Pod::from_bytes(&values).unwrap()];

    stream.connect(
        spa::utils::Direction::Output,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    // Cross-thread quit: the streamcraft side sends `()` on stop; quit the loop.
    let weak = mainloop.downgrade();
    let _recv = quit_rx.attach(mainloop.loop_(), move |_| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    mainloop.run();
    Ok(())
}

// --- the element -----------------------------------------------------------------------

/// Plays interleaved PCM through PipeWire. Construct with [`PipeWireAudioSink::new`].
#[derive(Default)]
pub struct PipeWireAudioSink {
    shared: Option<Arc<Shared>>,
    quit: Option<pw::channel::Sender<()>>,
    thread: Option<JoinHandle<()>>,
    started: bool,
}

impl PipeWireAudioSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the device from the negotiated `audio/raw` format and spawn the PipeWire
    /// thread. Idempotent — the first format wins (a mid-stream format change would need a
    /// device reconfigure, a follow-up).
    fn configure(&mut self, ctx: &Ctx, f: &FixedFormat) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }
        let rate = ctx
            .field_id("rate")
            .and_then(|id| f.get(id))
            .and_then(int_value)
            .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no rate"))? as u32;
        let channels = ctx
            .field_id("channels")
            .and_then(|id| f.get(id))
            .and_then(int_value)
            .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no channels"))?
            as u32;
        let sample = ctx
            .field_id("sample")
            .and_then(|id| f.get(id))
            .and_then(|v| match v {
                Value::Id(vid) => ctx.value_name(vid),
                _ => None,
            })
            .ok_or(Error::Todo("pipewireaudiosink: negotiated format has no sample format"))?;
        let fmt = Fmt::from_name(sample)
            .ok_or_else(|| Error::Resource(format!("pipewireaudiosink: unsupported sample '{sample}'")))?;

        let stride = fmt.bytes() * channels as usize;
        // ~0.5 s of jitter buffer, floored so tiny rates still buffer sensibly.
        let cap = stride * (rate as usize / 2).max(4096);
        let shared = Arc::new(Shared::new(cap));

        let (tx, rx) = pw::channel::channel::<()>();
        let thread_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("sc-pipewire".into())
            .spawn(move || {
                if let Err(e) = run_pw(Arc::clone(&thread_shared), rx, rate, channels, fmt, stride) {
                    eprintln!("pipewireaudiosink: PipeWire error: {e}");
                }
                thread_shared.close(); // unblock the producer/drainer if the loop exits
            })
            .map_err(|e| Error::Resource(format!("pipewireaudiosink: thread spawn: {e}")))?;

        self.shared = Some(shared);
        self.quit = Some(tx);
        self.thread = Some(handle);
        self.started = true;
        Ok(())
    }
}

fn int_value(v: Value) -> Option<i64> {
    match v {
        Value::Int(n) => Some(n),
        _ => None,
    }
}

impl Element for PipeWireAudioSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        // The device is configured lazily, once the format is known (FormatChange or a
        // fully-fixed link-time format read from `ctx.negotiated`).
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            // No FormatChange yet? Try a fully-fixed link-time format.
            match ctx.negotiated(PadId(0)).cloned() {
                Some(f) => self.configure(ctx, &f)?,
                None => return Ok(Flow::Ok), // wait for the format before touching the device
            }
        }
        while let Some(buf) = inputs.pop() {
            if let Some(shared) = &self.shared {
                shared.push(buf.memory.data()); // blocks when full → backpressure
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            // The pipeline delivers EOS at end-of-stream: play out the buffered audio.
            Event::Eos => {
                if let Some(shared) = &self.shared {
                    shared.drain();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        if let Some(shared) = &self.shared {
            shared.close();
        }
        if let Some(tx) = self.quit.take() {
            let _ = tx.send(()); // ask the PipeWire loop to quit
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
        self.shared = None;
        self.started = false;
    }
}

// The PipeWire integration itself needs a running server (verified by the `play` example
// on a real desktop); these tests cover the device-independent PCM hand-off ring, which is
// the part with the concurrency subtlety.
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn push_pull_roundtrips_bytes() {
        let s = Shared::new(1024);
        s.push(&[1, 2, 3, 4]);
        let mut out = [0u8; 4];
        s.pull(&mut out);
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn pull_underrun_is_silence() {
        let s = Shared::new(1024);
        s.push(&[9, 9]);
        let mut out = [7u8; 6];
        s.pull(&mut out);
        assert_eq!(out, [9, 9, 0, 0, 0, 0], "available bytes, then silence");
    }

    #[test]
    fn push_blocks_until_space_then_completes() {
        // A full ring makes the producer block until the consumer frees space — the
        // backpressure that paces the graph to the device.
        let s = Arc::new(Shared::new(4));
        s.push(&[1, 2, 3, 4]); // fills it
        let s2 = Arc::clone(&s);
        let producer = thread::spawn(move || s2.push(&[5, 6, 7, 8])); // blocks
        thread::sleep(Duration::from_millis(20));
        let mut a = [0u8; 4];
        s.pull(&mut a);
        assert_eq!(a, [1, 2, 3, 4]);
        producer.join().unwrap(); // unblocked by the pull, enqueues 5..8
        let mut b = [0u8; 4];
        s.pull(&mut b);
        assert_eq!(b, [5, 6, 7, 8]);
    }

    #[test]
    fn drain_returns_once_the_ring_empties_at_eos() {
        let s = Arc::new(Shared::new(1024));
        s.push(&[1, 2, 3, 4]);
        let s2 = Arc::clone(&s);
        let drainer = thread::spawn(move || s2.drain()); // blocks until empty
        thread::sleep(Duration::from_millis(20));
        let mut out = [0u8; 4];
        s.pull(&mut out); // empties → drain wakes and returns
        drainer.join().unwrap();
    }

    #[test]
    fn close_unblocks_a_full_producer() {
        let s = Arc::new(Shared::new(4));
        s.push(&[1, 2, 3, 4]);
        let s2 = Arc::clone(&s);
        let producer = thread::spawn(move || s2.push(&[5, 6, 7, 8])); // blocks (full)
        thread::sleep(Duration::from_millis(20));
        s.close(); // must wake the producer, which returns (dropping the data)
        producer.join().unwrap();
    }
}
