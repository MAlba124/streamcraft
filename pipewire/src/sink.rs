//! `pipewireaudiosink` — plays interleaved PCM through PipeWire (spec: Milestone
//! applications — play an audio file).
//!
//! It is an **active** element on its own thread. Its `process()` pushes decoded PCM into a
//! bounded lock-free SPSC byte ring ([`crate::ring`]); a dedicated PipeWire thread runs the
//! device loop and its **real-time** callback pulls from that ring to fill device buffers.
//! The RT callback must never lock or allocate, so the ring's consumer side is wait-free —
//! the only blocking (a full-ring producer parking; the EOS drain) happens on the streamcraft
//! render thread. When the ring is full, `process()` blocks, so the whole pipeline is paced
//! by the audio device (backpressure is the clock; full clock-slaving through PipeWire for
//! A/V sync is a follow-up). The PCM format is data-dependent, so the sink advertises a broad
//! `dynamic` `audio/raw` sink and configures itself from the runtime `FormatChange` a decoder
//! announces (spec: dynamic caps). On EOS the pipeline delivers `Event::Eos`, and the sink
//! drains the ring (plays out) before returning.

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

use crate::ring::{self, Consumer, Producer};

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

// --- the PipeWire device thread --------------------------------------------------------

/// Run the PipeWire device loop on its own thread. All PipeWire objects live here (they are
/// not `Send`); the shared state is the ring's [`Consumer`] (pulled from the RT callback) and
/// the quit channel. The `Consumer` is `Send` but the loop objects are not, so it is moved
/// in here and never leaves.
fn run_pw(
    consumer: Consumer,
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

    // The RT process callback: pull PCM from the ring to fill the device buffer. This runs
    // on PipeWire's real-time thread, so `Consumer::pull` is wait-free and allocation-free —
    // no mutex is taken here. The consumer is owned by the callback (moved in).
    let _listener = stream
        .add_local_listener_with_user_data(())
        .process(move |stream, ()| {
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if let Some(data) = datas.get_mut(0) {
                    let size = if let Some(slice) = data.data() {
                        let n = (slice.len() / stride) * stride; // whole frames only
                        consumer.pull(&mut slice[..n]);
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
    /// Producer end of the PCM ring; the render thread pushes decoded PCM here (blocking on
    /// a full ring → backpressure) and signals EOS via `drain`.
    producer: Option<Producer>,
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
        let (producer, consumer) = ring::spsc(cap);

        let (tx, rx) = pw::channel::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("sc-pipewire".into())
            .spawn(move || {
                // `consumer` is moved onto the PW thread and pulled from its RT callback; on
                // any early exit it drops here, closing the ring so a parked producer wakes.
                if let Err(e) = run_pw(consumer, rx, rate, channels, fmt, stride) {
                    eprintln!("pipewireaudiosink: PipeWire error: {e}");
                }
            })
            .map_err(|e| Error::Resource(format!("pipewireaudiosink: thread spawn: {e}")))?;

        self.producer = Some(producer);
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
            if let Some(producer) = &self.producer {
                producer.push(buf.memory.data()); // blocks when full → backpressure
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            Event::FormatChange(f) => self.configure(ctx, f)?,
            // The pipeline delivers EOS at end-of-stream: play out the buffered audio.
            Event::Eos => {
                if let Some(producer) = &self.producer {
                    producer.drain();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        if let Some(producer) = &self.producer {
            producer.close();
        }
        if let Some(tx) = self.quit.take() {
            let _ = tx.send(()); // ask the PipeWire loop to quit
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
        self.producer = None;
        self.started = false;
    }
}

// The PipeWire integration itself needs a running server (verified by the `play` example on a
// real desktop). The device-independent part with the concurrency subtlety — the lock-free
// PCM hand-off ring the RT callback pulls from — is unit-tested in [`crate::ring`].
