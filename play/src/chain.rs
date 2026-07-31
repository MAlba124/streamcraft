//! The **canonical-output audio chain** (spec: `gapless.md` Phase 2 — "Fixed canonical format
//! chosen at creation (f32 48 kHz stereo); every track chain is autoplugged to converge on it
//! — `wire_audio_chain`'s convert+resample fallback becomes the *forced* mode").
//!
//! [`crate::autoplug::wire_audio_chain`] is a *negotiation* strategy: try the cheapest wiring
//! (decoder straight into the sink), then insert `audioconvert`, then `audioconvert +
//! audioresample`, stopping at the first tier that links. That is exactly right when the sink
//! adapts to the content — which `pipewireaudiosink` does, since it configures its device from
//! the first `FormatChange` a decoder announces.
//!
//! It is exactly wrong when the sink **cannot** adapt. A shared `AudioOut` latches one device
//! format for the life of the application so that track *n+1*'s producer can attach behind
//! track *n*'s in-flight bytes without re-opening the device — that is the whole mechanism
//! that makes the handoff sample-continuous. A chain that negotiated 44 100 Hz because the
//! first tier happened to link would then be unattachable, and the failure would surface as a
//! dead second track rather than as a link error.
//!
//! So this module offers the *other* mode, alongside the tiered one rather than replacing it:
//! **every stage is forced, unconditionally, and the chain provably ends in
//! [`CANONICAL`] — f32, 48 000 Hz, 2 channels — whatever the decoder emits.**
//!
//! # The chain, and why the stages are in this order
//!
//! ```text
//! decoder → audioconvert(f32) → audiostereo → audioresample(48k) → [audiostretch] → [audiogain] → sink
//! ```
//!
//! 1. **`audioconvert(f32)` first — before, not after, the DSP.** Both later DSP stages work
//!    internally in float and re-encode to whatever sample format arrived: [`AudioResample`]
//!    decodes to `f32`, runs the polyphase FIR, and re-encodes to *its input format*
//!    (`resample_element.rs`), and [`AudioStretch`] does the same around its overlap-add. Feed
//!    them `s16` and each one silently re-quantises its own output to 16 bits — twice over,
//!    with the FIR's ringing and the stretch's cross-fade both landing on a 16-bit grid before
//!    anything downstream can see them. Converting up front makes the entire tail of the chain
//!    end-to-end float: exactly one quantisation, in the sink, at the device's depth. The
//!    BS.775 fold in step 2 likewise mixes at float precision and, in `f32`, is not narrowed
//!    back to an integer grid before the resampler reads it.
//! 2. **`audiostereo` second — the single place the channel count is decided.** It folds `>2`
//!    with the ITU-R BS.775-3 matrix and *duplicates* mono to stereo (see [`crate::stereo`] for
//!    why no existing element does the latter). Placing it ahead of the expensive stages means
//!    the resampler and the stretcher run exactly two lanes for every input layout — a 7.1
//!    track resamples 2 channels, not 8. The one case that pays is mono, which resamples two
//!    identical lanes instead of one; that is a deliberate trade (one extra 32-tap FIR lane, a
//!    few MFLOP/s) for having the stereo invariant established once, at a single point,
//!    instead of being re-derived at the end of the chain.
//! 3. **`audioresample(48k)` third.** The expensive stage, so it runs last among the mandatory
//!    ones — on two `f32` lanes, the cheapest shape it can be given.
//! 4. **`[audiostretch]` then `[audiogain]` last.** Both then operate on canonical
//!    f32/48 kHz/2 ch: cheapest (two lanes), and *format-stable* — the format they latch at
//!    their first negotiation is the canonical one and nothing upstream can change it, which
//!    matters because every glue element in this workspace latches its input format for good
//!    and only warns if it later disagrees.
//! 5. **Gain after stretch, not before.** A gain change is a *ramp* ([`RAMP_MS`] ms of
//!    declick); applied before the stretcher, that ramp would be smeared across the WSOLA
//!    overlap-add and stop being the linear ramp it was designed as. Last also means the live
//!    volume/mute knob acts on precisely the bytes the sink receives.
//!
//! # Identity stages are not special-cased
//!
//! An `audiogain` at unity forwards its buffer **by move** — no pool slot, no memcpy, no
//! arithmetic — and an `audiostretch` at rate 1.0 bypasses likewise, so a stage that is
//! requested but currently a no-op costs a comparison per buffer. The chain therefore inserts
//! a stage whenever the caller asks for one and lets the element's own fast path handle
//! identity, rather than growing graph-shape special cases whose absence would then have to be
//! discovered at the point a live property is set. (`audioconvert`, `audiostereo` and
//! `audioresample` do the same when their input already matches the target: byte-for-byte
//! pass-through, and `audiostereo` forwards already-stereo buffers by move.)
//!
//! # ReplayGain is decibels; this element takes a multiplier
//!
//! `audiogain`'s `gain` is a **linear** multiplier and deliberately owns no dB policy (a dB
//! knob would have to invent a floor for −∞). [`ChainSpec::gain_db`] is decibels because that
//! is what a tag carries, and [`gain_db_to_linear`] is the documented, tested conversion —
//! applied once here at build time, and again by [`ChainHandles::set_gain_db`] when the app
//! changes it live.

use profluens_core::element::{Direction, Element};
use profluens_core::error::Error;
use profluens_core::format::Value;
use profluens_core::id::ElementId;
use profluens_core::pipeline::Pipeline;
use profluens_core::props::PropHandle;

use profluens_audio::{
    AudioConvert, AudioFormat, AudioGain, AudioResample, AudioStretch, SampleFormat,
    StretchPosition, GAIN_MAX, GAIN_MIN, RATE_MAX, RATE_MIN,
};

use crate::autoplug::{SinkChoice, AUDIO_SLOT, AUDIO_SLOTS};
use crate::stereo::AudioStereo;

// --- the canonical format --------------------------------------------------------------

/// The sample rate every canonical chain converges on. 48 kHz is PipeWire's native graph rate,
/// so the server does no rate conversion of its own on the common path, and it is an integer
/// multiple of nothing awkward — the two rates that matter (44 100 and 48 000) both resample
/// cleanly with the polyphase filter's L/M reduction.
pub const CANONICAL_RATE: u32 = 48_000;
/// The channel count every canonical chain converges on.
pub const CANONICAL_CHANNELS: u16 = 2;
/// The sample format every canonical chain converges on. `f32` is PipeWire-native, is what both
/// DSP stages work in internally, and is the workspace's designated headroom format
/// (`audiogain` deliberately does not clamp it).
pub const CANONICAL_SAMPLE: SampleFormat = SampleFormat::F32;

/// `f32 48 000 Hz 2 ch` — the format a canonical chain's sink is guaranteed to be offered.
pub const CANONICAL: AudioFormat =
    AudioFormat::new(CANONICAL_RATE, CANONICAL_CHANNELS, CANONICAL_SAMPLE);

// --- dB → linear ------------------------------------------------------------------------

/// ReplayGain decibels to the linear multiplier `audiogain` takes: `10^(dB/20)`.
///
/// The amplitude (not power) form, which is what ReplayGain's `REPLAYGAIN_TRACK_GAIN` /
/// `R128_TRACK_GAIN` corrections mean — a −6.02 dB correction is a factor of ½.
///
/// Two edges are pinned here rather than left to `powf`:
/// * **`-inf` dB → 0.0** (silence) falls out of `powf` correctly and is kept.
/// * **A `NaN` dB → 1.0** (unity). A tag that parsed to garbage must not silence or blow up a
///   track; the honest response to "I could not read the correction" is "apply no correction".
///
/// The result is clamped to `audiogain`'s own `[GAIN_MIN, GAIN_MAX]` range so the value this
/// function reports is the value the element will actually apply (the element clamps too;
/// doing it here as well is what makes [`ChainHandles::gain_linear`] truthful).
pub fn gain_db_to_linear(db: f32) -> f32 {
    if db.is_nan() {
        return 1.0;
    }
    10f32.powf(db / 20.0).clamp(GAIN_MIN, GAIN_MAX)
}

// --- the spec ---------------------------------------------------------------------------

/// Where a canonical chain's audio ends up.
///
/// [`Policy`](Self::Policy) reproduces the existing per-[`SinkChoice`] sink construction
/// verbatim, so the canonical mode is usable with today's device/drop sinks;
/// [`Injected`](Self::Injected) is the seam the gapless engine needs — the caller builds the
/// sink (a producer attached to a shared `AudioOut`, a capture backend, a test recorder) and
/// this module wires the canonical audio into it.
pub enum SinkSpec {
    /// Build the sink from the policy: [`SinkChoice::Device`] (and the video-only
    /// `External*` choices, which also mean "audio to the device") → `pipewireaudiosink`;
    /// [`SinkChoice::Drop`] → a `TestSink` drop.
    Policy(SinkChoice),
    /// Use the caller's element as the sink. Its **first sink-direction pad** receives the
    /// canonical audio — the name is read off the element's own descriptor, so an element that
    /// calls its input something other than `"sink"` still links.
    Injected(Box<dyn Element>),
}

impl SinkSpec {
    /// The audio device sink (`pipewireaudiosink`).
    pub fn device() -> Self {
        SinkSpec::Policy(SinkChoice::Device)
    }

    /// A headless drop sink — decode-and-discard, no device grabbed.
    pub fn drop_sink() -> Self {
        SinkSpec::Policy(SinkChoice::Drop)
    }

    /// The `track <pad>: …` label for this sink. Borrowing (rather than consuming) matters:
    /// the spec owns a boxed element and cannot be cloned to be described.
    pub fn label(&self) -> &'static str {
        match self {
            SinkSpec::Policy(SinkChoice::Drop) => "canonical f32/48k/2ch → drop sink",
            SinkSpec::Policy(_) => "canonical f32/48k/2ch → audio device",
            SinkSpec::Injected(_) => "canonical f32/48k/2ch → injected sink",
        }
    }
}

/// What one track's canonical chain should contain, beyond the mandatory conform stages.
///
/// Both optional stages are `Option` rather than "1.0 means absent": the difference is
/// *observable*, because an inserted-but-identity stage still yields an [`ElementId`] the app
/// can drive live (a volume slider that only works once the track happens to have ReplayGain
/// metadata would be a trap). Ask for the stage when you intend to control it.
pub struct ChainSpec {
    /// Per-track ReplayGain in **decibels**, converted here by [`gain_db_to_linear`]. `Some`
    /// inserts an `audiogain`; `None` inserts none at all. Pass `Some(0.0)` for "I want a live
    /// volume knob but no initial correction".
    pub gain_db: Option<f32>,
    /// Playback rate for `audiostretch` (1.5 = 1.5× faster, pitch preserved). `Some` inserts
    /// the stage and returns a [`StretchPosition`]; `None` inserts none.
    pub stretch: Option<f32>,
    /// Where the canonical audio goes.
    pub sink: SinkSpec,
}

impl ChainSpec {
    /// A bare canonical chain into `sink` — no gain, no stretch.
    pub fn new(sink: SinkSpec) -> Self {
        ChainSpec { gain_db: None, stretch: None, sink }
    }

    /// A bare canonical chain into the audio device.
    pub fn device() -> Self {
        Self::new(SinkSpec::device())
    }

    /// A bare canonical chain into a headless drop sink.
    pub fn drop_sink() -> Self {
        Self::new(SinkSpec::drop_sink())
    }

    /// A bare canonical chain into the caller's sink element.
    // COLD: one boxed sink per track chain, at build time.
    #[allow(clippy::disallowed_methods)]
    pub fn injected(sink: impl Element + 'static) -> Self {
        Self::new(SinkSpec::Injected(Box::new(sink)))
    }

    /// Add a ReplayGain stage at `db` decibels (and a live volume knob).
    pub fn with_gain_db(mut self, db: f32) -> Self {
        self.gain_db = Some(db);
        self
    }

    /// Add a time-stretch stage at `rate` (pitch preserved).
    pub fn with_stretch(mut self, rate: f32) -> Self {
        self.stretch = Some(rate);
        self
    }
}

// --- the handles the engine needs --------------------------------------------------------

/// Everything an application needs to drive a built canonical chain: the ids of the stages it
/// may address live, and the out-of-band handles it cannot get from an id.
///
/// The ids double as the chain's shape, which is what the tests assert on — `stretch.is_some()`
/// is "a stretch stage exists", not a guess from a summary string.
pub struct ChainHandles {
    /// `audioconvert` → the canonical sample format.
    pub convert: ElementId,
    /// `audiostereo` → exactly [`CANONICAL_CHANNELS`] channels.
    pub stereo: ElementId,
    /// `audioresample` → [`CANONICAL_RATE`].
    pub resample: ElementId,
    /// The `audiostretch` stage, when one was requested. Set its `rate` prop live through
    /// [`PropHandle::set`] — see [`ChainHandles::set_rate`].
    pub stretch: Option<ElementId>,
    /// The stretcher's **source-time** position counter, taken before the element was moved
    /// into the pipeline. `Some` exactly when [`stretch`](Self::stretch) is.
    pub stretch_position: Option<StretchPosition>,
    /// The `audiogain` stage, when one was requested — the prop path for live volume and for
    /// ReplayGain changes (`"gain"`, `"mute"`; see [`set_gain_db`](Self::set_gain_db) /
    /// [`set_mute`](Self::set_mute)).
    pub gain: Option<ElementId>,
    /// The linear multiplier the gain stage was *built* with, post-clamp — i.e. exactly what
    /// [`gain_db_to_linear`] made of [`ChainSpec::gain_db`]. `Some` exactly when
    /// [`gain`](Self::gain) is.
    pub gain_linear: Option<f32>,
    /// The sink the chain ends in — the injected element, or the one built from the policy.
    pub sink: ElementId,
    /// The sink pad the chain linked into (`"sink"` for every in-tree sink; read off the
    /// injected element's descriptor otherwise).
    pub sink_pad: &'static str,
}

/// The denominator used to spell an `f32` gain as an exact [`Value::Rat`].
///
/// The value universe has no float, and `audiogain` accepts a rational, a whole multiple, or
/// decimal *text*. A rational is the only spelling that needs neither string interning at
/// runtime nor a lossy decimal round-trip. 1/65536 of a linear step is ~0.00013 dB near unity
/// — some four orders of magnitude below audibility, and below the resolution of any
/// ReplayGain tag.
const GAIN_RAT_DEN: i32 = 1 << 16;

impl ChainHandles {
    /// Spell a linear gain as the [`Value`] `audiogain` reads. Public because an app that
    /// drives the prop table itself (an introspection client, a scripted mixer) should not
    /// have to rediscover the rational encoding.
    pub fn gain_value(linear: f32) -> Value {
        let clamped = if linear.is_nan() { 1.0 } else { linear.clamp(GAIN_MIN, GAIN_MAX) };
        Value::Rat((clamped * GAIN_RAT_DEN as f32).round() as i32, GAIN_RAT_DEN)
    }

    /// Set the gain stage's **linear** multiplier live. `Err` when this chain has no gain stage
    /// (ask for one with [`ChainSpec::with_gain_db`]) or the property is rejected.
    ///
    /// Lands at the next batch boundary and starts `audiogain`'s declick ramp, so a volume
    /// slider driving this at UI frame rate is well-behaved.
    pub fn set_gain(&self, props: &PropHandle, linear: f32) -> Result<(), Error> {
        let id = self.gain.ok_or(Error::Todo(
            "this chain has no audiogain stage (build it with ChainSpec::with_gain_db)",
        ))?;
        props.set(id, "gain", Self::gain_value(linear))
    }

    /// Set the gain stage from **decibels** — the ReplayGain path. Converts with
    /// [`gain_db_to_linear`], then defers to [`set_gain`](Self::set_gain).
    pub fn set_gain_db(&self, props: &PropHandle, db: f32) -> Result<(), Error> {
        self.set_gain(props, gain_db_to_linear(db))
    }

    /// Mute or unmute live, without forgetting the configured gain.
    pub fn set_mute(&self, props: &PropHandle, mute: bool) -> Result<(), Error> {
        let id = self.gain.ok_or(Error::Todo(
            "this chain has no audiogain stage (build it with ChainSpec::with_gain_db)",
        ))?;
        props.set(id, "mute", Value::Int(i64::from(mute)))
    }

    /// Set the stretch stage's playback rate live. `Err` when this chain has no stretch stage.
    ///
    /// Spelled as an exact rational for the same reason as the gain (see [`gain_value`] —
    /// no float in the value universe, no runtime string interning).
    ///
    /// [`gain_value`]: ChainHandles::gain_value
    pub fn set_rate(&self, props: &PropHandle, rate: f32) -> Result<(), Error> {
        let id = self.stretch.ok_or(Error::Todo(
            "this chain has no audiostretch stage (build it with ChainSpec::with_stretch)",
        ))?;
        // Clamped to the element's own range before scaling, so the numerator cannot overflow
        // `i32` on a wild input (and so the value set is the value applied).
        let r = if rate.is_finite() { rate.clamp(RATE_MIN, RATE_MAX) } else { 1.0 };
        props.set(id, "rate", Value::Rat((r * GAIN_RAT_DEN as f32).round() as i32, GAIN_RAT_DEN))
    }

    /// Every stage of the chain in order, as element ids — for a tap/stats watcher.
    pub fn stages(&self) -> Vec<(&'static str, ElementId)> {
        // COLD: one small Vec per built chain, for a stats watcher — never on a buffer path.
        #[allow(clippy::disallowed_methods)]
        let mut v = Vec::with_capacity(6);
        v.push(("convert", self.convert));
        v.push(("stereo", self.stereo));
        v.push(("resample", self.resample));
        if let Some(id) = self.stretch {
            v.push(("stretch", id));
        }
        if let Some(id) = self.gain {
            v.push(("gain", id));
        }
        v.push(("sink", self.sink));
        v
    }
}

// --- the builder --------------------------------------------------------------------------

/// Wire the canonical chain from `head` (an element+pad emitting `audio/raw`, normally a
/// decoder's `"src"`) through to the spec's sink, and return the handles.
///
/// Every stage gets its **own** small-slot pool. That is not decoration: `try_alloc` hands out
/// whole slots, so a few-KB PCM buffer drawn from the shared 4 MiB video pool pins a 4 MiB slot,
/// and a couple of dozen in flight exhaust it — the measured failure mode documented at
/// [`crate::autoplug::link_audio`]. The tiered chain gives its downmix a private pool for
/// exactly this reason; the canonical chain has more stages, so it gives one to each.
///
/// Errors are one-liners naming the edge that would not negotiate. A failure here is a real
/// wiring bug (each stage advertises broad `audio/raw`), not a "try the next tier" signal —
/// there is no next tier, by design.
pub fn wire_canonical_audio_chain(
    p: &mut Pipeline,
    head: (ElementId, &str),
    spec: ChainSpec,
) -> Result<ChainHandles, String> {
    let ChainSpec { gain_db, stretch, sink } = spec;

    // 1. To the canonical sample format, before any DSP — see the module docs on why this is
    //    first and not last.
    let convert = p.add(AudioConvert::new(CANONICAL_SAMPLE));
    p.set_element_pool(convert, AUDIO_SLOT, AUDIO_SLOTS);
    p.link(head, (convert, "sink"))
        .map_err(|e| format!("canonical chain: {} ! audioconvert(f32): {e:?}", head.1))?;

    // 2. To exactly two channels — the one place the channel count is decided.
    let stereo = p.add(AudioStereo::new());
    p.set_element_pool(stereo, AUDIO_SLOT, AUDIO_SLOTS);
    p.link((convert, "src"), (stereo, "sink"))
        .map_err(|e| format!("canonical chain: audioconvert ! audiostereo: {e:?}"))?;

    // 3. To the canonical rate.
    let resample = p.add(AudioResample::new(CANONICAL_RATE));
    p.set_element_pool(resample, AUDIO_SLOT, AUDIO_SLOTS);
    p.link((stereo, "src"), (resample, "sink"))
        .map_err(|e| format!("canonical chain: audiostereo ! audioresample(48k): {e:?}"))?;

    let mut tail = resample;

    // 4. Optional time-stretch, on canonical f32/48k/2ch. The position handle must be taken
    //    before the element is moved into the pipeline.
    let (stretch_id, stretch_position) = match stretch {
        Some(rate) => {
            let el = AudioStretch::new(rate);
            let pos = el.position();
            let id = p.add(el);
            p.set_element_pool(id, AUDIO_SLOT, AUDIO_SLOTS);
            p.link((tail, "src"), (id, "sink"))
                .map_err(|e| format!("canonical chain: audioresample ! audiostretch: {e:?}"))?;
            tail = id;
            (Some(id), Some(pos))
        }
        None => (None, None),
    };

    // 5. Optional gain, last before the sink — see the module docs on ordering.
    let (gain_id, gain_linear) = match gain_db {
        Some(db) => {
            let linear = gain_db_to_linear(db);
            let id = p.add(AudioGain::new(linear));
            p.set_element_pool(id, AUDIO_SLOT, AUDIO_SLOTS);
            p.link((tail, "src"), (id, "sink"))
                .map_err(|e| format!("canonical chain: {} ! audiogain: {e:?}", stage_name(tail, stretch_id)))?;
            tail = id;
            (Some(id), Some(linear))
        }
        None => (None, None),
    };

    // 6. The sink: the caller's element, or one built from the policy.
    let (sink_id, sink_pad) = match sink {
        SinkSpec::Injected(el) => {
            // Read the input pad name off the element's own descriptor rather than assuming
            // `"sink"` — the descriptor is `&'static`, so the name outlives the move.
            let pad = first_sink_pad(el.as_ref()).ok_or_else(|| {
                format!(
                    "canonical chain: the injected sink '{}' declares no sink-direction pad",
                    el.desc().name
                )
            })?;
            (p.add_boxed(el), pad)
        }
        SinkSpec::Policy(SinkChoice::Drop) => {
            let (ts, _stats) = profluens_elements::testing::TestSink::new();
            (p.add(ts), "sink")
        }
        // Device — and the video-only External* choices, which also mean "audio to the device"
        // (the same rule `link_audio` follows).
        SinkSpec::Policy(_) => (p.add(pf_pipewire::PipeWireAudioSink::new()), "sink"),
    };
    p.link((tail, "src"), (sink_id, sink_pad))
        .map_err(|e| format!("canonical chain: chain tail ! sink.{sink_pad}: {e:?}"))?;

    Ok(ChainHandles {
        convert,
        stereo,
        resample,
        stretch: stretch_id,
        stretch_position,
        gain: gain_id,
        gain_linear,
        sink: sink_id,
        sink_pad,
    })
}

/// The name of the stage `tail` currently is, for an error message.
fn stage_name(tail: ElementId, stretch: Option<ElementId>) -> &'static str {
    if Some(tail) == stretch {
        "audiostretch"
    } else {
        "audioresample"
    }
}

/// The first sink-direction pad an element declares — the pad a chain links its tail into.
/// `None` for an element with no input at all (a source handed in as a sink by mistake).
fn first_sink_pad(el: &dyn Element) -> Option<&'static str> {
    el.desc().pads.iter().find(|p| p.direction == Direction::Sink).map(|p| p.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaygain_decibels_convert_to_the_documented_multipliers() {
        // The amplitude form, 10^(dB/20): −6.02 dB is a factor of one half, +6.02 dB is two,
        // 0 dB is unity. These are the numbers a ReplayGain tag means.
        assert!((gain_db_to_linear(0.0) - 1.0).abs() < 1e-6, "0 dB is unity");
        assert!((gain_db_to_linear(-6.0206) - 0.5).abs() < 1e-4, "−6.02 dB is a half");
        assert!((gain_db_to_linear(6.0206) - 2.0).abs() < 1e-3, "+6.02 dB is a double");
        assert!((gain_db_to_linear(-20.0) - 0.1).abs() < 1e-6, "−20 dB is a tenth");
    }

    #[test]
    fn gain_edges_are_pinned_rather_than_left_to_powf() {
        assert_eq!(gain_db_to_linear(f32::NEG_INFINITY), 0.0, "−inf dB is silence");
        assert_eq!(gain_db_to_linear(f32::NAN), 1.0, "an unreadable tag applies no correction");
        // Clamped into audiogain's own range, so `gain_linear` reports what will be applied.
        assert_eq!(gain_db_to_linear(60.0), GAIN_MAX, "+60 dB clamps to the element's ceiling");
        assert!(gain_db_to_linear(-200.0) >= GAIN_MIN);
    }

    #[test]
    fn the_gain_value_spelling_round_trips_within_a_rational_step() {
        for linear in [0.0f32, 0.25, 0.5, 0.7943, 1.0, 2.0, 4.0] {
            let Value::Rat(n, d) = ChainHandles::gain_value(linear) else {
                panic!("gain must be spelled as an exact rational");
            };
            let back = n as f32 / d as f32;
            assert!(
                (back - linear).abs() <= 1.0 / GAIN_RAT_DEN as f32,
                "{linear} round-tripped to {back}"
            );
        }
    }

    #[test]
    fn the_canonical_format_is_f32_48k_stereo() {
        // The one invariant every canonical chain exists to guarantee. Pinned as a test so a
        // change to it is a deliberate act with a failing assertion attached.
        assert_eq!(CANONICAL.sample_rate, 48_000);
        assert_eq!(CANONICAL.channels, 2);
        assert_eq!(CANONICAL.format, SampleFormat::F32);
        assert_eq!(CANONICAL.frame_stride(), 8, "2 ch × 4 bytes");
    }
}
