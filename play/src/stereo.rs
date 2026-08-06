//! `audiostereo` — conform **any** interleaved `audio/raw` channel count to exactly **two**
//! channels (spec: Formats — dynamic caps; Writing elements). This is the channel half of the
//! canonical-output chain ([`crate::chain`]): whatever a decoder emits — 1, 2, 5.1, 7.1 —
//! leaves this element as stereo, at the same rate and in the same sample format.
//!
//! # Why this element exists (and why `audiodownmix` is not enough)
//!
//! [`AudioDownmix`](profluens_audio::AudioDownmix) folds `>2` channels to stereo and passes
//! `<=2` through **unchanged** — mono stays mono, by explicit design ("duplicating it is the
//! converter's job, not the downmixer's", `audio/src/downmix_element.rs`). But
//! [`AudioConvert`](profluens_audio::AudioConvert) changes only the sample *representation*
//! and [`AudioResample`](profluens_audio::AudioResample) only the *rate*; both preserve the
//! channel count verbatim. So no element in the workspace performs a mono→stereo **upmix**,
//! and a mono track fed through `downmix → convert → resample` still arrives at the sink as
//! one channel.
//!
//! That is fine for the ordinary `pipewireaudiosink`, which negotiates whatever channel count
//! it is offered. It is *not* fine for a shared `AudioOut` whose device format is latched once
//! at creation (gapless.md, Phase 2): every track chain must converge on the same canonical
//! format, or the second track cannot attach. Hence: one element that owns the whole
//! "channels → 2" invariant, so the canonical chain has exactly one place where the channel
//! count is decided.
//!
//! # What it does
//!
//! | input channels | operation | library call |
//! |---|---|---|
//! | 1 | duplicate the sole channel into both outputs | [`remap_channels`] `(1, 2)` |
//! | 2 | pass through (by **move** when the buffer is whole frames) | — |
//! | > 2 | ITU-R BS.775-3 "Lo/Ro" fold | [`downmix_to_stereo`] |
//!
//! Both folds are the *same* tested functions `audiodownmix` uses, so a 5.1 stream folded here
//! is bit-identical to one folded there. Nothing else changes: same rate, same sample format,
//! and the output is written **straight into a pool slot** (no staging buffer), so `process()`
//! performs no heap allocation.
//!
//! A passive transform like the rest of the `audio/raw` glue: it inlines into the upstream
//! group and never blocks. It is both a consumer and a producer of dynamic caps — it learns
//! `rate`/`channels`/`sample` from the negotiated sink caps through the shared vocabulary (the
//! `audioconvert` pattern) and announces `channels = 2` on its src pad the moment the input is
//! known.
//!
//! # Where this element belongs
//!
//! Its natural home is `profluens-audio`, beside `audiodownmix` — it uses only that crate's
//! public library functions and nothing from `pf-play`. It lives here because the canonical
//! chain is the only consumer today and `play/` is the crate under change; promoting it is a
//! file move plus a `pub use`.

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

use profluens_audio::{
    downmix_to_stereo, negotiated_audio_format, remap_channels, AudioFormat, FAMILY,
    FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE,
};

const SINK: PadId = PadId(0);
const SRC: PadId = PadId(1);

/// The channel count every output buffer carries. Not a knob: this element exists to make the
/// canonical chain's stereo invariant unconditional, and a configurable target would put the
/// "which channel count?" decision back in two places.
pub const OUT_CHANNELS: u16 = 2;

// Both pads speak `audio/raw` (any rate/channels/sample) plus the `bytes` escape, mirroring
// `audioconvert`/`audiodownmix`. The sink takes whatever the upstream fixates; the src is
// `dynamic` and announces its concrete stereo output at runtime. Every sample-format name is
// offered so the `sample` value is interned for the announcement to resolve against.
static SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];
static FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: FIELD_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: FIELD_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: FIELD_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
static OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &FIELDS }, OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: true, // the stereo output format is announced at runtime
        validate: None,
    },
];

/// Config-free: the target is fixed at stereo (see [`OUT_CHANNELS`]).
static PROPS: [PropDesc; 0] = [];

// COLD: the `make_default` factory boxes one element at construction, not per buffer.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "audiostereo",
    pads: &PADS,
    props: &PROPS,
    // Passive: a pure transform that inlines into the upstream group (spec: Scheduling).
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        // Nothing is held back but at most one partial interchannel frame, which is not a
        // whole frame of delay.
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(AudioStereo::new())),
};

/// Conforms interleaved PCM of any channel count to stereo, preserving rate and sample format.
#[derive(Default)]
pub struct AudioStereo {
    /// Input format learned from negotiated caps (or pinned by [`with_input`](Self::with_input)).
    /// `None` until known.
    input: Option<AudioFormat>,
    /// Whether the stereo output format has been announced downstream (announce once).
    announced: bool,
    /// Input interchannel-frame stride in bytes (`channels * sample.bytes()`); `0` until known.
    in_stride: usize,
    /// Output interchannel-frame stride in bytes (`2 * sample.bytes()`); `0` until known.
    out_stride: usize,
    /// Carry for a partial input frame straddling two buffers, so channel lanes never desync
    /// (the `audioconvert`/`audiogain` idiom). Capacity is one frame — reserved at negotiation,
    /// so filling it never allocates.
    carry: Vec<u8>,
    /// The mid-stream format already complained about, so the warning is posted once per
    /// *distinct* change — see [`warn_if_relatched`](Self::warn_if_relatched).
    warned_relatch: Option<AudioFormat>,
}

impl AudioStereo {
    /// A stereo conformer discovering its input format at runtime from the upstream's
    /// negotiated `audio/raw` caps (spec: dynamic caps — consumer side).
    // COLD: constructs the element once; `carry` is a one-frame cross-buffer accumulator, not
    // per-buffer scratch.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            input: None,
            announced: false,
            in_stride: 0,
            out_stride: 0,
            carry: Vec::new(),
            warned_relatch: None,
        }
    }

    /// A conformer with the input format pinned at construction (statically-wired / testable
    /// form). Still announces its stereo output downstream.
    pub fn with_input(input: AudioFormat) -> Self {
        let mut s = Self::new();
        s.set_input(input);
        s
    }

    /// The input format, once known.
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.input
    }

    /// The format this element produces once the input is known: the input's rate and sample
    /// format, `channels` = [`OUT_CHANNELS`].
    pub fn output_format(&self) -> Option<AudioFormat> {
        self.input.map(|f| AudioFormat::new(f.sample_rate, OUT_CHANNELS, f.format))
    }

    /// Record the input format and derive both strides. Idempotent for the same format; a
    /// genuinely different one re-arms the downstream announcement.
    // COLD: `reserve` runs once per negotiation, sizing the one-frame carry so `process()`
    // never grows it.
    #[allow(clippy::disallowed_methods)]
    fn set_input(&mut self, input: AudioFormat) {
        if self.input == Some(input) {
            return;
        }
        self.in_stride = input.frame_stride();
        self.out_stride = OUT_CHANNELS as usize * input.format.bytes();
        self.input = Some(input);
        self.announced = false;
        self.carry.clear();
        self.carry.reserve(self.in_stride.max(1));
    }

    /// Infer the input format from the sink's negotiated `audio/raw` caps (spec: dynamic caps —
    /// consumer side). A pinned or already-learned format wins and short-circuits.
    fn learn_from_sink(&mut self, ctx: &Ctx) {
        if self.input.is_some() {
            return;
        }
        if let Some(fmt) = negotiated_audio_format(ctx, SINK) {
            self.set_input(fmt);
        }
    }

    /// Announce the stereo output (same rate, `channels = 2`, same sample) on the src pad, once
    /// the input is known (spec: dynamic caps — producer side).
    fn announce_output(&mut self, ctx: &mut Ctx) {
        if self.announced {
            return;
        }
        let Some(out) = self.output_format() else { return };
        ctx.announce_format(
            SRC,
            FAMILY,
            &[
                (FIELD_RATE, ValueDesc::Int(out.sample_rate as i64)),
                (FIELD_CHANNELS, ValueDesc::Int(out.channels as i64)),
                (FIELD_SAMPLE, ValueDesc::Id(out.format.caps_name())),
            ],
        );
        self.announced = true;
    }

    /// Complain on the bus when the sink's negotiated format has moved away from the latched
    /// one — the contract every `audio/raw` glue element holds.
    ///
    /// [`learn_from_sink`](Self::learn_from_sink) pins the format at the *first* negotiation, so
    /// a mid-stream change is **ignored**, and here the ignored field that matters most is
    /// `channels`: frames are read at the latched stride and folded with the matrix for the
    /// latched count, so a stream that changes channel count mid-way is conformed with the wrong
    /// matrix at the wrong stride. Following the change means re-announcing downstream and
    /// re-fixating the tail of the graph — the gapless design's job, not this element's. So the
    /// contract is: keep the old format, but say so, loudly.
    ///
    /// Posted once per *distinct* format and only from the event path, so it can never fire per
    /// buffer.
    // COLD: one bus message per distinct mid-stream format change, never per buffer.
    #[allow(clippy::disallowed_methods)]
    fn warn_if_relatched(&mut self, ctx: &mut Ctx) {
        let Some(latched) = self.input else { return };
        let Some(offered) = negotiated_audio_format(ctx, SINK) else { return };
        if offered == latched || self.warned_relatch == Some(offered) {
            return;
        }
        self.warned_relatch = Some(offered);
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element {
                element,
                message: format!(
                    "audiostereo: ignoring a mid-stream input format change ({latched} -> \
                     {offered}); the conform matrix and frame stride are fixed at the first \
                     negotiation and every later buffer is still conformed as the latched \
                     format. Rebuild the branch (or insert a fresh audiostereo) to follow the \
                     change."
                ),
            },
        });
    }

    /// Nanoseconds spanned by `frames` at the negotiated rate. `u128` intermediate so a
    /// multi-day stream cannot overflow the `frames * 1e9` product.
    fn frames_to_ts(&self, frames: u64) -> Timestamp {
        let Some(inp) = self.input else { return Timestamp::ZERO };
        if inp.sample_rate == 0 {
            return Timestamp::ZERO;
        }
        let ns = (frames as u128 * 1_000_000_000) / inp.sample_rate as u128;
        Timestamp::from_nanos(ns.min(Timestamp::MAX.0 as u128) as u64)
    }

    /// Conform one contiguous run of whole input frames into `dst` (exactly the matching run of
    /// output frames). `dst.len()` must be `frames * out_stride`.
    fn conform_bulk(&self, src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        let Some(inp) = self.input else {
            return Err(Error::Todo("audiostereo: input format unknown"));
        };
        match inp.channels {
            // Already stereo: a straight copy (the by-move fast path in `process` covers the
            // common case; this is the ragged-boundary remainder).
            2 => {
                dst.copy_from_slice(src);
                Ok(())
            }
            // Mono: duplicate the sole channel into both outputs.
            1 => remap_channels(inp.format, 1, OUT_CHANNELS as usize, src, dst)
                .map(|_| ())
                .ok_or(Error::Todo("audiostereo: mono->stereo upmix failed (ragged frame)")),
            // Multichannel: the ITU-R BS.775-3 Lo/Ro fold, the same call `audiodownmix` makes.
            n => downmix_to_stereo(inp.format, n as usize, src, dst)
                .map(|_| ())
                .ok_or(Error::Todo("audiostereo: multichannel fold failed (ragged frame)")),
        }
    }

    /// Conform `nframes` frames starting at frame `start` of the logical stream `head ++ body`
    /// into `dst`.
    ///
    /// `head` is the completed carried frame (0 or 1 whole frames) and `body` this buffer's
    /// whole frames, so a run never straddles the two in mid-frame — the split is at most one
    /// frame in, which is why this is two bulk calls and not a per-frame loop.
    fn conform_range(
        &self,
        head: &[u8],
        body: &[u8],
        start: usize,
        nframes: usize,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        let (i_s, o_s) = (self.in_stride, self.out_stride);
        let head_frames = head.len() / i_s.max(1);
        let mut written = 0usize;
        if start < head_frames {
            let take = (head_frames - start).min(nframes);
            self.conform_bulk(
                &head[start * i_s..(start + take) * i_s],
                &mut dst[..take * o_s],
            )?;
            written = take;
        }
        if written < nframes {
            let b_start = (start + written) - head_frames;
            let take = nframes - written;
            self.conform_bulk(
                &body[b_start * i_s..(b_start + take) * i_s],
                &mut dst[written * o_s..(written + take) * o_s],
            )?;
        }
        Ok(())
    }

    /// Conform `head ++ body` (whole interchannel frames) into pool slots and push them on the
    /// src pad, chunked to the slot size but always on a whole-frame boundary. The conform
    /// writes **straight into the slot**, so no staging buffer exists and `process()` performs
    /// no heap allocation (spec: performance — no steady-state heap traffic).
    ///
    /// `base` is the incoming buffer's `pts`; outputs carry it forward on the frame grid. The
    /// grid is shared by input and output because this element changes channels only — one
    /// input frame is always exactly one output frame.
    fn emit_conformed(
        &mut self,
        ctx: &mut Ctx,
        head: &[u8],
        body: &[u8],
        base: Timestamp,
    ) -> Result<(), Error> {
        let (i_s, o_s) = (self.in_stride, self.out_stride);
        if i_s == 0 || o_s == 0 {
            return Err(Error::Todo("audiostereo: frame stride unknown"));
        }
        let total = head.len() + body.len();
        debug_assert_eq!(total % i_s, 0, "emit_conformed takes whole frames");
        let total_frames = total / i_s;
        let (mut at, mut done) = (0usize, 0u64);
        while at < total_frames {
            let mut buf = ctx.alloc(SRC);
            let cap_frames = buf.memory.capacity() / o_s;
            if cap_frames == 0 {
                return Err(Error::Todo(
                    "audiostereo: pool slot smaller than one stereo interchannel frame",
                ));
            }
            let n = cap_frames.min(total_frames - at);
            let dst = &mut buf.memory.as_mut_full()[..n * o_s];
            self.conform_range(head, body, at, n, dst)?;
            buf.memory.set_len(n * o_s);
            // The carried frame belongs to the previous input buffer, so a ragged stream's pts
            // can be up to one frame early — below the resolution of any sink's scheduling, and
            // it keeps the grid gapless, which matters more (the `audiogain` rule).
            buf.pts = if base.is_some() {
                base.saturating_add(self.frames_to_ts(done))
            } else {
                Timestamp::NONE
            };
            buf.duration = self.frames_to_ts(n as u64);
            ctx.out(SRC).push(buf);
            at += n;
            done += n as u64;
        }
        Ok(())
    }
}

impl Element for AudioStereo {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // Link-time negotiation may already have fixed the sink format; infer it now so a
        // statically-negotiated graph needs no runtime event.
        self.learn_from_sink(ctx);
        self.carry.clear();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        self.announce_output(ctx);

        while let Some(buf) = inputs.pop() {
            if self.input.is_none() {
                return Err(Error::Todo(
                    "audiostereo: PCM arrived before an audio/raw input format was known \
                     (construct with AudioStereo::with_input, or feed an announcing upstream)",
                ));
            }
            let stride = self.in_stride;

            // Already-stereo fast path: the buffer is forwarded by *move*, no pool slot and no
            // memcpy (the `audiogain` unity idiom). Only when nothing is held back and the
            // buffer is whole frames, so ordering and frame alignment are preserved.
            if self.in_stride == self.out_stride
                && self.carry.is_empty()
                && buf.memory.data().len() % stride == 0
            {
                ctx.out(SRC).push(buf);
                continue;
            }

            // Whole interchannel frames only: a frame split across two buffers has to be
            // reassembled before either half is conformed, or the fold reads the wrong channel
            // into every matrix position and never resyncs. The carry holds at most one frame
            // and was reserved at negotiation, so this never allocates.
            let mut data = buf.memory.data();
            let mut frame = std::mem::take(&mut self.carry);
            let had_carry = !frame.is_empty();
            if had_carry {
                let take = (stride - frame.len()).min(data.len());
                frame.extend_from_slice(&data[..take]);
                data = &data[take..];
            }
            let complete = frame.len() == stride;
            let whole = data.len() - (data.len() % stride);
            let head: &[u8] = if complete { &frame } else { &[] };
            let r = self.emit_conformed(ctx, head, &data[..whole], buf.pts);
            // Put the carry back — `frame` first, so its one-frame capacity is restored either
            // way; then, unless it is still an incomplete frame in its own right, overwrite it
            // with this buffer's ragged tail.
            self.carry = frame;
            if complete || !had_carry {
                self.carry.clear();
                self.carry.extend_from_slice(&data[whole..]);
            }
            r?;
            // `buf` recycles here on drop.
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Seek (spec: flush/seek). `carry` holds the tail of a *pre-seek* interchannel
            // frame; prepending it to the first post-seek buffer shifts every later sample by
            // part of a frame, so the conform reads the wrong channel into every output lane
            // and never resyncs. The conform itself is memoryless (one output frame per input
            // frame, no history), so there is nothing else a seek invalidates — and the learned
            // format deliberately survives: a seek moves the read head, it does not change what
            // the upstream is sending.
            Event::FlushStart => self.carry.clear(),
            Event::FormatChange(_) => {
                self.learn_from_sink(ctx);
                self.warn_if_relatched(ctx);
            }
            _ => {}
        }
        if self.input.is_none() {
            self.learn_from_sink(ctx);
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // A leftover carry is an incomplete final interchannel frame (ragged input) — drop it
        // so the output stays frame-aligned.
        self.carry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profluens_audio::SampleFormat;

    #[test]
    fn output_is_always_stereo_whatever_the_input() {
        for (ch, rate, fmt) in [
            (1u16, 44_100u32, SampleFormat::S16),
            (2, 48_000, SampleFormat::F32),
            (6, 48_000, SampleFormat::S16),
            (8, 96_000, SampleFormat::S32),
        ] {
            let el = AudioStereo::with_input(AudioFormat::new(rate, ch, fmt));
            let out = el.output_format().expect("output format once the input is pinned");
            assert_eq!(out.channels, 2, "{ch} ch input must conform to stereo");
            assert_eq!(out.sample_rate, rate, "rate is preserved");
            assert_eq!(out.format, fmt, "sample format is preserved");
        }
    }

    #[test]
    fn mono_frames_are_duplicated_into_both_lanes() {
        // The load-bearing claim of this element: 1 ch in, 2 ch out, sample-for-sample.
        let el = AudioStereo::with_input(AudioFormat::new(48_000, 1, SampleFormat::S16));
        let src: Vec<u8> = [100i16, -200, 300, -400].iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut dst = vec![0u8; src.len() * 2];
        el.conform_bulk(&src, &mut dst).expect("mono upmix");
        let got: Vec<i16> = dst.as_chunks::<2>().0.iter().copied().map(i16::from_le_bytes).collect();
        assert_eq!(got, vec![100, 100, -200, -200, 300, 300, -400, -400]);
    }

    #[test]
    fn stereo_is_conformed_byte_for_byte() {
        let el = AudioStereo::with_input(AudioFormat::new(48_000, 2, SampleFormat::S16));
        let src: Vec<u8> = (0u8..16).collect();
        let mut dst = vec![0u8; src.len()];
        el.conform_bulk(&src, &mut dst).expect("stereo passthrough");
        assert_eq!(dst, src, "stereo input must be untouched");
    }

    #[test]
    fn multichannel_folds_exactly_like_audiodownmix() {
        // Same call, same coefficients — assert against the library directly so a divergence
        // between this element and `audiodownmix` can never go unnoticed.
        let el = AudioStereo::with_input(AudioFormat::new(48_000, 6, SampleFormat::S16));
        let src: Vec<u8> = (0i16..60).flat_map(|s| (s * 100).to_le_bytes()).collect();
        let mut mine = vec![0u8; src.len() / 3];
        let mut theirs = vec![0u8; src.len() / 3];
        el.conform_bulk(&src, &mut mine).expect("5.1 fold");
        downmix_to_stereo(SampleFormat::S16, 6, &src, &mut theirs).expect("library fold");
        assert_eq!(mine, theirs, "the fold must be bit-identical to audiodownmix's");
    }

    #[test]
    fn a_carried_partial_frame_is_conformed_with_the_next_buffer() {
        // `conform_range` reads a run that begins in the carried frame and continues into the
        // body — the ragged-boundary case that would otherwise rotate the channel lanes.
        let el = AudioStereo::with_input(AudioFormat::new(48_000, 1, SampleFormat::S16));
        let head: Vec<u8> = 7i16.to_le_bytes().into_iter().collect(); // one whole mono frame
        let body: Vec<u8> = [8i16, 9].iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut dst = vec![0u8; 3 * 4]; // 3 frames × 2 ch × 2 bytes
        el.conform_range(&head, &body, 0, 3, &mut dst).expect("carry + body");
        let got: Vec<i16> = dst.as_chunks::<2>().0.iter().copied().map(i16::from_le_bytes).collect();
        assert_eq!(got, vec![7, 7, 8, 8, 9, 9], "the carried frame leads, in order");
    }
}
