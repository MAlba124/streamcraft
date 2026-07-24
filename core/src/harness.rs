//! `Harness` — the threadless, single-element test rig (spec: Testing, benchmarks,
//! stress, fuzzing — "Element harness").
//!
//! > *wraps one element with no threads and a mock clock; push buffers/events, assert
//! > outputs. The passive/inline scheduling model makes the harness nearly free — it
//! > IS the inline caller.*
//!
//! A [`Harness`] owns one [`Element`] and the single [`Ctx`] it runs against,
//! configured exactly the way [`run_group`](crate::pipeline) configures a group
//! member: the per-pad output table sized from the element's descriptor
//! ([`configure_pads`](Ctx::configure_pads)), a bounded [`Pool`], a [`MockClock`]
//! installed ([`set_clock`](Ctx::set_clock)), and a [`Vocabulary`] built from the
//! element's *own* offers so `ctx.field_id("rate")` / `ctx.value_name(id)` resolve and
//! runtime announcements can be lowered — all without a pipeline, a thread, or a ring.
//!
//! Calling `process()` inline is the whole point: the pipeline's passive-element path
//! already runs transforms as plain function calls (spec: Scheduling — passive chains
//! run inline), so the harness reproduces the element's real runtime contract with none
//! of the scaffolding every hand-rolled test file re-derives (the `PacketSrc` /
//! `RecordSink` duplication in `flac/tests/transcode.rs`, `vp8/tests/vp8dec.rs`,
//! `elements/tests/dynamic_caps.rs`).
//!
//! ## What it reproduces from the scheduler
//!
//! - **Link-time formats.** [`fix_format`](Harness::fix_format) interns a family/fields
//!   through the harness vocabulary and installs the resulting [`FixedFormat`] on a pad
//!   — the analogue of the pipeline's link-time solve writing a fixed edge, read back by
//!   the element via `ctx.negotiated(pad)`.
//! - **Buffer + event delivery.** [`push`](Harness::push) appends a buffer to the
//!   element's input and runs `process()`; [`push_event`](Harness::push_event) delivers
//!   to `event()`. `crank()` runs `process()` with no new input (sources, carry drains).
//! - **Runtime announce → FormatChange.** After every `process()` the harness drains a
//!   pending [`announce_format`](Ctx::announce_format) and resolves it through the
//!   vocabulary with [`Vocabulary::build_fixed`], exactly as `run_group` does before
//!   riding a `FormatChange` downstream. [`announced`](Harness::announced) returns it.
//! - **EOS + carry.** [`eos`](Harness::eos) delivers [`Event::Eos`] and returns the
//!   remaining outputs — the tail-flush path a decoder takes at end of stream.
//! - **The clock.** [`clock`](Harness::clock) hands back the [`MockClock`]; advancing it
//!   drives `ctx.now()` / `ctx.wait_until()` with no wall-clock sleeping.
//!
//! ## What it deliberately does NOT simulate
//!
//! It is the *inline caller*, not a scheduler. There are **no rings** (no inter-group
//! SPSC queues, no blocking backpressure — `push` never blocks; the pool cap is the only
//! backpressure and it surfaces as the element's own `try_alloc` returning `None`), **no
//! thread groups** (one element, one thread — the calling test's), and **no seek
//! generations** (no out-of-band `SeekState`; deliver `Event::FlushStart` /
//! `Event::FlushStop` directly to test flush handling). Multi-element chains, ring
//! semantics, group termination, and seek-gen stamping belong to golden *pipeline* tests
//! (spec: Golden pipeline tests), not here. Re-validating an announcement against a
//! downstream pad's offers is likewise a two-element property — the harness surfaces the
//! raw announced format and leaves that assertion to the caller.
//!
//! ## Duplication note (for later dedup)
//!
//! The announce-drain in [`run_process`](Harness::run_process) mirrors `run_group`'s
//! post-`process()` hook (pipeline.rs, "Dynamic caps: if the element announced …"), and
//! the pad-shape derivation in [`new`](Harness::new) mirrors `run_group`'s `pad_infos`
//! walk. Kept replicated here per the harness's charter (touch no scheduler internals);
//! a shared helper is a fair follow-up once the transport is the ring-backed one.

use std::sync::Arc;

use crate::batch::{Batch, Inputs};
use crate::buffer::{Buffer, BufferFlags};
use crate::bus::{Bus, BusMessage, BusSender};
use crate::clock::MockClock;
use crate::ctx::Ctx;
use crate::element::{Direction, Element, Flow};
use crate::error::Error;
use crate::event::Event;
use crate::format::{FixedFormat, Vocabulary};
use crate::id::{Interner, PadId};
use crate::memory::Pool;
use crate::time::Timestamp;

/// The byte/passthrough format id every un-negotiated pad carries, matching the
/// pipeline's `BYTES` sentinel (pipeline.rs `const BYTES`). A harness buffer allocated
/// from the pool is stamped with it until a real format is fixed.
const BYTES: crate::id::FormatId = crate::id::FormatId(0);

/// The default pool slot size, in bytes — big enough for the small payloads a unit test
/// pushes, overridable via [`Harness::with_slot_size`].
const DEFAULT_SLOT: usize = 64 * 1024;

/// The default pool cap: generous, since a threadless unit test rarely needs
/// backpressure. Override with [`Harness::with_pool`] to exercise the `try_alloc`-returns-
/// `None` path an element takes under a full pool.
const DEFAULT_SLOTS: u32 = 1024;

/// A threadless, mock-clocked wrapper around one [`Element`] (spec: Element harness).
///
/// Construct with [`Harness::new`], fix formats with [`fix_format`](Harness::fix_format),
/// then drive the element with [`push`](Harness::push) / [`push_event`](Harness::push_event)
/// / [`crank`](Harness::crank) and read results with [`pull`](Harness::pull) /
/// [`announced`](Harness::announced) / [`bus_messages`](Harness::bus_messages).
pub struct Harness {
    element: Box<dyn Element>,
    ctx: Ctx,
    clock: MockClock,
    /// The interned tables the harness built from the element's offers, kept so
    /// [`fix_format`](Harness::fix_format) can intern new names on demand and the
    /// announce-drain can [`build_fixed`](Vocabulary::build_fixed) — the same
    /// `Arc<Vocabulary>` installed on the `Ctx`.
    vocabulary: Arc<Vocabulary>,
    /// The application side of the bus, drained by [`bus_messages`](Harness::bus_messages).
    bus: Bus,
    /// The pool the element allocates from, for [`alloc`](Harness::alloc) prefills.
    pool: Pool,
    /// The most recent runtime announcement, resolved through the vocabulary — set after
    /// any `process()` that called [`Ctx::announce_format`], as `run_group` would.
    announced: Option<FixedFormat>,
    /// Whether [`start`](Harness::start) has run — driving methods call it lazily on
    /// first use, matching the pipeline starting an element before streaming.
    started: bool,
    /// Sink pad index for delivering in-band events (first sink pad), or `None` for a
    /// pure source — the analogue of `run_group`'s `member_sink_pads`.
    sink_pad: Option<PadId>,
}

impl Harness {
    /// Wrap `element` in a fresh harness: derive its pad shape from `desc()`, size a
    /// default [`Pool`], install a [`MockClock`] and a [`Vocabulary`] built by interning
    /// every family/field/categorical value the element's pads offer (so the element can
    /// resolve its own caps by name), exactly as the pipeline does at run setup.
    pub fn new(element: impl Element + 'static) -> Self {
        Self::with_pool(element, DEFAULT_SLOT, DEFAULT_SLOTS)
    }

    /// Like [`new`](Self::new) but with a caller-chosen pool `slot_size` — for an element
    /// whose per-buffer output is larger than [`DEFAULT_SLOT`].
    pub fn with_slot_size(element: impl Element + 'static, slot_size: usize) -> Self {
        Self::with_pool(element, slot_size, DEFAULT_SLOTS)
    }

    /// The full constructor: `slot_size` bytes per pool slot, capped at `max_slots`
    /// outstanding. A small `max_slots` exercises the backpressure path — the element's
    /// `try_alloc` returns `None`, its own signal to yield (spec: submission credits).
    pub fn with_pool(element: impl Element + 'static, slot_size: usize, max_slots: u32) -> Self {
        let element: Box<dyn Element> = Box::new(element);
        let desc = element.desc();

        // Pad shape: total pad count and the src-pad indices — the same derivation
        // `run_group` runs from `pad_infos` (pipeline.rs). Static pads only: a harness
        // drives one element in isolation, before any dynamic-pad preroll.
        let npads = desc.pads.len();
        let src_pads: Vec<usize> = desc
            .pads
            .iter()
            .enumerate()
            .filter(|(_, p)| p.direction == Direction::Src)
            .map(|(i, _)| i)
            .collect();
        let sink_pad = desc
            .pads
            .iter()
            .position(|p| p.direction == Direction::Sink)
            .map(|i| PadId(i as u32));

        // Build the vocabulary by interning the element's own offers. The pipeline interns
        // across the whole graph at link time; here the one element is the whole graph, so
        // interning its pads' families/fields/values gives it a vocabulary rich enough to
        // resolve everything it can name (spec: Formats — interning happens once).
        let (mut formats, mut fields, mut values) =
            (Interner::new(), Interner::new(), Interner::new());
        for pad in desc.pads {
            for offer in pad.offers {
                formats.intern(offer.family);
                for fd in offer.fields {
                    fields.intern(fd.field);
                    intern_constraint_values(&fd.allowed, &mut values);
                    if let Some(vd) = fd.preferred {
                        intern_value_desc(&vd, &mut values);
                    }
                }
            }
        }
        let vocabulary = Arc::new(Vocabulary { formats, fields, values });

        let pool = Pool::bounded(slot_size, max_slots);
        let (bus_tx, bus) = Bus::channel();
        let mut ctx = new_ctx(&pool, &bus_tx);
        ctx.configure_pads(npads, &src_pads);
        ctx.set_vocabulary(Arc::clone(&vocabulary));
        let clock = MockClock::new();
        ctx.set_clock(Arc::new(clock.clone()), Timestamp::ZERO);
        // Install the property mailbox where declared, so `ctx.prop(name)` resolves —
        // parity with `run_group` wiring props on elements that declare them.
        if !desc.props.is_empty() {
            ctx.set_props(desc.props, Arc::new(crate::props::PropTable::new(desc.props.len())));
        }

        Self {
            element,
            ctx,
            clock,
            vocabulary,
            bus,
            pool,
            announced: None,
            started: false,
            sink_pad,
        }
    }

    /// Install a negotiated [`FixedFormat`] on the named pad — the link-time analogue:
    /// interns `family` and each field name/categorical value against the harness
    /// vocabulary, builds the fixed format, and writes it where the element reads it
    /// (`ctx.negotiated(pad)`). Panics if the pad name is unknown (a test bug).
    ///
    /// ```ignore
    /// h.fix_format("sink", "audio/raw", &[("rate", ValueDesc::Int(48_000))]);
    /// ```
    pub fn fix_format(
        &mut self,
        pad_name: &str,
        family: &str,
        fields: &[(&'static str, crate::format::ValueDesc)],
    ) {
        let pad = self.pad_id(pad_name);
        // Intern any names not already in the vocabulary — a caller may fix a value the
        // pad's static offer never named (a broad `Any` field pinned to a concrete value).
        // Rebuild the shared vocabulary so both `Ctx` and the harness see the new names.
        let vocab = make_mut_vocabulary(&mut self.vocabulary);
        vocab.formats.intern(family);
        for (name, vd) in fields {
            vocab.fields.intern(name);
            intern_value_desc(vd, &mut vocab.values);
        }
        self.ctx.set_vocabulary(Arc::clone(&self.vocabulary));

        let fixed = self
            .vocabulary
            .build_fixed(family, fields)
            .expect("fix_format: family/field/value must be internable");
        self.ctx.set_negotiated_one(pad, fixed);
    }

    /// Run the element's `start()` if it has not run yet. Called lazily by the driving
    /// methods, or explicitly to assert `start()` succeeds before any input.
    pub fn start(&mut self) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }
        self.element.start(&mut self.ctx)?;
        self.started = true;
        Ok(())
    }

    /// Deliver one input buffer to the named sink pad and run `process()` inline,
    /// returning its [`Flow`]. Starts the element first if needed. Single-input elements
    /// (the majority) ignore the pad name — the buffer lands on the primary input, as it
    /// would from an upstream ring.
    pub fn push(&mut self, _pad_name: &str, buf: Buffer) -> Result<Flow, Error> {
        self.start()?;
        let mut batch = Batch::new(BYTES);
        batch.push(buf);
        self.ctx.input_append(&mut batch);
        self.run_process()
    }

    /// Deliver a whole [`Batch`] of input to the primary sink pad and run `process()`
    /// once, returning its [`Flow`].
    pub fn push_batch(&mut self, mut batch: Batch) -> Result<Flow, Error> {
        self.start()?;
        self.ctx.input_append(&mut batch);
        self.run_process()
    }

    /// Deliver an event to the element's `event()` — a `FormatChange`, `Segment`, a
    /// `FlushStart`/`FlushStop` pair to test flush handling, etc. A `FormatChange`
    /// additionally installs its format on the sink pad, as the scheduler does before
    /// delivering it (spec: Formats — dynamic caps).
    pub fn push_event(&mut self, event: Event) -> Result<(), Error> {
        self.start()?;
        if let (Event::FormatChange(f), Some(pad)) = (&event, self.sink_pad) {
            self.ctx.set_negotiated_one(pad, f.clone());
        }
        self.element.event(&mut self.ctx, &event)
    }

    /// Run `process()` with no new input — for a source producing on each call, or a
    /// transform draining a backpressure carry. Returns the [`Flow`].
    pub fn crank(&mut self) -> Result<Flow, Error> {
        self.start()?;
        self.run_process()
    }

    /// Take the oldest emitted buffer from the named src pad (FIFO), or `None` when the
    /// element has produced nothing more. Single-src elements ignore the pad name.
    pub fn pull(&mut self, pad_name: &str) -> Option<Buffer> {
        let pad = self.pad_id(pad_name);
        // Route by pad only for a genuine branching element (>1 src pad); a single-src
        // element writes everything to its one src pad regardless of the argument, so read
        // there — mirroring `Ctx::out`'s single-src leniency.
        let pad = if self.src_pad_count() > 1 { pad } else { self.primary_src_pad() };
        // `take_output` swaps in a fresh empty batch; pop one and hand the tail back so a
        // later `pull` sees the rest.
        let mut batch = self.ctx.take_output(pad);
        let head = batch.pop_front();
        if !batch.is_empty() {
            self.ctx.output_on(pad).append(&mut batch);
        }
        head
    }

    /// The element's most recent runtime format announcement, resolved through the
    /// vocabulary exactly as the scheduler resolves it (`take_announcement` +
    /// [`Vocabulary::build_fixed`]). `None` until the element announces. Set after each
    /// driving call; a fresh announcement overwrites the last.
    pub fn announced(&self) -> Option<FixedFormat> {
        self.announced.clone()
    }

    /// The mock clock — advance it to drive `ctx.now()` and release `ctx.wait_until()`
    /// waits with no real sleeping (spec: Testing — nothing ever sleeps).
    pub fn clock(&self) -> &MockClock {
        &self.clock
    }

    /// The harness vocabulary — the interned tables built from the element's offers (plus
    /// any [`fix_format`](Self::fix_format) names). Resolve an [`announced`](Self::announced)
    /// format's fields by name: `h.vocabulary().field_id("rate")`,
    /// `h.vocabulary().value_name(id)` — the same resolution the element itself does through
    /// its `Ctx`.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// A pool buffer of `bytes.len()` usable bytes, prefilled from `bytes` — the helper
    /// every hand-rolled test re-writes to turn a byte slice into a pushable [`Buffer`].
    pub fn alloc(&mut self, bytes: &[u8]) -> Buffer {
        let mut memory = self.pool.acquire_exact(bytes.len());
        memory.as_mut_full()[..bytes.len()].copy_from_slice(bytes);
        memory.set_len(bytes.len());
        Buffer {
            memory,
            pts: Timestamp::NONE,
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags: BufferFlags::empty(),
            format: BYTES,
            sync: None,
        }
    }

    /// Deliver [`Event::Eos`] then return every buffer the element flushes, in FIFO
    /// order — the end-of-stream tail-drain (spec: Events — EOS reaches elements in
    /// order; a decoder flushes its remaining frames here).
    pub fn eos(&mut self) -> Result<Vec<Buffer>, Error> {
        self.push_event(Event::Eos)?;
        // An element may announce (a decoder that only saw its header at EOS) during the
        // flush; drain it too.
        self.drain_announcement();
        Ok(self.drain_outputs())
    }

    /// Drain every message the element posted to the bus since the last call.
    pub fn bus_messages(&mut self) -> Vec<BusMessage> {
        let mut out = Vec::new();
        while let Some(m) = self.bus.try_recv() {
            out.push(m);
        }
        out
    }

    /// Read a declared property's current value by name, for asserting a `start()`-time
    /// or event-driven prop read (`None` if never set / not declared).
    pub fn prop(&self, name: &str) -> Option<crate::format::Value> {
        self.ctx.prop(name)
    }

    /// Every buffer currently staged on the primary src pad, FIFO, leaving it empty.
    pub fn drain_outputs(&mut self) -> Vec<Buffer> {
        let pad = self.primary_src_pad();
        let mut batch = self.ctx.take_output(pad);
        let mut out = Vec::new();
        while let Some(b) = batch.pop_front() {
            out.push(b);
        }
        out
    }

    // --- internals ---

    /// Run `process()` inline and reproduce the scheduler's post-`process()` hooks the
    /// harness cares about: reset the scratch arena and drain any announcement (spec:
    /// dynamic caps). Mirrors the middle of `run_group`'s element loop.
    fn run_process(&mut self) -> Result<Flow, Error> {
        let mut input = self.ctx.take_input();
        let flow = self.element.process(&mut self.ctx, Inputs::owned(&mut input));
        self.ctx.set_input(input);
        self.ctx.reset_scratch();
        self.drain_announcement();
        flow
    }

    /// Drain a pending runtime announcement, resolving it to a [`FixedFormat`] through the
    /// vocabulary and caching it for [`announced`](Self::announced). The scheduler would
    /// additionally ride it downstream as a `FormatChange`; with no downstream here, the
    /// harness just records the resolved format.
    fn drain_announcement(&mut self) {
        if let Some(ann) = self.ctx.take_announcement() {
            let fixed = match ann.payload {
                crate::ctx::AnnouncePayload::Named { family, fields } => {
                    self.vocabulary.build_fixed(family, &fields)
                }
                crate::ctx::AnnouncePayload::Fixed(f) => Some(f),
            };
            if let Some(f) = fixed {
                self.announced = Some(f);
            }
        }
    }

    /// The `PadId` for a pad name (panics on an unknown name — a test-authoring bug, so a
    /// loud panic beats a silent misroute).
    fn pad_id(&self, name: &str) -> PadId {
        self.element
            .desc()
            .pads
            .iter()
            .position(|p| p.name == name)
            .map(|i| PadId(i as u32))
            .unwrap_or_else(|| {
                panic!("harness: no pad named '{name}' on '{}'", self.element.desc().name)
            })
    }

    fn src_pad_count(&self) -> usize {
        self.element
            .desc()
            .pads
            .iter()
            .filter(|p| p.direction == Direction::Src)
            .count()
    }

    /// The element's first src pad — where a single-src element writes everything.
    fn primary_src_pad(&self) -> PadId {
        self.element
            .desc()
            .pads
            .iter()
            .position(|p| p.direction == Direction::Src)
            .map(|i| PadId(i as u32))
            .unwrap_or(PadId(0))
    }
}

/// Construct a `Ctx` the way `run_group` does (pool clone, bus clone, the sentinel
/// element id, the `BYTES` out-format, one credit unit). There is only ever one element
/// in a harness, so a fixed element id suffices.
fn new_ctx(pool: &Pool, bus: &BusSender) -> Ctx {
    Ctx::new(pool.clone(), bus.clone(), crate::id::ElementId(0), BYTES, 1)
}

/// Intern every categorical name a [`ConstraintDesc`](crate::format::ConstraintDesc)
/// mentions, so the vocabulary can resolve an announcement / fixed value naming it.
fn intern_constraint_values(c: &crate::format::ConstraintDesc, values: &mut Interner) {
    use crate::format::ConstraintDesc;
    match c {
        ConstraintDesc::Any => {}
        ConstraintDesc::Eq(v) => intern_value_desc(v, values),
        ConstraintDesc::Range { min, max, step } => {
            intern_value_desc(min, values);
            intern_value_desc(max, values);
            intern_value_desc(step, values);
        }
        ConstraintDesc::Set(vs) => {
            for v in *vs {
                intern_value_desc(v, values);
            }
        }
    }
}

fn intern_value_desc(v: &crate::format::ValueDesc, values: &mut Interner) {
    if let crate::format::ValueDesc::Id(s) = v {
        values.intern(s);
    }
}

/// Give the harness a `&mut Vocabulary` even though the `Ctx` holds a second `Arc` clone:
/// the `Ctx`'s clone means `Arc::get_mut` would fail, so clone the (small) interner tables
/// into a fresh, uniquely-owned `Arc`, install that, and hand out `&mut`. Cheap and off
/// any hot path — only the rare `fix_format` naming something new hits this.
fn make_mut_vocabulary(this: &mut Arc<Vocabulary>) -> &mut Vocabulary {
    let cloned = Vocabulary {
        formats: this.formats.clone(),
        fields: this.fields.clone(),
        values: this.values.clone(),
    };
    *this = Arc::new(cloned);
    Arc::get_mut(this).expect("freshly-created Arc is unique")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementDesc, InputPolicy, LatencyDesc, PadDesc, SchedHint};
    use crate::format::{ConstraintDesc, FieldDesc, OfferDesc, Value, ValueDesc};

    // A tiny in-file element exercising the whole harness surface: it echoes each input
    // byte doubled, announces an `audio/raw` format on its first `process()`, stamps
    // output PTS from the clock, and flushes a fixed tail marker at EOS.
    static SAMPLE_VALUES: [ValueDesc; 2] = [ValueDesc::Id("s16"), ValueDesc::Id("s24")];
    static SRC_FIELDS: [FieldDesc; 3] = [
        FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None },
        FieldDesc { field: "channels", allowed: ConstraintDesc::Any, preferred: None },
        FieldDesc {
            field: "sample",
            allowed: ConstraintDesc::Set(&SAMPLE_VALUES),
            preferred: None,
        },
    ];
    static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc { family: "audio/raw", fields: &SRC_FIELDS }];
    static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
    static PADS: [PadDesc; 2] = [
        PadDesc {
            name: "sink",
            direction: Direction::Sink,
            offers: &SINK_OFFERS,
            dynamic: false,
            validate: None,
        },
        PadDesc {
            name: "src",
            direction: Direction::Src,
            offers: &SRC_OFFERS,
            dynamic: true,
            validate: None,
        },
    ];
    static DESC: ElementDesc = ElementDesc {
        name: "doubler",
        pads: &PADS,
        props: &[],
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

    #[derive(Default)]
    struct Doubler {
        announced: bool,
    }

    impl Element for Doubler {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
            if !self.announced {
                ctx.announce_format(
                    PadId(1),
                    "audio/raw",
                    &[
                        ("rate", ValueDesc::Int(48_000)),
                        ("channels", ValueDesc::Int(2)),
                        ("sample", ValueDesc::Id("s16")),
                    ],
                );
                self.announced = true;
            }
            while let Some(inb) = inputs.pop() {
                let data = inb.memory.data();
                let n = data.len();
                let mut out = ctx.alloc(PadId(1));
                let dst = out.memory.as_mut_full();
                for (i, &b) in data.iter().enumerate() {
                    dst[i] = b.wrapping_mul(2);
                }
                out.memory.set_len(n);
                // Stamp the PTS with the current running time, so the clock test can assert it.
                out.pts = ctx.now();
                ctx.out(PadId(1)).push(out);
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
            if matches!(event, Event::Eos) {
                // Flush a one-byte tail marker (0xEE) at EOS — the carry-drain analogue.
                let mut out = ctx.alloc(PadId(1));
                out.memory.as_mut_full()[0] = 0xEE;
                out.memory.set_len(1);
                ctx.out(PadId(1)).push(out);
            }
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }

    #[test]
    fn push_pull_doubles_bytes() {
        let mut h = Harness::new(Doubler::default());
        let buf = h.alloc(&[1, 2, 3]);
        let flow = h.push("sink", buf).unwrap();
        assert_eq!(flow, Flow::Ok);
        let out = h.pull("src").expect("one output buffer");
        assert_eq!(out.memory.data(), &[2, 4, 6], "each byte doubled");
        assert!(h.pull("src").is_none(), "exactly one buffer");
    }

    #[test]
    fn announce_resolves_through_the_vocabulary() {
        let mut h = Harness::new(Doubler::default());
        let buf = h.alloc(&[9]);
        let _ = h.push("sink", buf).unwrap();
        let f = h.announced().expect("the element announced audio/raw");
        // Resolve the announced fields by name through the harness vocabulary, proving
        // ctx.field_id / value_name are wired.
        let rate = h.vocabulary.field_id("rate").unwrap();
        let sample = h.vocabulary.field_id("sample").unwrap();
        let s16 = h.vocabulary.value_id("s16").unwrap();
        assert_eq!(f.get(rate), Some(Value::Int(48_000)));
        assert_eq!(f.get(sample), Some(Value::Id(s16)));
        assert_eq!(h.vocabulary.family_name(f.family), Some("audio/raw"));
    }

    #[test]
    fn fix_format_installs_a_negotiated_edge() {
        // The element reads its format from ctx.negotiated — fix one on the src pad (a
        // broad `Any` rate pinned to a concrete value) and prove it reads back.
        let mut h = Harness::new(Doubler::default());
        h.fix_format("src", "audio/raw", &[("rate", ValueDesc::Int(44_100))]);
        let rate = h.vocabulary.field_id("rate").unwrap();
        let neg = h.ctx.negotiated(PadId(1)).expect("format fixed on src");
        assert_eq!(neg.get(rate), Some(Value::Int(44_100)));
    }

    #[test]
    fn clock_advance_drives_running_time_on_pts() {
        let mut h = Harness::new(Doubler::default());
        h.clock().advance(Timestamp::from_millis(10));
        let buf = h.alloc(&[1]);
        let _ = h.push("sink", buf).unwrap();
        let out = h.pull("src").unwrap();
        assert_eq!(out.pts, Timestamp::from_millis(10), "pts stamped with clock now()");
    }

    #[test]
    fn eos_flushes_the_tail_marker() {
        let mut h = Harness::new(Doubler::default());
        let buf = h.alloc(&[1]);
        let _ = h.push("sink", buf).unwrap();
        // Drain the doubled buffer first so eos() returns only the tail.
        assert_eq!(h.pull("src").unwrap().memory.data(), &[2]);
        let tail = h.eos().unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].memory.data(), &[0xEE], "EOS tail marker flushed");
    }

    #[test]
    fn crank_runs_process_without_input() {
        // A crank with no input still runs process (which announces), producing no buffer.
        let mut h = Harness::new(Doubler::default());
        let flow = h.crank().unwrap();
        assert_eq!(flow, Flow::Ok);
        assert!(h.pull("src").is_none(), "no input ⇒ no output");
        assert!(h.announced().is_some(), "but the announce still fired");
    }

    #[test]
    fn unknown_pad_name_panics() {
        let mut h = Harness::new(Doubler::default());
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.pull("nope")));
        assert!(r.is_err(), "an unknown pad name is a loud test bug");
    }
}
