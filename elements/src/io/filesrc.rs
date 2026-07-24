//! `filesrc` — reads a file into pooled buffers via the reactor
//! (spec: Milestone applications §1; IO). Reactor-native: it registers its file,
//! keeps up to `credits` positioned reads in flight, and emits completed buffers.
//! Positioned reads (explicit offset) make it seek-ready.

use std::fs::File;
use std::path::{Path, PathBuf};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{Constraint, OfferDesc};
use streamcraft_core::id::PadId;
use streamcraft_core::io::{FileHandle, IoResult};
use streamcraft_core::log;
use streamcraft_core::log::Level;
use streamcraft_core::time::Timestamp;

/// A file is an untyped byte stream — offer the open `bytes` family, no fields.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

/// The file to read (spec: Plugins — `parse("filesrc path=…")`). A path is a free-form
/// string: `Constraint::Any` (paths ride `Value::Id`; see [`Pipeline::set_str`]).
/// Structural (`live: false`): opening the file is a `start()`-time act.
///
/// [`Pipeline::set_str`]: streamcraft_core::pipeline::Pipeline::set_str
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "path",
    allowed: Constraint::Any,
    live: false,
}];

static DESC: ElementDesc = ElementDesc {
    name: "filesrc",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(FileSrc::new(""))),
};

pub struct FileSrc {
    path: PathBuf,
    file: FileHandle,
    started: bool,
    offset: u64,
    eof: bool,
    in_flight: u32,
    seq: u64,
    /// Reads submitted before a seek carry a `user` (their `seq`) below this floor; their
    /// completions land at the *old* offset, so they are discarded (spec: flush/seek).
    valid_from: u64,
}

impl FileSrc {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            file: FileHandle(0),
            started: false,
            offset: 0,
            eof: false,
            in_flight: 0,
            seq: 0,
            valid_from: 0,
        }
    }
}

impl Element for FileSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A parsed `path=` overrides the constructor value (spec: Plugins — the string
        // rides `Value::Id`, resolved by name off the value vocabulary). Falls back to
        // the constructor path when unset (the typed `FileSrc::new(path)` path).
        if let Some(streamcraft_core::format::Value::Id(id)) = ctx.prop("path") {
            if let Some(s) = ctx.value_name(id) {
                self.path = PathBuf::from(s);
            }
        }
        let f = File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path.display())))?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        self.file = ctx.io().register(f);
        self.started = true;
        log!(&*ctx, Level::Debug, "open", len = len);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            return Err(Error::Todo("filesrc not started"));
        }

        // Drain completed reads: emit their buffers (or note EOF).
        loop {
            let completion = ctx.io().next_completion();
            let c = match completion {
                Some(c) => c,
                None => break,
            };
            self.in_flight -= 1;
            if c.user < self.valid_from {
                continue; // read submitted before a seek: wrong offset, discard (buf recycles)
            }
            match c.result {
                IoResult::Ok(0) => self.eof = true, // c.buf recycles on drop
                IoResult::Ok(n) => {
                    log!(&*ctx, Level::Debug, "read", bytes = n);
                    ctx.out(PadId(0)).push(c.buf);
                }
                IoResult::Cancelled => {}
                IoResult::Err(k) => return Err(Error::Resource(format!("read: {k:?}"))),
            }
        }

        // Top up positioned reads up to credits and pool availability.
        let credits = ctx.io().credits();
        while !self.eof && self.in_flight < credits {
            let buf = match ctx.try_alloc(PadId(0)) {
                Some(b) => b,
                None => break, // pool full → backpressure
            };
            let cap = buf.memory.capacity() as u64;
            let user = self.seq;
            self.seq += 1;
            ctx.io().submit_read(self.file, self.offset, buf, user);
            self.offset += cap;
            self.in_flight += 1;
        }

        if self.eof && self.in_flight == 0 {
            log!(&*ctx, Level::Info, "eos", offset = self.offset);
            Ok(Flow::Eos)
        } else {
            Ok(Flow::Ok)
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::FlushStart) {
            // Seek (spec: flush/seek): resume reading at the requested byte offset. Reads
            // already in flight will complete at the old offset — mark them stale via the
            // `seq` floor so their completions are discarded. `submit_read` takes the
            // offset explicitly, so the fd's cursor is irrelevant; nothing else to reset.
            if let Some(t) = ctx.seek_target() {
                self.offset = t.to_byte;
                self.eof = false;
                self.valid_from = self.seq;
            }
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns and closes the file
    }
}
