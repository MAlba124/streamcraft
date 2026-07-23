//! `filesrc` — reads a file into pooled buffers via the reactor
//! (spec: Milestone applications §1; IO). Reactor-native: it registers its file,
//! keeps up to `credits` positioned reads in flight, and emits completed buffers.
//! Positioned reads (explicit offset) make it seek-ready.

use std::fs::File;
use std::path::{Path, PathBuf};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::io::{FileHandle, IoResult};
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

static DESC: ElementDesc = ElementDesc {
    name: "filesrc",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

pub struct FileSrc {
    path: PathBuf,
    file: FileHandle,
    started: bool,
    offset: u64,
    eof: bool,
    in_flight: u32,
    seq: u64,
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
        }
    }
}

impl Element for FileSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let f = File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path.display())))?;
        self.file = ctx.io().register(f);
        self.started = true;
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
            match c.result {
                IoResult::Ok(0) => self.eof = true, // c.buf recycles on drop
                IoResult::Ok(_) => ctx.out(PadId(0)).push(c.buf),
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
            Ok(Flow::Eos)
        } else {
            Ok(Flow::Ok)
        }
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns and closes the file
    }
}
