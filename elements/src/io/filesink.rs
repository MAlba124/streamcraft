//! `filesink` — writes incoming buffers to a file via the reactor
//! (spec: Milestone applications §1; IO). Reactor-native: it takes ownership of
//! input buffers and submits positioned writes, recycling each buffer when its
//! write completes.

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
use streamcraft_core::io::{FileHandle, IoResult};
use streamcraft_core::time::Timestamp;

/// Writes an untyped byte stream — offer the open `bytes` family, no fields.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "filesink",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

pub struct FileSink {
    path: PathBuf,
    file: FileHandle,
    started: bool,
    write_offset: u64,
    in_flight: u32,
}

impl FileSink {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            file: FileHandle(0),
            started: false,
            write_offset: 0,
            in_flight: 0,
        }
    }
}

impl Element for FileSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let f = File::create(&self.path)
            .map_err(|e| Error::Resource(format!("create {}: {e}", self.path.display())))?;
        self.file = ctx.io().register(f);
        self.started = true;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        if !self.started {
            return Err(Error::Todo("filesink not started"));
        }

        // Drain completed writes: the returned buffers recycle on drop.
        loop {
            let completion = ctx.io().next_completion();
            let c = match completion {
                Some(c) => c,
                None => break,
            };
            self.in_flight -= 1;
            if let IoResult::Err(k) = c.result {
                return Err(Error::Resource(format!("write: {k:?}")));
            }
        }

        // Submit positioned writes for available input, up to credits.
        let credits = ctx.io().credits();
        while self.in_flight < credits {
            let buf = match inputs.pop() {
                Some(b) => b,
                None => break,
            };
            let n = buf.memory.len() as u64;
            let offset = self.write_offset;
            self.write_offset += n;
            ctx.io().submit_write(self.file, offset, buf, 0);
            self.in_flight += 1;
        }

        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns, flushes, and closes the file
    }
}
