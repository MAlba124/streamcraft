//! `filesrc` — reads a file into pooled buffers (spec: Milestone applications §1).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &[],
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
    file: Option<File>,
}

impl FileSrc {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            file: None,
        }
    }
}

impl Element for FileSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        let f = File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path.display())))?;
        self.file = Some(f);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        let file = self.file.as_mut().ok_or(Error::Todo("filesrc not started"))?;
        let mut buf = ctx.alloc(PadId(0));
        let n = file
            .read(buf.memory.as_mut_full())
            .map_err(|e| Error::Resource(format!("read {}: {e}", self.path.display())))?;
        if n == 0 {
            return Ok(Flow::Eos);
        }
        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.file = None;
    }
}
