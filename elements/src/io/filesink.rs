//! `filesink` — writes incoming buffers to a file (spec: Milestone applications §1).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::time::Timestamp;

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &[],
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
    file: Option<File>,
}

impl FileSink {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            file: None,
        }
    }
}

impl Element for FileSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        let f = File::create(&self.path)
            .map_err(|e| Error::Resource(format!("create {}: {e}", self.path.display())))?;
        self.file = Some(f);
        Ok(())
    }

    fn process(&mut self, _ctx: &mut Ctx, inputs: Inputs<'_>) -> Result<Flow, Error> {
        let file = self.file.as_mut().ok_or(Error::Todo("filesink not started"))?;
        if let Some(input) = inputs.single() {
            for i in 0..input.len() {
                file.write_all(input.memory(i).data())
                    .map_err(|e| Error::Resource(format!("write {}: {e}", self.path.display())))?;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        if let Some(mut f) = self.file.take() {
            let _ = f.flush();
        }
    }
}
