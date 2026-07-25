//! `filesink` — writes incoming buffers to a file via the reactor
//! (spec: Milestone applications §1; IO). Reactor-native: it takes ownership of
//! input buffers and submits positioned writes, recycling each buffer when its
//! write completes.

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

/// The file to write (spec: Plugins — `parse("filesink path=…")`); see [`filesrc`]'s
/// `path` prop. Structural: the file is created in `start()`.
///
/// [`filesrc`]: crate::io::FileSrc
static PROPS: [PropDesc; 1] = [PropDesc {
    name: "path",
    allowed: Constraint::Any,
    live: false,
}];

static DESC: ElementDesc = ElementDesc {
    name: "filesink",
    pads: &PADS,
    props: &PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: Some(|| Box::new(FileSink::new(""))),
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
        // A parsed `path=` overrides the constructor value (spec: Plugins — the string
        // rides `Value::Id`, resolved by name off the value vocabulary).
        if let Some(streamcraft_core::format::Value::Id(id)) = ctx.prop("path") {
            if let Some(s) = ctx.value_name(id) {
                self.path = PathBuf::from(s);
            }
        }
        // Unlink an existing regular file instead of truncating over it. Truncating a
        // large file frees its extents *synchronously through the journal*, and the new
        // run's first writes then stall behind that commit — measured on ext4-on-LUKS:
        // back-to-back 1.4 GB remuxes to the same path froze ~1.5–3 s at ~5 MiB, while
        // unlink-first runs were stall-free (the unlinked inode frees lazily via the
        // orphan list, off the writer's path). Symlinks are left alone (removing one
        // would unlink the link, not the target — truncate-through keeps that
        // behavior), and errors are ignored — `create` surfaces anything real.
        match std::fs::symlink_metadata(&self.path) {
            Ok(m) if m.file_type().is_file() => {
                let _ = std::fs::remove_file(&self.path);
            }
            _ => {}
        }
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

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // The single back-patch (spec: Events — `Patch`): an indexed container's front
        // reservation (a Matroska SeekHead) now that its target's offset is known.
        // In-band, so it arrives after every buffer whose bytes precede it was popped;
        // ordering versus still-in-flight appends is irrelevant — the offsets are
        // disjoint and positioned IO is order-independent.
        if let Event::Patch { offset, data } = event {
            if !self.started {
                return Err(Error::Todo("filesink not started"));
            }
            let buf = streamcraft_core::buffer::Buffer {
                memory: data.clone(),
                pts: Timestamp::NONE,
                dts: Timestamp::NONE,
                duration: Timestamp::NONE,
                flags: streamcraft_core::buffer::BufferFlags::empty(),
                format: streamcraft_core::id::FormatId(0),
                sync: None,
            };
            ctx.io().submit_write(self.file, *offset, buf, 0);
            self.in_flight += 1;
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.started = false; // the reactor owns, flushes, and closes the file
    }
}
