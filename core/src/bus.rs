//! The bus: out-of-band element/framework → application (spec: Events, queries, and
//! the bus). Drained by the app on the app's own thread — no callbacks on streaming
//! threads, ever.
//!
//! Milestone 1 backs it with `std::sync::mpsc` (spec-sanctioned: "replace
//! crossbeam-channel with std::sync::mpsc or, better, a purpose-built SPSC ring").
//! The bounded, class-aware version (spec) replaces the internals later.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use crate::error::Error;
use crate::event::TagList;
use crate::format::FixedFormat;
use crate::id::{ElementId, GroupId, LinkId, PadId};
use crate::time::Timestamp;

pub enum BusMessage {
    Error { element: ElementId, error: Error },
    Warning { element: ElementId, error: Error },
    Eos,
    StateChanged { old: State, new: State },
    PadAdded { element: ElementId, pad: PadId, format: FixedFormat },
    ElementAdded { element: ElementId, group: GroupId },
    ElementRemoved { element: ElementId },
    LinkChanged { link: LinkId },
    Tags { element: ElementId, tags: TagList },
    SubgraphJoined { group: GroupId, added_latency: Timestamp },
    LatencyChanged { old: Timestamp, new: Timestamp },
    Qos { sink: ElementId, lateness_ns: i64 },
    /// A branch failed and was isolated (spec: Supervision).
    BranchSealed { group: GroupId, error: Error },
}

/// Deliberately smaller than gst's. Pause is a clock op, not a state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Stopped,
    Ready,
    Playing,
}

/// The posting half, handed to elements via `Ctx`. Cheap to clone.
#[derive(Clone)]
pub struct BusSender {
    tx: Sender<BusMessage>,
}

impl BusSender {
    /// Post a message. Never blocks the data path; silently drops if the
    /// application has dropped the bus.
    pub fn send(&self, msg: BusMessage) {
        let _ = self.tx.send(msg);
    }
}

/// The receiving half, owned by the pipeline / drained by the application.
pub struct Bus {
    rx: Receiver<BusMessage>,
}

impl Bus {
    pub fn channel() -> (BusSender, Bus) {
        let (tx, rx) = std::sync::mpsc::channel();
        (BusSender { tx }, Bus { rx })
    }

    pub fn try_recv(&self) -> Option<BusMessage> {
        match self.rx.try_recv() {
            Ok(m) => Some(m),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    /// Blocking; app thread only. `None` once all senders are gone.
    pub fn recv(&self) -> Option<BusMessage> {
        self.rx.recv().ok()
    }
}
