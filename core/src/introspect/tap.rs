//! The two hot-path taps (spec: The two hot-path touches). Both are structured so the
//! feature-off build has no code, and the feature-on-but-no-client path is one branch:
//!
//! - **Bus tap**: a POD [`BusTapRow`] is built inside the bus's *already-held* lock,
//!   right before the message is posted, and copied into each attached slot's ring
//!   (try_push, drop-and-count on full). The tap copies, never steals — the app's
//!   `try_recv` is untouched.
//! - **Log tap**: a [`LogTapRegistry`] the [`LogDrainThread`](crate::pipeline) publishes
//!   each formatted record into; the fast path is one relaxed `nsubs` load == 0 → return,
//!   so streaming cost with no subscriber is byte-identical to logging alone.
//!
//! `LogRecord` is `Copy` with process-'static pointers, so it crosses the in-process
//! ring as-is; interning to string ids happens on the client thread (server.rs).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::bus::{BusMessage, MessageClass};
use crate::error::Error;
use crate::log::LogRecord;
use crate::ring::{Consumer, Producer};

use super::wire::BUS_MSG_TEXT;

// ---------------------------------------------------------------------------
// BusTapRow — a POD image of a BusMessage (spec: BusMsg 96 B row semantics)
// ---------------------------------------------------------------------------

/// A zero-alloc POD image of a [`BusMessage`], built under the bus lock. Mirrors the
/// wire [`BusMsgRow`](super::wire::BusMsgRow) a/b/c/d/msg layout, minus the wire `seq`
/// (the bus stamps that separately). `kind` is the `BusMessage` variant ordinal in
/// declaration order (bus.rs:26-41), pinned by a test that fails to compile if a
/// variant is added without a mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusTapRow {
    /// Per-bus monotonic sequence, stamped under the bus lock at build time — a total
    /// order across every tap ring. Copied straight onto the wire `BusMsgRow`.
    pub seq: u64,
    pub kind: u8,
    pub class: u8,
    pub msg_len: u8,
    pub a: u32,
    pub b: u32,
    pub c: u64,
    pub d: u64,
    pub msg: [u8; BUS_MSG_TEXT],
}

impl BusTapRow {
    /// Build the row from a message *without consuming it* (the bus still posts the
    /// original), stamping the under-lock `seq`. Copies at most [`BUS_MSG_TEXT`] bytes
    /// of any error text, truncated at a UTF-8 char boundary so `text()` never sees an
    /// invalid split.
    pub fn from_msg(msg: &BusMessage, seq: u64) -> BusTapRow {
        let class = match msg.class() {
            MessageClass::Droppable => 0u8,
            MessageClass::Critical => 1u8,
        };
        let mut row = BusTapRow {
            seq,
            kind: bus_kind_ordinal(msg),
            class,
            msg_len: 0,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            msg: [0u8; BUS_MSG_TEXT],
        };
        match msg {
            BusMessage::Error { element, error } => {
                row.a = element.0;
                row.copy_text(error_text(error));
            }
            BusMessage::Warning { element, error } => {
                row.a = element.0;
                row.copy_text(error_text(error));
            }
            BusMessage::Eos => {}
            BusMessage::StateChanged { old, new } => {
                row.a = state_ordinal(*old) as u32;
                row.b = state_ordinal(*new) as u32;
            }
            BusMessage::PadAdded { element, pad, format } => {
                row.a = element.0;
                row.b = pad.0;
                row.c = format.family.0 as u64;
            }
            BusMessage::ElementAdded { element, group } => {
                row.a = element.0;
                row.b = group.0;
            }
            BusMessage::ElementRemoved { element } => {
                row.a = element.0;
            }
            BusMessage::LinkChanged { link } => {
                row.a = link.0;
            }
            BusMessage::Tags { element, .. } => {
                // Tag payloads do NOT ride the row — ids only (spec).
                row.a = element.0;
            }
            BusMessage::SubgraphJoined { group, added_latency } => {
                row.a = group.0;
                row.c = added_latency.0;
            }
            BusMessage::LatencyChanged { old, new } => {
                row.c = old.0;
                row.d = new.0;
            }
            BusMessage::Qos { sink, lateness_ns } => {
                row.a = sink.0;
                row.c = *lateness_ns as u64;
            }
            BusMessage::BranchSealed { group, error } => {
                row.a = group.0;
                row.copy_text(error_text(error));
            }
            BusMessage::DurationChanged { element, ns } => {
                row.a = element.0;
                row.c = *ns;
            }
        }
        row
    }

    /// Copy `s` into the inline `msg`, truncated to [`BUS_MSG_TEXT`] at a char boundary.
    fn copy_text(&mut self, s: &str) {
        let mut end = s.len().min(BUS_MSG_TEXT);
        // Back off to a UTF-8 char boundary so `text()` decodes cleanly.
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        let bytes = &s.as_bytes()[..end];
        self.msg[..bytes.len()].copy_from_slice(bytes);
        self.msg_len = bytes.len() as u8;
    }
}

/// The `BusMessage` variant ordinal in declaration order (bus.rs:26-41). This `match`
/// is exhaustive with an explicit arm per variant, so adding a variant to `BusMessage`
/// without updating this fails to compile — the pin the spec asks for.
pub const fn bus_kind_ordinal(msg: &BusMessage) -> u8 {
    match msg {
        BusMessage::Error { .. } => 0,
        BusMessage::Warning { .. } => 1,
        BusMessage::Eos => 2,
        BusMessage::StateChanged { .. } => 3,
        BusMessage::PadAdded { .. } => 4,
        BusMessage::ElementAdded { .. } => 5,
        BusMessage::ElementRemoved { .. } => 6,
        BusMessage::LinkChanged { .. } => 7,
        BusMessage::Tags { .. } => 8,
        BusMessage::SubgraphJoined { .. } => 9,
        BusMessage::LatencyChanged { .. } => 10,
        BusMessage::Qos { .. } => 11,
        BusMessage::BranchSealed { .. } => 12,
        BusMessage::DurationChanged { .. } => 13,
    }
}

/// The `State` ordinal (bus.rs:79-83 declaration order): Stopped, Ready, Playing.
fn state_ordinal(s: crate::bus::State) -> u8 {
    match s {
        crate::bus::State::Stopped => 0,
        crate::bus::State::Ready => 1,
        crate::bus::State::Playing => 2,
    }
}

/// Human text for the inline bus message (Error has no `Display`, so match here).
fn error_text(e: &Error) -> &str {
    match e {
        Error::NegotiationFailed => "negotiation failed",
        Error::Element { message, .. } => message,
        Error::Resource(m) => m,
        Error::Todo(m) => m,
    }
}

// ---------------------------------------------------------------------------
// Bus tap slots + port (spec: bus.rs Inner gains taps; Bus gains tap_port)
// ---------------------------------------------------------------------------

/// The id of an attached bus tap, returned by [`attach`](BusTapPortState::attach) and
/// passed to [`detach`](BusTapPortState::detach).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapId(pub u32);

/// One attached bus tap: a producer ring plus a per-slot drop tally. Lives inside the
/// bus `Inner` (feature-gated) so publishing is one memcpy under the already-held lock.
pub struct BusTapSlot {
    pub id: TapId,
    pub tx: Producer<BusTapRow>,
    /// Rows dropped because this slot's ring was full (slow client). Read by the owner.
    pub dropped: u64,
}

impl BusTapSlot {
    /// Try to enqueue `row`; count a drop on a full ring. Never blocks (spec: the bus
    /// tap must never stall a streaming thread posting a message).
    pub fn publish(&mut self, row: BusTapRow) {
        if self.tx.try_push(row).is_err() {
            self.dropped += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// LogTapRegistry (spec: pipeline.rs only; log.rs unchanged)
// ---------------------------------------------------------------------------

/// One log subscription: a monotonic tag, its producer ring, and a drop tally.
struct LogSub {
    id: u64,
    tx: Producer<LogRecord>,
    dropped: u64,
}

/// A fan-out registry the log drain thread publishes each formatted record into.
/// Fast path is one relaxed `nsubs` load: 0 → return, so streaming with no client
/// costs one branch on top of the existing drain formatting.
pub struct LogTapRegistry {
    nsubs: AtomicUsize,
    next_id: AtomicU64,
    subs: Mutex<Vec<LogSub>>,
}

impl Default for LogTapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl LogTapRegistry {
    // Cold: one-time registry construction (the empty subs Vec), not per media buffer.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            nsubs: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
            subs: Mutex::new(Vec::new()),
        }
    }

    /// Attach a consumer for a client. Returns `(id, consumer)`; the id detaches later.
    pub fn attach(&self, cap: usize) -> (u64, Consumer<LogRecord>) {
        let (tx, rx) = crate::ring::spsc::<LogRecord>(cap);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        subs.push(LogSub { id, tx, dropped: 0 });
        self.nsubs.store(subs.len(), Ordering::Relaxed);
        (id, rx)
    }

    /// Detach a subscription, returning its final drop count.
    pub fn detach(&self, id: u64) -> u64 {
        let mut subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        let mut dropped = 0;
        if let Some(pos) = subs.iter().position(|s| s.id == id) {
            dropped = subs[pos].dropped;
            subs.remove(pos);
        }
        self.nsubs.store(subs.len(), Ordering::Relaxed);
        dropped
    }

    /// The current per-subscription drop count (for a `Dropped` push when it advances).
    pub fn dropped(&self, id: u64) -> u64 {
        let subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        subs.iter().find(|s| s.id == id).map(|s| s.dropped).unwrap_or(0)
    }

    /// Publish one record to every subscriber (drain thread). Fast path: no subscriber
    /// ⇒ one relaxed load and return. On a full ring the record is dropped and counted.
    #[inline]
    pub fn publish(&self, rec: &LogRecord) {
        if self.nsubs.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        for sub in subs.iter_mut() {
            if sub.tx.try_push(*rec).is_err() {
                sub.dropped += 1;
            }
        }
    }
}
