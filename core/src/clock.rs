//! Clocks, the interruptible `ClockWait`, and slaving (spec: Clocking and
//! synchronization — the flagship). Only sinks wait; everything upstream runs as
//! fast as backpressure allows.

use crate::time::Timestamp;

/// The one deliberately tiny trait. Core ships `InstantClock` and `MockClock`.
pub trait Clock: Send + Sync {
    fn now(&self) -> Timestamp;
}

/// Monotonic wall-clock, `std::time::Instant`-backed. TODO(step 2).
pub struct InstantClock {
    _priv: (),
}

/// Test clock: virtual time advanced by hand, so sync/latency tests take
/// microseconds of wall time (spec: Testing).
pub struct MockClock {
    _priv: (),
}

impl MockClock {
    pub fn advance(&self, _by: Timestamp) {
        todo!("spec: Testing")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaitOutcome {
    Reached,
    Interrupted,
}

/// Interruptible wait against the pipeline clock (spec: ClockWait). Written early,
/// small, hammered with mock-clock + loom tests. `as_waitable()` is timerfd-shaped
/// so it folds into a reactor's single wait.
pub struct ClockWait {
    _priv: (),
}

impl ClockWait {
    pub fn wait_until(&self, _deadline: Timestamp) -> WaitOutcome {
        todo!("spec: Clocking")
    }

    /// The flush / seek / shutdown path.
    pub fn interrupt(&self) {
        todo!("spec: Clocking")
    }
}
