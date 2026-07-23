//! Clocks, the interruptible `ClockWait`, and slaving (spec: Clocking and
//! synchronization — the flagship). Only sinks wait; everything upstream runs as
//! fast as backpressure allows.
//!
//! Two clocks ship in core:
//!
//! - [`InstantClock`]: the default real clock, monotonic and `std::time::Instant`
//!   backed.
//! - [`MockClock`]: virtual time advanced by hand via [`MockClock::advance`], so
//!   sync/latency/timeout tests run in microseconds of wall time instead of real
//!   sleeps (spec: Testing — "nothing ever sleeps").
//!
//! The flagship primitive is [`ClockWait`]: an *interruptible* wait. It blocks
//! until either the clock reaches a deadline ([`WaitOutcome::Reached`]) or another
//! thread calls [`ClockWait::interrupt`] ([`WaitOutcome::Interrupted`], the
//! flush / seek / shutdown path). Both wake sources notify the **same**
//! [`Condvar`]:
//!
//! - A wait is created by its clock via [`Clock::new_wait`], so the wait shares the
//!   clock's wake station. [`MockClock::advance`] and [`ClockWait::interrupt`] then
//!   `notify_all` the very same condvar the waiter is parked on — advancing virtual
//!   time past a deadline wakes the blocked thread with no wall-clock sleeping.
//! - For [`InstantClock`], time advances on its own, so the wait parks on a private
//!   condvar with a real [`Condvar::wait_timeout`] sized to `deadline - now`;
//!   [`ClockWait::interrupt`] notifies that same private condvar to cut the wait
//!   short.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::time::Timestamp;

/// The one deliberately tiny trait. Core ships [`InstantClock`] and [`MockClock`].
///
/// A clock is also the factory for waits against it: [`new_wait`](Clock::new_wait)
/// hands back a [`ClockWait`] wired to this clock's wake station, so the clock's
/// own progress (and interruption) can unblock the waiter. This keeps the trait
/// object-safe — every method takes `&self` and returns concrete types.
pub trait Clock: Send + Sync {
    /// Current time on this clock's timeline, in monotonic nanoseconds.
    fn now(&self) -> Timestamp;

    /// Create an interruptible [`ClockWait`] bound to this clock.
    ///
    /// The returned wait observes *this* clock's `now()` and, crucially, is woken
    /// when this clock makes progress (see [`MockClock::advance`]). Each call
    /// returns a fresh, independently-interruptible wait.
    fn new_wait(&self) -> ClockWait;
}

/// The shared parking station a [`ClockWait`] blocks on.
///
/// `interrupted` (and, for the mock clock, the virtual `now_ns`) live under the
/// same mutex as the predicate the waiter checks, so a notify can never slip
/// between "check predicate" and "start waiting" (the classic lost-wakeup). Both
/// [`MockClock::advance`] and [`ClockWait::interrupt`] mutate this state under the
/// lock and then `notify_all` `cvar`.
struct Wake {
    state: Mutex<WakeState>,
    cvar: Condvar,
}

struct WakeState {
    /// Virtual time in nanoseconds. Owned by [`MockClock`]; unused (stays `0`) for
    /// [`InstantClock`], whose `now()` reads a [`std::time::Instant`] instead.
    now_ns: u64,
    /// Sticky interrupt flag. Once set, current and future waits on this station
    /// return [`WaitOutcome::Interrupted`] immediately — the shutdown/flush latch.
    interrupted: bool,
}

impl Wake {
    fn new(now_ns: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(WakeState {
                now_ns,
                interrupted: false,
            }),
            cvar: Condvar::new(),
        })
    }

    /// Latch the interrupt and wake every parked waiter.
    fn interrupt(&self) {
        // A poisoned lock here means a waiter panicked mid-wait; recover the guard
        // and interrupt anyway — the shutdown path must never itself block.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.interrupted = true;
        drop(state);
        self.cvar.notify_all();
    }
}

/// Monotonic wall-clock, [`std::time::Instant`]-backed. The default real clock.
///
/// `now()` is nanoseconds elapsed since the instant this clock was created, so it
/// starts near zero and never goes backwards.
pub struct InstantClock {
    base: Instant,
}

impl InstantClock {
    /// A fresh clock whose zero is *now*.
    pub fn new() -> Self {
        Self {
            base: Instant::now(),
        }
    }
}

impl Default for InstantClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for InstantClock {
    fn now(&self) -> Timestamp {
        // Duration -> u64 ns saturates rather than panicking after ~584 years.
        let ns = u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Timestamp::from_nanos(ns)
    }

    fn new_wait(&self) -> ClockWait {
        // Real time advances on its own; the wait only needs a private condvar for
        // interrupts, plus this clock's base to measure the remaining duration.
        ClockWait {
            kind: WaitKind::Instant {
                base: self.base,
                wake: Wake::new(0),
            },
        }
    }
}

/// Test clock: virtual time advanced by hand, so sync/latency tests take
/// microseconds of wall time (spec: Testing).
///
/// Cheap to clone — every clone is an [`Arc`] handle onto the same virtual time and
/// the same wake station, so advancing time on one handle wakes waits created from
/// any handle.
pub struct MockClock {
    wake: Arc<Wake>,
}

impl MockClock {
    /// A mock clock starting at [`Timestamp::ZERO`].
    pub fn new() -> Self {
        Self { wake: Wake::new(0) }
    }

    /// A mock clock starting at `start`.
    pub fn starting_at(start: Timestamp) -> Self {
        Self {
            wake: Wake::new(start.0),
        }
    }

    /// Advance virtual time by `by` and wake every waiter parked on this clock.
    ///
    /// This is the mechanism that lets `wait_until` return without any real sleep:
    /// the new time and the wakeup are published under one lock, so a waiter whose
    /// deadline is now in the past observes it immediately. Advancing by
    /// [`Timestamp::NONE`] or `ZERO` is a no-op on time but still issues the wakeup.
    pub fn advance(&self, by: Timestamp) {
        let add = by.nanos().unwrap_or(0);
        let mut state = self.wake.state.lock().unwrap();
        state.now_ns = state.now_ns.saturating_add(add);
        drop(state);
        self.wake.cvar.notify_all();
    }
}

impl Default for MockClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MockClock {
    fn clone(&self) -> Self {
        Self {
            wake: Arc::clone(&self.wake),
        }
    }
}

impl Clock for MockClock {
    fn now(&self) -> Timestamp {
        Timestamp::from_nanos(self.wake.state.lock().unwrap().now_ns)
    }

    fn new_wait(&self) -> ClockWait {
        // Share the clock's wake station: `advance` notifies exactly this condvar.
        ClockWait {
            kind: WaitKind::Mock {
                wake: Arc::clone(&self.wake),
            },
        }
    }
}

/// How a [`ClockWait::wait_until`] ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaitOutcome {
    /// The clock reached (or was already past) the deadline.
    Reached,
    /// [`ClockWait::interrupt`] was called — the flush / seek / shutdown path.
    Interrupted,
}

/// Interruptible wait against a pipeline clock (spec: ClockWait). Written early,
/// small, and hammered with mock-clock tests, because an interruptible wait is
/// miserable to retrofit and cheap to build first.
///
/// Obtain one from [`Clock::new_wait`]. Two independent events end a
/// [`wait_until`](ClockWait::wait_until): the clock reaching the deadline, and
/// [`interrupt`](ClockWait::interrupt) from another thread. Interrupt is **sticky**
/// — once tripped, this wait's current and subsequent `wait_until` calls return
/// [`WaitOutcome::Interrupted`] at once; make a fresh wait to arm again.
pub struct ClockWait {
    kind: WaitKind,
}

enum WaitKind {
    /// Virtual time: park until `advance` pushes `now_ns` past the deadline (or an
    /// interrupt lands). No timeout — mock time only moves via `advance`.
    Mock { wake: Arc<Wake> },
    /// Real time: park with a `wait_timeout` sized to `deadline - now`, re-checking
    /// on each wake. `base` measures elapsed time; `wake` carries interrupts.
    Instant { base: Instant, wake: Arc<Wake> },
}

impl ClockWait {
    /// Block until the clock reaches `deadline`, or an interrupt lands.
    ///
    /// Returns [`WaitOutcome::Reached`] once the bound clock's `now()` is `>=
    /// deadline` — immediately, without parking, if the deadline is already in the
    /// past. Returns [`WaitOutcome::Interrupted`] if [`interrupt`](Self::interrupt)
    /// has been (or gets) called. A deadline of [`Timestamp::NONE`] is treated as
    /// "infinitely late": the wait can then only end via interrupt.
    pub fn wait_until(&self, deadline: Timestamp) -> WaitOutcome {
        match &self.kind {
            WaitKind::Mock { wake } => Self::wait_mock(wake, deadline),
            WaitKind::Instant { base, wake } => Self::wait_instant(*base, wake, deadline),
        }
    }

    fn wait_mock(wake: &Wake, deadline: Timestamp) -> WaitOutcome {
        // `NONE` (u64::MAX) reads as "infinitely late" here, so a plain `>=` on the
        // raw nanoseconds already means "never reached" — no special-casing needed.
        let deadline_ns = deadline.0;
        let mut state = wake.state.lock().unwrap();
        loop {
            if state.interrupted {
                return WaitOutcome::Interrupted;
            }
            if state.now_ns >= deadline_ns {
                return WaitOutcome::Reached;
            }
            // Park. `advance` and `interrupt` both notify this condvar under the
            // lock, so we cannot miss a wakeup between the checks above and here.
            state = wake.cvar.wait(state).unwrap();
        }
    }

    fn wait_instant(base: Instant, wake: &Wake, deadline: Timestamp) -> WaitOutcome {
        let deadline_ns = deadline.0;
        let mut state = wake.state.lock().unwrap();
        loop {
            if state.interrupted {
                return WaitOutcome::Interrupted;
            }
            let now_ns = u64::try_from(base.elapsed().as_nanos()).unwrap_or(u64::MAX);
            if now_ns >= deadline_ns {
                return WaitOutcome::Reached;
            }
            // Sleep only for the time left; on spurious wakeups or timeout we loop
            // and re-measure. An interrupt cuts this short via `notify_all`.
            let remaining = Duration::from_nanos(deadline_ns - now_ns);
            let (guard, _timed_out) = wake.cvar.wait_timeout(state, remaining).unwrap();
            state = guard;
        }
    }

    /// Trip the interrupt latch and promptly unblock the waiter (flush / seek /
    /// shutdown). Sticky: subsequent [`wait_until`](Self::wait_until) calls on this
    /// wait also return [`WaitOutcome::Interrupted`]. Safe to call from any thread,
    /// and idempotent.
    pub fn interrupt(&self) {
        match &self.kind {
            WaitKind::Mock { wake } | WaitKind::Instant { wake, .. } => wake.interrupt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Instant as StdInstant;

    // Generous ceiling: these assertions are about *not* hanging, not about tight
    // timing. Real work happens in microseconds; a full second means a deadlock.
    const SANITY: Duration = Duration::from_secs(1);

    /// Run `f` on a helper thread and assert it finishes within `SANITY`, returning
    /// its value. Fails loudly (rather than hanging the suite) if the wait deadlocks.
    fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let _ = tx.send(f());
        });
        let v = rx
            .recv_timeout(SANITY)
            .expect("wait did not complete in time — likely a missed wakeup / deadlock");
        handle.join().unwrap();
        v
    }

    #[test]
    fn mock_now_reflects_advance() {
        let clock = MockClock::new();
        assert_eq!(clock.now(), Timestamp::ZERO);
        clock.advance(Timestamp::from_millis(10));
        assert_eq!(clock.now(), Timestamp::from_millis(10));
        clock.advance(Timestamp::from_millis(5));
        assert_eq!(clock.now(), Timestamp::from_millis(15));
    }

    #[test]
    fn mock_starting_at_and_clone_share_time() {
        let a = MockClock::starting_at(Timestamp::from_secs(1));
        let b = a.clone();
        assert_eq!(b.now(), Timestamp::from_secs(1));
        // Advancing one handle is visible on the other — same virtual timeline.
        a.advance(Timestamp::from_secs(1));
        assert_eq!(b.now(), Timestamp::from_secs(2));
    }

    #[test]
    fn mock_past_deadline_returns_reached_without_blocking() {
        let clock = MockClock::starting_at(Timestamp::from_millis(100));
        let wait = clock.new_wait();
        // Deadline already behind the clock -> immediate, no parking.
        assert_eq!(
            wait.wait_until(Timestamp::from_millis(50)),
            WaitOutcome::Reached
        );
        // Exactly-equal deadline also counts as reached.
        assert_eq!(
            wait.wait_until(Timestamp::from_millis(100)),
            WaitOutcome::Reached
        );
    }

    #[test]
    fn mock_advance_wakes_blocked_waiter_with_reached() {
        let clock = MockClock::new();
        let wait = clock.new_wait();

        // A thread that first tells us it's about to block, then blocks.
        let (started_tx, started_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            wait.wait_until(Timestamp::from_millis(10))
        });

        // Make sure the waiter has reached the blocking point before we advance, so
        // this genuinely exercises the wakeup path (not the already-past shortcut).
        started_rx.recv().unwrap();
        // Not strictly necessary for correctness, but nudges the waiter to park.
        thread::yield_now();

        clock.advance(Timestamp::from_millis(10));

        let outcome = handle.join().expect("waiter thread panicked");
        assert_eq!(outcome, WaitOutcome::Reached);
    }

    #[test]
    fn mock_partial_advance_does_not_reach_but_full_advance_does() {
        let clock = MockClock::new();
        let wait = clock.new_wait();

        let (done_tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let outcome = wait.wait_until(Timestamp::from_millis(10));
            done_tx.send(outcome).unwrap();
        });

        // An advance that stays *below* the deadline must not wake the waiter.
        clock.advance(Timestamp::from_millis(5));
        assert_eq!(
            done_rx.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "partial advance wrongly satisfied the deadline"
        );

        // Crossing the deadline releases it as Reached.
        clock.advance(Timestamp::from_millis(5)); // now at 10ms == deadline
        assert_eq!(
            done_rx.recv_timeout(SANITY).expect("waiter never woke"),
            WaitOutcome::Reached
        );
        handle.join().unwrap();
    }

    #[test]
    fn mock_interrupt_unblocks_with_interrupted() {
        let clock = MockClock::new();
        let wait = Arc::new(clock.new_wait());

        let waiter = Arc::clone(&wait);
        let (started_tx, started_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            // Deadline the clock will never reach on its own (we never advance).
            waiter.wait_until(Timestamp::from_secs(3600))
        });

        started_rx.recv().unwrap();
        thread::yield_now();
        wait.interrupt();

        let outcome = handle.join().expect("waiter thread panicked");
        assert_eq!(outcome, WaitOutcome::Interrupted);
    }

    #[test]
    fn mock_interrupt_is_sticky() {
        let clock = MockClock::new();
        let wait = clock.new_wait();
        wait.interrupt();
        // Even a deadline already in the past yields Interrupted once latched:
        // shutdown wins over "reached".
        assert_eq!(wait.wait_until(Timestamp::ZERO), WaitOutcome::Interrupted);
        // And a future deadline returns immediately too, not blocking.
        assert_eq!(
            within(move || wait.wait_until(Timestamp::from_secs(3600))),
            WaitOutcome::Interrupted
        );
    }

    #[test]
    fn mock_none_deadline_only_ends_on_interrupt() {
        let clock = MockClock::new();
        let wait = Arc::new(clock.new_wait());

        let waiter = Arc::clone(&wait);
        let (started_tx, started_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            waiter.wait_until(Timestamp::NONE)
        });

        started_rx.recv().unwrap();
        thread::yield_now();
        // Advancing time — even a lot — must NOT satisfy a NONE ("infinitely late")
        // deadline; only interrupt can.
        clock.advance(Timestamp::from_secs(10_000));
        thread::yield_now();
        wait.interrupt();

        assert_eq!(handle.join().unwrap(), WaitOutcome::Interrupted);
    }

    #[test]
    fn instant_now_is_monotonic_and_starts_near_zero() {
        let clock = InstantClock::new();
        let a = clock.now();
        let b = clock.now();
        assert!(a.is_some() && b.is_some());
        assert!(b >= a, "InstantClock::now must be monotonic");
        // Starts near zero: well under a second at construction time.
        assert!(a < Timestamp::from_secs(1));
    }

    #[test]
    fn instant_wait_until_reaches_after_real_time() {
        let clock = InstantClock::new();
        let wait = clock.new_wait();
        let start = StdInstant::now();
        // A small, real ~5ms wait: the one place we touch real time.
        let deadline = clock.now().saturating_add(Timestamp::from_millis(5));
        assert_eq!(wait.wait_until(deadline), WaitOutcome::Reached);
        let elapsed = start.elapsed();
        // It actually waited (not a busy shortcut) but nowhere near the sanity cap.
        assert!(
            elapsed >= Duration::from_millis(4),
            "returned too early: {elapsed:?}"
        );
        assert!(elapsed < SANITY, "waited far too long: {elapsed:?}");
    }

    #[test]
    fn instant_past_deadline_returns_immediately() {
        let clock = InstantClock::new();
        let wait = clock.new_wait();
        let start = StdInstant::now();
        // Deadline already in the past -> no real sleep.
        assert_eq!(wait.wait_until(Timestamp::ZERO), WaitOutcome::Reached);
        assert!(start.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn instant_interrupt_unblocks_promptly() {
        let clock = InstantClock::new();
        let wait = Arc::new(clock.new_wait());

        let waiter = Arc::clone(&wait);
        let start = StdInstant::now();
        let (started_tx, started_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            // Far-future deadline: only interrupt should end this.
            waiter.wait_until(Timestamp::from_secs(3600))
        });

        started_rx.recv().unwrap();
        thread::yield_now();
        wait.interrupt();

        assert_eq!(handle.join().unwrap(), WaitOutcome::Interrupted);
        // Promptly: nowhere near the 3600s deadline.
        assert!(start.elapsed() < SANITY, "interrupt was not prompt");
    }

    /// The wait is usable behind `&dyn Clock` — the object-safety guarantee the
    /// pipeline relies on (it holds the master as a trait object).
    #[test]
    fn clock_is_object_safe() {
        let clock: Box<dyn Clock> = Box::new(MockClock::starting_at(Timestamp::from_millis(20)));
        assert_eq!(clock.now(), Timestamp::from_millis(20));
        let wait = clock.new_wait();
        assert_eq!(
            wait.wait_until(Timestamp::from_millis(10)),
            WaitOutcome::Reached
        );
    }
}
