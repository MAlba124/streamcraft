//! Lock-free single-producer / single-consumer **byte** ring — the PCM hand-off between
//! the profluens render thread (producer) and PipeWire's real-time process callback
//! (consumer).
//!
//! The RT callback must never lock a mutex or allocate, so this ring is designed around
//! an asymmetry the audio world takes for granted: **only the non-RT side ever blocks or
//! signals.** The consumer's [`Consumer::pull`] is wait-free — two atomic index reads, a pair
//! of `memcpy`s, one Release store, no mutex, no allocation, ever. The producer, when the
//! ring is full, parks on a condvar with a bounded timeout and re-polls; the device draining
//! the ring makes room, which the producer notices on its next wake. So the device paces the
//! pipeline (backpressure) without the RT thread having to hand-shake a wakeup back.
//!
//! Core has a general `spsc<T>` ring, but it is element-oriented (`try_push(val: T)` /
//! `try_pop() -> Option<T>`): using it for a byte stream means one Acquire/Release index
//! pair *and a `SeqCst` fence per byte*, which is untenable at PCM rates. This ring copies
//! whole slices under a single index publish, so a `pull` that fills a device buffer is two
//! `memcpy`s regardless of size. It follows core's memory-ordering discipline (monotonic
//! `head`/`tail`, Acquire the peer / Release your own, power-of-two `mask`, `UnsafeCell`
//! byte storage) — just specialized to bulk `Copy` bytes with a one-sided parking policy.

#![allow(unsafe_code)] // plugin crate: this module is the audited unsafe area, mirroring core::ring

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// How long a full/draining producer parks before re-polling the RT consumer's progress.
/// Bounded so we never miss the consumer making room even though it never signals us; short
/// enough that shutdown/backpressure stays responsive, long enough to avoid a busy spin.
const PARK: Duration = Duration::from_millis(2);

/// Force the two indices onto separate cache lines — the producer writes `tail` while the
/// consumer writes `head`; sharing a line would bounce it between cores (false sharing).
#[repr(align(64))]
struct CachePadded<T>(T);

struct Inner {
    /// Byte storage; `cap` is a power of two so indices mask cheaply. `UnsafeCell` because
    /// producer and consumer write disjoint regions concurrently (SPSC discipline).
    buf: Box<[UnsafeCell<u8>]>,
    cap: usize,
    mask: usize,
    /// Consumer-owned position (monotonic; masked for indexing).
    head: CachePadded<AtomicUsize>,
    /// Producer-owned position (monotonic; masked for indexing).
    tail: CachePadded<AtomicUsize>,
    /// Flush (spec: flush/seek). The producer bumps `flush_gen` and publishes the tail it
    /// held at flush time in `flush_to`; the RT consumer, seeing a new generation, jumps
    /// `head` forward to `flush_to`, dropping the buffered pre-seek PCM while preserving any
    /// post-seek bytes the producer has pushed since (index ≥ `flush_to`). Only the consumer
    /// writes `head`, so this stays SPSC-safe — the producer never touches the consumer's
    /// index. `seen_flush_gen` is the consumer's private record of the last generation it
    /// acted on (it alone writes it), read on the pull fast path — one relaxed load.
    flush_gen: AtomicUsize,
    flush_to: AtomicUsize,
    seen_flush_gen: CachePadded<AtomicUsize>,
    /// Either end went away (sink shutdown / PW loop exited); unblocks the parked producer.
    closed: AtomicBool,
    /// Parking gate for the producer's blocking path. The RT consumer never touches this.
    gate: Mutex<()>,
    cvar: Condvar,
}

// SAFETY: access is disciplined SPSC — exactly one producer touching [tail, ..) and one
// consumer touching [head, tail); the shared bytes each side reaches are provably disjoint
// (see `push`/`pull`), and index hand-off uses Acquire/Release. So sharing `Inner` across
// the two threads is sound even though `UnsafeCell` is not `Sync` by default.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    /// Bytes currently readable by the consumer. Acquire on `tail` so the producer's slot
    /// writes that precede its Release publish are visible before we read them.
    fn len(&self) -> usize {
        let tail = self.tail.0.load(Ordering::Acquire);
        let head = self.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }
}

/// The producing (render-thread) end of the byte ring.
pub struct Producer {
    inner: Arc<Inner>,
}

/// The consuming (PipeWire RT callback) end of the byte ring.
pub struct Consumer {
    inner: Arc<Inner>,
}

/// Create a byte ring holding up to `capacity` bytes (rounded up to a power of two).
pub fn spsc(capacity: usize) -> (Producer, Consumer) {
    let cap = capacity.max(1).next_power_of_two();
    let buf = (0..cap).map(|_| UnsafeCell::new(0u8)).collect::<Vec<_>>().into_boxed_slice();
    let inner = Arc::new(Inner {
        buf,
        cap,
        mask: cap - 1,
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
        flush_gen: AtomicUsize::new(0),
        flush_to: AtomicUsize::new(0),
        seen_flush_gen: CachePadded(AtomicUsize::new(0)),
        closed: AtomicBool::new(false),
        gate: Mutex::new(()),
        cvar: Condvar::new(),
    });
    (Producer { inner: Arc::clone(&inner) }, Consumer { inner })
}

impl Producer {
    /// Copy as many bytes of `src` into the ring as fit, returning how many were written.
    /// Non-blocking; the sole caller of the blocking [`push`](Self::push) wrapper.
    fn try_push(&self, src: &[u8]) -> usize {
        let tail = self.inner.tail.0.load(Ordering::Relaxed); // we own tail
        let head = self.inner.head.0.load(Ordering::Acquire); // peer's progress
        let free = self.inner.cap - tail.wrapping_sub(head);
        let n = free.min(src.len());
        if n == 0 {
            return 0;
        }
        // The free region [tail, tail+n) may wrap the physical buffer; split into up to two
        // contiguous runs. These bytes are past `head`, so the consumer cannot touch them —
        // we hold exclusive access and copy without atomics.
        let start = tail & self.inner.mask;
        let first = n.min(self.inner.cap - start);
        // SAFETY: [start, start+first) is a free, in-bounds run owned solely by the producer.
        unsafe {
            let dst = self.inner.buf.as_ptr().add(start) as *mut u8;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, first);
        }
        if first < n {
            // SAFETY: wrap remainder [0, n-first); also free and producer-owned this round.
            unsafe {
                let dst = self.inner.buf.as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src.as_ptr().add(first), dst, n - first);
            }
        }
        // Release: publish the new tail *after* the slot writes above, so a consumer that
        // Acquire-loads this tail is guaranteed to see the bytes.
        self.inner.tail.0.store(tail.wrapping_add(n), Ordering::Release);
        n
    }

    /// Append all of `bytes`, blocking while the ring is full so the audio device paces the
    /// pipeline (backpressure). A test-only convenience over
    /// [`push_interruptible`](Self::push_interruptible) (never aborts); production code uses the
    /// interruptible form so a seek can unblock a paused sink.
    #[cfg(test)]
    pub fn push(&self, bytes: &[u8]) {
        self.push_interruptible(bytes, || false);
    }

    /// Append all of `bytes`, blocking while the ring is full so the audio device paces the
    /// pipeline (backpressure), but give up (returning `false`, dropping the unwritten tail) as
    /// soon as `abort` reads true — so a **paused** sink, blocked here on a full ring the RT
    /// callback is deliberately not draining, can still notice a seek and return control to the
    /// scheduler to run the flush (spec: flush/seek). `abort` is re-checked each park cycle, so
    /// the wait ends within one `PARK` even though the consumer never signals. Returns `true`
    /// when every byte was pushed. Also returns early if the sink was closed (PW thread gone).
    pub fn push_interruptible(&self, mut bytes: &[u8], abort: impl Fn() -> bool) -> bool {
        while !bytes.is_empty() {
            if self.inner.closed.load(Ordering::Acquire) || abort() {
                return false;
            }
            let n = self.try_push(bytes);
            bytes = &bytes[n..];
            if !bytes.is_empty() {
                let g = self.inner.gate.lock().unwrap();
                if self.inner.len() == self.inner.cap
                    && !self.inner.closed.load(Ordering::Acquire)
                    && !abort()
                {
                    let _ = self.inner.cvar.wait_timeout(g, PARK).unwrap();
                }
            }
        }
        true
    }

    /// Discard the currently-buffered PCM on a seek (spec: flush/seek). Publishes the current
    /// tail as the flush point and bumps the flush generation; the RT consumer drops
    /// everything up to that point on its next `pull`. Bytes pushed *after* this call (the
    /// post-seek PCM) are preserved. Wait-free and safe to interleave with the RT consumer:
    /// the producer only writes its own `flush_to`/`flush_gen`, never the consumer's `head`.
    pub fn flush(&self) {
        let tail = self.inner.tail.0.load(Ordering::Relaxed); // we own tail
        self.inner.flush_to.store(tail, Ordering::Release);
        self.inner.flush_gen.fetch_add(1, Ordering::Release);
    }

    /// End-of-stream: block until the consumer has played out every buffered byte (or the sink
    /// closes). After this returns, the ring is empty and the stream is complete. "Signalling
    /// EOS" is exactly this wait — the producer stops pushing and blocks until drained, which
    /// the consumer effects by advancing `head` to `tail`.
    pub fn drain(&self) {
        loop {
            if self.inner.closed.load(Ordering::Acquire) || self.inner.len() == 0 {
                return;
            }
            // Same one-sided park as `push`: the consumer drains and never signals, so poll.
            let g = self.inner.gate.lock().unwrap();
            if self.inner.len() != 0 && !self.inner.closed.load(Ordering::Acquire) {
                let _ = self.inner.cvar.wait_timeout(g, PARK).unwrap();
            }
        }
    }
}

impl Consumer {
    /// Fill `out` with available PCM, zero-padding whatever is left (an underrun becomes
    /// silence, so the device never plays stale/garbage bytes). **Wait-free and
    /// allocation-free** — safe to call from the PipeWire RT thread: two index loads, up to
    /// two `memcpy`s, one Release store, no mutex.
    pub fn pull(&self, out: &mut [u8]) {
        // Flush check (spec: flush/seek): if the producer bumped the flush generation, jump
        // `head` to the published flush point, dropping the buffered pre-seek PCM. Guard with
        // a wrap-safe distance test so we never move `head` backward (if we have already
        // consumed past the flush point, there is nothing stale to drop). One relaxed load on
        // the fast path when no flush is pending — no fence, no lock; stays RT-safe.
        let g = self.inner.flush_gen.load(Ordering::Acquire);
        if g != self.inner.seen_flush_gen.0.load(Ordering::Relaxed) {
            let ft = self.inner.flush_to.load(Ordering::Acquire);
            let head = self.inner.head.0.load(Ordering::Relaxed);
            if ft.wrapping_sub(head) <= self.inner.cap {
                self.inner.head.0.store(ft, Ordering::Release);
            }
            self.inner.seen_flush_gen.0.store(g, Ordering::Relaxed);
        }

        let head = self.inner.head.0.load(Ordering::Relaxed); // we own head
        let tail = self.inner.tail.0.load(Ordering::Acquire); // peer's publish
        let avail = tail.wrapping_sub(head);
        let n = avail.min(out.len());
        if n > 0 {
            // [head, head+n) is initialized and published by the producer; it can't advance
            // into this region because it stops at `head`, so we read it without atomics.
            let start = head & self.inner.mask;
            let first = n.min(self.inner.cap - start);
            // SAFETY: [start, start+first) is an in-bounds published run owned by the consumer.
            unsafe {
                let src = self.inner.buf.as_ptr().add(start) as *const u8;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), first);
            }
            if first < n {
                // SAFETY: wrap remainder [0, n-first); also published and consumer-owned.
                unsafe {
                    let src = self.inner.buf.as_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(first), n - first);
                }
            }
            // Release: free these slots for the producer only after we've copied out of them.
            self.inner.head.0.store(head.wrapping_add(n), Ordering::Release);
        }
        for slot in &mut out[n..] {
            *slot = 0;
        }
    }

    /// Bytes currently available to pull. Used by tests to reassemble without trusting the
    /// silence padding; the RT callback itself just calls [`pull`](Self::pull).
    #[cfg(test)]
    fn available(&self) -> usize {
        self.inner.len()
    }

    /// Whether the producer has closed the ring (dropped / signalled shutdown). The ring may
    /// still hold undrained bytes — keep pulling until [`available`](Self::available) is 0.
    #[cfg(test)]
    fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

/// Close the ring from either end and wake a parked producer. Called on sink shutdown and
/// when the PipeWire loop exits, so `push`/`drain` return promptly instead of parking out.
impl Inner {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.cvar.notify_all();
    }
}

impl Producer {
    /// Mark the ring closed (shutdown). Idempotent.
    pub fn close(&self) {
        self.inner.close();
    }
}

// Dropping either end closes the ring, so a peer blocked on the other side (a parked
// producer, or a consumer's `is_closed` check) is released promptly at teardown. The RT
// consumer closes purely via this drop — it never calls a method that could park it — while
// the producer also exposes an explicit [`Producer::close`] for the sink's `stop` path.
impl Drop for Producer {
    fn drop(&mut self) {
        self.inner.close();
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        self.inner.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn push_pull_roundtrips_bytes() {
        let (p, c) = spsc(1024);
        assert_eq!(p.try_push(&[1, 2, 3, 4]), 4);
        let mut out = [0u8; 4];
        c.pull(&mut out);
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn pull_underrun_is_silence() {
        let (p, c) = spsc(1024);
        p.try_push(&[9, 9]);
        let mut out = [7u8; 6];
        c.pull(&mut out);
        assert_eq!(out, [9, 9, 0, 0, 0, 0], "available bytes, then silence");
    }

    #[test]
    fn pull_from_empty_is_all_silence() {
        let (_p, c) = spsc(16);
        let mut out = [0xAAu8; 8];
        c.pull(&mut out);
        assert_eq!(out, [0u8; 8], "empty ring → pure silence, no read");
    }

    #[test]
    fn flush_drops_buffered_and_preserves_post_flush() {
        // Seek: the pre-flush PCM is discarded, but bytes pushed after the flush (the
        // post-seek audio) survive and play — the flush point is the tail at flush time.
        let (p, c) = spsc(64);
        p.push(&[1, 2, 3, 4]); // pre-flush (stale after a seek)
        p.flush();
        p.push(&[5, 6, 7, 8]); // post-flush (the new position's audio)
        let mut out = [0u8; 4];
        c.pull(&mut out);
        assert_eq!(out, [5, 6, 7, 8], "pre-flush dropped, post-flush kept");
    }

    #[test]
    fn flush_then_empty_pulls_silence() {
        // Flush with nothing pushed after → the ring is empty → pure silence, no stale bytes.
        let (p, c) = spsc(64);
        p.push(&[1, 2, 3, 4]);
        p.flush();
        let mut out = [0x77u8; 4];
        c.pull(&mut out);
        assert_eq!(out, [0, 0, 0, 0], "flushed ring is empty → silence");
    }

    #[test]
    fn flush_after_partial_consume_drops_only_the_rest() {
        // The consumer has already played some pre-flush bytes; the flush drops just the
        // remaining buffered ones, never rewinding over what was consumed.
        let (p, c) = spsc(64);
        p.push(&[1, 2, 3, 4, 5, 6]);
        let mut out = [0u8; 2];
        c.pull(&mut out); // consume [1,2]
        assert_eq!(out, [1, 2]);
        p.flush(); // drop the buffered [3,4,5,6]
        p.push(&[7, 8]);
        let mut out2 = [0u8; 2];
        c.pull(&mut out2);
        assert_eq!(out2, [7, 8], "only post-flush bytes remain");
    }

    #[test]
    fn try_push_stops_at_capacity() {
        // Capacity rounds up to the next power of two (5 → 8); a full ring accepts no more.
        let (p, c) = spsc(5);
        assert_eq!(p.try_push(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 8, "fills exactly cap");
        assert_eq!(p.try_push(&[11]), 0, "full → nothing more fits");
        let mut out = [0u8; 8];
        c.pull(&mut out);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn wraps_around_past_capacity() {
        // Drive far more bytes through than the physical buffer holds, in offset chunks, so
        // both the write and read runs straddle the wrap boundary many times.
        let (p, c) = spsc(8); // cap 8
        let mut next_w: u8 = 0;
        let mut next_r: u8 = 0;
        for _ in 0..1000 {
            let chunk = [next_w, next_w.wrapping_add(1), next_w.wrapping_add(2)];
            assert_eq!(p.try_push(&chunk), 3);
            next_w = next_w.wrapping_add(3);
            let mut out = [0u8; 3];
            c.pull(&mut out);
            assert_eq!(out, [next_r, next_r.wrapping_add(1), next_r.wrapping_add(2)]);
            next_r = next_r.wrapping_add(3);
        }
    }

    #[test]
    fn partial_push_reports_bytes_written() {
        let (p, c) = spsc(4); // cap 4
        p.try_push(&[1, 2, 3]);
        // Only 1 byte of room left → try_push takes exactly 1.
        assert_eq!(p.try_push(&[4, 5, 6]), 1, "writes only what fits");
        let mut out = [0u8; 4];
        c.pull(&mut out);
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn push_blocks_until_space_then_completes() {
        // A full ring makes the producer block until the consumer frees space — the
        // backpressure that paces the graph to the device. The producer is `Send`, so it
        // moves onto its own thread (exactly as the sink does); the consumer stays here.
        let (p, c) = spsc(4);
        p.push(&[1, 2, 3, 4]); // fills it (cap 4)
        let producer = thread::spawn(move || {
            p.push(&[5, 6, 7, 8]); // blocks until we drain below
            p // hand it back so the ring isn't closed by dropping it
        });
        // Give the producer time to reach the park.
        thread::sleep(Duration::from_millis(20));
        let mut a = [0u8; 4];
        c.pull(&mut a);
        assert_eq!(a, [1, 2, 3, 4]);
        let _p = producer.join().unwrap(); // timed out of its park, saw room, enqueued 5..8
        let mut b = [0u8; 4];
        c.pull(&mut b);
        assert_eq!(b, [5, 6, 7, 8]);
    }

    #[test]
    fn drain_returns_once_the_ring_empties_at_eos() {
        let (p, c) = spsc(1024);
        p.push(&[1, 2, 3, 4]);
        let drainer = thread::spawn(move || {
            p.drain(); // blocks until the ring empties
            p
        });
        thread::sleep(Duration::from_millis(20));
        let mut out = [0u8; 4];
        c.pull(&mut out); // empties → drain's next poll sees len()==0 and returns
        let _p = drainer.join().unwrap();
    }

    #[test]
    fn drain_on_empty_ring_returns_immediately() {
        let (p, _c) = spsc(64);
        p.drain(); // nothing buffered → returns without parking
    }

    #[test]
    fn close_unblocks_a_full_producer() {
        // Shutdown path: the PW consumer going away must free a producer parked on "full".
        let (p, c) = spsc(4);
        p.push(&[1, 2, 3, 4]); // fills it
        let producer = thread::spawn(move || {
            p.push(&[5, 6, 7, 8]); // blocks (full); close() must wake it
            p
        });
        thread::sleep(Duration::from_millis(20));
        drop(c); // consumer end goes away → ring closes, waking the producer (data dropped)
        let _p = producer.join().unwrap();
    }

    #[test]
    fn close_unblocks_a_drainer() {
        let (p, c) = spsc(64);
        p.push(&[1, 2, 3, 4]);
        let drainer = thread::spawn(move || {
            p.drain(); // blocks: ring non-empty and consumer never pulls
            p
        });
        thread::sleep(Duration::from_millis(20));
        drop(c); // consumer went away → drainer must stop waiting
        let _p = drainer.join().unwrap();
    }

    #[test]
    fn producer_and_consumer_move_a_pattern_intact_and_in_order() {
        // The real hand-off: a known byte pattern streamed producer→consumer across threads
        // through a small ring, reassembled and checked for exact order and no loss/dup.
        // The producer streams in irregular chunks (partial writes + wrap splits), signals
        // EOS via `drain`, and the consumer pulls until it has drained every byte at EOS.
        const N: usize = 1 << 18; // 256 KiB — many wraps through a 256-byte ring
        let (p, c) = spsc(256);

        let producer = thread::spawn(move || {
            let mut sent = 0usize;
            let sizes = [1usize, 7, 3, 64, 100, 5, 200, 33]; // irregular, some > cap
            let mut si = 0;
            while sent < N {
                let want = sizes[si % sizes.len()].min(N - sent);
                si += 1;
                // Deterministic pattern (value == byte index mod 256) so any reorder,
                // duplicate, or drop shows up as a mismatch at reassembly.
                let chunk: Vec<u8> = (0..want).map(|k| (sent + k) as u8).collect();
                p.push(&chunk);
                sent += want;
            }
            p.drain(); // EOS: block until the consumer has taken everything, then close
        });

        let consumer = thread::spawn(move || {
            // Reassemble the whole stream. `pull` zero-pads on underrun, so we can't trust
            // padding bytes; instead we track our own position and use `try_take` — pull a
            // scratch buffer, then only accept the bytes the ring actually held this round.
            let mut received: Vec<u8> = Vec::with_capacity(N);
            let mut scratch = [0u8; 37]; // not a divisor of ring size or any chunk
            while received.len() < N {
                let before = c.available();
                if before == 0 {
                    if c.is_closed() {
                        break; // producer done and ring drained
                    }
                    thread::yield_now();
                    continue;
                }
                let take = before.min(scratch.len());
                c.pull(&mut scratch[..take]); // exactly `take` real bytes (we saw them)
                received.extend_from_slice(&scratch[..take]);
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();
        assert_eq!(received.len(), N, "every byte arrived exactly once");
        // Verify order + integrity: byte i must equal i mod 256.
        assert!(
            received.iter().enumerate().all(|(i, &b)| b == i as u8),
            "bytes arrived in order, uncorrupted"
        );
    }
}
