//! Where a [`Player`](crate::Player) gets its bytes: a complete file on disk, or one a
//! separate writer is still appending to (spec: `elements/src/io/growingfilesrc.rs` — "the
//! progressive-download case: an app downloads a podcast episode to disk while this pipeline
//! plays it back from the same path").
//!
//! # Why this is not just a path
//!
//! Everything `Player::open` does before the pipeline exists — typefind, the per-container head
//! walk, the seek index — is **blocking application-side IO against the file directly**
//! (`crate::head`). On a complete file that is unremarkable. On a *growing* file every one of
//! those reads is a race: the prefix may be shorter than typefind wants, the container head may
//! not have arrived, and — the sharp one — a read of "the tail" returns the bytes at the current
//! download frontier, which is emphatically **not** the tail of the finished file. An Ogg
//! duration derived from that tail is the position the download has reached, reported as the
//! length of the episode.
//!
//! So a growing source carries three things a path cannot: the [`FrontierHandle`] that says how
//! many leading bytes are safely readable, the caller's `total_hint` (a podcast's HTTP
//! `Content-Length`), and the [`GrowingFileSrc`] element itself.
//!
//! # The open policy, stated plainly
//!
//! 1. **Open succeeds as soon as the typefind prefix is readable.** That is
//!    [`PREFIX_LEN`](crate::probe::PREFIX_LEN) — 64 bytes. [`SourceSpec::wait_readable`] polls
//!    the frontier for it (and for the container head behind it) up to
//!    [`OPEN_TIMEOUT`]; a caller who has already buffered the first chunk never waits at all.
//! 2. **No read ever crosses the frontier.** Every head read is clamped to
//!    [`SourceSpec::readable`], so a half-written append can never be parsed. The clamp is why
//!    the growing path uses [`crate::head`]'s `*_from` byte-slice variants rather than its
//!    path-reading ones: the slice is the bounded window, and the walk cannot wander past it.
//! 3. **A container head that has not arrived fails the open, loudly.** You cannot demux a
//!    Matroska file whose `Tracks` element is still downloading, and pretending otherwise would
//!    surface as a track-less player rather than as an error. The wait in (1) covers the
//!    ordinary case of a download that is simply a second behind.
//! 4. **Duration and index prefer the hint over the file.** The length the seek index is built
//!    against is [`SourceSpec::index_len`]: `total_hint` if the caller supplied one, else the
//!    writer's declared total once finished, else the current frontier. A proportional index
//!    built against a partial length maps "half way through the episode" to half of what has
//!    *downloaded so far* — an answer that silently changes as bytes arrive. The hint is the
//!    only honest denominator.
//! 5. **Tail reads happen only when the whole file is readable** ([`SourceSpec::is_complete`]:
//!    the writer finished, or the frontier has reached `total_hint`). Until then, duration comes
//!    from the *head* — an MP3's Xing frame count, a FLAC's STREAMINFO, an MP4's `mvhd` — which
//!    is correct from the first kilobyte and does not change as the download proceeds. A format
//!    whose only duration source is its tail (Ogg's last granule) honestly reports none until
//!    the download completes, at which point a fresh open has it.
//!
//! # Shape note (a deliberate deviation)
//!
//! The design sketch had `SourceSpec` as an enum carrying a caller-supplied frontier handle.
//! That shape cannot be built: [`GrowingFileSrc::new`] *mints* the handle, so a caller cannot
//! have one before the element exists — and the caller genuinely needs it first, because the
//! download must already be running for step (1) to succeed. Hence a struct built by
//! [`SourceSpec::growing`], which constructs the element and hands the handle back:
//!
//! ```ignore
//! let (source, frontier) = SourceSpec::growing("/tmp/ep.mp3", Some(content_length));
//! std::thread::spawn(move || { /* download; frontier.advance(n); … frontier.finish(len) */ });
//! let player = Player::open_canonical(source, ChainSpec::device(), SinkChoice::Drop)?;
//! ```
//!
//! No dependency cycle is introduced: `pf-play` already depends on `profluens-elements`, which
//! is where `growingfilesrc` lives.

use std::time::{Duration, Instant};

use profluens_core::element::Element;
use profluens_elements::io::{FileSrc, FrontierHandle, GrowingFileSrc};

/// How long [`SourceSpec::wait_readable`] will wait for the download frontier to reach the
/// bytes an open needs, before giving up and reporting what is actually there.
///
/// Ten seconds is chosen against the failure it guards: a caller who starts the download and
/// opens the player in the same breath, where "the first 64 KiB have landed" is a
/// network-latency question, not a bandwidth one. Any real progressive download clears it by
/// orders of magnitude; a download that has genuinely stalled should fail the open rather than
/// hang a UI thread forever. The wait also ends early the moment the frontier reaches the
/// target or the writer calls `finish()`.
///
/// (A writer that calls `abort()` is not observable through [`FrontierHandle`] — only the
/// element sees it — so an aborted download waits out the full timeout here and then fails on
/// the bytes it has. That is the honest bound, not a hang.)
pub const OPEN_TIMEOUT: Duration = Duration::from_secs(10);

/// How often [`SourceSpec::wait_readable`] re-reads the frontier while waiting. 5 ms is far
/// below human perception and costs two relaxed atomic loads per poll.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// How much of a growing file the container head walk is allowed to read. Matches
/// [`crate::head::mkv_header_prefix`]'s own 8 MiB bound, which is the largest head budget any
/// supported container asks for.
pub const HEAD_BUDGET: usize = 8 * 1024 * 1024;

/// The byte source a [`Player`](crate::Player) is built on.
///
/// Construct with [`SourceSpec::path`] (a complete file — exactly what `Player::open` uses) or
/// [`SourceSpec::growing`] (a file a writer is still appending to).
pub struct SourceSpec {
    /// The file both the head reads and the source element read. Always a real path: the head
    /// walk is app-side IO against the same bytes the element streams.
    path: String,
    /// The writer's watermark, for a growing source. `None` for a complete file.
    frontier: Option<FrontierHandle>,
    /// The final size, when the caller knows it out of band (a podcast's `Content-Length`).
    total_hint: Option<u64>,
    /// The source element, taken once when the graph is built.
    element: Option<Box<dyn Element>>,
}

impl SourceSpec {
    /// A complete file on disk — byte-for-byte the source `Player::open` has always used.
    // COLD: one boxed source element per player, at build time.
    #[allow(clippy::disallowed_methods)]
    pub fn path(path: &str) -> Self {
        SourceSpec {
            path: path.into(),
            frontier: None,
            total_hint: None,
            element: Some(Box::new(FileSrc::new(path))),
        }
    }

    /// A file a separate writer is still appending to, together with the [`FrontierHandle`] the
    /// writer publishes progress through.
    ///
    /// Start the download **before** opening the player: the open waits for the typefind prefix
    /// (and, for a container, its head) to become readable, so a frontier that never moves is a
    /// [`OPEN_TIMEOUT`]-long wait followed by an honest failure.
    ///
    /// `total_hint` is the final file size when it is known out of band. Supply it whenever you
    /// can — see the module docs, policy (4): it is the denominator the seek index and the
    /// proportional duration are built against, and the alternative (the current frontier)
    /// silently changes meaning as bytes arrive.
    // COLD: one boxed source element per player, at build time.
    #[allow(clippy::disallowed_methods)]
    pub fn growing(path: &str, total_hint: Option<u64>) -> (Self, FrontierHandle) {
        let (el, frontier) = GrowingFileSrc::new(path);
        let spec = SourceSpec {
            path: path.into(),
            frontier: Some(frontier.clone()),
            total_hint,
            element: Some(Box::new(el)),
        };
        (spec, frontier)
    }

    /// The file path — what the head/probe/seek reads open.
    pub fn path_str(&self) -> &str {
        &self.path
    }

    /// The caller's declared final size, if any.
    pub fn total_hint(&self) -> Option<u64> {
        self.total_hint
    }

    /// Whether this is a growing (progressively downloaded) source.
    pub fn is_growing(&self) -> bool {
        self.frontier.is_some()
    }

    /// The writer's frontier, for a growing source.
    pub fn frontier(&self) -> Option<&FrontierHandle> {
        self.frontier.as_ref()
    }

    /// How many leading bytes are safe to read **right now**. `None` means "no bound" — a
    /// complete file, where the file's own length is the only limit.
    pub fn readable(&self) -> Option<u64> {
        self.frontier.as_ref().map(|f| f.downloaded())
    }

    /// Whether every byte of the file is readable: a complete file always; a growing one once
    /// the writer has called `finish()`, or once the frontier has reached the caller's
    /// `total_hint`.
    ///
    /// This is the gate on tail reads (module docs, policy 5). The `total_hint` arm matters:
    /// a download can deliver the last byte well before the writer gets round to announcing it,
    /// and an episode whose bytes are all present has a real, readable tail.
    pub fn is_complete(&self) -> bool {
        let Some(f) = &self.frontier else { return true };
        if f.is_finished() {
            return true;
        }
        matches!(self.total_hint, Some(t) if f.downloaded() >= t)
    }

    /// The length the seek index and any proportional duration are built against — the module
    /// docs' policy (4): the caller's hint first, then the writer's declared total, then the
    /// frontier, and for a complete file the file's own size.
    pub fn index_len(&self, file_len_on_disk: u64) -> u64 {
        let Some(f) = &self.frontier else { return file_len_on_disk };
        if let Some(t) = self.total_hint {
            return t;
        }
        // `FrontierHandle` exposes no `total()`; once finished the watermark *is* the total
        // (`finish` advances it), so the frontier is the answer in both remaining cases.
        f.downloaded()
    }

    /// Block until at least `bytes` leading bytes are readable, or [`OPEN_TIMEOUT`] elapses, or
    /// the writer finishes. Returns how many bytes are readable when it gives up waiting.
    ///
    /// A complete file returns immediately — there is nothing to wait for.
    pub fn wait_readable(&self, bytes: u64) -> u64 {
        let Some(f) = &self.frontier else { return u64::MAX };
        let deadline = Instant::now() + OPEN_TIMEOUT;
        loop {
            let have = f.downloaded();
            // `finish()` is the writer's promise that no more is coming, so waiting past it is
            // waiting for something that will never arrive.
            if have >= bytes || f.is_finished() || Instant::now() >= deadline {
                return f.downloaded();
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Read the first `want` bytes, clamped to what the frontier allows (waiting for them
    /// first, up to [`OPEN_TIMEOUT`]). The returned buffer may be shorter than `want` — because
    /// the file is shorter, or because that is all the writer has published.
    ///
    /// This is the *only* read the growing open path performs, and it is why nothing downstream
    /// can observe a torn append: every parse works on this bounded window.
    pub fn read_prefix(&self, want: usize) -> std::io::Result<Vec<u8>> {
        let bounded = match self.frontier {
            None => want,
            Some(_) => {
                let have = self.wait_readable(want as u64);
                usize::try_from(have.min(want as u64)).unwrap_or(want)
            }
        };
        crate::head::read_prefix(&self.path, bounded)
    }

    /// The head window a container walk gets on this source: everything readable **right now**,
    /// capped at [`HEAD_BUDGET`]. Never waits.
    ///
    /// Deliberately not "wait for [`HEAD_BUDGET`] bytes": that budget is an upper bound, not a
    /// requirement — a three-megabyte episode would never reach it, and waiting for it would
    /// turn every small growing file into a full [`OPEN_TIMEOUT`] stall. The caller instead
    /// walks what it has and, if the walk says "not yet", waits for *growth*
    /// ([`wait_for_growth`](Self::wait_for_growth)) and walks again. The head is then read
    /// exactly as many times as the download makes necessary, and no more.
    pub fn read_head_window(&self) -> std::io::Result<Vec<u8>> {
        let want = match self.readable() {
            None => HEAD_BUDGET,
            Some(have) => usize::try_from(have.min(HEAD_BUDGET as u64)).unwrap_or(HEAD_BUDGET),
        };
        crate::head::read_prefix(&self.path, want)
    }

    /// Wait (bounded by `deadline`) for the frontier to move past `have`. Returns `true` when
    /// it did — i.e. "there are new bytes, try the parse again" — and `false` when the writer
    /// has finished, the deadline passed, or this is not a growing source (in all three cases
    /// there is nothing more to wait for).
    pub fn wait_for_growth(&self, have: u64, deadline: Instant) -> bool {
        let Some(f) = &self.frontier else { return false };
        loop {
            if f.downloaded() > have {
                return true;
            }
            // `finish()` is the writer's promise that no more is coming.
            if f.is_finished() || Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// The deadline a bounded open-time wait should use: now + [`OPEN_TIMEOUT`].
    pub fn open_deadline() -> Instant {
        Instant::now() + OPEN_TIMEOUT
    }

    /// Take the source element for the graph. Called exactly once, by the player's build path;
    /// a second call is a controller bug and says so.
    pub(crate) fn take_element(&mut self) -> Result<Box<dyn Element>, String> {
        // COLD: one message on a build-path bug.
        #[allow(clippy::disallowed_methods)]
        self.element.take().ok_or_else(|| {
            format!("source '{}': the source element was already taken", self.path)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_path_has_no_frontier_and_is_always_complete() {
        let s = SourceSpec::path("/nonexistent/file.flac");
        assert!(!s.is_growing());
        assert!(s.frontier().is_none());
        assert_eq!(s.readable(), None, "a complete file has no readable bound");
        assert!(s.is_complete());
        assert_eq!(s.index_len(1234), 1234, "a complete file indexes against its own size");
        assert_eq!(s.wait_readable(u64::MAX), u64::MAX, "a complete file never waits");
    }

    #[test]
    fn a_growing_source_tracks_its_frontier() {
        let (s, f) = SourceSpec::growing("/nonexistent/file.mp3", None);
        assert!(s.is_growing());
        assert_eq!(s.readable(), Some(0));
        assert!(!s.is_complete(), "nothing downloaded, nothing declared");
        f.advance(4096);
        assert_eq!(s.readable(), Some(4096));
        assert_eq!(s.index_len(0), 4096, "with no hint, the frontier is the best denominator");
        f.finish(9000);
        assert!(s.is_complete(), "finish() makes the whole file readable");
        assert_eq!(s.index_len(0), 9000);
    }

    #[test]
    fn the_total_hint_wins_over_the_frontier_and_completes_early() {
        let (s, f) = SourceSpec::growing("/nonexistent/file.mp3", Some(10_000));
        f.advance(2_500);
        assert_eq!(
            s.index_len(2_500),
            10_000,
            "the index must be proportional to the FINAL length, not the downloaded prefix"
        );
        assert!(!s.is_complete());
        // Every byte has arrived, even though the writer has not said so yet.
        f.advance(10_000);
        assert!(s.is_complete(), "the frontier reaching the hint means the tail is readable");
    }

    #[test]
    fn waiting_ends_the_moment_the_frontier_covers_the_request() {
        let (s, f) = SourceSpec::growing("/nonexistent/file.mp3", None);
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            f.advance(64);
        });
        let started = Instant::now();
        let have = s.wait_readable(64);
        let waited = started.elapsed();
        writer.join().expect("writer thread");
        assert!(have >= 64, "waited but saw only {have} bytes");
        assert!(waited < OPEN_TIMEOUT, "returned on the frontier, not on the timeout");
    }

    #[test]
    fn the_source_element_is_taken_exactly_once() {
        let mut s = SourceSpec::path("/nonexistent/file.flac");
        assert!(s.take_element().is_ok());
        assert!(s.take_element().is_err(), "a second take is a controller bug");
    }
}
