//! Fan a path list across N threads: one [`Scanner`] each, one shared [`Pool`], one shared
//! cursor.
//!
//! No channels and no work queue. A library scan is embarrassingly parallel and its unit of
//! work is one array index, so the cheapest possible balancer — an [`AtomicUsize`] every
//! worker `fetch_add`s — beats anything with a send/receive on it, and self-balances: a
//! thread that lands on a 40 MB FLAC simply takes its next index later. `std::thread::scope`
//! lets the workers borrow the caller's `Vec` and callback directly, so nothing is cloned to
//! start a scan.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use profluens_core::io::ReactorFactory;

use crate::scanner::default_pool;
use crate::{ScanConfig, ScanOutcome, Scanner};

/// Scan `paths` across `threads` worker threads, calling `emit` from whichever worker
/// finished the file. Returns when every path has been reported.
///
/// `emit` must be `Sync` because it is shared by the workers; a caller that needs ordering
/// or single-threaded state locks (or channels) inside it.
pub fn scan_parallel(
    paths: Vec<PathBuf>,
    threads: usize,
    cfg: ScanConfig,
    emit: impl Fn(ScanOutcome<'_>) + Sync,
) {
    scan_parallel_with(paths, threads, cfg, None, emit);
}

/// [`scan_parallel`] with an explicit reactor factory — the same override
/// [`Scanner::with_reactor`] takes, applied to every worker. `None` gives each thread the
/// default backend.
pub fn scan_parallel_with(
    paths: Vec<PathBuf>,
    threads: usize,
    cfg: ScanConfig,
    factory: Option<ReactorFactory>,
    emit: impl Fn(ScanOutcome<'_>) + Sync,
) {
    let threads = threads.max(1);
    let pool = default_pool(cfg, threads);
    let cursor = AtomicUsize::new(0);
    let paths = &paths[..];
    let emit = &emit;
    let factory = factory.as_ref();

    std::thread::scope(|s| {
        for _ in 0..threads {
            let pool = pool.clone();
            let cursor = &cursor;
            s.spawn(move || {
                let mut scanner = match factory {
                    Some(f) => Scanner::with_reactor(cfg, pool, f),
                    None => Scanner::with_pool(cfg, pool),
                };
                scanner.scan(Cursor { paths, next: cursor }, emit);
            });
        }
    });
}

/// The work queue: `fetch_add` an index, hand back that path **borrowed**. Owning the path
/// would mean cloning it per file — one heap allocation per scanned file, in the one place
/// this crate is trying hardest not to allocate — so [`Scanner::scan`] takes
/// `IntoIterator<Item: AsRef<Path>>` and copies the bytes into the slot's reusable
/// `PathBuf` instead.
struct Cursor<'a> {
    paths: &'a [PathBuf],
    next: &'a AtomicUsize,
}

impl<'a> Iterator for Cursor<'a> {
    type Item = &'a Path;

    fn next(&mut self) -> Option<&'a Path> {
        // Relaxed is enough: the only invariant is that no two workers get the same index,
        // which `fetch_add` guarantees on its own — nothing else is published through it.
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        self.paths.get(i).map(PathBuf::as_path)
    }
}
