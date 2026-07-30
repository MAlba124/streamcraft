//! Streaming page-cache hygiene (spec: IO): the raw `fadvise`/`sync_file_range`
//! shims propagate errors and succeed on real files, and `SyncReactor`'s
//! per-window hygiene (SEQUENTIAL on registration, DONTNEED behind reads, the
//! sync+drop two-window dance behind writes) never perturbs the data itself.

use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

use profluens_core::buffer::{Buffer, BufferFlags};
use profluens_core::id::{ElementId, FormatId};
use profluens_core::io::{
    fadvise, sync_file_range, FileHandle, IoResult, OpId, OpKind, Reactor, StreamHygiene,
    Submission, SyncReactor, POSIX_FADV_DONTNEED, POSIX_FADV_SEQUENTIAL, SYNC_FILE_RANGE_WRITE,
};
use profluens_core::memory::Pool;
use profluens_core::time::Timestamp;

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pf_io_hygiene_{tag}_{}", std::process::id()))
}

/// An invalid fd must surface as an `Err` (EBADF on Linux/x86_64) — never a
/// panic, never silent success. On targets where the shims are cfg'd to no-ops
/// they report success by design.
#[test]
fn shims_propagate_errors_without_panicking() {
    let fa = fadvise(-1, 0, 0, POSIX_FADV_SEQUENTIAL);
    let sfr = sync_file_range(-1, 0, 0, SYNC_FILE_RANGE_WRITE);
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        assert_eq!(fa.unwrap_err().raw_os_error(), Some(9), "fadvise(-1) => EBADF");
        assert_eq!(sfr.unwrap_err().raw_os_error(), Some(9), "sync_file_range(-1) => EBADF");
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        assert!(fa.is_ok() && sfr.is_ok(), "no-op shims report success");
    }
}

/// The full hint sequence succeeds on a real file: SEQUENTIAL at open, an async
/// writeback kick, DONTNEED behind the position.
#[test]
fn shims_succeed_on_a_real_file() {
    let path = tmp_path("shims");
    let mut f = File::create(&path).expect("create temp file");
    f.write_all(&vec![7u8; 256 * 1024]).expect("write");
    let fd = f.as_raw_fd();
    fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL).expect("SEQUENTIAL");
    sync_file_range(fd, 0, 128 * 1024, SYNC_FILE_RANGE_WRITE).expect("writeback kick");
    fadvise(fd, 0, 128 * 1024, POSIX_FADV_DONTNEED).expect("DONTNEED behind");
    drop(f);
    std::fs::remove_file(&path).ok();
}

/// `StreamHygiene` is best-effort bookkeeping: shim failures (bad fd) are
/// swallowed, a backward op flips the side off, and huge offsets don't trip
/// arithmetic — none of it may panic.
#[test]
fn hygiene_tracker_survives_bad_fds_and_backward_seeks() {
    let mut h = StreamHygiene::with_window(1024);
    h.on_read(-1, 0, 2048); // window trips → fadvise on a bad fd → ignored
    h.on_read(-1, 1024, 100); // backward (1024 < end 2048): read hygiene off for good
    h.on_read(-1, 1 << 40, 4096); // ignored, still no panic

    let mut w = StreamHygiene::with_window(1024);
    w.on_write(-1, 0, 4096); // trips sync + (no-op) drop on a bad fd → ignored
    w.on_write(-1, 0, 4096); // overwrite: write hygiene off for good
    w.on_write(-1, u64::MAX - 4096, 4096); // ignored, no overflow
}

fn buf_from(pool: &Pool, data: &[u8]) -> Buffer {
    let mut m = pool.acquire();
    m.as_mut_full()[..data.len()].copy_from_slice(data);
    m.set_len(data.len());
    Buffer {
        memory: m,
        pts: Timestamp::ZERO,
        dts: Timestamp::ZERO,
        duration: Timestamp::ZERO,
        flags: BufferFlags::empty(),
        format: FormatId(0),
        sync: None,
    }
}

/// End-to-end through `SyncReactor` with a hygiene window far smaller than the
/// data, so the sync/drop branches fire many times mid-run: every completion is
/// `Ok`, and the bytes on disk (and read back through the reactor) are exactly
/// what was submitted — hygiene is advice, never allowed to perturb data.
#[test]
fn sync_reactor_hygiene_preserves_data() {
    const SLOT: usize = 16 * 1024;
    const SLOTS: usize = 24; // 384 KiB total, 12 windows deep

    let path = tmp_path("reactor");
    let pool = Pool::new(SLOT);
    let mut r = SyncReactor::new();
    r.set_hygiene_window(32 * 1024); // trip the windows every other op

    // Writer: element 1 owns the file, sequential 16 KiB writes.
    let writer = ElementId(1);
    r.set_file(writer, File::create(&path).expect("create"));
    let expect: Vec<u8> = (0..SLOT * SLOTS).map(|i| (i % 251) as u8).collect();
    for (i, chunk) in expect.chunks(SLOT).enumerate() {
        r.submit(&mut vec![Submission {
            op: OpId(i as u64),
            element: writer,
            kind: OpKind::Write,
            file: FileHandle(writer.0),
            offset: (i * SLOT) as u64,
            buf: buf_from(&pool, chunk),
            user: i as u64,
        }]);
        // Interleave run_once with submission like the scheduler does.
        if i % 3 == 2 {
            for (_, c) in {
        let mut completions = Vec::new();
        r.run_once(&mut completions);
        completions
    } {
                assert!(matches!(c.result, IoResult::Ok(SLOT)), "write completion ok");
            }
        }
    }
    for (_, c) in {
        let mut completions = Vec::new();
        r.run_once(&mut completions);
        completions
    } {
        assert!(matches!(c.result, IoResult::Ok(SLOT)), "write completion ok");
    }
    assert!(r.is_idle());

    // Bytes on disk are exactly what was submitted.
    let mut on_disk = Vec::new();
    File::open(&path).expect("reopen").read_to_end(&mut on_disk).expect("read back");
    assert_eq!(on_disk, expect, "hygiene must not perturb written data");

    // Reader: element 2 streams the file back through the reactor (DONTNEED
    // fires behind the position); contents must match.
    let reader = ElementId(2);
    r.set_file(reader, File::open(&path).expect("open for read"));
    let mut got = vec![0u8; 0];
    for i in 0..SLOTS {
        r.submit(&mut vec![Submission {
            op: OpId(1000 + i as u64),
            element: reader,
            kind: OpKind::Read,
            file: FileHandle(reader.0),
            offset: (i * SLOT) as u64,
            buf: buf_from(&pool, &[]),
            user: i as u64,
        }]);
    }
    let mut completions = Vec::new();
    r.run_once(&mut completions);
    completions.sort_by_key(|(_, c)| c.user);
    for (_, c) in completions {
        assert!(matches!(c.result, IoResult::Ok(SLOT)), "read completion ok");
        got.extend_from_slice(c.buf.memory.data());
    }
    assert_eq!(got, expect, "hygiene must not perturb read data");

    std::fs::remove_file(&path).ok();
}
