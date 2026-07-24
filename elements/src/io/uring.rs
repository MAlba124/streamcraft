//! A hand-rolled io_uring reactor backend (spec: IO: built for the io_uring era).
//!
//! Implements [`streamcraft_core::io::Reactor`] against Linux io_uring via raw
//! syscalls (`libc` only — no higher-level wrapper). Registered files are addressed
//! by fd; each element op becomes an `IORING_OP_READ`/`WRITE` SQE with `user_data`
//! set to the op id, and the owning `Buffer` is parked in an in-flight table until
//! its CQE arrives. Positioned (offset-carrying) ops make completion order
//! irrelevant. Feature-gated (`io-uring`) and Linux-only.
//!
//! Safety: this module manages mmap'd SQ/CQ rings and issues syscalls directly, so
//! it is necessarily `unsafe`. The ring head/tail indices are shared with the kernel
//! and accessed as atomics with acquire/release ordering.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::Error as IoError;
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use streamcraft_core::id::ElementId;
use streamcraft_core::io::{
    fadvise, Completion, IoResult, OpId, OpKind, Reactor, StreamHygiene, Submission,
    POSIX_FADV_SEQUENTIAL,
};

// --- ABI constants ---

const IORING_OFF_SQ_RING: u64 = 0;
const IORING_OFF_CQ_RING: u64 = 0x0800_0000;
const IORING_OFF_SQES: u64 = 0x1000_0000;

const IORING_ENTER_GETEVENTS: u32 = 1;

const IORING_OP_READ: u8 = 22;
const IORING_OP_WRITE: u8 = 23;

// --- ABI structs (repr(C), exact kernel layout) ---

#[repr(C)]
#[derive(Default)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

#[repr(C)]
struct IoUringSqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    op_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    __pad2: [u64; 2],
}

#[repr(C)]
struct IoUringCqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

// Layout guards: a wrong size here would corrupt the rings silently.
const _: () = assert!(core::mem::size_of::<IoUringSqe>() == 64);
const _: () = assert!(core::mem::size_of::<IoUringCqe>() == 16);

// --- syscall wrappers ---

fn io_uring_setup(entries: u32, params: &mut IoUringParams) -> std::io::Result<i32> {
    let ret = unsafe {
        libc::syscall(
            libc::SYS_io_uring_setup,
            entries as libc::c_long,
            params as *mut IoUringParams,
        )
    };
    if ret < 0 {
        Err(IoError::last_os_error())
    } else {
        Ok(ret as i32)
    }
}

fn io_uring_enter(fd: i32, to_submit: u32, min_complete: u32, flags: u32) -> std::io::Result<i32> {
    let ret = unsafe {
        libc::syscall(
            libc::SYS_io_uring_enter,
            fd as libc::c_long,
            to_submit as libc::c_long,
            min_complete as libc::c_long,
            flags as libc::c_long,
            ptr::null::<libc::c_void>(),
            0_usize as libc::c_long,
        )
    };
    if ret < 0 {
        Err(IoError::last_os_error())
    } else {
        Ok(ret as i32)
    }
}

fn mmap_ring(len: usize, fd: i32, offset: u64) -> std::io::Result<*mut u8> {
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            fd,
            offset as libc::off_t,
        )
    };
    if ptr == libc::MAP_FAILED {
        Err(IoError::last_os_error())
    } else {
        Ok(ptr as *mut u8)
    }
}

// --- reactor ---

struct Mmap {
    ptr: *mut u8,
    len: usize,
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

struct InflightOp {
    element: ElementId,
    op: OpId,
    user: u64,
    kind: OpKind,
    /// File + offset ride along so the completion can feed that file's
    /// [`StreamHygiene`] watermark.
    file: u32,
    offset: u64,
    buf: streamcraft_core::buffer::Buffer,
}

/// Contiguous-completion watermark: uring completes ops in any order, but cache
/// hygiene (`POSIX_FADV_DONTNEED` behind the position) is only safe below the
/// point where *every* prior byte's op has completed — dropping under an
/// in-flight write would discard the not-yet-dirty page for nothing, and
/// advising on a max-end mark could jump over holes. Completed ranges pool in a
/// min-heap and the watermark advances over whatever became contiguous.
#[derive(Default)]
struct Watermark {
    /// Everything below this has completed. Initialized to the first submitted
    /// offset (a stream need not start at 0 — a seek target).
    contig: Option<u64>,
    /// Completed `(start, end)` ranges above (and possibly at) `contig`.
    done: BinaryHeap<Reverse<(u64, u64)>>,
}

impl Watermark {
    /// Note the first submitted offset (fixes the origin).
    fn origin(&mut self, offset: u64) {
        if self.contig.is_none() {
            self.contig = Some(offset);
        }
    }

    /// Merge a completed range; returns the newly-contiguous `(from, to)` span
    /// if the watermark advanced.
    fn complete(&mut self, start: u64, end: u64) -> Option<(u64, u64)> {
        let mut contig = self.contig?;
        let from = contig;
        self.done.push(Reverse((start, end)));
        while let Some(&Reverse((s, e))) = self.done.peek() {
            if s > contig {
                break;
            }
            self.done.pop();
            contig = contig.max(e);
        }
        self.contig = Some(contig);
        (contig > from).then_some((from, contig))
    }
}

/// A registered file plus its streaming-hygiene state (see [`StreamHygiene`] —
/// same policy as `SyncReactor`, adapted to out-of-order completion).
struct UringFile {
    file: File,
    hygiene: StreamHygiene,
    /// Submission-order high-water marks: monotonicity must be judged on the
    /// *submitted* pattern (completion order is arbitrary here), so one backward
    /// submitted offset disables that side's hygiene for good.
    read_submit_end: u64,
    write_submit_end: u64,
    read_wm: Watermark,
    write_wm: Watermark,
}

pub struct IoUringReactor {
    ring_fd: i32,
    _sq_map: Mmap,
    _cq_map: Mmap,
    _sqes_map: Mmap,

    // SQ ring pointers (into _sq_map). The kernel-updated SQ head is not tracked:
    // the ring (64 entries) vastly exceeds our in-flight count, so it never fills.
    sq_ktail: *const AtomicU32,
    sq_mask: u32,
    sq_array: *mut u32,
    sqes: *mut IoUringSqe,
    sq_tail: u32, // local mirror of the tail we publish

    // CQ ring pointers (into _cq_map).
    cq_khead: *const AtomicU32,
    cq_ktail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const IoUringCqe,
    cq_head: u32, // local mirror of the head we consume

    files: HashMap<u32, UringFile>,
    pending: Vec<Submission>,
    /// Drain scratch for `submit_pending`: `pending` swaps in here so both vecs
    /// keep their capacity across passes (ZERO-COPY.md stage 4.2).
    scratch: Vec<Submission>,
    in_flight: HashMap<u64, InflightOp>,
}

impl IoUringReactor {
    /// Set up a fresh io_uring with `entries` (rounded up by the kernel) slots.
    pub fn new() -> std::io::Result<Self> {
        let entries: u32 = 64;
        let mut params = IoUringParams::default();
        let ring_fd = io_uring_setup(entries, &mut params)?;

        let sq_ring_len = (params.sq_off.array + params.sq_entries * 4) as usize;
        let cq_ring_len =
            (params.cq_off.cqes + params.cq_entries * core::mem::size_of::<IoUringCqe>() as u32) as usize;
        let sqes_len = params.sq_entries as usize * core::mem::size_of::<IoUringSqe>();

        let sq_map = Mmap { ptr: mmap_ring(sq_ring_len, ring_fd, IORING_OFF_SQ_RING)?, len: sq_ring_len };
        let cq_map = Mmap { ptr: mmap_ring(cq_ring_len, ring_fd, IORING_OFF_CQ_RING)?, len: cq_ring_len };
        let sqes_map = Mmap { ptr: mmap_ring(sqes_len, ring_fd, IORING_OFF_SQES)?, len: sqes_len };

        // SAFETY: the offsets come from the kernel and point inside the mappings.
        unsafe {
            let sq = sq_map.ptr;
            let cq = cq_map.ptr;
            let sq_ktail = sq.add(params.sq_off.tail as usize) as *const AtomicU32;
            let sq_mask = *(sq.add(params.sq_off.ring_mask as usize) as *const u32);
            let sq_array = sq.add(params.sq_off.array as usize) as *mut u32;
            let sq_tail = (*sq_ktail).load(Ordering::Acquire);

            let cq_khead = cq.add(params.cq_off.head as usize) as *const AtomicU32;
            let cq_ktail = cq.add(params.cq_off.tail as usize) as *const AtomicU32;
            let cq_mask = *(cq.add(params.cq_off.ring_mask as usize) as *const u32);
            let cqes = cq.add(params.cq_off.cqes as usize) as *const IoUringCqe;
            let cq_head = (*cq_khead).load(Ordering::Acquire);

            Ok(Self {
                ring_fd,
                sqes: sqes_map.ptr as *mut IoUringSqe,
                _sq_map: sq_map,
                _cq_map: cq_map,
                _sqes_map: sqes_map,
                sq_ktail,
                sq_mask,
                sq_array,
                sq_tail,
                cq_khead,
                cq_ktail,
                cq_mask,
                cqes,
                cq_head,
                files: HashMap::new(),
                pending: Vec::new(),
                scratch: Vec::new(),
                in_flight: HashMap::new(),
            })
        }
    }

    /// Fill SQEs for all pending ops, park their buffers, and submit. Any
    /// immediate errors (e.g. an op for an unregistered file) push into `out`.
    fn submit_pending(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        // Swap `pending` into the drain scratch so its capacity survives the pass.
        debug_assert!(self.scratch.is_empty());
        std::mem::swap(&mut self.pending, &mut self.scratch);
        let mut submitted: u32 = 0;

        for mut s in self.scratch.drain(..) {
            let fd = match self.files.get_mut(&s.file.0) {
                Some(rf) => {
                    // Hygiene bookkeeping happens at *submission*, where order still
                    // reflects the element's access pattern (completions reorder):
                    // fix the watermark origin and check monotonicity.
                    match s.kind {
                        OpKind::Read => {
                            rf.read_wm.origin(s.offset);
                            if s.offset < rf.read_submit_end {
                                rf.hygiene.disable_read();
                            }
                            rf.read_submit_end = s.offset + s.buf.memory.capacity() as u64;
                        }
                        OpKind::Write => {
                            rf.write_wm.origin(s.offset);
                            if s.offset < rf.write_submit_end {
                                rf.hygiene.disable_write();
                            }
                            rf.write_submit_end = s.offset + s.buf.memory.len() as u64;
                        }
                    }
                    rf.file.as_raw_fd()
                }
                None => {
                    out.push((
                        s.element,
                        Completion {
                            op: s.op,
                            user: s.user,
                            result: IoResult::Err(std::io::ErrorKind::NotFound),
                            buf: s.buf,
                        },
                    ));
                    continue;
                }
            };

            let (opcode, addr, len) = match s.kind {
                OpKind::Read => {
                    let slice = s.buf.memory.as_mut_full();
                    (IORING_OP_READ, slice.as_mut_ptr() as u64, slice.len() as u32)
                }
                OpKind::Write => {
                    let slice = s.buf.memory.data();
                    (IORING_OP_WRITE, slice.as_ptr() as u64, slice.len() as u32)
                }
            };

            let idx = (self.sq_tail & self.sq_mask) as usize;
            let sqe = IoUringSqe {
                opcode,
                flags: 0,
                ioprio: 0,
                fd,
                off: s.offset,
                addr,
                len,
                op_flags: 0,
                user_data: s.op.0,
                buf_index: 0,
                personality: 0,
                splice_fd_in: 0,
                __pad2: [0; 2],
            };
            // SAFETY: idx is masked into range; sqes/array live for the ring's life.
            unsafe {
                ptr::write(self.sqes.add(idx), sqe);
                ptr::write(self.sq_array.add(idx), idx as u32);
            }
            self.sq_tail = self.sq_tail.wrapping_add(1);
            submitted += 1;

            self.in_flight.insert(
                s.op.0,
                InflightOp {
                    element: s.element,
                    op: s.op,
                    user: s.user,
                    kind: s.kind,
                    file: s.file.0,
                    offset: s.offset,
                    buf: s.buf,
                },
            );
        }

        // Publish the new tail so the kernel sees the SQEs, then enter.
        unsafe {
            (*self.sq_ktail).store(self.sq_tail, Ordering::Release);
        }
        if submitted > 0 || !self.in_flight.is_empty() {
            // Block for at least one completion whenever ops are outstanding, so the
            // scheduler always makes progress rather than spinning.
            let min_complete = if self.in_flight.is_empty() { 0 } else { 1 };
            let flags = if min_complete > 0 { IORING_ENTER_GETEVENTS } else { 0 };
            // A spurious EINTR is fine — the next run_once retries.
            let _ = io_uring_enter(self.ring_fd, submitted, min_complete, flags);
        }
    }

    /// Reap all currently-available completions into `out`.
    fn reap(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        // SAFETY: ktail is the kernel-updated CQ tail; acquire orders the CQE reads.
        let ktail = unsafe { (*self.cq_ktail).load(Ordering::Acquire) };
        while self.cq_head != ktail {
            let idx = (self.cq_head & self.cq_mask) as usize;
            // SAFETY: idx masked into range; cqes lives for the ring's life.
            let (user_data, res) = unsafe {
                let cqe = &*self.cqes.add(idx);
                (cqe.user_data, cqe.res)
            };
            self.cq_head = self.cq_head.wrapping_add(1);

            if let Some(mut op) = self.in_flight.remove(&user_data) {
                let result = if res < 0 {
                    IoResult::Err(IoError::from_raw_os_error(-res).kind())
                } else {
                    let n = res as usize;
                    if let OpKind::Read = op.kind {
                        op.buf.memory.set_len(n);
                    }
                    // Feed the file's hygiene watermark: only spans that became
                    // *contiguously* complete reach StreamHygiene, in file order,
                    // so the DONTNEED/sync windows behave exactly as in the
                    // ordered SyncReactor despite arbitrary CQE order.
                    if n > 0 {
                        if let Some(rf) = self.files.get_mut(&op.file) {
                            let fd = rf.file.as_raw_fd();
                            let wm = match op.kind {
                                OpKind::Read => &mut rf.read_wm,
                                OpKind::Write => &mut rf.write_wm,
                            };
                            if let Some((from, to)) = wm.complete(op.offset, op.offset + n as u64) {
                                match op.kind {
                                    OpKind::Read => rf.hygiene.on_read(fd, from, (to - from) as usize),
                                    OpKind::Write => rf.hygiene.on_write(fd, from, (to - from) as usize),
                                }
                            }
                        }
                    }
                    IoResult::Ok(n)
                };
                out.push((op.element, Completion { op: op.op, user: op.user, result, buf: op.buf }));
            }
        }
        // Publish the consumed head back to the kernel.
        unsafe {
            (*self.cq_khead).store(self.cq_head, Ordering::Release);
        }
    }
}

impl Reactor for IoUringReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        // Registered files stream: double the readahead window up front
        // (`man 2 posix_fadvise`; best-effort — advice).
        let _ = fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_SEQUENTIAL);
        self.files.insert(
            element.0,
            UringFile {
                file,
                hygiene: StreamHygiene::new(),
                read_submit_end: 0,
                write_submit_end: 0,
                read_wm: Watermark::default(),
                write_wm: Watermark::default(),
            },
        );
    }

    fn submit(&mut self, subs: &mut Vec<Submission>) {
        // Drain, never take: the caller's outbox keeps its capacity for the next
        // pass; `pending`'s capacity is reactor-owned and survives submit_pending.
        self.pending.append(subs);
    }

    fn cancel(&mut self, op: OpId) {
        // Best-effort: drop it if still queued. In-flight cancellation (an
        // IORING_OP_ASYNC_CANCEL SQE) is a TODO — file copy never cancels.
        self.pending.retain(|s| s.op != op);
    }

    fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.in_flight.is_empty()
    }

    fn run_once(&mut self, out: &mut Vec<(ElementId, Completion)>) {
        // Caller-owned, reused across passes (ZERO-COPY.md stage 4.2); submit
        // errors and reaped completions share it.
        out.clear();
        self.submit_pending(out);
        self.reap(out);
    }
}

impl Drop for IoUringReactor {
    fn drop(&mut self) {
        // Mmaps are unmapped by their own Drop; just close the ring fd.
        unsafe {
            libc::close(self.ring_fd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Watermark;

    /// Out-of-order completions only advance the watermark once contiguous —
    /// the invariant the hygiene feed depends on.
    #[test]
    fn watermark_advances_only_when_contiguous() {
        let mut wm = Watermark::default();
        wm.origin(0);
        assert_eq!(wm.complete(10, 20), None); // hole at [0,10)
        assert_eq!(wm.complete(20, 30), None); // still blocked
        assert_eq!(wm.complete(0, 10), Some((0, 30))); // hole filled: all three merge
        assert_eq!(wm.complete(30, 40), Some((30, 40)));
    }

    /// The origin is the first submitted offset, not zero — a stream that starts
    /// mid-file (a seek target) must not wait for bytes nobody submitted.
    #[test]
    fn watermark_origin_is_first_offset() {
        let mut wm = Watermark::default();
        assert_eq!(wm.complete(0, 10), None); // no origin yet: ignored
        wm.origin(100);
        wm.origin(50); // later origins are no-ops
        assert_eq!(wm.complete(100, 150), Some((100, 150)));
    }
}
