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

use std::collections::HashMap;
use std::fs::File;
use std::io::Error as IoError;
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use streamcraft_core::id::ElementId;
use streamcraft_core::io::{Completion, IoResult, OpId, OpKind, Reactor, Submission};

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
    buf: streamcraft_core::buffer::Buffer,
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

    files: HashMap<u32, File>,
    pending: Vec<Submission>,
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
                in_flight: HashMap::new(),
            })
        }
    }

    /// Fill SQEs for all pending ops, park their buffers, and submit. Returns any
    /// immediate errors (e.g. an op for an unregistered file).
    fn submit_pending(&mut self) -> Vec<(ElementId, Completion)> {
        let mut errors = Vec::new();
        let subs = std::mem::take(&mut self.pending);
        let mut submitted: u32 = 0;

        for mut s in subs {
            let fd = match self.files.get(&s.file.0) {
                Some(f) => f.as_raw_fd(),
                None => {
                    errors.push((
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
                InflightOp { element: s.element, op: s.op, user: s.user, kind: s.kind, buf: s.buf },
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

        errors
    }

    /// Reap all currently-available completions.
    fn reap(&mut self) -> Vec<(ElementId, Completion)> {
        let mut out = Vec::new();
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
                    IoResult::Ok(n)
                };
                out.push((op.element, Completion { op: op.op, user: op.user, result, buf: op.buf }));
            }
        }
        // Publish the consumed head back to the kernel.
        unsafe {
            (*self.cq_khead).store(self.cq_head, Ordering::Release);
        }
        out
    }
}

impl Reactor for IoUringReactor {
    fn set_file(&mut self, element: ElementId, file: File) {
        self.files.insert(element.0, file);
    }

    fn submit(&mut self, subs: Vec<Submission>) {
        self.pending.extend(subs);
    }

    fn cancel(&mut self, op: OpId) {
        // Best-effort: drop it if still queued. In-flight cancellation (an
        // IORING_OP_ASYNC_CANCEL SQE) is a TODO — file copy never cancels.
        self.pending.retain(|s| s.op != op);
    }

    fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.in_flight.is_empty()
    }

    fn run_once(&mut self) -> Vec<(ElementId, Completion)> {
        let mut out = self.submit_pending();
        out.extend(self.reap());
        out
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
