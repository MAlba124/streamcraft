//! The one audited `unsafe` module (crate `#![deny(unsafe_code)]`, this module locally
//! `#![allow(unsafe_code)]`, mirroring `streamcraft-core`'s `memory`/`ring`). Every FFI
//! call here is a thin, checked wrapper over exactly the four syscalls the shm swapchain
//! and Wayland fd-passing need — nothing more:
//!
//! * [`memfd`] — `memfd_create` + `ftruncate`: an anonymous, sealed-capable shared file to
//!   back the pool.
//! * [`Mmap`] — `mmap`/`munmap`: a `MAP_SHARED` mapping of that file the compositor and we
//!   both see, so a CPU pixel write is visible to the server with no copy.
//! * [`send_with_fd`] — `sendmsg` with an `SCM_RIGHTS` control message: hand the pool fd to
//!   the compositor alongside the `wl_shm.create_pool` request bytes.
//!
//! Each `unsafe` block documents the invariant it upholds (valid pointers, correct
//! lengths, ownership of the fd). No raw pointer escapes this module: callers see `RawFd`,
//! `&mut [u8]`, and `io::Result`.

#![allow(unsafe_code)]

use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;

/// An owned anonymous shared-memory file (from `memfd_create`), sized with `ftruncate`.
/// Backs a `wl_shm` pool: we `mmap` it, write pixels, and pass its fd to the compositor,
/// which maps the same pages. Closed on drop.
pub struct Memfd {
    fd: RawFd,
    size: usize,
}

impl Memfd {
    /// Create an anonymous shared file of `size` bytes, `CLOEXEC` so it does not leak
    /// across an exec. The compositor receives a *dup* of this fd via `SCM_RIGHTS`; our
    /// copy stays owned here.
    pub fn new(size: usize) -> io::Result<Memfd> {
        // SAFETY: a fixed NUL-terminated name literal; `memfd_create` reads it as a
        // C string and returns a new fd or -1. We own the returned fd.
        let name = c"streamcraft-wl-shm";
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut mf = Memfd { fd, size: 0 };
        mf.resize(size)?;
        Ok(mf)
    }

    /// Grow (or shrink) the backing file to `size` bytes and record the new size.
    pub fn resize(&mut self, size: usize) -> io::Result<()> {
        // SAFETY: `self.fd` is an fd we own and keep open for the call; `size` fits an
        // `off_t` for any pool we allocate (frames are megabytes, not exabytes).
        let rc = unsafe { libc::ftruncate(self.fd, size as libc::off_t) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        self.size = size;
        Ok(())
    }

    /// The current file size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for Memfd {
    fn drop(&mut self) {
        // SAFETY: `self.fd` was returned by `memfd_create` and is owned solely by this
        // `Memfd`; no live `Mmap` borrows it (mappings survive the fd close, and are
        // unmapped independently). Closing an owned fd once is sound.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// A `MAP_SHARED`, read/write mapping of a [`Memfd`]. Writing here is immediately visible
/// to the compositor, which maps the same file — the zero-copy shm hand-off. Unmapped on
/// drop; the mapping outlives the fd, so the [`Memfd`] may be closed first.
pub struct Mmap {
    ptr: *mut u8,
    len: usize,
}

// The mapping is a plain shared byte region we own exclusively (one `Mmap` per pool); it
// is safe to move to and access from another thread. We never alias `ptr` outside the
// `&mut [u8]`/`&[u8]` views, which the borrow checker then serialises.
unsafe impl Send for Mmap {}

impl Mmap {
    /// Map `len` bytes of `fd` shared read/write at offset 0.
    pub fn new(fd: RawFd, len: usize) -> io::Result<Mmap> {
        // SAFETY: `fd` refers to a file at least `len` bytes long (the caller `ftruncate`d
        // it); `NULL` addr lets the kernel choose; `MAP_SHARED` on a valid fd returns a
        // mapping of exactly `len` bytes or `MAP_FAILED`. We own the returned region.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Mmap {
            ptr: ptr as *mut u8,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The mapped bytes, mutably. Pixel writes land here.
    pub fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr`/`len` come from a successful `mmap` and are unchanged; the region
        // is `PROT_READ|PROT_WRITE` and owned exclusively by this `Mmap`. `&mut self`
        // guarantees no other reference aliases it for the borrow's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The mapped bytes, immutably.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as above; `&self` permits shared reads with no concurrent `&mut`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are exactly what `mmap` returned and have not been unmapped;
        // no slice view outlives this `Mmap` (all views borrow `self`). One `munmap` of an
        // owned mapping is sound.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// Send `bytes` on `stream` with `fd` attached as a single-fd `SCM_RIGHTS` control message
/// — the Wayland fd-passing primitive (used for `wl_shm.create_pool`). The compositor
/// receives a duplicate of `fd`; ours stays open. At least one data byte must accompany the
/// control message, which is always true here (the request header is 8 bytes).
///
/// Returns an error if the whole payload could not be sent in one `sendmsg` (Wayland
/// messages are small — well under a socket buffer — so a short send would itself be a
/// protocol-level anomaly we surface rather than silently truncate).
pub fn send_with_fd(stream: &UnixStream, bytes: &[u8], fd: RawFd) -> io::Result<()> {
    if bytes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SCM_RIGHTS send needs at least one data byte",
        ));
    }

    // One iovec pointing at the request bytes.
    let mut iov = libc::iovec {
        iov_base: bytes.as_ptr() as *mut libc::c_void,
        iov_len: bytes.len(),
    };

    // Control buffer sized for exactly one RawFd, aligned for cmsghdr via a padded array.
    // CMSG_SPACE includes the header + alignment padding.
    const FD_BYTES: usize = std::mem::size_of::<RawFd>();
    let cmsg_space = unsafe { libc::CMSG_SPACE(FD_BYTES as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];

    // Zero-initialise the msghdr, then fill the fields we need by name (field order and
    // any padding are platform-specific — zeroing + named writes is portable).
    // SAFETY: `msghdr` is a plain-old-data struct with no invalid bit patterns; an
    // all-zero value is a valid (empty) header we then populate.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg_space;

    // Fill the single control message header and copy the fd into its data area.
    // SAFETY: `CMSG_FIRSTHDR` returns a pointer into `cmsg_buf`, which is `cmsg_space`
    // bytes and correctly sized/aligned for one cmsghdr + one fd. We write only within
    // that region: the header fields, then `FD_BYTES` at `CMSG_DATA`.
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "CMSG_FIRSTHDR returned null",
            ));
        }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(FD_BYTES as u32) as usize;
        std::ptr::copy_nonoverlapping(
            &fd as *const RawFd as *const u8,
            libc::CMSG_DATA(cmsg),
            FD_BYTES,
        );
    }

    // SAFETY: `msg` points at live, correctly-sized iov/control buffers that outlive the
    // call; `stream`'s fd is a connected `SOCK_STREAM` we borrow for the duration. `sendmsg`
    // reads through the pointers and returns the byte count or -1.
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        return Err(io::Error::last_os_error());
    }
    if sent as usize != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "short sendmsg on the Wayland socket",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memfd_maps_and_round_trips_bytes() {
        // A memfd is available on any modern Linux; map it, write, read back.
        let mf = Memfd::new(4096).expect("memfd_create");
        assert_eq!(mf.size(), 4096);
        let mut map = Mmap::new(mf.as_raw_fd(), 4096).expect("mmap");
        assert_eq!(map.len(), 4096);
        map.as_mut()[0] = 0xAB;
        map.as_mut()[4095] = 0xCD;
        assert_eq!(map.as_slice()[0], 0xAB);
        assert_eq!(map.as_slice()[4095], 0xCD);
    }

    #[test]
    fn memfd_resize_grows_the_file() {
        let mut mf = Memfd::new(64).expect("memfd_create");
        mf.resize(8192).expect("ftruncate grow");
        assert_eq!(mf.size(), 8192);
        let map = Mmap::new(mf.as_raw_fd(), 8192).expect("mmap grown");
        assert_eq!(map.len(), 8192);
    }

    #[test]
    fn send_with_fd_delivers_the_fd_and_bytes_over_a_socketpair() {
        // Prove the SCM_RIGHTS path end to end without a compositor: pass a memfd across a
        // socketpair, then read the received fd and confirm it maps the same content.
        let (a, b) = UnixStream::pair().expect("socketpair");
        let mf = Memfd::new(4096).expect("memfd");
        {
            let mut map = Mmap::new(mf.as_raw_fd(), 4096).expect("mmap");
            map.as_mut()[..4].copy_from_slice(&[1, 2, 3, 4]);
        }

        let payload = [0xDEu8, 0xAD, 0xBE, 0xEF];
        send_with_fd(&a, &payload, mf.as_raw_fd()).expect("send_with_fd");

        // Receive: read the 4 data bytes and the ancillary fd in one recvmsg.
        let (received_fd, data) = recv_one_fd(&b, payload.len()).expect("recv fd");
        assert_eq!(data, payload, "data bytes arrive alongside the fd");
        let map = Mmap::new(received_fd, 4096).expect("mmap received fd");
        assert_eq!(&map.as_slice()[..4], &[1, 2, 3, 4], "same shared pages");
        drop(map);
        // SAFETY: `received_fd` is owned by this test (dup'd into us by SCM_RIGHTS).
        unsafe {
            libc::close(received_fd);
        }
    }

    /// Test helper: receive exactly one fd plus the `n` data bytes via `recvmsg`. Lives in
    /// the test module so the production surface stays send-only.
    fn recv_one_fd(stream: &UnixStream, n: usize) -> io::Result<(RawFd, Vec<u8>)> {
        const FD_BYTES: usize = std::mem::size_of::<RawFd>();
        let mut data = vec![0u8; n.max(1)];
        let mut iov = libc::iovec {
            iov_base: data.as_mut_ptr() as *mut libc::c_void,
            iov_len: data.len(),
        };
        let cmsg_space = unsafe { libc::CMSG_SPACE(FD_BYTES as u32) } as usize;
        let mut cmsg_buf = vec![0u8; cmsg_space];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_space;

        let rc = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, 0) };
        if rc < 0 {
            return Err(io::Error::last_os_error());
        }
        data.truncate(rc as usize);
        let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
        if cmsg.is_null() {
            return Err(io::Error::new(io::ErrorKind::Other, "no control message"));
        }
        let mut fd: RawFd = -1;
        unsafe {
            std::ptr::copy_nonoverlapping(
                libc::CMSG_DATA(cmsg),
                &mut fd as *mut RawFd as *mut u8,
                FD_BYTES,
            );
        }
        Ok((fd, data))
    }
}
