//! The PipeWire daemon connection: an `AF_UNIX` stream to `pipewire-0`, with the `SCM_RIGHTS`
//! ancillary-data fd passing (buffer memfds, activation eventfds) that
//! `std::os::unix::net::UnixStream` cannot express. Requests are serialized into a reused send
//! buffer and flushed with one `sendmsg`; events are read into a reused receive buffer and
//! walked message-by-message — no per-message heap allocation on the steady path.
//!
//! Modeled on `pf-present`'s Wayland `conn.rs` (same raw-wire discipline, same reason:
//! [[alloc-ban]] — off the per-message heap churn a C client library imposes).

#![allow(unsafe_code)] // libc socket/`sendmsg`/`recvmsg` + `SCM_RIGHTS`; each call is justified.
// Connection setup + framing is one-time / control-path work, not an `Element::process()`
// frame loop (spec: performance #1 — allocation discipline; clippy.toml).
#![allow(clippy::disallowed_methods)]

use std::collections::VecDeque;
use std::io;
use std::os::unix::io::RawFd;

use crate::native::pod::{PodBuilder, PodReader};
use crate::native::wire::{self, Header, HEADER};

/// The Core object id — id 0, always present (the connection root; `Core.Hello` targets it).
pub const CORE_ID: u32 = 0;
/// The Client object id — id 1, the server's view of us (`Client.UpdateProperties` targets it).
pub const CLIENT_ID: u32 = 1;

/// A live connection to the PipeWire daemon.
pub struct Connection {
    fd: RawFd,
    /// Requests staged for the next [`flush`](Self::flush).
    send_buf: Vec<u8>,
    /// Fds (positionally paired to the `Fd` PODs in `send_buf`) for the next flush's `SCM_RIGHTS`.
    send_fds: Vec<RawFd>,
    /// Bytes read from the socket but not yet consumed into whole messages.
    recv_buf: Vec<u8>,
    /// Fds received via `SCM_RIGHTS`, in arrival order.
    recv_fds: VecDeque<RawFd>,
    /// The current message's own fds (exactly `n_fds` of them), isolated from `recv_fds` when the
    /// message is popped so a handler can never steal a neighbouring message's fds.
    msg_fds: VecDeque<RawFd>,
    /// The current message's full bytes (reused across messages — zero steady-state alloc).
    msg: Vec<u8>,
    /// `size` of the current message's payload (bytes after the 16-byte header).
    msg_size: usize,
    /// Next client-allocated object id (ids 2.. ; 0 = Core, 1 = Client are reserved).
    next_id: u32,
    /// Monotonic message sequence number stamped into each request's header.
    seq: u32,
}

impl Connection {
    /// Resolve and connect to the daemon socket. Path resolution mirrors libpipewire:
    /// `$PIPEWIRE_REMOTE` (a socket name or absolute path) else `pipewire-0`, located under
    /// `$PIPEWIRE_RUNTIME_DIR` else `$XDG_RUNTIME_DIR`.
    pub fn connect() -> io::Result<Connection> {
        let name = std::env::var("PIPEWIRE_REMOTE").unwrap_or_else(|_| "pipewire-0".into());
        let path = if name.starts_with('/') {
            std::path::PathBuf::from(name)
        } else {
            let dir = std::env::var("PIPEWIRE_RUNTIME_DIR")
                .or_else(|_| std::env::var("XDG_RUNTIME_DIR"))
                .map_err(|_| io::Error::other("PIPEWIRE_RUNTIME_DIR/XDG_RUNTIME_DIR unset"))?;
            std::path::Path::new(&dir).join(name)
        };
        // SAFETY: a plain `socket(2)`; the returned fd is checked and owned by `Connection`.
        let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let bytes = path.as_os_str().as_encoded_bytes();
        if bytes.len() >= addr.sun_path.len() {
            unsafe { libc::close(fd) };
            return Err(io::Error::other("pipewire socket path too long"));
        }
        // SAFETY: `bytes` fits (checked) and is copied into the NUL-initialised sun_path.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                addr.sun_path.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
            let len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
            if libc::connect(fd, &addr as *const _ as *const libc::sockaddr, len) < 0 {
                let e = io::Error::last_os_error();
                libc::close(fd);
                return Err(e);
            }
        }
        Ok(Connection {
            fd,
            send_buf: Vec::new(),
            send_fds: Vec::new(),
            recv_buf: Vec::new(),
            recv_fds: VecDeque::new(),
            msg_fds: VecDeque::new(),
            msg: Vec::new(),
            msg_size: 0,
            next_id: 2,
            seq: 0,
        })
    }

    /// Allocate a fresh client object id (for a Registry/Node/... proxy we create).
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Stage one request to object `id`, method `opcode`. `build` receives a [`PodBuilder`]
    /// already inside the payload `Struct` — write the args, no `push_struct`/`pop` needed. Any
    /// `fd` args ride the next flush's `SCM_RIGHTS`. Nothing is sent until [`flush`](Self::flush).
    pub fn send(&mut self, id: u32, opcode: u8, build: impl FnOnce(&mut PodBuilder)) {
        let header = wire::begin(&mut self.send_buf, id, self.seq);
        let fds_before = self.send_fds.len();
        {
            let mut b = PodBuilder::new(&mut self.send_buf, &mut self.send_fds);
            b.push_struct();
            build(&mut b);
            b.pop();
        }
        let n_fds = (self.send_fds.len() - fds_before) as u32;
        wire::finish(&mut self.send_buf, header, opcode, n_fds);
        self.seq = self.seq.wrapping_add(1);
    }

    /// Flush every staged request in one (or more, if the buffer is large) `sendmsg`, passing all
    /// staged fds via `SCM_RIGHTS` on the first chunk.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.send_buf.is_empty() {
            return Ok(());
        }
        let mut off = 0;
        while off < self.send_buf.len() {
            off += self.sendmsg_from(off)?;
        }
        self.send_buf.clear();
        self.send_fds.clear();
        Ok(())
    }

    /// One `sendmsg` of `send_buf[off..]`, attaching all staged fds (only on the first chunk).
    fn sendmsg_from(&self, off: usize) -> io::Result<usize> {
        let mut iov = libc::iovec {
            iov_base: self.send_buf[off..].as_ptr() as *mut libc::c_void,
            iov_len: self.send_buf.len() - off,
        };
        let fds: &[RawFd] = if off == 0 { &self.send_fds } else { &[] };
        let cmsg_space = if fds.is_empty() {
            0
        } else {
            unsafe { libc::CMSG_SPACE(std::mem::size_of_val(fds) as u32) as usize }
        };
        let mut cmsg_buf = vec![0u8; cmsg_space];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if !fds.is_empty() {
            msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = cmsg_space as _;
            // SAFETY: `cmsg_buf` is sized by CMSG_SPACE for these fds; we fill exactly one hdr.
            unsafe {
                let cmsg = libc::CMSG_FIRSTHDR(&msg);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len =
                    libc::CMSG_LEN(std::mem::size_of_val(fds) as u32) as _;
                std::ptr::copy_nonoverlapping(
                    fds.as_ptr(),
                    libc::CMSG_DATA(cmsg) as *mut RawFd,
                    fds.len(),
                );
            }
        }
        // SAFETY: `msg` describes `iov`/`cmsg_buf`, both alive for this call.
        let n = unsafe { libc::sendmsg(self.fd, &msg, libc::MSG_NOSIGNAL) };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(n as usize)
    }

    /// Block for at least one read from the socket, appending bytes to `recv_buf` and any
    /// received fds to `recv_fds`. Returns `Ok(false)` on a clean EOF (daemon gone).
    pub fn recv(&mut self) -> io::Result<bool> {
        let mut chunk = [0u8; 8192];
        let mut iov = libc::iovec {
            iov_base: chunk.as_mut_ptr() as *mut libc::c_void,
            iov_len: chunk.len(),
        };
        let mut cmsg_buf = [0u8; 512];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_buf.len() as _;
        // SAFETY: `msg` points at `chunk`/`cmsg_buf`, both live for the call.
        let n = unsafe { libc::recvmsg(self.fd, &mut msg, libc::MSG_CMSG_CLOEXEC) };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        if n == 0 {
            return Ok(false);
        }
        // SAFETY: walk the well-formed control buffer the kernel just filled.
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                    let data = libc::CMSG_DATA(cmsg) as *const RawFd;
                    let payload = (*cmsg).cmsg_len as usize
                        - (libc::CMSG_DATA(cmsg) as usize - cmsg as usize);
                    let count = payload / std::mem::size_of::<RawFd>();
                    for i in 0..count {
                        self.recv_fds.push_back(std::ptr::read(data.add(i)));
                    }
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }
        self.recv_buf.extend_from_slice(&chunk[..n as usize]);
        Ok(true)
    }

    /// Pop the next whole message into the reused scratch, returning its [`Header`]. Its args are
    /// then read via [`args`](Self::args). `None` when `recv_buf` holds only a partial message
    /// (call [`recv`](Self::recv) for more).
    pub fn next_message(&mut self) -> Option<Header> {
        let (hdr, total) = wire::parse(&self.recv_buf)?;
        self.msg.clear();
        self.msg.extend_from_slice(&self.recv_buf[..total]);
        self.recv_buf.drain(..total);
        self.msg_size = hdr.size as usize;
        // Isolate this message's fds: any left un-taken from the previous message are closed
        // (a handler that ignores an fd must not leak it), then move exactly `n_fds` from the
        // arrival queue. This keeps each message's fds indexable independent of the others.
        for fd in self.msg_fds.drain(..) {
            unsafe { libc::close(fd) };
        }
        let take = (hdr.n_fds as usize).min(self.recv_fds.len());
        self.msg_fds.extend(self.recv_fds.drain(..take));
        Some(hdr)
    }

    /// Block until a whole message is available, driving [`recv`](Self::recv) as needed. Returns
    /// `Ok(None)` on a clean EOF with no buffered message left (daemon gone).
    pub fn next_blocking(&mut self) -> io::Result<Option<Header>> {
        loop {
            if let Some(hdr) = self.next_message() {
                return Ok(Some(hdr));
            }
            if !self.recv()? {
                return Ok(None);
            }
        }
    }

    /// An argument reader over the current message (positioned at the first arg of its payload
    /// `Struct`; a trailing footer, if any, is naturally excluded). Read all args into locals
    /// before issuing requests — the borrow is read-only.
    pub fn args(&self) -> PodReader<'_> {
        let payload = &self.msg[HEADER..HEADER + self.msg_size];
        PodReader::new(payload).enter_struct().unwrap_or_else(PodReader::empty)
    }

    /// Take the next fd belonging to the *current* message (in `fd`-arg order). Returns `None`
    /// once this message's fds are exhausted — never bleeds into another message's fds.
    pub fn take_fd(&mut self) -> Option<RawFd> {
        self.msg_fds.pop_front()
    }

    /// The raw socket fd (for the reactor integration — a follow-up; see the module docs).
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // SAFETY: our owned socket fd; closed exactly once. Any un-taken received fds leak until
        // process exit — acceptable for the control connection (buffer fds are taken + closed on
        // the data path).
        unsafe { libc::close(self.fd) };
    }
}
