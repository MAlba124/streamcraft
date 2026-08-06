//! The Wayland display-socket connection: an `AF_UNIX` stream to the compositor, with the
//! [`SCM_RIGHTS`](libc::SCM_RIGHTS) ancillary-data fd passing (`shm` pool + `dmabuf` fds) that
//! `std::os::unix::net::UnixStream` cannot express. Requests are serialized into a reused send
//! buffer and flushed with one `sendmsg`; events are read into a reused receive buffer and
//! walked message-by-message through a reused scratch — no per-message heap allocation.

#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::io;
use std::os::unix::io::RawFd;

use crate::wire::{self, ArgReader, Writer};

/// The `wl_display` object id — id 1, always present (protocol spec §wl_display).
pub const DISPLAY_ID: u32 = 1;

/// A live connection to the Wayland compositor.
pub struct Connection {
    fd: RawFd,
    /// Requests staged for the next [`flush`](Self::flush).
    send_buf: Vec<u8>,
    /// Fds (positionally paired to `fd` args in `send_buf`) for the next flush's `SCM_RIGHTS`.
    send_fds: Vec<RawFd>,
    /// Bytes read from the socket but not yet consumed into whole messages.
    recv_buf: Vec<u8>,
    /// Fds received via `SCM_RIGHTS`, in arrival order, paired to fd-args of pending events.
    recv_fds: VecDeque<RawFd>,
    /// The current event's full bytes (reused across events — zero steady-state alloc).
    msg: Vec<u8>,
    /// Next client-allocated object id (ids 1.. ; 1 is `wl_display`).
    next_id: u32,
}

impl Connection {
    /// Connect to `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY` (or `$WAYLAND_DISPLAY` if it is an
    /// absolute path). Mirrors libwayland's `wl_display_connect` resolution.
    pub fn connect() -> io::Result<Connection> {
        let disp = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".into());
        let path = if disp.starts_with('/') {
            std::path::PathBuf::from(disp)
        } else {
            let dir = std::env::var("XDG_RUNTIME_DIR")
                .map_err(|_| io::Error::other("XDG_RUNTIME_DIR unset"))?;
            std::path::Path::new(&dir).join(disp)
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
            return Err(io::Error::other("wayland socket path too long"));
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
            msg: Vec::new(),
            next_id: 2,
        })
    }

    /// Allocate a fresh client object id.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Begin a request from `object`, request `opcode`. Write its args on the returned
    /// [`Writer`] and call [`Writer::finish`]; the bytes stage in the reused send buffer until
    /// [`flush`](Self::flush).
    pub fn request(&mut self, object: u32, opcode: u16) -> Writer<'_> {
        Writer::new(&mut self.send_buf, &mut self.send_fds, object, opcode)
    }

    /// Flush every staged request in one `sendmsg`, passing any staged fds via `SCM_RIGHTS`.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.send_buf.is_empty() {
            return Ok(());
        }
        let mut off = 0;
        while off < self.send_buf.len() {
            let n = self.sendmsg_from(off)?;
            off += n;
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
        // Ancillary buffer for the fds (only sent with the first chunk, where off == 0).
        let fds: &[RawFd] = if off == 0 { &self.send_fds } else { &[] };
        let cmsg_space = if fds.is_empty() {
            0
        } else {
            unsafe { libc::CMSG_SPACE((std::mem::size_of::<RawFd>() * fds.len()) as u32) as usize }
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
                    libc::CMSG_LEN((std::mem::size_of::<RawFd>() * fds.len()) as u32) as _;
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

    /// Block for at least one event batch from the socket, appending bytes to `recv_buf` and
    /// any received fds to `recv_fds`. Returns `Ok(false)` on a clean EOF (compositor gone).
    pub fn recv(&mut self) -> io::Result<bool> {
        let mut chunk = [0u8; 4096];
        let mut iov = libc::iovec {
            iov_base: chunk.as_mut_ptr() as *mut libc::c_void,
            iov_len: chunk.len(),
        };
        // Room for a handful of fds per read (keymap, dmabuf feedback — rare).
        let mut cmsg_buf = [0u8; 256];
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
        // Collect any passed fds.
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

    /// Pop the next whole event into the reused scratch, returning `(object, opcode)`. Its
    /// arguments are then read via [`args`](Self::args). `None` when `recv_buf` holds only a
    /// partial message (call [`recv`](Self::recv) for more).
    pub fn next_event(&mut self) -> Option<(u32, u16)> {
        let (object, opcode, size) = {
            let (ev, size) = wire::parse_message(&self.recv_buf)?;
            (ev.object, ev.opcode, size)
        };
        self.msg.clear();
        self.msg.extend_from_slice(&self.recv_buf[..size]);
        self.recv_buf.drain(..size);
        Some((object, opcode))
    }

    /// An argument reader over the current event (the one [`next_event`](Self::next_event) just
    /// returned). Read all args into locals before issuing requests (the borrow is read-only).
    pub fn args(&self) -> ArgReader<'_> {
        ArgReader::new(&self.msg[8..])
    }

    /// Take the oldest received fd (for an event carrying an `fd` arg, e.g. `wl_keyboard.keymap`).
    pub fn take_fd(&mut self) -> Option<RawFd> {
        self.recv_fds.pop_front()
    }

    /// The raw socket fd (for polling / the reactor integration later).
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // SAFETY: our owned socket fd; closed exactly once.
        unsafe { libc::close(self.fd) };
    }
}
