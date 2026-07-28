//! [`Window`] — an `xdg_toplevel` over the raw wire client, with an **recycling `wl_shm`
//! buffer pool** for software-rendered content and (via [`present_dmabuf`](Window::present_dmabuf))
//! zero-copy dmabuf video. This is the reusable core the sink element and the GUI drive.
//!
//! Buffers are reused on `wl_buffer.release` (double-buffered: the app renders into a free
//! buffer while the compositor still holds the last one), so the steady-state present path
//! allocates nothing — the whole reason for talking the wire ourselves.

#![allow(unsafe_code)]

use std::io;
use std::os::unix::io::RawFd;

use crate::conn::{Connection, DISPLAY_ID};
use crate::protocol as p;
use crate::raster::Canvas;

/// One `wl_shm` buffer: an mmap'd `memfd` region + its `wl_buffer` id, reused across frames.
struct ShmBuffer {
    buffer: u32,
    fd: RawFd,
    map: *mut u8,
    size: usize,
    width: i32,
    height: i32,
    /// `true` while the compositor may still be reading it (between commit and `release`).
    busy: bool,
}

impl ShmBuffer {
    /// The pixel bytes as a mutable slice (ARGB8888, `width*4` stride).
    fn pixels(&mut self) -> &mut [u8] {
        // SAFETY: `map` addresses `size` writable bytes for this buffer's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.map, self.size) }
    }
}

impl Drop for ShmBuffer {
    fn drop(&mut self) {
        // SAFETY: our mapping + owned fd, released exactly once.
        unsafe {
            libc::munmap(self.map as *mut libc::c_void, self.size);
            libc::close(self.fd);
        }
    }
}

/// A presentation window: one `xdg_toplevel` surface plus the compositor globals we bound.
pub struct Window {
    conn: Connection,
    // Bound globals (0 = absent).
    compositor: u32,
    shm: u32,
    wm_base: u32,
    dmabuf: u32,
    // Window objects.
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    // Geometry (compositor-configured; falls back to the requested default until then).
    width: i32,
    height: i32,
    // Lifecycle.
    configured: bool,
    closed: bool,
    /// The dmabuf `wl_buffer`s in flight, awaiting `release` (video path).
    dmabuf_inflight: Vec<u32>,
    /// The recycling software-render buffer pool.
    pool: Vec<ShmBuffer>,
}

/// A global advertised by the registry, captured during enumeration.
struct Global {
    name: u32,
    interface: [u8; 40],
    ilen: usize,
    version: u32,
}

impl Window {
    /// Open a titled window of the requested default size and run the initial handshake
    /// (registry → bind → `xdg_toplevel` → first configure). The window has no content until
    /// the first [`frame`](Self::frame) / [`present_dmabuf`](Self::present_dmabuf).
    pub fn open(title: &str, width: i32, height: i32) -> io::Result<Window> {
        let mut conn = Connection::connect()?;

        // get_registry + a sync barrier to know when all globals have been advertised.
        let registry = conn.alloc_id();
        conn.request(DISPLAY_ID, p::display::GET_REGISTRY).u32(registry).finish();
        let sync_cb = conn.alloc_id();
        conn.request(DISPLAY_ID, p::display::SYNC).u32(sync_cb).finish();
        conn.flush()?;

        // Collect globals into a small fixed buffer (interface names are short) — no per-global
        // heap growth beyond the bounded Vec.
        let mut globals: Vec<Global> = Vec::new();
        let mut done = false;
        while !done {
            if !conn.recv()? {
                return Err(io::Error::other("compositor closed during enumeration"));
            }
            while let Some((obj, op)) = conn.next_event() {
                if obj == registry && op == p::registry::EV_GLOBAL {
                    let mut a = conn.args();
                    let name = a.u32().unwrap_or(0);
                    let iface = a.string().unwrap_or("");
                    let version = a.u32().unwrap_or(0);
                    let mut interface = [0u8; 40];
                    let ilen = iface.len().min(interface.len());
                    interface[..ilen].copy_from_slice(&iface.as_bytes()[..ilen]);
                    globals.push(Global { name, interface, ilen, version });
                } else if obj == sync_cb && op == p::callback::EV_DONE {
                    done = true;
                } else if obj == DISPLAY_ID && op == p::display::EV_ERROR {
                    return Err(display_error(&conn));
                }
            }
        }

        let find = |iface: &str| -> Option<(u32, u32)> {
            globals
                .iter()
                .find(|g| &g.interface[..g.ilen] == iface.as_bytes())
                .map(|g| (g.name, g.version))
        };

        let mut w = Window {
            surface: 0,
            xdg_surface: 0,
            toplevel: 0,
            compositor: 0,
            shm: 0,
            wm_base: 0,
            dmabuf: 0,
            width,
            height,
            configured: false,
            closed: false,
            dmabuf_inflight: Vec::new(),
            pool: Vec::new(),
            conn,
        };

        // Bind the required globals; dmabuf is optional (only the video path needs it).
        let (cname, cver) = find(p::compositor::NAME).ok_or_else(|| miss("wl_compositor"))?;
        w.compositor = w.conn.alloc_id();
        w.conn
            .request(registry, p::registry::BIND)
            .u32(cname)
            .bind_new_id(p::compositor::NAME, cver.min(4), w.compositor)
            .finish();
        let (sname, sver) = find(p::shm::NAME).ok_or_else(|| miss("wl_shm"))?;
        w.shm = w.conn.alloc_id();
        w.conn
            .request(registry, p::registry::BIND)
            .u32(sname)
            .bind_new_id(p::shm::NAME, sver.min(1), w.shm)
            .finish();
        let (wname, wver) = find(p::wm_base::NAME).ok_or_else(|| miss("xdg_wm_base"))?;
        w.wm_base = w.conn.alloc_id();
        w.conn
            .request(registry, p::registry::BIND)
            .u32(wname)
            .bind_new_id(p::wm_base::NAME, wver.min(2), w.wm_base)
            .finish();
        if let Some((dname, dver)) = find(p::dmabuf::NAME) {
            w.dmabuf = w.conn.alloc_id();
            w.conn
                .request(registry, p::registry::BIND)
                .u32(dname)
                .bind_new_id(p::dmabuf::NAME, dver.min(4), w.dmabuf)
                .finish();
        }

        // Role the surface as an xdg_toplevel and commit once for the initial configure.
        w.surface = w.conn.alloc_id();
        w.conn.request(w.compositor, p::compositor::CREATE_SURFACE).u32(w.surface).finish();
        w.xdg_surface = w.conn.alloc_id();
        w.conn
            .request(w.wm_base, p::wm_base::GET_XDG_SURFACE)
            .u32(w.xdg_surface)
            .object(w.surface)
            .finish();
        w.toplevel = w.conn.alloc_id();
        w.conn.request(w.xdg_surface, p::xdg_surface::GET_TOPLEVEL).u32(w.toplevel).finish();
        w.conn.request(w.toplevel, p::xdg_toplevel::SET_TITLE).string(title).finish();
        w.conn.request(w.surface, p::surface::COMMIT).finish();
        w.conn.flush()?;

        // Block for the first configure so the caller can present immediately after `open`.
        while !w.configured && !w.closed {
            w.dispatch(true)?;
        }
        Ok(w)
    }

    /// The current window size (compositor-configured, or the open default).
    pub fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    /// Whether the compositor asked the window to close (`xdg_toplevel.close`).
    pub fn should_close(&self) -> bool {
        self.closed
    }

    /// Software-render one frame: acquire a free pool buffer sized `width×height`, hand its
    /// [`Canvas`] to `render`, then attach + damage + commit it. Reuses a released buffer when
    /// one is free (zero-alloc steady state), else grows the pool by one.
    pub fn frame<F: FnOnce(&mut Canvas)>(&mut self, render: F) -> io::Result<()> {
        let (w, h) = (self.width.max(1), self.height.max(1));
        let idx = self.acquire(w, h)?;
        {
            let b = &mut self.pool[idx];
            let stride = (b.width * 4) as u32;
            let (bw, bh) = (b.width as u32, b.height as u32);
            let mut canvas = Canvas::new(b.pixels(), bw, bh, stride);
            render(&mut canvas);
        }
        let (buffer, bw, bh) = {
            let b = &self.pool[idx];
            (b.buffer, b.width, b.height)
        };
        self.pool[idx].busy = true;
        self.conn.request(self.surface, p::surface::ATTACH).object(buffer).i32(0).i32(0).finish();
        self.conn.request(self.surface, p::surface::DAMAGE_BUFFER).i32(0).i32(0).i32(bw).i32(bh).finish();
        self.conn.request(self.surface, p::surface::COMMIT).finish();
        self.conn.flush()
    }

    /// Find a free pool buffer of the right size, or create one. Recycles released buffers.
    fn acquire(&mut self, w: i32, h: i32) -> io::Result<usize> {
        if let Some(i) = self.pool.iter().position(|b| !b.busy && b.width == w && b.height == h) {
            return Ok(i);
        }
        let buf = self.make_shm_buffer(w, h)?;
        self.pool.push(buf);
        Ok(self.pool.len() - 1)
    }

    /// Allocate a fresh `memfd`-backed `wl_shm` buffer of `w×h` ARGB8888.
    fn make_shm_buffer(&mut self, w: i32, h: i32) -> io::Result<ShmBuffer> {
        let stride = w * 4;
        let size = (stride * h) as usize;
        let (fd, map) = shm_alloc(size)?;
        let pool = self.conn.alloc_id();
        self.conn.request(self.shm, p::shm::CREATE_POOL).u32(pool).fd(fd).i32(size as i32).finish();
        let buffer = self.conn.alloc_id();
        self.conn
            .request(pool, p::shm_pool::CREATE_BUFFER)
            .u32(buffer)
            .i32(0)
            .i32(w)
            .i32(h)
            .i32(stride)
            .u32(p::shm::FORMAT_ARGB8888)
            .finish();
        // The buffer holds its own reference to the mmap'd memory, so the pool object can go
        // now (spec: wl_shm_pool.destroy) — frees the id, keeps the buffer valid.
        self.conn.request(pool, p::shm_pool::DESTROY).finish();
        Ok(ShmBuffer { buffer, fd, map: map as *mut u8, size, width: w, height: h, busy: false })
    }

    /// Pump the socket once and dispatch pending events (configure/ping/close/release). When
    /// `block` is true, wait for at least one event; otherwise return promptly if none are
    /// ready (the sink's non-blocking pump inside `process()`).
    pub fn dispatch(&mut self, block: bool) -> io::Result<()> {
        if block || readable(self.conn.as_raw_fd(), 0)? {
            if !self.conn.recv()? {
                self.closed = true;
                return Ok(());
            }
        }
        while let Some((obj, op)) = self.conn.next_event() {
            self.handle(obj, op)?;
        }
        self.conn.flush()
    }

    fn handle(&mut self, obj: u32, op: u16) -> io::Result<()> {
        if obj == self.wm_base && op == p::wm_base::EV_PING {
            let serial = self.conn.args().u32().unwrap_or(0);
            self.conn.request(self.wm_base, p::wm_base::PONG).u32(serial).finish();
        } else if obj == self.xdg_surface && op == p::xdg_surface::EV_CONFIGURE {
            let serial = self.conn.args().u32().unwrap_or(0);
            self.conn.request(self.xdg_surface, p::xdg_surface::ACK_CONFIGURE).u32(serial).finish();
            self.configured = true;
        } else if obj == self.toplevel && op == p::xdg_toplevel::EV_CONFIGURE {
            let mut a = self.conn.args();
            let w = a.i32().unwrap_or(0);
            let h = a.i32().unwrap_or(0);
            // A 0 dimension means "you choose" — keep the current size.
            if w > 0 {
                self.width = w;
            }
            if h > 0 {
                self.height = h;
            }
        } else if obj == self.toplevel && op == p::xdg_toplevel::EV_CLOSE {
            self.closed = true;
        } else if op == p::buffer::EV_RELEASE {
            // A pool buffer (recycle) or an in-flight dmabuf buffer (destroy) was released.
            if let Some(b) = self.pool.iter_mut().find(|b| b.buffer == obj) {
                b.busy = false;
            } else if let Some(pos) = self.dmabuf_inflight.iter().position(|&b| b == obj) {
                self.dmabuf_inflight.swap_remove(pos);
                self.conn.request(obj, p::buffer::DESTROY).finish();
            }
        } else if obj == DISPLAY_ID && op == p::display::EV_ERROR {
            return Err(display_error(&self.conn));
        }
        Ok(())
    }

    /// Mutable access to the raw connection (for the video path to mint dmabuf buffers).
    pub fn conn(&mut self) -> &mut Connection {
        &mut self.conn
    }

    /// The bound `zwp_linux_dmabuf_v1` id (0 if the compositor lacks it), the video surface id,
    /// and a slot to register an in-flight dmabuf buffer for release-driven destroy.
    pub fn dmabuf_global(&self) -> u32 {
        self.dmabuf
    }
    pub fn video_surface(&self) -> u32 {
        self.surface
    }
    pub fn register_dmabuf_inflight(&mut self, buffer: u32) {
        self.dmabuf_inflight.push(buffer);
    }
}

fn miss(iface: &str) -> io::Error {
    io::Error::other(format!("compositor is missing required global {iface}"))
}

fn display_error(c: &Connection) -> io::Error {
    let mut a = c.args();
    let object = a.u32().unwrap_or(0);
    let code = a.u32().unwrap_or(0);
    let msg = a.string().unwrap_or("");
    io::Error::other(format!("wayland protocol error on object {object}: code {code}: {msg}"))
}

/// A `memfd`-backed shared region of `size` bytes, mmap'd writable.
pub(crate) fn shm_alloc(size: usize) -> io::Result<(RawFd, *mut libc::c_void)> {
    // SAFETY: memfd_create with a static name; fd is checked before use.
    let fd = unsafe { libc::memfd_create(c"sc-present".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: sizing then mapping our own fresh memfd; on any failure we close and bail.
    unsafe {
        if libc::ftruncate(fd, size as libc::off_t) < 0 {
            let e = io::Error::last_os_error();
            libc::close(fd);
            return Err(e);
        }
        let map = libc::mmap(
            std::ptr::null_mut(),
            size.max(1),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        if map == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            libc::close(fd);
            return Err(e);
        }
        Ok((fd, map))
    }
}

/// Block up to `timeout_ms` (0 = poll) for the fd to be readable.
pub(crate) fn readable(fd: RawFd, timeout_ms: i32) -> io::Result<bool> {
    let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: one valid pollfd.
    let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n > 0 && (pfd.revents & libc::POLLIN) != 0)
}
