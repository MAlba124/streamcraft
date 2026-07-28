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
    /// The mmap base as an integer address (not a raw pointer) so `ShmBuffer` — and thus the
    /// `Window` and the sink element — stays `Send`: the mapping is only ever touched from the
    /// single thread that owns the `Window`, so passing the address across the element's move
    /// to its scheduler thread is sound.
    map_addr: usize,
    size: usize,
    width: i32,
    height: i32,
    /// `true` while the compositor may still be reading it (between commit and `release`).
    busy: bool,
}

impl ShmBuffer {
    /// The pixel bytes as a mutable slice (ARGB8888, `width*4` stride).
    fn pixels(&mut self) -> &mut [u8] {
        // SAFETY: `map_addr` is our mapping's base, valid+writable for `size` bytes for this
        // buffer's lifetime, and this `&mut self` is the sole access (the compositor only reads
        // it while `busy`, between commit and release).
        unsafe { std::slice::from_raw_parts_mut(self.map_addr as *mut u8, self.size) }
    }
}

impl Drop for ShmBuffer {
    fn drop(&mut self) {
        // SAFETY: our mapping + owned fd, released exactly once.
        unsafe {
            libc::munmap(self.map_addr as *mut libc::c_void, self.size);
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
    viewporter: u32,
    subcompositor: u32,
    seat: u32,
    keyboard: u32,
    // Window objects.
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    /// The video surface's `wp_viewport` (crop+scale), created lazily on the first dmabuf frame.
    viewport: u32,
    /// The GUI overlay `wl_surface` + its `wl_subsurface` over the video (created lazily on the
    /// first [`present_gui`](Self::present_gui)); the compositor blends it above the video.
    gui_surface: u32,
    gui_subsurface: u32,
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
            viewporter: 0,
            subcompositor: 0,
            seat: 0,
            keyboard: 0,
            viewport: 0,
            gui_surface: 0,
            gui_subsurface: 0,
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
        if let Some((vname, vver)) = find(p::viewporter::NAME) {
            w.viewporter = w.conn.alloc_id();
            w.conn
                .request(registry, p::registry::BIND)
                .u32(vname)
                .bind_new_id(p::viewporter::NAME, vver.min(1), w.viewporter)
                .finish();
        }
        if let Some((scname, scver)) = find(p::subcompositor::NAME) {
            w.subcompositor = w.conn.alloc_id();
            w.conn
                .request(registry, p::registry::BIND)
                .u32(scname)
                .bind_new_id(p::subcompositor::NAME, scver.min(1), w.subcompositor)
                .finish();
        }
        if let Some((sename, sever)) = find(p::seat::NAME) {
            w.seat = w.conn.alloc_id();
            w.conn
                .request(registry, p::registry::BIND)
                .u32(sename)
                .bind_new_id(p::seat::NAME, sever.min(5), w.seat)
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
        Ok(ShmBuffer { buffer, fd, map_addr: map as usize, size, width: w, height: h, busy: false })
    }

    /// Present a zero-copy dmabuf video frame: import its planes as a `wl_buffer`
    /// (`zwp_linux_dmabuf`), crop the coded surface to the display rect and scale it to the
    /// window (`wp_viewport`), then attach + commit. The `wl_buffer` is destroyed when the
    /// compositor releases it (tracked in `dmabuf_inflight`), so the decoder's surface stays
    /// pinned exactly until the compositor is done sampling it. Requires the compositor to
    /// expose `zwp_linux_dmabuf_v1` (checked) — [`has_dmabuf`](Self::has_dmabuf).
    pub fn present_dmabuf(&mut self, v: &DmabufVideo<'_>) -> io::Result<()> {
        if self.dmabuf == 0 {
            return Err(io::Error::other("compositor lacks zwp_linux_dmabuf_v1"));
        }
        // Accumulate the planes into a params object, then mint the buffer synchronously.
        let params = self.conn.alloc_id();
        self.conn.request(self.dmabuf, p::dmabuf::CREATE_PARAMS).u32(params).finish();
        let mod_hi = (v.drm_modifier >> 32) as u32;
        let mod_lo = v.drm_modifier as u32;
        for (i, pl) in v.planes.iter().enumerate() {
            let fd = v.fds[pl.object_index as usize];
            self.conn
                .request(params, p::dmabuf_params::ADD)
                .fd(fd)
                .u32(i as u32)
                .u32(pl.offset)
                .u32(pl.pitch)
                .u32(mod_hi)
                .u32(mod_lo)
                .finish();
        }
        let buffer = self.conn.alloc_id();
        self.conn
            .request(params, p::dmabuf_params::CREATE_IMMED)
            .u32(buffer)
            .i32(v.coded_w)
            .i32(v.coded_h)
            .u32(v.drm_format)
            .u32(0) // flags: no y-invert / interlace
            .finish();
        self.conn.request(params, p::dmabuf_params::DESTROY).finish();

        // Crop the coded surface to the (cropped) display rect and scale it to the window.
        self.ensure_viewport();
        if self.viewport != 0 {
            self.conn
                .request(self.viewport, p::viewport::SET_SOURCE)
                .fixed(fx(v.crop_x))
                .fixed(fx(v.crop_y))
                .fixed(fx(v.disp_w))
                .fixed(fx(v.disp_h))
                .finish();
            self.conn
                .request(self.viewport, p::viewport::SET_DESTINATION)
                .i32(self.width.max(1))
                .i32(self.height.max(1))
                .finish();
        }
        self.conn.request(self.surface, p::surface::ATTACH).object(buffer).i32(0).i32(0).finish();
        self.conn
            .request(self.surface, p::surface::DAMAGE_BUFFER)
            .i32(0)
            .i32(0)
            .i32(v.coded_w)
            .i32(v.coded_h)
            .finish();
        self.conn.request(self.surface, p::surface::COMMIT).finish();
        self.dmabuf_inflight.push(buffer);
        self.conn.flush()
    }

    /// Whether the compositor can import dmabuf video (else the caller must fall back to shm).
    pub fn has_dmabuf(&self) -> bool {
        self.dmabuf != 0
    }

    /// Lazily create the video surface's `wp_viewport` (once), if the compositor has viewporter.
    fn ensure_viewport(&mut self) {
        if self.viewport == 0 && self.viewporter != 0 {
            self.viewport = self.conn.alloc_id();
            let (vp, surf) = (self.viewport, self.surface);
            self.conn.request(self.viewporter, p::viewporter::GET_VIEWPORT).u32(vp).object(surf).finish();
        }
    }

    /// Render + present the **GUI overlay**: a `wl_subsurface` above the video at `(x, y)`,
    /// software-rendered `w×h` through the [`Canvas`] into a recycled shm buffer. The compositor
    /// blends it over the video — no GPU, no readback. The subsurface is desynchronized, so the
    /// GUI updates independently of the video rate (call it only on change). No-op if the
    /// compositor lacks `wl_subcompositor`. Whether the overlay is available:
    /// [`has_gui`](Self::has_gui).
    pub fn present_gui<F: FnOnce(&mut Canvas)>(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        render: F,
    ) -> io::Result<()> {
        if self.subcompositor == 0 {
            return Ok(());
        }
        self.ensure_gui_subsurface(x, y);
        let (w, h) = (w.max(1), h.max(1));
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
        let gs = self.gui_surface;
        self.conn.request(gs, p::surface::ATTACH).object(buffer).i32(0).i32(0).finish();
        self.conn.request(gs, p::surface::DAMAGE_BUFFER).i32(0).i32(0).i32(bw).i32(bh).finish();
        self.conn.request(gs, p::surface::COMMIT).finish();
        self.conn.flush()
    }

    /// Whether the compositor supports the GUI overlay (`wl_subcompositor`).
    pub fn has_gui(&self) -> bool {
        self.subcompositor != 0
    }

    /// Lazily create the GUI `wl_surface` + `wl_subsurface` over the video at `(x, y)`, once.
    fn ensure_gui_subsurface(&mut self, x: i32, y: i32) {
        if self.gui_subsurface != 0 || self.subcompositor == 0 {
            return;
        }
        self.gui_surface = self.conn.alloc_id();
        self.conn.request(self.compositor, p::compositor::CREATE_SURFACE).u32(self.gui_surface).finish();
        self.gui_subsurface = self.conn.alloc_id();
        let (sub, gs, parent) = (self.gui_subsurface, self.gui_surface, self.surface);
        self.conn
            .request(self.subcompositor, p::subcompositor::GET_SUBSURFACE)
            .u32(sub)
            .object(gs)
            .object(parent)
            .finish();
        self.conn.request(sub, p::subsurface::SET_POSITION).i32(x).i32(y).finish();
        // Desync: the GUI commits apply immediately, decoupled from the video surface's cadence.
        self.conn.request(sub, p::subsurface::SET_DESYNC).finish();
        // The subsurface add + position are parent-cached — a parent commit applies them (a
        // present_dmabuf commit follows anyway, but do one now so the placement takes effect).
        self.conn.request(self.surface, p::surface::COMMIT).finish();
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
        } else if obj == self.seat && op == p::seat::EV_CAPABILITIES {
            // Grab a keyboard once the seat advertises one (for Esc/q → close).
            let caps = self.conn.args().u32().unwrap_or(0);
            if caps & p::seat::CAP_KEYBOARD != 0 && self.keyboard == 0 {
                self.keyboard = self.conn.alloc_id();
                let (kb, seat) = (self.keyboard, self.seat);
                self.conn.request(seat, p::seat::GET_KEYBOARD).u32(kb).finish();
            }
        } else if obj == self.keyboard && op == p::keyboard::EV_KEY {
            // key(serial, time, key, state): Esc / q closes the window.
            let mut a = self.conn.args();
            let _serial = a.u32();
            let _time = a.u32();
            let key = a.u32().unwrap_or(0);
            let state = a.u32().unwrap_or(0);
            if state == p::keyboard::STATE_PRESSED
                && (key == p::keyboard::KEY_ESC || key == p::keyboard::KEY_Q)
            {
                self.closed = true;
            }
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

}

/// One dmabuf plane's geometry (mirrors `sc_vaapi::va::ExportedPlane`, kept local so the
/// presenter core does not depend on the VA-API crate — the sink maps `GpuFrame` into these).
#[derive(Clone, Copy)]
pub struct Plane {
    /// Index into [`DmabufVideo::fds`] — which dmabuf object holds this plane.
    pub object_index: u32,
    /// Byte offset of the plane's first row within that object.
    pub offset: u32,
    /// Row stride in bytes.
    pub pitch: u32,
}

/// A zero-copy video frame to import via `zwp_linux_dmabuf` — the decoded VA surface's dmabuf
/// fds + plane layout + DRM format/modifier + the coded/crop/display geometry.
pub struct DmabufVideo<'a> {
    /// The dup'd dmabuf fds (one per object); still owned by the caller (the sink closes them
    /// after this call — the compositor holds its own references once the buffer is created).
    pub fds: &'a [RawFd],
    pub planes: &'a [Plane],
    /// DRM FourCC (e.g. `DRM_FORMAT_NV12`).
    pub drm_format: u32,
    /// DRM format modifier (`DRM_FORMAT_MOD_INVALID` when none).
    pub drm_modifier: u64,
    /// Coded (allocation) size — the dmabuf plane geometry.
    pub coded_w: i32,
    pub coded_h: i32,
    /// Crop origin + display (cropped) size within the coded surface (the viewport source).
    pub crop_x: i32,
    pub crop_y: i32,
    pub disp_w: i32,
    pub disp_h: i32,
}

/// Integer → `wl_fixed` (24.8) for the viewport source rect.
fn fx(n: i32) -> crate::wire::Fixed {
    crate::wire::Fixed::from_int(n)
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
