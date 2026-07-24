//! A minimal Wayland client for a single video window (spec: `wayland/spec/wayland.xml`,
//! `xdg-shell.xml`). It speaks the hand-written wire format ([`crate::wire`]) over a
//! `std::os::unix::net::UnixStream`, binds only the globals a video window needs, drives an
//! `xdg_toplevel`, and presents frames from a double-buffered `wl_shm` swapchain.
//!
//! Object-id allocation is deliberately trivial: client ids count up from 2 (id 1 is the
//! `wl_display`; the server owns the odd/even split, but a client may use any id > 1 it has
//! not used). We never recycle ids except when the server sends `wl_display.delete_id`, and
//! since we allocate only a fixed handful (registry, compositor, shm, surface, xdg objects,
//! pool, N buffers, sync/frame callbacks) a monotonic counter never realistically overflows.
//!
//! All parsing is total — a malformed event is surfaced as [`Error::Resource`], never a
//! panic (per the task's "NEVER panic on malformed server bytes").

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use streamcraft_core::error::Error;

use crate::protocol as op;
use crate::sys::{Memfd, Mmap};
use crate::wire::{ArgReader, Header, MessageBuilder};

/// The wl_display object id is always 1.
const DISPLAY: u32 = op::WL_DISPLAY_ID;
/// Client object ids start here and count up.
const FIRST_CLIENT_ID: u32 = 2;

fn resource(msg: impl Into<String>) -> Error {
    Error::Resource(msg.into())
}

/// One shm buffer in the swapchain: the `wl_buffer` object id, its byte offset into the
/// pool, and whether the compositor currently holds it (released → free to reuse).
struct SwapBuffer {
    wl_buffer: u32,
    offset: usize,
    /// `true` while free for us to draw into; `false` between `attach`+`commit` and the
    /// compositor's `wl_buffer.release`.
    available: bool,
}

/// The double-buffered `wl_shm` pool + its buffers (spec: task deliverable 2). One memfd
/// backs `n` XRGB8888 buffers of `width*height`; we draw into a free one, attach it, and
/// wait for `release` before reusing it.
///
/// On a late/changed dimension announce we **destroy and rebuild** the pool + buffers
/// ([`build_swapchain`](WaylandClient::build_swapchain)) rather than issuing
/// `wl_shm_pool.resize` (opcode wired as [`op::WL_SHM_POOL_RESIZE`]): every buffer's
/// width/height/stride changes with the frame size, so the old `wl_buffer`s must be
/// recreated anyway, and a fresh memfd of the exact new size is the simplest correct path.
/// `resize` only helps when *adding* same-size buffers to an existing pool, which we never do.
struct Swapchain {
    memfd: Memfd,
    map: Mmap,
    pool: u32, // wl_shm_pool object id
    width: usize,
    height: usize,
    stride: usize,
    buffers: Vec<SwapBuffer>,
}

/// A minimal Wayland client owning one video window.
pub struct WaylandClient {
    stream: UnixStream,
    /// Receive buffer for partial messages spanning `read` boundaries.
    rx: Vec<u8>,
    next_id: u32,

    // Bound globals (0 = not yet bound).
    registry: u32,
    compositor: u32,
    shm: u32,
    xdg_wm_base: u32,
    surface: u32,
    xdg_surface: u32,
    xdg_toplevel: u32,

    // Registry discovery: (name, version) for each global we care about, learned from
    // wl_registry.global before we bind.
    compositor_global: Option<(u32, u32)>,
    shm_global: Option<(u32, u32)>,
    wm_base_global: Option<(u32, u32)>,

    /// wl_shm formats the server advertised (we require XRGB8888).
    shm_formats: Vec<u32>,

    swap: Option<Swapchain>,

    /// Set once the xdg_surface has been configured+acked (safe to attach the first buffer).
    configured: bool,
    /// The most recent xdg_surface.configure serial awaiting ack (0 = none pending).
    pending_configure: Option<u32>,
    /// Set when the compositor sends xdg_toplevel.close — the user closed the window.
    closed: bool,
    /// Set when a wl_display.error event arrives; the client is then unusable.
    protocol_error: Option<String>,

    /// Object id of an in-flight wl_display.sync callback we are waiting on (0 = none).
    sync_callback: u32,
    sync_done: bool,
}

impl WaylandClient {
    /// Connect to `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY` (or an absolute `$WAYLAND_DISPLAY`),
    /// then bind the registry and its globals. Returns [`Error::Resource`] if the socket is
    /// absent (no compositor), so a caller can skip cleanly.
    pub fn connect() -> Result<WaylandClient, Error> {
        let path = socket_path()?;
        let stream = UnixStream::connect(&path)
            .map_err(|e| resource(format!("wayland: connect {}: {e}", path.display())))?;
        stream
            .set_nonblocking(false)
            .map_err(|e| resource(format!("wayland: set blocking: {e}")))?;

        let mut c = WaylandClient {
            stream,
            rx: Vec::with_capacity(4096),
            next_id: FIRST_CLIENT_ID,
            registry: 0,
            compositor: 0,
            shm: 0,
            xdg_wm_base: 0,
            surface: 0,
            xdg_surface: 0,
            xdg_toplevel: 0,
            compositor_global: None,
            shm_global: None,
            wm_base_global: None,
            shm_formats: Vec::new(),
            swap: None,
            configured: false,
            pending_configure: None,
            closed: false,
            protocol_error: None,
            sync_callback: 0,
            sync_done: false,
        };
        c.init_registry()?;
        Ok(c)
    }

    /// Allocate the next client object id.
    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Send a request. If it carries an fd, use the SCM_RIGHTS path; otherwise a plain
    /// write. Wayland messages are small, so a single `write_all` suffices.
    fn send(&mut self, msg: &MessageBuilder) -> Result<(), Error> {
        let bytes = msg.finish();
        match msg.take_fd() {
            Some(fd) => crate::sys::send_with_fd(&self.stream, &bytes, fd)
                .map_err(|e| resource(format!("wayland: sendmsg: {e}")))?,
            None => self
                .stream
                .write_all(&bytes)
                .map_err(|e| resource(format!("wayland: write: {e}")))?,
        }
        Ok(())
    }

    // --- registry bring-up -------------------------------------------------------------

    /// get_registry, then roundtrip so every `global` has arrived, then bind what we need.
    fn init_registry(&mut self) -> Result<(), Error> {
        self.registry = self.alloc_id();
        // wl_display.get_registry(new_id registry)
        let mut m = MessageBuilder::new(DISPLAY, op::WL_DISPLAY_GET_REGISTRY);
        m.u32(self.registry);
        self.send(&m)?;

        // A sync roundtrip: the server answers callbacks in order, so once our sync
        // callback fires, every global advertised before it has been delivered.
        self.roundtrip()?;

        // Bind the globals we found.
        let (name, ver) = self
            .compositor_global
            .ok_or_else(|| resource("wayland: no wl_compositor global"))?;
        self.compositor = self.bind(name, op::WL_COMPOSITOR, ver.min(op::WL_COMPOSITOR_VERSION))?;

        let (name, ver) = self
            .shm_global
            .ok_or_else(|| resource("wayland: no wl_shm global"))?;
        self.shm = self.bind(name, op::WL_SHM, ver.min(op::WL_SHM_VERSION))?;

        let (name, ver) = self
            .wm_base_global
            .ok_or_else(|| resource("wayland: no xdg_wm_base global"))?;
        self.xdg_wm_base = self.bind(name, op::XDG_WM_BASE, ver.min(op::XDG_WM_BASE_VERSION))?;

        // A second roundtrip so wl_shm.format events (sent right after bind) arrive.
        self.roundtrip()?;
        if !self.shm_formats.contains(&op::WL_SHM_FORMAT_XRGB8888) {
            return Err(resource("wayland: compositor does not support XRGB8888 shm"));
        }
        Ok(())
    }

    /// wl_registry.bind(name, interface, version, new_id). The `new_id` arg is *unnamed* in
    /// the protocol, so we marshal the interface string + version + id ourselves (the
    /// standard bind encoding).
    fn bind(&mut self, name: u32, interface: &str, version: u32) -> Result<u32, Error> {
        let id = self.alloc_id();
        let mut m = MessageBuilder::new(self.registry, op::WL_REGISTRY_BIND);
        m.u32(name).string(interface).u32(version).u32(id);
        self.send(&m)?;
        Ok(id)
    }

    /// A `wl_display.sync` roundtrip: block dispatching events until the sync callback's
    /// `done` fires. Since the server processes requests and posts callbacks in order, this
    /// guarantees every earlier event has been dispatched.
    pub fn roundtrip(&mut self) -> Result<(), Error> {
        self.sync_callback = self.alloc_id();
        self.sync_done = false;
        let mut m = MessageBuilder::new(DISPLAY, op::WL_DISPLAY_SYNC);
        m.u32(self.sync_callback);
        self.send(&m)?;
        while !self.sync_done {
            self.read_and_dispatch(true)?;
            if let Some(e) = &self.protocol_error {
                return Err(resource(format!("wayland: protocol error: {e}")));
            }
        }
        self.sync_callback = 0;
        Ok(())
    }

    // --- window creation ---------------------------------------------------------------

    /// Create the surface + xdg_surface + xdg_toplevel, set the title, and commit once so
    /// the compositor sends the initial `configure`. After this returns the window exists
    /// but no buffer is attached yet — call [`present`](Self::present) with the first frame.
    pub fn create_window(&mut self, title: &str) -> Result<(), Error> {
        // wl_compositor.create_surface(new_id surface)
        self.surface = self.alloc_id();
        let mut m = MessageBuilder::new(self.compositor, op::WL_COMPOSITOR_CREATE_SURFACE);
        m.u32(self.surface);
        self.send(&m)?;

        // xdg_wm_base.get_xdg_surface(new_id, surface)
        self.xdg_surface = self.alloc_id();
        let mut m = MessageBuilder::new(self.xdg_wm_base, op::XDG_WM_BASE_GET_XDG_SURFACE);
        m.u32(self.xdg_surface).u32(self.surface);
        self.send(&m)?;

        // xdg_surface.get_toplevel(new_id)
        self.xdg_toplevel = self.alloc_id();
        let mut m = MessageBuilder::new(self.xdg_surface, op::XDG_SURFACE_GET_TOPLEVEL);
        m.u32(self.xdg_toplevel);
        self.send(&m)?;

        // xdg_toplevel.set_title(title)
        let mut m = MessageBuilder::new(self.xdg_toplevel, op::XDG_TOPLEVEL_SET_TITLE);
        m.string(title);
        self.send(&m)?;

        // Commit the surface with no buffer to elicit the first configure, then roundtrip
        // and ack it (spec: xdg_surface — the configure/ack handshake precedes the first
        // buffer commit).
        self.commit_surface()?;
        self.roundtrip()?;
        self.ack_pending_configure()?;
        Ok(())
    }

    fn commit_surface(&mut self) -> Result<(), Error> {
        let m = MessageBuilder::new(self.surface, op::WL_SURFACE_COMMIT);
        self.send(&m)
    }

    /// Ack a pending xdg_surface.configure, if any (spec: xdg_surface.ack_configure).
    fn ack_pending_configure(&mut self) -> Result<(), Error> {
        if let Some(serial) = self.pending_configure.take() {
            let mut m = MessageBuilder::new(self.xdg_surface, op::XDG_SURFACE_ACK_CONFIGURE);
            m.u32(serial);
            self.send(&m)?;
            self.configured = true;
        }
        Ok(())
    }

    // --- swapchain ---------------------------------------------------------------------

    /// (Re)build the shm swapchain for `width`×`height` XRGB8888 with `count` buffers,
    /// growing the memfd/pool as needed (spec: task deliverable 2 — resize on a late
    /// dimension announce). Destroys any prior pool + buffers first.
    fn build_swapchain(&mut self, width: usize, height: usize, count: usize) -> Result<(), Error> {
        // Tear down the old pool + buffers if present.
        if let Some(old) = self.swap.take() {
            for b in &old.buffers {
                let m = MessageBuilder::new(b.wl_buffer, op::WL_BUFFER_DESTROY);
                self.send(&m)?;
            }
            let m = MessageBuilder::new(old.pool, op::WL_SHM_POOL_DESTROY);
            self.send(&m)?;
        }

        let stride = width * 4;
        let buf_size = stride * height;
        let total = buf_size * count;

        let mut memfd = Memfd::new(total)
            .map_err(|e| resource(format!("wayland: memfd_create: {e}")))?;
        memfd
            .resize(total)
            .map_err(|e| resource(format!("wayland: ftruncate: {e}")))?;
        let map = Mmap::new(memfd.as_raw_fd(), total)
            .map_err(|e| resource(format!("wayland: mmap: {e}")))?;

        // wl_shm.create_pool(new_id pool, fd, size) — the fd rides in SCM_RIGHTS.
        let pool = self.alloc_id();
        let mut m = MessageBuilder::new(self.shm, op::WL_SHM_CREATE_POOL);
        m.u32(pool).fd(memfd.as_raw_fd()).i32(total as i32);
        self.send(&m)?;

        // wl_shm_pool.create_buffer for each swap slot.
        let mut buffers = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * buf_size;
            let wl_buffer = self.alloc_id();
            let mut m = MessageBuilder::new(pool, op::WL_SHM_POOL_CREATE_BUFFER);
            m.u32(wl_buffer)
                .i32(offset as i32)
                .i32(width as i32)
                .i32(height as i32)
                .i32(stride as i32)
                .u32(op::WL_SHM_FORMAT_XRGB8888);
            self.send(&m)?;
            buffers.push(SwapBuffer {
                wl_buffer,
                offset,
                available: true,
            });
        }

        self.swap = Some(Swapchain {
            memfd,
            map,
            pool,
            width,
            height,
            stride,
            buffers,
        });
        Ok(())
    }

    /// Ensure the swapchain matches `width`×`height`, (re)building it on first use or a
    /// dimension change. `COUNT` buffers double-buffer (2 is the minimum for tear-free
    /// presentation; a third smooths a slow compositor release).
    pub fn ensure_swapchain(&mut self, width: usize, height: usize) -> Result<(), Error> {
        const COUNT: usize = 3;
        let matches = self
            .swap
            .as_ref()
            .is_some_and(|s| s.width == width && s.height == height);
        if !matches {
            self.build_swapchain(width, height, COUNT)?;
        }
        Ok(())
    }

    /// Present one XRGB8888 frame by drawing it into a free swap buffer and committing.
    /// `draw` fills the buffer's bytes given `(dst, stride, width, height)`. If no buffer is
    /// currently released by the compositor, this pumps events until one frees (bounded by a
    /// roundtrip), so it never busy-spins. Requires the surface to be configured.
    ///
    /// Returns `Ok(true)` if a frame was presented, `Ok(false)` if the window is closing (a
    /// caller then treats further frames as drops).
    pub fn present<F>(&mut self, width: usize, height: usize, draw: F) -> Result<bool, Error>
    where
        F: FnOnce(&mut [u8], usize, usize, usize),
    {
        if self.closed {
            return Ok(false);
        }
        if let Some(e) = &self.protocol_error {
            return Err(resource(format!("wayland: protocol error: {e}")));
        }
        self.ensure_swapchain(width, height)?;
        if !self.configured {
            // Should not happen (create_window acks the first configure), but never attach
            // before configure — pump once and re-check.
            self.pump()?;
            self.ack_pending_configure()?;
        }

        // Find a free buffer, pumping (and roundtripping as a backstop) until one releases.
        let idx = self.acquire_free_buffer()?;
        let (offset, stride, wl_buffer) = {
            let s = self.swap.as_ref().expect("swapchain built");
            (s.buffers[idx].offset, s.stride, s.buffers[idx].wl_buffer)
        };

        // Draw directly into the mapped shm region for this slot.
        {
            let s = self.swap.as_mut().expect("swapchain built");
            let buf_bytes = stride * height;
            let dst = &mut s.map.as_mut()[offset..offset + buf_bytes];
            draw(dst, stride, width, height);
            s.buffers[idx].available = false;
        }

        // attach(buffer, 0, 0); damage_buffer(0,0,w,h); commit.
        let mut m = MessageBuilder::new(self.surface, op::WL_SURFACE_ATTACH);
        m.u32(wl_buffer).i32(0).i32(0);
        self.send(&m)?;
        let mut m = MessageBuilder::new(self.surface, op::WL_SURFACE_DAMAGE_BUFFER);
        m.i32(0).i32(0).i32(width as i32).i32(height as i32);
        self.send(&m)?;
        self.commit_surface()?;

        // Flush any pending server events (ping/pong, release, close) without blocking.
        self.pump()?;
        Ok(!self.closed)
    }

    /// Return the index of a free swap buffer, pumping events (and, as a backstop, doing a
    /// roundtrip) until the compositor releases one.
    fn acquire_free_buffer(&mut self) -> Result<usize, Error> {
        // Fast path: a buffer is already free.
        if let Some(i) = self.free_buffer_index() {
            return Ok(i);
        }
        // Pump non-blocking a few times, then fall back to a blocking roundtrip.
        for _ in 0..2 {
            self.pump()?;
            if let Some(i) = self.free_buffer_index() {
                return Ok(i);
            }
        }
        // Blocking backstop: roundtrip forces the server to flush releases to us.
        self.roundtrip()?;
        if let Some(i) = self.free_buffer_index() {
            return Ok(i);
        }
        // With 2+ buffers and one attach in flight, at least one is always free after a
        // roundtrip; if not, treat as a resource error rather than spin forever.
        Err(resource("wayland: no shm buffer released by the compositor"))
    }

    fn free_buffer_index(&self) -> Option<usize> {
        self.swap
            .as_ref()?
            .buffers
            .iter()
            .position(|b| b.available)
    }

    // --- event pump --------------------------------------------------------------------

    /// True once the compositor asked to close the window.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Read whatever is available (non-blocking) and dispatch it. Call between frames to
    /// answer pings, absorb buffer releases, and notice close/configure. Never blocks.
    pub fn pump(&mut self) -> Result<(), Error> {
        self.read_and_dispatch(false)?;
        // Apply any configure that arrived (keeps the surface valid across resizes).
        self.ack_pending_configure()?;
        Ok(())
    }

    /// Read from the socket (blocking iff `block`) and dispatch every complete message in the
    /// buffer. With `block == false`, an empty socket returns `Ok(())` at once.
    fn read_and_dispatch(&mut self, block: bool) -> Result<(), Error> {
        // Dispatch anything already buffered first, so a blocking caller whose awaited reply
        // (e.g. a sync `done`) is already in `rx` makes progress without blocking on more.
        // If that produced any message, return so the caller re-checks its wait condition
        // before we risk blocking on a further read.
        if self.dispatch_buffered()? > 0 && block {
            return Ok(());
        }
        self.stream
            .set_nonblocking(!block)
            .map_err(|e| resource(format!("wayland: set_nonblocking: {e}")))?;
        let mut tmp = [0u8; 4096];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => {
                    // The compositor closed the connection.
                    return Err(resource("wayland: server closed the connection"));
                }
                Ok(n) => {
                    self.rx.extend_from_slice(&tmp[..n]);
                    // In blocking mode we want just enough to make progress; dispatch what
                    // we have and, if we still need more (blocking roundtrip), loop to read
                    // again. In non-blocking mode, drain then return on WouldBlock.
                    self.dispatch_buffered()?;
                    if block {
                        // Keep reading until the specific condition the caller waits on is
                        // met (sync_done etc.); the caller's loop re-invokes us. One read is
                        // enough per call — return so the caller can re-check its condition.
                        return Ok(());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Nothing (more) to read; dispatch what we buffered and return.
                    self.dispatch_buffered()?;
                    return Ok(());
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(resource(format!("wayland: read: {e}"))),
            }
        }
    }

    /// Dispatch every complete message currently in `self.rx`, leaving any trailing partial
    /// message buffered for the next read. Returns how many messages were dispatched.
    fn dispatch_buffered(&mut self) -> Result<usize, Error> {
        let mut consumed = 0usize;
        let mut count = 0usize;
        while self.rx.len() - consumed >= Header::LEN {
            let header = match Header::parse(&self.rx[consumed..]) {
                Some(h) => h,
                None => break,
            };
            let size = header.size as usize;
            if size < Header::LEN {
                return Err(resource("wayland: message size smaller than header"));
            }
            if self.rx.len() - consumed < size {
                break; // partial message; wait for more bytes
            }
            let body_start = consumed + Header::LEN;
            let body_end = consumed + size;
            // Copy the body out so we don't hold an immutable borrow of self.rx while
            // dispatching (which needs &mut self).
            let body: Vec<u8> = self.rx[body_start..body_end].to_vec();
            self.dispatch_one(header, &body)?;
            consumed += size;
            count += 1;
        }
        if consumed > 0 {
            self.rx.drain(..consumed);
        }
        Ok(count)
    }

    /// Dispatch a single event by (object, opcode). Unknown object/opcode pairs are ignored
    /// (forward-compat: a compositor may send events for capabilities we did not use).
    fn dispatch_one(&mut self, h: Header, body: &[u8]) -> Result<(), Error> {
        let mut r = ArgReader::new(body);
        match h.object {
            DISPLAY => match h.opcode {
                op::WL_DISPLAY_ERROR => {
                    let obj = r.u32().unwrap_or(0);
                    let code = r.u32().unwrap_or(0);
                    let msg = r.string().unwrap_or_default();
                    self.protocol_error =
                        Some(format!("object {obj} code {code}: {msg}"));
                }
                op::WL_DISPLAY_DELETE_ID => {
                    // The server is done with an id; we could recycle it, but our monotonic
                    // allocator never runs out for a single window, so just note it.
                    let _ = r.u32();
                }
                _ => {}
            },
            _ if h.object == self.registry => match h.opcode {
                op::WL_REGISTRY_GLOBAL => {
                    let name = r.u32().ok_or_else(|| resource("wayland: bad global event"))?;
                    let interface = r
                        .string()
                        .ok_or_else(|| resource("wayland: bad global interface"))?;
                    let version = r.u32().unwrap_or(1);
                    match interface.as_str() {
                        op::WL_COMPOSITOR => self.compositor_global = Some((name, version)),
                        op::WL_SHM => self.shm_global = Some((name, version)),
                        op::XDG_WM_BASE => self.wm_base_global = Some((name, version)),
                        _ => {}
                    }
                }
                op::WL_REGISTRY_GLOBAL_REMOVE => {
                    let _ = r.u32();
                }
                _ => {}
            },
            _ if h.object == self.shm => {
                if h.opcode == op::WL_SHM_FORMAT {
                    if let Some(fmt) = r.u32() {
                        self.shm_formats.push(fmt);
                    }
                }
            }
            _ if h.object == self.xdg_wm_base => {
                if h.opcode == op::XDG_WM_BASE_PING {
                    // Answer immediately to prove liveness (spec: xdg_wm_base.ping/pong).
                    let serial = r.u32().ok_or_else(|| resource("wayland: bad ping"))?;
                    let mut m = MessageBuilder::new(self.xdg_wm_base, op::XDG_WM_BASE_PONG);
                    m.u32(serial);
                    self.send(&m)?;
                }
            }
            _ if h.object == self.xdg_surface => {
                if h.opcode == op::XDG_SURFACE_CONFIGURE {
                    let serial = r.u32().ok_or_else(|| resource("wayland: bad configure"))?;
                    self.pending_configure = Some(serial);
                }
            }
            _ if h.object == self.xdg_toplevel => match h.opcode {
                op::XDG_TOPLEVEL_CONFIGURE => {
                    // width, height, states[] — advisory; we keep the video's own size.
                    let _ = (r.i32(), r.i32(), r.array());
                }
                op::XDG_TOPLEVEL_CLOSE => self.closed = true,
                _ => {}
            },
            _ if h.object == self.sync_callback => {
                if h.opcode == op::WL_CALLBACK_DONE {
                    self.sync_done = true;
                }
            }
            _ => {
                // Might be a wl_buffer.release for one of our swap buffers.
                if let Some(swap) = self.swap.as_mut() {
                    if h.opcode == op::WL_BUFFER_RELEASE {
                        if let Some(b) =
                            swap.buffers.iter_mut().find(|b| b.wl_buffer == h.object)
                        {
                            b.available = true;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Destroy the toplevel/surface and pool cleanly on shutdown. Best-effort — errors are
    /// swallowed since the connection is going away anyway.
    pub fn destroy(&mut self) {
        if let Some(swap) = self.swap.take() {
            for b in &swap.buffers {
                let m = MessageBuilder::new(b.wl_buffer, op::WL_BUFFER_DESTROY);
                let _ = self.send(&m);
            }
            let m = MessageBuilder::new(swap.pool, op::WL_SHM_POOL_DESTROY);
            let _ = self.send(&m);
            // Drop the map/memfd explicitly (closes the fd).
            drop(swap.map);
            drop(swap.memfd);
        }
        let _ = self.stream.flush();
    }
}

impl Drop for WaylandClient {
    fn drop(&mut self) {
        self.destroy();
    }
}

/// Resolve the display socket path from the environment: `$WAYLAND_DISPLAY` if absolute,
/// else `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY` (default `wayland-0`). Returns
/// [`Error::Resource`] if `$XDG_RUNTIME_DIR` is needed but unset.
fn socket_path() -> Result<PathBuf, Error> {
    let display = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".to_string());
    let p = PathBuf::from(&display);
    if p.is_absolute() {
        return Ok(p);
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .map_err(|_| resource("wayland: XDG_RUNTIME_DIR unset and WAYLAND_DISPLAY not absolute"))?;
    let mut path = PathBuf::from(runtime);
    path.push(display);
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_path_prefers_absolute_wayland_display() {
        // Absolute path is used verbatim.
        // (We mutate process env under a serial guard-free test; these vars are process-wide
        // but the assertions read the resolved path immediately.)
        std::env::set_var("WAYLAND_DISPLAY", "/tmp/sc-abs.sock");
        let p = socket_path().unwrap();
        assert_eq!(p, PathBuf::from("/tmp/sc-abs.sock"));
        std::env::remove_var("WAYLAND_DISPLAY");
    }

    #[test]
    fn socket_path_joins_runtime_dir_for_relative_name() {
        std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1000");
        std::env::set_var("WAYLAND_DISPLAY", "wayland-1");
        let p = socket_path().unwrap();
        assert_eq!(p, PathBuf::from("/run/user/1000/wayland-1"));
        std::env::remove_var("WAYLAND_DISPLAY");
        std::env::remove_var("XDG_RUNTIME_DIR");
    }
}
