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

/// One imported dmabuf-backed buffer in the GPU present path (spec: task deliverable 3 —
/// present through zwp_linux_dmabuf_v1 with `wl_buffer.release`-gated reuse). The `wl_buffer`
/// wraps a GPU image the renderer exported as a dma-buf fd; we track only the handle and
/// whether the compositor currently holds it. Reuse is identical to the shm swapchain:
/// available → the renderer may draw into the underlying image and re-present it.
struct DmabufBuffer {
    wl_buffer: u32,
    /// The caller's opaque slot index (which GPU image this `wl_buffer` wraps), echoed back
    /// so the renderer knows which image freed.
    slot: u32,
    /// `true` while free for us to render into; `false` between `attach`+`commit` and the
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
    /// zwp_linux_dmabuf_v1, bound only if the compositor advertises it (0 = absent). The
    /// shm path never touches this — dmabuf import is a strictly additive capability.
    dmabuf: u32,

    // Registry discovery: (name, version) for each global we care about, learned from
    // wl_registry.global before we bind.
    compositor_global: Option<(u32, u32)>,
    shm_global: Option<(u32, u32)>,
    wm_base_global: Option<(u32, u32)>,
    dmabuf_global: Option<(u32, u32)>,

    /// wl_shm formats the server advertised (we require XRGB8888).
    shm_formats: Vec<u32>,
    /// (DRM fourcc, DRM format modifier) pairs the compositor advertised it can import via
    /// linux-dmabuf (`zwp_linux_dmabuf_v1.modifier`). Empty if dmabuf is absent or the
    /// compositor only sent the legacy modifier-less `format` events.
    dmabuf_modifiers: Vec<(u32, u64)>,

    /// Imported dmabuf-backed buffers, keyed by their `wl_buffer` id. Each carries the same
    /// compositor-holds-it / free-to-reuse flag as an shm [`SwapBuffer`], gated by
    /// `wl_buffer.release`. The GPU owns the underlying image memory; we only track the
    /// `wl_buffer` handle and its availability.
    dmabuf_buffers: Vec<DmabufBuffer>,
    /// Result of the most recent `create_immed`/`create` params import, learned from a
    /// `zwp_linux_buffer_params_v1.failed` event (async failure; `create_immed` success is
    /// silent). `true` means the last import failed.
    dmabuf_import_failed: bool,
    /// Object ids of `zwp_linux_buffer_params_v1` objects we created and have not yet
    /// destroyed, so a late `failed`/`created` event is routed to the dmabuf logic (and not
    /// mistaken for a `wl_buffer.release`, since opcode 0/1 overlap across interfaces).
    pending_params: Vec<u32>,

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
            dmabuf: 0,
            compositor_global: None,
            shm_global: None,
            wm_base_global: None,
            dmabuf_global: None,
            shm_formats: Vec::new(),
            dmabuf_modifiers: Vec::new(),
            dmabuf_buffers: Vec::new(),
            dmabuf_import_failed: false,
            pending_params: Vec::new(),
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

        // linux-dmabuf is *optional*: bind it only if advertised, so a compositor without it
        // still gives us a working shm sink. The GPU sink queries `has_dmabuf()` and falls
        // back to waylandvideosink when it is absent.
        if let Some((name, ver)) = self.dmabuf_global {
            self.dmabuf = self.bind(name, op::ZWP_LINUX_DMABUF, ver.min(op::ZWP_LINUX_DMABUF_VERSION))?;
        }

        // A second roundtrip so wl_shm.format events (and the dmabuf format/modifier
        // advertisements, sent right after their respective binds) arrive.
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

    // --- linux-dmabuf import + present (GPU path; spec: task deliverable 3) -------------
    //
    // This is the additive presenter the sc-vk renderer drives: it renders into an exported
    // dma-buf (a GPU image), imports each dma-buf fd once as a `wl_buffer` via
    // zwp_linux_dmabuf_v1, then presents them with the same `wl_buffer.release`-gated reuse
    // as the shm swapchain. The shm path above is untouched.

    /// Whether the compositor advertised linux-dmabuf and we bound it. The GPU sink checks
    /// this and falls back to `waylandvideosink` when it is `false`.
    pub fn has_dmabuf(&self) -> bool {
        self.dmabuf != 0
    }

    /// The (DRM fourcc, DRM format modifier) pairs the compositor said it can import. Empty
    /// if dmabuf is absent. Exposed so the renderer can log/negotiate modifier support.
    pub fn dmabuf_modifiers(&self) -> &[(u32, u64)] {
        &self.dmabuf_modifiers
    }

    /// Whether the compositor can import `format` under `modifier` (an exact membership test
    /// against the advertised set). v1 imports `DRM_FORMAT_XRGB8888` under
    /// `DRM_FORMAT_MOD_LINEAR`, so the sink pre-flights that pair before importing.
    pub fn dmabuf_supports(&self, format: u32, modifier: u64) -> bool {
        self.dmabuf_modifiers
            .iter()
            .any(|&(f, m)| f == format && m == modifier)
    }

    /// Record an advertised (fourcc, modifier) pair, de-duplicated. `INVALID` modifiers
    /// (all-ones) mean "implementation-defined"; we ignore those and only keep concrete
    /// values we can actually request (LINEAR in v1).
    fn record_dmabuf_modifier(&mut self, format: u32, modifier: u64) {
        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
        if modifier == DRM_FORMAT_MOD_INVALID {
            return;
        }
        if !self.dmabuf_modifiers.contains(&(format, modifier)) {
            self.dmabuf_modifiers.push((format, modifier));
        }
    }

    fn is_pending_params(&self, id: u32) -> bool {
        self.pending_params.contains(&id)
    }

    fn drop_pending_params(&mut self, id: u32) {
        self.pending_params.retain(|&p| p != id);
    }

    /// Import one exported dma-buf plane as a `wl_buffer` and register it under the caller's
    /// opaque `slot` index (which GPU image it wraps), so a later `wl_buffer.release` tells
    /// the renderer that image is free to redraw. A single 32-bit XRGB plane is the v1 case;
    /// `offset`/`stride`/`modifier` come from the Vulkan subresource layout + allocation.
    ///
    /// Uses `create_params` → `add` (fd via SCM_RIGHTS) → `create_immed`: the `wl_buffer` is
    /// live synchronously on success. A `zwp_linux_buffer_params_v1.failed` (async) marks the
    /// import failed; the caller roundtrips and checks [`take_dmabuf_import_failed`].
    ///
    /// The compositor duplicates the fd on receipt; the caller keeps ownership of its own fd.
    ///
    /// [`take_dmabuf_import_failed`]: Self::take_dmabuf_import_failed
    #[allow(clippy::too_many_arguments)]
    pub fn import_dmabuf(
        &mut self,
        slot: u32,
        width: usize,
        height: usize,
        format: u32,
        modifier: u64,
        fd: std::os::unix::io::RawFd,
        offset: u32,
        stride: u32,
    ) -> Result<u32, Error> {
        if self.dmabuf == 0 {
            return Err(resource("wayland: compositor has no linux-dmabuf support"));
        }

        // zwp_linux_dmabuf_v1.create_params(new_id params)
        let params = self.alloc_id();
        let mut m = MessageBuilder::new(self.dmabuf, op::ZWP_LINUX_DMABUF_CREATE_PARAMS);
        m.u32(params);
        self.send(&m)?;
        self.pending_params.push(params);

        // zwp_linux_buffer_params_v1.add(fd, plane_idx=0, offset, stride, mod_hi, mod_lo).
        // The fd rides in SCM_RIGHTS ancillary data, exactly like wl_shm.create_pool.
        let mod_hi = (modifier >> 32) as u32;
        let mod_lo = (modifier & 0xffff_ffff) as u32;
        let mut m = MessageBuilder::new(params, op::ZWP_LINUX_BUFFER_PARAMS_ADD);
        m.fd(fd)
            .u32(0) // plane_idx
            .u32(offset)
            .u32(stride)
            .u32(mod_hi)
            .u32(mod_lo);
        self.send(&m)?;

        // zwp_linux_buffer_params_v1.create_immed(new_id wl_buffer, w, h, format, flags=0).
        let wl_buffer = self.alloc_id();
        let mut m = MessageBuilder::new(params, op::ZWP_LINUX_BUFFER_PARAMS_CREATE_IMMED);
        m.u32(wl_buffer)
            .i32(width as i32)
            .i32(height as i32)
            .u32(format)
            .u32(0); // flags: no y_invert
        self.send(&m)?;

        self.dmabuf_buffers.push(DmabufBuffer {
            wl_buffer,
            slot,
            available: true,
        });
        Ok(wl_buffer)
    }

    /// Roundtrip and report whether the most recent import failed (a `params.failed` event),
    /// clearing the flag. Called after importing all buffers to detect a rejected format.
    pub fn take_dmabuf_import_failed(&mut self) -> Result<bool, Error> {
        self.roundtrip()?;
        // Destroy any params objects that already answered (or that create_immed consumed);
        // the wl_buffer outlives the params object, so this is always safe post-create.
        let stale: Vec<u32> = self.pending_params.clone();
        for p in stale {
            let m = MessageBuilder::new(p, op::ZWP_LINUX_BUFFER_PARAMS_DESTROY);
            let _ = self.send(&m);
            self.drop_pending_params(p);
        }
        Ok(std::mem::take(&mut self.dmabuf_import_failed))
    }

    /// Return the `slot` of a dmabuf buffer the compositor has released (free to redraw),
    /// pumping events (then a blocking roundtrip backstop) until one frees. Mirrors
    /// [`acquire_free_buffer`](Self::acquire_free_buffer) for the shm path.
    fn acquire_free_dmabuf(&mut self) -> Result<u32, Error> {
        if let Some(s) = self.free_dmabuf_slot() {
            return Ok(s);
        }
        for _ in 0..2 {
            self.pump()?;
            if let Some(s) = self.free_dmabuf_slot() {
                return Ok(s);
            }
        }
        self.roundtrip()?;
        if let Some(s) = self.free_dmabuf_slot() {
            return Ok(s);
        }
        Err(resource("wayland: no dmabuf buffer released by the compositor"))
    }

    fn free_dmabuf_slot(&self) -> Option<u32> {
        self.dmabuf_buffers
            .iter()
            .find(|b| b.available)
            .map(|b| b.slot)
    }

    /// Acquire a free dmabuf slot for the renderer to draw into, returning its `slot` index.
    /// The caller renders the frame into that slot's GPU image, then calls
    /// [`present_dmabuf`](Self::present_dmabuf) with the same slot. Returns `Ok(None)` when
    /// the window is closing (the caller then treats frames as drops).
    pub fn acquire_dmabuf_slot(&mut self) -> Result<Option<u32>, Error> {
        if self.closed {
            return Ok(None);
        }
        if let Some(e) = &self.protocol_error {
            return Err(resource(format!("wayland: protocol error: {e}")));
        }
        Ok(Some(self.acquire_free_dmabuf()?))
    }

    /// Present the dmabuf buffer registered for `slot` (attach + damage_buffer + commit),
    /// after the renderer has finished drawing into and syncing it. Marks the slot busy until
    /// the compositor's `wl_buffer.release`. Returns `Ok(false)` if the window is closing.
    pub fn present_dmabuf(&mut self, slot: u32, width: usize, height: usize) -> Result<bool, Error> {
        if self.closed {
            return Ok(false);
        }
        if let Some(e) = &self.protocol_error {
            return Err(resource(format!("wayland: protocol error: {e}")));
        }
        if !self.configured {
            self.pump()?;
            self.ack_pending_configure()?;
        }
        let wl_buffer = {
            let b = self
                .dmabuf_buffers
                .iter_mut()
                .find(|b| b.slot == slot)
                .ok_or_else(|| resource("wayland: present_dmabuf: unknown slot"))?;
            b.available = false;
            b.wl_buffer
        };

        // attach(buffer, 0, 0); damage_buffer(0,0,w,h); commit — identical to the shm path.
        let mut m = MessageBuilder::new(self.surface, op::WL_SURFACE_ATTACH);
        m.u32(wl_buffer).i32(0).i32(0);
        self.send(&m)?;
        let mut m = MessageBuilder::new(self.surface, op::WL_SURFACE_DAMAGE_BUFFER);
        m.i32(0).i32(0).i32(width as i32).i32(height as i32);
        self.send(&m)?;
        self.commit_surface()?;

        self.pump()?;
        Ok(!self.closed)
    }

    /// Destroy all imported dmabuf `wl_buffer`s (e.g. before re-importing at a new size).
    /// The GPU image memory is the renderer's to free; this only releases the wire objects.
    pub fn clear_dmabuf_buffers(&mut self) {
        for b in std::mem::take(&mut self.dmabuf_buffers) {
            let m = MessageBuilder::new(b.wl_buffer, op::WL_BUFFER_DESTROY);
            let _ = self.send(&m);
        }
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
                        op::ZWP_LINUX_DMABUF => self.dmabuf_global = Some((name, version)),
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
            _ if self.dmabuf != 0 && h.object == self.dmabuf => match h.opcode {
                // Legacy modifier-less advertisement: record the fourcc under the LINEAR
                // modifier, since a format-only advertisement implies LINEAR is importable.
                op::ZWP_LINUX_DMABUF_FORMAT => {
                    if let Some(fmt) = r.u32() {
                        self.record_dmabuf_modifier(fmt, op::DRM_FORMAT_MOD_LINEAR);
                    }
                }
                // (fourcc, modifier) pair — the modifier is a 64-bit value split hi/lo.
                op::ZWP_LINUX_DMABUF_MODIFIER => {
                    if let (Some(fmt), Some(hi), Some(lo)) = (r.u32(), r.u32(), r.u32()) {
                        let modifier = ((hi as u64) << 32) | lo as u64;
                        self.record_dmabuf_modifier(fmt, modifier);
                    }
                }
                _ => {}
            },
            // A zwp_linux_buffer_params_v1 we created is answering create_immed. On success
            // create_immed is silent (the wl_buffer is live already); only `failed` arrives.
            _ if self.is_pending_params(h.object) => match h.opcode {
                op::ZWP_LINUX_BUFFER_PARAMS_FAILED => {
                    self.dmabuf_import_failed = true;
                    self.drop_pending_params(h.object);
                }
                op::ZWP_LINUX_BUFFER_PARAMS_CREATED => {
                    // Not our normal path (we use create_immed), but handle it: the buffer
                    // is already recorded by the caller keyed on the id it allocated.
                    self.drop_pending_params(h.object);
                }
                _ => {}
            },
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
                // Might be a wl_buffer.release for one of our swap buffers (shm) or an
                // imported dmabuf buffer — the release-gated reuse is identical for both.
                if h.opcode == op::WL_BUFFER_RELEASE {
                    if let Some(swap) = self.swap.as_mut() {
                        if let Some(b) =
                            swap.buffers.iter_mut().find(|b| b.wl_buffer == h.object)
                        {
                            b.available = true;
                        }
                    }
                    if let Some(b) = self
                        .dmabuf_buffers
                        .iter_mut()
                        .find(|b| b.wl_buffer == h.object)
                    {
                        b.available = true;
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
        // Release any imported dmabuf wl_buffers (the GPU image memory is freed by the
        // renderer that owns it — this only tears down the wire objects).
        self.clear_dmabuf_buffers();
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
    fn socket_path_resolution() {
        // ONE test, not two: `WAYLAND_DISPLAY`/`XDG_RUNTIME_DIR` are process-wide, and
        // cargo runs a binary's tests on parallel threads — two tests mutating the
        // same env vars raced (an intermittent workspace-run failure). Sequential
        // assertions in a single body cannot.
        std::env::set_var("WAYLAND_DISPLAY", "/tmp/sc-abs.sock");
        let p = socket_path().unwrap();
        assert_eq!(p, PathBuf::from("/tmp/sc-abs.sock"), "absolute display used verbatim");

        std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1000");
        std::env::set_var("WAYLAND_DISPLAY", "wayland-1");
        let p = socket_path().unwrap();
        assert_eq!(
            p,
            PathBuf::from("/run/user/1000/wayland-1"),
            "relative display joins the runtime dir"
        );
        std::env::remove_var("WAYLAND_DISPLAY");
        std::env::remove_var("XDG_RUNTIME_DIR");
    }
}
