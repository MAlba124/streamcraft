//! Interface names, request opcodes, event opcodes and enum constants for the exact
//! Wayland subset a video window needs (spec: `wayland/spec/wayland.xml`,
//! `wayland/spec/xdg-shell.xml`). Opcodes are the 0-based declaration index of a
//! `<request>` / `<event>` within its `<interface>` — the wire opcode. Each constant cites
//! `interface.request`/`interface.event` so it can be checked against the spec at a glance.
//!
//! We bind only the objects and versions we drive: `wl_display` (1), `wl_registry` (1),
//! `wl_compositor` (≥4 for `damage_buffer`; we bind up to 6), `wl_shm` (1), `wl_shm_pool`
//! (1), `wl_buffer` (1), `wl_surface` (from `wl_compositor`), `xdg_wm_base` (≥1), and its
//! `xdg_surface` / `xdg_toplevel`.

// --- interface names (for wl_registry.global matching) --------------------------------

pub const WL_COMPOSITOR: &str = "wl_compositor";
pub const WL_SHM: &str = "wl_shm";
pub const XDG_WM_BASE: &str = "xdg_wm_base";

/// The wl_display object always has id 1 (spec: wayland.xml — "The server-side resource of
/// [the display] is always allocated the object ID 1").
pub const WL_DISPLAY_ID: u32 = 1;

// The versions we bind globals at. `wl_compositor` v4 first exposes `wl_surface.damage_buffer`
// (opcode 9), which we use, so bind at least v4; v6 is the current stable. `wl_shm`/`xdg_wm_base`
// low versions suffice for our subset.
pub const WL_COMPOSITOR_VERSION: u32 = 4;
pub const WL_SHM_VERSION: u32 = 1;
pub const XDG_WM_BASE_VERSION: u32 = 1;

// --- wl_display (v1) ------------------------------------------------------------------

/// wl_display.sync(callback: new_id<wl_callback>) — request 0.
pub const WL_DISPLAY_SYNC: u16 = 0;
/// wl_display.get_registry(registry: new_id<wl_registry>) — request 1.
pub const WL_DISPLAY_GET_REGISTRY: u16 = 1;
/// wl_display.error(object_id, code, message) — event 0.
pub const WL_DISPLAY_ERROR: u16 = 0;
/// wl_display.delete_id(id) — event 1. The server acknowledges an object we may reuse.
pub const WL_DISPLAY_DELETE_ID: u16 = 1;

// --- wl_registry (v1) -----------------------------------------------------------------

/// wl_registry.bind(name, id: new_id) — request 0. The `new_id` is unnamed, so per the
/// wire rules we marshal the interface string + version + the new object id ourselves.
pub const WL_REGISTRY_BIND: u16 = 0;
/// wl_registry.global(name, interface, version) — event 0.
pub const WL_REGISTRY_GLOBAL: u16 = 0;
/// wl_registry.global_remove(name) — event 1.
pub const WL_REGISTRY_GLOBAL_REMOVE: u16 = 1;

// --- wl_callback (v1) -----------------------------------------------------------------

/// wl_callback.done(callback_data) — event 0. Used for both wl_display.sync roundtrips and
/// wl_surface.frame throttling callbacks.
pub const WL_CALLBACK_DONE: u16 = 0;

// --- wl_compositor (v6) ---------------------------------------------------------------

/// wl_compositor.create_surface(id: new_id<wl_surface>) — request 0.
pub const WL_COMPOSITOR_CREATE_SURFACE: u16 = 0;

// --- wl_shm (v1) ----------------------------------------------------------------------

/// wl_shm.create_pool(id: new_id<wl_shm_pool>, fd, size) — request 0. `fd` rides in the
/// SCM_RIGHTS ancillary data, not the body.
pub const WL_SHM_CREATE_POOL: u16 = 0;
/// wl_shm.format(format) — event 0. The server advertises each supported buffer format.
pub const WL_SHM_FORMAT: u16 = 0;

/// wl_shm.format enum: `xrgb8888 = 1` (spec: wayland.xml wl_shm/format). The two special
/// values `argb8888 = 0` / `xrgb8888 = 1` are guaranteed present; all others are DRM
/// fourcc codes. We present opaque frames, so we use XRGB (the alpha byte is ignored).
pub const WL_SHM_FORMAT_XRGB8888: u32 = 1;
/// wl_shm.format enum: `argb8888 = 0`.
pub const WL_SHM_FORMAT_ARGB8888: u32 = 0;

// --- wl_shm_pool (v1) -----------------------------------------------------------------

/// wl_shm_pool.create_buffer(id, offset, width, height, stride, format) — request 0.
pub const WL_SHM_POOL_CREATE_BUFFER: u16 = 0;
/// wl_shm_pool.destroy() — request 1.
pub const WL_SHM_POOL_DESTROY: u16 = 1;
/// wl_shm_pool.resize(size) — request 2. Grows the pool when a larger frame is announced.
pub const WL_SHM_POOL_RESIZE: u16 = 2;

// --- wl_buffer (v1) -------------------------------------------------------------------

/// wl_buffer.destroy() — request 0.
pub const WL_BUFFER_DESTROY: u16 = 0;
/// wl_buffer.release() — event 0. The compositor is done reading this buffer; we may reuse
/// its shm storage.
pub const WL_BUFFER_RELEASE: u16 = 0;

// --- wl_surface (v6) ------------------------------------------------------------------

/// wl_surface.attach(buffer: object<wl_buffer>, x, y) — request 1.
pub const WL_SURFACE_ATTACH: u16 = 1;
/// wl_surface.damage(x, y, width, height) — request 2 (surface-local coords).
pub const WL_SURFACE_DAMAGE: u16 = 2;
/// wl_surface.frame(callback: new_id<wl_callback>) — request 3. A throttling hint: the
/// server signals `done` when the surface should next be drawn.
pub const WL_SURFACE_FRAME: u16 = 3;
/// wl_surface.commit() — request 6. Atomically applies the pending attach+damage.
pub const WL_SURFACE_COMMIT: u16 = 6;
/// wl_surface.damage_buffer(x, y, width, height) — request 9 (buffer-relative coords;
/// wl_compositor v4+). The modern damage request; buffer pixels, not surface units.
pub const WL_SURFACE_DAMAGE_BUFFER: u16 = 9;

// --- xdg_wm_base (v7) -----------------------------------------------------------------

/// xdg_wm_base.get_xdg_surface(id: new_id<xdg_surface>, surface: object<wl_surface>) —
/// request 2.
pub const XDG_WM_BASE_GET_XDG_SURFACE: u16 = 2;
/// xdg_wm_base.pong(serial) — request 3. Answers a ping to prove liveness.
pub const XDG_WM_BASE_PONG: u16 = 3;
/// xdg_wm_base.ping(serial) — event 0.
pub const XDG_WM_BASE_PING: u16 = 0;

// --- xdg_surface (v7) -----------------------------------------------------------------

/// xdg_surface.get_toplevel(id: new_id<xdg_toplevel>) — request 1.
pub const XDG_SURFACE_GET_TOPLEVEL: u16 = 1;
/// xdg_surface.ack_configure(serial) — request 4. Must precede the first buffer commit.
pub const XDG_SURFACE_ACK_CONFIGURE: u16 = 4;
/// xdg_surface.configure(serial) — event 0. The client acks then commits.
pub const XDG_SURFACE_CONFIGURE: u16 = 0;

// --- xdg_toplevel (v7) ----------------------------------------------------------------

/// xdg_toplevel.set_title(title: string) — request 2.
pub const XDG_TOPLEVEL_SET_TITLE: u16 = 2;
/// xdg_toplevel.configure(width, height, states: array) — event 0. A suggested size (0 =
/// "you choose").
pub const XDG_TOPLEVEL_CONFIGURE: u16 = 0;
/// xdg_toplevel.close() — event 1. The user asked to close the window.
pub const XDG_TOPLEVEL_CLOSE: u16 = 1;
