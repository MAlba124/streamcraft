//! Wayland interface names, request opcodes, and the event opcodes we act on — transcribed
//! from the stable protocol XML (`wayland.xml`, `xdg-shell.xml`). Only what the presenter
//! uses is listed; opcodes are interface-relative and stable across versions for the requests
//! we send. Grouped by interface so each `mod` reads like the XML.

/// `wl_display` — the id-1 singleton (protocol spec §wl_display).
pub mod display {
    pub const SYNC: u16 = 0; // -> new_id wl_callback
    pub const GET_REGISTRY: u16 = 1; // -> new_id wl_registry
    pub const EV_ERROR: u16 = 0; // object_id, code, message
    pub const EV_DELETE_ID: u16 = 1; // id
}

/// `wl_registry` — the global advertisement/bind object.
pub mod registry {
    pub const BIND: u16 = 0; // name, new_id(interface, version, id)
    pub const EV_GLOBAL: u16 = 0; // name, interface, version
    pub const EV_GLOBAL_REMOVE: u16 = 1; // name
}

/// `wl_callback` — one-shot done notifier (`sync`, `frame`).
pub mod callback {
    pub const EV_DONE: u16 = 0; // callback_data
}

/// `wl_compositor` — surface/region factory.
pub mod compositor {
    pub const NAME: &str = "wl_compositor";
    pub const CREATE_SURFACE: u16 = 0; // -> new_id wl_surface
}

/// `wl_shm` — shared-memory buffer factory.
pub mod shm {
    pub const NAME: &str = "wl_shm";
    pub const CREATE_POOL: u16 = 0; // new_id, fd, size
    pub const EV_FORMAT: u16 = 0; // format
    /// 32-bit ARGB, premultiplied — what the GUI subsurface uses (alpha over video).
    pub const FORMAT_ARGB8888: u32 = 0;
    /// 32-bit XRGB (opaque) — a solid fill with no transparency.
    pub const FORMAT_XRGB8888: u32 = 1;
}

/// `wl_shm_pool` — a mmap'd region carved into buffers.
pub mod shm_pool {
    pub const CREATE_BUFFER: u16 = 0; // new_id, offset, width, height, stride, format
    pub const DESTROY: u16 = 1;
    pub const RESIZE: u16 = 2; // size
}

/// `wl_surface` — a rectangle of pixels the compositor composites.
pub mod surface {
    pub const DESTROY: u16 = 0;
    pub const ATTACH: u16 = 1; // buffer, x, y
    pub const DAMAGE: u16 = 2; // x, y, w, h
    pub const FRAME: u16 = 3; // -> new_id wl_callback
    pub const COMMIT: u16 = 6;
    pub const SET_BUFFER_SCALE: u16 = 8; // scale
    pub const DAMAGE_BUFFER: u16 = 9; // x, y, w, h (buffer coords — preferred)
}

/// `wl_buffer` — a committed pixel source; `release` says the compositor is done reading it.
pub mod buffer {
    pub const DESTROY: u16 = 0;
    pub const EV_RELEASE: u16 = 0;
}

/// `wl_subcompositor` / `wl_subsurface` — the GUI-over-video layering.
pub mod subcompositor {
    pub const NAME: &str = "wl_subcompositor";
    pub const GET_SUBSURFACE: u16 = 1; // new_id, surface, parent
}
pub mod subsurface {
    pub const SET_POSITION: u16 = 1; // x, y
    pub const PLACE_ABOVE: u16 = 2; // sibling
    pub const SET_DESYNC: u16 = 5;
}

/// `xdg_wm_base` — the desktop window-management protocol root.
pub mod wm_base {
    pub const NAME: &str = "xdg_wm_base";
    pub const GET_XDG_SURFACE: u16 = 2; // new_id, surface
    pub const PONG: u16 = 3; // serial
    pub const EV_PING: u16 = 0; // serial
}

/// `xdg_surface` — the window role adapter over a `wl_surface`.
pub mod xdg_surface {
    pub const GET_TOPLEVEL: u16 = 1; // -> new_id xdg_toplevel
    pub const ACK_CONFIGURE: u16 = 4; // serial
    pub const EV_CONFIGURE: u16 = 0; // serial
}

/// `xdg_toplevel` — an application top-level window.
pub mod xdg_toplevel {
    pub const SET_TITLE: u16 = 2; // title
    pub const SET_APP_ID: u16 = 3; // app_id
    pub const EV_CONFIGURE: u16 = 0; // width, height, states(array)
    pub const EV_CLOSE: u16 = 1;
}

/// `wl_seat` — input device hub (pointer/keyboard for controls).
pub mod seat {
    pub const NAME: &str = "wl_seat";
    pub const GET_KEYBOARD: u16 = 1; // -> new_id wl_keyboard
    pub const EV_CAPABILITIES: u16 = 0; // capabilities (bitmask)
    pub const EV_NAME: u16 = 1; // name
    /// Capability bit: a keyboard is available.
    pub const CAP_KEYBOARD: u32 = 2;
}

/// `wl_keyboard` — key events. Keys are raw Linux evdev codes (`<linux/input-event-codes.h>`):
/// `KEY_ESC = 1`, `KEY_Q = 16`.
pub mod keyboard {
    pub const EV_KEY: u16 = 3; // serial, time, key, state
    /// `state == 1` ⇒ pressed.
    pub const STATE_PRESSED: u32 = 1;
    pub const KEY_ESC: u32 = 1;
    pub const KEY_Q: u32 = 16;
}

/// `zwp_linux_dmabuf_v1` — imports a VA-API decode surface's dmabuf as a `wl_buffer` (zero-copy
/// video).
pub mod dmabuf {
    pub const NAME: &str = "zwp_linux_dmabuf_v1";
    pub const CREATE_PARAMS: u16 = 1; // -> new_id zwp_linux_buffer_params_v1
}

/// `zwp_linux_buffer_params_v1` — accumulates dmabuf planes, then mints a `wl_buffer`.
pub mod dmabuf_params {
    pub const DESTROY: u16 = 0;
    /// `add(fd, plane_idx, offset, stride, modifier_hi, modifier_lo)`.
    pub const ADD: u16 = 1;
    /// `create_immed(new_id buffer, width, height, format, flags)` — synchronous (no
    /// `created`/`failed` round-trip); available since interface version 2.
    pub const CREATE_IMMED: u16 = 3;
    pub const EV_CREATED: u16 = 0; // buffer  (async `create` path — unused)
    pub const EV_FAILED: u16 = 1;
}

/// `wp_viewporter` / `wp_viewport` — crops (the coded surface → the display rect) and scales
/// (→ the window size), so a padded NV12 decode surface displays correctly.
pub mod viewporter {
    pub const NAME: &str = "wp_viewporter";
    pub const GET_VIEWPORT: u16 = 1; // new_id, surface
}
pub mod viewport {
    /// `set_source(x, y, width, height)` — all `wl_fixed` (crop rect in surface coords).
    pub const SET_SOURCE: u16 = 1;
    /// `set_destination(width, height)` — integers (the on-screen size).
    pub const SET_DESTINATION: u16 = 2;
}

/// DRM FourCC + flag constants for the dmabuf import (matches `<drm_fourcc.h>`).
pub mod drm {
    /// `DRM_FORMAT_MOD_INVALID` — "no explicit modifier"; the importer omits the modifier hint.
    pub const MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
}
