//! Phase-1 spike: bring up a real `xdg_toplevel` window over the raw wire client and paint it a
//! solid colour from a `wl_shm` buffer, driven by a time-bounded event loop. Proves the wire
//! codec + `SCM_RIGHTS` fd passing + the registry/xdg/shm handshake against a live compositor.
//!
//! Run:  `WAYLAND_DISPLAY=wayland-1 cargo run -p sc-present --example wl_spike`

#![allow(unsafe_code)]

use std::io;
use std::os::unix::io::RawFd;

use sc_present::conn::{Connection, DISPLAY_ID};
use sc_present::protocol as p;

const W: i32 = 640;
const H: i32 = 360;

/// Object ids we create + the globals we bind, tracked so the event loop can route by id.
#[derive(Default)]
struct State {
    registry: u32,
    compositor: u32,
    shm: u32,
    wm_base: u32,
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    // bound-global names discovered from the registry:
    g_compositor: Option<(u32, u32)>, // (name, version)
    g_shm: Option<(u32, u32)>,
    g_wm_base: Option<(u32, u32)>,
    painted: bool,
    should_close: bool,
    frames: u32,
}

fn main() -> io::Result<()> {
    let mut c = Connection::connect()?;
    let mut s = State::default();

    // wl_display.get_registry, then a sync to know when every global has been advertised.
    s.registry = c.alloc_id();
    c.request(DISPLAY_ID, p::display::GET_REGISTRY).u32(s.registry).finish();
    let sync_cb = c.alloc_id();
    c.request(DISPLAY_ID, p::display::SYNC).u32(sync_cb).finish();
    c.flush()?;

    // Pump until the sync callback fires — the registry is fully enumerated by then.
    let mut enumerated = false;
    while !enumerated {
        if !c.recv()? {
            return Err(io::Error::other("compositor closed during enumeration"));
        }
        while let Some((obj, op)) = c.next_event() {
            if obj == s.registry && op == p::registry::EV_GLOBAL {
                let mut a = c.args();
                let name = a.u32().unwrap_or(0);
                let iface = a.string().unwrap_or("").to_owned();
                let version = a.u32().unwrap_or(0);
                match iface.as_str() {
                    p::compositor::NAME => s.g_compositor = Some((name, version)),
                    p::shm::NAME => s.g_shm = Some((name, version)),
                    p::wm_base::NAME => s.g_wm_base = Some((name, version)),
                    _ => {}
                }
            } else if obj == sync_cb && op == p::callback::EV_DONE {
                enumerated = true;
            } else if obj == DISPLAY_ID && op == p::display::EV_ERROR {
                return Err(display_error(&c));
            }
        }
    }

    // Bind the three globals we need (clamp versions to what we support).
    let (cname, cver) = s.g_compositor.ok_or_else(|| io::Error::other("no wl_compositor"))?;
    s.compositor = c.alloc_id();
    c.request(s.registry, p::registry::BIND)
        .u32(cname)
        .bind_new_id(p::compositor::NAME, cver.min(4), s.compositor)
        .finish();

    let (sname, sver) = s.g_shm.ok_or_else(|| io::Error::other("no wl_shm"))?;
    s.shm = c.alloc_id();
    c.request(s.registry, p::registry::BIND)
        .u32(sname)
        .bind_new_id(p::shm::NAME, sver.min(1), s.shm)
        .finish();

    let (wname, wver) = s.g_wm_base.ok_or_else(|| io::Error::other("no xdg_wm_base"))?;
    s.wm_base = c.alloc_id();
    c.request(s.registry, p::registry::BIND)
        .u32(wname)
        .bind_new_id(p::wm_base::NAME, wver.min(2), s.wm_base)
        .finish();

    // Create the surface → xdg_surface → xdg_toplevel, title it, and commit once to elicit the
    // initial configure (we must not attach a buffer until the compositor configures us).
    s.surface = c.alloc_id();
    c.request(s.compositor, p::compositor::CREATE_SURFACE).u32(s.surface).finish();
    s.xdg_surface = c.alloc_id();
    c.request(s.wm_base, p::wm_base::GET_XDG_SURFACE).u32(s.xdg_surface).object(s.surface).finish();
    s.toplevel = c.alloc_id();
    c.request(s.xdg_surface, p::xdg_surface::GET_TOPLEVEL).u32(s.toplevel).finish();
    c.request(s.toplevel, p::xdg_toplevel::SET_TITLE).string("streamcraft (sc-present spike)").finish();
    c.request(s.surface, p::surface::COMMIT).finish();
    c.flush()?;

    // Time-bounded event loop (~2s) so the spike always exits.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !s.should_close && std::time::Instant::now() < deadline {
        if !poll_readable(c.as_raw_fd(), 100)? {
            continue; // timeout tick — re-check deadline
        }
        if !c.recv()? {
            break; // compositor gone
        }
        while let Some((obj, op)) = c.next_event() {
            dispatch(&mut c, &mut s, obj, op)?;
        }
        c.flush()?;
    }

    println!(
        "sc-present spike: window up, {} frame(s) presented, clean exit ({})",
        s.frames,
        if s.should_close { "toplevel close" } else { "2s deadline" }
    );
    Ok(())
}

fn dispatch(c: &mut Connection, s: &mut State, obj: u32, op: u16) -> io::Result<()> {
    if obj == s.wm_base && op == p::wm_base::EV_PING {
        let serial = c.args().u32().unwrap_or(0);
        c.request(s.wm_base, p::wm_base::PONG).u32(serial).finish();
    } else if obj == s.xdg_surface && op == p::xdg_surface::EV_CONFIGURE {
        let serial = c.args().u32().unwrap_or(0);
        c.request(s.xdg_surface, p::xdg_surface::ACK_CONFIGURE).u32(serial).finish();
        if !s.painted {
            paint_solid(c, s)?;
            s.painted = true;
        }
    } else if obj == s.toplevel && op == p::xdg_toplevel::EV_CLOSE {
        s.should_close = true;
    } else if obj == DISPLAY_ID && op == p::display::EV_ERROR {
        return Err(display_error(c));
    }
    // A `frame` callback's done would land here too (a fresh callback id each time); the spike
    // paints once and idles, so we don't request frames.
    let _ = &s.frames;
    Ok(())
}

/// Create a `wl_shm` buffer, fill it with an opaque solid colour, and commit it to the surface.
fn paint_solid(c: &mut Connection, s: &mut State) -> io::Result<()> {
    let stride = W * 4;
    let size = (stride * H) as usize;
    let (fd, map) = shm_alloc(size)?;
    // 0x00RRGGBB in XRGB8888 (native-endian u32) — a streamcraft dark slate.
    let color: u32 = 0x0020_3550;
    // SAFETY: `map` points at `size` writable bytes we own for this scope.
    unsafe {
        let px = map as *mut u32;
        for i in 0..(size / 4) {
            *px.add(i) = color;
        }
    }
    let pool = c.alloc_id();
    c.request(s.shm, p::shm::CREATE_POOL).u32(pool).fd(fd).i32(size as i32).finish();
    let buffer = c.alloc_id();
    c.request(pool, p::shm_pool::CREATE_BUFFER)
        .u32(buffer)
        .i32(0) // offset
        .i32(W)
        .i32(H)
        .i32(stride)
        .u32(p::shm::FORMAT_XRGB8888)
        .finish();
    c.request(s.surface, p::surface::ATTACH).object(buffer).i32(0).i32(0).finish();
    c.request(s.surface, p::surface::DAMAGE_BUFFER).i32(0).i32(0).i32(W).i32(H).finish();
    c.request(s.surface, p::surface::COMMIT).finish();
    c.flush()?;
    s.frames += 1;
    // The pool's fd is now owned by the compositor's mapping; close our copy (the mapping and
    // the pool keep the memory alive). `map` is intentionally leaked for the spike's short life.
    // SAFETY: our dup of the memfd; the compositor mmap'd it under CREATE_POOL.
    unsafe { libc::close(fd) };
    Ok(())
}

/// A `memfd`-backed anonymous shared region of `size` bytes, mmap'd writable. Returns the fd
/// (to hand to `wl_shm.create_pool`) and the mapping pointer.
fn shm_alloc(size: usize) -> io::Result<(RawFd, *mut libc::c_void)> {
    // SAFETY: memfd_create with a static name; fd checked below.
    let fd = unsafe { libc::memfd_create(c"sc-present".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: sizing then mapping our own fresh memfd.
    unsafe {
        if libc::ftruncate(fd, size as libc::off_t) < 0 {
            let e = io::Error::last_os_error();
            libc::close(fd);
            return Err(e);
        }
        let map = libc::mmap(
            std::ptr::null_mut(),
            size,
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

/// Read and format a `wl_display.error` event (object, code, message).
fn display_error(c: &Connection) -> io::Error {
    let mut a = c.args();
    let object = a.u32().unwrap_or(0);
    let code = a.u32().unwrap_or(0);
    let msg = a.string().unwrap_or("");
    io::Error::other(format!("wayland protocol error on object {object}: code {code}: {msg}"))
}

/// Block up to `timeout_ms` for the socket to become readable. `Ok(false)` on timeout.
fn poll_readable(fd: RawFd, timeout_ms: i32) -> io::Result<bool> {
    let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: single valid pollfd.
    let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n > 0 && (pfd.revents & libc::POLLIN) != 0)
}
