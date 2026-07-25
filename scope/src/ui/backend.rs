//! The SDL3 backend — the crate's only `unsafe` module.
//!
//! Mirrors the pattern established by `sc-sdl3`'s `video.rs`: a thin safe wrapper
//! over `sdl3-sys` with a SAFETY note per `unsafe` block, refcounted subsystem
//! init (an app may already use SDL for its video sink — the scope embeds into that
//! process), one window + renderer, and one static font-atlas texture.
//!
//! Responsibilities:
//! - init SDL video (refcounted), create a resizable window + renderer;
//! - upload the [`Font`] atlas to an RGBA8 texture (nearest sampling, alpha blend);
//! - pump SDL events into an [`Input`] snapshot each frame;
//! - flush a [`DrawList`] via `SDL_RenderGeometryRaw` — one call per batch
//!   (untextured solids, then the font-atlas text run), with `SDL_SetRenderClipRect`
//!   for scissoring.
//!
//! Everything above [`DrawList`] is pure and window-free; this file is where pixels
//! actually happen.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_void, CStr, CString};

use sdl3_sys::everything::*;

use crate::ui::draw::{Batch, DrawList, TexId};
use crate::ui::font::Font;
use crate::ui::{Arena, Input, Key, Mods, MouseButton};

/// The last SDL error as an owned `String`.
fn sdl_error() -> String {
    // SAFETY: SDL_GetError returns a thread-local NUL-terminated string valid until
    // the next SDL call; we copy it out immediately.
    unsafe { CStr::from_ptr(SDL_GetError()) }.to_string_lossy().into_owned()
}

/// A backend init/runtime error (kept local — scope does not depend on core's Error
/// for the UI layer).
#[derive(Debug)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scope-ui: {}", self.0)
    }
}
impl std::error::Error for BackendError {}

fn err(what: &str) -> BackendError {
    BackendError(format!("{what}: {}", sdl_error()))
}

/// SDL window + renderer + font-atlas texture for the scope UI.
pub struct Backend {
    window: *mut SDL_Window,
    renderer: *mut SDL_Renderer,
    font_tex: *mut SDL_Texture,
    /// Persistent input state (level fields survive between frames).
    input: Input,
    /// Reusable per-frame geometry arena (bump; reset each render).
    arena: Arena,
}

// SAFETY: the raw SDL handles are used only from the thread that owns the Backend;
// SDL video on the Linux drivers this targets is a single-threaded in-process
// protocol client (same posture as sc-sdl3::video). Backend is moved whole, never
// shared, so Send but not Sync.
unsafe impl Send for Backend {}

impl Backend {
    /// Open a window titled `title` at `width`×`height` and upload `font`'s atlas.
    /// Headless: set `SDL_VIDEODRIVER=dummy` and it still succeeds (SDL's dummy
    /// driver gives a real renderer), so CI can render frames without a display.
    pub fn open(title: &str, width: u32, height: u32, font: &Font) -> Result<Self, BackendError> {
        let title_c = CString::new(title).unwrap_or_default();
        // SAFETY: plain C calls; SDL_InitSubSystem is refcounted per subsystem, so
        // co-existing with an app's video sink is fine (each pairs its own Quit).
        unsafe {
            if !SDL_InitSubSystem(SDL_INIT_VIDEO) {
                return Err(err("init video"));
            }
            let window = SDL_CreateWindow(
                title_c.as_ptr(),
                width.max(1) as c_int,
                height.max(1) as c_int,
                SDL_WINDOW_RESIZABLE,
            );
            if window.is_null() {
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
                return Err(err("create window"));
            }
            let renderer = SDL_CreateRenderer(window, std::ptr::null());
            if renderer.is_null() {
                let e = err("create renderer");
                SDL_DestroyWindow(window);
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
                return Err(e);
            }
            // Alpha blending so the font atlas (transparent background) composites.
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

            let font_tex = match upload_font_atlas(renderer, font) {
                Ok(t) => t,
                Err(e) => {
                    SDL_DestroyRenderer(renderer);
                    SDL_DestroyWindow(window);
                    SDL_QuitSubSystem(SDL_INIT_VIDEO);
                    return Err(e);
                }
            };

            // Enable text input so we receive SDL_EVENT_TEXT_INPUT (property editing
            // later); harmless if never used.
            SDL_StartTextInput(window);

            let input = Input {
                window_w: width as f32,
                window_h: height as f32,
                ..Input::default()
            };

            Ok(Self {
                window,
                renderer,
                font_tex,
                input,
                arena: Arena::with_capacity(1 << 16),
            })
        }
    }

    /// Pump SDL's event queue into the [`Input`] snapshot for this frame and return
    /// it. Edge fields (pressed/released/wheel/text/keys/quit) describe only this
    /// frame; level fields (mouse pos, held buttons, window size) persist.
    pub fn begin_frame(&mut self) -> Input {
        self.input.begin_frame();
        // SAFETY: SDL_PollEvent fills the event union; we read only the fields
        // matching the discriminant we checked.
        unsafe {
            let mut ev: SDL_Event = std::mem::zeroed();
            while SDL_PollEvent(&mut ev) {
                let ty = SDL_EventType(ev.r#type);
                match ty {
                    SDL_EVENT_QUIT | SDL_EVENT_WINDOW_CLOSE_REQUESTED => {
                        self.input.quit = true;
                    }
                    SDL_EVENT_MOUSE_MOTION => {
                        self.input.mouse_x = ev.motion.x;
                        self.input.mouse_y = ev.motion.y;
                    }
                    SDL_EVENT_MOUSE_BUTTON_DOWN | SDL_EVENT_MOUSE_BUTTON_UP => {
                        if let Some(b) = map_button(ev.button.button) {
                            let i = Input::button_index(b);
                            self.input.mouse_x = ev.button.x;
                            self.input.mouse_y = ev.button.y;
                            if ev.button.down {
                                self.input.mouse_down[i] = true;
                                self.input.mouse_pressed[i] = true;
                            } else {
                                self.input.mouse_down[i] = false;
                                self.input.mouse_released[i] = true;
                            }
                        }
                    }
                    SDL_EVENT_MOUSE_WHEEL => {
                        self.input.wheel += ev.wheel.y;
                    }
                    SDL_EVENT_TEXT_INPUT => {
                        // SAFETY: `text` is a NUL-terminated UTF-8 C string owned by
                        // SDL, valid for the duration of this event handling.
                        let s = CStr::from_ptr(ev.text.text);
                        self.input.text.push_str(&s.to_string_lossy());
                    }
                    SDL_EVENT_KEY_DOWN => {
                        if let Some(k) = map_key(ev.key.key) {
                            let m = ev.key.r#mod;
                            let mods = Mods {
                                ctrl: (m & SDL_KMOD_CTRL) != SDL_Keymod(0),
                                shift: (m & SDL_KMOD_SHIFT) != SDL_Keymod(0),
                                alt: (m & SDL_KMOD_ALT) != SDL_Keymod(0),
                            };
                            self.input.keys.push((k, mods));
                        }
                    }
                    SDL_EVENT_WINDOW_RESIZED | SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED => {
                        let mut w: c_int = 0;
                        let mut h: c_int = 0;
                        SDL_GetWindowSize(self.window, &mut w, &mut h);
                        self.input.window_w = w as f32;
                        self.input.window_h = h as f32;
                    }
                    _ => {}
                }
            }
        }
        self.input.clone()
    }

    /// The current window size in logical pixels (level state, kept in sync by the
    /// event pump; queried directly here so the first frame is right).
    pub fn window_size(&self) -> (f32, f32) {
        let mut w: c_int = 0;
        let mut h: c_int = 0;
        // SAFETY: reads sizes of a window we own.
        unsafe {
            SDL_GetWindowSize(self.window, &mut w, &mut h);
        }
        (w as f32, h as f32)
    }

    /// Clear to `clear` (RGBA8), flush the draw list, and present. The whole frame's
    /// geometry comes from the internal per-frame arena, which is reset first, so a
    /// steady-state frame does no heap allocation for vertices.
    pub fn render(&mut self, dl: &DrawList, clear: crate::ui::Color) {
        self.arena.reset();
        let batches = dl.build(&mut self.arena);
        let font_tex = self.font_tex;
        let renderer = self.renderer;
        // SAFETY: `renderer` and `font_tex` are live handles we own; the batch
        // slices live in `self.arena`, which outlives this call and is not reset
        // until the next frame. Pointers/strides passed to SDL match the slice
        // layout exactly (interleaved xy/uv pairs, rgba quads, u16 indices).
        unsafe {
            let c = clear.to_f32();
            SDL_SetRenderDrawColorFloat(renderer, c[0], c[1], c[2], c[3]);
            SDL_RenderClear(renderer);
            for b in &batches {
                apply_clip(renderer, b);
                let tex = match b.tex {
                    TexId::Font => font_tex,
                    TexId::None => std::ptr::null_mut(),
                };
                render_batch(renderer, tex, b);
            }
            // Reset the clip so a later app draw isn't scissored.
            SDL_SetRenderClipRect(renderer, std::ptr::null());
            SDL_RenderPresent(renderer);
        }
    }

    /// Did the user request to close the window since the last `begin_frame`?
    pub fn should_close(&self) -> bool {
        self.input.quit
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: tearing down objects we own, then dropping our subsystem ref.
        unsafe {
            SDL_StopTextInput(self.window);
            if !self.font_tex.is_null() {
                SDL_DestroyTexture(self.font_tex);
            }
            SDL_DestroyRenderer(self.renderer);
            SDL_DestroyWindow(self.window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
    }
}

/// Create the RGBA8 font-atlas texture and upload the glyph pixels.
///
/// # Safety
/// `renderer` must be a live renderer.
unsafe fn upload_font_atlas(renderer: *mut SDL_Renderer, font: &Font) -> Result<*mut SDL_Texture, BackendError> {
    let (w, h) = font.atlas_size();
    let tex = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STATIC,
        w as c_int,
        h as c_int,
    );
    if tex.is_null() {
        return Err(err("create font texture"));
    }
    // Nearest sampling keeps the bitmap crisp; blend so transparent bg composites.
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_NEAREST);
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
    let pixels = font.atlas_rgba();
    let pitch = (w * 4) as c_int;
    // SAFETY: `pixels` is exactly w*h*4 bytes (checked by Font); the texture is
    // w*h RGBA32; NULL rect = whole texture.
    if !SDL_UpdateTexture(tex, std::ptr::null(), pixels.as_ptr() as *const c_void, pitch) {
        let e = err("upload font atlas");
        SDL_DestroyTexture(tex);
        return Err(e);
    }
    Ok(tex)
}

/// Set (or clear) the scissor rect for a batch.
///
/// # Safety
/// `renderer` must be live.
unsafe fn apply_clip(renderer: *mut SDL_Renderer, b: &Batch) {
    match b.clip {
        Some(c) => {
            let rect = SDL_Rect {
                x: c.x.floor() as c_int,
                y: c.y.floor() as c_int,
                w: c.w.ceil().max(0.0) as c_int,
                h: c.h.ceil().max(0.0) as c_int,
            };
            SDL_SetRenderClipRect(renderer, &rect);
        }
        None => {
            SDL_SetRenderClipRect(renderer, std::ptr::null());
        }
    }
}

/// Issue one `SDL_RenderGeometryRaw` for a batch.
///
/// # Safety
/// `renderer` live; `tex` a live texture or NULL; the batch slices are valid for
/// this call and their strides match the raw-geometry contract.
unsafe fn render_batch(renderer: *mut SDL_Renderer, tex: *mut SDL_Texture, b: &Batch) {
    if b.indices.is_empty() {
        return;
    }
    SDL_RenderGeometryRaw(
        renderer,
        tex,
        b.xy.as_ptr(),
        (2 * std::mem::size_of::<f32>()) as c_int,
        b.color.as_ptr() as *const SDL_FColor,
        (4 * std::mem::size_of::<f32>()) as c_int,
        b.uv.as_ptr(),
        (2 * std::mem::size_of::<f32>()) as c_int,
        b.num_vertices as c_int,
        b.indices.as_ptr() as *const c_void,
        b.indices.len() as c_int,
        std::mem::size_of::<u16>() as c_int,
    );
}

/// Map an SDL mouse button constant to our [`MouseButton`].
fn map_button(b: u8) -> Option<MouseButton> {
    match b as u32 {
        x if x == SDL_BUTTON_LEFT as u32 => Some(MouseButton::Left),
        x if x == SDL_BUTTON_MIDDLE as u32 => Some(MouseButton::Middle),
        x if x == SDL_BUTTON_RIGHT as u32 => Some(MouseButton::Right),
        _ => None,
    }
}

/// Map an SDL keycode to our [`Key`] (only the non-text keys the inspector uses).
fn map_key(k: SDL_Keycode) -> Option<Key> {
    match k {
        SDLK_RETURN | SDLK_KP_ENTER => Some(Key::Enter),
        SDLK_ESCAPE => Some(Key::Escape),
        SDLK_TAB => Some(Key::Tab),
        SDLK_BACKSPACE => Some(Key::Backspace),
        SDLK_UP => Some(Key::Up),
        SDLK_DOWN => Some(Key::Down),
        SDLK_LEFT => Some(Key::Left),
        SDLK_RIGHT => Some(Key::Right),
        SDLK_PAGEUP => Some(Key::PageUp),
        SDLK_PAGEDOWN => Some(Key::PageDown),
        SDLK_HOME => Some(Key::Home),
        SDLK_END => Some(Key::End),
        _ => None,
    }
}
