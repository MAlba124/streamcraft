//! Immediate-mode UI for pf-scope — the pipeline inspector's own tiny widget
//! toolkit, drawn through SDL3's `SDL_RenderGeometryRaw`.
//!
//! # Why immediate mode
//!
//! The inspector shows fast-changing live data (queue fill bars, counters, log
//! tails, latency histograms). A retained widget tree would mean diffing that state
//! into persistent nodes every frame; immediate mode instead *rebuilds the UI from
//! the data every frame*, which is both simpler and a natural fit for profluens's
//! per-frame-arena, no-steady-state-allocation ethos. The concept is Casey
//! Muratori's "Immediate-Mode Graphical User Interfaces" (2005 talk / RAD Game
//! Tools); this is a clean-room implementation of the idea, not a port of any
//! imgui/egui code.
//!
//! # Shape of a frame
//!
//! ```text
//! backend.begin_frame()  -> Input        // pump SDL events into an Input snapshot
//! ui = Ui::new(&input, &font, size)      // per-frame context over a fresh DrawList
//!   panel/label/button/... widgets       // append primitives, read Input, set ids
//! let dl = ui.finish()                   // the frame's DrawList (pure data)
//! backend.render(&dl, &mut arena)        // tessellate -> SDL_RenderGeometryRaw
//! ```
//!
//! Everything up to `finish()` is pure and window-free, so the whole widget layer
//! is tested by feeding a mock [`Input`] and asserting the emitted [`draw::Prim`]s.
//!
//! # Modules
//! - [`arena`]  — per-frame bump allocator for geometry/text.
//! - [`draw`]   — the draw list: primitives, clip stack, tessellation to batches.
//! - [`font`]   — embedded 8x8 bitmap font + atlas + measurement/layout.
//! - [`widgets`]— the [`Ui`] context and the immediate-mode widgets.
//! - [`backend`]— the one `unsafe` module: SDL init/window/renderer/events/flush.

pub mod arena;
pub mod backend;
pub mod dock;
pub mod draw;
pub mod font;
mod font_data;
pub mod widgets;

pub use arena::Arena;
pub use draw::{Color, DrawList, Rect};
pub use font::Font;
pub use widgets::{Theme, Ui};

/// A stable widget identity derived from its call-site label. Two widgets with the
/// same label under the same parent scope collide — callers disambiguate by giving
/// distinct labels (or a `"label##salt"` suffix, imgui-style, stripped for display).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Id(pub u64);

impl Id {
    /// FNV-1a of the label, mixed with a parent seed so the same label nests
    /// distinctly inside different panels. FNV is chosen for being tiny, allocation
    /// free, and stable across runs (Fowler–Noll–Vo, 1991); the UI never needs
    /// cryptographic strength, only good call-site dispersion.
    pub fn new(seed: u64, label: &str) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET ^ seed.wrapping_mul(PRIME);
        for b in label.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(PRIME);
        }
        Id(h)
    }
}

/// Which mouse button an interaction concerns. Kept small and explicit rather than
/// a bitmask so widget code reads plainly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

/// Keys the inspector cares about (navigation + text editing). Printable characters
/// arrive via [`Input::text`], not here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Enter,
    Escape,
    Tab,
    Backspace,
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
}

/// Per-frame input snapshot the backend fills from SDL events. Pure data — tests
/// construct it directly. "Edge" fields (`*_pressed`, `*_released`, `wheel`,
/// `text`, `keys`) describe *this frame only* and are cleared at `begin_frame`;
/// "level" fields (`mouse_x/y`, `*_down`, `window_w/h`) persist.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub mouse_x: f32,
    pub mouse_y: f32,
    /// Held state, indexed by [`MouseButton`] (`[left, middle, right]`).
    pub mouse_down: [bool; 3],
    pub mouse_pressed: [bool; 3],
    pub mouse_released: [bool; 3],
    /// Vertical wheel delta this frame (+ = away from user / scroll up).
    pub wheel: f32,
    /// UTF-8 text typed this frame (from SDL text-input events).
    pub text: String,
    /// Non-text keys pressed this frame, with the modifier state at press time.
    pub keys: Vec<(Key, Mods)>,
    pub window_w: f32,
    pub window_h: f32,
    /// The user asked to close the window this frame.
    pub quit: bool,
}

/// Keyboard modifier state at the moment of a key press.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Input {
    /// Reset the per-frame edge fields; keep level state. Called at frame start.
    pub fn begin_frame(&mut self) {
        self.mouse_pressed = [false; 3];
        self.mouse_released = [false; 3];
        self.wheel = 0.0;
        self.text.clear();
        self.keys.clear();
        self.quit = false;
    }

    pub fn button_index(b: MouseButton) -> usize {
        match b {
            MouseButton::Left => 0,
            MouseButton::Middle => 1,
            MouseButton::Right => 2,
        }
    }
    /// Was `b` pressed (down-edge) this frame?
    pub fn pressed(&self, b: MouseButton) -> bool {
        self.mouse_pressed[Self::button_index(b)]
    }
    /// Was `b` released (up-edge) this frame?
    pub fn released(&self, b: MouseButton) -> bool {
        self.mouse_released[Self::button_index(b)]
    }
    /// Is `b` currently held?
    pub fn down(&self, b: MouseButton) -> bool {
        self.mouse_down[Self::button_index(b)]
    }
    /// Did the given non-text key fire this frame?
    pub fn key_pressed(&self, k: Key) -> bool {
        self.keys.iter().any(|(kk, _)| *kk == k)
    }
}
