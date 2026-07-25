//! `ui_demo` — a standalone showcase of the scope immediate-mode UI with *fake*
//! pipeline data. No introspection protocol, no core pipeline: this exercises the
//! reusable `ui` layer (backend, arena, draw list, font, widgets) so it can be
//! iterated on and gated in headless CI before the real panels are wired.
//!
//! Layout: a top bar with tabs, a left "elements" panel of fake element names with
//! animating queue fill bars and counters, a right "latency" panel of key/value
//! rows and buttons, and a bottom log view fed by synthetic lines.
//!
//! Headless / CI:
//! ```text
//! SDL_VIDEODRIVER=dummy cargo run -p streamcraft-scope --bin ui_demo -- --frames 60
//! ```
//! renders 60 frames and exits 0. Without `--frames`, it runs until the window is
//! closed.

use streamcraft_scope::ui::backend::Backend;
use streamcraft_scope::ui::draw::Color;
use streamcraft_scope::ui::widgets::{LogView, UiState};
use streamcraft_scope::ui::{Font, Rect, Ui};

/// Fake per-element live state we animate to look like a running pipeline.
struct FakeElement {
    name: &'static str,
    /// Queue fill 0..1, oscillating.
    fill_phase: f32,
    fill_speed: f32,
    /// Monotonic buffer counter.
    buffers: u64,
    per_frame: u64,
}

fn main() {
    // Tiny arg parse: `--frames N` runs N frames then exits (headless CI).
    let mut max_frames: Option<u64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => {
                max_frames = args.next().and_then(|s| s.parse().ok());
            }
            other => {
                eprintln!("ui_demo: unknown arg {other:?} (only --frames N)");
            }
        }
    }

    let font = Font::new();
    let mut backend = match Backend::open("scraft-scope (demo)", 900, 620, &font) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ui_demo: could not open window: {e}");
            // A headless box with no driver at all should not hard-fail CI in a way
            // that hides real errors — but per the gate we expect the dummy driver
            // to work, so treat inability to open as failure.
            std::process::exit(1);
        }
    };

    // --- retained app state (immediate-mode: the app owns what outlives a frame) ---
    let mut ui_state = UiState::default();
    let mut selected_tab = 0usize;
    let mut show_dropped = false;
    let mut paused = false;
    let mut log = LogView::new(1024);
    log.push("scope demo started", Color::rgb(0x8a, 0x93, 0xa0));

    let mut elements = vec![
        FakeElement { name: "filesrc",      fill_phase: 0.1, fill_speed: 0.017, buffers: 0, per_frame: 1 },
        FakeElement { name: "mkvdemux",     fill_phase: 0.5, fill_speed: 0.023, buffers: 0, per_frame: 2 },
        FakeElement { name: "h264dec",      fill_phase: 0.8, fill_speed: 0.031, buffers: 0, per_frame: 2 },
        FakeElement { name: "aacdec",       fill_phase: 0.3, fill_speed: 0.013, buffers: 0, per_frame: 3 },
        FakeElement { name: "videoconvert", fill_phase: 0.0, fill_speed: 0.041, buffers: 0, per_frame: 2 },
        FakeElement { name: "sdl3videosink",fill_phase: 0.65,fill_speed: 0.019, buffers: 0, per_frame: 2 },
    ];

    let tabs = ["Graph", "Latency", "Events", "Debug"];
    let mut frame: u64 = 0;

    loop {
        let input = backend.begin_frame();
        if backend.should_close() {
            break;
        }
        let size = backend.window_size();

        // --- animate the fake data ---
        if !paused {
            for e in elements.iter_mut() {
                e.fill_phase += e.fill_speed;
                e.buffers += e.per_frame;
            }
            // Feed the log a synthetic line every few frames.
            if frame.is_multiple_of(6) {
                let (msg, col) = synthetic_log_line(frame);
                log.push(msg, col);
            }
        }

        // --- build the frame ---
        let mut ui = Ui::new(&input, &font, &mut ui_state, size);
        ui.clear_background();

        let pad = 8.0;
        let top_h = 30.0;

        // Top tab strip spanning the width (drawn without a panel — direct rect).
        {
            // A borrowless scope: tab_strip needs a layout cursor, so wrap it in a
            // thin full-width panel-less region by pushing a temporary cursor via a
            // one-row panel body. Simpler: draw tabs into explicit rects.
            let strip = Rect::new(pad, pad, size.0 - 2.0 * pad, top_h - 4.0);
            let n = tabs.len() as f32;
            let tw = strip.w / n;
            for (i, name) in tabs.iter().enumerate() {
                let r = Rect::new(strip.x + i as f32 * tw, strip.y, tw - 2.0, strip.h);
                if ui.button_in(name, r) {
                    selected_tab = i;
                }
                if i == selected_tab {
                    // Underline the active tab in the accent colour.
                    let accent = ui.theme().accent;
                    ui.draw_list_mut()
                        .fill_rect(Rect::new(r.x, r.bottom() - 2.0, r.w, 2.0), accent);
                }
            }
        }
        // Keep `selected_tab` meaningful even if unused visually below.
        let _ = selected_tab;

        let body_y = pad + top_h;
        let body_h = size.1 - body_y - pad;
        let log_h = 180.0;
        let cols_h = (body_h - log_h - pad).max(60.0);
        let col_w = (size.0 - 3.0 * pad) * 0.5;

        // Left: elements panel.
        let left = Rect::new(pad, body_y, col_w, cols_h);
        ui.begin_panel("Elements", left);
        for e in &elements {
            // sine oscillation in 0..1
            let frac = 0.5 + 0.5 * (e.fill_phase * std::f32::consts::TAU).sin();
            let color = if frac > 0.85 {
                Color::rgb(0xe0, 0x6c, 0x4c) // near-full: warm
            } else {
                ui.theme().bar_fill
            };
            ui.label(e.name);
            ui.fill_bar(frac, color, &format!("queue {:>3.0}%", frac * 100.0));
            ui.kv("buffers", &format!("{}", e.buffers));
        }
        ui.end_panel();

        // Right: latency + controls panel.
        let right = Rect::new(pad * 2.0 + col_w, body_y, col_w, cols_h);
        ui.begin_panel("Latency / Controls", right);
        ui.kv("pipeline", "PLAYING");
        ui.kv("path budget", "42.0 ms");
        ui.kv("h264dec", "18.3 ms");
        ui.kv("sink render", "6.1 ms");
        ui.kv("frame", &format!("{frame}"));
        let total: u64 = elements.iter().map(|e| e.buffers).sum();
        ui.kv("total buffers", &format!("{total}"));
        if ui.button(if paused { "Resume" } else { "Pause" }) {
            paused = !paused;
            log.push(if paused { "paused" } else { "resumed" }, Color::rgb(0x4a, 0xa8, 0xff));
        }
        if ui.button("Step") {
            for e in elements.iter_mut() {
                e.buffers += e.per_frame;
            }
            log.push("stepped one frame", Color::rgb(0x8a, 0x93, 0xa0));
        }
        ui.toggle("show dropped", &mut show_dropped);
        ui.end_panel();

        // Bottom: log view spanning the full width.
        let log_rect = Rect::new(pad, body_y + cols_h + pad, size.0 - 2.0 * pad, log_h);
        ui.begin_panel("Log", log_rect);
        // Fill the panel body with the log view.
        ui.log_view(&mut log);
        ui.end_panel();

        let dl = ui.finish();
        backend.render(&dl, Color::rgb(0x14, 0x16, 0x1a));

        frame += 1;
        if let Some(max) = max_frames {
            if frame >= max {
                break;
            }
        }
    }

    // Clean exit (Drop on Backend tears SDL down).
    eprintln!("ui_demo: rendered {frame} frames");
}

/// A rotating set of synthetic log lines with a few colours, so the log view shows
/// a realistic mix (info / warn / error).
fn synthetic_log_line(frame: u64) -> (String, Color) {
    let info = Color::rgb(0xc8, 0xcf, 0xd8);
    let warn = Color::rgb(0xe0, 0xb0, 0x4c);
    let err = Color::rgb(0xe0, 0x6c, 0x4c);
    match (frame / 6) % 7 {
        0 => (format!("[{frame:>5}] h264dec: decoded keyframe, dpb=4"), info),
        1 => (format!("[{frame:>5}] mkvdemux: cluster @ pts=1.234s"), info),
        2 => (format!("[{frame:>5}] aacdec: 1024 samples, 48000 Hz"), info),
        3 => (format!("[{frame:>5}] qos: sink late by 3.2ms, dropping"), warn),
        4 => (format!("[{frame:>5}] videoconvert: I420 -> IYUV"), info),
        5 => (format!("[{frame:>5}] pool: allocation stall, waiting"), warn),
        _ => (format!("[{frame:>5}] sink: renegotiate 1280x720@30 failed"), err),
    }
}
