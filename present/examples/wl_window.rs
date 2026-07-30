//! Phase-3 check: the reusable [`Window`] + the software [`Canvas`] + recycling `wl_shm`
//! buffers, drawing an animated GUI frame each vsync for ~2s. Proves the GUI present path
//! (no GPU, compositor blends) end to end against a live compositor.
//!
//! Run:  `WAYLAND_DISPLAY=wayland-1 cargo run -p pf-present --example wl_window`

use std::io;

use pf_present::window::Window;

fn main() -> io::Result<()> {
    let mut win = Window::open("profluens — pf-present GUI demo", 800, 450)?;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut frames = 0u32;
    while !win.should_close() && std::time::Instant::now() < deadline {
        let t = frames as f32 * 0.05;
        win.frame(|c| {
            let (w, h) = (c.width() as i32, c.height() as i32);
            c.clear(0xff10_1418); // opaque dark background
            // A control bar across the bottom.
            c.fill_rect(0, h - 48, w, 48, 0xff20_2830);
            c.blend_rect(16, h - 36, 120, 24, 0xa040_a0ff); // a translucent "button"
            // A moving scope-like triangle fan.
            let cx = (w / 2) as f32 + 120.0 * t.cos();
            let cy = (h / 2) as f32 + 60.0 * (t * 1.3).sin();
            c.fill_triangle(
                (cx as i32, (cy - 40.0) as i32),
                ((cx - 46.0) as i32, (cy + 30.0) as i32),
                ((cx + 46.0) as i32, (cy + 30.0) as i32),
                0xc030_ffa0,
            );
            // A framing line.
            c.draw_line(8, 8, w - 8, 8, 0xff50_6070);
        })?;
        // Pace roughly to the compositor: block briefly for events (release/configure).
        win.dispatch(true)?;
        frames += 1;
    }

    println!(
        "pf-present GUI demo: {frames} frame(s) rendered + recycled, clean exit ({})",
        if win.should_close() { "toplevel close" } else { "2s deadline" }
    );
    Ok(())
}
