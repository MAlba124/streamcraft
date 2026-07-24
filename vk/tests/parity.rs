//! GPU ↔ CPU conversion parity (the crate's load-bearing correctness claim): the
//! compute shader (`shaders/yuv2rgb.comp`) and the CPU reference
//! (`sc_wayland::convert::i420_to_xrgb`) implement the *same* integer BT.601
//! derivation (Rec. ITU-R BT.601-7 §2.5.1/§3.5 — see REFERENCES.md), and GLSL's
//! signed `>>` is arithmetic like Rust's (GLSL 4.50 §5.9) — so the two must agree
//! **exactly**, byte for byte, X channel included.
//!
//! Runs on any Vulkan device with a compute queue — including lavapipe, Mesa's
//! software implementation, which the workspace's ICD set provides — so this is a
//! headless-capable test. It skips (cleanly, with a message) only when no Vulkan
//! loader/device exists at all.

use sc_vk::gpu::{i420_size, Gpu, Renderer};
use sc_wayland::convert::i420_to_xrgb;

/// Deterministic planes exercising the full Y/C ranges, including studio-swing
/// out-of-range values the clamp must catch.
fn make_planes(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(i420_size(w, h));
    let mut x = seed | 1;
    let mut next = || {
        // xorshift32 — Marsaglia (2003), "Xorshift RNGs", J. Stat. Software 8(14).
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x >> 8) as u8
    };
    for _ in 0..i420_size(w, h) {
        v.push(next());
    }
    v
}

fn gpu_or_skip(require_export: bool) -> Option<Gpu> {
    match Gpu::new(require_export) {
        Ok(g) => {
            eprintln!("vulkan device: {}", g.device_name);
            Some(g)
        }
        Err(e) => {
            eprintln!("SKIP: no usable Vulkan ({e})");
            None
        }
    }
}

#[test]
fn shader_matches_cpu_reference_exactly() {
    let Some(gpu) = gpu_or_skip(false) else { return };
    // Odd dimensions on purpose: the ceil(w/2) chroma geometry must agree too.
    for (w, h, seed) in [(64, 48, 1u32), (33, 27, 0xBEEF), (128, 96, 42)] {
        let mut r = Renderer::new(&gpu, w, h, 2, false).expect("renderer");
        let planes = make_planes(w, h, seed);

        let mut cpu = vec![0u8; w * h * 4];
        assert!(i420_to_xrgb(&planes, w, h, &mut cpu, w * 4), "CPU reference converts");

        r.render(&gpu, &planes, 1).expect("gpu render");
        let gpu_out = r.read_back(&gpu, 1).expect("read back");

        assert_eq!(gpu_out.len(), cpu.len());
        assert_eq!(
            gpu_out, cpu,
            "GPU and CPU BT.601 paths must agree byte-for-byte ({w}x{h})"
        );
        r.destroy(&gpu);
    }
}

#[test]
fn short_frames_error_and_bad_slots_error_without_panicking() {
    let Some(gpu) = gpu_or_skip(false) else { return };
    let mut r = Renderer::new(&gpu, 32, 32, 1, false).expect("renderer");
    assert!(r.render(&gpu, &[0u8; 10], 0).is_err(), "short frame is a clean Err");
    let planes = make_planes(32, 32, 7);
    assert!(r.render(&gpu, &planes, 5).is_err(), "bad slot is a clean Err");
    r.render(&gpu, &planes, 0).expect("good frame still renders");
    r.destroy(&gpu);
}

#[test]
fn export_mode_requires_capability_and_yields_fds() {
    let Some(gpu) = gpu_or_skip(false) else { return };
    if !gpu.can_export {
        eprintln!("SKIP: device cannot export dma-bufs");
        return;
    }
    let r = Renderer::new(&gpu, 64, 64, 3, true).expect("export renderer");
    for s in 0..3 {
        assert!(r.slot_fd(s).is_some(), "slot {s} exported an fd");
    }
    assert_eq!(r.stride(), 64 * 4);
    assert!(
        r.read_back(&gpu, 0).is_err(),
        "read_back is refused in export mode (not host-visible)"
    );
    r.destroy(&gpu);
}
