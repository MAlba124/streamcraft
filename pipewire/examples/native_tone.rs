//! Export a mono `DSP_F32` client node through the from-scratch native ClientNode path (no
//! libpipewire) and feed it a sine tone. Exercises the whole RT setup against the live daemon:
//! node creation, format + cross-process buffer negotiation, shared-memory buffer mapping, the
//! activation/transport records, and `Command Start`.
//!
//! The node does not autoconnect (a bare client-node is not adapter-wrapped, so the session
//! manager leaves it unlinked). Link it to a device's left channel by hand to drive negotiation
//! to completion:
//!
//!   cargo run -p pf-pipewire --example native_tone &
//!   pw-link 'streamcraft tone' <sink>:playback_FL      # or use pw-link's port ids
//!
//! With `STREAMCRAFT_PW_DEBUG=1` it traces every step. NOTE: the final driver→client RT trigger
//! (the sink writing our eventfd each cycle) is only wired up by session-manager-managed
//! linking; see the module docs for the remaining work before this emits sound.
//!
//! Run (needs a running PipeWire daemon; Ctrl-C to stop):
//!   STREAMCRAFT_PW_DEBUG=1 cargo run -p pf-pipewire --example native_tone

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pf_pipewire::native::{play, AudioConfig, SampleFormat};

fn main() {
    let rate = 48_000u32;
    // DSP mode: one mono F32 port that links 1:1 to a device's playback_FL port (no adapter).
    let channels = 1u32;
    let freq = 440.0f32;
    let cfg = AudioConfig {
        rate,
        channels,
        format: SampleFormat::F32,
        quantum: 1024,
        app_name: "streamcraft-native-tone".into(),
        node_name: "streamcraft tone".into(),
        dsp: true,
    };

    let quit = Arc::new(AtomicBool::new(false));
    {
        let q = Arc::clone(&quit);
        let _ = ctrlc_lite(move || q.store(true, Ordering::Relaxed));
    }

    // Interleaved f32 sine; `phase` advances across process cycles.
    let mut phase = 0.0f32;
    let step = 2.0 * std::f32::consts::PI * freq / rate as f32;
    let fill = move |out: &mut [u8]| -> usize {
        let frames = out.len() / (channels as usize * 4);
        for f in 0..frames {
            let s = (phase.sin() * 0.2).clamp(-1.0, 1.0);
            phase += step;
            if phase > std::f32::consts::TAU {
                phase -= std::f32::consts::TAU;
            }
            let bytes = s.to_le_bytes();
            for c in 0..channels as usize {
                let off = (f * channels as usize + c) * 4;
                out[off..off + 4].copy_from_slice(&bytes);
            }
        }
        frames * channels as usize * 4
    };

    println!("exported a mono {freq} Hz-tone client node via the native protocol (no libpipewire)");
    println!("link it to hear it, e.g.:  pw-link 'streamcraft tone' <sink>:playback_FL");
    println!("Ctrl-C to stop.");
    match play(cfg, &quit, fill) {
        Ok(()) => println!("stopped."),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Minimal SIGINT handler (avoids a ctrlc crate dep for a one-file example).
fn ctrlc_lite(f: impl FnMut() + Send + 'static) -> std::io::Result<()> {
    use std::sync::Mutex;
    static HANDLER: Mutex<Option<Box<dyn FnMut() + Send>>> = Mutex::new(None);
    *HANDLER.lock().unwrap() = Some(Box::new(f));
    extern "C" fn on_sigint(_: libc::c_int) {
        if let Ok(mut g) = HANDLER.lock() {
            if let Some(h) = g.as_mut() {
                h();
            }
        }
    }
    // SAFETY: installing a signal handler for SIGINT.
    unsafe { libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t) };
    Ok(())
}
