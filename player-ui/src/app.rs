//! The application glue: build the [`Player`] with external video, splice in the
//! frame-slot sink, spawn the streaming thread, and drive the GUI loop on the main thread
//! (spec: streamcraft.md UI `<update>` §1209).
//!
//! # Threading (why this shape)
//!
//! `Player::run()` blocks to EOS and takes `&mut self`, and SDL wants its event pump on
//! the process main thread. So we:
//! 1. build the player, open it with `SinkPolicy { video: External, audio: Device }` —
//!    `External` makes the autoplug controller add **no** video sink and record the tap
//!    point (`player.video_out()`), and the audio device sink still provides the clock;
//! 2. add our [`FrameSlotSink`] to `player.pipeline` and `link(video_out, framesink.sink)`;
//! 3. pull every transport handle we need (`pause_handle`, `seek_handle`, `stop_handle`,
//!    `tap_handle`) *before* moving `player` onto a background thread that calls `run()`;
//! 4. run the GUI loop here on the main thread, reading the shared [`FrameSlot`] and the
//!    handles; on quit, trip the stop handle and join.
//!
//! The frame-slot sink touches no SDL and the GUI touches no pipeline internals beyond the
//! cloneable handles, so there is no cross-thread SDL and no shared mutable pipeline state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use sc_play::{Player, SinkChoice, SinkPolicy};
use sc_vaapi::gpuframe::GpuFrameChannel;
use streamcraft_core::bus::BusMessage;
use streamcraft_core::id::ElementId;
use streamcraft_core::pipeline::{SeekHandle, SeekIndex};
use streamcraft_core::time::Timestamp;

use streamcraft_scope::ui::widgets::UiState;
use streamcraft_scope::ui::{Font, Key, Ui};

use crate::backend::Backend;
use crate::framesink::{FrameSlot, FrameSlotSink};
use crate::ui::{self as player_ui, PlayerUiState, StatRow, TrackRow, UiActions, UiFrame};

/// How the player was invoked.
pub struct Config {
    /// The media file to open.
    pub file: String,
    /// When `Some(n)`, run headless: drive the compositor for `n` frames into the (dummy-
    /// driver) offscreen renderer, assert the video-texture + DrawList path produced output,
    /// print fps/drop numbers, and exit — the on-display-free validation proof.
    pub headless_frames: Option<u32>,
}

/// Run the player to completion. Returns a one-line error string on a build failure
/// (unknown container, no linkable video, etc.) — the binary maps it to a nonzero exit.
pub fn run(cfg: Config) -> Result<(), String> {
    // Build with the ZERO-COPY external video choice (no sink added — we tap `video_out`) +
    // a DEVICE audio sink (the pipeline clock). The controller builds a DMA-BUF-exporting
    // VA-API decoder when the codec is hardware-decodable and a VA device is present; it
    // falls back to the readback/software `video/raw` path otherwise, which we detect via
    // `player.video_zerocopy()`. Headless runs (`--headless-frames`) request zero-copy too,
    // but the dummy SDL driver has no GL, so the backend/framesink use the CPU path — and a
    // zero-copy DECODER wired to a CPU sink would mismatch (`video/gpu` vs `video/raw`). So in
    // headless mode we force the plain External path (see `want_zc`).
    let want_zc = cfg.headless_frames.is_none();
    let video_choice =
        if want_zc { SinkChoice::ExternalZeroCopy } else { SinkChoice::External };
    let policy = SinkPolicy { video: video_choice, audio: SinkChoice::Device };

    // The shared GPU-frame channel (decoder push side ↔ EGL sink/GUI pop+release side).
    let channel = GpuFrameChannel::new();
    let mut player = if want_zc {
        Player::open_zerocopy(&cfg.file, policy, Arc::clone(&channel))?
    } else {
        Player::open(&cfg.file, policy)?
    };

    if !player.any_track_linked() {
        return Err("no track could be linked to any decoder".to_string());
    }

    // The tap point emitting the final video frames. In zero-copy mode this is the decoder's
    // `video/gpu` src (DMA-BUF descriptors); otherwise the subtitle overlay's / decoder's
    // `video/raw` src. No video track ⇒ nothing to show.
    let Some((vid_el, vid_pad)) = player.video_out() else {
        return Err("this file has no video track to display (audio-only)".to_string());
    };
    // Did the controller actually wire zero-copy? (A requested zero-copy falls back to
    // readback for a software codec / no device.) This drives the sink + backend choice.
    let zerocopy = player.video_zerocopy();

    // Splice the clock-paced frame-slot sink onto the tap. Zero-copy: the sink pops the paired
    // GpuFrame from `channel` and hands it to the GUI; CPU: it copies pixels into the triple
    // buffer. Both bridge to the GUI thread through the same `slot`.
    let slot = if zerocopy {
        FrameSlot::new_zerocopy(Arc::clone(&channel))
    } else {
        FrameSlot::new()
    };
    let sink = if zerocopy {
        FrameSlotSink::new_zerocopy(Arc::clone(&slot), Arc::clone(&channel))
    } else {
        FrameSlotSink::new(Arc::clone(&slot))
    };
    let sink_id = player.pipeline.add(sink);
    player
        .pipeline
        .link((vid_el, vid_pad), (sink_id, "sink"))
        .map_err(|e| format!("could not link video_out → frameslotsink: {e:?}"))?;
    // Give the sink its own modest pool + queue so paced frames don't compete with the
    // decoder's shared pool (the same lesson autoplug applies to the SDL sink's peer).
    player.pipeline.set_queue_capacity(sink_id, 8);
    eprintln!(
        "scplay-ui: decode path = {}",
        if zerocopy { "VA-API zero-copy (DMA-BUF → EGL)" } else { "CPU frame-slot (readback/software)" }
    );

    // Collect the track menu rows and stats watch-list from the player BEFORE we move it.
    let tracks = build_track_rows(&player);
    let watched: Vec<(&'static str, ElementId)> = player.watched();
    let seek_index: SeekIndex = player.seek_index().clone();
    let duration: Timestamp = player.duration().unwrap_or(Timestamp::NONE);

    // Pull the transport handles (cloneable, Send) before the player moves to its thread.
    let pause = player.pipeline.pause_handle();
    let seek = player.pipeline.seek_handle();
    let stop = player.pipeline.stop_handle();
    let tap = player.pipeline.tap_handle();

    // Spawn the streaming thread: run() blocks to EOS/stop. We surface bus Warnings/Qos
    // after it returns, like scplay.
    let stream = std::thread::spawn(move || {
        let result = player.run();
        let mut warnings = Vec::new();
        while let Some(msg) = player.pipeline.bus().try_recv() {
            match msg {
                BusMessage::Warning { error, .. } => warnings.push(format!("warning: {error:?}")),
                BusMessage::Qos { lateness_ns, .. } => {
                    // Qos is already counted in the slot; only echo egregious ones.
                    let _ = lateness_ns;
                }
                _ => {}
            }
        }
        (result, warnings)
    });

    // The GUI loop (main thread). On any exit it trips `stop` and joins the streaming
    // thread. Headless mode drives a fixed frame count against the dummy driver.
    let loop_result = gui_loop(
        &cfg,
        &slot,
        zerocopy,
        GuiHandles { pause, seek, stop: &stop, tap, seek_index, duration },
        &tracks,
        &watched,
    );

    // Ensure the streaming thread stops and joins regardless of how the loop ended.
    stop.stop();
    let (run_result, warnings) = stream.join().unwrap_or_else(|_| (Ok(()), Vec::new()));
    for w in warnings {
        eprintln!("{w}");
    }
    loop_result?;
    run_result.map_err(|e| format!("run error: {e:?}"))
}

/// The transport handles + read-only tables the GUI loop drives against.
struct GuiHandles<'a> {
    pause: streamcraft_core::pipeline::PauseHandle,
    seek: SeekHandle,
    stop: &'a streamcraft_core::pipeline::StopHandle,
    tap: streamcraft_core::counters::TapHandle,
    seek_index: SeekIndex,
    duration: Timestamp,
}

/// Build the audio+subtitle track rows for the menu from `player.tracks()`. v1 marks the
/// first linked audio and the first shown subtitle as active (which is what autoplug
/// plays); live switching is future work.
fn build_track_rows(player: &Player) -> Vec<TrackRow> {
    let mut rows = Vec::new();
    let mut first_audio = true;
    let mut first_sub = true;
    for t in player.tracks() {
        let sub = t.summary.contains("subparse") || t.summary.contains("subtitle");
        let is_audio = t.summary.contains("audio device") || t.summary.contains("aacdec")
            || t.summary.contains("flacdec") || t.summary.contains("mp3dec");
        if sub {
            rows.push(TrackRow { label: short_summary(&t.summary), active: t.linked && first_sub, subtitle: true });
            if t.linked {
                first_sub = false;
            }
        } else if is_audio {
            rows.push(TrackRow { label: short_summary(&t.summary), active: t.linked && first_audio, subtitle: false });
            if t.linked {
                first_audio = false;
            }
        }
        // Video and dropped tracks are omitted from the menu (v1 shows audio/subs).
    }
    rows
}

/// Trim an autoplug summary to a short menu label (drop the sink chain tail).
fn short_summary(s: &str) -> String {
    s.split(" → ").next().unwrap_or(s).to_string()
}

/// The main GUI loop. Opens the window (dummy driver in headless mode), then per frame:
/// pump events, read the latest slot frame, build the immediate-mode UI, compose (video +
/// UI), present, and act on user intent. Returns `Ok` on a clean quit / headless finish.
fn gui_loop(
    cfg: &Config,
    slot: &FrameSlot,
    zerocopy: bool,
    h: GuiHandles,
    tracks: &[TrackRow],
    watched: &[(&'static str, ElementId)],
) -> Result<(), String> {
    // Headless: force SDL's dummy video driver so a display-less box still gets a real
    // renderer (the scope backend relies on the same property).
    if cfg.headless_frames.is_some() {
        // SAFETY: single-threaded set before any SDL init on this thread; the streaming
        // thread does not touch SDL. std::env::set_var is safe here (no other reader yet).
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("SDL_VIDEODRIVER", "dummy");
        }
    }

    let font = Font::new();
    // Ask for the GLES backend only when the decoder is zero-copy (its `video/gpu` frames need
    // the EGL importer). `Backend::open` falls back to the SDL_Renderer path if the GLES/EGL
    // DMA-BUF-import context does not stand up (e.g. the dummy headless driver), and `is_gl()`
    // reports what actually happened — but a zero-copy DECODER wired to a fallen-back
    // SDL_Renderer backend cannot display (`video/gpu` has no CPU pixels), so the caller only
    // requests zero-copy when it does not force headless (see `run`).
    let mut backend = Backend::open("streamcraft — scplay-ui", 1280, 720, &font, zerocopy)
        .map_err(|e| format!("open window: {e}"))?;
    if zerocopy && !backend.is_gl() {
        // The decoder is exporting DMA-BUFs but the GLES/EGL backend did not stand up — there
        // is no CPU-pixel path for `video/gpu`. Report it honestly; the video will not show,
        // but audio + UI still run (a degraded, not crashed, state the user can see).
        eprintln!(
            "scplay-ui: WARNING — zero-copy decoder is active but the GLES/EGL backend did not \
             initialize; video will not display. Re-run on a machine with EGL DMA-BUF import, or \
             the SDL_Renderer fallback (software/readback decode) will show video."
        );
    }
    let gl_active = backend.is_gl();

    let mut ui_state = PlayerUiState::default();
    // scope's hot/active bookkeeping — kept separate from `PlayerUiState` so `Ui::new`'s
    // borrow of it does not conflict with borrowing `ui_state` for the UI build.
    let mut ui_bookkeeping = UiState::default();
    // The seq of the frame currently in the GPU texture (0 = none), and whether any frame has
    // been uploaded (so `render` draws the persistent texture). No CPU-side frame copy is kept
    // — the slot uploads straight into the texture.
    let mut last_uploaded_seq: u64 = 0;
    let mut have_video = false;
    // Zero-copy: the token of the frame currently on screen. We release it only when the NEXT
    // frame replaces it — NOT right after import — so the decoder can't reclaim+overwrite a
    // surface that's still being displayed (that caused the on-seek stale-frame flashes).
    let mut held_token: Option<sc_vaapi::gpuframe::FrameToken> = None;

    // fps + delta tracking. `prev` holds the last stats snapshot per watched element.
    let mut prev_counters: Vec<(u64, u64)> = vec![(0, 0); watched.len()];
    let mut last_stats_at = Instant::now();
    let mut last_frame_at = Instant::now();
    // Video fps: measured from the slot's publication seq over a 1 s window. The zero-copy
    // path publishes GpuFrames (its own seq); the CPU path the triple-buffer seq.
    let video_seq = |s: &FrameSlot| if zerocopy { s.gpu_latest_seq() } else { s.latest_seq() };
    let mut fps_window_start = Instant::now();
    let mut fps_start_seq = video_seq(slot);
    let mut video_fps = 0.0f32;
    // Cache stat rows between the ~2 Hz stats refresh so the overlay is stable to read.
    let mut stat_rows: Vec<StatRow> = Vec::new();

    let headless = cfg.headless_frames;
    let mut frames_done: u32 = 0;
    let mut any_video_drawn = false;

    // Render-on-change bookkeeping: a media player should idle at ~0% CPU when the composited
    // frame is static (paused, chrome faded out, no input) instead of re-tessellating the UI
    // and re-presenting the same pixels every vsync. We hold the last-drawn state and only
    // redraw when something that affects the screen changes.
    let mut prev_mouse = (f32::NAN, f32::NAN);
    let mut prev_drawn_paused = !ui_state.paused; // force the first frame to draw
    let mut prev_drawn_eos = false;
    let mut prev_drawn_size = (0.0f32, 0.0f32);

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame_at).as_secs_f32();
        last_frame_at = now;

        let (input, events) = backend.begin_frame();
        if backend.should_close() {
            break;
        }

        // Keyboard transport (Space pause, ←/→ ∓10 s, Home 0, F fullscreen, M mute,
        // Tab/i stats, Q/Esc quit). Space/Q/M/I arrive as chars in `input.text`; the
        // navigation keys as `input.keys`; F out-of-band in `events`.
        let mut quit = false;
        for ch in input.text.chars() {
            match ch {
                ' ' => {
                    ui_state.paused = h.pause.toggle();
                }
                'q' => quit = true,
                'i' => ui_state.show_stats = !ui_state.show_stats,
                'm' => ui_state.muted = !ui_state.muted, // best-effort; no live volume prop
                _ => {}
            }
        }
        for (k, _mods) in &input.keys {
            match k {
                Key::Escape => quit = true,
                Key::Tab => ui_state.show_stats = !ui_state.show_stats,
                Key::Left => seek_relative(&h, slot, -10),
                Key::Right => seek_relative(&h, slot, 10),
                Key::Home => seek_absolute(&h, Timestamp(0)),
                _ => {}
            }
        }
        if events.toggle_fullscreen {
            backend.toggle_fullscreen();
        }
        if quit {
            break;
        }

        // Present the latest released frame.
        let mut new_frame_this_iter = false;
        if zerocopy && gl_active {
            // Zero-copy: take the paced GpuFrame, import its DMA-BUF straight into the GLES
            // external texture (no CPU touch), then release its token so the decoder reclaims
            // the surface (the anti-tear handshake). Latest-wins: we import only the freshest.
            if slot.gpu_latest_seq() != last_uploaded_seq {
                last_uploaded_seq = slot.gpu_latest_seq();
                if let Some(frame) = slot.take_gpu() {
                    let new_token = frame.token;
                    // SAFETY: the GL context is current on this (GUI) thread. `import_gpu_frame`
                    // replaces `self.current` — destroying the previous EGLImage and closing its
                    // fds — before returning.
                    let imported = unsafe { backend.import_gpu_frame(frame) };
                    if imported {
                        // The new frame is now on screen; release the PREVIOUS frame's token so
                        // the decoder reclaims that surface — but only now that it is no longer
                        // displayed. (Releasing at import time let the decoder overwrite a
                        // still-visible surface → the on-seek stale-frame flashes.)
                        if let Some(prev) = held_token.replace(new_token) {
                            slot.release_gpu_token(prev);
                        }
                        have_video = true;
                        new_frame_this_iter = true;
                    } else {
                        // Import failed → its fds are already closed; release its token now so
                        // the decoder does not stall on a presentation that will never come.
                        slot.release_gpu_token(new_token);
                    }
                }
            }
        } else {
            // CPU path: upload the latest released frame **straight into the GPU texture** —
            // no intermediate CPU copy (the slot hands its buffer to SDL_Update*Texture under a
            // brief lock).
            if let Some(seq) = slot.upload_latest_new(last_uploaded_seq, |w, h, pix, bytes| unsafe {
                backend.upload_video_raw(w, h, pix, bytes)
            }) {
                last_uploaded_seq = seq;
                have_video = true;
                new_frame_this_iter = true;
            }
        }
        if fps_window_start.elapsed() >= Duration::from_secs(1) {
            let seq = video_seq(slot);
            let elapsed = fps_window_start.elapsed().as_secs_f32();
            video_fps = (seq.saturating_sub(fps_start_seq)) as f32 / elapsed.max(1e-3);
            fps_window_start = Instant::now();
            fps_start_seq = seq;
        }

        // Refresh the stats rows ~2 Hz (cheap tap snapshots; deltas over the window).
        if ui_state.show_stats && last_stats_at.elapsed() >= Duration::from_millis(500) {
            stat_rows.clear();
            for (i, (label, id)) in watched.iter().enumerate() {
                if let Some(c) = h.tap.snapshot(*id) {
                    let (pi, po) = prev_counters[i];
                    stat_rows.push(StatRow {
                        label,
                        buffers_in: c.buffers_in,
                        buffers_out: c.buffers_out,
                        d_in: c.buffers_in.saturating_sub(pi),
                        d_out: c.buffers_out.saturating_sub(po),
                        queue_high_water: c.queue_high_water,
                        drops: c.drops,
                    });
                    prev_counters[i] = (c.buffers_in, c.buffers_out);
                }
            }
            last_stats_at = Instant::now();
        }

        // Position: running time from the tap clock (the audio DAC timeline), which is what
        // the transport bar and the timeline report — the same source scplay prints.
        let position = h.tap.now();
        let paused = h.pause.is_paused();
        let (ww, wh) = backend.window_size();

        // Render-on-change gate: redraw only when the composited frame would differ. Anything
        // that moves pixels forces a draw — a freshly presented video frame, any pointer/key
        // activity, the transport chrome still visible or mid-fade, the stats overlay (which
        // ticks on its own timer), a pause/EOS/resize state change, or startup before the
        // first video lands. Otherwise the frame is byte-identical to the last one: skip the UI
        // tessellation + GL present entirely and nap, so a paused or quiescent player costs
        // ~0% CPU instead of re-presenting the same pixels every vsync.
        let mouse_moved = prev_mouse.0 != input.mouse_x || prev_mouse.1 != input.mouse_y;
        prev_mouse = (input.mouse_x, input.mouse_y);
        let input_activity = mouse_moved
            || input.mouse_down.iter().any(|&b| b)
            || input.mouse_pressed.iter().any(|&b| b)
            || input.mouse_released.iter().any(|&b| b)
            || input.wheel != 0.0
            || !input.text.is_empty()
            || !input.keys.is_empty();
        let eos_now = slot.is_eos();
        let state_changed =
            paused != prev_drawn_paused || eos_now != prev_drawn_eos || (ww, wh) != prev_drawn_size;
        let must_draw = headless.is_some()
            || new_frame_this_iter
            || input_activity
            || !ui_state.chrome_settled_hidden()
            || ui_state.show_stats
            || state_changed
            || !any_video_drawn;

        if must_draw {
            // Build the immediate-mode UI for this frame.
            let mut ui = Ui::new(&input, &font, &mut ui_bookkeeping, (ww, wh));
            let uiframe = UiFrame {
                position: if position.is_some() { position } else { Timestamp(0) },
                duration: h.duration,
                paused,
                tracks,
                has_volume: false, // no live volume prop on the pipewire sink today (see notes)
                seek_index: &h.seek_index,
                stats: &stat_rows,
                video_fps,
                qos_drops: slot.qos_drops(),
                eos: eos_now,
            };
            // `ui` borrows `ui_bookkeeping`, `build` borrows `ui_state` — disjoint, so both
            // `&mut` borrows are live at once.
            let actions = player_ui::build(&mut ui_state, &mut ui, &uiframe, dt);
            let dl = ui.finish();

            // Compose: black clear, letterboxed video, UI on top, present.
            let drew = backend.render(have_video, &dl);
            any_video_drawn |= drew;

            // Act on user intent.
            apply_actions(&h, actions);

            prev_drawn_paused = paused;
            prev_drawn_eos = eos_now;
            prev_drawn_size = (ww, wh);
        } else {
            // Nothing on screen changed. Don't spin: nap so the CPU idles. The event pump at
            // the top of the loop still runs each wake, so input stays responsive — a longer
            // nap while paused (no frames are coming) than while merely between frames.
            let nap = if paused { 16 } else { 4 };
            std::thread::sleep(Duration::from_millis(nap));
        }

        // Headless: stop after N frames, once we have proof the compose path ran.
        if let Some(target) = headless {
            frames_done += 1;
            if frames_done >= target {
                break;
            }
            // In headless mode there is no vsync to pace us; a tiny sleep lets the streaming
            // thread produce frames between compositions so we actually exercise uploads.
            std::thread::sleep(Duration::from_millis(8));
        }
    }

    if let Some(n) = headless {
        // The headless proof: we composed `n` frames; report whether video reached the
        // texture path. A file that decodes video should have produced at least one upload.
        println!(
            "headless: composed {} frame(s); video reached the texture+letterbox path: {}",
            frames_done.min(n),
            any_video_drawn
        );
        println!(
            "headless: measured video ~{:.1} fps, paced drops {} (frame-slot Qos)",
            video_fps,
            slot.qos_drops()
        );
        if !any_video_drawn && !slot.is_eos() {
            // Not a hard failure (a very short clip may EOS before the first compose), but
            // surface it so the validation transcript is honest.
            eprintln!("headless: WARNING — no video frame was uploaded before finishing");
        }
    }
    let _ = h.stop; // used by the caller to join
    Ok(())
}

/// Carry out one frame's UI actions against the transport handles.
fn apply_actions(h: &GuiHandles, a: UiActions) {
    if a.toggle_pause {
        h.pause.toggle();
    }
    if let Some(t) = a.seek_to {
        seek_absolute(h, t);
    }
    // toggle_mute is a no-op today (no live volume prop); left here for when one lands.
    let _ = a.toggle_mute;
}

/// Seek to an absolute stream time via the seek index (resolve time→byte, then
/// `seek(byte, landed)` — seek to the RESOLVED cue, not the request, per SeekIndex::resolve
/// and scplay's digit-seek).
fn seek_absolute(h: &GuiHandles, target: Timestamp) {
    if let Some((byte, landed)) = h.seek_index.resolve(target, h.duration) {
        h.seek.seek(byte, landed);
    }
}

/// Seek by `delta` seconds relative to the current position (the ←/→ keys). Clamped to
/// `[0, duration]`.
fn seek_relative(h: &GuiHandles, _slot: &FrameSlot, delta: i64) {
    let pos = h.tap.now().nanos().unwrap_or(0) as i64;
    let dur = h.duration.nanos().map(|d| d as i64).unwrap_or(i64::MAX);
    let target = (pos + delta * 1_000_000_000).clamp(0, dur.max(0));
    seek_absolute(h, Timestamp(target as u64));
}
