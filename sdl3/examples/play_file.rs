//! `play_file` — play a real-world MKV's video track in a window (spec: Milestone
//! applications §5). `filesrc ! mkvdemux ! <matching video decoder> ! sdl3videosink`,
//! with every other discovered track (audio codecs we don't decode yet, subtitles)
//! linked to a drop-sink — an unlinked dynamic pad would otherwise accumulate output
//! without bound (the spec's buffer-then-drop policy for unlinked pads is future core
//! work; explicit drops are today's honest equivalent).
//!
//! Decoder selection is negotiation-driven mini-autoplugging: for each discovered pad,
//! try each registered video decoder — the link only succeeds when the demuxer's
//! announced family (e.g. `h265/annexb` for a V_MPEGH/ISO/HEVC track) matches the
//! decoder's sink offer. First video pad that links gets the window; the rest drop.
//!
//! ```text
//! cargo run --release -p sc-sdl3 --example play_file -- FILE.mkv
//! ```

use sc_mkv::ebml::id;
use sc_mkv::MkvDemux;
use sc_sdl3::Sdl3VideoSink;
use streamcraft_core::element::Element;
use streamcraft_core::error::Error;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSrc;
use streamcraft_elements::testing::TestSink;

use std::io::Read;

/// The stream head: everything before the first Cluster (EBML Header + Segment
/// metadata — SeekHead/Info/Tracks/Tags/…). 8 MiB is generous for real files.
fn header_prefix(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut head = vec![0u8; 8 * 1024 * 1024];
    let n = f.read(&mut head)?;
    head.truncate(n);
    let cluster = head
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("no Cluster in the first 8 MiB — not a (supported) MKV?");
    head.truncate(cluster);
    Ok(head)
}

/// Try each video decoder against a demux pad; `Some` once one negotiates.
fn try_video_decoders(
    p: &mut Pipeline,
    pad: (streamcraft_core::id::ElementId, &str),
) -> Option<streamcraft_core::id::ElementId> {
    // Constructed fresh per attempt: a failed link leaves an unlinked spare element
    // in the graph, which is harmless (it never runs), so try-in-place is fine.
    let attempts: Vec<(&str, Box<dyn Element>)> = vec![
        ("h265", Box::new(sc_h265::H265Dec::new())),
        ("h264", Box::new(sc_h264::H264Dec::new())),
        ("vp8", Box::new(sc_vp8::Vp8Dec::new())),
        ("vp9", Box::new(sc_vp9::Vp9Dec::new())),
        ("av1", Box::new(sc_av1::Av1Dec::new())),
    ];
    for (name, dec) in attempts {
        let dec_id = p.add_boxed(dec);
        match p.link(pad, (dec_id, "sink")) {
            Ok(_) => {
                println!("  {} -> {name}dec", pad.1);
                return Some(dec_id);
            }
            Err(Error::Resource(_)) | Err(Error::Todo(_)) => continue,
            Err(e) => {
                eprintln!("  link error on {}: {e:?}", pad.1);
                return None;
            }
        }
    }
    None
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--stats`: print per-element counter deltas every 3 s (buffers in/out, ring
    // high-water, drops) — a poor man's scope; a stalled pipeline shows as every
    // delta hitting zero while the process lives.
    let stats = if let Some(i) = args.iter().position(|a| a == "--stats") {
        args.remove(i);
        true
    } else {
        false
    };
    // `--probe`: demux only — every pad to a drop-sink, no decoder, no window.
    // Isolates container-side behavior (throughput, memory) from decode/display.
    let probe = if let Some(i) = args.iter().position(|a| a == "--probe") {
        args.remove(i);
        true
    } else {
        false
    };
    // `--max-secs N`: stop the pipeline cleanly after N wall seconds (cooperative
    // StopHandle) — bounded runs for diagnostics/profiling where SIGKILL would
    // truncate a heaptrack capture.
    let max_secs: Option<u64> = args
        .iter()
        .position(|a| a == "--max-secs")
        .map(|i| {
            let v = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
            args.drain(i..=i + 1);
            v
        });
    let Some(path) = args.first().cloned() else {
        eprintln!("usage: play_file [--probe] [--max-secs N] FILE.mkv");
        std::process::exit(2);
    };

    let header = header_prefix(&path).expect("read header");

    let mut p = Pipeline::new();
    // 1080p I420 is ~3.1 MiB/frame; 4 MiB slots leave headroom up to ~1600x1300.
    p.set_pool(4 * 1024 * 1024, 24);

    let src = p.add(FileSrc::new(&path));
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src ! demux");

    let added = p.preroll().expect("preroll");
    println!("{} track pad(s) discovered:", added.len());

    let mut video_dec = None;
    let mut audio_dec = None;
    for ap in &added {
        if !probe && video_dec.is_none() {
            if let Some(dec) = try_video_decoders(&mut p, (ap.element, &ap.name)) {
                video_dec = Some(dec);
                continue;
            }
        }
        // First audio track that negotiates gets decoded and played (milestone:
        // play A+V — the audio sink provides the pipeline clock, so video paces
        // on the DAC; spec: Clock providers, audio-master sync). Same
        // negotiation-driven autoplug as video: the link only succeeds when the
        // pad's per-track family matches the decoder's sink offer.
        if !probe && audio_dec.is_none() {
            let dec_id = p.add(sc_aac::AacDec::new());
            match p.link((ap.element, &ap.name), (dec_id, "sink")) {
                Ok(_) => {
                    println!("  {} -> aacdec", ap.name);
                    audio_dec = Some(dec_id);
                    continue;
                }
                Err(Error::Resource(_)) | Err(Error::Todo(_)) => {} // not aac — fall through
                Err(e) => eprintln!("  audio link error on {}: {e:?}", ap.name),
            }
        }
        // Everything else — codecs without decoders, subtitles, extra tracks —
        // drains into a drop-sink so nothing backs up.
        let (sink, _stats) = TestSink::new();
        let drop_id = p.add(sink);
        p.link((ap.element, &ap.name), (drop_id, "sink"))
            .unwrap_or_else(|e| panic!("drop-link {}: {e:?}", ap.name));
        println!("  {} -> (dropped: no decoder)", ap.name);
    }
    let dec = match video_dec {
        Some(d) => Some(d),
        None if probe => None,
        None => {
            eprintln!("no track linked to a video decoder");
            std::process::exit(1);
        }
    };

    let mut sink_id = None;
    if let Some(dec) = dec {
        // The decoder gets its OWN pool (spec: pool negotiation, explicit v1 —
        // set_element_pool): decoded frames must never compete with the shared
        // default pool that filesrc reads and the demuxer's held input backlog
        // drink from. With one shared pool the graph deadlocks: the output-blocked
        // demuxer pins its input slots, the pool hits its cap, and the decoder —
        // the only element that could unblock the chain — can never allocate the
        // frame it is carrying (measured: outstanding=24/24, all counters flat).
        p.set_element_pool(dec, 4 * 1024 * 1024, 24);
        let sink = p.add_boxed(Box::new(Sdl3VideoSink::new().with_title("streamcraft — play_file")));
        p.link((dec, "src"), (sink, "sink")).expect("dec ! sink");
        sink_id = Some(sink);
    }
    if let Some(adec) = audio_dec {
        // s16 interleaved audio/raw straight into the device sink (the same
        // chain shape as the flac "play an audio file" milestone). The pipewire
        // sink provides the AudioDeviceClock — sinks-first selection makes it
        // the pipeline clock, so the video sink's wait_until paces on the DAC.
        let asink = p.add(sc_pipewire::PipeWireAudioSink::new());
        p.link((adec, "src"), (asink, "sink")).expect("aacdec ! audiosink");
    }

    println!("playing {path} — 'p'⏎ pause/resume, 'q'⏎ quit (or close the window / Ctrl-C)…");
    if stats {
        // Watch the graph breathe: named per-element `buffers in→out (+delta)` lines
        // every 3 s. Reads live tap counters — zero cost to the streaming threads.
        let tap = p.tap_handle();
        let watched: Vec<(String, streamcraft_core::id::ElementId)> = [
            ("filesrc", Some(src)),
            ("mkvdemux", Some(demux)),
            ("dec", dec),
        ]
        .into_iter()
        .filter_map(|(n, id)| id.map(|id| (n.to_string(), id)))
        .chain(sink_id.map(|s| ("sink".to_string(), s)))
        .collect();
        std::thread::spawn(move || {
            let mut prev: Vec<(u64, u64)> = vec![(0, 0); watched.len()];
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3));
                let mut line = String::from("stats:");
                for (i, (name, id)) in watched.iter().enumerate() {
                    if let Some(c) = tap.snapshot(*id) {
                        let (pi, po) = prev[i];
                        line.push_str(&format!(
                            " {name} {}→{} (+{}/+{}) qhw={} drops={} |",
                            c.buffers_in,
                            c.buffers_out,
                            c.buffers_in - pi,
                            c.buffers_out - po,
                            c.queue_high_water,
                            c.drops,
                        ));
                        prev[i] = (c.buffers_in, c.buffers_out);
                    }
                }
                eprintln!("{line}");
            }
        });
    }
    if let Some(secs) = max_secs {
        let stop = p.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.stop();
        });
    }
    // Transport control from stdin (spec: Clocking — pause is a clock op): running
    // time freezes, the audio sink holds its hardware, video holds its frame; resume
    // continues exactly where it left off. Line-based — no raw-mode termios, no deps.
    {
        let pause = p.pause_handle();
        let stop = p.stop_handle();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if std::io::BufRead::read_line(&mut stdin.lock(), &mut line).unwrap_or(0) == 0 {
                    return;
                }
                match line.trim() {
                    "p" => {
                        let paused = pause.toggle();
                        println!("{}", if paused { "⏸ paused" } else { "▶ playing" });
                    }
                    "q" => {
                        stop.stop();
                        return;
                    }
                    _ => {}
                }
            }
        });
    }
    let result = p.run();
    // Surface sink warnings — a display/GPU fallback is reported here, and silence
    // plus no window would otherwise be undiagnosable.
    while let Some(msg) = p.bus().try_recv() {
        match msg {
            streamcraft_core::bus::BusMessage::Warning { error, .. } => {
                eprintln!("warning: {error:?}")
            }
            streamcraft_core::bus::BusMessage::Qos { lateness_ns, .. } => {
                eprintln!("qos: frame dropped {:.1} ms late", lateness_ns as f64 / 1e6)
            }
            _ => {}
        }
    }
    match result {
        Ok(()) => println!("done."),
        Err(e) => {
            eprintln!("run error: {e:?}");
            std::process::exit(1);
        }
    }
}
