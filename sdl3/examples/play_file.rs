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
//! cargo run --release -p pf-sdl3 --example play_file -- FILE.mkv
//! ```

use pf_mkv::ebml::id;
use pf_mkv::{parse_cues, parse_seek_head, MkvDemux};

// The adopted h264 decoder allocates ~1.5M times/s internally (upstream churn,
// see PLAN); glibc malloc makes that the decode bottleneck (one pegged core at
// ~12-22 fps for 720p25). mimalloc buys the headroom until the churn is fixed
// at the source. Example-only — the libraries stay allocator-agnostic.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
use pf_sdl3::Sdl3VideoSink;
use profluens_core::element::Element;
use profluens_core::error::Error;
use profluens_core::pipeline::{Pipeline, SeekIndex};
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSrc;
use profluens_elements::testing::TestSink;

use std::io::Read;

/// The stream head: everything before the first Cluster (EBML Header + Segment
/// metadata — SeekHead/Info/Tracks/Tags/…). 8 MiB is generous for real files.
/// Build the time→byte seek index for `path`: SeekHead (from the already-read
/// header) → pread the Cues range → parse. Returns the index (entries empty when
/// the file has no Cues — the proportional fallback then applies) and the
/// declared duration.
fn build_seek_index(path: &str, header: &[u8], file_len: u64) -> (SeekIndex, Option<u64>) {
    use std::os::unix::fs::FileExt;
    let mut index = SeekIndex { entries: Vec::new(), file_len: Some(file_len) };
    // Duration comes from the header's Segment Info (same probe the demuxer runs).
    let mut probe = pf_mkv::MatroskaReader::new();
    let _ = probe.push(header);
    let duration = probe.duration_ns();
    let Some(info) = parse_seek_head(header) else {
        return (index, duration);
    };
    if let Some(cues_pos) = info.cues_pos {
        let abs = info.segment_data_start + cues_pos;
        if abs < file_len {
            // The Cues master is small (a few bytes per keyframe cluster); read a
            // bounded chunk from its offset — parse_cues reads the element's own
            // declared size and ignores the tail.
            let want = ((file_len - abs) as usize).min(4 * 1024 * 1024);
            let mut buf = vec![0u8; want];
            if let Ok(f) = std::fs::File::open(path) {
                if f.read_exact_at(&mut buf, abs).is_ok() {
                    index.entries = parse_cues(&buf, info.segment_data_start, info.timestamp_scale);
                }
            }
        }
    }
    (index, duration)
}

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
    pad: (profluens_core::id::ElementId, &str),
) -> Option<profluens_core::id::ElementId> {
    // Constructed fresh per attempt: a failed link leaves an unlinked spare element
    // in the graph, which is harmless (it never runs), so try-in-place is fine.
    let mut attempts: Vec<(&str, Box<dyn Element>)> = vec![
        ("h265", Box::new(pf_h265::H265Dec::new())),
        ("h264", Box::new(pf_h264::H264Dec::new())),
        ("vp8", Box::new(pf_vp8::Vp8Dec::new())),
        ("vp9", Box::new(pf_vp9::Vp9Dec::new())),
        ("av1", Box::new(pf_av1::Av1Dec::new())),
    ];
    // Hardware first: the probe gates construction (no VA-API device or
    // `PF_NO_VAAPI` set → no attempt), and a non-h264 track simply fails the
    // link and falls through to the software decoders.
    if let Some(hw) = pf_vaapi::video_decoder_for("h264/annexb") {
        attempts.insert(0, ("vaapih264", hw));
    }
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

/// Live RTSP playback (spec: Milestone applications — network streaming).
///
/// The app is the controller (spec: no-bins): it runs the RTSP client dance —
/// DESCRIBE (RFC 2326 §10.2) for the SDP, SETUP (§10.4) per media with a
/// locally bound RTP/RTCP port pair, PLAY (§10.5) once the pipeline is
/// prerolled — then keeps the session alive from a helper thread and tears it
/// down on exit. The pipeline itself is the plain receive chain; the video
/// track feeds the same decoder autoplug (hardware-first) as file playback.
///
/// v1 scope: the first H.264 video media, UDP transport. Audio media are
/// skipped (no Opus decoder in-tree yet); other codecs fail loudly.
fn play_rtsp(url: &str, stats: bool, max_secs: Option<u64>) {
    use pf_rtp::elements::{RtpH264Depay, RtpSession, RtpStreamDesc, UdpSrc};
    use pf_rtsp::client::RtspClient;
    use pf_rtsp::sdp::{decode_base64, MediaKind};

    let mut client = RtspClient::connect(url).expect("rtsp connect");
    let described = client.describe().expect("DESCRIBE");
    let base = client.base_url().to_string();

    // The first H.264 video media (RFC 6184 §8.2.2: media subtype "H264").
    let media = described
        .sdp
        .media
        .iter()
        .find(|m| {
            m.kind == MediaKind::Video
                && m.payload_types.iter().any(|pt| {
                    m.rtpmap.get(pt).is_some_and(|r| r.encoding.eq_ignore_ascii_case("H264"))
                })
        })
        .expect("no H.264 video media in the SDP");
    let pt = *media
        .payload_types
        .iter()
        .find(|pt| media.rtpmap.get(pt).is_some_and(|r| r.encoding.eq_ignore_ascii_case("H264")))
        .unwrap();
    let clock_rate = media.rtpmap[&pt].clock_rate;
    // Out-of-band parameter sets (RFC 6184 §8.1 sprop-parameter-sets):
    // comma-separated base64 NAL units in the fmtp line.
    let sprop: Vec<Vec<u8>> = media
        .fmtp
        .get(&pt)
        .and_then(|f| f.params.get("sprop-parameter-sets"))
        .map(|v| v.split(',').filter_map(decode_base64).collect())
        .unwrap_or_default();

    // An even/odd local port pair (RFC 3550 §11: RTP on the even port, RTCP
    // one above). Bind ephemeral until the kernel hands us an even port.
    let (rtp_sock, _rtcp_sock, rtp_port) = loop {
        let a = std::net::UdpSocket::bind("0.0.0.0:0").expect("bind rtp");
        let port = a.local_addr().unwrap().port();
        if port % 2 == 0 && port < u16::MAX {
            if let Ok(b) = std::net::UdpSocket::bind(("0.0.0.0", port + 1)) {
                break (a, b, port);
            }
        }
    };

    let control = pf_rtsp::client::resolve_control(&base, media.control.as_deref().unwrap_or(""));
    let transport = client.setup(&control, rtp_port).expect("SETUP");
    println!(
        "rtsp: {} pt={pt} rate={clock_rate} sprop_nals={} transport={}",
        url,
        sprop.len(),
        transport.raw.trim()
    );

    // The receive pipeline. Decoded 720p NV12 ≈ 1.4 MiB — same pool shape as
    // file playback, RTP-side buffers are small.
    let mut p = Pipeline::new();
    p.set_pool(4 * 1024 * 1024, 24);
    let src = p.add(UdpSrc::from_socket(rtp_sock));
    let session = p.add(RtpSession::new(vec![RtpStreamDesc { payload_type: pt, clock_rate }]));
    p.link((src, "src"), (session, "sink")).expect("udpsrc ! session");
    let added = p.preroll().expect("preroll");
    let session_pad = added.first().expect("session stream pad");

    let depay = p.add(RtpH264Depay::new(sprop));
    p.link((session_pad.element, &session_pad.name), (depay, "sink")).expect("session ! depay");
    let dec = try_video_decoders(&mut p, (depay, "src")).expect("no decoder linked for h264");
    p.set_element_pool(dec, 4 * 1024 * 1024, 24);
    p.set_queue_capacity(dec, 16);
    let sink = p.add_boxed(Box::new(Sdl3VideoSink::new().with_title("profluens — rtsp")));
    p.link((dec, "src"), (sink, "sink")).expect("dec ! sink");

    // Everything is linked: start the media flowing, then run.
    client.play().expect("PLAY");
    println!("playing {url} — 'q'⏎ quits (or close the window / Ctrl-C)…");

    // Keepalive from a helper thread (RFC 2326 §12.37: the session times out —
    // 60 s default — without liveness); TEARDOWN when the pipeline stops.
    let stop = p.stop_handle();
    let keepalive_secs = client.session().map(|s| s.timeout_secs).unwrap_or(60).max(2) / 2;
    let ka = std::thread::spawn(move || {
        let mut last = std::time::Instant::now();
        while !stop.is_stopped() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if last.elapsed().as_secs() >= keepalive_secs {
                if client.keepalive().is_err() {
                    eprintln!("rtsp: keepalive failed — server gone?");
                    stop.stop();
                    return;
                }
                last = std::time::Instant::now();
            }
        }
        let _ = client.teardown();
    });

    if stats {
        let tap = p.tap_handle();
        let watched = vec![("udpsrc", src), ("session", session), ("depay", depay), ("dec", dec), ("sink", sink)];
        std::thread::spawn(move || {
            let mut prev = vec![(0u64, 0u64); watched.len()];
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3));
                let mut line = String::from("stats:");
                for (i, (name, id)) in watched.iter().enumerate() {
                    if let Some(c) = tap.snapshot(*id) {
                        let (pi, po) = prev[i];
                        line.push_str(&format!(
                            " {name} {}→{} (+{}/+{}) drops={} |",
                            c.buffers_in,
                            c.buffers_out,
                            c.buffers_in - pi,
                            c.buffers_out - po,
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
    {
        let stop = p.stop_handle();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if std::io::BufRead::read_line(&mut stdin.lock(), &mut line).unwrap_or(0) == 0 {
                    return;
                }
                if line.trim() == "q" {
                    stop.stop();
                    return;
                }
            }
        });
    }

    if let Err(e) = p.run() {
        eprintln!("pipeline error: {e:?}");
    }
    let _ = ka.join();
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
        eprintln!("usage: play_file [--probe] [--max-secs N] FILE.mkv | rtsp://HOST[:554]/PATH");
        std::process::exit(2);
    };

    // A network stream: the RTSP control dance is app policy (spec: no-bins —
    // controllers live in the application), the pipeline is
    // `udpsrc ! rtpsession ! rtph264depay ! <decoder> ! sdl3videosink`.
    if path.starts_with("rtsp://") {
        play_rtsp(&path, stats, max_secs);
        return;
    }

    let header = header_prefix(&path).expect("read header");

    // Time→byte seek index (spec: flush/seek — mapping time to a byte is the seek
    // issuer's job): SeekHead in the header points at the Cues (written after the
    // last Cluster); pread that range and parse. A cue-less file still seeks via
    // the proportional file_len fallback + the demuxer's cluster-scan resync.
    let file_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let (seek_index, duration_ns) = build_seek_index(&path, &header, file_len);
    println!(
        "seek index: {} cue points{}, duration {}",
        seek_index.entries.len(),
        if seek_index.entries.is_empty() { " (proportional fallback)" } else { "" },
        duration_ns.map(|ns| format!("{:.1}s", ns as f64 / 1e9)).unwrap_or_else(|| "unknown".into()),
    );

    let mut p = Pipeline::new();
    // 1080p I420 is ~3.1 MiB/frame; 4 MiB slots leave headroom up to ~1600x1300.
    p.set_pool(4 * 1024 * 1024, 24);

    p.set_seek_index(seek_index.clone());

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
            let dec_id = p.add(pf_aac::AacDec::new());
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
    let mut audio_sink_id = None;
    if let Some(dec) = dec {
        // The decoder gets its OWN pool (spec: pool negotiation, explicit v1 —
        // set_element_pool): decoded frames must never compete with the shared
        // default pool that filesrc reads and the demuxer's held input backlog
        // drink from. With one shared pool the graph deadlocks: the output-blocked
        // demuxer pins its input slots, the pool hits its cap, and the decoder —
        // the only element that could unblock the chain — can never allocate the
        // frame it is carrying (measured: outstanding=24/24, all counters flat).
        p.set_element_pool(dec, 4 * 1024 * 1024, 24);
        // A deep inbound ring (the classic post-demux queue): the demuxer's
        // thread emits BOTH tracks, and a full ring on either pad blocks it —
        // starving the other track. With the default 4-batch rings it ping-pongs
        // between blocked-on-video and blocked-on-audio: video halves, audio
        // underruns (audible stutter). Depth here is the interleave slack.
        p.set_queue_capacity(dec, 16);
        let sink = p.add_boxed(Box::new(Sdl3VideoSink::new().with_title("profluens — play_file")));
        p.link((dec, "src"), (sink, "sink")).expect("dec ! sink");
        sink_id = Some(sink);
    }
    if let Some(adec) = audio_dec {
        // The audio decoder gets its OWN small-slot pool (pool negotiation v1,
        // same lesson as the video decoder above but inverted): a decoded AAC
        // frame is ~4 KB, and `try_alloc` hands out whole slots — from the
        // shared pool that is a 4 MiB slot pinned per 4 KB frame, so a couple
        // dozen in-flight audio buffers exhausted the pool: the demuxer starved
        // (video froze) and the audio sink underran (stutter). Worse, the DAC
        // is the *pipeline* clock now — an underrun freezes video pacing too.
        p.set_element_pool(adec, 64 * 1024, 64);
        p.set_queue_capacity(adec, 64); // interleave slack — see the video note
        // Diagnostics: PF_AUDIO_DROP=1 decodes audio but drops it (no device, no
        // audio clock); PF_FORCE_WALL=1 keeps the device but paces on the wall
        // clock. Both isolate "audio chain CPU" from "audio-master clock" when
        // hunting pacing regressions.
        if std::env::var_os("PF_AUDIO_DROP").is_some() {
            let (tsink, _stats) = TestSink::new();
            let drop_id = p.add(tsink);
            p.link((adec, "src"), (drop_id, "sink")).expect("aacdec ! drop");
            println!("  (audio decoded but dropped — PF_AUDIO_DROP)");
        } else {
            // s16 interleaved audio/raw straight into the device sink (the same
            // chain shape as the flac "play an audio file" milestone). The pipewire
            // sink provides the AudioDeviceClock — sinks-first selection makes it
            // the pipeline clock, so the video sink's wait_until paces on the DAC.
            let asink = p.add(pf_pipewire::PipeWireAudioSink::new());
            p.link((adec, "src"), (asink, "sink")).expect("aacdec ! audiosink");
            audio_sink_id = Some(asink);
            if std::env::var_os("PF_FORCE_WALL").is_some() {
                p.set_clock(std::sync::Arc::new(profluens_core::clock::InstantClock::new()));
                println!("  (wall clock forced — PF_FORCE_WALL)");
            }
        }
    }

    println!("playing {path} — 'p'⏎ pause/resume, 'q'⏎ quit (or close the window / Ctrl-C)…");
    if stats {
        // Watch the graph breathe: named per-element `buffers in→out (+delta)` lines
        // every 3 s. Reads live tap counters — zero cost to the streaming threads.
        let tap = p.tap_handle();
        let watched: Vec<(String, profluens_core::id::ElementId)> = [
            ("filesrc", Some(src)),
            ("mkvdemux", Some(demux)),
            ("dec", dec),
            ("adec", audio_dec),
            ("asink", audio_sink_id),
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
        let seek = p.seek_handle();
        let tap = p.tap_handle();
        let index = seek_index;
        let duration = duration_ns.map(Timestamp).unwrap_or(Timestamp::NONE);
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
                    // Digits seek to n×10% of the duration (spec: flush/seek) —
                    // the scope-free test hook for time seeking.
                    d if d.len() == 1 && d.as_bytes()[0].is_ascii_digit() => {
                        let Some(dur) = duration.nanos() else {
                            println!("seek: unknown duration");
                            continue;
                        };
                        let frac = (d.as_bytes()[0] - b'0') as u64;
                        let target = Timestamp(dur / 10 * frac);
                        match index.resolve(target, duration) {
                            Some((byte, landed)) => {
                                // Seek to the RESOLVED cue time, not the request —
                                // rebasing to the request while content resumes at
                                // the earlier cue leaves video permanently late
                                // (frozen picture over playing audio).
                                seek.seek(byte, landed);
                                println!(
                                    "⇥ seek to {:.1}s → landed {:.1}s (byte {byte}); position now {:.1}s",
                                    target.0 as f64 / 1e9,
                                    landed.0 as f64 / 1e9,
                                    tap.now().nanos().map(|n| n as f64 / 1e9).unwrap_or(-1.0),
                                );
                            }
                            None => println!("seek: no mapping available"),
                        }
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
            profluens_core::bus::BusMessage::Warning { error, .. } => {
                eprintln!("warning: {error:?}")
            }
            profluens_core::bus::BusMessage::Qos { lateness_ns, .. } => {
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
