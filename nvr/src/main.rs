//! `scraft-nvr` — N RTSP cameras in, rotated self-contained MKV segments out,
//! with a live mosaic wall (spec: Milestone applications — the framework's
//! broadest stress test: live clocks, fan-out, fan-in, segmented muxing with
//! back-patches, long-run memory flatness).
//!
//! The app is the controller (spec: no-bins): it runs the RTSP client dance per
//! camera — DESCRIBE (RFC 2326 §10.2), SETUP (§10.4) with a local RTP/RTCP port
//! pair, PLAY (§10.5) once the pipeline stands — keeps sessions alive from a
//! helper thread, and owns ONE pipeline for all cameras:
//!
//! ```text
//! per cam i:  udpsrc_i ! rtpsession_i ! rtph264depay_i ! tee_i ┬ mkvsegmentsink_i
//!                                                              └ h264dec_i ! mosaic.sink_i
//! mosaic ! sdl3videosink            (--no-mosaic: depay_i ! mkvsegmentsink_i, N disjoint chains)
//! ```
//!
//! Pool discipline (the hard-won per-element-pool lessons): RTP packets ride
//! small-slot pools, access units mid-slot pools, decoded frames big-slot pools
//! — a held small buffer must never pin a frame-sized slot.
//!
//! ```text
//! scraft-nvr [--record DIR] [--segment SECS] [--no-mosaic] [--cell WxH]
//!            [--stats] [--max-secs N] rtsp://... [rtsp://... ...]
//! ```
//! `q`⏎ stops cleanly (segments finalize via the sink's teardown path).

// Decode churn: the adopted h264 decoder allocates heavily upstream; mimalloc
// buys the headroom (the play_file lesson). Example/app-only — libraries stay
// allocator-agnostic.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::net::UdpSocket;

use sc_rtp::elements::{RtpH264Depay, RtpSession, RtpStreamDesc, UdpSrc};
use sc_rtsp::client::RtspClient;
use sc_rtsp::sdp::{decode_base64, MediaKind};
use sc_vaapi::h264parse::{parse_sps, NAL_SPS};
use streamcraft_core::id::ElementId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_nvr::segment_sink::SegmentStats;
use streamcraft_nvr::mosaic::CamGeom;
use streamcraft_nvr::{MkvSegmentSink, Mosaic};

/// One camera's negotiated transport + stream parameters, ready to wire.
struct Cam {
    name: String,
    client: RtspClient,
    rtp_sock: UdpSocket,
    pt: u8,
    clock_rate: u32,
    sprop: Vec<Vec<u8>>,
    geom: CamGeom,
}

/// The RTSP dance for one camera through SETUP (PLAY happens once the pipeline
/// stands): the `play_file` client flow, factored for N cameras.
fn setup_cam(url: &str, index: usize) -> Result<Cam, String> {
    let mut client = RtspClient::connect(url).map_err(|e| format!("{url}: connect: {e:?}"))?;
    let described = client.describe().map_err(|e| format!("{url}: DESCRIBE: {e:?}"))?;
    let base = client.base_url().to_string();

    // The first H.264 video media (RFC 6184 §8.2.2: subtype "H264").
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
        .ok_or_else(|| format!("{url}: no H.264 video media in the SDP"))?;
    let pt = *media
        .payload_types
        .iter()
        .find(|pt| media.rtpmap.get(pt).is_some_and(|r| r.encoding.eq_ignore_ascii_case("H264")))
        .unwrap();
    let clock_rate = media.rtpmap[&pt].clock_rate;
    // Out-of-band parameter sets (RFC 6184 §8.1 sprop-parameter-sets).
    let sprop: Vec<Vec<u8>> = media
        .fmtp
        .get(&pt)
        .and_then(|f| f.params.get("sprop-parameter-sets"))
        .map(|v| v.split(',').filter_map(decode_base64).collect())
        .unwrap_or_default();
    // The wall's geometry comes from the SPS (the recorder parses it again for
    // its CodecPrivate — app-side policy either way).
    // Crop AND coded dims: the decoder emits MB-aligned planes (H.264
    // §7.4.2.1.1) — the wall needs both (mosaic module docs).
    let geom = sprop
        .iter()
        .find(|n| n.first().map(|b| b & 0x1F) == Some(NAL_SPS))
        .and_then(|n| parse_sps(n))
        .map(|s| CamGeom {
            crop_w: s.width(),
            crop_h: s.height(),
            coded_w: s.width_in_mbs() * 16,
            coded_h: s.height_in_mbs() * 16,
        })
        .ok_or_else(|| format!("{url}: SDP carries no parseable SPS (sprop-parameter-sets)"))?;

    // An even/odd local port pair (RFC 3550 §11: RTP even, RTCP one above).
    let (rtp_sock, _rtcp_sock, rtp_port) = loop {
        let a = UdpSocket::bind("0.0.0.0:0").map_err(|e| format!("bind rtp: {e}"))?;
        let port = a.local_addr().unwrap().port();
        if port % 2 == 0 && port < u16::MAX {
            if let Ok(b) = UdpSocket::bind(("0.0.0.0", port + 1)) {
                break (a, b, port);
            }
        }
    };
    let control = sc_rtsp::client::resolve_control(&base, media.control.as_deref().unwrap_or(""));
    client.setup(&control, rtp_port).map_err(|e| format!("{url}: SETUP: {e:?}"))?;
    println!("cam{index}: {url} — {}x{} pt={pt} rate={clock_rate}", geom.crop_w, geom.crop_h);
    Ok(Cam { name: format!("cam{index}"), client, rtp_sock, pt, clock_rate, sprop, geom })
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take_val = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).map(|i| {
            let v = args.get(i + 1).cloned().unwrap_or_default();
            args.drain(i..=i + 1);
            v
        })
    };
    let record_dir = take_val("--record").unwrap_or_else(|| "recordings".into());
    let segment_secs: f64 = take_val("--segment").and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let max_secs: Option<u64> = take_val("--max-secs").and_then(|s| s.parse().ok());
    let cell: (u32, u32) = take_val("--cell")
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((480, 270));
    let mut args = args; // release the closure borrow
    let mut take_flag = |flag: &str| -> bool {
        args.iter().position(|a| a == flag).map(|i| args.remove(i)).is_some()
    };
    let no_mosaic = take_flag("--no-mosaic");
    let stats = take_flag("--stats");
    let args = args;
    if args.is_empty() {
        eprintln!(
            "usage: scraft-nvr [--record DIR] [--segment SECS] [--no-mosaic] [--cell WxH] \
             [--stats] [--max-secs N] rtsp://URL..."
        );
        std::process::exit(2);
    }

    // --- RTSP dance per camera (through SETUP) --------------------------------
    let mut cams: Vec<Cam> = Vec::new();
    for (i, url) in args.iter().enumerate() {
        match setup_cam(url, i) {
            Ok(c) => cams.push(c),
            Err(e) => {
                eprintln!("scraft-nvr: {e}");
                std::process::exit(1);
            }
        }
    }
    let n = cams.len();

    // --- one pipeline for the whole wall --------------------------------------
    let mut p = Pipeline::new();
    // The shared/default pool serves the mosaic canvas and anything unpinned;
    // every high-traffic element below gets its own right-sized pool.
    p.set_pool(4 * 1024 * 1024, 16);

    // Sources + sessions first — preroll discovers every session's stream pad.
    let mut chain: Vec<(ElementId, ElementId)> = Vec::new(); // (udpsrc, session)
    for cam in &mut cams {
        let sock = cam.rtp_sock.try_clone().expect("clone rtp socket");
        let src = p.add(UdpSrc::from_socket(sock));
        // RTP packets are ≤ ~1.4 KB: small slots, deep — a held packet must
        // never pin a frame-sized slot (the sender-side deadlock lesson).
        p.set_element_pool(src, 2048, 512);
        let ses = p.add(RtpSession::new(vec![RtpStreamDesc {
            payload_type: cam.pt,
            clock_rate: cam.clock_rate,
        }]));
        p.set_element_pool(ses, 2048, 512);
        p.link((src, "src"), (ses, "sink")).expect("udpsrc ! session");
        chain.push((src, ses));
    }
    let added = p.preroll().expect("preroll");

    // Per camera: depay → (tee → recorder + decoder) or → recorder directly.
    let mosaic_geom: Vec<CamGeom> = cams.iter().map(|c| c.geom).collect();
    let mut seg_stats: Vec<std::sync::Arc<std::sync::Mutex<SegmentStats>>> = Vec::new();
    let mut watch: Vec<(String, ElementId)> = Vec::new();
    let mut mosaic_id = None;
    if !no_mosaic {
        let mos = Mosaic::new(mosaic_geom.clone(), cell.0, cell.1);
        let (ow, oh) = mos.out_dims();
        println!("mosaic: {n} cams → {ow}x{oh} ({}x{} cells)", cell.0, cell.1);
        let id = p.add(mos);
        p.set_element_pool(id, 4 * 1024 * 1024, 8);
        mosaic_id = Some(id);
    }
    for (i, cam) in cams.iter().enumerate() {
        let (_, ses) = chain[i];
        let pad = added
            .iter()
            .find(|ap| ap.element == ses)
            .unwrap_or_else(|| panic!("{}: session announced no pad", cam.name));
        let depay = p.add(RtpH264Depay::new(cam.sprop.clone()));
        // Access units: mid-size slots (a 1080p IDR is a few hundred KB).
        p.set_element_pool(depay, 1024 * 1024, 24);
        p.set_queue_capacity(depay, 16);
        p.link((pad.element, &pad.name), (depay, "sink")).expect("session ! depay");

        let seg = MkvSegmentSink::new(&record_dir, &cam.name, segment_secs, &cam.sprop);
        seg_stats.push(seg.stats_handle());
        let seg_id = p.add(seg);

        if let Some(mos) = mosaic_id {
            let tee = p.add(streamcraft_elements::flow::Tee::new(2));
            p.link((depay, "src"), (tee, "sink")).expect("depay ! tee");
            p.link((tee, "src_0"), (seg_id, "sink")).expect("tee ! segsink");
            let dec = p.add(sc_h264::H264Dec::new());
            // Decoded frames: big slots (1080p I420 ≈ 3.1 MB).
            p.set_element_pool(dec, 4 * 1024 * 1024, 12);
            p.set_queue_capacity(dec, 16);
            p.link((tee, "src_1"), (dec, "sink")).expect("tee ! h264dec");
            p.link((dec, "src"), (mos, &format!("sink_{i}"))).expect("dec ! mosaic");
        } else {
            p.link((depay, "src"), (seg_id, "sink")).expect("depay ! segsink");
        }
        watch.push((format!("{}:udpsrc", cam.name), chain[i].0));
        watch.push((format!("{}:depay", cam.name), depay));
        watch.push((format!("{}:seg", cam.name), seg_id));
    }
    if let Some(mos) = mosaic_id {
        let sink = p.add_boxed(Box::new(sc_sdl3::Sdl3VideoSink::new().with_title("scraft-nvr")));
        p.link((mos, "src"), (sink, "sink")).expect("mosaic ! sdl3videosink");
        watch.push(("mosaic".into(), mos));
        watch.push(("wall".into(), sink));
    }

    // --- start the media, keep sessions alive, run ----------------------------
    for cam in &mut cams {
        cam.client.play().unwrap_or_else(|e| panic!("{}: PLAY: {e:?}", cam.name));
    }
    println!(
        "recording {n} cam(s) → {record_dir}/ ({segment_secs}s segments) — 'q'⏎ stops cleanly"
    );

    let stop = p.stop_handle();
    let mut keepalives = Vec::new();
    for cam in cams {
        let stop = stop.clone();
        let mut client = cam.client;
        let name = cam.name;
        let secs = client.session().map(|s| s.timeout_secs).unwrap_or(60).max(2) / 2;
        keepalives.push(std::thread::spawn(move || {
            let mut last = std::time::Instant::now();
            while !stop.is_stopped() {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if last.elapsed().as_secs() >= secs {
                    if client.keepalive().is_err() {
                        eprintln!("{name}: keepalive failed — camera gone?");
                        return;
                    }
                    last = std::time::Instant::now();
                }
            }
            let _ = client.teardown();
        }));
    }

    if stats {
        let tap = p.tap_handle();
        let seg_stats = seg_stats.clone();
        std::thread::spawn(move || {
            let mut prev = vec![(0u64, 0u64); watch.len()];
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let mut line = String::from("stats:");
                for (i, (name, id)) in watch.iter().enumerate() {
                    if let Some(c) = tap.snapshot(*id) {
                        let (pi, po) = prev[i];
                        line.push_str(&format!(
                            " {name} +{}/+{} d{} |",
                            c.buffers_in - pi,
                            c.buffers_out - po,
                            c.drops
                        ));
                        prev[i] = (c.buffers_in, c.buffers_out);
                    }
                }
                let (mut segs, mut mb) = (0u64, 0u64);
                for h in &seg_stats {
                    let s = h.lock().unwrap();
                    segs += s.segments_closed;
                    mb += s.bytes_written >> 20;
                }
                line.push_str(&format!(" segments={segs} written={mb}MiB"));
                eprintln!("{line}");
            }
        });
    }
    if let Some(secs) = max_secs {
        let stop = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.stop();
        });
    }
    {
        let stop = stop.clone();
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

    let result = p.run();
    while let Some(msg) = p.bus().try_recv() {
        if let streamcraft_core::bus::BusMessage::Warning { error, .. } = msg {
            eprintln!("warning: {error:?}");
        }
    }
    for ka in keepalives {
        let _ = ka.join();
    }
    let mut total = SegmentStats::default();
    for h in &seg_stats {
        let s = h.lock().unwrap();
        total.segments_closed += s.segments_closed;
        total.frames_written += s.frames_written;
        total.bytes_written += s.bytes_written;
    }
    println!(
        "done: {} segments, {} frames, {:.1} MiB",
        total.segments_closed,
        total.frames_written,
        total.bytes_written as f64 / (1024.0 * 1024.0)
    );
    if let Err(e) = result {
        eprintln!("pipeline error: {e:?}");
        std::process::exit(1);
    }
}
