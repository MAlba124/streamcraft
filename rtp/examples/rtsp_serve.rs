//! `rtsp_serve` — a pure-streamcraft RTSP server streaming an MKV's H.264
//! track (spec: Milestone applications — network streaming, send side).
//!
//! The app is the controller (spec: no-bins): `sc_rtsp::server::RtspServer`
//! owns the *protocol* (DESCRIBE/SETUP/PLAY/TEARDOWN, sessions, timeouts) and
//! reports `ServerEvent`s; this app owns the *pipeline* — on `Play` it builds
//! `filesrc ! mkvdemux ! rtph264pay ! udpsink(client)` and runs it, clock-paced
//! by `udpsink`'s `wait_until` (the stream leaves at media rate, not disk
//! rate); on `Teardown` (client quit, or session timeout) it stops the
//! pipeline and waits for the next viewer.
//!
//! The other end can be pure sc too:
//! `play_file rtsp://127.0.0.1:8554/live` — the whole loop, both directions,
//! one codebase. ffplay/mpv consume it as well (interop check).
//!
//! ```text
//! cargo run --release -p sc-rtp --example rtsp_serve -- FILE.mkv [--port 8554]
//! ```

use std::io::Read;
use std::net::SocketAddr;
use std::sync::mpsc;

use sc_rtp::elements::{RtpH264Pay, UdpSink};
use sc_rtsp::server::{RtspServer, ServerConfig, ServerEvent};
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSrc;
use streamcraft_elements::testing::TestSink;

/// RFC 4648 §4 base64 (encode side — the SDP layer only ships a decoder).
fn encode_base64(data: &[u8]) -> String {
    const AL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        out.push(AL[(n >> 18) as usize & 63] as char);
        out.push(AL[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { AL[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { AL[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Split an Annex-B stream into NAL units (start codes stripped).
fn split_nals(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let mut e = starts.get(k + 1).map(|&n| n - 3).unwrap_or(stream.len());
        while e > s && stream[e - 1] == 0 {
            e -= 1; // trailing zeros belong to the next start code
        }
        if e > s {
            nals.push(&stream[s..e]);
        }
    }
    nals
}

/// The MKV head (everything before the first Cluster) — enough for Tracks.
fn header_prefix(path: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(path).expect("open input");
    let mut head = vec![0u8; 8 * 1024 * 1024];
    let n = f.read(&mut head).expect("read head");
    head.truncate(n);
    let cluster = head
        .windows(4)
        .position(|w| w == sc_mkv::ebml::id::CLUSTER)
        .expect("no Cluster in the first 8 MiB — not a (supported) MKV?");
    head.truncate(cluster);
    head
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .map(|i| {
            let v = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8554);
            args.drain(i..=i + 1);
            v
        })
        .unwrap_or(8554);
    let Some(path) = args.first().cloned() else {
        eprintln!("usage: rtsp_serve FILE.mkv [--port 8554]");
        std::process::exit(2);
    };

    // The H.264 track's out-of-band parameter sets, for the SDP's
    // sprop-parameter-sets (RFC 6184 §8.1): avcC → Annex-B head → SPS/PPS.
    let header = header_prefix(&path);
    let mut probe = sc_mkv::MatroskaReader::new();
    probe.push(&header).expect("parse header");
    let track = probe
        .tracks()
        .iter()
        .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")
        .cloned()
        .expect("no H.264 (V_MPEG4/ISO/AVC) track");
    let (annexb_head, _nal_len) =
        sc_mkv::nal_head_from_config(&track.codec_private, false).expect("avcC parses");
    let sprop: Vec<String> = split_nals(&annexb_head)
        .into_iter()
        .filter(|n| matches!(n[0] & 0x1F, 7 | 8)) // SPS / PPS
        .map(encode_base64)
        .collect();

    let sdp = format!(
        "v=0\r\n\
         o=- 0 0 IN IP4 127.0.0.1\r\n\
         s=streamcraft\r\n\
         t=0 0\r\n\
         m=video 0 RTP/AVP 96\r\n\
         a=rtpmap:96 H264/90000\r\n\
         a=fmtp:96 packetization-mode=1;sprop-parameter-sets={}\r\n\
         a=control:trackID=0\r\n",
        sprop.join(",")
    );

    let (tx, rx) = mpsc::channel::<ServerEvent>();
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let server =
        RtspServer::bind(addr, ServerConfig::new(sdp, "trackID=0"), tx).expect("bind rtsp");
    println!(
        "serving {path} at rtsp://127.0.0.1:{}/live — {}x{}, sprop NALs: {}",
        server.local_addr().port(),
        track.pixel_width,
        track.pixel_height,
        sprop.len()
    );

    // One viewer at a time (v1): Play builds and runs the pipeline from the
    // file start; Teardown stops it; then wait for the next viewer.
    let mut running: Option<(streamcraft_core::pipeline::StopHandle, std::thread::JoinHandle<()>)> =
        None;
    for event in rx {
        match event {
            ServerEvent::Play { session, client_rtp, .. } => {
                println!("PLAY (session {session}) → streaming to {client_rtp}");
                let path = path.clone();
                let header = header.clone();
                let mut p = Pipeline::new();
                let src = p.add(FileSrc::new(&path));
                let demux = p.add(sc_mkv::MkvDemux::new(header));
                p.link((src, "src"), (demux, "sink")).expect("src ! demux");
                let added = p.preroll().expect("preroll");
                let pay = p.add(RtpH264Pay::new(96));
                let mut linked = false;
                for ap in &added {
                    if !linked && p.link((ap.element, &ap.name), (pay, "sink")).is_ok() {
                        linked = true;
                        continue;
                    }
                    // Other tracks (audio, subs) drain into drop-sinks.
                    let (tsink, _stats) = TestSink::new();
                    let drop_id = p.add(tsink);
                    p.link((ap.element, &ap.name), (drop_id, "sink")).expect("drop link");
                }
                assert!(linked, "no pad linked to rtph264pay");
                let sink = p.add(UdpSink::new(client_rtp));
                p.link((pay, "src"), (sink, "sink")).expect("pay ! udpsink");
                let stop = p.stop_handle();
                let jh = std::thread::spawn(move || {
                    if let Err(e) = p.run() {
                        eprintln!("stream pipeline error: {e:?}");
                    }
                });
                running = Some((stop, jh));
            }
            ServerEvent::Pause { session } => {
                // v1: pause tears the stream down to a stop (a live restart on
                // the next PLAY would need server-side PLAY-after-PAUSE state).
                println!("PAUSE (session {session}) — stopping stream");
                if let Some((stop, jh)) = running.take() {
                    stop.stop();
                    let _ = jh.join();
                }
            }
            ServerEvent::Teardown { session } => {
                println!("TEARDOWN (session {session})");
                if let Some((stop, jh)) = running.take() {
                    stop.stop();
                    let _ = jh.join();
                }
            }
        }
    }
    drop(server);
}
