//! `rtsp_serve` — a pure-streamcraft RTSP server streaming an MKV's H.264
//! track (spec: Milestone applications — network streaming, send side).
//!
//! The app is the controller (spec: no-bins): `sc_rtsp::server::RtspServer`
//! owns the *protocol* (DESCRIBE/SETUP/PLAY/TEARDOWN, sessions, timeouts) and
//! reports `ServerEvent`s; this app owns the *pipeline* — on `Play` it builds
//! `camsrc ! rtph264pay ! udpsink(client)` and runs it, clock-paced by
//! `udpsink`'s `wait_until` (the stream leaves at media rate, not disk rate);
//! on `Teardown` (client quit, or session timeout) it stops the pipeline and
//! waits for the next viewer.
//!
//! `camsrc` is the app-side camera simulator: it walks the MKV with
//! `MatroskaReader` + the demuxer's NAL reframing and emits Annex-B access
//! units directly — and with `--loop` it rewinds at EOF while advancing pts by
//! the file's duration, so one PLAY session streams *forever* with continuous
//! timestamps and sequence numbers, exactly like a real IP camera. Parameter
//! sets are re-emitted in-band ahead of every IDR (camera convention; RFC 6184
//! §8.4 sanctions in-band repetition), so late joiners and recorders never
//! depend on the SDP alone.
//!
//! The other end can be pure sc too:
//! `play_file rtsp://127.0.0.1:8554/live` — the whole loop, both directions,
//! one codebase. ffplay/mpv consume it as well (interop check).
//!
//! ```text
//! cargo run --release -p sc-rtp --example rtsp_serve -- FILE.mkv [--port 8554] [--loop]
//! ```

use std::io::Read;
use std::net::SocketAddr;
use std::sync::mpsc;

use sc_mkv::codec::{nal_head_from_config, Reframer};
use sc_mkv::MatroskaReader;
use sc_rtp::elements::{RtpH264Pay, UdpSink};
use sc_rtsp::server::{RtspServer, ServerConfig, ServerEvent};
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::io::IoResult;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

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
// App-side startup probe on the main thread, not element IO (clippy.toml).
#[allow(clippy::disallowed_methods)]
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

// --- camsrc: the looping camera-simulator source -------------------------------

static CAM_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static CAM_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &CAM_OFFERS,
    dynamic: false,
    validate: None,
}];
static CAM_DESC: ElementDesc = ElementDesc {
    name: "camsrc",
    pads: &CAM_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};
const CAM_SRC: PadId = PadId(0);

/// MKV file → Annex-B H.264 access units, optionally looping forever with pts
/// continuing across iterations (`pts = file pts + loops × duration`). One AU
/// per buffer — the payloader's contract. Reads ride the reactor (spec: IO —
/// elements never block on data): one positioned read outstanding, an offset
/// cursor that wraps to 0 at EOF, the file registered once at `start`.
struct CamSrc {
    path: String,
    looping: bool,
    /// The H.264 track's Annex-B parameter-set head (from CodecPrivate),
    /// re-emitted ahead of every IDR (in-band repetition, RFC 6184 §8.4).
    head: Vec<u8>,
    track: u64,
    reframer: Reframer,
    reader: MatroskaReader,
    file_len: u64,
    read_offset: u64,
    read_in_flight: bool,
    /// Added to every emitted pts: `completed loops × file duration`.
    loop_offset_ns: u64,
    /// File-local pts of the last emitted frame + its predecessor's delta —
    /// the duration fallback when Info omits one.
    last_pts_ns: u64,
    prev_pts_ns: u64,
    duration_ns: Option<u64>,
    /// Pool-dry carry: (annexb AU, absolute pts ns).
    pending: Option<(Vec<u8>, u64)>,
    loops: u64,
}

impl CamSrc {
    /// Probes the header eagerly (panics on a file without a supported H.264
    /// track — this is the app's own startup path, not stream input).
    fn new(path: &str, looping: bool) -> CamSrc {
        let header = header_prefix(path);
        let mut probe = MatroskaReader::new();
        probe.push(&header).expect("parse header");
        let track = probe
            .tracks()
            .iter()
            .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")
            .cloned()
            .expect("no H.264 (V_MPEG4/ISO/AVC) track");
        let (head, length_size) =
            nal_head_from_config(&track.codec_private, false).expect("avcC parses");
        CamSrc {
            path: path.to_string(),
            looping,
            head,
            track: track.track_number,
            reframer: Reframer::Nal { length_size },
            reader: MatroskaReader::new(),
            file_len: 0,
            read_offset: 0,
            read_in_flight: false,
            loop_offset_ns: 0,
            last_pts_ns: 0,
            prev_pts_ns: 0,
            duration_ns: probe.duration_ns(),
            pending: None,
            loops: 0,
        }
    }

    /// The pts advance for one file iteration: the declared duration, else the
    /// observed span plus one mean frame delta.
    fn loop_span(&self) -> u64 {
        self.duration_ns
            .unwrap_or(self.last_pts_ns + self.last_pts_ns.saturating_sub(self.prev_pts_ns))
            .max(1)
    }

    /// Emit the carried AU if a slot frees; `false` = still stalled.
    fn flush_pending(&mut self, ctx: &mut Ctx) -> bool {
        let Some((au, pts)) = self.pending.take() else { return true };
        let Some(mut buf) = ctx.try_alloc(CAM_SRC) else {
            self.pending = Some((au, pts));
            return false;
        };
        assert!(
            au.len() <= buf.memory.capacity(),
            "camsrc: AU ({} B) exceeds the pool slot ({} B) — raise set_pool",
            au.len(),
            buf.memory.capacity()
        );
        buf.memory.as_mut_full()[..au.len()].copy_from_slice(&au);
        buf.memory.set_len(au.len());
        buf.pts = Timestamp::from_nanos(pts);
        ctx.out(CAM_SRC).push(buf);
        true
    }
}

impl Element for CamSrc {
    fn desc(&self) -> &'static ElementDesc {
        &CAM_DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let file = std::fs::File::open(&self.path)
            .map_err(|e| Error::Resource(format!("open {}: {e}", self.path)))?;
        self.file_len = file
            .metadata()
            .map_err(|e| Error::Resource(format!("stat {}: {e}", self.path)))?
            .len();
        ctx.io().register(file);
        self.reader = MatroskaReader::new();
        self.read_offset = 0;
        self.read_in_flight = false;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Completed reads feed the parser; the buffer recycles on drop.
        while let Some(c) = ctx.io().next_completion() {
            self.read_in_flight = false;
            match c.result {
                // EOF race (file shrank under us): clamp so the wrap fires.
                IoResult::Ok(0) => self.read_offset = self.file_len,
                IoResult::Ok(_) => {
                    self.read_offset += c.buf.memory.len() as u64;
                    self.reader
                        .push(c.buf.memory.data())
                        .map_err(|e| Error::Resource(format!("camsrc: parse: {e:?}")))?;
                }
                IoResult::Cancelled => {}
                IoResult::Err(k) => {
                    return Err(Error::Resource(format!("camsrc: read {}: {k:?}", self.path)))
                }
            }
        }
        // Emit parsed frames while output slots allow.
        loop {
            if !self.flush_pending(ctx) {
                return Ok(Flow::Ok); // pool dry — backpressure
            }
            let Some(frame) = self.reader.next_frame() else { break };
            if frame.track_number != self.track {
                continue;
            }
            let body = self
                .reframer
                .reframe_block(&frame.data)
                .map_err(|e| Error::Resource(format!("camsrc: reframe: {e:?}")))?;
            let mut au = Vec::with_capacity(self.head.len() + body.len());
            if frame.keyframe {
                au.extend_from_slice(&self.head);
            }
            au.extend_from_slice(&body);
            self.prev_pts_ns = self.last_pts_ns;
            self.last_pts_ns = frame.pts_ns;
            self.pending = Some((au, frame.pts_ns + self.loop_offset_ns));
        }
        // Parser starved: wrap (or end) at EOF, then keep one read in flight.
        if self.read_offset >= self.file_len && !self.read_in_flight {
            if !self.looping {
                return Ok(Flow::Eos); // pending/frames drained above
            }
            self.loop_offset_ns += self.loop_span();
            self.loops += 1;
            self.read_offset = 0;
            self.reader = MatroskaReader::new();
        }
        if !self.read_in_flight {
            if let Some(buf) = ctx.try_alloc(CAM_SRC) {
                let handle = streamcraft_core::io::FileHandle(ctx.element().0);
                ctx.io().submit_read(handle, self.read_offset, buf, 0);
                self.read_in_flight = true;
            }
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// --- the RTSP control loop -----------------------------------------------------

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
    let looping = if let Some(i) = args.iter().position(|a| a == "--loop") {
        args.remove(i);
        true
    } else {
        false
    };
    let Some(path) = args.first().cloned() else {
        eprintln!("usage: rtsp_serve FILE.mkv [--port 8554] [--loop]");
        std::process::exit(2);
    };

    // The H.264 track's out-of-band parameter sets, for the SDP's
    // sprop-parameter-sets (RFC 6184 §8.1): avcC → Annex-B head → SPS/PPS.
    let header = header_prefix(&path);
    let mut probe = MatroskaReader::new();
    probe.push(&header).expect("parse header");
    let track = probe
        .tracks()
        .iter()
        .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")
        .cloned()
        .expect("no H.264 (V_MPEG4/ISO/AVC) track");
    let (annexb_head, _nal_len) =
        nal_head_from_config(&track.codec_private, false).expect("avcC parses");
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
        "serving {path} at rtsp://127.0.0.1:{}/live — {}x{}, sprop NALs: {}{}",
        server.local_addr().port(),
        track.pixel_width,
        track.pixel_height,
        sprop.len(),
        if looping { ", looping" } else { "" }
    );

    // One viewer at a time (v1): Play builds and runs the pipeline from the
    // file start; Teardown stops it; then wait for the next viewer.
    let mut running: Option<(streamcraft_core::pipeline::StopHandle, std::thread::JoinHandle<()>)> =
        None;
    for event in rx {
        match event {
            ServerEvent::Play { session, client_rtp, .. } => {
                println!("PLAY (session {session}) → streaming to {client_rtp}");
                let mut p = Pipeline::new();
                // One AU per slot; 4 MiB dwarfs any AU this simulator serves.
                p.set_pool(4 * 1024 * 1024, 24);
                let src = p.add(CamSrc::new(&path, looping));
                let pay = p.add(RtpH264Pay::new(96));
                p.link((src, "src"), (pay, "sink")).expect("camsrc ! pay");
                // The payloader gets its OWN small-slot pool (the play_file
                // set_element_pool lesson, sender-side): RTP packets are ≤1.4 KB
                // and the paced udpsink HOLDS them across its clock waits — from
                // the shared pool each held packet would pin a whole 4 MiB AU
                // slot, the greedy source drinks the rest, and the graph
                // deadlocks pool-dry (measured: camsrc +25 AUs, pay stuck at 4
                // packets out, every group parked idle).
                p.set_element_pool(pay, 2048, 256);
                p.set_queue_capacity(pay, 64);
                // 100 µs/packet ≈ 112 Mbit/s shaped — keyframes stop bursting past
                // default receiver buffers (see UdpSink::with_packet_gap).
                let sink = p.add(UdpSink::new(client_rtp));
                p.link((pay, "src"), (sink, "sink")).expect("pay ! udpsink");
                // SC_SERVE_STATS=1: per-element counter deltas every 3 s — the
                // sender-side stall diagnostic (which element stopped moving).
                if std::env::var_os("SC_SERVE_STATS").is_some() {
                    let tap = p.tap_handle();
                    let watched = [("camsrc", src), ("pay", pay), ("udpsink", sink)];
                    std::thread::spawn(move || {
                        let mut prev = [(0u64, 0u64); 3];
                        loop {
                            std::thread::sleep(std::time::Duration::from_secs(3));
                            let mut line = String::from("serve stats:");
                            for (i, (name, id)) in watched.iter().enumerate() {
                                if let Some(c) = tap.snapshot(*id) {
                                    let (pi, po) = prev[i];
                                    line.push_str(&format!(
                                        " {name} {}→{} (+{}/+{}) qhw={} |",
                                        c.buffers_in,
                                        c.buffers_out,
                                        c.buffers_in - pi,
                                        c.buffers_out - po,
                                        c.queue_high_water,
                                    ));
                                    prev[i] = (c.buffers_in, c.buffers_out);
                                }
                            }
                            eprintln!("{line}");
                        }
                    });
                }
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
