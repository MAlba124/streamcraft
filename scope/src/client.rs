//! The introspection-protocol client (spec: Introspection protocol and scraft-scope).
//!
//! Owns the Unix-socket connection to a serving pipeline: handshake, a background
//! reader thread that decodes pushed and reply frames into a shared [`Model`], and a
//! request side the UI drives once per frame via [`Client::poll`]. Decoding reuses
//! `streamcraft_core::introspect::wire` — the same codecs the server's tests pin —
//! so client and server cannot drift.
//!
//! Threading: the reader thread blocks in `read_frame` (no read timeout, so framing
//! can never be torn by a partial read); requests go out on a `try_clone`d handle
//! from the UI thread. Both sides converge on `Arc<Mutex<Model>>`, which the UI
//! locks briefly once per frame.

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use streamcraft_core::introspect::wire::{self, kind, stream, submask};

/// How often [`Client::poll`] re-requests counters.
const COUNTER_POLL: Duration = Duration::from_millis(100);
/// Bounded feed of formatted event/log lines the UI drains each frame.
const FEED_CAP: usize = 4096;

/// One formatted line for the events-and-logs panel. `severity` is the log rank
/// (1 error .. 5 trace); bus messages map onto the same scale (events = 3).
pub struct FeedLine {
    pub text: String,
    pub severity: u8,
    /// True when this line came off the bus rather than the log stream.
    pub is_event: bool,
}

/// A topology element, strings resolved.
pub struct ElementView {
    pub id: u32,
    pub name: String,
    pub group: u32,
    pub flags: u8,
    pub npads: u16,
}

/// A negotiated edge, format resolved to display strings.
pub struct EdgeView {
    pub src: u32,
    pub src_pad: u32,
    pub sink: u32,
    pub sink_pad: u32,
    pub family: String,
    /// `name=value` pairs, pre-formatted for display.
    pub fields: Vec<String>,
}

/// The resolved topology tables (pads are folded into element/edge views for now).
pub struct TopoView {
    pub elements: Vec<ElementView>,
    pub edges: Vec<EdgeView>,
    /// Presentation duration when any negotiated format carried a `duration`
    /// field (ns) — e.g. the mkv demuxer announces `Info\Duration` on its pads.
    pub duration_ns: Option<u64>,
}

/// One counters reply: the server's `now_ns` plus a row per element. Rates are the
/// observer's arithmetic between two samples (spec: taps — cumulative counts, not rates).
pub struct CounterSample {
    pub now_ns: u64,
    pub rows: Vec<wire::CounterRow>,
}

/// Everything the UI renders, written by the reader thread, read once per frame.
#[derive(Default)]
pub struct Model {
    pub connected: bool,
    pub pid: u32,
    pub hello_flags: u32,
    pub topo: Option<TopoView>,
    /// Bumped on every new topology so the UI knows to re-layout.
    pub topo_gen: u64,
    pub cur: Option<CounterSample>,
    pub prev: Option<CounterSample>,
    pub feed: VecDeque<FeedLine>,
    /// Presentation duration, from the `DurationChanged` bus message, the sticky
    /// `Info` poll, or a negotiated `duration` format field (remux pipelines).
    pub duration_ns: Option<u64>,
    pub bus_dropped: u64,
    pub logs_dropped: u64,
    /// Set by the reader on a topology-mutation bus message; consumed by `poll`.
    want_topo: bool,
    /// A fatal connection error, for the UI to display.
    pub error: Option<String>,
}

impl Model {
    fn push_feed(&mut self, text: String, severity: u8, is_event: bool) {
        if self.feed.len() == FEED_CAP {
            self.feed.pop_front();
        }
        self.feed.push_back(FeedLine { text, severity, is_event });
    }
}

/// The connected client. Dropping it closes the socket; the reader thread exits on EOF.
pub struct Client {
    writer: UnixStream,
    pub model: Arc<Mutex<Model>>,
    seq: u16,
    last_counters: Instant,
    pub socket_path: String,
}

impl Client {
    /// Connect, handshake (Hello/ClientHello), subscribe to bus+logs, request the
    /// initial topology and counters, and start the reader thread.
    pub fn connect(path: &Path) -> std::io::Result<Client> {
        let mut stream = UnixStream::connect(path)?;

        // Server speaks first (spec: handshake).
        let f = wire::read_frame(&mut stream)?;
        if f.kind != kind::HELLO {
            return Err(bad("first frame was not Hello"));
        }
        let hello = wire::Hello::decode(&f.payload).ok_or_else(|| bad("bad Hello"))?;
        if hello.ver_major != wire::VER_MAJOR {
            return Err(bad("protocol major version mismatch"));
        }
        send(
            &mut stream,
            kind::CLIENT_HELLO,
            1,
            &wire::ClientHello { ver_major: wire::VER_MAJOR, ver_minor: wire::VER_MINOR }
                .encode(),
        )?;
        send(&mut stream, kind::SUBSCRIBE, 2, &wire::encode_sub(submask::BUS | submask::LOGS))?;
        send(&mut stream, kind::GET_TOPOLOGY, 3, &[])?;
        send(&mut stream, kind::GET_COUNTERS, 4, &[])?;

        let model = Arc::new(Mutex::new(Model {
            connected: true,
            pid: hello.pid,
            hello_flags: hello.flags,
            ..Model::default()
        }));

        let reader_stream = stream.try_clone()?;
        let reader_model = Arc::clone(&model);
        std::thread::Builder::new()
            .name("scope-reader".into())
            .spawn(move || reader_loop(reader_stream, reader_model))
            .map_err(|e| bad(&format!("spawn reader: {e}")))?;

        Ok(Client {
            writer: stream,
            model,
            seq: 5,
            last_counters: Instant::now(),
            socket_path: path.display().to_string(),
        })
    }

    fn next_seq(&mut self) -> u16 {
        // 0 is reserved for pushed frames; wrap past it.
        self.seq = self.seq.wrapping_add(1);
        if self.seq == 0 {
            self.seq = 1;
        }
        self.seq
    }

    /// Drive the request side; call once per UI frame. Re-requests counters on a
    /// fixed cadence and topology when a bus message said it changed.
    pub fn poll(&mut self) {
        if self.last_counters.elapsed() >= COUNTER_POLL {
            self.last_counters = Instant::now();
            let s = self.next_seq();
            let _ = send(&mut self.writer, kind::GET_COUNTERS, s, &[]);
            // Sticky facts a late attach missed on the bus; stop once known.
            let need_info = {
                let m = self.model.lock().unwrap_or_else(|e| e.into_inner());
                m.duration_ns.is_none()
            };
            if need_info {
                let s = self.next_seq();
                let _ = send(&mut self.writer, kind::GET_INFO, s, &[]);
            }
        }
        let want_topo = {
            let mut m = self.model.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut m.want_topo)
        };
        if want_topo {
            let s = self.next_seq();
            let _ = send(&mut self.writer, kind::GET_TOPOLOGY, s, &[]);
        }
    }

    /// Request a time seek (protocol v1.2). The server clamps to the known
    /// duration and maps time→byte through the app-installed `SeekIndex`; a
    /// pipeline without one answers Error(UNSUPPORTED), which lands in the feed.
    pub fn seek(&mut self, target_ns: u64) {
        let s = self.next_seq();
        let _ = send(&mut self.writer, kind::SEEK, s, &wire::encode_seek(target_ns));
    }

    /// Send Pause or Resume. The Ack is ignored; the UI reflects its own toggle.
    pub fn set_paused(&mut self, paused: bool) {
        let k = if paused { kind::PAUSE } else { kind::RESUME };
        let s = self.next_seq();
        let _ = send(&mut self.writer, k, s, &[]);
    }
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

fn send(w: &mut UnixStream, kind: u16, seq: u16, payload: &[u8]) -> std::io::Result<()> {
    w.write_all(&wire::encode_frame(kind, seq, payload))
}

// ---------------------------------------------------------------------------
// Reader thread: pushed + reply frames → Model
// ---------------------------------------------------------------------------

fn reader_loop(mut stream: UnixStream, model: Arc<Mutex<Model>>) {
    // The per-connection string table (spec: StrDef precedes first reference).
    let mut strings: HashMap<u32, String> = HashMap::new();
    loop {
        let frame = match wire::read_frame(&mut stream) {
            Ok(f) => f,
            Err(_) => break, // EOF or error: connection is over either way.
        };
        let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
        match frame.kind {
            kind::STR_DEF => {
                if let Some((id, s)) = wire::decode_str_def(&frame.payload) {
                    strings.insert(id, s);
                }
            }
            kind::TOPOLOGY => {
                if let Some(t) = decode_topology(&frame.payload, &strings) {
                    if t.duration_ns.is_some() {
                        m.duration_ns = t.duration_ns;
                    }
                    m.topo = Some(t);
                    m.topo_gen += 1;
                }
            }
            kind::INFO => {
                if let Some(Some(ns)) = wire::decode_info(&frame.payload) {
                    m.duration_ns = Some(ns);
                }
            }
            kind::COUNTERS => {
                if let Some(s) = decode_counters(&frame.payload) {
                    m.prev = m.cur.take();
                    m.cur = Some(s);
                }
            }
            kind::BUS_MSG => {
                if let Some(row) = wire::BusMsgRow::decode(&frame.payload) {
                    // DurationChanged: a=element, c=ns (wire.rs mapping table).
                    if row.kind == 13 {
                        m.duration_ns = Some(row.c);
                    }
                    let (text, severity) = format_bus(&row);
                    m.push_feed(text, severity, true);
                    // Topology-mutation kinds are the cue to re-GetTopology (wire.rs).
                    if (4..=7).contains(&row.kind) {
                        m.want_topo = true;
                    }
                }
            }
            kind::LOG_REC => {
                if let Some(row) = wire::LogRecRow::decode(&frame.payload) {
                    let text = format_log(&row, &strings);
                    m.push_feed(text, row.level.max(1), false);
                }
            }
            kind::DROPPED => {
                if let Some((sid, count)) = wire::decode_dropped(&frame.payload) {
                    let what = if sid == stream::BUS {
                        m.bus_dropped += count;
                        "bus messages"
                    } else {
                        m.logs_dropped += count;
                        "log records"
                    };
                    m.push_feed(format!("[scope] {count} {what} dropped (slow client)"), 2, true);
                }
            }
            kind::ERROR => {
                if let Some((code, msg)) = wire::decode_error(&frame.payload) {
                    if frame.seq == 0 {
                        // Fatal, server will close (spec: Error).
                        m.error = Some(format!("server error {code}: {msg}"));
                    } else {
                        m.push_feed(format!("[scope] request failed ({code}): {msg}"), 2, true);
                    }
                }
            }
            kind::BYE => {
                m.push_feed("[scope] server shut down".into(), 2, true);
                m.connected = false;
                return;
            }
            kind::ACK | kind::PONG => {}
            _ => {} // Unknown pushed kind: length prefix already skipped it. Forward-compatible.
        }
    }
    let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
    m.connected = false;
    if m.error.is_none() {
        m.error = Some("connection closed".into());
    }
}

fn resolve(strings: &HashMap<u32, String>, id: u32) -> String {
    match id {
        0 => String::new(),
        _ => strings.get(&id).cloned().unwrap_or_else(|| format!("#{id}")),
    }
}

/// Decode the Topology reply: three `TableHeader`-prefixed tables back-to-back
/// (elements, pads, edges), walking rows by the server's `row_size` stride.
fn decode_topology(payload: &[u8], strings: &HashMap<u32, String>) -> Option<TopoView> {
    let mut r = wire::Reader::new(payload);

    let mut elements = Vec::new();
    let th = wire::TableHeader::read(&mut r)?;
    for _ in 0..th.row_count {
        let start = r.pos();
        let row = wire::ElementRow::read(&mut r)?;
        r.skip((th.row_size as usize).saturating_sub(r.pos() - start))?;
        elements.push(ElementView {
            id: row.element,
            name: resolve(strings, row.name_str),
            group: row.group,
            flags: row.flags,
            npads: row.npads,
        });
    }

    // Pads: decoded for stride correctness; the simple UI doesn't render them yet.
    let th = wire::TableHeader::read(&mut r)?;
    for _ in 0..th.row_count {
        let start = r.pos();
        let _ = wire::PadRow::read(&mut r)?;
        r.skip((th.row_size as usize).saturating_sub(r.pos() - start))?;
    }

    let mut edges = Vec::new();
    let mut duration_ns: Option<u64> = None;
    let th = wire::TableHeader::read(&mut r)?;
    for _ in 0..th.row_count {
        let start = r.pos();
        let row = wire::EdgeRow::read(&mut r)?;
        r.skip((th.row_size as usize).saturating_sub(r.pos() - start))?;
        let mut fields = Vec::with_capacity(row.nfields as usize);
        for slot in row.fields.iter().take(row.nfields as usize) {
            let name = resolve(strings, slot.field_str);
            // A demuxer that knows the presentation duration announces it as a
            // format field (mkv: Info\Duration, ns) — surface it as the stream
            // duration rather than an edge-label detail.
            if name == "duration" && slot.tag as u32 == wire::WireValue::TAG_INT {
                duration_ns = Some(duration_ns.unwrap_or(0).max(slot.bits));
            }
            fields.push(format!("{name}={}", format_slot_value(slot, strings)));
        }
        edges.push(EdgeView {
            src: row.src,
            src_pad: row.src_pad,
            sink: row.sink,
            sink_pad: row.sink_pad,
            family: resolve(strings, row.family_str),
            fields,
        });
    }

    Some(TopoView { elements, edges, duration_ns })
}

fn format_slot_value(slot: &wire::FieldSlot, strings: &HashMap<u32, String>) -> String {
    match slot.tag as u32 {
        wire::WireValue::TAG_INT => format!("{}", slot.bits as i64),
        wire::WireValue::TAG_RAT => {
            let (n, d) = ((slot.bits >> 32) as u32 as i32, slot.bits as u32 as i32);
            format!("{n}/{d}")
        }
        wire::WireValue::TAG_ID => resolve(strings, slot.bits as u32),
        _ => "?".into(),
    }
}

fn decode_counters(payload: &[u8]) -> Option<CounterSample> {
    let mut r = wire::Reader::new(payload);
    let now_ns = r.get_u64()?;
    let th = wire::TableHeader::read(&mut r)?;
    let mut rows = Vec::with_capacity(th.row_count as usize);
    for _ in 0..th.row_count {
        let start = r.pos();
        rows.push(wire::CounterRow::read(&mut r)?);
        r.skip((th.row_size as usize).saturating_sub(r.pos() - start))?;
    }
    Some(CounterSample { now_ns, rows })
}

/// The BusMsg kind names, in the pinned ordinal order (wire.rs a/b/c/d table).
const BUS_KIND_NAMES: [&str; 14] = [
    "Error",
    "Warning",
    "Eos",
    "StateChanged",
    "PadAdded",
    "ElementAdded",
    "ElementRemoved",
    "LinkChanged",
    "Tags",
    "SubgraphJoined",
    "LatencyChanged",
    "Qos",
    "BranchSealed",
    "DurationChanged",
];

fn format_bus(row: &wire::BusMsgRow) -> (String, u8) {
    let name = BUS_KIND_NAMES.get(row.kind as usize).copied().unwrap_or("Unknown");
    let text = std::str::from_utf8(&row.msg[..row.msg_len as usize]).unwrap_or("");
    let line = match row.kind {
        0 | 1 => format!("[bus] {name} el={} {text}", row.a),
        2 => "[bus] Eos".into(),
        3 => format!("[bus] StateChanged {} -> {}", row.a, row.b),
        4 => format!("[bus] PadAdded el={} pad={}", row.a, row.b),
        5 => format!("[bus] ElementAdded el={} group={}", row.a, row.b),
        6 => format!("[bus] ElementRemoved el={}", row.a),
        7 => format!("[bus] LinkChanged link={}", row.a),
        9 => format!("[bus] SubgraphJoined group={} +{}ns latency", row.a, row.c),
        10 => format!("[bus] LatencyChanged {}ns -> {}ns", row.c, row.d),
        11 => format!("[bus] Qos sink={} lateness={}ns", row.a, row.c as i64),
        12 => format!("[bus] BranchSealed group={} {text}", row.a),
        13 => format!("[bus] DurationChanged el={} {}s", row.a, row.c / 1_000_000_000),
        _ => format!("[bus] {name} a={} b={}", row.a, row.b),
    };
    let severity = match row.kind {
        0 => 1, // Error
        1 => 2, // Warning
        _ => 3,
    };
    (line, severity)
}

fn format_log(row: &wire::LogRecRow, strings: &HashMap<u32, String>) -> String {
    use std::fmt::Write as _;
    let level = match row.level {
        1 => "E",
        2 => "W",
        3 => "I",
        4 => "D",
        _ => "T",
    };
    let mut line = format!(
        "{level} {} {}",
        resolve(strings, row.name_str),
        resolve(strings, row.event_str)
    );
    for f in row.fields.iter().take(row.nfields as usize) {
        let _ = write!(line, " {}=", resolve(strings, f.key_str));
        let _ = match f.tag {
            wire::LogField::TAG_INT => write!(line, "{}", f.bits as i64),
            wire::LogField::TAG_UINT => write!(line, "{}", f.bits),
            wire::LogField::TAG_BOOL => write!(line, "{}", f.bits != 0),
            wire::LogField::TAG_STR => write!(line, "{}", resolve(strings, f.bits as u32)),
            wire::LogField::TAG_FOURCC => {
                let b = (f.bits as u32).to_le_bytes();
                write!(line, "{}", String::from_utf8_lossy(&b))
            }
            wire::LogField::TAG_TIME => {
                if f.bits == wire::TS_NONE {
                    write!(line, "none")
                } else {
                    write!(line, "{}ms", f.bits / 1_000_000)
                }
            }
            _ => write!(line, "?"),
        };
    }
    line
}
