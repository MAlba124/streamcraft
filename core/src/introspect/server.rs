//! The introspection server: an accept thread plus one blocking thread per client
//! (spec: server architecture). Clients number ~1-2 and std has no readiness API, so a
//! parked blocking read costs zero CPU — a thread per client is the simplest correct
//! shape.
//!
//! Per-client loop: write `Hello`; read `ClientHello` (1 s timeout, version-gated);
//! then set a 5 ms read timeout and a 1 s write timeout and loop
//! `{ read a frame (≤5 ms block) → dispatch a reply (StrDefs first) → drain the bus tap
//! ring → drain the log tap ring → flush }` until stop or an IO error. A slow client's
//! rings drop-and-count (via the taps), and repeated write timeouts detach it.
//!
//! Everything a reply needs comes from [`IntrospectShared`]: a published
//! [`TopologySnapshot`], live [`Handles`], the [`LogTapRegistry`], and the bus tap
//! port — the server never touches the pipeline's interners or streaming threads.

use std::io::{BufWriter, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::bus::BusTapPort;
use crate::counters::TapHandle;
use crate::format::Value;
use crate::id::{ElementId, ValueId};
use crate::log::{Level, LevelFilter};
use crate::pipeline::PauseHandle;
use crate::props::PropHandle;

use super::snapshot::{SnapValue, TopologySnapshot};
use super::strtab::StrTab;
use super::tap::{BusTapRow, LogTapRegistry, TapId};
use super::wire::{self, errcode, kind, Writer};

/// The mutable control handles the server flips on request (spec: Shared state —
/// `Handles`). Filled by `serve_introspection`, refreshed by `run()` after preroll.
pub struct Handles {
    pub tap: TapHandle,
    pub props: PropHandle,
    pub pause: PauseHandle,
    pub seek: crate::pipeline::SeekHandle,
    /// The app-installed time→byte index; `None` = time seeking unavailable.
    pub seek_index: Option<std::sync::Arc<crate::pipeline::SeekIndex>>,
    pub tracing: Arc<AtomicBool>,
    /// Per-element log filters (one per element, index = ElementId), flipped live by
    /// `SetLogLevel`. Empty when logging channels were not wired this run.
    pub filters: Vec<Arc<LevelFilter>>,
    /// The global log filter (the shared gate), if channels were wired.
    pub global_filter: Option<Arc<LevelFilter>>,
    /// Whether log channels are wired this run (Hello flags bit1).
    pub log_channels: bool,
}

/// Server-shared state (spec: `Arc<IntrospectShared>`). Every field is either atomic or
/// behind a brief mutex; nothing here is touched on a streaming path.
pub struct IntrospectShared {
    pub stop: AtomicBool,
    pub topology: Mutex<Arc<TopologySnapshot>>,
    pub handles: Mutex<Handles>,
    pub log_taps: Arc<LogTapRegistry>,
    /// The bus tap port (server-internal; `BusTapPort` is `pub(crate)`).
    pub(crate) bus_port: BusTapPort,
    /// Sticky facts distilled from an always-attached internal bus tap, so a client
    /// attaching mid-run can poll (`GetInfo`) what it missed on the bus — a demuxer
    /// posts `DurationChanged` once at preroll. Server threads drain opportunistically.
    pub sticky: Mutex<Sticky>,
}

/// See [`IntrospectShared::sticky`]. `rx` is the internal tap's consumer; `None` in
/// tests that build shared state without a bus.
pub struct Sticky {
    pub(crate) rx: Option<crate::ring::Consumer<BusTapRow>>,
    pub duration_ns: Option<u64>,
}

impl IntrospectShared {
    /// Republish a fresh topology snapshot (spec: `run()` republishes after preroll so
    /// dynamic pads are visible).
    pub fn publish_topology(&self, snap: TopologySnapshot) {
        *self.topology.lock().unwrap_or_else(|e| e.into_inner()) = Arc::new(snap);
    }

    /// Replace the control handles (spec: `run()` republishes handles too).
    pub fn set_handles(&self, handles: Handles) {
        *self.handles.lock().unwrap_or_else(|e| e.into_inner()) = handles;
    }

    /// Drain the internal sticky bus tap into the cached facts. Cheap (the messages
    /// are rare) and called opportunistically from server threads, never streaming ones.
    pub fn drain_sticky(&self) {
        let mut s = self.sticky.lock().unwrap_or_else(|e| e.into_inner());
        let Some(rx) = &s.rx else { return };
        let mut duration = None;
        while let Some(row) = rx.try_pop() {
            // Ordinal 13 = DurationChanged (wire.rs mapping table): a=element, c=ns.
            if row.kind == 13 {
                duration = Some(row.c);
            }
        }
        if duration.is_some() {
            s.duration_ns = duration;
        }
    }
}

/// A running introspection server: the listener thread, the shared state, and the
/// socket path. Dropping it shuts everything down and unlinks the socket.
pub struct IntrospectServer {
    shared: Arc<IntrospectShared>,
    path: PathBuf,
    listener: Option<JoinHandle<()>>,
}

impl IntrospectServer {
    /// Bind the socket and spawn the accept thread. `path` is removed first if stale.
    pub fn spawn(
        path: impl AsRef<Path>,
        shared: Arc<IntrospectShared>,
    ) -> std::io::Result<IntrospectServer> {
        let path = path.as_ref().to_path_buf();
        // A stale socket file from a crashed run would make `bind` fail with EADDRINUSE.
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        // A short accept timeout lets the loop poll the stop flag; the `connect` trick in
        // Drop is the primary wakeup, the timeout is a backstop.
        listener.set_nonblocking(false)?;
        let shared_l = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("sc-introspect-accept".into())
            .spawn(move || accept_loop(listener, shared_l))
            .expect("spawn introspect accept thread");
        Ok(IntrospectServer { shared, path, listener: Some(handle) })
    }

    /// The shared state, for the pipeline to republish snapshots/handles into.
    pub fn shared(&self) -> &Arc<IntrospectShared> {
        &self.shared
    }

    /// The socket path this server is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for IntrospectServer {
    fn drop(&mut self) {
        // Signal stop, then wake the parked `accept()` by connecting to our own socket
        // (a std-only way to unblock a blocking accept — the accept thread checks the
        // stop flag right after accepting and exits, joining its clients).
        self.shared.stop.store(true, Ordering::Release);
        let _ = UnixStream::connect(&self.path);
        if let Some(h) = self.listener.take() {
            let _ = h.join();
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The accept loop: block on `accept`, spawn a client thread per connection, and exit
/// (joining all clients) once the stop flag is set.
fn accept_loop(listener: UnixListener, shared: Arc<IntrospectShared>) {
    let mut clients: Vec<JoinHandle<()>> = Vec::new();
    loop {
        match listener.accept() {
            Ok((stream, _addr)) => {
                if shared.stop.load(Ordering::Acquire) {
                    break; // the Drop wakeup connection — exit
                }
                let shared_c = Arc::clone(&shared);
                let h = std::thread::Builder::new()
                    .name("sc-introspect-client".into())
                    .spawn(move || {
                        let _ = client_loop(stream, shared_c);
                    })
                    .expect("spawn introspect client thread");
                clients.push(h);
                // Reap finished clients so the vec doesn't grow without bound.
                clients.retain(|h| !h.is_finished());
            }
            Err(_) => {
                if shared.stop.load(Ordering::Acquire) {
                    break;
                }
            }
        }
    }
    // Best-effort: clients notice the stop flag at their next 5 ms read boundary and
    // send a Bye, then exit. Join them for a clean shutdown.
    for h in clients {
        let _ = h.join();
    }
}

/// How many consecutive write timeouts before we give up on a wedged client.
const MAX_WRITE_TIMEOUTS: u32 = 5;
/// Per-client bus/log tap ring capacity (rows). Small — a slow client drops, it does not
/// grow memory. Power of two (the ring rounds up anyway).
const TAP_RING_CAP: usize = 1024;

/// One client's lifetime: handshake, then the request/push/flush loop.
fn client_loop(stream: UnixStream, shared: Arc<IntrospectShared>) -> std::io::Result<()> {
    let mut conn = Conn::new(stream, &shared)?;
    // Handshake: write Hello, read ClientHello (version-gated) within 1 s.
    conn.write_hello(&shared)?;
    conn.reader.set_read_timeout(Some(Duration::from_secs(1)))?;
    match wire::read_frame(&mut conn.reader) {
        Ok(f) if f.kind == kind::CLIENT_HELLO => {
            let ch = wire::ClientHello::decode(&f.payload);
            let major = ch.map(|c| c.ver_major).unwrap_or(0);
            if major != wire::VER_MAJOR {
                conn.send(kind::ERROR, f.seq, &wire::encode_error(
                    errcode::VERSION_MISMATCH,
                    "unsupported protocol major version",
                ))?;
                conn.flush()?;
                return Ok(()); // disconnect after the Error frame
            }
        }
        _ => {
            // A missing/garbled ClientHello: refuse and close.
            conn.send(kind::ERROR, 0, &wire::encode_error(
                errcode::BAD_FRAME,
                "expected ClientHello as the first frame",
            ))?;
            conn.flush()?;
            return Ok(());
        }
    }

    // Streaming phase: short read timeout so pushes stay responsive; long write timeout.
    conn.reader.set_read_timeout(Some(Duration::from_millis(5)))?;
    conn.writer.get_ref().set_write_timeout(Some(Duration::from_secs(1)))?;

    loop {
        if shared.stop.load(Ordering::Acquire) {
            // Best-effort Bye at shutdown.
            let _ = conn.send(kind::BYE, 0, &[]);
            let _ = conn.flush();
            break;
        }

        // 1) One request (blocks up to 5 ms). WouldBlock/TimedOut just means "no request
        //    this tick" — fall through to draining pushes.
        match wire::read_frame(&mut conn.reader) {
            Ok(frame) => {
                if let Err(e) = conn.dispatch(&shared, frame) {
                    // A real IO error tears the connection down.
                    if is_fatal(&e) {
                        break;
                    }
                }
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut
                {
                    // No request; continue to pushes.
                } else {
                    break; // EOF or a real error → client gone
                }
            }
        }

        // 2) Drain any subscribed streams (bus, logs) → push frames.
        if let Err(e) = conn.drain_pushes(&shared) {
            if is_fatal(&e) {
                break;
            }
        }

        // 3) Flush what we buffered this tick.
        if let Err(e) = conn.flush() {
            if is_fatal(&e) {
                break;
            }
        }

        if conn.write_timeouts >= MAX_WRITE_TIMEOUTS {
            break; // wedged client — give up
        }
    }

    conn.teardown(&shared);
    Ok(())
}

/// Whether an IO error should end the connection (a write timeout is recoverable up to
/// the count cap; a real error is not).
fn is_fatal(e: &std::io::Error) -> bool {
    !matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// Per-connection state: the socket halves, the string table, and subscription state.
struct Conn {
    reader: UnixStream,
    writer: BufWriter<UnixStream>,
    strtab: StrTab,
    write_timeouts: u32,

    // Subscriptions (RAII-detached in `teardown`).
    bus_sub: Option<(TapId, crate::ring::Consumer<BusTapRow>, u64)>, // (id, ring, last_dropped)
    log_sub: Option<(u64, crate::ring::Consumer<crate::log::LogRecord>, u64)>,

    /// Connection string id → pipeline `ValueId`, so a `SetProp` with tag=Id maps back
    /// to the categorical value the string named (spec: PropRow — record the mapping).
    id_to_valueid: std::collections::HashMap<u32, u32>,
}

impl Conn {
    fn new(stream: UnixStream, _shared: &Arc<IntrospectShared>) -> std::io::Result<Conn> {
        let reader = stream.try_clone()?;
        let writer = BufWriter::new(stream);
        Ok(Conn {
            reader,
            writer,
            strtab: StrTab::new(),
            write_timeouts: 0,
            bus_sub: None,
            log_sub: None,
            id_to_valueid: std::collections::HashMap::new(),
        })
    }

    /// Write a raw frame into the buffer, tracking write timeouts.
    fn send(&mut self, kind: u16, seq: u16, payload: &[u8]) -> std::io::Result<()> {
        let bytes = wire::encode_frame(kind, seq, payload);
        match self.writer.write_all(&bytes) {
            Ok(()) => {
                self.write_timeouts = 0;
                Ok(())
            }
            Err(e) => {
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) {
                    self.write_timeouts += 1;
                }
                Err(e)
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.writer.flush() {
            Ok(()) => {
                self.write_timeouts = 0;
                Ok(())
            }
            Err(e) => {
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) {
                    self.write_timeouts += 1;
                }
                Err(e)
            }
        }
    }

    /// Intern a `&'static str`, emitting a StrDef first if new. Returns the id.
    fn intern_static(&mut self, s: &'static str) -> std::io::Result<u32> {
        let (id, is_new) = self.strtab.intern_static(s);
        if is_new {
            self.send(kind::STR_DEF, 0, &wire::encode_str_def(id, s))?;
        }
        Ok(id)
    }

    /// Intern an owned string by content, emitting a StrDef first if new.
    fn intern_owned(&mut self, s: &str) -> std::io::Result<u32> {
        let (id, is_new) = self.strtab.intern_owned(s);
        if is_new {
            self.send(kind::STR_DEF, 0, &wire::encode_str_def(id, s))?;
        }
        Ok(id)
    }

    fn write_hello(&mut self, shared: &Arc<IntrospectShared>) -> std::io::Result<()> {
        let (nelements, flags) = {
            let topo = shared.topology.lock().unwrap_or_else(|e| e.into_inner());
            let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
            let mut flags = 0u32;
            if handles.tracing.load(Ordering::Acquire) {
                flags |= wire::helloflags::TRACING_ON;
            }
            if handles.log_channels {
                flags |= wire::helloflags::LOG_CHANNELS;
            }
            (topo.elements.len() as u32, flags)
        };
        let pid = std::process::id();
        let hello = wire::Hello {
            ver_major: wire::VER_MAJOR,
            ver_minor: wire::VER_MINOR,
            flags,
            pid,
            nelements,
        };
        self.send(kind::HELLO, 0, &hello.encode())?;
        self.flush()
    }

    /// Dispatch one request frame to its reply (spec: Frame catalog v1).
    fn dispatch(
        &mut self,
        shared: &Arc<IntrospectShared>,
        frame: wire::Frame,
    ) -> std::io::Result<()> {
        let seq = frame.seq;
        match frame.kind {
            kind::PING => self.send(kind::PONG, seq, &[]),
            kind::CLIENT_HELLO => Ok(()), // late duplicate; ignore
            kind::GET_TOPOLOGY => self.reply_topology(shared, seq),
            kind::GET_DOT => self.reply_dot(shared, seq),
            kind::GET_COUNTERS => self.reply_counters(shared, seq),
            kind::SEEK => {
                let Some(target_ns) = wire::decode_seek(&frame.payload) else {
                    return self.send(
                        kind::ERROR,
                        seq,
                        &wire::encode_error(errcode::BAD_FRAME, "bad Seek payload"),
                    );
                };
                // Clamp to the known duration (sticky) and map time→byte via the
                // app-installed index; without one, time seeking is unsupported.
                shared.drain_sticky();
                let duration = shared
                    .sticky
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .duration_ns
                    .map(crate::time::Timestamp)
                    .unwrap_or(crate::time::Timestamp::NONE);
                let target = crate::time::Timestamp(match duration.nanos() {
                    Some(d) => target_ns.min(d),
                    None => target_ns,
                });
                let g = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
                let resolved = g
                    .seek_index
                    .as_ref()
                    .and_then(|idx| idx.resolve(target, duration));
                match resolved {
                    // Seek to the RESOLVED time (the cue point the byte lands on),
                    // not the request — see SeekIndex::resolve for the frozen-video
                    // failure mode a requested-time rebase causes.
                    Some((b, landed)) => {
                        g.seek.seek(b, landed);
                        drop(g);
                        self.send(kind::ACK, seq, &[])
                    }
                    None => {
                        drop(g);
                        self.send(
                            kind::ERROR,
                            seq,
                            &wire::encode_error(
                                errcode::UNSUPPORTED,
                                "no seek index installed (Pipeline::set_seek_index)",
                            ),
                        )
                    }
                }
            }
            kind::GET_INFO => {
                shared.drain_sticky();
                let d = shared.sticky.lock().unwrap_or_else(|e| e.into_inner()).duration_ns;
                self.send(kind::INFO, seq, &wire::encode_info(d))
            }
            kind::GET_LATENCY => self.reply_latency(shared, seq, &frame.payload),
            kind::GET_LATENCY_REPORT => self.reply_latency_report(shared, seq),
            kind::GET_PROPS => self.reply_props(shared, seq, &frame.payload),
            kind::SET_PROP => self.handle_set_prop(shared, seq, &frame.payload),
            kind::PAUSE => {
                shared.handles.lock().unwrap_or_else(|e| e.into_inner()).pause.pause();
                self.send(kind::ACK, seq, &[])
            }
            kind::RESUME => {
                shared.handles.lock().unwrap_or_else(|e| e.into_inner()).pause.resume();
                self.send(kind::ACK, seq, &[])
            }
            kind::STEP => self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::UNSUPPORTED, "step() is reserved (not in v1)"),
            ),
            kind::SET_LOG_LEVEL => self.handle_set_log_level(shared, seq, &frame.payload),
            kind::SET_TRACING => self.handle_set_tracing(shared, seq, &frame.payload),
            kind::SUBSCRIBE => self.handle_subscribe(shared, seq, &frame.payload),
            kind::UNSUBSCRIBE => self.handle_unsubscribe(shared, seq, &frame.payload),
            _ => self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::UNKNOWN_KIND, "unknown frame kind"),
            ),
        }
    }

    // --- Snapshot replies --------------------------------------------------

    fn reply_topology(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
    ) -> std::io::Result<()> {
        let topo = {
            let g = shared.topology.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(&g)
        };

        // Elements table.
        let mut ew = Writer::new();
        wire::TableHeader { row_size: wire::ROW_SIZE_ELEMENT, row_count: topo.elements.len() as u16 }
            .write(&mut ew);
        for e in &topo.elements {
            let name_str = self.intern_owned(&e.name)?;
            let mut flags = 0u8;
            if e.is_source {
                flags |= wire::ElementRow::FLAG_SOURCE;
            }
            if e.is_sink {
                flags |= wire::ElementRow::FLAG_SINK;
            }
            if e.is_live {
                flags |= wire::ElementRow::FLAG_LIVE;
            }
            wire::ElementRow {
                element: e.id,
                name_str,
                group: e.group,
                sched: e.sched,
                flags,
                npads: e.pads.len() as u16,
                latency_min_ns: e.latency_min_ns,
                latency_max_ns: e.latency_max_ns,
                jitter_ns: e.jitter_ns,
            }
            .write(&mut ew);
        }

        // Pads table.
        let pad_count: usize = topo.elements.iter().map(|e| e.pads.len()).sum();
        let mut pw = Writer::new();
        wire::TableHeader { row_size: wire::ROW_SIZE_PAD, row_count: pad_count as u16 }
            .write(&mut pw);
        for e in &topo.elements {
            for p in &e.pads {
                let name_str = self.intern_owned(&p.name)?;
                let mut flags = 0u8;
                if p.dynamic {
                    flags |= wire::PadRow::FLAG_DYNAMIC;
                }
                if p.linked {
                    flags |= wire::PadRow::FLAG_LINKED;
                }
                wire::PadRow {
                    element: e.id,
                    pad: p.pad,
                    name_str,
                    direction: p.direction,
                    flags,
                }
                .write(&mut pw);
            }
        }

        // Edges table.
        let mut dw = Writer::new();
        wire::TableHeader { row_size: wire::ROW_SIZE_EDGE, row_count: topo.edges.len() as u16 }
            .write(&mut dw);
        for edge in &topo.edges {
            let family_str = self.intern_owned(&edge.family_name)?;
            let mut fields = [wire::FieldSlot::ZERO; wire::EDGE_MAX_FIELDS];
            let n = edge.fields.len().min(wire::EDGE_MAX_FIELDS);
            for (slot, f) in fields.iter_mut().zip(&edge.fields[..n]) {
                let field_str = self.intern_owned(&f.field_name)?;
                let bits = if f.tag as u32 == wire::WireValue::TAG_ID {
                    // Categorical id → connection string id.
                    self.intern_owned(&f.id_name)? as u64
                } else {
                    f.bits
                };
                *slot = wire::FieldSlot { field_str, tag: f.tag, bits };
            }
            wire::EdgeRow {
                src: edge.src,
                src_pad: edge.src_pad,
                sink: edge.sink,
                sink_pad: edge.sink_pad,
                family_str,
                nfields: n as u8,
                fields,
            }
            .write(&mut dw);
        }

        // Three tables back-to-back.
        let mut payload = ew.into_vec();
        payload.extend_from_slice(&pw.into_vec());
        payload.extend_from_slice(&dw.into_vec());
        self.send(kind::TOPOLOGY, seq, &payload)
    }

    fn reply_dot(&mut self, shared: &Arc<IntrospectShared>, seq: u16) -> std::io::Result<()> {
        let dot = {
            let g = shared.topology.lock().unwrap_or_else(|e| e.into_inner());
            g.dot.clone()
        };
        self.send(kind::DOT, seq, dot.as_bytes())
    }

    fn reply_counters(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
    ) -> std::io::Result<()> {
        let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
        let now = handles.tap.now();
        let now_ns = now.nanos().unwrap_or(wire::TS_NONE);
        let n = handles.tap.len();
        let mut w = Writer::new();
        wire::write_counters_prefix(&mut w, now_ns);
        wire::TableHeader { row_size: wire::ROW_SIZE_COUNTER, row_count: n as u16 }.write(&mut w);
        for i in 0..n {
            let el = ElementId(i as u32);
            let s = handles.tap.snapshot(el).unwrap_or_default();
            wire::CounterRow {
                element: i as u32,
                buffers_in: s.buffers_in,
                buffers_out: s.buffers_out,
                bytes_in: s.bytes_in,
                bytes_out: s.bytes_out,
                batches_in: s.batches_in,
                batches_out: s.batches_out,
                queue_high_water: s.queue_high_water,
                drops: s.drops,
            }
            .write(&mut w);
        }
        drop(handles);
        self.send(kind::COUNTERS, seq, &w.into_vec())
    }

    fn reply_latency(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let which = wire::decode_elem_req(payload).unwrap_or(wire::ELEM_ALL);
        let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
        let n = handles.tap.len();
        let indices: Vec<usize> = if which == wire::ELEM_ALL {
            (0..n).collect()
        } else if (which as usize) < n {
            vec![which as usize]
        } else {
            vec![]
        };
        let mut w = Writer::new();
        wire::TableHeader { row_size: wire::ROW_SIZE_LATENCY, row_count: indices.len() as u16 }
            .write(&mut w);
        for i in indices {
            let el = ElementId(i as u32);
            let (proc, queue, wait) = handles.tap.latency(el).unwrap_or_else(zero_hist_triplet);
            wire::LatencyRow {
                element: i as u32,
                process: to_wire_hist(&proc),
                queue: to_wire_hist(&queue),
                wait_lateness: to_wire_hist(&wait),
            }
            .write(&mut w);
        }
        drop(handles);
        self.send(kind::LATENCY, seq, &w.into_vec())
    }

    fn reply_latency_report(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
    ) -> std::io::Result<()> {
        let paths = {
            let g = shared.topology.lock().unwrap_or_else(|e| e.into_inner());
            g.latency_paths
                .iter()
                .map(|p| wire::LatencyPath {
                    sink: p.sink,
                    is_live: p.is_live as u8,
                    total_ns: p.total_ns,
                    elems: p
                        .elems
                        .iter()
                        .map(|(e, m)| wire::LatencyPathElem { element: *e, min_ns: *m })
                        .collect(),
                })
                .collect::<Vec<_>>()
        };
        self.send(kind::LATENCY_REPORT, seq, &wire::encode_latency_report(&paths))
    }

    fn reply_props(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let which = wire::decode_elem_req(payload).unwrap_or(wire::ELEM_ALL);
        let props: Vec<super::snapshot::SnapProp> = {
            let g = shared.topology.lock().unwrap_or_else(|e| e.into_inner());
            g.props
                .iter()
                .filter(|p| which == wire::ELEM_ALL || p.element == which)
                .cloned()
                .collect()
        };

        // Ask the live PropHandle for the *current* values (the snapshot's current may
        // be stale after a live set), addressed by index.
        let mut rows = Vec::with_capacity(props.len());
        for p in &props {
            let current = {
                let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
                handles.props.get(ElementId(p.element), p.prop_index as usize)
            };
            let name_str = self.intern_owned(&p.name)?;
            let current_wire = match current {
                Some(v) => self.value_to_wire(v, &p.allowed)?,
                None => wire::WireValue::UNSET,
            };
            let mut vals = Vec::with_capacity(p.allowed.len());
            for a in &p.allowed {
                vals.push(self.snapvalue_to_wire(a)?);
            }
            rows.push(wire::PropRow {
                element: p.element,
                prop_index: p.prop_index,
                live: p.live as u8,
                ckind: p.ckind,
                name_str,
                current: current_wire,
                vals,
            });
        }
        self.send(kind::PROPS, seq, &wire::encode_props(&rows))
    }

    /// Convert a live `Value` (the current property value read from the live
    /// `PropHandle`) to a `WireValue`. A categorical `Value::Id` carries only a pipeline
    /// `ValueId`, so its name is recovered from the prop's `allowed` snapshot values
    /// (which pair each `id_name` with its `ValueId` in `bits`) — the server never
    /// touches the pipeline interners. If the id isn't in `allowed` (a prop set to a
    /// value outside its own constraint, which the validated set path forbids), it falls
    /// back to `Unset` rather than emit a nameless id with no StrDef.
    fn value_to_wire(
        &mut self,
        v: Value,
        allowed: &[SnapValue],
    ) -> std::io::Result<wire::WireValue> {
        Ok(match v {
            Value::Int(i) => wire::WireValue::int(i),
            Value::Rat(n, d) => wire::WireValue::rat(n, d),
            Value::Id(ValueId(vid)) => {
                match allowed.iter().find(|a| a.tag as u32 == wire::WireValue::TAG_ID
                    && a.bits as u32 == vid)
                {
                    Some(a) => self.snapvalue_to_wire(a)?,
                    None => wire::WireValue::UNSET,
                }
            }
        })
    }

    /// Convert a resolved `SnapValue` to a `WireValue`, interning its categorical name
    /// and recording the mapping so a SetProp(tag=Id) maps back to the ValueId. The
    /// snapshot tag is a `u8`; the wire tag is a `u32`.
    fn snapvalue_to_wire(&mut self, v: &SnapValue) -> std::io::Result<wire::WireValue> {
        Ok(match v.tag as u32 {
            wire::WireValue::TAG_INT => wire::WireValue { tag: wire::WireValue::TAG_INT, bits: v.bits },
            wire::WireValue::TAG_RAT => wire::WireValue { tag: wire::WireValue::TAG_RAT, bits: v.bits },
            wire::WireValue::TAG_ID => {
                let conn_id = self.intern_owned(&v.id_name)?;
                // `bits` on a SnapValue::Id is the pipeline ValueId; map it.
                self.id_to_valueid.insert(conn_id, v.bits as u32);
                wire::WireValue::id(conn_id)
            }
            _ => wire::WireValue::UNSET,
        })
    }

    fn handle_set_prop(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let Some(req) = wire::SetPropReq::decode(payload) else {
            return self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::BAD_FRAME, "malformed SetProp"),
            );
        };
        // Map the wire value back to a pipeline `Value`.
        let value = match req.value.tag {
            wire::WireValue::TAG_INT => Some(Value::Int(req.value.bits as i64)),
            wire::WireValue::TAG_RAT => req
                .value
                .as_rat()
                .map(|(n, d)| Value::Rat(n, d)),
            wire::WireValue::TAG_ID => {
                // The bits are a connection string id; map to a pipeline ValueId. A new
                // string interned via SetProp is Error(Unsupported) in v1.
                match self.id_to_valueid.get(&(req.value.bits as u32)) {
                    Some(&vid) => Some(Value::Id(ValueId(vid))),
                    None => {
                        return self.send(
                            kind::ERROR,
                            seq,
                            &wire::encode_error(
                                errcode::UNSUPPORTED,
                                "interning a new categorical value via SetProp is not supported in v1",
                            ),
                        );
                    }
                }
            }
            _ => None,
        };
        let Some(value) = value else {
            return self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::BAD_FRAME, "SetProp value has no valid tag"),
            );
        };
        let result = {
            let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
            handles.props.set_by_index(
                ElementId(req.element),
                req.prop_index as usize,
                value,
            )
        };
        match result {
            Ok(()) => self.send(kind::ACK, seq, &[]),
            Err(e) => self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::REJECTED, &err_message(&e)),
            ),
        }
    }

    fn handle_set_log_level(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let Some(req) = wire::SetLogLevelReq::decode(payload) else {
            return self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::BAD_FRAME, "malformed SetLogLevel"),
            );
        };
        let level = if req.rank == 0 { None } else { Level::from_rank(req.rank) };
        if req.rank != 0 && level.is_none() {
            return self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::BAD_FRAME, "log rank out of range (0..5)"),
            );
        }
        let handles = shared.handles.lock().unwrap_or_else(|e| e.into_inner());
        if req.element == wire::SetLogLevelReq::GLOBAL {
            // Flip every filter live (per-element filters may alias the global one).
            for f in &handles.filters {
                f.set(level);
            }
            if let Some(g) = &handles.global_filter {
                g.set(level);
            }
        } else if let Some(f) = handles.filters.get(req.element as usize) {
            f.set(level);
        }
        drop(handles);
        self.send(kind::ACK, seq, &[])
    }

    fn handle_set_tracing(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let Some(on) = wire::decode_set_tracing(payload) else {
            return self.send(
                kind::ERROR,
                seq,
                &wire::encode_error(errcode::BAD_FRAME, "malformed SetTracing"),
            );
        };
        shared
            .handles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tracing
            .store(on, Ordering::Release);
        self.send(kind::ACK, seq, &[])
    }

    fn handle_subscribe(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let mask = wire::decode_sub(payload).unwrap_or(0);
        if mask & wire::submask::BUS != 0 && self.bus_sub.is_none() {
            let (tx, rx) = crate::ring::spsc::<BusTapRow>(TAP_RING_CAP);
            let id = shared.bus_port.attach(tx);
            self.bus_sub = Some((id, rx, 0));
        }
        if mask & wire::submask::LOGS != 0 && self.log_sub.is_none() {
            let (id, rx) = shared.log_taps.attach(TAP_RING_CAP);
            self.log_sub = Some((id, rx, 0));
        }
        self.send(kind::ACK, seq, &[])
    }

    fn handle_unsubscribe(
        &mut self,
        shared: &Arc<IntrospectShared>,
        seq: u16,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let mask = wire::decode_sub(payload).unwrap_or(0);
        if mask & wire::submask::BUS != 0 {
            if let Some((id, _, _)) = self.bus_sub.take() {
                shared.bus_port.detach(id);
            }
        }
        if mask & wire::submask::LOGS != 0 {
            if let Some((id, _, _)) = self.log_sub.take() {
                shared.log_taps.detach(id);
            }
        }
        self.send(kind::ACK, seq, &[])
    }

    // --- Pushed streams ----------------------------------------------------

    fn drain_pushes(&mut self, shared: &Arc<IntrospectShared>) -> std::io::Result<()> {
        // Bus tap → BusMsg frames (+ Dropped when the tap's drop counter advanced).
        if let Some((id, rx, last_dropped)) = self.bus_sub.take() {
            let mut rows = Vec::new();
            while let Some(row) = rx.try_pop() {
                rows.push(row);
            }
            for row in rows {
                let frame = wire::BusMsgRow {
                    seq: row.seq,
                    kind: row.kind,
                    class: row.class,
                    msg_len: row.msg_len,
                    a: row.a,
                    b: row.b,
                    c: row.c,
                    d: row.d,
                    msg: row.msg,
                };
                self.send(kind::BUS_MSG, 0, &frame.encode())?;
            }
            let now_dropped = shared.bus_port.take_dropped(id);
            if now_dropped != last_dropped {
                self.send(
                    kind::DROPPED,
                    0,
                    &wire::encode_dropped(wire::stream::BUS, now_dropped),
                )?;
            }
            self.bus_sub = Some((id, rx, now_dropped));
        }

        // Log tap → LogRec frames (intern name/event/str fields on this thread).
        if let Some((id, rx, last_dropped)) = self.log_sub.take() {
            let mut recs = Vec::new();
            while let Some(rec) = rx.try_pop() {
                recs.push(rec);
            }
            for rec in recs {
                let row = self.log_record_to_row(&rec)?;
                self.send(kind::LOG_REC, 0, &row.encode())?;
            }
            let now_dropped = shared.log_taps.dropped(id);
            if now_dropped != last_dropped {
                self.send(
                    kind::DROPPED,
                    0,
                    &wire::encode_dropped(wire::stream::LOGS, now_dropped),
                )?;
            }
            self.log_sub = Some((id, rx, now_dropped));
        }

        Ok(())
    }

    /// Convert an in-process `LogRecord` (process-'static pointers) to a wire row,
    /// interning the element name, event name, string-field keys, and any string values.
    fn log_record_to_row(
        &mut self,
        rec: &crate::log::LogRecord,
    ) -> std::io::Result<wire::LogRecRow> {
        let name_str = self.intern_static(rec.name)?;
        let event_str = self.intern_static(rec.event)?;
        let mut fields = [wire::LogField::ZERO; wire::LOG_MAX_FIELDS];
        let n = (rec.nfields as usize).min(wire::LOG_MAX_FIELDS);
        for (slot, f) in fields.iter_mut().zip(&rec.fields[..n]) {
            let key_str = self.intern_static(f.key)?;
            let (tag, bits) = match f.val {
                crate::log::FieldValue::Int(v) => (wire::LogField::TAG_INT, v as u64),
                crate::log::FieldValue::Uint(v) => (wire::LogField::TAG_UINT, v),
                crate::log::FieldValue::Bool(v) => (wire::LogField::TAG_BOOL, v as u64),
                crate::log::FieldValue::Str(s) => {
                    (wire::LogField::TAG_STR, self.intern_static(s)? as u64)
                }
                crate::log::FieldValue::Fourcc(b) => (
                    wire::LogField::TAG_FOURCC,
                    u32::from_le_bytes(b) as u64,
                ),
                crate::log::FieldValue::Time(t) => (wire::LogField::TAG_TIME, t.0),
            };
            *slot = wire::LogField { key_str, tag, bits };
        }
        Ok(wire::LogRecRow {
            ts: rec.ts.0,
            element: rec.element.0,
            name_str,
            event_str,
            level: rec.level.rank(),
            nfields: n as u8,
            fields,
        })
    }

    /// Detach subscriptions on teardown (RAII: unsubscribe/disconnect drops the taps).
    fn teardown(&mut self, shared: &Arc<IntrospectShared>) {
        if let Some((id, _, _)) = self.bus_sub.take() {
            shared.bus_port.detach(id);
        }
        if let Some((id, _, _)) = self.log_sub.take() {
            shared.log_taps.detach(id);
        }
    }
}

// --- helpers ---------------------------------------------------------------

fn to_wire_hist(s: &crate::counters::LatencySnapshot) -> wire::WireHistogram {
    wire::WireHistogram {
        count: s.count,
        sum_ns: s.sum_ns,
        max_ns: s.max_ns,
        buckets: s.buckets,
    }
}

fn zero_hist_triplet() -> (
    crate::counters::LatencySnapshot,
    crate::counters::LatencySnapshot,
    crate::counters::LatencySnapshot,
) {
    let z = crate::counters::LatencySnapshot {
        buckets: [0; wire::LATENCY_BUCKETS],
        count: 0,
        sum_ns: 0,
        max_ns: 0,
    };
    (z, z, z)
}

/// Human text for an `Error` (it has no `Display`), for a Rejected reply message.
fn err_message(e: &crate::error::Error) -> String {
    match e {
        crate::error::Error::NegotiationFailed => "negotiation failed".to_string(),
        crate::error::Error::Element { message, .. } => message.clone(),
        crate::error::Error::Resource(m) => m.clone(),
        crate::error::Error::Todo(m) => (*m).to_string(),
    }
}
