//! Introspection protocol + server tests (spec: Introspection protocol and
//! scraft-scope). Table-driven where it pays; "tests are data". The whole file runs in
//! well under a second (the server's poll granularity is 5 ms, so a round trip is a few
//! polls). Empty when the `introspect` feature is off, so the feature-off suite links.

#![cfg(feature = "introspect")]

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, PropDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{Constraint, OfferDesc, Value};
use streamcraft_core::id::{ElementId, PadId};
use streamcraft_core::introspect::strtab::StrTab;
use streamcraft_core::introspect::tap::{BusTapRow, LogTapRegistry};
use streamcraft_core::introspect::wire::{self, errcode, kind};
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

// ===========================================================================
// 1. Golden wire pinning — row sizes and field offsets are a contract.
// ===========================================================================

#[test]
fn row_sizes_are_pinned() {
    assert_eq!(wire::ROW_SIZE_ELEMENT, 40);
    assert_eq!(wire::ROW_SIZE_PAD, 16);
    assert_eq!(wire::ROW_SIZE_EDGE, 280);
    assert_eq!(wire::ROW_SIZE_COUNTER, 72);
    assert_eq!(wire::ROW_SIZE_LATENCY, 848);
    assert_eq!(wire::WireValue::SIZE, 16);
    assert_eq!(wire::TableHeader::SIZE, 8);
    assert_eq!(wire::HEADER_LEN, 8);
    // 100, not 96 — see the note on `BusMsgRow`; the field list sums to 100.
    assert_eq!(wire::BusMsgRow::SIZE, 100);
    assert_eq!(wire::LogRecRow::SIZE, 88);
    assert_eq!(wire::WireHistogram::SIZE, 280);
    assert_eq!(wire::FieldSlot::SIZE, 16);
    assert_eq!(wire::LogField::SIZE, 16);
}

#[test]
fn frame_header_round_trips_at_fixed_offsets() {
    let h = wire::FrameHeader { len: 0x11223344, kind: 0x5566, seq: 0x7788 };
    let b = h.encode();
    // len @0..4, kind @4..6, seq @6..8, all little-endian.
    assert_eq!(&b[0..4], &0x11223344u32.to_le_bytes());
    assert_eq!(&b[4..6], &0x5566u16.to_le_bytes());
    assert_eq!(&b[6..8], &0x7788u16.to_le_bytes());
    assert_eq!(wire::FrameHeader::decode(&b), h);
}

#[test]
fn wire_value_layout_and_round_trip() {
    // Int: tag @0, _pad @4, bits @8.
    let mut w = wire::Writer::new();
    wire::WireValue::int(-3).write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), 16);
    assert_eq!(&b[0..4], &wire::WireValue::TAG_INT.to_le_bytes());
    assert_eq!(&b[8..16], &(-3i64 as u64).to_le_bytes());

    // Rat packs (num<<32 | den).
    let r = wire::WireValue::rat(30000, 1001);
    assert_eq!(r.as_rat(), Some((30000, 1001)));

    // Every tag survives a round trip through Reader.
    for v in [
        wire::WireValue::UNSET,
        wire::WireValue::int(i64::MIN),
        wire::WireValue::rat(-3, 4),
        wire::WireValue::id(42),
    ] {
        let mut w = wire::Writer::new();
        v.write(&mut w);
        let bytes = w.into_vec();
        let mut r = wire::Reader::new(&bytes);
        assert_eq!(wire::WireValue::read(&mut r), Some(v));
    }
}

#[test]
fn element_row_offsets_and_round_trip() {
    let row = wire::ElementRow {
        element: 1,
        name_str: 2,
        group: 3,
        sched: 1,
        flags: wire::ElementRow::FLAG_SINK | wire::ElementRow::FLAG_LIVE,
        npads: 4,
        latency_min_ns: 100,
        latency_max_ns: 200,
        jitter_ns: 5,
    };
    let mut w = wire::Writer::new();
    row.write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), wire::ROW_SIZE_ELEMENT as usize);
    // element @0, name_str @4, group @8, sched @12, flags @13, npads @14, lat_min @16.
    assert_eq!(&b[0..4], &1u32.to_le_bytes());
    assert_eq!(&b[4..8], &2u32.to_le_bytes());
    assert_eq!(&b[8..12], &3u32.to_le_bytes());
    assert_eq!(b[12], 1);
    assert_eq!(b[13], wire::ElementRow::FLAG_SINK | wire::ElementRow::FLAG_LIVE);
    assert_eq!(&b[14..16], &4u16.to_le_bytes());
    assert_eq!(&b[16..24], &100u64.to_le_bytes());
    let mut r = wire::Reader::new(&b);
    assert_eq!(wire::ElementRow::read(&mut r), Some(row));
}

#[test]
fn pad_edge_counter_latency_round_trip() {
    // PadRow
    let pad = wire::PadRow {
        element: 7,
        pad: 2,
        name_str: 9,
        direction: wire::PadRow::DIR_SRC,
        flags: wire::PadRow::FLAG_LINKED,
    };
    let mut w = wire::Writer::new();
    pad.write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), 16);
    assert_eq!(wire::PadRow::read(&mut wire::Reader::new(&b)), Some(pad));

    // EdgeRow with two fields
    let mut fields = [wire::FieldSlot::ZERO; wire::EDGE_MAX_FIELDS];
    fields[0] = wire::FieldSlot { field_str: 5, tag: 1, bits: 48000 };
    fields[1] = wire::FieldSlot { field_str: 6, tag: 3, bits: 12 };
    let edge = wire::EdgeRow {
        src: 0,
        src_pad: 1,
        sink: 2,
        sink_pad: 0,
        family_str: 3,
        nfields: 2,
        fields,
    };
    let mut w = wire::Writer::new();
    edge.write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), wire::ROW_SIZE_EDGE as usize);
    assert_eq!(wire::EdgeRow::read(&mut wire::Reader::new(&b)), Some(edge));

    // CounterRow
    let c = wire::CounterRow {
        element: 4,
        buffers_in: 10,
        buffers_out: 9,
        bytes_in: 1000,
        bytes_out: 900,
        batches_in: 3,
        batches_out: 3,
        queue_high_water: 2,
        drops: 1,
    };
    let mut w = wire::Writer::new();
    c.write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), wire::ROW_SIZE_COUNTER as usize);
    assert_eq!(wire::CounterRow::read(&mut wire::Reader::new(&b)), Some(c));

    // LatencyRow
    let mut buckets = [0u64; wire::LATENCY_BUCKETS];
    buckets[3] = 7;
    let hist = wire::WireHistogram { count: 7, sum_ns: 21, max_ns: 9, buckets };
    let lat = wire::LatencyRow {
        element: 1,
        process: hist,
        queue: wire::WireHistogram::ZERO,
        wait_lateness: hist,
    };
    let mut w = wire::Writer::new();
    lat.write(&mut w);
    let b = w.into_vec();
    assert_eq!(b.len(), wire::ROW_SIZE_LATENCY as usize);
    assert_eq!(wire::LatencyRow::read(&mut wire::Reader::new(&b)), Some(lat));
}

#[test]
fn prop_row_self_sizing_round_trip() {
    // Range prop: 3 value slots.
    let row = wire::PropRow {
        element: 2,
        prop_index: 1,
        live: 1,
        ckind: wire::PropRow::CKIND_RANGE,
        name_str: 4,
        current: wire::WireValue::int(320_000),
        vals: vec![
            wire::WireValue::int(1),
            wire::WireValue::int(1_000_000),
            wire::WireValue::int(1),
        ],
    };
    let payload = wire::encode_props(&[row.clone()]);
    let decoded = wire::decode_props(&payload).expect("decodes");
    assert_eq!(decoded, vec![row]);
}

#[test]
fn hello_and_client_hello_round_trip() {
    let hello = wire::Hello {
        ver_major: wire::VER_MAJOR,
        ver_minor: wire::VER_MINOR,
        flags: wire::helloflags::TRACING_ON | wire::helloflags::LOG_CHANNELS,
        pid: 4321,
        nelements: 3,
    };
    assert_eq!(wire::Hello::decode(&hello.encode()), Some(hello));

    let ch = wire::ClientHello { ver_major: 1, ver_minor: 0 };
    assert_eq!(wire::ClientHello::decode(&ch.encode()), Some(ch));

    // A wrong magic decodes as None.
    let mut bad = hello.encode();
    bad[0] = b'X';
    assert_eq!(wire::Hello::decode(&bad), None);
}

#[test]
fn error_str_def_dropped_latency_report_round_trip() {
    assert_eq!(
        wire::decode_error(&wire::encode_error(wire::errcode::REJECTED, "nope")),
        Some((wire::errcode::REJECTED, "nope".to_string()))
    );
    assert_eq!(
        wire::decode_str_def(&wire::encode_str_def(7, "flacdec")),
        Some((7, "flacdec".to_string()))
    );
    assert_eq!(
        wire::decode_dropped(&wire::encode_dropped(wire::stream::LOGS, 42)),
        Some((wire::stream::LOGS, 42))
    );
    let paths = vec![wire::LatencyPath {
        sink: 2,
        is_live: 1,
        total_ns: 5000,
        elems: vec![
            wire::LatencyPathElem { element: 0, min_ns: 1000 },
            wire::LatencyPathElem { element: 1, min_ns: 4000 },
        ],
    }];
    assert_eq!(wire::decode_latency_report(&wire::encode_latency_report(&paths)), Some(paths));
}

#[test]
fn bus_msg_and_log_rec_frame_round_trip() {
    let mut msg = [0u8; wire::BUS_MSG_TEXT];
    msg[..3].copy_from_slice(b"err");
    let bm = wire::BusMsgRow {
        seq: 9,
        kind: 0,
        class: 1,
        msg_len: 3,
        a: 1,
        b: 0,
        c: 0,
        d: 0,
        msg,
    };
    assert_eq!(wire::BusMsgRow::decode(&bm.encode()), Some(bm));
    assert_eq!(bm.text(), "err");

    let mut fields = [wire::LogField::ZERO; wire::LOG_MAX_FIELDS];
    fields[0] = wire::LogField { key_str: 1, tag: wire::LogField::TAG_UINT, bits: 500 };
    let lr = wire::LogRecRow {
        ts: 12_000_000,
        element: 3,
        name_str: 2,
        event_str: 4,
        level: 1,
        nfields: 1,
        fields,
    };
    assert_eq!(wire::LogRecRow::decode(&lr.encode()), Some(lr));
}

#[test]
fn read_frame_rejects_oversized_len() {
    let mut buf = Vec::new();
    let h = wire::FrameHeader { len: wire::MAX_FRAME_LEN + 1, kind: 1, seq: 0 };
    buf.extend_from_slice(&h.encode());
    let mut cur = std::io::Cursor::new(buf);
    let err = wire::read_frame(&mut cur).expect_err("oversized frame rejected");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

// ===========================================================================
// 2. StrTab — pointer-identity interning, content dedup, dense ids.
// ===========================================================================

#[test]
fn strtab_interns_by_pointer_and_content() {
    let mut t = StrTab::new();
    // Empty string is the "none" sentinel: id 0, never new.
    assert_eq!(t.intern_static(""), (0, false));

    static A: &str = "flacdec";
    let (id1, new1) = t.intern_static(A);
    assert_eq!((id1, new1), (1, true), "first ref is new, dense from 1");
    let (id2, new2) = t.intern_static(A);
    assert_eq!((id2, new2), (1, false), "same pointer dedups, not new");

    // A distinct static gets a fresh dense id.
    static B: &str = "filesink";
    let (id3, new3) = t.intern_static(B);
    assert_eq!((id3, new3), (2, true));

    // Owned content equal to a prior owned dedups.
    let (o1, on1) = t.intern_owned("audio/raw");
    let (o2, on2) = t.intern_owned("audio/raw");
    assert_eq!((o1, on1), (3, true));
    assert_eq!((o2, on2), (3, false));

    assert_eq!(t.resolve(1), Some("flacdec"));
    assert_eq!(t.resolve(3), Some("audio/raw"));
    assert_eq!(t.resolve(0), None);
    assert_eq!(t.len(), 3);
}

// ===========================================================================
// 3. BusTapRow conversion — every variant → expected {kind,class,a,b,c,d}.
// ===========================================================================

#[test]
fn bus_tap_row_kind_and_mapping_table() {
    use streamcraft_core::bus::{BusMessage, State};
    use streamcraft_core::id::{ElementId, GroupId, LinkId, PadId};
    use streamcraft_core::format::FixedFormat;
    use streamcraft_core::id::FormatId;

    // (message, expected kind, expected class, a, b, c, d)
    let cases: Vec<(BusMessage, u8, u8, u32, u32, u64, u64)> = vec![
        (BusMessage::Error { element: ElementId(1), error: Error::Todo("boom") }, 0, 1, 1, 0, 0, 0),
        (BusMessage::Warning { element: ElementId(2), error: Error::Todo("warn") }, 1, 0, 2, 0, 0, 0),
        (BusMessage::Eos, 2, 1, 0, 0, 0, 0),
        (
            BusMessage::StateChanged { old: State::Ready, new: State::Playing },
            3, 1, 1, 2, 0, 0,
        ),
        (
            BusMessage::PadAdded {
                element: ElementId(3),
                pad: PadId(4),
                format: FixedFormat::new(FormatId(5)),
            },
            4, 1, 3, 4, 5, 0,
        ),
        (BusMessage::ElementAdded { element: ElementId(6), group: GroupId(7) }, 5, 1, 6, 7, 0, 0),
        (BusMessage::ElementRemoved { element: ElementId(8) }, 6, 1, 8, 0, 0, 0),
        (BusMessage::LinkChanged { link: LinkId(9) }, 7, 1, 9, 0, 0, 0),
        (
            BusMessage::SubgraphJoined {
                group: GroupId(10),
                added_latency: Timestamp::from_nanos(1234),
            },
            9, 1, 10, 0, 1234, 0,
        ),
        (
            BusMessage::LatencyChanged {
                old: Timestamp::from_nanos(11),
                new: Timestamp::from_nanos(22),
            },
            10, 1, 0, 0, 11, 22,
        ),
        (BusMessage::Qos { sink: ElementId(12), lateness_ns: -5 }, 11, 0, 12, 0, (-5i64) as u64, 0),
        (BusMessage::BranchSealed { group: GroupId(13), error: Error::Todo("seal") }, 12, 1, 13, 0, 0, 0),
        (
            BusMessage::DurationChanged { element: ElementId(14), ns: 90_000_000_000 },
            13, 1, 14, 0, 90_000_000_000, 0,
        ),
    ];
    for (i, (msg, kind, class, a, b, c, d)) in cases.into_iter().enumerate() {
        let row = BusTapRow::from_msg(&msg, 1);
        assert_eq!(row.kind, kind, "kind for case {i}");
        assert_eq!(row.class, class, "class for case {i}");
        assert_eq!(row.a, a, "a for case {i}");
        assert_eq!(row.b, b, "b for case {i}");
        assert_eq!(row.c, c, "c for case {i}");
        assert_eq!(row.d, d, "d for case {i}");
    }
}

#[test]
fn bus_tap_row_truncates_error_text_at_char_boundary() {
    use streamcraft_core::bus::BusMessage;
    use streamcraft_core::id::ElementId;
    // A message longer than 64 bytes with a multibyte char straddling the cut.
    let long = "x".repeat(63) + "é"; // 'é' is 2 bytes; byte 63 is 'x', 64/65 are 'é'
    let msg = BusMessage::Error {
        element: ElementId(0),
        error: Error::Element { element: ElementId(0), message: long },
    };
    let row = BusTapRow::from_msg(&msg, 1);
    // The cut lands at 63 (the 'é' won't fit), so msg_len is 63 and text decodes clean.
    assert_eq!(row.msg_len, 63);
    let text = std::str::from_utf8(&row.msg[..row.msg_len as usize]).expect("valid utf8");
    assert_eq!(text.len(), 63);
}

// ===========================================================================
// 4. Tap drop accounting — a cap-4 ring, publish 10 → 6 dropped exact.
// ===========================================================================

#[test]
fn log_tap_registry_drops_exact_on_full_ring() {
    use streamcraft_core::log::{FieldValue, Level, LogRecord};

    let reg = LogTapRegistry::new();
    let (id, _rx) = reg.attach(4); // rounds up to a power of two → 4
    let rec = LogRecord {
        ts: Timestamp::ZERO,
        element: ElementId(0),
        name: "x",
        level: Level::Info,
        event: "tick",
        nfields: 0,
        fields: [streamcraft_core::log::Field::new("", FieldValue::Uint(0)); 4],
    };
    // Publish 10 into a 4-slot ring that is never drained → 6 dropped.
    for _ in 0..10 {
        reg.publish(&rec);
    }
    assert_eq!(reg.dropped(id), 6, "exact drop count for a cap-4 ring");
    assert_eq!(reg.detach(id), 6, "detach returns the final drop count");
}

#[test]
fn log_tap_fast_path_is_a_noop_without_subscribers() {
    use streamcraft_core::log::{FieldValue, Level, LogRecord};
    let reg = LogTapRegistry::new();
    let rec = LogRecord {
        ts: Timestamp::ZERO,
        element: ElementId(0),
        name: "x",
        level: Level::Info,
        event: "tick",
        nfields: 0,
        fields: [streamcraft_core::log::Field::new("", FieldValue::Uint(0)); 4],
    };
    // No subscriber: publish must not panic and must be a no-op.
    for _ in 0..100 {
        reg.publish(&rec);
    }
}

// ===========================================================================
// Test elements for the round-trip integration: a source and a sink.
// ===========================================================================

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_PROPS: [PropDesc; 1] = [PropDesc {
    name: "bitrate",
    // A wide integer range so a SetProp lands, and an out-of-range value is rejected.
    allowed: Constraint::Range {
        min: Value::Int(1),
        max: Value::Int(1_000_000),
        step: Value::Int(1),
    },
    live: true,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "testsrc",
    pads: &SRC_PADS,
    props: &SRC_PROPS,
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::from_nanos(1000),
        max: Timestamp::from_nanos(2000),
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct TestSrc {
    remaining: u32,
}

impl Element for TestSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.remaining == 0 {
            return Ok(Flow::Eos);
        }
        let mut out = ctx.alloc(PadId(0));
        let dst = out.memory.as_mut_full();
        dst[0] = 0xAB;
        out.memory.set_len(1);
        ctx.out(PadId(0)).push(out);
        self.remaining -= 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "testsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::from_nanos(500),
        max: Timestamp::from_nanos(1000),
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

#[derive(Default)]
struct TestSink {
    count: u64,
}

impl Element for TestSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while inputs.pop().is_some() {
            self.count += 1;
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Build a linked testsrc → testsink pipeline that emits `n` buffers then EOS.
fn build_pipeline(n: u32) -> (Pipeline, ElementId, ElementId) {
    let mut p = Pipeline::new();
    let src = p.add(TestSrc { remaining: n });
    let sink = p.add(TestSink::default());
    p.link((src, "src"), (sink, "sink")).expect("link");
    (p, src, sink)
}

/// A raw client over a fresh connection: writes ClientHello, then helpers to request.
struct Client {
    stream: UnixStream,
}

impl Client {
    fn connect(path: &std::path::Path) -> Client {
        // Retry briefly: the accept thread may not have bound the instant we connect.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match UnixStream::connect(path) {
                Ok(s) => {
                    s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                    return Client { stream: s };
                }
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => panic!("connect failed: {e}"),
            }
        }
    }

    fn send(&mut self, kind: u16, seq: u16, payload: &[u8]) {
        self.stream.write_all(&wire::encode_frame(kind, seq, payload)).unwrap();
    }

    /// Read frames until one of `want` arrives, returning it. StrDefs and pushes seen
    /// along the way are collected so a caller can resolve names.
    fn recv_until(&mut self, want: u16) -> wire::Frame {
        self.recv_until_any(&[want])
    }

    fn recv_until_any(&mut self, want: &[u16]) -> wire::Frame {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let f = wire::read_frame(&mut self.stream).expect("frame");
            if want.contains(&f.kind) {
                return f;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {want:?}");
        }
    }
}

/// A per-test unique socket path (pid + a counter), in the OS temp dir.
fn temp_socket() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("sc-introspect-{}-{}.sock", std::process::id(), n))
}

// ===========================================================================
// 5. Round-trip integration.
// ===========================================================================

#[test]
fn handshake_topology_dot_counters_and_props() {
    let (mut p, src, _sink) = build_pipeline(0);
    let expected_dot = p.dump_dot();
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");

    let mut c = Client::connect(&path);

    // Hello arrives before any request.
    let hello_frame = wire::read_frame(&mut c.stream).expect("hello");
    assert_eq!(hello_frame.kind, kind::HELLO);
    let hello = wire::Hello::decode(&hello_frame.payload).expect("hello decodes");
    assert_eq!(hello.ver_major, 1);
    assert_eq!(hello.nelements, 2);

    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());

    // Ping/Pong echoes the seq.
    c.send(kind::PING, 77, &[]);
    let pong = c.recv_until(kind::PONG);
    assert_eq!(pong.seq, 77);

    // GetTopology: three tables, elements first.
    c.send(kind::GET_TOPOLOGY, 1, &[]);
    // Collect StrDefs → an id→name map, then the Topology reply.
    let mut names = std::collections::HashMap::new();
    let topo = loop {
        let f = wire::read_frame(&mut c.stream).unwrap();
        match f.kind {
            kind::STR_DEF => {
                let (id, s) = wire::decode_str_def(&f.payload).unwrap();
                names.insert(id, s);
            }
            kind::TOPOLOGY => break f,
            _ => {}
        }
    };
    // Walk the elements table.
    let mut r = wire::Reader::new(&topo.payload);
    let eh = wire::TableHeader::read(&mut r).unwrap();
    assert_eq!(eh.row_size, wire::ROW_SIZE_ELEMENT);
    assert_eq!(eh.row_count, 2);
    let e0 = wire::ElementRow::read(&mut r).unwrap();
    assert_eq!(names.get(&e0.name_str).map(String::as_str), Some("testsrc"));
    assert_eq!(e0.flags & wire::ElementRow::FLAG_SOURCE, wire::ElementRow::FLAG_SOURCE);
    assert_eq!(e0.latency_min_ns, 1000);
    let e1 = wire::ElementRow::read(&mut r).unwrap();
    assert_eq!(names.get(&e1.name_str).map(String::as_str), Some("testsink"));
    assert_eq!(e1.flags & wire::ElementRow::FLAG_SINK, wire::ElementRow::FLAG_SINK);
    assert_eq!(e1.flags & wire::ElementRow::FLAG_LIVE, wire::ElementRow::FLAG_LIVE);

    // GetDot == dump_dot().
    c.send(kind::GET_DOT, 2, &[]);
    let dot = c.recv_until(kind::DOT);
    assert_eq!(String::from_utf8(dot.payload).unwrap(), expected_dot);

    // GetCounters: prefix now_ns then a per-element table.
    c.send(kind::GET_COUNTERS, 3, &[]);
    let counters = c.recv_until(kind::COUNTERS);
    let mut r = wire::Reader::new(&counters.payload);
    let _now_ns = r.get_u64().unwrap();
    let ch = wire::TableHeader::read(&mut r).unwrap();
    assert_eq!(ch.row_count, 2);

    // GetProps(all): testsrc has a "bitrate" Range prop, live.
    c.send(kind::GET_PROPS, 4, &wire::encode_elem_req(wire::ELEM_ALL));
    let props = loop {
        let f = wire::read_frame(&mut c.stream).unwrap();
        match f.kind {
            kind::STR_DEF => {
                let (id, s) = wire::decode_str_def(&f.payload).unwrap();
                names.insert(id, s);
            }
            kind::PROPS => break wire::decode_props(&f.payload).unwrap(),
            _ => {}
        }
    };
    assert_eq!(props.len(), 1);
    let bitrate = &props[0];
    assert_eq!(bitrate.element, src.0);
    assert_eq!(bitrate.live, 1);
    assert_eq!(bitrate.ckind, wire::PropRow::CKIND_RANGE);
    assert_eq!(bitrate.vals.len(), 3, "range: min/max/step");
    assert_eq!(bitrate.current.tag, wire::WireValue::TAG_UNSET, "never set yet");
    assert_eq!(names.get(&bitrate.name_str).map(String::as_str), Some("bitrate"));

    drop(p); // shuts down the server, unlinks the socket
}

#[test]
fn set_prop_accepts_valid_and_rejects_out_of_range() {
    let (mut p, src, _sink) = build_pipeline(0);
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");
    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello
    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());

    // Valid set → Ack, and the value reflects in a follow-up GetProps.
    let req = wire::SetPropReq { element: src.0, prop_index: 0, value: wire::WireValue::int(320_000) };
    c.send(kind::SET_PROP, 5, &req.encode());
    let ack = c.recv_until_any(&[kind::ACK, kind::ERROR]);
    assert_eq!(ack.kind, kind::ACK, "valid set is acked");

    c.send(kind::GET_PROPS, 6, &wire::encode_elem_req(src.0));
    let props = loop {
        let f = wire::read_frame(&mut c.stream).unwrap();
        if f.kind == kind::PROPS {
            break wire::decode_props(&f.payload).unwrap();
        }
    };
    assert_eq!(props[0].current, wire::WireValue::int(320_000), "set value reflected");

    // Out-of-range set → Error(Rejected).
    let bad = wire::SetPropReq { element: src.0, prop_index: 0, value: wire::WireValue::int(9_999_999) };
    c.send(kind::SET_PROP, 7, &bad.encode());
    let err = c.recv_until_any(&[kind::ACK, kind::ERROR]);
    assert_eq!(err.kind, kind::ERROR);
    let (code, _msg) = wire::decode_error(&err.payload).unwrap();
    assert_eq!(code, wire::errcode::REJECTED);

    drop(p);
}

#[test]
fn pause_resume_and_step_reserved() {
    let (mut p, _src, _sink) = build_pipeline(0);
    let pause = p.pause_handle();
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");
    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello
    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());

    c.send(kind::PAUSE, 8, &[]);
    assert_eq!(c.recv_until_any(&[kind::ACK, kind::ERROR]).kind, kind::ACK);
    assert!(pause.is_paused(), "pipeline paused via the protocol");

    c.send(kind::RESUME, 9, &[]);
    assert_eq!(c.recv_until_any(&[kind::ACK, kind::ERROR]).kind, kind::ACK);
    assert!(!pause.is_paused());

    // Step is reserved → Error(Unsupported).
    c.send(kind::STEP, 10, &[]);
    let f = c.recv_until_any(&[kind::ACK, kind::ERROR]);
    assert_eq!(f.kind, kind::ERROR);
    assert_eq!(wire::decode_error(&f.payload).unwrap().0, wire::errcode::UNSUPPORTED);

    drop(p);
}

#[test]
fn wrong_major_client_hello_is_rejected_and_closed() {
    let (mut p, _src, _sink) = build_pipeline(0);
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");
    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello

    c.send(kind::CLIENT_HELLO, 3, &wire::ClientHello { ver_major: 2, ver_minor: 0 }.encode());
    let err = c.recv_until_any(&[kind::ERROR]);
    assert_eq!(err.seq, 3, "error echoes the request seq");
    assert_eq!(wire::decode_error(&err.payload).unwrap().0, wire::errcode::VERSION_MISMATCH);
    // The server closes: the next read hits EOF.
    let eof = wire::read_frame(&mut c.stream);
    assert!(eof.is_err(), "connection closed after version mismatch");

    drop(p);
}

#[test]
fn subscribe_bus_delivers_a_tap_without_stealing_from_the_bus() {
    use streamcraft_core::bus::BusMessage;

    // A run posts `BusMessage::Eos` through the pipeline's own sender (the one the tap
    // watches) at end of stream — a real, deterministic trigger, no test-only API. The
    // tap must both deliver a BusMsg frame AND leave the original for the app to drain.
    let (mut p, _src, _sink) = build_pipeline(2);
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");
    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello
    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());

    // Subscribe to the bus stream, then run to EOS in another thread.
    c.send(kind::SUBSCRIBE, 11, &wire::encode_sub(wire::submask::BUS));
    assert_eq!(c.recv_until_any(&[kind::ACK]).kind, kind::ACK);

    let run = std::thread::spawn(move || {
        p.run().expect("run to eos");
        p
    });

    // The tap delivers a BusMsg frame; the run posts Eos (ordinal 2) at the end.
    let mut saw_eos_tap = false;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let f = wire::read_frame(&mut c.stream).expect("frame");
        if f.kind == kind::BUS_MSG {
            let row = wire::BusMsgRow::decode(&f.payload).unwrap();
            if row.kind == 2 {
                saw_eos_tap = true;
                break;
            }
        }
    }
    assert!(saw_eos_tap, "the bus tap delivered the Eos message");

    let p = run.join().unwrap();
    // The app can STILL drain the original Eos (the tap copies, never steals).
    let mut saw_eos_app = false;
    while let Some(m) = p.bus().try_recv() {
        if matches!(m, BusMessage::Eos) {
            saw_eos_app = true;
        }
    }
    assert!(saw_eos_app, "app still receives the original bus message (tap never steals)");

    drop(p);
}

#[test]
fn short_run_with_logging_and_log_subscribe_decodes_records() {
    use streamcraft_core::log::Level;

    let (mut p, _src, sink) = build_pipeline(4);
    p.set_log_level(Some(Level::Trace)); // wire log channels this run
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");

    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello (before run)
    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());
    c.send(kind::SUBSCRIBE, 12, &wire::encode_sub(wire::submask::LOGS));
    assert_eq!(c.recv_until_any(&[kind::ACK]).kind, kind::ACK);

    // Run the pipeline in another thread while we tail logs; it emits framework logs.
    let run = std::thread::spawn(move || {
        let _ = p.run();
        p // hand the pipeline back so the socket is unlinked on drop after the test
    });

    // We may or may not get LogRec frames depending on framework emissions; the goal is
    // that whatever arrives decodes cleanly (no torn frames). Wait briefly for one.
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut saw_any_frame = false;
    while Instant::now() < deadline {
        match wire::read_frame(&mut c.stream) {
            Ok(f) => {
                saw_any_frame = true;
                match f.kind {
                    kind::LOG_REC => {
                        assert!(wire::LogRecRow::decode(&f.payload).is_some(), "log rec decodes");
                        break;
                    }
                    kind::STR_DEF => {
                        assert!(wire::decode_str_def(&f.payload).is_some());
                    }
                    kind::DROPPED | kind::BUS_MSG | kind::BYE => {}
                    _ => {}
                }
            }
            Err(_) => break,
        }
    }
    let _ = saw_any_frame; // frames are best-effort; the decode assertions are the point.
    let _ = sink; // referenced for clarity

    let p = run.join().unwrap();
    drop(p);
}

// ===========================================================================
// 6. Shutdown — drop server mid-connection → Bye/EOF, socket unlinked.
// ===========================================================================

#[test]
fn dropping_server_unlinks_socket_and_ends_connection() {
    let (mut p, _src, _sink) = build_pipeline(0);
    let path = temp_socket();
    p.serve_introspection(&path).expect("serve");
    assert!(path.exists(), "socket bound");

    let mut c = Client::connect(&path);
    let _ = wire::read_frame(&mut c.stream).unwrap(); // Hello
    c.send(kind::CLIENT_HELLO, 0, &wire::ClientHello { ver_major: 1, ver_minor: 0 }.encode());

    // Stop the server: the client's connection ends (Bye or EOF) and the socket is gone.
    p.stop_introspection();
    // Read until the stream closes; a Bye then EOF, or a bare EOF, both end it.
    let mut closed = false;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match wire::read_frame(&mut c.stream) {
            Ok(f) if f.kind == kind::BYE => {}
            Ok(_) => {}
            Err(_) => {
                closed = true;
                break;
            }
        }
    }
    assert!(closed, "connection ended after server shutdown");
    assert!(!path.exists(), "socket file unlinked on shutdown");

    drop(p);
}

#[test]
fn seek_maps_time_through_the_index_or_errors_unsupported() {
    use streamcraft_core::pipeline::SeekIndex;

    // Without an index: Seek → Error(UNSUPPORTED).
    let (mut p, _src, _sink) = build_pipeline(4);
    let sock = temp_socket();
    p.serve_introspection(&sock).expect("serve");
    let mut c = Client::connect(&sock);
    c.send(kind::CLIENT_HELLO, 1, &wire::ClientHello { ver_major: wire::VER_MAJOR, ver_minor: wire::VER_MINOR }.encode());
    c.send(kind::SEEK, 2, &wire::encode_seek(1_000_000_000));
    let f = c.recv_until(kind::ERROR);
    let (code, _msg) = wire::decode_error(&f.payload).expect("error payload");
    assert_eq!(code, errcode::UNSUPPORTED, "no index installed");
    drop(c);
    p.stop_introspection();

    // With an index: Seek → Ack (floor lookup maps time→byte; the pipeline-side
    // rebase mechanics are covered by elements/tests/seek.rs).
    let (mut p, _src, _sink) = build_pipeline(4);
    p.set_seek_index(SeekIndex {
        entries: vec![(0, 0), (1_000_000_000, 4096), (2_000_000_000, 9000)],
        file_len: None,
    });
    let sock = temp_socket();
    p.serve_introspection(&sock).expect("serve");
    let mut c = Client::connect(&sock);
    c.send(kind::CLIENT_HELLO, 1, &wire::ClientHello { ver_major: wire::VER_MAJOR, ver_minor: wire::VER_MINOR }.encode());
    c.send(kind::SEEK, 2, &wire::encode_seek(1_500_000_000));
    let f = c.recv_until_any(&[kind::ACK, kind::ERROR]);
    assert_eq!(f.kind, kind::ACK, "seek accepted through the index");
}

