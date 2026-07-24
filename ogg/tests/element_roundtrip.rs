//! End-to-end element tests: `OggMux` and `OggDemux` in a real [`Pipeline`].
//!
//! The load-bearing property is **boundary-preserving round-trip**: a source emits known
//! packets as buffers → `OggMux` wraps them into an Ogg byte stream → `OggDemux` reassembles
//! them → a sink captures buffers, and buffer *i* out must equal buffer *i* in — same bytes
//! *and* same packet boundaries. Separately, `OggMux`'s output is asserted to be a valid Ogg
//! stream by parsing it with the library [`demux_all`], including that the last page carries
//! the eos flag.

use std::sync::{Arc, Mutex};

use sc_ogg::header_flags;
use sc_ogg::{demux_all, OggDemux, OggMux, OggWriter, PageHeader};
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

// =====================================================================================
// A source that emits a fixed list of byte buffers (one per "packet"), then EOS.
// =====================================================================================

static SRC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &SRC_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "packetsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active, // a real source: its own group, drives the pipeline
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Emits each buffer in `packets` in order (one `process` call each), then `Flow::Eos`.
/// A buffer larger than the pool slot would not fit; the test packets stay within a slot
/// so each is one buffer — exactly the boundary the round-trip checks.
struct PacketSrc {
    packets: Vec<Vec<u8>>,
    next: usize,
}

impl PacketSrc {
    fn new(packets: Vec<Vec<u8>>) -> Self {
        Self { packets, next: 0 }
    }
}

impl Element for PacketSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.next = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.packets.len() {
            return Ok(Flow::Eos);
        }
        let pkt = &self.packets[self.next];
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok), // pool full → try again next turn
        };
        assert!(
            pkt.len() <= buf.memory.capacity(),
            "test packet {} ({} bytes) exceeds pool slot ({}) — would split the boundary",
            self.next,
            pkt.len(),
            buf.memory.capacity()
        );
        buf.memory.as_mut_full()[..pkt.len()].copy_from_slice(pkt);
        buf.memory.set_len(pkt.len());
        ctx.out(PadId(0)).push(buf);
        self.next += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// =====================================================================================
// A sink that captures each received buffer's bytes as a Vec (preserving boundaries).
// =====================================================================================

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "packetsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active, // its own group, so buffers cross a real ring
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Records every buffer it receives, in order, as one `Vec<u8>` each — so the test can
/// compare per-buffer (per-packet) boundaries, not just concatenated bytes.
struct PacketSink {
    got: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl Element for PacketSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while let Some(buf) = inputs.pop() {
            self.got.lock().unwrap().push(buf.memory.data().to_vec());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// A sink that concatenates **all** received bytes in order — for capturing the raw Ogg
/// byte stream `OggMux` produces (as opposed to `PacketSink`, which keeps per-buffer
/// boundaries). Reuses `PacketSink`'s pad descriptor.
struct ByteSink {
    got: Arc<Mutex<Vec<u8>>>,
}

impl Element for ByteSink {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            got.extend_from_slice(buf.memory.data());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

// =====================================================================================
// Tests
// =====================================================================================

/// Run `packetsrc(packets) ! oggmux ! oggdemux ! packetsink` and return the buffers the
/// sink captured (one Vec per received buffer).
fn run_roundtrip(packets: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(PacketSrc::new(packets));
    let mux = p.add(OggMux::new());
    let demux = p.add(OggDemux::new());
    let sink = p.add(PacketSink { got: Arc::clone(&got) });
    p.link((src, "src"), (mux, "sink")).expect("src->mux");
    p.link((mux, "src"), (demux, "sink")).expect("mux->demux");
    p.link((demux, "src"), (sink, "sink")).expect("demux->sink");
    p.run().expect("run");
    let out = got.lock().unwrap().clone();
    out
}

#[test]
fn roundtrip_preserves_packets_and_boundaries() {
    // A mix: ordinary packets, a zero-length (nil) packet, and a packet that spans more
    // than one Ogg page (70_000 > one page's 65_025-byte payload) but still fits a pool
    // slot so it stays one buffer end to end.
    let packets: Vec<Vec<u8>> = vec![
        vec![1, 2, 3, 4],
        vec![],                                          // nil packet
        (0..500u32).map(|i| i as u8).collect(),          // multi-segment, one page
        vec![0xAB; 10],
        (0..70_000u32).map(|i| (i * 7) as u8).collect(), // spans >1 page via 255-lacing
        vec![9],
    ];
    let got = run_roundtrip(packets.clone());
    assert_eq!(got.len(), packets.len(), "same number of packets out as in");
    for (i, (a, b)) in packets.iter().zip(got.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} bytes and boundary preserved");
    }
}

#[test]
fn roundtrip_many_small_packets() {
    // Many small packets pack into shared pages; each must still come out as its own
    // buffer with its own bytes (boundary preserved across page packing).
    let packets: Vec<Vec<u8>> = (0..1000u32).map(|i| vec![(i & 0xff) as u8; (i % 7) as usize]).collect();
    let got = run_roundtrip(packets.clone());
    assert_eq!(got.len(), packets.len());
    for (i, (a, b)) in packets.iter().zip(got.iter()).enumerate() {
        assert_eq!(a, b, "packet {i}");
    }
}

#[test]
fn roundtrip_all_zero_length_packets() {
    // A degenerate stream of only nil packets — each must survive as an empty buffer.
    let packets: Vec<Vec<u8>> = vec![vec![], vec![], vec![]];
    let got = run_roundtrip(packets.clone());
    assert_eq!(got, packets);
}

#[test]
fn roundtrip_single_packet() {
    let packets = vec![vec![42, 43, 44]];
    let got = run_roundtrip(packets.clone());
    assert_eq!(got, packets);
}

// --- OggMux output validity ---------------------------------------------------------

/// Run `packetsrc(packets) ! oggmux(serial) ! bytesink` and return the raw Ogg bytes the
/// mux produced.
fn mux_pipeline(packets: Vec<Vec<u8>>, serial: u32) -> Vec<u8> {
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(PacketSrc::new(packets));
    let mux = p.add(OggMux::with_serial(serial));
    let sink = p.add(ByteSink { got: Arc::clone(&got) });
    p.link((src, "src"), (mux, "sink")).expect("src->mux");
    p.link((mux, "src"), (sink, "sink")).expect("mux->sink");
    p.run().expect("run");
    let out = got.lock().unwrap().clone();
    out
}

/// Walk every page in `stream`, asserting each parses and the whole stream is consumed;
/// return the header_type flags of each page in order.
fn page_flags(stream: &[u8]) -> Vec<u8> {
    let mut off = 0;
    let mut flags = Vec::new();
    while off < stream.len() {
        let page = PageHeader::parse(&stream[off..]).expect("mux emitted a valid page");
        off += page.len();
        flags.push(page.header_type());
    }
    assert_eq!(off, stream.len(), "mux output is exactly a sequence of whole pages");
    flags
}

#[test]
fn mux_pipeline_output_parses_as_valid_ogg() {
    // The bytes OggMux emits *through a real pipeline* are a valid Ogg stream: every page
    // parses (CRC included, since `PageHeader::parse` verifies it) and the library
    // `demux_all` recovers every packet with its bytes and serial.
    let packets: Vec<Vec<u8>> = vec![
        vec![1, 2, 3, 4],
        vec![],
        (0..500u32).map(|i| i as u8).collect(),
        vec![7],
    ];
    let stream = mux_pipeline(packets.clone(), 0xABCD_1234);
    let flags = page_flags(&stream);
    assert!(!flags.is_empty(), "mux produced pages");
    // First page is bos (§6, flag 0x02).
    assert_eq!(flags[0] & header_flags::BOS, header_flags::BOS, "first page bos");

    let got = demux_all(&stream);
    assert_eq!(got.len(), packets.len(), "every packet recovered");
    for (i, (p, g)) in packets.iter().zip(got.iter()).enumerate() {
        assert_eq!(&g.data, p, "packet {i} bytes");
        assert_eq!(g.serial, 0xABCD_1234, "packet {i} serial stamped by mux");
    }
    assert!(got.first().unwrap().bos, "first packet tagged bos");
}

/// The **eos-terminated bytes** contract of `OggMux`'s finish path.
///
/// The muxer's `event(Eos)`/`stop()` flush the terminating eos page via
/// `OggWriter::finish` — but today's pipeline does not route that end-of-stream output
/// downstream (see the element's module docs), so it cannot be observed at a pipeline
/// sink yet. This test drives the **exact byte sequence the element performs**
/// (`write_packet` + `flush` per packet, then `finish`) through the same `OggWriter` and
/// asserts the resulting stream is eos-terminated: the last page carries the eos flag
/// (§6, flag 0x04) and the library tags the final packet `eos`.
///
/// Because every packet's page is flushed during `process`, `finish` finds no pending
/// page and appends a **nil eos page** — a documented, well-formed consequence of the
/// flush-per-packet strategy (`spec/NOTES.md`, "Nil eos page on an empty/flushed
/// stream"): the demuxer reads it back as one trailing zero-length packet. So the
/// eos-terminated stream carries the N input packets **plus** a trailing empty eos packet.
/// (This trailing packet is *not* seen by today's boundary-preserving round-trip because
/// the eos page is dropped, not routed; it will appear once the core routes EOS output,
/// at which point the muxer can drop the redundant per-packet flush and let `finish`
/// terminate the last real packet's page directly.)
#[test]
fn mux_finish_contract_terminates_with_eos() {
    let packets: Vec<&[u8]> = vec![&[1, 2, 3, 4], &[], &[9; 500], &[7]];
    let serial = 1234;

    // Mirror OggMux::process (write_packet + flush per packet) then finish_stream (finish).
    let mut w = OggWriter::new(serial);
    let mut stream = Vec::new();
    for &pkt in &packets {
        w.write_packet(&mut stream, pkt, 0).expect("write");
        w.flush(&mut stream); // per-packet page flush, exactly like the element
    }
    w.finish(&mut stream); // the eos page the element flushes at end-of-stream

    // Valid stream: the N input packets round-trip with the mux's serial, plus a trailing
    // nil eos packet (the documented finish semantics above).
    let got = demux_all(&stream);
    assert_eq!(got.len(), packets.len() + 1, "N packets + trailing nil eos packet");
    for (i, (p, g)) in packets.iter().zip(got.iter()).enumerate() {
        assert_eq!(&g.data, *p, "packet {i}");
        assert_eq!(g.serial, serial);
    }
    let trailing = got.last().unwrap();
    assert!(trailing.data.is_empty(), "trailing eos packet is nil");
    // Well-terminated: the last page has the eos flag and the last (nil) packet is eos.
    let flags = page_flags(&stream);
    assert_eq!(
        flags.last().copied().unwrap() & header_flags::EOS,
        header_flags::EOS,
        "last page carries eos"
    );
    assert!(trailing.eos, "final packet tagged eos");
    assert!(got.first().unwrap().bos, "first packet tagged bos");
}
