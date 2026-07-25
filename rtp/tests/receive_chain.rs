//! The full receive chain over a real socket: the checked-in ffmpeg RTP dump
//! (see `payload.rs` for its provenance) replayed through
//! `udpsrc ! rtpsession ! rtph264depay`, asserting the depayloaded Annex-B
//! stream is **byte-identical** to the encoder's own `.h264` output —
//! normalized to 4-byte start codes, since RFC 6184 transports NAL units, not
//! start-code framing.

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sc_rtp::elements::{RtpH264Depay, RtpSession, RtpStreamDesc, UdpSrc};
use sc_rtp::packet::RtpPacket;
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;

const RTPDUMP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/h264_testsrc.rtpdump");
const H264: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/h264_testsrc.h264");

/// The fixture's framing (documented in `payload.rs`): u32 LE length-prefixed
/// datagrams, concatenated.
fn read_datagrams(path: &str) -> Vec<Vec<u8>> {
    let raw = std::fs::read(path).expect("fixture present");
    let mut out = Vec::new();
    let mut at = 0;
    while at + 4 <= raw.len() {
        let len = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        out.push(raw[at..at + len].to_vec());
        at += len;
    }
    out
}

/// Normalize an Annex-B stream to 4-byte start codes (the mixed 3-/4-byte
/// forms both mark NAL unit boundaries; only the units are comparable).
fn normalize_annexb(stream: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(stream.len() + 64);
    let mut i = 0;
    let mut starts = Vec::new();
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    for (k, &s) in starts.iter().enumerate() {
        let begin = s + 3;
        let mut end = starts.get(k + 1).copied().unwrap_or(stream.len());
        if k + 1 < starts.len() && end > begin && stream[end - 1] == 0 {
            end -= 1; // the next code's 4-byte zero_byte
        }
        if end > begin {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&stream[begin..end]);
        }
    }
    out
}

static AU_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];
static AU_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &AU_OFFERS,
    dynamic: false,
    validate: None,
}];
static AU_DESC: ElementDesc = ElementDesc {
    name: "aucollect",
    pads: &AU_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

struct AuCollect {
    got: Arc<Mutex<Vec<(Vec<u8>, Timestamp)>>>,
}

impl Element for AuCollect {
    fn desc(&self) -> &'static ElementDesc {
        &AU_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            got.push((buf.memory.data().to_vec(), buf.pts));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

#[test]
fn fixture_replay_reconstructs_the_encoder_stream() {
    let datagrams = read_datagrams(RTPDUMP);
    assert!(!datagrams.is_empty(), "fixture parses");
    // The stream's payload type, read off the wire rather than assumed.
    let pt = RtpPacket::parse(&datagrams[0]).expect("fixture is RTP").payload_type();

    let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(UdpSrc::new(std::net::SocketAddr::from(([127, 0, 0, 1], port))));
    let session = p.add(RtpSession::new(vec![RtpStreamDesc {
        payload_type: pt,
        clock_rate: 90_000, // RFC 6184 §8.1: H.264's fixed RTP clock
    }]));
    p.link((src, "src"), (session, "sink")).expect("src -> session");
    let added = p.preroll().expect("preroll");
    assert_eq!(added.len(), 1, "one stream pad from the session");

    let depay = p.add(RtpH264Depay::new(Vec::new()));
    let sink = p.add(AuCollect { got: Arc::clone(&got) });
    p.link((added[0].element, &added[0].name), (depay, "sink")).expect("session -> depay");
    p.link((depay, "src"), (sink, "sink")).expect("depay -> sink");

    let stop = p.stop_handle();
    let sender = std::thread::spawn(move || {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect(("127.0.0.1", port)).unwrap();
        std::thread::sleep(Duration::from_millis(200)); // let the first Recv arm
        for d in &datagrams {
            s.send(d).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
    });

    let runner = std::thread::spawn(move || p.run());
    sender.join().unwrap();
    std::thread::sleep(Duration::from_millis(400)); // drain the tail
    stop.stop();
    runner.join().unwrap().expect("run");

    let got = got.lock().unwrap();
    assert!(!got.is_empty(), "access units decoded");
    for w in got.windows(2) {
        assert!(w[1].1 >= w[0].1, "AU pts monotonic (session clock mapping)");
    }

    // Byte-exactness against the encoder's own Annex-B output. Loopback UDP
    // does not lose packets in practice; if this ever flakes on a loaded
    // machine, the diagnosis is real loss (the depay resync path), not a bug.
    let ours: Vec<u8> = got.iter().flat_map(|(au, _)| au.iter().copied()).collect();
    let reference = normalize_annexb(&std::fs::read(H264).expect("h264 fixture"));
    assert_eq!(
        ours.len(),
        reference.len(),
        "depayloaded stream length matches the encoder stream"
    );
    assert_eq!(ours, reference, "depayloaded bytes identical to the encoder stream");
}
