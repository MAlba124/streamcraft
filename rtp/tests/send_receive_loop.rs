//! The pure-streamcraft RTP loop: a **sender pipeline**
//! (`ausrc ! rtph264pay ! udpsink`, clock-paced) streaming to a **receiver
//! pipeline** (`udpsrc ! rtpsession ! rtph264depay`) over a real localhost
//! socket — both ends this crate, no external tools. Byte-exact access units
//! out, in order, one per sent AU.

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sc_rtp::elements::{RtpH264Depay, RtpH264Pay, RtpSession, RtpStreamDesc, UdpSink, UdpSrc};
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

static AU_OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];

static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &AU_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "ausrc",
    pads: &SRC_PADS,
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

/// Emits the prepared access units at 40 ms pts spacing, then EOS.
struct AuSrc {
    aus: Vec<Vec<u8>>,
    at: usize,
}

impl Element for AuSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.at = 0;
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.at >= self.aus.len() {
            return Ok(Flow::Eos);
        }
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
        let au = &self.aus[self.at];
        let n = au.len().min(buf.memory.capacity());
        buf.memory.as_mut_full()[..n].copy_from_slice(&au[..n]);
        buf.memory.set_len(n);
        buf.pts = Timestamp::from_millis(self.at as u64 * 40);
        ctx.out(PadId(0)).push(buf);
        self.at += 1;
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &AU_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "aucollect",
    pads: &SINK_PADS,
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
    got: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl Element for AuCollect {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut got = self.got.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            got.push(buf.memory.data().to_vec());
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Synthetic Annex-B access units: an SPS/PPS-ish pair on the first, then per
/// AU one small NAL and one MTU-busting NAL (forces FU-A on the wire).
fn synth_aus(n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for i in 0..n {
        let mut au = Vec::new();
        if i == 0 {
            au.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1F]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        }
        au.extend_from_slice(&[0, 0, 0, 1, 0x06, 0x05, (i & 0xFF) as u8]); // SEI-ish
        au.extend_from_slice(&[0, 0, 0, 1, if i == 0 { 0x65 } else { 0x41 }]);
        // 3000 payload bytes > MTU → FU-A; deterministic non-start-code bytes.
        au.extend((0..3000).map(|b| (((b * 7 + i * 13) % 199) + 32) as u8));
        out.push(au);
    }
    out
}

#[test]
fn pure_sc_send_receive_loop_is_byte_exact() {
    let aus = synth_aus(25); // 1 s of 25 fps
    let want = aus.clone();

    let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    // Receiver first, so its socket is armed before the sender starts.
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut rx = Pipeline::new();
    let src = rx.add(UdpSrc::new(std::net::SocketAddr::from(([127, 0, 0, 1], port))));
    let session = rx.add(RtpSession::new(vec![RtpStreamDesc { payload_type: 96, clock_rate: 90_000 }]));
    rx.link((src, "src"), (session, "sink")).expect("src -> session");
    let added = rx.preroll().expect("preroll");
    let depay = rx.add(RtpH264Depay::new(Vec::new()));
    let sink = rx.add(AuCollect { got: Arc::clone(&got) });
    rx.link((added[0].element, &added[0].name), (depay, "sink")).expect("session -> depay");
    rx.link((depay, "src"), (sink, "sink")).expect("depay -> sink");
    let rx_stop = rx.stop_handle();
    let rx_run = std::thread::spawn(move || rx.run());
    std::thread::sleep(Duration::from_millis(200)); // arm the first Recv

    // Sender: clock-paced by udpsink's wait_until — ~1 s of wall time.
    let mut tx = Pipeline::new();
    let ausrc = tx.add(AuSrc { aus, at: 0 });
    let pay = tx.add(RtpH264Pay::new(96));
    let usink = tx.add(UdpSink::new(std::net::SocketAddr::from(([127, 0, 0, 1], port))));
    tx.link((ausrc, "src"), (pay, "sink")).expect("ausrc -> pay");
    tx.link((pay, "src"), (usink, "sink")).expect("pay -> udpsink");
    tx.run().expect("sender run");

    // Sender EOSed; give the receiver the jitter latency + slack to drain.
    std::thread::sleep(Duration::from_millis(400));
    rx_stop.stop();
    rx_run.join().unwrap().expect("receiver run");

    let got = got.lock().unwrap();
    assert_eq!(got.len(), want.len(), "one access unit out per access unit in");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "AU {i} byte-exact through pay → wire → depay");
    }
}
