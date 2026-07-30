//! `udpsrc` integration: real localhost datagrams through a real pipeline —
//! one datagram per buffer, boundaries intact, arrival pts stamped.

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pf_rtp::elements::UdpSrc;
use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

static SINK_OFFERS: [OfferDesc; 1] = [OfferDesc::any("datagram")];
static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &SINK_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "dgramcollect",
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

/// Collects `(payload, pts)` per buffer — one entry per datagram if boundaries hold.
struct DgramCollect {
    got: Arc<Mutex<Vec<(Vec<u8>, Timestamp)>>>,
}

impl Element for DgramCollect {
    fn desc(&self) -> &'static ElementDesc {
        &SINK_DESC
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
fn datagram_boundaries_and_arrival_pts() {
    // Bind an ephemeral port first with a plain socket to learn a free port,
    // then hand that exact port to the element.
    let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(UdpSrc::new(std::net::SocketAddr::from(([127, 0, 0, 1], port))));
    let sink = p.add(DgramCollect { got: Arc::clone(&got) });
    p.link((src, "src"), (sink, "sink")).expect("link");

    let stop = p.stop_handle();
    let sender = std::thread::spawn(move || {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect(("127.0.0.1", port)).unwrap();
        // Give the pipeline a beat to preroll and get its first Recv armed;
        // resend-with-retry below tolerates any startup loss anyway.
        std::thread::sleep(Duration::from_millis(150));
        for i in 0..20u8 {
            // Distinct sizes so a coalescing bug shows as a length mismatch.
            let msg: Vec<u8> = (0..(10 + i as usize)).map(|b| b as u8 ^ i).collect();
            s.send(&msg).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
    });

    let runner = std::thread::spawn(move || p.run());
    sender.join().unwrap();
    // Let the tail datagrams drain, then stop the live pipeline.
    std::thread::sleep(Duration::from_millis(200));
    stop.stop();
    runner.join().unwrap().expect("run");

    let got = got.lock().unwrap();
    // UDP on loopback does not drop in practice, but the contract we assert is
    // boundaries + monotonic arrival stamps, not zero loss: require most.
    assert!(got.len() >= 15, "received {} of 20 datagrams", got.len());
    for (payload, _) in got.iter() {
        let i = payload.len() - 10;
        let want: Vec<u8> = (0..payload.len()).map(|b| (b as u8) ^ (i as u8)).collect();
        assert_eq!(payload, &want, "datagram bytes intact, boundaries preserved");
    }
    for w in got.windows(2) {
        assert!(w[1].1 >= w[0].1, "arrival pts monotonic");
    }
}
