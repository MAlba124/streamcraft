// temporary probe — decode a fixture via the element at two chunk sizes, dump stats.
use std::sync::{Arc, Mutex};

use pf_mp3::Mp3Dec;
use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

static BYTES_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc { name: "src", direction: Direction::Src, offers: &BYTES_OFFERS, dynamic: false, validate: None }];
static SRC_DESC: ElementDesc = ElementDesc { name: "packetsrc", pads: &SRC_PADS, props: &[], sched: SchedHint::Active, inputs: InputPolicy::None, latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO }, make_default: None };
struct PacketSrc { packets: Vec<Vec<u8>>, next: usize }
impl Element for PacketSrc {
    fn desc(&self) -> &'static ElementDesc { &SRC_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { self.next = 0; Ok(()) }
    fn process(&mut self, ctx: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
        loop {
            if self.next >= self.packets.len() { return Ok(Flow::Eos); }
            let pkt = &self.packets[self.next];
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            buf.memory.as_mut_full()[..pkt.len()].copy_from_slice(pkt);
            buf.memory.set_len(pkt.len());
            ctx.out(PadId(0)).push(buf);
            self.next += 1;
        }
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}
static SINK_PADS: [PadDesc; 1] = [PadDesc { name: "sink", direction: Direction::Sink, offers: &BYTES_OFFERS, dynamic: false, validate: None }];
static SINK_DESC: ElementDesc = ElementDesc { name: "sink", pads: &SINK_PADS, props: &[], sched: SchedHint::Active, inputs: InputPolicy::Single, latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO }, make_default: None };
struct Sink { got: Arc<Mutex<Vec<u8>>> }
impl Element for Sink {
    fn desc(&self) -> &'static ElementDesc { &SINK_DESC }
    fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> { Ok(()) }
    fn process(&mut self, _c: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut g = self.got.lock().unwrap();
        while let Some(b) = inputs.pop() { g.extend_from_slice(b.memory.data()); }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> { Ok(()) }
    fn stop(&mut self, _c: &mut Ctx) {}
}

fn decode(mp3: &[u8], chunk: usize) -> Vec<i16> {
    let packets: Vec<Vec<u8>> = mp3.chunks(chunk).map(<[u8]>::to_vec).collect();
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    p.set_pool(4096, 512);
    let src = p.add(PacketSrc { packets, next: 0 });
    let dec = p.add(Mp3Dec::new());
    let sink = p.add(Sink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).unwrap();
    p.link((dec, "src"), (sink, "sink")).unwrap();
    p.run().unwrap();
    let b = got.lock().unwrap().clone();
    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn stats(tag: &str, v: &[i16]) {
    let first_nz = v.iter().position(|&s| s != 0).unwrap_or(v.len());
    let last_nz = v.iter().rposition(|&s| s != 0).map(|i| i+1).unwrap_or(0);
    let energy: f64 = v.iter().map(|&s| f64::from(s)*f64::from(s)).sum::<f64>() / v.len().max(1) as f64;
    println!("{tag}: len={} first_nz={} last_nz={} rms={:.1} head={:?}", v.len(), first_nz, last_nz, energy.sqrt(), &v[..v.len().min(8)]);
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let js = std::fs::read(format!("{dir}/js_128.mp3")).unwrap();
    eprintln!("=== STEREO DECODE chunk=1000 ({} bytes) ===", js.len());
    let s = decode(&js, 1000);
    stats("js chunk=1000  ", &s);
}
