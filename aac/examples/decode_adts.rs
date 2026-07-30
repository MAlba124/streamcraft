//! Decode an ADTS AAC file through `AacDec` to raw s16le — the direct-library
//! probe for real-world files (the `pf-h265` `decode_annexb` precedent: adoption
//! gates must include a real file, not just fixtures). Strips the ADTS framing
//! (ISO/IEC 14496-3 §1.A.2.2) into the raw AUs + synthesized ASC our demuxers
//! deliver, so it exercises exactly the container-shaped element contract.
//!
//! ```text
//! ffmpeg -i movie.mkv -map 0:a -c copy -f adts audio.adts
//! cargo run --release -p pf-aac --example decode_adts -- audio.adts out.s16le [max_aus]
//! ffmpeg -i audio.adts -f s16le ref.s16le   # then compare
//! ```

use std::io::Write;
use std::sync::{Arc, Mutex};

use pf_aac::AacDec;
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

fn adts_to_aus(data: &[u8], max: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut aus = Vec::new();
    let mut asc = Vec::new();
    let mut at = 0usize;
    while at + 7 <= data.len() && aus.len() < max {
        if data[at] != 0xFF || data[at + 1] & 0xF0 != 0xF0 {
            at += 1; // resync (junk between frames)
            continue;
        }
        let protection_absent = data[at + 1] & 1;
        let profile = data[at + 2] >> 6;
        let sf_index = (data[at + 2] >> 2) & 0xF;
        let chan_cfg = ((data[at + 2] & 1) << 2) | (data[at + 3] >> 6);
        let frame_len = ((data[at + 3] as usize & 0x3) << 11)
            | ((data[at + 4] as usize) << 3)
            | (data[at + 5] as usize >> 5);
        if frame_len < 7 || at + frame_len > data.len() {
            break;
        }
        let hdr = if protection_absent == 1 { 7 } else { 9 };
        if asc.is_empty() {
            let bits: u16 = (u16::from(profile + 1) << 11)
                | (u16::from(sf_index) << 7)
                | (u16::from(chan_cfg) << 3);
            asc = bits.to_be_bytes().to_vec();
        }
        aus.push(data[at + hdr..at + frame_len].to_vec());
        at += frame_len;
    }
    (asc, aus)
}

static BYTES_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static SRC_PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &BYTES_OFFERS,
    dynamic: false,
    validate: None,
}];
static SRC_DESC: ElementDesc = ElementDesc {
    name: "packetsrc",
    pads: &SRC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PacketSrc {
    packets: Vec<Vec<u8>>,
    next: usize,
}

impl Element for PacketSrc {
    fn desc(&self) -> &'static ElementDesc {
        &SRC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.next >= self.packets.len() {
            return Ok(Flow::Eos);
        }
        let pkt = &self.packets[self.next];
        let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
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

static SINK_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &BYTES_OFFERS,
    dynamic: false,
    validate: None,
}];
static SINK_DESC: ElementDesc = ElementDesc {
    name: "pcmrecordsink",
    pads: &SINK_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc { min: Timestamp::ZERO, max: Timestamp::ZERO, is_live: false, jitter: Timestamp::ZERO },
    make_default: None,
};

struct PcmRecordSink {
    got: Arc<Mutex<Vec<u8>>>,
}

impl Element for PcmRecordSink {
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(inp), Some(outp)) = (args.first(), args.get(1)) else {
        eprintln!("usage: decode_adts IN.adts OUT.s16le [max_aus]");
        std::process::exit(2);
    };
    let max: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let data = std::fs::read(inp).expect("read input");
    let (asc, aus) = adts_to_aus(&data, max);
    eprintln!("{} AUs, ASC {:02x?}", aus.len(), asc);
    let mut packets = vec![asc];
    packets.extend(aus);

    let got = Arc::new(Mutex::new(Vec::new()));
    let mut p = Pipeline::new();
    let src = p.add(PacketSrc { packets, next: 0 });
    let dec = p.add(AacDec::new());
    let sink = p.add(PcmRecordSink { got: Arc::clone(&got) });
    p.link((src, "src"), (dec, "sink")).expect("src ! aacdec");
    p.link((dec, "src"), (sink, "sink")).expect("aacdec ! sink");
    p.run().expect("run");
    while let Some(msg) = p.bus().try_recv() {
        if let profluens_core::bus::BusMessage::Warning { error, .. } = msg {
            eprintln!("warning: {error:?}");
        }
    }
    let bytes = got.lock().unwrap();
    std::fs::File::create(outp).and_then(|mut f| f.write_all(&bytes)).expect("write output");
    eprintln!("wrote {} bytes of s16le", bytes.len());
}
