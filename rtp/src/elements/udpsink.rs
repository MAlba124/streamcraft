//! `udpsink` — sends each buffer as one UDP datagram, **paced on the pipeline
//! clock**: the sink waits until running time reaches the buffer's `pts`
//! before sending (`ctx.wait_until`, the same discipline as a render sink).
//! That pacing is what turns a file-fed pipeline into a realtime sender — the
//! network is the "device", and backpressure through the pipeline follows the
//! clock, exactly like ffmpeg's `-re` but by construction.
//!
//! Sending uses the plain blocking `send(2)`: on a connected UDP socket it
//! does not block in any meaningful way (no flow control to wait on — a full
//! socket buffer drops, which *is* UDP's contract; RFC 3550 leaves loss to
//! RTCP accounting).

use std::net::{SocketAddr, UdpSocket};

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::time::Timestamp;

/// Accepts RTP packets or raw datagrams — either way, one buffer = one send.
static OFFERS: [OfferDesc; 2] = [OfferDesc::any("rtp"), OfferDesc::any("datagram")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "udpsink",
    pads: &PADS,
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

/// One datagram per buffer to a fixed peer, clock-paced on `pts`.
pub struct UdpSink {
    peer: SocketAddr,
    sock: Option<UdpSocket>,
}

impl UdpSink {
    /// A sink sending to `peer` (an RTSP server app got this from SETUP's
    /// `client_port` + the connection's address).
    pub fn new(peer: SocketAddr) -> UdpSink {
        UdpSink { peer, sock: None }
    }
}

impl Element for UdpSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        let sock = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| Error::Resource(format!("udpsink: bind: {e}")))?;
        sock.connect(self.peer)
            .map_err(|e| Error::Resource(format!("udpsink: connect {}: {e}", self.peer)))?;
        self.sock = Some(sock);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let Some(sock) = &self.sock else { return Ok(Flow::Ok) };
        while let Some(buf) = inputs.pop() {
            // Pace on the clock (a pts-less buffer sends immediately —
            // wait_until returns at once for NONE).
            let _ = ctx.wait_until(buf.pts);
            // A send error on UDP is local (unreachable peer, buffer full):
            // count-and-continue is the live contract, not a pipeline error.
            let _ = sock.send(buf.memory.data());
        }
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.sock = None;
    }
}
