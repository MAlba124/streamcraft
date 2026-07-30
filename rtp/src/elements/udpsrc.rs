//! `udpsrc` — receives UDP datagrams into pooled buffers, one datagram per
//! buffer (spec: IO; the pf-http source is the pattern).
//!
//! `start()` binds (and optionally connects) a [`std::net::UdpSocket`]
//! synchronously, then hands the fd to the reactor exactly like `httpsrc`
//! hands its TCP socket: `into_raw_fd` → `File::from_raw_fd` →
//! `ctx.io().register`. `process()` drains completed
//! [`OpKind::Recv`](profluens_core::io::OpKind::Recv) ops — **`read(2)` on a
//! UDP socket returns exactly one datagram per call** (truncating an oversized
//! one, which pool slots ≫ MTU make moot), so each completion is pushed
//! downstream as-is, boundaries intact. The ride is single-op like httpsrc's:
//! one `Recv` outstanding, resubmitted per pass.
//!
//! Datagram boundaries are load-bearing (RTP has no framing of its own —
//! RFC 3550 §11: one packet per UDP datagram), so the src pad offers a
//! dedicated **`datagram`** family rather than `bytes`: a byte-stream consumer
//! (a demuxer, a file sink) must not accidentally link to a packet source.
//!
//! Each buffer's `pts` is stamped with the **arrival running time**
//! (`ctx.now()` at completion drain) — the jitter buffer downstream keys its
//! latency wait off this, and no other element on the path rewrites it.
//!
//! Live source: no EOS — the element runs until the pipeline stops it
//! (`LatencyDesc.is_live = true`; sinks schedule with the path's declared
//! latency budget).

use std::net::{SocketAddr, UdpSocket};
use std::os::unix::io::{FromRawFd, IntoRawFd};

use profluens_core::batch::Inputs;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::io::{FileHandle, IoResult};
use profluens_core::time::Timestamp;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("datagram")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "udpsrc",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: true,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// Where the socket comes from: bound at `start()`, or handed in pre-bound
/// (an RTSP client negotiated its port in SETUP — rebinding would race).
enum Sock {
    Bind(SocketAddr),
    Ready(Option<UdpSocket>),
}

/// One datagram per pooled buffer from a bound UDP socket.
pub struct UdpSrc {
    sock: Sock,
    /// Optional peer to `connect(2)` to — kernel-side source filtering (only
    /// datagrams from this address are delivered; RFC 3550 §11's transport
    /// address pairing).
    peer: Option<SocketAddr>,
    file: Option<FileHandle>,
    read_in_flight: bool,
    /// Monotonic completion tag (httpsrc's pattern; no flush path yet, the
    /// floor exists so one can be added without a protocol change).
    seq: u64,
    /// For error messages after the socket is consumed into the reactor.
    label: String,
}

impl UdpSrc {
    /// A source bound to `bind` (typed construction — resolution is the
    /// caller's concern; an SDP/RTSP layer knows real addresses already).
    // COLD: the diagnostic `label` is built once at construction, never per packet.
    #[allow(clippy::disallowed_methods)]
    pub fn new(bind: SocketAddr) -> UdpSrc {
        UdpSrc {
            sock: Sock::Bind(bind),
            peer: None,
            file: None,
            read_in_flight: false,
            seq: 0,
            label: bind.to_string(),
        }
    }

    /// A source over an already-bound socket — the RTSP path: the port was
    /// negotiated in SETUP against this exact socket, so it must be used, not
    /// re-bound.
    // COLD: the diagnostic `label` is built once at construction, never per packet.
    #[allow(clippy::disallowed_methods)]
    pub fn from_socket(sock: UdpSocket) -> UdpSrc {
        let label = sock.local_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".into());
        UdpSrc {
            sock: Sock::Ready(Some(sock)),
            peer: None,
            file: None,
            read_in_flight: false,
            seq: 0,
            label,
        }
    }

    /// Additionally `connect(2)` the socket to `peer` so the kernel drops
    /// datagrams from other sources.
    pub fn with_peer(mut self, peer: SocketAddr) -> UdpSrc {
        self.peer = Some(peer);
        self
    }

    /// Keep one receive outstanding (single-op streaming, see module docs).
    fn submit_read(&mut self, ctx: &mut Ctx) {
        if self.read_in_flight {
            return;
        }
        let Some(file) = self.file else { return };
        // Pool dry = backpressure: retry next pass (live data meanwhile queues
        // in the socket's kernel buffer; overflow there is UDP loss, which the
        // jitter buffer accounts for — the correct live-source degradation).
        let Some(buf) = ctx.try_alloc(PadId(0)) else { return };
        let user = self.seq;
        self.seq += 1;
        ctx.io().submit_recv(file, buf, user);
        self.read_in_flight = true;
    }
}

impl Element for UdpSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let sock = match &mut self.sock {
            Sock::Bind(addr) => UdpSocket::bind(*addr)
                .map_err(|e| Error::Resource(format!("udpsrc: bind {addr}: {e}")))?,
            Sock::Ready(s) => s.take().ok_or_else(|| {
                Error::Resource("udpsrc: pre-bound socket already consumed (restart)".into())
            })?,
        };
        if let Some(peer) = self.peer {
            sock.connect(peer)
                .map_err(|e| Error::Resource(format!("udpsrc: connect {peer}: {e}")))?;
        }
        // A receive timeout so a blocked `read(2)` in the sync reactor wakes
        // periodically: `StopHandle` is cooperative, and a live source whose
        // sender went quiet would otherwise wedge its group in the kernel
        // forever (the framework's hard-interrupt reactor cancel is a named
        // follow-up; until then the timeout makes stop observable ≤100 ms).
        // The timed-out read completes as `Err(WouldBlock)` and is resubmitted.
        sock.set_read_timeout(Some(std::time::Duration::from_millis(100)))
            .map_err(|e| Error::Resource(format!("udpsrc: set_read_timeout: {e}")))?;
        // Hand the socket to the reactor (fd ownership transfers to the
        // registered File — the httpsrc/filesrc registration path).
        // SAFETY: the fd came from `into_raw_fd`, which transferred ownership.
        let file = unsafe { std::fs::File::from_raw_fd(sock.into_raw_fd()) };
        self.file = Some(ctx.io().register(file));
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Drain completed receives: each is one whole datagram, stamped with
        // its arrival running time and pushed as-is.
        while let Some(c) = ctx.io().next_completion() {
            self.read_in_flight = false;
            match c.result {
                // 0-byte datagrams are legal UDP but meaningless here; recycle.
                IoResult::Ok(0) | IoResult::Cancelled => {}
                IoResult::Ok(_) => {
                    let mut buf = c.buf;
                    buf.pts = ctx.now();
                    ctx.out(PadId(0)).push(buf);
                }
                // The 100 ms receive timeout firing on a quiet socket (see
                // `start`) — not an error; resubmit below.
                IoResult::Err(std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                IoResult::Err(k) => {
                    return Err(Error::Resource(format!(
                        "udpsrc: recv on {}: {k:?}",
                        self.label
                    )));
                }
            }
        }
        self.submit_read(ctx);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.file = None;
        self.read_in_flight = false;
    }
}
