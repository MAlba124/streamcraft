//! The ClientNode real-time audio path: export a `client-node`, negotiate a `Format` and
//! `Buffers`, map the shared-memory buffers, and drive the graph off the activation eventfd —
//! all over the native protocol, no libpipewire (see `REFERENCES.md` → ClientNode; layouts from
//! PipeWire 1.6 `module-client-node/protocol-native.c` + `remote-node.c` + `private.h`).
//!
//! # How it runs
//!
//! [`play`] owns a [`Connection`] and runs a single loop that `poll(2)`s both the control socket
//! and the node's **readfd** eventfd. Control messages (format negotiation, buffer setup, node
//! commands) are dispatched as they arrive; when the graph driver writes our eventfd we run one
//! **process cycle**: pull a quantum of PCM from the caller's `fill` callback into a shared
//! buffer, publish it through the port's `spa_io_buffers`, then trigger the downstream (sink)
//! node by decrementing its activation and waking its eventfd. This mirrors what libpipewire's
//! data loop does for a `pw_stream`, minus the C library.
//!
//! The loop is synchronous and single-threaded — the caller runs it on a dedicated thread (as
//! `PipeWireAudioSink` does), pacing the pipeline by what `fill` returns. Moving the socket onto
//! the reactor and the process cycle onto a real RT thread are follow-ups (see [[reactor-io-rule]]).
//!
//! # Status (validated against the live 1.6.5 daemon)
//!
//! Everything up to and including `Command Start` works and is verified against the running
//! daemon: `Core.CreateObject` a `client-node`, `Update`/`PortUpdate` with a node- and port-level
//! `EnumFormat`, format fixation (`port_set_param(Format)`), cross-process `Buffers` negotiation
//! (`dataType` MemFd), `AddMem` memfd mapping, `PortUseBuffers` + `PortSetIO(Buffers)`, the
//! `Transport`/`SetActivation` records, and the node joining the running graph as a follower of
//! the real sink driver (`pw-top` shows it under the device, `driver_id` = the sink).
//!
//! **Remaining before it emits sound:** the per-cycle driver→client RT trigger. Our activation
//! sits at `status=FINISHED`, `required=1`, `pending=0`, but the sink driver never writes our
//! eventfd — that per-cycle "prepare + trigger the follower" step is only wired up by the
//! session manager's *managed* linking. A bare client-node is not adapter-wrapped, so the
//! session manager will not autoconnect it, and a hand-made `pw-link` creates the link without
//! the driver-schedule integration. Closing the gap means one of: (a) presenting an
//! **audioconvert adapter** (interleaved→per-channel) the way `pw_stream` does, so the session
//! manager wraps and links us; or (b) presenting **DSP mono ports** the session manager's
//! pro-audio policy links 1:1 to the device — either way letting *managed* linking set up the
//! driver's follower schedule. The DSP-format path here ([`AudioConfig::dsp`]) gets furthest.

#![allow(unsafe_code)] // mmap'd shared memory + libc socket/eventfd/poll syscalls
#![allow(clippy::disallowed_methods)] // control-path setup + device thread, not a frame loop

use std::collections::HashMap;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

use crate::native::conn::{Connection, CLIENT_ID, CORE_ID};
use crate::native::proto::{
    self, PwError, CLIENT_METHOD_UPDATE_PROPERTIES, CORE_METHOD_CREATE_OBJECT, CORE_METHOD_HELLO,
    CORE_METHOD_PONG,
};
use crate::native::spa::{self, SampleFormat};

// --- ClientNode interface (a client-allocated proxy id) ------------------------------------

const CLIENT_NODE_VERSION: i32 = 6;
const CN_METHOD_UPDATE: u8 = 2;
const CN_METHOD_PORT_UPDATE: u8 = 3;
const CN_METHOD_SET_ACTIVE: u8 = 4;

const CN_EVENT_TRANSPORT: u8 = 0;
const CN_EVENT_SET_PARAM: u8 = 1;
const CN_EVENT_COMMAND: u8 = 4;
const CN_EVENT_PORT_SET_PARAM: u8 = 7;
const CN_EVENT_PORT_USE_BUFFERS: u8 = 8;
const CN_EVENT_PORT_SET_IO: u8 = 9;
const CN_EVENT_SET_ACTIVATION: u8 = 10;

const CORE_EVENT_ADD_MEM: u8 = 6;
const CORE_EVENT_ERROR: u8 = 3;
const CORE_EVENT_PING: u8 = 2;
const CORE_EVENT_BOUND_ID: u8 = 5;
const CORE_EVENT_BOUND_PROPS: u8 = 8;

/// `client_version` field offset in `pw_node_activation` (set on Transport, per remote-node.c).
const ACT_CLIENT_VERSION: usize = 540;

/// Playback parameters, known up front from the negotiated `audio/raw` format.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub rate: u32,
    pub channels: u32,
    pub format: SampleFormat,
    /// Desired buffer size in frames (~one quantum); bounds latency.
    pub quantum: u32,
    pub app_name: String,
    pub node_name: String,
    /// Present a single mono `DSP_F32` output port instead of an interleaved `audio/raw` port.
    /// A DSP port links 1:1 to an audio device's per-channel port with no adapter — the only
    /// path that plays through the native client today (`channels` must be 1). Interleaved
    /// multi-channel needs the session manager's audioconvert adapter, which we do not provide.
    pub dsp: bool,
}

fn dbg_enabled() -> bool {
    std::env::var_os("STREAMCRAFT_PW_DEBUG").is_some()
}
macro_rules! trace {
    ($($a:tt)*) => { if dbg_enabled() { eprintln!("[pw-native] {}", format!($($a)*)); } };
}

// --- shared-memory mapping -----------------------------------------------------------------

struct Mapping {
    base: *mut libc::c_void,
    len: usize,
}

/// One mapped audio buffer: where to write PCM and where its `spa_chunk` descriptor lives.
struct Buf {
    data: *mut u8,
    maxsize: usize,
    chunk: *mut u8,
}

/// A downstream node we must wake after producing: its activation record + signal eventfd.
struct Peer {
    node: u32,
    act: *mut u8,
    signalfd: RawFd,
}

/// The client-node playback session: connection + negotiated + RT state. `F` is the caller's
/// PCM source (fills a slice of whole interleaved frames, returns bytes written).
struct Node<F> {
    conn: Connection,
    cfg: AudioConfig,
    stride: usize,
    node_id: u32,
    /// Our node's global id (from `Core.BoundProps`) — used to never trigger ourselves.
    global_id: u32,
    /// memfds from `Core.AddMem`, by mem id (mapped on demand).
    mems: HashMap<u32, RawFd>,
    mappings: Vec<Mapping>,
    // real-time hand-off state, filled in as setup events arrive
    self_act: *mut u8,
    readfd: RawFd,
    writefd: RawFd,
    io_buffers: *mut u8,
    buffers: Vec<Buf>,
    peers: Vec<Peer>,
    next_buf: usize,
    running: bool,
    format_configured: bool,
    cycles: u64,
    idle_ticks: u64,
    error: Option<PwError>,
    fill: F,
}

// SeqCst matches libpipewire's `SPA_ATOMIC_*` (`__ATOMIC_SEQ_CST`) on the shared records.
#[inline]
unsafe fn a_u32(p: *mut u8, off: usize) -> &'static AtomicU32 {
    &*(p.add(off) as *const AtomicU32)
}
#[inline]
unsafe fn a_i32(p: *mut u8, off: usize) -> &'static AtomicI32 {
    &*(p.add(off) as *const AtomicI32)
}
#[inline]
unsafe fn read_u64(p: *mut u8, off: usize) -> u64 {
    (p.add(off) as *const u64).read_unaligned()
}
#[inline]
unsafe fn write_u32(p: *mut u8, off: usize, v: u32) {
    (p.add(off) as *mut u32).write(v);
}

impl<F: FnMut(&mut [u8]) -> usize> Node<F> {
    /// Map `[offset, offset+size)` of mem `mem_id` (page-aligning the mmap under the hood).
    fn map(&mut self, mem_id: u32, offset: u32, size: u32) -> Option<*mut u8> {
        let fd = *self.mems.get(&mem_id)?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let start = (offset as usize) & !(page - 1);
        let delta = offset as usize - start;
        let len = delta + size as usize;
        // SAFETY: fd is an AddMem memfd; MAP_SHARED at a page-aligned offset for `len` bytes.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                start as libc::off_t,
            )
        };
        if base == libc::MAP_FAILED {
            trace!("mmap mem {mem_id} failed: {}", std::io::Error::last_os_error());
            return None;
        }
        self.mappings.push(Mapping { base, len });
        Some(unsafe { (base as *mut u8).add(delta) })
    }

    // -- setup: create the node, declare ports/format, activate --

    fn hello(&mut self) -> Result<(), PwError> {
        self.conn.send(CORE_ID, CORE_METHOD_HELLO, |b| {
            b.int(proto::CORE_VERSION);
        });
        let app = self.cfg.app_name.clone();
        self.conn.send(CLIENT_ID, CLIENT_METHOD_UPDATE_PROPERTIES, |b| {
            b.dict(&[("application.name", app.as_str())]);
        });
        Ok(())
    }

    fn create_node(&mut self) {
        self.node_id = self.conn.alloc_id();
        let name = self.cfg.node_name.clone();
        let app = self.cfg.app_name.clone();
        // node.latency / node.rate give the session manager a defined quantum so it can
        // configure the graph and link the stream (a bare client-node without them is created
        // but never linked); values mirror what `pw_stream` sets.
        let latency = format!("{}/{}", self.cfg.quantum, self.cfg.rate);
        let rate = format!("1/{}", self.cfg.rate);
        let id = self.node_id;
        self.conn.send(CORE_ID, CORE_METHOD_CREATE_OBJECT, |b| {
            b.string("client-node");
            b.string("PipeWire:Interface:ClientNode");
            b.int(CLIENT_NODE_VERSION);
            b.dict(&[
                // media.class is what the session manager keys on to autoconnect a playback
                // stream to the default sink (without it the node is created but never linked).
                ("media.class", "Stream/Output/Audio"),
                ("media.type", "Audio"),
                ("media.category", "Playback"),
                ("media.role", "Music"),
                ("node.name", name.as_str()),
                ("node.description", name.as_str()),
                ("media.name", name.as_str()),
                ("application.name", app.as_str()),
                ("node.autoconnect", "true"),
                ("node.want-driver", "true"),
                // Keep the graph (and the sink driver) running so our eventfd is scheduled —
                // otherwise a linked-but-idle sink stays suspended and never triggers us.
                ("node.always-process", "true"),
                ("stream.is-live", "true"),
                ("node.latency", latency.as_str()),
                ("node.rate", rate.as_str()),
            ]);
            b.int(id as i32);
        });
    }

    /// `ClientNode.Update` — one output port, real-time capable, advertising a node-level
    /// `EnumFormat` (the session manager's audio adapter enumerates the *node* to find a usable
    /// format — a per-port EnumFormat alone leaves it with "no usable format", like `pw_stream`).
    fn update_node(&mut self) {
        let id = self.node_id;
        let (fmt, rate, ch, dsp) = (self.cfg.format, self.cfg.rate, self.cfg.channels, self.cfg.dsp);
        self.conn.send(id, CN_METHOD_UPDATE, |b| {
            b.int(1 | 2); // change_mask = PARAMS | INFO
            b.int(1); // n_params
            if dsp {
                spa::build_format_dsp(b, spa::PARAM_ENUM_FORMAT);
            } else {
                spa::build_format(b, spa::PARAM_ENUM_FORMAT, fmt, rate, ch);
            }
            // info struct: max_in, max_out, change_mask(Long), flags(Long), n_items, n_info_params
            b.push_struct();
            b.int(0); // max_input_ports
            b.int(1); // max_output_ports
            b.long((spa::NODE_CHANGE_MASK_FLAGS | spa::NODE_CHANGE_MASK_PARAMS) as i64);
            b.long(spa::NODE_FLAG_RT as i64);
            b.int(0); // n_items (node props)
            b.int(2); // n_info_params
            b.id(spa::PARAM_ENUM_FORMAT); b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_FORMAT);      b.int(spa::PARAM_INFO_WRITE as i32);
            b.pop();
        });
    }

    /// Initial `ClientNode.PortUpdate` — advertise our single fixed `EnumFormat` on the output
    /// port and the param set the server may get/set (notably `Format`).
    fn port_update_enum_format(&mut self) {
        let id = self.node_id;
        let (fmt, rate, ch, dsp) = (self.cfg.format, self.cfg.rate, self.cfg.channels, self.cfg.dsp);
        self.conn.send(id, CN_METHOD_PORT_UPDATE, |b| {
            b.int(spa::DIRECTION_OUTPUT as i32);
            b.int(0); // port_id
            b.int(1 | 2); // change_mask = PARAMS | INFO
            b.int(1); // n_params
            if dsp {
                spa::build_format_dsp(b, spa::PARAM_ENUM_FORMAT);
            } else {
                spa::build_format(b, spa::PARAM_ENUM_FORMAT, fmt, rate, ch);
            }
            // port info
            b.push_struct();
            b.long((spa::PORT_CHANGE_MASK_FLAGS | spa::PORT_CHANGE_MASK_RATE | spa::PORT_CHANGE_MASK_PARAMS) as i64);
            b.long(0); // port flags
            b.int(0); // rate.num
            b.int(0); // rate.denom
            // port props: a DSP port names its channel so it links to the device's FL port.
            if dsp {
                b.int(3);
                b.string("port.name").string("output_FL");
                b.string("audio.channel").string("FL");
                b.string("format.dsp").string("32 bit float mono audio");
            } else {
                b.int(0); // n_items (port props)
            }
            // param_info: which params the server can read/write on us
            b.int(5);
            b.id(spa::PARAM_ENUM_FORMAT); b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_META);        b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_IO);          b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_FORMAT);      b.int(spa::PARAM_INFO_RW as i32);
            b.id(spa::PARAM_BUFFERS);     b.int(spa::PARAM_INFO_READ as i32);
            b.pop();
        });
    }

    /// After the server fixates our `Format` on the port, declare our `Buffers` constraints so
    /// it can allocate the shared buffers. We do NOT re-send a `Format` param here — the server
    /// just set it, and re-proposing one restarts negotiation (mirrors pw_stream, which only
    /// stores the set format and exposes Buffers).
    fn port_update_buffers(&mut self) {
        let id = self.node_id;
        let size = (self.cfg.quantum as usize * self.stride) as u32;
        let stride = self.stride as u32;
        self.conn.send(id, CN_METHOD_PORT_UPDATE, |b| {
            b.int(spa::DIRECTION_OUTPUT as i32);
            b.int(0);
            b.int(1 | 2); // PARAMS | INFO
            b.int(1); // n_params: Buffers
            spa::build_buffers(b, size, stride);
            b.push_struct();
            b.long((spa::PORT_CHANGE_MASK_FLAGS | spa::PORT_CHANGE_MASK_RATE | spa::PORT_CHANGE_MASK_PARAMS) as i64);
            b.long(0);
            b.int(0);
            b.int(0);
            b.int(0); // n_items
            b.int(5);
            b.id(spa::PARAM_ENUM_FORMAT); b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_META);        b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_IO);          b.int(spa::PARAM_INFO_READ as i32);
            b.id(spa::PARAM_FORMAT);      b.int(spa::PARAM_INFO_RW as i32);
            b.id(spa::PARAM_BUFFERS);     b.int(spa::PARAM_INFO_READ as i32);
            b.pop();
        });
    }

    fn set_active(&mut self) {
        let id = self.node_id;
        self.conn.send(id, CN_METHOD_SET_ACTIVE, |b| {
            b.bool(true);
        });
    }

    /// The Format was fixated (node- or port-level): declare our concrete `Format` + `Buffers`
    /// on the port so the server allocates the shared buffers. Idempotent.
    fn configure_format(&mut self) {
        if self.format_configured {
            return;
        }
        self.format_configured = true;
        self.port_update_buffers();
    }

    // -- event dispatch --

    fn dispatch(&mut self, id: u32, opcode: u8) -> Result<(), PwError> {
        if id == self.node_id || id == CORE_ID {
            trace!("recv id={id} op={opcode}");
        }
        match (id, opcode) {
            (CORE_ID, CORE_EVENT_ADD_MEM) => {
                let mem_id = {
                    let mut a = self.conn.args();
                    let mem_id = a.int().unwrap_or(0) as u32;
                    let _type = a.id();
                    a.skip(); // Fd pod (index); the fd rides SCM_RIGHTS
                    let _flags = a.int();
                    mem_id
                };
                if let Some(fd) = self.conn.take_fd() {
                    trace!("add_mem {mem_id} fd={fd}");
                    self.mems.insert(mem_id, fd);
                }
            }
            (CORE_ID, CORE_EVENT_PING) => {
                let (pid, seq) = {
                    let mut a = self.conn.args();
                    (a.int().unwrap_or(0), a.int().unwrap_or(0))
                };
                self.conn.send(CORE_ID, CORE_METHOD_PONG, |b| {
                    b.int(pid);
                    b.int(seq);
                });
            }
            (CORE_ID, CORE_EVENT_BOUND_ID) | (CORE_ID, CORE_EVENT_BOUND_PROPS) => {
                // Struct{ Int(proxy_id), Int(global_id), [dict] } — learn our node's global id.
                let (pid, gid) = {
                    let mut a = self.conn.args();
                    (a.int().unwrap_or(0) as u32, a.int().unwrap_or(0) as u32)
                };
                if pid == self.node_id {
                    self.global_id = gid;
                    trace!("our node bound to global id {gid}");
                }
            }
            (CORE_ID, CORE_EVENT_ERROR) => {
                let mut a = self.conn.args();
                let eid = a.int().unwrap_or(0) as u32;
                let seq = a.int().unwrap_or(0);
                let res = a.int().unwrap_or(0);
                let message = a.string().unwrap_or("").to_owned();
                self.error = Some(PwError::Server { id: eid, seq, res, message });
            }
            (id, CN_EVENT_TRANSPORT) if id == self.node_id => {
                let (memid, off, sz) = {
                    let mut a = self.conn.args();
                    a.skip(); // Fd readfd
                    a.skip(); // Fd writefd
                    let memid = a.int().unwrap_or(0) as u32;
                    let off = a.int().unwrap_or(0) as u32;
                    let sz = a.int().unwrap_or(0) as u32;
                    (memid, off, sz)
                };
                let readfd = self.conn.take_fd().unwrap_or(-1);
                let writefd = self.conn.take_fd().unwrap_or(-1);
                self.readfd = readfd;
                self.writefd = writefd;
                self.self_act = self.map(memid, off, sz).unwrap_or(std::ptr::null_mut());
                if !self.self_act.is_null() {
                    unsafe { write_u32(self.self_act, ACT_CLIENT_VERSION, 1) };
                }
                trace!("transport readfd={readfd} memid={memid} off={off} act={:?}", self.self_act);
            }
            (id, CN_EVENT_SET_PARAM) if id == self.node_id => {
                // Node-level set_param is informational (the session manager pokes node Format /
                // ProcessLatency here). The real, link-time format negotiation happens on the
                // *port* (port_set_param below) — mirror pw_stream, which ignores node set_param.
                let param_id = {
                    let mut a = self.conn.args();
                    a.id().unwrap_or(0)
                };
                trace!("set_param id={param_id} (node, informational)");
            }
            (id, CN_EVENT_PORT_SET_PARAM) if id == self.node_id => {
                let param_id = {
                    let mut a = self.conn.args();
                    let _dir = a.int();
                    let _port = a.int();
                    let pid = a.id().unwrap_or(0);
                    let _flags = a.int();
                    // Log the fixated format the server chose (channels/rate reveal a
                    // mono-vs-stereo mismatch with the sink port).
                    if pid == spa::PARAM_FORMAT && dbg_enabled() {
                        if let Some((_ty, _oid, mut obj)) = a.enter_object() {
                            let (mut sfmt, mut srate, mut sch) = (0u32, 0i32, 0i32);
                            while let Some((key, _f, mut v)) = obj.next_prop() {
                                match key {
                                    k if k == spa::FORMAT_AUDIO_FORMAT => sfmt = v.id().unwrap_or(0),
                                    k if k == spa::FORMAT_AUDIO_RATE => srate = v.int().unwrap_or(0),
                                    k if k == spa::FORMAT_AUDIO_CHANNELS => sch = v.int().unwrap_or(0),
                                    _ => {}
                                }
                            }
                            trace!("  fixated format: fmt=0x{sfmt:x} rate={srate} channels={sch}");
                        }
                    }
                    pid
                };
                trace!("port_set_param id={param_id}");
                if param_id == spa::PARAM_FORMAT {
                    self.configure_format();
                }
            }
            (id, CN_EVENT_PORT_USE_BUFFERS) if id == self.node_id => {
                self.use_buffers()?;
            }
            (id, CN_EVENT_PORT_SET_IO) if id == self.node_id => {
                let (io_id, memid, off, sz) = {
                    let mut a = self.conn.args();
                    let _dir = a.int();
                    let _port = a.int();
                    let _mix = a.int();
                    let io_id = a.id().unwrap_or(0);
                    let memid = a.int().unwrap_or(0) as u32;
                    let off = a.int().unwrap_or(0) as u32;
                    let sz = a.int().unwrap_or(0) as u32;
                    (io_id, memid, off, sz)
                };
                if io_id == spa::IO_BUFFERS {
                    self.io_buffers = if memid == spa::ID_INVALID {
                        std::ptr::null_mut()
                    } else {
                        self.map(memid, off, sz).unwrap_or(std::ptr::null_mut())
                    };
                    trace!(
                        "port_set_io Buffers memid={memid} off={off} sz={sz} known_mems={:?} io={:?}",
                        self.mems.keys().collect::<Vec<_>>(),
                        self.io_buffers
                    );
                }
            }
            (id, CN_EVENT_SET_ACTIVATION) if id == self.node_id => {
                let (peer_node, memid, off, sz) = {
                    let mut a = self.conn.args();
                    let peer_node = a.int().unwrap_or(0) as u32;
                    a.skip(); // Fd signalfd
                    let memid = a.int().unwrap_or(0) as u32;
                    let off = a.int().unwrap_or(0) as u32;
                    let sz = a.int().unwrap_or(0) as u32;
                    (peer_node, memid, off, sz)
                };
                let signalfd = self.conn.take_fd().unwrap_or(-1);
                if memid != spa::ID_INVALID && sz != 0 {
                    if let Some(act) = self.map(memid, off, sz) {
                        trace!("set_activation peer_node={peer_node} signalfd={signalfd} memid={memid} off={off} act={act:?}");
                        self.peers.push(Peer { node: peer_node, act, signalfd });
                    }
                }
            }
            (id, CN_EVENT_COMMAND) if id == self.node_id => {
                // args = Struct{ Object(command) }; the object id is the node command.
                let cmd = {
                    let mut a = self.conn.args();
                    a.enter_object().map(|(_ty, cid, _p)| cid)
                };
                match cmd {
                    Some(spa::NODE_COMMAND_START) => {
                        self.running = true;
                        // Become schedulable: INACTIVE -> FINISHED (v6 requirement).
                        if !self.self_act.is_null() {
                            unsafe {
                                a_u32(self.self_act, spa::ACT_STATUS)
                                    .store(spa::ACTIVATION_FINISHED, Ordering::SeqCst)
                            };
                        }
                        trace!("command Start (running)");
                    }
                    Some(spa::NODE_COMMAND_PAUSE) | Some(spa::NODE_COMMAND_SUSPEND) => {
                        self.running = false;
                        trace!("command Pause/Suspend");
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Map the buffers the server allocated (`port_use_buffers`). Layout per remote-node.c:
    /// the buffer's `(mem_id, offset)` region holds metas then the `spa_chunk`s; each data block
    /// is either inline (`MemPtr`, at an offset in that region) or a separate memfd (`MemId`).
    fn use_buffers(&mut self) -> Result<(), PwError> {
        // Parse the buffer descriptors into owned records first (releases the args borrow).
        struct RawData {
            ty: u32,
            data_id: u32,
            mapoffset: u32,
            maxsize: u32,
        }
        struct RawBuf {
            mem_id: u32,
            offset: u32,
            size: u32,
            n_metas: u32,
            datas: Vec<RawData>,
        }
        let raws: Vec<RawBuf> = {
            let mut a = self.conn.args();
            let _dir = a.int();
            let _port = a.int();
            let _mix = a.int();
            let _flags = a.int();
            let n = a.int().unwrap_or(0).max(0) as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mem_id = a.int().unwrap_or(0) as u32;
                let offset = a.int().unwrap_or(0) as u32;
                let size = a.int().unwrap_or(0) as u32;
                let n_metas = a.int().unwrap_or(0).max(0) as u32;
                for _ in 0..n_metas {
                    a.id(); // meta type
                    a.int(); // meta size
                }
                let n_datas = a.int().unwrap_or(0).max(0) as usize;
                let mut datas = Vec::with_capacity(n_datas);
                for _ in 0..n_datas {
                    let ty = a.id().unwrap_or(0);
                    let data_id = a.int().unwrap_or(0) as u32;
                    let _flags = a.int();
                    let mapoffset = a.int().unwrap_or(0) as u32;
                    let maxsize = a.int().unwrap_or(0) as u32;
                    datas.push(RawData { ty, data_id, mapoffset, maxsize });
                }
                v.push(RawBuf { mem_id, offset, size, n_metas, datas });
            }
            v
        };

        self.buffers.clear();
        self.next_buf = 0;
        for rb in &raws {
            let Some(meta_base) = self.map(rb.mem_id, rb.offset, rb.size.max(1)) else {
                return Err(PwError::Protocol("pipewire: buffer metadata map failed"));
            };
            // Chunks sit right after the metas (each meta is 8-byte padded; we requested none).
            let metas_size = rb.n_metas as usize * 8; // conservative; n_metas is 0 for us
            let chunk0 = unsafe { meta_base.add(metas_size) };
            // Single audio block (blocks=1).
            let Some(d) = rb.datas.first() else { continue };
            let data_ptr = if d.ty == spa::DATA_MEM_ID {
                // Data lives in a separate memfd identified by data_id.
                match self.map(d.data_id, d.mapoffset, d.maxsize) {
                    Some(p) => p,
                    None => return Err(PwError::Protocol("pipewire: buffer data map failed")),
                }
            } else if d.ty == spa::DATA_MEM_PTR {
                // Data is inline in the metadata region at offset data_id.
                unsafe { meta_base.add(d.data_id as usize) }
            } else {
                return Err(PwError::Protocol("pipewire: unsupported buffer data type"));
            };
            trace!("buffer data={data_ptr:?} maxsize={} chunk={chunk0:?}", d.maxsize);
            self.buffers.push(Buf { data: data_ptr, maxsize: d.maxsize as usize, chunk: chunk0 });
        }
        Ok(())
    }

    // -- the real-time process cycle (our eventfd fired) --

    fn process_cycle(&mut self) {
        // Consume the eventfd signal (drain both ends; the driver may write either).
        let mut ev = [0u8; 8];
        unsafe {
            libc::read(self.readfd, ev.as_mut_ptr() as *mut libc::c_void, 8);
            if self.writefd >= 0 {
                libc::read(self.writefd, ev.as_mut_ptr() as *mut libc::c_void, 8);
            }
        }

        if dbg_enabled() && self.cycles < 3 {
            let st = if self.self_act.is_null() {
                u32::MAX
            } else {
                unsafe { a_u32(self.self_act, spa::ACT_STATUS).load(Ordering::SeqCst) }
            };
            trace!(
                "woke: running={} act_status={} io_null={} nbufs={} peers={}",
                self.running,
                st,
                self.io_buffers.is_null(),
                self.buffers.len(),
                self.peers.len()
            );
        }

        if self.self_act.is_null() {
            return;
        }
        // TRIGGERED -> AWAKE.
        unsafe { a_u32(self.self_act, spa::ACT_STATUS).store(spa::ACTIVATION_AWAKE, Ordering::SeqCst) };

        if self.running && !self.io_buffers.is_null() && !self.buffers.is_empty() {
            self.produce();
            self.cycles += 1;
            if dbg_enabled() && self.cycles % 200 == 1 {
                trace!("process cycle #{} (audio flowing)", self.cycles);
            }
        }

        // Trigger downstream, then AWAKE -> FINISHED.
        self.trigger_peers();
        unsafe {
            a_u32(self.self_act, spa::ACT_STATUS).store(spa::ACTIVATION_FINISHED, Ordering::SeqCst)
        };
    }

    /// Fill one buffer with a quantum of PCM and publish it through the port's `spa_io_buffers`.
    fn produce(&mut self) {
        let stride = self.stride;
        // Frames this cycle: the driver's clock duration, clamped to the buffer (sane fallback
        // if the offset ever drifts — never an OOB read, the activation record is far larger).
        let maxframes = self.buffers[self.next_buf].maxsize / stride;
        let dur = unsafe { read_u64(self.self_act, spa::ACT_CLOCK_DURATION) } as usize;
        let frames = if dur > 0 && dur <= maxframes { dur } else { maxframes };
        let nbytes = frames * stride;

        let bi = self.next_buf;
        self.next_buf = (self.next_buf + 1) % self.buffers.len();
        let buf = &self.buffers[bi];

        // SAFETY: `data`/`chunk` point into the buffer's mmap; `nbytes <= maxsize`.
        let out = unsafe { std::slice::from_raw_parts_mut(buf.data, nbytes) };
        let got = (self.fill)(out);

        unsafe {
            // spa_chunk { offset, size, stride, flags }
            write_u32(buf.chunk, 0, 0);
            write_u32(buf.chunk, 4, got as u32);
            write_u32(buf.chunk, 8, stride as u32);
            write_u32(buf.chunk, 12, 0);
            // spa_io_buffers { i32 status; u32 buffer_id } — publish buffer_id then status.
            write_u32(self.io_buffers, 4, bi as u32);
            a_i32(self.io_buffers, 0).store(spa::STATUS_HAVE_DATA, Ordering::SeqCst);
        }
    }

    /// Wake each downstream node whose last dependency we are (trigger_target_v1).
    fn trigger_peers(&self) {
        for peer in &self.peers {
            if peer.node == self.global_id {
                continue; // never trigger our own node
            }
            let pending = unsafe { a_i32(peer.act, spa::ACT_STATE0_PENDING).fetch_sub(1, Ordering::SeqCst) };
            if pending - 1 == 0 {
                let ok = unsafe {
                    a_u32(peer.act, spa::ACT_STATUS)
                        .compare_exchange(
                            spa::ACTIVATION_NOT_TRIGGERED,
                            spa::ACTIVATION_TRIGGERED,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                };
                if ok {
                    let one = 1u64.to_ne_bytes();
                    unsafe { libc::write(peer.signalfd, one.as_ptr() as *const libc::c_void, 8) };
                }
            }
        }
    }

    /// Run until `quit` is set, an error arrives, or the daemon disconnects.
    fn run(&mut self, quit: &AtomicBool) -> Result<(), PwError> {
        while !quit.load(Ordering::Relaxed) {
            let mut fds = [
                libc::pollfd { fd: self.conn.as_raw_fd(), events: libc::POLLIN, revents: 0 },
                libc::pollfd { fd: self.readfd, events: libc::POLLIN, revents: 0 },
                libc::pollfd { fd: self.writefd, events: libc::POLLIN, revents: 0 },
            ];
            let nfds = if self.readfd >= 0 { 3 } else { 1 };
            let r = unsafe { libc::poll(fds.as_mut_ptr(), nfds as libc::nfds_t, 100) };
            if r < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(PwError::Io(e));
            }
            // RT eventfd first (latency-sensitive).
            if dbg_enabled() && self.running && r == 0 {
                // Poll timed out with no eventfd wake — show what the driver left our activation
                // status at (3=FINISHED means the driver never re-armed us; 1=TRIGGERED means it
                // signalled but we missed the fd).
                self.idle_ticks += 1;
                if self.idle_ticks % 10 == 1 && !self.self_act.is_null() {
                    let st = unsafe { a_u32(self.self_act, spa::ACT_STATUS).load(Ordering::SeqCst) };
                    let req = unsafe { a_i32(self.self_act, 12).load(Ordering::SeqCst) };
                    let pend = unsafe { a_i32(self.self_act, spa::ACT_STATE0_PENDING).load(Ordering::SeqCst) };
                    let driver = unsafe { a_u32(self.self_act, 552).load(Ordering::SeqCst) };
                    trace!("idle: status={st} state0.required={req} pending={pend} driver_id={driver}");
                }
            }
            if nfds == 3 && (fds[1].revents | fds[2].revents) & libc::POLLIN != 0 {
                if dbg_enabled() && self.cycles < 2 {
                    trace!("wake on read={:#x} write={:#x}", fds[1].revents, fds[2].revents);
                }
                self.process_cycle();
            }
            if fds[0].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
                if !self.conn.recv()? {
                    return Ok(()); // daemon closed
                }
                while let Some(hdr) = self.conn.next_message() {
                    self.dispatch(hdr.id, hdr.opcode)?;
                    if let Some(e) = self.error.take() {
                        return Err(e);
                    }
                }
                self.conn.flush()?;
            }
        }
        Ok(())
    }
}

impl<F> Drop for Node<F> {
    fn drop(&mut self) {
        for m in &self.mappings {
            unsafe { libc::munmap(m.base, m.len) };
        }
        for (_, fd) in self.mems.drain() {
            unsafe { libc::close(fd) };
        }
        if self.readfd >= 0 {
            unsafe { libc::close(self.readfd) };
        }
        for p in &self.peers {
            if p.signalfd >= 0 {
                unsafe { libc::close(p.signalfd) };
            }
        }
    }
}

/// Export a client node over the native protocol and run its message + RT loop: create the node,
/// negotiate the `cfg` format and buffers, map the shared memory, and — once the graph driver
/// schedules us — pull PCM from `fill` (fill the slice with whole interleaved frames, return
/// bytes written) each cycle, until `quit` is set (or the daemon disconnects / errors). Blocks
/// the calling thread; run it on a dedicated audio thread. See the module docs for the current
/// limit on the driver→client RT trigger.
pub fn play<F: FnMut(&mut [u8]) -> usize>(
    cfg: AudioConfig,
    quit: &AtomicBool,
    fill: F,
) -> Result<(), PwError> {
    let conn = Connection::connect()?;
    let stride = cfg.format.bytes() * cfg.channels as usize;
    let mut node = Node {
        conn,
        cfg,
        stride,
        node_id: 0,
        global_id: 0,
        mems: HashMap::new(),
        mappings: Vec::new(),
        self_act: std::ptr::null_mut(),
        readfd: -1,
        writefd: -1,
        io_buffers: std::ptr::null_mut(),
        buffers: Vec::new(),
        peers: Vec::new(),
        next_buf: 0,
        running: false,
        format_configured: false,
        cycles: 0,
        idle_ticks: 0,
        error: None,
        fill,
    };
    node.hello()?;
    node.create_node();
    node.update_node();
    node.port_update_enum_format();
    node.set_active();
    node.conn.flush()?;
    node.run(quit)
}
