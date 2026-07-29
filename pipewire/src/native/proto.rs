//! The PipeWire Core / Client / Registry interfaces and the connection handshake, spoken
//! directly over [`Connection`]. Interface opcodes and version numbers are from the native
//! protocol (see `REFERENCES.md`; `pw_core_methods`/`pw_core_events` etc. in PipeWire).
//!
//! Handshake (client → server): open the socket, `Core.Hello(version=3)`, then
//! `Client.UpdateProperties` with our application name; a `Core.Sync`/`Core.Done` round-trip
//! flushes the exchange and confirms liveness. `Core.Ping` is answered with `Core.Pong`,
//! `Core.Error` is surfaced. `Core.GetRegistry` + a round-trip enumerates the server's globals.
//!
//! This is the control path — everything runs on one thread, one message at a time. The
//! real-time audio path (ClientNode: export a node, negotiate `Format`/`Buffers`, drive the
//! shared-memory ring off the activation eventfd) is the next milestone and layers on top of
//! this same POD + framing foundation.

// String-building for server info / registry props is control-path, one-time work — not an
// `Element::process()` frame loop (spec: performance #1; clippy.toml).
#![allow(clippy::disallowed_methods)]

use std::fmt;
use std::io;

use crate::native::conn::{Connection, CLIENT_ID, CORE_ID};
use crate::native::wire::Header;

// --- interface versions (the `version` we announce; min that supports the modern protocol) --

/// `Core.Hello` / `Core.GetRegistry` version. v3 enables the footer-bearing wire protocol; the
/// server sends footers we skip, and does not require us to send any.
pub const CORE_VERSION: i32 = 3;
/// `Registry` interface version.
pub const REGISTRY_VERSION: i32 = 3;

// --- Core interface (object id 0) ----------------------------------------------------------

pub const CORE_METHOD_HELLO: u8 = 1;
pub const CORE_METHOD_SYNC: u8 = 2;
pub const CORE_METHOD_PONG: u8 = 3;
pub const CORE_METHOD_ERROR: u8 = 4;
pub const CORE_METHOD_GET_REGISTRY: u8 = 5;
pub const CORE_METHOD_CREATE_OBJECT: u8 = 6;
pub const CORE_METHOD_DESTROY: u8 = 7;

pub const CORE_EVENT_INFO: u8 = 0;
pub const CORE_EVENT_DONE: u8 = 1;
pub const CORE_EVENT_PING: u8 = 2;
pub const CORE_EVENT_ERROR: u8 = 3;
pub const CORE_EVENT_REMOVE_ID: u8 = 4;
pub const CORE_EVENT_BOUND_ID: u8 = 5;
pub const CORE_EVENT_ADD_MEM: u8 = 6;
pub const CORE_EVENT_REMOVE_MEM: u8 = 7;
pub const CORE_EVENT_BOUND_PROPS: u8 = 8;

// --- Client interface (object id 1) --------------------------------------------------------

pub const CLIENT_METHOD_ERROR: u8 = 1;
pub const CLIENT_METHOD_UPDATE_PROPERTIES: u8 = 2;

// --- Registry interface (a client-allocated id) --------------------------------------------

pub const REGISTRY_METHOD_BIND: u8 = 1;
pub const REGISTRY_METHOD_DESTROY: u8 = 2;

pub const REGISTRY_EVENT_GLOBAL: u8 = 0;
pub const REGISTRY_EVENT_GLOBAL_REMOVE: u8 = 1;

// --- data returned to callers --------------------------------------------------------------

/// The server's identity, from the `Core.Info` event received during the handshake.
#[derive(Clone, Debug, Default)]
pub struct ServerInfo {
    pub id: u32,
    pub cookie: u32,
    pub user_name: String,
    pub host_name: String,
    /// The daemon's PipeWire version string (e.g. `"1.6.7"`).
    pub version: String,
    pub name: String,
}

/// One global object the daemon exposes (a node, port, device, factory, …), from a
/// `Registry.Global` event.
#[derive(Clone, Debug)]
pub struct Global {
    pub id: u32,
    pub permissions: u32,
    /// The interface type, e.g. `"PipeWire:Interface:Node"`.
    pub type_: String,
    pub version: u32,
    pub props: Vec<(String, String)>,
}

impl Global {
    /// A convenience lookup over the global's properties.
    pub fn prop(&self, key: &str) -> Option<&str> {
        self.props.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

/// Errors from the native client.
#[derive(Debug)]
pub enum PwError {
    /// A socket / syscall failure.
    Io(io::Error),
    /// The daemon closed the connection.
    Eof,
    /// A `Core.Error` event: the server rejected something.
    Server { id: u32, seq: i32, res: i32, message: String },
    /// A malformed message from the server (a bug on either side).
    Protocol(&'static str),
}

impl fmt::Display for PwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PwError::Io(e) => write!(f, "pipewire io: {e}"),
            PwError::Eof => write!(f, "pipewire daemon closed the connection"),
            PwError::Server { id, seq, res, message } => {
                write!(f, "pipewire server error on id {id} (seq {seq}, res {res}): {message}")
            }
            PwError::Protocol(m) => write!(f, "pipewire protocol error: {m}"),
        }
    }
}

impl std::error::Error for PwError {}

impl From<io::Error> for PwError {
    fn from(e: io::Error) -> Self {
        PwError::Io(e)
    }
}

// --- the client ----------------------------------------------------------------------------

/// A connected PipeWire client: owns the [`Connection`] and dispatches Core/Client/Registry
/// events. Single-threaded, synchronous — the control surface the RT audio path builds on.
pub struct PwClient {
    conn: Connection,
    info: ServerInfo,
    /// The Registry proxy id once [`registry_globals`](Self::registry_globals) has run (0 = not
    /// yet requested).
    registry_id: u32,
    globals: Vec<Global>,
    /// Seq of the last `Core.Done` seen — a round-trip completes when this reaches its target.
    last_done: Option<i32>,
    /// Our `Core.Sync` argument seq counter (echoed back in `Core.Done`).
    sync_seq: i32,
    /// A pending `Core.Error`, surfaced by the next dispatch loop.
    error: Option<PwError>,
}

impl PwClient {
    /// Connect to the daemon and complete the handshake (`Hello` + `UpdateProperties` +
    /// a sync round-trip). `app_name` becomes the client's `application.name`.
    pub fn connect(app_name: &str) -> Result<PwClient, PwError> {
        let conn = Connection::connect()?;
        let mut c = PwClient {
            conn,
            info: ServerInfo::default(),
            registry_id: 0,
            globals: Vec::new(),
            last_done: None,
            sync_seq: 0,
            error: None,
        };
        c.hello(app_name)?;
        Ok(c)
    }

    /// The server identity captured during the handshake.
    pub fn server_info(&self) -> &ServerInfo {
        &self.info
    }

    fn hello(&mut self, app_name: &str) -> Result<(), PwError> {
        self.conn.send(CORE_ID, CORE_METHOD_HELLO, |b| {
            b.int(CORE_VERSION);
        });
        // The args are one dict (a nested Struct{ Int(n), key/value pairs }).
        self.conn.send(CLIENT_ID, CLIENT_METHOD_UPDATE_PROPERTIES, |b| {
            b.dict(&[
                ("application.name", app_name),
                ("application.process.binary", "streamcraft"),
            ]);
        });
        self.conn.flush()?;
        // Info arrives unprompted after Hello; the sync round-trip waits for it + confirms.
        self.roundtrip()
    }

    /// A `Core.Sync`/`Core.Done` barrier: everything the server queued before our sync has been
    /// delivered when this returns (registry dumps, error reports, …).
    pub fn roundtrip(&mut self) -> Result<(), PwError> {
        self.sync_seq = self.sync_seq.wrapping_add(1);
        let target = self.sync_seq;
        self.conn.send(CORE_ID, CORE_METHOD_SYNC, |b| {
            b.int(CORE_ID as i32);
            b.int(target);
        });
        self.conn.flush()?;
        loop {
            let hdr = self.conn.next_blocking()?.ok_or(PwError::Eof)?;
            self.dispatch(hdr)?;
            if let Some(e) = self.error.take() {
                return Err(e);
            }
            if self.last_done == Some(target) {
                return Ok(());
            }
        }
    }

    /// Enumerate the daemon's global objects (`GetRegistry` + a round-trip). Meaningful once —
    /// the server emits each global a single time; the result is cached.
    pub fn registry_globals(&mut self) -> Result<&[Global], PwError> {
        if self.registry_id == 0 {
            let rid = self.conn.alloc_id();
            self.registry_id = rid;
            self.conn.send(CORE_ID, CORE_METHOD_GET_REGISTRY, |b| {
                b.int(REGISTRY_VERSION);
                b.int(rid as i32);
            });
            self.conn.flush()?;
            self.roundtrip()?; // globals arrive in order, before the Done
        }
        Ok(&self.globals)
    }

    /// Handle one incoming message, updating client state. Reads all args into locals before any
    /// reply (`Pong`), so the read-only args borrow is released first.
    fn dispatch(&mut self, hdr: Header) -> Result<(), PwError> {
        match (hdr.id, hdr.opcode) {
            (CORE_ID, CORE_EVENT_INFO) => {
                let mut a = self.conn.args();
                let mut info = ServerInfo {
                    id: a.int().unwrap_or(0) as u32,
                    cookie: a.int().unwrap_or(0) as u32,
                    user_name: a.string().unwrap_or("").to_owned(),
                    host_name: a.string().unwrap_or("").to_owned(),
                    version: a.string().unwrap_or("").to_owned(),
                    name: a.string().unwrap_or("").to_owned(),
                };
                let _change_mask = a.long(); // advance past the change_mask before the props dict
                let _ = a.dict(|_k, _v| {}); // props (unused here)
                // (borrow of `a`/self.conn ends at the close of this arm)
                std::mem::swap(&mut self.info, &mut info);
            }
            (CORE_ID, CORE_EVENT_DONE) => {
                let mut a = self.conn.args();
                let _id = a.int();
                self.last_done = a.int();
            }
            (CORE_ID, CORE_EVENT_PING) => {
                let (id, seq) = {
                    let mut a = self.conn.args();
                    (a.int().unwrap_or(0), a.int().unwrap_or(0))
                };
                // Answer promptly so the daemon does not consider us unresponsive.
                self.conn.send(CORE_ID, CORE_METHOD_PONG, |b| {
                    b.int(id);
                    b.int(seq);
                });
                self.conn.flush()?;
            }
            (CORE_ID, CORE_EVENT_ERROR) => {
                let mut a = self.conn.args();
                let id = a.int().unwrap_or(0) as u32;
                let seq = a.int().unwrap_or(0);
                let res = a.int().unwrap_or(0);
                let message = a.string().unwrap_or("").to_owned();
                self.error = Some(PwError::Server { id, seq, res, message });
            }
            (id, REGISTRY_EVENT_GLOBAL) if id == self.registry_id && id != 0 => {
                let g = {
                    let mut a = self.conn.args();
                    let id = a.int().unwrap_or(0) as u32;
                    let permissions = a.int().unwrap_or(0) as u32;
                    let type_ = a.string().unwrap_or("").to_owned();
                    let version = a.int().unwrap_or(0) as u32;
                    let mut props = Vec::new();
                    let _ = a.dict(|k, v| props.push((k.to_owned(), v.to_owned())));
                    Global { id, permissions, type_, version, props }
                };
                self.globals.push(g);
            }
            // RemoveId / AddMem / RemoveMem / BoundId / BoundProps / Client events /
            // Registry.GlobalRemove: not needed for the handshake + dump. The RT audio path
            // will handle AddMem/RemoveMem (shared buffers) and BoundProps (our node's id).
            _ => {}
        }
        Ok(())
    }
}
