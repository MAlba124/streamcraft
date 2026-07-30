//! Core's own logging — no `log`/`tracing` dependency (spec: Debuggability, "Logging
//! cont'd"). Logging is the third face of the observability plane (with counters and the
//! bus) and obeys the same rule: *observation never perturbs what it observes.*
//!
//! The shape follows the spec:
//! - Emission writes a fixed-size POD [`LogRecord`] — never a formatted string. The human
//!   text is produced at the *drain*, from the record's `&'static` event name plus typed
//!   [`Field`]s, so the streaming thread never touches bytes or allocates.
//! - Transport is a per-group SPSC ring ([`crate::ring`]); a low-priority [`LogDrain`]
//!   consumes. Ring full → **drop and count**, never block a streaming thread (that would
//!   violate the latency guarantee).
//! - [`Level`] filtering is one relaxed atomic compare behind the [`log!`] macro, so a
//!   disabled record costs a predictable branch and does not even evaluate its arguments.
//!
//! What is *not* wired here yet: the `Ctx`/pipeline integration (the group owns a [`Log`]
//! and stamps the current element via [`Log::set_element`]; `Ctx` will forward to it) and
//! the introspection-protocol drain. Those touch `ctx.rs`/`pipeline.rs`; this module is
//! the self-contained primitive underneath them. TODO(step 7): wire into `Ctx`.

use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use crate::id::ElementId;
use crate::ring::{spsc, Consumer, Producer};
use crate::time::Timestamp;

/// Severity, most severe first. The derived `Ord` is *not* used for filtering (a numeric
/// [`Level::rank`] is, so "off" has a natural zero below `Error`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    /// Verbosity rank, `1..=5`. `0` is reserved for "off" (nothing passes), which is why
    /// this is not just `self as u8`.
    pub const fn rank(self) -> u8 {
        match self {
            Level::Error => 1,
            Level::Warn => 2,
            Level::Info => 3,
            Level::Debug => 4,
            Level::Trace => 5,
        }
    }

    pub const fn from_rank(r: u8) -> Option<Level> {
        Some(match r {
            1 => Level::Error,
            2 => Level::Warn,
            3 => Level::Info,
            4 => Level::Debug,
            5 => Level::Trace,
            _ => return None,
        })
    }

    /// Fixed-width uppercase tag for drain output.
    pub const fn as_str(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }

    /// Parse a level *name* (case-insensitive). Numbers and "off" are handled by
    /// [`parse_level_token`], which the env syntax uses.
    /// The canonical display name (uppercase, as the drain prints it).
    pub fn name(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }

    pub fn from_name(s: &str) -> Option<Level> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "error" | "err" => Level::Error,
            "warn" | "warning" => Level::Warn,
            "info" => Level::Info,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => return None,
        })
    }
}

/// A runtime-adjustable verbosity gate (spec: `PROFLUENS_DEBUG=element:level`). One
/// relaxed atomic; `0` = off. Per-element overrides live in [`DebugSpec::targets`] and are
/// applied by the pipeline once element names resolve — this global gate is the fast path.
pub struct LevelFilter {
    threshold: AtomicU8,
}

impl LevelFilter {
    /// Silent by default: logging costs nothing until a level is set.
    pub const fn new() -> Self {
        Self {
            threshold: AtomicU8::new(0),
        }
    }

    pub fn with_level(level: Level) -> Self {
        let f = Self::new();
        f.set(Some(level));
        f
    }

    /// Set the maximum enabled level, or `None` to silence.
    pub fn set(&self, level: Option<Level>) {
        self.threshold
            .store(level.map_or(0, Level::rank), Relaxed);
    }

    pub fn level(&self) -> Option<Level> {
        Level::from_rank(self.threshold.load(Relaxed))
    }

    /// The one hot-path check: a record at `level` passes iff its rank is within threshold.
    #[inline]
    pub fn enabled(&self, level: Level) -> bool {
        level.rank() <= self.threshold.load(Relaxed)
    }

    /// Apply the global part of `PROFLUENS_DEBUG`, if set. Per-target rules are returned
    /// for the caller (pipeline) to apply later; here we only move the global gate.
    pub fn apply_env(&self) -> DebugSpec {
        let spec = std::env::var("PROFLUENS_DEBUG")
            .ok()
            .map(|s| DebugSpec::parse(&s))
            .unwrap_or_default();
        if let Some(g) = spec.global {
            self.threshold.store(g, Relaxed);
        }
        spec
    }
}

impl Default for LevelFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed `PROFLUENS_DEBUG` value: a global catch-all level plus per-target overrides
/// (`name:level`). Levels are stored as ranks (`0..=5`); invalid tokens are ignored.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DebugSpec {
    pub global: Option<u8>,
    pub targets: Vec<(String, u8)>,
}

impl DebugSpec {
    /// Parse the gst-style syntax: comma-separated tokens, each either a bare level
    /// (global) or `name:level`. Whitespace-tolerant; malformed tokens are skipped.
    // Parses the PROFLUENS_DEBUG env var once at startup, not per frame.
    #[allow(clippy::disallowed_methods)]
    pub fn parse(s: &str) -> DebugSpec {
        let mut spec = DebugSpec::default();
        for tok in s.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            match tok.split_once(':') {
                Some((name, lvl)) => {
                    if let Some(rank) = parse_level_token(lvl) {
                        spec.targets.push((name.trim().to_string(), rank));
                    }
                }
                None => {
                    if let Some(rank) = parse_level_token(tok) {
                        spec.global = Some(rank);
                    }
                }
            }
        }
        spec
    }
}

/// Parse one level token — a number `0..=5` or a name (`off`/`error`/…/`trace`) — to a
/// rank. Out-of-range numbers and unknown names return `None`.
pub fn parse_level_token(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Ok(n) = s.parse::<u8>() {
        return (n <= 5).then_some(n);
    }
    if s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("none") {
        return Some(0);
    }
    Level::from_name(s).map(Level::rank)
}

/// A typed structured field on a log record. Keys are `&'static str` (the call site names
/// them); values are POD. Arbitrary runtime strings on a hot path are a smell — use an
/// event name plus typed fields, not string interpolation.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub key: &'static str,
    pub val: FieldValue,
}

impl Field {
    pub const fn new(key: &'static str, val: FieldValue) -> Self {
        Self { key, val }
    }
}

const EMPTY_FIELD: Field = Field {
    key: "",
    val: FieldValue::Uint(0),
};

/// A POD field value. `Str` is `&'static` on purpose (see [`Field`]); the open sub-decision
/// in the spec — a side arena for cold variable-length payloads — is deliberately not here.
#[derive(Clone, Copy, Debug)]
pub enum FieldValue {
    Int(i64),
    Uint(u64),
    Bool(bool),
    Str(&'static str),
    Fourcc([u8; 4]),
    Time(Timestamp),
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldValue::Int(v) => write!(f, "{v}"),
            FieldValue::Uint(v) => write!(f, "{v}"),
            FieldValue::Bool(v) => write!(f, "{v}"),
            FieldValue::Str(s) => write!(f, "{s}"),
            FieldValue::Time(t) => match t.nanos() {
                Some(n) => write!(f, "{n}ns"),
                None => write!(f, "none"),
            },
            FieldValue::Fourcc(b) => {
                for &c in b {
                    write!(f, "{}", c as char)?;
                }
                Ok(())
            }
        }
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for FieldValue {
            fn from(v: $t) -> Self { FieldValue::Int(v as i64) }
        }
    )*};
}
macro_rules! from_uint {
    ($($t:ty),*) => {$(
        impl From<$t> for FieldValue {
            fn from(v: $t) -> Self { FieldValue::Uint(v as u64) }
        }
    )*};
}
from_int!(i8, i16, i32, i64, isize);
from_uint!(u8, u16, u32, u64, usize);

impl From<bool> for FieldValue {
    fn from(v: bool) -> Self {
        FieldValue::Bool(v)
    }
}
impl From<&'static str> for FieldValue {
    fn from(v: &'static str) -> Self {
        FieldValue::Str(v)
    }
}
impl From<[u8; 4]> for FieldValue {
    fn from(v: [u8; 4]) -> Self {
        FieldValue::Fourcc(v)
    }
}
impl From<Timestamp> for FieldValue {
    fn from(v: Timestamp) -> Self {
        FieldValue::Time(v)
    }
}
impl From<ElementId> for FieldValue {
    fn from(v: ElementId) -> Self {
        FieldValue::Uint(v.0 as u64)
    }
}

/// Maximum structured fields carried inline on a record (keeps it small and cache-dense).
pub const MAX_FIELDS: usize = 4;

/// One log line as a fixed-size, `Copy` POD record — the unit that crosses the ring. Text
/// is reconstructed at the drain from `event` + `fields`, never built on the hot path.
#[derive(Clone, Copy, Debug)]
pub struct LogRecord {
    pub ts: Timestamp,
    pub element: ElementId,
    /// The element's descriptor name (`""` for framework internals) — a `&'static`
    /// pointer stamped at emit, formatted only at the drain.
    pub name: &'static str,
    pub level: Level,
    pub event: &'static str,
    pub nfields: u8,
    pub fields: [Field; MAX_FIELDS],
}

impl LogRecord {
    /// The populated fields (drops the trailing filler).
    pub fn fields(&self) -> &[Field] {
        &self.fields[..self.nfields as usize]
    }
}

/// The producer side of a log channel: one per thread group, written by that group's single
/// thread only. `push` is non-blocking — a full ring drops and bumps the shared counter.
pub struct LogSink {
    tx: Producer<LogRecord>,
    dropped: Arc<AtomicU64>,
}

impl LogSink {
    /// Enqueue a record, or drop it (and count the drop) if the ring is full. Never blocks.
    pub fn push(&self, rec: LogRecord) {
        if self.tx.try_push(rec).is_err() {
            self.dropped.fetch_add(1, Relaxed);
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Relaxed)
    }
}

/// The consumer side: a low-priority drain. `next_blocking` parks until a record arrives or
/// the sink closes; the format helpers turn records into human text off the hot path.
pub struct LogDrain {
    rx: Consumer<LogRecord>,
    dropped: Arc<AtomicU64>,
}

impl LogDrain {
    pub fn try_next(&self) -> Option<LogRecord> {
        self.rx.try_pop()
    }

    /// Block until the next record, or `None` once the sink is dropped and drained.
    pub fn next_blocking(&self) -> Option<LogRecord> {
        self.rx.pop()
    }

    /// Whether this drain's [`LogSink`] has been dropped. The ring may still hold
    /// undrained records — keep calling [`try_next`](Self::try_next) until it returns
    /// `None` before treating the channel as finished. Used by a multi-drain poller to
    /// know when a channel is done without blocking on it.
    pub fn is_closed(&self) -> bool {
        self.rx.is_closed()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Relaxed)
    }

    /// Drain everything currently queued into `w`, returning how many records were written.
    pub fn write_pending(&self, w: &mut impl Write) -> io::Result<usize> {
        let mut n = 0;
        while let Some(rec) = self.rx.try_pop() {
            format_record(w, &rec)?;
            n += 1;
        }
        Ok(n)
    }

    /// Run this drain to completion, formatting each record to stderr — with ANSI
    /// colors when stderr is a terminal (see [`format_record_styled`]; `NO_COLOR` and
    /// `PROFLUENS_LOG_COLOR=always|never` override). Intended for a dedicated
    /// low-priority thread (see [`LogDrain::spawn_stderr`]).
    pub fn run_to_stderr(self) {
        let styled = stderr_colors_enabled();
        let stderr = io::stderr();
        while let Some(rec) = self.rx.pop() {
            let mut lock = stderr.lock();
            let _ = format_record_styled(&mut lock, &rec, styled);
        }
    }

    /// Spawn the stderr drain on its own thread. This is the "one line in `main`" dev path;
    /// the thread lives until the sink is dropped.
    pub fn spawn_stderr(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("pf-log-drain".into())
            .spawn(move || self.run_to_stderr())
            .expect("spawn log drain")
    }
}

/// Create a log channel sized to `capacity` records. One per thread group.
pub fn log_channel(capacity: usize) -> (LogSink, LogDrain) {
    let (tx, rx) = spsc::<LogRecord>(capacity);
    let dropped = Arc::new(AtomicU64::new(0));
    (
        LogSink {
            tx,
            dropped: dropped.clone(),
        },
        LogDrain { rx, dropped },
    )
}

/// An element's (or the framework's) handle for emitting logs. Owns the group's [`LogSink`]
/// and the shared [`LevelFilter`]; the current [`ElementId`] is stamped on each record and
/// updated by the group as it runs each element in turn.
pub struct Log {
    sink: LogSink,
    filter: Arc<LevelFilter>,
    element: ElementId,
    /// The element's descriptor name (`""` for framework internals). A `&'static`
    /// pointer, so stamping it per record is free and humanization stays at the drain.
    name: &'static str,
    start: Instant,
}

impl Log {
    pub fn new(sink: LogSink, filter: Arc<LevelFilter>) -> Self {
        Self {
            sink,
            filter,
            element: ElementId(0),
            name: "",
            start: Instant::now(),
        }
    }

    /// The dev "one line": build a channel, spawn the stderr drain, gate at `level`, and
    /// return the emit handle. The drain thread runs until the returned `Log` is dropped.
    pub fn to_stderr(capacity: usize, level: Level) -> Log {
        let (sink, drain) = log_channel(capacity);
        drain.spawn_stderr();
        Log::new(sink, Arc::new(LevelFilter::with_level(level)))
    }

    /// Stamp subsequent records with `element` and its descriptor `name` (the pipeline
    /// sets this once per element at run setup; framework-internal `Log`s leave it and
    /// render as `elem#N`).
    pub fn set_element(&mut self, element: ElementId, name: &'static str) {
        self.element = element;
        self.name = name;
    }

    pub fn element(&self) -> ElementId {
        self.element
    }

    pub fn filter(&self) -> &LevelFilter {
        &self.filter
    }

    /// Records dropped so far because the ring was full.
    pub fn dropped(&self) -> u64 {
        self.sink.dropped()
    }

    /// Running-time stamp for a record. Cheap monotonic read; once wired into `Ctx` this
    /// will read the pipeline clock instead, so records share the media timebase.
    fn now(&self) -> Timestamp {
        Timestamp::from_nanos(self.start.elapsed().as_nanos() as u64)
    }
}

/// The emit surface the [`log!`] macro targets. Implemented for [`Log`] now; `Ctx` will
/// implement it later (forwarding to the group's `Log`), so call sites don't change.
pub trait Loggable {
    fn enabled(&self, level: Level) -> bool;
    fn emit(&self, level: Level, event: &'static str, fields: &[Field]);
}

impl Loggable for Log {
    #[inline]
    fn enabled(&self, level: Level) -> bool {
        self.filter.enabled(level)
    }

    fn emit(&self, level: Level, event: &'static str, fields: &[Field]) {
        let n = fields.len().min(MAX_FIELDS);
        let mut buf = [EMPTY_FIELD; MAX_FIELDS];
        buf[..n].copy_from_slice(&fields[..n]);
        self.sink.push(LogRecord {
            ts: self.now(),
            element: self.element,
            name: self.name,
            level,
            event,
            nfields: n as u8,
            fields: buf,
        });
    }
}

/// Emit a structured log record if the target's level gate is open.
///
/// ```ignore
/// log!(&log, Level::Debug, "header_parsed", len = n, codec = "flac");
/// ```
///
/// The gate is checked *before* the field expressions are evaluated, so a disabled record
/// costs one relaxed atomic load and a branch — its arguments are never touched.
#[macro_export]
macro_rules! log {
    ($target:expr, $level:expr, $event:expr $(, $key:ident = $val:expr)* $(,)?) => {{
        let __log = $target;
        let __lvl = $level;
        if $crate::log::Loggable::enabled(__log, __lvl) {
            $crate::log::Loggable::emit(
                __log,
                __lvl,
                $event,
                &[$($crate::log::Field::new(
                    ::core::stringify!($key),
                    $crate::log::FieldValue::from($val),
                )),*],
            );
        }
    }};
}

/// Format a record as one human line: `[    12.000ms] INFO  flacdec#3 event key=val …`
/// (`elem#3` when no element name was stamped — framework internals). Plain, no ANSI —
/// the machine/file/test-safe form; the stderr drain uses
/// [`format_record_styled`] when the terminal supports it.
pub fn format_record(w: &mut impl Write, rec: &LogRecord) -> io::Result<()> {
    format_record_styled(w, rec, false)
}

/// ANSI reset.
const SGR_RESET: &str = "\x1b[0m";
/// Dim — timestamps and field keys (context, not content).
const SGR_DIM: &str = "\x1b[2m";

/// The level's SGR: severity at a glance (error red jumps out of a scroll).
pub(crate) fn level_sgr(level: Level) -> &'static str {
    match level {
        Level::Error => "\x1b[1;31m", // bold red
        Level::Warn => "\x1b[33m",    // yellow
        Level::Info => "\x1b[32m",    // green
        Level::Debug => "\x1b[36m",   // cyan
        Level::Trace => "\x1b[2m",    // dim
    }
}

/// A stable per-element hue, so one element's lines group visually in interleaved
/// multi-group output. The palette avoids the level colors (no red/yellow/green).
fn element_sgr(id: ElementId) -> &'static str {
    const PALETTE: [&str; 6] = [
        "\x1b[94m", // bright blue
        "\x1b[95m", // bright magenta
        "\x1b[96m", // bright cyan
        "\x1b[34m", // blue
        "\x1b[35m", // magenta
        "\x1b[36m", // cyan
    ];
    PALETTE[id.0 as usize % PALETTE.len()]
}

/// Whether the stderr drain should emit ANSI colors: `PROFLUENS_LOG_COLOR`
/// (`always`/`never`) wins, then the `NO_COLOR` convention (https://no-color.org),
/// then "is stderr actually a terminal" — so piping to a file stays clean bytes.
/// Public so every stderr drain (the pipeline's multi-channel one included) applies
/// one policy.
pub fn stderr_colors_enabled() -> bool {
    match std::env::var("PROFLUENS_LOG_COLOR").as_deref() {
        Ok("always") => return true,
        Ok("never") => return false,
        _ => {}
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    use std::io::IsTerminal;
    io::stderr().is_terminal()
}

/// [`format_record`] with optional ANSI styling: dim timestamp, level-colored level,
/// per-element hue on `name#id`, dim field keys. Styling is a drain-side concern —
/// the hot path only ever moves POD records.
pub fn format_record_styled(w: &mut impl Write, rec: &LogRecord, styled: bool) -> io::Result<()> {
    let ms = rec.ts.nanos().map(|n| n as f64 / 1e6).unwrap_or(f64::NAN);
    let name = if rec.name.is_empty() { "elem" } else { rec.name };
    if styled {
        write!(
            w,
            "{SGR_DIM}[{ms:>10.3}ms]{SGR_RESET} {}{:<5}{SGR_RESET} {}{name}#{}{SGR_RESET} {}",
            level_sgr(rec.level),
            rec.level.as_str(),
            element_sgr(rec.element),
            rec.element.0,
            rec.event
        )?;
        for f in rec.fields() {
            write!(w, " {SGR_DIM}{}={SGR_RESET}{}", f.key, f.val)?;
        }
    } else {
        write!(
            w,
            "[{ms:>10.3}ms] {:<5} {name}#{} {}",
            rec.level.as_str(),
            rec.element.0,
            rec.event
        )?;
        for f in rec.fields() {
            write!(w, " {}={}", f.key, f.val)?;
        }
    }
    writeln!(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ranks_round_trip() {
        for l in [Level::Error, Level::Warn, Level::Info, Level::Debug, Level::Trace] {
            assert_eq!(Level::from_rank(l.rank()), Some(l));
        }
        assert_eq!(Level::from_rank(0), None);
        assert_eq!(Level::from_name("WARN"), Some(Level::Warn));
        assert_eq!(Level::from_name("nonsense"), None);
    }

    #[test]
    fn filter_thresholds_are_inclusive_downward() {
        let f = LevelFilter::new(); // off
        assert!(!f.enabled(Level::Error));
        f.set(Some(Level::Info));
        assert!(f.enabled(Level::Error));
        assert!(f.enabled(Level::Info));
        assert!(!f.enabled(Level::Debug));
        assert!(!f.enabled(Level::Trace));
        f.set(Some(Level::Trace));
        assert!(f.enabled(Level::Trace));
        f.set(None);
        assert!(!f.enabled(Level::Error));
    }

    #[test]
    fn debug_spec_parsing() {
        assert_eq!(DebugSpec::parse("3").global, Some(3));
        assert_eq!(DebugSpec::parse("info").global, Some(Level::Info.rank()));
        assert_eq!(DebugSpec::parse("off").global, Some(0));

        let s = DebugSpec::parse(" filesrc:debug , 2 , http:trace ");
        assert_eq!(s.global, Some(2));
        assert_eq!(
            s.targets,
            vec![
                ("filesrc".to_string(), Level::Debug.rank()),
                ("http".to_string(), Level::Trace.rank()),
            ]
        );

        // Out-of-range numbers and unknown names are ignored, not clamped.
        let s = DebugSpec::parse("garbage:nonsense,,7");
        assert_eq!(s.global, None);
        assert!(s.targets.is_empty());
    }

    #[test]
    fn field_value_conversions() {
        assert!(matches!(FieldValue::from(-5i32), FieldValue::Int(-5)));
        assert!(matches!(FieldValue::from(5u64), FieldValue::Uint(5)));
        assert!(matches!(FieldValue::from(7usize), FieldValue::Uint(7)));
        assert!(matches!(FieldValue::from(true), FieldValue::Bool(true)));
        assert!(matches!(FieldValue::from("x"), FieldValue::Str("x")));
        assert!(matches!(FieldValue::from(ElementId(9)), FieldValue::Uint(9)));
    }

    #[test]
    fn format_line_is_greppable() {
        let mut rec = LogRecord {
            ts: Timestamp::from_millis(12),
            element: ElementId(3),
            name: "flacdec",
            level: Level::Info,
            event: "header_parsed",
            nfields: 2,
            fields: [
                Field::new("len", FieldValue::Uint(1024)),
                Field::new("codec", FieldValue::Str("flac")),
                EMPTY_FIELD,
                EMPTY_FIELD,
            ],
        };
        let mut buf = Vec::new();
        format_record(&mut buf, &rec).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("INFO"), "{s}");
        assert!(s.contains("flacdec#3"), "{s}");
        assert!(s.contains("header_parsed"), "{s}");
        assert!(s.contains("len=1024"), "{s}");
        assert!(s.contains("codec=flac"), "{s}");

        // Framework internals carry no name and fall back to the bare id.
        rec.name = "";
        let mut buf = Vec::new();
        format_record(&mut buf, &rec).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("elem#3"), "{s}");
    }

    #[test]
    fn disabled_level_is_silent_and_does_not_evaluate_args() {
        let filter = Arc::new(LevelFilter::new()); // off
        let (sink, drain) = log_channel(64);
        let log = Log::new(sink, filter);

        let mut calls = 0u32;
        let mut arg = || {
            calls += 1;
            42u64
        };
        log!(&log, Level::Error, "suppressed", v = arg());
        assert_eq!(calls, 0, "arguments must not be evaluated when the gate is closed");
        assert!(drain.try_next().is_none(), "nothing emitted");
    }

    #[test]
    fn enabled_level_emits_a_structured_record() {
        let filter = Arc::new(LevelFilter::with_level(Level::Warn));
        let (sink, drain) = log_channel(64);
        let mut log = Log::new(sink, filter);
        log.set_element(ElementId(2), "testsink");

        log!(&log, Level::Debug, "below", x = 1); // below threshold → suppressed
        log!(&log, Level::Error, "boom", code = 500u32, retry = true);

        let rec = drain.try_next().expect("error record present");
        assert_eq!(rec.event, "boom");
        assert_eq!(rec.element, ElementId(2));
        assert_eq!(rec.level, Level::Error);
        assert_eq!(rec.fields().len(), 2);
        assert_eq!(rec.fields()[0].key, "code");
        assert!(matches!(rec.fields()[0].val, FieldValue::Uint(500)));
        assert!(matches!(rec.fields()[1].val, FieldValue::Bool(true)));
        assert!(drain.try_next().is_none(), "debug record was suppressed");
    }

    #[test]
    fn records_round_trip_through_the_ring_in_order() {
        let filter = Arc::new(LevelFilter::with_level(Level::Trace));
        let (sink, drain) = log_channel(1024);
        let mut log = Log::new(sink, filter);
        log.set_element(ElementId(7), "testsrc");

        let n = 1000u64;
        let producer = std::thread::spawn(move || {
            for i in 0..n {
                log!(&log, Level::Info, "tick", i = i, half = i / 2);
            }
            // Dropping `log` here closes the ring, so the drain's blocking pop ends.
        });

        let mut got = Vec::new();
        while let Some(rec) = drain.next_blocking() {
            got.push(rec);
        }
        producer.join().unwrap();

        assert_eq!(got.len() as u64, n, "every record arrived exactly once");
        for (i, rec) in got.iter().enumerate() {
            assert_eq!(rec.event, "tick");
            assert_eq!(rec.element, ElementId(7));
            assert!(matches!(rec.fields()[0].val, FieldValue::Uint(v) if v == i as u64));
            assert!(matches!(rec.fields()[1].val, FieldValue::Uint(v) if v == i as u64 / 2));
        }
    }

    #[test]
    fn full_ring_drops_and_counts_every_lost_record() {
        let filter = Arc::new(LevelFilter::with_level(Level::Error));
        let (sink, drain) = log_channel(4); // tiny, and never drained during the burst
        let log = Log::new(sink, filter);

        let emitted = 100u64;
        for i in 0..emitted {
            log!(&log, Level::Error, "x", i = i);
        }

        let dropped = log.dropped();
        assert!(dropped > 0, "a 4-slot ring must drop under a 100-record burst");

        let mut kept = 0u64;
        while drain.try_next().is_some() {
            kept += 1;
        }
        assert_eq!(kept + dropped, emitted, "kept + dropped == emitted (no record vanishes)");
    }
}
