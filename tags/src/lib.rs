//! pf-tags — a parallel, zero-copy tag/metadata scanner.
//!
//! The engine behind a library scan: point it at ten thousand paths and it hands back
//! each file's format, duration, tags and cover art, **borrowed** — text out of a bump
//! arena, pictures as refcounted [`Memory`](profluens_core::memory::Memory) views into
//! the very buffer the bytes were read into. Nothing is copied that the caller does not
//! ask to keep, and the steady-state per-file path allocates nothing on the heap
//! (spec: performance #1 — no steady-state heap traffic).
//!
//! ## Shape
//!
//! ```no_run
//! use pf_tags::{ScanConfig, Scanner};
//! let mut sc = Scanner::new(ScanConfig::default());
//! // Anything `AsRef<Path>`: `String`, `&str`, `PathBuf`, `&Path`.
//! sc.scan(std::env::args().skip(1), |out| {
//!     match out.result {
//!         Ok(r) => println!("{} {:?} {:?}", out.path.display(), r.format, r.tags.get("TITLE")),
//!         Err(e) => eprintln!("{}: {e:?}", out.path.display()),
//!     }
//! });
//! ```
//!
//! One [`Scanner`] per thread. It owns a reactor, a [`Pool`](profluens_core::memory::Pool)
//! (shareable across threads — see [`Scanner::with_pool`]), a scratch
//! [`Arena`](profluens_core::memory::Arena), and a fixed array of state-machine slots.
//! [`scan_parallel`] fans a path list across N threads over one shared pool.
//!
//! ## Why a reactor and not a thread pool doing `read()`
//!
//! A library scan is a few small positioned reads per file across thousands of files:
//! latency-bound, not bandwidth-bound. Keeping `in_flight` reads outstanding at once and
//! reaping completions is exactly the shape the framework's [`Reactor`](profluens_core::io::Reactor) contract exists
//! for (spec: IO — built for the io_uring era), and with the `io-uring` feature the same
//! engine issues them as real async ops without changing a line of the state machine.
//!
//! ## Blocking IO, and where it is sanctioned
//!
//! `open(2)` and `stat(2)` have no reactor op — [`Submission`](profluens_core::io::Submission)
//! covers `Read`/`Write`/`Recv` only — so opening a file and reading its length/mtime are
//! blocking `std::fs` calls. That is the workspace's documented app-side exception (see
//! `clippy.toml`'s header and `play/src/head.rs`): this is a standalone application-side
//! engine, not an `Element` inside a scheduler group, so there is no group to hold hostage.
//! Every such call carries an item-level `#[allow(clippy::disallowed_methods)]` and a
//! one-line justification. All *data* — every byte parsed — rides the reactor.
//!
//! ## Format coverage
//!
//! | [`Format`] | tags | duration | rate / channels | reads for a typical file |
//! |---|---|---|---|---|
//! | `Wav` | `LIST`/`INFO` | exact (`data` size ÷ `nAvgBytesPerSec`) | `fmt ` | 1 (2 when the tagger appended `INFO` after `data`) |
//! | `Aiff` | `ID3 ` chunk, then `NAME`/`AUTH`/`ANNO`/`(c) ` | exact (`COMM` sample-frame count ÷ the 80-bit rate) — AIFF-C too | `COMM` | 1 (2 when a tag chunk sits behind `SSND`) |
//! | `Flac` | `VORBIS_COMMENT` + `PICTURE` | exact (STREAMINFO sample count) | STREAMINFO | 1 (2 with cover art past the prefix) |
//! | `Mp3` | ID3v2 (+ `APIC`), APEv2, ID3v1 | exact with Xing/VBRI, else a CBR estimate — [`Props::duration_exact`] says which | first frame header | 2 (3 with an ID3v2 tag past the prefix) |
//! | `Mp4` | iTunes `ilst` (`©nam`, `trkn`, `covr`, `----` freeform) | exact (`mdhd`, else `mvhd`) | `stsd` sample entry | 1 `faststart`, 2 with a trailing `moov` |
//! | `Ape` | APEv2, ID3v1 | exact (total blocks ÷ rate) | `APE_HEADER`, or the derived pre-3.98 one | 2 |
//! | `WavPack` | APEv2, ID3v1 | exact (`total_samples` ÷ rate); none when the file declares an unknown length | block header flags, `ID_CHANNEL_INFO`/`ID_SAMPLE_RATE` sub-blocks | 2 |
//! | `Musepack` | APEv2, ID3v1 | SV8 exact (sample count − beginning silence); SV7 exact when `TrueGapless`, else ±1 frame | SV8 `SH` packet; SV7 fixed header (always stereo) | 2 |
//! | `OggOpus` | `OpusTags` | exact (last granule − `pre_skip`, at 48 kHz) | `OpusHead` channels; rate reported as the 48 kHz decode rate | 2 (3 with cover art past the prefix) |
//! | `OggVorbis` | `\x03vorbis` comments | exact (last granule ÷ rate) | identification header | 2 |
//! | `OggFlac` | `VORBIS_COMMENT` + `PICTURE` | exact (last granule ÷ rate) | STREAMINFO | 2 |
//! | `OggSpeex` | a bare Vorbis comment — no magic, no framing bit | exact (last granule ÷ rate); 5–11 ms under `ffprobe`, which adds the encoder lookahead back | `"Speex   "` header | 2 |
//! | `Mkv` | `\Segment\Tags` + cover art from `\Segment\Attachments` | exact (`Info` `Duration` × `TimestampScale`) | first audio `TrackEntry`'s `Audio` master | 1; 2 when a muxer wrote `Tags`/`Attachments` behind the frames |
//! | `Unknown` | — | — | — | 1 |
//!
//! "Reads" counts reactor ops, not bytes: the prefix, any positioned metadata extent, and
//! the one tail read the formats that keep metadata at *both* ends need. A file smaller than
//! the prefix never costs a tail read — the prefix already is one.
//!
//! Four of these formats — MP3, Monkey's Audio, WavPack and Musepack — end with the same
//! APEv2-then-ID3v1 pair, measured and parsed once in `tail.rs` rather than three more times.
//!
//! Each of the hand-written parsers cites both its external specification and an in-tree copy
//! or field table under `spec/` (`AIFF.md`, `APE.md`, `WAVPACK.md`, `MPC.md`, plus the
//! vendored WavPack, Monkey's Audio and libmpc documents); Speex's lives in `ogg/spec/`.
//!
//! Matroska is the one format whose metadata is not all in one place: `Info` and `Tracks` sit
//! ahead of the frames while `Tags` and `Attachments` may be written behind them. It is also
//! the only one that never needs a *search* for them anyway — the Segment's front `SeekHead`
//! (RFC 9559 §5.1.1) names each master's position, so the walk that finds a trailing tag block
//! in an 80 MB file reads a few hundred bytes of index and then exactly one range.
//!
//! Recognised-only formats still report their [`Format`], length and mtime; they simply carry
//! no tags. One known gap, deliberate:
//!
//! * **Bare ADTS AAC** (`.aac`) — sniffs as [`Format::Unknown`] on purpose: its sync word is
//!   the same 12 bits as MPEG audio's (ISO/IEC 14496-3 §1.A.2.2.1), so claiming it as MP3
//!   would report a wrong duration from a wrong frame table. ADTS carries no tags of its own;
//!   what a file like that needs is its own props probe.
//!
//! Every magic number and structure below cites its specification at the point of use
//! (workspace standing rule). Parsing is total: malformed, truncated or hostile input
//! yields fewer tags, never a panic.

#![deny(unsafe_code)]
#![feature(allocator_api)]

pub mod fixture;
mod aiff;
mod ape;
mod flac;
mod mkv;
mod mp3;
mod mp4;
mod mpc;
mod ogg;
mod parallel;
mod scanner;
mod sniff;
mod store;
mod tail;
mod wav;
mod wv;

pub use parallel::{scan_parallel, scan_parallel_with};
pub use scanner::{ReactorKind, Scanner};
pub use sniff::Format;
pub use store::{PictureRef, Tags};

/// Default prefix-read size, and the default pool's slot size: the head of every file is
/// read in one op of this many bytes. 128 KiB covers a WAV header, an entire FLAC
/// metadata chain without cover art, and an ID3v2 tag of any sane size, so the common
/// file needs exactly one read (spec: IO — one op per file is the whole point).
pub const DEFAULT_PREFIX: usize = 128 * 1024;

/// Hard ceiling on concurrent in-flight files, and therefore on outstanding reactor ops.
///
/// Set by the io_uring backend, not by taste: `elements/src/io/uring.rs:248-257` sizes the
/// submission queue at 64 entries and deliberately does **not** track the kernel-updated SQ
/// head ("the ring (64 entries) vastly exceeds our in-flight count, so it never fills").
/// Submitting more than 64 SQEs between two `io_uring_enter` calls would therefore wrap the
/// ring and silently overwrite live entries. 32 keeps a 2x margin under that bound even if a
/// future engine pass issues two ops per file per pass, and it is far past the point where
/// more queue depth buys anything on NVMe.
pub const MAX_IN_FLIGHT: usize = 32;

/// How the scan is tuned. [`Default`] is the sane library-scan configuration.
#[derive(Clone, Copy, Debug)]
pub struct ScanConfig {
    /// Files (and therefore reactor ops) in flight at once. Clamped to `1..=`[`MAX_IN_FLIGHT`].
    pub in_flight: usize,
    /// How many bytes to read when a format's metadata sits at the end of the file (a WAV
    /// `LIST`/`INFO` chunk after `data`, an ID3v1 trailer). One op, from the last chunk
    /// boundary or the end of the file.
    pub tail_len: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self { in_flight: 16, tail_len: 64 * 1024 }
    }
}

impl ScanConfig {
    /// `in_flight`, clamped to the range the io_uring backend can actually carry
    /// (see [`MAX_IN_FLIGHT`]).
    pub fn slots(&self) -> usize {
        self.in_flight.clamp(1, MAX_IN_FLIGHT)
    }
}

/// What went wrong with one path. Everything else — a truncated header, a tag block that
/// runs off the end of the file, bytes that are not audio at all — is *not* an error: the
/// scan reports [`Format::Unknown`] or fewer tags. Only the filesystem can fail a file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScanError {
    /// `open(2)` or `stat(2)` failed: no such file, no permission, not a regular file.
    Open(std::io::ErrorKind),
    /// A read failed after the file was opened (EIO on a dying disk, a vanished network mount).
    Io(std::io::ErrorKind),
    /// The op was cancelled, or the reactor stopped making progress with ops outstanding.
    Cancelled,
}

/// One file's outcome, handed to the scan callback. Borrowed for the duration of the call
/// only: the arena behind [`Tags`] is reset the moment it returns (a caller that wants to
/// keep tags calls [`Tags::to_tag_list`]).
pub struct ScanOutcome<'a> {
    pub path: &'a std::path::Path,
    pub result: Result<ScanResult<'a>, ScanError>,
}

/// Everything the scan learned about one file.
pub struct ScanResult<'a> {
    pub format: Format,
    pub file: FileMeta,
    pub props: Props,
    pub tags: Tags<'a>,
}

/// Filesystem facts, from the one `stat(2)` the scan does per file. A library indexer keys
/// its "unchanged since last scan" check on exactly this pair.
#[derive(Clone, Copy, Debug)]
pub struct FileMeta {
    pub len: u64,
    pub mtime: std::time::SystemTime,
}

/// Stream properties, as far as the metadata (never the audio data) reveals them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Props {
    /// Playing time in nanoseconds.
    pub duration_ns: Option<u64>,
    /// Whether `duration_ns` came from an authoritative sample count (FLAC STREAMINFO, a
    /// WAV `data` chunk size) rather than a bitrate estimate. Estimated durations are what
    /// makes a VBR MP3's reported length wrong; the caller deserves to know which it got.
    pub duration_exact: bool,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
}
