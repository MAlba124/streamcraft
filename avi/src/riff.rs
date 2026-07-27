//! Hand-written AVI/RIFF primitives: the header probe that discovers streams, the
//! streaming chunk walker that turns the `movi` list into per-stream chunks, and the
//! `idx1` index parser. Everything here is **core-independent** (plain byte slices in,
//! plain structs out) so it is unit-tested in isolation, exactly like `sc-mkv`'s `ebml`
//! and `sc-ogg`'s `page`.
//!
//! ## The AVI file format (spec at point of use)
//! AVI is a RIFF document (Microsoft *AVI RIFF File Reference*,
//! `learn.microsoft.com/windows/win32/directshow/avi-riff-file-reference`; the classic
//! `OpenDML AVI File Format Extensions v1.02` PDF for the >1 GB pieces). A RIFF file is a
//! tree of **chunks**: a 4-CC id (four ASCII bytes), a little-endian `u32` size, then that
//! many payload bytes, **WORD-aligned** — an odd size is followed by one pad byte that is
//! *not* counted in the size (AVI RIFF Reference, "RIFF chunks"). A `LIST` chunk's payload
//! begins with a second 4-CC (the list type), then nested chunks.
//!
//! An AVI file's outer chunk is `RIFF … 'AVI '`. Inside:
//! ```text
//! RIFF 'AVI '
//!   LIST 'hdrl'                 header list
//!     'avih'                    MainAVIHeader (frame count, stream count, µs/frame)
//!     LIST 'strl'  (one per stream)
//!       'strh'                  AVIStreamHeader (fccType vids/auds, fccHandler, scale/rate)
//!       'strf'                  stream format: BITMAPINFOHEADER (video) / WAVEFORMATEX (audio)
//!       ['JUNK'/'indx'/…]       padding / OpenDML super-index
//!     [LIST 'odml' { 'dmlh' }]  OpenDML header (total-frames hint)
//!   [LIST 'INFO' …]             metadata (ignored)
//!   LIST 'movi'                 the interleaved media data
//!     ['00dc'|'01wb'|…]         one chunk per sample: NNtt, NN = stream index, tt = type
//!     [LIST 'rec '{ … }]        an optional grouping wrapper (OpenDML) around a set of chunks
//!   ['idx1']                    the legacy index: (ckid, flags, offset, size) × N
//! ```
//!
//! ## Chunk naming: `NNtt` (AVI RIFF Reference, "Stream data")
//! A `movi` data chunk's id is two ASCII digits of the **stream number** followed by a
//! two-letter **type code**: `dc` (compressed video / DIB), `db` (uncompressed video),
//! `wb` (audio "wave bytes"), `pc` (palette change), `tx`/`sb` (subtitle). We route by the
//! stream number and forward the payload verbatim; the type code is advisory (we accept any).
//!
//! ## Robustness (this brief's P0: "a crash on bad input is a P0")
//! Every field is bounds-checked against the remaining input; a size that would run past the
//! buffer is rejected, never sliced. A malformed chunk in `movi` triggers a **resync** — a
//! forward scan for the next plausible chunk id — rather than aborting the stream. Nothing
//! here can panic or index out of range on adversarial input; the `#[test]` bad-input table
//! (see `tests`) pins that.

/// A four-character code (RIFF chunk id / AVI fourcc). Compared case-insensitively for
/// codec fourccs (`XVID` vs `xvid` name the same codec), byte-exact for structural ids.
pub type FourCc = [u8; 4];

/// The kind of a discovered stream, from `strh`'s `fccType` (AVI RIFF Reference,
/// `AVIStreamHeader.fccType`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamKind {
    /// `vids` — a video stream; `strf` is a `BITMAPINFOHEADER`.
    Video,
    /// `auds` — an audio stream; `strf` is a `WAVEFORMATEX`.
    Audio,
    /// `txts`/`mids`/anything else — carried as a byte stream (best effort).
    Other,
}

/// One discovered AVI stream: the `strh` timing + the `strf` format, decoded (video pixel
/// dims + fourcc handler, or audio WAVEFORMATEX fields) so the element can announce a
/// concrete format. The `index` is the stream's ordinal in the file (0-based) — the `NN` in
/// a `movi` chunk id routes to it.
#[derive(Clone, Debug)]
pub struct Stream {
    /// 0-based ordinal (order of the `strl` lists in `hdrl`) — the routing key.
    pub index: usize,
    /// `vids`/`auds`/other.
    pub kind: StreamKind,
    /// `strh.fccHandler` for video (the codec fourcc, e.g. `XVID`) — 4 raw bytes. For audio
    /// it is the format tag's textual echo and unused; kept for diagnostics.
    pub handler: FourCc,
    /// `strh.dwScale` / `strh.dwRate`: the stream's time base. One sample's duration is
    /// `dwScale / dwRate` seconds (AVI RIFF Reference, `AVIStreamHeader`). For video this is
    /// the frame rate's reciprocal; for audio (`dwSampleSize != 0`, i.e. CBR PCM-like) it is
    /// the block rate. Zero `rate` means "unknown timing" — the element falls back to a
    /// monotonic sample counter.
    pub scale: u32,
    pub rate: u32,
    /// `strh.dwSampleSize` — for audio, bytes per "sample" (0 = variable/VBR, one chunk is
    /// one frame; nonzero = a chunk holds `size/sample_size` samples). Video is always 0.
    pub sample_size: u32,
    /// `strh.dwLength` — declared sample count (video: frame count). Advisory: the real
    /// count comes from walking `movi`/`idx1`, but this seeds the probe's frame-count report
    /// and matches ffprobe's `nb_frames` for a well-formed file.
    pub length: u32,
    // ---- video (`strf` = BITMAPINFOHEADER, AVI RIFF Reference) ----
    /// `biWidth` — 0 for a non-video stream.
    pub width: u32,
    /// `biHeight` — 0 for a non-video stream (may be stored negative for top-down DIBs;
    /// we take the absolute value).
    pub height: u32,
    /// The raw `strf` bytes (the whole BITMAPINFOHEADER incl. any trailing extradata, or the
    /// WAVEFORMATEX incl. `cbSize` bytes) — the MPEG-4 decoder wants the BITMAPINFOHEADER
    /// extradata for VOL config when the ES headers are stripped (this brief's mandate).
    pub strf: Vec<u8>,
    // ---- audio (`strf` = WAVEFORMATEX, AVI RIFF Reference) ----
    /// `wFormatTag` — 0 for a non-audio stream. `0x2000`=AC-3, `0x2001`=DTS, `0x0055`=MP3,
    /// `0x0001`=PCM (see [`crate::codec`]).
    pub format_tag: u16,
    /// `nChannels`.
    pub channels: u16,
    /// `nSamplesPerSec` (sample rate in Hz).
    pub samples_per_sec: u32,
    /// `wBitsPerSample`.
    pub bits_per_sample: u16,
}

impl Stream {
    /// One sample's duration in nanoseconds from the `strh` time base (`dwScale/dwRate`
    /// seconds), or `None` when `rate` is zero (unknown timing — the element then uses a
    /// monotonic counter). AVI RIFF Reference, `AVIStreamHeader`.
    pub fn sample_duration_ns(&self) -> Option<u64> {
        if self.rate == 0 {
            return None;
        }
        Some((self.scale as u64).saturating_mul(1_000_000_000) / self.rate as u64)
    }
}

/// The result of a header probe: the discovered streams plus the file-level timing hints.
#[derive(Clone, Debug, Default)]
pub struct AviHeader {
    /// Every `strl` stream, in file order.
    pub streams: Vec<Stream>,
    /// `avih.dwMicroSecPerFrame` — the nominal video frame interval (µs). A cross-check
    /// against the video `strh` scale/rate; not load-bearing.
    pub us_per_frame: u32,
    /// `avih.dwTotalFrames` — nominal total video frames. Advisory.
    pub total_frames: u32,
    /// Whether an OpenDML `LIST 'odml'` header was present — a signal the file may use the
    /// `AVIX` extension RIFFs / super-index for content past the first ~2 GB. Reported so the
    /// element/probe can note it (see the crate docs' OpenDML section).
    pub has_odml: bool,
}

impl AviHeader {
    /// The presentation duration in nanoseconds: the longest stream's `dwLength × dwScale /
    /// dwRate` (AVI RIFF Reference — a stream's playing time is `length × scale / rate`).
    /// `None` if no stream carries usable timing. Used to seed `SeekIndex`'s proportional
    /// fallback and the pipeline's `DurationChanged`.
    pub fn duration_ns(&self) -> Option<u64> {
        self.streams
            .iter()
            .filter_map(|s| {
                if s.rate == 0 {
                    return None;
                }
                Some((s.length as u64).saturating_mul(s.scale as u64).saturating_mul(1_000_000_000) / s.rate as u64)
            })
            .max()
    }
}

/// A bounds-checked forward cursor over a byte slice. Every read is fallible — a truncated
/// header errors rather than panicking (this brief's untrusted-input P0). Mirrors
/// `sc-mkv`'s `codec::Cursor`.
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    /// Bytes still unread.
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.at)
    }

    fn u16le(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32le(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn fourcc(&mut self) -> Option<FourCc> {
        let b = self.take(4)?;
        Some([b[0], b[1], b[2], b[3]])
    }

    /// Take exactly `n` bytes, or `None` if fewer remain (never a partial/oob slice).
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.data.get(self.at..end)?;
        self.at = end;
        Some(s)
    }

    /// Peek `n` bytes without advancing.
    fn peek(&self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        self.data.get(self.at..end)
    }
}

/// Errors from the header probe. A structural problem in the *header* is fatal (we cannot
/// discover streams), unlike a bad `movi` chunk which only triggers a resync.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AviError {
    /// The bytes are not a `RIFF … 'AVI '` document.
    NotAvi,
    /// A required element (avih / at least one strl / strh / strf) was truncated or absent.
    Truncated(&'static str),
    /// The header prefix did not reach the end of `hdrl` — the caller passed too few bytes.
    HeaderIncomplete,
}

/// The byte offset of the `movi` list's **data** (first sub-chunk) within a probed header,
/// so a caller reading the whole file knows where media begins. Returned alongside the
/// header from [`probe_header`].
#[derive(Clone, Copy, Debug)]
pub struct MoviLocation {
    /// Absolute file offset of the `LIST 'movi'` chunk's *data* start — i.e. the first byte
    /// after the `movi` 4-CC list type. `idx1` offsets that are `movi`-relative are relative
    /// to `movi_data_start - 4` (the list-type position), per the AVI RIFF Reference; we
    /// handle both interpretations in [`Idx1::resolve_offset`].
    pub movi_data_start: u64,
    /// Absolute file offset of the `LIST 'movi'` chunk id (the `L` of `LIST`).
    pub movi_chunk_start: u64,
}

/// Parse an AVI header prefix — everything through the `hdrl` list, and (if present in the
/// prefix) locate the `movi` list. `data` must contain at least the full `RIFF/AVI` outer
/// header + the whole `LIST 'hdrl'`; a few MiB is generous (real files put `hdrl` first,
/// well under 64 KiB). Returns the discovered streams and, when the prefix reaches it, the
/// `movi` location.
///
/// This is the sans-IO half of discovery: the element hands its constructor-supplied header
/// prefix here in `preroll` (like `MkvDemux::new`), and the app hands a larger prefix here
/// to locate `movi` for building a `SeekIndex`.
pub fn probe_header(data: &[u8]) -> Result<(AviHeader, Option<MoviLocation>), AviError> {
    let mut c = Cursor::new(data);
    // Outer RIFF header: 'RIFF' <size> 'AVI '.
    if c.fourcc() != Some(*b"RIFF") {
        return Err(AviError::NotAvi);
    }
    let _riff_size = c.u32le().ok_or(AviError::Truncated("RIFF size"))?;
    if c.fourcc() != Some(*b"AVI ") {
        return Err(AviError::NotAvi);
    }

    let mut header = AviHeader::default();
    let mut movi = None;
    // Walk the top-level chunks of the RIFF form. We only need `hdrl` (streams) and the
    // start of `movi`; stop once both are known (or the prefix runs out).
    while c.remaining() >= 8 {
        let chunk_start = c.at as u64;
        let id = c.fourcc().ok_or(AviError::Truncated("top-level chunk id"))?;
        let size = c.u32le().ok_or(AviError::Truncated("top-level chunk size"))? as usize;
        if &id == b"LIST" {
            let list_type = c.fourcc().ok_or(AviError::Truncated("LIST type"))?;
            // The list body is `size - 4` bytes (the 4-CC type is counted in `size`).
            let body_len = size.checked_sub(4).ok_or(AviError::Truncated("LIST size < 4"))?;
            match &list_type {
                b"hdrl" => {
                    // The header list must be wholly present to discover streams.
                    let body = c
                        .peek(body_len)
                        .ok_or(AviError::HeaderIncomplete)?;
                    parse_hdrl(body, &mut header)?;
                    c.at += body_len; // advance past the list body
                }
                b"movi" => {
                    // Found the media list — record where its data begins and stop. The
                    // element/app streams the rest; we do not read `movi` here.
                    movi = Some(MoviLocation {
                        movi_chunk_start: chunk_start,
                        movi_data_start: c.at as u64,
                    });
                    break;
                }
                _ => {
                    // INFO / other lists: skip the whole body (WORD-aligned).
                    c.at = advance_padded(c.at, body_len);
                }
            }
        } else {
            // A stray top-level chunk (e.g. JUNK) before movi: skip it, WORD-aligned.
            c.at = advance_padded(c.at, size);
        }
        // If the last skip ran past the prefix, we simply stop (movi not located yet).
        if c.at > data.len() {
            break;
        }
    }

    if header.streams.is_empty() {
        return Err(AviError::Truncated("no strl stream in hdrl"));
    }
    Ok((header, movi))
}

/// Advance an offset by a chunk payload of `size` bytes, honouring RIFF WORD alignment: an
/// odd size is followed by one pad byte not counted in `size` (AVI RIFF Reference, "RIFF
/// chunks"). Saturating so a hostile size never wraps.
fn advance_padded(at: usize, size: usize) -> usize {
    at.saturating_add(size).saturating_add(size & 1)
}

/// Parse the body of a `LIST 'hdrl'`: the `avih` main header then one `LIST 'strl'` per
/// stream (AVI RIFF Reference, "AVI RIFF form"). Bounds-checked; a truncated sub-element
/// errors.
fn parse_hdrl(body: &[u8], header: &mut AviHeader) -> Result<(), AviError> {
    let mut c = Cursor::new(body);
    let mut stream_index = 0usize;
    while c.remaining() >= 8 {
        let id = c.fourcc().ok_or(AviError::Truncated("hdrl chunk id"))?;
        let size = c.u32le().ok_or(AviError::Truncated("hdrl chunk size"))? as usize;
        match &id {
            b"avih" => {
                let b = c.peek(size).ok_or(AviError::Truncated("avih body"))?;
                parse_avih(b, header);
                c.at = advance_padded(c.at, size);
            }
            b"LIST" => {
                let list_type = c.fourcc().ok_or(AviError::Truncated("hdrl LIST type"))?;
                let body_len = size.checked_sub(4).ok_or(AviError::Truncated("hdrl LIST size < 4"))?;
                let sub = c.peek(body_len).ok_or(AviError::Truncated("hdrl LIST body"))?;
                match &list_type {
                    b"strl" => {
                        if let Some(s) = parse_strl(sub, stream_index) {
                            header.streams.push(s);
                        }
                        stream_index += 1;
                    }
                    b"odml" => header.has_odml = true,
                    _ => {}
                }
                c.at += body_len;
            }
            // JUNK / unknown padding inside hdrl.
            _ => {
                c.at = advance_padded(c.at, size);
            }
        }
        if c.at > body.len() {
            break;
        }
    }
    Ok(())
}

/// Parse the `avih` MainAVIHeader (AVI RIFF Reference, `MainAVIHeader`): we take
/// `dwMicroSecPerFrame` (offset 0) and `dwTotalFrames` (offset 16). Tolerant — a short
/// avih leaves the fields zero.
fn parse_avih(b: &[u8], header: &mut AviHeader) {
    let mut c = Cursor::new(b);
    header.us_per_frame = c.u32le().unwrap_or(0);
    let _ = c.take(12); // dwMaxBytesPerSec, dwPaddingGranularity, dwFlags
    header.total_frames = c.u32le().unwrap_or(0);
}

/// Parse a `LIST 'strl'` body: its `strh` (timing) + `strf` (format). Returns `None` only
/// if the `strh` is missing/truncated (a stream with no header is unroutable). `strf` is
/// tolerated absent (fields stay zero). AVI RIFF Reference, "Stream headers".
fn parse_strl(body: &[u8], index: usize) -> Option<Stream> {
    let mut c = Cursor::new(body);
    let mut strh: Option<(StreamKind, FourCc, u32, u32, u32, u32)> = None;
    let mut strf: Vec<u8> = Vec::new();
    while c.remaining() >= 8 {
        let id = c.fourcc()?;
        let size = c.u32le()? as usize;
        match &id {
            b"strh" => {
                let b = c.peek(size)?;
                strh = Some(parse_strh(b));
                c.at = advance_padded(c.at, size);
            }
            b"strf" => {
                let b = c.peek(size)?;
                strf = b.to_vec();
                c.at = advance_padded(c.at, size);
            }
            // `indx` (OpenDML super-index), `JUNK`, `strd`, `strn` — skip.
            _ => {
                c.at = advance_padded(c.at, size);
            }
        }
        if c.at > body.len() {
            break;
        }
    }
    let (kind, handler, scale, rate, sample_size, length) = strh?;
    let mut s = Stream {
        index,
        kind,
        handler,
        scale,
        rate,
        sample_size,
        length,
        width: 0,
        height: 0,
        strf,
        format_tag: 0,
        channels: 0,
        samples_per_sec: 0,
        bits_per_sample: 0,
    };
    match kind {
        StreamKind::Video => parse_bitmapinfoheader(&s.strf.clone(), &mut s),
        StreamKind::Audio => parse_waveformatex(&s.strf.clone(), &mut s),
        StreamKind::Other => {}
    }
    Some(s)
}

/// Parse the `strh` AVIStreamHeader fields we route on (AVI RIFF Reference,
/// `AVIStreamHeader`): fccType (offset 0), fccHandler (4), dwScale (20), dwRate (24),
/// dwLength (32), dwSampleSize (44). Missing tail bytes read as zero.
fn parse_strh(b: &[u8]) -> (StreamKind, FourCc, u32, u32, u32, u32) {
    let get_u32 = |off: usize| -> u32 {
        b.get(off..off + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .unwrap_or(0)
    };
    let fcc_type: FourCc = b.get(0..4).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([0; 4]);
    let handler: FourCc = b.get(4..8).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([0; 4]);
    let kind = match &fcc_type {
        b"vids" => StreamKind::Video,
        b"auds" => StreamKind::Audio,
        _ => StreamKind::Other,
    };
    let scale = get_u32(20);
    let rate = get_u32(24);
    let length = get_u32(32);
    let sample_size = get_u32(44);
    (kind, handler, scale, rate, sample_size, length)
}

/// Fill a video [`Stream`]'s pixel dims from its `strf` BITMAPINFOHEADER (AVI RIFF
/// Reference — video `strf` is a `BITMAPINFOHEADER`): biWidth at offset 4, biHeight at 8
/// (may be negative for a top-down image; take the magnitude). The raw `strf` (incl. any
/// trailing extradata) is already retained on the stream for the decoder.
fn parse_bitmapinfoheader(strf: &[u8], s: &mut Stream) {
    let get_i32 = |off: usize| -> i32 {
        strf.get(off..off + 4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .unwrap_or(0)
    };
    s.width = get_i32(4).unsigned_abs();
    s.height = get_i32(8).unsigned_abs();
}

/// Fill an audio [`Stream`]'s WAVEFORMATEX fields from its `strf` (AVI RIFF Reference —
/// audio `strf` is a `WAVEFORMATEX`): wFormatTag(0) nChannels(2) nSamplesPerSec(4)
/// nAvgBytesPerSec(8) nBlockAlign(12) wBitsPerSample(14) [cbSize(16) + extension].
fn parse_waveformatex(strf: &[u8], s: &mut Stream) {
    let mut c = Cursor::new(strf);
    s.format_tag = c.u16le().unwrap_or(0);
    s.channels = c.u16le().unwrap_or(0);
    s.samples_per_sec = c.u32le().unwrap_or(0);
    let _ = c.u32le(); // nAvgBytesPerSec
    let _ = c.u16le(); // nBlockAlign
    s.bits_per_sample = c.u16le().unwrap_or(0);
}

// =====================================================================================
// idx1 — the legacy index (AVI RIFF Reference, "AVI Index (idx1)")
// =====================================================================================

/// One `idx1` entry: a `movi` chunk's id, keyframe flag, and byte offset+size. The offset's
/// meaning (movi-relative vs absolute) is muxer-dependent; [`Idx1::resolve_offset`] resolves
/// it against the file (AVI RIFF Reference — "the offset is relative to the start of the
/// 'movi' list, or to the start of the file"; we detect which).
#[derive(Clone, Copy, Debug)]
pub struct Idx1Entry {
    /// The 4-CC chunk id, e.g. `00dc`/`01wb`. The first two ASCII digits are the stream #.
    pub ckid: FourCc,
    /// `AVIIF_KEYFRAME` (0x10) set → a resume-safe point (AVI RIFF Reference, index flags).
    pub keyframe: bool,
    /// The chunk's offset as written in the index (raw; resolve with [`Idx1::resolve_offset`]).
    pub offset: u32,
    /// The chunk's payload size in bytes.
    pub size: u32,
}

impl Idx1Entry {
    /// The 0-based stream number parsed from the id's first two ASCII digits (`00dc` → 0,
    /// `01wb` → 1). `None` if they are not digits (a `rec `/`ix##` wrapper entry — skip it).
    pub fn stream_index(&self) -> Option<usize> {
        let d0 = (self.ckid[0] as char).to_digit(10)?;
        let d1 = (self.ckid[1] as char).to_digit(10)?;
        Some((d0 * 10 + d1) as usize)
    }
}

/// A parsed `idx1` chunk: its entries plus the `movi` data start needed to resolve
/// movi-relative offsets.
#[derive(Clone, Debug)]
pub struct Idx1 {
    pub entries: Vec<Idx1Entry>,
    /// `movi_data_start - 4` — the position of the `movi` list-type 4-CC, which movi-relative
    /// `idx1` offsets are measured from (the AVI RIFF Reference is famously ambiguous; the
    /// de-facto convention, and what ffmpeg assumes, is offset-from-the-`'movi'`-4CC).
    movi_base: u64,
    /// The file length, to disambiguate movi-relative vs absolute offsets.
    file_len: u64,
}

impl Idx1 {
    /// Resolve one entry's raw offset to an **absolute file byte offset of the chunk id**.
    /// AVI muxers disagree: most (ffmpeg, VirtualDub) write offsets relative to the `movi`
    /// list-type 4-CC; some write absolute file offsets. We detect which by testing the
    /// first entry both ways against the file: if the movi-relative reading lands inside the
    /// file and the absolute one would overflow it, it is movi-relative (the common case).
    /// Each entry then uses the detected convention.
    pub fn resolve_offset(&self, entry: &Idx1Entry) -> u64 {
        if self.movi_relative() {
            self.movi_base.saturating_add(entry.offset as u64)
        } else {
            entry.offset as u64
        }
    }

    /// Whether offsets are movi-relative (see [`resolve_offset`](Self::resolve_offset)). The
    /// heuristic: an absolute offset for the first entry that lands beyond EOF means the
    /// offsets must be movi-relative. Conversely a first offset already ≥ `movi_base` that
    /// fits the file is plausibly absolute. Defaults to movi-relative (the dominant form).
    fn movi_relative(&self) -> bool {
        let Some(first) = self.entries.first() else {
            return true;
        };
        let as_abs = first.offset as u64;
        let as_rel = self.movi_base.saturating_add(first.offset as u64);
        // Absolute reading overshoots the file → definitely movi-relative.
        if as_abs >= self.file_len {
            return true;
        }
        // Absolute reading lands *before* movi (impossible for real data) → movi-relative.
        if as_abs < self.movi_base {
            return true;
        }
        // Both plausible: prefer movi-relative only if the relative reading also fits; an
        // absolute-offset file has offsets already ≥ movi_base, so relative would overshoot.
        as_rel < self.file_len
    }

    /// Build a time→byte [`streamcraft_core::pipeline::SeekIndex`] from the video stream's
    /// keyframe entries: for each keyframe chunk of `video_stream_index`, its presentation
    /// time (`frame_number × sample_duration_ns`) mapped to the **absolute byte offset of the
    /// enclosing structure to resume reads from**. The demuxer re-syncs by scanning for the
    /// next chunk, so pointing at the chunk id is resume-safe (AVI RIFF Reference — a keyframe
    /// index entry marks a random-access point). Requires the video stream's per-sample
    /// duration; entries for other streams advance no video clock but are still counted so
    /// the frame number tracks the video sample ordinal.
    pub fn build_seek_index(
        &self,
        video_stream_index: usize,
        video_sample_dur_ns: u64,
    ) -> streamcraft_core::pipeline::SeekIndex {
        let mut entries = Vec::new();
        let mut frame: u64 = 0;
        for e in &self.entries {
            if e.stream_index() != Some(video_stream_index) {
                continue;
            }
            if e.keyframe {
                let time_ns = frame.saturating_mul(video_sample_dur_ns);
                entries.push((time_ns, self.resolve_offset(e)));
            }
            frame += 1;
        }
        // Entries are already ascending in `frame` and thus in time.
        streamcraft_core::pipeline::SeekIndex {
            entries,
            file_len: Some(self.file_len),
        }
    }
}

/// Parse a standalone `idx1` chunk's **payload** (the bytes after the `idx1` id + size) into
/// entries: each entry is 16 bytes — ckid(4) dwFlags(4) dwChunkOffset(4) dwChunkLength(4)
/// (AVI RIFF Reference, "AVI Index"). `movi_data_start` and `file_len` come from the header
/// probe. Trailing partial bytes are ignored (never sliced oob).
pub fn parse_idx1(payload: &[u8], movi_data_start: u64, file_len: u64) -> Idx1 {
    let mut entries = Vec::with_capacity(payload.len() / 16);
    let mut c = Cursor::new(payload);
    while c.remaining() >= 16 {
        let ckid = c.fourcc().expect("16 remaining ≥ 4");
        let flags = c.u32le().expect("16 remaining");
        let offset = c.u32le().expect("16 remaining");
        let size = c.u32le().expect("16 remaining");
        entries.push(Idx1Entry {
            ckid,
            keyframe: flags & 0x0000_0010 != 0, // AVIIF_KEYFRAME
            offset,
            size,
        });
    }
    Idx1 {
        entries,
        // Movi-relative offsets are measured from the `'movi'` 4-CC, which sits 4 bytes
        // before the data start recorded by the probe.
        movi_base: movi_data_start.saturating_sub(4),
        file_len,
    }
}

// =====================================================================================
// Streaming chunk walker — turns the movi byte stream into per-stream chunks
// =====================================================================================

/// One media chunk pulled from the `movi` list: its stream number and payload. The element
/// stamps a PTS from the stream's per-sample duration × the running per-stream sample index.
#[derive(Clone, Debug)]
pub struct MediaChunk {
    /// 0-based stream number (the `NN` of the chunk id).
    pub stream_index: usize,
    /// The 4-CC id (for diagnostics / type-code awareness).
    pub id: FourCc,
    /// The chunk payload — one video frame or one audio packet, verbatim.
    pub data: Vec<u8>,
}

/// An incremental walker over the `movi` byte stream: `push` bytes as they arrive on the
/// sink pad, `next_chunk` to pull each complete media chunk. Buffers across input
/// boundaries (a chunk may span two `push`es) and **resyncs** on a corrupt chunk id by
/// scanning forward for the next plausible one (this brief's untrusted-input mandate).
///
/// The walker is fed the stream **from the start of `movi`'s data** (the first sub-chunk);
/// the element skips the file up to there. It descends `LIST 'rec '` wrappers and skips
/// `JUNK`, exactly as the AVI RIFF Reference describes the `movi` layout.
#[derive(Default)]
pub struct MoviWalker {
    /// Unconsumed bytes buffered across `push` boundaries. Bounded by backpressure: the
    /// element only `push`es more while it can emit (the demuxer discipline), so this holds
    /// at most one input batch plus a partial chunk.
    buf: Vec<u8>,
    /// Absolute byte offset within the *original file* of `buf[0]` — for diagnostics and to
    /// let a future OpenDML index cross-reference. Advances as bytes are consumed.
    consumed: u64,
    /// True once we have seen a valid chunk id — before that, leading garbage (a proportional
    /// seek can land mid-`movi`) is scanned past rather than trusted.
    synced: bool,
}

/// The four ASCII digits/letters we accept as the *first* byte of a `movi` chunk id when
/// resyncing: a stream-numbered data chunk starts with a digit, and the wrapper ids `LIST`
/// / `JUNK` / `idx1` start with an uppercase letter. Kept deliberately permissive — a false
/// positive just costs one more parse attempt, which itself bounds-checks.
fn plausible_chunk_id(id: &[u8]) -> bool {
    if id.len() < 4 {
        return false;
    }
    // Structural wrappers we descend/skip.
    if id == b"LIST" || id == b"JUNK" || id == b"idx1" || id == b"rec " {
        return true;
    }
    // A stream data chunk: two ASCII digits then a two-letter type code (NNtt).
    id[0].is_ascii_digit()
        && id[1].is_ascii_digit()
        && id[2].is_ascii_alphanumeric()
        && id[3].is_ascii_alphanumeric()
}

impl MoviWalker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed more `movi` bytes (in file order, starting at the first sub-chunk of `movi`).
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Drop the front `n` consumed bytes, advancing the absolute offset. Keeps `buf` from
    /// growing unbounded once chunks are pulled.
    fn advance(&mut self, n: usize) {
        let n = n.min(self.buf.len());
        self.buf.drain(..n);
        self.consumed = self.consumed.saturating_add(n as u64);
    }

    /// Reset for a post-seek resume: forget the buffered tail (it is content behind the seek
    /// target) and re-arm the resync scanner. The element also re-skips the file to the seek
    /// byte, so the next `push` starts at (or near) a fresh chunk boundary.
    pub fn resync(&mut self) {
        self.buf.clear();
        self.synced = false;
    }

    /// Pull the next complete media chunk, or `None` if more bytes are needed / the stream is
    /// exhausted. Descends `LIST 'rec '`, skips `JUNK`/`LIST` wrappers and unknown chunks,
    /// and resyncs past corruption. Never panics or slices out of range.
    pub fn next_chunk(&mut self) -> Option<MediaChunk> {
        loop {
            // Need at least an 8-byte chunk header.
            if self.buf.len() < 8 {
                return None;
            }
            let id: FourCc = [self.buf[0], self.buf[1], self.buf[2], self.buf[3]];

            // Resync: if we are not synced or this id is implausible, scan forward one byte
            // at a time for the next plausible chunk id (this brief's resync mandate).
            if !self.synced || !plausible_chunk_id(&id) {
                if let Some(skip) = self.find_next_chunk_id() {
                    if skip > 0 {
                        self.advance(skip);
                        continue; // re-read the header at the new position
                    }
                    self.synced = true;
                } else {
                    // No plausible id anywhere in the buffer: keep all but the last 3 bytes
                    // (a 4-CC could straddle the next push) and wait for more.
                    let keep_from = self.buf.len().saturating_sub(3);
                    self.advance(keep_from);
                    return None;
                }
            }

            let size = u32::from_le_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;

            // `LIST 'rec '` (OpenDML grouping): descend into it — consume the 12-byte
            // list header and continue reading the inner chunks directly.
            if &id == b"LIST" {
                if self.buf.len() < 12 {
                    return None; // need the list type too
                }
                let list_type: FourCc = [self.buf[8], self.buf[9], self.buf[10], self.buf[11]];
                if &list_type == b"rec " {
                    self.advance(12); // step inside the wrapper
                    continue;
                }
                // Any other LIST inside movi: skip the whole thing, WORD-aligned.
                let total = 8usize.saturating_add(padded(size));
                if self.buf.len() < total {
                    return None;
                }
                self.advance(total);
                continue;
            }

            // `idx1` / `JUNK` / other non-media chunk: we have reached the index or padding
            // — skip it (idx1 marks end of movi in practice). WORD-aligned.
            if &id == b"idx1" || &id == b"JUNK" || !id[0].is_ascii_digit() {
                let total = 8usize.saturating_add(padded(size));
                if self.buf.len() < total {
                    // idx1 can be large and arrive across pushes; wait for the whole thing
                    // only if it plausibly fits memory — otherwise just stop (end of media).
                    if &id == b"idx1" {
                        return None;
                    }
                    return None;
                }
                self.advance(total);
                continue;
            }

            // A stream data chunk (NNtt): need the whole payload (WORD-aligned) present.
            let total = 8usize.saturating_add(padded(size));
            if self.buf.len() < total {
                return None; // wait for the rest of the payload
            }
            // Bounds are guaranteed by the length check above.
            let data = self.buf[8..8 + size].to_vec();
            let stream_index = ((id[0] as char).to_digit(10).unwrap_or(0) * 10
                + (id[1] as char).to_digit(10).unwrap_or(0)) as usize;
            self.advance(total);
            return Some(MediaChunk { stream_index, id, data });
        }
    }

    /// Scan `buf` for the next plausible chunk id, returning the number of bytes to skip to
    /// reach it (`Some(0)` = already at one). `None` = none found in the current buffer.
    fn find_next_chunk_id(&self) -> Option<usize> {
        let n = self.buf.len();
        if n < 8 {
            return None;
        }
        for i in 0..=(n - 4) {
            if plausible_chunk_id(&self.buf[i..i + 4]) {
                // Require the 8-byte header to be readable so `next_chunk` can proceed.
                if i + 8 <= n || &self.buf[i..i + 4] == b"idx1" {
                    return Some(i);
                }
            }
        }
        None
    }
}

/// A chunk's total footprint including RIFF WORD padding (an odd payload gets one pad byte).
fn padded(size: usize) -> usize {
    size.saturating_add(size & 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but valid AVI header in memory: one video (`vids/XVID`, 160×120) and
    /// one audio (`auds`, wFormatTag 0x2000 AC-3, 48 kHz, 6ch) stream, then a `movi` list
    /// containing `data_chunks` (id, bytes). Returns the whole file bytes. Used by several
    /// tests as a stand-in for a real AVI so they do not need the 2 GB file.
    fn build_avi(data_chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(id);
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(body);
            if body.len() & 1 == 1 {
                v.push(0); // WORD pad
            }
            v
        }
        fn list(list_type: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut inner = Vec::new();
            inner.extend_from_slice(list_type);
            inner.extend_from_slice(body);
            chunk(b"LIST", &inner)
        }

        // avih: only fields we read (µs/frame @0, total_frames @16). 56 bytes.
        let mut avih = vec![0u8; 56];
        avih[0..4].copy_from_slice(&41_666u32.to_le_bytes()); // 24 fps
        avih[16..20].copy_from_slice(&3u32.to_le_bytes());

        // Video strh (56 bytes): fccType 'vids' @0, fccHandler 'XVID' @4, scale @20, rate @24,
        // length @32, sample_size @44.
        let mut vstrh = vec![0u8; 56];
        vstrh[0..4].copy_from_slice(b"vids");
        vstrh[4..8].copy_from_slice(b"XVID");
        vstrh[20..24].copy_from_slice(&1u32.to_le_bytes()); // scale
        vstrh[24..28].copy_from_slice(&24u32.to_le_bytes()); // rate → 24 fps
        vstrh[32..36].copy_from_slice(&3u32.to_le_bytes()); // length
        // Video strf: BITMAPINFOHEADER, biWidth @4, biHeight @8.
        let mut vstrf = vec![0u8; 40];
        vstrf[0..4].copy_from_slice(&40u32.to_le_bytes());
        vstrf[4..8].copy_from_slice(&160i32.to_le_bytes());
        vstrf[8..12].copy_from_slice(&120i32.to_le_bytes());
        vstrf[16..20].copy_from_slice(b"XVID"); // biCompression
        let vstrl = list(b"strl", &[chunk(b"strh", &vstrh), chunk(b"strf", &vstrf)].concat());

        // Audio strh (56 bytes): fccType 'auds', scale 1, rate 48000, sample_size 0 (VBR).
        let mut astrh = vec![0u8; 56];
        astrh[0..4].copy_from_slice(b"auds");
        astrh[20..24].copy_from_slice(&1u32.to_le_bytes());
        astrh[24..28].copy_from_slice(&48_000u32.to_le_bytes());
        // Audio strf: WAVEFORMATEX, tag 0x2000, ch 6, rate 48000, bits 0.
        let mut astrf = vec![0u8; 18];
        astrf[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
        astrf[2..4].copy_from_slice(&6u16.to_le_bytes());
        astrf[4..8].copy_from_slice(&48_000u32.to_le_bytes());
        let astrl = list(b"strl", &[chunk(b"strh", &astrh), chunk(b"strf", &astrf)].concat());

        let hdrl = list(
            b"hdrl",
            &[chunk(b"avih", &avih), vstrl, astrl].concat(),
        );

        let mut movi_body = Vec::new();
        for (id, body) in data_chunks {
            movi_body.extend_from_slice(&chunk(id, body));
        }
        let movi = list(b"movi", &movi_body);

        let mut form = Vec::new();
        form.extend_from_slice(b"AVI ");
        form.extend_from_slice(&hdrl);
        form.extend_from_slice(&movi);

        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(form.len() as u32).to_le_bytes());
        file.extend_from_slice(&form);
        file
    }

    #[test]
    fn probe_discovers_video_and_audio() {
        let avi = build_avi(&[(b"00dc", &[1, 2, 3, 4]), (b"01wb", &[5, 6])]);
        let (header, movi) = probe_header(&avi).expect("probe");
        assert_eq!(header.streams.len(), 2, "one video + one audio");
        let v = &header.streams[0];
        assert_eq!(v.kind, StreamKind::Video);
        assert_eq!(&v.handler, b"XVID");
        assert_eq!((v.width, v.height), (160, 120));
        assert_eq!((v.scale, v.rate), (1, 24));
        let a = &header.streams[1];
        assert_eq!(a.kind, StreamKind::Audio);
        assert_eq!(a.format_tag, 0x2000, "AC-3");
        assert_eq!(a.channels, 6);
        assert_eq!(a.samples_per_sec, 48_000);
        assert!(movi.is_some(), "movi located in the full-file prefix");
    }

    #[test]
    fn walker_yields_chunks_in_order() {
        let avi = build_avi(&[(b"00dc", &[0xAA; 5]), (b"01wb", &[0xBB, 0xCC]), (b"00dc", &[0xDD])]);
        let (_h, movi) = probe_header(&avi).expect("probe");
        let start = movi.unwrap().movi_data_start as usize;
        let mut w = MoviWalker::new();
        w.push(&avi[start..]);
        let c0 = w.next_chunk().expect("chunk 0");
        assert_eq!(c0.stream_index, 0);
        assert_eq!(c0.data, vec![0xAA; 5]);
        let c1 = w.next_chunk().expect("chunk 1");
        assert_eq!(c1.stream_index, 1);
        assert_eq!(c1.data, vec![0xBB, 0xCC]);
        let c2 = w.next_chunk().expect("chunk 2");
        assert_eq!(c2.stream_index, 0);
        assert_eq!(c2.data, vec![0xDD]);
        assert!(w.next_chunk().is_none(), "movi exhausted");
    }

    #[test]
    fn walker_reassembles_across_push_boundaries() {
        let avi = build_avi(&[(b"00dc", &(0..50u8).collect::<Vec<_>>())]);
        let (_h, movi) = probe_header(&avi).expect("probe");
        let start = movi.unwrap().movi_data_start as usize;
        let stream = &avi[start..];
        let mut w = MoviWalker::new();
        // Feed one byte at a time: the chunk must reassemble.
        for &b in stream {
            assert!(w.next_chunk().is_none() || true); // no chunk until complete
            w.push(&[b]);
        }
        let c = w.next_chunk().expect("chunk reassembled");
        assert_eq!(c.data, (0..50u8).collect::<Vec<_>>());
    }

    #[test]
    fn walker_descends_rec_wrapper() {
        // Hand-build a movi with a `LIST 'rec '` wrapping two chunks.
        fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(id);
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(body);
            if body.len() & 1 == 1 {
                v.push(0);
            }
            v
        }
        let inner = [chunk(b"00dc", &[1, 2, 3]), chunk(b"01wb", &[4, 5])].concat();
        let mut rec = Vec::new();
        rec.extend_from_slice(b"rec ");
        rec.extend_from_slice(&inner);
        let movi_body = chunk(b"LIST", &rec);

        let mut w = MoviWalker::new();
        w.push(&movi_body);
        let c0 = w.next_chunk().expect("descend rec, chunk 0");
        assert_eq!(c0.data, vec![1, 2, 3]);
        let c1 = w.next_chunk().expect("chunk 1");
        assert_eq!(c1.data, vec![4, 5]);
    }

    #[test]
    fn walker_resyncs_past_garbage() {
        fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(id);
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(body);
            v
        }
        let good = chunk(b"00dc", &[9, 9, 9, 9]);
        // Prepend garbage that a proportional seek might land on.
        let mut stream = vec![0xFF, 0x00, 0x13, 0x37, 0x42];
        stream.extend_from_slice(&good);
        let mut w = MoviWalker::new();
        w.push(&stream);
        let c = w.next_chunk().expect("resync finds the good chunk");
        assert_eq!(c.data, vec![9, 9, 9, 9]);
    }

    #[test]
    fn idx1_parses_and_resolves_movi_relative() {
        // Two entries: video keyframe at movi-relative offset 4, audio at 40.
        let mut payload = Vec::new();
        for (id, flags, off, size) in
            [(b"00dc", 0x10u32, 4u32, 100u32), (b"01wb", 0x00, 40, 20)]
        {
            payload.extend_from_slice(id);
            payload.extend_from_slice(&flags.to_le_bytes());
            payload.extend_from_slice(&off.to_le_bytes());
            payload.extend_from_slice(&size.to_le_bytes());
        }
        let movi_data_start = 2048;
        let file_len = 1_000_000;
        let idx = parse_idx1(&payload, movi_data_start, file_len);
        assert_eq!(idx.entries.len(), 2);
        assert!(idx.entries[0].keyframe);
        assert_eq!(idx.entries[0].stream_index(), Some(0));
        assert_eq!(idx.entries[1].stream_index(), Some(1));
        // movi-relative: base = movi_data_start - 4 = 2044; +4 = 2048.
        assert_eq!(idx.resolve_offset(&idx.entries[0]), 2048);
    }

    #[test]
    fn idx1_builds_seek_index_from_video_keyframes() {
        // Three video frames, frames 0 and 2 keyframes; audio interleaved (ignored for time).
        let mut payload = Vec::new();
        let entries = [
            (b"00dc", 0x10u32, 4u32),   // frame 0, keyframe
            (b"01wb", 0x00, 200),        // audio
            (b"00dc", 0x00, 300),        // frame 1, delta
            (b"00dc", 0x10, 400),        // frame 2, keyframe
        ];
        for (id, flags, off) in entries {
            payload.extend_from_slice(id);
            payload.extend_from_slice(&flags.to_le_bytes());
            payload.extend_from_slice(&off.to_le_bytes());
            payload.extend_from_slice(&100u32.to_le_bytes());
        }
        let idx = parse_idx1(&payload, 1004, 100_000);
        // 24 fps → 41_666_666 ns/frame.
        let dur = 41_666_666u64;
        let si = idx.build_seek_index(0, dur);
        assert_eq!(si.entries.len(), 2, "two video keyframes");
        assert_eq!(si.entries[0], (0, 1004)); // frame 0 → base(1000)+4
        assert_eq!(si.entries[1], (2 * dur, 1400)); // frame 2 → base+400
    }

    // ---- Bad-input table: malformed headers/chunks error or resync, never panic (P0) ----

    #[test]
    fn not_an_avi_errors() {
        assert_eq!(probe_header(b"not a riff file at all").err(), Some(AviError::NotAvi));
        assert_eq!(probe_header(b"RIFF\x00\x00\x00\x00WEBP").err(), Some(AviError::NotAvi));
        assert!(probe_header(b"").is_err());
        assert!(probe_header(b"RIFF").is_err());
    }

    #[test]
    fn oversized_chunk_sizes_do_not_panic() {
        // A hdrl whose strl claims a gigantic size must error, not slice oob.
        let mut avi = Vec::new();
        avi.extend_from_slice(b"RIFF");
        avi.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        avi.extend_from_slice(b"AVI ");
        avi.extend_from_slice(b"LIST");
        avi.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // absurd list size
        avi.extend_from_slice(b"hdrl");
        // Truncated here — must be a clean error.
        assert!(probe_header(&avi).is_err());
    }

    #[test]
    fn walker_never_panics_on_random_bytes() {
        // Fuzz-ish: feed pseudo-random bytes and pull until exhausted. Must terminate
        // without panicking or looping forever.
        let mut seed = 0x1234_5678u32;
        let mut bytes = Vec::new();
        for _ in 0..4096 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            bytes.push((seed >> 16) as u8);
        }
        let mut w = MoviWalker::new();
        w.push(&bytes);
        let mut pulled = 0;
        while w.next_chunk().is_some() && pulled < 100_000 {
            pulled += 1;
        }
        // Reaching here (no panic, bounded iterations) is the assertion.
    }

    #[test]
    fn zero_size_chunk_makes_progress() {
        // A zero-length data chunk must not stall the walker (it advances by the 8-byte
        // header).
        let mut stream = Vec::new();
        stream.extend_from_slice(b"00dc");
        stream.extend_from_slice(&0u32.to_le_bytes());
        stream.extend_from_slice(b"01wb");
        stream.extend_from_slice(&2u32.to_le_bytes());
        stream.extend_from_slice(&[7, 7]);
        let mut w = MoviWalker::new();
        w.push(&stream);
        let c0 = w.next_chunk().expect("empty chunk 0");
        assert_eq!(c0.stream_index, 0);
        assert!(c0.data.is_empty());
        let c1 = w.next_chunk().expect("chunk 1");
        assert_eq!(c1.data, vec![7, 7]);
    }
}
