//! `mp3dec` — the incremental MPEG-1/2 Audio Layer III decoder element (spec:
//! Milestone applications §3; Formats — dynamic caps). MP3 bytes arrive on the sink
//! pad; raw interleaved PCM leaves on the src pad, decoded frame-by-frame as bytes
//! arrive (never buffering the whole stream — latency #2), inlining into the upstream
//! group like a transform — the flacdec pattern.
//!
//! MP3 is a **self-syncing byte stream**: an MPEG audio frame begins with an 11-bit
//! syncword (ISO/IEC 11172-3 §2.4.1.3), each frame carries its own byte length, and a
//! reader locates frames by scanning for that syncword. So the sink accepts a raw
//! `filesrc` byte stream at arbitrary chunk boundaries (buffer internally, resync on
//! the syncword) *and* pre-framed packets from a future demuxer. Bytes are accumulated
//! and walked with the upstream [`FrameWalker`] (self-delimiting, resyncing framer);
//! each complete frame is handed to the upstream [`Mp3CoreDecoder`] (`oxideav_core`
//! packet-in / frame-out) as one packet. A leading ID3v2 tag is skipped up front by
//! its synchsafe size (§ID3v2.4 header) so its body — which can contain false
//! syncword bytes (e.g. embedded album art) — never confuses the framer; ID3v1/APE
//! trailers at EOS are simply the untrailed tail the framer never resyncs into.
//!
//! The PCM format (rate / channels) lives in the frame header, not a static
//! descriptor, so `mp3dec` advertises a broad, `dynamic` `audio/raw` template and
//! **announces** the concrete format downstream at runtime — via
//! [`Ctx::announce_format`] — the moment the first frame decodes (spec: Formats —
//! dynamic caps). MP3 decode output is always `s16` (the `oxideav-mp3` decoder emits
//! interleaved i16, §2.4.3.4.7). Real files do not change rate/channels mid-stream; a
//! header that does is a loud per-buffer warn + drop rather than a silent mislabel
//! (spec: Supervision).
//!
//! Adoption note (see `lib.rs`): `oxideav-mp3` is a pure-Rust MPEG-1/2/2.5 Layer III
//! decoder vetted against an ffmpeg oracle (84–102 dB SNR across 44.1/48 kHz, mono +
//! joint-stereo, 128/320 kbps CBR + VBR). Gapless trim (Xing/LAME encoder-delay
//! padding) is explicitly **out of v1**: the decoder emits every reconstructed sample,
//! including the ~half-frame codec-delay priming and the frame-padded tail, so the
//! output is a few hundred samples longer than a gapless-trimmed reference (documented,
//! not a defect).

use oxideav_core::{CodecId, CodecParameters, Frame as AvFrame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::{parse_header, Mp3FrameHeader};

use profluens_core::batch::Inputs;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
use profluens_core::id::PadId;
use profluens_core::time::Timestamp;

// `audio/raw` family/field/value names. Kept as literals (not a dep on
// profluens-audio) so pf-mp3 stays core-only, exactly as pf-flac / pf-vp8 do; the
// pipeline interns by string, so the ids line up with any audio peer using the same
// names.
const FAMILY: &str = "audio/raw";
const F_RATE: &str = "rate";
const F_CHANNELS: &str = "channels";
const F_SAMPLE: &str = "sample";
// `oxideav-mp3` converts its float decode to interleaved i16 (§2.4.3.4.7): the output
// sample format is always signed 16-bit.
const SAMPLE_S16: &str = "s16";

const SRC_PAD: PadId = PadId(1);

/// How many MP3 frames `process()` decodes before yielding its output batch
/// downstream. Small on purpose: it bounds the batch so the first audio reaches the
/// sink after ~one frame's decode rather than a whole read's worth (spec: latency #2).
/// An MPEG-1 frame is 1152 samples (~26 ms at 44.1 kHz), so a handful still leaves
/// plenty of jitter buffer downstream.
const FRAMES_PER_BATCH: u32 = 4;

/// Cap on buffered-but-not-yet-framed bytes. A stream of pure garbage (no syncword
/// ever) would otherwise grow this without bound; past the cap we drop the oldest
/// bytes, keeping only a syncword-sized tail so a real frame straddling the boundary is
/// not lost. The largest legal MPEG frame is ~1440 bytes (320 kbps @ 32 kHz + pad), so
/// this holds many whole frames — it only bites on non-MP3 input.
const MAX_BUFFERED: usize = 1 << 16;

static SAMPLE_VALUES: [ValueDesc; 1] = [ValueDesc::Id(SAMPLE_S16)];

// A broad `audio/raw` template: any rate/channels, s16 sample format. The concrete
// values are announced at runtime from the first frame header — the src pad is
// `dynamic` for exactly this reason (spec: Formats — dynamic caps).
static SRC_FIELDS: [FieldDesc; 3] = [
    FieldDesc { field: F_RATE, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_CHANNELS, allowed: ConstraintDesc::Any, preferred: None },
    FieldDesc { field: F_SAMPLE, allowed: ConstraintDesc::Set(&SAMPLE_VALUES), preferred: None },
];
// Typed `audio/raw` first; then the `bytes` bridge, so a byte sink (`filesink` — dump
// decoded PCM to a file) still links and the announcement rides it tolerantly (spec:
// Formats — an `audio/raw` refinement riding a `bytes` bridge to a byte sink). Mirrors
// flacdec.
static SRC_OFFERS: [OfferDesc; 2] =
    [OfferDesc { family: FAMILY, fields: &SRC_FIELDS }, OfferDesc::any("bytes")];
// The sink takes an `mp3` elementary stream (a future demuxer's per-frame packets) or a
// raw `bytes` stream (a `filesrc` reading a `.mp3` — arbitrary chunk boundaries). MP3
// self-syncs, so both are the same to the framer.
static SINK_OFFERS: [OfferDesc; 2] = [OfferDesc::any("mp3"), OfferDesc::any("bytes")];

static PADS: [PadDesc; 2] = [
    PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &SINK_OFFERS,
        dynamic: false,
        validate: None,
    },
    PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &SRC_OFFERS,
        dynamic: true, // format is data-dependent — announced at runtime
        validate: None,
    },
];

// `make_default` boxes the element once at registry construction (spec: Plugins) — cold.
#[allow(clippy::disallowed_methods)]
static DESC: ElementDesc = ElementDesc {
    name: "mp3dec",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Passive,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    // Name-constructible (spec: Plugins): the decoder is config-free — it learns its
    // format from the MP3 stream and announces it via dynamic caps.
    make_default: Some(|| Box::new(Mp3Dec::new())),
};

/// Length of a leading ID3v2 tag in `buf` (10-byte header + synchsafe body size,
/// plus the v2.4 footer if flagged), or 0 if there is none. Only inspected at the
/// very start of the stream; a mid-stream `ID3` run is left to the framer. Returns
/// `None` when the header is present but the full tag is not yet buffered (wait for
/// more bytes before deciding).
///
/// The header layout itself lives in [`crate::id3::v2_total_len`] — one implementation,
/// shared with the tag parser (ID3v2.4 §3.1: `"ID3"`, 2 version bytes, 1 flags byte, then
/// a 4-byte **synchsafe** size, plus 10 more when the flags mark a footer). This wrapper
/// adds only the *streaming* tri-state the framer needs on top of it. A header that is
/// present but whose size is not synchsafe (§6.2) is not a tag: `v2_total_len` rejects it
/// and the bytes go to the framer as audio.
fn id3v2_len(buf: &[u8]) -> Option<usize> {
    // A leading `"ID3"` magic marks a tag. If the buffer is still a strict prefix of
    // that magic, we can't yet tell tag-from-audio — wait for more bytes.
    const MAGIC: &[u8; 3] = b"ID3";
    if buf.len() < 3 {
        return if MAGIC.starts_with(buf) {
            None // prefix of "ID3" — undecided
        } else {
            Some(0) // definitely not a tag (first bytes diverge from the magic)
        };
    }
    if &buf[..3] != MAGIC {
        return Some(0);
    }
    if buf.len() < 10 {
        return None; // header itself not fully buffered yet
    }
    Some(crate::id3::v2_total_len(buf).unwrap_or(0))
}

/// The result of trying to frame the next MPEG audio frame out of a byte buffer.
///
/// `pub(crate)` so [`crate::props`] probes with the *same* framer the decoder runs on:
/// one definition of "the first frame", not a second header parser that could disagree.
pub(crate) enum Framed {
    /// A whole frame occupies `buf[offset..offset + len]` (junk before `offset` is
    /// skippable). Ready to decode.
    Frame { offset: usize, len: usize, header: Mp3FrameHeader },
    /// A valid frame header sits at `offset`, but the frame's bytes are not all buffered
    /// yet — drop the junk before `offset` and wait for more input. *Do not* scan past
    /// this header for a shorter frame deeper in: that would latch onto a false syncword
    /// inside the real (incomplete) frame's data — the classic partial-frame bug.
    NeedMore { offset: usize },
    /// No frame header found anywhere in the buffer — pure junk.
    NoSync,
}

/// Frame the next MPEG audio frame out of `buf`, boundary-safely.
///
/// Unlike a whole-buffer walker, this stops at the **first** valid, length-derivable
/// frame header and reports `NeedMore` if that frame is not yet fully buffered, instead
/// of scanning deeper (where a false 11-bit syncword inside the frame's own payload
/// would masquerade as a shorter frame — corrupting the framing whenever a read splits
/// the stream mid-frame). A candidate header is only accepted once **two** consecutive
/// frames line up (the next frame's syncword falls exactly at `offset + len`), the
/// standard resync heuristic that rejects a chance `FF Ex` byte pair inside audio data;
/// at end-of-buffer a single complete frame is accepted so the stream tail still decodes.
pub(crate) fn next_frame(buf: &[u8]) -> Framed {
    let mut i = 0usize;
    while i + 4 <= buf.len() {
        // Cheap 11-bit frame-sync pre-filter (0xFF, then top 3 bits of byte 1 set),
        // covering MPEG-1 / MPEG-2 LSF / the MPEG-2.5 extension (frame.rs §parse_header).
        if buf[i] != 0xFF || (buf[i + 1] & 0xE0) != 0xE0 {
            i += 1;
            continue;
        }
        let Ok(header) = parse_header(&buf[i..]) else {
            i += 1;
            continue;
        };
        let Some(len) = header.frame_len() else {
            // Free-format (no derivable length). The demuxer would measure it; the raw
            // byte path can't, so treat it as a non-frame and scan on.
            i += 1;
            continue;
        };
        if len < 4 {
            i += 1;
            continue;
        }
        if i + len > buf.len() {
            // Header is valid but the frame overruns the buffer: it may be the real next
            // frame, still arriving. Wait — never scan deeper into its payload.
            return Framed::NeedMore { offset: i };
        }
        // A complete candidate frame is buffered. Confirm it by checking the next frame's
        // syncword lands exactly at its end (rejects a false in-payload sync). If the
        // next header isn't buffered yet we can't confirm — but the frame is whole, so
        // accept it (the decoder's own CRC/side-info parse is the backstop, and the
        // stream tail must decode).
        let next = i + len;
        let confirmed = next + 2 > buf.len()
            || (buf[next] == 0xFF && (buf[next + 1] & 0xE0) == 0xE0);
        if confirmed {
            return Framed::Frame { offset: i, len, header };
        }
        // Not confirmed: this was a false sync. Scan one byte on.
        i += 1;
    }
    // No length-derivable header anywhere. If the buffer still holds a partial header
    // prefix (`FF Ex …`) at the tail, ask for more; otherwise it is pure junk.
    Framed::NoSync
}

/// Incrementally decodes an MP3 byte stream to interleaved PCM.
pub struct Mp3Dec {
    /// The `oxideav_core::Decoder` trait object doing the actual Layer III decode.
    dec: Box<dyn oxideav_core::Decoder>,
    /// Accumulated bytes not yet framed into whole MP3 frames.
    buf: Vec<u8>,
    /// Set once the leading ID3v2 tag (if any) has been resolved and skipped.
    id3_done: bool,
    /// Runtime format, learned from the first decoded frame header.
    announced: bool,
    rate: u32,
    channels: u16,
    /// PCM samples-per-channel emitted so far — the zero-based PTS grid.
    samples_emitted: u64,
    /// Decoded interleaved-i16 PCM not yet emitted — the backpressure carry. Bounded to
    /// one frame: the framing loop never decodes another until this drains, so a slow
    /// sink cannot make the decoder run ahead and balloon the pool.
    pending: Vec<i16>,
    pending_pos: usize,
    /// Reused packet-payload buffer. `oxideav_core::Packet::new` takes an owned
    /// `Vec<u8>`, so a frame's bytes must live in a `Vec`; this one is moved into the
    /// packet and reclaimed after `send_packet` (which copies what it needs and keeps
    /// no reference to the buffer), so its capacity persists — no per-frame heap alloc
    /// (spec: performance #1).
    pkt_buf: Vec<u8>,
}

impl Mp3Dec {
    // One-time element setup: `buf`/`pending` are reused-and-cleared across frames, not
    // re-allocated per frame (spec: performance #1) — cold.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            dec: build_decoder(),
            buf: Vec::new(),
            id3_done: false,
            announced: false,
            rate: 0,
            channels: 0,
            samples_emitted: 0,
            pending: Vec::new(),
            pending_pos: 0,
            pkt_buf: Vec::new(),
        }
    }

    fn reset_state(&mut self) {
        self.dec = build_decoder();
        self.buf.clear();
        self.id3_done = false;
        self.announced = false;
        self.rate = 0;
        self.channels = 0;
        self.samples_emitted = 0;
        self.pending.clear();
        self.pending_pos = 0;
        self.pkt_buf.clear();
    }

    /// PTS in nanoseconds for the current `samples_emitted` on the sample grid from a
    /// zero base (like wavparse). `NONE` until the rate is known.
    fn pts_now(&self) -> Timestamp {
        if self.rate == 0 {
            return Timestamp::NONE;
        }
        Timestamp::from_nanos((self.samples_emitted * 1_000_000_000) / u64::from(self.rate))
    }

    /// Emit the pending frame's not-yet-written samples as interleaved little-endian
    /// s16 PCM, directly into pool buffers. With `bounded` it uses the capped pool
    /// (`try_alloc`) and returns `Ok(false)` when the pool is full — the caller then
    /// yields so the sink drains (backpressure). With `!bounded` it uses the unbounded
    /// pool (never fails), to guarantee the stream tail is flushed at EOS. Returns
    /// `Ok(true)` once fully drained. Each emitted buffer is PTS-stamped on the sample
    /// grid; `samples_emitted` advances per channel.
    fn emit_pending(&mut self, ctx: &mut Ctx, bounded: bool) -> Result<bool, Error> {
        let ch = self.channels.max(1) as usize;
        while self.pending_pos < self.pending.len() {
            let mut buf = if bounded {
                match ctx.try_alloc(SRC_PAD) {
                    Some(b) => b,
                    None => return Ok(false), // pool full → backpressure
                }
            } else {
                ctx.alloc(SRC_PAD)
            };
            // Whole interleaved *frames* (ch samples) per buffer, so a buffer never
            // splits a multichannel sample group.
            let cap_samples = ((buf.memory.capacity() / 2) / ch * ch).max(ch);
            let end = (self.pending_pos + cap_samples).min(self.pending.len());
            let dst = buf.memory.as_mut_full();
            let mut n = 0;
            for &s in &self.pending[self.pending_pos..end] {
                dst[n..n + 2].copy_from_slice(&s.to_le_bytes());
                n += 2;
            }
            buf.memory.set_len(n);
            buf.pts = self.pts_now();
            self.samples_emitted += ((end - self.pending_pos) / ch) as u64;
            buf.duration = self.pts_now().saturating_sub(buf.pts);
            ctx.out(SRC_PAD).push(buf);
            self.pending_pos = end;
        }
        self.pending.clear();
        self.pending_pos = 0;
        Ok(true)
    }

    /// Decode every whole MP3 frame currently buffered, appending interleaved i16 into
    /// `self.pending` (which the caller then drains). Returns the number of frames
    /// decoded this call. Per-frame error scope: a frame that fails to decode is warned
    /// + dropped and framing continues at the next syncword (spec: Supervision).
    ///
    /// Only whole frames are consumed; the trailing partial frame stays in `self.buf`
    /// for the next call (incremental decode over arbitrary chunk boundaries).
    fn frame_and_decode_one(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        // Resolve + skip a leading ID3v2 tag before the first frame walk.
        if !self.id3_done {
            match id3v2_len(&self.buf) {
                Some(len) => {
                    if len > self.buf.len() {
                        return Ok(false); // tag not fully buffered yet
                    }
                    self.buf.drain(..len);
                    self.id3_done = true;
                }
                None => return Ok(false), // header present but incomplete — wait
            }
        }

        // Frame the next whole MPEG audio frame, boundary-safely (see `next_frame`): it
        // never scans into an incomplete frame's payload for a false syncword, so the
        // framing does not depend on where the source split its reads.
        let (offset, len) = match next_frame(&self.buf) {
            Framed::Frame { offset, len, .. } => (offset, len),
            Framed::NeedMore { offset } => {
                // A valid header at `offset` whose frame is still arriving: drop the junk
                // before it, keep the (partial) frame, and wait for more input.
                if offset > 0 {
                    self.buf.drain(..offset);
                }
                return Ok(false);
            }
            Framed::NoSync => {
                // No frame header anywhere — pure junk. Bound the buffer against a flood
                // (keep a syncword-sized tail so a frame straddling the cut survives).
                if self.buf.len() > MAX_BUFFERED {
                    let drop_to = self.buf.len() - 3;
                    self.buf.drain(..drop_to);
                }
                return Ok(false);
            }
        };
        // Copy the frame's bytes into the reused packet buffer and consume through its
        // end (dropping pre-sync junk). `Packet::new` needs an owned `Vec<u8>`; we take
        // the reused buffer out (leaving an empty Vec), refill it, and reclaim it below —
        // so its capacity persists and no per-frame heap allocation occurs.
        let mut pkt_buf = std::mem::take(&mut self.pkt_buf);
        pkt_buf.clear();
        pkt_buf.extend_from_slice(&self.buf[offset..offset + len]);
        self.buf.drain(..offset + len);

        // Feed the whole frame to the trait decoder as one packet, then drain its
        // frame(s). A frame that fails (bad CRC, corrupt main-data) is dropped with a
        // bus Warning; the reservoir/overlap state stays, and the next syncword resyncs.
        let mut pkt = Packet::new(0, TimeBase::from_rate(self.rate.max(1)), pkt_buf);
        let send_result = self.dec.send_packet(&pkt);
        // Reclaim the buffer: `send_packet` copies what it needs and keeps no reference
        // to the packet payload, so its allocation returns to `self.pkt_buf` for reuse.
        self.pkt_buf = std::mem::take(&mut pkt.data);
        if let Err(e) = send_result {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element { element, message: format!("mp3dec: frame dropped: {e}") },
            });
            return Ok(true); // consumed the frame; resync continues on the next call
        }
        loop {
            match self.dec.receive_frame() {
                Ok(AvFrame::Audio(a)) => self.take_audio(ctx, a)?,
                Ok(_) => {} // non-audio never occurs for an MP3 decoder
                Err(e) if e.is_need_more() => break,
                Err(e) if e.is_eof() => break,
                Err(e) => {
                    let element = ctx.element();
                    ctx.post(BusMessage::Warning {
                        element,
                        error: Error::Element {
                            element,
                            message: format!("mp3dec: frame dropped: {e}"),
                        },
                    });
                    break;
                }
            }
        }
        Ok(true)
    }

    /// Append one decoded `AudioFrame`'s interleaved i16 into `self.pending`, announcing
    /// the runtime format on the first frame and warn+dropping a mid-stream
    /// rate/channel *change*.
    fn take_audio(&mut self, ctx: &mut Ctx, a: oxideav_core::AudioFrame) -> Result<(), Error> {
        let nch = a.data.len();
        if nch == 0 {
            return Ok(());
        }
        // `self.rate` / `self.channels` were set from the frame header by
        // `announce_if_needed` before this frame was decoded (the plane count here
        // equals that header's `channel_count`). A frame whose *channel count* differs
        // from the announced one is the mid-stream change case: real files never do it,
        // and silently mislabeling it downstream would corrupt the interleave — warn and
        // drop instead (spec: Supervision; a loud per-buffer weather event).
        if nch as u16 != self.channels {
            let element = ctx.element();
            ctx.post(BusMessage::Warning {
                element,
                error: Error::Element {
                    element,
                    message: format!(
                        "mp3dec: mid-stream channel change {} → {} — frame dropped",
                        self.channels, nch
                    ),
                },
            });
            return Ok(());
        }
        // Interleave the planar S16 output (`a.data[ch]` is i16 LE for channel `ch`).
        let per: Vec<&[u8]> = a.data.iter().map(Vec::as_slice).collect();
        let n = per[0].len() / 2;
        self.pending.reserve(n * nch);
        for i in 0..n {
            for pl in &per {
                let lo = pl.get(2 * i).copied().unwrap_or(0);
                let hi = pl.get(2 * i + 1).copied().unwrap_or(0);
                self.pending.push(i16::from_le_bytes([lo, hi]));
            }
        }
        Ok(())
    }
}

/// Build a fresh `oxideav-mp3` trait decoder. The concrete rate/channels are re-read
/// from every frame header, so the construction hint here is nominal.
fn build_decoder() -> Box<dyn oxideav_core::Decoder> {
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(44_100);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    make_decoder(&params).expect("oxideav-mp3 make_decoder is infallible for 2ch")
}

impl Default for Mp3Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Mp3Dec {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        self.reset_state();
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Flush any carry from a prior backpressured call before decoding more.
        if !self.emit_pending(ctx, true)? {
            return Ok(Flow::Ok);
        }
        let mut emitted = 0u32;
        loop {
            // Learn rate/channels from the next buffered frame's header *before* it is
            // decoded, so `take_audio` and the PTS grid have the right values and the
            // format is announced exactly once (the flacdec announce-on-first-frame
            // pattern, adapted: MP3 carries its format in every frame header).
            self.announce_if_needed(ctx)?;

            match self.frame_and_decode_one(ctx)? {
                true => {
                    if !self.emit_pending(ctx, true)? {
                        return Ok(Flow::Ok); // pool filled mid-frame — carry, yield
                    }
                    emitted += 1;
                    if emitted >= FRAMES_PER_BATCH {
                        return Ok(Flow::Ok); // bound the batch (spec: latency #2)
                    }
                }
                false => match inputs.pop() {
                    Some(inbuf) => self.buf.extend_from_slice(inbuf.memory.data()),
                    None => {
                        return Ok(Flow::Ok); // no whole frame and no more input
                    }
                },
            }
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        match event {
            // Seek (spec: flush/seek): drop the buffered bytes, the pending carry, and
            // the decoder's carry-over reservoir / IMDCT overlap / synthesis state, and
            // re-sync to the next frame boundary on the bytes that arrive after the
            // source's byte-seek. `id3_done` stays set (no ID3 tag mid-stream); the
            // announced format is kept (every frame header re-confirms it).
            Event::FlushStart => {
                let _ = self.dec.reset();
                self.buf.clear();
                self.pending.clear();
                self.pending_pos = 0;
                // …and re-base the sample counter the PTS is derived from. An MP3 frame
                // carries no timestamp — its position in the stream *is* its timestamp —
                // so a decoder that keeps counting from where it was interrupted stamps
                // the post-seek audio with pre-seek times, and every consumer that reads
                // `pts` (a muxer, a video sink synchronising to this audio, a test) sees
                // the stream jump backwards. The seek target is the only place the new
                // position exists, which is exactly what `seek_target` is for: the source
                // reads `to_byte` from it, and the decoder — the element that owns this
                // stream's time base — reads `to_time`.
                if let Some(t) = ctx.seek_target() {
                    if let (Some(ns), true) = (t.to_time.nanos(), self.rate > 0) {
                        self.samples_emitted =
                            (u128::from(ns) * u128::from(self.rate) / 1_000_000_000) as u64;
                    }
                }
            }
            // End of stream: flush the carry and every remaining whole frame through the
            // unbounded pool, so the tail is emitted even if we were backpressured here.
            // A truncated trailing frame (and any ID3v1/APE trailer) is left in `buf`.
            Event::Eos => {
                self.emit_pending(ctx, false)?;
                loop {
                    self.announce_if_needed(ctx)?;
                    match self.frame_and_decode_one(ctx)? {
                        true => {
                            self.emit_pending(ctx, false)?;
                        }
                        false => break,
                    }
                }
                let _ = self.dec.flush();
            }
            _ => {}
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        self.reset_state();
    }
}

impl Mp3Dec {
    /// Peek the next buffered whole frame's header (without consuming it) to learn the
    /// runtime format and announce it once. MP3 carries rate/channels in *every* frame
    /// header, so the first frame is enough. No-op once announced (rate/channels then
    /// stay fixed; a later change is caught in `take_audio`).
    fn announce_if_needed(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        if self.announced {
            return Ok(());
        }
        // ID3 must be skipped before the header offset is meaningful.
        if !self.id3_done {
            match id3v2_len(&self.buf) {
                Some(len) if len <= self.buf.len() => {
                    self.buf.drain(..len);
                    self.id3_done = true;
                }
                _ => return Ok(()), // wait for more bytes
            }
        }
        // Peek the first framed header (whole *or* still-arriving — a header is enough to
        // announce the format; the frame itself is decoded later by `frame_and_decode_one`).
        let hdr = match next_frame(&self.buf) {
            Framed::Frame { header, .. } => header,
            Framed::NeedMore { offset } => match parse_header(&self.buf[offset..]) {
                Ok(h) => h,
                Err(_) => return Ok(()),
            },
            Framed::NoSync => return Ok(()), // no header yet
        };
        self.rate = hdr.sample_rate_hz;
        self.channels = u16::from(hdr.channel_count());
        ctx.announce_format(
            SRC_PAD,
            FAMILY,
            &[
                (F_RATE, ValueDesc::Int(i64::from(self.rate))),
                (F_CHANNELS, ValueDesc::Int(i64::from(self.channels))),
                (F_SAMPLE, ValueDesc::Id(SAMPLE_S16)),
            ],
        );
        self.announced = true;
        Ok(())
    }
}
