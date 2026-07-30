//! `mkvsegmentsink` — the NVR's recording sink: H.264 Annex-B access units in,
//! a directory of self-contained, seekable MKV segment files out, rotated at
//! keyframe boundaries every `segment` seconds (the `splitmuxsink` role, as an
//! app-side element — spec: no-bins, controllers live in the application).
//!
//! Each segment is a complete indexed Matroska file: Cues + front SeekHead
//! (RFC 9559 §5.1.1/§5.1.5) and a back-patched `Info\Duration` (§5.1.2) via
//! [`MatroskaWriter::reserve_duration`] — both applied as positioned writes.
//! Timestamps rebase per segment (first frame = 0), so every file plays
//! standalone; the recording wall-clock start rides in the filename (unix
//! seconds).
//!
//! Keyframes are detected from the bitstream, not buffer flags: an AU
//! containing an IDR slice (H.264 §7.4.1 Table 7-1, `nal_unit_type` 5) starts a
//! recovery point. A segment only *opens* on an IDR with parameter sets in hand
//! (SPS/PPS from the constructor — SDP `sprop-parameter-sets`, RFC 6184 §8.1 —
//! or in-band), because a file whose first frame needs earlier state would be
//! undecodable. Matroska stores AVC as length-prefixed NALs with the parameter
//! sets in CodecPrivate (`AVCDecoderConfigurationRecord`, ISO/IEC 14496-15
//! §5.3.3.1), so each AU is reframed start-codes → 4-octet lengths (§5.3.4.2)
//! on the way in.
//!
//! ## IO: reactor-native (spec: IO — elements never block on data)
//!
//! All segment bytes leave through `ctx.io()` positioned writes — cluster runs
//! and frame payloads (the payload [`Memory`] forwarded by refcount, no copy),
//! and the finalize patches. The reactor holds **one file per element**, so
//! rotation is a drain-then-swap state machine: at a boundary the sink stops
//! popping input (the ring is the backpressure buffer), lets in-flight writes
//! complete, submits the finalize tail + patches, waits again, and only then
//! registers the next file — a straggler write can never land in the wrong
//! segment. The one sanctioned blocking exception is the `stop()` path: a live
//! recording never sees EOS and a stopped pipeline runs no further reactor
//! passes, so the element finalizes through a dup'ed fd there (teardown, not
//! streaming).

use std::fs::File;
use std::path::PathBuf;

use pf_mkv::writer::{MatroskaWriter, MuxOut, MuxPiece, SeekHeadPatch, TrackConfig};
use pf_vaapi::h264parse::{parse_sps, split_nals, NAL_PPS, NAL_SLICE_IDR, NAL_SPS};
use profluens_core::batch::Inputs;
use profluens_core::buffer::{Buffer, BufferFlags};
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::FormatId;
use profluens_core::io::IoResult;
use profluens_core::memory::{Memory, Pool};
use profluens_core::time::Timestamp;

static OFFERS: [OfferDesc; 1] = [OfferDesc::any("h264/annexb")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "mkvsegmentsink",
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

/// The muxing state of the segment being written. The `File` itself lives in
/// the reactor; `spare` is the dup used only by the `stop()` teardown path.
struct OpenSegment {
    writer: MatroskaWriter,
    out: MuxOut,
    path: PathBuf,
    spare: File,
    /// First frame's pipeline pts (ns) — segment time 0.
    base_pts: u64,
    /// Last frame's pipeline pts (ns), for the duration patch.
    last_pts: u64,
    frames: u64,
    /// `finalize_scatter` has run: `out` holds the Cues tail, `patches` the
    /// SeekHead/Duration back-patches not yet submitted.
    finalized: bool,
    patches: Vec<SeekHeadPatch>,
}

/// Where rotation stands (see the module docs' drain-then-swap protocol).
#[derive(Clone, Copy)]
enum Phase {
    /// No file open — waiting for a decodable entry point (IDR + SPS/PPS).
    Idle,
    /// Streaming cluster writes into the registered file.
    Writing,
    /// Boundary hit: submit the finalize tail + patches (positioned and
    /// mutually disjoint — order-free), then wait for a full drain before the
    /// registration swap. `reopen` = rotation (the carried buffer's AU opens
    /// the next segment); `false` = EOS, back to Idle.
    Finalizing { reopen: bool },
}

/// Totals a supervisor can print; updated as segments close.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentStats {
    pub segments_closed: u64,
    pub frames_written: u64,
    pub bytes_written: u64,
}

pub struct MkvSegmentSink {
    dir: PathBuf,
    prefix: String,
    segment_ns: u64,
    /// Latest parameter sets (NAL bytes incl. header): seeded out-of-band from
    /// the SDP, refreshed by in-band SPS/PPS between segments.
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    phase: Phase,
    seg: Option<OpenSegment>,
    /// The rotation keyframe's buffer, carried across the Finalizing phase —
    /// it belongs to the *next* segment.
    carry: Option<Buffer>,
    /// Next byte offset in the registered file (positioned writes).
    write_offset: u64,
    /// Writes submitted and not yet completed (the swap gate).
    in_flight: u32,
    seq: u64,
    /// Exact-size allocations for reframed AU payloads and mux byte runs
    /// ([`Pool::acquire_exact`]'s heap path — slot size 1 so no slot ever
    /// qualifies; these buffers must not budget against any pipeline pool).
    scratch: Pool,
    stats: std::sync::Arc<std::sync::Mutex<SegmentStats>>,
    /// AUs dropped while waiting for the opening IDR (normal at join) or for
    /// parameter sets (a stream problem if it persists).
    skipped: u64,
}

impl MkvSegmentSink {
    /// Record into `dir` as `{prefix}-{seq:04}-{unix_secs}.mkv`, rotating at the
    /// first IDR at or past `segment_secs`. `sprop`: out-of-band SPS/PPS NALs
    /// (SDP `sprop-parameter-sets`, RFC 6184 §8.1), may be empty when the
    /// stream carries them in-band.
    pub fn new(dir: impl Into<PathBuf>, prefix: &str, segment_secs: f64, sprop: &[Vec<u8>]) -> Self {
        let mut s = Self {
            dir: dir.into(),
            prefix: prefix.to_string(),
            segment_ns: (segment_secs * 1e9) as u64,
            sps: None,
            pps: None,
            phase: Phase::Idle,
            seg: None,
            carry: None,
            write_offset: 0,
            in_flight: 0,
            seq: 0,
            scratch: Pool::new(1),
            stats: Default::default(),
            skipped: 0,
        };
        for nal in sprop {
            s.stash_param_set(nal);
        }
        s
    }

    /// A live handle onto the running totals (for the app's stats line).
    pub fn stats_handle(&self) -> std::sync::Arc<std::sync::Mutex<SegmentStats>> {
        self.stats.clone()
    }

    fn stash_param_set(&mut self, nal: &[u8]) {
        match nal.first().map(|b| b & 0x1F) {
            Some(NAL_SPS) => self.sps = Some(nal.to_vec()),
            Some(NAL_PPS) => self.pps = Some(nal.to_vec()),
            _ => {}
        }
    }

    /// `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.3.1) from the
    /// stashed SPS/PPS, `lengthSizeMinusOne` = 3 (4-octet NAL lengths).
    fn build_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(11 + sps.len() + pps.len());
        v.push(1); // configurationVersion
        v.extend_from_slice(&sps[1..4]); // AVCProfileIndication, compat, level from the SPS
        v.push(0xFC | 3); // reserved ´111111´ + lengthSizeMinusOne
        v.push(0xE0 | 1); // reserved ´111´ + numOfSequenceParameterSets
        v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        v.extend_from_slice(sps);
        v.push(1); // numOfPictureParameterSets
        v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        v.extend_from_slice(pps);
        v
    }

    /// A reactor-shaped metadata-less buffer around `mem` (the `filesink`
    /// Patch-path construction).
    fn wrap(mem: Memory) -> Buffer {
        Buffer {
            memory: mem,
            pts: Timestamp::NONE,
            dts: Timestamp::NONE,
            duration: Timestamp::NONE,
            flags: BufferFlags::empty(),
            format: FormatId(0),
            sync: None,
        }
    }

    /// Create + register the next segment file and submit its header. Must only
    /// run with no writes in flight (the registration swap closes the previous
    /// file — see the module docs).
    fn open_segment(&mut self, ctx: &mut Ctx, base_pts: u64) -> Result<(), Error> {
        debug_assert_eq!(self.in_flight, 0, "swap with writes in flight");
        let (sps, pps) = match (&self.sps, &self.pps) {
            (Some(s), Some(p)) => (s.clone(), p.clone()),
            _ => return Err(Error::Todo("mkvsegmentsink: open without SPS/PPS")),
        };
        let sps_info = parse_sps(&sps)
            .ok_or_else(|| Error::Resource("mkvsegmentsink: SPS did not parse".into()))?;
        let avcc = Self::build_avcc(&sps, &pps);
        let track =
            TrackConfig::video(1, "V_MPEG4/ISO/AVC", avcc, sps_info.width(), sps_info.height());
        let mut writer = MatroskaWriter::new(vec![track]);
        writer.enable_cues();
        writer.reserve_duration();

        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = self.dir.join(format!("{}-{:04}-{unix}.mkv", self.prefix, self.seq));
        self.seq += 1;
        let file = File::create(&path)
            .map_err(|e| Error::Resource(format!("create {}: {e}", path.display())))?;
        let spare = file
            .try_clone()
            .map_err(|e| Error::Resource(format!("dup {}: {e}", path.display())))?;
        ctx.io().register(file);
        self.write_offset = 0;

        let mut out = MuxOut::new();
        writer.write_header_scatter(&mut out).map_err(|e| {
            Error::Resource(format!("mkvsegmentsink: header {}: {e:?}", path.display()))
        })?;
        self.seg = Some(OpenSegment {
            writer,
            out,
            path,
            spare,
            base_pts,
            last_pts: base_pts,
            frames: 0,
            finalized: false,
            patches: Vec::new(),
        });
        self.phase = Phase::Writing;
        self.submit_pieces(ctx);
        Ok(())
    }

    /// Submit queued [`MuxOut`] pieces as positioned writes, up to the credit
    /// budget; leftovers stay queued for the next pass. Byte runs copy into
    /// scratch memory (they alias the arena, which `reclaim` reuses); payloads
    /// forward the frame's own [`Memory`] refcount — no copy.
    fn submit_pieces(&mut self, ctx: &mut Ctx) {
        let handle = profluens_core::io::FileHandle(ctx.element().0);
        let credits = ctx.io().credits();
        let Self { seg, scratch, in_flight, write_offset, .. } = self;
        let Some(seg) = seg.as_mut() else { return };
        while *in_flight < credits {
            let Some(piece) = seg.out.pop_front() else { break };
            let (buf, len) = match piece {
                MuxPiece::Bytes { start, end } => {
                    let mut mem = scratch.acquire_exact(end - start);
                    mem.as_mut_full()[..end - start]
                        .copy_from_slice(&seg.out.header_bytes()[start..end]);
                    mem.set_len(end - start);
                    (Self::wrap(mem), end - start)
                }
                MuxPiece::Payload(mem) => {
                    let n = mem.len();
                    (Self::wrap(mem), n)
                }
            };
            ctx.io().submit_write(handle, *write_offset, buf, 0);
            *write_offset += len as u64;
            *in_flight += 1;
        }
        seg.out.reclaim();
    }

    /// Drain write completions; an IO error fails the pipeline loudly.
    fn drain_completions(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        while let Some(c) = ctx.io().next_completion() {
            self.in_flight -= 1;
            match c.result {
                IoResult::Ok(n) => self.stats.lock().unwrap().bytes_written += n as u64,
                IoResult::Cancelled => {}
                IoResult::Err(k) => {
                    let path = self.seg.as_ref().map(|s| s.path.display().to_string());
                    return Err(Error::Resource(format!(
                        "mkvsegmentsink: write {:?}: {k:?}",
                        path.unwrap_or_default()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Run `finalize_scatter` once for the open segment: the Cues tail lands in
    /// `out`, the SeekHead + Duration back-patches (RFC 9559 §5.1.1; §5.1.2) in
    /// `patches`.
    fn compute_finalize(seg: &mut OpenSegment) {
        if seg.finalized {
            return;
        }
        // Duration: last frame's start plus one nominal frame (the mean delta —
        // the true last-frame duration is unknowable from timestamps alone).
        let span = seg.last_pts - seg.base_pts;
        let mean_delta = span / seg.frames.saturating_sub(1).max(1);
        let sh = seg.writer.finalize_scatter(&mut seg.out);
        let dur = seg.writer.duration_patch(span + mean_delta);
        seg.patches = sh.into_iter().chain(dur).collect();
        seg.finalized = true;
    }

    /// Advance the Finalizing state machine: queue the tail + patches (credit-
    /// bounded, resumable), and once everything is submitted *and* completed,
    /// close the books and swap to the next segment (rotation) or go idle.
    fn drive_finalize(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let Phase::Finalizing { reopen } = self.phase else { return Ok(()) };
        let Some(seg) = &mut self.seg else {
            self.phase = Phase::Idle;
            return Ok(());
        };
        Self::compute_finalize(seg);
        self.submit_pieces(ctx); // the Cues tail (resumes across passes)
        let handle = profluens_core::io::FileHandle(ctx.element().0);
        let Self { seg, scratch, in_flight, .. } = self;
        let seg = seg.as_mut().expect("finalizing segment");
        while *in_flight < ctx.io().credits() {
            let Some(p) = seg.patches.pop() else { break };
            let mut mem = scratch.acquire_exact(p.bytes.len().max(1));
            mem.as_mut_full()[..p.bytes.len()].copy_from_slice(&p.bytes);
            mem.set_len(p.bytes.len());
            ctx.io().submit_write(handle, p.offset, Self::wrap(mem), 0);
            *in_flight += 1;
        }
        if !seg.out.is_empty() || !seg.patches.is_empty() || *in_flight > 0 {
            return Ok(()); // not fully on disk yet — resume next pass
        }
        // Everything is on disk: close the books and move on.
        let seg = self.seg.take().expect("finalizing segment");
        {
            let mut st = self.stats.lock().unwrap();
            st.segments_closed += 1;
            st.frames_written += seg.frames;
        }
        self.phase = Phase::Idle;
        if reopen {
            let carried = self.carry.take().expect("rotation carries the boundary AU");
            self.handle_au(ctx, carried)?;
        }
        Ok(())
    }

    /// One Annex-B AU: stash parameter sets, classify, open/rotate segments,
    /// reframe to length-prefixed NALs (ISO/IEC 14496-15 §5.3.4.2) and submit
    /// the block. May stash the buffer in `carry` when it triggers a rotation.
    fn handle_au(&mut self, ctx: &mut Ctx, buf: Buffer) -> Result<(), Error> {
        let au = buf.memory.data();
        let nals = split_nals(au);
        let mut keyframe = false;
        for n in &nals {
            match n.unit_type {
                NAL_SPS | NAL_PPS => {
                    // Refresh only between segments: a segment's CodecPrivate
                    // must describe every frame in it.
                    if self.seg.is_none() {
                        self.stash_param_set(n.raw);
                    }
                }
                NAL_SLICE_IDR => keyframe = true,
                _ => {}
            }
        }
        // The pipeline pts for this AU; a NONE (a depayloader's EOS tail
        // flush) reuses the last seen time rather than inventing a gap.
        let pts = buf
            .pts
            .nanos()
            .unwrap_or_else(|| self.seg.as_ref().map(|s| s.last_pts).unwrap_or(0));

        match &self.seg {
            None => {
                // Wait for a decodable entry point.
                if !(keyframe && self.sps.is_some() && self.pps.is_some()) {
                    self.skipped += 1;
                    return Ok(());
                }
                self.open_segment(ctx, pts)?;
            }
            Some(seg) => {
                if keyframe && pts.saturating_sub(seg.base_pts) >= self.segment_ns {
                    // Boundary: this AU opens the NEXT segment once the current
                    // one has fully drained and finalized.
                    self.carry = Some(buf);
                    self.phase = Phase::Finalizing { reopen: true };
                    return self.drive_finalize(ctx);
                }
            }
        }

        // Reframe: 4-octet big-endian length per NAL, start codes dropped.
        let total: usize = nals.iter().map(|n| 4 + n.raw.len()).sum();
        let mut mem = self.scratch.acquire_exact(total.max(1));
        {
            let dst = mem.as_mut_full();
            let mut at = 0;
            for n in &nals {
                dst[at..at + 4].copy_from_slice(&(n.raw.len() as u32).to_be_bytes());
                at += 4;
                dst[at..at + n.raw.len()].copy_from_slice(n.raw);
                at += n.raw.len();
            }
        }
        mem.set_len(total);

        let seg = self.seg.as_mut().expect("segment open");
        seg.last_pts = pts.max(seg.last_pts);
        seg.frames += 1;
        seg.writer
            .write_frame_scatter(&mut seg.out, 1, pts - seg.base_pts, mem, keyframe)
            .map_err(|e| Error::Resource(format!("mkvsegmentsink: frame: {e:?}")))?;
        self.submit_pieces(ctx);
        Ok(())
    }
}

impl Element for MkvSegmentSink {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| Error::Resource(format!("mkdir {}: {e}", self.dir.display())))?;
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        self.drain_completions(ctx)?;
        self.drive_finalize(ctx)?;
        loop {
            match self.phase {
                // Mid-rotation: input waits in the ring (the backpressure
                // buffer) until the swap completes.
                Phase::Finalizing { .. } => return Ok(Flow::Ok),
                Phase::Idle | Phase::Writing => {
                    // Retry pieces a full credit budget deferred last pass.
                    self.submit_pieces(ctx);
                    let Some(buf) = inputs.pop() else { return Ok(Flow::Ok) };
                    self.handle_au(ctx, buf)?;
                }
            }
        }
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if matches!(event, Event::Eos) {
            if matches!(self.phase, Phase::Writing) {
                self.phase = Phase::Finalizing { reopen: false };
            }
            self.drain_completions(ctx)?;
            self.drive_finalize(ctx)?;
        }
        Ok(())
    }

    // The sanctioned teardown exception (module docs): positioned writes through
    // the dup'ed fd because no reactor pass runs after stop.
    #[allow(clippy::disallowed_methods)]
    fn stop(&mut self, ctx: &mut Ctx) {
        // The teardown exception (module docs): a StopHandle stop outruns the
        // reactor, but the open segment must still end as a valid indexed file
        // (losing the tail GOP would be a recording gap). The scheduler settles
        // the reactor before stop, so draining here reconciles `in_flight` with
        // what actually reached the disk — a nonzero remainder would mean the
        // append offset lies (the 26 KB-hole lesson), so it is loud.
        use std::os::unix::fs::FileExt;
        if let Err(e) = self.drain_completions(ctx) {
            eprintln!("mkvsegmentsink: stop-drain: {e:?}");
        }
        if self.in_flight != 0 {
            eprintln!(
                "mkvsegmentsink: {} write(s) unaccounted at stop — the tail may be torn",
                self.in_flight
            );
        }
        let Some(mut seg) = self.seg.take() else { return };
        if std::env::var_os("PF_SEG_DEBUG").is_some() {
            let backlog: usize = seg
                .out
                .pieces()
                .map(|p| match p {
                    MuxPiece::Bytes { start, end } => end - start,
                    MuxPiece::Payload(m) => m.len(),
                })
                .sum();
            eprintln!(
                "segsink stop: write_offset={} in_flight={} backlog={} completed_bytes={}",
                self.write_offset,
                self.in_flight,
                backlog,
                self.stats.lock().unwrap().bytes_written,
            );
        }
        Self::compute_finalize(&mut seg);
        let mut offset = self.write_offset;
        while let Some(piece) = seg.out.pop_front() {
            let r = match piece {
                MuxPiece::Bytes { start, end } => {
                    let b = &seg.out.header_bytes()[start..end];
                    seg.spare.write_all_at(b, offset).map(|()| end - start)
                }
                MuxPiece::Payload(mem) => {
                    seg.spare.write_all_at(mem.data(), offset).map(|()| mem.len())
                }
            };
            match r {
                Ok(n) => offset += n as u64,
                Err(e) => {
                    eprintln!("mkvsegmentsink: stop-flush {}: {e}", seg.path.display());
                    return;
                }
            }
        }
        for p in &seg.patches {
            if let Err(e) = seg.spare.write_all_at(&p.bytes, p.offset) {
                eprintln!("mkvsegmentsink: stop-patch {}: {e}", seg.path.display());
                return;
            }
        }
        let mut st = self.stats.lock().unwrap();
        st.segments_closed += 1;
        st.frames_written += seg.frames;
        self.phase = Phase::Idle;
    }
}
