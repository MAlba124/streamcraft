//! [`Player`] — the thin controller API over probe → head → seek-index → autoplug (spec:
//! Milestone applications §5 — "give it any supported file and it plays it"; no-bins — the
//! player *is* the controller, the pipeline stays a flat graph). Deliberately thin: `pfplay`
//! is the only consumer today, so this exposes just what a CLI needs — open, inspect the
//! discovered tracks and duration, drive the transport handles, and run.
//!
//! The build sequence, in order:
//! 1. read a small prefix, [`probe`](crate::probe) the magic → a [`Kind`];
//! 2. per-container [`head`](crate::head) prep (MKV cluster scan / MP4 box walk);
//! 3. build the [`SeekIndex`](profluens_core::pipeline::SeekIndex) from container metadata;
//! 4. `filesrc → demux` (or `filesrc → parser/decoder` for an elementary stream), then
//!    [`preroll`](profluens_core::pipeline::Pipeline::preroll) to discover pads;
//! 5. [`autoplug`](crate::autoplug) each pad to a decoder + sink (or a drop sink).

use std::collections::HashMap;

use profluens_core::pipeline::{Pipeline, SeekIndex};
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSrc;

use crate::autoplug::{self, SinkChoice, TrackOutcome, Wiring};
use crate::probe::{self, Kind};
use crate::{head, seek};

/// Which real sinks to grab (a headless / silent run drops instead — see the CLI's
/// `--no-window` / `--no-audio`).
#[derive(Clone, Copy)]
pub struct SinkPolicy {
    pub video: SinkChoice,
    pub audio: SinkChoice,
}

impl Default for SinkPolicy {
    fn default() -> Self {
        SinkPolicy { video: SinkChoice::Device, audio: SinkChoice::Device }
    }
}

/// A built, prerolled, fully-linked player. Hold it, read [`tracks`](Self::tracks) /
/// [`duration`](Self::duration), install transport controls off the handles, then
/// [`run`](Self::run). The [`Pipeline`] is public so the CLI can pull the pause/seek/stop/
/// tap handles it needs — keeping [`Player`] thin rather than re-exporting each.
pub struct Player {
    pub pipeline: Pipeline,
    kind: Kind,
    wiring: Wiring,
    seek_index: SeekIndex,
    duration_ns: Option<u64>,
}

impl Player {
    /// Probe `path`, build and preroll the pipeline, and autoplug every track. Returns the
    /// ready-to-run player, or a one-line error (unknown container, unreadable head, or a
    /// preroll failure). A file with recognized container but *no* linkable track is NOT an
    /// error here — [`any_track_linked`](Self::any_track_linked) reports that, so the CLI
    /// can distinguish "unknown file" from "known file, nothing to play".
    pub fn open(path: &str, policy: SinkPolicy) -> Result<Player, String> {
        Self::open_inner(path, policy, None)
    }

    /// Open a player wired for the **zero-copy** VA-API display path: when `policy.video` is
    /// [`SinkChoice::ExternalZeroCopy`] and the video codec is hardware-decodable, the
    /// controller builds a DMA-BUF-exporting decoder whose `video/gpu` src `channel` the EGL
    /// frame-slot sink imports. Falls back to the readback External path (reported via
    /// [`video_zerocopy`](Self::video_zerocopy)) when the codec is software-only or no VA
    /// device is present. The caller shares `channel` with its EGL sink.
    pub fn open_zerocopy(
        path: &str,
        policy: SinkPolicy,
        channel: std::sync::Arc<pf_vaapi::gpuframe::GpuFrameChannel>,
    ) -> Result<Player, String> {
        Self::open_inner(path, policy, Some(channel))
    }

    fn open_inner(
        path: &str,
        policy: SinkPolicy,
        zc_channel: Option<std::sync::Arc<pf_vaapi::gpuframe::GpuFrameChannel>>,
    ) -> Result<Player, String> {
        let prefix = head::read_prefix(path, probe::PREFIX_LEN)
            .map_err(|e| format!("cannot read '{path}': {e}"))?;
        let kind = probe::probe(&prefix)?;
        let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        let mut p = Pipeline::new();
        // 1080p I420 is ~3.1 MiB/frame; 4 MiB shared slots leave headroom up to ~1600×1300.
        // Per-decoder pools (set in autoplug) keep decoded frames off this shared pool.
        p.set_pool(4 * 1024 * 1024, 24);

        let (wiring, seek_index, duration_ns) = if kind.is_elementary() {
            Self::build_elementary(&mut p, path, kind, file_len, policy)?
        } else {
            Self::build_container(&mut p, path, &prefix, kind, file_len, policy, zc_channel.as_ref())?
        };

        p.set_seek_index(seek_index.clone());
        Ok(Player { pipeline: p, kind, wiring, seek_index, duration_ns })
    }

    /// Build the container path: `filesrc → demux`, preroll, autoplug the discovered pads.
    #[allow(clippy::too_many_arguments)]
    fn build_container(
        p: &mut Pipeline,
        path: &str,
        prefix: &[u8],
        kind: Kind,
        file_len: u64,
        policy: SinkPolicy,
        zc_channel: Option<&std::sync::Arc<pf_vaapi::gpuframe::GpuFrameChannel>>,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        let src = p.add(FileSrc::new(path));

        // Per-container demuxer construction + seek index. `mp4_families` is the explicit
        // pad→family map MP4 needs (its demux src pads share one all-families offer menu that
        // leads with `bytes`, so the pad's declared family cannot select a decoder — the
        // per-track family lives on the reader instead; MKV publishes per-track menus and
        // needs no override).
        let mut mp4_families: HashMap<String, &'static str> = HashMap::new();
        // Per-audio-track channel counts (pad name → channels), so autoplug prefers a stereo
        // track over a 5.1 (avoids the flakier multichannel-AAC decode + the downmix).
        let mut audio_channels: HashMap<String, u16> = HashMap::new();
        let (demux, seek_info) = match kind {
            Kind::Mkv => {
                let header = head::mkv_header_prefix(path)
                    .map_err(|e| format!("mkv header: {e}"))?;
                let info = seek::mkv_seek_index(path, &header, file_len);
                // A throwaway reader parses the same header for each track's channel count;
                // the pad name mirrors the demuxer's `src_track<track_number>`.
                let mut probe = pf_mkv::MatroskaReader::new();
                let _ = probe.push(&header);
                for t in probe.tracks() {
                    if t.channels > 0 {
                        audio_channels.insert(format!("src_track{}", t.track_number), t.channels as u16);
                    }
                }
                (p.add(pf_mkv::MkvDemux::new(header)), info)
            }
            Kind::Mp4 => {
                let hb = head::mp4_head(path).map_err(|e| format!("mp4 head: {e}"))?;
                // The reader resolves the sample tables from the same head bytes — reused for
                // the (keyframe-exact) seek index AND the pad→family map, before the bytes are
                // handed to the demuxer. The pad name mirrors the demuxer's `src_track<id>`.
                let reader = pf_mp4::Mp4Reader::new(&hb)
                    .map_err(|e| format!("mp4 tables: {e:?}"))?;
                for t in reader.tracks() {
                    mp4_families.insert(format!("src_track{}", t.track_id), t.family());
                }
                let info = seek::mp4_seek_index(&reader, file_len);
                // Decode mode (`new`, not `passthrough`): NAL tracks reframe to Annex B, which
                // is what the software/hardware decoders take.
                (p.add(pf_mp4::Mp4Demux::new(hb)), info)
            }
            Kind::Ogg => {
                // Ogg's demuxer emits raw `bytes` packets on a single static pad (v1: first
                // logical bitstream only). The codec rides in the BOS page — sniff it from the
                // prefix so we can wire the right de-framer/decoder (or drop cleanly).
                return Self::build_ogg(p, src, prefix, file_len, policy);
            }
            Kind::Avi => {
                // AVI: `RIFF('AVI ' …)`, `hdrl` (stream headers) then `movi` (interleaved
                // chunks). `avidemux` publishes per-stream offer menus (like MKV — the pad's
                // declared family is the codec), so no MP4-style family override is needed.
                let header = head::avi_head(path).map_err(|e| format!("avi head: {e}"))?;
                let info = seek::avi_seek_index(&header, file_len);
                // Per-stream channel counts for the stereo preference (pad `src_stream<index>`).
                if let Ok((h, _)) = pf_avi::probe_header(&header) {
                    for s in &h.streams {
                        if s.channels > 0 {
                            audio_channels.insert(format!("src_stream{}", s.index), s.channels);
                        }
                    }
                }
                (p.add(pf_avi::AviDemux::new(header)), info)
            }
            _ => unreachable!("build_container called with an elementary kind"),
        };

        p.link((src, "src"), (demux, "sink")).map_err(|e| format!("filesrc ! demux: {e:?}"))?;
        let pads = p.preroll().map_err(|e| format!("preroll: {e:?}"))?;
        let resolver = |name: &str| mp4_families.get(name).copied();
        let family_of: Option<&autoplug::FamilyResolver<'_>> =
            if mp4_families.is_empty() { None } else { Some(&resolver) };
        let ch_resolver = |name: &str| audio_channels.get(name).copied();
        let audio_ch: Option<&autoplug::AudioChannels<'_>> =
            if audio_channels.is_empty() { None } else { Some(&ch_resolver) };
        let wiring = autoplug::autoplug_container(
            p, demux, &pads, family_of, audio_ch, policy.video, policy.audio, zc_channel,
        );
        Ok((wiring, seek_info.index, seek_info.duration_ns))
    }

    /// The Ogg path: sniff the first logical bitstream's codec magic from the prefix and wire
    /// `filesrc → oggdemux → oggflacdeframe → flacdec` for FLAC-in-Ogg; anything else (Opus,
    /// Vorbis, Theora — no decoder in-tree) is reported and the file plays nothing.
    fn build_ogg(
        p: &mut Pipeline,
        src: profluens_core::id::ElementId,
        prefix: &[u8],
        file_len: u64,
        policy: SinkPolicy,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        let codec = ogg_first_codec(prefix);
        let seek_info = seek::proportional(file_len, None); // no keyframe index for ogg v1
        let mut w = Wiring::default();
        match codec {
            OggCodec::Flac => {
                // filesrc → oggdemux → oggflacdeframe → flacdec → audio chain. The de-framer
                // strips the Ogg-FLAC mapping header back to a native `fLaC` stream.
                let demux = p.add(pf_ogg::OggDemux::new());
                let deframe = p.add(pf_flac::OggFlacDeframe::new());
                p.link((src, "src"), (demux, "sink")).map_err(|e| format!("filesrc ! oggdemux: {e:?}"))?;
                p.link((demux, "src"), (deframe, "sink")).map_err(|e| format!("oggdemux ! oggflacdeframe: {e:?}"))?;
                // The de-framer's byte src is the elementary FLAC stream — reuse the
                // elementary-audio wiring from its src pad.
                let sub = autoplug::wire_elementary_audio(
                    p,
                    deframe,
                    Box::new(pf_flac::FlacDec::new()),
                    "ogg/flac",
                    policy.audio,
                );
                w = sub;
            }
            other => {
                w.tracks.push(TrackOutcome {
                    pad: "ogg".to_string(),
                    summary: format!("(dropped: no decoder for {})", other.label()),
                    linked: false,
                });
            }
        }
        Ok((w, seek_info.index, seek_info.duration_ns))
    }

    /// Build an elementary (container-less) audio stream: `filesrc → parser/decoder → chain`.
    fn build_elementary(
        p: &mut Pipeline,
        path: &str,
        kind: Kind,
        file_len: u64,
        policy: SinkPolicy,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        let src = p.add(FileSrc::new(path));
        // No keyframe index; duration is unknown pre-decode, so time seeking is proportional
        // only when a duration turns up (it doesn't, here) — the honest v1.
        let seek_info = seek::proportional(file_len, None);

        let wiring = match kind {
            Kind::Flac => autoplug::wire_elementary_audio(
                p, src, Box::new(pf_flac::FlacDec::new()), "flac", policy.audio,
            ),
            Kind::Mp3 => autoplug::wire_elementary_audio(
                p, src, Box::new(pf_mp3::Mp3Dec::new()), "mp3", policy.audio,
            ),
            Kind::Wav => Self::build_wav(p, src, path, policy)?,
            Kind::AdtsAac => {
                // Bare ADTS: `filesrc → adtsparse → aacdec → audio chain`. `adtsparse`
                // synthesizes the ASC head from the first ADTS frame and strips each frame's
                // 7/9-byte header to the raw AU `aacdec` expects (spec: elementary-stream
                // reframing — the counterpart of a container's CodecPrivate + framing).
                let parse = p.add(pf_aac::AdtsParse::new());
                p.link((src, "src"), (parse, "sink"))
                    .map_err(|e| format!("filesrc ! adtsparse: {e:?}"))?;
                autoplug::wire_elementary_audio(
                    p, parse, Box::new(pf_aac::AacDec::new()), "adts", policy.audio,
                )
            }
            _ => unreachable!("build_elementary called with a container kind"),
        };
        Ok((wiring, seek_info.index, seek_info.duration_ns))
    }

    /// WAV: `filesrc → wavparse → audioconvert(→s16) → audio device`. `wavparse` emits raw
    /// `bytes` PCM and does not announce `audio/raw`, so the converter's input format is
    /// pinned from the header parsed controller-side (app IO, sanctioned) via
    /// [`AudioConvert::with_input`]. For a drop sink the bytes ride straight through.
    fn build_wav(
        p: &mut Pipeline,
        src: profluens_core::id::ElementId,
        path: &str,
        policy: SinkPolicy,
    ) -> Result<Wiring, String> {
        use profluens_audio::{AudioConvert, SampleFormat, WavParse};

        let parse = p.add(WavParse::new());
        p.link((src, "src"), (parse, "sink")).map_err(|e| format!("filesrc ! wavparse: {e:?}"))?;

        let mut w = Wiring::default();
        match policy.audio {
            // Device/External* all use the audio device (External* are video-only choices).
            SinkChoice::Device | SinkChoice::External | SinkChoice::ExternalZeroCopy => {
                // Parse the header up front (a small prefix read) to pin the converter input.
                let hdr = wav_header(path).ok_or_else(|| "wavparse: unreadable WAV header".to_string())?;
                let conv = p.add(AudioConvert::with_input(hdr, SampleFormat::S16));
                let sink = p.add(pf_pipewire::PipeWireAudioSink::new());
                p.link((parse, "src"), (conv, "sink")).map_err(|e| format!("wavparse ! audioconvert: {e:?}"))?;
                p.link((conv, "src"), (sink, "sink")).map_err(|e| format!("audioconvert ! audiosink: {e:?}"))?;
                w.audio_sink = Some(sink);
                w.tracks.push(TrackOutcome {
                    pad: "wav".to_string(),
                    summary: "wavparse → audioconvert(s16) → audio device".to_string(),
                    linked: true,
                });
            }
            SinkChoice::Drop => {
                use profluens_elements::testing::TestSink;
                let (ts, _s) = TestSink::new();
                let sink = p.add(ts);
                p.link((parse, "src"), (sink, "sink")).map_err(|e| format!("wavparse ! dropsink: {e:?}"))?;
                w.audio_sink = Some(sink);
                w.tracks.push(TrackOutcome {
                    pad: "wav".to_string(),
                    summary: "wavparse → drop sink".to_string(),
                    linked: true,
                });
            }
        }
        Ok(w)
    }

    /// The detected container/elementary kind.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// One [`TrackOutcome`] per discovered pad — the CLI prints these as `track <pad>: …`.
    pub fn tracks(&self) -> &[TrackOutcome] {
        &self.wiring.tracks
    }

    /// The raw-wire presenter's window↔app control channel (`PF_PRESENT`), if the
    /// `waylandvideosink` is in use — the CLI drains its clicks into pause/seek and publishes
    /// duration/pause back for the HUD.
    pub fn player_control(&self) -> Option<std::sync::Arc<pf_present::PlayerControl>> {
        self.wiring.player_control.clone()
    }

    /// Whether any track linked to a decoder/sink — the CLI's success predicate (audio-only
    /// and video-only files both count; only *zero* linked tracks is a failure).
    pub fn any_track_linked(&self) -> bool {
        self.wiring.any_linked()
    }

    /// The declared stream duration, if known (for the digit-seek and the summary line).
    pub fn duration(&self) -> Option<Timestamp> {
        self.duration_ns.map(Timestamp)
    }

    /// The seek index (byte↔time map), for the CLI's digit-seek resolution.
    pub fn seek_index(&self) -> &SeekIndex {
        &self.seek_index
    }

    /// The `(element, pad)` emitting the final composited `video/raw` frames when the player
    /// was opened with `SinkPolicy { video: SinkChoice::External, .. }` — the tap point an
    /// external presenter (the SDL3 GUI's clock-paced frame-slot appsink) links its own sink
    /// to. `None` unless `External` video was requested and a video track linked.
    pub fn video_out(&self) -> Option<(profluens_core::id::ElementId, &'static str)> {
        self.wiring.video_out
    }

    /// Whether the video was wired **zero-copy** (VA-API DMA-BUF export) — so
    /// [`video_out`](Self::video_out) emits `video/gpu` descriptors the EGL frame-slot sink
    /// imports, not `video/raw` pixels. `false` for the CPU/software path (including a
    /// requested zero-copy that fell back to readback). The GUI reads this to pick the
    /// matching frame-slot sink + presentation backend.
    pub fn video_zerocopy(&self) -> bool {
        self.wiring.video_zerocopy
    }

    /// The subtitle overlay element, if one was inserted (a video file with a text-subtitle
    /// track). A UI can toggle its visibility or inspect it.
    pub fn overlay(&self) -> Option<profluens_core::id::ElementId> {
        self.wiring.overlay
    }

    /// The ids the stats tap watches: `(label, id)` for each present decoder/sink.
    pub fn watched(&self) -> Vec<(&'static str, profluens_core::id::ElementId)> {
        let w = &self.wiring;
        [
            ("vdec", w.video_dec),
            ("adec", w.audio_dec),
            ("vsink", w.video_sink),
            ("asink", w.audio_sink),
        ]
        .into_iter()
        .filter_map(|(n, id)| id.map(|id| (n, id)))
        .collect()
    }

    /// Drive the pipeline to EOS / stop (delegates to [`Pipeline::run`]). Blocks the calling
    /// thread; install transport controls off `self.pipeline`'s handles beforehand.
    pub fn run(&mut self) -> Result<(), profluens_core::error::Error> {
        self.pipeline.run()
    }
}

/// Parse a WAV file's `AudioFormat` from a small prefix (app-side controller IO, before the
/// pipeline — the sanctioned `clippy.toml` exception). `None` on any read/parse failure.
#[allow(clippy::disallowed_methods)] // app-side WAV header parse before the pipeline; see clippy.toml
fn wav_header(path: &str) -> Option<profluens_audio::AudioFormat> {
    use std::io::Read;
    // 4 KiB comfortably covers the RIFF/fmt/data-header region before PCM data.
    let mut buf = vec![0u8; 4096];
    let mut f = std::fs::File::open(path).ok()?;
    let n = f.read(&mut buf).ok()?;
    buf.truncate(n);
    profluens_audio::parse_wav_header(&buf).ok().map(|h| h.format)
}

/// The Ogg first-bitstream codec, from the BOS page's mapping-header magic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OggCodec {
    Flac,
    Opus,
    Vorbis,
    Theora,
    Unknown,
}

impl OggCodec {
    fn label(self) -> &'static str {
        match self {
            OggCodec::Flac => "Ogg/FLAC",
            OggCodec::Opus => "Opus",
            OggCodec::Vorbis => "Vorbis",
            OggCodec::Theora => "Theora",
            OggCodec::Unknown => "an unknown Ogg codec",
        }
    }
}

/// Sniff the first logical bitstream's codec from an Ogg prefix by scanning for its BOS
/// mapping-header magic (each xiph mapping fixes a distinctive first-packet signature):
/// - FLAC-in-Ogg: `0x7F "FLAC"` (xiph "Ogg Mapping for FLAC" §mapping);
/// - Opus: `"OpusHead"` (RFC 7845 §5.1);
/// - Vorbis: `0x01 "vorbis"` (Vorbis I §4.2.1, the identification header);
/// - Theora: `0x80 "theora"` (Theora spec §6.2).
///
/// The magic sits in the first BOS page's packet, well within the typefind prefix. A pure
/// byte scan of the prefix is enough — we do not need to parse the page structure to *route*.
fn ogg_first_codec(prefix: &[u8]) -> OggCodec {
    let has = |needle: &[u8]| prefix.windows(needle.len()).any(|w| w == needle);
    if has(&[0x7F, b'F', b'L', b'A', b'C']) {
        OggCodec::Flac
    } else if has(b"OpusHead") {
        OggCodec::Opus
    } else if has(&[0x01, b'v', b'o', b'r', b'b', b'i', b's']) {
        OggCodec::Vorbis
    } else if has(&[0x80, b't', b'h', b'e', b'o', b'r', b'a']) {
        OggCodec::Theora
    } else {
        OggCodec::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ogg_codec_sniff() {
        // A synthetic BOS page: `OggS` header filler, then the mapping magic.
        let mut flac = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00".to_vec();
        flac.extend_from_slice(&[0x7F, b'F', b'L', b'A', b'C', 1, 0]);
        assert_eq!(ogg_first_codec(&flac), OggCodec::Flac);

        let mut opus = b"OggS\x00\x02".to_vec();
        opus.extend_from_slice(b"OpusHead\x01\x02");
        assert_eq!(ogg_first_codec(&opus), OggCodec::Opus);

        let mut vorbis = b"OggS\x00\x02".to_vec();
        vorbis.extend_from_slice(&[0x01, b'v', b'o', b'r', b'b', b'i', b's']);
        assert_eq!(ogg_first_codec(&vorbis), OggCodec::Vorbis);

        assert_eq!(ogg_first_codec(b"OggS\x00\x02nothing-known-here"), OggCodec::Unknown);
    }
}
