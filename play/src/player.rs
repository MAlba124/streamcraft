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
use crate::autoplug::{self, AudioTarget, SinkChoice, TrackOutcome, Wiring, ZeroCopyChannel};
use crate::chain::{ChainHandles, ChainSpec};
use crate::probe::{self, Kind};
use crate::source::SourceSpec;
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
    /// Whether this player was opened on a growing source.
    ///
    /// Deliberately a `bool` and **not** a retained [`FrontierHandle`]. `growingfilesrc` treats
    /// "every handle dropped without `finish()`/`abort()`" as an abandoned download and winds
    /// the stream down (`End::Abandoned`) — it detects that by the handle `Arc`'s strong count
    /// reaching one. A copy squirrelled away in here would hold that count at two forever, so a
    /// caller who dropped their handle (or whose downloader panicked) would get a pipeline
    /// parked at the frontier until the process died, instead of a clean EOS and a bus warning.
    /// The convenience of `player.frontier()` is not worth disarming that.
    growing: bool,
}

impl Player {
    /// Probe `path`, build and preroll the pipeline, and autoplug every track. Returns the
    /// ready-to-run player, or a one-line error (unknown container, unreadable head, or a
    /// preroll failure). A file with recognized container but *no* linkable track is NOT an
    /// error here — [`any_track_linked`](Self::any_track_linked) reports that, so the CLI
    /// can distinguish "unknown file" from "known file, nothing to play".
    pub fn open(path: &str, policy: SinkPolicy) -> Result<Player, String> {
        Self::open_inner(
            SourceSpec::path(path),
            policy.video,
            AudioTarget::Policy(policy.audio),
            None,
        )
    }

    /// Open with the **canonical-output audio chain** (spec: `gapless.md` Phase 2): the audio
    /// branch is forced through `audioconvert(f32) → audiostereo → audioresample(48k)` and the
    /// spec's optional stretch/gain stages, so whatever the file contains, the sink is offered
    /// [`CANONICAL`](crate::chain::CANONICAL) — f32, 48 kHz, 2 ch.
    ///
    /// This is the mode a shared `AudioOut` requires, because its device format is latched once
    /// for the life of the application and every track's producer must speak it. The sink
    /// itself is the spec's ([`SinkSpec::Injected`](crate::chain::SinkSpec::Injected) for the
    /// engine's producer element; a policy sink otherwise).
    ///
    /// `source` may be a complete file or one still downloading — see [`SourceSpec`] for the
    /// growing-file open policy. `video` chooses what happens to any video track in the
    /// container (audio-only sources ignore it); pass [`SinkChoice::Drop`] for a music player.
    ///
    /// Drive the built chain through [`audio_chain`](Self::audio_chain).
    pub fn open_canonical(
        source: SourceSpec,
        chain: ChainSpec,
        video: SinkChoice,
    ) -> Result<Player, String> {
        Self::open_inner(source, video, AudioTarget::Canonical(chain), None)
    }

    /// Open a player wired for the **zero-copy** VA-API display path: when `policy.video` is
    /// [`SinkChoice::ExternalZeroCopy`] and the video codec is hardware-decodable, the
    /// controller builds a DMA-BUF-exporting decoder whose `video/gpu` src `channel` the EGL
    /// frame-slot sink imports. Falls back to the readback External path (reported via
    /// [`video_zerocopy`](Self::video_zerocopy)) when the codec is software-only or no VA
    /// device is present. The caller shares `channel` with its EGL sink.
    ///
    /// Without the `video` feature [`ZeroCopyChannel`] is uninhabited, so this entry point
    /// keeps its signature but is uncallable — there is no VA-API to export a surface from.
    pub fn open_zerocopy(
        path: &str,
        policy: SinkPolicy,
        channel: std::sync::Arc<ZeroCopyChannel>,
    ) -> Result<Player, String> {
        Self::open_inner(
            SourceSpec::path(path),
            policy.video,
            AudioTarget::Policy(policy.audio),
            Some(channel),
        )
    }

    fn open_inner(
        mut source: SourceSpec,
        video: SinkChoice,
        audio: AudioTarget,
        zc_channel: Option<std::sync::Arc<ZeroCopyChannel>>,
    ) -> Result<Player, String> {
        // Typefind. For a growing source this is the one blocking wait in the whole open: it
        // returns as soon as the (64-byte) prefix is readable — see `crate::source`, policy 1.
        let prefix = source
            .read_prefix(probe::PREFIX_LEN)
            .map_err(|e| format!("cannot read '{}': {e}", source.path_str()))?;
        let kind = probe::probe(&prefix)?;
        // The length everything downstream measures against: the caller's `total_hint` for a
        // growing source, the file's own size otherwise (`crate::source`, policy 4).
        let on_disk = std::fs::metadata(source.path_str()).map(|m| m.len()).unwrap_or(0);
        let file_len = source.index_len(on_disk);
        let growing = source.is_growing();

        let mut p = Pipeline::new();
        // 1080p I420 is ~3.1 MiB/frame; 4 MiB shared slots leave headroom up to ~1600×1300.
        // Per-decoder pools (set in autoplug) keep decoded frames off this shared pool.
        p.set_pool(4 * 1024 * 1024, 24);

        let (wiring, seek_index, duration_ns) = if kind.is_elementary() {
            Self::build_elementary(&mut p, &mut source, kind, file_len, audio)?
        } else {
            Self::build_container(
                &mut p,
                &mut source,
                &prefix,
                kind,
                file_len,
                video,
                audio,
                zc_channel.as_ref(),
            )?
        };

        p.set_seek_index(seek_index.clone());
        Ok(Player { pipeline: p, kind, wiring, seek_index, duration_ns, growing })
    }

    /// Build the container path: `filesrc → demux`, preroll, autoplug the discovered pads.
    #[allow(clippy::too_many_arguments)]
    fn build_container(
        p: &mut Pipeline,
        source: &mut SourceSpec,
        prefix: &[u8],
        kind: Kind,
        file_len: u64,
        video: SinkChoice,
        audio: AudioTarget,
        zc_channel: Option<&std::sync::Arc<ZeroCopyChannel>>,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        // COLD: one path copy per open. `source` is borrowed mutably below (`take_element`),
        // so the path cannot stay a borrow of it.
        #[allow(clippy::disallowed_methods)]
        let path = source.path_str().to_string();
        let path = path.as_str();
        let src = p.add_boxed(source.take_element()?);

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
                let header = container_head(source, Kind::Mkv)?;
                // A throwaway reader parses the same header for each track's channel count;
                // the pad name mirrors the demuxer's `src_track<track_number>`.
                let mut probe = pf_mkv::MatroskaReader::new();
                let _ = probe.push(&header);
                // The Cues chain is resolved with a positioned read *of the file*, which a
                // growing source may not have received yet — so an incomplete download gets the
                // proportional index (over the hinted final length) plus the header's declared
                // duration, rather than a `pread` past the frontier.
                let info = if source.is_complete() {
                    seek::mkv_seek_index(path, &header, file_len)
                } else {
                    seek::proportional(file_len, probe.duration_ns())
                };
                for t in probe.tracks() {
                    if t.channels > 0 {
                        audio_channels.insert(format!("src_track{}", t.track_number), t.channels as u16);
                    }
                }
                (p.add(pf_mkv::MkvDemux::new(header)), info)
            }
            Kind::Mp4 => {
                let hb = container_head(source, Kind::Mp4)?;
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
                return Self::build_ogg(p, src, source, prefix, file_len, audio);
            }
            Kind::Avi => {
                // AVI: `RIFF('AVI ' …)`, `hdrl` (stream headers) then `movi` (interleaved
                // chunks). `avidemux` publishes per-stream offer menus (like MKV — the pad's
                // declared family is the codec), so no MP4-style family override is needed.
                let header = container_head(source, Kind::Avi)?;
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
        let wiring = autoplug::autoplug_container_with(
            p, demux, &pads, family_of, audio_ch, video, audio, zc_channel,
        );
        Ok((wiring, seek_info.index, seek_info.duration_ns))
    }

    /// The Ogg path: sniff the first logical bitstream's codec magic from the prefix and wire
    /// the matching chain — `filesrc → oggdemux → oggflacdeframe → flacdec` for FLAC-in-Ogg,
    /// `filesrc → oggdemux → opusdec` for Ogg-Opus. Anything else (Vorbis, Theora — no
    /// decoder in-tree) is reported and the file plays nothing.
    ///
    /// The two audio mappings need different amounts of de-framing, and the difference is
    /// the mapping specs', not ours:
    /// - **FLAC-in-Ogg** re-uses the *native* FLAC byte stream, so its first packet is a
    ///   native stream head behind a 9-byte `0x7F "FLAC"` mapping prefix (xiph "Ogg Mapping
    ///   for FLAC" §1). `flacdec` wants the native stream, so `oggflacdeframe` strips that
    ///   prefix and concatenates the packets back into one.
    /// - **Ogg-Opus** is packet-for-packet: "each Opus packet … is placed directly into an
    ///   Ogg packet" (RFC 7845 §3), and its two header packets (`OpusHead` §5.1, `OpusTags`
    ///   §5.2) are Opus's own, not an Ogg wrapper. `oggdemux` already emits **one Ogg packet
    ///   per buffer**, which *is* `opusdec`'s input contract, and `opusdec` consumes both
    ///   header packets itself — including arming the RFC 7845 §4.2 pre-skip trim from
    ///   `OpusHead`. So there is nothing left for a de-framer to do, and inserting one would
    ///   only risk breaking the packet boundaries the decoder depends on.
    // Everything here runs **once**, while the pipeline is being built: boxing a decoder to hand
    // to `wire_elementary_audio`, and naming a track for the report. Neither is on any streaming
    // path (spec: allocation discipline — one-time setup takes the documented exception).
    #[allow(clippy::disallowed_methods)]
    fn build_ogg(
        p: &mut Pipeline,
        src: profluens_core::id::ElementId,
        source: &SourceSpec,
        prefix: &[u8],
        file_len: u64,
        audio: AudioTarget,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        let path = source.path_str();
        let codec = ogg_first_codec(prefix);
        // No keyframe index for ogg v1, but both ends of the file give a duration — and a
        // duration is what turns the proportional fallback from "unavailable" into a
        // working seek bar. Both reads are best-effort: an unreadable end costs the
        // duration, not the playback.
        //
        // The tail read is gated on the whole file being readable (`crate::source`, policy 5).
        // Ogg's duration lives in the *last* page's granule position, so on a half-downloaded
        // file "the tail" is the download frontier and its granule is how far the download has
        // got — which would be reported as the length of the episode. Reporting no duration
        // until the download completes is the honest answer; there is no head-side alternative
        // for this container.
        let head = head::read_prefix(path, head::OGG_HEAD_LEN).unwrap_or_default();
        let tail = if source.is_complete() {
            head::read_tail(path, head::OGG_TAIL_LEN).unwrap_or_default().0
        } else {
            Vec::new()
        };
        let seek_info = seek::ogg_seek_index(&head, &tail, file_len);
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
                let sub = autoplug::wire_elementary_audio_with(
                    p,
                    deframe,
                    Box::new(pf_flac::FlacDec::new()),
                    "ogg/flac",
                    audio,
                );
                w = sub;
            }
            OggCodec::Opus => {
                // filesrc → oggdemux → opusdec → audio chain. No de-framer stage: see the
                // mapping note on this function. `oggdemux`'s src pad offers `bytes`, which
                // `opusdec`'s sink accepts alongside `opus`, so the link needs no bridge.
                let demux = p.add(pf_ogg::OggDemux::new());
                p.link((src, "src"), (demux, "sink"))
                    .map_err(|e| format!("filesrc ! oggdemux: {e:?}"))?;
                w = autoplug::wire_elementary_audio_with(
                    p,
                    demux,
                    Box::new(pf_opus::OpusDec::new()),
                    "ogg/opus",
                    audio,
                );
            }
            // Vorbis is the one common Ogg audio mapping still without an in-tree decoder
            // (adoption is being evaluated separately); Theora likewise for video. Both fall
            // through to the honest "no decoder" report rather than a half-wired chain.
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
        source: &mut SourceSpec,
        kind: Kind,
        file_len: u64,
        audio: AudioTarget,
    ) -> Result<(Wiring, SeekIndex, Option<u64>), String> {
        // COLD: one path copy per open (see `build_container`).
        #[allow(clippy::disallowed_methods)]
        let path = source.path_str().to_string();
        let path = path.as_str();
        let src = p.add_boxed(source.take_element()?);
        // An elementary stream has no header field for its duration and no element for its
        // index, but the *codec* leaves both lying around somewhere — in an MP3's first
        // frame, in a FLAC's metadata chain, in a WAV's `fmt `/`data` pair. Reading those
        // windows here (app-side IO before the pipeline, the sanctioned pattern) is what
        // gives the player a seek bar over a bare music file at all: without a duration,
        // `SeekIndex::resolve` has nothing to be proportional to and time seeking is simply
        // unavailable. ADTS AAC is the one that stays unknown — see the module docs.
        let seek_info = elementary_seek_info(source, kind, file_len);

        let wiring = match kind {
            Kind::Flac => autoplug::wire_elementary_audio_with(
                p, src, Box::new(pf_flac::FlacDec::new()), "flac", audio,
            ),
            Kind::Mp3 => autoplug::wire_elementary_audio_with(
                p, src, Box::new(pf_mp3::Mp3Dec::new()), "mp3", audio,
            ),
            Kind::Wav => Self::build_wav(p, src, path, audio)?,
            Kind::AdtsAac => {
                // Bare ADTS: `filesrc → adtsparse → aacdec → audio chain`. `adtsparse`
                // synthesizes the ASC head from the first ADTS frame and strips each frame's
                // 7/9-byte header to the raw AU `aacdec` expects (spec: elementary-stream
                // reframing — the counterpart of a container's CodecPrivate + framing).
                let parse = p.add(pf_aac::AdtsParse::new());
                p.link((src, "src"), (parse, "sink"))
                    .map_err(|e| format!("filesrc ! adtsparse: {e:?}"))?;
                autoplug::wire_elementary_audio_with(
                    p, parse, Box::new(pf_aac::AacDec::new()), "adts", audio,
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
        audio: AudioTarget,
    ) -> Result<Wiring, String> {
        use profluens_audio::{AudioConvert, SampleFormat, WavParse};

        let parse = p.add(WavParse::new());
        p.link((src, "src"), (parse, "sink")).map_err(|e| format!("filesrc ! wavparse: {e:?}"))?;

        let mut w = Wiring::default();
        match audio {
            // The canonical chain needs an announcing upstream, and `wavparse` emits raw
            // `bytes` without announcing `audio/raw` at all — so the *pinned* converter that
            // the policy path already uses becomes the chain's head here. Its output is
            // `audio/raw` at the header's rate/channels in the canonical sample format, which
            // is exactly what `wire_canonical_audio_chain` expects to be handed; the chain's
            // own `audioconvert(f32)` then sees f32 already and passes through byte for byte.
            // COLD: naming the track for the report, once per build (see `link_audio`).
            #[allow(clippy::disallowed_methods)]
            AudioTarget::Canonical(spec) => {
                let hdr = wav_header(path)
                    .ok_or_else(|| "wavparse: unreadable WAV header".to_string())?;
                let conv = p.add(AudioConvert::with_input(hdr, crate::chain::CANONICAL_SAMPLE));
                p.link((parse, "src"), (conv, "sink"))
                    .map_err(|e| format!("wavparse ! audioconvert: {e:?}"))?;
                let label = spec.sink.label();
                let h = crate::chain::wire_canonical_audio_chain(p, (conv, "src"), spec)?;
                w.audio_sink = Some(h.sink);
                w.audio_chain = Some(h);
                w.tracks.push(TrackOutcome {
                    pad: "wav".to_string(),
                    summary: format!("wavparse → audioconvert → {label}"),
                    linked: true,
                });
                return Ok(w);
            }
            // Device/External* all use the audio device (External* are video-only choices).
            AudioTarget::Policy(
                SinkChoice::Device | SinkChoice::External | SinkChoice::ExternalZeroCopy,
            ) => {
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
            AudioTarget::Policy(SinkChoice::Drop) => {
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
    ///
    /// Present only with the `video` feature: the presenter, and therefore the window this
    /// controls, is part of the video stack.
    #[cfg(feature = "video")]
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

    /// The canonical audio chain's handles, when this player was opened with
    /// [`open_canonical`](Self::open_canonical) and an audio track linked — the `audiogain` /
    /// `audiostretch` element ids for live property changes, the stretcher's source-time
    /// position counter, and the sink the chain ends in.
    ///
    /// Drive them against [`Pipeline::prop_handle`](profluens_core::pipeline::Pipeline::prop_handle),
    /// taken from [`self.pipeline`](Self::pipeline) after the topology is built:
    ///
    /// ```ignore
    /// let props = player.pipeline.prop_handle();
    /// player.audio_chain().unwrap().set_gain_db(&props, -3.5)?;
    /// ```
    ///
    /// `None` on the tiered (`Player::open`) path, which builds no such stages.
    pub fn audio_chain(&self) -> Option<&ChainHandles> {
        self.wiring.audio_chain.as_ref()
    }

    /// Whether this player is reading a still-downloading file.
    ///
    /// The [`FrontierHandle`] itself is *not* re-exposed here — the caller who built the spec
    /// with [`SourceSpec::growing`](crate::source::SourceSpec::growing) already owns it, and it
    /// is the last one alive besides the element's. That is what lets `growingfilesrc` notice
    /// an abandoned download and end the stream instead of parking on it forever; see the
    /// [`growing`](Self::growing) field note.
    pub fn is_growing(&self) -> bool {
        self.growing
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

/// Duration + time→byte index for an elementary stream, from whatever window its format
/// keeps the answers in ([`head`] reads, [`seek`] parses).
///
/// Every read here is best-effort: a head or tail that cannot be read costs the file its
/// seek bar, never its playback, so a failure degrades to the empty index rather than
/// failing the open. That matters because these reads are pure enrichment — the pipeline
/// itself needs none of them.
///
/// ADTS AAC is deliberately absent: an ADTS stream declares no duration anywhere and has no
/// index, so the only route to either is walking every frame header in the file. That is a
/// full read at open time for a format no music library uses, and reporting no duration is
/// the honest alternative.
/// On a **growing** source every read here is additionally clamped to the download frontier and
/// the tail read is skipped entirely until the whole file is readable (`crate::source`, policies
/// 2 and 5). That costs nothing that matters: all three formats state their duration in the
/// *head* — MP3 in the Xing/LAME frame, FLAC in STREAMINFO, WAV in the `fmt `/`data` pair — so
/// the answer is correct from the first kilobyte and does not drift as bytes arrive. The tail
/// only carries trailing ID3v1/APEv2 tags, whose absence costs a few bytes of accuracy in the
/// audio-length estimate, and reading it early would find the download frontier instead.
// COLD: app-side head/tail windows read once per open, before the pipeline exists — the
// documented `clippy.toml` exception (same as the container head builders in `crate::head`).
#[allow(clippy::disallowed_methods)]
fn elementary_seek_info(source: &SourceSpec, kind: Kind, file_len: u64) -> seek::SeekInfo {
    let path = source.path_str();
    // A growing source may not read past the frontier; `read_prefix` clamps to it (and to the
    // budget), so a head that has not arrived simply yields a shorter window and the parse
    // degrades to the proportional fallback rather than reading a torn append.
    let bounded_head = |want: usize| source.read_prefix(want).ok();
    match kind {
        Kind::Mp3 => {
            let head = if source.is_growing() {
                match bounded_head(head::MAX_MP3_HEAD.min(1024 * 1024) as usize) {
                    Some(h) => h,
                    None => return seek::proportional(file_len, None),
                }
            } else {
                let Ok(h) = head::mp3_head(path) else {
                    return seek::proportional(file_len, None);
                };
                h
            };
            let (tail, base) = if source.is_complete() {
                head::read_tail(path, head::MP3_TAIL_LEN).unwrap_or_default()
            } else {
                (Vec::new(), 0)
            };
            seek::mp3_seek_index(&head, &tail, base, file_len)
        }
        Kind::Flac => {
            let head = if source.is_growing() {
                bounded_head(1024 * 1024)
            } else {
                head::flac_head(path).ok()
            };
            match head {
                Some(h) => seek::flac_seek_index(&h, file_len),
                None => seek::proportional(file_len, None),
            }
        }
        Kind::Wav => match bounded_head(WAV_HEAD_LEN) {
            Some(head) => seek::wav_seek_index(&head, file_len),
            None => seek::proportional(file_len, None),
        },
        // ADTS AAC, and anything else that reaches here.
        _ => seek::proportional(file_len, None),
    }
}

/// The container head bytes for `kind`, honouring a growing source's frontier.
///
/// A complete file takes the existing path-based walk unchanged — same reads, same bounds, same
/// errors. A growing one walks a **bounded window** of what has downloaded
/// ([`SourceSpec::read_head_window`]) with [`head`]'s byte-slice variants, and, when the walk
/// says "the head is not all here yet", waits for the frontier to move and walks again, up to
/// [`crate::source::OPEN_TIMEOUT`]. A head that never arrives fails the open with the walk's own
/// message — you cannot demux a container whose track metadata is still downloading, and a
/// track-less player would be a far worse way to learn that.
fn container_head(source: &SourceSpec, kind: Kind) -> Result<Vec<u8>, String> {
    let path = source.path_str();
    if !source.is_growing() {
        return match kind {
            Kind::Mkv => head::mkv_header_prefix(path).map_err(|e| format!("mkv header: {e}")),
            Kind::Mp4 => head::mp4_head(path).map_err(|e| format!("mp4 head: {e}")),
            Kind::Avi => head::avi_head(path).map_err(|e| format!("avi head: {e}")),
            _ => Err(format!("no container head walk for {}", kind.label())),
        };
    }
    let deadline = SourceSpec::open_deadline();
    let mut last_len = 0u64;
    loop {
        let window = source
            .read_head_window()
            .map_err(|e| format!("cannot read '{path}': {e}"))?;
        let walked = match kind {
            Kind::Mkv => head::mkv_header_from(&window).map_err(|e| format!("mkv header: {e}")),
            Kind::Mp4 => head::mp4_head_from(&window).map_err(|e| format!("mp4 head: {e}")),
            Kind::Avi => head::avi_head_from(&window).map_err(|e| format!("avi head: {e}")),
            _ => return Err(format!("no container head walk for {}", kind.label())),
        };
        match walked {
            Ok(h) => return Ok(h),
            Err(e) => {
                // Not there yet — wait for the download to deliver more and walk again. When
                // the frontier will not move (finished, or the deadline passed), the walk's
                // error is the final answer.
                let have = source.readable().unwrap_or(0).max(window.len() as u64);
                if !source.wait_for_growth(have.max(last_len), deadline) {
                    return Err(e);
                }
                last_len = have;
            }
        }
    }
}

/// How much of a WAV to read for its header: the `RIFF`/`fmt `/`data` chunk region, plus
/// room for the `LIST`/`INFO` metadata a tagger may put in front of `data`.
const WAV_HEAD_LEN: usize = 4096;

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
