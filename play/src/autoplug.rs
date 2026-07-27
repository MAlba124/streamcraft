//! The autoplug controller (spec: streamcraft.md §no-bins — "decodebin's actual job …
//! is bus-listening logic issuing first-class relink operations. Ship an autoplug
//! controller as a **library, not as a magic self-modifying graph node**"; roadmap item 8
//! lists the "autoplug controller"). Given a prerolled pipeline and its discovered demux
//! pads, this picks a decoder per track by *negotiation* and wires the sinks — the exact
//! job `sdl3/examples/play_file.rs` did by hand for MKV, generalized across containers and
//! codecs.
//!
//! ## The negotiation-is-truth model, and its one sharp edge
//! A candidate decoder is chosen by trying to `link` it: a failed link with
//! [`Error::Resource`]/[`Error::Todo`] means "not this codec, try the next"; any other
//! error is real (spec: Formats — the intersection empties loudly at link time). This is
//! why we construct a fresh candidate per attempt (a failed link leaves an inert, never-run
//! spare element in the graph — harmless).
//!
//! The sharp edge: every demux src pad offers `[<codec-family>, bytes]` (the `bytes` escape
//! lets a generic byte sink tap any track — see `sc-mkv`'s `codec::offers_for`), and the
//! audio decoders accept `bytes` on their sink (`flacdec`/`mp3dec`/`aacdec` all bridge raw
//! bytes so a `filesrc`-of-a-`.flac` links). So *any* audio decoder byte-bridge-links to
//! *any* audio pad — pure negotiation cannot tell FLAC from AAC. We resolve this with the
//! pad's declared family ([`Pipeline::pad_families`]): the first offer is the real codec
//! family, so we only offer a decoder the pad whose codec it actually decodes. Video
//! decoders declare *only* their specific family on their sink (no `bytes` bridge), so for
//! video the link failure alone is a sufficient, self-correcting filter (an h264 track
//! simply won't link `vp8dec`).
//!
//! ## Sink policy (v1, kept simple)
//! First video pad that links gets the video sink; first audio pad gets the audio sink;
//! every other discovered pad (extra tracks, subtitles, codecs with no decoder) is linked
//! to a drop sink so nothing accumulates unboundedly (spec: unlinked dynamic pads buffer
//! without bound — an explicit drop is today's honest equivalent). The audio sink provides
//! the pipeline clock (sinks-first selection), so video paces on the DAC.
//!
//! ## Pool/queue policy (carried verbatim from play_file, with the WHY)
//! The numbers below encode measured deadlock/stutter post-mortems, not guesses — see the
//! inline comments at each `set_element_pool`/`set_queue_capacity`.

use streamcraft_core::element::{Element, SchedHint};
use streamcraft_core::error::Error;
use streamcraft_core::id::ElementId;
use streamcraft_core::pipeline::Pipeline;

use streamcraft_audio::{AudioConvert, AudioDownmix, AudioResample, SampleFormat};
use streamcraft_elements::flow::Queue;
use streamcraft_elements::testing::TestSink;

use sc_text::{PgsDec, SubParse, SubtitleOverlay};

/// Where a decoder's output goes: a real device sink, or a drop sink (headless / no
/// device). The controller stays agnostic to which — the CLI picks per `--no-window` /
/// `--no-audio`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SinkChoice {
    /// A real presentation sink (SDL3 window for video, PipeWire device for audio).
    Device,
    /// A drop sink (`TestSink`) — decode-and-discard, no window/device grabbed.
    Drop,
    /// Video only: the controller adds **no** sink and instead records the tap point
    /// ([`Wiring::video_out`]) — the element+pad emitting decoded (subtitle-composited)
    /// `video/raw` frames. An external presenter (the SDL3 GUI's clock-paced frame-slot
    /// appsink) links its own sink there. The pipeline still needs an audio device sink for
    /// the clock, so this is a *video* choice paired with `audio: Device`.
    External,
    /// Like [`External`](Self::External), but **zero-copy**: try to build a VA-API decoder in
    /// DMA-BUF-export mode (`sc_vaapi::video_decoder_zerocopy_for`) so the decoded surface
    /// reaches the GUI without a CPU round-trip. The subtitle overlay cannot touch a GPU-only
    /// frame, so this path taps `decoder.src` **directly** (no suboverlay — subtitles are a
    /// documented follow-up here). Falls back to the plain `External` path (software/readback
    /// decoder + `video/raw` tap) when the family is not hardware-decodable or no device is
    /// present, and reports which happened via [`Wiring::video_zerocopy`]. Requires the
    /// zero-copy channel be supplied to [`autoplug_container`].
    ExternalZeroCopy,
}

/// What autoplug decided for one discovered pad, for the CLI's `track <pad>: <what>` line.
pub struct TrackOutcome {
    /// The demux pad's name (`src_track<N>`, or the elementary synthetic name).
    pub pad: String,
    /// A human summary: which decoder linked, or why the pad was dropped.
    pub summary: String,
    /// True when a decoder was linked (any track linking to a decoder is "success" for the
    /// process exit code; a file with zero linked tracks is the failure case).
    pub linked: bool,
}

/// The result of wiring a whole file: the per-pad outcomes plus the ids the CLI's stats tap
/// wants to watch.
#[derive(Default)]
pub struct Wiring {
    pub tracks: Vec<TrackOutcome>,
    pub video_dec: Option<ElementId>,
    pub audio_dec: Option<ElementId>,
    pub video_sink: Option<ElementId>,
    pub audio_sink: Option<ElementId>,
    /// The `(element, pad)` emitting the final `video/raw` frames when [`SinkChoice::External`]
    /// is used — the subtitle overlay's `src` if subtitles are composited, else the video
    /// decoder's `src`. An external presenter links its own sink here. `None` for the
    /// Device/Drop sink choices (which wire their own sink).
    pub video_out: Option<(ElementId, &'static str)>,
    /// The subtitle overlay element, when one was inserted (a video + text-subtitle file).
    /// Exposed so a UI can toggle/inspect it; `None` when there is no subtitle to composite.
    pub overlay: Option<ElementId>,
    /// True when the video decoder was wired **zero-copy** (VA-API DMA-BUF export) and
    /// [`video_out`](Self::video_out) emits `video/gpu` descriptors, not `video/raw` pixels —
    /// the presenter must be the EGL frame-slot sink. `false` for the CPU/software path
    /// (including a [`SinkChoice::ExternalZeroCopy`] that fell back to readback).
    pub video_zerocopy: bool,
}

impl Wiring {
    /// Whether any pad linked to a decoder — the "did we play anything" predicate the exit
    /// code keys on (audio-only and video-only files both count as success).
    pub fn any_linked(&self) -> bool {
        self.tracks.iter().any(|t| t.linked)
    }
}

/// The video decoder candidate order (software), by the demux family they decode. Hardware
/// (`sc_vaapi::video_decoder_for`) is tried first, ahead of these, and is capability-gated
/// (no device or `SC_NO_VAAPI` set → skipped). Each family string is exactly what the
/// demuxers announce (`sc-mkv`/`sc-mp4` `codec::family_for` / decode-mode families).
fn make_video_decoder(family: &str) -> Option<Box<dyn Element>> {
    Some(match family {
        // MP4 decode-mode and MKV both emit Annex B for AVC/HEVC (Mp4Demux::new, not
        // passthrough), so both software decoders take `h264/annexb` / `h265/annexb`.
        "h264/annexb" => Box::new(sc_h264::H264Dec::new()),
        "h265/annexb" => Box::new(sc_h265::H265Dec::new()),
        "vp8" => Box::new(sc_vp8::Vp8Dec::new()),
        "vp9" => Box::new(sc_vp9::Vp9Dec::new()),
        "av1" => Box::new(sc_av1::Av1Dec::new()),
        // MPEG-4 Part 2 ASP (XviD/DivX) — the video of classic `.avi` rips.
        "mpeg4/asp" => Box::new(sc_mpeg4p2::Mpeg4p2Dec::new()),
        _ => return None,
    })
}

/// The audio decoder for a demux family, or `None` for a codec with no in-tree decoder
/// (Opus/Vorbis — dropped cleanly with a printed reason). The family is the pad's first
/// declared offer, so this is codec-exact: we never hand `flacdec` an `aac` pad just
/// because both accept `bytes`.
fn make_audio_decoder(family: &str) -> Option<Box<dyn Element>> {
    Some(match family {
        "aac" => Box::new(sc_aac::AacDec::new()),
        "flac" => Box::new(sc_flac::FlacDec::new()),
        // `mp3` appears as a demux family via the elementary path and mkv/avi `A_MPEG/L3`.
        "mp3" => Box::new(sc_mp3::Mp3Dec::new()),
        // Dolby: AC-3 (ATSC A/52) and E-AC-3 (Annex E) — the 5.1 tracks in BluRay/WEB-DL/AVI
        // rips. Decode to interleaved S16 in canonical L,R,C,LFE,Ls,Rs order; a `>2ch` sink
        // path inserts the BS.775 downmix (see `wire_audio_chain`).
        "ac3" => Box::new(sc_ac3::Ac3Dec::new()),
        "eac3" => Box::new(sc_ac3::Eac3Dec::new()),
        _ => return None,
    })
}

/// The video-pool policy: a decoded 1080p I420 frame is ~3.1 MiB; 4 MiB slots hold up to
/// ~1600×1300. `24` slots is the shared-pool default the pipeline is set to; a video
/// decoder gets its **own** 4 MiB × 24 pool (see [`wire_video`] for why) and a queue of 16.
const VIDEO_SLOT: usize = 4 * 1024 * 1024;
const VIDEO_SLOTS: u32 = 24;
const VIDEO_QUEUE: usize = 16;
/// The audio-pool policy: a decoded frame is a few KB, so a decoder gets a small-slot pool
/// (64 KiB × 64) and a deep queue (64) — the interleave slack (see [`wire_audio`]).
const AUDIO_SLOT: usize = 64 * 1024;
const AUDIO_SLOTS: u32 = 64;
const AUDIO_QUEUE: usize = 64;

/// How autoplug learns a discovered pad's real codec family. Some demuxers publish a
/// *per-track* offer menu whose first family is the codec ([`sc_mkv::MkvDemux`] via
/// `codec::offers_for`) — [`Pipeline::pad_families`] reads it directly. Others share one
/// all-families menu across every dynamic pad ([`sc_mp4::Mp4Demux`], whose menu leads with
/// `bytes`) and expose the per-track family only through their reader — the controller then
/// supplies an explicit `pad → family` map. This closure is that override, consulted before
/// the pad's declared menu.
pub type FamilyResolver<'a> = dyn Fn(&str) -> Option<&'static str> + 'a;

/// Resolves a discovered audio pad's declared channel count from the container's track
/// metadata, so autoplug can prefer a stereo/mono track over a multichannel one. A 5.1 AAC
/// track usually ships beside an explicit **stereo** AAC mix; decoding the stereo one avoids
/// both the flakier multichannel-AAC decode path and the downmix. `None` for a pad whose
/// channel count the container did not declare (autoplug then falls back to track order).
pub type AudioChannels<'a> = dyn Fn(&str) -> Option<u16> + 'a;

/// Autoplug and wire every discovered pad (spec: no-bins — the controller). `pads` are the
/// preroll-discovered demux src pads; `demux` is the demuxer element they live on.
/// `family_of`, when it returns `Some`, overrides the pad's declared family menu (see
/// [`FamilyResolver`] — MP4's shared menu makes this necessary). `video_sink`/`audio_sink`
/// choose device vs drop.
///
/// The shared pipeline pool is expected to already be sized (4 MiB × 24) by the caller; this
/// only sets the *per-decoder* pools/queues, whose sizing is the load-bearing part.
#[allow(clippy::too_many_arguments)]
pub fn autoplug_container(
    p: &mut Pipeline,
    demux: ElementId,
    pads: &[streamcraft_core::pipeline::AddedPadInfo],
    family_of: Option<&FamilyResolver<'_>>,
    audio_channels: Option<&AudioChannels<'_>>,
    video_sink: SinkChoice,
    audio_sink: SinkChoice,
    zc_channel: Option<&std::sync::Arc<sc_vaapi::gpuframe::GpuFrameChannel>>,
) -> Wiring {
    let mut w = Wiring::default();

    // Resolve every pad's real codec family once (the explicit override — MP4 — wins; else the
    // pad's declared menu, whose first family is the codec for per-track-menu demuxers — MKV).
    let fam: Vec<&'static str> = pads
        .iter()
        .map(|ap| {
            family_of
                .and_then(|f| f(&ap.name))
                .or_else(|| p.pad_families(demux, &ap.name).and_then(|fs| fs.first().copied()))
                .unwrap_or("bytes")
        })
        .collect();

    // Subtitle pre-scan: an overlay only makes sense with BOTH a video track to composite onto
    // and a subtitle track to show. The overlay must exist before the video is wired (video is
    // usually the first pad), so we decide up front and create it here — the video branch links
    // the decoder to `overlay.video`, and the subtitle branch (below) feeds `overlay.text`.
    let has_video = fam.iter().any(|f| make_video_decoder(f).is_some() || is_hw_video_family(f));
    let sub_idx = fam.iter().position(|f| is_handleable_subtitle(f));

    // Audio track selection: among the decodable audio tracks, prefer a stereo/mono one when
    // the container declares channel counts (Legally Blonde/Coneheads ship a stereo AAC beside
    // the 5.1 — decoding it dodges the flakier multichannel-AAC path and the downmix). Falls
    // back to the first decodable audio track (the downmix then folds any >2ch to stereo).
    let audio_cands: Vec<usize> =
        (0..pads.len()).filter(|&i| make_audio_decoder(fam[i]).is_some()).collect();
    let audio_idx = audio_channels
        .and_then(|res| {
            audio_cands.iter().copied().find(|&i| matches!(res(&pads[i].name), Some(1..=2)))
        })
        .or_else(|| audio_cands.first().copied());
    let overlay = if has_video && sub_idx.is_some() {
        let ov = p.add(SubtitleOverlay::new());
        w.overlay = Some(ov);
        Some(ov)
    } else {
        None
    };

    for (i, ap) in pads.iter().enumerate() {
        let family = fam[i];

        // Video first, once: the first video pad that links gets the window (through the
        // subtitle overlay when present). A video decoder only declares its own family, so a
        // non-matching link fails cleanly and we fall through.
        if w.video_dec.is_none() {
            // Zero-copy first (when requested + a channel supplied): a VA-API decoder in
            // DMA-BUF-export mode taps `decoder.src` DIRECTLY (no suboverlay — a CPU overlay
            // cannot touch a GPU-only frame; subtitles are the documented follow-up). Its src
            // is `video/gpu`, so it can only link a `video/gpu` presenter.
            if video_sink == SinkChoice::ExternalZeroCopy {
                if let Some(ch) = zc_channel {
                    if let Some((dec_id, name)) =
                        try_link_video_zerocopy(p, (ap.element, &ap.name), family, ch)
                    {
                        wire_video_zerocopy(p, dec_id, &mut w);
                        w.tracks.push(TrackOutcome {
                            pad: ap.name.clone(),
                            summary: format!("{family} → {name}(zero-copy) → external sink"),
                            linked: true,
                        });
                        w.video_dec = Some(dec_id);
                        w.video_zerocopy = true;
                        continue;
                    }
                }
            }
            // The readback/software path (also the ExternalZeroCopy fallback: a software codec
            // or no device → a normal decoder + the `video/raw` External tap).
            let effective_sink = if video_sink == SinkChoice::ExternalZeroCopy {
                SinkChoice::External // fell back — present via the CPU path
            } else {
                video_sink
            };
            if let Some((dec_id, name)) = try_link_video(p, (ap.element, &ap.name), family) {
                wire_video(p, dec_id, effective_sink, overlay, &mut w);
                let via = if overlay.is_some() { " → suboverlay" } else { "" };
                w.tracks.push(TrackOutcome {
                    pad: ap.name.clone(),
                    summary: format!(
                        "{family} → {name}{via} → {}",
                        sink_label(effective_sink, true)
                    ),
                    linked: true,
                });
                w.video_dec = Some(dec_id);
                continue;
            }
        }

        // Then audio: only the chosen track (stereo-preferred, else first) gets decoded and
        // sunk; other audio tracks fall through to a drop. Family-exact selection (the byte
        // bridge would otherwise let any audio decoder claim any audio pad).
        if Some(i) == audio_idx {
            if let Some(dec) = make_audio_decoder(family) {
                match link_audio(p, (ap.element, &ap.name), dec, audio_sink, &mut w) {
                    Ok(name) => {
                        w.tracks.push(TrackOutcome {
                            pad: ap.name.clone(),
                            summary: format!("{family} → {name} → {}", sink_label(audio_sink, false)),
                            linked: true,
                        });
                        continue;
                    }
                    Err(e) => {
                        // A real negotiation failure on a family we *should* decode — report
                        // it, then drop the pad so nothing backs up.
                        drop_pad(p, (ap.element, &ap.name));
                        w.tracks.push(TrackOutcome {
                            pad: ap.name.clone(),
                            summary: format!("(dropped: {family} decoder failed to link: {e})"),
                            linked: false,
                        });
                        continue;
                    }
                }
            }
        }

        // The chosen text-subtitle track, when an overlay is present: route it through a
        // `queue → subparse → overlay.text` branch so it burns into the video (spec: subtitle
        // support — the overlay is a non-blocking fan-in; the `queue` heads its own active
        // group, a passive demux fan-out cannot). Only the FIRST subtitle track is shown.
        if let (Some(ov), true) = (overlay, Some(i) == sub_idx) {
            match wire_subtitle(p, (ap.element, &ap.name), ov, family, &mut w) {
                Ok(()) => {
                    let path = if family == "subtitle/pgs" {
                        "pgsdec → suboverlay.image"
                    } else {
                        "subparse → suboverlay.text"
                    };
                    w.tracks.push(TrackOutcome {
                        pad: ap.name.clone(),
                        summary: format!("{family} → {path}"),
                        linked: true,
                    });
                    continue;
                }
                Err(e) => {
                    drop_pad(p, (ap.element, &ap.name));
                    w.tracks.push(TrackOutcome {
                        pad: ap.name.clone(),
                        summary: format!("(dropped: subtitle wiring failed: {e})"),
                        linked: false,
                    });
                    continue;
                }
            }
        }

        // Everything else: extra tracks, an unshown/undecodable subtitle, or a codec with no
        // in-tree decoder (Opus/Vorbis/Theora → the demuxer's `bytes` family). Drop it so its
        // buffers never accumulate.
        drop_pad(p, (ap.element, &ap.name));
        w.tracks.push(TrackOutcome {
            pad: ap.name.clone(),
            summary: format!("(dropped: {})", drop_reason(family)),
            linked: false,
        });
    }
    w
}

/// The text-subtitle families the [`SubtitleOverlay`] shows via [`SubParse`] (`subtitle/srt`,
/// `subtitle/vtt`, `subtitle/ass`).
fn is_text_subtitle(family: &str) -> bool {
    matches!(family, "subtitle/srt" | "subtitle/vtt" | "subtitle/ass")
}

/// Every subtitle family the overlay can composite: the text families plus PGS
/// (`subtitle/pgs`, image-based, via [`PgsDec`] into the overlay's `image` pad).
fn is_handleable_subtitle(family: &str) -> bool {
    is_text_subtitle(family) || family == "subtitle/pgs"
}

/// Wire a subtitle pad into the overlay: `pad → queue → {subparse|pgsdec} → overlay.{text|image}`.
/// The `queue` (Active passthrough) heads the branch's own scheduler group — the demuxer is a
/// passive fan-out and cannot head two branches — and the parser/decoder (passive) inlines into
/// it. Text (SRT/VTT/ASS) normalizes to `subtitle/events` for the overlay's `text` pad; PGS
/// decodes the RLE bitmap to `subtitle/bitmap` RGBA for the overlay's `image` pad.
fn wire_subtitle(
    p: &mut Pipeline,
    pad: (ElementId, &str),
    overlay: ElementId,
    family: &str,
    w: &mut Wiring,
) -> Result<(), String> {
    let q = p.add(Queue::new());
    p.link(pad, (q, "sink")).map_err(|e| format!("demux ! queue: {e:?}"))?;
    if family == "subtitle/pgs" {
        // Bitmap (PGS/BluRay): queue → pgsdec → overlay.image — RGBA alpha-over the frame.
        let dec = p.add(PgsDec::new());
        p.link((q, "src"), (dec, "sink")).map_err(|e| format!("queue ! pgsdec: {e:?}"))?;
        p.link((dec, "src"), (overlay, "image")).map_err(|e| format!("pgsdec ! overlay.image: {e:?}"))?;
    } else {
        // Text (SRT/VTT/ASS): queue → subparse → overlay.text.
        let sp = p.add(SubParse::new());
        p.link((q, "src"), (sp, "sink")).map_err(|e| format!("queue ! subparse: {e:?}"))?;
        p.link((sp, "src"), (overlay, "text")).map_err(|e| format!("subparse ! overlay.text: {e:?}"))?;
    }
    w.overlay = Some(overlay);
    Ok(())
}

/// Why a pad was dropped, for the `track` line. Distinguishes "no decoder for this codec"
/// from "we already have a track of this kind" (v1 plays one video + one audio) — this is
/// only reached once a video/audio slot is already filled or the family is unknown.
fn drop_reason(family: &str) -> String {
    if make_video_decoder(family).is_some() || is_hw_video_family(family) {
        return "extra video track (v1 plays the first)".into();
    }
    if make_audio_decoder(family).is_some() {
        return "extra audio track (v1 plays the first)".into();
    }
    if family.starts_with("subtitle/") {
        // A subtitle beyond the first shown (text or PGS), or a subtitle format with no
        // overlay path (e.g. DVB/VobSub — a future `*dec` onto the `subtitle/bitmap` family).
        return if is_handleable_subtitle(family) {
            "extra subtitle track (v1 shows the first)".into()
        } else {
            format!("no overlay for {family}")
        };
    }
    format!("no decoder for {family}")
}

/// Whether a family could be hardware-decoded (so an unlinked one is an "extra track", not
/// "unsupported"). h264 + h265 today (matching `sc_vaapi::video_decoder_for`); both also have
/// software decoders, so this only refines the drop message for a *second* such track.
fn is_hw_video_family(family: &str) -> bool {
    matches!(family, "h264/annexb" | "h265/annexb")
}

/// Try to link a video decoder for `family` to `pad`: hardware first (capability-gated),
/// then the software decoder for the family. Returns the decoder id and a display name.
/// A `Resource`/`Todo` link error means "not this codec" — but since we pre-filter by
/// family, that should only happen on a genuine bitstream/profile mismatch, which we let
/// fall through to a drop.
fn try_link_video(
    p: &mut Pipeline,
    pad: (ElementId, &str),
    family: &str,
) -> Option<(ElementId, &'static str)> {
    // Hardware-first (spec: play_file — the probe gates construction; `SC_NO_VAAPI` or no
    // device → `None`). A hardware decoder that constructs but fails to link (unsupported
    // profile) falls through to software.
    if let Some(hw) = sc_vaapi::video_decoder_for(family) {
        let id = p.add_boxed(hw);
        let hw_name = match family {
            "h265/annexb" => "vaapih265dec",
            _ => "vaapih264dec",
        };
        match p.link(pad, (id, "sink")) {
            Ok(_) => return Some((id, hw_name)),
            Err(Error::Resource(_)) | Err(Error::Todo(_)) => {} // fall through to software
            Err(_) => {} // real error — still fall through; software may cope
        }
    }
    let sw = make_video_decoder(family)?;
    let name = video_dec_name(family);
    let id = p.add_boxed(sw);
    match p.link(pad, (id, "sink")) {
        Ok(_) => Some((id, name)),
        Err(_) => None, // genuine mismatch — caller drops the pad
    }
}

/// Try to link a **zero-copy** VA-API decoder (`video_decoder_zerocopy_for`) for `family`
/// to `pad`. `None` when the family is not hardware-decodable, no VA device is present, or
/// the link fails (the caller then falls back to the readback/software path). The returned
/// decoder's src is `video/gpu` — only an EGL frame-slot presenter can link it.
fn try_link_video_zerocopy(
    p: &mut Pipeline,
    pad: (ElementId, &str),
    family: &str,
    channel: &std::sync::Arc<sc_vaapi::gpuframe::GpuFrameChannel>,
) -> Option<(ElementId, &'static str)> {
    let hw = sc_vaapi::video_decoder_zerocopy_for(family, std::sync::Arc::clone(channel))?;
    let hw_name = match family {
        "h265/annexb" => "vaapih265dec",
        _ => "vaapih264dec",
    };
    let id = p.add_boxed(hw);
    match p.link(pad, (id, "sink")) {
        Ok(_) => Some((id, hw_name)),
        // A HW decoder that constructs but won't link (unsupported profile) — fall back.
        Err(_) => None,
    }
}

/// Wire a zero-copy video decoder as the External tap: `decoder.src` (`video/gpu`) is the
/// `video_out` point the EGL frame-slot sink links to. Applies the same per-decoder video
/// pool/queue policy as [`wire_video`] (the `video/gpu` buffers are tiny descriptors, but
/// the decoder still needs its own pool + a deep inbound queue for the demux interleave).
fn wire_video_zerocopy(p: &mut Pipeline, dec: ElementId, w: &mut Wiring) {
    p.set_element_pool(dec, VIDEO_SLOT, VIDEO_SLOTS);
    p.set_queue_capacity(dec, VIDEO_QUEUE);
    // No sink, no overlay — the tap is the decoder's `video/gpu` src.
    w.video_out = Some((dec, "src"));
}

/// A stable display name for a software video decoder family.
fn video_dec_name(family: &str) -> &'static str {
    match family {
        "h264/annexb" => "h264dec",
        "h265/annexb" => "h265dec",
        "vp8" => "vp8dec",
        "vp9" => "vp9dec",
        "av1" => "av1dec",
        "mpeg4/asp" => "mpeg4p2dec",
        _ => "videodec",
    }
}

/// Wire a linked video decoder to its sink and apply the video pool/queue policy. When
/// `overlay` is `Some`, the frame's path to the sink runs through the subtitle overlay
/// (`dec → overlay.video`, then `overlay.src → sink`) so subtitles burn into the picture.
fn wire_video(
    p: &mut Pipeline,
    dec: ElementId,
    sink: SinkChoice,
    overlay: Option<ElementId>,
    w: &mut Wiring,
) {
    // The decoder gets its OWN pool (spec: pool negotiation, explicit v1): decoded frames
    // must never compete with the shared default pool that filesrc reads and the demuxer's
    // held input backlog drink from. With one shared pool the graph deadlocks — the
    // output-blocked demuxer pins its input slots, the pool hits its cap, and the decoder
    // (the only element that could unblock the chain) can never allocate the frame it is
    // carrying (measured: outstanding=24/24, all counters flat). See play_file.
    p.set_element_pool(dec, VIDEO_SLOT, VIDEO_SLOTS);
    // A deep inbound ring: the demuxer's thread emits BOTH tracks, and a full ring on either
    // pad blocks it — starving the other. With the default 4-batch rings it ping-pongs
    // between blocked-on-video and blocked-on-audio: video halves, audio underruns. Depth
    // here is the interleave slack.
    p.set_queue_capacity(dec, VIDEO_QUEUE);

    // Route through the subtitle overlay when present; the overlay composites in place and
    // forwards `video/raw`, so no extra pool is needed (it reuses the decoder's frame).
    let (out_el, out_pad): (ElementId, &'static str) = match overlay {
        Some(ov) => {
            p.link((dec, "src"), (ov, "video")).expect("videodec ! suboverlay.video");
            (ov, "src")
        }
        None => (dec, "src"),
    };

    match sink {
        SinkChoice::Device => {
            let s = p.add_boxed(Box::new(
                sc_sdl3::Sdl3VideoSink::new().with_title("streamcraft — scplay"),
            ));
            // The decoder/overlay emits `video/raw`; the SDL sink takes i420/gray8/nv12.
            // Direct link — the renderer owns any CSC, so no CPU convert element is needed.
            p.link((out_el, out_pad), (s, "sink")).expect("video ! sdl3videosink");
            w.video_sink = Some(s);
        }
        SinkChoice::Drop => {
            // A decoded frame is `video/raw`, which `TestSink` (bytes only) cannot take —
            // `VideoCkSink` accepts `video/raw` and just checksums/counts frames, the honest
            // headless drop for `--no-window`.
            let id = p.add(streamcraft_video::VideoCkSink::new_element());
            p.link((out_el, out_pad), (id, "sink")).expect("video ! videocksink (headless drop)");
            w.video_sink = Some(id);
        }
        // External (also an ExternalZeroCopy that fell back to readback here — `wire_video`
        // only ever sees the readback path, never zero-copy, which uses `wire_video_zerocopy`).
        SinkChoice::External | SinkChoice::ExternalZeroCopy => {
            // No sink added — record the tap point (overlay.src or dec.src) so an external
            // presenter (the SDL3 GUI's clock-paced frame-slot appsink) links its own sink.
            w.video_out = Some((out_el, out_pad));
        }
    }
}

/// Link an audio decoder and wire it to its sink, inserting `audioconvert`/`audioresample`
/// only if the direct decoder→sink link does not negotiate. Returns the decoder display
/// name on success. The decoder is `add`ed here; on a link failure it is left inert (never
/// run) and the error propagates to the caller, which drops the pad.
fn link_audio(
    p: &mut Pipeline,
    pad: (ElementId, &str),
    dec: Box<dyn Element>,
    sink: SinkChoice,
    w: &mut Wiring,
) -> Result<&'static str, String> {
    let name = audio_dec_name(dec.desc().name);
    // A *passive* decoder (ac3/eac3/flac/mp3 inline into their upstream's group for
    // efficiency on an elementary stream) would, on a demux **fan-out**, inline into the
    // demuxer's group and make the (passive) demuxer a non-tail branch point — which the
    // scheduler rejects (an inter-group edge must leave a group's tail). Head the branch with
    // a `queue` (Active): the passive decoder then inlines into the queue's group, and the
    // demuxer's edge to the queue is a legal tail branch — the same reason the subtitle branch
    // uses a queue. An Active decoder (aac) already heads its own group, so it links direct.
    let passive = dec.desc().sched == SchedHint::Passive;
    let dec_id = p.add_boxed(dec);
    if passive {
        let q = p.add(Queue::new());
        p.link(pad, (q, "sink")).map_err(|e| format!("demux ! queue: {e:?}"))?;
        p.link((q, "src"), (dec_id, "sink")).map_err(|e| format!("queue ! {name}: {e:?}"))?;
    } else {
        // Decoder sink ← demux pad. A family-filtered pad means this should succeed; a genuine
        // failure (a truncated codec head, say) surfaces here.
        p.link(pad, (dec_id, "sink")).map_err(|e| format!("demux ! {name}: {e:?}"))?;
    }

    // Audio pool/queue policy (spec: pool negotiation v1, the inverse of the video note):
    // a decoded frame is a few KB, and `try_alloc` hands out whole slots — from the shared
    // 4 MiB pool that is a 4 MiB slot pinned per 4 KB frame, so a couple dozen in-flight
    // audio buffers exhausted the pool: the demuxer starved (video froze) and the audio sink
    // underran. Worse, the DAC is the pipeline clock now — an underrun freezes video pacing
    // too. So: a small-slot 64 KiB × 64 pool, and a deep queue for interleave slack.
    p.set_element_pool(dec_id, AUDIO_SLOT, AUDIO_SLOTS);
    p.set_queue_capacity(dec_id, AUDIO_QUEUE);

    match sink {
        // Device — and External/ExternalZeroCopy, which are *video*-only choices: audio always
        // uses the device sink (it provides the pipeline clock). The GUI pairs an External video
        // choice with `audio: Device`; a stray `audio: External*` grabs the device.
        SinkChoice::Device | SinkChoice::External | SinkChoice::ExternalZeroCopy => {
            let s = p.add(sc_pipewire::PipeWireAudioSink::new());
            wire_audio_chain(p, dec_id, s, "pipewireaudiosink")?;
            w.audio_sink = Some(s);
        }
        SinkChoice::Drop => {
            let (ts, _stats) = TestSink::new();
            let s = p.add(ts);
            // A drop sink speaks `bytes` — the decoder's `audio/raw`→`bytes` bridge links
            // it directly, no convert/resample needed.
            p.link((dec_id, "src"), (s, "sink")).map_err(|e| format!("{name} ! dropsink: {e:?}"))?;
            w.audio_sink = Some(s);
        }
    }
    w.audio_dec = Some(dec_id);
    Ok(name)
}

/// Link `dec → device_sink`, inserting `audioconvert` then `audioconvert+audioresample`
/// when the direct link does not negotiate (spec: Formats — a decoder's `audio/raw` may not
/// match the device's sample format/rate; the convert/resample glue closes the gap).
///
/// The PipeWire sink takes s16 `audio/raw`; a decoder that emits s16 links directly (aac,
/// mp3), while a decoder that emits s24/s32 (flac) needs the convert to s16 first — and if
/// a device demanded a specific rate, the resampler too. We try the cheapest wiring first.
fn wire_audio_chain(
    p: &mut Pipeline,
    dec: ElementId,
    device_sink: ElementId,
    sink_name: &str,
) -> Result<(), String> {
    // Fold to stereo first: `audiodownmix` passes mono/stereo through unchanged and folds a
    // >2-channel stream (5.1 AC-3/E-AC-3, multichannel AAC) to a BS.775 stereo mix, so the
    // device — driven at stereo — always receives ≤2 channels. It is a runtime no-op for
    // stereo content (~free), so inserting it unconditionally is simpler and safe than
    // deciding channel count at wiring time (which isn't known until caps negotiate). Its
    // own small-slot pool keeps tiny audio buffers off the shared (video) pool.
    let dmix = p.add(AudioDownmix::new());
    p.link((dec, "src"), (dmix, "sink"))
        .map_err(|e| format!("{sink_name}: decoder ! audiodownmix: {e:?}"))?;
    p.set_element_pool(dmix, AUDIO_SLOT, AUDIO_SLOTS);
    let head = dmix; // the chain head feeding the sink is now the (post-fold) downmix output

    // 1) Direct: downmix → sink. Works when the (post-fold) sample format already matches the
    //    device's s16 (aac/mp3/ac3/eac3 emit s16, so this is the common path).
    if p.link((head, "src"), (device_sink, "sink")).is_ok() {
        return Ok(());
    }
    // 2) One convert to the device's s16: downmix → audioconvert(s16) → sink. The converter
    //    reads its input format off the negotiated `audio/raw` caps at runtime (e.g. flac s24).
    let conv = p.add(AudioConvert::new(SampleFormat::S16));
    if p.link((head, "src"), (conv, "sink")).is_ok()
        && p.link((conv, "src"), (device_sink, "sink")).is_ok()
    {
        return Ok(());
    }
    // 3) Convert + resample: downmix → audioconvert(s16) → audioresample(48k) → sink. 48 kHz
    //    is the device's near-universal default; the resampler discovers its input rate from
    //    the upstream caps. This is the last resort before reporting failure.
    let conv2 = p.add(AudioConvert::new(SampleFormat::S16));
    let resamp = p.add(AudioResample::new(48_000));
    if p.link((head, "src"), (conv2, "sink")).is_ok()
        && p.link((conv2, "src"), (resamp, "sink")).is_ok()
        && p.link((resamp, "src"), (device_sink, "sink")).is_ok()
    {
        return Ok(());
    }
    Err(format!("could not negotiate the audio chain into {sink_name} (tried direct, +convert, +convert+resample, all after the stereo downmix)"))
}

/// A stable display name for an audio decoder from its element desc name.
fn audio_dec_name(desc_name: &str) -> &'static str {
    match desc_name {
        "aacdec" => "aacdec",
        "flacdec" => "flacdec",
        "mp3dec" => "mp3dec",
        "ac3dec" => "ac3dec",
        "eac3dec" => "eac3dec",
        _ => "audiodec",
    }
}

/// Link a pad to a fresh drop sink (`TestSink`) so its buffers never accumulate (spec:
/// unlinked dynamic pads buffer without bound — an explicit drop is the honest v1).
fn drop_pad(p: &mut Pipeline, pad: (ElementId, &str)) {
    let (s, _stats) = TestSink::new();
    let id = p.add(s);
    // A drop sink speaks `bytes`; every demux pad offers the `bytes` escape, so this always
    // links. A failure here is a topology bug — surface it loudly.
    p.link(pad, (id, "sink")).unwrap_or_else(|e| panic!("drop-link {}: {e:?}", pad.1));
}

/// The sink label for a `track` line (`window`/`drop` for video, `audio device`/`drop`
/// for audio).
fn sink_label(choice: SinkChoice, video: bool) -> &'static str {
    match (choice, video) {
        (SinkChoice::Device, true) => "window",
        (SinkChoice::Device, false) => "audio device",
        (SinkChoice::Drop, _) => "drop sink",
        (SinkChoice::External | SinkChoice::ExternalZeroCopy, _) => "external sink",
    }
}

// ---------------------------------------------------------------------------------------
// Elementary streams (no container): filesrc → parser/decoder → audio chain.
// ---------------------------------------------------------------------------------------

/// Wire an elementary FLAC / MP3 stream: `filesrc → <decoder> → audio chain`. `dec` is the
/// already-constructed decoder; `pad_name` is the synthetic track name for the printout.
/// No demux stage — the decoder self-syncs on raw bytes (spec: `flacdec`/`mp3dec` accept a
/// `bytes` sink for exactly this `filesrc`-of-a-`.flac`/`.mp3` case).
pub fn wire_elementary_audio(
    p: &mut Pipeline,
    src: ElementId,
    dec: Box<dyn Element>,
    pad_name: &str,
    audio_sink: SinkChoice,
) -> Wiring {
    let mut w = Wiring::default();
    let name = audio_dec_name(dec.desc().name);
    let dec_id = p.add_boxed(dec);
    if let Err(e) = p.link((src, "src"), (dec_id, "sink")) {
        w.tracks.push(TrackOutcome {
            pad: pad_name.to_string(),
            summary: format!("(dropped: filesrc ! {name}: {e:?})"),
            linked: false,
        });
        return w;
    }
    p.set_element_pool(dec_id, AUDIO_SLOT, AUDIO_SLOTS);
    p.set_queue_capacity(dec_id, AUDIO_QUEUE);
    match audio_sink {
        // Device/External* all use the audio device (see the note in `link_audio`).
        SinkChoice::Device | SinkChoice::External | SinkChoice::ExternalZeroCopy => {
            let s = p.add(sc_pipewire::PipeWireAudioSink::new());
            match wire_audio_chain(p, dec_id, s, "pipewireaudiosink") {
                Ok(()) => {
                    w.audio_sink = Some(s);
                    w.audio_dec = Some(dec_id);
                    w.tracks.push(TrackOutcome {
                        pad: pad_name.to_string(),
                        summary: format!("{name} → audio device"),
                        linked: true,
                    });
                }
                Err(e) => w.tracks.push(TrackOutcome {
                    pad: pad_name.to_string(),
                    summary: format!("(dropped: {e})"),
                    linked: false,
                }),
            }
        }
        SinkChoice::Drop => {
            let (ts, _stats) = TestSink::new();
            let s = p.add(ts);
            p.link((dec_id, "src"), (s, "sink")).expect("audiodec ! dropsink");
            w.audio_sink = Some(s);
            w.audio_dec = Some(dec_id);
            w.tracks.push(TrackOutcome {
                pad: pad_name.to_string(),
                summary: format!("{name} → drop sink"),
                linked: true,
            });
        }
    }
    w
}
