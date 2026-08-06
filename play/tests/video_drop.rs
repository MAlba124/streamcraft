//! The audio-only build's load-bearing claim: **a video-bearing container still opens, and its
//! audio still plays.**
//!
//! `pf-play`'s `video` feature (default on) carries every video decoder. A music player turns it
//! off — and the failure mode that would matter is not "no picture", it is a video-bearing MKV
//! in the library refusing to open, or opening with a video pad that nothing consumes and that
//! therefore buffers without bound behind the audio. So this file muxes a real two-track MKV
//! (a `V_VP8` video track beside an `A_MPEG/L3` audio track) and drives it through the whole
//! controller, `Player::open` included.
//!
//! The same test runs in both feature configurations and asserts the difference explicitly:
//!
//! - with `video`: the video pad links a decoder (`vp8` → `vp8dec`);
//! - without `video`: the video pad is reported `(dropped: no decoder for vp8)` — the same
//!   shape an unknown codec has always produced — and the pipeline is then run to EOS with the
//!   audio decoder's tap counters checked, proving the drop sink really consumes the video
//!   track rather than stalling the audio behind it.
//!
//! Only the audio-only config runs the pipeline: the video payload here is a stand-in, not a
//! decodable VP8 bitstream, so a build *with* `vp8dec` would be asserting on the decoder's
//! reaction to garbage rather than on the wiring. The wiring is what this file is about.

// Tests own their fixtures: temp files, staging `Vec`s, format! in assertions.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;

use pf_play::{Player, SinkChoice, SinkPolicy};

const RATE: u32 = 44_100;
/// MPEG-1 Layer III: 1152 PCM frames per audio frame.
const SAMPLES_PER_FRAME: u64 = 1152;
/// 128 kbps @ 44.1 kHz, no padding: `floor(144 * 128000 / 44100)`.
const MP3_FRAME_LEN: usize = 417;
const MP3_FRAMES: usize = 20;

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("video_drop");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// One silent MPEG-1 Layer III frame: a 4-byte header followed by zeroed side info and main
/// data (ISO/IEC 11172-3 §2.4.1.7 / §2.4.2.7).
///
/// Zeroed side info means `part2_3_length == 0` for both granules of both channels, i.e. a
/// frame that legally codes silence — the exact shape of the padding frames an encoder emits
/// around a stream. That is all this fixture needs: the point is that the audio branch decodes
/// and the buffers flow, not what they sound like. Synthesising it here keeps the test free of
/// an ffmpeg dependency (the workspace's other MP3 fixtures skip when the tool is absent).
///
/// Header bits: `FF FB` = sync + MPEG-1 + Layer III + no CRC; `0x90` = bitrate index 9
/// (128 kbps) + sampling index 0 (44.1 kHz) + no padding; `0x04` = stereo, original.
fn silent_mp3_frame() -> Vec<u8> {
    let mut f = vec![0u8; MP3_FRAME_LEN];
    f[0] = 0xFF;
    f[1] = 0xFB;
    f[2] = 0x90;
    f[3] = 0x04;
    f
}

/// Mux `track 1 = V_VP8` (video) + `track 2 = A_MPEG/L3` (audio) into an MKV on disk,
/// interleaved on a shared nanosecond timeline, and return its path.
///
/// The video Blocks carry a stand-in payload: what is under test is the *controller's* handling
/// of a declared video track, and the demuxer's family announcement (`V_VP8` → `vp8`) comes
/// from the Tracks header, not from the bitstream.
fn two_track_mkv(name: &str) -> PathBuf {
    use pf_mkv::{MatroskaWriter, TrackConfig};

    let video = TrackConfig::video(1, "V_VP8", Vec::new(), 320, 240);
    let audio = TrackConfig::audio(2, "A_MPEG/L3", Vec::new(), RATE as f64, 2, 16);
    let mut w = MatroskaWriter::new(vec![video, audio]);

    let frame = silent_mp3_frame();
    let mut out = Vec::new();
    w.write_header(&mut out).expect("mkv header");
    let block_ns = SAMPLES_PER_FRAME * 1_000_000_000 / RATE as u64;
    for i in 0..MP3_FRAMES {
        let ts = i as u64 * block_ns;
        // One video Block per audio Block, video first, so the demuxer really fans out to two
        // pads with the video pad ahead of the audio one.
        w.write_frame(&mut out, 1, ts, &[0x10, 0x00, 0x9D, 0x01, 0x2A], true).expect("video block");
        w.write_frame(&mut out, 2, ts, &frame, true).expect("audio block");
    }
    w.finalize(&mut out);

    let path = tmp_dir().join(name);
    std::fs::write(&path, &out).expect("write mkv fixture");
    path
}

/// Find the outcome for a demux pad by the MKV demuxer's `src_track<N>` naming.
fn outcome(player: &Player, track: u64) -> &pf_play::autoplug::TrackOutcome {
    let want = format!("src_track{track}");
    player
        .tracks()
        .iter()
        .find(|t| t.pad == want)
        .unwrap_or_else(|| panic!("no outcome for {want}: {:?}", summaries(player)))
}

fn summaries(player: &Player) -> Vec<String> {
    player.tracks().iter().map(|t| format!("{}: {}", t.pad, t.summary)).collect()
}

#[test]
fn video_bearing_mkv_plays_its_audio_with_the_video_dropped() {
    let path = two_track_mkv("vp8_plus_mp3.mkv");
    let policy = SinkPolicy { video: SinkChoice::Drop, audio: SinkChoice::Drop };
    let player = Player::open(path.to_str().unwrap(), policy)
        .expect("a video-bearing MKV must open in every build");

    // Both tracks are discovered in both builds — the demuxers are never feature-gated.
    assert_eq!(player.tracks().len(), 2, "two tracks discovered: {:?}", summaries(&player));

    // The audio is what a music player came for, and it links either way.
    let audio = outcome(&player, 2);
    assert!(audio.linked, "the MP3 track must link: {:?}", summaries(&player));
    assert!(
        audio.summary.starts_with("mp3 → mp3dec"),
        "the MP3 track goes through mp3dec: {}",
        audio.summary
    );

    let video = outcome(&player, 1);

    #[cfg(feature = "video")]
    {
        assert!(video.linked, "with `video`, the vp8 track links a decoder: {}", video.summary);
        assert!(
            video.summary.starts_with("vp8 → vp8dec"),
            "with `video`, vp8 selects vp8dec: {}",
            video.summary
        );
    }

    #[cfg(not(feature = "video"))]
    {
        assert!(!video.linked, "without `video` there is no decoder to link: {}", video.summary);
        assert_eq!(
            video.summary, "(dropped: no decoder for vp8)",
            "an audio-only build reports the video track with the standard no-decoder shape"
        );

        // The drop must be a real consumer, not an unlinked pad: an unconsumed demux pad would
        // buffer without bound and stall the audio behind it. So run to EOS and then read the
        // audio decoder's own counters off the live tap — reaching EOS *and* having decoded
        // every muxed frame is the proof that the video track went somewhere harmless.
        let adec = player
            .watched()
            .into_iter()
            .find(|(name, _)| *name == "adec")
            .expect("a linked audio decoder is watched")
            .1;
        let tap = player.pipeline.tap_handle();
        let mut player = player;
        player.run().expect("play the audio to EOS with the video track dropped");
        assert!(player.any_track_linked());

        let counters = tap.snapshot(adec).expect("the audio decoder reports counters");
        assert_eq!(
            counters.buffers_in, MP3_FRAMES as u64,
            "every muxed MP3 frame reached the decoder past the video fan-out"
        );
        assert!(counters.buffers_out > 0, "the audio decoder produced PCM");
    }
}
