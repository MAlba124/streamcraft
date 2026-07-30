# mp3dec test fixtures — how they were generated

These are TINY real MPEG-1 Audio Layer III streams, produced once with system
**libmp3lame** (via ffmpeg 8.1) during development and committed as goldens, together
with ffmpeg's own interleaved-`s16le` decode of each as the oracle reference. Each
`.mp3` is ~6.7 KB (0.35 s of a two-tone sine); the whole fixture set is ~106 KB.

The source signal is a fixed two-tone sine (440 Hz + 660 Hz on the left / mono channel,
523 Hz + 784 Hz on the right), deterministic so the fixtures are reproducible.

- **`mono_128.mp3`** — MPEG-1 Layer III, 44.1 kHz, single-channel, 128 kbps CBR. Bare
  frames (no ID3): starts with the `FF FB` syncword.
- **`js_128.mp3`** — MPEG-1 Layer III, 44.1 kHz, joint-stereo, 128 kbps CBR. Bare
  frames.
- **`mono_128_id3.mp3`** — the same mono content, but with a leading **ID3v2** tag
  (title + artist metadata) so the element's synchsafe ID3v2 skip is exercised. Starts
  with `49 44 33` (`"ID3"`).
- **`mono_128.ref.s16le` / `js_128.ref.s16le`** — ffmpeg's decode of the two bare `.mp3`
  fixtures to raw interleaved little-endian `i16` PCM (`-f s16le -ac {1,2}`). The oracle
  the SNR gate scores against.

The reference `.s16le` files are ffmpeg's **gapless-trimmed** decode: ffmpeg trims the
LAME encoder-delay priming and the frame-padded tail, so they are a few hundred samples
shorter than `mp3dec`'s untrimmed output. The test aligns the two by a best-shift
cross-correlation (encoder/decoder-delay compensation) before scoring SNR, and skips the
first frame's transient — so it measures decode fidelity, not the (out-of-v1) gapless
trim.

## Recipe

```sh
# 0.35 s two-tone sine sources.
ffmpeg -f lavfi -i "aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*660*t):s=44100:d=0.35" \
  -c:a pcm_s16le fx_mono.wav
ffmpeg -f lavfi -i "aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*660*t)|0.35*sin(2*PI*523*t)+0.2*sin(2*PI*784*t):s=44100:d=0.35" \
  -c:a pcm_s16le fx_stereo.wav

# Bare-frame CBR fixtures (no ID3).
ffmpeg -i fx_mono.wav   -c:a libmp3lame -b:a 128k               -id3v2_version 0 -write_id3v1 0 mono_128.mp3
ffmpeg -i fx_stereo.wav -c:a libmp3lame -b:a 128k -joint_stereo 1 -id3v2_version 0 -write_id3v1 0 js_128.mp3

# ID3v2-prefixed variant of the mono content.
ffmpeg -i fx_mono.wav -c:a libmp3lame -b:a 128k -write_id3v2 1 \
  -metadata title="profluens mp3dec fixture" -metadata artist="oxideav-mp3 gate" mono_128_id3.mp3

# Oracle decodes (interleaved s16le).
ffmpeg -i mono_128.mp3 -f s16le -c:a pcm_s16le -ac 1 mono_128.ref.s16le
ffmpeg -i js_128.mp3   -f s16le -c:a pcm_s16le -ac 2 js_128.ref.s16le
```
