# fixtures — profluens validation corpus

A small, fast-to-generate set of test-media clips plus a matrix runner, used to
smoke-test the general-purpose player (`pfplay`) across the container/codec
combinations profluens is expected to handle — including a few **negative**
fixtures that must fail cleanly.

Nothing here is checked in except the scripts: the media lives in `out/`, which
is git-ignored and regenerated on demand.

## Layout

| file             | what                                                        |
|------------------|-------------------------------------------------------------|
| `gen.sh`         | generate the corpus into `out/` with ffmpeg (idempotent)    |
| `run-matrix.sh`  | run every fixture through `pfplay`, tabulate PASS/FAIL      |
| `out/`           | generated media (git-ignored)                               |

## Regenerating

```sh
./gen.sh            # generate whatever is missing, then ffprobe-sanity-check
FORCE=1 ./gen.sh    # rebuild every fixture from scratch
FFMPEG=/path/to/ffmpeg ./gen.sh
```

All clips are ~5 s, 320x180, encoded with the fastest presets (`-preset
ultrafast`; `-deadline realtime -cpu-used 8` for libvpx; SVT-AV1 `-preset 12`
or libaom `-cpu-used 8 -usage realtime`). Video is lavfi `testsrc2`, audio is
lavfi `sine`. If this ffmpeg build lacks a required encoder, that row is
reported and **skipped** — `gen.sh` keeps going and produces a partial corpus.
After generating it re-reads every file with `ffprobe` and prints a summary
table (streams + size), so a corrupt/empty write is caught immediately.

### The matrix

| fixture                     | intent                                             |
|-----------------------------|----------------------------------------------------|
| `mkv_h264_aac.mkv`          | H.264 + AAC in Matroska                             |
| `mkv_h265_aac.mkv`          | H.265 + AAC in Matroska                             |
| `webm_vp8_vorbis.webm`      | VP8 video **plays**, Vorbis audio **drops cleanly** |
| `webm_vp9.webm`             | VP9, video-only                                    |
| `mkv_av1.mkv`               | AV1, video-only (if an AV1 encoder exists)         |
| `mp4_h264_aac.mp4`          | H.264 + AAC, **+faststart** (moov before mdat)     |
| `mp4_h265.mp4`              | H.265, video-only, +faststart                      |
| `mp4_h264_nofaststart.mp4`  | **negative** — moov *after* mdat                   |
| `ogg_flac.oga`              | FLAC-in-Ogg                                         |
| `ogg_vorbis.ogg`            | **negative** — Vorbis-only Ogg                     |
| `audio.mp3`                 | MP3, 44.1 kHz stereo CBR                            |
| `audio.flac`                | FLAC (native)                                       |
| `audio.wav`                 | WAV, s16le PCM                                      |
| `audio.aac`                 | raw ADTS AAC (no container)                         |

## Running the matrix

```sh
./run-matrix.sh /path/to/pfplay          # default (hardware) decode path
./run-matrix.sh --sw /path/to/pfplay     # software path (PF_NO_VAAPI=1)
```

The player CLI contract (frozen — `pfplay` is built to this spec):

```
pfplay [--no-window] [--no-audio] [--stats] [--max-secs N] FILE
```

* prints `track <padname>: ...` lines to stdout;
* exits 0 on clean EOS or a `--max-secs` stop;
* exits nonzero with **one** stderr line when the container is unknown or no
  track links to any decoder.

For each expected-**pass** fixture the runner invokes
`pfplay --no-window --no-audio --max-secs 10 <file>` and counts exit 0 as PASS.
Expected-**fail** fixtures PASS when the exit code is nonzero *and* the process
did not hang. Every invocation is wrapped in `timeout 30`, so a hang surfaces as
a FAIL (exit 124) rather than wedging the whole matrix. `--sw` prepends
`PF_NO_VAAPI=1` to every run to exercise the pure-software decoders.

The runner prints one aligned row per fixture (name, expectation, exit code,
PASS/FAIL, first `track` line or first stderr line), a summary count, and exits
nonzero if any row failed. A fixture that wasn't generated (missing encoder)
shows as `SKIP` and does not count against the run.

## Why the negative fixtures fail (by design)

* **`ogg_vorbis.ogg`** and the Vorbis audio track in **`webm_vp8_vorbis.webm`** —
  profluens has **no in-tree Vorbis decoder** (nor Opus, nor Theora). In the
  WebM the VP8 video still plays and the Vorbis track drops into a
  drop-sink, so the run PASSES. The Vorbis-only Ogg has nothing left to play, so
  no track links to a decoder and the player exits nonzero — that nonzero exit is
  the PASS condition for that row.
* **`mp4_h264_nofaststart.mp4`** — `Mp4Demux` resolves **progressive / faststart**
  MP4 only: it needs the file *head* bytes through `moov` to build the sample
  tables before it can address samples in `mdat`. When `moov` sits *after*
  `mdat` (no faststart) it cannot resolve the tables and rejects the file with a
  clear error. The positive `mp4_*` fixtures are all written with
  `-movflags +faststart` for exactly this reason.
