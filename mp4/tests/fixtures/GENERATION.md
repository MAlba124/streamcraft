# MP4 test fixture provenance

Three tiny committed MP4 fixtures, generated once with the system ffmpeg so the
oracle + end-to-end tests have deterministic, few-hundred-KB-max inputs. All are
**progressive** (single `moov` + `mdat`, `moov` first via `+faststart`); the video is
128x96, 15 frames at 15 fps, yuv420p (the 4:2:0 8-bit subset `pf-h264` decodes).

## `tiny_h264.mp4` (12.7 KB)

Baseline profile, **no B-frames** ⇒ **no `ctts`** box, so composition time == decode time
(pts == dts). A closed GOP of 15 with `stss` marking the single sync sample. This is the
primary end-to-end decode fixture: every sample decodes through `pf-h264` (Annex B, 4:2:0
8-bit) with no reorder caveat.

```sh
ffmpeg -v error -y -f lavfi -i "testsrc2=size=128x96:rate=15:duration=1" \
  -c:v libx264 -profile:v baseline -pix_fmt yuv420p -g 15 -bf 0 \
  -x264-params "keyint=15:min-keyint=15:scenecut=0" \
  -movflags +faststart tiny_h264.mp4
```

## `bframes_h264.mp4` (12.5 KB)

Main profile, **2 B-frames** ⇒ **`ctts` present**, so this exercises the composition-offset
arithmetic (`pts = dts + ctts`) and decode-order ≠ presentation-order. Used by the oracle
test for the `stts`/`ctts` cross-check; not required to decode end-to-end (the reorder
caveat in `pf-h264` is documented, so the oracle asserts pts/dts, not decoded planes).

```sh
ffmpeg -v error -y -f lavfi -i "testsrc2=size=128x96:rate=15:duration=1" \
  -c:v libx264 -profile:v main -pix_fmt yuv420p -g 15 -bf 2 \
  -x264-params "keyint=15:min-keyint=15:scenecut=0" \
  -movflags +faststart bframes_h264.mp4
```

## `tiny_av.mp4` (21.4 KB)

**Two tracks** — the same baseline no-B-frame video plus an AAC-LC stereo sine tone
(48 kHz, 48 audio access units), so track 1 is `avc1` (with `avcC`) and track 2 is
`mp4a` with an `esds`-carried AudioSpecificConfig. This is the multi-track remux
fixture: `mp4demux(passthrough) ! mkvmuxn` must carry both tracks' bytes bit-exact
(video length-prefixed NALs + record, audio raw AAC AUs + ASC — RFC 9559 §12 shapes).

```sh
ffmpeg -v error -y -f lavfi -i "testsrc2=size=128x96:rate=15:duration=1" \
  -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=1" \
  -c:v libx264 -profile:v baseline -pix_fmt yuv420p -g 15 -bf 0 \
  -x264-params "keyint=15:min-keyint=15:scenecut=0" \
  -c:a aac -b:a 64k -ac 2 -shortest -movflags +faststart tiny_av.mp4
```

## System ffmpeg

```
ffmpeg version 8.1 (libx264)
```

The tests that consult a live ffmpeg as a second-opinion oracle **skip if ffmpeg/ffprobe
is absent** (they do not regenerate these committed fixtures — those are frozen).
