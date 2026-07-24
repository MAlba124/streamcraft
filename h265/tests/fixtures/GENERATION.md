# h265 test fixtures — how they were generated

These are TINY real HEVC Annex B elementary streams, produced once with system
**x265** (via ffmpeg 8.1, libx265) during development and committed as goldens. They
are stripped to VPS/SPS/PPS + VCL NAL units (the ~2.3 KB x265 user-data SEI prefix is
removed — the decoder ignores SEI, and ffmpeg decodes the stripped stream identically),
so each is only a few hundred bytes.

The `tests/h265dec.rs` reference-decode test does not need a committed expected-YUV
file: it decodes each golden through the `oxideav-h265` library directly as its own
reference (byte-exact against the pipeline output), and separately cross-checks against
ffmpeg's decoder at test time (skipping when ffmpeg is absent). These `.hevc` bytes are
the only committed artifacts.

## Recipe

```sh
# tiny_i.hevc — a single 16x16 Main-profile IDR (intra).
ffmpeg -hide_banner -loglevel error -f lavfi -i "testsrc2=size=16x16:rate=1:duration=1" \
  -pix_fmt yuv420p -c:v libx265 \
  -x265-params "keyint=1:min-keyint=1:no-open-gop=1:no-sao=1:log-level=none" \
  -f hevc -vframes 1 raw.hevc

# ip.hevc — a 16x16 IDR followed by a TRAIL_R P-picture (inter).
ffmpeg -hide_banner -loglevel error -f lavfi -i "testsrc2=size=16x16:rate=2:duration=1" \
  -pix_fmt yuv420p -c:v libx265 \
  -x265-params "keyint=2:min-keyint=2:bframes=0:no-open-gop=1:no-sao=1:log-level=none" \
  -f hevc -vframes 2 raw.hevc
```

Then strip the SEI-prefix NAL (types 39/40) — scan for start codes (`00 00 01` /
`00 00 00 01`), read `nal_unit_type = (byte_after_start_code >> 1) & 0x3f`, and drop
units 39/40 — keeping VPS(32)/SPS(33)/PPS(34) and the VCL NALs. The stripped stream
decodes identically under ffmpeg (`cmp` the two raw-YUV decodes to confirm).
