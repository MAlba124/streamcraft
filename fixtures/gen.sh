#!/usr/bin/env bash
#
# gen.sh — generate the profluens validation corpus.
#
# Short (~5 s), tiny (320x180) test clips covering the container/codec matrix
# that pfplay is expected to handle, plus a few negative fixtures that MUST
# fail cleanly (no in-tree vorbis decoder; Mp4Demux needs a faststart layout).
#
# Idempotent: an existing, non-empty output is left alone. Set FORCE=1 to
# regenerate everything. A missing encoder in this ffmpeg build is reported
# and skipped — the script keeps going and still exits 0 (a partial corpus is
# useful; run-matrix.sh only tests the files that exist).
#
# Usage:
#   ./gen.sh            # generate what's missing, then sanity-check + summarise
#   FORCE=1 ./gen.sh    # rebuild every fixture from scratch
#   FFMPEG=/path ./gen.sh

set -uo pipefail  # NOT -e: a single failed encode must not abort the corpus.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"

FFMPEG="${FFMPEG:-/etc/profiles/per-user/merb/bin/ffmpeg}"
if ! command -v "$FFMPEG" >/dev/null 2>&1; then
    FFMPEG="$(command -v ffmpeg || true)"
fi
if [ -z "${FFMPEG:-}" ] || ! command -v "$FFMPEG" >/dev/null 2>&1; then
    echo "FATAL: ffmpeg not found (set FFMPEG=/path/to/ffmpeg)" >&2
    exit 1
fi
FFPROBE="${FFPROBE:-${FFMPEG%ffmpeg}ffprobe}"
command -v "$FFPROBE" >/dev/null 2>&1 || FFPROBE="$(command -v ffprobe || echo ffprobe)"

DUR=5
SIZE=320x180
FPS=25

# Common lavfi sources. testsrc2 gives motion (real inter-frame residual so the
# encoders actually do work); sine is a plain tone.
VSRC="testsrc2=size=${SIZE}:rate=${FPS}:duration=${DUR}"
ASRC="sine=frequency=440:sample_rate=48000:duration=${DUR}"

# --- encoder availability -------------------------------------------------
ENCODERS="$("$FFMPEG" -hide_banner -encoders 2>/dev/null)"
has_enc() { grep -qE "^\s*[A-Z.]+\s+$1\b" <<<"$ENCODERS"; }

# Pick the fastest available AV1 encoder (svt-av1 preferred, else libaom).
AV1_ENC=""
AV1_ARGS=()
if has_enc libsvtav1; then
    AV1_ENC="libsvtav1"
    # preset 12 = fastest; low-latency-ish knobs keep the encode brief.
    AV1_ARGS=(-c:v libsvtav1 -preset 12 -svtav1-params "fast-decode=1")
elif has_enc libaom-av1; then
    AV1_ENC="libaom-av1"
    AV1_ARGS=(-c:v libaom-av1 -cpu-used 8 -usage realtime)
fi

declare -a GEN_LOG=()   # "name|status|note" for the summary table
note() { printf '  %s\n' "$*"; }

# emit NAME BUILD_FN — runs BUILD_FN (which calls ffmpeg) unless the output
# already exists. BUILD_FN must return nonzero on encoder-missing so we can
# record SKIP vs OK vs FAIL distinctly.
emit() {
    local name="$1"; shift
    local path="$OUT/$name"
    if [ -z "${FORCE:-}" ] && [ -s "$path" ]; then
        GEN_LOG+=("$name|EXISTS|kept (FORCE=1 to rebuild)")
        note "exists: $name"
        return 0
    fi
    rm -f "$path"
    if "$@" "$path"; then
        if [ -s "$path" ]; then
            GEN_LOG+=("$name|OK|generated")
            note "wrote:  $name"
        else
            GEN_LOG+=("$name|FAIL|ffmpeg produced empty file")
            note "FAIL:   $name (empty output)"
            rm -f "$path"
        fi
    else
        local rc=$?
        if [ "$rc" -eq 42 ]; then
            GEN_LOG+=("$name|SKIP|encoder missing in this ffmpeg build")
            note "skip:   $name (encoder missing)"
        else
            GEN_LOG+=("$name|FAIL|ffmpeg exit $rc")
            note "FAIL:   $name (ffmpeg exit $rc)"
        fi
        rm -f "$path"
    fi
}

run() { "$FFMPEG" -hide_banner -loglevel error -y "$@"; }

# ---------------------------------------------------------------------------
# Build functions. Each takes the output path as its last argument.
# Return 42 to signal "required encoder missing" (→ graceful SKIP).
# ---------------------------------------------------------------------------

b_mkv_h264_aac() {
    has_enc libx264 || return 42; has_enc aac || return 42
    run -f lavfi -i "$VSRC" -f lavfi -i "$ASRC" \
        -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
        -c:a aac -b:a 96k -shortest -f matroska "$1"
}

b_mkv_h265_aac() {
    has_enc libx265 || return 42; has_enc aac || return 42
    run -f lavfi -i "$VSRC" -f lavfi -i "$ASRC" \
        -c:v libx265 -preset ultrafast -pix_fmt yuv420p -tag:v hvc1 \
        -c:a aac -b:a 96k -shortest -f matroska "$1"
}

b_webm_vp8_vorbis() {
    has_enc libvpx || return 42; has_enc libvorbis || return 42
    run -f lavfi -i "$VSRC" -f lavfi -i "$ASRC" \
        -c:v libvpx -deadline realtime -cpu-used 8 -b:v 400k -pix_fmt yuv420p \
        -c:a libvorbis -b:a 96k -shortest -f webm "$1"
}

b_webm_vp9() {
    has_enc libvpx-vp9 || return 42
    run -f lavfi -i "$VSRC" \
        -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -b:v 400k -pix_fmt yuv420p \
        -f webm "$1"
}

b_mkv_av1() {
    [ -n "$AV1_ENC" ] || return 42
    run -f lavfi -i "$VSRC" "${AV1_ARGS[@]}" -pix_fmt yuv420p -f matroska "$1"
}

b_mp4_h264_aac() {
    has_enc libx264 || return 42; has_enc aac || return 42
    run -f lavfi -i "$VSRC" -f lavfi -i "$ASRC" \
        -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
        -c:a aac -b:a 96k -shortest -movflags +faststart -f mp4 "$1"
}

b_mp4_h265() {
    has_enc libx265 || return 42
    run -f lavfi -i "$VSRC" \
        -c:v libx265 -preset ultrafast -pix_fmt yuv420p -tag:v hvc1 \
        -movflags +faststart -f mp4 "$1"
}

b_mp4_h264_nofaststart() {
    # Negative fixture: moov AFTER mdat. Mp4Demux resolves the head bytes only
    # (progressive layout) and must reject this with a clear error.
    has_enc libx264 || return 42; has_enc aac || return 42
    run -f lavfi -i "$VSRC" -f lavfi -i "$ASRC" \
        -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
        -c:a aac -b:a 96k -shortest -movflags -faststart -f mp4 "$1"
}

b_ogg_flac() {
    has_enc flac || return 42
    run -f lavfi -i "$ASRC" -c:a flac -f oga "$1"
}

b_ogg_vorbis() {
    # Negative fixture: vorbis-in-ogg, no other track. profluens has no
    # vorbis decoder → nothing links → clean nonzero exit.
    has_enc libvorbis || return 42
    run -f lavfi -i "$ASRC" -c:a libvorbis -b:a 96k -f ogg "$1"
}

b_audio_mp3() {
    has_enc libmp3lame || return 42
    # 44.1 kHz stereo CBR.
    run -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=${DUR}" \
        -af "aeval=val(0)|val(0):c=stereo" \
        -c:a libmp3lame -b:a 128k -ar 44100 -ac 2 "$1"
}

b_audio_flac() {
    has_enc flac || return 42
    run -f lavfi -i "$ASRC" -c:a flac "$1"
}

b_audio_wav() {
    has_enc pcm_s16le || return 42
    run -f lavfi -i "$ASRC" -c:a pcm_s16le -f wav "$1"
}

b_audio_aac() {
    has_enc aac || return 42
    # Raw ADTS AAC (self-framing, no container).
    run -f lavfi -i "$ASRC" -c:a aac -b:a 128k -f adts "$1"
}

echo "== generating fixtures into $OUT =="
echo "   ffmpeg: $FFMPEG"
echo "   av1 encoder: ${AV1_ENC:-<none available>}"
echo

emit mkv_h264_aac.mkv          b_mkv_h264_aac
emit mkv_h265_aac.mkv          b_mkv_h265_aac
emit webm_vp8_vorbis.webm      b_webm_vp8_vorbis
emit webm_vp9.webm             b_webm_vp9
emit mkv_av1.mkv               b_mkv_av1
emit mp4_h264_aac.mp4          b_mp4_h264_aac
emit mp4_h265.mp4              b_mp4_h265
emit mp4_h264_nofaststart.mp4  b_mp4_h264_nofaststart
emit ogg_flac.oga              b_ogg_flac
emit ogg_vorbis.ogg            b_ogg_vorbis
emit audio.mp3                 b_audio_mp3
emit audio.flac                b_audio_flac
emit audio.wav                 b_audio_wav
emit audio.aac                 b_audio_aac

# ---------------------------------------------------------------------------
# Sanity check: read back each file with ffprobe and print stream shorthand.
# ---------------------------------------------------------------------------
echo
echo "== ffprobe sanity check =="
probe_streams() {
    # -> "video:h264 audio:aac" style summary, or "UNREADABLE".
    # ffprobe emits stream= entries in its own schema order (codec_name before
    # codec_type here), so read them in that order and print type:name.
    local f="$1" out=""
    local codec_name codec_type
    while IFS='|' read -r codec_name codec_type; do
        [ -n "$codec_type" ] || continue
        out+="${codec_type}:${codec_name} "
    done < <("$FFPROBE" -v error -show_entries stream=codec_name,codec_type \
                        -of csv=p=0 "$f" 2>/dev/null | tr ',' '|')
    [ -n "$out" ] && printf '%s' "${out% }" || printf 'UNREADABLE'
}

printf '%-26s %-10s %-8s  %s\n' "FILE" "STATUS" "SIZE" "STREAMS (ffprobe)"
printf '%-26s %-10s %-8s  %s\n' "----" "------" "----" "-----------------"
for entry in "${GEN_LOG[@]}"; do
    name="${entry%%|*}"; rest="${entry#*|}"; status="${rest%%|*}"
    path="$OUT/$name"
    if [ -s "$path" ]; then
        sz=$(du -h "$path" | cut -f1)
        streams="$(probe_streams "$path")"
        printf '%-26s %-10s %-8s  %s\n' "$name" "$status" "$sz" "$streams"
    else
        printf '%-26s %-10s %-8s  %s\n' "$name" "$status" "-" "(not produced)"
    fi
done

echo
n_ok=$(printf '%s\n' "${GEN_LOG[@]}" | grep -cE '\|(OK|EXISTS)\|' || true)
n_skip=$(printf '%s\n' "${GEN_LOG[@]}" | grep -c '|SKIP|' || true)
n_fail=$(printf '%s\n' "${GEN_LOG[@]}" | grep -c '|FAIL|' || true)
echo "done: $n_ok present, $n_skip skipped (encoder missing), $n_fail failed"
# A partial corpus is fine; only a hard ffmpeg FAIL is worth a nonzero exit.
[ "$n_fail" -eq 0 ]
