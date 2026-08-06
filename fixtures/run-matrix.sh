#!/usr/bin/env bash
#
# run-matrix.sh — drive every fixture through pfplay and tabulate PASS/FAIL.
#
# Player CLI contract (frozen — pfplay is being built to this spec):
#   pfplay [--no-window] [--no-audio] [--stats] [--max-secs N] FILE
#     * prints "track <padname>: ..." lines to stdout
#     * exit 0 on clean EOS or a --max-secs stop
#     * nonzero, with ONE stderr line, when the container is unknown or no
#       track links to any decoder
#
# Each expected-PASS fixture is run headless/mute with a max-secs cap and PASSES
# on exit 0. Each expected-FAIL fixture (nofaststart mp4, vorbis-only ogg)
# PASSES when the exit code is nonzero AND the run did not hang. Every run is
# wrapped in `timeout` so a hang surfaces as FAIL instead of wedging the matrix.
#
# Usage:
#   ./run-matrix.sh /path/to/pfplay          # hardware path (as built)
#   ./run-matrix.sh --sw /path/to/pfplay     # software path (PF_NO_VAAPI=1)
#
# Exit status: 0 iff every row passed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

SW=0
if [ "${1:-}" = "--sw" ]; then
    SW=1
    shift
fi

SCPLAY="${1:-}"
if [ -z "$SCPLAY" ]; then
    echo "usage: $0 [--sw] <path-to-pfplay-binary>" >&2
    exit 2
fi
if [ ! -x "$SCPLAY" ]; then
    # Allow a bare name resolved via PATH, but insist it exists.
    if ! command -v "$SCPLAY" >/dev/null 2>&1; then
        echo "error: pfplay binary not found or not executable: $SCPLAY" >&2
        exit 2
    fi
fi
if [ ! -d "$OUT" ]; then
    echo "error: no fixtures in $OUT — run ./gen.sh first" >&2
    exit 2
fi

MAX_SECS=10          # player self-stops here
HARD_TIMEOUT=30      # wall clock per invocation; a hang trips this → FAIL

# Fixture matrix: "name expectation". expectation ∈ {pass, fail}.
# Order matches gen.sh. Missing files (skipped encoder) are reported SKIP.
MATRIX=(
    "mkv_h264_aac.mkv          pass"
    "mkv_h265_aac.mkv          pass"
    "webm_vp8_vorbis.webm      pass"   # video plays; vorbis audio drops cleanly
    "webm_vp9.webm             pass"
    "mkv_av1.mkv               pass"
    "mp4_h264_aac.mp4          pass"
    "mp4_h265.mp4              pass"
    "mp4_h264_nofaststart.mp4  fail"   # moov after mdat — Mp4Demux must reject
    "ogg_flac.oga              pass"
    "ogg_vorbis.ogg            fail"   # no vorbis decoder, no other track
    "audio.mp3                 pass"
    "audio.flac                pass"
    "audio.wav                 pass"
    "audio.aac                 pass"
)

if [ "$SW" -eq 1 ]; then
    echo "== pfplay matrix (software path: PF_NO_VAAPI=1) =="
else
    echo "== pfplay matrix (default path) =="
fi
echo "   pfplay: $SCPLAY"
echo

hdr_fmt='%-26s %-6s %-5s %-6s  %s\n'
printf "$hdr_fmt" "FIXTURE" "EXPECT" "EXIT" "RESULT" "FIRST TRACK / STDERR LINE"
printf "$hdr_fmt" "-------" "------" "----" "------" "-------------------------"

n_pass=0 n_fail=0 n_skip=0

# run_one <file> -> sets globals RC, DETAIL
run_one() {
    local file="$1"
    local out err rc
    out="$(mktemp)"; err="$(mktemp)"
    local -a env=()
    [ "$SW" -eq 1 ] && env=(env PF_NO_VAAPI=1)

    # `timeout` returns 124 on TERM-kill (hang). Bound every run.
    "${env[@]}" timeout "$HARD_TIMEOUT" \
        "$SCPLAY" --no-window --no-audio --max-secs "$MAX_SECS" "$file" \
        >"$out" 2>"$err"
    rc=$?

    # Detail: prefer the first "track ..." stdout line; else the first stderr
    # line (the one-line error for the fail rows / real failures).
    local detail
    detail="$(grep -m1 -iE '^[[:space:]]*track' "$out" 2>/dev/null)"
    if [ -z "$detail" ]; then
        detail="$(grep -m1 . "$out" 2>/dev/null)"   # any stdout line
    fi
    if [ -z "$detail" ]; then
        detail="$(grep -m1 . "$err" 2>/dev/null)"   # first stderr line
    fi
    # Note a hang explicitly.
    [ "$rc" -eq 124 ] && detail="TIMEOUT after ${HARD_TIMEOUT}s (hang) — ${detail}"

    RC=$rc
    DETAIL="$detail"
    rm -f "$out" "$err"
}

for row in "${MATRIX[@]}"; do
    read -r name expect <<<"$row"
    file="$OUT/$name"

    if [ ! -s "$file" ]; then
        printf "$hdr_fmt" "$name" "$expect" "-" "SKIP" "(fixture absent — run gen.sh)"
        n_skip=$((n_skip + 1))
        continue
    fi

    run_one "$file"

    # Decide pass/fail against the expectation. A hang (124) never passes.
    local_result="FAIL"
    if [ "$RC" -eq 124 ]; then
        local_result="FAIL"
    elif [ "$expect" = "pass" ] && [ "$RC" -eq 0 ]; then
        local_result="PASS"
    elif [ "$expect" = "fail" ] && [ "$RC" -ne 0 ]; then
        local_result="PASS"
    fi

    # Trim detail to keep the table aligned.
    detail_trim="${DETAIL:0:64}"
    printf "$hdr_fmt" "$name" "$expect" "$RC" "$local_result" "$detail_trim"

    if [ "$local_result" = "PASS" ]; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done

echo
echo "summary: $n_pass passed, $n_fail failed, $n_skip skipped"
[ "$n_fail" -eq 0 ]
