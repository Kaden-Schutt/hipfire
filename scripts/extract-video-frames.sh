#!/usr/bin/env bash
# Extract every coded frame of each video to lossless PNG, byte-identically to
# hipfire-media::decode_frames (same ffmpeg invocation) so the frames match what
# the gemma3-vl video path produces — i.e. they can be cached/compared 1:1.
#
# Usage:
#   scripts/extract-video-frames.sh <dir-or-file>...   [--out <root>]
# Default --out is "<video-dir>/frames/<video-stem>/f_00001.png".
set -euo pipefail

command -v ffmpeg  >/dev/null || { echo "extract-video-frames: ffmpeg not found" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "extract-video-frames: ffprobe not found" >&2; exit 1; }

VIDEO_EXTS="webm mp4 mkv mov avi m4v ogv wmv"
OUT_ROOT=""
INPUTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_ROOT="$2"; shift 2 ;;
    *) INPUTS+=("$1"); shift ;;
  esac
done
[ ${#INPUTS[@]} -gt 0 ] || { echo "usage: extract-video-frames.sh <dir-or-file>... [--out <root>]" >&2; exit 1; }

is_video() {
  local ext="${1##*.}"; ext="${ext,,}"
  for e in $VIDEO_EXTS; do [ "$ext" = "$e" ] && return 0; done
  return 1
}

# Mirror hipfire-media::scale_filter: expand limited→full luma range; full stays full.
scale_filter() {
  local r; r="$(ffprobe -v error -select_streams v:0 -show_entries stream=color_range \
                 -of default=nk=1:nw=1 "$1" 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
  case "$r" in
    pc|jpeg|full) echo "scale=iw:ih:in_range=full:out_range=full" ;;
    *)            echo "scale=iw:ih:in_range=tv:out_range=full" ;;
  esac
}

extract_one() {
  local video="$1" stem outdir
  stem="$(basename "$video")"; stem="${stem%.*}"
  if [ -n "$OUT_ROOT" ]; then outdir="$OUT_ROOT/$stem"; else outdir="$(dirname "$video")/frames/$stem"; fi
  mkdir -p "$outdir"
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$video" \
    -vsync 0 -vf "$(scale_filter "$video")" -pix_fmt rgb24 "$outdir/f_%05d.png"
  local count; count="$(find "$outdir" -maxdepth 1 -name 'f_*.png' | wc -l)"
  printf '  %-50s -> %4s frames  (%s)\n' "$stem" "$count" "$outdir"
}

total=0
for inp in "${INPUTS[@]}"; do
  if [ -d "$inp" ]; then
    while IFS= read -r -d '' v; do extract_one "$v"; total=$((total+1)); done \
      < <(find "$inp" -type f \( $(printf -- '-iname *.%s -o ' $VIDEO_EXTS | sed 's/ -o $//') \) -print0)
  elif [ -f "$inp" ] && is_video "$inp"; then
    extract_one "$inp"; total=$((total+1))
  else
    echo "  (skip non-video: $inp)" >&2
  fi
done
echo "done: $total video(s) extracted"
