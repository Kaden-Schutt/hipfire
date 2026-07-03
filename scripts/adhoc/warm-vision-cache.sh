#!/usr/bin/env bash
# Pre-warm the gemma3-vl vision-embedding cache for every image under the given
# paths, for a specific model. The cache is per-model (keyed by model path +
# vision config), so pass the exact model you'll serve. Uses the daemon's
# `vision_cache_only` mode: SigLIP encode + cache insert, NO LM decode — cheap
# even on a large model.
#
# Usage:
#   scripts/warm-vision-cache.sh <model.hfq|name> <dir-or-image>...
set -euo pipefail

MODEL="${1:?usage: warm-vision-cache.sh <model> <dir-or-image>...}"; shift
[ $# -gt 0 ] || { echo "warm-vision-cache: no input paths" >&2; exit 1; }

DAEMON="${HIPFIRE_DAEMON_BIN:-./target/release/hipfire-daemon}"
[ -x "$DAEMON" ] || { echo "warm-vision-cache: daemon not found at $DAEMON" >&2; exit 1; }

# Collect images (recursively for dirs).
mapfile -d '' IMGS < <(find "$@" -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.jfif' \
     -o -iname '*.avif' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.gif' \) -print0)
echo "warm-vision-cache: ${#IMGS[@]} image(s) -> model $MODEL"
[ "${#IMGS[@]}" -gt 0 ] || { echo "  (no images found)"; exit 0; }

# Build a JSONL session (python for safe escaping of paths with spaces) and pipe
# it to the daemon. HIPFIRE_VISION_CACHE must stay enabled (the default).
python3 - "$MODEL" "${IMGS[@]}" <<'PY' | "$DAEMON" 2>/dev/null \
  | grep -E '"type":"(done|error)"' \
  | python3 -c '
import sys, json
ok=err=hit=0
for line in sys.stdin:
    try: e=json.loads(line)
    except Exception: continue
    if e.get("type")=="done":
        ok+=1; hit+=int(e.get("cache_hits",0) or 0)
    elif e.get("type")=="error":
        err+=1; print("  error:", e.get("message","")[:120], file=sys.stderr)
print(f"warm-vision-cache: encoded {ok} ok, {err} error(s), {hit} were already cached")
'
import sys, json
model = sys.argv[1]
imgs = sys.argv[2:]
print(json.dumps({"type": "load", "model": model,
                  "params": {"max_seq": 512, "kv_cache": "q8"}, "request_id": "load"}))
for i, p in enumerate(imgs):
    print(json.dumps({"type": "generate", "id": f"g{i}", "prompt": "x",
                      "image": p, "vision_cache_only": True, "max_tokens": 0,
                      "request_id": f"g{i}"}))
print(json.dumps({"type": "unload", "request_id": "unload"}))
PY
