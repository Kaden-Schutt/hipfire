#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-/home/sadara/.hipfire/models/qwen3.5-397b-a17b.mq6.hfq}"
RSS_LIMIT_KB="${HIPFIRE_EXPERT_MODULE_PROBE_RSS_LIMIT_KB:-2097152}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

cargo build -p hipfire-runtime --example qwen35_hfq_modules >/dev/null

tmp_json="$(mktemp)"
tmp_time="$(mktemp)"
trap 'rm -f "$tmp_json" "$tmp_time"' EXIT

/usr/bin/time -v \
  ./target/debug/examples/qwen35_hfq_modules probe "$MODEL" \
  >"$tmp_json" 2>"$tmp_time"

cat "$tmp_json"

max_rss_kb="$(awk -F: '/Maximum resident set size/ { gsub(/^[ \t]+/, "", $2); print $2 }' "$tmp_time" | tail -n1)"
if [[ -z "$max_rss_kb" ]]; then
  echo "could not read max RSS from /usr/bin/time output" >&2
  cat "$tmp_time" >&2
  exit 1
fi

payload_read_bytes="$(python3 - "$tmp_json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data.get("payload_read_bytes", -1))
PY
)"

skipped_full_payload="$(python3 - "$tmp_json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print("true" if data.get("full_payload_allocation_skipped") is True else "false")
PY
)"

if (( max_rss_kb > RSS_LIMIT_KB )); then
  echo "probe RSS ${max_rss_kb} KiB exceeded limit ${RSS_LIMIT_KB} KiB" >&2
  exit 1
fi

if [[ "$payload_read_bytes" != "0" || "$skipped_full_payload" != "true" ]]; then
  echo "probe did not prove metadata-only load: payload_read_bytes=$payload_read_bytes skipped=$skipped_full_payload" >&2
  exit 1
fi

echo "probe ok: max_rss_kb=$max_rss_kb limit_kb=$RSS_LIMIT_KB"
