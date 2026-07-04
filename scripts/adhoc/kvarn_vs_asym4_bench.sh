#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# asym4 vs kvarn side-by-side: PPL + KLD (vs f32-KV ref) at short/long ctx, and
# single-stream decode tok/s at short vs long KV. See the header note on batched
# decode: quantized KV decodes via the serial per-session loop on dense models
# (the fused/routed path is f32-KV-gated), so aggregate batched tok/s is flat for
# both asym4 and kvarn today — true batching needs the quantized-routed dense path.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/rocm/lib ROCM_PATH=/opt/rocm
MODEL=${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b-oq4awq.hfq}
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
PPLBIN=./target/release/examples/perplexity
DAEMON=./target/release/hipfire-daemon
OUT=${OUT:-/tmp/kvarn_vs_asym4.md}

echo "# asym4 vs kvarn — $(basename "$MODEL")" >"$OUT"
echo "" >>"$OUT"

# ── PPL + KLD matrix ───────────────────────────────────────────────────────
echo "## PPL + KLD (vs oq4awq+f32-KV reference)" >>"$OUT"
echo "" >>"$OUT"
printf '| ctx | mode | PPL | KLD/tok |\n|---|---|---|---|\n' >>"$OUT"
for CTX in 512 2048; do
    REF=/tmp/kref_${CTX}.bin
    $PPLBIN "$MODEL" "$CORPUS" --ctx "$CTX" --kv-mode f32 --dump-ref "$REF" >/tmp/p.log 2>&1
    refppl=$(grep -oE 'PPL: *[0-9.]+' /tmp/p.log | grep -oE '[0-9.]+' | head -1)
    printf '| %s | f32 (ref) | %s | — |\n' "$CTX" "$refppl" >>"$OUT"
    for KV in asym4 kvarn; do
        $PPLBIN "$MODEL" "$CORPUS" --ctx "$CTX" --kv-mode "$KV" --kld-ref "$REF" >/tmp/p.log 2>&1
        ppl=$(grep -oE 'PPL: *[0-9.]+' /tmp/p.log | grep -oE '[0-9.]+' | head -1)
        kld=$(grep -oE 'KLD/tok: *[0-9.]+' /tmp/p.log | grep -oE '[0-9.]+' | head -1)
        printf '| %s | %s | %s | %s |\n' "$CTX" "$KV" "$ppl" "$kld" >>"$OUT"
    done
done
echo "" >>"$OUT"

# ── Single-stream decode tok/s at short vs long KV ─────────────────────────
# Long prompt = first ~6000 chars of the corpus (~1500 tok) → decode attends to long KV.
LONGP=$(head -c 6000 "$CORPUS" | tr '\n' ' ' | sed 's/"/ /g')
echo "## Single-stream tok/s (warm)" >>"$OUT"
echo "" >>"$OUT"
printf '| kv-ctx | mode | prefill tok/s | decode tok/s |\n|---|---|---|---|\n' >>"$OUT"
cat >/tmp/toks_parse.py <<'PY'
import sys,json
label,kv=sys.argv[1],sys.argv[2]
pt=dt=0.0
for line in sys.stdin:
    try:o=json.loads(line)
    except:continue
    if o.get("type")=="done" and o.get("id")=="m":
        pt=o.get("prefill_tok_s",0); dt=o.get("decode_tok_s",0)
print("| %s | %s | %.1f | %.1f |"%(label,kv,pt,dt))
PY
for KV in asym4 kvarn; do
    python3 - "$MODEL" "$KV" "Write a detailed paragraph about the ocean and its ecosystems." 160 <<'PY' >/tmp/gen.jsonl
import json,sys
m,kv,prompt,mx=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
print(json.dumps({"type":"load","model":m,"params":{"max_seq":4096,"kv_mode":kv}}))
print(json.dumps({"type":"generate","id":"warm","prompt":"Count to three.","temperature":0.0,"max_tokens":8}))
print(json.dumps({"type":"generate","id":"m","prompt":prompt,"temperature":0.0,"max_tokens":mx}))
PY
    timeout 300 $DAEMON </tmp/gen.jsonl 2>/dev/null | python3 /tmp/toks_parse.py "short-kv" "$KV" | tee -a "$OUT"
done
for KV in asym4 kvarn; do
    python3 - "$MODEL" "$KV" "$LONGP" 120 <<'PY' >/tmp/gen.jsonl
import json,sys
m,kv,prompt,mx=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
print(json.dumps({"type":"load","model":m,"params":{"max_seq":4096,"kv_mode":kv}}))
print(json.dumps({"type":"generate","id":"warm","prompt":"Count to three.","temperature":0.0,"max_tokens":8}))
print(json.dumps({"type":"generate","id":"m","prompt":prompt,"temperature":0.0,"max_tokens":mx}))
PY
    timeout 400 $DAEMON </tmp/gen.jsonl 2>/dev/null | python3 /tmp/toks_parse.py "long-kv" "$KV" | tee -a "$OUT"
done
echo "" >>"$OUT"
echo "Batched decode (quantized KV) runs via the serial per-session loop on dense" >>"$OUT"
echo "models → aggregate tok/s is flat (≈ single-stream) for both modes; true" >>"$OUT"
echo "batching needs the quantized-routed dense path (kvarn routed kernel built)." >>"$OUT"
cat "$OUT"
