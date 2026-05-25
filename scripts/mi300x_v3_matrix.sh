#!/usr/bin/env bash
# mi300x_v3_matrix.sh — apply the v3 recipe across all supported Qwen3
# trunk models and emit a comparable result table.
#
# Per model:
#   1. hipfire-quantize ... --format mq4 --awq --awq-alpha 0.5 --imatrix
#      → AWQ base HFQ (F1 scope, 184 sidecars on 9B-class)
#   2. mq4_masked_calib.py quantize --method gptq --gpu 0 --awq-aware-hessian
#      → GPTQ-corrected v3 HFQ at byte-identical size
#   3. Generate or reuse <model>-bf16.kldref.bin
#   4. eval_hipfire → KLD + PPL @ c512 q8 prefill
#   5. coherence_probe --max-tokens 200 → hard/soft fail counts
#   6. bench: AR decode tok/s (--max 120)
#
# Output: $WORK/results/v3-matrix/<ts>/<model>/ with JSON summary per model
# plus a top-level table.md. Idempotent: skips already-quantized models.

set -euo pipefail

WORK="${WORK:-/workspace}"
HIPFIRE="${HIPFIRE_DIR:-${WORK}/hipfire}"
HF_HOME="${HF_HOME:-${WORK}/hf-cache}"
IMATRIX_DIR="${WORK}/imatrix"
MODELS_OUT="${WORK}/models/v3"
TS="${TS:-$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="${WORK}/results/v3-matrix/${TS}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$MODELS_OUT" "$RUN_DIR"
cd "$HIPFIRE"

# ── Matrix definition ──────────────────────────────────────────────────────
# Lines: <slug> <hf_repo> <hf_revision> <imatrix_subdir>
# slug is used for output filenames + result key.
read -r -d '' MATRIX <<'EOF' || true
qwen3.5-0.8b    Qwen/Qwen3.5-0.8B           2fc06364715b967f1860aea9cf38778875588b17  Qwen3.5-0.8B-GGUF
qwen3.5-9b      Qwen/Qwen3.5-9B             c202236235762e1c871ad0ccb60c8ee5ba337b9a  Qwen3.5-9B-GGUF
qwen3.5-27b     Qwen/Qwen3.5-27B            b7ca741b86de18df552fd2cc952861e04621a4bd  Qwen3.5-27B-GGUF
qwen3.6-27b     Qwen/Qwen3.6-27B            6a9e13bd6fc8f0983b9b99948120bc37f49c13e9  Qwen3.6-27B-GGUF
qwen3.6-35b-a3b Qwen/Qwen3.6-35B-A3B        7da1103448ba36029c34ce1a9a741dfe93ee0c50  Qwen3.6-35B-A3B-GGUF
EOF

# ── Per-model overrides ────────────────────────────────────────────────────
# A3B has a MoE router that's in F1 scope by default and triggered tool-call
# schema corruption in PR #225/MFP4. v3 uses MQ4 not MFP4 — risk unknown.
# For safety the first pass excludes the router; flip A3B_INCLUDE_ROUTER=1 to
# also try the inclusive variant.
A3B_INCLUDE_ROUTER="${A3B_INCLUDE_ROUTER:-0}"

# Eval params (canonical from CLAUDE.md):
KLD_PROMPT_NORMALIZE="${KLD_PROMPT_NORMALIZE:-true}"
EVAL_KV_MODE="${EVAL_KV_MODE:-q8}"
EVAL_CTX="${EVAL_CTX:-512}"
BENCH_MAX="${BENCH_MAX:-120}"
COHERENCE_MAX_TOKENS="${COHERENCE_MAX_TOKENS:-200}"

# ── Helpers ────────────────────────────────────────────────────────────────
phase() { echo; echo "═══ [$(date +%H:%M:%S)] $* ═══"; }
ok()    { printf "    \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "    \033[33m!\033[0m %s\n" "$*"; }
die()   { printf "    \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# Resolve a pinned HF snapshot to a local path.
resolve_snapshot() {
    local repo="$1" rev="$2"
    $PYTHON -c "
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id='$repo', revision='$rev',
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'],
))
"
}

# Generate kldref.bin from BF16 source if not present.
ensure_kldref() {
    local slug="$1" bf16_dir="$2"
    local out_dir="$WORK/kldref"
    mkdir -p "$out_dir"
    local kldref="$out_dir/${slug}-bf16.kldref.bin"
    if [ -s "$kldref" ]; then
        ok "kldref cached: $kldref"
        echo "$kldref"
        return
    fi
    echo "    [kldref] generating from BF16 — this is the slowest single step" >&2
    if [ ! -x target/release/examples/make_kldref ]; then
        cargo build --release --example make_kldref -p hipfire-runtime 2>&1 | tail -5 >&2 || \
            warn "make_kldref not in workspace; computing via PyTorch fallback"
    fi
    if [ -x target/release/examples/make_kldref ]; then
        ./target/release/examples/make_kldref \
            --bf16 "$bf16_dir" \
            --calib benchmarks/calib/calib-1m.txt \
            --ctx "$EVAL_CTX" \
            --out "$kldref" 2>&1 | tail -5 >&2
    else
        # PyTorch fallback for kldref generation
        $PYTHON scripts/make_kldref_torch.py \
            --model "$bf16_dir" \
            --calib benchmarks/calib/calib-1m.txt \
            --ctx "$EVAL_CTX" \
            --out "$kldref" 2>&1 | tail -5 >&2 \
        || die "no kldref generation path available — need make_kldref binary or scripts/make_kldref_torch.py"
    fi
    [ -s "$kldref" ] || die "kldref generation failed: $kldref"
    echo "$kldref"
}

# ── Run a single model ──────────────────────────────────────────────────────
run_one() {
    local slug="$1" repo="$2" rev="$3" im_subdir="$4"
    phase "Model: $slug ($repo @ ${rev:0:8})"

    local model_out_dir="$RUN_DIR/$slug"
    mkdir -p "$model_out_dir"

    local imatrix="$IMATRIX_DIR/$im_subdir/imatrix_unsloth.gguf_file"
    [ -s "$imatrix" ] || die "missing imatrix: $imatrix"

    local awq_base="$MODELS_OUT/${slug}.mq4-awq"
    local v3_out="$MODELS_OUT/${slug}.mq4-v3"

    # Stage 1: AWQ-prescaled MQ4 base
    if [ ! -f "$awq_base" ]; then
        local bf16_dir
        bf16_dir=$(resolve_snapshot "$repo" "$rev")
        ok "BF16 source: $bf16_dir"

        local awq_args=( --input "$bf16_dir" --output "$awq_base"
                         --format mq4 --awq --awq-alpha 0.5
                         --imatrix "$imatrix" )
        # MoE router exclusion for A3B unless explicitly opted in
        if [ "$slug" = "qwen3.6-35b-a3b" ] && [ "$A3B_INCLUDE_ROUTER" != "1" ]; then
            awq_args+=( --awq-exclude-pattern "router.weight" )
            ok "A3B: excluding router from AWQ (set A3B_INCLUDE_ROUTER=1 to override)"
        fi
        ( cd "$HIPFIRE" && ./target/release/hipfire-quantize "${awq_args[@]}" ) 2>&1 \
            | tail -15 | tee "$model_out_dir/stage1_awq.log"
        [ -f "$awq_base" ] || die "stage 1 produced no output"
        local sz=$(stat -c%s "$awq_base")
        ok "stage 1 done: $awq_base ($sz bytes)"
    else
        ok "stage 1 cached: $awq_base"
    fi

    # Stage 2: AWQ-aware GPTQ
    if [ ! -f "$v3_out" ]; then
        local bf16_dir
        bf16_dir=$(resolve_snapshot "$repo" "$rev")
        $PYTHON scripts/mq4_masked_calib.py quantize \
            --base "$awq_base" \
            --source-dir "$bf16_dir" \
            --output "$v3_out" \
            --out "$model_out_dir/stage2_gptq" \
            --method gptq --gpu 0 \
            --awq-aware-hessian "$awq_base" \
            --skip-unsupported \
            2>&1 | tail -10 | tee "$model_out_dir/stage2_gptq.log"
        [ -f "$v3_out" ] || die "stage 2 produced no output"
        local sz=$(stat -c%s "$v3_out")
        ok "stage 2 done: $v3_out ($sz bytes)"
    else
        ok "stage 2 cached: $v3_out"
    fi

    # Stage 3: ensure kldref.bin
    local bf16_dir
    bf16_dir=$(resolve_snapshot "$repo" "$rev")
    local kldref
    kldref=$(ensure_kldref "$slug" "$bf16_dir")

    # Stage 4: KLD eval (c512 q8 prefill)
    local kld_json="$model_out_dir/kld.json"
    if [ ! -s "$kld_json" ]; then
        ./target/release/examples/eval_hipfire \
            --model "$v3_out" \
            --kldref "$kldref" \
            --kv-mode "$EVAL_KV_MODE" \
            --ctx "$EVAL_CTX" \
            --scoring-mode prefill \
            --emit-json "$kld_json" \
            2>&1 | tail -10 | tee "$model_out_dir/kld.log"
    fi
    [ -s "$kld_json" ] && ok "KLD eval: $(jq -r '"KLD=" + (.kld_mean|tostring) + " PPL=" + (.ppl|tostring)' "$kld_json")"

    # Stage 5: coherence_probe
    local coh_json="$model_out_dir/coherence.json"
    if [ ! -s "$coh_json" ]; then
        ./target/release/examples/coherence_probe \
            --model "$v3_out" \
            --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
            --max-tokens "$COHERENCE_MAX_TOKENS" \
            --temperature 0.0 \
            --emit-json "$coh_json" \
            2>&1 | tail -10 | tee "$model_out_dir/coherence.log" || true
    fi
    [ -s "$coh_json" ] && ok "coherence: $(jq -r '"hard=" + (.hard_fails|tostring) + " soft=" + (.soft_fails|tostring)' "$coh_json" 2>/dev/null || echo unknown)"

    # Stage 6: tok/s
    local bench_json="$model_out_dir/bench.json"
    if [ ! -s "$bench_json" ]; then
        ./target/release/examples/eval_hipfire \
            --model "$v3_out" \
            --bench \
            --max "$BENCH_MAX" \
            --kv-mode "$EVAL_KV_MODE" \
            --no-chatml \
            --emit-json "$bench_json" \
            2>&1 | tail -10 | tee "$model_out_dir/bench.log" || true
    fi
    [ -s "$bench_json" ] && ok "bench: $(jq -r '"decode=" + (.decode_tok_s|tostring) + " prefill=" + (.prefill_tok_s|tostring)' "$bench_json" 2>/dev/null || echo unknown)"

    # Final summary blob
    $PYTHON - "$slug" "$kld_json" "$coh_json" "$bench_json" > "$model_out_dir/summary.json" <<'PY'
import json, sys
slug, kld_p, coh_p, bench_p = sys.argv[1:5]
def safe(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
out = {
    "slug": slug,
    "kld": safe(kld_p),
    "coherence": safe(coh_p),
    "bench": safe(bench_p),
}
print(json.dumps(out, indent=2, sort_keys=True))
PY
    ok "summary: $model_out_dir/summary.json"
}

# ── Run all ────────────────────────────────────────────────────────────────
phase "v3 matrix start — run dir: $RUN_DIR"
echo "$MATRIX" | while read -r slug repo rev im_subdir; do
    [ -n "$slug" ] || continue
    run_one "$slug" "$repo" "$rev" "$im_subdir" \
        || { warn "$slug FAILED — continuing with next model"; continue; }
done

# ── Roll up into table.md ──────────────────────────────────────────────────
phase "Roll up results"
$PYTHON - "$RUN_DIR" > "$RUN_DIR/table.md" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
rows = []
for d in sorted(run_dir.iterdir()):
    s = d / "summary.json"
    if not s.exists():
        continue
    j = json.load(s.open())
    kld = j.get("kld") or {}
    coh = j.get("coherence") or {}
    bench = j.get("bench") or {}
    rows.append({
        "slug": j["slug"],
        "kld": kld.get("kld_mean"),
        "ppl": kld.get("ppl"),
        "decode": bench.get("decode_tok_s"),
        "prefill": bench.get("prefill_tok_s"),
        "coh_hard": coh.get("hard_fails"),
        "coh_soft": coh.get("soft_fails"),
    })
print("# v3 matrix on MI300x")
print()
print("| Model | KLD | PPL | Decode | Prefill | Coh hard | Coh soft |")
print("|---|---:|---:|---:|---:|---:|---:|")
def fmt(v, w=".4f"):
    return "—" if v is None else format(v, w)
for r in rows:
    print(f"| {r['slug']} | {fmt(r['kld'])} | {fmt(r['ppl'],'.3f')} | "
          f"{fmt(r['decode'],'.1f')} | {fmt(r['prefill'],'.1f')} | "
          f"{r.get('coh_hard','—')} | {r.get('coh_soft','—')} |")
PY
echo
cat "$RUN_DIR/table.md"
echo
ok "Done. Full results: $RUN_DIR"
echo
echo "Next: bash $HIPFIRE/scripts/mi300x_sub_0_10_attempt.sh"
