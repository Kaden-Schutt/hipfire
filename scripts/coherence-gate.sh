#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery — replaces the byte-exact MQ4 quality gate.
#
# Rationale: byte-exact comparison blocks legitimate numerical-correctness
# improvements (e.g., norm convention fixes that change output for the
# better). This gate instead runs a small fixed matrix of (model × prompt)
# through the daemon and writes a markdown report that a human/agent
# reviewer reads before committing. The gate itself only fails on hard
# daemon/error signals (panics, non-zero exit, zero tokens emitted);
# correctness is assessed qualitatively on the report.
#
# Exit codes:
#   0  battery ran clean — open the report and inspect coherence
#   1  a test hit a hard error (daemon panic / zero tokens / timeout)
#   2  build or environment error
#
# Report destination: /tmp/coherence-<timestamp>.md (or $HIPFIRE_COHERENCE_OUT)
#
# Modes:
#   ./scripts/coherence-gate.sh          # short battery (~2-4 min)
#   ./scripts/coherence-gate.sh --full   # add A3B tests (~6-10 min)

set -u
cd "$(dirname "$0")/.."

FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        -h|--help)
            sed -n '3,21p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild daemon if any relevant source is newer than the binary ────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/hfq.rs crates/hipfire-runtime/examples/daemon.rs \
               crates/rdna-compute/src/dispatch.rs \
               crates/hipfire-dispatch/src/families/moe.rs \
               crates/hipfire-dispatch/src/pipeline/mod.rs \
               crates/hipfire-dispatch/src/pipeline/steps.rs \
               crates/hipfire-dispatch/src/families/gemv.rs \
               crates/hipfire-dispatch/src/families/attention.rs \
               crates/hipfire-dispatch/src/families/fused_qkv.rs \
               crates/hipfire-arch-deepseek4/src/arch.rs \
               crates/hipfire-arch-deepseek4/src/deepseek4.rs \
               crates/hipfire-arch-deepseek4/src/forward.rs \
               crates/hipfire-arch-deepseek4/src/spec_decode.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "coherence-gate: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Test matrix ───────────────────────────────────────────────────────────
# Format: "model_file|id|prompt|max_tokens[|system_prompt_file]"
# The optional 5th field names a file under benchmarks/prompts/ to be read
# verbatim and passed as the daemon's `system` field. Used for tool-call
# coverage (see #87 — auto-MMQ regression slipped through previous gates
# because none exercised tool-emission shapes).
# Short battery: the three dense sizes + a small multi-turn recall check
# + a tool-call shape (auto-MMQ regression detector for #87 redo).
# Full battery (--full): adds A3B MoE tests (loads large models, ~2-3 min each).
SHORT_TESTS=(
    "qwen3.5-0.8b.mq4|cap|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.5-4b.mq4|code|Write a one-line Python function named square that returns n*n.|180"
    "qwen3.5-9b.mq4|reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "qwen3.5-9b.mq4|tool-call|What does the file /tmp/fibonacci.c contain?|180|tool_call_system.txt"
    # MQ3 coverage (gfx11+gfx12 only — refused on other archs at load).
    # Verifies WMMA prefill family + K4-unroll decode + fused residual all
    # dispatch and stay coherent. Same prompts as the MQ4 rows so output
    # drift between bit-widths is comparable.
    "qwen3.5-9b.mq3|reason-mq3|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "qwen3.5-27b.mq3|cap-mq3-27b|What is the capital of France? Answer in one short sentence.|80"
    # MQ3-Lloyd coverage (PR #115 — research-gated format, --allow-mq3-lloyd
    # at quantize time only; no runtime gate). 4B + 9B exercise the K4 +
    # fp32-LDS-codebook gfx1100 kernel + tail-rotation logic. Runs anywhere
    # with the model file present.
    "qwen3.5-4b.mq3-lloyd|cap-mq3-lloyd-4b|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.5-9b.mq3-lloyd|reason-mq3-lloyd-9b|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    # MQ3-Lloyd batched-prefill coverage (companion to issue #116 Phase B2).
    # Uses a ~180-token prompt (well above MIN_BATCH=2) to exercise the
    # batched-prefill path's new WMMA fused kernels (qkv, qkvza, gate_up,
    # residual) under a realistic single-chunk forward. Prompt is loaded
    # from benchmarks/prompts/coherence_lloyd_long.txt — referenced by
    # md5 below to detect drift (per CLAUDE.md prompt-md5 rule).
    #   md5(coherence_lloyd_long.txt) = f20bbc4f5b88ab5f7b44fe7c7da0e2e3
    "qwen3.5-4b.mq3-lloyd|long-prefill-mq3-lloyd-4b|@coherence_lloyd_long.txt|220"
    "qwen3.5-9b.mq3-lloyd|long-prefill-mq3-lloyd-9b|@coherence_lloyd_long.txt|220"
    # MQ4-Lloyd coverage (issue #182 Phase B3 — regression-prevention
    # companion to B2's batched WMMA prefill wiring). Exercises
    # gemm_*_mq4g256_lloyd_wmma family + nibble-pair decode + per-row LDS
    # codebook (16 entries × 16 rows = 512 B/workgroup). 9B row catches
    # any mid-layer corruption that would surface as token-loop attractor.
    # gfx11 + gfx12 only (refused on other archs by is_batchable_la).
    "qwen3.5-9b.mq4-lloyd|reason-mq4-lloyd-9b|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    # Q8_0 batched-prefill coverage (companion to docs/plans/q8-fused-prefill-kernels.md
    # T3-0 prerequisite). Exercises the Tier 2 dispatch arms
    # (gemm_q8_0_batched_chunked at qkv/qkvza/gate_up/wo+residual/w_down+residual)
    # via the daemon's prefill path on a 9B Q8 model. Tier 3 will replace
    # those arms with fused WMMA kernels — this row is the regression detector.
    # Same long prompt as the MQ3-Lloyd rows so output drift across quant
    # formats is comparable. Single-chunk because the prompt fits comfortably
    # inside one PrefillBatchScratch chunk (PREFILL_MAX_BATCH=256).
    "qwen3.5-9b.q8f16|long-prefill-q8-9b|@coherence_lloyd_long.txt|220"
    # MQ6 coverage — different quant family (HFQ6-G256, 200 B/group). Used
    # as a regression-safety check that gfx906's new HFQ4 dp4a/prefetch
    # defaults don't disturb the mq6 dispatch routing. Skipped if model
    # absent (download via `hipfire pull qwen3.5-9b.mq6`).
    "qwen3.5-9b.mq6|reason-mq6|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    # MQ3-AWQ coverage — regression catcher for the 2026-05-18 loader bug
    # where `qwen35.rs:907` gated AWQ-sidecar attachment on
    # `matches!(wt.gpu_dtype, DType::MQ4G256)` and silently dropped sidecars
    # for MQ3G256 weights (fixed by extending to MQ4G256|MQ3G256 and then
    # centralized behind `DType::supports_awq_sidecar` — see the
    # `mq3-awq-paris` case below for the corresponding hard-fail check).
    # Uses the 4B-awq-only variant because it produces a verbose answer
    # containing "Paris" reliably at temp=0 + repeat_penalty=1.05; the
    # awq-gptq sibling terses out on this prompt (7 tokens, no Paris), so
    # it would false-positive the gate. Skipped if the canonical file
    # isn't symlinked into MODELS_DIR — symlink it as:
    #   ln -s /data/hipfire/mq3-sweep/qwen3.5-4b.mq3-awq-only.hfq \
    #         "${HIPFIRE_DIR:-$HOME/.hipfire}/models/qwen3.5-4b.mq3-awq-only"
    "qwen3.5-4b.mq3-awq-only|mq3-awq-paris|What is the capital of France? Answer in one short sentence.|80"
    # lm_head AWQ coverage — regression catcher for the AWQ-aware lm_head
    # dispatch landed in 2026-05-18's lm-head-awq-runtime PR. The 9B file
    # ships `lm_head.awq_scale.weight` alongside the 248 per-layer
    # sidecars; the runtime applies `x /= s` before the lm_head matmul
    # via `weight_gemv` → `rotate_x_mq_for` (decode) and
    # `speculative.rs::rotate_x_mq_batched_for` (spec-verify). With
    # those wirings in place, the model produces coherent output;
    # without them, the lm_head computes `(W·s)·x ≠ W·x` and emits
    # gibberish at the output projection (KLD 0.67 → 13.5 class —
    # see docs/plans/awq_fix_claude.md). max_tokens=300 because 9B
    # Qwen3.5 thinking mode spends more tokens in `<think>` before
    # emitting the final answer. Symlink to enable:
    #   ln -s /data/hipfire/qwen3.5-9b.mq4-awq-gptq-f2-lmhead-a100.hfq \
    #         "${HIPFIRE_DIR:-$HOME/.hipfire}/models/qwen3.5-9b.mq4-awq-gptq-f2-lmhead"
    "qwen3.5-9b.mq4-awq-gptq-f2-lmhead|lmhead-awq-paris|What is the capital of France? Answer in one short sentence.|300"
    # ParoQ4G128 dispatch smoke — regression catcher for the GemvResidual
    # Givens-rotation fix (0912c73a): steps.rs GemvResidual else-branch was
    # calling gemv.run(Plain) which skipped Givens rotation for Paro weights,
    # silently producing wrong o_proj output. Fixed to gemv.run_auto().
    # Uses shisa-ai's packed A3B-PARO checkpoint converted to HFQ format.
    # Two prompts: fast capital check (exercises QKV + o_proj dispatch) and
    # sheep reasoning (longer decode, stresses gate_up + o_proj repeatedly).
    # Skipped if the HFQ file is absent. To produce it:
    #   python3 scripts/paroquant_import.py import \
    #     --model ~/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-packed \
    #     --output ~/.hipfire/models/qwen3.6-35b-a3b-paro.hfq \
    #     --local-only
    # (requires numpy, torch, safetensors in the active Python environment)
    "qwen3.6-35b-a3b-paro.hfq|paro-a3b-cap|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.6-35b-a3b-paro.hfq|paro-a3b-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|500"
)
FULL_EXTRA=(
    "qwen3.5-35b-a3b.mq4|moe-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|500"
    # gfx12/RDNA4 Q8_0-wo MoE coverage — the gate gap that let a9e8dfda
    # corrupt RDNA4 MoE output for ~100 commits before ae13aa75 fixed it.
    # a9e8dfda aliased the GemvResidual fallback's `out` onto the residual
    # buffer; on RDNA4 the Q8_0 wo / dn_out projections (which take that
    # GemvResidual fallback rather than a fused *_residual_wmma kernel)
    # then read-after-write the same buffer → silent wrong MoE output.
    # NONE of the prior FULL_EXTRA rows exercised a *Q8_0* MoE (the only
    # MoE rows were MQ4), so the aliasing regression slipped every gate.
    # This row is a qwen3_5_moe (arch) A3B whose router + shared-gate +
    # wo/dn_out projections are Q8_0, forcing the GemvResidual fallback on
    # gfx12. A healthy answer states the final number (8); aliasing
    # corruption produces an attractor / off-topic loop. It is symlink-gated
    # exactly like the other rows (skipped where the file is absent, so it
    # is a no-op on boxes without the Q8 A3B trunk). Produce / symlink it:
    #   ln -s /data/hipfire-models/qwen3.5-35b-a3b.q8f16.hfq \
    #         "${HIPFIRE_DIR:-$HOME/.hipfire}/models/qwen3.5-35b-a3b.q8f16"
    "qwen3.5-35b-a3b.q8f16|moe-q8-wo-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|500"
    "qwen3.6-35b-a3b.mq4|moe36-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|800"
    "qwen3.6-27b.mq4|tool-call-27b|What does the file /tmp/fibonacci.c contain?|220|tool_call_system.txt"
    # DeepSeek V4 Flash (arch_id=9, hipfire-arch-deepseek4). Loads the
    # 80 GB base from the MTP-sidecar split + the MTP addon via
    # HIPFIRE_DEEPSEEK4_MTP_ADDON. Skipped unless the symlink is present in
    # MODELS_DIR; symlink with:
    #   ln -s /data/hipfire-models/deepseek-v4-flash.mq2lloyd \
    #         "${HIPFIRE_DIR:-$HOME/.hipfire}/models/deepseek-v4-flash.mq2lloyd"
    #   ln -s /data/hipfire-models/deepseek-v4-flash-mtp.mq2lloyd \
    #         "${HIPFIRE_DIR:-$HOME/.hipfire}/models/deepseek-v4-flash-mtp.mq2lloyd"
    # HIPFIRE_DEEPSEEK4_MTP_ADDON is auto-set by the run-case when the sibling
    # addon file exists.
    "deepseek-v4-flash.mq2lloyd|deepseek4-cap|What is the capital of France? Answer in one short sentence.|80"
    "deepseek-v4-flash.mq2lloyd|deepseek4-reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "deepseek-v4-flash.mq2lloyd|deepseek4-long-prefill|@coherence_lloyd_long.txt|220"
)
tests=("${SHORT_TESTS[@]}")
if [ "$FULL" -eq 1 ]; then
    tests+=("${FULL_EXTRA[@]}")
fi

# ── Run ───────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Coherence battery"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FULL" -eq 1 ] && echo full || echo short )"
    echo
    echo "Review each output for coherence (fluent English, on-topic, not stuck"
    echo "in verbatim loops). Hard errors fail the gate; soft output changes do not."
    echo
} > "$OUT"

for entry in "${tests[@]}"; do
    IFS='|' read -r model_file prompt_id prompt max_tok system_file <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [ ! -f "$model_path" ]; then
        echo "## $model_file — $prompt_id — SKIPPED (model not present)" >> "$OUT"
        echo >> "$OUT"
        continue
    fi

    # `@filename` syntax: read the user prompt from benchmarks/prompts/<file>.
    # Used for long-prompt batched-prefill rows where embedding the prompt
    # inline would violate CLAUDE.md's prompt-md5 rule (heredocs in scripts
    # are reformatting-sensitive, breaking byte-identical reproduction).
    prompt_md5=""
    prompt_ref=""
    if [ "${prompt:0:1}" = "@" ]; then
        prompt_ref="${prompt:1}"
        prompt_path="benchmarks/prompts/$prompt_ref"
        if [ ! -f "$prompt_path" ]; then
            echo "## $model_file — $prompt_id — SKIPPED (prompt file $prompt_path not found)" >> "$OUT"
            continue
        fi
        prompt=$(cat "$prompt_path")
        prompt_md5=$(md5sum "$prompt_path" | awk '{print $1}')
    fi

    # Optional system prompt: load from benchmarks/prompts/ if specified
    # in the test entry. Used for tool-call shape coverage (#87) where the
    # system block contains the tools <tools>...</tools> definition.
    system_json=""
    if [ -n "${system_file:-}" ]; then
        system_path="benchmarks/prompts/$system_file"
        if [ -f "$system_path" ]; then
            system_text=$(python3 -c "import sys,json; print(json.dumps(open(sys.argv[1]).read()))" "$system_path")
            system_json=",\"system\":${system_text}"
        else
            echo "## $model_file — $prompt_id — SKIPPED (system prompt $system_path not found)" >> "$OUT"
            continue
        fi
    fi

    echo "== $model_file / $prompt_id =="
    # JSONL input for daemon. Use python json.dumps for the user prompt so
    # special tokens / quotes / backslashes in the fixture survive intact
    # (the previous sed-based escape only handled `"`, missing tabs, JSON
    # control chars, and `\n` literals — a tool-emit fixture would need
    # those).
    in_file="/tmp/coh_in_$$.jsonl"
    out_file="/tmp/coh_out_$$.log"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.05${system_json}}
{"type":"unload"}
JL
    # DeepSeek V4 (arch_id=9): point HIPFIRE_DEEPSEEK4_MTP_ADDON at the sibling `-mtp`
    # file if one exists. MoE / expert upload / deterministic MoE-down
    # are default-on in the DeepSeek V4 arch crate. Other arches inherit the
    # caller's env unchanged.
    declare -a deepseek4_env=()
    if [[ "$model_file" == deepseek-v4* ]]; then
        addon="${model_path%.mq2lloyd}-mtp.mq2lloyd"
        if [ -f "$addon" ]; then
            deepseek4_env+=("HIPFIRE_DEEPSEEK4_MTP_ADDON=$addon")
        fi
    fi

    t0=$(date +%s.%N)
    env "${deepseek4_env[@]}" timeout 240 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    fi

    # Tool-call shape: hard-fail on ChatML special-token leakage in the
    # visible output. Healthy tool-call emit looks like:
    #   <tool_call>{"name":"read","arguments":{"path":"/tmp/foo.c"}}</tool_call><|im_end|>
    # The corruption signature from #87 (auto-MMQ regression on gfx1151) is
    # `<|im_start|>` *inside* the tool_call body, e.g.
    #   <tool_call>\n<|im_start|>box", "bash","command":"..."
    # Count `<|im_start|>` in the assembled token text; healthy = 0,
    # corrupted ≥ 1. Trailing `<|im_end|>` is fine and stripped by clients;
    # we only flag im_start as a hard failure since legitimate output
    # never embeds it.
    case "$prompt_id" in
        tool-call*)
            text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')
            im_start_leaks=$(printf '%s' "$text" | grep -oE '<\|im_start\|>' | wc -l | tr -d ' ')
            if [ "${im_start_leaks:-0}" -gt 0 ]; then
                status="HARD_ERROR (tool-call corruption: ${im_start_leaks}× <|im_start|> leaked into visible output — see #87)"
                hard_errors=$((hard_errors + 1))
            elif ! printf '%s' "$text" | grep -qE '<tool_call>'; then
                # Soft warn — model didn't emit a tool_call at all. Could
                # be quantization noise (small model decided to answer
                # directly); not a corruption signal.
                status="OK (soft: no <tool_call> emitted; model answered inline)"
            fi
            ;;
        mq3-awq-paris|lmhead-awq-paris)
            # Regression catcher for AWQ-sidecar load + dispatch bugs.
            # Two flavors share this hard-fail check:
            #   - mq3-awq-paris (4B): catches the 2026-05-18 loader bug
            #     where `qwen35.rs:907` gated AWQ-sidecar attachment on
            #     `matches!(_, DType::MQ4G256)` and silently dropped MQ3
            #     sidecars.
            #   - lmhead-awq-paris (9B): catches the 2026-05-18 spec-verify
            #     dispatch bug where `speculative.rs`'s lm_head path
            #     called the non-AWQ `rotate_x_mq_batched` even when an
            #     lm_head sidecar was attached. Same dispatch class as
            #     the DFlash drafter bug fixed in PR #290.
            # A healthy Qwen3.5 MQ4-AWQ answer always contains "Paris"
            # verbatim at temp=0 — even the lowest-PPL calibration gets
            # the capital right. Anything else means the AWQ rescale is
            # not being applied at some stage.
            text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')
            if ! printf '%s' "$text" | grep -q 'Paris'; then
                status="HARD_ERROR (AWQ regression: answer missing 'Paris' — AWQ rescale likely not applied; check DType::supports_awq_sidecar gate and rotate_x_mq_for / rotate_x_mq_batched_for dispatch)"
                hard_errors=$((hard_errors + 1))
            fi
            ;;
    esac

    {
        echo "## $model_file — $prompt_id"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        if [ -n "$done_line" ]; then
            echo "- stats: \`$done_line\`"
        fi
        if [ -n "$prompt_md5" ]; then
            echo "- prompt: \`@$prompt_ref\` (md5: \`$prompt_md5\`)"
        else
            echo "- prompt: \"$prompt\""
        fi
        echo
        if [ -n "$panic" ]; then
            echo '**PANIC/ERROR DETECTED:**'
            echo
            echo '```'
            echo "$panic"
            echo '```'
            echo
        fi
        echo '**Output:**'
        echo
        echo '```'
        grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))'
        echo '```'
        echo
    } >> "$OUT"

    rm -f "$in_file" "$out_file"
done

echo
echo "coherence report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"

# ── PFlash regression stage ─────────────────────────────────────────────
# Optional follow-up stage that asserts PFlash bench wall-clock and
# verdicts haven't regressed against the committed baseline. Skipped
# when target/drafter aren't present or when HIPFIRE_SKIP_PFLASH_GATE=1.
# Release the daemon GPU lock first so pflash-gate.sh can acquire its own.
gpu_release 2>/dev/null || true
trap - EXIT

if [ "${HIPFIRE_SKIP_PFLASH_GATE:-0}" = "1" ]; then
    echo
    echo "pflash-gate: SKIPPED (HIPFIRE_SKIP_PFLASH_GATE=1)"
    exit 0
fi
PFLASH_TARGET="${HIPFIRE_PFLASH_TARGET:-$MODELS_DIR/qwen3.5-27b.mq3}"
PFLASH_DRAFTER="${HIPFIRE_PFLASH_DRAFTER:-$MODELS_DIR/qwen3.5-0.8b.mq4}"
if [ ! -f "$PFLASH_TARGET" ] || [ ! -f "$PFLASH_DRAFTER" ]; then
    echo
    echo "pflash-gate: SKIPPED (target or drafter not present)"
    exit 0
fi

echo
echo "── pflash regression stage ────────────────────────────────────────"
HIPFIRE_PFLASH_TARGET="$PFLASH_TARGET" HIPFIRE_PFLASH_DRAFTER="$PFLASH_DRAFTER" \
    ./scripts/pflash-gate.sh
pflash_rc=$?
if [ "$pflash_rc" -ne 0 ]; then
    echo "pflash-gate: FAILED (exit $pflash_rc) — combined gate FAILED"
    exit 1
fi
exit 0
