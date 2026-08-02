#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# MI300X DS4 MQ2R compressed-cache cap measurement campaign.
# Each subprocess is a fresh process and banks a standalone log. The cap sweep
# uses a synthetic absolute position: it measures allocation and decode scaling,
# not cache quality. Top-k runs use real prefill contents.
set -u

export PATH="${PATH}:/root/.cargo/bin:/opt/rocm/core-7.14/bin"
ROOT="${HIPFIRE_ROOT:-/root/hipfire-work/ds4-parent-gate}"
OUT="${DS4_CAP_OUT:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-cap-campaign}"
MODEL="${MQ2R_MODEL:-/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r}"
EXPECT_SHA="${MQ2R_SHA:-cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce}"
BIN="${ROOT}/target/release/examples/ds4_longctx_probe"
PROFILE_PY="${ROOT}/crates/hipfire-arch-deepseek4/scripts/ds4_longctx_profile.py"
mkdir -p "$OUT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$OUT/campaign.log"; }
run_logged() {
    local name="$1"
    shift
    log "START ${name}: $*"
    "$@" 2>&1 | tee "$OUT/${name}.log"
    local rc=${PIPESTATUS[0]}
    log "END ${name}: rc=${rc}"
    return "$rc"
}

cmd="${1:-all}"
wall0=$(date +%s)
printf 'campaign_start=%s\nhost=%s\nmodel=%s\nexpected_sha256=%s\n' \
    "$(stamp)" "$(hostname)" "$MODEL" "$EXPECT_SHA" >"$OUT/provenance.txt"

if [[ "${DS4_TRUST_VERIFIED_SHA:-0}" == "1" && -f "$OUT/model.sha256.ok" ]]; then
    model_sha=$(cat "$OUT/model.sha256.ok")
    log "trusting prior campaign hash ${model_sha}"
else
    log "hashing 82 GB MQ2R artifact once"
    model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
    if [[ "$model_sha" != "$EXPECT_SHA" ]]; then
        log "FATAL model sha mismatch got=${model_sha} expected=${EXPECT_SHA}"
        exit 3
    fi
    printf '%s\n' "$model_sha" >"$OUT/model.sha256.ok"
fi
printf 'verified_sha256=%s\n' "$model_sha" >>"$OUT/provenance.txt"
log "model sha verified ${model_sha}"

if [[ ! -x "$BIN" ]]; then
    log "building ds4_longctx_probe"
    (cd "$ROOT" && cargo build --release -p hipfire-arch-deepseek4 --example ds4_longctx_probe) \
        >"$OUT/build.log" 2>&1
    build_rc=$?
    if [[ $build_rc -ne 0 ]]; then
        log "FATAL build failed rc=${build_rc}; see $OUT/build.log"
        exit "$build_rc"
    fi
fi
sha256sum "$BIN" >"$OUT/probe_binary.sha256"

run_sweep() {
    local caps=(2048 8192 32768 131072 262144)
    : >"$OUT/sweep_status.tsv"
    printf 'cap\tposition\trc\n' >>"$OUT/sweep_status.tsv"
    for cap in "${caps[@]}"; do
        position=$((cap * 4))
        run_logged "cap_${cap}" "$BIN" "$MODEL" --cap "$cap" --position "$position" --decode-reps 3
        rc=$?
        printf '%s\t%s\t%s\n' "$cap" "$position" "$rc" >>"$OUT/sweep_status.tsv"
        if [[ $rc -ne 0 ]]; then
            log "first cap failure cap=${cap} position=${position}; stopping climb"
            break
        fi
    done
}

run_profiles() {
    local caps=(2048 32768 262144)
    for cap in "${caps[@]}"; do
        position=$((cap * 4))
        pdir="$OUT/rocprof_cap_${cap}"
        rm -rf "$pdir"
        mkdir -p "$pdir"
        run_logged "rocprof_cap_${cap}" \
            /opt/rocm/core-7.14/bin/rocprofv3 --kernel-trace --stats \
            --output-directory "$pdir" --output-file "cap_${cap}" -- \
            "$BIN" "$MODEL" --cap "$cap" --position "$position" --decode-reps 2
        rc=$?
        if [[ $rc -eq 0 ]]; then
            python3 "$PROFILE_PY" "$pdir" --decode-steps 4 \
                --json "$OUT/profile_cap_${cap}.json" \
                | tee "$OUT/profile_cap_${cap}.txt"
        else
            log "profile failed cap=${cap}; continuing with banked sweep"
        fi
    done
}

run_topk() {
    local contexts=(4096 16384 65536)
    for context in "${contexts[@]}"; do
        cap=$(((context + 3) / 4))
        extra=()
        if [[ "$context" == "65536" ]]; then
            extra=(--generate 32)
        fi
        run_logged "topk_${context}" "$BIN" "$MODEL" --cap "$cap" \
            --prefill "$context" --batch 1024 --decode-reps 2 --topk-sanity "${extra[@]}"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            log "top-k real-prefill failure context=${context}; stopping top-k climb"
            break
        fi
    done
}

case "$cmd" in
    sweep) run_sweep ;;
    profile) run_profiles ;;
    topk) run_topk ;;
    all)
        run_sweep
        run_profiles
        run_topk
        ;;
    *)
        echo "usage: $0 [all|sweep|profile|topk]" >&2
        exit 2
        ;;
esac

wall1=$(date +%s)
printf 'campaign_end=%s\ntotal_wall_s=%s\n' "$(stamp)" "$((wall1 - wall0))" >>"$OUT/provenance.txt"
log "campaign cmd=${cmd} total_wall_s=$((wall1 - wall0))"
