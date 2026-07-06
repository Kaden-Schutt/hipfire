#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 alpineq
# hipfire — see LICENSE and NOTICE in the project root.

# Multi-GPU pipeline-parallel gate.
#
# Validates that pp>1 paths still match pp=1 byte-for-byte under the
# deterministic flag. Skips silently when fewer than 2 usable HIP devices
# are visible — that's the expected state in CI / single-GPU dev boxes, and
# we don't want pp-gate to block commits that don't touch PP code.
#
# "Usable" means: AMD GPU, not an APU iGPU, sharing an ISA family with the
# other visible GPU. See docs/plans/pp-gate-fix.md for the rationale.
#
# Three barriers (all skipped on <2 usable GPU):
#
#   1. pp_parity_chatml example — per-token forward_scratch{,_multi}
#      bit-equivalence (50 decode tokens after ChatML prefill, asym3 KV).
#      This is the floor: if forward_scratch_multi diverges from
#      forward_scratch even with HIPFIRE_DETERMINISTIC=1, pp=2 is broken.
#
#   2. daemon pp=1 vs pp=2 byte-equivalence on dense 0.8B mq4 — the
#      end-to-end smoke. Catches regressions in the load/prefill/decode/
#      sample chain that the example doesn't exercise (top_p sampler,
#      repeat penalty, attractor block, ChatML wrap, etc.). Run with
#      HIPFIRE_DETERMINISTIC=1 to opt out of the inherited ksplit
#      atomicAdd reduction non-det.
#
#   3. Refusal sanity — DFlash + pp=2 and CASK + pp=2 must still
#      produce clean error messages at load. These are v1 contract
#      promises (see plan v2 stages 7 + 9).
#
# Environment knobs (precedence: PP_GATE_DEVICES > INCLUDE_IGPU > HETEROGENEOUS):
#   PP_GATE_DEVICES=0,1                 # operator pin; bypasses all filters
#   HIPFIRE_PP_GATE_INCLUDE_IGPU=1      # don't filter known APU iGPUs
#   HIPFIRE_PP_GATE_HETEROGENEOUS=1     # don't skip on mixed ISA families
#   HIPFIRE_PP_GATE_REQUIRE_SYSFS=1     # hard-fail instead of falling back to rocm-smi
#   HIPFIRE_PP_GATE_MODEL=<path>        # override default model
#   HIPFIRE_PP_GATE_TEST_NO_SYSFS=1     # force sysfs probe to "fail" (validation harness)
#
# Exit codes:
#   0  passed, or skipped (<2 usable GPU, iGPU-only host, etc.)
#   1  hard failure (parity broken, refusal missing, daemon panic)
#   2  build / environment / operator error (e.g. bad PP_GATE_DEVICES)
#
# Run by .githooks/pre-commit when staged diff touches multi_gpu.rs,
# pp_*, peer_access, or other pipeline-related code.
#
# Manual invocation:
#   ./scripts/pp-gate.sh                       # full battery
#   ./scripts/pp-gate.sh --skip-end-to-end     # parity example only
#   ./scripts/pp-gate.sh --dry-run             # parser/filter trace, no GPU work
#   ./scripts/pp-gate.sh --dry-run --simulate=0:gfx1100,1:gfx1101
#                                              # feed canned topology

set -u
cd "$(dirname "$0")/.." || { echo "pp-gate: failed to cd to repo root" >&2; exit 2; }

# ── Platform gate ───────────────────────────────────────────────────────
# pp-gate's underlying daemon + dual-GPU dispatch is Linux-ROCm-only.
# Git Bash / MSYS2 / Cygwin lack /sys/class/kfd and have no working PP
# story on Windows ROCm. WSL is Linux from the script's POV (uname → Linux).
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "pp-gate: Windows host ($(uname -s)) — skipping (Linux ROCm only)"
        exit 0
        ;;
esac

SKIP_E2E=0
DRY_RUN=0
SIMULATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-end-to-end) SKIP_E2E=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --simulate=*) SIMULATE="${1#--simulate=}" ;;
        -h|--help)
            sed -n '3,52p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# --simulate replaces real device probing; only meaningful in --dry-run.
# Without this guard, a typo like `--simulate=0:gfx1100` (no --dry-run)
# would actually build and run with the simulated indices.
if [ -n "$SIMULATE" ] && [ "$DRY_RUN" != "1" ]; then
    echo "pp-gate: --simulate requires --dry-run (it bypasses real device probing)" >&2
    exit 2
fi

# ── ROCm env ────────────────────────────────────────────────────────────
# Auto-source rocm-env.sh so HIP libraries land on the loader path on
# NixOS-style hosts. No-op if already loaded.
if [ -r "./scripts/rocm-env.sh" ]; then
    # shellcheck disable=SC1091
    . ./scripts/rocm-env.sh
fi

# ── iGPU arch allowlist ─────────────────────────────────────────────────
# Known APU iGPU gfx archs. Verified against LLVM AMDGPU.td (LLVM 18+) +
# ROCm device-libs. Adding a new APU is a one-line change here.
#
# Source of truth for engine-side arch handling:
#   crates/rdna-compute/src/kernels.rs:80-139 (WMMA family groups)
#   crates/rdna-compute/src/dispatch.rs       (per-arch dispatch)
#
# NOT in this list (intentionally):
#   gfx1150 (Strix Point)  — 16-CU iGPU, plausibly a useful PP partner
#   gfx1151 (Strix Halo)   — dGPU-class memory (32+ GB unified)
#   gfx1152                — classification unclear; let homogeneity check handle it
IGPU_ARCHS=(
    gfx902    # Raven Ridge / Picasso (Ryzen 2000G/3000G)
    gfx909    # Raven2 / Dali
    gfx90c    # Renoir / Lucienne / Cezanne / Barcelo (4xxx-5xxx APU)
    gfx1013   # Van Gogh (Steam Deck) / BC-250
    gfx1033   # Rembrandt (6xxx mobile)
    gfx1034   # Rembrandt-R
    gfx1035   # Rembrandt-R refresh
    gfx1036   # Raphael desktop iGPU (AM5 7xxx)
    gfx1103   # Phoenix / Phoenix2 / Hawk Point
)

# ── ISA family table ────────────────────────────────────────────────────
# Two devices are "heterogeneous" only when their families differ.
# rdna3/rdna4 groupings mirror crates/rdna-compute/src/kernels.rs:80-139
# WMMA matchsets. cdna/rdna1/rdna2 groupings come from the wider per-arch
# dispatch in crates/rdna-compute/src/dispatch.rs (kernels.rs doesn't
# cover those families at line 80-139). The apu_* buckets are unreachable
# under default flags (stage 2 filters iGPUs first) and only matter when
# INCLUDE_IGPU=1 forces them into the homogeneity check.
arch_family() {
    case "$1" in
        gfx906|gfx908|gfx90a|gfx942)            echo "cdna" ;;
        gfx1010|gfx1011|gfx1012)                echo "rdna1" ;;
        gfx1030|gfx1031|gfx1032)                echo "rdna2" ;;
        gfx1100|gfx1101|gfx1102|gfx1150|gfx1151) echo "rdna3" ;;
        gfx1200|gfx1201)                        echo "rdna4" ;;
        gfx902|gfx909|gfx90c)                   echo "apu_gcn5" ;;
        gfx1013|gfx1033|gfx1034|gfx1035|gfx1036|gfx1103) echo "apu_rdna" ;;
        *)                                       echo "unknown_$1" ;;
    esac
}

# ── gfx_target_version integer → gfx string ─────────────────────────────
# Integer encoding from amdkfd: major * 10000 + minor * 100 + stepping.
# Stepping is hex for gfx90a (10) and gfx90c (12). Hand-coded for known archs.
gfx_int_to_str() {
    case "$1" in
        90002)  echo "gfx902" ;;
        90006)  echo "gfx906" ;;
        90008)  echo "gfx908" ;;
        90009)  echo "gfx909" ;;
        90010)  echo "gfx90a" ;;
        90012)  echo "gfx90c" ;;
        90402)  echo "gfx942" ;;
        100100) echo "gfx1010" ;;
        100101) echo "gfx1011" ;;
        100102) echo "gfx1012" ;;
        100103) echo "gfx1013" ;;
        100300) echo "gfx1030" ;;
        100301) echo "gfx1031" ;;
        100302) echo "gfx1032" ;;
        100303) echo "gfx1033" ;;
        100304) echo "gfx1034" ;;
        100305) echo "gfx1035" ;;
        100306) echo "gfx1036" ;;
        110000) echo "gfx1100" ;;
        110001) echo "gfx1101" ;;
        110002) echo "gfx1102" ;;
        110003) echo "gfx1103" ;;
        110500) echo "gfx1150" ;;
        110501) echo "gfx1151" ;;
        110502) echo "gfx1152" ;;
        120000) echo "gfx1200" ;;
        120001) echo "gfx1201" ;;
        0)      echo "" ;;   # CPU host node
        *)      echo "gfx_unknown_$1" ;;
    esac
}

# ── Device probe: sysfs primary ─────────────────────────────────────────
# Read /sys/class/kfd/kfd/topology/nodes/*/properties for each node.
# CPU host nodes have simd_count==0 — filter those out. Remaining nodes
# in node-id order map 1:1 to HIP device indices in encounter order.
#
# Output format: one "idx:gfx_str" per line (e.g. "0:gfx906\n1:gfx90c").
# Empty output means no GPUs found (or sysfs unavailable).
probe_sysfs() {
    [ "${HIPFIRE_PP_GATE_TEST_NO_SYSFS:-0}" = "1" ] && return 1
    [ -d /sys/class/kfd/kfd/topology/nodes ] || return 1
    local idx=0
    local found=0
    # Iterate in numeric node-id order. The glob expands in shell order
    # (lexical on most systems); for ≤ ~16 nodes this matches numeric order.
    for nodedir in /sys/class/kfd/kfd/topology/nodes/*/; do
        local props="$nodedir/properties"
        [ -r "$props" ] || continue
        local simd gfx
        simd=$(awk '/^simd_count/ {print $2; exit}' "$props" 2>/dev/null)
        gfx=$(awk '/^gfx_target_version/ {print $2; exit}' "$props" 2>/dev/null)
        # CPU host: simd_count=0 or gfx_target_version=0
        [ -z "$simd" ] && continue
        [ "$simd" = "0" ] && continue
        [ -z "$gfx" ] && continue
        [ "$gfx" = "0" ] && continue
        local gfx_str
        gfx_str=$(gfx_int_to_str "$gfx")
        [ -z "$gfx_str" ] && continue
        printf '%d:%s\n' "$idx" "$gfx_str"
        idx=$((idx + 1))
        found=1
    done
    [ "$found" -eq 1 ] || return 1
    return 0
}

# ── Device probe: rocm-smi fallback ─────────────────────────────────────
# Used only when sysfs is unavailable. rocm-smi --showhw produces a
# tabular layout; header has embedded-space column names (GFX VER, GFX RAS)
# but data rows are whitespace-stable.
probe_rocm_smi() {
    command -v rocm-smi >/dev/null 2>&1 || return 1
    local raw
    raw=$(rocm-smi --showhw 2>/dev/null) || return 1
    [ -z "$raw" ] && return 1
    # Match data rows: idx node did(0x...) guid gfxVERR ...
    # Capture group 1 = GPU index, group 2 = gfx string
    local got=0
    while IFS= read -r line; do
        if [[ $line =~ ^([0-9]+)[[:space:]]+[0-9]+[[:space:]]+0x[0-9a-f]+[[:space:]]+[0-9]+[[:space:]]+(gfx[0-9a-z]+)[[:space:]] ]]; then
            printf '%d:%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
            got=1
        fi
    done <<< "$raw"
    [ "$got" -eq 1 ] || return 1
    return 0
}

# ── Device probe: --simulate (testing) ──────────────────────────────────
# --simulate is dry-run-only — using it without --dry-run is rejected at
# arg-parse time. Each entry MUST match idx:gfxNNNN[NNN] strictly.
probe_simulate() {
    [ -n "$SIMULATE" ] || return 1
    local entry
    IFS=',' read -ra entries <<< "$SIMULATE"
    for entry in "${entries[@]}"; do
        if [[ ! "$entry" =~ ^[0-9]+:gfx[0-9a-z]+$ ]]; then
            echo "pp-gate: --simulate entry '$entry' must match idx:gfxNNNN (e.g. 0:gfx1100)" >&2
            return 2
        fi
        printf '%s\n' "$entry"
    done
    return 0
}

# ── Top-level probe with fallback chain ─────────────────────────────────
probe_source=""
probe_devices=""

if [ -n "$SIMULATE" ]; then
    if ! probe_devices=$(probe_simulate); then
        exit 2
    fi
    probe_source="simulate"
elif probe_devices=$(probe_sysfs); then
    probe_source="sysfs"
elif [ "${HIPFIRE_PP_GATE_REQUIRE_SYSFS:-0}" = "1" ]; then
    echo "pp-gate: sysfs topology unavailable and HIPFIRE_PP_GATE_REQUIRE_SYSFS=1 — failing" >&2
    exit 2
elif probe_devices=$(probe_rocm_smi); then
    probe_source="rocm-smi"
    echo "pp-gate: WARNING — sysfs topology unavailable, falling back to rocm-smi." >&2
    echo "                   Run with --dry-run to see the filter pipeline trace." >&2
    echo "                   Set HIPFIRE_PP_GATE_REQUIRE_SYSFS=1 to detect future sysfs loss as a hard error." >&2
else
    # No GPUs found at all. Preserve the old HIP_VISIBLE_DEVICES fallback
    # so CI / no-GPU containers exit cleanly.
    probe_source="none"
fi

# ── Filter pipeline ─────────────────────────────────────────────────────
# Input: probe_devices ("0:gfx906\n1:gfx90c"). Output: filtered_indices
# ("0,1") + filter_log (human-readable trace for --dry-run / diagnostics).
filter_log=""
log_filter() { filter_log="${filter_log}${1}\n"; }
WOULD_SKIP_REASON=""

# Build associative arrays: idx → gfx_str, and ordered idx list
ALL_IDX=()
declare -A IDX_ARCH
if [ -n "$probe_devices" ]; then
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        dev_idx="${line%%:*}"
        dev_arch="${line#*:}"
        ALL_IDX+=("$dev_idx")
        IDX_ARCH["$dev_idx"]="$dev_arch"
    done <<< "$probe_devices"
fi

# Stage 1: PP_GATE_DEVICES wins precedence
FILTERED_IDX=()
igpu_removed_count=0

if [ -n "${PP_GATE_DEVICES:-}" ]; then
    log_filter "  stage 1: PP_GATE_DEVICES='${PP_GATE_DEVICES}' — validating indices"
    IFS=',' read -ra requested <<< "$PP_GATE_DEVICES"
    for req in "${requested[@]}"; do
        if [ -z "${IDX_ARCH[$req]:-}" ]; then
            echo "pp-gate: PP_GATE_DEVICES=$PP_GATE_DEVICES includes index $req, but only ${#ALL_IDX[@]} device(s) are visible." >&2
            echo "         Visible indices: ${ALL_IDX[*]:-(none)}" >&2
            exit 2
        fi
        FILTERED_IDX+=("$req")
    done
    log_filter "  stage 2: iGPU filter — SKIPPED (PP_GATE_DEVICES wins)"
    log_filter "  stage 3: homogeneity check — SKIPPED (PP_GATE_DEVICES wins)"
else
    log_filter "  stage 1: PP_GATE_DEVICES unset — using all visible devices"

    # Stage 2: iGPU filter
    if [ "${HIPFIRE_PP_GATE_INCLUDE_IGPU:-0}" = "1" ]; then
        if [ "${#ALL_IDX[@]}" -gt 0 ]; then
            FILTERED_IDX=("${ALL_IDX[@]}")
        fi
        log_filter "  stage 2: iGPU filter — SKIPPED (HIPFIRE_PP_GATE_INCLUDE_IGPU=1)"
    else
        removed=""
        for idx in "${ALL_IDX[@]:-}"; do
            [ -z "$idx" ] && continue
            arch="${IDX_ARCH[$idx]}"
            is_igpu=0
            for ig in "${IGPU_ARCHS[@]}"; do
                if [ "$arch" = "$ig" ]; then is_igpu=1; break; fi
            done
            if [ "$is_igpu" = "1" ]; then
                removed="${removed} ${idx}:${arch}"
                igpu_removed_count=$((igpu_removed_count + 1))
            else
                FILTERED_IDX+=("$idx")
            fi
        done
        if [ -n "$removed" ]; then
            log_filter "  stage 2: iGPU filter — removed${removed}"
        else
            log_filter "  stage 2: iGPU filter — no iGPUs found"
        fi
    fi

    # Stage 3: homogeneity check
    if [ "${#FILTERED_IDX[@]}" -ge 2 ]; then
        first_family=$(arch_family "${IDX_ARCH[${FILTERED_IDX[0]}]}")
        hetero=0
        families_seen="$first_family"
        for idx in "${FILTERED_IDX[@]:1}"; do
            fam=$(arch_family "${IDX_ARCH[$idx]}")
            if [ "$fam" != "$first_family" ]; then
                hetero=1
                families_seen="$families_seen, $fam"
            fi
        done
        if [ "$hetero" = "1" ]; then
            if [ "${HIPFIRE_PP_GATE_HETEROGENEOUS:-0}" = "1" ]; then
                log_filter "  stage 3: homogeneity check — heterogeneous ($families_seen) but HIPFIRE_PP_GATE_HETEROGENEOUS=1, running anyway"
            else
                if [ "$DRY_RUN" = "1" ]; then
                    log_filter "  stage 3: homogeneity check — heterogeneous ($families_seen), would skip"
                    WOULD_SKIP_REASON="heterogeneous ISA families ($families_seen)"
                else
                    echo "pp-gate: heterogeneous ISA families ($families_seen) — skipping."
                    echo "         Kernel cache is keyed per-family; pp>1 across mismatched families isn't supported."
                    echo "         Hints:"
                    echo "           - pin a homogeneous pair via PP_GATE_DEVICES=0,2"
                    echo "           - or set HIPFIRE_PP_GATE_HETEROGENEOUS=1 to force-run"
                    exit 0
                fi
            fi
        else
            log_filter "  stage 3: homogeneity check — family $first_family, OK"
        fi
    else
        log_filter "  stage 3: homogeneity check — n/a (<2 devices remaining)"
    fi
fi

# Compose filtered indices comma-list
filtered_csv=$(IFS=,; echo "${FILTERED_IDX[*]:-}")

# ── Dry-run output ──────────────────────────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
    echo "pp-gate: dry-run mode"
    if [ -n "$SIMULATE" ]; then
        echo "  source: simulate ('$SIMULATE')"
    else
        echo "  source: $probe_source"
    fi
    if [ -n "$probe_devices" ]; then
        parsed_str=$(echo "$probe_devices" | tr '\n' ' ')
        echo "  parsed: $parsed_str"
    else
        echo "  parsed: (no GPUs detected)"
    fi
    echo "  filter pipeline:"
    printf '%b' "$filter_log"
    echo "  decision: ${#FILTERED_IDX[@]} GPU(s) usable"
    if [ -n "$WOULD_SKIP_REASON" ]; then
        echo "  would skip ($WOULD_SKIP_REASON)"
        echo "  HIP_VISIBLE_DEVICES: (not exported)"
    elif [ "${#FILTERED_IDX[@]}" -lt 2 ]; then
        echo "  would skip (need ≥2 usable)"
        echo "  HIP_VISIBLE_DEVICES: (not exported)"
    else
        echo "  would run gate"
        echo "  HIP_VISIBLE_DEVICES: $filtered_csv"
    fi
    exit 0
fi

# ── Final count gate (preserves old <2 GPU skip semantics) ──────────────
gpu_count=${#FILTERED_IDX[@]}

# Last-resort fallback: HIP_VISIBLE_DEVICES parsing when no probe worked.
# Keeps CI / containers with no rocm-smi + no sysfs functional.
if [ "$gpu_count" -eq 0 ] && [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    fallback_count=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    if [ "$fallback_count" -ge 2 ]; then
        echo "pp-gate: WARNING — no sysfs/rocm-smi info; using HIP_VISIBLE_DEVICES='$HIP_VISIBLE_DEVICES'" >&2
        echo "                   iGPU + homogeneity filtering NOT applied." >&2
        filtered_csv="$HIP_VISIBLE_DEVICES"
        gpu_count="$fallback_count"
    fi
fi

if [ "$gpu_count" -lt 2 ]; then
    if [ "$gpu_count" -eq 0 ] && [ "$igpu_removed_count" -gt 0 ]; then
        echo "pp-gate: all visible GPUs are iGPUs ($igpu_removed_count device(s) filtered);"
        echo "         set HIPFIRE_PP_GATE_INCLUDE_IGPU=1 to test on iGPUs anyway."
    elif [ "$igpu_removed_count" -gt 0 ]; then
        echo "pp-gate: $igpu_removed_count iGPU(s) filtered, only $gpu_count usable GPU(s) — skipping"
        echo "         set HIPFIRE_PP_GATE_INCLUDE_IGPU=1 to include iGPUs"
    else
        echo "pp-gate: only $gpu_count usable GPU(s) — skipping"
    fi
    exit 0
fi

# Export for the test runs. Replaces the old hardcoded `0,1` default.
export HIP_VISIBLE_DEVICES="$filtered_csv"
export HIPFIRE_DETERMINISTIC=1

EXE="./target/release/examples/daemon"
EXAMPLES_DIR="./target/release/examples"
MODEL="${HIPFIRE_PP_GATE_MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

if [ ! -f "$MODEL" ]; then
    echo "pp-gate: model not found at $MODEL — skipping"
    echo "         set HIPFIRE_PP_GATE_MODEL or install qwen3.5-0.8b.mq4"
    exit 0
fi

# ── Rebuild ──────────────────────────────────────────────────────────────
rebuild=0
if [ ! -x "$EXE" ] || [ ! -x "$EXAMPLES_DIR/pp_parity_chatml" ]; then
    rebuild=1
else
    # Post-0.1.20 modular topology: forward-pass + multi-GPU sources are
    # split across hipfire-runtime (KvCache, Gpus, daemon) and
    # hipfire-arch-qwen35 (qwen35 forward, prefill batch). Watch both.
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs \
               crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/multi_gpu.rs \
               crates/hipfire-runtime/examples/daemon.rs \
               crates/hipfire-runtime/examples/pp_parity_chatml.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then rebuild=1; break; fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "pp-gate: rebuilding..."
    if ! cargo build --release --features deltanet -p hipfire-runtime \
            --example daemon --example pp_parity_chatml >&2; then
        echo "pp-gate: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ─────────────────────────────────────────────────────────────
# gpu_acquire is reentrant: if a parent gate already holds the lock it
# no-ops (HIPFIRE_GPU_LOCK_OWNER), and gpu_release in this child won't
# release the ancestor's lock. So acquire unconditionally.
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "pp-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

fail=0
say() { printf '\n── %s ──\n' "$1"; }

# ── 1. pp_parity_chatml ─────────────────────────────────────────────────
say "pp_parity_chatml (per-token forward bit-equivalence, 50 decode tokens)"
if "$EXAMPLES_DIR/pp_parity_chatml" "$MODEL" 2>&1 | tee /tmp/pp-gate-parity.log | \
   grep -qE 'ALL [0-9]+ tokens identical'; then
    echo "PASS"
else
    echo "FAIL — see /tmp/pp-gate-parity.log"
    fail=1
fi

if [ "$SKIP_E2E" -eq 1 ]; then
    [ "$fail" -ne 0 ] && exit 1
    echo
    echo "pp-gate: parity-only mode passed."
    exit 0
fi

# ── 2. daemon pp=1 vs pp=2 byte-equivalence ─────────────────────────────
# SHA prefix of empty string — used to detect "daemon emitted no text events".
EMPTY_SHA="e3b0c44298fc1c14"

# Run the daemon for one pp value, return: "<count> <sha16> <stderr_logfile>"
# count = number of "text" events on stdout. sha16 = sha256[:16] of joined text.
gen_summary() {
    local pp_arg="$1"
    local params='{"max_seq":2048}'
    [ "$pp_arg" = "2" ] && params='{"max_seq":2048,"pp":2}'
    local logfile
    logfile=$(mktemp -t "pp-gate-pp${pp_arg}.XXXXXX.log")
    local result
    result=$(
        (printf '%s\n' \
            '{"type":"load","model":"'"$MODEL"'","params":'"$params"'}' \
            '{"type":"generate","id":"r1","prompt":"Write a one-sentence greeting.","temperature":0.0,"max_tokens":40}' \
            '{"type":"unload"}'
        ) | "$EXE" 2>"$logfile" \
          | grep '"text"' \
          | python3 -c '
import sys, json, hashlib
toks = []
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        toks.append(obj.get("text", ""))
    except Exception:
        pass
joined = "".join(toks)
print(f"{len(toks)} {hashlib.sha256(joined.encode()).hexdigest()[:16]}")
'
    )
    # Validate result format: "<count> <sha16>". Empty or malformed means
    # the python interpreter crashed, the heredoc broke, etc. — surface
    # that explicitly rather than letting awk-parsing return garbage.
    if [[ ! "$result" =~ ^[0-9]+\ [0-9a-f]{16}$ ]]; then
        echo "0 ${EMPTY_SHA} $logfile"
        echo "pp-gate: gen_summary(pp=$pp_arg) produced malformed result '$result'" >&2
        echo "         daemon stderr → $logfile" >&2
        return
    fi
    echo "$result $logfile"
}

say "daemon pp=1 vs pp=2 byte-identical (greedy, ChatML, HIPFIRE_DETERMINISTIC=1)"

PP1_RESULT=$(gen_summary 1)
PP2_RESULT=$(gen_summary 2)
PP1_COUNT=$(echo "$PP1_RESULT" | awk '{print $1}')
PP1_SHA=$(echo "$PP1_RESULT" | awk '{print $2}')
PP1_LOG=$(echo "$PP1_RESULT" | awk '{print $3}')
PP2_COUNT=$(echo "$PP2_RESULT" | awk '{print $1}')
PP2_SHA=$(echo "$PP2_RESULT" | awk '{print $2}')
PP2_LOG=$(echo "$PP2_RESULT" | awk '{print $3}')

echo "pp=1: count=$PP1_COUNT sha=$PP1_SHA log=$PP1_LOG"
echo "pp=2: count=$PP2_COUNT sha=$PP2_SHA log=$PP2_LOG"

# Detect load/dispatch failure on EITHER run (count==0 or empty-SHA).
# This must precede the equality check — two empty runs are NOT a PASS.
zero_run=""
if [ "$PP1_COUNT" = "0" ] || [ "$PP1_SHA" = "$EMPTY_SHA" ]; then
    zero_run="${zero_run}pp=1 "
fi
if [ "$PP2_COUNT" = "0" ] || [ "$PP2_SHA" = "$EMPTY_SHA" ]; then
    zero_run="${zero_run}pp=2 "
fi
zero_run="${zero_run% }"

if [ -n "$zero_run" ]; then
    echo "FAIL"
    echo "pp-gate: $zero_run emitted 0 text events — daemon load or dispatch failed."
    echo "         Hints:"
    echo "           - check the per-PID stderr logs:"
    echo "               $PP1_LOG"
    echo "               $PP2_LOG"
    echo "           - PP_GATE_DEVICES=0 to pin to one device (gate falls through the <2-GPU skip)"
    echo "           - HIPFIRE_PP_GATE_INCLUDE_IGPU=1 if you intend to test iGPU dispatch deliberately"
    echo "           - common causes: iGPU dispatch, OOM, ISA mismatch, missing model"
    fail=1
elif [ "$PP1_COUNT" != "$PP2_COUNT" ]; then
    echo "FAIL — token count mismatch (pp=1: $PP1_COUNT, pp=2: $PP2_COUNT)"
    echo "       this is a real parity divergence; see $PP1_LOG and $PP2_LOG"
    fail=1
elif [ "$PP1_SHA" != "$PP2_SHA" ]; then
    echo "FAIL — pp=2 ≢ pp=1 byte-identical (same token count, different content)"
    echo "       this is the genuine multi-GPU parity bug the gate was built for"
    fail=1
else
    echo "PASS"
    # Clean up successful-run logs to avoid /tmp growth.
    rm -f "$PP1_LOG" "$PP2_LOG" 2>/dev/null || true
fi

# ── 3. refusal contracts ─────────────────────────────────────────────────
# Capture stdout + stderr; older daemons emit the refusal on stdout, newer
# versions may move it to stderr.
say "refusal: DFlash + pp=2 must error at load"
DFLASH_REFUSAL=$( (printf '%s\n' \
    '{"type":"load","model":"'"$MODEL"'","params":{"max_seq":2048,"pp":2,"draft":"/nonexistent.hfq"}}'
) | "$EXE" 2>&1 | grep -c 'DFlash speculative decode requires pp=1' || true)
if [ "$DFLASH_REFUSAL" -ge 1 ]; then echo "PASS"; else echo "FAIL"; fail=1; fi

say "refusal: CASK + pp=2 must error at load"
CASK_REFUSAL=$( (printf '%s\n' \
    '{"type":"load","model":"'"$MODEL"'","params":{"max_seq":2048,"pp":2,"cask_sidecar":"/nonexistent.bin"}}'
) | "$EXE" 2>&1 | grep -c 'CASK / TriAttention eviction requires pp=1' || true)
if [ "$CASK_REFUSAL" -ge 1 ]; then echo "PASS"; else echo "FAIL"; fail=1; fi

echo
if [ "$fail" -ne 0 ]; then
    echo "pp-gate: FAIL"
    exit 1
fi
echo "pp-gate: PASS"
exit 0
