#!/usr/bin/env bash
# run-ep-parity.sh — emulated-EP2 parity runner (STEP-002 Task 8, Phase 2B).
#
# One explicit mode enum: exactly one of
#   --probe   NON-ACCEPTANCE: reports observed deltas + digests, exits 0.
#   --accept  ACCEPTANCE: refuses to run until ALL Phase 3 pins below are
#             filled (model SHA-256, binary SHA-256, max-logit-delta) and
#             the actual digests match them.  The max-logit-delta is a
#             PINNED CONSTANT in this script — never a runtime argument.
#
# Self-test (no GPU, no model):
#   bash .agent-progress/run-ep-parity.sh --self-test
#
# Exit codes: 0 = probe report OR acceptance pass; 1 = acceptance fail;
# 2 = configuration/pinning/usage error; 3 = GPU lock failure; 4 = build failure.
set -u

# Repo-relative root discovery — no hardcoded worktree path.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.." || exit 2

# ── Phase 3 pins ────────────────────────────────────────────────────
# The binary digest and maximum delta below were pinned from five fresh-process
# probes of the exact canonical scenario. The delta is the next representable
# f32 above the maximum observed value (0.7081642).
PINNED_MODEL_SHA256="1dc1c7964de415e0040a540a4300b9518e11b00c13d99c23f576f2b9fe1e8bca"
PINNED_BINARY_SHA256="f4b82b109e779f8518332dd86e31371e9a46a5cf50c0ec87360d3a75c95dbd6f"
PINNED_MAX_LOGIT_DELTA="0.708164275"
# Pinned from the canonical fixture bytes; update only together with the file.
PINNED_PROMPT_MD5="1aacd3c05cf9695cc799acc59581938d"

# ── Canonical acceptance scenario (pinned constants) ──────────────────
# ACCEPTANCE runs EXACTLY this scenario and nothing else: 16 decode steps,
# the canonical second-turn suffix (byte-exact), Q8 KV.  --steps/--suffix/
# --kv-mode overrides are REFUSED in acceptance mode; probe may override.
# The model path may be overridden only because acceptance enforces the
# pinned model SHA-256 equality.
PINNED_STEPS=16
PINNED_KV_MODE=q8
PINNED_SUFFIX_BYTES=32  # byte length of the canonical SUFFIX below

MODEL=${EP2_PARITY_MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4}
PROMPT_FILE=benchmarks/prompts/qwen35_moe_ep_parity.txt
STEPS=$PINNED_STEPS
KV_MODE=$PINNED_KV_MODE
SUFFIX=$'\n\nWhat follows is a second turn.'

require_acceptance_pins() {
    local missing=()
    [[ -n "$PINNED_MODEL_SHA256" ]] || missing+=("PINNED_MODEL_SHA256")
    [[ -n "$PINNED_BINARY_SHA256" ]] || missing+=("PINNED_BINARY_SHA256")
    [[ -n "$PINNED_MAX_LOGIT_DELTA" ]] || missing+=("PINNED_MAX_LOGIT_DELTA")
    if ((${#missing[@]} > 0)); then
        echo "run-ep-parity: ACCEPTANCE IMPOSSIBLE — Phase 3 pins empty: ${missing[*]}" >&2
        return 2
    fi
    if [[ ! "$PINNED_MAX_LOGIT_DELTA" =~ ^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]; then
        echo "run-ep-parity: PINNED_MAX_LOGIT_DELTA is not a finite positive number: '$PINNED_MAX_LOGIT_DELTA'" >&2
        return 2
    fi
}

# Fail-only internal hook used by --self-test. It exercises the real pin
# validator after source constants are populated, without touching a model or
# GPU. It cannot bypass acceptance; it only forces refusal.
if [[ ${EP2_RUNNER_SELFTEST_UNPIN:-0} == 1 ]]; then
    PINNED_MODEL_SHA256=""
    PINNED_BINARY_SHA256=""
    PINNED_MAX_LOGIT_DELTA=""
    require_acceptance_pins
    exit $?
fi

# ── Idempotent cleanup + terminating signal traps ─────────────────────
# `install_cleanup_traps <fn>` runs <fn> EXACTLY ONCE on any exit path
# (normal exit, error exit, or signal).  INT/TERM run the cleanup and then
# terminate with 130/143 (the conventional signal exit codes); the EXIT
# trap that follows the signal handler observes the idempotence flag and
# does not run the cleanup a second time.
CLEANUP_FN=:
CLEANUP_DONE=0
install_cleanup_traps() {
    CLEANUP_FN=$1
    CLEANUP_DONE=0
    # The EXIT trap preserves the pending exit status even when the cleanup
    # function returns non-zero (a parity verdict is never masked by a
    # secondary release failure — the failure is still logged).
    trap 'rc=$?; if ((CLEANUP_DONE == 0)); then CLEANUP_DONE=1; "$CLEANUP_FN"; fi; exit "$rc"' EXIT
    trap 'if ((CLEANUP_DONE == 0)); then CLEANUP_DONE=1; "$CLEANUP_FN"; fi; exit 130' INT
    trap 'if ((CLEANUP_DONE == 0)); then CLEANUP_DONE=1; "$CLEANUP_FN"; fi; exit 143' TERM
}

# ── Argument parsing: one explicit mode enum ─────────────────────────
MODE=""  # "", "probe", "accept"
MODE_ARGS=()
# Set when an acceptance-canonical parameter is overridden on the command
# line.  Probe may override; acceptance REFUSES any override (the pinned
# canonical scenario must run byte-exact).
ACCEPT_OVERRIDE=0

while (($# > 0)); do
    case "$1" in
        --probe)
            if [[ -n "$MODE" ]]; then
                echo "run-ep-parity: conflicting/repeated mode: '--probe' after mode '$MODE'" >&2
                exit 2
            fi
            MODE=probe
            ;;
        --accept)
            if [[ -n "$MODE" ]]; then
                echo "run-ep-parity: conflicting/repeated mode: '--accept' after mode '$MODE'" >&2
                exit 2
            fi
            MODE=accept
            ;;
        --steps)
            if (($# < 2)); then
                echo "run-ep-parity: --steps requires a value" >&2
                exit 2
            fi
            shift
            STEPS="$1"
            ACCEPT_OVERRIDE=1
            if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || ((STEPS == 0)); then
                echo "run-ep-parity: --steps must be a positive integer, got '$STEPS'" >&2
                exit 2
            fi
            ;;
        --model)
            if (($# < 2)); then
                echo "run-ep-parity: --model requires a value" >&2
                exit 2
            fi
            shift
            MODEL="$1"
            ;;
        --kv-mode)
            if (($# < 2)); then
                echo "run-ep-parity: --kv-mode requires a value" >&2
                exit 2
            fi
            shift
            KV_MODE="$1"
            ACCEPT_OVERRIDE=1
            ;;
        --suffix)
            if (($# < 2)); then
                echo "run-ep-parity: --suffix requires a value" >&2
                exit 2
            fi
            shift
            SUFFIX="$1"
            ACCEPT_OVERRIDE=1
            ;;
        --self-test)
            if [[ -n "$MODE" ]]; then
                echo "run-ep-parity: --self-test is not a mode and cannot be combined with --probe/--accept" >&2
                exit 2
            fi
            MODE=self-test
            ;;
        *)
            echo "run-ep-parity: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$MODE" == "self-test" ]]; then
    # ── Lightweight self-test: mode enum, pins, canonical-scenario
    # refusal, terminating-signal traps (no GPU/model). ──
    failures=0
    expect_exit() { # expect_exit <label> <want> <argv...>
        local label=$1 want=$2
        shift 2
        local out got
        out=$("$@" 2>&1); got=$?
        if [[ $got -ne $want ]]; then
            echo "runner self-test FAIL: $label (exit $got, want $want)" >&2
            echo "$out" >&2
            failures=$((failures + 1))
        else
            echo "runner self-test ok: $label"
        fi
    }
    expect_exit_msg() { # expect_exit_msg <label> <want> <needle> <argv...>
        local label=$1 want=$2 needle=$3
        shift 3
        local out got
        out=$("$@" 2>&1); got=$?
        if [[ $got -ne $want ]]; then
            echo "runner self-test FAIL: $label (exit $got, want $want)" >&2
            echo "$out" >&2
            failures=$((failures + 1))
        elif ! grep -qF "$needle" <<<"$out"; then
            echo "runner self-test FAIL: $label (missing message '$needle')" >&2
            echo "$out" >&2
            failures=$((failures + 1))
        else
            echo "runner self-test ok: $label"
        fi
    }
    SELF="$SCRIPT_DIR/run-ep-parity.sh"
    expect_exit "absent mode" 2 bash "$SELF"
    expect_exit "repeated --probe" 2 bash "$SELF" --probe --probe
    expect_exit "repeated --accept" 2 bash "$SELF" --accept --accept
    expect_exit "conflicting --probe --accept" 2 bash "$SELF" --probe --accept
    expect_exit "conflicting --accept --probe" 2 bash "$SELF" --accept --probe
    expect_exit "--accept takes no runtime delta" 2 bash "$SELF" --accept 0.5
    expect_exit "unknown argument" 2 bash "$SELF" --bogus
    expect_exit "missing --steps value" 2 bash "$SELF" --probe --steps
    expect_exit "zero steps" 2 bash "$SELF" --probe --steps 0
    # Exercise the pin validator directly with blanked values. This remains a
    # no-GPU test after Phase 3 fills the source constants above.
    expect_exit_msg "unpinned acceptance refusal" 2 "ACCEPTANCE IMPOSSIBLE" \
        env EP2_RUNNER_SELFTEST_UNPIN=1 bash "$SELF"
    # ── Canonical-scenario override refusal (acceptance) ────────────────
    # Acceptance must run the pinned scenario byte-exact: any
    # --steps/--suffix/--kv-mode override is refused, in either argument
    # order.  Probe may override (verified below by reaching the model check).
    expect_exit_msg "acceptance refuses --steps override" 2 "acceptance refuses" \
        bash "$SELF" --accept --steps 8
    expect_exit_msg "acceptance refuses --steps override (reversed order)" 2 "acceptance refuses" \
        bash "$SELF" --steps 8 --accept
    expect_exit_msg "acceptance refuses --suffix override" 2 "acceptance refuses" \
        bash "$SELF" --accept --suffix "x"
    expect_exit_msg "acceptance refuses --kv-mode override" 2 "acceptance refuses" \
        bash "$SELF" --accept --kv-mode fp16
    # Probe overrides are allowed: the run proceeds past the override gate
    # and fails later at the (explicitly nonexistent) model check instead.
    expect_exit_msg "probe allows --steps override" 2 "model not found" \
        bash "$SELF" --probe --steps 8 --model /nonexistent/ep2-parity-model.mq4
    expect_exit_msg "probe allows --kv-mode override" 2 "model not found" \
        bash "$SELF" --probe --kv-mode fp16 --model /nonexistent/ep2-parity-model.mq4
    # ── Canonical suffix pin sanity ─────────────────────────────────────
    canonical_bytes=$(printf '%s' "$SUFFIX" | wc -c)
    if [[ "$canonical_bytes" != "$PINNED_SUFFIX_BYTES" ]]; then
        echo "runner self-test FAIL: canonical suffix bytes $canonical_bytes != pinned $PINNED_SUFFIX_BYTES" >&2
        failures=$((failures + 1))
    else
        echo "runner self-test ok: canonical suffix pin matches the default suffix"
    fi
    # ── Terminating-signal traps: cleanup exactly once; INT=130, TERM=143 ─
    trap_signal_test() { # trap_signal_test <signal> <want_exit>
        local signal=$1 want=$2 log
        log=$(mktemp) || return 1
        local rc count
        (
            cleanup_probe() { printf 'x' >> "$log"; }
            install_cleanup_traps cleanup_probe
            kill -"$signal" "$BASHPID"
            exit 99 # unreachable: the signal handler must terminate
        ) 2>/dev/null
        rc=$?
        count=$(wc -c < "$log" 2>/dev/null || echo 0)
        rm -f "$log"
        [[ $rc -eq $want && $count -eq 1 ]]
    }
    if trap_signal_test INT 130 && trap_signal_test TERM 143; then
        echo "runner self-test ok: INT/TERM traps release exactly once and exit 130/143"
    else
        echo "runner self-test FAIL: INT/TERM traps must release exactly once and exit 130/143" >&2
        failures=$((failures + 1))
    fi
    # A failing cleanup must not mask the pending exit status (the parity
    # verdict is never replaced by a secondary release failure).
    trap_fail_test() {
        local rc
        (
            cleanup_fail() { echo "cleanup failed" >&2; return 1; }
            install_cleanup_traps cleanup_fail
            exit 7
        ) 2>/dev/null
        rc=$?
        [[ $rc -eq 7 ]]
    }
    if trap_fail_test; then
        echo "runner self-test ok: failing cleanup does not mask the pending exit status"
    else
        echo "runner self-test FAIL: failing cleanup masks the pending exit status" >&2
        failures=$((failures + 1))
    fi
    # Prompt MD5 pin must match the committed bytes.
    actual_md5=$(md5sum "$PROMPT_FILE" 2>/dev/null | cut -d' ' -f1)
    if [[ "$actual_md5" != "$PINNED_PROMPT_MD5" ]]; then
        echo "runner self-test FAIL: prompt MD5 pin mismatch ($actual_md5 != $PINNED_PROMPT_MD5)" >&2
        failures=$((failures + 1))
    else
        echo "runner self-test ok: prompt MD5 pin matches committed bytes"
    fi
    if ((failures > 0)); then
        echo "runner self-test FAILED: $failures assertion(s)" >&2
        exit 1
    fi
    echo "runner self-test passed"
    exit 0
fi

# Exactly one mode is mandatory.
if [[ -z "$MODE" ]]; then
    echo "run-ep-parity: exactly one mode required: --probe or --accept" >&2
    exit 2
fi

# ── Acceptance refuses canonical-scenario overrides ───────────────────
# The pinned acceptance scenario is byte-exact: steps=$PINNED_STEPS, the
# canonical suffix, Q8 KV.  Probe may override; acceptance must not.  The
# model path may be overridden only because acceptance enforces the pinned
# model SHA-256 equality.
if [[ "$MODE" == "accept" ]]; then
    if ((ACCEPT_OVERRIDE == 1)); then
        echo "run-ep-parity: acceptance refuses --steps/--suffix/--kv-mode overrides" >&2
        echo "run-ep-parity: canonical acceptance scenario: steps=$PINNED_STEPS, suffix=$PINNED_SUFFIX_BYTES bytes, kv_mode=$PINNED_KV_MODE" >&2
        exit 2
    fi
    if [[ "$STEPS" != "$PINNED_STEPS" || "$KV_MODE" != "$PINNED_KV_MODE" ]]; then
        echo "run-ep-parity: acceptance scenario drift (steps=$STEPS kv_mode=$KV_MODE) — must equal the pins" >&2
        exit 2
    fi
    ACTUAL_SUFFIX_BYTES=$(printf '%s' "$SUFFIX" | wc -c)
    if [[ "$ACTUAL_SUFFIX_BYTES" != "$PINNED_SUFFIX_BYTES" ]]; then
        echo "run-ep-parity: acceptance suffix byte length $ACTUAL_SUFFIX_BYTES != pinned $PINNED_SUFFIX_BYTES" >&2
        exit 2
    fi
fi

# ── Prompt file + pinned MD5 (verified for BOTH modes) ───────────────
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "run-ep-parity: prompt file missing: $PROMPT_FILE" >&2
    exit 2
fi
PROMPT_MD5=$(md5sum "$PROMPT_FILE" | cut -d' ' -f1)
if [[ "$PROMPT_MD5" != "$PINNED_PROMPT_MD5" ]]; then
    echo "run-ep-parity: prompt MD5 mismatch: got $PROMPT_MD5, pinned $PINNED_PROMPT_MD5" >&2
    exit 2
fi

# ── Acceptance preconditions: every Phase 3 pin must be filled ───────
if [[ "$MODE" == "accept" ]]; then
    require_acceptance_pins || exit $?
fi

if [[ ! -f "$MODEL" ]]; then
    echo "run-ep-parity: model not found: $MODEL (set EP2_PARITY_MODEL or pass --model)" >&2
    exit 2
fi

LOG=.agent-progress/ep-parity.log; : > "$LOG"; exec 3>&2; exec >>"$LOG" 2>&1
echo "== emulated-EP2 parity start $(date -Is) HEAD $(git rev-parse --short HEAD) mode=$MODE =="

# ── Digests ──────────────────────────────────────────────────────────
MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
ACTUAL_SUFFIX_BYTES=$(printf '%s' "$SUFFIX" | wc -c)
echo "model_path:   $MODEL"
echo "model_sha256: $MODEL_SHA"
echo "prompt_md5:   $PROMPT_MD5 (pinned $PINNED_PROMPT_MD5)"
echo "prompt_file:  $PROMPT_FILE"
echo "canonical_steps:        $PINNED_STEPS (pinned; actual $STEPS)"
echo "canonical_kv_mode:      $PINNED_KV_MODE (pinned; actual $KV_MODE)"
echo "canonical_suffix_bytes: $PINNED_SUFFIX_BYTES (pinned; actual $ACTUAL_SUFFIX_BYTES)"

# ── Physical GPU identity / gfx arch / ROCm-HIP evidence ─────────────
# Probe may print `unavailable` only when the tooling is truly absent;
# acceptance REFUSES missing identity/arch/version evidence.
GPU_IDENTITY="unavailable"
GFX_ARCH="unavailable"
ROCM_VERSION="unavailable"
HIP_VERSION="unavailable"
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_IDENTITY=$(rocm-smi --showproductname 2>/dev/null | grep -iE "card|gpu" | head -3 | tr '\n' ';' | sed 's/;$//')
    [[ -n "$GPU_IDENTITY" ]] || GPU_IDENTITY="unavailable"
fi
if command -v rocm_agent_enumerator >/dev/null 2>&1; then
    GFX_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -E "^gfx" | head -1)
    [[ -n "$GFX_ARCH" ]] || GFX_ARCH="unavailable"
fi
if [[ -r /opt/rocm/.info/version ]]; then
    ROCM_VERSION=$(tr -d '\n' < /opt/rocm/.info/version 2>/dev/null)
    [[ -n "$ROCM_VERSION" ]] || ROCM_VERSION="unavailable"
elif command -v rocm-smi >/dev/null 2>&1; then
    ROCM_VERSION=$(rocm-smi --version 2>/dev/null | grep -iE "rocm" | head -1 | tr -s ' ')
    [[ -n "$ROCM_VERSION" ]] || ROCM_VERSION="unavailable"
fi
if command -v hipconfig >/dev/null 2>&1; then
    HIP_VERSION=$(hipconfig --version 2>/dev/null | head -1)
    [[ -n "$HIP_VERSION" ]] || HIP_VERSION="unavailable"
fi
echo "gpu_identity: $GPU_IDENTITY"
echo "gfx_arch:     $GFX_ARCH"
echo "rocm_version: $ROCM_VERSION"
echo "hip_version:  $HIP_VERSION"

if [[ "$MODE" == "accept" ]]; then
    missing_evidence=()
    [[ "$GPU_IDENTITY" != "unavailable" ]] || missing_evidence+=("gpu_identity")
    [[ "$GFX_ARCH" != "unavailable" ]] || missing_evidence+=("gfx_arch")
    [[ "$ROCM_VERSION" != "unavailable" ]] || missing_evidence+=("rocm_version")
    [[ "$HIP_VERSION" != "unavailable" ]] || missing_evidence+=("hip_version")
    if ((${#missing_evidence[@]} > 0)); then
        echo "run-ep-parity: ACCEPTANCE REFUSED — missing identity/arch/version evidence: ${missing_evidence[*]}" >&3
        exit 2
    fi
fi

# Acceptance verifies the model digest against the pin BEFORE running.
if [[ "$MODE" == "accept" && "$MODEL_SHA" != "$PINNED_MODEL_SHA256" ]]; then
    echo "run-ep-parity: ACCEPTANCE REFUSED — model SHA-256 mismatch" >&3
    echo "  actual: $MODEL_SHA" >&3
    echo "  pinned: $PINNED_MODEL_SHA256" >&3
    exit 2
fi

# ── GPU lock (idempotent trap-safe release on ANY exit path) ─────────
source scripts/gpu-lock.sh
export GPU_LOCK_TIMEOUT=2400
gpu_acquire "device-mesh-ep-parity" || { echo "lock FAIL"; exit 3; }
# Traps are installed ONLY after the lock is held.  EXIT releases once on
# every normal/error exit; INT/TERM release once and then terminate with
# the conventional 130/143.
install_cleanup_traps gpu_release
echo "-- lock acquired $(date -Is) --"

# Deterministic parity: FP32 DeltaNet state, direct execution (no graph
# capture — a baseline-only capture would fake the comparison).
export HIPFIRE_DETERMINISTIC=1
export HIPFIRE_GRAPH=0
export HIPFIRE_PREFILL_MAX_BATCH=256

FEATURES="deltanet,emulated-ep2-harness"
nix develop -c cargo build --release --features "$FEATURES" -p hipfire-runtime \
    --example ep_decode_parity || { echo "BUILD FAIL"; exit 4; }
BIN=target/release/examples/ep_decode_parity
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
echo "binary_sha256: $BIN_SHA"

# Acceptance verifies the binary digest against the pin BEFORE running.
if [[ "$MODE" == "accept" && "$BIN_SHA" != "$PINNED_BINARY_SHA256" ]]; then
    echo "run-ep-parity: ACCEPTANCE REFUSED — binary SHA-256 mismatch" >&3
    echo "  actual: $BIN_SHA" >&3
    echo "  pinned: $PINNED_BINARY_SHA256" >&3
    exit 2
fi

# ── ONE shell-escaped replay command line (env + nix develop + binary) ─
# The SAME command array is executed below, so the logged line always
# reproduces the run exactly: device pinning, deterministic gates, batch
# cap, topology, scenario, and mode are all explicit.
TOPO="single-GPU logical-EP2 (2 ranks, stride-2), HIP_VISIBLE_DEVICES=0"
if [[ "$MODE" == "accept" ]]; then
    MODE_ARGS=(--max-logit-delta "$PINNED_MAX_LOGIT_DELTA")
else
    MODE_ARGS=(--probe)
fi
REPLAY_CMD=(env HIP_VISIBLE_DEVICES=0 HIPFIRE_DETERMINISTIC=1 HIPFIRE_GRAPH=0 \
    HIPFIRE_PREFILL_MAX_BATCH=256 nix develop -c "$BIN" "$MODEL" \
    --prompt-file "$PROMPT_FILE" --steps "$STEPS" --kv-mode "$KV_MODE" \
    --state-quant fp32 --suffix "$SUFFIX" "${MODE_ARGS[@]}")
echo "topology:     $TOPO"
printf 'replay_command: '
printf '%q ' "${REPLAY_CMD[@]}"
printf '\n'

"${REPLAY_CMD[@]}"
PARITY=$?
echo "== parity exit: $PARITY (mode: $MODE) $(date -Is) =="

# ── Truthful final message + exit code (verdicts also on the terminal) ─
if [[ "$MODE" == "probe" ]]; then
    echo "run-ep-parity: RESULT: PROBE (NON-ACCEPTANCE) — deltas/digests reported above;" >&3
    echo "run-ep-parity: nothing was accepted. Pin model/binary SHA-256 + max-logit-delta for acceptance." >&3
    if [[ $PARITY -eq 0 ]]; then exit 0; fi
    exit "$PARITY"
fi
# acceptance
case $PARITY in
    0)
        echo "run-ep-parity: RESULT: ACCEPTANCE PASS" >&3
        exit 0
        ;;
    1)
        echo "run-ep-parity: RESULT: ACCEPTANCE FAIL (see $LOG)" >&3
        exit 1
        ;;
    *)
        echo "run-ep-parity: ACCEPTANCE CONFIG ERROR (exit=$PARITY, see $LOG)" >&3
        exit "$PARITY"
        ;;
esac
