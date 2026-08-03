#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Thin bootstrap: obtain source, build hipfire-cli, hand off to `hipfire setup`.
# Usage: curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash
# Branch: curl -fsSL .../master/scripts/install.sh | bash -s -- --branch beta
set -euo pipefail

HIPFIRE_HOME="${HIPFIRE_HOME:-$HOME/.hipfire}"
# Internal override for deterministic local-remote tests; default is the official repo.
GITHUB_URL="${HIPFIRE_GITHUB_URL:-https://github.com/warpfront/hipfire.git}"

# Managed-source transaction state (curl|bash reusing $HIPFIRE_HOME/src).
_TX_ARMED=0
_TX_REPO=""
_TX_PRIOR_HEAD=""
_TX_PRIOR_BRANCH=""
_TX_PRIOR_DETACHED=0
_TX_ORIGIN_HAD=0
_TX_ORIGIN_PRIOR=""
_TX_ORIGIN_CREATED=0
_TX_TARGET_BRANCH=""
_TX_TARGET_EXISTED=0
_TX_TARGET_PRIOR_SHA=""

_tx_disarm() {
    _TX_ARMED=0
}

# Restore checkpoint after a failed checkout/build/setup (or signal). Best-effort;
# never masks the original failure status of the caller.
_tx_rollback() {
    [ "${_TX_ARMED:-0}" -eq 1 ] || return 0
    _TX_ARMED=0
    local repo="${_TX_REPO:-}"
    [ -n "$repo" ] && [ -d "$repo/.git" ] || return 0

    # Restore any moved/created target branch ref from checkout -B first, while
    # still able to move refs freely; then restore HEAD/worktree.
    if [ -n "${_TX_TARGET_BRANCH:-}" ]; then
        if [ "${_TX_TARGET_EXISTED:-0}" -eq 1 ] && [ -n "${_TX_TARGET_PRIOR_SHA:-}" ]; then
            git -C "$repo" update-ref "refs/heads/${_TX_TARGET_BRANCH}" "${_TX_TARGET_PRIOR_SHA}" >/dev/null 2>&1 || true
        elif git -C "$repo" rev-parse --verify --quiet "refs/heads/${_TX_TARGET_BRANCH}" >/dev/null 2>&1; then
            # Created by checkout -B; drop only if we are not still sitting on it.
            # Switch away first when needed so the delete can succeed.
            if [ -n "${_TX_PRIOR_HEAD:-}" ]; then
                git -C "$repo" checkout -f --detach "${_TX_PRIOR_HEAD}" >/dev/null 2>&1 || true
            fi
            git -C "$repo" update-ref -d "refs/heads/${_TX_TARGET_BRANCH}" >/dev/null 2>&1 || true
        fi
    fi

    # Discard only post-checkout worktree/index mutations (e.g. Cargo.lock writes).
    # Done after target-ref restore so reset lands on the restored tip when possible.
    git -C "$repo" reset --hard HEAD >/dev/null 2>&1 || true
    git -C "$repo" clean -fd >/dev/null 2>&1 || true

    # Restore prior branch or detached HEAD.
    if [ -n "${_TX_PRIOR_HEAD:-}" ]; then
        if [ "${_TX_PRIOR_DETACHED:-0}" -eq 1 ] || [ -z "${_TX_PRIOR_BRANCH:-}" ]; then
            git -C "$repo" checkout -f --detach "${_TX_PRIOR_HEAD}" >/dev/null 2>&1 || true
        else
            git -C "$repo" checkout -f -B "${_TX_PRIOR_BRANCH}" "${_TX_PRIOR_HEAD}" >/dev/null 2>&1 || true
        fi
    fi

    # Restore prior origin URL (or remove origin we created).
    if [ "${_TX_ORIGIN_HAD:-0}" -eq 1 ] && [ -n "${_TX_ORIGIN_PRIOR:-}" ]; then
        git -C "$repo" remote set-url origin "${_TX_ORIGIN_PRIOR}" >/dev/null 2>&1 || true
    elif [ "${_TX_ORIGIN_CREATED:-0}" -eq 1 ]; then
        git -C "$repo" remote remove origin >/dev/null 2>&1 || true
    fi
}

_tx_fail() {
    # Preserve original failure status after rollback.
    local status="${1:-1}"
    _tx_rollback
    exit "$status"
}

# Retain original argv for forwarding to `hipfire setup`.
ORIGINAL_ARGS=("$@")

# Quiet step state: temp log + spinner PID (if any).
_STEP_LOG=""
_SPINNER_PID=""

_cleanup_step() {
    if [ -n "${_SPINNER_PID:-}" ]; then
        kill "${_SPINNER_PID}" 2>/dev/null || true
        wait "${_SPINNER_PID}" 2>/dev/null || true
        _SPINNER_PID=""
        # Clear spinner line when stderr is a TTY.
        if [ -t 2 ]; then
            printf '\r\033[K' >&2
        fi
    fi
    if [ -n "${_STEP_LOG:-}" ] && [ -f "${_STEP_LOG}" ]; then
        rm -f "${_STEP_LOG}"
        _STEP_LOG=""
    fi
}

_on_signal() {
    _cleanup_step
    if [ "${_TX_ARMED:-0}" -eq 1 ]; then
        _tx_rollback
    fi
    exit 130
}

_on_exit() {
    _cleanup_step
    # If still armed on shell exit, roll back (failed path that didn't call _tx_fail).
    if [ "${_TX_ARMED:-0}" -eq 1 ]; then
        _tx_rollback
    fi
}

trap '_on_signal' INT TERM
trap '_on_exit' EXIT

_spinner() {
    local frames='|/-\' i=0
    while true; do
        printf '\r[%c] ' "${frames:i++%4:1}" >&2
        sleep 0.1
    done
}

# Run a command quietly: capture output, optional TTY spinner, dump on failure.
# Usage: quiet_step <label> <command> [args...]
quiet_step() {
    local label="$1"
    shift
    local status=0

    _STEP_LOG="$(mktemp "${TMPDIR:-/tmp}/hipfire-bootstrap.XXXXXX")"

    if [ -t 2 ]; then
        _spinner &
        _SPINNER_PID=$!
    fi

    set +e
    "$@" >"${_STEP_LOG}" 2>&1
    status=$?
    set -e

    if [ -n "${_SPINNER_PID:-}" ]; then
        kill "${_SPINNER_PID}" 2>/dev/null || true
        wait "${_SPINNER_PID}" 2>/dev/null || true
        _SPINNER_PID=""
        printf '\r\033[K' >&2
    fi

    if [ "$status" -ne 0 ]; then
        echo "ERROR: ${label} failed (exit ${status})" >&2
        cat "${_STEP_LOG}" >&2
        rm -f "${_STEP_LOG}"
        _STEP_LOG=""
        return "$status"
    fi

    rm -f "${_STEP_LOG}"
    _STEP_LOG=""
    return 0
}

usage() {
    cat <<'EOF'
Usage: install.sh [options]

Options:
  --branch NAME              Install from branch NAME
  --ref REF                  Install from git ref REF
  --tag TAG                  Install from tag TAG
  --commit SHA               Install from commit SHA
  --rocm-root PATH           ROCm installation root (forwarded to setup)
  --gpu-arch ARCH            GPU architecture (forwarded to setup)
  --profile auto|hip|redline Replay profile (forwarded to setup)
  --yes, -y, --non-interactive
                             Non-interactive mode
  --help                     Show this help

Bootstrap only initializes source and builds hipfire-cli, then execs:
  hipfire setup --source <repo> [original args...]

Examples:
  bash -s -- --branch beta
  HIPFIRE_INSTALL_REF=beta bash install.sh --yes
EOF
}

BRANCH=""
REF=""
TAG=""
COMMIT=""
YES=0
SELECTOR=""
SELECTOR_KIND=""
SELECTOR_COUNT=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --branch)
            [ "$#" -ge 2 ] || { echo "ERROR: --branch requires a value" >&2; exit 1; }
            BRANCH="$2"
            SELECTOR="$2"
            SELECTOR_KIND="branch"
            SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
            shift 2
            ;;
        --ref)
            [ "$#" -ge 2 ] || { echo "ERROR: --ref requires a value" >&2; exit 1; }
            REF="$2"
            SELECTOR="$2"
            SELECTOR_KIND="ref"
            SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
            shift 2
            ;;
        --tag)
            [ "$#" -ge 2 ] || { echo "ERROR: --tag requires a value" >&2; exit 1; }
            TAG="$2"
            SELECTOR="$2"
            SELECTOR_KIND="tag"
            SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
            shift 2
            ;;
        --commit)
            [ "$#" -ge 2 ] || { echo "ERROR: --commit requires a value" >&2; exit 1; }
            COMMIT="$2"
            SELECTOR="$2"
            SELECTOR_KIND="commit"
            SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
            shift 2
            ;;
        --profile)
            [ "$#" -ge 2 ] || { echo "ERROR: --profile requires a value" >&2; exit 1; }
            case "$2" in
                auto|hip|redline) ;;
                *)
                    echo "ERROR: --profile must be exactly auto, hip, or redline (got: $2)" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --rocm-root|--gpu-arch)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            shift 2
            ;;
        --yes|-y|--non-interactive)
            YES=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            # Unknown flags are forwarded unchanged to setup.
            shift
            ;;
    esac
done

if [ "$SELECTOR_COUNT" -gt 1 ]; then
    echo "ERROR: choose only one of --branch, --ref, --tag, or --commit" >&2
    exit 2
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required but was not found on PATH." >&2
    echo "Install git (e.g. sudo apt install git) and re-run." >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo is required but was not found on PATH." >&2
    echo "Install Rust from https://rustup.rs/ and re-run." >&2
    exit 1
fi

# Validate a git revision token. Kind-specific: commit is 7-40 hex only;
# branch/tag/ref reject option-like values, path traversal, and other unsafe forms.
validate_revision() {
    local value="$1"
    local kind="$2"

    if [ -z "$value" ]; then
        echo "ERROR: unsafe or invalid git revision '$value'" >&2
        exit 2
    fi

    case "$kind" in
        commit)
            if ! [[ "$value" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
                echo "ERROR: --commit requires a 7-40 character hexadecimal git commit" >&2
                exit 2
            fi
            return 0
            ;;
    esac

    # Reject option-like values, path components, and git-special sequences.
    # Mirrors hipfire-cli validate_revision character rules.
    case "$value" in
        -* | .* | /* | *. | */ | *..* | *@{* | *//* | *\\* | *:* | *\?* | *\** | *\[* | *^* | *~*)
            echo "ERROR: unsafe or invalid git revision '$value'" >&2
            exit 2
            ;;
    esac
    case "$value" in
        *" "* | *$'\t'* | *$'\n'* | *$'\r'*)
            echo "ERROR: unsafe or invalid git revision '$value'" >&2
            exit 2
            ;;
    esac

    # git check-ref-format for branch/tag names; --ref allows branch/tag shape or hex.
    case "$kind" in
        branch)
            if ! git check-ref-format --branch "$value" >/dev/null 2>&1; then
                echo "ERROR: unsafe or invalid git revision '$value'" >&2
                exit 2
            fi
            ;;
        tag)
            if ! git check-ref-format "refs/tags/$value" >/dev/null 2>&1; then
                echo "ERROR: unsafe or invalid git revision '$value'" >&2
                exit 2
            fi
            ;;
        ref)
            if ! git check-ref-format --branch "$value" >/dev/null 2>&1 \
                && ! git check-ref-format "refs/tags/$value" >/dev/null 2>&1; then
                if ! [[ "$value" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
                    echo "ERROR: unsafe or invalid git revision '$value'" >&2
                    exit 2
                fi
            fi
            ;;
    esac
}

# CLI selector wins; otherwise HIPFIRE_INSTALL_REF is generic --ref for automation.
ENV_REF_APPLIED=0
if [ "$SELECTOR_COUNT" -eq 0 ] && [ -n "${HIPFIRE_INSTALL_REF:-}" ]; then
    SELECTOR="$HIPFIRE_INSTALL_REF"
    SELECTOR_KIND="ref"
    SELECTOR_COUNT=1
    ENV_REF_APPLIED=1
fi

if [ -n "$SELECTOR" ] && [ -n "$SELECTOR_KIND" ]; then
    validate_revision "$SELECTOR" "$SELECTOR_KIND"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FORWARDED_ARGS=("${ORIGINAL_ARGS[@]}")
if [ "$ENV_REF_APPLIED" -eq 1 ]; then
    FORWARDED_ARGS=("--ref" "$SELECTOR" "${ORIGINAL_ARGS[@]}")
fi

if [ -f "$REPO_DIR/Cargo.toml" ]; then
    # Local checkout (including `hipfire update`): use as-is, never fetch/switch.
    :
else
    # curl|bash path: managed install under $HIPFIRE_HOME/src
    REPO_DIR="$HIPFIRE_HOME/src"

    if [ -z "$SELECTOR" ]; then
        if [ "$YES" -eq 0 ] && [ -t 0 ]; then
            printf 'Source channel [master/beta/<branch>] (master): '
            read -r answer || true
            answer="${answer:-master}"
            SELECTOR="$answer"
            SELECTOR_KIND="branch"
        else
            SELECTOR="master"
            SELECTOR_KIND="branch"
        fi
        validate_revision "$SELECTOR" "$SELECTOR_KIND"
        # Include the chosen/default branch in forwarded args for install metadata.
        FORWARDED_ARGS=("--branch" "$SELECTOR" "${ORIGINAL_ARGS[@]}")
    fi

    mkdir -p "$REPO_DIR"

    MANAGED_EXISTING=0
    if [ -d "$REPO_DIR/.git" ]; then
        MANAGED_EXISTING=1
    fi

    if [ "$MANAGED_EXISTING" -eq 1 ]; then
        # Existing managed source is user state: refuse dirty work before mutation.
        if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=normal 2>/dev/null || true)" ]; then
            echo "ERROR: managed source at $REPO_DIR is dirty (including untracked files); commit, stash, or clean before re-running the installer" >&2
            exit 1
        fi

        if git -C "$REPO_DIR" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
            _TX_PRIOR_HEAD="$(git -C "$REPO_DIR" rev-parse HEAD)"
            if _TX_PRIOR_BRANCH="$(git -C "$REPO_DIR" symbolic-ref -q --short HEAD 2>/dev/null)"; then
                _TX_PRIOR_DETACHED=0
            else
                _TX_PRIOR_BRANCH=""
                _TX_PRIOR_DETACHED=1
            fi
        else
            # Empty git dir (init without commits): treat as fresh.
            MANAGED_EXISTING=0
        fi
    fi

    if [ "$MANAGED_EXISTING" -eq 1 ]; then
        _TX_REPO="$REPO_DIR"
        if git -C "$REPO_DIR" remote get-url origin >/dev/null 2>&1; then
            _TX_ORIGIN_HAD=1
            _TX_ORIGIN_PRIOR="$(git -C "$REPO_DIR" remote get-url origin)"
        else
            _TX_ORIGIN_HAD=0
            _TX_ORIGIN_PRIOR=""
        fi

        # Arm before origin/fetch/checkout mutations.
        _TX_ARMED=1

        if [ "$_TX_ORIGIN_HAD" -eq 1 ]; then
            if [ "$_TX_ORIGIN_PRIOR" != "$GITHUB_URL" ]; then
                quiet_step "git remote set-url" git -C "$REPO_DIR" remote set-url origin "$GITHUB_URL" || _tx_fail $?
            fi
        else
            quiet_step "git remote add" git -C "$REPO_DIR" remote add origin "$GITHUB_URL" || _tx_fail $?
            _TX_ORIGIN_CREATED=1
        fi

        case "$SELECTOR_KIND" in
            branch)
                _TX_TARGET_BRANCH="$SELECTOR"
                if git -C "$REPO_DIR" rev-parse --verify --quiet "refs/heads/$SELECTOR" >/dev/null 2>&1; then
                    _TX_TARGET_EXISTED=1
                    _TX_TARGET_PRIOR_SHA="$(git -C "$REPO_DIR" rev-parse "refs/heads/$SELECTOR")"
                    # Full fetch so local tip can be tested as ancestor of FETCH_HEAD.
                    quiet_step "git fetch" git -C "$REPO_DIR" fetch origin "$SELECTOR" || _tx_fail $?
                    if ! git -C "$REPO_DIR" merge-base --is-ancestor "refs/heads/$SELECTOR" FETCH_HEAD 2>/dev/null; then
                        echo "ERROR: local branch '$SELECTOR' has commits not contained in the fetched tip; refusing to reset $REPO_DIR" >&2
                        _tx_fail 1
                    fi
                else
                    _TX_TARGET_EXISTED=0
                    _TX_TARGET_PRIOR_SHA=""
                    quiet_step "git fetch" git -C "$REPO_DIR" fetch --depth 1 origin "$SELECTOR" || _tx_fail $?
                fi
                quiet_step "git checkout" git -C "$REPO_DIR" checkout -B "$SELECTOR" FETCH_HEAD || _tx_fail $?
                ;;
            tag)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch origin "refs/tags/$SELECTOR:refs/tags/$SELECTOR" || _tx_fail $?
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach "refs/tags/$SELECTOR" || _tx_fail $?
                ;;
            commit)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch origin "$SELECTOR" || _tx_fail $?
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach "$SELECTOR" || _tx_fail $?
                ;;
            ref)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch origin "$SELECTOR" || _tx_fail $?
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach FETCH_HEAD || _tx_fail $?
                ;;
            *)
                echo "ERROR: internal: unknown selector kind '$SELECTOR_KIND'" >&2
                _tx_fail 1
                ;;
        esac
    else
        # Fresh managed source: simple init/fetch/checkout (nothing user-owned to restore).
        if [ ! -d "$REPO_DIR/.git" ]; then
            quiet_step "git init" git -C "$REPO_DIR" init
        fi

        if git -C "$REPO_DIR" remote get-url origin >/dev/null 2>&1; then
            quiet_step "git remote set-url" git -C "$REPO_DIR" remote set-url origin "$GITHUB_URL"
        else
            quiet_step "git remote add" git -C "$REPO_DIR" remote add origin "$GITHUB_URL"
        fi

        case "$SELECTOR_KIND" in
            branch)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch --depth 1 origin "$SELECTOR"
                quiet_step "git checkout" git -C "$REPO_DIR" checkout -B "$SELECTOR" FETCH_HEAD
                ;;
            tag)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch --depth 1 origin "refs/tags/$SELECTOR:refs/tags/$SELECTOR"
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach "refs/tags/$SELECTOR"
                ;;
            commit)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch --depth 1 origin "$SELECTOR"
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach "$SELECTOR"
                ;;
            ref)
                quiet_step "git fetch" git -C "$REPO_DIR" fetch --depth 1 origin "$SELECTOR"
                quiet_step "git checkout" git -C "$REPO_DIR" checkout --detach FETCH_HEAD
                ;;
            *)
                echo "ERROR: internal: unknown selector kind '$SELECTOR_KIND'" >&2
                exit 1
                ;;
        esac
    fi
fi

export CARGO_TARGET_DIR="$REPO_DIR/target"
if [ "${_TX_ARMED:-0}" -eq 1 ]; then
    quiet_step "cargo build hipfire-cli" bash -c "cd \"\$1\" && cargo build --release -p hipfire-cli" _ "$REPO_DIR" || _tx_fail $?
else
    quiet_step "cargo build hipfire-cli" bash -c "cd \"\$1\" && cargo build --release -p hipfire-cli" _ "$REPO_DIR"
fi

HIPFIRE_BIN="$REPO_DIR/target/release/hipfire"
if [ ! -x "$HIPFIRE_BIN" ]; then
    echo "ERROR: expected binary missing: $HIPFIRE_BIN" >&2
    if [ "${_TX_ARMED:-0}" -eq 1 ]; then
        _tx_fail 1
    fi
    exit 1
fi

if [ "${_TX_ARMED:-0}" -eq 1 ]; then
    # Do not exec while rollback is armed: run setup, restore on failure, disarm only on success.
    set +e
    "$HIPFIRE_BIN" setup --source "$REPO_DIR" "${FORWARDED_ARGS[@]}"
    setup_status=$?
    set -e
    if [ "$setup_status" -ne 0 ]; then
        _tx_fail "$setup_status"
    fi
    _tx_disarm
    # Clear EXIT trap noise; setup already succeeded.
    trap - EXIT
    exit 0
fi

# Clear EXIT trap so exec does not remove state the child never needs.
trap - EXIT
exec "$HIPFIRE_BIN" setup --source "$REPO_DIR" "${FORWARDED_ARGS[@]}"
