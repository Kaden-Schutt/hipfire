#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# hipfire uninstaller for installs created by scripts/install.sh.
# By default it removes the installed program and preserves models/settings.
set -euo pipefail

if [ -z "${HOME:-}" ]; then
    echo "ERROR: HOME is not set; refusing to choose an uninstall target." >&2
    exit 1
fi
if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    echo "ERROR: do not run this script with sudo; run it as the user who installed hipfire." >&2
    exit 1
fi

USER_HOME="${HOME%/}"
if [ -z "$USER_HOME" ] || [ "$USER_HOME" = "/" ]; then
    echo "ERROR: unsafe HOME value '$HOME'; refusing to uninstall." >&2
    exit 1
fi

HIPFIRE_DIR="$USER_HOME/.hipfire"
BIN_DIR="$HIPFIRE_DIR/bin"
SRC_DIR="$HIPFIRE_DIR/src"
LEGACY_CLI_DIR="$HIPFIRE_DIR/cli"
PURGE=0
ASSUME_YES=0
DRY_RUN=0

usage() {
    cat <<'EOF'
hipfire uninstaller

Usage:
  uninstall.sh [--dry-run] [--purge] [--yes]

Options:
  --dry-run   Show what would be removed without changing anything
  --purge     Also delete models, configuration, and all other ~/.hipfire data
  --yes       Skip the interactive confirmation required by --purge
  -h, --help  Show this help

Default behavior:
  Removes installed binaries, kernels, managed source, runtime PID/log files,
  and the PATH line added by install.sh. Models and settings are preserved.

Examples:
  curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
  curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash -s -- --dry-run
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --purge)
            PURGE=1
            ;;
        --yes)
            ASSUME_YES=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown uninstaller option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "$HIPFIRE_DIR" in
    "$USER_HOME/.hipfire") ;;
    *)
        echo "ERROR: unsafe uninstall target '$HIPFIRE_DIR'." >&2
        exit 1
        ;;
esac

path_exists() {
    [ -e "$1" ] || [ -L "$1" ]
}

remove_tree() {
    local target="$1"
    case "$target" in
        "$BIN_DIR"|"$SRC_DIR"|"$LEGACY_CLI_DIR"|"$HIPFIRE_DIR") ;;
        *)
            echo "ERROR: refusing unexpected recursive removal target '$target'." >&2
            exit 1
            ;;
    esac
    path_exists "$target" || return 0
    if [ "$DRY_RUN" = "1" ]; then
        echo "Would remove: $target"
    else
        rm -rf -- "$target"
        echo "Removed: $target"
    fi
}

remove_file() {
    local target="$1"
    case "$target" in
        "$HIPFIRE_DIR/serve.pid"|"$HIPFIRE_DIR/daemon.pid"|"$HIPFIRE_DIR/serve.log") ;;
        *)
            echo "ERROR: refusing unexpected file removal target '$target'." >&2
            exit 1
            ;;
    esac
    path_exists "$target" || return 0
    if [ "$DRY_RUN" = "1" ]; then
        echo "Would remove: $target"
    else
        rm -f -- "$target"
        echo "Removed: $target"
    fi
}

stop_installed_processes() {
    if [ "$DRY_RUN" = "1" ]; then
        if path_exists "$HIPFIRE_DIR/serve.pid" || path_exists "$HIPFIRE_DIR/daemon.pid"; then
            echo "Would stop running hipfire processes owned by this install"
        fi
        return
    fi

    if path_exists "$HIPFIRE_DIR/serve.pid" && [ -x "$BIN_DIR/hipfire" ]; then
        if "$BIN_DIR/hipfire" stop >/dev/null 2>&1; then
            echo "Stopped: hipfire serve"
        else
            echo "WARNING: hipfire serve did not stop cleanly; continuing with ownership-safe cleanup." >&2
        fi
    fi

    # A directly launched daemon is not covered by `hipfire stop`. Only signal
    # the PID when /proc proves it is this install's daemon binary.
    if path_exists "$HIPFIRE_DIR/daemon.pid"; then
        local daemon_pid expected_exe running_exe attempt
        daemon_pid="$(tr -d '[:space:]' < "$HIPFIRE_DIR/daemon.pid" 2>/dev/null || true)"
        case "$daemon_pid" in
            ""|*[!0-9]*) return ;;
        esac
        if ! kill -0 "$daemon_pid" 2>/dev/null; then
            return
        fi
        expected_exe="$(readlink -f "$BIN_DIR/daemon" 2>/dev/null || true)"
        running_exe="$(readlink -f "/proc/$daemon_pid/exe" 2>/dev/null || true)"
        if [ -z "$expected_exe" ] || [ "$running_exe" != "$expected_exe" ]; then
            echo "WARNING: PID $daemon_pid is not $BIN_DIR/daemon; refusing to signal it." >&2
            return
        fi
        kill -TERM "$daemon_pid"
        for ((attempt = 0; attempt < 50; attempt++)); do
            kill -0 "$daemon_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$daemon_pid" 2>/dev/null; then
            echo "WARNING: daemon PID $daemon_pid is still running; no force-kill was attempted." >&2
        else
            echo "Stopped: hipfire daemon (PID $daemon_pid)"
        fi
    fi
}

profile_has_install_path() {
    local profile="$1"
    [ -f "$profile" ] || return 1
    # shellcheck disable=SC2016  # Match the installer's literal shell expression.
    grep -Fqx 'export PATH="$HOME/.hipfire/bin:$PATH"' "$profile"
}

clean_profile() {
    local profile="$1" temporary
    profile_has_install_path "$profile" || return 0
    if [ "$DRY_RUN" = "1" ]; then
        echo "Would remove hipfire PATH entry from: $profile"
        return
    fi

    temporary="$(mktemp "${profile}.hipfire-uninstall.XXXXXX")"
    awk '
        $0 == "# hipfire" {
            marker = $0
            next
        }
        $0 == "export PATH=\"$HOME/.hipfire/bin:$PATH\"" {
            marker = ""
            next
        }
        {
            if (marker != "") {
                print marker
                marker = ""
            }
            print
        }
        END {
            if (marker != "") {
                print marker
            }
        }
    ' "$profile" > "$temporary"
    chmod --reference="$profile" "$temporary"
    mv -- "$temporary" "$profile"
    echo "Updated PATH: $profile"
}

source_is_managed() {
    [ -d "$SRC_DIR/.git" ] || return 1
    command -v git >/dev/null 2>&1 || return 1
    local origin
    origin="$(git -C "$SRC_DIR" remote get-url origin 2>/dev/null || true)"
    # The canonical repo moved from Kaden-Schutt/hipfire to warpfront/hipfire.
    # Checkouts created by an installer from before the transfer still carry the
    # old origin, so BOTH namespaces must be recognized as managed — otherwise
    # `uninstall` silently refuses to clean up every pre-transfer install.
    case "$origin" in
        https://github.com/warpfront/hipfire|\
        https://github.com/warpfront/hipfire.git|\
        git@github.com:warpfront/hipfire|\
        git@github.com:warpfront/hipfire.git|\
        ssh://git@github.com/warpfront/hipfire|\
        ssh://git@github.com/warpfront/hipfire.git|\
        https://github.com/Kaden-Schutt/hipfire|\
        https://github.com/Kaden-Schutt/hipfire.git|\
        git@github.com:Kaden-Schutt/hipfire|\
        git@github.com:Kaden-Schutt/hipfire.git|\
        ssh://git@github.com/Kaden-Schutt/hipfire|\
        ssh://git@github.com/Kaden-Schutt/hipfire.git)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

source_has_local_work() {
    [ -n "$(git -C "$SRC_DIR" status --porcelain --untracked-files=normal 2>/dev/null)" ] && return 0
    git -C "$SRC_DIR" rev-parse --quiet --verify refs/stash >/dev/null 2>&1 && return 0

    local branch upstream ahead
    while IFS=$'\t' read -r branch upstream; do
        [ -n "$branch" ] || continue
        if [ -z "$upstream" ]; then
            # A managed checkout may have a local branch without upstream
            # metadata. It is safe only when a remote branch or fetched tag
            # still contains its tip; otherwise it may hold unpublished work.
            [ -n "$(git -C "$SRC_DIR" for-each-ref --contains="$branch" \
                --format='%(refname)' refs/remotes refs/tags 2>/dev/null)" ] || return 0
            continue
        fi
        git -C "$SRC_DIR" rev-parse --verify "$upstream" >/dev/null 2>&1 || return 0
        ahead="$(git -C "$SRC_DIR" rev-list --count "$upstream..$branch" 2>/dev/null || echo 1)"
        [ "$ahead" = "0" ] || return 0
    done < <(git -C "$SRC_DIR" for-each-ref \
        --format='%(refname:short)%09%(upstream:short)' refs/heads 2>/dev/null)
    return 1
}

remove_managed_source() {
    path_exists "$SRC_DIR" || return 0
    if ! source_is_managed; then
        echo "Preserved: $SRC_DIR (not a recognized managed hipfire checkout)"
        return
    fi
    if source_has_local_work; then
        echo "Preserved: $SRC_DIR (contains local work or git stashes)"
        echo "  Review it manually, or use --purge to delete all hipfire data."
        return
    fi
    remove_tree "$SRC_DIR"
}

confirm_purge() {
    [ "$PURGE" = "1" ] || return 0
    [ "$DRY_RUN" = "0" ] || return 0
    [ "$ASSUME_YES" = "0" ] || return 0

    local reply
    echo "PURGE will permanently delete models, settings, and every file under:"
    echo "  $HIPFIRE_DIR"
    if ! printf "Type 'delete' to continue: " >/dev/tty 2>/dev/null; then
        echo "ERROR: --purge needs an interactive terminal or --yes." >&2
        exit 2
    fi
    read -r reply </dev/tty 2>/dev/null || reply=""
    if [ "$reply" != "delete" ]; then
        echo "Purge cancelled."
        exit 1
    fi
}

echo "=== hipfire uninstaller ==="
echo "Install root: $HIPFIRE_DIR"
if [ "$DRY_RUN" = "1" ]; then
    echo "Mode: dry run"
elif [ "$PURGE" = "1" ]; then
    echo "Mode: purge"
else
    echo "Mode: preserve models and settings"
fi
echo ""

confirm_purge
stop_installed_processes
clean_profile "$USER_HOME/.bashrc"
clean_profile "$USER_HOME/.zshrc"

if [ "$PURGE" = "1" ]; then
    remove_tree "$HIPFIRE_DIR"
else
    remove_tree "$BIN_DIR"
    remove_tree "$LEGACY_CLI_DIR"
    remove_managed_source
    remove_file "$HIPFIRE_DIR/serve.pid"
    remove_file "$HIPFIRE_DIR/daemon.pid"
    remove_file "$HIPFIRE_DIR/serve.log"
fi

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete; nothing was changed."
elif [ "$PURGE" = "1" ]; then
    echo "hipfire and all ~/.hipfire data were removed."
else
    echo "hipfire was uninstalled."
    echo "Models and settings were preserved under $HIPFIRE_DIR."
    echo "Run this script with --purge to remove that data too."
fi
echo "ROCm, Rust, and other shared system dependencies were not removed."
