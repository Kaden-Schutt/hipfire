#!/usr/bin/env bash
# kld_daemon.sh — drive the hipfire-daemon `kld_eval` op from shell.
#
# Replaces the removed `eval_hipfire` / `build_kld_ref` examples and the
# cross-engine HFKLDR `.kldref.bin` format. All KLD references are now
# hipfire-self HFKREF built by the daemon's own forward (mode=build_ref); a
# candidate is scored against one (mode=score) through the IDENTICAL resident
# forward + the shared `hipfire-kld` core, so the historical two-binary drift
# is impossible by construction.
#
# Usage:  source "$(dirname "$0")/lib/kld_daemon.sh"
#   kld_build_ref <ref_model.hfq> <corpus.txt> <out.kldref> [max_chunks] [kv_mode] [n_ctx]
#   kld_score     <model.hfq>     <ref.kldref>  <out.kldseq> [max_chunks] [kv_mode] [n_ctx]
#   kld_field     '<kld_evaled-json>' <field>          # mean_kld | p99_kld | ppl | ...
#
# Both functions echo the daemon's final `{"type":"kld_evaled",...}` JSON line
# (and write the .kldref / .kldseq artifact). Non-zero return on daemon error.
# Env: HIPFIRE_DAEMON_BIN overrides the daemon path; KLD_DAEMON_LOG captures
#      daemon stderr; HIPFIRE_RESOURCE_LOCK_WAIT_MS tunes the GPU lock wait.

kld_daemon_bin() {
    if [ -n "${HIPFIRE_DAEMON_BIN:-}" ] && [ -x "${HIPFIRE_DAEMON_BIN}" ]; then
        echo "${HIPFIRE_DAEMON_BIN}"; return 0
    fi
    local c
    for c in target/release/hipfire-daemon \
             "${HOME}/.hipfire/bin/hipfire-daemon" \
             "$(command -v hipfire-daemon 2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
    done
    echo "kld_daemon: hipfire-daemon not found (build: cargo build --release -p hipfire-daemon --features deltanet)" >&2
    return 2
}

_kld_run() {  # internal: feed the two JSON lines to the daemon, emit kld_evaled
    local bin; bin=$(kld_daemon_bin) || return 2
    HIPFIRE_RESOURCE_LOCK_WAIT_MS="${HIPFIRE_RESOURCE_LOCK_WAIT_MS:-600000}" \
        "$bin" 2>>"${KLD_DAEMON_LOG:-/dev/null}" \
        | grep '"type":"kld_evaled"\|"type":"error"'
}

kld_build_ref() {  # <ref_model> <corpus> <out.kldref> [max_chunks] [kv_mode] [n_ctx]
    local model="$1" corpus="$2" out="$3" max_chunks="${4:-}" kv="${5:-q8}" nctx="${6:-2048}"
    local mc=""; [ -n "$max_chunks" ] && mc=",\"max_chunks\":$max_chunks"
    local line
    line=$(
        { printf '{"type":"load","model":"%s","params":{"max_seq":%d,"kv_cache":"%s"}}\n' "$model" $((nctx * 2)) "$kv"
          printf '{"type":"kld_eval","mode":"build_ref","corpus":"%s","ref_path":"%s","n_ctx":%d%s}\n' "$corpus" "$out" "$nctx" "$mc"
        } | _kld_run
    )
    echo "$line"
    case "$line" in *'"type":"error"'*|'') return 1 ;; esac
    return 0
}

kld_score() {  # <model> <ref.kldref> <out.kldseq> [max_chunks] [kv_mode] [n_ctx]
    local model="$1" ref="$2" out="$3" max_chunks="${4:-}" kv="${5:-q8}" nctx="${6:-2048}"
    local mc=""; [ -n "$max_chunks" ] && mc=",\"max_chunks\":$max_chunks"
    local line
    line=$(
        { printf '{"type":"load","model":"%s","params":{"max_seq":%d,"kv_cache":"%s"}}\n' "$model" $((nctx * 2)) "$kv"
          printf '{"type":"kld_eval","mode":"score","ref_path":"%s","output":"%s"%s}\n' "$ref" "$out" "$mc"
        } | _kld_run
    )
    echo "$line"
    case "$line" in *'"type":"error"'*|'') return 1 ;; esac
    return 0
}

kld_field() {  # '<kld_evaled-json>' <field>
    python3 -c 'import json,sys
try: d=json.loads(sys.argv[1])
except Exception: sys.exit(0)
v=d.get(sys.argv[2])
print("" if v is None else v)' "$1" "$2"
}
