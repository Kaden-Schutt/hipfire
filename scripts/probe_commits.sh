#!/usr/bin/env bash
# Probe a list of commits for 9B decode perf. Stashes/restores Cargo.lock
# automatically. Skips commits whose build fails. Output per commit:
#   <hash>  <gen_tok_s>  <short message>
set -u
COMMITS=("$@")
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# Stash any dirty state
git stash push -u -m "probe_commits_autostash_$$" >/dev/null 2>&1
STASHED=$?

trap 'git checkout "$START_BRANCH" >/dev/null 2>&1 || true; [ "$STASHED" -eq 0 ] && git stash pop >/dev/null 2>&1 || true' EXIT

for h in "${COMMITS[@]}"; do
    msg=$(git show --no-patch --format="%s" "$h" | head -c 50)
    echo -n "$h  "
    # Cargo may have dirtied Cargo.lock during previous iteration's build.
    # -f discards that so the checkout can proceed.
    if ! git checkout -f "$h" >/dev/null 2>&1; then
        echo "CHECKOUT_FAIL  $msg"
        continue
    fi
    rm -f target/release/examples/bench_qwen35_mq4
    if ! cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_mq4 >/tmp/probe_build.log 2>&1; then
        echo "BUILD_FAIL  $msg"
        continue
    fi
    out=$(HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
        target/release/examples/bench_qwen35_mq4 "$HOME/.hipfire/models/qwen3.5-9b.mq4" \
        --prefill 16 --warmup 3 --gen 30 2>&1)
    tok_s=$(echo "$out" | grep -oE 'gen_tok_s=[0-9.]+' | sed 's/gen_tok_s=//')
    if [ -z "$tok_s" ]; then
        echo "BENCH_FAIL  $msg"
    else
        printf '%7s tok/s  %s\n' "$tok_s" "$msg"
    fi
done
