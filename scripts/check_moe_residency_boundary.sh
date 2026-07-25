#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for symbol in \
    ExpertShardResident \
    ExpertShardAssembly \
    ExpertShardResourceKind \
    ExpertShardResource \
    ExpertShardTarget \
    ExpertShardSlot; do
    if matches=$(git grep -n -F -e "$symbol" -- ':(glob)crates/**/*.rs'); then
        printf 'MOE residency boundary check failed: %s remains in tracked Rust sources:\n%s\n' \
            "$symbol" "$matches" >&2
        found=1
    else
        status=$?
        if [[ $status -ne 1 ]]; then
            printf 'MOE residency boundary check failed: git grep exited with status %d while checking %s\n' \
                "$status" "$symbol" >&2
            exit "$status"
        fi
    fi
done

if [[ ${found:-0} -ne 0 ]]; then
    exit 1
fi

printf 'MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.\n'
