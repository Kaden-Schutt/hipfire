#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
weight_store="$repo_root/crates/hipfire-runtime/src/weight_store.rs"

forbidden_symbols=(
  WeightStoreView
  WeightStoreKey
  WeightStoreAuxiliary
  WeightStorePoolTarget
  DurableWeightStoreAssembly
  DurableWeightStoreGuard
  begin_durable_assembly
  free_all_to_pool
  auxiliary_len
  auxiliary_view
  prepare_finalize
  allocate_auxiliary
  register_allocated
  register_host_auxiliary
  register_placement
)

for symbol in "${forbidden_symbols[@]}"; do
  if rg --fixed-strings --line-number -- "$symbol" "$weight_store" >/dev/null; then
    printf 'forbidden hybrid foundation symbol remains: %s\n' "$symbol" >&2
    exit 1
  fi
done

printf 'weight_store hybrid foundation boundary: passed\n'
