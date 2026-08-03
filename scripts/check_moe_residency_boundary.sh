#!/usr/bin/env bash
# check_moe_residency_boundary.sh — Phase C MoE residency + Qwen Frozen boundaries
#
# Assertions:
#
# 1. Rejected ownership vocabulary is absent from Rust sources under crates/:
#    the ExpertShard* family (reset boundary), the raw-pointer `WeightStoreView`,
#    ownership booleans (`store_owned`), `EpArch::Qwen35`, and the forgeable
#    `WeightCellId::for_test` constructor.
#
# 2. The Qwen35 ID-only projection is ID-only: `Qwen35MoeResident` and
#    `MoeFfnBindings` struct fields carry no `GpuTensor` / `WeightTensor`.
#
# 3. The Frozen staging/publication path never reconstructs raw ownership:
#    no `DeviceBuffer::from_raw` and no cloneable `.alias()` view inside
#    `build_frozen_moe_resident`, `impl Qwen35MoeResident`, or any
#    `impl MoeFfnBindings` region.  `DeviceBuffer::from_raw` may appear only
#    inside the whitelisted legacy-domain regions
#    (`convert_handle_forward_ready`, `free_resident_buffer_retaining_owner`,
#    `assemble_qwen35_weights_inner_with_mode`) or in doc comments.
#
# 4. No public raw allocation / adoption / typed-free-authority exposure in the
#    qwen35 crate: `WeightStoreAllocation` never appears there, and no `pub fn`
#    signature exposes `SingleFrozenWeightStore`, `SingleWeightStoreBuilder`,
#    `WeightStoreAllocation`, or an adopt/alloc_raw/into_raw/take_raw/leak
#    surface.
#
# 5. Frozen is refused at every multi-device entry: `reject_frozen_multi` is
#    defined and is called from `forward_scratch_layers_multi`,
#    `forward_scratch_multi`, and `forward_prefill_batch_multi`.
#
# 6. Qwen35 EP remains Planned/refused before allocation: `EpArchKind` has no
#    Qwen35 variant, `validate_ep_layout` refuses non-DS4/MiniMax architectures,
#    and the capability matrix keeps `(Qwen35Moe, Ep) => Planned` owned by
#    AXIS-002 with no Admitted/NormalizeToSingle row.
#
# Usage: bash scripts/check_moe_residency_boundary.sh [--self-test]
#   --self-test runs every assertion against a controlled violation fixture in
#   a temporary directory, asserts each expected violation category
#   independently (not just a nonzero failure total), and exits 0 only when
#   every expected category was caught.
#
# The script fails closed at startup if any required tool (rg, git, awk, sed,
# grep, cut, head, dirname; plus mktemp/rm in --self-test mode) is missing,
# so a missing `rg` can never make the from_raw whitelist check pass
# silently.
set -euo pipefail

# ── Fail-closed tool prerequisites (before anything else: a missing tool
# must never let a check pass silently, e.g. rg-127 in the from_raw
# whitelist) ─────────────────────────────────────────────────────────────
required_tools=(rg git awk sed grep cut head dirname)
missing_tools=()
for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        missing_tools+=("$tool")
    fi
done
if ((${#missing_tools[@]} > 0)); then
    printf 'MOE residency boundary check failed: required tool(s) missing: %s\n' \
        "${missing_tools[*]}" >&2
    exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

mode=tree
if [[ ${1:-} == "--self-test" ]]; then
    mode=self-test
fi

case "$mode" in
    tree)
        ROOT="$repo_root"
        ;;
    self-test)
        for tool in mktemp rm; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                printf 'MOE residency boundary check failed: required tool(s) missing: %s\n' \
                    "$tool" >&2
                exit 1
            fi
        done
        ROOT=$(mktemp -d)
        # Populate a controlled violation fixture: every assertion group is
        # violated so the self-test proves the checks are not vacuous.  Each
        # expected category is asserted independently afterwards.
        mkdir -p "$ROOT/crates/hipfire-arch-qwen35/src" "$ROOT/crates/hipfire-loader/src"
        cat > "$ROOT/crates/hipfire-arch-qwen35/src/store.rs" <<'FIXTURE_EOF'
// deliberate boundary-violation fixture for --self-test
pub struct Qwen35MoeResident {
    store: SingleFrozenWeightStore,
    raw_tensor: GpuTensor,
}

pub(crate) fn build_frozen_moe_resident() -> Result<(), ()> {
    let raw = unsafe { DeviceBuffer::from_raw(ptr, size) };
    let view = GpuTensor { buf: unsafe { t.buf.alias() }, shape: vec![], dtype: DType::F32 };
    let _ = (raw, view);
    Ok(())
}

struct WeightStoreView {
    ptr: *mut u8,
}

fn legacy_domain() {
    let store_owned = true;
    let _ = store_owned;
    let _ = EpArch::Qwen35;
}

// Check 4 triggers: typed free authority in the crate, and public raw
// ownership/adoption/free-authority signatures.
pub fn freeze_handle() -> SingleFrozenWeightStore {
    unimplemented!()
}

pub fn adopt(t: WeightStoreAllocation) {
    let _ = t;
}

pub fn leak(x: WeightStoreAllocation) {
    let _ = x;
}
FIXTURE_EOF
        cat > "$ROOT/crates/hipfire-arch-qwen35/src/vocab_fixture.rs" <<'FIXTURE_EOF'
// deliberate boundary-violation fixture: reset-boundary vocabulary
struct ExpertShardResident;
struct ExpertShardAssembly;
struct ExpertShardResourceKind;
struct ExpertShardResource;
struct ExpertShardTarget;
struct ExpertShardSlot;
fn forge() -> WeightCellId {
    WeightCellId::for_test(0)
}
FIXTURE_EOF
        cat > "$ROOT/crates/hipfire-arch-qwen35/src/qwen35.rs" <<'FIXTURE_EOF'
fn forward_scratch_layers_multi() {
    // deliberately missing reject_frozen_multi call
}

pub fn forward_scratch_multi() {
    // deliberately missing reject_frozen_multi call
}

pub fn forward_prefill_batch_multi() {
    // deliberately missing reject_frozen_multi call
}
FIXTURE_EOF
        cat > "$ROOT/crates/hipfire-loader/src/lib.rs" <<'FIXTURE_EOF'
enum EpArchKind {
    Ds4,
    Minimax,
    Qwen35,
}

fn validate_ep_layout() -> Result<(), ()> {
    Err(())
}
FIXTURE_EOF
        cat > "$ROOT/crates/hipfire-loader/src/parallel_capability.rs" <<'FIXTURE_EOF'
(Qwen35Moe, Ep) => Admitted,
FIXTURE_EOF
        git -C "$ROOT" init -q
        git -C "$ROOT" add -A
        ;;
esac

# Self-test capture file for per-category assertion (created after the fixture
# so the EXIT trap can clean both).
capture=""
if [[ "$mode" == "self-test" ]]; then
    capture=$(mktemp)
    trap 'rm -rf "$ROOT" "$capture"' EXIT
fi

found=0

fail() {
    printf 'MOE residency boundary check failed: %s\n' "$1" >&2
    found=$((found + 1))
}

# git_grep_symbol <symbol>: literal search of tracked Rust sources under
# crates/ of $ROOT (working-tree content).  Exit 0 = found, 1 = absent,
# other = error.
git_grep_symbol() {
    git -C "$ROOT" grep -n -F -e "$1" -- ':(glob)crates/**/*.rs'
}

# region_span <file> <start_line>: prints "start end" line numbers of the
# brace-balanced region beginning at start_line (comment-only lines are
# skipped for brace accounting).  A multi-line signature is supported: the
# region closes only after an opening brace has been seen and depth returns
# to zero.  Empty output on failure.
region_span() {
    local file=$1 start=$2
    awk -v start="$start" '
        NR < start { next }
        {
            line = $0
            sub(/^[[:space:]]*/, "", line)
            if (line ~ /^\/\//) { next }
            depth += gsub(/\{/, "{")
            depth -= gsub(/\}/, "}")
            if (depth > 0) { opened = 1 }
            if (NR == start) { first = NR }
            if (opened && depth <= 0) { print first, NR; exit }
        }
    ' "$file"
}

# region_lines <file> <header_regex>: line numbers of every region header
# matching header_regex (extended regex, whole-line anchored by caller).
# Never fails under `set -e`: a missing header yields empty output.
region_lines() {
    local file=$1 header=$2
    grep -nE "$header" "$file" | cut -d: -f1 || true
}

# region_missing <file> <header_regex> <label>: fail unless the region header
# exists; returns the start line(s) on stdout.
require_region() {
    local file=$1 header=$2 label=$3
    local lines
    lines=$(region_lines "$file" "$header")
    if [[ -z "$lines" ]]; then
        fail "expected region header '$header' ($label) not found in $file"
        return 1
    fi
    printf '%s\n' "$lines"
}

# check_region_absent <file> <symbol> <header_regex> <label>...
# For every region whose header matches, assert the symbol is absent inside.
check_region_absent() {
    local file=$1 symbol=$2
    shift 2
    local header label span start end body
    while (($# >= 2)); do
        header=$1
        label=$2
        shift 2
        while read -r start; do
            [[ -n "$start" ]] || continue
            span=$(region_span "$file" "$start")
            if [[ -z "$span" ]]; then
                fail "region extraction failed for '$header' ($label) at $file:$start (script bug?)"
                continue
            fi
            end=${span#* }
            if ! sed -n "${end}p" "$file" | grep -q '}'; then
                fail "region extraction sanity check failed for '$header' ($label) at $file:$start (script bug?)"
                continue
            fi
            body=$(sed -n "${start},${end}p" "$file")
            if grep -qF "$symbol" <<<"$body"; then
                fail "$symbol appears inside $label region ($header) at $file:$start"
            fi
        done < <(require_region "$file" "$header" "$label" || true)
    done
}

# check_region_contains <file> <symbol> <header_regex> <label>...
# For every region whose header matches, assert the symbol appears inside.
check_region_contains() {
    local file=$1 symbol=$2
    shift 2
    local header label span start end body
    while (($# >= 2)); do
        header=$1
        label=$2
        shift 2
        while read -r start; do
            [[ -n "$start" ]] || continue
            span=$(region_span "$file" "$start")
            if [[ -z "$span" ]]; then
                fail "region extraction failed for '$header' ($label) at $file:$start (script bug?)"
                continue
            fi
            end=${span#* }
            if ! sed -n "${end}p" "$file" | grep -q '}'; then
                fail "region extraction sanity check failed for '$header' ($label) at $file:$start (script bug?)"
                continue
            fi
            body=$(sed -n "${start},${end}p" "$file")
            if ! grep -qF "$symbol" <<<"$body"; then
                fail "$symbol missing from $label region ($header) at $file:$start"
            fi
        done < <(require_region "$file" "$header" "$label" || true)
    done
}

# check_from_raw_whitelist <file> <header_regex> <label> [<header_regex> <label>...]
# Every `DeviceBuffer::from_raw` hit in <file> must be a doc-comment line or
# fall inside one of the whitelisted legacy regions.
check_from_raw_whitelist() {
    local file=$1
    shift
    local whitelisted=()
    local header label
    while (($# >= 2)); do
        whitelisted+=("$1")
        shift 2
    done
    local hits hitline text start end in_region span
    local status=0
    hits=$(rg -n -F "DeviceBuffer::from_raw" "$file" 2>/dev/null) || status=$?
    if [[ $status -eq 127 ]]; then
        fail "rg is not available; cannot verify the DeviceBuffer::from_raw whitelist in $file"
        return 0
    fi
    if [[ $status -eq 2 ]]; then
        fail "rg failed while scanning $file for DeviceBuffer::from_raw (exit 2); whitelist unverified"
        return 0
    fi
    if [[ -z "$hits" ]]; then
        return 0
    fi
    while IFS=: read -r hitline _rest; do
        text=$(sed -n "${hitline}p" "$file")
        if [[ "$text" =~ ^[[:space:]]*/// ]]; then
            continue
        fi
        in_region=0
        for header in "${whitelisted[@]}"; do
            start=$(region_lines "$file" "$header" | head -1)
            if [[ -z "$start" ]]; then
                continue
            fi
            span=$(region_span "$file" "$start")
            [[ -n "$span" ]] || continue
            end=${span#* }
            if (( hitline >= start && hitline <= end )); then
                in_region=1
                break
            fi
        done
        if [[ $in_region -ne 1 ]]; then
            fail "DeviceBuffer::from_raw at $file:$hitline is outside the whitelisted legacy regions"
        fi
    done <<<"$hits"
}

# ── run_checks: all assertion groups ────────────────────────────────────
run_checks() {
    # ── Check 1: rejected ownership vocabulary absent (crates/**/*.rs) ────
    for symbol in \
        ExpertShardResident \
        ExpertShardAssembly \
        ExpertShardResourceKind \
        ExpertShardResource \
        ExpertShardTarget \
        ExpertShardSlot \
        WeightStoreView \
        store_owned \
        EpArch::Qwen35 \
        WeightCellId::for_test; do
        status=0
        matches=$(git_grep_symbol "$symbol" 2>/dev/null) || status=$?
        if [[ $status -eq 0 ]]; then
            fail "$symbol remains in Rust sources under crates/:\n$matches"
        elif [[ $status -ne 1 ]]; then
            fail "git grep exited with status $status while checking $symbol"
        fi
    done

    # ── Check 2: ID-only projection (no GpuTensor/WeightTensor fields) ────
    local qwen_store="$ROOT/crates/hipfire-arch-qwen35/src/store.rs"
    local qwen_weights="$ROOT/crates/hipfire-arch-qwen35/src/qwen35.rs"
    local loader_lib="$ROOT/crates/hipfire-loader/src/lib.rs"
    local capability="$ROOT/crates/hipfire-loader/src/parallel_capability.rs"
    local qwen_crate="$ROOT/crates/hipfire-arch-qwen35/src"

    check_region_absent "$qwen_store" "GpuTensor" \
        '^pub struct Qwen35MoeResident \{' "Qwen35MoeResident struct (ID-only projection)"
    check_region_absent "$qwen_store" "WeightTensor" \
        '^pub struct Qwen35MoeResident \{' "Qwen35MoeResident struct (ID-only projection)"
    check_region_absent "$qwen_store" "GpuTensor" \
        '^pub struct MoeFfnBindings' "MoeFfnBindings struct (borrowed bindings)"
    check_region_absent "$qwen_store" "WeightTensor" \
        '^pub struct MoeFfnBindings' "MoeFfnBindings struct (borrowed bindings)"

    # ── Check 3: Frozen staging/publication path purity ───────────────────
    check_region_absent "$qwen_store" "DeviceBuffer::from_raw" \
        '^pub\(crate\) fn build_frozen_moe_resident\(' "Frozen staging (build_frozen_moe_resident)"
    check_region_absent "$qwen_store" ".alias(" \
        '^pub\(crate\) fn build_frozen_moe_resident\(' "Frozen staging (build_frozen_moe_resident)"
    check_region_absent "$qwen_store" "DeviceBuffer::from_raw" \
        '^impl Qwen35MoeResident \{' "Frozen resident impl"
    check_region_absent "$qwen_store" ".alias(" \
        '^impl Qwen35MoeResident \{' "Frozen resident impl"
    check_region_absent "$qwen_store" "DeviceBuffer::from_raw" \
        '^impl(<[^>]*>)? MoeFfnBindings' "MoeFfnBindings impls"
    check_region_absent "$qwen_store" ".alias(" \
        '^impl(<[^>]*>)? MoeFfnBindings' "MoeFfnBindings impls"

    # Every remaining DeviceBuffer::from_raw in the qwen35 crate must be
    # inside a whitelisted legacy-domain region.
    check_from_raw_whitelist "$qwen_store" \
        '^fn convert_handle_forward_ready\(' "legacy conversion" \
        '^fn free_resident_buffer_retaining_owner\(' "legacy retaining-owner free" \
        '^pub\(crate\) fn assemble_qwen35_weights_inner_with_mode\(' "legacy common assembly"

    # ── Check 4: no public raw allocation / adoption / token exposure ─────
    status=0
    matches=$(rg -n -F "WeightStoreAllocation" "$qwen_crate" 2>/dev/null) || status=$?
    if [[ $status -eq 0 ]]; then
        fail "typed free authority WeightStoreAllocation appears in the qwen35 crate:\n$matches"
    elif [[ $status -ne 1 ]]; then
        fail "rg exited with status $status while checking WeightStoreAllocation in the qwen35 crate"
    fi
    status=0
    matches=$(rg -n -e 'pub fn [^(]*\b(SingleFrozenWeightStore|SingleWeightStoreBuilder|WeightStoreAllocation)\b' \
        -e 'pub fn (adopt|alloc_raw|into_raw|take_raw|leak)\b' "$qwen_crate" 2>/dev/null) || status=$?
    if [[ $status -eq 0 ]]; then
        fail "public raw ownership surface in the qwen35 crate:\n$matches"
    elif [[ $status -ne 1 ]]; then
        fail "rg exited with status $status while checking the qwen35 public ownership surface"
    fi

    # ── Check 5: Frozen refused at every multi-device entry ───────────────
    require_region "$qwen_weights" '^pub\(crate\) fn reject_frozen_multi\(' \
        "multi-device Frozen refusal" >/dev/null || true
    check_region_contains "$qwen_weights" "reject_frozen_multi(" \
        '^fn forward_scratch_layers_multi\(' "multi-device entry forward_scratch_layers_multi"
    check_region_contains "$qwen_weights" "reject_frozen_multi(" \
        '^pub fn forward_scratch_multi\(' "multi-device entry forward_scratch_multi"
    check_region_contains "$qwen_weights" "reject_frozen_multi(" \
        '^pub fn forward_prefill_batch_multi\(' "multi-device entry forward_prefill_batch_multi"

    # ── Check 6: Qwen35 EP remains Planned/refused before allocation ──────
    check_region_absent "$loader_lib" "Qwen35" '^enum EpArchKind \{' "EpArchKind enum (no Qwen35 variant)"
    check_region_contains "$loader_lib" "unsupported EP architecture" \
        '^fn validate_ep_layout\(' "EP layout refusal for unsupported architectures"

    status=0
    matches=$(rg -n -F '(Qwen35Moe, Ep) => Admitted' "$capability" 2>/dev/null) || status=$?
    if [[ $status -eq 0 ]]; then
        fail "Qwen35 MoE EP is Admitted in the capability matrix:\n$matches"
    elif [[ $status -ne 1 ]]; then
        fail "rg exited with status $status while checking the capability matrix"
    fi
    status=0
    matches=$(rg -n -F '(Qwen35Moe, Ep) => NormalizeToSingle' "$capability" 2>/dev/null) || status=$?
    if [[ $status -eq 0 ]]; then
        fail "Qwen35 MoE EP normalizes to single in the capability matrix (must stay Planned):\n$matches"
    elif [[ $status -ne 1 ]]; then
        fail "rg exited with status $status while checking the capability matrix"
    fi

    ep_row=$(rg -n -F '(Qwen35Moe, Ep) => Planned {' "$capability" 2>/dev/null | head -1 || true)
    if [[ -z "$ep_row" ]]; then
        fail "capability matrix is missing the '(Qwen35Moe, Ep) => Planned' refusal row"
    else
        ep_line=${ep_row%%:*}
        owner_lines=$(sed -n "$((ep_line + 1)),$((ep_line + 8))p" "$capability")
        if ! grep -qF 'owner: "AXIS-002"' <<<"$owner_lines"; then
            fail "capability matrix '(Qwen35Moe, Ep)' row is not owned by AXIS-002"
        fi
    fi
}

# ── Result ────────────────────────────────────────────────────────────────
if [[ "$mode" == "tree" ]]; then
    run_checks
    if [[ $found -ne 0 ]]; then
        printf 'MOE residency boundary check failed with %d assertion failure(s); see messages above.\n' "$found"
        exit 1
    fi
    printf 'MOE residency boundary check passed: no forbidden ownership symbols in tracked Rust sources under crates/.\n'
    printf 'Also passed: ID-only projection fields, Frozen staging-path purity (no from_raw/alias),\n'
    printf 'from_raw legacy whitelist, no public ownership-surface exposure, multi-device Frozen refusal,\n'
    printf 'and Qwen35 EP Planned/refused admission (no EpArch::Qwen35, no daemon admission).\n'
    exit 0
fi

# ── Self-test: per-category assertion ────────────────────────────────────
# Each expected category is an entry "label|pattern|pattern|..." ; every
# pattern must appear in the captured failure output for the category to
# count as caught.  Exit 0 only when ALL categories were caught.
expected_categories=(
    "expert-shard-family|ExpertShardResident remains|ExpertShardAssembly remains|ExpertShardResourceKind remains|ExpertShardResource remains|ExpertShardTarget remains|ExpertShardSlot remains"
    "for-test-token|WeightCellId::for_test remains"
    "raw-view-vocab|WeightStoreView remains"
    "ownership-boolean|store_owned remains"
    "ep-arch-symbol|EpArch::Qwen35 remains"
    "id-only-projection|GpuTensor appears inside Qwen35MoeResident struct"
    "frozen-from-raw|DeviceBuffer::from_raw appears inside Frozen staging"
    "frozen-alias|\.alias\( appears inside Frozen staging"
    "from-raw-whitelist|outside the whitelisted legacy regions"
    "check4-token|typed free authority WeightStoreAllocation appears in the qwen35 crate"
    "check4-surface|public raw ownership surface in the qwen35 crate"
    "multi-entry-refusal|missing from multi-device entry"
    "ep-arch-kind|Qwen35 appears inside EpArchKind enum"
    "ep-layout-refusal|unsupported EP architecture missing from"
    "ep-matrix-admitted|Qwen35 MoE EP is Admitted in the capability matrix"
    "ep-matrix-planned|capability matrix is missing the '\(Qwen35Moe, Ep\) => Planned' refusal row"
)

run_checks 2> "$capture"

uncaught=0
for entry in "${expected_categories[@]}"; do
    IFS='|' read -r -a parts <<<"$entry"
    label=${parts[0]}
    missing=0
    for ((i = 1; i < ${#parts[@]}; i++)); do
        if ! grep -qE "${parts[$i]}" "$capture"; then
            printf 'MOE residency boundary self-test: category "%s" not caught (missing pattern: %s)\n' \
                "$label" "${parts[$i]}"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        uncaught=$((uncaught + 1))
    fi
done

if [[ $uncaught -eq 0 ]]; then
    printf 'MOE residency boundary self-test passed: %d assertion failure(s) total; all %d expected category/categories caught.\n' \
        "$found" "${#expected_categories[@]}"
    exit 0
fi

printf 'MOE residency boundary self-test failed: %d of %d expected categories were not caught.\n' \
    "$uncaught" "${#expected_categories[@]}"
exit 1
