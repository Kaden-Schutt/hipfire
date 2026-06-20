#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/tmp/hipfire-lds-gemm-isa-compare}"
ARCH="${ARCH:-gfx1103}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
READOBJ="${READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}"
OBJDUMP="${OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}"
SRC="${SRC:-$ROOT/lds_gemm_standalone_probe.hip}"
SINGLE_INSTANTIATION="${SINGLE_INSTANTIATION:-0}"

DEFAULT_SYMBOLS=(
    "_Z72gemm_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned_probeILi6EEviiii"
    "_Z52gemm_lds_counter_noextra_load4_consume4_pinned_probeILi6EEviiii"
    "_Z49gemm_lds_snop_noextra_load4_consume4_pinned_probeILi6EEviiii"
    "_Z54gemm_lds_tail_snop_noextra_load4_consume4_pinned_probeILi6EEviiii"
    "_Z57gemm_lds_counter_mask_noextra_load4_consume4_pinned_probeILi6EEviiii"
)

if [[ ! -f "$SRC" ]]; then
    echo "missing source: $SRC" >&2
    exit 1
fi

for tool in "$HIPCC" "$READOBJ" "$OBJDUMP"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing executable: $tool" >&2
        exit 1
    fi
done

if ! command -v rg >/dev/null 2>&1; then
    echo "missing executable: rg" >&2
    exit 1
fi

mkdir -p "$OUT"
bin="$OUT/lds_gemm_standalone_probe"
temps="$OUT/save-temps"
sections="$OUT/sections"
rm -rf "$temps" "$sections" "$OUT/single"
mkdir -p "$temps" "$sections"

{
    echo "source=$SRC"
    echo "arch=$ARCH"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hipcc=$HIPCC"
    echo "llvm-readobj=$READOBJ"
    echo "llvm-objdump=$OBJDUMP"
} >"$OUT/meta.txt"

cp "$SRC" "$OUT/lds_gemm_standalone_probe.hip"

if [[ -n "${SYMBOLS:-}" ]]; then
    read -r -a symbols <<<"$SYMBOLS"
else
    symbols=("${DEFAULT_SYMBOLS[@]}")
fi

variant_for_symbol() {
    case "$1" in
        "_Z72gemm_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned_probeILi6EEviiii")
            echo "gemm_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned_probe"
            ;;
        "_Z52gemm_lds_counter_noextra_load4_consume4_pinned_probeILi6EEviiii")
            echo "gemm_lds_counter_noextra_load4_consume4_pinned_probe"
            ;;
        "_Z49gemm_lds_snop_noextra_load4_consume4_pinned_probeILi6EEviiii")
            echo "gemm_lds_snop_noextra_load4_consume4_pinned_probe"
            ;;
        "_Z54gemm_lds_tail_snop_noextra_load4_consume4_pinned_probeILi6EEviiii")
            echo "gemm_lds_tail_snop_noextra_load4_consume4_pinned_probe"
            ;;
        "_Z57gemm_lds_counter_mask_noextra_load4_consume4_pinned_probeILi6EEviiii")
            echo "gemm_lds_counter_mask_noextra_load4_consume4_pinned_probe"
            ;;
        *)
            return 1
            ;;
    esac
}

emit_tool_dumps() {
    local root="$1"
    local prefix="$2"
    local generated="$root/generated-files.txt"

    find "$root" -maxdepth 2 -type f \( -name '*.hsaco' -o -name '*.o' -o -name '*.s' -o -name '*.ll' \) \
        >"$generated" 2>/dev/null || true

    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        rel="${f#$root/}"
        base="${prefix}__${rel//\//__}"
        cp "$f" "$temps/$base" 2>/dev/null || true
        if file "$f" | grep -qi ELF; then
            "$READOBJ" --notes --sections --symbols "$f" \
                >"$temps/$base.readobj.txt" 2>&1 || true
            "$OBJDUMP" -d --mcpu="$ARCH" "$f" \
                >"$temps/$base.isa.txt" 2>&1 || true
        fi
    done <"$generated"
}

if [[ "$SINGLE_INSTANTIATION" == "1" ]]; then
    : >"$OUT/object-list.txt"
    for symbol in "${symbols[@]}"; do
        kernel="$(variant_for_symbol "$symbol")" || {
            echo "no single-instantiation mapping for symbol: $symbol" >&2
            exit 1
        }
        safe="${symbol//[^A-Za-z0-9_]/_}"
        single_dir="$OUT/single/$safe"
        mkdir -p "$single_dir"
        single_src="$single_dir/single.hip"
        single_obj="$single_dir/single.o"
        cat >"$single_src" <<EOF
#define HIPFIRE_LDS_PROBE_NO_MAIN
#include "$SRC"
template __global__ void ${kernel}<6>(int, int, int, int);
EOF
        (
            cd "$single_dir"
            "$HIPCC" -O3 --offload-arch="$ARCH" -save-temps=obj -c \
                "$single_src" -o "$single_obj" >"$single_dir/build.log" 2>&1
        )
        emit_tool_dumps "$single_dir" "$safe"
        obj="$(find "$single_dir" -maxdepth 1 -type f -name "*hip-amdgcn-amd-amdhsa-${ARCH}.o" | head -1)"
        if [[ -z "$obj" ]]; then
            echo "no amdgpu object found under $single_dir" >&2
            exit 1
        fi
        printf '%s\t%s\n' "$symbol" "$obj" >>"$OUT/object-list.txt"
    done
else
    (
        cd "$OUT"
        "$HIPCC" -O3 --offload-arch="$ARCH" -save-temps=obj \
            "$SRC" -o "$bin" >"$OUT/build.log" 2>&1
    )
    emit_tool_dumps "$OUT" "full"
    obj="$(find "$OUT" -maxdepth 1 -type f -name "*hip-amdgcn-amd-amdhsa-${ARCH}.o" | head -1)"
    if [[ -z "$obj" ]]; then
        echo "no amdgpu object found under $OUT" >&2
        exit 1
    fi
    for symbol in "${symbols[@]}"; do
        printf '%s\t%s\n' "$symbol" "$obj"
    done >"$OUT/object-list.txt"
fi

function_value_after_name() {
    local readobj_txt="$1"
    local symbol="$2"
    local key="$3"
    awk -v symbol="$symbol" -v key="$key" '
        $0 ~ "Name: " symbol " \\(" { in_symbol = 1; next }
        in_symbol && $1 == key ":" { print $2; exit }
        in_symbol && /^    Name:/ { exit }
    ' "$readobj_txt"
}

metadata_value_near_name() {
    local readobj_txt="$1"
    local symbol="$2"
    local key="$3"
    local line
    line="$(rg -n "\\.name: +${symbol}$" "$readobj_txt" | cut -d: -f1 | head -1)"
    if [[ -z "$line" ]]; then
        return 0
    fi
    sed -n "$((line - 30)),$((line + 20))p" "$readobj_txt" |
        awk -v key="$key" '$1 == "." key ":" { print $2; exit }'
}

extract_section() {
    local isa_txt="$1"
    local symbol="$2"
    local section="$3"
    local start next end
    start="$(rg -n "Disassembly of section \\.text\\.${symbol}:" "$isa_txt" | cut -d: -f1 | head -1)"
    if [[ -z "$start" ]]; then
        return 1
    fi
    next="$(tail -n +"$((start + 1))" "$isa_txt" | rg -n '^Disassembly of section' | head -1 | cut -d: -f1)"
    if [[ -z "$next" ]]; then
        end="$(wc -l <"$isa_txt")"
    else
        end="$((start + next - 1))"
    fi
    sed -n "${start},${end}p" "$isa_txt" >"$section"
}

count_op() {
    local pattern="$1"
    local section="$2"
    local count
    count="$(rg -c "$pattern" "$section" || true)"
    echo "${count:-0}"
}

summary="$OUT/isa-summary.tsv"
printf 'symbol\tsize\tinstructions\ts_nop\tds_store\tds_load\ts_barrier\ts_cbranch\tgroup_segment\tprivate_segment\tsgpr\tvgpr\twavefront\tsection\n' >"$summary"

while IFS=$'\t' read -r symbol obj; do
    safe="${symbol//[^A-Za-z0-9_]/_}"
    if [[ "$SINGLE_INSTANTIATION" == "1" ]]; then
        obj_base="${safe}__$(basename "$obj")"
    else
        obj_base="full__$(basename "$obj")"
    fi
    readobj_txt="$temps/$obj_base.readobj.txt"
    isa_txt="$temps/$obj_base.isa.txt"

    if [[ ! -s "$readobj_txt" || ! -s "$isa_txt" ]]; then
        echo "missing readobj/isa dump for $obj" >&2
        exit 1
    fi

    section="$sections/$safe.isa.txt"
    if ! extract_section "$isa_txt" "$symbol" "$section"; then
        echo "missing ISA section for symbol: $symbol" >&2
        exit 1
    fi

    keyops="$sections/$safe.key-ops.txt"
    rg -n "s_nop|ds_store|ds_load|s_waitcnt|s_barrier|buffer_gl0_inv|s_add_i32|s_cmp_ge_i32|s_cbranch" \
        "$section" >"$keyops" || true

    size="$(function_value_after_name "$readobj_txt" "$symbol" "Size")"
    instructions="$(count_op '^\s+[A-Za-z_].*// 000000' "$section")"
    snop="$(count_op '^\s+s_nop' "$section")"
    ds_store="$(count_op '^\s+ds_store' "$section")"
    ds_load="$(count_op '^\s+ds_load' "$section")"
    s_barrier="$(count_op '^\s+s_barrier' "$section")"
    s_cbranch="$(count_op '^\s+s_cbranch' "$section")"
    group_segment="$(metadata_value_near_name "$readobj_txt" "$symbol" "group_segment_fixed_size")"
    private_segment="$(metadata_value_near_name "$readobj_txt" "$symbol" "private_segment_fixed_size")"
    sgpr="$(metadata_value_near_name "$readobj_txt" "$symbol" "sgpr_count")"
    vgpr="$(metadata_value_near_name "$readobj_txt" "$symbol" "vgpr_count")"
    wavefront="$(metadata_value_near_name "$readobj_txt" "$symbol" "wavefront_size")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$symbol" "$size" "$instructions" "$snop" "$ds_store" "$ds_load" \
        "$s_barrier" "$s_cbranch" "$group_segment" "$private_segment" \
        "$sgpr" "$vgpr" "$wavefront" "$section" >>"$summary"
done <"$OUT/object-list.txt"

{
    echo "single_instantiation=$SINGLE_INSTANTIATION"
    echo "object_list=$OUT/object-list.txt"
    echo "summary=$summary"
} >"$OUT/summary.txt"

cat "$OUT/summary.txt"
cat "$summary"
