#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/tmp/hipfire-lds-gemm-isa-compare}"
ARCH="${ARCH:-gfx1103}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
READOBJ="${READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}"
OBJDUMP="${OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}"
SRC="${SRC:-$ROOT/lds_gemm_standalone_probe.hip}"

DEFAULT_SYMBOLS=(
    "_Z72gemm_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned_probeILi6EEviiii"
    "_Z54gemm_lds_tail_snop_noextra_load4_consume4_pinned_probeILi6EEviiii"
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

(
    cd "$OUT"
    "$HIPCC" -O3 --offload-arch="$ARCH" -save-temps=obj \
        "$SRC" -o "$bin" >"$OUT/build.log" 2>&1
)

find "$OUT" -maxdepth 2 -type f \( -name '*.hsaco' -o -name '*.o' -o -name '*.s' -o -name '*.ll' \) \
    >"$OUT/generated-files.txt" 2>/dev/null || true

while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    cp "$f" "$temps/$base" 2>/dev/null || true
    if file "$f" | grep -qi ELF; then
        "$READOBJ" --notes --sections --symbols "$f" \
            >"$temps/$base.readobj.txt" 2>&1 || true
        "$OBJDUMP" -d --mcpu="$ARCH" "$f" \
            >"$temps/$base.isa.txt" 2>&1 || true
    fi
done <"$OUT/generated-files.txt"

obj="$(find "$OUT" -maxdepth 2 -type f -name "*hip-amdgcn-amd-amdhsa-${ARCH}.o" | head -1)"
if [[ -z "$obj" ]]; then
    echo "no amdgpu object found under $OUT" >&2
    exit 1
fi

obj_base="$(basename "$obj")"
readobj_txt="$temps/$obj_base.readobj.txt"
isa_txt="$temps/$obj_base.isa.txt"

if [[ ! -s "$readobj_txt" || ! -s "$isa_txt" ]]; then
    echo "missing readobj/isa dump for $obj" >&2
    exit 1
fi

if [[ -n "${SYMBOLS:-}" ]]; then
    read -r -a symbols <<<"$SYMBOLS"
else
    symbols=("${DEFAULT_SYMBOLS[@]}")
fi

function_value_after_name() {
    local symbol="$1"
    local key="$2"
    awk -v symbol="$symbol" -v key="$key" '
        $0 ~ "Name: " symbol " \\(" { in_symbol = 1; next }
        in_symbol && $1 == key ":" { print $2; exit }
        in_symbol && /^    Name:/ { exit }
    ' "$readobj_txt"
}

metadata_value_near_name() {
    local symbol="$1"
    local key="$2"
    local line
    line="$(rg -n "\\.name: +${symbol}$" "$readobj_txt" | cut -d: -f1 | head -1)"
    if [[ -z "$line" ]]; then
        return 0
    fi
    sed -n "$((line - 30)),$((line + 20))p" "$readobj_txt" |
        awk -v key="$key" '$1 == "." key ":" { print $2; exit }'
}

extract_section() {
    local symbol="$1"
    local section="$2"
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
    rg -c "$pattern" "$section" || true
}

summary="$OUT/isa-summary.tsv"
printf 'symbol\tsize\tinstructions\ts_nop\tds_store\tds_load\ts_barrier\ts_cbranch\tgroup_segment\tprivate_segment\tsgpr\tvgpr\twavefront\tsection\n' >"$summary"

for symbol in "${symbols[@]}"; do
    safe="${symbol//[^A-Za-z0-9_]/_}"
    section="$sections/$safe.isa.txt"
    if ! extract_section "$symbol" "$section"; then
        echo "missing ISA section for symbol: $symbol" >&2
        exit 1
    fi

    keyops="$sections/$safe.key-ops.txt"
    rg -n "s_nop|ds_store|ds_load|s_waitcnt|s_barrier|buffer_gl0_inv|s_add_i32|s_cmp_ge_i32|s_cbranch" \
        "$section" >"$keyops" || true

    size="$(function_value_after_name "$symbol" "Size")"
    instructions="$(count_op '^\s+[A-Za-z_].*// 000000' "$section")"
    snop="$(count_op '^\s+s_nop' "$section")"
    ds_store="$(count_op '^\s+ds_store' "$section")"
    ds_load="$(count_op '^\s+ds_load' "$section")"
    s_barrier="$(count_op '^\s+s_barrier' "$section")"
    s_cbranch="$(count_op '^\s+s_cbranch' "$section")"
    group_segment="$(metadata_value_near_name "$symbol" "group_segment_fixed_size")"
    private_segment="$(metadata_value_near_name "$symbol" "private_segment_fixed_size")"
    sgpr="$(metadata_value_near_name "$symbol" "sgpr_count")"
    vgpr="$(metadata_value_near_name "$symbol" "vgpr_count")"
    wavefront="$(metadata_value_near_name "$symbol" "wavefront_size")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$symbol" "$size" "$instructions" "$snop" "$ds_store" "$ds_load" \
        "$s_barrier" "$s_cbranch" "$group_segment" "$private_segment" \
        "$sgpr" "$vgpr" "$wavefront" "$section" >>"$summary"
done

{
    echo "object=$obj"
    echo "readobj=$readobj_txt"
    echo "isa=$isa_txt"
    echo "summary=$summary"
} >"$OUT/summary.txt"

cat "$OUT/summary.txt"
cat "$summary"
