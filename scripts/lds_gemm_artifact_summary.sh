#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/tmp/hipfire-lds-tail-snop-780m}"
OUT_PREFIX="${2:-$ROOT/lds-artifact-summary}"
TSV="${OUT_PREFIX}.tsv"
MD="${OUT_PREFIX}.md"

if [[ ! -d "$ROOT" ]]; then
    echo "missing artifact root: $ROOT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_PREFIX")"

meta_value() {
    local key="$1"
    local file="$2"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

meta_contains_value() {
    local pattern="$1"
    local file="$2"
    rg -m1 "$pattern" "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true
}

sanitize() {
    tr '\t\n' '  ' | sed 's/[[:space:]][[:space:]]*/ /g;s/^ //;s/ $//'
}

dmesg_delta_count() {
    local pattern="$1"
    local before="$2"
    local after="$3"

    if [[ ! -r "$after" ]]; then
        echo 0
        return
    fi
    if [[ ! -r "$before" ]]; then
        rg -c "$pattern" "$after" || true
        return
    fi
    awk 'NR == FNR { seen[$0]++; next } seen[$0] > 0 { seen[$0]--; next } { print }' "$before" "$after" |
        rg -c "$pattern" || true
}

short_sha256() {
    local path="$1"
    if [[ -r "$path" ]]; then
        sha256sum "$path" | awk '{ print substr($1, 1, 16) }'
    fi
}

normalized_isa_sha256() {
    local path="$1"
    if [[ -r "$path" ]]; then
        sed '/file format/d' "$path" | sha256sum | awk '{ print substr($1, 1, 16) }'
    fi
}

selected_isa_symbol() {
    local variant="$1"
    local tile suffix

    case "$variant" in
        tile5|tile6|tile16)
            tile="${variant#tile}"
            echo "gemm_lds_probeILi${tile}"
            ;;
        tile[0-9]*_*)
            suffix="${variant#tile[0-9]*_}"
            echo "gemm_${suffix}_probe"
            ;;
    esac
}

selected_isa_sha256() {
    local path="$1"
    local symbol="$2"
    local section
    if [[ -r "$path" && -n "$symbol" ]]; then
        section="$(awk -v symbol="$symbol" '
            /^Disassembly of section / {
                emit = index($0, symbol) > 0;
            }
            emit { print }
        ' "$path")"
        if [[ -n "$section" ]]; then
            printf '%s\n' "$section" | sha256sum | awk '{ print substr($1, 1, 16) }'
        fi
    fi
}

devcore_contains() {
    local pattern="$1"
    local path="$2"
    if [[ -r "$path" ]] && rg -a -q "$pattern" "$path"; then
        echo 1
    else
        echo 0
    fi
}

devcore_colon_value() {
    local pattern="$1"
    local path="$2"
    if [[ -r "$path" ]]; then
        rg -a -m1 "$pattern" "$path" | sed 's/.*:[[:space:]]*//' | sanitize || true
    fi
}

devcore_reg_value() {
    local reg="$1"
    local path="$2"
    if [[ -r "$path" ]]; then
        rg -a -m1 "^${reg}[[:space:]]+" "$path" | awk '{ print $NF }' | sanitize || true
    fi
}

hex_to_dec() {
    local value="$1"
    if [[ "$value" =~ ^0[xX][0-9a-fA-F]+$ ]]; then
        printf '%u\n' "$((value))"
    fi
}

join_flags() {
    local joined=""
    local flag
    for flag in "$@"; do
        [[ -n "$flag" ]] || continue
        if [[ -n "$joined" ]]; then
            joined="${joined},${flag}"
        else
            joined="$flag"
        fi
    done
    printf '%s\n' "$joined"
}

gds_fault_flags() {
    local raw
    local flags=()
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    ((raw & 0x1)) && flags+=("WRITE_DIS")
    ((raw & 0x2)) && flags+=("FAULT_DETECTED")
    ((raw & 0x4)) && flags+=("GRBM")
    join_flags "${flags[@]}"
}

gds_vm_fault_flags() {
    local raw
    local flags=()
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    ((raw & 0x1)) && flags+=("WRITE_DIS")
    ((raw & 0x2)) && flags+=("FAULT_DETECTED")
    ((raw & 0x4)) && flags+=("GWS")
    ((raw & 0x8)) && flags+=("OA")
    ((raw & 0x10)) && flags+=("GRBM")
    ((raw & 0x20)) && flags+=("TMZ")
    join_flags "${flags[@]}"
}

gds_fault_addr() {
    local raw
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    printf '0x%x\n' "$(((raw & 0xfffc0000) >> 18))"
}

gds_vm_fault_vmid() {
    local raw
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    printf '%u\n' "$(((raw & 0x00000f00) >> 8))"
}

gds_vm_fault_addr() {
    local raw
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    printf '0x%x\n' "$(((raw & 0xffff0000) >> 16))"
}

first_artifact_file() {
    local dir="$1"
    local name="$2"
    {
        if [[ -d "$dir/save-temps" ]]; then
            find "$dir/save-temps" -maxdepth 1 -type f -name "$name"
        fi
        find "$dir" -maxdepth 1 -type f -name "$name"
    } 2>/dev/null | sort | head -1 || true
}

printf 'artifact\tvariant\tmode\tlaunches\tshape\tk_limit\tarch\tbuild_only\texit\tsync_failure\thip_error\tgit_commit\tgit_dirty\thipcc\tdriver\tgpu\tsource_sha256\tamdgpu_obj_sha256\tamdgpu_isa_sha256\tdmesg_remove_queue\tdmesg_mode2\tdmesg_gds\tdevcoredump\tisa_files\tamdgpu_isa_norm_sha256\tselected_isa_symbol\tselected_isa_norm_sha256\tdevcore_gfxhub_page_fault\tdevcore_fault_addr\tdevcore_prot_status\tdevcore_gds_protection_fault\tdevcore_gds_vm_protection_fault\tdevcore_gds_flags\tdevcore_gds_addr\tdevcore_gds_vm_flags\tdevcore_gds_vm_vmid\tdevcore_gds_vm_addr\n' >"$TSV"

while IFS= read -r meta; do
    dir="$(dirname "$meta")"
    rel="${dir#$ROOT/}"
    [[ "$rel" == "$dir" ]] && rel="."

    run_log="$dir/run.log"
    exit_file="$dir/exit_code.txt"
    dmesg_before="$dir/dmesg.before.txt"
    dmesg_after="$dir/dmesg.after.txt"

    variant="$(meta_value variant "$meta" | sanitize)"
    mode="$(meta_value mode "$meta" | sanitize)"
    launches="$(meta_value launches "$meta" | sanitize)"
    shape="$(meta_value shape "$meta" | sanitize)"
    k_limit="$(meta_value k_limit "$meta" | sanitize)"
    arch="$(meta_value arch "$meta" | sanitize)"
    build_only="$(meta_value build_only "$meta" | sanitize)"
    exit_code="$([[ -r "$exit_file" ]] && tr -d '\n\r\t ' <"$exit_file" || true)"

    sync_failure=""
    hip_error=""
    if [[ -r "$run_log" ]]; then
        sync_failure="$(rg -m1 'sync [0-9]+ failed:' "$run_log" | sanitize || true)"
        hip_error="$(rg -o -m1 '\([0-9]+\)' "$run_log" | tr -d '()' || true)"
    fi

    git_commit="$(meta_value git_commit "$meta" | cut -c1-12 | sanitize)"
    git_status="$(meta_value git_status_short "$meta" | sanitize)"
    git_dirty=0
    [[ -n "$git_status" ]] && git_dirty=1

    hipcc="$(meta_value hipcc "$meta" | sanitize)"
    driver="$(meta_contains_value 'Driver version:' "$meta" | sed 's/.*Driver version:[[:space:]]*//' | sanitize)"
    gpu="$(meta_contains_value 'Marketing Name:' "$meta" | sed 's/.*Marketing Name:[[:space:]]*//' | sanitize)"

    dmesg_remove_queue=0
    dmesg_mode2=0
    dmesg_gds=0
    dmesg_remove_queue="$(dmesg_delta_count 'REMOVE_QUEUE|remove queue' "$dmesg_before" "$dmesg_after")"
    dmesg_mode2="$(dmesg_delta_count 'MODE2|mode2' "$dmesg_before" "$dmesg_after")"
    dmesg_gds="$(dmesg_delta_count 'GDS|regGDS' "$dmesg_before" "$dmesg_after")"
    dmesg_remove_queue="${dmesg_remove_queue:-0}"
    dmesg_mode2="${dmesg_mode2:-0}"
    dmesg_gds="${dmesg_gds:-0}"

    devcore="$dir/devcoredump.data"
    devcoredump=0
    [[ -s "$devcore" ]] && devcoredump=1
    devcore_gfxhub_pf="$(devcore_contains '\[gfxhub\] Page fault observed' "$devcore")"
    devcore_fault_addr="$(devcore_colon_value '^Faulty page starting at address:' "$devcore")"
    devcore_prot_status="$(devcore_colon_value '^Protection fault status register:' "$devcore")"
    devcore_gds_pf="$(devcore_reg_value 'regGDS_PROTECTION_FAULT' "$devcore")"
    devcore_gds_vm_pf="$(devcore_reg_value 'regGDS_VM_PROTECTION_FAULT' "$devcore")"
    devcore_gds_flags="$(gds_fault_flags "$devcore_gds_pf")"
    devcore_gds_addr="$(gds_fault_addr "$devcore_gds_pf")"
    devcore_gds_vm_flags="$(gds_vm_fault_flags "$devcore_gds_vm_pf")"
    devcore_gds_vm_vmid="$(gds_vm_fault_vmid "$devcore_gds_vm_pf")"
    devcore_gds_vm_addr="$(gds_vm_fault_addr "$devcore_gds_vm_pf")"
    isa_files=0
    if [[ -d "$dir/save-temps" ]]; then
        isa_files="$(find "$dir/save-temps" -maxdepth 1 -type f -name '*.isa.txt' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    source_sha="$(short_sha256 "$dir/lds_gemm_standalone_probe.hip")"
    amdgpu_obj="$(first_artifact_file "$dir" '*hip-amdgcn-amd-amdhsa-*.o')"
    amdgpu_isa="$(first_artifact_file "$dir" '*hip-amdgcn-amd-amdhsa-*.o.isa.txt')"
    amdgpu_obj_sha="$(short_sha256 "$amdgpu_obj")"
    amdgpu_isa_sha="$(short_sha256 "$amdgpu_isa")"
    amdgpu_isa_norm_sha="$(normalized_isa_sha256 "$amdgpu_isa")"
    selected_symbol="$(selected_isa_symbol "$variant")"
    selected_isa_norm_sha="$(selected_isa_sha256 "$amdgpu_isa" "$selected_symbol")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rel" "$variant" "$mode" "$launches" "$shape" "$k_limit" "$arch" \
        "$build_only" "$exit_code" "$sync_failure" "$hip_error" "$git_commit" \
        "$git_dirty" "$hipcc" "$driver" "$gpu" "$source_sha" \
        "$amdgpu_obj_sha" "$amdgpu_isa_sha" "$dmesg_remove_queue" \
        "$dmesg_mode2" "$dmesg_gds" "$devcoredump" "$isa_files" \
        "$amdgpu_isa_norm_sha" "$selected_symbol" "$selected_isa_norm_sha" \
        "$devcore_gfxhub_pf" "$devcore_fault_addr" "$devcore_prot_status" \
        "$devcore_gds_pf" "$devcore_gds_vm_pf" "$devcore_gds_flags" \
        "$devcore_gds_addr" "$devcore_gds_vm_flags" "$devcore_gds_vm_vmid" \
        "$devcore_gds_vm_addr" >>"$TSV"
done < <(find "$ROOT" -type f -name meta.txt | sort)

{
    echo "# LDS GEMM Artifact Summary"
    echo
    echo "- root: \`$ROOT\`"
    echo "- generated: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- tsv: \`$TSV\`"
    echo
    echo "| artifact | variant | exit | sync failure | git | driver | gpu | src | obj | isa | isa_norm | selected_isa | remove_queue | mode2 | gds | devcoredump | gfxhub_pf | gds_pf | gds_addr | gds_vm_pf | vmid | vm_addr |"
    echo "|---|---|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|"
    awk -F '\t' 'NR > 1 {
        dirty = ($13 == "1") ? " dirty" : "";
        sync_failure = ($10 == "") ? " " : $10;
        printf "| `%s` | `%s` | `%s` | %s | `%s%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | `%s` | `%s` | `%s` | %s | `%s` |\n", \
            $1, $2, $9, sync_failure, $12, dirty, $15, $16, $17, $18, $19, $25, $27, $20, $21, $22, $23, $28, $31, $34, $32, $36, $37;
    }' "$TSV"
} >"$MD"

echo "tsv=$TSV"
echo "markdown=$MD"
