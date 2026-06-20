#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/tmp/hipfire-lds-direct-ab-multi-exec-artifacts}"
OUT_PREFIX="${2:-$ROOT/direct-ab-artifact-summary}"
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

isa_count() {
    local pattern="$1"
    local path="$2"
    if [[ -r "$path" ]]; then
        rg -c "$pattern" "$path" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

isa_ds_store_offset1() {
    local path="$1"
    if [[ -r "$path" ]]; then
        rg -o 'offset1:[0-9]+' "$path" 2>/dev/null |
            sed 's/offset1://' |
            sort -n -u |
            paste -sd, - || true
    fi
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

gcvm_fault_field() {
    local value="$1"
    local mask="$2"
    local shift="$3"
    local raw
    raw="$(hex_to_dec "$value")"
    [[ -n "$raw" ]] || return 0
    printf '%u\n' "$(((raw & mask) >> shift))"
}

gcvm_fault_flags() {
    local raw
    local flags=()
    raw="$(hex_to_dec "$1")"
    [[ -n "$raw" ]] || return 0
    ((raw & 0x00000001)) && flags+=("MORE_FAULTS")
    ((raw & 0x0000000e)) && flags+=("WALKER_ERROR")
    ((raw & 0x000000f0)) && flags+=("PERMISSION_FAULTS")
    ((raw & 0x00000100)) && flags+=("MAPPING_ERROR")
    ((raw & 0x00040000)) && flags+=("RW")
    ((raw & 0x00080000)) && flags+=("ATOMIC")
    ((raw & 0x01000000)) && flags+=("VF")
    ((raw & 0x20000000)) && flags+=("PRT")
    ((raw & 0x40000000)) && flags+=("FED")
    join_flags "${flags[@]}"
}

first_devcoredump() {
    local dir="$1"
    find "$dir/coredumps" -maxdepth 1 -type f -name '*.data' 2>/dev/null | sort | head -1 || true
}

printf 'artifact\tactive\tblock\treads\titers\tchunks\tmode\tgrid\tarch\tbuild_only\texit\tsync_failure\thip_error\tdriver\tgpu\thipcc\tsource_sha256\tamdgpu_obj_sha256\tamdgpu_isa_sha256\tamdgpu_isa_norm_sha256\tgroup_segment\tprivate_segment\tsgpr\tvgpr\twavefront\tdmesg_remove_queue\tdmesg_mode2\tdmesg_gds\tdevcoredump\tdevcore_file\tdevcore_gfxhub_page_fault\tdevcore_fault_addr\tdevcore_prot_status\tdevcore_gcvm_flags\tdevcore_gcvm_cid\tdevcore_gcvm_rw\tdevcore_gcvm_vmid\tdevcore_gds_protection_fault\tdevcore_gds_flags\tdevcore_gds_addr\tdevcore_gds_vm_protection_fault\tdevcore_gds_vm_flags\tdevcore_gds_vm_vmid\tdevcore_gds_vm_addr\tisa_s_barrier\tisa_ds\tisa_s_waitcnt\tisa_s_cbranch\tisa_ds_store_offset1\tlayout\tactive_start\tforce_wrap_cndmask\n' >"$TSV"

while IFS= read -r meta; do
    dir="$(dirname "$meta")"
    rel="${dir#$ROOT/}"
    [[ "$rel" == "$dir" ]] && rel="."

    run_log="$dir/run.log"
    exit_file="$dir/exit_code.txt"
    dmesg_before="$dir/dmesg.before.txt"
    dmesg_after="$dir/dmesg.after.txt"

    active="$(meta_value active "$meta" | sanitize)"
    active_start="$(meta_value active_start "$meta" | sanitize)"
    [[ -n "$active_start" ]] || active_start="0 x 0"
    block="$(meta_value block "$meta" | sanitize)"
    layout="$(meta_value layout "$meta" | sanitize)"
    [[ -n "$layout" ]] || layout="$active"
    reads="$(meta_value reads "$meta" | sanitize)"
    iters="$(meta_value iters "$meta" | sanitize)"
    chunks="$(meta_value chunks "$meta" | sanitize)"
    mode="$(meta_value mode "$meta" | sanitize)"
    force_wrap_cndmask="$(meta_value force_wrap_cndmask "$meta" | sanitize)"
    [[ -n "$force_wrap_cndmask" ]] || force_wrap_cndmask=0
    grid="$(meta_value grid "$meta" | sanitize)"
    arch="$(meta_value arch "$meta" | sanitize)"
    build_only="$(meta_value build_only "$meta" | sanitize)"
    hipcc="$(meta_value hipcc "$meta" | sanitize)"
    driver="$(meta_contains_value 'Driver version:' "$meta" | sed 's/.*Driver version:[[:space:]]*//' | sanitize)"
    gpu="$(meta_contains_value 'Marketing Name:' "$meta" | sed 's/.*Marketing Name:[[:space:]]*//' | sanitize)"
    exit_code="$([[ -r "$exit_file" ]] && tr -d '\n\r\t ' <"$exit_file" || true)"

    sync_failure=""
    hip_error=""
    if [[ -r "$run_log" ]]; then
        sync_failure="$(rg -m1 'sync [0-9]+ global [0-9]+ failed:' "$run_log" | sanitize || true)"
        hip_error="$(rg -o -m1 '\([0-9]+\)' "$run_log" | tr -d '()' || true)"
    fi

    dmesg_remove_queue="$(dmesg_delta_count 'REMOVE_QUEUE|remove queue' "$dmesg_before" "$dmesg_after")"
    dmesg_mode2="$(dmesg_delta_count 'MODE2|mode2' "$dmesg_before" "$dmesg_after")"
    dmesg_gds="$(dmesg_delta_count 'GDS|regGDS' "$dmesg_before" "$dmesg_after")"
    dmesg_remove_queue="${dmesg_remove_queue:-0}"
    dmesg_mode2="${dmesg_mode2:-0}"
    dmesg_gds="${dmesg_gds:-0}"

    source_sha="$(short_sha256 "$dir/lds_direct_ab_phase_probe.hip")"
    amdgpu_obj="$(first_artifact_file "$dir" '*hip-amdgcn-amd-amdhsa-*.o')"
    amdgpu_isa="$(first_artifact_file "$dir" '*hip-amdgcn-amd-amdhsa-*.o.isa.txt')"
    amdgpu_readobj="$(first_artifact_file "$dir" '*hip-amdgcn-amd-amdhsa-*.o.readobj.txt')"
    amdgpu_obj_sha="$(short_sha256 "$amdgpu_obj")"
    amdgpu_isa_sha="$(short_sha256 "$amdgpu_isa")"
    amdgpu_isa_norm_sha="$(normalized_isa_sha256 "$amdgpu_isa")"
    group_segment="$(metadata_value_near_name "$amdgpu_readobj" '_Z25lds_direct_ab_phase_probev' 'group_segment_fixed_size')"
    private_segment="$(metadata_value_near_name "$amdgpu_readobj" '_Z25lds_direct_ab_phase_probev' 'private_segment_fixed_size')"
    sgpr="$(metadata_value_near_name "$amdgpu_readobj" '_Z25lds_direct_ab_phase_probev' 'sgpr_count')"
    vgpr="$(metadata_value_near_name "$amdgpu_readobj" '_Z25lds_direct_ab_phase_probev' 'vgpr_count')"
    wavefront="$(metadata_value_near_name "$amdgpu_readobj" '_Z25lds_direct_ab_phase_probev' 'wavefront_size')"
    isa_s_barrier="$(isa_count '\bs_barrier\b' "$amdgpu_isa")"
    isa_ds="$(isa_count '\bds_' "$amdgpu_isa")"
    isa_s_waitcnt="$(isa_count '\bs_waitcnt\b' "$amdgpu_isa")"
    isa_s_cbranch="$(isa_count '\bs_cbranch' "$amdgpu_isa")"
    isa_ds_offset1="$(isa_ds_store_offset1 "$amdgpu_isa" | sanitize)"

    devcore="$(first_devcoredump "$dir")"
    devcoredump=0
    [[ -s "$devcore" ]] && devcoredump=1
    devcore_file="${devcore#$dir/}"
    [[ "$devcore_file" == "$devcore" ]] && devcore_file=""
    devcore_gfxhub_pf="$(devcore_contains '\[gfxhub\] Page fault observed' "$devcore")"
    devcore_fault_addr="$(devcore_colon_value '^Faulty page starting at address:' "$devcore")"
    devcore_prot_status="$(devcore_colon_value '^Protection fault status register:' "$devcore")"
    devcore_gcvm_flags="$(gcvm_fault_flags "$devcore_prot_status")"
    devcore_gcvm_cid="$(gcvm_fault_field "$devcore_prot_status" 0x0003fe00 9)"
    devcore_gcvm_rw="$(gcvm_fault_field "$devcore_prot_status" 0x00040000 18)"
    devcore_gcvm_vmid="$(gcvm_fault_field "$devcore_prot_status" 0x00f00000 20)"
    devcore_gds_pf="$(devcore_reg_value 'regGDS_PROTECTION_FAULT' "$devcore")"
    devcore_gds_flags="$(gds_fault_flags "$devcore_gds_pf")"
    devcore_gds_addr="$(gds_fault_addr "$devcore_gds_pf")"
    devcore_gds_vm_pf="$(devcore_reg_value 'regGDS_VM_PROTECTION_FAULT' "$devcore")"
    devcore_gds_vm_flags="$(gds_vm_fault_flags "$devcore_gds_vm_pf")"
    devcore_gds_vm_vmid="$(gds_vm_fault_vmid "$devcore_gds_vm_pf")"
    devcore_gds_vm_addr="$(gds_vm_fault_addr "$devcore_gds_vm_pf")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rel" "$active" "$block" "$reads" "$iters" "$chunks" "$mode" "$grid" \
        "$arch" "$build_only" "$exit_code" "$sync_failure" "$hip_error" \
        "$driver" "$gpu" "$hipcc" "$source_sha" "$amdgpu_obj_sha" \
        "$amdgpu_isa_sha" "$amdgpu_isa_norm_sha" "$group_segment" \
        "$private_segment" "$sgpr" "$vgpr" "$wavefront" "$dmesg_remove_queue" \
        "$dmesg_mode2" "$dmesg_gds" "$devcoredump" "$devcore_file" \
        "$devcore_gfxhub_pf" "$devcore_fault_addr" "$devcore_prot_status" \
        "$devcore_gcvm_flags" "$devcore_gcvm_cid" "$devcore_gcvm_rw" \
        "$devcore_gcvm_vmid" "$devcore_gds_pf" "$devcore_gds_flags" \
        "$devcore_gds_addr" "$devcore_gds_vm_pf" "$devcore_gds_vm_flags" \
        "$devcore_gds_vm_vmid" "$devcore_gds_vm_addr" "$isa_s_barrier" \
        "$isa_ds" "$isa_s_waitcnt" "$isa_s_cbranch" "$isa_ds_offset1" \
        "$layout" "$active_start" "$force_wrap_cndmask" >>"$TSV"
done < <(find "$ROOT" -type f -name meta.txt | sort)

{
    echo "# LDS Direct-AB Artifact Summary"
    echo
    echo "- root: \`$ROOT\`"
    echo "- generated: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- tsv: \`$TSV\`"
    echo
    echo "| artifact | exit | active | start | layout | wrap | reads | iters | chunks | grid | obj | isa_norm | sgpr | vgpr | barrier | ds | wait | branch | offset1 | dmesg | devcore | gcvm | gds | sync |"
    echo "|---|---:|---|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---|"
    awk -F '\t' 'NR > 1 {
        sync_failure = ($12 == "") ? " " : $12;
        dmesg = "rq=" $26 ",mode2=" $27 ",gds=" $28;
        gds = $38 "/" $41;
        layout = ($50 == "") ? $2 : $50;
        active_start = ($51 == "") ? "0 x 0" : $51;
        force_wrap = ($52 == "") ? 0 : $52;
        printf "| `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | %s | `%s` | `%s` | %s | `%s` | `%s` | %s |\n", \
            $1, $11, $2, active_start, layout, force_wrap, $4, $5, $6, $8, $18, $20, $23, $24, $45, $46, $47, $48, $49, dmesg, $29, $34, gds, sync_failure;
    }' "$TSV"
} >"$MD"

echo "tsv=$TSV"
echo "markdown=$MD"
