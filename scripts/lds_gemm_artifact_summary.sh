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
    awk 'NR == FNR { seen[$0] = 1; next } !($0 in seen)' "$before" "$after" |
        rg -c "$pattern" || true
}

short_sha256() {
    local path="$1"
    if [[ -r "$path" ]]; then
        sha256sum "$path" | awk '{ print substr($1, 1, 16) }'
    fi
}

printf 'artifact\tvariant\tmode\tlaunches\tshape\tk_limit\tarch\tbuild_only\texit\tsync_failure\thip_error\tgit_commit\tgit_dirty\thipcc\tdriver\tgpu\tsource_sha256\tamdgpu_obj_sha256\tamdgpu_isa_sha256\tdmesg_remove_queue\tdmesg_mode2\tdmesg_gds\tdevcoredump\tisa_files\n' >"$TSV"

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

    devcoredump=0
    [[ -s "$dir/devcoredump.data" ]] && devcoredump=1
    isa_files="$(find "$dir/save-temps" -maxdepth 1 -type f -name '*.isa.txt' 2>/dev/null | wc -l | tr -d ' ')"
    source_sha="$(short_sha256 "$dir/lds_gemm_standalone_probe.hip")"
    amdgpu_obj="$(find "$dir/save-temps" "$dir" -maxdepth 1 -type f -name '*hip-amdgcn-amd-amdhsa-*.o' 2>/dev/null | sort | head -1)"
    amdgpu_isa="$(find "$dir/save-temps" "$dir" -maxdepth 1 -type f -name '*hip-amdgcn-amd-amdhsa-*.o.isa.txt' 2>/dev/null | sort | head -1)"
    amdgpu_obj_sha="$(short_sha256 "$amdgpu_obj")"
    amdgpu_isa_sha="$(short_sha256 "$amdgpu_isa")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rel" "$variant" "$mode" "$launches" "$shape" "$k_limit" "$arch" \
        "$build_only" "$exit_code" "$sync_failure" "$hip_error" "$git_commit" \
        "$git_dirty" "$hipcc" "$driver" "$gpu" "$source_sha" \
        "$amdgpu_obj_sha" "$amdgpu_isa_sha" "$dmesg_remove_queue" \
        "$dmesg_mode2" "$dmesg_gds" "$devcoredump" "$isa_files" >>"$TSV"
done < <(find "$ROOT" -type f -name meta.txt | sort)

{
    echo "# LDS GEMM Artifact Summary"
    echo
    echo "- root: \`$ROOT\`"
    echo "- generated: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- tsv: \`$TSV\`"
    echo
    echo "| artifact | variant | exit | sync failure | git | driver | gpu | src | obj | isa | remove_queue | mode2 | gds | devcoredump |"
    echo "|---|---|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|"
    awk -F '\t' 'NR > 1 {
        dirty = ($13 == "1") ? " dirty" : "";
        sync_failure = ($10 == "") ? " " : $10;
        printf "| `%s` | `%s` | `%s` | %s | `%s%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s |\n", \
            $1, $2, $9, sync_failure, $12, dirty, $15, $16, $17, $18, $19, $20, $21, $22, $23;
    }' "$TSV"
} >"$MD"

echo "tsv=$TSV"
echo "markdown=$MD"
