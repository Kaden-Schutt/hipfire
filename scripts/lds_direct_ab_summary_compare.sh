#!/usr/bin/env bash
set -euo pipefail

LEFT="${1:-}"
RIGHT="${2:-}"
OUT="${3:-/tmp/hipfire-lds-direct-ab-summary-compare.tsv}"

if [[ -z "$LEFT" || -z "$RIGHT" ]]; then
    echo "usage: $0 left-direct-ab-summary.tsv right-direct-ab-summary.tsv [out.tsv]" >&2
    exit 2
fi
if [[ ! -r "$LEFT" ]]; then
    echo "missing left summary: $LEFT" >&2
    exit 1
fi
if [[ ! -r "$RIGHT" ]]; then
    echo "missing right summary: $RIGHT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"

awk -F '\t' '
function read_header(    i) {
    for (i = 1; i <= NF; ++i) {
        col[$i] = i;
    }
}
function require_columns(    need, n, i) {
    n = split("active block reads iters chunks mode grid arch build_only exit sync_failure source_sha256 amdgpu_obj_sha256 amdgpu_isa_sha256 amdgpu_isa_norm_sha256 group_segment private_segment sgpr vgpr wavefront driver gpu hipcc dmesg_remove_queue dmesg_mode2 dmesg_gds devcoredump devcore_gfxhub_page_fault devcore_fault_addr devcore_prot_status devcore_gcvm_flags devcore_gcvm_cid devcore_gcvm_rw devcore_gcvm_vmid devcore_gds_protection_fault devcore_gds_flags devcore_gds_addr devcore_gds_vm_protection_fault devcore_gds_vm_flags devcore_gds_vm_vmid devcore_gds_vm_addr", need, " ");
    for (i = 1; i <= n; ++i) {
        if (!(need[i] in col)) {
            printf "missing required column %s in %s\n", need[i], FILENAME >"/dev/stderr";
            exit 1;
        }
    }
}
function key(row) {
    return row["active"] "|" row["block"] "|" row["reads"] "|" row["iters"] "|" \
        row["chunks"] "|" row["mode"] "|" row["grid"] "|" row["arch"];
}
function load_row(dst,    i) {
    for (i in col) {
        dst[i] = $(col[i]);
    }
}
function same(a, b) {
    return a == b ? "same" : "diff";
}
function resource_sig(side, k) {
    return side[k, "group_segment"] "/" side[k, "private_segment"] "/" \
        side[k, "sgpr"] "/" side[k, "vgpr"] "/" side[k, "wavefront"];
}
function dmesg_sig(side, k) {
    return side[k, "dmesg_remove_queue"] "/" side[k, "dmesg_mode2"] "/" side[k, "dmesg_gds"];
}
function devcore_sig(side, k) {
    if (side[k, "devcoredump"] != "1") {
        return "";
    }
    return side[k, "devcore_gfxhub_page_fault"] "/" side[k, "devcore_fault_addr"] "/" \
        side[k, "devcore_prot_status"] "/" side[k, "devcore_gds_protection_fault"] "/" \
        side[k, "devcore_gds_vm_protection_fault"];
}
function gcvm_sig(side, k) {
    if (side[k, "devcoredump"] != "1" || side[k, "devcore_gcvm_flags"] == "") {
        return "";
    }
    return side[k, "devcore_gcvm_flags"] "/cid=" side[k, "devcore_gcvm_cid"] \
        "/rw=" side[k, "devcore_gcvm_rw"] "/vmid=" side[k, "devcore_gcvm_vmid"];
}
function gds_sig(side, k) {
    if (side[k, "devcoredump"] != "1") {
        return "";
    }
    return side[k, "devcore_gds_protection_fault"] "/" side[k, "devcore_gds_flags"] "/" \
        side[k, "devcore_gds_addr"] "/" side[k, "devcore_gds_vm_protection_fault"] "/" \
        side[k, "devcore_gds_vm_flags"] "/" side[k, "devcore_gds_vm_vmid"] "/" \
        side[k, "devcore_gds_vm_addr"];
}
function code_same(k) {
    if (left[k, "amdgpu_isa_norm_sha256"] != "" && right[k, "amdgpu_isa_norm_sha256"] != "") {
        return same(left[k, "amdgpu_isa_norm_sha256"], right[k, "amdgpu_isa_norm_sha256"]);
    }
    return same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]);
}
function classify(k,    source_same, code_same_result, resource_same, exit_same, env_same, build_same) {
    source_same = same(left[k, "source_sha256"], right[k, "source_sha256"]);
    code_same_result = code_same(k);
    resource_same = same(resource_sig(left, k), resource_sig(right, k));
    exit_same = same(left[k, "exit"], right[k, "exit"]);
    env_same = (same(left[k, "driver"], right[k, "driver"]) == "same" && \
        same(left[k, "gpu"], right[k, "gpu"]) == "same" && \
        same(left[k, "hipcc"], right[k, "hipcc"]) == "same") ? "same" : "diff";
    build_same = same(left[k, "build_only"], right[k, "build_only"]);

    if (source_same == "diff") {
        return "source-drift";
    }
    if (resource_same == "diff") {
        return "resource-drift";
    }
    if (code_same_result == "diff") {
        return "codegen-drift";
    }
    if (exit_same == "diff") {
        return "same-codegen-runtime-diff";
    }
    if (build_same == "diff") {
        return "same-codegen-build-mode-diff";
    }
    if (env_same == "diff") {
        return "same-result-env-diff";
    }
    return "same";
}
FNR == 1 {
    file_no++;
    delete col;
    read_header();
    require_columns();
    next;
}
file_no == 1 {
    delete row;
    load_row(row);
    k = key(row);
    if (!(k in left_keys)) {
        left_order[++left_n] = k;
    }
    left_keys[k] = 1;
    for (i in col) {
        left[k, i] = row[i];
    }
    next;
}
file_no == 2 {
    delete row;
    load_row(row);
    k = key(row);
    if (!(k in right_keys)) {
        right_order[++right_n] = k;
    }
    right_keys[k] = 1;
    for (i in col) {
        right[k, i] = row[i];
    }
    next;
}
function print_missing_left(k) {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
        k, "right-only", "missing-left", "", right[k, "exit"], "", right[k, "sync_failure"], \
        "", "", "", "", "", "", "", "", dmesg_sig(right, k), "", "", \
        "", devcore_sig(right, k), "", gcvm_sig(right, k), "", gds_sig(right, k);
}
function print_missing_right(k) {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
        k, "left-only", "missing-right", left[k, "exit"], "", left[k, "sync_failure"], "", \
        "", "", "", "", "", "", "", dmesg_sig(left, k), "", "", "", \
        devcore_sig(left, k), "", gcvm_sig(left, k), "", gds_sig(left, k), "";
}
function print_both(k,    verdict) {
    verdict = classify(k);
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
        k, "both", verdict, left[k, "exit"], right[k, "exit"], \
        left[k, "sync_failure"], right[k, "sync_failure"], \
        same(left[k, "build_only"], right[k, "build_only"]), \
        same(left[k, "source_sha256"], right[k, "source_sha256"]), \
        same(left[k, "amdgpu_obj_sha256"], right[k, "amdgpu_obj_sha256"]), \
        same(left[k, "amdgpu_isa_norm_sha256"], right[k, "amdgpu_isa_norm_sha256"]), \
        same(resource_sig(left, k), resource_sig(right, k)), \
        same(left[k, "driver"], right[k, "driver"]), \
        same(left[k, "hipcc"], right[k, "hipcc"]), \
        same(dmesg_sig(left, k), dmesg_sig(right, k)), \
        same(devcore_sig(left, k), devcore_sig(right, k)), \
        same(gcvm_sig(left, k), gcvm_sig(right, k)), \
        same(gds_sig(left, k), gds_sig(right, k)), \
        devcore_sig(left, k), devcore_sig(right, k), \
        gcvm_sig(left, k), gcvm_sig(right, k), \
        gds_sig(left, k), gds_sig(right, k);
}
END {
    print "key\tstatus\tverdict\tleft_exit\tright_exit\tleft_sync\tright_sync\tbuild_only\tsource\tobj\tisa_norm\tresources\tdriver\thipcc\tdmesg_sig\tdevcore_sig\tgcvm_sig\tgds_sig\tleft_devcore\tright_devcore\tleft_gcvm\tright_gcvm\tleft_gds\tright_gds";
    for (i = 1; i <= left_n; ++i) {
        k = left_order[i];
        if (k in right_keys) {
            print_both(k);
        } else {
            print_missing_right(k);
        }
    }
    for (i = 1; i <= right_n; ++i) {
        k = right_order[i];
        if (!(k in left_keys)) {
            print_missing_left(k);
        }
    }
}
' "$LEFT" "$RIGHT" >"$OUT"

cat "$OUT"
