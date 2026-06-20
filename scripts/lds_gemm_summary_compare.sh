#!/usr/bin/env bash
set -euo pipefail

LEFT="${1:-}"
RIGHT="${2:-}"
OUT="${3:-/tmp/hipfire-lds-summary-compare.tsv}"

if [[ -z "$LEFT" || -z "$RIGHT" ]]; then
    echo "usage: $0 left-summary.tsv right-summary.tsv [out.tsv]" >&2
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
function key(row) {
    return row["variant"] "|" row["mode"] "|" row["launches"] "|" row["shape"] "|" row["k_limit"] "|" row["arch"];
}
function read_header(    i) {
    for (i = 1; i <= NF; i++) {
        col[$i] = i;
    }
}
function load_row(dst,    i) {
    for (i in col) {
        dst[i] = $(col[i]);
    }
}
function same(a, b) {
    return a == b ? "same" : "diff";
}
function code_same(k) {
    if (left[k, "selected_isa_norm_sha256"] != "" && right[k, "selected_isa_norm_sha256"] != "") {
        return same(left[k, "selected_isa_norm_sha256"], right[k, "selected_isa_norm_sha256"]);
    }
    if (left[k, "amdgpu_isa_norm_sha256"] != "" && right[k, "amdgpu_isa_norm_sha256"] != "") {
        return same(left[k, "amdgpu_isa_norm_sha256"], right[k, "amdgpu_isa_norm_sha256"]);
    }
    return same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]);
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
function classify(k,    source_same, code_same_result, metadata_same, exit_same, driver_same, hipcc_same, verdict) {
    source_same = same(left[k, "source_sha256"], right[k, "source_sha256"]);
    code_same_result = code_same(k);
    metadata_same = (same(left[k, "amdgpu_obj_sha256"], right[k, "amdgpu_obj_sha256"]) == "same" && \
        same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]) == "same") ? "same" : "diff";
    exit_same = same(left[k, "exit"], right[k, "exit"]);
    driver_same = same(left[k, "driver"], right[k, "driver"]);
    hipcc_same = same(left[k, "hipcc"], right[k, "hipcc"]);

    if (source_same == "diff") {
        verdict = "source-drift";
    } else if (code_same_result == "diff") {
        verdict = "codegen-drift";
    } else if (exit_same == "diff") {
        verdict = "same-codegen-runtime-diff";
    } else if (driver_same == "diff" || hipcc_same == "diff") {
        verdict = "same-result-env-diff";
    } else if (metadata_same == "diff") {
        verdict = "codegen-metadata-drift";
    } else {
        verdict = "same";
    }
    return verdict;
}
FNR == 1 {
    file_no++;
    delete col;
    read_header();
    next;
}
file_no == 1 {
    delete row;
    load_row(row);
    k = key(row);
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
    right_keys[k] = 1;
    for (i in col) {
        right[k, i] = row[i];
    }
    next;
}
END {
    print "key\tstatus\tverdict\tleft_exit\tright_exit\tleft_sync\tright_sync\tsource\tobj\tisa\tisa_norm\tselected_isa\tdriver\thipcc\tdmesg_sig\tdevcore_sig\tgcvm_sig\tleft_devcore\tright_devcore\tleft_gcvm\tright_gcvm\tleft_git\tright_git";
    for (k in left_keys) {
        if (!(k in right_keys)) {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                k, "left-only", "missing-right", left[k, "exit"], "", \
                left[k, "sync_failure"], "", "", "", "", "", "", "", "", \
                dmesg_sig(left, k), "", "", devcore_sig(left, k), "", \
                gcvm_sig(left, k), "", left[k, "git_commit"], "";
            continue;
        }
        verdict = classify(k);
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
            k, "both", verdict, left[k, "exit"], right[k, "exit"], \
            left[k, "sync_failure"], right[k, "sync_failure"], \
            same(left[k, "source_sha256"], right[k, "source_sha256"]), \
            same(left[k, "amdgpu_obj_sha256"], right[k, "amdgpu_obj_sha256"]), \
            same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]), \
            same(left[k, "amdgpu_isa_norm_sha256"], right[k, "amdgpu_isa_norm_sha256"]), \
            same(left[k, "selected_isa_norm_sha256"], right[k, "selected_isa_norm_sha256"]), \
            same(left[k, "driver"], right[k, "driver"]), \
            same(left[k, "hipcc"], right[k, "hipcc"]), \
            same(dmesg_sig(left, k), dmesg_sig(right, k)), \
            same(devcore_sig(left, k), devcore_sig(right, k)), \
            same(gcvm_sig(left, k), gcvm_sig(right, k)), \
            devcore_sig(left, k), devcore_sig(right, k), \
            gcvm_sig(left, k), gcvm_sig(right, k), \
            left[k, "git_commit"], right[k, "git_commit"];
    }
    for (k in right_keys) {
        if (!(k in left_keys)) {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                k, "right-only", "missing-left", "", right[k, "exit"], \
                "", right[k, "sync_failure"], "", "", "", "", "", "", "", \
                dmesg_sig(right, k), "", "", "", devcore_sig(right, k), \
                "", gcvm_sig(right, k), "", right[k, "git_commit"];
        }
    }
}
' "$LEFT" "$RIGHT" >"$OUT"

cat "$OUT"
