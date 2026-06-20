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
function classify(k,    source_same, obj_same, isa_same, exit_same, driver_same, hipcc_same, verdict) {
    source_same = same(left[k, "source_sha256"], right[k, "source_sha256"]);
    obj_same = same(left[k, "amdgpu_obj_sha256"], right[k, "amdgpu_obj_sha256"]);
    isa_same = same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]);
    exit_same = same(left[k, "exit"], right[k, "exit"]);
    driver_same = same(left[k, "driver"], right[k, "driver"]);
    hipcc_same = same(left[k, "hipcc"], right[k, "hipcc"]);

    if (source_same == "diff") {
        verdict = "source-drift";
    } else if (obj_same == "diff" || isa_same == "diff") {
        verdict = "codegen-drift";
    } else if (exit_same == "diff") {
        verdict = "same-codegen-runtime-diff";
    } else if (driver_same == "diff" || hipcc_same == "diff") {
        verdict = "same-result-env-diff";
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
    print "key\tstatus\tverdict\tleft_exit\tright_exit\tleft_sync\tright_sync\tsource\tobj\tisa\tdriver\thipcc\tleft_git\tright_git";
    for (k in left_keys) {
        if (!(k in right_keys)) {
            print k "\tleft-only\tmissing-right\t" left[k, "exit"] "\t\t" left[k, "sync_failure"] "\t\t\t\t\t\t\t" left[k, "git_commit"] "\t";
            continue;
        }
        verdict = classify(k);
        print k "\tboth\t" verdict "\t" left[k, "exit"] "\t" right[k, "exit"] "\t" \
            left[k, "sync_failure"] "\t" right[k, "sync_failure"] "\t" \
            same(left[k, "source_sha256"], right[k, "source_sha256"]) "\t" \
            same(left[k, "amdgpu_obj_sha256"], right[k, "amdgpu_obj_sha256"]) "\t" \
            same(left[k, "amdgpu_isa_sha256"], right[k, "amdgpu_isa_sha256"]) "\t" \
            same(left[k, "driver"], right[k, "driver"]) "\t" \
            same(left[k, "hipcc"], right[k, "hipcc"]) "\t" \
            left[k, "git_commit"] "\t" right[k, "git_commit"];
    }
    for (k in right_keys) {
        if (!(k in left_keys)) {
            print k "\tright-only\tmissing-left\t\t" right[k, "exit"] "\t\t" right[k, "sync_failure"] "\t\t\t\t\t\t\t" right[k, "git_commit"];
        }
    }
}
' "$LEFT" "$RIGHT" >"$OUT"

cat "$OUT"
