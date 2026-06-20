#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 left-isa-summary.tsv right-isa-summary.tsv [out.tsv]" >&2
    exit 2
fi

left="$1"
right="$2"
out="${3:-/dev/stdout}"

if [[ ! -f "$left" ]]; then
    echo "missing left summary: $left" >&2
    exit 1
fi
if [[ ! -f "$right" ]]; then
    echo "missing right summary: $right" >&2
    exit 1
fi

awk -F '\t' '
function load_header(    i) {
    for (i = 1; i <= NF; ++i) {
        col[$i] = i;
    }
    required["symbol"];
    required["size"];
    required["instructions"];
    required["s_nop"];
    required["ds_store"];
    required["ds_load"];
    required["s_barrier"];
    required["s_cbranch"];
    required["pre_ds_s_nop"];
    required["pre_ds_s_add_i32"];
    required["tail_s_nop"];
    required["tail_s_add_i32"];
    required["tail_window"];
    required["group_segment"];
    required["private_segment"];
    required["sgpr"];
    required["vgpr"];
    required["wavefront"];
    for (i in required) {
        if (!(i in col)) {
            printf "missing required column %s in %s\n", i, FILENAME >"/dev/stderr";
            exit 1;
        }
    }
}
function capture(side,    sym, i) {
    sym = $col["symbol"];
    if (sym == "") {
        return;
    }
    if (!(sym in keys)) {
        keys[sym] = 1;
        order[++n_keys] = sym;
    }
    side_keys[side, sym] = 1;
    for (i in col) {
        value[side, sym, i] = $col[i];
    }
}
function same(field, sym) {
    return value["left", sym, field] == value["right", sym, field] ? "same" : "diff";
}
function any_diff(sym, fields,    n, i, a) {
    n = split(fields, a, ",");
    for (i = 1; i <= n; ++i) {
        if (value["left", sym, a[i]] != value["right", sym, a[i]]) {
            return 1;
        }
    }
    return 0;
}
function classify(sym) {
    if (!(("left" SUBSEP sym) in side_keys)) {
        return "missing-left";
    }
    if (!(("right" SUBSEP sym) in side_keys)) {
        return "missing-right";
    }
    if (any_diff(sym, "tail_window,pre_ds_s_nop,pre_ds_s_add_i32,tail_s_nop,tail_s_add_i32")) {
        return "placement-drift";
    }
    if (any_diff(sym, "ds_store,ds_load,s_barrier,s_cbranch,s_nop")) {
        return "lds-control-drift";
    }
    if (any_diff(sym, "group_segment,private_segment,sgpr,vgpr,wavefront")) {
        return "resource-drift";
    }
    if (any_diff(sym, "size,instructions")) {
        return "size-drift";
    }
    return "same";
}
FNR == 1 {
    delete col;
    delete required;
    load_header();
    file_no++;
    next;
}
file_no == 1 {
    capture("left");
    next;
}
file_no == 2 {
    capture("right");
    next;
}
function emit_row(sym,    verdict) {
    verdict = classify(sym);
    if (verdict == "missing-left") {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
            sym, verdict, "", "", "", "", "", "", "", "", "", "", \
            value["right", sym, "tail_window"], "", \
            value["right", sym, "pre_ds_s_nop"] "/" value["right", sym, "pre_ds_s_add_i32"], \
            "", \
            value["right", sym, "tail_s_nop"] "/" value["right", sym, "tail_s_add_i32"];
        return;
    }
    if (verdict == "missing-right") {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
            sym, verdict, "", "", "", "", "", "", "", "", "", \
            value["left", sym, "tail_window"], "", \
            value["left", sym, "pre_ds_s_nop"] "/" value["left", sym, "pre_ds_s_add_i32"], \
            "", \
            value["left", sym, "tail_s_nop"] "/" value["left", sym, "tail_s_add_i32"], "";
        return;
    }
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
        sym, verdict, same("size", sym), same("instructions", sym), same("s_nop", sym), \
        same("ds_store", sym), same("ds_load", sym), same("s_barrier", sym), \
        same("s_cbranch", sym), \
        any_diff(sym, "tail_window,pre_ds_s_nop,pre_ds_s_add_i32,tail_s_nop,tail_s_add_i32") ? "diff" : "same", \
        any_diff(sym, "group_segment,private_segment,sgpr,vgpr,wavefront") ? "diff" : "same", \
        value["left", sym, "tail_window"], value["right", sym, "tail_window"], \
        value["left", sym, "pre_ds_s_nop"] "/" value["left", sym, "pre_ds_s_add_i32"], \
        value["right", sym, "pre_ds_s_nop"] "/" value["right", sym, "pre_ds_s_add_i32"], \
        value["left", sym, "tail_s_nop"] "/" value["left", sym, "tail_s_add_i32"], \
        value["right", sym, "tail_s_nop"] "/" value["right", sym, "tail_s_add_i32"];
}
END {
    print "symbol\tverdict\tsize\tinstructions\ts_nop\tds_store\tds_load\ts_barrier\ts_cbranch\tplacement\tresources\tleft_tail_window\tright_tail_window\tleft_pre_ds\tright_pre_ds\tleft_tail\tright_tail";
    for (i = 1; i <= n_keys; ++i) {
        emit_row(order[i]);
    }
}
' "$left" "$right" >"$out"
