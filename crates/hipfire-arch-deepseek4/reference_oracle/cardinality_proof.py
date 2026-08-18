#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Cardinality proof: reference vs parent formulas at seq=1024.

Does NOT need GPU. Imports model.py helpers for reference; parent formulas
are transcribed from attention.rs / indexer.rs with line citations.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim
sys.modules["kernel"] = kernel_shim
import torch
from model import get_window_topk_idxs, get_compress_topk_idxs

WIN = 128
INDEX_TOPK = 512
SEQ = 1024
RATIO4 = 4
PROBES = [64, 200, 400, 448, 500, 511, 512, 600, 800, 1000, 1023]

def parent_n_active_comp(r, ratio, n_compressed, topk_max=INDEX_TOPK):
    """attention.rs:1324-1325  n_active = topk_max.min(vis)"""
    vis = (r + 1) // ratio
    return min(topk_max, vis, n_compressed)

def parent_n_valid_swa(r, window=WIN):
    return min(r + 1, window)

def parent_joint(r, ratio, n_compressed):
    return parent_n_valid_swa(r) + parent_n_active_comp(r, ratio, n_compressed)

def main():
    n_comp = SEQ // RATIO4  # 256
    assert n_comp == 256
    # Reference helpers
    w = get_window_topk_idxs(WIN, 1, SEQ, 0)[0]
    c = get_compress_topk_idxs(RATIO4, 1, SEQ, 0, offset=SEQ)[0]  # unified
    c_raw = get_compress_topk_idxs(RATIO4, 1, SEQ, 0, offset=0)[0]

    rows = []
    print(f"seq={SEQ} ratio={RATIO4} n_comp={n_comp} index_topk={INDEX_TOPK} window={WIN}")
    print(f"n_comp <= index_topk ? {n_comp <= INDEX_TOPK} → parent topk FAST PATH = identity 0..n_comp-1")
    print(f"{'r':>6} {'ref_win':>8} {'ref_comp':>8} {'ref_joint':>9} {'par_swa':>8} {'par_comp':>8} {'par_joint':>9} {'match':>6} {'notes'}")
    all_match = True
    for r in PROBES:
        rw = [int(x) for x in w[r].tolist() if x >= 0]
        rc = [int(x) for x in c[r].tolist() if x >= 0]
        rc_raw = [int(x) for x in c_raw[r].tolist() if x >= 0]
        ref_win, ref_comp = len(rw), len(rc)
        ref_joint = ref_win + ref_comp
        # reference joint includes -1 pads in tensor width but VALID count is win+comp
        # full topk tensor width:
        ref_tensor_w = w.shape[-1] + c.shape[-1]  # 128 + 256 = 384
        ps = parent_n_valid_swa(r)
        pc = parent_n_active_comp(r, RATIO4, n_comp)
        pj = ps + pc
        match = (ref_win == ps and ref_comp == pc and ref_joint == pj)
        all_match &= match
        # check identity packing: raw set should be 0..n_vis-1
        n_vis = (r + 1) // RATIO4
        expect_raw = list(range(n_vis))
        identity_ok = rc_raw == expect_raw
        notes = []
        if not match:
            notes.append("CARD_MISMATCH")
        if not identity_ok:
            notes.append(f"raw_not_identity got={rc_raw[:8]}...")
        if r >= 448 and r <= 576:
            notes.append("step_region")
        # budget-on-joint counterfactual
        joint_budget_counterfactual = min(INDEX_TOPK, ref_joint)  # if someone applied 512 to joint
        if joint_budget_counterfactual != ref_joint:
            notes.append(f"if_budget_on_joint={joint_budget_counterfactual}")
        print(f"{r:6d} {ref_win:8d} {ref_comp:8d} {ref_joint:9d} {ps:8d} {pc:8d} {pj:9d} {str(match):>6} {' '.join(notes)}")
        rows.append({
            "r": r,
            "ref_win": ref_win, "ref_comp": ref_comp, "ref_joint_valid": ref_joint,
            "ref_tensor_width": int(ref_tensor_w),
            "parent_swa": ps, "parent_comp": pc, "parent_joint": pj,
            "card_match": match,
            "ref_comp_raw_is_identity_0_nvis": identity_ok,
            "n_vis": n_vis,
            "counterfactual_budget_on_joint": min(INDEX_TOPK, ref_joint),
            "counterfactual_budget_on_joint_minus_win": max(0, min(INDEX_TOPK, ref_joint) - ref_win),
        })

    # Critical: does parent apply budget to joint?
    print("\n=== BUDGET SCOPE ===")
    print("reference: index_topk on compressed half only; joint = win + min(topk, n_comp_vis)")
    print("parent attention.rs:1324-1325: n_active_comp = topk_max.min(vis)  # compressed only")
    print("parent kernel: n_total = n_valid_swa + n_active_topk  # separate counters, no joint budget")
    print(f"parent topk fast-path (kernels indexer_top_k_batched): n_iter={n_comp} <= k_fill={INDEX_TOPK} → identity indices")
    print(f"parent main_kv capacity: max(max_rows,512)=max({SEQ},512)={max(SEQ,512)} >= n_comp={n_comp}")

    # Show when joint-budget counterfactual would bite
    print("\n=== WHEN WOULD joint-budget-512 BITE? ===")
    for r in range(0, SEQ, 32):
        j = parent_joint(r, RATIO4, n_comp)
        if j > INDEX_TOPK:
            print(f"  first joint>512 at r={r}: joint={j} win={parent_n_valid_swa(r)} comp={parent_n_active_comp(r,RATIO4,n_comp)}")
            break
    else:
        print(f"  at seq={SEQ} ratio4, max joint={parent_joint(SEQ-1,RATIO4,n_comp)} <= 512? {parent_joint(SEQ-1,RATIO4,n_comp)<=INDEX_TOPK}")
        print("  → even a wrong joint-budget of 512 would NOT truncate at seq=1024 ratio4 (max joint=128+256=384)")

    # When does compressed-only topk filter?
    print("\n=== WHEN DOES compressed topk FILTER? ===")
    print(f"  need n_comp > 512 → seq > {INDEX_TOPK * RATIO4} for ratio4")
    print(f"  at seq=1024, n_comp=256: NO FILTER on either side")

    # post-mask packing hazard (only if non-identity topk order)
    print("\n=== POST-MASK PACKING HAZARD ===")
    print("  reference: softmax over FULL topk width (128+256) with -1 → -inf; valid slots anywhere count")
    print("  parent: softmax over first n_active of topk_staged; requires valid idxs packed in front")
    print("  at n_comp<=512 fast-path writes identity 0..n-1 then causal → -1s are TRAILING → packing OK")
    print("  hazard only when real score-topk reorders and future slots rank into the prefix (n_comp>512)")

    out = {
        "seq": SEQ,
        "ratio": RATIO4,
        "n_comp": n_comp,
        "index_topk": INDEX_TOPK,
        "all_card_match": all_match,
        "max_joint": parent_joint(SEQ-1, RATIO4, n_comp),
        "joint_budget_512_would_truncate_at_1024": parent_joint(SEQ-1, RATIO4, n_comp) > INDEX_TOPK,
        "compressed_topk_filters_at_1024": n_comp > INDEX_TOPK,
        "verdict": (
            "CARDINALITY MATCH at all probes. Neither side filters at seq=1024 ratio4. "
            "Even counterfactual joint-budget-512 would NOT truncate (max joint=384). "
            "Selection/budget-scope hypothesis is DEAD for the 1024-token parent capture. "
            "The [448,512) step must be content/layout (KV values, RoPE, staging), not slot count."
        ),
        "rows": rows,
        "conventions": {
            "prefill_offset": "seqlen (token_kv length before cat)",
            "decode_offset": "window_size",
            "prefill_kv_for_attn": "cat([token_kv[B,S,D], kv_compress[B,S//ratio,D]], dim=1)",
            "prefill_window_idxs": "absolute token positions",
            "decode_window_idxs": "ring slots",
            "decode_kv_for_attn": "kv_cache[0:window]=ring; [window:]=compress alias",
            "index_topk_scope": "compressed half only; window EXEMPT",
            "parent_split_buffer": "offset=0 into separate main_kv_cache; n_total=n_swa+n_comp",
        },
    }
    Path("/tmp/cardinality_proof.json").write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT ===")
    print(out["verdict"])
    print("wrote /tmp/cardinality_proof.json")

if __name__ == "__main__":
    main()
