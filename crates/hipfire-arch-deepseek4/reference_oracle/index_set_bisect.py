#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Index-set bisect: reference ground truth for DSA top-k selection.

Priority check from Main: at query positions past 512 on tokens.bin (1024),
compare the *selected index sets* (not norms). Reports:
  - whether window (128) is inside or outside the index_topk=512 budget
  - n_comp vs index_topk (is top-k filtering active?)
  - the actual chosen compressed indices at probe positions
  - L2 (ratio=4, indexer) and L3 (ratio=128, identity gather)
  - L0 control (ratio=0, SWA only)

Does NOT import parent Rust. Optional --parent-idxs JSON for set diff.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _kernel_shim
sys.modules["kernel"] = _kernel_shim
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F

from model import (
    ModelArgs,
    Transformer,
    Block,
    Attention,
    apply_rotary_emb,
    get_window_topk_idxs,
    get_compress_topk_idxs,
    rotate_activation,
)
import model as model_mod
from weight_loader import load_state_into_model
from kernel_shim import act_quant, fp4_act_quant, sparse_attn

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
)
PROBE_ROWS = [0, 63, 127, 255, 511, 512, 768, 1023]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_tokens(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    assert len(raw) >= n * 4, (len(raw), n)
    ids = struct.unpack("<" + "i" * n, raw[: n * 4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)


def make_args(config_path: Path, max_seq_len: int) -> ModelArgs:
    with open(config_path) as f:
        cfg = json.load(f)
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k, v in cfg.items():
        if k not in fields:
            continue
        if k in ("compress_ratios", "dspark_target_layer_ids"):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    args = ModelArgs(**kwargs)
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    return args


def build_model(args: ModelArgs, model_dir: Path, layers: List[int], device: str) -> Transformer:
    n_build = max(layers) + 1
    full = args.n_layers
    args.n_layers = n_build
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full
    load_state_into_model(model, model_dir, layers=range(n_build), load_experts=False, device=device, verbose=True)
    # experts not needed for index-set path (attn only); save RAM
    model.eval()
    return model


def set_stats(a: Set[int], b: Set[int]) -> dict:
    inter = a & b
    only_a = a - b
    only_b = b - a
    return {
        "|a|": len(a),
        "|b|": len(b),
        "|intersect|": len(inter),
        "|only_a|": len(only_a),
        "|only_b|": len(only_b),
        "jaccard": len(inter) / max(len(a | b), 1),
        "only_a_sample": sorted(only_a)[:16],
        "only_b_sample": sorted(only_b)[:16],
    }


def classify_disagreement(ref: Set[int], other: Set[int], scores_ref: Optional[torch.Tensor] = None) -> str:
    if ref == other:
        return "identical_sets"
    only_r = ref - other
    only_o = other - ref
    if not only_r and not only_o:
        return "identical_sets"
    # boundary vs scattered: if we have scores, check if only_r are near the cut
    if scores_ref is not None and only_r:
        # scores_ref: [n_comp] for this row
        vals = [(int(i), float(scores_ref[i])) for i in only_r if 0 <= i < scores_ref.numel()]
        if vals:
            vals.sort(key=lambda t: -t[1])
            # if all missing are among the lowest scores in the selected set, boundary
            sel_scores = sorted([float(scores_ref[i]) for i in ref if 0 <= i < scores_ref.numel()])
            if sel_scores:
                cut = sel_scores[0]  # lowest selected
                near = sum(1 for _, s in vals if s <= cut * 1.01 + 1e-6)
                if near >= 0.8 * len(vals):
                    return "boundary_of_ranking"
    # order-only?
    if ref == other:
        return "identical_sets"
    return "scattered_or_scoring"


@torch.inference_mode()
def capture_layer_indexes(
    block: Block,
    x_hc: torch.Tensor,
    tokens: torch.Tensor,
    layer_id: int,
    probe_rows: List[int],
) -> dict:
    """Run attn half of block; return window + compress index sets and scores."""
    attn: Attention = block.attn
    ratio = int(attn.compress_ratio)
    win = int(attn.window_size)
    bsz, seqlen, _ = x_hc.shape[0], x_hc.shape[1], x_hc.shape[-1]
    # hc_pre + attn_norm → x for attention
    y, post, comb = block.hc_pre(x_hc, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    x = block.attn_norm(y)

    # Mirror Attention.forward index construction with captures
    freqs_cis = attn.freqs_cis[:seqlen]
    rd = attn.rope_head_dim
    if ratio and attn.compressor.kv_cache is None:
        attn.compressor.kv_cache = attn.kv_cache[:, win:]
        attn.compressor.freqs_cis = attn.freqs_cis
        if attn.indexer is not None:
            attn.indexer.freqs_cis = attn.freqs_cis

    qr = q = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(q).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    apply_rotary_emb(q[..., -rd:], freqs_cis)

    kv = attn.kv_norm(attn.wkv(x))
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    act_quant(kv[..., :-rd], 64, model_mod.scale_fmt, model_mod.scale_dtype, True)

    window_idxs = get_window_topk_idxs(win, bsz, seqlen, 0)  # [1,S,W]
    compress_idxs = None
    index_scores = None  # raw scores before topk, compressed space
    n_comp = 0
    topk_k = 0
    offset = kv.size(1)  # seqlen on prefill

    if ratio:
        n_comp = seqlen // ratio
        if attn.indexer is not None:
            # Reproduce Indexer.forward with score capture
            idxr = attn.indexer
            if idxr.compressor.kv_cache is None:
                idxr.compressor.kv_cache = idxr.kv_cache
                idxr.compressor.freqs_cis = idxr.freqs_cis
            q_i = idxr.wq_b(qr)
            q_i = q_i.unflatten(-1, (idxr.n_local_heads, idxr.head_dim))
            apply_rotary_emb(q_i[..., -rd:], freqs_cis)
            q_i = rotate_activation(q_i)
            fp4_act_quant(q_i, 32, True)
            idxr.compressor(x, 0)
            weights = idxr.weights_proj(x) * (idxr.softmax_scale * idxr.n_heads ** -0.5)
            # scores: [B,S,T_comp]
            kv_c = idxr.kv_cache[:bsz, :n_comp]
            index_score = torch.einsum("bshd,btd->bsht", q_i, kv_c)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
            # causal mask
            mask = torch.arange(n_comp).repeat(seqlen, 1) >= (
                torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, torch.tensor(float("-inf")), torch.tensor(0.0))
            index_scores = index_score.detach()
            topk_k = min(idxr.index_topk, n_comp)
            topk_idxs = index_score.topk(topk_k, dim=-1)[1]
            mask2 = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            compress_idxs = torch.where(mask2, torch.tensor(-1), topk_idxs + offset).int()
            # also keep un-offset compressed-space ids for set compare
            compress_idxs_raw = torch.where(mask2, torch.tensor(-1), topk_idxs).int()
        else:
            # identity gather path (ratio=128)
            compress_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, 0, offset)
            compress_idxs_raw = get_compress_topk_idxs(ratio, bsz, seqlen, 0, 0)
            topk_k = n_comp
    else:
        compress_idxs_raw = None

    joint = window_idxs
    if compress_idxs is not None:
        joint = torch.cat([window_idxs, compress_idxs], dim=-1)

    # Budget analysis (reference semantics)
    budget = {
        "window_size": win,
        "index_topk_config": int(getattr(getattr(attn, "indexer", None), "index_topk", 0) or 0),
        "compress_ratio": ratio,
        "n_comp_slots": n_comp,
        "topk_k_used": topk_k,
        "topk_filters": bool(ratio and attn.indexer is not None and n_comp > int(attn.indexer.index_topk)),
        "window_in_topk_budget": False,  # reference: window is OUTSIDE topk; cat after
        "joint_width": int(joint.size(-1)),
        "note": (
            "reference cats window_idxs || compress_topk_idxs; "
            "index_topk applies ONLY to compressed slots, never to SWA window"
        ),
    }

    probes = {}
    for r in probe_rows:
        if r >= seqlen:
            continue
        w_row = window_idxs[0, r].tolist()
        w_set = {int(v) for v in w_row if v >= 0}
        entry: Dict[str, Any] = {
            "window_set_size": len(w_set),
            "window_set": sorted(w_set),
            "window_expect_size": r + 1 if r + 1 <= win else win,
        }
        if compress_idxs_raw is not None:
            c_row = compress_idxs_raw[0, r].tolist()
            c_set = {int(v) for v in c_row if v >= 0}
            c_off = compress_idxs[0, r].tolist()
            c_off_set = {int(v) for v in c_off if v >= 0}
            entry["compress_raw_set_size"] = len(c_set)
            entry["compress_raw_set_sample"] = sorted(c_set)[:32]
            entry["compress_raw_set_max"] = max(c_set) if c_set else None
            entry["compress_offset_set_sample"] = sorted(c_off_set)[:32]
            # causal n_visible compressed
            n_vis = (r + 1) // ratio if ratio else 0
            entry["n_visible_comp"] = n_vis
            if index_scores is not None:
                sc = index_scores[0, r, :n_comp]
                # ranking of selected
                if c_set:
                    sel_scores = sorted(((i, float(sc[i])) for i in c_set), key=lambda t: -t[1])
                    entry["selected_score_min"] = sel_scores[-1][1]
                    entry["selected_score_max"] = sel_scores[0][1]
                    # how many non-selected have score >= min selected?
                    min_sel = sel_scores[-1][1]
                    n_above = int(((sc >= min_sel) & torch.isfinite(sc)).sum()) - len(c_set)
                    entry["n_nonselected_ge_min_selected"] = max(n_above, 0)
            # full set list only for small
            if len(c_set) <= 64:
                entry["compress_raw_set"] = sorted(c_set)
            else:
                entry["compress_raw_set_head"] = sorted(c_set)[:64]
                entry["compress_raw_set_tail"] = sorted(c_set)[-16:]
        j_row = joint[0, r].tolist()
        j_set = {int(v) for v in j_row if v >= 0}
        entry["joint_set_size"] = len(j_set)
        entry["joint_width_incl_minus1"] = len(j_row)
        probes[str(r)] = entry

    # compressed KV fingerprint if any
    kv_comp_l2 = None
    if ratio:
        attn.kv_cache.zero_()
        # re-run compressor path via full forward piece
        if attn.compressor.kv_cache is not None:
            # already filled by indexer compressor or main — main compressor separate
            pass
        # call main compressor fresh
        # reset main compressor state roughly by zeroing caches
        main_kv = attn.kv_cache[:, win:win + max(n_comp, 1)].clone()

    return {
        "layer_id": layer_id,
        "ratio": ratio,
        "has_indexer": attn.indexer is not None,
        "budget": budget,
        "probes": probes,
        "window_idxs_shape": list(window_idxs.shape),
        "compress_idxs_shape": list(compress_idxs.shape) if compress_idxs is not None else None,
        "joint_shape": list(joint.shape),
    }


@torch.inference_mode()
def run(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass

    model_dir = Path(args.model)
    config_path = Path(args.config) if args.config else model_dir / "inference" / "config.json"
    if not config_path.exists():
        config_path = HERE / "config.json"

    log(f"model={model_dir}")
    log(f"config={config_path}")
    log(f"tokens={args.tokens} seq={args.seq}")
    log(f"torch={torch.__version__} device=cpu")

    margs = make_args(config_path, max_seq_len=max(args.seq, 256))
    layers = [int(x) for x in args.layers.split(",")]
    log(f"layers={layers}  compress_ratios={[margs.compress_ratios[i] for i in layers]}")
    log(
        f"window={margs.window_size} index_topk={margs.index_topk} "
        f"compress_rope_theta={margs.compress_rope_theta}"
    )

    t0 = time.time()
    model = build_model(margs, model_dir, layers, "cpu")
    log(f"load done in {time.time()-t0:.1f}s")

    tokens = load_tokens(Path(args.tokens), args.seq)
    log(f"tokens[0,:8]={tokens[0,:8].tolist()} sha_note=caller_verified")

    # embed + expand
    h = model.embed(tokens)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    log(f"h0 L2={float(h.float().norm()):.6f}")

    probe_rows = [int(x) for x in args.probes.split(",")]
    probe_rows = [r for r in probe_rows if r < args.seq]

    summary: Dict[str, Any] = {
        "seq": args.seq,
        "index_topk": margs.index_topk,
        "window_size": margs.window_size,
        "reference_budget_rule": (
            "topk_idxs = cat([window_idxs, compress_topk_idxs], dim=-1); "
            "index_topk applies only to compressed half; window is exempt"
        ),
        "layers": {},
    }

    # Run layers sequentially; for non-target layers just forward to advance state
    # We only need per-layer fresh forward from same h0 for index compare of that layer alone
    # (indexer reads post-attn_norm activations of THAT layer). So for each target layer,
    # re-embed and run layers 0..L to get correct input activations.
    for L in layers:
        log(f"\n######## layer {L} ratio={margs.compress_ratios[L]} ########")
        # fresh forward to layer L
        for layer in model.layers:
            layer.attn.kv_cache.zero_()
            if getattr(layer.attn, "compress_ratio", 0):
                if hasattr(layer.attn, "compressor"):
                    layer.attn.compressor.kv_state.zero_()
                    layer.attn.compressor.score_state.fill_(float("-inf"))
                if getattr(layer.attn, "indexer", None) is not None:
                    layer.attn.indexer.kv_cache.zero_()
                    layer.attn.indexer.compressor.kv_state.zero_()
                    layer.attn.indexer.compressor.score_state.fill_(float("-inf"))
                    layer.attn.indexer.compressor.kv_cache = None
                layer.attn.compressor.kv_cache = None

        hh = h.clone()
        for i in range(L):
            # full block forward to produce correct residual into layer L
            # Need experts for correctness of residual — but we loaded experts=False!
            # For INDEX SETS we only need attn_norm(x) of layer L, which depends on
            # residual streams through prior layers. Without experts residual is wrong.
            #
            # Reload with experts for layers we traverse, OR only test each layer
            # from a synthetic/captured residual.
            #
            # Pragmatic: load experts for build. Re-build if needed.
            pass

        # If experts missing, MoE out is wrong. Detect:
        has_exp = any(
            p.numel() > 0 and p.abs().sum() > 0
            for n, p in model.layers[0].ffn.named_parameters()
            if "experts.0" in n
        )
        if not has_exp and L > 0:
            log("  WARNING: experts not loaded; layer input for L>0 is wrong if we chain.")
            log("  For index-set on layer L we still run hc_pre+attn on WHATEVER residual we have.")
            log("  Prefer --layers 2 with experts loaded for true scores.")

        # chain with whatever weights we have
        for i in range(L):
            hh = model.layers[i](hh, 0, tokens)

        cap = capture_layer_indexes(model.layers[L], hh, tokens, L, probe_rows)
        summary["layers"][str(L)] = cap
        b = cap["budget"]
        log(
            f"  budget: ratio={b['compress_ratio']} n_comp={b['n_comp_slots']} "
            f"topk_k={b['topk_k_used']} filters={b['topk_filters']} "
            f"window_in_budget={b['window_in_topk_budget']} joint_width={b['joint_width']}"
        )
        log(f"  NOTE: {b['note']}")
        for r, e in cap["probes"].items():
            log(
                f"  row {r:>4}: win_set={e['window_set_size']} (expect {e['window_expect_size']}) "
                + (
                    f"comp_set={e.get('compress_raw_set_size')} n_vis={e.get('n_visible_comp')} "
                    f"joint={e['joint_set_size']}"
                    if "compress_raw_set_size" in e
                    else f"joint={e['joint_set_size']}"
                )
            )
            if e.get("compress_raw_set_sample") is not None:
                log(f"           comp_sample={e['compress_raw_set_sample'][:16]}")

    # Cross-layer structural asserts
    log("\n=== STRUCTURAL VERDICTS (reference) ===")
    for Ls, cap in summary["layers"].items():
        b = cap["budget"]
        log(
            f"  L{Ls}: topk_filters_active={b['topk_filters']} "
            f"(n_comp={b['n_comp_slots']} vs index_topk={b['index_topk_config']})"
        )
        if b["compress_ratio"] and not b["topk_filters"]:
            log(
                f"       → at seq={args.seq}, top-k is a NO-OP (selects all visible compressed slots). "
                f"Cannot localise a pure selection bug here; need seq > index_topk * ratio "
                f"= {b['index_topk_config'] * max(b['compress_ratio'],1)} to exercise filtering."
            )
        if b["window_in_topk_budget"]:
            log("       → WINDOW COUNTED IN TOPK BUDGET (unexpected for reference)")
        else:
            log("       → window EXEMPT from topk budget (cat composition) — reference ground truth")

    # Optional parent set diff
    if args.parent_idxs and Path(args.parent_idxs).exists():
        parent = json.loads(Path(args.parent_idxs).read_text())
        log("\n=== PARENT SET DIFF ===")
        for Ls, cap in summary["layers"].items():
            if Ls not in parent:
                continue
            for r, e in cap["probes"].items():
                if r not in parent[Ls]:
                    continue
                pref = set(e.get("compress_raw_set") or e.get("compress_raw_set_sample") or [])
                # if only sample, skip full diff
                if "compress_raw_set" not in e and e.get("compress_raw_set_size", 0) > 64:
                    # rebuild full set from dump file if parent has full
                    pass
                pset = set(parent[Ls][r].get("compress_raw_set", []))
                if not pset:
                    continue
                # For large sets parent must provide full lists
                st = set_stats(set(parent[Ls][r].get("compress_raw_set_full", pset)), pset)
                log(f"  L{Ls} row {r}: {st}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nwrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default="0,2,3", help="comma layers")
    ap.add_argument("--probes", default=",".join(str(p) for p in PROBE_ROWS))
    ap.add_argument("--out", default="/tmp/index_set_bisect.json")
    ap.add_argument("--parent-idxs", default=None)
    ap.add_argument("--with-experts", action="store_true", help="load MoE experts (needed to chain L>0 accurately)")
    args = ap.parse_args()
    # monkeypatch build if experts
    global build_model
    if args.with_experts:
        _orig = build_model
        def build_model(a, m, layers, device):
            n_build = max(layers) + 1
            full = a.n_layers
            a.n_layers = n_build
            a.n_mtp_layers = 0
            a.dspark_block_size = 0
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cpu"):
                model = Transformer(a)
            a.n_layers = full
            load_state_into_model(model, m, layers=range(n_build), load_experts=True, device=device, verbose=True)
            model.eval()
            return model
    run(args)


if __name__ == "__main__":
    main()
