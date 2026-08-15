#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Index-set bisect against model.py (verbatim) for DSA top-k selection.

Priority: compare selected compressed index *sets* at probe rows past 512
on tokens.bin (1024). Layers: L0 (ratio=0 control), L2 (ratio=4+indexer),
L3 (ratio=128 identity).

Reference budget rule (model.py:511-520):
  topk_idxs = cat([window_idxs, compress_topk_idxs], -1)
  index_topk applies ONLY to the compressed half; SWA window is EXEMPT.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _kernel_shim
sys.modules["kernel"] = _kernel_shim

import torch

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
from kernel_shim import act_quant, fp4_act_quant

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
)
PROBE_ROWS = [0, 63, 127, 255, 511, 512, 768, 1023]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_tokens(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    assert len(raw) >= n * 4
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


def build_model(args: ModelArgs, model_dir: Path, layers: List[int], device: str, with_experts: bool) -> Transformer:
    n_build = max(layers) + 1
    full = args.n_layers
    args.n_layers = n_build
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full
    load_state_into_model(
        model, model_dir, layers=range(n_build), load_experts=with_experts, device=device, verbose=True
    )
    model.eval()
    return model


def reset_layer_caches(model: Transformer) -> None:
    for layer in model.layers:
        attn = layer.attn
        attn.kv_cache.zero_()
        comp = getattr(attn, "compressor", None)
        if comp is not None:
            comp.kv_state.zero_()
            comp.score_state.fill_(float("-inf"))
            comp.kv_cache = None
        idxr = getattr(attn, "indexer", None)
        if idxr is not None:
            idxr.kv_cache.zero_()
            idxr.compressor.kv_state.zero_()
            idxr.compressor.score_state.fill_(float("-inf"))
            idxr.compressor.kv_cache = None


@torch.inference_mode()
def capture_layer_indexes(
    block: Block,
    x_hc: torch.Tensor,
    layer_id: int,
    probe_rows: List[int],
    index_topk_cfg: int,
) -> dict:
    attn: Attention = block.attn
    ratio = int(attn.compress_ratio)
    win = int(attn.window_size)
    bsz = x_hc.size(0)
    seqlen = x_hc.size(1)
    indexer = getattr(attn, "indexer", None)

    y, post, comb = block.hc_pre(x_hc, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    x = block.attn_norm(y)

    freqs_cis = attn.freqs_cis[:seqlen]
    rd = attn.rope_head_dim
    if ratio and getattr(attn, "compressor", None) is not None and attn.compressor.kv_cache is None:
        attn.compressor.kv_cache = attn.kv_cache[:, win:]
        attn.compressor.freqs_cis = attn.freqs_cis
        if indexer is not None:
            indexer.freqs_cis = attn.freqs_cis

    qr = attn.q_norm(attn.wq_a(x))

    kv = attn.kv_norm(attn.wkv(x))
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    act_quant(kv[..., :-rd], 64, model_mod.scale_fmt, model_mod.scale_dtype, True)

    window_idxs = get_window_topk_idxs(win, bsz, seqlen, 0)
    compress_idxs = None
    compress_idxs_raw = None
    index_scores = None
    n_comp = 0
    topk_k = 0
    offset = int(kv.size(1))

    if ratio:
        n_comp = seqlen // ratio
        if indexer is not None:
            if indexer.compressor.kv_cache is None:
                indexer.compressor.kv_cache = indexer.kv_cache
                indexer.compressor.freqs_cis = indexer.freqs_cis
            q_i = indexer.wq_b(qr).unflatten(-1, (indexer.n_local_heads, indexer.head_dim))
            apply_rotary_emb(q_i[..., -rd:], freqs_cis)
            q_i = rotate_activation(q_i)
            fp4_act_quant(q_i, 32, True)
            indexer.compressor(x, 0)
            weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)
            kv_c = indexer.kv_cache[:bsz, :n_comp]
            index_score = torch.einsum("bshd,btd->bsht", q_i, kv_c)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
            mask = torch.arange(n_comp).repeat(seqlen, 1) >= (
                torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(
                mask, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            index_scores = index_score.detach()
            topk_k = min(int(indexer.index_topk), n_comp)
            topk_idxs = index_score.topk(topk_k, dim=-1)[1]
            mask2 = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            compress_idxs = torch.where(mask2, torch.tensor(-1), topk_idxs + offset).int()
            compress_idxs_raw = torch.where(mask2, torch.tensor(-1), topk_idxs).int()
        else:
            compress_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, 0, offset)
            compress_idxs_raw = get_compress_topk_idxs(ratio, bsz, seqlen, 0, 0)
            topk_k = n_comp

    joint = window_idxs if compress_idxs is None else torch.cat([window_idxs, compress_idxs], dim=-1)

    budget = {
        "window_size": win,
        "index_topk_config": int(index_topk_cfg),
        "compress_ratio": ratio,
        "n_comp_slots": n_comp,
        "topk_k_used": topk_k,
        "topk_filters": bool(indexer is not None and n_comp > int(index_topk_cfg)),
        "window_in_topk_budget": False,
        "joint_width": int(joint.size(-1)),
        "offset_used": offset,
        "note": (
            "reference cats window_idxs || compress_topk_idxs; "
            "index_topk applies ONLY to compressed slots; window EXEMPT"
        ),
    }

    probes = {}
    for r in probe_rows:
        if r >= seqlen:
            continue
        w_set = {int(v) for v in window_idxs[0, r].tolist() if v >= 0}
        entry: Dict[str, Any] = {
            "window_set_size": len(w_set),
            "window_expect_size": min(r + 1, win),
            "window_set_ok": len(w_set) == min(r + 1, win),
        }
        if compress_idxs_raw is not None:
            c_list = compress_idxs_raw[0, r].tolist()
            c_set = {int(v) for v in c_list if v >= 0}
            c_off_set = {int(v) for v in compress_idxs[0, r].tolist() if v >= 0}
            n_vis = (r + 1) // ratio
            entry["compress_raw_set_size"] = len(c_set)
            entry["n_visible_comp"] = n_vis
            entry["compress_selects_all_visible"] = len(c_set) == min(topk_k, n_vis) or (
                topk_k >= n_vis and len(c_set) == n_vis
            )
            # full set for set-diff (always; 256 ints is fine)
            entry["compress_raw_set"] = sorted(c_set)
            entry["compress_offset_set"] = sorted(c_off_set)
            entry["compress_raw_sample"] = sorted(c_set)[:16]
            if index_scores is not None and c_set:
                sc = index_scores[0, r, :n_comp]
                sel_scores = sorted((float(sc[i]) for i in c_set))
                entry["selected_score_min"] = sel_scores[0]
                entry["selected_score_max"] = sel_scores[-1]
                min_sel = sel_scores[0]
                n_ge = int(((sc >= min_sel) & torch.isfinite(sc)).sum().item())
                entry["n_scores_ge_min_selected"] = n_ge
                entry["n_nonselected_ge_min_selected"] = max(n_ge - len(c_set), 0)
        j_set = {int(v) for v in joint[0, r].tolist() if v >= 0}
        entry["joint_set_size"] = len(j_set)
        # window and compressed must be disjoint in unified index space
        if compress_idxs is not None:
            c_off = {int(v) for v in compress_idxs[0, r].tolist() if v >= 0}
            entry["window_comp_overlap"] = sorted(w_set & c_off)
            entry["window_comp_disjoint"] = len(w_set & c_off) == 0
        probes[str(r)] = entry

    return {
        "layer_id": layer_id,
        "ratio": ratio,
        "has_indexer": indexer is not None,
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
    log(f"layers={layers} compress_ratios={[margs.compress_ratios[i] for i in layers]}")
    log(
        f"window={margs.window_size} index_topk={margs.index_topk} "
        f"compress_rope_theta={margs.compress_rope_theta}"
    )

    t0 = time.time()
    model = build_model(margs, model_dir, layers, "cpu", with_experts=args.with_experts)
    log(f"load done in {time.time()-t0:.1f}s experts={args.with_experts}")

    tokens = load_tokens(Path(args.tokens), args.seq)
    log(f"tokens[0,:8]={tokens[0,:8].tolist()}")

    h0 = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    log(f"h0 L2={float(h0.float().norm()):.6f}")

    probe_rows = [int(x) for x in args.probes.split(",") if int(x) < args.seq]

    summary: Dict[str, Any] = {
        "seq": args.seq,
        "index_topk": margs.index_topk,
        "window_size": margs.window_size,
        "reference_budget_rule": (
            "topk_idxs = cat([window_idxs, compress_topk_idxs], dim=-1); "
            "index_topk applies only to compressed half; window is EXEMPT"
        ),
        "seq_needed_for_topk_filter_ratio4": margs.index_topk * 4,
        "seq_needed_for_topk_filter_ratio128": margs.index_topk * 128,
        "layers": {},
    }

    for L in layers:
        log(f"\n######## layer {L} ratio={margs.compress_ratios[L]} ########")
        reset_layer_caches(model)
        hh = h0.clone()
        for i in range(L):
            t1 = time.time()
            hh = model.layers[i](hh, 0, tokens)
            log(f"  ran L{i} in {time.time()-t1:.1f}s residual_L2={float(hh.float().norm()):.4f}")

        cap = capture_layer_indexes(
            model.layers[L], hh, L, probe_rows, margs.index_topk
        )
        summary["layers"][str(L)] = cap
        b = cap["budget"]
        log(
            f"  budget: ratio={b['compress_ratio']} n_comp={b['n_comp_slots']} "
            f"topk_k={b['topk_k_used']} filters={b['topk_filters']} "
            f"window_in_budget={b['window_in_topk_budget']} joint_width={b['joint_width']} "
            f"offset={b['offset_used']}"
        )
        log(f"  NOTE: {b['note']}")
        for r, e in cap["probes"].items():
            extra = ""
            if "compress_raw_set_size" in e:
                extra = (
                    f" comp={e['compress_raw_set_size']}/{e['n_visible_comp']} "
                    f"all_vis={e['compress_selects_all_visible']} "
                    f"disjoint={e.get('window_comp_disjoint')} "
                    f"sample={e.get('compress_raw_sample')}"
                )
            log(
                f"  row {r:>4}: win={e['window_set_size']}/{e['window_expect_size']} "
                f"ok={e['window_set_ok']} joint={e['joint_set_size']}{extra}"
            )

    log("\n=== STRUCTURAL VERDICTS (reference ground truth) ===")
    for Ls, cap in summary["layers"].items():
        b = cap["budget"]
        log(
            f"  L{Ls}: topk_filters_active={b['topk_filters']} "
            f"(n_comp={b['n_comp_slots']} vs index_topk={b['index_topk_config']})"
        )
        if b["compress_ratio"] and not b["topk_filters"]:
            need = b["index_topk_config"] * max(b["compress_ratio"], 1)
            log(
                f"       → at seq={args.seq}, top-k is a NO-OP on compressed slots. "
                f"Need seq > {need} to exercise selection filtering on this ratio."
            )
        log(
            f"       → window EXEMPT from topk budget: {not b['window_in_topk_budget']} "
            f"(reference cat composition)"
        )
        # probe past-512 summary
        for r in ("512", "768", "1023"):
            if r in cap["probes"] and "compress_raw_set_size" in cap["probes"][r]:
                e = cap["probes"][r]
                log(
                    f"       row {r}: comp_set={e['compress_raw_set_size']} "
                    f"n_vis={e['n_visible_comp']} selects_all_visible={e['compress_selects_all_visible']}"
                )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default="0,2,3")
    ap.add_argument("--probes", default=",".join(str(p) for p in PROBE_ROWS))
    ap.add_argument("--out", default="/tmp/index_set_bisect.json")
    ap.add_argument("--with-experts", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
