#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Independent DeepSeek-V4 reference oracle.

Imports model.py VERBATIM with kernel_shim replacing tilelang kernels.
Gates:
  1. Floor — Linear fp8 path, reported in BOTH f32-diagnostic and bf16-fidelity
  2. hc_post fix agreement + loud fail on deliberately transposed comb
  3. Layer-0 stage bisect (hc_pre, attn_norm, attention internals, hc_post, ffn, out)
  4. Residual L2 trajectory for as many layers as memory allows

Default device: CPU. Stay off the GPU while parent captures are chaining.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent

def _find_ref_infer() -> Path:
    if (HERE / "model.py").is_file() and (HERE / "model.py").stat().st_size > 0:
        # symlink or real file with content
        return HERE
    for parent in HERE.parents:
        cand = parent / ".codeinsight+research" / "ds4-parent-ref" / "inference"
        if (cand / "model.py").is_file():
            return cand
    return HERE

REF_INFER = _find_ref_infer()
sys.path.insert(0, str(HERE))

import kernel_shim as _kernel_shim  # noqa: E402
sys.modules["kernel"] = _kernel_shim

if (HERE / "model.py").exists():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(REF_INFER))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import (  # noqa: E402
    ModelArgs,
    Transformer,
    Block,
    Attention,
    Linear,
    linear as model_linear,
    apply_rotary_emb,
    get_window_topk_idxs,
    precompute_freqs_cis,
    set_dtype,
)
import model as model_mod  # noqa: E402
from weight_loader import (  # noqa: E402
    build_tensor_index,
    dequant_fp8_block128,
    load_state_into_model,
    _load_tensor,
    _ue8m0_to_f32,
)
from parent_hc_post_ref import parent_hc_post_explicit  # noqa: E402
from kernel_shim import act_quant, fp8_gemm, sparse_attn, hc_split_sinkhorn  # noqa: E402

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens_128.bin"
)
# Post-combfix parent residual L2 at 128 tok (from Gate narrative)
POST_FIX_PARENT_L2_128 = [178.19]  # geo-mean growth 1.168 over 43 layers ends 122766
PRE_FIX_PARENT_L2 = [
    494.179871, 474.714539, 483.457733, 482.975098, 486.401825, 777.972900, 1188.696289,
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_tokens(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    assert len(raw) >= n * 4, (len(raw), n)
    ids = struct.unpack("<" + "i" * n, raw[: n * 4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64 = a.detach().float().reshape(-1).double()
    b64 = b.detach().float().reshape(-1).double()
    diff = (a64 - b64).abs()
    denom = b64.norm().clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(a64.unsqueeze(0), b64.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rel_fro": float(diff.norm() / denom),
        "cosine": float(cos),
        "l2_a": float(a64.norm()),
        "l2_b": float(b64.norm()),
    }


def make_args(config_path: Path, max_seq_len: int = 256) -> ModelArgs:
    with open(config_path) as f:
        cfg = json.load(f)
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k, v in cfg.items():
        if k not in fields:
            continue
        if k == "compress_ratios":
            kwargs[k] = tuple(v)
        elif k == "dspark_target_layer_ids":
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    args = ModelArgs(**kwargs)
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    return args


def configure_model_globals(args: ModelArgs, force_bf16_weights: bool = False) -> None:
    """Mirror Transformer.__init__ global side effects without building full model."""
    model_mod.world_size = 1
    model_mod.rank = 0
    if force_bf16_weights:
        model_mod.default_dtype = torch.bfloat16
    else:
        model_mod.default_dtype = (
            torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        )
    model_mod.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    model_mod.scale_dtype = (
        torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    )
    torch.set_default_dtype(torch.bfloat16)


# ===========================================================================
# Step 1 — floors (f32 diagnostic + bf16 fidelity)
# ===========================================================================

def _linear_floor_pair(
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    *,
    label: str,
) -> dict:
    """Side A: model_linear (act_quant+fp8_gemm shim). Side B: full dequant f32 matmul."""
    w = w.clone()
    w.scale = s if s.dtype == torch.float8_e8m0fnu else s.view(torch.float8_e8m0fnu)
    y_a = model_linear(x, w, None).float()
    # Side B: dequant act via same act_quant codes, dequant weight, f32 matmul
    x_q, x_s = act_quant(
        x.float() if x.dtype != torch.bfloat16 else x,
        128,
        model_mod.scale_fmt,
        model_mod.scale_dtype,
        inplace=False,
    )
    in_f = w.shape[1]
    xs = _ue8m0_to_f32(x_s).reshape(-1, in_f // 128)
    xa = x_q.to(torch.float32).reshape(-1, in_f)
    xa = xa * xs.repeat_interleave(128, dim=-1)
    wd = dequant_fp8_block128(w, w.scale)
    y_b = (xa @ wd.t()).reshape(y_a.shape)
    m = metrics(y_a, y_b)
    log(
        f"  [{label}] max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
        f"rel_fro={m['rel_fro']:.6e} cosine={m['cosine']:.8f}"
    )
    return m


def step1_floor(model_dir: Path, device: str) -> dict:
    log("=== STEP 1: floors (Linear fp8) ===")
    index = build_tensor_index(model_dir)
    w_name = "layers.0.attn.wq_a.weight"
    s_name = "layers.0.attn.wq_a.scale"
    w = _load_tensor(index, w_name)
    s = _load_tensor(index, s_name)
    assert w.dtype == torch.float8_e4m3fn, w.dtype
    out_f, in_f = w.shape
    log(f"  weight: {w_name} {tuple(w.shape)} scale {tuple(s.shape)}")

    torch.manual_seed(0)
    # --- f32 diagnostic floor: f32 activations into act_quant+fp8 path vs dequant ---
    model_mod.block_size = 128
    model_mod.scale_fmt = "ue8m0"
    model_mod.scale_dtype = torch.float8_e8m0fnu
    model_mod.default_dtype = torch.bfloat16  # gemm out dtype
    torch.set_default_dtype(torch.bfloat16)

    x_f32 = torch.randn(8, in_f, dtype=torch.float32)
    # Cast path: reference linear always act_quants then fp8_gemms; input dtype only
    # affects the pre-quant representation. f32 input → tighter pre-quant amax.
    m_f32 = _linear_floor_pair(x_f32, w, s, label="f32-input diagnostic")

    x_bf = torch.randn(8, in_f, dtype=torch.bfloat16)
    m_bf = _linear_floor_pair(x_bf, w, s, label="bf16-input fidelity")

    # Plain f32 matmul identity (no quant) — absolute zero floor for pure arithmetic
    wd = dequant_fp8_block128(w, s if s.dtype == torch.float8_e8m0fnu else s.view(torch.float8_e8m0fnu))
    y1 = x_f32 @ wd.t()
    y2 = F.linear(x_f32, wd)
    m_id = metrics(y1, y2)
    log(f"  plain f32 matmul identity max_abs={m_id['max_abs']:.6e} (expect 0)")

    # Pure f32 Linear (no quant): known-correct both sides identical ops
    w_bf = wd.to(torch.bfloat16)
    y_ref = (x_f32 @ w_bf.float().t())
    y_lin = F.linear(x_f32, w_bf.float())
    m_pure = metrics(y_ref, y_lin)
    log(f"  pure f32 F.linear identity max_abs={m_pure['max_abs']:.6e}")

    log(f"  FLOOR_F32_DIAG max_abs = {m_f32['max_abs']:.6e}")
    log(f"  FLOOR_BF16_FIDELITY max_abs = {m_bf['max_abs']:.6e}")
    log(
        "  note: fp8 act_quant+GEMM cannot floor below ~bf16/fp8 quant noise; "
        "pure f32 stages (hc_post, softmax, rope) floor near 0 / 1e-6."
    )
    return {
        "floor_f32_diag_max_abs": m_f32["max_abs"],
        "floor_f32_diag": m_f32,
        "floor_bf16_fidelity_max_abs": m_bf["max_abs"],
        "floor_bf16_fidelity": m_bf,
        "floor_pure_f32_identity": m_pure["max_abs"],
        # backward-compat key used by step2 threshold narrative
        "floor_max_abs": m_f32["max_abs"],
    }


# ===========================================================================
# Model build / load
# ===========================================================================

def build_partial_model(
    args: ModelArgs, model_dir: Path, n_layers: int, device: str
) -> Transformer:
    full_layers = args.n_layers
    args.n_layers = n_layers
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full_layers
    log(f"  constructed Transformer with {n_layers} blocks (mtp disabled)")
    load_state_into_model(
        model,
        model_dir,
        layers=range(n_layers),
        load_experts=True,
        device=device,
        verbose=True,
    )
    model.eval()
    return model


# ===========================================================================
# Step 2 — hc_post teeth
# ===========================================================================

@torch.inference_mode()
def step2_hc_post(model: Transformer, floor_f32: float, device: str) -> dict:
    log("=== STEP 2: hc_post contraction (fix vs deliberate transpose) ===")
    block: Block = model.layers[0]
    torch.manual_seed(1)
    rows, hc, dim = 32, 4, 4096
    # Work entirely in f32 for the contraction check — model.hc_post does
    # type_as(x) at the end; feed f32 x so comparison is not polluted by bf16 cast.
    residual = torch.randn(1, rows, hc, dim, dtype=torch.float32, device=device)
    y_stream, post, comb = block.hc_pre(
        residual.to(torch.bfloat16),  # hc_pre path uses model weights in their native path
        block.hc_attn_fn,
        block.hc_attn_scale,
        block.hc_attn_base,
    )
    # Recompute post/comb in f32 from the same residual for clean algebra
    # (hc_pre returns bf16 y_stream; post/comb are f32 from sinkhorn)
    post = post.float()
    comb = comb.float()
    x_attn = torch.randn(1, rows, dim, dtype=torch.float32, device=device)
    residual_f = residual.float()

    y_model = block.hc_post(x_attn, residual_f, post, comb).float()
    y_parent = parent_hc_post_explicit(
        x_attn, residual_f, post, comb, transpose_comb=False
    ).float()
    y_bug = parent_hc_post_explicit(
        x_attn, residual_f, post, comb, transpose_comb=True
    ).float()

    m_ok = metrics(y_model, y_parent)
    m_bug = metrics(y_model, y_bug)
    log(
        f"  hc_post FIXED       max_abs={m_ok['max_abs']:.6e} rel_fro={m_ok['rel_fro']:.6e} "
        f"cosine={m_ok['cosine']:.8f}"
    )
    log(
        f"  hc_post TRANSPOSED  max_abs={m_bug['max_abs']:.6e} rel_fro={m_bug['rel_fro']:.6e} "
        f"cosine={m_bug['cosine']:.8f}"
    )

    # Pure f32 algebra: fixed must be ~0; transpose must be O(1) relative
    ok_pass = m_ok["max_abs"] < 1e-5
    bug_fail = m_bug["max_abs"] > 1e-2 and m_bug["max_abs"] > 100 * max(m_ok["max_abs"], 1e-12)
    log(
        f"  VERDICT fixed:             {'PASS' if ok_pass else 'FAIL'} "
        f"(threshold 1e-5; pure-f32 algebra floor ~0; linear-floor was {floor_f32:.3e})"
    )
    log(
        f"  VERDICT transpose-detect:  {'PASS (loud fail)' if bug_fail else 'FAIL (silent)'}"
    )
    log(
        f"  magnitude ratio transposed/fixed = "
        f"{m_bug['max_abs'] / max(m_ok['max_abs'], 1e-30):.1f}x"
    )

    # Inline one-liner identity
    y_inline = (
        post.unsqueeze(-1) * x_attn.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual_f.unsqueeze(-2), dim=2)
    )
    m_inline = metrics(y_model, y_inline)
    log(f"  model.hc_post vs inline one-liner max_abs={m_inline['max_abs']:.6e} (expect 0)")

    return {
        "hc_post_fixed": m_ok,
        "hc_post_transposed": m_bug,
        "hc_post_fixed_pass": ok_pass,
        "hc_post_transpose_detected": bug_fail,
        "ratio_bug_over_fix": m_bug["max_abs"] / max(m_ok["max_abs"], 1e-30),
    }


# ===========================================================================
# Step 3 — layer-0 stage bisect
# ===========================================================================

@torch.inference_mode()
def _attn_stages(attn: Attention, x: torch.Tensor, start_pos: int = 0) -> Dict[str, torch.Tensor]:
    """Mirror Attention.forward with intermediate captures. Layer-0: compress_ratio=0."""
    bsz, seqlen, _ = x.size()
    freqs_cis = attn.freqs_cis[start_pos : start_pos + seqlen]
    win = attn.window_size
    rd = attn.rope_head_dim
    eps = attn.eps

    out: Dict[str, torch.Tensor] = {}
    # q path
    qr = q = attn.q_norm(attn.wq_a(x))
    out["q_lat"] = qr.detach()
    q = attn.wq_b(q).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    out["q_post_wb"] = q.detach()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    out["q_post_head_rms"] = q.detach()
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    out["q_post_rope"] = q.detach()

    # kv path
    kv = attn.wkv(x)
    kv = attn.kv_norm(kv)
    out["kv_post_norm"] = kv.detach()
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    out["kv_post_rope"] = kv.detach()
    act_quant(kv[..., :-rd], 64, model_mod.scale_fmt, model_mod.scale_dtype, True)
    out["kv_post_quant"] = kv.detach()

    topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    out["window_idxs"] = topk_idxs.detach()

    # L0: compress_ratio == 0 — no compressor/indexer
    assert not attn.compress_ratio, (
        f"layer compress_ratio={attn.compress_ratio}; this bisect expects L0 ratio=0"
    )

    if start_pos == 0:
        if seqlen <= win:
            attn.kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            attn.kv_cache[:bsz, cutoff:win], attn.kv_cache[:bsz, :cutoff] = kv[
                :, -win:
            ].split([win - cutoff, cutoff], dim=1)
        o = sparse_attn(q, kv, attn.attn_sink, topk_idxs, attn.softmax_scale)
    else:
        attn.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        o = sparse_attn(
            q, attn.kv_cache[:bsz], attn.attn_sink, topk_idxs, attn.softmax_scale
        )
    out["attn_raw"] = o.detach()
    apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    out["attn_inv_rope"] = o.detach()

    o_g = o.view(bsz, seqlen, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o_a = torch.einsum("bsgd,grd->bsgr", o_g, wo_a)
    out["wo_a_out"] = o_a.detach()
    x_out = attn.wo_b(o_a.flatten(2))
    out["attn_out"] = x_out.detach()
    return out


@torch.inference_mode()
def step3_layer0_bisect(
    model: Transformer,
    tokens: torch.Tensor,
    floors: dict,
    device: str,
    parent_stage_l2: Optional[Dict[str, float]] = None,
) -> dict:
    log("=== STEP 3: layer-0 stage bisect ===")
    tokens = tokens.to(device)
    block: Block = model.layers[0]
    attn: Attention = block.attn
    assert attn.compress_ratio == 0, attn.compress_ratio

    h0 = model.embed(tokens)
    h0 = h0.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    out: Dict[str, Any] = {"embed_hc_l2": float(h0.float().norm())}
    log(f"  embed+hc_expand L2={out['embed_hc_l2']:.6f} shape={tuple(h0.shape)}")

    # --- attn half ---
    residual = h0
    y, post, comb = block.hc_pre(
        residual, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    out["hc_pre_attn"] = {
        "y_l2": float(y.float().norm()),
        "post_l2": float(post.float().norm()),
        "comb_l2": float(comb.float().norm()),
    }
    log(
        f"  hc_pre_attn  y_l2={out['hc_pre_attn']['y_l2']:.6f} "
        f"post_l2={out['hc_pre_attn']['post_l2']:.6f} comb_l2={out['hc_pre_attn']['comb_l2']:.6f}"
    )

    x_norm = block.attn_norm(y)
    out["attn_norm_l2"] = float(x_norm.float().norm())
    log(f"  attn_norm    L2={out['attn_norm_l2']:.6f}")

    # Attention internals
    stages = _attn_stages(attn, x_norm, 0)
    stage_l2s = {k: float(v.float().norm()) for k, v in stages.items() if torch.is_floating_point(v)}
    out["attn_stages_l2"] = stage_l2s
    for k, v in stage_l2s.items():
        log(f"  attn.{k:16s} L2={v:.6f}")

    attn_out = stages["attn_out"]
    out["attn_out_l2"] = float(attn_out.float().norm())

    # Sanity: full Attention.forward matches staged path
    # (need fresh kv_cache — zero it)
    attn.kv_cache.zero_()
    attn_out_ref = attn(x_norm, 0)
    m_attn = metrics(attn_out, attn_out_ref)
    out["staged_vs_attn_forward"] = m_attn
    log(
        f"  staged attn vs Attention.forward max_abs={m_attn['max_abs']:.6e} "
        f"(expect ~0; domain f32 compare of bf16 outs)"
    )

    h_post = block.hc_post(attn_out, residual, post, comb)
    out["hc_post_attn_l2"] = float(h_post.float().norm())
    log(f"  hc_post_attn L2={out['hc_post_attn_l2']:.6f}")

    # --- ffn half ---
    residual2 = h_post
    y2, post2, comb2 = block.hc_pre(
        residual2, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
    )
    out["hc_pre_ffn_l2"] = float(y2.float().norm())
    x_ffn = block.ffn_norm(y2)
    out["ffn_norm_l2"] = float(x_ffn.float().norm())
    log(f"  hc_pre_ffn   L2={out['hc_pre_ffn_l2']:.6f}")
    log(f"  ffn_norm     L2={out['ffn_norm_l2']:.6f}")

    t0 = time.time()
    moe_out = block.ffn(x_ffn, tokens)
    dt = time.time() - t0
    out["moe_out_l2"] = float(moe_out.float().norm())
    log(f"  moe_out      L2={out['moe_out_l2']:.6f}  ({dt:.1f}s)")

    h1 = block.hc_post(moe_out, residual2, post2, comb2)
    out["layer_out_l2"] = float(h1.float().norm())
    log(f"  layer_out    L2={out['layer_out_l2']:.6f}")

    # Full Block.forward cross-check
    attn.kv_cache.zero_()
    if hasattr(attn, "compressor") and attn.compress_ratio:
        pass
    h1_full = block(h0, 0, tokens)
    m_block = metrics(h1, h1_full)
    out["staged_vs_block_forward"] = m_block
    log(
        f"  staged layer vs Block.forward max_abs={m_block['max_abs']:.6e} "
        f"rel_fro={m_block['rel_fro']:.6e}"
    )

    # Per-stage floor notes (what comparison CAN resolve)
    per_stage_floor = {
        "hc_pre / hc_post / sinkhorn": "pure f32 algebra → floor ~0 (1e-7)",
        "attn_norm / ffn_norm": "f32 RMSNorm → floor ~1e-7",
        "q/k/v projections (fp8 Linear)": f"fp8 quant+gemm → floor ~{floors.get('floor_f32_diag_max_abs', float('nan')):.3e} (f32-in) / {floors.get('floor_bf16_fidelity_max_abs', float('nan')):.3e} (bf16-in)",
        "RoPE": "f32 complex mul → floor ~1e-7",
        "act_quant kv nope": "fp8 block-64 quant → ~bf16 unit 3.9e-3 on those dims",
        "sparse_attn softmax+sink": "f32 online softmax → floor ~1e-6; shim uses full softmax not tiled online",
        "wo_a einsum": "bf16 weight (dequant on load) × bf16 act → ~bf16",
        "wo_b (fp8 Linear)": f"fp8 → floor ~ linear floor",
        "moe experts (fp4)": "fp4 gemm → higher; not the L0 attention suspect",
    }
    out["per_stage_floor_notes"] = per_stage_floor
    log("  --- per-stage floor notes ---")
    for k, v in per_stage_floor.items():
        log(f"    {k}: {v}")

    # Compare L2 trajectory vs parent stage dump if provided
    if parent_stage_l2:
        log("  --- vs parent stage L2 (same tokens) ---")
        mapping = {
            "hc_pre_attn": out["hc_pre_attn"]["y_l2"],
            "attn_norm": out["attn_norm_l2"],
            "attn_out": out["attn_out_l2"],
            "hc_post_attn": out["hc_post_attn_l2"],
            "hc_pre_ffn": out["hc_pre_ffn_l2"],
            "ffn_norm": out["ffn_norm_l2"],
            "moe_out": out["moe_out_l2"],
            "hc_post_ffn": out["layer_out_l2"],
        }
        rows = []
        first_leave = None
        for name, ref_l2 in mapping.items():
            par = parent_stage_l2.get(name)
            if par is None:
                continue
            rel = abs(ref_l2 - par) / max(par, 1e-30)
            abs_d = abs(ref_l2 - par)
            # L2 scalar compare is coarse; flag >1% relative as leave
            leave = rel > 0.01 and abs_d > 0.5
            mark = "LEAVE" if leave else "ok"
            if leave and first_leave is None:
                first_leave = name
            log(
                f"    {name:14s}  ref={ref_l2:.6f}  parent={par:.6f}  "
                f"abs_d={abs_d:.6f} rel={rel:.6e}  {mark}"
            )
            rows.append(
                {
                    "stage": name,
                    "ref_l2": ref_l2,
                    "parent_l2": par,
                    "abs_d": abs_d,
                    "rel": rel,
                    "leave": leave,
                }
            )
        out["parent_l2_compare"] = rows
        out["first_stage_leaving_l2"] = first_leave
        log(f"  FIRST stage leaving L2-floor vs parent: {first_leave}")

    # Internal attention stage self-consistency floors (recompute key pieces two ways)
    log("  --- attention internal self-checks (domain named) ---")
    # RoPE roundtrip: apply + inverse on q should recover head-rms (nope dims exact; rope ~1e-6)
    q_rms = stages["q_post_head_rms"].float()
    q_rope = stages["q_post_rope"].clone()
    freqs = attn.freqs_cis[: tokens.size(1)]
    q_back = q_rope.clone()
    apply_rotary_emb(q_back[..., -attn.rope_head_dim :], freqs, True)
    m_rope = metrics(q_back.float(), q_rms)
    log(
        f"  rope forward+inverse vs pre-rope max_abs={m_rope['max_abs']:.6e} "
        f"(f32 complex; expect ~1e-6)"
    )
    out["rope_roundtrip"] = m_rope

    # Sink contribution: softmax without sink vs with sink should differ
    # (proves sink term is active)
    q = stages["q_post_rope"]
    kv = stages["kv_post_quant"]
    idxs = stages["window_idxs"]
    sink = attn.attn_sink
    o_with = sparse_attn(q, kv, sink, idxs, attn.softmax_scale)
    o_nosink = sparse_attn(q, kv, torch.full_like(sink, -1e9), idxs, attn.softmax_scale)
    m_sink = metrics(o_with, o_nosink)
    log(
        f"  sink active? with vs -1e9-sink max_abs={m_sink['max_abs']:.6e} "
        f"(must be >> 0 if sink contributes)"
    )
    out["sink_delta"] = m_sink

    # Window coverage at 128: every row should see 1..128 valid keys (causal)
    valid_counts = (idxs >= 0).sum(dim=-1).float().squeeze(0)
    log(
        f"  window valid counts: min={int(valid_counts.min())} max={int(valid_counts.max())} "
        f"(expect min=1 max=128 at seq=128 window=128)"
    )
    out["window_valid_min"] = int(valid_counts.min())
    out["window_valid_max"] = int(valid_counts.max())

    return out


# ===========================================================================
# Step 4 — residual trajectory
# ===========================================================================

@torch.inference_mode()
def step4_trajectory(
    model: Transformer, tokens: torch.Tensor, n_layers: int, device: str
) -> list:
    log(f"=== STEP 4: residual L2 trajectory layers 0..{n_layers - 1} ===")
    tokens = tokens.to(device)
    h = model.embed(tokens)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    # zero all kv caches
    for layer in model.layers:
        layer.attn.kv_cache.zero_()
    traj = []
    for i in range(n_layers):
        t0 = time.time()
        h = model.layers[i](h, 0, tokens)
        l2 = float(h.float().norm())
        traj.append(l2)
        dt = time.time() - t0
        ratio = l2 / traj[i - 1] if i > 0 else float("nan")
        log(f"  L{i}: residual_L2={l2:.6f}  ratio_prev={ratio:.6f}  ({dt:.1f}s)")
    if len(traj) > 1:
        geo = math.exp(sum(math.log(max(traj[i] / traj[i - 1], 1e-30)) for i in range(1, len(traj))) / (len(traj) - 1))
        log(f"  geo-mean growth/layer = {geo:.6f}  (parent post-fix 1.168; production 1.114)")
    log("trajectory: " + ", ".join(f"{v:.6f}" for v in traj))
    return traj


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument(
        "--step",
        choices=["all", "1", "2", "3", "4"],
        default="all",
        help="1=floor 2=hc_post 3=layer0 bisect 4=trajectory",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--parent-stages",
        default=None,
        help="Optional path to parent_stages_only.txt for L2 compare",
    )
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

    model_dir = Path(args.model)
    if args.config:
        config_path = Path(args.config)
    elif (model_dir / "inference" / "config.json").exists():
        config_path = model_dir / "inference" / "config.json"
    elif (HERE / "config.json").exists():
        config_path = HERE / "config.json"
    else:
        config_path = REF_INFER / "config.json"

    log(f"model_dir={model_dir}")
    log(f"config={config_path}")
    log(
        f"device={args.device}  torch={torch.__version__}  "
        f"cuda_available={torch.cuda.is_available()}"
    )
    log(f"tokens={args.tokens}  seq={args.seq}  layers={args.layers}")

    summary: Dict[str, Any] = {"torch": torch.__version__, "device": args.device}

    if args.step in ("all", "1"):
        summary["step1"] = step1_floor(model_dir, args.device)
        floor_f32 = summary["step1"]["floor_f32_diag_max_abs"]
    else:
        floor_f32 = float("nan")

    need_model = args.step in ("all", "2", "3", "4")
    if need_model:
        margs = make_args(config_path, max_seq_len=max(args.seq, 256))
        log(
            f"ModelArgs: n_layers={margs.n_layers} dim={margs.dim} "
            f"route_scale={margs.route_scale} score_func={margs.score_func} "
            f"expert_dtype={margs.expert_dtype} scale_fmt={margs.scale_fmt} "
            f"window={margs.window_size} index_topk={margs.index_topk}"
        )
        n_build = 1 if args.step in ("2", "3") else args.layers
        if args.step == "all":
            n_build = args.layers
        model = build_partial_model(margs, model_dir, n_build, args.device)
        tokens = load_tokens(Path(args.tokens), args.seq)
        log(f"tokens[0,:8]={tokens[0, :8].tolist()}")

        if args.step in ("all", "2"):
            summary["step2"] = step2_hc_post(model, floor_f32, args.device)

        if args.step in ("all", "3"):
            parent_l2 = None
            # Built-in post-combfix parent L0 stage L2s at 128 tok if file given
            if args.parent_stages and Path(args.parent_stages).exists():
                parent_l2 = {}
                for line in Path(args.parent_stages).read_text().splitlines():
                    if "layer=0 " not in line or "PARENT_STAGE" not in line:
                        continue
                    # PARENT_STAGE layer=0 ratio=0 hc_pre_attn=... attn_norm=...
                    parts = line.split()
                    for p in parts:
                        if "=" in p and p.split("=")[0] not in ("layer", "ratio"):
                            k, v = p.split("=", 1)
                            try:
                                parent_l2[k] = float(v)
                            except ValueError:
                                pass
                    break
                log(f"  loaded parent L0 stages: {parent_l2}")
            floors = summary.get("step1", {})
            summary["step3"] = step3_layer0_bisect(
                model, tokens, floors, args.device, parent_l2
            )
            # persist hidden
            torch.save(
                {"tokens": tokens.cpu(), "step3": {k: v for k, v in summary["step3"].items() if not isinstance(v, torch.Tensor)}},
                HERE / "layer0_bisect_summary.pt",
            )

        if args.step in ("all", "4"):
            if len(model.layers) < args.layers:
                model = build_partial_model(margs, model_dir, args.layers, args.device)
            summary["step4_trajectory"] = step4_trajectory(
                model, tokens, args.layers, args.device
            )

    log("=== SUMMARY ===")

    def fmt(o):
        if isinstance(o, float):
            return round(o, 8) if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: fmt(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fmt(x) for x in o]
        if isinstance(o, bool):
            return o
        if isinstance(o, (int, str)) or o is None:
            return o
        return str(type(o))

    print(json.dumps(fmt(summary), indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(fmt(summary), indent=2))
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
