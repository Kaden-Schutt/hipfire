#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Self-calibrating L0 attn_out quantization floor (ref vs ref).

Runs the SAME L0 Attention forward twice on identical inputs:
  A) fp8 simulation ON  (kernel_shim act_quant + fp8_gemm as used by Linear)
  B) exact f32 path      (act_quant no-op dequant; Linear uses bf16/f32 matmul)

Cosine(A,B) is the quantization noise floor for attn_out with our port
not involved. Compare against parent-vs-ref attn_out cosine (~0.9993).

Also measures a single fp8 Linear floor on a known-shape GEMM matching
wq_a (4096→1536) using the same input activations from attn_norm.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernel_shim as _ks
sys.modules["kernel"] = _ks

import torch
import numpy as np

# residual_content helpers
from residual_content_dump import (
    load_tokens, make_args, free_cuda, load_embed_only, load_block_weights,
    DEFAULT_MODEL, DEFAULT_TOKENS,
)
from model import Block, Transformer, linear as model_linear
import model as model_mod
import kernel_shim

DEFAULT_POSITIONS = [0, 1, 64, 400, 448, 512, 800, 1023]


def log(m=""):
    print(m, flush=True)


def metrics(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0, 0.0, 1.0
    cos = float(np.dot(a, b) / (na * nb + 1e-300))
    rel = float(np.linalg.norm(a - b) / (nb + 1e-300))
    ratio = float(na / (nb + 1e-300))
    return cos, rel, ratio


def capture_pos(t: torch.Tensor, positions):
    x = t.detach().float()
    if x.dim() == 3 and x.size(0) == 1:
        x = x[0]
    return np.stack([x[p].cpu().numpy().astype(np.float32) for p in positions], 0)


# ── quant toggles ──────────────────────────────────────────────────────────

_ORIG_ACT_QUANT = kernel_shim.act_quant
_ORIG_FP8_GEMM = kernel_shim.fp8_gemm
_ORIG_LINEAR = None  # filled after model import


def _act_quant_exact(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """No-op: leave activations unchanged (no fp8 round-trip)."""
    if inplace:
        return x
    # non-inplace returns (q, s); callers of non-inplace rare in attn path
    N = x.size(-1)
    groups = N // block_size
    # dummy scales of ones so any fp8_gemm path dequants as identity * weight_scale
    s = torch.ones(*x.shape[:-1], groups, dtype=scale_dtype, device=x.device)
    return x.to(torch.float8_e4m3fn) if False else (x, s)  # never take this branch for inplace


def install_exact_mode():
    """Bypass act_quant inplace and force Linear through bf16 matmul.

    Strategy:
      1. act_quant(..., inplace=True) becomes no-op.
      2. Monkeypatch model.linear to dequant FP8/FP4 weights to bf16 once and
         matmul in bf16/f32 without act_quant on the activation side.
    """
    def act_quant_exact(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if inplace:
            # leave x untouched (still bf16)
            return x
        # non-inplace: return fake fp8 that is just cast (lossy!) — avoid by
        # also patching linear. Provide high-precision path via linear patch.
        N = x.size(-1)
        groups = N // block_size
        y = x.to(torch.float8_e4m3fn)  # still quantizes — linear patch is the real bypass
        s = torch.ones(*x.shape[:-1], groups, device=x.device, dtype=torch.float32)
        if scale_dtype == torch.float8_e8m0fnu:
            # encode 2^0 = 1.0 as exp 127
            s = torch.full((*x.shape[:-1], groups), 127, device=x.device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return y, s

    def linear_exact(x, weight, bias=None):
        """Dequant weight to f32 and matmul without act_quant on x."""
        xf = x.float()
        if weight.element_size() > 1 and weight.dtype not in (torch.float8_e4m3fn, getattr(torch, "float4_e2m1fn_x2", torch.uint8)):
            # bf16/f32 weight
            y = xf @ weight.float().t()
        elif weight.dtype == torch.float8_e4m3fn:
            # dequant weight with its scale
            w = weight.float()
            s = weight.scale
            # scale is ue8m0 [out_blocks, in_blocks]
            from kernel_shim import _ue8m0_to_f32
            group = 128
            N, K = w.shape
            Bs = _ue8m0_to_f32(s)
            if Bs.dim() == 2 and Bs.size(0) == (N + group - 1) // group:
                Bs_nk = Bs.repeat_interleave(group, dim=0)[:N].repeat_interleave(group, dim=-1)
            else:
                Bs_nk = Bs.reshape(N, K // group).repeat_interleave(group, dim=-1)
            w_deq = w * Bs_nk
            y = xf @ w_deq.t()
        else:
            # fp4 path via kernel_shim unpack
            from kernel_shim import _unpack_fp4, _ue8m0_to_f32
            N = weight.size(0)
            K = weight.size(1) * 2
            B = _unpack_fp4(weight).reshape(N, K).to(device=x.device)
            Bs = _ue8m0_to_f32(weight.scale).reshape(N, K // 32)
            B_deq = B * Bs.repeat_interleave(32, dim=-1)
            y = xf @ B_deq.t()
        if bias is not None:
            y = y + bias.float()
        return y.to(x.dtype)

    kernel_shim.act_quant = act_quant_exact
    model_mod.act_quant = act_quant_exact
    # model.py does `from kernel import act_quant` and also uses local linear()
    import model as M
    M.act_quant = act_quant_exact
    # patch linear function used by Linear.forward
    global _ORIG_LINEAR
    if _ORIG_LINEAR is None:
        _ORIG_LINEAR = M.linear
    M.linear = linear_exact
    log("EXACT mode installed (act_quant no-op inplace; linear dequant-matmul f32)")


def install_fp8_mode():
    kernel_shim.act_quant = _ORIG_ACT_QUANT
    model_mod.act_quant = _ORIG_ACT_QUANT
    import model as M
    M.act_quant = _ORIG_ACT_QUANT
    if _ORIG_LINEAR is not None:
        M.linear = _ORIG_LINEAR
    log("FP8 mode restored (kernel_shim defaults)")


@torch.inference_mode()
def run_l0_attn(block, h_stream, start_pos=0):
    """Run only the attention half of Block 0; return attn_out (post-wo)."""
    # h_stream is [B,S,D] after hc_pre + attn_norm (the Attention input)
    return block.attn(h_stream, start_pos)


@torch.inference_mode()
def prepare_attn_input(block, h_hc, tokens):
    """From HC residual [B,S,hc,D] produce Attention input stream [B,S,D]."""
    # mirror Block.forward attn half up to attn call
    residual = h_hc
    y, post, comb = block.hc_pre(h_hc, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    y = block.attn_norm(y)
    return y, residual, post, comb


@torch.inference_mode()
def run(args):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log(f"torch={torch.__version__} device={device}")
    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    positions = [int(x) for x in args.positions.split(",") if int(x) < args.seq]
    log(f"seq={args.seq} positions={positions} layer=0 (ratio={margs.compress_ratios[0]})")

    full_n = margs.n_layers
    margs.n_layers = 0
    margs.n_mtp_layers = 0
    margs.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        shell = Transformer(margs)
    margs.n_layers = full_n
    load_embed_only(shell, model_dir, device)
    shell.eval()
    tokens = load_tokens(args.tokens, args.seq).to(device)
    h = shell.embed(tokens).unsqueeze(2).repeat(1, 1, shell.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 global_L2={float(h.float().norm()):.6f}")
    del shell.head, shell.norm, shell.hc_head_fn, shell.hc_head_base, shell.hc_head_scale
    free_cuda()

    with torch.device("cpu"):
        block = Block(0, margs)
    load_block_weights(block, model_dir, 0, device, verbose=True)
    block.eval()
    assert int(block.attn.compress_ratio) == 0, "L0 must be ratio-0 pure SWA"

    # Prepare attn input once under FP8 mode (hc_pre uses f32 GEMM mostly)
    install_fp8_mode()
    if device.type == "cuda":
        torch.set_default_device("cuda")
    try:
        attn_in, residual, post, comb = prepare_attn_input(block, h, tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        attn_in = attn_in.detach().clone()
        log(f"attn_in L2={float(attn_in.float().norm()):.6f} shape={tuple(attn_in.shape)}")

        # --- Run A: FP8 simulation (default) ---
        # reset kv cache
        block.attn.kv_cache.zero_()
        t0 = time.time()
        out_fp8 = run_l0_attn(block, attn_in, 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_fp8 = time.time() - t0
        arr_fp8 = capture_pos(out_fp8, positions)
        log(f"FP8 attn_out dump_l2={np.linalg.norm(arr_fp8.astype(np.float64)):.4f} wall={dt_fp8:.2f}s")

        # --- Run B: exact f32 path ---
        install_exact_mode()
        block.attn.kv_cache.zero_()
        t0 = time.time()
        out_f32 = run_l0_attn(block, attn_in, 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_f32 = time.time() - t0
        arr_f32 = capture_pos(out_f32, positions)
        log(f"F32 attn_out dump_l2={np.linalg.norm(arr_f32.astype(np.float64)):.4f} wall={dt_f32:.2f}s")

        # --- Also: single Linear floor wq_a ---
        install_fp8_mode()
        wq_a = block.attn.wq_a
        x_pos = attn_in[0, positions, :].float()  # [P, dim]
        # fp8 path via module
        y_fp8 = wq_a(attn_in[0, positions, :].to(attn_in.dtype)).float()
        install_exact_mode()
        y_f32 = wq_a(attn_in[0, positions, :].to(attn_in.dtype)).float()
        install_fp8_mode()

    finally:
        torch.set_default_device("cpu")

    rows = []
    log(f"\n{'pos':>6} {'cos_fp8_vs_f32':>14} {'rel_l2':>10} {'norm_ratio':>12}")
    for i, pos in enumerate(positions):
        cos, rel, nr = metrics(arr_fp8[i], arr_f32[i])
        rows.append({"stage": "attn_out", "pos": pos, "cosine": cos, "rel_l2": rel, "norm_ratio": nr})
        log(f"{pos:6d} {cos:14.8f} {rel:10.6f} {nr:12.6f}")
    cos_all, rel_all, nr_all = metrics(arr_fp8, arr_f32)
    rows.append({"stage": "attn_out", "pos": "ALL", "cosine": cos_all, "rel_l2": rel_all, "norm_ratio": nr_all})
    log(f"{'ALL':>6} {cos_all:14.8f} {rel_all:10.6f} {nr_all:12.6f}")

    # Linear wq_a floor
    y8 = y_fp8.cpu().numpy().astype(np.float64)
    y3 = y_f32.cpu().numpy().astype(np.float64)
    cos_l, rel_l, nr_l = metrics(y8, y3)
    log(f"\nwq_a Linear floor (fp8 vs exact dequant matmul) ALL-pos: cos={cos_l:.8f} rel={rel_l:.6e} nr={nr_l:.6f}")

    # Compare to parent-vs-ref measured attn_out
    parent_ref_cos = 0.999302  # L0 attn_out ALL from residual_stage_content_compare
    parent_ref_rel = 0.037424
    log("\n=== VERDICT ===")
    log(f"quant floor (ref-fp8 vs ref-f32) attn_out ALL: cos={cos_all:.8f} rel={rel_all:.6e}")
    log(f"parent vs ref                        attn_out ALL: cos={parent_ref_cos:.8f} rel={parent_ref_rel:.6e}")
    if cos_all <= parent_ref_cos + 1e-6 and rel_all >= parent_ref_rel * 0.5:
        # floor is at or worse than observed → observed is floor
        verdict = "AT_FLOOR"
        msg = (
            f"Parent-vs-ref L0 attn_out cos={parent_ref_cos:.6f} does NOT clear the "
            f"ref-fp8-vs-ref-f32 floor cos={cos_all:.6f}. Attention is at quantization "
            f"noise; defect is elsewhere. Do NOT bisect L0 attn internals."
        )
    elif cos_all >= 0.99999 or rel_all < parent_ref_rel * 0.25:
        verdict = "CLEARS_FLOOR"
        msg = (
            f"Quant floor cos={cos_all:.8f} rel={rel_all:.3e} is well above parent-vs-ref "
            f"cos={parent_ref_cos:.6f} rel={parent_ref_rel:.3e} (~{parent_ref_rel/max(rel_all,1e-30):.1f}x floor). "
            f"L0 attn_out is a REAL direction defect. Proceed to internals bisect."
        )
    else:
        verdict = "MARGINAL"
        msg = (
            f"Floor cos={cos_all:.8f} rel={rel_all:.3e}; parent-vs-ref cos={parent_ref_cos:.6f} "
            f"rel={parent_ref_rel:.3e} (ratio rel/floor={parent_ref_rel/max(rel_all,1e-30):.2f}). "
            f"Borderline — report both and use per-substage floors for bisect."
        )
    log(msg)

    out = {
        "layer": 0,
        "compress_ratio": 0,
        "seq": args.seq,
        "positions": positions,
        "domain": {
            "fp8": "kernel_shim act_quant + Linear fp8_gemm (default harness)",
            "f32": "act_quant inplace no-op; Linear dequants weights to f32 and matmuls without act quant",
            "note": "same attn_in, same weights, same sparse_attn; only quant path differs",
        },
        "attn_out_fp8_vs_f32": rows,
        "attn_out_all": {"cosine": cos_all, "rel_l2": rel_all, "norm_ratio": nr_all},
        "wq_a_linear_floor": {"cosine": cos_l, "rel_l2": rel_l, "norm_ratio": nr_l},
        "parent_vs_ref_l0_attn_out_all": {"cosine": parent_ref_cos, "rel_l2": parent_ref_rel},
        "verdict": verdict,
        "verdict_msg": msg,
        "wall_s": {"fp8": dt_fp8, "f32": dt_f32},
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attn_out_quant_floor.json"
    path.write_text(json.dumps(out, indent=2))
    log(f"wrote {path}")
    # also dump tensors
    np.savez_compressed(out_dir / "attn_out_quant_floor.npz", fp8=arr_fp8, f32=arr_f32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--positions", default=",".join(str(x) for x in DEFAULT_POSITIONS))
    ap.add_argument("--out-dir", default="/tmp/attn_out_quant_floor")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
