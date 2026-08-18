#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Reference end-to-end PPL on tokens.bin (1024) — fp8 shim vs exact f32.

Streams all 43 Blocks (~4.7 GiB peak), then hc_head + norm + head with
full_logits=True. Scores row t against token_ids[t+1] (parent::plog::compare
convention). Optionally writes HFPLOG01 .plog files.

This is the measurement that decides whether parent PPL~59 is a port bug or
the reference recipe itself.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks

import torch
import torch.nn.functional as F
import numpy as np

from residual_content_dump import (
    load_tokens, make_args, free_cuda, load_embed_only, load_block_weights,
    DEFAULT_MODEL, DEFAULT_TOKENS,
)
from weight_loader import build_tensor_index, _load_tensor
from model import Block, Transformer, RMSNorm
import model as model_mod
import kernel_shim

PLOG_MAGIC = b"HFPLOG01"
VOCAB = 129280

_ORIG_ACT_QUANT = kernel_shim.act_quant
_ORIG_FP4_ACT = kernel_shim.fp4_act_quant
_ORIG_LINEAR = None


def log(m=""):
    print(m, flush=True)


def install_fp8_mode():
    global _ORIG_LINEAR
    kernel_shim.act_quant = _ORIG_ACT_QUANT
    kernel_shim.fp4_act_quant = _ORIG_FP4_ACT
    model_mod.act_quant = _ORIG_ACT_QUANT
    if hasattr(model_mod, "fp4_act_quant"):
        model_mod.fp4_act_quant = _ORIG_FP4_ACT
    import model as M
    M.act_quant = _ORIG_ACT_QUANT
    if _ORIG_LINEAR is not None:
        M.linear = _ORIG_LINEAR
    # also restore kernel imports used inside model.linear closure
    M.fp8_gemm = kernel_shim.fp8_gemm
    M.fp4_gemm = kernel_shim.fp4_gemm
    log("MODE=fp8 (kernel_shim defaults)")


def install_exact_mode():
    """No act quant; dequant weights to f32 and matmul."""
    global _ORIG_LINEAR

    def act_quant_exact(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if inplace:
            return x
        N = x.size(-1)
        groups = N // block_size
        # unused if linear is patched; keep shape-compatible
        y = x  # keep full precision; linear_exact ignores quant dtype
        s = torch.ones(*x.shape[:-1], groups, device=x.device, dtype=torch.float32)
        return y, s

    def fp4_act_quant_exact(x, block_size=32, inplace=False):
        if inplace:
            return x
        N = x.size(-1)
        groups = N // block_size
        s = torch.ones(*x.shape[:-1], groups, device=x.device, dtype=torch.float32)
        return x, s

    def linear_exact(x, weight, bias=None):
        assert bias is None
        xf = x.float()
        from kernel_shim import _ue8m0_to_f32, _unpack_fp4
        if weight.dtype == torch.float8_e4m3fn:
            w = weight.float()
            s = weight.scale
            group = 128
            N, K = w.shape
            Bs = _ue8m0_to_f32(s)
            if Bs.dim() == 2 and Bs.size(0) == (N + group - 1) // group:
                Bs_nk = Bs.repeat_interleave(group, dim=0)[:N].repeat_interleave(group, dim=-1)
            else:
                Bs_nk = Bs.reshape(N, K // group).repeat_interleave(group, dim=-1)
            w_deq = w * Bs_nk
            y = xf @ w_deq.t()
        elif str(weight.dtype).endswith("float4_e2m1fn_x2") or weight.dtype == getattr(torch, "float4_e2m1fn_x2", type(None)):
            N = weight.size(0)
            K = weight.size(1) * 2
            B = _unpack_fp4(weight).reshape(N, K).to(device=x.device)
            Bs = _ue8m0_to_f32(weight.scale).reshape(N, K // 32)
            B_deq = B * Bs.repeat_interleave(32, dim=-1)
            y = xf @ B_deq.t()
        else:
            y = xf @ weight.float().t()
        return y.to(dtype=torch.bfloat16 if x.dtype == torch.bfloat16 else x.dtype)

    import model as M
    if _ORIG_LINEAR is None:
        _ORIG_LINEAR = M.linear
    kernel_shim.act_quant = act_quant_exact
    kernel_shim.fp4_act_quant = fp4_act_quant_exact
    model_mod.act_quant = act_quant_exact
    M.act_quant = act_quant_exact
    M.linear = linear_exact
    log("MODE=exact (act_quant no-op; linear dequant-f32 matmul)")


def load_head_and_hc(shell: Transformer, model_dir: Path, device):
    """Load norm, head.weight, hc_head_* onto shell (already has embed)."""
    index = build_tensor_index(Path(model_dir))
    mapping = {
        "norm.weight": shell.norm.weight,
        "head.weight": shell.head.weight,
        "hc_head_fn": shell.hc_head_fn,
        "hc_head_base": shell.hc_head_base,
        "hc_head_scale": shell.hc_head_scale,
    }
    for name, param in mapping.items():
        t = _load_tensor(index, name, device="cpu")
        if t.dtype != param.dtype:
            t = t.to(param.dtype)
        if tuple(t.shape) != tuple(param.shape):
            raise RuntimeError(f"{name}: ckpt {tuple(t.shape)} vs model {tuple(param.shape)}")
        with torch.no_grad():
            param.copy_(t)
        log(f"  loaded {name} {tuple(param.shape)} {param.dtype}")
    shell.norm.to(device)
    shell.head.to(device)
    shell.hc_head_fn.data = shell.hc_head_fn.data.to(device)
    shell.hc_head_base.data = shell.hc_head_base.data.to(device)
    shell.hc_head_scale.data = shell.hc_head_scale.data.to(device)


def write_plog_header(f, n_tokens: int, vocab: int):
    f.write(PLOG_MAGIC)
    f.write(struct.pack("<I", n_tokens))
    f.write(struct.pack("<I", vocab))
    f.write(struct.pack("<Q", 0))


def ppl_from_logits(logits_f32: torch.Tensor, token_ids: torch.Tensor) -> dict:
    """logits: [S, V] f32; token_ids: [S] long. Score row t vs token t+1."""
    # Host-side f64 NLL — avoids device mismatch and keeps GPU free for plog path.
    if logits_f32.device.type != "cpu":
        logits_f32 = logits_f32.detach().cpu()
    token_ids = token_ids.detach().cpu().long().view(-1)
    S, V = logits_f32.shape
    assert token_ids.numel() == S
    x = logits_f32.double()
    m = x.max(dim=-1, keepdim=True).values
    logZ = torch.log(torch.exp(x - m).sum(dim=-1)) + m.squeeze(-1)
    logp = x - logZ.unsqueeze(-1)
    targets = token_ids[1:]
    nll = -logp[:-1].gather(1, targets.view(-1, 1)).squeeze(1)
    mean_nll = float(nll.mean().item())
    ppl = math.exp(mean_nll)
    pred = logits_f32[:-1].argmax(dim=-1)
    top1 = float((pred == targets).double().mean().item())
    return {
        "n_scored": int(nll.numel()),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "top1": top1,
        "nll_sum": float(nll.sum().item()),
    }


@torch.inference_mode()
def run_one(mode: str, args, device) -> dict:
    if mode == "fp8":
        install_fp8_mode()
    elif mode == "exact":
        install_exact_mode()
    else:
        raise ValueError(mode)

    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    n_layers = margs.n_layers
    assert margs.vocab_size == VOCAB or True
    vocab = int(margs.vocab_size)
    log(f"=== run mode={mode} seq={args.seq} n_layers={n_layers} vocab={vocab} ===")

    # Build shell with 0 layers; stream blocks
    full_n = margs.n_layers
    margs.n_layers = 0
    margs.n_mtp_layers = 0
    margs.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        shell = Transformer(margs)
    margs.n_layers = full_n

    load_embed_only(shell, model_dir, device)
    load_head_and_hc(shell, model_dir, device)
    shell.eval()

    tokens = load_tokens(args.tokens, args.seq).to(device)
    token_ids_1d = tokens.view(-1).cpu()
    # sha check optional
    h = shell.embed(tokens).unsqueeze(2).repeat(1, 1, shell.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 L2={float(h.float().norm()):.4f}")

    t0 = time.time()
    last_block = None
    for i in range(n_layers):
        with torch.device("cpu"):
            block = Block(i, margs)
        load_block_weights(block, model_dir, i, device, verbose=(i < 2 or i == n_layers - 1 or i % 10 == 0))
        block.eval()
        if device.type == "cuda":
            torch.set_default_device("cuda")
        try:
            h = block(h, 0, tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            torch.set_default_device("cpu")
        log(f"L{i:02d} L2={float(h.float().norm()):.4f}")
        last_block = block
        # free previous block weights (keep last for hc_head method)
        if i < n_layers - 1:
            del block
            free_cuda()

    # hc_head uses last layer\'s method + shell params
    assert last_block is not None
    h = last_block.hc_head(h, shell.hc_head_fn, shell.hc_head_scale, shell.hc_head_base)
    h = shell.norm(h)
    # full sequence logits: [1, S, V] or [S, V]
    logits = shell.head(h, full_logits=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    log(f"logits shape={tuple(logits.shape)} dtype={logits.dtype} wall_layers+head={wall:.1f}s")

    # squeeze batch
    if logits.dim() == 3:
        logits = logits[0]
    logits_f32 = logits.float().contiguous()
    S, V = logits_f32.shape
    assert S == args.seq, (S, args.seq)
    assert V == vocab, (V, vocab)

    stats = ppl_from_logits(logits_f32, token_ids_1d)
    log(f"PPL[{mode}] = {stats['ppl']:.6f}  mean_nll={stats['mean_nll']:.6f}  top1={stats['top1']:.4f}  n_scored={stats['n_scored']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plog_path = out_dir / f"ref_{mode}_{args.seq}.plog"
    if args.write_plog:
        log(f"writing plog {plog_path} ({S}×{V} f32 ≈ {S*V*4/1e9:.2f} GiB)...")
        tw = time.time()
        with open(plog_path, "wb") as f:
            write_plog_header(f, S, V)
            # stream rows to avoid huge intermediate
            row_bytes = V * 4
            cpu = logits_f32.cpu().numpy().astype("<f4", copy=False)
            for t in range(S):
                f.write(cpu[t].tobytes())
        log(f"plog wrote in {time.time()-tw:.1f}s size={plog_path.stat().st_size}")

    # free big tensors before next mode
    del logits, logits_f32, h, last_block, shell, tokens
    free_cuda()

    return {
        "mode": mode,
        "seq": args.seq,
        "vocab": vocab,
        "n_layers": n_layers,
        "wall_s": wall,
        "ppl": stats["ppl"],
        "mean_nll": stats["mean_nll"],
        "top1": stats["top1"],
        "n_scored": stats["n_scored"],
        "plog": str(plog_path) if args.write_plog else None,
        "tokens": args.tokens,
        "model": args.model,
        "references": {
            "parent_ppl_1024": 59.507,
            "mq2r_ppl_1024": 14.703,
            "lloyd_ppl_1024": 14.564,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out-dir", default="/tmp/ref_ppl_1024")
    ap.add_argument("--modes", default="fp8,exact", help="comma list: fp8,exact")
    ap.add_argument("--write-plog", action="store_true", default=True)
    ap.add_argument("--no-plog", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.no_plog:
        args.write_plog = False

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    log(f"torch={torch.__version__} device={device}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    t_all = time.time()
    for mode in modes:
        r = run_one(mode, args, device)
        results.append(r)
        # dump per-mode json immediately
        out_dir = Path(args.out_dir)
        (out_dir / f"ref_ppl_{mode}.json").write_text(json.dumps(r, indent=2))

    summary = {"results": results, "wall_total_s": time.time() - t_all}
    if len(results) == 2:
        by = {r["mode"]: r for r in results}
        if "fp8" in by and "exact" in by:
            summary["delta"] = {
                "ppl_fp8": by["fp8"]["ppl"],
                "ppl_exact": by["exact"]["ppl"],
                "ppl_ratio_fp8_over_exact": by["fp8"]["ppl"] / max(by["exact"]["ppl"], 1e-30),
                "top1_fp8": by["fp8"]["top1"],
                "top1_exact": by["exact"]["top1"],
            }
            log("\n=== SUMMARY ===")
            log(f"ref fp8   PPL={by['fp8']['ppl']:.4f}  top1={by['fp8']['top1']:.4f}")
            log(f"ref exact PPL={by['exact']['ppl']:.4f}  top1={by['exact']['top1']:.4f}")
            log(f"parent    PPL=59.507")
            log(f"mq2r      PPL=14.703")
            log(f"lloyd     PPL=14.564")
            # verdict
            fp8_p, ex_p = by["fp8"]["ppl"], by["exact"]["ppl"]
            if fp8_p > 40 and ex_p < 25:
                v = "FP8_RECIPE_IS_THE_GAP"
                msg = (
                    f"Reference fp8 PPL={fp8_p:.2f} ≈ parent 59.5; exact PPL={ex_p:.2f} ≈ quants. "
                    f"Parent is faithful to the fp8-activation recipe; the recipe loses to MQ2R."
                )
            elif fp8_p < 25 and abs(fp8_p - 59.5) > 20:
                v = "PARENT_STILL_BUGGY"
                msg = (
                    f"Reference fp8 PPL={fp8_p:.2f} is far below parent 59.5 — port still defective."
                )
            else:
                v = "INCONCLUSIVE_OR_MIXED"
                msg = f"fp8 PPL={fp8_p:.2f} exact PPL={ex_p:.2f} parent=59.5 mq2r=14.7 — inspect numbers."
            summary["verdict"] = v
            summary["verdict_msg"] = msg
            log(f"VERDICT: {v}")
            log(msg)

    out_path = Path(args.out_dir) / "ref_ppl_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
