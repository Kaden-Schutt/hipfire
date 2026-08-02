#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Independent DeepSeek-V4 reference oracle.

Runs the bundled `model.py` **verbatim** under PyTorch, with `kernel_shim.py`
supplying eager replacements for every tilelang entry point. Three gates:

  1. Floor — Linear (fp8 dequant + GEMM) self-consistency / codec agreement
  2. Layer-0 Block.forward + hc_post contraction (fix vs deliberate transpose)
  3. Multi-layer residual L2 trajectory (layers 0..L)

Always prefers CPU unless --device cuda is given. CombFixVerify owns the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: shim kernel + hadamard before model import
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# Resolve research inference tree: local symlink layout OR adjacent model.py
def _find_ref_infer() -> Path:
    # 1) model.py already next to us (synced harness or symlink)
    if (HERE / "model.py").is_file():
        return HERE
    # 2) walk up looking for .codeinsight+research
    for parent in HERE.parents:
        cand = parent / ".codeinsight+research" / "ds4-parent-ref" / "inference"
        if (cand / "model.py").is_file():
            return cand
    return HERE
REF_INFER = _find_ref_infer()
REPO = REF_INFER.parents[2] if REF_INFER != HERE else HERE

sys.path.insert(0, str(HERE))  # fast_hadamard_transform, weight_loader, parent_hc_post_ref

# Install kernel shim as `kernel` BEFORE importing model.
import kernel_shim as _kernel_shim  # noqa: E402
sys.modules["kernel"] = _kernel_shim

# model.py lives next to us as a symlink/copy OR we import from REF_INFER
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
    Linear,
    linear as model_linear,
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
from kernel_shim import act_quant, fp8_gemm, hc_split_sinkhorn  # noqa: E402


DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens_128.bin"
)
# Pre-fix parent residual L2 (rows=1024, from Gate-6). For narrative only.
PRE_FIX_PARENT_L2 = [
    494.179871, 474.714539, 483.457733, 482.975098, 486.401825, 777.972900, 1188.696289,
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_tokens(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    # int32 little-endian
    assert len(raw) >= n * 4
    ids = struct.unpack("<" + "i" * n, raw[: n * 4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64 = a.detach().float().reshape(-1).double()
    b64 = b.detach().float().reshape(-1).double()
    diff = (a64 - b64).abs()
    denom = b64.norm().clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(a64.unsqueeze(0), b64.unsqueeze(0)).item()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_fro": float(diff.norm() / denom),
        "cosine": float(cos),
        "l2_a": float(a64.norm()),
        "l2_b": float(b64.norm()),
    }


def make_args(config_path: Path, max_seq_len: int = 256) -> ModelArgs:
    with open(config_path) as f:
        cfg = json.load(f)
    # inference/config.json keys match ModelArgs fields
    args = ModelArgs(**{k: v for k, v in cfg.items() if k in ModelArgs.__dataclass_fields__})
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    # ModelArgs.scale_dtype default "fp8" → ue8m0 scales; keep it.
    return args


# ===========================================================================
# Step 1 — floor
# ===========================================================================

def step1_floor(model_dir: Path, device: str) -> dict:
    """Establish Linear agreement floor when both sides are believed correct.

    Side A: model.py Linear path (act_quant + fp8_gemm via kernel_shim).
    Side B: explicit dequant of the same weight + f32 matmul (codec-style).

    Both use the same kernel_shim dequant math, so this is the *arithmetic*
    floor of f32 accumulation over K=4096 with BF16 I/O — not a cross-impl
    check. Cross-impl (vs Rust codec) is reported if parent dump is present.
    """
    log("=== STEP 1: floor (Linear fp8) ===")
    index = build_tensor_index(model_dir)
    w_name = "layers.0.attn.wq_a.weight"
    s_name = "layers.0.attn.wq_a.scale"
    w = _load_tensor(index, w_name)
    s = _load_tensor(index, s_name)
    assert w.dtype == torch.float8_e4m3fn
    out_f, in_f = w.shape  # [1024, 4096]

    torch.manual_seed(0)
    x = torch.randn(8, in_f, dtype=torch.bfloat16)

    # Force model globals used by linear()
    model_mod.block_size = 128
    model_mod.scale_fmt = "ue8m0"
    model_mod.scale_dtype = torch.float8_e8m0fnu
    model_mod.default_dtype = torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)

    # Attach scale the way Linear does
    w.scale = s if s.dtype == torch.float8_e8m0fnu else s.view(torch.float8_e8m0fnu)

    # Side A: model linear dispatch
    y_a = model_linear(x, w, None).float()

    # Side B: dequant weight to f32, act to f32 (with same act_quant dequant),
    # then plain matmul — still through shim act_quant for the activation path
    x_q, x_s = act_quant(x.float(), 128, "ue8m0", torch.float8_e8m0fnu, inplace=False)
    # reconstruct act in f32 from codes
    # act_quant returned fp8 codes + scales; dequant:
    xs = _ue8m0_to_f32(x_s).reshape(-1, in_f // 128)
    xa = x_q.to(torch.float32).reshape(-1, in_f)
    xa = xa * xs.repeat_interleave(128, dim=-1)
    wd = dequant_fp8_block128(w, w.scale)
    y_b = (xa @ wd.t()).reshape(y_a.shape)

    m = metrics(y_a, y_b)
    log(f"  weight: {w_name} {tuple(w.shape)} scale {tuple(s.shape)}")
    log(f"  domain: bf16 activations, fp8 weights, f32 accumulate, compare in f32")
    log(f"  max_abs={m['max_abs']:.6e}  mean_abs={m['mean_abs']:.6e}  rel_fro={m['rel_fro']:.6e}  cosine={m['cosine']:.8f}")
    log(f"  FLOOR max_abs = {m['max_abs']:.6e}  (use as lower bound for later PASS/FAIL)")

    # Self-consistency of two dequants of the same weight (codec identity)
    wd2 = dequant_fp8_block128(w, s)
    md = metrics(wd, wd2)
    log(f"  dequant self-check max_abs={md['max_abs']:.6e} (expect 0)")

    # Known-correct synthetic: identical bf16 GEMM both paths without quant
    w_bf = wd.to(torch.bfloat16)
    x_bf = x
    y1 = F.linear(x_bf.float(), w_bf.float())
    y2 = x_bf.float() @ w_bf.float().t()
    ms = metrics(y1, y2)
    log(f"  plain f32 matmul identity max_abs={ms['max_abs']:.6e}")

    return {"floor_max_abs": m["max_abs"], "floor_rel_fro": m["rel_fro"], "metrics": m}


# ===========================================================================
# Step 2 — layer 0 + hc_post
# ===========================================================================

def build_partial_model(args: ModelArgs, model_dir: Path, n_layers: int, device: str) -> Transformer:
    """Construct Transformer but only materialize first n_layers blocks' weights."""
    # Temporarily shrink n_layers so we don't allocate 43 MoE expert ModuleLists
    # of empty Parameters (still heavy). We keep real n_layers in config for
    # compress_ratios indexing by building only n_layers blocks.
    full_layers = args.n_layers
    args.n_layers = n_layers
    # Also drop mtp to save RAM
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    # Construct on CPU
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full_layers  # restore for bookkeeping
    log(f"  constructed Transformer with {n_layers} blocks (mtp disabled)")
    load_state_into_model(
        model, model_dir, layers=range(n_layers), load_experts=True, device=device, verbose=True
    )
    model.eval()
    return model


@torch.inference_mode()
def step2_layer0(model: Transformer, tokens: torch.Tensor, floor: float, device: str) -> dict:
    log("=== STEP 2: layer-0 Block.forward + hc_post ===")
    tokens = tokens.to(device)
    # embed + HC expand (Transformer.forward head)
    h = model.embed(tokens)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    log(f"  input tokens shape={tuple(tokens.shape)} h={tuple(h.shape)} dtype={h.dtype}")

    t0 = time.time()
    h1 = model.layers[0](h, 0, tokens)
    dt = time.time() - t0
    log(f"  layer0 forward done in {dt:.1f}s  out_l2={h1.float().norm().item():.6f}")

    # residual L2 after layer 0 (multi-stream)
    res_l2 = float(h1.float().norm())
    log(f"  residual L2 after L0 (all streams) = {res_l2:.6f}")

    # ---- hc_post direct check ----
    log("--- hc_post contraction check ---")
    torch.manual_seed(1)
    rows, hc, dim = 16, 4, 4096
    # Use real post/comb from a short hc_pre on random residual, for realism
    block: Block = model.layers[0]
    x_hc = torch.randn(1, rows, hc, dim, dtype=torch.bfloat16, device=device)
    # hc_pre → get post, comb
    y_stream, post, comb = block.hc_pre(x_hc, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    # synthetic attention-out
    x_attn = torch.randn_like(y_stream)
    residual = x_hc

    y_model = block.hc_post(x_attn, residual, post, comb)
    y_parent = parent_hc_post_explicit(x_attn, residual, post, comb, transpose_comb=False)
    y_bug = parent_hc_post_explicit(x_attn, residual, post, comb, transpose_comb=True)

    m_ok = metrics(y_model, y_parent)
    m_bug = metrics(y_model, y_bug)
    log(f"  hc_post FIXED  max_abs={m_ok['max_abs']:.6e} rel_fro={m_ok['rel_fro']:.6e} cosine={m_ok['cosine']:.8f}")
    log(f"  hc_post TRANSPOSED-comb max_abs={m_bug['max_abs']:.6e} rel_fro={m_bug['rel_fro']:.6e} cosine={m_bug['cosine']:.8f}")

    # Verdicts against floor
    # hc_post is pure f32 arithmetic on identical inputs → floor should be ~0
    # (exact). Allow tiny f32 noise.
    ok_pass = m_ok["max_abs"] < 1e-5
    bug_fail = m_bug["max_abs"] > 1e-2  # must be loud
    log(f"  VERDICT fixed:  {'PASS' if ok_pass else 'FAIL'} (threshold 1e-5, floor was {floor:.3e})")
    log(f"  VERDICT transpose-detect: {'PASS (loud fail)' if bug_fail else 'FAIL (did not detect bug)'}")

    # Also compare model hc_post against the exact model.py one-liner inline
    y_inline = (
        post.unsqueeze(-1) * x_attn.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    ).type_as(x_attn)
    m_inline = metrics(y_model, y_inline)
    log(f"  model.hc_post vs inline one-liner max_abs={m_inline['max_abs']:.6e} (expect 0)")

    return {
        "layer0_residual_l2": res_l2,
        "h1": h1.detach().cpu(),
        "hc_post_fixed": m_ok,
        "hc_post_transposed": m_bug,
        "hc_post_fixed_pass": ok_pass,
        "hc_post_transpose_detected": bug_fail,
    }


# ===========================================================================
# Step 3 — multi-layer trajectory
# ===========================================================================

@torch.inference_mode()
def step3_trajectory(model: Transformer, tokens: torch.Tensor, n_layers: int, device: str) -> list:
    log(f"=== STEP 3: residual L2 trajectory layers 0..{n_layers-1} ===")
    tokens = tokens.to(device)
    h = model.embed(tokens)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    traj = []
    for i in range(n_layers):
        t0 = time.time()
        h = model.layers[i](h, 0, tokens)
        l2 = float(h.float().norm())
        traj.append(l2)
        dt = time.time() - t0
        pref = PRE_FIX_PARENT_L2[i] if i < len(PRE_FIX_PARENT_L2) else float("nan")
        ratio = l2 / traj[i - 1] if i > 0 else float("nan")
        log(f"  L{i}: residual_L2={l2:.6f}  (pre-fix parent={pref:.6f})  ratio_prev={ratio:.6f}  ({dt:.1f}s)")
    log("trajectory: " + ", ".join(f"{v:.6f}" for v in traj))
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--config", default=None, help="inference config.json (default: model/inference/config.json or research copy)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--layers", type=int, default=7, help="how many layers to run for step 3 (default 7 = 0..6)")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--step", choices=["all", "1", "2", "3"], default="all")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    # Force CPU threads; never silently grab GPU when device=cpu
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        # ROCm torch still sees GPU; pin ops to cpu explicitly
        torch.set_default_device("cpu")

    model_dir = Path(args.model)
    if args.config:
        config_path = Path(args.config)
    elif (model_dir / "inference" / "config.json").exists():
        config_path = model_dir / "inference" / "config.json"
    else:
        config_path = REF_INFER / "config.json"

    log(f"model_dir={model_dir}")
    log(f"config={config_path}")
    log(f"device={args.device}  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    log(f"tokens={args.tokens}  seq={args.seq}  layers={args.layers}")

    summary = {"torch": torch.__version__, "device": args.device}

    # Step 1 always first when all/1
    if args.step in ("all", "1"):
        summary["step1"] = step1_floor(model_dir, args.device)
        floor = summary["step1"]["floor_max_abs"]
    else:
        floor = float("nan")

    if args.step in ("all", "2", "3"):
        margs = make_args(config_path, max_seq_len=max(args.seq, 256))
        log(f"ModelArgs: n_layers={margs.n_layers} dim={margs.dim} route_scale={margs.route_scale} "
            f"score_func={margs.score_func} expert_dtype={margs.expert_dtype} scale_fmt={margs.scale_fmt}")
        n_build = args.layers if args.step != "2" else max(1, args.layers)
        if args.step == "2":
            n_build = 1
        model = build_partial_model(margs, model_dir, n_build, args.device)
        tokens = load_tokens(Path(args.tokens), args.seq)
        log(f"tokens[0,:8]={tokens[0,:8].tolist()}")

        if args.step in ("all", "2"):
            summary["step2"] = step2_layer0(model, tokens, floor, args.device)
            # drop bulky tensor from summary json
            h1 = summary["step2"].pop("h1")
            # save for parent compare
            out_h = HERE / "layer0_hidden.pt"
            torch.save({"h": h1, "tokens": tokens.cpu()}, out_h)
            log(f"  wrote {out_h}")

        if args.step in ("all", "3"):
            if args.step == "3" or n_build < args.layers:
                # need rebuild if we only built 1 layer
                if len(model.layers) < args.layers:
                    model = build_partial_model(margs, model_dir, args.layers, args.device)
            summary["step3_trajectory"] = step3_trajectory(model, tokens, args.layers, args.device)
            summary["pre_fix_parent"] = PRE_FIX_PARENT_L2[: args.layers]

    log("=== SUMMARY ===")
    # compact print
    def fmt(o):
        if isinstance(o, float):
            return round(o, 8)
        if isinstance(o, dict):
            return {k: fmt(v) for k, v in o.items() if k != "h1"}
        if isinstance(o, list):
            return [fmt(x) for x in o]
        return o
    print(json.dumps(fmt(summary), indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(fmt(summary), indent=2))
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
