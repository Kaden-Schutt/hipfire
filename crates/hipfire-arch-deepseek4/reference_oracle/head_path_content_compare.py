#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt <kaden@hipfire.dev>
"""Head-path stage cosine: hc_head → final RMSNorm → lm_head.

Feeds the SAME residual into both the torch reference path and the parent
host oracle (`hc_head_ref` / `rms_norm_ref` / BF16-staged head GEMM) so a
defect outside Block.forward cannot hide behind residual drift.

Also reports the end-to-end residual-pair case (parent residual vs ref residual
through the torch head) and a pure-f32 identity floor on the torch path.

Stages (names match model.py):
  residual   [n_pos, hc_mult, dim]   input
  hc_out     [n_pos, dim]            after Block.hc_head (plain sigmoid)
  norm_out   [n_pos, dim]            after Transformer.norm
  logits     [n_pos, vocab]          after ParallelHead (F.linear f32)

Usage (on mi300x):
  PYTHONPATH=/mnt/scratch/torch_oracle_rocm:... \\\n    python3 head_path_content_compare.py \\\n      --ref-npz /tmp/residual_content_ref/residual_content_ref.npz \\\n      --parent-bin /tmp/residual_content_parent/layer_42.f32 \\\n      --out-dir /tmp/head_path_content
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks  # noqa: F401
sys.modules["kernel"] = _ks

import numpy as np
import torch
import torch.nn.functional as F

from residual_content_dump import DEFAULT_MODEL, DEFAULT_TOKENS, make_args
from weight_loader import build_tensor_index, _load_tensor
from model import RMSNorm

DEFAULT_POSITIONS = [0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023]
HC_MULT = 4
DIM = 4096
VOCAB = 129280
NORM_EPS = 1e-6
HC_EPS = 1e-6


def log(m=""):
    print(m, flush=True)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return {"cosine": 1.0, "rel_l2": 0.0, "norm_ratio": 1.0, "norm_a": 0.0, "norm_b": 0.0}
    cos = float(np.dot(a, b) / (na * nb + 1e-300))
    rel = float(np.linalg.norm(a - b) / (nb + 1e-300))
    return {
        "cosine": cos,
        "rel_l2": rel,
        "norm_ratio": float(na / (nb + 1e-300)),
        "norm_a": na,
        "norm_b": nb,
    }


def bf16_round_np(x: np.ndarray) -> np.ndarray:
    """Round f32/f64 array to bf16 via bit truncation (round-to-nearest-even approx via torch)."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    return t.to(torch.bfloat16).float().numpy()


def load_head_weights(model_dir: Path, device):
    index = build_tensor_index(model_dir)
    def get(name, dtype=None):
        t = _load_tensor(index, name, device="cpu")
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    hc_fn = get("hc_head_fn", torch.float32)          # [4, 16384]
    hc_base = get("hc_head_base", torch.float32)      # [4]
    hc_scale = get("hc_head_scale", torch.float32)    # [1]
    norm_w = get("norm.weight", torch.float32)        # [4096]  (ckpt bf16 → f32)
    head_w = get("head.weight", torch.float32)        # [V, 4096]  (ckpt bf16 → f32)
    return {
        "hc_fn": hc_fn,
        "hc_base": hc_base,
        "hc_scale": hc_scale,
        "norm_w": norm_w,
        "head_w": head_w,
    }


@torch.inference_mode()
def torch_hc_head(x_hc, hc_fn, hc_scale, hc_base, norm_eps=NORM_EPS, hc_eps=HC_EPS):
    """model.py Block.hc_head. x_hc: [S,H,D] or [B,S,H,D] → returns [S,D]."""
    squeeze_b = False
    if x_hc.dim() == 3:
        x_hc = x_hc.unsqueeze(0)
        squeeze_b = True
    shape, dtype = x_hc.size(), x_hc.dtype
    x = x_hc.flatten(2).float()  # [B,S,H*D]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_hc.float().view(shape), dim=2)
    y = y.to(dtype)
    if squeeze_b:
        y = y.squeeze(0)
    return y  # [S,D]


@torch.inference_mode()
def torch_rms_norm(x, weight, eps=NORM_EPS):
    # model.py RMSNorm
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


@torch.inference_mode()
def torch_head_proj(x, weight):
    """ParallelHead: F.linear(x.float(), weight_f32)."""
    return F.linear(x.float(), weight.float())


def parent_host_hc_head(x_np, hc_fn, hc_scale, hc_base, eps=NORM_EPS, hc_eps=HC_EPS):
    """Host mirror of hc_head_ref / model.py hc_head. x_np: [S,H,D] f32."""
    S, H, D = x_np.shape
    assert H == HC_MULT and D == DIM
    x_flat = x_np.reshape(S, H * D).astype(np.float64)  # [S, 16384]
    # rsqrt over last dim
    mean_sq = (x_flat * x_flat).mean(axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)
    # mixes = x @ W.T  ; W is [H, H*D]
    W = hc_fn.astype(np.float64)  # [4, 16384]
    mixes = (x_flat @ W.T) * rsqrt  # [S, 4]
    scale = float(np.asarray(hc_scale).reshape(-1)[0])
    base = hc_base.astype(np.float64).reshape(1, H)
    pre = 1.0 / (1.0 + np.exp(-(mixes * scale + base))) + hc_eps  # [S,4]
    y = (pre[:, :, None] * x_np.astype(np.float64)).sum(axis=1)  # [S,D]
    return y.astype(np.float32), pre.astype(np.float32)


def parent_host_rms_norm(x_np, weight, eps=NORM_EPS):
    x = x_np.astype(np.float64)
    var = (x * x).mean(axis=-1, keepdims=True)
    y = x * (1.0 / np.sqrt(var + eps))
    return (y * weight.astype(np.float64).reshape(1, -1)).astype(np.float32)


def parent_host_head_bf16(x_normed, head_w_f32):
    """Parent path: round acts to bf16, weight already bf16-origin f32, f32 accum."""
    x_b = bf16_round_np(x_normed).astype(np.float32)
    # head_w_f32 already widened from bf16 checkpoint
    return x_b @ head_w_f32.T


def parent_host_head_f32(x_normed, head_w_f32):
    """Reference ParallelHead: full f32 acts × f32 weight."""
    return x_normed.astype(np.float32) @ head_w_f32.T


def load_residuals(ref_npz, parent_bin, positions):
    z = np.load(ref_npz)
    ref = np.asarray(z["layer_42"]).astype(np.float32)
    if ref.ndim == 2:
        ref = ref.reshape(len(positions), HC_MULT, DIM)
    assert ref.shape == (len(positions), HC_MULT, DIM), ref.shape
    parent = np.fromfile(parent_bin, dtype="<f4")
    parent = parent.reshape(len(positions), HC_MULT, DIM)
    return ref, parent


def stage_bundle(name, residual_np, w, device):
    """Run torch + parent-host on one residual tensor [S,H,D]."""
    x_t = torch.from_numpy(residual_np).to(device=device, dtype=torch.float32)
    # torch path
    hc_t = torch_hc_head(x_t, w["hc_fn"], w["hc_scale"], w["hc_base"])  # [S,D]
    norm_t = torch_rms_norm(hc_t, w["norm_w"])
    logits_t = torch_head_proj(norm_t, w["head_w"])

    # parent host path (matches head.rs comments)
    hc_fn = w["hc_fn"].detach().float().cpu().numpy()
    hc_base = w["hc_base"].detach().float().cpu().numpy()
    hc_scale = w["hc_scale"].detach().float().cpu().numpy()
    norm_w = w["norm_w"].detach().float().cpu().numpy()
    head_w = w["head_w"].detach().float().cpu().numpy()

    hc_p, pre_p = parent_host_hc_head(residual_np, hc_fn, hc_scale, hc_base)
    norm_p = parent_host_rms_norm(hc_p, norm_w)
    # Parent stages bf16 acts before GEMM
    logits_p = parent_host_head_bf16(norm_p, head_w)
    # Pure-f32 head (ref ParallelHead) on parent-normed — isolates BF16 staging
    logits_p_f32head = parent_host_head_f32(norm_p, head_w)
    # Pure-f32 head on torch-normed
    logits_t_np = logits_t.detach().float().cpu().numpy()
    norm_t_np = norm_t.detach().float().cpu().numpy()
    hc_t_np = hc_t.detach().float().cpu().numpy()
    logits_tf32 = parent_host_head_f32(norm_t_np, head_w)

    return {
        "hc_torch": hc_t_np,
        "norm_torch": norm_t_np,
        "logits_torch": logits_t_np,
        "hc_parent_host": hc_p,
        "norm_parent_host": norm_p,
        "logits_parent_host_bf16act": logits_p,
        "logits_parent_host_f32act": logits_p_f32head,
        "logits_torch_via_f32_matmul": logits_tf32,
        "pre_parent_host": pre_p,
    }


def compare_pair(label, a, b):
    m = metrics(a, b)
    m["label"] = label
    return m


def per_pos_table(a, b, positions, reduce_last=None):
    rows = []
    for i, p in enumerate(positions):
        aa = a[i] if reduce_last is None else a[i]
        bb = b[i] if reduce_last is None else b[i]
        m = metrics(aa, bb)
        m["pos"] = int(p)
        rows.append(m)
    return rows


def summarize(rows):
    cos = [r["cosine"] for r in rows]
    rel = [r["rel_l2"] for r in rows]
    return {
        "mean_cosine": float(np.mean(cos)),
        "min_cosine": float(np.min(cos)),
        "mean_rel_l2": float(np.mean(rel)),
        "max_rel_l2": float(np.max(rel)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ref-npz", required=True)
    ap.add_argument("--parent-bin", required=True)
    ap.add_argument("--positions", default=",".join(str(p) for p in DEFAULT_POSITIONS))
    ap.add_argument("--out-dir", default="/tmp/head_path_content")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    positions = [int(x) for x in args.positions.split(",")]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log(f"device={device} positions={positions}")

    ref_res, parent_res = load_residuals(args.ref_npz, args.parent_bin, positions)
    log(f"ref residual L42 {ref_res.shape} L2={np.linalg.norm(ref_res.astype(np.float64)):.4f}")
    log(f"parent residual L42 {parent_res.shape} L2={np.linalg.norm(parent_res.astype(np.float64)):.4f}")
    m_res = metrics(parent_res, ref_res)
    log(f"residual parent vs ref_fp8: cos={m_res['cosine']:.8f} rel={m_res['rel_l2']:.6e} nr={m_res['norm_ratio']:.6f}")

    w = load_head_weights(Path(args.model), device)
    log(f"weights: hc_fn={tuple(w['hc_fn'].shape)} head={tuple(w['head_w'].shape)} scale={w['hc_scale'].detach().cpu().numpy().tolist()}")

    # 1) Identity floor: torch path vs itself (trivial) and vs parent-host on REF residual
    t0 = time.time()
    bundle_ref = stage_bundle("ref_residual", ref_res, w, device)
    bundle_par = stage_bundle("parent_residual", parent_res, w, device)
    log(f"staged in {time.time()-t0:.2f}s")

    report = {
        "positions": positions,
        "residual_parent_vs_ref_fp8": m_res,
        "comparisons": {},
        "per_pos": {},
        "notes": [
            "parent_host mirrors head.rs: hc_head_ref + rms_norm_ref + bf16-round acts + f32 head weight",
            "torch path mirrors model.py: hc_head + RMSNorm + F.linear(x.float(), weight_f32)",
            "floor_host_vs_torch_on_ref_residual: pure arithmetic agreement with identical input",
            "If floor cos≈1 at every stage but e2e logits diverge, residual content is the cause",
            "If floor cos≪1 at a stage, that stage's parent host math disagrees with torch — port bug",
            "BF16 act staging delta = logits_parent_host_bf16act vs logits_parent_host_f32act",
        ],
    }

    def add(key, a, b, tag_pos=True):
        all_m = compare_pair(key, a, b)
        report["comparisons"][key] = all_m
        if tag_pos:
            rows = per_pos_table(a, b, positions)
            report["per_pos"][key] = rows
            report["comparisons"][key]["per_pos_summary"] = summarize(rows)
        log(
            f"{key:55s} cos={all_m['cosine']:.8f} rel={all_m['rel_l2']:.6e} nr={all_m['norm_ratio']:.6f}"
            + (
                f"  mean_pos_cos={report['comparisons'][key].get('per_pos_summary',{}).get('mean_cosine', float('nan')):.8f}"
                if tag_pos
                else ""
            )
        )

    # FLOOR: parent-host vs torch on identical REF residual
    add("floor_hc_parenthost_vs_torch_on_ref", bundle_ref["hc_parent_host"], bundle_ref["hc_torch"])
    add("floor_norm_parenthost_vs_torch_on_ref", bundle_ref["norm_parent_host"], bundle_ref["norm_torch"])
    add("floor_logits_f32act_parenthost_vs_torch_on_ref", bundle_ref["logits_parent_host_f32act"], bundle_ref["logits_torch"])
    add("floor_logits_bf16act_parenthost_vs_torch_on_ref", bundle_ref["logits_parent_host_bf16act"], bundle_ref["logits_torch"])
    # BF16 staging self-delta on torch-normed acts (isolates act rounding only)
    head_w_np = w["head_w"].detach().float().cpu().numpy()
    logits_bf16_on_torch_norm = parent_host_head_bf16(bundle_ref["norm_torch"], head_w_np)
    add("delta_bf16_act_staging_on_torch_norm", logits_bf16_on_torch_norm, bundle_ref["logits_torch"])

    # Same floors on PARENT residual (should match floors if residual only differs)
    add("floor_hc_parenthost_vs_torch_on_parent", bundle_par["hc_parent_host"], bundle_par["hc_torch"])
    add("floor_norm_parenthost_vs_torch_on_parent", bundle_par["norm_parent_host"], bundle_par["norm_torch"])
    add("floor_logits_bf16act_on_parent", bundle_par["logits_parent_host_bf16act"], bundle_par["logits_torch"])

    # Cross residual through SAME torch head: how much residual gap becomes logit gap
    add("e2e_torch_hc_parent_res_vs_ref_res", bundle_par["hc_torch"], bundle_ref["hc_torch"])
    add("e2e_torch_norm_parent_res_vs_ref_res", bundle_par["norm_torch"], bundle_ref["norm_torch"])
    add("e2e_torch_logits_parent_res_vs_ref_res", bundle_par["logits_torch"], bundle_ref["logits_torch"])

    # Parent-host on parent residual vs torch on ref residual (closest to real parent vs teacher head)
    add("cross_hc_parenthost_parentres_vs_torch_refres", bundle_par["hc_parent_host"], bundle_ref["hc_torch"])
    add("cross_norm_parenthost_parentres_vs_torch_refres", bundle_par["norm_parent_host"], bundle_ref["norm_torch"])
    add("cross_logits_bf16_parenthost_parentres_vs_torch_refres", bundle_par["logits_parent_host_bf16act"], bundle_ref["logits_torch"])

    # Verdict helpers
    floor_hc = report["comparisons"]["floor_hc_parenthost_vs_torch_on_ref"]["cosine"]
    floor_log_bf16 = report["comparisons"]["floor_logits_bf16act_parenthost_vs_torch_on_ref"]["cosine"]
    floor_log_f32 = report["comparisons"]["floor_logits_f32act_parenthost_vs_torch_on_ref"]["cosine"]
    e2e_log = report["comparisons"]["e2e_torch_logits_parent_res_vs_ref_res"]["cosine"]
    cross_log = report["comparisons"]["cross_logits_bf16_parenthost_parentres_vs_torch_refres"]["cosine"]

    if floor_hc < 0.999999 or floor_log_f32 < 0.999999:
        verdict = "HEAD_HOST_MATH_BUG"
        msg = (
            f"parent host hc/norm/f32-head disagrees with torch on identical residual "
            f"(hc_cos={floor_hc:.8f}, logits_f32_cos={floor_log_f32:.8f})"
        )
    elif floor_log_bf16 < 0.999 and e2e_log > floor_log_bf16:
        verdict = "BF16_ACT_STAGING_MATERIAL"
        msg = (
            f"BF16 act staging alone drops logits cos to {floor_log_bf16:.8f}; "
            f"check whether parent GEMM path matches ParallelHead f32"
        )
    elif e2e_log < 0.99 and floor_log_f32 > 0.999999:
        verdict = "RESIDUAL_DRIVES_LOGIT_GAP"
        msg = (
            f"head math matches on identical residual (floor logits cos={floor_log_f32:.8f}) "
            f"but torch head on parent residual vs ref residual logits cos={e2e_log:.8f} — "
            f"logit gap is residual content, not head"
        )
    elif cross_log < 0.99 and floor_log_bf16 > 0.999:
        verdict = "RESIDUAL_DRIVES_LOGIT_GAP"
        msg = f"cross logits cos={cross_log:.8f} with healthy head floor — residual content"
    else:
        verdict = "HEAD_PATH_NEEDS_GPU_PARENT_COMPARE"
        msg = (
            f"host floors healthy (hc={floor_hc:.8f}, logits_bf16={floor_log_bf16:.8f}, "
            f"logits_f32={floor_log_f32:.8f}); e2e_logits_cos={e2e_log:.8f} cross={cross_log:.8f}. "
            f"Still need GPU parent_head vs host on same residual if logits remain bad."
        )

    report["verdict"] = verdict
    report["verdict_msg"] = msg
    log(f"VERDICT: {verdict}")
    log(msg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "head_path_content_compare.json"
    path.write_text(json.dumps(report, indent=2))
    log(f"wrote {path}")

    # tiny npz of intermediate tensors for follow-up
    np.savez_compressed(
        out / "head_path_stages.npz",
        ref_residual=ref_res,
        parent_residual=parent_res,
        ref_hc_torch=bundle_ref["hc_torch"],
        ref_norm_torch=bundle_ref["norm_torch"],
        parent_hc_torch=bundle_par["hc_torch"],
        parent_norm_torch=bundle_par["norm_torch"],
        ref_hc_parenthost=bundle_ref["hc_parent_host"],
        parent_hc_parenthost=bundle_par["hc_parent_host"],
        # logits are large-ish but 11*129280*4 ≈ 5.6MB each — keep a few
        ref_logits_torch=bundle_ref["logits_torch"],
        parent_logits_torch=bundle_par["logits_torch"],
        parent_logits_parenthost_bf16=bundle_par["logits_parent_host_bf16act"],
    )
    log(f"wrote {out / 'head_path_stages.npz'}")


if __name__ == "__main__":
    main()
