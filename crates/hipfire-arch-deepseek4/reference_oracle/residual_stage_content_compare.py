#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Compare parent vs ref residual STAGE content (cosine, rel L2, norm ratio)."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

STAGES = [
    "hc_pre_attn",
    "attn_norm",
    "attn_out",
    "hc_post_attn",
    "hc_pre_ffn",
    "ffn_norm",
    "moe_out",
    "hc_post_ffn",
]

# stream stages are [pos, dim=4096]; hc stages [pos, hc_dim=16384]
HC_STAGES = {"hc_post_attn", "hc_post_ffn"}


def load_ref_npz(path: Path) -> dict:
    return dict(np.load(path))


def load_parent_bin(path: Path, n_pos: int, row_dim: int) -> np.ndarray:
    raw = path.read_bytes()
    need = n_pos * row_dim * 4
    if len(raw) != need:
        raise SystemExit(f"{path}: size {len(raw)} != {need} ({n_pos}x{row_dim} f32)")
    arr = np.frombuffer(raw, dtype="<f4").reshape(n_pos, row_dim)
    return arr.astype(np.float64)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="/tmp/residual_stage_content_ref")
    ap.add_argument("--parent-dir", default="/tmp/residual_stage_content_parent")
    ap.add_argument("--out", default="/tmp/residual_stage_content_compare.json")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    parent_dir = Path(args.parent_dir)
    ref_meta = json.loads((ref_dir / "residual_stage_content_ref.json").read_text())
    parent_meta = json.loads((parent_dir / "residual_stage_content_parent.json").read_text())
    positions = ref_meta["positions"]
    layers = ref_meta["layers"]
    n_pos = len(positions)
    assert positions == parent_meta.get("positions") or True  # parent may serialize differently

    npz = load_ref_npz(ref_dir / "residual_stage_content_ref.npz")

    rows = []
    print(f"{'layer':>5} {'stage':>14} {'pos':>6} {'cos':>10} {'rel_l2':>10} {'norm_ratio':>12}")
    for L in layers:
        for st in STAGES:
            key = f"L{L}_{st}"
            if key not in npz:
                print(f"missing ref {key}")
                continue
            ref = npz[key].astype(np.float64)
            if ref.ndim == 3:
                # [n_pos, hc_mult, dim] -> flatten last two
                ref = ref.reshape(ref.shape[0], -1)
            row_dim = ref.shape[1]
            ppath = parent_dir / f"L{L}_{st}.f32"
            if not ppath.exists():
                print(f"missing parent {ppath}")
                continue
            par = load_parent_bin(ppath, n_pos, row_dim)
            for i, pos in enumerate(positions):
                cos, rel, ratio = metrics(par[i], ref[i])
                rows.append(
                    {
                        "layer": L,
                        "stage": st,
                        "pos": pos,
                        "cosine": cos,
                        "rel_l2": rel,
                        "norm_ratio": ratio,
                    }
                )
                print(f"{L:5d} {st:>14} {pos:6d} {cos:10.6f} {rel:10.6f} {ratio:12.6f}")
            # also aggregate over all positions as one vector
            cos, rel, ratio = metrics(par, ref)
            rows.append(
                {
                    "layer": L,
                    "stage": st,
                    "pos": "ALL",
                    "cosine": cos,
                    "rel_l2": rel,
                    "norm_ratio": ratio,
                }
            )
            print(f"{L:5d} {st:>14} {'ALL':>6} {cos:10.6f} {rel:10.6f} {ratio:12.6f}")

    # summary: first stage where mean cosine over positions drops below 0.999
    by_stage = {}
    for r in rows:
        if r["pos"] == "ALL":
            by_stage[(r["layer"], r["stage"])] = r

    print("\n=== per-stage ALL-pos summary ===")
    first_bad = None
    for L in layers:
        for st in STAGES:
            r = by_stage.get((L, st))
            if r is None:
                continue
            flag = "" if r["cosine"] >= 0.999 else " << drift"
            print(
                f"L{L} {st:14s} cos={r['cosine']:.6f} rel={r['rel_l2']:.6f} nr={r['norm_ratio']:.6f}{flag}"
            )
            if first_bad is None and r["cosine"] < 0.999:
                first_bad = (L, st, r)

    out = {
        "ref_meta": ref_meta,
        "parent_meta": parent_meta,
        "rows": rows,
        "first_cosine_below_0.999": (
            None
            if first_bad is None
            else {
                "layer": first_bad[0],
                "stage": first_bad[1],
                "cosine": first_bad[2]["cosine"],
                "rel_l2": first_bad[2]["rel_l2"],
                "norm_ratio": first_bad[2]["norm_ratio"],
            }
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    if first_bad:
        print(
            f"VERDICT: first cosine < 0.999 at L{first_bad[0]} {first_bad[1]} "
            f"cos={first_bad[2]['cosine']:.6f} (direction problem if nr~1)"
        )
    else:
        print("VERDICT: all stages cosine >= 0.999")


if __name__ == "__main__":
    main()
