#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

"""Per-stage diff: hipfire vision_forward dumps vs captured HF reference.

Reads the sampled HF reference activations from
`benchmarks/references/<image>_activations/` (sampled at fixed
row-indices recorded in index.json), and compares them against the
full hipfire stage dumps produced by `HIPFIRE_DOTS_OCR_DUMP_DIR=...`
sampled at the same row-indices.

Reports per-stage cosine + max abs diff, identifies the FIRST stage
that diverges (i.e. where the bug enters), and prints the per-row
breakdown for that stage.

Usage:
  python3 scripts/diff_dots_ocr_stages.py \\
      --hf-ref benchmarks/references/dots_ocr_smoke_001_activations \\
      --hipfire-dump /data/cache/hipfire/dots_ocr_hipfire_dump
"""

import argparse
import json
import pathlib
import sys

import numpy as np


STAGES = [
    "patch_embed",
    "block_00_attn_out", "block_00",
    "block_01_attn_out", "block_01",
    "block_02_attn_out", "block_02",
    "block_04_attn_out", "block_04",
    "block_08_attn_out", "block_08",
    "block_12_attn_out", "block_12",
    "block_16_attn_out", "block_16",
    "block_21_attn_out", "block_21",
    "block_41_attn_out", "block_41",
    "post_trunk_norm", "merger",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (or row-wise mean if 2-D)."""
    if a.ndim == 1:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    # 2-D: per-row cosines, return mean
    nums = (a * b).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    mask = (na > 0) & (nb > 0)
    cos = np.zeros(a.shape[0], dtype=np.float64)
    cos[mask] = nums[mask] / (na[mask] * nb[mask])
    return cos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ref", required=True, type=pathlib.Path)
    ap.add_argument("--hipfire-dump", required=True, type=pathlib.Path)
    ap.add_argument("--worst-n", type=int, default=5, help="Worst-N rows for the divergence stage")
    args = ap.parse_args()

    idx = json.loads((args.hf_ref / "index.json").read_text())
    captures = {c["name"]: c for c in idx["captures"]}

    print(f"HF capture:   {args.hf_ref}")
    print(f"hipfire dump: {args.hipfire_dump}")
    print(f"image:        {idx.get('image_path', '?')}")
    print(f"grid_thw:     {idx['image_grid_thw']}")
    print()
    print(f"{'stage':18s}  {'hf shape':>15s}  {'our shape':>15s}  {'mean cos':>9s}  {'min cos':>9s}  {'max |Δ|':>10s}  {'mean |Δ|':>10s}  status")
    print("-" * 110)

    first_diverge = None
    diverge_details = None

    for stage in STAGES:
        cap = captures.get(stage)
        if cap is None:
            print(f"{stage:18s}  (no HF capture)")
            continue
        sample_idx = np.asarray(cap["sample_indices"], dtype=np.int64)
        hf_ref = np.load(args.hf_ref / cap["file"]).astype(np.float64)  # [N_sample, D]
        full = np.load(args.hipfire_dump / f"{stage}.npy")
        our = full[sample_idx].astype(np.float64)

        cos = cosine(our, hf_ref)
        diffs = np.abs(our - hf_ref)
        mean_cos = float(cos.mean())
        min_cos = float(cos.min())
        max_d = float(diffs.max())
        mean_d = float(diffs.mean())

        # Treat as DIVERGED if mean cos < 0.99 — anything worse than
        # F16-cast slack is a real bug.
        diverged = mean_cos < 0.99
        status = "FAIL" if diverged else "ok"
        print(
            f"{stage:18s}  {str(list(hf_ref.shape)):>15s}  {str(list(our.shape)):>15s}  "
            f"{mean_cos:>9.5f}  {min_cos:>9.5f}  {max_d:>10.3f}  {mean_d:>10.3f}  {status}"
        )

        if diverged and first_diverge is None:
            first_diverge = stage
            # Sort rows by cosine (ascending → worst first)
            row_order = np.argsort(cos)
            diverge_details = {
                "stage": stage,
                "cos": cos,
                "diffs": diffs,
                "row_order": row_order,
                "hf_ref": hf_ref,
                "our": our,
                "sample_idx": sample_idx,
            }

    print()
    if first_diverge is None:
        print("✓ all stages within tolerance — pipeline matches HF reference")
        return 0

    print(f"FIRST diverging stage: {first_diverge}")
    print()
    d = diverge_details
    print(f"Worst {args.worst_n} rows by cosine (lowest similarity first):")
    print(f"  {'sample':>6s}  {'row':>5s}  {'cos':>9s}  {'max |Δ|':>10s}  {'hf mean':>10s}  {'our mean':>10s}  {'hf norm':>10s}  {'our norm':>10s}")
    for r in d["row_order"][: args.worst_n]:
        gi = int(d["sample_idx"][r])
        cos_r = d["cos"][r]
        max_d_r = d["diffs"][r].max()
        hf_mean = d["hf_ref"][r].mean()
        our_mean = d["our"][r].mean()
        hf_norm = np.linalg.norm(d["hf_ref"][r])
        our_norm = np.linalg.norm(d["our"][r])
        print(
            f"  {r:>6d}  {gi:>5d}  {cos_r:>9.5f}  {max_d_r:>10.3f}  "
            f"{hf_mean:>10.4f}  {our_mean:>10.4f}  {hf_norm:>10.3f}  {our_norm:>10.3f}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
