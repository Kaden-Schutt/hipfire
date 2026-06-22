#!/usr/bin/env python3
"""Diff HF-reference vs hipfire vision-tower dumps for one image, stage by stage.

Family-agnostic: matches stages by name across the two dirs and diffs every
common one in pipeline order, flagging the FIRST divergent stage (the most
likely root cause). No model-family-specific shapes or permutations.

  HF dir      (from dump_hf_reference.py):  <stage>.npy
  hipfire dir (from HIPFIRE_VISION_DUMP):   <stage>.bin + <stage>.json {"shape": [...]}

Stage-name aliases bridge the two sides' vocab (e.g. HF "pixel_values" ==
hipfire "patches", HF "post_merger" == hipfire "image_embeds").

Usage:
  diff_dumps.py <image_stem> --hf-dir hf-ref --hipfire-dir hipfire-dump
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

# Canonical stage order for reporting. Block stages (block_NN) are expanded
# numerically between post_pos_embed and pre_merger.
STAGE_ORDER = ["pixel_values", "patch_embed", "post_pos_embed", "pre_merger", "post_merger"]

# Map differing names on each side to a shared canonical stage key.
ALIASES = {
    "patches": "pixel_values",
    "image_embeds": "post_merger",
    "img_embeds": "post_merger",
    "post_projector": "post_merger",
    "post_layernorm": "pre_merger",
}


def canon(name: str) -> str:
    return ALIASES.get(name, name)


def load_npy_dir(d: Path) -> dict:
    out = {}
    for p in d.glob("*.npy"):
        out[canon(p.stem)] = np.load(p)
    return out


def load_hipfire_dir(d: Path) -> dict:
    """hipfire dumps as <stem>.bin (f32) + <stem>.json {"shape": [...]}."""
    out = {}
    for j in d.glob("*.json"):
        if j.stem == "meta":
            continue
        try:
            meta = json.loads(j.read_text())
            shape = meta.get("shape")
            if shape is None:
                continue
            arr = np.fromfile(j.with_suffix(".bin"), dtype=np.float32).reshape(shape)
            out[canon(j.stem)] = arr
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {j.name}: {e})")
    return out


def ordered_stages(keys) -> list:
    blocks = sorted(k for k in keys if re.fullmatch(r"block_\d+", k))
    out = []
    for s in STAGE_ORDER:
        if s == "pre_merger":
            out.extend(blocks)
        if s in keys:
            out.append(s)
    # Any leftover keys not in the canonical order, appended for visibility.
    out.extend(sorted(k for k in keys if k not in out and k not in blocks))
    return out


def diff(a, b, name) -> bool:
    """Return True if this stage is the first 'divergent' one."""
    if a.shape != b.shape:
        # Try a flatten-compatible compare before declaring shape mismatch.
        if a.size == b.size:
            print(f"  {name}: shape differs {a.shape} vs {b.shape} (same size — compared flat)")
            a, b = a.reshape(-1), b.reshape(-1)
        else:
            print(f"  {name}: SHAPE MISMATCH a={a.shape} b={b.shape}  <<< divergent")
            return True
    d = np.abs(a - b)
    rel = d.sum() / (np.abs(a).sum() + 1e-9)
    print(f"  {name:14s} shape={str(tuple(a.shape)):20s} "
          f"HF(mean={a.mean():+.4f},std={a.std():.4f}) "
          f"hf-hp: rel-L1={rel:.3e} max|Δ|={d.max():.3e}")
    return rel >= 0.1  # >10% relative L1 → genuinely divergent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_stem")
    ap.add_argument("--hf-dir", default="hf-ref")
    ap.add_argument("--hipfire-dir", default="hipfire-dump")
    args = ap.parse_args()

    hf = load_npy_dir(Path(args.hf_dir) / args.image_stem)
    hp = load_hipfire_dir(Path(args.hipfire_dir) / args.image_stem)
    common = set(hf) & set(hp)
    print(f"HF stages: {len(hf)}  hipfire stages: {len(hp)}  common: {len(common)}")
    only_hf = sorted(set(hf) - set(hp))
    only_hp = sorted(set(hp) - set(hf))
    if only_hf:
        print(f"  only in HF:      {only_hf}")
    if only_hp:
        print(f"  only in hipfire: {only_hp}")
    print()

    first_divergent = None
    for stage in ordered_stages(common):
        if diff(hf[stage], hp[stage], stage) and first_divergent is None:
            first_divergent = stage
    print()
    if first_divergent:
        print(f">>> FIRST DIVERGENT STAGE: {first_divergent} — start the fix here.")
    elif common:
        print(">>> all common stages match within tolerance.")
    else:
        print(">>> no common stages — check that hipfire dumped the same names.")


if __name__ == "__main__":
    main()
