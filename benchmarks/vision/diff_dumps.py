#!/usr/bin/env python3
"""Diff HF reference vs hipfire dumped tensors for a single image."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_hipfire_blob(stem: Path):
    """Load <stem>.bin + <stem>.json into a numpy array."""
    bin_path = stem.with_suffix(".bin")
    json_path = stem.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(meta["shape"])
    return arr, meta


def diff(a, b, name):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={a.shape} b={b.shape}")
        return
    d = a - b
    rel = np.abs(d).sum() / (np.abs(a).sum() + 1e-9)
    max_abs = np.abs(d).max()
    print(f"  {name}: shape={a.shape}")
    print(f"    HF      mean={a.mean():+.5f} std={a.std():.5f} min={a.min():+.5f} max={a.max():+.5f}")
    print(f"    hipfire mean={b.mean():+.5f} std={b.std():.5f} min={b.min():+.5f} max={b.max():+.5f}")
    print(f"    diff:   rel-L1={rel:.4e}  max|Δ|={max_abs:.4e}  mean|Δ|={np.abs(d).mean():.4e}")
    if max_abs < 1e-4:
        print(f"    >>> NUMERICALLY IDENTICAL")
    elif rel < 1e-3:
        print(f"    >>> very close (likely just precision)")
    elif rel < 0.1:
        print(f"    >>> CLOSE but not identical — worth investigating")
    else:
        print(f"    >>> DIVERGENT — likely root cause")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_stem", help="e.g. barney_cigar")
    ap.add_argument("--hf-dir", default="hf-ref")
    ap.add_argument("--hipfire-dir", default="hipfire-dump")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir) / args.image_stem
    hf_dir = hf_dir if hf_dir.is_absolute() else Path.cwd() / hf_dir
    hp_dir = Path(args.hipfire_dir) / args.image_stem
    hp_dir = hp_dir if hp_dir.is_absolute() else Path.cwd() / hp_dir

    print(f"HF dir:      {hf_dir}")
    print(f"hipfire dir: {hp_dir}")
    print()

    # 1. Compare patches (post-preprocessor, pre-vision-tower).
    hf_pixels = np.load(hf_dir / "pixel_values.npy")
    hp_patches, hp_meta = load_hipfire_blob(hp_dir / "hipfire_patches")
    print(f"hipfire patches meta: {hp_meta}")
    print(f"HF grid_thw: {json.load((hf_dir / 'meta.json').open())['grid_thw']}")
    print()

    print("=== Patches (post-preprocessor) ===")
    diff(hf_pixels, hp_patches, "patches")

    # 2. If they don't match, check various permutations to localize.
    if hf_pixels.shape == hp_patches.shape:
        d = np.abs(hf_pixels - hp_patches)
        if d.max() > 1e-4:
            print("\n--- additional checks (patches differ) ---")
            # Check if the difference is uniform vs concentrated in one dim.
            per_patch_max = d.max(axis=1)
            print(f"  per-patch max |Δ|: min={per_patch_max.min():.4e} max={per_patch_max.max():.4e}")
            per_elem_max = d.max(axis=0)
            print(f"  per-elem max |Δ|:  min={per_elem_max.min():.4e} max={per_elem_max.max():.4e}")

            # Try RGB-vs-RBG swap on hipfire side. Hipfire stores (R, B, G); HF stores (R, G, B).
            # Each row is [t0_C0, t0_C1, t0_C2, t1_C0, t1_C1, t1_C2] with each Ci flattened to patch_h*patch_w.
            # For temporal=2, channels=3, patch=16x16: 2*3*16*16 = 1536, with strides (768, 256, 1) for (t, c, ph*pw).
            ps = 16
            t_stride = 3 * ps * ps  # 768
            c_stride = ps * ps       # 256
            hp_unflat = hp_patches.reshape(-1, 2, 3, ps, ps)  # (n, t, c, ph, pw)
            # If hipfire has R,B,G and HF expects R,G,B, swap channels 1<->2:
            hp_rgb = hp_unflat[:, :, [0, 2, 1], :, :].reshape(-1, 1536)
            d_rgb = np.abs(hf_pixels - hp_rgb)
            print(f"  hipfire RGB-corrected (swap ch1<->ch2):  max|Δ|={d_rgb.max():.4e}  rel-L1={d_rgb.sum() / (np.abs(hf_pixels).sum() + 1e-9):.4e}")

            # Try patch-grid transpose (h<->w): reshape hipfire patches as (ph, pw, elems), transpose to (pw, ph, elems), reflatten.
            grid_thw = json.load((hf_dir / 'meta.json').open())['grid_thw']
            _, gh, gw = grid_thw[0]
            try:
                hp_t = hp_patches.reshape(gh, gw, 1536).transpose(1, 0, 2).reshape(-1, 1536)
                d_t = np.abs(hf_pixels - hp_t)
                print(f"  hipfire patch-grid transposed (h<->w):   max|Δ|={d_t.max():.4e}  rel-L1={d_t.sum() / (np.abs(hf_pixels).sum() + 1e-9):.4e}")
            except Exception as e:
                print(f"  patch-grid transpose check skipped: {e}")

            # Try both: RGB-swap + transpose.
            try:
                hp_both = hp_rgb.reshape(gh, gw, 1536).transpose(1, 0, 2).reshape(-1, 1536)
                d_both = np.abs(hf_pixels - hp_both)
                print(f"  hipfire RGB-swap + transposed:           max|Δ|={d_both.max():.4e}  rel-L1={d_both.sum() / (np.abs(hf_pixels).sum() + 1e-9):.4e}")
            except Exception as e:
                pass

            # HYPOTHESIS: HF reorders patches into 2x2 spatial-merge blocks.
            # For grid (gh, gw), reshape (gh, gw, 1536) → (gh/2, 2, gw/2, 2, 1536)
            # then permute axes (0, 2, 1, 3, 4) → flatten (gh/2 * gw/2 * 2 * 2, 1536).
            # Equivalent: groups are (py//2, px//2); within group order is (py%2, px%2).
            assert gh % 2 == 0 and gw % 2 == 0, f"grid {gh}x{gw} not divisible by 2"
            sms = 2
            hp_grid = hp_patches.reshape(gh, gw, 1536)
            hp_2x2 = hp_grid.reshape(gh // sms, sms, gw // sms, sms, 1536) \
                            .transpose(0, 2, 1, 3, 4) \
                            .reshape(-1, 1536)
            d_2x2 = np.abs(hf_pixels - hp_2x2)
            print(f"  hipfire 2x2-block reordered:             max|Δ|={d_2x2.max():.4e}  rel-L1={d_2x2.sum() / (np.abs(hf_pixels).sum() + 1e-9):.4e}")

            # Same 2x2 reorder PLUS RGB swap.
            hp_2x2_rgb = hp_2x2.reshape(-1, 2, 3, 16, 16)[:, :, [0, 2, 1], :, :].reshape(-1, 1536)
            d_2x2_rgb = np.abs(hf_pixels - hp_2x2_rgb)
            print(f"  hipfire 2x2 reorder + RGB swap:          max|Δ|={d_2x2_rgb.max():.4e}  rel-L1={d_2x2_rgb.sum() / (np.abs(hf_pixels).sum() + 1e-9):.4e}")

            # Same 2x2 reorder PLUS B<->G swap (hipfire's deliberate (R,B,G) order
            # would be HF's (R,G,B); applying the reverse permutation to hipfire
            # is equivalent to swapping channels 1 and 2).
            # Already tested above as hp_2x2_rgb.


if __name__ == "__main__":
    main()
