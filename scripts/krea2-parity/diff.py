#!/usr/bin/env python3
"""Diff two Krea2 parity dump dirs (hipfire vs diffusers reference).

Loads every .npy present in BOTH dirs and reports shape match + max/mean abs diff
and cosine similarity, so a convention mismatch (RoPE, layer indexing, adaLN
order, etc.) shows up as a large diff on the first-affected tensor.

Usage: python diff.py ref_dir hipfire_dir [--rtol 1e-2 --atol 1e-3]
"""
import argparse, glob, os
import numpy as np


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("hipfire")
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    names = sorted(
        {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.ref, "*.npy"))}
        & {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.hipfire, "*.npy"))}
    )
    if not names:
        print("no overlapping .npy files between the two dirs")
        return

    print(f"{'tensor':<24}{'shapes':<28}{'max_abs':>10}{'mean_abs':>10}{'cos':>8}  ok")
    all_ok = True
    for name in names:
        r = np.load(os.path.join(args.ref, f"{name}.npy")).astype(np.float64)
        h = np.load(os.path.join(args.hipfire, f"{name}.npy")).astype(np.float64)
        shp = f"{r.shape} vs {h.shape}"
        if r.shape != h.shape:
            print(f"{name:<24}{shp:<28}{'SHAPE MISMATCH':>28}  X")
            all_ok = False
            continue
        d = np.abs(r - h)
        ok = np.allclose(r, h, rtol=args.rtol, atol=args.atol)
        all_ok &= ok
        print(
            f"{name:<24}{shp:<28}{d.max():>10.4g}{d.mean():>10.4g}"
            f"{cos(r, h):>8.4f}  {'ok' if ok else 'X'}"
        )
    print("\nALL WITHIN TOLERANCE" if all_ok else "\nMISMATCHES FOUND (see rows marked X)")


if __name__ == "__main__":
    main()
