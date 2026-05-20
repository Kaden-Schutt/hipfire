#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Per-position drift breakdown for a single layer between HF oracle and
hipfire dumps. Distinguishes "drift grows with state-step index" (recurrent
accumulation) vs "drift roughly constant" (input-amplification at the layer
level).

Usage:
    compare_layer_positions.py --hf <hf_dump.bin> --hipfire <hipfire_dump.bin> \
                               --layer N
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--hipfire", required=True)
    p.add_argument("--layer", type=int, required=True)
    return p.parse_args()


def read_hfhs(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"HFHS\0\0\0\0":
            sys.exit(f"{path}: bad magic")
        n_layers, n_pos, hidden, _ = struct.unpack("<IIII", f.read(16))
        body = np.frombuffer(f.read(), dtype=np.float32)
    return n_layers, n_pos, hidden, body.reshape(n_layers, n_pos, hidden)


def main():
    args = parse_args()
    n_l_a, n_p_a, h_a, hf = read_hfhs(Path(args.hf))
    n_l_b, n_p_b, h_b, hip = read_hfhs(Path(args.hipfire))
    if (n_l_a, n_p_a, h_a) != (n_l_b, n_p_b, h_b):
        sys.exit("shape mismatch")
    if not (0 <= args.layer < n_l_a):
        sys.exit(f"layer {args.layer} out of range [0, {n_l_a})")

    h = hf[args.layer]  # [n_pos, hidden]
    p = hip[args.layer]
    diff = h - p
    h_norm = np.linalg.norm(h.astype(np.float64), axis=-1)
    p_norm = np.linalg.norm(p.astype(np.float64), axis=-1)
    diff_norm = np.linalg.norm(diff.astype(np.float64), axis=-1)
    rel_l2 = diff_norm / np.maximum(h_norm, 1e-12)
    dot = np.einsum("ij,ij->i", h.astype(np.float64), p.astype(np.float64))
    cos = dot / np.maximum(h_norm * p_norm, 1e-12)

    n_pos = h.shape[0]
    bucket_size = max(1, n_pos // 16)  # ~16 buckets
    print(f"layer {args.layer}: per-position drift across {n_pos} positions")
    print(f"{'pos_range':>14} {'mean_rel_L2':>12} {'min_cos':>10} {'mean_cos':>10}")
    print("-" * 50)
    for s in range(0, n_pos, bucket_size):
        e = min(s + bucket_size, n_pos)
        m_rel = float(np.mean(rel_l2[s:e]))
        mn_cos = float(np.min(cos[s:e]))
        mc = float(np.mean(cos[s:e]))
        print(f"  [{s:4d}..{e:4d}] {m_rel:>12.4f} {mn_cos:>10.4f} {mc:>10.6f}")

    print()
    print(f"  overall mean rel_L2 = {float(np.mean(rel_l2)):.4f}")
    print(f"  pos 0..127 mean = {float(np.mean(rel_l2[:128])):.4f}")
    print(f"  pos 1024..1151 mean = {float(np.mean(rel_l2[1024:1152])):.4f}")
    print(f"  pos 1920..2047 mean = {float(np.mean(rel_l2[1920:2048])):.4f}")


if __name__ == "__main__":
    main()
