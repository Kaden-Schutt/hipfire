#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Compare per-layer hidden states between HF transformers oracle and
hipfire dump (Step A phase 3).

Reads two HFHS-format binary dumps produced by
  scripts/dump_hf_hidden_states.py    (HF transformers BF16 oracle)
  examples/dump_qwen35_hidden_states  (hipfire forward, captured via
                                       HiddenStateRingBuffer)

Each dump must have the SAME (n_layers, n_pos, hidden_dim) for the same
chunk. For each layer the comparator reports:

  - RMS of HF, hipfire, and (HF - hipfire)
  - mean cosine similarity per position
  - relative L2 error  ||hf - hipfire|| / ||hf||  averaged over positions
  - min cosine across positions (worst-case alignment per layer)

The drift profile across layers tells us: (a) which layer first
diverges materially, (b) whether the gap grows monotonically (drift
accumulates layer-on-layer) or has a sharp step (one layer is broken).

Usage:
    compare_hidden_states.py --hf <hf_dump.bin> --hipfire <hipfire_dump.bin>
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
    return p.parse_args()


def read_hfhs(path: Path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"HFHS\0\0\0\0":
            sys.exit(f"{path}: bad magic {magic!r}")
        n_layers, n_pos, hidden_dim, _reserved = struct.unpack("<IIII", f.read(16))
        # body: n_layers * [n_pos, hidden_dim] f32
        body = np.frombuffer(f.read(), dtype=np.float32)
    expected = n_layers * n_pos * hidden_dim
    if body.size != expected:
        sys.exit(
            f"{path}: body has {body.size} f32s, expected {expected} "
            f"({n_layers}*{n_pos}*{hidden_dim})"
        )
    body = body.reshape(n_layers, n_pos, hidden_dim)
    return n_layers, n_pos, hidden_dim, body


def main():
    args = parse_args()
    n_layers_a, n_pos_a, hidden_a, hf = read_hfhs(Path(args.hf))
    n_layers_b, n_pos_b, hidden_b, hip = read_hfhs(Path(args.hipfire))
    if (n_layers_a, n_pos_a, hidden_a) != (n_layers_b, n_pos_b, hidden_b):
        sys.exit(
            f"shape mismatch HF {(n_layers_a, n_pos_a, hidden_a)} vs "
            f"hipfire {(n_layers_b, n_pos_b, hidden_b)}"
        )
    n_layers, n_pos, hidden = n_layers_a, n_pos_a, hidden_a
    print(
        f"comparing {n_layers} layers x {n_pos} positions x {hidden} hidden",
        flush=True,
    )

    print(
        f"\n{'layer':>5} {'hf_rms':>10} {'hip_rms':>10} {'diff_rms':>10} "
        f"{'rel_L2':>10} {'mean_cos':>10} {'min_cos':>10}",
        flush=True,
    )
    print("-" * 70, flush=True)

    for layer in range(n_layers):
        h = hf[layer]  # [n_pos, hidden]
        p = hip[layer]
        diff = h - p
        hf_rms = float(np.sqrt(np.mean(h.astype(np.float64) ** 2)))
        hip_rms = float(np.sqrt(np.mean(p.astype(np.float64) ** 2)))
        diff_rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        # Per-position relative L2 + cosine
        h_norm = np.linalg.norm(h.astype(np.float64), axis=-1)
        p_norm = np.linalg.norm(p.astype(np.float64), axis=-1)
        diff_norm = np.linalg.norm(diff.astype(np.float64), axis=-1)
        # Avoid div-by-zero (a layer-0-pre-RMSNorm row could be ~0)
        rel_l2 = diff_norm / np.maximum(h_norm, 1e-12)
        mean_rel_l2 = float(np.mean(rel_l2))
        # Cosine via dot / (||h|| ||p||)
        dot = np.einsum("ij,ij->i", h.astype(np.float64), p.astype(np.float64))
        cos = dot / np.maximum(h_norm * p_norm, 1e-12)
        mean_cos = float(np.mean(cos))
        min_cos = float(np.min(cos))
        print(
            f"{layer:>5} {hf_rms:>10.4f} {hip_rms:>10.4f} {diff_rms:>10.4f} "
            f"{mean_rel_l2:>10.4f} {mean_cos:>10.6f} {min_cos:>10.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
