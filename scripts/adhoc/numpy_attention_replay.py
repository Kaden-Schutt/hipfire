#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read

"""Replay block-N attention in numpy F32 using HF's captured QKV inputs.

Goal: tell whether our GPU attn_out divergence comes from
  (a) our attention COMPUTE being wrong (RoPE / softmax / V mult), or
  (b) our QKV linear input differing from HF's just enough to flip
      softmax winners downstream.

Strategy:
  1. Load HF's full block_N_qkv (post-linear, pre-reshape, pre-RoPE).
  2. Reshape per HF's `(seq, 3, n_heads, head_dim).permute(1,0,2,3).unbind(0)`.
  3. Apply our 2-D RoPE (using our build_rope_2d_tables semantics).
  4. Compute non-causal full attention in F32 numpy.
  5. Compare attention output (pre-proj) to HF's attn_out (post-proj).
     — since HF's hook is on the .attn module, attn_out *includes* proj.
     — we'd need proj weight from HFQ to match exactly, so we just
       compare per-row cosines to see if attention is "matching shape".

Better: also compute the same on OUR qkv. Then:
  - numpy(HF_qkv) ≈ HF_attn_out (modulo proj precision)
  - numpy(OUR_qkv) ≈ OUR_attn_out
  - numpy(HF_qkv) vs numpy(OUR_qkv): tells us how much attn divergence
    comes purely from QKV input differences, in our SAME numerical
    environment.

Run from repo root:
    python3 scripts/numpy_attention_replay.py 1   # block 1
"""

import json
import pathlib
import sys

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
HF_FULL = pathlib.Path("/data/cache/hipfire/dots_ocr_activations_full")
OUR_DUMP = pathlib.Path("/data/cache/hipfire/dots_ocr_hipfire_dump")
REF_DIR = REPO / "benchmarks" / "references" / "dots_ocr_smoke_001_activations"


# Model constants for dots.ocr vision tower
N_HEADS = 12
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM  # 1536
EMBED = HIDDEN
GRID_H = 160
GRID_W = 122
N_PATCHES = GRID_H * GRID_W  # 19520
SPATIAL_MERGE = 2
ROPE_THETA = 10000.0
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def build_rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """Mirrors `crates/hipfire-arch-dots-ocr/src/rope.rs::build_rope_2d_tables`."""
    quarter = HEAD_DIM // 4
    half = HEAD_DIM // 2
    denom = float(HEAD_DIM // 2)
    inv_freq = np.array([ROPE_THETA ** (-(2.0 * k) / denom) for k in range(quarter)], dtype=np.float64)

    cos = np.zeros((N_PATCHES, HEAD_DIM), dtype=np.float64)
    sin = np.zeros((N_PATCHES, HEAD_DIM), dtype=np.float64)

    outer_w = GRID_W // SPATIAL_MERGE
    sm = SPATIAL_MERGE
    patch_idx = 0
    for oy in range(GRID_H // sm):
        for ox in range(outer_w):
            for dy in range(sm):
                for dx in range(sm):
                    hpos = oy * sm + dy
                    wpos = ox * sm + dx
                    for k in range(quarter):
                        h_angle = hpos * inv_freq[k]
                        w_angle = wpos * inv_freq[k]
                        hc, hs = np.cos(h_angle), np.sin(h_angle)
                        wc, ws = np.cos(w_angle), np.sin(w_angle)
                        cos[patch_idx, k] = hc
                        cos[patch_idx, quarter + k] = wc
                        cos[patch_idx, half + k] = hc
                        cos[patch_idx, half + quarter + k] = wc
                        sin[patch_idx, k] = hs
                        sin[patch_idx, quarter + k] = ws
                        sin[patch_idx, half + k] = hs
                        sin[patch_idx, half + quarter + k] = ws
                    patch_idx += 1
    return cos, sin


def apply_rope_halfsplit(qk: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """In-place RoPE rotation. qk shape: [n_patches, n_heads, head_dim]."""
    half = HEAD_DIM // 2
    # Broadcast cos/sin across heads
    cos_b = cos[:, None, :]  # [n_patches, 1, head_dim]
    sin_b = sin[:, None, :]
    # Split into two halves
    qk1 = qk[..., :half]
    qk2 = qk[..., half:]
    # rotate_half(x) = concat(-x2, x1)
    rotated = np.concatenate([-qk2, qk1], axis=-1)
    return qk * cos_b + rotated * sin_b


def split_qkv(qkv_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HF's split: reshape (seq, 3, n_heads, head_dim) then permute(1,0,2,3).unbind(0).

    Equivalent to: q = qkv[:, 0:HIDDEN].reshape(seq, n_heads, head_dim), etc.
    """
    n = qkv_flat.shape[0]
    q = qkv_flat[:, 0:HIDDEN].reshape(n, N_HEADS, HEAD_DIM)
    k = qkv_flat[:, HIDDEN:2 * HIDDEN].reshape(n, N_HEADS, HEAD_DIM)
    v = qkv_flat[:, 2 * HIDDEN:3 * HIDDEN].reshape(n, N_HEADS, HEAD_DIM)
    return q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)


def attention_numpy(qkv: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Replays VisionAttention.forward (sans proj). qkv: [n, 3*hidden].

    Per-head processing to keep memory bounded: full Q/K^T at N=19520 is
    19520² × 4 bytes = 1.5 GB per head — 18 GB across heads if done at
    once. Processing one head at a time peaks at ~3 GB.
    """
    n = qkv.shape[0]
    q, k, v = split_qkv(qkv)              # [n, n_heads, head_dim]
    q = apply_rope_halfsplit(q, cos, sin)
    k = apply_rope_halfsplit(k, cos, sin)
    # Use float32 for the big intermediate to keep memory under ~3 GB/head.
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)

    out = np.zeros((n, HIDDEN), dtype=np.float64)
    for h in range(N_HEADS):
        # Per-head: scores [n, n] in F32
        scores = (q32[:, h, :] @ k32[:, h, :].T) * np.float32(SCALE)
        # Softmax in F32 along axis=1
        m = scores.max(axis=1, keepdims=True)
        np.subtract(scores, m, out=scores)
        np.exp(scores, out=scores)
        s = scores.sum(axis=1, keepdims=True)
        np.divide(scores, s, out=scores)
        # Apply V: [n, n] @ [n, head_dim] → [n, head_dim]
        head_out = scores @ v32[:, h, :]
        out[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] = head_out.astype(np.float64)
        if h % 3 == 0:
            print(f"    [head {h}/{N_HEADS} done]")
    return out


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    dot = (a * b).sum(axis=-1)
    mask = (na > 0) & (nb > 0)
    cos = np.zeros(a.shape[0], dtype=np.float64)
    cos[mask] = dot[mask] / (na[mask] * nb[mask])
    return cos


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <block_index>")
        sys.exit(1)
    block = int(sys.argv[1])

    print(f"=== block {block:02d} attention replay (F32 numpy) ===\n")

    # Load FULL tensors
    hf_qkv_full = np.load(HF_FULL / f"block_{block:02d}_qkv.npy")
    our_qkv_full = np.load(OUR_DUMP / f"block_{block:02d}_qkv.npy")
    print(f"HF qkv shape:  {hf_qkv_full.shape}  dtype={hf_qkv_full.dtype}")
    print(f"Our qkv shape: {our_qkv_full.shape}  dtype={our_qkv_full.dtype}")

    # Build RoPE tables
    print("Building RoPE tables...")
    cos_tbl, sin_tbl = build_rope_tables()
    print(f"cos[0,0]={cos_tbl[0,0]:.6f}  cos[19519, 64]={cos_tbl[-1, 64]:.6f}")

    # Compute attention via numpy on both qkv inputs
    print("\nComputing numpy attention with HF qkv (this is slow — F32 19520x19520)...")
    numpy_out_hf = attention_numpy(hf_qkv_full, cos_tbl, sin_tbl)

    print("Computing numpy attention with our qkv...")
    numpy_out_ours = attention_numpy(our_qkv_full, cos_tbl, sin_tbl)

    # Load attn_out (which includes proj — so numpy_out doesn't equal it directly)
    # Instead compare numpy_out_hf vs numpy_out_ours: tells us how much divergence
    # comes from qkv input differences in the *same* (numpy) compute environment.
    print("\nnumpy_out_hf  vs numpy_out_ours  (same algorithm, different qkv inputs):")
    cos_rows = cos_per_row(numpy_out_hf, numpy_out_ours)
    print(f"  mean cos: {cos_rows.mean():.5f}  min cos: {cos_rows.min():.5f}")
    print(f"  norm ratio (ours/hf): mean={np.linalg.norm(numpy_out_ours, axis=-1).mean() / np.linalg.norm(numpy_out_hf, axis=-1).mean():.4f}")

    # Worst-N rows
    order = np.argsort(cos_rows)[:5]
    print(f"  worst 5 rows: " + ", ".join(f"r={int(r)} cos={cos_rows[int(r)]:.4f}" for r in order))

    # Compare numpy(our_qkv) vs OUR GPU pre-proj attn output
    # (both with same qkv inputs, both pre-proj — direct diff of
    # our GPU attention KERNEL vs the F32 reference algorithm).
    our_preproj_path = OUR_DUMP / f"block_{block:02d}_attn_pre_proj.npy"
    if our_preproj_path.exists():
        our_preproj = np.load(our_preproj_path).astype(np.float64)
        cos_preproj = cos_per_row(numpy_out_ours, our_preproj)
        print(f"\n*** numpy(our_qkv) vs OUR GPU pre-proj attn (same qkv input, F32 numpy vs GPU kernel) ***")
        print(f"  mean cos: {cos_preproj.mean():.5f}  min cos: {cos_preproj.min():.5f}")
        ratio_preproj = np.linalg.norm(our_preproj, axis=-1).mean() / np.linalg.norm(numpy_out_ours, axis=-1).mean()
        print(f"  norm ratio (our_preproj / numpy_out_ours): {ratio_preproj:.4f}")
        order = np.argsort(cos_preproj)[:5]
        print(f"  worst 5 rows: " + ", ".join(f"r={int(r)} cos={cos_preproj[int(r)]:.4f}" for r in order))

    # Load HF/our attn_out and project numpy_out forward by computing the
    # delta numpy_out_hf - numpy_out_ours. If our QKV is the only divergence
    # source, this should equal HF_attn_out - OUR_attn_out (modulo proj).
    print("\n--- For reference: HF vs ours at attn_out (post-proj) ---")
    hf_attn = np.load(HF_FULL / f"block_{block:02d}_attn_out.npy")
    our_attn = np.load(OUR_DUMP / f"block_{block:02d}_attn_out.npy")
    cos_attn = cos_per_row(hf_attn.astype(np.float64), our_attn.astype(np.float64))
    print(f"  HF_attn_out vs our_attn_out: mean cos {cos_attn.mean():.5f} min cos {cos_attn.min():.5f}")


if __name__ == "__main__":
    main()
