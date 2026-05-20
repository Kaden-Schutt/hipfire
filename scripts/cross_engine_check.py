#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Cross-engine sanity check (Step B for engine-drift floor investigation).

Reads the first chunk's tokens from a llama.cpp-produced .kldref, runs
those exact tokens through HF transformers BF16, and computes the per-
position KL divergence between llama.cpp's top-K reference distribution
and HF transformers' distribution on the same chunk.

If KLD is small everywhere -> the two BF16 engines agree on Qwen3.5
hidden-state evolution; hipfire's ~0.4-nat drift is fixable in
principle. If KLD is large -> implementations diverge inherently; no
ground truth to chase.

Usage:
    cross_engine_check.py --model <hf_snapshot_dir> \
                          --ref <path-to-kldref.bin> \
                          [--chunk N=0]
"""
import argparse
import math
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF safetensors snapshot dir")
    p.add_argument("--ref", required=True, help=".kldref binary from build_kld_ref")
    p.add_argument("--chunk", type=int, default=0, help="chunk index to compare (default 0)")
    return p.parse_args()


def read_kldref_chunk(ref_path: Path, chunk_idx: int):
    """Return (n_ctx, n_vocab, top_k, chunk_tokens, scored_blocks).

    scored_blocks is a list of (indices, log_probs, residual_p) for each
    scored position in the chunk. Mirrors the on-disk HFKLDR-beta layout
    used by eval_hipfire/score_position.
    """
    with open(ref_path, "rb") as f:
        magic = f.read(8)
        if magic != b"HFKLDR\0\0":
            sys.exit(f"bad ref magic: {magic!r}")
        hdr = f.read(24)
        version = struct.unpack("<I", hdr[0:4])[0]
        n_ctx = struct.unpack("<I", hdr[4:8])[0]
        n_vocab = struct.unpack("<I", hdr[8:12])[0]
        n_chunk = struct.unpack("<I", hdr[12:16])[0]
        top_k = struct.unpack("<H", hdr[16:18])[0]
        if version != 1:
            sys.exit(f"unsupported version {version}")
        scored_per_chunk = n_ctx - 1 - n_ctx // 2
        block_bytes = 8 + 8 * top_k

        # Tokens: n_ctx * n_chunk u32s, all chunks contiguous
        f.seek(32 + chunk_idx * n_ctx * 4)
        if chunk_idx >= n_chunk:
            sys.exit(f"chunk {chunk_idx} >= n_chunk {n_chunk}")
        # Need to seek back to the START of tokens (offset 32) and
        # skip to the desired chunk
        f.seek(32)
        all_token_bytes_to_skip = chunk_idx * n_ctx * 4
        f.seek(32 + all_token_bytes_to_skip)
        chunk_tokens = list(struct.unpack(f"<{n_ctx}I", f.read(n_ctx * 4)))

        # Per-position blocks. Block stream starts after ALL tokens.
        tokens_end = 32 + n_ctx * n_chunk * 4
        blocks_start = tokens_end + chunk_idx * scored_per_chunk * block_bytes
        f.seek(blocks_start)

        scored_blocks = []
        for _ in range(scored_per_chunk):
            buf = f.read(block_bytes)
            indices = list(struct.unpack(f"<{top_k}I", buf[: top_k * 4]))
            log_probs = list(
                struct.unpack(f"<{top_k}f", buf[top_k * 4 : top_k * 8])
            )
            residual = struct.unpack("<f", buf[top_k * 8 : top_k * 8 + 4])[0]
            scored_blocks.append((indices, log_probs, residual))

    return n_ctx, n_vocab, top_k, chunk_tokens, scored_blocks


def main():
    args = parse_args()

    print(f"loading reference: {args.ref}", flush=True)
    n_ctx, n_vocab, top_k, chunk_tokens, scored_blocks = read_kldref_chunk(
        Path(args.ref), args.chunk
    )
    print(
        f"  n_ctx={n_ctx} n_vocab={n_vocab} top_k={top_k} "
        f"scored_per_chunk={len(scored_blocks)} chunk={args.chunk}",
        flush=True,
    )

    print(f"loading HF model BF16 (CPU): {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(
        f"  config: hidden={model.config.text_config.hidden_size if hasattr(model.config,'text_config') else model.config.hidden_size} "
        f"layers={model.config.text_config.num_hidden_layers if hasattr(model.config,'text_config') else model.config.num_hidden_layers}",
        flush=True,
    )

    print(f"running forward on {n_ctx} tokens...", flush=True)
    input_ids = torch.tensor([chunk_tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    # out.logits: [1, n_ctx, n_vocab]
    logits_bf16 = out.logits[0].float()  # cast to f32 for stable math
    print(f"  logits shape: {tuple(logits_bf16.shape)}", flush=True)

    scoring_start = n_ctx // 2
    klds_llama_to_hf = []
    klds_hf_to_llama = []
    for j, (ll_indices, ll_log_probs, ll_residual) in enumerate(scored_blocks):
        pos = scoring_start + j
        hf_logits = logits_bf16[pos]
        # Compute HF's log-softmax once
        hf_log_probs_all = torch.log_softmax(hf_logits, dim=-1)
        # HF's top-K (for the reverse-direction KL)
        hf_topk = torch.topk(hf_log_probs_all, top_k)
        hf_top_indices = hf_topk.indices.tolist()
        hf_top_log_probs = hf_topk.values.tolist()

        # === KL(llama || HF) using llama's top-K as anchor ===
        kld_a = 0.0
        sum_p_hf_at_ll_top = 0.0
        for i, idx in enumerate(ll_indices):
            log_p_ll = ll_log_probs[i]
            log_p_hf = hf_log_probs_all[idx].item()
            p_ll = math.exp(log_p_ll)
            kld_a += p_ll * (log_p_ll - log_p_hf)
            sum_p_hf_at_ll_top += math.exp(log_p_hf)
        p_resid_ll = ll_residual
        p_resid_hf = max(1.0 - sum_p_hf_at_ll_top, 0.0)
        if p_resid_ll > 1e-9 and p_resid_hf > 1e-9:
            kld_a += p_resid_ll * (math.log(p_resid_ll) - math.log(p_resid_hf))
        klds_llama_to_hf.append(max(kld_a, 0.0))

        # === KL(HF || llama) using HF's top-K as anchor ===
        # llama log_probs as a lookup: index -> log_prob
        ll_lookup = dict(zip(ll_indices, ll_log_probs))
        # Approximate llama log_prob for HF's top-K indices not in llama's top-K
        # by uniformly spreading p_resid_ll over the missing vocab.
        kld_b = 0.0
        sum_p_ll_at_hf_top = 0.0
        for i, idx in enumerate(hf_top_indices):
            log_p_hf = hf_top_log_probs[i]
            if idx in ll_lookup:
                log_p_ll = ll_lookup[idx]
            else:
                # Best-effort estimate: uniform over residual.
                remaining_vocab = max(n_vocab - len(ll_indices), 1)
                p_uniform = max(p_resid_ll / remaining_vocab, 1e-30)
                log_p_ll = math.log(p_uniform)
            p_hf = math.exp(log_p_hf)
            kld_b += p_hf * (log_p_hf - log_p_ll)
            sum_p_ll_at_hf_top += math.exp(log_p_ll)
        klds_hf_to_llama.append(max(kld_b, 0.0))

        if j < 5 or j == len(scored_blocks) - 1:
            print(
                f"  pos {pos:4d}  KL(ll||hf)={kld_a:.6f}  KL(hf||ll)={kld_b:.6f}",
                flush=True,
            )

    mean_a = sum(klds_llama_to_hf) / len(klds_llama_to_hf)
    mean_b = sum(klds_hf_to_llama) / len(klds_hf_to_llama)
    max_a = max(klds_llama_to_hf)
    max_b = max(klds_hf_to_llama)
    print()
    print("=== cross-engine KL summary (chunk", args.chunk, ") ===", flush=True)
    print(f"  KL(llama || HF):  mean={mean_a:.6f}  max={max_a:.6f}", flush=True)
    print(f"  KL(HF || llama):  mean={mean_b:.6f}  max={max_b:.6f}", flush=True)
    print(
        f"  symmetric (mean of the two): {(mean_a + mean_b) / 2:.6f}",
        flush=True,
    )

    # Context: matched hipfire q8f16 floor for Q3.5-0.8B per-token kv-q8 is 0.4945
    print()
    print(
        "Context: hipfire q8f16 KLD (per-token kv-q8, n=20 chunks) for matched-arch "
        "Q3.5-0.8B = 0.4945. If cross-engine KL << that, hipfire is the drifter."
    )


if __name__ == "__main__":
    main()
