#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""HF transformers per-layer hidden-state oracle (Step A phase 1).

Loads the HF checkpoint in BF16, runs one chunk's tokens (extracted
from a llama.cpp kldref so they match the hipfire eval pipeline byte-
for-byte), and dumps each transformer layer's output hidden state to a
binary file. Pair with a matching hipfire dumper + offline comparator
to localize which layer first diverges by how much.

Output format (raw little-endian):
  - header: 16 bytes
      magic 8B = b"HFHS\\0\\0\\0\\0"
      n_layers u32
      n_pos u32  (= n_ctx; we dump *every* position so cosine is
                  per-position per-layer)
      hidden_dim u32
      reserved u32 = 0
  - body: n_layers blocks, each block is [n_pos, hidden_dim] f32 row-major
    (HF returns hidden_states tuple of length n_layers+1: the first is
     the embedding; we dump layers 1..n_layers, i.e. POST-transformer
     output per layer — to match what hipfire writes post-FFN residual.)

Usage:
    dump_hf_hidden_states.py --model <hf_dir> --ref <kldref> \
                             --chunk N --out <path>
"""
import argparse
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="forward device; 'cuda' is also the PyTorch ROCm device name",
    )
    return p.parse_args()


def read_chunk_tokens(ref_path: Path, chunk_idx: int):
    with open(ref_path, "rb") as f:
        if f.read(8) != b"HFKLDR\0\0":
            sys.exit("bad ref magic")
        hdr = f.read(24)
        n_ctx = struct.unpack("<I", hdr[4:8])[0]
        n_chunk = struct.unpack("<I", hdr[12:16])[0]
        if chunk_idx >= n_chunk:
            sys.exit(f"chunk {chunk_idx} >= n_chunk {n_chunk}")
        f.seek(32 + chunk_idx * n_ctx * 4)
        tokens = list(struct.unpack(f"<{n_ctx}I", f.read(n_ctx * 4)))
        return n_ctx, tokens


def main():
    args = parse_args()
    n_ctx, tokens = read_chunk_tokens(Path(args.ref), args.chunk)
    print(f"loaded {n_ctx} tokens from chunk {args.chunk}", flush=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"loading HF model BF16 ({device}): {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if device == "cuda":
        model.to("cuda")
    model.eval()

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    hidden_dim = tcfg.hidden_size
    n_layers = tcfg.num_hidden_layers
    print(f"  hidden_dim={hidden_dim} n_layers={n_layers}", flush=True)

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    print(f"running forward with output_hidden_states=True...", flush=True)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # out.hidden_states is a tuple of length n_layers+1.
    # transformers convention (see Qwen2Model.forward):
    #   hidden_states[0]            = pre-layer-0  (= embedding output)
    #   hidden_states[i] for 1..n   = pre-layer-i  (= POST decoder-layer-(i-1))
    #   hidden_states[n_layers]     = POST FINAL NORM (model.norm applied)
    #
    # For an apples-to-apples comparison with hipfire's
    # HiddenStateRingBuffer (which writes the post-residual state at the
    # END of each decoder layer, BEFORE the final model norm), we want
    # the post-layer-i value for i = 0..n_layers-1. That is:
    #   hipfire layer k output  <->  hs[k + 1] for k = 0..n_layers-2
    #   hipfire layer (n_layers-1) output  <->  has no direct entry in
    #     the standard hidden_states tuple — hs[n_layers] is POST norm,
    #     hs[n_layers-1] is BEFORE the last layer (not after).
    #
    # To get the post-last-layer-pre-norm state we hook the final norm
    # and capture its input. Workaround: temporarily replace
    # model.model.norm with an identity wrapper that records the input,
    # rerun, then restore. Or simpler — read out via model.model.norm's
    # forward via a forward hook before this main forward() returns.
    hs = out.hidden_states
    print(f"  hidden_states tuple length: {len(hs)}", flush=True)

    # Capture pre-final-norm state via a forward-pre-hook on model.model.norm.
    # We replay with the hook attached.
    pre_norm_capture = {}
    def _capture(module, inputs):
        pre_norm_capture["x"] = inputs[0].detach()
    norm_module = model.model.norm if hasattr(model, "model") and hasattr(model.model, "norm") else None
    if norm_module is None:
        # qwen3_5 wraps in model.language_model.norm
        norm_module = model.model.language_model.norm
    handle = norm_module.register_forward_pre_hook(_capture)
    with torch.no_grad():
        _ = model(input_ids, output_hidden_states=False)
    handle.remove()
    post_last_layer = pre_norm_capture["x"][0]  # [n_ctx, hidden_dim], pre-final-norm
    print(
        f"  captured pre-final-norm via hook: shape={tuple(post_last_layer.shape)} "
        f"rms={float((post_last_layer.float() ** 2).mean() ** 0.5):.4f}",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, n_ctx, hidden_dim, 0))
        for layer_idx in range(n_layers):
            if layer_idx < n_layers - 1:
                tensor = hs[layer_idx + 1][0]  # post-layer-`layer_idx` output
            else:
                tensor = post_last_layer       # post-last-layer, pre-norm
            assert tensor.shape == (n_ctx, hidden_dim), f"layer {layer_idx} shape {tensor.shape}"
            arr = tensor.detach().float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
            if layer_idx < 3 or layer_idx == n_layers - 1:
                norm = arr.reshape(-1).astype("float64")
                rms = float((norm ** 2).mean() ** 0.5)
                print(
                    f"  layer {layer_idx}: shape={arr.shape} rms={rms:.4f}",
                    flush=True,
                )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
