#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Dump per-stage residual CONTENT inside selected Blocks (ref).

Step-4 follow-up when residual cosine falls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks
import torch
import numpy as np
from model import Block, Transformer
from residual_content_dump import (
    load_tokens, make_args, free_cuda, load_embed_only, load_block_weights,
    DEFAULT_MODEL, DEFAULT_TOKENS,
)

DEFAULT_LAYERS = [0, 2]
DEFAULT_POSITIONS = [0, 1, 64, 400, 448, 512, 800, 1023]

def log(m=""):
    print(m, flush=True)

def capture_pos(t, positions):
    x = t.detach().float()
    if x.dim() == 4:
        x = x[0]
    elif x.dim() == 3 and x.size(0) == 1:
        x = x[0]
    return np.stack([x[p].cpu().numpy().astype(np.float32) for p in positions], 0)

class StageCatcher:
    def __init__(self, positions):
        self.positions = positions
        self.stages = {}
    def grab(self, name, t):
        self.stages[name] = capture_pos(t, self.positions)

def instrument_block(block, catcher):
    def wrapped(x, start_pos, input_ids=None, *attn_args):
        residual = x
        y, post, comb = block.hc_pre(x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        catcher.grab("hc_pre_attn", y)
        y = block.attn_norm(y)
        catcher.grab("attn_norm", y)
        y = block.attn(y, start_pos, *attn_args)
        catcher.grab("attn_out", y)
        x = block.hc_post(y, residual, post, comb)
        catcher.grab("hc_post_attn", x)
        residual = x
        y, post, comb = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
        catcher.grab("hc_pre_ffn", y)
        y = block.ffn_norm(y)
        catcher.grab("ffn_norm", y)
        y = block.ffn(y, input_ids)
        catcher.grab("moe_out", y)
        x = block.hc_post(y, residual, post, comb)
        catcher.grab("hc_post_ffn", x)
        return x
    block.forward = wrapped

@torch.inference_mode()
def run(args):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log(f"torch={torch.__version__} device={device}")
    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    layers = [int(x) for x in args.layers.split(",")]
    positions = [int(x) for x in args.positions.split(",") if int(x) < args.seq]
    last_needed = max(layers)
    log(f"seq={args.seq} layers={layers} positions={positions}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_n = margs.n_layers
    margs.n_layers = 0
    margs.n_mtp_layers = 0
    margs.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        shell = Transformer(margs)
    margs.n_layers = full_n
    load_embed_only(shell, model_dir, device)
    shell.eval()
    tokens = load_tokens(args.tokens, args.seq).to(device)
    h = shell.embed(tokens).unsqueeze(2).repeat(1, 1, shell.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 global_L2={float(h.float().norm()):.6f}")
    del shell.head, shell.norm, shell.hc_head_fn, shell.hc_head_base, shell.hc_head_scale
    free_cuda()
    npz = {}
    meta = {"seq": args.seq, "layers": layers, "positions": positions,
            "stages": ["hc_pre_attn","attn_norm","attn_out","hc_post_attn","hc_pre_ffn","ffn_norm","moe_out","hc_post_ffn"],
            "domain": "ref bf16 block; dump f32"}
    want = set(layers)
    t0 = time.time()
    for i in range(last_needed + 1):
        with torch.device("cpu"):
            block = Block(i, margs)
        load_block_weights(block, model_dir, i, device, verbose=True)
        block.eval()
        catcher = None
        if i in want:
            catcher = StageCatcher(positions)
            instrument_block(block, catcher)
        if device.type == "cuda":
            torch.set_default_device("cuda")
        try:
            h = block(h, 0, tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            torch.set_default_device("cpu")
        log(f"L{i:02d} global_L2={float(h.float().norm()):.4f}")
        if catcher is not None:
            for name, arr in catcher.stages.items():
                key = f"L{i}_{name}"
                npz[key] = arr
                log(f"  {key} shape={arr.shape}")
        del block
        free_cuda()
    meta["wall_s"] = time.time() - t0
    npz_path = out_dir / "residual_stage_content_ref.npz"
    np.savez_compressed(npz_path, **npz)
    (out_dir / "residual_stage_content_ref.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {npz_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--positions", default=",".join(str(x) for x in DEFAULT_POSITIONS))
    ap.add_argument("--out-dir", default="/tmp/residual_stage_content_ref")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
