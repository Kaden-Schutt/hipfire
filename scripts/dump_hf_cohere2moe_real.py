#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Real-weight per-layer oracle for Cohere2Moe bring-up.

Loads the ACTUAL checkpoint in HF transformers (bf16, CPU), forwards a short
prompt, and dumps per-layer post-residual hidden states (HFHS) + the token chunk
(HFKLDR) — the same format the hipfire dumper + compare_hidden_states.py use.
This localizes a structural bug that the tiny RANDOM 2-layer oracle masks
(systematic per-layer error compounds only over the real depth).

Usage:
  dump_hf_cohere2moe_real.py --model ~/bls-mini-code --out ~/cohere-real-oracle \
      --prompt "def fibonacci(n):"
Then on the hipfire side:
  dump_cohere2moe_hidden_states --model <hfq> --ref <out>/tokens.hfkldr --out <out>/hipfire.hfhs
  compare_hidden_states.py --hf <out>/oracle_hidden.hfhs --hipfire <out>/hipfire.hfhs
"""
import argparse, struct, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="def fibonacci(n):")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading model (bf16, CPU) …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"loaded: {n_layers} layers, hidden {hidden}", flush=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    n_ctx = ids.shape[1]
    tokens = ids[0].tolist()
    print(f"prompt {args.prompt!r} → {n_ctx} tokens: {tokens}", flush=True)
    with open(out / "tokens.hfkldr", "wb") as f:
        f.write(b"HFKLDR\0\0")
        hdr = bytearray(24)
        struct.pack_into("<I", hdr, 4, n_ctx)
        struct.pack_into("<I", hdr, 12, 1)
        f.write(hdr)
        f.write(struct.pack(f"<{n_ctx}I", *tokens))

    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
    hs = res.hidden_states  # embeddings, post-layer-0, …, post-layer-(n-1)

    cap = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, i: cap.__setitem__("x", i[0].detach())
    )
    with torch.no_grad():
        _ = model(ids)
    h.remove()
    post_last = cap["x"][0]

    # Also report the model's own first-token argmax (sanity: what SHOULD hipfire predict).
    logits = res.logits[0, -1].float()
    top = torch.topk(logits, 5)
    print("HF next-token top-5:", [(int(i), tok.decode([int(i)])) for i in top.indices], flush=True)

    with open(out / "oracle_hidden.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, n_ctx, hidden, 0))
        for k in range(n_layers):
            t = hs[k + 1][0] if k < n_layers - 1 else post_last
            assert tuple(t.shape) == (n_ctx, hidden), (k, t.shape)
            arr = t.float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
            if k < 4 or k >= n_layers - 2:
                print(f"  layer {k}: rms={float((arr.astype('float64')**2).mean()**0.5):.4f}", flush=True)
    print(f"DONE → {out}/oracle_hidden.hfhs", flush=True)


if __name__ == "__main__":
    main()
