#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate a tiny random-weight MiniMax-M2 oracle for hipfire arch validation.

Uses the BUILT-IN transformers `MiniMaxM2` (compatible with the installed
transformers; the checkpoint's bundled custom modeling needs a newer API).
The built-in impl is a faithful MiniMax-M2: per-LAYER QK-norm (RMSNorm on the
flat n_heads*head_dim / n_kv*head_dim vector), partial rotate_half RoPE
(rotary_dim = head_dim*partial_rotary_factor), sigmoid+bias top-k routing
(gather-unbiased + normalize, no scaling), SwiGLU experts, no shared expert.

Dims keep attention in the real regime (head_dim=128, rotary_dim=64) and make
every 2D weight k%256==0 (HFQ4G256-quantizable): hidden=256, 4Q/2KV hd128,
inter=256, 8 experts top-2, 2 layers, vocab=512.

Built-in stores experts PACKED (gate_up_proj[E,2I,H], down_proj[E,H,I]); the
real ckpt + hipfire loader want SPLIT (block_sparse_moe.experts.E.w{1,2,3}).
So we RE-SPLIT the state_dict before saving model.safetensors (numerically
identical, just reorganized): w1=gate_up[:I], w3=gate_up[I:], w2=down.

Outputs into <out>:
  model.safetensors + config.json  (SPLIT layout, flat arch fields → hipfire)
  oracle_hidden.hfhs               (HF per-layer post-residual, pre-final-norm)
  tokens.hfkldr                    (fixed token chunk for both dumpers)
"""
import argparse, json, struct, sys
from pathlib import Path
import torch
from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM
from safetensors.torch import save_file

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/minimax-tiny")
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    n_layers, inter, hidden = 2, 256, 256
    # top-8 to match the hardcoded "k8" indexed-MoE GEMV kernels (the real
    # MiniMax-M2 is also top-8). 16 experts → genuine top-8 sparsity.
    cfg = MiniMaxM2Config(
        vocab_size=512, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, num_local_experts=16, num_experts_per_tok=8,
        max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                         "partial_rotary_factor": 0.5},  # rotary_dim = 128*0.5 = 64
    )
    model = MiniMaxM2ForCausalLM(cfg).to(torch.float32).eval()
    # Spread the routing bias so top-k selection is non-trivial.
    with torch.no_grad():
        for layer in model.model.layers:
            b = layer.mlp.e_score_correction_bias
            b.copy_(torch.randn_like(b) * 0.1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Fixed token chunk (seeded).
    g = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(0, cfg.vocab_size, (args.n_ctx,), generator=g).tolist()
    print(f"tokens ({args.n_ctx}): {tokens}", flush=True)
    with open(out / "tokens.hfkldr", "wb") as f:
        f.write(b"HFKLDR\0\0"); hdr = bytearray(24)
        struct.pack_into("<I", hdr, 4, args.n_ctx); struct.pack_into("<I", hdr, 12, 1)
        f.write(hdr); f.write(struct.pack(f"<{args.n_ctx}I", *tokens))

    # Forward (+ hidden states) and pre-final-norm hook.
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        res = model(input_ids, output_hidden_states=True)
    hs = res.hidden_states
    cap = {}
    h = model.model.norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()
    post_last = cap["x"][0]
    with open(out / "oracle_hidden.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0"); f.write(struct.pack("<IIII", n_layers, args.n_ctx, hidden, 0))
        for k in range(n_layers):
            t = hs[k + 1][0] if k < n_layers - 1 else post_last
            assert tuple(t.shape) == (args.n_ctx, hidden), (k, t.shape)
            arr = t.float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
            print(f"  layer {k}: rms={float((arr.astype('float64')**2).mean()**0.5):.4f}", flush=True)

    # Post-ATTENTION residual oracle (input to each post_attention_layernorm),
    # for attention-vs-MoE divergence localization.
    pa = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            (lambda idx: (lambda m, i: pa.__setitem__(idx, i[0].detach())))(li)))
    with torch.no_grad():
        _ = model(input_ids)
    for h2 in handles:
        h2.remove()
    with open(out / "oracle_postattn.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0"); f.write(struct.pack("<IIII", n_layers, args.n_ctx, hidden, 0))
        for k in range(n_layers):
            arr = pa[k][0].float().cpu().contiguous().numpy()
            assert arr.shape == (args.n_ctx, hidden), (k, arr.shape)
            f.write(arr.tobytes())
    print(f"wrote {out}/oracle_postattn.hfhs", flush=True)

    # Re-split PACKED → SPLIT and save model.safetensors.
    sd = model.state_dict()
    split = {}
    for name, t in sd.items():
        t = t.detach().to(torch.float32).contiguous()
        if name.endswith("mlp.experts.gate_up_proj"):           # [E, 2I, H]
            pre = name[:-len("mlp.experts.gate_up_proj")]
            for e in range(cfg.num_local_experts):
                split[f"{pre}block_sparse_moe.experts.{e}.w1.weight"] = t[e][:inter, :].contiguous()
                split[f"{pre}block_sparse_moe.experts.{e}.w3.weight"] = t[e][inter:, :].contiguous()
        elif name.endswith("mlp.experts.down_proj"):             # [E, H, I]
            pre = name[:-len("mlp.experts.down_proj")]
            for e in range(cfg.num_local_experts):
                split[f"{pre}block_sparse_moe.experts.{e}.w2.weight"] = t[e].contiguous()
        elif name.endswith("mlp.gate.weight"):
            split[name.replace("mlp.gate.weight", "block_sparse_moe.gate.weight")] = t
        elif name.endswith("mlp.e_score_correction_bias"):
            split[name.replace("mlp.e_score_correction_bias",
                               "block_sparse_moe.e_score_correction_bias")] = t
        else:
            split[name] = t  # attn / norms / embed / lm_head — unchanged HF names
    save_file(split, str(out / "model.safetensors"))
    print(f"re-split → {len(split)} tensors", flush=True)

    # config.json with flat arch fields (real-ckpt convention hipfire parses).
    conf = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    conf.update(dict(
        architectures=["MiniMaxM2ForCausalLM"], model_type="minimax_m2",
        vocab_size=512, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, num_local_experts=16, num_experts_per_tok=8,
        rotary_dim=64, rope_theta=5_000_000.0, rms_norm_eps=1e-6,
        use_qk_norm=True, use_routing_bias=True, scoring_func="sigmoid",
        max_position_embeddings=512, num_mtp_modules=0, tie_word_embeddings=False,
    ))
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    print(f"wrote {out}/config.json (flat arch fields)", flush=True)
    print(f"DONE → {out}", flush=True)

if __name__ == "__main__":
    main()
