#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Tiny dimension-faithful LFM2.5-MoE (arch_id 11) oracle generator.
#
# Builds a small random-weight Lfm2MoeForCausalLM that EXERCISES ALL PATHS:
#   * conv mixer (LIV short-conv, K=3) AND attention mixer (GQA + qk-norm)
#   * dense SwiGLU MLP (the first num_dense_layers) AND sparse top-4 MoE
# then dumps per-layer POST-residual hidden states in the shared HFHS format
# for comparison against the hipfire decode_step (see
# docs/methodology/arch-port-validation.md and compare_hidden_states.py).
#
# Pitfalls baked in (per the methodology checklist):
#   * REAL head_dim (64) and REAL conv kernel size (conv_L_cache=3); we shrink
#     the *number* of heads/layers, not head_dim.
#   * num_experts_per_tok = 4 to match the hipfire indexed-MoE GEMV K_TOP=4.
#   * every quantized 2D weight has k % 256 == 0 (hidden=256, moe_inter=256,
#     dense inter=256, 3*hidden=768).
#   * weights persisted as bf16 (matching the real checkpoint dtype) and the
#     reference forward is run from the RELOADED bf16 values upcast to f32, so
#     the cosine isolates hipfire quant error, not bf16 rounding.
#
# Usage:
#   python3 scripts/gen_tiny_lfm2moe.py --out /tmp/tiny_lfm2moe
# Produces:
#   <out>/hf/            tiny HF checkpoint (bf16 safetensors + config.json)
#   <out>/oracle.hfhs    per-layer hidden states (HFHS)
#   <out>/tokens.json    the fixed input token ids
import argparse, json, os, struct, sys

import torch
from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

# ----- fixed tiny input -----
TOKENS = [1, 5, 9, 3, 7, 2, 8, 4]  # within tiny vocab; 8 positions


def build_tiny_config() -> Lfm2MoeConfig:
    # 5 layers: cover conv+dense, attn+dense, conv+moe, attn+moe, conv+moe.
    # num_dense_layers=2 -> layers 0,1 dense; 2,3,4 MoE.
    layer_types = ["conv", "full_attention", "conv", "full_attention", "conv"]
    return Lfm2MoeConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,        # 4 * head_dim(64) = 256
        num_key_value_heads=2,        # GQA 2:1
        head_dim=64,                  # REAL head_dim
        max_position_embeddings=1024,
        norm_eps=1e-5,
        rope_theta=1_000_000.0,
        # conv (LIV short-conv)
        conv_L_cache=3,               # REAL kernel size
        conv_bias=False,
        # dense MLP dim -> force 256 (k % 256 == 0); disable auto-adjust
        intermediate_size=256,
        block_ff_dim=256,
        block_multiple_of=256,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        # MoE
        moe_intermediate_size=256,    # k % 256 == 0
        num_experts=8,                # > num_experts_per_tok
        num_experts_per_tok=4,        # match indexed-MoE K_TOP=4
        num_dense_layers=2,
        norm_topk_prob=True,
        use_expert_bias=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        layer_types=layer_types,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        torch_dtype="bfloat16",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tiny_lfm2moe")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_tiny_config()
    print("tiny config:", json.dumps({
        "layer_types": cfg.layer_types,
        "num_dense_layers": cfg.num_dense_layers,
        "hidden_size": cfg.hidden_size,
        "head_dim": cfg.head_dim,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "conv_L_cache": cfg.conv_L_cache,
        "intermediate_size": cfg.intermediate_size,
        "moe_intermediate_size": cfg.moe_intermediate_size,
    }, indent=1))

    model = Lfm2MoeForCausalLM(cfg)
    model.eval()

    # Give expert_bias and norms non-degenerate values so routing & norms
    # actually do something (default init may leave bias at 0 / norms at 1).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("expert_bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            elif name.endswith("_norm.weight") or name.endswith("norm.weight") \
                    or name.endswith("layernorm.weight"):
                # perturb RMSNorm weights away from 1.0 to catch +1/Gemma bugs
                p.copy_(1.0 + torch.randn_like(p) * 0.05)

    hf_dir = os.path.join(args.out, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    # Persist as bf16 (real checkpoint dtype), then reload upcast to f32 so the
    # reference forward uses exactly the bf16-rounded values.
    model.to(torch.bfloat16).save_pretrained(hf_dir, safe_serialization=True)
    cfg.save_pretrained(hf_dir)
    ref = Lfm2MoeForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32)
    ref.eval()

    # Print expert/param shapes for divisibility sanity.
    for n, p in ref.named_parameters():
        if any(s in n for s in ("layers.0.", "layers.2.")) and "weight" in n:
            print(f"  param {n}: {tuple(p.shape)}")

    # ----- per-layer post-residual capture via forward hooks -----
    captures = {}

    def mk_hook(i):
        def hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captures[i] = hs.detach().float()[0].cpu()  # [seq, hidden]
        return hook

    handles = [layer.register_forward_hook(mk_hook(i))
               for i, layer in enumerate(ref.model.layers)]

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        ref(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    n_layers = cfg.num_hidden_layers
    n_pos = len(TOKENS)
    hidden = cfg.hidden_size
    # ----- write HFHS -----
    # magic "HFHS\0\0\0\0", <IIII> n_layers,n_pos,hidden,reserved,
    # then n_layers x [n_pos, hidden] f32 row-major.
    out_path = os.path.join(args.out, "oracle.hfhs")
    with open(out_path, "wb") as f:
        f.write(b"HFHS\x00\x00\x00\x00")
        f.write(struct.pack("<IIII", n_layers, n_pos, hidden, 0))
        for i in range(n_layers):
            arr = captures[i].contiguous().numpy().astype("float32")
            assert arr.shape == (n_pos, hidden), (i, arr.shape)
            f.write(arr.tobytes())
    with open(os.path.join(args.out, "tokens.json"), "w") as f:
        json.dump(TOKENS, f)

    print(f"wrote {out_path}  ({n_layers} layers x {n_pos} pos x {hidden} hidden)")
    print(f"wrote tiny HF checkpoint to {hf_dir}")
    print("RMS of last-layer hidden:", float(captures[n_layers - 1].pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
