#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate a tiny random-weight Cohere2Moe oracle for hipfire arch validation.

Reference: CohereLabs/BLS-Mini-Code-1.0 (`model_type = cohere2_moe`) via the
built-in transformers `Cohere2Moe*`. The arch is a Command-R-style decoder with
a **parallel block** (one input_layernorm shared by attention + MLP, both summed
into the residual), interleaved full-dim RoPE, standard RMSNorm, a dense SwiGLU
prefix layer (`first_k_dense_replace`), then 128-expert top-8 sigmoid MoE with
NO routing bias and `norm_topk_prob = False` (no top-k renormalization).

Tiny dims keep attention in the real regime (head_dim=128) and make every 2D
weight k%256==0 (HFQ4G256-quantizable): hidden=256, 4Q/2KV hd128, moe_inter=256,
dense_inter=512, 16 experts top-8 (matches the hardcoded `_k8` indexed-MoE GEMV
kernels), 2 layers (layer 0 dense, layer 1 MoE), vocab=512.

Built-in transformers stores routed experts PACKED (`mlp.experts.gate_up_proj`
[E,2I,H], `mlp.experts.down_proj` [E,H,I]); the hipfire loader wants SPLIT
(`mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`). We RE-SPLIT before
saving (numerically identical, just reorganized). Dense-layer MLP + the router
(`mlp.gate.weight`) pass through unchanged.

Outputs into <out>:
  model.safetensors + config.json  (SPLIT layout, flat arch fields → hipfire)
  oracle_hidden.hfhs               (HF per-layer post-residual, pre-final-norm)
  tokens.hfkldr                    (fixed token chunk for both dumpers)

Then: quantize model.safetensors → a .hfq (hipfire-quantize, cohere2moe arm —
see NEXT-STEPS.md), run examples/dump_cohere2moe_hidden_states, and compare with
scripts/compare_hidden_states.py.
"""
import argparse, json, struct, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

try:
    from transformers import Cohere2MoeConfig, Cohere2MoeForCausalLM
except Exception as e:  # pragma: no cover - import-time guard
    sys.exit(
        f"need a transformers with Cohere2Moe (model_type=cohere2_moe): {e}\n"
        "try: pip install -U 'transformers>=5.8'"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/cohere2moe-tiny")
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # FAITHFUL tiny dims (so the oracle reproduces real-model bugs the original
    # 2-layer/2:1 config masked): 8:1 GQA (16 q / 2 kv heads), multi-group FWHT
    # (hidden 512 = 2× G256), and the 1:3 full:sliding pattern with a NoPE
    # full-attention MoE layer (L4) plus a force-RoPE dense layer (L0).
    n_layers, hidden = 5, 512
    moe_inter, dense_inter = 256, 512
    n_exp, k_top = 16, 8  # top-8 matches the hardcoded `_k8` indexed-MoE kernels
    layer_types = [
        "full_attention",     # L0 dense → force_rope (prefix_dense pattern==1)
        "sliding_attention",  # L1 MoE  → RoPE
        "sliding_attention",  # L2 MoE  → RoPE
        "sliding_attention",  # L3 MoE  → RoPE
        "full_attention",     # L4 MoE  → NoPE (global, non-dense)
    ]

    # Build the config defensively: different transformers point-releases name a
    # few fields slightly differently. Start from the known BLS-Mini-Code fields.
    cfg_kwargs = dict(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=moe_inter,
        prefix_dense_intermediate_size=dense_inter,
        num_hidden_layers=n_layers,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=n_exp,
        num_experts_per_tok=k_top,
        first_k_dense_replace=1,
        prefix_dense_sliding_window_pattern=1,
        layer_types=layer_types,
        expert_selection_fn="sigmoid",
        norm_topk_prob=False,
        use_parallel_block=True,
        use_qk_norm=False,
        logit_scale=1.0,
        rms_norm_eps=1e-6,
        rope_theta=50_000.0,
        max_position_embeddings=512,
        sliding_window=4096,
        tie_word_embeddings=True,
    )
    cfg = Cohere2MoeConfig(**cfg_kwargs)
    model = Cohere2MoeForCausalLM(cfg).to(torch.float32).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Fixed token chunk (seeded).
    g = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(0, cfg.vocab_size, (args.n_ctx,), generator=g).tolist()
    print(f"tokens ({args.n_ctx}): {tokens}", flush=True)
    with open(out / "tokens.hfkldr", "wb") as f:
        f.write(b"HFKLDR\0\0")
        hdr = bytearray(24)
        struct.pack_into("<I", hdr, 4, args.n_ctx)
        struct.pack_into("<I", hdr, 12, 1)
        f.write(hdr)
        f.write(struct.pack(f"<{args.n_ctx}I", *tokens))

    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        res = model(input_ids, output_hidden_states=True)
    hs = res.hidden_states  # tuple: embeddings, then post-layer-0, post-layer-1, ...

    # Pre-final-norm residual (last layer's post-residual) via a pre-hook on norm.
    cap = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, i: cap.__setitem__("x", i[0].detach())
    )
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()
    post_last = cap["x"][0]

    with open(out / "oracle_hidden.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, args.n_ctx, hidden, 0))
        for k in range(n_layers):
            t = hs[k + 1][0] if k < n_layers - 1 else post_last
            assert tuple(t.shape) == (args.n_ctx, hidden), (k, t.shape)
            arr = t.float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
            print(
                f"  layer {k}: rms={float((arr.astype('float64')**2).mean()**0.5):.4f}",
                flush=True,
            )
    print(f"wrote {out}/oracle_hidden.hfhs", flush=True)

    # Re-split PACKED experts → SPLIT and save model.safetensors. Dense-layer MLP
    # (mlp.gate_proj/up_proj/down_proj), the router (mlp.gate.weight), attn,
    # norms, embed and lm_head pass through unchanged.
    sd = model.state_dict()
    # First, surface the expert layout (helps if a transformers release differs).
    expert_keys = sorted({k for k in sd if ".experts." in k})
    print(f"expert state_dict keys ({len(expert_keys)}):", flush=True)
    for kn in expert_keys[:8]:
        print(f"  {kn}  {tuple(sd[kn].shape)}", flush=True)

    split = {}
    for name, t in sd.items():
        t = t.detach().to(torch.float32).contiguous()
        if name.endswith("mlp.experts.gate_up_proj"):  # [E, 2I, H]
            pre = name[: -len("mlp.experts.gate_up_proj")]
            assert t.shape[1] == 2 * moe_inter, (name, t.shape)
            for e in range(n_exp):
                split[f"{pre}mlp.experts.{e}.gate_proj.weight"] = t[e][:moe_inter, :].contiguous()
                split[f"{pre}mlp.experts.{e}.up_proj.weight"] = t[e][moe_inter:, :].contiguous()
        elif name.endswith("mlp.experts.down_proj"):  # [E, H, I]
            pre = name[: -len("mlp.experts.down_proj")]
            for e in range(n_exp):
                split[f"{pre}mlp.experts.{e}.down_proj.weight"] = t[e].contiguous()
        else:
            # .clone() breaks tied-weight storage sharing (lm_head ↔
            # embed_tokens), which safetensors.save_file refuses to serialize.
            # The HFQ then carries lm_head + embed as distinct (identical) tensors.
            split[name] = t.clone()  # attn / norms / router / dense MLP / embed / lm_head

    save_file(split, str(out / "model.safetensors"))
    print(f"re-split → {len(split)} tensors", flush=True)

    # config.json with flat arch fields (real-ckpt convention hipfire parses).
    conf = dict(
        architectures=["Cohere2MoeForCausalLM"],
        model_type="cohere2_moe",
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=moe_inter,
        prefix_dense_intermediate_size=dense_inter,
        num_hidden_layers=n_layers,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=n_exp,
        num_experts_per_tok=k_top,
        first_k_dense_replace=1,
        prefix_dense_sliding_window_pattern=1,
        layer_types=layer_types,
        expert_selection_fn="sigmoid",
        norm_topk_prob=False,
        use_parallel_block=True,
        use_qk_norm=False,
        logit_scale=1.0,
        rms_norm_eps=1e-6,
        rope_theta=50_000.0,
        max_position_embeddings=512,
        sliding_window=4096,
        tie_word_embeddings=True,
    )
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    print(f"wrote {out}/config.json (flat arch fields)", flush=True)
    print(f"DONE → {out}", flush=True)


if __name__ == "__main__":
    main()
