#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate a tiny random-weight Gemma4 (text) oracle for hipfire arch validation.

Reference: google/gemma-4-12B-it text decoder (model_type gemma4_unified /
gemma4_unified_text) via transformers `Gemma4UnifiedForCausalLM`. Faithful tiny
dims reproduce the real model's distinctive quirks so the oracle catches bugs the
generic path would mask:
  * DUAL head_dim: sliding hd=256 (2:1 GQA), full hd=512 with num_global_kv=1 (MQA)
  * DUAL RoPE: sliding theta=1e4 full-rotate; full theta=1e6 proportional-0.25
  * 4 sandwich norms/layer (plain x*w, NO +1), QK-norm, attention_k_eq_v (full)
  * GeGLU gelu_pytorch_tanh, embed x sqrt(hidden), final_logit_softcapping 30
  * SWA: sliding_window=8 < n_ctx=16 → actively TESTS the window cap on sliding layers
  * layer_types = 5 sliding + 1 full (last) — tests both attention kinds incl NoPE-partial

Tensors are renamed to the unified `model.language_model.*` prefix (matching the
real ckpt + the hipfire loader/converter). lm_head is tied (aliased at load).

Outputs into <out>: model.safetensors + config.json (gemma4_unified w/ text_config)
+ oracle_hidden.hfhs + tokens.hfkldr.
"""
import argparse, json, struct, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

try:
    from transformers import Gemma4UnifiedTextConfig, Gemma4UnifiedForCausalLM
except Exception as e:  # pragma: no cover
    sys.exit(f"need transformers with gemma4_unified (main): {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/gemma4-tiny")
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sliding-window", type=int, default=8,
                   help="small (<n_ctx) exercises SWA; large (>=n_ctx) makes it a no-op")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    n_layers, hidden = 6, 512
    inter = 512
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                           "partial_rotary_factor": 0.25},
    }
    cfg_kwargs = dict(
        vocab_size=512, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=1,  # full layers = MQA (1 KV head)
        head_dim=256, global_head_dim=512, attention_k_eq_v=True,
        hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6,
        sliding_window=args.sliding_window,  # < n_ctx exercises SWA windowing
        final_logit_softcapping=30.0, layer_types=layer_types,
        rope_parameters=rope_parameters, tie_word_embeddings=True,
        attention_bias=False, num_kv_shared_layers=0, use_double_wide_mlp=False,
        max_position_embeddings=512, bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
    )
    cfg = Gemma4UnifiedTextConfig(**cfg_kwargs)
    model = Gemma4UnifiedForCausalLM(cfg).to(torch.float32).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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
    hs = res.hidden_states

    cap = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, i: cap.__setitem__("x", i[0].detach())
    )
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()
    post_last = cap["x"][0]

    logits = res.logits[0, -1].float()
    top = torch.topk(logits, 5)
    print("HF next-token top-5:", [int(i) for i in top.indices], flush=True)

    with open(out / "oracle_hidden.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, args.n_ctx, hidden, 0))
        for k in range(n_layers):
            t = hs[k + 1][0] if k < n_layers - 1 else post_last
            assert tuple(t.shape) == (args.n_ctx, hidden), (k, t.shape)
            arr = t.float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
            print(f"  layer {k}: rms={float((arr.astype('float64')**2).mean()**0.5):.4f}", flush=True)
    print(f"wrote {out}/oracle_hidden.hfhs", flush=True)

    # Re-prefix text tensors to the unified `model.language_model.*` layout and
    # clone to break tied lm_head/embed storage sharing. lm_head is tied →
    # dropped (the hipfire loader aliases embed_tokens for the head).
    sd = model.state_dict()
    out_sd = {}
    for name, t in sd.items():
        if name.startswith("lm_head."):
            continue  # tied; loader aliases embed
        t = t.detach().to(torch.float32).contiguous().clone()
        if name.startswith("model."):
            out_sd["model.language_model." + name[len("model."):]] = t
        else:
            out_sd[name] = t
    save_file(out_sd, str(out / "model.safetensors"))
    print(f"saved {len(out_sd)} tensors (model.language_model.* layout)", flush=True)

    text_cfg = dict(
        model_type="gemma4_unified_text", **cfg_kwargs,
    )
    conf = dict(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        model_type="gemma4_unified",
        num_hidden_layers=n_layers,
        text_config=text_cfg,
    )
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    print(f"wrote {out}/config.json (gemma4_unified + text_config)", flush=True)
    print(f"DONE → {out}", flush=True)


if __name__ == "__main__":
    main()
