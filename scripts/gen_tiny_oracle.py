#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generalized tiny random-weight ORACLE generator for hipfire arch ports.

Builds a dimension-faithful *tiny* model from the reference (HF transformers)
implementation and dumps per-layer hidden states, so a new hipfire arch port's
forward pass can be validated to cosine≈1 in seconds, with no real weights and
no GPU-hours. See docs/methodology/arch-port-validation.md for the full method.

Outputs into <out>:
  model.safetensors + config.json   (the layout hipfire ingests; re-split if needed)
  oracle_hidden.hfhs                 (HF per-layer post-residual, PRE final-norm)
  oracle_postattn.hfhs               (HF per-layer post-ATTENTION residual; for bisection)
  tokens.hfkldr                      (fixed token chunk both dumpers read)

HFHS = magic "HFHS\\0\\0\\0\\0" + <IIII>(n_layers,n_pos,hidden,0) + n_layers×[n_pos,hidden] f32.
HFKLDR = magic "HFKLDR\\0\\0" + 24B hdr (n_ctx@[4:8], n_chunk@[12:16]) + n_ctx u32 tokens @32.

──────────────────────────────────────────────────────────────────────────────
PORTING A NEW ARCH: edit only the `ARCH ADAPTATION` block below. Pick dims per
the pitfall checklist (docs/methodology/arch-port-validation.md):
  • keep REAL head_dim / rotary_dim (shrink #heads/#layers, not head_dim)
  • every quantized 2D weight: k % group_size == 0 (usually 256)
  • match any hardcoded kernel k_top (e.g. _k8 kernels → num_experts_per_tok=8)
  • routing/precision-sensitive tensors stay Q8 (handled at quantize time)
  • re-split packed→split if the loader/kernels expect split experts
──────────────────────────────────────────────────────────────────────────────
"""
import argparse, json, struct, sys
from pathlib import Path
import torch


# ===================== ARCH ADAPTATION (edit per arch) ======================
def build_model_and_config(seed: int):
    """Return (model, hf_config, n_layers, hidden). Tiny, real head_dim/rotary,
    dims divisible by the quant group size, top-k matching the kernel."""
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM  # ← arch import
    torch.manual_seed(seed)
    n_layers, inter, hidden = 2, 256, 256
    cfg = MiniMaxM2Config(
        vocab_size=512, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128,                       # REAL head_dim (not shrunk)
        num_local_experts=16, num_experts_per_tok=8,   # top-8 matches the _k8 kernels
        max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                         "partial_rotary_factor": 0.5},  # rotary_dim = 128*0.5 = 64
    )
    model = MiniMaxM2ForCausalLM(cfg).to(torch.float32).eval()
    # Optional: perturb routing bias so top-k selection is non-trivial.
    with torch.no_grad():
        for layer in model.model.layers:
            b = layer.mlp.e_score_correction_bias
            b.copy_(torch.randn_like(b) * 0.1)
    return model, cfg, n_layers, hidden


def post_attn_modules(model):
    """Modules whose INPUT is the post-attention residual (pre-MoE/FFN), one per
    layer, for the bisection oracle. Typically the post_attention_layernorm."""
    return [layer.post_attention_layernorm for layer in model.model.layers]


def final_norm_module(model):
    """The final norm; its INPUT is the post-last-layer pre-norm hidden state."""
    return model.model.norm


def resplit_state_dict(sd, cfg):
    """Reorganize the saved state_dict to the layout hipfire ingests. Return a
    new dict. Identity if the reference already matches. MiniMax: HF packs
    experts (gate_up_proj[E,2I,H]/down_proj[E,H,I]) but the real ckpt + hipfire
    want split block_sparse_moe.experts.E.{w1,w2,w3}."""
    inter = cfg.intermediate_size
    out = {}
    for name, t in sd.items():
        t = t.detach().to(torch.float32).contiguous()
        if name.endswith("mlp.experts.gate_up_proj"):
            pre = name[:-len("mlp.experts.gate_up_proj")]
            for e in range(cfg.num_local_experts):
                out[f"{pre}block_sparse_moe.experts.{e}.w1.weight"] = t[e][:inter, :].contiguous()
                out[f"{pre}block_sparse_moe.experts.{e}.w3.weight"] = t[e][inter:, :].contiguous()
        elif name.endswith("mlp.experts.down_proj"):
            pre = name[:-len("mlp.experts.down_proj")]
            for e in range(cfg.num_local_experts):
                out[f"{pre}block_sparse_moe.experts.{e}.w2.weight"] = t[e].contiguous()
        elif name.endswith("mlp.gate.weight"):
            out[name.replace("mlp.gate.weight", "block_sparse_moe.gate.weight")] = t
        elif name.endswith("mlp.e_score_correction_bias"):
            out[name.replace("mlp.e_score_correction_bias",
                             "block_sparse_moe.e_score_correction_bias")] = t
        else:
            out[name] = t
    return out


def flat_config_fields(cfg):
    """Flat config fields hipfire's `*Config::from_hfq` parses (the real-ckpt
    convention), in case the HF config nests them differently."""
    return dict(
        architectures=["MiniMaxM2ForCausalLM"], model_type="minimax_m2",
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=128, num_local_experts=cfg.num_local_experts,
        num_experts_per_tok=cfg.num_experts_per_tok, rotary_dim=64, rope_theta=5_000_000.0,
        rms_norm_eps=1e-6, use_qk_norm=True, use_routing_bias=True, scoring_func="sigmoid",
        max_position_embeddings=512, num_mtp_modules=0, tie_word_embeddings=False,
    )
# =================== END ARCH ADAPTATION ====================================


def write_hfhs(path, tensors, n_layers, n_ctx, hidden):
    with open(path, "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, n_ctx, hidden, 0))
        for k in range(n_layers):
            arr = tensors[k].float().cpu().contiguous().numpy()
            assert arr.shape == (n_ctx, hidden), (k, arr.shape)
            f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/minimax-tiny")
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg, n_layers, hidden = build_model_and_config(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Fixed token chunk.
    g = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(0, cfg.vocab_size, (args.n_ctx,), generator=g).tolist()
    with open(out / "tokens.hfkldr", "wb") as f:
        f.write(b"HFKLDR\0\0"); hdr = bytearray(24)
        struct.pack_into("<I", hdr, 4, args.n_ctx); struct.pack_into("<I", hdr, 12, 1)
        f.write(hdr); f.write(struct.pack(f"<{args.n_ctx}I", *tokens))
    print(f"tokens: {tokens}", flush=True)

    input_ids = torch.tensor([tokens], dtype=torch.long)

    # Per-layer POST-residual (pre final-norm): hs[k+1] for k<last, hook for last.
    with torch.no_grad():
        hs = model(input_ids, output_hidden_states=True).hidden_states
    cap = {}
    h = final_norm_module(model).register_forward_pre_hook(
        lambda m, i: cap.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()
    layer_out = [hs[k + 1][0] if k < n_layers - 1 else cap["x"][0] for k in range(n_layers)]
    write_hfhs(out / "oracle_hidden.hfhs", layer_out, n_layers, args.n_ctx, hidden)
    for k in range(n_layers):
        rms = float((layer_out[k].float() ** 2).mean() ** 0.5)
        print(f"  layer {k}: post-residual rms={rms:.4f}", flush=True)

    # Per-layer POST-ATTENTION residual (input to each post_attn norm) for bisection.
    pa, handles = {}, []
    for li, mod in enumerate(post_attn_modules(model)):
        handles.append(mod.register_forward_pre_hook(
            (lambda idx: (lambda m, i: pa.__setitem__(idx, i[0].detach())))(li)))
    with torch.no_grad():
        _ = model(input_ids)
    for hd in handles:
        hd.remove()
    write_hfhs(out / "oracle_postattn.hfhs", [pa[k][0] for k in range(n_layers)],
               n_layers, args.n_ctx, hidden)

    # Save model.safetensors in the hipfire-ingest layout + flat config.json.
    from safetensors.torch import save_file
    save_file(resplit_state_dict(model.state_dict(), cfg), str(out / "model.safetensors"))
    conf = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    conf.update(flat_config_fields(cfg))
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    print(f"DONE → {out}", flush=True)


if __name__ == "__main__":
    main()
