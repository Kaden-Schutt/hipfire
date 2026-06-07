#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Tiny random-weight Gemma4 EAGLE-drafter oracle for hipfire arch-22 validation.

Generates a matched pair:
  * tiny TARGET  (model_type gemma4_unified, arch_id 13) — small dims, hybrid
    sliding/full attention, K=V sharing on the full layer.
  * tiny ASSISTANT/DRAFTER (model_type gemma4_unified_assistant, arch_id 22) —
    hidden 128, 4 layers [sliding,sliding,sliding,full], backbone_hidden =
    target hidden, num_kv_shared_layers = 4, the two projections, tied embed.

Runs the HF reference EXACTLY as
generation/candidate_generator.py::SinglePositionMultiTokenCandidateGenerator
does for the Gemma 4 MTP head:
  1. Run the target over a short token seq with output_hidden_states=True and
     return_shared_kv_states=True → shared_kv_states (last sliding + last full
     slot, already k_norm'd + RoPE'd + v_norm'd) and last hidden state.
  2. Drafter autoregressive loop at CONSTANT position_ids = seq_len-1:
       inputs_embeds = cat[ target_embed(prev_tok)·√bb  ‖  hidden ]
       x = pre_projection(inputs_embeds); backbone layers; lm_head argmax;
       post_projection feeds the next step's hidden half.
  Dumps per step: prev_token, pre_proj input (7680) + output (1024), final
  normed hidden (1024), logits (vocab), argmax, post_projection output (bb).

The TARGET is written in the SAME format gen_tiny_gemma4.py uses (so the hipfire
arch-13 loader + quantizer read it). The ASSISTANT is written with the FLAT
model.* names + the two top-level projections, and a config.json whose
model_type = gemma4_unified_assistant so the arch-22 converter picks it up.

Outputs into <out>/{target,assistant}/ + <out>/drafter_oracle.npz.
Run with the transformers-MAIN venv (cohere-gen-venv on hiptrx).
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

try:
    from transformers import (
        Gemma4UnifiedTextConfig,
        Gemma4UnifiedForCausalLM,
        Gemma4UnifiedAssistantConfig,
        Gemma4UnifiedAssistantForCausalLM,
    )
except Exception as e:  # pragma: no cover
    sys.exit(f"need transformers-main with gemma4_unified + gemma4_unified_assistant: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/kaden/gemma4-drafter-tiny")
    p.add_argument("--n-ctx", type=int, default=12)
    p.add_argument("--n-draft", type=int, default=4, help="drafter steps to run")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_target(seed):
    """Tiny gemma4_unified target. sliding_window >= n_ctx so SWA is a no-op
    (isolates the drafter math from the window-edge flip). Dims kept modest but
    large enough that Q8-KV quantization noise doesn't flip near-tied logits in
    the random-weight head (a pure tiny-scale degeneracy, not a math error)."""
    hidden = 768
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                           "partial_rotary_factor": 0.25},
    }
    cfg_kwargs = dict(
        vocab_size=4096, hidden_size=hidden, intermediate_size=1024,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=1, head_dim=256, global_head_dim=512,
        attention_k_eq_v=True, hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6, sliding_window=4096, final_logit_softcapping=30.0,
        layer_types=layer_types, rope_parameters=rope_parameters,
        tie_word_embeddings=True, attention_bias=False, num_kv_shared_layers=0,
        use_double_wide_mlp=False, max_position_embeddings=512,
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
    )
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(**cfg_kwargs)
    model = Gemma4UnifiedForCausalLM(cfg).to(torch.float32).eval()
    return model, cfg, cfg_kwargs, hidden


def build_assistant(seed, backbone_hidden, vocab):
    """Tiny gemma4_unified_assistant drafter. hidden 128, 4 layers
    [sliding,sliding,sliding,full], head_dim/n_kv MATCH the target (the drafter
    attends the target's shared K/V), num_kv_shared_layers=4 (all shared)."""
    layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                           "partial_rotary_factor": 0.25},
    }
    text_kwargs = dict(
        vocab_size=vocab, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=1, head_dim=256, global_head_dim=512,
        attention_k_eq_v=True, hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6, sliding_window=4096, final_logit_softcapping=None,
        layer_types=layer_types, rope_parameters=rope_parameters,
        tie_word_embeddings=True, attention_bias=False, num_kv_shared_layers=4,
        use_double_wide_mlp=False, max_position_embeddings=512,
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
    )
    torch.manual_seed(seed + 100)
    text_cfg = Gemma4UnifiedTextConfig(**text_kwargs)
    acfg = Gemma4UnifiedAssistantConfig(
        text_config=text_cfg, backbone_hidden_size=backbone_hidden,
        num_centroids=2048, centroid_intermediate_top_k=32,
        use_ordered_embeddings=False, tie_word_embeddings=True,
    )
    model = Gemma4UnifiedAssistantForCausalLM(acfg).to(torch.float32).eval()
    return model, acfg, text_kwargs


def write_target_dir(out, model, cfg_kwargs, hidden, n_layers, tokens, hs_post_last, hs):
    out.mkdir(parents=True, exist_ok=True)
    # tokens.hfkldr
    with open(out / "tokens.hfkldr", "wb") as f:
        f.write(b"HFKLDR\0\0")
        hdr = bytearray(24)
        struct.pack_into("<I", hdr, 4, len(tokens))
        struct.pack_into("<I", hdr, 12, 1)
        f.write(hdr)
        f.write(struct.pack(f"<{len(tokens)}I", *tokens))
    # oracle_hidden.hfhs (per-layer post-residual + final post-norm-input)
    with open(out / "oracle_hidden.hfhs", "wb") as f:
        f.write(b"HFHS\0\0\0\0")
        f.write(struct.pack("<IIII", n_layers, len(tokens), hidden, 0))
        for k in range(n_layers):
            t = hs[k + 1][0] if k < n_layers - 1 else hs_post_last
            arr = t.float().cpu().contiguous().numpy()
            f.write(arr.tobytes())
    # safetensors with model.language_model.* prefix, drop tied lm_head
    sd = model.state_dict()
    out_sd = {}
    for name, t in sd.items():
        if name.startswith("lm_head."):
            continue
        t = t.detach().to(torch.float32).contiguous().clone()
        if name.startswith("model."):
            out_sd["model.language_model." + name[len("model."):]] = t
        else:
            out_sd[name] = t
    save_file(out_sd, str(out / "model.safetensors"))
    text_cfg = dict(model_type="gemma4_unified_text", **cfg_kwargs)
    conf = dict(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        model_type="gemma4_unified",
        num_hidden_layers=n_layers,
        text_config=text_cfg,
    )
    (out / "config.json").write_text(json.dumps(conf, indent=2))


def write_assistant_dir(out, model, text_kwargs, backbone_hidden):
    out.mkdir(parents=True, exist_ok=True)
    # FLAT model.* names + the two top-level projections; drop tied lm_head.
    sd = model.state_dict()
    out_sd = {}
    for name, t in sd.items():
        if name.startswith("lm_head."):
            continue
        if name.startswith("masked_embedding."):
            continue  # use_ordered_embeddings=False → not used; skip
        t = t.detach().to(torch.float32).contiguous().clone()
        out_sd[name] = t
    save_file(out_sd, str(out / "model.safetensors"))
    text_cfg = dict(model_type="gemma4_unified_text", **text_kwargs)
    conf = dict(
        architectures=["Gemma4UnifiedAssistantForCausalLM"],
        model_type="gemma4_unified_assistant",
        backbone_hidden_size=backbone_hidden,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        use_ordered_embeddings=False,
        tie_word_embeddings=True,
        num_hidden_layers=text_kwargs["num_hidden_layers"],
        text_config=text_cfg,
    )
    (out / "config.json").write_text(json.dumps(conf, indent=2))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build models ──
    target, tcfg, tcfg_kwargs, hidden = build_target(args.seed)
    assistant, acfg, text_kwargs = build_assistant(args.seed, hidden, tcfg.vocab_size)
    backbone_hidden = hidden

    # ── Tokens ──
    g = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(0, tcfg.vocab_size, (args.n_ctx,), generator=g).tolist()
    print(f"tokens ({args.n_ctx}): {tokens}", flush=True)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    # ── Run target: hidden states + shared KV states ──
    cap = {}
    h = target.model.norm.register_forward_pre_hook(
        lambda m, i: cap.__setitem__("x", i[0].detach())
    )
    with torch.no_grad():
        res = target(input_ids, output_hidden_states=True, return_shared_kv_states=True)
    h.remove()
    hs = res.hidden_states
    hs_post_last = cap["x"][0]
    shared_kv_states = res.shared_kv_states
    assert shared_kv_states is not None, "target did not return shared_kv_states"
    print("shared_kv keys:", list(shared_kv_states.keys()), flush=True)
    for k, (kk, vv) in shared_kv_states.items():
        print(f"  {k}: K{tuple(kk.shape)} V{tuple(vv.shape)}", flush=True)

    # last hidden state of the last token (= hidden_states[-1], post final norm)
    last_hidden_state = res.hidden_states[-1][:, -1:].clone()  # [1,1,hidden]
    print("last_hidden_state shape:", tuple(last_hidden_state.shape), flush=True)

    # ── Write model dirs ──
    write_target_dir(out / "target", target, tcfg_kwargs, hidden, tcfg.num_hidden_layers,
                     tokens, hs_post_last, hs)
    write_assistant_dir(out / "assistant", assistant, text_kwargs, backbone_hidden)
    print(f"wrote {out}/target and {out}/assistant", flush=True)

    # ── Drafter loop (mirrors SinglePositionMultiTokenCandidateGenerator) ──
    target_embed = target.get_input_embeddings()  # ScaledWordEmbedding (·√hidden)
    position_ids = torch.tensor([[args.n_ctx - 1]], dtype=torch.long)
    last_token_id = input_ids[:, -1:]
    cur_hidden = last_hidden_state  # [1,1,hidden]

    steps = []
    for step in range(args.n_draft):
        last_token_embedding = target_embed(last_token_id)            # [1,1,bb], ·√bb baked in
        inputs_embeds = torch.cat([last_token_embedding, cur_hidden], dim=-1)  # [1,1,2bb]
        with torch.no_grad():
            outputs = assistant(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                position_ids=position_ids,
                shared_kv_states=shared_kv_states,
                use_cache=False,
                output_hidden_states=True,
            )
        logits = outputs.logits[0, -1].detach().float().cpu().numpy()          # [vocab]
        argmax = int(logits.argmax())
        post_proj = outputs.last_hidden_state[0, -1].detach().float().cpu().numpy()  # [bb]
        # pre_projection in/out for cross-check
        pre_in = inputs_embeds[0, -1].detach().float().cpu().numpy()           # [2bb]
        with torch.no_grad():
            pre_out = assistant.pre_projection(inputs_embeds)[0, -1].float().cpu().numpy()  # [hidden]
        # final normed hidden = backbone last_hidden_state (output of model.norm)
        normed = outputs.hidden_states[-1][0, -1].detach().float().cpu().numpy()  # [hidden]
        steps.append(dict(
            prev_token=int(last_token_id.item()),
            pre_in=pre_in, pre_out=pre_out, normed=normed,
            logits=logits, argmax=argmax, post_proj=post_proj,
        ))
        print(f"step {step}: prev={int(last_token_id.item())} argmax={argmax} "
              f"normed_rms={float((normed.astype('float64')**2).mean()**0.5):.4f} "
              f"post_rms={float((post_proj.astype('float64')**2).mean()**0.5):.4f}", flush=True)
        # feed next
        last_token_id = torch.tensor([[argmax]], dtype=torch.long)
        cur_hidden = outputs.last_hidden_state.clone()  # [1,1,bb]

    # ── Dump npz oracle ──
    npz = dict(
        n_ctx=np.int64(args.n_ctx),
        n_draft=np.int64(args.n_draft),
        hidden=np.int64(text_kwargs["hidden_size"]),
        backbone_hidden=np.int64(backbone_hidden),
        vocab=np.int64(tcfg.vocab_size),
        tokens=np.array(tokens, dtype=np.int64),
        query_pos=np.int64(args.n_ctx - 1),
        # the step-0 hidden half = target last_hidden_state (post final norm)
        target_last_hidden=last_hidden_state[0, -1].detach().float().cpu().numpy(),
    )
    for i, s in enumerate(steps):
        npz[f"s{i}_prev_token"] = np.int64(s["prev_token"])
        npz[f"s{i}_pre_in"] = s["pre_in"].astype(np.float32)
        npz[f"s{i}_pre_out"] = s["pre_out"].astype(np.float32)
        npz[f"s{i}_normed"] = s["normed"].astype(np.float32)
        npz[f"s{i}_logits"] = s["logits"].astype(np.float32)
        npz[f"s{i}_argmax"] = np.int64(s["argmax"])
        npz[f"s{i}_post_proj"] = s["post_proj"].astype(np.float32)
    np.savez(out / "drafter_oracle.npz", **npz)
    print(f"wrote {out}/drafter_oracle.npz ({args.n_draft} steps)", flush=True)

    # ── Also dump a flat little-endian binary the Rust harness reads ──
    #   magic "HFDRAFT\0" | u32 n_ctx | u32 n_draft | u32 hidden | u32 backbone |
    #   u32 vocab | u32 query_pos | [n_ctx u32 tokens] |
    #   [backbone f32 target_last_hidden] |
    #   per step: u32 prev_token | u32 argmax |
    #             [2*backbone f32 pre_in] | [hidden f32 pre_out] |
    #             [hidden f32 normed] | [backbone f32 post_proj] | [vocab f32 logits]
    h = text_kwargs["hidden_size"]
    bb = backbone_hidden
    V = int(tcfg.vocab_size)
    # shared K/V (last sliding + last full slot), post-k_norm + RoPE + v_norm.
    # HF layout: [1, n_kv, seq, head_dim]; we dump row-major [seq, n_kv*head_dim]
    # to match hipfire's pre-Q8 K/V vector layout (per position, n_kv heads × hd).
    ks, vs = shared_kv_states["sliding_attention"]
    kf, vf = shared_kv_states["full_attention"]
    n_kv_s, hd_s = ks.shape[1], ks.shape[3]
    n_kv_f, hd_f = kf.shape[1], kf.shape[3]

    def kv_seq_major(t):  # [1,n_kv,seq,hd] -> [seq, n_kv*hd] float32
        return t[0].permute(1, 0, 2).reshape(t.shape[2], -1).detach().float().cpu().numpy()

    ks_m = kv_seq_major(ks); vs_m = kv_seq_major(vs)
    kf_m = kv_seq_major(kf); vf_m = kv_seq_major(vf)
    with open(out / "drafter_oracle.bin", "wb") as f:
        f.write(b"HFDRAFT\0")
        f.write(struct.pack("<IIIIII", args.n_ctx, args.n_draft, h, bb, V, args.n_ctx - 1))
        f.write(struct.pack("<IIII", n_kv_s, hd_s, n_kv_f, hd_f))
        f.write(struct.pack(f"<{args.n_ctx}I", *tokens))
        f.write(npz["target_last_hidden"].astype("<f4").tobytes())
        # shared K/V blocks ([seq, n_kv*hd] each)
        f.write(ks_m.astype("<f4").tobytes())
        f.write(vs_m.astype("<f4").tobytes())
        f.write(kf_m.astype("<f4").tobytes())
        f.write(vf_m.astype("<f4").tobytes())
        for s in steps:
            f.write(struct.pack("<II", s["prev_token"], s["argmax"]))
            f.write(s["pre_in"].astype("<f4").tobytes())
            f.write(s["pre_out"].astype("<f4").tobytes())
            f.write(s["normed"].astype("<f4").tobytes())
            f.write(s["post_proj"].astype("<f4").tobytes())
            f.write(s["logits"].astype("<f4").tobytes())
    print(f"wrote {out}/drafter_oracle.bin", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
