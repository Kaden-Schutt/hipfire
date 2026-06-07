#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""CPU reference checker for dense Qwen3.5 9B native MTP.

This intentionally validates the smallest v1 contract: a single-token prompt
dump from `mtp_probe` against the original HuggingFace safetensors MTP module.
Multi-token prompts need a target-hidden trace for every prompt token to rebuild
the MTP KV cache, so they are rejected instead of checked approximately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def text_config(snapshot: Path) -> dict:
    cfg = json.loads((snapshot / "config.json").read_text())
    return cfg.get("text_config", cfg)


def load_tensor(snapshot: Path, name: str) -> torch.Tensor:
    for path in sorted(snapshot.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).to(torch.float32)
    raise KeyError(f"tensor not found in safetensors snapshot: {name}")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def topk_ids(logits: torch.Tensor, k: int) -> list[int]:
    return torch.topk(logits, k).indices.cpu().tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path, help="HF safetensors snapshot directory")
    ap.add_argument("dump_prefix", type=Path, help="HIPFIRE_MTP_DUMP_PREFIX used by mtp_probe")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--atol", type=float, default=0.1)
    ap.add_argument(
        "--use-norm-hidden",
        action="store_true",
        help="Use <prefix>.target_hidden_norm.f32 instead of raw target_hidden.f32",
    )
    args = ap.parse_args()

    cfg = text_config(args.snapshot)
    dim = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", dim // n_heads))
    hidden_dim = int(cfg["intermediate_size"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    prompt_tokens = [
        int(x)
        for x in (args.dump_prefix.with_suffix(".prompt_tokens.txt")).read_text().split()
        if x
    ]
    if len(prompt_tokens) != 1:
        raise SystemExit(
            "mtp_cpu_ref.py currently checks only single-token prompt dumps; "
            f"got {len(prompt_tokens)} tokens. Re-run mtp_probe with a one-token prompt."
        )
    token = prompt_tokens[-1]

    hidden_name = ".target_hidden_norm.f32" if args.use_norm_hidden else ".target_hidden.f32"
    target_hidden = torch.from_file(
        str(args.dump_prefix.with_suffix(hidden_name)), dtype=torch.float32, size=dim
    ).clone()
    hipfire_logits = torch.from_file(
        str(args.dump_prefix.with_suffix(".mtp_logits.f32")),
        dtype=torch.float32,
        size=int(cfg["vocab_size"]),
    ).clone()

    names = {
        "embed": "model.language_model.embed_tokens.weight",
        "lm_head": "lm_head.weight",
        "pre_embed": "mtp.pre_fc_norm_embedding.weight",
        "pre_hidden": "mtp.pre_fc_norm_hidden.weight",
        "fc": "mtp.fc.weight",
        "attn_norm": "mtp.layers.0.input_layernorm.weight",
        "q_proj": "mtp.layers.0.self_attn.q_proj.weight",
        "v_proj": "mtp.layers.0.self_attn.v_proj.weight",
        "o_proj": "mtp.layers.0.self_attn.o_proj.weight",
        "ffn_norm": "mtp.layers.0.post_attention_layernorm.weight",
        "gate_proj": "mtp.layers.0.mlp.gate_proj.weight",
        "up_proj": "mtp.layers.0.mlp.up_proj.weight",
        "down_proj": "mtp.layers.0.mlp.down_proj.weight",
        "mtp_norm": "mtp.norm.weight",
    }
    w = {key: load_tensor(args.snapshot, name) for key, name in names.items()}

    with torch.inference_mode():
        embed = w["embed"][token]
        fc_in = torch.cat(
            [
                rmsnorm(embed, w["pre_embed"], eps),
                rmsnorm(target_hidden, w["pre_hidden"], eps),
            ]
        )
        x = torch.mv(w["fc"], fc_in)

        tmp = rmsnorm(x, w["attn_norm"], eps)
        q_full = torch.mv(w["q_proj"], tmp).reshape(n_heads, 2, head_dim)
        gate_vec = q_full[:, 1, :].reshape(n_heads * head_dim)
        v = torch.mv(w["v_proj"], tmp).reshape(n_kv_heads, head_dim)
        v = v.repeat_interleave(n_heads // n_kv_heads, dim=0).reshape(n_heads * head_dim)
        attn_out = v * torch.sigmoid(gate_vec)
        x = x + torch.mv(w["o_proj"], attn_out)

        tmp = rmsnorm(x, w["ffn_norm"], eps)
        gate = torch.mv(w["gate_proj"], tmp)
        up = torch.mv(w["up_proj"], tmp)
        x = x + torch.mv(w["down_proj"], torch.nn.functional.silu(gate) * up)

        mtp_hidden = rmsnorm(x, w["mtp_norm"], eps)
        cpu_logits = torch.mv(w["lm_head"], mtp_hidden)

    cpu_top = topk_ids(cpu_logits, args.topk)
    hip_top = topk_ids(hipfire_logits, args.topk)
    overlap = len(set(cpu_top) & set(hip_top))
    cpu_argmax = cpu_top[0]
    hip_argmax = hip_top[0]
    max_abs_top = torch.max(torch.abs(cpu_logits[hip_top] - hipfire_logits[hip_top])).item()

    print(f"cpu_argmax={cpu_argmax}")
    print(f"hipfire_argmax={hip_argmax}")
    print(f"top{args.topk}_overlap={overlap}")
    print(f"cpu_top{args.topk}={cpu_top}")
    print(f"hipfire_top{args.topk}={hip_top}")
    print(f"max_abs_delta_on_hipfire_top{args.topk}={max_abs_top:.6g}")

    if cpu_argmax != hip_argmax or cpu_top != hip_top or max_abs_top > args.atol:
        print("status=FAIL")
        return 1
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
