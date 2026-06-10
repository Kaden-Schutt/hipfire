#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
dflash_diag_iter1_tau.py — Hermes cheap diagnostic #2 (2026-04-19).

Eval τ of a trained draft with max_cycles=1 (one decode cycle only).

Rationale: at iter 1, past_kv_draft is EMPTY, so the draft's forward sees
  k_cat = [k_ctx (full prompt features), k_noise (current block)]
which is STRUCTURALLY IDENTICAL to our training forward pass for a single-
anchor case. Iter 2+ at inference has past_kv accumulated with cropped ctx;
training never exercised this pattern (we always use_cache=False).

If τ_iter1 much larger than τ_aggregate (e.g. τ_iter1=3 vs τ_aggregate=0.1),
Hypothesis B (KV-cache distribution mismatch) is effectively proven: draft
learned to speculate at iter-1 distribution, fails at iter 2+.

If τ_iter1 also near zero, Hypothesis B is refuted; look elsewhere (mask,
config, γ).

Run on MI300X:
  python3 scripts/dflash_diag_iter1_tau.py \
      --target-repo Qwen/Qwen3.5-4B \
      --draft-dir /root/dflash_4b_agentic \
      --prompt-file /root/hermes_test_prompt.txt \
      --max-cycles 1   # and again with --max-cycles 50 for aggregate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".dflash-reference"))

from dflash.model import DFlashDraftModel, extract_context_feature, sample  # type: ignore[import-not-found]  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--draft-dir", required=True,
                   help="Local HF-format draft dir (config.json + model.safetensors).")
    p.add_argument("--prompt-file", default=None,
                   help="Path to a txt file with a prompt. Default: a simple tool-call prompt.")
    p.add_argument("--max-cycles", type=int, default=1,
                   help="Cap on number of decode cycles. 1 isolates iter-1 τ; set high for aggregate.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-chatml", action="store_true",
                   help="Skip ChatML wrapping. Default: wrap (matches --chatml in dflash_spec_demo).")
    return p.parse_args()


DEFAULT_PROMPT = (
    "You are an AI assistant. Call the `get_weather` tool to find today's "
    "weather in Tokyo, then summarize the result in two sentences."
)


@torch.inference_mode()
def spec_generate_capped(draft, target, tokenizer, input_ids, max_new, max_cycles, device):
    """Copy of reference spec_generate but with a hard cap on decode cycles.

    Returns (output_ids, acceptance_lengths list).
    """
    from transformers import DynamicCache

    draft.eval()
    num_input = input_ids.shape[1]
    max_length = num_input + max_new
    B = draft.block_size
    output_ids = torch.full(
        (1, max_length + B), draft.mask_token_id, dtype=torch.long, device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    # Qwen3.5 target has linear_attention layers — DynamicCache must be
    # initialized WITH the target's config so it creates the hybrid cache
    # layout. Plain DynamicCache() errors on has_previous_state().
    pkv_t = DynamicCache(config=target.config)
    pkv_d = DynamicCache()

    out = target(
        input_ids, position_ids=position_ids[:, :num_input],
        past_key_values=pkv_t, use_cache=True,
        logits_to_keep=1, output_hidden_states=True,
    )
    output_ids[:, :num_input] = input_ids
    output_ids[:, num_input:num_input + 1] = sample(out.logits, 0.0)
    target_hidden = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    accept_lengths = []
    start = num_input
    cycles = 0
    while start < max_length and cycles < max_cycles:
        cycles += 1
        block_out = output_ids[:, start:start + B].clone()
        block_pos = position_ids[:, start:start + B]
        noise_emb = target.model.embed_tokens(block_out)
        dh = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=position_ids[:, pkv_d.get_seq_length():start + B],
            past_key_values=pkv_d, use_cache=True,
        )
        draft_logits = target.lm_head(dh[:, -B + 1:, :])
        pkv_d.crop(start)
        block_out[:, 1:] = sample(draft_logits, 0.0)
        out = target(
            block_out, position_ids=block_pos,
            past_key_values=pkv_t, use_cache=True,
            output_hidden_states=True,
        )
        post = sample(out.logits, 0.0)
        accept = (block_out[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start:start + accept + 1] = block_out[:, :accept + 1]
        output_ids[:, start + accept + 1] = post[:, accept]
        start += accept + 1
        pkv_t.crop(start)
        target_hidden = extract_context_feature(out.hidden_states, draft.target_layer_ids)[:, :accept + 1, :]
        accept_lengths.append(accept + 1)
    return output_ids, accept_lengths


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    print(f"[target] loading {args.target_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_repo)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_repo, torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"[draft] loading from {args.draft_dir}...")
    # Skip AutoConfig (stale auto_map with "module.submodule.Class" triples
    # chokes transformers 5.x's class_reference.split parser). Load the
    # config.json manually into a fresh Qwen3Config — DFlashDraftModel's
    # config_class is Qwen3Config anyway.
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    import json as _json
    cfg_dict = _json.loads((Path(args.draft_dir) / "config.json").read_text())
    dflash_cfg = cfg_dict.pop("dflash_config", {"mask_token_id": 248070, "target_layer_ids": []})
    block_size = cfg_dict.pop("block_size", 16)
    num_target_layers = cfg_dict.pop("num_target_layers", 32)
    # Strip attrs Qwen3Config doesn't understand
    for k in ("auto_map", "architectures", "dflash_config", "id2label", "label2id",
              "problem_type", "num_target_layers", "block_size"):
        cfg_dict.pop(k, None)
    draft_cfg = Qwen3Config(**{k: v for k, v in cfg_dict.items() if k not in ("transformers_version",)})
    draft_cfg.dflash_config = dflash_cfg
    draft_cfg.block_size = block_size
    draft_cfg.num_target_layers = num_target_layers
    # Guard DFlashDraftModel's layer_types access — force all full_attention
    draft_cfg.layer_types = ["full_attention"] * draft_cfg.num_hidden_layers
    draft = DFlashDraftModel(draft_cfg).to(device=device, dtype=dtype)
    sf = next(p for p in Path(args.draft_dir).glob("*.safetensors"))
    sd = load_file(str(sf))
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    print(f"[draft]   missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[draft]   missing keys (first 3): {missing[:3]}")
    if unexpected[:3]:
        print(f"[draft]   unexpected keys (first 3): {unexpected[:3]}")
    draft.eval()

    # Build prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    else:
        prompt = DEFAULT_PROMPT

    if args.no_chatml:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        u = tokenizer.encode("user", add_special_tokens=False)
        a = tokenizer.encode("assistant", add_special_tokens=False)
        nl = tokenizer.encode("\n", add_special_tokens=False)
        body = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = im_start + u + nl + body + im_end + nl + im_start + a + nl
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    print(f"[prompt] {input_ids.shape[1]} tokens")

    print(f"\n[eval] running spec_generate with max_cycles={args.max_cycles}, "
          f"max_new_tokens={args.max_new_tokens}...")
    output_ids, accept_lengths = spec_generate_capped(
        draft, target, tokenizer, input_ids,
        args.max_new_tokens, args.max_cycles, device,
    )

    tau = sum(accept_lengths) / max(1, len(accept_lengths))
    emitted = sum(accept_lengths)
    print(f"\n{'='*60}")
    print(f"cycles: {len(accept_lengths)}")
    print(f"accept_lengths: {accept_lengths}")
    print(f"τ = {tau:.3f}  (emitted {emitted} tokens in {len(accept_lengths)} cycles)")
    print(f"{'='*60}")

    # Show generated text for sanity
    gen = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:input_ids.shape[1] + emitted + 1].cpu().tolist(),
        skip_special_tokens=False,
    )
    print(f"\n[gen] {gen!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
