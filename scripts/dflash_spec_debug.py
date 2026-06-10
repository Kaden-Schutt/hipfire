#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Debug inline spec_generate step-by-step: dump posterior, block_out, etc.

The goal: figure out why the draft+target loop produces 'useruser...'
while target.generate alone produces '<think>\\nThinking Process:'.
"""
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO_ROOT = Path(__file__).resolve().parent.parent
DFLASH_REFERENCE = Path(os.environ.get("HIPFIRE_DFLASH_REFERENCE", REPO_ROOT / ".dflash-reference"))
if not (DFLASH_REFERENCE / "dflash" / "model.py").exists():
    raise SystemExit(f"missing DFlash reference at {DFLASH_REFERENCE}")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(DFLASH_REFERENCE))

from dflash_train_poc import build_draft_config  # noqa: E402
from dflash.model import DFlashDraftModel, extract_context_feature, sample  # type: ignore[import-not-found]  # noqa: E402

device = torch.device("cuda")
dtype = torch.bfloat16

print("[target] loading...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
tgt_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=dtype, attn_implementation="eager",
).to(device)  # type: ignore[arg-type]
target.eval()

print("[draft] loading z-lab weights...", flush=True)
cfg = build_draft_config(tgt_cfg, 5, 16, 248070, match_zlab_arch=True)
draft = DFlashDraftModel(cfg).to(device=device, dtype=dtype)
sd = load_file(
    os.environ.get(
        "HIPFIRE_DFLASH_ZLAB_SAFETENSORS",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots"
            / "96899cc270945f554998309580b08a04a05a3187/model.safetensors"
        ),
    )
)
miss, unex = draft.load_state_dict(sd, strict=False)
print(f"[draft]   missing={len(miss)} unexpected={len(unex)}", flush=True)
draft.eval()

prompt = "You are an AI assistant. Call the get_weather tool to find today weather in Tokyo, then summarize the result in two sentences."
im_s = tok.encode("<|im_start|>", add_special_tokens=False)
im_e = tok.encode("<|im_end|>", add_special_tokens=False)
u = tok.encode("user", add_special_tokens=False)
a = tok.encode("assistant", add_special_tokens=False)
nl = tok.encode("\n", add_special_tokens=False)
body = tok.encode(prompt, add_special_tokens=False)
ids = im_s + u + nl + body + im_e + nl + im_s + a + nl
ids = torch.tensor([ids], dtype=torch.long, device=device)
num_in = ids.shape[1]
print(f"[prompt] {num_in} tokens: {tok.decode(ids[0].cpu().tolist(), skip_special_tokens=False)!r}", flush=True)

@torch.inference_mode()
def debug_spec(max_cycles=3, B=16):
    maxL = num_in + 96
    out = torch.full((1, maxL + B), draft.mask_token_id, dtype=torch.long, device=device)
    pos = torch.arange(out.shape[1], device=device).unsqueeze(0)
    pkv_t = DynamicCache(config=target.config)
    pkv_d = DynamicCache()

    print("\n=== PREFILL ===", flush=True)
    o = target(ids, position_ids=pos[:, :num_in],
               past_key_values=pkv_t, use_cache=True,
               logits_to_keep=1, output_hidden_states=True)
    first_tok = sample(o.logits, 0.0)
    print(f"prefill sample (position {num_in}): "
          f"id={first_tok[0,0].item()} tok={tok.decode([first_tok[0,0].item()])!r}", flush=True)
    out[:, :num_in] = ids
    out[:, num_in:num_in+1] = first_tok
    th = extract_context_feature(o.hidden_states, draft.target_layer_ids)
    print(f"target_hidden shape: {th.shape}  expected (1, {num_in}, 5*2560=12800)", flush=True)

    start = num_in
    for cy in range(max_cycles):
        print(f"\n=== CYCLE {cy+1} (start={start}) ===", flush=True)
        block_out = out[:, start:start+B].clone()
        block_pos = pos[:, start:start+B]
        print(f"block_out input: {block_out[0, :5].tolist()}... (first 5)", flush=True)
        print(f"  = {[tok.decode([t]) for t in block_out[0, :5].tolist()]!r}", flush=True)
        print(f"block_pos: {block_pos[0, :5].tolist()}...", flush=True)

        ne = target.model.embed_tokens(block_out)
        d_pos = pos[:, pkv_d.get_seq_length():start+B]
        print(f"draft position_ids: [{d_pos[0, 0].item()}..{d_pos[0, -1].item()}] len={d_pos.shape[1]}", flush=True)
        print(f"draft target_hidden shape: {th.shape}", flush=True)

        dh = draft(target_hidden=th, noise_embedding=ne,
                   position_ids=d_pos, past_key_values=pkv_d, use_cache=True)
        print(f"draft output shape: {dh.shape}", flush=True)
        draft_logits = target.lm_head(dh[:, -B+1:, :])
        print(f"draft_logits shape: {draft_logits.shape}", flush=True)
        pkv_d.crop(start)
        draft_preds = sample(draft_logits, 0.0)
        print(f"draft predictions (positions {start+1}..{start+B-1}): {draft_preds[0, :5].tolist()}...", flush=True)
        print(f"  = {[tok.decode([t]) for t in draft_preds[0, :5].tolist()]!r}", flush=True)

        block_out[:, 1:] = draft_preds
        print(f"block_out after draft sub: {[tok.decode([t]) for t in block_out[0, :6].tolist()]!r}", flush=True)

        o = target(block_out, position_ids=block_pos,
                   past_key_values=pkv_t, use_cache=True, output_hidden_states=True)
        print(f"target out logits shape: {o.logits.shape}", flush=True)
        post = sample(o.logits, 0.0)
        print(f"target posterior (first 5): {post[0, :5].tolist()}", flush=True)
        print(f"  = {[tok.decode([t]) for t in post[0, :5].tolist()]!r}", flush=True)

        # accept: block_out[:, 1:] vs posterior[:, :-1]
        match = block_out[:, 1:] == post[:, :-1]
        al = match.cumprod(dim=1).sum(dim=1)[0].item()
        print(f"accept_length: {al}  (match[0,:5]={match[0, :5].tolist()})", flush=True)

        out[:, start:start+al+1] = block_out[:, :al+1]
        out[:, start+al+1] = post[:, al]
        print(f"setting out[{start+al+1}] = {post[0, al].item()} = {tok.decode([post[0, al].item()])!r}", flush=True)
        start += al + 1
        pkv_t.crop(start)
        th = extract_context_feature(o.hidden_states, draft.target_layer_ids)[:, :al+1, :]

    print("\n=== FINAL DECODE ===", flush=True)
    print(f"out[{num_in}:{start+1}] = {tok.decode(out[0, num_in:start+1].cpu().tolist(), skip_special_tokens=False)!r}", flush=True)

debug_spec(max_cycles=3)
