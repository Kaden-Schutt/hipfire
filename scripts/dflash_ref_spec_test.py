#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Run z-lab's unmodified reference spec_generate on our prompt.

If this gives τ=1, the issue is NOT in our tau_probe reproduction —
there's something fundamentally broken with how the draft interacts
with our Qwen3.5 target (e.g. cache layout, position_ids, etc).

If this gives τ>1, our tau_probe has a reproducible bug to find.
"""
import sys, torch
sys.path.insert(0, "/root/hipfire/.dflash-reference")
from dflash.model import DFlashDraftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")
dtype = torch.bfloat16

print("[target] loading...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=dtype, attn_implementation="eager",
).to(device)
target.eval()

print("[draft] loading z-lab 4B DFlash...", flush=True)
draft = DFlashDraftModel.from_pretrained(
    "z-lab/Qwen3.5-4B-DFlash", torch_dtype=dtype, trust_remote_code=True,
).to(device)
draft.eval()
print(f"[draft]   block_size={draft.block_size}  mask_token_id={draft.mask_token_id}", flush=True)

prompt = "You are an AI assistant. Call the get_weather tool to find today weather in Tokyo, then summarize the result in two sentences."
im_s = tok.encode("<|im_start|>", add_special_tokens=False)
im_e = tok.encode("<|im_end|>", add_special_tokens=False)
u = tok.encode("user", add_special_tokens=False)
a = tok.encode("assistant", add_special_tokens=False)
nl = tok.encode("\n", add_special_tokens=False)
body = tok.encode(prompt, add_special_tokens=False)
ids = im_s + u + nl + body + im_e + nl + im_s + a + nl
ids = torch.tensor([ids], dtype=torch.long, device=device)
print(f"[prompt] {ids.shape[1]} tokens", flush=True)

try:
    out = draft.spec_generate(
        target=target,
        input_ids=ids,
        max_new_tokens=96,
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else [],
        temperature=0.0,
    )
    gen_ids = out[0, ids.shape[1]:].cpu().tolist()
    gen = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"[out] shape={out.shape}  new_len={len(gen_ids)}", flush=True)
    print(f"[gen] {gen[:400]!r}", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
