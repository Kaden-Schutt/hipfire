#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Sanity check: does Qwen3.5-4B target alone produce reasonable greedy
output on our agentic prompt? If target outputs 'useruser...' itself, the
draft failure might just be a reflection of target's weird behavior.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")
dtype = torch.bfloat16

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=dtype, attn_implementation="eager",
).to(device)
target.eval()

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
print("[target] greedy generation, 48 new tokens...", flush=True)
with torch.inference_mode():
    out = target.generate(
        ids, max_new_tokens=48, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
gen = tok.decode(out[0, ids.shape[1]:].cpu().tolist(), skip_special_tokens=False)
print(f"[out] {gen!r}", flush=True)
