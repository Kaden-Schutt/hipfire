#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Isolate target's cache-vs-no-cache behavior.

Test A: target(full_prompt, use_cache=False). logits[:, -1] argmax = next token.
Test B: target(prompt, past_kv=DynamicCache(config)), then target(block_out, past_kv, position_ids=[L..L+B-1]).
        logits[:, 0] argmax = should be SAME as Test A's next token (since we're predicting the same position).

If Test A = '\n' (198) and Test B = 'user' (846), there's a cache state bug.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
num_in = ids.shape[1]
print(f"[prompt] {num_in} tokens", flush=True)

# TEST A: no cache, full prompt + the first predicted token as input
think_tok = tok.encode("<think>", add_special_tokens=False)[0]
seq_with_think = torch.cat([ids, torch.tensor([[think_tok]], device=device)], dim=1)
print(f"\n[Test A] target(prompt+'<think>', no_cache). logits[:, -1] argmax:", flush=True)
with torch.inference_mode():
    out_A = target(seq_with_think, use_cache=False)
arg_A = out_A.logits[:, -1].argmax(dim=-1)
print(f"  token_id={arg_A.item()}  token={tok.decode([arg_A.item()])!r}", flush=True)
print(f"  expected '\\n' (198)", flush=True)

# TEST B: stepwise cache
print(f"\n[Test B] prefill + cycle with DynamicCache(config=target.config):", flush=True)
pkv_t = DynamicCache(config=target.config)
position_ids_full = torch.arange(num_in + 16, device=device).unsqueeze(0)
with torch.inference_mode():
    out_pre = target(
        ids, position_ids=position_ids_full[:, :num_in],
        past_key_values=pkv_t, use_cache=True,
        logits_to_keep=1, output_hidden_states=True,
    )
pre_arg = out_pre.logits[:, -1].argmax(dim=-1)
print(f"  prefill last logit argmax: {pre_arg.item()} = {tok.decode([pre_arg.item()])!r} (expect '<think>')", flush=True)

block = torch.tensor([[think_tok] + [248070] * 15], device=device)
with torch.inference_mode():
    out_B = target(
        block, position_ids=position_ids_full[:, num_in:num_in+16],
        past_key_values=pkv_t, use_cache=True, output_hidden_states=True,
    )
arg_B = out_B.logits[:, 0].argmax(dim=-1)
print(f"  cycle logits[:, 0] argmax: {arg_B.item()} = {tok.decode([arg_B.item()])!r} (expect '\\n')", flush=True)
print(f"  cycle logits[:, 1] argmax: {out_B.logits[:, 1].argmax(dim=-1).item()} = "
      f"{tok.decode([out_B.logits[:, 1].argmax(dim=-1).item()])!r}", flush=True)

# TEST C: same but WITHOUT logits_to_keep=1 on prefill
print(f"\n[Test C] prefill WITHOUT logits_to_keep=1:", flush=True)
pkv_t2 = DynamicCache(config=target.config)
with torch.inference_mode():
    out_pre2 = target(
        ids, position_ids=position_ids_full[:, :num_in],
        past_key_values=pkv_t2, use_cache=True,
        output_hidden_states=True,
    )
    out_C = target(
        block, position_ids=position_ids_full[:, num_in:num_in+16],
        past_key_values=pkv_t2, use_cache=True, output_hidden_states=True,
    )
print(f"  cycle logits[:, 0] argmax: {out_C.logits[:, 0].argmax(dim=-1).item()} = "
      f"{tok.decode([out_C.logits[:, 0].argmax(dim=-1).item()])!r}", flush=True)

# TEST D: plain DynamicCache (no config)
print(f"\n[Test D] prefill with plain DynamicCache() (no config=):", flush=True)
try:
    pkv_t3 = DynamicCache()
    with torch.inference_mode():
        out_pre3 = target(
            ids, position_ids=position_ids_full[:, :num_in],
            past_key_values=pkv_t3, use_cache=True,
            output_hidden_states=True,
        )
        out_D = target(
            block, position_ids=position_ids_full[:, num_in:num_in+16],
            past_key_values=pkv_t3, use_cache=True, output_hidden_states=True,
        )
    print(f"  cycle logits[:, 0] argmax: {out_D.logits[:, 0].argmax(dim=-1).item()} = "
          f"{tok.decode([out_D.logits[:, 0].argmax(dim=-1).item()])!r}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
