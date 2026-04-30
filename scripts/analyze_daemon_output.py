#!/usr/bin/env python3
"""Parse daemon JSONL output: extract tokens, check for corruption."""
import sys, json

tokens = []
done = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "token":
        tokens.append(d.get("text", ""))
    elif d.get("type") == "done":
        done = d

text = "".join(tokens)
im_leaks = text.count("<|im_start|>")
has_tc = "<tool_call>" in text

if done:
    pp = done.get("prefill_tokens", "?")
    pp_toks = done.get("prefill_tok_s", 0)
    dec_toks = done.get("decode_tok_s", 0)
    ntok = done.get("tokens", "?")
    print(f"  prefill={pp} pp_tok/s={pp_toks:.1f} decode_tok/s={dec_toks:.1f} tokens={ntok}")

print(f"  im_start_leaks={im_leaks} tool_call={has_tc}")
print(f"  text: {text[:250]}")

if im_leaks > 0:
    print("  ** FAIL: ChatML corruption **")
