#!/usr/bin/env python3
"""HF transformers reference dump for the 9B reasoning-loop investigation.

Loads Qwen3.5-9B BF16 in BF16 on GPU (any backend torch detects),
tokenizes the same 44-token chat-templated prompt that produced
/tmp/hidden_chat.bin via hipfire, runs one forward pass with
`output_hidden_states=True`, writes per-layer residual stream states
in the HFHS format that /tmp/diff_hidden.py expects:

  magic  8B = b"HFHS\\0\\0\\0\\0"
  n_layers u32  (post-block residual streams, layers 0..N-1)
  n_pos    u32  (= prompt token count)
  hidden_dim u32
  reserved u32 = 0
  body: n_layers blocks of (n_pos, hidden_dim) f32 row-major

HF returns hidden_states tuple of length (num_hidden_layers + 1) — index 0
is the embedding pre-block, index k is the residual stream AFTER block k-1.
We dump indices 1..=N (post-block-0 .. post-block-(N-1)) to match what
HiddenStateRingBuffer in hipfire captures.
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

# Limit transformers' verbose chatter
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="local model dir")
ap.add_argument("--prompt-tokens", required=True,
                help="JSON file with array of token ids (the exact 44 tokens)")
ap.add_argument("--out", required=True, help="HFHS output path")
ap.add_argument("--device", default="cuda",
                help="cuda (rocm) or cpu (CPU is slow but works)")
ap.add_argument("--dtype", default="bfloat16",
                choices=["bfloat16", "float16", "float32"])
args = ap.parse_args()

dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
         "float32": torch.float32}[args.dtype]

print(f"loading model from {args.model} dtype={args.dtype} device={args.device}",
      file=sys.stderr)
from transformers import AutoModelForCausalLM, AutoConfig

cfg = AutoConfig.from_pretrained(args.model)
# Qwen3.5 is a nested config (text_config holds the language model fields)
tc = cfg.text_config
n_layers = tc.num_hidden_layers
hidden_dim = tc.hidden_size
print(f"text config: n_layers={n_layers} hidden={hidden_dim} vocab={tc.vocab_size}",
      file=sys.stderr)

# Load text-only — we don't need the vision tower for this experiment.
# Qwen3_5ForConditionalGeneration is the multimodal wrapper; the LM lives at
# `.language_model` (Qwen3_5ForCausalLM-style). Try the wrapper first; fall
# back to ForCausalLM if available.
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"AutoModelForCausalLM failed ({e}); retrying with AutoModel", file=sys.stderr)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True)

print(f"loaded model: {type(model).__name__}", file=sys.stderr)
model = model.to(args.device).eval()

# Find the underlying text-LM module so we can call its forward with
# output_hidden_states cleanly (and skip any vision branches).
def find_lm(m):
    # Common HF patterns: m.language_model, m.model, m.text_model
    for attr in ("language_model", "model", "text_model"):
        if hasattr(m, attr):
            sub = getattr(m, attr)
            if hasattr(sub, "embed_tokens") or hasattr(sub, "model"):
                return sub
    return m

lm = find_lm(model)
print(f"LM module: {type(lm).__name__}", file=sys.stderr)

# Load tokens
with open(args.prompt_tokens) as f:
    tok_ids = json.load(f)
assert isinstance(tok_ids, list) and all(isinstance(t, int) for t in tok_ids), \
    f"expected JSON array of ints, got {type(tok_ids)}"
n_pos = len(tok_ids)
print(f"prompt: {n_pos} tokens, first 10 = {tok_ids[:10]}", file=sys.stderr)

input_ids = torch.tensor([tok_ids], dtype=torch.long, device=args.device)

# Forward pass — text-only path. Some Qwen3_5 wrappers require an attention
# mask + position_ids; build them explicitly.
attention_mask = torch.ones_like(input_ids)
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

hs = out.hidden_states
print(f"got {len(hs)} hidden_states (expected n_layers+1 = {n_layers+1})",
      file=sys.stderr)
# Drop index 0 (pre-block embedding) so output matches hipfire's
# "post-block residual stream per layer" convention.
post_block = hs[1:]
assert len(post_block) == n_layers, f"{len(post_block)} != {n_layers}"

# Write HFHS
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "wb") as f:
    f.write(b"HFHS\0\0\0\0")
    f.write(struct.pack("<I", n_layers))
    f.write(struct.pack("<I", n_pos))
    f.write(struct.pack("<I", hidden_dim))
    f.write(struct.pack("<I", 0))
    for L, h in enumerate(post_block):
        # h is [1, n_pos, hidden_dim] in the model's dtype
        arr = h[0].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        assert arr.shape == (n_pos, hidden_dim), f"layer {L}: {arr.shape}"
        f.write(arr.tobytes())
        if L < 3 or L == n_layers - 1:
            last = arr[-1]
            rms = float(np.sqrt((last * last).mean()))
            print(f"  layer {L}: last-pos rms={rms:.4f}", file=sys.stderr)

sz = os.path.getsize(args.out)
print(f"wrote {args.out} ({sz/1024/1024:.1f} MB)", file=sys.stderr)
