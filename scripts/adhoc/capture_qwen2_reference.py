#!/usr/bin/env python3
"""Capture Qwen2-1.5B-Instruct reference logits + token IDs for hipfire validation.

Phase 0 / phase 1 of the dots.ocr + Qwen2 bring-up plan (see
docs/plans/dots-ocr-prd.md).

What this captures, for the prompt at benchmarks/prompts/qwen2_smoke.txt:

  - prompt_token_ids: tokenizer output, no chat template
  - first_16_completion_token_ids: greedy-decoded continuation
  - logits_top100_at_pos_{0,8,16}: top-100 token IDs and their f32 logit
    values at the listed positions of the prompt's *forward pass*
    (NOT the completion). Position i is the model's distribution over
    the token that would follow prompt_token_ids[:i+1].

The output JSON is committed to benchmarks/references/ so the future
hipfire-arch-qwen2 forward pass has a fixed comparison target.

Run from repo root:
    .venv/bin/python scripts/capture_qwen2_reference.py

Idempotent — overwrites the output file. Recompute when transformers
version or model snapshot changes; the artifact records both for
reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO / "benchmarks" / "prompts" / "qwen2_smoke.txt"
OUT_PATH = REPO / "benchmarks" / "references" / "qwen2_1p5b_instruct_smoke.json"
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
SNAPSHOT = (
    "/data/cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/"
    "snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
)


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not PROMPT_PATH.exists():
        print(f"error: prompt not found: {PROMPT_PATH}", file=sys.stderr)
        return 1
    if not Path(SNAPSHOT).is_dir():
        print(
            f"error: model snapshot not found: {SNAPSHOT}\n"
            f"  hint: huggingface-cli download {MODEL_ID}",
            file=sys.stderr,
        )
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompt_bytes = PROMPT_PATH.read_bytes()
    prompt_md5 = md5(PROMPT_PATH)
    print(f"prompt: {PROMPT_PATH}  ({len(prompt_bytes)} bytes, md5={prompt_md5})")

    print(f"loading tokenizer from {SNAPSHOT}...")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=False)

    print(f"loading model from {SNAPSHOT} (cpu, float32)...")
    # transformers 5.x renamed `torch_dtype` to `dtype`; using the new
    # name. `device_map="cpu"` would require `accelerate` — CPU is the
    # default placement so omit it. The whole model fits in RAM at f32
    # (~6 GB) on this host.
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.eval()

    prompt_text = prompt_bytes.decode("utf-8")
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    n_prompt = input_ids.shape[1]
    print(f"prompt tokens: {n_prompt}")

    # Forward pass over the prompt.
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits[0].float()  # [n_prompt, vocab_size]
    print(f"logits shape: {tuple(logits.shape)}")

    # Top-100 at the requested positions. Position i is the distribution
    # over the token that follows prompt[:i+1] — i.e. the model's
    # prediction for prompt[i+1] (or for the first completion token at
    # i = n_prompt-1).
    top_positions = [0, 8, 16, n_prompt - 1]
    top_positions = [p for p in top_positions if 0 <= p < n_prompt]
    logit_dump: dict[str, list[dict[str, float]]] = {}
    for p in top_positions:
        vals, ids = torch.topk(logits[p], k=100)
        logit_dump[f"pos_{p}"] = [
            {"token_id": int(t), "logit": float(v)}
            for t, v in zip(ids.tolist(), vals.tolist())
        ]

    # Greedy decode 32 continuation tokens.
    print("greedy decoding 32 continuation tokens...")
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            num_beams=1,
        )
    completion_ids = gen[0, n_prompt:].tolist()
    first_16 = completion_ids[:16]
    completion_text = tok.decode(completion_ids, skip_special_tokens=False)
    print(f"completion (32 tokens): {completion_text!r}")
    print(f"first 16 IDs: {first_16}")

    artifact = {
        "model_id": MODEL_ID,
        "snapshot": SNAPSHOT,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "torch_dtype": "float32",
        "device": "cpu",
        "prompt_path": str(PROMPT_PATH.relative_to(REPO)),
        "prompt_md5": prompt_md5,
        "prompt_byte_count": len(prompt_bytes),
        "prompt_token_ids": input_ids[0].tolist(),
        "n_prompt_tokens": n_prompt,
        "completion_token_ids_32": completion_ids,
        "first_16_completion_token_ids": first_16,
        "completion_text": completion_text,
        "logits_top100_at_positions": logit_dump,
        "greedy_decode": {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "num_beams": 1,
        },
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
