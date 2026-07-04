#!/usr/bin/env python3
"""Run a small prompt through HF Transformers as an external reference.

This is tooling only; it is not part of Hipfire's inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if os.environ.get("HIPFIRE_HIDE_FLASH_ATTN") == "1":
    _find_spec = importlib.util.find_spec

    def _find_spec_without_flash_attn(name: str, *args: Any, **kwargs: Any):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return _find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec_without_flash_attn

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = (
    "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="Answer in one short sentence: What is 2+2?")
    ap.add_argument("--system", default=None)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stop-on-eos", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def render_prompt(tok: AutoTokenizer, args: argparse.Namespace) -> str:
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.text})
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking == "on",
    )


def forward_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    try:
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits[0, -1, :]


def greedy_decode(
    model,
    prompt_ids: list[int],
    args: argparse.Namespace,
    stop_ids: set[int],
) -> tuple[list[int], list[dict[str, Any]], list[float]]:
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    durations: list[float] = []
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    with torch.no_grad():
        for step in range(args.max_new_tokens):
            t0 = time.time()
            logits = forward_last_logits(model, cur)
            vals, ids = torch.topk(logits.float(), k=args.top_k)
            next_id = int(ids[0].item())
            generated.append(next_id)
            elapsed = time.time() - t0
            durations.append(elapsed)
            steps.append(
                {
                    "step": step,
                    "token_id": next_id,
                    "top_ids": [int(x) for x in ids.tolist()],
                    "top_logits": [float(x) for x in vals.tolist()],
                    "elapsed_s": elapsed,
                }
            )
            print(f"step {step}: token={next_id} elapsed={elapsed:.2f}s", flush=True)
            if args.stop_on_eos and next_id in stop_ids:
                break
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
            cur = torch.cat([cur, next_token], dim=1)
    return generated, steps, durations


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"error: model not found: {model_dir}", file=sys.stderr)
        return 2

    print(f"loading tokenizer: {model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = render_prompt(tok, args)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    stop_ids = {int(x) for x in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), 2) if x is not None}
    print("prompt_ids", prompt_ids, flush=True)
    print("prompt_repr", repr(prompt), flush=True)
    print("stop_ids", sorted(stop_ids), flush=True)

    print(
        f"loading model dtype={args.dtype} device={args.device} torch={torch.__version__}",
        flush=True,
    )
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(args.device)
    first_param = next(model.parameters())
    print(
        f"model loaded class={type(model).__name__} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )

    generated, steps, durations = greedy_decode(model, prompt_ids, args, stop_ids)
    text = tok.decode(generated, skip_special_tokens=False)
    payload = {
        "model": str(model_dir),
        "config_sha256": sha256_file(model_dir / "config.json") if (model_dir / "config.json").exists() else None,
        "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json")
        if (model_dir / "tokenizer.json").exists()
        else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "model_class": type(model).__name__,
        "dtype": args.dtype,
        "device": args.device,
        "first_parameter_dtype": str(first_param.dtype),
        "first_parameter_device": str(first_param.device),
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "generated_ids": generated,
        "generated_text": text,
        "steps": steps,
        "tokens_per_second": (len(generated) / sum(durations)) if durations else 0.0,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
