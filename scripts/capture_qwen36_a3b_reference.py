#!/usr/bin/env python3
"""Capture Qwen3.6-35B-A3B PyTorch greedy/top-k reference artifacts.

The output shape intentionally matches `greedy_dump_top5`:

  <out-prefix>.prompt_tokens       one prompt token ID per line
  <out-prefix>.prompt_tokens.json  same IDs as a JSON array
  <out-prefix>.tokens              one generated greedy token ID per line
  <out-prefix>.top5.csv            per-step top-5 logits
  <out-prefix>.meta.json           reproducibility metadata

Use `--dtype float16` when validating Hipfire's fp16 HFQ container path.
Use `--dtype bfloat16` when checking against the upstream source dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)


DEFAULT_MODEL = (
    "/home/sadara/Models/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DEFAULT_PROMPT = "A farmer has 17 sheep. All but 9 die. State the final number."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file")
    p.add_argument("--mode", choices=["raw", "chat", "thinking"], default="chat")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--step-mode",
        choices=["full", "stream"],
        default="full",
        help=(
            "full recomputes the whole prefix each step; stream feeds one token "
            "at a time through the model cache, matching Hipfire decode semantics."
        ),
    )
    p.add_argument(
        "--causal-lm-class",
        action="store_true",
        help="Use the text-only CausalLM class. Default uses the checkpoint's conditional-generation wrapper.",
    )
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_no_special(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def prompt_ids(tok, prompt: str, mode: str) -> list[int]:
    if mode == "raw":
        return encode_no_special(tok, prompt)
    ids: list[int] = []
    ids += encode_no_special(tok, "<|im_start|>")
    ids += encode_no_special(tok, "user")
    ids += encode_no_special(tok, "\n")
    ids += encode_no_special(tok, prompt)
    ids += encode_no_special(tok, "<|im_end|>")
    ids += encode_no_special(tok, "\n")
    ids += encode_no_special(tok, "<|im_start|>")
    ids += encode_no_special(tok, "assistant")
    ids += encode_no_special(tok, "\n")
    if mode == "thinking":
        ids += encode_no_special(tok, "<think>")
        ids += encode_no_special(tok, "\n")
    return ids


def top5_row(step: int, logits: torch.Tensor) -> str:
    vals, ids = torch.topk(logits.float(), k=5)
    ids_l = [int(x) for x in ids.tolist()]
    vals_l = [float(x) for x in vals.tolist()]
    margin = vals_l[0] - vals_l[1]
    return (
        f"{step},{ids_l[0]},{vals_l[0]:.8f},{ids_l[1]},{vals_l[1]:.8f},"
        f"{ids_l[2]},{vals_l[2]:.8f},{ids_l[3]},{vals_l[3]:.8f},"
        f"{ids_l[4]},{vals_l[4]:.8f},{margin:.8f}"
    )


def capture_full_prefix(model, ids: list[int], args: argparse.Namespace) -> tuple[list[int], list[str]]:
    generated: list[int] = []
    csv_lines = [
        "step,r1_id,r1_logit,r2_id,r2_logit,r3_id,r3_logit,"
        "r4_id,r4_logit,r5_id,r5_logit,margin_top12"
    ]
    cur = torch.tensor([ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            t_step = time.time()
            out = model(cur, use_cache=False, logits_to_keep=1)
            logits = out.logits[0, -1, :]
            csv_lines.append(top5_row(step, logits))
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            cur = torch.cat(
                [cur, torch.tensor([[next_id]], dtype=torch.long, device=args.device)],
                dim=1,
            )
            print(f"step {step}: token={next_id} elapsed={time.time() - t_step:.2f}s", flush=True)
    return generated, csv_lines


def capture_streaming(model, ids: list[int], args: argparse.Namespace) -> tuple[list[int], list[str]]:
    generated: list[int] = []
    csv_lines = [
        "step,r1_id,r1_logit,r2_id,r2_logit,r3_id,r3_logit,"
        "r4_id,r4_logit,r5_id,r5_logit,margin_top12"
    ]
    past = None
    logits = None

    with torch.no_grad():
        for token in ids:
            x = torch.tensor([[token]], dtype=torch.long, device=args.device)
            out = model(x, use_cache=True, past_key_values=past, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[0, -1, :]

        if logits is None:
            raise ValueError("empty prompt token sequence")

        for step in range(args.max_new_tokens):
            t_step = time.time()
            if step > 0:
                x = torch.tensor([[generated[-1]]], dtype=torch.long, device=args.device)
                out = model(x, use_cache=True, past_key_values=past, logits_to_keep=1)
                past = out.past_key_values
                logits = out.logits[0, -1, :]

            csv_lines.append(top5_row(step, logits))
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            print(f"step {step}: token={next_id} elapsed={time.time() - t_step:.2f}s", flush=True)
    return generated, csv_lines


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"error: model dir not found: {model_dir}", file=sys.stderr)
        return 2

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    dtype = torch_dtype(args.dtype)

    print(f"loading tokenizer: {model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ids = prompt_ids(tok, prompt, args.mode)
    print(f"prompt mode={args.mode} tokens={len(ids)}", flush=True)

    prompt_txt = "\n".join(str(x) for x in ids) + "\n"
    out_prefix.with_suffix(".prompt_tokens").write_text(prompt_txt)
    out_prefix.with_suffix(".prompt_tokens.json").write_text(json.dumps(ids) + "\n")

    print(
        f"loading model dtype={args.dtype} device={args.device} "
        f"torch={torch.__version__}",
        flush=True,
    )
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if args.causal_lm_class and hasattr(config, "text_config"):
        for key, value in config.text_config.to_dict().items():
            if not hasattr(config, key):
                setattr(config, key, value)
    t0 = time.time()
    if args.causal_lm_class:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_dir,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.to(args.device)
    first_param = next(model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    if args.step_mode == "stream":
        generated, csv_lines = capture_streaming(model, ids, args)
    else:
        generated, csv_lines = capture_full_prefix(model, ids, args)

    out_prefix.with_suffix(".tokens").write_text("\n".join(str(x) for x in generated) + "\n")
    out_prefix.with_suffix(".top5.csv").write_text("\n".join(csv_lines) + "\n")
    meta = {
        "model_dir": str(model_dir),
        "config_sha256": file_sha256(model_dir / "config.json"),
        "tokenizer_sha256": file_sha256(model_dir / "tokenizer.json"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "dtype": args.dtype,
        "device": args.device,
        "step_mode": args.step_mode,
        "model_class": type(model).__name__,
        "first_parameter_dtype": str(first_param.dtype),
        "first_parameter_device": str(first_param.device),
        "parameter_count": param_count,
        "mode": args.mode,
        "prompt": prompt,
        "prompt_token_count": len(ids),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated,
    }
    out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote prefix: {out_prefix}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
