#!/usr/bin/env python3
"""Run a small Nemotron-H prompt through vLLM as an external reference.

This is tooling only; it is not part of Hipfire's inference path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

if os.environ.get("HIPFIRE_VLLM_HIDE_FLASH_ATTN") == "1":
    import importlib.util

    _find_spec = importlib.util.find_spec

    def _find_spec_without_flash_attn(name: str, *args, **kwargs):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return _find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec_without_flash_attn

from vllm import LLM, SamplingParams


DEFAULT_MODEL = (
    "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="Answer in one short sentence: What is 2+2?")
    ap.add_argument("--system", default=None)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--logprobs", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--language-model-only", action="store_true")
    ap.add_argument("--skip-mm-profiling", action="store_true")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    return ap.parse_args()


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


def main() -> None:
    args = parse_args()
    model = str(args.model)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    prompt = render_prompt(tok, args)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")]
    eos_id = tok.eos_token_id
    if eos_id is not None and eos_id not in stop_ids:
        stop_ids.append(eos_id)
    if 2 not in stop_ids:
        stop_ids.append(2)

    print("prompt_ids", prompt_ids, flush=True)
    print("prompt_repr", repr(prompt), flush=True)
    print("stop_ids", stop_ids, flush=True)

    print("llm_init_start", flush=True)
    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        language_model_only=args.language_model_only,
        skip_mm_profiling=args.skip_mm_profiling,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    print("llm_init_done", flush=True)
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=False,
        logprobs=args.logprobs,
    )
    print("generate_start", flush=True)
    result = llm.generate([prompt], params)[0]
    print("generate_done", flush=True)

    payload = {
        "model": model,
        "prompt": prompt,
        "prompt_ids": list(result.prompt_token_ids or prompt_ids),
        "outputs": [],
    }
    for out in result.outputs:
        entry = {
            "text": out.text,
            "token_ids": list(out.token_ids),
            "finish_reason": out.finish_reason,
            "stop_reason": out.stop_reason,
        }
        if out.logprobs is not None:
            entry["logprobs"] = [
                {
                    str(tok_id): {
                        "rank": lp.rank,
                        "logprob": lp.logprob,
                        "decoded_token": lp.decoded_token,
                    }
                    for tok_id, lp in step.items()
                }
                for step in out.logprobs
            ]
        payload["outputs"].append(entry)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
