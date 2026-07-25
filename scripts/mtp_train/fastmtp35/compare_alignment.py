#!/usr/bin/env python3
"""Compare official/checkpoint MTP weights under legacy and serving alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from mtp_module import Qwen35MtpBlock, load_mtp_from_safetensors
from train_head import (
    ALIGNMENTS,
    RECURRENCE_INPUTS,
    FeatureShard,
    evaluate,
    find_tensor,
    load_checkpoint_weights,
    load_vocab_map,
    snapshot_dir,
)


def named_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/path/to/weights.safetensors")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--vocab-map", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=named_checkpoint, default=[])
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    shards = sorted(args.validation_features.glob("*.rwf"))
    if not shards:
        raise SystemExit(f"no validation shards under {args.validation_features}")
    header = FeatureShard(shards[0]).header
    dim = int(header["hidden_dim"])
    k = int(header["recursive_steps"])

    config = AutoConfig.from_pretrained(snapshot_dir(args.hf_model), trust_remote_code=True)
    text_config = config.text_config if hasattr(config, "text_config") else config
    full_vocab = int(text_config.vocab_size)
    vocab_map_cpu, inverse_cpu = load_vocab_map(args.vocab_map, full_vocab)
    embed_weight = find_tensor(
        args.hf_model,
        [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
    ).to(torch.bfloat16).to(device)
    lm_weight = find_tensor(
        args.hf_model,
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "model.language_model.lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
    ).to(torch.bfloat16)[vocab_map_cpu].contiguous().to(device)
    vocab_map = vocab_map_cpu.to(device)
    inverse_vocab = inverse_cpu.to(device)

    warm_start = load_mtp_from_safetensors(str(args.hf_model))
    candidates: list[tuple[str, Path | None]] = [("official", None), *args.checkpoint]
    for label, checkpoint in candidates:
        model = Qwen35MtpBlock(text_config)
        missing, unexpected = model.load_pretrained_(warm_start)
        if missing or unexpected:
            raise ValueError(
                f"official MTP warm start mismatch: missing={missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
        if checkpoint is not None:
            load_checkpoint_weights(model, checkpoint)
        model = model.to(device=device, dtype=torch.bfloat16)
        for alignment in ALIGNMENTS:
            for recurrence_input in RECURRENCE_INPUTS:
                metrics = evaluate(
                    model,
                    embed_weight,
                    lm_weight,
                    vocab_map,
                    inverse_vocab,
                    shards,
                    0,
                    1,
                    args.seed,
                    args.micro_batch_size,
                    dim,
                    k,
                    args.eval_batches,
                    alignment,
                    recurrence_input,
                )
                print(
                    json.dumps(
                        {
                            "event": "alignment-eval",
                            "weights": label,
                            "checkpoint": str(checkpoint) if checkpoint else "official",
                            "alignment": alignment,
                            "recurrence_input": recurrence_input,
                            "metrics": metrics,
                        }
                    ),
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
