#!/usr/bin/env python3
"""Join Hipfire completion shards to prompt tokens and create train splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import xxhash


def hash64(value: str) -> int:
    return xxhash.xxh3_64_intdigest(value.encode("utf-8"))


def repeated(tokens: list[int]) -> bool:
    if len(tokens) < 64:
        return False
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    return len(set(grams)) / len(grams) < 0.25


def split_for(identifier: object) -> str:
    value = hash64(json.dumps(identifier, sort_keys=True, ensure_ascii=False)) % 100
    if value == 0:
        return "validation"
    if value == 1:
        return "test"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=400_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: (args.output / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    counts = {split: 0 for split in outputs}
    rejected: dict[str, int] = {}
    accepted_finish_reasons: dict[str, int] = {}
    seen_outputs: set[int] = set()

    try:
        for job_path in sorted((args.root / "jobs").glob("*.jsonl")):
            prompts: dict[int, dict] = {}
            with job_path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    prompts[index] = json.loads(line)
            completion_paths = sorted(
                (args.root / "completions").glob(f"{job_path.stem}.gpu*.jsonl")
            )
            if len(completion_paths) != 4:
                raise SystemExit(
                    f"{job_path.name}: expected four completion shards, "
                    f"found {len(completion_paths)}"
                )
            for completion_path in completion_paths:
                with completion_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        completion = json.loads(line)
                        prompt = prompts.get(int(completion["index"]))
                        if prompt is None:
                            raise SystemExit(
                                f"{completion_path}: unknown input index "
                                f"{completion['index']}"
                            )
                        tokens = [int(token) for token in completion["completion_tokens"]]
                        reason = str(completion.get("finish_reason") or "unknown")
                        reject_reason = None
                        # A length-capped trunk trajectory is valid MTP
                        # supervision: training masks the final recursive
                        # targets instead of requiring an EOS-complete SFT
                        # answer. Reject only trajectories whose tokens are
                        # intrinsically unusable.
                        if len(tokens) < 32:
                            reject_reason = "too_short"
                        elif repeated(tokens):
                            reject_reason = "repetitive"
                        output_hash = xxhash.xxh3_64_intdigest(
                            bytes().join(int(token).to_bytes(4, "little") for token in tokens)
                        )
                        if output_hash in seen_outputs:
                            reject_reason = "duplicate_output"
                        if reject_reason:
                            rejected[reject_reason] = rejected.get(reject_reason, 0) + 1
                            continue
                        seen_outputs.add(output_hash)
                        input_ids = [int(token) for token in prompt["tokens"]] + tokens
                        row = {
                            "id": prompt["id"],
                            "input_ids": input_ids,
                            "assistant_start": len(prompt["tokens"]),
                            "completion_tokens": len(tokens),
                            "finish_reason": reason,
                            "sampling": completion["sampling"],
                        }
                        split = split_for(prompt["id"])
                        outputs[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[split] += 1
                        accepted_finish_reasons[reason] = (
                            accepted_finish_reasons.get(reason, 0) + 1
                        )
    finally:
        for handle in outputs.values():
            handle.close()

    accepted = sum(counts.values())
    manifest = {
        "schema_version": 1,
        "accepted": accepted,
        "target": args.target,
        "shortfall": max(0, args.target - accepted),
        "splits": counts,
        "accepted_finish_reasons": accepted_finish_reasons,
        "rejected": rejected,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    if accepted < args.target:
        raise SystemExit(
            f"clean corpus has {accepted} examples, below target {args.target}"
        )


if __name__ == "__main__":
    main()
