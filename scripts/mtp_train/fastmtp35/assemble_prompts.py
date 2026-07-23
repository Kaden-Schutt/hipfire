#!/usr/bin/env python3
"""Assemble pretokenized, globally deduplicated FastMTP distillation jobs.

This script streams prompt sources from Hugging Face. Source responses are
discarded. The emitted JSONL is accepted directly by qwen35_batch_generate and
is sorted by exact prompt-token length to unlock equal-length batched prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import xxhash
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer


BUCKET_ORDER = ("short", "medium", "long")
PROFILE_ORDER = ("serve", "fastmtp", "greedy")
EVAL_MARKERS = (
    "livecodebench",
    "math-500",
    "mt-bench",
    "cnn/daily mail",
    "c-eval",
    "natural questions benchmark",
)


def hash64(value: str) -> int:
    return xxhash.xxh3_64_intdigest(value.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here / "corpus.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale every source count for a smoke run; 1.0 emits 440K rows",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=Path,
        help="Optional Hugging Face cache root; local cached files are preferred",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    parser.add_argument("--keep-sqlite", action="store_true")
    return parser.parse_args()


def allocate(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight / sum(weights) for weight in weights]
    out = [int(value) for value in raw]
    for _, index in sorted(
        ((raw[i] - out[i], i) for i in range(len(raw))), reverse=True
    )[: total - sum(out)]:
        out[index] += 1
    return out


def canonical_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        re.sub(r"\s+", " ", str(message.get("content", ""))).strip().lower()
        for message in messages
    ).strip()


def bottom_k_signature(text: str, k: int = 32) -> tuple[int, ...]:
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < 5:
        return (hash64(text),)
    count = len(words) - 4
    stride = max(1, count // 512)
    heap: list[int] = []
    seen: set[int] = set()
    for index in range(0, count, stride):
        value = hash64("\x1f".join(words[index : index + 5]))
        if value in seen:
            continue
        seen.add(value)
        if len(heap) < k:
            heapq.heappush(heap, -value)
        elif value < -heap[0]:
            removed = -heapq.heapreplace(heap, -value)
            seen.discard(removed)
    return tuple(sorted(-value for value in heap))


class GlobalDeduper:
    """Exact + bottom-k MinHash LSH deduper with bounded candidate checks."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self.exact: set[int] = set()
        self.signatures: list[frozenset[int]] = []
        self.bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    def accept(self, text: str) -> bool:
        exact = hash64(text)
        if exact in self.exact:
            return False
        signature = bottom_k_signature(text)
        sig_set = frozenset(signature)
        candidate_ids: set[int] = set()
        for band_index in range(0, len(signature), 4):
            key = (band_index // 4, signature[band_index : band_index + 4])
            candidate_ids.update(self.bands.get(key, ())[-32:])
        for candidate_id in candidate_ids:
            other = self.signatures[candidate_id]
            union = len(sig_set | other)
            if union and len(sig_set & other) / union >= self.threshold:
                return False
        row_id = len(self.signatures)
        self.exact.add(exact)
        self.signatures.append(sig_set)
        for band_index in range(0, len(signature), 4):
            key = (band_index // 4, signature[band_index : band_index + 4])
            self.bands[key].append(row_id)
        return True


def strip_final_assistant(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [
        {
            "role": {
                "human": "user",
                "gpt": "assistant",
                "tool_response": "tool",
            }.get(str(message.get("role", message.get("from", ""))), str(message.get("role", message.get("from", "")))),
            "content": str(message.get("content", message.get("value", ""))),
        }
        for message in messages
        if message.get("content", message.get("value", ""))
    ]
    while normalized and normalized[-1]["role"] == "assistant":
        normalized.pop()
    return normalized


def adapt_row(
    source: dict[str, Any], row: dict[str, Any], row_index: int
) -> tuple[list[dict[str, str]], str, dict[str, Any]] | None:
    name = source["name"]
    attrs: dict[str, Any] = {}
    if name == "ultrachat":
        source_id = str(row.get("prompt_id") or row_index)
        multiturn_threshold = round(
            float(source.get("multiturn_fraction", 1.0)) * (1 << 64)
        )
        if hash64(source_id) < multiturn_threshold:
            messages = strip_final_assistant(row.get("messages") or [])
            attrs["prompt_shape"] = "multiturn_prefix"
        else:
            messages = [{"role": "user", "content": str(row.get("prompt", ""))}]
            attrs["prompt_shape"] = "single_turn"
    elif name == "openmath":
        messages = [{"role": "user", "content": str(row.get("problem", ""))}]
        source_id = f"{row.get('problem_source', 'unknown')}:{row_index}"
        attrs["problem_source"] = row.get("problem_source")
    elif name == "opencode":
        messages = [{"role": "user", "content": str(row.get("input", ""))}]
        source_id = str(row.get("id") or row_index)
        attrs["domain"] = row.get("domain")
    elif name == "chinese_r1":
        if int(row.get("score") or 0) < 6:
            return None
        messages = [{"role": "user", "content": str(row.get("input", ""))}]
        source_id = f"{row.get('repo_name', 'unknown')}:{row_index}"
        attrs["score"] = row.get("score")
    elif name == "aya":
        messages = [{"role": "user", "content": str(row.get("inputs", ""))}]
        source_id = str(row.get("user_id") or row_index)
        attrs["language_code"] = row.get("language_code")
    elif name == "rag":
        documents = row.get("documents") or []
        context = "\n\n".join(
            f"[Document {index + 1}]\n{document}"
            for index, document in enumerate(documents)
        )
        content = (
            f"Question:\n{row.get('question', '')}\n\n"
            "Use the following documents to answer the question. "
            "If the documents do not support a claim, say so.\n\n"
            f"{context}"
        )
        messages = [{"role": "user", "content": content}]
        source_id = str(row_index)
        attrs["document_count"] = len(documents)
    elif name == "hermes_agent":
        # The trace's system message embeds a large, Hermes-specific tool
        # catalog and harness contract. Retain the actual task, then let the
        # deployed trunk produce a clean response for our runtime instead.
        messages = [{"role": "user", "content": str(row.get("task", ""))}]
        source_id = str(row.get("id") or row_index)
        attrs["category"] = row.get("category")
        attrs["subcategory"] = row.get("subcategory")
    else:
        raise ValueError(f"no adapter for {name}")
    if not messages or not any(message["role"] == "user" for message in messages):
        return None
    if not all(message["content"].strip() for message in messages):
        return None
    return messages, source_id, attrs


def stream_source(source: dict[str, Any], seed: int, buffer_size: int) -> Iterator[dict[str, Any]]:
    configs = [part.strip() for part in source["config"].split(",")]
    streams = [
        load_dataset(
            source["dataset"],
            config,
            split=source["split"],
            streaming=True,
        )
        for config in configs
    ]
    stream = streams[0] if len(streams) == 1 else concatenate_datasets(streams)
    yield from stream.shuffle(seed=seed, buffer_size=buffer_size)


def render_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": None,
    }
    try:
        tokens = tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        tokens = tokenizer.apply_chat_template(messages, **kwargs)
    # Transformers 5 returns a BatchEncoding here even without tensors, while
    # Transformers 4 returned the input-id list directly.
    if hasattr(tokens, "get"):
        input_ids = tokens.get("input_ids")
        if input_ids is not None:
            tokens = input_ids
    if tokens and hasattr(tokens[0], "ids"):
        tokens = tokens[0].ids
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def choose_remaining(
    remaining: dict[str, int],
    compatible: set[str],
    key: int,
) -> str | None:
    choices = [name for name in remaining if remaining[name] > 0 and name in compatible]
    if not choices:
        return None
    choices.sort(
        key=lambda name: (
            remaining[name],
            hash64(f"{key}:{name}"),
        ),
        reverse=True,
    )
    return choices[0]


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not (0 < args.scale <= 1):
        raise SystemExit("--scale must be in (0, 1]")
    config = tomllib.loads(args.config.read_text())
    seed = int(config["seed"])
    output = args.output.resolve()
    jobs_dir = output / "jobs"
    output.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.tokenizer_cache:
        tokenizer_kwargs["cache_dir"] = str(args.tokenizer_cache)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], **tokenizer_kwargs)

    sqlite_path = output / "assembly.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    database = sqlite3.connect(sqlite_path)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute(
        "CREATE TABLE jobs (profile TEXT, bucket TEXT, token_len INTEGER, payload TEXT)"
    )

    deduper = GlobalDeduper()
    accepted_by_source: dict[str, int] = {}
    rejected_by_source: dict[str, dict[str, int]] = {}
    target_total = 0

    profile_weights = [
        float(config["profiles"][profile]["fraction"]) for profile in PROFILE_ORDER
    ]

    for source_number, source in enumerate(config["source"]):
        name = source["name"]
        target = max(1, round(int(source["raw_count"]) * args.scale))
        target_total += target
        bucket_counts = allocate(target, [float(value) for value in source["bucket_mix"]])
        bucket_remaining = dict(zip(BUCKET_ORDER, bucket_counts))
        profile_remaining: dict[str, dict[str, int]] = {}
        for bucket, count in zip(BUCKET_ORDER, bucket_counts):
            profile_remaining[bucket] = dict(
                zip(PROFILE_ORDER, allocate(count, profile_weights))
            )

        accepted = 0
        rejected = defaultdict(int)
        for row_index, row in enumerate(
            stream_source(source, seed + source_number * 1009, args.shuffle_buffer)
        ):
            adapted = adapt_row(source, row, row_index)
            if adapted is None:
                rejected["adapter"] += 1
                continue
            messages, source_id, attrs = adapted
            canonical = canonical_text(messages)
            if len(canonical) < 16:
                rejected["too_short"] += 1
                continue
            if any(marker in canonical for marker in EVAL_MARKERS):
                rejected["eval_marker"] += 1
                continue
            if not deduper.accept(canonical):
                rejected["duplicate"] += 1
                continue
            try:
                tokens = render_tokens(tokenizer, messages)
            except Exception as error:
                rejection = f"tokenizer:{type(error).__name__}"
                rejected[rejection] += 1
                if rejected[rejection] <= 3:
                    print(
                        f"{name}: rejected row {row_index} after tokenizer "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            token_len = len(tokens)
            compatible = {
                bucket
                for bucket in BUCKET_ORDER
                if token_len <= int(config["buckets"][bucket]["max_prompt_tokens"])
            }
            stable_key = hash64(f"{name}:{source_id}")
            bucket = choose_remaining(bucket_remaining, compatible, stable_key)
            if bucket is None:
                rejected["length_or_bucket"] += 1
                continue
            profile = choose_remaining(
                profile_remaining[bucket], set(PROFILE_ORDER), stable_key ^ 0xA5A5A5A5
            )
            if profile is None:
                raise RuntimeError(f"internal profile quota exhausted for {name}/{bucket}")

            payload = {
                "id": {
                    "source": name,
                    "dataset": source["dataset"],
                    "source_id": source_id,
                    "category": source["category"],
                    "profile": profile,
                    "bucket": bucket,
                    "attributes": attrs,
                },
                "tokens": tokens,
                "max_new_tokens": int(config["buckets"][bucket]["max_new_tokens"]),
            }
            database.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?)",
                (profile, bucket, token_len, json.dumps(payload, ensure_ascii=False)),
            )
            bucket_remaining[bucket] -= 1
            profile_remaining[bucket][profile] -= 1
            accepted += 1
            if accepted % 1000 == 0:
                database.commit()
                print(
                    f"{name}: {accepted}/{target} accepted "
                    f"(seen={row_index + 1}, rejected={sum(rejected.values())})",
                    flush=True,
                )
            if accepted == target:
                break
        database.commit()
        if accepted != target:
            raise RuntimeError(
                f"{name}: source exhausted at {accepted}/{target}; "
                f"remaining buckets={bucket_remaining}, rejected={dict(rejected)}"
            )
        accepted_by_source[name] = accepted
        rejected_by_source[name] = dict(rejected)

    job_manifest: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        for profile in PROFILE_ORDER:
            path = jobs_dir / f"{bucket}-{profile}.jsonl"
            count = 0
            with path.open("w", encoding="utf-8") as handle:
                cursor = database.execute(
                    """
                    SELECT payload FROM jobs
                    WHERE bucket = ? AND profile = ?
                    ORDER BY token_len, payload
                    """,
                    (bucket, profile),
                )
                for (payload,) in cursor:
                    handle.write(payload)
                    handle.write("\n")
                    count += 1
            job_manifest[path.name] = {
                "rows": count,
                "sha256": sha256(path),
                "max_seq": int(config["buckets"][bucket]["max_seq"]),
                "max_new_tokens": int(config["buckets"][bucket]["max_new_tokens"]),
                "sampling": config["profiles"][profile],
            }

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "scale": args.scale,
        "git_head": git_head(),
        "tokenizer": config["tokenizer"],
        "target_rows": target_total,
        "accepted_rows": sum(accepted_by_source.values()),
        "sources": accepted_by_source,
        "rejections": rejected_by_source,
        "jobs": job_manifest,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    license_path = output / "LICENSES.md"
    with license_path.open("w", encoding="utf-8") as handle:
        handle.write("# Prompt-source license ledger\n\n")
        handle.write("Source responses are discarded; only prompts are retained.\n\n")
        handle.write("| Source | Dataset | License | Accepted prompts |\n")
        handle.write("|---|---|---:|---:|\n")
        for source in config["source"]:
            handle.write(
                f"| {source['name']} | [{source['dataset']}](https://huggingface.co/datasets/"
                f"{source['dataset']}) | {source['license']} | "
                f"{accepted_by_source[source['name']]} |\n"
            )

    if not args.keep_sqlite:
        database.close()
        sqlite_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(sqlite_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    else:
        database.close()

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Python 3.14 plus the HF/Arrow stack on hiptrx can leave native worker
    # pools blocked in interpreter teardown after every artifact is closed.
    # The assembler is a one-shot CLI, so bypass extension finalizers only
    # after a successful main() and explicit stream flush.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
