#!/usr/bin/env python3
"""Sample diverse prompts from local HuggingFace datasets for trunk-argmax
distillation (FastMTP-style v2 sidecar generator).

Reads JSONL blobs directly from `~/.cache/huggingface/hub/datasets--*` so it
runs without the `datasets` Python package. Writes one prompt per .txt file
into `--output-dir/`, numbered for the parallel runner to consume.

Default mix targets diverse coverage across reasoning/code/dialogue:
- Roman1111111-claude-opus-10000x: 60% (general assistant, broad domains)
- Jackrong-Qwen3.5-reasoning-700x: 20% (reasoning + code)
- nohurry-Opus-Reasoning-3000x:    20% (filtered reasoning)

The trunk's AR output on these prompts is what populates the v2 sidecar.

Hipfire is greedy temp=0 in this path, so the output is fully determined
by the prompts. Sampling diverse prompts -> diverse argmax output ->
better top-32K coverage.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_dataset_blob(dataset_name: str) -> Path | None:
    """Locate the JSONL blob for `datasets--<name>` cache dir."""
    cache_dir = HF_CACHE / f"datasets--{dataset_name.replace('/', '--')}"
    if not cache_dir.exists():
        return None
    blobs = list((cache_dir / "blobs").glob("*"))
    if not blobs:
        return None
    return max(blobs, key=lambda p: p.stat().st_size)


def load_jsonl(path: Path, max_rows: int | None = None) -> list[dict]:
    """Read any JSONL into a list of dicts. Schema-agnostic at this layer."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def extract_user_prompt(row: dict) -> str | None:
    """Pull a user prompt out of a row. Handles three on-disk schemas:
    1. `messages: [{role, content}, ...]`        (Roman1111111-opus)
    2. `conversation: [{from, value}, ...]`      (Jackrong-Qwen3.5-reasoning)
    3. `problem: "..."` (top-level flat field)   (nohurry-Opus-Reasoning-filtered)
    Returns the LAST user/human turn, or the flat field if present.
    """
    if "messages" in row and isinstance(row["messages"], list):
        text: str | None = None
        for m in row["messages"]:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    text = c
        if text:
            return text

    if "conversation" in row and isinstance(row["conversation"], list):
        text = None
        for m in row["conversation"]:
            if m.get("from") in ("human", "user"):
                v = m.get("value", "")
                if isinstance(v, str) and v.strip():
                    text = v
        if text:
            return text

    for key in ("problem", "prompt", "question", "input", "instruction"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v

    return None


def sample_from_dataset(
    dataset_name: str, n: int, rng: random.Random,
) -> list[tuple[str, str]]:
    """Return up to n (label, prompt_text) tuples sampled from the dataset."""
    blob = find_dataset_blob(dataset_name)
    if blob is None:
        print(f"WARNING: dataset '{dataset_name}' not found in HF cache", file=sys.stderr)
        return []
    rows = load_jsonl(blob, max_rows=None)
    print(f"  {dataset_name}: loaded {len(rows)} rows from {blob.name[:16]}",
          file=sys.stderr)
    prompts: list[tuple[str, str]] = []
    for r in rows:
        p = extract_user_prompt(r)
        if p is None:
            continue
        if len(p) < 30 or len(p) > 8000:
            continue
        prompts.append((dataset_name.split("/")[-1], p))
    if len(prompts) > n:
        prompts = rng.sample(prompts, n)
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write numbered prompt .txt files")
    ap.add_argument("--n-prompts", type=int, default=400,
                    help="Total number of prompts to sample (default 400)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", default="opus60,reasoning20,filtered20",
                    help="Comma-sep weights for dataset mix. Each token "
                         "is `<short>NN` where NN is %% (sums to 100). "
                         "Defaults to opus60,reasoning20,filtered20.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    DATASET_ALIASES = {
        "opus":      "Roman1111111/claude-opus-4.6-10000x",
        "reasoning": "Jackrong/Qwen3.5-reasoning-700x",
        "filtered":  "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "hermes":    "lambda/hermes-agent-reasoning-traces",
    }

    weights: dict[str, int] = {}
    for tok in args.mix.split(","):
        tok = tok.strip()
        for i, ch in enumerate(tok):
            if ch.isdigit():
                short, pct = tok[:i], int(tok[i:])
                weights[short] = pct
                break
    if sum(weights.values()) != 100:
        print(f"ERROR: weights must sum to 100, got {weights}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("prompt_*.txt"):
        old.unlink()

    all_prompts: list[tuple[str, str]] = []
    for short, pct in weights.items():
        if short not in DATASET_ALIASES:
            print(f"ERROR: unknown dataset alias '{short}'; valid: "
                  f"{list(DATASET_ALIASES)}", file=sys.stderr)
            return 1
        n = int(args.n_prompts * pct / 100)
        ds_name = DATASET_ALIASES[short]
        sampled = sample_from_dataset(ds_name, n, rng)
        all_prompts.extend(sampled)

    rng.shuffle(all_prompts)
    if len(all_prompts) > args.n_prompts:
        all_prompts = all_prompts[: args.n_prompts]

    manifest: list[dict] = []
    for i, (label, text) in enumerate(all_prompts):
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        path = out_dir / f"prompt_{i:04d}_{label}_{digest}.txt"
        path.write_text(text)
        manifest.append({
            "id": i,
            "label": label,
            "md5_8": digest,
            "path": str(path.relative_to(out_dir)),
            "char_len": len(text),
        })

    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "mix": args.mix,
        "n_prompts": len(all_prompts),
        "prompts": manifest,
    }, indent=2))

    print(f"\nwrote {len(all_prompts)} prompts to {out_dir}", file=sys.stderr)
    print(f"manifest: {out_dir / 'manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
