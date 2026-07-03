#!/usr/bin/env python3
"""Aggregate trunk-argmax token IDs from `dflash_spec_demo --ar-baseline`
runner outputs and build a FastMTP-style top-K vocab sidecar.

Reads the `AR tokens: [...]` line written to stderr by each AR-baseline
invocation, counts token-id frequencies across all runs, and emits a
sidecar JSON in the same schema as `scripts/build_mtp_vocab_sidecar.py`
(drop-in for `mtp_extract --vocab-sidecar`).

This is the v2 sidecar generator — much higher τ ceiling than the v1
input-corpus version because the frequency map reflects the trunk's
ACTUAL emit distribution (FastMTP self-distillation pattern).

Usage:
  python3 scripts/distill/aggregate_argmax.py \
    --output-dir /tmp/distill_outputs \
    --tokenizer ~/.cache/.../models--Qwen--Qwen3.5-27B/snapshots/<...>/ \
    --sidecar-out qwen35_27b_trunk_argmax_v1.json \
    --top-k 32768
"""

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # special-token list still works without tokenizer

AR_TOKENS_RE = re.compile(r"^AR tokens:\s*\[([\d,\s]+)\]\s*$", re.MULTILINE)


def parse_token_ids(stderr_text: str) -> list[int] | None:
    m = AR_TOKENS_RE.search(stderr_text)
    if m is None:
        return None
    body = m.group(1).strip()
    if not body:
        return []
    try:
        return [int(x.strip()) for x in body.split(",") if x.strip()]
    except ValueError:
        return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True,
                   help="Directory of *.stderr.txt files from run_distill_parallel.sh")
    p.add_argument("--sidecar-out", required=True,
                   help="Output sidecar JSON path (drop-in for mtp_extract --vocab-sidecar)")
    p.add_argument("--top-k", type=int, default=32768)
    p.add_argument("--tokenizer", default=None,
                   help="Path to tokenizer.json directory (for special-token "
                        "must-include list). Optional; if absent, special "
                        "tokens are still inferred via counter heuristics.")
    p.add_argument("--full-vocab-size", type=int, default=248320,
                   help="Full trunk vocab size (default 248320 for Qwen3.5/3.6). "
                        "Used for the metadata field; pad-fill if top-k > observed.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"ERROR: output dir not found: {out_dir}", file=sys.stderr)
        return 1

    stderr_files = sorted(out_dir.glob("prompt_*.stderr.txt"))
    if not stderr_files:
        print(f"ERROR: no prompt_*.stderr.txt in {out_dir}", file=sys.stderr)
        return 1

    counter: Counter[int] = Counter()
    n_parsed = 0
    n_failed = 0
    total_tokens = 0
    per_file_token_counts: list[int] = []

    for sf in stderr_files:
        text = sf.read_text(errors="replace")
        ids = parse_token_ids(text)
        if ids is None:
            n_failed += 1
            continue
        counter.update(ids)
        per_file_token_counts.append(len(ids))
        total_tokens += len(ids)
        n_parsed += 1

    if n_parsed == 0:
        print(f"ERROR: no parseable AR tokens in {len(stderr_files)} stderr files",
              file=sys.stderr)
        return 1

    unique_observed = len(counter)
    print(f"parsed {n_parsed}/{len(stderr_files)} files "
          f"({n_failed} failed)", file=sys.stderr)
    print(f"  total tokens emitted: {total_tokens}", file=sys.stderr)
    print(f"  unique tokens observed: {unique_observed}", file=sys.stderr)
    if per_file_token_counts:
        avg = total_tokens / len(per_file_token_counts)
        mn = min(per_file_token_counts)
        mx = max(per_file_token_counts)
        print(f"  per-file: avg={avg:.1f}  min={mn}  max={mx}", file=sys.stderr)

    most_common = counter.most_common(args.top_k)

    must_include: list[int] = []
    if args.tokenizer is not None and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
            for sp_attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
                sp = getattr(tok, sp_attr, None)
                if isinstance(sp, int) and sp >= 0:
                    must_include.append(sp)
            if hasattr(tok, "all_special_ids"):
                must_include.extend(int(i) for i in tok.all_special_ids)
        except Exception as e:
            print(f"  tokenizer load failed: {e}; skipping must-include",
                  file=sys.stderr)
    must_include = sorted(set(must_include))

    selected = [tid for tid, _ in most_common]
    selected_set = set(selected)
    for tid in must_include:
        if tid not in selected_set:
            selected.append(tid)
            selected_set.add(tid)

    if len(selected) < args.top_k:
        for t in range(args.full_vocab_size):
            if t not in selected_set:
                selected.append(t)
                selected_set.add(t)
                if len(selected) == args.top_k:
                    break

    if len(selected) > args.top_k:
        selected = selected[: args.top_k]

    assert len(selected) == args.top_k, f"selection size {len(selected)} != {args.top_k}"

    covered = sum(counter[t] for t in selected if t in counter)
    coverage = covered / total_tokens if total_tokens > 0 else 0.0
    print(f"top-{args.top_k} covers {coverage*100:.2f}% of emitted tokens",
          file=sys.stderr)

    out = {
        "draft_to_full": selected,
        "compressed_vocab_size": args.top_k,
        "full_vocab_size": args.full_vocab_size,
        "stats": {
            "source": "trunk-argmax distillation v2 (FastMTP self-distill pattern)",
            "n_runs_parsed": n_parsed,
            "n_runs_failed": n_failed,
            "total_tokens_emitted": total_tokens,
            "unique_tokens_observed": unique_observed,
            "coverage_top_k": coverage,
            "must_include_specials": must_include,
        },
    }

    out_path = Path(args.sidecar_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
