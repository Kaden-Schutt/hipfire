#!/usr/bin/env python3
"""Build a top-K token-frequency vocab sidecar for FastMTP-style MTP head compression.

Reads a representative corpus (canonical bench prompt + Python stdlib + small
English/code samples), tokenizes with the trunk's tokenizer, counts token-id
frequencies, and emits a JSON sidecar with the top-K most common token IDs.

The sidecar is consumed by `mtp_extract.rs --vocab-sidecar <path>` to build
a compressed `lm_head_draft` of shape [K, n_embd] that the MTP head dispatches
in place of the trunk's full [vocab, n_embd] head — ~7.7x BW reduction at K=32K
on a 248K-vocab Qwen3.5/3.6 model.

Verifier path is unchanged (uses trunk's full vocab head), so any out-of-vocab
draft proposal is automatically rejected by argmax mismatch — lossless greedy
preserved, just hurts τ if the corpus is unrepresentative.

This is the v1 sidecar generator: INPUT-CORPUS frequency only (no GPU needed,
runs on CPU in ~seconds). FastMTP's empirical decomposition shows vocab
compression alone is ~12% of their 2.03x lift — enough to validate the
architectural change. If v1 lands the BW reduction but τ is weak due to
coverage gaps, escalate to v2: trunk-argmax capture across a wide corpus
(parallel-friendly across 4x R9700 on hiptrx, ~100K-token argmax corpus
in well under an hour). v2 generator not yet implemented.

Output JSON schema:
    {
        "draft_to_full": [u32; K],          # draft idx -> full vocab idx
        "compressed_vocab_size": K,
        "full_vocab_size": V,
        "stats": {
            "corpus_files": [str, ...],
            "total_tokens": int,
            "unique_tokens": int,
            "coverage_top_k": float,        # fraction of corpus tokens covered by top-K
        }
    }
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.stderr.write("ERROR: transformers not installed. Run: pip install transformers\n")
    sys.exit(1)


def gather_corpus(prompt_dir: Path, repo_root: Path) -> list[tuple[str, str]]:
    """Return [(file_label, text), ...] for tokenization."""
    corpus: list[tuple[str, str]] = []

    canonical = prompt_dir / "lru_cache_pep8_strict.txt"
    if canonical.exists():
        corpus.append(("canonical_lru_pep8", canonical.read_text()))

    for he in sorted(prompt_dir.glob("humaneval_*.txt")):
        corpus.append((he.name, he.read_text()))

    for stdlib_name in ("functools.py", "collections/__init__.py", "heapq.py", "bisect.py"):
        stdlib_path = Path("/usr/lib/python3/dist-packages") / stdlib_name
        if not stdlib_path.exists():
            for py_root in ("/usr/lib/python3.12", "/usr/lib/python3.11", "/usr/lib/python3.10"):
                cand = Path(py_root) / stdlib_name
                if cand.exists():
                    stdlib_path = cand
                    break
        if stdlib_path.exists():
            corpus.append((f"stdlib_{stdlib_name}", stdlib_path.read_text()))

    for rust_name in ("crates/hipfire-runtime/src/lib.rs",
                      "crates/hipfire-arch-qwen35/src/qwen35.rs"):
        rust_path = repo_root / rust_name
        if rust_path.exists():
            text = rust_path.read_text()
            corpus.append((f"rust_{rust_name.split('/')[-1]}", text[:50_000]))

    english_samples = [
        ("english_short_1",
         "The quick brown fox jumps over the lazy dog. "
         "Pack my box with five dozen liquor jugs. "
         "How vexingly quick daft zebras jump."),
        ("english_explanation",
         "When you implement a least recently used cache, you typically combine "
         "a hash table for O(1) lookup with a doubly linked list to track recency. "
         "Each access promotes the node to the head; eviction removes the tail. "
         "This gives constant time for both get and put operations."),
    ]
    corpus.extend(english_samples)

    return corpus


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True,
                   help="Path to tokenizer.json directory (HF format)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--top-k", type=int, default=32768,
                   help="Top-K most frequent token IDs to keep (default 32768)")
    p.add_argument("--prompt-dir", default="benchmarks/prompts",
                   help="Directory containing canonical bench prompts")
    p.add_argument("--repo-root", default=".",
                   help="Repo root (for additional corpus files)")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    full_vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "added_tokens_encoder"):
        full_vocab_size = max(full_vocab_size,
                              max(tokenizer.added_tokens_encoder.values(), default=0) + 1)
    print(f"tokenizer vocab size: {full_vocab_size}", file=sys.stderr)

    if args.top_k > full_vocab_size:
        sys.stderr.write(f"ERROR: top-k {args.top_k} exceeds vocab {full_vocab_size}\n")
        return 1

    prompt_dir = Path(args.prompt_dir)
    repo_root = Path(args.repo_root)
    corpus = gather_corpus(prompt_dir, repo_root)
    if not corpus:
        sys.stderr.write("ERROR: no corpus files found\n")
        return 1

    counter: Counter[int] = Counter()
    files_used: list[str] = []
    for label, text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(ids)
        files_used.append(f"{label}:{len(ids)}toks")
        print(f"  {label}: {len(ids)} tokens", file=sys.stderr)

    total = sum(counter.values())
    unique = len(counter)
    print(f"corpus total: {total} tokens, {unique} unique", file=sys.stderr)

    most_common = counter.most_common(args.top_k)

    must_include = []
    for sp_attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        sp = getattr(tokenizer, sp_attr, None)
        if isinstance(sp, int) and sp >= 0:
            must_include.append(sp)
    if hasattr(tokenizer, "all_special_ids"):
        must_include.extend(int(i) for i in tokenizer.all_special_ids)
    must_include = sorted(set(must_include))

    selected_ids = set(tid for tid, _ in most_common)
    for tid in must_include:
        selected_ids.add(tid)

    if len(selected_ids) > args.top_k:
        ranked_present = [tid for tid, _ in most_common if tid in selected_ids]
        ranked_present_set = set(ranked_present)
        keep = list(ranked_present)
        for tid in must_include:
            if tid not in ranked_present_set:
                keep.append(tid)
        ranked = sorted(set(keep), key=lambda t: (-counter.get(t, 0), t))[:args.top_k]
        for tid in must_include:
            if tid not in ranked:
                ranked = ranked[:-1] + [tid]
        selected_ids = ranked
    else:
        ranked_full = [tid for tid, _ in most_common]
        for tid in must_include:
            if tid not in ranked_full:
                ranked_full.insert(0, tid)
        if len(ranked_full) < args.top_k:
            unused = sorted(t for t in range(full_vocab_size) if t not in selected_ids)
            ranked_full.extend(unused[: args.top_k - len(ranked_full)])
        selected_ids = ranked_full[: args.top_k]

    assert len(selected_ids) == args.top_k, \
        f"selection size {len(selected_ids)} != top-k {args.top_k}"

    covered = sum(counter[tid] for tid in selected_ids if tid in counter)
    coverage = covered / total if total > 0 else 0.0
    print(f"top-{args.top_k} covers {coverage*100:.2f}% of corpus tokens",
          file=sys.stderr)

    out = {
        "draft_to_full": selected_ids,
        "compressed_vocab_size": args.top_k,
        "full_vocab_size": full_vocab_size,
        "stats": {
            "corpus_files": files_used,
            "total_tokens": total,
            "unique_tokens": unique,
            "coverage_top_k": coverage,
            "must_include_specials": must_include,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
