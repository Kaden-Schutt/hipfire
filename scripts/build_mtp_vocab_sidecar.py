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

Custom corpus (recommended for a real workload): generate assistant outputs from
a genuine Qwen3.6-27B endpoint with scripts/dump_corpus_openai.py, then:
    build_mtp_vocab_sidecar.py --tokenizer <hf-dir> --output cvs.json \
        --top-k 32768 --no-default-corpus --corpus-jsonl 'corpus/*.jsonl'
Only the assistant OUTPUT side is counted (that's what the draft head predicts);
all special/added tokens (chatml frame, <think>, <tool_call>) are force-included.

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
import glob
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


def _extract_output_text(obj: dict) -> str:
    """Pull the ASSISTANT-side text from a dumped JSONL record. Prefers a
    pre-assembled `output` field (written by dump_corpus_openai.py); otherwise
    assembles reasoning + content + serialized tool_calls. The user prompt is
    never counted — the draft head predicts the model's output distribution."""
    if isinstance(obj.get("output"), str) and obj["output"]:
        return obj["output"]
    parts: list[str] = []
    for k in ("reasoning_content", "reasoning", "content", "response", "text"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    tcs = obj.get("tool_calls")
    if isinstance(tcs, list):
        parts.extend(json.dumps(tc, ensure_ascii=False) for tc in tcs)
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                if isinstance(m.get("content"), str):
                    parts.append(m["content"])
                for tc in (m.get("tool_calls") or []):
                    parts.append(json.dumps(tc, ensure_ascii=False))
    return "\n".join(parts)


def load_custom_corpus(jsonl_globs, corpus_dir) -> list[tuple[str, str]]:
    """Load a real corpus from dumped JSONL (assistant outputs) and/or a dir of
    *.jsonl / *.txt files. JSONL records are reduced to assistant-output text."""
    corpus: list[tuple[str, str]] = []
    jsonl_paths: list[str] = []
    for pat in (jsonl_globs or []):
        jsonl_paths.extend(sorted(glob.glob(pat)))
    txt_paths: list[Path] = []
    if corpus_dir:
        d = Path(corpus_dir)
        jsonl_paths.extend(str(p) for p in sorted(d.glob("**/*.jsonl")))
        txt_paths.extend(sorted(d.glob("**/*.txt")))
    for fp in jsonl_paths:
        chunks: list[str] = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = _extract_output_text(obj)
                if t:
                    chunks.append(t)
        if chunks:
            corpus.append((f"jsonl:{Path(fp).name}({len(chunks)}rec)", "\n".join(chunks)))
    for tp in txt_paths:
        corpus.append((f"txt:{tp.name}", tp.read_text(encoding="utf-8")))
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
    p.add_argument("--corpus-jsonl", nargs="+", default=None,
                   help="One or more JSONL files (or globs) of dumped assistant "
                        "outputs (see scripts/dump_corpus_openai.py). Assistant "
                        "OUTPUT text is counted, not user prompts.")
    p.add_argument("--corpus-dir", default=None,
                   help="Directory scanned recursively for *.jsonl (assistant "
                        "dumps) and *.txt (raw text) corpus files.")
    p.add_argument("--no-default-corpus", action="store_true",
                   help="Skip the small built-in seed corpus; use only "
                        "--corpus-jsonl / --corpus-dir. Use once you have a real "
                        "workload corpus so the tiny seed set doesn't dilute it.")
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
    corpus: list[tuple[str, str]] = []
    if not args.no_default_corpus:
        corpus.extend(gather_corpus(prompt_dir, repo_root))
    corpus.extend(load_custom_corpus(args.corpus_jsonl, args.corpus_dir))
    if not corpus:
        sys.stderr.write("ERROR: no corpus files found (default corpus disabled "
                         "and no --corpus-jsonl/--corpus-dir matched)\n")
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
    # Force-include ALL added tokens (chatml frame + <think>/<tool_call> markers,
    # etc.) — high-frequency structural tokens in agentic/chat output that must
    # be in the draft K-set or the head can never propose them.
    if hasattr(tokenizer, "added_tokens_encoder"):
        must_include.extend(int(v) for v in tokenizer.added_tokens_encoder.values()
                            if isinstance(v, int) and 0 <= v < full_vocab_size)
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
            "used_default_corpus": not args.no_default_corpus,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
