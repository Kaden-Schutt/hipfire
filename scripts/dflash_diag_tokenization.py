#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
dflash_diag_tokenization.py — Hermes diagnostic #0 (2026-04-19).

Verify that the DFlash training pipeline sees the same TOKEN IDs as the
inference pipeline for matching prompt text. A mismatch (e.g. <|im_start|>
tokenized as literal `<`, `|`, `im`, ..., `>` in training but as single
special-token ID 151644 at inference) would produce our exact symptom:
training loss converges cleanly on the wrong distribution; draft never learns
the distribution it needs at inference; τ stays near zero.

What it prints:
  1. Tokenizer meta (vocab_size, special tokens, bos/eos/pad).
  2. A sample of special-token round-trips (encode→decode).
  3. The first 500 token IDs of the training corpus, decoded back to text.
  4. The same prompt wrapped two ways (training-corpus style vs inference
     dflash_spec_demo style), with per-token side-by-side comparison.

Usage (no GPU needed):
  python3 scripts/dflash_diag_tokenization.py \
      --target-repo Qwen/Qwen3.5-4B \
      --corpus /root/calibration_corpus.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--corpus", default=None,
                   help="Path to calibration corpus. Skipped if absent.")
    p.add_argument("--sample-prompt",
                   default="Call the get_weather tool for Tokyo and summarize.",
                   help="Arbitrary prompt used to compare training vs inference token IDs.")
    p.add_argument("--n-dump", type=int, default=500,
                   help="Decode the first N token IDs from the corpus.")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.target_repo)

    print("=" * 72)
    print(f"Tokenizer: {type(tok).__name__} for {args.target_repo}")
    print(f"  vocab_size={tok.vocab_size}  model_max_length={tok.model_max_length}")
    print(f"  bos={tok.bos_token_id}({tok.bos_token!r})  "
          f"eos={tok.eos_token_id}({tok.eos_token!r})  "
          f"pad={tok.pad_token_id}")
    print(f"  added_tokens_decoder size: {len(tok.added_tokens_decoder)}")

    print("\n--- Special token round-trips -----------------------------------")
    for s in ["<|im_start|>", "<|im_end|>", "<|im_start|>user",
              "<|im_start|>assistant", "user", "assistant", "\n"]:
        ids = tok.encode(s, add_special_tokens=False)
        back = tok.decode(ids, skip_special_tokens=False)
        status = "✓ single id" if len(ids) == 1 else f"⚠ {len(ids)} tokens"
        print(f"  encode({s!r:32}) = {ids!r}  decode={back!r}  [{status}]")

    # ── Compare corpus-style wrap vs inference-style wrap ───────────────
    # The training corpus (fetch_calibration_corpus.sh) uses ChatML as:
    #     <|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n...
    # dflash_spec_demo's --chatml path (daemon.rs production path) builds:
    #     [<|im_start|>, "user", "\n", prompt_tokens, <|im_end|>, "\n",
    #      <|im_start|>, "assistant", "\n"]
    # The concatenated STRING form should tokenize to the same IDs as the
    # explicit-encode form — if not, training and inference see different
    # token ID streams for byte-identical prompt text.
    print("\n--- Training vs inference tokenization parity -------------------")
    prompt = args.sample_prompt

    train_str = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    train_ids = tok.encode(train_str, add_special_tokens=False)

    # Inference-style: encode each chunk separately and concatenate IDs.
    im_start = tok.encode("<|im_start|>", add_special_tokens=False)
    im_end = tok.encode("<|im_end|>", add_special_tokens=False)
    u = tok.encode("user", add_special_tokens=False)
    a = tok.encode("assistant", add_special_tokens=False)
    nl = tok.encode("\n", add_special_tokens=False)
    body = tok.encode(prompt, add_special_tokens=False)
    infer_ids = im_start + u + nl + body + im_end + nl + im_start + a + nl

    n = max(len(train_ids), len(infer_ids))
    print(f"training-style encode : {len(train_ids)} tokens")
    print(f"inference-style encode: {len(infer_ids)} tokens")
    if train_ids == infer_ids:
        print("✓ IDENTICAL — tokenization parity confirmed.")
    else:
        print("✗ DIVERGENT — tokenization mismatch between training and inference!")
        for i in range(n):
            t = train_ids[i] if i < len(train_ids) else None
            f = infer_ids[i] if i < len(infer_ids) else None
            marker = " " if t == f else "≠"
            t_str = repr(tok.decode([t], skip_special_tokens=False)) if t is not None else "—"
            f_str = repr(tok.decode([f], skip_special_tokens=False)) if f is not None else "—"
            print(f"  [{i:3d}] train={t!s:>7} {t_str:20}  infer={f!s:>7} {f_str:20}  {marker}")
            if i > 30:
                print(f"  ... ({n - i} more positions)")
                break

    # ── Dump first N tokens of corpus ──────────────────────────────────
    if args.corpus and Path(args.corpus).exists():
        print(f"\n--- First {args.n_dump} tokens of {args.corpus} ---------------")
        text = Path(args.corpus).read_text()
        docs = [d.strip() for d in text.split("\n\n") if d.strip()]
        print(f"corpus: {len(docs)} docs, first doc length {len(docs[0])} chars")

        # Replicate the training-loop's tokenize-then-flatten behavior.
        bos = tok.bos_token_id
        ids = []
        for d in docs[:10]:  # first 10 docs is plenty for 500 tokens
            if bos is not None:
                ids.append(bos)
            ids.extend(tok.encode(d, add_special_tokens=False))
            if len(ids) >= args.n_dump:
                break
        ids = ids[:args.n_dump]

        print(f"\ntoken IDs (first {args.n_dump}):")
        print(ids)

        print(f"\ndecoded text (skip_special_tokens=False):")
        decoded = tok.decode(ids, skip_special_tokens=False)
        # Preserve newlines so you can eyeball ChatML structure
        print(repr(decoded[:1500]) + ("..." if len(decoded) > 1500 else ""))

        # Scan for special-token usage in the corpus — ratio of IDs that are
        # "added" (special) vs "regular" text tokens.
        added_ids = set(tok.added_tokens_decoder.keys())
        n_special = sum(1 for i in ids if i in added_ids)
        print(f"\nspecial-token density in first {args.n_dump} corpus tokens: "
              f"{n_special}/{args.n_dump} = {n_special/len(ids):.1%}")
        if n_special == 0:
            print("⚠ NO special tokens in corpus prefix — ChatML may be tokenizing as literal text.")
        else:
            # Show which special tokens appear (most common first)
            from collections import Counter
            c = Counter(i for i in ids if i in added_ids)
            print(f"special tokens seen (top 5): "
                  + ", ".join(f"{tok.added_tokens_decoder[i]!r}×{n}" for i, n in c.most_common(5)))
    else:
        print(f"\n(corpus {args.corpus!r} not found — skipping corpus dump)")


if __name__ == "__main__":
    main()
