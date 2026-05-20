# Eval slice — wikitext-2 train, 1024 sequences × 2048 context

The frozen prompt bytes used by every quant-quality eval. Committed
directly so all comparable cohorts share byte-identical inputs (per
CLAUDE.md's "Prompt-structure τ sensitivity" rule, generalized:
shared bytes = comparable numbers).

## Files

- `make_slice.sh` — deterministic generator. Fetches wikitext-2 train,
  writes ~10 MB of concatenated text into
  `wikitext2-1024s-2048ctx.txt`. Records md5 in `slice.md5`. Run once
  when first standing up the eval; output committed. Uses
  `.venv/bin/python3` (project venv, not pip --user).
- `wikitext2-1024s-2048ctx.txt` — the slice. ~10 MB. **Generated; do
  not edit by hand.**
- `slice.md5` — md5 tripwire (`83b0205a304bf4e52172ecdb05f2e895`).
  `build_kld_ref` and `eval_gguf` md5sum the slice against this
  sibling and abort on mismatch (M4 of the consolidated review).

There is no `tokens.bin` in this directory and there isn't expected
to be one. Step 1.5 (tokenizer-parity, ran 2026-05-08) measured a
45.9% structural divergence between hipfire's HF-Qwen BPE and
llama.cpp's GGUF-bundled BPE on this slice. **The divergence does
not block the eval pipeline:** `eval_hipfire` reads token IDs from
the reference file (written by llama-perplexity during the BF16 ref
dump), and `eval_gguf` consumes its candidate's tokens from the same
llama-perplexity tokenization. Neither re-tokenizes the slice text
itself. See plan §"Step 1.5 verdict" and §"GGUF anchor architecture
(rev-3.3)" for the reasoning.

## Why wikitext-2 train, not test?

Wikitext-2 test is ~245K tokens — far short of the 2.1M tokens the
matrix needs (1024 × 2048). Train is ~2.5M tokens, enough with no
overlap. Using train for eval is unconventional but acceptable here:
- We're evaluating quants of pretrained models — the models have
  presumably seen wikitext-2 train during pretraining (most large
  open models scrape Wikipedia), so PPLs / KLDs measure
  *quantization-induced perturbation*, not generalization.
- We need the same slice across every (model, variant, arch) to make
  numbers comparable; the eval is a cross-quant comparison, not a
  generalization claim.

## Why not wikitext-103?

Wikitext-103 has more text but uses a different vocabulary
construction; cross-comparison with published WT2 PPLs (which exist
for many models) is harder. WT2-train + the matrix's slice size is
the sweet spot.

## Reproducibility caveat

Once `make_slice.sh` runs and `wikitext2-1024s-2048ctx.txt` lands in
git, the slice is byte-stable and the md5 tripwire catches drift. If
the upstream HF wikitext-2 corpus changes, `make_slice.sh` re-run
would produce different bytes — the md5 mismatch surfaces this.
**The slice text in git is the source of truth**; the script is the
recipe, not the canonical artifact.

PR #115's predecessor corpus (`dev/bench/data/wikitext2-test.txt`) is
not in git and produces unreproducible historical PPLs. This new
slice resets the comparable cohort.
