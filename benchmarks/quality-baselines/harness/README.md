# Harness — quant-quality eval scripts

This directory holds the tools that build / consume / aggregate the
KLD references and produce the result tables.

## Files

| File | Purpose | Status |
|---|---|---|
| `manifest.json`         | SHA-pinned index of legacy BF16 reference dumps | legacy `.kldref.bin`; current local refs should be regenerated as `.pkld` via `perplexity --dump-ref` or as HFQM calibration bundles via `collect_artifacts --kldref` |
| `kldref_format.py`      | Reader/writer for the legacy hipfire HFKLDR ref format + HFKSEQ per-sequence sidecar (v2: adds mean_nll for PPL) | legacy compatibility |
| `kld_reduce.py`         | Bootstrap CI + result-table emitter (incl. PPL column) | done |
| `tokenizer_parity.py`   | Step 1.5 tokenizer-parity check (hipfire vs llama.cpp BPE) | done; ran 2026-05-08 — see plan §"Step 1.5 verdict" |
| `canary.md`             | 11-sequence harness-output reproducibility fixture | sequences populated; expected KLDs land after Step 5's first canary candidate |

The current local-HFQ quality path lives in
`crates/hipfire-runtime/examples/perplexity.rs`, with optional HFQM
calibration capture in `collect_artifacts.rs`. Older references to
`build_kld_ref.rs`, `eval_hipfire.rs`, and `eval_gguf.rs` are historical for
this checkout; those examples are not currently present.

## Reference fetch

`scripts/fetch-eval-refs.sh` (at repo root) reads `manifest.json` and
either verifies (if locally present) or downloads (if `.hf_repo` is
set) each legacy raw ref into `../refs/<name>`. Use this only for
historical compatibility checks. New local baseline-quality claims should use
GPU-generated refs from the local BF16 HFQ model:

```bash
cargo build --release --features deltanet -p hipfire-runtime \
  --example perplexity

target/release/examples/perplexity \
  ~/.hipfire/models/<model>-bf16.hfq \
  benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
  --ctx 2048 --warmup 8 --offset 0 --kv-mode fp32 --top-k 128 \
  --dump-ref ~/.hipfire/datasets/kldref/<model>-bf16.pkld
```

## How to add a new quant variant

1. Make sure the BF16 reference for the model exists. For the current
   examples executor, produce a `.pkld` ref with `perplexity --dump-ref` from
   the matching local BF16 HFQ artifact and store it under
   `~/.hipfire/datasets/kldref/` or next to the model. For HFQM calibration
   bundles, run `collect_artifacts --kldref`; that output is a calibration
   package, not the legacy standalone `.kldref.bin` format.

2. Run the candidate against the cached reference:

   - hipfire variants through the current eval battery:
     ```
     HIPFIRE_EVAL_PERPLEXITY_CTX=2048 \
     target/release/hipfire-eval \
       --model <path-to-hfq> \
       --battery perplexity --executor examples --kv-mode q8 \
       --kldref ~/.hipfire/datasets/kldref/<model>-bf16.pkld
     ```

   - GGUF anchor variants are historical in this checkout. Reintroduce or
     replace the GGUF evaluator before using legacy `.kldref.bin` anchors for
     new claims.

   Output filename convention: `<variant>__<arch>.kldseq` —
   `kld_reduce.py` parses `rsplit("__", 1)`.

3. Aggregate:

   ```
   python3 kld_reduce.py --result-dir ../results/<date>/per-seq/ \
                         --out-md   ../results/<date>/result-table.md \
                         --out-json ../results/<date>/result-data.json
   ```

4. Eyeball the markdown table; commit alongside the run's
   `2026-MM-DD-quant-pareto.md` write-up.

## Plan reference

`docs/plans/issue-113-quant-quality-eval.md` is the canonical PRD —
source of truth for binary format, eval matrix, scoring modes,
validation methodology, and pivot decisions.

## Pinned llama.cpp commit

`9dcf83552887bb898b4a98a5761361e504e31fc3` (master, 2026-05-08).

Historical GGUF producer/evaluator flows invoked `<bin> --version` before
spawning, parsed the parenthesized short hash, and asserted it was a prefix of
the pinned 40-char commit. If those flows are reintroduced, keep that guard; a
different llama.cpp build can change tokenizer or scoring behavior.
