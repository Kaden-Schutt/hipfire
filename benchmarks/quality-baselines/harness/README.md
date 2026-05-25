# Harness — quant-quality eval scripts

This directory holds the tools that build / consume / aggregate the
KLD references and produce the result tables.

## Files

| File | Purpose | Status |
|---|---|---|
| `manifest.json`         | SHA-pinned index of BF16 reference dumps | populated for 9B (qwen3.5); 27B entry pending Step 6 dump completion |
| `kldref_format.py`      | Reader/writer for the hipfire HFKLDR ref format + HFKSEQ per-sequence sidecar (v2: adds mean_nll for PPL) | done |
| `kld_reduce.py`         | Bootstrap CI + result-table emitter (incl. PPL column) | done |
| `tokenizer_parity.py`   | Step 1.5 tokenizer-parity check (hipfire vs llama.cpp BPE) | done; ran 2026-05-08 — see plan §"Step 1.5 verdict" |
| `canary.md`             | 11-sequence harness-output reproducibility fixture | sequences populated; expected KLDs land after Step 5's first canary candidate |

The corresponding Rust binaries live at
`crates/hipfire-runtime/examples/{build_kld_ref,eval_hipfire,eval_gguf,tokenize_slice}.rs`
and are gated by `required-features` on the example targets in
`hipfire-runtime/Cargo.toml`. Build with `cargo build --release
--features deltanet --example <name>` for the deltanet-requiring ones
(eval_hipfire, eval_gguf via the deltanet feature path indirectly —
build_kld_ref is GPU-free I/O only).

## Reference fetch

`scripts/fetch-eval-refs.sh` (at repo root) reads `manifest.json` and
either verifies (if locally present) or downloads (if `.hf_repo` is
set) each ref into `../refs/<name>`. Run before any eval if you don't
already have the ref locally.

## How to add a new quant variant

1. Make sure the BF16 reference for the model exists. If not, run
   `build_kld_ref` (see plan §"Reference dump methodology") on the
   gfx1151 host (only host with enough UMA RAM for 27B BF16). Upload
   to `hipfire-models/hipfire-eval-refs` and add an entry to
   `manifest.json` with `sha256`, `hf_repo`, `producer_cmd`,
   `llamacpp_commit`, `slice_md5`, and shape metadata.

2. Run the candidate against the cached reference:

   - hipfire variants:
     ```
     cargo run --release -p hipfire-runtime --example eval_hipfire \
       --features deltanet -- \
       --model <path-to-hfq> \
       --ref ../refs/<model>-bf16.kldref.bin \
       --output ../results/<date>/per-seq/<variant>__<arch>.kldseq \
       --kv-mode asym3
     ```

   - GGUF anchor variants:
     ```
     cargo run --release -p hipfire-runtime --example eval_gguf -- \
       --candidate-gguf <path-to-cand.gguf> \
       --ref ../refs/<model>-bf16.kldref.bin \
       --slice ../slice/wikitext2-1024s-2048ctx.txt \
       --output ../results/<date>/per-seq/<variant>__<arch>.kldseq \
       --llama-perplexity-bin <path-to-llama-perplexity>
     ```

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

`build_kld_ref` and `eval_gguf` both invoke `<bin> --version` before
spawning, parse the parenthesized short hash, and assert it's a
prefix of the pinned 40-char commit. If the user's llama.cpp build is
from a different commit, the format may have drifted; bail loudly.
