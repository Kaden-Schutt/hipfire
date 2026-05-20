# Critical review — KLD eval branch (chore/113-quant-eval-plan)

**Reviewer:** claude-opus-4-7
**Date:** 2026-05-11
**Scope:** all commits on this branch since `upstream/master`, plus the PRD rewrite.
**Status:** scaffolding. Per `feedback_drop_review_files_after_fold_in.md`, delete after fold-in.

---

## Sibling-review action table (2026-05-11)

Cross-references the glm-5 review at `quant-code-rev-glm5.md`. Each finding marked V (validated + fixed in this branch), V-D (validated + deferred), or R (rejected).

| # | Severity | Finding | Status | Fix landed in |
|---|---|---|---|---|
| 2.1 | HIGH | `scored_per_chunk` shadow | V | eval_hipfire.rs:248 dedup |
| 2.2 | HIGH | Negative-KLD clamp masks bugs | V | `.max(0.0)` + `debug_assert!(>= -1e-9)` in eval_hipfire.rs + eval_gguf.rs |
| 2.3 | MED | `unsafe set_var` UB risk | V-D | env writes run before any thread spawn; ergonomic fix needs API restructure |
| 2.4 | MED | FIFO `/tmp` PID collision | V-D | low-likelihood; defer to a `tempfile`-crate refactor |
| 2.5 | MED | Commit prefix `starts_with` allows drift | V | Equal-length prefix compare in `eval_common::verify_llama_commit`; demand ≥ 7 chars |
| 3.1 | HIGH | 5 copy-pasted helpers across examples | V | Extracted to `crates/hipfire-runtime/src/eval_common.rs`; all three examples call into it |
| 3.2 | MED | Hand-rolled argv parsing | V-D | Invasive; clap migration is a separate refactor |
| 3.3 | MED | `score_position` mutates captured state | V-D | Same finding as claude H1; cosmetic refactor deferred (works correctly) |
| 3.4 | LOW | `tokenizer_parity.py` writes to `/tmp` | R | AGENTS.md rule #4 is about canonical *prompts*, not intermediate scratch |
| 4.1 | HIGH | Top-K=256 bias is variant-dependent | V | PRD §2 + §8 caveat preamble updated; all KLD now framed as lower-bound |
| 4.2 | MED | Mixed-mode rows in same table | V | `kld_reduce.py` emits mode-collision warning + groups by mode |
| 4.3 | MED | Canary fixture all TBD | V | PRD §10 V2 now explicitly informational-not-enforced |
| 4.4 | LOW | Slice size discrepancy | R | Misread of `git diff --stat` (shows lines, not bytes); actual slice is 10.5 MB |
| 5.3 | MED | PRD references deleted plan | V | Last source-code docstring ref updated to point to PRD §5 |
| 5.4 | MED | Deferred list lacks dates/owners | V-D | PRD stylistic; rev in next pass when matrix gets actively scheduled again |
| 6.1 | LOW | External `sha256sum`/`md5sum` | V-D | Linux-only fine for now; `sha2` crate migration is a separate refactor |
| 6.2 | LOW | Inline Python in shell scripts | V-D | Cosmetic |

Plus from my own review:
| # | Severity | Finding | Status |
|---|---|---|---|
| C-H2 | HIGH | `kld_diff.py` 90% one-sided heuristic missed canonical V1 | V — replaced with binomial sign-test (p < 0.001); confirmed firing at p=6.59e-101 on gfx1100 MQ4 |
| C-H3 | HIGH | PRD §13 referenced `/tmp/smoke/` (now gone) | V — citation removed |
| C-M3 | MED | `--max-chunks` progress shows `n_chunk` not `effective_n_chunk` | V — fixed in both branches of the chunk loop |

Findings carried forward to follow-up work:
- claude H1 (allocator churn in `score_position`) — deferred; perf invisible on full slice with ~1h prefill wall-clock.
- claude H4 (HFP4/MFP4 readiness asserted not validated) — deferred; depends on .hfq files being produced.
- claude H5 (V1 SOFT FAIL didn't trigger V0 microtests; methodology drift) — the sign-test fix to `kld_diff.py` now correctly fires HARD FAIL with a clear "expected for prefill-vs-per-token; investigate via V0 if you see this on same-mode A/B" verdict, so the methodology change is now self-documenting.
- claude M1 (prefill-is-more-accurate framing) — PRD §5.3 unchanged; soften in next pass.
- claude M5 (Q8 never V1-validated) — deferred; needs Q8 .hfq + a re-run.
- claude M6 (no unit tests for KLD math / kld_diff gate logic) — separate PR.
- All claude L items — opportunistic.

Severity ladder: **C** (blocks merge) · **H** (must-fix before next data run) · **M** (resolve in rev-2 of PRD or in follow-up) · **L** (note, OK to wave through).

The work is substantively sound — the eval pipeline is real, the data is committed, the PRD is honest about scope. The issues below are the things I'd want fixed before treating this as a permanent reference, not blockers to merging the PR.

---

## High findings

### H1 — Allocator churn in `score_position` closure (eval_hipfire.rs:272-332)

`score_position` is called 1,202,025 times per full-slice run (1175 chunks × 1023 scored positions). On each call it allocates two `Vec<u32>` and `Vec<f32>` of capacity 256, fills them by parsing the ref block, then drops them. That's ~600 MB of allocator churn across a run, plus an ``Vec::with_capacity`` + 256 push'es overhead per call.

Per-call cost is probably <10 µs so not catastrophic, but the closure body is reading bytes that already exist in `block_buf` — there's no reason to allocate. Use `bytemuck::cast_slice::<u8, u32>` (or a manual unaligned load loop) to view the block buffer directly. Or hoist a single pair of reusable `Vec`s to outer scope, clear-and-fill.

For a full-slice run at the new ~1 h prefill wall-clock, the closure's overhead is invisible. For the partial-canary-fast iteration loops, it matters more — and the bug is the same in both paths.

### H2 — `kld_diff.py` one-sided heuristic mis-calibrated (kld_diff.py:117)

The "systematic bias" alarm fires when `> 90%` of per-seq deltas are one-signed. On the canonical V1 gfx1100 MQ4 result (mean delta −6.75%, Pearson 0.949, CIs do NOT overlap) the verdict was `SOFT FAIL — no systematic bias detected`. That's wrong: the bias is statistically definitive but the one-signed fraction landed somewhere in 80–90%, so the heuristic didn't fire.

The plan's escalation path uses this verdict to decide whether to root-cause via V0 microtests. If the heuristic misses real biases at this magnitude, the escalation tree is partially broken. Either:

- Lower the threshold to 80% (more sensitive but more false positives on noise-dominated runs).
- Replace with a sign-test p-value: under no-bias H0, sign-count follows Binomial(n, 0.5); for n=1175 a 9:1 split is wildly significant. Use a proper one-sample sign test or Wilcoxon.

Also, the `>` should probably be `>=` so 90.0% counts.

### H3 — PRD references `/tmp/smoke/` as authoritative

PRD §13 "Current State" and §10 V3 still reference "partial 9B-MQ3 + 9B-MQ4 50-chunk data lives in /tmp/smoke/" — but those files were not committed and `/tmp/smoke/` is session-local on gfx1151. A reviewer reading the PRD will see broken citations.

Fix: either commit the 50-chunk data under `results/2026-05-11/per-seq/` with a `-50c` suffix in the variant name (so it's a labeled-as-partial sample), or remove the `/tmp/smoke/` reference and rephrase as "gfx1151 50-chunk smoke runs were exercised during V1 development but not archived; reproducible via `--max-chunks 50`."

### H4 — HFP4G32 / MFP4G32 readiness is asserted, not validated

The PRD §5.4 eligibility table and §6 matrix both treat HFP4G32 / MFP4G32 as eval-ready (auto-fallback to per-token). But there is no committed evidence that `eval_hipfire` actually completes a run on an .hfq file of those types. The auto-fallback path in `forward_prefill_batch` is exercised via `is_batchable_la` returning false, fine — but other parts of `eval_hipfire` could fail before that point:

- `Qwen35Scratch::new` allocates scratch sized for the model's config; if HFP4G32 introduces a new scratch path requirement it could panic at load.
- `KvCache::new_gpu_asym3` doesn't depend on weight dtype, OK.
- The lm_head row at `weights.output` is dispatched via `weight_gemv(...)` which has match arms for `DType::HFP4G32` and `DType::MFP4G32` (verified at llama.rs:565-572), good — but only if the model's lm_head is itself one of those types, which depends on what `hipfire-quantize --format hfp4` does to the output tensor.

Mitigation: add a "smoke target" to the harness README — once a 9B HFP4G32 .hfq exists, the first run should be 5-chunk `--scoring-mode prefill` to confirm the load path completes. Until then the PRD claim that HFP4/MFP4 "slot into the matrix" is unverified.

### H5 — V1 SOFT FAIL didn't trigger V0; the gate's purpose got rewritten in-flight

The PRD §10's V0 block says microtests trigger "if V1 on any variant reports a HARD FAIL or persistent SOFT FAIL." The gfx1100 MQ4 V1 result *did* SOFT FAIL by the tolerance schedule (every gate except CI-overlap failed: rel delta 6.75% vs 1% threshold, Pearson 0.949 vs 0.9999, p99 27.7% vs 5%). We should have run V0 microtests as the root-cause step.

We didn't. Instead, we re-interpreted V1 as a measurement, not a gate — by §5.3 "the gates intentionally fail … the diff tool is the diagnostic; the verdict is editorial." That's a defensible move, but it's a *change* to the validation methodology mid-stream. The PRD now reads coherently after the rewrite, but a reader doing the work fresh would follow §10's V0 trigger and run V0, only to find it's been removed.

Either: (a) explicitly document the methodology change ("V1 PASS/FAIL applied to mode-equivalence questions; for hipfire MQ vs prefill MQ the verdict is editorial because the modes are known-different and we're characterising the difference"), or (b) restore V0 as a real gate and run it. Recommend (a).

---

## Medium findings

### M1 — "Prefill is more accurate against BF16" is a hypothesis dressed as a finding

PRD §5.3 says prefill produces "logits systematically closer to BF16 ground truth." This is true *by definition of the test* — we measure prefill-KLD-vs-BF16-ref and per-token-KLD-vs-BF16-ref, prefill comes out lower, "closer to BF16." But the BF16 reference was produced by llama-perplexity's float kernel chain, which has its own multi-acc accumulation pattern. The bias could equally be "prefill happens to round in the direction llama-perplexity's float code does, and per-token's GEMV rounds the other way." That's not "more accurate"; it's "aligned by happenstance to the reference's roundoff."

To genuinely show prefill is more accurate, we'd need an *independent* ground truth — e.g., fp64 reference on a CPU implementation. We don't have that. The framing should soften: "prefill measures lower KLD than per-token against this BF16 reference; the direction is reproducible but whether it represents 'more accurate' inference is open until we cross-check against an independent oracle."

This is a tone-of-claim issue, not a load-bearing error. The Pivot decision still stands (prefill is faster and the measured numbers are what hipfire's prefill path actually emits). But the editorial recommendation in §5 should not lean on the "more accurate" claim.

### M2 — `scored_per_chunk` declared twice (eval_hipfire.rs:188, 248)

Same value (`n_ctx - 1 - n_ctx / 2`), the second shadows the first. Compile passes, semantics unchanged, but it's a stink. Remove the second declaration; line 248 should reference the earlier binding.

### M3 — `chunk` progress reports use `n_chunk` instead of `effective_n_chunk` (eval_hipfire.rs:372, 429)

When `--max-chunks 50` is set, the progress line still reads "chunk 5/1175" instead of "chunk 5/50". Cosmetic, but confusing for users of `--max-chunks`.

### M4 — `HIPFIRE_PREFILL_REUSE_PBS=1` ordering dependency

`Qwen35Scratch::new` reads `HIPFIRE_PREFILL_REUSE_PBS` *during construction* and only then allocates `PrefillBatchScratch`. In `eval_hipfire`, the env var is set before `Gpu::init` (good), then scratch is allocated (good). But this is an implicit ordering contract: if a future contributor reorders code to allocate scratch before the env-var block, `prefill_batch` stays `None`, every chunk pays alloc/free for ~25 tensors, run gets ~10% slower, no warning.

Add an `assert!(scratch.prefill_batch.is_some(), "PBS not pre-allocated; set HIPFIRE_PREFILL_REUSE_PBS=1 *before* Qwen35Scratch::new")` in the prefill mode branch. Compile-time cheap; catches the regression.

### M5 — Q8 was never V1-validated; the canonical-mode claim is unproven for the lossless case

The V1 measurement was only run on MQ4 (and partial on MQ3 / MQ3-Lloyd). The PRD treats prefill as canonical for ALL hipfire variants in §5, but Q8 — the lossless reference proxy where any kernel-path noise should be tiniest — has never had its prefill-vs-per-token bias measured. The 7% MQ-side bias might be 0.01% on Q8, or it might be 5% on Q8; we don't know.

This matters because Q8 is the cleanest case to argue "prefill is canonical." If Q8 shows a tiny bias (<0.1%), it supports the broader case; if Q8 shows a large bias, the canonical decision is suspect.

Suggested follow-up (not blocking): once an .hfq Q8 is on hand, a 50-chunk V1 A/B on gfx1100 takes ~25 min and resolves this.

### M6 — KLD math has no unit tests

The `score_position` closure's math is verified only by end-to-end runs (gfx1100 MQ4 prefill ≈ 0.817, eyeballed as "looks right"). A 10-line unit test with two hand-constructed distributions and a hand-computed expected KLD would lock the math down against future refactors. Same for the residual cross-term's edge cases (one residual zero, both zero, etc.).

`kld_diff.py` similarly has no unit tests beyond the self-diff smoke. The gate-decision logic for boundary cases (epsilon-equal-to-threshold, etc.) is untested.

### M7 — Storage & cost table mixes pre-Pivot and post-Pivot numbers (PRD §12)

The cost table lists "Hipfire 9B per-token: ~7 h/run × 10 = ~70 GPU-h" alongside "Hipfire 9B prefill: ~1 h/run × 10 = ~10 GPU-h" — the per-token line is historical, the prefill line is the canonical path. Reading the table linearly, a stakeholder might think hipfire 9B costs 80 GPU-h. It costs ~10 GPU-h (prefill canonical) + the 28 GPU-h already spent on the historical per-token data.

Restructure: split "future cost" (prefill canonical) from "historical cost already spent" (per-token rows committed).

---

## Low findings

### L1 — Residual cross-term silently skips when `sum_p_residual_cand <= 1e-9`

When the candidate believes all probability mass is concentrated in the reference's top-K (so `1 - Σ p_cand[ref_top_K] ≈ 0`), the cross-term is dropped. This is correct (the term goes to 0× anything-finite), but the asymmetric clamp introduces a tiny bias on near-deterministic positions toward "no residual penalty." Probably invisible in practice; document or tolerate.

### L2 — Negative-KLD clamp range (eval_hipfire.rs:324)

`if kld_token < 0.0 && kld_token > -1e-6 { kld_token = 0.0; }` clamps tiny fp roundoff. But KLD values below -1e-6 would survive untouched and propagate into the mean. KLD is mathematically ≥ 0; any negative value is roundoff. Better: clamp at 0 unconditionally, or `kld_token.max(0.0)`.

### L3 — `chunk_klds.clone()` per chunk for p99 sorting (eval_hipfire.rs:443)

Clones 8 KB per chunk for the sort. 10 MB churn over 1175 chunks. Sort in place with `chunk_klds.sort_by(...)` — order isn't needed after p99 is read, and the mean was already computed.

### L4 — `eval_gguf` H7 token-equality assumes byte-identical tokenization

`eval_gguf` aborts if the candidate GGUF's tokens (produced by llama-perplexity on the slice) don't byte-match the reference's tokens. This is correct for catching mismatched tokenizers, but the failure message at qwen35.rs's call site is "ERROR: cand n_vocab {cand} != ref n_vocab {ref}" — could be clearer about which side disagrees.

### L5 — `fetch-eval-refs.sh` Python version unspecified

The script requires `huggingface_hub`, which requires Python ≥ 3.8. On older systems the venv create succeeds but pip install or hf_hub_download fails with cryptic errors. Add a `python3 --version` check + `>= 3.8` guard.

### L6 — HF upload recipe is undocumented

The script downloads from HF but the corresponding upload step (`hf upload --repo-type dataset hipfire-models/qwen-kldref ...`) is not in any committed README. If anyone needs to upload a new ref, they have to reverse-engineer the recipe from the hf CLI docs. Add a §"Producing a new ref" subsection to the harness README.

### L7 — No regression test for HFKSEQ output format

`eval_hipfire` writes HFKSEQ v2 by hardcoding `version = 2` (eval_hipfire.rs:461). If the schema changes (e.g., a v3 adds median NLL), the writer must be updated in lockstep with the reader. No test enforces. Add a round-trip test: write a known HFKSEQ, read it back via `kldref_format.py`, assert per-seq values match.

### L8 — `prefill_microbench` has no oracle

The microbench measures dispatch time but doesn't check that `forward_prefill_batch` produced correct output. If a kernel regression returned zeros silently, the microbench would just measure "zero-write time." Add a single-position output comparison against `forward_scratch` on a synthetic input (5-line check, fp16 tolerance).

### L9 — Auto-mkdir on output path (eval_hipfire.rs:457-461, eval_gguf.rs:382, build_kld_ref.rs:165)

`create_dir_all` on the output's parent silently creates any missing chain. Saves the "lost run to typo" case the commit message describes, but if a user typos a path like `--output benchmark/quality-baseline/foo.kldseq` (singular instead of plural), they'll get a fresh tree created without warning. Probably fine since the run produces output regardless and the user can rm-rf the typo, but worth noting.

### L10 — Slice fixture has single point of failure

`benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt` is committed and pinned by md5 in `slice.md5`. If wikitext-2 ever changes (the upstream HF dataset), or if `make_slice.sh` is re-run on a new machine and produces a byte-different slice, every reference becomes invalid and the eval matrix collapses. Mitigation is implicit (the slice IS committed, the md5 IS checked) but the failure mode would be catastrophic and silent until the first eval run fails with a SHA mismatch. Worth a sentence in the PRD: "the committed slice + md5 + sha256 references form a triple-pin; do NOT regenerate `make_slice.sh` casually."

---

## Summary

The core eval pipeline works, the data flows are honest, and the PRD captures the substance well. The notable risks are:

1. **Methodology drift on V1.** We re-defined V1 from "equivalence gate" to "characterisation measurement" mid-stream. Document the choice or it'll trip the next reviewer.
2. **Bias-interpretation framing.** "Prefill is more accurate" is a charitable read of the V1 result; the careful read is "prefill is biased toward llama-perplexity's float roundoff direction." Soften.
3. **HFP4/MFP4 unverified.** PRD claims they slot in; nothing has actually run end-to-end on those formats.
4. **kld_diff.py systematic-bias detector mis-calibrated.** Missed the canonical V1 result; needs sensitivity bump.

None of the above blocks merge. They're the things to fix before the eval matrix is treated as a permanent reference for HFP4/MFP4 results.

Smaller polish (M2–M7, L1–L10) is opportunistic.
