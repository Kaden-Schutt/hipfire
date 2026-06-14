# Quant Quality Eval (KLD-Primary) PRD

**Status:** Active (canonical reference)
**Last updated:** 2026-05-11
**Owner:** hipfire eval
**Tracking:** #113 (uniform MQ), #116 (Lloyd-MQ), #182 (MQ4-Lloyd)
**Pinned llama.cpp commit:** `9dcf83552887bb898b4a98a5761361e504e31fc3`

---

## 1. Executive Summary

A KLD-primary quality eval for hipfire quant formats against a BF16 reference, with a llama.cpp GGUF anchor track for cross-codebase calibration. KLD against BF16 measures the exact perturbation a quant introduces over the whole output distribution, where PPL — top-1 probability alone — would miss tail collapses that hurt DFlash speculative acceptance, RAG retrieval, factual recall, and sampling diversity.

**Deliverable:** a 2-axis Pareto plot (size GB on x, mean-KLD on y) with hipfire quant points (HFP4G32 / MFP4G32 as canonical going forward; MQ family rows retained as historical) and GGUF Q-family anchor points, bootstrap-CI error bars, p99-KLD shading. The result *table* below the plot is the reproducibility artifact; the plot is the user-facing message.

**Active focus (post 2026-05-11 Pivot):** HFP4G32 + MFP4G32 (PR #224 / #225) are the RDNA-optimal successors to the MQ family. The MQ family's coarse single-fp32-scale-per-256 grouping is structurally noisier than community K-quants by a wide margin (hipfire MQ4 KLD 0.876 vs vanilla Q4_K_M 0.125 at near-equivalent size on the 9B slice). Once `Qwen3.5-9B-HFP4G32.hfq` / `.MFP4G32.hfq` exist, they slot into the matrix and become the canonical hipfire-side rows. MQ rows are kept for historical continuity.

**Non-goal — KLD is not a ship gate.** Coherence-gate and perf-gate remain the gating signals for shipping a kernel/format. KLD informs editorial decisions: promotion-to-default, deprecation notes, recalibrated thresholds. It does not automatically revert or block any PR.

---

## 2. Why KLD over PPL

PPL collapses the full output distribution at each position to one scalar — the probability of the *actual* next token. A quant can preserve top-1 perfectly while scrambling the tail; PPL won't see it. KLD against BF16 measures `Σ P_ref(i) · (log P_ref(i) − log P_cand(i))` over the full vocab, surfacing exactly the perturbation the quant introduces.

llama.cpp's `llama-perplexity --kl-divergence` mode emits both PPL and mean/median/p99 KLD in one pass. PPL stays as a secondary sanity column in the result table.

**Top-K=256 truncation caveat (M1).** Qwen 3.5/3.6 has a 248,320-token vocab. Top-K=256 covers ~0.103% of vocab IDs but typically the bulk of the probability mass: empirical sampling of the 9B BF16 reference shows median `sum_p_residual` = 0.41% (under the 0.5% gate), mean 1.81%, p99 17.94%, max 72.0%. The residual cross-term (§4.4) corrects mean-KLD in expectation; ~1% of all scored tokens have residual mass >17% (flat distributions) where the top-K approximation is loosest.

**All reported KLD values are *lower bounds* on the true full-vocab KLD.** The residual cross-term assumes both distributions miss similarly in the tail — true for high-bit quants whose tail closely tracks BF16, less true for low-bit quants whose tail is precisely what gets scrambled. The bias is therefore **variant-dependent**: a coarser quant has more tail-mass divergence that the cross-term under-counts. Cross-variant ordering (e.g. "MQ4 worse than Q4_K_M") remains valid in direction; the *magnitude* of the gap is a lower bound and the true gap may be wider. Captured in the §8 result-table caveats preamble.

---

## 3. Gate Structure

| Gate | Type | Decision rule |
|---|---|---|
| Coherence-gate (existing) | hard ship gate | unchanged |
| Perf-gate (existing) | hard ship gate | unchanged |
| KLD eval (this PRD) | editorial signal | informs comments on #113, #116, #197; does not gate PRs |

For #113 (uniform MQ): positioning comment on the issue (now mostly historical post-Pivot). For #116 (Lloyd-MQ): positioning + recalibration input. For #182 / PR #197 (MQ4-Lloyd): editorial decision on promotion-to-default; PR #197 ships on coherence + perf alone.

---

## 4. Reference Format & Pipeline

### 4.1 Why a hipfire-internal top-K reference (not llama.cpp's native)

llama.cpp's `--kl-divergence-base` format is full-vocab uint16-quantized log-probs, ~318 GB per 9B-class reference at our slice size (1175 chunks × 1023 scored tokens/chunk × 304 KB/token). Unhostable on HF. We pipe llama-perplexity's full-vocab output through `build_kld_ref` which top-K-reduces (k=256) and writes a custom format (~2.48 GB/ref, ~150× smaller) that fits on HF Hub.

### 4.2 Binary format (HFKLDR β)

```
Header (32 bytes):
  bytes  0-7   magic "HFKLDR\0\0"   (8 ASCII chars, null-padded)
  bytes  8-11  version              (uint32, currently 1)
  bytes 12-15  n_ctx                (uint32)
  bytes 16-19  n_vocab              (uint32, sanity vs candidate)
  bytes 20-23  n_chunk              (uint32)
  bytes 24-25  top_k                (uint16, e.g. 256)
  bytes 26-27  flags                (uint16, currently 0)
  bytes 28-31  reserved             (uint32, zero)

Tokens:
  n_ctx × n_chunk × uint32 token IDs

Per-chunk × per-scored-token (n_ctx − 1 − n_ctx/2 tokens per chunk):
  uint32 top_indices[top_k]     vocab IDs, descending log-prob
  fp32   top_log_probs[top_k]   log P(i), descending
  fp32   sum_p_residual         Σ P(i) for i NOT in top-K, in [0, 1]
  fp32   reserved_pad           zero, for 8-byte alignment
```

Per-token block size: `8 + 8*top_k` bytes. At top_k=256: 2,056 B/token.

**Why log-probs (β) not raw logits.** llama.cpp's native format already encodes log-probs (via `min_log_prob` + `scale` reconstructing to log-probs, see `tools/perplexity/perplexity.cpp:222-225` on the pinned commit). Storing log-probs directly avoids backing out `max_logit` / `log_sum_exp` from the encoding (underdetermined when the logit range is clipped at 16). KLD operates on log-probs natively.

### 4.3 Producer: `build_kld_ref`

```bash
cargo run --release -p hipfire-runtime --example build_kld_ref --features deltanet -- \
  --bf16-gguf <path>.gguf \
  --slice benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
  --top-k 256 \
  --output refs/<name>.kldref.bin
```

Architecture:

1. `mkfifo /tmp/kldref-<pid>.fifo`
2. spawn `llama-perplexity -m <gguf> -f <slice> --kl-divergence-base /tmp/kldref-<pid>.fifo -c <n_ctx> --kl-divergence`
3. read llama.cpp header from FIFO (16 bytes); parse n_ctx / n_vocab / n_chunk
4. read tokens from FIFO (n_ctx × n_chunk × uint32); write hipfire header + tokens
5. for each per-token block (nv × uint16 from FIFO): reconstruct log-probs → top-K-reduce (nlargest by log_p) → compute `sum_p_residual` → write β-format block
6. join llama-perplexity; unlink fifo

The FIFO sidesteps the 318 GB transient: llama.cpp streams full-vocab uint16, hipfire reduces in-flight, ~2.48 GB lands on disk. Throughput: 375 reduced tok/s on gfx1151 for 9B BF16, 162 tok/s for 27B BF16.

**Numerics.** Log-prob reconstruction, top-K accumulators, and residual computation use fp64 internally, written as fp32. Matches llama.cpp's accumulator precision (`double sum_exp` + `log_sum_exp` formulation). 27B BF16 producer requires `--no-mmap` (passed automatically by `build_kld_ref`): demand-paging on 50 GB BF16 weights stalls in eviction-cycle on the 124 GB UMA host.

### 4.4 Consumers: `eval_hipfire` and `eval_gguf`

Both consumers read the same β-format reference. They differ only in how they obtain candidate logits at each scored position:

| binary | candidate-logit source | per-token KLD math |
|---|---|---|
| `eval_hipfire` | hipfire's runtime (forward_scratch or forward_prefill_batch → batched lm_head) | identical |
| `eval_gguf` | spawn `llama-perplexity --kl-divergence-base <fifo>` on a GGUF; read full-vocab uint16 from FIFO; reconstruct candidate's log-probs | identical |

Per-position KLD math (top-K-of-reference + residual cross-term):

```
KLD_token = Σ_{i in ref_top_K} P_ref(i) · (log_p_ref(i) − log_p_cand(i))
          + sum_p_residual_ref · (log sum_p_residual_ref − log sum_p_residual_cand)
where  sum_p_residual_cand = max(0, 1 − Σ_{i in ref_top_K} P_cand(i))
```

The residual cross-term assumes both distributions miss similarly in the tail. Reduces bias on the ~1% of flat-distribution tokens with high residual mass. Source: rev-3.3 architecture decision (`benchmarks/results/devlog_20260507_mq3_lloyd_gfx1151.md`).

Both consumers emit HFKSEQ (per-sequence mean + p99 KLD + mean NLL for PPL), aggregated by `kld_reduce.py`.

### 4.5 Tokenization (Step 1.5 verdict)

Tokenizer parity check ran 2026-05-08: hipfire's BPE encoder vs llama.cpp's GGUF-bundled BPE encoder disagree on 45.9% of token positions over the slice — a known structural divergence in how Qwen's BPE merge-priority tiebreaker resolves a specific byte sequence (hipfire emits `[2071, 110]` where llama.cpp emits `[220, 28495]`). Streams realign within 2 tokens.

**Why this doesn't block the pipeline:** `eval_hipfire.rs` reads token IDs from the reference file (written by llama-perplexity during the ref dump) and feeds those IDs directly into the candidate model. **It never re-tokenizes the slice.** All measurements use llama-perplexity's tokenization end-to-end; the KLD axis is internally consistent.

| | producer | consumer | parity needed? |
|---|---|---|---|
| BF16 reference | llama-perplexity → ref file | (none) | — |
| hipfire candidate | hipfire forward | KLD vs ref | **No** — reads token IDs from ref |
| GGUF candidate | llama-perplexity on same slice | KLD vs ref | only that llama-perplexity is deterministic, which it is |

---

## 5. Scoring Modes (hipfire candidates)

`eval_hipfire` supports two scoring modes for hipfire candidates, selected via `--scoring-mode`. **Prefill is the canonical default since 2026-05-11.**

### 5.1 Per-token mode (`--scoring-mode per-token`)

Loop `forward_scratch` once per position. Bandwidth-bound at ~44–107 tok/s depending on arch + variant (gfx1100 ≈ 2.4× faster than gfx1151 on the same model due to higher GDDR6 bandwidth). Full 1175-chunk wall-clock: ~7 h/run on gfx1100, longer on gfx1151.

### 5.2 Prefill mode (`--scoring-mode prefill`, canonical)

For each chunk:

1. **Prefix call:** `forward_prefill_batch(tokens[0..n_ctx/2])` with no logit capture. Writes KV positions `[0, n_ctx/2)`.
2. **Scored call:** `forward_prefill_batch(tokens[n_ctx/2..n_ctx-1])` with `per_token_hidden_out=Some(&hidden_buf)`. Writes the post-output-norm hidden state per row into the caller's buffer.
3. **lm_head fan-out:** loop `weight_gemv(weights.output, hidden_row, scratch.logits)` per scored position; download + compute KLD per position (existing per-position math).

Scratch reuse: `HIPFIRE_PREFILL_REUSE_PBS=1` is set automatically so the caller-owned `PrefillBatchScratch` is allocated once and reused across all chunks (avoids ~25 alloc/free pairs per chunk).

**Wall-clock speedup measured** (prefill_microbench, n_ctx=2048, kv_mode=asym3, mean of 3 iters):

| arch | variant | per-token (tok/s) | prefill (tok/s) | inference speedup |
|---|---|---:|---:|---:|
| gfx1100 | 9B-MQ4 | 107.9 | 2162.1 | 20.0× |
| gfx1151 | 9B-MQ4 | 44.4 | 842.0 | 19.0× |
| gfx1151 | 9B-MQ3-Lloyd | 46.1 | 391.7 | 8.5× |
| gfx1151 | 9B-MQ3 | 52.2 | 396.7 | 7.6× |

The ~19-20× ratio is arch-independent — speedup is structural (GEMM-vs-GEMV at 2048-token batch), not a chip-specific tuning artifact. Including the per-position KLD compute overhead (top-K extract + residual + NLL), the **end-to-end wall-clock speedup is ~7×** for full-slice runs.

### 5.3 The two modes are not equivalent — kernel-path numerical divergence

V1 (gfx1100 MQ4 full slice, 1175 chunks, 2026-05-11) measured a **−6.75% mean-KLD shift** of prefill vs per-token (Pearson per-seq 0.949; bootstrap CIs do not overlap). Confirmed on gfx1151 at n=50 (−7.86% MQ4, −8.64% MQ3) with consistent direction.

**Mechanism.** Prefill mode's batched GEMM kernels use multi-accumulator reductions that average fp16/fp32 partial-sum noise across more terms than per-token GEMV's single-accumulator pattern. The result is logits systematically closer to BF16 ground truth — prefill is both faster *and* more accurate against the BF16 reference, but the bias makes the two modes separate measurement classes.

**Implication.** Mode is recorded in the result table as a first-class column. **Rows of different mode are not cross-comparable.** Pre-existing per-token gfx1100 kldseqs from 2026-05-08 (mq3, mq4, mq3-lloyd, mq6) are retained as historical-only rows; re-running them in prefill mode is on the menu but deferred per the Pivot (§11).

### 5.4 Eligibility — auto-fallback

`forward_prefill_batch` consults `is_batchable_la(dtype, arch)` (qwen35.rs:3823) before dispatching. Variants outside the batchable set auto-fall-back to per-token internally — the new default is safe.

| variant | gfx1100 / gfx1151 / gfx1101-1102 / gfx1150 | gfx1200/1201 | other archs |
|---|---|---|---|
| Q8 / MQ4 / MQ6 / HFQ4-G256 / HFQ6-G256 | prefill | prefill | prefill |
| MQ3 (uniform) | prefill (gfx11 WMMA) | prefill (gfx12 WMMA) | per-token fallback |
| MQ3-Lloyd | prefill (gfx11) | prefill if `HIPFIRE_LLOYD_GFX12=1` | per-token fallback |
| MQ4-Lloyd | per-token (issue #182 not yet in batchable set) | per-token | per-token |
| HFP4G32 | per-token (no GEMM kernel; PR #224 v2 deferred) | per-token | per-token |
| MFP4G32 | per-token (same kernel family as HFP4G32) | per-token | per-token |

When HFP4/MFP4 lands a batched WMMA kernel (PR #224 v2), those variants enter the batchable set and inherit prefill scoring automatically.

---

## 6. Eval Matrix

Per model: 9B (qwen3.5) / 27B (qwen3.6). qwen3.5-27B was dropped post-rev-3.1 (superseded by qwen3.6-27B; same parameter budget, newer training corpus).

| Track | Variants | Scoring | Status |
|---|---|---|---|
| Hipfire (canonical, post-Pivot) | HFP4G32, MFP4G32 | prefill (when batched kernel lands) / per-token (today) | Awaiting .hfq files |
| Hipfire (historical) | MQ3, MQ4, MQ3-Lloyd, MQ6 | per-token (2026-05-08 runs), + MQ4 prefill (2026-05-11) | Partial committed |
| GGUF anchors (9B) | Q8_0, Q6_K, Q4_K_M, UD-Q3/Q4/Q5/Q6_K_XL | n/a (single path) | All 7 committed (gfx1151) |
| GGUF anchors (27B) | Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q3_K_M, Q3_K_S | n/a | Deferred per Pivot |

Q8 plays a dual role for hipfire: closest-to-BF16 reference proxy AND a harness sanity check that the eval reproduces near-zero KLD for a known-good quant.

### 6.1 Arch coverage

Each hipfire variant is evaluated on **two archs**: gfx1100 (canonical ship target, GDDR6) and gfx1151 (deployment, Strix Halo APU, LPDDR5x). BF16 reference dumps are produced on gfx1151 only (sole host with enough RAM for 27B BF16 in 137 GB UMA).

References are arch-independent (just bytes); a quant scored on either arch compares against the same reference. The result table grows a per-arch column. **Cross-arch KLD divergence > canary tolerance is itself a finding** — decode-path multi-acc drift would surface as a quant-quality artifact.

---

## 7. Eval-Mode Hipfire Flags

`eval_hipfire` always sets the following so future replays match byte-for-byte:

```
HIPFIRE_NORMALIZE_PROMPT=0     # raw byte-stream-through; eval is byte-deterministic
HIPFIRE_GRAPH=0                # capture-mode adds capture-illegal paths; eval doesn't need it
HIPFIRE_KV_MODE=asym3          # canonical KV-mode for ship benches
HIPFIRE_LLOYD_GFX12=1          # if running on gfx1200/1201 — PR #195 gate
HIPFIRE_PREFILL_REUSE_PBS=1    # set automatically by --scoring-mode prefill
```

And the CLI default since 2026-05-11:

```
--scoring-mode prefill         # canonical hipfire scoring mode
```

`HIPFIRE_GRAPH=0` is currently a determinism style choice, not a correctness requirement (byte-equality between graph=0 and graph=1 was verified on 2026-05-08 against a dense Qwen3.5-9B-mq4 prefill of 64 tokens, kv_mode=asym3). `HIPFIRE_NORMALIZE_PROMPT=0` matches eval-determinism intent; the canary fixture runs both ON and OFF as a hedge.

**Historical-baseline drift caveat.** `HIPFIRE_NORMALIZE_PROMPT` was flipped to ON by default on 2026-04-26. Lloyd findings (`benchmarks/results/lloyd_max_findings_20260501.md`) were likely run with normalize ON. The new eval forces OFF — PPLs in the new table will not reproduce historical baselines on the same corpus.

---

## 8. Result Table Format

```markdown
| Model | Variant     | Arch    | Mode      | Size GB | Mean KLD ± 95% CI | p99 KLD | PPL    | DFlash τ | Notes |
|-------|-------------|---------|-----------|---------|--------------------|---------|--------|----------|-------|
| 9B    | Q8-uniform  | gfx1100 | prefill   | 9.4     | 0.0008 ± 0.0001    | 0.012   | 9.81   | n/a      | reference proxy |
| 9B    | HFP4G32     | gfx1100 | per-token | 5.2     | TBD                |         |        |          | awaiting .hfq |
| 9B    | MFP4G32     | gfx1100 | per-token | 5.2     | TBD                |         |        |          | awaiting .hfq |
| 9B    | MQ4         | gfx1100 | prefill   | 5.2     | 0.817 ± 0.013      | 19.52   | 14.89  | TBD      | historical comparison |
| 9B    | MQ4         | gfx1100 | per-token | 5.2     | 0.876 ± 0.013      | 19.99   | —      | TBD      | HISTORICAL (do not cross-compare mode) |
... (per model, per arch)
```

**`Mode` column.** First-class field. `prefill` (canonical hipfire), `per-token` (historical hipfire), `gguf` (eval_gguf). Mandatory; do NOT compare rows of different mode (see §5.3).

**Filename convention (post-2026-05-11):**

```
<variant>__<arch>__<scoring_mode>.kldseq      preferred 3-segment form
<variant>__<arch>.kldseq                      legacy 2-segment form (auto-tagged
                                              gguf if variant contains "gguf-",
                                              else per-token)
```

**`Mean KLD ± 95% CI`** uses 10,000-resample bootstrap on per-sequence means. Per-seq KLDs are persisted by default (8 KB per row at 1175 chunks), so CIs are reproducible without re-running.

**`DFlash τ` column** (deferred per Pivot): for each variant where a draft model exists, τ from the canonical `merge_sort` prompt (per CLAUDE.md "Prompt-structure τ sensitivity"; byte-identical prompt + md5 in the result file's preamble).

**Mandatory caveats preamble** for any result file:

```
## Caveats for direct comparison

- PPL column was measured with HIPFIRE_NORMALIZE_PROMPT=0 and
  HIPFIRE_GRAPH=0. Historical baselines in
  benchmarks/results/lloyd_max_findings_20260501.md were likely run
  with normalize ON (default flipped 2026-04-26) and may not match.
- The eval slice (wikitext2-1024s-2048ctx.txt) is a frozen committed
  fixture; PR #115's corpus is not committed.
- Both archs use HIPFIRE_KV_MODE=asym3.
- `Mode` column: hipfire MQ rows with mode="per-token" are historical
  and measure ~7% higher mean-KLD than prefill (kernel-path numerical
  difference, see §5.3). Do NOT cross-compare rows of different mode.
- **All KLD values are lower bounds.** The top-K=256 residual cross-term
  assumes both distributions miss similarly in the tail — looser for
  coarser quants whose tail is more scrambled. Cross-variant ordering
  (direction) is reliable; absolute magnitudes are conservative. See §2.
```

---

## 9. Reference Distribution

BF16 reference dumps are SHA-pinned, content-addressed, and uploaded to HF Hub at dataset repo [`hipfire-models/qwen-kldref`](https://huggingface.co/datasets/hipfire-models/qwen-kldref):

```
qwen3.5-9b-bf16.kldref.bin     2.48 GB  sha256 06948cd3…
qwen3.6-27b-bf16.kldref.bin    2.48 GB  sha256 8af83b38…
```

(Both refs are the same size at top_k=256: per-token block is `8 + 8·top_k = 2,056 B` independent of n_vocab. n_vocab is in the header for sanity.)

In-tree `benchmarks/quality-baselines/harness/manifest.json` carries per-reference `{sha256, size_bytes, host_arch, llamacpp_commit, slice_md5, top_k, n_ctx, n_vocab, n_chunk, producer_cmd, hf_repo, hf_repo_type, comment}`.

**Download recipe (agent + human readable):**

```bash
# From repo root, one-time:
python3 -m venv .venv
.venv/bin/pip install huggingface_hub

# Pull both refs into benchmarks/quality-baselines/refs/, verify SHA256:
./scripts/fetch-eval-refs.sh
```

`scripts/fetch-eval-refs.sh` is idempotent — files already present + matching SHA256 are skipped. The runtime examples' `verify_ref_sha256` re-checks SHA on each invocation, so a corrupted local ref is caught at run start. Alternative paths (inline `hf_hub_download` for single-file, `hf download --repo-type dataset` for CLI) documented in the harness README.

**Reference-drift canary** is a separate guard from the SHA256 check:
- SHA256 in manifest = reference identity guard (catches re-uploads, corruption)
- Canary fixture = harness output reproducibility guard (catches `eval_hipfire` regression that yields different KLD on the same model + same reference)

Canary is 11 wikitext sequences — 10 short (≤500 tokens) + 1 near-max ctx (1800–2000 tokens, catches RoPE / KV-cache drift). Per-sequence expected KLDs in `harness/canary.md` (populated from first canary run).

---

## 10. Validation

Sequenced gates from cheapest to most expensive.

### V0 — Pre-validation kernel-equivalence microtests (deferred to V1 failure)

Two of the originally-planned three V0 microtests are automatically true or unnecessary:

- **`rmsnorm_batched ≡ rmsnorm_f32`** verified by code inspection. `dispatch.rs:10704` and `dispatch.rs:10817` dispatch the same kernel (`kernels::RMSNORM_SRC` named `"rmsnorm_f32"`) with the same launch config. No runtime test needed.
- **DN-state byte-equality** and **KV-cache continuity across split prefill calls** are end-to-end correctness properties that V1 validates implicitly at full-slice scale (1175 chunks × 1023 positions). If V1 passes, both hold by induction; if V1 fails, V0 fires as the localization tool.

**Trigger:** if V1 on any variant reports a HARD FAIL or persistent SOFT FAIL, implement and run the DN-state + KV-cache microtests as the root-cause step.

### V1 — Same-variant, same-arch A/B (when applicable)

When a variant has been scored in two modes (e.g., MQ4 per-token vs prefill on gfx1100), `kld_diff.py` computes per-sequence:

| metric | tolerance |
|---|---|
| mean delta abs / rel | per-variant, empirically calibrated |
| Pearson per-seq | ≥ 0.9999 expected, but kernel-path mode differences relax this (see §5.3) |
| p99 \|Δseq\| | ≤ 5% of variant's mean KLD |
| 95% bootstrap CI overlap | yes |

For prefill-vs-per-token A/B on hipfire MQ formats, the gates intentionally fail — that's how we discovered the kernel-path divergence is real (§5.3). The diff tool is the diagnostic; the verdict is editorial.

### V2 — Canary-fixture pre-commit gate (currently informational, not enforced)

`eval_hipfire` on the 11-seq canary fixture, then `kld_diff.py` against the committed expected-KLDs in `canary.md`. Designed to run in a few minutes and catch harness regressions on the live decode path.

**Status:** `canary.md` ships with expected-KLD slots TBD-populated. Until a baseline is run and the slots filled in, V2 cannot fire — it's informational only, not enforced. Populating the canary is on the deferred list (§13).

Limitation acknowledged even once populated: V2 catches divergence between modes / between releases, but cannot detect a shared-mode bug that affects both paths identically. issue-113's existing coherence-gate is the backstop for shared regressions in the live decode path.

### V3 — Cross-arch sanity (gfx1100 ≡ gfx1151)

Same variant in same mode on both archs. Divergence > canary tolerance is a finding (multi-acc decode drift surfaces as quant-quality artifact). Required when both archs have a committed row.

### V5 — `--max-chunks N` for dev smoke

50-chunk runs (~5–8 min) for code-correctness smoke during development. **Not a substitute** for full-slice statistics — Pearson ≥ 0.9999 over 50 chunks is a much weaker test than over 1175 (variance ~1/N).

---

## 11. Pivot (2026-05-11)

Two scope changes from rev-3.3:

1. **Prefill canonical for hipfire scoring.** V1 measured a ~7% systematic kernel-path bias between prefill and per-token (gfx1100 9B-MQ4 full slice; Pearson 0.949; CIs do not overlap). Prefill is both ~7× faster AND ~7% closer to BF16. The four pre-existing per-token gfx1100 kldseqs are renamed `*__per-token.kldseq` and retained as historical-only rows.
2. **MQ matrix is paused; HFP4G32 / MFP4G32 are the canonical hipfire focus.** Partial Pareto data is conclusive: hipfire MQ4 KLD 0.876 vs vanilla Q4_K_M 0.125 at near-equivalent size on the 9B slice — the MQ family's coarse single-fp32-scale-per-256 grouping is structurally noisier than community K-quants. PR #224 / #225 introduced HFP4 / MFP4 — RDNA-optimal successors with two-level scaling (UE8M0 g32 + FP16 row). Once 9B .hfq files exist for those formats, they slot in as the canonical hipfire-side rows.

---

## 12. Storage & Cost

| line item | wall-clock | storage |
|---|---|---|
| Slice text | — | ~10 MB in git |
| BF16 reference dumps (gfx1151, one-time per model) | 2 models × ~1.5–2 h | ~2.48 GB/ref × 2 = 4.96 GB on HF |
| Reference-build transient (FIFO, never disk) | — | ~318 GB streamed |
| Per-quant per-sequence KLDs | — | ~28 KB/run (1175 × 24 B) |
| GGUF anchors (7 × 9B, gfx1151) | ~50 min/run × 7 = ~6 GPU-h | done; committed |
| Hipfire 9B per-token (5 variants × 2 archs) | ~7 h/run × 10 = ~70 GPU-h | historical, partial committed |
| Hipfire 9B prefill (5 variants × 2 archs) | ~1 h/run × 10 = ~10 GPU-h | partial (MQ4 gfx1100 committed) |
| Hipfire 27B (5 variants × 2 archs) | ~14–18 h per-token / ~2–5 h prefill | deferred per Pivot |

---

## 13. Current State

**Done:**
- Harness skeleton: slice fixture (md5 `83b0205a304bf4e52172ecdb05f2e895`), `kldref_format.py`, `kld_reduce.py` (Mode column), `kld_diff.py`, `tokenizer_parity.py`, `make_slice.sh`, `canary.md` skeleton.
- Runtime binaries: `build_kld_ref` (FIFO + top-K reduce + auto-`--no-mmap`), `eval_hipfire` (`--scoring-mode prefill` default + `--max-chunks N` + auto-mkdir), `eval_gguf` (FIFO + ref-token-equality check + residual cross-term), `prefill_microbench` (Step 0 gate), `tokenize_slice` (used by tokenizer_parity.py).
- BF16 references: 9B (gfx1151, 53 min, 375 reduced tok/s) + 27B (gfx1151, 2h 3m, 162 reduced tok/s). Uploaded to `hipfire-models/qwen-kldref` (dataset); manifest wired; fetch script smoke-tested end-to-end.
- Hipfire data: 4 historical per-token kldseqs (MQ3, MQ4, MQ3-Lloyd, MQ6 on gfx1100); 1 canonical prefill kldseq (MQ4 on gfx1100 full slice = 0.817).
- GGUF anchors: all 7 9B anchors on gfx1151 (Q8_0/Q6_K/Q4_K_M + UD-Q3/Q4/Q5/Q6_K_XL). Q8_0 establishes the near-lossless floor (KLD 0.0163); UD-Q4_K_XL beats vanilla Q4_K_M by ~1.9× (0.067 vs 0.125).
- Scoring-mode A/B: V1 gfx1100 MQ4 full slice (canonical) + gfx1151 MQ3/MQ4 partial 50-chunk smoke runs (not archived; sign-test p<1e-100 on the gfx1100 result confirms kernel-path divergence is not noise).

**Deferred (post-Pivot, not blocking PR):**
- 27B hipfire matrix (~150 GPU-h).
- 9B-Q8 + 9B-MQ4-Lloyd hipfire rows on gfx1100.
- gfx1151 hipfire full-slice pass (the 50-chunk smoke runs from V1 development were not archived; reproducible via `--max-chunks 50`).
- DFlash τ column (Step 8) — editorial signal mooted by Pivot.
- Pareto writeup (Step 9) — wait until HFP4/MFP4 .hfq files exist.
- HFP4G32 / MFP4G32 .hfq files for 9B (depends on `hipfire-quantize --format hfp4`; eval-side is wired through auto-fallback).

---

## 14. References

### Issues / PRs

- #113 — uniform MQ3 quality (this PRD supersedes the PPL-only / 5%-gate scoping).
- #116 — Lloyd-MQ3 ship gates (this PRD adds *positioning*, not a third gate).
- #182 / PR #197 — MQ4-Lloyd WMMA prefill.
- PR #195 — MQ3-Lloyd WMMA prefill (merged 2026-05-08).
- PR #201 — O(N²) → O(N log N) BPE encoder (cherry-picked into this branch; made `tokenize_slice` tractable).
- PR #224 — HFP4G32 quant family (RDNA-optimal FP4 + UE8M0 g32 + FP16 row scale).
- PR #225 — MFP4G32 (HFP4G32 + offline FWHT, drop-in MQ4 replacement).

### Repo docs

- `docs/quant-formats/hfp4.md` — HFP4G32 binary format spec.
- `docs/plans/mq-sub4bit-prd.md` — MQ family research PRD.
- `benchmarks/results/lloyd_max_findings_20260501.md` — historical Lloyd writeup (corpus not committed; PPL numbers not reproducible).
- CLAUDE.md — "Prompt-structure τ sensitivity" rule.

### External

- llama.cpp `llama-perplexity` `--kl-divergence` / `--kl-divergence-base` modes.
- llama.cpp pinned commit: `9dcf83552887bb898b4a98a5761361e504e31fc3`.
- Unsloth Qwen3.6 GGUF Pareto plot: https://unsloth.ai/docs/models/qwen3.6.
- HF dataset: https://huggingface.co/datasets/hipfire-models/qwen-kldref.
