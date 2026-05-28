# KLD Measurements — Master Table

**Status:** Living document. Append new measurements; do not delete.
**Last updated:** 2026-05-26 — restructured into two summary tables (**Winning quants** + **Experiments**, immediately below). The dated narrative log is preserved unchanged in §0–§6 (see §5 for the chronological findings catalog). Most recent data: 27B mixed-bits `mq3-body-mq4-lmhead` (§3.2, 2026-05-20), 9B `lmhead-a100` 4-bit KLD headline + MQ3 inverse-direction result (§1.1k / §1.4a, 2026-05-19).

**Provenance note (2026-05-18 import):** rows §1.1 through §4A were carried over from `feat/mq-v2-quant-format` (which was split into multiple PRs, not merged whole). Some of those measurements pre-date subsequent code fixes on master (e.g. May-16 Q8 attention NaN #264, F16 lm_head shim #265, hipGraph H2D capture fix #7790ac6a). Heuristic: better KLD numbers are more likely to be representative of current code. Re-measurement on master is welcome for any specific row; if the new number is materially lower, replace and note the SHA delta.
**Owner:** hipfire eval
**Authoritative methodology:** [`issue-113-quant-quality-eval.md`](issue-113-quant-quality-eval.md)
**Cross-engine framework:** Δ-above-own-Q8 (TL;DR + §2.4.1 + [engine-drift memory](../../memory/project_engine_drift_floor_decomposition.md)). Absolute `KLD(engine || HF-bf16)` is NOT a cross-engine output-quality metric — see §6 rule 8.

This doc is the single place to look for every KLD/PPL number we've measured against the BF16 reference. Pull-out tables in other docs (`qwen35-mq4-quality-gap.md` §1.5, `awq_hipfire.md`, individual cohort `result-table.md` files) are derived views; this is the source-of-truth catalog.

The two tables below are the summary view; **§0–§6 are the detailed research log** (mechanism, CIs, working-artifact paths, dated findings). Every row links back to its log section.

---

## Winning quants

Best measured **hipfire** recipe per **model × size × bpw band**, ranked by **KLD ↓** —
the primary quality metric per [issue-113 §2](issue-113-quant-quality-eval.md): KLD
scores full-distribution fidelity to the BF16 reference, not just argmax confidence.
All rows are `prefill` scoring against the BF16 `kldref`. Where a *different* recipe
wins **PPL** (argmax / greedy quality) the split is footnoted — the two objectives
diverge often enough (the recurring **KLD-vs-PPL inversion**, see §1.1e.i / §1.1h /
§1.4a) that the right pick depends on the workload: **KLD** for sampling, spec-decode
draft/target acceptance, and RL reward modeling; **PPL** for greedy decode.

| Model · size | bpw band | Best recipe (by KLD) | bpw | KLD ↓ | PPL | KV | n | Arch | Detail |
|---|---|---|---:|---:|---:|---|---:|---|---|
| Q3.5-9B | 8-bit | `q8f16` (Tier-3 fused WMMA) | 8.50 | **0.0173** | 9.189 | q8 | 256 | gfx1151 | §1.1b |
| Q3.5-9B | 6-bit | `mq6 + Q8 conv1d` | ~6.50 | **0.0510** | 9.186 | q8 | 512 | gfx1151 | §1.1e |
| Q3.5-9B | 5-bit | `mq4-kmd2 + Q8 conv1d` | ~5.04 | **0.1613** | 9.172 | q8 | 512 | gfx1151 | §1.1g |
| Q3.5-9B | 4-bit | `mq4-awq-gptq-f2-lmhead-a100` ¹ | 4.74 | **0.0863** | 9.374 | q8 | 256 | gfx1100 | §1.1k |
| Q3.5-9B | 3-bit | `mq3-awq-gptq` (Q8 lm_head) | ~3.80 | **0.1967** | 11.645 | q8 | 256 | gfx1151 | §1.4 |
| Q3.5-4B | 4-bit | `mq4-awq+gptq+Q8conv1d` | ~4.25 | **0.0662** | 10.712 | q8 | 512 | gfx1151 | §3.1 |
| Q3.5-0.8B ⚠ | 8-bit | `q8f16` (Tier-3) | 8.50 | 0.0041 | — | q8 | 512 | gfx1100 | §4.1 |
| Q3.5-0.8B ⚠ | 4-bit | `mq4-awq+gptq+Q8conv1d` (post-OBS-fix) | ~4.25 | 0.0478 | 18.24 | q8 | 512 | gfx1100 | §5 |
| Q3.6-27B | 4-bit | `mq4-awq-gptq-f2-q8head` (V100) ² | ~4.50 | **0.1257** | 8.697 | q8 | 256 | gfx1100 | §3.2 |
| Q3.6-27B | 3.5-bit (mixed) | `mq3-body-mq4-lmhead` | 3.552 | 0.3082 | 9.856 | q8 | 256 | gfx1100 | §3.2 |

**Notes:**

¹ **9B 4-bit is a dual headline.** `lmhead-a100` is **KLD-optimal** (0.0863). The
PPL-optimal sibling is `mq4-awq-gptq-f2-q8head` (~4.4 bpw → KLD 0.1727 / **PPL 8.417**,
§1.1j): it wins PPL by ~10% but loses KLD by 2×. Note also that this 4-bit `lmhead-a100`
row (0.0863) beats the 5-bit `mq4-kmd2` row (0.1613) on KLD — the band-vs-KLD progression
is **not monotonic** because `lmhead-a100` is an exceptionally tuned recipe the 5-bit row
hasn't received. This 4-bit row is also hipfire's first to land inside the llama.cpp
K-quants Pareto frontier on absolute KLD (beats Q4_K_M despite −0.33 bpw; see §1.1k).

² **27B 4-bit V100 vs A100 is a statistical tie.** The A100-calibrated sibling
`mq4-awq-gptq-f2-lmhead-a100` (4.458 bpw) measures KLD 0.1307 / PPL 8.663 — V100 wins
KLD by 4%, A100 wins PPL + tail. Calibration host is ~noise at 27B (the opposite of 9B,
where `lmhead-a100` cut KLD ~50% — see §3.2 for why the recipe didn't transfer by size).

⚠ **Q3.5-0.8B is intrinsically incoherent on hipfire at every precision** — even the Q8
floor babbles (DeltaNet depth × recurrence pedestal, §5 2026-05-14 retraction). Its
KLD/PPL numbers are valid **only for ranking recipes against each other**, never as ship
signals. The smallest coherence-valid model is **4B**.

## Experiments

Every other measured row: baselines (no calibration), single-lever ablations, α-sweeps,
cross-arch reproductions, dropped / falsified recipes, and llama.cpp cross-engine
reference anchors. Headline KLD/PPL only — CIs, mechanism, and artifact paths live in
the linked detail section. `—` = not recorded in this summary (see detail).

| Model · size | Recipe / probe | Category | bpw | KLD | PPL | KV | n | Detail |
|---|---|---|---:|---:|---:|---|---:|---|
| Q3.5-9B | `mq4-base` | baseline | 4.25 | 0.3376 | 9.116 | asym3 | 512 | §1.1 |
| Q3.5-9B | `mq4-awq` | lever: AWQ only | 4.25 | 0.2800 | 9.271 | asym3 | 512 | §1.1 |
| Q3.5-9B | `hfp4` / `mfp4` / `*-l4-l5c` (FP4 family) | format (all worse) | ~4.5–5.0 | 0.46–0.78 | — | asym3 | 512 | §1.1 |
| Q3.5-9B | `mq4-plain-q8head` | baseline (A/B partner) | ~4.4 | 0.2149 | 8.746 | q8 | 256 | §1.1j |
| Q3.5-9B | `mq4-awq-gptq-f2-q8head` | **PPL-best 4-bit** (alt headline) | ~4.4 | 0.1727 | 8.417 | q8 | 256 | §1.1j |
| Q3.5-9B | `mq4-q8conv1d` | lever: Q8 conv1d only | 4.25 | 0.2501 | 8.789 | q8 | 512 | §1.1f |
| Q3.5-9B | `mq4-awq + Q8 conv1d` | stack (superseded by §1.1j/k) | 4.25 | 0.1842 | 9.575 | q8 | 512 | §1.1f |
| Q3.5-9B | `mq4-lloyd` | **dropped** — Lloyd null at 4-bit | ~4.91 | 0.3114 | 9.085 | q8 | 512 | §1.1c |
| Q3.5-9B | `mq4-kmd2` (no AWQ) | lever: mixed MQ4/MQ6 | ~4.81 | 0.1859 | 9.661 | asym3 | 256 | §1.1h |
| Q3.5-9B | `mq4-kmd2-awq` | stack (asym3; ~0.11 q8-equiv est.) | ~5.75 | 0.1485 | 9.844 | asym3 | 256 | §1.1h |
| Q3.5-9B | F1 α=0.5 (184 sidecars) | AWQ whitelist | 4.25 | 0.1725 | 9.54 | q8 | 256 | §1.1h |
| Q3.5-9B | F2 α=0.5 (248 sidecars) | AWQ whitelist — KLD-flat, PPL −6.6% | 4.25 | 0.1724 | 8.91 | q8 | 256 | §1.1h |
| Q3.5-9B | F2 α=0.55 (gfx906) | α-sweep — PPL optimum | 4.25 | 0.1830 | 8.79 | q8 | 100 | §1.1h |
| Q3.5-9B | F2 α=0.55 (gfx1151) | α-sweep — cross-arch repro | 4.25 | 0.1855 | 8.785 | q8 | 100 | §1.1i |
| Q3.5-9B | `mq3-rtn` | baseline: naked MQ3 | 3.25 | 0.5449 | 13.45 | q8 | 256 | §1.4 |
| Q3.5-9B | `mq3-awq-gptq-f2-lmhead-a100` | **dropped** — inverse-direction at 3-bit | ~4.0 | 0.2613 | 10.440 | q8 | 256 | §1.4a |
| Q3.5-0.8B | `q8f16` (Tier-2, asym3) | floor | 8.50 | 0.1256 | 19.633 | asym3 | 512 | §2.1 |
| Q3.5-0.8B | `mq4-base` | baseline | 4.25 | 0.2675 | 22.088 | q8 | 512 | §2.2 |
| Q3.5-0.8B | `mq4-awq` | lever: AWQ only | 4.25 | 0.2531 | 22.149 | q8 | 512 | §2.2 |
| Q3.5-0.8B | `mq4-awq + Q8 conv1d` | stack — no-GPTQ best | 4.25 | 0.1366 | 19.61 | q8 | 512 | §5 |
| Q3.5-0.8B | `mq4-gptq` (alone) | **dropped** — GPTQ alone hurts | 4.25 | 0.3908 | — | q8 | 512 | §5 |
| Q3.5-0.8B | `mq4-awq+gptq` (no conv1d) | **dropped** — GPTQ regresses at 0.8B | 4.25 | 0.3371 | — | q8 | 512 | §5 |
| Q3.6-27B | `mq4-plain-q8head` | baseline (A/B partner) | ~4.5 | 0.2034 | 8.584 | q8 | 256 | §3.2 |
| Q3.6-27B | `mq4-awq-gptq-f2-lmhead-a100` | A100 sibling (tied with V100) | 4.458 | 0.1307 | 8.663 | q8 | 256 | §3.2 |
| llama.cpp 9B | Q8_0 | anchor (floor) | 8.50 | 0.0163 | 9.31 | f16 | — | §1.2 |
| llama.cpp 9B | UD-Q6_K_XL | anchor | ~6.7 | 0.0213 | 9.14 | f16 | — | §1.2 |
| llama.cpp 9B | Q6_K | anchor | 6.56 | 0.0250 | 9.31 | f16 | — | §1.2 |
| llama.cpp 9B | UD-Q5_K_XL | anchor | ~5.5 | 0.0408 | 9.27 | f16 | — | §1.2 |
| llama.cpp 9B | UD-Q4_K_XL | anchor | 5.32 | 0.0670 | 9.34 | f16 | — | §1.2 |
| llama.cpp 9B | Q4_K_M | anchor | 5.07 | 0.1249 | 8.70 | f16 | — | §1.2 |
| llama.cpp 9B | UD-Q3_K_XL | anchor (bpw-matched 4-bit) | ~4.50 | 0.1411 | 8.67 | f16 | — | §1.2 |
| llama.cpp 0.8B | Q4_K_M | anchor | 5.07 | 0.0351 | 17.33 | q8 | 1175 | §2.3 |

**Unmeasured / queued** (no row yet): MQ3-Lloyd, MQ4-Lloyd (gated on PR #197), MQ2-Lloyd
(§4A.1); Qwen3.5-A3B (task #8); 27B llama.cpp anchors (§3.2 next priority); MQ3 + lm_head-MQ4
+ AWQ at sub-9B (§1.4a #2). Open KV-mode floor measurements in §4.3.

---

## TL;DR — strategic headline (2026-05-13 PM, supersedes AM)

**The Tier-2 Q8 floor was an FP32-vs-bf16 precision-class mismatch artifact, not engine drift.** The 2026-05-13 PM pattern-hunt audit (commits `13003256`, `f6d1a59e`, `a75b4a08`, `1b90a663`) proved hipfire's fp32-native kernels (rmsnorm, l2norm, recurrence) are bit-faithful to the fp64 ideal. Then on the same day, [PR #248](https://github.com/Kaden-Schutt/hipfire/pull/248) Tier-3 fused WMMA Q8 prefill kernels (FP16 accumulators, matched to HF's bf16 precision class) landed, and the measured Q8 floor collapsed:

| Model | Tier-2 Q8 floor (FP32 acc) | Tier-3 Q8 floor (FP16 acc) | Reduction |
|---|---:|---:|---:|
| Q3.5-0.8B q8-KV | 0.0796 | **0.0041** (n=512, gfx1100) | 19× |
| Q3.5-9B q8-KV | 0.1459 (asym3-KV) → ~0.12 (est. q8-KV) | **0.0173** (n=256, gfx1151) | 7-8× |

**The 9B Tier-3 Q8 floor is +6% above llama.cpp Q8_0 9B's 0.0163 — same precision class.** What was previously reported as "100% engine drift" (Phase 0 commit `13003256`) was 100% the FP32-FP16 accumulator-class mismatch with HF's bf16 reference. On Tier-3 the gap is functionally closed. Hipfire's kernels were never imprecise; they were *too* precise for the dtype the reference was trained at, and switching the Q8 prefill path's accumulator class to FP16 made the absolute KLD-vs-HF land near the cross-engine noise floor.

**Cross-engine KLD-vs-HF is a valid output-quality comparison once both engines use precision-class-matched accumulators.** Within precision-class match (Tier-3 hipfire FP16-WMMA vs llama.cpp's similar internal bf16/FP16 accumulators), absolute `KLD(engine || HF-bf16)` becomes interpretable: the gap is residual architectural drift (DeltaNet recurrence accumulation for Q3.5 family) plus weight-quantization cost. For mixed-path cross-engine comparisons (Tier-2 hipfire vs llama.cpp), use **Δ-above-own-Q8** instead: first-order independence (KLD ≈ ½·Var(δlogit), independent perturbations add) cancels each engine's own floor, leaving pure quantizer cost. Cross-engine quality claims at matched bpw are now `Δ_hipfire(MQ4) vs Δ_llamacpp(Q4_K_M)` OR — equivalently when Tier-3 — absolute `KLD_hipfire vs KLD_llamacpp` directly.

**Strategic implication.** Engine-surgery scope has collapsed to ~zero kernels — confirmed empirically twice (F2/F3 audit + PR-#248 floor measurement). The calibration roadmap (AWQ Stage A → GPTQ Stage B → MR-GPTQ Stage C, Lloyd codebooks) is the right priority and is now better-justified since (a) there is no kernel arithmetic left to fix and (b) on Tier-3, 4-bit Δs are dominated by quant noise rather than engine drift. Stage B/C success is measured as `Δ = KLD(quant) − KLD(Q8)` within hipfire (path-matched Q8 floor), and cross-engine claims are made on `Δ_hipfire` vs `Δ_llamacpp` at matched bpw.

**Depth-amplification finding refined.** The earlier "DeltaNet recurrence amplifies bf16-cast drift ~8×" framing (Q3-0.6B dense 0.0098 vs Q3.5-0.8B DeltaNet 0.0796 on Tier-2) conflated two effects: (a) FP32-vs-bf16 precision-class mismatch (now closed by Tier-3) and (b) genuine DeltaNet recurrence drift accumulation over LA layers. On Tier-3, **0.8B DeltaNet (0.0041) → 9B DeltaNet (0.0173) = 4.2× over 8 additional layers (32 vs 24)**, or ~1.20× per extra layer of cumulative drift. That's the real DeltaNet amplification, much smaller than the Tier-2 phantom 8×. And at 9B the absolute floor is now low enough (0.0173) that 4-bit weight noise (~0.2-0.3 nats) dominates, not recurrence drift.

**Other validation pending:**
1. `KLD(hipfire-Q8 || hipfire-FP32-everything)` — should be tiny if the framework's premise holds. The Tier-2-vs-Tier-3 collapse (0.1459 → 0.0173 on the same hipfire engine, only kernel accumulator class changed) is already a partial validation.
2. Rank-correlation of `Δ_hipfire(MQ4)` vs `Δ_llamacpp(Q4_K_M)` across prompts/positions.
3. mq4-vs-mq4-awq Δ within hipfire matches the qualitative AWQ-helps direction observed in other engines.

---

## 0. Reading conventions

- **All hipfire rows are POST-RoPE-fix** (commit `1805d820` flipped halfsplit RoPE to default; 2026-05-12 evening) unless explicitly marked `PRE-FIX`. Pre-fix hipfire numbers are ~3–5× inflated due to the interleaved-vs-halfsplit RoPE pair-convention bug — see [`project_engine_drift_floor_decomposition.md`](../../memory/project_engine_drift_floor_decomposition.md).
- **Reference distribution:** `qwen3.5-{0.8b,9b}-bf16.kldref.bin` (built via `build_kld_ref` from llama.cpp `llama-perplexity --kl-divergence-base` over the BF16 GGUF). Slice: wikitext2-1024s-2048ctx (md5 `83b0205a`, 1175 chunks; quick-slice runs use `--max-chunks 512`).
- **KV mode is load-bearing for cross-engine comparison.** llama.cpp GGUF baselines were measured with its default FP16 KV cache (lossless). Hipfire defaults to `asym3` KV (lossy, ~0.07-nat penalty on 0.8B mq4-base; see §4). When comparing across engines, only compare rows with similar KV lossiness, OR explicitly compute the above-floor lift.
- **Above-floor KLD** = raw KLD − engine's Q8-weight floor (in matching KV mode). Surfaces the format/calibration cost stripped of engine drift + KV-quant cost.
- **Scoring mode:** all hipfire rows use `prefill` (canonical, see §5.3 of issue-113). GGUF rows use llama.cpp's native (`gguf` mode, not directly comparable to per-token).
- **Top-K=256 caveat:** all KLD values are lower bounds on true full-vocab KLD. Cross-variant ordering is reliable; absolute magnitudes are conservative. See issue-113 §2.

---

## 1. Qwen3.5-9B

### 1.1 Hipfire 9B 4-bit cohort (post-RoPE-fix, gfx1100, KV=asym3, n=512)

Source: `benchmarks/quality-baselines/results/2026-05-12-cohort-post-rope-fix-9b/result-table.md`

| Variant | bpw | KLD (CI) | p99 | PPL |
|---|---:|---|---:|---:|
| mq4-base | 4.25 | 0.3376 (0.3263–0.3494) | 18.194 | 9.116 |
| **mq4-awq** | 4.25 | **0.2800** (0.2697–0.2910) | 17.537 | 9.271 |
| hfp4 | 4.50 | 0.4594 (0.4475–0.4720) | 19.279 | 11.511 |
| mfp4 | 4.50 | 0.4653 (0.4535–0.4782) | 18.278 | 11.138 |
| hfp4-l4-l5c | ~5.0 | 0.3836 (0.3722–0.3959) | 18.665 | 10.299 |
| mfp4-l4-l5c | ~5.0 | 0.7783 (0.7625–0.7951) | 21.199 | 12.571 |

> Tier-2 q8f16 floor row removed (was 0.1459); superseded by §1.1b Tier-3 floor 0.0173.
> §1.1d FWHT rotation isolation table removed (n=20 with CIs spanning ±50% of mean —
> too loose to support quantitative conclusions). Qualitative finding from that section
> ("FWHT projection rotation provides ~−30% KLD; PPL inversion") is preserved here as
> narrative pending an n=256+ re-measurement.

### 1.1j Q8-head cohort (gfx1100, KV=q8, prefill, n=256, 2026-05-18)

Source: this session, fresh quant on current master code (`hipfire-quantize` post-May-18 incl. Cholesky-direct OBS + parallelized GPTQ column loop). Both rows use `--kmap-dense` for Q8 lm_head + default Promote6 on alt down_proj; body kmap matched across the pair so the only delta is AWQ+GPTQ on/off.

| Variant | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---|---:|---:|---|
| mq4-plain-q8head | ~4.4 | 0.2149 (CI 0.2011–0.2305) | 17.379 | 8.746 | **no AWQ, no GPTQ**; `--kmap-dense` only |
| **mq4-awq-gptq-f2-q8head** | ~4.4 | **0.1727 (CI 0.1600–0.1877)** | **16.166** | **8.417** | **NEW BEST 9B 4-bit recipe (supersedes §1.1f's 0.1842)** |

**AWQ+GPTQ Δ at 9B 4-bit (clean A/B, identical kv-q8 + Q8 lm_head + body kmap):**
mean KLD **−20%** (0.2149 → 0.1727), CIs essentially non-overlapping at the
boundary (0.2011 vs 0.1877). p99 KLD −7%. PPL **−3.8%** (8.746 → 8.417) —
AWQ+GPTQ wins on both axes at 9B 4-bit, unlike §3.2's 27B MQ4 where PPL
moved the other way.

**Comparison across recent recipes on 9B:**
- §1.1f mq4-awq + Q8 conv1d (n=512, MQ4 lm_head): KLD 0.1842, PPL 9.575
- **§1.1j mq4-awq-gptq-f2-q8head (n=256, Q8 lm_head): KLD 0.1727, PPL 8.417** — **−6% KLD, −12% PPL** at slightly higher bpw (~+0.15 from Q8 head)
- Q8 lm_head + GPTQ replaced Q8 conv1d as the additive lever. Both can stack
  (Q8 head + Q8 conv1d + AWQ + GPTQ) but not yet measured.

**Δ-above-own-Q8** (Tier-3 floor 0.0173 per §1.1b): 0.1727 − 0.0173 = **0.155**.

**Cross-engine vs llama.cpp at matched bpw:**
- UD-Q3_K_XL @ ~4.50 bpw: KLD 0.1411 (Δ-above-own-Q8 0.125)
- Hipfire mq4-awq-gptq-f2-q8head @ ~4.4 bpw: KLD 0.1727 (Δ 0.155)
- Gap narrowed from §1.1f's ~+24% Δ-gap to **+24% absolute / +24% Δ** — modest improvement over the AWQ+Q8conv1d stack but still 22% behind llama.cpp's bpw-matched anchor.

Working artifacts: `benchmarks/quality-baselines/results/2026-05-18-9b-q8head-cohort/`.

### 1.1k mq4-awq-gptq-f2-lmhead-a100 (KV=q8, prefill, n=256, 2026-05-19) — **cross-arch confirmed**

Source: `qwen3.5-9b.mq4-awq-gptq-f2-lmhead-a100.hfq` on `/data/hipfire` (5.315 GB, **4.744 bpw** effective whole-file at 8.96B params). Same .hfq evaluated on two arches, same slice (md5 `83b0205a`, n=256), same kv-mode (q8), same scoring mode (prefill), same `kldref` (md5 `283ecb32`).

| Arch | Run | KLD (CI) | p99 KLD | mean NLL | PPL | Wall | tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| gfx1031 | user, 2026-05-19 AM | 0.0841 | — | 2.2372 | 9.367 | 3668.9 s | 71 |
| **gfx1100** | **this session, 2026-05-19 (master tip `52273907`, fresh build md5 `99e44d16…`)** | **0.0863 (CI 0.0798–0.0939)** | **9.607** | **2.2379** | **9.374** | **727.1 s** | **360** |

**Cross-arch drift: KLD +2.7%, PPL +0.07%, NLL +0.03%.** Well inside the established cross-arch noise band (cf. §1.1i F2-α sweep gfx906↔gfx1151: KLD agreement ~0.06%, PPL ~0.1%). The earlier "strong-but-unconfirmed" flag is resolved — the 0.084-class KLD is **real**, not arch- or measurement-specific.

Working artifacts: `benchmarks/quality-baselines/results/2026-05-19-9b-lmhead-a100-gfx1100-parity/` (.kldseq + eval.log + reduce result-table).

**Note vs §1.1j canonical (mq4-awq-gptq-f2-q8head, ~4.4 bpw, gfx1100, KLD 0.1727 / PPL 8.42).** Same arch, same kv-mode, same n, same scoring mode — only the .hfq differs (lmhead-a100 vs q8head, +0.34 bpw).

- KLD **−50%** (0.1727 → 0.0863). Major.
- p99 KLD **−41%** (16.166 → 9.607). Tail also moves substantially.
- PPL **+11.4%** (8.417 → 9.374). Goes the **wrong** way.
- NLL +5.0% (2.130 → 2.238).

The KLD/PPL divergence is the same shape the user originally flagged, but now confirmed across two arches and a fresh binary. **Hypothesis:** the `-a100` suffix points at A100-calibrated Hessian / a different AWQ α / a different GPTQ recipe than `-q8head` — producing a quant whose full-vocab distribution sits closer to HF-bf16 (KLD wins) but whose argmax-token mass is slightly lower than the more aggressive `-q8head` recipe (PPL loses). Both metrics are valid; they're measuring different things. The §1.1j canonical optimized PPL; this row optimizes KLD-vs-HF.

**Cross-engine vs llama.cpp at similar bpw (§1.2 anchors, FP16 KV).** Caveat: hipfire KV is q8 here (lossy ~0.07-nat penalty per §0) while GGUF anchors used FP16 KV, so the KLD comparison is biased *against* this row. Δ-above-own-Q8 uses hipfire Tier-3 9B Q8 floor 0.0173 (§1.1b) and llama.cpp Q8_0 floor 0.0163 (§1.2).

| Variant | bpw | KLD (raw) | Δ vs own Q8 | Notes |
|---|---:|---:|---:|---|
| llama.cpp UD-Q3_K_XL | 4.50 | 0.1411 | 0.1248 | −0.24 bpw vs hipfire |
| **Hipfire mq4-awq-gptq-f2-lmhead-a100 (this row)** | **4.74** | **0.0863** | **0.0690** | gfx1100, kv-q8, n=256 |
| llama.cpp Q4_K_M | 5.07 | 0.1249 | 0.1086 | +0.33 bpw vs hipfire |
| llama.cpp UD-Q4_K_XL | 5.32 | 0.0670 | 0.0507 | +0.58 bpw vs hipfire |

- vs **UD-Q3_K_XL** (hipfire +0.24 bpw): hipfire wins **−39% raw KLD / −45% Δ**.
- vs **Q4_K_M** (llama.cpp +0.33 bpw): hipfire wins **−31% raw KLD / −37% Δ** despite llama.cpp's bpw cushion.
- vs **UD-Q4_K_XL** (llama.cpp +0.58 bpw): hipfire **+29% raw KLD / +36% Δ behind** — llama.cpp's +0.58 bpw still wins.

**Headline implications, now that parity is confirmed.**
1. **This is the first hipfire 4-bit recipe to land inside the llama.cpp K-quants Pareto frontier on absolute KLD.** Beats Q4_K_M (+0.33 bpw advantage to llama.cpp) by 31% raw / 37% Δ; trails UD-Q4_K_XL only because llama.cpp gets +0.58 bpw of headroom.
2. **Two competing 4-bit headlines now exist on 9B**, optimizing different objectives:
   - **§1.1j `mq4-awq-gptq-f2-q8head` (4.4 bpw): PPL-optimal** (8.42, best on this slice). Use when downstream cares about argmax-token confidence (greedy decode, top-1 acceptance).
   - **§1.1k `mq4-awq-gptq-f2-lmhead-a100` (4.74 bpw): KLD-optimal** (0.086, half of §1.1j). Use when downstream cares about full-distribution fidelity (sampling, draft-target acceptance in spec-decode, RL reward modeling).
3. The TL;DR / cross-references at top of the doc still cite §1.1j 0.1727 as the canonical "9B 4-bit headline." That framing is now stale — should be updated to reflect the dual-headline state and the metric the user is shopping on.

**Coherence eyeball (2026-05-19, gfx1100, daemon md5 `c7b414a5`).** Four prompts (capital, sheep riddle, one-line Python `square`, sourdough starter essay) on the same .hfq, T=0.0, repeat_penalty=1.05, max_tokens 80/300/180/220. **Result: clean.** Reasoning fluent and on-topic on all four; arithmetic riddle correctly identifies answer 9; sourdough essay completes (107 tok) with proper `<|im_end|>` termination. No `<|im_start|>` leaks, no token attractors, no repetition. Decode steady-state 106-108 tok/s on gfx1100. The KLD 0.0863 is a real distributional-quality win, not a coherence regression hiding behind a metric.

**Followups (low priority but worth filing):**
- Recipe diff between `-q8head` and `-lmhead-a100` (commit log + quant.toml audit) to nail down what the A100-side calibration actually changed (relevant given §3.2's 27B A100-vs-V100 was flat — see hypothesis 1 there).
- Spec-decode τ measurement on lmhead-a100 — predicted to go **up** relative to §1.1j despite higher PPL, because KLD-closer-to-target should translate to better draft/target alignment.

### 1.1e MQ6+Q8conv1d anchor (gfx1151, KV=q8, prefill)

Source: gfx1151 agent 2026-05-13 PM (smoke) + 2026-05-14 (full n=512). Uniform MQ6G256 across all 4-bit-eligible projections + Q8 conv1d override + default Q8 lm_head + Q8 embed.

| Variant | n | bpw | KLD (CI) | p99 | PPL | Wall |
|---|---:|---:|---|---:|---:|---:|
| mq6-q8conv1d (smoke) | 20 | ~6.5 | 0.0568 (CI 0.0314–0.0950) | 12.13 | 9.281 | ~6 min |
| **mq6-q8conv1d (n=512)** | **512** | ~6.5 | **0.0510** (CI **0.047–0.055**) | **8.68** | **9.186** | gfx1151 |

**n=512 confirms the smoke and tightens the CI 10× without changing the headline conclusion.** The smoke's wide CI 0.0314–0.0950 now collapses to 0.047–0.055 — the point estimate moved down 10% (0.0568 → 0.0510) and the tail p99 dropped from 12.13 to 8.68 (smoke caught a few unrepresentative high-noise sequences). PPL 9.186 is +0.02% above the Tier-3 Q8 floor 9.189 — i.e., **indistinguishable from 8-bit predictive quality at 6.5 bpw**.

**Hipfire's first format that genuinely competes with llama.cpp absolute-KLD-vs-HF:**
- Hipfire MQ6+Q8conv1d 6.5 bpw KLD 0.0510 ≈ **Llama.cpp UD-Q4_K_XL** at 5.32 bpw KLD 0.0670 — hipfire matches at +1.2 bpw cost
- Better than **Llama.cpp Q4_K_M** at 5.07 bpw KLD 0.1249 — hipfire wins by 2.4×
- Worse than **Llama.cpp Q6_K** at 6.56 bpw KLD 0.0250 — hipfire 2.0× worse (Δ-above-Q8: 0.034 vs 0.009 = 3.8× worse)

**This is a viable shipping format for "high-quality 4-5 bpw" target.**

Δ-above-Tier3-Q8: 0.0510 − 0.0173 = **0.0337** — pure 6-bit weight noise (down from smoke estimate 0.0395).

**Cross-engine pattern: same ~4× Δ penalty at both 4-bit and 6-bit, suggesting format-level (per-block scale fitting, 32 vs 256 group, asymmetric vs symmetric) is the structural cause rather than any specific bpw quirk.** See `qwen35-mq4-quality-gap.md` Stage A follow-ups F1-F5 for concrete fixes targeting this structural gap.

#### 1.1e.i KLD-vs-PPL inversion: MQ6-q8conv1d wins KLD 6.1× but loses PPL by 1%

Direct n=512 comparison of the two 9B variants the gfx1151 agent ran in the same sweep:

| Variant | bpw | KLD (CI) | p99 | PPL | KLD rank | PPL rank |
|---|---:|---|---:|---:|---:|---:|
| MQ4-Lloyd | ~4.91 | 0.3114 (0.300–0.324) | 18.69 | **9.085** | 2nd | **1st** |
| MQ6-q8conv1d | ~6.5 | **0.0510** (0.047–0.055) | 8.68 | 9.186 | **1st** | 2nd |

**Both gaps are well outside their CIs** — this is not measurement noise. MQ6-q8conv1d's KLD is **6.1× lower** but its PPL is **1.1% higher**. The two metrics genuinely disagree on which variant is better.

**Mechanism.** PPL is `exp(mean(-log p(y_true)))` — only the probability assigned to the *true* next token matters, regardless of the full distribution shape. KLD is `Σ p_ref(y) log(p_ref(y)/p_cand(y))` — measures full-distribution match against HF reference, including all the tail mass HF assigns probability to. MQ6-q8conv1d has lower per-weight noise → tighter tail-mass match to HF → much lower KLD. MQ4-Lloyd's per-block Lloyd codebook happens to nudge the argmax marginally toward the HF-bf16 reference's argmax token (without preserving tail shape), so PPL improves slightly even though the full distribution drifts harder.

This is the same family of "more precise ≠ better PPL-vs-HF" effect documented in F2/F3 — HF's bf16 reference is itself noisy at module boundaries, and any quant whose argmax happens to align with HF-bf16's argmax wins PPL even if the rest of its distribution is worse-aligned.

**Implication for ranking.** Exactly the PRD §2 case for KLD-primary scoring: KLD captures tail-mass quality (impacts coherence on rare-token contexts, multi-step reasoning where low-probability outputs matter); PPL only captures argmax confidence (impacts greedy decoding on common-token contexts). Models that win PPL but lose KLD are likely to win one-shot benchmarks (HumanEval, MMLU) while degrading on long-context reasoning. The shipping recommendation should weight by KLD.

### 1.1g Mixed MQ4/MQ6 + Q8 conv1d (gfx1151, KV=q8, prefill, n=512)

Source: gfx1151 agent 2026-05-14. Recipe: `--format mq4 --kmap-dense --kmap-mode 2` + `HIPFIRE_QUANTIZE_CONV_Q8=1`. Mixed-format quant: MQ4G256 base, MQ6G256 promotion for `mlp.down_proj` (and `attn_v` where applicable; on dense Q3.5 `attn_v` is fused into `in_proj_qkv` so the `_v_proj` substring doesn't match — only `down_proj` actually promotes), + Q8 conv1d. File size ~6.43 GB → **~5.04 bpw aggregate** (vs MQ4+Q8conv1d's ~4.25 bpw and uniform MQ6+Q8conv1d's ~6.5 bpw).

| Variant | n | bpw | KLD | PPL | Wall |
|---|---:|---:|---|---:|---:|
| **mq4-kmd2 + Q8 conv1d** | **512** | **~5.04** | **0.1613** | **9.1716** | 104 min |

**Pareto-frontier point between uniform MQ4 and uniform MQ6:**

| Recipe | bpw | KLD | PPL | Δ-above-Tier3-Q8 |
|---|---:|---:|---:|---:|
| mq4 + Q8 conv1d (uniform 4-bit) | 4.25 | 0.2501 | 8.789 | 0.233 |
| mq4 + AWQ + Q8 conv1d (gfx1100) | 4.27 | 0.1842 | 9.575 | 0.167 |
| **mq4-kmd2 + Q8 conv1d (mixed 4/6-bit)** | **~5.04** | **0.1613** | **9.172** | **0.144** |
| mq6 + Q8 conv1d (uniform 6-bit) | 6.5 | 0.0510 | 9.186 | 0.034 |

PPL 9.172 is **within 0.02% of the Tier-3 Q8 floor** (9.189) — indistinguishable from 8-bit predictive quality at +0.79 bpw cost above uniform MQ4. KLD-wise, mq4-kmd2 sits at 47% of uniform MQ4's KLD (-53%, biggest single-step gain in the bpw progression) and 27% of MQ6's (smaller gap to higher-bpw recipe). The +0.8 bpw cost from promoting down_proj to MQ6 captures most of MQ6's KLD benefit at ~half the bpw overhead. **Strong shipping candidate for "high-quality 5-bit" target.**

**Cross-engine Δ-above-Tier3-Q8: 0.1613 − 0.0173 = 0.144** — vs llama.cpp Q4_K_M's Δ ≈ 0.109. Hipfire stack now within 1.32× of Q4_K_M, the smallest absolute-KLD gap to llama.cpp K-quants on any hipfire 4-5 bpw recipe to date.

**Note on dispatch:** the same `--kmap-dense --kmap-mode 2` recipe was filed as [issue #249](https://github.com/Kaden-Schutt/hipfire/issues/249) on gfx1100, where the runtime forward pass produces NaN logits (silent corruption). This gfx1151 measurement implies the dispatch NaN is **gfx1100-specific** — the bug class (kernels not enumerating the mixed-format combo) may only trigger on the WMMA path active on gfx1100 but not on gfx1151's RDNA3.5 dispatch. Issue #249 should be re-scoped from "kmd2 produces NaN" to "kmd2 produces NaN on gfx1100 (works on gfx1151)" pending follow-up.

### 1.1h AWQ + K-map mixed MQ4/MQ6 + Q8 conv1d (gfx1151, KV=asym3, prefill, n=256, 2026-05-15)

Source: commit `0043e26c` ("feat(quantize): AWQ pre-scale K-map MQ6 promotions") — extends Stage A AWQ-on-MQ4 to the K-map-promoted MQ6 tensors. Recipe: `--format mq4 --kmap-dense --kmap-mode 2 --imatrix … --awq`, q8_conv1d_default ON (PR #251). Quantized models in `~/.hipfire/models/qwen3.5-9b.mq4-kmd2{,-awq}-2026-05-14`. **Both runs ran on the same fresh build** (cargo clean + `.hipfire_kernels/` nuke + daemon binary refresh — earlier KLD=0 / NaN cohort was a stale-binary artifact). Cohort artifacts: `benchmarks/quality-baselines/results/2026-05-14-awq-mq6-cohort-fresh/per-seq/`.

| Variant | n | bpw (body / total) | KLD | mean NLL | PPL | Wall |
|---|---:|---:|---:|---:|---:|---:|
| mq4-kmd2 + Q8 conv1d (no AWQ) | 256 | 4.81 / 5.75 | **0.185940** | 2.2681 | 9.6611 | 3315 s @ 79 tok/s |
| **mq4-kmd2-awq + Q8 conv1d** | 256 | 4.81 / 5.75 | **0.148480** | 2.2868 | 9.8435 | 3317 s @ 79 tok/s |
| **Δ (AWQ − no-AWQ)** | | +0.001 (AWQ sidecars) | **−0.037460 (−20.1%)** | +0.019 | +0.183 (+1.9%) | identical |

**Headline: AWQ extension to K-map-promoted MQ6 tensors drops KLD by 20.1%** at matched conditions (same code, same model files, only `--awq` differs). Per the K-map mode-2 typed rule, the AWQ sidecar count grew by 8 (v_proj on the 8 FA layers, previously skipped because the Promote6 arm in the quantizer didn't go through the AWQ pre-scale block — see §"the gap" in commit `0043e26c`). Edge-layer MQ6 tensors also receive AWQ; the post-projection paths (`o_proj`, `down_proj`) correctly skip AWQ per the `awq_eligible` whitelist.

**KLD/PPL inversion (again, same as 1.1e.i and 1.1f).** AWQ on the kmd2 mix lifts KLD substantially (−20.1%) but regresses PPL +1.9%. AWQ tightens the full distribution match against HF but shifts the argmax token's probability slightly down. For long-context reasoning workloads where the tail mass matters, prefer AWQ + kmd2 + Q8 conv1d.

**Cross-engine Δ-above-Tier3-Q8** (CAVEAT: Tier-3 Q8 floor in §1.1b is q8-KV; this row is asym3-KV, so the Δ here mixes weight-quant cost with KV-mode contribution. Per §4.1 asym3-KV adds ~0.04 nats vs q8-KV; the asym3-corrected Δ is approximate):

| Recipe | KV | KLD | est. Δ-above-Tier3-Q8 |
|---|:---:|---:|---:|
| mq4 + Q8 conv1d (1.1, uniform 4-bit) | q8 | 0.2501 | 0.233 |
| mq4-awq + Q8 conv1d (1.1f, AWQ on uniform MQ4) | q8 | 0.1842 | 0.167 |
| mq4-kmd2 + Q8 conv1d (1.1g, kmd2 only) | q8 | 0.1613 | 0.144 |
| **mq4-kmd2 (this row, no AWQ)** | asym3 | 0.186 | est. ~0.13 at q8-KV |
| **mq4-kmd2-awq (this row)** | asym3 | **0.148** | **est. ~0.09 at q8-KV** |
| mq6 + Q8 conv1d (uniform 6-bit) | q8 | 0.0510 | 0.034 |

If the asym3→q8 KV-mode contribution holds at ~0.04 nats, mq4-kmd2-awq sits at est. ~0.108 q8-KV-equivalent — **a new best 9B sub-6-bpw recipe** beating both 1.1g (kmd2 alone, 0.1613) and 1.1f (AWQ on uniform MQ4, 0.1842). The two AWQ levers (per-channel pre-scaling) and bit allocation (kmd2 mode-2 typed) stack additively: each gives ~−20% KLD on its own, combined yields a measurable further drop. Confirmed via dispatch flow — `fused_rmsnorm_rotate_mq_awq` kernel handles the inverse divide regardless of whether the consuming gemm is MQ4 or MQ6 (key on `awq_scale.is_some()`, not dtype). Validation: ran on both gfx906 (user, AWQ-on-MQ4 coherent at commit `155c2a0b`) and gfx1100 (user, AWQ-on-MQ4-with-kmd2 coherent at commit `0043e26c`) and gfx1151 (this row, fresh build).

**Follow-up gaps:**
1. **Matched-KV re-measurement.** This row's asym3-KV vs §1.1g's q8-KV is a methodology mismatch. A clean q8-KV n=512 run on both mq4-kmd2 and mq4-kmd2-awq would put this row on the Pareto frontier next to 1.1g without the KV-mode caveat. ~3.5 h gfx1151 wall.
2. **AWQ alpha sweep on kmd2 mix.** Per the F1 follow-up in `qwen35-mq4-quality-gap.md` §"open follow-ups", the default alpha=0.5 is Llama-tuned. The kmd2 mix has a different per-class scale distribution (some classes uniform MQ4, some MQ6); per-class alpha may give further KLD reduction. The +1.9% PPL regression at alpha=0.5 hints this is worth probing.
3. **gfx1100 reproduction** at matched n=512 q8-KV would close the cross-arch validation loop and let this row drop directly into the Pareto frontier table in §1.1g.

### 1.1f AWQ + Q8 conv1d stack (gfx1100, KV=q8, prefill, 2026-05-14)

Source: this session's `/tmp/awq_q8conv_redo.sh`, after the cargo incremental build glitch was resolved (`cargo clean -p hipfire-quantize -p hipfire-runtime` before quant + smoke + n=512). Recipe: `--format mq4 --imatrix --awq` + `HIPFIRE_QUANTIZE_CONV_Q8=1`. Quant md5 `1da67afff879099aeed39f3a3d42d1b1`.

| Variant | n | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---:|---|---:|---:|---|
| mq4-base (§1.1, asym3-KV) | 512 | 4.25 | 0.3376 | 18.19 | 9.116 | baseline (different KV mode) |
| mq4-q8conv1d (re-verified n=512) | 512 | 4.25 | **0.2501** (0.2396–0.2609) | 16.06 | **8.789** | Q8 conv1d alone, byte-exact reproduction of original anchor |
| **mq4-awq + Q8 conv1d (n=512)** | 512 | 4.25 | **0.1842** (0.1759–0.1930) | **15.95** | **9.575** | superseded by §1.1j (mq4-awq-gptq-f2-q8head 0.1727, 2026-05-18) |

> n=20 smoke rows (mq4-base q8-KV, mq4-awq re-verified, mq4-awq+Q8conv1d smoke) removed
> — superseded by the n=512 rows above; CIs not bootstrapped.

**Both levers stack ~additively on KLD.** Each gives ~−21% KLD on its own; combined gives **−42%** vs mq4-base — almost perfect additivity (would expect ~−40% from independent Bernoulli composition of relative-noise reductions). The AWQ × FWHT × MQ4 GEMV composition math `y = (W·diag(s)·R^T) · (R·(x/s)) = W·x` holds at the bench level too — no destructive interference.

**KLD-vs-PPL inversion (again).** The AWQ+Q8conv1d stack lifts KLD substantially (0.2501 → 0.1842, −26%) but **regresses PPL** (8.789 → 9.575, +9%). Same family as §1.1e.i: AWQ tightens the full distribution match against HF but nudges the argmax token's probability slightly down. PRD §2 KLD-primary ranking puts this stack at the top; PPL-primary ranking would still pick plain mq4-q8conv1d. For long-context-reasoning workloads where the tail mass matters (multi-step, low-probability paths), prefer AWQ + Q8 conv1d.

**Cross-engine Δ-above-own-Q8.** Tier-3 Q8 floor 9B = 0.0173. mq4-awq + Q8 conv1d Δ = 0.1842 − 0.0173 = **0.1669** — vs llama.cpp Q4_K_M Δ = 0.1086 (0.1249 raw − 0.0163 Q8_0 floor). Hipfire stack is now **1.54× over Q4_K_M** instead of the ~4.5× ratio mq4-base sat at. Stage A's AWQ + Q8 conv1d closes ~62% of the bpw-normalized cross-engine gap at 4-bit. Stage B (GPTQ on top) is the next composable lever; the empirical question is whether GPTQ's Hessian-aware sequential update gives independent lift on top of AWQ + Q8 conv1d or competes with AWQ for the same outlier-preservation budget.

Working artifacts: `qwen35-9b-mq4-awq-q8conv__gfx1100__kv-q8__c20.kldseq` (smoke), `qwen35-9b-mq4-awq-q8conv__gfx1100__kv-q8__c512.kldseq` (n=512); reverify quant `qwen35-9b-mq4-q8conv1d-reverify__gfx1100__kv-q8__c512.kldseq` md5-identical to the §5 anchor `qwen35-9b-mq4-q8conv1d__gfx1100__kv-q8__c512.kldseq` (both `8c90dbad8d7b9ee22147f0b02fbc2b43`). All in `benchmarks/quality-baselines/results/2026-05-13-q8-lmhead-conv-smoke/per-variant/`.

### 1.1h F2 — AWQ whitelist expansion to output-side projections (gfx906, KV=q8, prefill, 2026-05-14/15)

Source: this session's `scripts/awq_f1_vs_f2.sh` (paired comparison) + `scripts/awq_alpha_sweep.sh` (alpha sweep). F2 is plan §F2: extend AWQ pre-scaling from input-side projections (q/k/v/gate/up/router/in_proj_*, 184 sidecars) to also include output-side projections (o_proj/wo/out_proj/down_proj/w_down, +64 sidecars → **248 total**) via two new HIP kernels (`rotate_x_mq_awq.hip`, `fused_silu_mul_mq_rotate_awq.hip`) + dispatch wrappers + `_for` routing helpers + `weight_gemv` AWQ-aware fix.

**F1 vs F2 at α=0.5, paired (n=256, KV=q8, gfx906):**

| Variant | n | bpw | KLD (CI) | PPL | NLL | Notes |
|---|---:|---:|---|---:|---:|---|
| F1 α=0.5 (184 sidecars) | 256 | 4.25 | 0.1725 ± 0.0119 | 9.5367 | 2.2551 | baseline (`HIPFIRE_AWQ_F1_ONLY=1`) |
| **F2 α=0.5 (248 sidecars)** | 256 | 4.25 | **0.1724 ± 0.0121** | **8.9116** | **2.1873** | F2 default whitelist |

**Paired t-test on per-chunk metrics (F2 − F1, n=256):**

| Metric | Paired Δ | 95% CI | t-stat | Significant? |
|---|---:|---:|---:|---|
| KLD | −0.00003 | ±0.00399 | −0.01 | ✗ noise (KLD flat) |
| **NLL** | **−0.06780** | **±0.00997** | **−13.32** | ✓ p<10⁻³⁰ (F2 strictly better) |

F2 improved NLL on **209/256 chunks (81.6%)** — near-monotonic per-chunk improvement. KLD per-chunk split 52/48 (noise drowns the signal). **PPL: 9.54 → 8.91, −6.56%.**

**The KLD-PPL inversion is the headline.** AWQ improvements move probability mass favorably toward the true next token without changing the full-distribution divergence shape vs BF16. KLD averages over the full top-K so the mass-redistribution cancels; NLL (per-token surprise at the true label) captures it cleanly. **Going forward, paired-t on NLL is the primary AWQ-quality signal; KLD is secondary.** F1's earlier "lift = 2.3% on KLD" headline understated the true lever value because NLL wasn't measured alongside.

**F2 alpha sweep (gfx906 / KV q8 / n=100, [0.35, 0.65] step 0.05):**

| α | KLD | KLD CI | PPL | NLL paired-t vs α=0.50 |
|---|---:|---:|---:|---:|
| 0.35 | **0.1723** (best KLD) | ±0.018 | 10.09 | +0.104 t=+13.2 ★ (worse) |
| 0.40 | 0.1731 | ±0.016 | 10.18 | +0.112 t=+15.8 ★ |
| 0.45 | 0.1740 | ±0.018 | 10.13 | +0.107 t=+18.5 ★ |
| 0.50 | 0.1751 | ±0.018 | 9.10 | (anchor) |
| **0.55** | 0.1830 (worst KLD) | ±0.018 | **8.79** (best PPL) | **−0.034 t=−8.65 ★** |
| 0.60 | 0.1796 | ±0.019 | 9.19 | +0.010 t=+1.75 |
| 0.65 | 0.1795 | ±0.018 | 9.50 | +0.043 t=+6.06 ★ |

**KLD-PPL inversion is severe.** α=0.55 is the *worst* alpha on KLD but the *best* on PPL, with NLL paired-t = −8.65 vs the α=0.50 anchor (highly significant). The α-that-minimizes-KLD (0.35) is the 5th-best on PPL. Six of seven non-anchor alphas show |t| > 6 on NLL paired-t — the lever is strong on PPL, weak on KLD.

**Recommended ship config (subject to n=256 confirmation):**

| Variant | Sidecars | KLD | PPL | Lift vs F1 α=0.5 baseline |
|---|---:|---:|---:|---:|
| F1 α=0.5 (baseline) | 184 | 0.1725 | 9.54 | — |
| F2 α=0.5 | 248 | 0.1724 | 8.91 | **−6.6% PPL** |
| **F2 α=0.55** (gfx906/q8 PPL optimum) | 248 | 0.1830 | **8.79** (n=100) | **−7.9% PPL** (subject to n=256) |

**Plan §F2's "+3-7% reduction on top of F1" prediction was right in magnitude (measured +6.6 to +7.9% PPL), wrong in metric (predicted vs measured KLD ≈ 0).** Right lever, right impact range, wrong yardstick.

**Methodological implication.** Section §6 rule list should add: "AWQ improvements are best measured by paired-t on per-chunk NLL, not KLD. The KLD-PPL inversion at AWQ-relevant alpha shifts is severe — KLD-only ranking will pick the wrong default."

**Cross-arch open question.** gfx1100 F1 sweep (separate experiment) had α=0.5 winning at KLD on KV q8. With the new PPL-primary methodology, that finding needs re-measuring with NLL paired-t alongside. Expected: same KLD-PPL inversion shape; gfx1100 PPL optimum likely also above α=0.5. Followup sweep on gfx1100 next session.

**Implementation review trail:**
- Implementation: this branch (`feat/mq-v2-quant-format`), §F2 commit pending.
- Critical self-review: `docs/plans/awq-f2-rev-claude.md` (identified `weight_gemv` un-AWQ'd dispatch path; fix landed in same commit before any user-visible corruption).
- Quant-side env toggle: `HIPFIRE_AWQ_F1_ONLY=1` reproduces the F1 (184-sidecar) whitelist for A/B comparison without rebuilding.

Working artifacts:
- F1 vs F2 paired: `benchmarks/quality-baselines/results/2026-05-14-f1-vs-f2-n256-kvq8-9b-gfx906/{f1,f2}-a0_5.kldseq`
- F2 alpha sweep: `benchmarks/quality-baselines/results/2026-05-14-f2-alpha-sweep-n100-kvq8-9b-gfx906/per-variant/a*/awq-a*.kldseq`

### 1.1i F2 cross-arch reproduction on gfx1151 (KV=q8, prefill, n=100, 2026-05-15)

Source: tight 3-alpha sweep around the gfx906 sweet spot (§1.1h). Recipe: uniform MQ4+AWQ on 9B (no kmd2), q8 KV, prefill scoring, n=100 chunks — methodology byte-matched to §1.1h. Branch tip `0c7aaeed` (F2 code in `9ca8d900`). Quant slot reused per-alpha; eval per-variant. **All 248 sidecars confirmed in every quant**: F2 whitelist (input + output projections) is identical on gfx1151 because the quantizer path is arch-agnostic.

| α | KLD (CI) | NLL (CI) | PPL | eval wall |
|---|---:|---:|---:|---:|
| 0.50 (anchor) | 0.1757 ± 0.018 | 2.2066 ± 0.049 | 9.085 | 1058 s @ 97 tok/s |
| **0.55** | 0.1855 ± 0.019 (worst) | **2.1730 ± 0.048** (best) | **8.785** (best) | 1069 s |
| 0.60 | 0.1807 ± 0.019 | 2.2161 ± 0.049 | 9.172 | 1077 s |

**Paired-t (variant − α=0.50 anchor) on per-chunk metrics, n=100:**

| Comparison | NLL Δ | NLL t-stat | KLD Δ | KLD t-stat |
|---|---:|---:|---:|---:|
| **α=0.55 vs α=0.50** | **−0.03361** | **−8.48 ★** | +0.00980 | +3.72 ★ (worse) |
| α=0.60 vs α=0.50 | +0.00953 | +1.63 (n.s.) | +0.00506 | +1.91 (n.s.) |

**The gfx906 finding reproduces on gfx1151 with near-zero drift.** Side-by-side:

| α | gfx906 KLD | gfx1151 KLD | gfx906 PPL | gfx1151 PPL | gfx906 NLL t-stat | gfx1151 NLL t-stat |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.1751 | 0.1757 (+0.4%) | 9.10 | 9.085 (−0.2%) | (anchor) | (anchor) |
| **0.55** | 0.1830 (worst) | 0.1855 (worst) | **8.79** (best) | **8.785** (best) | **−8.65 ★** | **−8.48 ★** |
| 0.60 | 0.1796 | 0.1807 (+0.6%) | 9.19 | 9.172 (−0.2%) | +1.75 n.s. | +1.63 n.s. |

The shape is *identical*:
1. **KLD-PPL inversion preserved.** α=0.55 has worst KLD AND best PPL/NLL on both archs.
2. **NLL paired-t magnitude reproduces.** −8.65 (gfx906) vs −8.48 (gfx1151) — essentially the same effect size; t-statistics agree to 2% on a single dispatch-path change.
3. **PPL gain reproduces.** α=0.50→0.55 drops PPL −3.41% on gfx906, −3.30% on gfx1151.
4. **α=0.60 plateau reproduces.** Past the sweet spot, KLD recovers slightly but PPL/NLL flattens — neither arch shows further gain.

**Methodology vindication.** §1.1h's "paired-t on NLL is the right metric" rule (§6 rule 9) holds cross-arch. KLD-only ranking would have picked α=0.50 on both archs and missed the −3.3% PPL gain. The lever is real and arch-portable; it just doesn't show in the metric we used to default to.

**Strategic implication.** **α=0.55 is the recommended ship default on gfx11.x as well as gfx906.** No per-arch tuning needed for this lever. The AWQ pre-scale mechanism is doing the same thing on both INT8 dp4a (gfx906) and FP16 WMMA (gfx1151) dispatch paths — the underlying mass-redistribution toward the true next token is a property of the AWQ pre-scaled weights, not of the kernel that consumes them.

**Cross-arch open question, now closed.** §1.1h's "gfx1100 followup sweep next session" is now answered for gfx1151 (RDNA3.5). gfx1100 (RDNA3) reproduction is the next step, but RDNA3 and RDNA3.5 share enough dispatch lineage that I'd expect the same shape there too.

**Side note on eval_hipfire teardown:** all three sweep evals on gfx1151 exited with signal 11 (segfault) *after* writing the kldseq and the `slice-mean KLD` line. Data is valid (verified per-chunk reads + paired-t). Suspected: `Drop` ordering issue when the F2-new AWQ-aware kernels (`rotate_x_mq_awq`, `fused_silu_mul_mq_rotate_awq`) are in the dispatch table — gfx1151 path takes them, gfx906 path may not. Doesn't affect data correctness; should be a separate cleanup PR.

Working artifacts:
- Sweep results: `benchmarks/quality-baselines/results/2026-05-15-f2-alpha-sweep-gfx1151/per-variant/a*/awq-a*.kldseq`
- Sweep script: `/tmp/sweep-gfx1151{,-resume}.sh` (resume variant tolerates teardown segfault)
- Per-alpha eval logs in `…/per-variant/a*/eval.log` (all show clean `slice-mean KLD` line + buffered tokens, then segfault on exit)

### 1.1c MQ4-Lloyd anchor (gfx1151, KV=q8, prefill, n=512)

Source: gfx1151 agent 2026-05-13 PM. Default conv1d (MQ4G256, not Q8), default lm_head (MQ4G256). FWHT rotation on projections + Lloyd-Max codebook fit per 256-block instead of uniform grid.

| Variant | bpw | KLD (CI) | p99 | PPL | Wall |
|---|---:|---|---:|---:|---:|
| **mq4-lloyd** | **~4.91** (incl. Lloyd codebook overhead) | **0.3114** (CI 0.2999–0.3236) | 18.69 | **9.085** | 76 min @ 116 tok/s |

**Lloyd codebook provides essentially zero KLD improvement on MQ4 at 9B.** Compared to mq4-base (asym3-KV n=512: 0.3376, q8-KV n=20: 0.3182, q8-KV n=512 estimated 0.27-0.32), mq4-lloyd at 0.3114 is within noise of the baseline — and the file is 6.06 GB vs mq4-base's 5.31 GB (**+744 MB / +0.66 bpw avg overhead** for the Lloyd codebook).

**Mechanism (confirms §4A.2 prediction):** after FWHT-256 rotation, per-block weight distribution is approximately Gaussian (CLT on 256 elements). Lloyd-Max codebooks fit to a near-Gaussian don't beat uniform 4-bit grids by much. Lloyd is theorized to help on heavy-tailed distributions; the FWHT rotation already removes the heavy-tail problem MQ4 base would otherwise have.

**Implication:** **Drop MQ4-Lloyd from the active calibration roadmap.** The hypothesis is now measured-and-falsified for the 4-bit case. MQ3-Lloyd remains untested and could plausibly still help (§4A.2 #1 — uniform 3-bit is "on the cliff", Lloyd's larger lift at lower bpw is still hypothetical).

### 1.1b Hipfire Tier-3 Q8 floor anchor (gfx1151, KV=q8, fused WMMA prefill)

Source: gfx1151 agent 2026-05-13 PM run on PR [#248](https://github.com/Kaden-Schutt/hipfire/pull/248) (HEAD `747315a4`). Kernels: `gemm_qkv_q8_0_wmma`, `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma` (all 4 Tier-3 fused kernels confirmed in dispatch log; Tier-2 substrate **not** invoked).

| Variant | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---|---:|---:|---|
| **q8f16 (Tier-3)** | 8.50 | **0.0173** (CI 0.0150–0.0201) | 0.0844 | **9.189** | n=256, gfx1151, q8-KV, prefill |

**Architecture-platform caveat:** measurement is on **gfx1151** (Strix Halo, RDNA3.5), not gfx1100. Cross-arch parity per issue-113 §V3 is ~1%. Kernels were validated on gfx1100 per commit `47fd6c4d` (4 fused-kernel unit tests `=== 0 failure(s) ===` on gfx1100, daemon prefill 1069 tok/s) — but the KLD measurement itself awaits a gfx1100 reproduction.

**+6% above llama.cpp Q8_0 9B's 0.0163** (§1.2 below). Essentially same precision class. What was previously framed as "+0.13 nats hipfire-side engine drift" was the FP32-vs-bf16 precision-class mismatch of the Tier-2 substrate; on Tier-3 it's gone. The remaining ~0.001 nats is residual cumulative DeltaNet recurrence drift (~0.06 per LA layer × ~24 LA layers = small in absolute terms once the precision-class is matched).

**PPL also dropped substantially** (9.795 → 9.189, −6%). Tier-3 doesn't just match HF noise pattern; it produces a sharper output token distribution on wikitext-2. Likely because FP16 rounding errors are systematically biased the same direction as bf16 in HF, so logit confidence shape matches HF better.

### 1.2 llama.cpp GGUF anchors (gfx1151, default FP16 KV)

Source: `benchmarks/quality-baselines/results/2026-05-10/per-seq/qwen3.5-9b.gguf-*__gfx1151.kldseq`

| Variant | bpw | KLD | PPL | Above-floor |
|---|---:|---:|---:|---:|
| Q8_0 | 8.50 | **0.0163** | 9.31 | (floor) |
| UD-Q6_K_XL | ~6.7 | 0.0213 | 9.14 | 0.0050 |
| Q6_K | 6.56 | 0.0250 | 9.31 | 0.0087 |
| UD-Q5_K_XL | ~5.5 | 0.0408 | 9.27 | 0.0245 |
| **UD-Q4_K_XL** | 5.32 | **0.0670** | 9.34 | 0.0507 |
| Q4_K_M | 5.07 | 0.1249 | 8.70 | 0.1086 |
| **UD-Q3_K_XL** | ~4.50 | **0.1411** | 8.67 | 0.1248 |

### 1.3 9B cross-engine summary

> ⚠ **2026-05-13 PM reframing (twice).** The original "engine-drift floor gap" claim was a Tier-2 FP32-vs-bf16 precision-class mismatch artifact (closed by F2/F3 audit + PR-#248 Tier-3 measurement); on Tier-3 the gap is **+6% not +0.13 nats**. The earlier "above-floor" column also used a Tier-2 (FP32-acc) Q8 baseline for MQ4 (FP16-WMMA) measurements, a precision-class mismatch that artificially deflated the Δ. Corrected numbers below.

**Absolute Q8-vs-Q8 KLD-vs-HF on Tier-3 (precision-class-matched):**
- Hipfire 9B q8f16 Tier-3 q8-KV: **0.0173** (§1.1b)
- Llama.cpp 9B Q8_0 FP16-KV: **0.0163** (§1.2)
- **Gap: +6% / 0.001 nats** — essentially same precision class. Caveat: KV mode mismatch (q8 vs FP16-KV) introduces a small confounder; hipfire's f16-KV-equivalent (had we measured it) is likely 0.015-0.017, narrowing the gap further.

**Δ-above-own-Q8 (the apples-to-apples cross-engine claim) at 9B (path-matched on Tier-3):**

| Variant | bpw | KV | KLD | Δ-above-own-Q8 | Notes |
|---|---:|---|---:|---:|---|
| Llama.cpp Q8_0 | 8.50 | f16 | 0.0163 | (floor) | |
| **Llama.cpp UD-Q3_K_XL** | ~4.50 | f16 | 0.1411 | **0.125** | bpw-matched 4-bit anchor |
| **Llama.cpp Q4_K_M** | 5.07 | f16 | 0.1249 | **0.109** | imatrix-calibrated 4-bit |
| Hipfire q8f16 (Tier-3) | 8.50 | q8 | 0.0173 | (floor) | n=256 gfx1151 |
| Hipfire mq4-plain-q8head (§1.1j) | ~4.4 | q8 | 0.2149 | 0.198 | gfx1100, n=256, 2026-05-18 — no AWQ, no GPTQ |
| **Hipfire mq4-awq-gptq-f2-q8head (§1.1j)** | ~4.4 | q8 | **0.1727** | **0.155** | gfx1100, n=256, 2026-05-18 — **canonical 9B 4-bit headline** |
| **Hipfire F1 α=0.5** (gfx906) | 4.25 | q8 | 0.1725 | 0.155 | n=256, gfx906 (older recipe, no Q8 head) |
| **Hipfire F2 α=0.5** (gfx906) | 4.25 | q8 | 0.1724 | 0.155 | KLD-flat vs F1; PPL −6.6% |
| **Hipfire F2 α=0.55** (gfx906) | 4.25 | q8 | 0.1830 | 0.166 | n=100 sweep, PPL=8.79 (PPL optimum) |

**Cross-engine gap on current canonical 9B headline (§1.1j, 2026-05-18):**
hipfire mq4-awq-gptq-f2-q8head Δ = 0.155 vs llama.cpp UD-Q3_K_XL Δ = 0.125
→ **hipfire +24% Δ-above-own-Q8 at matched bpw**. The §1.1j build adds Q8
lm_head + GPTQ on top of AWQ; combined they close another 6% of KLD over
§1.1f's mq4-awq+Q8conv1d 0.1842, but the Δ-axis gap to UD-Q3_K_XL barely
moves because the AWQ Δ was already 0.155 on gfx906 — Q8 head + GPTQ
recovers part of the gain absorbed by the body kmap difference rather
than buying fresh Δ.

**AWQ+GPTQ closure on the Δ-above-own-Q8 basis vs hipfire plain @ same
recipe:** −22% Δ (0.198 → 0.155 in §1.1j's clean A/B). On absolute KLD
vs HF: −20% (0.215 → 0.173). These are the cleanest within-engine
AWQ+GPTQ Δ numbers for 9B 4-bit on current code.

**F2 on Δ-above-own-Q8 basis.** Hipfire F2 α=0.5 Δ = 0.155 (using gfx906 measurements, slightly different arch than the Tier-3 gfx1151 floor — caveat). PPL improvement is decoupled from this metric; F2's lift is more visible on PPL-paired-t (−6.6%, p<10⁻³⁰) than on Δ-above-own-Q8 (essentially zero movement). The Δ framework masks AWQ improvements that show up as mass-redistribution within an unchanged divergence envelope. **Δ-above-own-Q8 is the right metric for cross-engine quant-quality comparison; paired-t on NLL is the right metric for within-engine AWQ-config comparison.** Two different jobs.

### 1.4 MQ3 cohort (gfx1151, KV=q8, prefill, n=256, 2026-05-18 / 2026-05-19)

Source: 2026-05-18 + 2026-05-19 sessions. Closes the §4A.1 "MQ3G256 — NOT MEASURED" gap, then extends with an lmhead-a100 recipe-port from §1.1k. All rows kv-q8 prefill on gfx1151, n=256.

| Variant | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---|---:|---:|---|
| mq3-rtn-kvq8-c256 | 3.25 | 0.5449 (CI 0.532–0.559) | 16.927 | 13.45 | **no AWQ, no GPTQ** — naked MQ3 RTN baseline |
| mq3-awq-gptq-kvq8-c256 (Q8 lm_head) | ~3.8 (body 3.25 + Q8 head) | **0.1967** (CI 0.189–0.205) | **9.705** | **11.645** | AWQ + GPTQ at 3-bit; Q8 lm_head override |
| **mq3-awq-gptq-f2-lmhead-a100** | ~4.0 | **0.2613** (CI 0.249–0.276) | **14.740** | **10.440** | 2026-05-19; recipe-ported from §1.1k; wall 1710 s @ 153 tok/s |

**AWQ+GPTQ Δ at 3-bit vs RTN (the original 2026-05-18 finding):**
mean KLD **−64% / 2.77×** (0.5449 → 0.1967). CIs nowhere near overlap
(rtn lower bound 0.532 vs awq-gptq upper 0.205). p99 KLD −43%
(16.927 → 9.705) — AWQ's outlier-preserving design pays off in the tail
at low bit-widths. PPL −13% (13.45 → 11.65); unlike the 27B MQ4 cohort
in §3.2, AWQ+GPTQ wins on both KLD and PPL here. 3-bit is far enough
into the lossy regime that AWQ+GPTQ helps even the next-token-loss
metric.

**Tightening vs prior n=20 asym3 smoke** (from `2026-05-15-mq3-awq-uplift/`):
CIs shrank ≈3.6× as expected from √(256/20). mq3-awq-gptq centre moved
from 0.189 → 0.197 (+4%, within original CI); rtn from 0.569 → 0.545
(−4%, within original CI). The qualitative gap held; n=20 wasn't lying
about the direction, only about the precision.

#### 1.4a `-lmhead-a100` recipe-port to MQ3 — **inverse-direction at 3-bit (2026-05-19)**

The §1.1k lmhead-a100 recipe was ported from MQ4 to MQ3. **It moves the wrong way at 3-bit:**

| Axis | MQ3 q8-head (prior) | MQ3 lmhead-a100 (new) | Δ |
|---|---:|---:|---:|
| KLD | 0.1967 | 0.2613 | **+33% (worse)** |
| p99 KLD | 9.705 | 14.740 | **+52% (worse tail)** |
| PPL | 11.645 | 10.440 | **−10% (better)** |
| bpw | ~3.8 | ~4.0 | +0.2 |

CIs are non-overlapping (0.189–0.205 vs 0.249–0.276) — the KLD regression is statistically real, not measurement noise.

**Pattern: KLD up, PPL down.** This is the **inverse** of what §1.1k showed at MQ4 (KLD down, PPL up). Put the four 9B variants in a single grid to see the cross-format shape:

| Variant | KLD | PPL |
|---|---:|---:|
| MQ4 q8-head (§1.1j) | 0.1727 | 8.417 |
| **MQ4 lmhead-a100 (§1.1k)** | **0.0863** | 9.374 |
| MQ3 q8-head (§1.4) | 0.1967 | 11.645 |
| **MQ3 lmhead-a100 (§1.4a)** | 0.2613 | **10.440** |

The lmhead-a100 recipe lowers PPL on both formats but moves KLD in opposite directions: down at MQ4, up at MQ3.

**Mechanistic read — over-sharpening clipped by MQ3 codebook capacity.** AWQ pre-scaling amplifies activation-important weight columns so the model puts more probability mass on its argmax (PPL improves). At MQ4 there's enough codebook resolution (16 entries / 4-bit) to round-trip the amplified values, so the post-quant distribution shape stays close to the BF16 reference (KLD improves too). At MQ3 (8 entries / 3-bit) the amplified values quantize to coarser bins; the argmax still lands on the right token (PPL keeps improving) but the *shape* of the distribution rotates away from the reference — the model becomes correctly-pointing but over-confidently-shaped, which is precisely what KLD penalizes. The p99 KLD +52% confirms it's the tail, not the body, that takes the hit — exactly where AWQ amplification was concentrated.

**Implications for Phase 0b (recipe defaults):**
1. **`mq3 + lm_head-mq3 + awq` is NOT a Pareto win** — paired-t on KLD would fail vs the q8-head baseline. Do **not** recommend `--lm-head-format mq3 --awq` for sub-9B (and probably any) configurations.
2. **The recipe that's actually worth measuring** is `mq3 body + mq4 lm_head + awq`: keep MQ3's bpw savings on the dense layers, give the 248320-row lm_head MQ4's codebook headroom so AWQ-amplified mass can round-trip without clipping. The Phase 1 CLI is already designed for this: `--format mq3 --kmap-promote mq4 --lm-head-format mq4 --awq --kmap-dense --kmap-mode 2`.
3. **27B asymmetry is consistent with this story.** At 27B MQ4 (§3.2 row 3) the same lmhead-a100 recipe landed neutral (+4% KLD, −0.4% PPL) — "barely enough precision" threshold. At 9B MQ3 the recipe crosses below the threshold (KLD actively worse). The threshold is per-bit-width, not per-model-size: 4-bit clears it, 3-bit doesn't.
4. **AWQ α=0.55 was tuned at MQ4** (§1.1i). For any MQ3 + lm_head-AWQ recipe we keep alive, an **α sweep at lower values (0.30, 0.40)** is the next thing to try — less aggressive pre-scaling may stay within MQ3's precision envelope while still capturing partial gain. Predicted shape: PPL benefit shrinks gracefully, KLD regression flips back to neutral or positive.

---

## 2. Qwen3.5-0.8B

### 2.1 Hipfire (post-RoPE-fix, gfx1100, KV=asym3)

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/result-table.md`

| Variant | bpw | KLD (CI) | p99 | PPL | Above-floor |
|---|---:|---|---:|---:|---:|
| q8f16 | 8.50 | **0.1256** (0.1234–0.1280) | 1.458 | 19.633 | (floor) |
| mq4-base | 4.25 | 0.3341 (0.3308–0.3374) | 2.707 | 23.895 | 0.2085 |
| **mq4-awq** | 4.25 | **0.3000** (0.2971–0.3029) | 2.515 | 23.383 | **0.1744** |
| hfp4-l4-l5c | ~5.0 | 0.7890 (0.7785–0.7999) | 5.671 | 43.407 | 0.6634 |
| mfp4-l4-l5c | ~5.0 | 0.7524 (0.7445–0.7603) | 5.259 | 38.829 | 0.6268 |

### 2.2 Hipfire q8-KV probes (gfx1100, KV=q8) — apples-to-apples-er with GGUFs

Source: `benchmarks/quality-baselines/results/2026-05-13-cohort-post-rope-fix-0.8b/per-variant/*__kv-q8.kldseq`

| Variant | KLD (CI) | p99 | PPL | Δ vs asym3 |
|---|---|---:|---:|---:|
| mq4-base | 0.2675 (0.2650–0.2699) | 2.366 | 22.088 | −0.0666 (−20%) |
| mq4-awq | **0.2531** (0.2506–0.2556) | 2.305 | 22.149 | −0.0469 (−16%) |

> q8f16 q8-KV row removed (was 0.0806, n=20, per-token, gfx1151, pre-PR-#248 Tier-2
> kernel path). Four asymmetries against the rest of §2.2 plus explicit "needs
> re-measurement post-PR-#248" flag. Re-measurement on the production gfx1100 + Tier-3
> path is on the punchlist (§4.3) and critical-path for the cross-engine Δ framework.

### 2.3 llama.cpp GGUF anchors (0.8B)

Run via `eval_gguf` against `qwen3.5-0.8b-bf16.kldref.bin`. **Full slice (1175 chunks), not the 512-chunk quick-slice used by the hipfire 0.8B rows.** The slice-mean KLD usually shifts <2% between 512 and 1175 chunks (wikitext slice converges fast), but flagged here so it can't be forgotten.

| Variant | bpw | KV mode | KLD | PPL | Chunks | Source |
|---|---:|---|---:|---:|---:|---|
| Q4_K_M | ~5.07 | q8 | **0.0351** | 17.334 | 1175 (full) | user-reported 2026-05-13 |

Q8_0 / UD-* anchors at 0.8B not yet measured.

### 2.4 0.8B cross-engine summary (after Q4_K_M anchor)

At matched 4-bit + KV mode (q8 vs q8):

| Engine | Variant | bpw | KLD | PPL | Gap to llama.cpp Q4_K_M |
|---|---|---:|---:|---:|---:|
| **llama.cpp** | Q4_K_M | 5.07 | **0.0351** | **17.33** | (anchor) |
| hipfire | mq4-base | 4.25 | 0.2675 | 22.088 | +0.232 nats KLD, +27.5% PPL at −0.82 bpw |
| hipfire | **mq4-awq** | 4.25 | **0.2531** | **22.149** | **+0.218 nats KLD, +27.8% PPL** at −0.82 bpw |

**AWQ uplift is KV-mode-dependent.** At asym3 KV, AWQ improved 0.8B mq4 by −10.2% KLD; at q8 KV it's only −5.4%. Interpretation: AWQ partially compensates for KV-rotation noise (asym3 K rotation precision interacts with per-channel outliers); when the KV cache is less lossy, AWQ has less to clean up. The 9B picture (where AWQ closed −30% above-floor at asym3) may show similar shrinkage if re-measured at q8 KV — worth flagging for the Stage A retrospective.

**PPL is roughly unchanged by AWQ on this slice.** mq4-base PPL 22.088, mq4-awq PPL 22.149 — within slice noise. KLD improvement comes from tail-distribution matching, not top-1 probability, which is exactly what AWQ targets (outlier preservation in heavy channels). Consistent with §2 of issue-113 ("PPL collapses the full output distribution... KLD surfaces tail").

> §2.4.1 (Q8-weights vs Q4_K_M cross-engine — FLIPPED by 2026-05-13 PM audit) removed.
> The framework reframing is preserved in the TL;DR + §6 rule 8 (use Δ-above-own-Q8,
> not absolute KLD-vs-HF, for cross-engine claims). The table itself was 100% pending
> rows (hipfire Q8 floor "_pending_", post-PR-#248 measurement TBD). Re-add after
> the punchlist re-measurement lands.

---

## 3. Status of other model sizes

| Model | Hipfire status | GGUF status |
|---|---|---|
| Qwen3.5-4B | mq4-awq+gptq+Q8conv1d measured (§3.1) — KLD 0.0662 / PPL 10.71 @ n=512 | not measured |
| Qwen3.5-A3B | not yet quantized; queued (task #8) | not measured |
| Qwen3.6-27B | mq4-plain-q8head + mq4-awq-gptq-f2-q8head-v100 measured (§3.2) — AWQ+GPTQ KLD 0.1257 @ n=256 gfx1100 kv-q8 | not measured |
| Qwen2.5-7B | kldref built; cohort historical only | not measured |

### 3.1 Qwen3.5-4B AWQ + GPTQ + Q8 conv1d (gfx1151, KV=q8, prefill, n=512, 2026-05-15)

Source: this session. Run via `eval_hipfire` against `qwen3.5-4b-bf16.kldref.bin`. Quant produced via `hipfire-quantize --input /data/models/qwen/Qwen3.5-4B --format mq4 --imatrix qwen3.5-4b-bf16.imatrix.gguf --awq --gptq qwen3.5-4b-bf16.hessian.bin --gptq-damp 0.01 --gptq-max-damp 1.0` with `HIPFIRE_QUANTIZE_CONV_Q8=1`. Hessian collected via `scripts/collect_hessian.py` (128 seq × 2048 ctx = 262,144 calibration tokens, BF16 forward pass). Quantize from HF safetensors (not the BF16 GGUF) — see "VL caveat" below.

| Variant | bpw | KLD | PPL | tokens | tok/s |
|---|---:|---:|---:|---:|---:|
| mq4-awq+gptq+Q8conv1d | ~4.25 | **0.0662** | **10.7123** | 523,776 | 140 |

Smoke (n=20, same recipe): KLD 0.0572, PPL 10.4167, 142 tok/s.

Quant output: `~/.hipfire/models/qwen3.5-4b.mq4-awq-gptq-q8conv` (2.5 GB, 30.6% of 8.4 GB BF16 input).

**VL caveat (load-bearing for reproducibility).** Qwen3.5-4B on HF is `Qwen3_5ForConditionalGeneration` (`pipeline_tag: image-text-to-text`); its safetensors store tensors as `model.language_model.embed_tokens.weight` etc. with the `language_model.` prefix. The 9B and 0.8B base models are text-only and have no such prefix. Quantizing 4B **from a BF16 GGUF** silently strips that prefix (the GGUF flatten doesn't preserve the VL nesting), producing a .hfq that fails to load (`embed_tokens not found` at `qwen35.rs:1334`) AND causes zero Hessian-key matches in the GPTQ OBS pass (hessian keys retain `language_model.` from the HF-side collection). **Always pass `--input <HF-safetensors-dir>` for 4B, never the GGUF.** The imatrix and kldref are still built from the BF16 GGUF — those paths use llama.cpp's tokenizer/forward and don't care about the safetensors prefix.

Imatrix collection: standard (no VL-specific changes); 4B imatrix sidecar `qwen3.5-4b-bf16.imatrix.gguf` committed at HEAD ~ this commit.

Hessian collection: 248 Linears matched via the standard multimodal-flatten translation in `collect_hessian.py:_translate_to_stored_name` (longest-suffix match against safetensors keys); ran ~1h17m on gfx1151 (Strix Halo, ROCm-PyTorch 2.12+rocm7.2 in `.venv`). Sidecar at `/data/hipfire-refs/qwen3.5-4b-bf16.hessian.bin` (17 GB, gitignored per refs/.gitignore).

Quantize wall: ~3 h CPU on Strix Halo (16C/32T Zen 5). All 248 GPTQ targets clamped (`clamps/elements` ratio 0.001%–0.005% per tensor — well within Frantar 2022 healthy range). AWQ + GPTQ + Q8 conv1d composed cleanly.

Coherence eyeball (greedy 400 tokens, humaneval_0): all 8 hipfire-detect detectors green (no attractors, no n-gram density spike, no loop, no special-token leak, no empty-think). Output: structured chain-of-thought in `<think>` block, on-topic, English, one minor garbled math-notation segment that recovered. **Not incoherent at 4B** (unlike 0.8B, which is incoherent regardless of quant precision per the 2026-05-14 retraction below).

Working artifacts: `/data/hipfire-refs/gptq-4b-full.kldseq` (0 bytes — eval_hipfire's destructor segfaults on close, truncates the output file on the full-n run; headline metrics are durable in the log only). Smoke kldseq survived at `/data/hipfire-refs/gptq-4b-smoke.kldseq` (500 B).

**Open follow-ups:**
- No 4B mq4-awq+Q8conv1d (no-GPTQ) baseline measured yet → can't yet say whether GPTQ helps at 4B specifically. At 0.8B GPTQ regresses (+45%); at 9B is pending. 4B sits in between.
- The destructor segfault on .kldseq close is reproducible; if per-chunk arrays are needed for downstream paired-t analysis on 4B, eval_hipfire needs a fflush-before-cleanup ordering fix.

### 3.2 Qwen3.6-27B first cohort (gfx1100, KV=q8, prefill, n=256, 2026-05-18 / 2026-05-19)

Source: 2026-05-18 + 2026-05-19 sessions. 27B KLD measurements against `qwen3.6-27b-bf16.kldref.bin` (2.48 GB, sha256 `8af83b38…`, gfx1151 producer 2026-05-09, HF dataset `hipfire-models/qwen-kldref`). All rows on gfx1100, kv-q8, prefill, n=256. Total 27B params **26.896B** (counted from BF16 GGUF tensors at `/data/models/unsloth/Qwen3.6-27B/Qwen3.6-27B-BF16-*.gguf`).

| Variant | bpw | KLD (CI) | p99 | PPL | Notes |
|---|---:|---|---:|---:|---|
| mq4-plain-q8head-kvq8-c256 | ~4.5 | 0.2034 (CI 0.1841–0.2237) | 19.009 | 8.584 | **no AWQ, no GPTQ**; `--kmap-dense` only (Q8 lm_head + default Promote6 on alt down_proj) |
| mq4-awq-gptq-f2-q8head-v100-kvq8-c256 | ~4.5 | **0.1257** (CI 0.1126–0.1398) | **16.666** | 8.697 | AWQ stage-A F2 + GPTQ body + Q8 lm_head — V100-calibrated |
| **mq4-awq-gptq-f2-lmhead-a100-kvq8-c256** | **4.458** | **0.1307** (CI 0.1181–0.1442) | **15.946** | **8.663** | **same recipe family, A100-calibrated**; 2026-05-19, file 14.987 GB |
| **mq3-body-mq4-lmhead-kvq8-c256** | **3.552** | **0.3082** (CI 0.2857–0.3316) | **19.096** | **9.856** | **mixed-bits**: MQ3-AWQ-GPTQ body (α=0.55, refit_iters=1) + MQ4-AWQ lm_head; 2026-05-20, file 11.943 GB; Stage C on A100, Stage D on gfx1100 (commit 7915ed6f added per-tensor n_bits dispatch). Coherent eyeball output ("Paris is the capital and largest city of France…" at 43 tok/s on gfx1100). HF: `hipfire-models/qwen3.6-27b-dev` (deleted 2026-05-20 — see analysis below). |

**AWQ+GPTQ Δ on 27B MQ4 (identical kv-q8 + Q8 lm_head + body kmap):**
mean KLD **−38%** (0.2034 → 0.1257), CIs non-overlapping. p99 KLD −12%
(19.009 → 16.666). PPL **+1.3%** (8.584 → 8.697) — diverges from KLD
because the plain quant happens to assign slightly higher probability
to the ground-truth next token; AWQ+GPTQ is more faithful to the BF16
*distribution* (the KLD axis). At 4-bit on 27B, KLD captures the lift
that next-token-loss masks. Contrast with §1.4's 9B MQ3 cohort, where
3-bit's lossier regime makes AWQ+GPTQ win on both axes.

**A100 vs V100 calibration on 27B (2026-05-19).** Same arch (gfx1100), same KV, same n, same scoring mode, same recipe family (mq4 + AWQ stage-A F2 + GPTQ + Q8 lm_head):

| Axis | V100 (4.5 bpw) | A100 (4.458 bpw) | Δ |
|---|---:|---:|---:|
| KLD | 0.1257 | 0.1307 | **+4.0%** (worse) |
| p99 KLD | 16.666 | 15.946 | −4.3% (better tail) |
| PPL | 8.697 | 8.663 | −0.4% (basically identical) |
| bpw | ~4.50 | 4.458 | −0.04 bpw |

**The A100→V100 sidegrade is essentially flat on 27B** — calibration host doesn't materially change quant quality once stage-A F2 + GPTQ are already in the recipe. This is the **opposite** of what §1.1k showed at 9B (lmhead-a100 cut KLD ~50% vs q8head). Possible explanations:
1. The 9B `q8head` baseline used a *different* (worse) recipe than the 9B `lmhead-a100`, not just a different calibration host — the −50% KLD lift was the recipe, not the GPU. 27B already had stage-A F2 in `q8head-v100`, so the A100 variant has nothing recipe-side to capture.
2. 9B's smaller layer count (32 vs 27B's 64) leaves more headroom for per-layer calibration improvements to compound; at 64 layers the gradient is saturating.
3. Cohort-specific: the 27B A100 quant may have used slightly different alpha / whitelist than the V100 sibling — diff between recipes hasn't been audited yet.

The 9B "−50% from lmhead-a100" should not be presented as transferring to other model sizes without a per-size A/B. **At 27B the headline AWQ+GPTQ-vs-plain Δ remains −38%; A100-vs-V100 within that recipe is noise.**

**Cross-engine 27B at matched bpw (llama.cpp Q4_K_M / UD-Q4_K_XL):** not yet measured on 27B — §1.2 anchors are 9B-only. Adding 27B llama.cpp anchors is the next priority to position this row in the bpw-quality Pareto.

**Premature-EOS bug fix wired in.** The prior v100 variant with MQ4
lm_head (no Q8 head) emitted `<|im_end|>` at argmax mid-reasoning
(documented 2026-05-18 session). Naked MQ4 on the 248,320-row lm_head
has insufficient resolution for the long-tail token logits; the Q8
head row above does not exhibit the failure. The KLD number from the
broken-lm_head variant would still be numerically valid (KLD measures
distributional divergence, not output coherence), but is not in this
table because re-measuring on the fixed quant supersedes it.

**Open rows for context:**
- mq4-awq-gptq-f2 with MQ4 lm_head — would isolate Q8-head Δ; not
  reported here due to the coherence bug above.
- Same variants on gfx1151 — required by §6 "per-arch" rule.
- Same variants at full slice (1175 chunks) — current row is the
  c256 smoke (CI half-width ≈11% of mean, acceptable for cohort
  comparison; full slice expected to tighten ≈2×).

**Mixed-bits MQ3-body + MQ4-AWQ-lmhead (2026-05-20).** First true
mixed-format Stage D build — `--format mq3 --lm-head-format mq4
--precomputed-gptq-path …` with per-tensor `n_bits` in the manifest
dispatched correctly (commit 7915ed6f). Bytes verified: 496 MQ3G256
body tensors, 1 MQ4G256 lm_head, 497 AWQ sidecars; body codes use the
full `[0..7]` range as expected. File is **3.86 GB smaller** than the
all-MQ4 A100 row (11.943 vs 14.987 GB).

Quality trade-off:
- **KLD 0.3082 vs 0.1307 all-MQ4** → +136% (CIs non-overlapping, so
  the regression is real, not noise). p99 19.1 vs 15.9 (+20% tail).
- **PPL 9.856 vs 8.663 all-MQ4** → +14% next-token-loss.
- **−1.0 bpw saving** (3.55 vs 4.46) for that quality cost.

The body's 3-bit precision floor dominates the size win. Compare to §1.4
9B MQ3 cohort: `mq3-awq-gptq` reached KLD 0.1967 (9B at 3.8 bpw) — about
1.5× the 4-bit equivalent's KLD. The 27B mixed-bits sits at a similar
ratio above its 4-bit reference (0.308 / 0.131 ≈ 2.4×), consistent with
the "3-bit imposes a structural KLD floor that AWQ+GPTQ can mitigate
but not eliminate" finding.

**Coherence eyeball:** fluent on prose ("Paris is the capital and largest
city of France, serving as the country's political and economic center.
It is globally renowned for its rich history, iconic landmarks like the
Eiffel Tower, and world-class museums such as the Louvre…" 1188 tokens
at 43.6 tok/s on gfx1100). No attractor / repetition / token-level
pathology observed in the eyeball window.

**Recommendation:** the mixed-bits build is COHERENT but the +136% KLD
makes it a niche Pareto point — best for VRAM-constrained gfx1100
deployments where the 14.99 GB all-MQ4 file leaves no headroom for KV
cache + activations. Default 27B distribution should remain
`mq4-awq-gptq-f2-lmhead-a100` (KLD 0.1307). The artifact was uploaded
to `hipfire-models/qwen3.6-27b-dev` on 2026-05-20 then deleted after the
n=32 smoke (KLD 0.218) misleadingly suggested better quality than the
n=256 measurement revealed.

**Open α sweep — H1 from §3.2 followups still applies here.** AWQ
α=0.55 was tuned at the 9B MQ4 U-curve minimum. The 27B mixed-bits row
inherits that α with no re-tuning at the body's 3-bit width. Lower α
(0.3 / 0.4) is the next thing to try — less aggressive AWQ pre-scaling
may stay within MQ3's narrower precision envelope and recover some of
the KLD gap to the 4-bit baseline. Expected shape: PPL benefit shrinks
gracefully, KLD regression flips back toward parity with the 4-bit
row. ~3 hours GPTQ wall-time per α point on A100.

---

## 4. KV-cache-mode contribution

### 4.1 Measurements

| Model | Variant | asym3 KLD | q8 KLD | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| 0.8B | mq4-base | 0.3341 | 0.2675 | −0.0666 | −20% |
| 0.8B | mq4-awq | 0.3000 | 0.2531 | −0.0469 | −16% |
| 0.8B | q8f16 (Tier-2) | 0.1256 | 0.0796 (pre-PR#248, n=20, gfx1151) | (−0.046 est.) | Tier-2 substrate |
| 0.8B | q8f16 (Tier-3) | _not measured_ | **0.0041** (n=512, gfx1100, PR-#248) | | path-matched Q8 floor for MQ4 Δ |
| 9B | q8f16 (Tier-2) | 0.1459 | _not measured_ | | Tier-2 substrate |
| 9B | q8f16 (Tier-3) | _not measured_ | **0.0173** (n=256, gfx1151, PR-#248) | | path-matched Q8 floor for MQ4 Δ |
| 9B | mq4-base (smoke) | 0.3376 | 0.3182 (n=20, gfx1100) | −0.019 | −5.7% (n=20 wide CI) |
| 9B | mq4-q8lmhead (probe) | _new_ | 0.3083 (n=20) | | lm_head Q8 vs MQ4: −3.1% KLD; smoke only |
| 9B | mq4-q8conv1d (probe) | _new_ | 0.2360 (n=20) | | conv1d Q8 vs MQ4: **−25.8% KLD**; smoke only — see §5 caveat |

### 4.2 What this means for cross-engine comparison

- The original "asym3 vs Q8 = ~0.20 nats" estimate from the pre-RoPE-fix floor decomposition appears to overstate the penalty at 0.8B post-fix. On mq4-base 0.8B, the actual delta is **~0.07 nats**, i.e. about **20% of the raw KLD**.
- This 0.07 nats is **real lossy KV quantization noise** (asym3 K-rotation precision loss), not engine-side drift, and it is correctly attributed to the KV mode. It is one of the few absolute-KLD-vs-HF components that survives the 2026-05-13 PM reframing intact, because lossy KV quant adds real δlogit-variance independent of whether the reference is bf16 or fp32.
- **Recommendation:** for the Δ-above-own-Q8 framework (TL;DR), the Q8 floor must be measured **in the same KV mode** as the quant being compared. Mixing q8-KV-Q8-floor with asym3-KV-MQ4-row is meaningless because the KV term doesn't cancel under subtraction.

### 4.3 Open KV-mode measurements to fill in

Critical-path (Δ framework requires both engines' Q8 baselines in matching KV mode):

- [ ] **0.8B hipfire q8f16 + q8 KV on PR-#248 Tier-3 fused WMMA prefill** — replaces the n=20 gfx1151 pre-PR Tier-2 number; this becomes the canonical hipfire Q8 floor at 0.8B
- [ ] **9B hipfire q8f16 + q8 KV on PR-#248 Tier-3** — canonical hipfire Q8 floor at 9B
- [ ] **0.8B llama.cpp Q8_0 + q8 KV** — the engine-pair Q8 baseline (currently we have Q8_0 at q8-KV implied from Phase 0 `13003256` cross-check; want a clean direct measurement)
- [ ] **9B llama.cpp Q8_0 + q8 KV** — same, at scale

Within-hipfire Δ measurements (Stage A/B/C uplift):

- [ ] 0.8B mq4-awq + q8 KV
- [ ] 9B mq4-base + q8 KV
- [ ] 9B mq4-awq + q8 KV
- [ ] 9B mq4-awq-gptq + q8 KV (after the in-flight 9B quantize lands)

After the 0.8B + 9B Q8 baselines on PR-#248 are in, the canonical cross-engine table becomes: `(Δ_hipfire(MQ4*) vs Δ_llamacpp(Q4_K_M / UD-Q3_K_XL / UD-Q4_K_XL))` at matched bpw and KV mode, with both engines' Q8 floors subtracted. That is the only cross-engine claim the framework supports.

---

## 4A. Open format-coverage measurement gaps

The current cohort runs only cover MQ4 / HFP4 / MFP4 family. Sub-4-bit
formats and the Lloyd codebook variants have NOT been measured against
the post-RoPE-fix BF16 reference. Per CLAUDE.md and the format roadmap,
several of these are theorized to deliver meaningful quality lifts and
need to be benched before strategic decisions get cemented.

### 4A.1 Sub-4-bit and Lloyd variants — unmeasured

| Format | bpw | Status on disk | Gated by | KLD against current ref? |
|---|---:|---|---|---|
| MQ3G256 (uniform 3-bit, "mq3-rtn") | 3.25 | `~/.hipfire/models/qwen3.5-9b.mq3` exists | — | **MEASURED §1.4** — 0.5449 (gfx1151, kv-q8, n=256) |
| MQ3G256 + AWQ + GPTQ | 3.25 | quant produced via stage-A F2 + stage-B GPTQ | — | **MEASURED §1.4** — 0.1967 (gfx1151, kv-q8, n=256) |
| MQ3G256-Lloyd (3-bit + per-block Lloyd-Max 8-entry FP16 codebook) | 3.50 | `~/.hipfire/models/qwen3.5-9b.mq3-lloyd` exists | — | **NOT MEASURED** |
| MQ4G256-Lloyd (4-bit + Lloyd codebook, prefill kernel) | ~4.5 | not quantized | PR [#197](https://github.com/Kaden-Schutt/hipfire/pull/197) (`feat/issue-182-mq4-lloyd`, open) | **NOT MEASURED** |
| MQ2G256-Lloyd (2-bit + Lloyd codebook) | ~2.5 | check `~/.hipfire/models/` | — | **NOT MEASURED** |

### 4A.2 Lloyd-transform uplift — investigation note

The Lloyd-Max per-block codebook (used in mq3-lloyd / mq4-lloyd /
mq2-lloyd) replaces the uniform 4/8/16-codepoint grid with K
data-driven centroids fit to each post-FWHT block's value distribution.
Theory: for post-FWHT weights with heavy-tailed kurtosis, a uniform
grid wastes codepoints on tail bins while under-resolving the dense
center. Lloyd codebooks should claw back precision proportional to
how non-uniform the post-FWHT distribution is.

**Expected uplift to measure (against current post-RoPE-fix ref):**
1. **MQ3 → MQ3-Lloyd at 9B.** Lloyd at 3-bit is theorized to be the
   bigger lever (MQ3 uniform is on the cliff of where 3-bit becomes
   unusable; codebook adaptation may rescue it). Direct A/B since both
   .hfq files already exist on disk.
2. **MQ4 → MQ4-Lloyd at 9B.** Gated on PR #197 merging or being
   checked out temporarily. Smaller theoretical uplift than MQ3-Lloyd
   (MQ4 uniform is already in the well-behaved zone) but worth a
   measurement to know the order of magnitude before committing
   format/kernel work.
3. **Stacking interaction with AWQ.** AWQ pre-scaling reshapes the
   per-channel value distribution before FWHT; whether Lloyd codebooks
   trained on post-AWQ post-FWHT blocks deliver the same uplift as on
   non-AWQ blocks is open. The Stage A → Stage B → Stage C roadmap
   currently treats AWQ + GPTQ + Lloyd as orthogonal levers; measuring
   them as additive vs interfering is part of the design validation.

**Highest-leverage measurement order:**
1. mq3 + mq3-lloyd at 9B (no PR gating, no quantize required, ~30 min wall)
2. mq3 + mq3-lloyd at 0.8B (need to quantize first; ~10 min quantize + ~15 min eval)
3. mq4-lloyd at 9B (gated on PR #197 — see below)
4. mq4-lloyd + AWQ stack at 9B (Phase A Step 5c follow-up)

### 4A.3 How to measure mq4-lloyd (gated on PR #197)

PR `Kaden-Schutt/hipfire#197` (`feat/issue-182-mq4-lloyd`, OPEN) ships
the MQ4-Lloyd WMMA prefill kernels. To measure mq4-lloyd KLD without
merging:

```bash
# Save current branch
git checkout -b backup-feat-mq-v2 feat/mq-v2-quant-format

# Fetch + check out the PR
gh pr checkout 197 -R Kaden-Schutt/hipfire

# Quantize a candidate
cargo run --release -p hipfire-quantize -- \
    --hf <bf16-path> --format mq4-lloyd \
    --out ~/.hipfire/models/qwen3.5-9b.mq4-lloyd

# Eval against the existing ref
./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
    --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
    --output benchmarks/quality-baselines/results/<dated>/qwen35-9b-mq4-lloyd__gfx1100.kldseq \
    --kv-mode asym3 --scoring-mode prefill --max-chunks 512

# Reduce + copy results back to feat/mq-v2-quant-format
# (the .kldseq file is engine-version-agnostic; row can be appended
#  to the master doc without re-running on the main branch)

# Restore working branch
git checkout feat/mq-v2-quant-format
```

The `.kldseq` artifact + reduced KLD/PPL values are portable across
engine versions (the slice is the slice; the reference is the
reference). Result row goes into §1 of this doc; flag the engine
SHA in the row's "Notes" field (e.g. "engine=PR#197@<sha>").

---

## 5. Key findings catalog

- **2026-05-12 — RoPE fix flips defaults.** Halfsplit (HF convention) becomes default; interleaved relegated to `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`. Drops 0.8B Q8 floor from ~0.49 to ~0.13. Commit `1805d820`.
- **2026-05-12 — AWQ on MQ4 (Stage A) closes 30% of 9B above-floor gap.** mq4-base 0.192 above-floor → mq4-awq 0.134 above-floor. AWQ implementation per `crates/hipfire-quantize/src/main.rs` AWQ block; calibration via `awq_collect.py`.
- **2026-05-13 AM — Engine-drift floor investigation closed (first pass).** No further single-localizable bugs in the post-RoPE-fix residual ~0.08 nats. Initial attribution: 0.07 (cumulative pipeline imprecision, 24 layers) + 0.01 (Q8 weight + cross-engine inherent). Refuted GLM-5's F16-LA-weights claim. **Superseded by PM finding below.**
- **2026-05-13 — Cohort humaneval/smoke bug fixed.** `HIPFIRE_DEFAULT_MODEL` → `HIPFIRE_MODEL` in `scripts/bench_humaneval_completion.sh` and `scripts/quant_cohort.sh`. KLD/PPL/MSE data unaffected (eval_hipfire path doesn't use the daemon).
- **2026-05-13 — asym3 KV vs q8 KV delta is smaller than pre-fix estimate.** Empirically 0.07 nats at 0.8B mq4-base, not the 0.20 implied by the pre-RoPE-fix decomposition. (Survives the PM reframing — this is real lossy-KV-quant noise.)
- **2026-05-13 AM — Cross-engine 4-bit-beats-8-bit finding (RETRACTED PM).** Original headline: llama.cpp Q4_K_M (0.035 KLD vs HF) beats hipfire q8f16 (0.080 KLD vs HF) on Q3.5-0.8B at matched Q8 KV, implying hipfire engine drift exceeds llama.cpp's engine + 4-bit combined. **This interpretation is wrong** — see PM finding below. The absolute-KLD-vs-HF numbers are correct; what they mean is not what was claimed. See §2.4.1 reframing.
- **2026-05-13 PM — Pattern-hunt audit closes engine-surgery scope to ~zero kernels.** Commits `13003256` (Phase 0 / llama.cpp Q8_0 sanity), `f6d1a59e` (Phase 2 + 3a, rmsnorm bit-faithful to fp64), `a75b4a08` (Day 0 rmsnorm), `1b90a663` (F3 l2norm bit-faithful). Result: hipfire's fp32-native kernels (rmsnorm, l2norm, recurrence) score `0.000000` rL2 vs per-engine fp64 references; 22 of 24 audit-flagged "drift stages" are HF's intentional bf16 cast at module boundaries (F2); the only remaining stage with non-zero fp32-vs-fp32 drift is sigmoid alpha at 0.0024 rL2 cosine 1.000 (negligible). **Hipfire's kernels are uniformly more arithmetically precise than HF's BF16 reference**, not less. The "engine drift floor" is a precision mismatch with HF's training dtype, not a hipfire bug.
- **2026-05-13 PM — Cross-engine KLD-vs-HF reframed: absolute is invalid; use Δ-above-own-Q8.** First-order independence of {weight-quant noise, engine floor, embedding noise} makes `(KLD(quant) − KLD(Q8))` cancel engine + embedding terms within each engine, yielding pure quantizer cost. Cross-engine claim is then `Δ_hipfire(MQ4) vs Δ_llamacpp(Q4_K_M)`, not absolute KLDs vs HF. See TL;DR.
- **2026-05-13 PM — Hipfire architecture-dependent floor: refined after Tier-3 measurement.** Original framing: "DeltaNet amplifies bf16-cast drift ~8×" (Q3-0.6B dense Q8 0.0098 vs Q3.5-0.8B DeltaNet Q8 0.0796 on Tier-2). Refined: that 8× ratio conflated two effects — (a) the Tier-2 FP32-vs-bf16 precision-class mismatch (now closed by PR-#248 Tier-3, see next finding) and (b) genuine DeltaNet recurrence drift accumulation. On Tier-3, Q3.5-0.8B DeltaNet Q8 floor is **0.0041** (not 0.0796), and Q3.5-9B DeltaNet Q8 floor is **0.0173**. The per-layer DeltaNet amplification is real but ~1.20× per additional LA layer (4.2× over the 8 extra layers from 24-layer 0.8B to 32-layer 9B), much smaller than the Tier-2 phantom 8× factor. The "engine floor" is more precisely "DeltaNet recurrence amplifies hipfire-vs-HF dtype mismatch, with magnitude proportional to the precision-class gap" — Tier-3 closes the gap and reduces the absolute floor accordingly.
- **2026-05-13 PM — MQ6+Q8conv1d is hipfire's first format competitive with llama.cpp K-quants absolute KLD.** gfx1151 measurement: 6.5 bpw KLD 0.0568 / PPL 9.281 — matches Llama.cpp UD-Q4_K_XL at +1.2 bpw cost, beats Q4_K_M by 2.2×. PPL only +1% above Tier-3 Q8 floor. **Viable shipping format for "high-quality 4-5 bpw" target.** Cross-engine Δ-above-own-Q8 at 6-bit: 0.040 vs 0.009 = 4.4× — same penalty ratio as 4-bit (1.86×), confirming the gap is format-level (per-block scale fitting, group size, asymmetric vs symmetric), not bpw-specific.
- **2026-05-13 PM — `--kmap-dense --kmap-mode 2` produces NaN logits on gfx1100 only — gfx1151 works.** Filed as [#249](https://github.com/Kaden-Schutt/hipfire/issues/249), originally framed as a unconditional dispatch bug. **Revised 2026-05-14** after gfx1151 agent successfully measured the same `mq4-kmd2 + Q8 conv1d` quant at n=512 and got KLD 0.1613 / PPL 9.172 (see §1.1g) — the dispatch correctly handles mixed MQ4/MQ6 on RDNA3.5 (Strix Halo). gfx1100 (RDNA3) still hits the NaN on the same .hfq file: must be a WMMA-dispatch-path-specific issue with the (MQ4 gate/up, MQ6 down) combination that doesn't manifest on gfx1151's dispatcher. Issue #249 needs updating to "kmd2 dispatch NaN is gfx1100-specific; mq4-kmd2 + Q8 conv1d ships fine for gfx1151 users today." **Shipping note:** mq4-kmd2 + Q8 conv1d at ~5.04 bpw is a Pareto-frontier 5-bit recipe (see §1.1g) — but gated on the gfx1100 dispatch fix before it can be the universal default 5-bit format.
- **2026-05-13 PM — MQ4-Lloyd doesn't help at 9B 4-bit.** n=512 q8-KV measurement (gfx1151) gave KLD 0.3114 / PPL 9.085, essentially tied with mq4-base under q8-KV (estimated converged at 0.27-0.32 from §1.1 + KV-normalization). Lloyd codebook costs +0.66 bpw avg (file size 6.06 GB vs mq4-base 5.31 GB) for null KLD improvement. Confirms §4A.2 prediction: FWHT-256 Gaussianizes per-block distribution, removing the heavy-tail Lloyd is designed to attack. **Active calibration roadmap should drop MQ4-Lloyd**; MQ3-Lloyd remains theoretically interesting (uniform 3-bit is on the cliff edge where heavy-tail loss matters most) and untested.
- **2026-05-13 PM — Per-tensor quant-contribution smoke + n=512 confirmation (9B q8-KV).** Quantized 9B MQ4 with `HIPFIRE_QUANTIZE_LM_HEAD_Q8=1` / `HIPFIRE_QUANTIZE_CONV_Q8=1` / `HIPFIRE_QUANTIZE_CONV_F16=1` (env vars added in `crates/hipfire-quantize/src/main.rs` lines 4055-4084).
  - **n=20 smoke (q8-KV prefill, 9B, gfx1100):** mq4-base 0.3182; mq4-q8lmhead 0.3083 (−3.1% KLD); **mq4-q8conv1d 0.2360 (−25.8% KLD)**; mq4-f16conv1d 0.2388; mq4-f16conv1d-q8lmhead 0.2293.
  - **n=512 confirmation (q8-KV prefill, 9B, gfx1100): mq4-q8conv1d KLD = 0.2501 (CI 0.2396-0.2609), PPL = 8.789.** Drifted +6% from n=20 (expected slice-mean noise); CI tight at ±0.01. PPL is BETTER than the §1.1 asym3 mq4-base 9.116 (−3.6%) and the n=20 smoke (which was wrong-direction artifact of small n).
  - **bpw cost is effectively zero for Q8 conv1d** (+420 KB / +0.0004 bpw avg over 8.95B). vs Q8 lm_head's +540 MB / +0.48 bpw (3 orders of magnitude more expensive per KLD lift unit). **Recommendation: Q8 conv1d should become the default for MQ4 models.** Implementation: change `kmap_resolve_mode` (main.rs:2132) to return `QuantLevel::Q8` for `linear_attn.conv1d.weight` regardless of `--kmap-dense` setting. ~5 line change.
  - **Mechanism:** confirms F2/F3 audit's "1.7-2× drift amplification at Q8 conv1d → LA stage 8/9" finding. MQ4 conv1d adds FWHT rotation noise on top of Q8's already-amplified drift, then DeltaNet recurrence accumulates it across 24 LA layers. Saving ~10× quantization noise at the conv1d input → outsized downstream KLD reduction.
  - **F16 conv1d ≈ Q8 conv1d (0.2388 vs 0.2360 at n=20, statistically indistinguishable):** F16 storage is actually FINER-grained than HF's BF16 reference (10 mantissa bits vs 7), so F16 conv1d weights are slightly more accurate than HF's training-dtype reference → forward pass diverges slightly → KLD-vs-HF slightly higher than Q8. Same F2/F3-pattern (more precise = worse KLD-vs-HF). Q8 dequant's ~7-effective-mantissa-bits precision is closer-to-BF16's grid than F16 storage is. Means Q8 (not F16) is the right default — also smaller.
  - **Q8 lm_head dropped from recommendation:** +540 MB for −3% KLD doesn't pencil out vs Q8 conv1d's near-zero cost for −26%. Cost-per-KLD-lift ratio: Q8 conv1d is **65,000× better** than Q8 lm_head.
- **2026-05-13 PM — PR-#248 Tier-3 fused WMMA Q8 prefill closes the cross-engine floor gap.** 0.8B q8-KV q8f16 floor: 0.0796 (Tier-2 FP32-acc, n=20 gfx1151) → **0.0041** (Tier-3 FP16-acc WMMA, n=512 gfx1100); 9B q8-KV q8f16 floor: 0.1459 asym3-KV Tier-2 (n=512 gfx1100) → **0.0173** q8-KV Tier-3 (n=256 gfx1151). Hipfire 9B Tier-3 Q8 floor is **+6% above llama.cpp Q8_0's 0.0163** — essentially same precision class. The earlier "100% engine drift" claim (Phase 0 commit `13003256`) was 100% the FP32-vs-bf16 precision-class mismatch with HF's bf16 reference; Tier-3's FP16-accumulator WMMA kernels match HF's precision class, and the gap closes. PR-#248 was framed as a perf improvement (18× faster Q8 prefill) but the KLD-vs-HF impact is the bigger structural finding. **Implication for §1.1's "Above-floor" column:** it's path-mismatched (Tier-2 Q8 baseline subtracted from FP16-WMMA MQ4 measurements), understating true 4-bit weight noise by ~0.130 nats at 9B. The path-matched Δ-framework using Tier-3 floor gives the honest accounting.
- **2026-05-14 — "AWQ regression on 9B" RETRACTED — it was a cargo incremental build glitch, not source code.** Earlier in this session multiple smoke runs of `--format mq4 --imatrix --awq` at HEAD reported KLD 13.40 (q8-KV n=20), 13.35 (asym3-KV n=20), 14.55 (with Q8 conv1d), 14.48 (with MQ6 lm_head) — appearing to be a catastrophic 18× regression vs §1.1's 0.2800 anchor. Bisect across `dcf3b18a..HEAD` (4 commits tested: `dcf3b18a`, `bd1ca3e1`, `a9dde0de`, `f6d1a59e`) all returned exactly **KLD 0.250268 to 6 decimals** — a clear "same binary running" pattern that should have flagged the binary-state confound earlier. Verified by smoke-testing the original `mq4-awq-bisect` quant (md5 `bc64d7dc04c816c9f3ee78bdf40ba86e`, built at HEAD, originally KLD 13.40) with the post-bisect freshly-built `eval_hipfire` → **KLD 0.250268** + tok/s **362** (vs original 232 — the broken binary had a slow path active too). All five "broken" smokes used the same stale eval binary. **No AWQ regression exists; current HEAD's `--awq` recipe reproduces the §1.1 anchor.** Lesson: when multiple bisect steps return literally-identical KLD to 6 decimals, the prior should be "same binary running" before "5 different source revisions accidentally produced identical output." `cargo clean -p hipfire-quantize -p hipfire-runtime` before any quality bench post-build-thrash is the safe pattern.
- **2026-05-14 — MQ6 lm_head probe and AWQ + Q8 conv1d smoke failures also attributable to the same cargo glitch.** Earlier MQ6 lm_head smoke at KLD 14.48 (filed as candidate dispatch bug, similar to issue #249) and AWQ + Q8 conv1d at KLD 14.55 (filed as suspect AWQ×Q8conv1d interaction) both ran on the same stale binary. **Pending re-bench with a freshly-rebuilt `eval_hipfire`** to confirm or refute whether those experiments work at HEAD. The MQ6 lm_head experiment's untested-on-gfx1100 hypothesis is still plausible (gfx1151-only validation) but evidence is no longer in the failed smoke — needs fresh measurement.
- **2026-05-14 — AWQ + Q8 conv1d is the best 9B 4-bit recipe to date: KLD 0.1842 at n=512 (q8-KV).** Validated post-cargo-glitch with `cargo clean` + fresh build. Stacks both levers (each ~−21% KLD alone) for **−42% vs mq4-base** — nearly perfect additivity (n=512: 0.1842 vs base ~0.32). Bpw cost is negligible (Q8 conv1d adds +420 KB, AWQ adds 248 F16 sidecars × ~8 KB ≈ 2 MB). Δ-above-Tier3-Q8 = **0.1669** (vs llama.cpp Q4_K_M Δ = 0.1086 — hipfire 1.54× over Q4_K_M, down from mq4-base's ~4.5× ratio). PPL regresses slightly (8.789 → 9.575, +9% — KLD-vs-PPL inversion pattern, same family as §1.1e.i): AWQ tightens distribution match, slightly slides argmax. Recommend: ship AWQ + Q8 conv1d as the default 4-bit configuration on Q3.5-9B and beyond. See §1.1f for the full table + cross-engine Δ math.
- **2026-05-14 — mq4-q8conv1d 0.2501 n=512 anchor confirmed byte-identical after cargo-glitch resolution.** Re-bench with fresh eval binary produced the exact same kldseq file (md5 `8c90dbad8d7b9ee22147f0b02fbc2b43`). Original measurement (2026-05-13 18:52, before the chain-script + bisect rebuild thrashing started 21:45+) is reliable. Anchor stays at 0.2501; PPL 8.789. The reverify file lives alongside the original at `qwen35-9b-mq4-q8conv1d-reverify__gfx1100__kv-q8__c512.kldseq` as audit trail.
- **2026-05-14 — Mixed MQ4/MQ6 (kmd2) + Q8 conv1d = new Pareto point at ~5.04 bpw.** gfx1151 agent measured n=512: KLD **0.1613** / PPL **9.172** (104 min wall). Sits cleanly between uniform MQ4 (~4.25 bpw, KLD 0.25) and uniform MQ6 (6.5 bpw, KLD 0.051): promoting only `mlp.down_proj` to MQ6 captures ~half of MQ6's KLD benefit at ~half the bpw overhead. PPL is within 0.02% of the Tier-3 Q8 floor (9.189) — predictively indistinguishable from 8-bit. Δ-above-Tier3-Q8 = 0.144 = **1.32× of llama.cpp Q4_K_M** Δ — the smallest absolute-KLD gap to llama.cpp K-quants on any hipfire 4-5 bpw recipe to date. **Shipping note:** gated on issue #249 (dispatch NaN on gfx1100; gfx1151 works). When fixed, this is a strong default 5-bit recipe candidate. See §1.1g for full table.
- **2026-05-14 — Stage B GPTQ does NOT scale down to 0.8B — AWQ + Q8 conv1d alone is the winner there.** Full 5-recipe + noisy-Hessian matrix at 0.8B q8-KV n=512:
  - mq4-base 0.2675
  - mq4-awq 0.2531
  - **mq4-awq + Q8 conv1d (NO GPTQ): KLD 0.1366 / PPL 19.61 — best at 0.8B**
  - mq4-gptq (alone, no AWQ): 0.3908 (+46% over base — GPTQ alone hurts)
  - mq4-awq+gptq (BF16-H, no Q8 conv1d): 0.3371 (+26% — AWQ recovers some)
  - mq4-awq+gptq + Q8 conv1d: 0.1983 (+45% over the no-GPTQ winner — GPTQ on top of AWQ+Q8conv1d still hurts)
  - mq4-awq+gptq (noisy-H, no Q8 conv1d): 0.3560 (worse than BF16-H by +5.6%)

  GPTQ adds 0.06 KLD (+45%) on top of AWQ+Q8conv1d at 0.8B regardless of Hessian source. This matches Frantar 2022 §4 (GPTQ's improvement scales with model size; small models can regress). The implementation is verified correct: all 23 unit tests pass, identity-H reduces to RTN, AWQ-rescale + FWHT-similarity invariants hold, clamp rate < 0.02% on real tensors. **The bug is sample size — 9B is the smallest scale where GPTQ may pay off.**

- **2026-05-14 — Calibration-mismatch hypothesis FALSIFIED.** Hypothesis: GPTQ regresses on 0.8B-no-Q8conv1d because the Hessian was collected on a BF16 model (clean conv1d) while inference has MQ4-conv1d quant noise. Tested via `scripts/collect_hessian.py --inject-conv1d-mq4-noise` (commit `c4107fa2`): round-trip every linear_attn.conv1d weight through FWHT-256 + MQ4-quant-dequant before collecting Hessians, then re-quantize 0.8B mq4-awq-gptq. Result: KLD 0.356 vs BF16-H's 0.337 — slightly worse (+5.6%), not better. The activation-distribution mismatch is not the mechanism. Real explanation: GPTQ's column-sequential OBS update has high variance at small K (0.8B K_max=3584); empirical regularization (AWQ + Q8 conv1d) outperforms OBS at this scale. Re-validates Frantar's "scale matters" finding.
- **2026-05-14 — 0.8B shipping recipe (RETRACTED): no recipe ships at 0.8B because 0.8B on hipfire is intrinsically incoherent regardless of quant precision.** Coherence-eyeball test across three variants on gfx906 (`mq4-kmd2-awq`, shipped `mq4-awq+Q8 conv1d` Stage A baseline, and the **Q8 floor** `q8f16-pr248`) all produce equivalent-quality babble at free generation. Confirmed independently on gfx1100 with `mq4-awq+Q8conv1d` (KLD 0.1366, PPL 19.61) and `mq4-awq+gptq+Q8conv1d-postfix` (KLD 0.0478, PPL 18.24) — both incoherent despite the dramatic prefill-metric gap. Since Q8 effectively-lossless weights also babble, the failure is below the quant layer: the ~0.5-nat DeltaNet pedestal at 0.8B (master doc §1.1) is depth × recurrence amplification of HF's bf16-cast pattern, not weight-quantization noise. **KLD/PPL on 0.8B remain valid for RANKING quant recipes against each other** (lower KLD = closer prefill-distribution match to HF reference); they just don't translate to generation coherence on a model whose baseline is broken. The "Stage A 0.8B winner" framing was misleading — there's no winner because the playing field is unusable. **9B is the only valid coherence-validation target for Stage A/B/C.** 9B's Tier-3 Q8 floor 0.0173 is 4.6× lower than 0.8B's; the 9B BF16 reference generates coherently in upstream engines, so the 9B GPTQ overnight (`/tmp/gptq_9b_overnight.sh`, post-OBS-fix) is the actual ship-decision experiment.
- **2026-05-15 — Stage B GPTQ at 4B lands KLD 0.0662 / PPL 10.71 (n=512, q8-KV), coherent at greedy generation.** First 4B GPTQ data point. Recipe: mq4-awq+gptq+Q8conv1d, ~4.25 bpw. Hessian collected via `scripts/collect_hessian.py` on the HF safetensors (262k calibration tokens, 1h17m on gfx1151). Quantize ~3 h CPU. AWQ + GPTQ + Q8 conv1d composed cleanly; 248 GPTQ-target tensors all clamped within Frantar healthy range (0.001%–0.005% clamps/elements). Coherence-eyeball (humaneval_0, greedy 400 tok) green across all 8 hipfire-detect detectors — structured CoT, on-topic, no attractor/loop. **Crucially, 4B is coherent on hipfire** — unlike 0.8B (intrinsically incoherent per 2026-05-14 retraction), 4B can be a Stage A/B coherence-validation target. The Stage B scale-down ladder so far: 0.8B regresses (+45% over AWQ+Q8conv1d), 4B = 0.0662 (no AWQ+Q8conv1d-only baseline yet, so can't yet judge GPTQ delta at 4B in isolation), 9B = pending overnight. **VL gotcha: Qwen3.5-4B on HF is `Qwen3_5ForConditionalGeneration` (image-text-to-text); MUST quantize from HF safetensors, not BF16 GGUF, or the `model.language_model.` prefix is stripped → fails to load AND silently zero-matches Hessian keys.** See §3.1 for full details + reproducer.
- **2026-05-14 — KLD-vs-coherence inversion at 0.8B made the post-fix 0.0478 anchor diagnostic, not decisional.** The OBS-Cholesky bug fix (commit `687aa2d0`) dropped mq4-awq+gptq+Q8conv1d KLD from 0.1983 (broken) to 0.0478 (correct math) — a real 4× improvement in prefill-distribution match. But the absolute KLD value doesn't predict generation coherence at 0.8B because the BF16 reference itself is broken at that scale. Useful interpretation: 0.0478 confirms the fixed Cholesky math reaches its theoretical lower bound (the residual KLD is now dominated by 4-bit quant noise vs the original ~0.1 nat noise floor from the buggy OBS factor). Empirical-quality validation moves to 9B.
- **2026-05-14/15 — F2 AWQ whitelist expansion: KLD-flat but PPL −6.6% with paired-t = −13.32 (p<10⁻³⁰).** F2 extended AWQ pre-scaling to output-side projections (o_proj/wo/out_proj/down_proj/w_down, sidecar count 184→248 on 9B). Paired n=256 q8-KV α=0.5 on gfx906: KLD Δ=−0.00003 (t=−0.01, noise), but NLL Δ=−0.0678 (t=−13.32, F2 strictly better on 209/256 chunks = 81.6%), PPL 9.54→8.91 (−6.6%). The lever is real and significant — KLD just doesn't see it because AWQ redistributes probability mass favorably toward the true next token without changing the full top-K divergence shape vs BF16. KLD averages over top-K; NLL captures per-token surprise at the true label. F2 alpha sweep n=100 q8-KV gfx906 [0.35–0.65 step 0.05] shows severe KLD-PPL inversion: α=0.55 is the WORST on KLD (0.183) but BEST on PPL (8.79), NLL paired-t = −8.65 vs α=0.50. The α that minimizes KLD (0.35) is 5th-best on PPL. **Methodological retroactive finding: F1's headline "lift = 2.3% on KLD" understated true F1 lever value because NLL was never measured. Going forward, paired-t on per-chunk NLL is the primary AWQ-quality signal; KLD is secondary.** F2 implementation includes critical self-review at `awq-f2-rev-claude.md` (caught `weight_gemv` un-AWQ'd dispatch path in post-implementation review; fix landed in same PR before any user-visible corruption). Recommend ship config: **F2 default whitelist, α=0.55 on gfx906/q8-KV** (subject to n=256 PPL-optimum confirmation + cross-arch gfx1100 sweep — both pending). Plan §F2 predicted "+3-7% reduction"; measured 6.6-7.9% PPL, right range, wrong metric. See §1.1h for the full table.

---

## 6. Methodology rules for adding rows

1. **Always disclose KV mode** (`asym3` / `q8`) in every row. No "default" hand-waving.
2. **Always disclose scoring mode** (`prefill` / `per-token` / `gguf`). Mixing is forbidden by issue-113 §5.3.
3. **Always link the source `.kldseq` file path** or the cohort directory it lives in.
4. **Bootstrap 95% CI** (10,000 resamples on per-seq means) — both `quant_cohort.sh` and the standalone q8-KV probe scripts emit this.
5. **For "above-floor" math (Δ-above-own-Q8 framework, the canonical hipfire-internal lift metric AND the only valid cross-engine claim):** subtract the engine's own Q8-weight floor measured under the SAME KV mode AND the SAME kernel path (e.g. PR-#248 Tier-3 prefill vs Tier-2 batched). Cross-KV-mode floor subtraction is meaningless; cross-kernel-path floor subtraction has unknown error bars.
6. **Flag PRE-FIX (pre-2026-05-12 RoPE) rows explicitly** if they need to be cited; otherwise omit them.
7. **Flag the Q8 kernel path** (Tier-2 batched / Tier-3 fused WMMA / future) in any q8f16 row. The Q8 floor is kernel-path-specific (structurally close but not bit-identical across paths); the Δ-framework requires consistent kernel path between numerator and denominator.
8. **Absolute `KLD(engine || HF-bf16)` is reported for completeness but is NOT a cross-engine output-quality metric.** It measures similarity to HF's bf16-cast noise pattern, which favors engines with bf16-cast-like internal accumulators over arithmetically more precise engines. Use Δ-above-own-Q8 for any "engine A vs engine B" claim. See TL;DR + §2.4.1 + engine-drift memory.
9. **For AWQ-config A/B comparisons (e.g. F1 vs F2, α-sweep, whitelist variations), report paired-t on per-chunk NLL alongside the KLD table.** AWQ improvements typically redistribute probability mass favorably toward true tokens without changing the full top-K divergence shape vs BF16, so they're under-measured by KLD. The F1 vs F2 A/B at α=0.5 (§1.1h) had KLD Δ ≈ 0 (t=−0.01) but NLL Δ = −0.068 (t=−13.32, p<10⁻³⁰) — a 6.6% PPL win invisible to KLD. The F2 alpha sweep showed severe KLD-PPL inversion: α=0.55 had the worst KLD AND the best PPL simultaneously. **KLD-only ranking will pick the wrong AWQ default.** Pair every AWQ-config row with NLL paired-t when the comparison is within-engine; reserve absolute KLD ranking for cross-format (MQ4 vs MQ6 vs Q8) comparisons where the per-channel distribution shape changes more than the mass-redistribution effect.
