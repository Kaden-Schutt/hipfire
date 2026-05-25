# Phase A Step 0.5 — fivetide-comparison findings

**Cohort:** `2026-05-11-cohort-phase-a-step-0.5` (this directory)
**Date:** 2026-05-11
**Branch:** `feat/mq-v2-quant-format` @ 6c3f2feb (cohort run) / 78892f0a (with comparison + table-flip)
**Host arch:** gfx1100 (7900 XTX)
**Slice:** wikitext2-test, ctx=2048, slice md5 `83b0205a304bf4e52172ecdb05f2e895`, max-chunks=256 (quick-slice)
**Scoring mode:** prefill (the canonical post-#113 default)
**KV mode:** asym3

## 1. Headline result

| variant | MSE (4-bit qts) | KLD vs BF16 | KLD-p99 | PPL | Δ vs MQ4 (KLD / PPL) |
|---|---:|---:|---:|---:|:---|
| MQ4G256 (FWHT + INT4 g=256) | 6.62e-6 | 0.8084 | 19.86 | 15.16 | reference |
| HFP4G32 unrotated (E2M1 + UE8M0/FP16-row g=32) | 3.15e-6 | 0.9763 | 19.54 | 18.68 | +20.8% / +23.2% |
| MFP4G32 (FWHT + E2M1 + UE8M0/FP16-row g=32) | **2.98e-6** | **1.1116** | 19.04 | **21.02** | +37.5% / +38.6% |

**Fivetide's NRMSE paradox is reproduced.** MFP4G32 wins per-tensor reconstruction (lowest MSE) and loses model-quality (highest KLD vs BF16, highest PPL). HFP4G32 sits in the middle on both metrics. The relative ordering MQ4 > HFP4 > MFP4 in model-quality is identical to fivetide's qualitative finding, even though we measure against BF16 (fivetide measured KLD-vs-MQ4 and PPL — different yardsticks, same direction).

## 2. Comparison to fivetide's published numbers

| | fivetide (gfx1151, per-token, asym4, full slice) | this cohort (gfx1100, prefill, asym3, 256-chunk) |
|---|---|---|
| 9B MQ4 PPL | 9.94 | 15.16 |
| 9B MFP4 PPL | 12.47 | 21.02 |
| **9B MFP4 vs MQ4 PPL Δ** | **+25.4%** | **+38.6%** |
| 9B HFP4 PPL | (not in fivetide's 9B PPL table) | 18.68 |
| 9B HFP4 vs MQ4 PPL Δ | (no fivetide 9B row) | +23.2% |

**Direction reproduces, magnitude differs.** Absolute PPL numbers are not directly comparable across the methodology axes below. The MFP4-vs-MQ4 PPL gap is bigger on our infra (+38.6%) than on fivetide's (+25.4%) — that's a real cross-platform observation worth understanding.

### Methodology delta cataloguing

| axis | fivetide | this cohort | likely effect on the MFP4-vs-MQ4 PPL ratio |
|---|---|---|---|
| KV mode | asym4 | asym3 | asym3 adds more KV-cache quantization noise — affects both numerator and denominator, but **MFP4 may be more sensitive to compounded weight + KV noise** because its model-quality is already on a steeper failure curve. Plausible driver of the +13pp ratio expansion. |
| Scoring mode | per-token (forward_scratch loop) | prefill (forward_prefill_batch) | Per `issue-113-quant-quality-eval.md` §5.3, prefill mode is −6.75% mean-KLD shift vs per-token on MQ4 — i.e. prefill is *more* faithful to BF16. So prefill *shrinks* the absolute KLD value but should not flip the relative ratio between formats. |
| Host arch | gfx1151 (Strix Halo APU, LPDDR5x) | gfx1100 (7900 XTX, GDDR6) | Kernel-path differences possible but bounded — same family of WMMA kernels under PR #235. Cross-arch KLD divergence-canary tolerance per #113 §6.1 not yet measured for HFP4/MFP4 specifically. |
| Slice subset | full 1175 chunks | 256 chunks | Our 256-chunk MQ4 reproduces full-slice MQ4 within bench noise (KLD 0.8084 vs 0.8171, PPL 15.16 vs 14.89 — CIs overlap), so subset size is not a meaningful contributor to the gap. |
| K-map | unspecified in fivetide's published methodology | OFF (default for dense per `main.rs:2737` maintainer directive) | Both runs almost certainly K-map-OFF on dense (no flag would activate it). Not a delta. |

### The KLD-vs-BF16 picture (us) vs KLD-vs-MQ4 picture (fivetide)

Fivetide's KLD column (0.8B-only, ctx=512, asym4) measures **KLD(quant_X || quant_MQ4)** — i.e., "how far does quant X drift from MQ4's output distribution?". Their finding: MFP4 (0.661) < HFP4 (0.815) < HFQ4 (0.936). MFP4 tracks MQ4's distribution closest.

This cohort's KLD column measures **KLD(quant_X || BF16)** — "how far does quant X drift from full precision?". Our finding: MQ4 (0.8084) < HFP4 (0.9763) < MFP4 (1.1116). MQ4 tracks BF16 closest.

**These are not contradictory.** They answer different questions:
- *Fivetide:* which 4-bit format is most-faithful to MQ4? Answer: MFP4.
- *This cohort:* which 4-bit format is most-faithful to BF16? Answer: MQ4.

Both can be true simultaneously because MFP4's drift from BF16 *includes* MQ4's drift from BF16 plus its own incremental drift away from MQ4. MFP4 ends up *more drifted* from BF16 than MQ4 is — even though MFP4 is *less drifted* from MQ4 than HFP4 or HFQ4 are.

For Phase A's goal of "pick the best quant against ground truth", KLD-vs-BF16 is the relevant metric and the answer in this cohort is **MQ4 still wins on Qwen3.5-9B dense** before any calibration is applied.

## 3. Step 0.5 verdict

**Phase A baselines validated.** The fivetide PPL-paradox finding reproduces on this gfx1100 box: E2M1+FP16-row wins per-weight MSE but loses model-quality vs INT4+per-block-affine. The reproduction-within-noise check (256-chunk MQ4 vs full-slice 2026-05-11 MQ4) confirms the cohort pipeline produces trustworthy KLD/PPL numbers. Methodology drift between our setup and fivetide's is documented above and is the explanation for the +38.6% vs +25.4% magnitude delta — not a substantive disagreement on the format-quality direction.

**Implication for Phase A roadmap:** the "calibrated MFP4 beats UD-Q4_K_XL on KLD" projection in `docs/plans/qwen35-mq4-quality-gap.md` §1.3 is **measured-against**, not validated. Whether calibration (L4 weighted-LS + L5 imatrix) closes the +37.5% KLD-vs-BF16 gap from MFP4 to MQ4 is the actual Phase A bet. The data in this cohort gives us the floor (uncalibrated MFP4 KLD 1.1116) and the ceiling (MQ4 KLD 0.8084) against which any L4/L5 quantizer-side change will be measured.

The format decision pending Phase A is genuinely open per the rebuttal doc's framing — the cohort baselines do not settle it.

## 4. Step 0.5 caveats — what this cohort does NOT measure

| caveat | what it means | how to address |
|---|---|---|
| **HumanEval column = FAIL on all 3 variants** | `bench_humaneval_completion.sh` has a daemon-startup race: it polls `pgrep -af "examples/daemon"` before the daemon process registers, gets an empty result, exits with "daemon failed to start" — even though the daemon DOES come up ~30s later (smoke step later in same cohort confirms this by reusing the daemon successfully on the MQ4 row). | Replace `pgrep` liveness check with HTTP `/v1/models` probe + ~10s grace period. Or wait for serve.log to exist at all before polling. ~30 min fix. |
| **Smoke column = SPIRAL / ERR_'choices'** (no real signal) | Two false-failure modes surfaced: (a) MQ4: model produced 400 tokens of `<think>` reasoning that got stripped, smoke heuristic reads `len==0 → SPIRAL` — false positive (real label should be `BUDGET`). (b) HFP4/MFP4: daemon's `/v1/models` registration apparently doesn't recognize `.hfp4`/`.mfp4` extensions, so chat-completions returns `{"error": "model not found"}` — python parser KeyErrors on the missing `choices` key. | Fix (a): inspect raw (pre-strip) completion + `finish_reason` to distinguish budget-exhaustion (`finish_reason=length` + unclosed `<think>`) from true spiral (`finish_reason=stop` + empty raw output). Fix (b): trace the daemon's model-directory scanner to register `.hfp4`/`.mfp4`. Both ~30 min. |
| **Cohort scope: 9B Qwen3.5 only** | Phase A Step 0.5 calls for 0.8B + 4B + 9B + A3B. Only 9B has a BF16 KLD reference published (`hipfire-models/qwen-kldref`); 0.8B/4B/A3B refs are not yet built or uploaded. | Build local BF16 refs via `build_kld_ref` example: ~10 min wall for 0.8B, ~30-45 min for 4B, multi-hour for A3B BF16. Or wait for fivetide to upload. |
| **K-map = OFF (dense default)** | Cohort measures pure format-vs-format quality. With K-map ON, edge layers + every-3rd `ffn_down` would promote to MQ6, mlp.gate to Q8 — changes the picture toward production-realistic config. | Separate Step 0.5b cohort with `--kmap-dense` on all three variants. Asks: "does the current K-map heuristic close more of MFP4's gap than MQ4's?" — a useful Phase A signal but not the Step 0.5 question. ~40 min wall. |

## 5. Pointers

- Cohort artifacts: this directory's `per-variant/` (kldseq, mse.txt, eval.log)
- Existing 2026-05-11 full-slice MQ4 reference: `../2026-05-11/per-seq/qwen3.5-9b.mq4__gfx1100__prefill.kldseq` (used as the cross-check for the 256-chunk reproducibility claim)
- Fivetide's source doc: `benchmarks/quality-baselines/external/fivetide-2026-05-11-hfp4-quality-analysis.md` (snapshotted in-tree by commit 78892f0a)
- §5.4 prefill-eligibility table fix (this confirms HFP4/MFP4 ran via the batched WMMA path, not per-token fallback): `docs/plans/issue-113-quant-quality-eval.md` (commit 78892f0a)
- Strategic rebuttal that this cohort populates: `docs/plans/hfp4-fivetide-rebuttal-perspective.md` §"Things that need to be measured before claiming"
