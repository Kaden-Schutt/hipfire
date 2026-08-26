# Qwen3.8 27B MQ4G256 — lever exhaustion, plus two gfx942 blockers

**Lifecycle:** `historical`. Fixture-bound measured evidence from the same
MI300X BF16-teacher campaign as
[`2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md`](2026-08-16-qwen38-27b-mq4-awq-alpha-kld.md).
It is **not** a current default, an automatic baseline, a product claim, or an
admission decision.

**Relationship to the alpha record:** companion, not correction. That record
stands unmodified; this one adds the AG-side curve it did not yet have, plus
two negative results and two hardware blockers found afterward.

**Disposition:** At fixed MQ4G256, the accessible recipe levers are exhausted at
`alpha≈0.05-0.15`. Increasing calibration tokens by 50 % changes KLD by less
than 1 % — the Hessian is already well-conditioned at N/K≈15. Separately, the
dense Lloyd codebook family (Tier-3/4 work) **cannot execute on gfx942 at all**,
and the gemma4/glimmer KLD path produces an implausible oracle, so neither is a
quality result pending fixes.

---

## 1. Calibration token count — no effect (negative result)

Same parent, same `alpha=0.05`, same references, same binaries. Only the
calibration token budget differs. At `down_proj` K=17408 this moves N/K from
15.1 to 22.6.

| calibration tokens | N/K @ down_proj | WT2 KLD | WT2 PPL | AG KLD | AG PPL |
|---|---|---|---|---|---|
| 262,144 | 15.1 | 0.063102 | 6.5260 | 0.224205 | 7.4737 |
| 393,216 | 22.6 | 0.063133 | 6.5242 | 0.226272 | 7.4405 |

Deltas are +0.05 % (WT2) and +0.9 % (AG) — both within the flat band already
observed across `alpha in [0.05, 0.25]`, i.e. indistinguishable from no change.

**Reading:** the second-moment statistics AWQ consumes (`in_sum2` -> `RMS_act`)
converge well before 262 k tokens. Spending GPU hours on a larger calibration
corpus is not a lever at this format. This does **not** transfer to LDLQ/GPTQ,
which consumes the full K×K Hessian rather than its diagonal and is far more
sensitive to conditioning; that arm is unmeasured here.

Artifacts: `qwen3.8-27b.calib393k.hfq` (496/496 tensors, uniform 393,216
tokens, 31,481,139,200 B); arm md5 `3cd3fada19df`.

## 2. AWQ alpha — AG-side curve

The alpha record selected on WT2 (out-of-distribution) and reported AG for three
points. The remaining AG points, same fixture:

| alpha | WT2 KLD | AG KLD |
|---|---|---|
| 0.05 | **0.063102** | 0.224205 |
| 0.10 | 0.064720 | 0.239629 |
| 0.15 | 0.064898 | **0.216983** |
| 0.20 | 0.064102 | 0.272399 |
| 0.25 | 0.065860 | 0.247375 |
| 0.30 | 0.068296 | 0.269824 |
| 0.55 (default) | 0.083329 | 0.263730 |

The two references disagree on the interior minimum — WT2 says 0.05, AG says
0.15 — while agreeing decisively that the shipped 0.55 is worse than both. That
disagreement is the expected signature of a **flat** optimum: within
`[0.05, 0.25]` the ranking is reference-dependent noise, and only the gap to
0.55 is real. The selection stays with WT2's 0.05 because AG is the held-out
tail of the calibration corpus and must not be used to choose a recipe.

## 3. Blocker — dense Lloyd is not implemented on gfx942

Tier-3 (MQ4-GL / MQ4-Lloyd) and Tier-4 (MQ3-Lloyd, MQ2-Lloyd) work cannot be
measured on this host. Quantization succeeds:

| format | bytes | md5 |
|---|---|---|
| `mq4-lloyd` | 17,382,419,456 | `dfa4fd8dbe75` |
| `mq2-lloyd` | 8,581,696,512 | `e049b42bf43f` |

Execution does not. Both scoring modes fail identically:

```
--scoring-mode prefill   -> no implementation for FusedQkvzaMq4G256Lloyd
--scoring-mode per-token -> no implementation for FusedQkvzaMq4G256Lloyd
```

Cause is kernel coverage, not a defect: every Lloyd kernel carrying a `.gfx942`
suffix in `kernels/src/` is **MoE-specific** —
`gemm_mq2g256_lloyd_moe_grouped_mfma.gfx942.hip`,
`gemv_mq2g256_lloyd_moe_{down,gate_up}_indexed*.gfx942.hip`. There is no dense
Lloyd kernel for CDNA3, and qwen3.8 is a dense hybrid whose DeltaNet layers need
the fused QKVZA variant. Closing this needs new gfx942 dense Lloyd kernels, not
a dispatch fix.

Consequence: sub-4-bit dense evaluation for this model must run on gfx11/gfx12,
or wait for CDNA3 dense Lloyd kernels.

## 4. Blocker — gemma4 / glimmer KLD path yields an implausible oracle

`build_kld_ref_native` gained llama / gemma4 / glimmer / lfm2 branches during
this campaign (it was qwen35-only, as were `eval_hipfire` and `greedy_dump`).
The gemma4 branch runs, but the reference it produces is not usable:

| teacher | slice | oracle NLL | oracle PPL |
|---|---|---|---|
| qwen3.8-27b BF16 | WT2 | 1.830742 | **6.2385** |
| gemma4-12b BF16 | WT2 | 7.238736 | **1392.33** |
| gemma4-12b BF16 | AG | 5.442017 | 230.91 |

A lossless BF16 passthrough of the upstream parent scoring 1392 on wikitext-2 is
broken, not weak. Two hypotheses were tested and **both eliminated**:

- *Batched/per-token seam.* Replacing the hybrid (batched 128-chunk prefix +
  per-token scored window) with a pure per-token loop moved PPL to 2012 — worse.
- *BF16 execution.* Re-running the same teacher with `HIPFIRE_CALIB_BF16=0`, so
  weights widen to F32 and the MFMA path is bypassed entirely, gives PPL 1423 —
  unchanged. The defect is dtype-independent.

Remaining suspicion, untested: gemma4's interleaved Sliding/Full attention with
two separately-sized KV caches and per-layer window masking. NLL 7.24 sits
between correct (~2.3) and uniform (`ln 262144 = 12.5`), which is the signature
of a forward that is partially right — consistent with attention context being
wrong while embeddings and MLP still carry signal.

**This retroactively qualifies the non-qwen calibration artifacts.** gemma4
(328/328), glimmer (416/416), and both LFM2 models (92/92) all passed coverage
verification against artifact bytes, but *coverage proves the capture tap fired
on the right weight pointers, not that the forward computed correct values*.
Until an arch's oracle PPL is plausible, its Hessians are unvalidated. Only
qwen3.8 currently has that proof (PPL 6.2385 plus coherent greedy decode).

## 5. Blocker — LFM2 `--format mq4` ignores calibration entirely

For lfm2.5-350m and lfm2.5-1.2b, all four arms (`base`, `awq`, `ldlq`, `both`)
are byte-identical (md5 `7fcd86de6836` for the 350m). The quantizer logs
`imatrix derived from HFQM calibration: 184 keys` and
`AWQ pre-scaling: ENABLED (alpha=0.05)`, then emits tensors labelled **`MQ4-LFM`**,
not `MQ4G256`, and reports `GPTQ-MQ4: 0 tensors GPTQ'd, 0 fell back` — the dense
MQ4G256 GPTQ arm never sees these tensors at all. The LFM-specific MQ4 recipe
consumes neither the imatrix nor the Hessian.

By contrast gemma4 and glimmer *do* respond (distinct md5s, 88 and 416
`MQ4G256` tensors respectively), so this is specific to the LFM path.

---

## Method (identical to the companion record unless noted)

| field | value |
|---|---|
| GPU / ROCm | MI300X `gfx942`; HIP 7.0.51831-7c9236b16 |
| parent | `/scratch/parents/qwen3.8-27b` |
| references | WT2 + AG, 24 chunks / 24,552 scored tokens each, top-k 256, n-ctx 2048 |
| oracle | WT2 NLL 1.830742 / PPL 6.2385; AG NLL 2.052225 / PPL 7.7852 |
| scoring | `eval_hipfire --scoring-mode prefill --kv-mode q8` |
| repeat | scoring verified deterministic (repeat run bit-identical); one run per arm is the full result |
| calib corpus | `bartowski_v5.calib.txt` md5 `fcf30d293b369f71f0eded4e8f62cf74` |

## What is still unmeasured

- **LDLQ/GPTQ at `alpha=0.05`.** Running at time of writing; measured rate is
  ~15 tensors/hour against 497 tensors (K up to 17408), i.e. roughly 30 h. No
  GPTQ number appears in either record.
- Clip search and group-wise vs tensor-wise AWQ: not exposed as CLI flags on
  `hipfire-quantize`, so untestable without quantizer changes.
- Any throughput measurement. Neither record makes one.
