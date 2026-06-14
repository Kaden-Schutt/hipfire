# MQ Sub-4-Bit Research & Development PRD

**Status:** Draft (post-Q0 verification)  
**Last updated:** 2026-04-30  
**Owner:** hipfire research  
**Companion docs:** `mq-sub4bit-research-queue.md` (prioritized task queue)

---

## 1. Executive Summary

hipfire's MagnumQuant (MQ) family — MQ4, MQ3, MQ2 — applies a Fast Walsh-Hadamard Transform (FWHT) rotation to weight blocks before uniform affine quantization. The rotation spreads outlier mass across the block, enabling lower bit widths than flat quantization at the same perceptual quality. An empirical sweep on master `c448d5e` (2026-04-30, gfx1100, 7900 XTX) established that **engine wiring is correct** (zero panics, monotonic quality vs. model size) but that **format lossiness dominates below MQ4** on small models (≤9B).

A bit-exact kernel verification (Q0) confirmed that the GPU kernels reproduce the quantizer's arithmetic to within fp16 rounding tolerance, closing the "kernel bug" hypothesis. The remaining quality gap is therefore in the quantizer math itself.

This PRD lays out five phased research/engineering tracks (Q1–Q5) backed by internal empirical data and external literature. The goal is to either **rescue MQ3 quality at 4B/9B** (move borderline → fluent) or **rescue MQ2 at any size** (move all-fail → at-least-9B-pass), while preserving the bandwidth wins that make sub-4-bit attractive (~3.25 bpw for MQ3, ~2.25 bpw for MQ2).

---

## 2. Background: The MQ Pipeline

### 2.1 How MQ works today

1. **FWHT rotation (offline, quant-time):** Each 256-element weight block is multiplied by a PRNG-sign vector, run through an in-place butterfly Hadamard transform, scaled by `1/16`, then multiplied by a second PRNG-sign vector. The signs are deterministic (`seed1=42`, `seed2=1042` for all models).
2. **Uniform affine quantization:** The rotated block is mapped to `N` levels via `scale = (max-min)/(2^bits-1)`, `zero = min`. For MQ3: 8 levels (0–7); for MQ2: 4 levels (0–3).
3. **Storage:** Per-block header holds `f32 scale` + `f32 zero` (8 bytes). Data follows in packed little-endian bitstreams: MQ3 = 96 B (104 B total), MQ2 = 64 B (72 B total).
4. **Runtime:** The activation vector `x` is rotated once per layer via `mq_rotate_x` (same FWHT math, ~9 µs on gfx1100), then a standard HFQ GEMV kernel dequantizes weights on-the-fly with `recon = scale * q + zero`.

### 2.2 Empirical sweep recap (internal)

Run on 2026-04-30 against Qwen 3.5/3.6 family, 32 generations, **0 hard errors**:

| size | MQ3 | MQ2 |
|------|-----|-----|
| 0.8B | gibberish | mojibake |
| 4B   | partial (intent recognised, `<think>` loops, language drift) | symbol soup |
| 9B   | borderline (factual ✓, multi-step reasoning loops) | symbol soup |
| 27B 3.5 | fluent | — |
| 27B 3.6 | **cleanest output of the sweep** | — |

**Interpretation:** The collapse is not a kernel bug (no panics, monotonic quality, 27B fluent). It is format-lossiness: uniform 4-level or 8-level grids cannot represent the post-FWHT weight distribution with enough fidelity on small models, where each parameter carries more representational burden.

---

## 3. Smoke Test: Q0 — Bit-Exact Kernel Verification

### 3.1 Goal

Prove (or disprove) that `gemv_mq3g256_with_rotate` and `gemv_mq2g256_with_rotate` produce values bit-exact (or within fp16 rounding) to a CPU reference of the same math.

### 3.2 Method

- **Harness:** `crates/engine/examples/verify_mq_kernel.rs` (committed 2026-04-30).
- **Inputs:** Deterministic pseudo-random weights and activations (`fract_sin` PRNG) at shapes `(4,256)`, `(4,512)`, `(8,1024)`.
- **CPU reference:**
  1. Rotate `x` with `cpu_fwht_256(signs1, signs2)` matching quantizer math exactly.
  2. Dequantize each weight block: unpack bits, reconstruct with `scale * q + zero`.
  3. Compute `y = Σ w_rot * x_rot`.
- **GPU path:** Upload quantized bytes, run `gpu.gemv_mq{3,2}g256_with_rotate`, read back `y`.
- **Metrics:** `max_abs_err`, `mean_abs_err`, `bit_exact_count`.

### 3.3 Results (gfx1100, release build)

| shape   | format | max_abs_err | mean_abs_err | bit_exact | verdict |
|---------|--------|-------------|--------------|-----------|---------|
| 4×256   | MQ3    | 9.16e-05    | 6.10e-05     | 0/4       | PASS    |
| 4×256   | MQ2    | 9.16e-05    | 3.81e-05     | 1/4       | PASS    |
| 4×512   | MQ3    | 1.22e-04    | 7.63e-05     | 1/4       | PASS    |
| 4×512   | MQ2    | 3.05e-04    | 1.53e-04     | 1/4       | PASS    |
| 8×1024  | MQ3    | 7.32e-04    | 4.20e-04     | 1/8       | PASS    |
| 8×1024  | MQ2    | 9.16e-04    | 4.20e-04     | 0/8       | PASS    |
| all     | rot    | 0.00e+00    | 0.00e+00     | 256–1024/256–1024 | PASS |

The FWHT `rotate_x_mq` step is **fully bit-exact** (all elements exact match). All GEMV deltas sit well inside the acceptance threshold of `1e-3`.

### 3.4 Conclusion

**Kernel is correct.** The small-model quality collapse is purely format-lossiness. Close Q0 and move to quantizer-side improvements.

---

## 4. External Research Landscape

### 4.1 llama.cpp K-quants and IQ-quants

llama.cpp's `Q2_K` and `Q3_K` formats are the most widely deployed sub-4-bit schemes. They use a **hierarchical super-block** design:

- **Super-block:** 256 weights.
- **Sub-blocks:** 16 weights each (Q2_K, Q3_K, Q6_K) or 32 weights each (Q4_K, Q5_K).
- **Scale/min quantization:** Sub-block scales are themselves quantized to 4–6 bits, and a per-super-block fp16 scale (`d`) rescales them. This two-level hierarchy keeps the metadata overhead small while allowing per-sub-block adaptation.
- **Formulas:**
  - Q2_K: `x = a·q + b` (scale + offset), **2.625 bpw** (84 bytes / 256 weights).
  - Q3_K: `x = a·q` (scale only), **3.4375 bpw** (110 bytes / 256 weights).

**Relevance to hipfire:** MQ2/MQ3 already beat K-quants on bandwidth (MQ2 = 2.25 bpw vs Q2_K = 2.625 bpw; MQ3 = 3.25 bpw vs Q3_K = 3.4375 bpw). However, K-quants achieve better quality at the cost of higher bpw. The IQ (implied-quant) family introduced in late 2023/early 2024 goes further by using **non-uniform codebooks** derived from an importance matrix (imatrix). `IQ2_XXS` achieves **2.0625 bpw** with a true 2-bit codebook plus fp16 scale — but still requires calibration data.

### 4.2 QuIP# and Hadamard Incoherence Processing

**QuIP** (Chee et al., 2023) and **QuIP#** (Tseng et al., ICML 2024) are the SOTA academic baselines for extreme LLM quantization.

**Key ideas:**
1. **Incoherence processing:** Multiply weights by a random orthogonal matrix (randomized Hadamard transform, RHT). This makes the weight distribution approximately Gaussian and eliminates axis-aligned outliers.
2. **Lattice codebooks (QuIP#):** For 2-bit, use an E₈ lattice codebook — a hardware-friendly vector quantizer that packs 8-dimensional Gaussian vectors optimally. For 3-bit, combine the 2-bit E8P codebook with a 1-bit E8 codebook.
3. **Theoretical guarantees:** QuIP is the first LLM quantization algorithm with a formal convergence analysis.

**Empirical comparison (llama.cpp community replication, 2023):**

| Model | QuIP# PPL | QuIP# Size | Q2_K PPL | Q2_K Size |
|-------|-----------|------------|----------|-----------|
| LLaMA-2-7B | 8.201 | 2.15 GB | 6.025 | 2.23 GB |
| LLaMA-2-13B | 6.003 | 3.83 GB | 5.152 | 4.26 GB |
| LLaMA-2-70B | 4.156 | 18.2 GB | 3.671 | 22.9 GB |

**Key insight:** Q2_K *outperforms* QuIP# on smaller models despite similar size. QuIP# only becomes competitive at 70B+. This aligns with our internal finding that sub-4-bit quality is fundamentally size-dependent; small models need every bit of representational capacity.

**Relevance to hipfire:** hipfire already uses FWHT (a deterministic orthogonal transform), which is a close cousin of QuIP#'s RHT. The difference is:
- QuIP# uses **vector quantization** (E8P lattice) rather than scalar uniform quantization.
- QuIP# requires **calibration data** (Hessian-aware) for best results.
- QuIP# is not optimized for GPU GEMV throughput; lattice lookups are expensive.

Our Q1 (Lloyd-Max per-block codebooks) is essentially a scalar, hardware-friendly approximation of QuIP#'s vector codebook idea.

### 4.3 Lloyd-Max and Locally Optimal Block Clustered Quantization (LO-BCQ)

**Lloyd-Max** (Lloyd 1957 / Max 1960) iteratively refines quantization centroids to minimize mean squared error. For a 1D distribution, it is equivalent to k-means with `k = 2^bits`.

**LO-BCQ** (Rakka et al./OpenReview 2024) explicitly applies 1D Lloyd-Max and 2D k-means to weight blocks for 4-bit quantization. Their finding:

> "Minimizing quantization MSE using the 1D (Lloyd-Max) and 2D K-means clustering has been explored in (Han et al., 2016; Cho et al., 2021; 2023) ... LO-BCQ extends this to per-block locally optimal centroids."

**Relevance to hipfire:** A true 4-entry Lloyd-Max codebook per 256-weight block (Q1) is mathematically distinct from the uniform affine map `{scale*0+zero, ..., scale*3+zero}`. For skewed or bimodal post-FWHT distributions, Lloyd-Max can place centroids where the mass actually lives, reducing MSE by 15–40% vs. uniform quantization on heavy-tailed data (this is a well-established result in information theory).

The storage cost is identical to uniform MQ2: 4 fp16 centroids = 8 bytes header + 64 bytes data = 72 B/group. The VGPR cost in the kernel is +2 fp16 registers (4 vs. 2), which is negligible on RDNA (well under the spill threshold).

### 4.4 GPTQ-Style Block-Wise Error Compensation

**GPTQ** (Frantar et al., ICLR 2023) is the canonical OBS-inspired PTQ method. It processes weights column-by-column, quantizes each column, computes the error, and propagates it forward into unquantized columns via the inverse Hessian of the activation covariance:

```
error = w_quantized - w_original
w_remaining -= error * H_inv_block
```

**Key properties:**
- Requires a small calibration dataset (~128 samples of activations).
- Uses Cholesky-decomposed Hessian inverse for numerical stability.
- Standard implementation uses fp64 for the Hessian; fp32 is often sufficient for rotated distributions.
- Routinely rescues sub-4-bit quality on ≥7B models (e.g., OPTQ/GPTQ at 3-bit on LLaMA-7B recovers >95% of fp16 perplexity).

**Relevance to hipfire:** Q2 proposes adding a calibration pass to `hipfire-quantize` that captures activation Hessians per layer and applies the GPTQ inner loop to MQ3/MQ2 quantization. Because our weights are already FWHT-rotated, the Hessian may be better conditioned (the rotation whitens the covariance), potentially allowing fp32 throughout.

### 4.5 Mixed-Precision Quantization Policies

The 2025 survey *Mixed-Precision Quantization for Language Models* (Rakka et al., arXiv 2510.16805) formalizes three categories:

1. **MPW** — Mixed-precision weights, full-precision activations.
2. **MPW+UPA** — Mixed-precision weights, uniform-precision activations.
3. **MPW+MPA** — Mixed-precision weights and activations.

**Key finding for hipfire:**

> "Uniform low-bit quantization can degrade accuracy in sensitive transformer components. Mixed-precision quantization selectively allocates precision across layers/tensors to balance efficiency and accuracy."

The llama.cpp Q2_K format already uses a **mixed-precision policy internally**: attention K/Q may be Q2_K, output.weight is Q6_K, and token embeddings are often Q5_K. Similarly, `IQ3_S` (a successor to Q3_K) uses a mix of `IQ3_XXS` and `IQ3_S` per tensor class.

A unified evaluation on Llama-3.1-8B (Kurt, 2026) shows that **task sensitivity varies by tensor class**: GSM8K (math) is most sensitive to quantization, while HellaSwag is nearly impervious. This suggests that a policy allocating higher precision to `o_proj`, `lm_head`, and `down_proj` while aggressively quantizing `gate_proj`/`up_proj` is well-motivated.

**Relevance to hipfire:** Q4 (mixed-MQ) implements exactly this policy: attention projections at MQ4 (or MQ6), FFN gate/up at MQ3, down_proj at MQ3, lm_head at MQ4, embeddings at Q8F16. Average ~3.3 bpw — same bandwidth as uniform MQ3, but with the most sensitive tensors protected.

### 4.6 Per-Model Calibrated Sign Vectors (Incoherence Randomization Tuning)

QuIP# uses a **random** Hadamard transform (random sign flips + butterfly). hipfire currently uses **deterministic** PRNG seeds (42, 1042) shared across all models.

The post-FWHT weight distribution depends on the sign pattern. A sign vector that happens to align an outlier-rich region with a destructive butterfly path will spread mass more evenly, tightening the dynamic range that quantization must cover.

**Relevance to hipfire:** Q3 proposes a random-restart search over candidate seed pairs, measuring per-block dynamic range or kurtosis. This is computationally cheap (<60s per model) and requires no format or kernel changes — only a metadata field storing the chosen seeds.

---

## 5. Phased Roadmap

### Phase 1 — Q1: True Non-Uniform Lloyd-Max Codebook for MQ2 (executed 2026-05-01: research-only)

**Outcome (2026-05-01):** Implemented as `qt=19` (`MQ2G256Lloyd`) and
ppl-validated. The codebook delivers a 41–55× ppl reduction over uniform
MQ2 (e.g. 9B 120,108 → 2,163) but the absolute floor is still
text-collapse — bit-width is the binding constraint, not codebook
shape. **Stays research-only**, gated behind `--allow-mq2-lloyd` /
`HIPFIRE_ALLOW_MQ2_LLOYD=1`. The format is plumbed for future
combinations with GPTQ (Phase 2 / Q2) or QuIP#-style RHT (queue Q5).

The win moved to **Phase 1.5 (Lloyd-Max MQ3)** below, which is what
actually ships.

**Original spec (kept for reference):**
- 4 fp16 centroids = 8 bytes header.
- 64 bytes data (256 × 2 bits).
- Total: **72 bytes / group** — bit-exact same bandwidth as uniform MQ2.
- Kernel: 4-entry register-resident lookup `recon = cb[q & 3]`.
- VGPR: +2 fp16 (19 → 21 on gfx1100), no spill.
- Quantize: percentile init at 12.5/37.5/62.5/87.5 → Lloyd's iterations
  (max 8, early-exit on stable assignment) → sort centroids ascending.
- On-disk: `qt=19`.

**Empirical wikitext2-test ppl** (gfx1100, ctx=2048, scored=2039):

| size | uniform MQ2 | Lloyd-MQ2 | Lloyd factor |
|---|---:|---:|---:|
| 0.8B | 803,852 | 19,651 | 40.9× |
| 9B | 120,108 | 2,163 | 55.5× |

**Acceptance gates failed:** target was 9B Lloyd-MQ2 fluent + ppl ≤ MQ4
× 1.06; actual 9B Lloyd-MQ2 ppl=2,163 vs MQ4 10.34 = 209× — far below
fluency floor. Original criteria not reachable at 2 bpw on Qwen3.5
without bigger algorithmic change (Phase 2 GPTQ stack).

### Phase 1.5 — Q1.5: Lloyd-Max MQ3 (validated 2026-05-01, ships pending dual gate)

**Outcome:** This is what the Lloyd-Max work actually delivers.
**Status:** Implemented as `qt=20` (`MQ3G256Lloyd`); ppl-validated; ships
as the 3-bit default once both gates land.

**Storage layout:**
- Header: **8 fp16 centroids = 16 bytes** (vs uniform MQ3's 8 B fp32
  scale + zero — the codebook header doubles).
- Data: 96 bytes packed 3-bit indices (cross-byte layout unchanged
  from uniform MQ3, so kernel unpack code is identical except for the
  reconstruction step).
- Total: **112 bytes / group** vs uniform MQ3's 104 B → **+7.7 %
  bandwidth cost**.

The +7.7% is the cost; the win is a 2.27× ppl reduction at 9B (42.03 →
18.52). Bandwidth comparison against MQ4 (136 B/group) is **−17.6%**:
Lloyd-MQ3 is materially smaller than MQ4 while sitting at 1.79× MQ4
ppl (vs uniform MQ3's 4.07×).

**Kernel:** `gemv_mq3g256_lloyd.hip` with 8-entry register-resident
codebook lookup (current implementation uses an 8-way switch which
costs 3.2× decode vs uniform MQ3 — see ship gate 1 below).

**Quantizer:** `quantize_mq3g256_lloyd` in `hipfire-quantize`. Per
256-element block: percentile init at 1/16, 3/16, …, 15/16 → Lloyd's
iterations (max 8, early-exit) → sort centroids ascending → pack 3-bit
indices same cross-byte layout as uniform MQ3. Parallelized via rayon
`par_chunks_mut` over output blocks (9B quantize ~85s wall on 24-core).

**On-disk:** `qt=20` (`MQ3G256Lloyd`).

**Empirical wikitext2-test ppl:**

| size | MQ4 | uniform MQ3 | **Lloyd-MQ3** | Lloyd factor | vs MQ4 |
|------|---:|---:|---:|---:|---:|
| 0.8B | 25.65 | 301.06 | **155.22** | 1.94× | 6.05× |
| 4B   | 12.73 | 45.24  | **22.56**  | 2.01× | 1.77× |
| 9B   | 10.34 | 42.03  | **18.52**  | 2.27× | 1.79× |

**Ship gates** (both required before flipping `qt=20` to default and
removing the `--allow-mq3-lloyd` guard):

1. **Decode perf:** K4-unroll kernel restores 9B decode to ≥120 tok/s
   on gfx1100 (current 8-way switch implementation: 44 tok/s vs uniform
   MQ3's 141). Don't ship the 44 tok/s path — the user-visible 3.2×
   latency regression cancels the quality win in chat usage.
2. **Coherence eyeball:** 4-prompt coherence battery passes on 4B and
   9B Lloyd-MQ3 with no attractor loops at the new ppl floor.

Until both clear, Lloyd-MQ3 stays research-gated behind
`--allow-mq3-lloyd` / `HIPFIRE_ALLOW_MQ3_LLOYD=1`.

**Sub-9B status:** 0.8B Lloyd-MQ3 still text-collapse (155 vs MQ4 26 =
6× worse). Below 9B, Lloyd alone is insufficient — needs Phase 2
(GPTQ) or Phase 4 (mixed-MQ with MQ4 critical layers) on top.

**See also:**
- `docs/plans/mq-sub4bit-research-queue.md` Q1.5 — canonical research log.
- `benchmarks/results/lloyd_max_findings_20260501.md` — empirical writeup.
- `docs/plans/mq3-rounding-out-precompute-leverage.prd` §A — perf-leverage view.

### Phase 2 — Q2: GPTQ-Style Block-Wise Error Compensation

**Effort:** 1–2 weeks  
**Impact:** High — could rescue MQ3 at 4B/9B without changing kernel or storage  
**Risk:** Moderate (adds calibration data dependency)

**Description:** When quantizing block-by-block, propagate the quantization error of already-quantized blocks forward into the still-unquantized blocks via the inverse Hessian of the activation covariance.

**Implementation:**
1. Add `--calibrate <samples>` flag to `hipfire-quantize`.
2. Load source model in f16, run ~128 samples (WikiText-2, C4, or cached data), record per-layer activation Hessians.
3. Cholesky-factor the Hessian inverse per layer.
4. In `quantize_mq3g256` (and `_mq4` for completeness), replace independent per-block quantization with:
   ```
   quantize column i
   error = quantized - original
   remaining_columns -= error * H_inv_block
   ```
5. Record calibration config in `.hfq` metadata for reproducibility.

**Acceptance:**
- 4B MQ3-GPTQ produces fluent output (current vanilla: partial collapse).
- 9B MQ3-GPTQ produces fluent multi-step reasoning (current vanilla: loops).
- 27B MQ3 unchanged or improved (already fluent; floor not regressed).

### Phase 3 — Q3: Per-Model Calibrated Sign Vectors

**Effort:** 1 week  
**Impact:** Medium — cheaper than GPTQ, smaller gains  
**Risk:** Low

**Description:** Replace global PRNG-seeded sign vectors (seeds 42 / 1042) with per-model calibrated vectors that minimize post-FWHT block-level dynamic range.

**Method:**
1. Random-restart search: try N ∈ [16, 64] candidate seed pairs.
2. For each, apply FWHT to a sample of the model's weights and measure:
   - `block_dynamic_range = max(|w_rot|) / median(|w_rot|)` averaged across blocks (smaller = better), OR
   - per-block kurtosis (lower = more uniform-tolerant).
3. Pick the seed pair minimizing the metric.
4. Store seeds in `.hfq` metadata header (2 × u64).
5. Engine loader defaults to old constants when field is absent.

**Acceptance:**
- 4B MQ3 with calibrated seeds outperforms vanilla 4B MQ3 on the coherence battery.
- Calibration runs in <60s per model.
- Engine load is a no-op slowdown.

### Phase 4 — Q4: Mixed-Precision MQ-Hybrid

**Effort:** 3–5 days  
**Impact:** Medium — near-term release lever, no new math  
**Risk:** Very low

**Description:** Add `--format mixed-mq` to `hipfire-quantize` with a per-tensor policy table. **The 3-bit slot uses Lloyd-MQ3 (qt=20, 112 B/group), not uniform MQ3** — see Phase 1.5 for the empirical justification:

| Tensor class | Format | B/group |
|---|---|---:|
| q_proj, k_proj, v_proj, o_proj | MQ4 (or MQ6 for sensitive heads) | 136 |
| gate_proj, up_proj | Lloyd-MQ3 | 112 |
| down_proj | Lloyd-MQ3 (widest matrix → biggest bandwidth win) | 112 |
| lm_head | MQ4 (logits sensitivity) | 136 |
| Embeddings | Q8F16 (existing) | — |
| Norms | F16 (existing) | — |

**Expected average:** ~3.5 bpw — slightly above uniform MQ3's 3.25 bpw because Lloyd-MQ3 carries +7.7% header. The quality reclaim (2.27× ppl reduction on the 3-bit slot) makes the bandwidth bump trivially worth it.

**Phase 4 is gated on Phase 1.5 ship gates clearing** — Lloyd-MQ3 must
have its K4-unroll perf fix and coherence eyeball pass before mixed-MQ
ships, or mixed-MQ inherits the 44 tok/s decode regression in its
3-bit slot.

**Acceptance:**
- 4B mixed-MQ passes coherence battery where 4B MQ3 partially collapses.
- Average bpw ~3.3–3.5.

### Phase 5 — Q5: WMMA Prefill Kernels for MQ3 / MQ2

**Effort:** 1–2 weeks  
**Impact:** Pure performance, not quality  
**Risk:** Low (engineering, not research)

**Description:** Currently MQ3/MQ2 prefill falls back to per-row GEMV (~43 tok/s prefill at 27B vs. ~70 for batched MQ4 + WMMA). Add WMMA prefill kernels:

- `gemm_qkvza_mq3g256_wmma_gfx12.hip`
- `gemm_gate_up_mq3g256_wmma`
- `gemm_mq3g256_residual_wmma`

Same shape as existing MQ4 WMMA family but with 3-bit unpack inside the K-tile loop. Add to `is_batchable_la` eligibility check.

**Acceptance:**
- 27B MQ3 prefill ≥1.5× current per-row GEMV speed.
- Channel-tests pass on gfx1201 (R9700) + fp16 reference matches gfx1100 dot2 fallback.
- Coherence-gate clean post-merge.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Lloyd-Max MQ2 still collapses at 4B/9B | Medium | High | Escalate to activation-weighted Lloyd's (GPTQ-calibrated centroids) or drop MQ2 entirely. |
| GPTQ calibration too memory-heavy | Low | Medium | Use fp32 Hessian (validate against fp64 on a small model); cap samples at 64. |
| Calibrated sign seeds yield <1% gain | Medium | Low | Cheap to run; if no gain, abandon Q3 and document. |
| Mixed-MQ policy overfits to Qwen3.5 | Low | Medium | Validate on Qwen3.6 and Llama-style GGUF paths before shipping default. |
| WMMA MQ3 kernel adds compile-time | Low | Low | Guard behind feature flag until channel-tested. |

---

## 7. Cross-Cutting Agent Guidance

- **Always use `scripts/mq3-mq2-sweep.sh`** for empirical verdicts. Eyeball the report; hard-fail predicates (panic / 0 tokens / timeout) won't catch quality regressions.
- **Always record source-input md5 + binary md5** alongside any quality / perf claim.
- **27B 3.6 MQ3 is the canonical fluent sub-4-bit reference.** Compare new variants against it on the same 4 prompts.
- **Don't touch `is_batchable_la` to admit MQ3/MQ2 until Q5 lands.** The current per-token fallback is the reason zero panics surfaced.
- **Do not store canonical artifacts under `/tmp`.** Use `benchmarks/prompts/`, `~/.hipfire/datasets/`, or committed scripts.

---

## 8. References

1. **QuIP:** Chee et al., *QuIP: 2-Bit Quantization of Large Language Models With Guarantees*, arXiv:2307.13304, 2023.
2. **QuIP#:** Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*, ICML 2024. arXiv:2402.04396.
3. **GPTQ:** Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023. arXiv:2210.17323.
4. **LO-BCQ:** *Locally Optimal Block Clustered Quantization for 4-bit LLMs*, OpenReview 2024 / arXiv:2502.05376.
5. **Mixed-Precision Survey:** Rakka et al., *Mixed-Precision Quantization for Language Models: Techniques and Prospects*, arXiv:2510.16805, 2025.
6. **llama.cpp K-quants:** ikawrakow, "New SOTA 2-Bit Quant released: QuIP-Sharp", GitHub Discussion #4327, 2023. Community replication and improved Q2_K results.
7. **llama.cpp Quant Structures:** `ggml-quants.c` definitions (Q2_K–Q8_K, IQ2_XXS/XS). See eadst.com/blog/232 for summary.
8. **Unified llama.cpp Evaluation:** Kurt, *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*, arXiv:2601.14277, 2026.
9. **Lloyd-Max:** Lloyd (1957), *Least Squares Quantization in PCM*; Max (1960), *Quantizing for Minimum Distortion*.
10. **hipfire internal sweep:** `docs/plans/mq-sub4bit-research-queue.md`, 2026-04-30.

---

## 9. Appendix: Q0 Test Reproduction

```bash
cargo run --release --example verify_mq_kernel -p hipfire-runtime
```

Expected output (gfx1100, release):
```
[PASS] All MQ3/MQ2 kernels bit-exact within 1e-3.
```

If `max_abs_err > 1e-3` on any shape, the kernel has a bug. Bisect to:
- FWHT scaling factor (`1/16` vs `1/sqrt(256)`)
- Sign-vector application order
- Byte unpack ordering (cross-byte vs. linear)
- Scale/zero-point header read alignment
