# MQ Sub-4-Bit Research Queue (post-2026-04-30 sweep)

Companion to `mq-sub4bit-roadmap.prd`. The roadmap defines the *paths*;
this document is the prioritized *research queue* generated from the
empirical sweep on master `c448d5e` (2026-04-30, gfx1100, 7900 XTX).

## Sweep recap (32 generations, 0 hard errors)

| size | MQ3 | MQ2 |
|------|-----|-----|
| 0.8B | gibberish | mojibake |
| 4B   | partial (intent recognised, `<think>` loops, language drift) | symbol soup |
| 9B   | borderline (factual ✓, multi-step reasoning loops) | symbol soup |
| 27B 3.5 | fluent | _not tested past 9B fail_ |
| 27B 3.6 | **cleanest output of the sweep** | _not tested past 9B fail_ |

**Engine wiring is correct** (zero panics, monotonic quality vs size,
27B fluent). The collapse is format-lossiness on small models, not a
kernel bug. PR #109 added refuse-by-default for MQ2 and quality
advisory for MQ3 < 9B.

This queue identifies quality-lever research tasks that could:
- **Rescue MQ3 quality at 4B / 9B** (move borderline → fluent)
- **Rescue MQ2 at any size** (move all-fail → at-least-9B-pass)

---

## Q0 — Bit-exact kernel verification (forensic, 1-2 days)

**Status:** Pending.
**Effort:** 1-2 days.
**Quality risk:** None — purely diagnostic.
**Why first:** Locks down the question of "is the kernel doing what we
think it's doing" before we tune anything. Cheap to run, eliminates the
biggest unknown the user surfaced.

### Goal

Prove (or disprove) that `gemv_mq3g256_with_rotate` and
`gemv_mq2g256_with_rotate` produce values bit-exact (or within fp16
rounding) to a CPU reference of the same math.

### Plan

1. Pick the smallest .mq3 / .mq2 artifact (`qwen3.5-0.8b.mq3`).
2. Write a CPU reference: `cpu_reference_gemv(W_rot, x, m, k) -> y`
   that loads MQ3 bytes, dequantizes per the kernel's reconstruction
   formula (`scale * q + zero`), runs `cpu_fwht_256(signs ⊙ x) / 16`,
   then a plain f32 matvec.
3. Capture the GPU dispatch's `y` for the same inputs (enable a debug
   tap in `gemv_mq3g256_with_rotate`).
4. Compare element-wise: `max_abs_err`, `mean_abs_err`,
   `max_rel_err`, `bit_exact_count / total`.

### Acceptance

- **`max_abs_err <= 1e-3`** in fp16 mixed-precision: kernel is correct;
  the small-model collapse is purely format-lossiness. Close this task.
- **`max_abs_err > 1e-3`**: kernel has a bug. Bisect to: (a) FWHT
  scaling, (b) sign-vector application, (c) byte unpack ordering,
  (d) scale/zero-point header read.

### Owner / files

- New: `crates/engine/examples/verify_mq3_kernel.rs` (CPU/GPU compare).
- Reads: `crates/rdna-compute/src/dispatch.rs` (kernel wrapper), 
  `kernels/src/gemv_hfq3g256.hip` (decode formula),
  `crates/hipfire-quantize/src/main.rs` (`quantize_mq3g256` output layout).

---

## Q1 — True non-uniform 4-entry codebook for MQ2 (revised; supersedes PRD §5.4 framing)

**Status:** Pending.
**Effort:** 1-2 weeks.
**Quality risk:** Low if implemented carefully — the engine already does
codebook-style reconstruction for KV cache (`asym3` / `asym4`).
**Why second:** Highest-impact quality lever for MQ2. The sweep failed at
every size tested; if a true non-uniform codebook doesn't rescue it,
MQ2 is likely off the table for hipfire entirely.

### Caveat on the PRD's framing (and why this revision exists)

PRD §5.4 describes "Lloyd-Max" as: *"store `cb_min` and `cb_max` (2 floats).
Reconstruction is `lerp(cb_min, cb_max, q / (2^K - 1))`."*

This is **mathematically identical to the existing uniform `scale * q +
zero`** — just an affine map renamed (`scale = (cb_max - cb_min)/(2^K-1)`,
`zero = cb_min`). It buys nothing over what we already do. The PRD's
ternary alternative `{-thresh, 0, +thresh}` only uses 3 of 4 codes,
wasting 25% of the bit budget — strictly worse than uniform MQ2.

True Lloyd-Max (Lloyd 1957 / Max 1960) is iterative refinement of a
**codebook of N reconstruction values** (centroids), not just two
boundary parameters. For 2-bit quant that means **4 fp16 centroids per
block**, indexed by the 2-bit `q` via direct lookup.

### Goal

Replace uniform MQ2's `{scale*0+zero, scale*1+zero, scale*2+zero,
scale*3+zero}` per-block reconstruction with a **per-block 4-entry
codebook** of fp16 values produced by Lloyd's algorithm minimizing
`E[(w − cb[q])²]` over the FWHT-rotated weights of that block.

This is genuinely non-uniform — the codebook can fit bimodal,
asymmetric, or heavy-tailed distributions that uniform-grid quant
cannot.

### Storage layout (preserves bandwidth budget)

- **Header**: 4 fp16 centroids = **8 bytes** (same byte count as the
  uniform MQ2 header: 4-byte fp16 scale + 4-byte fp16 zero).
- **Data**: 64 bytes (256 × 2 bits) — unchanged.
- **Total**: **72 bytes / group** — bit-exact same as uniform MQ2.

The bandwidth win of MQ2 (0.281 bytes/param) is preserved completely.

### Plan

1. **Quantizer**: `quantize_mq2g256_lloyd(f32_data, signs1, signs2)` in
   `crates/hipfire-quantize/src/main.rs`. Per 256-element block:
   1. Apply `cpu_fwht_256(signs ⊙ block) / 16`.
   2. Initialize 4 centroids at the 12.5/37.5/62.5/87.5 percentiles.
   3. **Lloyd's algorithm** to convergence (typically <10 iterations):
      assign each weight to its nearest centroid; recompute each
      centroid as the mean of its assigned weights; repeat until
      assignments stabilize.
   4. Pack the 4 final centroids as fp16 (8 bytes header).
   5. Pack each weight's centroid index as 2 bits.
2. **On-disk layout**: bump quant_type to qt 19 (`MQ2G256_LLOYD`). 8 B
   header + 64 B data = 72 B/group.
3. **Engine**: new `DType::MQ2G256Lloyd`. New kernel
   `gemv_mq2g256_lloyd.hip` — copy `gemv_hfq2g256.hip`, replace the
   `recon = scale*q + zero` line with a 4-entry register-resident
   codebook lookup:
   ```c
   // Header: load 4 fp16 codes once into vector registers.
   half4 cb = *(const half4*)(group_ptr);
   // Per-weight: index into codebook by the 2-bit q.
   half recon = cb[q & 3];
   ```
   VGPR cost: 4 fp16 in registers instead of 1 scale + 1 zero = 2 fp16
   — net +2 VGPRs per wave. Negligible (HFQ2 wave32 kernel has ~19
   VGPRs / 20 waves/SIMD; +2 stays well under the spill threshold).
4. **Sweep**: re-run `scripts/mq3-mq2-sweep.sh` against 9B / 27B Lloyd
   variants. Compare against uniform MQ2 (refused-by-default per #109)
   and MQ4 baselines.

### Acceptance

- 9B MQ2-Lloyd produces fluent output on the 4-prompt battery.
- 27B MQ2-Lloyd produces fluent output.
- Perplexity delta vs MQ4 ≤ 6%.
- Bytes/param == uniform MQ2 (no bandwidth regression).

### What asym3/asym4 KV cache already does (precedent + key difference)

The KV-cache `asym3`/`asym4` modes use a **single global non-uniform
codebook** of 8 (asym3) or 16 (asym4) fp32 values fitted to the
expected post-Givens-rotation distribution `N(0, 1/256)`. See
`kernels/src/turbo_common.h:23`:

```c
__constant__ float TURBO_C3_256[8] = {-0.134860f, -0.083320f, -0.046469f,
                                      -0.015176f,  0.015176f,  0.046469f,
                                       0.083320f,  0.134860f};
```

The kernel reconstructs as `recon = TURBO_C3_256[idx]` — direct
indexed lookup. This works because every K vector is unit-normalized
and Givens-rotated, so they all share approximately the same
distribution; one global codebook fits all blocks.

**Weight quantization is harder**: weights have heterogeneous
per-block distributions (different scales, shapes, modalities). A
single global codebook cannot fit all of them — we need a per-block
codebook. The math is identical to the asym3 lookup; the difference
is that the 4 (for MQ2) or 8 (for MQ3 via Q1.5 below) codebook
entries live in the per-block header instead of `__constant__`
memory.

The asym3/asym4 modes are good evidence that the engine's lookup
path is fast and the kernel toolchain handles small register-resident
codebooks fine. They're NOT per-block Lloyd-Max — that's what Q1 adds.

### Risks

- **Codebook ordering**: the 4 centroids are unordered by construction
  (Lloyd's converges to a permutation). Decide whether to sort them
  ascending in the header, or assign indices 0-3 by ordered position
  during quantize. Sorting at quant-time is simplest; the kernel does
  not need to know.
- **Outlier blocks**: blocks with extreme distributions (1 weight 100×
  larger than the rest) will pull a centroid to the outlier and waste
  the other 3. May need a fallback to uniform when codebook variance
  is too low — measure on real distributions first.
- **Calibration**: pure weight-distribution Lloyd's is data-free
  (good — preserves the deterministic property of MQ formats). If
  quality is still insufficient, escalate to GPTQ-style activation-
  weighted Lloyd's (Q2 below).

### Files

- `crates/hipfire-quantize/src/main.rs`: new `quantize_mq2g256_lloyd`.
- `crates/engine/src/hfq.rs`: qt 19 → `DType::MQ2G256Lloyd`.
- `crates/rdna-compute/src/dispatch.rs`: new dispatch arm + GEMV
  wrapper paralleling `gemv_mq2g256_with_rotate`.
- `kernels/src/gemv_mq2g256_lloyd.hip`: new — codebook-lookup variant.
- Reads: literature on llama.cpp's `Q2_K` / `Q3_K` / `Q4_K` (which use
  per-block codebooks already; portable algorithms, ROCm-compatible
  storage layout) for reference; `kernels/src/turbo_common.h` (existing
  global non-uniform codebooks for asym3/4 KV) for the lookup-style
  reconstruction precedent; `kernels/src/kv_fold_asym3.hip:60-62` for
  the `recon = CODEBOOK[idx]` access pattern that already runs on the
  hot path.

### Out-of-band: PRD §5.4 should be revised

If Q1 is approved, update `mq-sub4bit-roadmap.prd` §5.4 to drop the
`(cb_min, cb_max)` / lerp framing — it's a uniform-quant scheme under
a non-uniform-quant name. The actual Path D should be this 4-entry
codebook lookup. Don't sit on incorrect docs.

---

## Q2 — GPTQ-style block-wise error compensation (NOT in PRD)

**Status:** Pending — new proposal, not in roadmap.
**Effort:** 1-2 weeks.
**Quality risk:** Moderate — well-established technique but adds
calibration data dependency.
**Why third:** Could rescue MQ3 at 4B / 9B (move borderline → fluent)
**without changing the kernel or storage format**. The quantizer becomes
smarter; runtime is unchanged.

### Goal

When quantizing block-by-block, propagate the quantization error of
already-quantized blocks forward into the still-unquantized blocks via
the inverse Hessian of the activation covariance. Standard since GPTQ
(Frantar et al., ICLR 2023); routinely rescues sub-4-bit quality on
≥7B models.

### Plan

1. **Calibration data**: add a small calibration pass to
   `hipfire-quantize` that runs ~128 samples through the f16 model and
   records activation Hessians per layer (Cholesky factor, per layer).
   Source: WikiText-2, C4, or whatever's already cached.
2. **GPTQ inner loop**: in `quantize_mq3g256` (and `_mq4`, since the
   technique applies to all bit widths), replace per-block independent
   quantization with: for each column block, quantize, compute error,
   subtract `(error * H_inv_block)` from remaining columns.
3. **Calibration dataset & seeds**: deterministic. Record the calibration
   config in the .hfq metadata so models can be reproduced.
4. **Sweep**: run `scripts/mq3-mq2-sweep.sh` against GPTQ-MQ3 versions
   of 4B and 9B. Compare against vanilla MQ3.

### Acceptance

- 4B MQ3-GPTQ produces fluent output (current vanilla: partial
  collapse). Even modest improvement here is a strong signal.
- 9B MQ3-GPTQ produces fluent multi-step reasoning (current vanilla:
  loops).
- 27B MQ3 unchanged or improved (already fluent; floor not regressed).

### Risks

- Adds a "calibration model load" step to `hipfire-quantize` — need to
  load the source model in f16 to capture activations. ~2× memory at
  quant time. Acceptable since quantize is offline.
- GPTQ's standard implementation uses fp64 for the Hessian inverse. We
  may need to validate fp32 is enough on rotated weight distributions.

### Files

- `crates/hipfire-quantize/src/main.rs`: add `--calibrate <samples>`
  flag, GPTQ inner loop.
- New: `crates/hipfire-quantize/src/gptq.rs` (Hessian capture +
  inversion + block-update math).
- Reads: literature — GPTQ paper (arxiv 2210.17323); marlin / autogptq
  reference code (CUDA but the algorithm is portable).

---

## Q3 — Per-model calibrated sign vectors (incoherence randomization tuning)

**Status:** Pending — new proposal, not in roadmap.
**Effort:** 1 week.
**Quality risk:** Low.
**Why fourth:** Smaller-impact than GPTQ but cheaper. Currently the
FWHT sign vectors are deterministic PRNG seeds (42, 1042) shared
across all models. Per-model calibration could pick sign vectors that
push the post-FWHT weight distribution closer to uniform within each
block — tightening the dynamic range that downstream quantization
(uniform or Lloyd-Max codebook) has to cover.

### Note on terminology

This is NOT AWQ (Activation-aware Weight Quantization). AWQ scales
specific weight columns by per-channel activation magnitude — that
would be a separate task (Q3.5 below if pursued). The technique
proposed here is closer to QuIP-style "incoherence randomization":
search the space of orthogonal-equivalent rotations for the one that
minimizes post-rotation outlier mass.

### Goal

Replace the global PRNG-seeded sign vectors (seeds 42 / 1042) with
per-model (or per-layer-class) calibrated vectors that minimize the
post-FWHT block-level dynamic range or outlier-to-median ratio.

### Plan

1. Random-restart search: try N ∈ [16, 64] candidate seed pairs;
   for each, apply FWHT to a sample of the model's weights and
   measure either:
   - `block_dynamic_range = max(|w_rot|) / median(|w_rot|)` per block,
     averaged across blocks (smaller = better)
   - or per-block kurtosis of the rotated distribution (lower kurtosis
     means more uniform-tolerant)
2. Pick the seed pair minimizing the chosen metric.
3. Store the chosen seeds in the .hfq metadata header (2 × u64).
4. At engine load, read the seeds from metadata and reproduce the
   sign vectors instead of using the hard-coded constants.

### Acceptance

- 4B MQ3 with calibrated seeds outperforms vanilla 4B MQ3 on the
  coherence battery.
- Seed calibration runs in <60s per model.
- Engine load is a no-op slowdown (same FWHT, just different signs).

### Risks

- Backward compat: existing `.mq3` / `.mq2` artifacts use seeds
  42 / 1042. New artifacts would have a different metadata field.
  Loader must default to old constants when the field is absent.
- Diminishing returns: the FWHT rotation is already designed to
  decorrelate weight statistics. Random-restart on top may yield
  only marginal gains. Run Q0 first to confirm the kernel is
  correctly applying the rotation; if rotation fidelity is the
  bottleneck (unlikely but possible), Q3 is more impactful than
  if the rotation is already near-optimal.

---

## Q4 — Mixed-precision MQ-hybrid (PRD §5.3 Path C)

**Status:** Pending — PRD Phase 3.
**Effort:** 3-5 days.
**Quality risk:** Very low.
**Why fifth:** Mostly an engineering task. The PRD already specifies the
policy. Lower research priority than Q1-Q3 but unblocks a near-term
0.1.9 release that improves sub-9B sub-4-bit quality without any new
math.

### Goal

`--format mixed-mq` policy in `hipfire-quantize`:

| Tensor class | Bpw |
|---|---|
| q_proj, k_proj, v_proj, o_proj | MQ4 (or MQ6 for sensitive heads) |
| gate_proj, up_proj | MQ3 |
| down_proj | MQ3 (widest matrix → biggest bandwidth win) |
| lm_head | MQ4 (logits sensitivity) |
| Embeddings | Q8F16 (existing) |
| Norms | F16 (existing) |

Average ~3.3 bpw — same bandwidth as uniform MQ3, hopefully better
quality on sub-9B.

### Acceptance

- 4B mixed-MQ on the coherence battery passes where 4B MQ3 partially
  collapses.
- Average bpw ~3.3-3.5 (target for the bandwidth window).

### Files

- `crates/hipfire-quantize/src/main.rs`: per-tensor policy table,
  `--format mixed-mq` flag.

---

## Q5 — WMMA prefill kernels for MQ3 / MQ2 (PRD Phase 3)

**Status:** Pending — engineering, not research.
**Effort:** 1-2 weeks.
**Why last in this queue:** Pure perf, not quality. After Q1-Q4 we
hopefully have shippable MQ3 (and maybe Lloyd-Max MQ2); only then is
prefill-perf worth the kernel-development cost. Currently MQ3 prefill
falls back to per-row GEMV (43 tok/s prefill at 27B vs ~70 for
batched MQ4 + WMMA).

### Goal

Add `gemm_qkvza_mq3g256_wmma_gfx12.hip`, `gemm_gate_up_mq3g256_wmma`,
`gemm_mq3g256_residual_wmma`, etc. Same shape as the existing MQ4
WMMA family but with 3-bit unpack inside the K-tile loop. Add to
`is_batchable_la` so the eligibility check no longer falls back.

### Acceptance

- 27B MQ3 prefill ≥1.5× current per-row GEMV speed.
- Channel-tests pass for all WMMA variants on gfx1201 (R9700) +
  fp16 reference matches gfx1100 dot2 fallback.
- Coherence-gate clean post-merge.

---

## Cross-cutting agent guidance

- **Always use `scripts/mq3-mq2-sweep.sh`** for empirical verdicts.
  Eyeball the report; the gate's hard-fail predicate (panic / 0
  tokens / timeout) won't catch quality regressions.
- **Always record source-input md5 + binary md5** alongside any
  quality / perf claim (per CLAUDE.md prompt-shape rule, applied here
  to quant artifacts).
- **27B 3.6 MQ3 is the canonical fluent sub-4-bit reference.** When
  testing a new quant variant, compare against 27B 3.6 MQ3 outputs on
  the same 4 prompts.
- **Don't touch `is_batchable_la` to admit MQ3/MQ2 until Q5 lands.**
  The current per-token fallback is the reason zero panics surfaced;
  adding MQ3 to the batched path without a real WMMA kernel would
  reintroduce the HFQ4 batched-kernel-on-MQ3 hazard PR #108 closed.

## Out of scope for this queue

- Path E (TCQ / QTIP-style trellis codes) — PRD §5.5 says research-grade,
  pursue only if Q1-Q4 collectively fail. Empirically Q1+Q2 should
  unlock enough quality headroom that we don't need TCQ.
- Path F (BitNet-style native ternary) — requires a training stack
  hipfire doesn't have. Out of scope.

## Reproducing this sweep before any agent run

```bash
# Quantize the canonical sweep set (NAS HF cache → ~/.hipfire/models/)
hipfire-quantize --input <Qwen3.5-0.8B-snapshot> --output ~/.hipfire/models/qwen3.5-0.8b.mq3 --format mq3
hipfire-quantize --input <Qwen3.5-9B-snapshot>   --output ~/.hipfire/models/qwen3.5-9b.mq3   --format mq3
hipfire-quantize --input <Qwen3.5-27B-snapshot>  --output ~/.hipfire/models/qwen3.5-27b.mq3  --format mq3
hipfire-quantize --input <Qwen3.6-27B-snapshot>  --output ~/.hipfire/models/qwen3.6-27b.mq3  --format mq3
# (MQ2 needs --allow-mq2 per PR #109 guard)

./scripts/mq3-mq2-sweep.sh
# report → ~/.hipfire/mq3-tests/sweep-<timestamp>.md
```
