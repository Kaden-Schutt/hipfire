# vLLM AWQ → hipfire integration findings

**Date:** 2026-05-12
**Companion to:** `/home/kread/mygit/vllm/vllm_awq.md` (deep-dive analysis of vLLM + compressed-tensors AWQ; treat that as the upstream reference). This doc is the **hipfire-specific translation** — what changes vs vLLM's design when targeting MQ4 + FWHT-256, what stays the same, and what's open.

**Status:** Research / design synthesis. Companion doc is "lightly validated"; spot-checks below confirm key claims. This doc is the action plan derived from those findings.

---

## 1. What's reusable from vLLM's design vs what needs rethinking

vLLM's AWQ implementation is **forward-pass-only** — scales are pre-computed offline by AutoAWQ and baked into the safetensor checkpoint as `(qweight, qzeros, scales)` triples. The α-grid-search calibration lives entirely in AutoAWQ / llm-compressor. **vLLM never invokes alpha search.**

For hipfire, this is the right architectural split: keep the alpha tuning in `hipfire-quantize` (one-shot offline), keep the runtime path minimal. But our offline path differs from AutoAWQ's in three ways that matter:

| dimension | AutoAWQ (vLLM ingests) | hipfire-quantize |
|---|---|---|
| Calibration data | Raw activation max per channel (`max\|X_j\|`) over ~128 calib seqs | We have `Σ_token act²[j]` from `imatrix_collect.rs` (llama.cpp `--imatrix`) |
| Alpha search | 20-step grid over [0, 1], per-layer | Not implemented; spec needed |
| Quantizer entry point | Per-linear-layer scaling + per-group min-max | Existing `quantize_mq4g256` + FWHT-256 |
| Wire format storage | safetensor `qweight/qzeros/scales` tensors | hipfire `.hfq` per-tensor metadata + sidecar 1D F16 tensor |

The hipfire-specific piece is **what AWQ formula to ship given we have `Σ act²` instead of `max\|X\|`**.

## 2. Hipfire AWQ scale formula — derived from imatrix data

AutoAWQ uses `max|X_j|`. Hipfire's `imatrix_collect` (Step 4 Tier 2) dumps `Σ_token act²[j]`. These are related but not equivalent:

- `max|X_j|` is dominated by outlier tokens; sensitive to corpus shape.
- `RMS(X_j) = sqrt(Σ act²[j] / n_tokens)` is the per-channel L2 statistic; smoother estimator, equal to `max|X_j|` × constant if `|X_j|` is approximately uniform but lower for heavy-tailed distributions.

For a Gaussian-ish channel: `max|X_j| ≈ 3-4 × RMS(X_j)`. The constant factor cancels in the AWQ formula because alpha tuning absorbs it. So **using `RMS(X_j)` in place of `max|X_j|` is mathematically valid — the optimal alpha shifts but the optimum is reachable.**

Concrete formula for hipfire AWQ:

```
RMS_act[j]   = sqrt(in_sum2[j] / n_tokens)    # n_tokens recoverable from
                                              # imatrix.counts entries
RMS_w[j]     = sqrt((1/m) * Σ_i W[i,j]²)      # per-input-channel weight RMS
                                              # (m = output dim of this tensor)

# AWQ-paper variant (original LLM-AWQ form):
s_raw[j]     = (RMS_act[j])^α / (RMS_w[j])^α

# OR AutoAWQ variant (algebraically equivalent up to alpha reparameterization):
s_raw[j]     = (RMS_act[j])^α * (RMS_w[j])^(1-α)

# Either way, normalize to keep geometric mean = 1 so the scaled-weight
# magnitudes stay in a similar range (important for the post-scaling
# per-group min-max quantizer):
log_geo_mean = (1/k) * Σ_j log(s_raw[j])
s[j]         = s_raw[j] / exp(log_geo_mean)
```

The geometric-mean normalization prevents the post-AWQ weights from ballooning or shrinking systematically — keeps the per-group `(scale, ZP)` quantizer happy.

**Implementation note**: AutoAWQ uses `max|X_j|`, AutoAWQ does NOT use `(RMS_w[j])^(1-α)` (it uses just `RMS_w` or its variant). For the first hipfire ship, use the simpler form `s[j] = (RMS_act[j])^α` (no weight-magnitude term), normalize to geo-mean=1, and tune α. The weight-magnitude term in the formula is small-effect; can be added later if grid search shows it helps.

## 3. Alpha tuning — hipfire approach

AutoAWQ does **per-linear-layer** alpha grid search over 20 values in [0, 1]. Per-layer search means: for each linear layer, try each α, simulate quantize+dequantize, compute reconstruction error against full-precision matmul, pick the α that minimizes it.

For hipfire's first ship, this is too much engineering. **Use a global α (single value, e.g., 0.5) for all tensors**. Empirically AWQ-paper-original showed α ≈ 0.5 is a strong default. Add per-layer search as a Phase B improvement once the global-α version is benched.

**CLI surface for hipfire-quantize:**

```
--awq                    # enable AWQ at default alpha (0.5), requires --imatrix
--awq-alpha <f>          # explicit alpha (overrides default); 0=off, 1=pure act
--awq-formula auto|paper|autoawq    # which formula variant (default: paper)
```

`--awq` without `--imatrix` errors out with a clear message.

## 4. Rotation interaction (the hipfire-specific lever)

vLLM doesn't ship AWQ + rotation, but the architecture pattern is `CompressedTensorsLinearTransformMethod` (`linear.py:155-165`): `x → input_transform → quant_method.apply → output_transform`. The transform is **completely separate from the quantization method.**

For hipfire's MQ4 (FWHT-256 baked into weights at quantize time), the math composition is:

```
Offline (quantize time):
  1. Compute AWQ scales s[j] from imatrix data (rotated or unrotated?)
  2. Pre-scale: W' = W · diag(s)
  3. Apply FWHT-256 to W' rows over input dim: W'' = W' · H^T
  4. Quantize W'' as standard MQ4 with min-max scale per 256-group

Runtime (inference):
  1. RMSNorm: x_norm = norm(x) * γ
  2. AWQ divide: x_awq = x_norm / s             ← NEW step
  3. FWHT: x_rot = H · x_awq                    ← existing kernel
  4. Quantized MQ4 GEMM: y = MQ4_gemm(W''_q, x_rot)
```

The composition works because FWHT and per-channel scaling don't commute trivially (FWHT applied to `x/s` ≠ FWHT(x) / s), but **the math cancels at the GEMM step** since we apply the AWQ scale on x BEFORE the FWHT — same as we baked it on W BEFORE the FWHT. Detailed derivation in `mq4v2-format-proposal.md` §6.5 (or in the current `qwen35-mq4-quality-gap.md` §5 Phase A revised).

**Critical question: should the AWQ scale be computed on the unrotated activation statistics, or on rotated activation statistics?**

The imatrix is captured against unrotated activations (it's just llama.cpp's forward pass on the BF16 model). For AWQ-on-MQ4-with-FWHT, we want `s` to reflect the importance of channels in the **runtime** activation flow — which means unrotated activations (because we apply `x/s` before the FWHT in step 2 above). So **use unrotated imatrix directly**. Matches AutoAWQ's behavior (they don't apply rotation at calibration).

This is **opposite** to what we tried in Step 5a-prime for MFP4 weighted-LS (we attempted to FWHT the imatrix and got nonsense from the sign bug). For AWQ, the divide happens in the unrotated basis on the inference side, so the unrotated imatrix is correct.

## 5. Wire format choice for hipfire

vLLM stores `qweight`, `qzeros`, `scales` as separate safetensor entries. Compressed-tensors stores `weight_packed`, `weight_scale`, `weight_zero_point` with a slightly different convention. Hipfire's `.hfq` format is its own thing — but the same "scale is a separate tensor" pattern transfers.

**Recommended hipfire AWQ wire format**:

- Existing `MQ4G256` quant data unchanged (still per-256 INT4 + FP32 scale + FP32 ZP)
- New **sidecar 1D F16 tensor** per AWQ-enabled weight, named `<weight_name>.awq_scale`
  - For `model.layers.0.self_attn.q_proj.weight` → `model.layers.0.self_attn.q_proj.awq_scale`
  - Shape: `[K]` (length = input dim of the parent weight)
  - Dtype: F16 (matches scale precision; ~256 KB total for 9B)
- Runtime loader: when loading a weight tensor, look up the companion `.awq_scale`; pass to the forward path

**Why sidecar tensor rather than embed in the per-block layout**: keeps MQ4 wire format unchanged (no kernel changes for non-AWQ users), composable with HFP4/MFP4 if we want AWQ there too in future, easy for `--help` output to enumerate. Cost is a one-extra lookup at model load time, negligible.

## 6. Kernel-side application — where the divide happens

vLLM applies AWQ scales inside the GEMM (Path A: fused into the inner loop; Path B: dequantize-then-matmul; Path C: Marlin's `scale_and_sub`). vLLM specifically does **NOT** fold AWQ scaling into rmsnorm/layernorm.

For hipfire's MQ4 with FWHT, the geometry is different. The AWQ divide must happen **before** the FWHT — and FWHT lives inside `fused_rmsnorm_rotate_mq` (a fused kernel). So the cleanest hipfire integration is:

- **Extend `fused_rmsnorm_rotate_mq`** to optionally take a per-channel divide vector. When AWQ scale is loaded for the next linear, pass it; the kernel does `(rmsnorm_output / s) → FWHT → output`.
- **Backward-compatible**: when no AWQ scale, behaves identically to today.

This is a small kernel mod, not a new kernel family. The fused-rmsnorm-rotate path is fp16/bf16; adding an elementwise divide by a per-channel f16 vector is `~3-4 fp16 ops` per element, negligible vs the existing rmsnorm + FWHT cost.

For HFP4G32 / MFP4G32 (different forward path — no fused rmsnorm rotate), the divide would go into a different fused kernel. Out of scope for Stage A (which targets MQ4).

## 7. Concrete implementation plan for hipfire AWQ on MQ4

**Phase 1 — Quantizer-only patch (~3-5 days, no GPU needed)**

1. `crates/hipfire-quantize/src/main.rs`:
   - Add `--awq [<alpha>]` CLI flag (default α=0.5, requires `--imatrix`)
   - Add `awq_compute_scales(imatrix: &[f32], weights: &[f32], m, k, alpha) -> Vec<f32>`
     - Computes `RMS_act = sqrt(in_sum2[j] / n_tokens)`
     - Computes `s[j] = (RMS_act[j])^α`
     - Normalizes to geo-mean=1
   - Modify `quantize_mq4g256` (or add an AWQ-aware wrapper) to accept optional scales and apply `W' = W · diag(s)` before FWHT
   - Emit the per-tensor `.awq_scale` F16 1D tensor alongside the weight tensor in `.hfq`

2. Tests (`#[cfg(test)]`):
   - Round-trip: AWQ-scale then dequant then AWQ-unscale → recovers W within MQ4 quantization error
   - Math identity: for a random W and x, `(W·s) · (x/s) ≈ W·x` within FP16 precision
   - The unmodified path (no `--awq`) produces byte-identical output to pre-Stage-A code

**Phase 2 — Runtime path (~1-1.5 weeks)**

3. `crates/hipfire-runtime/src/hfq.rs`:
   - Loader reads `.awq_scale` sidecar tensors at model open
   - Stores in `LinearLayer` struct (alongside W, scale, ZP)

4. `crates/hipfire-arch-qwen35/src/qwen35.rs` (or wherever fused_rmsnorm_rotate_mq is called):
   - Pass `Option<&AwqScale>` to the fused kernel calls
   - Linear layers WITH AWQ scale use the new path; WITHOUT, fall back to existing kernel

5. Kernel mod (kernels/src/`fused_rmsnorm_rotate_mq.gfx11.hip` etc):
   - Add optional per-channel divide before FWHT
   - Maintain backward compatibility (when scale=nullptr, behaves identically)

**Phase 3 — Bench (~3-5 days)**

6. Re-quantize 9B + 0.8B with `--awq` enabled. Compare per-tensor MSE — expect slight increase (AWQ optimizes downstream output, not per-tensor reconstruction).
7. Run cohort on each variant. Compare KLD/PPL vs uncalibrated MQ4. **Target: ≥15% PPL improvement** (literature baseline for AWQ on Q4 INT4 quants).
8. Stack with Step 6b UD-kmap → measure additivity.

**Total: ~2-2.5 weeks** to validate AWQ on MQ4 end-to-end. Most of the work is the runtime kernel mod (Phase 2 step 5); the quantizer-side logic (Phase 1) is straightforward.

## 8. Per-layer alpha search — Phase B improvement, not blocking

AutoAWQ's 20-step per-layer alpha grid search adds significant offline cost (per layer: 20 × forward-pass-on-calibration-data ≈ 20× the cost of computing a single scale). For hipfire's first ship, global α = 0.5 is good enough. Per-layer search becomes a Phase B improvement when we have:
- Bench infrastructure proving global α=0.5 leaves quality on the table
- A faster forward-pass-simulation harness (could reuse the imatrix collector's forward path)

Estimated cost of per-layer search: ~1 week dev + ~1 hour wall to compute alphas for a 9B model.

## 9. What we still don't know

Open questions that should be resolved by Stage A's bench (or Stage 0's Q8 floor cohort):

1. **AWQ + FWHT-256 interaction at small group sizes** — the MR-GPTQ paper's "small-group neutralizes outlier mitigation" finding applies to MXFP4 g=32. AWQ on MQ4 g=256 should escape this — but until benched on Qwen3.5-9B specifically, we don't know if hipfire's specific FWHT-256 + INT4 + g=256 combination has its own gotcha.

2. **RMS-vs-max for the activation statistic** — using `RMS(X)` (from imatrix `Σ act²`) instead of `max|X|` (from AutoAWQ) might shift the optimal alpha by a constant factor. Could matter if alpha=0.5 happens to be a degenerate point for RMS-based scaling.

3. **Whether to also compute `RMS_w` term** — the AWQ-paper-original formula `(RMS_act)^α / (RMS_w)^α` is fully equivalent to `(RMS_act)^α * (RMS_w)^(-α)`. The weight-magnitude term is small-effect; first hipfire ship without it; add if grid search shows it helps.

4. **Tokenizer parity floor** — every measurement we make on hipfire is contaminated by the ~46% tokenizer disagreement with llama.cpp. The Q8 baseline cohort (Stage 0) measures the floor; AWQ should close more of the gap above the floor, not below it.

## 10. References (vLLM + compressed-tensors, spot-checked 2026-05-12)

| File | Verified claim |
|---|---|
| `vllm/model_executor/layers/quantization/awq.py:88-95` | `AWQConfig.from_config()` reads `w_bit`, `q_group_size`, `zero_point` — no `transform_config` or alpha tuning |
| `vllm/model_executor/layers/quantization/awq.py:262-289` | `AWQLinearMethod.apply()` dispatches to `awq_gemm` (tokens<256) or `awq_dequantize + torch.matmul` (tokens≥256) — pure inference, no calibration |
| `vllm/model_executor/layers/quantization/utils/quant_utils.py:794-813` | `awq_pack()` uses interleave `[0, 2, 4, 6, 1, 3, 5, 7]` for 4-bit packing |
| `vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py:39-165` | `CompressedTensorsLinearTransformMethod` wraps any `LinearMethodBase` with `input_transform` + `output_transform`; calls them around `quant_method.apply()` |
| `vllm/model_executor/layers/quantization/awq_marlin.py` (no transform_config reads) | Confirms AWQ standalone path does NOT consume `transform_config` |
| `csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu` | Sylvester FHT kernel (recursive butterflies, no transform-weight needed); power-of-2 sizes 2-2^15; in-place fp16/bf16 |

The companion file `/home/kread/mygit/vllm/vllm_awq.md` has full code-line references; this doc is the hipfire-specific synthesis.

---

## Summary

- vLLM's AWQ is forward-pass-only — its design is the right architectural split for hipfire too.
- The α-grid search lives offline; hipfire-quantize gets a `--awq [α]` flag, default α=0.5.
- Use imatrix's `Σ act²` → derive `RMS_act` → AWQ scale formula. Math equivalent to AutoAWQ's `max|X|` up to an absorbed constant.
- Store AWQ scales as a sidecar 1D F16 tensor per weight (`<weight>.awq_scale`). No `.hfq` wire format change.
- At inference: divide activations by `s` before FWHT, inside `fused_rmsnorm_rotate_mq`. Small kernel mod, backward-compatible.
- Total implementation budget: ~2-2.5 weeks (quantizer + runtime + kernel mod + bench).
- Skip per-layer alpha search for first ship; global α=0.5; per-layer search as Phase B if needed.
