# AWQ Bug Hunt — Root Cause Found

**Date:** 2026-05-12
**Status:** Bug confirmed — quantizer applies AWQ pre-scaling to weights whose runtime path has no AWQ inverse.

---

## The Bug

The quantizer applies `W' = W · diag(s)` to **every** MQ4G256 tensor that has imatrix data. But the runtime only applies the inverse `x /= s` in the `fused_rmsnorm_rotate_mq_awq` kernel path. Two sets of weights are AWQ-pre-scaled at quantize time but **never compensated** at runtime:

### `wo` (o_proj) — consumed via `weight_gemv_residual`

```
llama.rs:877  (DType::MQ4G256 arm):
    rotate_x_mq(x, &x_rot_alias, w.k)?;
    gemv_hfq4g256_residual(&w.buf, &x_rot_alias, y, w.m, w.k)
```

`rotate_x_mq` does FWHT rotation only — **no AWQ divide**. The weights have `W·diag(s)` baked in, but the activations are not divided by `s`. Result: computes `(W·s) · x ≠ W·x`.

### `w_down` (down_proj) — consumed via `weight_gemv_swiglu_residual`

```
llama.rs:959  (DType::MQ4G256 arm):
    fused_silu_mul_rotate_mq(gate, up, &x_rot_alias, w_down.k)?;
    gemv_hfq4g256_residual(&w_down.buf, &x_rot_alias, x, w_down.m, w_down.k)
```

`fused_silu_mul_rotate_mq` does SiLU(gate)*up + FWHT rotation — **no AWQ divide**. Same corruption: scaled weights meet unscaled activations.

### Every layer compounds the error

Every layer's output passes through both `wo` and `w_down`. Error accumulates across all 32 layers → garbage output.

---

## Why α=0 passes (confirming the diagnosis)

| Configuration | KLD | Why |
|---|---|---|
| mq4-base (no AWQ) | 0.6721 | Baseline |
| mq4-awq, α=0 (s[j]=1.0 ∀j) | 0.6721 | AWQ kernel fires but divides by 1.0 (no-op). `wo`/`w_down` also have s=1.0 baked in (no-op). Identical to baseline. |
| mq4-awq, α=0.5 (real scales) | 13.4893 | Non-trivial s ∈ [0.76, 3.9] baked into `wo`/`w_down` with no inverse → catastrophic. |

This is the smoking gun: the bug is **invisible at s=1.0** (which is why α=0 matches baseline exactly) and **catastrophic at any non-trivial s** (which is why α=0.5 blows up to 13 nats).

---

## The four correctly-handled weights (for contrast)

These are preceded by `fused_rmsnorm_rotate_for_mq` or `fused_rmsnorm_rotate_mq_batched_for`, which check `awq_scale` and dispatch to the AWQ kernel that divides by s before the FWHT:

| Weight | Runtime path | AWQ inverse applied? |
|---|---|---|
| q_proj, k_proj, v_proj (or wqkv) | `fused_rmsnorm_rotate_for_mq` → AWQ kernel | ✓ Yes |
| gate_proj, up_proj | `fused_rmsnorm_rotate_for_mq` → AWQ kernel | ✓ Yes |
| **o_proj (wo)** | `weight_gemv_residual` → `rotate_x_mq` | ✗ **No** |
| **down_proj (w_down)** | `weight_gemv_swiglu_residual` → `fused_silu_mul_rotate_mq` | ✗ **No** |

---

## FWHT normalization — not the issue

For completeness: the quantizer's `cpu_fwht_256` and the GPU kernel's FWHT use identical normalization (`1/16 = 0.0625 = 1/√256`) and identical sign tables (`gen_fwht_signs` with same seeds 42 and 1042). The α=0=baseline match at 4 decimal places confirms there's no FWHT normalization discrepancy. The bug is purely about missing AWQ inverse on `wo`/`w_down`.

---

## Fix

### Option A: Quantizer-side filter (recommended, minimal change)

Skip AWQ pre-scaling (and sidecar emission) for tensors whose runtime path lacks AWQ inverse. In `hipfire-quantize/src/main.rs`, around line 4018, add a guard before the AWQ block:

```rust
// Only AWQ-pre-scale weights whose runtime path will apply the inverse.
// o_proj / down_proj go through rotate_x_mq / fused_silu_mul_rotate_mq
// which have no AWQ support — pre-scaling them corrupts the output.
let awq_eligible = !name.contains("o_proj") && !name.contains("down_proj");

let q = if awq_eligible
    && let (Some(alpha), Some(im_weights)) = (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
{
    // existing AWQ path unchanged
    let scales = compute_awq_scales(im_weights, alpha);
    awq_sidecar_scales = Some(scales.clone());
    let m_dim = meta.shape[0];
    let mut scaled = f32_data.clone();
    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
    quantize_mq4g256(&scaled, &signs1, &signs2)
} else {
    quantize_mq4g256(&f32_data, &signs1, &signs2)
};
```

After this change, re-quantize and re-run the cohort. Expected: KLD should drop from 13.49 back to ~0.67 (near baseline), possibly slightly better if AWQ helps the eligible projections.

### Option B: Runtime-side AWQ for all paths (more invasive, future work)

Add AWQ-aware variants of `rotate_x_mq` and `fused_silu_mul_rotate_mq` that accept an optional `awq_scale` parameter and divide before the FWHT. This would allow AWQ to benefit `wo` and `w_down` too, but requires new kernel variants and dispatch wiring. Not worth it for the first ship — the AWQ paper's dominant benefit is on the attention-input and FFN-input projections.

---

## Repro steps

```bash
# Quantize with AWQ at α=0.5 (broken):
hipfire-quantize model.safetensors --format mq4 --imatrix imatrix.bin --awq-alpha 0.5 -o model-awq.hfq

# Quantize baseline (no AWQ, fine):
hipfire-quantize model.safetensors --format mq4 --imatrix imatrix.bin -o model-base.hfq

# Compare via eval_hipfire or cohort — expect ~20× KLD blowup on awq variant.
```

---

## Files involved

| File | Role |
|---|---|
| `crates/hipfire-quantize/src/main.rs:4018` | Bug origin — AWQ applied to all MQ4G256 tensors unconditionally |
| `crates/hipfire-runtime/src/llama.rs:877` | `wo` MQ4G256 path — `rotate_x_mq` (no AWQ) |
| `crates/hipfire-runtime/src/llama.rs:959` | `w_down` MQ4G256 path — `fused_silu_mul_rotate_mq` (no AWQ) |
| `crates/hipfire-runtime/src/llama.rs:722` | Correctly-handled decode path — checks `awq_scale` |
| `crates/hipfire-runtime/src/llama.rs:696` | Correctly-handled batched path — checks `awq_scale` |
| `crates/rdna-compute/src/dispatch.rs:3641` | `rotate_x_mq` — no AWQ parameter |
| `crates/rdna-compute/src/dispatch.rs:3515` | `fused_silu_mul_rotate_mq` — no AWQ parameter |
| `crates/rdna-compute/src/dispatch.rs:3316` | `fused_rmsnorm_rotate_mq_awq` — has AWQ parameter (correct) |
