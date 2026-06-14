# NPU Kernel Results

## Platform

- **Machine**: nix1
- **NPU**: NPU1 (XDNA / Ryzen AI, Phoenix silicon) — AIE2 / AIE-ML, 16 TOPS
- **Tile grid**: 4 compute columns × 4 core rows (column_width=4)
- **Driver**: amdxdna-dkms 7.0.0-rc1+git20260310.6b13cb8f4
- **XRT**: 2.25.0 (2026-06-01)
- **Date**: 2026-06-13

## SwiGLU (silu_mul_bf16) — Qwen3.5 dense FFN

Kernel: `tools/npu/silu_mul_bf16.cc`, AIE2 LUT-based tanh.
Computes `out[i] = silu(gate[i]) * up[i]` in BF16 across all 4 NPU columns.
Warmup: 20 iterations. Timed: 200 iterations.

| hidden_size | model  | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------------|--------|-----------|---------------|---------|---------|----------------|-----------|
| 8960        | 1.5B   | 52.5      | 201           | 198     | 302     | 247            | 0.27      |
| 18944       | 7B     | 111.0     | 216           | 209     | 359     | 262            | 0.53      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × hidden_size × 2 bytes / npu_mean.

### Observations

The ~190 µs floor is fixed dispatch/DMA overhead. The 7B size does 2.1× the
data in only 7% more time, confirming compute is not the bottleneck. The NPU
path is intended for pipelined use where this overhead is hidden by concurrent
GPU work, not for synchronous calls.

## Correctness (oracle test)

Reference: `silu(gate) * up` computed in float32, cast to bfloat16.  
Tolerance: atol=0.02, rtol=0.02. Max abs error ~0.047 (≈3 bfloat16 ULPs at 1.0,
consistent with LUT-based tanh rounding).

| hidden_size | max_abs_err | mean_abs_err | max_rel_err | result |
|-------------|-------------|--------------|-------------|--------|
| 8960        | 0.04688     | 0.00291      | 0.0234      | PASS   |
| 18944       | 0.04688     | 0.00290      | 0.0235      | PASS   |

## RMSNorm (rms_norm_weighted_bf16) — Qwen3.5 hidden norm

Kernel: `tools/npu/rms_norm_weighted_bf16.cc`, AIE2, single-tile full-row design.
Computes `out[i] = (x[i] / rms(x)) * weight[i]` in BF16.
Single-tile design: tile_size=hidden_size so the entire row lands in one AIE core for
the reduction pass. Uses `aie::invsqrt(vector<float, N>)` (hardware RSQRT).
Warmup: 20 iterations. Timed: 200 iterations.

| hidden_size | model | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------------|-------|-----------|---------------|---------|---------|----------------|-----------|
| 1536        | 1.5B  | 9.0       | 187           | 178     | 290     | 243            | 0.05      |
| 3584        | 7B    | 21.0      | 167           | 161     | 265     | 204            | 0.13      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × hidden_size × 2 bytes / npu_mean.

### Observations

The dispatch floor is ~160–190 µs, same as SwiGLU. BW is lower than SwiGLU because the
data footprint (3 × hidden_size ≈ 9–21 KB) is smaller than the FFN sizes. The single-tile
design uses only 1 of 4 NPU columns — this is unavoidable for a reduction operation where
all elements must be visible for the sum(x²) pass. Pipelined with GPU compute, the dispatch
overhead is hidden.

### Correctness (oracle test)

Reference: `(x / sqrt(mean(x²) + 1e-5)) * weight` in float32, cast to bfloat16.  
Tolerance: atol=0.02, rtol=0.02. Max abs error ~0.031 (≈2 bfloat16 ULPs at 1.0,
from bfloat16 broadcast of float inv_rms).

| hidden_size | max_abs_err | mean_abs_err | max_rel_err | result |
|-------------|-------------|--------------|-------------|--------|
| 1536        | 0.03125     | 0.00512      | 0.0193      | PASS   |
| 3584        | 0.03125     | 0.00471      | 0.0155      | PASS   |

## RoPE (rope_rotate_bf16) — Qwen3.5 rotary position embedding

Kernel: `tools/npu/rope_rotate_bf16.cc`, AIE2, single-tile half-split design.
Applies `x_rot = x*cos - y*sin, y_rot = y*cos + x*sin` in BF16 for dims [0, n_rot),
pass-through for dims [n_rot, head_dim). Half-split layout: x at [0, n_rot/2),
y at [n_rot/2, n_rot). Separate Q and K xclbins (tile_size=head_dim).
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, n_kv_heads=2,
head_dim=256, n_rot=64 (Qwen3.5-1.5B dense).

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 185           | 171     | 304     | 246            | 0.05      |
| K      | 2       | 512       | 1.0       | 166           | 161     | 255     | 211            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × total_elem × 2 bytes / wall_mean.

### Observations

Same ~160–190 µs dispatch floor as SwiGLU and RMSNorm. The Q tensor (8 heads × 256 = 2048 elements, 4 KiB) and K tensor (2 heads × 256 = 512 elements, 1 KiB) are both small; BW is dispatch-floor-dominated. The cs param (64 elements, 128 B) is acquired once per dispatch and reused for all head iterations, avoiding 8× or 2× redundant transfers. Single-tile design (one AIE column) since all heads are processed serially.

### Correctness (oracle test)

Reference: half-split float32 RoPE with `freq_base=500000` (Qwen3.5 theta), `pos=1`.
Tolerance: atol=0.02, rtol=0.02. max_abs ≤ 1 bfloat16 ULP (0.03125 at magnitude 1–2).

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.03125     | 0.00064      | 0.1034      | PASS   |
| K      | 0.01562     | 0.00057      | 0.0667      | PASS   |

## QK Head Norm (rms_norm_head_bf16) — per-head RMSNorm on Q and K

Kernel: `tools/npu/rms_norm_head_bf16.cc`, AIE2, single-tile design.
Applies `out[h][i] = (x[h][i] / rms(x[h])) * weight[i]` per head in BF16.
Shared weight `[head_dim]` is a tensor param acquired once for all head iterations.
Mirrors `gpu.rmsnorm_batched()` in the Qwen3.5 forward pass (runs after QKV projection, before RoPE).
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, n_kv_heads=2, head_dim=256.

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 187           | 177     | 291     | 235            | 0.05      |
| K      | 2       | 512       | 1.0       | 175           | 169     | 259     | 218            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × total_elem × 2 bytes / wall_mean.

### Observations

Same ~170–190 µs dispatch floor. The weight tensor param pattern (acquired once, reused across 8 or 2 head iterations) amortizes the weight DMA cost rather than paying it per head. Single-tile design — each head requires the full vector for the reduction pass.

### Correctness (oracle test)

Reference: per-head float32 RMSNorm, `eps=1e-5`, random weight in [0.5, 1.5].  
Tolerance: atol=0.02, rtol=0.02.

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.04688     | 0.00448      | 0.0199      | PASS   |
| K      | 0.06250     | 0.00449      | 0.0148      | PASS   |

## Attn Output Gate (sigmoid_mul_bf16) — Qwen3.5 attention gating

Kernel: `tools/npu/sigmoid_mul_bf16.cc`, AIE2 LUT-based tanh.
Computes `out[i] = sigmoid(gate[i]) * x[i]` across all 4 NPU columns.
Replaces `gpu.sigmoid_f32 + gpu.mul_f32` when `config.attn_output_gate=true`.
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, head_dim=256, q_dim=2048.

| q_dim | n_heads | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------|---------|-----------|---------------|---------|---------|----------------|-----------|
| 2048  | 8       | 4.0       | 183           | 176     | 268     | 227            | 0.05      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 × q_dim × 2 bytes / wall_mean.

### Observations

Same ~183 µs dispatch floor. Uses 4 NPU columns (parallel), same as SwiGLU. Data footprint (3 × 4 KiB = 12 KiB) is small; BW-dominated by dispatch. One kernel per decode step (only present when `attn_output_gate=true`).

### Correctness (oracle test)

Reference: `sigmoid(gate) * x` in float32 where `sigmoid(x) = 1/(1+exp(-x))`.  
Tolerance: atol=0.02, rtol=0.02. max_abs=0.016 (≈1 ULP at magnitude ≤1, from LUT tanh rounding).

| q_dim | max_abs_err | mean_abs_err | max_rel_err | result |
|-------|-------------|--------------|-------------|--------|
| 2048  | 0.01562     | 0.00211      | 0.0216      | PASS   |
