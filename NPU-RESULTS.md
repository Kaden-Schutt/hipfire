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
