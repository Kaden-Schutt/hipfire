# NPU Kernel Results

## Strix Halo NPU (aie2p) — the ceiling is FEEDING, not compute

> **halo** box (RYZEN AI MAX+ 395, NPU Strix Halo **aie2p/npu2**, 4 compute rows
> × 8 cols = 32 cores). Investigation: 2026-07-04. Full detail + reproducible
> harness in `benchmarks/npu_gemm_tuning/` (`findings.md`, `tune.sh`).

**The Strix Halo NPU is a real, un-throttled 58 TOPS int8 — but the hard part is
feeding the cores, not the cores.** Evidence:

- **Hardware peak = 58 TOPS** (`hipfire-xdna` resource_info: `npu_tops_max=58`,
  `npu_clk_max=1800`). First-principles: 32 cores × 512 int8 MAC/mmul (8×8×8) ×
  1.8 GHz ≈ 59 TMAC. `crates/hipfire-xdna/examples/npu_info.rs` dumps it live.
- **NOT power/clock-throttled.** Under GEMM load, `default` pmode already boosts
  to the full 58-TOPS budget with the AIE compute clock maxed at 1800 MHz.
  `xrt-smi configure --pmode turbo` is a **confirmed no-op** (15.2 vs 15.7 TOPS).
- **Real GEMM caps at ~12–27% of peak — it's feed/overhead-bound.** A tuned int8
  matmul (mlir-aie `whole_array`) tops out at **15.7 TOPS (27%)**; every knob
  explored (output-tile size is the only lever and it's L1-capped at 64 KB/core;
  columns maxed; k/fifo_depth/OPT_PERF/pre-tiled-weights all no-op or marginal;
  the `mm.cc` microkernel is AMD-optimal). Throughput scales with **output-tile
  size** because it amortizes per-tile feed/sync overhead (DMA setup, C
  accumulator load/store, objectFIFO acquire/release, software-pipeline
  fill/drain) — the cores are **starved**, not saturated.
- **AMD's own shipped kernel confirms it.** Built DynamicDispatch from source and
  ran the production `mladf` int4 gemm on the NPU: flat **~7 TOPS** across LLM
  shapes (a memory-bound weight-quant *decode* kernel) — *below* our int8
  reference. No measured real GEMM — reference or production — approaches 58.

**Takeaway:** treat 58 TOPS as a theoretical ceiling; budget **~15–16 TOPS int8**
for a real Strix Halo GEMM. The lever for more is a better *dataflow* that keeps
the cores fed (larger effective tiles / less per-tile overhead), not compute,
power, columns, or datatype. Bottleneck = feeding the AIE array.

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

## Kernel 7: Softmax (`softmax_bf16`)

- **Date**: 2026-06-14
- **Status**: PASS (all ctx_len variants)
- **Config**: n_heads=8, Qwen3.5-1.5B dense, NPU1 (Phoenix/AIE2)
- **Algorithm**: 3-pass scalar poly-exp (from mlir-aie bf16_softmax.cc), + max-subtraction pre-pass
- **Exp method**: range-reduction `2^trunc(x*log2e) * 2^frac(x*log2e)`, degree-2 poly for fractional; clamped for underflow (`ix < -127 → 0`). Handles masked -inf positions correctly (produce exactly 0).
- **Note on LUT exp**: `getExpBf16` from lut_based_ops.h only handles positive inputs (LUT covers [0, 7.97]); negative inputs (all values after max-subtraction) give `exp(7.97)` via truncation — unusable here.

| ctx_len | max_abs  | npu_mean | wall_mean |
|---------|----------|----------|-----------|
| 64      | 0.00391  | 608 µs   | 672 µs    |
| 128     | 0.00269  | 1343 µs  | 1486 µs   |
| 256     | 0.00122  | 1881 µs  | 1953 µs   |
| 512     | 0.00073  | 3860 µs  | 4134 µs   |

**Dispatch floor: ~170 µs; compute scales ~7.4 µs/element (scalar loop).** Softmax is compute-heavy (exp + sum + normalize) and the scalar implementation dominates. Not NPU-competitive with GPU DFlash for typical decode contexts (DFlash fuses QK^T + softmax + AV in one kernel). Useful only for the non-DFlash fallback paths (Q8 FA, gqa4) at short context lengths (≤128) where standalone GPU softmax dispatch overhead dominates.

## Kernel 8: Fused HeadNorm + RoPE (`headnorm_rope_bf16`)

- **Date**: 2026-06-14 (built; awaiting hardware run)
- **Status**: pending first run
- **Config**: n_heads=8, n_kv_heads=2, head_dim=256, Qwen3.5-1.5B dense, NPU1 (Phoenix/AIE2)
- **Algorithm**: 3-pass vectorized (VEC=16):
  - Pass 1: `sum_sq = Σ x_i²`, `inv_rms = 1/sqrt(mean_sq + 1e-5)` via `aie::invsqrt`
  - Pass 2 (rotation region [0, n_rot)): normalize (x * inv_rms * weight) then rope-rotate (half-split layout)
  - Pass 3 (passthrough [n_rot, head_dim)): normalize only
- **Tensor param**: single packed buffer `[weight (head_dim), cs (n_rot)]` — avoids a new FFI signature
- **Dispatch savings**: replaces 4 separate dispatches (headnorm_q + rope_q + headnorm_k + rope_k)
  with 2 dispatches (Q and K), saving 2 × ~170 µs × 28 layers ≈ **9.5 ms/step**
- **Artifact**: `qwen35-headnorm-rope-{q,k}-{n_heads}h{head_dim}d.{xclbin,instr.bin}`

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 189           | 180     | 287     | 236            | 0.03      |
| K      | 2       | 512       | 1.0       | 176           | 161     | 297     | 227            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).
**wall time**: end-to-end per-call latency measured on the host.
**BW**: effective memory bandwidth = 2 tensors × total_elem × 2 bytes / wall_mean.

### Correctness (oracle test)

Reference: per-head float32 RMSNorm + half-split RoPE applied to normalized output.
Tolerance: atol=0.02, rtol=0.02.

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.03125     | 0.00544      | 3.3603*     | PASS   |
| K      | 0.03125     | 0.00574      | 0.0243      | PASS   |

*max_rel=3.36 on Q is a near-zero element (numerator ~0.03, denominator ~0.009) — passes absolute tolerance; consistent with prior headnorm and rope results individually.

### Strategic note

This kernel is Stage 1 of the MLIR-AIE migration roadmap approved 2026-06-14.
The remaining 2-dispatch reduction (Q and K in a single dispatch) requires raw
MLIR-AIE to assign different weight tensors to separate tile columns; that is
Stage 2 (single pre-attention dispatch per layer).

## Inference Integration — Qwen3.5-0.8B SwiGLU NPU Bench

- **Date**: 2026-06-14
- **Model**: `qwen3.5-0.8b.mq4.hfq` (MQ4 quantized, 0.50 GiB HFQ payload)
- **xclbin**: `qwen35-swiglu-3584.xclbin` (hidden_size=3584, intermediate FFN dim)
- **Activation**: `HIPFIRE_QWEN35_FFN_BF16=xdna1 HIPFIRE_QWEN35_FFN_BF16_LAYER=all`
- **Path**: `forward_scratch_layers` → `weight_gemv_swiglu_residual_bf16_probe` → xdna1

### Changes needed to wire xdna1 on MQ4 models

1. **Load bypass**: `load_bf16_down_shadow_for` returned error for non-BF16 tensors when `FFN_BF16=xdna1`.
   Fixed: early-return `Ok(None)` for xdna1 mode — the down GEMV uses the original MQ4 tensor on GPU, the shadow w_down data is never needed.

2. **Forward-pass bypass**: `weight_gemv_swiglu_residual_bf16_probe` gated on shadow presence.
   Fixed: dispatch to xdna1 path before the shadow guard, using `w_down.k` for hidden_size.

3. **XRT session limit**: creating one XRT handle per layer_idx (24 total) hits the NPU context limit on NPU1, crashing with `free(): invalid pointer` after the 2nd handle.
   Fixed: all layers with the same hidden_size share one handle (cache key = hidden_size).

### Results

| mode                         | decode tok/s | ms/tok | wall tok/s |
|------------------------------|-------------|--------|------------|
| GPU only (baseline)          | 60.8        | 16.45  | 59.9       |
| SwiGLU NPU, layer=0 only     | 60.6        | 16.50  | 59.7       |
| SwiGLU NPU, all 24 layers    | 59.3        | 16.85  | 54.3       |

**The NPU SwiGLU path is ~2.4% slower than GPU-only on the 0.8B model.**

### Analysis

The 24 × ~180 µs dispatch floor = ~4.3 ms extra per token (serial host→NPU→GPU→NPU...).
The GPU SwiGLU for 24 layers takes ~0.3 ms total (tiny elementwise op on RDNA).
Net: +4.0 ms/token = -1.5 tok/s.

NPU SwiGLU only makes sense when:
- GPU is fully compute-saturated by GEMVs and the elementwise SwiGLU contends for waves
- Dispatches overlap with concurrent GPU work (not currently the case — dispatch is synchronous)
- On larger models where GEMV latency makes the ~4 ms dispatch overhead small relative to step time

For the 0.8B model (16 ms/tok baseline), the dispatch overhead is 25% of step time — not viable.
For a 7B model (~100 ms/tok), the same 4 ms would be ~4% — marginal.
Real NPU benefit requires async NPU dispatch or hardware-level DMA overlap.

---

## Performance Tuning — Qwen3.5-0.8B Decode (2026-06-14)

Systematic A/B benchmarks to identify easy decode gains. Baseline: FP32 state (previous session, run.rs).

Bench tool: `bench_qwen35_speed`, MQ4, gfx1103, `--gen 80 --warmup 5 --prefill 64`.

### DeltaNet State Quantization

| State quant | tok/s (gen) | vs Q8 |
|-------------|-------------|-------|
| Q8 (daemon default) | 62.2 | — |
| FP32 | 58.9 | -5.4% |
| Q4 | 58.7 | -5.6% |

**Q8 is optimal.** FP32 wastes 18 × 1MB = 18MB/step of bandwidth (read+write per DeltaNet layer).
Q4 is slower than Q8 — requant overhead outweighs bandwidth savings at this scale.

### KV Cache Mode

| KV mode | tok/s (gen) |
|---------|-------------|
| q8 | 62.0 |
| asym3 | 62.2 |
| asym4 | 61.1 |

Negligible on 0.8B — only 6 FullAttention layers, 2 KV heads, tiny KV footprint.

### Weight Format

| Format | tok/s |
|--------|-------|
| MQ4 | 62.2 |
| MQ6 | 51.2 |

MQ4 already correct choice. MQ6 = -18% (higher dequant cost per weight).

### hipGraph

Hardcoded `let use_graph = false;` at qwen35.rs:8331 (disabled 2026-05-15, token-0 attractor on gfx11+ROCm 7.x). `HIPFIRE_GRAPH=1` is a no-op on this branch. Would be +0.6–0.7% per prior measurements if re-enabled.

### Takeaway

The daemon already uses Q8 state by default. The 60.8 tok/s in the previous session was from `run.rs` which forced FP32 state. Real decode ceiling with current code: **~62 tok/s on gfx1103 MQ4**. Further gains require re-enabling hipGraph (needs bug investigation) or kernel-level optimization.

## Branch integration history — `NpuKernel` API union (2026-07-06)

When the local NPU line (R5–R15 + async dispatch) was rebased onto `chaingun`,
it met a parallel NPU effort already upstream. Both had independently extended
`NpuKernel` from the same base:

- **upstream**: blocking `submit(-> u64)` / `wait(seq)`, `submit_synced`
  (selective per-arg flush), `sync_output` (pipelined read-back cache reconcile),
  `import_dmabuf`, multi-slot command-BO cache.
- **local (kept)**: the async `NpuInFlight` owning-handle path for GPU∥NPU
  overlap with scheduler correlation tags.

Both were kept (union, nothing dropped). The only clash was the `submit`/`wait`
names, resolved by renaming the async pair to **`submit_inflight` /
`wait_inflight`** (`submit_tagged` / `poll` / `NpuInFlight` unchanged). The sole
async caller is `examples/async_smoke.rs`.

**Bisect note:** the reconciliation landed as a separate tip commit (`merge-fix(npu):
unify local async NpuKernel API …`), so the three commits that introduce/inherit
the async API before it — `feat(npu): async NPU dispatch split …` through
`refactor(rdna): single-source kernarg lists …` — do **not** individually compile
(duplicate `submit`/`wait` in `hipfire-xdna`). This is inherent to the divergent
rebase; the branch tip is green. `git bisect skip` that span when bisecting a
`hipfire-xdna` build across it.
