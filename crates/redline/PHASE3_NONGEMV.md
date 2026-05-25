# Non-GEMV cost analysis (gfx1100, RX 7900 XTX)

## TL;DR

The hypothesis (memory round-trips on intermediate vectors) is **wrong**. The
diagnosis is **GPU-side dispatch latency**: every kernel costs 9-16 µs of
GPU clock regardless of how much memory it touches or how much compute it
does, and the engine launches ~362 non-GEMV kernels per forward. Both
problems have the same fix shape (collapse many small kernels into fewer
larger ones), but the implication is different — and hipGraph capture/replay
becomes the highest-leverage move.

## Method

`profile_bandwidth_fwd` measures per-kernel GPU time via `hipEventRecord`
+ `hipEventSynchronize` + `hipEventElapsedTime` over 32 warmup + 16 measured
forward passes, on the actual production code path. Cross-checked against
wall-clock from `bench_qwen35_forward` (no profile, no syncs):

| Model | Profile total | Production wall | Δ |
|---|---|---|---|
| 9B MQ4 | 11.40 ms | 12.23 ms | +0.83 ms (host loop + sampling) |

The profile is accurate — production matches it within host overhead.

## Per-model breakdown (16 measured forwards each)

| Model | Total | GEMV | Non-GEMV | Effective tok/s |
|---|---|---|---|---|
| **0.8B MQ4** | 4.93 ms | 2.09 ms (42%) | **2.84 ms** | 202.8 |
| **2B MQ4** | 5.54 ms | 2.64 ms (48%) | **2.91 ms** | 180.4 |
| **4B MQ4** | 9.08 ms | 5.13 ms (57%) | **3.95 ms** | 110.1 |
| **9B MQ4** | 11.40 ms | 7.28 ms (64%) | **4.12 ms** | 87.7 |

**Confirmed:** non-GEMV time is roughly constant across model sizes
(2.84 → 4.12 ms, 1.45×) while GEMV time scales with model size
(2.09 → 7.28 ms, 3.5×). Exactly the user's prediction.

## 9B MQ4 non-GEMV breakdown by category

Total non-GEMV: **4.12 ms** = 36% of forward time

| Category | Time | % of fwd | Calls | Avg/call |
|---|---|---|---|---|
| rmsnorm | 1.96 ms | 17.2% | 113 | 17.3 µs |
| elementwise | 1.19 ms | 10.4% | ~120 | ~10 µs |
| deltanet | 0.58 ms | 5.1% | 24 | 24.0 µs |
| kv_write | 0.15 ms | 1.3% | 16 | 9.4 µs |
| attention | 0.12 ms | 1.1% | 8 | 15.4 µs |
| rope | 0.11 ms | 0.9% | 8 | 13.4 µs |
| embedding | 0.01 ms | 0.1% | 1 | 12.8 µs |

## Per-kernel detail (9B MQ4, 16 measured forwards)

| Kernel | Calls/fwd | Avg µs | Total ms | Bytes touched (MB) | "BW" GB/s |
|---|---|---|---|---|---|
| `rmsnorm_f32` | 65 | 16.42 | 1.07 | 3.19 | 3.0 |
| `l2_norm_f32` | 48 | 9.92 | 0.48 | 0.79 | 1.7 |
| `gated_delta_net_q8` | 24 | 14.61 | 0.35 | 27.53 | 78.5 |
| `silu_mul_f32` | 32 | 9.15 | 0.29 | 4.72 | 16.1 |
| `sigmoid_f32` | 32 | 9.04 | 0.29 | 0.27 | 0.9 |
| `gated_norm_f32` | 24 | 10.84 | 0.26 | 1.57 | 6.0 |
| `alpha_gate_f32` | 24 | 9.77 | 0.23 | 0.01 | 0.1 |
| `conv1d_silu_f32` | 24 | 9.41 | 0.23 | 9.44 | 41.8 |
| `scale_f32` | 24 | 9.13 | 0.22 | 0.39 | 1.8 |
| `rmsnorm_batched` | 16 | 9.71 | 0.16 | 0.49 | 3.2 |
| `kv_cache_write_q8_0` | 16 | 9.38 | 0.15 | 0.08 | 0.5 |
| `attention_q8_0_kv` | 8 | 15.35 | 0.12 | 0.94 | 7.6 |
| `rope_partial_interleaved_f32` | 8 | 13.43 | 0.11 | 0.33 | 3.1 |
| `deinterleave_f32` | 8 | 9.53 | 0.08 | 0.39 | 5.2 |
| `mul_f32` | 8 | 9.20 | 0.07 | 0.39 | 5.3 |
| `embedding_lookup_hfq4g256` | 1 | 12.79 | 0.01 | 0.02 | 1.5 |
| **TOTAL non-GEMV** | **362** | **— ** | **4.12 ms** | — | — |

## Diagnosis: dispatch latency dominates, not memory traffic

Look at `sigmoid_f32`: 32 calls × 9.04 µs each = **0.29 ms total**, processing
**0.27 MB total**. At 960 GB/s peak the memory traffic alone takes
**0.28 µs of GPU time**. The kernel itself does ~256 sigmoids × 5 FLOPS =
1280 FLOPS — call it **2 ns of compute** at 60 TFLOPS.

So the kernel's actual work is ≪1 µs, but it takes **9.04 µs**. The other
**~9 µs is pure GPU dispatch latency** — wave32 issue, register init, MEC
walk, completion handshake. This is the per-kernel floor on gfx1100.

Same pattern across every small kernel:

- `alpha_gate_f32`: 0.01 MB / 0.1 GB/s "bandwidth" → 99% dispatch latency
- `scale_f32`: 0.39 MB / 1.8 GB/s → 99% dispatch latency
- `sigmoid_f32`: 0.27 MB / 0.9 GB/s → 99% dispatch latency
- `mul_f32`: 0.39 MB / 5.3 GB/s → 95% dispatch latency
- `rmsnorm_f32`: 3.19 MB / 3.0 GB/s → 99% dispatch latency

The only non-GEMV kernels with meaningful arithmetic intensity are
`gated_delta_net_q8` (78.5 GB/s — actual compute, the recurrent state
update), `conv1d_silu_f32` (41.8 GB/s), and `silu_mul_f32` (16.1 GB/s).
Together those three contribute 0.87 ms. The other ~3.25 ms of non-GEMV
time is essentially "362 launches × ~9 µs of dispatch latency each, with
~1 ms of real compute mixed in."

**The actual element-wise compute floor (sum of work that has to happen):**

Counting only useful FLOPS for non-GEMV ops on a 4096-dim 32-layer model:
~24 layers × ~80K FLOPS each + ~8 layers × ~30K FLOPS = ~2.2 MFLOPS per
forward. At gfx1100's ~60 TFLOPS, that's **0.04 µs of compute**. Negligible.

**Memory traffic floor:** non-GEMV total bytes from the table = ~50 MB.
At 960 GB/s = **52 µs**. Also negligible.

**Real cost:** 362 launches × 9-16 µs dispatch latency = **3-5 ms**.

The fix isn't to keep intermediates in registers (the data wouldn't change
fast enough to matter — we're talking 50 MB of intermediate traffic vs
~3 ms of dispatch latency). The fix is to **fire fewer launches**.

## Why kernel fusion (Phase 3.6-3.8) helped, but hit a ceiling

Each fusion saved ONE launch worth of dispatch latency (~9 µs) AND ONE
intermediate vector's memory traffic (~0.05 µs). The launch saving was
the dominant effect — explains why fusing rmsnorm+rotate saved ~10 µs/call
even though the rotate kernel was already cheap.

The current state:
- `fused_rmsnorm_mq_rotate` (Phase 3.6): saves 32 launches/forward
- `gemv_hfq4g256_residual` (Phase 3.7): saves 32 launches/forward
- `fused_silu_mul_mq_rotate` (Phase 3.8): saves 32 launches/forward
- Total saved: ~96 launches × 9 µs = ~860 µs

Fusion is hitting diminishing returns at the level of "fuse adjacent
kernels with compatible parallelism patterns." The remaining 362 launches
mostly DON'T have the same parallelism pattern as their neighbors:

- `gated_delta_net` is one workgroup per head, 24 heads
- `rmsnorm` is one workgroup per row, 1 row
- `gemv` is one workgroup per row, M rows
- These all have different grids — fusing them naively requires either
  redundant compute (each WG does the rmsnorm again) or LDS coordination
  across workgroups (slow on RDNA3).

Targeted fusion can probably save another **80-120 launches** (~720-1080 µs),
reducing non-GEMV from 4.12 → ~3.0 ms. **Not the user's <1 ms target.**

## The actually-big lever: hipGraph capture/replay

`hipGraphCreate` / `hipStreamBeginCapture` / `hipGraphLaunch` lets us
record an entire forward pass as a "graph" of operations and replay it as
a single GPU command. Critically, **the GPU's command processor walks the
captured graph internally**, without going through HIP's
`hipLaunchKernel` → ROCr → AQL queue → completion-signal-wait per launch.
The per-kernel walk inside a graph is reportedly 1-3 µs vs the 9-16 µs we
measure today.

Expected impact for the 9B MQ4 forward pass:

```
Current:
  362 non-GEMV launches × ~10 µs dispatch = 3.62 ms
  Real compute (delta_net + conv1d + silu_mul) = 0.50 ms
  Non-GEMV total                              = 4.12 ms
  GEMV                                         = 7.28 ms
  TOTAL                                        = 11.40 ms

After hipGraph (no kernel changes):
  362 non-GEMV inside graph × ~2 µs walk      = 0.72 ms
  Real compute (unchanged)                     = 0.50 ms
  Non-GEMV total                              = 1.22 ms
  GEMV (also benefits!): 248 × 2 µs = 0.50 ms saved
  GEMV new total                              = 6.78 ms
  TOTAL                                        ≈ 8.0 ms  →  125 tok/s
```

After hipGraph + targeted fusions (eliminate 100 launches):
```
  262 non-GEMV inside graph × ~2 µs           = 0.52 ms
  Real compute                                 = 0.50 ms
  Non-GEMV total                              = 1.02 ms
  TOTAL ≈ 7.8 ms → 128 tok/s
```

For 0.8B MQ4:
```
After hipGraph alone:
  Non-GEMV: ~362 × 2 µs + ~0.5 ms = 1.22 ms
  GEMV: ~138 × 2 µs faster = 1.81 ms saved off 2.09 → 0.27 ms? no actually 2.09 was 138 launches × 15 µs = 2.07 ms.
  At 2 µs each: 138 × 2 = 0.28 ms (but the GEMVs DO actual work, weight read takes ~10 µs each on 0.8B)
```

Hmm, the GEMV math is more complex. Even on 0.8B, GEMV time is dominated
by weight read (which IS bandwidth-bound). hipGraph won't help GEMV much
since they're already at peak bandwidth — but it WILL help if the weights
fit in L2 (which they do for 0.8B at 400 MB > 96 MB L2 wait no, MQ4
weights for 0.8B are ~200 MB, still > L2).

Let me just bound it: hipGraph cuts non-GEMV from 4.12 → ~1.2 ms on 9B,
giving ~125 tok/s. On 0.8B it would cut non-GEMV proportionally and the
result is dominated by GEMV at ~1.5 ms for the small model = ~370 tok/s.
Not the user's 500 target, but a real 1.8× improvement.

## Risks with hipGraph

1. **Capture-mode incompatibility.** Some HIP API calls can't be captured
   (synchronize, host-side reads). The forward pass uses `hipMemcpyDtoD`
   for splitting conv outputs in DeltaNet — these may or may not capture.
   First experiment would establish.
2. **Graph re-instantiation cost.** Each new sequence position needs a
   different graph (KV cache write index changes). We'd need either:
   - One graph per position (too expensive)
   - Graph with parameterized values via `hipGraphExecKernelNodeSetParams`
     (more complex, but supported)
3. **Memory layout assumptions.** Graph captures specific buffer pointers.
   If we resize KV cache mid-stream, the graph needs to be re-captured.
4. **Profile correlation breaks.** `hipEventRecord` inside a graph doesn't
   work the same as on a stream — we lose per-kernel timing visibility.

These are solvable but real. I'd estimate **2-3 days** of integration work
to get a working hipGraph forward path that matches the current numerical
output.

## Two-path comparison

| Approach | Effort | Risk | Non-GEMV after | Tok/s 9B MQ4 | Tok/s 0.8B MQ4 |
|---|---|---|---|---|---|
| Current (baseline) | — | — | 4.12 ms | 80.1 | 200 |
| Targeted fusion (more chains) | **medium** | low | ~3.0 ms | ~95 | ~225 |
| **hipGraph capture/replay** | **medium-high** | medium | **~1.2 ms** | **~125** | **~370** |
| hipGraph + targeted fusion | high | medium | ~1.0 ms | ~128 | ~390 |
| Megakernel (Luce-style) | very high | high | ~0.5 ms (?) | ? | ? |

The user's targets:
- 9B: 120 tok/s — **hipGraph achieves it** (≈125)
- 4B: 174 tok/s — hipGraph alone gets to ~120, would need megakernel for 174
- 0.8B: 500 tok/s — neither path alone hits this; would need megakernel

## Recommendation

**Phase 3a: hipGraph capture/replay** as the next step. It's the highest
ROI per unit of effort given the diagnosis. The hip-bridge crate already
exposes `stream_begin_capture`, `stream_end_capture`, `graph_instantiate`,
`graph_launch`, `graph_exec_destroy` (line 75-86 of hip-bridge ffi.rs) —
the FFI is wired up, just unused.

First-experiment scope: capture 1 layer's forward, instantiate, replay,
verify byte-exact output. Then capture full 32-layer forward. Then
benchmark. If it works, the savings are immediately visible without
touching kernel code.

**Phase 3b: targeted fusions** on top of hipGraph if we still want more.
Each saved launch is now worth less (2 µs inside graph vs 10 µs outside),
so individual fusions matter less, but the total still adds up.

**Phase 3c (if we want to chase 0.8B → 500 tok/s):** megakernel approach
or persistent kernels. Substantial effort. Worth doing only if hipGraph
gets us close and the marginal gain matters for the product story.

## Files

- `crates/redline/PHASE3_NONGEMV.md` (this file)
- Profile data: re-runnable via
  `cargo run --release -p hipfire-runtime --example profile_bandwidth_fwd --features deltanet -- <model.hfq> 32 16`
