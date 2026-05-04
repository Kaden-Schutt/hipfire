# gfx906 MMQ — daemon CPU usage investigation

Date: 2026-05-04
Hardware: gfx906 (MI50). HIP 6.4.3.
Workload: Qwen 3.5 9B mq4, daemon prefill at pp512+.

## Hypothesis

User observation: "daemon never takes more than 200% CPU." After
the default-on commit (`52eb6bb`), pp256 (561 tok/s) and pp512
(554 tok/s) plateau at essentially the same throughput, suggesting
some non-kernel bottleneck is the floor — possibly CPU-side launch
overhead, since AMD/HIP's default stream synchronizes per-launch
on the host.

## Investigation

### 1. Per-call latency stays flat across pp sweep

`rocprofv3 --kernel-trace` at pp256 vs pp512 (same workload, same
binary, only `--prefill` differs):

| Kernel | pp256 per-call | pp512 per-call |
|---|---|---|
| `_full_set_x64` (gate_up + qkv) | **1.93 ms** | **1.94 ms** |
| `_full_add_x64` (residual K=12288 mlp-down) | **2.31 ms** | **2.36 ms** |
| `_full_add_x16` (residual K=4096 attn-out) | 0.37 ms | 0.40 ms |
| `attention_q8_0_kv_batched` | 0.66 ms | **1.29 ms** |
| `gemm_hfq4g256_residual_fp16_wave64` | 0.64 ms | 0.66 ms |

The MMQ kernels are flat per-call; only `attention_q8_0_kv_batched`
scales (2× per-call as sequence doubles, which is expected if KV is
being attended sequentially). MMQ scaling is linear per-token —
that's correct.

### 2. Wallclock scales linearly

Clean bench, no rocprof:

| pp | wallclock | tok/s | per-token |
|---|---|---|---|
| 256 | 456 ms | 561 | 1.78 ms/tok |
| 512 | 923 ms | 555 | 1.80 ms/tok |

**Wallclock doubles when pp doubles.** Per-token cost is essentially
identical (1.78 vs 1.80 ms). The plateau in tok/s is **not a
bottleneck** — it's the linear-scaling regime where each additional
token costs the same fixed amount of GPU work.

### 3. CPU usage during prefill

Top sampling at 50 ms granularity during a pp600+ prefill:

| Phase | %CPU | State | Threads | Notes |
|---|---|---|---|---|
| Tokenize/embed (start) | 90-100 % | R | 2-3 | single-threaded setup |
| Prefill — launch burst | 100 % | R | 3 | brief ms-scale spikes |
| Prefill — sync wait | 18-40 % | **D** | 3 | mostly here, GPU work |
| Post-prefill (logits/sample/emit/unload) | 100-200 % | R | 3 | brief, not on the throughput path |

The "200% peak" observation was correct but **misattributed**. The
peak happens *after* prefill ends — during the single-token sample,
JSON output, and model unload. During actual prefill, CPU is
**idle 60-80 % of the time** waiting on the GPU (frequent `D`-state
which is uninterruptible kernel-mode sync).

### 4. Default-stream hypothesis test (negative)

Setting `gpu.active_stream = Some(gpu.hip.stream_create()?)` at the
start of `bench_qwen35_mq4`:

| pp | default stream | explicit stream |
|---|---|---|
| 32 | 276 | 277 |
| 128 | 461 | 462 |
| 256 | 561 | 561 |
| 512 | 554 | 554 |

**No change.** Either modern HIP runs default-stream as
"per-thread default" (async w.r.t. host on AMD ROCm 6.4), or the
sync overhead is below measurement noise. Either way, this is not
the bottleneck.

## Conclusion

There is no CPU bottleneck during prefill on this workload. The
~554 tok/s pp512 plateau is the actual GPU per-token cost given
the current MMQ kernel implementation — about 1.8 ms of GPU work
per token. To break through requires per-call MMQ improvements:

1. **Stock per-call comparison**: stock llama.cpp's
   `mul_mat_q<Q4_K, 64>` runs at ~0.85 ms/call extrapolated for
   K=4096. Our `_full_set_x64` runs at ~1.93 ms/call. **Our calls
   are ~2× longer than stock** at the same shape.
2. **Suspected cause**: Option B's 8 `__syncthreads()` per HFQ4
   group vs stock's ~2/group. With many groups per call (K=4096
   → 16 groups, K=12288 → 48 groups), this multiplies into a
   significant overhead.
3. **Best lever** to attack: sync-frequency reduction. Tracked
   as plan §P2.

## What surprised me

I expected to find a launch-loop bottleneck (CPU dispatch faster
than GPU consumes). Instead I found the opposite — GPU is so busy
that CPU is mostly waiting. The 200 % peak that triggered this
investigation is post-prefill housekeeping, not the prefill
throughput floor.

The default-stream-vs-explicit-stream test was particularly useful
as a falsifier: if HIP's default stream were synchronizing per
launch as it does on CUDA, switching streams would produce a
visible improvement. The flat result rules that out.

## Re-attribution: top GPU work at pp512

| Kernel | Calls | Avg | Share |
|---|---|---|---|
| `_full_set_x64` (gate_up + qkv) | 272 | 1.94 ms | 44.75 % |
| `_full_add_x64` (residual K=12288) | 128 | 2.36 ms | 25.66 % |
| `residual_fp16_wave64` (decode-side mostly) | 200 | 0.66 ms | 11.26 % |
| `_full_add_x16` (residual K=4096) | 200 | 0.40 ms | 6.78 % |
| `gated_delta_net_q8` | 48 | 1.28 ms | 5.20 % |
| `attention_q8_0_kv_batched` | 16 | 1.29 ms | 1.75 % |
| (rest) | — | — | <5 % |

The two `_full_*_x64` MMQ kernels alone are **70.4 % of GEMM time
at pp512** (44.75 + 25.66). Any per-call improvement on these is
the highest-leverage next step.

## Code touched

- `crates/engine/examples/bench_qwen35_mq4.rs`: explicit
  `active_stream` setup retained (no perf change but matches the
  post-prefill graph-capture path).

(Plan v2.13 reflects the re-prioritization.)
