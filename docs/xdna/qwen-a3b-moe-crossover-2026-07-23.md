# Qwen3.6 A3B MQ4P route and XDNA leverage on gfx1151

Date: 2026-07-23
Host: `hipx`
GPU: gfx1151 (`AMD Radeon 8060S Graphics`, HIP 7.14)
Source baseline: `ef8761eb9f321a60c71d58edd1b26c81e84cb96c`

## Production MQ4P result

The production target is `qwen3.6-35b-a3b.mq4p`, not MQ4R. MQ4P uses graded
mixed expert weights, so a pure-MQ4 route crossover does not establish a safe
production dispatch cutoff.

The existing grouped route remains unchanged. A trial that sent the tiny MQ4P
MTP verification batch through the existing merged mixed-indexed kernels was
rejected:

| MQ4P MTP route | Decode tok/s (3 runs) | Median | Result |
|---|---:|---:|---|
| unchanged grouped | 80.7, 83.5, 77.3 | 80.7 | coherent |
| mixed indexed trial | 37.6, 37.7, 37.9 | 37.7 | degraded output; acceptance depth 1.00 |

The trial was not retained. Plain autoregressive MQ4P measured 62.4 tok/s in
three runs, so the unchanged grouped MTP route is a real 29.3% throughput gain
over MQ4P AR. Speculative verification is therefore already useful for MQ4P,
but its small verification batch is a HIP latency regime rather than the first
XDNA target.

## Actual MQ4P prefill profile

The internal serialized kernel profiler was run against plain MQ4P with Q8 KV.
Percentages are kernel-attribution percentages, not end-to-end wall time; the
profiler synchronizes launches and therefore inflates absolute latency.

| Kernel bucket | 512 tokens | 2048 tokens |
|---|---:|---:|
| mixed grouped MoE GEMM | 38.5% | 35.2% |
| QKVZA Q8 GEMM | 24.2% | 22.3% |
| Q8 residual / WO GEMM | 7.1% | 6.7% |
| QKV Q8 GEMM | 5.8% | 5.3% |
| Q8 attention | 2.6% | 10.4% |
| DeltaNet recurrence | 4.4% | 4.1% |

The complete Q8 projection bundle is 37.1% at 512 and 34.3% at 2048. MoE and
Q8 projections are therefore nearly equal first-order ceilings at 512; at 2048
attention becomes the next scaling pressure.

The MoE kernel averaged 1145.25 us across 160 calls at 512 and 1142.05 us
across 640 calls at 2048. Longer prompts multiply the same per-chunk work
instead of increasing reuse inside a call. This is direct evidence for the
layer-major, cross-chunk aggregation lever.

MQ4P's 20,480 routed-expert tensors have this exact on-disk tier distribution:

| HFQ quant type | Format | Tensors | Share |
|---:|---|---:|---:|
| 13 | MQ4G256 | 6,160 | 30.1% |
| 15 | MQ6G256 | 4,080 | 19.9% |
| 20 | MQ3G256Lloyd | 10,240 | 50.0% |

The count includes gate/up and down weights. An MQ4P MoE overlay therefore
needs three real packed decoders (or separately certified artifact variants);
an MQ4-only implementation cannot claim MQ4P coverage.

## Pure-MQ4 diagnostic

`bench_moe_a3b_route_crossover` remains a synthetic diagnostic for the route
geometry. It measures the complete pure-MQ4 indexed and grouped projection
bundles for Qwen3.6 A3B (256 experts, top-8, K=2048, expert intermediate=512).
It does **not** select the MQ4P production route.

| Batch | Real slots | Grouped rows | Indexed | Grouped | Grouped speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 128 | 30.2 us | 97.3 us | 0.31x |
| 3 | 24 | 384 | 151.9 us | 323.7 us | 0.47x |
| 8 | 64 | 1024 | 470.3 us | 836.3 us | 0.56x |
| 32 | 256 | 4096 | 1836.5 us | 2591.4 us | 0.71x |
| 64 | 512 | 4352 | 3643.0 us | 3174.1 us | 1.15x |
| 128 | 1024 | 4864 | 7259.1 us | 3339.1 us | 2.17x |
| 256 | 2048 | 5888 | 14488.4 us | 3746.7 us | 3.87x |

The useful lesson is geometric: launch and padding dominate at tiny batches,
while grouped reuse wins once enough routed rows accumulate. It is evidence for
full-prompt expert aggregation, not for an MQ4P dtype shortcut.

## XDNA decision

Keep HIP for decode and speculative verification. For XDNA, the first MoE
candidate is a layer-major, full-prompt expert-major mixed low-bit pipeline:

1. Aggregate the routed rows for one layer across every 256-token chunk.
2. Process each active expert's gate/up and down panels across the accumulated
   rows while the imported weight panel is retained.
3. Preserve token-order recurrence and commit the residual transactionally.

At 512 prompt tokens the mean routed depth is 16 rows per expert; at 2048 it is
64. That moves the work toward the grouped side of the measured transmission
curve and reuses each imported expert panel across the full prompt instead of
once per chunk.

The existing Q8 projection overlay remains the lower-risk first kernel because
its layouts are uniform and it covers roughly the same kernel share. The mixed
low-bit MoE pipeline is the higher-reuse follow-on for MQ4P and must support its
actual dtype tags rather than assuming uniform MQ4.

An isolated XDNA2 i8 compute-ceiling probe also established that the exact
M=256, K=N=2048 chunk shape is viable after making the output-DMA transfer
group tail-aware. Across five fresh processes it measured 312.886 us aggregate
p50 but 474.449 us p99. This supports continuing the shape-specific packed-Q8
kernel, while the failed tail and missing HIP/Q8 conversion costs keep
automatic routing locked.

The AIE2P kernel API has no BF16 x i8 MMUL specialization. The Q8 projection
candidate should keep activations in BF16 and decode each imported Q8_0 B
microtile to BF16 in on-chip memory, reuse it across all prompt chunks, and
accumulate in F32. This is a better match for MQ4P prefill than globally
requantizing activations to i8: it preserves the existing activation contract,
does not duplicate model weights, and makes the Q8 decode overhead shrink as
the number of 256-token chunks grows.
