# Exp #8: hipGraph multi-node launch-overhead microbench

**Date:** 2026-05-07
**Status:** VERDICT — LOSS (hypothesis refuted)

## Hypothesis under test

Per Kaden's design note: when emulating WMMA via meta-instruction macros that expand to a sequence of N small kernels, capturing that sequence as a multi-node hipGraph should amortize per-launch dispatch overhead so the cluster does not cost as much as N sequential native launches.

This is distinct from the PR3 verdict (single-kernel-per-graph capture) and the BC-250 monolithic PoC (whole-forward-pass single graph). The intermediate granularity (N small launches per graph, where N is a small integer like 4-16) was empirically untested on RDNA1 prior to this experiment.

## Lever

Wrap N tiny kernel launches as one captured hipGraph, replay, vs the same N native sequential launches, on identical conditions.

## Test infrastructure

`crates/rdna-compute/examples/hip_graph_poc.rs` already implements exactly this microbench (committed 2026-04 era for the original PR3 motivation). It uses `mul_f32` as the per-node kernel with kernarg-blob path to keep capture correctness intact (per the documented `hipStreamBeginCapture` stack-pointer gotcha).

The bench captures N copies of a single mul_f32 launch into one graph, replays the graph 100 times, sorts 20 trials and takes the median wall-clock. Per-node cost is `(median total time / 100 iter) / N nodes`. Sequential reference is a `launch_kernel_blob` in a 200-launch loop with single sync, median of 50 trials.

No code changes required for this experiment.

## Hardware state

- hipx, RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1).
- amdgpu auto DPM, no manual clock overrides.
- Same hw state as Exp #1 / Exp #4 / Exp #7 (verified via `/tmp/perf-research/hw-state/01-pr3-graph-cache-rebench.txt`).
- ROCm 7.2.2 / LLVM 18.

## Bench results

```
--- Reference (single-launch baseline) ---
sequential blob-direct burst: 862.4 µs total / 200 launches → 4.31 µs/launch
single-node graph replay: 13892.5 µs / 200 → 69.46 µs/replay (16.1× worse than native)

--- Multi-node graph (N kernels per graph_launch) ---
  N=  1 nodes:   62.75 µs/graph_launch,  62.75 µs/node  (14.6× worse than native)
  N= 10 nodes:  231.26 µs/graph_launch,  23.13 µs/node  ( 5.4× worse than native)
  N= 50 nodes:  940.76 µs/graph_launch,  18.82 µs/node  ( 4.4× worse than native)
  N=200 nodes: 3650.01 µs/graph_launch,  18.25 µs/node  ( 4.2× worse than native)
```

Native sequential reference: **4.31 µs/launch**.

Per-node graph cost: **18.25 µs at N=200** (asymptotic floor).

Amortization is real — per-node cost drops 14.6× → 4.2× as N grows from 1 to 200 — but asymptotes well above the native cost.

## Verdict

**LOSS.** Pre-registered criterion required `t_graph(N) / t_native(N) <= 0.75` at N=16 and `<= 0.6` at N=64 for a WIN. Empirical ratio is **5.4× at N=10 and 4.4× at N=50**. Graph form is ALWAYS worse, at every granularity tested. Loss criterion (`>= 1.05`) fires by 4-15×.

## Generalization confirmed

Three independent measurements now point to the same structural cause:

| Test | Granularity | Outcome |
|---|---|---|
| BC-250 monolithic PoC (memory-only entry) | ~75-200 launches per graph (whole forward pass) | -12.9% LOSS vs native |
| PR3 graph cache (Exp #1 today + prior) | 1 launch per graph (per-shape cache) | -17.95% / -6.26% LOSS vs native |
| **Exp #8 (today)** | N ∈ {1, 10, 50, 200} launches per graph, microbench | 4-15× per-node LOSS vs native |

The structural cause holds at all granularities: native ROCm 7.2 burst-mode launch pipelining is materially faster than hipGraph replay on RDNA1 silicon, regardless of how many launches are batched into one graph or how the graph is structured. The "graph-amortization" intuition that smaller N might lose to overhead but larger N would amortize is empirically false on this hardware. Even at N=200 the asymptote is 4× worse than native.

## Implication for WMMA-emulation proposal

Even with perfect graph-internal amortization, WMMA emulation would lose to the equivalent native sequential launches by ~4× per-node on RDNA1. There is no granularity where graph capture wins.

Combined with the silicon ceiling (no real WMMA on gfx1010, so emulation maxes at v_pk_fma_f16 throughput = 64 FMAs/wave/cycle vs native WMMA's ~256 FMAs/wave/cycle), the WMMA-emulation-via-hipGraph path is doubly dead on RDNA1:

1. **Silicon ceiling**: emulation can never exceed v_pk_fma_f16 throughput.
2. **Graph overhead**: amortizing emulation via hipGraph is 4× SLOWER than native dispatch.

If WMMA-emulation is ever pursued for cross-arch portability (single source compiles to native WMMA on gfx1100+ and emulated on gfx1010), the gfx1010 lowering should NOT use hipGraph — it should use inline (`__device__ __forceinline__`) macros within a single kernel that dispatches once via native launch.

## Action

- No code changes. Master unchanged.
- Document this verdict. Update memory entry to extend the generalization to small-N multi-node graphs.
- DO NOT propose hipGraph capture as an amortization mechanism for fixed-sequence kernel macros on RDNA1.

## Closure

The "WMMA-emulation as multi-node hipGraph" proposal is closed with empirical refutation. The architectural intuition (multi-node graphs amortize boundary sync) is correct in principle but does not translate to net wins on RDNA1 silicon in practice. AMD's burst-mode launch pipelining on RDNA1 is just very good — graphs cannot compete with it at any granularity tested.

If/when a future ROCm version publishes a fix for graph-internal launch scheduling that closes this gap, all three experiments (BC-250 monolithic, PR3 per-shape, Exp #8 microbench) should be re-run together. Until then: native launches, every time.
