# Exp #4: __reduce_*_sync (or __shfl_xor) replacing LDS-tree reduction in rmsnorm.hip

**Date:** 2026-05-07
**Status:** VERDICT — NO_CHANGE (reverted)

## Lever

Replace the block-level LDS-tree reduction in `kernels/src/rmsnorm.hip` with intra-wave butterfly reduce via `__shfl_xor` (5 shuffle ops per wave) plus a single cross-wave LDS step. `__reduce_add_sync` was originally proposed but the agent's HIP 7 release-note research turned out to be optimistic — the intrinsic is **not present in our HIP 7.2.2 headers** (`grep __reduce_add_sync /opt/rocm/include/hip/` returns empty). Fell back to `__shfl_xor` butterfly which is HIP-portable and lowers to `ds_swizzle_b32` / DPP on gfx10/11.

## Implementation

```cpp
__device__ __forceinline__ float wave_reduce_add_f32(float v) {
    v += __shfl_xor(v, 16);
    v += __shfl_xor(v,  8);
    v += __shfl_xor(v,  4);
    v += __shfl_xor(v,  2);
    v += __shfl_xor(v,  1);
    return v;
}

extern "C" __global__ void rmsnorm_f32(...) {
    // ... compute per-thread sum_sq ...
    sum_sq = wave_reduce_add_f32(sum_sq);
    if (lane == 0) sdata[wave_id] = sum_sq;
    __syncthreads();
    if (wave_id == 0) {
        float v = (lane < n_waves) ? sdata[lane] : 0.0f;
        v = wave_reduce_add_f32(v);
        if (lane == 0) sdata[0] = v;
    }
    __syncthreads();
    // ... rsqrt + scaled output ...
}
```

LDS pressure reduced from 8 round-trip iterations (256-thread block) to ~1 cross-wave step (8 entries written, 8 read).

## Quality gate

`./scripts/coherence-gate.sh` (short battery, 5 model × prompt combinations) ran on `exp/reduce-sync-rmsnorm` branch. **PASS — no hard errors**, all outputs fluent and on-topic:

- qwen3.5-0.8b.mq4 / cap: "Paris."
- qwen3.5-4b.mq4 / code: clean Python one-liner
- qwen3.5-9b.mq4 / reason: 9 with correct logic
- qwen3.5-9b.mq4 / tool-call: clean tool invocation
- qwen3.5-9b.mq3 / reason-mq3: clean

Numerical equivalence is approximate (different reduction order), but the coherence battery confirms it's within bf16 ULP of the prior output for production purposes.

## Bench results

Hardware state: same as Exp #1 (recorded in `/tmp/perf-research/hw-state/01-pr3-graph-cache-rebench.txt`; bench run on the same gfx1010 with no power/thermal change since Exp #1 completed minutes earlier).

### qwen3.5-9b.mq4 decode tok/s

| condition | run 1 | run 2 | run 3 | median | mean | σ |
|---|---|---|---|---|---|---|
| baseline (LDS-tree) | 56.0 | 55.8 | 55.8 | 55.8 | 55.87 | 0.094 |
| treatment (wave-reduce + 1 cross-wave LDS) | 55.9 | 55.9 | 56.0 | 55.9 | 55.93 | 0.047 |

**Delta: +0.10% (mean), +0.18% (median).** Within noise band of -2% to +5%.

Both σ < 0.1 tok/s. The treatment median is well within 1σ of the baseline mean. Statistically indistinguishable.

## Verdict

**NO_CHANGE.** Pre-registered no-change band of -2% to +5% applies. Quality gate PASS. Performance equivalent.

Why the predicted 0.4-0.6% lift didn't materialize: rmsnorm runs once per layer per token. On 9B with 32 layers, that's 32 invocations per decode token. With decode at 56 tok/s = ~17.9 ms/token, rmsnorm gets ~0.5 ms/token total. Saving even half of the LDS overhead is well below the noise floor of our 3-run bench.

The estimate assumed rmsnorm was a non-trivial fraction of decode time. It isn't. Decode on gfx1010 9B is 88% peak DRAM BW and the LDS reductions sit in the shadow of memory-bound GEMV operations. They don't make the critical path.

## Action taken

- **Revert.** Per the pre-registered plan ("Default: revert to keep master diff minimal in autoresearch context. Document the negative perf result.")
- `exp/reduce-sync-rmsnorm` branch deleted on hipx.
- Master is unchanged.
- Documenting the negative perf result here so future sessions don't re-ask the question.

## Notes for future work

The rewritten kernel is genuinely cleaner code (less LDS pressure, easier to reason about cross-arch). If a separate "kernel modernization" track is opened later, this is a good template — paste the `wave_reduce_add_f32` helper and apply the same pattern to other reduction-shape kernels. But on its own merits it's not a perf lever on gfx1010 RDNA1; reduction overhead is in the noise of BW-bound decode.

A different way to spend reduction-cleanup effort: target the FlashAttention attention-block reductions (Exp #5), where the reduction shape is along a different axis and the overhead may matter more relative to the kernel's compute time.
