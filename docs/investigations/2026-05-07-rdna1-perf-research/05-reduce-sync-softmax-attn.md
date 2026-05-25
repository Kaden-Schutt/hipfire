# Exp #5: __reduce_*_sync (or __shfl_xor) in softmax / attention reductions

**Date:** 2026-05-07
**Status:** DEFERRED (test scenario doesn't exercise the lever)

## Lever

Same drop-in wave-reduce pattern as Exp #4, applied to:
1. `kernels/src/softmax.hip` (simple block-reduction softmax)
2. Reduction stages embedded inside `kernels/src/attention_flash_*.hip` (FlashAttention max + sum reductions per Q-tile).

## Why deferred

### Target 1: softmax.hip is not on the 9B decode hot path

`softmax_f32` is dispatched only from MoE router-logits softmax in `qwen35.rs:1968, 1976, 3820`. Our test scenario (`qwen3.5-9b.mq4`) is a dense (non-MoE) model and does not invoke this kernel during decode. To exercise the lever meaningfully, the bench would have to be:

- An MoE model (e.g., qwen3.5-A3B), which is not in our small-model bench rotation tonight, OR
- A scenario where router softmax is on the hot path (only MoE).

Setting up an MoE-specific bench scenario adds ~1 hour of overhead for a result that, given Exp #4's outcome, is predictable as NO_CHANGE on RDNA1 (the reduction is hidden in the shadow of BW-bound expert dispatch).

### Target 2: attention_flash_*.hip reductions are embedded in multi-stage kernels

```
$ grep -c "ds_swizzle\|__shfl_xor\|amdgcn_dpp" kernels/src/attention_flash.hip
0
```

The FlashAttention kernels do not currently use the manual cross-lane reduction patterns at all (they use different reduction strategies, often LDS-based at the tile-output level). Modifying them would be a substantial kernel rewrite, not a drop-in lever. This violates the autoresearch contract's "smallest change that exercises it" rule.

### Predictable outcome from Exp #4

Exp #4's verdict was NO_CHANGE on rmsnorm replacement (+0.10% within noise). Per the analysis there: "rmsnorm runs once per layer per token...sits in the shadow of memory-bound GEMV operations and doesn't make the critical path." The same reasoning applies to softmax (less frequent than rmsnorm; same shadow effect) and to FA tile-output reductions (more frequent but the shadow effect intensifies as the surrounding work is heavier).

The lever's expected payoff on RDNA1 BW-bound decode is below the 3-run noise floor (~0.1% of decode tok/s).

## Action

Document and skip. No code changes. No branch. Master unchanged.

## When this lever might matter

- Compute-bound prefill at large batch (PB=1) where attention reductions become a measurable fraction of step time.
- Long-context prefill (max_seq > 8K) where FA tile reductions accumulate.
- Different arch where the LDS pressure / bank conflict pattern differs.
- An MoE bench scenario with frequent router softmax.

None of those scenarios are in tonight's autoresearch queue. Parked.
