# Devlog 2026-05-28 — Hetero MTP perf analysis: not the kernels, the τ

Hetero MTP shipped earlier today (commit fe11669c). On the validation
prompt "Hello", hetero ran -47% tok/s vs single-gpu (11.64 vs 21.93).
Initial assumption: gfx1031 (RDNA2) kernels for the MTP head are
under-optimized vs gfx906. Profile data falsifies that.

## Profile setup

Same prompt for both runs (md5 fbe1cc14880397f613ba81aa84bac201,
md5-stable across normalize):
"Write a Python function to compute the sum of an array using a loop."
qwen3.6-27b AWQ trunk + cvs16384 .mtp head, --compressed-serial,
max=64, temp=0, HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=8.

## Numbers

| metric                   | single-gpu | hetero (gfx906+gfx1031) | delta  |
| ---                      | ---        | ---                     | ---    |
| **per-cycle kernel tot** | 167 ms     | 169 ms                  | +1.5%  |
| trunk gemm_gate_up       | 793 µs/call| 798 µs/call             | +0.6%  |
| trunk gemm_residual      | 378 µs/call| 379 µs/call             | +0.3%  |
| trunk gemm_qkvza         | 399 µs/call| 404 µs/call             | +1.3%  |
| total decode wall        | 3.67 s     | 5.45 s                  | +48%   |
| cycles needed (66 tok)   | 20         | 29                      | +45%   |
| **τ**                    | 3.25       | 2.24                    | **-31%** |

## What the data actually says

The per-cycle kernel cost is **statistically identical** (+1.5% is well
within run-to-run noise). MTP head kernels (rmsnorm_f32 etc.) account
for <1% of cycle time on both paths — the cycle is dominated by trunk
verify (gemm_gate_up + gemm_residual + gemm_qkvza = ~70%).

The 48% wall-clock regression is **entirely from needing +45% more
cycles** to produce the same 66-token output. That's a τ collapse
(3.25 → 2.24), not a kernel slowdown.

Trunk + head are bit-identical between single-gpu and hetero on the
weight side (same .hfq, same .mtp file). The drafter's chain produces
different draft tokens that the trunk rejects more often, leading to
more cycles.

## Hypothesis

**Numerical drift on RDNA2 kernels.** Same algorithm, slightly
different FP rounding due to:
- wave32 (RDNA) vs wave64 (gfx906/Vega) reduction order
- Different fma/multiply-add intrinsics
- Possibly different rmsnorm shared-memory reduction tree shape

A 1e-4 magnitude drift in the MTP head's `t_mtp_out` would produce a
slightly different distribution over 16k compressed-vocab logits;
argmax can shift on close calls. Compounded over a 4-step chain, this
shifts which candidate tokens get proposed, and trunk verify
(unchanged on gfx906) rejects more of them.

The `feedback_attention_precision` memory note from May 2026 is the
canary: "5% attention error cascades into attractor within ~10 tokens
under greedy decode." Our drift is tiny (no attractor in coherence
output, identical text actually produced), but a small drift is
enough to shift τ.

## Next investigation steps

1. **Element-wise compare MTP draft logits across paths.** Dump
   `state.mtp_lm_logits_compressed` (or drafter_state's equivalent) at
   cycle 0 step 0 from both runs and compute max-abs-diff. If <1e-4
   we're seeing a meaningful but small drift; if ~1e-2 we have a kernel
   issue.
2. **Walk the chain backwards.** Capture `prev_hidden` (post-peer-copy
   on hetero, post-capture on single) and verify byte-identical (should
   be — same memcpy source). Then capture `t_mtp_out` after the first
   head forward on both and compare. Then the rmsnorm output. The first
   divergence point isolates the culprit kernel.
3. **Re-run hetero with all RDNA2 kernels swapped to higher precision.**
   E.g. force rmsnorm_f32 to use full f32 (vs any internal f16
   shortcuts). If τ recovers, precision is the lever.

## What this does NOT mean

- It does NOT mean "RDNA2 kernels are bad." They run at the same wall
  time as gfx906 here (which is genuinely interesting on its own —
  gfx1031 with its smaller chip + smaller memory bandwidth matches
  gfx906 on this workload).
- It does NOT mean the hetero plumbing is wrong. The plumbing is
  measurably correct: bit-identical output text, same peer-copy cost
  the microbench predicted.
- It does NOT mean the multi-GPU MTP project is dead. The path to
  +12% sync ROI was always conditional on τ holding within 5-10% of
  single-gpu. A -31% τ drop kills that — but the fix is in the
  numerical correctness of the drafter-side kernels, not in the
  cross-device orchestration.

## What this DOES mean

RDNA2 kernel work IS the right direction for hetero MTP perf — but
not the optimization direction we expected (faster kernels). The
direction is **numerical equivalence** with gfx906's output.

If we can match gfx906's MTP head output bit-for-bit (or within a
tight tolerance) on RDNA2, τ should recover, and hetero MTP becomes a
straight win (frees ~800 MB of gfx906 VRAM, gets the projected ~12%
ROI from offloading the MTP head's compute share).
