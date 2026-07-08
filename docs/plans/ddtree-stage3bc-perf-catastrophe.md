# DDTree Stage 3b/3c on-GPU tree build/follow — REVERTED (87% perf catastrophe)

**Date:** 2026-06-30. **Branch:** feature/speculator-ddtree.
**Reverted commits:** 77d4cbaa (3b, on-GPU tree build), 0ab24be7 (3c, on-GPU greedy follow).
**Revert commits:** b8ae0900, 93c81e2e. **3a (201315b8, on-GPU attn-mask) KEPT.**

## What happened

Stages 3b/3c moved the ddtree candidate-tree BUILD (best-first heap expansion) and
the greedy accept-FOLLOW (tree walk) from the host onto the GPU, to eliminate the
per-cycle tree-control copies (~top-K D2H + parent/positions H2D, <1 KB/cycle). Both
kernels were, by necessity, **single-workgroup / single-thread** — the algorithms are
inherently sequential (heap pop/push, ancestor walk). The shadow-assert validation
proved them BYTE-IDENTICAL to the host (correct output) and the gates passed.

But **a single GPU thread doing sequential work is ms-scale**, vs the host's ~15 µs.
The per-cycle tree build/follow cost exploded, and since ddtree runs it every cycle:

| ddtree temp0 tok/s | before (3df5da3d) | after 3b+3c | after revert | τ (all) |
|---|---|---|---|---|
| code    | 24.2 | **3.1** (−87%) | 24.6 | 4.12 |
| reason  | 32.2 | **4.1** (−87%) | 32.8 | 5.63 |
| prose   | 23.0 | **2.8** (−88%) | 23.2 | 3.52 |
| factual | 32.0 | **4.0** (−88%) | 32.4 | 5.61 |

τ is bit-identical across all three columns → the kernels were correct; the output
never changed. The collapse was pure throughput: an ~8× longer cycle to eliminate a
sub-µs copy.

## Why it wasn't caught at commit time

The per-stage validation checked byte-identity + coherence gates + copy-COUNTS, and
reported "perf neutral." It never measured ddtree **tok/s**. The byte-identical /
copy-count signals were all green while throughput cratered. A user-requested
before/after tok/s A/B (the whole reason for measuring) surfaced it.

## Lessons

1. **A copy-elimination that adds a sequential GPU kernel is almost always a loss.**
   Tree build, heap expansion, and tree walks are the GPU's worst case (1 lane of 1
   wave, rest idle). The host does them in µs; a single-thread kernel takes ms.
2. **"perf neutral" is a measurement, not a deduction.** Byte-identity + gates +
   copy-counts do NOT imply throughput parity. Always A/B tok/s on the daemon before
   claiming neutral — especially when moving work between host and device.
3. **Sub-µs copies are not worth eliminating** if the elimination costs more than the
   copy. The tree-control copies (<1 KB/cycle, sub-µs even on dGPU PCIe) were never
   worth a kernel. Stage 3 was flagged as over-engineering by cost/benefit before
   implementation; it turned out to be strongly negative, not merely zero.

## What was kept (the real wins, all measured flat-or-better on UMA, PCIe wins on dGPU)

- Stage 3a (201315b8): attn-mask built on-GPU by a **parallel** kernel (grid[big_n]
  × block[big_n]) — perf-neutral (ddtree 24.6 ≈ 24.2), removes the 15 KB mask H2D.
  Parallel ≠ sequential: this one is a legitimate GPU workload.
- Stages 1/2 (670eb12f/137e7600): ddtree hidden-state D2D, naive 37 MB D2H deleted,
  device-drain → stream-sync — neutral on UMA, real PCIe wins on dGPU.
- C8 (3f056f1c/1317d16f): chain temp>0 on-GPU sampler — kills 2×9 MB probs D2H.
- ca6691b0: gfx1151 HFQ4G256 occupancy — marginal/noise on chain in the clean A/B,
  correct + harmless.

## If revisited

A GPU-resident ddtree build would need a genuinely **parallel** formulation (the
build is hard to parallelize; the follow could be a parallel ancestor-scan). Given
the copies it would save are sub-µs, this is not worth pursuing. The tree-control
data should stay host-side.
