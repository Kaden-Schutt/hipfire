# RDNA1/2 Perf Autoresearch Summary

**Date:** 2026-05-07
**Hardware:** hipx (Strix Halo gfx1151 iGPU + 2× RX 5700 XT gfx1010 + 1× RX 6950 XT gfx1030)
**ROCm:** 7.2.2 / LLVM 18+
**Methodology:** Pre-registered criteria, 3 fresh-process runs per condition, σ recorded, hardware state recorded, quality-gate alongside perf-gate, auto-revert on loss/no-change.

## Wake-up deliverable

All 7 experiments + audit have committed verdict docs. Master has the one win squash-merged. No master pollution from losses.

## Results table

| # | Experiment | Verdict | Master change | Code branch state |
|---|---|---|---|---|
| Audit | LLVM 20 atomic fadd scan | CLEAN | doc commit | n/a |
| 1 | PR3 graph-cache re-bench | LOSS | doc only | left on hipx for reference |
| 2 | hipMemcpyBatchAsync for boundary_copy | DEFERRED | doc only | no branch |
| 3 | hipHostMallocUncached for boundary_copy | DEFERRED | doc only | no branch |
| 4 | wave-reduce RMSNorm replacement | NO_CHANGE | doc only | branch deleted |
| 5 | wave-reduce softmax/attention | DEFERRED | doc only | no branch |
| 6 | Stream priority for KV writeback (DFlash) | DEFERRED | doc only | no branch |
| 7 | gfx10-1-generic compile target | **WIN** | **CODE MERGED** | branch deleted |

## The single win: HIPFIRE_TARGET_ARCH env override

Commit `ac3ff2c` on master. Adds `HIPFIRE_TARGET_ARCH="<arch>"` env override to `dispatch.rs::init_with_device`. Empty / unset preserves prior behavior. When set, value flows through `KernelCompiler::new` → `hipcc --offload-arch` → JIT cache key.

Validated empirically:
- hipcc compiles all hipfire kernels for `gfx10-1-generic` without error.
- Daemon loads cleanly.
- Output byte-identical to gfx1010-specific build (md5 ccbefe413d7f8b68ecef9fd06a16d62b on canonical 9B prompt).
- Decode tok/s 55.83 vs 55.93 baseline = -0.18% (within ±1% tolerance, both σ 0.047).

Unblocks: BC-160 (gfx1011) plan no longer needs separate per-arch JIT cache. Forward compatibility for future Navi-1.x silicon. Default flip to auto-promote gfx101x → gfx10-1-generic is a follow-up.

## Two negative results worth recording

### Exp #1 (PR3 graph-cache LOSS confirmed)

Re-bench under ROCm 7.2.2 with doorbell-batched runtime path active confirms the prior LOSS verdict. Numbers match prior measurement to <1 percentage point:

| model | baseline | treatment | delta | prior delta |
|---|---|---|---|---|
| 0.8B | 207.10 | 169.93 | **-17.95%** | -18.3% |
| 9B | 55.87 | 52.43 | **-6.26%** | -5.4% |

Both >70σ below baseline mean. ROCm 7.1+ doorbell batching + 7.2.0 AQL-batch memset + async-handler lock-contention removal don't move the needle. The structural cause (graph-boundary sync per atomic graph unit overrides the runtime's cross-shape pipelining) holds.

**Don't re-test under future ROCm versions** unless AMD explicitly publishes a fix for graph-boundary sync semantics.

### Exp #4 (wave-reduce RMSNorm NO_CHANGE)

Drop-in replacement of LDS-tree reduction in `rmsnorm.hip` with `__shfl_xor` butterfly + 1 cross-wave LDS step.

| condition | median | mean | σ |
|---|---|---|---|
| baseline (LDS-tree) | 55.8 | 55.87 | 0.094 |
| treatment (wave-reduce) | 55.9 | 55.93 | 0.047 |

Delta +0.10% (within noise). Coherence gate PASS. The estimated 0.4-0.6% lift didn't materialize because rmsnorm reduction overhead is in the shadow of BW-bound GEMV in the decode hot path. Reverted per pre-registered default action.

The rewritten kernel is genuinely cleaner code; if a separate kernel-modernization track is opened later, this is a good template, but on its own merits it's not a perf lever on gfx1010.

### Notable: `__reduce_*_sync` is NOT in our HIP 7.2.2 headers

The agent's research called out `__reduce_{add,min,max}_sync` as ROCm 7.0+. Empirical check showed our installed HIP 7.2.2 does not expose it (`grep __reduce_add_sync /opt/rocm/include/hip/` is empty). Fell back to `__shfl_xor` which is HIP-portable.

## Three deferred levers (same architectural blocker)

Exp #2 (`hipMemcpyBatchAsync`) and Exp #3 (`hipHostMallocUncached`) target the host-staged `boundary_copy` path, but both require bypassing `hipMemcpyPeerAsync` with explicit two-stage copy + per-pair pinned host buffer pool + stream/event coordination. Multi-hour invasive change in correctness-critical PP code path. Quantitatively, host-staging is 1.8% of decode time; optimistic 1-3% improvement of that is below our 3-run noise floor.

Exp #6 (DFlash stream priority) targets a workload (DFlash on RDNA1) that is NET NEGATIVE today and refused on pp>1. Out of scope for tonight's PP=2/3 queue.

These are parked for design call. The levers exist; they just don't have a same-night hot path on RDNA1 inference.

## One deferred lever (test scenario doesn't exercise)

Exp #5 (wave-reduce softmax/attention) — `softmax_f32` is MoE-router only, not on the dense 9B decode hot path. FA tile reductions are embedded in multi-stage kernels exceeding "smallest change" rule. Predicted NO_CHANGE on 9B per Exp #4's analysis. Skipped; documented.

## Hardware state across the session

Recorded to `/tmp/perf-research/hw-state/`. No DPM, power, or thermal anomalies between baseline and treatment runs of any experiment. Same gfx1010 silicon (HVD=1, 0000:05:00.0 healthy fans) throughout. Each treatment was bench'd within minutes of its baseline.

## File policy compliance

- `/tmp/perf-research/{baselines,treatments,hw-state,logs}/` — raw bench data, never committed. ✓
- `docs/investigations/2026-05-07-rdna1-perf-research/` — pre-registration + verdict docs, all committed. ✓ (this index, plus 7 experiment docs + 1 audit doc)
- One branch per experiment, deleted on revert / squash-merged on win. ✓
- Master updated only by validated wins. ✓ (only Exp #7 touched code)

## Forward pointers

- `feedback_hip7_levers_for_gfx1010_2026_05_07.md` — original lever inventory; Tier 1 items #1-3 now empirically validated as not actionable on RDNA1 inference.
- `project_gemv_graph_cache_pr3_2026_05_07.md` — graph-cache memory entry; updated with formal re-bench data.
- BC-160 procurement: Exp #7 win means no per-arch cache fragmentation if the cards land. The kernel JIT path is generic-target-ready.

## Halt rationale

Queue exhausted with verdicts on all items. No experiments require Kaden's design call to begin (some require it to *progress*, but all have committed verdicts at the deferral point). No three-strike systemic environment failure occurred (the LLVM `__reduce_*_sync` finding was a one-strike that pivoted cleanly to `__shfl_xor`). Stopping cleanly per the contract's wake-up deliverable spec.
