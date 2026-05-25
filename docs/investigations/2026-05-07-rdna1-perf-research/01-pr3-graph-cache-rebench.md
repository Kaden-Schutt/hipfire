# Exp #1: PR3 graph-cache re-bench on ROCm 7.2.2

**Date:** 2026-05-07
**Status:** VERDICT — LOSS

## Hypothesis under test

The previous LOSS verdict for PR3 (per-shape hipGraph cache extending PR2 to all 4 fused families + plain GEMV) was recorded as `project_gemv_graph_cache_pr3_2026_05_07.md` with measurements of 0.8B -18.3% / 9B -5.4% on single-card 5700 XT decode. That measurement may have been taken before ROCm 7.2.2's runtime improvements were active. Three landed-since changes are the candidate explanation:

- ROCm 7.1+ doorbell-ring batching for graph launches.
- ROCm 7.2.0 AQL-batch memset graph node optimization (variable AQL packets per memset).
- ROCm 7.2.0 async-handler lock-contention removal.

Hypothesis: under current ROCm 7.2.2, PR3 may now yield neutral or positive results on the same scenario.

## Lever

`HIPFIRE_GEMV_GRAPH=1` (env-gated) on the `feat/gemv-graph-cache-pr3` branch. The branch contains commits PR1 (skeleton), PR2 (plain GEMV cache), and PR3 (fused families: fused_qkv, fused_qkvza, fused_gate_up, gemv_hfq4g256_residual + plain GEMV).

## Scenario

- Hardware: hipx, single RX 5700 XT (gfx1010, ROCR_VISIBLE_DEVICES=1 = 0000:05:00.0 healthy fans).
- Power state: amdgpu default (auto DPM), no manual clock/power overrides.
- Models: `qwen3.5-0.8b.mq4` and `qwen3.5-9b.mq4`.
- KV mode: asym3.
- Prompt: literal `"Why is the sky blue? Answer in two sentences."` (19 tokens).
- max_seq: 4096; max_tokens: 120; temperature: 0.0; deterministic.
- Bench harness: hipfire daemon JSON protocol, fresh process per run.
- 3 fresh-process runs per condition.

## Win criterion (pre-registered)

PR3 (`HIPFIRE_GEMV_GRAPH=1`) shows ≥5% decode tok/s improvement vs baseline on at least one of the two models, with the median outside 2σ of the baseline distribution. PP=1 byte-equivalence held (existing PR3 correctness test passes).

## Loss criterion (pre-registered)

PR3 shows ≥2% decode tok/s regression vs baseline on either model.

## No-change band (pre-registered)

Between -2% and +5%, or within 2σ noise.

## Quality gate

`test_gemv_graph_cache_correctness_pr3` ran prior to perf bench. PASS. All 4 fused families × 100 calls bit-exact across cache disabled vs enabled. Cache stats: hits=48, misses=52, captures=26, evictions=25, hit_rate=48.0%.

## Hardware state at bench time

Recorded to `/tmp/perf-research/hw-state/01-pr3-graph-cache-rebench.txt`. Key values:
- HVD=1 5700 XT (0000:05:00.0): pp_dpm_sclk levels {300, 800, 2100} MHz, started at level 1 (800 MHz, idle-warm).
- Power cap: 220 W. Idle PPT: 9 W.
- Junction temp at idle: well below thermal limits.
- Kernel: `7.0.0-15-generic #15-Ubuntu PREEMPT_DYNAMIC`.
- ROCm: 7.2.2.

DPM ramps under load to level 2 (2100 MHz) during decode.

## Bench results

Raw runs at `/tmp/perf-research/baselines/01-pr3-graph-cache-rebench/results.log` and `/tmp/perf-research/treatments/01-pr3-graph-cache-rebench/results.log`.

### qwen3.5-0.8b.mq4

| condition | run 1 | run 2 | run 3 | median | mean | σ |
|---|---|---|---|---|---|---|
| baseline (graph=0) | 207.1 | 207.7 | 206.5 | 207.1 | 207.10 | 0.49 |
| treatment (graph=1) | 169.9 | 169.8 | 170.1 | 169.9 | 169.93 | 0.12 |

**Delta: -17.95%** (median 169.9 vs baseline median 207.1).
- Treatment median is 76σ below baseline mean.
- Crosses the loss threshold (-2%) by an order of magnitude.

### qwen3.5-9b.mq4

| condition | run 1 | run 2 | run 3 | median | mean | σ |
|---|---|---|---|---|---|---|
| baseline (graph=0) | 55.8 | 55.9 | 55.9 | 55.9 | 55.87 | 0.05 |
| treatment (graph=1) | 52.4 | 52.4 | 52.5 | 52.4 | 52.43 | 0.05 |

**Delta: -6.26%** (median 52.4 vs baseline median 55.9).
- Treatment median is 70σ below baseline mean.
- Crosses the loss threshold (-2%) by 3x.

## Verdict

**LOSS.** Both models show statistically significant regression. Pre-registered loss criterion (≥2% regression on either model) fires unambiguously.

## Comparison vs prior measurement

Prior measurement (see `project_gemv_graph_cache_pr3_2026_05_07.md`):
- 0.8B: -18.3% (now -17.95%; delta 0.35 percentage points, within noise)
- 9B: -5.4% (now -6.26%; delta 0.86 percentage points)

The two measurements agree to <1 pp despite being taken hours apart with no controlled hardware state in the prior measurement. The structural cause (per the original entry) holds:

> ROCm 7.2's native burst-mode launch pipelining beats hipGraph replay for multi-launch forward-pass workloads. Each `hipGraphLaunch` is an "atomic" unit; its kernels run in stream order but graph-to-graph boundaries force implicit syncs. Multi-family forward pass (75+ launches across 5+ shapes per token) loses to the runtime's natural cross-shape pipelining.

ROCm 7.1+ doorbell batching + 7.2.0 AQL-batch memset + async-handler lock-contention removal **did not invert the verdict.** The graph-boundary sync overhead at launch transitions outweighs whatever doorbell-batching savings the runtime now provides, and the multi-family decode pattern doesn't exhibit the access pattern that would benefit from those specific optimizations.

## Action taken

- Code: no commits to master from this experiment.
- Branch: `exp/pr3-rebench` was not created (treatment was tested directly against existing `feat/gemv-graph-cache-pr3` branch with env-gated A/B).
- Memory: existing entry `project_gemv_graph_cache_pr3_2026_05_07.md` was already updated with the informal re-bench note. This formal re-bench confirms the verdict with proper hardware-state recording and pre-registered criterion.
- Open PR? No. The infrastructure remains opt-in default-off on its own branch; do not propose merging to master as a perf optimization.

## Closure

The PR3 graph cache LOSS verdict is now closed with:
- Pre-registered criterion.
- 3 fresh-process runs per condition.
- σ recorded.
- Hardware state recorded.
- Quality gate passed (correctness preserved).
- Two independent measurements (informal + formal) agreeing to within 1 pp.

**DO NOT re-test under future ROCm versions** unless AMD explicitly publishes a fix for graph-boundary sync semantics. That is the actual blocker. Doorbell/AQL/lock-contention improvements at the launch level cannot move the needle while the per-graph implicit sync cost remains.
