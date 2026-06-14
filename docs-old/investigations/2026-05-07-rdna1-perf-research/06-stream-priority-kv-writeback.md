# Exp #6: hipStreamSetAttribute priority for KV writeback during DFlash verify

**Date:** 2026-05-07
**Status:** DEFERRED (target workload not in scope tonight)

## Lever

`hipStreamSetAttribute` (HIP 7.1+) — set per-stream priority. The proposed application is to elevate the KV writeback stream's priority relative to the main compute stream during DFlash's verify phase, where the prior profiling at `project_27b_dflash_perf_analysis_2026_04_22` showed 21.4% d2h-bound time during verify.

## Why deferred

### Blocker 1: DFlash is refused on pp>1

The autoresearch contract scopes us to "dual-RDNA1 (PP=2 dense), heterogeneous (RDNA1+RDNA2 PP=3), and single-card RDNA1 as a control." DFlash is currently refused at load with `pp>1` per the v1 contract (`daemon.rs:594`). On PP=2 / PP=3 we cannot run DFlash. The lever's target workload doesn't exist in those scenarios.

### Blocker 2: DFlash on single-card RDNA1 is NET NEGATIVE

Per `project_gfx1010_5700xt_validated_2026_05_06`: "DFlash NET NEGATIVE on RDNA1 (-41%, block=16 wastes BW)." DFlash is fundamentally not viable on RDNA1 today. Optimizing the d2h-bound fraction of a workload that's already 41% slower than AR doesn't move it past AR — it just makes the regression slightly less bad.

### Blocker 3: Pre-registered criterion can't be measured cleanly

The criterion was "≥10% reduction in observed d2h-bound profile time during DFlash verify." To validate, we'd need to:
1. Run DFlash on a scenario where it's competitive (gfx1100, not in our test rig tonight).
2. Profile with rocprofiler v3 to attribute d2h time per stream.
3. Apply the lever and re-profile.

This requires a different hardware target (gfx1100) and substantial profiling infrastructure setup. Out of scope for tonight's RDNA1/2 queue.

## Action

Document and skip. No code changes. No branch. Master unchanged.

## When to revisit

- If DFlash gets a per-arch tuning track that makes it competitive on RDNA1 (e.g., block=4 or block=8 draft).
- When benching on gfx1100 / gfx1151 where DFlash is competitive and the d2h profile point can be measured.
- If a different KV writeback scenario emerges (e.g., async d2h overlap during AR decode for telemetry / paging).
