# Exp #3: hipHostMallocUncached for boundary_copy staging buffer

**Date:** 2026-05-07
**Status:** DEFERRED (requires bypassing hipMemcpyPeerAsync)

## Lever

`hipHostMallocUncached` (HIP 7.0+) — allocates pinned host memory with the cache-attribute set to write-combined / uncached, avoiding CPU cache pollution for staging buffers that are write-once-read-once on the host side.

## Why deferred

Same architectural blocker as Exp #2: our `boundary_copy` uses `hipMemcpyPeerAsync` which performs HIP-internal host staging when peer access is unavailable. The staging buffer is allocated and managed by the runtime — we don't control it. To use `hipHostMallocUncached` for that buffer, we'd need to:

1. Bypass `hipMemcpyPeerAsync` and roll our own two-stage copy: `[hipMemcpyAsync DtoH (src → uncached pinned host buf), hipMemcpyAsync HtoD (uncached pinned host buf → dst)]`.
2. Maintain a per-pair pinned-host staging buffer pool (sized to the maximum boundary tensor across all band transitions).
3. Lifecycle management: allocate at `Gpus::init_layers`, free at drop, handle multi-stream concurrency.
4. Stream + event coordination matching the existing `BoundaryEvent` semantics.
5. Env-gate the new path so it can be A/B'd against the current `hipMemcpyPeerAsync` baseline.

This is multi-hour invasive work in a correctness-critical PP code path. Not appropriate for an unattended autoresearch session under the contract's "smallest change" rule.

## Quantitative justification for deferral

Per the fabric ablation memory (`project_5700xt_fabric_ablation_2026_05_07`), single-GPU fabric BW is +0.88% within noise on 9B decode. Per the PP=2 baseline memory (`project_pp2_2x5700xt_first_rdna1_2026_05_07`), 9B PP=2 host-staged overhead is -1.8% vs single-card. The total PP overhead attributable to host-staging is small. The pre-registered criterion is "≥2% PP boundary improvement on hetero PP=3."

Even an optimistic estimate puts the `hipHostMallocUncached` lever at 1-3% improvement of host-staging cost = 0.018-0.054% of total wall-clock. This is below the noise floor of our 3-run measurements (typical σ is 0.05-0.5% of median).

The lever is unlikely to clear the criterion even with perfect implementation. Combined with the multi-hour implementation cost, ROI is poor.

## Action

Same as Exp #2 — requires Kaden's design call:

1. Whether to invest in the explicit two-stage copy infrastructure.
2. Whether to re-pre-register against a different scenario where uncached pinned memory matters more (e.g., very large activations crossing the host bus during PP TP=N>2 with FA layer co-location).

Until a design decision lands, parked.

## No code changes

No branch. No bench. Master unchanged.
