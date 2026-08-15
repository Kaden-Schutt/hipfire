<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# weight_pager eviction policy: measured traces, then staleness-aware eviction + router-guided prefetch

| Field | Value |
|---|---|
| State | **planned** |
| Date | 2026-07-19 |
| Author | Kaden Schutt (drafted with agent assistance) |
| Validation route | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Key external input | arXiv:2602.03921 (SpecMD), arXiv:2605.11537 (predictive prefetch + replication), arXiv:2606.15453 (spatio-temporal expert prefetching) |
| Code surface | `crates/hipfire-runtime/src/weight_pager.rs` |
| Companion | `docs/specs/2026-07-19-cross-arch-utilization-campaign.md` P4 (gfx1151 hold + >VRAM MoE prep) |

## 1. Background

`weight_pager.rs` is v0.1 by design: `WeightPager` maintains a residency map
and `ensure_resident` over a `PreadH2DTransport`, with `WeightId` covering
MoE experts, shared experts, router, norms, embed, and lm_head — but only
`Expert` actually pages, and the module doc states **"no real eviction yet —
assumes VRAM is large enough."** `PagerConfig.vram_soft_cap` exists
(`u64::MAX` disables eviction); an LRU order structure exists in-tree, and
the file handle runs under `posix_fadvise(POSIX_FADV_RANDOM)`.

SpecMD (arXiv:2602.03921, Apple) benchmarked MoE expert cache policies
exhaustively and found **expert access does not follow temporal locality**:
LRU/LFU underperform because per-layer expert reuse is *predictable from
routing structure*, not from recency. Their Least-Stale policy evicts the
expert with the longest expected time-to-next-use estimated from observed
routing patterns, and beats recency policies across hardware configs. The
predictive-prefetch cluster (arXiv:2605.11537, 2606.15453) adds the
complementary arm: compute or estimate router decisions slightly ahead of
the consuming layer and prefetch H2D before the miss.

This matters to hipfire exactly where weights exceed VRAM: >VRAM MoE on
gfx1151-class unified-memory machines (the campaign spec's P4) and any
forced-cap paging scenario. On fits-in-VRAM SKUs the pager is dormant, so
the entire work is **measured-first**: no policy change ships without trace
evidence that LRU actually hurts on our routing distributions.

## 2. Work items

### P0 — Access-trace instrumentation (gate; pure observability)

1. Optional trace mode on `WeightPager`: per-token log of
   `(token_idx, layer, expert_id, hit/miss, evicted_id)` behind an env flag
   (off by default; zero cost when unset).
2. Trace fixtures: A3B MoE decode (TG512, greedy + sampled) on a dGPU with
   `vram_soft_cap` forced to 25% / 50% / 75% of expert bytes — simulates
   >VRAM pressure without new hardware.
3. Analysis (scripted, committed under `autoresearch/` or `scripts/`):
   reuse-distance distribution per layer; LRU vs optimal (Belady) miss-rate
   gap per cap; cross-token routing correlation (layer-L expert at token t
   vs t+1..t+8). **Output decides whether policy work is warranted at
   all** — if LRU is within ~2% of Belady at 50% cap, record that and stop.

### P1 — EvictionPolicy abstraction + Least-Stale

1. `enum EvictionPolicy { Lru, LeastStale }` on `PagerConfig`; default stays
   `Lru` until P2 results.
2. Least-Stale per SpecMD: per-expert staleness score from observed
   time-between-reuse (exponential moving statistic per `(layer, expert)`);
   eviction picks the resident expert with the longest predicted
   time-to-next-use. Numerics-neutral by construction — eviction changes
   *which* bytes are resident, never their content; greedy outputs must
   remain byte-exact vs the LRU arm.
3. Unit tests: synthetic traces with known periodicities (property tests
   fit the existing proptest setup); the residency-map/LRU consistency
   invariant already asserted in-tree (`LRU contained {id} but residency
   map did not`) extends to the new policy.

### P2 — Router-guided prefetch (only if P0 shows miss-bound decode)

1. Router early-compute: router GEMV for layer L issued during layer L−1
   (router weights are tiny and always-resident), giving top-k expert
   identity one layer ahead — the prefetch window.
2. Overlapped H2D: prefetch issued on the transport's async path (or a
   dedicated copy stream) against the predicted set; wrong predictions fall
   back to demand-miss semantics unchanged.
3. Fixture: the P0 forced-cap matrix; acceptance = decode tok/s uplift vs
   Least-Stale-without-prefetch at equal cap, miss rate reduced ≥ 30% at
   50% cap, byte-exact greedy parity.

### P3 — gfx1151 >VRAM MoE fixture (with campaign P4)

On Strix Halo: run a MoE artifact sized above the GPU carve-out through the
pager under real UMA conditions; record tok/s per policy arm and per cap.
This is the fixture the paper's UMA section and the cross-arch campaign
both cite.

## 3. Redline interaction (explicit)

Paged H2D traffic is a **dynamic binding** against the retained tape: an
eviction invalidates any tape-baked pointer to the evicted buffer. Until a
certified interaction exists: retained replay refuses to arm when the pager
is under memory pressure (soft cap < resident working set), falling back to
ordinary dispatch — fail closed, per `docs/REDLINE.md` capability-vs-
admission rules. Certifying paged-expert replay is out of scope here.

## 4. Acceptance summary

| Gate | Criterion |
|---|---|
| P0 | Trace stats published (measured checkpoint); LRU-vs-Belady gap quantified at three caps; proceed/stop decision recorded |
| P1 | Least-Stale behind config, byte-exact parity, proptest invariants green; policy A/B on P0 matrix shows miss-rate reduction where SpecMD predicts it |
| P2 | ≥30% miss reduction at 50% cap with decode uplift and zero coherence regressions (8-turn battery) |
| P3 | gfx1151 >VRAM fixture measured per arm; numbers feed the cross-arch campaign's UMA analysis |

## 5. Non-goals

- No expert replication (arXiv:2605.11537's second arm) in v1 — duplication
  trades capacity for locality and needs the P0 evidence first.
- No changes to quantization formats, router math, or expert layouts;
  policy is transport-side only.
- No promotion of any paged route to a product default; `admissions.yml`
  rows remain out of scope for this spec.
