# Chaingun MoE Module Layout Notes

Date: 2026-06-08
Status: planning notes

## Goal

Chaingun should support MoE models whose routed expert pool is too large to
keep fully resident in the fastest GPU allocation tier. The layout must let the
runtime keep always-used MoE components hot while treating routed experts as a
managed working set.

The first target is Qwen3.6-35B-A3B-style MoE on UMA systems. The same layout
should scale to larger MoE models such as a future q4 model with hundreds of
billions of parameters and 512 routed experts.

## Module Shape

Use a module schema that can hold one or more experts:

```text
ExpertModule {
  layer_id,
  experts: [expert_id...],
  tensors: {
    gate_up_proj,
    down_proj,
  },
  packed_size_bytes,
  placement_policy,
  histogram_metadata,
}
```

The v1 packing policy can emit exactly one expert per module:

```text
layer 12, experts=[147]
```

The schema must not bake in that restriction. Later histogram passes can
coalesce experts into multi-expert modules without changing the scheduler
contract:

```text
layer 12, experts=[147, 66, 123, 218]
```

Single-expert modules are an initial accounting and residency policy, not the
permanent physical layout limit.

## Residency Tiers

For UMA systems, expert movement should not be modeled as direct disk-to-GPU
swaps in the common case. Distinguish these states:

- `pinned`: always resident in the fast model working set.
- `hot`: currently preferred/prefetched for decode or prefill.
- `warm_gtt`: GPU-addressable through GTT/UMA memory, but not assumed hot in
  GPU caches.
- `cold_disk`: recoverable from the model file or a staging cache only when
  memory pressure forces it.

This distinction matters because a large UMA machine may keep a meaningful
number of routed experts in GTT even when it cannot keep them in the fastest
resident allocation tier. Chaingun should optimize for fewer cold faults and
better prefetch ordering, not just for minimizing disk reads.

## Always-Hot MoE Components

For quantized MoE models, keep the stable path resident whenever possible:

- routers,
- shared expert gates,
- shared experts,
- attention and linear-attention weights,
- norms and other per-layer dense projections,
- model metadata needed for routing and scheduling.

For the inspected Qwen3.6-35B-A3B MQ4 artifact, approximate packed sizes were:

| component | approximate packed size |
|---|---:|
| all routers | 22 MiB |
| all shared experts | 67 MiB |
| all shared expert gates | less than 1 MiB |
| one routed expert | 1.59 MiB |
| one layer's 256 routed experts | 408 MiB |
| all routed experts | 15.94 GiB |

On a 32 MiB Infinity Cache system, routers alone are near the full cache size
and shared experts exceed it. Do not assume these components remain in Infinity
Cache across layers. The useful goal is model residency and UMA locality, not
literal last-level-cache residency.

## Routed Expert Policy

Routed experts should be the flexible working set:

1. Start with one expert per module for precise residency accounting.
2. Collect subject-labeled router histograms.
3. Add per-layer top-k co-occurrence data.
4. Build per-layer expert groups from observed co-fire patterns.
5. Promote stable groups into multi-expert physical modules where they improve
   locality without overcommitting residency.

Avoid coarse default chunks such as 32 experts. On Qwen3.6-35B-A3B MQ4, a
32-expert chunk is roughly 51 MiB per layer, which is too large for fine-grained
residency decisions on small-cache UMA systems. Smaller modules also let the
scheduler keep a broader warm pool in GTT.

## Subject-Labeled Histograms

A first scratch sweep using the existing global MoE router histogram showed
different hot expert bands for different prompt categories:

- WikiText generic prose favored bands such as `64-71`, `200-207`, and `0-7`.
- long code favored `144-151`, `168-175`, and `112-119`.
- long prose favored `56-63`, `40-47`, and `32-39`.
- a short reasoning probe favored `72-79`, `128-135`, and `160-167`.

This confirms that subject matter should be part of the grouping data. A
deployment layout may need either a blended prior or multiple profile-specific
packing hints.

The initial runtime histogram collapsed all layers together. That was useful
for global expert hotness but not sufficient for final Chaingun packing. The
dynamic recorder should emit:

- per-layer top-1 and top-k histograms,
- per-layer selected-expert co-occurrence,
- subject/category labels,
- optional windowed locality summaries for prefetch planning.

The cheap always-on candidate is per-layer top-1/top-k hits and weight sums.
Top-k co-occurrence is still small for top-8 routing, but should remain
sampled or evidence-gated until profiling proves the hash-map update cost is
noise in production decode.

Scratch evidence from the initial sweep is under
`.codeinsight+research/chaingun-hist/`.

## Scheduler Implications

The scheduler should treat model modules separately from session state:

- session-state eviction handles KV/recurrent/prefix state;
- model-module residency handles expert/shared/router placement.

The same priority ideas can apply to both, but the counters and policies should
remain separate. A background request may tolerate recomputing or spilling
session state, while still benefiting from a warm GTT pool of likely experts.

Useful future telemetry:

- pinned/shared/router bytes,
- hot routed expert bytes,
- warm GTT expert bytes,
- cold expert load count,
- expert module prefetch hit/miss count,
- per-subject profile id used for prefetch.

## Next Implementation Slice

The next code slice after per-layer evidence should add a module-residency
simulator over the existing HFQ artifact. It should report hot-set hit rate,
warm-GTT hit rate, cold load count, and prefetch usefulness without changing
the on-disk format.
