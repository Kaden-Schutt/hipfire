# Redline suballocation wait census

**Date:** 2026-07-12

**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`)

**Model:** `qwen3.6-35b-a3b.mq4r`, FWHT3 KV, ordinary AR

**Clock policy:** automatic; no clock or performance-level override

## Question

Redline's PM4 hazard frontier intentionally treats a whole HIP allocation as
one region. This fails closed when multiple scratch tensors are subviews of one
allocation, but it could also retain compute-idle waits between provably
disjoint subviews. Before adding kernel-specific byte extents, determine whether
any wait in the real A3B tape is blocked only by that allocation-wide aliasing.

## Census

The recorder now retains each pointer's exact device start in addition to the
allocation base and size returned by `hipMemGetAddressRange`. This does **not**
change the scheduler's allocation-wide conflict test. During PM4 preparation a
second audit asks whether a blocked frontier would become independent under the
weaker exact-pointer-start test.

A boundary is a `suballocation_candidate` only when:

1. all pointer effects and allocation lookups are covered;
2. the allocation-wide frontier requires a wait; and
3. no outstanding write/read or write/write pair shares the same pointer start.

This is only a candidate detector, not permission to remove a wait: overlapping
ranges can have different starts. Any non-empty pair would still require a
source-derived byte-extent proof.

## Result

The retained FWHT3 A3B capture reported:

```text
boundaries=832
covered=832
allowlist_independent=80
resource_independent=130
resource_only={
  (kv_cache_write_asym_k_fwht3, kv_cache_write_q8_0): 10,
  (moe_topk_renorm_k8, fused_silu_mul_mq_rotate): 40
}
suballocation_candidates={}
```

All 702 remaining inter-dispatch waits therefore include at least one true
dependency at the exact same device pointer. There is no hot repeated pair for
which byte-range metadata could prove the entire frontier independent.

The 15-position retained-PM4 gate remained exact against direct HIP:

```text
launches=833 kernels=27 sequence_hash=6f56f88512659cba
shadow backend=pm4_ib exact=True pass=True
```

The replay unit subset passes 11/11, including a synthetic discriminator that
separates different pointer starts inside one allocation from a true same-start
dependency.

The sampled eight-turn serve harness averaged 168.8 tok/s. It reproduced the
prior control byte-for-byte for assistant content, generated-token count,
reasoning-token count, and finish reason; it had zero attractors, zero empty
responses, and the expected 2/3 recall on turns 7 and 8. MTP/spec remained off.

## Verdict

Do not add a kernel-by-kernel byte-extent catalog for this tape. It would add a
large ABI maintenance surface and cannot remove a single current wait. Keep the
allocation-wide policy and the exact-start census as a cheap diagnostic for
future kernel sequences; unknown pointers and kernels continue to serialize.

The next ordinary-AR lever is the 256-token FWHT attention tile body. The prior
context-bucket experiment already proved that merely reducing empty grid rows
is not valuable, so that arm must win through useful-tile memory traffic or
instruction efficiency.
