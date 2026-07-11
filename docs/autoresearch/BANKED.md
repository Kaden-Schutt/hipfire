# Banked autoresearch wins (gfx1201, a3b-mq4r decode)

## 2026-07-05 — baseline_v3: attention register-ring (jetfuel swarm → codex loop), UNIVERSAL byte-exact

Found by the v3-queue loop (3-round Claude brainstorm swarm → codex implement + ISA-precheck +
permutation-sweep → autonomous fold). The lever that landed where 3 months of pedestrian sweeps
missed: a P=4 register-ring software-pipeline on attention's `__shfl_xor`-serialized score/V loops
(loads batched to front, reductions deferred) — MLP 1→4 on a loop the compiler CANNOT auto-pipeline.
BYTE-EXACT (token-id parity), ISA-verified (MLP=4, 0 spills). Everything the compiler could already
pipeline (rmsnorm/residual/gfx12-multirow) correctly no-op'd.

| arch | kernels | lever | A/B Δ | verify |
|---|---|---|---|---|
| **gfx1100** | attention_flash_q8_0_tile (+moe_down tail) | P=4 KV/V register-ring prefetch + branch-free tail | **+6.95%** | stack_verify 163.3→174.7, f=1.0; rollover compose +7.66% |
| **gfx1201** | attention_flash_q8_0_tile | P=4 KV/V register-ring (transferred) | **+2.9%** | ab_certify_v2 f=1.0, coh=OK |

Folded to **baseline_v3**: gfx1100 `ec4ece82`(v2) → `62200a00`(v3) @ 175.6 tok/s; gfx1201
`77e1dfe4`(v2) → `78276f03`(v3). Both on `loop_baseline_$ARCH`; lineage v1→v2→v3, next fold = v4.
Rule banked: kernel MLP levers win ONLY on loops the compiler can't auto-pipeline
(shuffle/reduction-serialized, OOB-guarded tail). See `harness/v3-queue/README.md`.

## 2026-07-04 — round 1 (autonomous Codex loop), COMPOSED +7.35% A/B / +8.1% multiturn
All 3 gfx1201-validated (composed A/B f=1.0 coherent clock-matched; serve_harness multiturn
durable — coherence at baseline level, decode gain holds). NOT yet applied to kernels/src/ —
per-arch validation + dispatch-gating required before fleet adoption (variants are gfx1201-tuned).

| variant | kernel | lever | A/B Δ |
|---|---|---|---|
| gfx1201_R20260704c3.hip | gemv_hfq4g256_residual_scaled | pair adjacent rows: reuse x + sigmoid across 2 rows | +5.20% |
| gfx1201_R20260704c2.hip | fused_rmsnorm_mq_rotate | fold norm into rotation load; kill intermediate LDS writeback+barrier | +2.09% |
| gfx1201_R20260704c1.hip | gemv_hfq4g256_moe_down_k8_indexed_batched_expanded | adjacent-row pair reuses rot_batch x | +1.96% |

Composed (all 3): base 130.0 → 139.55 tok/s = **+7.35%** decode (f=1.0, coherent).
Multiturn (serve_harness chain): base 122.9 → 132.9 = **+8.1%**, attractor 2→3 (baseline-level).

## Anti-clobber promotion rule (per-arch gating) — MANDATORY before any kernels/src/ change

Kernel source is SHARED: one `<k>.hip` JIT-compiles per arch. A gfx12-tuned variant dropped
into a shared-base `<k>.hip` reshapes EVERY RDNA1/2/3 path too. Proven hazard: r2lds was
coherent on gfx1201 but INCOHERENT on gfx1151. So a gfx12 win is NEVER promoted into a
shared-base kernel without a cross-arch clobber check.

Mechanism already in tree: 58 `.gfx12`-suffixed kernels + `.gfx1151/.gfx1100/.gfx1030/.gfx1200`
forks in kernels/src/, resolved by arch at dispatch (`<k>.<arch>.hip` else fall back to `<k>.hip`).

Promotion gate for a gfx12 win on a SHARED-BASE kernel:
1. Run the variant on the RDNA1/2/3 fleet (hipx: gfx1010/1030/1100/1151) — coherence + perf A/B
   vs baseline, per arch, model chosen to fit each card's VRAM.
2a. Coherent AND perf-neutral-or-better on ALL arches → universal win → may replace shared `<k>.hip`.
2b. Incoherent OR perf-regression on ANY arch → CLOBBER → fork to `kernels/src/<k>.gfx12.hip`
    + dispatch predicate (gfx12 uses variant, all other arches keep baseline). STRICTLY gated.

Wins on already-suffixed kernels (`<k>.gfx12.hip`) are inherently gated → bank direct, no check.

STATUS: the 3 round-1 wins are ALL shared-base with no gfx12 fork → each REQUIRES the gate above
before touching kernels/src/. Until checked: archive-only in autoresearch/variants/ (current state).
