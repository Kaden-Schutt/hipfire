# Banked autoresearch wins (gfx1201, a3b-mq4r decode)

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
