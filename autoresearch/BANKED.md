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
