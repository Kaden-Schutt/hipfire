# Cross-arch transfer map — baseline_v2 (8 folded kernels)

**2026-07-04.** Applied baseline_v2's 8 gfx12-tuned folded kernels (as a 1266-line patch,
`61a8716d → 77e1dfe4`) on hipx and A/B'd folded-vs-baseline per arch. Combined-fold test
(all 8 together), a3b-mq4r, kv q8, 4-round A/B, coherence-checked.

| arch | dev | base tok/s | folded tok/s | Δ | f | coh | verdict |
|---|---|---|---|---|---|---|---|
| gfx1100 (7900XTX) | 0 | 157.8 | 165.9 | **+5.13%** | 1.0 | OK | **UNIVERSAL_WIN** |
| gfx1151 (Strix Halo) | 1 | 85.8 | 87.6 | **+2.16%** | 1.0 | OK | **UNIVERSAL_WIN** |
| gfx1010 / gfx1030 | 2/3 | — | — | — | — | — | TBD (a3b won't fit; smaller-model smoke) |
| gfx1201 (R9700, source arch) | — | ~130 | ~150 | +15.9% | — | OK | (baseline_v2 origin) |

## Finding

The 8 gfx12-tuned kernels are **coherent and faster on both gfx1100 and gfx1151** — the
memory levers (LDS-writeback removal, adjacent-row x-reuse, warp-topk, direct-global
rmsnorm) are **arch-portable**, not gfx12-specific. No clobber on either RDNA3 arch.

**Routing decision:** baseline_v2 → **SHARED** `kernels/src/` (helps the whole a3b-capable
fleet), NOT gfx12-only. Per-arch forking (`<k>.gfx12.hip` + dispatch predicate) is reserved
only for *future* kernels that measurably clobber another arch (r2lds precedent) — default
shared, fork on conflict, exactly the intended anti-clobber design.

**Fleet headroom note:** gfx1100 baseline (157.8) already **exceeds gfx1201's folded 150** —
the 7900XTX's ~1.5× memory bandwidth. Folded → 165.9; native-run ceiling ~200+. gfx1100 is
the highest-value next autoresearch campaign. gfx1151 gains less (+2.16%) — Strix Halo's
shared LPDDR5 is a different memory regime.

**Still open:** gfx1010/1030 smoke on a fitting model (qwen3.6-9b / smaller) before declaring
fleet-universal; per-kernel bisection only if a future combined-fold shows a clobber.
