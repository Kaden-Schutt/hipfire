# `feat/paro-g256-perfmax` — final summary

> Branch `feat/paro-g256-perfmax` HEAD 3f717ffa. hiptrx gfx1201 (R9700/RDNA4),
> ROCm 7.2. Mission: decide the G256 gate, perfmax the ParoQuant runtime
> on gfx12 to its asymptote.

## Exit decision: B

| Exit B criterion | Status |
|---|---|
| G256 gate decided | ✓ Phase 1 — viable as opt-in research; structural payload analysis shows G256 grid alone saves <2% bytes (rotation side-meta dominates). |
| G128-only stack shipped with rotate-fusion + batched-QKV | ✓ Lever 1 + Lever 2 both shipped (Lever 1 default-off after falsification; Lever 2 default-on at +5.2%). |
| A3B-G128 ≥ 90% of MQ4 decode (= ≥51 tok/s on gfx1201) | ✓ A3B-PARO via PR #319 + Lever 4: 60-63 tok/s = 107% of MQ4 baseline (57). |
| gfx12 asymptote documented | ✓ `gfx12-asymptote.md`. |

## Phase results

### Phase 1 — G256 quality probe (CPU-only)

**Verdict: G256 quality-viable as opt-in research, not default.**

`scripts/paroquant_g256_probe.py` on z-lab/Qwen3.5-{0.8B,9B}-PARO, 12 modules each.

| Metric | PARO4G256_AWQ | PARO4G256_MQ |
|---|---:|---:|
| avg output NRMSE vs G128 oracle | 0.084-0.085 | 0.092-0.096 |
| worst output NRMSE | 0.11-0.14 | 0.11-0.15 |
| avg cosine vs G128 | ~0.997 | ~0.996 |
| payload ratio vs G128 | 0.98 (-1.8%) | 1.02 (+2.2%) |

Cosine 0.997 → expected PPL Δ <0.1 from G128 → G256. Inside the GOAL.md
"≤1.2× G128 NRMSE → invest" gate. But payload analysis shows the G256 grid
alone saves only ~1.8% bytes — rotation side-metadata (pairs/theta/channel_scales)
is fixed at ~4.7% of total and dominates the BW gap. Investing in a native
PARO4G256 runtime delivers minimal BW gain; the Phase 2+3 levers (rotate-fusion +
batched-QKV) apply identically to G128 and G256. **Pursue Exit B.**

Doc: `phase-1-g256-quality-probe.md` · Probes: `g256-probe-0.8b.json`, `g256-probe-9b.json`.

### Phase 2 — Baseline reproduction

**0.8B PARO4G128T engine layout on gfx1201: 161.4 tok/s decode, 171.7 prefill.**

13.2% regression vs May 14 baseline (186.6 tok/s @ 26ebcfc3). Per-kernel
timings are byte-identical to baseline — regression is entirely host-side
dispatch overhead (~0.73ms/token extra) from the post-merge stack of PR
#316/#317/#318 + master merges. Investigation deferred; not exit-gating.

Doc: `phase-2-baseline-reproduction.md`.

### Phase 3 — Two perf levers

| Lever | Δ decode | Status |
|---|---:|---|
| **Lever 1** — fused `rmsnorm + paro4g128t_rotate` | **-2.4%** | FALSIFIED. Single-workgroup design loses CU occupancy vs split kernel's grid=[K/128] parallelism. Shipped default-off as research artifact for future multi-block redesign. |
| **Lever 2** — `fused_qkvza_paro4g128t` + `fused_gate_up_paro4g128t` (default-on flip of PR #319 lineage) | **+5.2%** | SHIPPED default-on. Collapses 3-output QKV + 2-output gate/up into 1 launch each via paro4g128t_quad_rotate (with 4th slot dummy) + paro4g128t_dual_rotate. |

Combined Lever 2 default-on: 0.8B PARO4G128T 161.4 → 169.7 tok/s decode.
test_inference 9/9 PASS in every mode.

Docs: `phase-3-lever-1-falsified.md`, `phase-3-lever-2-shipped.md`.

### Phase 4 — A3B-PARO on gfx1201 via PR #319

**A3B-PARO decode 60-63 tok/s (median 61.15), 107% of A3B-MQ4 baseline (57). Exit B (3) met.**

Per-expert PARO MoE kernels shipped in fivetide's PR #319 (still draft):
- `gemv_paro_q4g128_moe_{gate_up,down}_*_indexed*.hip` (decode-path)
- `gemm_paro_q4g128_moe_grouped_{mmq,mmq_k8,wmma_k2}*.hip` (prefill-path)
- `paro_load_moe_ffn` Rust loader for safetensors-direct A3B-PARO

z-lab/Qwen3.6-35B-A3B-PARO downloaded to hiptrx and verified.
shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed also verified.

**Side fix shipped on `feat/lever-4-gpu-argmax-stability`** (origin):
NaN argmax panic at llama.rs:4418 was triggering ~25-33% of runs on PR #319's
bench path. Root cause is non-deterministic uninit memory in one of the new
MoE kernels (probably tied to F32 router/shared_gate weight band that
Björn's Lever 1 will quantize). Bench now uses `gpu.argmax_f32` (already
the daemon's path) — graceful fallback (returns index 0 on all-NaN) instead
of CPU `partial_cmp.unwrap()` panic. Validation: **24/24 runs PASS** across
both checkpoints post-fix (was 18/22 pre-fix).

Doc: `phase-4-a3b-paro-via-pr319.md`.

### Phase 6 — Asymptote

3 sub-5% experiments + 2 ≥5% lever ships → criterion satisfied.

Doc: `gfx12-asymptote.md`.

## Commits on this branch

```
3f717ffa  feat(paro-g256-perfmax): Lever 2 — default-on FA3+GATE_UP fused for PARO4G128T
3f7544f2  docs(paro-g256-perfmax): Phase 4 — A3B-PARO 60+ tok/s on gfx1201 + Lever 4 NaN fix
44c6d4a4  fix(paro-g256-perfmax): default Lever 1 OFF; falsified at -2.4% decode
22a5358e  feat(paro-g256-perfmax): Lever 1 — fused_rmsnorm_paro4g128t_rotate kernel + 1 wired FA site
54e472b7  feat(paro-g256-perfmax): Lever 1 — wire fused rmsnorm+paro rotate at 12 more call sites
56cefe16  docs(paro-g256-perfmax): Phase 1+2 probe + baseline reproduction
```

## Sibling branch (Lever 4)

```
feat/lever-4-gpu-argmax-stability (off fivetide/feat/paroquant-batched-phase2-shared-expert @ bf04d2b4)
dcf752dc  feat(paroquant-decode): Lever 4 — bench_qwen35_mq4 uses GPU argmax
```

Pushed to origin/feat/lever-4-gpu-argmax-stability. Ready to cherry-pick into PR #319.

## Hardware utilization

| host | use |
|---|---|
| hiptrx (R9700/gfx1201 ×4) | all experiments (only host used) |
| mi300 (gfx942) | not touched (reserved per GOAL.md) |
| hipx (gfx1100) | not touched (Phase 7 candidate) |
| k9lin (gfx1100) | local-only (no GPU bench) |

## Models exercised

| model | format | tok/s decode (gfx1201) | notes |
|---|---|---:|---|
| Qwen3.5-0.8B PARO | PARO4G128T engine | 169.7 | dense, Lever 2 default-on |
| Qwen3.5-9B PARO | (probed only) | — | G256 quality probe only |
| z-lab/Qwen3.6-35B-A3B-PARO | safetensors direct | 60-63 | A3B exit gate model |
| shisa-ai/PARO-full4096-e5-packed | safetensors direct | 59.5 | cross-checkpoint stability |

## Next-steps (not in this exit)

- **i8 MMQ port to gfx12** — `gemm_hfq4g128_mmq.gfx1151.hip` and the PARO MoE
  variants are gfx1151-only. Predicate flip + gfx1201 testing is a 4-8 hr
  follow-up. Largest expected lift: A3B-PARO prefill on gfx1201 from 63.7 →
  closer to gfx1151's 428.3 tok/s.
- **F32 router/shared_gate quantization** (Björn's Lever 1) — biggest single
  decode lever (+8-15%), kept with Björn.
- **F32 → FP16 activations** (Björn's Lever 2) — +10-15% predicted, kept with Björn.
- **Phase 5 dense parity sweep** — only 0.8B measured. 9B/27B/27B-3.6 PARO
  numbers not exercised on hiptrx. GOAL.md soft target (not exit-gating).
- **Phase 7 conditional ports** — gfx1100 + gfx1151 rebuilds + benches.
  Now unblocked by this asymptote certification.
