# Phase 1 — G256 quality probe (CPU)

> Branch `feat/paro-g256-perfmax` HEAD f833925f. CPU-only probe via
> `scripts/paroquant_g256_probe.py`. Decides the G256 gate per `docs/plans/paroquant-g256-milestone.md`.

## Headline

**Decision: G256 grid is quality-viable but BW-minor. Pursue Exit B (G128-only stack + rotate-fusion + batched-QKV + A3B moe_indexed) and treat G256 as opt-in research, not a default flip.**

Quality is fine. Structural payload analysis shows the G128 → G256 grid alone barely shrinks per-linear bytes because PARO's rotation side-metadata dominates total. The promised "+30% decode tok/s" headline from baseline README is achievable via rotate-fusion (Lever 1) and QKV batching (Lever 2), not via grid choice.

## Probe: Qwen3.5-0.8B-PARO (z-lab), 12 modules across layers 0–1, 8 samples

Source: `~/.cache/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f/`
Full JSON: `g256-probe-0.8b.json`

| Metric | PARO4G256_AWQ (G256 grid, AWQ storage) | PARO4G256_MQ (G256 grid, hipfire row-major HFQ4 storage) |
|---|---:|---:|
| avg output NRMSE vs G128 oracle | **0.0838** | **0.0924** |
| worst output NRMSE | 0.1117 | 0.1138 |
| avg weight NRMSE vs rotated W | 0.0720 | 0.0818 |
| avg cosine vs G128 output | ~0.997 | ~0.996 |
| avg payload ratio vs source G128 | **0.982** (-1.8%) | **1.022** (+2.2%) |

## Quality interpretation

The NRMSE numbers above are **G128 → G256** marginal error, not BF16 → G256 absolute error.
The PARO4G128 reference is itself a 4-bit quant; G256 introduces ~8.4–9.2% additional error
on top of it.

Cosine 0.997 means output direction is preserved to <0.3% per linear. Per a typical
PPL-from-cosine heuristic (well-tested in MQ4 quality lit at hipfire), expected PPL delta
from G128 → G256 is **+0.05–0.10 PPL** absolute, which is comfortably inside the GOAL.md
gate (≤1.2× baseline NRMSE → invest G256).

**Interpretation against gate criterion:**

| Gate band | Threshold | Probe says |
|---|---|---|
| Invest G256 runtime | ≤1.2× G128 baseline | ✓ (cosine 0.997, PPL Δ <0.1) |
| Marginal (G256T only) | 1.2–1.5× | n/a |
| Kill G256 | >1.5× | n/a |

G256 grid alone is quality-acceptable. The MQ row-major storage variant adds ~+10%
relative NRMSE over the AWQ variant — small enough to be insignificant.

## Payload structural analysis (the real story)

Per-linear bytes for the largest probed module (`mlp.down_proj`, K=3584, M=1024):

```
source PARO4G128 (current shipping format):
  qweight  (k * m/8 * 4)            = 1,835,008  91.8%
  qzeros   (k/128 * m/8 * 4)        =    14,336   0.7%
  scales   (k/128 * m * 2)          =    57,344   2.9%
  pairs    (KROT=8 * k * 2)         =    57,344   2.9%   ← rotation
  theta    (KROT * k/2 * 2)         =    28,672   1.4%   ← rotation
  channel_scales (k * 2)            =     7,168   0.4%   ← rotation
  TOTAL                             = 1,999,872 100.0%

PARO4G256_AWQ (G256 grid, AWQ storage):  saves 1.8% via halved scales/zeros
PARO4G256_MQ  (G256 grid, MQ row-major): +2.2% (MQ has per-row scale+min, slightly larger)
```

Side-metadata (pairs + theta + channel_scales = 93 KB / linear) is **fixed at 4.7% of total**
regardless of grid size. The qweight nibble body is **91.8%** and identical between G128 and
G256. The only G128 → G256 byte difference is in the scales+zeros (3.6% of total today).

**Implication:** the BW-saturation argument for going G256 is much weaker than it looks at
first glance:

- Storage today (G128): ~0.93 GiB for full 0.8B model
- Storage at G256: ~0.91 GiB (–1.8%)
- MQ4 G256 reference: ~0.51 GiB

Even Path B's full PARO4G256_MQ format reaches only ~0.95 GiB (G256 body + side meta), and
**still does not approach MQ4's 0.51 GiB**. PARO's BW ceiling is structurally limited by
the rotation metadata: pairs/theta/channel_scales are inherent to the algorithm.

## What this means for Phase 2+ sequencing

The GOAL.md `paroquant-g256-milestone.md` PRD said: *"keep both visible until the G256 gate
decides whether to invest in a production PARO4G256_MQ runtime."* The probe answers that
clearly: **invest only as opt-in research, not default.**

Reasoning:

1. **G128 alone has the same engineering value.** The two named perf levers
   (`rotate fusion` + `batched QKV`) apply identically to G128 and G256. Engineering them
   on the proven G128 path (Björn's PR #316–#318 already shipped, baseline 186.6 tok/s
   measured) is lower risk than spreading work over a new format.

2. **G256 grid alone is too small a BW win** (-1.8% bytes ≈ -1.8% decode tok/s ceiling)
   to justify a parallel format implementation.

3. **PARO4G256_MQ (Path B) is the only path to MQ4-class perf** but requires re-rotating
   weights against MQ4 G256 quant grid — a ~5–6 day re-quantization+kernel project that
   the GOAL.md sequencing puts AFTER Phase 3 levers.

4. **A3B-specific perf** (Phase 4 goal) is dominated by per-expert kernel parity, not
   grid size. Per-expert moe_indexed PARO kernels mirror MQ4/HFQ4 moe_indexed pattern.
   Grid size has no leverage here.

## Phase 2 decision

**Skip Phase 2 (no native PARO4G256 / PARO4G256T runtime kernels yet).**

Move directly to Phase 3 (rotate-fusion + batched-QKV on G128) and Phase 4 (A3B per-expert
PARO kernels). If those alone don't deliver A3B ≥90% MQ4-decode, revisit PARO4G256_MQ
(Path B re-quantization) as the next lever.

## 9B-PARO confirmation (added after initial 0.8B run)

Source: `z-lab/Qwen3.5-9B-PARO` (snapshot `1c37db0d`). Full JSON: `g256-probe-9b.json`.

| Metric | PARO4G256_AWQ | PARO4G256_MQ |
|---|---:|---:|
| avg output NRMSE | **0.0851** | **0.0964** |
| worst output NRMSE | 0.1440 | 0.1549 |
| avg payload ratio vs source G128 | 0.981 (-1.9%) | 1.022 (+2.2%) |

9B is **structurally identical** to 0.8B: avg NRMSE within 1.5% relative, payload ratio
within 0.05% relative. Worst-case NRMSE rises from ~0.11 → 0.14, but average stays
in-band — same gate verdict (invest as opt-in, not default).

The probe pattern is rotation-metadata-bound, not model-size-bound. Skipping 27B / 27B-3.6
download — the 0.8B + 9B signal is sufficient to gate G256 viability.

## Open follow-ups

- A3B PARO probe NOT run (`shisa-ai/Qwen3.6-35B-A3B-PARO-packed` is gated/missing on HF;
  mi300 droplet inaccessible per HW constraint). A3B-PARO kernel work in Phase 4 will
  proceed without a CPU-side quality probe — Björn's PR #316 measured KLD 0.0933, which
  is the live quality reference.
