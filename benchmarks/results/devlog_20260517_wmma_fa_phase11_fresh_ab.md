# Phase 1.1 WMMA-FA fresh-process A/B (gfx1151)

**Date:** 2026-05-17
**Branch:** `feat/wmma-fa-prefill` @ `e7a1a983`
**GPU:** gfx1151 (Radeon 8060S, 40 CUs × 2 SIMDs, ~200 GB/s LPDDR5x)
**ROCm:** 7.12
**Kernel:** `attention_flash_asym4_wmma_tile_batched.hip`, gated via `HIPFIRE_WMMA_FA=1`

## Methodology

Per CLAUDE.md Δ≥5% rule. Fresh `prefill_microbench` process per measurement
to defeat DPM / thermal residue. **Interleaved** (scalar-first / wmma-first
alternating by round) to defeat local trend bias. Paired-by-round t-stat
reported alongside median Δ.

Script: `.tmp/wmma-fa-ab/probe.sh` — measures
`forward_prefill_batch` tok/s at `--n-ctx 2048 --kv-mode asym4
--warmup-iters 0 --measure-iters 1`. Each run: fresh model load + JIT
kernel cache + 1 measure iter. The model-load and JIT costs are excluded
from the reported tok/s (the bench timer wraps only the prefill call).

## Result — Qwen 3.5 9B mq3 (hd=256, n_kv_heads=4, 32 FA layers of 32)

| | scalar | WMMA |
|---|---:|---:|
| n          | 5      | 5      |
| median     | 645.10 | **656.80** |
| min        | 642.80 | 652.90 |
| max        | 649.70 | 662.80 |
| stdev      | 2.69   | 3.27   |

- **Δ median: +1.81%** (645.10 → 656.80 tok/s)
- **Paired Δ: +12.04 tok/s ± 2.55** over 5 paired rounds
- **Paired t-stat: +9.46** (|t|>2 ≈ significant at p<0.05)

Every paired round showed WMMA > scalar (no inversions). Within-session
3-iter benches earlier showed +2.0% with similar variance, suggesting the
fresh-process methodology is bounding noise tightly.

Raw rows (round, config, tok/s):

```
1,scalar,642.8   1,wmma,652.9
2,wmma,656.8     2,scalar,645.1
3,scalar,647.6   3,wmma,656.6
4,wmma,662.8     4,scalar,649.7
5,scalar,642.9   5,wmma,659.2
```

## Result — Qwen 3.5 0.8B mq4 (hd=256, n_kv_heads=4, smaller stack)

| | scalar | WMMA |
|---|---:|---:|
| n          | 5       | 5       |
| median     | 4873.00 | **5071.00** |
| min        | 4856.20 | 5058.30 |
| max        | 4922.70 | 5133.20 |
| stdev      | 25.38   | 34.31   |

- **Δ median: +4.06%** (4873.00 → 5071.00 tok/s)
- **Paired Δ: +203.52 tok/s ± 11.67** over 5 paired rounds
- **Paired t-stat: +34.87** (massively significant)

Raw rows:

```
1,scalar,4911.0  1,wmma,5133.2
2,wmma,5131.1    2,scalar,4922.7
3,scalar,4873.0  3,wmma,5071.0
4,wmma,5058.3    4,scalar,4856.2
5,scalar,4872.0  5,wmma,5058.9
```

## Interpretation

Two clean wins, both with t-stats well above the significance threshold.
**Magnitude scales with model size — opposite to what you might expect:**

| model | n_layers | Δ pipeline | Δ paired |
|---|---:|---:|---:|
| 9B mq3   | 32 | +1.81% | +12.04 tok/s |
| 0.8B mq4 | (smaller) | **+4.06%** | +203.52 tok/s |

The bigger pipeline lift on the smaller model is consistent with FA being
a larger fraction of total prefill time when there's less non-FA work
(fewer/smaller GEMMs, FFNs, etc.). On 9B the FA kernel is roughly 7.5% of
prefill time (per earlier rocprof kernel_stats); on 0.8B it's likely
20-30% of prefill. The WMMA-FA kernel-level lift is similar across
models — it just dilutes more on bigger ones.

Implied kernel-level FA speedup, if we attribute the entire pipeline lift
to the FA kernel:
- 9B: pipeline +1.81% / FA-fraction 7.5% ≈ **+24% on FA kernel alone**
- 0.8B: pipeline +4.06% / FA-fraction ≈ 25% ≈ **+16% on FA kernel alone**

These two estimates are within ~50% of each other, which is plausible
given the rough FA-fraction estimates. Both well below the spike's 5.91×
(but that was on hd=128 fp16-K, no asym dequant, no Givens — a different
kernel doing a different problem).

## Disposition

**Both runs pass the statistical-significance bar (t > 2) but neither
clears the CLAUDE.md Δ≥5% ship gate.** The kernel:

- Compiles clean, zero spills, zero private memory (per `gfx-kernel-metadata`)
- Produces numerically correct output within fp16-Q-narrow precision envelope
  (argmax preserved on Qwen 3.5 9B prefill 256; top-5 ranks 2-4 swap)
- Runs without crashes through full prefill chunks
- Gives a real but small pipeline win

**Recommended disposition: keep default-off (`HIPFIRE_WMMA_FA=1` opt-in)
as the plan already specifies.** The branch can serve as research
scaffolding for:
- gfx1100 dGPU testing (different bandwidth-vs-compute balance — may show
  a bigger win on the more bandwidth-rich part)
- Further kernel tuning (online FA-2 to drop scores LDS, eliminating
  lds_q_rot, etc.)
- Phase 1.2 asym2 extension (gfx1151's default KV mode)

## Probe script

`.tmp/wmma-fa-ab/probe.sh` — committed alongside this devlog for reproduction
under `benchmarks/results/wmma-fa-probe.sh`. Run with
`N=5 NCTX=2048 MODEL=<path> bash benchmarks/results/wmma-fa-probe.sh`.

## Coherence smoke test (2026-05-18)

`dump_logits_qwen35` on Qwen 3.5 9B mq3 + asym4 KV at multiple prefill
lengths, scalar vs WMMA, comparing last-position logits.

| prefill | argmax match | top-5 overlap | max \|Δ\| logits | WMMA fired? |
|---:|---|---:|---:|---|
| 64   | ✓ (token 44576)  | 5/5 | 0.347 | yes |
| 256  | ✓ (token 107685) | 5/5 | 0.403 | yes |
| 1024 | ✓ (token 78)     | 5/5 | 0.000 | no (sub_batch=72/36/24 not 16-aligned → scalar fallback) |

The argmax + top-5 set match across firing lengths confirms WMMA-FA
output ranking is preserved — drift is in the logit MAGNITUDES (fp16-Q-
narrow envelope, consistent with the Phase 1.0 precision spike result)
but the model's next-token prediction is unchanged.

At prefill=1024, the auto-route gate correctly falls back to scalar
because `sub_batch` (sized by partials capacity / per-pos bytes for the
8-tile context) drops to 24-72, none of which are multiples of
WMMA_BLOCK_M=16. This is a real-deployment limitation: the WMMA path
only fires when the chunking happens to align. For Qwen 3.5 9B with
default scratch sizing, alignment fails past ~256-512 prefill. Phase 1.2
or later should either re-size partials to ensure 16-aligned sub_batch,
or add a chunk_size kernel arg so partial chunks can still WMMA.

## Full coherence-gate.sh status

Attempted with `HIPFIRE_KV_MODE=asym4 HIPFIRE_WMMA_FA=1 ./scripts/coherence-gate.sh`,
killed at the 15-minute timeout. Full battery (~15 model+prompt rows
including 27B and 35B-A3B) takes longer than this on gfx1151 even
without WMMA; not specific to this branch. A narrow single-model run
would suffice for ship validation; deferred.
