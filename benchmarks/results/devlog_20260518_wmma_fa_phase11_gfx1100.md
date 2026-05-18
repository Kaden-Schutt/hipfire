# Phase 1.1 WMMA-FA fresh-process A/B (gfx1100)

**Date:** 2026-05-18
**Branch:** `feat/wmma-fa-prefill` @ `0b9fcdb4`
**GPU:** gfx1100 (Radeon RX 7900 XT, 84 CUs × 2 SIMDs, ~800 GB/s GDDR6)
**Host:** Linux 6.19.12-arch1-1, Intel Core i9-14900F
**ROCm:** 7.2.53211 (Arch system package)
**Kernel:** `attention_flash_asym4_wmma_tile_batched.hip`, gated via `HIPFIRE_WMMA_FA=1`
**Binary md5:** `e477adf40af88a3866c3f950e522c7fa` (force-rebuilt from sources touched 2026-05-18; pre-rebuild was 2026-05-11)

## Why this run

`devlog_20260517_wmma_fa_phase11_fresh_ab.md` measured the same kernel on
the gfx1151 bench host (Strix Halo APU). The plan
(`docs/plans/wmma-flash-attention-prefill.md`) names **gfx1100/1101/1102
as target #1**, calling out gfx1151 as conditional on bandwidth not being
the bottleneck. gfx1100 is the canonical RDNA3 dGPU and was expected to
show the strongest lift because scalar FA there is ALU-bound, not BW-bound.

## Methodology

Same protocol as the gfx1151 run: fresh `prefill_microbench` process per
measurement, interleaved scalar-first / wmma-first per round, `--n-ctx
2048 --kv-mode asym4 --warmup-iters 0 --measure-iters 1`. N=5 paired
rounds. Median Δ + paired-by-round t-stat reported.

Probe script: `.tmp/wmma-fa-ab/probe-gfx1100.sh` (adapted from the
gfx1151 `benchmarks/results/wmma-fa-probe.sh` — only ROCm path differs;
gfx1100 box has `/opt/rocm` not `/opt/rocm-7.12`).

Force-rebuilt before measuring (cargo incremental cache had a May-11
prefill_microbench binary — predates the WMMA-FA kernel commit
`e7a1a983` entirely). Pre-rebuild md5 `0610cd85…`, post-rebuild md5
`e477adf4…`. Per memory `project_awq_dual_bug_picture.md`, stale-binary
shipping has bitten this project twice — confirming the binary md5
moved is now standard practice.

## Result — Qwen 3.5 4B mq4 (hd=128, `qwen3.5-4b.mq4-cuda.hfq`)

Model file md5: `bf4063ded4182d8b5a7cd275c06641e5` (2.59 GB).

| | scalar | WMMA |
|---|---:|---:|
| n          | 5       | 5       |
| median     | 3116.50 | **3180.00** |
| min        | 3110.90 | 3166.50 |
| max        | 3146.50 | 3206.90 |
| stdev      | 12.78   | 13.40   |

- **Δ median: +2.04%** (3116.50 → 3180.00 tok/s)
- **Paired Δ: +61.30 tok/s ± 6.61** over 5 paired rounds
- **Paired t-stat: +18.55** (every round WMMA > scalar)

Raw rows:

```
1,scalar,3146.5  1,wmma,3206.9
2,wmma,3187.1    2,scalar,3116.5
3,scalar,3123.4  3,wmma,3180.0
4,wmma,3177.5    4,scalar,3110.9
5,scalar,3114.2  5,wmma,3166.5
```

## Result — Qwen 3.5 0.8B mq4 (hd=128, `qwen3.5-0.8b.mq4`)

Model file md5: `0769fdeaa08e82bd6ed555e3f151d04b` (549 MB).

| | scalar | WMMA |
|---|---:|---:|
| n          | 5       | 5       |
| median     | 9548.80 | **9639.50** |
| min        | 9537.70 | 9628.10 |
| max        | 9572.20 | 9647.20 |
| stdev      | 12.10   | 6.74    |

- **Δ median: +0.95%** (9548.80 → 9639.50 tok/s)
- **Paired Δ: +88.12 tok/s ± 13.72** over 5 paired rounds
- **Paired t-stat: +12.85** (every round WMMA > scalar)

Raw rows:

```
1,scalar,9572.2  1,wmma,9639.5
2,wmma,9647.2    2,scalar,9537.7
3,scalar,9549.7  3,wmma,9632.9
4,wmma,9628.1    4,scalar,9540.7
5,scalar,9548.8  5,wmma,9642.0
```

## Interpretation

Both runs are statistically clean (every paired round positive, |t|≫2)
but small in magnitude — neither clears the CLAUDE.md Δ≥5% ship gate.

Side-by-side with gfx1151 (`devlog_20260517_wmma_fa_phase11_fresh_ab.md`):

| GPU | model | Δ pipeline | paired t |
|---|---|---:|---:|
| gfx1151 | 9B mq3   | +1.81% | +9.46 |
| gfx1151 | 0.8B mq4 | **+4.06%** | +34.87 |
| **gfx1100** | **4B mq4** | **+2.04%** | **+18.55** |
| **gfx1100** | **0.8B mq4** | **+0.95%** | **+12.85** |

Two observations:

1. **gfx1100 didn't outperform gfx1151's lift, despite being ALU-bound.**
   The plan's "+15-25% on RDNA3 dGPU prefill" target is not met. The
   kernel commit message (`e7a1a983`) already flagged the dilution
   mechanism: FA is only ~7.5% of 9B-class prefill time, so even a 4×
   FA-kernel win caps the pipeline lift around 5%. gfx1100 just has
   more total compute, so non-FA work runs faster too — the FA fraction
   shrinks rather than expanding.

2. **0.8B on gfx1100 shows a SMALLER lift than 4B**, inverting the
   gfx1151 pattern (where smaller model → bigger lift). Likely cause:
   gfx1100's 0.8B prefill runs at ~9.6k tok/s — kernel-launch overhead
   and fixed per-call costs eat a bigger fraction of per-tile time at
   this scale, so even if the FA kernel arithmetic speeds up, the
   per-launch overhead floor blunts the pipeline lift. gfx1151's 0.8B
   only hit 4.9k tok/s, so its launch-overhead fraction was lower in
   proportion to compute.

Implied FA-kernel-only speedup assuming 7.5%/25% FA-fractions
(very rough, just to sanity-check):
- gfx1100 4B (hd=128): +2.04% / ~10% FA-fraction ≈ **+20% on FA kernel**
- gfx1100 0.8B (hd=128): +0.95% / ~20% FA-fraction ≈ **+5% on FA kernel** ← suspicious-low

The 0.8B gfx1100 number suggests the WMMA win is being eaten by something
that isn't present on the 4B path — most likely the kernel-launch /
sub_batch chunking overhead, which is a fixed cost per `launch_maybe_blob`
call and proportionally bigger on a model whose per-layer FA work is
small.

## Open questions for Phase 1.1.5+ / Phase 1.2

1. **Does the WMMA route fully fire across all sub_batches at n_ctx=2048
   on gfx1100?** The gfx1151 coherence smoke test (commit `0946260f`)
   found `sub_batch` becomes < 16 past ~256-512 prefill on 9B with default
   scratch sizing — the gate falls back to scalar there. We have NOT
   re-run that smoke test on gfx1100 yet. If a fraction of the prefill
   is silently running on the scalar fallback, the +2.04% number
   understates the kernel win. Phase 1.1.5 (sub_batch alignment fix,
   plan `0b9fcdb4`) directly addresses this.

2. **Should we measure long-prefill (n_ctx ≥ 4096) where FA is a bigger
   fraction of total compute?** The plan rev notes FA is ~15-25% of
   prefill time at long context (vs ~7.5% at n_ctx=2048 on 9B), so the
   pipeline-level WMMA win should compound there — but only after Phase
   1.1.5 ships chunk-size kernel-arg support, otherwise scalar fallback
   dominates.

3. **Is the +5% gfx1100-0.8B kernel implication telling us the kernel is
   bottlenecked by something other than FA arithmetic on small models?**
   Worth a `rocprof` kernel-trace on the 0.8B WMMA path to verify the
   WMMA kernel actually runs in less wall-time per call than the scalar
   tile, separately from pipeline-level numbers.

## Disposition

**Phase 1.1 ships gfx1100 numbers as expected: real, statistically
significant, but below the +5% engineering-ship bar.** Keeps
default-off (`HIPFIRE_WMMA_FA=1` opt-in) per the original Phase 1.1 plan.

The interesting result is the **dilution pattern**: gfx1100's larger
total compute throughput shrinks FA as a fraction of prefill faster
than the WMMA kernel can lift it. Two paths forward:

- **Phase 1.1.5 (sub_batch alignment).** Highest leverage — guarantees
  WMMA fires across all sub_batches on long-prefill, which is where FA
  is the largest fraction of prefill time. Plan commit `0b9fcdb4`
  already specifies B-first approach (chunk_size kernel arg + OOB
  masking, ~1.5-2 hours).

- **Long-prefill A/B post-Phase-1.1.5.** Re-measure at n_ctx ≥ 4096 on
  both gfx1151 and gfx1100. Hypothesis: pipeline lift should grow to
  +5-10% once the WMMA path stops falling back to scalar past ~512.

Phase 1.2 (asym2) remains worth doing for gfx1151 (its default KV mode),
but on gfx1100 the user-side incentive is weaker — asym4 is the typical
KV mode here.

## Probe scripts

- gfx1151: `benchmarks/results/wmma-fa-probe.sh`
- gfx1100 (this run): `benchmarks/results/wmma-fa-probe-gfx1100.sh` — only
  diff is the ROCm path. The two probes can be unified in a follow-up
  once the PATH/LD_LIBRARY_PATH detection is generalized.

Run with: `N=5 NCTX=2048 MODEL=<path> bash benchmarks/results/wmma-fa-probe-gfx1100.sh`
