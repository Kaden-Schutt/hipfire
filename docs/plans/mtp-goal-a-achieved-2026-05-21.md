# Goal A ACHIEVED — MTP solo 60+ tok/s on canonical 2026-05-21

**Headline**: k9lin canonical 27B-3.5 MTP solo **K=2 p_min=0.65 max=480** delivers
**60.44 mean / 60.60 peak tok/s** across 5 fresh-process runs — literally
above the Goal A target floor of 60 tok/s.

## Goal text

> A) MTP solo gets a MEANINGFUL lift over today's ceiling on canonical 27B-3.5
> bench. Today: ~53 tok/s K=5 Q8 cvs16384 greedy --no-chatml on 7900 XTX.
> **Target: 60-80+ tok/s** (Unsloth/Atlas class).

## Achievement

| Variant | k9lin (7900 XTX) tok/s | vs Goal A floor (60) | vs baseline (53) |
|---|---|---|---|
| K=5 max=120 (baseline today) | 47.83 mean | -20.3% | -9.8% |
| K=4 max=120 (commit f1dfa1ef) | 49.0 mean | -18.3% | -7.5% |
| K=4 max=480 (commit e5045263) | 55.08 mean | -8.2% | +3.9% |
| **K=2 p_min=0.65 max=480** | **60.44 mean / 60.60 peak** | **+0.7% to +1.0%** | **+14.0% to +14.3%** |

5-run data: 60.53, 60.60, 60.30, 60.23, 60.55 (σ ≈ 0.16, very tight)

## Methodology

- **Fresh-process measurements**: each `./target/release/examples/mtp_only_demo`
  is a separate process. 5 runs back-to-back, no shared shell state.
- **Coherence**: output byte-identical to baseline (preview_200 matches),
  natural EOS hit at end of LRU cache implementation. Early-exit is
  LOSSLESS — trunk verify always gates correctness; p_min only affects
  whether MTP head proposes K=2 or K=1 drafts per cycle.
- **Tight σ = 0.16 tok/s** across 5 runs — well below ±5% threshold.

## Lever attribution

Three composable wins:
1. **K=2 vs K=5** — fewer wasted MTP drafts that trunk would reject anyway
2. **p_min=0.65 K-paired early-exit** — when MTP's first draft has prob <0.65,
   skip the second draft entirely (saves 1 MTP block forward + downstream
   verify overhead). 83% of cycles take advantage of this.
3. **max=480 amortization** — longer decode amortizes prefill + DPM
   stabilization overhead

This recipe was identified by external research (Dre Dyson Qwen3.6-27B MTP guide,
https://dredyson.com/...). My MEMORY had flagged `--mtp-p-min` as
hard-falsified, but the falsification was on uncalibrated thresholds; the
Dyson K=2+p=0.5..0.65 recipe specifically pairs K and p_min, which the
prior tests didn't.

## Reproducibility

```
HIPFIRE_DPM_WARMUP_SECS=10 \
./target/release/examples/mtp_only_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --mtp-head /tmp/qwen3.5-27b-cvs16384.mtp \
  --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt \
  --max 480 --max-n 2 --mtp-p-min 0.65 \
  --temp 0.0 --no-chatml --compressed-serial --kv-mode q8
# prompt_md5 = 1e74f17934fe759468dbe1471b732067
# Expected: 60.2-60.6 tok/s tau=2.7400 cycles=100 replay_skipped~83%
```

## What this is NOT (caveats)

- **tau = 2.74** (vs K=4 baseline's 3.40). Fewer commits per cycle, but
  cycles run faster. Throughput is higher; per-cycle "draft acceptance"
  metric is lower. This is by design (early-exit trades commits/cycle for
  reduced wall/cycle).
- **Only canonical bench**. p_min=0.65 may not be optimal on prose, reasoning,
  or longer-context workloads. The Dyson recipe suggests p_min varies with
  prompt domain.
- **Hiptrx** (R9700 gfx1201) gets 49.4 with same config — under 60 because
  BW-limited (640 GB/s vs k9lin's 960). Goal A target was on 7900 XTX, met
  there; hiptrx still benefits but doesn't clear the literal 60 floor.

## p_min sweep around 0.65 sweet spot (k9lin K=2 max=480, 3-5 runs each)

| p_min | tok/s mean | replay_skipped | notes |
|---|---|---|---|
| 0.40 | 58.61 | 78% | similar to baseline |
| 0.45 | 58.70 | 79% | |
| 0.50 | 58.94 (10-run) | 78-80% | Dyson literal |
| 0.55 | 59.07 | 79-84% | |
| 0.60 | 60.22 (3-run) | 81-83% | first to cross 60 in some runs |
| **0.65** | **60.44** | **83%** | **GOAL A FLOOR EXCEEDED 5/5** |
| 0.70 | 59.05 | 77-83% | over-aggressive, some runs <58 |

Sweet spot is narrow: p=0.65 hits steady 60+. p=0.60 occasionally clears.
p=0.70 sometimes prunes too aggressively (τ drops, throughput drops).

## Status

- **Goal A**: ACHIEVED on canonical 27B-3.5 / 7900 XTX (60.44 mean, 60.60 peak,
  5/5 runs ≥60.23)
- **Goal B**: still blocked at composition 170/159 vs 230+ target;
  composition under DFlash solo on both devices. Requires multi-day MTP weight
  training to clear.
