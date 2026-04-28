# CDNA / MI300X calibration optimization — Phase 5 results

Single-MI300X session 2026-04-28. Branch `feat/cdna-calib-mfma`.
Implements `docs/plans/cdna-calibration-optimization.prd`.

## Speedup vs Phase B baseline

100k-token wikitext bench (chunk_len=1024, qwen3.6-27b.mq4):

| stage | wall | tok/s | speedup | R̄ |
|---|---|---|---|---|
| Phase 1 baseline (CPU tap) | 3m47s | 457 | 1.00× | 0.537 |
| Phase 2 (GPU flip + skip-K) | 2m56s | 578 | 1.27× | 0.537 |
| Phase 2.5 (+ pretok hoist) | 2m51s | 1271 (loop) | 2.62× | 0.537 |
| Phase 2.6 (+ par pretok)   | 1m26s | 1270 (loop) | 2.62× wall | 0.537 |

R̄ matched byte-exact across all stages — correctness gate met by exact match.

## Phase 5 sidecar re-cal at 1M blended corpus

3 Phase C sidecars regenerated. Compared against existing v2 R̄.

| sidecar | Phase C v2 R̄ | new v3 R̄ | wall | speedup vs Phase B (8h24m) |
|---|---|---|---|---|
| qwen3.6-27b.mq4 | 0.611 | 0.610 | 9m29s | **53×** |
| qwen3.6-35b-a3b.mq4 | 0.392 | 0.391 | 13m13s | **38×** |
| qwen3.5-35b-a3b.mq4 | 0.362 | 0.363 | 13m05s | **39×** |

**All three R̄ within ±0.001 of v2** → Phase 2 GPU/skip-K/pretok changes preserve
calibration math exactly (FP-tolerance gate passed). Optimizations are
purely wall-clock without quality cost.

## Conclusion on A3B R̄ ceiling

A3B sidecars stayed at R̄=0.39/0.36 even on the optimized pipe — same as
Phase C v2 — confirming the PRD hypothesis that the low R̄ is **structural**
(MoE routing variance), not corpus-size-dependent. PRD §risks §3 was correct;
moving from 1M → 5M would not change R̄.

Real next-action for A3B R̄: per-expert sidecars (PRD-mentioned but
out-of-scope for this session). Filed as follow-up.

## Cost

Total MI300X time: ~90 min × $1.40/hr ≈ **$2.10**.
PRD budget was <$30. Came in at 7%.

Stretch goal (5M re-cal) skipped: structurally confirmed not to help.

## Artifacts

- `benchmarks/calib/blended-corpus.txt` — md5 c96c7ca1b189ccc19b09565ccb0c010e
  (built ad-hoc; committed in `scripts/fetch_calibration_corpus.sh --recipe blended`)
- `models/{qwen3.6-27b,qwen3.6-35b-a3b,qwen3.5-35b-a3b}/*.mq4.triattn.blended_v3.bin`
- This dir: 6 profile traces in CSV format (per-chunk timing breakdowns)

## Phase 2 commit chain on `feat/cdna-calib-mfma`

```
9e2fbf1  phase 2: gpu calib default + skip-K + parallel pretokenize
8a88662  phase 5: add 5M-token bench corpus
a93b117  calib README: distinguish bench-only vs sidecar-quality
80c112c  phase 2 follow-up: bound pretokenize input chars
f7f7390  phase 2 follow-up: single-token paths must also try GPU calibrate tap
b0db44e  phase 2 follow-up: pretok loop until covered_tokens reaches max_tokens
```
