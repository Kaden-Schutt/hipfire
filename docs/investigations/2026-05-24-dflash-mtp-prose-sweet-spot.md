# DFlash/MTP prose + sampling sweet-spot (k9lin 7900XTX, 27B MQ4) — 2026-05-24

Pin the **configs**, not the exact tok/s. Prior sessions lost the working DFlash config and burned hours re-deriving it.

## Reproducibility caveat (read first)
All tok/s below are SINGLE warm runs, ±3-5% noise — directional, not pinned medians.
DFlash code ~220 is the only median-validated bar; everything else is indicative.
Before quoting any cell: 5-run fresh-process median, prompt md5 recorded, `scripts/probe_commits.sh` for cross-commit. Blessed code prompt md5 253c7ac50857fe6d0e10fb0d2c5e35c0 (merge_sort), prose md5 unpinned. Prose temp deltas (49-53) are within noise — treat ranks as ties.

## Known-good configs (DON'T re-derive)
- **DFlash code (the real win):** `--no-chatml --kv-mode q8 --ctx 4096 --temp 0.0`, merge_sort_thinking_off prompt → τ≈10.9, ~220 tok/s (~5× AR). rp MUST be 1.0.
- **MTP-solo code greedy:** K=3 p_min0.5 q8 → ~70-74 tok/s (1.6× AR). K=5 plateaus.
- **AR baseline:** ~45 tok/s.

## Prose is ≈AR — spec gives nothing in prod
prose (water-cycle), chatml on, rp1.0; greedy code-shaped wins evaporate under sampling:

| temp | DFlash | MTP n5 |
|--:|--:|--:|
| greedy | ~70 | 57-58 |
| 0.3 | 49.7 τ3.0 | **53.3** |
| 0.5 | 52.4 | 49.5 |
| 0.7 | 49.6 | 46.7 |

AR≈45. So sampled prose ~50 both → ~1.1×, neither helps. MTP edges DFlash only at temp0.3. rp>1.0 craters τ.

## Sweet spot / daemon policy
- Code+tools: greedy + rp1.0 → DFlash 5×. Lock it. (keep repeat-window guard vs temp0 loops.)
- Prose/creative: spec≈AR; route to AR or sample freely. If sampled, temp0.3 rp1.0 (MTP slight edge).
- KV mode is perf-irrelevant (q8≈fwht3≈asym3); fwht3 best quality. p_min greedy is genuine greedy (argmax accept). temp accept = real residual rejection, ~floor — not a bug.

## Win requires a better drafter, not tuning
prose=low-mutual-info → acceptance fundamentally small. Only lever: drafter aligned to prod chatml/sampled dist, or larger non-AR drafter. Tuning exhausted. See settled asymptote.
