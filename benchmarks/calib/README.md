# Calibration bench corpus

Stable input for TriAttention sidecar calibration optimization runs.

## `calib-1m.txt`

- Source: HuggingFace `Salesforce/wikitext` config `wikitext-103-raw-v1`,
  composed of `test-00000-of-00001.parquet` (full) + a prefix of
  `train-00000-of-00002.parquet` truncated to ~4.8 MB total text.
- Size: 4,798,009 bytes — ~1.2M est tokens at 4 chars/tok.
- md5: `c1879341cb2d4bcf06ead9d1c02ef5fa`

Used by `scripts/calib-mi300x-baseline.sh` and any benchmarking that
compares calibration optimizations across commits. Do not regenerate
in-place: keeping the byte-exact corpus stable is what makes
cross-session timing comparisons meaningful (per
`feedback_perf_bench_tmp_fragile.md` + the τ prompt-shape rule in
`CLAUDE.md`).

If a larger corpus is needed for Phase 5 re-calibration of A3B sidecars
at 5M tokens, generate `calib-5m.txt` separately rather than overwriting
this file.

## `calib-5m.txt`

- Source: HuggingFace `Salesforce/wikitext` config `wikitext-103-raw-v1`,
  composed of `train-00000-of-00002.parquet` (full prefix to 20 MB).
- Size: 19,996,814 bytes — ~5M est tokens at 4 chars/tok.
- md5: `5dc7dc29676eb591869378b3ddc17815`

Used by Phase 5 sidecar re-calibration on MI300X. Stable, byte-identical
across rental sessions for cross-commit comparison.
