# Residual CONTENT cosine (parent vs reference)

SPDX-License-Identifier: Apache-2.0

## Why

Parent residual **norms** match reference to sub-1% at every layer/position
(`ParentPosTraj`), yet PPL is 59.5 vs mq2r 14.7. This measurement compares
**directions** (cosine + rel L2 + norm ratio) of the HC residual content,
not scalar L2 summaries.

## Floor domain

Embed floor (layer=-1): mean_cosine=1.0000000000 max_rel_l2=0.000000e+00 over 11 positions. Domain: parent F32 residual vs ref bf16-embed-widened-to-f32. Earlier probe claimed n_diff=0 bit-identical.

- ref domain: `bf16 block forward, dump cast to f32; layout [n_pos,4,4096]`
- parent domain: `F32 residual internal; dump f32 LE; layout [n_pos,4,4096]`
- compare: `cosine + rel_l2 + norm_ratio on flattened hc*dim vectors per position`

bf16 unit roundoff floor is ~3.9e-3; pure-f32 identity floor is 0. Embed
hits the pure-f32 identity floor (cos=1, rel_l2=0) at every position —
comparison harness and token/layout alignment are bit-clean.

## Verdict

**COSINE FALLS first at layer=2 pos=64 cosine=0.99870957 rel_l2=5.093678e-02 norm_ratio=1.002630 (floor_min_cos=1.00000000).**

Cosine falls with **depth**, not at a position-448 cliff. Norm ratios stay
near 1 while cosine drifts → **direction problem**, not scale.

## Per-layer summary (min / mean cosine over 11 positions)

| layer | min cos | mean cos | min nr | max |rel_l2| |
|------:|--------:|---------:|-------:|------------:|
| -1 | 1.000000 | 1.000000 | 1.0000 | 0.0000 |
| 0 | 0.999619 | 0.999699 | 0.9874 | 0.0293 |
| 2 | 0.998710 | 0.999514 | 0.9935 | 0.0509 |
| 10 | 0.992351 | 0.997588 | 0.9973 | 0.1371 |
| 20 | 0.988889 | 0.993261 | 0.9815 | 0.1487 |
| 30 | 0.991527 | 0.994674 | 0.9583 | 0.1303 |
| 38 | 0.984463 | 0.991430 | 0.9717 | 0.1768 |
| 42 | 0.979795 | 0.992214 | 0.9388 | 0.2000 |

## First departures

| threshold | layer | pos | cosine | rel_l2 | norm_ratio |
|----------:|------:|----:|-------:|-------:|-----------:|
| 0.9999 | 0 | 0 | 0.999726 | 2.3602e-02 | 1.002659 |
| 0.999 | 2 | 64 | 0.998710 | 5.0937e-02 | 1.002630 |
| 0.99 | 20 | 448 | 0.988889 | 1.4873e-01 | 0.993534 |
| 0.95 | — | — | — | — | — |

## Full table (selected)

| L | pos | cosine | rel_l2 | norm_ratio |
|--:|----:|-------:|-------:|-----------:|
| -1 | 0 | 1.000000 | 0.0000e+00 | 1.000000 |
| -1 | 1 | 1.000000 | 0.0000e+00 | 1.000000 |
| -1 | 64 | 1.000000 | 0.0000e+00 | 1.000000 |
| -1 | 448 | 1.000000 | 0.0000e+00 | 1.000000 |
| -1 | 512 | 1.000000 | 0.0000e+00 | 1.000000 |
| -1 | 1023 | 1.000000 | 0.0000e+00 | 1.000000 |
| 0 | 0 | 0.999726 | 2.3602e-02 | 1.002659 |
| 0 | 1 | 0.999692 | 2.5149e-02 | 1.003782 |
| 0 | 64 | 0.999647 | 2.6564e-02 | 1.000094 |
| 0 | 448 | 0.999762 | 2.1967e-02 | 1.002389 |
| 0 | 512 | 0.999769 | 2.1664e-02 | 0.996949 |
| 0 | 1023 | 0.999812 | 1.9378e-02 | 1.000163 |
| 2 | 0 | 0.999855 | 1.7066e-02 | 1.001116 |
| 2 | 1 | 0.999369 | 3.5807e-02 | 1.003939 |
| 2 | 64 | 0.998710 | 5.0937e-02 | 1.002630 |
| 2 | 448 | 0.999588 | 2.8757e-02 | 1.001072 |
| 2 | 512 | 0.999620 | 2.7574e-02 | 1.000232 |
| 2 | 1023 | 0.999761 | 2.1853e-02 | 1.000538 |
| 10 | 0 | 0.999989 | 4.6794e-03 | 1.000500 |
| 10 | 1 | 0.997185 | 7.5032e-02 | 0.999813 |
| 10 | 64 | 0.992351 | 1.3705e-01 | 1.051888 |
| 10 | 448 | 0.998031 | 6.2841e-02 | 1.001821 |
| 10 | 512 | 0.996698 | 8.1248e-02 | 0.999415 |
| 10 | 1023 | 0.997602 | 6.9425e-02 | 1.002959 |
| 20 | 0 | 0.999986 | 5.3415e-03 | 1.000081 |
| 20 | 1 | 0.999486 | 3.2551e-02 | 0.993900 |
| 20 | 64 | 0.993928 | 1.1043e-01 | 1.003322 |
| 20 | 448 | 0.988889 | 1.4873e-01 | 0.993534 |
| 20 | 512 | 0.990198 | 1.4170e-01 | 1.014094 |
| 20 | 1023 | 0.989388 | 1.4702e-01 | 1.011838 |
| 42 | 0 | 0.999983 | 6.3003e-03 | 0.997776 |
| 42 | 1 | 0.997323 | 7.3236e-02 | 1.001362 |
| 42 | 64 | 0.983600 | 1.8111e-01 | 1.000088 |
| 42 | 448 | 0.990802 | 1.3537e-01 | 0.994605 |
| 42 | 512 | 0.994952 | 1.0287e-01 | 0.972333 |
| 42 | 1023 | 0.994513 | 1.1022e-01 | 0.959805 |

## Assets

- `residual_content_dump.py` (ref)
- `examples/ds4_parent_residual_content.rs` (parent)
- `residual_content_compare.py`
- artifacts: `artifacts/residual_content/`

## Step-4 follow-up

Stage content dumps at L0 and L2 (first clear departure zone) via
`residual_stage_content_dump.py` / `ds4_parent_residual_stage_content`
and `residual_stage_content_compare.py`.
