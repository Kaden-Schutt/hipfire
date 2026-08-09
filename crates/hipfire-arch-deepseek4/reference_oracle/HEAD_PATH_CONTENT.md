# Head-path content cosine -- CLOSED AT FLOOR

SPDX-License-Identifier: Apache-2.0

## Verdict

**`GPU_HEAD_AT_FLOOR` -- the head path does not explain PPL 59.5.**

Host formulas and GPU `parent_head` both match the torch teacher at the
BF16-act-staging floor on identical L42 residuals. Investigation of
`parent/*` **stops here** per scope decision; deep-layer stage bisect
is out of scope. `parent/*` must not be used as a calibration reference
(see `docs/investigations/2026-08-02-ds4-parent-not-calibration-ref.md`
and the marker on `src/parent/mod.rs`).

## Floors (identical residual -> parent host vs torch)

| stage | cosine | rel L2 |
|-------|-------:|-------:|
| hc_head | 1.00000000 | 5.872e-08 |
| final RMSNorm | 1.00000000 | 7.566e-08 |
| logits f32 acts (ParallelHead) | 1.00000000 | 1.098e-06 |
| logits BF16-staged acts | 0.99999944 | 1.060e-03 |
| BF16 act staging alone | 0.99999944 | 1.060e-03 |

## GPU `parent_head` vs torch

| input residual | logits cos | rel L2 | top1 agree |
|----------------|----------:|-------:|-----------:|
| parent L42 | **0.99999942** | 1.079e-03 | 11/11 |
| ref L42 | **0.99999944** | 1.060e-03 | 11/11 |

Matches BF16-act-staging floor exactly. No head port bug.

## Residual gap through the SAME torch head (not explanatory)

Parent vs ref residual -> torch logits cos **0.994145** (rel 1.083e-01). Real drift, far too small for 12.7x PPL.

## Parent-vs-teacher full-seq KLD

| pair | KLD mean | p50 | p95 | max | top1 agree |
|------|---------:|----:|----:|----:|-----------:|
| ref_fp8 || parent | **2.718** | 1.642 | 8.517 | 22.63 | 0.482 |
| ref_fp8 || ref_exact | **0.040** | 0.0065 | 0.146 | 7.11 | 0.930 |

## Assets

- `head_path_content_compare.py`
- `examples/ds4_parent_head_residual_compare.rs`
- `artifacts/head_path_content/`
- `artifacts/ref_vs_parent_kld/`
