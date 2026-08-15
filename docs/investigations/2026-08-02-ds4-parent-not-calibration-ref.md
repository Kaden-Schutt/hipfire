# `parent/*` is NOT a calibration reference

SPDX-License-Identifier: Apache-2.0
Date: 2026-08-02
Branch context: `ds4-cdna-test-fail` / local worktree `ds4-mi300x-agentmaxx`

## One-line

`crates/hipfire-arch-deepseek4/src/parent/*` scores **PPL 59.507** on the
canonical 1024-token baseline against the torch teacher's **PPL 4.693**.
**Do not use it as a calibration reference.** Gates 6-9 calibrate against
the torch harness (`reference_oracle/ref_ppl_e2e.py` -> `ref_fp8_*.plog`).

## Numbers that close the case

| system | PPL @1024 | top-1 | KLD mean vs ref_fp8 | KLD p50 / p95 / max |
|--------|----------:|------:|--------------------:|---------------------|
| torch ref fp8 (teacher) | **4.693** | 0.640 | 0 | -- |
| torch ref exact | 4.624 | 0.649 | **0.040** | 0.0065 / 0.146 / 7.11 |
| **parent combfix** | **59.507** | ~0.48 agree w/ teacher | **2.718** | 1.64 / 8.52 / 22.63 |
| mq2r (student, prior) | 14.703 | -- | (was prior yardstick) | -- |
| lloyd (student, prior) | 14.564 | -- | -- | -- |

Teacher self-consistency (fp8 vs exact) is mean KLD **0.040**. Parent-vs-teacher
is **2.718** (~67x that control). Every earlier KLD in this effort was measured
against mq2r/lloyd, which are themselves ~3x off the teacher -- those numbers
are retired as yardsticks.

Tokens: `/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin`
(sha `48b0f834a7a60656db79b3784c96e1ed131c501ae5b0d35c2066f45fb5bb86dd`).
Parent plog: `.../combfix/parent_1024_combfix.plog`.
Teacher plogs: `/tmp/ref_ppl_1024/ref_{fp8,exact}_1024.plog`.

## What was eliminated (do not reopen without new evidence)

- HC / MoE / weight loading / embed / `Block.forward` composition / sinkhorn
- Inter-layer plumbing, indexer, RoPE, residual **growth** and **position norms**
- L0 `attn_out` port bug -- parent-vs-ref cos 0.9993 is **at** the ref-fp8-vs-ref-f32
  quant floor 0.9995 (`ATTN_OUT_QUANT_FLOOR.md`)
- **Head path** (this stop):
  - host `hc_head` / final `RMSNorm` / f32 `ParallelHead` vs torch: cos **1.0**
  - BF16 act staging before head GEMM: cos **0.99999944**, rel ~1e-3
  - GPU `parent_head` vs torch on identical residual: cos **0.99999944**, top1 11/11
  - Report: `reference_oracle/HEAD_PATH_CONTENT.md`

One real defect was found and fixed earlier: `Block.hc_post` contracted the
wrong axis of the sinkhorn `comb` matrix (PPL 163.9 -> 59.5). That is not enough.

## Residual magnitude is not quality

ref-fp8 final residual L2 = **124858.6**, ref-exact = **23899.6** (~5.2x) while
both score PPL ~4.6. Residual L2 / `stack_stability` is not a correctness
proxy. See `REF_PPL_E2E.md`.

## Why gates can be green while PPL is 59

Component smokes and per-stage floors test local operators. The full forward's
logit distribution can still be wrong when residual **directions** drift with
depth while norms match (observed: L42 mean pos cos ~0.992), or when some
full-sequence interaction is not visible at L0/L2 stage dumps.

The head path -- the only region outside `Block.forward` previously uncompared --
is now closed at floor. Per project decision, **deep-layer stage bisect was not
pursued**; the torch teacher unblocks Gates 6-9 without a faithful `parent/*`.

## What to use instead

| need | use |
|------|-----|
| Teacher logits / PPL / plog | `reference_oracle/ref_ppl_e2e.py` (fp8 mode) |
| KLD / top-1 vs teacher | `ds4_parent_kld --reference ref_fp8_*.plog` |
| Position accuracy shape | `scripts/plog_fine_scan.py` |
| Residual content dumps | `residual_content_dump.py` / stage siblings |
| Calibration Hessians | torch teacher activations -- **not** `parent/*` |

## Marker in tree

Module-level warning at the top of `src/parent/mod.rs`:
`NOT A CALIBRATION REFERENCE (2026-08-02)`.
