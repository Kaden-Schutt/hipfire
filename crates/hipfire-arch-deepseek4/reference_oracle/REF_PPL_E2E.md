# Reference end-to-end PPL on tokens.bin (1024)

SPDX-License-Identifier: Apache-2.0

## Question

Is parent PPL 59.5 a port bug, or does the reference fp8-activation recipe
itself score ~59 on these tokens?

## Method

- Same `tokens.bin` (1024 ids, sha `48b0f834…`)
- Stream all 43 Blocks via residual harness (~4.7 GiB), then `hc_head` + `norm` + `head(full_logits=True)`
- Score row `t` against `token_ids[t+1]` over 1023 rows (`parent::plog::compare`)
- Two arithmetic modes, same weights:
  1. **fp8** — default `kernel_shim` (`act_quant` + `fp8_gemm` / `fp4_gemm`)
  2. **exact** — act_quant no-op; Linear dequants weights to f32 and matmuls
- Write `HFPLOG01` `.plog` for each mode

## Results

| system | PPL | top-1 | mean NLL | wall (layers+head) |
|--------|----:|------:|---------:|-------------------:|
| **ref fp8** | **4.6928** | 0.6403 | 1.5460 | 633.9s |
| **ref exact** | **4.6238** | 0.6491 | 1.5312 | 613.2s |
| parent (prior) | 59.507 | — | — | — |
| mq2r (prior) | 14.703 | — | — | — |
| lloyd (prior) | 14.564 | — | — | — |

fp8/exact PPL ratio = 1.0149 (activation fp8 barely moves PPL on this sequence).

Final residual L2 after L42: fp8 `124858.6` (matches residual_content_ref);
exact `23899.6` — magnitudes differ a lot under exact, but both PPL≈4.6.

## Verdict

**PARENT_STILL_BUGGY**

Reference fp8 PPL=4.69 is far below parent 59.5 — port still defective.

The premise that "fp8-activation reference might itself be ~59" is **false**.
Reference fp8 PPL **4.69** is *better* than both quants (~14.6) and ~12.7× better
than the parent. The parent is still badly defective relative to its own teacher.

Combined with the L0 `attn_out` floor (parent-vs-ref cos 0.9993 is **at** the
ref-fp8-vs-ref-f32 floor 0.9995): the bug is **not** L0 attention quant noise,
Head path was subsequently closed at floor (`HEAD_PATH_CONTENT.md`); residual-content / full-seq path remains open but is out of scope for `parent/*` until explicitly reopened. **Do not calibrate against parent/*.**

## Assets

- `ref_ppl_e2e.py`
- `artifacts/ref_ppl_1024/ref_ppl_summary.json`
- remote plogs: `/tmp/ref_ppl_1024/ref_fp8_1024.plog`, `ref_exact_1024.plog`
  (0.53 GiB each, HFPLOG01) — ready for `ds4_parent_kld` vs quant plogs

## Residual magnitude is not predictive of quality

Final residual L2 after L42 differs by **~5.2×** between arithmetic modes
(fp8 `124858.6` vs exact `23899.6`) while both score PPL ≈ 4.6 (ratio 1.015).

That single fact:

1. **Retires residual-growth / stack-stability as a correctness proxy.** Even
   median per-row L2 has no demonstrated relationship to output quality once
   the teacher can score the same tokens at both magnitudes.
2. Explains why `ParentPosTraj` matching norms at every layer was compatible
   with a broken parent: magnitude agreement is cheap; direction and the head
   path are not.
3. Means any gate keyed on residual L2 (including `stack_stability`) is
   measuring something with no proven link to PPL/top-1. Do not rebuild that
   dead end.

## Parent vs teacher KLD (ref_fp8 plog)

Yardstick is now the teacher, not mq2r/lloyd.

| pair | KLD mean | p50 | p95 | max | top-1 agree | top-5 overlap | PPL ref | PPL cand |
|------|---------:|----:|----:|----:|------------:|--------------:|--------:|---------:|
| **ref_fp8 ∥ parent_combfix** | **2.718** | 1.642 | 8.517 | 22.631 | **0.482** | 0.389 | 4.693 | 59.507 |
| ref_fp8 ∥ ref_exact (teacher self) | **0.0404** | 0.0065 | 0.146 | 7.114 | **0.930** | 0.904 | 4.693 | 4.624 |

Teacher self-consistency under the only arithmetic choice available is mean
KLD **0.040**. Parent-vs-teacher is mean KLD **2.72** (~67× the control).
The old quant-vs-quant 0.106 control is in the same ballpark as teacher-self;
parent is not.

### Position fine-scan (bucket width 64, every row)

Teacher itself steps down near 448 (0.77 → 0.48) then holds ~0.50–0.60 — so
the mid-sequence dip is partly corpus/structure, not purely a parent bug.
Parent tracks below teacher in every bucket and falls harder after 448
(~0.31 plateau vs teacher ~0.53).

| bucket | ref_fp8 top1 | parent top1 | ref_exact top1 |
|--------|-------------:|------------:|---------------:|
| [0,64) | 0.810 | 0.698 | 0.794 |
| [64,128) | 0.922 | 0.641 | 0.859 |
| [384,448) | 0.766 | 0.625 | 0.797 |
| [448,512) | 0.484 | 0.312 | 0.484 |
| [512,576) | 0.438 | 0.219 | 0.469 |
| [960,1024) | 0.540 | 0.302 | 0.540 |

## Teacher harness status

The torch residual/PPL harness is now **production tooling for Gates 6–9
calibration**, not a throwaway diagnostic:

- Verified teacher: PPL 4.693 (fp8) / 4.624 (exact) on `tokens.bin`
- Emits `HFPLOG01` `.plog` consumable by `ds4_parent_kld` / fine-scan
- ~634 s / 1024 tokens at 4.7 GiB peak on a 192 GiB card — headroom exists
  (cache dequantized layers) when Gate 8 needs 8K–32K; do not treat 634 s as a floor
- Land all harness changes in the LOCAL worktree; keep this README current

## Head path closed (stop)

Host + GPU head path match torch at the BF16-act floor on identical residuals
(`HEAD_PATH_CONTENT.md`, verdict `GPU_HEAD_AT_FLOOR`). Head does **not**
explain PPL 59.5. Per scope decision, `parent/*` investigation stops here;
deep-layer bisect not pursued. **Do not use `parent/*` as a calibration
reference** -- marker on `src/parent/mod.rs` and
`docs/investigations/2026-08-02-ds4-parent-not-calibration-ref.md`.

Teacher for Gates 6-9 remains this harness (`ref_fp8_*.plog`).
