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
and residual-content cosine drift at depth still needs an **above-floor** cause
somewhere else — or a logits/head path defect that residual cosine understates.

## Assets

- `ref_ppl_e2e.py`
- `artifacts/ref_ppl_1024/ref_ppl_summary.json`
- remote plogs: `/tmp/ref_ppl_1024/ref_fp8_1024.plog`, `ref_exact_1024.plog`
  (0.53 GiB each, HFPLOG01) — ready for `ds4_parent_kld` vs quant plogs
