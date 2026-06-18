# RoughQuant importance generality — 9B confirms the 0.8B finding

Date: 2026-06-18. Box: gfx1151 / Strix Halo.

## Question

Task #9 (ablation oracle on `qwen3.5-0.8b`) established that **diag(H) residual-channel
energy is a near-optimal cheap importance selector** — voiding the highest-diag
channels does the most KLD damage, the tail is ~harmless, so "protect the head"
is sound and only a bounded tail gain is left on the table. Does this generalize
beyond 0.8B / 1024-dim?

## Method

Re-ran the *same* ablation oracle (`scripts/roughquant_ablation_oracle.sh`, now
parameterized) on `qwen3.5-9B` (dense, residual dim 4096):

- Hessian: native collector → `~/.hipfire/calib/qwen3.5-9b.calib.hfq` (248 dense
  Hessians, 256 calib tokens, `diag(Σxxᵀ)==Σx²` consistent). Read by the
  quantizer via the HFQM `hessian_io` reader (HFHS retired).
- diag rank-map: `HIPFIRE_RQ4_DUMP_RANK=1` (resid_energy, descending).
- For 14 ranks sampled head→tail: void exactly that one residual channel
  (`HIPFIRE_RQ4_BULK=void HIPFIRE_RQ4_VOID_ONLY=<ch>`, all else exact bf16),
  quantize `roughquant4-sim`, measure teacher-forced top-K KLD vs the bf16 ref
  (CTX 512).

## Result

| diag rank | channel | diag_energy | ablation_KLD |
|----------:|--------:|------------:|-------------:|
| 0    | 3994 | 2.409e5 | **5.948** |
| 1    | 310  | 1.140e4 | 0.1213 |
| 2    | 4042 | 8.850e3 | 0.0125 |
| 4    | 253  | 5.438e3 | 0.0491 |
| 8    | 2028 | 2.636e3 | 0.0218 |
| 16   | 1089 | 1.353e3 | 0.0076 |
| 64   | 2653 | 5.941e2 | 0.0025 |
| 256  | 2854 | 2.284e2 | 0.0021 |
| 1024 | 3064 | 8.711e1 | 0.00106 |
| 2048 | 481  | 6.690e1 | 0.00045 |
| 3072 | 2626 | 5.951e1 | 0.00101 |
| 4032 | 2253 | 5.062e1 | 0.00048 |
| 4090 | 1862 | 4.699e1 | 0.00030 |
| 4095 | 4029 | 4.505e1 | 0.00056 |

## Verdict: GENERALIZES

1. **One hyper-dominant outlier channel.** Rank 0 (channel 3994) has ~20× the
   diag energy of rank 1 and voiding it costs KLD 5.95 vs 0.12 for the next —
   the "massive activation" phenomenon, now confirmed at 9B / 4096-dim, same
   shape as 0.8B.
2. **diag energy strongly predicts ablation damage.** Broadly monotonic over 4
   orders of magnitude; the tail (rank ≥1024) is ~harmless (KLD ~3e-4–1e-3), so
   tail channels are genuinely low-importance.
3. **Near-optimal, not perfect.** Local inversions in the mid/head (rank 4 KLD
   0.049 > rank 2 KLD 0.012) confirm a *bounded* tail gain over the pure diag
   selector — matching the #9 conclusion. Nothing here motivates replacing diag
   as the production selector.

This is also the first cross-model exercise of the full native pipeline end to
end: collector → `.calib.hfq` → quantizer LDLQ/roughquant Hessian read (HFQM),
on a model other than 0.8B.

## Repro

```
MODEL=/srv/huggingface/models--Qwen--Qwen3.5-9B/snapshots/<snap> \
BF16_MODEL=~/.hipfire/models/qwen3.5-9b-bf16.hfq \
HESS=~/.hipfire/calib/qwen3.5-9b.calib.hfq \
DMODEL=4096 CTX=512 \
RANKS="0 1 2 4 8 16 64 256 1024 2048 3072 4032 4090 4095" \
REF=/tmp/bf16-9b.pkld RANKMAP=/tmp/diag_rank_9b.tsv OUT=/tmp/ablation_oracle_9b.tsv \
scripts/roughquant_ablation_oracle.sh
```
