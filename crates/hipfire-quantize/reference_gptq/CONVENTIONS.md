# GPTQ convention hypothesis list

Written **before** equivalence runs. Each row is a place a silent transpose,
scale factor, or off-by-one can hide. Hipfire answers are from reading the
code listed in the Evidence column — not from matching the reference.

Paper / DASLab answers are from Frantar et al. arXiv:2210.17323 and
https://github.com/IST-DASLab/gptq/blob/main/gptq.py (`add_batch`,
`fasterquant`).

| # | Convention | Paper / DASLab GPTQ | Hipfire answer | Evidence |
|---|---|---|---|---|
| 1 | Hessian definition | `H = (2/N) X X^T` with `X` shaped `[K, N]` (features × samples). Factor 2 from `∂²/∂W² ‖WX−QX‖²`. | **HFHS:** `H = (1/N) X^T X` — **no factor 2**. Accumulator does `H += x.T @ x` then `H/n_tokens`. **E8H1:** raw `sum_t x_b x_b^T` per 256-block, **no `/N`**, no factor 2. | `scripts/collect_hessian.py` `HessianAccumulator`; `collect_e8_hessian::BlockHessian`; `parent/hessian.rs` |
| 2 | Hessian normalisation by sample count | Yes — running mean in `add_batch` ends at `(2/N) XX^T`. | HFHS: yes (`/ n_tokens`). E8H1: **no** (raw sum; damping uses `mean(diag)` so scale still cancels in ratios if damp is fractional). | same |
| 3 | Weight matrix orientation | `W` is `[rows=out, cols=in]` = `nn.Linear.weight`. Columns are input channels; Hessian is over columns. | Same: row-major `M×K` with `K = in_features`. GPTQ walks **columns**. | `gptq_column_sequential` |
| 4 | Damping fraction | `percdamp=0.01`; `damp = 0.01 * mean(diag(H))` added to diagonal **after** any actorder permute, **before** Cholesky. | **e8_gptq:** `LAMBDA=0.01`, `damp = LAMBDA * mean(diag)` — paper-correct. **gptq.rs:** takes `initial_damp` as an **absolute** addend; `clamped_initial_damp` floors at `eps*mean(diag)`. Callers must pre-multiply by `mean(diag)` themselves. Adaptive `×10` up to `max_damp_multiplier * mean(diag)`. | `e8_gptq.rs:LAMBDA`; `gptq.rs:clamped_initial_damp`, `cholesky_with_adaptive_damping` |
| 5 | Damping before vs after scaling | Damp on the Hessian that enters Cholesky (post any AWQ/FWHT transform). | Same intent: damp on `H_target` in the GPTQ basis (post-AWQ rescale + FWHT similarity). | `gptq_pipeline_mq4g256` |
| 6 | Inverse-Cholesky form | `H = chol(H)`; `H = chol_inv(H)`; `H = chol(H, upper=True)` → `U` with **`U^T U = H^{-1}`**. OBS uses `U[j,k]/U[j,j]`. | Same invariant after 2026-05-14 fix: `compute_damped_inv_cholesky_upper` returns `U = L_HI^T` so `U^T U = H_inv`. Prior bug returned `L_H^{-T}` (`U U^T = H_inv`) and regressed quality. | `gptq.rs` doc on `compute_damped_inv_cholesky_upper`; test `compute_damped_inv_cholesky_upper_satisfies_identity` |
| 7 | Act-order | Optional. `perm = argsort(diag(H), descending)`; permute **both** `W` columns and `H`; after loop, `Q = Q[:, invperm]`. | **WEIGHT-mode actorder always on** in `gptq_column_sequential`. Sort by descending `diag(H)`. Does **not** permute weight storage: `perm[step]=original_col`; `U` on `P^T H P`; writes stay in original column order → **no invperm** needed (and no `g_idx` in the container). | `weight_mode_actorder`, `gptq_column_sequential` |
| 8 | Block size (OBS tiling) | Default `blocksize=128`; within-block column loop + trailing `W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]`. | `gptq.rs`: naive full-K column loop (blocksize comment says 128 is a follow-up). `e8_gptq`: **block-diagonal-256** Hessian, group size 8 (E8), feedback only inside the 256-block. | `gptq_column_sequential`; `ldlq_row_block` |
| 9 | Error feedback update | `W[:, j:] -= err[:, None] * Hinv_U[j, j:]` with `err = (w-q)/Hinv_U[j,j]`. | Same formula using `U[step, next]` into residual columns (original indexing). e8 path: per-column residual times feedback row, with `V_CLAMP=12` runaway guard and per-column cap `6*s`. | `gptq_column_sequential` Phase B; `ldlq_row_block` |
| 10 | Quantizer grid freeze | Official: `find_params` once before loop (unless `groupsize` retunes). | Frozen per-256-block `(scale, min_val)` from **pre-GPTQ** weights (`compute_frozen_block_grids`). e8: frozen row scale + E4M3 block scales before LDLQ. | `compute_frozen_block_grids`; e8 header comment |
| 11 | Dead columns | `diag(H)==0` → set `H[d,d]=1`, `W[:,d]=0`. | Not explicitly mirrored in `gptq_column_sequential`; Cholesky adaptive damp is the recovery path. e8: `mean_diag<=0` → RTN fallback. | DASLab `fasterquant`; `block_feedback` |
| 12 | Arithmetic domain | Official CUDA path often f32/tf32-off; Cholesky in float. | Hipfire Cholesky + OBS in **f64**. HFHS payload f32 (promoted at use). E8H1 payload f32 (accum f64→store f32). | module docs |
| 13 | E8H1 file layout | n/a (paper has no block file). | Magic `u32 LE = 0x45384831` (`E8H1`), `n_blocks:u32`, `K:u32`, then `n_blocks*256*256` f32 LE row-major. `K == n_blocks*256`. | `parent/hessian.rs` `E8H1_MAGIC`, `HBLK_HEADER_BYTES=12`; `load_hessian_blocks` |
| 14 | HFHS file layout | n/a. | Magic `b"HFHS"`, ver=1, `n_tensors:u64`, reserved=0; records `name_len|name|expert_idx|K|dtype_flag|payload`. dtype 1=f32, 2=f64. Name **without** `.weight`. | `docs/plans/gptq-hessian-format.md`; `hessian_io.rs` |

## Scale invariance note (critical for Gate 9)

OBS ratios `H_inv[j,k] / H_inv[j,j]` are invariant under `H → c H` for `c>0`,
**provided** damping scales the same way (`damp ∝ mean(diag(H))`). Therefore:

- HFHS missing factor 2 is **benign for GPTQ weights** if damp is fractional.
- E8H1 missing `/N` is **benign** under the same rule (`e8_gptq` multiplies
  `LAMBDA * mean(diag)` on the raw-sum H).
- Passing absolute `damp=0.01` into `gptq.rs` when `mean(diag) ≫ 1` is **not**
  equivalent — that is a real footgun and is probed in Stage 2.

## What this oracle cannot protect

See the fault-injection table in `README.md` / `last_run.json` after a harness
run. Any injection listed as MISSED is a blind spot: do not treat a clean
PASS as proof against that defect class.
