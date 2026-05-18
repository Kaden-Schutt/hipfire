//! GPTQ column-sequential quantization for hipfire's MQ4G256 wire format.
//!
//! Phase A Stage B per `docs/plans/gptq.md` v2. Consumes per-tensor input
//! Hessians from `scripts/collect_hessian.py` (read via `hessian_io::HessianSidecar`),
//! produces MQ4G256 codewords optimized for activation-aware reconstruction.
//!
//! ## Architecture summary
//!
//! 1. **`transform_hessian_for_gptq`** — given the unscaled Hessian
//!    `H_unrot = E[x · x^T]` and the AWQ scale vector `s`, produce
//!    `H_target = FWHT_per_256_similarity( diag(1/s) · H_unrot · diag(1/s) )`,
//!    i.e. the Hessian in the actual coordinate system the matmul kernel
//!    operates in.
//!
//! 2. **`compute_frozen_block_grids`** — pre-compute per-256-block
//!    `(scale, min_val)` pairs from the FWHT-rotated, AWQ-scaled
//!    weights BEFORE running GPTQ. Frozen through the loop to avoid the
//!    circular dependency where post-GPTQ weights would change the
//!    per-block min/max (per GLM5 C1 in the synthesis review).
//!
//! 3. **`gptq_column_sequential`** — main loop. WEIGHT-mode actorder
//!    (sort columns by `diag(H_target)` descending), block-wise OBS
//!    column update via FP64 Cholesky from `faer`, per-element
//!    asymmetric INT4 quantize using the frozen per-256-block grids.
//!
//! 4. **`compute_damped_inv_cholesky_upper`** — Cholesky-direct (per
//!    Frantar et al. 2210.17323 Algorithm 1): compute `U` such that
//!    `U · U^T = (P^T (H+λI) P)^-1`, where P is the WEIGHT-mode-actorder
//!    permutation. Avoids materializing the dense inverse + the O(K³)
//!    `solve(I)` back-substitution (the latter was single-threaded in
//!    faer 0.24 and dominated wall time at K=12288). Inversion of L is
//!    rayon-parallel column-wise. Defensive adaptive damping (10× per
//!    retry up to `max_damp_multiplier * mean(diag(H))`); if even the
//!    cap fails, returns `Err(SingularEvenWithMaxDamp)` and the caller
//!    skips GPTQ for that tensor (falls through to plain MQ4 in main.rs).
//!
//! All linear algebra is FP64 (per Claude M2 + GLM5 M2 reviews) — FP32
//! Cholesky on K=12288 with cond=1e6+ has zero effective precision.

#![cfg_attr(not(test), allow(dead_code))]  // suppress until main.rs wires it

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, Side};
use rayon::prelude::*;

/// Per-element asymmetric MQ4 quantize step.
///
/// Mirrors the formula in `quantize_mq4g256` (main.rs:566-567):
/// `q = round((w - min_val) / scale)` clamped to `[0, 15]`,
/// then `dequant = q * scale + min_val`. Returns the dequantized FP32
/// value (i.e. what the runtime sees as the effective weight).
///
/// `scale` and `min_val` are from the FROZEN per-256-block grid computed
/// before the GPTQ loop (per `compute_frozen_block_grids`).
#[inline]
pub fn quantize_mq4_element(w: f64, scale: f64, min_val: f64) -> f64 {
    if scale == 0.0 {
        return min_val;
    }
    let inv_scale = 1.0 / scale;
    let q = ((w - min_val) * inv_scale + 0.5).floor().clamp(0.0, 15.0);
    q * scale + min_val
}

/// Variant of `quantize_mq4_element` that also reports the clamp state of
/// the pre-clamp grid index. Returns `(q_value, clamp_state)` where
/// `clamp_state` is:
///   - `-1` if `floor((w - min_val) / scale + 0.5) < 0` (clamped to 0),
///   - `+1` if it `> 15` (clamped to 15),
///   - `0` if the value was in range.
///
/// Used by the GPTQ inner loop's clamp diagnostic — the frozen per-256-
/// block grid is fit to the ORIGINAL weights, but OBS error compensation
/// can push the residual outside that range. When clamping fires, the
/// per-column quantization error contract (`|err| ≤ ½·scale`) is
/// violated, and the cascading OBS propagation in
/// `gptq_column_sequential` operates on an inflated error → quality
/// regression. Counting clamps per-tensor surfaces this case.
#[inline]
pub fn quantize_mq4_element_with_clamp(w: f64, scale: f64, min_val: f64) -> (f64, i8) {
    if scale == 0.0 {
        return (min_val, 0);
    }
    let inv_scale = 1.0 / scale;
    let q_raw = ((w - min_val) * inv_scale + 0.5).floor();
    let clamp_state: i8 = if q_raw < 0.0 { -1 } else if q_raw > 15.0 { 1 } else { 0 };
    let q = q_raw.clamp(0.0, 15.0);
    (q * scale + min_val, clamp_state)
}

/// FP64 Cholesky of `H + damp * I` with adaptive damping fallback.
///
/// Returns `(L, effective_damp)` where `L` is the lower-triangular
/// Cholesky factor and `effective_damp` is the damping value that
/// actually made `H + damp*I` PSD-decomposable. If even
/// `damp = max_damp_multiplier * mean(diag(H))` fails, returns
/// `Err(CholeskyError::SingularEvenWithMaxDamp)`.
///
/// Per the GPTQ paper, damping is critical for numerical stability —
/// the Hessian's null space (low-activation channels) makes naive
/// Cholesky fail without it.
///
/// Provided for testability; production GPTQ uses
/// `compute_damped_inv_cholesky_upper` (returns upper-tri U with
/// U^T·U = H^-1 — the Frantar-Algorithm-1 invariant).
pub fn cholesky_with_adaptive_damping(
    h: &Mat<f64>,
    initial_damp: f64,
    max_damp_multiplier: f64,
) -> Result<(Mat<f64>, f64), CholeskyError> {
    let k = h.nrows();
    assert_eq!(h.nrows(), h.ncols(), "Hessian must be square");
    let diag_mean: f64 = (0..k).map(|i| h[(i, i)]).sum::<f64>() / k as f64;

    let mut damp = clamped_initial_damp(initial_damp, diag_mean);
    let damp_cap = max_damp_multiplier * diag_mean;
    loop {
        let mut a = h.clone();
        for i in 0..k {
            a[(i, i)] += damp;
        }
        match a.llt(Side::Lower) {
            Ok(decomp) => {
                let l_ref = decomp.L();
                let mut l = Mat::<f64>::zeros(k, k);
                for j in 0..k {
                    for i in j..k {
                        l[(i, j)] = l_ref[(i, j)];
                    }
                }
                return Ok((l, damp));
            }
            Err(_) => {
                if damp >= damp_cap {
                    return Err(CholeskyError::SingularEvenWithMaxDamp {
                        max_damp: damp,
                        k,
                        diag_mean,
                    });
                }
                damp = (damp * 10.0).min(damp_cap);
            }
        }
    }
}

/// Snap `initial_damp` away from zero relative to the Hessian's scale.
/// Without this, `damp *= 10` stays at 0 forever when the caller passes
/// zero against a singular matrix. The clamp is inert for any practical
/// non-zero `initial_damp` (it lives at the `f64::EPSILON * diag_mean`
/// floor), so well-conditioned Cholesky outputs don't shift measurably.
#[inline]
fn clamped_initial_damp(initial_damp: f64, diag_mean: f64) -> f64 {
    initial_damp.max(f64::EPSILON * diag_mean.max(1.0))
}

/// Adaptive-damping search + the upper Cholesky factor of `H_inv` such
/// that `U^T · U = H_inv` — the form Frantar et al. 2210.17323 Algorithm
/// 1 uses for the OBS error-propagation cascade.
///
/// Returns `(U, effective_damp)` where `U` is K×K upper-triangular with
/// `U^T · U = (P^T (H + λI) P)^-1`, P the permutation in `perm` (identity
/// if None). `effective_damp` is the damping value that worked.
///
/// **Why `U^T · U = H_inv` and not `U · U^T = H_inv`** (a bug fixed
/// 2026-05-14). The seminal GPTQ algorithm propagates each step's
/// quantization error using `U[step, next_step] / U[step, step]` of
/// THIS upper Cholesky factor. The reason is the *Schur-complement
/// submatrix property* — `(U[i:K, i:K])^T · (U[i:K, i:K]) =
/// Schur_complement(H_inv, [0:i, 0:i])` — which makes the trailing
/// rows of U the right factor of the residual Hessian for unprocessed
/// columns. The transpose-flipped variant `U · U^T = H_inv` (which an
/// earlier hipfire iteration returned via `L_H^{-T}`) IS a valid
/// factorization of `H_inv`, but its trailing submatrix does NOT
/// satisfy the Schur property — so the row-j ratios systematically
/// differ from `H_inv[j, k] / H_inv[j, j]` by factors of 1.5–3.5×
/// (verified numerically against direct dense H_inv). Using
/// `L_H^{-T}` in the OBS loop produced GPTQ quality REGRESSIONS at
/// every model size we tested (0.8B mq4-awq-gptq+Q8conv1d 0.198 vs
/// AWQ-alone 0.137). Bug isolated by external review on 2026-05-14;
/// fix lands here.
///
/// Computation:
///   1. `L = chol(H + λI, lower)` so `L · L^T = H + λI`
///   2. `L_inv = L^-1` (lower-tri, by forward sub)
///   3. `H_inv = L_inv^T · L_inv` (materialize K×K, symmetric)
///   4. `L_HI = chol(H_inv, lower)` so `L_HI · L_HI^T = H_inv`
///   5. Return `U = L_HI^T` (upper-tri, `U^T · U = L_HI · L_HI^T = H_inv` ✓)
///
/// Cost vs prior `L_H^{-T}` form: +K²/2 storage (H_inv), +K³/3 flops
/// (matmul + second Cholesky). At K=12288 that's ~1.2 GB + ~2 minutes
/// extra per-tensor wall — acceptable for correctness.
pub fn compute_damped_inv_cholesky_upper(
    h: &Mat<f64>,
    perm: Option<&[usize]>,
    initial_damp: f64,
    max_damp_multiplier: f64,
) -> Result<(Mat<f64>, f64), CholeskyError> {
    let k = h.nrows();
    assert_eq!(h.nrows(), h.ncols(), "Hessian must be square");
    if let Some(p) = perm {
        assert_eq!(p.len(), k, "permutation length must equal Hessian dim");
    }

    // Materialize H_eff = P^T H P (or H itself when perm is None).
    // Cholesky's column order must match the GPTQ inner loop's processing
    // order; the upper-triangular U is only "upper" relative to THIS order.
    let h_eff: Mat<f64> = if let Some(p) = perm {
        Mat::<f64>::from_fn(k, k, |i, j| h[(p[i], p[j])])
    } else {
        h.clone()
    };

    let diag_mean: f64 = (0..k).map(|i| h_eff[(i, i)]).sum::<f64>() / k as f64;
    let mut damp = clamped_initial_damp(initial_damp, diag_mean);
    let damp_cap = max_damp_multiplier * diag_mean;

    loop {
        let mut a = h_eff.clone();
        for i in 0..k {
            a[(i, i)] += damp;
        }
        match a.llt(Side::Lower) {
            Ok(decomp) => {
                // Materialize L = lower Cholesky of (H_eff + λI), so L·L^T = H_eff+λI.
                let l_view = decomp.L();
                let mut l_mat = Mat::<f64>::zeros(k, k);
                for j in 0..k {
                    for i in j..k {
                        l_mat[(i, j)] = l_view[(i, j)];
                    }
                }

                // Invert L (lower-triangular): each column j of L_inv is
                // the solution to L · x = e_j by forward substitution.
                // Columns are independent → rayon-parallel.
                //
                // For column j:
                //   x[i] = 0           for i < j  (lower-tri)
                //   x[j] = 1 / L[j, j]
                //   x[i] = -(Σ_{m=j..i} L[i, m] · x[m]) / L[i, i]   for i > j
                let l_mat_ref = &l_mat;
                let l_inv_cols: Vec<Vec<f64>> = (0..k).into_par_iter().map(|j| {
                    let mut col = vec![0.0_f64; k];
                    let l_jj = l_mat_ref[(j, j)];
                    if l_jj <= 0.0 {
                        return col;  // defensive: should not happen after successful LLT
                    }
                    col[j] = 1.0 / l_jj;
                    for i in (j + 1)..k {
                        let mut s = 0.0;
                        for m in j..i {
                            s += l_mat_ref[(i, m)] * col[m];
                        }
                        col[i] = -s / l_mat_ref[(i, i)];
                    }
                    col
                }).collect();

                // Step 3: materialize H_inv = L_inv^T · L_inv (symmetric, K×K).
                //
                // `l_inv_cols[j]` holds column j of L_inv (lower-tri), i.e.
                // L_inv[i, j] = l_inv_cols[j][i] for i >= j, 0 otherwise.
                // (L_inv^T · L_inv)[i, j] = Σ_m L_inv[m, i] · L_inv[m, j].
                // L_inv lower-tri ⇒ L_inv[m, i] != 0 only when m >= i, and
                // L_inv[m, j] != 0 only when m >= j; intersection m >= max(i,j).
                // Result is symmetric. Per-row parallel via rayon.
                let l_inv_cols_ref = &l_inv_cols;
                let h_inv_upper_rows: Vec<Vec<f64>> = (0..k).into_par_iter().map(|i| {
                    let mut row = vec![0.0_f64; k];
                    for j in i..k {  // upper triangle (incl. diagonal)
                        let mut s = 0.0_f64;
                        // m ranges over max(i,j)=j .. k (since j >= i in this loop)
                        for m in j..k {
                            s += l_inv_cols_ref[i][m] * l_inv_cols_ref[j][m];
                        }
                        row[j] = s;
                    }
                    row
                }).collect();

                let mut h_inv = Mat::<f64>::zeros(k, k);
                for i in 0..k {
                    for j in i..k {
                        let v = h_inv_upper_rows[i][j];
                        h_inv[(i, j)] = v;
                        if i != j { h_inv[(j, i)] = v; }
                    }
                }

                // Step 4: second Cholesky on H_inv → L_HI (lower-tri),
                // L_HI · L_HI^T = H_inv. Should never fail by construction
                // (H_inv is SPD), but propagate any failure as the adaptive
                // damping cascade would for the outer Cholesky.
                //
                // Step 5: U = L_HI^T (upper-tri). U^T · U = L_HI · L_HI^T
                // = H_inv, the correct Frantar-Algorithm-1 form.
                //
                // The decomp owns the underlying buffer; bind it for the
                // entire scope of the materialization below so .L() stays
                // valid while we read entries into our owned `u`.
                let h_inv_decomp = match h_inv.llt(Side::Lower) {
                    Ok(d) => d,
                    Err(_) => {
                        // Should not happen. If it does, signal failure
                        // with the same SingularEvenWithMaxDamp variant the
                        // outer Cholesky uses — caller falls back to plain
                        // MQ4 RTN for this tensor (see main.rs:4336-4339).
                        return Err(CholeskyError::SingularEvenWithMaxDamp {
                            max_damp: damp, k, diag_mean,
                        });
                    }
                };
                let l_hi_view = h_inv_decomp.L();
                let mut u = Mat::<f64>::zeros(k, k);
                for j in 0..k {
                    for i in 0..=j {
                        u[(i, j)] = l_hi_view[(j, i)];  // transpose: U[i,j] = L_HI[j,i]
                    }
                }

                return Ok((u, damp));
            }
            Err(_) => {
                if damp >= damp_cap {
                    return Err(CholeskyError::SingularEvenWithMaxDamp {
                        max_damp: damp,
                        k,
                        diag_mean,
                    });
                }
                damp = (damp * 10.0).min(damp_cap);
            }
        }
    }
}

#[derive(Debug)]
pub enum CholeskyError {
    SingularEvenWithMaxDamp {
        max_damp: f64,
        k: usize,
        diag_mean: f64,
    },
}

impl std::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CholeskyError::SingularEvenWithMaxDamp { max_damp, k, diag_mean } => write!(
                f,
                "Cholesky of K={k} Hessian failed even at damp={max_damp:.6e} (diag mean={diag_mean:.6e}); skip GPTQ for this tensor"
            ),
        }
    }
}

impl std::error::Error for CholeskyError {}

/// Order-of-magnitude condition number estimate via diag(H+λI) min/max.
///
/// Returns a *lower bound* on the true condition number — a real
/// estimate would need the full eigenvalue decomposition. This is a
/// cheap guard against pathological Hessians (e.g. truncated download,
/// model weight corruption) without paying O(K³) for a real SVD.
///
/// For the actual decision "is this Hessian usable for GPTQ?",
/// `cholesky_with_adaptive_damping` is the better signal — it fails
/// definitively when the matrix is too singular.
pub fn diag_condition_lower_bound(h: &Mat<f64>, damp: f64) -> f64 {
    let k = h.nrows();
    let mut min_d = f64::INFINITY;
    let mut max_d = f64::NEG_INFINITY;
    for i in 0..k {
        let d = h[(i, i)] + damp;
        if d < min_d {
            min_d = d;
        }
        if d > max_d {
            max_d = d;
        }
    }
    if min_d <= 0.0 {
        f64::INFINITY
    } else {
        max_d / min_d
    }
}

/// Apply per-256-block FWHT similarity transform to a K×K matrix in-place.
///
/// For each block-pair `(b_row, b_col)` of 256 consecutive K-axis
/// channels, applies `H'[b_row, b_col] = H_256_FWHT · H[b_row, b_col] · H_256_FWHT^T`.
/// Because `H_256_FWHT` is orthogonal (and the hipfire kernel applies a
/// `1/sqrt(256)` normalization), this is `<H_256, H, H_256^T>` exactly
/// — a similarity transform that doesn't change the matrix's spectrum,
/// only its basis.
///
/// `signs1`, `signs2` are the per-pre/post-FWHT sign vectors that
/// hipfire's kernel applies (gen_fwht_signs with seeds 42 and 1042 —
/// see `quantize_mq4g256`).
///
/// This is the FWHT half of the Hessian transformation chain
/// (Topic 1 + Topic 2 of the v2 plan).
pub fn fwht_similarity_per_256(
    h: &mut Mat<f64>,
    signs1: &[f64],
    signs2: &[f64],
) {
    let k = h.nrows();
    assert_eq!(h.nrows(), h.ncols(), "FWHT similarity requires square matrix");
    assert!(k % 256 == 0, "K={k} must be divisible by 256");
    assert_eq!(signs1.len(), 256);
    assert_eq!(signs2.len(), 256);
    let n_blocks = k / 256;

    // Stage 1: apply FWHT to each 256-element ROW slice (in-place per row)
    // for every row of H. This computes H' = H · H_256_FWHT^T block-by-
    // block on the column axis.
    for row in 0..k {
        for bc in 0..n_blocks {
            let mut buf = [0.0_f64; 256];
            for j in 0..256 {
                buf[j] = h[(row, bc * 256 + j)];
            }
            fwht_256_inplace_f64(&mut buf, signs1, signs2);
            for j in 0..256 {
                h[(row, bc * 256 + j)] = buf[j];
            }
        }
    }

    // Stage 2: apply FWHT to each 256-element COL slice for every col of H'.
    // Computes H'' = H_256_FWHT · H' = (H_256_FWHT · H · H_256_FWHT^T).
    for col in 0..k {
        for br in 0..n_blocks {
            let mut buf = [0.0_f64; 256];
            for i in 0..256 {
                buf[i] = h[(br * 256 + i, col)];
            }
            fwht_256_inplace_f64(&mut buf, signs1, signs2);
            for i in 0..256 {
                h[(br * 256 + i, col)] = buf[i];
            }
        }
    }
}

/// FWHT-256 in FP64, in-place, matching `cpu_fwht_256` in main.rs
/// (which is FP32). Same sign convention, same 1/16 = 1/sqrt(256)
/// normalization at the end — keeps the round-trip identity:
/// `<FWHT(a), FWHT(b)> = <a, b>` for orthogonal FWHT.
fn fwht_256_inplace_f64(x: &mut [f64; 256], signs1: &[f64], signs2: &[f64]) {
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1usize;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    const SCALE: f64 = 1.0 / 16.0;
    for i in 0..256 {
        x[i] *= SCALE * signs2[i];
    }
}

/// Apply AWQ rescaling to a Hessian: `H' = diag(1/s) · H · diag(1/s)`.
///
/// Per Gemini's review finding (gptq_plan_rev_synthesis.md Topic 1):
/// when the runtime divides activations by `s` before the matmul, the
/// effective Hessian seen by the matmul kernel is `E[(x/s)(x/s)^T] =
/// diag(1/s) · E[xx^T] · diag(1/s)`. GPTQ must optimize against THIS
/// Hessian, not the unscaled one.
///
/// For non-AWQ tensors (Stage B widened coverage per GLM5 M5), pass
/// `s = [1.0; K]` — the function is then a no-op (multiplies by 1
/// row-wise + col-wise).
pub fn apply_awq_rescaling(h: &mut Mat<f64>, awq_scales: &[f64]) {
    let k = h.nrows();
    assert_eq!(h.nrows(), h.ncols());
    assert_eq!(awq_scales.len(), k);
    for &s in awq_scales {
        assert!(s > 0.0, "AWQ scales must be strictly positive (got {s})");
    }
    for i in 0..k {
        let inv_i = 1.0 / awq_scales[i];
        for j in 0..k {
            let inv_j = 1.0 / awq_scales[j];
            h[(i, j)] *= inv_i * inv_j;
        }
    }
}

/// Symmetrize a square matrix in place: `M[i,j] = M[j,i] = (M[i,j] + M[j,i]) / 2`.
/// Used to scrub the FP-error asymmetry that accumulates across the
/// row-pass + col-pass of `fwht_similarity_per_256` (which is exactly
/// symmetric in exact arithmetic but drifts by O(ε·K·log K) at K=12288).
pub fn symmetrize_in_place(h: &mut Mat<f64>) {
    let k = h.nrows();
    assert_eq!(h.nrows(), h.ncols());
    for i in 0..k {
        for j in (i + 1)..k {
            let avg = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = avg;
            h[(j, i)] = avg;
        }
    }
}

/// Per-256-block (scale, min_val) pair, frozen before the GPTQ loop.
#[derive(Clone, Copy, Debug)]
pub struct BlockGrid {
    pub scale: f64,
    pub min_val: f64,
}

/// Apply per-256-block FWHT to a row-major M×K f64 weight matrix in place.
///
/// Mirrors the per-block FWHT that `quantize_mq4g256` (main.rs:553-554)
/// does internally, but in FP64 so it composes with GPTQ's FP64 pipeline.
/// Used by the GPTQ pipeline to rotate weights once at the start of the
/// per-tensor work — Option β per the v2 plan §2.2.
pub fn apply_fwht_per_256_to_weights_f64(
    weights: &mut [f64],
    m: usize,
    k: usize,
    signs1: &[f64],
    signs2: &[f64],
) {
    assert_eq!(weights.len(), m * k);
    assert_eq!(k % 256, 0, "K={k} must be divisible by 256 for FWHT-256");
    assert_eq!(signs1.len(), 256);
    assert_eq!(signs2.len(), 256);
    let blocks_per_row = k / 256;
    for r in 0..m {
        for b in 0..blocks_per_row {
            let start = r * k + b * 256;
            let mut buf = [0.0_f64; 256];
            buf.copy_from_slice(&weights[start..start + 256]);
            fwht_256_inplace_f64(&mut buf, signs1, signs2);
            weights[start..start + 256].copy_from_slice(&buf);
        }
    }
}

/// Pack rotated FP64 weights into MQ4G256 INT4 codewords using the FROZEN
/// per-256-block grids. Output byte layout matches `quantize_mq4g256`
/// exactly: per 256-block, 4-byte FP32 scale + 4-byte FP32 min_val +
/// 128 bytes of packed 4-bit codewords (2 per byte).
///
/// Used as the final packing step of the GPTQ pipeline. The input
/// `weights` are post-FWHT (rotated by the same FWHT that the existing
/// MQ4 GEMV kernel rotates `x` against at inference). Output is byte-
/// equivalent to what `quantize_mq4g256` would have produced, except
/// the codewords reflect GPTQ's Hessian-aware column updates instead
/// of plain RTN on the same rotated input.
pub fn pack_mq4g256_from_rotated_f64(weights: &[f64], grids: &[BlockGrid]) -> Vec<u8> {
    let n = weights.len();
    assert_eq!(n % 256, 0);
    let n_blocks = n / 256;
    assert_eq!(grids.len(), n_blocks);

    let block_bytes = 136usize;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let grid = grids[b];
        let scale_f32 = grid.scale as f32;
        let min_f32 = grid.min_val as f32;
        let inv_scale = if grid.scale > 0.0 { 1.0 / grid.scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_f32.to_le_bytes());

        let group = &weights[b * 256..(b + 1) * 256];
        for i in 0..128 {
            // Round-half-up to MQ4 grid (matches quantize_mq4g256 main.rs:568).
            let lo_q = (((group[2 * i] - grid.min_val) * inv_scale) + 0.5).floor() as i32;
            let hi_q = (((group[2 * i + 1] - grid.min_val) * inv_scale) + 0.5).floor() as i32;
            let lo = lo_q.clamp(0, 15) as u8;
            let hi = hi_q.clamp(0, 15) as u8;
            output[out_off + 8 + i] = lo | (hi << 4);
        }
    }

    output
}

/// High-level GPTQ pipeline for one MQ4G256 tensor.
///
/// Input is the post-AWQ-prescaled FP32 weight matrix (row-major M × K),
/// plus the unrotated/unscaled Hessian `H_unrot` from the sidecar, plus
/// the AWQ scale vector `s` (or `vec![1.0; K]` for non-AWQ tensors).
///
/// Performs the full quantize-time GPTQ chain:
///   1. AWQ-rescale H (no-op if s = 1)
///   2. FWHT-per-256 similarity transform on H → H_target in the basis
///      the matmul kernel actually operates in (Option β).
///   3. FWHT-per-256 on weights → W_rot in same basis.
///   4. Pre-compute FROZEN per-256-block grids from W_rot.
///   5. Run gptq_column_sequential on W_rot with H_target + frozen grids.
///   6. Pack post-GPTQ weights using the SAME frozen grids → MQ4 codewords.
///
/// Returns the packed MQ4G256 bytes (same layout as `quantize_mq4g256`).
/// On Cholesky failure even after adaptive damping, falls back to plain
/// `quantize_mq4g256` (with a warning passed via the `on_fallback` callback).
pub fn gptq_pipeline_mq4g256(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_unrot_f32: &[f32],     // K*K row-major
    awq_scales: &[f64],      // length K; pass [1.0; K] for non-AWQ
    signs1_f32: &[f32],
    signs2_f32: &[f32],
    initial_damp: f64,
    max_damp_multiplier: f64,
    tensor_name: &str,
) -> Result<Vec<u8>, CholeskyError> {
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_unrot_f32.len(), k * k);
    assert_eq!(awq_scales.len(), k);

    // Cast to f64 for the GPTQ pipeline. AWQ pre-scaling has already
    // been applied to weights upstream; we only need to rescale H here.
    let mut h = Mat::<f64>::from_fn(k, k, |i, j| h_unrot_f32[i * k + j] as f64);
    apply_awq_rescaling(&mut h, awq_scales);

    let signs1: Vec<f64> = signs1_f32.iter().map(|&v| v as f64).collect();
    let signs2: Vec<f64> = signs2_f32.iter().map(|&v| v as f64).collect();
    fwht_similarity_per_256(&mut h, &signs1, &signs2);

    // Defensive symmetrization. `F · diag(1/s) · H · diag(1/s) · F^T` is
    // symmetric in exact arithmetic but the row-pass and col-pass in
    // `fwht_similarity_per_256` accumulate FP error differently, so
    // (i,j) and (j,i) can drift. faer's `llt(Side::Lower)` ignores the
    // upper triangle but `gptq_column_sequential` reads `h_inv[(j, kk)]`
    // for arbitrary (j, kk) — silent asymmetry there corrupts OBS
    // propagation. Average them once here, cheap O(K²).
    symmetrize_in_place(&mut h);

    let mut weights = vec![0.0_f64; m * k];
    for (i, &w) in weights_f32.iter().enumerate() {
        weights[i] = w as f64;
    }
    apply_fwht_per_256_to_weights_f64(&mut weights, m, k, &signs1, &signs2);

    let frozen_grids = compute_frozen_block_grids(&weights);

    gptq_column_sequential(
        &mut weights,
        &h,
        m,
        k,
        &frozen_grids,
        initial_damp,
        max_damp_multiplier,
        tensor_name,
    )?;

    Ok(pack_mq4g256_from_rotated_f64(&weights, &frozen_grids))
}

/// Compute the FROZEN per-256-block grids from the FWHT-rotated, AWQ-scaled
/// weights — exactly the same per-block min/max scheme that
/// `quantize_mq4g256` uses in main.rs:554-559. Frozen through the GPTQ
/// loop to avoid the circular dependency where the post-GPTQ weights
/// would change the block's min/max (per GLM5 C1 in the synthesis review).
///
/// `weights_flat` is the row-major `M × K` weight matrix as a flat slice
/// of length `M * K`. Blocks are sequential 256-element chunks of this
/// flat buffer, matching the `for b in 0..n_blocks { group = data[b*256..]` }`
/// pattern in `quantize_mq4g256`.
pub fn compute_frozen_block_grids(weights_flat: &[f64]) -> Vec<BlockGrid> {
    let n = weights_flat.len();
    assert_eq!(n % 256, 0, "weight buffer length {n} must be divisible by 256");
    let n_blocks = n / 256;
    let mut grids = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let block = &weights_flat[b * 256..(b + 1) * 256];
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in block {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        grids.push(BlockGrid { scale, min_val });
    }
    grids
}

/// Map (row, original_col) of a weight matrix → its frozen-grid index.
///
/// In the row-major `M × K` flat layout, element `(row, col)` is at
/// flat index `row * K + col`, which lives in block `(row * K + col) / 256`.
/// `original_col` is the un-permuted column index (the permutation only
/// affects the GPTQ loop ORDER, not the storage layout).
#[inline]
fn block_idx_for(row: usize, original_col: usize, k_dim: usize) -> usize {
    (row * k_dim + original_col) / 256
}

/// WEIGHT-mode actorder: returns the permutation that orders the K
/// columns by descending `diag(H)`. Apply to both H and W (columns)
/// before the GPTQ loop, then un-apply to W after. Storage layout is
/// unchanged (no g_idx needed in the .hfq), satisfying the runtime's
/// "no kernel changes" constraint per the GPTQ plan §2.2.
///
/// Per the compressed-tensors `ActivationOrdering::WEIGHT` mode
/// (cf. gptq_plan_rev_synthesis.md Topic 3).
pub fn weight_mode_actorder(h_diag: &[f64]) -> Vec<usize> {
    let k = h_diag.len();
    let mut perm: Vec<usize> = (0..k).collect();
    // Sort indices by descending diag(H). Stable to keep deterministic order
    // for tied diagonals (matters for unit-test reproducibility).
    perm.sort_by(|&a, &b| h_diag[b].partial_cmp(&h_diag[a]).unwrap_or(std::cmp::Ordering::Equal));
    perm
}

/// Inverse permutation: `inverse[perm[i]] = i`.
pub fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Core GPTQ column-sequential update.
///
/// Mutates `weights_flat` (row-major M×K) in place: each column j is
/// snapped to the per-256-block grid (frozen pre-loop) with OBS error
/// compensation propagated to columns > j via the inverse Hessian.
///
/// **Inputs:**
/// - `weights_flat` — row-major FP64 weight matrix, length `M * K`.
///   Must already be in the GPTQ basis (post-AWQ-scaling, post-FWHT
///   if Option β; or pre-FWHT if Option α — see plan §2.2).
/// - `h_target` — K×K Hessian in the same basis as `weights_flat`
///   (transformed via `apply_awq_rescaling` + `fwht_similarity_per_256`).
/// - `m`, `k_dim` — matrix dimensions; `weights_flat.len() == m * k_dim`.
/// - `frozen_grids` — output of `compute_frozen_block_grids` on the
///   PRE-GPTQ weights; FROZEN through the loop (per GLM5 C1).
/// - `damp` — initial damping for Cholesky (typically `0.01 * mean(diag(H))`).
///
/// **Returns:** the effective damping used (after adaptive escalation),
/// for caller diagnostics + post-mortem.
///
/// **Algorithm:** standard GPTQ Algorithm 1 (Frantar et al., arXiv 2210.17323
/// §3.1). For each column j in WEIGHT-mode-actorder order:
///   1. Snap `weights[:, j]` to MQ4 grid (per-element, using frozen grids).
///   2. `err[:, j] = (w_orig[:, j] - w_q[:, j]) / H_inv[j, j]`
///   3. For each remaining column k > j: `weights[:, k] -= err[:, j] * H_inv[j, k]`
///
/// Naive O(K² · M) — suitable for K up to ~8K. For K=12288 the inter-tile
/// blocking optimization (paper §3.2 with `block_size=128`) is a follow-up.
pub fn gptq_column_sequential(
    weights_flat: &mut [f64],
    h_target: &Mat<f64>,
    m: usize,
    k_dim: usize,
    frozen_grids: &[BlockGrid],
    initial_damp: f64,
    max_damp_multiplier: f64,
    tensor_name: &str,
) -> Result<f64, CholeskyError> {
    assert_eq!(weights_flat.len(), m * k_dim, "weight shape mismatch");
    assert_eq!(h_target.nrows(), k_dim);
    assert_eq!(h_target.ncols(), k_dim);
    assert_eq!(frozen_grids.len(), (m * k_dim) / 256);

    // WEIGHT-mode actorder: sort columns by descending diag(H_target).
    // The Cholesky-direct upper factor U is computed on the PERMUTED
    // Hessian P^T H P; weights + frozen grids stay in original indexing
    // (perm[step] = original column index processed at step `step`).
    let h_diag: Vec<f64> = (0..k_dim).map(|i| h_target[(i, i)]).collect();
    let perm = weight_mode_actorder(&h_diag);

    // U is upper-tri K×K with U^T·U = (P^T(H+λI)P)^-1 — the
    // Frantar-Algorithm-1 form (fixed 2026-05-14; prior implementation
    // returned L_H^{-T} which satisfies U·U^T = H_inv instead, a
    // different upper-tri factor that breaks the Schur-complement
    // submatrix property OBS relies on). OBS inner loop below uses
    // U[step, step] as the divisor and U[step, next_step] as the
    // propagation weight; with this U, the ratio
    // `U[j, k] / U[j, j] = H_inv[j, k] / H_inv[j, j]` for the residual
    // Schur-complement Hessian — exactly the textbook GPTQ correction.
    // See `compute_damped_inv_cholesky_upper` doc for the full math.
    let (u, effective_damp) = compute_damped_inv_cholesky_upper(
        h_target, Some(&perm), initial_damp, max_damp_multiplier,
    )?;

    // Working copy of the post-quantize "residual" weights. We need to
    // keep the original values to compute the error for OBS propagation.
    // `weights_residual[row, col]` evolves as columns get processed and
    // future columns absorb the error compensation.
    let mut weights_residual: Vec<f64> = weights_flat.to_vec();

    // Clamp diagnostic — atomic counters incremented inside the rayon
    // closures. See `quantize_mq4_element_with_clamp` for rationale: the
    // frozen per-256-block grid is fit to ORIGINAL weights, but OBS error
    // propagation can push residuals outside the grid range. When the
    // pre-clamp grid index is < 0 or > 15, the clamp inflates per-column
    // error beyond GPTQ's ±½·scale assumption and the cascade amplifies.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let total_count = AtomicUsize::new(0);
    let clamps_above = AtomicUsize::new(0);
    let clamps_below = AtomicUsize::new(0);

    // Output: snapped values per (row, original_col). The outer column-
    // sequential pass is intrinsically serial (column step+1 depends on
    // column step's residual update), but both inner row-loops are
    // independent across rows — `par_chunks_mut` over `k_dim`-sized rows
    // gives M-way parallelism.
    for step in 0..k_dim {
        let j_orig = perm[step];
        let u_ss = u[(step, step)];
        if u_ss <= 0.0 {
            // Defensive: U's diagonal entries are reciprocals of L's diagonal
            // entries (positive for any SPD H). Hitting ≤0 means numerical
            // breakdown — skip this column, leave residual unchanged (rare).
            continue;
        }
        // Phase A: quantize column j_orig to the MQ4 grid + compute err_col,
        // parallel across rows. Each row writes its own disjoint
        // `weights_flat[row * k_dim + j_orig]` slot.
        let err_col: Vec<f64> = weights_flat
            .par_chunks_mut(k_dim)
            .zip(weights_residual.par_chunks(k_dim))
            .enumerate()
            .map(|(row, (out_row, res_row))| {
                let block_idx = block_idx_for(row, j_orig, k_dim);
                let grid = frozen_grids[block_idx];
                let w = res_row[j_orig];
                let (q, clamp_state) = quantize_mq4_element_with_clamp(w, grid.scale, grid.min_val);
                total_count.fetch_add(1, Ordering::Relaxed);
                if clamp_state < 0 {
                    clamps_below.fetch_add(1, Ordering::Relaxed);
                } else if clamp_state > 0 {
                    clamps_above.fetch_add(1, Ordering::Relaxed);
                }
                out_row[j_orig] = q;
                (w - q) / u_ss
            })
            .collect();

        // Phase B: OBS propagation via U[step, next_step] (upper-triangular
        // entries of U for unprocessed columns). Per-row update is
        // independent → rayon-parallel; inner sweep over remaining
        // permuted-column indices is serial per-row.
        let u_ref = &u;
        let perm_ref = &perm;
        let err_ref = &err_col;
        weights_residual
            .par_chunks_mut(k_dim)
            .enumerate()
            .for_each(|(row, res_row)| {
                let err = err_ref[row];
                if err == 0.0 {
                    return;
                }
                for next_step in (step + 1)..k_dim {
                    let kk_orig = perm_ref[next_step];
                    let u_sn = u_ref[(step, next_step)];
                    if u_sn != 0.0 {
                        res_row[kk_orig] -= err * u_sn;
                    }
                }
            });
    }

    // Diagnostic: per-tensor clamp stats. Print to stderr so the pipeline
    // log + bench script's tail-grep can correlate clamp rates with
    // downstream quality regressions.
    let total = total_count.load(Ordering::Relaxed);
    let cab = clamps_above.load(Ordering::Relaxed);
    let cbe = clamps_below.load(Ordering::Relaxed);
    let pct = 100.0 * (cab + cbe) as f64 / total.max(1) as f64;
    eprintln!(
        "[gptq-clamp] {tensor_name} M={m} K={k_dim} elements={total} \
         clamps={}/{} ({:.3}%)  above={cab}  below={cbe}",
        cab + cbe, total, pct,
    );

    Ok(effective_damp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity scale: `quantize_mq4_element` rounds to multiples of `scale`
    /// when `min_val = 0`.
    #[test]
    fn quantize_mq4_element_rounds_to_grid() {
        // Grid: 16 values 0, 0.25, 0.5, ..., 3.75 (scale=0.25, min_val=0)
        assert_eq!(quantize_mq4_element(0.0, 0.25, 0.0), 0.0);
        assert_eq!(quantize_mq4_element(0.1, 0.25, 0.0), 0.0);    // rounds down
        assert_eq!(quantize_mq4_element(0.15, 0.25, 0.0), 0.25);  // rounds up (>= 0.125)
        assert_eq!(quantize_mq4_element(3.5, 0.25, 0.0), 3.5);
        assert_eq!(quantize_mq4_element(3.74, 0.25, 0.0), 3.75);
        assert_eq!(quantize_mq4_element(10.0, 0.25, 0.0), 3.75);  // clamp to 15
        assert_eq!(quantize_mq4_element(-1.0, 0.25, 0.0), 0.0);   // clamp to 0
    }

    /// Asymmetric grid: `min_val` shifts the entire bucket array.
    #[test]
    fn quantize_mq4_element_handles_negative_min() {
        // Grid: -1.0, -0.875, ..., 0.875 (scale=0.125, min_val=-1.0)
        assert_eq!(quantize_mq4_element(-1.0, 0.125, -1.0), -1.0);
        assert_eq!(quantize_mq4_element(0.0, 0.125, -1.0), 0.0);
        assert_eq!(quantize_mq4_element(0.875, 0.125, -1.0), 0.875);
        assert_eq!(quantize_mq4_element(1.5, 0.125, -1.0), 0.875);  // clamp
        assert_eq!(quantize_mq4_element(-1.5, 0.125, -1.0), -1.0);  // clamp
    }

    /// Cholesky on a tiny SPD matrix: H = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]].
    #[test]
    fn cholesky_succeeds_on_spd() {
        let h = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 4.0, (0, 1) => 2.0,
            (1, 0) => 2.0, (1, 1) => 3.0,
            _ => unreachable!(),
        });
        let (l, damp) = cholesky_with_adaptive_damping(&h, 0.0, 1.0).unwrap();
        // Effective damp is at the ε·diag_mean floor (clamped_initial_damp),
        // not literally zero — that floor exists to prevent the damp=0
        // infinite-loop on singular inputs. Cosmetic shift; the Cholesky
        // result is unchanged to FP precision.
        assert!(damp < 1e-14, "SPD damp should be at the ε·diag_mean floor, got {damp}");
        // L[0][0] = sqrt(4) = 2.0
        assert!((l[(0, 0)] - 2.0).abs() < 1e-12, "L[0][0] = {}", l[(0, 0)]);
        // L[1][0] = 2 / 2 = 1.0
        assert!((l[(1, 0)] - 1.0).abs() < 1e-12, "L[1][0] = {}", l[(1, 0)]);
        // L[1][1] = sqrt(3 - 1) = sqrt(2)
        assert!((l[(1, 1)] - 2.0_f64.sqrt()).abs() < 1e-12, "L[1][1] = {}", l[(1, 1)]);
        // Above-diag entries should be zero
        assert_eq!(l[(0, 1)], 0.0);
    }

    /// Singular matrix (rank-deficient) requires damping. Verify the
    /// adaptive schedule kicks in and reports the effective damp used.
    #[test]
    fn cholesky_dampens_singular_matrix() {
        // [[1, 1], [1, 1]] — rank 1, singular.
        let h = Mat::<f64>::from_fn(2, 2, |_i, _j| 1.0);
        let result = cholesky_with_adaptive_damping(&h, 0.01, 1.0).unwrap();
        assert!(result.1 > 0.0, "expected non-zero damping");
        // diag_mean = 1.0; initial_damp=0.01 should succeed in one shot
        // since 0.01 * I makes a rank-2 matrix easily.
        assert_eq!(result.1, 0.01);
    }

    /// AWQ rescaling: identity scales → no-op.
    #[test]
    fn awq_rescaling_identity_is_noop() {
        let mut h = Mat::<f64>::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
        let h_orig = h.clone();
        apply_awq_rescaling(&mut h, &[1.0, 1.0, 1.0]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(h[(i, j)], h_orig[(i, j)]);
            }
        }
    }

    /// AWQ rescaling: doubling-scale halves Hessian entries.
    #[test]
    fn awq_rescaling_doubles_inverse_squared() {
        // H[i,j] = 4 for all i,j; s = [2, 2, 2] → H'[i,j] = 4 / 4 = 1.
        let mut h = Mat::<f64>::from_fn(3, 3, |_i, _j| 4.0);
        apply_awq_rescaling(&mut h, &[2.0, 2.0, 2.0]);
        for i in 0..3 {
            for j in 0..3 {
                assert!((h[(i, j)] - 1.0).abs() < 1e-12, "H[{i},{j}] = {}", h[(i, j)]);
            }
        }
    }

    /// FWHT-256 round-trip via similarity: applying the transform twice
    /// to a Hessian is NOT identity (it's `H_256² · H · H_256^{-2}`),
    /// but applying it once to a DIAGONAL Hessian preserves the trace.
    /// Lighter sanity check: the trace is preserved exactly.
    #[test]
    fn fwht_similarity_preserves_trace_on_diagonal() {
        let k = 256;
        let mut h = Mat::<f64>::zeros(k, k);
        for i in 0..k {
            h[(i, i)] = (i + 1) as f64;
        }
        let trace_before: f64 = (0..k).map(|i| h[(i, i)]).sum();

        let signs1: Vec<f64> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f64> = (0..256).map(|i| if (i / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        fwht_similarity_per_256(&mut h, &signs1, &signs2);

        let trace_after: f64 = (0..k).map(|i| h[(i, i)]).sum();
        // Orthogonal similarity preserves trace exactly.
        assert!(
            (trace_after - trace_before).abs() < 1e-9,
            "trace mismatch: before={trace_before}, after={trace_after}"
        );
    }

    /// Diagonal condition lower bound on a well-conditioned matrix.
    #[test]
    fn diag_condition_lower_bound_well_conditioned() {
        let h = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        // diag values: 1, 2, 3 → cond lower bound = 3/1 = 3.
        let cond = diag_condition_lower_bound(&h, 0.0);
        assert!((cond - 3.0).abs() < 1e-12);
    }

    #[test]
    fn diag_condition_handles_zero_diag_with_damping() {
        let h = Mat::<f64>::zeros(3, 3);
        // diag values all 0; damp=0.1 → 0.1/0.1 = 1.0
        let cond = diag_condition_lower_bound(&h, 0.1);
        assert!((cond - 1.0).abs() < 1e-12);
    }

    /// Frozen-block-grid: matches `quantize_mq4g256`'s scheme (main.rs:554-559).
    #[test]
    fn frozen_grid_matches_quantize_mq4g256_formula() {
        // 256 values: 0.0, 0.1, 0.2, ..., 25.5
        let weights: Vec<f64> = (0..256).map(|i| i as f64 * 0.1).collect();
        let grids = compute_frozen_block_grids(&weights);
        assert_eq!(grids.len(), 1);
        // min = 0.0, max = 25.5, range = 25.5, scale = 25.5/15 = 1.7
        assert!((grids[0].scale - 1.7).abs() < 1e-12);
        assert_eq!(grids[0].min_val, 0.0);
    }

    /// WEIGHT-mode actorder produces descending-diag permutation.
    #[test]
    fn weight_mode_actorder_sorts_descending() {
        let h_diag = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let perm = weight_mode_actorder(&h_diag);
        // Largest-first: index 1 (5.0), index 4 (4.0), index 2 (3.0), index 3 (2.0), index 0 (1.0)
        assert_eq!(perm, vec![1, 4, 2, 3, 0]);
    }

    /// inverse_perm round-trip identity.
    #[test]
    fn inverse_perm_roundtrip() {
        let perm = vec![3, 0, 4, 1, 2];
        let inv = inverse_perm(&perm);
        // Apply perm then inv → identity.
        let mut v: Vec<usize> = (0..5).collect();
        let permuted: Vec<usize> = perm.iter().map(|&i| v[i]).collect();
        let unpermuted: Vec<usize> = (0..5).map(|i| permuted[inv[i]]).collect();
        v.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        assert_eq!(unpermuted, v);
    }

    /// **GPTQ identity test:** when `H = I`, GPTQ should reduce to plain
    /// RTN (round-to-nearest) — no error propagation, since `H^-1 = I`
    /// has zero off-diagonal entries.
    #[test]
    fn gptq_identity_hessian_equals_rtn() {
        let m = 4;
        let k = 256;  // one frozen-block per row
        let weights_orig: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let frozen = compute_frozen_block_grids(&weights_orig);

        // H = I → no off-diagonal correction.
        let h = Mat::<f64>::identity(k, k);

        let mut weights = weights_orig.clone();
        let damp = gptq_column_sequential(&mut weights, &h, m, k, &frozen, 0.0, 1.0, "test:identity_H").unwrap();
        // Identity H trivially Cholesky'd — effective damp lands on the
        // ε·diag_mean=ε floor from clamped_initial_damp, not literally 0.
        assert!(damp < 1e-14, "identity H damp should be at the ε floor, got {damp}");

        // Compare to plain RTN on the same weights+grids.
        let mut rtn = weights_orig.clone();
        for row in 0..m {
            for col in 0..k {
                let flat = row * k + col;
                let block = block_idx_for(row, col, k);
                let g = frozen[block];
                rtn[flat] = quantize_mq4_element(weights_orig[flat], g.scale, g.min_val);
            }
        }

        // With H = I, GPTQ should produce identical output to RTN.
        for i in 0..m * k {
            assert!(
                (weights[i] - rtn[i]).abs() < 1e-9,
                "mismatch at flat[{i}]: gptq={}, rtn={}",
                weights[i],
                rtn[i]
            );
        }
    }

    /// Pack helper round-trips: packing then unpacking the codewords
    /// recovers the snapped grid values (within the per-block grid).
    #[test]
    fn pack_mq4g256_from_rotated_round_trip() {
        // Build 256 known values that snap to a 16-bucket grid.
        let weights: Vec<f64> = (0..256).map(|i| (i as f64) * 0.1).collect();
        let grids = compute_frozen_block_grids(&weights);
        // grid: scale=1.7, min_val=0.0
        let packed = pack_mq4g256_from_rotated_f64(&weights, &grids);
        assert_eq!(packed.len(), 136);
        // Decode the per-block header
        let scale = f32::from_le_bytes(packed[0..4].try_into().unwrap()) as f64;
        let min_val = f32::from_le_bytes(packed[4..8].try_into().unwrap()) as f64;
        assert!((scale - 1.7).abs() < 1e-6);
        assert_eq!(min_val, 0.0);
        // Decode every code, verify it matches a fresh per-element quantize.
        for i in 0..128 {
            let byte = packed[8 + i];
            let lo = (byte & 0xF) as f64;
            let hi = ((byte >> 4) & 0xF) as f64;
            let lo_dec = lo * scale + min_val;
            let hi_dec = hi * scale + min_val;
            let lo_expected = quantize_mq4_element(weights[2 * i], scale, min_val);
            let hi_expected = quantize_mq4_element(weights[2 * i + 1], scale, min_val);
            assert!(
                (lo_dec - lo_expected).abs() < 1e-9,
                "pack/decode mismatch at bucket {i} lo: got {lo_dec}, expected {lo_expected}"
            );
            assert!(
                (hi_dec - hi_expected).abs() < 1e-9,
                "pack/decode mismatch at bucket {i} hi: got {hi_dec}, expected {hi_expected}"
            );
        }
    }

    /// FWHT-per-256 preserves Parseval inner products. With asymmetric
    /// signs1/signs2 (as the actual MQ4 kernel uses via different seeds
    /// 42/1042), the FWHT is NOT self-inverse — but it is Parseval-orthogonal:
    /// `<FWHT(a), FWHT(b)> = <a, b>`. That's the only identity GPTQ + the
    /// MQ4 dot-product correctness rely on.
    #[test]
    fn fwht_per_256_weights_preserves_parseval() {
        let k = 256;
        // Two distinct random-ish vectors
        let a_orig: Vec<f64> = (0..k).map(|i| (i as f64 * 0.7).sin()).collect();
        let b_orig: Vec<f64> = (0..k).map(|i| (i as f64 * 0.3).cos() + 0.5).collect();
        let dot_before: f64 = (0..k).map(|i| a_orig[i] * b_orig[i]).sum();

        // Use deterministic ±1 sign tables (asymmetric — like the real kernel).
        let signs1: Vec<f64> = (0..256).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f64> = (0..256).map(|i| if (i / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        // Treat each as a 1×K row-major matrix; FWHT in place
        apply_fwht_per_256_to_weights_f64(&mut a, 1, k, &signs1, &signs2);
        apply_fwht_per_256_to_weights_f64(&mut b, 1, k, &signs1, &signs2);
        let dot_after: f64 = (0..k).map(|i| a[i] * b[i]).sum();

        // Parseval: <FWHT(a), FWHT(b)> = <a, b> exactly (modulo FP).
        assert!(
            (dot_after - dot_before).abs() / dot_before.abs().max(1e-30) < 1e-9,
            "Parseval failed: <a,b>={dot_before:.10e}, <FWHT(a),FWHT(b)>={dot_after:.10e}"
        );
    }

    /// **End-to-end GPTQ pipeline test:** AWQ-noop (s=1) + GPTQ with
    /// identity Hessian must produce the same bytes as plain RTN through
    /// the rotated grid. Validates the full chain: AWQ rescale → FWHT
    /// similarity → FWHT weights → frozen grids → GPTQ-identity → pack.
    #[test]
    fn gptq_pipeline_identity_matches_rtn_on_rotated() {
        let m = 2;
        let k = 256;
        let weights_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();

        // H = I (k×k), AWQ scales = 1.0 → entire pipeline reduces to
        // FWHT → frozen grids → RTN → pack.
        let h_unrot: Vec<f32> = (0..k * k).map(|i| if i / k == i % k { 1.0 } else { 0.0 }).collect();
        let awq_scales = vec![1.0_f64; k];
        let signs1: Vec<f32> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f32> = (0..256).map(|i| if (i / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let gptq_packed = gptq_pipeline_mq4g256(
            &weights_f32, m, k, &h_unrot, &awq_scales, &signs1, &signs2, 1e-6, 1.0,
            "test:pipeline_identity",
        )
        .expect("identity-H pipeline should not need damping");

        // Independently compute RTN on the same rotated weights via the
        // same packer (skip GPTQ).
        let signs1_f64: Vec<f64> = signs1.iter().map(|&v| v as f64).collect();
        let signs2_f64: Vec<f64> = signs2.iter().map(|&v| v as f64).collect();
        let mut rotated_f64: Vec<f64> = weights_f32.iter().map(|&v| v as f64).collect();
        apply_fwht_per_256_to_weights_f64(&mut rotated_f64, m, k, &signs1_f64, &signs2_f64);
        let grids = compute_frozen_block_grids(&rotated_f64);
        let rtn_packed = pack_mq4g256_from_rotated_f64(&rotated_f64, &grids);

        assert_eq!(gptq_packed.len(), rtn_packed.len(), "byte-length mismatch");
        assert_eq!(gptq_packed, rtn_packed, "GPTQ with identity-H should byte-equal plain rotated RTN");
    }

    /// **GPTQ reconstruction test:** for a well-conditioned diagonal-dominant H,
    /// GPTQ's quantization error against `H` should be ≤ plain RTN's
    /// error against `H` (where "error" = sum of `<H_jj, (w - w_q)^2>`
    /// per channel — the activation-weighted L2 reconstruction loss).
    #[test]
    fn gptq_improves_activation_weighted_reconstruction() {
        let m = 32;
        let k = 256;
        // Build a weight matrix with one "outlier" column that benefits
        // from error compensation. Other columns are tame.
        let mut weights_orig = vec![0.0_f64; m * k];
        for row in 0..m {
            for col in 0..k {
                let flat = row * k + col;
                weights_orig[flat] = if col == 100 {
                    // Outlier column with values that don't snap to a tight grid
                    1.234 + 0.001 * row as f64
                } else {
                    0.1 * (col as f64 / 256.0)
                };
            }
        }
        let frozen = compute_frozen_block_grids(&weights_orig);

        // Diagonal-dominant Hessian with one channel (100) heavily weighted.
        let h = Mat::<f64>::from_fn(k, k, |i, j| {
            if i == j {
                if i == 100 { 100.0 } else { 1.0 }
            } else {
                0.001  // small off-diagonals to give GPTQ something to do
            }
        });

        // Plain RTN.
        let mut rtn = weights_orig.clone();
        for row in 0..m {
            for col in 0..k {
                let flat = row * k + col;
                let block = block_idx_for(row, col, k);
                let g = frozen[block];
                rtn[flat] = quantize_mq4_element(weights_orig[flat], g.scale, g.min_val);
            }
        }

        // GPTQ.
        let mut gptq = weights_orig.clone();
        gptq_column_sequential(&mut gptq, &h, m, k, &frozen, 1e-6, 1.0, "test:improves_aw").unwrap();

        // Activation-weighted error: sum over (i,j,k) of (w[i,j]-w_q[i,j]) * H[j,k] * (w[i,k]-w_q[i,k]).
        // Approximate via per-channel diagonal (the dominant term):
        // sum_i sum_j H[j,j] * (w[i,j]-w_q[i,j])^2
        let aw_err = |q: &[f64]| -> f64 {
            let mut total = 0.0;
            for row in 0..m {
                for col in 0..k {
                    let flat = row * k + col;
                    let dq = weights_orig[flat] - q[flat];
                    total += h[(col, col)] * dq * dq;
                }
            }
            total
        };

        let rtn_err = aw_err(&rtn);
        let gptq_err = aw_err(&gptq);
        // GPTQ should reduce activation-weighted error (or at least not
        // make it worse by more than a tiny floating-point margin).
        assert!(
            gptq_err <= rtn_err * 1.01,
            "GPTQ should match or beat RTN on activation-weighted error: \
             rtn={rtn_err:.6e}, gptq={gptq_err:.6e}"
        );
    }

    /// Regression guard for the `initial_damp = 0` + singular H infinite-loop
    /// case. Prior to the `clamped_initial_damp` floor, `damp *= 10` stayed
    /// at zero forever and this call never returned.
    #[test]
    fn cholesky_terminates_on_singular_h_with_zero_initial_damp() {
        let h = Mat::<f64>::from_fn(4, 4, |_i, _j| 1.0);  // rank-1, singular
        let (_l, damp) = cholesky_with_adaptive_damping(&h, 0.0, 1.0)
            .expect("must terminate with successful damp on rank-1 H");
        assert!(damp > 0.0, "damp must be > 0 to make singular H invertible");

        let (_u, damp2) = compute_damped_inv_cholesky_upper(&h, None, 0.0, 1.0)
            .expect("compute_damped_inv_cholesky_upper must also terminate");
        assert!(damp2 > 0.0);
    }

    /// `compute_damped_inv_cholesky_upper` satisfies `U^T · U = (H+λI)^-1`,
    /// the Frantar-Algorithm-1 form. (Was previously `U · U^T = H_inv`
    /// before the 2026-05-14 fix — wrong invariant for GPTQ propagation.)
    #[test]
    fn compute_damped_inv_cholesky_upper_satisfies_identity() {
        let h = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 4.0, (0, 1) => 1.0, (0, 2) => 0.5,
            (1, 0) => 1.0, (1, 1) => 3.0, (1, 2) => 0.25,
            (2, 0) => 0.5, (2, 1) => 0.25, (2, 2) => 2.0,
            _ => unreachable!(),
        });
        let (u, damp) = compute_damped_inv_cholesky_upper(&h, None, 0.01, 1.0).unwrap();

        // U is upper-triangular: U[i, j] = 0 for i > j.
        for i in 0..3 {
            for j in 0..i {
                assert_eq!(u[(i, j)], 0.0, "U should be upper-tri: U[{i},{j}]={}", u[(i, j)]);
            }
        }

        // Compute (U^T · U) and (H + damp·I)^-1; compare.
        let mut utu = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += u[(k, i)] * u[(k, j)];  // U^T · U
                }
                utu[i][j] = s;
            }
        }

        // (H + damp·I) · utu should be I.
        let mut a = h.clone();
        for i in 0..3 { a[(i, i)] += damp; }
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[(i, k)] * utu[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-10,
                    "(H+damp·I)·(U^T·U) [{i},{j}] = {s}, expected {expected}"
                );
            }
        }
    }

    /// Regression test for the 2026-05-14 OBS-propagation bug. Two
    /// checks:
    ///   1. **Step 0**: `U[0, k] / U[0, 0]` must equal
    ///      `H_inv[0, k] / H_inv[0, 0]` (the direct first-row ratio).
    ///   2. **All steps via Schur complements**: at step j, the OBS
    ///      ratio `U[j, k] / U[j, j]` must equal the Schur-complement
    ///      ratio `S_j[0, k-j] / S_j[0, 0]` where S_j is the Schur
    ///      complement of H_inv after eliminating rows/cols 0..j-1.
    ///      This is the *full* Frantar-Algorithm-1 property — what
    ///      makes GPTQ-via-Cholesky correct.
    ///
    /// Prior to the fix, hipfire's `compute_damped_inv_cholesky_upper`
    /// returned `L_H^{-T}` whose row ratios diverged from the
    /// Schur-complement ratios by factors of 1.5–3.5× — silently breaking
    /// GPTQ's OBS cascade and producing quality regressions at every
    /// tested model size.
    #[test]
    fn obs_propagation_ratios_match_direct_h_inv() {
        // 4×4 SPD H for a stricter cross-check.
        let h = Mat::<f64>::from_fn(4, 4, |i, j| {
            ((i + 1) as f64) * ((j + 1) as f64) * 0.1
                + if i == j { 2.0 } else { 0.0 }
                + 0.05 * ((i as f64) - (j as f64)).sin()
        });
        // Symmetrize to be exactly SPD-compatible.
        let mut hs = h.clone();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let avg = 0.5 * (hs[(i, j)] + hs[(j, i)]);
                hs[(i, j)] = avg; hs[(j, i)] = avg;
            }
        }
        let damp = 1e-8;
        let (u, _eff_damp) = compute_damped_inv_cholesky_upper(&hs, None, damp, 1.0).unwrap();

        // Reference H_inv via (H + damp·I)^-1 from an independent path.
        let mut a = hs.clone();
        for i in 0..4 { a[(i, i)] += damp; }
        let l = a.llt(Side::Lower).unwrap();
        let identity = Mat::<f64>::identity(4, 4);
        let h_inv = l.solve(&identity);

        // Check 1 — step 0 row ratios (direct first-row of H_inv).
        for next in 1..4 {
            let u_ratio = u[(0, next)] / u[(0, 0)];
            let direct_ratio = h_inv[(0, next)] / h_inv[(0, 0)];
            assert!(
                (u_ratio - direct_ratio).abs() < 1e-9,
                "step 0 → col {next}: U={u_ratio:.9}, H_inv={direct_ratio:.9}",
            );
        }

        // Check 2 — full Schur-complement property at all steps.
        // S_j is the (4-j) × (4-j) Schur complement of H_inv after
        // eliminating leading principal submatrix [0:j, 0:j].
        // Build S_j by sequential Gaussian elimination on H_inv.
        let mut s = vec![vec![0.0_f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = h_inv[(i, j)];
            }
        }
        for j_step in 0..4 {
            // Verify ratios from current Schur block against U.
            for k in (j_step + 1)..4 {
                let u_ratio = u[(j_step, k)] / u[(j_step, j_step)];
                let schur_ratio = s[j_step][k] / s[j_step][j_step];
                assert!(
                    (u_ratio - schur_ratio).abs() < 1e-9,
                    "step {j_step} → col {k}: \
                     U[{j_step},{k}]/U[{j_step},{j_step}] = {u_ratio:.9}, \
                     Schur ratio = {schur_ratio:.9}",
                );
            }
            // Eliminate row/col j_step → next Schur complement.
            let pivot = s[j_step][j_step];
            for r in (j_step + 1)..4 {
                let factor = s[r][j_step] / pivot;
                for c in (j_step + 1)..4 {
                    s[r][c] -= factor * s[j_step][c];
                }
            }
        }
    }

    /// Permuted variant: `U · U^T = (P^T (H+λI) P)^-1`.
    #[test]
    fn compute_damped_inv_cholesky_upper_with_permutation() {
        let h = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 4.0, (0, 1) => 1.0, (0, 2) => 0.5,
            (1, 0) => 1.0, (1, 1) => 3.0, (1, 2) => 0.25,
            (2, 0) => 0.5, (2, 1) => 0.25, (2, 2) => 2.0,
            _ => unreachable!(),
        });
        let perm = vec![2_usize, 0, 1];  // arbitrary permutation
        let (u, damp) = compute_damped_inv_cholesky_upper(&h, Some(&perm), 0.01, 1.0).unwrap();

        // Build H_perm = P^T H P + damp·I (the matrix Cholesky operated on).
        let mut h_perm = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                h_perm[(i, j)] = h[(perm[i], perm[j])];
            }
        }
        for i in 0..3 { h_perm[(i, i)] += damp; }

        // (P^T H P + damp·I) · (U^T · U) should be I.
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    let mut utu_kj = 0.0;
                    for m in 0..3 {
                        utu_kj += u[(m, k)] * u[(m, j)];  // U^T · U
                    }
                    s += h_perm[(i, k)] * utu_kj;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-10,
                    "(H_perm) · (U^T·U) [{i},{j}] = {s}, expected {expected}"
                );
            }
        }
    }

    /// `symmetrize_in_place` produces an exactly symmetric matrix from a
    /// near-symmetric input — guard for the defensive scrub applied to
    /// `H_target` before Cholesky.
    #[test]
    fn symmetrize_in_place_produces_exact_symmetry() {
        let mut h = Mat::<f64>::from_fn(4, 4, |i, j| {
            let base = ((i * 4 + j) as f64) * 0.1;
            // Inject deterministic asymmetric perturbation
            base + if i < j { 1e-12 } else { 0.0 }
        });
        symmetrize_in_place(&mut h);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    h[(i, j)], h[(j, i)],
                    "after symmetrize: [{i},{j}] = {}, [{j},{i}] = {}",
                    h[(i, j)], h[(j, i)]
                );
            }
        }
    }

    /// FWHT similarity is symmetric in exact arithmetic but drifts in FP.
    /// Verify our defensive `symmetrize_in_place` clamp restores exact
    /// symmetry without changing the spectrum meaningfully (trace preserved).
    #[test]
    fn fwht_similarity_then_symmetrize_is_exactly_symmetric() {
        let k = 256;
        let mut h = Mat::<f64>::from_fn(k, k, |i, j| {
            // Random-ish symmetric input
            let v = ((i as f64) * 0.7 + (j as f64) * 0.31).sin();
            v
        });
        // Ensure exact symmetry of the input
        for i in 0..k {
            for j in (i + 1)..k {
                h[(j, i)] = h[(i, j)];
            }
        }
        let trace_before: f64 = (0..k).map(|i| h[(i, i)]).sum();

        let signs1: Vec<f64> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let signs2: Vec<f64> = (0..256).map(|i| if (i / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        fwht_similarity_per_256(&mut h, &signs1, &signs2);
        symmetrize_in_place(&mut h);

        for i in 0..k {
            for j in 0..k {
                assert_eq!(h[(i, j)], h[(j, i)]);
            }
        }
        let trace_after: f64 = (0..k).map(|i| h[(i, i)]).sum();
        assert!(
            (trace_after - trace_before).abs() < 1e-9,
            "trace shifted: before={trace_before}, after={trace_after}"
        );
    }

    /// `apply_awq_rescaling` panics defensively on a zero scale (would
    /// otherwise produce inf entries and corrupt the Hessian silently).
    #[test]
    #[should_panic(expected = "AWQ scales must be strictly positive")]
    fn apply_awq_rescaling_rejects_zero_scale() {
        let mut h = Mat::<f64>::from_fn(2, 2, |_i, _j| 1.0);
        apply_awq_rescaling(&mut h, &[1.0, 0.0]);
    }
}
