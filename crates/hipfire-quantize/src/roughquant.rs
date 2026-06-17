// SPDX-License-Identifier: Apache-2.0
// hipfire — RoughQuant Phase 2: PCA rotation into the activation-Hessian
// eigenbasis. Clean-room; `faer` is only the symmetric-eigensolver + matmul
// backend (same dependency the LDLQ path already uses).
//
// Weight-only specialization (hipfire layout: weight [m×k] = [out×in], forward
// y = W·x, x∈ℝ^k). C = XᵀX is the k×k input-channel covariance (the Hessian
// sidecar). Eigendecompose C = P·Λ·Pᵀ; columns of P are input directions sorted
// by eigenvalue (energy) descending. In the rotated frame x = P·x̃, the weight
// is W̃ = W·P, whose columns are now energy-ranked: column 0 = highest-variance
// activation direction. RoughQuant protects the top columns (super-weight /
// ResQ subspace) and crushes the tail. The sim inverse-rotates the dequantized
// weight back to the original frame (W_q = W̃_q·Pᵀ) so the normal forward is
// numerically invariant — no runtime rotation needed for the PPL verdict.

use faer::{Mat, Side};

/// PCA basis from the per-input-channel Hessian `C = XᵀX` (row-major `k×k`,
/// symmetric PSD). Returns `(P, eigvals)` with `P` row-major `k×k` whose COLUMNS
/// are eigenvectors sorted by eigenvalue DESCENDING (column 0 = highest energy),
/// and `eigvals` the matching eigenvalues (descending). A small diagonal ridge
/// `damp` (fraction of mean diag) stabilizes near-singular C. `None` if the
/// eigensolve fails.
pub fn pca_basis(c_rowmajor: &[f32], k: usize, damp_frac: f64) -> Option<(Vec<f32>, Vec<f32>)> {
    assert_eq!(c_rowmajor.len(), k * k);
    let mut diag_sum = 0.0f64;
    for i in 0..k {
        diag_sum += c_rowmajor[i * k + i] as f64;
    }
    let damp = damp_frac * (diag_sum / k as f64).max(1e-12);
    let cd = Mat::<f64>::from_fn(k, k, |i, j| {
        c_rowmajor[i * k + j] as f64 + if i == j { damp } else { 0.0 }
    });
    let eig = cd.self_adjoint_eigen(Side::Lower).ok()?;
    let s = eig.S(); // eigenvalues (faer: ascending), as a column vector
    let u = eig.U(); // k×k eigenvectors, column j ↔ eigenvalue s[j]
                     // Reorder columns by eigenvalue descending.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| {
        s[a].partial_cmp(&s[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
    });
    let mut p = vec![0.0f32; k * k];
    let mut ev = vec![0.0f32; k];
    for (newc, &oldc) in order.iter().enumerate() {
        ev[newc] = s[oldc] as f32;
        for r in 0..k {
            p[r * k + newc] = u[(r, oldc)] as f32;
        }
    }
    Some((p, ev))
}

/// Rotate weight `W` [m×k] by the k×k basis `P` (row-major): returns `W·P`
/// (`transpose=false`) or `W·Pᵀ` (`transpose=true`), row-major m×k. Uses faer's
/// optimized matmul. `W·P` maps to the PCA frame; `W·Pᵀ` maps back.
pub fn rotate_w(w: &[f32], p: &[f32], m: usize, k: usize, transpose: bool) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    assert_eq!(p.len(), k * k);
    let wmat = Mat::<f32>::from_fn(m, k, |i, j| w[i * k + j]);
    let pmat = Mat::<f32>::from_fn(k, k, |i, j| p[i * k + j]);
    let out = if transpose {
        wmat * pmat.transpose()
    } else {
        wmat * pmat.as_ref()
    };
    let mut v = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            v[i * k + j] = out[(i, j)];
        }
    }
    v
}
