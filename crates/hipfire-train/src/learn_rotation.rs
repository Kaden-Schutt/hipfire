// SPDX-License-Identifier: Apache-2.0
//! SpinQuant Phase 2: **learn** the residual rotation R1 (Cayley SGD on the
//! Stiefel manifold), the step from the fixed +0.9 dB tier to the learned +6.8 dB
//! tier.
//!
//! ## Objective — the differentiable incoherence proxy
//!
//! The natural target is the *quantized* CE loss, but a plain straight-through
//! estimator on `Q(X Rᵀ)·Q(W Rᵀ)ᵀ` has a near-zero gradient w.r.t. `R`: the clean
//! term `X Rᵀ·R Wᵀ = X Wᵀ` is rotation-invariant, and STE zeroes the derivative
//! of the quant-noise term — so the loss looks flat to first order exactly where
//! it isn't. Instead we minimize a smooth, dense surrogate whose minimizer is a
//! quant-friendly rotation: the **per-element 4th moment** of the rotated
//! activations,
//!
//! `L(R) = Σ_r Σ_i (X Rᵀ)_{r,i}⁴ .`
//!
//! Because an orthonormal `R` preserves each row's energy `Σ_i x̃²`, minimizing
//! `Σ x̃⁴` minimizes the **kurtosis** — it flattens the heavy tails (the outlier
//! channels) that inflate a shared int4 group scale and crush the bulk. This is
//! the incoherence/Gaussianization that QuaRot/SpinQuant rotations are *for*,
//! made into a differentiable objective with a clean dense gradient
//! `G = dL/dR = 4·(X̃∘³)ᵀ X` (`X̃ = X Rᵀ`, `∘³` elementwise cube).
//!
//! ## Optimizer — Cayley SGD on the Stiefel manifold
//!
//! A Euclidean step off the manifold would break orthonormality. The Cayley
//! transform keeps `R Rᵀ = I` exactly: from the skew part `A = Ĝ − Ĝᵀ`,
//! `Ĝ = G Rᵀ`, the update is
//!
//! `R' = (I + α/2·A)⁻¹ (I − α/2·A) R ,`
//!
//! computed by a fixed-point iteration for the inverse (no explicit solve, ~2×
//! SGD cost/iter). `A` skew ⇒ the Cayley factor is orthogonal ⇒ `R'` stays on the
//! manifold. Host-side and offline (the rotation is baked into the weights by
//! [`crate::rotation::apply_r1`] once learned); `O(h³)`/step, fine for a bake.

use crate::rotation::Rotation;

/// Row-major `[p,q]·[q,c] → [p,c]`.
fn matmul(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; p * c];
    for i in 0..p {
        for k in 0..q {
            let aik = a[i * q + k];
            if aik == 0.0 {
                continue;
            }
            let brow = &b[k * c..k * c + c];
            let orow = &mut out[i * c..i * c + c];
            for (o, &bv) in orow.iter_mut().zip(brow.iter()) {
                *o += aik * bv;
            }
        }
    }
    out
}

/// Row-major `A·Bᵀ`: `[p,q]·([c,q])ᵀ → [p,c]`.
fn matmul_bt(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; p * c];
    for i in 0..p {
        let arow = &a[i * q..i * q + q];
        for j in 0..c {
            let brow = &b[j * q..j * q + q];
            let mut acc = 0.0f32;
            for (&av, &bv) in arow.iter().zip(brow.iter()) {
                acc += av * bv;
            }
            out[i * c + j] = acc;
        }
    }
    out
}

/// `Aᵀ·B`: `([p,q])ᵀ·[p,c] → [q,c]`.
fn matmul_at(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; q * c];
    for i in 0..p {
        let arow = &a[i * q..i * q + q];
        let brow = &b[i * c..i * c + c];
        for (k, &av) in arow.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let orow = &mut out[k * c..k * c + c];
            for (o, &bv) in orow.iter_mut().zip(brow.iter()) {
                *o += av * bv;
            }
        }
    }
    out
}

/// One Cayley-SGD descent step on the Stiefel manifold. `r` (orthonormal `[h,h]`,
/// row-major) is updated **in place** to reduce the objective whose Euclidean
/// gradient is `g = dL/dR` `[h,h]`. `lr` is the step size; `fp_iters` the number
/// of fixed-point iterations for the Cayley inverse (3–5 suffices for small `lr`).
pub fn cayley_step(r: &mut [f32], g: &[f32], h: usize, lr: f32, fp_iters: usize) {
    // Ĝ = G Rᵀ, then the skew part A = Ĝ − Ĝᵀ. Descending: use +A so the Cayley
    // curve moves along −grad on the manifold (verified by the Procrustes test).
    let ghat = matmul_bt(g, r, h, h, h); // G Rᵀ
    let mut a = vec![0.0f32; h * h];
    for i in 0..h {
        for j in 0..h {
            a[i * h + j] = ghat[i * h + j] - ghat[j * h + i];
        }
    }
    // Normalize A by its Frobenius norm so `lr` is a bounded per-step rotation
    // angle, independent of the gradient magnitude. This keeps the fixed-point
    // inverse a contraction (½·lr·‖A‖ = ½·lr < 1) — the raw gradient can be large
    // enough to make the iteration diverge otherwise.
    let fro: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if fro > 1e-20 {
        let inv = 1.0 / fro;
        for v in a.iter_mut() {
            *v *= inv;
        }
    }
    let half = lr * 0.5;
    // Y = (I − half·A) R = R − half·(A R).
    let ar = matmul(&a, r, h, h, h);
    let mut y = vec![0.0f32; h * h];
    for idx in 0..h * h {
        y[idx] = r[idx] - half * ar[idx];
    }
    // Solve (I + half·A) R' = Y by fixed point: R' ← Y − half·(A R').
    let mut rp = r.to_vec(); // warm start at R
    for _ in 0..fp_iters {
        let arp = matmul(&a, &rp, h, h, h);
        for idx in 0..h * h {
            rp[idx] = y[idx] - half * arp[idx];
        }
    }
    r.copy_from_slice(&rp);
}

/// Sum of per-element 4th powers of the rotated rows `X Rᵀ` — the kurtosis
/// surrogate objective (lower ⇒ flatter tails ⇒ quant-friendlier).
pub fn kurtosis_objective(x: &[f32], rot: &Rotation, rows: usize) -> f64 {
    let xr = crate::rotation::rotate_rows(x, rot, rows);
    xr.iter().map(|&v| (v as f64).powi(4)).sum()
}

/// Learn an orthonormal `R [h,h]` minimizing the 4th-moment (kurtosis) surrogate
/// of the rows of `X [rows,h]` via Cayley SGD, starting from `init` (e.g. a
/// Hadamard warm start, or identity). Returns the learned [`Rotation`]. `lr` is a
/// bounded per-step rotation angle (the gradient magnitude is normalized away in
/// [`cayley_step`]), so one `lr` transfers across activation scales.
pub fn learn_rotation_kurtosis(
    x: &[f32],
    rows: usize,
    h: usize,
    init: Rotation,
    iters: usize,
    lr: f32,
    fp_iters: usize,
) -> Rotation {
    assert_eq!(init.h, h);
    let mut rot = init;
    for it in 0..iters {
        let xr = crate::rotation::rotate_rows(x, &rot, rows); // X Rᵀ
                                                              // G = 4 (X̃∘³)ᵀ X.
        let mut cube = vec![0.0f32; rows * h];
        for (c, &v) in cube.iter_mut().zip(xr.iter()) {
            *c = 4.0 * v * v * v;
        }
        let g = matmul_at(&cube, x, rows, h, h); // (X̃³)ᵀ X  → [h,h]
        cayley_step(&mut rot.r, &g, h, lr, fp_iters);
        // The approximate Cayley inverse drifts slightly off the manifold;
        // re-project periodically so error doesn't accumulate over many steps.
        if it % 16 == 15 {
            rot.reorthonormalize();
        }
    }
    rot.reorthonormalize();
    rot
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orthonormality_err(r: &[f32], h: usize) -> f32 {
        let mut worst = 0.0f32;
        for i in 0..h {
            for j in 0..h {
                let mut dot = 0.0f32;
                for k in 0..h {
                    dot += r[i * h + k] * r[j * h + k];
                }
                let t = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - t).abs());
            }
        }
        worst
    }

    /// Procrustes: minimize `L(R) = ‖R A − B‖²` with `B = R_true A`. Cayley SGD
    /// from identity must drive `L→0` while holding orthonormality — a
    /// self-contained optimizer check independent of the quant objective.
    /// `dL/dR = 2 (RA − B) Aᵀ`.
    #[test]
    fn cayley_solves_procrustes() {
        let (h, n) = (8usize, 24usize);
        let r_true = Rotation::random(h, 3).r;
        // A [h,n] random, B = R_true A.
        let a: Vec<f32> = (0..h * n)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2)
            .collect();
        let b = matmul(&r_true, &a, h, h, n);
        let loss = |r: &[f32]| -> f32 {
            let ra = matmul(r, &a, h, h, n);
            ra.iter().zip(&b).map(|(&p, &q)| (p - q) * (p - q)).sum()
        };
        let mut r = Rotation::identity(h).r;
        let l0 = loss(&r);
        let mut lr = 0.08f32;
        for step in 0..800 {
            let ra = matmul(&r, &a, h, h, n);
            let mut resid = vec![0.0f32; h * n];
            for i in 0..h * n {
                resid[i] = 2.0 * (ra[i] - b[i]);
            }
            let g = matmul_bt(&resid, &a, h, n, h); // (RA−B)·Aᵀ → [h,h]
            cayley_step(&mut r, &g, h, lr, 6);
            lr *= 0.996; // decay so the normalized step settles into the optimum
            if step % 16 == 15 {
                let mut rr = Rotation { h, r: r.clone() };
                rr.reorthonormalize();
                r = rr.r;
            }
        }
        let l1 = loss(&r);
        assert!(orthonormality_err(&r, h) < 1e-3, "lost orthonormality");
        assert!(
            l1 < l0 * 1e-2,
            "Procrustes not solved: L {l0:.3e} → {l1:.3e}"
        );
    }

    /// Heavy-tailed rows: learning must reduce the kurtosis objective below the
    /// identity start and hold orthonormality.
    #[test]
    fn learns_to_reduce_kurtosis() {
        let (rows, h) = (32usize, 16usize);
        // A few shared outlier channels over a small Gaussian-ish bulk.
        let mut s = 12345u64;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        };
        let outliers = [2usize, 7, 11];
        let mut x = vec![0.0f32; rows * h];
        for r in 0..rows {
            for c in 0..h {
                let base = nxt();
                x[r * h + c] = if outliers.contains(&c) {
                    base + nxt().signum() * 8.0
                } else {
                    base
                };
            }
        }
        let l_id = kurtosis_objective(&x, &Rotation::identity(h), rows);
        let learned = learn_rotation_kurtosis(&x, rows, h, Rotation::hadamard(h, 1), 200, 0.5, 4);
        let l_learned = kurtosis_objective(&x, &learned, rows);
        println!("kurtosis  identity={l_id:.3e}  learned={l_learned:.3e}");
        assert!(learned.orthonormality_error() < 1e-3, "lost orthonormality");
        assert!(
            l_learned < l_id,
            "learned kurtosis {l_learned:.3e} not below identity {l_id:.3e}"
        );
    }
}
