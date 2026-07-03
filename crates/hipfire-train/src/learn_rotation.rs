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
use rayon::prelude::*;

// These are parallelized over OUTPUT ROWS: each row's inner sum stays serial within one
// thread, so results are bit-identical to the serial version (no float-reassociation) —
// only the wall-clock changes. This matters at h=2048 where the learn was the long pole.

/// Row-major `[p,q]·[q,c] → [p,c]`.
fn matmul(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; p * c];
    out.par_chunks_mut(c).enumerate().for_each(|(i, orow)| {
        for k in 0..q {
            let aik = a[i * q + k];
            if aik == 0.0 {
                continue;
            }
            let brow = &b[k * c..k * c + c];
            for (o, &bv) in orow.iter_mut().zip(brow.iter()) {
                *o += aik * bv;
            }
        }
    });
    out
}

/// Row-major `A·Bᵀ`: `[p,q]·([c,q])ᵀ → [p,c]`.
fn matmul_bt(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; p * c];
    out.par_chunks_mut(c).enumerate().for_each(|(i, orow)| {
        let arow = &a[i * q..i * q + q];
        for (j, o) in orow.iter_mut().enumerate() {
            let brow = &b[j * q..j * q + q];
            *o = arow.iter().zip(brow).map(|(&av, &bv)| av * bv).sum();
        }
    });
    out
}

/// `Aᵀ·B`: `([p,q])ᵀ·[p,c] → [q,c]`. Parallel over the `q` output rows (k), each
/// summing over all `i` in i-order — identical accumulation to the serial loop.
fn matmul_at(a: &[f32], b: &[f32], p: usize, q: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; q * c];
    out.par_chunks_mut(c).enumerate().for_each(|(k, orow)| {
        for i in 0..p {
            let av = a[i * q + k];
            if av == 0.0 {
                continue;
            }
            let brow = &b[i * c..i * c + c];
            for (o, &bv) in orow.iter_mut().zip(brow.iter()) {
                *o += av * bv;
            }
        }
    });
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

/// Euclidean gradient of the kurtosis surrogate `Σ(M Rᵀ)⁴` w.r.t. `R`:
/// `G = 4·(M̃∘³)ᵀ M`, `M̃ = M Rᵀ`. `m` is `[rows,h]` row-major.
fn kurtosis_grad(m: &[f32], rot: &Rotation, rows: usize, h: usize) -> Vec<f32> {
    let mt = crate::rotation::rotate_rows(m, rot, rows); // M Rᵀ
    let mut cube = vec![0.0f32; rows * h];
    for (c, &v) in cube.iter_mut().zip(mt.iter()) {
        *c = 4.0 * v * v * v;
    }
    matmul_at(&cube, m, rows, h, h) // (M̃³)ᵀ M → [h,h]
}

/// Scale a `[h,h]` gradient to unit Frobenius norm (so two objective terms can be
/// mixed by a clean directional weight independent of their raw magnitudes).
fn frob_normalize(g: &mut [f32]) {
    let n: f32 = g.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if n > 1e-20 {
        let inv = 1.0 / n;
        for v in g.iter_mut() {
            *v *= inv;
        }
    }
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
        let g = kurtosis_grad(x, &rot, rows, h);
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

/// Like [`learn_rotation_kurtosis`] but minimizes a **joint** activation+weight
/// kurtosis: the deployed W4A4 quantizes *both* the rotated activation `X Rᵀ` and
/// the rotated reader weight `W Rᵀ`, so a rotation that only flattens activations
/// leaves the weight int4 error on the table. Each per-step gradient term is
/// Frobenius-normalized before mixing, so `lambda ∈ [0,1]` is a clean directional
/// blend (`0` = activations only, `1` = weights only, `0.5` = balanced). `x_act`
/// is `[rows_act,h]`, `x_wt` is the stacked reader weights `[rows_wt,h]`.
#[allow(clippy::too_many_arguments)]
pub fn learn_rotation_joint(
    x_act: &[f32],
    rows_act: usize,
    x_wt: &[f32],
    rows_wt: usize,
    h: usize,
    init: Rotation,
    iters: usize,
    lr: f32,
    fp_iters: usize,
    lambda: f32,
) -> Rotation {
    assert_eq!(init.h, h);
    let mut rot = init;
    for it in 0..iters {
        let mut ga = kurtosis_grad(x_act, &rot, rows_act, h);
        let mut gw = kurtosis_grad(x_wt, &rot, rows_wt, h);
        frob_normalize(&mut ga);
        frob_normalize(&mut gw);
        let mut g = vec![0.0f32; h * h];
        for i in 0..h * h {
            g[i] = (1.0 - lambda) * ga[i] + lambda * gw[i];
        }
        cayley_step(&mut rot.r, &g, h, lr, fp_iters);
        if it % 16 == 15 {
            rot.reorthonormalize();
        }
    }
    rot.reorthonormalize();
    rot
}

/// Per-256-group (or `group`-wide) symmetric `bits`-bit fake-quant **residual**
/// `Q(v) − v` of a row-major `[rows,h]` matrix, matching the Oq codec's symmetric
/// per-group grid (scale = max|·|/qmax, round-to-nearest, clamp `[-qmax,qmax]`).
/// This is the differentiable stand-in for the deployed weight quantizer.
fn fake_quant_residual(v: &[f32], rows: usize, h: usize, group: usize, bits: u32) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // 4-bit → 7
    let mut d = vec![0.0f32; rows * h];
    for r in 0..rows {
        let mut g0 = 0;
        while g0 < h {
            let g1 = (g0 + group).min(h);
            let seg = &v[r * h + g0..r * h + g1];
            let amax = seg.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            if amax > 0.0 {
                let scale = amax / qmax;
                for (j, &x) in seg.iter().enumerate() {
                    let q = (x / scale).round().clamp(-qmax, qmax) * scale;
                    d[r * h + g0 + j] = q - x;
                }
            }
            g0 = g1;
        }
    }
    d
}

/// Hessian-weighted weight-quant-error objective (the **decode-path** term of the
/// phase-joint rotation): with the reader weight `W [out,h]` stored as `Q_b(W Rᵀ)`
/// and the activation exact (`X Rᵀ`, f16), the decode output error is
/// `‖ΔW·(X Rᵀ)ᵀ‖²` = `tr(ΔW R H Rᵀ ΔWᵀ)`, `ΔW = Q_b(W Rᵀ) − W Rᵀ`, `H = XᵀX`.
/// Lower ⇒ the (one, shared) int-weight buffer degrades the f16-activation decode
/// path less. This is the term that *protects* decode while the activation kurtosis
/// term buys the A4 prefill win.
pub fn hess_wquant_objective(
    w: &[f32],
    out: usize,
    h: usize,
    hess: &[f32],
    rot: &Rotation,
    group: usize,
    bits: u32,
) -> f64 {
    let wr = crate::rotation::rotate_rows(w, rot, out); // W Rᵀ  [out,h]
    let dw = fake_quant_residual(&wr, out, h, group, bits); // ΔW  [out,h]
    let a = matmul(&dw, &rot.r, out, h, h); // ΔW·R  [out,h]
    let ah = matmul(&a, hess, out, h, h); // (ΔW R)·H  [out,h]
    a.iter()
        .zip(ah.iter())
        .map(|(&p, &q)| p as f64 * q as f64)
        .sum()
}

/// Euclidean gradient of [`hess_wquant_objective`] w.r.t. `R`, **QAT convention**:
/// stop-gradient on the quantizer (`Q(U)` frozen, so `ΔW = sg(Q(U)) − U` flows the
/// `−U` path, `dΔW/dU = −I`). This is the direction that matters — rotating changes
/// the residual *magnitude* via incoherence, which the STE-detached `2·C·R·H` misses
/// (it steers a frozen residual and walks off the good Hadamard). With `U = W Rᵀ`,
/// `M = R H Rᵀ`, `dL/dU = −2·ΔW·M`, and `dL/dR = (dL/dU)ᵀ W = −2·R H Rᵀ·ΔWᵀ·W`.
fn hess_wquant_grad(
    w: &[f32],
    out: usize,
    h: usize,
    hess: &[f32],
    rot: &Rotation,
    group: usize,
    bits: u32,
) -> Vec<f32> {
    let wr = crate::rotation::rotate_rows(w, rot, out); // U = W Rᵀ  [out,h]
    let dw = fake_quant_residual(&wr, out, h, group, bits); // ΔW  [out,h]
    let rh = matmul(&rot.r, hess, h, h, h); // R H
    let rhrt = matmul_bt(&rh, &rot.r, h, h, h); // R H Rᵀ  [h,h]
    let dwtw = matmul_at(&dw, w, out, h, h); // ΔWᵀ W  [h,h]
    let mut g = matmul(&rhrt, &dwtw, h, h, h); // R H Rᵀ ΔWᵀ W
    for v in g.iter_mut() {
        *v *= -2.0;
    }
    g
}

/// **Phase-joint** rotation: one orthonormal `R` (one stored int-weight buffer) that
/// serves *both* the compute-bound prefill (W4A4 — needs the rotated activation
/// `X Rᵀ` to quantize to int4, driven by the kurtosis term) and the bandwidth-bound
/// decode (W-int/A-f16 — needs `Q_b(W Rᵀ)` to stay a good grid, driven by the
/// Hessian-weighted weight-quant term). The two per-step gradients are each
/// Frobenius-normalized, then blended by `alpha ∈ [0,1]` (`0` = activation-only =
/// today's `--rotate`; `1` = weight-quant-only ≈ decode-optimal; between = the
/// single-buffer compromise). `hess` is `H = XᵀX` `[h,h]` from the calib collector.
#[allow(clippy::too_many_arguments)]
pub fn learn_rotation_phase_joint(
    x_act: &[f32],
    rows_act: usize,
    w: &[f32],
    out: usize,
    hess: &[f32],
    h: usize,
    group: usize,
    bits: u32,
    init: Rotation,
    iters: usize,
    lr: f32,
    fp_iters: usize,
    alpha: f32,
) -> Rotation {
    assert_eq!(init.h, h);
    assert_eq!(hess.len(), h * h, "H must be [h,h]");
    let mut rot = init;
    for it in 0..iters {
        // Geometric lr decay: the frob-normalized Cayley step is a fixed rotation
        // angle, so without decay it orbits a good optimum (e.g. the Hadamard start,
        // near-optimal for weight quant) instead of settling into it.
        let lr_t = lr * 0.99f32.powi(it as i32);
        let mut ga = kurtosis_grad(x_act, &rot, rows_act, h);
        let mut gw = hess_wquant_grad(w, out, h, hess, &rot, group, bits);
        frob_normalize(&mut ga);
        frob_normalize(&mut gw);
        let mut g = vec![0.0f32; h * h];
        for i in 0..h * h {
            g[i] = (1.0 - alpha) * ga[i] + alpha * gw[i];
        }
        cayley_step(&mut rot.r, &g, h, lr_t, fp_iters);
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

    /// The joint objective must reduce *both* the activation and the (distinct)
    /// weight kurtosis below their identity starts — a balanced blend, not just
    /// one term. Uses two independent heavy-tailed sets with outliers in
    /// different channels so a rotation good for one isn't automatically good for
    /// the other.
    #[test]
    fn joint_reduces_both_kurtoses() {
        let h = 16usize;
        let mut s = 999u64;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        };
        let heavy = |nxt: &mut dyn FnMut() -> f32, rows: usize, outliers: &[usize]| {
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
            x
        };
        let act = heavy(&mut nxt, 24, &[1, 5, 9]);
        let wt = heavy(&mut nxt, 40, &[3, 12, 14]);
        let a_id = kurtosis_objective(&act, &Rotation::identity(h), 24);
        let w_id = kurtosis_objective(&wt, &Rotation::identity(h), 40);
        let learned = learn_rotation_joint(
            &act,
            24,
            &wt,
            40,
            h,
            Rotation::hadamard(h, 1),
            200,
            0.4,
            4,
            0.5,
        );
        let a_l = kurtosis_objective(&act, &learned, 24);
        let w_l = kurtosis_objective(&wt, &learned, 40);
        println!("joint  act {a_id:.3e}→{a_l:.3e}  wt {w_id:.3e}→{w_l:.3e}");
        assert!(learned.orthonormality_error() < 1e-3, "lost orthonormality");
        assert!(a_l < a_id, "act kurtosis not reduced: {a_id:.3e}→{a_l:.3e}");
        assert!(w_l < w_id, "wt kurtosis not reduced: {w_id:.3e}→{w_l:.3e}");
    }

    /// Phase-joint rotation: the two phase objectives genuinely trade off, and each
    /// end of `alpha` optimizes its own. Activation outliers and weight outliers sit
    /// in different channels, so no single rotation is best for both. Asserts:
    /// (a) orthonormality held; (b) `alpha=1` drives the Hessian-weighted
    /// weight-quant error below the identity/Hadamard start (decode protected);
    /// (c) a real frontier — the activation-only end has lower activation kurtosis
    /// but higher weight-quant error than the weight-only end.
    #[test]
    fn phase_joint_trades_activation_for_weight() {
        let h = 32usize;
        let mut s = 4242u64;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        };
        let heavy = |nxt: &mut dyn FnMut() -> f32, rows: usize, outliers: &[usize]| {
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
            x
        };
        let rows_act = 48usize;
        let act = heavy(&mut nxt, rows_act, &[3, 11, 20]);
        let out = 64usize;
        let w = heavy(&mut nxt, out, &[6, 17, 25]); // reader weight [out,h], different channels
                                                    // H = XᵀX from the activations (the calib collector's Hessian).
        let hess = matmul_at(&act, &act, rows_act, h, h);
        let (group, bits) = (h, 4u32); // one group (h<256), int4

        let start = Rotation::hadamard(h, 1);
        let w_id = hess_wquant_objective(&w, out, h, &hess, &start, group, bits);

        let learn = |alpha: f32| {
            learn_rotation_phase_joint(
                &act,
                rows_act,
                &w,
                out,
                &hess,
                h,
                group,
                bits,
                Rotation::hadamard(h, 1),
                240,
                0.4,
                4,
                alpha,
            )
        };
        let act_only = learn(0.0);
        let wt_only = learn(1.0);

        let ka = |r: &Rotation| kurtosis_objective(&act, r, rows_act);
        let qw = |r: &Rotation| hess_wquant_objective(&w, out, h, &hess, r, group, bits);
        println!(
            "phase-joint  act-kurt: α0={:.3e} α1={:.3e} | wquant: id={:.3e} α0={:.3e} α1={:.3e}",
            ka(&act_only),
            ka(&wt_only),
            w_id,
            qw(&act_only),
            qw(&wt_only)
        );
        assert!(
            act_only.orthonormality_error() < 1e-3,
            "α0 lost orthonormality"
        );
        assert!(
            wt_only.orthonormality_error() < 1e-3,
            "α1 lost orthonormality"
        );
        // (b) weight-only end protects decode: weight-quant error below the start.
        assert!(
            qw(&wt_only) < w_id,
            "α1 didn't reduce wquant: {:.3e}→{:.3e}",
            w_id,
            qw(&wt_only)
        );
        // (c) a real frontier: act-only has lower activation kurtosis...
        assert!(ka(&act_only) < ka(&wt_only), "no act-kurt frontier");
        // ...but pays with higher weight-quant error than the weight-only end.
        assert!(qw(&act_only) > qw(&wt_only), "no wquant frontier");
    }
}
