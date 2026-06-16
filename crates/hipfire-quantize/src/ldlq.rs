// SPDX-License-Identifier: Apache-2.0
// hipfire — QTIP-LDLQ (Phase C1e): output-aware (Hessian) trellis quantization.
//
// Clean-room: the LDLQ algorithm is implemented here; `faer` is only the
// SIMD/parallel Cholesky backend. MSE-optimal QTIP-2 failed PPL (125.6 vs MQ4
// 14.0) because reconstruction-optimal ≠ output-optimal; LDLQ minimizes the
// activation-weighted output error ‖(W−Ŵ)·√H‖ via OBS error feedback.

use faer::prelude::Solve;
use faer::{Mat, Side};

/// Lower factor `L` with `L·Lᵀ = (H + λI)⁻¹` (H row-major `k×k`, SPD).
///
/// The GPTQ/OBS upper factor `U` (`UᵀU = (H+λI)⁻¹`, used as: divisor
/// `U[step,step]`, propagation weight `U[step,next]`) is exactly `Uᵀ = L`, so
/// `U[step,step] = L[step,step]` and `U[step,next>step] = L[next,step]`. We
/// return `L` and index it transposed in the OBS loop — avoids a second
/// transpose pass. `None` on non-SPD breakdown.
pub fn inv_cholesky_lower(h_rowmajor: &[f32], k: usize, damp: f64) -> Option<Mat<f64>> {
    assert_eq!(h_rowmajor.len(), k * k);
    let hd = Mat::<f64>::from_fn(k, k, |i, j| {
        h_rowmajor[i * k + j] as f64 + if i == j { damp } else { 0.0 }
    });
    let chol = hd.llt(Side::Lower).ok()?;
    let hinv = chol.solve(Mat::<f64>::identity(k, k)); // H⁻¹ via Cholesky solve
    let chol2 = hinv.llt(Side::Lower).ok()?;
    Some(chol2.L().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG for a reproducible SPD matrix.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        }
    }

    #[test]
    fn inv_cholesky_reconstructs_inverse() {
        let k = 64;
        let mut rng = Lcg(0xABCD);
        // A = random; H = AᵀA (SPD) + ridge.
        let mut a = vec![0.0f64; k * k];
        for v in a.iter_mut() {
            *v = rng.next();
        }
        let mut h = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for t in 0..k {
                    s += a[t * k + i] * a[t * k + j];
                }
                h[i * k + j] = s as f32;
            }
        }
        let damp = 1e-3;
        for i in 0..k {
            h[i * k + i] += damp as f32;
        }
        let l = inv_cholesky_lower(&h, k, 0.0).expect("SPD"); // ridge already in h

        // Verify (H)·(L·Lᵀ) ≈ I  ⇔  L·Lᵀ = H⁻¹.
        // Compute M = H · (L·Lᵀ), check ≈ identity.
        let llt = |r: usize, c: usize| -> f64 {
            // (L·Lᵀ)[r,c] = Σ_t L[r,t] L[c,t]
            let mut s = 0.0;
            for t in 0..=r.min(c) {
                s += l[(r, t)] * l[(c, t)];
            }
            s
        };
        let mut max_off = 0.0f64;
        let mut max_diag_err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let mut m = 0.0;
                for t in 0..k {
                    m += h[i * k + t] as f64 * llt(t, j);
                }
                if i == j {
                    max_diag_err = max_diag_err.max((m - 1.0).abs());
                } else {
                    max_off = max_off.max(m.abs());
                }
            }
        }
        eprintln!("inv-cholesky: max|diag-1|={max_diag_err:.2e} max|offdiag|={max_off:.2e}");
        assert!(max_diag_err < 1e-4, "diag err {max_diag_err}");
        assert!(max_off < 1e-4, "offdiag err {max_off}");
    }
}
