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

/// Per-256-block FWHT similarity transform of the Hessian, in place:
/// `H ← R H Rᵀ` where `R` is the engine's per-256 signed FWHT
/// (`crate::cpu_fwht_256`). Row-pass then column-pass (same R both sides),
/// putting H in the same incoherent domain as the FWHT-rotated weights so the
/// OBS feedback is consistent.
fn rotate_hessian(h: &mut [f64], k: usize, signs1: &[f32], signs2: &[f32]) {
    let nb = k / 256;
    let mut buf = [0.0f32; 256];
    for r in 0..k {
        for b in 0..nb {
            for c in 0..256 {
                buf[c] = h[r * k + b * 256 + c] as f32;
            }
            crate::cpu_fwht_256(&mut buf, signs1, signs2);
            for c in 0..256 {
                h[r * k + b * 256 + c] = buf[c] as f64;
            }
        }
    }
    for col in 0..k {
        for b in 0..nb {
            for r in 0..256 {
                buf[r] = h[(b * 256 + r) * k + col] as f32;
            }
            crate::cpu_fwht_256(&mut buf, signs1, signs2);
            for r in 0..256 {
                h[(b * 256 + r) * k + col] = buf[r] as f64;
            }
        }
    }
}

/// QTIP-LDLQ: block-sequential **trellis** quantization with OBS error
/// feedback. Returns the dequantized weights in the ORIGINAL (un-rotated)
/// domain — the effective weight a fused QTIP kernel would produce — as
/// row-major `m×k` f32, for the simulated-quant PPL path. `None` on
/// Cholesky breakdown (caller falls back to plain QTIP).
///
/// FWHT-rotate H + W into the incoherent domain → `L` with `LLᵀ=(H_rot+λI)⁻¹`
/// (so OBS divisor `=L[c,c]`, propagation weight `U[c,f]=L[f,c]`) → for each
/// 256-column block in order: per row, trellis-encode the OBS-adjusted
/// `residual`, record dequant, propagate `(w−ŵ)/L[c,c]` to all later columns.
/// Intra-block coupling handled jointly by the trellis; inter-block by OBS.
#[allow(clippy::too_many_arguments)]
pub fn qtip2_ldlq_dequant(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    beam_width: usize,
    damp: f64,
) -> Option<Vec<f32>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "qtip2_ldlq_dequant requires k % 256 == 0");

    // Rotate the Hessian, then L with L·Lᵀ = (H_rot + λI)⁻¹.
    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let hd = Mat::<f64>::from_fn(k, k, |i, j| h[i * k + j] + if i == j { damp } else { 0.0 });
    let chol = hd.llt(Side::Lower).ok()?;
    let hinv = chol.solve(Mat::<f64>::identity(k, k));
    let l = hinv.llt(Side::Lower).ok()?.L().to_owned();

    // Rotate the weights into the same domain.
    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual.par_chunks_mut(k).enumerate().for_each(|(row, rr)| {
        let base = row * k;
        let mut buf = [0.0f32; 256];
        for b in 0..nb {
            for c in 0..256 {
                buf[c] = weights_f32[base + b * 256 + c];
            }
            crate::cpu_fwht_256(&mut buf, signs1, signs2);
            for c in 0..256 {
                rr[b * 256 + c] = buf[c] as f64;
            }
        }
    });

    let cb = crate::qtip::build_codebook();
    let mut dequant = vec![0.0f64; m * k];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let errs: Vec<Vec<f64>> = dequant
            .par_chunks_mut(k)
            .enumerate()
            .map(|(row, dr)| {
                let rbase = row * k;
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = residual[rbase + c0 + c] as f32;
                }
                let s0 = crate::qtip::group_scale(&grp);
                let sym = crate::qtip::beam_encode_group(&grp, s0, &cb, beam_width);
                let s = crate::qtip::optimal_scale(&grp, &sym, &cb);
                let deq = crate::qtip::decode_group(&sym, s, &cb);
                let mut err = vec![0.0f64; 256];
                for c in 0..256 {
                    dr[c0 + c] = deq[c] as f64;
                    let ucc = l[(c0 + c, c0 + c)];
                    err[c] = if ucc > 0.0 {
                        (residual[rbase + c0 + c] - deq[c] as f64) / ucc
                    } else {
                        0.0
                    };
                }
                err
            })
            .collect();

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(errs.par_iter())
                .for_each(|(rr, err)| {
                    for c in 0..256 {
                        let ec = err[c];
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)]; // U[col,f] = L[f,col]
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    // Un-rotate (FWHT orthogonal → swap sign args).
    let mut out = vec![0.0f32; m * k];
    out.par_chunks_mut(k).enumerate().for_each(|(row, orow)| {
        let mut buf = [0.0f32; 256];
        for b in 0..nb {
            for c in 0..256 {
                buf[c] = dequant[row * k + b * 256 + c] as f32;
            }
            crate::cpu_fwht_256(&mut buf, signs2, signs1);
            orow[b * 256..b * 256 + 256].copy_from_slice(&buf);
        }
    });
    Some(out)
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

    /// The LDLQ claim: OBS feedback (real H) reduces the H-weighted *output*
    /// error vs no feedback (identity H) on column-correlated weights — the
    /// fix for the PPL-125 MSE-only QTIP-2.
    #[test]
    fn qtip2_ldlq_obs_beats_no_feedback() {
        let (m, k) = (1024usize, 512usize);
        let mut rng = Lcg(0x1357);
        let mut w = vec![0.0f32; m * k];
        for row in 0..m {
            let mut prev = 0.0f64;
            for c in 0..k {
                prev = 0.85 * prev + rng.next(); // AR(1) across columns
                w[row * k + c] = prev as f32;
            }
        }
        // H = (1/m) Σ w_rowᵀ w_row + ridge.
        let mut h = vec![0.0f32; k * k];
        for row in 0..m {
            let base = row * k;
            for i in 0..k {
                let wi = w[base + i];
                if wi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    h[i * k + j] += wi * w[base + j];
                }
            }
        }
        for v in h.iter_mut() {
            *v /= m as f32;
        }
        for i in 0..k {
            h[i * k + i] += 1e-2;
        }
        let ident: Vec<f32> = (0..k * k).map(|x| if x / k == x % k { 1.0 } else { 0.0 }).collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let deq_h = qtip2_ldlq_dequant(&w, m, k, &h, &s1, &s2, 64, 1e-2).expect("ldlq H");
        let deq_i = qtip2_ldlq_dequant(&w, m, k, &ident, &s1, &s2, 64, 1e-2).expect("ldlq I");

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k).map(|c| (w[base + c] - deq[base + c]) as f64).collect();
                for i in 0..k {
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for j in 0..k {
                        acc += h[i * k + j] as f64 * d[j];
                    }
                    tot += d[i] * acc;
                }
            }
            tot
        };
        let (eh, ei) = (out_err(&deq_h), out_err(&deq_i));
        eprintln!("qtip2-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}", eh / ei);
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }
}
