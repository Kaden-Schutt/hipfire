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

fn inv_cholesky_lower_rotated(h: &[f64], k: usize, damp: f64) -> Option<Mat<f64>> {
    let base = damp.max(1e-12);
    for mult in [1.0, 10.0, 100.0, 1000.0, 10000.0] {
        let lambda = base * mult;
        let hd = Mat::<f64>::from_fn(k, k, |i, j| {
            h[i * k + j] + if i == j { lambda } else { 0.0 }
        });
        let Ok(chol) = hd.llt(Side::Lower) else {
            continue;
        };
        let hinv = chol.solve(Mat::<f64>::identity(k, k));
        let Ok(chol2) = hinv.llt(Side::Lower) else {
            continue;
        };
        return Some(chol2.L().to_owned());
    }
    None
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
/// 2-bit LDLQ wrapper (preserves the original signature / callers).
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
    qtip_ldlq_dequant_bits(
        weights_f32,
        m,
        k,
        h_rowmajor_f32,
        signs1,
        signs2,
        beam_width,
        damp,
        2,
        &crate::qtip::build_codebook(),
    )
}

/// Bit-parametric QTIP-LDLQ: same block-trellis OBS encode for any bit-rate.
/// The bitshift codebook is indexed by the 12-bit trellis *state* (independent
/// of bits-per-weight), so only the per-step symbol count differs — encode /
/// optimal-scale / decode route to the `_bits` variants.
#[allow(clippy::too_many_arguments)]
pub fn qtip_ldlq_dequant_bits(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    beam_width: usize,
    damp: f64,
    bits: u32,
    cb: &[f32],
) -> Option<Vec<f32>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "qtip_ldlq_dequant_bits requires k % 256 == 0");

    // Rotate the Hessian, then L with L·Lᵀ = (H_rot + λI)⁻¹.
    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    // Rotate the weights into the same domain.
    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    // Codebook threaded from the caller (1MAD or 3INST) — must match the
    // beam_encode / decode the on-device kernel computes.
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
                let sym = crate::qtip::beam_encode_group_bits(&grp, s0, &cb, beam_width, bits);
                let s = crate::qtip::optimal_scale_bits(&grp, &sym, &cb, bits);
                let deq = crate::qtip::decode_group_bits(&sym, s, &cb, bits);
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

/// Opus W4A4 (oq4) LDLQ: Hessian error-feedback **symmetric int4** weight quant,
/// emitting the SAME packed `[f16 scale][128 signed nibbles]` per-256-group layout
/// as `codecs::quantize_oq4g256` (130 B/group, row-major), but with the GPTQ/OBS
/// error feedback the plain RTN codec lacks.
///
/// Same machinery as `qtip_ldlq_dequant_bits`: FWHT-rotate H + W into the
/// incoherent domain (oq4 stores PRE-rotated weights, so the packed output stays
/// in the rotated domain — no un-rotate), `L` with `LLᵀ=(H_rot+λI)⁻¹`, then for
/// each 256-column block in order, per row: clip-search scale + round the
/// OBS-adjusted residual to signed int4, pack, and propagate `(w−ŵ)/L[c,c]` to
/// all later columns. The inner per-group quant is oq4's symmetric round (the only
/// substitution vs the QTIP trellis encode). `None` on Cholesky breakdown →
/// caller falls back to plain `quantize_oq4g256`.
pub fn oq4_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq4_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    let nb = k / 256;
    // FWHT-rotate weights into the same domain (this is the residual we feed).
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    const BLOCK_BYTES: usize = 130; // 2 (f16 scale) + 128 nibbles
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        // Per row: clip-search scale, round to int4, pack, compute OBS error.
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..128 {
                    let qlo = (grp[2 * i] * inv).round().clamp(-7.0, 7.0);
                    let qhi = (grp[2 * i + 1] * inv).round().clamp(-7.0, 7.0);
                    block[2 + i] = ((qlo as i8 as u8) & 0xf) | (((qhi as i8 as u8) & 0xf) << 4);
                    let (clo, chi) = (c0 + 2 * i, c0 + 2 * i + 1);
                    let ulo = l[(clo, clo)];
                    let uhi = l[(chi, chi)];
                    err[2 * i] = if ulo > 0.0 {
                        (grp[2 * i] as f64 - (qlo * scale) as f64) / ulo
                    } else {
                        0.0
                    };
                    err[2 * i + 1] = if uhi > 0.0 {
                        (grp[2 * i + 1] as f64 - (qhi * scale) as f64) / uhi
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// Opus W3A4 (oq3) LDLQ: Hessian error-feedback **symmetric int3** weight quant,
/// emitting the SAME bit-plane `[f16 scale][8×(3 u32)]` per-256-group layout as
/// `codecs::quantize_oq3g256` (98 B/group = 3.0625 b/w), but with GPTQ/OBS error
/// feedback the plain RTN codec lacks. At 3 bits the quant grid is coarse (±3), so
/// error feedback matters MORE than at int4 — this is the calibrated `oq3++` packer.
///
/// Same machinery as [`oq4_ldlq_pack`]: FWHT-rotate H + W into the incoherent domain
/// (oq3 stores PRE-rotated weights, so the packed output stays rotated — no
/// un-rotate), `L` with `LLᵀ=(H_rot+λI)⁻¹`, then for each 256-column block in order,
/// per row: clip-search scale + round the OBS-adjusted residual to signed int3
/// `[-3,3]`, bit-plane pack, and propagate `(w−ŵ)/L[c,c]` to all later columns. Only
/// the inner quant range + packing differ from `oq4_ldlq_pack`. `None` on Cholesky
/// breakdown → caller falls back to plain `quantize_oq3g256`.
pub fn oq3_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq3_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    const BLOCK_BYTES: usize = 98; // 2 (f16 scale) + 8×3 u32 bit-planes
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 3.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for s in 0..8 {
                    let (mut p0, mut p1, mut p2) = (0u32, 0u32, 0u32);
                    for i in 0..32 {
                        let idx = s * 32 + i;
                        let q = (grp[idx] * inv).round().clamp(-3.0, 3.0);
                        let u = (q as i8 as u8) & 7; // 3-bit two's-complement
                        p0 |= ((u & 1) as u32) << i;
                        p1 |= (((u >> 1) & 1) as u32) << i;
                        p2 |= (((u >> 2) & 1) as u32) << i;
                        let col = c0 + idx;
                        let ucc = l[(col, col)];
                        err[idx] = if ucc > 0.0 {
                            (grp[idx] as f64 - (q * scale) as f64) / ucc
                        } else {
                            0.0
                        };
                    }
                    let bo = 2 + s * 12;
                    block[bo..bo + 4].copy_from_slice(&p0.to_le_bytes());
                    block[bo + 4..bo + 8].copy_from_slice(&p1.to_le_bytes());
                    block[bo + 8..bo + 12].copy_from_slice(&p2.to_le_bytes());
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// Opus W8A8 (oq8) LDLQ: Hessian error-feedback symmetric int8 weight quant,
/// emitting the same `[f16 scale][256 signed int8]` per-256-group layout as
/// `codecs::quantize_oq8g256`. This is the calibrated `op8+` packer: FWHT-rotate
/// H and W into the stored OQ8 domain, round the OBS-adjusted residual to int8,
/// and propagate the remaining error to later groups.
pub fn oq8_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oq8_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    const BLOCK_BYTES: usize = 258; // 2 (f16 scale) + 256 int8
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 127.0);
                let inv = 1.0 / scale;
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..256 {
                    let q = (grp[i] * inv).round().clamp(-127.0, 127.0);
                    block[2 + i] = q as i8 as u8;
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// LDLQ (GPTQ/OBS error-feedback) packer for the magnitude-tiered OQ+ (Opus
/// Plus) format: bulk int4 + top-`w8_frac` weights per group at int8, emitting the
/// **Oq8 on-disk layout** (`[f16 scale][256 signed int8]`, 258 B/group) so it
/// loads via the existing qt=35 path and the single iu8 W8A8 kernel.
///
/// Same machinery as [`oq4_ldlq_pack`] — FWHT-rotate H + W into the incoherent
/// domain, `L` with `LLᵀ=(H_rot+λI)⁻¹`, then per 256-block (in order) per row:
/// clip-search an INT4-tuned scale, pick the top-`w8_frac` positions by
/// int8-upgrade gain, round those to int8 `[-127,127]` and the bulk to int4
/// `[-7,7]` (same scale), and propagate `(w−ŵ)/L[c,c]` to later blocks. The win
/// vs plain tiering: the int8 outlier positions have ~zero residual, so the OBS
/// feedback spends its budget compensating only the int4 bulk.
#[allow(clippy::too_many_arguments)]
pub fn oqplus_tiered_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
    w8_frac: f32,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oqplus_tiered_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    const BLOCK_BYTES: usize = 258; // 2 (f16 scale) + 256 int8
    let n_out = ((w8_frac as f64 * 256.0).round() as usize).clamp(1, 256);
    let mut out = vec![0u8; m * nb * BLOCK_BYTES];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, [u8; BLOCK_BYTES])> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                // Outliers: top n_out by int8-upgrade gain (int4_err² − int8_err²).
                let gain = |i: usize| -> f32 {
                    let v = grp[i];
                    let q4 = (v * inv).round().clamp(-7.0, 7.0);
                    let q8 = (v * inv).round().clamp(-127.0, 127.0);
                    let e4 = v - q4 * scale;
                    let e8 = v - q8 * scale;
                    e4 * e4 - e8 * e8
                };
                let mut idx: [usize; 256] = core::array::from_fn(|i| i);
                idx.sort_unstable_by(|&a, &c| {
                    gain(c)
                        .partial_cmp(&gain(a))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                let mut is_w8 = [false; 256];
                for &i in &idx[..n_out] {
                    is_w8[i] = true;
                }
                let mut block = [0u8; BLOCK_BYTES];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                for i in 0..256 {
                    let lim = if is_w8[i] { 127.0 } else { 7.0 };
                    let q = (grp[i] * inv).round().clamp(-lim, lim);
                    block[2 + i] = q as i8 as u8;
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * BLOCK_BYTES;
            out[off..off + BLOCK_BYTES].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

/// COMPACT (~4 b/w) variant of [`oqplus_tiered_ldlq_pack`]: identical GPTQ/OBS
/// error-feedback and tiered quantization, but emits the compact per-256-group
/// layout `[f16 scale][128 int4 nibbles][N_out × (u8 idx, i8 val)]` (matching
/// `codecs::quantize_oqplus_compact`) instead of the int8 Oq8 layout. The OBS
/// residual uses the actual tiered value (int8 for outliers → ~0 error → little
/// propagation; int4 for the bulk), so the feedback compensates the bulk exactly
/// as in the int8 packer — same dequantized values, ~half the bytes.
#[allow(clippy::too_many_arguments)]
pub fn oqplus_compact_ldlq_pack(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    h_rowmajor_f32: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    damp: f64,
    w8_frac: f32,
) -> Option<Vec<u8>> {
    use rayon::prelude::*;
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(h_rowmajor_f32.len(), k * k);
    assert_eq!(k % 256, 0, "oqplus_compact_ldlq_pack requires k % 256 == 0");

    let mut h: Vec<f64> = h_rowmajor_f32.iter().map(|&v| v as f64).collect();
    rotate_hessian(&mut h, k, signs1, signs2);
    let l = inv_cholesky_lower_rotated(&h, k, damp)?;

    let nb = k / 256;
    let mut residual = vec![0.0f64; m * k];
    residual
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, rr)| {
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

    let n_out = ((w8_frac as f64 * 256.0).round() as usize).clamp(1, 255);
    let block_bytes = 130 + 2 * n_out; // [f16][128 nibbles][n_out×(u8 idx, i8 val)]
    let mut out = vec![0u8; m * nb * block_bytes];

    for blk in 0..nb {
        let c0 = blk * 256;
        let c1 = c0 + 256;
        let results: Vec<(Vec<f64>, Vec<u8>)> = residual
            .par_chunks(k)
            .map(|rr_row| {
                let mut grp = [0.0f32; 256];
                for (c, g) in grp.iter_mut().enumerate() {
                    *g = rr_row[c0 + c] as f32;
                }
                let scale = crate::codecs::symmetric_clipsearch(&grp, 7.0);
                let inv = 1.0 / scale;
                let gain = |i: usize| -> f32 {
                    let v = grp[i];
                    let q4 = (v * inv).round().clamp(-7.0, 7.0);
                    let q8 = (v * inv).round().clamp(-127.0, 127.0);
                    let e4 = v - q4 * scale;
                    let e8 = v - q8 * scale;
                    e4 * e4 - e8 * e8
                };
                let mut idx: [usize; 256] = core::array::from_fn(|i| i);
                idx.sort_unstable_by(|&a, &c| {
                    gain(c)
                        .partial_cmp(&gain(a))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                let mut is_w8 = [false; 256];
                for &i in &idx[..n_out] {
                    is_w8[i] = true;
                }
                let mut block = vec![0u8; block_bytes];
                let s16 = crate::f32_to_f16(scale);
                block[0] = (s16 & 0xFF) as u8;
                block[1] = (s16 >> 8) as u8;
                let mut err = vec![0.0f64; 256];
                // Quantize each position to its tier; nibbles store the int4 clamp
                // (outlier slots overridden by the sparse table on load), err uses
                // the ACTUAL tiered value.
                for i in 0..128 {
                    let q4lo = (grp[2 * i] * inv).round().clamp(-7.0, 7.0) as i8;
                    let q4hi = (grp[2 * i + 1] * inv).round().clamp(-7.0, 7.0) as i8;
                    block[2 + i] = ((q4lo as u8) & 0xf) | (((q4hi as u8) & 0xf) << 4);
                }
                for i in 0..256 {
                    let lim = if is_w8[i] { 127.0 } else { 7.0 };
                    let q = (grp[i] * inv).round().clamp(-lim, lim);
                    let u = l[(c0 + i, c0 + i)];
                    err[i] = if u > 0.0 {
                        (grp[i] as f64 - (q * scale) as f64) / u
                    } else {
                        0.0
                    };
                }
                let tbl = 130;
                for (s, &pos) in idx[..n_out].iter().enumerate() {
                    let q8 = (grp[pos] * inv).round().clamp(-127.0, 127.0) as i8;
                    block[tbl + 2 * s] = pos as u8;
                    block[tbl + 2 * s + 1] = q8 as u8;
                }
                (err, block)
            })
            .collect();

        for (row, (_, block)) in results.iter().enumerate() {
            let off = (row * nb + blk) * block_bytes;
            out[off..off + block_bytes].copy_from_slice(block);
        }

        if c1 < k {
            residual
                .par_chunks_mut(k)
                .zip(results.par_iter())
                .for_each(|(rr, (err, _))| {
                    for (c, &ec) in err.iter().enumerate() {
                        if ec == 0.0 {
                            continue;
                        }
                        let col = c0 + c;
                        for f in c1..k {
                            let usf = l[(f, col)];
                            if usf != 0.0 {
                                rr[f] -= ec * usf;
                            }
                        }
                    }
                });
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG for a reproducible SPD matrix.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        let ident: Vec<f32> = (0..k * k)
            .map(|x| if x / k == x % k { 1.0 } else { 0.0 })
            .collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let deq_h = qtip2_ldlq_dequant(&w, m, k, &h, &s1, &s2, 64, 1e-2).expect("ldlq H");
        let deq_i = qtip2_ldlq_dequant(&w, m, k, &ident, &s1, &s2, 64, 1e-2).expect("ldlq I");

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k)
                    .map(|c| (w[base + c] - deq[base + c]) as f64)
                    .collect();
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
        eprintln!(
            "qtip2-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}",
            eh / ei
        );
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }

    /// The int3 packer's OBS feedback (real H) must reduce the H-weighted *output*
    /// error vs no feedback (identity H) on column-correlated weights — the same
    /// LDLQ claim as the QTIP path, for the coarse ±3 grid where it matters most.
    /// Round-trips through the codec's `dequant_oq3g256` oracle (bit-plane decode).
    #[test]
    fn oq3_ldlq_obs_beats_no_feedback() {
        let (m, k) = (512usize, 512usize);
        let mut rng = Lcg(0x2468);
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
        let ident: Vec<f32> = (0..k * k)
            .map(|x| if x / k == x % k { 1.0 } else { 0.0 })
            .collect();
        let s1 = crate::gen_fwht_signs(42, 256);
        let s2 = crate::gen_fwht_signs(1042, 256);

        let pk_h = oq3_ldlq_pack(&w, m, k, &h, &s1, &s2, 1e-2).expect("oq3 ldlq H");
        let pk_i = oq3_ldlq_pack(&w, m, k, &ident, &s1, &s2, 1e-2).expect("oq3 ldlq I");
        let deq_h = crate::codecs::dequant_oq3g256(&pk_h, m * k, &s1, &s2);
        let deq_i = crate::codecs::dequant_oq3g256(&pk_i, m * k, &s1, &s2);

        let out_err = |deq: &[f32]| -> f64 {
            let mut tot = 0.0f64;
            for row in 0..m {
                let base = row * k;
                let d: Vec<f64> = (0..k)
                    .map(|c| (w[base + c] - deq[base + c]) as f64)
                    .collect();
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
        eprintln!(
            "oq3-ldlq output-err: H-OBS={eh:.4} no-fb={ei:.4} ratio={:.3}",
            eh / ei
        );
        assert!(eh < ei, "OBS feedback must beat no-feedback: {eh} !< {ei}");
    }
}
