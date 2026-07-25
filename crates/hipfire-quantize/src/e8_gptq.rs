// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — GPTQ/LDLQ-on-E8 (Hessian-aware sequential rounding for mfp{4,3,2}-E8).
//
// Completes the QuIP# recipe for the MFP{4,3,2}G32E8 wire formats:
//   (1) incoherence rotation = offline FWHT (already in quantize_mfp4g32_e8_2d)
//   (2) E8 lattice codebook   (e8.rs)
//   (3) Hessian-aware rounding = THIS module (vector-LDLQ, block-diagonal-256)
//
// Output bytes are BIT-IDENTICAL in format to the RTN path for each n:
//   n=4: 16-byte header + per-32-block (1-byte E4M3 scale + 4 u32 codewords = 17 B)
//   n=3: 16-byte header + per-32-block (1-byte E4M3 scale + 4 u24 codewords = 13 B)
//   n=2: 16-byte header + per-32-block (1-byte E4M3 scale + 4 u16 codewords =  9 B)
// Scales are computed from the data FIRST and FROZEN; GPTQ ONLY changes which
// lattice point each 8-group picks, so the existing kernel decoders are UNCHANGED.
// With H = I (after damping) the assignment collapses exactly to RTN.
//
// All linear algebra is FP64. The per-256-block Cholesky is hand-rolled
// (256×256 is trivial — no external dependency needed; this keeps the quantize
// crate free of `faer`, unlike the orphan `gptq.rs` module).

#![allow(dead_code)] // wired conditionally from main.rs

use crate::e8;

/// Damping factor: H'_b += LAMBDA * mean(diag(H'_b)) * I. Essential for the
/// rare/singular per-expert Hessians that heavy-tailed MoE routing produces.
pub const LAMBDA: f64 = 0.01;
/// Feedback runaway guard: if a group's normalized target exceeds this after
/// error compensation, drop the feedback for that group (use RTN value).
pub const V_CLAMP: f32 = 12.0;

/// Per-256-block diagonal Hessian (row-major 256×256, F32 on disk / in-memory).
/// Captured PRE-rotation as XX^T over the input channels of one 256-block.
pub type HBlock = Vec<f32>; // len 256*256

/// FP64 FWHT-256 in place, matching `cpu_fwht_256` (FP32) bit-convention:
/// sign-flip by signs1, unnormalized Hadamard butterfly, then *1/16*signs2.
fn fwht_256_f64(x: &mut [f64; 256], signs1: &[f32], signs2: &[f32]) {
    for i in 0..256 {
        x[i] *= signs1[i] as f64;
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
    const SCALE: f64 = 1.0 / 16.0; // 1/sqrt(256)
    for i in 0..256 {
        x[i] *= SCALE * signs2[i] as f64;
    }
}

/// Rotate a captured 256×256 Hessian into the quantization basis:
///   H' = R · H · R^T,   R = diag(signs2)·(1/16)H_256·diag(signs1).
/// Implemented by applying fwht_256_f64 to every COLUMN (left-multiply by R)
/// then every ROW (right-multiply by R^T). R is orthogonal so R^T = R^-1.
/// Returns a symmetrized 256×256 row-major f64.
pub fn rotate_hblock(h: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<f64> {
    assert_eq!(h.len(), 256 * 256);
    let mut m = vec![0.0f64; 256 * 256];
    for i in 0..256 * 256 {
        m[i] = h[i] as f64;
    }
    // Left-multiply by R: FWHT each column.
    for c in 0..256 {
        let mut col = [0.0f64; 256];
        for r in 0..256 {
            col[r] = m[r * 256 + c];
        }
        fwht_256_f64(&mut col, signs1, signs2);
        for r in 0..256 {
            m[r * 256 + c] = col[r];
        }
    }
    // Right-multiply by R^T (== apply R to each row, R symmetric-in-action):
    for r in 0..256 {
        let mut row = [0.0f64; 256];
        row.copy_from_slice(&m[r * 256..r * 256 + 256]);
        fwht_256_f64(&mut row, signs1, signs2);
        m[r * 256..r * 256 + 256].copy_from_slice(&row);
    }
    // Symmetrize to kill FP asymmetry.
    for i in 0..256 {
        for j in (i + 1)..256 {
            let avg = 0.5 * (m[i * 256 + j] + m[j * 256 + i]);
            m[i * 256 + j] = avg;
            m[j * 256 + i] = avg;
        }
    }
    m
}

/// In-place lower Cholesky of a 256×256 row-major SPD matrix `h` (overwrites
/// the lower triangle with L; upper triangle is left as-is/ignored). Returns
/// false if a non-positive pivot is hit (not PD).
fn cholesky_lower_inplace(h: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut diag = h[j * n + j];
        for k in 0..j {
            let ljk = h[j * n + k];
            diag -= ljk * ljk;
        }
        if !(diag > 0.0) {
            return false;
        }
        let ljj = diag.sqrt();
        h[j * n + j] = ljj;
        let inv = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = h[i * n + j];
            for k in 0..j {
                s -= h[i * n + k] * h[j * n + k];
            }
            h[i * n + j] = s * inv;
        }
    }
    true
}

/// GPTQ feedback matrix for one 256-block. We need the OBS ratios
/// `Hinv[c, j] / Hinv[c, c]` for j > c, where Hinv = (H')^-1. Frantar's
/// Cholesky-direct form: with U = upper-Cholesky of Hinv (U^T U = Hinv), the
/// ratio Hinv[c,j]/Hinv[c,c] simplifies to U[c,j]/U[c,c] for j>=c (Schur
/// property). We compute Hinv explicitly (256×256 is cheap), then its
/// upper Cholesky U. `feedback[c*256+j] = U[c,j]/U[c,c]` for j>=c, else 0.
///
/// Returns None if H' is not PD even after damping escalation (caller → RTN).
pub fn block_feedback(hp_rot: &[f64]) -> Option<Vec<f64>> {
    const N: usize = 256;
    // mean(diag) for damping
    let mut mean_diag = 0.0f64;
    for i in 0..N {
        mean_diag += hp_rot[i * N + i];
    }
    mean_diag /= N as f64;
    if !(mean_diag > 0.0) {
        return None; // expert never routed / degenerate → RTN
    }

    // Adaptive damping: start at LAMBDA*mean_diag, escalate ×10 up to 3×.
    let mut damp = LAMBDA * mean_diag;
    for _attempt in 0..4 {
        let mut h = hp_rot.to_vec();
        for i in 0..N {
            h[i * N + i] += damp;
        }
        // Lower Cholesky of H' = L L^T.
        let mut l = h.clone();
        if !cholesky_lower_inplace(&mut l, N) {
            damp *= 10.0;
            continue;
        }
        // Invert L (lower-triangular) → Linv (lower). Then Hinv = Linv^T Linv.
        let mut linv = vec![0.0f64; N * N];
        for i in 0..N {
            linv[i * N + i] = 1.0 / l[i * N + i];
            for j in 0..i {
                let mut s = 0.0f64;
                for k in j..i {
                    s += l[i * N + k] * linv[k * N + j];
                }
                linv[i * N + j] = -s / l[i * N + i];
            }
        }
        // Hinv = Linv^T · Linv (upper triangle is enough; symmetric).
        // Hinv[a,b] = sum_k linv[k,a]*linv[k,b]; both lower so k>=max(a,b).
        let mut hinv = vec![0.0f64; N * N];
        for a in 0..N {
            for b in a..N {
                let mut s = 0.0f64;
                for k in b..N {
                    s += linv[k * N + a] * linv[k * N + b];
                }
                hinv[a * N + b] = s;
                hinv[b * N + a] = s;
            }
        }
        // Upper Cholesky of Hinv: U with U^T U = Hinv. Equivalent to
        // lower-Cholesky of the reversed matrix; simplest: compute the
        // standard upper-triangular Cholesky directly.
        let mut u = vec![0.0f64; N * N];
        let mut ok = true;
        for j in 0..N {
            // diagonal
            let mut diag = hinv[j * N + j];
            for k in 0..j {
                diag -= u[k * N + j] * u[k * N + j];
            }
            if !(diag > 0.0) {
                ok = false;
                break;
            }
            let ujj = diag.sqrt();
            u[j * N + j] = ujj;
            let inv = 1.0 / ujj;
            for i in (j + 1)..N {
                let mut s = hinv[j * N + i];
                for k in 0..j {
                    s -= u[k * N + j] * u[k * N + i];
                }
                u[j * N + i] = s * inv;
            }
        }
        if !ok {
            damp *= 10.0;
            continue;
        }
        // feedback[c,j] = U[c,j]/U[c,c] for j>=c (incl. c==j → 1.0).
        let mut fb = vec![0.0f64; N * N];
        for c in 0..N {
            let ucc = u[c * N + c];
            if !(ucc.abs() > 0.0) {
                return None;
            }
            let inv = 1.0 / ucc;
            for j in c..N {
                fb[c * N + j] = u[c * N + j] * inv;
            }
        }
        return Some(fb);
    }
    None
}

/// LDLQ-on-E8 for a single output row over one 256-block, parameterized on `n`.
///
/// `w_rot_block`  : the FWHT-rotated weights for this row's 256-block (f32[256]).
/// `row_scale_a`  : the frozen row scale (exact, unrounded) — used to reconstruct
///                  the (row*block) product for the feedback residual.
/// `inv_row_scale`: the frozen 1/row_scale_a (exact) — the FIRST reciprocal of the
///                  normalization, byte-matching the RTN row encoder.
/// `block_scale`  : the 8 frozen DECODED E4M3 block scales (one per 32-weight
///                  sub-block, a 256-block contains 8 sub-blocks).
/// `fb`           : the 256×256 OBS feedback matrix from `block_feedback`, or
///                  None to run pure RTN (degenerate-Hessian fallback).
/// `n`            : lattice bit-width (2, 3, or 4). Controls which E8 codec is used.
/// `quant_step`   : QUANT_STEP constant for the chosen n (QUANT_STEP for n=4,
///                  QUANT_STEP_MFP3 for n=3, QUANT_STEP_MFP2 for n=2).
/// Writes 32 E8 codewords (4 per 32-block × 8 sub-blocks) into `out_idx`.
/// For n=4 each codeword occupies 32 bits; for n=3 24 bits; for n=2 16 bits
/// (upper 32-8n bits are zero, guaranteed by encode_index_n — see e8.rs).
///
/// NORMALIZATION PARITY: the normalized target is computed as
///   v = w * inv_row_scale * inv_block_scale   (TWO separate reciprocals)
/// — byte-identical to the RTN encoder `quantize_mfpn_e8_row` (main.rs), NOT the
/// pre-multiplied `w / (row_scale*block_scale)` form (which differs by ≤1 ULP and
/// would make the fb=None == RTN parity gate seed-fragile). With fb=None this
/// yields byte-identical codewords to RTN; with fb=Some the same two-reciprocal
/// normalization is applied to the error-compensated target.
fn ldlq_row_block(
    w_rot_block: &[f32],
    row_scale_a: f32,
    inv_row_scale: f32,
    block_scale: &[f32; 8],
    fb: Option<&[f64]>,
    n: u32,
    quant_step: f32,
    out_idx: &mut [u32; 32],
) {
    const N: usize = 256;
    // Error-compensated residual targets in the ROTATED-WEIGHT domain.
    let mut w_res = [0.0f64; N];
    for i in 0..N {
        w_res[i] = w_rot_block[i] as f64;
    }
    // 32 groups of 8 columns, ascending.
    for g in 0..32 {
        let col0 = g * 8;
        let sub = g / 4; // which 32-weight sub-block (8 sub-blocks per 256)
        let bs = block_scale[sub]; // decoded E4M3 block scale for this sub-block
                                   // TWO-reciprocal normalization, byte-matching RTN (main.rs:1556).
        let inv_block_scale = if bs > 0.0 { 1.0 / bs } else { 0.0 };
        // (row_scale*block_scale) product — used ONLY for the feedback residual
        // (rotated-weight-domain reconstruction) and the per-column push cap.
        let s = (row_scale_a as f64) * (bs as f64);

        // Form the error-compensated, normalized 8-vector.
        let mut v = [0.0f32; 8];
        let mut over = false;
        for i in 0..8 {
            let wc = w_res[col0 + i] as f32;
            let vn = wc * inv_row_scale * inv_block_scale; // ≈[-6,6]
            v[i] = vn;
            if vn.abs() > V_CLAMP {
                over = true;
            }
        }
        // Feedback-runaway guard: if compensation blew the group past the
        // representable lattice extent, drop the feedback and round the raw
        // rotated weight (never worse than RTN by more than a lattice quantum).
        if over {
            for i in 0..8 {
                v[i] = w_rot_block[col0 + i] * inv_row_scale * inv_block_scale;
            }
        }

        // n-bit E8 nearest-point search: shared path for n=2,3,4.
        let idx = e8::quantize8_n(&v, quant_step, n);
        out_idx[g] = idx;

        // Dequant back to normalized then to rotated-weight domain.
        let dq = e8::dequantize8_n(idx, quant_step, n);
        // Per-column residual in rotated-weight space, propagate via feedback.
        if let Some(fbm) = fb {
            // Quantized value in rotated-weight space = dq * s.
            // Residual is computed against the COMPENSATED target we just
            // rounded (w_res), so the error that propagates is the part the
            // lattice could not absorb.
            for i in 0..8 {
                let c = col0 + i;
                let wq = dq[i] as f64 * s;
                let r = w_res[c] - wq; // rotated-weight-domain residual
                if r == 0.0 {
                    continue;
                }
                // Propagate to all later columns of THIS 256-block.
                let row = &fbm[c * N..c * N + N];
                // cap |E| push per column at 6*s (matches the validated n=4 GPTQ
                // path; |normalized v|<=6 by the /6.0 scale convention, so 6*s is
                // the rotated-domain lattice full-scale — n-loose but empirically
                // a no-op at n=3/2 per adversarial review, see module header).
                let cap = 6.0 * s;
                for j in (c + 1)..N {
                    let coeff = row[j];
                    if coeff == 0.0 {
                        continue;
                    }
                    let mut delta = r * coeff;
                    if delta > cap {
                        delta = cap;
                    } else if delta < -cap {
                        delta = -cap;
                    }
                    w_res[j] -= delta;
                }
            }
        }
    }
}

/// Frozen per-32-block scale derivation, IDENTICAL to quantize_mfp4g32_e8_row.
/// Returns (row_scale_a_f16_decoded, [block_scale_factor; n_blocks],
///          scale_bytes). All computed from the rotated row, frozen.
struct FrozenScales {
    row_scale_a: f32, // exact (unrounded) — used for inv in encode, matching RTN
    inv_row_scale: f32,
    scale_bytes: Vec<u8>,
    block_scale: Vec<f32>, // decoded e4m3 per 32-block
}

fn freeze_row_scales(
    row_rot: &[f32],
    e4m3_encode: &dyn Fn(f32) -> u8,
    e4m3_decode: &dyn Fn(u8) -> f32,
) -> FrozenScales {
    let k = row_rot.len();
    let n_blocks = k / 32;
    let row_max_abs = row_rot.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };
    let mut scale_bytes = vec![0u8; n_blocks];
    let mut block_scale = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let block = &row_rot[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let sb = e4m3_encode(s);
        scale_bytes[b] = sb;
        block_scale[b] = e4m3_decode(sb);
    }
    FrozenScales {
        row_scale_a,
        inv_row_scale,
        scale_bytes,
        block_scale,
    }
}

/// Full GPTQ-E8 quantize of one weight matrix W[m, k] (row-major), already in
/// the ORIGINAL (un-rotated) domain. Generic on `n` (2, 3, or 4 bits).
/// Mirrors quantize_mfpn_e8_2d byte-for-byte EXCEPT the lattice assignment uses
/// block-diagonal-256 vector-LDLQ against the provided per-256-block Hessians.
///
/// `h_blocks[b]` is the captured 256×256 (pre-rotation) Hessian for input
/// 256-block b (b in 0..k/256). If `h_blocks` is empty OR a block's Hessian is
/// degenerate, that block falls back to RTN — guaranteeing GPTQ-E8 is never
/// worse than RTN by more than a lattice quantum.
///
/// `cpu_fwht`, `e4m3_encode`, `e4m3_decode`, `f32_to_f16` are passed in from
/// main.rs so this module reuses the EXACT production helpers (no drift).
///
/// Per-block layout (per-32-block payload):
///   n=4: 1 scale byte + 4 × 4-byte codewords = 17 B
///   n=3: 1 scale byte + 4 × 3-byte codewords = 13 B
///   n=2: 1 scale byte + 4 × 2-byte codewords =  9 B
/// This matches `quantize_mfpn_e8_row` (RTN) exactly.
#[allow(clippy::too_many_arguments)]
fn quantize_mfpn_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[HBlock],
    n: u32,
    quant_step: f32,
    cpu_fwht: &(dyn Fn(&mut [f32], &[f32], &[f32]) + Sync),
    e4m3_encode: &(dyn Fn(f32) -> u8 + Sync),
    e4m3_decode: &(dyn Fn(u8) -> f32 + Sync),
    f32_to_f16: &(dyn Fn(f32) -> u16 + Sync),
) -> Vec<u8> {
    assert!(
        n == 2 || n == 3 || n == 4,
        "n must be 2, 3, or 4; got n={}",
        n
    );
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfpN-E8 requires k%256==0, got k={}", k);
    let n_256 = k / 256;
    let n_32 = k / 32;
    // Per-block payload: 1 E4M3 scale byte + 4 codewords of n bytes each.
    let block_bytes = 1 + 4 * n as usize;
    let row_bytes = 16 + block_bytes * n_32;

    // Precompute per-256-block feedback ONCE (shared across all m rows).
    // None means RTN for that block.
    let block_fb: Vec<Option<Vec<f64>>> = (0..n_256)
        .map(|b| {
            if b < h_blocks.len() && h_blocks[b].len() == 256 * 256 {
                let hp = rotate_hblock(&h_blocks[b], signs1, signs2);
                block_feedback(&hp)
            } else {
                None
            }
        })
        .collect();

    let mut out = vec![0u8; m * row_bytes];

    // Rows are independent once the block feedback matrices are frozen.
    // Parallel row execution is byte-deterministic and is essential for
    // single, tall tensors such as a 69K-vocabulary output head.
    use rayon::prelude::*;
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(r, row_out)| {
            let mut row_buf = vec![0.0f32; k];
            row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
            for seg in 0..n_256 {
                cpu_fwht(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            // Freeze scales from the rotated row (identical to RTN).
            let fs = freeze_row_scales(&row_buf, e4m3_encode, e4m3_decode);

            row_out[..2].copy_from_slice(&f32_to_f16(fs.row_scale_a).to_le_bytes());
            row_out[2..4].copy_from_slice(&0u16.to_le_bytes());
            row_out[4..6].copy_from_slice(&(n_32 as u16).to_le_bytes());
            row_out[6] = 0x05; // same FWHT flag as RTN path
            row_out[7] = 0u8;

            // Per 256-block LDLQ.
            for b256 in 0..n_256 {
                let w_rot_block = &row_buf[b256 * 256..b256 * 256 + 256];
                // 8 sub-blocks of 32; their frozen DECODED E4M3 block scales.
                // The row scale + the two reciprocals are derived inside
                // ldlq_row_block to byte-match the RTN normalization exactly.
                let mut block_scale = [0.0f32; 8];
                for sub in 0..8 {
                    let blk = b256 * 8 + sub;
                    block_scale[sub] = fs.block_scale[blk];
                }
                let mut idx32 = [0u32; 32];
                ldlq_row_block(
                    w_rot_block,
                    fs.row_scale_a,
                    fs.inv_row_scale,
                    &block_scale,
                    block_fb[b256].as_deref(),
                    n,
                    quant_step,
                    &mut idx32,
                );
                // Emit: 8 sub-blocks × (1 scale byte + 4 n-byte codewords).
                // Packing: encode_index_n guarantees the upper 32-8n bits of each
                // u32 codeword are ZERO — only the low n bytes are written, exactly
                // mirroring quantize_mfpn_e8_row (RTN path in main.rs).
                for sub in 0..8 {
                    let blk = b256 * 8 + sub;
                    let po = 16 + blk * block_bytes;
                    row_out[po] = fs.scale_bytes[blk];
                    for gg in 0..4 {
                        let idx = idx32[sub * 4 + gg];
                        let bytes = idx.to_le_bytes();
                        let cw_off = po + 1 + gg * n as usize;
                        row_out[cw_off..cw_off + n as usize].copy_from_slice(&bytes[..n as usize]);
                    }
                }
            }
        });
    out
}

/// GPTQ-E8 entry point for n=4 (mfp4-E8). Byte-layout identical to
/// `quantize_mfp4g32_e8_2d` (RTN). Delegates to the generic `_mfpn_` path.
#[allow(clippy::too_many_arguments)]
pub fn quantize_mfp4g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[HBlock],
    cpu_fwht: &(dyn Fn(&mut [f32], &[f32], &[f32]) + Sync),
    e4m3_encode: &(dyn Fn(f32) -> u8 + Sync),
    e4m3_decode: &(dyn Fn(u8) -> f32 + Sync),
    f32_to_f16: &(dyn Fn(f32) -> u16 + Sync),
) -> Vec<u8> {
    quantize_mfpn_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        4,
        e8::QUANT_STEP,
        cpu_fwht,
        e4m3_encode,
        e4m3_decode,
        f32_to_f16,
    )
}

/// GPTQ-E8 entry point for n=3 (mfp3-E8, 3.25 bpw). Byte-layout identical to
/// `quantize_mfp3g32_e8_2d` (RTN): 16-byte header + 13 B per 32-block.
/// With h_blocks empty (or degenerate Hessians) degrades to byte-identical RTN.
#[allow(clippy::too_many_arguments)]
pub fn quantize_mfp3g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[HBlock],
    cpu_fwht: &(dyn Fn(&mut [f32], &[f32], &[f32]) + Sync),
    e4m3_encode: &(dyn Fn(f32) -> u8 + Sync),
    e4m3_decode: &(dyn Fn(u8) -> f32 + Sync),
    f32_to_f16: &(dyn Fn(f32) -> u16 + Sync),
) -> Vec<u8> {
    quantize_mfpn_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        3,
        e8::QUANT_STEP_MFP3,
        cpu_fwht,
        e4m3_encode,
        e4m3_decode,
        f32_to_f16,
    )
}

/// GPTQ-E8 entry point for n=2 (mfp2-E8, 2.25 bpw). Byte-layout identical to
/// `quantize_mfp2g32_e8_2d` (RTN): 16-byte header + 9 B per 32-block.
/// With h_blocks empty (or degenerate Hessians) degrades to byte-identical RTN.
#[allow(clippy::too_many_arguments)]
pub fn quantize_mfp2g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[HBlock],
    cpu_fwht: &(dyn Fn(&mut [f32], &[f32], &[f32]) + Sync),
    e4m3_encode: &(dyn Fn(f32) -> u8 + Sync),
    e4m3_decode: &(dyn Fn(u8) -> f32 + Sync),
    f32_to_f16: &(dyn Fn(f32) -> u16 + Sync),
) -> Vec<u8> {
    quantize_mfpn_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        2,
        e8::QUANT_STEP_MFP2,
        cpu_fwht,
        e4m3_encode,
        e4m3_decode,
        f32_to_f16,
    )
}

// =====================================================================
// Synthetic discriminator tests (CPU-only, no GPU).
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- Local copies of the production helpers (kept bit-identical) ----
    fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                if (state >> 16) & 1 == 1 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect()
    }
    fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
        assert!(x.len() == 256);
        for i in 0..256 {
            x[i] *= signs1[i];
        }
        let mut stride = 1;
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
        let scale = 0.0625;
        for i in 0..256 {
            x[i] *= scale * signs2[i];
        }
    }
    fn e4m3_scale_decode(byte: u8) -> f32 {
        let exp = ((byte >> 3) & 0xf) as i32;
        let mant = (byte & 0x7) as u32;
        if exp == 0 {
            return (2.0f32).powi(-6) * (mant as f32) / 8.0;
        }
        if exp == 0xf && mant == 7 {
            return 448.0;
        }
        (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
    }
    fn e4m3_scale_encode_roundup(s: f32) -> u8 {
        if !(s > 0.0) {
            return 0x00;
        }
        if s >= 448.0 {
            return 0x7E;
        }
        for code in 0u8..=0x7E {
            if e4m3_scale_decode(code) >= s {
                return code;
            }
        }
        0x7E
    }
    fn f32_to_f16(val: f32) -> u16 {
        half_round(val)
    }
    // minimal round-to-nearest-even f32->f16 (good enough for tests; the
    // production path uses main.rs::f32_to_f16, but scales here are coarse).
    fn half_round(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7fffff;
        if exp <= 0 {
            return sign;
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        let mant16 = (mant >> 13) as u16;
        let _ = &mut exp;
        sign | ((exp as u16) << 10) | mant16
    }

    // RTN E8 reference (mirrors quantize_mfpn_e8_row). For n=4 uses the dedicated
    // quantize8/dequantize8 wrappers; for n=2,3 uses quantize8_n/dequantize8_n.
    fn rtn_row_codewords_n(row_rot: &[f32], n: u32, quant_step: f32) -> (Vec<u32>, Vec<f32>) {
        let k = row_rot.len();
        let n_blocks = k / 32;
        let row_max_abs = row_rot.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let row_scale_a = if row_max_abs > 0.0 {
            row_max_abs / 6.0
        } else {
            1.0
        };
        let inv_row_scale = if row_max_abs > 0.0 {
            1.0 / row_scale_a
        } else {
            0.0
        };
        let mut codes = Vec::with_capacity(n_blocks * 4);
        let mut recon = vec![0.0f32; k]; // rotated-domain reconstruction
        for b in 0..n_blocks {
            let block = &row_rot[b * 32..b * 32 + 32];
            let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
            let block_max_normalized = block_max_abs * inv_row_scale;
            let s = if block_max_normalized > 0.0 {
                block_max_normalized / 6.0
            } else {
                0.0
            };
            let sb = e4m3_scale_encode_roundup(s);
            let bsf = e4m3_scale_decode(sb);
            let inv_block_scale = if bsf > 0.0 { 1.0 / bsf } else { 0.0 };
            let scale = row_scale_a * bsf;
            for g in 0..4 {
                let mut v = [0.0f32; 8];
                for i in 0..8 {
                    v[i] = block[g * 8 + i] * inv_row_scale * inv_block_scale;
                }
                let idx = e8::quantize8_n(&v, quant_step, n);
                codes.push(idx);
                let dq = e8::dequantize8_n(idx, quant_step, n);
                for i in 0..8 {
                    recon[b * 32 + g * 8 + i] = dq[i] * scale;
                }
            }
        }
        (codes, recon)
    }

    // RTN E8 reference for n=4 (convenience wrapper kept for existing tests).
    fn rtn_row_codewords(row_rot: &[f32]) -> (Vec<u32>, Vec<f32>) {
        rtn_row_codewords_n(row_rot, 4, e8::QUANT_STEP)
    }

    // GPTQ-E8 single-tensor wrapper (generic on n), returning codewords + rotated recon.
    fn gptq_codewords_n(
        f32_data: &[f32],
        m: usize,
        k: usize,
        signs1: &[f32],
        signs2: &[f32],
        h_blocks: &[HBlock],
        n: u32,
        quant_step: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        let bytes = quantize_mfpn_e8_gptq_2d(
            f32_data,
            m,
            k,
            signs1,
            signs2,
            h_blocks,
            n,
            quant_step,
            &cpu_fwht_256,
            &e4m3_scale_encode_roundup,
            &e4m3_scale_decode,
            &f32_to_f16,
        );
        // Decode codewords + rotated recon from bytes.
        // Per-block payload: 1 E4M3 scale + 4 codewords of n bytes.
        let n_32 = k / 32;
        let block_bytes = 1 + 4 * n as usize;
        let row_bytes = 16 + block_bytes * n_32;
        let mut codes = Vec::with_capacity(m * n_32 * 4);
        let mut recon = vec![0.0f32; m * k];
        for r in 0..m {
            let base = r * row_bytes;
            let row_scale_a = {
                let b = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
                f16_to_f32(b)
            };
            for b in 0..n_32 {
                let po = base + 16 + b * block_bytes;
                let scale = row_scale_a * e4m3_scale_decode(bytes[po]);
                for g in 0..4 {
                    // Read n bytes and assemble u32 (upper bits are zero per encode_index_n).
                    let cw_off = po + 1 + g * n as usize;
                    let mut raw = [0u8; 4];
                    raw[..n as usize].copy_from_slice(&bytes[cw_off..cw_off + n as usize]);
                    let idx = u32::from_le_bytes(raw);
                    codes.push(idx);
                    let dq = e8::dequantize8_n(idx, quant_step, n);
                    for i in 0..8 {
                        recon[r * k + b * 32 + g * 8 + i] = dq[i] * scale;
                    }
                }
            }
        }
        (codes, recon)
    }

    // GPTQ-E8 n=4 convenience wrapper (used by existing tests).
    fn gptq_codewords(
        f32_data: &[f32],
        m: usize,
        k: usize,
        signs1: &[f32],
        signs2: &[f32],
        h_blocks: &[HBlock],
    ) -> (Vec<u32>, Vec<f32>) {
        gptq_codewords_n(f32_data, m, k, signs1, signs2, h_blocks, 4, e8::QUANT_STEP)
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let mant = (bits & 0x3ff) as u32;
        let f = if exp == 0 {
            if mant == 0 {
                sign << 31
            } else {
                // subnormal
                let mut e = exp as i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let fe = (e + (127 - 15) + 1) as u32;
                (sign << 31) | (fe << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | (0xff << 23) | (mant << 13)
        } else {
            let fe = exp + (127 - 15);
            (sign << 31) | (fe << 23) | (mant << 13)
        };
        f32::from_bits(f)
    }

    // Deterministic LCG (file convention).
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let hi = (*state >> 32) as u32;
        (hi as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    }

    // H-weighted error J = sum_rows (w_rot - wq)^T H' (w_rot - wq), with H'
    // the ROTATED Hessian (the objective LDLQ minimizes). Computed per
    // 256-block (block-diagonal).
    fn h_weighted_err(
        f32_data: &[f32],
        recon_rot: &[f32],
        m: usize,
        k: usize,
        signs1: &[f32],
        signs2: &[f32],
        h_blocks: &[HBlock],
    ) -> f64 {
        let n_256 = k / 256;
        // rotated weights
        let mut w_rot = vec![0.0f32; m * k];
        let mut buf = vec![0.0f32; k];
        for r in 0..m {
            buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
            for seg in 0..n_256 {
                cpu_fwht_256(&mut buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            w_rot[r * k..(r + 1) * k].copy_from_slice(&buf);
        }
        let mut j = 0.0f64;
        for r in 0..m {
            for b in 0..n_256 {
                let hp = rotate_hblock(&h_blocks[b], signs1, signs2);
                let mut e = [0.0f64; 256];
                for i in 0..256 {
                    let c = b * 256 + i;
                    e[i] = (w_rot[r * k + c] - recon_rot[r * k + c]) as f64;
                }
                for a in 0..256 {
                    if e[a] == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0f64;
                    for bb in 0..256 {
                        acc += hp[a * 256 + bb] * e[bb];
                    }
                    j += e[a] * acc;
                }
            }
        }
        j
    }

    fn unweighted_mse(
        f32_data: &[f32],
        recon_rot: &[f32],
        m: usize,
        k: usize,
        signs1: &[f32],
        signs2: &[f32],
    ) -> f64 {
        let n_256 = k / 256;
        let mut buf = vec![0.0f32; k];
        let mut acc = 0.0f64;
        for r in 0..m {
            buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
            for seg in 0..n_256 {
                cpu_fwht_256(&mut buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            for c in 0..k {
                let d = (buf[c] - recon_rot[r * k + c]) as f64;
                acc += d * d;
            }
        }
        acc / (m * k) as f64
    }

    // ------- TEST 0: R-rotation correctness (load-bearing identity) -------
    #[test]
    fn r_rotation_matches_explicit_matrix() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        // Build explicit R: column j = FWHT(e_j).
        let mut rmat = vec![0.0f64; 256 * 256];
        for j in 0..256 {
            let mut col = [0.0f64; 256];
            col[j] = 1.0;
            fwht_256_f64(&mut col, &signs1, &signs2);
            for i in 0..256 {
                rmat[i * 256 + j] = col[i];
            }
        }
        // R R^T == I to 1e-4
        let mut max_off = 0.0f64;
        for i in 0..256 {
            for j in 0..256 {
                let mut s = 0.0f64;
                for k in 0..256 {
                    s += rmat[i * 256 + k] * rmat[j * 256 + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                max_off = max_off.max((s - target).abs());
            }
        }
        assert!(
            max_off < 1e-4,
            "R not orthogonal: max|RR^T - I| = {max_off}"
        );

        // Random symmetric H = A A^T.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut a = vec![0.0f32; 256 * 256];
        for v in a.iter_mut() {
            *v = lcg_next(&mut state);
        }
        let mut h = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            for jj in 0..256 {
                let mut s = 0.0f64;
                for k in 0..256 {
                    s += a[i * 256 + k] as f64 * a[jj * 256 + k] as f64;
                }
                h[i * 256 + jj] = s as f32;
            }
        }
        // Hp_ref = R H R^T (explicit), Hp_impl = rotate_hblock.
        let mut rh = vec![0.0f64; 256 * 256];
        for i in 0..256 {
            for jj in 0..256 {
                let mut s = 0.0f64;
                for k in 0..256 {
                    s += rmat[i * 256 + k] * h[k * 256 + jj] as f64;
                }
                rh[i * 256 + jj] = s;
            }
        }
        let mut hp_ref = vec![0.0f64; 256 * 256];
        for i in 0..256 {
            for jj in 0..256 {
                let mut s = 0.0f64;
                for k in 0..256 {
                    s += rh[i * 256 + k] * rmat[jj * 256 + k];
                } // R^T col == R row
                hp_ref[i * 256 + jj] = s;
            }
        }
        let hp_impl = rotate_hblock(&h, &signs1, &signs2);
        let mut maxref = 0.0f64;
        let mut maxdiff = 0.0f64;
        for i in 0..256 * 256 {
            maxref = maxref.max(hp_ref[i].abs());
            maxdiff = maxdiff.max((hp_ref[i] - hp_impl[i]).abs());
        }
        assert!(
            maxdiff < 1e-3 * maxref,
            "rotate_hblock != R H R^T: maxdiff={maxdiff}, maxref={maxref}"
        );
    }

    // ------- TEST 1: DISCRIMINATOR — correlated H, GPTQ beats RTN -------
    #[test]
    fn correlated_hessian_gptq_beats_rtn() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let m = 64usize;
        let k = 256usize;
        // W = randn ~0.5 std (seeded).
        let mut state = 0xfeed_face_dead_beefu64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.9;
        }

        // Build a strongly-correlated rotated-domain H' (AR(1) rho=0.95),
        // then set the captured pre-rotation H = R^T H' R so the impl
        // recovers H' after rotate_hblock.
        let rho = 0.95f64;
        let var = 1.0f64;
        let mut hprime = vec![0.0f64; 256 * 256];
        for a in 0..256 {
            for b in 0..256 {
                hprime[a * 256 + b] = var * rho.powi((a as i32 - b as i32).abs());
            }
        }
        // H = R^T H' R. R^T x == apply R to columns? We use rotate by R^T then
        // R. Since rotate_hblock computes R H R^T, to recover H'=R H R^T we set
        // H = R^T H' R. Compute via explicit R.
        let mut rmat = vec![0.0f64; 256 * 256];
        for j in 0..256 {
            let mut col = [0.0f64; 256];
            col[j] = 1.0;
            fwht_256_f64(&mut col, &signs1, &signs2);
            for i in 0..256 {
                rmat[i * 256 + j] = col[i];
            }
        }
        // R^T H' : (R^T H')[i,j] = sum_k R[k,i] H'[k,j]
        let mut rth = vec![0.0f64; 256 * 256];
        for i in 0..256 {
            for j in 0..256 {
                let mut s = 0.0f64;
                for kk in 0..256 {
                    s += rmat[kk * 256 + i] * hprime[kk * 256 + j];
                }
                rth[i * 256 + j] = s;
            }
        }
        // (R^T H') R : [i,j] = sum_k rth[i,k] R[k,j]
        let mut hcap = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            for j in 0..256 {
                let mut s = 0.0f64;
                for kk in 0..256 {
                    s += rth[i * 256 + kk] * rmat[kk * 256 + j];
                }
                hcap[i * 256 + j] = s as f32;
            }
        }
        let h_blocks = vec![hcap];

        let (codes_rtn, recon_rtn) = {
            // RTN via the GPTQ path with empty Hessian (forces RTN fallback).
            gptq_codewords(&w, m, k, &signs1, &signs2, &[])
        };
        let (codes_gptq, recon_gptq) = gptq_codewords(&w, m, k, &signs1, &signs2, &h_blocks);

        // Sanity: codewords must round-trip to valid E8 (bijection).
        for &idx in codes_gptq.iter() {
            let p = e8::decode_index(idx);
            let idx2 = e8::encode_index(&p);
            assert_eq!(idx, idx2, "GPTQ codeword not a valid E8 bijection point");
        }
        // GPTQ must actually CHANGE some codewords vs RTN (anti-no-op).
        let changed = codes_rtn
            .iter()
            .zip(codes_gptq.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            changed > 0,
            "GPTQ produced byte-identical output to RTN on correlated H (silent no-op)"
        );

        let j_rtn = h_weighted_err(&w, &recon_rtn, m, k, &signs1, &signs2, &h_blocks);
        let j_gptq = h_weighted_err(&w, &recon_gptq, m, k, &signs1, &signs2, &h_blocks);
        eprintln!(
            "J_rtn={j_rtn:.6} J_gptq={j_gptq:.6} ratio={:.4} changed={changed}",
            j_gptq / j_rtn
        );
        assert!(
            j_gptq <= 0.90 * j_rtn,
            "GPTQ-E8 did not reduce H-weighted error by >=10%: J_gptq={j_gptq:.6} vs J_rtn={j_rtn:.6} (ratio {:.4})",
            j_gptq / j_rtn
        );

        // Unweighted (Euclidean) MSE is EXPECTED to rise a bit: GPTQ trades
        // Euclidean fidelity for H-weighted (output-error) fidelity. With this
        // strong (rho=0.95) correlation the trade is large; we only guard
        // against an INSANE blow-up (a real feedback runaway), not the
        // legitimate trade. The H-weighted bound above is the true gate.
        let mse_rtn = unweighted_mse(&w, &recon_rtn, m, k, &signs1, &signs2);
        let mse_gptq = unweighted_mse(&w, &recon_gptq, m, k, &signs1, &signs2);
        eprintln!(
            "mse_rtn={mse_rtn:.6} mse_gptq={mse_gptq:.6} (ratio {:.3})",
            mse_gptq / mse_rtn
        );
        assert!(
            mse_gptq <= 2.5 * mse_rtn,
            "GPTQ unweighted MSE blew up (runaway, not a legit H-trade): {mse_gptq:.6} vs {mse_rtn:.6}"
        );
    }

    // ------- TEST 2: DEGENERACY — H=I, GPTQ ~= RTN -------
    #[test]
    fn identity_hessian_equals_rtn() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let m = 32usize;
        let k = 256usize;
        let mut state = 0x0bad_c0de_1234_5678u64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.7;
        }

        // H = I  ⇒  after rotate (R I R^T = I) and damping, feedback is
        // diagonal ⇒ zero off-diagonal propagation ⇒ byte-identical to RTN.
        let mut hi = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            hi[i * 256 + i] = 1.0;
        }
        let h_blocks = vec![hi];

        let (codes_rtn, _) = gptq_codewords(&w, m, k, &signs1, &signs2, &[]);
        let (codes_gptq, _) = gptq_codewords(&w, m, k, &signs1, &signs2, &h_blocks);
        let changed = codes_rtn
            .iter()
            .zip(codes_gptq.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            changed, 0,
            "H=I must collapse to RTN exactly, but {changed} codewords differ (feedback firing on uncorrelated H)"
        );
    }

    // ------- TEST 3: byte-format / output validity (n=4) -------
    #[test]
    fn output_bytes_valid_format() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let m = 8usize;
        let k = 512usize; // 2 blocks of 256
        let mut state = 0xaaaa_bbbb_cccc_ddddu64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.6;
        }
        let bytes = quantize_mfp4g32_e8_gptq_2d(
            &w,
            m,
            k,
            &signs1,
            &signs2,
            &[],
            &cpu_fwht_256,
            &e4m3_scale_encode_roundup,
            &e4m3_scale_decode,
            &f32_to_f16,
        );
        let n_32 = k / 32;
        let block_bytes = 1 + 4 * 4usize; // n=4: 17 B
        let row_bytes = 16 + block_bytes * n_32;
        assert_eq!(bytes.len(), m * row_bytes, "wrong output byte length (n=4)");
        // Every block header flag must be 0x05.
        for r in 0..m {
            assert_eq!(bytes[r * row_bytes + 6], 0x05);
        }
    }

    // ------- TEST 4: PARITY GATE — GPTQ-mfp3 (fb=None) == RTN-mfp3 byte-for-byte -------
    //
    // CRITICAL INVARIANT: with an empty Hessian (h_blocks=[]), ldlq_row_block runs
    // pure RTN (fb=None path), so the GPTQ entry-point must produce byte-identical
    // output to the RTN path for the same weights. This guards against any divergence
    // between the GPTQ and RTN scale derivation or codeword packing for n=3.
    #[test]
    fn gptq_mfp3_none_hessian_matches_rtn_bytes() {
        let signs1 = gen_fwht_signs(17, 256);
        let signs2 = gen_fwht_signs(117, 256);
        let m = 16usize;
        let k = 512usize; // 2 × 256 blocks
        let mut state = 0x1234_abcd_ef01_2345u64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.8;
        }

        // GPTQ with no Hessian — must degrade to RTN.
        let gptq_bytes = quantize_mfp3g32_e8_gptq_2d(
            &w,
            m,
            k,
            &signs1,
            &signs2,
            &[], // empty h_blocks
            &cpu_fwht_256,
            &e4m3_scale_encode_roundup,
            &e4m3_scale_decode,
            &f32_to_f16,
        );

        // RTN reference: manually apply FWHT then call rtn_row_codewords_n.
        // We replicate what quantize_mfp3g32_e8_2d does (FWHT + row encode).
        let n_32 = k / 32;
        let block_bytes_n3 = 1 + 4 * 3usize; // 13 B
        let row_bytes_n3 = 16 + block_bytes_n3 * n_32;
        let mut rtn_bytes = vec![0u8; m * row_bytes_n3];
        {
            // Build RTN output by re-implementing quantize_mfpn_e8_row inline.
            let mut row_buf = vec![0.0f32; k];
            for r in 0..m {
                row_buf.copy_from_slice(&w[r * k..(r + 1) * k]);
                for seg in 0..(k / 256) {
                    cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], &signs1, &signs2);
                }
                let (codes, _) = rtn_row_codewords_n(&row_buf, 3, e8::QUANT_STEP_MFP3);
                // Re-emit to bytes using the same scale derivation.
                let row_max_abs = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
                let row_scale_a = if row_max_abs > 0.0 {
                    row_max_abs / 6.0
                } else {
                    1.0
                };
                let inv_rs = if row_max_abs > 0.0 {
                    1.0 / row_scale_a
                } else {
                    0.0
                };
                let base = r * row_bytes_n3;
                rtn_bytes[base..base + 2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
                rtn_bytes[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
                rtn_bytes[base + 4..base + 6].copy_from_slice(&(n_32 as u16).to_le_bytes());
                rtn_bytes[base + 6] = 0x05;
                rtn_bytes[base + 7] = 0u8;
                let mut code_idx = 0usize;
                for b in 0..n_32 {
                    let block = &row_buf[b * 32..b * 32 + 32];
                    let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
                    let bmax_n = bmax * inv_rs;
                    let s = if bmax_n > 0.0 { bmax_n / 6.0 } else { 0.0 };
                    let sb = e4m3_scale_encode_roundup(s);
                    let po = base + 16 + b * block_bytes_n3;
                    rtn_bytes[po] = sb;
                    for g in 0..4 {
                        let idx = codes[code_idx];
                        code_idx += 1;
                        let bytes = idx.to_le_bytes();
                        let cw_off = po + 1 + g * 3;
                        rtn_bytes[cw_off..cw_off + 3].copy_from_slice(&bytes[..3]);
                    }
                }
            }
        }

        assert_eq!(
            gptq_bytes.len(),
            rtn_bytes.len(),
            "GPTQ-mfp3 (fb=None) output length mismatch: {} vs {}",
            gptq_bytes.len(),
            rtn_bytes.len()
        );
        // Byte-exact comparison — any difference is a packing or scale divergence bug.
        let diffs: usize = gptq_bytes
            .iter()
            .zip(rtn_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "GPTQ-mfp3 (fb=None) is NOT byte-identical to RTN-mfp3: {diffs} byte(s) differ. \
             Layout parity BROKEN — the GPTQ path uses a different scale or codeword packing than RTN."
        );
    }

    // Build the full RTN-mfpN-E8 byte stream for `f32_data`, mirroring the
    // production `quantize_mfpn_e8_2d` (FWHT + per-row encode via the
    // two-reciprocal `rtn_row_codewords_n` normalization).  Used by the
    // multi-seed parity sweep below so it compares against the EXACT RTN bytes.
    fn rtn_mfpn_e8_2d_bytes(
        f32_data: &[f32],
        m: usize,
        k: usize,
        signs1: &[f32],
        signs2: &[f32],
        n: u32,
        quant_step: f32,
    ) -> Vec<u8> {
        let n_32 = k / 32;
        let block_bytes = 1 + 4 * n as usize;
        let row_bytes = 16 + block_bytes * n_32;
        let mut out = vec![0u8; m * row_bytes];
        let mut row_buf = vec![0.0f32; k];
        for r in 0..m {
            row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            let (codes, _) = rtn_row_codewords_n(&row_buf, n, quant_step);
            let row_max_abs = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let row_scale_a = if row_max_abs > 0.0 {
                row_max_abs / 6.0
            } else {
                1.0
            };
            let inv_rs = if row_max_abs > 0.0 {
                1.0 / row_scale_a
            } else {
                0.0
            };
            let base = r * row_bytes;
            out[base..base + 2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
            out[base + 4..base + 6].copy_from_slice(&(n_32 as u16).to_le_bytes());
            out[base + 6] = 0x05;
            out[base + 7] = 0u8;
            let mut code_idx = 0usize;
            for b in 0..n_32 {
                let block = &row_buf[b * 32..b * 32 + 32];
                let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
                let bmax_n = bmax * inv_rs;
                let s = if bmax_n > 0.0 { bmax_n / 6.0 } else { 0.0 };
                let sb = e4m3_scale_encode_roundup(s);
                let po = base + 16 + b * block_bytes;
                out[po] = sb;
                for g in 0..4 {
                    let idx = codes[code_idx];
                    code_idx += 1;
                    let bytes = idx.to_le_bytes();
                    let cw_off = po + 1 + g * n as usize;
                    out[cw_off..cw_off + n as usize].copy_from_slice(&bytes[..n as usize]);
                }
            }
        }
        out
    }

    // ------- TEST 5b: PARITY GATE (MULTI-SEED) — fb=None == RTN across MANY seeds -------
    //
    // The single-seed gates (TEST 4/5) catch a *structural* layout/scale bug but
    // pass by seed luck against a ~1e-7-per-group ULP normalization drift (a prior
    // GPTQ build used the combined `w/(row_scale*block_scale)` reciprocal which
    // differs from RTN's two-reciprocal `w*inv_row*inv_block` by <=1 ULP, flipping
    // ~3-in-10,000 random seeds). This sweep makes the parity gate DETERMINISTIC:
    // with the correct two-reciprocal normalization, EVERY seed is byte-identical.
    // If anyone reintroduces the combined form, this test reds (it found 8/2000
    // failing-seed deltas empirically against the broken arithmetic).
    #[test]
    fn gptq_mfp3_mfp2_none_hessian_matches_rtn_multiseed() {
        let signs1 = gen_fwht_signs(31, 256);
        let signs2 = gen_fwht_signs(131, 256);
        let m = 16usize;
        let k = 512usize;
        for &(n, qs) in &[(3u32, e8::QUANT_STEP_MFP3), (2u32, e8::QUANT_STEP_MFP2)] {
            let mut total_diffs = 0usize;
            let mut failing_seeds: Vec<u64> = Vec::new();
            for seed in 0u64..2000 {
                let mut state = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678);
                let mut w = vec![0.0f32; m * k];
                for v in w.iter_mut() {
                    *v = lcg_next(&mut state) * 0.8;
                }

                let gptq = if n == 3 {
                    quantize_mfp3g32_e8_gptq_2d(
                        &w,
                        m,
                        k,
                        &signs1,
                        &signs2,
                        &[],
                        &cpu_fwht_256,
                        &e4m3_scale_encode_roundup,
                        &e4m3_scale_decode,
                        &f32_to_f16,
                    )
                } else {
                    quantize_mfp2g32_e8_gptq_2d(
                        &w,
                        m,
                        k,
                        &signs1,
                        &signs2,
                        &[],
                        &cpu_fwht_256,
                        &e4m3_scale_encode_roundup,
                        &e4m3_scale_decode,
                        &f32_to_f16,
                    )
                };
                let rtn = rtn_mfpn_e8_2d_bytes(&w, m, k, &signs1, &signs2, n, qs);
                let d = gptq.iter().zip(rtn.iter()).filter(|(a, b)| a != b).count();
                if d > 0 {
                    total_diffs += d;
                    if failing_seeds.len() < 8 {
                        failing_seeds.push(seed);
                    }
                }
            }
            assert_eq!(
                total_diffs,
                0,
                "GPTQ-mfp{n} (fb=None) diverged from RTN-mfp{n} on {} seed(s) ({} total byte diffs); \
                 examples: {:?}. Normalization parity BROKEN — the fb=None path must use the SAME \
                 two-reciprocal `w*inv_row*inv_block` form as the RTN encoder, NOT a combined reciprocal.",
                failing_seeds.len(),
                total_diffs,
                failing_seeds
            );
        }
    }

    // ------- TEST 5: PARITY GATE — GPTQ-mfp2 (fb=None) == RTN-mfp2 byte-for-byte -------
    #[test]
    fn gptq_mfp2_none_hessian_matches_rtn_bytes() {
        let signs1 = gen_fwht_signs(21, 256);
        let signs2 = gen_fwht_signs(121, 256);
        let m = 16usize;
        let k = 512usize;
        let mut state = 0xdead_beef_cafe_f00du64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.8;
        }

        let gptq_bytes = quantize_mfp2g32_e8_gptq_2d(
            &w,
            m,
            k,
            &signs1,
            &signs2,
            &[], // empty h_blocks
            &cpu_fwht_256,
            &e4m3_scale_encode_roundup,
            &e4m3_scale_decode,
            &f32_to_f16,
        );

        let n_32 = k / 32;
        let block_bytes_n2 = 1 + 4 * 2usize; // 9 B
        let row_bytes_n2 = 16 + block_bytes_n2 * n_32;
        let mut rtn_bytes = vec![0u8; m * row_bytes_n2];
        {
            let mut row_buf = vec![0.0f32; k];
            for r in 0..m {
                row_buf.copy_from_slice(&w[r * k..(r + 1) * k]);
                for seg in 0..(k / 256) {
                    cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], &signs1, &signs2);
                }
                let (codes, _) = rtn_row_codewords_n(&row_buf, 2, e8::QUANT_STEP_MFP2);
                let row_max_abs = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
                let row_scale_a = if row_max_abs > 0.0 {
                    row_max_abs / 6.0
                } else {
                    1.0
                };
                let inv_rs = if row_max_abs > 0.0 {
                    1.0 / row_scale_a
                } else {
                    0.0
                };
                let base = r * row_bytes_n2;
                rtn_bytes[base..base + 2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
                rtn_bytes[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
                rtn_bytes[base + 4..base + 6].copy_from_slice(&(n_32 as u16).to_le_bytes());
                rtn_bytes[base + 6] = 0x05;
                rtn_bytes[base + 7] = 0u8;
                let mut code_idx = 0usize;
                for b in 0..n_32 {
                    let block = &row_buf[b * 32..b * 32 + 32];
                    let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
                    let bmax_n = bmax * inv_rs;
                    let s = if bmax_n > 0.0 { bmax_n / 6.0 } else { 0.0 };
                    let sb = e4m3_scale_encode_roundup(s);
                    let po = base + 16 + b * block_bytes_n2;
                    rtn_bytes[po] = sb;
                    for g in 0..4 {
                        let idx = codes[code_idx];
                        code_idx += 1;
                        let bytes = idx.to_le_bytes();
                        let cw_off = po + 1 + g * 2;
                        rtn_bytes[cw_off..cw_off + 2].copy_from_slice(&bytes[..2]);
                    }
                }
            }
        }

        assert_eq!(
            gptq_bytes.len(),
            rtn_bytes.len(),
            "GPTQ-mfp2 (fb=None) output length mismatch: {} vs {}",
            gptq_bytes.len(),
            rtn_bytes.len()
        );
        let diffs: usize = gptq_bytes
            .iter()
            .zip(rtn_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "GPTQ-mfp2 (fb=None) is NOT byte-identical to RTN-mfp2: {diffs} byte(s) differ."
        );
    }

    // ------- TEST 6: GPTQ-mfp3/mfp2 decode via decode_index_n gives finite values -------
    //
    // Verifies that GPTQ output with a real Hessian decodes to finite f32 via the
    // SAME decode_index_n path the kernels use, and that the layout (header + n-byte
    // codeword packing) is readable identically to the RTN format.
    #[test]
    fn gptq_mfp3_mfp2_decode_finite() {
        let signs1 = gen_fwht_signs(99, 256);
        let signs2 = gen_fwht_signs(199, 256);
        let m = 8usize;
        let k = 256usize;
        let mut state = 0x5555_aaaa_5555_aaaau64;
        let mut w = vec![0.0f32; m * k];
        for v in w.iter_mut() {
            *v = lcg_next(&mut state) * 0.7;
        }

        // Build a mildly correlated H (same form as TEST 1).
        let rho = 0.5f64;
        let mut hp = vec![0.0f64; 256 * 256];
        for a in 0..256 {
            for b in 0..256 {
                hp[a * 256 + b] = rho.powi((a as i32 - b as i32).abs());
            }
        }
        // H = R^T H' R.
        let mut rmat = vec![0.0f64; 256 * 256];
        for j in 0..256 {
            let mut col = [0.0f64; 256];
            col[j] = 1.0;
            fwht_256_f64(&mut col, &signs1, &signs2);
            for i in 0..256 {
                rmat[i * 256 + j] = col[i];
            }
        }
        let mut rth = vec![0.0f64; 256 * 256];
        for i in 0..256 {
            for j in 0..256 {
                let mut s = 0.0f64;
                for kk in 0..256 {
                    s += rmat[kk * 256 + i] * hp[kk * 256 + j];
                }
                rth[i * 256 + j] = s;
            }
        }
        let mut hcap = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            for j in 0..256 {
                let mut s = 0.0f64;
                for kk in 0..256 {
                    s += rth[i * 256 + kk] * rmat[kk * 256 + j];
                }
                hcap[i * 256 + j] = s as f32;
            }
        }
        let h_blocks = vec![hcap];

        for &(n, qs) in &[(3u32, e8::QUANT_STEP_MFP3), (2u32, e8::QUANT_STEP_MFP2)] {
            let (codes, recon) = gptq_codewords_n(&w, m, k, &signs1, &signs2, &h_blocks, n, qs);
            // All codewords must decode via decode_index_n without NaN/Inf.
            for (i, &idx) in codes.iter().enumerate() {
                let p = e8::decode_index_n(idx, n);
                for (j, &c) in p.iter().enumerate() {
                    assert!(c.is_finite(), "n={n} code[{i}] coord[{j}] not finite: {c}");
                }
            }
            // Recon must also be finite (scale * lattice point).
            for (i, &v) in recon.iter().enumerate() {
                assert!(v.is_finite(), "n={n} recon[{i}] not finite: {v}");
            }
        }
    }
}
