// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Portable correctness harness for the HFQ4-G256 i8-WMMA MMQ GEMM, run on
//! whatever GPU `rdna_compute::Gpu::init()` brings up. Cross-checks both MMQ
//! entry points against a pure-CPU dequant oracle:
//!
//!   * `gemm_hfq4g256_residual_mmq`  — add=1, Y pre-zeroed → Y == W·X
//!   * `gemm_hfq4g256_mmq_set`       — add=0 (set)         → Y == W·X
//!
//! over several shapes, including a deliberately partial tile (130×256×70) that
//! exercises the bounds-clamped (non-`_full`) kernel on both RDNA3 (128-row LDS
//! tile) and gfx12 (16-row single-wave tile).
//!
//! Why a CPU oracle and NOT `gemm_hfq4g256_wmma` as a reference: the f16 WMMA
//! intrinsic `llvm.amdgcn.wmma.f32.16x16x16.f16` used by that kernel does NOT
//! JIT on gfx12, so it cannot serve as a gfx12 ground truth. The CPU oracle
//! works on every arch. (`gemm_hfq4g256_residual_fp16_wave64` would also work
//! as an on-GPU reference, but the CPU oracle keeps this harness device-free.)
//!
//! Regression this guards: the gfx12 MMQ module-name cache collision, where
//! `ensure_q8_1_mmq_x` pre-loads the RDNA3 source under module
//! "gemm_hfq4g256_residual_mmq" and the cache (keyed by module name only)
//! short-circuits the later gfx12-source load → the gfx12 #if-excluded body
//! resolves to an empty stub → the kernel writes NOTHING and Y stays at its
//! input value. With add=1 over a zeroed Y, the stub signature is therefore
//! `output == input buffer` (all zeros); we flag that explicitly.

use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;
const NRMSE_THRESHOLD: f32 = 0.01; // 1%

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            eprintln!("HFQ4G256 MMQ PORTABLE PASS");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("HFQ4G256 MMQ PORTABLE SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("HFQ4G256 MMQ PORTABLE FAIL: {msg}");
            ExitCode::from(1)
        }
    }
}

enum Outcome {
    Skip(String),
    Fail(String),
}

fn run() -> Result<(), Outcome> {
    let mut gpu = rdna_compute::Gpu::init()
        .map_err(|e| Outcome::Skip(format!("GPU init unavailable: {e}")))?;
    eprintln!("arch = {}", gpu.arch);

    // (m, k, n): rows of W, contraction, batch (= rows of X).
    //   - 128×256×128  : exact full tile on both RDNA3 (128/128) and gfx12 (16/16)
    //   -  16×256× 16  : minimal full tile on gfx12
    //   - 130×256× 70  : PARTIAL tile (bounds-clamped, non-`_full`) on both archs
    //   - 256×512×48   : multi-K-block (K=512 → 2 G256 blocks), partial N
    let shapes: [(usize, usize, usize); 4] =
        [(128, 256, 128), (16, 256, 16), (130, 256, 70), (256, 512, 48)];

    let mut any_fail = false;
    for &(m, k, n) in &shapes {
        match check_shape(&mut gpu, m, k, n) {
            Ok(()) => {}
            Err(Outcome::Skip(msg)) => return Err(Outcome::Skip(msg)),
            Err(Outcome::Fail(msg)) => {
                eprintln!("  FAIL [{m}x{k}x{n}]: {msg}");
                any_fail = true;
            }
        }
    }

    if any_fail {
        return Err(Outcome::Fail("one or more shapes failed".to_string()));
    }
    Ok(())
}

fn check_shape(gpu: &mut rdna_compute::Gpu, m: usize, k: usize, n: usize) -> Result<(), Outcome> {
    assert!(k % 256 == 0, "K must be a multiple of the G256 group size");

    // Deterministic pseudo-random weights and activations.
    let mut w = vec![0.0f32; m * k];
    for row in 0..m {
        for col in 0..k {
            w[row * k + col] = ((row * 31 + col * 17) as f32 * 0.013).sin() * 0.7
                + (row as f32 * 0.005)
                - 0.35;
        }
    }
    let mut x = vec![0.0f32; n * k];
    for r in 0..n {
        for col in 0..k {
            x[r * k + col] = ((r * 23 + col * 11) as f32 * 0.017).cos() * 0.5;
        }
    }

    let q = quantize_hfq4g256(&w, k);

    // CPU oracle over the *dequantized* weights (matches what the kernel sees),
    // laid out column-major [n * m + row] to match the kernel's Y layout.
    let oracle = cpu_dequant_matmul(&q, &x, m, k, n);

    let d_a = gpu
        .upload_raw(&q, &[q.len()])
        .map_err(|e| Outcome::Fail(format!("upload A: {e}")))?;
    let d_x = gpu
        .upload_f32(&x, &[n * k])
        .map_err(|e| Outcome::Fail(format!("upload X: {e}")))?;

    // ── Path 1: residual MMQ, add=1, Y pre-zeroed → Y should equal W·X ──────
    let d_y_add = gpu
        .zeros(&[n * m], rdna_compute::DType::F32)
        .map_err(|e| Outcome::Fail(format!("alloc Y(add): {e}")))?;
    let y_in_add = gpu
        .download_f32(&d_y_add)
        .map_err(|e| Outcome::Fail(format!("download Y(add) pre: {e}")))?;
    gpu.gemm_hfq4g256_residual_mmq(&d_a, &d_x, &d_y_add, m, k, n)
        .map_err(|e| Outcome::Fail(format!("gemm_hfq4g256_residual_mmq: {e}")))?;
    let y_add = gpu
        .download_f32(&d_y_add)
        .map_err(|e| Outcome::Fail(format!("download Y(add): {e}")))?;

    // Stub signature: an empty-stub kernel writes nothing → output == input
    // buffer (the zeroed Y we passed in).
    if buffers_equal(&y_add, &y_in_add) {
        return Err(Outcome::Fail(
            "STUB SIGNATURE: residual_mmq output == input buffer (kernel wrote nothing — \
             gfx12 module-name cache collision / empty stub)"
                .to_string(),
        ));
    }
    let nrmse_add = nrmse(&y_add, &oracle);

    // ── Path 2: mmq_set, add=0 (set) → Y should equal W·X regardless of init ─
    let d_y_set = gpu
        .zeros(&[n * m], rdna_compute::DType::F32)
        .map_err(|e| Outcome::Fail(format!("alloc Y(set): {e}")))?;
    gpu.gemm_hfq4g256_mmq_set(&d_a, &d_x, &d_y_set, m, k, n)
        .map_err(|e| Outcome::Fail(format!("gemm_hfq4g256_mmq_set: {e}")))?;
    let y_set = gpu
        .download_f32(&d_y_set)
        .map_err(|e| Outcome::Fail(format!("download Y(set): {e}")))?;
    let nrmse_set = nrmse(&y_set, &oracle);

    eprintln!(
        "  [{m}x{k}x{n}] arch={} nrmse_add={:.6} nrmse_set={:.6}",
        gpu.arch, nrmse_add, nrmse_set
    );

    gpu.free_tensor(d_a)
        .map_err(|e| Outcome::Fail(format!("free A: {e}")))?;
    gpu.free_tensor(d_x)
        .map_err(|e| Outcome::Fail(format!("free X: {e}")))?;
    gpu.free_tensor(d_y_add)
        .map_err(|e| Outcome::Fail(format!("free Y(add): {e}")))?;
    gpu.free_tensor(d_y_set)
        .map_err(|e| Outcome::Fail(format!("free Y(set): {e}")))?;

    if nrmse_add > NRMSE_THRESHOLD {
        return Err(Outcome::Fail(format!(
            "residual_mmq (add=1) NRMSE {:.6} > {:.6}",
            nrmse_add, NRMSE_THRESHOLD
        )));
    }
    if nrmse_set > NRMSE_THRESHOLD {
        return Err(Outcome::Fail(format!(
            "mmq_set (add=0) NRMSE {:.6} > {:.6}",
            nrmse_set, NRMSE_THRESHOLD
        )));
    }
    Ok(())
}

/// Two buffers are bit-for-bit equal (used for stub detection: kernel wrote
/// nothing → output buffer unchanged from its input value).
fn buffers_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Normalized RMSE: sqrt(mean((a-b)^2)) / (rms(b) + eps).
fn nrmse(got: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(got.len(), reference.len());
    let n = got.len() as f32;
    let mse: f32 = got
        .iter()
        .zip(reference)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / n;
    let ref_ms: f32 = reference.iter().map(|b| b * b).sum::<f32>() / n;
    (mse.sqrt()) / (ref_ms.sqrt() + 1e-8)
}

/// CPU reference: dequantize the HFQ4-G256 weight blocks and run the GEMM,
/// producing the column-major [n*m + row] layout the MMQ kernels write.
fn cpu_dequant_matmul(q: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let block_bytes = 136usize;
    let group = 256usize;
    let blocks_per_row = k / group;
    let mut out = vec![0.0f32; n * m];
    for row in 0..m {
        // Dequantize this row's weights once.
        let mut wrow = vec![0.0f32; k];
        for b in 0..blocks_per_row {
            let off = (row * blocks_per_row + b) * block_bytes;
            let scale = f32::from_le_bytes([q[off], q[off + 1], q[off + 2], q[off + 3]]);
            let zero = f32::from_le_bytes([q[off + 4], q[off + 5], q[off + 6], q[off + 7]]);
            for i in 0..group {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    q[off + 8 + byte_idx] & 0xF
                } else {
                    q[off + 8 + byte_idx] >> 4
                };
                wrow[b * group + i] = scale * nibble as f32 + zero;
            }
        }
        for r in 0..n {
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += wrow[col] * x[r * k + col];
            }
            out[r * m + row] = acc;
        }
    }
    out
}

/// HFQ4-G256 affine 4-bit quantizer (136-byte blocks: f32 scale, f32 min,
/// 128 packed-nibble bytes per 256-element group). Identical layout to the
/// project QA helper; `k` may span multiple G256 groups per row.
fn quantize_hfq4g256(f32_data: &[f32], _k: usize) -> Vec<u8> {
    let group_size = 256usize;
    let block_bytes = 136usize;
    let n_blocks = (f32_data.len() + group_size - 1) / group_size;
    let mut out = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(f32_data.len());
        let grp = &f32_data[start..end];
        let min_val = grp.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = grp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let off = b * block_bytes;
        out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
        out[off + 4..off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..128 {
            let lo_idx = 2 * i;
            let hi_idx = 2 * i + 1;
            let lo_val = if lo_idx < actual_len { grp[lo_idx] } else { min_val };
            let hi_val = if hi_idx < actual_len { grp[hi_idx] } else { min_val };
            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;
            out[off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }
    out
}
