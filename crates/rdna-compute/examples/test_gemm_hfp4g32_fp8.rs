//! Correctness + perf test for the gfx12 FP8-WMMA QKV HFP4G32 kernel.
//!
//! Compares `gemm_qkv_hfp4g32_wmma_fp8_gfx12` against the FP16-WMMA
//! sibling `gemm_qkv_hfp4g32_wmma_gfx12` (already validated against the
//! gemv scalar reference by `test_gemm_hfp4g32`). FP8 path uses E4M3
//! for both operands so per-element precision is ~12.5%; over K=1024
//! accumulation the relative error should be sub-1%, but we allow 5%
//! to cover worst-case rows.
//!
//! Skips silently with PASS on non-gfx12 archs.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        eprintln!("=== SKIP === FP8 WMMA is gfx12-only (arch={arch})");
        return;
    }
    eprintln!("=== gemm_qkv_hfp4g32_wmma_fp8_gfx12 ===");
    eprintln!("  arch={arch}");

    // Sweep both small N (decode-batch-like) and realistic prefill N
    // (256-2048) at production K=2048 to see whether the constant
    // per-warp dequant overhead amortizes against the 1.87x raw WMMA
    // throughput advantage when M tile count grows.
    let n_list: Vec<usize> = vec![2048, 512, 256, 64, 16, 4, 1];
    let k: usize = 2048;
    let q_m: usize = 2048;
    let k_m: usize = 512;
    let v_m: usize = 512;
    let row_bytes = 16 + (k / 32) * 17;

    let w_q = gpu.upload_raw(&synth(q_m, k, 0xAA), &[q_m * row_bytes]).unwrap();
    let w_k = gpu.upload_raw(&synth(k_m, k, 0xBB), &[k_m * row_bytes]).unwrap();
    let w_v = gpu.upload_raw(&synth(v_m, k, 0xCC), &[v_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = make_x(max_n * k, 0x1111);

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_q_ref = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_ref = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_ref = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();
    let y_q_fp8 = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_fp8 = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_fp8 = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    // n_list descending order: ensure_fp16_x is keyed by src_ptr — the
    // first call at max_n fills the scratch; later smaller N reuses it.
    let mut all_pass = true;
    for &n in &n_list {
        // FP16 reference (direct entry point — bypasses HIPFIRE_FP8_WMMA gate).
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkv_hfp4g32_wmma_gfx12(
            &w_q, &w_k, &w_v, &x_gemm,
            &y_q_ref, &y_k_ref, &y_v_ref,
            q_m, k_m, v_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let f16_us = t.elapsed().as_secs_f64() * 1e6;

        // FP8 candidate.
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkv_hfp4g32_wmma_fp8_gfx12(
            &w_q, &w_k, &w_v, &x_gemm,
            &y_q_fp8, &y_k_fp8, &y_v_fp8,
            q_m, k_m, v_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let fp8_us = t.elapsed().as_secs_f64() * 1e6;

        let ok_q = cmp_tol(&mut gpu, &y_q_ref, &y_q_fp8, n, q_m, "q");
        let ok_k = cmp_tol(&mut gpu, &y_k_ref, &y_k_fp8, n, k_m, "k");
        let ok_v = cmp_tol(&mut gpu, &y_v_ref, &y_v_fp8, n, v_m, "v");
        let ok = ok_q && ok_k && ok_v;
        eprintln!(
            "  N={n:3}  fp16: {:8.1} µs   fp8: {:8.1} µs   speedup: {:5.2}x   [{}]",
            f16_us, fp8_us, f16_us / fp8_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

fn cmp_tol(gpu: &mut Gpu, y_ref: &GpuTensor, y_kernel: &GpuTensor, n: usize, m: usize, label: &str) -> bool {
    let r = gpu.download_f32(y_ref).unwrap();
    let k = gpu.download_f32(y_kernel).unwrap();

    let mut max_abs: f64 = 0.0;
    let mut max_abs_ref: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut sum_sq_ref: f64 = 0.0;

    for b in 0..n {
        for row in 0..m {
            let r_v = r[b * m + row] as f64;
            let k_v = k[b * m + row] as f64;
            let abs = (r_v - k_v).abs();
            if abs > max_abs { max_abs = abs; }
            if r_v.abs() > max_abs_ref { max_abs_ref = r_v.abs(); }
            sum_sq_err += (r_v - k_v) * (r_v - k_v);
            sum_sq_ref += r_v * r_v;
        }
    }
    let nrmse = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
    // 5% relative tol on element max, plus 1e-3 abs floor. NRMSE shown for context.
    let tol_abs = 0.05 * max_abs_ref.max(1e-3);
    let mut bad = 0usize;
    for b in 0..n {
        for row in 0..m {
            let r_v = r[b * m + row] as f64;
            let k_v = k[b * m + row] as f64;
            if (r_v - k_v).abs() > tol_abs { bad += 1; }
        }
    }

    if bad > 0 {
        eprintln!(
            "  {label}: FAIL  {bad} elements outside tol  max_abs={:.3e}  tol={:.3e}  max_|y_ref|={:.3e}  NRMSE={:.3e}",
            max_abs, tol_abs, max_abs_ref, nrmse
        );
        return false;
    }
    eprintln!(
        "  {label}: PASS  max_abs={:.3e}  max_|y_ref|={:.3e}  NRMSE={:.3e}",
        max_abs, max_abs_ref, nrmse
    );
    true
}

fn make_x(n: usize, seed: i64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as i64).wrapping_mul(seed.wrapping_add(0x91c2_a73d)).wrapping_add(seed) & 0xFFFFFF) as f32 * 1e-7 - 0.5)
        .collect()
}

fn synth(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let row_off = row * row_bytes;
        let rs_f32 = 0.02f32 + ((next() & 0xFF) as f32) * 1e-4;
        let rs_f16 = f32_to_f16_bits(rs_f32);
        out[row_off..row_off + 2].copy_from_slice(&rs_f16.to_le_bytes());
        let bc = blocks_per_row as u16;
        out[row_off + 4..row_off + 6].copy_from_slice(&bc.to_le_bytes());
        for b in 0..blocks_per_row {
            let bp = row_off + 16 + b * 17;
            let e = 120 + (next() & 0x7) as u8;
            out[bp] = e;
            for i in 0..16 {
                out[bp + 1 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0 { return sign; }
    if exp >= 143 { return sign | 0x7C00; }
    if exp <= 112 { return sign; }
    let new_exp = (exp - 127 + 15) as u16;
    let new_mant = (mant >> 13) as u16;
    sign | (new_exp << 10) | new_mant
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
