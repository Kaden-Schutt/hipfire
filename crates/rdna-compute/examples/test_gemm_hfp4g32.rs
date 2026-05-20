//! Unified correctness + perf test for the three batched HFP4-G32
//! GEMM kernels:
//!   * gemm_qkv_hfp4g32           (QKV preamble, 3-way fused)
//!   * gemm_gate_up_hfp4g32       (FFN preamble, 2-way fused)
//!   * gemm_hfp4g32_residual      (wo / w_down with += semantics)
//!
//! Reference for each: gemv_hfp4g32 × N (one call per batch row).
//! Tolerance: 1e-3 relative against max|y_ref|, plus 1e-4 abs floor.
//! WMMA accumulates in FP16, scalar GEMV in FP32 — exact byte match
//! is not expected.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn main() {
    // n_list runs in DESCENDING order: each test reuses one x_gemm tensor
    // across all N values, but `ensure_fp16_x` caches by src_ptr — so smaller
    // N after larger N reads already-converted data, while smaller N before
    // larger N would skip reconversion of the larger range. Doing max N first
    // guarantees full coverage of the fp16 scratch on the first call.
    let n_list: Vec<usize> = vec![64, 16, 4, 1];
    let k: usize = 1024;

    let mut gpu = Gpu::init().expect("gpu init");

    let mut all_pass = true;
    all_pass &= test_qkv(&mut gpu, &n_list, k);
    all_pass &= test_gate_up(&mut gpu, &n_list, k);
    all_pass &= test_residual(&mut gpu, &n_list, k);

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

fn row_bytes_for(k: usize) -> usize {
    16 + (k / 32) * 17
}

fn test_qkv(gpu: &mut Gpu, n_list: &[usize], k: usize) -> bool {
    let q_m: usize = 2048;
    let k_m: usize = 512;
    let v_m: usize = 512;
    let row_bytes = row_bytes_for(k);

    eprintln!("=== gemm_qkv_hfp4g32 ===");
    eprintln!("  q_m={q_m} k_m={k_m} v_m={v_m} K={k}");

    let w_q = gpu.upload_raw(&synth(q_m, k, 0xAA), &[q_m * row_bytes]).unwrap();
    let w_k = gpu.upload_raw(&synth(k_m, k, 0xBB), &[k_m * row_bytes]).unwrap();
    let w_v = gpu.upload_raw(&synth(v_m, k, 0xCC), &[v_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = make_x(max_n * k, 0x1111);

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_q_1 = gpu.alloc_tensor(&[q_m], DType::F32).unwrap();
    let y_k_1 = gpu.alloc_tensor(&[k_m], DType::F32).unwrap();
    let y_v_1 = gpu.alloc_tensor(&[v_m], DType::F32).unwrap();
    let y_q_col = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_col = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_col = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_q_gemm = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_gemm = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_gemm = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    let mut all_pass = true;
    for &n in n_list {
        let mut gemv_us = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.gemv_hfp4g32(&w_q, &x_gemv, &y_q_1, q_m, k).unwrap();
            gpu.gemv_hfp4g32(&w_k, &x_gemv, &y_k_1, k_m, k).unwrap();
            gpu.gemv_hfp4g32(&w_v, &x_gemv, &y_v_1, v_m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;
            gpu.hip.memcpy_dtod_at(&y_q_col.buf, i * q_m * 4, &y_q_1.buf, 0, q_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_k_col.buf, i * k_m * 4, &y_k_1.buf, 0, k_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_v_col.buf, i * v_m * 4, &y_v_1.buf, 0, v_m * 4).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkv_hfp4g32(
            &w_q, &w_k, &w_v, &x_gemm,
            &y_q_gemm, &y_k_gemm, &y_v_gemm,
            q_m, k_m, v_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok_q = cmp_tol(gpu, &y_q_col, &y_q_gemm, n, q_m, "q");
        let ok_k = cmp_tol(gpu, &y_k_col, &y_k_gemm, n, k_m, "k");
        let ok_v = cmp_tol(gpu, &y_v_col, &y_v_gemm, n, v_m, "v");
        let ok = ok_q && ok_k && ok_v;
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   gemm×1: {:8.1} µs   speedup: {:5.2}x   [{}]",
            gemv_us, gemm_us, gemv_us / gemm_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }
    all_pass
}

fn test_gate_up(gpu: &mut Gpu, n_list: &[usize], k: usize) -> bool {
    let gate_m: usize = 4096;
    let up_m: usize = 4096;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== gemm_gate_up_hfp4g32 ===");
    eprintln!("  gate_m={gate_m} up_m={up_m} K={k}");

    let w_g = gpu.upload_raw(&synth(gate_m, k, 0xDD), &[gate_m * row_bytes]).unwrap();
    let w_u = gpu.upload_raw(&synth(up_m, k, 0xEE), &[up_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = make_x(max_n * k, 0x2222);

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_g_1 = gpu.alloc_tensor(&[gate_m], DType::F32).unwrap();
    let y_u_1 = gpu.alloc_tensor(&[up_m], DType::F32).unwrap();
    let y_g_col = gpu.alloc_tensor(&[max_n * gate_m], DType::F32).unwrap();
    let y_u_col = gpu.alloc_tensor(&[max_n * up_m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_g_gemm = gpu.alloc_tensor(&[max_n * gate_m], DType::F32).unwrap();
    let y_u_gemm = gpu.alloc_tensor(&[max_n * up_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    let mut all_pass = true;
    for &n in n_list {
        let mut gemv_us = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.gemv_hfp4g32(&w_g, &x_gemv, &y_g_1, gate_m, k).unwrap();
            gpu.gemv_hfp4g32(&w_u, &x_gemv, &y_u_1, up_m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;
            gpu.hip.memcpy_dtod_at(&y_g_col.buf, i * gate_m * 4, &y_g_1.buf, 0, gate_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_u_col.buf, i * up_m * 4, &y_u_1.buf, 0, up_m * 4).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_gate_up_hfp4g32(
            &w_g, &w_u, &x_gemm,
            &y_g_gemm, &y_u_gemm,
            gate_m, up_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok_g = cmp_tol(gpu, &y_g_col, &y_g_gemm, n, gate_m, "gate");
        let ok_u = cmp_tol(gpu, &y_u_col, &y_u_gemm, n, up_m, "up");
        let ok = ok_g && ok_u;
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   gemm×1: {:8.1} µs   speedup: {:5.2}x   [{}]",
            gemv_us, gemm_us, gemv_us / gemm_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }
    all_pass
}

fn test_residual(gpu: &mut Gpu, n_list: &[usize], k: usize) -> bool {
    let m: usize = 2048;
    let row_bytes = row_bytes_for(k);

    eprintln!("\n=== gemm_hfp4g32_residual (+= semantics) ===");
    eprintln!("  M={m} K={k}");

    let w = gpu.upload_raw(&synth(m, k, 0xFF), &[m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = make_x(max_n * k, 0x3333);
    // Residual seed: small constant so += semantics is testable.
    let res_seed: Vec<f32> = (0..max_n * m)
        .map(|i| ((i as i32) % 7 - 3) as f32 * 1e-3)
        .collect();

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_1 = gpu.alloc_tensor(&[m], DType::F32).unwrap();
    let y_col = gpu.alloc_tensor(&[max_n * m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_gemm = gpu.alloc_tensor(&[max_n * m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    let mut all_pass = true;
    for &n in n_list {
        let mut gemv_us = 0.0;
        // Build the host-side reference: ref[b][r] = seed[b][r] + gemv(w, x[b])[r].
        let mut ref_host: Vec<f32> = vec![0.0; n * m];
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.gemv_hfp4g32(&w, &x_gemv, &y_1, m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;
            let y_1_host = gpu.download_f32(&y_1).unwrap();
            for r in 0..m {
                ref_host[i * m + r] = res_seed[i * m + r] + y_1_host[r];
            }
        }
        gpu.hip.memcpy_htod(&y_col.buf, bytes_of(&ref_host)).unwrap();
        // Seed y_gemm with res_seed; the residual GEMM accumulates into it.
        gpu.hip.memcpy_htod(&y_gemm.buf, bytes_of(&res_seed[..n * m])).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_hfp4g32_residual(&w, &x_gemm, &y_gemm, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok = cmp_tol(gpu, &y_col, &y_gemm, n, m, "res");
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   gemm×1: {:8.1} µs   speedup: {:5.2}x   [{}]",
            gemv_us, gemm_us, gemv_us / gemm_us,
            if ok { "PASS" } else { "FAIL" }
        );
        all_pass &= ok;
    }
    all_pass
}

fn cmp_tol(gpu: &mut Gpu, y_ref: &GpuTensor, y_kernel: &GpuTensor, n: usize, m: usize, label: &str) -> bool {
    let r = gpu.download_f32(y_ref).unwrap();
    let k = gpu.download_f32(y_kernel).unwrap();

    let mut max_abs: f64 = 0.0;
    let mut max_abs_ref: f64 = 0.0;
    let mut bad = 0usize;

    for b in 0..n {
        for row in 0..m {
            let r_v = r[b * m + row] as f64;
            let k_v = k[b * m + row] as f64;
            let abs = (r_v - k_v).abs();
            if abs > max_abs { max_abs = abs; }
            if r_v.abs() > max_abs_ref { max_abs_ref = r_v.abs(); }
        }
    }
    let tol_abs = 1e-3 * max_abs_ref.max(1e-4);
    for b in 0..n {
        for row in 0..m {
            let r_v = r[b * m + row] as f64;
            let k_v = k[b * m + row] as f64;
            if (r_v - k_v).abs() > tol_abs { bad += 1; }
        }
    }

    if bad > 0 {
        eprintln!(
            "  {label}: FAIL  {bad} elements outside tol  max_abs={:.3e}  tol={:.3e}  max_|y_ref|={:.3e}",
            max_abs, tol_abs, max_abs_ref
        );
        return false;
    }
    eprintln!(
        "  {label}: PASS  max_abs={:.3e}  max_|y_ref|={:.3e}",
        max_abs, max_abs_ref
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
