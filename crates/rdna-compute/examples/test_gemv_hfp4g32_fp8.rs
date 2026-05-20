//! Correctness + perf test for the gfx12 FP8-dot4 HFP4G32 decode GEMV.
//!
//! Compares `gemv_hfp4g32_fp8_gfx12` against the fallback `gemv_hfp4g32`
//! across production Qwen 9B decode shapes. Reports speedup, NRMSE, and
//! % of theoretical peak BW.
//!
//! Skips silently on non-gfx12 archs.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const PEAK_GBPS: f64 = 800.0;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        eprintln!("=== SKIP === FP8 dot4 GEMV is gfx12-only (arch={arch})");
        return;
    }
    eprintln!("=== gemv_hfp4g32_fp8_gfx12 vs fallback gemv_hfp4g32 ===");
    eprintln!("  arch={arch}  peak_bw_gbps={PEAK_GBPS}");

    // 9B Qwen3.5 decode-path GEMV shapes:
    let shapes: Vec<(usize, usize, &str)> = vec![
        (2048,  2048,  "qkv-q     M=2048 K=2048"),
        (512,   2048,  "qkv-kv    M=512  K=2048"),
        (11008, 2048,  "gate_up   M=11008 K=2048"),
        (2048,  11008, "w_down    M=2048  K=11008"),
        (4096,  2048,  "med       M=4096 K=2048"),
        (1024,  2048,  "small     M=1024 K=2048"),
    ];

    let trials = 200;
    let warmup = 20;

    let mut all_pass = true;
    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        let row_bytes = 16 + (k / 32) * 17;
        let total_w_bytes = m * row_bytes;

        let w = gpu.upload_raw(&synth(m, k, 0xAA00 | (m as u64) ^ (k as u64)), &[total_w_bytes]).unwrap();
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_ref = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let y_fp8 = gpu.alloc_tensor(&[m], DType::F32).unwrap();

        let x_host = make_x(k, 0x1111);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

        // Warmup both kernels
        for _ in 0..warmup {
            gpu.gemv_hfp4g32(&w, &x, &y_ref, m, k).unwrap();
            gpu.gemv_hfp4g32_fp8_gfx12(&w, &x, &y_fp8, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Time fallback
        let t = Instant::now();
        for _ in 0..trials {
            gpu.gemv_hfp4g32(&w, &x, &y_ref, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let ref_us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;

        // Time FP8 dot4
        let t = Instant::now();
        for _ in 0..trials {
            gpu.gemv_hfp4g32_fp8_gfx12(&w, &x, &y_fp8, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let fp8_us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;

        let ref_bw = (total_w_bytes as f64) / (ref_us * 1e-6) / 1e9;
        let fp8_bw = (total_w_bytes as f64) / (fp8_us * 1e-6) / 1e9;

        // Correctness check
        let ok = cmp_tol(&mut gpu, &y_ref, &y_fp8, m, label);
        all_pass &= ok;

        let speedup = ref_us / fp8_us;
        eprintln!(
            "  {label:30}  ref: {ref_us:6.2}µs ({ref_bw:5.1} GB/s = {:4.1}%)  fp8: {fp8_us:6.2}µs ({fp8_bw:5.1} GB/s = {:4.1}%)  speedup: {speedup:4.2}×",
            ref_bw / PEAK_GBPS * 100.0,
            fp8_bw / PEAK_GBPS * 100.0,
        );
    }

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

fn cmp_tol(gpu: &mut Gpu, y_ref: &GpuTensor, y_fp8: &GpuTensor, m: usize, label: &str) -> bool {
    let r = gpu.download_f32(y_ref).unwrap();
    let k = gpu.download_f32(y_fp8).unwrap();

    let mut max_abs: f64 = 0.0;
    let mut max_abs_ref: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut sum_sq_ref: f64 = 0.0;
    for i in 0..m {
        let r_v = r[i] as f64;
        let k_v = k[i] as f64;
        let abs = (r_v - k_v).abs();
        if abs > max_abs { max_abs = abs; }
        if r_v.abs() > max_abs_ref { max_abs_ref = r_v.abs(); }
        sum_sq_err += (r_v - k_v) * (r_v - k_v);
        sum_sq_ref += r_v * r_v;
    }
    let nrmse = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
    let tol_abs = 0.05 * max_abs_ref.max(1e-3);
    let bad: usize = (0..m).filter(|&i| ((r[i] as f64) - (k[i] as f64)).abs() > tol_abs).count();
    if bad > 0 {
        eprintln!(
            "  {label}: FAIL  {bad}/{m}  max_abs={:.3e}  tol={:.3e}  max_|y|={:.3e}  NRMSE={:.3e}",
            max_abs, tol_abs, max_abs_ref, nrmse
        );
        return false;
    }
    eprintln!("  {label}: OK  NRMSE={:.3e}", nrmse);
    true
}

fn make_x(n: usize, seed: i64) -> Vec<f32> {
    (0..n).map(|i| ((i as i64).wrapping_mul(seed.wrapping_add(0x91c2_a73d)).wrapping_add(seed) & 0xFFFFFF) as f32 * 1e-7 - 0.5).collect()
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
