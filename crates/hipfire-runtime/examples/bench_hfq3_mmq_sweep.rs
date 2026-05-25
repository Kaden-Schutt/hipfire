//! Microbench: HFQ3 MMQ vs dot2 residual GEMM across batch sizes.
//!
//! Probes the crossover point where MMQ (mmq_x=32) starts winning over
//! dot2 on RDNA2. Sets the batch-size gate for the shipping dispatcher.
//!
//! Synthetic weights/X — measures kernel-only time. Sync barrier via
//! a small download_f32 between iters.
//!
//! Run: cargo run --release -p hipfire-runtime --example bench_hfq3_mmq_sweep
//!      (set HIPFIRE_HFQ3_MMQ via the dispatcher itself — bench bypasses
//!       routing by calling _dot2 and _mmq variants directly.)

use rdna_compute::{DType, GpuTensor};
use std::time::Instant;

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

fn synth_hfq3_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let groups_per_row = k / 256;
    let mut bytes = vec![0u8; m * groups_per_row * 104];
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = (row * groups_per_row + g) * 104;
            let scale = fract_sin(seed as f32 + (row * 7 + g * 11) as f32 * 0.131);
            let zero = fract_sin(seed as f32 + (row * 13 + g * 17) as f32 * 0.211);
            bytes[off..off + 4].copy_from_slice(&(scale * 0.1).to_le_bytes());
            bytes[off + 4..off + 8].copy_from_slice(&(zero * 0.1).to_le_bytes());
            for i in 0..96 {
                let v = (((seed.wrapping_mul(2654435761))
                    .wrapping_add((row as u32) * 257 + (g as u32) * 19 + i as u32))
                    & 0xFF) as u8;
                bytes[off + 8 + i] = v;
            }
        }
    }
    bytes
}

fn synth_x(n: usize, k: usize, seed: u32) -> Vec<f32> {
    (0..n * k)
        .map(|i| fract_sin(seed as f32 + i as f32 * 0.317))
        .collect()
}

fn run_kernel(
    gpu: &mut rdna_compute::Gpu,
    d_w: &GpuTensor,
    d_x: &GpuTensor,
    d_y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    method: &str,
) {
    match method {
        "scalar" => gpu.gemm_hfq3g256_residual(d_w, d_x, d_y, m, k, n).unwrap(),
        "dot2"   => gpu.gemm_hfq3g256_residual_dot2(d_w, d_x, d_y, m, k, n).unwrap(),
        "mmq8"   => gpu.gemm_hfq3g256_residual_mmq_x8(d_w, d_x, d_y, m, k, n).unwrap(),
        "mmq16"  => gpu.gemm_hfq3g256_residual_mmq_x16(d_w, d_x, d_y, m, k, n).unwrap(),
        "mmq32"  => gpu.gemm_hfq3g256_residual_mmq_x32(d_w, d_x, d_y, m, k, n).unwrap(),
        "mmq32_y64" => gpu.gemm_hfq3g256_residual_mmq_x32_y64(d_w, d_x, d_y, m, k, n).unwrap(),
        "mmq32_y32" => gpu.gemm_hfq3g256_residual_mmq_x32_y32(d_w, d_x, d_y, m, k, n).unwrap(),
        _ => unreachable!(),
    };
}

fn time_one(
    gpu: &mut rdna_compute::Gpu,
    d_w: &GpuTensor,
    d_x: &GpuTensor,
    d_y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    method: &str,
    iters: usize,
) -> f64 {
    for _ in 0..3 { run_kernel(gpu, d_w, d_x, d_y, m, k, n, method); }
    let _ = gpu.download_f32(d_y).unwrap();

    let t0 = Instant::now();
    for _ in 0..iters { run_kernel(gpu, d_w, d_x, d_y, m, k, n, method); }
    let _ = gpu.download_f32(d_y).unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("=== bench_hfq3_mmq_sweep on {} ===", gpu.arch);

    // Qwen3.5 9B FA preamble shape: K=2048, m up to 4096 for q; we use
    // a representative residual call shape (m=4096, k=2048).
    let m = 4096usize;
    let k = 2048usize;
    let weight_bytes = synth_hfq3_bytes(m, k, 42);
    let d_w = gpu.upload_raw(&weight_bytes, &[weight_bytes.len()]).unwrap();

    // Batch sizes spanning short to long prefill.
    let batches: &[usize] = &[1, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 240, 512, 1024];

    println!("# m={m} k={k}  (times in microseconds)");
    println!("{:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>11}  {:>11}  {:>10}  {:>10}  {:>10}",
             "N", "scalar", "dot2", "mmq_x16", "mmq_x32", "mmq_x32_y64", "mmq_x32_y32",
             "best", "y64/y128", "y32/y128");
    println!("{}", "-".repeat(125));

    for &n in batches {
        let x = synth_x(n, k, 17);
        let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
        let d_y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

        let iters = if n <= 16 { 100 } else if n <= 64 { 50 } else { 20 };

        let t_scal  = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "scalar", iters);
        let t_dot2  = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "dot2",   iters);
        let t_mmq16 = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "mmq16",  iters);
        let t_mmq32 = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "mmq32",  iters);
        let t_mmq32_y64 = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "mmq32_y64", iters);
        let t_mmq32_y32 = time_one(&mut gpu, &d_w, &d_x, &d_y, m, k, n, "mmq32_y32", iters);

        let methods = [
            ("scalar", t_scal),
            ("dot2",   t_dot2),
            ("mmq16",  t_mmq16),
            ("mmq32",  t_mmq32),
            ("mmq32_y64", t_mmq32_y64),
            ("mmq32_y32", t_mmq32_y32),
        ];
        let (best_name, _best_t) = methods.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied().unwrap();
        let r64 = t_mmq32_y64 / t_mmq32;
        let r32 = t_mmq32_y32 / t_mmq32;

        println!("{:>6}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>11.1}  {:>11.1}  {:>10}  {:>9.3}x  {:>9.3}x",
                 n,
                 t_scal * 1e6,
                 t_dot2 * 1e6,
                 t_mmq16 * 1e6,
                 t_mmq32 * 1e6,
                 t_mmq32_y64 * 1e6,
                 t_mmq32_y32 * 1e6,
                 best_name,
                 r64,
                 r32);

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y).unwrap();
    }
    gpu.free_tensor(d_w).unwrap();

    // ─── Gate_up sweep: y128 vs y64 across batch sizes ───────────────────
    // The daemon-level test at N=240 showed gate_up y64 regresses (-4%) but
    // the residual sweep showed y64 wins more at moderate N (128-512) than
    // very large N (1024 tied). Need to find if there's any gate_up batch
    // size where y64 wins.
    eprintln!("\n=== gate_up sweep: y128 vs y64 ===");
    let weight_bytes_gu = synth_hfq3_bytes(m, k, 43);
    let d_w_gate = gpu.upload_raw(&weight_bytes_gu, &[weight_bytes_gu.len()]).unwrap();
    let weight_bytes_gu2 = synth_hfq3_bytes(m, k, 44);
    let d_w_up = gpu.upload_raw(&weight_bytes_gu2, &[weight_bytes_gu2.len()]).unwrap();

    println!("# gate_up: m={m}+{m}, k={k}");
    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
             "N", "y128_us", "y96_us", "y64_us", "y96/y128", "y64/y128");
    println!("{}", "-".repeat(70));

    for &n in &[64usize, 96, 128, 192, 240, 384, 512, 768, 1024] {
        let x = synth_x(n, k, 17);
        let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
        let d_yg = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_yu = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

        let iters = if n <= 64 { 50 } else if n <= 256 { 20 } else { 10 };

        // Warmup
        for _ in 0..3 {
            gpu.gemm_gate_up_hfq3g256_mmq_x32(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
            gpu.gemm_gate_up_hfq3g256_mmq_x32_y96(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
            gpu.gemm_gate_up_hfq3g256_mmq_x32_y64(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
        }
        let _ = gpu.download_f32(&d_yg).unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_gate_up_hfq3g256_mmq_x32(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
        }
        let _ = gpu.download_f32(&d_yg).unwrap();
        let t_y128 = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_gate_up_hfq3g256_mmq_x32_y96(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
        }
        let _ = gpu.download_f32(&d_yg).unwrap();
        let t_y96 = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_gate_up_hfq3g256_mmq_x32_y64(
                &d_w_gate, &d_w_up, &d_x, &d_yg, &d_yu, m, m, k, n,
            ).unwrap();
        }
        let _ = gpu.download_f32(&d_yg).unwrap();
        let t_y64 = t0.elapsed().as_secs_f64() / iters as f64;

        let r96 = t_y96 / t_y128;
        let r64 = t_y64 / t_y128;
        println!("{:>6}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.3}  {:>10.3}",
                 n, t_y128 * 1e6, t_y96 * 1e6, t_y64 * 1e6, r96, r64);

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_yg).unwrap();
        gpu.free_tensor(d_yu).unwrap();
    }
    gpu.free_tensor(d_w_gate).unwrap();
    gpu.free_tensor(d_w_up).unwrap();
}
