//! Microbench: gfx12 HFQ4v4 iu4 GEMM vs v1 iu4 GEMM vs FP16 WMMA baseline.
//!
//! Compares throughput on realistic prefill GEMM dimensions for Qwen3.5
//! sizes (4B, 9B, 27B). Three paths:
//!   - FP16 WMMA (baseline / "safe" path)
//!   - v1 iu4 K=32 (HFQ4-G256 + Q4_1 acts; quality-fail per PR #140)
//!   - v4 iu4 K=32 (HFQ4v4 + per-row mu correction; the new path)
//!
//! Run on gfx1201 (hiptrx):
//!   cargo run --release -p engine --example bench_hfq4v4_gemm
//!
//! Env knobs:
//!   ITERS=10  — iters per measurement (default 10)
//!   WARMUP=3  — warmup iters (default 3)
//!   ROTATE=1  — emit MQ4v4 variant (FWHT-32 weights)

use engine::hfq4v4::{convert_hfq4g256_to_hfq4v4, MuStrategy};
use rdna_compute::{DType, Gpu};

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let warmup: usize = std::env::var("WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let rotate = std::env::var("ROTATE").ok().as_deref() == Some("1");

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!("SKIP: requires gfx1200/gfx1201 (RDNA4). Current: {arch}");
        std::process::exit(0);
    }

    eprintln!("=== gfx12 HFQ4v4 GEMM microbench ===");
    eprintln!("iters={iters}, warmup={warmup}, rotate={rotate}");
    eprintln!();

    // Realistic shapes from Qwen3.5 prefill (residual GEMMs: wo and w_down).
    // (M, K, N): M is output dim, K is input dim, N is batch (prefill tokens).
    //   Qwen3.5-4B:  hidden=2560, ffn=10240
    //     wo:     M=2560, K=2560
    //     w_down: M=2560, K=10240
    //   Qwen3.5-9B:  hidden=4096, ffn=14336
    //     wo:     M=4096, K=4096
    //     w_down: M=4096, K=14336
    //   Qwen3.5-27B: hidden=5120, ffn=27648
    //     wo:     M=5120, K=5120
    //     w_down: M=5120, K=27648
    //
    // Use N=128 (typical pp128 prefill batch) and N=512 (long prefill).
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("4B-wo-pp128",    2560,  2560, 128),
        ("4B-down-pp128",  2560, 10240, 128),
        ("9B-wo-pp128",    4096,  4096, 128),
        ("9B-down-pp128",  4096, 14336, 128),
        ("27B-wo-pp128",   5120,  5120, 128),
        ("27B-down-pp128", 5120, 27648, 128),
        ("9B-wo-pp512",    4096,  4096, 512),
        ("9B-down-pp512",  4096, 14336, 512),
        ("27B-wo-pp512",   5120,  5120, 512),
        ("27B-down-pp512", 5120, 27648, 512),
    ];

    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "label",
        "M", "K", "N",
        "fp16 ms", "v1 ms", "v4 ms",
        "v4 GFLOPS",
        "v4/v1",
        "v4/fp16",
    );

    for (label, m, k, n) in &shapes {
        // Skip oversized shapes that don't fit comfortably.
        let weight_mb_v1 = m * k / 256 * 136 / 1_000_000;
        if weight_mb_v1 > 5_000 {
            eprintln!("SKIP {label}: weight too large ({weight_mb_v1} MB v1)");
            continue;
        }
        let groups_per_row = k / 256;
        let row_bytes = groups_per_row * 136;
        let weight_bytes_v1: Vec<u8> =
            synth_hfq4g256_weights(*m, groups_per_row, 0xC0DE_FACE);
        let a_v1 = gpu
            .upload_raw(&weight_bytes_v1, &[m * row_bytes])
            .expect("upload v1");

        let (w_v4, mu_v4) = convert_hfq4g256_to_hfq4v4(
            &weight_bytes_v1, *m, *k, rotate, &MuStrategy::WeightMean,
        );
        let a_v4 = gpu.upload_raw(&w_v4, &[w_v4.len()]).expect("upload v4");
        let mu_t = gpu.upload_raw(&mu_v4, &[mu_v4.len()]).expect("upload mu");

        let x_host: Vec<f32> = (0..n * k)
            .map(|i| {
                let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
                (v * 1e-9) % 2.0 - 1.0
            })
            .collect();
        let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
        gpu.hip
            .memcpy_htod(&x_gpu.buf, bytes_of(&x_host))
            .expect("htod x");

        // Pre-allocate y tensors (we reuse across iters).
        let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
        let y_v1 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_v1");
        let y_v4 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_v4");

        // Build a y_init buffer once.
        let y_init: Vec<f32> = vec![0.0; n * m];
        let _ = gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init));
        let _ = gpu.hip.memcpy_htod(&y_v1.buf, bytes_of(&y_init));
        let _ = gpu.hip.memcpy_htod(&y_v4.buf, bytes_of(&y_init));

        // Warmup all three paths.
        for _ in 0..warmup {
            let _ = gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_v1, &x_gpu, &y_fp16, *m, *k, *n);
            if let Ok(xq) = gpu.ensure_q4_1_x(&x_gpu, *n, *k) {
                let _ = gpu.gemm_hfq4g256_residual_iu4_gfx12(&a_v1, xq, &y_v1, *m, *k, *n, true);
                let _ = gpu.gemm_hfq4v4_residual_iu4_gfx12(
                    &a_v4, &mu_t, xq, &y_v4, *m, *k, *n, true,
                );
            }
        }
        gpu.hip.device_synchronize().expect("sync");

        // Time fp16.
        let fp16_ms = time_n(&mut gpu, iters, |g| {
            let _ = g.gemm_hfq4g256_residual_wmma_gfx12(&a_v1, &x_gpu, &y_fp16, *m, *k, *n);
        });

        // Time v1.
        let v1_ms = time_n(&mut gpu, iters, |g| {
            // Re-quantize per iter (matches real dispatch behaviour).
            let xq = g
                .ensure_q4_1_x(&x_gpu, *n, *k)
                .expect("q4_1 ensure");
            let _ = g.gemm_hfq4g256_residual_iu4_gfx12(&a_v1, xq, &y_v1, *m, *k, *n, true);
        });

        // Time v4.
        let v4_ms = time_n(&mut gpu, iters, |g| {
            let xq = g
                .ensure_q4_1_x(&x_gpu, *n, *k)
                .expect("q4_1 ensure");
            let _ = g.gemm_hfq4v4_residual_iu4_gfx12(&a_v4, &mu_t, xq, &y_v4, *m, *k, *n, true);
        });

        let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
        let v4_gflops = flops / (v4_ms as f64 * 1e6);

        println!(
            "{:<20} {:>8} {:>8} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.1} {:>8.3} {:>8.3}",
            label,
            m, k, n,
            fp16_ms, v1_ms, v4_ms,
            v4_gflops,
            v1_ms / v4_ms,
            fp16_ms / v4_ms,
        );

        let _ = gpu.free_tensor(a_v1);
        let _ = gpu.free_tensor(a_v4);
        let _ = gpu.free_tensor(mu_t);
        let _ = gpu.free_tensor(x_gpu);
        let _ = gpu.free_tensor(y_fp16);
        let _ = gpu.free_tensor(y_v1);
        let _ = gpu.free_tensor(y_v4);
    }
}

fn time_n<F>(gpu: &mut Gpu, iters: usize, mut work: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        work(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
    elapsed_ms / (iters as f32)
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale_bits = 0x3a000000u32 | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0x80) << 24) | 0x39000000u32 | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 {
                scale
            } else {
                1e-3
            };
            let zp_ok = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
