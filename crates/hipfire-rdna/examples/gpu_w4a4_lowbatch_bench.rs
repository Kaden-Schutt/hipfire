//! GPU W4A4 (`gemm_iu4_i32_wmma_lds`, the tuned LDS kernel — a W4A8 proxy: W4A4≈W4A8
//! within ~1.05× on gfx1151) throughput vs BATCH, to resolve the concurrent NPU‖GPU split
//! go/no-go. The split's aggregate win = 1 + NPU/GPU; the NPU is flat ~1.9 TOPS at every
//! batch, so the decider is the GPU's rate at LOW batch (small B can't fill the WMMA
//! pipeline). Logical GEMM `[M,K]·[B,K]ᵀ → [B,M]`, B = tokens; shape matches the NPU bench
//! (M = N = 4096, K = 512). See docs/npu/concurrent-prefill-split-design.md.
//!
//! Run (needs the GPU):
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "gpu-w4a4-lowbatch"
//!   cargo run -p hipfire-rdna --release --example gpu_w4a4_lowbatch_bench [M K]
//!   hipfire lock release

use hipfire_rdna::Gpu;
use std::time::Instant;

fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u8
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return Ok(());
    }
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (m, k) = (p(1, 4096), p(2, 512)); // M = N (out dim), K = contraction
    assert_eq!(k % 16, 0);

    println!(
        "arch: {}  |  gemm_iu4_i32_wmma_lds, [M={m},K={k}]·[B,K]ᵀ→[B,M], iters=50",
        gpu.arch
    );
    println!("  NPU reference: flat ~1.9 TOPS at every B (weight-bandwidth-bound)");
    println!(
        "  {:>6} | {:>10} {:>9} | {:>18}",
        "B(tok)", "ms", "TOPS", "split win=1+1.9/GPU"
    );
    println!("  {}", "-".repeat(52));

    let (iters, warmup) = (50u32, 10u32);
    for &b in &[64usize, 128, 256, 512, 768, 2048, 4096, 8192] {
        let w4 = gpu.upload_raw(&rand_bytes(m * (k / 2), 1), &[m, k / 2])?;
        let x4 = gpu.upload_raw(&rand_bytes(b * (k / 2), 2), &[b, k / 2])?;
        let y4 = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m])?;
        for _ in 0..warmup {
            gpu.gemm_iu4_i32_wmma_lds(&w4, &x4, &y4, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_iu4_i32_wmma_lds(&w4, &x4, &y4, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * m as f64 * k as f64 * b as f64 / (ms * 1e-3) / 1e12;
        let win = 100.0 * 1.9 / (tops + 1.9);
        println!("  {b:>6} | {ms:>10.4} {tops:>9.2} | {win:>16.1}%",);
        for t in [w4, x4, y4] {
            gpu.free_tensor(t)?;
        }
    }
    println!(
        "\n  Concurrent NPU‖GPU split adds the NPU's flat ~1.9 TOPS on top of the GPU row.\n  \
              Build the split only where 'win' is a meaningful double-digit % (low B)."
    );
    Ok(())
}
