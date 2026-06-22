//! Standalone correctness + timing for gemm_f32_train at realistic large dims
//! (the gradcheck only covers tiny shapes). Isolates the tiled GEMM from the
//! full capture pipeline.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gemm-bench"
//!   cargo run -p hipfire-train --release --example gemm_f32_train_bench

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    // Sweep all forward-NT shapes a 3B capture actually uses (the capture faulted
    // with HIP 719; the bench's single 512×3072×3072 shape passed).
    let shapes: &[(usize, usize, usize, &str)] = &[
        (512, 3072, 3072, "q/o_proj"),
        (512, 1024, 3072, "k/v_proj"),
        (512, 8192, 3072, "mlp gate/up"),
        (512, 3072, 8192, "mlp down"),
    ];
    for &(sm, sn, sk, name) in shapes {
        let xa: Vec<f32> = (0..sm * sk).map(|_| 0.01f32).collect();
        let wa: Vec<f32> = (0..sn * sk).map(|_| 0.01f32).collect();
        let xt = gpu.upload_f32(&xa, &[sm * sk])?;
        let wt = gpu.upload_f32(&wa, &[sn * sk])?;
        let yt = gpu.zeros(&[sm * sn], DType::F32)?;
        gpu.gemm_f32_train(&xt, &wt, &yt, sm, sn, sk, sk, sk, false, true)?;
        let _ = gpu.download_f32(&yt)?; // sync barrier → surfaces any fault here
        println!("  shape {name:14} M={sm} N={sn} K={sk}  OK");
    }
    println!("all capture forward-NT shapes OK\n");

    // forward shape: Y[M,N] = X[M,K] · W[N,K]ᵀ  (transB=1), 3B-ish dims
    let (m, n, k) = (512usize, 3072usize, 3072usize);
    let mut s: u64 = 0x1234567;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / (1u64 << 23) as f32 - 0.5
    };
    let xa: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let wa: Vec<f32> = (0..n * k).map(|_| rng()).collect();
    let x = gpu.upload_f32(&xa, &[m * k])?;
    let w = gpu.upload_f32(&wa, &[n * k])?;
    let y = gpu.zeros(&[m * n], DType::F32)?;

    println!("launching gemm_f32_train M={m} N={n} K={k} (transB=1)…");
    gpu.gemm_f32_train(&x, &w, &y, m, n, k, k, k, false, true)?;
    let yh = gpu.download_f32(&y)?;
    println!("kernel returned; checking correctness on a few elements…");

    // CPU reference for a handful of output elements
    let refel = |mi: usize, ni: usize| -> f32 {
        let mut acc = 0.0f64;
        for kk in 0..k {
            acc += xa[mi * k + kk] as f64 * wa[ni * k + kk] as f64;
        }
        acc as f32
    };
    let mut maxrel = 0.0f32;
    for &(mi, ni) in &[(0, 0), (1, 5), (511, 3071), (100, 200), (255, 1000)] {
        let got = yh[mi * n + ni];
        let want = refel(mi, ni);
        let rel = (got - want).abs() / want.abs().max(1e-3);
        maxrel = maxrel.max(rel);
        println!("  Y[{mi},{ni}] = {got:.4}  ref {want:.4}  rel {rel:.2e}");
    }
    if maxrel > 1e-2 {
        return Err(format!("CORRECTNESS FAIL: max rel {maxrel:.2e}").into());
    }
    println!("correctness OK (max rel {maxrel:.2e})");

    // timing: 20 iters of the three training matmuls, report GFLOP/s
    let dy = gpu.zeros(&[m * n], DType::F32)?;
    let dx = gpu.zeros(&[m * k], DType::F32)?;
    let dw = gpu.zeros(&[n * k], DType::F32)?;
    // warmup + per-case isolation (download after each → pinpoints a faulting case)
    gpu.gemm_f32_train(&x, &w, &y, m, n, k, k, k, false, true)?; // fwd NT
    let _ = gpu.download_f32(&y)?;
    println!("  fwd NT ok");
    gpu.gemm_f32_train(&dy, &w, &dx, m, k, n, n, k, false, false)?; // dX NN
    let _ = gpu.download_f32(&dx)?;
    println!("  dX NN ok");
    gpu.gemm_f32_train(&dy, &x, &dw, n, k, m, n, k, true, false)?; // dW TN
    let _ = gpu.download_f32(&dw)?;
    println!("  dW TN ok");

    let iters = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        gpu.gemm_f32_train(&x, &w, &y, m, n, k, k, k, false, true)?;
        gpu.gemm_f32_train(&dy, &w, &dx, m, k, n, n, k, false, false)?;
        gpu.gemm_f32_train(&dy, &x, &dw, n, k, m, n, k, true, false)?;
    }
    let _ = gpu.download_f32(&dw)?; // blocks until the stream drains (sync barrier)
    let dt = t0.elapsed().as_secs_f64();
    let flop = 3.0 * 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
    println!(
        "\n3 matmuls × {iters} iters: {:.3}s → {:.1} GFLOP/s",
        dt,
        flop / dt / 1e9
    );
    Ok(())
}
