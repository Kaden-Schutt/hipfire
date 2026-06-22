//! GPU `gemm_f32_train` correctness vs CPU reference (hipfire-train Phase 0, M0).
//!
//! `gemm_f32_train` is the single matmul primitive the training path's forward
//! AND backward rely on. A bug here silently poisons every gradient, so this is
//! the M0 gate per docs/plans/2026-06-17-hipfire-train-phase0.md. Costs seconds;
//! a wrong gradient would otherwise show up only as a non-converging loss curve
//! hours later.
//!
//! Validates all three linear-layer products through one kernel via transpose
//! flags: forward `Y=X·Wᵀ`, `dX=dY·W`, `dW=dYᵀ·X`, plus the `_accum` variant.
//!
//! Run:
//!   source ./scripts/rocm-env.sh
//!   hipfire gpu-lock acquire "gemm-train-test"
//!   cargo run -p rdna-compute --release --example test_gemm_f32_train_gpu_vs_cpu
//!   hipfire gpu-lock release

use rdna_compute::{DType, Gpu};

/// Row-major reference: C[M,N] = op(A)·op(B), matching the kernel's index math.
///   op(A)[m,k] = trans_a ? A[k*lda+m] : A[m*lda+k]
///   op(B)[k,n] = trans_b ? B[n*ldb+k] : B[k*ldb+n]
#[allow(clippy::too_many_arguments)]
fn cpu_gemm(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let av = if trans_a {
                    a[ki * lda + mi]
                } else {
                    a[mi * lda + ki]
                };
                let bv = if trans_b {
                    b[ni * ldb + ki]
                } else {
                    b[ki * ldb + ni]
                };
                acc += av * bv;
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    // Non-square shapes catch index/transpose bugs that square shapes hide.
    // Model the linear: X[M=tokens, K=in], W[Nout, K].
    let (tokens, kin, nout) = (5usize, 7usize, 3usize);

    // Deterministic pseudo-random inputs (no Math.random in this env anyway).
    let x: Vec<f32> = (0..tokens * kin)
        .map(|i| ((i * 37 % 23) as f32) * 0.05 - 0.5)
        .collect();
    let w: Vec<f32> = (0..nout * kin)
        .map(|i| ((i * 19 % 17) as f32) * 0.07 - 0.4)
        .collect();
    let dy: Vec<f32> = (0..tokens * nout)
        .map(|i| ((i * 13 % 11) as f32) * 0.1 - 0.3)
        .collect();

    let dx = gpu.upload_f32(&x, &[tokens * kin])?;
    let dw = gpu.upload_f32(&w, &[nout * kin])?;
    let ddy = gpu.upload_f32(&dy, &[tokens * nout])?;

    let tol = 1e-4f32;
    let mut failures = 0;

    // ── 1. forward: Y[tokens,nout] = X·Wᵀ ────────────────────────────────────
    {
        let y = gpu.zeros(&[tokens * nout], DType::F32)?;
        gpu.gemm_f32_train(&dx, &dw, &y, tokens, nout, kin, kin, kin, false, true)?;
        let got = gpu.download_f32(&y)?;
        let want = cpu_gemm(&x, &w, tokens, nout, kin, kin, kin, false, true);
        let err = max_abs_err(&got, &want);
        let ok = err < tol;
        failures += !ok as i32;
        println!(
            "forward  Y=X·Wᵀ      max_abs_err={err:.2e} {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // ── 2. backward dX[tokens,kin] = dY·W ────────────────────────────────────
    {
        let gx = gpu.zeros(&[tokens * kin], DType::F32)?;
        gpu.gemm_f32_train(&ddy, &dw, &gx, tokens, kin, nout, nout, kin, false, false)?;
        let got = gpu.download_f32(&gx)?;
        let want = cpu_gemm(&dy, &w, tokens, kin, nout, nout, kin, false, false);
        let err = max_abs_err(&got, &want);
        let ok = err < tol;
        failures += !ok as i32;
        println!(
            "backward dX=dY·W     max_abs_err={err:.2e} {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // ── 3. backward dW[nout,kin] = dYᵀ·X ─────────────────────────────────────
    {
        let gw = gpu.zeros(&[nout * kin], DType::F32)?;
        gpu.gemm_f32_train(&ddy, &dx, &gw, nout, kin, tokens, nout, kin, true, false)?;
        let got = gpu.download_f32(&gw)?;
        let want = cpu_gemm(&dy, &x, nout, kin, tokens, nout, kin, true, false);
        let err = max_abs_err(&got, &want);
        let ok = err < tol;
        failures += !ok as i32;
        println!(
            "backward dW=dYᵀ·X    max_abs_err={err:.2e} {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // ── 4. accum variant: C = beta*C + X·Wᵀ ──────────────────────────────────
    {
        let init: Vec<f32> = (0..tokens * nout).map(|i| (i as f32) * 0.01).collect();
        let c = gpu.upload_f32(&init, &[tokens * nout])?;
        let beta = 0.5f32;
        gpu.gemm_f32_train_accum(&dx, &dw, &c, tokens, nout, kin, kin, kin, false, true, beta)?;
        let got = gpu.download_f32(&c)?;
        let prod = cpu_gemm(&x, &w, tokens, nout, kin, kin, kin, false, true);
        let want: Vec<f32> = init
            .iter()
            .zip(&prod)
            .map(|(c0, p)| beta * c0 + p)
            .collect();
        let err = max_abs_err(&got, &want);
        let ok = err < tol;
        failures += !ok as i32;
        println!(
            "accum    C=βC+X·Wᵀ   max_abs_err={err:.2e} {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    if failures == 0 {
        println!("\nALL PASS — gemm_f32_train matches CPU reference.");
        Ok(())
    } else {
        Err(format!("{failures} gemm_f32_train case(s) FAILED").into())
    }
}
