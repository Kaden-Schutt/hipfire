#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! Finite-difference gradient check for the `linear` op (Phase 0, M1).
//!
//! Validates the analytic backward (`gemm_f32_train`-based) against central
//! differences, independent of any reference framework. Scalar loss
//! `L = Σ_ij Y_ij · G_ij` for a fixed random G, so `dL/dY = G` exactly — then
//! analytic `dX = G·W`, `dW = Gᵀ·X` must match `(L(θ+ε) − L(θ−ε)) / 2ε`.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "gradcheck-linear"
//!   cargo run -p hipfire-train --release --example gradcheck_linear
//!   hipfire gpu-lock release

use hipfire_train::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

const M: usize = 4; // tokens
const K: usize = 6; // in features
const N: usize = 5; // out features

/// L = Σ Y∘G, with Y = X·Wᵀ computed on GPU; returns the scalar.
fn loss(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, g: &[f32]) -> HipResult<f32> {
    let y = gpu.zeros(&[M * N], DType::F32)?;
    linear_forward(gpu, x, w, &y, M, K, N)?;
    let yv = gpu.download_f32(&y)?;
    Ok(yv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    // Deterministic inputs.
    let x_host: Vec<f32> = (0..M * K)
        .map(|i| ((i * 31 % 17) as f32) * 0.07 - 0.4)
        .collect();
    let w_host: Vec<f32> = (0..N * K)
        .map(|i| ((i * 23 % 13) as f32) * 0.05 - 0.25)
        .collect();
    let g_host: Vec<f32> = (0..M * N)
        .map(|i| ((i * 11 % 7) as f32) * 0.1 - 0.2)
        .collect();

    let x = gpu.upload_f32(&x_host, &[M * K])?;
    let w = gpu.upload_f32(&w_host, &[N * K])?;

    // ── Analytic gradients ───────────────────────────────────────────────────
    // dL/dY = G, so upload G as dY.
    let dy = gpu.upload_f32(&g_host, &[M * N])?;
    let dx = gpu.zeros(&[M * K], DType::F32)?;
    let dw = gpu.zeros(&[N * K], DType::F32)?;
    linear_backward_x(&mut gpu, &dy, &w, &dx, M, K, N, false)?;
    linear_backward_w(&mut gpu, &dy, &x, &dw, M, K, N, false)?;
    let dx_analytic = gpu.download_f32(&dx)?;
    let dw_analytic = gpu.download_f32(&dw)?;

    // ── Numeric gradients (central differences) ──────────────────────────────
    let eps = 1e-3f32;
    let mut max_err_x = 0.0f32;
    for i in 0..M * K {
        let mut xp = x_host.clone();
        xp[i] += eps;
        let xpd = gpu.upload_f32(&xp, &[M * K])?;
        let lp = loss(&mut gpu, &xpd, &w, &g_host)?;
        let mut xm = x_host.clone();
        xm[i] -= eps;
        let xmd = gpu.upload_f32(&xm, &[M * K])?;
        let lm = loss(&mut gpu, &xmd, &w, &g_host)?;
        let num = (lp - lm) / (2.0 * eps);
        max_err_x = max_err_x.max((num - dx_analytic[i]).abs());
    }

    let mut max_err_w = 0.0f32;
    for i in 0..N * K {
        let mut wp = w_host.clone();
        wp[i] += eps;
        let wpd = gpu.upload_f32(&wp, &[N * K])?;
        let lp = loss(&mut gpu, &x, &wpd, &g_host)?;
        let mut wm = w_host.clone();
        wm[i] -= eps;
        let wmd = gpu.upload_f32(&wm, &[N * K])?;
        let lm = loss(&mut gpu, &x, &wmd, &g_host)?;
        let num = (lp - lm) / (2.0 * eps);
        max_err_w = max_err_w.max((num - dw_analytic[i]).abs());
    }

    println!("linear dX  max|analytic-numeric| = {max_err_x:.2e}");
    println!("linear dW  max|analytic-numeric| = {max_err_w:.2e}");

    let tol = 1e-2f32; // fp32 + central-diff truncation; linear is exact in theory
    if max_err_x < tol && max_err_w < tol {
        println!("\nGRADCHECK PASS — linear backward matches finite differences.");
        Ok(())
    } else {
        Err(
            format!("gradcheck FAIL (tol {tol:.0e}): dX {max_err_x:.2e}, dW {max_err_w:.2e}")
                .into(),
        )
    }
}
