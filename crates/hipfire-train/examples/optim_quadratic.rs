//! AdamW convergence test (Phase 0, M3). Minimize L = ½Σ(p − target)² (so the
//! gradient is exactly p − target) and confirm AdamW drives p → target. This
//! validates the optimizer kernel + bias correction independently of the model.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "optim-quad"
//!   cargo run -p hipfire-train --release --example optim_quadratic
//!   gpu_release

use hipfire_train::optim::AdamW;
use rdna_compute::Gpu;

const N: usize = 16;
const STEPS: usize = 400;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let target: Vec<f32> = (0..N).map(|i| ((i * 7 % 11) as f32) * 0.3 - 1.5).collect();
    let p_host: Vec<f32> = (0..N).map(|i| ((i * 5 % 9) as f32) * 0.2 + 0.5).collect();
    let p = gpu.upload_f32(&p_host, &[N])?;

    // AdamW with sft.py-style betas; lr higher for a fast unit test.
    let mut opt = AdamW::new(&mut gpu, &[N], 0.1, 0.9, 0.999, 1e-8, 0.0)?;

    let mut loss = 0.0f32;
    for step in 0..STEPS {
        let pv = gpu.download_f32(&p)?;
        // grad = p - target  (host-computed; isolates the optimizer)
        let grad: Vec<f32> = pv.iter().zip(&target).map(|(a, b)| a - b).collect();
        loss = grad.iter().map(|d| 0.5 * d * d).sum();
        let g = gpu.upload_f32(&grad, &[N])?;
        opt.step(&mut gpu, &[&p], &[&g])?;
        if step % 100 == 0 {
            println!("step {step:4}: loss = {loss:.6}");
        }
    }

    let pv = gpu.download_f32(&p)?;
    let max_err = pv.iter().zip(&target).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("final loss = {loss:.3e}, max|p-target| = {max_err:.3e} after {STEPS} steps");

    if max_err < 1e-3 {
        println!("\nPASS — AdamW converged p → target.");
        Ok(())
    } else {
        Err(format!("FAIL — AdamW did not converge (max_err {max_err:.3e})").into())
    }
}
