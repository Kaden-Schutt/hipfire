//! Flaky-719 SHAPE ISOLATION (LDS gemm_f32_train). The single 3072 forward-NT
//! never faults, but the full bench (mlp 8192 shapes + NN/TN backward) faults
//! ~70%. Step through each shape/case in order, printing before each launch+sync;
//! the 719 kills the process AT the faulting step, so running this N times tallies
//! which shape is the trigger.

use hipfire_rdna::{DType, Gpu};

fn step(
    gpu: &mut Gpu,
    label: &str,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
    c: &hipfire_rdna::GpuTensor,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ta: bool,
    tb: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("STEP {label}: launch M={m} N={n} K={k} ta={ta} tb={tb}");
    gpu.gemm_f32_train(a, b, c, m, n, k, lda, ldb, ta, tb)?;
    let _ = gpu.download_f32(c)?; // sync barrier — surfaces a fault here
    println!("STEP {label}: OK");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("init");
    let big = 8192usize;
    // buffers sized for the LARGEST output of any step (TN is [3072,3072]=9.4M,
    // gate is [512,8192]=4.2M). Over-allocate so no step writes OOB.
    let cap = big * big; // 67M elems, covers every shape below
    let x = gpu.upload_f32(&vec![0.01f32; cap], &[cap])?;
    let w = gpu.upload_f32(&vec![0.01f32; cap], &[cap])?;
    let c = gpu.zeros(&[cap], DType::F32)?;

    // RECOVERABILITY: 100 launches, each wrapped in a per-launch retry that
    // clears the error + syncs + relaunches on fault. If the process completes
    // all 100, the LDS fault is RECOVERABLE → a retry wrapper makes it reliable.
    let n_launch = 100;
    let mut total_retries = 0u32;
    for i in 0..n_launch {
        let mut ok = false;
        for r in 0..8 {
            let _ = gpu.gemm_f32_train(&x, &w, &c, 512, 3072, 3072, 3072, 3072, false, true);
            match gpu.device_synchronize() {
                Ok(()) => {
                    ok = true;
                    break;
                }
                Err(_) => {
                    total_retries += 1;
                    let _ = gpu.clear_last_error();
                    let _ = gpu.device_synchronize();
                    let _ = gpu.clear_last_error();
                    if r == 7 {
                        println!("UNRECOVERABLE at launch {i} after 8 retries");
                        std::process::exit(2);
                    }
                }
            }
        }
        if !ok {
            std::process::exit(2);
        }
    }
    println!("ALL {n_launch} LAUNCHES OK (total_retries={total_retries}) — RECOVERABLE");
    let _ = step;
    Ok(())
}
