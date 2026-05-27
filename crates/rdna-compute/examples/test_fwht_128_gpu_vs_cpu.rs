//! GPU FWHT-128 correctness vs CPU reference.
//!
//! This is the critical-path correctness gate for the MQ4G128 dispatch path.
//! If GPU and CPU disagree here, the codec (Task 9) will encode weights with a
//! FWHT that the kernel cannot invert, producing silent numerical corruption at
//! decode time. KLD validation would catch it eventually — but that costs hours,
//! while this test costs seconds.
//!
//! Run:
//!   source ./scripts/rocm-env.sh
//!   source ./scripts/gpu-lock.sh && gpu_acquire "fwht128-test"
//!   cargo run -p rdna-compute --release --example test_fwht_128_gpu_vs_cpu
//!   gpu_release

use rdna_compute::{gen_fwht_signs, DType, Gpu};

const GROUP: usize = 128;

fn cpu_fwht_128(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), GROUP);
    // Pre-multiply by signs1
    for i in 0..GROUP {
        x[i] *= signs1[i];
    }
    // Walsh-Hadamard butterfly
    let mut stride = 1;
    while stride < GROUP {
        let mut i = 0;
        while i < GROUP {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    // Scale and post-multiply by signs2
    let scale = 1.0f32 / (GROUP as f32).sqrt(); // 0.0883883476...
    for i in 0..GROUP {
        x[i] *= scale * signs2[i];
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    gpu.ensure_mq_signs_128()?;

    // 4 groups of 128 floats = K=512
    let k = 512usize;
    let n_groups = k / GROUP;
    let x_host: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 0.25).collect();

    // CPU reference
    let signs1 = gen_fwht_signs(43, GROUP);
    let signs2 = gen_fwht_signs(1043, GROUP);
    let mut cpu_out = x_host.clone();
    for g in 0..n_groups {
        let start = g * GROUP;
        cpu_fwht_128(&mut cpu_out[start..start + GROUP], &signs1, &signs2);
    }

    // GPU path — rotate_x_mq_128 (T5-preview stub, minimal dispatch added in dispatch.rs)
    let d_x = gpu.upload_f32(&x_host, &[k])?;
    let d_x_rot = gpu.zeros(&[k], DType::F32)?;
    gpu.rotate_x_mq_128(&d_x, &d_x_rot, k)?;
    let gpu_out = gpu.download_f32(&d_x_rot)?;

    // Compare
    let max_err = cpu_out
        .iter()
        .zip(&gpu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("CPU vs GPU max abs error: {:e}", max_err);
    println!("First 8 elements:");
    for i in 0..8 {
        println!(
            "  [{i}]  CPU={:.6}  GPU={:.6}  diff={:.2e}",
            cpu_out[i],
            gpu_out[i],
            (cpu_out[i] - gpu_out[i]).abs()
        );
    }
    if max_err >= 1e-4 {
        eprintln!("\nFAIL: max abs error {:e} >= 1e-4", max_err);
        std::process::exit(1);
    }
    println!("\nPASS: GPU FWHT-128 matches CPU reference within 1e-4");
    Ok(())
}
