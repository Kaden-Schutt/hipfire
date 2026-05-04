//! Side-by-side: FP16 wmma reference vs v1 iu4 vs v4 iu4 on real Qwen3.5-9B
//! layer-0 down_proj weights with realistic activations.
//!
//! Goal: diagnose whether v4's mu correction reduces or amplifies the per-
//! element error vs v1 (which has no mu term and smaller K=256 group dim).
//! If v4 is no better than v1 on real weights, the v4 design doesn't fix
//! the v1 quality blocker — falsifying-bar fails.

use engine::hfq::HfqFile;
use engine::hfq4v4::{convert_hfq4g256_to_hfq4v4, MuStrategy};
use rdna_compute::{DType, Gpu};
use std::path::PathBuf;

fn main() {
    let models_dir: PathBuf = std::env::var("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            eprintln!("HIPFIRE_MODELS_DIR not set; SKIP");
            std::process::exit(0);
        });
    let model = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5-9b.mq4".to_string());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let rotate = std::env::var("ROTATE").ok().as_deref() == Some("1");

    let path = models_dir.join(&model);
    if !path.exists() {
        eprintln!("model not found: {}; SKIP", path.display());
        std::process::exit(0);
    }

    let mut gpu = Gpu::init().expect("gpu init");
    if !(gpu.arch == "gfx1200" || gpu.arch == "gfx1201") {
        eprintln!("SKIP: requires gfx1200/gfx1201");
        std::process::exit(0);
    }
    eprintln!("=== v1 vs v4 vs FP16 — real layer-0 down_proj ===");

    let hfq = HfqFile::open(&path).expect("open hfq");
    let candidates: Vec<&str> = hfq
        .tensor_names()
        .filter(|nn| nn.contains("layers.0") && nn.ends_with("down_proj.weight"))
        .collect();
    let tensor = candidates[0];
    let (info, data) = hfq.tensor_data(tensor).unwrap();
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    eprintln!("tensor: {tensor} m={m} k={k}");

    let groups = k / 256;
    let row_bytes = groups * 136;
    let a_v1 = gpu.upload_raw(data, &[m * row_bytes]).unwrap();
    let (w_v4, mu_v4) =
        convert_hfq4g256_to_hfq4v4(data, m, k, rotate, &MuStrategy::WeightMean);
    let a_v4 = gpu.upload_raw(&w_v4, &[w_v4.len()]).unwrap();
    let mu_t = gpu.upload_raw(&mu_v4, &[mu_v4.len()]).unwrap();

    // Gaussian zero-mean activations.
    let mut state: u64 = 0xBEEF_CAFE_FACE_F00D;
    let mut rand_f32 = || -> f32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32) as f32 / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x_host: Vec<f32> = {
        let mut buf = vec![0.0f32; n * k];
        for col in 0..n {
            for ki in 0..k {
                let u1 = rand_f32().abs().max(1e-7);
                let u2 = rand_f32();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                buf[col * k + ki] = g;
            }
            let off = col * k;
            let m_x: f32 = buf[off..off + k].iter().sum::<f32>() / k as f32;
            for v in &mut buf[off..off + k] {
                *v -= m_x;
            }
        }
        buf
    };

    let x_for_q4_1: Vec<f32> = if rotate {
        let mut buf = x_host.clone();
        for col in 0..n {
            let off = col * k;
            for g in 0..(k / 32) {
                let go = off + g * 32;
                let mut grp = [0f32; 32];
                grp.copy_from_slice(&buf[go..go + 32]);
                engine::hfq4v4::fwht_32(&mut grp);
                buf[go..go + 32].copy_from_slice(&grp);
            }
        }
        buf
    } else {
        x_host.clone()
    };

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    let x_gpu_q4_1 = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    let y_v1 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    let y_v4 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    let zero = vec![0.0f32; n * m];

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();
    gpu.hip.memcpy_htod(&x_gpu_q4_1.buf, bytes_of(&x_for_q4_1)).unwrap();

    // FP16 reference.
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&zero)).unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_v1, &x_gpu, &y_fp16, m, k, n).unwrap();

    // v1 iu4.
    gpu.hip.memcpy_htod(&y_v1.buf, bytes_of(&zero)).unwrap();
    let xq_v1 = gpu.ensure_q4_1_x(&x_gpu_q4_1, n, k).unwrap();
    gpu.gemm_hfq4g256_residual_iu4_gfx12(&a_v1, xq_v1, &y_v1, m, k, n, true).unwrap();

    // v4 iu4 + mu.
    gpu.hip.memcpy_htod(&y_v4.buf, bytes_of(&zero)).unwrap();
    let xq_v4 = gpu.ensure_q4_1_x(&x_gpu_q4_1, n, k).unwrap();
    gpu.gemm_hfq4v4_residual_iu4_gfx12(&a_v4, &mu_t, xq_v4, &y_v4, m, k, n, true).unwrap();

    gpu.hip.device_synchronize().unwrap();

    let y_fp16_h: Vec<f32> = gpu.download_f32(&y_fp16).unwrap();
    let y_v1_h: Vec<f32> = gpu.download_f32(&y_v1).unwrap();
    let y_v4_h: Vec<f32> = gpu.download_f32(&y_v4).unwrap();

    fn measure(name: &str, a: &[f32], b: &[f32], n: usize, m: usize) {
        let mut max_abs: f32 = 0.0;
        let mut sum_abs: f64 = 0.0;
        let mut sum_rel: f64 = 0.0;
        let mut over_10: usize = 0;
        let mut elig: usize = 0;
        for col in 0..n {
            for row in 0..m {
                let i = col * m + row;
                let err = (a[i] - b[i]).abs();
                if err > max_abs {
                    max_abs = err;
                }
                sum_abs += err as f64;
                if a[i].abs() > 0.1 {
                    let rel = err / a[i].abs();
                    sum_rel += rel as f64;
                    elig += 1;
                    if rel > 0.1 {
                        over_10 += 1;
                    }
                }
            }
        }
        let mean_abs = sum_abs / (n * m) as f64;
        let mean_rel = if elig > 0 { sum_rel / elig as f64 } else { 0.0 };
        let pct = 100.0 * over_10 as f32 / elig.max(1) as f32;
        println!(
            "{:>20}: max_abs={max_abs:.4}  mean_abs={mean_abs:.4}  mean_rel={mean_rel:.4}  pct>10%={pct:>5.1}%",
            name
        );
    }

    println!();
    println!("Comparing each path against FP16 reference:");
    println!("(rotate={rotate})");
    println!();
    measure("v1 iu4 vs FP16", &y_fp16_h, &y_v1_h, n, m);
    measure("v4 iu4 vs FP16", &y_fp16_h, &y_v4_h, n, m);
    println!();
    measure("v1 vs v4", &y_v1_h, &y_v4_h, n, m);
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
