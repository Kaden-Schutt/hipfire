//! Diagnostic: compare FP16 vs calibrated-iu4 output for layer-0 wo on a
//! REAL model + REAL calibration sidecar. This is the bridge between the
//! synthetic-data correctness test (gfx1201) and the daemon coherence
//! gate. If this diverges by > Q4_1-noise on a single layer, the layer-0
//! daemon falsification will too.
//!
//! Flow:
//!   1. Load model + sidecar.
//!   2. Construct a synthetic activation `x` with the same per-channel
//!      stats as the captured mu/s.
//!   3. Path A: FP16 dequant + WMMA (gfx12) on the original weight.
//!   4. Path B: bake weight scales, preshift x, q4_1 quant, iu4 GEMM,
//!      bias_add. (Same dispatch as the daemon.)
//!   5. Compare outputs element-wise; print per-row stats.
//!
//! Usage:
//!   cargo run --release --features deltanet --example
//!     test_iu4_calibration_layer0 -- --model <path> --site 0

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("requires --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::quant::iu4_calibration::{Iu4Calibration, upload_to_gpu};
    use engine::qwen35;
    use rdna_compute::Gpu;

    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut site_idx: usize = 0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--site" => {
                site_idx = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
                i += 2;
            }
            _ => i += 1,
        }
    }
    let Some(model_path) = model_path else {
        eprintln!("--model required");
        std::process::exit(2);
    };
    let sidecar_path = format!("{model_path}.iu4cal");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    if !(gpu.arch == "gfx1200" || gpu.arch == "gfx1201") {
        eprintln!("SKIP: requires gfx1200/gfx1201");
        std::process::exit(0);
    }
    let hfq = HfqFile::open(std::path::Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("Loading weights...");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let cal = Iu4Calibration::read_path(std::path::Path::new(&sidecar_path))
        .expect("read sidecar");
    eprintln!("sidecar: {} sites", cal.n_sites());

    // Inspect site
    let site = cal.sites.get(site_idx).expect("site idx out of range");
    let m = site.n_output_rows as usize;
    let k = site.n_channels as usize;
    eprintln!(
        "Site {site_idx}: layer={}, proj={}, m={m}, k={k}",
        site.layer_idx, site.proj_id
    );

    // Find the corresponding weight tensor.
    let layer_idx = site.layer_idx as usize;
    let proj_id = site.proj_id;
    let weight = match (&weights.layers[layer_idx], proj_id) {
        (qwen35::LayerWeights::FullAttn(l), 0) => &l.wo,
        (qwen35::LayerWeights::FullAttn(l), 1) => &l.w_down,
        (qwen35::LayerWeights::DeltaNet(l), 0) => &l.wo,
        (qwen35::LayerWeights::DeltaNet(l), 1) => &l.w_down,
        _ => panic!("unsupported layer kind for diagnostic"),
    };

    // Generate a small synthetic activation matrix matching the capture
    // shape: a few random tokens with realistic magnitudes (≈ mu_a's
    // magnitude scale to mimic the post-rmsnorm distribution).
    let n: usize = 16;
    let mu_a_f32 = site.mu_a_f32();
    let (s_a_f32, _inv) = site.s_a_f32_with_inv(1e-6);

    // x[t][c] = mu[c] + s[c] * z, z ~ Uniform(-1, 1).  With a 1% outlier at ±5×.
    let mut state: u64 = 0xCAFEBABE;
    let mut nrand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (u32::MAX as f32)
    };
    let mut x_host = vec![0.0f32; n * k];
    for t in 0..n {
        for c in 0..k {
            let u = nrand();
            let z = nrand() * 2.0 - 1.0;
            let mag = if u > 0.99 { 5.0 } else { 1.0 };
            x_host[t * k + c] = mu_a_f32[c] + s_a_f32[c] * z * mag;
        }
    }

    let x_gpu = gpu
        .alloc_tensor(&[n * k], rdna_compute::DType::F32)
        .expect("alloc x");
    gpu.hip
        .memcpy_htod(&x_gpu.buf, bytes_of(&x_host))
        .expect("htod x");
    let y_fp16 = gpu
        .alloc_tensor(&[n * m], rdna_compute::DType::F32)
        .expect("alloc y_fp16");
    let y_iu4 = gpu
        .alloc_tensor(&[n * m], rdna_compute::DType::F32)
        .expect("alloc y_iu4");
    // Zero y so the residual-add lands on a known baseline.
    gpu.hip.memset(&y_fp16.buf, 0, n * m * 4).unwrap();
    gpu.hip.memset(&y_iu4.buf, 0, n * m * 4).unwrap();

    // Path A: FP16 wmma reference
    eprintln!("--- path A: FP16 reference ---");
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&weight.buf, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: calibrated iu4 (load sidecar + go through dispatcher)
    eprintln!("--- path B: calibrated iu4 ---");
    let gpu_cal = upload_to_gpu(&cal, &mut gpu).expect("upload sidecar");
    gpu.load_iu4_calibration(gpu_cal);
    // Manual dispatch: skip layers 0..site_idx by incrementing the counter
    // so site_idx is the next call.
    gpu.iu4_dispatch_call_idx.set(site_idx);
    std::env::set_var("HIPFIRE_GFX12_IU4_CALIBRATED", "1");
    std::env::set_var("HIPFIRE_GFX12_IU4_MAX_CALL", &(site_idx + 1).to_string());
    gpu.gemm_hfq4g256_residual(&weight.buf, &x_gpu, &y_iu4, m, k, n)
        .expect("calibrated iu4");
    std::env::remove_var("HIPFIRE_GFX12_IU4_CALIBRATED");
    std::env::remove_var("HIPFIRE_GFX12_IU4_MAX_CALL");
    gpu.hip.device_synchronize().unwrap();
    let y_iu4_host = gpu.download_f32(&y_iu4).expect("download y_iu4");

    // Compare
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_loc = (0usize, 0usize);
    for t in 0..n {
        for r in 0..m {
            let idx = t * m + r;
            let a = y_fp16_host[idx];
            let b = y_iu4_host[idx];
            let e = (a - b).abs();
            if e > max_abs {
                max_abs = e;
                max_loc = (t, r);
            }
            sum_abs += e as f64;
        }
    }
    let mean_abs = sum_abs / (n * m) as f64;
    eprintln!("\n=== Comparison ===");
    eprintln!("  max abs err:  {max_abs:.6} at (t={}, r={})", max_loc.0, max_loc.1);
    eprintln!("  mean abs err: {mean_abs:.6}");
    let mut max_a: f32 = 0.0;
    let mut max_b: f32 = 0.0;
    for &v in &y_fp16_host {
        if v.abs() > max_a {
            max_a = v.abs();
        }
    }
    for &v in &y_iu4_host {
        if v.abs() > max_b {
            max_b = v.abs();
        }
    }
    eprintln!("  max |y_fp16|: {max_a:.4}");
    eprintln!("  max |y_iu4|:  {max_b:.4}");
    eprintln!("\n--- Sample triples (t=0..2, r=0..6) ---");
    for t in 0..2.min(n) {
        for r in 0..6.min(m) {
            let idx = t * m + r;
            let a = y_fp16_host[idx];
            let b = y_iu4_host[idx];
            eprintln!("  t={t} r={r}: fp16={a:>10.4}  cal={b:>10.4}  err={:.4}", (a - b).abs());
        }
    }

    eprintln!("\n--- Site summary ---");
    eprintln!("  mu_a: range [{:.4}, {:.4}]", min_of(&mu_a_f32), max_of(&mu_a_f32));
    eprintln!("  s_a:  range [{:.4}, {:.4}]", min_of(&s_a_f32), max_of(&s_a_f32));
    eprintln!("  bias: range [{:.4}, {:.4}]",
        min_of(&site.w_mu_bias_f32()), max_of(&site.w_mu_bias_f32()));
}

#[cfg(feature = "deltanet")]
fn min_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::INFINITY, f32::min)
}
#[cfg(feature = "deltanet")]
fn max_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
#[cfg(feature = "deltanet")]
fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
