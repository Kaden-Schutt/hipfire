//! Verify the HFQ4v4 + iu4 K=32 GEMM on a REAL Qwen3.5-9B layer-0 weight,
//! not synthetic LCG bytes. This catches the v1 quality-fail class of bug:
//! the v1 iu4 kernel passed the synthetic correctness gate (PR #140 has the
//! same kind of bench) but produced complete garbage on real weights.
//!
//! Methodology:
//!   1. Open a real .hfq model file (e.g. qwen3.5-9b.mq4).
//!   2. Locate layer 0's wo or w_down weight (HFQ4-G256 / MQ4-G256).
//!   3. Synthesize a "post-RMSNorm" activation distribution (zero-mean,
//!      unit-ish variance per token, K elements per token).
//!   4. Compute FP16-WMMA reference output (path A).
//!   5. Convert the weight to HFQ4v4 + mu sidecar.
//!   6. Run the v4 GEMM (path B).
//!   7. Compare element-wise. Same thresholds as the synthetic test.
//!
//! Gate behavior: if real-weight v4 fails BUT synthetic v4 passed, the
//! conversion code (or kernel address arithmetic) is wrong on real
//! distributions. STOP — don't assume the kernel is correct.
//!
//! Run on hiptrx:
//!   HIPFIRE_MODELS_DIR=$HOME/.hipfire/models \
//!     cargo run --release -p engine --example test_hfq4v4_real_layer
//!
//! Env knobs:
//!   MODEL=qwen3.5-9b.mq4  — model filename in HIPFIRE_MODELS_DIR
//!   TENSOR=model.layers.0.self_attn.o_proj.weight  — weight to test
//!   N=128                 — batch size
//!   ROTATE=1              — apply FWHT-32

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
    let tensor = std::env::var("TENSOR")
        .unwrap_or_else(|_| "model.layers.0.self_attn.o_proj.weight".to_string());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let rotate = std::env::var("ROTATE").ok().as_deref() == Some("1");

    let path = models_dir.join(&model);
    if !path.exists() {
        eprintln!("model not found: {}; SKIP", path.display());
        std::process::exit(0);
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!("SKIP: requires gfx1200/gfx1201 (RDNA4)");
        std::process::exit(0);
    }

    eprintln!("=== gfx12 HFQ4v4 real-weight correctness test ===");
    eprintln!("model: {}", path.display());
    eprintln!("tensor: {tensor}");
    eprintln!("N={n}, rotate={rotate}");

    let hfq = HfqFile::open(&path).expect("open hfq");
    let (info, data) = hfq.tensor_data(&tensor).unwrap_or_else(|| {
        eprintln!("tensor not found: {tensor}");
        std::process::exit(1);
    });
    if info.quant_type != 6 && info.quant_type != 13 {
        eprintln!(
            "tensor {tensor} has quant_type {} (need HFQ4-G256=6 or MQ4-G256=13)",
            info.quant_type
        );
        std::process::exit(1);
    }
    if info.shape.len() != 2 {
        eprintln!("tensor {tensor} is not 2-D");
        std::process::exit(1);
    }
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    eprintln!("  m={m}, k={k}, bytes={}", data.len());

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    assert_eq!(data.len(), m * row_bytes, "tensor data size mismatch");

    let a_v1 = gpu
        .upload_raw(data, &[m * row_bytes])
        .expect("upload v1 weights");

    let (w_v4, mu_v4) =
        convert_hfq4g256_to_hfq4v4(data, m, k, rotate, &MuStrategy::WeightMean);
    let a_v4 = gpu.upload_raw(&w_v4, &[w_v4.len()]).expect("upload v4");
    let mu_t = gpu.upload_raw(&mu_v4, &[mu_v4.len()]).expect("upload mu");

    eprintln!(
        "  v4 weight blob: {} bytes ({:.3} bits/weight)",
        w_v4.len(),
        (w_v4.len() as f32 * 8.0) / (m * k) as f32
    );

    // Synthesize a post-RMSNorm-like activation distribution: zero-mean per
    // token, std-dev around 1.0. This is a plausible match for what the
    // kernel sees in the real forward pass.
    let mut state: u64 = 0xBEEF_CAFE_FACE_F00D;
    let mut rand_f32 = || -> f32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((state >> 33) as u32) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
        u
    };
    let x_host: Vec<f32> = {
        let mut buf = vec![0.0f32; n * k];
        for col in 0..n {
            // Box-Muller for ~Gaussian samples per token.
            for ki in 0..k {
                let u1 = rand_f32().abs().max(1e-7);
                let u2 = rand_f32();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                buf[col * k + ki] = g;
            }
            // Center the per-token row.
            let off = col * k;
            let mu_x: f32 = buf[off..off + k].iter().sum::<f32>() / k as f32;
            for v in &mut buf[off..off + k] {
                *v -= mu_x;
            }
        }
        buf
    };

    let y_init: Vec<f32> = vec![0.0f32; n * m];

    // For mq4v4 (rotate), CPU-rotate the activations the same way the kernel
    // expects (matches what production runtime would do via a GPU FWHT-32
    // kernel — TODO follow-up).
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

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    let x_gpu_for_q4_1 = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x_rot");
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
    let y_v4 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_v4");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();
    gpu.hip.memcpy_htod(&x_gpu_for_q4_1.buf, bytes_of(&x_for_q4_1)).unwrap();

    // Path A: FP16 reference (uses ORIGINAL non-rotated weights).
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_v1, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: HFQ4v4 + iu4 K=32 with mu correction.
    gpu.hip.memcpy_htod(&y_v4.buf, bytes_of(&y_init)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let xq_ptr = gpu.ensure_q4_1_x(&x_gpu_for_q4_1, n, k).expect("ensure_q4_1_x");
    gpu.gemm_hfq4v4_residual_iu4_gfx12(&a_v4, &mu_t, xq_ptr, &y_v4, m, k, n, true)
        .expect("hfq4v4 iu4 gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_v4_host: Vec<f32> = gpu.download_f32(&y_v4).expect("download y_v4");

    const REL_FLOOR: f32 = 0.1;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_rel_err: f64 = 0.0;
    let mut samples_above_10pct: usize = 0;
    let mut rel_eligible: usize = 0;
    let mut max_loc: (usize, usize) = (0, 0);

    for col in 0..n {
        for row in 0..m {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_v4_host[idx];
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
                max_loc = (col, row);
            }
            sum_abs_err += err as f64;
            if a.abs() > REL_FLOOR {
                let rel = err / a.abs();
                if rel > max_rel_err {
                    max_rel_err = rel;
                }
                sum_rel_err += rel as f64;
                rel_eligible += 1;
                if rel > 0.10 {
                    samples_above_10pct += 1;
                }
            }
        }
    }

    let total = (n * m) as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 { sum_rel_err / rel_eligible as f64 } else { 0.0 };
    let pct_above = 100.0 * samples_above_10pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-channel error (n*m = {} elements) ---", n * m);
    eprintln!(
        "  max abs err:                       {:.6}  at (col={}, row={})",
        max_abs_err, max_loc.0, max_loc.1
    );
    eprintln!("  mean abs err:                      {:.6}", mean_abs_err);
    eprintln!(
        "  rel-err eligible (|out| > {:.2}):    {} / {} ({:.1}%)",
        REL_FLOOR,
        rel_eligible,
        n * m,
        100.0 * rel_eligible as f32 / (n * m) as f32
    );
    eprintln!("  max rel err†:                      {:.4}", max_rel_err);
    eprintln!("  mean rel err†:                     {:.4}", mean_rel_err);
    eprintln!(
        "  samples > 10% rel†:                {} / {} ({:.3}%)",
        samples_above_10pct,
        rel_eligible.max(1),
        pct_above
    );

    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_v4_host[idx];
            eprintln!(
                "  col={col} row={row}: fp16={a:>10.4}  v4={b:>10.4}  err={:.4}",
                (a - b).abs()
            );
        }
    }

    let max_abs_thresh = 0.30;
    let mean_abs_thresh = 0.03;
    let mean_rel_thresh = 0.10;
    let pct_thresh = 40.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:  {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}:  {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >10% rel† < {pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: gfx12 HFQ4v4 GEMM stays within tolerance on real weights.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: gfx12 HFQ4v4 GEMM diverges on real weights — kernel design or converter is wrong.");
        std::process::exit(1);
    }
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
