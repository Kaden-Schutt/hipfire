//! gfx12 (RDNA4) HFQ4-G256 MMQ correctness test (issue #136 part B).
//!
//! Compares the new gfx12 MMQ residual GEMM kernel
//! (`gemm_hfq4g256_residual_mmq_gfx12`, commit 5757283) against the existing
//! gfx12 FP16 dequant→WMMA reference path (`gemm_hfq4g256_residual_wmma_gfx12`).
//! Both compute `Y += W·X` for HFQ4 weights and FP32 activations on the same
//! random inputs.
//!
//! The MMQ path quantizes activations to Q8_1 mid-flight (loses ~0.4% per
//! element vs FP32), so outputs differ by Q8_1 rounding noise — bounded by
//! roughly 0.4% × √K accumulated. For K=512 that's ~9% upper bound; in
//! practice (with structured-but-random data) we usually see <2% per row.
//!
//! PASS thresholds (per channel = per output element):
//!   max abs err        < 0.10  (tracks the default mmq_screen_threshold)
//!   mean abs err       < 0.01  (1% of typical output magnitude)
//!   max rel err†       < 0.05  (5% per element, only for |output| > 0.05)
//!   mean rel err†      < 0.01  (1% per element on average)
//!   pct rel-err > 5%†  < 1.0%  (less than 1% of elements significantly off)
//!
//! † Rel-err is computed only for elements where |fp16_output| > 0.05.
//!   For near-zero outputs (where Q8_1 rounding noise dominates), abs-err
//!   matters; rel-err blows up by a factor of (1 / output_magnitude) and
//!   stops being meaningful. The previous version floored the denominator
//!   at 1e-3, which produced spurious 73% "rel err" on outputs of 0.002.
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example test_mmq_gfx12_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);

    assert!(k % 256 == 0, "K must be a multiple of 256 for HFQ4-G256");
    assert!(m % 16 == 0, "M should be aligned to wmma tile (16)");
    assert!(n % 16 == 0, "N should be aligned to wmma tile (16)");

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!(
            "SKIP: this test requires gfx1200/gfx1201 (RDNA4). \
             Current arch: {arch}. The gfx12 MMQ kernel only exists on RDNA4."
        );
        std::process::exit(0);
    }

    eprintln!("=== gfx12 MMQ vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    eprintln!("weight tensor: {} MiB", (m * row_bytes) as f64 / (1024.0 * 1024.0));

    // --- Random HFQ4-G256 weights on GPU.
    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    // --- Random FP32 activations + initial Y. Deterministic PRNG.
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
    let y_mmq = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_mmq");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();

    // --- Path A: FP16 dequant -> WMMA reference (existing production gfx12 path).
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a_raw, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // --- Path B: MMQ via the new gfx12 kernel (Q8_1 activation quant + iu8 K=16 wmma).
    gpu.hip.memcpy_htod(&y_mmq.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_mmq(&a_raw, &x_gpu, &y_mmq, m, k, n)
        .expect("mmq gfx12");
    gpu.hip.device_synchronize().unwrap();
    let y_mmq_host: Vec<f32> = gpu.download_f32(&y_mmq).expect("download y_mmq");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_mmq_host.len(), n * m);

    // --- Compare per channel.
    // Rel-err is computed only on elements where |fp16_output| > REL_FLOOR
    // (= 0.05). Near-zero outputs have abs-err in the noise floor of Q8_1
    // rounding (~1e-3) but explode rel-err by 1/|output|, producing spurious
    // failures. Abs-err is the meaningful metric across the whole tensor.
    const REL_FLOOR: f32 = 0.05;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_rel_err: f64 = 0.0;
    let mut max_loc: (usize, usize) = (0, 0);
    let mut samples_above_5pct: usize = 0;
    let mut rel_eligible: usize = 0;

    for col in 0..n {
        for row in 0..m {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_mmq_host[idx];
            let err = (a - b).abs();

            if err > max_abs_err {
                max_abs_err = err;
                max_loc = (col, row);
            }
            sum_abs_err += err as f64;

            // Rel-err only on non-near-zero outputs.
            if a.abs() > REL_FLOOR {
                let rel = err / a.abs();
                if rel > max_rel_err {
                    max_rel_err = rel;
                }
                sum_rel_err += rel as f64;
                rel_eligible += 1;
                if rel > 0.05 {
                    samples_above_5pct += 1;
                }
            }
        }
    }

    let total = (n * m) as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 {
        sum_rel_err / (rel_eligible as f64)
    } else {
        0.0
    };

    eprintln!("\n--- per-channel error (n*m = {} elements) ---", n * m);
    eprintln!("  max abs err:                       {:.6}  at (col={}, row={})", max_abs_err, max_loc.0, max_loc.1);
    eprintln!("  mean abs err:                      {:.6}", mean_abs_err);
    eprintln!("  rel-err eligible (|out| > {:.2}):    {} / {} ({:.1}%)",
              REL_FLOOR, rel_eligible, n * m,
              100.0 * rel_eligible as f32 / (n * m) as f32);
    eprintln!("  max rel err†:                      {:.4}", max_rel_err);
    eprintln!("  mean rel err†:                     {:.4}", mean_rel_err);
    eprintln!("  samples > 5% rel†:                 {} / {} ({:.3}%)",
              samples_above_5pct, rel_eligible.max(1),
              100.0 * samples_above_5pct as f32 / rel_eligible.max(1) as f32);
    eprintln!("  † counted only on non-near-zero outputs (|out| > {REL_FLOOR})");

    // --- Show a small sample of (fp16, mmq, err) triples for human eyeball.
    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_mmq_host[idx];
            eprintln!("  col={col} row={row}: fp16={a:>10.4}  mmq={b:>10.4}  err={:.4}", (a - b).abs());
        }
    }

    // --- Pass / fail.
    let max_abs_thresh = 0.10;
    let mean_abs_thresh = 0.01;
    let max_rel_thresh = 0.05;
    let mean_rel_thresh = 0.01;
    let pct_above_5pct_thresh = 1.0;

    let pct_above = 100.0 * samples_above_5pct as f32 / rel_eligible.max(1) as f32;
    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_above_5pct_thresh;

    eprintln!("\n--- PASS criteria ---");
    eprintln!("  max abs err   < {max_abs_thresh}:    {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}:   {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err†  < {max_rel_thresh}:    {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}:   {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >5% rel†  < {pct_above_5pct_thresh}%: {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: gfx12 MMQ kernel is numerically equivalent to FP16 WMMA \
                   reference within Q8_1 activation-quantization tolerance.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: gfx12 MMQ kernel diverges from FP16 WMMA reference \
                   beyond Q8_1 tolerance. Investigate before flipping the \
                   HIPFIRE_GFX12_MMQ default.");
        std::process::exit(1);
    }
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            // Scale: small finite positive FP32. Clamp magnitude to [1e-3, 1e-2].
            let scale_bits = 0x3a000000u32 | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0x80) << 24) | 0x39000000u32 | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok    = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
