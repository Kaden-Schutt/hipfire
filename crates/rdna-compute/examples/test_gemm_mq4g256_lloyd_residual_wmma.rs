//! Phase A parity test for gemm_mq4g256_lloyd_residual_wmma.
//!
//! Sibling of `test_gemm_mq3g256_lloyd_residual_wmma.rs` (commit 869236d).
//! Single-variant test (fp16-LDS only) — fp32-LDS sibling skipped per
//! MQ3 Phase A's empirical conclusion (fp16 wins 7.15% with bit-identical
//! numerical output; see benchmarks/results/devlog_20260507_lloyd_wmma_phase_a.md).
//!
//! Sweeps the same (M, K, N) ∈ {64, 256, 1024} × {1024, 4096, 12288} × {16, 64, 256}
//! 8-shape grid. CPU reference is fp64-accumulated; X is fp16-roundtripped to
//! match the GPU's view after `ensure_fp16_x`.
//!
//! Tolerance starts at 1.75e-4 (3× MQ3's observed 5.83e-5 max-abs at K=12288).
//! Phase A acceptance includes logging the actual MQ4 max-abs and confirming it
//! stays in the same envelope.

use rdna_compute::{Gpu, DType, LLOYD_MQ4_GROUP_BYTES};

/// f32 → IEEE 754 binary16 little-endian, RTNE on dropped 13 mantissa bits.
fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h: u16 = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) {
            new_mant += 1;
        }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 {
            new_mant = 0;
            exp_bits += 1;
        }
        (sign << 15) | (exp_bits << 10) | new_mant
    };
    h.to_le_bytes()
}

fn f16_le_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let mant32 = (m & 0x3ff) << 13;
            let exp32 = (e + 127 - 14) as u32;
            (sign << 31) | (exp32 << 23) | mant32
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// MQ4-Lloyd nibble-pair packing: 256 indices → 128 bytes.
/// byte[i] low nibble = idx[2i], high nibble = idx[2i+1] —
/// matches `quantize_mq4g256_lloyd` in crates/hipfire-quantize/src/main.rs
/// and the kernel decode in gemm_mq4g256_lloyd_residual_wmma.hip.
fn pack_4bit_group(qs: &[u8; 256]) -> [u8; 128] {
    let mut out = [0u8; 128];
    for i in 0..128 {
        let lo = qs[2 * i] & 0x0F;
        let hi = qs[2 * i + 1] & 0x0F;
        out[i] = lo | (hi << 4);
    }
    out
}

/// Builds row bytes (160 B/group: 32 B fp16 codebook header + 128 B nibble
/// indices) AND returns f16-roundtripped codebooks so the CPU reference uses
/// exactly the values the GPU will dequant.
fn build_lloyd_row(
    groups_per_row: usize,
    codebooks: &[[f32; 16]],
    indices: &[[u8; 256]],
) -> (Vec<u8>, Vec<[f32; 16]>) {
    let mut out = Vec::with_capacity(groups_per_row * LLOYD_MQ4_GROUP_BYTES);
    let mut roundtripped = Vec::with_capacity(groups_per_row);
    for g in 0..groups_per_row {
        let cb = &codebooks[g];
        let mut cb_rt = [0.0f32; 16];
        for (i, &v) in cb.iter().enumerate() {
            let bytes = f32_to_f16_le(v);
            out.extend_from_slice(&bytes);
            cb_rt[i] = f16_le_to_f32(bytes);
        }
        roundtripped.push(cb_rt);
        let packed = pack_4bit_group(&indices[g]);
        out.extend_from_slice(&packed);
    }
    assert_eq!(out.len(), groups_per_row * LLOYD_MQ4_GROUP_BYTES);
    (out, roundtripped)
}

/// CPU reference GEMM with residual. fp64-accumulated; X is f16-roundtripped
/// to match the GPU's view after `ensure_fp16_x`.
fn cpu_reference_gemm(
    m: usize, k: usize, n: usize,
    codebooks_per_row: &[Vec<[f32; 16]>],
    indices_per_row: &[Vec<[u8; 256]>],
    x_fp32: &[f32],
    y_init: &[f32],
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let x_rt: Vec<f32> = x_fp32.iter().map(|&v| f16_le_to_f32(f32_to_f16_le(v))).collect();
    let mut y = y_init.to_vec();
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            let cbs = &codebooks_per_row[row];
            let idxs = &indices_per_row[row];
            for g in 0..groups_per_row {
                let cb = &cbs[g];
                let qs = &idxs[g];
                for i in 0..256 {
                    let q = qs[i] as usize & 0xF;
                    let k_idx = g * 256 + i;
                    acc += cb[q] as f64 * x_rt[col * k + k_idx] as f64;
                }
            }
            y[col * m + row] += acc as f32;
        }
    }
    y
}

/// Bench harness for one variant. `bench_fn` is the kernel to measure;
/// `name` is just for the printf. Returns (max_abs, max_rel, rms, us/call).
fn bench_variant(
    gpu: &mut Gpu,
    m: usize, _k: usize, n: usize,
    d_a: &rdna_compute::GpuTensor,
    d_x: &rdna_compute::GpuTensor,
    y_init: &[f32],
    y_ref: &[f32],
    bench_fn: impl Fn(&mut Gpu, &rdna_compute::GpuTensor, &rdna_compute::GpuTensor, &rdna_compute::GpuTensor),
) -> (f32, f32, f32, f64) {
    let d_y = gpu.upload_f32(y_init, &[n, m]).unwrap();
    bench_fn(gpu, d_a, d_x, &d_y);
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let n_warmup = 3usize;
    let n_iter = 20usize;
    let d_y_bench = gpu.zeros(&[n, m], DType::F32).unwrap();
    for _ in 0..n_warmup {
        bench_fn(gpu, d_a, d_x, &d_y_bench);
    }
    let _ = gpu.download_f32(&d_y_bench).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        bench_fn(gpu, d_a, d_x, &d_y_bench);
    }
    let _ = gpu.download_f32(&d_y_bench).unwrap();
    let elapsed_us_per_call = t0.elapsed().as_secs_f64() * 1e6 / n_iter as f64;
    gpu.free_tensor(d_y_bench).unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_sq_err = 0.0f64;
    let total = n * m;
    for i in 0..total {
        let abs = (y_gpu[i] - y_ref[i]).abs();
        let denom = y_ref[i].abs().max(1e-3);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
        sum_sq_err += (abs as f64) * (abs as f64);
    }
    let rms_err = (sum_sq_err / total as f64).sqrt() as f32;

    gpu.free_tensor(d_y).unwrap();
    (max_abs, max_rel, rms_err, elapsed_us_per_call)
}

fn run_one(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> (
    (f32, f32, f32, f64),  // _wmma     (Phase A)
    Option<(f32, f32, f32, f64)>,  // _wmma_mb2 (Phase D experiment)
    Option<(f32, f32, f32, f64)>,  // _wmma_mb4 (Phase D-A)
) {
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;

    let mut a_rows = Vec::with_capacity(m);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            // Synthetic codebook: 16 ascending centroids around zero, varied per
            // (row, g) so different rows produce distinguishable outputs.
            let base = ((row.wrapping_mul(7) + g.wrapping_mul(11)) % 19) as f32 * 0.013 - 0.1;
            let cb: [f32; 16] = std::array::from_fn(|i| base + (i as f32 - 7.5) * 0.018);
            cbs.push(cb);

            // Synthetic indices in [0, 16).
            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)) & 0xF) as u8;
            }
            idxs.push(q);
        }
        let (row_bytes, cbs_rt) = build_lloyd_row(groups_per_row, &cbs, &idxs);
        a_rows.push(row_bytes);
        codebooks_per_row.push(cbs_rt);
        indices_per_row.push(idxs);
    }

    let x: Vec<f32> = (0..(n * k))
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    let y_init: Vec<f32> = (0..(n * m))
        .map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.001)
        .collect();

    let mut a_flat: Vec<u8> = Vec::with_capacity(m * groups_per_row * LLOYD_MQ4_GROUP_BYTES);
    for row in &a_rows {
        a_flat.extend_from_slice(row);
    }

    let d_a = gpu.upload_raw(&a_flat, &[a_flat.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

    let y_ref = cpu_reference_gemm(m, k, n, &codebooks_per_row, &indices_per_row, &x, &y_init);

    let phase_a = bench_variant(
        gpu, m, k, n, &d_a, &d_x, &y_init, &y_ref,
        |gpu, d_a, d_x, d_y| {
            // Force MB4=0 to skip the size-gated routing.
            std::env::set_var("HIPFIRE_LLOYD_MB4", "0");
            gpu.gemm_mq4g256_lloyd_residual_wmma(d_a, d_x, d_y, m, k, n).unwrap();
            std::env::remove_var("HIPFIRE_LLOYD_MB4");
        },
    );

    let supports_gfx11_fanout = gpu.arch_caps.has_wmma_w32();
    let phase_d_mb2 = if supports_gfx11_fanout {
        Some(bench_variant(
            gpu, m, k, n, &d_a, &d_x, &y_init, &y_ref,
            |gpu, d_a, d_x, d_y| {
                gpu.gemm_mq4g256_lloyd_residual_wmma_mb2(d_a, d_x, d_y, m, k, n).unwrap();
            },
        ))
    } else {
        None
    };

    let phase_d_mb4 = if supports_gfx11_fanout {
        Some(bench_variant(
            gpu, m, k, n, &d_a, &d_x, &y_init, &y_ref,
            |gpu, d_a, d_x, d_y| {
                gpu.gemm_mq4g256_lloyd_residual_wmma_mb4(d_a, d_x, d_y, m, k, n).unwrap();
            },
        ))
    } else {
        None
    };

    gpu.free_tensor(d_a).unwrap();
    gpu.free_tensor(d_x).unwrap();
    (phase_a, phase_d_mb2, phase_d_mb4)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let cases: &[(usize, usize, usize)] = &[
        (64,   1024,  16),
        (64,   1024,  64),
        (256,  1024,  64),
        (64,   4096,  64),
        (256,  4096,  16),
        (256,  4096, 256),
        (1024, 4096,  64),
        (1024, 12288, 64),  // qwen3.5-9b mlp.down_proj K dim
        // Production prefill shapes — the regime where _mb4's 4× weight
        // reuse should pay off. These mirror the per-kernel sizes seen in
        // the gfx1151 9B prefill profile (devlog 2026-05-09).
        (4096, 4096, 256),
        (4096, 12288, 256),
        (14336, 4096, 256),  // 9B-Lloyd FFN gate/up output dim
    ];

    // Phase A starting tolerance: 1.75e-4 = 3× MQ3 Phase A's observed max-abs
    // (5.83e-5 at K=12288). MQ4's 16-entry codebook may produce slightly
    // tighter reconstruction noise per element than MQ3's 8-entry; the
    // K=12288 envelope is dominated by WMMA accumulation noise either way.
    // If MQ4 observes consistently smaller errors, tighten tolerance to
    // ~3× MQ4's actual max-abs in Phase B1.
    let phase_a_tolerance = 1.75e-4f32;

    let mut all_pass = true;
    let mut total_us_a = 0.0f64;
    let mut total_us_mb2 = 0.0f64;
    let mut total_us_mb4 = 0.0f64;

    println!("{:>5} {:>6} {:>4}  {:>5}  {:>11}  {:>11}  {:>10}  {}",
             "M", "K", "N", "kern", "max_abs", "rms", "us/call", "verdict");

    let emit_row = |label: &str, m: usize, k: usize, n: usize,
                    result: (f32, f32, f32, f64), ref_us: f64| -> bool {
        let (max_abs, _max_rel, rms, us_per_call) = result;
        let pass = max_abs < phase_a_tolerance;
        let tag = if pass {
            if (us_per_call - ref_us).abs() < 1.0 {
                "PASS  (ref)".to_string()
            } else {
                let speedup = ref_us / us_per_call;
                if speedup >= 1.0 {
                    format!("PASS  ({:.2}× faster)", speedup)
                } else {
                    format!("PASS  ({:.2}× slower)", 1.0 / speedup)
                }
            }
        } else { "FAIL".to_string() };
        println!(
            "{:>5} {:>6} {:>4}  {:>5}  {:>11.3e}  {:>11.3e}  {:>10.1}  {tag}",
            m, k, n, label, max_abs, rms, us_per_call
        );
        pass
    };
    let emit_skip_row = |label: &str, m: usize, k: usize, n: usize| {
        println!(
            "{:>5} {:>6} {:>4}  {:>5}  {:>11}  {:>11}  {:>10}  SKIP  (gfx11-only)",
            m, k, n, label, "-", "-", "-"
        );
    };

    for &(m, k, n) in cases {
        let (phase_a, phase_d_mb2, phase_d_mb4) = run_one(&mut gpu, m, k, n);
        let ref_us = phase_a.3;
        all_pass &= emit_row("_wmma", m, k, n, phase_a, ref_us);
        if let Some(phase_d_mb2) = phase_d_mb2 {
            all_pass &= emit_row("_mb2", m, k, n, phase_d_mb2, ref_us);
            total_us_mb2 += phase_d_mb2.3;
        } else {
            emit_skip_row("_mb2", m, k, n);
        }
        if let Some(phase_d_mb4) = phase_d_mb4 {
            all_pass &= emit_row("_mb4", m, k, n, phase_d_mb4, ref_us);
            total_us_mb4 += phase_d_mb4.3;
        } else {
            emit_skip_row("_mb4", m, k, n);
        }
        println!();
        total_us_a += phase_a.3;
    }
    println!("Phase A tolerance (initial)      : {:.3e}", phase_a_tolerance);
    println!("Aggregate us/call (_wmma)        : {:.1}", total_us_a);
    if total_us_mb2 > 0.0 {
        println!("Aggregate us/call (_mb2)         : {:.1}  (vs _wmma: {:.2}×)",
                 total_us_mb2, total_us_a / total_us_mb2);
    } else {
        println!("Aggregate us/call (_mb2)         : SKIP  (gfx11-only)");
    }
    if total_us_mb4 > 0.0 {
        println!("Aggregate us/call (_mb4)         : {:.1}  (vs _wmma: {:.2}×)",
                 total_us_mb4, total_us_a / total_us_mb4);
    } else {
        println!("Aggregate us/call (_mb4)         : SKIP  (gfx11-only)");
    }

    if !all_pass {
        eprintln!("\nFAIL: one or more shapes exceeded {:.3e} absolute", phase_a_tolerance);
        std::process::exit(1);
    }
    println!("\nALL PASS");
}
