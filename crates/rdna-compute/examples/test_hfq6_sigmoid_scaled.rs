//! Correctness test for `gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched`.
//!
//! Synthesizes random HFQ6 weights (matching the producer layout used by
//! `quantize_hfq6g256` in `hipfire-quantize`: 200 B / group, 4 B scale +
//! 4 B zero + 192 B packed 6-bit nibbles, 4 weights per 3 bytes), random
//! X (per-token rows), random gate scalars c_batch, and a non-zero
//! initial Y. Computes the same operation on the CPU and compares.
//!
//! The kernel does:
//!   for bid in 0..N:
//!     for row in 0..M:
//!       acc = A[row] · x_batch[bid]            (HFQ6 dequant)
//!       y_batch[bid, row] += sigmoid(c_batch[bid]) * acc
//!
//! No atomic collisions (each (bid, row) is unique), so a 1e-2 abs tol
//! is generous — the only source of slop is f32 reduction order inside
//! the warp shuffle vs sequential CPU sum.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example test_hfq6_sigmoid_scaled \
//!     -- [M] [K] [N]
//!
//! Defaults: M=128, K=512, N=4. K must be a multiple of 256.

use rdna_compute::Gpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    assert!(k % 256 == 0, "K must be a multiple of 256 (HFQ6 group size)");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 200;

    eprintln!("=== HFQ6 batched sigmoid_scaled GEMV correctness test ===");
    eprintln!("M={m} K={k} N={n} (groups_per_row={groups_per_row}, row_bytes={row_bytes})");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    // Synthesize HFQ6 weights matching producer layout.
    let weight_bytes = synth_hfq6g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    assert_eq!(weight_bytes.len(), m * row_bytes);

    let a_raw = gpu
        .upload_raw(&weight_bytes, &[m * row_bytes])
        .expect("upload weights");

    // Per-token X rows: deterministic seed → host-visible repro.
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let x_tensor = gpu.upload_f32(&x_host, &[n * k]).expect("upload x");

    // c_batch: scalar per token, centered around 0 (sigmoid ~0.5).
    let c_host: Vec<f32> = (0..n)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2654435761).wrapping_add(0x9E37_79B9_u32 as i64)) as f32;
            ((v * 1e-9) % 4.0) - 2.0
        })
        .collect();
    let c_tensor = gpu.upload_f32(&c_host, &[n]).expect("upload c_batch");

    // Non-zero initial Y so the residual `+=` is observable.
    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();
    let y_tensor = gpu
        .upload_f32(&y_init_host, &[n * m])
        .expect("alloc + upload y");

    // ─── GPU run ─────────────────────────────────────────────────────
    eprintln!("\n--- gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched ---");
    gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
        &a_raw, &x_tensor, &y_tensor, &c_tensor, m, k, n,
    )
    .expect("kernel launch");
    gpu.hip.device_synchronize().expect("sync");
    let gpu_out = gpu.download_f32(&y_tensor).expect("download y");

    // ─── CPU reference ───────────────────────────────────────────────
    eprintln!("--- CPU reference (full dequant + GEMV + sigmoid-scaled +=) ---");
    let cpu_out = cpu_reference(&weight_bytes, &x_host, &y_init_host, &c_host, m, k, n);

    // ─── Compare ──────────────────────────────────────────────────────
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut worst_idx = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for i in 0..n * m {
        let r = cpu_out[i];
        let q = gpu_out[i];
        let err = (r - q).abs();
        if err > max_abs_err {
            max_abs_err = err;
            worst_idx = i;
            worst_pair = (r, q);
        }
        let rel = if r.abs() > 1e-6 { err / r.abs() } else { 0.0 };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (r as f64).powi(2);
    }

    let rms_err = (sum_sq_err / (n * m) as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / (n * m) as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);

    let ref_min = cpu_out.iter().copied().fold(f32::INFINITY, f32::min);
    let ref_max = cpu_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let g_min = gpu_out.iter().copied().fold(f32::INFINITY, f32::min);
    let g_max = gpu_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let worst_bid = worst_idx / m;
    let worst_row = worst_idx % m;
    eprintln!("\nmax_abs_err  = {:.6e}", max_abs_err);
    eprintln!("max_rel_err  = {:.4}%", max_rel_err * 100.0);
    eprintln!("rms_err      = {:.6e}", rms_err);
    eprintln!("rms_ref      = {:.6e}", rms_ref);
    eprintln!("NRMSE        = {:.4}%", nrmse * 100.0);
    eprintln!("worst (bid,row) = ({worst_bid}, {worst_row})");
    eprintln!("                  cpu={:.6e}  gpu={:.6e}", worst_pair.0, worst_pair.1);
    eprintln!("cpu range: [{ref_min:.4e}, {ref_max:.4e}]");
    eprintln!("gpu range: [{g_min:.4e}, {g_max:.4e}]");

    eprintln!("\n--- First 8 output cells (bid=0, rows=0..7) ---");
    for i in 0..8.min(m) {
        eprintln!(
            "  row {i}: cpu={:.6e}  gpu={:.6e}  diff={:.6e}",
            cpu_out[i],
            gpu_out[i],
            (cpu_out[i] - gpu_out[i]).abs()
        );
    }

    // Pass criteria:
    //  - max_abs_err < 1e-2: order-of-add slop from warp shuffle vs sequential
    //    sum is the only error source (no quant noise, both paths dequant the
    //    same bytes the same way).
    //  - GPU output is non-zero and differs from y_init (kernel actually ran
    //    and the `+=` fired).
    let gpu_nonzero = gpu_out.iter().any(|&v| v.abs() > 1e-12);
    let gpu_wrote_residual = gpu_out
        .iter()
        .zip(y_init_host.iter())
        .any(|(o, init)| (o - init).abs() > 1e-6);
    let pass = max_abs_err < 1e-2 && gpu_nonzero && gpu_wrote_residual;
    if pass {
        eprintln!("\nPASS (max_abs_err < 1e-2, residual write observed)");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL");
        if !gpu_nonzero {
            eprintln!("  gpu output is all-zero — kernel may not have run");
        }
        if !gpu_wrote_residual {
            eprintln!("  gpu output matches y_init — residual `+=` did not fire");
        }
        if max_abs_err >= 1e-2 {
            eprintln!("  max_abs_err {max_abs_err:.6e} exceeds 1e-2 threshold");
        }
        std::process::exit(1);
    }
}

/// CPU reference: full HFQ6 dequant + scalar GEMV + sigmoid(c_batch) scale +
/// initial-Y `+=`. Matches the kernel byte-for-byte modulo f32 add order.
fn cpu_reference(
    weight_bytes: &[u8],
    x: &[f32],
    y_init: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 200;
    let mut out = y_init.to_vec();

    for bid in 0..n {
        let x_off = bid * k;
        let gate = 1.0f32 / (1.0f32 + (-c[bid]).exp());
        for row in 0..m {
            let row_ptr = row * row_bytes;
            let mut acc = 0.0f32;
            for g in 0..groups_per_row {
                let gp = row_ptr + g * 200;
                let scale = f32::from_le_bytes([
                    weight_bytes[gp],
                    weight_bytes[gp + 1],
                    weight_bytes[gp + 2],
                    weight_bytes[gp + 3],
                ]);
                let zero = f32::from_le_bytes([
                    weight_bytes[gp + 4],
                    weight_bytes[gp + 5],
                    weight_bytes[gp + 6],
                    weight_bytes[gp + 7],
                ]);
                let base_data = gp + 8;
                let base_x = x_off + g * 256;
                // Loop the same way the kernel does: each "thread" tid in
                // 0..32 reads 6 bytes at byte_off = tid*6 → 8 quants per
                // thread → 256 weights per group.
                for tid in 0..32 {
                    let byte_off = tid * 6;
                    let b0 = weight_bytes[base_data + byte_off];
                    let b1 = weight_bytes[base_data + byte_off + 1];
                    let b2 = weight_bytes[base_data + byte_off + 2];
                    let b3 = weight_bytes[base_data + byte_off + 3];
                    let b4 = weight_bytes[base_data + byte_off + 4];
                    let b5 = weight_bytes[base_data + byte_off + 5];

                    let q0 = (b0 & 63) as f32;
                    let q1 = ((b0 >> 6) as u32 | (((b1 & 0xF) as u32) << 2)) as f32;
                    let q2 = ((b1 >> 4) as u32 | (((b2 & 3) as u32) << 4)) as f32;
                    let q3 = (b2 >> 2) as f32;
                    let q4 = (b3 & 63) as f32;
                    let q5 = ((b3 >> 6) as u32 | (((b4 & 0xF) as u32) << 2)) as f32;
                    let q6 = ((b4 >> 4) as u32 | (((b5 & 3) as u32) << 4)) as f32;
                    let q7 = (b5 >> 2) as f32;

                    let base = base_x + tid * 8;
                    acc += (scale * q0 + zero) * x[base]
                        + (scale * q1 + zero) * x[base + 1]
                        + (scale * q2 + zero) * x[base + 2]
                        + (scale * q3 + zero) * x[base + 3]
                        + (scale * q4 + zero) * x[base + 4]
                        + (scale * q5 + zero) * x[base + 5]
                        + (scale * q6 + zero) * x[base + 6]
                        + (scale * q7 + zero) * x[base + 7];
                }
            }
            out[bid * m + row] += gate * acc;
        }
    }
    out
}

/// Synthesize random HFQ6 weights matching the producer layout used by
/// `quantize_hfq6g256` in `hipfire-quantize`:
///   per-group (200 B): [f32 scale][f32 zero][192 B packed 6-bit nibbles]
///   packing: 4 quants per 3 bytes, byte_off = (i/4)*3
///     byte0 = q0 | (q1 << 6)
///     byte1 = (q1 >> 2) | (q2 << 4)
///     byte2 = (q2 >> 4) | (q3 << 2)
fn synth_hfq6g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 200;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    // Pick mild scales/zeros — keeps dequant values in a typical range so
    // the f32 acc doesn't grow huge and amplify add-order slop.
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 200;
            let scale = 1e-3 * (0.5 + (next_u32() & 0xFFFF) as f32 / 65535.0 * 1.5);
            let zero = ((next_u32() & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.1;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zero.to_le_bytes());
            // 256 weights / 4 = 64 byte-triplets per group → 192 B packed.
            for i in (0..256).step_by(4) {
                let q0 = (next_u32() & 63) as u8;
                let q1 = (next_u32() & 63) as u8;
                let q2 = (next_u32() & 63) as u8;
                let q3 = (next_u32() & 63) as u8;
                let byte_off = 8 + (i / 4) * 3;
                out[gp + byte_off]     = q0 | (q1 << 6);
                out[gp + byte_off + 1] = (q1 >> 2) | (q2 << 4);
                out[gp + byte_off + 2] = (q2 >> 4) | (q3 << 2);
            }
        }
    }
    out
}
