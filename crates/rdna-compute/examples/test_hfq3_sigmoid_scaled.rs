//! Correctness test for `gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched`.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example test_hfq3_sigmoid_scaled -- [M] [K] [N]
//!
//! Defaults: M=128, K=512, N=4. K must be a multiple of 256.

use rdna_compute::Gpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    assert!(
        k % 256 == 0,
        "K must be a multiple of 256 (HFQ3 group size)"
    );
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 104;

    eprintln!("=== HFQ3 batched sigmoid_scaled GEMV correctness test ===");
    eprintln!("M={m} K={k} N={n} (groups_per_row={groups_per_row}, row_bytes={row_bytes})");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    let weight_bytes = synth_hfq3g256_weights(m, groups_per_row, 0xA11C_E123u64);
    assert_eq!(weight_bytes.len(), m * row_bytes);
    let a_raw = gpu
        .upload_raw(&weight_bytes, &[m * row_bytes])
        .expect("upload weights");

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let x_tensor = gpu.upload_f32(&x_host, &[n * k]).expect("upload x");

    let c_host: Vec<f32> = (0..n)
        .map(|i| {
            let v = ((i as i64)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9E37_79B9_u32 as i64)) as f32;
            ((v * 1e-9) % 4.0) - 2.0
        })
        .collect();
    let c_tensor = gpu.upload_f32(&c_host, &[n]).expect("upload c_batch");

    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();
    let y_tensor = gpu
        .upload_f32(&y_init_host, &[n * m])
        .expect("alloc + upload y");

    gpu.gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
        &a_raw, &x_tensor, &y_tensor, &c_tensor, m, k, n,
    )
    .expect("kernel launch");
    gpu.hip.device_synchronize().expect("sync");
    let gpu_out = gpu.download_f32(&y_tensor).expect("download y");

    let cpu_out = cpu_reference(&weight_bytes, &x_host, &y_init_host, &c_host, m, k, n);

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut worst_idx = 0usize;
    for i in 0..n * m {
        let r = cpu_out[i];
        let q = gpu_out[i];
        let err = (r - q).abs();
        if err > max_abs_err {
            max_abs_err = err;
            worst_idx = i;
        }
        let rel = if r.abs() > 1e-6 { err / r.abs() } else { 0.0 };
        max_rel_err = max_rel_err.max(rel);
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (r as f64).powi(2);
    }

    let rms_err = (sum_sq_err / (n * m) as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / (n * m) as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);
    let worst_bid = worst_idx / m;
    let worst_row = worst_idx % m;
    eprintln!("max_abs_err  = {:.6e}", max_abs_err);
    eprintln!("max_rel_err  = {:.4}%", max_rel_err * 100.0);
    eprintln!("NRMSE        = {:.4}%", nrmse * 100.0);
    eprintln!(
        "worst (bid,row)=({worst_bid},{worst_row}) cpu={:.6e} gpu={:.6e}",
        cpu_out[worst_idx], gpu_out[worst_idx]
    );

    let gpu_wrote_residual = gpu_out
        .iter()
        .zip(y_init_host.iter())
        .any(|(o, init)| (o - init).abs() > 1e-6);
    if max_abs_err < 1e-2 && gpu_wrote_residual {
        eprintln!("PASS (max_abs_err < 1e-2, residual write observed)");
    } else {
        eprintln!("FAIL");
        std::process::exit(1);
    }
}

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
    let row_bytes = groups_per_row * 104;
    let mut out = y_init.to_vec();

    for bid in 0..n {
        let x_off = bid * k;
        let gate = 1.0f32 / (1.0f32 + (-c[bid]).exp());
        for row in 0..m {
            let row_ptr = row * row_bytes;
            let mut acc = 0.0f32;
            for g in 0..groups_per_row {
                let gp = row_ptr + g * 104;
                let scale = f32::from_le_bytes(weight_bytes[gp..gp + 4].try_into().unwrap());
                let zero = f32::from_le_bytes(weight_bytes[gp + 4..gp + 8].try_into().unwrap());
                let base_data = gp + 8;
                let base_x = x_off + g * 256;
                for chunk in 0..32 {
                    let dp = base_data + chunk * 3;
                    let pk = weight_bytes[dp] as u32
                        | ((weight_bytes[dp + 1] as u32) << 8)
                        | ((weight_bytes[dp + 2] as u32) << 16);
                    for i in 0..8 {
                        let q = ((pk >> (3 * i)) & 7) as f32;
                        acc += (scale * q + zero) * x[base_x + chunk * 8 + i];
                    }
                }
            }
            out[bid * m + row] += gate * acc;
        }
    }
    out
}

fn synth_hfq3g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 104;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 104;
            let scale = 1e-3 * (0.5 + (next_u32() & 0xFFFF) as f32 / 65535.0 * 1.5);
            let zero = ((next_u32() & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.1;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zero.to_le_bytes());
            for chunk in 0..32 {
                let mut packed = 0u32;
                for i in 0..8 {
                    packed |= (next_u32() & 7) << (3 * i);
                }
                let off = gp + 8 + chunk * 3;
                out[off] = (packed & 0xFF) as u8;
                out[off + 1] = ((packed >> 8) & 0xFF) as u8;
                out[off + 2] = ((packed >> 16) & 0xFF) as u8;
            }
        }
    }
    out
}
