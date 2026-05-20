// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness check: fused_rmsnorm_rotate vs sequential rmsnorm + rotate.

fn main() {
    use rdna_compute::DType;
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    for &k in &[4096usize, 12288] {
        eprintln!("\n=== K = {k} ===");

        // Random input, weight
        let x: Vec<f32> = (0..k).map(|i| ((i * 7 + 13) % 97) as f32 / 47.0 - 0.8).collect();
        let w: Vec<f32> = (0..k).map(|i| 1.0 + ((i * 11 + 3) % 51) as f32 / 233.0).collect();

        let d_x = gpu.upload_f32(&x, &[k]).unwrap();
        let d_w = gpu.upload_f32(&w, &[k]).unwrap();
        let d_out_split = gpu.zeros(&[k], DType::F32).unwrap();
        let d_out_fused = gpu.zeros(&[k], DType::F32).unwrap();
        let d_tmp = gpu.zeros(&[k], DType::F32).unwrap();

        gpu.ensure_mq_signs().unwrap();

        // Path A: rmsnorm_f32 → rotate_x_mq
        gpu.rmsnorm_f32(&d_x, &d_w, &d_tmp, 1e-6).unwrap();
        gpu.rotate_x_mq(&d_tmp, &d_out_split, k).unwrap();

        // Path B: fused kernel
        gpu.fused_rmsnorm_rotate_mq(&d_x, &d_w, &d_out_fused, k, 1e-6).unwrap();

        let a = gpu.download_f32(&d_out_split).unwrap();
        let b = gpu.download_f32(&d_out_fused).unwrap();

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut n_finite = 0;
        for i in 0..k {
            if a[i].is_finite() && b[i].is_finite() { n_finite += 1; }
            let d = (a[i] - b[i]).abs();
            if d > max_abs { max_abs = d; }
            let denom = a[i].abs().max(1e-6);
            let r = d / denom;
            if r > max_rel { max_rel = r; }
        }
        eprintln!("  n_finite:  {n_finite}/{k}");
        eprintln!("  max_abs:   {max_abs:.6e}");
        eprintln!("  max_rel:   {max_rel:.6e}");
        eprintln!("  split[0..4]: {:?}", &a[..4]);
        eprintln!("  fused[0..4]: {:?}", &b[..4]);

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_w).unwrap();
        gpu.free_tensor(d_out_split).unwrap();
        gpu.free_tensor(d_out_fused).unwrap();
        gpu.free_tensor(d_tmp).unwrap();
    }
}
