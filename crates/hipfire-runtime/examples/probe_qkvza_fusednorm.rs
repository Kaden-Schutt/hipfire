// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Numerical probe for the qkvza consumer-fold (HIPFIRE_QKVZA_FUSEDNORM):
//! producer->consumer vs folded fused_qkvza_hfq4g256_fusednorm on identical
//! synthetic input; diffs all four outputs bitwise. K=2560 (Qwen3.5-4B shape).
//! Also emits the fold kernel's internal rms via an eps<0 sentinel tap.
//!
//! Usage: probe_qkvza_fusednorm [qkv_m] [z_m]

use rdna_compute::{DType, Gpu};

fn main() {
    let qkv_m = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);
    let z_m = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);
    let beta_m = 16usize;
    let alpha_m = 16usize;
    let k = 2560usize;
    let groups = k / 256;
    let eps = 1e-6f32;

    let mut gpu = Gpu::init().expect("GPU init failed");

    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f64 * 0.017).sin() * 1.9 + (i as f64 * 0.005).cos() * 0.7) as f32)
        .collect();
    let gamma: Vec<f32> = (0..k)
        .map(|i| (0.8f64 + 0.4 * (i as f64 * 0.011).sin()) as f32)
        .collect();
    let awq: Vec<f32> = (0..k)
        .map(|i| (0.5f64 + 1.5 * (0.5 + 0.5 * (i as f64 * 0.007).cos())) as f32)
        .collect();

    let total_m = qkv_m + z_m + beta_m + alpha_m;
    let row_bytes = groups * 136;
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut mk_weights = |m: usize| -> Vec<u8> {
        let mut b = vec![0u8; m * row_bytes];
        for r in 0..m {
            for g in 0..groups {
                let o = r * row_bytes + g * 136;
                let sc = 0.01f32 + (next() % 100) as f32 * 0.0005;
                let mn = -1.0f32 - (next() % 50) as f32 * 0.01;
                b[o..o + 4].copy_from_slice(&sc.to_le_bytes());
                b[o + 4..o + 8].copy_from_slice(&mn.to_le_bytes());
                for i in 0..128 {
                    b[o + 8 + i] = (next() & 0xFF) as u8;
                }
            }
        }
        b
    };
    let wq = mk_weights(qkv_m);
    let wz = mk_weights(z_m);
    let wb = mk_weights(beta_m);
    let wa = mk_weights(alpha_m);

    let s1v = rdna_compute::gen_fwht_signs(42, 256);
    let s2v = rdna_compute::gen_fwht_signs(1042, 256);

    let xt = gpu.upload_f32(&x, &[k]).expect("x");
    let gt = gpu.upload_f32(&gamma, &[k]).expect("gamma");
    let at = gpu.upload_f32(&awq, &[k]).expect("awq");
    let wqt = gpu.upload_raw(&wq, &[wq.len()]).expect("wq");
    let wzt = gpu.upload_raw(&wz, &[wz.len()]).expect("wz");
    let wbt = gpu.upload_raw(&wb, &[wb.len()]).expect("wb");
    let wat = gpu.upload_raw(&wa, &[wa.len()]).expect("wa");

    let x_rot = gpu.alloc_tensor(&[k], DType::F32).expect("x_rot");
    let yq_a = gpu.alloc_tensor(&[qkv_m], DType::F32).expect("yq a");
    let yz_a = gpu.alloc_tensor(&[z_m], DType::F32).expect("yz a");
    let yb_a = gpu.alloc_tensor(&[beta_m], DType::F32).expect("yb a");
    let ya_a = gpu.alloc_tensor(&[alpha_m], DType::F32).expect("ya a");
    let yq_b = gpu.alloc_tensor(&[qkv_m], DType::F32).expect("yq b");
    let yz_b = gpu.alloc_tensor(&[z_m], DType::F32).expect("yz b");
    let yb_b = gpu.alloc_tensor(&[beta_m], DType::F32).expect("yb b");
    let ya_b = gpu.alloc_tensor(&[alpha_m], DType::F32).expect("ya b");

    // ── Arm A: producer -> base consumer.
    gpu.fused_rmsnorm_rotate_mq_awq(&xt, &gt, &at, &x_rot, k, eps)
        .expect("producer");
    gpu.fused_qkvza_hfq4g256(
        &wqt, &wzt, &wbt, &wat, &x_rot, &yq_a, &yz_a, &yb_a, &ya_a, qkv_m, z_m, beta_m, alpha_m, k,
    )
    .expect("consumer A");
    gpu.hip.device_synchronize().expect("sync A");

    // ── Arm B: folded consumer (eps > 0 -> normal mode).
    gpu.fused_qkvza_hfq4g256_fusednorm(
        &wqt, &wzt, &wbt, &wat, &xt, &gt, &at, &yq_b, &yz_b, &yb_b, &ya_b, qkv_m, z_m, beta_m,
        alpha_m, k, eps,
    )
    .expect("consumer B");
    gpu.hip.device_synchronize().expect("sync B");

    // ── Compare BEFORE any debug-tap call runs.
    let names = [
        ("y_qkv", qkv_m, &yq_a, &yq_b),
        ("y_z", z_m, &yz_a, &yz_b),
        ("y_beta", beta_m, &yb_a, &yb_b),
        ("y_alpha", alpha_m, &ya_a, &ya_b),
    ];
    let mut all_ok = true;
    for (name, m, ta, tb) in names {
        let a = gpu.download_f32(ta).unwrap();
        let b = gpu.download_f32(tb).unwrap();
        let mut mism = 0usize;
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for i in 0..m {
            if a[i].to_bits() != b[i].to_bits() {
                mism += 1;
                let rel = (a[i] - b[i]).abs() / a[i].abs().max(1e-30);
                if rel > worst {
                    worst = rel;
                    worst_i = i;
                }
            }
        }
        println!(
            "{name}: {} mism={mism}/{} max_rel={:.3e} @{}",
            if mism == 0 { "BITEXACT" } else { "MISMATCH" },
            m,
            worst,
            worst_i
        );
        if mism > 0 {
            all_ok = false;
        }
    }

    // ── rms diagnostics via the eps<0 sentinel tap. Every matrix's row-0
    // block writes its own output's [0]; use throwaway buffers everywhere
    // except the one we read.
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms_host = 1.0 / (sum_sq / k as f32 + eps).sqrt();
    let dz = gpu.alloc_tensor(&[z_m], DType::F32).expect("dz");
    let db = gpu.alloc_tensor(&[beta_m], DType::F32).expect("db");
    let da = gpu.alloc_tensor(&[alpha_m], DType::F32).expect("da");
    let y_rms = gpu.alloc_tensor(&[qkv_m], DType::F32).expect("y rms");
    gpu.fused_qkvza_hfq4g256_fusednorm(
        &wqt, &wzt, &wbt, &wat, &xt, &gt, &at, &y_rms, &dz, &db, &da, qkv_m, z_m, beta_m, alpha_m,
        k, -eps,
    )
    .expect("rms tap");
    gpu.hip.device_synchronize().expect("sync rms");
    let rms_k = gpu.download_f32(&y_rms).unwrap()[0];
    let rms_host_n = 1.0 / (sum_sq / k as f32).sqrt();
    // naive discriminator: eps <= -1e-4 switches pass A to a plain strided sum.
    let dzb = gpu.alloc_tensor(&[z_m], DType::F32).expect("dzb");
    gpu.fused_qkvza_hfq4g256_fusednorm(
        &wqt, &wzt, &wbt, &wat, &xt, &gt, &at, &y_rms, &dzb, &db, &da, qkv_m, z_m, beta_m, alpha_m,
        k, -1.0,
    )
    .expect("rms naive");
    gpu.hip.device_synchronize().expect("sync naive");
    let rms_n = gpu.download_f32(&y_rms).unwrap()[0];
    println!("rms(host)={rms_host:.6e}");
    println!("rms(fold slot/tree)={rms_k:.6e} ratio={:.4}", rms_k / rms_host);
    println!(
        "rms(fold naive strided)={rms_n:.6e} ratio_vs_host={:.4}",
        rms_n / rms_host_n
    );

    if !all_ok {
        eprintln!("PROBE: FAIL - outputs differ");
        std::process::exit(1);
    }
    println!("PROBE: PASS - all outputs bitwise identical");
}
