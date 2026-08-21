// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Numerical probe for the gfx1100 AWQ wavegrid RMSNorm kernel: runs the
//! enabled AWQ-norm dispatch (HIPFIRE_AWQ_NORM_WAVEGRID=0/1 selects the arm)
//! on deterministic input and dumps x_rot to the given path for a bitwise
//! cross-arm diff.
//!
//! Usage: probe_awq_wavegrid K OUT.bin

use rdna_compute::{DType, Gpu};

fn main() {
    let k: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2560);
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/x_rot.bin".into());

    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!(
        "GPU: {} K={k} wavegrid={:?}",
        gpu.arch,
        std::env::var("HIPFIRE_AWQ_NORM_WAVEGRID").as_deref()
    );

    // Deterministic input in a realistic magnitude range.
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f64 * 0.017).sin() * 0.7 + (i as f64 * 0.003).cos() * 0.2) as f32)
        .collect();
    let weight: Vec<f32> = (0..k)
        .map(|i| (0.85 + 0.3 * ((i as f64 * 0.011).sin())) as f32)
        .collect();
    let awq: Vec<f32> = (0..k)
        .map(|i| (0.9 + 0.25 * ((i as f64 * 0.007).cos()).abs()) as f32)
        .collect();

    let x_t = gpu.upload_f32(&x, &[k]).expect("upload x");
    let w_t = gpu.upload_f32(&weight, &[k]).expect("upload w");
    let a_t = gpu.upload_f32(&awq, &[k]).expect("upload awq");
    let mut x_rot = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x_rot");

    gpu.fused_rmsnorm_rotate_mq_awq(&x_t, &w_t, &a_t, &mut x_rot, k, 1e-6)
        .expect("awq norm failed");
    // Second call on evolved input — exercises the epoch reset path.
    let x2: Vec<f32> = (0..k)
        .map(|i| ((i as f64 * 0.031).sin() * 1.3) as f32)
        .collect();
    let x2_t = gpu.upload_f32(&x2, &[k]).expect("upload x2");
    gpu.fused_rmsnorm_rotate_mq_awq(&x2_t, &w_t, &a_t, &mut x_rot, k, 1e-6)
        .expect("awq norm 2 failed");
    gpu.hip.device_synchronize().expect("sync");

    let host = gpu.download_f32(&x_rot).expect("download");
    eprintln!("wrote {} floats to {out_path}", host.len());
    eprintln!(
        "first 4: {:?}  last 4: {:?}",
        &host[..4],
        &host[host.len() - 4..]
    );
    std::fs::write(&out_path, unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
    })
    .expect("write");
}
