// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! gfx1201 channel for the DeepSeek4 low-LDS fused RMSNorm + FWHT route.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn bytes_of_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: f32 is plain data and the byte slice does not outlive `values`.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn fixture(n: usize, seed: u64, offset: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) * 2.3e-10 + offset
        })
        .collect()
}

fn main() {
    const K: usize = 4096;
    const CASES: usize = 8;
    const WARMUPS: usize = 100;
    const TRIALS: usize = 2000;
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201", "channel requires exact gfx1201");

    let mut plain_exact = 0usize;
    let mut rotate_exact = 0usize;
    let mut baseline_us = 0.0;
    let mut candidate_us = 0.0;
    for case in 0..CASES {
        let x_host = fixture(K, 0x1201_1000 ^ case as u64, -0.5);
        let weight_host = fixture(K, 0x1201_2000 ^ case as u64, 0.75);
        let x = gpu.upload_f32(&x_host, &[K]).expect("upload x");
        let weight = gpu.upload_f32(&weight_host, &[K]).expect("upload weight");
        let base_rot = gpu.alloc_tensor(&[K], DType::F32).expect("base rot");
        let base_plain = gpu.alloc_tensor(&[K], DType::F32).expect("base plain");
        let cand_rot = gpu.alloc_tensor(&[K], DType::F32).expect("candidate rot");
        let cand_plain = gpu.alloc_tensor(&[K], DType::F32).expect("candidate plain");

        gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
            &x,
            &weight,
            &base_rot,
            &base_plain,
            K,
            1.0e-6,
            false,
        )
        .expect("baseline");
        gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
            &x,
            &weight,
            &cand_rot,
            &cand_plain,
            K,
            1.0e-6,
            true,
        )
        .expect("candidate");
        gpu.hip.device_synchronize().expect("parity synchronize");

        let mut bp = vec![0.0f32; K];
        let mut cp = vec![0.0f32; K];
        let mut br = vec![0.0f32; K];
        let mut cr = vec![0.0f32; K];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut bp), &base_plain.buf)
            .expect("download base plain");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut cp), &cand_plain.buf)
            .expect("download candidate plain");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut br), &base_rot.buf)
            .expect("download base rotate");
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut cr), &cand_rot.buf)
            .expect("download candidate rotate");
        plain_exact += bp
            .iter()
            .zip(&cp)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();
        rotate_exact += br
            .iter()
            .zip(&cr)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();

        if case == 0 {
            for _ in 0..WARMUPS {
                gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
                    &x,
                    &weight,
                    &base_rot,
                    &base_plain,
                    K,
                    1.0e-6,
                    false,
                )
                .expect("warm baseline");
                gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
                    &x,
                    &weight,
                    &cand_rot,
                    &cand_plain,
                    K,
                    1.0e-6,
                    true,
                )
                .expect("warm candidate");
            }
            gpu.hip.device_synchronize().expect("warm synchronize");
            let start = Instant::now();
            for _ in 0..TRIALS {
                gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
                    &x,
                    &weight,
                    &base_rot,
                    &base_plain,
                    K,
                    1.0e-6,
                    false,
                )
                .expect("timed baseline");
            }
            gpu.hip.device_synchronize().expect("baseline synchronize");
            baseline_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
            let start = Instant::now();
            for _ in 0..TRIALS {
                gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
                    &x,
                    &weight,
                    &cand_rot,
                    &cand_plain,
                    K,
                    1.0e-6,
                    true,
                )
                .expect("timed candidate");
            }
            gpu.hip.device_synchronize().expect("candidate synchronize");
            candidate_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
        }
    }

    let total = CASES * K;
    eprintln!(
        "gfx1201 DS4 rmsnorm+rotate: plain {plain_exact}/{total} raw-bit exact; rotate {rotate_exact}/{total} raw-bit exact; baseline {baseline_us:.3} us; nox {candidate_us:.3} us; speedup {:.3}x",
        baseline_us / candidate_us,
    );
}
