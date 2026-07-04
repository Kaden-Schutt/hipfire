#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! HFQ6-G256 fused QKVZA channel test.
//! Compares WMMA variants against the FP16-packed fused substrate.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_hfq6_qkvza_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}");
        std::process::exit(0);
    }

    let shapes: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        (64, 32, 16, 16, 256, "tiny"),
        (512, 256, 16, 16, 512, "medium"),
        (4096, 1024, 16, 16, 4096, "9B LA"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (qkv_m, z_m, beta_m, alpha_m, k, label) in shapes {
        eprintln!("\n--- {label} ---");
        let w_qkv = build_hfq6g256(qkv_m, k, 0x11);
        let w_z = build_hfq6g256(z_m, k, 0x22);
        let w_beta = build_hfq6g256(beta_m, k, 0x33);
        let w_alpha = build_hfq6g256(alpha_m, k, 0x44);

        let d_qkv = gpu.upload_raw(&w_qkv, &[w_qkv.len()]).unwrap();
        let d_z = gpu.upload_raw(&w_z, &[w_z.len()]).unwrap();
        let d_beta = gpu.upload_raw(&w_beta, &[w_beta.len()]).unwrap();
        let d_alpha = gpu.upload_raw(&w_alpha, &[w_alpha.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_y_qkv_ref = gpu.zeros(&[max_n * qkv_m], DType::F32).unwrap();
        let d_y_z_ref = gpu.zeros(&[max_n * z_m], DType::F32).unwrap();
        let d_y_beta_ref = gpu.zeros(&[max_n * beta_m], DType::F32).unwrap();
        let d_y_alpha_ref = gpu.zeros(&[max_n * alpha_m], DType::F32).unwrap();
        let d_y_qkv_w = gpu.zeros(&[max_n * qkv_m], DType::F32).unwrap();
        let d_y_z_w = gpu.zeros(&[max_n * z_m], DType::F32).unwrap();
        let d_y_beta_w = gpu.zeros(&[max_n * beta_m], DType::F32).unwrap();
        let d_y_alpha_w = gpu.zeros(&[max_n * alpha_m], DType::F32).unwrap();
        let d_y_qkv_4w = gpu.zeros(&[max_n * qkv_m], DType::F32).unwrap();
        let d_y_z_4w = gpu.zeros(&[max_n * z_m], DType::F32).unwrap();
        let d_y_beta_4w = gpu.zeros(&[max_n * beta_m], DType::F32).unwrap();
        let d_y_alpha_4w = gpu.zeros(&[max_n * alpha_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let rq = d_y_qkv_ref.sub_offset(0, n * qkv_m);
            let rz = d_y_z_ref.sub_offset(0, n * z_m);
            let rb = d_y_beta_ref.sub_offset(0, n * beta_m);
            let ra = d_y_alpha_ref.sub_offset(0, n * alpha_m);
            let wq = d_y_qkv_w.sub_offset(0, n * qkv_m);
            let wz = d_y_z_w.sub_offset(0, n * z_m);
            let wb = d_y_beta_w.sub_offset(0, n * beta_m);
            let wa = d_y_alpha_w.sub_offset(0, n * alpha_m);
            let q4 = d_y_qkv_4w.sub_offset(0, n * qkv_m);
            let z4 = d_y_z_4w.sub_offset(0, n * z_m);
            let b4 = d_y_beta_4w.sub_offset(0, n * beta_m);
            let a4 = d_y_alpha_4w.sub_offset(0, n * alpha_m);

            gpu.gemm_qkvza_hfq6g256_fp16(
                &d_qkv, &d_z, &d_beta, &d_alpha, &x_n, &rq, &rz, &rb, &ra, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            )
            .unwrap();

            if arch.starts_with("gfx12") {
                gpu.gemm_qkvza_hfq6g256_wmma_gfx12(
                    &d_qkv, &d_z, &d_beta, &d_alpha, &x_n, &wq, &wz, &wb, &wa, qkv_m, z_m, beta_m,
                    alpha_m, k, n,
                )
                .unwrap();
            } else {
                gpu.gemm_qkvza_hfq6g256_wmma(
                    &d_qkv, &d_z, &d_beta, &d_alpha, &x_n, &wq, &wz, &wb, &wa, qkv_m, z_m, beta_m,
                    alpha_m, k, n,
                )
                .unwrap();
            }

            let s = [
                compare(
                    &gpu.download_f32(&wq).unwrap(),
                    &gpu.download_f32(&rq).unwrap(),
                ),
                compare(
                    &gpu.download_f32(&wz).unwrap(),
                    &gpu.download_f32(&rz).unwrap(),
                ),
                compare(
                    &gpu.download_f32(&wb).unwrap(),
                    &gpu.download_f32(&rb).unwrap(),
                ),
                compare(
                    &gpu.download_f32(&wa).unwrap(),
                    &gpu.download_f32(&ra).unwrap(),
                ),
            ];
            let pass = s.iter().all(|x| x.mean_rel < 4e-3 && x.max_rel < 9e-2);
            let mark = if pass {
                "PASS"
            } else {
                total_fail += 1;
                "FAIL"
            };
            eprintln!(
                "  N={n:4}  {mark}   QKV: {:.2e}/{:.2e}  Z: {:.2e}/{:.2e}  beta: {:.2e}/{:.2e}  alpha: {:.2e}/{:.2e}",
                s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel,
                s[2].mean_rel, s[2].max_rel, s[3].mean_rel, s[3].max_rel,
            );

            if arch == "gfx1151" && n % 64 == 0 {
                gpu.gemm_qkvza_hfq6g256_wmma_4w_gfx1151(
                    &d_qkv, &d_z, &d_beta, &d_alpha, &x_n, &q4, &z4, &b4, &a4, qkv_m, z_m, beta_m,
                    alpha_m, k, n,
                )
                .unwrap();
                let s4 = [
                    compare(
                        &gpu.download_f32(&q4).unwrap(),
                        &gpu.download_f32(&rq).unwrap(),
                    ),
                    compare(
                        &gpu.download_f32(&z4).unwrap(),
                        &gpu.download_f32(&rz).unwrap(),
                    ),
                    compare(
                        &gpu.download_f32(&b4).unwrap(),
                        &gpu.download_f32(&rb).unwrap(),
                    ),
                    compare(
                        &gpu.download_f32(&a4).unwrap(),
                        &gpu.download_f32(&ra).unwrap(),
                    ),
                ];
                let pass4 = s4.iter().all(|x| x.mean_rel < 4e-3 && x.max_rel < 9e-2);
                let mark4 = if pass4 {
                    "PASS"
                } else {
                    total_fail += 1;
                    "FAIL"
                };
                eprintln!(
                    "          4w {mark4} QKV: {:.2e}/{:.2e}  Z: {:.2e}/{:.2e}  beta: {:.2e}/{:.2e}  alpha: {:.2e}/{:.2e}",
                    s4[0].mean_rel, s4[0].max_rel, s4[1].mean_rel, s4[1].max_rel,
                    s4[2].mean_rel, s4[2].max_rel, s4[3].mean_rel, s4[3].max_rel,
                );
            }
        }
    }

    eprintln!("\n=== {total_fail} failure(s) ===");
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

struct Stats {
    mean_rel: f64,
    max_rel: f64,
}

fn compare(a: &[f32], b: &[f32]) -> Stats {
    let max_ref = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut n) = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b.iter()) {
        if y.abs() > thr {
            let r = ((x - y).abs() / y.abs()) as f64;
            sum += r;
            if r > max_r {
                max_r = r;
            }
            n += 1;
        }
    }
    Stats {
        mean_rel: if n == 0 { 0.0 } else { sum / n as f64 },
        max_rel: max_r,
    }
}

fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}

fn build_hfq6g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 200;
    let mut out = vec![0u8; m * bytes_per_row];

    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;

    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 200;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.005 + (((r1 as u32) % 1500) as f32) * 1e-5;
            let zero = (((r2 as u32) % 12000) as f32) * 1e-4 - 0.6;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());

            let mut vals = [0u8; 256];
            for (i, slot) in vals.iter_mut().enumerate() {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (i as u64));
                *slot = (r & 0x3f) as u8;
            }
            for chunk in 0..64 {
                let v0 = vals[chunk * 4] as u32;
                let v1 = vals[chunk * 4 + 1] as u32;
                let v2 = vals[chunk * 4 + 2] as u32;
                let v3 = vals[chunk * 4 + 3] as u32;
                let bits = v0 | (v1 << 6) | (v2 << 12) | (v3 << 18);
                out[off + 8 + chunk * 3] = (bits & 0xff) as u8;
                out[off + 8 + chunk * 3 + 1] = ((bits >> 8) & 0xff) as u8;
                out[off + 8 + chunk * 3 + 2] = ((bits >> 16) & 0xff) as u8;
            }
        }
    }
    out
}
