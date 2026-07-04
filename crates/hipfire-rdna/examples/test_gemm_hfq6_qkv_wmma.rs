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

//! HFQ6-G256 fused QKV channel test.
//! Compares WMMA variants against the FP16-packed fused substrate.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_hfq6_qkv_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}");
        std::process::exit(0);
    }

    let shapes: Vec<(usize, usize, usize, usize, &str)> = vec![
        (64, 32, 32, 256, "tiny"),
        (512, 128, 128, 512, "medium"),
        (4096, 1024, 1024, 4096, "9B FA"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (q_m, k_m, v_m, k, label) in shapes {
        eprintln!("\n--- {label} ---");
        let w_q = build_hfq6g256(q_m, k, 0x11);
        let w_k = build_hfq6g256(k_m, k, 0x22);
        let w_v = build_hfq6g256(v_m, k, 0x33);

        let d_q = gpu.upload_raw(&w_q, &[w_q.len()]).unwrap();
        let d_k = gpu.upload_raw(&w_k, &[w_k.len()]).unwrap();
        let d_v = gpu.upload_raw(&w_v, &[w_v.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_y_q_ref = gpu.zeros(&[max_n * q_m], DType::F32).unwrap();
        let d_y_k_ref = gpu.zeros(&[max_n * k_m], DType::F32).unwrap();
        let d_y_v_ref = gpu.zeros(&[max_n * v_m], DType::F32).unwrap();
        let d_y_q_w = gpu.zeros(&[max_n * q_m], DType::F32).unwrap();
        let d_y_k_w = gpu.zeros(&[max_n * k_m], DType::F32).unwrap();
        let d_y_v_w = gpu.zeros(&[max_n * v_m], DType::F32).unwrap();
        let d_y_q_4w = gpu.zeros(&[max_n * q_m], DType::F32).unwrap();
        let d_y_k_4w = gpu.zeros(&[max_n * k_m], DType::F32).unwrap();
        let d_y_v_4w = gpu.zeros(&[max_n * v_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let rq = d_y_q_ref.sub_offset(0, n * q_m);
            let rk = d_y_k_ref.sub_offset(0, n * k_m);
            let rv = d_y_v_ref.sub_offset(0, n * v_m);
            let wq = d_y_q_w.sub_offset(0, n * q_m);
            let wk = d_y_k_w.sub_offset(0, n * k_m);
            let wv = d_y_v_w.sub_offset(0, n * v_m);
            let q4 = d_y_q_4w.sub_offset(0, n * q_m);
            let k4 = d_y_k_4w.sub_offset(0, n * k_m);
            let v4 = d_y_v_4w.sub_offset(0, n * v_m);

            gpu.gemm_qkv_hfq6g256_fp16(&d_q, &d_k, &d_v, &x_n, &rq, &rk, &rv, q_m, k_m, v_m, k, n)
                .unwrap();

            if arch.starts_with("gfx12") {
                gpu.gemm_qkv_hfq6g256_wmma_gfx12(
                    &d_q, &d_k, &d_v, &x_n, &wq, &wk, &wv, q_m, k_m, v_m, k, n,
                )
                .unwrap();
            } else {
                gpu.gemm_qkv_hfq6g256_wmma(
                    &d_q, &d_k, &d_v, &x_n, &wq, &wk, &wv, q_m, k_m, v_m, k, n,
                )
                .unwrap();
            }

            let s = [
                compare(
                    &gpu.download_f32(&wq).unwrap(),
                    &gpu.download_f32(&rq).unwrap(),
                ),
                compare(
                    &gpu.download_f32(&wk).unwrap(),
                    &gpu.download_f32(&rk).unwrap(),
                ),
                compare(
                    &gpu.download_f32(&wv).unwrap(),
                    &gpu.download_f32(&rv).unwrap(),
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
                "  N={n:4}  {mark}   Q: {:.2e}/{:.2e}  K: {:.2e}/{:.2e}  V: {:.2e}/{:.2e}",
                s[0].mean_rel,
                s[0].max_rel,
                s[1].mean_rel,
                s[1].max_rel,
                s[2].mean_rel,
                s[2].max_rel,
            );

            if arch == "gfx1151" && n % 64 == 0 {
                gpu.gemm_qkv_hfq6g256_wmma_4w_gfx1151(
                    &d_q, &d_k, &d_v, &x_n, &q4, &k4, &v4, q_m, k_m, v_m, k, n,
                )
                .unwrap();
                let s4 = [
                    compare(
                        &gpu.download_f32(&q4).unwrap(),
                        &gpu.download_f32(&rq).unwrap(),
                    ),
                    compare(
                        &gpu.download_f32(&k4).unwrap(),
                        &gpu.download_f32(&rk).unwrap(),
                    ),
                    compare(
                        &gpu.download_f32(&v4).unwrap(),
                        &gpu.download_f32(&rv).unwrap(),
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
                    "          4w {mark4} Q: {:.2e}/{:.2e}  K: {:.2e}/{:.2e}  V: {:.2e}/{:.2e}",
                    s4[0].mean_rel,
                    s4[0].max_rel,
                    s4[1].mean_rel,
                    s4[1].max_rel,
                    s4[2].mean_rel,
                    s4[2].max_rel,
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
