// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-parity test for `fused_qkvza_q8_0` (4-way) and `fused_qkv_q8_0` (3-way)
//! decode (n=1) kernels vs four / three sequential `gemv_q8_0` calls.
//!
//! Pass criteria (mirrors fused_gate_up_q8_0 threshold):
//!   mean_rel < 2e-3  AND  max_rel < 3.5e-2 for every output tensor.
//!
//! Run:
//!   cargo run --release --example test_fused_q8_0_decode -p rdna-compute

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_fused_q8_0_decode ===\n  arch = {arch}");

    // ── Realistic Qwen3.5-A3B .mq4p DeltaNet LA preamble shapes ──
    // 0.6B: K=1024, qkv_m=1024, z_m=512, beta_m=16, alpha_m=16
    // 9B  : K=4096, qkv_m=4096, z_m=1024, beta_m=16, alpha_m=16
    let shapes_4way: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        (64, 32, 16, 16, 128, "tiny"),
        (1024, 512, 16, 16, 1024, "0.6B LA (qkv=1024 z=512 K=1024)"),
        (4096, 1024, 16, 16, 4096, "9B LA (qkv=4096 z=1024 K=4096)"),
    ];

    // ── Full-attention QKV shapes ──
    let shapes_3way: Vec<(usize, usize, usize, usize, &str)> = vec![
        (64, 32, 32, 128, "tiny"),
        (1024, 256, 256, 1024, "0.6B FA (q=1024 k=256 v=256 K=1024)"),
        (
            4096,
            1024,
            1024,
            4096,
            "9B FA (q=4096 k=1024 v=1024 K=4096)",
        ),
    ];

    let mut total_fail = 0usize;

    // ── 4-way QKVZA parity ──────────────────────────────────────────
    eprintln!("\n=== 4-way QKVZA (fused_qkvza_q8_0 vs 4× gemv_q8_0) ===");
    for (qkv_m, z_m, beta_m, alpha_m, k, label) in &shapes_4way {
        let (qkv_m, z_m, beta_m, alpha_m, k) = (*qkv_m, *z_m, *beta_m, *alpha_m, *k);
        assert!(k % 32 == 0, "K must be a multiple of 32 for Q8_0");
        eprintln!("\n--- {label} ---");

        let w_qkv_host = synth_q8(qkv_m, k, 0xA1B2C3D4);
        let w_z_host = synth_q8(z_m, k, 0xE5F60718);
        let w_beta_host = synth_q8(beta_m, k, 0x9ABCDEF0);
        let w_alpha_host = synth_q8(alpha_m, k, 0x12345678);

        let d_w_qkv = gpu.upload_raw(&w_qkv_host, &[w_qkv_host.len()]).unwrap();
        let d_w_z = gpu.upload_raw(&w_z_host, &[w_z_host.len()]).unwrap();
        let d_w_beta = gpu.upload_raw(&w_beta_host, &[w_beta_host.len()]).unwrap();
        let d_w_alpha = gpu
            .upload_raw(&w_alpha_host, &[w_alpha_host.len()])
            .unwrap();

        let x_host: Vec<f32> = (0..k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[k]).unwrap();

        // Fused kernel outputs.
        let d_fq = gpu.zeros(&[qkv_m], DType::F32).unwrap();
        let d_fz = gpu.zeros(&[z_m], DType::F32).unwrap();
        let d_fb = gpu.zeros(&[beta_m], DType::F32).unwrap();
        let d_fa = gpu.zeros(&[alpha_m], DType::F32).unwrap();

        // Reference: 4 sequential gemv_q8_0 calls.
        let d_rq = gpu.zeros(&[qkv_m], DType::F32).unwrap();
        let d_rz = gpu.zeros(&[z_m], DType::F32).unwrap();
        let d_rb = gpu.zeros(&[beta_m], DType::F32).unwrap();
        let d_ra = gpu.zeros(&[alpha_m], DType::F32).unwrap();

        gpu.fused_qkvza_q8_0(
            &d_w_qkv, &d_w_z, &d_w_beta, &d_w_alpha, &d_x, &d_fq, &d_fz, &d_fb, &d_fa, qkv_m, z_m,
            beta_m, alpha_m, k,
        )
        .unwrap();

        gpu.gemv_q8_0(&d_w_qkv, &d_x, &d_rq, qkv_m, k).unwrap();
        gpu.gemv_q8_0(&d_w_z, &d_x, &d_rz, z_m, k).unwrap();
        gpu.gemv_q8_0(&d_w_beta, &d_x, &d_rb, beta_m, k).unwrap();
        gpu.gemv_q8_0(&d_w_alpha, &d_x, &d_ra, alpha_m, k).unwrap();

        let s = [
            compare(
                "qkv",
                &gpu.download_f32(&d_fq).unwrap(),
                &gpu.download_f32(&d_rq).unwrap(),
            ),
            compare(
                "z",
                &gpu.download_f32(&d_fz).unwrap(),
                &gpu.download_f32(&d_rz).unwrap(),
            ),
            compare(
                "beta",
                &gpu.download_f32(&d_fb).unwrap(),
                &gpu.download_f32(&d_rb).unwrap(),
            ),
            compare(
                "alpha",
                &gpu.download_f32(&d_fa).unwrap(),
                &gpu.download_f32(&d_ra).unwrap(),
            ),
        ];

        let pass = s.iter().all(|x| x.mean_rel < 2e-3 && x.max_rel < 3.5e-2);
        let mark = if pass {
            "PASS"
        } else {
            total_fail += 1;
            "FAIL"
        };
        eprintln!(
            "  {mark}  QKV: mean={:.2e}/max={:.2e}  Z: {:.2e}/{:.2e}  β: {:.2e}/{:.2e}  α: {:.2e}/{:.2e}",
            s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel,
            s[2].mean_rel, s[2].max_rel, s[3].mean_rel, s[3].max_rel,
        );
        if !pass {
            for (i, st) in s.iter().enumerate() {
                let name = ["qkv", "z", "beta", "alpha"][i];
                if st.mean_rel >= 2e-3 || st.max_rel >= 3.5e-2 {
                    eprintln!(
                        "    FAIL detail [{name}]: mean_rel={:.4e} max_rel={:.4e}",
                        st.mean_rel, st.max_rel
                    );
                }
            }
        }
    }

    // ── 3-way QKV parity ────────────────────────────────────────────
    eprintln!("\n=== 3-way QKV (fused_qkv_q8_0 vs 3× gemv_q8_0) ===");
    for (q_m, k_m, v_m, k, label) in &shapes_3way {
        let (q_m, k_m, v_m, k) = (*q_m, *k_m, *v_m, *k);
        assert!(k % 32 == 0, "K must be a multiple of 32 for Q8_0");
        eprintln!("\n--- {label} ---");

        let w_q_host = synth_q8(q_m, k, 0xDEADBEEF);
        let w_k_host = synth_q8(k_m, k, 0xC0FFEE00);
        let w_v_host = synth_q8(v_m, k, 0xFEEDFACE);

        let d_wq = gpu.upload_raw(&w_q_host, &[w_q_host.len()]).unwrap();
        let d_wk = gpu.upload_raw(&w_k_host, &[w_k_host.len()]).unwrap();
        let d_wv = gpu.upload_raw(&w_v_host, &[w_v_host.len()]).unwrap();

        let x_host: Vec<f32> = (0..k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[k]).unwrap();

        let d_fq = gpu.zeros(&[q_m], DType::F32).unwrap();
        let d_fk = gpu.zeros(&[k_m], DType::F32).unwrap();
        let d_fv = gpu.zeros(&[v_m], DType::F32).unwrap();

        let d_rq = gpu.zeros(&[q_m], DType::F32).unwrap();
        let d_rk = gpu.zeros(&[k_m], DType::F32).unwrap();
        let d_rv = gpu.zeros(&[v_m], DType::F32).unwrap();

        gpu.fused_qkv_q8_0(
            &d_wq, &d_wk, &d_wv, &d_x, &d_fq, &d_fk, &d_fv, q_m, k_m, v_m, k,
        )
        .unwrap();

        gpu.gemv_q8_0(&d_wq, &d_x, &d_rq, q_m, k).unwrap();
        gpu.gemv_q8_0(&d_wk, &d_x, &d_rk, k_m, k).unwrap();
        gpu.gemv_q8_0(&d_wv, &d_x, &d_rv, v_m, k).unwrap();

        let s = [
            compare(
                "q",
                &gpu.download_f32(&d_fq).unwrap(),
                &gpu.download_f32(&d_rq).unwrap(),
            ),
            compare(
                "k",
                &gpu.download_f32(&d_fk).unwrap(),
                &gpu.download_f32(&d_rk).unwrap(),
            ),
            compare(
                "v",
                &gpu.download_f32(&d_fv).unwrap(),
                &gpu.download_f32(&d_rv).unwrap(),
            ),
        ];

        let pass = s.iter().all(|x| x.mean_rel < 2e-3 && x.max_rel < 3.5e-2);
        let mark = if pass {
            "PASS"
        } else {
            total_fail += 1;
            "FAIL"
        };
        eprintln!(
            "  {mark}  Q: mean={:.2e}/max={:.2e}  K: {:.2e}/{:.2e}  V: {:.2e}/{:.2e}",
            s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel, s[2].mean_rel, s[2].max_rel,
        );
        if !pass {
            for (i, st) in s.iter().enumerate() {
                let name = ["q", "k", "v"][i];
                if st.mean_rel >= 2e-3 || st.max_rel >= 3.5e-2 {
                    eprintln!(
                        "    FAIL detail [{name}]: mean_rel={:.4e} max_rel={:.4e}",
                        st.mean_rel, st.max_rel
                    );
                }
            }
        }
    }

    eprintln!(
        "\n=== {total_fail} failure(s) — {} ===",
        if total_fail == 0 {
            "ALL PASS"
        } else {
            "FAILURES"
        }
    );
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

struct Stats {
    mean_rel: f64,
    max_rel: f64,
}

fn compare(label: &str, fused: &[f32], reference: &[f32]) -> Stats {
    assert_eq!(fused.len(), reference.len(), "{label}: length mismatch");
    let max_ref = reference.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut n) = (0.0f64, 0.0f64, 0usize);
    for (f, r) in fused.iter().zip(reference.iter()) {
        if r.abs() > thr {
            let rel = ((f - r).abs() / r.abs()) as f64;
            sum += rel;
            if rel > max_r {
                max_r = rel;
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
