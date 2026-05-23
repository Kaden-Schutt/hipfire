// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test for `attention_dflash_f32` against a CPU naive softmax reference.
//!
//! Sweeps L × head_dim per the reviewer matrix on PR #222:
//!   L        ∈ {1, 127, 128, 13951, 13952, 13953, 16384}
//!   head_dim ∈ {64, 128, 256, 512}
//!
//! The boundary cases at L = 13951..13953 cover the single-tile/multi-tile
//! transition: tile_size for head_dim=128 is 13952, so n_tiles=1 at L=13952
//! and n_tiles=2 at L=13953. head_dim=512 forces nthreads(=256) < head_dim,
//! exercising the strided V-accumulation in Phase C.
//!
//! Tolerance is max-abs-diff < 1e-3. Inputs are bounded in [-0.1, 0.1) via a
//! deterministic LCG so accumulated FP error stays well below tolerance even
//! at L=16384.

use rdna_compute::{DType, Gpu};

fn lcg_data(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = (s >> 16) & 0x7fff;
            (u as f32 / 32_768.0 - 0.5) * 0.2
        })
        .collect()
}

fn cpu_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    l: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rep = n_heads / n_kv_heads;
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let mut out = vec![0.0f32; b * n_heads * head_dim];
    let mut scores = vec![0.0f32; l];

    for qi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / rep;
            let q_off = qi * q_stride + h * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for j in 0..l {
                let k_off = j * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                let s = dot * scale;
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for j in 0..l {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }
            let inv_sum = 1.0f32 / sum_exp;

            let out_off = qi * q_stride + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..l {
                    let v_off = j * kv_stride + kv_h * head_dim;
                    acc += scores[j] * v[v_off + d];
                }
                out[out_off + d] = acc * inv_sum;
            }
        }
    }
    out
}

fn compute_n_tiles(l: usize, head_dim: usize) -> usize {
    let block_size = std::cmp::min(256, std::cmp::max(l, head_dim));
    let block_size = (block_size as u32).next_power_of_two() as usize;
    const LDS_BUDGET_F32: usize = 14_336;
    let fixed = block_size + head_dim;
    let max_tile_room = LDS_BUDGET_F32.saturating_sub(fixed).max(1);
    let tile_size = std::cmp::min(l.max(1), max_tile_room);
    (l + tile_size - 1) / tile_size.max(1)
}

fn run_case(
    gpu: &mut Gpu,
    kernel: &str,
    b: usize, l: usize, n_heads: usize, n_kv_heads: usize, hd: usize,
    out_ref: &[f32],
) -> f32 {
    let q = lcg_data(0xa5a5_a5a5 ^ ((l as u32).wrapping_mul(31)), b * n_heads * hd);
    let k = lcg_data(0xc3c3_c3c3 ^ ((l as u32).wrapping_mul(17)), l * n_kv_heads * hd);
    let v = lcg_data(0x9696_9696 ^ ((l as u32).wrapping_mul(13)), l * n_kv_heads * hd);

    let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();
    let d_out = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();

    match kernel {
        "scalar" => gpu
            .attention_dflash_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd)
            .unwrap(),
        "wmma" => gpu
            .attention_dflash_wmma_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd)
            .unwrap(),
        "wmma_m32" => gpu
            .attention_dflash_wmma_m32_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd)
            .unwrap(),
        "wmma_n64" => gpu
            .attention_dflash_wmma_n64_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd)
            .unwrap(),
        "wmma_n64_f16kv" => {
            // Cast K and V to f16 first, then attention.
            let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
            gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();
            gpu.attention_dflash_wmma_n64_f16kv_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out,
                b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
            gpu.free_tensor(d_k_f16).unwrap();
            gpu.free_tensor(d_v_f16).unwrap();
        }
        "wmma_n128_f16kv" => {
            let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
            gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();
            gpu.attention_dflash_wmma_n128_f16kv_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out,
                b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
            gpu.free_tensor(d_k_f16).unwrap();
            gpu.free_tensor(d_v_f16).unwrap();
        }
        "wmma_m64_n128_f16kv" => {
            let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
            gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();
            gpu.attention_dflash_wmma_m64_n128_f16kv_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out,
                b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
            gpu.free_tensor(d_k_f16).unwrap();
            gpu.free_tensor(d_v_f16).unwrap();
        }
        "wmma_m64_n128_v2" => {
            let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
            gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();
            gpu.attention_dflash_wmma_m64_n128_f16kv_v2_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out,
                b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
            gpu.free_tensor(d_k_f16).unwrap();
            gpu.free_tensor(d_v_f16).unwrap();
        }
        "wmma_m64_n128_v3" => {
            let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
            gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
            gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();
            gpu.attention_dflash_wmma_m64_n128_f16kv_v3_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out,
                b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
            gpu.free_tensor(d_k_f16).unwrap();
            gpu.free_tensor(d_v_f16).unwrap();
        }
        _ => unreachable!(),
    }

    let out_gpu = gpu.download_f32(&d_out).unwrap();
    let max_abs_diff = out_ref
        .iter()
        .zip(out_gpu.iter())
        .map(|(r, g)| (g - r).abs())
        .fold(0.0f32, f32::max);

    gpu.free_tensor(d_q).unwrap();
    gpu.free_tensor(d_k).unwrap();
    gpu.free_tensor(d_v).unwrap();
    gpu.free_tensor(d_out).unwrap();
    max_abs_diff
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    println!("GPU initialized: {}", gpu.arch);

    // Existing scalar coverage: B=1 sweep across small + large L.
    // New WMMA coverage: same plus B=16, 17, 32 to exercise B-tiling
    // (single tile, partial second tile, exact two tiles).
    let l_values = [1usize, 127, 128, 13_951, 13_952, 13_953, 16_384];
    let hd_values = [64usize, 128, 256, 512];
    let b_values: &[(usize, &str)] = &[
        (1,  "scalar+wmma"),
        (16, "wmma_only"),
        (17, "wmma_only"),
        (32, "wmma_only"),
    ];
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let tol = 1.0e-3f32;

    let mut total = 0;
    let mut failed = 0;
    let mut max_err_seen = 0.0f32;

    println!("tolerance: max-abs-diff < {tol:.0e}");
    println!("kernels:   scalar = attention_dflash_f32   wmma = attention_dflash_wmma_f32   wmma_m32 = attention_dflash_wmma_m32_f32 (hd<=128 only)   wmma_n64 = attention_dflash_wmma_n64_f32 (hd==128 only)   wmma_n64_f16kv = attention_dflash_wmma_n64_f16kv_f32 (hd==128 only, K/V cast to f16)");
    println!();
    println!(
        "{:>3}  {:>5}  {:>3}  {:>6}  {:>11}  {:>11}  {:>4}",
        "B", "L", "hd", "kernel", "max_diff", "vs_scalar", "stat"
    );
    println!("{}", "-".repeat(70));

    for &(b, mode) in b_values {
        for &l in &l_values {
            for &hd in &hd_values {
                let q = lcg_data(0xa5a5_a5a5 ^ ((l as u32).wrapping_mul(31)), b * n_heads * hd);
                let k = lcg_data(0xc3c3_c3c3 ^ ((l as u32).wrapping_mul(17)), l * n_kv_heads * hd);
                let v = lcg_data(0x9696_9696 ^ ((l as u32).wrapping_mul(13)), l * n_kv_heads * hd);
                let out_ref = cpu_attention_ref(&q, &k, &v, b, l, n_heads, n_kv_heads, hd);

                let run_scalar = mode.contains("scalar");
                let scalar_diff = if run_scalar {
                    let d = run_case(&mut gpu, "scalar", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(d);
                    if d >= tol { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>6}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "scalar", d, "—",
                        if d < tol { "PASS" } else { "FAIL" }
                    );
                    Some(d)
                } else { None };

                // WMMA kernel caps at head_dim <= 256 (LDS budget). Skip
                // larger head_dim — those stay on the scalar path.
                if hd <= 256 {
                    let wmma_diff = run_case(&mut gpu, "wmma", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(wmma_diff);
                    if wmma_diff >= tol { failed += 1; }
                    let vs = match scalar_diff {
                        Some(_) => format!("{:.2e}", wmma_diff),
                        None => "—".into(),
                    };
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>6}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma", wmma_diff, vs,
                        if wmma_diff < tol { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    // hd>256, wmma_only mode — nothing to test.
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>6}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma", "—", "—", "SKIP (hd>256)"
                    );
                }

                // M=32 WMMA kernel caps at head_dim <= 128 (tighter LDS
                // budget than the M=16 variant). Skip larger.
                if hd <= 128 {
                    let m32_diff = run_case(&mut gpu, "wmma_m32", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(m32_diff);
                    if m32_diff >= tol { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>8}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_m32", m32_diff, "—",
                        if m32_diff < tol { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>8}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_m32", "—", "—", "SKIP (hd>128)"
                    );
                }

                // N=64 WMMA kernel is hard-coded to head_dim==128 so the
                // dc loop unrolls with d_chunks=8 and Q_frags promotes to
                // registers (v1 with runtime d_chunks regressed +19%
                // because Q_frags lived in 544 B/lane scratch instead).
                if hd == 128 {
                    let n64_diff = run_case(&mut gpu, "wmma_n64", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(n64_diff);
                    if n64_diff >= tol { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>8}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_n64", n64_diff, "—",
                        if n64_diff < tol { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>8}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_n64", "—", "—", "SKIP (hd!=128)"
                    );
                }

                // N=64 f16-K/V variant. K/V cast to fp16 before
                // attention; output stays F32. Tolerance must allow
                // for the f16 precision loss on K and V (≈ 5e-3 worst
                // case at moderate input magnitudes).
                if hd == 128 {
                    let n64_f16kv_diff = run_case(&mut gpu, "wmma_n64_f16kv", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(n64_f16kv_diff);
                    // f16 K/V introduces a 1/2048 relative quantisation
                    // on inputs in [-0.1, 0.1] (LCG range), so allow up
                    // to 5e-3 absolute diff for this variant only.
                    let tol_f16 = 5.0e-3f32;
                    if n64_f16kv_diff >= tol_f16 { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>14}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_n64_f16kv", n64_f16kv_diff, "—",
                        if n64_f16kv_diff < tol_f16 { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>14}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_n64_f16kv", "—", "—", "SKIP (hd!=128)"
                    );
                }

                // N=128 f16-K/V variant. Same tolerance band as N=64
                // f16-K/V — softmax intermediate is f16-LDS but full
                // softmax math runs in f32 per row before write-back.
                if hd == 128 {
                    let n128_f16kv_diff = run_case(&mut gpu, "wmma_n128_f16kv", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(n128_f16kv_diff);
                    let tol_f16 = 5.0e-3f32;
                    if n128_f16kv_diff >= tol_f16 { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>15}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_n128_f16kv", n128_f16kv_diff, "—",
                        if n128_f16kv_diff < tol_f16 { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>15}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_n128_f16kv", "—", "—", "SKIP (hd!=128)"
                    );
                }

                // M=64 N=128 f16-K/V variant (O register-resident).
                if hd == 128 {
                    let m64_diff = run_case(&mut gpu, "wmma_m64_n128_f16kv", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(m64_diff);
                    let tol_f16 = 5.0e-3f32;
                    if m64_diff >= tol_f16 { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>19}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_f16kv", m64_diff, "—",
                        if m64_diff < tol_f16 { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>19}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_f16kv", "—", "—", "SKIP (hd!=128)"
                    );
                }

                // M=64 N=128 v2 — padded S_lds + cooperative softmax.
                if hd == 128 {
                    let v2_diff = run_case(&mut gpu, "wmma_m64_n128_v2", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(v2_diff);
                    let tol_f16 = 5.0e-3f32;
                    if v2_diff >= tol_f16 { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>17}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_v2", v2_diff, "—",
                        if v2_diff < tol_f16 { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>17}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_v2", "—", "—", "SKIP (hd!=128)"
                    );
                }

                // M=64 N=128 v3 — hoisted S_lds reads in phase C.
                if hd == 128 {
                    let v3_diff = run_case(&mut gpu, "wmma_m64_n128_v3", b, l, n_heads, n_kv_heads, hd, &out_ref);
                    total += 1;
                    max_err_seen = max_err_seen.max(v3_diff);
                    let tol_f16 = 5.0e-3f32;
                    if v3_diff >= tol_f16 { failed += 1; }
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>17}  {:>11.3e}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_v3", v3_diff, "—",
                        if v3_diff < tol_f16 { "PASS" } else { "FAIL" }
                    );
                } else if !run_scalar {
                    println!(
                        "{:>3}  {:>5}  {:>3}  {:>17}  {:>11}  {:>11}  {}",
                        b, l, hd, "wmma_m64_n128_v3", "—", "—", "SKIP (hd!=128)"
                    );
                }
            }
        }
    }

    println!();
    println!("=== Summary ===");
    println!(
        "{} cases, {} failed, max-abs-diff seen: {:.3e} (tolerance {:.0e})",
        total, failed, max_err_seen, tol
    );
    if failed > 0 {
        std::process::exit(1);
    }
}
