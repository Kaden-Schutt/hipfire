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

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    println!("GPU initialized: {}", gpu.arch);

    let b = 1usize;
    let n_heads = 2usize;
    let n_kv_heads = 1usize;

    let l_values = [1usize, 127, 128, 13_951, 13_952, 13_953, 16_384];
    let hd_values = [64usize, 128, 256, 512];
    let tol = 1.0e-3f32;

    let mut total = 0;
    let mut failed = 0;
    let mut max_err_seen = 0.0f32;

    println!(
        "matrix: B={b} n_heads={n_heads} n_kv_heads={n_kv_heads} (rep={})",
        n_heads / n_kv_heads
    );
    println!("tolerance: max-abs-diff < {tol:.0e}\n");
    println!("{:>5}  {:>3}  {:>7}  {:>11}  {:>4}", "L", "hd", "n_tiles", "max_diff", "stat");
    println!("{}", "-".repeat(40));

    for &l in &l_values {
        for &hd in &hd_values {
            total += 1;
            let n_tiles = compute_n_tiles(l, hd);

            let q = lcg_data(0xa5a5_a5a5 ^ ((l as u32).wrapping_mul(31)), b * n_heads * hd);
            let k = lcg_data(0xc3c3_c3c3 ^ ((l as u32).wrapping_mul(17)), l * n_kv_heads * hd);
            let v = lcg_data(0x9696_9696 ^ ((l as u32).wrapping_mul(13)), l * n_kv_heads * hd);

            let out_ref = cpu_attention_ref(&q, &k, &v, b, l, n_heads, n_kv_heads, hd);

            let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
            let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
            let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();
            let d_out = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();

            gpu.attention_dflash_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd)
                .unwrap();

            let out_gpu = gpu.download_f32(&d_out).unwrap();

            let mut max_abs_diff = 0.0f32;
            for i in 0..out_ref.len() {
                let diff = (out_gpu[i] - out_ref[i]).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }
            max_err_seen = max_err_seen.max(max_abs_diff);
            let pass = max_abs_diff < tol;
            if !pass {
                failed += 1;
            }
            println!(
                "{:>5}  {:>3}  {:>7}  {:>11.3e}  {}",
                l,
                hd,
                n_tiles,
                max_abs_diff,
                if pass { "PASS" } else { "FAIL" }
            );

            gpu.free_tensor(d_q).unwrap();
            gpu.free_tensor(d_k).unwrap();
            gpu.free_tensor(d_v).unwrap();
            gpu.free_tensor(d_out).unwrap();
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
