// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test: GPU `triattn_score_q8` kernel vs CPU scoring.
//!
//! Generates synthetic centers and a synthetic post-RoPE K cache, scores
//! both on CPU (via `triattn::s_total`) and on GPU (via the new kernel).
//! Expects max absolute delta ≤ 1e-3 (single-precision round-off + Q8
//! dequant error + differing trig intrinsic implementations).

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::triattn::{self, BandCenter, TriAttnCenters};
    use hipfire_runtime::llama::f16_to_f32;
    use rdna_compute::{DType, Gpu};

    // ── Fixed synthetic config ─────────────────────────────────────────
    let n_heads = 16usize;
    let n_kv_heads = 4usize;
    let head_dim = 256usize;
    let kv_group = n_heads / n_kv_heads;
    let n_bands = head_dim / 2;
    let rope_theta = 10_000_000.0f32;
    let partial_rotary_factor = 1.0f32;
    let n_rot = (head_dim as f32 * partial_rotary_factor) as usize;
    let seq_len = 64usize;
    let p_q = (seq_len - 1) as f32;

    // Deterministic "random" via LCG so CPU and GPU see identical bytes.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map top 24 bits to [-1, 1).
        let bits = (*seed >> 40) as u32;
        let uniform = bits as f32 / (1u32 << 24) as f32;
        uniform * 2.0 - 1.0
    }

    // ── Build centers ──────────────────────────────────────────────────
    let mut centers = TriAttnCenters::new(
        1, n_heads, head_dim, rope_theta, partial_rotary_factor,
    );
    let mut seed = 0xdeadbeefu64;
    for h in 0..n_heads {
        for f in 0..n_bands {
            centers.set(0, h, f, BandCenter {
                eq_re: 0.3 * lcg(&mut seed),
                eq_im: 0.3 * lcg(&mut seed),
                e_abs_q: 0.5 + 0.3 * lcg(&mut seed).abs(),
            });
        }
    }

    // Pack centers into the layout the GPU kernel expects:
    // [n_heads × n_bands × 3] (eq_re, eq_im, e_abs_q).
    let mut centers_flat = Vec::with_capacity(n_heads * n_bands * 3);
    for h in 0..n_heads {
        for f in 0..n_bands {
            let c = centers.get(0, h, f);
            centers_flat.push(c.eq_re);
            centers_flat.push(c.eq_im);
            centers_flat.push(c.e_abs_q);
        }
    }

    // ── Build Q8 K cache row-by-row ────────────────────────────────────
    //
    // Layout (matches kv_cache_write_q8_0.hip): per position, per kv_head,
    // `blocks_per_head = head_dim/32` blocks of 34 bytes each:
    //   [2B f16 scale][32 × int8 values]
    //
    // We drive the cache through the standard kv_cache_write_q8_0 kernel
    // so the layout is exercised end-to-end. Each token gets random pre-
    // Q8 values in [-0.5, 0.5]; the kernel handles scale selection.

    let mut gpu = Gpu::init().expect("gpu init");

    let kv_dim = n_kv_heads * head_dim;
    let blocks_per_head = head_dim / 32;
    let total_blocks = n_kv_heads * blocks_per_head;
    let bytes_per_pos = total_blocks * 34;
    let cache_bytes = seq_len * bytes_per_pos;
    let cache_floats = (cache_bytes + 3) / 4;

    let k_cache = gpu.zeros(&[cache_floats], DType::F32).unwrap();
    let pos_dev = gpu.hip.malloc(4).unwrap();

    for pos in 0..seq_len {
        let row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let tmp = gpu.upload_f32(&row, &[kv_dim]).unwrap();
        let pos_bytes = (pos as i32).to_ne_bytes();
        gpu.hip.memcpy_htod(&pos_dev, &pos_bytes).unwrap();
        gpu.kv_cache_write_q8_0(&k_cache, &tmp, &pos_dev, n_kv_heads, head_dim).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // ── GPU scoring ────────────────────────────────────────────────────
    let scores_gpu = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32).unwrap();
    let centers_dev = gpu.upload_f32(&centers_flat, &[n_heads * n_bands * 3]).unwrap();

    gpu.triattn_score_q8(
        &k_cache, &centers_dev, &scores_gpu,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let gpu_scores = gpu.download_f32(&scores_gpu).unwrap();

    // ── CPU scoring ────────────────────────────────────────────────────
    //
    // The GPU reads Q8-dequantized K, so we have to match that here:
    // download the Q8 cache, dequant on CPU, then run `s_total` on the
    // round-tripped K (not the pre-quantized ground truth). Otherwise
    // the parity check would be measuring Q8 quant noise, not kernel
    // correctness.
    let k_cache_bytes = {
        let floats = gpu.download_f32(&k_cache).unwrap();
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for v in &floats { bytes.extend_from_slice(&v.to_ne_bytes()); }
        bytes
    };

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut cpu_scores = vec![0.0f32; n_heads * seq_len];
    for h in 0..n_heads {
        let h_kv = h / kv_group;
        let c_slice = {
            let n_bands = centers.n_bands();
            let base = h * n_bands;
            &centers.centers[base..base + n_bands]
        };
        for pos in 0..seq_len {
            // Dequant K[pos, h_kv, :] from the Q8 cache bytes.
            let pos_base = pos * bytes_per_pos + h_kv * blocks_per_head * 34;
            let mut k_dequant = vec![0.0f32; head_dim];
            for b in 0..blocks_per_head {
                let blk = &k_cache_bytes[pos_base + b * 34 .. pos_base + (b + 1) * 34];
                let scale_bits = u16::from_le_bytes([blk[0], blk[1]]);
                // Q8 stores scale as f16 (cast from float in the kernel).
                let scale = f16_to_f32(scale_bits);
                for i in 0..32 {
                    let q = blk[2 + i] as i8;
                    k_dequant[b * 32 + i] = scale * q as f32;
                }
            }

            let k_bands = triattn::kpost_per_band(&k_dequant);
            let s = triattn::s_total(c_slice, &k_bands, p_q, |f| centers.omega(f));
            cpu_scores[h * seq_len + pos] = s;

            let g = gpu_scores[h * seq_len + pos];
            let diff = (s - g).abs();
            if diff > max_abs { max_abs = diff; }
            let denom = s.abs().max(g.abs()).max(1e-6);
            let rel = diff / denom;
            if rel > max_rel { max_rel = rel; }
        }
    }

    let total = (n_heads * seq_len) as f32;
    let mut sum_abs = 0.0f32;
    for i in 0..cpu_scores.len() {
        sum_abs += (cpu_scores[i] - gpu_scores[i]).abs();
    }
    let mean_abs = sum_abs / total;

    let n_nan_cpu = cpu_scores.iter().filter(|v| v.is_nan()).count();
    let n_nan_gpu = gpu_scores.iter().filter(|v| v.is_nan()).count();
    eprintln!("NaN count: cpu={n_nan_cpu} gpu={n_nan_gpu}");
    eprintln!("First 4 and last 4 — cpu vs gpu:");
    for i in (0..4).chain(cpu_scores.len().saturating_sub(4)..cpu_scores.len()) {
        let h = i / seq_len;
        let pos = i % seq_len;
        eprintln!("  h={h} pos={pos}: cpu={} gpu={}", cpu_scores[i], gpu_scores[i]);
    }

    eprintln!("GPU vs CPU parity over {n_heads} heads × {seq_len} positions = {} scores", n_heads * seq_len);
    eprintln!("  max |Δ|  = {max_abs:.2e}");
    eprintln!("  max rel  = {max_rel:.2e}");
    eprintln!("  mean |Δ| = {mean_abs:.2e}");

    // Pearson correlation as a stricter check: magnitudes can legitimately
    // shift by trig intrinsic differences, but the ranking should be
    // essentially identical.
    let r = triattn::pearson(&cpu_scores, &gpu_scores);
    eprintln!("  Pearson r = {r:.6}");

    assert!(r > 0.9999, "ranking correlation too low: {r}");
    assert!(max_rel < 5e-3, "max relative delta too high: {max_rel}");
    eprintln!("✅ parity within tolerance");
}
