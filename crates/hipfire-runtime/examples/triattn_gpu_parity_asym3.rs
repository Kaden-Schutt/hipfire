// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU parity test for `triattn_score_asym3`.
//!
//! Unlike the Q8_0 parity test (whose reconstruction is a straight linear
//! dequant), asym3 stacks three non-trivial transformations on top of K:
//!   1. normalize by the original L2 norm,
//!   2. per-band Givens rotation,
//!   3. Lloyd-Max 3-bit quantization,
//! and bookkeeps a per-head `cnorm` to restore magnitude. If any piece
//! of the recovery is wrong — endianness of the 3-byte packed words, the
//! Givens inverse, the codebook index math — the CPU and GPU scores
//! diverge by orders of magnitude. Passing this test means the kernel's
//! dequant + un-Givens matches exactly.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::llama::{f16_to_f32, KvCache};
    use hipfire_runtime::triattn::{self, BandCenter, TriAttnCenters};
    use rdna_compute::{DType, Gpu};

    // ── Config matching Qwen3.5 FA layer shape ─────────────────────────
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

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*seed >> 40) as u32;
        let uniform = bits as f32 / (1u32 << 24) as f32;
        uniform * 2.0 - 1.0
    }

    // ── Centers ────────────────────────────────────────────────────────
    let mut centers = TriAttnCenters::new(1, n_heads, head_dim, rope_theta, partial_rotary_factor);
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
    let mut centers_flat = Vec::with_capacity(n_heads * n_bands * 3);
    for h in 0..n_heads {
        for f in 0..n_bands {
            let c = centers.get(0, h, f);
            centers_flat.push(c.eq_re);
            centers_flat.push(c.eq_im);
            centers_flat.push(c.e_abs_q);
        }
    }

    // ── Allocate asym3 KV cache (just for its cos/sin Givens tables) ───
    let mut gpu = Gpu::init().expect("gpu init");
    // Use the standard constructor so cos_theta/sin_theta match what the
    // production kernels generate (seed=42).
    let kv = KvCache::new_gpu_asym3(&mut gpu, 1, n_kv_heads, head_dim, seq_len)
        .expect("asym3 kv cache");
    let cos_theta = kv.givens_cos.as_ref().expect("asym3 has cos table");
    let sin_theta = kv.givens_sin.as_ref().expect("asym3 has sin table");

    let k_cache = &kv.k_gpu[0];
    let v_cache = &kv.v_gpu[0]; // populated as a byproduct of the fused write
    let pos_dev = gpu.hip.malloc(4).unwrap();
    let kv_dim = n_kv_heads * head_dim;

    for pos in 0..seq_len {
        let k_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let k_tmp = gpu.upload_f32(&k_row, &[kv_dim]).unwrap();
        let v_tmp = gpu.upload_f32(&v_row, &[kv_dim]).unwrap();
        let pos_bytes = (pos as i32).to_ne_bytes();
        gpu.hip.memcpy_htod(&pos_dev, &pos_bytes).unwrap();
        gpu.kv_cache_write_asym3_fused(
            k_cache, v_cache, &k_tmp, &v_tmp, &pos_dev,
            cos_theta, sin_theta, n_kv_heads, head_dim,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // ── GPU scoring ────────────────────────────────────────────────────
    let scores_gpu = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32).unwrap();
    let centers_dev = gpu.upload_f32(&centers_flat, &[n_heads * n_bands * 3]).unwrap();
    gpu.triattn_score_asym3(
        k_cache, &centers_dev, cos_theta, sin_theta, &scores_gpu,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gpu_scores = gpu.download_f32(&scores_gpu).unwrap();

    // ── CPU reconstruction ─────────────────────────────────────────────
    let cos_vals = gpu.download_f32(cos_theta).unwrap();
    let sin_vals = gpu.download_f32(sin_theta).unwrap();
    let k_cache_bytes = {
        let floats = gpu.download_f32(k_cache).unwrap();
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for v in &floats { bytes.extend_from_slice(&v.to_ne_bytes()); }
        bytes
    };

    // Lloyd-Max codebook for N(0, 1/256) — matches turbo_common.h TURBO_C3_256.
    const LLOYD_C3_256: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176,
         0.015176,  0.046469,  0.083320,  0.134860,
    ];

    let k_bytes_per_head = 4 + (head_dim * 3) / 8;
    let k_bytes_per_pos = n_kv_heads * k_bytes_per_head;

    let mut cpu_scores = vec![0.0f32; n_heads * seq_len];
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let _ = f16_to_f32; // silence unused import warning if the helper is also elsewhere

    for h in 0..n_heads {
        let h_kv = h / kv_group;
        let centers_slice = {
            let base = h * n_bands;
            &centers.centers[base..base + n_bands]
        };
        for pos in 0..seq_len {
            let head_off = pos * k_bytes_per_pos + h_kv * k_bytes_per_head;
            let cnorm = f32::from_le_bytes([
                k_cache_bytes[head_off + 0],
                k_cache_bytes[head_off + 1],
                k_cache_bytes[head_off + 2],
                k_cache_bytes[head_off + 3],
            ]);

            // Dequant 8 values per thread × 32 threads = 256 dims.
            let mut k_post_recovered = vec![0.0f32; head_dim];
            for tid in 0..32usize {
                let base_off = head_off + 4 + tid * 3;
                let b0 = k_cache_bytes[base_off + 0] as u32;
                let b1 = k_cache_bytes[base_off + 1] as u32;
                let b2 = k_cache_bytes[base_off + 2] as u32;
                let packed = b0 | (b1 << 8) | (b2 << 16);
                let d0 = tid * 8;
                let b0_band = tid * 4;
                // Dequantize first, then un-Givens per band pair.
                let mut v = [0.0f32; 8];
                for i in 0..8 {
                    let idx = ((packed >> (i * 3)) & 7) as usize;
                    v[i] = cnorm * LLOYD_C3_256[idx];
                }
                for j in 0..4 {
                    let f = b0_band + j;
                    let cb = cos_vals[f];
                    let sb = sin_vals[f];
                    let a_gv = v[j * 2 + 0];
                    let b_gv = v[j * 2 + 1];
                    let a =  cb * a_gv + sb * b_gv;
                    let b = -sb * a_gv + cb * b_gv;
                    k_post_recovered[d0 + j * 2 + 0] = a;
                    k_post_recovered[d0 + j * 2 + 1] = b;
                }
            }

            let k_bands = triattn::kpost_per_band(&k_post_recovered);
            let s = triattn::s_total(centers_slice, &k_bands, p_q, |f| centers.omega(f));
            cpu_scores[h * seq_len + pos] = s;

            let g = gpu_scores[h * seq_len + pos];
            let diff = (s - g).abs();
            if diff > max_abs { max_abs = diff; }
            let denom = s.abs().max(g.abs()).max(1e-6);
            let rel = diff / denom;
            if rel > max_rel { max_rel = rel; }
        }
    }

    let r = triattn::pearson(&cpu_scores, &gpu_scores);
    eprintln!("asym3 GPU vs CPU parity over {} heads × {} positions = {} scores", n_heads, seq_len, n_heads * seq_len);
    eprintln!("  max |Δ|  = {max_abs:.2e}");
    eprintln!("  max rel  = {max_rel:.2e}");
    eprintln!("  Pearson r = {r:.6}");

    assert!(r > 0.9999, "asym3 score ranking correlation too low: {r}");
    assert!(max_rel < 5e-3, "asym3 max relative delta too high: {max_rel}");
    eprintln!("✅ asym3 parity within tolerance");
}
