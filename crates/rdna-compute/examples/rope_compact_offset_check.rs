// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// H4 verification: the batched RoPE `pos_offset` param (carries
// kv_cache.compact_offset) must rotate at ABSOLUTE phase = positions[b] + offset.
// Proves, without needing a CASK-eviction setup:
//   T1 (offset applied / equivalence): batched(positions=[K..K+B], offset=0)
//        == batched(positions=[0..B], offset=K)   — both rotate at K..K+B.
//   T2 (matches the absolute-phase reference): batched(B=1, positions=[0], offset=K)
//        == per-token rope at pos=K  — i.e. the fix makes batched Q/K rotate at
//        the same absolute phase the cached (per-token-written) keys carry.
//
// Run: cargo run --release --example rope_compact_offset_check   (gfx10xx ok; no model)

use rdna_compute::{DType, Gpu};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() {
    let b = 4usize;
    let n_heads_q = 4usize;
    let n_heads_k = 2usize;
    let head_dim = 64usize;
    let n_rot = 64usize;
    let freq_base = 1.0e6f32;
    let k_off = 137i32; // stand-in for a post-eviction compact_offset

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  (b={b} nhq={n_heads_q} nhk={n_heads_k} hd={head_dim} n_rot={n_rot} K={k_off})", gpu.arch);

    let q_src = lcg(0xA5A5, b * n_heads_q * head_dim);
    let k_src = lcg(0xC3C3, b * n_heads_k * head_dim);

    // positions tensor helper (i32 bytes into a 4-byte/elem tensor, as the kernel reads int*).
    let pos_tensor = |gpu: &mut Gpu, vals: &[i32]| {
        let t = gpu.alloc_tensor(&[vals.len()], DType::F32).unwrap();
        let bytes: Vec<u8> = vals.iter().flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap();
        t
    };

    // ---- T1: offset applied == shifting positions ----
    // Run A: positions = [K, K+1, K+2, K+3], offset = 0
    let qa = gpu.upload_f32(&q_src, &[q_src.len()]).unwrap();
    let ka = gpu.upload_f32(&k_src, &[k_src.len()]).unwrap();
    let pos_a = pos_tensor(&mut gpu, &(0..b as i32).map(|i| k_off + i).collect::<Vec<_>>());
    gpu.rope_partial_interleaved_f32_batched(&qa, &ka, &pos_a, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, b, 0).unwrap();
    let qa_out = gpu.download_f32(&qa).unwrap();
    let ka_out = gpu.download_f32(&ka).unwrap();

    // Run B: positions = [0,1,2,3], offset = K
    let qb = gpu.upload_f32(&q_src, &[q_src.len()]).unwrap();
    let kb = gpu.upload_f32(&k_src, &[k_src.len()]).unwrap();
    let pos_b = pos_tensor(&mut gpu, &(0..b as i32).collect::<Vec<_>>());
    gpu.rope_partial_interleaved_f32_batched(&qb, &kb, &pos_b, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, b, k_off).unwrap();
    let qb_out = gpu.download_f32(&qb).unwrap();
    let kb_out = gpu.download_f32(&kb).unwrap();

    let t1q = max_abs_diff(&qa_out, &qb_out);
    let t1k = max_abs_diff(&ka_out, &kb_out);
    eprintln!("T1 offset-equivalence  max|dq|={t1q:.3e}  max|dk|={t1k:.3e}");

    // ---- T2: batched(B=1, pos=[0], offset=K) == per-token(pos=K) ----
    let q1 = q_src[..n_heads_q * head_dim].to_vec();
    let k1 = k_src[..n_heads_k * head_dim].to_vec();

    // per-token reference at absolute pos = K (what a cached key carries)
    let qt = gpu.upload_f32(&q1, &[q1.len()]).unwrap();
    let kt = gpu.upload_f32(&k1, &[k1.len()]).unwrap();
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &k_off.to_ne_bytes()).unwrap();
    gpu.rope_partial_interleaved_f32(&qt, &kt, &pos_buf, n_heads_q, n_heads_k, head_dim, n_rot, freq_base).unwrap();
    let qt_out = gpu.download_f32(&qt).unwrap();
    let kt_out = gpu.download_f32(&kt).unwrap();

    // batched B=1, positions=[0], offset=K  -> rotates at pos 0+K = K
    let qc = gpu.upload_f32(&q1, &[q1.len()]).unwrap();
    let kc = gpu.upload_f32(&k1, &[k1.len()]).unwrap();
    let pos_c = pos_tensor(&mut gpu, &[0]);
    gpu.rope_partial_interleaved_f32_batched(&qc, &kc, &pos_c, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, 1, k_off).unwrap();
    let qc_out = gpu.download_f32(&qc).unwrap();
    let kc_out = gpu.download_f32(&kc).unwrap();

    let t2q = max_abs_diff(&qt_out, &qc_out);
    let t2k = max_abs_diff(&kt_out, &kc_out);
    eprintln!("T2 batched+offset==per-token@K  max|dq|={t2q:.3e}  max|dk|={t2k:.3e}");

    let eps = 1e-4f32;
    let pass = t1q < eps && t1k < eps && t2q < eps && t2k < eps;
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
