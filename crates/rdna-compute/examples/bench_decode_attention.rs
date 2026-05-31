// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Isolated microbench for the Qwen2/dots.ocr decode attention kernels.
//!
//! Single-token query against a long F32 KV cache (the dots.ocr decode
//! shape: n_heads=12, n_kv_heads=2 GQA, head_dim=128). Tiny dispatch
//! count so it's safe under `rocprofv3 --pmc` (the full daemon's ~53k
//! dispatches crash rocprofv3's packet tracking).
//!
//! Usage:
//!   ./target/release/examples/bench_decode_attention [--seq N] [--iters N]
//!
//! Benches both `attention_flash` (split-K) and `attention_f32` (naive)
//! at the given KV length so they can be compared directly and profiled.

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| args.iter().position(|a| a == k)
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(d);
    let seq_len = argval("--seq", 5100);
    let iters = argval("--iters", 100);

    let n_heads = 12usize;
    let n_kv_heads = 2usize;
    let head_dim = 128usize;
    let max_seq = 12000usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  seq_len={seq_len}  iters={iters}  (n_heads={n_heads} kv={n_kv_heads} hd={head_dim})", gpu.arch);

    let d_q = gpu.upload_f32(&lcg(0xa5a5, q_dim), &[q_dim]).unwrap();
    let d_k = gpu.upload_f32(&lcg(0xc3c3, max_seq * kv_dim), &[max_seq * kv_dim]).unwrap();
    let d_v = gpu.upload_f32(&lcg(0x9696, max_seq * kv_dim), &[max_seq * kv_dim]).unwrap();
    let d_out = gpu.zeros(&[q_dim], DType::F32).unwrap();

    let n_chunks_max = (max_seq + 127) / 128;
    let d_part = gpu.zeros(&[n_heads * n_chunks_max * (2 + head_dim)], DType::F32).unwrap();

    let pos_i32 = (seq_len - 1) as i32;
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes()).unwrap();

    // attention_flash (split-K)
    gpu.attention_flash(&d_q, &d_k, &d_v, &d_out, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_flash(&d_q, &d_k, &d_v, &d_out, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_flash:  {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // attention_f32 (naive, grid [n_heads])
    gpu.attention_f32(&d_q, &d_k, &d_v, &d_out, &pos_buf, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_f32(&d_q, &d_k, &d_v, &d_out, &pos_buf, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_f32:    {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // attention_q8_0_kv (Q8 KV cache: 4× fewer KV bytes) — build a Q8 cache
    // from the dummy F32 KV, then bench. Same grid [n_heads]; the only diff
    // is KV byte volume, so this isolates the KV-quant lever.
    let d_kq8 = gpu.alloc_tensor(&[max_seq * kv_dim], DType::Q8_0).unwrap();
    let d_vq8 = gpu.alloc_tensor(&[max_seq * kv_dim], DType::Q8_0).unwrap();
    let pos_all: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    let pos_all_t = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&pos_all_t.buf, &pos_all).unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_kq8, &d_k, &pos_all_t, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_vq8, &d_v, &pos_all_t, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.attention_q8_0_kv(&d_q, &d_kq8, &d_vq8, &d_out, &pos_buf, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_q8_0_kv(&d_q, &d_kq8, &d_vq8, &d_out, &pos_buf, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_q8_0_kv:{:.1} us/call (Q8 KV)", t.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // attention_flash_gqa (one K/V load per kv_head, reused across group)
    let d_out2 = gpu.zeros(&[q_dim], DType::F32).unwrap();
    gpu.attention_flash(&d_q, &d_k, &d_v, &d_out, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.attention_flash_gqa(&d_q, &d_k, &d_v, &d_out2, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let a = gpu.download_f32(&d_out).unwrap();
    let b = gpu.download_f32(&d_out2).unwrap();
    let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_flash_gqa(&d_q, &d_k, &d_v, &d_out2, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_flash_gqa:{:.1} us/call (vs flash maxdiff={maxdiff:.2e})", t.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // attention_gqa_warp (warp-cooperative GQA, chunked partials + reduce)
    let d_out4 = gpu.zeros(&[q_dim], DType::F32).unwrap();
    gpu.attention_gqa_warp(&d_q, &d_k, &d_v, &d_out4, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let d = gpu.download_f32(&d_out4).unwrap();
    let maxdiff4 = a.iter().zip(&d).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_gqa_warp(&d_q, &d_k, &d_v, &d_out4, &d_part, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_gqa_warp:{:.1} us/call (vs flash maxdiff={maxdiff4:.2e})", t.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // attention_gqa_warp_dv: same math, seq_len read from a device pointer
    // for hipGraph capture paths.
    let d_out5 = gpu.zeros(&[q_dim], DType::F32).unwrap();
    let seq_i32 = seq_len as i32;
    let seq_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&seq_buf, &seq_i32.to_ne_bytes()).unwrap();
    let chunk_size = 128usize;
    let n_chunks = (seq_len + chunk_size - 1) / chunk_size;
    gpu.attention_gqa_warp_dv(&d_q, &d_k, &d_v, &d_out5, &d_part, &seq_buf, n_heads, n_kv_heads, head_dim, max_seq, chunk_size, n_chunks).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let e = gpu.download_f32(&d_out5).unwrap();
    let maxdiff5 = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("attention_gqa_warp_dv: smoke PASS (vs flash maxdiff={maxdiff5:.2e})");

    // attention_flash_gqa_fused (single launch, no partials/reduce)
    let d_out3 = gpu.zeros(&[q_dim], DType::F32).unwrap();
    gpu.attention_flash_gqa_fused(&d_q, &d_k, &d_v, &d_out3, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let c = gpu.download_f32(&d_out3).unwrap();
    let maxdiff3 = a.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_flash_gqa_fused(&d_q, &d_k, &d_v, &d_out3, seq_len, n_heads, n_kv_heads, head_dim, max_seq).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("attention_flash_gqa_fused:{:.1} us/call (vs flash maxdiff={maxdiff3:.2e})", t.elapsed().as_secs_f64() * 1e6 / iters as f64);
}
