// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Debug: compare dequantized tensor values against a simple check.

use hipfire_runtime::gguf::GgufFile;
use hipfire_runtime::llama;
use std::path::Path;

fn main() {
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = GgufFile::open(Path::new(path)).unwrap();

    // Check F32 tensor (should be exact)
    let norm_info = gguf.find_tensor("blk.0.attn_norm.weight").unwrap();
    let norm_data = gguf.tensor_data(norm_info);
    println!("=== blk.0.attn_norm.weight (F32, {} elements) ===", norm_info.numel());
    println!("  dtype: {:?}, bytes: {}", norm_info.dtype, norm_data.len());
    for i in 0..8 {
        let v = f32::from_le_bytes([
            norm_data[i*4], norm_data[i*4+1], norm_data[i*4+2], norm_data[i*4+3]
        ]);
        print!("  [{i}]={v:.6}");
    }
    println!();

    // Check Q4_K tensor - first block
    let q_info = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let q_data = gguf.tensor_data(q_info);
    println!("\n=== blk.0.attn_q.weight (Q4_K, {} elements) ===", q_info.numel());
    println!("  dtype: {:?}, raw bytes: {}, computed byte_size: {}",
        q_info.dtype, q_data.len(), q_info.byte_size());

    // Dequantize first block (256 elements)
    let deq = llama::dequantize_q4_k(q_data, 256);
    println!("  First 16 dequantized values:");
    for i in 0..16 {
        print!("  [{i}]={:.6}", deq[i]);
    }
    println!();

    // Stats
    let deq_full = llama::dequantize_q4_k(q_data, q_info.numel());
    let min = deq_full.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = deq_full.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = deq_full.iter().sum::<f32>() / deq_full.len() as f32;
    let nonzero = deq_full.iter().filter(|&&v| v.abs() > 1e-8).count();
    println!("  Full tensor stats: min={min:.6} max={max:.6} mean={mean:.8} nonzero={nonzero}/{}",
        deq_full.len());

    // Also check Q6_K
    let v_info = gguf.find_tensor("blk.0.attn_v.weight").unwrap();
    let v_data = gguf.tensor_data(v_info);
    println!("\n=== blk.0.attn_v.weight (Q6_K, {} elements) ===", v_info.numel());
    let deq_v = llama::dequantize_q6_k(v_data, 256);
    println!("  First 16 dequantized values:");
    for i in 0..16 {
        print!("  [{i}]={:.6}", deq_v[i]);
    }
    println!();

    let deq_v_full = llama::dequantize_q6_k(v_data, v_info.numel());
    let min = deq_v_full.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = deq_v_full.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = deq_v_full.iter().sum::<f32>() / deq_v_full.len() as f32;
    println!("  Stats: min={min:.6} max={max:.6} mean={mean:.8}");

    // Check token_embd
    let embd_info = gguf.find_tensor("token_embd.weight").unwrap();
    let embd_data = gguf.tensor_data(embd_info);
    println!("\n=== token_embd.weight (Q4_K, {} elements) ===", embd_info.numel());
    let deq_embd = llama::dequantize_q4_k(embd_data, embd_info.numel());
    let min = deq_embd.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = deq_embd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = deq_embd.iter().sum::<f32>() / deq_embd.len() as f32;
    println!("  Stats: min={min:.6} max={max:.6} mean={mean:.8}");

    // Check embedding for token 1 (BOS)
    let dim = 2048;
    let bos_embd: Vec<f32> = deq_embd[0..dim].to_vec();
    let hello_embd: Vec<f32> = deq_embd[15043 * dim..(15043 + 1) * dim].to_vec();
    println!("  BOS embedding first 8: {:?}", &bos_embd[..8]);
    println!("  Hello(15043) embedding first 8: {:?}", &hello_embd[..8]);
}
