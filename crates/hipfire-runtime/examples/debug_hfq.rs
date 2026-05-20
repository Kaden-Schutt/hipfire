// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Quick debug: load HFQ, run one forward pass, print logits stats
fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    
    // Load HFQ
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-q4f16.hfq")).unwrap();
    let config = hipfire_runtime::hfq::config_from_hfq(&hfq).unwrap();
    let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu).unwrap();
    
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv = hipfire_runtime::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap();
    
    // Forward pass with token 1 (BOS)
    let logits = hipfire_runtime::llama::forward(&mut gpu, &weights, &config, 1, 0, &mut kv).unwrap();
    
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let top = hipfire_runtime::llama::argmax(&logits);
    println!("HFQ logits: min={min:.4} max={max:.4} mean={mean:.6} argmax={top}");
    println!("  first 10: {:?}", &logits[..10]);
    
    // Now compare with GGUF
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")).unwrap();
    let config2 = hipfire_runtime::llama::LlamaConfig::from_gguf(&gguf).unwrap();
    let weights2 = hipfire_runtime::llama::load_weights(&gguf, &config2, &mut gpu).unwrap();
    let mut kv2 = hipfire_runtime::llama::KvCache::new_gpu(
        &mut gpu, config2.n_layers, config2.n_kv_heads, config2.head_dim, kv_seq_len).unwrap();
    
    let logits2 = hipfire_runtime::llama::forward(&mut gpu, &weights2, &config2, 1, 0, &mut kv2).unwrap();
    let min2 = logits2.iter().cloned().fold(f32::INFINITY, f32::min);
    let max2 = logits2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean2 = logits2.iter().sum::<f32>() / logits2.len() as f32;
    let top2 = hipfire_runtime::llama::argmax(&logits2);
    println!("GGUF logits: min={min2:.4} max={max2:.4} mean={mean2:.6} argmax={top2}");
    println!("  first 10: {:?}", &logits2[..10]);
}
