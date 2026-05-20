// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    
    // HFQ Q8_FP16 path
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-q8f16.hfq")).unwrap();
    let config = hipfire_runtime::hfq::config_from_hfq(&hfq).unwrap();
    let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu).unwrap();
    
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv = hipfire_runtime::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap();
    
    let logits = hipfire_runtime::llama::forward(&mut gpu, &weights, &config, 1, 0, &mut kv).unwrap();
    let top = hipfire_runtime::llama::argmax(&logits);
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("HFQ Q8: argmax={top} min={min:.4} max={max:.4}");
    println!("  first 10: {:?}", &logits[..10]);
    
    // GGUF Q8_0 path 
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")).unwrap();
    let config2 = hipfire_runtime::llama::LlamaConfig::from_gguf(&gguf).unwrap();
    let weights2 = hipfire_runtime::llama::load_weights(&gguf, &config2, &mut gpu).unwrap();
    let mut kv2 = hipfire_runtime::llama::KvCache::new_gpu(
        &mut gpu, config2.n_layers, config2.n_kv_heads, config2.head_dim, kv_seq_len).unwrap();
    
    let logits2 = hipfire_runtime::llama::forward(&mut gpu, &weights2, &config2, 1, 0, &mut kv2).unwrap();
    let top2 = hipfire_runtime::llama::argmax(&logits2);
    let min2 = logits2.iter().cloned().fold(f32::INFINITY, f32::min);
    let max2 = logits2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("GGUF Q8: argmax={top2} min={min2:.4} max={max2:.4}");
    println!("  first 10: {:?}", &logits2[..10]);
    
    // Also GGUF Q4_K_M
    let gguf3 = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")).unwrap();
    let config3 = hipfire_runtime::llama::LlamaConfig::from_gguf(&gguf3).unwrap();
    let weights3 = hipfire_runtime::llama::load_weights(&gguf3, &config3, &mut gpu).unwrap();
    let mut kv3 = hipfire_runtime::llama::KvCache::new_gpu(
        &mut gpu, config3.n_layers, config3.n_kv_heads, config3.head_dim, kv_seq_len).unwrap();
    
    let logits3 = hipfire_runtime::llama::forward(&mut gpu, &weights3, &config3, 1, 0, &mut kv3).unwrap();
    let top3 = hipfire_runtime::llama::argmax(&logits3);
    let min3 = logits3.iter().cloned().fold(f32::INFINITY, f32::min);
    let max3 = logits3.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("GGUF Q4: argmax={top3} min={min3:.4} max={max3:.4}");
    println!("  first 10: {:?}", &logits3[..10]);
}
