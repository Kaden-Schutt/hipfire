//! Test: parse a GGUF file, print model config and tensor info.

use engine::gguf::GgufFile;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string()
        });

    println!("Opening: {path}");
    let gguf = GgufFile::open(Path::new(&path)).expect("failed to parse GGUF");

    println!("GGUF version: {}", gguf.version);
    println!("Tensors: {}", gguf.tensors.len());
    println!("Tensor data offset: 0x{:x}", gguf.tensor_data_offset);
    println!();

    // Print key metadata
    println!("=== Key Metadata ===");
    let keys = [
        "general.architecture",
        "general.name",
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.feed_forward_length",
        "llama.context_length",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.vocab_size",
    ];
    for key in &keys {
        if let Some(val) = gguf.meta(key) {
            println!("  {key}: {val:?}");
        }
    }
    println!();

    // Count tensor types
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for t in &gguf.tensors {
        *type_counts.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    println!("=== Tensor Types ===");
    for (dtype, count) in &type_counts {
        println!("  {dtype}: {count} tensors");
    }
    println!();

    // Print first 20 tensors
    println!("=== Tensors (first 20) ===");
    for t in gguf.tensors.iter().take(20) {
        println!(
            "  {:40} {:?} {:>12} bytes  {:?}",
            t.name,
            t.dtype,
            t.byte_size(),
            t.shape
        );
    }
    if gguf.tensors.len() > 20 {
        println!("  ... and {} more", gguf.tensors.len() - 20);
    }

    // Try to get LLaMA config
    println!();
    if let Some(config) = engine::llama::LlamaConfig::from_gguf(&gguf) {
        println!("=== LLaMA Config ===");
        println!("  dim: {}", config.dim);
        println!("  hidden_dim: {}", config.hidden_dim);
        println!("  n_layers: {}", config.n_layers);
        println!("  n_heads: {}", config.n_heads);
        println!("  n_kv_heads: {}", config.n_kv_heads);
        println!("  vocab_size: {}", config.vocab_size);
        println!("  head_dim: {}", config.head_dim);
        println!("  norm_eps: {}", config.norm_eps);
        println!("  max_seq_len: {}", config.max_seq_len);

        // Estimate VRAM needed if dequantized to F32
        let total_params: usize = gguf.tensors.iter().map(|t| t.numel()).sum();
        let f32_bytes = total_params * 4;
        println!("\n  Total params: {total_params} ({:.1}M)", total_params as f64 / 1e6);
        println!("  F32 VRAM needed: {:.1} GB", f32_bytes as f64 / 1e9);
    }
}
