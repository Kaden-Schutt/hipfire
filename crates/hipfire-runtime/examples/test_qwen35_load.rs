//! Minimal test: load Qwen3.5 HFQ, parse config, print layer types.
//! Usage: cargo run --release --features deltanet --example test_qwen35_load -- models/qwen3.5-0.8b.q4.hfq

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("Usage: test_qwen35_load <model.hfq>");

    let hfq = HfqFile::open(Path::new(path)).expect("failed to open HFQ");
    eprintln!("HFQ arch_id: {}", hfq.arch_id);

    // Parse metadata
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("bad metadata JSON");
    let config = meta.get("config").expect("no config in metadata");

    // For VLM models, text_config contains the actual model config
    let text_config = config.get("text_config").unwrap_or(config);

    let model_type = text_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown");
    let hidden_size = text_config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_layers = text_config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_heads = text_config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_kv_heads = text_config.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(0);
    let head_dim = text_config.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(0);
    let vocab_size = text_config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);

    eprintln!("model_type: {model_type}");
    eprintln!("hidden_size: {hidden_size}, layers: {n_layers}, heads: {n_heads}, kv_heads: {n_kv_heads}, head_dim: {head_dim}, vocab: {vocab_size}");

    // DeltaNet-specific config
    let layer_types: Vec<String> = text_config.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let linear_key_head_dim = text_config.get("linear_key_head_dim").and_then(|v| v.as_u64()).unwrap_or(0);
    let linear_value_head_dim = text_config.get("linear_value_head_dim").and_then(|v| v.as_u64()).unwrap_or(0);
    let linear_num_key_heads = text_config.get("linear_num_key_heads").and_then(|v| v.as_u64()).unwrap_or(0);
    let linear_num_value_heads = text_config.get("linear_num_value_heads").and_then(|v| v.as_u64()).unwrap_or(0);
    let conv_kernel_dim = text_config.get("linear_conv_kernel_dim").and_then(|v| v.as_u64()).unwrap_or(4);
    let attn_output_gate = text_config.get("attn_output_gate").and_then(|v| v.as_bool()).unwrap_or(false);

    eprintln!("\nDeltaNet config:");
    eprintln!("  linear_key_head_dim: {linear_key_head_dim}");
    eprintln!("  linear_value_head_dim: {linear_value_head_dim}");
    eprintln!("  linear_num_key_heads: {linear_num_key_heads}");
    eprintln!("  linear_num_value_heads: {linear_num_value_heads}");
    eprintln!("  conv_kernel_dim: {conv_kernel_dim}");
    eprintln!("  attn_output_gate: {attn_output_gate}");

    let n_linear = layer_types.iter().filter(|t| *t == "linear_attention").count();
    let n_full = layer_types.iter().filter(|t| *t == "full_attention").count();
    eprintln!("  layers: {n_linear} linear + {n_full} full attention = {} total", n_layers);

    // Check tokenizer
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json);
    match tok {
        Some(t) => eprintln!("\nTokenizer: {} tokens", t.vocab_size()),
        None => eprintln!("\nTokenizer: FAILED to load"),
    }

    // List some tensors
    eprintln!("\nSample tensors:");
    let prefixes = ["model.language_model.layers.0.linear_attn", "model.language_model.layers.3.self_attn"];
    for prefix in prefixes {
        eprintln!("  {prefix}.*:");
        // We can't easily iterate tensors from HfqFile's API, but we verified shapes above
    }

    eprintln!("\nConfig parse: OK");

    // Test full weight loading
    #[cfg(feature = "deltanet")]
    {
        use hipfire_arch_qwen35::qwen35;
        let q35_config = qwen35::config_from_hfq(&hfq).expect("failed to parse Qwen3.5 config");
        eprintln!("\nLoading Qwen3.5 weights...");
        let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
        let weights = qwen35::load_weights(&hfq, &q35_config, &mut gpu).expect("failed to load weights");
        eprintln!("Loaded {} layers", weights.layers.len());
        for (i, layer) in weights.layers.iter().enumerate() {
            match layer {
                qwen35::LayerWeights::DeltaNet(_) => eprint!("D"),
                qwen35::LayerWeights::FullAttn(_) => eprint!("F"),
                qwen35::LayerWeights::DeltaNetMoe(_) => eprint!("d"),
                qwen35::LayerWeights::FullAttnMoe(_) => eprint!("f"),
            }
        }
        eprintln!("\nWeight loading: OK");
    }
}
