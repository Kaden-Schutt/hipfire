//! Inspect a Qwen2 HFQ file: open it, parse the config, optionally
//! load weights to GPU and report the layer count. Rev-1 smoke utility
//! to verify the loader works end-to-end against real HFQ output.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example inspect_hfq -p hipfire-arch-qwen2 -- \
//!     ~/.hipfire/models/qwen2-1.5b.hfq4
//!
//! # Add --load to upload all weights to GPU (~820 MB for 1.5B-Instruct):
//! cargo run --release --example inspect_hfq -p hipfire-arch-qwen2 -- \
//!     ~/.hipfire/models/qwen2-1.5b.hfq4 --load
//! ```

use std::path::Path;

use hipfire_arch_qwen2::qwen2;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or("usage: inspect_hfq <path.hfq> [--load]")?;
    let do_load = args.iter().any(|a| a == "--load");

    let mut hfq = HfqFile::open(Path::new(path))?;
    println!("opened: {path}");
    println!("  arch_id (from HFQ header): {}", hfq.arch_id);
    println!("  metadata_json length: {} bytes", hfq.metadata_json.len());

    let cfg = qwen2::config_from_hfq(&hfq)
        .ok_or("config_from_hfq returned None")?;

    println!("\nparsed Qwen2Config:");
    println!("  hidden_size:             {}", cfg.hidden_size);
    println!("  num_hidden_layers:       {}", cfg.num_hidden_layers);
    println!("  num_attention_heads:     {}", cfg.num_attention_heads);
    println!("  num_key_value_heads:     {}", cfg.num_key_value_heads);
    println!("  head_dim:                {}", cfg.head_dim);
    println!("  intermediate_size:       {}", cfg.intermediate_size);
    println!("  vocab_size:              {}", cfg.vocab_size);
    println!("  max_position_embeddings: {}", cfg.max_position_embeddings);
    println!("  rope_theta:              {}", cfg.rope_theta);
    println!("  rms_norm_eps:            {}", cfg.rms_norm_eps);
    println!("  attention_bias:          {}", cfg.attention_bias);
    println!("  tie_word_embeddings:     {}", cfg.tie_word_embeddings);
    println!("  eos_token_id:            {}", cfg.eos_token_id);

    if do_load {
        println!("\nloading weights to GPU...");
        let mut gpu = Gpu::init()?;
        let weights = qwen2::load_weights(&mut hfq, &cfg, &mut gpu)?;
        println!("\nloaded successfully:");
        println!("  layers:        {}", weights.layers.len());
        println!("  tied_lm_head:  {}", weights.tied_lm_head);
        println!("  embd_format:   {:?}", weights.embd_format);
        // Sanity: the first layer's WQ weight should match config dims.
        let l0 = &weights.layers[0];
        println!("  layer 0 wq:    m={}, k={}, dtype={:?}", l0.wq.m, l0.wq.k, l0.wq.gpu_dtype);
        println!("  layer 0 wk:    m={}, k={}, dtype={:?}", l0.wk.m, l0.wk.k, l0.wk.gpu_dtype);
        println!("  output:        m={}, k={}, dtype={:?}", weights.output.m, weights.output.k, weights.output.gpu_dtype);
    }

    Ok(())
}
