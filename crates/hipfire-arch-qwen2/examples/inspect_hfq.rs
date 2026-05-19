//! Inspect a Qwen2 HFQ file: open it, parse the config via
//! [`hipfire_arch_qwen2::qwen2::config_from_hfq`], and print the
//! resulting [`Qwen2Config`]. Rev-0 smoke utility to verify the parser
//! works against real HFQ output (vs. the in-memory string fixtures the
//! unit tests use).
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example inspect_hfq -p hipfire-arch-qwen2 -- \
//!     ~/.hipfire/models/qwen2-1.5b.hfq4
//! ```

use std::path::Path;

use hipfire_arch_qwen2::qwen2;
use hipfire_runtime::hfq::HfqFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: inspect_hfq <path.hfq>")?;
    let hfq = HfqFile::open(Path::new(&path))?;
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

    Ok(())
}
