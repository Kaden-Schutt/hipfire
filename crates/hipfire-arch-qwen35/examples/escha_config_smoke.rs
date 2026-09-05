//! Prove the converted .hfq's metadata satisfies arch-6's config loader.
fn main() {
    let path = std::env::args().nth(1).expect("usage: <model.hfq>");
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path))
        .expect("open hfq");
    println!("arch_id = {}", hfq.arch_id);
    println!("tensors = {}", hfq.tensors().len());
    match hipfire_arch_qwen35::qwen35::config::config_from_hfq(&hfq) {
        Ok(c) => println!(
            "config OK: dim={} layers={} experts={} top_k={} moe_inter={} vocab={} is_vl_text={}",
            c.dim, c.n_layers, c.num_experts, c.num_experts_per_tok,
            c.moe_intermediate_size, c.vocab_size, c.is_vl_text
        ),
        Err(e) => { eprintln!("config FAILED: {e}"); std::process::exit(1); }
    }
}
