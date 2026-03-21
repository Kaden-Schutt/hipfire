//! Profile one forward pass: time each operation category.
fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");

    let path = std::env::args().nth(1)
        .unwrap_or("/home/kaden/llama.cpp/models/Qwen3-8B-Q4_K_M.gguf".to_string());
    let gguf = engine::gguf::GgufFile::open(std::path::Path::new(&path)).unwrap();
    let config = engine::llama::LlamaConfig::from_gguf(&gguf).unwrap();
    let weights = engine::llama::load_weights(&gguf, &config, &mut gpu).unwrap();
    eprintln!("Loaded: {} layers, dim={}", config.n_layers, config.dim);

    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv = engine::llama::KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap();

    // Warm up with a few tokens
    for pos in 0..3 {
        engine::llama::forward(&mut gpu, &weights, &config, 1, pos, &mut kv).unwrap();
    }

    // Profile token 3
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();

    let n_iter = 20;
    gpu.hip.event_record(&start, None).unwrap();
    for i in 0..n_iter {
        engine::llama::forward(&mut gpu, &weights, &config, 1, 3 + i, &mut kv).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let ms_per_token = ms / n_iter as f32;

    eprintln!("\n{n_iter} tokens: {ms:.1}ms total, {ms_per_token:.1}ms/token ({:.1} tok/s)",
        1000.0 / ms_per_token);
}
