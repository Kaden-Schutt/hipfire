//! Per-layer profiling of hipfire inference.
//! Measures every kernel with hipDeviceSynchronize barriers.
//! Usage: cargo run --release --example profile_layers <model.hfq> [n_tokens]

use engine::hfq::{self, HfqFile};
use engine::llama::{self, KvCache, ForwardScratch, weight_gemv};
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::time::Instant;

fn sync_us(gpu: &Gpu, t: Instant) -> f64 {
    gpu.hip.device_synchronize().unwrap();
    t.elapsed().as_nanos() as f64 / 1000.0
}

#[derive(Default, Clone, serde::Serialize)]
struct LayerTimings {
    attn_norm_us: f64,
    q_proj_us: f64,
    k_proj_us: f64,
    v_proj_us: f64,
    qk_norm_us: f64,
    rope_us: f64,
    kv_cache_us: f64,
    attention_us: f64,
    o_proj_us: f64,
    attn_residual_us: f64,
    ffn_norm_us: f64,
    gate_proj_us: f64,
    up_proj_us: f64,
    silu_mul_us: f64,
    down_proj_us: f64,
    ffn_residual_us: f64,
    total_us: f64,
}

#[derive(Default, Clone, serde::Serialize)]
struct TokenTimings {
    embedding_us: f64,
    layers: Vec<LayerTimings>,
    output_norm_us: f64,
    output_proj_us: f64,
    sampling_us: f64,
    total_us: f64,
    pos: usize,
}

#[derive(serde::Serialize)]
struct SystemSnapshot {
    gpu_temp_c: f64,
    gpu_power_w: f64,
    gpu_sclk_mhz: u64,
    gpu_mclk_mhz: u64,
    gpu_util_pct: f64,
    gpu_vram_used_mb: f64,
    gpu_vram_total_mb: f64,
    cpu_temp_c: f64,
    ram_used_mb: f64,
    ram_total_mb: f64,
}

fn system_snapshot() -> SystemSnapshot {
    let mut snap = SystemSnapshot {
        gpu_temp_c: 0.0, gpu_power_w: 0.0, gpu_sclk_mhz: 0, gpu_mclk_mhz: 0,
        gpu_util_pct: 0.0, gpu_vram_used_mb: 0.0, gpu_vram_total_mb: 0.0,
        cpu_temp_c: 0.0, ram_used_mb: 0.0, ram_total_mb: 0.0,
    };
    // GPU metrics from sysfs
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/hwmon/hwmon2/temp1_input") {
        snap.gpu_temp_c = s.trim().parse::<f64>().unwrap_or(0.0) / 1000.0;
    } else if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/hwmon/hwmon1/temp1_input") {
        snap.gpu_temp_c = s.trim().parse::<f64>().unwrap_or(0.0) / 1000.0;
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/hwmon/hwmon2/power1_average") {
        snap.gpu_power_w = s.trim().parse::<f64>().unwrap_or(0.0) / 1_000_000.0;
    } else if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/hwmon/hwmon1/power1_average") {
        snap.gpu_power_w = s.trim().parse::<f64>().unwrap_or(0.0) / 1_000_000.0;
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/pp_dpm_sclk") {
        for line in s.lines() { if line.contains('*') {
            snap.gpu_sclk_mhz = line.split_whitespace().nth(1).and_then(|v| v.trim_end_matches("Mhz").parse().ok()).unwrap_or(0);
        }}
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/pp_dpm_mclk") {
        for line in s.lines() { if line.contains('*') {
            snap.gpu_mclk_mhz = line.split_whitespace().nth(1).and_then(|v| v.trim_end_matches("Mhz").parse().ok()).unwrap_or(0);
        }}
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/gpu_busy_percent") {
        snap.gpu_util_pct = s.trim().parse().unwrap_or(0.0);
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_vram_used") {
        snap.gpu_vram_used_mb = s.trim().parse::<f64>().unwrap_or(0.0) / 1_048_576.0;
    }
    if let Ok(s) = std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_vram_total") {
        snap.gpu_vram_total_mb = s.trim().parse::<f64>().unwrap_or(0.0) / 1_048_576.0;
    }
    // CPU temp
    for i in 0..10 {
        let p = format!("/sys/class/hwmon/hwmon{i}/temp1_input");
        if let Ok(s) = std::fs::read_to_string(&p) {
            let name_path = format!("/sys/class/hwmon/hwmon{i}/name");
            let name = std::fs::read_to_string(&name_path).unwrap_or_default();
            if name.contains("k10temp") || name.contains("coretemp") || name.contains("zenpower") {
                snap.cpu_temp_c = s.trim().parse::<f64>().unwrap_or(0.0) / 1000.0;
                break;
            }
        }
    }
    // RAM from /proc/meminfo
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let mut total = 0u64; let mut avail = 0u64;
        for line in s.lines() {
            if line.starts_with("MemTotal:") { total = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0); }
            if line.starts_with("MemAvailable:") { avail = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0); }
        }
        snap.ram_total_mb = total as f64 / 1024.0;
        snap.ram_used_mb = (total - avail) as f64 / 1024.0;
    }
    snap
}

fn profile_token(
    gpu: &mut Gpu, weights: &llama::LlamaWeights, config: &llama::LlamaConfig,
    token: u32, pos: usize, kv_cache: &mut KvCache, scratch: &ForwardScratch,
    temperature: f32, top_p: f32, rng_state: u32,
    repeat_window: usize, repeat_penalty: f32,
) -> (TokenTimings, u32, u32) {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dim = config.dim;

    let tok_start = Instant::now();
    let mut timings = TokenTimings { pos, ..Default::default() };

    // Embedding
    gpu.hip.device_synchronize().unwrap();
    let t = Instant::now();
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes()).unwrap();
    match weights.embd_format {
        llama::EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim).unwrap(),
        llama::EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim).unwrap(),
        llama::EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim).unwrap(),
        llama::EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &scratch.x, token, dim).unwrap(),
        llama::EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim).unwrap(),
    }
    timings.embedding_us = sync_us(gpu, t);

    // Layers
    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let mut lt = LayerTimings::default();
        let layer_start = Instant::now();

        let t = Instant::now();
        gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps).unwrap();
        lt.attn_norm_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.wq, &scratch.tmp, &scratch.q).unwrap();
        lt.q_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.wk, &scratch.tmp, &scratch.k).unwrap();
        lt.k_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.wv, &scratch.tmp, &scratch.v).unwrap();
        lt.v_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(&scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps).unwrap();
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(&scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps).unwrap();
            }
        }
        lt.qk_norm_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf, n_heads, n_kv_heads, head_dim, config.rope_freq_base).unwrap();
        lt.rope_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim).unwrap();
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim).unwrap();
        lt.kv_cache_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.attention_f32(
            &scratch.q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &scratch.attn_out, &scratch.pos_buf, pos + 1, n_heads, n_kv_heads, head_dim, kv_cache.max_seq,
        ).unwrap();
        lt.attention_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o).unwrap();
        lt.o_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.add_inplace_f32(&scratch.x, &scratch.o).unwrap();
        lt.attn_residual_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps).unwrap();
        lt.ffn_norm_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.w_gate, &scratch.tmp, &scratch.gate).unwrap();
        lt.gate_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.w_up, &scratch.tmp, &scratch.up).unwrap();
        lt.up_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden).unwrap();
        lt.silu_mul_us = sync_us(gpu, t);

        let t = Instant::now();
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out).unwrap();
        lt.down_proj_us = sync_us(gpu, t);

        let t = Instant::now();
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out).unwrap();
        lt.ffn_residual_us = sync_us(gpu, t);

        gpu.hip.device_synchronize().unwrap();
        lt.total_us = layer_start.elapsed().as_nanos() as f64 / 1000.0;
        timings.layers.push(lt);
    }

    // Output head
    let t = Instant::now();
    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps).unwrap();
    timings.output_norm_us = sync_us(gpu, t);

    let t = Instant::now();
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits).unwrap();
    timings.output_proj_us = sync_us(gpu, t);

    let t = Instant::now();
    let (tok_id, new_rng) = gpu.sample_top_p(
        &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
        config.vocab_size, temperature, top_p, rng_state,
        repeat_window, repeat_penalty,
    ).unwrap();
    timings.sampling_us = sync_us(gpu, t);

    timings.total_us = tok_start.elapsed().as_nanos() as f64 / 1000.0;
    (timings, tok_id, new_rng)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: profile_layers <model.hfq> [n_tokens]");
    let n_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = hfq::config_from_hfq(&hfq).expect("failed to read config");
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("no tokenizer in HFQ");

    eprintln!("Model: {model_path}");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, head_dim={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.head_dim, config.vocab_size);
    eprintln!("Profiling {n_tokens} tokens with per-op sync barriers");

    let mut gpu = Gpu::init().expect("GPU init failed");
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("failed to load weights");
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap();
    let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();

    // Prompt tokens
    let mut prompt_tokens = tokenizer.encode("Hello");
    if config.arch == llama::ModelArch::Qwen3 {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user_tok = tokenizer.encode("user");
        let asst_tok = tokenizer.encode("assistant");
        let nl_tok = tokenizer.encode("\n");
        let sys_tok = tokenizer.encode("system");
        let sys_msg = tokenizer.encode("You are a helpful assistant.");
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&sys_tok); chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&sys_msg); chat.extend_from_slice(&im_end); chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&user_tok); chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&prompt_tokens); chat.extend_from_slice(&im_end); chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&asst_tok); chat.extend_from_slice(&nl_tok);
        prompt_tokens = chat;
    }

    // Warmup: run prompt through without profiling
    let mut rng_state = 42u32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let (_, _, rng) = profile_token(&mut gpu, &weights, &config, token, pos, &mut kv_cache, &scratch,
            0.6, 0.8, rng_state, 0, 1.0);
        rng_state = rng;
    }
    eprintln!("Prompt processed ({} tokens), now profiling {} generation tokens...", prompt_tokens.len(), n_tokens);

    // Get first gen token
    let mut out_bytes = [0u8; 8];
    gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
    let mut next_token = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);

    // Profile generation tokens
    let mut all_timings: Vec<TokenTimings> = Vec::new();
    let mut all_snapshots: Vec<SystemSnapshot> = Vec::new();
    let mut token_history: Vec<u32> = prompt_tokens.clone();

    for i in 0..n_tokens {
        token_history.push(next_token);
        let hist_start = token_history.len().saturating_sub(64);
        let hist_slice = &token_history[hist_start..];
        let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

        let pos = prompt_tokens.len() + i;
        let (timings, tok, rng) = profile_token(
            &mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &scratch,
            0.6, 0.8, rng_state, hist_slice.len(), 1.1,
        );

        let snap = system_snapshot();
        let text = tokenizer.decode(&[next_token]);
        eprintln!("  token {i}: id={next_token} text={text:?} total={:.0}µs", timings.total_us);

        all_timings.push(timings);
        all_snapshots.push(snap);
        next_token = tok;
        rng_state = rng;

        if next_token == config.eos_token { break; }
    }

    // Output JSON
    let output = serde_json::json!({
        "model": model_path,
        "config": {
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "hidden_dim": config.hidden_dim,
        },
        "n_prompt_tokens": prompt_tokens.len(),
        "n_gen_tokens": all_timings.len(),
        "token_timings": all_timings,
        "system_snapshots": all_snapshots,
    });

    let json_str = serde_json::to_string_pretty(&output).unwrap();
    println!("{json_str}");
    eprintln!("\nDone. {} tokens profiled.", all_timings.len());
}
