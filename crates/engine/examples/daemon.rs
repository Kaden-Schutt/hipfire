//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Bun CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this","image":"/path/to/img.png","temperature":0.3,"max_tokens":512}
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use engine::hfq::HfqFile;
use engine::llama;
use engine::qwen35;
use engine::qwen35::DeltaNetState;
use engine::qwen35_vl;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

const IMAGE_SIZE: usize = 448;
const IMAGE_PAD_ID: u32 = 248056;
const VISION_START_ID: u32 = 248053;
const VISION_END_ID: u32 = 248054;

struct LoadedModel {
    arch_id: u32,
    // Qwen3.5 state
    q35_config: Option<qwen35::Qwen35Config>,
    q35_weights: Option<qwen35::Qwen35Weights>,
    q35_scratch: Option<qwen35::Qwen35Scratch>,
    kv_cache: Option<llama::KvCache>,
    dn_state: Option<DeltaNetState>,
    // Qwen3 state
    llama_config: Option<llama::LlamaConfig>,
    llama_weights: Option<llama::LlamaWeights>,
    llama_scratch: Option<llama::ForwardScratch>,
    llama_kv: Option<llama::KvCache>,
    // Vision state (VL models only)
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
    // Shared
    tokenizer: Option<engine::tokenizer::Tokenizer>,
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let mut model: Option<LoadedModel> = None;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() { continue; }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"invalid JSON: {}"}}"#, e);
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match msg_type {
            "load" => {
                // Unload previous if any
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let max_seq = msg.get("params").and_then(|p| p.get("max_seq")).and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
                let turbo_bits = msg.get("turbo").and_then(|v| v.as_u64()).unwrap_or(4) as u8;

                match load_model(path, max_seq, turbo_bits, &mut gpu) {
                    Ok(m) => {
                        let arch = if m.arch_id == 5 { "qwen3_5" } else { "qwen3" };
                        let vl = m.vision_config.is_some();
                        let (dim, layers, vocab) = if let Some(ref c) = m.q35_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.llama_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else { (0, 0, 0) };
                        let _ = writeln!(stdout, r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{}}}"#, arch, dim, layers, vocab, vl);
                        model = Some(m);
                    }
                    Err(e) => {
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"load failed: {}. GPU: {} ({} MB free / {} MB total)"}}"#, e, gpu.arch, free_mb, total_mb);
                    }
                }
                let _ = stdout.flush();
            }

            "generate" => {
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };

                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                let prompt = msg.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello");
                let image = msg.get("image").and_then(|v| v.as_str());
                let temp = msg.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
                let max_tokens = msg.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
                let top_p = msg.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
                let repeat_penalty = msg.get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.3) as f32;
                let repeat_window = msg.get("repeat_window").and_then(|v| v.as_u64()).unwrap_or(128) as usize;

                if image.is_some() && m.vision_config.is_some() {
                    generate_vl(m, &mut gpu, &mut stdout, id, prompt, image.unwrap(), temp, top_p, max_tokens, repeat_penalty, repeat_window);
                } else {
                    generate(m, &mut gpu, &mut stdout, id, prompt, temp, top_p, max_tokens, repeat_penalty, repeat_window);
                }
            }

            "unload" => {
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }
                let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                let _ = stdout.flush();
            }

            "ping" => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            _ => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"unknown type: {}"}}"#, msg_type);
                let _ = stdout.flush();
            }
        }
    }
}

fn load_model(path: &str, max_seq: usize, turbo_bits: u8, gpu: &mut rdna_compute::Gpu) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer not found")?;

    if hfq.arch_id == 5 {
        // Qwen3.5 DeltaNet
        let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;

        // Detect VL model: check if vision config AND vision tensors are present
        // Text-only models may have vision config in metadata but no actual vision weights
        let vision_config = qwen35_vl::vision_config_from_hfq(&hfq);
        let has_vision_tensors = hfq.tensor_data("model.visual.patch_embed.proj.weight").is_some();
        let (vision_config, vision_weights) = if let Some(vc) = vision_config {
            if has_vision_tensors {
                let vw = qwen35_vl::load_vision_weights(&hfq, &vc, gpu).map_err(|e| format!("{e}"))?;
                eprintln!("  VL model: vision encoder (hidden={}, layers={})", vc.hidden_size, vc.num_layers);
                (Some(vc), Some(vw))
            } else {
                (None, None) // text-only model, no vision tensors
            }
        } else {
            (None, None)
        };

        let weights = qwen35::load_weights(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        let kv = if turbo_bits >= 2 && turbo_bits <= 4 {
            eprintln!("  KV cache: turbo{turbo_bits}");
            llama::KvCache::new_gpu_turbo(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, turbo_bits).map_err(|e| format!("{e}"))?
        } else {
            eprintln!("  KV cache: Q8");
            llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
        };
        let dn = DeltaNetState::new(gpu, &config).map_err(|e| format!("{e}"))?;
        let scratch = qwen35::Qwen35Scratch::new(gpu, &config, 128).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: 5,
            q35_config: Some(config), q35_weights: Some(weights), q35_scratch: Some(scratch),
            kv_cache: Some(kv), dn_state: Some(dn),
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            vision_config, vision_weights,
            tokenizer: Some(tokenizer),
        })
    } else {
        // Qwen3 / LLaMA
        let config = engine::hfq::config_from_hfq(&hfq).ok_or("failed to read LLaMA config")?;
        let weights = engine::hfq::load_weights_hfq(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        let kv = if turbo_bits >= 2 && turbo_bits <= 4 {
            eprintln!("  KV cache: turbo{turbo_bits}");
            llama::KvCache::new_gpu_turbo(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, turbo_bits).map_err(|e| format!("{e}"))?
        } else {
            eprintln!("  KV cache: Q8");
            llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?
        };
        let scratch = llama::ForwardScratch::new(gpu, &config).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: Some(config), llama_weights: Some(weights), llama_scratch: Some(scratch), llama_kv: Some(kv),
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
        })
    }
}

fn unload_model(m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if let Some(kv) = m.kv_cache { kv.free_gpu(gpu); }
    if let Some(dn) = m.dn_state { dn.free_gpu(gpu); }
    if let Some(s) = m.q35_scratch { s.free_gpu(gpu); }
    if let Some(kv) = m.llama_kv { kv.free_gpu(gpu); }
    // Weights and ForwardScratch also hold GPU tensors but we don't have free_gpu for them yet
    // TODO: add free_gpu for Qwen35Weights, LlamaWeights, ForwardScratch
    gpu.drain_pool();
}

fn generate(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize) {
    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Build ChatML prompt
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let q_tokens = tokenizer.encode(prompt);

    let mut prompt_tokens = Vec::new();
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&user_tok);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&q_tokens);
    prompt_tokens.extend_from_slice(&im_end);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&asst_tok);
    prompt_tokens.extend_from_slice(&nl);

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let t0 = Instant::now();

    if m.arch_id == 5 {
        // Qwen3.5 path
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let kv = m.kv_cache.as_mut().unwrap();
        let dn = m.dn_state.as_mut().unwrap();

        // Prefill
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(gpu, weights, config, tok, pos, kv, dn, scratch).unwrap();
        }

        // Generate
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next_token = llama::sample_top_p(&logits, temp, top_p);
        let mut generated = 0;
        let mut token_history = prompt_tokens.clone();

        for _ in 0..max_tokens {
            generated += 1;
            token_history.push(next_token);
            let text = tokenizer.decode(&[next_token]);
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }

            let pos = prompt_tokens.len() + generated - 1;
            qwen35::forward_scratch(gpu, weights, config, next_token, pos, kv, dn, scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_ngram_block(&mut logits, &token_history);
            llama::apply_repeat_penalty(&mut logits, &token_history, repeat_window, repeat_penalty);
            next_token = llama::sample_top_p(&logits, temp, top_p);
        }

        let tok_s = generated as f64 / t0.elapsed().as_secs_f64();
        let _ = writeln!(stdout, r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1}}}"#, id, generated, tok_s);
        let _ = stdout.flush();
    } else {
        // Qwen3 / LLaMA path
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

        let mut rng_state = 42u32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let (_, rng) = llama::forward_scratch(gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0).unwrap();
            rng_state = rng;
        }

        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
        let mut next_token = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        rng_state = u32::from_ne_bytes([out_bytes[4], out_bytes[5], out_bytes[6], out_bytes[7]]);

        let mut generated = 0;
        let mut token_history = prompt_tokens.clone();

        for _ in 0..max_tokens {
            generated += 1;
            token_history.push(next_token);
            let text = tokenizer.decode(&[next_token]);
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }

            // Qwen3/LLaMA scratch repeat_buf is 64 slots — clamp window to fit
            let rw = repeat_window.min(64);
            let hist_start = token_history.len().saturating_sub(rw);
            let hist_slice = &token_history[hist_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

            let pos = prompt_tokens.len() + generated - 1;
            let (tok, rng) = llama::forward_scratch(gpu, weights, config, next_token, pos, kv, scratch, temp, top_p, rng_state, hist_slice.len(), repeat_penalty).unwrap();
            next_token = tok;
            rng_state = rng;
        }

        let tok_s = generated as f64 / t0.elapsed().as_secs_f64();
        let _ = writeln!(stdout, r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1}}}"#, id, generated, tok_s);
        let _ = stdout.flush();
    }
}

fn generate_vl(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, image_path: &str, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.q35_config.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();
    let vision_weights = m.vision_weights.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();

    // Load and preprocess image
    let pixels = engine::image::load_and_preprocess(Path::new(image_path), IMAGE_SIZE);
    let grid_h = IMAGE_SIZE / vision_config.patch_size;
    let grid_w = IMAGE_SIZE / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Extract patches and run vision encoder
    let patches = engine::image::extract_patches(
        &pixels, 3, IMAGE_SIZE, IMAGE_SIZE,
        vision_config.patch_size, vision_config.temporal_patch_size,
    );
    let visual_tokens = qwen35_vl::vision_forward(gpu, vision_weights, vision_config, &patches, grid_h, grid_w)
        .expect("vision forward failed");

    // Build VL prompt: <|im_start|>user\n<|vision_start|><|image_pad|>×N<|vision_end|>\n{question}<|im_end|>\n<|im_start|>assistant\n
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let q_tokens = tokenizer.encode(prompt);

    let mut prompt_tokens: Vec<u32> = Vec::new();
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&user_tok);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.push(VISION_START_ID);
    for _ in 0..n_visual_tokens {
        prompt_tokens.push(IMAGE_PAD_ID);
    }
    prompt_tokens.push(VISION_END_ID);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&q_tokens);
    prompt_tokens.extend_from_slice(&im_end);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&asst_tok);
    prompt_tokens.extend_from_slice(&nl);

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let t0 = Instant::now();

    // Prefill with vision token embedding for IMAGE_PAD positions
    let mut visual_idx = 0usize;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        if token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            qwen35::forward_scratch_embed(gpu, weights, config, emb, pos, kv, dn, scratch)
                .expect("forward_scratch_embed failed");
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(gpu, weights, config, token, pos, kv, dn, scratch)
                .expect("forward_scratch failed");
        }
    }

    // Generate
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::sample_top_p(&logits, temp, top_p);
    let mut generated = 0;
    let mut token_history = prompt_tokens.clone();

    for _ in 0..max_tokens {
        generated += 1;
        token_history.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
        let _ = stdout.flush();

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }

        let pos = prompt_tokens.len() + generated - 1;
        qwen35::forward_scratch(gpu, weights, config, next_token, pos, kv, dn, scratch).unwrap();
        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &token_history);
        llama::apply_repeat_penalty(&mut logits, &token_history, repeat_window, repeat_penalty);
        next_token = llama::sample_top_p(&logits, temp, top_p);
    }

    let tok_s = generated as f64 / t0.elapsed().as_secs_f64();
    let _ = writeln!(stdout, r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1}}}"#, id, generated, tok_s);
    let _ = stdout.flush();
}
