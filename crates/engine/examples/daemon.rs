//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Bun CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use engine::hfq::HfqFile;
use engine::llama;
use engine::qwen35;
use engine::qwen35::DeltaNetState;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

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

                match load_model(path, max_seq, &mut gpu) {
                    Ok(m) => {
                        let arch = if m.arch_id == 5 { "qwen3_5" } else { "qwen3" };
                        let (dim, layers, vocab) = if let Some(ref c) = m.q35_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.llama_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else { (0, 0, 0) };
                        let _ = writeln!(stdout, r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{}}}"#, arch, dim, layers, vocab);
                        model = Some(m);
                    }
                    Err(e) => {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"load failed: {}"}}"#, e);
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
                let temp = msg.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
                let max_tokens = msg.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;

                generate(m, &mut gpu, &mut stdout, id, prompt, temp, max_tokens);
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

fn load_model(path: &str, max_seq: usize, gpu: &mut rdna_compute::Gpu) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer not found")?;

    if hfq.arch_id == 5 {
        // Qwen3.5 DeltaNet
        let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;
        let weights = qwen35::load_weights(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        let kv = llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?;
        let dn = DeltaNetState::new(gpu, &config).map_err(|e| format!("{e}"))?;
        let scratch = qwen35::Qwen35Scratch::new(gpu, &config, 128).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: 5,
            q35_config: Some(config), q35_weights: Some(weights), q35_scratch: Some(scratch),
            kv_cache: Some(kv), dn_state: Some(dn),
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            tokenizer: Some(tokenizer),
        })
    } else {
        // Qwen3 / LLaMA
        let config = engine::hfq::config_from_hfq(&hfq).ok_or("failed to read LLaMA config")?;
        let weights = engine::hfq::load_weights_hfq(&hfq, &config, gpu).map_err(|e| format!("{e}"))?;
        let kv = llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?;
        let scratch = llama::ForwardScratch::new(gpu, &config).map_err(|e| format!("{e}"))?;
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: Some(config), llama_weights: Some(weights), llama_scratch: Some(scratch), llama_kv: Some(kv),
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

fn generate(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, temp: f32, max_tokens: usize) {
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
        let mut next_token = llama::sample_top_p(&logits, temp, 0.8);
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
            llama::apply_repeat_penalty(&mut logits, &token_history, 128, 1.3);
            next_token = llama::sample_top_p(&logits, temp, 0.8);
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
            let (_, rng) = llama::forward_scratch(gpu, weights, config, tok, pos, kv, scratch, temp, 0.8, rng_state, 0, 1.0).unwrap();
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

            let hist_start = token_history.len().saturating_sub(64);
            let hist_slice = &token_history[hist_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

            let pos = prompt_tokens.len() + generated - 1;
            let (tok, rng) = llama::forward_scratch(gpu, weights, config, next_token, pos, kv, scratch, temp, 0.8, rng_state, hist_slice.len(), 1.3).unwrap();
            next_token = tok;
            rng_state = rng;
        }

        let tok_s = generated as f64 / t0.elapsed().as_secs_f64();
        let _ = writeln!(stdout, r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1}}}"#, id, generated, tok_s);
        let _ = stdout.flush();
    }
}
