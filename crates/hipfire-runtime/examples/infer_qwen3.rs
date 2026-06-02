// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Run inference on a .hfq (hipfire-quantized) model.
//! Usage: cargo run --release --example infer_qwen3 <model.hfq> [flags] [prompt text...]
//! Flags: --q8kv, --fp32kv, --givens4, --givens2, --hfq4kv, --temp T, --guards on|off
//!
//! `--guards` (default: off) opts the bare example into the production
//! generation guards owned by the daemon: ChatML framing via
//! `hipfire_runtime::prompt_frame::ChatFrame`, top-p sampling via
//! `hipfire_runtime::sampler::sample`, output-stream filtering via
//! `hipfire_runtime::eos_filter::EosFilter`, and the n-gram repetition detector
//! via `hipfire_runtime::loop_guard::LoopGuard`. The default keeps today's bare
//! semantics so kernel/loading sanity probes are unchanged byte-for-byte.

use hipfire_arch_llama::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use hipfire_runtime::loop_guard::LoopGuard;
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame};
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);
extern "C" fn handle_sigint(_: libc::c_int) { RUNNING.store(false, Ordering::SeqCst); }

fn main() {
    unsafe { libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t); }
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1)
        .unwrap_or_else(|| { eprintln!("Usage: infer_qwen3 <model.hfq> [--q8kv|--fp32kv|--givens4|--givens2|--hfq4kv] [--temp T] [--guards on|off] [prompt...]"); std::process::exit(1); });

    // Parse flags
    let use_givens4 = args.iter().any(|a| a == "--givens4");
    let use_givens2 = args.iter().any(|a| a == "--givens2");
    let use_q8kv = args.iter().any(|a| a == "--q8kv");
    let use_fp32kv = args.iter().any(|a| a == "--fp32kv");
    let use_hfq4kv = args.iter().any(|a| a == "--hfq4kv");
    let temp: f32 = args.iter().position(|a| a == "--temp")
        .map(|i| args[i + 1].parse().unwrap_or(0.6))
        .unwrap_or(0.0);
    let top_p: f32 = if temp == 0.0 { 1.0 } else { 0.8 };

    // Parse `--guards on|off` (default: off) and reserve both tokens
    // from the prompt assembly below. `--guards` without a value
    // defaults to `on` — matches conventional CLI shape.
    let use_guards: bool = match args.iter().position(|a| a == "--guards") {
        Some(i) => {
            let val = args.get(i + 1).map(|s| s.as_str()).unwrap_or("on");
            matches!(val, "on" | "1" | "true")
        }
        None => false,
    };

    let prompt_text = {
        let skip_flags = ["--q8kv", "--fp32kv", "--hfq4kv", "--givens4", "--givens2", "--temp", "--maxgen", "--guards"];
        let mut skip_next = false;
        let parts: Vec<&str> = args[2..].iter().filter(|a| {
            if skip_next { skip_next = false; return false; }
            if skip_flags.contains(&a.as_str()) {
                // --temp, --maxgen, and --guards take a value argument.
                skip_next = matches!(a.as_str(), "--temp" | "--maxgen" | "--guards");
                return false;
            }
            true
        }).map(|s| s.as_str()).collect();
        if parts.is_empty() { "Hello".to_string() } else { parts.join(" ") }
    };

    eprintln!("=== hipfire inference engine (HFQ) ===");
    eprintln!("Model: {model_path}");
    if temp == 0.0 { eprintln!("Sampling: GREEDY"); }
    else { eprintln!("Sampling: temp={temp}, top_p={top_p}"); }
    if use_guards {
        eprintln!("Guards: ON (prompt_frame + sampler + eos_filter + loop_guard)");
    }

    // Parse HFQ. PR 11: route bring-up triple through `Architecture` trait
    // dispatch — same hybrid pattern as PR 8's qwen35 daemon path. Forward
    // calls below stay direct `llama::*` static dispatch for the hot loop.
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = <Llama as Architecture>::config_from_hfq(&hfq)
        .expect("failed to read model config");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    // Load tokenizer from HFQ metadata, fallback to GGUF
    let tokenizer: hipfire_runtime::tokenizer::Tokenizer = if let Ok(t) = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        eprintln!("Tokenizer: {} tokens (from HFQ)", t.vocab_size());
        t
    } else {
        let gguf_path = if config.arch == llama::ModelArch::Qwen3 {
            "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf"
        } else {
            "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        };
        let gguf = hipfire_runtime::gguf::GgufFile::open(Path::new(gguf_path)).expect("need GGUF for tokenizer");
        let t = hipfire_runtime::tokenizer::Tokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
        eprintln!("Tokenizer: {} tokens (from GGUF)", t.vocab_size());
        t
    };

    let prompt_tokens = if use_guards {
        // Production framing path: route through hipfire_runtime::prompt_frame so
        // the example's prompt assembly matches the daemon byte-for-byte.
        // Plain assistant prefix — this example is intended for
        // non-thinking models (Qwen3 0.6B, TinyLlama).
        ChatFrame {
            tokenizer: &tokenizer,
            system: None,
            user: &prompt_text,
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build()
    } else {
        // Bare default path — preserved verbatim from the pre-PR5 example.
        let mut prompt_tokens = tokenizer.encode(&prompt_text);
        // ChatML: auto-detect
        let has_chatml = tokenizer.encode("<|im_start|>").len() == 1
            && tokenizer.encode("<|im_end|>").len() == 1;
        if has_chatml {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user_tok = tokenizer.encode("user");
            let asst_tok = tokenizer.encode("assistant");
            let nl_tok = tokenizer.encode("\n");
            let mut chat = Vec::new();
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user_tok);
            chat.extend_from_slice(&nl_tok);
            chat.extend_from_slice(&prompt_tokens);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl_tok);
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst_tok);
            chat.extend_from_slice(&nl_tok);
            prompt_tokens = chat;
        }
        prompt_tokens
    };

    eprintln!("Prompt: \"{}\" → {} tokens", prompt_text, prompt_tokens.len());

    // Init GPU
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Load weights via the trait — same dispatch path as the daemon.
    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = if use_givens4 {
        KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_givens2 {
        KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_hfq4kv {
        KvCache::new_gpu_hfq4kv(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_q8kv {
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_fp32kv {
        KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else {
        // Default: Q8 KV cache
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    };

    // Persistent scratch buffers (zero-alloc forward pass) via the trait.
    let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).unwrap();
    let mut rng_state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    if rng_state == 0 { rng_state = 1; }

    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let repeat_penalty: f32 = 1.1;
    let repeat_window: usize = 64;

    // Prefill: try batched (skip for non-standard KV formats), fallback to sequential
    let t1 = Instant::now();
    let prefill_logits = if use_givens4 || use_givens2 || use_hfq4kv {
        Err(hip_bridge::HipError::new(0, "quantized KV requires sequential prefill"))
    } else {
        llama::prefill_forward(&mut gpu, &weights, &config, &prompt_tokens, &mut kv_cache)
    };

    // Production sampler config (used only when --guards on). Mirrors
    // the daemon's per-token cfg. `repeat_window` is bounded by the
    // GPU repeat_buf capacity inside sampler::sample.
    let repeat_buf_cap = scratch.repeat_buf.buf.size() / 4;
    let make_sampler_cfg = |t: f32| SamplerConfig {
        temperature: t,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap.min(repeat_window),
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
    };
    let mut rng_state_u32: u32 = rng_state;

    let mut next_token = if let Ok(logits) = prefill_logits {
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [batched]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        llama::argmax(&logits)
    } else if use_guards {
        // Guards-on sequential prefill: use forward_scratch_compute so
        // the GPU sampler kernel does NOT also fire (we'll call
        // sampler::sample explicitly below). This keeps the production
        // sampler module as the single sample call site.
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            llama::forward_scratch_embed(&mut gpu, &weights, &config, token, pos, &scratch)
                .expect("forward_scratch_embed failed");
            llama::forward_scratch_compute(&mut gpu, &weights, &config, pos, &mut kv_cache, &scratch)
                .expect("forward_scratch_compute failed");
        }
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [sequential, guards]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        let cfg = make_sampler_cfg(temp.max(0.01));
        sampler::sample(
            &mut gpu,
            &scratch.logits,
            &scratch.sample_buf,
            &scratch.repeat_buf,
            config.vocab_size,
            &token_history,
            &cfg,
            &mut rng_state_u32,
        )
    } else {
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let (_, rng) = llama::forward_scratch(
                &mut gpu, &weights, &config, token, pos, &mut kv_cache,
                &scratch, temp.max(0.01), top_p, rng_state, 0, 1.0,
            ).expect("forward_scratch failed");
            rng_state = rng;
        }
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [sequential]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
        u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]])
    };

    // Generate
    let max_gen: usize = args.iter().position(|a| a == "--maxgen")
        .map(|i| args[i + 1].parse().unwrap_or(128)).unwrap_or(128);
    eprintln!("\nGenerating (max {max_gen} tokens)...\n");
    let t2 = Instant::now();
    let mut generated = Vec::new();

    // EosFilter / LoopGuard / streamed-tokens state — only used when
    // --guards on, but constructed unconditionally to keep the loop
    // body simple. Construction is cheap (no allocations until use).
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let loop_guard = LoopGuard::from_config(hipfire_runtime::config::get());
    let mut bytes_fed_to_filter = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);

        if use_guards {
            // Production output-stream path: feed only the new bytes to
            // the filter so partial UTF-8 codepoints / marker prefixes
            // are buffered until the next token disambiguates them.
            streamed_tokens.push(next_token);
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                if let Ok(text) = std::str::from_utf8(&text_bytes) {
                    print!("{text}");
                    std::io::stdout().flush().ok();
                }
            }
        } else {
            let text = tokenizer.decode(&[next_token]);
            print!("{text}");
            std::io::stdout().flush().ok();
        }

        if next_token == config.eos_token || !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        // N-gram loop detector (guards on only).
        if use_guards {
            if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
                loop_guard.check(&streamed_tokens)
            {
                let window_len = loop_guard.window_len(streamed_tokens.len());
                eprintln!(
                    "\n[guards] ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS",
                    count, window_len
                );
                break;
            }
        }

        token_history.push(next_token);
        let pos = prompt_tokens.len() + generated.len() - 1;

        if use_guards {
            // Production decode step: forward_scratch_embed +
            // forward_scratch_compute + sampler::sample. Same kernel
            // path as the daemon AR loop.
            llama::forward_scratch_embed(&mut gpu, &weights, &config, next_token, pos, &scratch)
                .expect("forward_scratch_embed failed");
            llama::forward_scratch_compute(&mut gpu, &weights, &config, pos, &mut kv_cache, &scratch)
                .expect("forward_scratch_compute failed");
            let cfg = make_sampler_cfg(temp.max(0.01));
            next_token = sampler::sample(
                &mut gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                &token_history,
                &cfg,
                &mut rng_state_u32,
            );
        } else {
            // Bare default: forward_scratch does forward + GPU sample
            // in one call. The repeat-window upload below mirrors the
            // pre-PR5 inline behavior verbatim.
            let hist_start = token_history.len().saturating_sub(repeat_window);
            let hist_slice = &token_history[hist_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

            let (tok, rng) = llama::forward_scratch(
                &mut gpu, &weights, &config, next_token, pos, &mut kv_cache,
                &scratch, temp.max(0.01), top_p, rng_state,
                hist_slice.len(), repeat_penalty,
            ).expect("forward_scratch failed");
            next_token = tok;
            rng_state = rng;
        }
    }

    // Drain any bytes the EosFilter was holding back at end-of-stream.
    if use_guards {
        let drained = filter.flush_pending();
        if !drained.is_empty() {
            if let Ok(text) = std::str::from_utf8(&drained) {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
        }
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 {
        generated.len() as f64 / (gen_ms as f64 / 1000.0)
    } else { 0.0 };

    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===",
        generated.len(), gen_ms, tok_s);
}
