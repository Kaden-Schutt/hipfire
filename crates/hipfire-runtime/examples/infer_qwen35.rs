// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 (DeltaNet) inference — matches ollama quality settings.
//! Usage: infer_qwen35 <model.hfq> [--guards on|off] [prompt text...]
//!
//! `--guards` (default: off) opts the bare example into the production
//! generation guards owned by the daemon: ChatML framing via
//! `hipfire_runtime::prompt_frame::ChatFrame`, top-p sampling via
//! `hipfire_runtime::sampler::sample`, output-stream filtering via
//! `hipfire_runtime::eos_filter::EosFilter`, and the n-gram repetition detector
//! via `hipfire_runtime::loop_guard::LoopGuard`. The default keeps today's bare
//! semantics so kernel/loading sanity probes are unchanged byte-for-byte.

use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::loop_guard::LoopGuard;
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame};
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::DeltaNetState;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);
extern "C" fn handle_sigint(_: libc::c_int) { RUNNING.store(false, Ordering::SeqCst); }

/// Parse `--guards on|off` from argv WITHOUT consuming positional prompt
/// args when the flag is absent. Returns (use_guards, prompt_args).
///
/// Strategy: scan for `--guards`; if found, take the next token as the
/// value and elide both from the prompt-arg slice. If absent, the
/// prompt-arg slice equals `args[2..]` byte-for-byte and the bare
/// default behavior is preserved.
fn parse_guards_flag(args: &[String]) -> (bool, Vec<String>) {
    let mut use_guards = false;
    let mut prompt_args: Vec<String> = Vec::new();
    let mut i = 2; // args[0] = exe, args[1] = model_path
    while i < args.len() {
        if args[i] == "--guards" {
            // Consume the value if present; default to "on" if missing.
            let val = args.get(i + 1).map(|s| s.as_str()).unwrap_or("on");
            use_guards = matches!(val, "on" | "1" | "true");
            i += 2;
        } else {
            prompt_args.push(args[i].clone());
            i += 1;
        }
    }
    (use_guards, prompt_args)
}

fn main() {
    unsafe { libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t); }
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: infer_qwen35 <model.hfq> [--guards on|off] [prompt...]");
        std::process::exit(1);
    });

    let (use_guards, prompt_args) = parse_guards_flag(&args);
    let prompt_text = if !prompt_args.is_empty() {
        prompt_args.join(" ")
    } else {
        "Hello".to_string()
    };

    eprintln!("=== hipfire Qwen3.5 inference ===");
    eprintln!("Model: {model_path}");
    if use_guards {
        eprintln!("Guards: ON (prompt_frame + sampler + eos_filter + loop_guard)");
    }

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read Qwen3.5 config");
    eprintln!("Config: dim={}, layers={}, heads={}, vocab={}", config.dim, config.n_layers, config.n_heads, config.vocab_size);

    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .unwrap_or_else(|_| {
            let gguf = hipfire_runtime::gguf::GgufFile::open(Path::new("/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf")).expect("need GGUF for tokenizer");
            hipfire_runtime::tokenizer::Tokenizer::from_gguf(&gguf).expect("tokenizer failed")
        });

    let prompt_tokens = if use_guards {
        // Production framing path: route through hipfire_runtime::prompt_frame so
        // the example's prompt assembly matches the daemon byte-for-byte.
        // OpenThink keeps the `<think>` opener for thinking-mode models;
        // it falls back to Plain when the tokenizer doesn't register
        // `<think>` as a special token.
        ChatFrame {
            tokenizer: &tokenizer,
            system: None,
            user: &prompt_text,
            assistant_prefix: AssistantPrefix::OpenThink,
            raw: false,
        }
        .build()
    } else {
        // Bare default path — preserved verbatim from the pre-PR5 example.
        let mut prompt_tokens = tokenizer.encode(&prompt_text);
        let has_chatml = tokenizer.encode("<|im_start|>").len() == 1;
        if has_chatml {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user = tokenizer.encode("user");
            let asst = tokenizer.encode("assistant");
            let nl = tokenizer.encode("\n");
            let think = tokenizer.encode("<think>");

            let mut chat = Vec::new();
            // No system message (matches ollama defaults)
            // User message
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&prompt_tokens);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl);
            // Assistant start with <think>
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&think);
            chat.extend_from_slice(&nl);
            prompt_tokens = chat;
        }
        prompt_tokens
    };
    eprintln!("Prompt: \"{}\" ({} tokens)", prompt_text, prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading weights...");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("failed to load weights");

    let kv_seq = 2048usize;
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    let mut kv_cache = match kv_mode.as_str() {
        "givens4" => { eprintln!("KV cache: givens4"); llama::KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap() }
        "givens2" => { eprintln!("KV cache: givens2"); llama::KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap() }
        _ => llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
    };
    let mut dn_state = if std::env::var("FP32_STATE").is_ok() {
        DeltaNetState::new_with_quant(&mut gpu, &config, hipfire_arch_qwen35::qwen35::StateQuant::FP32).unwrap()
    } else {
        DeltaNetState::new(&mut gpu, &config).unwrap()
    };

    // Phase 3a-A: use forward_scratch path (avoids per-call alloc/free + uses
    // the fused repeat_interleave kernel). Allocate scratch once, reuse forever.
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // ── GPU-assisted top-K sampler (opt-in, experimental) ──
    //
    // HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU
    // sample_top_p_from_candidates path, which avoids the 600 KB logits
    // DtoH per token. It is BIT-EXACT with the full-CPU path for
    // sampling-from-full-logits because apply_repeat_penalty can only
    // decrease logits, so the pre-penalty top-128 ⊇ post-penalty top-20.
    //
    // HIPFIRE_SAMPLE_COMPARE=1 additionally runs the old full-CPU path
    // every step and panics on token divergence — long-generation safety
    // check for the opt-in flag before enabling it by default.
    //
    // Note: --guards on takes precedence over these env knobs because
    // the production sampler module does its own GPU top-p kernel call
    // and the env-flag paths below assume the bare CPU sampler shape.
    let use_gpu_topk = std::env::var("HIPFIRE_GPU_TOPK").ok().as_deref() == Some("1");
    let sample_compare = std::env::var("HIPFIRE_SAMPLE_COMPARE").ok().as_deref() == Some("1");
    if (use_gpu_topk || sample_compare) && !use_guards {
        eprintln!("sampler: gpu_topk={} compare={}", use_gpu_topk, sample_compare);
    }
    const TOPK: usize = 1024;  // 256 threads × top-4 each
    // Single 8 KB device buffer laid out as [1024 × u32 indices | 1024 × f32 values]
    // so the whole top-K candidate set downloads in one memcpy_dtoh call.
    let topk_buf = gpu.alloc_tensor(&[2 * TOPK], rdna_compute::DType::F32).unwrap();
    let mut topk_host = vec![0u8; 2 * TOPK * 4];  // reused across steps

    // Sequential prefill
    let t1 = Instant::now();
    let mut logits = vec![0.0f32; config.vocab_size];
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &weights, &config, token, pos, &mut kv_cache, &mut dn_state, &scratch)
            .expect("forward failed");
        if !use_guards {
            // Bare path needs the host-side logits for CPU sampling.
            // The guards path defers the D2H to the sampler module.
            logits = gpu.download_f32(&scratch.logits).unwrap();
        }
    }
    let prefill_ms = t1.elapsed().as_millis();
    eprintln!("Prefill: {}ms ({} tokens, {:.0} tok/s)", prefill_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0));

    // Detect special tokens
    let think_end_id = tokenizer.encode("</think>");
    let think_end_token = if think_end_id.len() == 1 { Some(think_end_id[0]) } else { None };
    let im_end_id = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end_id.len() == 1 { Some(im_end_id[0]) } else { None };

    let sc = llama::SamplingConfig::text_thinking();
    let max_gen = 2048;

    let t2 = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut in_thinking = true;
    let mut generated = Vec::new();

    // Production guards path: stream output through EosFilter and watch
    // for n-gram loops via LoopGuard. Set up here so the loop body can
    // route bytes/tokens through them when use_guards is true.
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let loop_guard = LoopGuard::from_config(hipfire_runtime::config::get());
    let mut bytes_fed_to_filter = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();

    // RNG for the production sampler path. Seeded matching the daemon
    // default so behavior is reproducible across runs.
    let mut rng_state_u32: u32 = 0x13579BDFu32;

    if !use_guards {
        eprint!("<think>");
    }
    let mut next_token = if use_guards {
        // First sample via the production sampler module — same kernel
        // path the daemon uses, with its `temp/top_p/repeat_penalty`
        // policy. `repeat_window` is bounded by the GPU repeat_buf
        // capacity so we don't outrun the on-device buffer.
        let repeat_buf_cap = scratch.repeat_buf.buf.size() / 4;
        let cfg = SamplerConfig {
            temperature: sc.think_temp,
            top_p: sc.top_p,
            repeat_penalty: sc.repeat_penalty,
            repeat_window: repeat_buf_cap.min(sc.repeat_window),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
        };
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
        llama::sample_top_p(&logits, sc.think_temp, sc.top_p)
    };

    for _gi in 0..max_gen {
        generated.push(next_token);
        token_history.push(next_token);
        streamed_tokens.push(next_token);

        if use_guards {
            // Production output-stream path: feed only the new bytes to
            // the filter so partial UTF-8 codepoints / marker prefixes
            // are buffered until the next token disambiguates them.
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
            // Bare default behavior — unchanged.
            if in_thinking && think_end_token == Some(next_token) {
                in_thinking = false;
                eprint!("</think>\n");
            } else {
                let text = tokenizer.decode(&[next_token]);
                if in_thinking {
                    eprint!("{text}");
                } else {
                    print!("{text}");
                    std::io::stdout().flush().ok();
                }
            }
        }

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }
        if !RUNNING.load(Ordering::Relaxed) { break; }

        // N-gram loop detector (guards on only). When the streamed
        // tokens trigger the LoopGuard, force EOS — same shape as the
        // daemon's loop-guard early-exit.
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
            // Track think state for guards-on path so the temperature
            // mode switches at </think>, even though the EosFilter is
            // not configured to strip it (we want the example to still
            // SHOW the think block).
            if in_thinking && think_end_token == Some(next_token) {
                in_thinking = false;
            }
        }

        let pos = prompt_tokens.len() + generated.len() - 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &mut dn_state, &scratch)
            .expect("forward failed");

        let temp = if in_thinking { sc.think_temp } else { sc.answer_temp };

        next_token = if use_guards {
            // Route through the production sampler module. Mirrors the
            // daemon's per-step sampler call exactly.
            let repeat_buf_cap = scratch.repeat_buf.buf.size() / 4;
            let cfg = SamplerConfig {
                temperature: temp,
                top_p: sc.top_p,
                repeat_penalty: sc.repeat_penalty,
                repeat_window: repeat_buf_cap.min(sc.repeat_window),
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
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
        } else if use_gpu_topk || sample_compare {
            // GPU-assisted path: run topk_logits_f32 kernel, ONE DtoH,
            // run the same CPU sampler on the top-1024 subset.
            gpu.topk_logits_f32(&scratch.logits, &topk_buf, config.vocab_size)
                .expect("topk_logits");
            gpu.hip.memcpy_dtoh(&mut topk_host, &topk_buf.buf).unwrap();

            // (memcpy_dtoh already done above for timing)
            let mut cand_ids = vec![0u32; TOPK];
            let mut cand_vals = vec![0.0f32; TOPK];
            for i in 0..TOPK {
                cand_ids[i] = u32::from_ne_bytes([
                    topk_host[i*4], topk_host[i*4+1], topk_host[i*4+2], topk_host[i*4+3],
                ]);
                let v_off = TOPK * 4 + i * 4;
                cand_vals[i] = f32::from_ne_bytes([
                    topk_host[v_off], topk_host[v_off+1], topk_host[v_off+2], topk_host[v_off+3],
                ]);
            }

            if sample_compare {
                // Snapshot RNG so both samplers see the same state.
                let rng_before = llama::sampler_rng_snapshot();

                // GPU-assisted path (advances RNG)
                let mut cand_vals_gpu = cand_vals.clone();
                let gpu_tok = llama::sample_top_p_from_candidates(
                    &cand_ids, &mut cand_vals_gpu, &token_history,
                    sc.repeat_window, sc.repeat_penalty, temp, sc.top_p,
                );
                let rng_after_gpu = llama::sampler_rng_snapshot();

                // Restore and run full-CPU path
                llama::sampler_rng_restore(rng_before);
                logits = gpu.download_f32(&scratch.logits).unwrap();
                llama::apply_repeat_penalty(&mut logits, &token_history, sc.repeat_window, sc.repeat_penalty);
                let cpu_tok = llama::sample_top_p(&logits, temp, sc.top_p);
                let rng_after_cpu = llama::sampler_rng_snapshot();

                if cpu_tok != gpu_tok || rng_after_cpu != rng_after_gpu {
                    eprintln!("\n!! SAMPLE_COMPARE divergence at step {}: cpu={} gpu={} (pos={})",
                        generated.len(), cpu_tok, gpu_tok, pos);
                    eprintln!("   rng state: before={rng_before:#x} after_gpu={rng_after_gpu:#x} after_cpu={rng_after_cpu:#x}");
                    // Show the top-128 candidate set + which of them the CPU's top-20 came from
                    let mut sorted: Vec<(u32, f32)> = cand_ids.iter().copied().zip(cand_vals.iter().copied()).collect();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    eprintln!("   gpu top-8 candidates (by raw logit):");
                    for (i, (tok, val)) in sorted.iter().take(8).enumerate() {
                        eprintln!("     [{i}] tok={tok} raw_logit={val}");
                    }
                    panic!("sample comparison failed");
                }
                // Both matched and advanced RNG to the same state. Leave it at rng_after_gpu.
                gpu_tok
            } else {
                llama::sample_top_p_from_candidates(
                    &cand_ids, &mut cand_vals, &token_history,
                    sc.repeat_window, sc.repeat_penalty, temp, sc.top_p,
                )
            }
        } else {
            logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_repeat_penalty(&mut logits, &token_history, sc.repeat_window, sc.repeat_penalty);
            llama::sample_top_p(&logits, temp, sc.top_p)
        };
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
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}
