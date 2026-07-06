// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure greedy token dump for byte-exact regression comparison.
//! Supports both HFQ (.hfq/.mq4) and safetensors-directory (PaRo) models.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use hipfire_runtime::safetensors_source::SafetensorsSource;
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: greedy_dump <model.(hfq|mq4|dir)> [out_tokens.txt] [prompt...]");
        eprintln!("  If out_tokens.txt is omitted, tokens go to stdout.");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_path = args.get(2).cloned();
    let prompt_text = if args.len() > 3 {
        args[3..].join(" ")
    } else {
        "Write a 500-word essay about Federalist No. 10 by James Madison.".to_string()
    };

    let mode = std::env::var("PROMPT_MODE").unwrap_or_else(|_| "thinking".to_string());
    eprintln!("greedy_dump: {model_path} mode={mode}");

    // ---- GPU init ----
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // ---- Load model (HFQ or safetensors) ----
    let model = Path::new(model_path);
    let (config, weights, hfq_opt) = if model.is_dir() {
        let source = SafetensorsSource::open(model).expect("safetensors open");
        let config = qwen35::config_from_safetensors(&source).expect("read config");
        let weights = {
            let mut paro_source =
                qwen35::ParoSource::new(&source, &config).expect("ParoSource::new");
            let paro_layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(
                &mut paro_source,
                std::slice::from_mut(&mut gpu),
                &paro_layout,
            )
            .expect("load paro")
        };
        (config, weights, None::<HfqFile>)
    } else {
        let mut hfq = HfqFile::open(model).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        let weights = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
                .expect("load weights")
        };
        (config, weights, Some(hfq))
    };

    // ---- Tokenizer ----
    let metadata_json = hfq_opt
        .as_ref()
        .map(|h| h.metadata_json.as_str())
        .unwrap_or("");
    let tokenizer = if !metadata_json.is_empty() {
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(metadata_json).expect("tok")
    } else {
        // Safetensors path: read tokenizer.json from the directory
        let tok_path = model.join("tokenizer.json");
        let tok_str = std::fs::read_to_string(&tok_path)
            .unwrap_or_else(|_| panic!("missing tokenizer.json at {}", tok_path.display()));
        hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&tok_str).expect("tok")
    };

    // ---- Build prompt tokens ----
    let mut prompt_tokens: Vec<u32> = match mode.as_str() {
        "raw" => tokenizer.encode(&prompt_text),
        _ => {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user = tokenizer.encode("user");
            let asst = tokenizer.encode("assistant");
            let nl = tokenizer.encode("\n");
            let user_body = tokenizer.encode(&prompt_text);
            let mut chat = Vec::new();
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&user_body);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst);
            chat.extend_from_slice(&nl);
            if mode == "thinking" {
                chat.extend_from_slice(&tokenizer.encode("<think>"));
                chat.extend_from_slice(&nl);
            }
            chat
        }
    };
    eprintln!("prompt: {} tokens", prompt_tokens.len());

    // ---- KV / DeltaNet / scratch ----
    let kv_seq = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let max_gen = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| kv_seq.saturating_sub(prompt_tokens.len() + 8));
    let mut out: Box<dyn Write> = if let Some(ref path) = out_path {
        Box::new(std::fs::File::create(path).expect("create out"))
    } else {
        Box::new(std::io::stdout())
    };

    // ---- Prefill ----
    qwen35::forward_prefill_batch(
        &mut gpu,
        &weights,
        &config,
        &prompt_tokens,
        0,
        &mut kv_cache,
        &mut dn_state,
        &scratch,
        None,
        None,
        None,
        None,
    )
    .expect("prefill forward failed");

    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);
    writeln!(out, "{next_token}").ok();
    prompt_tokens.push(next_token);

    // ---- Decode loop ----
    for step in 0..max_gen {
        let pos = prompt_tokens.len() - 1;
        if pos >= kv_seq {
            break;
        }
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            next_token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
        writeln!(out, "{next_token}").ok();
        prompt_tokens.push(next_token);
        if next_token == config.eos_token {
            break;
        }
        if step % 500 == 0 {
            eprintln!("  step {step:4}");
        }
    }
    eprintln!("done");
}
