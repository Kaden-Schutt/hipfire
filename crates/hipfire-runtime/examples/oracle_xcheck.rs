// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! oracle_xcheck — F1 native bf16/F32 reference-oracle cross-check tool.
//!
//! Loads a hipfire .hfq model (the F32/bf16 oracle), tokenizes a prompt,
//! runs a PER-TOKEN forward (forward_scratch) over the real tokens with a
//! true un-quantized FP32 KV cache, and dumps per-position logits to a raw
//! f32 file laid out [n_pos, vocab]. Pair with llama.cpp logit dumps on the
//! IDENTICAL tokens to validate the native forward (cosine + top-1 agreement).
//!
//! Also greedily continues for HIPFIRE_GEN_STEPS tokens and decodes them as a
//! coherence signal (fluent text => the F32 forward is numerically correct).
//!
//! Usage:
//!   oracle_xcheck <model.hfq> --prompt-file <txt> --n-pos N --out <logits.f32>
//!   env: HIPFIRE_GEN_STEPS (default 16), HIPFIRE_KV (default f32; f32|q8|asym3)

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::io::Write;
    use std::path::Path;

    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!("Usage: oracle_xcheck <model.hfq> [--prompt-file <txt>] [--prompt <str>] [--n-pos N] [--out <f32>]");
        std::process::exit(1);
    }
    let model_path = argv[1].clone();
    let mut prompt_file: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut n_pos: usize = 256;
    let mut out_path: Option<String> = None;
    let mut tokens_csv: Option<String> = None;
    let mut i = 2;
    while i < argv.len() {
        match argv[i].as_str() {
            "--prompt-file" => { prompt_file = Some(argv[i+1].clone()); i += 2; }
            "--prompt"      => { prompt_str = Some(argv[i+1].clone()); i += 2; }
            "--n-pos"       => { n_pos = argv[i+1].parse().unwrap(); i += 2; }
            "--out"         => { out_path = Some(argv[i+1].clone()); i += 2; }
            "--tokens-csv"  => { tokens_csv = Some(argv[i+1].clone()); i += 2; }
            o => { eprintln!("unknown arg {o}"); std::process::exit(1); }
        }
    }
    let gen_steps: usize = std::env::var("HIPFIRE_GEN_STEPS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(16);
    let kv_mode = std::env::var("HIPFIRE_KV").unwrap_or_else(|_| "f32".to_string());

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("oracle_xcheck: arch={} kv={} n_pos={} gen_steps={}", gpu.arch, kv_mode, n_pos, gen_steps);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("loaded {} layers, vocab={}", weights.layers.len(), config.vocab_size);

    // Tokenize prompt.
    let prompt = if let Some(pf) = prompt_file {
        std::fs::read_to_string(&pf).expect("read prompt file")
    } else {
        prompt_str.unwrap_or_else(|| "The history of computing began when".to_string())
    };
    let mut tokens: Vec<u32> = if let Some(tc) = tokens_csv {
        let raw = std::fs::read_to_string(&tc).expect("read tokens csv");
        raw.trim().split(',').filter(|s| !s.is_empty())
            .map(|s| s.trim().parse::<u32>().expect("token parse")).collect()
    } else {
        tokenizer.encode(&prompt)
    };
    if tokens.len() > n_pos { tokens.truncate(n_pos); }
    let n = tokens.len();
    eprintln!("prompt tokens: {} (first 16: {:?})", n, &tokens[..n.min(16)]);

    let kv_max = (n + gen_steps + 16).max(512);
    let mut kv_cache = match kv_mode.as_str() {
        "f32" | "f16" => KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).unwrap(),
        "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).unwrap(),
        o => panic!("unknown HIPFIRE_KV {o}"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).unwrap();

    // Per-position forward; capture logits for each position.
    let mut all_logits: Vec<f32> = Vec::with_capacity(n * config.vocab_size);
    let mut argmax_tokens: Vec<u32> = Vec::with_capacity(n);
    for pos in 0..n {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tokens[pos], pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("forward_scratch");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        // argmax
        let mut am = 0usize; let mut mv = f32::NEG_INFINITY;
        for (j, &v) in logits.iter().enumerate() { if v > mv { mv = v; am = j; } }
        argmax_tokens.push(am as u32);
        all_logits.extend_from_slice(&logits);
    }
    eprintln!("forward complete over {} positions.", n);

    // Coherence signal: greedily continue from the last position.
    eprintln!("--- greedy continuation ({gen_steps} steps) ---");
    let mut next = argmax_tokens[n-1];
    let mut gen: Vec<u32> = vec![next];
    for s in 0..gen_steps {
        let pos = n + s;
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("forward_scratch gen");
        let logits = gpu.download_f32(&scratch.logits).expect("download");
        let mut am = 0usize; let mut mv = f32::NEG_INFINITY;
        for (j, &v) in logits.iter().enumerate() { if v > mv { mv = v; am = j; } }
        next = am as u32;
        gen.push(next);
    }
    eprintln!("gen token ids: {:?}", gen);
    eprintln!("gen decoded: {:?}", tokenizer.decode(&gen));

    if let Some(op) = out_path {
        let mut out = std::fs::File::create(&op).expect("create out");
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(all_logits.as_ptr() as *const u8, all_logits.len()*4)
        };
        out.write_all(bytes).expect("write");
        // header sidecar: positions, vocab, token ids
        let mut meta = std::fs::File::create(format!("{op}.meta")).expect("create meta");
        writeln!(meta, "n_pos={n}\nvocab={}\ntokens={:?}\nargmax={:?}", config.vocab_size, tokens, argmax_tokens).unwrap();
        eprintln!("wrote {} f32 logits ([{n} x {}]) to {op}", all_logits.len(), config.vocab_size);
    }
}
