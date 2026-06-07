// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! collect_imatrix_native — produce a hipfire-NATIVE imatrix (HFIM).
//!
//! The AWQ scale `s[j] = rms_act[j]^α` needs `Σ_token act²[j]` per input channel
//! of each linear. Historically hipfire borrowed this from a llama.cpp `--imatrix`
//! GGUF, which carried two confounds: (1) the llama.cpp tokenizer disagrees with
//! hipfire on ~46% of token positions, and (2) the GGUF↔safetensors name remap
//! silently no-op'd AWQ on 27B-3.6 hybrid `linear_attn` names. HFIM removes both:
//! it runs hipfire's OWN f32 oracle forward + tokenizer over the calibration
//! corpus and keys the stats by the SAME canonical tensor names the quantizer
//! looks up.
//!
//! Mechanism: set `gpu.imatrix_capture = Some(..)`, then every `weight_gemv`
//! (the single chokepoint) accumulates `Σ act²` for the linear's RAW input
//! (pre-rotation, pre-AWQ-scale — the f32 path has neither), keyed by
//! `WeightTensor.name`. After the pass, serialize to HFIM.
//!
//! Usage:
//! ```text
//! collect_imatrix_native \
//!     --model <f32-oracle.hfq>  --slice <calib.txt>  --output <out.hfim> \
//!     [--n-ctx 512] [--max-chunks N]
//! ```
//! Build with `--features arch-qwen35,deltanet`.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features arch-qwen35,deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_imatrix::Imatrix;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::ImatrixCapture;
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_ctx: usize = 512;
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice" => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--n-ctx" => { n_ctx = argv[i + 1].parse().expect("--n-ctx int"); i += 2; }
            "--max-chunks" => { max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int")); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: collect_imatrix_native --model <f32-oracle.hfq> --slice <calib.txt> --output <out.hfim> [--n-ctx 512] [--max-chunks N]");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let slice = slice.expect("--slice required");
    let output = output.expect("--output required");

    // Force determinism knobs (mirror build_kld_ref_native / eval_hipfire).
    // SAFETY: single-threaded init phase.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", "f32");
    }

    // -------- load oracle model + tokenizer --------
    let mut hfq = HfqFile::open(&model).expect("open oracle model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("collect_imatrix_native: arch={} model={}", gpu.arch, model.display());
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("loaded {} layers, vocab={}, n_ctx={}", weights.layers.len(), config.vocab_size, n_ctx);

    // -------- tokenize the calibration slice with hipfire's own BPE --------
    let text = std::fs::read_to_string(&slice).expect("read slice");
    let tokens: Vec<u32> = tokenizer.encode(&text);
    eprintln!("hipfire tokenize: {} tokens from {}", tokens.len(), slice.display());

    let mut n_chunk = tokens.len() / n_ctx;
    if let Some(m) = max_chunks {
        n_chunk = n_chunk.min(m);
    }
    assert!(n_chunk >= 1, "not enough tokens for one n_ctx={n_ctx} chunk");
    let tokens: Vec<u32> = tokens[..n_chunk * n_ctx].to_vec();
    eprintln!("calibrating over {} chunks of n_ctx={} ({} tokens)", n_chunk, n_ctx, tokens.len());

    // -------- KV cache + DeltaNet + scratch (true F32 KV) --------
    let kv_max = n_ctx + 16;
    let mut kv_cache = KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
    ).expect("new_gpu f32 kv");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).expect("scratch");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn_state");

    // -------- enable capture, run the oracle forward, accumulate Σact² --------
    gpu.imatrix_capture = Some(ImatrixCapture::default());
    let t0 = Instant::now();
    let mut steps = 0u64;
    for c in 0..n_chunk {
        dn_state.reset(&mut gpu);
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, chunk[pos], pos,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch");
            steps += 1;
            if let Some(cap) = gpu.imatrix_capture.as_mut() {
                cap.n_tokens = steps;
            }
            if steps % 64 == 0 {
                let el = t0.elapsed().as_secs_f64();
                eprint!(
                    "\r  chunk {:4}/{}  step {:7}  ({:.1} tok/s)   ",
                    c + 1, n_chunk, steps, steps as f64 / el.max(1e-9)
                );
            }
        }
    }
    eprintln!();

    // -------- serialize HFIM --------
    let cap = gpu.imatrix_capture.take().expect("capture present");
    eprintln!(
        "captured {} tensors over {} forward steps in {:.1}s",
        cap.entries.len(), cap.n_tokens, t0.elapsed().as_secs_f64()
    );
    if cap.entries.is_empty() {
        eprintln!("error: no tensors captured — did the oracle load with named weights?");
        std::process::exit(1);
    }
    // Spot-check by category so a name-mismatch can't pass silently. Names are
    // the canonical safetensors form (== the quantizer's lookup key): the
    // transformer body keeps the `model.language_model.` prefix, lm_head is bare.
    for (label, suffix) in [
        ("self_attn.q_proj  (full-attn)", "self_attn.q_proj.weight"),
        ("linear_attn.in_proj_qkv     ", "linear_attn.in_proj_qkv.weight"),
        ("mlp.down_proj               ", "mlp.down_proj.weight"),
        ("lm_head                     ", "lm_head.weight"),
    ] {
        let hit = cap
            .entries
            .iter()
            .filter(|(k, _)| k.ends_with(suffix) && !k.starts_with("mtp."))
            .count();
        let example = cap
            .entries
            .iter()
            .find(|(k, _)| k.ends_with(suffix) && !k.starts_with("mtp."));
        match example {
            Some((k, (s, _))) => eprintln!("  ✓ {label}  ×{hit}  K={}  e.g. {k}", s.len()),
            None => eprintln!("  · {label}  not present (arch may differ — ok if expected)"),
        }
    }

    let im = Imatrix::from_accum(&cap.entries, cap.n_tokens);
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent");
        }
    }
    im.write_to_file(&output).expect("write HFIM");
    let sz = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "collect_imatrix_native: wrote {} ({:.1} MB, {} tensors, {} calib tokens)",
        output.display(), sz as f64 / 1e6, im.tensors.len(), im.n_tokens
    );
}
