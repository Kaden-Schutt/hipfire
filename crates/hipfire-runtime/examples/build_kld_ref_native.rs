// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! build_kld_ref_native — llama-free KLD reference producer.
//!
//! F2 deliverable. Replaces `build_kld_ref`'s llama-perplexity dependency:
//! runs hipfire's OWN F32 reference oracle (the F1 .hfq, all weights widened
//! bf16/f32 -> F32, true un-quantized FP32 KV) forward over the eval corpus
//! and writes per-token top-K reference log-probs in the EXACT SAME HFKLDR β
//! binary format that `eval_hipfire` already consumes (so eval_hipfire reads
//! it with NO changes).
//!
//! Why: every prior hipfire quant KLD was scored against a llama-generated
//! bf16 reference => cross-harness, carrying llama's different DeltaNet/RoPE/
//! norm port as a hidden ~0.30-0.36 nat floor (see F1 cross-check). Sourcing
//! the reference from hipfire's own F32 forward makes quant-vs-oracle clean:
//! the engine-port difference cancels (candidate and reference share the
//! identical forward path, differing only in weight precision).
//!
//! Tokenization (default `--tokenize-mode hipfire`): the slice is tokenized
//! with hipfire's OWN BPE (from the oracle .hfq metadata) and chunked into
//! n_ctx-token chunks. eval_hipfire reads tokens FROM the ref and feeds the
//! candidate forward, so the candidate is scored on the IDENTICAL token
//! stream the reference was built on — fully self-consistent, no cross-
//! tokenizer divergence.
//!
//! Alternatively `--tokens-bin <llama _logits_ dump>` reuses llama's exact
//! token IDs (header magic "_logits_") so the native reference is built on
//! the SAME positions as a llama kldref — enabling a clean per-token native-
//! vs-llama-ref delta that isolates purely the reference distribution shape
//! (the cross-engine confound).
//!
//! The forward matches llama-perplexity's chunking semantics: DeltaNet state
//! is reset per chunk, KV positions overwrite from 0 each chunk, and only the
//! second-half window [n_ctx/2 .. n_ctx-1) is scored (scored_per_chunk =
//! n_ctx - 1 - n_ctx/2). It also reports the oracle's mean NLL / PPL over the
//! scored window (Step 1 soundness number).
//!
//! Usage:
//!   build_kld_ref_native --model <f32-oracle.hfq> \
//!       --slice <slice.txt> --top-k 256 --n-ctx 512 \
//!       --output <name>-f32-native.kldref.bin \
//!       [--tokenize-mode hipfire|tokens-bin] [--tokens-bin <llama.bin>] \
//!       [--max-chunks N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::cmp::Ordering;
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const HIPFIRE_MAGIC: &[u8; 8] = b"HFKLDR\0\0";
    const HIPFIRE_VERSION: u32 = 1;
    const LLAMA_MAGIC: &[u8; 8] = b"_logits_";

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut top_k: usize = 256;
    let mut n_ctx: usize = 512;
    let mut tokenize_mode = "hipfire".to_string();
    let mut tokens_bin: Option<PathBuf> = None;
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice" => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--top-k" => { top_k = argv[i + 1].parse().expect("--top-k int"); i += 2; }
            "--n-ctx" => { n_ctx = argv[i + 1].parse().expect("--n-ctx int"); i += 2; }
            "--tokenize-mode" => { tokenize_mode = argv[i + 1].clone(); i += 2; }
            "--tokens-bin" => { tokens_bin = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--max-chunks" => { max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int")); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: build_kld_ref_native --model <f32-oracle.hfq> --slice <txt> --output <bin> [--top-k 256] [--n-ctx 512] [--tokenize-mode hipfire|tokens-bin] [--tokens-bin <llama.bin>] [--max-chunks N]");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let output = output.expect("--output required");

    // Force determinism knobs (mirror eval_hipfire).
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
    eprintln!("build_kld_ref_native: arch={} model={}", gpu.arch, model.display());
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!(
        "loaded {} layers, vocab={}, n_ctx={}, top_k={}",
        weights.layers.len(), config.vocab_size, n_ctx, top_k
    );

    // -------- build the token stream --------
    let tokens: Vec<u32> = if tokenize_mode == "tokens-bin" {
        let tb = tokens_bin.expect("--tokens-bin required when --tokenize-mode tokens-bin");
        let mut f = File::open(&tb).expect("open tokens-bin");
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).expect("read magic");
        assert_eq!(&magic, LLAMA_MAGIC, "tokens-bin not a llama _logits_ dump");
        let mut hdr = [0u8; 12];
        f.read_exact(&mut hdr).expect("read hdr");
        let llama_n_ctx = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let _n_vocab = i32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let llama_n_chunk = i32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        assert_eq!(llama_n_ctx, n_ctx, "--n-ctx must match llama dump n_ctx");
        let n = llama_n_ctx * llama_n_chunk;
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf).expect("read tokens");
        eprintln!("tokens-bin: reusing {} llama tokens ({} chunks)", n, llama_n_chunk);
        buf.chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
            .collect()
    } else {
        // Tokenize the slice with hipfire's own BPE.
        let text = std::fs::read_to_string(slice.expect("--slice required (hipfire mode)"))
            .expect("read slice");
        let toks = tokenizer.encode(&text);
        eprintln!("hipfire tokenize: {} tokens from slice", toks.len());
        toks
    };

    // Chunk into n_ctx-token chunks (drop the trailing partial chunk).
    let mut n_chunk = tokens.len() / n_ctx;
    if let Some(m) = max_chunks {
        n_chunk = n_chunk.min(m);
    }
    assert!(n_chunk >= 1, "not enough tokens for one n_ctx chunk");
    let tokens: Vec<u32> = tokens[..n_chunk * n_ctx].to_vec();
    eprintln!("chunked into {} chunks of n_ctx={}", n_chunk, n_ctx);

    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let scoring_start = n_ctx / 2;
    let total_scored = scored_per_chunk * n_chunk;

    // -------- open output, write HFKLDR header + tokens --------
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent");
        }
    }
    let out_file = File::create(&output).expect("create output");
    let mut out = BufWriter::with_capacity(4 * 1024 * 1024, out_file);
    out.write_all(HIPFIRE_MAGIC).unwrap();
    out.write_all(&HIPFIRE_VERSION.to_le_bytes()).unwrap();
    out.write_all(&(n_ctx as u32).to_le_bytes()).unwrap();
    out.write_all(&(config.vocab_size as u32).to_le_bytes()).unwrap();
    out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
    out.write_all(&(top_k as u16).to_le_bytes()).unwrap();
    out.write_all(&0u16.to_le_bytes()).unwrap(); // flags
    out.write_all(&0u32.to_le_bytes()).unwrap(); // reserved
    for &t in &tokens {
        out.write_all(&t.to_le_bytes()).unwrap();
    }

    // -------- KV cache + DeltaNet + scratch (true F32 KV, like F1-KV) --------
    let kv_max = n_ctx + 16;
    let mut kv_cache = KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
    ).expect("new_gpu f32 kv");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).expect("scratch");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn_state");

    // -------- per-chunk forward + top-K reduce --------
    let k = top_k;
    let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(config.vocab_size);
    let mut nll_sum = 0.0f64;
    let mut nll_count = 0usize;
    let t0 = Instant::now();
    let mut scored_done = 0usize;

    for c in 0..n_chunk {
        dn_state.reset(&mut gpu);
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, chunk[pos], pos,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch");
            if pos < scoring_start {
                continue;
            }
            let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");

            // Convert logits -> full log-prob vector (fp64 log-softmax).
            let mut max_logit = f32::NEG_INFINITY;
            for &v in cand_logits.iter() { if v > max_logit { max_logit = v; } }
            let mut sum_exp = 0.0f64;
            for &v in cand_logits.iter() { sum_exp += ((v - max_logit) as f64).exp(); }
            let log_z = (max_logit as f64) + sum_exp.ln();

            // NLL on the actual next token (matches eval_hipfire / llama-ppl).
            let actual_next = chunk[pos + 1] as usize;
            if actual_next < cand_logits.len() {
                let lp = (cand_logits[actual_next] as f64) - log_z;
                nll_sum += -lp;
                nll_count += 1;
            }

            // top-K reduce on log-probs.
            log_probs.clear();
            for (idx, &v) in cand_logits.iter().enumerate() {
                let lp = (v as f64 - log_z) as f32;
                log_probs.push((idx as u32, lp));
            }
            let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            };
            if k < log_probs.len() {
                log_probs.select_nth_unstable_by(k - 1, cmp_desc);
            }
            log_probs[..k].sort_by(cmp_desc);

            let top_p_sum: f64 = log_probs[..k]
                .iter()
                .map(|&(_, lp)| (lp as f64).exp())
                .sum();
            let sum_p_residual = (1.0 - top_p_sum).max(0.0) as f32;

            for &(idx, _) in &log_probs[..k] {
                out.write_all(&idx.to_le_bytes()).unwrap();
            }
            for &(_, lp) in &log_probs[..k] {
                out.write_all(&lp.to_le_bytes()).unwrap();
            }
            out.write_all(&sum_p_residual.to_le_bytes()).unwrap();
            out.write_all(&0f32.to_le_bytes()).unwrap(); // pad

            scored_done += 1;
            if scored_done % 64 == 0 || scored_done == total_scored {
                let pct = scored_done as f64 * 100.0 / total_scored as f64;
                let el = t0.elapsed().as_secs_f64();
                eprint!(
                    "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                    c + 1, n_chunk, scored_done, total_scored, pct,
                    scored_done as f64 / el.max(1e-9)
                );
            }
        }
    }
    eprintln!();

    out.flush().unwrap();
    drop(out);

    let mean_nll = if nll_count > 0 { nll_sum / nll_count as f64 } else { f64::NAN };
    let ppl = mean_nll.exp();
    let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
        output.display(), out_size as f64 / 1e9, scored_done, t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
        mean_nll, ppl, nll_count
    );
    let _ = Path::new("/dev/null");
}
