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
    use hipfire_runtime::hfq::HfqFile;
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
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--slice" => {
                slice = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--top-k" => {
                top_k = argv[i + 1].parse().expect("--top-k int");
                i += 2;
            }
            "--n-ctx" => {
                n_ctx = argv[i + 1].parse().expect("--n-ctx int");
                i += 2;
            }
            "--tokenize-mode" => {
                tokenize_mode = argv[i + 1].clone();
                i += 2;
            }
            "--tokens-bin" => {
                tokens_bin = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--max-chunks" => {
                max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int"));
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: build_kld_ref_native --model <f32-oracle.hfq> --slice <txt> --output <bin> [--top-k 256] [--n-ctx 512] [--tokenize-mode hipfire|tokens-bin] [--tokens-bin <llama.bin>] [--max-chunks N]");
                std::process::exit(0);
            }
            o => {
                eprintln!("unknown arg: {o}");
                std::process::exit(1);
            }
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
        // Keep BF16 teacher in BF16 on MFMA path when the teacher is BF16.
        // Calibration runs with HIPFIRE_CALIB_BF16=1 for BF16 teachers; the
        // reference must match that path. Respect an explicit user setting,
        // otherwise default to 1 so a BF16 teacher does not silently widen.
        if std::env::var("HIPFIRE_CALIB_BF16").is_err() {
            std::env::set_var("HIPFIRE_CALIB_BF16", "1");
        }
    }

    // -------- load oracle hfq, detect arch, tokenizer, gpu --------
    let mut hfq = HfqFile::open(&model).expect("open oracle model");
    let arch_id = hfq.arch_id;
    let is_llama = matches!(arch_id, 0 | 1);
    let is_qwen35 = matches!(arch_id, 5 | 6);
    let is_gemma = arch_id == 13;
    let is_glimmer = arch_id == 14;
    let is_lfm2 = arch_id == 11;
    if !(is_llama || is_qwen35 || is_gemma || is_glimmer || is_lfm2) {
        eprintln!(
            "unsupported arch id {} (model {}): build_kld_ref_native supports dense arches 0/1 (llama), 5/6 (qwen35), 13 (gemma4), 14 (glimmer), 11 (lfm2 dense). Refusing to fall through to qwen35 to avoid misleading `tensor not found: norm.weight` panic.",
            arch_id,
            model.display()
        );
        std::process::exit(1);
    }
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "build_kld_ref_native: arch={} model={}",
        arch_id,
        model.display()
    );
    eprintln!("GPU: {}", gpu.arch);

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
        eprintln!(
            "tokens-bin: reusing {} llama tokens ({} chunks)",
            n, llama_n_chunk
        );
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

    // Shared scoring helper: arch-independent top-k + NLL + serialization.
    // Each arch only supplies `logits: &[f32]` per scored position.
    fn write_scored_position(
        logits: &[f32],
        actual_next: usize,
        k: usize,
        out: &mut BufWriter<File>,
        log_probs: &mut Vec<(u32, f32)>,
        nll_sum: &mut f64,
        nll_count: &mut usize,
    ) {
        // log-softmax in f64 for stability
        let mut max_logit = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut sum_exp = 0.0f64;
        for &v in logits.iter() {
            sum_exp += ((v - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();

        if actual_next < logits.len() {
            let lp = (logits[actual_next] as f64) - log_z;
            *nll_sum += -lp;
            *nll_count += 1;
        }

        log_probs.clear();
        for (idx, &v) in logits.iter().enumerate() {
            let lp = (v as f64 - log_z) as f32;
            log_probs.push((idx as u32, lp));
        }
        let cmp_desc =
            |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
        if k < log_probs.len() {
            log_probs.select_nth_unstable_by(k - 1, cmp_desc);
        }
        let kk = k.min(log_probs.len());
        log_probs[..kk].sort_by(cmp_desc);

        let top_p_sum: f64 = log_probs[..kk]
            .iter()
            .map(|&(_, lp)| (lp as f64).exp())
            .sum();
        let sum_p_residual = (1.0 - top_p_sum).max(0.0) as f32;

        for &(idx, _) in &log_probs[..kk] {
            out.write_all(&idx.to_le_bytes()).unwrap();
        }
        // pad if vocab < k (should not happen for 256)
        if kk < k {
            for _ in kk..k {
                out.write_all(&0u32.to_le_bytes()).unwrap();
            }
        }
        for &(_, lp) in &log_probs[..kk] {
            out.write_all(&lp.to_le_bytes()).unwrap();
        }
        if kk < k {
            for _ in kk..k {
                out.write_all(&f32::NEG_INFINITY.to_le_bytes()).unwrap();
            }
        }
        out.write_all(&sum_p_residual.to_le_bytes()).unwrap();
        out.write_all(&0f32.to_le_bytes()).unwrap(); // pad
    }

    // Dispatch per arch — each branch loads its config/weights, writes the
    // HFKLDR header with its vocab, allocates KV/scratch per calib_sweep's
    // proven geometry, then drives forward and calls the shared scorer.
    // qwen35 path is byte-identical to the pre-existing code moved into the
    // new dispatch (verified by diffing the qwen35 block).

    if is_qwen35 {
        // -------- qwen35 (arch 5/6) — preserved byte-identical --------
        use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
        use hipfire_runtime::llama::KvCache;

        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        let weights = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        eprintln!(
            "loaded {} layers, vocab={}, n_ctx={}, top_k={}",
            weights.layers.len(),
            config.vocab_size,
            n_ctx,
            top_k
        );

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
        out.write_all(&(config.vocab_size as u32).to_le_bytes())
            .unwrap();
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
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
        )
        .expect("new_gpu f32 kv");
        let scratch =
            Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).expect("scratch");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn_state");

        // -------- per-chunk forward + top-K reduce (shared scorer) --------
        let k = top_k;
        let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(config.vocab_size);
        let mut nll_sum = 0.0f64;
        let mut nll_count = 0usize;
        let t0 = Instant::now();
        let mut scored_done = 0usize;

        for c in 0..n_chunk {
            dn_state.reset(&mut gpu).expect("dn_state reset");
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            for pos in 0..(n_ctx - 1) {
                qwen35::forward_scratch(
                    &mut gpu,
                    &weights,
                    &config,
                    chunk[pos],
                    pos,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                )
                .expect("forward_scratch");
                if pos < scoring_start {
                    continue;
                }
                let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                let actual_next = chunk[pos + 1] as usize;
                write_scored_position(
                    &cand_logits,
                    actual_next,
                    k,
                    &mut out,
                    &mut log_probs,
                    &mut nll_sum,
                    &mut nll_count,
                );

                scored_done += 1;
                if scored_done % 64 == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let el = t0.elapsed().as_secs_f64();
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        scored_done as f64 / el.max(1e-9)
                    );
                }
            }
        }
        eprintln!();

        out.flush().unwrap();
        drop(out);

        let mean_nll = if nll_count > 0 {
            nll_sum / nll_count as f64
        } else {
            f64::NAN
        };
        let ppl = mean_nll.exp();
        let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
            output.display(),
            out_size as f64 / 1e9,
            scored_done,
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
            mean_nll, ppl, nll_count
        );
        let _ = Path::new("/dev/null");
    } else if is_llama {
        // -------- llama (arch 0/1) — prefill_forward / forward_scratch --------
        use hipfire_runtime::llama::{ForwardScratch, KvCache};

        let cfg = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("llama config_from_hfq");
        let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &cfg, &mut gpu)
            .expect("llama load_weights");
        eprintln!(
            "loaded llama {} layers, vocab={}, n_ctx={}, top_k={} dim={} head_dim={}",
            cfg.n_layers, cfg.vocab_size, n_ctx, top_k, cfg.dim, cfg.head_dim
        );

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
        out.write_all(&(cfg.vocab_size as u32).to_le_bytes())
            .unwrap();
        out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
        out.write_all(&(top_k as u16).to_le_bytes()).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for &t in &tokens {
            out.write_all(&t.to_le_bytes()).unwrap();
        }

        let kv_max = n_ctx + 16;
        let mut kv_cache =
            KvCache::new_gpu(&mut gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_max)
                .expect("llama kv");
        let scratch =
            ForwardScratch::new_with_max_seq(&mut gpu, &cfg, kv_max).expect("llama scratch");

        let k = top_k;
        let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(cfg.vocab_size);
        let mut nll_sum = 0.0f64;
        let mut nll_count = 0usize;
        let t0 = Instant::now();
        let mut scored_done = 0usize;

        for c in 0..n_chunk {
            kv_cache.clear_gpu(&mut gpu).expect("llama kv clear");
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            // Warm prefix via batched prefill could use forward_prefill_batch,
            // but per-token forward_scratch is the reference that exposes
            // per-position logits directly.
            for pos in 0..(n_ctx - 1) {
                // llama forward_scratch returns (token, rng) but we only need logits side-effect
                hipfire_runtime::llama::forward_scratch(
                    &mut gpu,
                    &weights,
                    &cfg,
                    chunk[pos],
                    pos,
                    &mut kv_cache,
                    &scratch,
                    0.0,
                    0.0,
                    0,
                    0,
                    0.0,
                )
                .expect("llama forward_scratch");
                if pos < scoring_start {
                    continue;
                }
                let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                let actual_next = chunk[pos + 1] as usize;
                write_scored_position(
                    &cand_logits,
                    actual_next,
                    k,
                    &mut out,
                    &mut log_probs,
                    &mut nll_sum,
                    &mut nll_count,
                );
                scored_done += 1;
                if scored_done % 64 == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let el = t0.elapsed().as_secs_f64();
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        scored_done as f64 / el.max(1e-9)
                    );
                }
            }
        }
        eprintln!();
        out.flush().unwrap();
        drop(out);
        let mean_nll = if nll_count > 0 {
            nll_sum / nll_count as f64
        } else {
            f64::NAN
        };
        let ppl = mean_nll.exp();
        let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
            output.display(),
            out_size as f64 / 1e9,
            scored_done,
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
            mean_nll, ppl, nll_count
        );
        let _ = Path::new("/dev/null");
    } else if is_gemma {
        // -------- gemma4 (arch 13) — q8 KV, sub-chunk 128, forward_prefill_batch --------
        use hipfire_arch_gemma4::lowered::{self, Gemma4Scratch};
        use hipfire_runtime::llama::KvCache;

        let cfg = lowered::config_from_hfq(&hfq).expect("gemma4 config_from_hfq");
        eprintln!(
            "gemma4 arch=13 n_layers={} dim={} vocab={} hidden={} n_ctx={} top_k={}",
            cfg.n_layers, cfg.dim, cfg.vocab_size, cfg.hidden_dim, n_ctx, top_k
        );
        if cfg.enable_moe_block {
            eprintln!("gemma4 MoE (26B-A4B) not supported for BF16 KLD dense reference; refusing.");
            std::process::exit(1);
        }
        let weights = lowered::load_weights(&mut hfq, &cfg, &mut gpu).expect("gemma4 load_weights");
        eprintln!("gemma4 loaded {} layers", cfg.n_layers);

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
        out.write_all(&(cfg.vocab_size as u32).to_le_bytes())
            .unwrap();
        out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
        out.write_all(&(top_k as u16).to_le_bytes()).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for &t in &tokens {
            out.write_all(&t.to_le_bytes()).unwrap();
        }

        let kv_max = n_ctx + 16;
        let scratch = Gemma4Scratch::new(&mut gpu, &cfg, 1).expect("gemma4 scratch");
        lowered::init_scratch_constants(&mut gpu, &scratch, cfg.full_head_dim)
            .expect("gemma4 init_scratch_constants");
        let mut kv_sliding = KvCache::new_gpu_q8(
            &mut gpu,
            cfg.n_layers,
            cfg.sliding_n_kv_heads,
            cfg.sliding_head_dim,
            kv_max,
        )
        .expect("gemma sliding q8 kv");
        let mut kv_full = KvCache::new_gpu_q8(
            &mut gpu,
            cfg.n_layers,
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            kv_max,
        )
        .expect("gemma full q8 kv");
        // An F32 cache would error with `no implementation for KvWriteF32` — proven in calib_sweep.

        let k = top_k;
        let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(cfg.vocab_size);
        let mut nll_sum = 0.0f64;
        let mut nll_count = 0usize;
        let t0 = Instant::now();
        let mut scored_done = 0usize;

        for c in 0..n_chunk {
            kv_sliding
                .clear_gpu(&mut gpu)
                .expect("gemma kv sliding clear");
            kv_full.clear_gpu(&mut gpu).expect("gemma kv full clear");
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            // Per-token decode for the entire chunk, matching qwen35's proven
            // per-token loop and calib_sweep's correctness reference (forward_scratch
            // via Gemma4Bindings, not the batched prefill). The previous
            // hybrid (batched prefill for prefix + per-token for scored window)
            // left KV/position state misaligned at the 256 boundary — see
            // lowered.rs forward_prefill_batch_v2 (batched Q/K/V + batched full
            // attention) vs forward_scratch_inner_lowered (per-token). Using
            // the single per-token path eliminates the boundary class.
            for pos in 0..(n_ctx - 1) {
                lowered::forward_scratch(
                    &mut gpu,
                    &weights,
                    &cfg,
                    chunk[pos],
                    pos,
                    &mut kv_sliding,
                    &mut kv_full,
                    &scratch,
                )
                .expect("gemma forward_scratch");
                if pos < scoring_start {
                    continue;
                }
                let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                let actual_next = chunk[pos + 1] as usize;
                write_scored_position(
                    &cand_logits,
                    actual_next,
                    k,
                    &mut out,
                    &mut log_probs,
                    &mut nll_sum,
                    &mut nll_count,
                );
                scored_done += 1;
                if scored_done % 64 == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let el = t0.elapsed().as_secs_f64();
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        scored_done as f64 / el.max(1e-9)
                    );
                }
            }
        }
        eprintln!();
        out.flush().unwrap();
        drop(out);
        let mean_nll = if nll_count > 0 {
            nll_sum / nll_count as f64
        } else {
            f64::NAN
        };
        let ppl = mean_nll.exp();
        let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
            output.display(),
            out_size as f64 / 1e9,
            scored_done,
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
            mean_nll, ppl, nll_count
        );
        let _ = Path::new("/dev/null");
    } else if is_glimmer {
        // -------- glimmer (arch 14) — prefill_with_capture, chunk 192 --------
        use hipfire_arch_muse_glimmer::config::GlimmerConfig;
        use hipfire_arch_muse_glimmer::forward;
        use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};

        let cfg = GlimmerConfig::from_hfq(&hfq).expect("glimmer config");
        eprintln!(
            "glimmer arch=14 n_layers={} dim={} vocab={} n_ctx={} top_k={} sliding_window={}",
            cfg.n_layers, cfg.dim, cfg.vocab_size, n_ctx, top_k, cfg.sliding_window
        );
        let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("glimmer load");
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
        out.write_all(&(cfg.vocab_size as u32).to_le_bytes())
            .unwrap();
        out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
        out.write_all(&(top_k as u16).to_le_bytes()).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for &t in &tokens {
            out.write_all(&t.to_le_bytes()).unwrap();
        }

        let kv_max = n_ctx + 16;
        let mut state =
            GlimmerState::new_with_max_seq(&mut gpu, &cfg, kv_max).expect("glimmer state");
        // KV is inside GlimmerState (q8 per calib_sweep); reset() clears n_tokens.

        let k = top_k;
        let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(cfg.vocab_size);
        let mut nll_sum = 0.0f64;
        let mut nll_count = 0usize;
        let t0 = Instant::now();
        let mut scored_done = 0usize;

        for c in 0..n_chunk {
            state.reset();
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            // Per-token decode for the entire chunk, matching qwen35's proven
            // per-token loop and calib_sweep's correctness reference
            // (decode_step, not the batched prefill). The previous hybrid
            // (batched prefill_with_capture for prefix + per-token for scored
            // window) misaligned KV/position at the 256 boundary — the batched
            // path's internal chunking (glimmer_prefill_chunk_size) vs the
            // manual 192 chunking produced different batch sizes than
            // calib_sweep's single-call prefill_with_capture:3650 path, and the
            // decode tail's KV was not contiguous with the prefilled prefix.
            for pos in 0..(n_ctx - 1) {
                let cand_logits = forward::decode_step(
                    &cfg, &weights, &mut state, &mut gpu, chunk[pos], pos as u32,
                )
                .expect("glimmer decode_step");
                if pos < scoring_start {
                    continue;
                }
                let actual_next = chunk[pos + 1] as usize;
                write_scored_position(
                    &cand_logits,
                    actual_next,
                    k,
                    &mut out,
                    &mut log_probs,
                    &mut nll_sum,
                    &mut nll_count,
                );
                scored_done += 1;
                if scored_done % 64 == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let el = t0.elapsed().as_secs_f64();
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        scored_done as f64 / el.max(1e-9)
                    );
                }
            }
        }
        eprintln!();
        out.flush().unwrap();
        drop(out);
        let mean_nll = if nll_count > 0 {
            nll_sum / nll_count as f64
        } else {
            f64::NAN
        };
        let ppl = mean_nll.exp();
        let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
            output.display(),
            out_size as f64 / 1e9,
            scored_done,
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
            mean_nll, ppl, nll_count
        );
        let _ = Path::new("/dev/null");
    } else if is_lfm2 {
        // -------- lfm2 (arch 11) — forward_decode_batch_lfm with explicit positions, q8 KV --------
        use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
        use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
        use hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm;
        use hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights;

        let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("lfm2 config");
        eprintln!(
            "lfm2 arch=11 n_layers={} hidden={} vocab={} head_dim={} n_ctx={} top_k={} layer_types={:?}",
            cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.head_dim, n_ctx, top_k, cfg.layer_types
        );
        if cfg.num_experts != 0 {
            eprintln!("lfm2 MoE (arch 11 num_experts={} >0) not supported for dense KLD reference (only 1.2b/350m dense).", cfg.num_experts);
            std::process::exit(1);
        }
        let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("lfm2 load");
        if let Err(e) =
            hipfire_arch_lfm2moe::forward_batch::batch_weight_formats_supported(&weights)
        {
            eprintln!("lfm2 batched weight formats unsupported: {e}");
            std::process::exit(1);
        }
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
        out.write_all(&(cfg.vocab_size as u32).to_le_bytes())
            .unwrap();
        out.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
        out.write_all(&(top_k as u16).to_le_bytes()).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for &t in &tokens {
            out.write_all(&t.to_le_bytes()).unwrap();
        }

        let kv_max = n_ctx + 16;
        let max_batch = n_ctx;
        let mut state = Lfm2DecodeBatchState::new(&mut gpu, &cfg, max_batch, kv_max, 32)
            .expect("lfm2 batch state");
        // KV is q8 and only attention layers carry KV (conv layers use per-lane conv_state) — same as calib_sweep.

        let k = top_k;
        let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(cfg.vocab_size);
        let mut nll_sum = 0.0f64;
        let mut nll_count = 0usize;
        let t0 = Instant::now();
        let mut scored_done = 0usize;

        for c in 0..n_chunk {
            state.reset(&mut gpu).expect("lfm2 state reset");
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            // LFM2 uses a single batched call for the whole chunk (0..n_ctx-1)
            // — no hybrid prefix/per-token split, so it does NOT share the
            // batched↔per-token boundary defect that gemma/glimmer had. The
            // batched path is the calibration-proven chokepoint
            // forward_decode_batch_lfm:290 → batched_proj:61 (tap3), with
            // explicit positions 0..b-1 per lane. Keeping batched here is
            // correct; per-token fallback would be slower and unnecessary.
            let b = n_ctx - 1;
            let toks = &chunk[0..b];
            let positions: Vec<usize> = (0..b).collect();
            forward_decode_batch_lfm(&mut gpu, &weights, &cfg, toks, &positions, &mut state)
                .expect("lfm2 forward_decode_batch_lfm");
            // Download batched logits: [max_batch x vocab] row-major, row `pos`
            // holds logits for `toks[pos]` at its position. We slice per scored pos.
            let all_logits = gpu
                .download_f32(&state.logits)
                .expect("lfm2 download logits");
            let vocab = cfg.vocab_size;
            for pos in scoring_start..b {
                let row_start = pos * vocab;
                let row_end = row_start + vocab;
                if row_end > all_logits.len() {
                    eprintln!(
                        "lfm2 logits buffer too small: pos={} vocab={} len={}",
                        pos,
                        vocab,
                        all_logits.len()
                    );
                    std::process::exit(1);
                }
                let cand_logits = &all_logits[row_start..row_end];
                let actual_next = chunk[pos + 1] as usize;
                write_scored_position(
                    cand_logits,
                    actual_next,
                    k,
                    &mut out,
                    &mut log_probs,
                    &mut nll_sum,
                    &mut nll_count,
                );
                scored_done += 1;
                if scored_done % 64 == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let el = t0.elapsed().as_secs_f64();
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:7}/{:7}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        scored_done as f64 / el.max(1e-9)
                    );
                }
            }
        }
        eprintln!();
        out.flush().unwrap();
        drop(out);
        let mean_nll = if nll_count > 0 {
            nll_sum / nll_count as f64
        } else {
            f64::NAN
        };
        let ppl = mean_nll.exp();
        let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "build_kld_ref_native: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
            output.display(),
            out_size as f64 / 1e9,
            scored_done,
            t0.elapsed().as_secs_f64()
        );
        eprintln!(
            "build_kld_ref_native: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
            mean_nll, ppl, nll_count
        );
        let _ = Path::new("/dev/null");
    } else {
        unreachable!("arch dispatch missing");
    }
}
