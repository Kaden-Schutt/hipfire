// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! build_kld_ref_native_gemma4 — llama-free KLD reference producer, gemma4
//! (arch 13) arm. Mirrors `build_kld_ref_native` (the qwen35/F2 deliverable):
//! runs hipfire's OWN F32 reference oracle (`--format oracle` .hfq, all
//! weights widened bf16/f32 -> F32) forward over the eval corpus and writes
//! per-token top-K reference log-probs in the EXACT SAME HFKLDR β binary
//! format that `eval_hipfire_gemma4` consumes.
//!
//! Differences from the qwen35 arm, all gemma4-architectural:
//!   * forward is `hipfire_arch_gemma4::lowered::forward_scratch` with the
//!     dual KV (sliding F32 `new_gpu` + full asym3 `new_gpu_asym3`) exactly
//!     as `gemma4_oracle.rs` (the HF-parity-validated config). The KV setup
//!     is IDENTICAL between this ref builder and the candidate eval, so KV
//!     noise cancels and the KLD isolates weight precision.
//!   * gemma has add_bos=true: each n_ctx chunk is materialized as
//!     [BOS] + (n_ctx-1) corpus tokens (llama-perplexity add_bos semantics).
//!     The materialized chunk tokens (BOS included) are what's written to the
//!     HFKLDR token block, so the eval forwards byte-identical chunks.
//!   * no DeltaNet state — gemma4 carries no recurrent state across
//!     positions; per-chunk isolation comes from KV position overwrite.
//!
//! Chunking semantics match llama-perplexity: only the second-half window
//! [n_ctx/2 .. n_ctx-1) is scored (scored_per_chunk = n_ctx - 1 - n_ctx/2).
//!
//! Usage:
//!   build_kld_ref_native_gemma4 --model <f32-oracle.hfq> \
//!       --slice <slice.txt> --top-k 256 --n-ctx 512 \
//!       --output <name>-f32-native.kldref.bin [--max-chunks N]

fn main() {
    use hipfire_arch_gemma4::lowered::{self as gemma4, Gemma4Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::cmp::Ordering;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    const HIPFIRE_MAGIC: &[u8; 8] = b"HFKLDR\0\0";
    const HIPFIRE_VERSION: u32 = 1;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut top_k: usize = 256;
    let mut n_ctx: usize = 512;
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice" => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--top-k" => { top_k = argv[i + 1].parse().expect("--top-k int"); i += 2; }
            "--n-ctx" => { n_ctx = argv[i + 1].parse().expect("--n-ctx int"); i += 2; }
            "--max-chunks" => { max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int")); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: build_kld_ref_native_gemma4 --model <f32-oracle.hfq> --slice <txt> --output <bin> [--top-k 256] [--n-ctx 512] [--max-chunks N]");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let output = output.expect("--output required");

    // Force determinism knobs (mirror build_kld_ref_native / eval_hipfire).
    // SAFETY: single-threaded init phase.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GEMMA4_GRAPH", "0");
    }

    // -------- load oracle model + tokenizer --------
    let mut hfq = HfqFile::open(&model).expect("open oracle model");
    let config = gemma4::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("build_kld_ref_native_gemma4: arch={} model={}", gpu.arch, model.display());
    let weights = gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!(
        "loaded {} layers, vocab={}, n_ctx={}, top_k={}, bos={}",
        config.n_layers, config.vocab_size, n_ctx, top_k, config.bos_token
    );

    // -------- build the token stream --------
    let text = std::fs::read_to_string(slice.expect("--slice required"))
        .expect("read slice");
    let stream = tokenizer.encode(&text);
    eprintln!("hipfire tokenize: {} tokens from slice", stream.len());

    // Materialize BOS-prefixed chunks: chunk c = [BOS] + stream[c*(n_ctx-1)..(c+1)*(n_ctx-1)].
    let per_chunk_stream = n_ctx - 1;
    let mut n_chunk = stream.len() / per_chunk_stream;
    if let Some(m) = max_chunks {
        n_chunk = n_chunk.min(m);
    }
    assert!(n_chunk >= 1, "not enough tokens for one chunk");
    let mut tokens: Vec<u32> = Vec::with_capacity(n_chunk * n_ctx);
    for c in 0..n_chunk {
        tokens.push(config.bos_token);
        tokens.extend_from_slice(&stream[c * per_chunk_stream..(c + 1) * per_chunk_stream]);
    }
    eprintln!("chunked into {} chunks of n_ctx={} (BOS-prefixed)", n_chunk, n_ctx);

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

    // -------- scratch + dual KV (gemma4_oracle config: sliding F32, full asym3) --------
    let scratch = Gemma4Scratch::new(&mut gpu, &config, 1).expect("scratch");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim)
        .expect("init_scratch_constants");
    let kv_max = n_ctx + 16;
    let mut kv_sliding = KvCache::new_gpu(
        &mut gpu, config.n_layers, config.sliding_n_kv_heads,
        config.sliding_head_dim, kv_max,
    ).expect("kv sliding alloc");
    // FULL KV = F32, NOT asym3: on gfx942/CDNA the asym3 full-KV path is
    // catastrophically wrong (grows with depth; PPL 3826 at 512 ctx) while
    // F32 full-KV is HF-EXACT (top-5 logits match HF to 1e-4 at 128 ids,
    // 2026-06-10). F32 both sides also removes the shared KV-noise floor.
    let mut kv_full = KvCache::new_gpu(
        &mut gpu, config.n_layers, config.full_n_kv_heads,
        config.full_head_dim, kv_max,
    ).expect("kv full alloc");

    // -------- per-chunk forward + top-K reduce --------
    let k = top_k;
    let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(config.vocab_size);
    let mut nll_sum = 0.0f64;
    let mut nll_count = 0usize;
    let t0 = Instant::now();
    let mut scored_done = 0usize;

    for c in 0..n_chunk {
        // KV positions are passed explicitly via `pos` — overwriting from
        // position 0 each chunk is sufficient (same invariant the qwen35
        // eval relies on; gemma4 has no recurrent state to reset).
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            gemma4::forward_scratch(
                &mut gpu, &weights, &config, chunk[pos], pos,
                &mut kv_sliding, &mut kv_full, &scratch,
            ).expect("forward_scratch");
            if pos < scoring_start {
                continue;
            }
            let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
            let cand_logits = &cand_logits[..config.vocab_size.min(cand_logits.len())];

            // Convert logits -> full log-prob vector (fp64 log-softmax).
            let mut max_logit = f32::NEG_INFINITY;
            for &v in cand_logits.iter() { if v > max_logit { max_logit = v; } }
            let mut sum_exp = 0.0f64;
            for &v in cand_logits.iter() { sum_exp += ((v - max_logit) as f64).exp(); }
            let log_z = (max_logit as f64) + sum_exp.ln();

            // NLL on the actual next token (matches eval / llama-ppl).
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
        "build_kld_ref_native_gemma4: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
        output.display(), out_size as f64 / 1e9, scored_done, t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "build_kld_ref_native_gemma4: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
        mean_nll, ppl, nll_count
    );
}
