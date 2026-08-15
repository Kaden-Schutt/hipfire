// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! build_kld_ref_native_glimmer — llama-free KLD reference producer, glimmer
//! (arch 14) arm. Mirrors `build_kld_ref_native_gemma4` (the arch-13 arm) and
//! `build_kld_ref_native` (the qwen35/F2 deliverable): runs hipfire's OWN F32
//! reference oracle (`--format oracle` .hfq, all weights widened bf16/f32 -> F32)
//! forward over the eval corpus and writes per-token top-K reference log-probs
//! in the EXACT SAME HFKLDR β binary format that `eval_hipfire_glimmer`
//! consumes.
//!
//! Differences from the gemma4 arm, all glimmer-architectural:
//!   * forward is `hipfire_arch_muse_glimmer::forward::decode_step` with the
//!     dual Q8 KV (sliding + full) owned by `GlimmerState::new_with_max_seq`.
//!     Unlike gemma4's `Gemma4Scratch + two KvCache::new_gpu(F32/asym3)` split,
//!     Glimmer's KV lives INSIDE the state (see `glimmer.rs:1638-1644` and
//!     `new_with_max_seq` at `glimmer.rs:1721`). The two caches share uniform
//!     head_dim=128, n_kv=2 geometry and differ only in layer count (39 vs 13)
//!     and window at attention dispatch (sliding 2048 vs full 0). The KV shape
//!     is allocated with `kv_max = n_ctx + 16` via `new_with_max_seq` so the
//!     KLD reference and candidate forward use IDENTICAL KV quantization (both
//!     Q8, never asym3) — KV noise cancels and KLD isolates weight precision.
//!     Glimmer has no F32 KV path; `new_with_max_seq` unconditionally creates
//!     `KvCache::new_gpu_q8` / `new_gpu_q8_vmm_capped_filtered` (see
//!     `glimmer.rs:1870,1898`). Both sides must mirror this.
//!   * glimmer has BOS (bos_token 200000, `config.rs:43,75`) and the daemon's
//!     chat path explicitly prepends it (`daemon.rs:31029`:
//!     `if ids.first() != Some(&bos_tok) { ids.insert(0, bos_tok); }`) and the
//!     DFlash demo does the same (`dflash_spec_demo.rs:844`). The Glimmer
//!     crate itself carries no `add_bos` symbol; `tokenizer::from_hfq_metadata`
//!     for glimmer returns `add_bos=false` (no `tokenizer_config.add_bos_token`
//!     in the HFQ, no arch default for muse_glimmer), so raw `encode()` does
//!     NOT prepend BOS. We therefore materialize BOS-prefixed chunks exactly as
//!     gemma4 does: `[BOS] + (n_ctx-1)` corpus tokens (llama-perplexity add_bos
//!     semantics). The materialized chunk tokens (BOS included) are what is
//!     written to the HFKLDR token block, so the eval forwards byte-identical
//!     chunks. Documented here so the choice is auditable.
//!   * no DeltaNet / recurrent / conv state — glimmer is dense (52 layers,
//!     GQA, sandwich RMSNorm). Verified by inspecting `forward.rs` (`decode_step`
//!     at `forward.rs:319`, `decode_step_body`, `glimmer_layer_decode`) and
//!     `glimmer.rs` `GlimmerState` fields (`glimmer.rs:1638-1713`): only
//!     `n_tokens`, `kv_sliding`, `kv_full`, and scratch tensors; `drafter.rs`
//!     is a separate 5-layer speculative head (arch 23) not used in this path.
//!     Per-chunk isolation comes from KV position overwrite from 0 each chunk,
//!     same invariant the qwen35/gemma4 evals rely on; no per-chunk reset of
//!     recurrent state is needed.
//!   * determinism: `HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_GRAPH=0` are forced
//!     before GPU init, mirroring gemma4. A grep of
//!     `crates/hipfire-arch-muse-glimmer/` and `crates/hipfire-runtime/` for
//!     `HIPFIRE_GLIMMER*GRAPH` finds no Glimmer-specific graph env var (unlike
//!     `HIPFIRE_GEMMA4_GRAPH`); only the two generic knobs are set.
//!
//! Chunking semantics match llama-perplexity: only the second-half window
//! [n_ctx/2 .. n_ctx-1) is scored (scored_per_chunk = n_ctx - 1 - n_ctx/2).
//!
//! Usage:
//!   build_kld_ref_native_glimmer --model <f32-oracle.hfq> \
//!       --slice <slice.txt> --top-k 256 --n-ctx 512 \
//!       --output <name>-f32-native.kldref.bin [--max-chunks N]

fn main() {
    use hipfire_arch_muse_glimmer::config::GlimmerConfig;
    use hipfire_arch_muse_glimmer::forward as glimmer;
    use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
    use hipfire_runtime::hfq::HfqFile;
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
                eprintln!("Usage: build_kld_ref_native_glimmer --model <f32-oracle.hfq> --slice <txt> --output <bin> [--top-k 256] [--n-ctx 512] [--max-chunks N]");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let output = output.expect("--output required");

    // Force determinism knobs (mirror build_kld_ref_native / build_kld_ref_native_gemma4).
    // SAFETY: single-threaded init phase.
    // Grep for HIPFIRE_GLIMMER*GRAPH in crates/hipfire-arch-muse-glimmer/ and
    // crates/hipfire-runtime/ finds no Glimmer-specific graph var (unlike
    // HIPFIRE_GEMMA4_GRAPH). Only the two generic knobs are set.
    //
    // HIPFIRE_GLIMMER_KV_VMM is pinned explicitly rather than left to its
    // default. Glimmer's dual Q8 KV allocates through either
    // `new_gpu_q8_vmm_capped_filtered` or `new_gpu_q8` depending on this var
    // (glimmer.rs:1864, default enabled). The KLD invariant requires the
    // reference and `eval_hipfire_glimmer` to allocate KV IDENTICALLY so Q8
    // quantization noise cancels and the divergence isolates weight precision.
    // Leaving it implicit here while the eval pins it would silently break that
    // invariant for anyone with the var already set in their environment.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GLIMMER_KV_VMM", "1");
    }

    // -------- load oracle model + tokenizer --------
    let hfq = HfqFile::open(&model).expect("open oracle model");
    let config = GlimmerConfig::from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("build_kld_ref_native_glimmer: arch={} model={}", gpu.arch, model.display());
    let weights = GlimmerWeights::load(&hfq, &config, &mut gpu).expect("load weights");
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
    // Glimmer's daemon explicitly prepends BOS if missing (daemon.rs:31029
    // `if ids.first() != Some(&bos_tok) { ids.insert(0, bos_tok); }`) and the
    // DFlash demo mirrors it (dflash_spec_demo.rs:844). The Glimmer crate has
    // no `add_bos` symbol; `Tokenizer::from_hfq_metadata` for this HFQ returns
    // add_bos=false (no tokenizer_config.add_bos_token, no arch default for
    // muse_glimmer), so raw encode() does NOT include BOS. We therefore
    // materialize BOS-prefixed chunks, matching gemma4's add_bos=true semantics
    // and ensuring the HFKLDR token block is what the eval forwards verbatim.
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

    // -------- state + dual KV (glimmer: sliding + full, Q8, logical split) --------
    // Glimmer uses a dual-cache topology (sliding 39 layers + full 13 layers)
    // despite uniform head_dim=128 / n_kv=2 geometry; see glimmer.rs:1638-1644
    // and GlimmerState::new_with_max_seq at glimmer.rs:1721 which allocates
    // KvCache::new_gpu_q8 / new_gpu_q8_vmm_capped_filtered for both caches
    // (glimmer.rs:1870,1898,1909). There is no F32 KV path; the Q8 allocation
    // is identical on ref-builder and candidate eval, so KV quantization noise
    // cancels and KLD isolates weight precision. The gemma4 arm's F32/asym3
    // discussion (build_kld_ref_native_gemma4.rs:147-150) is not applicable
    // here — glimmer never uses asym3; both caches are Q8.
    // kv_max = n_ctx + 16 mirrors gemma4's `kv_max` sizing.
    let kv_max = n_ctx + 16;
    let mut state = GlimmerState::new_with_max_seq(&mut gpu, &config, kv_max).expect("GlimmerState alloc");

    // -------- per-chunk forward + top-K reduce --------
    let k = top_k;
    let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(config.vocab_size);
    let mut nll_sum = 0.0f64;
    let mut nll_count = 0usize;
    let t0 = Instant::now();
    let mut scored_done = 0usize;

    for c in 0..n_chunk {
        // KV positions are passed explicitly via `pos` — overwriting from
        // position 0 each chunk is sufficient. Glimmer is dense with no
        // recurrent / conv state to reset: verified by inspecting
        // forward.rs:decode_step (forward.rs:319), decode_step_body, and
        // glimmer.rs:GlimmerState fields (glimmer.rs:1638-1713) — only
        // n_tokens + kv_sliding/kv_full + scratch; drafter.rs is a separate
        // 5-layer assistant head (arch 23) not involved in this path.
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            let cand_logits = glimmer::decode_step(
                &config, &weights, &mut state, &mut gpu, chunk[pos], pos as u32,
            ).expect("decode_step");
            if pos < scoring_start {
                continue;
            }
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
        "build_kld_ref_native_glimmer: wrote {} ({:.3} GB) — {} scored tokens in {:.1}s",
        output.display(), out_size as f64 / 1e9, scored_done, t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "build_kld_ref_native_glimmer: ORACLE mean NLL = {:.6}  PPL = {:.4}  (scored window, {} tokens)",
        mean_nll, ppl, nll_count
    );
}
