// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! eval_hipfire_fullvocab — FULL-VOCAB KLD of a hipfire quant vs the hipfire
//! F32 oracle, with NO reference file and NO top-K approximation.
//!
//! F3 refinement (matched-harness program). The existing `eval_hipfire` scores
//! a candidate against the native HFKLDR ref, whose distribution is stored only
//! as the oracle's top-256 log-probs + a residual cross-term — i.e. a top-K
//! approximation of KL(P_oracle || P_cand). To make a hipfire candidate's KLD
//! directly comparable to llama-perplexity's FULL-VOCAB `--kl-divergence`
//! (which sums over the entire vocabulary), this tool runs BOTH the F32 oracle
//! AND the quant candidate forward over the SAME token stream and computes the
//! exact full-vocab KL per scored position:
//!
//!     KL = Σ_v P_oracle(v) · (log P_oracle(v) − log P_cand(v))      (fp64)
//!
//! summed over ALL `vocab_size` entries. No GGUF anywhere — this is purely
//! hipfire's own oracle vs hipfire's own quant, the clean self-consistent side
//! of the matched comparison.
//!
//! Token stream: read FROM an HFKLDR ref (so it sits on the IDENTICAL tokens
//! eval_hipfire/build_kld_ref_native used). Only the ref's header + token block
//! are consumed; the stored top-K blocks are skipped (we recompute the oracle
//! distribution exactly from the oracle forward).
//!
//! Scoring window + chunking match build_kld_ref_native exactly: DeltaNet reset
//! per chunk, KV from pos 0 per chunk, score the second-half window
//! [n_ctx/2 .. n_ctx-1). Both models use true FP32 KV (the f32 oracle regime).
//!
//! Usage:
//!   eval_hipfire_fullvocab --oracle <f32-oracle.hfq> --candidate <quant.hfq|dir> \
//!       --ref <hfkldr.bin> [--max-chunks N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features arch-qwen35,deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut oracle: Option<PathBuf> = None;
    let mut candidate: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--oracle" => { oracle = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--candidate" => { candidate = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--max-chunks" => { max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int")); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire_fullvocab --oracle <f32.hfq> --candidate <quant.hfq> --ref <hfkldr.bin> [--max-chunks N]");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {o}"); std::process::exit(1); }
        }
    }
    let oracle = oracle.expect("--oracle required");
    let candidate = candidate.expect("--candidate required");
    let ref_path = ref_path.expect("--ref required");

    // Determinism knobs (mirror build_kld_ref_native / eval_hipfire).
    // SAFETY: single-threaded init phase.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", "f32");
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("eval_hipfire_fullvocab: arch={}", gpu.arch);

    // -------- read ref header + tokens (tokens only; skip the stored blocks) --------
    let ref_file = File::open(&ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);
    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    assert_eq!(&magic, b"HFKLDR\0\0", "bad ref magic");
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("read ref header");
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let ref_n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_chunk_total = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let n_chunk = match max_chunks { Some(m) => m.min(n_chunk_total), None => n_chunk_total };
    let n_tokens = n_ctx * n_chunk_total;
    let mut tok_raw = vec![0u8; n_tokens * 4];
    ref_in.read_exact(&mut tok_raw).expect("read ref tokens");
    let tokens: Vec<u32> = tok_raw.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    drop(ref_in);

    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let scoring_start = n_ctx / 2;
    let total_scored = scored_per_chunk * n_chunk;
    eprintln!("ref: n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk(total)={n_chunk_total} eval_chunks={n_chunk} scored={total_scored}");

    // -------- load BOTH models --------
    let mut hfq_o = HfqFile::open(&oracle).expect("open oracle");
    let config = qwen35::config_from_hfq(&hfq_o).expect("oracle config");
    assert_eq!(config.vocab_size, ref_n_vocab, "oracle vocab != ref vocab");
    let weights_o = qwen35::load_weights(&mut hfq_o, &config, &mut gpu).expect("load oracle");
    eprintln!("loaded oracle ({} layers)", weights_o.layers.len());

    let (cfg_c, weights_c) = if candidate.is_dir() {
        use hipfire_runtime::safetensors_source::SafetensorsSource;
        let source = SafetensorsSource::open(&candidate).expect("safetensors open");
        let cfg_c = qwen35::config_from_safetensors(&source).expect("cand config");
        let w = qwen35::load_weights_paroquant(&source, &cfg_c, &mut gpu).expect("load cand paroquant");
        (cfg_c, w)
    } else {
        let mut hfq_c = HfqFile::open(&candidate).expect("open candidate");
        let cfg_c = qwen35::config_from_hfq(&hfq_c).expect("cand config");
        let w = qwen35::load_weights(&mut hfq_c, &cfg_c, &mut gpu).expect("load cand");
        (cfg_c, w)
    };
    assert_eq!(cfg_c.vocab_size, config.vocab_size, "candidate vocab mismatch");
    eprintln!("loaded candidate ({} layers)", weights_c.layers.len());

    // -------- two KV caches / DeltaNet states / scratches (true FP32 KV) --------
    let kv_max = n_ctx + 16;
    let mut kv_o = KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).expect("kv_o");
    let mut kv_c = KvCache::new_gpu(&mut gpu, cfg_c.n_layers, cfg_c.n_kv_heads, cfg_c.head_dim, kv_max).expect("kv_c");
    let scratch_o = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).expect("scratch_o");
    let scratch_c = Qwen35Scratch::new_with_kv_max(&mut gpu, &cfg_c, 128, kv_max).expect("scratch_c");
    let mut dn_o = DeltaNetState::new(&mut gpu, &config).expect("dn_o");
    let mut dn_c = DeltaNetState::new(&mut gpu, &cfg_c).expect("dn_c");

    let t0 = Instant::now();
    let mut kld_sum = 0.0f64;
    let mut nll_sum = 0.0f64;
    let mut scored_done = 0usize;

    // log-softmax in fp64 -> returns (log_probs: Vec<f64>, log_z, max_logit).
    let log_probs_f64 = |logits: &[f32]| -> Vec<f64> {
        let mut max_l = f32::NEG_INFINITY;
        for &v in logits { if v > max_l { max_l = v; } }
        let mut sum_exp = 0.0f64;
        for &v in logits { sum_exp += ((v - max_l) as f64).exp(); }
        let log_z = (max_l as f64) + sum_exp.ln();
        logits.iter().map(|&v| (v as f64) - log_z).collect()
    };

    for c in 0..n_chunk {
        dn_o.reset(&mut gpu);
        dn_c.reset(&mut gpu);
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            qwen35::forward_scratch(&mut gpu, &weights_o, &config, chunk[pos], pos, &mut kv_o, &mut dn_o, &scratch_o).expect("fwd oracle");
            qwen35::forward_scratch(&mut gpu, &weights_c, &cfg_c, chunk[pos], pos, &mut kv_c, &mut dn_c, &scratch_c).expect("fwd cand");
            if pos < scoring_start { continue; }
            let lo = gpu.download_f32(&scratch_o.logits).expect("dl oracle logits");
            let lc = gpu.download_f32(&scratch_c.logits).expect("dl cand logits");
            let lp_o = log_probs_f64(&lo);
            let lp_c = log_probs_f64(&lc);
            // Full-vocab KL(P_oracle || P_cand).
            let mut kl = 0.0f64;
            for v in 0..config.vocab_size {
                let p = lp_o[v].exp();
                if p > 0.0 {
                    kl += p * (lp_o[v] - lp_c[v]);
                }
            }
            kld_sum += kl.max(0.0);
            // candidate NLL on actual next token (sanity / PPL parity).
            let actual = chunk[pos + 1] as usize;
            if actual < lp_c.len() { nll_sum += -lp_c[actual]; }
            scored_done += 1;
            if scored_done % 256 == 0 || scored_done == total_scored {
                let el = t0.elapsed().as_secs_f64();
                eprint!("\r  chunk {:4}/{}  scored {:7}/{:7}  ({:.0} tok/s)  KLD~{:.6}   ",
                    c + 1, n_chunk, scored_done, total_scored, scored_done as f64 / el.max(1e-9),
                    kld_sum / scored_done as f64);
            }
        }
    }
    eprintln!();
    let mean_kld = kld_sum / scored_done as f64;
    let mean_nll = nll_sum / scored_done as f64;
    eprintln!("eval_hipfire_fullvocab: FULL-VOCAB KLD = {:.6}  cand mean NLL = {:.6}  cand PPL = {:.4}  ({} scored, {:.1}s)",
        mean_kld, mean_nll, mean_nll.exp(), scored_done, t0.elapsed().as_secs_f64());
}
