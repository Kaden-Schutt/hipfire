// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! eval_hipfire_llama — KLD eval for llama-arch hipfire models (Qwen3, Qwen2,
//! Llama / Mistral / TinyLlama) against a BF16 reference.
//!
//! Twin of `eval_hipfire` but plumbed through `hipfire_arch_llama::Llama`
//! and `hipfire_runtime::llama::{forward_scratch_embed, forward_scratch_compute}`
//! instead of the qwen35 arch. Same KLD top-K math, same HFKSEQ output
//! format, same `kld_reduce.py` consumer pipeline — only the bring-up
//! triple (config / weights / scratch) and per-position forward call
//! signature change.
//!
//! Per-token scoring only. The llama-arch `forward_prefill_batch` does
//! not (yet) expose a `per_token_hidden_out` capture hook the way the
//! qwen35 arch does, so eval's prefill-mode batched-stack + lm_head
//! fan-out trick isn't available here. Per-token mode is the canonical
//! baseline scoring path anyway (the 2026-05-08 kldseqs under
//! `results/2026-05-08/per-seq/*__per-token.kldseq` were all per-token).
//!
//! Usage:
//!   eval_hipfire_llama --model <path-to-hfq-model> \
//!                      --ref   <path-to-hipfire-β-ref> \
//!                      --output <path-to-output.kldseq> \
//!                      [--kv-mode <mode>=q8] \
//!                      [--max-chunks N]
//!
//! `--kv-mode` accepts `q8 | asym2 | asym3 | asym4`. Default is `q8` for
//! cross-arch portability — `asym3` is hard-coded to head_dim=256 in
//! `crates/hipfire-runtime/src/llama.rs` (Qwen3.5-specific), so plain-
//! Qwen3 / Llama / Mistral models with head_dim != 256 must use `q8`.
//!
//! Output format: HFKSEQ (see eval_hipfire.rs and kldref_format.py).

use hipfire_arch_llama::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    // -------- args --------
    struct Args {
        model: PathBuf,
        ref_path: PathBuf,
        output: PathBuf,
        kv_mode: String,
        max_chunks: Option<usize>,
    }
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut kv_mode = "q8".to_string();
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref"   => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4") {
                    eprintln!("--kv-mode must be one of: q8 asym2 asym3 asym4 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "--scoring-mode" => {
                let v = argv[i + 1].clone();
                if v != "per-token" {
                    eprintln!(
                        "eval_hipfire_llama supports only --scoring-mode per-token \
                         (got {v}). For prefill mode use eval_hipfire (qwen35 arch)."
                    );
                    std::process::exit(1);
                }
                i += 2;
            }
            "--max-chunks" => {
                max_chunks = Some(argv[i + 1].parse().expect("--max-chunks must be integer"));
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire_llama --model <path> --ref <path> --output <path> [--kv-mode q8] [--max-chunks N]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let args = Args {
        model: model.expect("--model required"),
        ref_path: ref_path.expect("--ref required"),
        output: output.expect("--output required"),
        kv_mode,
        max_chunks,
    };

    // -------- eval-mode env vars (must precede Gpu::init / forward) --------
    // Mirrors eval_hipfire.rs: force OFF prompt normalize + graph capture;
    // record kv-mode for downstream tooling. SAFETY: single-threaded init
    // phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
    }
    eprintln!(
        "eval_hipfire_llama: forced HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 \
         HIPFIRE_KV_MODE={} scoring_mode=per-token",
        args.kv_mode
    );

    // -------- ref sha256 sanity (shared with eval_hipfire) --------
    hipfire_runtime::eval_common::verify_ref_sha256(&args.ref_path, "eval_hipfire_llama");

    // -------- load model via Architecture trait --------
    let mut hfq = HfqFile::open(&args.model).expect("open model");
    let config = <Llama as Architecture>::config_from_hfq(&hfq)
        .expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "eval_hipfire_llama: arch={} model={}",
        gpu.arch, args.model.display()
    );
    if gpu.arch.starts_with("gfx12") {
        unsafe { std::env::set_var("HIPFIRE_LLOYD_GFX12", "1"); }
        eprintln!("eval_hipfire_llama: arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
    }
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("load weights");

    // -------- read reference (HFKLDR β) header + tokens --------
    let ref_file = File::open(&args.ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);

    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic: {magic:?}"); std::process::exit(2);
    }
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("read ref header");
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let ref_n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let top_k = u16::from_le_bytes(hdr[16..18].try_into().unwrap()) as usize;
    let _flags = u16::from_le_bytes(hdr[18..20].try_into().unwrap());
    if version != 1 {
        eprintln!("unsupported ref version {version}"); std::process::exit(2);
    }
    if ref_n_vocab != config.vocab_size {
        eprintln!(
            "vocab mismatch: ref says {ref_n_vocab}, model says {}",
            config.vocab_size
        );
        std::process::exit(2);
    }
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let effective_n_chunk = match args.max_chunks {
        Some(m) => m.min(n_chunk),
        None => n_chunk,
    };
    if let Some(m) = args.max_chunks {
        eprintln!(
            "eval_hipfire_llama: --max-chunks {m} → effective_n_chunk = {effective_n_chunk}/{n_chunk}"
        );
    }
    let total_scored = scored_per_chunk * effective_n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire_llama: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B"
    );

    let n_tokens = n_ctx * n_chunk;
    let mut tokens_raw = vec![0u8; n_tokens * 4];
    ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
    let tokens: Vec<u32> = tokens_raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // -------- KV cache + scratch --------
    let kv_max = n_ctx + 16;
    let mut kv_cache = match args.kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).unwrap(),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).unwrap(),
        other => panic!("unknown --kv-mode: {other}"),
    };
    let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).unwrap();

    // -------- per-chunk loop (per-token only) --------
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut p99_kld_per_seq:  Vec<f64> = Vec::with_capacity(n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;

    // Per-position KLD + NLL — identical math to eval_hipfire's
    // `score_position`. Reads the next ref block, downloads `scratch.logits`
    // to host, computes top-K-of-ref KLD with residual cross-term + NLL on
    // the actual next token. Per-chunk state stays outside.
    let score_position = |gpu: &mut rdna_compute::Gpu,
                          scratch_logits: &rdna_compute::GpuTensor,
                          ref_in: &mut BufReader<File>,
                          block_buf: &mut [u8],
                          actual_next: usize| -> (f64, Option<f64>) {
        ref_in.read_exact(block_buf).expect("read ref block");
        let mut top_indices: Vec<u32> = Vec::with_capacity(top_k);
        let mut top_log_probs: Vec<f32> = Vec::with_capacity(top_k);
        for j in 0..top_k {
            top_indices.push(u32::from_le_bytes(block_buf[j * 4..j * 4 + 4].try_into().unwrap()));
        }
        let lp_off = top_k * 4;
        for j in 0..top_k {
            top_log_probs.push(f32::from_le_bytes(
                block_buf[lp_off + j * 4..lp_off + j * 4 + 4].try_into().unwrap(),
            ));
        }
        let resid_off = top_k * 8;
        let sum_p_residual = f32::from_le_bytes(
            block_buf[resid_off..resid_off + 4].try_into().unwrap()
        );

        let cand_logits = gpu.download_f32(scratch_logits).expect("download logits");

        let mut max_logit = f32::NEG_INFINITY;
        for &v in cand_logits.iter() {
            if v > max_logit { max_logit = v; }
        }
        let mut sum_exp = 0.0f64;
        for &v in cand_logits.iter() {
            sum_exp += ((v - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();

        let mut kld_token = 0.0f64;
        let mut sum_p_cand_at_ref_top = 0.0f64;
        for j in 0..top_k {
            let ref_idx = top_indices[j] as usize;
            if ref_idx >= cand_logits.len() { continue; }
            let log_p_ref = top_log_probs[j] as f64;
            let log_p_cand = (cand_logits[ref_idx] as f64) - log_z;
            let p_ref = log_p_ref.exp();
            let p_cand = log_p_cand.exp();
            kld_token += p_ref * (log_p_ref - log_p_cand);
            sum_p_cand_at_ref_top += p_cand;
        }
        let sum_p_residual_ref = sum_p_residual as f64;
        let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
        if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
            kld_token += sum_p_residual_ref
                * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
        }
        debug_assert!(
            kld_token >= -1e-9,
            "negative KLD beyond fp roundoff: {kld_token}"
        );
        let kld_token = kld_token.max(0.0);

        let nll = if actual_next < cand_logits.len() {
            Some(-((cand_logits[actual_next] as f64) - log_z))
        } else {
            None
        };
        (kld_token, nll)
    };

    let scoring_start = n_ctx / 2;
    for c in 0..effective_n_chunk {
        let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
        let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
        let mut chunk_nll_sum: f64 = 0.0;
        let mut chunk_nll_count: usize = 0;

        for pos in 0..(n_ctx - 1) {
            llama::forward_scratch_embed(
                &mut gpu, &weights, &config, chunk_tokens[pos], pos, &scratch,
            ).expect("forward_scratch_embed");
            llama::forward_scratch_compute(
                &mut gpu, &weights, &config, pos, &mut kv_cache, &scratch,
            ).expect("forward_scratch_compute");
            if pos < scoring_start {
                continue;
            }
            let actual_next = chunk_tokens[pos + 1] as usize;
            let (kld, nll) = score_position(
                &mut gpu, &scratch.logits, &mut ref_in, &mut block_buf, actual_next,
            );
            chunk_klds.push(kld);
            if let Some(n) = nll {
                chunk_nll_sum += n;
                chunk_nll_count += 1;
            }
            total_scored_done += 1;
            if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = total_scored_done as f64 / elapsed.max(1e-9);
                eprint!(
                    "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                    c + 1, effective_n_chunk, total_scored_done, total_scored, pct, rate
                );
            }
        }

        if chunk_klds.is_empty() {
            mean_kld_per_seq.push(0.0);
            p99_kld_per_seq.push(0.0);
            mean_nll_per_seq.push(f64::NAN);
            continue;
        }
        let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
        let mut sorted = chunk_klds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p99 = sorted[p99_idx];
        let mean_nll = if chunk_nll_count > 0 {
            chunk_nll_sum / chunk_nll_count as f64
        } else { f64::NAN };
        mean_kld_per_seq.push(mean);
        p99_kld_per_seq.push(p99);
        mean_nll_per_seq.push(mean_nll);
    }
    eprintln!();
    eprintln!(
        "eval_hipfire_llama: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s)",
        t0.elapsed().as_secs_f64(),
        total_scored_done as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );

    // -------- write HFKSEQ output (v2: adds mean_nll per chunk) --------
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    let out_file = File::create(&args.output).expect("create output");
    let mut out = BufWriter::new(out_file);
    out.write_all(b"HFKSEQ\0\0").unwrap();
    out.write_all(&2u32.to_le_bytes()).unwrap();
    out.write_all(&(effective_n_chunk as u32).to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();
    for ((m, p), n) in mean_kld_per_seq.iter()
        .zip(p99_kld_per_seq.iter())
        .zip(mean_nll_per_seq.iter())
    {
        out.write_all(&m.to_le_bytes()).unwrap();
        out.write_all(&p.to_le_bytes()).unwrap();
        out.write_all(&n.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();

    let overall_mean: f64 = mean_kld_per_seq.iter().copied().sum::<f64>()
        / mean_kld_per_seq.len() as f64;
    let nll_finite: Vec<f64> = mean_nll_per_seq.iter().copied()
        .filter(|x| x.is_finite()).collect();
    let overall_nll: f64 = if nll_finite.is_empty() {
        f64::NAN
    } else {
        nll_finite.iter().copied().sum::<f64>() / nll_finite.len() as f64
    };
    let overall_ppl = overall_nll.exp();
    eprintln!(
        "eval_hipfire_llama: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4}",
        overall_mean, overall_nll, overall_ppl
    );
    eprintln!("eval_hipfire_llama: wrote {}", args.output.display());
}
