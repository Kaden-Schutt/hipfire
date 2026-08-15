// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! eval_hipfire_glimmer — KLD eval for muse-glimmer (arch 14) quant variants
//! against a hipfire-native F32 reference (HFKLDR β, produced by
//! `build_kld_ref_native_glimmer`).
//!
//! Mirrors `eval_hipfire_gemma4` (arch 13) with glimmer-architectural swaps:
//!   * forward is `hipfire_arch_muse_glimmer::forward::decode_step`
//!     (forward.rs:319) with the dual Q8 KV owned by
//!     `GlimmerState::new_with_max_seq` (glimmer.rs:1721). Unlike gemma4's
//!     `Gemma4Scratch + two KvCache::new_gpu(F32/asym3)` split, Glimmer's KV
//!     lives INSIDE the state (glimmer.rs:1638-1644,1721) and is allocated via
//!     `KvCache::new_gpu_q8` / `new_gpu_q8_vmm_capped_filtered`
//!     (glimmer.rs:1870,1898,1909). The two caches share uniform head_dim=128,
//!     n_kv=2 geometry and differ only in layer count (39 sliding vs 13 full)
//!     and window at dispatch (sliding 2048 vs full 0 = NoPE). The KV is
//!     created with `kv_max = n_ctx + 16` via `new_with_max_seq` so the KLD
//!     reference and candidate forward use IDENTICAL Q8 quantization (both Q8,
//!     never asym3) — Q8 noise cancels and KLD isolates weight precision.
//!     There is no F32 KV path for Glimmer; the gemma4 F32/asym3 discussion
//!     does not apply — both caches are Q8.
//!   * additionally reports top-1 agreement (candidate argmax == ref top-1).
//!   * no recurrent state — Glimmer is dense; per-chunk isolation is explicit
//!     pos overwrite from 0 (no DeltaNet/conv reset needed).
//!
//! Glimmer arch numerics (split eps 1e-5/1e-8, scale-less QK-norm +
//! qk_scale_factor 3.87, gated attention pre-o_proj, output_multiplier
//! 0.196116135, tanh softcap 20.0, 39 sliding + 13 full at rope theta
//! 500000/0) live inside the Glimmer forward; this file just calls that forward.
//!
//! Output: HFKSEQ v2 — per-sequence (mean, p99, mean_nll) fp64 triples.
//!
//! Usage:
//!   eval_hipfire_glimmer --model <quant.hfq> --ref <ref.kldref.bin> \
//!       --output <out.kldseq> [--max-chunks N]

fn main() {
    use hipfire_arch_muse_glimmer::config::GlimmerConfig;
    use hipfire_arch_muse_glimmer::forward::decode_step;
    use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--max-chunks" => { max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int")); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire_glimmer --model <hfq> --ref <kldref.bin> --output <kldseq> [--max-chunks N]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let ref_path = ref_path.expect("--ref required");
    let output = output.expect("--output required");

    // Force determinism knobs (mirror build_kld_ref_native_glimmer).
    // SAFETY: single-threaded init phase.
    // No HIPFIRE_GLIMMER*GRAPH variable exists in crates/hipfire-arch-muse-glimmer/
    // or crates/hipfire-runtime/ (grep HIPFIRE_GLIMMER*GRAPH returns nothing;
    // unlike HIPFIRE_GEMMA4_GRAPH for gemma4). Only the two generic knobs are set.
    // The Q8 KV allocator can be flipped via HIPFIRE_GLIMMER_KV_VMM; to keep the
    // KLD invariant (identical KV quantization) we pin it to the same value the
    // builder uses (VMM enabled = 1, the default when unset). See
    // glimmer.rs:1864 `use_vmm = env != "0" unwrap_or(true)`.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GLIMMER_KV_VMM", "1");
    }

    // -------- validate the reference header BEFORE touching the GPU --------
    // Magic and version are 32 cheap bytes. Checking them first means a wrong
    // --ref path or a stale format fails in milliseconds rather than after a
    // multi-minute weight upload (111 GB for the F32 oracle).
    let ref_file = File::open(&ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);

    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic: expected \"HFKLDR\\0\\0\" (bytes {:?}), found {:?}", b"HFKLDR\0\0", magic);
        std::process::exit(2);
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
        eprintln!("unsupported ref version: expected 1, found {version}");
        std::process::exit(2);
    }

    // -------- load model --------
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("eval_hipfire_glimmer: arch={} model={}", gpu.arch, model.display());
    if gpu.arch.starts_with("gfx12") {
        unsafe { std::env::set_var("HIPFIRE_LLOYD_GFX12", "1"); }
        eprintln!("eval_hipfire_glimmer: arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
    }
    let hfq = HfqFile::open(&model).expect("open model");
    let config = GlimmerConfig::from_hfq(&hfq).expect("read config");
    // Vocab cross-check BEFORE the weight upload — a mismatched reference is a
    // hard stop, so catching it here avoids uploading the whole model first.
    if ref_n_vocab != config.vocab_size {
        eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", config.vocab_size);
        std::process::exit(2);
    }
    let weights = GlimmerWeights::load(&hfq, &config, &mut gpu).expect("load weights");
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let effective_n_chunk = match max_chunks {
        Some(m) => m.min(n_chunk),
        None => n_chunk,
    };
    let total_scored = scored_per_chunk * effective_n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire_glimmer: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B"
    );

    // Read tokens (n_ctx * n_chunk u32s) — materialized chunks as written by
    // the ref builder; forwarded verbatim, never re-tokenized.
    let n_tokens = n_ctx * n_chunk;
    let mut tokens_raw = vec![0u8; n_tokens * 4];
    ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
    let tokens: Vec<u32> = tokens_raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // -------- KV via GlimmerState (identical to ref builder) --------
    // GlimmerState owns dual Q8 VMM KV (39 sliding + 13 full, head_dim 128,
    // n_kv 2). See glimmer.rs:1638-1644 and new_with_max_seq at 1721 which
    // allocates KvCache::new_gpu_q8 / new_gpu_q8_vmm_capped_filtered for both
    // caches (glimmer.rs:1870,1898,1909). kv_max = n_ctx + 16 mirrors the
    // builder so the two sides use IDENTICAL Q8 quantization — Q8 noise cancels
    // and KLD isolates weight precision. The VMM allocator is pinned to 1 above
    // so the two arms cannot silently differ via HIPFIRE_GLIMMER_KV_VMM.
    let kv_max = n_ctx + 16;
    let mut state = GlimmerState::new_with_max_seq(&mut gpu, &config, kv_max).expect("GlimmerState");

    // -------- per-chunk loop --------
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut p99_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    let mut top1_agree = 0usize;
    let mut top1_total = 0usize;
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;

    let scoring_start = n_ctx / 2;
    for c in 0..effective_n_chunk {
        // KV positions are explicit via `pos`; overwrite from 0 each chunk.
        // Glimmer is dense (no DeltaNet/conv state) so no per-chunk recurrent
        // reset is needed; per-chunk isolation comes solely from pos overwrite.
        let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
        let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
        let mut chunk_nll_sum: f64 = 0.0;
        let mut chunk_nll_count: usize = 0;

        for pos in 0..(n_ctx - 1) {
            let cand_logits = decode_step(
                &config, &weights, &mut state, &mut gpu, chunk_tokens[pos], pos as u32,
            ).expect("decode_step");
            if pos < scoring_start {
                continue;
            }

            // Read the next ref block.
            ref_in.read_exact(&mut block_buf).expect("read ref block");
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
            let sum_p_residual =
                f32::from_le_bytes(block_buf[resid_off..resid_off + 4].try_into().unwrap());

            let cand_logits = &cand_logits[..config.vocab_size.min(cand_logits.len())];

            // Candidate log-Z (fp64) + argmax for top-1 agreement.
            let mut max_logit = f32::NEG_INFINITY;
            let mut argmax = 0usize;
            for (idx, &v) in cand_logits.iter().enumerate() {
                if v > max_logit { max_logit = v; argmax = idx; }
            }
            let mut sum_exp = 0.0f64;
            for &v in cand_logits.iter() {
                sum_exp += ((v - max_logit) as f64).exp();
            }
            let log_z = (max_logit as f64) + sum_exp.ln();

            // ref top-1 is top_indices[0] (builder sorts descending).
            if argmax as u32 == top_indices[0] {
                top1_agree += 1;
            }
            top1_total += 1;

            // KLD = Σ_{i in top_K_P_ref} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
            //     + residual cross-term.
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
            chunk_klds.push(kld_token);

            let actual_next = chunk_tokens[pos + 1] as usize;
            if actual_next < cand_logits.len() {
                chunk_nll_sum += -((cand_logits[actual_next] as f64) - log_z);
                chunk_nll_count += 1;
            }

            total_scored_done += 1;
            if total_scored_done % 256 == 0 || total_scored_done == total_scored {
                let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = total_scored_done as f64 / elapsed.max(1e-9);
                eprint!(
                    "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                    c + 1, effective_n_chunk, total_scored_done, total_scored, pct, rate
                );
            }
        }

        // Per-chunk aggregates
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
        "eval_hipfire_glimmer: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s)",
        t0.elapsed().as_secs_f64(),
        total_scored_done as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );

    // -------- write HFKSEQ output (v2: adds mean_nll per chunk) --------
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    let out_file = File::create(&output).expect("create output");
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

    let overall_mean: f64 = mean_kld_per_seq.iter().copied().sum::<f64>() / mean_kld_per_seq.len() as f64;
    let nll_finite: Vec<f64> = mean_nll_per_seq.iter().copied().filter(|x| x.is_finite()).collect();
    let overall_nll: f64 = if nll_finite.is_empty() {
        f64::NAN
    } else {
        nll_finite.iter().copied().sum::<f64>() / nll_finite.len() as f64
    };
    let overall_ppl = overall_nll.exp();
    let top1_pct = if top1_total > 0 {
        top1_agree as f64 * 100.0 / top1_total as f64
    } else { f64::NAN };
    eprintln!(
        "eval_hipfire_glimmer: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4}  top1-agree = {:.2}% ({}/{})",
        overall_mean, overall_nll, overall_ppl, top1_pct, top1_agree, top1_total
    );
    eprintln!("eval_hipfire_glimmer: wrote {}", output.display());
}
