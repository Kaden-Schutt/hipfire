// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! eval_hipfire — KLD eval for hipfire quant variants against a BF16 reference.
//!
//! Loads a hipfire model, reads the slice (or pre-tokenized tokens), reads
//! the BF16 reference in hipfire β format (HFKLDR), runs forward inference
//! chunk-by-chunk over the matched eval tokens, computes per-token KLD via
//! a top-K-of-reference approximation, bins per-sequence, emits HFKSEQ
//! output that `kld_reduce.py` aggregates.
//!
//! Usage:
//!   eval_hipfire --model <path-to-hfq-model> \
//!                --ref   <path-to-hipfire-β-ref> \
//!                --output <path-to-output.kldseq> \
//!                [--variant <name>=auto-from-model-path] \
//!                [--arch <name>=auto-from-gpu] \
//!                [--kv-mode <mode>=asym3] \
//!                [--scoring-mode <per-token|prefill>=per-token]
//!
//! Scoring modes (per `docs/plans/issue-113-quant-quality-eval.md` §5):
//!   prefill:   (default, canonical since 2026-05-11) forward_prefill_batch
//!              (transformer stack batched, lm_head fan-out per scored
//!              position). ~7× wall-clock vs per-token on gfx1100/gfx1151
//!              9B Q3/Q4. Requires the model's LA dtype to be in
//!              `is_batchable_la`'s OK set; auto-falls-back to per-token
//!              inside `forward_prefill_batch` otherwise (e.g., MQ4-Lloyd,
//!              HFP4G32, MFP4G32 — no batched kernel yet).
//!   per-token: forward_scratch in a per-position loop. Historical baseline,
//!              retained for direct comparison against the 2026-05-08 kldseqs
//!              under `results/2026-05-08/per-seq/*__per-token.kldseq`.
//!
//! Output: HFKSEQ format (see kldref_format.py) — per-sequence (mean, p99)
//! KLD as fp64 pairs.
//!
//! Plan: docs/plans/issue-113-quant-quality-eval.md (rev-3.2).

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{DispatchCtx, KvCache, VMode, gemv_family, weight_gemv};
    use rdna_compute::DType;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    struct Args {
        model: PathBuf,
        ref_path: PathBuf,
        output: PathBuf,
        kv_mode: String,
        kv_v: String,
        scoring_mode: String,
        max_chunks: Option<usize>,
    }
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut kv_mode = "asym3".to_string();
    let mut kv_v = "q8".to_string();
    let mut scoring_mode = "prefill".to_string();
    let mut max_chunks: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref"   => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4" | "fwht2" | "fwht3" | "fwht4" | "f32" | "f16") {
                    eprintln!("--kv-mode must be one of: q8 asym2 asym3 asym4 fwht2 fwht3 fwht4 f32 f16 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "--kv-v" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "lloyd2" | "lloyd3" | "lloyd4") {
                    eprintln!("--kv-v must be one of: q8 lloyd2 lloyd3 lloyd4 (got {v})");
                    std::process::exit(1);
                }
                kv_v = v;
                i += 2;
            }
            "--scoring-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "per-token" | "prefill") {
                    eprintln!("--scoring-mode must be one of: per-token prefill (got {v})");
                    std::process::exit(1);
                }
                scoring_mode = v;
                i += 2;
            }
            "--max-chunks" => {
                max_chunks = Some(argv[i + 1].parse().expect("--max-chunks must be integer"));
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire --model <path> --ref <path> --output <path> [--kv-mode asym3] [--kv-v q8] [--scoring-mode prefill] [--max-chunks N]");
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
        kv_v,
        scoring_mode,
        max_chunks,
    };

    // -------- eval-mode env vars (must precede Gpu::init / forward) --------
    // Per plan §"Eval-mode hipfire flags": force OFF for prompt normalize +
    // graph capture; record kv-mode in env for downstream tooling. Logged so
    // a user reading the run output sees the override explicitly.
    //
    // Note on HIPFIRE_GRAPH=0: byte-equality between graph=0 and graph=1
    // was verified on 2026-05-08 against this binary's forward path
    // (dense Qwen3.5-9B mq4, prefill 64 tokens, kv_mode=asym3) — sha256
    // matched, 0/248320 logits differed. The plan's force-OFF is therefore
    // a determinism *style* choice, not a correctness requirement: a
    // future contributor can safely flip this to opt-out (respect a
    // pre-existing env value) for cards where graph mode would shave
    // kernel-launch overhead. On 2026-05-08's gfx1100 baseline run the
    // card was power-capped at the kernel-throughput ceiling, so graph
    // mode wouldn't have helped — but that's hardware-specific.
    // The MoE-config drift documented in
    // hipfire-arch-qwen35/src/qwen35.rs:2906-2932 still applies and is
    // already gated by `config.num_experts == 0`, so dense models are
    // unaffected.
    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
        std::env::set_var("HIPFIRE_KV_V", &args.kv_v);
        // For prefill scoring, pre-allocate the PrefillBatchScratch via
        // Qwen35Scratch's HIPFIRE_PREFILL_REUSE_PBS hook so the 1175 chunk
        // calls don't each pay 25-tensor alloc/free overhead. (Plan §M1.)
        if args.scoring_mode == "prefill" {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
        }
    }
    eprintln!(
        "eval_hipfire: forced HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 \
         HIPFIRE_KV_MODE={} HIPFIRE_KV_V={} scoring_mode={}",
        args.kv_mode, args.kv_v, args.scoring_mode
    );

    // -------- ref sha256 sanity (M1) --------
    hipfire_runtime::eval_common::verify_ref_sha256(&args.ref_path, "eval_hipfire");

    // -------- load model --------
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("eval_hipfire: arch={} model={}", gpu.arch, args.model.display());
    // gfx12 Lloyd kernels are gated by HIPFIRE_LLOYD_GFX12 (see PR #195).
    // Set if running on gfx12; harmless on other arches.
    if gpu.arch.starts_with("gfx12") {
        unsafe { std::env::set_var("HIPFIRE_LLOYD_GFX12", "1"); }
        eprintln!("eval_hipfire: arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
    }

    // Auto-route safetensors directories (ParoQuant / AWQ / HF native) — mirrors
    // daemon.rs:1500-1504. HFQ files take the canonical HFQ path below.
    let (config, weights) = if args.model.is_dir() {
        use hipfire_runtime::safetensors_source::SafetensorsSource;
        let source = SafetensorsSource::open(&args.model)
            .expect("safetensors open");
        let config = qwen35::config_from_safetensors(&source)
            .expect("config_from_safetensors");
        eprintln!("  loading via safetensors (ParoQuant path)");
        let weights = qwen35::load_weights_paroquant(&source, &config, &mut gpu)
            .expect("load_weights_paroquant");
        (config, weights)
    } else {
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
        (config, weights)
    };

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
        eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", config.vocab_size);
        std::process::exit(2);
    }
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    // Effective chunk count after --max-chunks cap (V5 / dev-smoke). Tokens
    // and ref blocks for chunks beyond the cap are read but not scored —
    // the ref-block stream advances per scored position, not per chunk, so
    // the cap is enforced at the outer loop.
    let effective_n_chunk = match args.max_chunks {
        Some(m) => m.min(n_chunk),
        None => n_chunk,
    };
    if let Some(m) = args.max_chunks {
        eprintln!("eval_hipfire: --max-chunks {m} → effective_n_chunk = {effective_n_chunk}/{n_chunk}");
    }
    let total_scored = scored_per_chunk * effective_n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B"
    );

    // Read tokens (n_ctx * n_chunk u32s).
    let n_tokens = n_ctx * n_chunk;
    let mut tokens_raw = vec![0u8; n_tokens * 4];
    ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
    let tokens: Vec<u32> = tokens_raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // -------- KV cache + DeltaNet state + scratch --------
    let kv_max = n_ctx + 16;
    // FWHT KV modes only have layer-filtered ctors; build the FA-layer mask
    // from layer_types. Filtering is KLD-neutral (DeltaNet layers never read KV).
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == qwen35::LayerType::FullAttention)
        .collect();
    let mut kv_cache = match args.kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "fwht4" => KvCache::new_gpu_fwht4_filtered(
            &mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "fwht3" => KvCache::new_gpu_fwht3_filtered(
            &mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "fwht2" => KvCache::new_gpu_fwht2_filtered(
            &mut gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        // F1-KV: true un-quantized KV. `new_gpu` allocates raw FP32 K/V with all
        // quant_* flags false. The prefill batched-FA gate (`fa_batched_ok`) requires
        // a q8/asym cache, so an F32 cache forces FA layers through the per-token
        // `run_fa_layer_body` fallback whose else-branch does raw `kv_cache_write` +
        // `attention_f32` — i.e. the cache is NOT Q8-quantized during scoring. "f16"
        // is accepted as an alias and maps to the same FP32-storage path (there is no
        // separate quantized-f16 cache ctor; FP32 is the higher-precision oracle).
        "f32" | "f16" => KvCache::new_gpu(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        other => panic!("unknown --kv-mode: {other}"),
    };
    let v_mode = match args.kv_v.as_str() {
        "q8" => VMode::Q8,
        "lloyd2" => VMode::Lloyd2,
        "lloyd3" => VMode::Lloyd3,
        "lloyd4" => VMode::Lloyd4,
        other => panic!("unknown --kv-v: {other}"),
    };
    if v_mode != VMode::Q8 {
        kv_cache.set_v_mode_realloc(&mut gpu, v_mode).unwrap();
    }
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    // DeltaNet state allocated once and reset in place per chunk. Allocating
    // per chunk leaks ~6 MB × n_la_layers/chunk because DeltaNetState has no
    // Drop impl (only an explicit free_gpu) — OOM'd at ~chunk 1013/1175 in a
    // prior gfx1100 run with 21.5 GB VRAM.
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

    // Hidden-state capture buffer for prefill mode. forward_prefill_batch
    // with per_token_hidden_out=Some(buf) writes one row per scored token
    // (post-output-norm hidden state); we then loop weight_gemv per row to
    // recover logits — the "option C / per-token GPU lm_head fan-out"
    // resolution from PRD §5 lm_head fan-out options.
    // Shape: [scored_per_chunk, dim]; ~16 MB at scored_per_chunk=1023, dim=4096.
    // `scored_per_chunk` already bound above at line ~188.
    let hidden_buf = if args.scoring_mode == "prefill" {
        Some(
            gpu.alloc_tensor(&[scored_per_chunk, config.dim], DType::F32)
                .expect("alloc hidden_buf"),
        )
    } else {
        None
    };

    // -------- per-chunk loop --------
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut p99_kld_per_seq:  Vec<f64> = Vec::with_capacity(n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;

    // Per-position KLD + NLL inner body. Reads the next ref block, downloads
    // the candidate logits from scratch.logits (caller is responsible for
    // having populated those), computes top-K-of-ref KLD with residual
    // cross-term, and returns (kld_token, optional_nll). The closure
    // explicitly mutates `ref_in` and `block_buf` so per-chunk state stays
    // outside; the rest is read-only. Same math both modes use.
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
        let sum_p_residual =
            f32::from_le_bytes(block_buf[resid_off..resid_off + 4].try_into().unwrap());

        let cand_logits = gpu.download_f32(scratch_logits).expect("download logits");

        // Candidate's log-Z = log Σ exp(logit_i) — fp64 throughout.
        let mut max_logit = f32::NEG_INFINITY;
        for &v in cand_logits.iter() { if v > max_logit { max_logit = v; } }
        let mut sum_exp = 0.0f64;
        for &v in cand_logits.iter() {
            sum_exp += ((v - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();

        // KLD = Σ_{i in top_K_P_ref} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
        //     + residual cross-term  (sum_p_residual_ref * Δlog_residual)
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
        // KLD ≥ 0 by Gibbs' inequality. Tiny negatives are fp64 roundoff on
        // ~257-term sums; >1e-9 magnitudes indicate a math bug. debug_assert
        // surfaces the latter in dev builds; release runs clamp at 0.
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
        // KvCache positions are passed explicitly via `pos` (or `start_pos`)
        // — overwriting from position 0 each chunk is sufficient.
        dn_state.reset(&mut gpu);

        let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
        let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
        let mut chunk_nll_sum: f64 = 0.0;
        let mut chunk_nll_count: usize = 0;

        if args.scoring_mode == "per-token" {
            // Canonical per-token path: forward_scratch per position; the
            // scoring window is [scoring_start, n_ctx-2] inclusive.
            for pos in 0..(n_ctx - 1) {
                qwen35::forward_scratch(
                    &mut gpu, &weights, &config, chunk_tokens[pos], pos,
                    &mut kv_cache, &mut dn_state, &scratch,
                ).expect("forward_scratch");
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
        } else {
            // Prefill mode: batch the transformer stack via two
            // forward_prefill_batch calls (prefix + scored region), then
            // weight_gemv per scored position on the captured hidden states.
            let h_buf = hidden_buf.as_ref().expect("hidden_buf in prefill mode");

            // 1. Prefix: positions [0, scoring_start), no logit capture.
            //    Writes KV positions [0, scoring_start).
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config,
                &chunk_tokens[0..scoring_start],
                0,
                &mut kv_cache, &mut dn_state, &scratch,
                None, None, None, None,
            ).expect("forward_prefill_batch prefix");

            // 2. Scored region: tokens [scoring_start, n_ctx-1) at positions
            //    [scoring_start, n_ctx-2]. Captures post-output-norm hidden
            //    state per row. The slice length is scored_per_chunk
            //    (= n_ctx - 1 - n_ctx/2) so h_buf is exactly filled.
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config,
                &chunk_tokens[scoring_start..(n_ctx - 1)],
                scoring_start,
                &mut kv_cache, &mut dn_state, &scratch,
                None, Some(h_buf), None, None,
            ).expect("forward_prefill_batch scored");

            // 3. lm_head fan-out + KLD per scored position.
            //
            // F16 fast path: when the lm_head weight is stored as F16 on
            // GPU (e.g. q8f16 models with a separate F16 lm_head), one
            // batched `gemm_f16_batched_lmhead` reads the lm_head matrix
            // ONCE for all `scored_per_chunk` positions instead of
            // `scored_per_chunk` independent GEMV calls. On gfx11 this
            // routes through the WMMA `gemm_mw16_residual_wmma` kernel;
            // on non-gfx11 it falls back to a scalar F32 GEMM. Drops eval
            // wall-clock for F16 lm_head from ~280 s/chunk to 0.8 s/chunk
            // measured on gfx1151 (Qwen3.5-0.8B per-token kv-q8).
            //
            // Other lm_head dtypes (Q8, MQ4, etc.) keep the per-position
            // weight_gemv path until a batched variant is wired for each.
            let f16_lmhead = weights.output.gpu_dtype == DType::F16;
            let batched_logits: Option<rdna_compute::GpuTensor> = if f16_lmhead {
                let alloc = gpu.alloc_tensor(
                    &[scored_per_chunk, config.vocab_size],
                    DType::F32,
                ).expect("alloc batched lm_head logits");
                gpu.gemm_f16_batched_lmhead(
                    &weights.output.buf, h_buf, &alloc,
                    config.vocab_size, config.dim, scored_per_chunk,
                ).expect("gemm_f16_batched_lmhead");
                Some(alloc)
            } else {
                None
            };

            for j in 0..scored_per_chunk {
                let logits_view = if let Some(ref all_logits) = batched_logits {
                    all_logits.sub_offset(j * config.vocab_size, config.vocab_size)
                } else {
                    let row_view = h_buf.sub_offset(j * config.dim, config.dim);
                    let ctx = DispatchCtx::new(&gpu);
                    let _ = gemv_family().run_auto(&ctx, &mut gpu, &weights.output.dispatch_ref(), &row_view, &scratch.logits)
                        .expect("gemv lm_head");
                    scratch.logits.sub_offset(0, config.vocab_size)
                };
                let pos = scoring_start + j;
                let actual_next = chunk_tokens[pos + 1] as usize;
                let (kld, nll) = score_position(
                    &mut gpu, &logits_view, &mut ref_in, &mut block_buf, actual_next,
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

            if let Some(alloc) = batched_logits {
                let _ = gpu.free_tensor(alloc);
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
        "eval_hipfire: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s)",
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
    out.write_all(&2u32.to_le_bytes()).unwrap();             // version = 2
    out.write_all(&(effective_n_chunk as u32).to_le_bytes()).unwrap(); // n_chunk (post --max-chunks)
    out.write_all(&0u32.to_le_bytes()).unwrap();             // reserved
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
    eprintln!(
        "eval_hipfire: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4}",
        overall_mean, overall_nll, overall_ppl
    );
    eprintln!("eval_hipfire: wrote {}", args.output.display());
}

// (verify_ref_sha256 now lives in hipfire_runtime::eval_common — see
// crates/hipfire-runtime/src/eval_common.rs)
