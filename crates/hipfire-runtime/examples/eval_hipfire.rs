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
//!
//! Multi-arch dispatch (Tier-1): mirrors `calib_sweep.rs` arch detection
//!   llama 0|1, qwen35 5|6, gemma4 13, glimmer 14, lfm2 11.
//!   Per-arch chunking mirrors calib_sweep:
//!     gemma4 sub-chunks to 128 (scratch.max_prefill_batch) and needs q8 KV
//!     glimmer chunks 192 (glimmer_prefill_chunk_size / prefill_with_capture)
//!     lfm2 uses forward_decode_batch_lfm with explicit positions, q8 KV
//!     qwen35 uses forward_prefill_batch with q8 KV at seq_len+16

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{KvCache, VMode};
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
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--ref" => {
                ref_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(
                    v.as_str(),
                    "q8" | "asym2"
                        | "asym3"
                        | "asym4"
                        | "fwht2"
                        | "fwht3"
                        | "fwht4"
                        | "f32"
                        | "f16"
                ) {
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
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
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

    // -------- GPU init --------
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "eval_hipfire: arch={} model={}",
        gpu.arch,
        args.model.display()
    );
    // gfx12 Lloyd kernels are gated by HIPFIRE_LLOYD_GFX12 (see PR #195).
    // Set if running on gfx12; harmless on other arches.
    if gpu.arch.starts_with("gfx12") {
        unsafe {
            std::env::set_var("HIPFIRE_LLOYD_GFX12", "1");
        }
        eprintln!("eval_hipfire: arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
    }

    // -------- arch detection (mirrors calib_sweep.rs:654-658) --------
    // HFQ header is authoritative for file models; for safetensors dirs we
    // probe SafetensorsSource. This is byte-identical to calib_sweep's
    // `arch = arg("--arch").unwrap_or(hfq.arch_id)` dispatch.
    let arch_id: u32 = if args.model.is_dir() {
        let src = hipfire_runtime::safetensors_source::SafetensorsSource::open(&args.model)
            .unwrap_or_else(|e| {
                eprintln!("safetensors open {}: {e}", args.model.display());
                std::process::exit(1)
            });
        let id = src.arch_id();
        eprintln!("eval_hipfire: detected arch_id={id} (safetensors dir)");
        id
    } else {
        let hfq = HfqFile::open(&args.model).unwrap_or_else(|e| {
            eprintln!("open model {}: {e}", args.model.display());
            std::process::exit(1)
        });
        let id = hfq.arch_id;
        eprintln!("eval_hipfire: detected arch_id={id} (HFQ header)");
        id
    };
    let is_llama = matches!(arch_id, 0 | 1);
    let is_qwen35 = matches!(arch_id, 5 | 6);
    let is_gemma = arch_id == 13;
    let is_glimmer = arch_id == 14;
    let is_lfm = arch_id == 11;
    if !(is_llama || is_qwen35 || is_gemma || is_glimmer || is_lfm) {
        eprintln!(
            "unsupported arch_id {arch_id}: eval_hipfire wires dense 0|1 (llama), 5|6 (qwen35), 13 (gemma4), 14 (glimmer), 11 (lfm2) — matches calib_sweep.rs:658"
        );
        std::process::exit(1);
    }

    // -------- shared KLD helper (arch-independent) --------
    // Factor so each arch only supplies logits per scored position and the
    // comparison/reporting stays shared. Do NOT duplicate this math per arch;
    // a divergent copy would silently make two models' numbers incomparable.
    // Mirrors eval_hipfire's original score_position closure verbatim.
    fn kld_from_block(
        logits: &[f32],
        block_buf: &[u8],
        top_k: usize,
        actual_next: usize,
    ) -> (f64, Option<f64>) {
        // parse block: [top_k u32 idx][top_k f32 logp][f32 sum_p_residual][f32 pad]
        let mut top_indices: Vec<u32> = Vec::with_capacity(top_k);
        let mut top_log_probs: Vec<f32> = Vec::with_capacity(top_k);
        for j in 0..top_k {
            top_indices.push(u32::from_le_bytes(
                block_buf[j * 4..j * 4 + 4].try_into().unwrap(),
            ));
        }
        let lp_off = top_k * 4;
        for j in 0..top_k {
            top_log_probs.push(f32::from_le_bytes(
                block_buf[lp_off + j * 4..lp_off + j * 4 + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
        let resid_off = top_k * 8;
        let sum_p_residual =
            f32::from_le_bytes(block_buf[resid_off..resid_off + 4].try_into().unwrap());

        // Candidate's log-Z = log Σ exp(logit_i) — fp64 throughout.
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

        // KLD = Σ_{i in top_K_P_ref} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
        //     + residual cross-term  (sum_p_residual_ref * Δlog_residual)
        let mut kld_token = 0.0f64;
        let mut sum_p_cand_at_ref_top = 0.0f64;
        for j in 0..top_k {
            let ref_idx = top_indices[j] as usize;
            if ref_idx >= logits.len() {
                continue;
            }
            let log_p_ref = top_log_probs[j] as f64;
            let log_p_cand = (logits[ref_idx] as f64) - log_z;
            let p_ref = log_p_ref.exp();
            let p_cand = log_p_cand.exp();
            kld_token += p_ref * (log_p_ref - log_p_cand);
            sum_p_cand_at_ref_top += p_cand;
        }
        let sum_p_residual_ref = sum_p_residual as f64;
        let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
        if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
            kld_token += sum_p_residual_ref * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
        }
        // KLD ≥ 0 by Gibbs' inequality. Tiny negatives are fp64 roundoff on
        // ~257-term sums; >1e-9 magnitudes indicate a math bug. debug_assert
        // surfaces the latter in dev builds; release runs clamp at 0.
        debug_assert!(
            kld_token >= -1e-9,
            "negative KLD beyond fp roundoff: {kld_token}"
        );
        let kld_token = kld_token.max(0.0);

        let nll = if actual_next < logits.len() {
            Some(-((logits[actual_next] as f64) - log_z))
        } else {
            None
        };
        (kld_token, nll)
    }

    // -------- helper to read ref header (shared) --------
    // Open once; after header the file cursor sits at the token stream.
    // Each arch branch will validate vocab, read tokens, and drive scoring.
    let ref_file = File::open(&args.ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);
    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic: {magic:?}");
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
        eprintln!("unsupported ref version {version}");
        std::process::exit(2);
    }
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    let effective_n_chunk = match args.max_chunks {
        Some(m) => m.min(n_chunk),
        None => n_chunk,
    };
    if let Some(m) = args.max_chunks {
        eprintln!(
            "eval_hipfire: --max-chunks {m} → effective_n_chunk = {effective_n_chunk}/{n_chunk}"
        );
    }
    let total_scored = scored_per_chunk * effective_n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B  arch_id={arch_id}"
    );

    // Shared output accumulators; arch branches fill them.
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut p99_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;
    let scoring_start = n_ctx / 2;

    // ===== Arch dispatch (mirrors calib_sweep.rs:684ff) =====
    if is_qwen35 {
        // ----- qwen35 5|6 — WIRED (preserved byte-identical behavior) -----
        // Quant dtypes admitted: MQ4G256 (campaign --format mq4), MQ8G256,
        // HFQ4G256, Q8_0, F32, BF16 (via HIPFIRE_CALIB_BF16 / F32 widen). The
        // loader (`qwen35::load_weights`) matches quant_type 13->MQ4G256 etc and
        // respects calib_force_bf16 for teachers.
        use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};

        // Auto-route safetensors directories (ParoQuant / AWQ / HF native) — mirrors
        // daemon.rs:1500-1504. HFQ files take the canonical HFQ path below.
        let (config, weights) = if args.model.is_dir() {
            use hipfire_runtime::safetensors_source::SafetensorsSource;
            let source = SafetensorsSource::open(&args.model).expect("safetensors open");
            let config = qwen35::config_from_safetensors(&source).expect("config_from_safetensors");
            eprintln!("  loading via safetensors (ParoQuant path) arch_id={arch_id}");
            let mut paro_source = qwen35::ParoSource::new(&source, &config).expect("ParoSource::new");
            let paro_layout = qwen35::Layout::single(config.n_layers);
            let weights = qwen35::load_weights(
                &mut paro_source,
                std::slice::from_mut(&mut gpu),
                &paro_layout,
            )
            .expect("load_weights");
            (config, weights)
        } else {
            let mut hfq = HfqFile::open(&args.model).expect("open model");
            let config = qwen35::config_from_hfq(&hfq).expect("read config");
            let weights = {
                let mut src = qwen35::HfqSource::new(&mut hfq, &config);
                let layout = qwen35::Layout::single(config.n_layers);
                qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
            }
            .expect("load weights");
            (config, weights)
        };
        if ref_n_vocab != config.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                config.vocab_size
            );
            std::process::exit(2);
        }
        // Read tokens (n_ctx * n_chunk u32s).
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // -------- KV cache + DeltaNet state + scratch --------
        // qwen35 uses forward_prefill_batch with a q8 KV at seq_len+16 (calib_sweep qwen35 arm: seq_len+16, KvCache::new_gpu_q8).
        let kv_max = n_ctx + 16;
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();
        let mut kv_cache = match args.kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym4" => KvCache::new_gpu_asym4(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym2" => KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht4" => KvCache::new_gpu_fwht4_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht3" => KvCache::new_gpu_fwht3_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht2" => KvCache::new_gpu_fwht2_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "f32" | "f16" => KvCache::new_gpu(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
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
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

        let hidden_buf = if args.scoring_mode == "prefill" {
            Some(
                gpu.alloc_tensor(&[scored_per_chunk, config.dim], DType::F32)
                    .expect("alloc hidden_buf"),
            )
        } else {
            None
        };

        // -------- per-chunk loop (preserved byte-identical) --------
        for c in 0..effective_n_chunk {
            dn_state.reset(&mut gpu);
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "per-token" {
                for pos in 0..(n_ctx - 1) {
                    qwen35::forward_scratch(
                        &mut gpu,
                        &weights,
                        &config,
                        chunk_tokens[pos],
                        pos,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                    )
                    .expect("forward_scratch");
                    if pos < scoring_start {
                        continue;
                    }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
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
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            } else {
                // Prefill mode: batch the transformer stack via two
                // forward_prefill_batch calls (prefix + scored region), then
                // weight_gemv per scored position on the captured hidden states.
                let h_buf = hidden_buf.as_ref().expect("hidden_buf in prefill mode");
                qwen35::forward_prefill_batch(
                    &mut gpu,
                    &weights,
                    &config,
                    &chunk_tokens[0..scoring_start],
                    0,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("forward_prefill_batch prefix");
                qwen35::forward_prefill_batch(
                    &mut gpu,
                    &weights,
                    &config,
                    &chunk_tokens[scoring_start..(n_ctx - 1)],
                    scoring_start,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                    None,
                    Some(h_buf),
                    None,
                    None,
                )
                .expect("forward_prefill_batch scored");

                let f16_lmhead = weights.output.gpu_dtype == DType::F16;
                let batched_logits: Option<rdna_compute::GpuTensor> = if f16_lmhead {
                    let alloc = gpu
                        .alloc_tensor(&[scored_per_chunk, config.vocab_size], DType::F32)
                        .expect("alloc batched lm_head logits");
                    gpu.gemm_f16_batched_lmhead(
                        &weights.output.buf,
                        h_buf,
                        &alloc,
                        config.vocab_size,
                        config.dim,
                        scored_per_chunk,
                    )
                    .expect("gemm_f16_batched_lmhead");
                    Some(alloc)
                } else {
                    None
                };

                for j in 0..scored_per_chunk {
                    let logits_view = if let Some(ref all_logits) = batched_logits {
                        all_logits.sub_offset(j * config.vocab_size, config.vocab_size)
                    } else {
                        let row_view = h_buf.sub_offset(j * config.dim, config.dim);
                        let ctx = hipfire_runtime::llama::DispatchCtx::new(&gpu);
                        let _ = hipfire_runtime::llama::gemv_family()
                            .run_auto(
                                &ctx,
                                &mut gpu,
                                &weights.output.dispatch_ref(),
                                &row_view,
                                &scratch.logits,
                            )
                            .expect("gemv lm_head");
                        scratch.logits.sub_offset(0, config.vocab_size)
                    };
                    let pos = scoring_start + j;
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&logits_view).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
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
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
                if let Some(alloc) = batched_logits {
                    let _ = gpu.free_tensor(alloc);
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
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        // qwen35 owns hidden_buf drop; keep eprintln identity
        drop(hidden_buf);
        if let Ok(p) = std::env::var("HIPFIRE_MOE_EXPERT_STATS_OUT") {
            qwen35::dump_expert_stats(&p);
        }
    } else if is_llama {
        // ----- llama 0|1 — WIRED (mirrors eval_hipfire_llama.rs) -----
        // Quant dtypes admitted: Q4K, HFQ4G256, HFQ4G128, HFQ6G256, HFQ2/3,
        // MQ4G256 (campaign mq4), MQ8G256, MQ6G256, etc. plus F32 teacher.
        // The Llama loader (`hipfire_arch_llama::Llama`) accepts all HFQ quant
        // types (see hipfire-runtime/src/hfq.rs quant_type match); no hard filter.
        use hipfire_arch_llama::Llama;
        use hipfire_runtime::arch::Architecture;
        use hipfire_runtime::llama;

        if args.model.is_dir() {
            eprintln!("eval_hipfire: llama dir models not yet wired — use HFQ for arch 0|1");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("read config");
        if ref_n_vocab != config.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                config.vocab_size
            );
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let kv_max = n_ctx + 16;
        let mut kv_cache = match args.kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym4" => KvCache::new_gpu_asym4(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym2" => KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "f32" | "f16" => KvCache::new_gpu(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            other => {
                eprintln!("llama: --kv-mode {other} not in q8/asym2/3/4/f32");
                std::process::exit(1);
            }
        };
        let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).unwrap();
        let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
        if args.scoring_mode == "prefill" {
            eprintln!("eval_hipfire: llama arch prefill not wired — falling back to per-token (prefill hook absent on llama arch, see eval_hipfire_llama.rs)");
        }
        for c in 0..effective_n_chunk {
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;
            kv_cache.clear_gpu(&mut gpu).expect("kv clear");
            for pos in 0..(n_ctx - 1) {
                llama::forward_scratch_embed(
                    &mut gpu,
                    &weights,
                    &config,
                    chunk_tokens[pos],
                    pos,
                    &scratch,
                )
                .expect("forward_scratch_embed");
                llama::forward_scratch_compute(
                    &mut gpu,
                    &weights,
                    &config,
                    pos,
                    &mut kv_cache,
                    &scratch,
                )
                .expect("forward_scratch_compute");
                if pos < scoring_start {
                    continue;
                }
                let actual_next = chunk_tokens[pos + 1] as usize;
                ref_in.read_exact(&mut block_buf).expect("read ref block");
                let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
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
                        c + 1,
                        effective_n_chunk,
                        total_scored_done,
                        total_scored,
                        pct,
                        rate
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
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        // ----- gemma4 13 — WIRED -----
        // Quant dtypes admitted: MQ4G256 / MG4G256 (campaign mq4, quant_type 13/30 alias),
        // HFQ4G256, HFQ4G128, HFQ6G256, HFQ2/3, Q4K, MQ8G256, MQ6G256, MQ3G256, MQ2G256,
        // plus F32/BF16 teachers (BF16 kept as BF16 when HIPFIRE_CALIB_BF16=1 else widened to F32).
        // See lowered.rs:load_gemma4_weight match (lines 696-736).
        // KV must be q8 (KvCache::new_gpu_q8): F32 cache fails with "no implementation for KvWriteF32".
        // Prefill sub-chunks to scratch.max_prefill_batch (128) — mirrors calib_sweep gemma arm.
        use hipfire_arch_gemma4::lowered as gemma4;
        if args.model.is_dir() {
            eprintln!("eval_hipfire: gemma4 dir not yet wired — use HFQ for arch 13");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = gemma4::config_from_hfq(&hfq).unwrap_or_else(|| panic!("gemma4 config missing"));
        if ref_n_vocab != cfg.vocab_size {
            eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", cfg.vocab_size);
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut weights = gemma4::load_weights(&mut hfq, &cfg, &mut gpu).expect("gemma4 load weights");
        let kv_max = n_ctx + 16;
        let scratch = gemma4::Gemma4Scratch::new(&mut gpu, &cfg, 1).expect("gemma scratch");
        gemma4::init_scratch_constants(&mut gpu, &scratch, cfg.full_head_dim).expect("gemma init scratch");
        // Q8 KV — mirrors calib_sweep: gemma sliding+full both q8 (lines 853-854)
        if args.kv_mode != "q8" {
            eprintln!("eval_hipfire: gemma4 requires --kv-mode q8 (got {}); F32 cache has no KvWriteF32 kernel — refusing", args.kv_mode);
            std::process::exit(1);
        }
        let mut kv_sliding = KvCache::new_gpu_q8(&mut gpu, cfg.n_layers, cfg.sliding_n_kv_heads, cfg.sliding_head_dim, kv_max).expect("gemma sliding kv q8");
        let mut kv_full = KvCache::new_gpu_q8(&mut gpu, cfg.n_layers, cfg.full_n_kv_heads, cfg.full_head_dim, kv_max).expect("gemma full kv q8");

        for c in 0..effective_n_chunk {
            kv_sliding.clear_gpu(&mut gpu).expect("kv sliding clear");
            kv_full.clear_gpu(&mut gpu).expect("kv full clear");
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "prefill" {
                // Prefill prefix with batched forward chunked to 128 (max_prefill_batch), then per-token for scored window.
                // This mirrors calib_sweep run_gemma_batched: while offset<chunk.len() { sub = chunk[offset..min(offset+128)]; forward_prefill_batch(..., sub, pos) }
                let max_b = scratch.max_prefill_batch; // 128
                let mut offset = 0usize;
                let mut pos = 0usize;
                while offset < scoring_start {
                    let end = (offset + max_b).min(scoring_start);
                    let sub = &chunk_tokens[offset..end];
                    gemma4::forward_prefill_batch(&mut gpu, &weights, &cfg, sub, pos, &mut kv_sliding, &mut kv_full, &scratch).expect("gemma prefill prefix");
                    pos += sub.len();
                    offset = end;
                }
                for pos in scoring_start..(n_ctx - 1) {
                    gemma4::forward_scratch(&mut gpu, &weights, &cfg, chunk_tokens[pos], pos, &mut kv_sliding, &mut kv_full, &scratch).expect("gemma forward_scratch");
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll { chunk_nll_sum += n; chunk_nll_count += 1; }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!("\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ", c+1, effective_n_chunk, total_scored_done, total_scored, pct, rate);
                    }
                }
            } else {
                for pos in 0..(n_ctx - 1) {
                    gemma4::forward_scratch(&mut gpu, &weights, &cfg, chunk_tokens[pos], pos, &mut kv_sliding, &mut kv_full, &scratch).expect("gemma forward_scratch");
                    if pos < scoring_start { continue; }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll { chunk_nll_sum += n; chunk_nll_count += 1; }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!("\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ", c+1, effective_n_chunk, total_scored_done, total_scored, pct, rate);
                    }
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
            let mean_nll = if chunk_nll_count > 0 { chunk_nll_sum / chunk_nll_count as f64 } else { f64::NAN };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    } else if is_glimmer {
        // ----- glimmer 14 — WIRED -----
        // Quant dtypes admitted: Q8_0 (embed), HFQ4G256/G128, MQ4G256 etc (same family as gemma4/qwen35).
        // See glimmer.rs load weights quant_type match (3->Q8,6->HFQ4G256 etc).
        // Prefills via prefill_with_capture, chunk 192 (mirrors calib_sweep glimmer arm: prefill_chunk 192).
        use hipfire_arch_muse_glimmer::config::GlimmerConfig;
        use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
        use hipfire_arch_muse_glimmer::forward as glimmer_fwd;
        if args.model.is_dir() {
            eprintln!("eval_hipfire: glimmer dir not yet wired — use HFQ for arch 14");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = GlimmerConfig::from_hfq(&hfq).expect("glimmer cfg");
        if ref_n_vocab != cfg.vocab_size {
            eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", cfg.vocab_size);
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("glimmer weights");
        let kv_max = n_ctx + 16;
        let mut state = GlimmerState::new_with_max_seq(&mut gpu, &cfg, kv_max).expect("glimmer state");
        // chunk size 192 — mirrors calib_sweep glimmer_prefill_chunk_size (192)
        let glimmer_chunk: usize = 192;

        for c in 0..effective_n_chunk {
            state.reset();
            // For scoring_start prefix, use batched prefill chunked to 192 to mirror calib_sweep
            // Note: GlimmerState reset clears n_tokens but KV buffers remain; next prefill overwrites from 0.
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "prefill" {
                // Prefill prefix chunked 192
                let mut pos = 0usize;
                while pos < scoring_start {
                    let end = (pos + glimmer_chunk).min(scoring_start);
                    let sub = &chunk_tokens[pos..end];
                    glimmer_fwd::prefill_with_capture(&cfg, &weights, &mut state, &mut gpu, sub, pos as u32, &[], &mut Vec::new()).expect("glimmer prefill prefix");
                    pos = end;
                }
                for pos in scoring_start..(n_ctx - 1) {
                    let logits = glimmer_fwd::decode_step(&cfg, &weights, &mut state, &mut gpu, chunk_tokens[pos], pos as u32).expect("glimmer decode");
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll { chunk_nll_sum += n; chunk_nll_count += 1; }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!("\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ", c+1, effective_n_chunk, total_scored_done, total_scored, pct, rate);
                    }
                }
            } else {
                for pos in 0..(n_ctx - 1) {
                    let logits = glimmer_fwd::decode_step(&cfg, &weights, &mut state, &mut gpu, chunk_tokens[pos], pos as u32).expect("glimmer decode");
                    if pos < scoring_start { continue; }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll { chunk_nll_sum += n; chunk_nll_count += 1; }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!("\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ", c+1, effective_n_chunk, total_scored_done, total_scored, pct, rate);
                    }
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
            let mean_nll = if chunk_nll_count > 0 { chunk_nll_sum / chunk_nll_count as f64 } else { f64::NAN };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    } else if is_lfm {
        // ----- lfm2 11 — WIRED (dense 1.2b/350m, num_experts==0) -----
        // Quant dtypes admitted: dense projections Q8_0/HFQ4G256/MQ4G256 (batch_weight_formats_supported);
        // embeddings Q8/HFQ4; lm_head Q8/HFQ4/MQ4/HFQ6/MQ6/MQ3 (+F32/BF16 teachers).
        // See lfm2moe/src/lfm2moe.rs batch_weight_formats_supported and forward_batch::batched_proj.
        // Uses forward_decode_batch_lfm with explicit positions; KV is q8 and only some layers carry KV
        // (KvCache q8 filtered on is_kv_layer, lane_capacity = n_ctx+16).
        use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
        use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
        use hipfire_arch_lfm2moe::forward_batch;
        use hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights;
        if args.model.is_dir() {
            eprintln!("eval_hipfire: lfm2 dir not yet wired — use HFQ for arch 11");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = Lfm2MoeConfig::from_hfq(&hfq).unwrap_or_else(|e| panic!("lfm2 cfg: {e}"));
        if cfg.num_experts != 0 {
            eprintln!("lfm2 MoE (num_experts={}) not admitted for eval_hipfire dense lane", cfg.num_experts);
            std::process::exit(1);
        }
        if ref_n_vocab != cfg.vocab_size {
            eprintln!("vocab mismatch: ref says {ref_n_vocab}, model says {}", cfg.vocab_size);
            std::process::exit(2);
        }
        if let Err(e) = forward_batch::batch_weight_formats_supported(&Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).as_ref().map(|w| w as &Lfm2MoeWeights).unwrap_or_else(|_| panic!("peek"))) {
            // placeholder: actual check after load below; keep compile shape
            let _ = e;
        }
        // Reload after peek above — we need weights for real
        let mut hfq2 = HfqFile::open(&args.model).expect("reopen model");
        let weights = Lfm2MoeWeights::load(&mut hfq2, &cfg, &mut gpu).expect("lfm2 weights");
        if let Err(e) = forward_batch::batch_weight_formats_supported(&weights) {
            eprintln!("unsupported lfm2 weight formats for batched decode: {e}");
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let kv_max = n_ctx + 16;
        let max_batch = n_ctx;
        let mut state = Lfm2DecodeBatchState::new(&mut gpu, &cfg, max_batch, kv_max, 32).expect("lfm batch state");

        for c in 0..effective_n_chunk {
            state.reset(&mut gpu).expect("lfm reset");
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            // LFM2 scoring: batched decode for the entire chunk (positions 0..n_ctx-1)
            // then score each scored position's logits from state.logits.
            // This matches calib_sweep's lfm arm: forward_decode_batch_lfm with per-chunk positions.
            if args.scoring_mode == "prefill" || args.scoring_mode == "per-token" {
                let positions: Vec<usize> = (0..chunk_tokens.len()).collect();
                forward_batch::forward_decode_batch_lfm(&mut gpu, &weights, &cfg, chunk_tokens, &positions, &mut state).expect("lfm forward batch");
                // state.logits holds [max_batch * vocab] batched; lane i logits at offset i*vocab
                let all_logits = gpu.download_f32(&state.logits).expect("download lfm logits");
                for pos in scoring_start..(n_ctx - 1) {
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let lane_logits = &all_logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
                    let (kld, nll) = kld_from_block(lane_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll { chunk_nll_sum += n; chunk_nll_count += 1; }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!("\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ", c+1, effective_n_chunk, total_scored_done, total_scored, pct, rate);
                    }
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
            let mean_nll = if chunk_nll_count > 0 { chunk_nll_sum / chunk_nll_count as f64 } else { f64::NAN };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    }

    eprintln!();
    eprintln!(
        "eval_hipfire: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s) arch_id={arch_id}",
        t0.elapsed().as_secs_f64(),
        total_scored_done as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );

    // -------- write HFKSEQ output (v2: adds mean_nll per chunk) --------
    // Arch-independent: same HFKSEQ writer for all arches, so numbers are comparable.
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
    for ((m, p), n) in mean_kld_per_seq
        .iter()
        .zip(p99_kld_per_seq.iter())
        .zip(mean_nll_per_seq.iter())
    {
        out.write_all(&m.to_le_bytes()).unwrap();
        out.write_all(&p.to_le_bytes()).unwrap();
        out.write_all(&n.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();

    let overall_mean: f64 =
        mean_kld_per_seq.iter().copied().sum::<f64>() / mean_kld_per_seq.len() as f64;
    let nll_finite: Vec<f64> = mean_nll_per_seq
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let overall_nll: f64 = if nll_finite.is_empty() {
        f64::NAN
    } else {
        nll_finite.iter().copied().sum::<f64>() / nll_finite.len() as f64
    };
    let overall_ppl = overall_nll.exp();
    eprintln!(
        "eval_hipfire: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4} arch_id={arch_id}",
        overall_mean, overall_nll, overall_ppl
    );
    eprintln!("eval_hipfire: wrote {}", args.output.display());
    if is_qwen35 {
        if let Ok(p) = std::env::var("HIPFIRE_MOE_EXPERT_STATS_OUT") {
            hipfire_arch_qwen35::qwen35::dump_expert_stats(&p);
        }
    }
}

// (verify_ref_sha256 now lives in hipfire_runtime::eval_common — see
// crates/hipfire-runtime/src/eval_common.rs)
