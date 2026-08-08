// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
//
//! qwen3_dspark_bench: A/B/C benchmark harness for DSpark vs AR vs n-gram
//! spec-decode on the Qwen3-8B target, driving the EXACT generic `Speculator`
//! path the daemon uses.
//!
//! Loads the Qwen3-8B trunk + auto-discovers `<stem>-dspark.mq4` sidecar,
//! builds a `LlamaBundle` (the `SpecTarget`), constructs the requested
//! speculator, prefills a chat-framed prompt, then runs the generic
//! prefill → step decode loop greedily for `--max` tokens, reporting:
//!   - decode tok/s (TG only, fresh-process, post-warm)
//!   - acceptance τ = committed_tokens / windows (avg tokens emitted per window)
//!   - draft accept fraction = accepted / proposed
//!   - decoded text + prompt md5 (mandatory for cross-session comparisons)
//!
//! Drafter selection (env var):
//!   HIPFIRE_QWEN3_BENCH_MODE=ar      → AR baseline (no speculation)
//!   HIPFIRE_QWEN3_BENCH_MODE=ngram   → model-free n-gram speculator
//!   HIPFIRE_QWEN3_BENCH_MODE=dspark  → DSpark speculator (default if sidecar found)
//!
//! ENV:
//!   HIPFIRE_QWEN3_MODEL     trunk MQ4 path (default ~/.hipfire/models/qwen3-8b.mq4)
//!   HIPFIRE_QWEN3_PROMPT    prompt text  (default: lru_cache_pep8_strict.txt content)
//!   HIPFIRE_QWEN3_MAX       max decode tokens (default 160)
//!   HIPFIRE_QWEN3_WARMUP    throwaway warmup tokens before timed run (default 128)
//!   HIPFIRE_QWEN3_RAW       =1 bypass ChatML framing (base-model completion mode)

use hipfire_arch_llama::dspark_body::{build_qwen3_dspark_body, load_qwen3_dspark};
use hipfire_arch_llama::{load_llama_bundle, LlamaBundle};
use hipfire_runtime::dspark_core::build_dspark_speculator;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame};
use hipfire_runtime::spec::{PrefillOutcome, Speculator};
use hipfire_runtime::spec_ngram::{ChainSpeculator, NgramDrafter};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::time::Instant;

fn no_abort() -> bool {
    false
}

/// Run the generic spec loop for `max` tokens, returns
/// (generated_tokens, windows, drafts_proposed, drafts_accepted).
#[allow(clippy::too_many_arguments)]
fn decode_loop(
    spec: &mut dyn Speculator,
    bundle: &mut LlamaBundle,
    gpu: &mut Gpu,
    first_token: u32,
    start_pos: usize,
    max: usize,
    eos: u32,
    raw: bool,
    temp: f32,
    top_p: f32,
    top_k: usize,
) -> Result<(Vec<u32>, u64, u64, u64), String> {
    let mut generated: Vec<u32> = Vec::with_capacity(max);
    let mut position = start_pos;
    let mut seed = first_token;
    let mut windows: u64 = 0;
    let mut proposed: u64 = 0;
    let mut accepted: u64 = 0;

    // Stash sampling on the speculator (top_p/top_k); temp also rides each step.
    spec.set_sampling(temp, top_p, top_k, 0.0);

    if !raw && first_token == eos {
        return Ok((generated, windows, proposed, accepted));
    }
    generated.push(first_token);

    while generated.len() < max {
        let step = spec.step(
            gpu,
            bundle,
            position,
            seed,
            &generated,
            None,
            temp,
            max.saturating_sub(generated.len()).max(1),
        )?;
        windows += 1;
        proposed += step.proposed as u64;
        accepted += step.accepted as u64;
        let mut hit_eos = false;
        for &t in step.emit.iter() {
            if generated.len() >= max {
                break;
            }
            if !raw && t == eos {
                hit_eos = true;
                break;
            }
            generated.push(t);
        }
        position += step.emit.len();
        seed = step.next_seed;
        if hit_eos || (!raw && seed == eos) {
            break;
        }
    }
    Ok((generated, windows, proposed, accepted))
}

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_QWEN3_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.hipfire/models/qwen3-8b.mq4",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let max: usize = std::env::var("HIPFIRE_QWEN3_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(160);
    // Default 128: gfx1151 DPM ramps over several seconds; 24 tokens is too short.
    let warmup: usize = std::env::var("HIPFIRE_QWEN3_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let raw = std::env::var("HIPFIRE_QWEN3_RAW").ok().as_deref() == Some("1");
    let mode = std::env::var("HIPFIRE_QWEN3_BENCH_MODE").unwrap_or_else(|_| "dspark".to_string());
    // Sampling: temp=0 (default) → greedy verify; temp>0 → DSpark sampled verify.
    let temp: f32 = std::env::var("HIPFIRE_QWEN3_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let top_p: f32 = std::env::var("HIPFIRE_QWEN3_TOP_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let top_k: usize = std::env::var("HIPFIRE_QWEN3_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_seq = 4096usize;

    eprintln!("Loading Qwen3-8B trunk from {path}...");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found in HFQ metadata: {e:?}"))?;
    // Do NOT drop_mmap on the trunk: load_llama_bundle → Architecture::load_weights
    // uses tensor_data (mmap path). The sidecar does call drop_mmap because its
    // loader (load_qwen3_dspark) uses tensor_data_pread throughout.

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU: arch={}", gpu.arch_caps.arch());

    let cask = CaskConfig::default();
    let src = ModelSource::Hfq(hfq);
    let mut ctx = LoadCtx {
        path: &path,
        max_seq,
        draft_path: None,
        kv_mode_override: None,
        kv_backend: hipfire_runtime::kv_backend::KvBackend::Contiguous,
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_bands: None,
        mtp_mode: "auto",
        mtp_k: 3,
        spec: SpecLoadCfg::default(),
        kv_physical_cap: None,
        gpu: &mut gpu,
    };
    let mut bundle = load_llama_bundle(src, &mut ctx)?;

    // ── Auto-discover the -dspark.mq4 sidecar ─────────────────────────────────
    // Derive the sidecar path from the trunk: replace the extension.
    let sidecar_path = {
        let p = Path::new(&path);
        let stem = p
            .file_stem()
            .ok_or("path has no stem")?
            .to_str()
            .ok_or("stem not utf8")?;
        let ext = p
            .extension()
            .ok_or("path has no extension")?
            .to_str()
            .ok_or("ext not utf8")?;
        let parent = p.parent().unwrap_or(Path::new("."));
        parent
            .join(format!("{stem}-dspark.{ext}"))
            .to_string_lossy()
            .to_string()
    };
    let sidecar_present = Path::new(&sidecar_path).exists();
    if sidecar_present && mode != "ar" && mode != "ngram" {
        eprintln!("Found sidecar at {sidecar_path}");
    }

    // ── Load sidecar if DSpark mode ────────────────────────────────────────────
    let dspark_loaded = if mode == "dspark" && sidecar_present {
        let mut sidecar_hfq =
            HfqFile::open(Path::new(&sidecar_path)).map_err(|e| format!("sidecar open: {e:?}"))?;
        sidecar_hfq.drop_mmap();
        match load_qwen3_dspark(&sidecar_hfq, ctx.gpu)? {
            Some((dspark_weights, dspark_assets)) => {
                eprintln!(
                    "DSpark sidecar loaded: block_size={} target_layers={:?} markov_rank={} enable_confidence={}",
                    dspark_weights.cfg.block_size,
                    dspark_weights.cfg.target_layer_ids,
                    dspark_weights.cfg.markov_rank,
                    dspark_weights.cfg.enable_confidence,
                );
                // Set target-layer extraction on the bundle so spec_advance
                // captures hidden states for the drafter.
                bundle.set_dflash_extract_layers(dspark_weights.cfg.target_layer_ids.clone());
                bundle.dspark_weights = Some(dspark_weights);
                bundle.dspark_assets = Some(dspark_assets);
                true
            }
            None => {
                eprintln!("Sidecar has no dspark_* metadata — falling back to AR");
                false
            }
        }
    } else {
        false
    };

    // ── Prompt ─────────────────────────────────────────────────────────────────
    // Prompt text comes from env or the committed benchmark file.  Both paths
    // are committed files — byte-identical prompts are mandatory for cross-session
    // comparisons (CLAUDE.md: prompt md5 must be recorded alongside tok/s).
    let prompt = std::env::var("HIPFIRE_QWEN3_PROMPT").unwrap_or_else(|_| {
        // Default: the canonical LRU-code bench prompt used across all spec-decode gates.
        std::fs::read_to_string("benchmarks/prompts/lru_cache_pep8_strict.txt")
            .unwrap_or_else(|_| "Implement a least-recently-used cache in Python.".to_string())
    });

    // Build chat-framed token sequence (or raw for base-model mode).
    let prompt_tokens: Vec<u32> = if raw {
        tokenizer.encode(&prompt)
    } else {
        ChatFrame {
            tokenizer: &tokenizer,
            system: None,
            user: &prompt,
            assistant_prefix: AssistantPrefix::ClosedThink,
            raw: false,
        }
        .build()
    };

    // Stable FNV fingerprint for cross-session identity check (NOT cryptographic).
    let prompt_md5 = fnv64(&prompt_tokens);

    let eos_tok = bundle.config.eos_token;
    eprintln!(
        "prompt: {} tokens  token-md5={prompt_md5:016x}  mode={mode}  max={max}  warmup={warmup}",
        prompt_tokens.len()
    );

    // ── AR baseline path ───────────────────────────────────────────────────────
    if mode == "ar" {
        use hipfire_arch_llama::llama;
        let ar_run =
            |bundle: &mut LlamaBundle, gpu: &mut Gpu, n_max: usize| -> Result<Vec<u32>, String> {
                bundle.kv.compact_offset = 0;
                gpu.invalidate_graph_state();
                // Prefill token-by-token (matches AR path; batched prefill not bit-exact).
                for (i, &tok) in prompt_tokens.iter().enumerate() {
                    llama::forward_scratch_embed(
                        gpu,
                        &bundle.weights,
                        &bundle.config,
                        tok,
                        i,
                        &bundle.scratch,
                    )
                    .map_err(|e| format!("ar embed: {e:?}"))?;
                    llama::forward_scratch_compute_capture(
                        gpu,
                        &bundle.weights,
                        &bundle.config,
                        i,
                        &mut bundle.kv,
                        &bundle.scratch,
                        None,
                    )
                    .map_err(|e| format!("ar compute: {e:?}"))?;
                }
                let logits = gpu
                    .download_f32(&bundle.scratch.logits)
                    .map_err(|e| format!("ar logits: {e:?}"))?;
                let mut tok = llama::argmax(&logits) as u32;
                let mut pos = prompt_tokens.len();
                let mut gen = Vec::with_capacity(n_max);
                if raw || tok != eos_tok {
                    gen.push(tok);
                }
                while gen.len() < n_max {
                    llama::forward_scratch_embed(
                        gpu,
                        &bundle.weights,
                        &bundle.config,
                        tok,
                        pos,
                        &bundle.scratch,
                    )
                    .map_err(|e| format!("ar step embed: {e:?}"))?;
                    llama::forward_scratch_compute_capture(
                        gpu,
                        &bundle.weights,
                        &bundle.config,
                        pos,
                        &mut bundle.kv,
                        &bundle.scratch,
                        None,
                    )
                    .map_err(|e| format!("ar step compute: {e:?}"))?;
                    let lg = gpu
                        .download_f32(&bundle.scratch.logits)
                        .map_err(|e| format!("ar step logits: {e:?}"))?;
                    tok = llama::argmax(&lg) as u32;
                    pos += 1;
                    if !raw && tok == eos_tok {
                        break;
                    }
                    gen.push(tok);
                }
                Ok(gen)
            };

        // Warmup run.
        let _ = ar_run(&mut bundle, ctx.gpu, warmup)?;
        ctx.gpu
            .hip
            .device_synchronize()
            .map_err(|e| format!("ar warmup sync: {e:?}"))?;

        // Timed run.
        let t0 = Instant::now();
        let generated = ar_run(&mut bundle, ctx.gpu, max)?;
        ctx.gpu
            .hip
            .device_synchronize()
            .map_err(|e| format!("ar post sync: {e:?}"))?;
        let dt = t0.elapsed().as_secs_f64();
        let n = generated.len();
        let tok_s = if dt > 0.0 { n as f64 / dt } else { 0.0 };
        let text = tokenizer.decode(&generated);

        println!("=== qwen3_dspark_bench ===");
        println!(
            "drafter=AR  prompt_md5={prompt_md5:016x}  prompt_tokens={}",
            prompt_tokens.len()
        );
        println!(
            "tokens={n}  time={dt:.3}s  tok/s={tok_s:.2}  | windows={n}  tau=1.000  accept=0.000"
        );
        println!("--- decoded ({n} tokens) ---");
        println!("{text}");
        return Ok(());
    }

    // ── Speculator path (DSpark or n-gram) ─────────────────────────────────────
    let mut spec: Box<dyn Speculator> = if mode == "dspark" && dspark_loaded {
        let dspark_weights = bundle.dspark_weights.take().unwrap();
        let assets = bundle.dspark_assets.take().unwrap();
        let block = dspark_weights.cfg.block_size;
        let vocab = assets.config.vocab_size;

        // stage_norm = drafter's final norm (output_norm in sidecar).
        let stage_norm = assets.weights.output_norm.shallow_clone();

        // lm_head: upload_raw sets dtype=Raw; fix to F16 (carrier pattern).
        let mut lm_head = assets.weights.output.buf.shallow_clone();
        lm_head.dtype = DType::F16;
        lm_head.shape = vec![vocab];

        let conf_threshold = std::env::var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1f32);

        eprintln!("DSpark speculator: block={block}  conf_threshold={conf_threshold:.2}");

        let body = build_qwen3_dspark_body(assets, &dspark_weights.cfg, ctx.gpu)
            .map_err(|e| format!("DSpark body build: {e}"))?;

        build_dspark_speculator(
            body,
            dspark_weights,
            stage_norm,
            lm_head,
            block,
            max_seq,
            conf_threshold,
            true, // llama supports sampled verify → temp>0 testable
        )
    } else {
        // n-gram (or DSpark fallback when sidecar absent).
        let effective_mode = if mode == "dspark" && !dspark_loaded {
            eprintln!("No DSpark sidecar; falling back to n-gram");
            "ngram"
        } else {
            &mode
        };
        eprintln!("Speculator: {effective_mode}");
        let block_size: usize = std::env::var("HIPFIRE_QWEN3_NGRAM_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        Box::new(ChainSpeculator::new(
            NgramDrafter::new(2, block_size),
            block_size,
            max_seq,
            false, // greedy-only (no sampled-verify kernel on llama yet)
        ))
    };

    // ── WARMUP: full prefill + short throwaway decode (JIT + DPM ramp) ────────
    {
        let outcome = spec
            .prefill(
                ctx.gpu,
                &mut bundle,
                &prompt_tokens,
                &prompt_tokens,
                0,
                false,
                None,
                &no_abort,
            )
            .map_err(|e| format!("warmup prefill: {e}"))?;
        let first = match outcome {
            PrefillOutcome::Ready { first_token } => first_token,
            PrefillOutcome::Aborted => return Err("warmup prefill aborted".into()),
        };
        let _ = decode_loop(
            spec.as_mut(),
            &mut bundle,
            ctx.gpu,
            first,
            prompt_tokens.len(),
            warmup,
            eos_tok,
            raw,
            temp,
            top_p,
            top_k,
        )?;
        ctx.gpu
            .hip
            .device_synchronize()
            .map_err(|e| format!("warmup sync: {e:?}"))?;
    }

    // ── TIMED RUN: fresh prefill, then timed decode ────────────────────────────
    // Mirror the daemon's reset contract: spec.reset + kv.compact_offset=0 +
    // gpu.invalidate_graph_state() so the warmup-shaped HIP graph doesn't replay.
    let _ = spec.reset(ctx.gpu);
    bundle.kv.compact_offset = 0;
    ctx.gpu.invalidate_graph_state();

    let outcome = spec
        .prefill(
            ctx.gpu,
            &mut bundle,
            &prompt_tokens,
            &prompt_tokens,
            0,
            false,
            None,
            &no_abort,
        )
        .map_err(|e| format!("prefill: {e}"))?;
    let first = match outcome {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("prefill aborted".into()),
    };
    ctx.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("pre-timer sync: {e:?}"))?;

    let t0 = Instant::now();
    let (generated, windows, proposed, accepted) = decode_loop(
        spec.as_mut(),
        &mut bundle,
        ctx.gpu,
        first,
        prompt_tokens.len(),
        max,
        eos_tok,
        raw,
        temp,
        top_p,
        top_k,
    )?;
    ctx.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("post-timer sync: {e:?}"))?;
    let dt = t0.elapsed().as_secs_f64();

    let n = generated.len();
    let tok_s = if dt > 0.0 { n as f64 / dt } else { 0.0 };
    // τ = avg tokens emitted per window (committed.len()).
    let tau = if windows > 0 {
        n as f64 / windows as f64
    } else {
        0.0
    };
    let accept_frac = if proposed > 0 {
        accepted as f64 / proposed as f64
    } else {
        0.0
    };

    let text = tokenizer.decode(&generated);
    println!("=== qwen3_dspark_bench ===");
    println!(
        "drafter={mode}  temp={temp:.2} top_p={top_p:.2} top_k={top_k}  prompt_md5={prompt_md5:016x}  prompt_tokens={}",
        prompt_tokens.len()
    );
    println!(
        "tokens={n}  time={dt:.3}s  tok/s={tok_s:.2}  | windows={windows}  tau={tau:.3}  accept={accept_frac:.3}  (proposed={proposed} accepted={accepted})"
    );
    println!("--- decoded ({n} tokens) ---");
    println!("{text}");
    println!("--- token ids ---");
    println!("{generated:?}");

    spec.free(ctx.gpu);
    Ok(())
}

/// Stable FNV-1a-64 fingerprint of the token sequence.
/// NOT cryptographic — just a stable per-session cross-run identity check.
fn fnv64(toks: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
