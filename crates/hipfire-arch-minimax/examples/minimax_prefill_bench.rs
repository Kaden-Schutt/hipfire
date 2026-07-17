// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prefill (prompt-processing) throughput benchmark for MiniMax-M2.
//!
//! Drives the EXACT serving path: process an N-token prompt at pos 0 by
//! chunking it through `forward_batch` (mirrors daemon.rs `prefill_ids
//! .chunks(CHUNK)`), and report tokens / wall-second. Does a throwaway
//! warmup prefill first to JIT the kernels + ramp DPM (a cold first chunk
//! is 3-7x slower — see CLAUDE.md), then resets state and measures `--reps`
//! times, reporting the median.
//!
//! The `--chunk` flag (CSV) sweeps prefill chunk sizes so the expert-weight
//! read-amortization curve is directly measurable (B=64 vs 512 vs 1024).
//!
//! Usage:
//!   minimax_prefill_bench <model.mq2> [--prompt FILE] [--tokens N]
//!       [--reps R] [--warmup W] [--chunk C[,C2,...]] [--synthetic-tokens]
//!
//! Defaults: --tokens 2048, --reps 3, --warmup 1, --chunk 64.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() -> Result<(), String> {
    use hipfire_arch_minimax as minimax;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::Gpu;
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "/home/bjoern/.hipfire/models/MiniMax-M2.7.mq2".to_string());

    let mut prompt_file: Option<String> = None;
    let mut target_tokens: usize = 2048;
    let mut reps: usize = 3;
    let mut warmup: usize = 1;
    let mut chunks: Vec<usize> = vec![64];
    let mut synthetic_tokens = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--prompt" => prompt_file = Some(args.next().expect("--prompt FILE")),
            "--tokens" => target_tokens = args.next().expect("--tokens N").parse().unwrap(),
            "--reps" => reps = args.next().expect("--reps R").parse().unwrap(),
            "--warmup" => warmup = args.next().expect("--warmup W").parse().unwrap(),
            "--chunk" => {
                chunks = args
                    .next()
                    .expect("--chunk C[,C2,...]")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--synthetic-tokens" => synthetic_tokens = true,
            other => panic!("unknown flag: {other}"),
        }
    }

    eprintln!("Loading MiniMax-M2 from {model_path}...");
    let mut hfq =
        HfqFile::open(std::path::Path::new(&model_path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let tokenizer = if synthetic_tokens {
        None
    } else {
        Some(
            Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| format!("tokenizer: {e:?}"))?,
        )
    };

    // Deterministic token sequence of exactly target_tokens length (tiled).
    let base: Vec<u32> = if synthetic_tokens {
        (1..cfg.vocab_size.min(257) as u32).collect()
    } else if let Some(pf) = &prompt_file {
        let text = std::fs::read_to_string(pf).map_err(|e| format!("read prompt: {e}"))?;
        tokenizer.as_ref().unwrap().encode(&text)
    } else {
        tokenizer.as_ref().unwrap().encode(
            "The quick brown fox jumps over the lazy dog. \
             Pack my box with five dozen liquor jugs. ",
        )
    };
    assert!(!base.is_empty(), "empty prompt token stream");
    let mut tokens: Vec<u32> = Vec::with_capacity(target_tokens);
    while tokens.len() < target_tokens {
        let take = (target_tokens - tokens.len()).min(base.len());
        tokens.extend_from_slice(&base[..take]);
    }
    tokens.truncate(target_tokens);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU: {}", gpu.arch.clone());
    let t_load = Instant::now();
    let weights = <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!("Loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());
    eprintln!(
        "Config: layers={} hidden={} experts={}/{} vocab={} | prefill tokens={} chunks={:?}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size,
        target_tokens,
        chunks
    );
    let max_seq = target_tokens + 64;

    // ── Coherence mode: prefill a REAL prompt via forward_batch (the changed
    //    path, chunked at 64 like the daemon), then greedy-decode and print
    //    text. Validates the WMMA-projection prefill produces fluent output. ──
    if std::env::var_os("MINIMAX_GEN").is_some() {
        let tokenizer = tokenizer
            .as_ref()
            .ok_or("MINIMAX_GEN requires tokenizer metadata")?;
        let gen_n: usize = std::env::var("MINIMAX_GEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(80);
        let text = "Explain in two sentences why the sky is blue.";
        let ids = tokenizer.encode(text);
        let mut state =
            minimax::MiniMaxState::new_with_max_seq(&mut gpu, &cfg, ids.len() + gen_n + 16)?;
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        for ck in ids.chunks(64) {
            logits =
                minimax::forward::forward_batch(&cfg, &weights, &mut state, &mut gpu, ck, pos)?;
            pos += ck.len();
        }
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let mut gen: Vec<u32> = Vec::new();
        for _ in 0..gen_n {
            let next = am(&logits) as u32;
            if matches!(next, 200020 | 151643 | 151645 | 2) {
                break;
            }
            gen.push(next);
            logits = minimax::forward::decode_step(
                &cfg, &weights, &mut state, &mut gpu, next, pos as u32,
            )?;
            pos += 1;
        }
        println!(
            "=== PROMPT ===\n{text}\n=== GENERATION ({} tok) ===\n{}",
            gen.len(),
            tokenizer.decode(&gen)
        );
        return Ok(());
    }

    let argmax = |v: &[f32]| -> usize {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len().min(b.len()) {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        d / (na.sqrt() * nb.sqrt() + 1e-12)
    };
    // (chunk, final-token argmax, final logits) — correctness fingerprint.
    let mut verify: Vec<(usize, usize, Vec<f32>)> = Vec::new();

    for &chunk in &chunks {
        let run_once = |gpu: &mut Gpu| -> Result<f64, String> {
            let mut state = minimax::MiniMaxState::new_with_max_seq(gpu, &cfg, max_seq)?;
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("pre-sync: {e:?}"))?;
            let t = Instant::now();
            let mut pos = 0usize;
            for ck in tokens.chunks(chunk) {
                let _ = minimax::forward::forward_batch(&cfg, &weights, &mut state, gpu, ck, pos)?;
                pos += ck.len();
            }
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("post-sync: {e:?}"))?;
            Ok(t.elapsed().as_secs_f64())
        };

        for w in 0..warmup {
            let s = run_once(&mut gpu)?;
            eprintln!(
                "[chunk {} warmup {}] {} tok in {:.3}s = {:.2} tok/s",
                chunk,
                w,
                target_tokens,
                s,
                target_tokens as f64 / s
            );
        }

        let mut secs: Vec<f64> = Vec::with_capacity(reps);
        for r in 0..reps {
            let s = run_once(&mut gpu)?;
            eprintln!(
                "[chunk {} measure {}] {} tok in {:.3}s = {:.2} tok/s",
                chunk,
                r,
                target_tokens,
                s,
                target_tokens as f64 / s
            );
            secs.push(s);
        }
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = secs[secs.len() / 2];
        let best = secs[0];
        println!(
            "PREFILL chunk={} median {:.2} tok/s (best {:.2}) | {} tokens",
            chunk,
            target_tokens as f64 / median,
            target_tokens as f64 / best,
            target_tokens
        );

        // Correctness fingerprint: capture final-token logits for this chunk.
        let mut state = minimax::MiniMaxState::new_with_max_seq(&mut gpu, &cfg, max_seq)?;
        let mut pos = 0usize;
        let mut last: Vec<f32> = Vec::new();
        for ck in tokens.chunks(chunk) {
            last = minimax::forward::forward_batch(&cfg, &weights, &mut state, &mut gpu, ck, pos)?;
            pos += ck.len();
        }
        let am = argmax(&last);
        verify.push((chunk, am, last));
    }

    // ── Cross-chunk correctness: final-token argmax + cosine vs chunks[0]. ──
    // forward_batch is mathematically chunk-size-independent (per-row causal
    // masking via positions[]); larger chunks exercise intra-chunk causality
    // for rows >64 that the 64-cap never reached. This catches a masking bug.
    if verify.len() > 1 {
        let (ref_chunk, ref_am, ref_logits) = verify[0].clone();
        eprintln!("\n=== CORRECTNESS (vs chunk={ref_chunk}, argmax={ref_am}) ===");
        let mut all_ok = true;
        for (chunk, am, logits) in &verify {
            let cos = cosine(&ref_logits, logits);
            let ok = *am == ref_am && cos > 0.9999;
            all_ok &= ok;
            eprintln!(
                "  chunk={chunk:<5} argmax={am:<7} match={:<5} cosine={cos:.6} {}",
                *am == ref_am,
                if ok { "OK" } else { "<<< MISMATCH" }
            );
        }
        println!(
            "VERIFY {}",
            if all_ok {
                "PASS (all chunk sizes byte-coherent)"
            } else {
                "FAIL (chunk-size changes output — masking bug)"
            }
        );
    }
    Ok(())
}
