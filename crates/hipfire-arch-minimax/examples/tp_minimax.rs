// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 TP-of-experts parity harness — Task 3.
//!
//! Runs the same prompt twice:
//!   1. tp=1 (whole experts, single logical rank) — reference output.
//!   2. tp=2 (every rank holds all experts, column/row-split by inter/tp) —
//!      via `MiniMaxWeights::load(.., tp_slice=Some(..))`  + `forward_tp`.
//!
//! Asserts argmax-exact across all generated tokens AND logit max|Δ| < 1e-2.
//! Prints prompt md5, per-token argmax comparison, and load time.
//!
//! Run:
//!   HIPFIRE_DETERMINISTIC=1 HIPFIRE_EMULATE_GPUS=2 \
//!   HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 \
//!   cargo run --release -p hipfire-arch-minimax --example tp_minimax -- \
//!     --model ~/.hipfire/models/MiniMax-M2.7.mq2 --max 32

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn fnv1a_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(feature = "deltanet")]
fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax::forward;
    use hipfire_arch_minimax::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::TpExpertSlice;
    use rdna_compute::{DType, GpuTensor};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 32;
    let mut selfcheck_tp1 = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            // THROWAWAY diagnostic: run tp=1 twice to measure atomic-add nondeterminism floor.
            "--selfcheck-tp1" => {
                selfcheck_tp1 = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let prompt_fnv = fnv1a_bytes(prompt.as_bytes());
    eprintln!("prompt fnv1a_bytes: 0x{prompt_fnv:016x}  max: {max}");

    // ── config + tokenizer ─────────────────────────────────────────────────────
    let hfq0 = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq0).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    drop(hfq0);

    eprintln!(
        "minimax: hidden={} layers={} experts={}/{} vocab={} inter={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size,
        cfg.intermediate_size,
    );

    let prompt_ids = tok.encode(&prompt);
    let max_seq = prompt_ids.len() + max + 16;
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());

    // ── tp=1 reference run — forward_tp with n_ranks=1 and the int64 down path ──
    // Uses forward_tp (not decode_step) so both tp=1 and tp=2 share the same
    // int64 kernel path. The i64 kernel is partition-invariant, so tp=1 (K=inter)
    // and tp=2 (K=inter/2 each, i64 all-reduce) produce BIT-IDENTICAL f32 logits.
    eprintln!("\n=== tp=1 reference run (forward_tp, i64 down) ===");
    let tp1_tokens;
    let tp1_logits_all;
    {
        let tp1 = 1usize;
        let mut gpus1 = Gpus::init_tp(tp1, cfg.num_hidden_layers).expect("init_tp(1)");
        gpus1.ensure_rank_streams().expect("rank_streams tp1");
        let _ = gpus1.enable_peer_all().expect("peer_all tp1");

        let t_load = std::time::Instant::now();
        let mut hfq = HfqFile::open(&model).expect("open model tp1");
        let ts = TpExpertSlice { tp: tp1, rank: 0 };
        let w = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpus1.devices[0], None, Some(ts))
            .expect("load tp1");
        eprintln!("  tp=1 loaded in {:.1}s", t_load.elapsed().as_secs_f64());

        let mut state_per_rank: Vec<MiniMaxState> =
            vec![
                MiniMaxState::new_with_max_seq(&mut gpus1.devices[0], &cfg, max_seq)
                    .expect("state tp1"),
            ];
        let mut partials: Vec<GpuTensor> = vec![gpus1.devices[0]
            .zeros(&[cfg.hidden_size], DType::F32)
            .expect("partial tp1")];
        // int64 scratch: hidden * 8 bytes per element (raw bytes, no DType::I64).
        let mut partials_i64: Vec<GpuTensor> = vec![gpus1.devices[0]
            .zeros(&[cfg.hidden_size * 8], DType::Raw)
            .expect("partial_i64 tp1")];
        let weights_per_rank: Vec<MiniMaxWeights> = vec![w];

        // Prefill via forward_tp (i64 down path).
        for (pos, &t) in prompt_ids.iter().enumerate() {
            forward::forward_tp(
                &mut gpus1,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                &partials_i64,
                t,
                pos as u32,
            )
            .expect("forward_tp prefill tp1");
        }
        gpus1.devices[0].bind_thread().expect("bind0 tp1");
        let mut logits = gpus1.devices[0]
            .download_f32(&state_per_rank[0].logits)
            .expect("dl tp1");

        let mut tokens = Vec::new();
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        let mut pos = prompt_ids.len();
        all_logits.push(logits.clone());
        for _step in 0..max {
            let next = argmax(&logits);
            tokens.push(next);
            if matches!(next, 200020 | 151643 | 151645 | 2) {
                break;
            }
            forward::forward_tp(
                &mut gpus1,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                &partials_i64,
                next,
                pos as u32,
            )
            .expect("forward_tp tp1");
            gpus1.devices[0].bind_thread().expect("bind0 tp1");
            logits = gpus1.devices[0]
                .download_f32(&state_per_rank[0].logits)
                .expect("dl tp1");
            all_logits.push(logits.clone());
            pos += 1;
        }
        eprintln!("  tp=1 generated {} tokens", tokens.len());
        tp1_tokens = tokens;
        tp1_logits_all = all_logits;

        if selfcheck_tp1 {
            eprintln!("\n=== selfcheck-tp1: second tp=1 pass (nondeterminism floor) ===");
            let mut state2 =
                vec![
                    MiniMaxState::new_with_max_seq(&mut gpus1.devices[0], &cfg, max_seq)
                        .expect("state tp1 second pass"),
                ];
            let mut logits2 = Vec::new();
            for (pos, &t) in prompt_ids.iter().enumerate() {
                forward::forward_tp(
                    &mut gpus1,
                    &weights_per_rank,
                    &cfg,
                    &mut state2,
                    &partials,
                    &partials_i64,
                    t,
                    pos as u32,
                )
                .expect("forward_tp prefill tp1 second pass");
                gpus1.devices[0].bind_thread().expect("bind0");
                logits2 = gpus1.devices[0]
                    .download_f32(&state2[0].logits)
                    .expect("dl tp1 second pass");
            }
            let mut tokens2 = Vec::new();
            let mut all_logits2: Vec<Vec<f32>> = Vec::new();
            let mut pos2 = prompt_ids.len();
            all_logits2.push(logits2.clone());
            for _step in 0..max {
                let next2 = argmax(&logits2);
                tokens2.push(next2);
                if matches!(next2, 200020 | 151643 | 151645 | 2) {
                    break;
                }
                forward::forward_tp(
                    &mut gpus1,
                    &weights_per_rank,
                    &cfg,
                    &mut state2,
                    &partials,
                    &partials_i64,
                    next2,
                    pos2 as u32,
                )
                .expect("forward_tp tp1 second pass");
                gpus1.devices[0].bind_thread().expect("bind0");
                logits2 = gpus1.devices[0]
                    .download_f32(&state2[0].logits)
                    .expect("dl tp1 second pass");
                all_logits2.push(logits2.clone());
                pos2 += 1;
            }
            eprintln!("  tp1-pass2 generated {} tokens", tokens2.len());

            let n_steps2 = tp1_tokens.len().min(tokens2.len());
            let mut argmax_ok2 = true;
            let mut max_logit_delta_tp1: f32 = 0.0;
            for step in 0..n_steps2 {
                if tp1_tokens[step] != tokens2[step] {
                    eprintln!(
                        "  ARGMAX MISMATCH (tp1 vs tp1) step {step}: pass1={} pass2={}",
                        tp1_tokens[step], tokens2[step]
                    );
                    argmax_ok2 = false;
                }
                if step < tp1_logits_all.len() && step < all_logits2.len() {
                    let l1 = &tp1_logits_all[step];
                    let l2 = &all_logits2[step];
                    let delta = l1
                        .iter()
                        .zip(l2.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if delta > max_logit_delta_tp1 {
                        max_logit_delta_tp1 = delta;
                    }
                }
            }
            if let Some(l0) = tp1_logits_all.first() {
                let mut top5: Vec<f32> = l0.iter().map(|x| x.abs()).collect();
                top5.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let top5: Vec<f32> = top5.into_iter().take(5).collect();
                eprintln!("  logit magnitude scale (step 0 top-5 |logit|): {top5:.2?}");
                let argmax_logit = l0[argmax(l0) as usize];
                eprintln!("  argmax logit value at step 0: {argmax_logit:.4}");
            }
            eprintln!("  tp1-vs-tp1 argmax-exact: {argmax_ok2}");
            eprintln!("  tp1-vs-tp1 logit max|Δ|: {max_logit_delta_tp1:.4e}");
            eprintln!("\nSELFCHECK-TP1 DONE");
            return;
        }

        // Free GPU memory before tp=2 load (so the same physical device can
        // hold the tp=2 weights under HIPFIRE_EMULATE_GPUS=2).
        let mut gpu0 = &mut gpus1.devices[0];
        for s in state_per_rank {
            s.free_gpu(gpu0);
        }
        for w in weights_per_rank {
            w.free_gpu(gpu0);
        }
        // partials and partials_i64 are plain GpuTensors; return them to the pool.
        gpu0.bind_thread().expect("bind0 tp1 free");
        gpu0.drain_pool();
    }

    // ── tp=2 run (TP-of-experts: every rank owns all experts, inter/2 each) ───
    let tp = 2usize;
    eprintln!("\n=== tp={tp} TP-of-experts run ===");
    let tp2_tokens;
    let tp2_logits_all;
    {
        let mut gpus2 = Gpus::init_tp(tp, cfg.num_hidden_layers).expect("init_tp(2)");
        gpus2.ensure_rank_streams().expect("rank_streams");
        let _ = gpus2.enable_peer_all().expect("peer_all");

        let mut weights_per_rank: Vec<MiniMaxWeights> = Vec::with_capacity(tp);
        let t_load = std::time::Instant::now();
        for r in 0..tp {
            gpus2.devices[r].bind_thread().expect("bind");
            let mut hfq = HfqFile::open(&model).expect("open model tp2");
            let ts = TpExpertSlice { tp, rank: r };
            let w = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpus2.devices[r], None, Some(ts))
                .expect("load tp2");
            weights_per_rank.push(w);
        }
        let load_elapsed = t_load.elapsed().as_secs_f64();
        eprintln!("  tp=2 all ranks loaded in {load_elapsed:.1}s (down row-gather included)");

        let mut state_per_rank: Vec<MiniMaxState> = Vec::with_capacity(tp);
        let mut partials: Vec<GpuTensor> = Vec::with_capacity(tp);
        let mut partials_i64: Vec<GpuTensor> = Vec::with_capacity(tp);
        for r in 0..tp {
            gpus2.devices[r].bind_thread().expect("bind");
            state_per_rank.push(
                MiniMaxState::new_with_max_seq(&mut gpus2.devices[r], &cfg, max_seq)
                    .expect("state tp2"),
            );
            partials.push(
                gpus2.devices[r]
                    .zeros(&[cfg.hidden_size], DType::F32)
                    .expect("partial tp2"),
            );
            // int64 scratch: hidden * 8 bytes (DType::Raw, element = 1 byte).
            partials_i64.push(
                gpus2.devices[r]
                    .zeros(&[cfg.hidden_size * 8], DType::Raw)
                    .expect("partial_i64 tp2"),
            );
        }

        // Prefill.
        for (pos, &t) in prompt_ids.iter().enumerate() {
            forward::forward_tp(
                &mut gpus2,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                &partials_i64,
                t,
                pos as u32,
            )
            .expect("forward_tp prefill");
        }
        gpus2.devices[0].bind_thread().expect("bind0");
        let mut logits = gpus2.devices[0]
            .download_f32(&state_per_rank[0].logits)
            .expect("dl tp2");

        let mut tokens = Vec::new();
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        let mut pos = prompt_ids.len();
        all_logits.push(logits.clone());
        for step in 0..max {
            let next = argmax(&logits);
            tokens.push(next);
            if matches!(next, 200020 | 151643 | 151645 | 2) {
                break;
            }
            let _ = step;
            forward::forward_tp(
                &mut gpus2,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                &partials_i64,
                next,
                pos as u32,
            )
            .expect("forward_tp decode");
            gpus2.devices[0].bind_thread().expect("bind0");
            logits = gpus2.devices[0]
                .download_f32(&state_per_rank[0].logits)
                .expect("dl tp2");
            all_logits.push(logits.clone());
            pos += 1;
        }
        eprintln!("  tp=2 generated {} tokens", tokens.len());
        tp2_tokens = tokens;
        tp2_logits_all = all_logits;
    }

    // ── Parity check ──────────────────────────────────────────────────────────
    eprintln!("\n=== Parity check tp1 vs tp2 ===");
    let n_steps = tp1_tokens.len().min(tp2_tokens.len());
    let mut argmax_ok = true;
    let mut max_logit_delta: f32 = 0.0;

    for step in 0..n_steps {
        let t1 = tp1_tokens[step];
        let t2 = tp2_tokens[step];
        if t1 != t2 {
            eprintln!("  ARGMAX MISMATCH at step {step}: tp1={t1} tp2={t2}");
            argmax_ok = false;
        }
        // Compare logits at this step (logits_all[step] = logits BEFORE token step is decoded).
        if step < tp1_logits_all.len() && step < tp2_logits_all.len() {
            let l1 = &tp1_logits_all[step];
            let l2 = &tp2_logits_all[step];
            let delta = l1
                .iter()
                .zip(l2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if delta > max_logit_delta {
                max_logit_delta = delta;
            }
        }
    }
    if tp1_tokens.len() != tp2_tokens.len() {
        eprintln!(
            "  token count mismatch: tp1={} tp2={}",
            tp1_tokens.len(),
            tp2_tokens.len()
        );
        argmax_ok = false;
    }

    eprintln!("  argmax-exact: {argmax_ok}");
    eprintln!("  logit max|Δ|: {max_logit_delta:.2e}");
    // Report logit magnitude scale at step 0 for context.
    if let Some(l0) = tp1_logits_all.first() {
        let mut top5: Vec<f32> = l0.iter().map(|x| x.abs()).collect();
        top5.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top5: Vec<f32> = top5.into_iter().take(5).collect();
        eprintln!("  logit magnitude scale (step 0 top-5 |logit|): {top5:.2?}");
        eprintln!(
            "  argmax logit value at step 0: {:.4}",
            l0[argmax(l0) as usize]
        );
    }
    eprintln!("  tp=1 generation:\n{}", tok.decode(&tp1_tokens));
    eprintln!("  tp=2 generation:\n{}", tok.decode(&tp2_tokens));

    assert!(
        argmax_ok,
        "PARITY FAIL: argmax mismatch between tp=1 and tp=2 (int64 path has a leak?)"
    );
    // Both tp=1 and tp=2 use forward_tp with the int64 down path
    // (DownResidualI64 → AllReduceI64Tp → ConvertI64ToF32). The i64 kernel is
    // partition-invariant: sum(i64_rank0 + i64_rank1) == i64_full, so tp=1 and tp=2
    // produce bit-identical int64 accumulators → identical f32 after convert.
    assert!(
        max_logit_delta == 0.0,
        "PARITY FAIL: logit max|Δ|={max_logit_delta:.2e} != 0.0 (FP leak in int64 TP path)"
    );

    eprintln!("\nPARITY PASS: tp=1 == tp=2 (argmax-exact, logit max|Δ|={max_logit_delta:.2e})");
}
