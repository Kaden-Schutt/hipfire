// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek-V4-Flash TP-of-experts parity harness — Task 4 (mirror of
//! `tp_minimax.rs`, structured like `ep_deepseek4.rs`).
//!
//! Runs the same prompt twice, both via `forward_tp` (the int64 down path):
//!   1. tp=1 — every expert whole (`TpExpertSlice{tp:1,rank:0}` == full load).
//!   2. tp=2 — every rank owns ALL experts, each column/row-split to inter/2.
//!
//! Both use `DownResidualI64`: tp=1 accumulates the full-inter i64 then
//! `AllReduce{Ep}` over 1 rank (no-op); tp=2 accumulates two inter/2 i64 halves
//! then `AllReduceI64Tp` sums them. Fixed-point accumulation is
//! partition-invariant, so tp=1 and tp=2 produce BIT-IDENTICAL f32 logits →
//! argmax-exact AND logit max|Δ| == 0. Hash layers (0..num_hash_layers) and
//! bias layers (num_hash_layers..) both exercise the same step.
//!
//! With `--mtp`, also drafts the next-next token under TP via `mtp_forward_tp`
//! and reports the MTP-EP accept (draft vs true gen[1]) for both tp counts.
//!
//! Run (emulated Tp-2 on one gfx1151, --no-dspark forced):
//!   HIPFIRE_DETERMINISTIC=1 HIPFIRE_EMULATE_GPUS=2 HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 \
//!   cargo run --release -p hipfire-arch-deepseek4 --example tp_deepseek4 -- \
//!     --model ~/.hipfire/models/deepseek-v4-flash.mq2lloyd --max 32

use hipfire_arch_deepseek4::forward;
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::TpExpertSlice;
use rdna_compute::{DType, GpuTensor};
use std::path::PathBuf;

fn fnv1a_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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

/// One end-to-end run at the given `tp`. Returns (generated tokens, per-step
/// logits [BEFORE each token], optional MTP draft of the next-next token).
/// Loads column/row-sliced experts, prefills + greedily decodes via
/// `forward_tp`, then frees all GPU memory (so the same physical device can
/// hold the next run under `HIPFIRE_EMULATE_GPUS`).
#[allow(clippy::too_many_arguments)]
fn run_tp_n(
    model: &PathBuf,
    cfg: &DeepseekV4Config,
    prompt_ids: &[u32],
    max: usize,
    do_mtp: bool,
    eos_tok: u32,
    tp: usize,
) -> (Vec<u32>, Vec<Vec<f32>>, Option<u32>) {
    let mut gpus = Gpus::init_tp(tp, cfg.num_hidden_layers).expect("init_tp");
    let n = gpus.devices.len();
    assert_eq!(n, tp, "init_tp gave {n} devices (check HIPFIRE_EMULATE_GPUS)");
    gpus.ensure_rank_streams().expect("ensure_rank_streams");
    let _ = gpus.enable_peer_all().expect("enable_peer_all");

    let t_load = std::time::Instant::now();
    let mut weights_per_rank: Vec<DeepseekV4Weights> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        let mut hfq = HfqFile::open(model).expect("reopen model");
        let ts = TpExpertSlice { tp, rank: r };
        let w = DeepseekV4::load_weights_tp(&mut hfq, cfg, &mut gpus.devices[r], ts)
            .expect("load_weights_tp");
        weights_per_rank.push(w);
    }
    eprintln!(
        "  tp={tp} all ranks loaded in {:.1}s (down row-gather included)",
        t_load.elapsed().as_secs_f64()
    );

    let mut state_per_rank: Vec<DeepseekV4State> = Vec::with_capacity(n);
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(n);
    let mut partials_i64: Vec<GpuTensor> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        state_per_rank.push(DeepseekV4State::new(cfg).expect("state"));
        partials.push(
            gpus.devices[r]
                .zeros(&[cfg.hidden_size], DType::F32)
                .expect("partial"),
        );
        // int64 scratch: hidden * 8 bytes (DType::Raw, 1 byte/elem).
        partials_i64.push(
            gpus.devices[r]
                .zeros(&[cfg.hidden_size * 8], DType::Raw)
                .expect("partial_i64"),
        );
    }

    // Prefill.
    for (pos, &t) in prompt_ids.iter().enumerate() {
        forward::forward_tp(
            &mut gpus,
            &weights_per_rank,
            cfg,
            &mut state_per_rank,
            &partials,
            &partials_i64,
            t,
            pos as u32,
        )
        .expect("forward_tp prefill");
    }
    gpus.devices[0].bind_thread().expect("bind0");
    let mut logits = {
        let l = state_per_rank[0].logits.as_ref().expect("logits");
        gpus.devices[0].download_f32(l).expect("dl")
    };

    // MTP draft (optional): capture h_n per rank, draft next-next via mtp_forward_tp.
    let mut mtp_draft: Option<u32> = None;
    if do_mtp {
        let t0 = argmax(&logits);
        let mut h_n_per_rank: Vec<GpuTensor> = Vec::with_capacity(n);
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind");
            let streams = state_per_rank[r]
                .residual_streams
                .as_ref()
                .expect("residual_streams");
            let h = gpus.devices[r]
                .alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], DType::F32)
                .expect("alloc h_n");
            gpus.devices[r]
                .memcpy_dtod_auto(&h.buf, &streams.buf, cfg.hc_mult * cfg.hidden_size * 4)
                .expect("copy h_n");
            h_n_per_rank.push(h);
        }
        let mtp_logits = forward::mtp_forward_tp(
            &mut gpus,
            &weights_per_rank,
            cfg,
            &mut state_per_rank,
            &partials,
            &partials_i64,
            &h_n_per_rank,
            t0,
            prompt_ids.len() as u32,
        )
        .expect("mtp_forward_tp");
        mtp_draft = Some(argmax(&mtp_logits));
        for h in h_n_per_rank {
            gpus.devices[0].free_tensor(h).ok();
        }
    }

    // Greedy decode.
    let mut tokens = Vec::new();
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    let mut pos = prompt_ids.len();
    all_logits.push(logits.clone());
    for _step in 0..max {
        let next = argmax(&logits);
        tokens.push(next);
        if next == eos_tok {
            break;
        }
        forward::forward_tp(
            &mut gpus,
            &weights_per_rank,
            cfg,
            &mut state_per_rank,
            &partials,
            &partials_i64,
            next,
            pos as u32,
        )
        .expect("forward_tp decode");
        gpus.devices[0].bind_thread().expect("bind0");
        logits = {
            let l = state_per_rank[0].logits.as_ref().expect("logits");
            gpus.devices[0].download_f32(l).expect("dl")
        };
        all_logits.push(logits.clone());
        pos += 1;
    }
    eprintln!("  tp={tp} generated {} tokens", tokens.len());

    // Free every rank's GPU memory + drain the pool so the next run fits.
    for r in (0..n).rev() {
        gpus.devices[r].bind_thread().expect("bind free");
        let w = weights_per_rank.pop().unwrap();
        w.free_gpu(&mut gpus.devices[r]);
        let s = state_per_rank.pop().unwrap();
        s.free_gpu(&mut gpus.devices[r]);
    }
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind drain");
        gpus.devices[r].drain_pool();
    }

    (tokens, all_logits, mtp_draft)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 32;
    let mut no_bos = false;
    let mut mtp = false;
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
            "--no-bos" => {
                no_bos = true;
                i += 1;
            }
            "--mtp" => {
                mtp = true;
                i += 1;
            }
            // Accepted for command-line symmetry with ep_deepseek4; DSpark is
            // ALWAYS disabled in this harness (TP-of-experts parity is AR-only,
            // and the drafter stages are not sliced).
            "--no-dspark" => {
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
    eprintln!("prompt fnv1a_bytes: 0x{prompt_fnv:016x}  max: {max}  mtp: {mtp}");

    // ── config + tokenizer ─────────────────────────────────────────────────
    let hfq0 = HfqFile::open(&model).expect("open model");
    let mut cfg = DeepseekV4::config_from_hfq(&hfq0).expect("config");
    // TP-of-experts parity is AR-only; the DSpark drafter stages are not sliced.
    cfg.load_dspark = false;
    eprintln!("  DSpark disabled (TP-of-experts parity is AR-only)");
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    eprintln!(
        "deepseek4 TP: hidden={} layers={} hash_layers={} experts={}/{} inter={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.vocab_size,
    );

    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tok.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup_id("<｜begin▁of▁sentence｜>");
    let eos_tok = lookup_id("<｜end▁of▁sentence｜>").unwrap_or(tok.eos_id);
    drop(hfq0);

    let mut prompt_ids: Vec<u32> = Vec::new();
    if !no_bos {
        if let Some(b) = bos_tok {
            prompt_ids.push(b);
        }
    }
    prompt_ids.extend(tok.encode(&prompt));
    eprintln!(
        "prompt {:?} → {} tokens (bos-prepended={})",
        prompt,
        prompt_ids.len(),
        !no_bos
    );

    // ── tp=1 reference, then tp=2 ──────────────────────────────────────────
    eprintln!("\n=== tp=1 reference run (forward_tp, i64 down) ===");
    let (tp1_tokens, tp1_logits_all, tp1_mtp) =
        run_tp_n(&model, &cfg, &prompt_ids, max, mtp, eos_tok, 1);
    eprintln!("\n=== tp=2 TP-of-experts run ===");
    let (tp2_tokens, tp2_logits_all, tp2_mtp) =
        run_tp_n(&model, &cfg, &prompt_ids, max, mtp, eos_tok, 2);

    // ── Parity check ───────────────────────────────────────────────────────
    eprintln!("\n=== Parity check tp1 vs tp2 ===");
    let n_steps = tp1_tokens.len().min(tp2_tokens.len());
    let mut argmax_ok = true;
    let mut max_logit_delta: f32 = 0.0;
    for step in 0..n_steps {
        if tp1_tokens[step] != tp2_tokens[step] {
            eprintln!(
                "  ARGMAX MISMATCH at step {step}: tp1={} tp2={}",
                tp1_tokens[step], tp2_tokens[step]
            );
            argmax_ok = false;
        }
        if step < tp1_logits_all.len() && step < tp2_logits_all.len() {
            let delta = tp1_logits_all[step]
                .iter()
                .zip(tp2_logits_all[step].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            max_logit_delta = max_logit_delta.max(delta);
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
    eprintln!("  tp=1 generation:\n{}", tok.decode(&tp1_tokens));
    eprintln!("  tp=2 generation:\n{}", tok.decode(&tp2_tokens));

    if mtp {
        // MTP-EP accept: draft predicted the token AFTER t0 → true gen[1].
        let acc = |tokens: &[u32], draft: Option<u32>| -> String {
            match (draft, tokens.get(1).copied()) {
                (Some(d), Some(t)) if d == t => format!("ACCEPT ✓ (draft={d})"),
                (Some(d), t) => format!("reject (draft={d} vs gen[1]={t:?})"),
                (None, _) => "no draft".to_string(),
            }
        };
        eprintln!("  MTP tp=1 accept: {}", acc(&tp1_tokens, tp1_mtp));
        eprintln!("  MTP tp=2 accept: {}", acc(&tp2_tokens, tp2_mtp));
        eprintln!(
            "  MTP draft parity tp1==tp2: {} (tp1={:?} tp2={:?})",
            tp1_mtp == tp2_mtp,
            tp1_mtp,
            tp2_mtp
        );
        assert_eq!(tp1_mtp, tp2_mtp, "MTP draft diverged between tp=1 and tp=2");
    }

    assert!(
        argmax_ok,
        "PARITY FAIL: argmax mismatch between tp=1 and tp=2 (an inter_local site was missed?)"
    );
    // The int64 down path is partition-invariant: sum of per-rank i64 == full i64,
    // so tp=1 and tp=2 produce bit-identical f32 logits.
    assert!(
        max_logit_delta == 0.0,
        "PARITY FAIL: logit max|Δ|={max_logit_delta:.2e} != 0.0 (FP leak in int64 TP path)"
    );
    eprintln!("\nPARITY PASS: tp=1 == tp=2 (argmax-exact, logit max|Δ|={max_logit_delta:.2e})");
}
