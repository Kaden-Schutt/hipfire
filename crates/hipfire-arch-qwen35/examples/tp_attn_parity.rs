// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! TP Stage 3 acceptance: full-model TP=2 vs TP=1 logits + greedy-token
//! parity on qwen3.5-0.8b.mq4 (hybrid DeltaNet + FullAttn).
//!
//! TP=1 reference: `forward_scratch` (single GPU). TP=2: `forward_scratch_tp`
//! — FullAttn layers sharded (per-rank masked attention → partial `wo` →
//! all-reduce → residual add → FFN), DeltaNet layers replicated. The
//! reference greedy-decodes a ChatML prompt; TP is force-fed that same token
//! path so logit deltas are apples-to-apples per position. PASS iff
//! per-step max|Δlogit|/max|ref| < 1e-4 AND TP's argmax matches ref at every
//! position. The cross-rank all-reduce only reorders fp adds (~1e-6), so the
//! TP forward MATH is exact to reassociation.
//!
//! **Precision mode (default fp32).** The gate runs fp32 KV + fp32 DeltaNet
//! state to isolate the TP forward math. With q8 KV/state the model is
//! chaotically sensitive to TP's ~1e-7 all-reduce reassociation: a 1e-7 nudge
//! crosses q8 quantization buckets in the KV cache and DeltaNet recurrent
//! state, and the recurrence compounds it to ~1e-2 within a few positions
//! (single-GPU is immune only because it is perturbation-free/deterministic).
//! That is a model-precision property, NOT a TP bug — the fp32 gate confirms
//! the wo-shard reconstruction is correct (~2e-6). Set HIPFIRE_PARITY_Q8=1 to
//! reproduce the q8 sensitivity; it will report a large delta by design.
//!
//! Diagnostic env knobs: HIPFIRE_PARITY_NTOK=N (truncate prompt — N=1 isolates
//! pos-0), HIPFIRE_PARITY_REFREF=1 (single-GPU determinism check),
//! HIPFIRE_PARITY_REFSTREAM=1 (active_stream vs null-stream single-GPU),
//! HIPFIRE_PARITY_RANKDIFF=1 (rank0-vs-rank1 residual-stream sync trace).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 cargo run --release -p hipfire-arch-qwen35 \
//!       --features deltanet --example tp_attn_parity -- \
//!       ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch, StateQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::Gpu;
use std::path::Path;

const TP: usize = 2;
const N_DECODE: usize = 32;
const KV_MAX: usize = 1024;
const REL_TOL: f32 = 1e-4;
const PROMPT: &str = "Write a one-sentence greeting.";

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

// KV / DeltaNet-state precision knobs (default fp32 — the parity gate).
// Independent: HIPFIRE_PARITY_KV={fp32|q8|fwht}, HIPFIRE_PARITY_STATE={fp32|q8}.
// HIPFIRE_PARITY_Q8=1 is a legacy shortcut meaning "both q8".
fn kv_mode() -> &'static str {
    match std::env::var("HIPFIRE_PARITY_KV").as_deref() {
        Ok("q8") => "q8",
        Ok("fwht") | Ok("fwht4") => "fwht",
        Ok("fp32") | Ok("f32") => "fp32",
        _ => if std::env::var("HIPFIRE_PARITY_Q8").is_ok() { "q8" } else { "fp32" },
    }
}
fn make_kv(gpu: &mut Gpu, config: &qwen35::Qwen35Config) -> KvCache {
    let (nl, nk, hd) = (config.n_layers, config.n_kv_heads, config.head_dim);
    match kv_mode() {
        "q8" => KvCache::new_gpu_q8(gpu, nl, nk, hd, KV_MAX).expect("kv q8"),
        "fwht" => KvCache::new_gpu_fwht4(gpu, nl, nk, hd, KV_MAX).expect("kv fwht4"),
        _ => KvCache::new_gpu(gpu, nl, nk, hd, KV_MAX).expect("kv fp32"),
    }
}
fn state_is_q8() -> bool {
    match std::env::var("HIPFIRE_PARITY_STATE").as_deref() {
        Ok("q8") => true,
        Ok("fp32") | Ok("f32") => false,
        _ => std::env::var("HIPFIRE_PARITY_Q8").is_ok(),
    }
}

fn build_prompt_tokens(tok: &Tokenizer) -> Vec<u32> {
    // Diagnostic: HIPFIRE_PARITY_NTOK=N truncates the prompt to its first N
    // tokens (N=1 isolates pos-0, removing KV / DeltaNet-state accumulation).
    if let Ok(n) = std::env::var("HIPFIRE_PARITY_NTOK") {
        if let Ok(n) = n.parse::<usize>() {
            let full = build_prompt_tokens_full(tok);
            return full[..n.min(full.len())].to_vec();
        }
    }
    build_prompt_tokens_full(tok)
}

fn build_prompt_tokens_full(tok: &Tokenizer) -> Vec<u32> {
    let im_start = tok.encode("<|im_start|>");
    let im_end = tok.encode("<|im_end|>");
    let nl = tok.encode("\n");
    let user = tok.encode("user");
    let asst = tok.encode("assistant");
    let q = tok.encode(PROMPT);
    let mut t = Vec::new();
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&user);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&q);
    t.extend_from_slice(&im_end);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&asst);
    t.extend_from_slice(&nl);
    t
}

// TP=1 reference: per-token prefill + greedy decode on a single GPU.
// `use_stream`: set an active_stream (diagnostic — mirrors the TP context to
// isolate stream-dependent divergence from the TP sharding itself).
fn run_single_gpu_opt(path: &str, prompt_tokens: &[u32], use_stream: bool) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("Gpu::init");
    if use_stream {
        gpu.bind_thread().expect("bind");
        let st = gpu.hip.stream_create().expect("stream");
        gpu.active_stream = Some(st);
    }
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    // KV + DeltaNet-state precision are toggled INDEPENDENTLY (default fp32)
    // so the isolation test can pin which one amplifies TP's ~1e-7 all-reduce
    // reassociation. See module header for the q8-sensitivity rationale.
    let state_q8 = state_is_q8();
    let mut kv = make_kv(&mut gpu, &config);
    let mut dn = DeltaNetState::new_with_quant(
        &mut gpu, &config, if state_q8 { StateQuant::Q8 } else { StateQuant::FP32 },
    ).expect("dn");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, i, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch prefill");
    }
    if use_stream { gpu.hip.device_synchronize().unwrap(); }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        let next = argmax(&logits);
        all_logits.push(logits);
        next
    };
    tokens.push(tok);
    for step in 1..N_DECODE {
        let pos = prompt_tokens.len() + step - 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch)
            .expect("forward_scratch decode");
        if use_stream { gpu.hip.device_synchronize().unwrap(); }
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        tok = argmax(&logits);
        tokens.push(tok);
        all_logits.push(logits);
    }
    scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    (tokens, all_logits)
}

fn run_single_gpu(path: &str, prompt_tokens: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    run_single_gpu_opt(path, prompt_tokens, false)
}

// TP=2: replicated weights, sharded FullAttn (wo partial + all-reduce),
// replicated DeltaNet. Per-token prefill, then for each decode step FEEDS
// `forced[step-1]` (the reference's chosen token) so both models walk the
// IDENTICAL input path — isolating forward-pass numerical parity from
// greedy-divergence amplification. Returns (TP's own argmax per step,
// TP logits per step).
fn run_tp(path: &str, prompt_tokens: &[u32], forced: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let attn_dim = config.n_heads * config.head_dim;

    let shard = ShardConfig::new(TP, false, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();

    let mut gpus = Gpus::init_tp(TP, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let st = dev.hip.stream_create().expect("stream");
        dev.active_stream = Some(st);
    }

    // Replicated weights + per-rank scratch/kv/dn.
    let mut weights = Vec::with_capacity(TP);
    let mut scratches = Vec::with_capacity(TP);
    let mut kvs = Vec::with_capacity(TP);
    let mut dns = Vec::with_capacity(TP);
    let mut masks = Vec::with_capacity(TP);
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap());
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &config, 128).unwrap());
        // KV + DeltaNet-state precision toggled independently (default fp32);
        // must match run_single_gpu_opt for a meaningful comparison.
        let state_q8 = state_is_q8();
        kvs.push(make_kv(&mut gpus.devices[r], &config));
        dns.push(DeltaNetState::new_with_quant(
            &mut gpus.devices[r], &config, if state_q8 { StateQuant::Q8 } else { StateQuant::FP32 },
        ).unwrap());
        // mask[r]: 1.0 on this rank's local Q-heads over the attention output.
        let range = shard.wo_col_range(r, config.n_heads, config.head_dim);
        let mut m = vec![0.0f32; attn_dim];
        m[range].iter_mut().for_each(|v| *v = 1.0);
        masks.push(gpus.devices[r].upload_f32(&m, &[attn_dim]).unwrap());
    }

    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch_tp(
            &mut gpus, &shard, &weights, &config, tok, i, &mut kvs, &mut dns, &scratches, &masks,
        )
        .expect("forward_scratch_tp prefill");
    }
    let mut tokens = Vec::with_capacity(N_DECODE);
    let mut tok = {
        gpus.devices[0].bind_thread().unwrap();
        let logits = gpus.devices[0].download_f32(&scratches[0].logits).expect("download logits");
        let next = argmax(&logits);
        all_logits.push(logits);
        next
    };
    tokens.push(tok);
    let _ = tok; // TP's own step-0 argmax recorded; input is forced below.
    let rankdiff = std::env::var("HIPFIRE_PARITY_RANKDIFF").is_ok();
    for step in 1..N_DECODE {
        let pos = prompt_tokens.len() + step - 1;
        let in_tok = forced[step - 1]; // walk the reference path
        qwen35::forward_scratch_tp(
            &mut gpus, &shard, &weights, &config, in_tok, pos, &mut kvs, &mut dns, &scratches, &masks,
        )
        .expect("forward_scratch_tp decode");
        // Diagnostic: do the two ranks' residual streams stay in sync? They
        // start each layer identical and re-sync after every all-reduce, so
        // s.x should be bit-identical across ranks unless replication breaks.
        if rankdiff && TP >= 2 {
            gpus.devices[0].bind_thread().unwrap();
            let x0 = gpus.devices[0].download_f32(&scratches[0].x).unwrap();
            gpus.devices[1].bind_thread().unwrap();
            let x1 = gpus.devices[1].download_f32(&scratches[1].x).unwrap();
            let mut d = 0.0f32;
            let mut mx = 0.0f32;
            for (a, b) in x0.iter().zip(x1.iter()) {
                d = d.max((a - b).abs());
                mx = mx.max(a.abs());
            }
            if step <= 6 || d / (mx + 1e-12) > 1e-4 {
                println!("  [rankdiff] step {step} (pos {pos}): max|s.x0 - s.x1| = {d:.3e}  rel = {:.3e}", d / (mx + 1e-12));
            }
        }
        gpus.devices[0].bind_thread().unwrap();
        let logits = gpus.devices[0].download_f32(&scratches[0].logits).expect("download logits");
        tokens.push(argmax(&logits)); // TP's own pick (for divergence reporting)
        all_logits.push(logits);
    }

    // Minimal cleanup: destroy the per-rank streams (process exit reclaims
    // the rest; Gpus Drop frees devices/comms in declared order).
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().unwrap();
        if let Some(st) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(st);
        }
    }
    (tokens, all_logits)
}

fn main() {
    // Deterministic reduction (matches pp_parity / coherence gates).
    std::env::set_var("HIPFIRE_DETERMINISTIC", "1");
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3.5-0.8b.mq4",
            std::env::var("HOME").unwrap_or_else(|_| "/home/kaden".into())
        )
    });

    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let prompt_tokens = build_prompt_tokens(&tokenizer);
    drop(hfq);

    println!("=== TP=2 ↔ TP=1 full-model parity (Stage 3 acceptance) ===");
    println!("prompt: {PROMPT:?}  (prompt_len={}, N_DECODE={N_DECODE})", prompt_tokens.len());
    println!(
        "precision: KV={}  DeltaNet-state={}",
        kv_mode(),
        if state_is_q8() { "q8" } else { "fp32" },
    );

    // Diagnostic: ref-vs-ref determinism check. Two independent single-GPU
    // runs along the same forced path; if they diverge like TP does, the
    // forward pass (DeltaNet Q8 state update) is non-deterministic and that —
    // not the TP wo-shard — is the divergence source.
    if std::env::var("HIPFIRE_PARITY_REFREF").is_ok() {
        let (rt, rl) = run_single_gpu(&path, &prompt_tokens);
        let (rt3, rl3) = run_single_gpu(&path, &prompt_tokens);
        let mut worst = 0.0f32;
        for (a, b) in rl.iter().zip(rl3.iter()) {
            let m = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let mut d = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) { d = d.max((x - y).abs()); }
            worst = worst.max(d / (m + 1e-12));
        }
        let mism = (0..rt.len()).filter(|&s| rt[s] != rt3[s]).count();
        println!("REF-vs-REF: worst rel Δ = {worst:.3e}, argmax mismatches = {mism}/{}", rt.len());
        return;
    }

    // Diagnostic: does setting an active_stream alone (no TP sharding) change
    // the single-GPU forward? Isolates a stream-context bug (e.g. an async
    // memset that only fires when active_stream is Some) from the TP logic.
    if std::env::var("HIPFIRE_PARITY_REFSTREAM").is_ok() {
        let (rt_ns, rl_ns) = run_single_gpu_opt(&path, &prompt_tokens, false); // null stream
        let (rt_st, rl_st) = run_single_gpu_opt(&path, &prompt_tokens, true);  // active stream
        let mut worst = 0.0f32;
        let mut worst_step = 0usize;
        for (s, (a, b)) in rl_ns.iter().zip(rl_st.iter()).enumerate() {
            let m = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let mut d = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) { d = d.max((x - y).abs()); }
            let r = d / (m + 1e-12);
            if r > worst { worst = r; worst_step = s; }
        }
        let mism = (0..rt_ns.len()).filter(|&s| rt_ns[s] != rt_st[s]).count();
        println!("REF(null-stream) vs REF(active-stream): worst rel Δ = {worst:.3e} (step {worst_step}), argmax mismatches = {mism}/{}", rt_ns.len());
        println!("  null-stream first 12: {:?}", &rt_ns[..12.min(rt_ns.len())]);
        println!("  active-stream first 12: {:?}", &rt_st[..12.min(rt_st.len())]);
        return;
    }

    println!("\n── TP=1 reference (defines the input path) ──");
    let (toks1, logits1) = run_single_gpu(&path, &prompt_tokens);
    println!("TP=1 tokens: {:?}", &toks1[..20.min(toks1.len())]);
    println!("TP=1 text  : {:?}", tokenizer.decode(&toks1));

    // TP walks the IDENTICAL input path (force-fed toks1), so logit deltas
    // are apples-to-apples per position — no greedy-divergence blowup.
    println!("\n── TP=2 (forced onto the TP=1 path) ──");
    let (toks2, logits2) = run_tp(&path, &prompt_tokens, &toks1);
    println!("TP=2 own argmax: {:?}", &toks2[..20.min(toks2.len())]);

    // ── Per-step logit delta along the identical path. ──
    let mut worst_rel = 0.0f32;
    let mut worst_step = 0usize;
    let mut per_step_rel = Vec::with_capacity(logits1.len());
    for (step, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        let ref_max = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut max_abs = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        let rel = max_abs / (ref_max + 1e-12);
        per_step_rel.push(rel);
        if rel > worst_rel {
            worst_rel = rel;
            worst_step = step;
        }
    }

    // ── Argmax agreement at each position (given identical inputs). ──
    let argmax_mismatches: Vec<usize> = (0..toks1.len())
        .filter(|&s| toks1[s] != toks2[s])
        .collect();

    println!("\n── parity (identical-path) ──");
    print!("per-step max rel logit Δ: ");
    for (s, r) in per_step_rel.iter().enumerate().take(8) {
        print!("[{s}]={r:.2e} ");
    }
    println!("...");
    println!("worst per-step max rel logit Δ = {worst_rel:.3e} (step {worst_step}, tol {REL_TOL:.0e})");
    println!("argmax agreement: {}/{} positions match", toks1.len() - argmax_mismatches.len(), toks1.len());

    // Detail any argmax mismatch with the reference top-1 margin — a flip
    // with margin ≲ |Δlogit| is a roundoff razor-edge, not a forward bug.
    for &s in &argmax_mismatches {
        let l = &logits1[s];
        let a0 = toks1[s] as usize;
        let a1 = toks2[s] as usize;
        // ref top-1 margin = ref(winner) − ref(runner-up among the two picks)
        let margin = (l[a0] - l[a1]).abs();
        println!(
            "  step {s}: TP=1 {} ({:?}) vs TP=2 {} ({:?}); ref margin between them = {:.3e}, step relΔ = {:.3e}",
            toks1[s], tokenizer.decode(&[toks1[s]]),
            toks2[s], tokenizer.decode(&[toks2[s]]),
            margin, per_step_rel[s],
        );
    }

    assert!(
        worst_rel < REL_TOL,
        "logit parity FAILED: worst rel Δ {worst_rel:.3e} ≥ {REL_TOL:.0e} at step {worst_step} (identical input path)"
    );
    assert!(
        argmax_mismatches.is_empty(),
        "argmax parity FAILED: {} of {} positions flipped on the identical path",
        argmax_mismatches.len(), toks1.len()
    );
    println!("\ntp_attn_parity: PASS  (argmax identical at all {} positions, max rel logit Δ {worst_rel:.3e})", toks1.len());
}
