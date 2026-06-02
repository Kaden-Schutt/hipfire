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
    let t_decode = std::time::Instant::now();
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
    let dsecs = t_decode.elapsed().as_secs_f64();
    eprintln!(
        "[timing] TP=1 (1 GPU) decode: {:.1} tok/s ({} tok / {:.3}s, incl per-step logits D2H)",
        (N_DECODE - 1) as f64 / dsecs, N_DECODE - 1, dsecs
    );
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
    // --slice (HIPFIRE_PARITY_SLICE=1): Stage 3b — per-rank sliced weights
    // (load_weights_tp), local-head config, local kv, no masks. Otherwise:
    // Stage 3 — replicated full weights + per-rank head masks.
    // Stage 3e EP-MoE: an MoE model (num_experts>0) ALWAYS uses load_weights_tp
    // (which shards routed experts) but the FULL config — EP v1 replicates
    // attention, so heads/hidden are NOT sliced. KV replicated (attention full).
    let is_moe = config.num_experts > 0;
    let slice = std::env::var("HIPFIRE_PARITY_SLICE").is_ok() || is_moe;
    // Stage 3f: MoE + HIPFIRE_EP_SHARD_ATTN → shard the attention too (local
    // heads), matching the loader's MoE-attention slicing + forward_scratch_tp's
    // sharded MoE arms. Without the flag → 3e (replicated attention, full config).
    let moe_shard_attn = is_moe && std::env::var_os("HIPFIRE_EP_SHARD_ATTN").is_some();

    // tp_kv_replicate: replicate KV heads ONLY when they can't shard cleanly
    // (n_kv_heads not divisible by tp). The clean KV-shard (replicate=false) is
    // what 3b/3c validated; replicate=true with sharded Q heads gives a WRONG
    // GQA head→KV mapping on rank>0 (local Q heads vs full KV heads). For A3B
    // FA-MoE this matters under 3f sharded attention.
    let kv_replicate = config.n_kv_heads % TP != 0;
    let shard = ShardConfig::new(TP, kv_replicate, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();
    if is_moe {
        shard.validate_moe(config.num_experts).unwrap();
    }

    let mut gpus = Gpus::init_tp(TP, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let st = dev.hip.stream_create().expect("stream");
        dev.active_stream = Some(st);
    }

    // Per-rank config: LOCAL (sliced heads) for dense --slice (3b/3c) AND for
    // MoE 3f (moe_shard_attn — sharded attention); FULL for dense-replicated
    // (Stage 3) and MoE 3e (replicated attention). Drives scratch/kv/dn sizing.
    // For MoE 3f, keep hidden_dim FULL: it sizes the (unused-by-MoE) dense-FFN
    // scratch; local_attn_config would shrink it (correct for dense 3c, moot/
    // risky for MoE since the MoE FFN scratch is sized by num_experts/mi).
    let configs: Vec<qwen35::Qwen35Config> = (0..TP)
        .map(|_| {
            if moe_shard_attn {
                let mut c = qwen35::local_attn_config(&config, &shard);
                c.hidden_dim = config.hidden_dim;
                c
            } else if slice && !is_moe {
                qwen35::local_attn_config(&config, &shard)
            } else {
                config.clone()
            }
        })
        .collect();

    // Per-rank weights/scratch/kv/dn. In --slice mode `weights[r]` is this
    // rank's attention slice and `kvs[r]` is sized for LOCAL kv heads.
    let mut weights = Vec::with_capacity(TP);
    let mut scratches = Vec::with_capacity(TP);
    let mut kvs = Vec::with_capacity(TP);
    let mut dns = Vec::with_capacity(TP);
    let mut masks: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(TP);
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(if slice {
            qwen35::load_weights_tp(&mut hfq, &config, &mut gpus.devices[r], &shard, r).unwrap()
        } else {
            qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap()
        });
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &configs[r], 128).unwrap());
        // KV + DeltaNet-state precision toggled independently (default fp32);
        // must match run_single_gpu_opt. DeltaNet state is full (replicated).
        let state_q8 = state_is_q8();
        kvs.push(make_kv(&mut gpus.devices[r], &configs[r]));
        // DeltaNet recurrent state sized from the per-rank config: LOCAL value
        // heads in --slice mode (3c), full otherwise. Must match the head
        // counts run_dn_layer_body computes from configs[r].
        dns.push(DeltaNetState::new_with_quant(
            &mut gpus.devices[r], &configs[r], if state_q8 { StateQuant::Q8 } else { StateQuant::FP32 },
        ).unwrap());
        if !slice {
            // mask[r]: 1.0 on this rank's local Q-heads over the full
            // attention output (replicated-weights Stage 3 path).
            let range = shard.wo_col_range(r, config.n_heads, config.head_dim);
            let mut m = vec![0.0f32; attn_dim];
            m[range].iter_mut().for_each(|v| *v = 1.0);
            masks.push(gpus.devices[r].upload_f32(&m, &[attn_dim]).unwrap());
        }
    }
    let fa_masks: Option<&[rdna_compute::GpuTensor]> = if slice { None } else { Some(&masks) };

    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch_tp(
            &mut gpus, &shard, &weights, &configs, tok, i, &mut kvs, &mut dns, &scratches, fa_masks,
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
    let t_decode = std::time::Instant::now();
    for step in 1..N_DECODE {
        let pos = prompt_tokens.len() + step - 1;
        let in_tok = forced[step - 1]; // walk the reference path
        qwen35::forward_scratch_tp(
            &mut gpus, &shard, &weights, &configs, in_tok, pos, &mut kvs, &mut dns, &scratches, fa_masks,
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
    let dsecs = t_decode.elapsed().as_secs_f64();
    eprintln!(
        "[timing] TP={} ({}) decode: {:.1} tok/s ({} tok / {:.3}s, incl 2 all-reduces/FA layer + rank-0 logits D2H)",
        TP, if slice { "3b sliced" } else { "3 replicated" },
        (N_DECODE - 1) as f64 / dsecs, N_DECODE - 1, dsecs
    );

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

// ── Stage 3d: batched-TP PREFILL parity + timing ──────────────────────────
//
// `run_prefill(path, tokens, tp)`: batched prefill via `forward_prefill_chunk_tp`.
//   tp=1 → fair single-GPU baseline (Full-phase bodies, NO all-reduce).
//   tp≥2 → sharded (sliced weights + local config + 2 all-reduces/layer).
// Same kernels both ways — tp just slices — so it's an apples-to-apples bench.
// Returns rank-0 LAST-token logits + prints prefill tok/s. Warms (kernel cache
// + DPM) with a throwaway prefill, then resets dn so the measured run starts at
// state=0 / pos=0 (KV slots are overwritten by the measured prefill).
fn run_prefill(path: &str, prompt_tokens: &[u32], tp: usize) -> Vec<f32> {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let dim = config.dim;
    let n = prompt_tokens.len();
    // Floor max_batch at 16: a batched GEMM can over-read/write to its tile
    // boundary (BLOCK_M up to 16); a 1-row pbs HIP-700s. The chunk still
    // processes only `n` rows — extra capacity is unused scratch.
    let pb_batch = n.max(16);

    let shard = ShardConfig::new(tp, false, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();

    let mut gpus = Gpus::init_tp(tp, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let st = dev.hip.stream_create().expect("stream");
        dev.active_stream = Some(st);
    }

    // tp=1 → FULL weights + FULL config on 1 GPU (the batched single-GPU
    // baseline). tp≥2 → per-rank SLICED weights + local config (the sharded
    // path). Both run forward_prefill_chunk_tp's batched bodies; tp=1 takes the
    // Full phase (no all-reduce). Using the production full loader at tp=1
    // avoids exercising load_weights_tp's degenerate tp=1 slice ranges.
    let configs: Vec<qwen35::Qwen35Config> =
        (0..tp).map(|_| if tp == 1 { config.clone() } else { qwen35::local_attn_config(&config, &shard) }).collect();

    let mut weights = Vec::with_capacity(tp);
    let mut scratches = Vec::with_capacity(tp);
    let mut kvs = Vec::with_capacity(tp);
    let mut dns = Vec::with_capacity(tp);
    let mut pbs_vec = Vec::with_capacity(tp);
    let mut partials = Vec::with_capacity(tp);
    for r in 0..tp {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(if tp == 1 {
            qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap()
        } else {
            qwen35::load_weights_tp(&mut hfq, &config, &mut gpus.devices[r], &shard, r).unwrap()
        });
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &configs[r], 128).unwrap());
        kvs.push(make_kv(&mut gpus.devices[r], &configs[r]));
        // Prefill is ALWAYS q8 DeltaNet state: the batched chunk kernel
        // `gated_delta_net_q8_batch_seq` reads the q8 layout (s_q8 int8 +
        // s_scales) and there is no fp32 batched variant (fp32 state OOB-reads
        // s_scales — faults at full head count). Production prefill is q8 state
        // too (DeltaNetState defaults to Q8). The HIPFIRE_PARITY_STATE knob
        // only applies to the single-token DECODE parity path.
        dns.push(DeltaNetState::new_with_quant(
            &mut gpus.devices[r], &configs[r], StateQuant::Q8,
        ).unwrap());
        pbs_vec.push(qwen35::PrefillBatchScratch::new(&mut gpus.devices[r], &configs[r], pb_batch).unwrap());
        // partial sized at pb_batch (not n): the wo/w_down GEMM can write to a
        // tile boundary past row n; the all-reduce/add only touch the live n·dim.
        partials.push(gpus.devices[r].zeros(&[pb_batch * dim], rdna_compute::DType::F32).unwrap());
    }

    // Warm, then reset dn for a clean measured run (state=0, pos=0).
    qwen35::forward_prefill_chunk_tp(
        &mut gpus, &shard, &weights, &configs, prompt_tokens, 0,
        &mut kvs, &mut dns, &pbs_vec, &partials, &scratches,
    ).expect("warm prefill_tp");
    gpus.devices[0].bind_thread().unwrap();
    gpus.devices[0].hip.device_synchronize().unwrap();
    for r in 0..tp {
        gpus.devices[r].bind_thread().unwrap();
        dns[r].reset(&mut gpus.devices[r]);
    }

    let t = std::time::Instant::now();
    qwen35::forward_prefill_chunk_tp(
        &mut gpus, &shard, &weights, &configs, prompt_tokens, 0,
        &mut kvs, &mut dns, &pbs_vec, &partials, &scratches,
    ).expect("prefill_tp");
    // forward_prefill_chunk_tp device-synchronizes rank 0 before returning.
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "[timing] TP={} ({}) prefill: {:.1} tok/s ({} tok / {:.4}s)",
        tp, if tp == 1 { "baseline, 1 GPU" } else { "3d sharded" },
        n as f64 / secs, n, secs,
    );

    gpus.devices[0].bind_thread().unwrap();
    let logits = gpus.devices[0].download_f32(&scratches[0].logits).expect("download logits");

    // Teardown: free every per-rank allocation + drain the pool so this call
    // leaves VRAM clean. Otherwise the leftover (~15 GB of 27B weights) makes
    // the NEXT run_prefill's init_tp preflight_vram trip on the device imbalance
    // and eventually OOM as allocations accumulate across the 4 bench calls.
    for r in (0..tp).rev() {
        gpus.devices[r].bind_thread().unwrap();
        if let Some(st) = gpus.devices[r].active_stream.take() {
            let _ = gpus.devices[r].hip.stream_destroy(st);
        }
        if let Some(p) = partials.pop() { let _ = gpus.devices[r].free_tensor(p); }
        if let Some(pb) = pbs_vec.pop() { pb.free_gpu(&mut gpus.devices[r]); }
        if let Some(s) = scratches.pop() { s.free_gpu(&mut gpus.devices[r]); }
        if let Some(d) = dns.pop() { d.free_gpu(&mut gpus.devices[r]); }
        if let Some(k) = kvs.pop() { k.free_gpu(&mut gpus.devices[r]); }
        if let Some(w) = weights.pop() { w.free_gpu(&mut gpus.devices[r]); }
        gpus.devices[r].drain_pool();
    }
    logits
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

    // Stage 3e EP-MoE LOADER smoke test (HIPFIRE_PARITY_MOE_LOADTEST=1): load the
    // (A3B) model via load_weights_tp on each rank and verify routed experts are
    // sharded (compacted to num_experts/tp per layer) without crashing. Tests the
    // loader in isolation — the EP forward (forward_scratch_tp MoE arms) is a
    // separate step; this does NOT run a forward.
    if std::env::var("HIPFIRE_PARITY_MOE_LOADTEST").is_ok() {
        let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
        let config = qwen35::config_from_hfq(&hfq).expect("config");
        println!("=== Stage 3e EP-MoE loader smoke test ===");
        println!(
            "model: num_experts={} num_experts_per_tok={} tp={TP}, expect {} experts/layer/rank",
            config.num_experts, config.num_experts_per_tok, config.num_experts / TP,
        );
        assert!(config.num_experts > 0, "MOE_LOADTEST needs an MoE model (num_experts>0)");
        // tp_kv_replicate=true: A3B attention KV heads may not divide tp; EP v1
        // replicates attention anyway, so KV replication is the right setting.
        let shard = ShardConfig::new(TP, true, config.num_experts, ExpertAssign::Stride).unwrap();
        let mut gpus = Gpus::init_tp(TP, config.n_layers).expect("init_tp");
        for r in 0..TP {
            gpus.devices[r].bind_thread().unwrap();
            let w = qwen35::load_weights_tp(&mut hfq, &config, &mut gpus.devices[r], &shard, r)
                .expect("load_weights_tp (A3B EP)");
            let mut moe_layers = 0usize;
            let mut first_count: Option<usize> = None;
            let mut all_match = true;
            // Stage 3f: also capture a MoE-layer attention dim (FullAttnMoe wq.m
            // output rows; DeltaNetMoe wo.k input cols) — halved when
            // HIPFIRE_EP_SHARD_ATTN is set (sharded attn), full otherwise (3e).
            let mut fa_wq_m: Option<usize> = None;
            let mut dn_wo_k: Option<usize> = None;
            for lw in &w.layers {
                let experts = match lw {
                    qwen35::LayerWeights::DeltaNetMoe(l) => { dn_wo_k.get_or_insert(l.wo.k); &l.ffn.experts }
                    qwen35::LayerWeights::FullAttnMoe(l) => { fa_wq_m.get_or_insert(l.wq.m); &l.ffn.experts }
                    _ => continue,
                };
                moe_layers += 1;
                first_count.get_or_insert(experts.len());
                if experts.len() != config.num_experts / TP { all_match = false; }
            }
            let shard_attn = std::env::var_os("HIPFIRE_EP_SHARD_ATTN").is_some();
            println!(
                "  rank {r}: {moe_layers} MoE layers, experts/layer={} {} | attn: FullAttnMoe wq.m={:?} DeltaNetMoe wo.k={:?} (shard_attn={})",
                first_count.unwrap_or(0),
                if all_match && first_count == Some(config.num_experts / TP) { "✓" } else { "✗ MISMATCH" },
                fa_wq_m, dn_wo_k, shard_attn,
            );
            assert!(all_match, "rank {r}: some MoE layer not compacted to {} experts", config.num_experts / TP);
            assert_eq!(first_count, Some(config.num_experts / TP), "rank {r}: expert count wrong");
        }
        println!("\ntp_attn_parity (MoE loader): PASS — A3B experts sharded {}-way, non-owned freed + dummy-filled ptr table.", TP);
        return;
    }

    println!("=== TP=2 ↔ TP=1 full-model parity (Stage 3 acceptance) ===");
    println!("prompt: {PROMPT:?}  (prompt_len={}, N_DECODE={N_DECODE})", prompt_tokens.len());
    println!(
        "precision: KV={}  DeltaNet-state={}",
        kv_mode(),
        if state_is_q8() { "q8" } else { "fp32" },
    );
    println!(
        "TP mode: {}",
        if std::env::var("HIPFIRE_PARITY_SLICE").is_ok() {
            "SLICED (3b — per-rank weight slices, local-head compute, no masks)"
        } else {
            "replicated (3 — full weights on every rank + head masks)"
        }
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

    // Stage 3d: batched-TP PREFILL parity + timing. HIPFIRE_PARITY_PREFILL=1
    // runs single-chunk prefill on 1 GPU (forward_prefill_batch) vs TP=2
    // (forward_prefill_chunk_tp). HIPFIRE_PARITY_PREFILL_LEN=L tiles/truncates
    // the prompt to L tokens for a meatier perf datapoint (token VALUES don't
    // affect compute cost or parity — both paths see the identical sequence).
    //
    // TWO-TIER gate (the batched chunk kernel `gated_delta_net_q8_batch_seq`
    // is q8-ONLY — no fp32 batched variant — so it quantizes the recurrent
    // state internally over the N-step chunk regardless of the STORED state
    // dtype; this amplifies TP's unavoidable ~1e-7 all-reduce reassociation to
    // ~1e-3 at N≥2, argmax-stable — the 3c "q8 state amplifies TP" finding in
    // the prefill chunk):
    //   • N=1 sub-check exercises the FULL sharding pipeline (sliced weights,
    //     all-reduces, FA+DeltaNet) with no chunk compounding → the
    //     fp32-equivalent sharding-MATH gate, must be clean (< REL_TOL ~1e-6).
    //   • N=L run: assert argmax match (output correctness); rel is q8-chunk
    //     amplification (informational, grows with L, NOT a sharding bug).
    if std::env::var("HIPFIRE_PARITY_PREFILL").is_ok() {
        // The bench reloads the model per (tp, length) call; tp=1 loads full
        // weights on 1 GPU while tp=2 loads slices on 2 — a deliberately uneven
        // VRAM profile across the bench's fresh Gpus instances. Relax init_tp's
        // uniform-VRAM preflight (it guards real multi-GPU serving, not a bench).
        if std::env::var("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB").is_err() {
            std::env::set_var("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB", "64");
        }
        // Closure: run baseline (tp=1) + TP (tp=2) prefill at the given tokens.
        let parity_at = |toks: &[u32]| -> (f32, u32, u32) {
            let r = run_prefill(&path, toks, 1);
            let t = run_prefill(&path, toks, 2);
            let rmax = r.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let mut mab = 0.0f32;
            for (x, y) in r.iter().zip(t.iter()) { mab = mab.max((x - y).abs()); }
            (mab / (rmax + 1e-12), argmax(&r), argmax(&t))
        };

        let pt: Vec<u32> = match std::env::var("HIPFIRE_PARITY_PREFILL_LEN").ok().and_then(|s| s.parse::<usize>().ok()) {
            Some(l) if l > 0 => (0..l).map(|i| prompt_tokens[i % prompt_tokens.len()]).collect(),
            _ => prompt_tokens.clone(),
        };
        println!("\n=== Stage 3d: batched-TP PREFILL parity + timing ===");
        println!(
            "prefill_len={} (one chunk), KV={}, DeltaNet-state=q8 (forced — batched chunk kernel requires q8 state)",
            pt.len(), kv_mode(),
        );

        // ── Tier 1: N=1 sharding-math gate (must be clean) ──
        println!("\n── N=1 sharding-math gate (full sliced pipeline, no chunk compounding) ──");
        let (rel1, am1r, am1t) = parity_at(&[pt[0]]);
        println!("N=1 last-token rel Δ = {rel1:.3e} (tol {REL_TOL:.0e}); argmax ref={am1r} tp={am1t} {}",
            if am1r == am1t { "✓" } else { "✗" });
        assert!(rel1 < REL_TOL, "N=1 sharding-math parity FAILED: {rel1:.3e} ≥ {REL_TOL:.0e} — the SLICING/all-reduce math is wrong");
        assert_eq!(am1r, am1t, "N=1 argmax parity FAILED");

        // ── Tier 2: N=L perf-length run (argmax-strict; rel = q8-chunk amplification) ──
        println!("\n── N={} prefill (perf + argmax; rel reflects q8 chunk-recurrence) ──", pt.len());
        println!("\n── TP=1 reference prefill (forward_prefill_batch) ──");
        let ref_logits = run_prefill(&path, &pt, 1);
        println!("\n── TP=2 sharded prefill (forward_prefill_chunk_tp) ──");
        let tp_logits = run_prefill(&path, &pt, 2);
        let ref_max = ref_logits.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut max_abs = 0.0f32;
        for (x, y) in ref_logits.iter().zip(tp_logits.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        let rel = max_abs / (ref_max + 1e-12);
        let am_ref = argmax(&ref_logits);
        let am_tp = argmax(&tp_logits);
        // Gross-error tripwire: q8-chunk amplification is ~1e-3; anything past
        // 1e-1 means a real bug, not quantization.
        const Q8_CHUNK_TRIP: f32 = 1e-1;
        println!("\n── prefill parity (last-token logits) ──");
        println!("last-token max rel logit Δ = {rel:.3e}  (q8-chunk amplification; N=1 math gate was {rel1:.3e})");
        println!(
            "last-token argmax: ref={am_ref} tp={am_tp}  {}",
            if am_ref == am_tp { "✓ match" } else { "✗ MISMATCH" },
        );
        assert!(rel < Q8_CHUNK_TRIP, "prefill rel {rel:.3e} ≥ {Q8_CHUNK_TRIP:.0e} — too large for q8-chunk amplification, likely a real bug");
        assert_eq!(am_ref, am_tp, "prefill argmax parity FAILED");
        println!("\ntp_attn_parity (prefill): PASS  (N=1 math gate {rel1:.3e} < {REL_TOL:.0e}; N={} argmax match, rel {rel:.3e})", pt.len());
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
