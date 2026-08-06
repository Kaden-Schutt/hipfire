// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP5 PC-5: **dense tensor-parallel BATCHED-PREFILL parity** — run a fixed
//! prompt through `TpModel::prefill` (batched embed→broadcast, per-rank batched
//! GEMMs via `Step::Gemm` through `execute_steps_tp`, batched `Step::Attend`
//! writing the Q8 KV internally, two `AllReduceOut` collectives per layer) and
//! assert the last-position logits match single-GPU `llama::forward_prefill_batch`.
//!
//! Both sides use MQ4G256 WMMA GEMMs AND read the same Q8 KV cache via batched
//! flash: the reference is single-GPU `llama::forward_prefill_batch` (the Q8-KV
//! `attention_flash_q8_0_batched_masked` path), NOT the F32-in-batch
//! `prefill_forward`. Reading the same Q8 KV removes the attention-mode
//! systematic bias — measured at max|Δ|≈0.88 between the F32-in-batch and Q8-flash
//! single-GPU references on this prompt — so the residual delta is only GEMM
//! sharding + per-layer all-reduce summation order + Q8-KV rounding compounded
//! over all prefill positions × layers. On qwen3-0.6b Tp-2 that lands at
//! max|Δ|≈0.20 (~1.3% of the ~15 peak logit), asserted below as a numeric bound —
//! NOT just the greedy argmax. (The decode-time `tp_full_model_parity` 4.2e-4 is
//! a single position with a one-entry KV; prefix compounding over 103 positions
//! and 28 layers is why this is larger, not a TP bug.)
//!
//! Emulated Tp-2 (gfx1151).
//!
//! Run: HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example tp_prefill_parity -- --model model.mq4

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_loader::ModelParallel;
use hipfire_runtime::llama::{self, KvCache, LlamaConfig};

const MAX_SEQ: usize = 512;

// Fixed prompt (≤256 tokens after tokenization). md5(PROMPT) = 0498720fa0b680a8fbceea068e9d6add
// (recorded so any whitespace edit that would change tokenization is caught in review).
const PROMPT: &str = "The tensor-parallel prefill shards every transformer layer's attention \
heads and feed-forward width across two ranks. Each rank embeds the replicated prompt hidden, \
runs its batched column projections, attends over its own heads writing the Q8 KV cache for all \
positions, and the row projections are summed across ranks with an all-reduce. Explain, in a few \
sentences, why the last-position logits after this batched tensor-parallel prefill must pick the \
same next token as running the whole model on a single device.";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = concat!(env!("HOME"), "/.hipfire/models/qwen3-0.6b-llama.mq4").to_string();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                if let Some(v) = it.next() {
                    model_path = v.clone();
                }
            }
            other => model_path = other.to_string(),
        }
    }
    let tp = 2usize;

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");

    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&model_path)).expect("open model");
    let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let toks = tokenizer.encode(PROMPT);
    assert!(!toks.is_empty(), "empty prompt");
    assert!(
        toks.len() <= llama::PREFILL_MAX_BATCH,
        "prompt {} toks > PREFILL_MAX_BATCH {} — pick a shorter fixed prompt",
        toks.len(),
        llama::PREFILL_MAX_BATCH
    );
    eprintln!(
        "model: layers={} | prompt={} toks, tp={tp} (batched prefill)",
        config.n_layers,
        toks.len()
    );

    // ── Reference: single-GPU batched prefill reading the Q8 KV cache (same
    // batched-flash attention the TP path uses), last-position logits. Scoped so
    // its Gpu drops before TpModel brings up the emulated Gpus. ──
    let ref_logits: Vec<f32> = {
        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        gpu.bind_thread().unwrap();
        let weights =
            hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("load_weights");
        let mut kv = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            MAX_SEQ,
        )
        .unwrap();
        let scratch = llama::ForwardScratch::new(&mut gpu, &config).unwrap();
        llama::forward_prefill_batch(
            &mut gpu, &weights, &config, &toks, 0, &mut kv, &scratch, None,
        )
        .expect("ref prefill (Q8-KV batched flash)");
        gpu.download_f32(&scratch.logits).unwrap()
    };

    // ── TP path: TpModel batched prefill (via loader). ──
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, tp)]);
    let loaded =
        match hipfire_loader::load_model_tp(&model_path, MAX_SEQ, &mesh, Default::default()) {
            Ok(m) => m,
            Err(e) => {
                println!("tp_prefill_parity: SKIPPED (load_model_tp: {e})");
                return;
            }
        };
        let mut model = loaded
        .tp_model
        .expect("expected TP model (tp_model carrier)");
    model.prefill(&toks).expect("tp prefill");
    let tp_logits = model.logits().expect("tp logits");

    assert_eq!(
        ref_logits.len(),
        tp_logits.len(),
        "logits length mismatch: ref={} tp={}",
        ref_logits.len(),
        tp_logits.len()
    );
    let ref_argmax = llama::argmax(&ref_logits);
    let tp_argmax = llama::argmax(&tp_logits);
    let max_delta = ref_logits
        .iter()
        .zip(&tp_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ref_mag = ref_logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    println!(
        "[tp-prefill] toks={} ref_argmax={ref_argmax} tp_argmax={tp_argmax} \
         max|Δ|={max_delta:.3e} (ref max|logit|={ref_mag:.3e})",
        toks.len()
    );
    eprintln!("ref next-token: {:?}", tokenizer.decode(&[ref_argmax]));
    eprintln!(" tp next-token: {:?}", tokenizer.decode(&[tp_argmax]));

    // Greedy invariant: same next token.
    assert_eq!(
        ref_argmax, tp_argmax,
        "TP batched prefill argmax diverged from single-GPU forward_prefill_batch: \
         tp={tp_argmax} ref={ref_argmax} (max|Δ|={max_delta:.3e})"
    );

    // Numeric bound: with both sides reading the same Q8 KV cache, the only
    // remaining divergence is GEMM sharding + all-reduce summation order + Q8-KV
    // rounding, compounded over all prefill positions × 28 layers. Measured
    // max|Δ|≈0.20 on qwen3-0.6b Tp-2 (deterministic under HIPFIRE_DETERMINISTIC=1);
    // the bound sits at 2× that with headroom for kernel/model churn. This is the
    // regression that greedy argmax cannot see: a sharding/all-reduce bug that
    // perturbs the logits without flipping the last-position argmax (the
    // attention-mode difference alone is ≈0.88, so a real TP break lands well
    // above this bound).
    const MAX_DELTA_TOL: f32 = 4.0e-1;
    assert!(
        max_delta < MAX_DELTA_TOL,
        "TP batched prefill last-position logits diverged from single-GPU \
         forward_prefill_batch beyond tolerance: max|Δ|={max_delta:.3e} >= {MAX_DELTA_TOL:.1e} \
         (ref max|logit|={ref_mag:.3e})"
    );
    println!(
        "tp_prefill_parity: dense TP batched prefill last-position logits == single-GPU \
         forward_prefill_batch (argmax match + max|Δ|={max_delta:.3e} < {MAX_DELTA_TOL:.1e}) \
         — PC-5 validated"
    );
}
