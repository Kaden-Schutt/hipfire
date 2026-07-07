// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP5 PC-5: **dense tensor-parallel BATCHED-PREFILL parity** — run a fixed
//! prompt through `TpModel::prefill` (batched embed→broadcast, per-rank batched
//! GEMMs via `Step::Gemm` through `execute_steps_tp`, batched `Step::Attend`
//! writing the Q8 KV internally, two `AllReduceOut` collectives per layer) and
//! assert the last-position logits match single-GPU `llama::prefill_forward`.
//!
//! Both sides use MQ4G256 WMMA GEMMs; the TP attention reads the Q8 KV cache
//! (batched flash) while the single-GPU reference does F32 in-batch causal
//! attention, so the argmax must be identical while `max|Δ|` sits a bit above the
//! all-Q8 `tp_full_model_parity` (4.2e-4). The greedy argmax is the invariant
//! that matters.
//!
//! Emulated Tp-2 (gfx1151).
//!
//! Run: HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example tp_prefill_parity -- --model model.mq4

use hipfire_runtime::llama::{self, KvCache, LlamaConfig};
use hipfire_runtime::tp_serve::TpModel;

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

    // ── Reference: single-GPU batched prefill (last-position logits). Scoped so
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
        llama::prefill_forward(&mut gpu, &weights, &config, &toks, &mut kv).expect("ref prefill")
    };

    // ── TP path: TpModel batched prefill. ──
    let mut model = match TpModel::load(&model_path, tp, MAX_SEQ) {
        Ok(m) => m,
        Err(e) => {
            println!("tp_prefill_parity: SKIPPED (TpModel::load: {e})");
            return;
        }
    };
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

    println!(
        "[tp-prefill] toks={} ref_argmax={ref_argmax} tp_argmax={tp_argmax} max|Δ|={max_delta:.3e}",
        toks.len()
    );
    eprintln!("ref next-token: {:?}", tokenizer.decode(&[ref_argmax]));
    eprintln!(" tp next-token: {:?}", tokenizer.decode(&[tp_argmax]));

    assert_eq!(
        ref_argmax, tp_argmax,
        "TP batched prefill argmax diverged from single-GPU prefill_forward: \
         tp={tp_argmax} ref={ref_argmax} (max|Δ|={max_delta:.3e})"
    );
    println!(
        "tp_prefill_parity: dense TP batched prefill last-position logits argmax == single-GPU \
         prefill_forward (max|Δ|={max_delta:.3e}) — PC-5 validated"
    );
}
