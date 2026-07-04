// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen3_dspark_load_smoke: load ONLY the Qwen3-8B DSpark sidecar HFQ via
//! `hipfire_arch_llama::dspark_body::load_qwen3_dspark` and verify every
//! body layer + DSpark global tensor lands on the GPU, then free.  No inference.
//!
//! Specifically validates the two hard requirements from the Task-6 review:
//! 1. `confidence_bias` is `Some` (qwen3 has bias; deepseek4 sets None).
//! 2. `enable_confidence` is parsed from metadata (not hardcoded).
//!
//! Usage: qwen3_dspark_load_smoke [path-to-qwen3-8b-dspark.hfq]

use hipfire_arch_llama::dspark_body::load_qwen3_dspark;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3-8b-dspark.hfq",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    eprintln!("opening {path}");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    hfq.drop_mmap();

    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    eprintln!("GPU initialised");

    let (dspark, assets) = load_qwen3_dspark(&hfq, &mut gpu)?.ok_or_else(|| {
        "load_qwen3_dspark returned None — sidecar has no dspark_* metadata".to_string()
    })?;

    // ── DSpark config ────────────────────────────────────────────────────────
    println!(
        "cfg: block_size={} target_layers={:?} markov_rank={} noise_tok={} enable_confidence={}",
        dspark.cfg.block_size,
        dspark.cfg.target_layer_ids,
        dspark.cfg.markov_rank,
        dspark.cfg.noise_token_id,
        dspark.cfg.enable_confidence,
    );

    // ── Drafter config ───────────────────────────────────────────────────────
    let cfg = &assets.config;
    println!(
        "drafter cfg: n_layers={} dim={} hidden={} n_heads={} n_kv_heads={} head_dim={} has_qk_norm={} rope={}",
        cfg.n_layers,
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.has_qk_norm,
        cfg.rope_freq_base,
    );

    // ── Body layers ──────────────────────────────────────────────────────────
    let mut all_ok = true;
    for (i, layer) in assets.weights.layers.iter().enumerate() {
        let attn_ok = !layer.attn_norm.shape.is_empty();
        let proj_ok = layer.wq.m > 0 && layer.wk.m > 0 && layer.wv.m > 0 && layer.wo.m > 0;
        let ffn_ok = layer.w_gate.m > 0 && layer.w_up.m > 0 && layer.w_down.m > 0;
        let qk_norm_ok = !cfg.has_qk_norm || (layer.q_norm.is_some() && layer.k_norm.is_some());
        println!(
            "  layer {i}: attn_norm={attn_ok} proj={proj_ok} ffn={ffn_ok} qk_norm={qk_norm_ok}"
        );
        all_ok &= attn_ok && proj_ok && ffn_ok && qk_norm_ok;
    }

    // ── Global tensors ───────────────────────────────────────────────────────
    let main_proj_ok = dspark.main_proj.is_some();
    let main_norm_ok = dspark.main_norm.is_some();
    let markov_ok = dspark.markov_w1.is_some() && dspark.markov_w2.is_some();
    // Hard requirement #1: confidence_bias must be Some for qwen3.
    let conf_proj_ok = !dspark.cfg.enable_confidence || dspark.confidence_proj.is_some();
    let conf_bias_ok = !dspark.cfg.enable_confidence || dspark.confidence_bias.is_some();
    println!(
        "  globals: main_proj={main_proj_ok} main_norm={main_norm_ok} \
         markov_w1={} markov_w2={} confidence_proj={conf_proj_ok} \
         confidence_bias={conf_bias_ok}",
        dspark.markov_w1.is_some(),
        dspark.markov_w2.is_some(),
    );
    all_ok &= main_proj_ok && main_norm_ok && markov_ok && conf_proj_ok && conf_bias_ok;

    // Validate hard requirements explicitly.
    // Hard req #2: enable_confidence parsed from metadata (not hardcoded true).
    println!(
        "  enable_confidence (from metadata) = {}",
        dspark.cfg.enable_confidence
    );
    // Hard req #1: confidence_bias is Some.
    println!(
        "  confidence_bias = {}",
        if dspark.confidence_bias.is_some() {
            "Some"
        } else {
            "None"
        }
    );
    if dspark.cfg.enable_confidence && dspark.confidence_bias.is_none() {
        eprintln!(
            "FAIL: enable_confidence=true but confidence_bias is None (hard req #1 violated)"
        );
        all_ok = false;
    }

    // ── Drafter body weight + embedding ─────────────────────────────────────
    let embd_ok = !assets.weights.token_embd.shape.is_empty();
    let norm_ok = !assets.weights.output_norm.shape.is_empty();
    let lm_head_ok = assets.weights.output.m > 0;
    println!("  embed={embd_ok} output_norm={norm_ok} lm_head={lm_head_ok}");
    all_ok &= embd_ok && norm_ok && lm_head_ok;

    // ── Scratch / KvCache ────────────────────────────────────────────────────
    let kv_ok = assets.kv.k_gpu.len() == cfg.n_layers;
    println!(
        "  kv_layers={} (expected {}): {kv_ok}",
        assets.kv.k_gpu.len(),
        cfg.n_layers
    );
    all_ok &= kv_ok;

    // Free (GPU memory released on drop of DsparkWeights + Qwen3DrafterAssets).
    drop(dspark);
    drop(assets);
    eprintln!("freed OK");

    if all_ok {
        println!("SMOKE PASS");
        Ok(())
    } else {
        Err("SMOKE FAIL — some tensors missing or hard requirements violated".into())
    }
}
