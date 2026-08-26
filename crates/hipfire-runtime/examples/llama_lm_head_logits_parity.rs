// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity test for Task 2b: `LlamaBundle::lm_head_logits`.
//!
//! Loads `bielik-minitron-7b.mq4` (40-layer dense LLaMA, hidden=4096),
//! prefills a single-token prompt via `SpecTarget::spec_advance` (uses the
//! per-token decode path since n=1 < MIN_BATCH=4, leaving the final hidden
//! state in `scratch.x` and the last-row logits downloadable as `last_argmax`),
//! then calls `lm_head_logits` on the same `scratch.x` tensor and asserts:
//!
//!   `argmax(lm_head_logits(scratch.x, 1)) == last_argmax`
//!
//! This proves the implementation is bit-identical to the existing per-token
//! forward path used by `spec_advance`.
//!
//! Usage:
//!   cargo run --release --example llama_lm_head_logits_parity
//!   cargo run --release --example llama_lm_head_logits_parity -- <model.mq4>

use hipfire_arch_llama::load_llama_bundle;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::spec::{SpecAdvance, SpecTarget};

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i as u32)
        .expect("empty logits")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_path = format!(
        "{}/.hipfire/models/bielik-minitron-7b.mq4",
        std::env::var("HOME").unwrap_or_else(|_| ".".into())
    );
    let model_path = args.get(1).cloned().unwrap_or(default_path);

    eprintln!("[task2b] opening {model_path}");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    let src = ModelSource::from_path(&model_path).expect("open model source");
    let cask = CaskConfig::default();
    let mut ctx = LoadCtx {
        path: &model_path,
        max_seq: 2048,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
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
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
    };

    let mut bundle = load_llama_bundle(src, &mut ctx).expect("load llama bundle");
    let hidden = bundle.config.dim;
    let vocab = bundle.config.vocab_size;
    let n_layers = bundle.config.n_layers;
    eprintln!("[task2b] loaded: hidden={hidden}, vocab={vocab}, n_layers={n_layers}");

    // Single-token prompt: uses the per-token fallback path (n=1 < MIN_BATCH=4),
    // which calls `forward_scratch_embed + forward_scratch_compute`. After the
    // call, `scratch.x` holds the final hidden state (pre-norm) and
    // `scratch.logits` holds vocab logits (both on GPU).
    let prompt: Vec<u32> = vec![1u32];
    let no_abort = || false;

    let adv = bundle
        .spec_advance(&mut gpu, &prompt, 0, true, &no_abort, None)
        .expect("spec_advance");

    let last_argmax = match adv {
        SpecAdvance::Ready { last_argmax, .. } => {
            eprintln!("[task2b] spec_advance Ready, last_argmax={last_argmax}");
            last_argmax
        }
        other => panic!("expected SpecAdvance::Ready, got {other:?}"),
    };

    // `scratch.x` (on GPU, F32, dim-element) is the final hidden state from
    // `forward_scratch_compute` — use it as the 1-row input to `lm_head_logits`.
    // `shallow_clone` gives a non-owning view so we can still call `&mut bundle`.
    let hidden_tensor = bundle.scratch.x.shallow_clone();

    let logits = bundle
        .lm_head_logits(&mut gpu, &hidden_tensor, 1)
        .expect("lm_head_logits returned Err");

    assert_eq!(
        logits.len(),
        vocab,
        "expected {} logits (1 row × vocab), got {}",
        vocab,
        logits.len()
    );

    let got_argmax = argmax(&logits);
    eprintln!("[task2b] lm_head_logits argmax={got_argmax}, expected={last_argmax}");

    if got_argmax == last_argmax {
        println!("PASS");
    } else {
        eprintln!(
            "[task2b] FAIL: argmax mismatch — lm_head_logits={got_argmax}, spec_advance={last_argmax}"
        );
        std::process::exit(1);
    }
}
