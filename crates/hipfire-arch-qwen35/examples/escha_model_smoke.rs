// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 end-to-end smoke (Task 10): load the WHOLE `.hfq` single-GPU and
//! run a few decode steps through the production forward path.
//!
//! The G4 block gate (`escha_moe_block_gate`) calls the routed executor
//! directly, with routing injected. That proves the maths; it does NOT prove
//! that a real `qwen35::forward` ever reaches it. This does: it asserts layer
//! 0 came through the escha loader, then decodes and reads the H128 launch
//! counter, which must be exactly `4 * n_layers` per token — the batched
//! budget. A regression to a per-expert wiring shows up here as `4 * k *
//! n_layers` (1280 at A3B) rather than 160, with no numerical change at all.
//!
//! Token ids are arbitrary — the converter does not embed a tokenizer, so
//! `hipfire run` cannot drive this checkpoint yet. The assertions are
//! therefore structural (finite logits, a non-degenerate argmax, the launch
//! budget), not semantic.
//!
//! COST: the Q8_0 experts are ~32 GiB resident (40 layers x 256 experts x
//! 3.19 MiB) plus ~4 GiB of everything else. On a 128 GB workstation with
//! other applications running this is close to the limit — it OOMs with
//! ~60 GB already in use. Not something to run casually.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_model_smoke -- /data/hipfire-models/escha-35b.hfq
use hipfire_arch_qwen35::qwen35;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/hipfire-models/escha-35b.hfq".to_string());
    let hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let cask = CaskConfig::default();
    let src = ModelSource::Hfq(hfq);
    let mut ctx = LoadCtx {
        path: &path,
        max_seq: 512,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        kv_mode_override: None,
        kv_backend: hipfire_runtime::kv_backend::KvBackend::Contiguous,
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
    };
    let t0 = std::time::Instant::now();
    let mut b = hipfire_arch_qwen35::load_qwen35_bundle(src, &mut ctx)?;
    eprintln!("loaded in {:?}", t0.elapsed());

    // Layer 0 must have come through the escha loader, and its experts must be
    // the Q8_0 the trellis decoded into — not whatever the generic per-expert
    // path would have found.
    match &b.weights.layers[0] {
        qwen35::LayerWeights::DeltaNetMoe(l) => {
            assert!(l.ffn.escha.is_some(), "layer 0 carries no escha tables");
            assert_eq!(
                l.ffn.experts[0].gate_up.gpu_dtype,
                rdna_compute::DType::Q8_0
            );
            eprintln!(
                "layer0: escha=Some experts={} gate_up dtype={:?} m={} k={}",
                l.ffn.experts.len(),
                l.ffn.experts[0].gate_up.gpu_dtype,
                l.ffn.experts[0].gate_up.m,
                l.ffn.experts[0].gate_up.k
            );
        }
        _ => panic!("layer 0 is not a DeltaNet+MoE layer"),
    }

    let want_launches =
        hipfire_dispatch::pipeline::escha::escha_launches_per_token(b.config.n_layers);
    let mut prev = rdna_compute::escha_h128_launches();
    for (pos, &tok) in [1000u32, 2000, 3000, 4000].iter().enumerate() {
        let t = std::time::Instant::now();
        let logits = qwen35::forward(
            ctx.gpu,
            &b.weights,
            &b.config,
            tok,
            pos,
            &mut b.kv_cache,
            &mut b.dn_state,
        )
        .map_err(|e| format!("forward: {e:?}"))?;

        let n_bad = logits.iter().filter(|v| !v.is_finite()).count();
        let mut best = f32::NEG_INFINITY;
        let mut argmax = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                argmax = i;
            }
        }
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let now = rdna_compute::escha_h128_launches();
        let launches = now - prev;
        prev = now;
        eprintln!(
            "pos {pos} tok {tok}: {} logits, non-finite={n_bad}, argmax={argmax} ({best:.4}), \
             mean={mean:.4}, H128 launches={launches}, {:?}",
            logits.len(),
            t.elapsed()
        );
        assert_eq!(n_bad, 0, "non-finite logits at pos {pos}");
        assert!(best > mean, "degenerate logit distribution at pos {pos}");
        assert_eq!(
            launches as usize, want_launches,
            "H128 launches per token drifted from the batched budget"
        );
    }
    eprintln!("escha_model_smoke: PASS");
    Ok(())
}
