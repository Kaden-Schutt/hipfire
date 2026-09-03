// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 end-to-end smoke (Task 10): load the WHOLE `.hfq` single-GPU and
//! run PREFILL and then decode through the production forward paths.
//!
//! The G4 block gate (`escha_moe_block_gate`) calls the routed executor
//! directly, with routing injected. That proves the maths; it does NOT prove
//! that a real `qwen35::forward` ever reaches it. This does: it asserts layer
//! 0 came through the escha loader, then decodes and reads the H128 launch
//! counter, which must be exactly `4 * n_layers` per token — the batched
//! budget. A regression to a per-expert wiring shows up here as `4 * k *
//! n_layers` (1280 at A3B) rather than 160, with no numerical change at all.
//!
//! # The prefill phase, and why the launch counter is the load-bearing assert
//!
//! Escha layers are expected to route to the escha executor in prefill too,
//! but by a completely different mechanism from decode. In prefill,
//! `moe_ffn_batched_admissible_for_dtypes` admits no `Q8_0` routed arm, so
//! `prefill_batch_pbs_eligible` comes out false for the WHOLE model and
//! `forward_prefill_batch` falls to its per-token `forward_scratch` loop —
//! which is byte-identical to decode and therefore reaches
//! `run_moe_decode` → `run_moe_decode_cpu_fallback` → the escha executor.
//!
//! That chain was INFERRED from source, not observed, and the failure mode if
//! the inference is wrong is silent wrong output: a batched MoE prefill body
//! would run the Q8_0 experts with no H128 pair and produce finite, fluent,
//! ~1e-1-wrong hidden state. Finiteness alone would not catch it. The
//! `escha_h128_launches()` delta would: on the batched body it is ZERO, on the
//! correct per-token path it is exactly `n_prompt * 4 * n_layers`. So the
//! prefill phase below asserts BOTH, and the counter is what actually proves
//! the escha executor ran for every layer of every prompt token.
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

    // ── Phase 1: PREFILL ─────────────────────────────────────────────────
    // 8 tokens, matching the G4 fixture width, through the real batched
    // prefill entry point (which is expected to fall through to its per-token
    // loop — see the module docs).
    const PROMPT: [u32; 8] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
    let before_prefill = rdna_compute::escha_h128_launches();
    let t = std::time::Instant::now();
    qwen35::forward_prefill_batch(
        ctx.gpu,
        &b.weights,
        &b.config,
        &PROMPT,
        0,
        &mut b.kv_cache,
        &mut b.dn_state,
        &b.scratch,
        None, // hidden ring
        None, // per-token hidden out — keep last-token logits enabled
        None, // gdn tape
        None, // tree verify
    )
    .map_err(|e| format!("prefill: {e:?}"))?;
    ctx.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let prefill_launches = rdna_compute::escha_h128_launches() - before_prefill;
    let prefill_logits = ctx
        .gpu
        .download_f32(&b.scratch.logits)
        .map_err(|e| format!("download prefill logits: {e:?}"))?;
    let prefill_bad = prefill_logits
        .iter()
        .take(b.config.vocab_size)
        .filter(|v| !v.is_finite())
        .count();
    eprintln!(
        "prefill n={}: H128 launches={prefill_launches} ({} per token, want {want_launches}), \
         non-finite logits={prefill_bad}/{}, {:?}",
        PROMPT.len(),
        prefill_launches / PROMPT.len() as u64,
        b.config.vocab_size,
        t.elapsed()
    );
    assert_eq!(
        prefill_bad,
        0,
        "non-finite logits after an {}-token prefill",
        PROMPT.len()
    );
    assert_eq!(
        prefill_launches as usize,
        PROMPT.len() * want_launches,
        "PREFILL took a path that did not run the escha executor for every (layer, token). \
         Expected {} H128 launches ({} tokens x 4 x {} layers), saw {prefill_launches}. Zero \
         means the model was admitted to a BATCHED MoE prefill body, which has no escha \
         awareness and would emit finite-but-~1e-1-wrong hidden state — check \
         `moe_ffn_batched_admissible_for_dtypes` for a newly-added Q8_0 routed arm.",
        PROMPT.len() * want_launches,
        PROMPT.len(),
        b.config.n_layers
    );

    // ── Phase 2: DECODE, continuing from the prefilled context ───────────
    let mut prev = rdna_compute::escha_h128_launches();
    for (i, &tok) in [9000u32, 10000, 11000, 12000].iter().enumerate() {
        let pos = PROMPT.len() + i;
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
        for (j, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                argmax = j;
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
