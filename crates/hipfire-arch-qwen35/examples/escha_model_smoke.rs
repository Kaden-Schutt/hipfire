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
//! Escha layers route to an escha executor in prefill too, but through
//! `escha_routed_prefill_indexed` rather than the decode one. The H128 launch
//! budget is what identifies which of THREE things happened, and the three are
//! indistinguishable by looking at the logits:
//!
//! | launches (8-token prompt, 40 layers) | what ran |
//! |---|---|
//! | **160** = `4 * n_layers` per CHUNK | batched escha prefill — correct |
//! | 1 280 = `n * 4 * n_layers` | silently fell back to the per-token loop |
//! | 0 | a batched MoE body with NO escha awareness |
//!
//! Zero is the dangerous one: the generic batched routed body would run the
//! Q8_0 experts without the H128 pair and emit finite, fluent, ~1e-1-wrong
//! hidden state, which finiteness alone would never catch. 1 280 is not wrong,
//! just 3.6x slower — but a silent fallback is exactly how a performance fix
//! rots, so it fails here too rather than being tolerated.
//!
//! The count is per CHUNK because that is the whole point of the batched body:
//! one launch of `escha_h128_in_batched` covers `n_tokens * k` slots. A prompt
//! longer than the prefill chunk ceiling would legitimately show one budget
//! per chunk; this gate keeps the prompt inside one chunk so the expected
//! number is exact.
//!
//! Token ids are arbitrary here on purpose: this gate is about the launch
//! budget and the structural invariants (finite logits, a non-degenerate
//! argmax), not about semantics. Semantic checking is Task 11's — the
//! converter now embeds the tokenizer, chat template and generation_config, so
//! the daemon DOES drive this checkpoint (`scripts/_coherence_runner.py`, and
//! §10.4 of the design doc).
//!
//! COST: **37.5 GB resident**, measured as an amdgpu GTT delta on gfx1151
//! (`scripts/escha-gtt-probe.sh`). 34.2 GB of that is the Q8_0 routed experts
//! and ~3.3 GB is everything else. It was 67.9 GB until the experts were
//! packed one device buffer per (layer, projection): while each of the 20,480
//! per-expert buffers was its own allocation, the HIP allocator's 2 MiB
//! granule rounded the 2.125 MiB gate_up up to 4 MiB and the 1.0625 MiB down
//! up to 2 MiB, spending 30 GB on rounding. Still not free on a 128 GB
//! workstation with other applications running — check headroom first.
//! See design doc §10.3.
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
        "prefill n={}: H128 launches={prefill_launches} (want {want_launches} for one \
         batched chunk; {} would be the per-token fallback), non-finite logits={prefill_bad}/{}, \
         {:?}",
        PROMPT.len(),
        PROMPT.len() * want_launches,
        b.config.vocab_size,
        t.elapsed()
    );
    assert_eq!(
        prefill_bad,
        0,
        "non-finite logits after an {}-token prefill",
        PROMPT.len()
    );
    // BOTH correct routes are accepted, and everything else fails.
    //
    // `HIPFIRE_PREFILL_BATCHED=0` is a supported escape hatch, so a gate that
    // demanded the batched count would fail the model under a configuration it
    // is meant to survive — and a gate that has to be run with one specific
    // env is a gate people stop running. What must never be accepted is ZERO:
    // that is the generic batched MoE body running escha weights without the
    // H128 pair, the finite-fluent-wrong case no logit check would catch.
    let per_token_total = PROMPT.len() * want_launches;
    let route = match prefill_launches as usize {
        n if n == want_launches => "batched escha prefill body",
        n if n == per_token_total => "per-token fallback (HIPFIRE_PREFILL_BATCHED=0?)",
        _ => "UNKNOWN",
    };
    eprintln!("prefill route: {route}");
    assert!(
        prefill_launches as usize == want_launches || prefill_launches as usize == per_token_total,
        "PREFILL issued {prefill_launches} H128 launches, which is neither the batched \
         budget ({want_launches} = 4 x {} layers, once for the whole chunk) nor the \
         per-token one ({per_token_total}). ZERO in particular means the model reached a \
         BATCHED MoE body with NO escha awareness: it omits both Hadamard transforms and \
         emits finite, fluent, ~1e-1-wrong hidden state that no finiteness or argmax check \
         would catch. Check that the escha branch at the top of `run_moe_prefill` still \
         fires before Path 1 / Path 2.",
        b.config.n_layers
    );
    // Under the DEFAULT configuration the batched body is the expected route;
    // a silent fall back to per-token is correct but 3.6x slower, and a
    // performance fix that quietly stops applying is how this regresses.
    if hipfire_runtime::config::get().prefill_batched {
        assert_eq!(
            prefill_launches as usize, want_launches,
            "default config, but prefill took the per-token route ({prefill_launches} \
             launches). Run with HIPFIRE_DEBUG_BATCH=1 to see which layer refused."
        );
    }

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
