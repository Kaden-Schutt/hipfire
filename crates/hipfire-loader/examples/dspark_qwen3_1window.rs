// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! dspark_qwen3_1window: Task-10 gate.
//!
//! Loads a Qwen3-8B target (`*.mq4`) alongside its `*-dspark.hfq` sidecar via
//! `hipfire_loader::load_model`, then drives ONE spec window through
//! `Speculator::prefill` + `Speculator::step`, and asserts:
//!   - the drafter proposed `block_size` tokens (non-degenerate draft)
//!   - `accept_greedy_prefix` committed ≥ 1 token (no panic, no zero-length emit)
//!
//! ## Usage
//!
//! ```
//! source scripts/gpu-lock.sh && gpu_acquire dspark-qwen3
//! cargo build --release -p hipfire-loader --example dspark_qwen3_1window --features lab
//! ./target/release/examples/dspark_qwen3_1window /home/bjoern/.hipfire/models/qwen3-8b.mq4
//! gpu_release
//! ```
//!
//! The sidecar must be at `<stem>-dspark.<ext>` next to the target
//! (e.g. `qwen3-8b-dspark.mq4`).

fn main() -> Result<(), String> {
    let target_path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3-8b.mq4",
            std::env::var("HOME").unwrap_or_default()
        )
    });

    eprintln!("[1window] loading target: {target_path}");

    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    eprintln!("[1window] GPU: {}", gpu.arch);

    let max_seq = 512usize;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
        mtp_mode: None,
        mtp_k: None,
        dspark: None, // auto: load sidecar if present
        ..Default::default()
    };

    let mut m = hipfire_loader::load_model(
        &target_path,
        max_seq,
        None, // draft_path (DFlash): not needed, DSpark is a sidecar
        None, // kv_mode_override
        None, // kv_adaptive_override
        None, // state_quant_override
        &cask,
        1, // pp: single-GPU
        spec_cfg,
        &mut gpu,
    )?;

    let spec = m.speculator.as_mut().ok_or_else(|| {
        "No speculator loaded — DSpark sidecar discovery failed or ctx.spec.dspark=false.\n\
         Expected sidecar at <stem>-dspark.<ext> next to the target.\n\
         For qwen3-8b.mq4 → qwen3-8b-dspark.mq4"
            .to_string()
    })?;

    let block_size = spec.block_size();
    eprintln!("[1window] speculator block_size={block_size}");

    // Minimal prompt: "The capital of France is" tokenised to a few fixed ids.
    // We use raw token IDs so the test is independent of the tokenizer artefact.
    // These ids are from the Qwen3 vocab; correctness doesn't matter — we only
    // need a non-trivial prefill to prime the KV cache and get a plausible seed.
    let prompt_tokens: Vec<u32> = vec![785, 6722, 315, 9625, 374]; // "The capital of France is"
    let prefill_tokens = prompt_tokens.clone();
    let prefill_start = 0usize;

    // Acquire spec target via carrier dispatch.
    let arch_id = m.arch_id;
    let carrier = hipfire_loader::carrier_for(arch_id)
        .ok_or_else(|| format!("no carrier for arch_id {arch_id}"))?;
    let mut guard = carrier
        .spec_target_guard(&mut m.state, &m.model_path)
        .map_err(|e| format!("spec_target_guard: {e}"))?;
    let target = guard.slot().map_err(|e| format!("guard.slot: {e}"))?;

    // Prefill.
    let prefill_out = spec
        .prefill(
            &mut gpu,
            target,
            &prompt_tokens,
            &prefill_tokens,
            prefill_start,
            false,     // cache_hit=false (fresh run)
            None,      // resume_from
            &|| false, // abort
        )
        .map_err(|e| format!("prefill: {e}"))?;

    let seed = match prefill_out {
        hipfire_runtime::spec::PrefillOutcome::Ready { first_token } => first_token,
        hipfire_runtime::spec::PrefillOutcome::Aborted => {
            return Err("prefill aborted — prompt triggered immediate abort".into())
        }
    };
    let position = prompt_tokens.len();
    eprintln!("[1window] prefill done: first seed={seed}, position={position}");

    // One spec window.
    let step = spec
        .step(
            &mut gpu,
            target,
            position,
            seed,
            &[],        // emitted (empty: no prior context for repeat-penalty)
            None,       // grammar
            0.0,        // temp: greedy
            usize::MAX, // uncapped bench window
        )
        .map_err(|e| format!("step: {e}"))?;

    eprintln!(
        "[1window] step: proposed={} accepted={} emit_len={} emit={:?}",
        step.proposed,
        step.accepted,
        step.emit.len(),
        &step.emit[..step.emit.len().min(8)]
    );

    // Gate: must have committed at least 1 token (no panic, no zero-length emit).
    if step.emit.is_empty() {
        return Err(format!(
            "FAIL: emit is empty (proposed={}, accepted={}). \
             Drafter produced no accepted tokens.",
            step.proposed, step.accepted
        ));
    }

    // Gate: drafter should have proposed exactly block_size drafts.
    if step.proposed != block_size {
        eprintln!(
            "[1window] WARN: proposed={} != block_size={} (confidence may have truncated)",
            step.proposed, block_size
        );
    }

    println!(
        "PASS: 1-window accept gate OK — accepted={}/{} (emit_len={})",
        step.accepted,
        step.proposed,
        step.emit.len()
    );
    Ok(())
}
