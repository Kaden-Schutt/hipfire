// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-qwen35: Qwen3.5 architecture (dense + MoE / A3B / A10B / A17B).
//!
//! This crate implements the [`hipfire_runtime::arch::Architecture`] trait
//! for Qwen3.5. It owns the model forward pass, weight loading, KV-state
//! layout, and the speculative-decoding glue that today is qwen35-specific
//! (`speculative.rs`; `pflash.rs` evacuated to `hipfire-pflash` per lean-up
//! map B3 — retained legacy research, not mainline).
//!
//! Future work (per docs/plans/engine-modularization.prd Phase 2):
//!   - `speculative.rs` will become arch-generic and move back into
//!     `hipfire-runtime`. It lives here today because the existing impl is
//!     deeply coupled to `qwen35::*` symbols (config, weights, scratch,
//!     forward functions). `pflash.rs` was evacuated to `hipfire-pflash`
//!     per lean-up map B3 (§5.1) — retained legacy research, historical
//!     reproduction only. PR 8 freezes the dep direction `arch-qwen35 →
//!     runtime`, but accepts that today's spec is not generic enough to
//!     live above the arch boundary.
//!
//! The `arch` module exposes the trait impl for use by the runtime's
//! daemon and other consumers via `hipfire_arch_qwen35::Qwen35`.

// Qwen3.5 is a hybrid DeltaNet + FullAttention architecture; all the
// runtime infrastructure it touches is `deltanet`-gated. When the parent
// build doesn't enable the feature, the crate is a no-op stub. This keeps
// `cargo build --no-default-features` working and matches the gating that
// was on `engine::qwen35` pre-Phase-2.
#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod arch_model;
#[cfg(feature = "deltanet")]
pub mod carrier;
/// Qwen3.5 DFlash / DDTree speculative-decode state (`DflashState`,
/// `load_dflash_state`) and the `DflashSpeculator` impl of the arch-generic
/// `hipfire_runtime::spec::Speculator`. Deltanet-gated — it owns `ModelSlot`-
/// based draft verify.
#[cfg(feature = "deltanet")]
pub mod dflash_spec;
/// Retained-PM4 route state for the fixed B=16 DFlash2 target-verify forward
/// (`DflashVerifyPm4`). Owns the phase machine, admission binding, and
/// route-proof counters; `speculative` owns the GPU half.
/// Not deltanet-gated, matching `speculative`, which consumes it.
pub mod dflash_verify_pm4;
/// SP3 Task 2 — `forward_batch_slots`, the N-slot forward pass. A PARALLEL
/// entry point to `qwen35::forward_prefill_batch_with_pbs_opts` (never a
/// modification of it — see the module doc for why), routing attention
/// and KV-write through SP1's slot-aware `_slots` kernels and DeltaNet
/// through SP2's per-slot `DeltaNetState`. Q8_0-only; depends on `qwen35`
/// for weight/config/scratch types, hence deltanet-gated like it.
#[cfg(feature = "deltanet")]
pub mod forward_slots;
#[cfg(feature = "deltanet")]
pub(crate) mod layer_driver;
#[cfg(feature = "deltanet")]
pub mod mtp_compose;
#[cfg(feature = "deltanet")]
pub mod mtp_head;
#[cfg(feature = "deltanet")]
pub mod mtp_probe;
#[cfg(feature = "deltanet")]
pub mod mtp_spec;
/// Qwen3.5 `MtpDrafter` impl (the arch half of the unified MTP spec-decode
/// core). Deltanet-gated — it touches `ModelSlot` + `MtpSpecState`.
#[cfg(feature = "deltanet")]
pub mod mtp_speculator;
#[cfg(feature = "deltanet")]
pub(crate) mod paro_moe;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
#[cfg(feature = "deltanet")]
pub mod serve_engine;
/// Qwen3.5 impls of the arch-generic `hipfire_runtime::spec` seam
/// (`impl SpecTarget for ModelSlot`). Deltanet-gated — it touches `ModelSlot`.
#[cfg(feature = "deltanet")]
mod spec_impl;
pub mod speculative;

/// Grammar-guided decoding for tool-call format — re-exported from `saddle_core::grammar::json`.
///
/// Unified in `saddle-core` (lean-up map B1). This re-export preserves the
/// `hipfire_arch_qwen35::grammar` path for existing consumers while the
/// implementation lives in `saddle_core::grammar::json`.
pub use saddle_core::grammar::json as grammar;

/// qwen35 grammar `Config` resolver that restores the `HIPFIRE_QWEN35_*` env
/// overrides lost in the B1 unification. Reads the two tunables via
/// `hipfire_config::developer_var` with the same parse/bounds as the
/// pre-merge `grammar.rs:128-153` and falls back to `Config::default()`.
/// Exposed here so the daemon example (outside this crate) can use the same
/// single source of truth.
pub mod grammar_config;
pub use grammar_config::{resolve_grammar_config, resolve_qwen35_grammar_config};

/// Per-token spec-decode emission (`SpecEmit`). Pure CPU; named here because it
/// drives the qwen35 `grammar` matcher. Built via [`spec_emit::Qwen35Emit::from_ctx`].
pub mod spec_emit;

/// `SlotBatch` — one forward step's ragged work across N slots. Pure CPU
/// data structure; no GPU dependencies. See module docs for the
/// per-slot-absolute `positions[]` invariant.
pub mod slot_batch;

/// `Scheduler` — decides what goes into each step's `SlotBatch`. Pure CPU
/// logic; no GPU dependencies. Round-robin, chunked prefill mixed with
/// decode; deliberately minimal — see module docs for why.
pub mod scheduler;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35;

#[cfg(feature = "deltanet")]
pub use carrier::{free_qwen35_bundle, load_bundle as load_qwen35_bundle, Qwen35Bundle};
#[cfg(feature = "deltanet")]
pub use mtp_compose::{spec_step_dflash_mtp_tree, MtpComposeTreeResult, MtpComposeTreeState};
#[cfg(feature = "deltanet")]
pub use mtp_speculator::{build_qwen35_mtp_speculator, Qwen35MtpDrafter};

/// G4b (Escha-W2 port, Task 9): expose the arch-6 MoE router selection step
/// so `examples/escha_router_contract.rs` can call the *actual* production
/// router — not a reimplementation of it — on an arbitrary `.hfq` layer and
/// compare its top-K expert SET against EschaLabs' shipped fixture.
///
/// This calls exactly the same primitives the production decode/prefill path
/// calls (`hipfire_runtime::llama::weight_gemv` for the router GEMV, then
/// whichever top-K kernel `hipfire_dispatch::pipeline::run_moe_decode` would
/// pick for this GPU's arch — the fused `moe_router_softmax_topk_k8_wave64_exact`
/// on gfx1100/gfx1151, or the reference two-launch `softmax_f32` +
/// `moe_topk_renorm_k8` everywhere else). No selection math is duplicated
/// here; only the same arch check the pipeline itself uses to pick a kernel.
#[cfg(feature = "deltanet")]
pub fn escha_router_topk_for_test(
    hfq_path: &str,
    layer: usize,
    x: &[f32],
    n_tokens: usize,
    hidden: usize,
    top_k: usize,
) -> Result<Vec<u32>, String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_runtime::hfq::{load_weight_tensor_pread, HfqFile};
    use hipfire_runtime::llama::weight_gemv;
    use rdna_compute::{DType, Gpu};

    if x.len() != n_tokens * hidden {
        return Err(format!(
            "escha_router_topk_for_test: x.len()={} != n_tokens*hidden={}",
            x.len(),
            n_tokens * hidden
        ));
    }

    let hfq = HfqFile::open(std::path::Path::new(hfq_path)).map_err(|e| e.to_string())?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let n_exp = config.num_experts;
    let norm_topk = config.norm_topk_prob;

    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let ctx = DispatchCtx::new(&gpu);
    // Same arch predicate `run_moe_decode` (hipfire-dispatch/src/pipeline/mod.rs)
    // uses to pick the fused exact-wave64 router kernel over the reference
    // two-launch path. HIPFIRE_MOE_ROUTER_SHARED_FUSE-gated shared-expert
    // fusion is a pure perf variant of the same math and is left out here —
    // it doesn't change which experts get selected.
    let use_exact_wave64 = ctx.arch.is_gfx1151() || ctx.arch.is_gfx1100();

    fn exact_name(s: &str) -> Vec<String> {
        vec![s.to_string()]
    }
    let weight_name = format!("model.language_model.layers.{layer}.mlp.gate.weight");
    let router = load_weight_tensor_pread(&hfq, &gpu, &weight_name, n_exp, hidden, exact_name)
        .map_err(|e| e.to_string())?;

    let logits = gpu
        .alloc_tensor(&[n_exp], DType::F32)
        .map_err(|e| e.to_string())?;
    // topk_idx carries raw i32 selections in an f32-tagged buffer — the same
    // "i32-in-F32 alias" convention `moe_ffn_decode_impl` uses for its scratch
    // (see qwen35/forward.rs `capture_expert_stats`'s `ti[krank].to_bits()`).
    let topk_idx = gpu
        .alloc_tensor(&[top_k], DType::F32)
        .map_err(|e| e.to_string())?;
    let topk_w = gpu
        .alloc_tensor(&[top_k], DType::F32)
        .map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        let row = &x[t * hidden..(t + 1) * hidden];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
        let x_gpu = gpu
            .upload_raw(bytes, &[hidden])
            .map_err(|e| e.to_string())?;

        weight_gemv(&mut gpu, &router, &x_gpu, &logits).map_err(|e| e.to_string())?;

        if use_exact_wave64 {
            gpu.moe_router_softmax_topk_k8_wave64_exact(
                &logits, &topk_idx, &topk_w, n_exp, norm_topk,
            )
            .map_err(|e| e.to_string())?;
        } else {
            gpu.softmax_f32(&logits).map_err(|e| e.to_string())?;
            gpu.moe_topk_renorm_k8(&logits, &topk_idx, &topk_w, n_exp, norm_topk)
                .map_err(|e| e.to_string())?;
        }

        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        let idx_f32 = gpu.download_f32(&topk_idx).map_err(|e| e.to_string())?;
        for v in idx_f32.iter().take(top_k) {
            out.push((v.to_bits() as i32) as u32);
        }

        let _ = gpu.free_tensor(x_gpu);
    }

    let _ = gpu.free_tensor(logits);
    let _ = gpu.free_tensor(topk_idx);
    let _ = gpu.free_tensor(topk_w);
    router.free_all(&mut gpu);

    Ok(out)
}
