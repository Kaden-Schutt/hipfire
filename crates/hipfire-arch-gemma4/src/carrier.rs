// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 carrier bundle loader — HFQ path.
//!
//! Verbatim relocation of the model-loading work from
//! `hipfire-loader/src/carriers.rs::Gemma4Carrier::load`. The loader retains
//! `LoadedModel` assembly, `SourceMeta`/`resolve_source_meta`, chat-template
//! and tokenizer handling, `Gemma4EagleState` side-car load, and
//! `spec_build::build_speculator`. This module owns the GPU bundle construction
//! (lowered vs eager decision, weight/state/KV allocation) with error strings
//! byte-identical to the prior inline block.

use crate::config::Gemma4Config;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use crate::lowered;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

// ─── Helpers moved verbatim from carriers.rs ─────────────────────────────

fn gemma4_use_lowered(
    enable_moe_block: bool,
    want_batched: bool,
    has_drafter: bool,
    is_e_series: bool,
) -> bool {
    enable_moe_block || (want_batched && !has_drafter && !is_e_series)
}

fn gemma4_validate_drafter_route(is_e_series: bool, has_drafter: bool) -> Result<(), String> {
    if is_e_series && has_drafter {
        return Err(
            "gemma4: E2B/E4B EAGLE spec-decode is not yet supported; load the E-series target without params.drafter"
                .into(),
        );
    }
    Ok(())
}

// ─── Bundle types ─────────────────────────────────────────────────────────

pub struct Gemma4EagerBundle {
    pub config: Gemma4Config,
    pub weights: Gemma4Weights,
    pub state: Gemma4State,
}

pub struct Gemma4LoweredBundle {
    pub config: lowered::Gemma4Config,
    pub weights: lowered::Gemma4Weights,
    pub scratch: lowered::Gemma4Scratch,
    pub kv_sliding: KvCache,
    pub kv_full: KvCache,
}

pub enum Gemma4Bundle {
    Eager(Gemma4EagerBundle),
    Lowered(Gemma4LoweredBundle),
}
impl Gemma4LoweredBundle {
    /// Actual bytes owned by the lowered bundle, excluding borrowed aliases
    /// (tied LM head and pool-backed expert views).
    pub fn owner_bytes(&self) -> usize {
        self.weights.owner_bytes()
            + self.scratch.owner_bytes()
            + lowered::kv_owner_bytes(&self.kv_sliding)
            + lowered::kv_owner_bytes(&self.kv_full)
    }
}

/// Lowered bundle construction owner. It borrows the destination GPU for the
/// entire load and keeps every completed resource private until publication.
/// Destruction is deliberately reverse-ordered: full KV, sliding KV, scratch,
/// then weights.
struct Gemma4LoweredStaging<'a> {
    gpu: &'a mut rdna_compute::Gpu,
    weights: Option<lowered::Gemma4Weights>,
    scratch: Option<lowered::Gemma4Scratch>,
    kv_sliding: Option<KvCache>,
    kv_full: Option<KvCache>,
}

fn emit_rollback_boundary(phase: &'static str, owner_bytes: usize, gpu: &rdna_compute::Gpu) {
    if lowered::allocation_telemetry_enabled() {
        lowered::Gemma4AllocationTelemetry::emit_from_gpu(
            phase,
            lowered::allocation_telemetry_cycle(),
            owner_bytes,
            gpu,
            Vec::new(),
        );
    }
}

impl<'a> Gemma4LoweredStaging<'a> {
    fn new(gpu: &'a mut rdna_compute::Gpu) -> Self {
        Self {
            gpu,
            weights: None,
            scratch: None,
            kv_sliding: None,
            kv_full: None,
        }
    }

    fn gpu_mut(&mut self) -> &mut rdna_compute::Gpu {
        self.gpu
    }

    fn publish(mut self, config: lowered::Gemma4Config) -> Gemma4LoweredBundle {
        Gemma4LoweredBundle {
            config,
            weights: self.weights.take().expect("lowered weights not staged"),
            scratch: self.scratch.take().expect("lowered scratch not staged"),
            kv_sliding: self
                .kv_sliding
                .take()
                .expect("lowered sliding KV not staged"),
            kv_full: self.kv_full.take().expect("lowered full KV not staged"),
        }
    }

    fn release(&mut self) {
        if let Some(kv_full) = self.kv_full.take() {
            emit_rollback_boundary(
                "rollback_full_kv_before",
                lowered::kv_owner_bytes(&kv_full),
                self.gpu,
            );
            let _ = kv_full.free_gpu(self.gpu);
            emit_rollback_boundary("rollback_full_kv_after", 0, self.gpu);
        }
        if let Some(kv_sliding) = self.kv_sliding.take() {
            emit_rollback_boundary(
                "rollback_sliding_kv_before",
                lowered::kv_owner_bytes(&kv_sliding),
                self.gpu,
            );
            let _ = kv_sliding.free_gpu(self.gpu);
            emit_rollback_boundary("rollback_sliding_kv_after", 0, self.gpu);
        }
        if let Some(scratch) = self.scratch.take() {
            emit_rollback_boundary("rollback_scratch_before", scratch.owner_bytes(), self.gpu);
            scratch.free_gpu(self.gpu);
            emit_rollback_boundary("rollback_scratch_after", 0, self.gpu);
        }
        if let Some(weights) = self.weights.take() {
            emit_rollback_boundary("rollback_weights_before", weights.owner_bytes(), self.gpu);
            weights.free_gpu(self.gpu);
            emit_rollback_boundary("rollback_weights_after", 0, self.gpu);
        }
    }
}

impl Drop for Gemma4LoweredStaging<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

/// Gemma 4 source/option preconditions — the single authority shared by the
/// bundle loader and the daemon-side preflight: HFQ-only source shape,
/// E-series variant validity, and the E-series × EAGLE-drafter refusal.
pub fn preflight_gemma4(hfq: &HfqFile, has_drafter: bool) -> Result<(), String> {
    let lowered_cfg = lowered::config_from_hfq(hfq);
    let lowered_is_moe = lowered_cfg
        .as_ref()
        .is_some_and(|lcfg| lcfg.enable_moe_block);
    let eager_config = if lowered_is_moe {
        None
    } else {
        Some(Gemma4Config::from_hfq(hfq)?)
    };
    let is_e_series = eager_config
        .as_ref()
        .is_some_and(|cfg| cfg.hidden_size_per_layer_input != 0 || cfg.num_kv_shared_layers != 0);
    if is_e_series {
        eager_config.as_ref().unwrap().e_series_variant()?;
    }
    gemma4_validate_drafter_route(is_e_series, has_drafter)
}

/// Build the Gemma 4 GPU bundle from an HFQ source.
///
/// `ModelSource::Dir` returns the same error string the carrier previously
/// emitted inline. HFQ path is verbatim: lowered/eager selection,
/// `want_batched` env gate, E-series validation, weight/state/KV allocation,
/// and the preserved `eprintln!` diagnostics for the chosen path.
pub fn load_gemma4_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Gemma4Bundle, String> {
    let hfq = match src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(_) => {
            return Err("gemma4: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 13) or add config_from_source to hipfire-arch-gemma4".into());
        }
    };
    preflight_gemma4(&hfq, ctx.gemma4_drafter_path.is_some())?;

    // ── Lowered vs eager selection (MoE or batched prefill opt-in) ──
    // Arch-13 MoE (26B-A4B `enable_moe_block`) must go through `lowered`, which
    // carries the parallel-MoE branch. We also route DENSE models through
    // `lowered` when the operator opts into batched/WMMA prefill — that path
    // lives only in `lowered::forward_prefill_batch`. E2B/E4B stay on eager
    // because lowered does not implement PLE, KV sharing, or E2B's double-wide
    // shared-layer FFN. EAGLE spec-decode (`params.drafter`) requires the eager
    // `Gemma4State`, so a drafter request always wins and keeps the eager path
    // (batched prefill opt-in is ignored when a drafter is present).
    let lowered_cfg = lowered::config_from_hfq(&hfq);
    let want_batched = lowered::batched_prefill_enabled() || lowered::wmma_prefill_enabled();
    let lowered_is_moe = lowered_cfg
        .as_ref()
        .is_some_and(|lcfg| lcfg.enable_moe_block);
    let eager_config = if lowered_is_moe {
        None
    } else {
        Some(Gemma4Config::from_hfq(&hfq)?)
    };
    let is_e_series = eager_config
        .as_ref()
        .is_some_and(|cfg| cfg.hidden_size_per_layer_input != 0 || cfg.num_kv_shared_layers != 0);
    if is_e_series {
        eager_config.as_ref().unwrap().e_series_variant()?;
    }
    gemma4_validate_drafter_route(is_e_series, ctx.gemma4_drafter_path.is_some())?;
    let use_lowered = if let Some(ref lcfg) = lowered_cfg {
        gemma4_use_lowered(
            lcfg.enable_moe_block,
            want_batched,
            ctx.gemma4_drafter_path.is_some(),
            is_e_series,
        )
    } else {
        false
    };
    if use_lowered {
        let lcfg = lowered_cfg.unwrap();
        let mut hfq2 = hfq;
        let mut staging = Gemma4LoweredStaging::new(ctx.gpu);

        let weights = lowered::load_weights(&mut hfq2, &lcfg, staging.gpu_mut())
            .map_err(|e| format!("gemma4 (lowered) load_weights: {e:?}"))?;
        staging.weights = Some(weights);
        // All model tensor reads are complete. Release the HFQ mapping before
        // rollback can return the weights to the pool; on UMA this mapping's
        // resident pages share the same physical budget as hipMalloc owners.
        hfq2.drop_mmap();
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::Weights)
            .map_err(|e| format!("gemma4 (lowered) weights stage: {e:?}"))?;

        let scratch = lowered::Gemma4Scratch::new(staging.gpu_mut(), &lcfg, 1)
            .map_err(|e| format!("gemma4 (lowered) scratch: {e:?}"))?;
        staging.scratch = Some(scratch);
        {
            let gpu = &mut *staging.gpu;
            let scratch = staging.scratch.as_ref().expect("lowered scratch staged");
            lowered::init_scratch_constants(gpu, scratch, lcfg.full_head_dim)
                .map_err(|e| format!("gemma4 (lowered) init_scratch_constants: {e:?}"))?;
        }
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::Scratch)
            .map_err(|e| format!("gemma4 (lowered) scratch stage: {e:?}"))?;

        let kv_sliding = KvCache::new_gpu_q8_capped(
            staging.gpu_mut(),
            lcfg.n_layers,
            lcfg.sliding_n_kv_heads,
            lcfg.sliding_head_dim,
            ctx.max_seq,
            lcfg.sliding_window,
        )
        .map_err(|e| format!("gemma4 (lowered) sliding KV alloc (q8 ring): {e:?}"))?;
        staging.kv_sliding = Some(kv_sliding);
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::SlidingKv)
            .map_err(|e| format!("gemma4 (lowered) sliding KV stage: {e:?}"))?;

        let kv_full = KvCache::new_gpu_asym3_gemma4(
            staging.gpu_mut(),
            lcfg.n_layers,
            lcfg.full_n_kv_heads,
            lcfg.full_head_dim,
            ctx.max_seq,
        )
        .map_err(|e| format!("gemma4 (lowered) full KV alloc: {e:?}"))?;
        staging.kv_full = Some(kv_full);
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::FullKv)
            .map_err(|e| format!("gemma4 (lowered) full KV stage: {e:?}"))?;

        eprintln!(
            "  gemma4 lowered path: moe={} batched_opt_in={} (sliding q8-ring + full asym3 KV)",
            lcfg.enable_moe_block, want_batched,
        );
        return Ok(Gemma4Bundle::Lowered(staging.publish(lcfg)));
    }
    // ── Eager dense / E-series path ──
    let config = match eager_config {
        Some(c) => c,
        None => Gemma4Config::from_hfq(&hfq)?,
    };
    if is_e_series {
        eprintln!(
            "  gemma4 E-series eager path: {:?} (PLE + shared KV)",
            config.e_series_variant()?
        );
    }
    let weights = Gemma4Weights::load(&hfq, &config, ctx.gpu)?;
    let state = Gemma4State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
        .map_err(|e| format!("gemma4: Gemma4State::new_with_max_seq failed: {e}"))?;
    let _ = &weights;
    Ok(Gemma4Bundle::Eager(Gemma4EagerBundle {
        config,
        weights,
        state,
    }))
}

// Alias for task's naming convention if callers use `load_bundle`.
pub use load_gemma4_bundle as load_bundle;
