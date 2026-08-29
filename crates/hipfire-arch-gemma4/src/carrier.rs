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
use hipfire_runtime::gpu_cleanup::enqueue_cleanup_failure;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{KvCache, KvCacheExt};
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

fn gemma4_validate_drafter_route(
    is_e_series: bool,
    is_moe: bool,
    has_drafter: bool,
) -> Result<(), String> {
    if is_moe && has_drafter {
        return Err(
            "gemma4: lowered/MoE EAGLE spec-decode is not supported; load the MoE target without params.drafter"
                .into(),
        );
    }
    if is_e_series && has_drafter {
        return Err(
            "gemma4: E2B/E4B EAGLE spec-decode is not yet supported; load the E-series target without params.drafter"
                .into(),
        );
    }
    Ok(())
}
#[inline]
fn lowered_sliding_physical_cap(max_seq: usize) -> usize {
    // The lowered sliding attention kernel applies `sliding_window` as its
    // logical mask, while its position addressing remains absolute. Allocate
    // the physical rows to the logical horizon so positions beyond the window
    // cannot index past the cache.
    max_seq
}

// ─── Bundle types ─────────────────────────────────────────────────────────

pub struct Gemma4EagerBundle {
    pub config: Gemma4Config,
    pub weights: Gemma4Weights,
    pub state: Gemma4State,
}

impl Gemma4EagerBundle {
    /// Actual bytes owned by eager target weights and state. Tied aliases are
    /// excluded by the underlying owner accounting helpers.
    pub fn owner_bytes(&self) -> usize {
        self.weights.owner_bytes() + self.state.owner_bytes()
    }
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

/// Requested execution route for Gemma 4 diagnostic callers.
///
/// `Auto` is the production policy used by [`load_gemma4_bundle`]. The
/// explicit variants are intentionally only a loader override for diagnostics;
/// architecture-incompatible requests fail before any GPU allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4Route {
    Auto,
    Eager,
    Lowered,
}

fn select_gemma4_route(
    requested: Gemma4Route,
    auto_use_lowered: bool,
    is_moe: bool,
    is_e_series: bool,
    has_drafter: bool,
) -> Result<bool, String> {
    match requested {
        Gemma4Route::Auto => Ok(auto_use_lowered),
        Gemma4Route::Eager => {
            if is_moe {
                return Err(
                    "gemma4: --route eager is incompatible with MoE; use --route lowered or --route auto"
                        .into(),
                );
            }
            Ok(false)
        }
        Gemma4Route::Lowered => {
            if is_e_series {
                return Err(
                    "gemma4: --route lowered is incompatible with E-series PLE/KV sharing; use --route eager or --route auto"
                        .into(),
                );
            }
            if has_drafter {
                return Err(
                    "gemma4: --route lowered is incompatible with EAGLE; use --route eager or --route auto"
                        .into(),
                );
            }
            Ok(true)
        }
    }
}
impl Gemma4Bundle {
    /// Actual bytes owned by whichever Gemma execution route was selected.
    pub fn owner_bytes(&self) -> usize {
        match self {
            Self::Eager(eager) => eager.owner_bytes(),
            Self::Lowered(lowered) => lowered.owner_bytes(),
        }
    }
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

fn release_lowered_kv(label: &str, kv: KvCache, gpu: &mut rdna_compute::Gpu) {
    let owner_bytes = lowered::kv_owner_bytes(&kv);
    emit_rollback_boundary(
        match label {
            "full" => "rollback_full_kv_before",
            "sliding" => "rollback_sliding_kv_before",
            _ => "rollback_kv_before",
        },
        owner_bytes,
        gpu,
    );
    let remaining_bytes = match kv.free_checked(gpu) {
        Ok(()) => {
            lowered::unregister_live_owner_bytes(owner_bytes);
            0
        }
        Err(remaining) => {
            let failure = lowered::kv_cleanup_failure_from_remaining(remaining);
            let remaining_bytes = lowered::kv_cleanup_failure_bytes(&failure);
            lowered::unregister_live_owner_bytes(owner_bytes.saturating_sub(remaining_bytes));
            enqueue_cleanup_failure(lowered::tracked_kv_cleanup_failure(
                failure,
                remaining_bytes,
            ));
            remaining_bytes
        }
    };
    emit_rollback_boundary(
        match label {
            "full" => "rollback_full_kv_after",
            "sliding" => "rollback_sliding_kv_after",
            _ => "rollback_kv_after",
        },
        remaining_bytes,
        gpu,
    );
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
            release_lowered_kv("full", kv_full, self.gpu);
        }
        if let Some(kv_sliding) = self.kv_sliding.take() {
            release_lowered_kv("sliding", kv_sliding, self.gpu);
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
    gemma4_validate_drafter_route(is_e_series, lowered_is_moe, has_drafter)
}

/// Build the Gemma 4 GPU bundle from an HFQ source using the production
/// architecture policy.
///
/// This is intentionally the default `Auto` route. Diagnostic callers that
/// need to compare the two implementations should use
/// [`load_gemma4_bundle_with_route`] instead.
pub fn load_gemma4_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Gemma4Bundle, String> {
    load_gemma4_bundle_with_route(src, ctx, Gemma4Route::Auto)
}

/// Build the Gemma 4 GPU bundle with an explicit diagnostic route override.
///
/// `Auto` is the same policy used by the production loader. Forced routes are
/// checked against the model topology before loading any GPU weights, so an
/// E-series PLE/KV-sharing model cannot silently enter the lowered path and a
/// MoE model cannot silently lose its expert branch.
pub fn load_gemma4_bundle_with_route(
    src: ModelSource,
    ctx: &mut LoadCtx,
    route: Gemma4Route,
) -> Result<Gemma4Bundle, String> {
    // `ModelSource::Dir` returns the same error string the carrier previously
    // emitted inline. HFQ path is verbatim: lowered/eager selection,
    // `want_batched` env gate, E-series validation, weight/state/KV allocation,
    // and the preserved `eprintln!` diagnostics for the chosen path.
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
    gemma4_validate_drafter_route(
        is_e_series,
        lowered_is_moe,
        ctx.gemma4_drafter_path.is_some(),
    )?;
    let auto_use_lowered = lowered_cfg.as_ref().is_some_and(|lcfg| {
        gemma4_use_lowered(
            lcfg.enable_moe_block,
            want_batched,
            ctx.gemma4_drafter_path.is_some(),
            is_e_series,
        )
    });
    let use_lowered = select_gemma4_route(
        route,
        auto_use_lowered,
        lowered_is_moe,
        is_e_series,
        ctx.gemma4_drafter_path.is_some(),
    )?;
    if use_lowered && lowered_cfg.is_none() {
        return Err(
            "gemma4: --route lowered requested, but the lowered config could not be parsed".into(),
        );
    }
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
            lowered_sliding_physical_cap(ctx.max_seq),
        )
        .map_err(|e| format!("gemma4 (lowered) sliding KV alloc (q8 ring): {e:?}"))?;
        staging.kv_sliding = Some(kv_sliding);
        lowered::register_live_owner_bytes(lowered::kv_owner_bytes(
            staging.kv_sliding.as_ref().expect("sliding KV staged"),
        ));
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::SlidingKv)
            .map_err(|e| format!("gemma4 (lowered) sliding KV stage: {e:?}"))?;

        let kv_full = if ctx.kv_mode_override == Some("fwht3") {
            eprintln!("  gemma4 lowered full KV: FWHT-512 3-bit K + Q8_0 V");
            staging
                .gpu_mut()
                .ensure_mq_signs()
                .map_err(|e| format!("gemma4 (lowered) fwht3 signs: {e:?}"))?;
            let n_full = lcfg
                .layer_types
                .iter()
                .filter(|layer| matches!(layer, lowered::LayerType::Full))
                .count();
            let all_true = vec![true; n_full];
            KvCache::new_gpu_fwht3_capped_filtered_gemma4(
                staging.gpu_mut(),
                &all_true,
                lcfg.full_n_kv_heads,
                lcfg.full_head_dim,
                ctx.max_seq,
                ctx.max_seq,
            )
            .map_err(|e| format!("gemma4 (lowered) full KV (fwht3): {e:?}"))?
        } else {
            KvCache::new_gpu_asym3_gemma4(
                staging.gpu_mut(),
                lcfg.n_layers,
                lcfg.full_n_kv_heads,
                lcfg.full_head_dim,
                ctx.max_seq,
            )
            .map_err(|e| format!("gemma4 (lowered) full KV alloc: {e:?}"))?
        };
        staging.kv_full = Some(kv_full);
        lowered::register_live_owner_bytes(lowered::kv_owner_bytes(
            staging.kv_full.as_ref().expect("full KV staged"),
        ));
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::FullKv)
            .map_err(|e| format!("gemma4 (lowered) full KV stage: {e:?}"))?;

        let full_kv_mode = if ctx.kv_mode_override == Some("fwht3") {
            "fwht3"
        } else {
            "asym3"
        };
        eprintln!(
            "  gemma4 lowered path: moe={} batched_opt_in={} (sliding q8-ring + full {full_kv_mode} KV)",
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
    let state = if ctx.kv_mode_override == Some("fwht3") {
        eprintln!("  gemma4 eager full KV: FWHT-512 3-bit K + Q8_0 V");
        Gemma4State::new_with_fwht3_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("gemma4: Gemma4State::new_with_fwht3_max_seq failed: {e}"))?
    } else {
        Gemma4State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("gemma4: Gemma4State::new_with_max_seq failed: {e}"))?
    };
    Ok(Gemma4Bundle::Eager(Gemma4EagerBundle {
        config,
        weights,
        state,
    }))
}

// Alias for task's naming convention if callers use `load_bundle`.
pub use load_gemma4_bundle as load_bundle;

#[cfg(test)]
mod tests {
    use super::{lowered_sliding_physical_cap, select_gemma4_route, Gemma4Route};

    #[test]
    fn lowered_sliding_kv_uses_logical_context_capacity() {
        assert_eq!(lowered_sliding_physical_cap(2048), 2048);
        assert!(
            lowered_sliding_physical_cap(4096) > 1024,
            "configured contexts beyond the 1024-token window must remain allocatable"
        );
    }

    #[test]
    fn explicit_routes_follow_architecture_capabilities() {
        assert_eq!(
            select_gemma4_route(Gemma4Route::Auto, false, false, true, false).unwrap(),
            false,
            "E-series auto route must stay eager for PLE/KV sharing"
        );
        assert_eq!(
            select_gemma4_route(Gemma4Route::Auto, true, true, false, false).unwrap(),
            true,
            "MoE auto route must retain the lowered expert branch"
        );
        assert_eq!(
            select_gemma4_route(Gemma4Route::Lowered, false, false, true, false)
                .unwrap_err()
                .to_string(),
            "gemma4: --route lowered is incompatible with E-series PLE/KV sharing; use --route eager or --route auto"
        );
        assert_eq!(
            select_gemma4_route(Gemma4Route::Eager, true, true, false, false)
                .unwrap_err()
                .to_string(),
            "gemma4: --route eager is incompatible with MoE; use --route lowered or --route auto"
        );
        assert_eq!(
            select_gemma4_route(Gemma4Route::Lowered, false, false, false, false).unwrap(),
            true,
            "dense diagnostics may explicitly select lowered"
        );
        assert_eq!(
            select_gemma4_route(Gemma4Route::Eager, true, false, false, false).unwrap(),
            false,
            "dense diagnostics may explicitly select eager"
        );
    }

    #[test]
    fn lowered_route_rejects_eagle_drafter_even_for_dense_models() {
        let error =
            select_gemma4_route(Gemma4Route::Lowered, false, false, false, true).unwrap_err();
        assert!(error.contains("--route lowered"));
        assert!(error.contains("EAGLE"));
    }
}
