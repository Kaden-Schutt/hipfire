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
//! (lowered vs eager decision, weight/state/KV allocation), including the
//! topology-specific lowered KV quality policy and fail-closed budget gate.

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
/// Lowered Gemma KV policy selected from the model topology.
///
/// The MoE path is intentionally pinned to F32 for both attention families:
/// compressed KV was the root cause of the canonical 26B-A4B divergence. Dense
/// lowered remains on its established Q8 sliding + Asym3 full policy; the
/// explicit `fwht3` override is still available for dense diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4LoweredKvPolicy {
    MoeF32,
    DenseCompressed,
}

impl Gemma4LoweredKvPolicy {
    pub const fn label(self) -> &'static str {
        match self {
            Self::MoeF32 => "moe-f32",
            Self::DenseCompressed => "dense-compressed",
        }
    }
}

#[inline]
pub fn lowered_kv_policy(enable_moe_block: bool) -> Gemma4LoweredKvPolicy {
    if enable_moe_block {
        Gemma4LoweredKvPolicy::MoeF32
    } else {
        Gemma4LoweredKvPolicy::DenseCompressed
    }
}

fn checked_product(parts: &[usize], label: &str) -> Result<usize, String> {
    parts.iter().try_fold(1usize, |value, part| {
        value
            .checked_mul(*part)
            .ok_or_else(|| format!("gemma4 lowered {label} byte calculation overflowed"))
    })
}

fn rounded_tensor_bytes(raw_bytes: usize, label: &str) -> Result<usize, String> {
    raw_bytes
        .checked_add(3)
        .and_then(|bytes| bytes.checked_div(4))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| format!("gemma4 lowered {label} byte calculation overflowed"))
}

fn f32_kv_bytes(
    layers: usize,
    max_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<usize, String> {
    checked_product(&[layers, max_seq, n_kv_heads, head_dim, 2, 4], "F32 KV")
}

fn compressed_kv_bytes(
    layers: usize,
    max_seq: usize,
    n_kv_heads: usize,
    k_bytes_per_head: usize,
    v_bytes_per_head: usize,
) -> Result<usize, String> {
    let k_raw = checked_product(&[max_seq, n_kv_heads, k_bytes_per_head], "compressed K")?;
    let v_raw = checked_product(&[max_seq, n_kv_heads, v_bytes_per_head], "compressed V")?;
    let k_bytes = rounded_tensor_bytes(k_raw, "compressed K")?;
    let v_bytes = rounded_tensor_bytes(v_raw, "compressed V")?;
    let per_layer = k_bytes
        .checked_add(v_bytes)
        .ok_or_else(|| "gemma4 lowered compressed KV byte calculation overflowed".to_string())?;
    checked_product(&[layers, per_layer], "compressed KV")
}

/// Return the exact K/V owner bytes required by a lowered Gemma allocation.
///
/// Counts only layers that actually carry each attention family and both K/V
/// tensors per layer. `max_seq` is both the logical and physical capacity on
/// this route. The dense compressed estimate also includes the two shared
/// rotation tables owned by the full cache.
pub fn required_kv_bytes(
    config: &lowered::Gemma4Config,
    max_seq: usize,
    policy: Gemma4LoweredKvPolicy,
) -> Result<usize, String> {
    let sliding_layers = config
        .layer_types
        .iter()
        .filter(|layer| matches!(layer, lowered::LayerType::Sliding))
        .count();
    let full_layers = config
        .layer_types
        .iter()
        .filter(|layer| matches!(layer, lowered::LayerType::Full))
        .count();
    match policy {
        Gemma4LoweredKvPolicy::MoeF32 => {
            let sliding = f32_kv_bytes(
                sliding_layers,
                max_seq,
                config.sliding_n_kv_heads,
                config.sliding_head_dim,
            )?;
            let full = f32_kv_bytes(
                full_layers,
                max_seq,
                config.full_n_kv_heads,
                config.full_head_dim,
            )?;
            sliding
                .checked_add(full)
                .ok_or_else(|| "gemma4 lowered F32 KV byte calculation overflowed".to_string())
        }
        Gemma4LoweredKvPolicy::DenseCompressed => {
            let sliding_bytes_per_head =
                checked_product(&[config.sliding_head_dim / 32, 34], "Q8 sliding")?;
            let sliding = compressed_kv_bytes(
                sliding_layers,
                max_seq,
                config.sliding_n_kv_heads,
                sliding_bytes_per_head,
                sliding_bytes_per_head,
            )?;
            let full_k_bytes_per_head =
                checked_product(&[3, config.full_head_dim], "Asym3 full K")?
                    .checked_div(8)
                    .and_then(|bytes| bytes.checked_add(4))
                    .ok_or_else(|| {
                        "gemma4 lowered Asym3 full K byte calculation overflowed".to_string()
                    })?;
            let full_v_bytes_per_head =
                checked_product(&[config.full_head_dim / 32, 34], "Asym3 full V")?;
            let full_tensors = compressed_kv_bytes(
                full_layers,
                max_seq,
                config.full_n_kv_heads,
                full_k_bytes_per_head,
                full_v_bytes_per_head,
            )?;
            let full_tables = checked_product(
                &[2, config.full_head_dim / 2, 4],
                "compressed rotation tables",
            )?;
            let full = full_tensors.checked_add(full_tables).ok_or_else(|| {
                "gemma4 lowered compressed KV byte calculation overflowed".to_string()
            })?;
            sliding.checked_add(full).ok_or_else(|| {
                "gemma4 lowered compressed KV byte calculation overflowed".to_string()
            })
        }
    }
}

/// Reject a lowered KV allocation before it can partially construct a cache.
///
/// This is deliberately fail-closed for the MoE F32 policy: an OOM must not
/// silently retry with compressed KV, which would reintroduce the quality bug.
pub fn preflight_lowered_kv_budget(
    policy: Gemma4LoweredKvPolicy,
    required: usize,
    free: usize,
    max_seq: usize,
) -> Result<(), String> {
    if required > free {
        return Err(format!(
            "gemma4 lowered KV allocation refused: policy={} required={} free={} max_seq={}",
            policy.label(),
            required,
            free,
            max_seq,
        ));
    }
    Ok(())
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
        let kv_policy = lowered_kv_policy(lcfg.enable_moe_block);
        let required_kv = required_kv_bytes(&lcfg, ctx.max_seq, kv_policy)
            .map_err(|e| format!("gemma4 (lowered) KV budget calculation: {e}"))?;
        let (free_kv, _total_kv) = staging
            .gpu_mut()
            .hip
            .get_vram_info()
            .map_err(|e| format!("gemma4 (lowered) KV budget query: {e:?}"))?;
        preflight_lowered_kv_budget(kv_policy, required_kv, free_kv, ctx.max_seq)?;

        let n_sliding = lcfg
            .layer_types
            .iter()
            .filter(|layer| matches!(layer, lowered::LayerType::Sliding))
            .count();
        let n_full = lcfg
            .layer_types
            .iter()
            .filter(|layer| matches!(layer, lowered::LayerType::Full))
            .count();
        if matches!(kv_policy, Gemma4LoweredKvPolicy::MoeF32) {
            if let Some(mode) = ctx.kv_mode_override {
                eprintln!(
                    "  gemma4 lowered KV override {mode:?} ignored for MoE; \
                     policy=moe-f32 is required for canonical parity"
                );
            }
        }
        let sliding_physical_cap = lowered_sliding_physical_cap(ctx.max_seq);
        let kv_sliding = match kv_policy {
            Gemma4LoweredKvPolicy::MoeF32 => KvCache::new_gpu(
                staging.gpu_mut(),
                n_sliding,
                lcfg.sliding_n_kv_heads,
                lcfg.sliding_head_dim,
                ctx.max_seq,
            )
            .map_err(|e| format!("gemma4 (lowered) sliding KV alloc (f32): {e:?}"))?,
            Gemma4LoweredKvPolicy::DenseCompressed => KvCache::new_gpu_q8_capped(
                staging.gpu_mut(),
                n_sliding,
                lcfg.sliding_n_kv_heads,
                lcfg.sliding_head_dim,
                ctx.max_seq,
                sliding_physical_cap,
            )
            .map_err(|e| format!("gemma4 (lowered) sliding KV alloc (q8 ring): {e:?}"))?,
        };
        staging.kv_sliding = Some(kv_sliding);
        lowered::register_live_owner_bytes(lowered::kv_owner_bytes(
            staging.kv_sliding.as_ref().expect("sliding KV staged"),
        ));
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::SlidingKv)
            .map_err(|e| format!("gemma4 (lowered) sliding KV stage: {e:?}"))?;

        let kv_full = match kv_policy {
            Gemma4LoweredKvPolicy::MoeF32 => KvCache::new_gpu(
                staging.gpu_mut(),
                n_full,
                lcfg.full_n_kv_heads,
                lcfg.full_head_dim,
                ctx.max_seq,
            )
            .map_err(|e| format!("gemma4 (lowered) full KV alloc (f32): {e:?}"))?,
            Gemma4LoweredKvPolicy::DenseCompressed => {
                if ctx.kv_mode_override == Some("fwht3") {
                    eprintln!("  gemma4 lowered full KV: FWHT-512 3-bit K + Q8_0 V");
                    staging
                        .gpu_mut()
                        .ensure_mq_signs()
                        .map_err(|e| format!("gemma4 (lowered) fwht3 signs: {e:?}"))?;
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
                        n_full,
                        lcfg.full_n_kv_heads,
                        lcfg.full_head_dim,
                        ctx.max_seq,
                    )
                    .map_err(|e| format!("gemma4 (lowered) full KV alloc: {e:?}"))?
                }
            }
        };
        staging.kv_full = Some(kv_full);
        lowered::register_live_owner_bytes(lowered::kv_owner_bytes(
            staging.kv_full.as_ref().expect("full KV staged"),
        ));
        lowered::fail_after_construction_stage(lowered::Gemma4ConstructionStage::FullKv)
            .map_err(|e| format!("gemma4 (lowered) full KV stage: {e:?}"))?;

        let full_kv_mode = match kv_policy {
            Gemma4LoweredKvPolicy::MoeF32 => "f32",
            Gemma4LoweredKvPolicy::DenseCompressed => {
                if ctx.kv_mode_override == Some("fwht3") {
                    "fwht3"
                } else {
                    "asym3"
                }
            }
        };
        let sliding_kv_mode = match kv_policy {
            Gemma4LoweredKvPolicy::MoeF32 => "f32",
            Gemma4LoweredKvPolicy::DenseCompressed => "q8-ring",
        };
        eprintln!(
            "  gemma4 lowered path: moe={} batched_opt_in={} kv_policy={} \
             (sliding {sliding_kv_mode} + full {full_kv_mode} KV; \
             sliding_layers={} full_layers={} max_seq={} physical_cap={} required_kv_bytes={})",
            lcfg.enable_moe_block,
            want_batched,
            kv_policy.label(),
            n_sliding,
            n_full,
            ctx.max_seq,
            sliding_physical_cap,
            required_kv,
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
    use super::{
        lowered_kv_policy, lowered_sliding_physical_cap, preflight_lowered_kv_budget,
        required_kv_bytes, select_gemma4_route, Gemma4LoweredKvPolicy, Gemma4Route,
    };
    use crate::lowered::{Gemma4Config, LayerType, RopeType};

    fn config(layer_types: Vec<LayerType>) -> Gemma4Config {
        Gemma4Config {
            dim: 8,
            n_layers: layer_types.len(),
            vocab_size: 32,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 2,
            sliding_head_dim: 32,
            sliding_n_kv_heads: 1,
            sliding_rope_theta: 10_000.0,
            sliding_window: 2,
            full_head_dim: 64,
            full_n_kv_heads: 1,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            hidden_dim: 16,
            enable_moe_block: false,
            moe_intermediate_size: 0,
            num_experts: 0,
            top_k_experts: 0,
            final_logit_softcapping: 0.0,
            tie_word_embeddings: true,
            embed_scale: 1.0,
            layer_types,
            has_vision: false,
            image_token_id: 0,
            boi_token_id: 0,
            eoi_token_id: 0,
            audio_token_id: 0,
            video_token_id: 0,
        }
    }

    #[test]
    fn lowered_moe_policy_is_f32_and_dense_policy_stays_compressed() {
        assert_eq!(lowered_kv_policy(true), Gemma4LoweredKvPolicy::MoeF32);
        assert_eq!(
            lowered_kv_policy(false),
            Gemma4LoweredKvPolicy::DenseCompressed
        );
        assert_eq!(Gemma4LoweredKvPolicy::MoeF32.label(), "moe-f32");
        assert_eq!(
            Gemma4LoweredKvPolicy::DenseCompressed.label(),
            "dense-compressed"
        );
    }

    #[test]
    fn f32_budget_counts_only_real_sliding_and_full_layers_and_both_tensors() {
        let cfg = config(vec![
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
        ]);
        // 3 sliding × 4 tokens × 1 head × 32 dims × 2 tensors × 4 bytes
        // + 2 full × 4 tokens × 1 head × 64 dims × 2 tensors × 4 bytes.
        let expected = 3 * 4 * 1 * 32 * 2 * 4 + 2 * 4 * 1 * 64 * 2 * 4;
        assert_eq!(
            required_kv_bytes(&cfg, 4, Gemma4LoweredKvPolicy::MoeF32).unwrap(),
            expected
        );
    }

    #[test]
    fn dense_budget_matches_q8_sliding_and_asym3_full_storage() {
        let cfg = config(vec![LayerType::Sliding, LayerType::Full]);
        // Sliding Q8: one head × one 32-value block × 34 bytes/token,
        // rounded to the F32 allocation unit for each K/V tensor.
        let sliding = 4usize * 34 * 2;
        // Full asym3: K=(4 + 3*64/8)=28 bytes/head and V=2*34 bytes/head,
        // each tensor rounded to four bytes, plus the two 32-float Givens tables.
        let full = 4usize * 28 + 4usize * (2 * 34) + 2 * (64 / 2) * 4;
        assert_eq!(
            required_kv_bytes(&cfg, 4, Gemma4LoweredKvPolicy::DenseCompressed).unwrap(),
            sliding + full
        );
    }

    #[test]
    fn kv_budget_rejects_overflow_before_gpu_allocation() {
        let cfg = config(vec![LayerType::Sliding]);
        let error = required_kv_bytes(&cfg, usize::MAX, Gemma4LoweredKvPolicy::MoeF32).unwrap_err();
        assert!(error.contains("overflow"), "{error}");
    }

    #[test]
    fn kv_budget_error_is_explicit_and_names_policy_need_free_and_capacity() {
        let error = preflight_lowered_kv_budget(Gemma4LoweredKvPolicy::MoeF32, 4096, 1024, 8192)
            .unwrap_err();
        assert!(error.contains("moe-f32"), "{error}");
        assert!(error.contains("required=4096"), "{error}");
        assert!(error.contains("free=1024"), "{error}");
        assert!(error.contains("max_seq=8192"), "{error}");
    }

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
