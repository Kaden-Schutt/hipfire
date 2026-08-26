// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek4 policy-aware MoE manifest + sealed-lowering adapter
//! (STEP-002 Phase 3, Task 8).
//!
//! The production functions in this module are implemented by the Task 8
//! adapter lane: policy-aware layer manifests (`ds4_expert_group_manifest` /
//! `ds4_expert_group_spec`), the pure routing-kind classifier
//! (`ds4_route_kind`), the typed router plans (`ds4_router_plan`), the
//! per-rank routed phases (`ds4_decode_phases`), and the model-owned
//! authority (`ds4_cached_moe_plans` / `ds4_resident_router_profiles` +
//! the full-matrix key/entry) consumed by forward.rs through borrowed plans
//! (`ds4_lower_borrowed_plan`). There is no local plan fabricator: every
//! lowered program consumes an already-resolved borrowed
//! [`ExpertGroupPlan`] from the authority.
//!
//! The test module below is the TDD contract: it must fail to compile (or
//! fail behaviorally) before the adapter exists, and must pass once it does.

// ─────────────────────────────────────────────────────────────────────────
// Production surface (implemented by the Task 8 lane)
// ─────────────────────────────────────────────────────────────────────────

use hipfire_dispatch::families::moe::{ExpertExecutionPlan, MoeExpertRef, RouterPlan};
use hipfire_dispatch::pipeline::{GemvInput, MoeActivationVariant, MoeProj, Step};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::moe_plan::{
    lower_moe_steps, MoEExecutionKind, MoEExecutionPolicy, MoeLowerError, MoeProgramParts,
    RoutedMoePhases, RoutedMoeStepPhases,
};
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_manifest::{
    resolve_expert_manifest_for_policy, ExpertExecutionIdentity, ExpertGroupPlan,
    ExpertManifestResolution, ExpertParallelism, ExpertResourceRequirements, ExpertSourceLayout,
};
use rdna_compute::GpuTensor;

use crate::deepseek4::{DeepseekV4Config, DeepseekV4LayerWeights, DeepseekV4Weights};

/// Canonical declared router identity for DS4 score-routed layers.
pub const ROUTER_BIAS_AWARE: &str = "bias_aware_topk";
/// Canonical declared router identity for DS4 hash-routed layers (device table).
pub const ROUTER_HASH: &str = "hash";
/// Declared identity for host-completed hash routing — the effective router
/// plan is `RouterPlan::Precomputed`, and `hash` is NOT an alias for it.
pub const ROUTER_PRECOMPUTED: &str = "precomputed";

/// Typed outcome of one DS4 MoE layer's routing (pre-down) phase.
///
/// `SharedOnly` means the layer contributes no routed program at all (MoE
/// disabled, expert blobs absent, or a hash layer without its tid2eid table) —
/// the shared expert alone has already seeded `ffn_out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ds4RouteSelection {
    /// No routed experts will run; no program may be built.
    SharedOnly,
    /// Bias-aware top-K (score layers, `layer_idx >= num_hash_layers`).
    BiasAware,
    /// Hash routing with the tid2eid table on device.
    Hash,
    /// Hash routing completed on the host (no `tid2eid_dev`); the effective
    /// router plan is `RouterPlan::Precomputed` with declared identity
    /// `precomputed` — never a `Hash` alias.
    PrecomputedHost,
}

/// Pure routing-kind classifier — transcribes the pre-change pre-down gating
/// (`ds4_bias_pre_down` / `ds4_hash_pre_down`):
/// - MoE disabled (`HIPFIRE_DEEPSEEK4_MOE=0`) or expert blobs absent →
///   shared-only layer;
/// - hash layer (`layer_idx < num_hash_layers`) without a tid2eid table →
///   shared-only layer;
/// - hash layer with the table on device → `Hash`;
/// - hash layer with the table host-only → `PrecomputedHost` (host fallback);
/// - every other routed layer → `BiasAware`.
pub fn ds4_route_kind(
    cfg: &DeepseekV4Config,
    layer_idx: usize,
    moe_on: bool,
    has_expert_blobs: bool,
    has_tid2eid: bool,
    tid2eid_on_device: bool,
) -> Ds4RouteSelection {
    if !moe_on || !has_expert_blobs {
        return Ds4RouteSelection::SharedOnly;
    }
    if layer_idx < cfg.num_hash_layers {
        if !has_tid2eid {
            return Ds4RouteSelection::SharedOnly;
        }
        return if tid2eid_on_device {
            Ds4RouteSelection::Hash
        } else {
            Ds4RouteSelection::PrecomputedHost
        };
    }
    Ds4RouteSelection::BiasAware
}

/// Parallelism + group size derived from the caller's execution policy.
pub fn ds4_parallelism(policy: &MoEExecutionPolicy) -> (ExpertParallelism, usize) {
    match policy.kind() {
        MoEExecutionKind::Single => (ExpertParallelism::Single, 1),
        MoEExecutionKind::Tp => (ExpertParallelism::TensorParallel, policy.rank_count()),
        MoEExecutionKind::Ep => (ExpertParallelism::ExpertParallel, policy.rank_count()),
    }
}

/// One layer-local DeepSeek4 expert-group declaration with the given declared
/// router identity. `n_experts` / `resources` / `source_layout` / `router`
/// mirror the DS4 loader's routed-expert surface (manifest names from
/// [`crate::arch::DeepseekV4::weight_manifest`]); `parallelism` is policy-
/// derived. The declared sources are the three **logical** packed projections
/// `experts_gate` / `experts_up` / `experts_down` (`PackedSeparate`): the
/// loader fuses gate+up into one runtime blob, but the manifest claims every
/// logical source so no expert projection can keep a stray per-weight
/// collective. The `precomputed` identity is selected during the model-owned
/// cache's cold effective-spec resolution
/// ([`ds4_effective_expert_group_manifest`]) for host-fallback layers; the
/// canonical manifest hook always declares the static identities
/// (`bias_aware_topk` / `hash`).
pub fn ds4_expert_group_spec(
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
    layer_idx: usize,
    declared_identity: &str,
) -> hipfire_runtime::weight_manifest::ExpertGroupSpec {
    let (parallelism, _) = ds4_parallelism(policy);
    hipfire_runtime::weight_manifest::ExpertGroupSpec {
        group: format!("deepseek4.moe.l{layer_idx}"),
        layer: Some(layer_idx),
        n_experts: cfg.n_routed_experts,
        parallelism,
        assignment: ExpertAssign::Stride,
        source_layout: ExpertSourceLayout::PackedSeparate {
            gate: "experts_gate".into(),
            up: "experts_up".into(),
            down: "experts_down".into(),
            sidecars: Vec::new(),
        },
        resources: ExpertResourceRequirements {
            // Exact F16 byte footprint of one expert's three separate
            // projections — gate (im·hidden), up (im·hidden), down
            // (hidden·im) — at 2 bytes per element: 3·im·hidden·2. Placement-
            // neutral for the lowering contract (the DS4 loader does not use
            // manifest fulfillment).
            bytes_per_expert: 3 * cfg.moe_intermediate_size * cfg.hidden_size * 2,
            alignment: 256,
        },
        router: "router_gate".into(),
        router_identity: declared_identity.to_string(),
        allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
    }
}

/// Policy-aware per-layer manifest (the `Architecture::expert_group_manifest`
/// hook): one layer-local group per MoE layer with the canonical static
/// identities (`hash` for layers `< num_hash_layers`, `bias_aware_topk`
/// otherwise), plus — when `num_nextn_predict_layers == 1` — one MTP group at
/// layer `num_hidden_layers` with the explicit bias-aware router identity.
/// MTP counts other than 0/1 are refused at plan resolution
/// ([`ds4_resolve_expert_plans`]), never declared here.
pub fn ds4_expert_group_manifest(
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
) -> Vec<hipfire_runtime::weight_manifest::ExpertGroupSpec> {
    let mut specs: Vec<_> = (0..cfg.num_hidden_layers)
        .map(|l| {
            let identity = if l < cfg.num_hash_layers {
                ROUTER_HASH
            } else {
                ROUTER_BIAS_AWARE
            };
            ds4_expert_group_spec(cfg, policy, l, identity)
        })
        .collect();
    if cfg.num_nextn_predict_layers == 1 {
        specs.push(ds4_expert_group_spec(
            cfg,
            policy,
            cfg.num_hidden_layers,
            ROUTER_BIAS_AWARE,
        ));
    }
    specs
}

/// The routed-expert scratch tensors one DS4 MoE program reads/writes, lifted
/// out of `DeepseekV4State` so the program builders stay CPU-testable.
#[derive(Clone, Copy)]
pub struct Ds4ProgramTensors<'a> {
    /// `[k_top]` i32-in-F32 selected expert ids (written by the router).
    pub topk_indices: &'a GpuTensor,
    /// `[k_top]` f32 normalized routing weights.
    pub topk_weights: &'a GpuTensor,
    /// `[hidden]` FWHT-rotated FFN input (gate_up reads this).
    pub ffn_x_rot: &'a GpuTensor,
    /// `[k_top, inter]` gate output.
    pub gate_batch: &'a GpuTensor,
    /// `[k_top, inter]` up output.
    pub up_batch: &'a GpuTensor,
    /// `[k_top, inter]` FWHT-rotated activation output (down input).
    pub rot_batch: &'a GpuTensor,
}

/// Where the routed down writes, per protocol (transcribed from the
/// pre-change paths):
/// - `ExpandedF32`: expanded per-expert write + a separate `MoeCombine` into
///   `out` — the single-GPU deterministic bias path (and its routed partial).
/// - `ResidualF32`: self-combining f32 residual into `out` — the single-GPU
///   hash path and the non-deterministic bias fallback.
/// - `I64`: reproducible int64 accumulator + `ConvertI64ToF32` into `partial`
///   — the parallel (EP/TP) decode and batched-TP prefill path.
#[derive(Clone, Copy)]
pub enum Ds4DownTarget<'a> {
    ExpandedF32 {
        down_expanded: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    ResidualF32 {
        out: &'a GpuTensor,
    },
    I64 {
        partial_i64: &'a GpuTensor,
        partial: &'a GpuTensor,
    },
}

/// Build one rank's typed routed phases for a DS4 MoE layer: the pre-change
/// op order [GateUp, Activation, down, (Combine | Convert)] with the router
/// (top-K) already executed arch-side. Reuses the shared indexed
/// MQ2-Lloyd Steps; no new kernels.
pub fn ds4_decode_phases<'a>(
    tensors: &Ds4ProgramTensors<'a>,
    experts: &'a MoeExpertRef<'a>,
    down: Ds4DownTarget<'a>,
    k_top: usize,
    inter: usize,
    hidden: usize,
    swiglu_limit: f32,
    batch_size: usize,
) -> Result<RoutedMoeStepPhases<'a>, String> {
    let gate_up = Step::IndexedMoeGemv {
        experts,
        which: MoeProj::GateUp {
            up_out: tensors.up_batch,
        },
        topk_indices: tensors.topk_indices,
        input: GemvInput::Prerotated(tensors.ffn_x_rot),
        out: tensors.gate_batch,
        k_top,
        batch_size,
    };
    // Routed activation rows: `batch_size * k_top` when batched (the batched
    // indexed I64 protocol requires the activation to span exactly the
    // batch×topk product; the sealed executor launches the batched silu·clamp
    // with this row count). batch_size == 1 keeps the scalar decode rows.
    let act_rows = batch_size
        .checked_mul(k_top)
        .ok_or_else(|| "ds4_decode_phases: batch_size*k_top overflow".to_string())?;
    let activation = Step::MoeActivation {
        variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit },
        gate: tensors.gate_batch,
        up: tensors.up_batch,
        rot_out: tensors.rot_batch,
        inter,
        k_top: act_rows,
    };
    let (down_step, combine, finish) = match down {
        Ds4DownTarget::ExpandedF32 { down_expanded, out } => {
            let step = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownExpanded,
                topk_indices: tensors.topk_indices,
                input: GemvInput::Prerotated(tensors.rot_batch),
                out: down_expanded,
                k_top,
                batch_size,
            };
            let combine = Step::MoeCombine {
                down_out: down_expanded,
                topk_weights: tensors.topk_weights,
                out,
                k: k_top,
                hidden,
                batch_size,
                inverse_perm: None,
            };
            (step, Some(combine), Vec::new())
        }
        Ds4DownTarget::ResidualF32 { out } => {
            let step = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidual {
                    topk_weights: tensors.topk_weights,
                },
                topk_indices: tensors.topk_indices,
                input: GemvInput::Prerotated(tensors.rot_batch),
                out,
                k_top,
                batch_size,
            };
            (step, None, Vec::new())
        }
        Ds4DownTarget::I64 {
            partial_i64,
            partial,
        } => {
            let step = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: tensors.topk_weights,
                },
                topk_indices: tensors.topk_indices,
                input: GemvInput::Prerotated(tensors.rot_batch),
                out: partial_i64,
                k_top,
                batch_size,
            };
            let finish = vec![Step::ConvertI64ToF32 {
                src: partial_i64,
                dst: partial,
                n: hidden
                    .checked_mul(batch_size)
                    .ok_or_else(|| "ds4_decode_phases: hidden*batch_size overflow".to_string())?,
            }];
            (step, None, finish)
        }
    };
    Ok(RoutedMoePhases {
        router: Vec::new(),
        gate_up: vec![gate_up],
        activation: vec![activation],
        down: vec![down_step],
        combine: combine.into_iter().collect(),
        finish,
    })
}

/// Build the typed router plan for one DS4 layer from its routing outcome.
///
/// The plan is typed identity metadata for the lowerer (only `selection()` is
/// consumed); the top-K kernels themselves already ran arch-side in the
/// pre-down. For `Hash` the `tokens` slot names the device token-id buffer
/// when present (the kernarg-variant fallback has no device buffer; the slot
/// is inert for lowering).
pub fn ds4_router_plan<'a>(
    kind: Ds4RouteSelection,
    scores: &'a GpuTensor,
    gate_bias: Option<&'a GpuTensor>,
    tid2eid: Option<&'a GpuTensor>,
    token_ids: Option<&'a GpuTensor>,
    topk_indices: &'a GpuTensor,
    topk_weights: &'a GpuTensor,
    k_top: usize,
    route_scale: f32,
) -> Result<RouterPlan<'a>, String> {
    match kind {
        Ds4RouteSelection::SharedOnly => Err("shared-only layers build no router plan".into()),
        Ds4RouteSelection::BiasAware => Ok(RouterPlan::BiasAwareTopK {
            scores,
            gate_bias: gate_bias
                .ok_or_else(|| "bias-aware router plan requires gate_bias".to_string())?,
            topk_indices,
            topk_weights,
            k_top,
            normalize: true,
            route_scale,
        }),
        Ds4RouteSelection::Hash => Ok(RouterPlan::Hash {
            scores,
            tokens: token_ids.unwrap_or(scores),
            tid2eid: tid2eid
                .ok_or_else(|| "hash router plan requires the tid2eid table".to_string())?,
            topk_indices,
            topk_weights,
            k_top,
            normalize: true,
            route_scale,
        }),
        Ds4RouteSelection::PrecomputedHost => Ok(RouterPlan::Precomputed {
            topk_indices,
            topk_weights,
            k_top,
            normalize: true,
            route_scale,
        }),
    }
}

/// Lower one DS4 MoE program through the runtime lowerer: validate the
/// declared identity against the typed router selection, derive the launch
/// schedule (collectives / zeroing / conversion placement) from the concrete
/// Steps, and seal it. Nothing is launched before this returns `Ok`.
///
/// The consumed [`ExpertGroupPlan`] is ALWAYS an already-resolved borrowed
/// plan from the model-owned authority (`ds4_cached_moe_plans` →
/// [`Ds4PlanCacheEntry::plan`]); no manifest authority is bypassed because
/// the plan originates from [`resolve_expert_manifest_for_policy`] (see
/// [`ds4_resolve_expert_plans`]).
pub fn ds4_lower_borrowed_plan<'mesh, 'step>(
    plan: &ExpertGroupPlan,
    policy: &'mesh MoEExecutionPolicy,
    router: RouterPlan<'step>,
    ranks: Vec<RoutedMoeStepPhases<'step>>,
) -> Result<hipfire_runtime::moe_plan::LoweredMoeProgram<'mesh, 'step>, MoeLowerError> {
    lower_moe_steps(
        plan,
        policy,
        MoeProgramParts {
            router,
            execution: ExpertExecutionPlan::IndexedQuantized,
            deferred_combine: false,

            ranks,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────
// Lane A: resident router profiles + model-owned manifest/cache resolution
// ─────────────────────────────────────────────────────────────────────────

/// Typed resource/cardinality failure of ONE layer's actual runtime bundle
/// (layer-scoped; the aggregate maps it into [`Ds4PlanCacheError`] with the
/// rank via [`Ds4BundleError::at_rank`]). Partial or misshapen bundles never
/// seed the plan cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ds4BundleError {
    /// One routed projection has a blob without its paired pointer table (or
    /// the pointer table without its blob), or a score layer has expert
    /// blobs but no router gate weight/bias — partial bundles are refused,
    /// never classified as shared-only.
    PartialExpertBundle {
        /// The affected layer (main index or `num_hidden_layers` for MTP).
        layer: usize,
        /// The projection: `"gate_up"`, `"down"`, or `"router_gate"`.
        projection: &'static str,
    },
    /// The device pointer table's F32 capacity (two F32 slots per u64
    /// pointer, matching the qwen35 convention) is below `2 * n_routed_experts`.
    ExpertPointerCapacity {
        layer: usize,
        projection: &'static str,
        /// Actual F32 slot capacity (`shape` product).
        capacity: usize,
        /// Required F32 slots: `2 * n_routed_experts`.
        required: usize,
    },
    /// The main-layer count of the weights bundle differs from the configured
    /// `num_hidden_layers` (too few OR too many) — checked before any
    /// profile indexing so a misshapen model never panics.
    MainLayerCount {
        /// Actual `layers.len()`.
        got: usize,
        /// Configured `num_hidden_layers`.
        expected: usize,
    },
    /// Hash LUT length != `vocab_size * num_experts_per_tok`.
    HashLutLength {
        layer: usize,
        got: usize,
        expected: usize,
    },
    /// Host-cached router gate bias length != `n_routed_experts` (the
    /// bias-aware route selection reads it on the CPU top-K path).
    RouterBiasLength {
        layer: usize,
        got: usize,
        expected: usize,
    },
}

impl Ds4BundleError {
    /// Map a layer-scoped bundle failure into the cache error, naming the
    /// rank whose weights produced it.
    pub(crate) fn at_rank(self, rank: usize) -> Ds4PlanCacheError {
        match self {
            Ds4BundleError::PartialExpertBundle { layer, projection } => {
                Ds4PlanCacheError::PartialExpertBundle {
                    rank,
                    layer,
                    projection,
                }
            }
            Ds4BundleError::ExpertPointerCapacity {
                layer,
                projection,
                capacity,
                required,
            } => Ds4PlanCacheError::ExpertPointerCapacity {
                rank,
                layer,
                projection,
                capacity,
                required,
            },
            Ds4BundleError::MainLayerCount { got, expected } => {
                Ds4PlanCacheError::LayerCountMismatch {
                    rank,
                    expected,
                    got,
                }
            }
            Ds4BundleError::HashLutLength {
                layer,
                got,
                expected,
            } => Ds4PlanCacheError::HashLutLength {
                rank,
                layer,
                got,
                expected,
            },
            Ds4BundleError::RouterBiasLength {
                layer,
                got,
                expected,
            } => Ds4PlanCacheError::RouterBiasLength {
                rank,
                layer,
                got,
                expected,
            },
        }
    }
}

/// Resident router profile of one MoE layer, derived from the ACTUAL loaded
/// residency (routed blobs, pointer tables, tid2eid host table, device LUT,
/// router gate) — the cache-key component that makes cached plans invalid
/// when residency changes. Mirrors [`Ds4RouteSelection`] with `Unavailable`
/// standing for layers that carry NO routed expert bundle at all
/// (shared-only). Runtime switches (e.g. MoE enablement) are deliberately
/// NOT part of residency: complete weights always classify truthfully.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ds4RouterProfile {
    /// No routed program (no routed expert bundle resident on this layer).
    Unavailable,
    /// Score-routed bias-aware top-K (main score layers and the MTP head).
    BiasAware,
    /// Hash-routed with the tid2eid table on device.
    HashDevice,
    /// Hash-routed with the table host-only (host-completed precomputed).
    PrecomputedHost,
}

impl Ds4RouterProfile {
    /// The effective routing selection this residency produces at forward
    /// time (mirrors [`ds4_route_kind`]'s transcription of the pre-change
    /// pre-down gating).
    ///
    /// Currently unconsumed by production: the forward path builds the typed
    /// router plan directly from [`Ds4RouteSelection`] (route state), and the
    /// effective-spec path maps the profile to the declared identity inline.
    /// The mapping is asserted by the canonical-identity test and remains
    /// the single source for the route-selection mapping; retain the narrow
    /// allowance until a production consumer exists.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn to_route_selection(self) -> Ds4RouteSelection {
        match self {
            Ds4RouterProfile::Unavailable => Ds4RouteSelection::SharedOnly,
            Ds4RouterProfile::BiasAware => Ds4RouteSelection::BiasAware,
            Ds4RouterProfile::HashDevice => Ds4RouteSelection::Hash,
            Ds4RouterProfile::PrecomputedHost => Ds4RouteSelection::PrecomputedHost,
        }
    }
}

/// Resident router profile of one MoE layer, transcribed from the ACTUAL
/// loaded layer bundle (`layer_idx < num_hidden_layers`, or the MTP slot at
/// `num_hidden_layers`) — FALLIBLE: partial or misshapen runtime bundles are
/// typed errors ([`Ds4BundleError`]), never silent classifications:
/// - NO routed expert resources at all — ALL FOUR of gate-up blob, gate-up
///   pointer table, down blob, down pointer table absent →
///   [`Ds4RouterProfile::Unavailable`] (shared-only layer);
/// - ANY other incomplete pairing of the four (blob without pointer table,
///   pointer table without blob, pointer-only) →
///   [`Ds4BundleError::PartialExpertBundle`];
/// - pointer-table F32 capacity below `2 * n_routed_experts` →
///   [`Ds4BundleError::ExpertPointerCapacity`];
/// - hash layer: missing `gate_weight` (the runtime computes hash-routing
///   gate scores from it) → [`Ds4BundleError::PartialExpertBundle`] with
///   projection `router_gate`; LUT length != `vocab_size * num_experts_per_tok`
///   → [`Ds4BundleError::HashLutLength`]; table on device → `HashDevice`,
///   host-only → `PrecomputedHost`;
/// - score layer: expert blobs without the complete bias-aware router data
///   (gate_weight + gate_bias + host gate_bias twin) →
///   [`Ds4BundleError::PartialExpertBundle`] / [`Ds4BundleError::RouterBiasLength`];
///   complete → `BiasAware`.
///
/// Runtime switches (MoE on/off) are NOT part of residency: the authority
/// must stay truthful regardless of the runtime enable switch; Lane B
/// bypasses plan lookup when MoE is disabled at forward time.
///
/// Consumed by the model-owned cache's profile derivation
/// (`ds4_resident_router_profiles`), which the forward entries reach through
/// the authority seams.
pub fn ds4_resident_router_profile(
    cfg: &DeepseekV4Config,
    layer: &DeepseekV4LayerWeights,
    layer_idx: usize,
) -> Result<Ds4RouterProfile, Ds4BundleError> {
    // 1. Routed expert bundle: BOTH projections must be present as PAIRED
    //    blob + pointer table. `Unavailable` (shared-only) ONLY when ALL
    //    FOUR expert fields are absent — any other incomplete pairing
    //    (blob-only, pointer-only, half-pairs) is a partial bundle.
    let gate_up_blob = layer.expert_gate_up_blob.is_some();
    let gate_up_ptrs = layer.expert_gate_up_ptrs.is_some();
    let down_blob = layer.expert_w2_blob.is_some();
    let down_ptrs = layer.expert_w2_ptrs.is_some();
    if !gate_up_blob && !gate_up_ptrs && !down_blob && !down_ptrs {
        return Ok(Ds4RouterProfile::Unavailable);
    }
    if !(gate_up_blob && gate_up_ptrs) {
        return Err(Ds4BundleError::PartialExpertBundle {
            layer: layer_idx,
            projection: "gate_up",
        });
    }
    if !(down_blob && down_ptrs) {
        return Err(Ds4BundleError::PartialExpertBundle {
            layer: layer_idx,
            projection: "down",
        });
    }
    // 2. Pointer-table capacity: F32 tensor of length `2 * n_routed_experts`
    //    (two F32 slots per u64 pointer, qwen35 convention).
    let required = 2 * cfg.n_routed_experts;
    let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().expect("paired above");
    let down_ptrs = layer.expert_w2_ptrs.as_ref().expect("paired above");
    for (ptrs, projection) in [(gate_up_ptrs, "gate_up"), (down_ptrs, "down")] {
        let capacity = ptrs.shape.iter().product::<usize>();
        if capacity < required {
            return Err(Ds4BundleError::ExpertPointerCapacity {
                layer: layer_idx,
                projection,
                capacity,
                required,
            });
        }
    }
    // 3. Hash layer: the runtime computes hash-routing gate scores from
    //    `gate_weight`, so it is required for BOTH the device and the
    //    host-completed hash paths; the LUT must cover the exact token×topk
    //    space. Bias requirements apply only to bias-aware (score) layers.
    if layer_idx < cfg.num_hash_layers {
        if layer.gate_weight.is_none() {
            return Err(Ds4BundleError::PartialExpertBundle {
                layer: layer_idx,
                projection: "router_gate",
            });
        }
        let expected = cfg.vocab_size * cfg.num_experts_per_tok;
        if layer.tid2eid_host.len() != expected {
            return Err(Ds4BundleError::HashLutLength {
                layer: layer_idx,
                got: layer.tid2eid_host.len(),
                expected,
            });
        }
        return Ok(if layer.tid2eid_dev.is_some() {
            Ds4RouterProfile::HashDevice
        } else {
            Ds4RouterProfile::PrecomputedHost
        });
    }
    // 4. Score-routed: the bias-aware top-K selection reads the router gate
    //    weight (score GEMV), the gate bias tensor (device top-k) and the
    //    host-cached gate_bias twin (CPU top-k-with-bias). All three must be
    //    actually resident at the exact cardinality.
    if layer.gate_weight.is_none() || layer.gate_bias.is_none() {
        return Err(Ds4BundleError::PartialExpertBundle {
            layer: layer_idx,
            projection: "router_gate",
        });
    }
    if layer.gate_bias_host.len() != cfg.n_routed_experts {
        return Err(Ds4BundleError::RouterBiasLength {
            layer: layer_idx,
            got: layer.gate_bias_host.len(),
            expected: cfg.n_routed_experts,
        });
    }
    Ok(Ds4RouterProfile::BiasAware)
}

/// Per-layer resident router profiles of the MAIN layers `0..num_hidden_layers`
/// in layer order, derived from the actual residency of the model-owned
/// weights, plus — when `num_nextn_predict_layers == 1` — the MTP slot at
/// layer `num_hidden_layers` appended last (mirroring the resolved plan
/// indexing). Fallible: the main-layer count is certified against
/// `num_hidden_layers` FIRST (too few OR too many → [`Ds4BundleError::MainLayerCount`],
/// before any profile indexing, so a misshapen model never panics); a missing
/// `mtp_layer` on a count-1 config is truthfully reported as
/// [`Ds4RouterProfile::Unavailable`] (refused later by the MTP-routeable
/// check). Runtime MoE enablement is NOT an input — complete weights always
/// classify truthfully.
///
/// Consumed by the canonical aggregate (`ds4_cached_moe_plans`) on every
/// acquisition — the complete per-rank residency certification.
pub fn ds4_resident_router_profiles(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
) -> Result<Vec<Ds4RouterProfile>, Ds4BundleError> {
    #[cfg(test)]
    ds4_resolve_seam::count_profile_if_instrumented();
    // Cardinality BEFORE indexing: a misshapen bundle is a typed error, not
    // a panic (resolve_layer is only safe when layers.len() == num_hidden_layers).
    if weights.layers.len() != cfg.num_hidden_layers {
        return Err(Ds4BundleError::MainLayerCount {
            got: weights.layers.len(),
            expected: cfg.num_hidden_layers,
        });
    }
    let mut profiles = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        profiles.push(ds4_resident_router_profile(
            cfg,
            weights.resolve_layer(l),
            l,
        )?);
    }
    if cfg.num_nextn_predict_layers == 1 {
        let mtp = match &weights.mtp_layer {
            Some(layer) => ds4_resident_router_profile(cfg, layer, cfg.num_hidden_layers)?,
            None => Ds4RouterProfile::Unavailable,
        };
        profiles.push(mtp);
    }
    Ok(profiles)
}

/// The config identity the DS4 static manifest + expert-group manifest are
/// derived from — every field either manifest's construction reads. A change
/// in any of these fields invalidates a cached resolution.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ds4ManifestConfigIdentity {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub n_routed_experts: usize,
    pub moe_intermediate_size: usize,
    pub num_hash_layers: usize,
    pub num_nextn_predict_layers: usize,
}

impl Ds4ManifestConfigIdentity {
    /// Capture the manifest-config identity of a full config.
    ///
    /// Consumed by the model-owned plan cache's cold key construction.
    pub fn of(cfg: &DeepseekV4Config) -> Self {
        Self {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            q_lora_rank: cfg.q_lora_rank,
            o_lora_rank: cfg.o_lora_rank,
            o_groups: cfg.o_groups,
            n_routed_experts: cfg.n_routed_experts,
            moe_intermediate_size: cfg.moe_intermediate_size,
            num_hash_layers: cfg.num_hash_layers,
            num_nextn_predict_layers: cfg.num_nextn_predict_layers,
        }
    }
}

/// The exact cache key of one model-owned DS4 manifest resolution: the full
/// [`MoEExecutionPolicy`] (whose `DeviceMesh` equality is epoch-sensitive, so
/// a fresh mesh with identical topology is a different key), the full
/// manifest config identity, and the COMPLETE validated per-rank residency
/// matrix — one profile vector per rank (main layers in order, then the
/// optional MTP slot). Cross-rank agreement is validated before the key is
/// built, but the FULL matrix is stored (not one rank's column), so any
/// rank's residency change invalidates the cached resolution.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ds4PlanCacheKey {
    pub policy: MoEExecutionPolicy,
    pub manifest_config: Ds4ManifestConfigIdentity,
    pub router_profiles: Vec<Vec<Ds4RouterProfile>>,
}

impl Ds4PlanCacheKey {
    /// Describe the FIRST differing key component, or `None` when the keys
    /// are equal. Deterministic field order: policy (kind + mesh epoch),
    /// manifest config identity, router profiles.
    ///
    /// Consumed by the model-owned plan cache's complete-key comparison on
    /// every acquisition.
    pub fn first_mismatch(&self, other: &Ds4PlanCacheKey) -> Option<String> {
        if self.policy != other.policy {
            return Some(
                "execution policy differs (kind or mesh epoch); cached plans are bound to the \
                 exact policy that resolved them"
                    .to_string(),
            );
        }
        if self.manifest_config != other.manifest_config {
            return Some(
                "manifest config identity differs (hidden/layer/expert/intermediate/hash/MTP \
                 geometry changed since the cached resolution)"
                    .to_string(),
            );
        }
        if self.router_profiles != other.router_profiles {
            return Some(
                "per-rank router residency matrix differs (any rank's main-layer LUT residency \
                 or MTP residency changed since the cached resolution)"
                    .to_string(),
            );
        }
        None
    }
}

/// The cached outcome of one exact-key DS4 manifest resolution. Success OR
/// failure is cached: the same key borrows this entry forever, a different
/// key is refused (no retry, no replacement, no global cache). Consumed by
/// Lane B's forward cutover (`plan(l)` / `key()` borrows).
#[derive(Debug)]
pub struct Ds4PlanCacheEntry {
    key: Ds4PlanCacheKey,
    result: Result<ExpertManifestResolution, String>,
}

/// Requesting the MTP plan when none is available — an explicit error, never
/// a panic. Live: `mtp_plan()` is called by the MTP public entries'
/// authority seams (forward.rs) before any GPU work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ds4MtpPlanError {
    /// `num_nextn_predict_layers == 0`: no MTP slot is declared.
    Unconfigured,
    /// `num_nextn_predict_layers > 1`: MTP is refused before resolution.
    Unsupported { count: usize },
    /// The cached resolution failed, so no MTP plan exists.
    ResolutionFailed(String),
    /// `num_nextn_predict_layers == 1` but the loaded weights' MTP slot is
    /// not routeable (absent or partial). The MAIN-layer authority tolerates
    /// an absent MTP slot; only the MTP selectors reach this accessor, and
    /// they must refuse with the typed error.
    MtpNotRouteable {
        rank: usize,
        profile: Ds4RouterProfile,
    },
}

impl Ds4PlanCacheEntry {
    /// Construct one cache entry inside the canonical model-owned cache path.
    fn new(key: Ds4PlanCacheKey, result: Result<ExpertManifestResolution, String>) -> Self {
        Self { key, result }
    }

    /// The exact key this entry was resolved under.
    pub fn key(&self) -> &Ds4PlanCacheKey {
        &self.key
    }

    /// The cached resolution outcome: `Ok` plans (main layers indexed then
    /// optional MTP) or the cached failure message.
    ///
    /// Currently unconsumed by production: the forward path borrows
    /// `plan(l)` / `mtp_plan()` directly; the full-resolution view is
    /// asserted by the cache tests. Retain the narrow allowance until a
    /// production consumer exists.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn resolution(&self) -> Result<&ExpertManifestResolution, &str> {
        self.result.as_ref().map_err(|e| e.as_str())
    }

    /// Borrow the plan for `layer` — main layers `0..num_hidden_layers` and
    /// the MTP slot at `num_hidden_layers` are indexed in that order. `None`
    /// when the resolution failed or the layer is out of range. The per-layer
    /// zero-allocation borrow Lane B's forward cutover consumes.
    pub fn plan(&self, layer: usize) -> Option<&ExpertGroupPlan> {
        self.result.as_ref().ok()?.plans.get(layer)
    }

    /// Borrow a MAIN-layer plan (`layer < num_hidden_layers`).
    ///
    /// Currently unconsumed by production: the forward path borrows the
    /// uniform `plan(l)` (main + MTP slots share one index space). Asserted
    /// by the cache tests; retain the narrow allowance until a production
    /// consumer exists.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn main_plan(&self, layer: usize) -> Option<&ExpertGroupPlan> {
        let main_layers = self.key.manifest_config.num_hidden_layers;
        if layer >= main_layers {
            return None;
        }
        self.plan(layer)
    }

    /// Borrow the optional MTP plan (the last plan, at layer
    /// `num_hidden_layers`). Missing/unsupported/failed MTP is an explicit
    /// error — never a panic. Live: the MTP public entries select it through
    /// the authority seams (forward.rs) before any GPU work.
    pub fn mtp_plan(&self) -> Result<&ExpertGroupPlan, Ds4MtpPlanError> {
        match self.key.manifest_config.num_nextn_predict_layers {
            0 => Err(Ds4MtpPlanError::Unconfigured),
            count if count > 1 => Err(Ds4MtpPlanError::Unsupported { count }),
            _ => {
                // The main-layer authority tolerates an absent/partial MTP
                // slot (recorded in the entry key's per-rank profile matrix);
                // only the MTP selectors reach this accessor, and a
                // non-routeable slot is an explicit typed refusal — never a
                // silent plan borrow.
                let mtp_idx = self.key.manifest_config.num_hidden_layers;
                let profile = self
                    .key
                    .router_profiles
                    .first()
                    .and_then(|profiles| profiles.get(mtp_idx))
                    .copied()
                    .unwrap_or(Ds4RouterProfile::Unavailable);
                if profile != Ds4RouterProfile::BiasAware {
                    return Err(Ds4MtpPlanError::MtpNotRouteable { rank: 0, profile });
                }
                let resolution = self
                    .result
                    .as_ref()
                    .map_err(|e| Ds4MtpPlanError::ResolutionFailed(e.clone()))?;
                let layer = self.key.manifest_config.num_hidden_layers;
                resolution.plans.get(layer).ok_or_else(|| {
                    Ds4MtpPlanError::ResolutionFailed(
                        "resolved plan list has no MTP slot".to_string(),
                    )
                })
            }
        }
    }
}

/// Errors from the model-owned DS4 plan cache entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ds4PlanCacheError {
    /// `num_nextn_predict_layers > 1` — refused BEFORE any resolution attempt
    /// (no MTP manifest entries or group are declared for unsupported counts).
    MtpCountUnsupported { count: usize },
    /// No weights-per-rank entries (or no rank profile vectors) were
    /// supplied.
    EmptyRankProfiles,
    /// The supplied weights-per-rank count does not match the execution
    /// policy's rank count — the per-rank residency matrix is incomplete.
    RankCountMismatch { expected: usize, got: usize },
    /// One rank's profile vector does not cover the expected layers (main
    /// layers then the optional MTP slot) — the per-rank residency matrix is
    /// misshapen.
    RankLayerCountMismatch {
        /// The disagreeing rank.
        rank: usize,
        /// Expected layers for that rank (main + optional MTP slot).
        expected: usize,
        /// Profiles supplied for that rank.
        got: usize,
    },
    /// The supplied rank router profiles disagree — resolution is refused
    /// until every rank's resident routing agrees.
    RankRouterProfileDisagreement {
        /// The first rank whose profile vector differs from rank 0's.
        first_disagreeing_rank: usize,
    },
    /// `num_nextn_predict_layers == 1` but a rank's MTP slot is NOT
    /// genuinely routeable [`Ds4RouterProfile::BiasAware`] (expert blobs +
    /// router gate weight + gate bias missing or partial). Refused BEFORE
    /// any resolution attempt; the cache is never seeded by the call.
    MtpNotRouteable {
        /// The first rank whose MTP slot is not routeable.
        rank: usize,
        /// That rank's actual MTP-slot residency profile.
        profile: Ds4RouterProfile,
    },
    /// One rank's actual main-layer count differs from the configured
    /// `num_hidden_layers` (too few OR too many) — checked before any
    /// profile indexing so a misshapen model never panics. Configured MTP
    /// presence is validated separately (the MTP-slot profile + routeable
    /// check).
    LayerCountMismatch {
        rank: usize,
        expected: usize,
        got: usize,
    },
    /// One routed projection has a blob without its paired pointer table (or
    /// vice versa), or a score layer has expert blobs without the router
    /// gate weight/bias — refused before any resolution; never seeded.
    PartialExpertBundle {
        rank: usize,
        layer: usize,
        projection: &'static str,
    },
    /// A pointer table's F32 capacity is below `2 * n_routed_experts`.
    ExpertPointerCapacity {
        rank: usize,
        layer: usize,
        projection: &'static str,
        capacity: usize,
        required: usize,
    },
    /// Hash LUT length != `vocab_size * num_experts_per_tok`.
    HashLutLength {
        rank: usize,
        layer: usize,
        got: usize,
        expected: usize,
    },
    /// Host-cached router gate bias length != `n_routed_experts`.
    RouterBiasLength {
        rank: usize,
        layer: usize,
        got: usize,
        expected: usize,
    },
    /// A different key is already cached on this weights instance; the
    /// mismatch is explicit and no retry/replacement is attempted.
    KeyMismatch { detail: String },
}

/// Validate that a resolved DS4 plan list is indexed main layers then the
/// optional MTP slot, with every plan carrying its own layer index.
///
/// Consumed by the shared resolution core
/// (`ds4_resolve_expert_plans_with_specs`) on every cold initialization.
pub fn ds4_validate_plan_order(
    cfg: &DeepseekV4Config,
    resolution: &ExpertManifestResolution,
) -> Result<(), String> {
    let mtp_slots = usize::from(cfg.num_nextn_predict_layers == 1);
    let expected = cfg.num_hidden_layers + mtp_slots;
    if resolution.plans.len() != expected {
        return Err(format!(
            "deepseek4: resolved expert plan count {} does not match expected {expected} \
             (num_hidden_layers={}, MTP slots={mtp_slots})",
            resolution.plans.len(),
            cfg.num_hidden_layers
        ));
    }
    for (idx, plan) in resolution.plans.iter().enumerate() {
        if plan.layer != Some(idx) {
            return Err(format!(
                "deepseek4: resolved expert plan[{idx}] has layer {:?}; expected Some({idx}) \
                 (plans are indexed main layers then the optional MTP slot)",
                plan.layer
            ));
        }
    }
    Ok(())
}

/// Resolve the DS4 expert manifest for one exact execution policy through the
/// shared policy-aware projection ([`resolve_expert_manifest_for_policy`]),
/// then validate plan length/order. The static manifest and specs are never
/// mutated. MTP counts other than 0/1 are refused here — before any
/// resolution attempt. Uses the CANONICAL static identities
/// ([`ds4_expert_group_manifest`]); the model-owned cache resolves through
/// the residency-aware TEMPORARY specs instead
/// ([`ds4_resolve_expert_plans_with_specs`]).
///
/// Narrow suppression: the canonical resolver is consumed by the
/// canonical-resolution tests (11–13); the production forward consumes the
/// effective-spec resolution through the cache. Retain until a production
/// consumer exists.
#[cfg_attr(not(test), allow(dead_code))]
pub fn ds4_resolve_expert_plans(
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
) -> Result<ExpertManifestResolution, String> {
    let specs = ds4_expert_group_manifest(cfg, policy);
    ds4_resolve_expert_plans_with_specs(cfg, policy, &specs)
}

/// Shared resolution core: validate the caller-supplied expert-group specs
/// (canonical OR temporary effective) against the static manifest through
/// the policy-aware projection, then validate plan length/order. The static
/// manifest and specs are never mutated. MTP counts other than 0/1 are
/// refused here — before any resolution attempt.
fn ds4_resolve_expert_plans_with_specs(
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
    specs: &[hipfire_runtime::weight_manifest::ExpertGroupSpec],
) -> Result<ExpertManifestResolution, String> {
    #[cfg(test)]
    ds4_resolve_seam::count_if_instrumented();
    if cfg.num_nextn_predict_layers > 1 {
        return Err(format!(
            "deepseek4: num_nextn_predict_layers={} is unsupported for the static MTP manifest \
             declaration; only 0 or 1 may be declared (refused before resolution)",
            cfg.num_nextn_predict_layers
        ));
    }
    let manifest = crate::arch::DeepseekV4::weight_manifest(cfg);
    let resolution = resolve_expert_manifest_for_policy(specs, &manifest, policy)
        .map_err(|e| format!("deepseek4: expert manifest resolution: {e}"))?;
    ds4_validate_plan_order(cfg, &resolution)?;
    Ok(resolution)
}

/// TEMPORARY effective expert-group specs for ONE cold cache initialization:
/// declared router identities derived from the VALIDATED residency profile
/// vector (rank 0's — the cell enforces cross-rank agreement before this is
/// called): `PrecomputedHost` layers resolve with the `precomputed` semantic
/// identity, device-hash layers keep `hash`, bias-aware layers keep
/// `bias_aware_topk`, and shared-only (`Unavailable`) layers keep the
/// canonical static identity (no routed program is ever lowered for them).
/// The canonical/static manifest declarations
/// ([`ds4_expert_group_manifest`]) are NEVER mutated — these specs are a
/// local temporary dropped after the initializer runs.
fn ds4_effective_expert_group_manifest(
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
    profiles: &[Ds4RouterProfile],
) -> Vec<hipfire_runtime::weight_manifest::ExpertGroupSpec> {
    let mtp_slots = usize::from(cfg.num_nextn_predict_layers == 1);
    let mut specs = Vec::with_capacity(cfg.num_hidden_layers + mtp_slots);
    for (l, &profile) in profiles.iter().take(cfg.num_hidden_layers).enumerate() {
        let identity = match profile {
            Ds4RouterProfile::HashDevice => ROUTER_HASH,
            Ds4RouterProfile::PrecomputedHost => ROUTER_PRECOMPUTED,
            Ds4RouterProfile::BiasAware => ROUTER_BIAS_AWARE,
            Ds4RouterProfile::Unavailable => {
                if l < cfg.num_hash_layers {
                    ROUTER_HASH
                } else {
                    ROUTER_BIAS_AWARE
                }
            }
        };
        specs.push(ds4_expert_group_spec(cfg, policy, l, identity));
    }
    if cfg.num_nextn_predict_layers == 1 {
        specs.push(ds4_expert_group_spec(
            cfg,
            policy,
            cfg.num_hidden_layers,
            ROUTER_BIAS_AWARE,
        ));
    }
    specs
}

/// Graph-safety gate for the single decode path: refuse HIP-graph capture
/// when the acquired reachable routed profile includes a host-completed hash
/// fallback (`PrecomputedHost` — host LUT, no device LUT; the per-step host
/// gathers are not capture-safe). The forward entry calls this BEFORE
/// warmup / `begin_graph_capture`, so no capture is ever started for a
/// graph-unsafe profile. `HashDevice` / `BiasAware` remain graph-admissible;
/// direct/eager mode remains supported.
pub fn ds4_graph_refuse_host_fallback(profiles: &[Ds4RouterProfile]) -> Result<(), String> {
    if profiles.contains(&Ds4RouterProfile::PrecomputedHost) {
        return Err(
            "deepseek4 graph mode refused BEFORE capture: the reachable routed profile \
             includes a host-completed hash fallback (PrecomputedHost — host LUT, no device \
             LUT); host gathers are not capture-safe. Run eager/direct mode or load the \
             tid2eid table on device."
                .to_string(),
        );
    }
    Ok(())
}

/// Lane-A model-owned plan cache entry point — canonical AGGREGATE accessor.
///
/// Callers pass the ordered `weights_per_rank` slice (Lane B's canonical
/// weights-per-rank slice) and the model's policy (the canonical stable
/// `weights_per_rank[0].moe_policy` on the Single path); the accessor
/// validates the rank count against the execution policy, derives the
/// COMPLETE per-rank residency matrix from the actual rank-local weights
/// itself, and ALWAYS owns/serves the cache cell of `weights_per_rank[0]`
/// (the rank-0 weights instance). No caller can choose which individual rank
/// cache resolves, and no caller claims rank 0 by an integer argument.
/// `Single` uses a one-element aggregate and owner 0.
///
/// Refusals happen BEFORE any resolution attempt, in deterministic order:
/// MTP count > 1 ([`Ds4PlanCacheError::MtpCountUnsupported`], at the
/// aggregate boundary before the empty/rank checks), empty / rank-count
/// mismatch ([`Ds4PlanCacheError::EmptyRankProfiles`] /
/// [`Ds4PlanCacheError::RankCountMismatch`]), per-rank main-layer
/// cardinality ([`Ds4PlanCacheError::LayerCountMismatch`]) and bundle
/// certification ([`Ds4PlanCacheError::PartialExpertBundle`] /
/// [`Ds4PlanCacheError::ExpertPointerCapacity`] /
/// [`Ds4PlanCacheError::HashLutLength`] /
/// [`Ds4PlanCacheError::RouterBiasLength`]), per-rank layer-count mismatch
/// ([`Ds4PlanCacheError::RankLayerCountMismatch`]), non-routeable MTP
/// residency on a count-1 config ([`Ds4PlanCacheError::MtpNotRouteable`]),
/// and rank router-profile disagreement
/// ([`Ds4PlanCacheError::RankRouterProfileDisagreement`]) — the complete
/// per-rank residency matrix is bound before any resolution, and a refused
/// call never seeds the cache.
///
/// COMPLETE-KEY AUTHORITY ON EVERY ACQUISITION: there is NO policy/config
/// fast path. Every call re-derives the full per-rank residency matrix
/// (rank × layer cardinality + bundle certification included) and the cell
/// compares the COMPLETE stored matrix key against the request, so changed
/// residency, partial bundles, rank disagreement, and cardinality are
/// caught on the very next acquisition — never a stale cached entry.
/// `get_or_init` still guarantees exactly-ONCE resolution per cell, and
/// same-key acquisitions borrow the identical entry (cached success OR
/// failure replayed verbatim).
///
/// LANE B CONTRACT: call the aggregate ONCE per forward pass and borrow the
/// returned [`Ds4PlanCacheEntry`] — per-layer `plan(l)` / `main_plan(l)` /
/// `mtp_plan()` borrowing is zero-allocation. Do NOT call this per layer.
///
/// Consumed by the forward entries' authority seams (`acquire_moe_authority_*`
/// in forward.rs) — the single production acquisition point.
#[cfg_attr(not(test), allow(dead_code))]
pub fn ds4_cached_moe_plans<'a>(
    weights_per_rank: &'a [DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
) -> Result<&'a Ds4PlanCacheEntry, Ds4PlanCacheError> {
    // 0. Unsupported MTP count is refused at the AGGREGATE boundary, before
    //    the empty/rank checks (a combined-invalid request reports the MTP
    //    refusal first).
    if cfg.num_nextn_predict_layers > 1 {
        return Err(Ds4PlanCacheError::MtpCountUnsupported {
            count: cfg.num_nextn_predict_layers,
        });
    }
    // 1. The aggregate must cover exactly the policy's ranks.
    if weights_per_rank.is_empty() {
        return Err(Ds4PlanCacheError::EmptyRankProfiles);
    }
    let ranks = policy.rank_count();
    if weights_per_rank.len() != ranks {
        return Err(Ds4PlanCacheError::RankCountMismatch {
            expected: ranks,
            got: weights_per_rank.len(),
        });
    }
    // 2. Derive the complete per-rank residency matrix from the ordered rank
    //    weights — on EVERY acquisition (layer cardinality + bundle
    //    certification included; typed errors, never a panic, never a seed).
    let mut matrix = Vec::with_capacity(ranks);
    for (rank, weights) in weights_per_rank.iter().enumerate() {
        let profiles = ds4_resident_router_profiles(cfg, weights).map_err(|e| e.at_rank(rank))?;
        matrix.push(profiles);
    }
    // 3. Cell: full matrix validation + complete-key comparison + exactly-
    //    once resolution on the rank-0 cell.
    ds4_cache_cell_moe_plans(&weights_per_rank[0].moe_plan_cache, cfg, policy, &matrix)
}

/// Main-layer-tolerant aggregate: identical to [`ds4_cached_moe_plans`]
/// EXCEPT the count-1 MTP slot is not required to be routeable — a config
/// declaring MTP whose loaded weights carry no MTP layer still resolves the
/// main layers (plain AR decode / mesh decode). The MTP slot's residency
/// stays in the entry key, so [`Ds4PlanCacheEntry::mtp_plan`] refuses it
/// with the typed `MtpNotRouteable` — only the MTP selectors
/// (`select_mtp_authority_*`) enforce the MTP requirement. Consumed by the
/// forward authority seams (`acquire_moe_authority_*`).
pub fn ds4_cached_moe_plans_main<'a>(
    weights_per_rank: &'a [DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
) -> Result<&'a Ds4PlanCacheEntry, Ds4PlanCacheError> {
    // 0. Unsupported MTP count is refused at the AGGREGATE boundary, before
    //    the empty/rank checks (a combined-invalid request reports the MTP
    //    refusal first).
    if cfg.num_nextn_predict_layers > 1 {
        return Err(Ds4PlanCacheError::MtpCountUnsupported {
            count: cfg.num_nextn_predict_layers,
        });
    }
    // 1. The aggregate must cover exactly the policy's ranks.
    if weights_per_rank.is_empty() {
        return Err(Ds4PlanCacheError::EmptyRankProfiles);
    }
    let ranks = policy.rank_count();
    if weights_per_rank.len() != ranks {
        return Err(Ds4PlanCacheError::RankCountMismatch {
            expected: ranks,
            got: weights_per_rank.len(),
        });
    }
    // 2. Derive the complete per-rank residency matrix from the ordered rank
    //    weights — on EVERY acquisition.
    let mut matrix = Vec::with_capacity(ranks);
    for (rank, weights) in weights_per_rank.iter().enumerate() {
        let profiles = ds4_resident_router_profiles(cfg, weights).map_err(|e| e.at_rank(rank))?;
        matrix.push(profiles);
    }
    // 3. Cell (main-tolerant): full matrix validation minus the MTP
    //    routeability refusal + complete-key comparison + exactly-once
    //    resolution on the rank-0 cell.
    ds4_cache_cell_moe_plans_impl(
        &weights_per_rank[0].moe_plan_cache,
        cfg,
        policy,
        &matrix,
        false,
    )
}

/// Low-level cache-cell entry (PRIVATE test seam): full matrix validation
/// plus exactly-once resolution on ONE cell. The aggregate
/// ([`ds4_cached_moe_plans`]) is the only production caller; it derives the
/// matrix from the ordered rank weights and pins the rank-0 cell, so this
/// seam cannot be reached with a forged owner or a caller-chosen matrix —
/// there is no crate-wide bypass. Tests in this module exercise the seam
/// directly to prove the concurrency and validation contract on a bare cell.
///
/// Only the aggregate calls this in production (private test seam).
#[cfg_attr(not(test), allow(dead_code))]
fn ds4_cache_cell_moe_plans<'a>(
    cache: &'a std::sync::OnceLock<Ds4PlanCacheEntry>,
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
    matrix: &[Vec<Ds4RouterProfile>],
) -> Result<&'a Ds4PlanCacheEntry, Ds4PlanCacheError> {
    ds4_cache_cell_moe_plans_impl(cache, cfg, policy, matrix, true)
}

/// Cell core shared by the strict aggregate ([`ds4_cache_cell_moe_plans`])
/// and the main-tolerant aggregate ([`ds4_cached_moe_plans_main`]):
/// `require_mtp_routeable = false` skips the count-1 MTP-slot routeability
/// refusal so plain AR decode works on configs whose MTP layer is absent;
/// the MTP slot's residency stays in the entry key and
/// [`Ds4PlanCacheEntry::mtp_plan`] refuses it with the typed
/// `MtpNotRouteable`.
fn ds4_cache_cell_moe_plans_impl<'a>(
    cache: &'a std::sync::OnceLock<Ds4PlanCacheEntry>,
    cfg: &DeepseekV4Config,
    policy: &MoEExecutionPolicy,
    matrix: &[Vec<Ds4RouterProfile>],
    require_mtp_routeable: bool,
) -> Result<&'a Ds4PlanCacheEntry, Ds4PlanCacheError> {
    // 1. MTP count > 1 is refused before anything else (no manifest
    //    entries/group are ever declared for unsupported counts).
    if cfg.num_nextn_predict_layers > 1 {
        return Err(Ds4PlanCacheError::MtpCountUnsupported {
            count: cfg.num_nextn_predict_layers,
        });
    }
    // 2. The per-rank residency matrix must be complete: exactly one profile
    //    vector per policy rank.
    let ranks = policy.rank_count();
    let first = matrix.first().ok_or(Ds4PlanCacheError::EmptyRankProfiles)?;
    if matrix.len() != ranks {
        return Err(Ds4PlanCacheError::RankCountMismatch {
            expected: ranks,
            got: matrix.len(),
        });
    }
    // 3. Each rank's vector must cover the expected layers: main layers then
    //    the optional MTP slot (mirroring the resolved plan indexing).
    let expected_layers = cfg.num_hidden_layers + usize::from(cfg.num_nextn_predict_layers == 1);
    for (rank, profiles) in matrix.iter().enumerate() {
        if profiles.len() != expected_layers {
            return Err(Ds4PlanCacheError::RankLayerCountMismatch {
                rank,
                expected: expected_layers,
                got: profiles.len(),
            });
        }
    }
    // 4. count==1: when the strict aggregate requires it, EVERY rank's MTP
    //    slot must be genuinely routeable BiasAware (expert blobs + complete
    //    router data). A missing or partial MTP is refused BEFORE any
    //    resolution attempt (and before the cross-rank agreement check, so
    //    the failing RANK is reported) and never seeds the cache. The
    //    main-tolerant aggregate skips this so plain AR decode works with an
    //    absent MTP layer; `mtp_plan()` still refuses the slot typed.
    if require_mtp_routeable && cfg.num_nextn_predict_layers == 1 {
        let mtp_idx = cfg.num_hidden_layers;
        for (rank, profiles) in matrix.iter().enumerate() {
            let profile = profiles[mtp_idx];
            if profile != Ds4RouterProfile::BiasAware {
                return Err(Ds4PlanCacheError::MtpNotRouteable { rank, profile });
            }
        }
    }
    // 5. All rank router profiles must agree before resolution; the cache is
    //    never seeded by a disagreed call.
    for (rank, profiles) in matrix.iter().enumerate() {
        if profiles != first {
            return Err(Ds4PlanCacheError::RankRouterProfileDisagreement {
                first_disagreeing_rank: rank,
            });
        }
    }
    // 6. The exact key: full policy (mesh epoch-sensitive equality), full
    //    manifest config identity, the COMPLETE validated matrix. Built on
    //    EVERY acquisition (one certification per aggregate call); the cell
    //    compares it against the stored entry's complete key below.
    #[cfg(test)]
    ds4_resolve_seam::count_key_if_instrumented();
    let key = Ds4PlanCacheKey {
        policy: policy.clone(),
        manifest_config: Ds4ManifestConfigIdentity::of(cfg),
        router_profiles: matrix.to_vec(),
    };
    // 7. Resolve AT MOST ONCE: `get_or_init` runs the initializer exactly
    //    once (winning caller's key resolved and stored; losing concurrent
    //    initializers never run). The initializer resolves through TEMPORARY
    //    effective specs derived from the validated residency matrix — a
    //    `PrecomputedHost` layer resolves with the `precomputed` semantic
    //    identity, device-hash and bias-aware layers keep their exact
    //    identities; the canonical/static manifest declarations are never
    //    mutated. Every caller then compares the COMPLETE stored key against
    //    its own request key: same key borrows the entry forever, a
    //    different key is an explicit mismatch.
    let entry = cache.get_or_init(|| {
        let specs = ds4_effective_expert_group_manifest(cfg, policy, &matrix[0]);
        Ds4PlanCacheEntry::new(
            key.clone(),
            ds4_resolve_expert_plans_with_specs(cfg, policy, &specs),
        )
    });
    match entry.key().first_mismatch(&key) {
        None => Ok(entry),
        Some(detail) => Err(Ds4PlanCacheError::KeyMismatch { detail }),
    }
}

/// Test-only DS4 manifest-resolution instrumentation (call-count seam).
///
/// The model-owned cache contract is: `OnceLock::get_or_init` resolves the
/// static expert manifest AT MOST ONCE per weights instance, even when
/// concurrent first calls race. This seam counts `ds4_resolve_expert_plans`
/// invocations so tests can prove the exactly-once contract with a counter
/// instead of inspecting cache internals.
///
/// Counting is per-THREAD, not global: only threads the seam test arms (its
/// own thread and every worker it spawns) increment [`RESOLUTIONS`], so the
/// suite's other tests running in parallel can never pollute the delta.
/// `SeamGuard` serializes seam tests with each other (poison-tolerant) and
/// clears the armed set on drop.
#[cfg(test)]
pub(crate) mod ds4_resolve_seam {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::thread::ThreadId;

    /// Serializes seam tests so their delta assertions cannot observe each
    /// other's increments (poison-tolerant: a failed seam test does not
    /// brick later ones).
    pub static LOCK: Mutex<()> = Mutex::new(());
    /// Threads whose `super::ds4_resolve_expert_plans` calls increment
    /// [`RESOLUTIONS`]. Guarded by the mutex itself.
    static ARMED: std::sync::LazyLock<Mutex<HashSet<ThreadId>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashSet::new()));
    /// Number of manifest resolutions performed by armed threads.
    static RESOLUTIONS: AtomicUsize = AtomicUsize::new(0);
    /// Number of per-rank profile derivations
    /// (`super::ds4_resident_router_profiles`) performed by armed threads.
    static PROFILE_DERIVATIONS: AtomicUsize = AtomicUsize::new(0);
    /// Number of owned cache-key constructions (`super::Ds4PlanCacheKey`)
    /// performed by armed threads.
    static KEY_CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

    /// Reset the instrumentation state (call while holding [`LOCK`]).
    pub fn reset() {
        RESOLUTIONS.store(0, Ordering::Relaxed);
        PROFILE_DERIVATIONS.store(0, Ordering::Relaxed);
        KEY_CONSTRUCTIONS.store(0, Ordering::Relaxed);
        ARMED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// The number of resolutions counted so far (call while holding [`LOCK`]).
    pub fn count() -> usize {
        RESOLUTIONS.load(Ordering::Relaxed)
    }

    /// The number of profile derivations counted so far (call while holding
    /// [`LOCK`]).
    pub fn profiles() -> usize {
        PROFILE_DERIVATIONS.load(Ordering::Relaxed)
    }

    /// The number of owned key constructions counted so far (call while
    /// holding [`LOCK`]).
    pub fn keys() -> usize {
        KEY_CONSTRUCTIONS.load(Ordering::Relaxed)
    }

    /// Arm the calling thread: its `ds4_resolve_expert_plans` calls now
    /// increment [`RESOLUTIONS`]. Spawned workers must arm themselves inside
    /// their closures.
    pub fn arm() {
        ARMED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(std::thread::current().id());
    }

    /// RAII guard: arms the creating thread for its lifetime (holding
    /// [`LOCK`] so delta assertions are race-free against other seam tests)
    /// and clears the armed set on drop.
    pub struct SeamGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl SeamGuard {
        pub fn on() -> Self {
            let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            reset();
            arm();
            SeamGuard { _lock }
        }
    }

    impl Drop for SeamGuard {
        fn drop(&mut self) {
            reset();
        }
    }

    /// Increment the resolution counter when the CALLING thread is armed.
    pub fn count_if_instrumented() {
        let armed = ARMED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&std::thread::current().id());
        if armed {
            RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment the profile-derivation counter when the CALLING thread is
    /// armed.
    pub fn count_profile_if_instrumented() {
        let armed = ARMED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&std::thread::current().id());
        if armed {
            PROFILE_DERIVATIONS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment the key-construction counter when the CALLING thread is
    /// armed.
    pub fn count_key_if_instrumented() {
        let armed = ARMED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&std::thread::current().id());
        if armed {
            KEY_CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// #[cfg(test)] mod tests { ... } — see bottom of file.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek4::DeepseekV4Config;
    use hipfire_dispatch::families::moe::{MoeExpertRef, RouterPlan};
    use hipfire_dispatch::pipeline::{GemvInput, MoeActivationVariant, MoeProj, Step};
    use hipfire_runtime::moe_plan::{
        MoEExecutionKind, MoEExecutionPolicy, MoeLowerError, RoutedMoePhases,
    };
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    use hipfire_runtime::weight_manifest::{
        ExpertExecutionIdentity, ExpertParallelism, ExpertPostCombineAllReduce, ExpertSourceLayout,
    };
    use rdna_compute::{DType, GpuTensor};

    // ── Lane A test scaffolding ──────────────────────────────────────────
    fn cfg_count0() -> DeepseekV4Config {
        let mut cfg = tiny_cfg();
        cfg.num_nextn_predict_layers = 0;
        cfg
    }

    fn cfg_count2() -> DeepseekV4Config {
        let mut cfg = tiny_cfg();
        cfg.num_nextn_predict_layers = 2;
        cfg
    }

    /// TP-eligible tiny config: `moe_intermediate_size` must satisfy
    /// `inter % ranks == 0` and `(inter / ranks) % 256 == 0` for ranks 1..=2
    /// (the projection-path TP role-dimension gate), so 512 works for Tp=2.
    fn tp_cfg() -> DeepseekV4Config {
        let mut cfg = tiny_cfg();
        cfg.moe_intermediate_size = 512;
        cfg
    }

    /// Model-owned weights with routed-expert residency derived from `cfg`:
    /// EVERY routed layer carries the COMPLETE certified bundle — paired
    /// blob + pointer table (F32 capacity `2 * n_routed_experts`) for both
    /// gate-up and down, PLUS the router `gate_weight` (the runtime computes
    /// hash-routing AND bias-aware gate scores from it). Hash layers
    /// (`l < num_hash_layers`) additionally get the tid2eid host table
    /// (exactly `vocab_size * num_experts_per_tok`) + device LUT, with no
    /// gate BIAS (bias applies only to bias-aware score layers); score
    /// layers get the complete bias-aware router data (gate weight tensor,
    /// gate bias tensor, host-cached gate_bias twin of exactly
    /// `n_routed_experts`). When `num_nextn_predict_layers == 1`, an MTP
    /// layer with the same complete routed residency is loaded so the MTP
    /// slot classifies `BiasAware`. `moe_plan_cache` starts empty (the
    /// Lane-A model-owned cache); `moe_policy` is the canonical stable
    /// Single policy.
    fn test_weights(cfg: &DeepseekV4Config) -> crate::deepseek4::DeepseekV4Weights {
        use crate::deepseek4::DeepseekV4LayerWeights;
        let ptrs = || owned_f32(2 * cfg.n_routed_experts); // two F32 slots per u64 pointer
        let routed_layer = || {
            let mut layer = DeepseekV4LayerWeights::new_empty(0);
            layer.expert_gate_up_blob = Some(owned_f32(1));
            layer.expert_gate_up_ptrs = Some(ptrs());
            layer.expert_w2_blob = Some(owned_f32(1));
            layer.expert_w2_ptrs = Some(ptrs());
            layer.gate_weight = Some(owned_f32(1));
            layer.gate_bias = Some(owned_f32(1));
            layer.gate_bias_host = vec![0.0; cfg.n_routed_experts];
            layer
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let mut layer = routed_layer();
            if l < cfg.num_hash_layers {
                // Hash layers keep `gate_weight` (gate-score GEMV) but carry
                // no gate bias and no bias twin (bias is bias-aware-only).
                layer.gate_bias = None;
                layer.gate_bias_host.clear();
                layer.tid2eid_host = vec![0u32; cfg.vocab_size * cfg.num_experts_per_tok];
                layer.tid2eid_dev = Some(owned_f32(1));
            }
            layers.push(layer);
        }
        let mtp_layer = if cfg.num_nextn_predict_layers == 1 {
            Some(routed_layer())
        } else {
            None
        };
        crate::deepseek4::DeepseekV4Weights {
            // Test fixture: no frozen MQ2R backend (CPU-side plan resolution).
            mq2r_backend: crate::backend::Mq2rBackend::Portable,
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0,
            layers,
            mtp_layer,
            dspark: None,
            moe_load_layout: crate::deepseek4::Ds4MoeLoadLayout::Single,
            moe_policy: MoEExecutionPolicy::single(),
            moe_plan_cache: std::sync::OnceLock::new(),
            _scaffold: (),
        }
    }

    /// An owned single-element F32 CPU tensor (test-only fake GPU buffer).
    fn owned_f32(numel: usize) -> GpuTensor {
        let bytes = numel * 4;
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype: DType::F32,
        }
    }

    // ── tiny synthetic config (4 layers; layers 0..2 hash-routed) ─────────
    fn tiny_cfg() -> DeepseekV4Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 16,
            "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            "q_lora_rank": 32, "o_lora_rank": 32, "qk_rope_head_dim": 8, "o_groups": 2,
            "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "moe_intermediate_size": 48, "routed_scaling_factor": 1.0,
            "topk_method": "noaux_tc", "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true, "swiglu_limit": 7.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 3, "hc_eps": 1e-6,
            "index_n_heads": 4, "index_head_dim": 16, "index_topk": 8,
            "compress_ratios": [], "compress_rope_theta": 10000.0,
            "rope_theta": 10000.0, "rope_scaling_factor": 16.0,
            "rope_scaling_original_max_position_embeddings": 4096,
            "rope_scaling_beta_fast": 32, "rope_scaling_beta_slow": 1,
            "sliding_window": 128, "num_nextn_predict_layers": 1, "num_hash_layers": 2
        }))
        .expect("tiny deepseek4 config")
    }

    fn synth_with_bytes(dtype: DType, numel: usize, bytes: usize) -> &'static GpuTensor {
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        Box::leak(Box::new(GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype,
        }))
    }

    fn synth_f32(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::F32, numel, numel * 4)
    }

    fn synth_i64(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::Raw, numel, numel * 8)
    }

    /// Raw I64 tensor with the `[batch, hidden]` shape the sealed batched
    /// TpI64 protocol requires for the down output (shallow-alias semantics).
    fn synth_i64_2d(rows: usize, cols: usize) -> &'static GpuTensor {
        let numel = rows * cols;
        let bytes = numel * 8;
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        Box::leak(Box::new(GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![rows, cols],
            dtype: DType::Raw,
        }))
    }

    fn expert_ref(expert_m: usize, expert_k: usize) -> &'static MoeExpertRef<'static> {
        Box::leak(Box::new(MoeExpertRef {
            gate_up_ptrs: synth_f32(4),
            down_ptrs: synth_f32(4),
            dummy_gate_up: None,
            dtype: DType::MQ2G256Lloyd,
            n_experts: 8,
            expert_m,
            expert_k,
            owned: &[],
        }))
    }

    /// Batched-protocol expert ref: pointer tables with the full
    /// `n_experts × 8`-byte u64 capacity the sealed batched chain checks.
    fn expert_ref_batched(expert_m: usize, expert_k: usize) -> &'static MoeExpertRef<'static> {
        Box::leak(Box::new(MoeExpertRef {
            gate_up_ptrs: synth_f32(2 * 8),
            down_ptrs: synth_f32(2 * 8),
            dummy_gate_up: None,
            dtype: DType::MQ2G256Lloyd,
            n_experts: 8,
            expert_m,
            expert_k,
            owned: &[],
        }))
    }

    fn decode_tensors(k_top: usize, im: usize) -> Ds4ProgramTensors<'static> {
        Ds4ProgramTensors {
            topk_indices: synth_f32(k_top),
            topk_weights: synth_f32(k_top),
            ffn_x_rot: synth_f32(64),
            gate_batch: synth_f32(k_top * im),
            up_batch: synth_f32(k_top * im),
            rot_batch: synth_f32(k_top * im),
        }
    }

    /// Batched-protocol scratch tensors: every routed buffer sized for
    /// `batch·k_top` rows (the sealed batched chain capacity-checks the real
    /// byte sizes of top-k slots, gate-up input, and gate/up/rotation).
    fn decode_tensors_batched(
        n: usize,
        k_top: usize,
        im: usize,
        hidden: usize,
    ) -> Ds4ProgramTensors<'static> {
        Ds4ProgramTensors {
            topk_indices: synth_f32(n * k_top),
            topk_weights: synth_f32(n * k_top),
            ffn_x_rot: synth_f32(n * hidden),
            gate_batch: synth_f32(n * k_top * im),
            up_batch: synth_f32(n * k_top * im),
            rot_batch: synth_f32(n * k_top * im),
        }
    }

    /// Typed `StepCollective` equality (the enum carries no derives).
    fn collective_eq(
        a: &hipfire_dispatch::pipeline::StepCollective,
        b: &hipfire_dispatch::pipeline::StepCollective,
    ) -> bool {
        match (a, b) {
            (
                hipfire_dispatch::pipeline::StepCollective::None,
                hipfire_dispatch::pipeline::StepCollective::None,
            ) => true,
            (
                hipfire_dispatch::pipeline::StepCollective::AllReduceI64Tp { dim: x },
                hipfire_dispatch::pipeline::StepCollective::AllReduceI64Tp { dim: y },
            ) => x == y,
            (
                hipfire_dispatch::pipeline::StepCollective::ZeroI64Only { dim: x },
                hipfire_dispatch::pipeline::StepCollective::ZeroI64Only { dim: y },
            ) => x == y,
            (
                hipfire_dispatch::pipeline::StepCollective::AllReduce { kind: xk, dim: xd },
                hipfire_dispatch::pipeline::StepCollective::AllReduce { kind: yk, dim: yd },
            ) => xk == yk && xd == yd,
            _ => false,
        }
    }

    fn single_policy() -> MoEExecutionPolicy {
        MoEExecutionPolicy::single()
    }

    fn tp_policy(ranks: usize) -> MoEExecutionPolicy {
        MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
        )
        .unwrap()
    }

    fn ep_policy(ranks: usize) -> MoEExecutionPolicy {
        MoEExecutionPolicy::new(
            MoEExecutionKind::Ep,
            DeviceMesh::rect(&[(DimKind::Ep, ranks)]),
        )
        .unwrap()
    }

    /// Build the sealed program through the PRODUCTION authority + builder:
    /// the plan for `layer_idx` comes from the model-owned cache under the
    /// canonical Single policy, and the program is constructed with the same
    /// builder the mesh MoE step consumes. Returns the program for the
    /// approved inspection API (no Debug traces).
    fn production_single_program<'a, 'b>(
        cfg: &DeepseekV4Config,
        weights: &'a crate::deepseek4::DeepseekV4Weights,
        layer_idx: usize,
        router: RouterPlan<'b>,
        phases: RoutedMoePhases<Step<'b>>,
    ) -> hipfire_runtime::moe_plan::LoweredMoeProgram<'a, 'b> {
        let entry = ds4_cached_moe_plans(std::slice::from_ref(weights), cfg, &weights.moe_policy)
            .expect("single aggregate resolves");
        let plan = entry.plan(layer_idx).expect("cached plan for the layer");
        crate::forward::build_ds4_parallel_program(plan, &weights.moe_policy, router, vec![phases])
            .expect("sealed program")
    }

    // ── 1. bias router: old sequencing vs the lowered program ──────────────
    #[test]
    fn ds4_bias_router_old_vs_lowered() {
        // OLD (pre-change) single-GPU bias decode — transcribed from
        // `ffn_routed` + `run_moe_decode_bias_aware`: after `moe_route`
        // (router GEMV + sqrt_softplus, which stays arch-side) the routed ops
        // were [topk_bias_aware, gate_up, silu·clamp, rotate, down_expanded,
        // combine] with the combine accumulating into `ffn_out` (already
        // seeded by the shared expert). The five routed ops map to four
        // Steps: [GateUp, Activation, DownExpanded, Combine].
        let cfg = tiny_cfg();
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let down_expanded = synth_f32(cfg.num_experts_per_tok * cfg.hidden_size);
        let ffn_out = synth_f32(cfg.hidden_size);
        let phases = ds4_decode_phases(
            &tensors,
            expert_ref(cfg.moe_intermediate_size, cfg.hidden_size),
            Ds4DownTarget::ExpandedF32 {
                down_expanded,
                out: ffn_out,
            },
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            cfg.hidden_size,
            cfg.swiglu_limit,
            1,
        )
        .unwrap();
        // OLD phase order: router (topk, arch-side) is empty; the program is
        // [GateUp, Activation, DownExpanded, Combine].
        assert_eq!(phases.router.len(), 0);
        assert_eq!(phases.gate_up.len(), 1);
        assert_eq!(phases.activation.len(), 1);
        assert_eq!(phases.down.len(), 1);
        assert_eq!(phases.combine.len(), 1);
        assert_eq!(phases.finish.len(), 0);
        // The gate-up step reads the FWHT-rotated FFN input; the down step
        // writes the expanded per-expert buffer; the combine folds it into
        // the shared-expert-seeded ffn_out (no zeroing on the Single path —
        // identical to the old run_bias_aware accumulation).
        assert!(matches!(
            &phases.gate_up[0],
            Step::IndexedMoeGemv {
                which: MoeProj::GateUp { .. },
                input: GemvInput::Prerotated(_),
                ..
            }
        ));
        assert!(matches!(
            &phases.activation[0],
            Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { .. },
                ..
            }
        ));
        assert!(matches!(
            &phases.down[0],
            Step::IndexedMoeGemv {
                which: MoeProj::DownExpanded,
                out,
                ..
            } if std::ptr::eq(*out, down_expanded)
        ));
        assert!(matches!(
            &phases.combine[0],
            Step::MoeCombine {
                inverse_perm: None,
                ..
            }
        ));

        let router = ds4_router_plan(
            Ds4RouteSelection::BiasAware,
            synth_f32(cfg.n_routed_experts),
            Some(synth_f32(cfg.n_routed_experts)),
            None,
            None,
            tensors.topk_indices,
            tensors.topk_weights,
            cfg.num_experts_per_tok,
            2.2,
        )
        .unwrap();
        let w = test_weights(&cfg);
        let program = production_single_program(&cfg, &w, 3, router, phases);
        // The old code accumulated in-place into ffn_out on the single GPU:
        // a Single program with exactly the four routed Steps; the Single
        // executor carries no parallel schedule (collectives/zeroing None).
        use hipfire_runtime::moe_plan::MoeExecutorKind;
        assert_eq!(program.executor_kind(), MoeExecutorKind::SingleMesh);
        assert_eq!(program.rank_count(), 1);
        assert_eq!(program.step_count(0), Some(4));
        assert!(program.collective(0, 0).is_none());
        assert!(program.zero_before(0, 0).is_none());
    }

    // ── 2. hash router: old sequencing vs the lowered program ─────────────
    #[test]
    fn ds4_hash_router_old_vs_lowered() {
        // OLD (pre-change) single-GPU hash decode — transcribed from
        // `ffn_hash_routed`: after `moe_route`, the routed ops were
        // [hash_router_normalize (device) | host gather (precomputed),
        // gate_up, silu·clamp, rotate, down_residual_scaled] with the
        // self-combining f32 down accumulating into `ffn_out`. The four
        // routed ops map to three Steps: [GateUp, Activation, DownResidual].
        let cfg = tiny_cfg();
        let policy = single_policy();
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let ffn_out = synth_f32(cfg.hidden_size);
        let phases = ds4_decode_phases(
            &tensors,
            expert_ref(cfg.moe_intermediate_size, cfg.hidden_size),
            Ds4DownTarget::ResidualF32 { out: ffn_out },
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            cfg.hidden_size,
            cfg.swiglu_limit,
            1,
        )
        .unwrap();
        assert_eq!(phases.router.len(), 0);
        assert_eq!(phases.gate_up.len(), 1);
        assert_eq!(phases.activation.len(), 1);
        assert_eq!(phases.down.len(), 1);
        assert_eq!(phases.combine.len(), 0);
        assert_eq!(phases.finish.len(), 0);
        assert!(matches!(
            &phases.down[0],
            Step::IndexedMoeGemv {
                which: MoeProj::DownResidual { .. },
                out,
                ..
            } if std::ptr::eq(*out, ffn_out)
        ));

        // Device-path hash routing: RouterPlan::Hash + the authoritative
        // layer-1 plan (canonical declared identity "hash").
        let router_dev = ds4_router_plan(
            Ds4RouteSelection::Hash,
            synth_f32(cfg.n_routed_experts),
            None,
            Some(synth_f32(cfg.n_routed_experts)),
            Some(synth_f32(1)),
            tensors.topk_indices,
            tensors.topk_weights,
            cfg.num_experts_per_tok,
            2.2,
        )
        .unwrap();
        let rank_phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, cfg.hidden_size),
                Ds4DownTarget::ResidualF32 { out: ffn_out },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let w = test_weights(&cfg);
        let program = production_single_program(&cfg, &w, 1, router_dev, rank_phases());
        use hipfire_runtime::moe_plan::MoeExecutorKind;
        assert_eq!(program.executor_kind(), MoeExecutorKind::SingleMesh);
        assert_eq!(program.step_count(0), Some(3));
        assert!(program.collective(0, 0).is_none());

        // The host-completed fallback side (declared identity "precomputed"
        // + RouterPlan::Precomputed, and its no-alias rejections) is covered
        // by `ds4_borrowed_plan_host_fallback_precomputed_identity` through
        // the model-owned cache/effective-spec path — the canonical plan
        // below is "hash", so a Precomputed router plan must be rejected
        // (a "hash" declaration is NOT an alias for "precomputed").
        let host_router = || {
            ds4_router_plan(
                Ds4RouteSelection::PrecomputedHost,
                synth_f32(cfg.n_routed_experts),
                None,
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap()
        };
        let resolution = ds4_resolve_expert_plans(&cfg, &policy).unwrap();
        let err = ds4_lower_borrowed_plan(
            &resolution.plans[1],
            &policy,
            host_router(),
            vec![rank_phases()],
        )
        .unwrap_err();
        assert!(
            matches!(err, MoeLowerError::RouterIdentityMismatch { .. }),
            "{err}"
        );
    }

    // ── 3. shared-only / disabled MoE behavior preserved ──────────────────
    #[test]
    fn ds4_shared_only_and_disabled_moe_preserved() {
        // Transcribed from the pre-change pre-down gating (`ds4_bias_pre_down`
        // / `ds4_hash_pre_down`): MoE disabled or expert blobs absent → the
        // layer is shared-only (no routed program); a hash layer without its
        // tid2eid table is shared-only; a hash layer with the table on device
        // routes via Hash; without it, routing is host-completed → Precomputed.
        let cfg = tiny_cfg();

        // MoE disabled (HIPFIRE_DEEPSEEK4_MOE=0) — no routed program.
        assert_eq!(
            ds4_route_kind(&cfg, 3, false, true, false, false),
            Ds4RouteSelection::SharedOnly
        );
        // Expert blobs absent — no routed program.
        assert_eq!(
            ds4_route_kind(&cfg, 3, true, false, false, false),
            Ds4RouteSelection::SharedOnly
        );
        // Hash layer without the tid2eid table — shared expert alone.
        assert_eq!(
            ds4_route_kind(&cfg, 0, true, true, false, false),
            Ds4RouteSelection::SharedOnly
        );
        // Hash layer, table on device → Hash.
        assert_eq!(
            ds4_route_kind(&cfg, 0, true, true, true, true),
            Ds4RouteSelection::Hash
        );
        // Hash layer, host-completed → PrecomputedHost (never a Hash alias).
        assert_eq!(
            ds4_route_kind(&cfg, 0, true, true, true, false),
            Ds4RouteSelection::PrecomputedHost
        );
        // Score layer (l >= num_hash_layers) → BiasAware.
        assert_eq!(
            ds4_route_kind(&cfg, 3, true, true, true, true),
            Ds4RouteSelection::BiasAware
        );

        // The canonical manifest identities are exactly the typed selections:
        // bias_aware_topk / hash / precomputed — no aliases. The route-selection
        // mapping is the single source (the effective-spec path selects the
        // identity from the resident profile, never a hash alias).
        assert_eq!(ROUTER_BIAS_AWARE, "bias_aware_topk");
        assert_eq!(ROUTER_HASH, "hash");
        assert_eq!(ROUTER_PRECOMPUTED, "precomputed");
        assert_eq!(
            Ds4RouterProfile::HashDevice.to_route_selection(),
            Ds4RouteSelection::Hash
        );
        assert_eq!(
            Ds4RouterProfile::PrecomputedHost.to_route_selection(),
            Ds4RouteSelection::PrecomputedHost
        );

        // A SharedOnly layer never declares a lowering identity: the declared
        // manifest for the layer keeps the canonical hash identity while the
        // runtime lowering path returns SharedOnly before any program exists.
        // With num_nextn_predict_layers == 1 the manifest additionally
        // declares one bias-aware MTP group at layer num_hidden_layers.
        let policy = single_policy();
        let specs = ds4_expert_group_manifest(&cfg, &policy);
        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].router_identity, "hash");
        assert_eq!(specs[1].router_identity, "hash");
        assert_eq!(specs[2].router_identity, "bias_aware_topk");
        assert_eq!(specs[3].router_identity, "bias_aware_topk");
        assert_eq!(specs[4].router_identity, "bias_aware_topk");
        assert_eq!(specs[4].layer, Some(cfg.num_hidden_layers));
    }

    // ── 4. prefill program shape preserved (batched TpI64, rows=batch·k) ──
    #[test]
    fn ds4_prefill_program_shape_preserved() {
        // OLD (pre-change) batched-TP prefill schedule — transcribed from
        // `ds4_prefill_moe_step_tp`: over the 4-op program [GateUp,
        // Activation, DownResidualI64 (batched, n tokens), ConvertI64ToF32]
        // the sequencing hardcoded collectives
        //   [None, None, AllReduceI64Tp { dim: n*hidden }, None]
        // and zero_before [false, false, true, false] — the i64 accumulator
        // is pre-zeroed before the batched down, then one partition-invariant
        // int64 all-reduce over the Tp group.
        //
        // PRODUCTION SHAPE: the authoritative plan comes from the shared
        // resolution under the caller's Tp policy (tp_cfg: inter=512 so the
        // TP projection gate passes; hidden=256 so the sealed batched
        // protocol's 256-alignment gates pass), and the activation/gate-up
        // rows are the checked `batch * num_experts_per_tok` product
        // (6 = 3·2) — the sealed batched TpI64 protocol rejects anything else.
        let mut cfg = tp_cfg();
        cfg.hidden_size = 256;
        let policy = tp_policy(2);
        let n = 3usize;
        let hidden = cfg.hidden_size;
        let tensors = decode_tensors_batched(
            n,
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            hidden,
        );
        let partial_i64 = synth_i64_2d(n, hidden);
        let partial = synth_f32(n * hidden);
        let phases = ds4_decode_phases(
            &tensors,
            expert_ref_batched(cfg.moe_intermediate_size, hidden),
            Ds4DownTarget::I64 {
                partial_i64,
                partial,
            },
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
            hidden,
            cfg.swiglu_limit,
            n,
        )
        .unwrap();
        assert_eq!(phases.gate_up.len(), 1);
        assert_eq!(phases.activation.len(), 1);
        assert_eq!(phases.down.len(), 1);
        assert_eq!(phases.combine.len(), 0);
        assert_eq!(phases.finish.len(), 1);
        // The activation step must span exactly batch·k rows (the batched
        // protocol's activation-rows check).
        assert!(matches!(
            &phases.activation[0],
            Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { .. },
                gate,
                up,
                rot_out,
                inter,
                k_top,
            } if std::ptr::eq(*gate, tensors.gate_batch)
                && std::ptr::eq(*up, tensors.up_batch)
                && std::ptr::eq(*rot_out, tensors.rot_batch)
                && *inter == cfg.moe_intermediate_size
                && *k_top == n * cfg.num_experts_per_tok
        ));
        let router = ds4_router_plan(
            Ds4RouteSelection::BiasAware,
            synth_f32(cfg.n_routed_experts),
            Some(synth_f32(cfg.n_routed_experts)),
            None,
            None,
            tensors.topk_indices,
            tensors.topk_weights,
            cfg.num_experts_per_tok,
            2.2,
        )
        .unwrap();
        let rank_phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref_batched(cfg.moe_intermediate_size, hidden),
                Ds4DownTarget::I64 {
                    partial_i64,
                    partial,
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                hidden,
                cfg.swiglu_limit,
                n,
            )
            .unwrap()
        };
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let entry = ds4_cached_moe_plans(&ranks, &cfg, &policy).expect("tp2 aggregate resolves");
        let plan = entry.plan(3).expect("cached layer-3 plan");
        let program = crate::forward::build_ds4_parallel_program(
            plan,
            &policy,
            router,
            vec![rank_phases(), rank_phases()],
        )
        .expect("sealed batched TpI64 program");
        // Approved inspection: Parallel executor, two ranks, exact 4-step
        // program, exact typed collectives + zero-before (no Debug parsing).
        use hipfire_runtime::moe_plan::MoeExecutorKind;
        assert_eq!(program.executor_kind(), MoeExecutorKind::Parallel);
        assert_eq!(program.rank_count(), 2);
        let expected_dim = n * hidden;
        for rank in 0..2 {
            assert_eq!(program.step_count(rank), Some(4), "rank {rank}");
            assert!(collective_eq(
                program.collective(rank, 0).unwrap(),
                &hipfire_dispatch::pipeline::StepCollective::None
            ));
            assert!(collective_eq(
                program.collective(rank, 1).unwrap(),
                &hipfire_dispatch::pipeline::StepCollective::None
            ));
            assert!(collective_eq(
                program.collective(rank, 2).unwrap(),
                &hipfire_dispatch::pipeline::StepCollective::AllReduceI64Tp { dim: expected_dim }
            ));
            assert!(collective_eq(
                program.collective(rank, 3).unwrap(),
                &hipfire_dispatch::pipeline::StepCollective::None
            ));
            assert_eq!(program.zero_before(rank, 0), Some(false), "rank {rank}");
            assert_eq!(program.zero_before(rank, 1), Some(false), "rank {rank}");
            assert_eq!(program.zero_before(rank, 2), Some(true), "rank {rank}");
            assert_eq!(program.zero_before(rank, 3), Some(false), "rank {rank}");
        }
    }

    // ── 5. MTP reuses the lowered program ─────────────────────────────────
    #[test]
    fn ds4_mtp_reuses_lowered_program() {
        // The MTP head is a SEPARATE layer at index == num_hidden_layers
        // whose MoE runs through the SAME sealed-lowering path as the main
        // layers (mtp_forward_ep / mtp_forward_tp call the same
        // `ds4_ep_moe_step`-shaped program).
        let cfg = tiny_cfg();
        let mtp_layer = cfg.num_hidden_layers;
        assert!(mtp_layer >= cfg.num_hash_layers); // MTP layer is score-routed
        let policy = ep_policy(2);
        // The authoritative MTP plan comes from the shared resolution (the
        // same seam `mtp_forward_ep` / `mtp_forward_tp` consume).
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let entry = ds4_cached_moe_plans(&ranks, &cfg, &policy).expect("ep2 aggregate resolves");
        let group = entry.mtp_plan().expect("cached MTP plan");
        assert_eq!(group.layer, Some(mtp_layer));
        assert_eq!(group.parallelism, ExpertParallelism::ExpertParallel);
        assert_eq!(
            group.collective,
            Some(ExpertPostCombineAllReduce::ExpertParallel)
        );

        let hidden = cfg.hidden_size;
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let partial_i64 = synth_i64(hidden);
        let partial = synth_f32(hidden);
        let rank_phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, hidden),
                Ds4DownTarget::I64 {
                    partial_i64,
                    partial,
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                hidden,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let router = ds4_router_plan(
            Ds4RouteSelection::BiasAware,
            synth_f32(cfg.n_routed_experts),
            Some(synth_f32(cfg.n_routed_experts)),
            None,
            None,
            tensors.topk_indices,
            tensors.topk_weights,
            cfg.num_experts_per_tok,
            2.2,
        )
        .unwrap();
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let entry = ds4_cached_moe_plans(&ranks, &cfg, &policy).expect("ep2 aggregate resolves");
        let group = entry.mtp_plan().expect("cached MTP plan");
        let program = crate::forward::build_ds4_parallel_program(
            group,
            &policy,
            router,
            vec![rank_phases(), rank_phases()],
        )
        .expect("sealed EpLocalI64 program");
        // Approved inspection: the SAME EpLocalI64 protocol as the main
        // layers — ZeroI64Only pre-zeros the local i64 accumulator, the FP32
        // EP all-reduce lands on the ConvertI64ToF32 destination.
        use hipfire_dispatch::pipeline::StepCollective as SC;
        use hipfire_runtime::moe_plan::MoeExecutorKind;
        use hipfire_runtime::multi_gpu::DimKind;
        assert_eq!(program.executor_kind(), MoeExecutorKind::Parallel);
        assert_eq!(program.rank_count(), 2);
        for rank in 0..2 {
            assert_eq!(program.step_count(rank), Some(4), "rank {rank}");
            assert!(collective_eq(
                program.collective(rank, 2).unwrap(),
                &SC::ZeroI64Only { dim: hidden }
            ));
            assert!(collective_eq(
                program.collective(rank, 3).unwrap(),
                &SC::AllReduce {
                    kind: DimKind::Ep,
                    dim: hidden
                }
            ));
            assert_eq!(program.zero_before(rank, 2), Some(true), "rank {rank}");
            assert_eq!(program.zero_before(rank, 3), Some(false), "rank {rank}");
        }
    }

    // ── 6. routed add precedes the HC mix ─────────────────────────────────
    #[test]
    fn ds4_routed_add_precedes_hc_mix() {
        // The routed contribution must be fully assembled BEFORE the arch-side
        // HC FFN mix runs. In the lowered program the routed partial is
        // completed by the LAST step: the EP i64 program ends with
        // ConvertI64ToF32 carrying the AllReduce{Ep} collective, and the DS4
        // production tail (`ffn_out += partial`, then `hc_ffn_mix`) runs only
        // after `execute_lowered_moe` returns.
        let cfg = tiny_cfg();
        let policy = ep_policy(2);
        let hidden = cfg.hidden_size;
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let partial_i64 = synth_i64(hidden);
        let partial = synth_f32(hidden);
        let phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, hidden),
                Ds4DownTarget::I64 {
                    partial_i64,
                    partial,
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                hidden,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let router = ds4_router_plan(
            Ds4RouteSelection::BiasAware,
            synth_f32(cfg.n_routed_experts),
            Some(synth_f32(cfg.n_routed_experts)),
            None,
            None,
            tensors.topk_indices,
            tensors.topk_weights,
            cfg.num_experts_per_tok,
            2.2,
        )
        .unwrap();
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let entry = ds4_cached_moe_plans(&ranks, &cfg, &policy).expect("ep2 aggregate resolves");
        let program = crate::forward::build_ds4_parallel_program(
            entry.plan(3).expect("cached layer-3 plan"),
            &policy,
            router,
            vec![phases(), phases()],
        )
        .expect("sealed EpLocalI64 program");
        // The collective (routed-sum assembly) is on the program's LAST step —
        // the finish phase (step 3) carries AllReduce{Ep} and its zero_before
        // is false (the conversion writes its destination fresh). Nothing
        // follows the routed assembly inside the program; the production tail
        // then folds the routed partial BEFORE the HC mix (the action
        // sequence in test 41 is the same seam the mesh step dispatches).
        use hipfire_dispatch::pipeline::StepCollective as SC;
        use hipfire_runtime::multi_gpu::DimKind;
        for rank in 0..2 {
            assert!(collective_eq(
                program.collective(rank, 3).unwrap(),
                &SC::AllReduce {
                    kind: DimKind::Ep,
                    dim: hidden
                }
            ));
            assert_eq!(program.zero_before(rank, 3), Some(false), "rank {rank}");
            assert_eq!(program.zero_before(rank, 2), Some(true), "rank {rank}");
        }
        // The production tail-order sequence (same helper the mesh MoE step
        // dispatches): routed add precedes the HC mix.
        use crate::forward::{ds4_tail_actions, Ds4TailAction as T};
        assert_eq!(
            ds4_tail_actions(&crate::moe_lower::Ds4RouteSelection::BiasAware),
            &[T::AddRouted, T::HcMix]
        );
    }

    // ── 7. the five multi-device entries validate the caller's policy ────
    #[test]
    fn ds4_forward_entries_require_caller_policy() {
        // The entries EXECUTE the production policy-validation logic (the
        // same seam `validate_mesh_entry_policy` calls at their start,
        // before any GPU work): each entry's required kind must accept its
        // own kind and refuse the other kind. A correct policy for the
        // entry's kind passes; the wrong kind refuses.
        use crate::forward::validate_mesh_policy_binding;
        use hipfire_runtime::multi_gpu::DeviceMesh as Dm;
        use hipfire_runtime::multi_gpu::DimKind as Dk;

        // Entries and the kind they require: EP decode, TP decode, EP MTP,
        // TP MTP, TP batched prefill (kind pairs exercised both directions).
        let cases: &[(MoEExecutionKind, MoEExecutionKind)] = &[
            (MoEExecutionKind::Ep, MoEExecutionKind::Tp),
            (MoEExecutionKind::Tp, MoEExecutionKind::Ep),
        ];
        for (entry_kind, other_kind) in cases {
            let axis = match entry_kind {
                MoEExecutionKind::Ep => Dk::Ep,
                MoEExecutionKind::Tp => Dk::Tp,
                MoEExecutionKind::Single => unreachable!(),
            };
            let mesh = Dm::rect(&[(axis, 2)]);
            let policy = MoEExecutionPolicy::new(*entry_kind, mesh.clone()).unwrap();
            // The entry's own kind + the exact bound mesh/epoch passes.
            validate_mesh_policy_binding(&policy, *entry_kind, Some(mesh.epoch()), 2)
                .expect("correct kind + bound mesh must pass");
            // The other kind refuses (kind is part of the entry contract).
            let err = validate_mesh_policy_binding(&policy, *other_kind, Some(mesh.epoch()), 2)
                .unwrap_err();
            assert!(
                err.contains("expected a") && err.contains("execution policy"),
                "{err}"
            );
            // Stale/different epoch refuses.
            let stale = Dm::rect(&[(Dk::Tp, 2)]);
            assert!(
                validate_mesh_policy_binding(&policy, *entry_kind, Some(stale.epoch()), 2).is_err()
            );
            // Device-count mismatch refuses.
            assert!(
                validate_mesh_policy_binding(&policy, *entry_kind, Some(mesh.epoch()), 1).is_err()
            );
        }
        // A correct rank-one NAMED mesh passes (rank-one Tp/Ep support).
        let mesh1 = Dm::rect(&[(Dk::Tp, 1)]);
        let policy1 = MoEExecutionPolicy::new(MoEExecutionKind::Tp, mesh1.clone()).unwrap();
        validate_mesh_policy_binding(&policy1, MoEExecutionKind::Tp, Some(mesh1.epoch()), 1)
            .expect("rank-one named mesh must pass");
        // An unbound Gpus (no from_mesh epoch) refuses — the binding is
        // required even for shared-only / MoE-disabled runs.
        let err =
            validate_mesh_policy_binding(&policy1, MoEExecutionKind::Tp, None, 1).unwrap_err();
        assert!(err.contains("not bound"), "{err}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Lane A (final cutover) — PackedSeparate authority + MTP declaration
    // ═══════════════════════════════════════════════════════════════════

    // ── 8. PackedSeparate claims ALL three logical expert sources ────────
    #[test]
    fn ds4_spec_declares_packed_separate_sources() {
        let cfg = tiny_cfg();
        let policy = single_policy();
        let spec = ds4_expert_group_spec(&cfg, &policy, 3, ROUTER_BIAS_AWARE);
        assert_eq!(spec.router, "router_gate");
        assert!(
            matches!(
                &spec.source_layout,
                ExpertSourceLayout::PackedSeparate {
                    gate,
                    up,
                    down,
                    sidecars,
                } if gate == "experts_gate"
                    && up == "experts_up"
                    && down == "experts_down"
                    && sidecars.is_empty()
            ),
            "source_layout must claim gate/up/down separately: {:?}",
            spec.source_layout
        );
        assert_eq!(
            spec.allowed_executions,
            vec![ExpertExecutionIdentity::IndexedQuantized]
        );
    }

    // ── 8a. PackedSeparate bytes_per_expert counts exact F16 bytes ───────
    #[test]
    fn ds4_spec_declares_exact_f16_bytes_per_expert() {
        let cfg = tiny_cfg(); // moe_intermediate_size = 48, hidden_size = 64
        let policy = single_policy();
        let spec = ds4_expert_group_spec(&cfg, &policy, 3, ROUTER_BIAS_AWARE);
        assert_eq!(
            spec.resources.bytes_per_expert,
            3 * cfg.moe_intermediate_size * cfg.hidden_size * 2,
            "PackedSeparate must count F16 bytes (3·im·hidden·2), not elements"
        );
        // Exact value for the tiny fixture: 3 · 48 · 64 · 2 = 18432.
        assert_eq!(spec.resources.bytes_per_expert, 18432);
        // Every declared group (main + MTP) carries the same exact footprint.
        let specs = ds4_expert_group_manifest(&cfg, &policy);
        assert!(specs.iter().all(|s| s.resources.bytes_per_expert == 18432));
    }

    // ── 9. MTP group declaration: count==1 appends one bias-aware group ──
    #[test]
    fn ds4_mtp_group_manifest_count1_appends_bias_aware_group() {
        let cfg = tiny_cfg(); // num_nextn_predict_layers == 1
        let policy = single_policy();
        let specs = ds4_expert_group_manifest(&cfg, &policy);
        assert_eq!(specs.len(), cfg.num_hidden_layers + 1);
        let mtp = &specs[cfg.num_hidden_layers];
        assert_eq!(mtp.layer, Some(cfg.num_hidden_layers));
        assert_eq!(mtp.router_identity, ROUTER_BIAS_AWARE);
        assert_eq!(mtp.router, "router_gate");
        assert_eq!(
            mtp.allowed_executions,
            vec![ExpertExecutionIdentity::IndexedQuantized]
        );
        assert!(
            matches!(
                &mtp.source_layout,
                ExpertSourceLayout::PackedSeparate { .. }
            ),
            "MTP group must use the same PackedSeparate authority: {:?}",
            mtp.source_layout
        );
        assert_eq!(mtp.n_experts, cfg.n_routed_experts);
    }

    // ── 10. MTP count > 1 refused before resolution ──────────────────────
    #[test]
    fn ds4_mtp_resolution_count2_refused_before_resolution() {
        let cfg = cfg_count2();
        let policy = single_policy();
        let err = ds4_resolve_expert_plans(&cfg, &policy).unwrap_err();
        assert!(
            err.contains("num_nextn_predict_layers") && err.contains("unsupported"),
            "{err}"
        );
        // The model-owned cache path reports the same refusal as a typed
        // error before any resolution attempt (aggregate, one-element).
        let weights = test_weights(&cfg);
        let err = ds4_cached_moe_plans(std::slice::from_ref(&weights), &cfg, &policy).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::MtpCountUnsupported { count: 2 }),
            "{err:?}"
        );
        assert!(weights.moe_plan_cache.get().is_none());
    }

    // ── 11. effective placements per policy; NO residual expert collectives
    #[test]
    fn ds4_resolution_placements_and_no_residual_expert_collectives() {
        use hipfire_runtime::weight_manifest::ExpertParallelism as P;
        use hipfire_runtime::weight_manifest::ExpertPostCombineAllReduce as C;
        let cases: Vec<(DeepseekV4Config, MoEExecutionPolicy, P, Option<C>)> = vec![
            (tiny_cfg(), single_policy(), P::Single, None),
            (
                tp_cfg(),
                tp_policy(1),
                P::TensorParallel,
                Some(C::TensorParallel),
            ),
            (
                tp_cfg(),
                tp_policy(2),
                P::TensorParallel,
                Some(C::TensorParallel),
            ),
            (
                tiny_cfg(),
                ep_policy(1),
                P::ExpertParallel,
                Some(C::ExpertParallel),
            ),
            (
                tiny_cfg(),
                ep_policy(2),
                P::ExpertParallel,
                Some(C::ExpertParallel),
            ),
        ];
        for (cfg, policy, parallelism, collective) in cases {
            let resolution = ds4_resolve_expert_plans(&cfg, &policy)
                .unwrap_or_else(|e| panic!("{parallelism:?} resolution failed: {e}"));
            assert!(
                resolution.layer_collectives.is_empty(),
                "{parallelism:?}: all three expert sources (gate/up/down) are claimed by the \
                 PackedSeparate groups; the projected residual schedule must be empty, got {:?}",
                resolution.layer_collectives
            );
            assert_eq!(resolution.plans.len(), cfg.num_hidden_layers + 1);
            let ranks = policy.rank_count();
            for plan in &resolution.plans[..cfg.num_hidden_layers] {
                assert_eq!(plan.parallelism, parallelism);
                assert_eq!(plan.group_size, ranks);
                assert_eq!(plan.collective, collective);
                match parallelism {
                    P::Single => {
                        assert_eq!(plan.experts.len(), cfg.n_routed_experts);
                        for (i, e) in plan.experts.iter().enumerate() {
                            assert_eq!((e.global_id, e.owner, e.local_slot), (i, 0, i));
                        }
                    }
                    P::TensorParallel => {
                        assert_eq!(plan.experts.len(), cfg.n_routed_experts * ranks);
                        for g in 0..cfg.n_routed_experts {
                            for owner in 0..ranks {
                                let e = plan.experts[g * ranks + owner];
                                assert_eq!((e.global_id, e.owner, e.local_slot), (g, owner, g));
                            }
                        }
                    }
                    P::ExpertParallel => {
                        assert_eq!(plan.experts.len(), cfg.n_routed_experts);
                        for (i, e) in plan.experts.iter().enumerate() {
                            assert_eq!(e.global_id, i);
                            assert_eq!(e.owner, i % ranks);
                            assert_eq!(e.local_slot, i / ranks);
                        }
                    }
                }
            }
        }
    }

    // ── 12. plans are indexed main layers then the optional MTP slot ─────
    #[test]
    fn ds4_resolution_plans_indexed_main_then_mtp() {
        let cfg = tiny_cfg();
        let resolution = ds4_resolve_expert_plans(&cfg, &single_policy()).unwrap();
        assert_eq!(resolution.plans.len(), cfg.num_hidden_layers + 1);
        for (i, plan) in resolution.plans.iter().enumerate() {
            assert_eq!(
                plan.layer,
                Some(i),
                "plan[{i}] must carry its own layer index"
            );
        }
        assert_eq!(
            resolution.plans[cfg.num_hidden_layers].group,
            format!("deepseek4.moe.l{}", cfg.num_hidden_layers)
        );
        assert_eq!(
            resolution.plans[cfg.num_hidden_layers].router_identity,
            ROUTER_BIAS_AWARE
        );
    }

    // ── 13. resolution leaves the static manifest + specs untouched ──────
    #[test]
    fn ds4_static_manifest_and_specs_immutable_after_resolution() {
        use hipfire_runtime::arch::Architecture;
        let cfg = tiny_cfg();
        let policy = single_policy();
        let specs = ds4_expert_group_manifest(&cfg, &policy);
        let manifest = crate::arch::DeepseekV4::weight_manifest(&cfg);
        let specs_before = specs.clone();
        let manifest_before = manifest.clone();
        ds4_resolve_expert_plans(&cfg, &policy).unwrap();
        assert_eq!(
            specs, specs_before,
            "specs must not be mutated by resolution"
        );
        assert_eq!(
            manifest, manifest_before,
            "static manifest must not be mutated by resolution"
        );
    }

    // ── 14. resident router profiles derive from actual LUT/router residency
    #[test]
    fn ds4_resident_router_profiles_from_actual_residency() {
        let cfg = tiny_cfg(); // hash layers 0,1; score layers 2,3; MTP=1
        let weights = test_weights(&cfg);
        // Main layers in order, then the MTP slot at num_hidden_layers (the
        // fixture loads a fully routeable MTP layer → BiasAware).
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &weights).unwrap(),
            vec![
                Ds4RouterProfile::HashDevice,
                Ds4RouterProfile::HashDevice,
                Ds4RouterProfile::BiasAware,
                Ds4RouterProfile::BiasAware,
                Ds4RouterProfile::BiasAware,
            ]
        );
        // Host-only LUT → PrecomputedHost (never a Hash alias).
        let mut w2 = test_weights(&cfg);
        w2.layers[0].tid2eid_dev = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w2).unwrap()[0],
            Ds4RouterProfile::PrecomputedHost
        );
        // Hash layer with a MISSING/wrong-length LUT → typed error (the LUT
        // must cover exactly vocab_size × num_experts_per_tok).
        let mut w3 = test_weights(&cfg);
        w3.layers[0].tid2eid_host.clear();
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w3),
            Err(Ds4BundleError::HashLutLength {
                layer: 0,
                got: 0,
                expected: cfg.vocab_size * cfg.num_experts_per_tok,
            })
        );
        // Hash layer WITHOUT the gate weight → typed error (the runtime
        // computes hash-routing gate scores from it).
        let mut w3b = test_weights(&cfg);
        w3b.layers[0].gate_weight = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w3b),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 0,
                projection: "router_gate",
            })
        );
        // PARTIAL score-routed router data → typed errors (each of the three
        // route-selection inputs is load-bearing).
        let mut w4 = test_weights(&cfg);
        w4.layers[2].gate_weight = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w4),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "router_gate",
            })
        );
        let mut w5 = test_weights(&cfg);
        w5.layers[2].gate_bias = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w5),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "router_gate",
            })
        );
        let mut w6 = test_weights(&cfg);
        w6.layers[2].gate_bias_host.clear();
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w6),
            Err(Ds4BundleError::RouterBiasLength {
                layer: 2,
                got: 0,
                expected: cfg.n_routed_experts,
            })
        );
        // Blob WITHOUT its paired pointer table → typed error (never a
        // silent shared-only classification).
        let mut w8 = test_weights(&cfg);
        w8.layers[2].expert_gate_up_ptrs = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w8),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "gate_up",
            })
        );
        // Pointer table with insufficient F32 capacity → typed error
        // (required: 2 × n_routed_experts slots).
        let mut w9 = test_weights(&cfg);
        w9.layers[2].expert_w2_ptrs = Some(owned_f32(2 * cfg.n_routed_experts - 1));
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w9),
            Err(Ds4BundleError::ExpertPointerCapacity {
                layer: 2,
                projection: "down",
                capacity: 2 * cfg.n_routed_experts - 1,
                required: 2 * cfg.n_routed_experts,
            })
        );
        // A missing MTP layer on a count-1 config → MTP slot Unavailable.
        let mut w7 = test_weights(&cfg);
        w7.mtp_layer = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w7).unwrap()[cfg.num_hidden_layers],
            Ds4RouterProfile::Unavailable
        );
        // A count-0 config derives no MTP slot.
        let cfg0 = cfg_count0();
        let w0 = test_weights(&cfg0);
        assert_eq!(
            ds4_resident_router_profiles(&cfg0, &w0).unwrap().len(),
            cfg0.num_hidden_layers
        );
    }

    #[test]
    fn ds4_router_profile_maps_to_route_selection() {
        use crate::moe_lower::Ds4RouteSelection as R;
        assert_eq!(
            Ds4RouterProfile::Unavailable.to_route_selection(),
            R::SharedOnly
        );
        assert_eq!(
            Ds4RouterProfile::BiasAware.to_route_selection(),
            R::BiasAware
        );
        assert_eq!(Ds4RouterProfile::HashDevice.to_route_selection(), R::Hash);
        assert_eq!(
            Ds4RouterProfile::PrecomputedHost.to_route_selection(),
            R::PrecomputedHost
        );
    }

    // ── 15. model-owned cache: same key borrows forever ──────────────────
    #[test]
    fn ds4_cache_same_key_borrows_forever() {
        let cfg = tiny_cfg();
        let weights = test_weights(&cfg);
        let policy = weights.moe_policy.clone(); // canonical stable Single policy
        let one = std::slice::from_ref(&weights);
        let e1 = ds4_cached_moe_plans(one, &cfg, &policy).unwrap();
        let e2 = ds4_cached_moe_plans(one, &cfg, &policy).unwrap();
        assert!(
            std::ptr::eq(e1, e2),
            "same key must return the same cached entry"
        );
        assert!(e1.resolution().is_ok());
        assert!(e2.resolution().is_ok());
        // The same policy instance cloned (same mesh epoch) still matches.
        let policy_clone = policy.clone();
        let e3 = ds4_cached_moe_plans(one, &cfg, &policy_clone).unwrap();
        assert!(std::ptr::eq(e1, e3));
    }

    // ── 16. model-owned cache: failure cached and reused with same key ───
    #[test]
    fn ds4_cache_failure_cached_and_reused_with_same_key() {
        // tiny_cfg's inter=48 fails the TP projection role-dimension gate
        // (local slice 24 % 256 != 0) — a deterministic resolution failure
        // that is cached and replayed, never retried.
        let cfg = tiny_cfg();
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let policy = tp_policy(2);
        let e1 = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap();
        let err1 = e1.resolution().unwrap_err();
        assert!(err1.contains("256"), "{err1}");
        let e2 = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap();
        assert!(
            std::ptr::eq(e1, e2),
            "cached failure must be reused, not retried"
        );
        assert_eq!(e1.resolution().unwrap_err(), e2.resolution().unwrap_err());
    }

    // ── 17. model-owned cache: different key → explicit mismatch, no retry
    #[test]
    fn ds4_cache_key_mismatch_refused_no_replacement() {
        // (a) mesh-epoch / policy mismatch on a TP key (aggregate).
        let cfg = tp_cfg();
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let p1 = tp_policy(2);
        ds4_cached_moe_plans(&ranks, &cfg, &p1).unwrap();
        // Fresh policy with the SAME topology but a new mesh epoch → mismatch.
        let p2 = tp_policy(2);
        let err = ds4_cached_moe_plans(&ranks, &cfg, &p2).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );
        // A Single policy is also a mismatch (kind differs).
        let err = ds4_cached_moe_plans(std::slice::from_ref(&ranks[0]), &cfg, &single_policy())
            .unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );

        // (b) manifest config identity mismatch on a Single key.
        let cfg_s = tiny_cfg();
        let ws = test_weights(&cfg_s);
        let one = std::slice::from_ref(&ws);
        let policy_s = single_policy();
        ds4_cached_moe_plans(one, &cfg_s, &policy_s).unwrap();
        let mut cfg2 = cfg_s.clone();
        cfg2.hidden_size = 128;
        let err = ds4_cached_moe_plans(one, &cfg2, &policy_s).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );

        // (c) per-rank residency change on the same policy/config is a
        //     complete-key mismatch (cell seam: the aggregate derives the
        //     matrix from the weights, so only the cell can replay a changed
        //     matrix). The MTP slot stays BiasAware — an Unavailable MTP is
        //     a refusal, covered by the MTP refusal test.
        let ps = ds4_resident_router_profiles(&cfg_s, &ws).unwrap();
        let mut ps2 = ps.clone();
        ps2[0] = Ds4RouterProfile::PrecomputedHost;
        let err =
            ds4_cache_cell_moe_plans(&ws.moe_plan_cache, &cfg_s, &policy_s, &[ps2]).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );

        // The original cache entry survives every mismatch: same key still
        // borrows the first result and nothing was replaced.
        let again = ds4_cached_moe_plans(one, &cfg_s, &policy_s).unwrap();
        assert!(again.resolution().is_ok());
    }

    // ── 18. rank router profiles must agree before resolution ────────────
    #[test]
    fn ds4_cache_rank_router_profile_disagreement_refused() {
        // Aggregate-level: rank 1's ACTUAL residency differs from rank 0's —
        // the derived matrix disagrees and resolution is refused.
        let cfg = tiny_cfg();
        let w0 = test_weights(&cfg);
        let mut w1 = test_weights(&cfg);
        w1.layers[1].tid2eid_dev = None; // rank 1 hash layer host-completed
        let ranks = [w0, w1];
        let policy = ep_policy(2);
        let err = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap_err();
        assert!(
            matches!(
                err,
                Ds4PlanCacheError::RankRouterProfileDisagreement {
                    first_disagreeing_rank: 1
                }
            ),
            "{err:?}"
        );
        // The cache was never seeded by the refused calls.
        assert!(ranks[0].moe_plan_cache.get().is_none());
        assert!(ranks[1].moe_plan_cache.get().is_none());
    }

    // ── 18a. the aggregate must cover exactly the policy's ranks ─────────
    #[test]
    fn ds4_aggregate_rank_count_mismatch_refused() {
        let cfg = tiny_cfg();
        let w0 = test_weights(&cfg);
        // Single policy (rank count 1) with a two-element aggregate → mismatch.
        let ranks2 = [test_weights(&cfg), test_weights(&cfg)];
        let err = ds4_cached_moe_plans(&ranks2, &cfg, &single_policy()).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::RankCountMismatch {
                expected: 1,
                got: 2
            }
        );
        // EP2 policy (rank count 2) with a one-element aggregate → mismatch.
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w0), &cfg, &ep_policy(2)).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::RankCountMismatch {
                expected: 2,
                got: 1
            }
        );
        // Empty aggregate → EmptyRankProfiles.
        let empty: [DeepseekV4Weights; 0] = [];
        let err = ds4_cached_moe_plans(&empty, &cfg, &single_policy()).unwrap_err();
        assert_eq!(err, Ds4PlanCacheError::EmptyRankProfiles);
        // The cache was never seeded by the refused calls.
        assert!(w0.moe_plan_cache.get().is_none());
    }

    #[test]
    fn ds4_cache_rank_layer_count_mismatch_refused() {
        // Cell seam: a misshapen per-rank vector (truncated or padded) is
        // refused before resolution — the matrix shape is binding.
        let cfg = tiny_cfg(); // 4 main layers + 1 MTP slot = 5 expected
        let weights = test_weights(&cfg);
        let profiles = ds4_resident_router_profiles(&cfg, &weights).unwrap();
        assert_eq!(profiles.len(), cfg.num_hidden_layers + 1);
        let truncated = profiles[..cfg.num_hidden_layers].to_vec();
        let err = ds4_cache_cell_moe_plans(
            &weights.moe_plan_cache,
            &cfg,
            &single_policy(),
            &[truncated],
        )
        .unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::RankLayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers + 1,
                got: cfg.num_hidden_layers,
            }
        );
        // An over-long vector is refused the same way.
        let mut padded = profiles.clone();
        padded.push(Ds4RouterProfile::BiasAware);
        let err =
            ds4_cache_cell_moe_plans(&weights.moe_plan_cache, &cfg, &single_policy(), &[padded])
                .unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::RankLayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers + 1,
                got: cfg.num_hidden_layers + 2,
            }
        );
        // A truncated MATRIX (missing rank vectors) is refused too.
        let err = ds4_cache_cell_moe_plans(
            &weights.moe_plan_cache,
            &cfg,
            &ep_policy(2),
            &[profiles.clone()],
        )
        .unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::RankCountMismatch {
                expected: 2,
                got: 1
            }
        );
        // The cache was never seeded by the refused calls.
        assert!(weights.moe_plan_cache.get().is_none());
    }

    // ── 18b. rank 0 owns/serves the cache; nonzero cells stay empty ──────
    #[test]
    fn ds4_aggregate_rank0_owns_cache() {
        let cfg = tiny_cfg();
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let policy = ep_policy(2);
        let e1 = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap();
        // Rank 0's cell owns the resolution; the nonzero-rank cell is empty.
        let owner = ranks[0]
            .moe_plan_cache
            .get()
            .expect("rank-0 cell must be seeded");
        assert!(std::ptr::eq(e1, owner));
        assert!(
            ranks[1].moe_plan_cache.get().is_none(),
            "nonzero-rank cells must stay empty forever"
        );
        // Repeated calls still serve from rank 0 only.
        let e2 = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap();
        assert!(std::ptr::eq(e1, e2));
        assert!(ranks[1].moe_plan_cache.get().is_none());
        // Single uses a one-element aggregate and owner 0.
        let s = test_weights(&cfg);
        let e3 = ds4_cached_moe_plans(std::slice::from_ref(&s), &cfg, &single_policy()).unwrap();
        assert!(std::ptr::eq(
            e3,
            s.moe_plan_cache.get().expect("single owner-0 cell seeded")
        ));
    }

    // ── 18c. configured-absent / partial MTP refuses BEFORE resolution ───
    #[test]
    fn ds4_cache_mtp_absent_refused_before_resolution() {
        let cfg = tiny_cfg(); // num_nextn_predict_layers == 1
        let policy = single_policy();
        // (a) count1 with NO MTP layer loaded at all.
        let mut absent = test_weights(&cfg);
        absent.mtp_layer = None;
        let err = ds4_cached_moe_plans(std::slice::from_ref(&absent), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::MtpNotRouteable {
                rank: 0,
                profile: Ds4RouterProfile::Unavailable
            }
        );
        assert!(
            absent.moe_plan_cache.get().is_none(),
            "a refused MTP call must never seed the cache"
        );
        // (b) partial MTP: blobs but no router gate weight → bundle error.
        let mut partial1 = test_weights(&cfg);
        partial1.mtp_layer.as_mut().unwrap().gate_weight = None;
        let err = ds4_cached_moe_plans(std::slice::from_ref(&partial1), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::PartialExpertBundle {
                rank: 0,
                layer: cfg.num_hidden_layers,
                projection: "router_gate",
            }
        );
        assert!(partial1.moe_plan_cache.get().is_none());
        // (c) partial MTP: no gate bias → bundle error.
        let mut partial2 = test_weights(&cfg);
        partial2.mtp_layer.as_mut().unwrap().gate_bias = None;
        let err = ds4_cached_moe_plans(std::slice::from_ref(&partial2), &cfg, &policy).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::PartialExpertBundle { .. }),
            "{err:?}"
        );
        // (d) partial MTP: wrong-length host-cached gate_bias twin → bundle error.
        let mut partial3 = test_weights(&cfg);
        partial3.mtp_layer.as_mut().unwrap().gate_bias_host.clear();
        let err = ds4_cached_moe_plans(std::slice::from_ref(&partial3), &cfg, &policy).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::RouterBiasLength { .. }),
            "{err:?}"
        );
        // (e) EVERY rank's MTP slot must be routeable: rank 1's absent MTP is
        //     refused with the disagreeing rank reported.
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let mut ranks_bad = [test_weights(&cfg), test_weights(&cfg)];
        ranks_bad[1].mtp_layer = None;
        let err = ds4_cached_moe_plans(&ranks_bad, &cfg, &ep_policy(2)).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::MtpNotRouteable {
                rank: 1,
                profile: Ds4RouterProfile::Unavailable
            }
        );
        assert!(ranks_bad[0].moe_plan_cache.get().is_none());
        // (f) count0 has no MTP slot and no MTP requirement.
        let cfg0 = cfg_count0();
        let w0 = test_weights(&cfg0);
        let e = ds4_cached_moe_plans(std::slice::from_ref(&w0), &cfg0, &single_policy()).unwrap();
        assert!(e.resolution().is_ok());
        // (g) a routeable aggregate seeds; the refused absent-MTP matrix
        //     stays refused and never disturbs the seeded entry.
        let good = test_weights(&cfg);
        let e1 = ds4_cached_moe_plans(std::slice::from_ref(&good), &cfg, &policy).unwrap();
        let bad_matrix = vec![ds4_resident_router_profiles(&cfg, &absent).unwrap()];
        let err =
            ds4_cache_cell_moe_plans(&good.moe_plan_cache, &cfg, &policy, &bad_matrix).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::MtpNotRouteable {
                rank: 0,
                profile: Ds4RouterProfile::Unavailable
            }
        );
        let e2 = ds4_cached_moe_plans(std::slice::from_ref(&good), &cfg, &policy).unwrap();
        assert!(std::ptr::eq(e1, e2) && e2.resolution().is_ok());
        // The two-rank routeable aggregate also seeds fine (e).
        assert!(ds4_cached_moe_plans(&ranks, &cfg, &ep_policy(2)).is_ok());
    }

    // ── 19. missing MTP plan is an explicit error, never a panic ─────────
    #[test]
    fn ds4_mtp_plan_accessor_explicit_error_never_panics() {
        // count0: main execution allowed, MTP plan explicitly unconfigured.
        let cfg0 = cfg_count0();
        let w0 = test_weights(&cfg0);
        let e = ds4_cached_moe_plans(std::slice::from_ref(&w0), &cfg0, &single_policy()).unwrap();
        assert!(e.main_plan(0).is_some());
        assert!(e.main_plan(cfg0.num_hidden_layers).is_none());
        assert!(matches!(e.mtp_plan(), Err(Ds4MtpPlanError::Unconfigured)));
        assert!(matches!(e.plan(cfg0.num_hidden_layers), None));

        // count1 success: the MTP plan is the last plan at layer N.
        let cfg1 = tiny_cfg();
        let w1 = test_weights(&cfg1);
        let e = ds4_cached_moe_plans(std::slice::from_ref(&w1), &cfg1, &single_policy()).unwrap();
        assert_eq!(e.mtp_plan().unwrap().layer, Some(cfg1.num_hidden_layers));
        assert!(std::ptr::eq(
            e.mtp_plan().unwrap(),
            e.plan(cfg1.num_hidden_layers).unwrap()
        ));

        // count1 with a cached resolution failure: explicit ResolutionFailed.
        let w2 = test_weights(&cfg1);
        let ranks = [w2, test_weights(&cfg1)];
        let e = ds4_cached_moe_plans(&ranks, &cfg1, &tp_policy(2)).unwrap();
        assert!(matches!(
            e.mtp_plan(),
            Err(Ds4MtpPlanError::ResolutionFailed(_))
        ));
        assert!(e.main_plan(0).is_none());
    }

    // ── 21. exactly-once resolution under concurrent first calls ─────────
    #[test]
    fn ds4_cache_concurrent_same_key_resolves_exactly_once() {
        const THREADS: usize = 8;
        let cfg = tiny_cfg();
        let policy = single_policy();
        let profiles = ds4_resident_router_profiles(&cfg, &test_weights(&cfg)).unwrap();
        let matrix = vec![profiles.clone()];
        let lock: std::sync::OnceLock<Ds4PlanCacheEntry> = std::sync::OnceLock::new();
        let _seam = ds4_resolve_seam::SeamGuard::on();
        let barrier = std::sync::Barrier::new(THREADS);
        let results: Vec<Result<usize, Ds4PlanCacheError>> = std::thread::scope(|s| {
            (0..THREADS)
                .map(|_| {
                    s.spawn(|| {
                        ds4_resolve_seam::arm();
                        barrier.wait();
                        ds4_cache_cell_moe_plans(&lock, &cfg, &policy, &matrix)
                            .map(|entry| entry as *const Ds4PlanCacheEntry as usize)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("worker panicked"))
                .collect()
        });
        // `get_or_init` ran the initializer exactly once: losing callers'
        // initializers never ran, so no discarded second resolution.
        assert_eq!(
            ds4_resolve_seam::count(),
            1,
            "concurrent same-key first calls must resolve the manifest exactly once"
        );
        // Every caller borrowed the SAME stored entry (address-identical).
        let winner = lock.get().expect("cache seeded") as *const Ds4PlanCacheEntry as usize;
        for result in &results {
            let addr = result
                .as_ref()
                .expect("same-key concurrent call must succeed");
            assert_eq!(
                *addr, winner,
                "every caller must borrow the one cached entry"
            );
        }
    }

    #[test]
    fn ds4_cache_concurrent_different_keys_resolve_once_and_refuse_mismatch() {
        const PER_KEY: usize = 4;
        let cfg = tiny_cfg();
        let profiles = ds4_resident_router_profiles(&cfg, &test_weights(&cfg)).unwrap();
        let lock: std::sync::OnceLock<Ds4PlanCacheEntry> = std::sync::OnceLock::new();
        let _seam = ds4_resolve_seam::SeamGuard::on();
        // Key A: Single policy (one-element matrix); key B: EP2 policy
        // (two-element matrix). Both are routeable; the keys differ in
        // policy kind AND matrix shape.
        let policy_a = single_policy();
        let policy_b = ep_policy(2);
        let matrix_a = vec![profiles.clone()];
        let matrix_b = vec![profiles.clone(), profiles.clone()];
        let key_a = Ds4PlanCacheKey {
            policy: policy_a.clone(),
            manifest_config: Ds4ManifestConfigIdentity::of(&cfg),
            router_profiles: matrix_a.clone(),
        };
        let key_b = Ds4PlanCacheKey {
            policy: policy_b.clone(),
            manifest_config: Ds4ManifestConfigIdentity::of(&cfg),
            router_profiles: matrix_b.clone(),
        };
        let barrier = std::sync::Barrier::new(2 * PER_KEY);
        let results: Vec<Result<usize, Ds4PlanCacheError>> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            let lock_ref = &lock;
            let cfg_ref = &cfg;
            let barrier_ref = &barrier;
            for _ in 0..PER_KEY {
                let matrix = matrix_a.clone();
                let policy_ref = &policy_a;
                handles.push(s.spawn(move || {
                    ds4_resolve_seam::arm();
                    barrier_ref.wait();
                    ds4_cache_cell_moe_plans(lock_ref, cfg_ref, policy_ref, &matrix)
                        .map(|entry| entry as *const Ds4PlanCacheEntry as usize)
                }));
            }
            for _ in 0..PER_KEY {
                let matrix = matrix_b.clone();
                let policy_ref = &policy_b;
                handles.push(s.spawn(move || {
                    ds4_resolve_seam::arm();
                    barrier_ref.wait();
                    ds4_cache_cell_moe_plans(lock_ref, cfg_ref, policy_ref, &matrix)
                        .map(|entry| entry as *const Ds4PlanCacheEntry as usize)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("worker panicked"))
                .collect()
        });
        // Exactly one resolution: whichever key won the `get_or_init` race,
        // the losing key's initializer never ran.
        assert_eq!(
            ds4_resolve_seam::count(),
            1,
            "concurrent different-key first calls must still resolve exactly once"
        );
        // Exactly PER_KEY successes and PER_KEY explicit KeyMismatchs.
        let successes = results.iter().filter(|r| r.is_ok()).count();
        let mismatches = results
            .iter()
            .filter(|r| matches!(r, Err(Ds4PlanCacheError::KeyMismatch { .. })))
            .count();
        assert_eq!(successes, PER_KEY, "exactly the winner-key callers succeed");
        assert_eq!(
            mismatches, PER_KEY,
            "exactly the loser-key callers get an explicit KeyMismatch"
        );
        // Successes correspond to the stored winner key, mismatchs to the
        // loser key, by caller index.
        let winner_key = lock
            .get()
            .expect("one of the two keys won the race")
            .key()
            .clone();
        let winner = lock.get().unwrap() as *const Ds4PlanCacheEntry as usize;
        for (i, result) in results.iter().enumerate() {
            let expected_key = if i < PER_KEY { &key_a } else { &key_b };
            match result {
                Ok(addr) => {
                    assert_eq!(*addr, winner, "winner-key callers share the entry");
                    assert!(
                        winner_key.first_mismatch(expected_key).is_none(),
                        "successful caller {i} must carry the stored winner key"
                    );
                }
                Err(Ds4PlanCacheError::KeyMismatch { .. }) => {
                    assert!(
                        winner_key.first_mismatch(expected_key).is_some(),
                        "mismatched caller {i} must carry the loser key"
                    );
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
    }

    // ── 21a. a concurrent FAILING key is also resolved exactly once ──────
    #[test]
    fn ds4_cache_concurrent_failing_key_resolves_exactly_once() {
        const THREADS: usize = 8;
        let cfg = tiny_cfg();
        let policy = tp_policy(2); // inter=48 fails the TP projection gate
        let profiles = ds4_resident_router_profiles(&cfg, &test_weights(&cfg)).unwrap();
        let matrix = vec![profiles.clone(), profiles.clone()];
        let lock: std::sync::OnceLock<Ds4PlanCacheEntry> = std::sync::OnceLock::new();
        let _seam = ds4_resolve_seam::SeamGuard::on();
        let barrier = std::sync::Barrier::new(THREADS);
        let results: Vec<Result<usize, Ds4PlanCacheError>> = std::thread::scope(|s| {
            let lock_ref = &lock;
            let cfg_ref = &cfg;
            let policy_ref = &policy;
            let barrier_ref = &barrier;
            (0..THREADS)
                .map(|_| {
                    let matrix = matrix.clone();
                    s.spawn(move || {
                        ds4_resolve_seam::arm();
                        barrier_ref.wait();
                        ds4_cache_cell_moe_plans(lock_ref, cfg_ref, policy_ref, &matrix)
                            .map(|entry| entry as *const Ds4PlanCacheEntry as usize)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("worker panicked"))
                .collect()
        });
        // The failure is cached exactly once and replayed to every caller.
        assert_eq!(
            ds4_resolve_seam::count(),
            1,
            "concurrent failing-key first calls must resolve exactly once"
        );
        let winner = lock.get().expect("failure entry cached") as *const Ds4PlanCacheEntry as usize;
        for result in &results {
            let addr = result
                .as_ref()
                .expect("same failing key must borrow the cached failure");
            assert_eq!(*addr, winner, "every caller borrows the one failure entry");
        }
        let err = lock.get().unwrap().resolution().unwrap_err();
        assert!(err.contains("256"), "{err}");
    }

    // ── 22. the key stores the COMPLETE per-rank matrix ───────────────────
    #[test]
    fn ds4_cache_key_stores_full_matrix() {
        let cfg = tiny_cfg();
        let weights = test_weights(&cfg);
        let profiles = ds4_resident_router_profiles(&cfg, &weights).unwrap();
        // EP2 seed: the stored key must carry BOTH rank vectors verbatim.
        let policy = ep_policy(2);
        let matrix = vec![profiles.clone(), profiles.clone()];
        let entry = ds4_cache_cell_moe_plans(&weights.moe_plan_cache, &cfg, &policy, &matrix)
            .expect("EP2 routeable seed succeeds");
        assert_eq!(
            entry.key().router_profiles,
            matrix,
            "the key must store the complete validated matrix, not one rank's column"
        );
        // A matrix change on ANY rank is a different key (agreement keeps
        // the matrix uniform, so a changed vector is a changed key).
        let mut changed = profiles;
        changed[0] = Ds4RouterProfile::PrecomputedHost;
        let changed_matrix = vec![changed.clone(), changed];
        let err = ds4_cache_cell_moe_plans(&weights.moe_plan_cache, &cfg, &policy, &changed_matrix)
            .unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "a changed matrix on the same cell must be an explicit mismatch: {err:?}"
        );
        // A Single seed stores its one-element matrix.
        let single = test_weights(&cfg);
        let s_matrix = vec![ds4_resident_router_profiles(&cfg, &single).unwrap()];
        let s_entry =
            ds4_cache_cell_moe_plans(&single.moe_plan_cache, &cfg, &single_policy(), &s_matrix)
                .expect("Single routeable seed succeeds");
        assert_eq!(s_entry.key().router_profiles, s_matrix);
    }

    // ── 23. authority → borrowed-plan lowering: residency-aware identities ─
    // A host-fallback layer must resolve with the `precomputed` semantic
    // identity so `RouterPlan::Precomputed` lowers WITHOUT a
    // `RouterIdentityMismatch`; device-hash and bias-aware layers retain
    // their exact identities. The canonical/static manifest is never mutated.
    #[test]
    fn ds4_borrowed_plan_host_fallback_precomputed_identity() {
        let cfg = tiny_cfg();
        let mut w = test_weights(&cfg);
        w.layers[0].tid2eid_dev = None; // host-completed hash fallback
        let policy = single_policy();
        let entry = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap();
        let plan = entry
            .plan(0)
            .expect("layer 0 plan from the model-owned cache");
        assert_eq!(
            plan.router_identity, ROUTER_PRECOMPUTED,
            "a host-fallback layer must resolve with the `precomputed` semantic identity"
        );
        // Lowering with RouterPlan::Precomputed → Ok (no RouterIdentityMismatch).
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let ffn_out = synth_f32(cfg.hidden_size);
        let phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, cfg.hidden_size),
                Ds4DownTarget::ResidualF32 { out: ffn_out },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let precomputed_router = || {
            ds4_router_plan(
                Ds4RouteSelection::PrecomputedHost,
                synth_f32(cfg.n_routed_experts),
                None,
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap()
        };
        let program =
            ds4_lower_borrowed_plan(plan, &policy, precomputed_router(), vec![phases()]).unwrap();
        // Approved inspection: the host-fallback program is a Single program
        // with exactly the three routed steps (GateUp, Activation,
        // DownResidual).
        assert_eq!(program.step_count(0), Some(3));
        assert!(program.collective(0, 0).is_none());
        // The SAME plan lowered with RouterPlan::Hash → explicit mismatch
        // (the effective identity is really `precomputed`, never a hash alias).
        let hash_router = || {
            ds4_router_plan(
                Ds4RouteSelection::Hash,
                synth_f32(cfg.n_routed_experts),
                None,
                Some(synth_f32(cfg.n_routed_experts)),
                Some(synth_f32(1)),
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap()
        };
        let err =
            ds4_lower_borrowed_plan(plan, &policy, hash_router(), vec![phases()]).unwrap_err();
        assert!(
            matches!(err, MoeLowerError::RouterIdentityMismatch { .. }),
            "{err}"
        );
        // Device-hash layer keeps the `hash` identity and lowers with
        // RouterPlan::Hash; bias-aware layer keeps `bias_aware_topk` and
        // lowers with RouterPlan::BiasAware.
        let entry2 = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap();
        assert_eq!(entry2.plan(1).unwrap().router_identity, ROUTER_HASH);
        let program = ds4_lower_borrowed_plan(
            entry2.plan(1).unwrap(),
            &policy,
            hash_router(),
            vec![phases()],
        )
        .unwrap();
        // Approved inspection: the device-hash Single program has exactly the
        // three routed steps (GateUp, Activation, DownResidual) and no
        // parallel schedule.
        assert_eq!(program.step_count(0), Some(3));
        assert!(program.collective(0, 0).is_none());
        assert_eq!(entry2.plan(3).unwrap().router_identity, ROUTER_BIAS_AWARE);
        let bias_router = || {
            ds4_router_plan(
                Ds4RouteSelection::BiasAware,
                synth_f32(cfg.n_routed_experts),
                Some(synth_f32(cfg.n_routed_experts)),
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap()
        };
        // Bias-aware phases use the expanded per-expert down + combine
        // (4 steps: GateUp, Activation, DownExpanded, Combine).
        let down_expanded = synth_f32(cfg.num_experts_per_tok * cfg.hidden_size);
        let bias_phases = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, cfg.hidden_size),
                Ds4DownTarget::ExpandedF32 {
                    down_expanded,
                    out: ffn_out,
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let program = ds4_lower_borrowed_plan(
            entry2.plan(3).unwrap(),
            &policy,
            bias_router(),
            vec![bias_phases()],
        )
        .unwrap();
        // Approved inspection: the bias-aware program has the exact four
        // routed steps (GateUp, Activation, DownExpanded, Combine).
        assert_eq!(program.step_count(0), Some(4));
        assert!(program.collective(0, 0).is_none());
    }

    // ── 24. partial / misshapen bundles refuse at the aggregate ───────────
    #[test]
    fn ds4_cache_partial_bundle_refused() {
        let cfg = tiny_cfg();
        let policy = single_policy();
        // One missing resource at a time — every case must be a typed error
        // BEFORE any resolution and must never seed the cache.
        let cases: Vec<(
            &str,
            Box<dyn Fn(&mut crate::deepseek4::DeepseekV4Weights) + '_>,
        )> = vec![
            (
                "gate_up blob w/o pointer table",
                Box::new(|w| w.layers[2].expert_gate_up_ptrs = None),
            ),
            (
                "down blob w/o pointer table",
                Box::new(|w| w.layers[2].expert_w2_ptrs = None),
            ),
            (
                "pointer table w/o gate_up blob",
                Box::new(|w| w.layers[2].expert_gate_up_blob = None),
            ),
            (
                "down pointer table w/o blob",
                Box::new(|w| w.layers[2].expert_w2_blob = None),
            ),
            (
                "gate_up pointer table ONLY (no blobs, no down)",
                Box::new(|w| {
                    w.layers[2].expert_gate_up_blob = None;
                    w.layers[2].expert_w2_blob = None;
                    w.layers[2].expert_w2_ptrs = None;
                }),
            ),
            (
                "down pointer table ONLY (no blobs, no gate_up)",
                Box::new(|w| {
                    w.layers[2].expert_gate_up_blob = None;
                    w.layers[2].expert_gate_up_ptrs = None;
                    w.layers[2].expert_w2_blob = None;
                }),
            ),
            (
                "BOTH pointer tables, no blobs at all",
                Box::new(|w| {
                    w.layers[2].expert_gate_up_blob = None;
                    w.layers[2].expert_w2_blob = None;
                }),
            ),
            (
                "gate_up ptr capacity short",
                Box::new(|w| {
                    w.layers[2].expert_gate_up_ptrs = Some(owned_f32(2 * cfg.n_routed_experts - 1))
                }),
            ),
            (
                "down ptr capacity short",
                Box::new(|w| {
                    w.layers[2].expert_w2_ptrs = Some(owned_f32(2 * cfg.n_routed_experts - 1))
                }),
            ),
            (
                "hash LUT wrong length",
                Box::new(|w| {
                    w.layers[0].tid2eid_host =
                        vec![0u32; cfg.vocab_size * cfg.num_experts_per_tok - 1]
                }),
            ),
            (
                "hash layer w/o gate weight",
                Box::new(|w| w.layers[0].gate_weight = None),
            ),
            (
                "router gate bias wrong length",
                Box::new(|w| w.layers[2].gate_bias_host = vec![0.0; cfg.n_routed_experts - 1]),
            ),
            (
                "score layer w/o gate weight",
                Box::new(|w| w.layers[2].gate_weight = None),
            ),
            (
                "score layer w/o gate bias",
                Box::new(|w| w.layers[2].gate_bias = None),
            ),
        ];
        for (name, mutate) in cases {
            let mut w = test_weights(&cfg);
            mutate(&mut w);
            let err = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap_err();
            assert!(
                matches!(
                    err,
                    Ds4PlanCacheError::PartialExpertBundle { .. }
                        | Ds4PlanCacheError::ExpertPointerCapacity { .. }
                        | Ds4PlanCacheError::HashLutLength { .. }
                        | Ds4PlanCacheError::RouterBiasLength { .. }
                ),
                "{name}: unexpected error {err:?}"
            );
            assert!(
                w.moe_plan_cache.get().is_none(),
                "{name}: a refused bundle must never seed the cache"
            );
        }
    }

    // ── 25. main-layer cardinality refuses BEFORE any indexing ────────────
    #[test]
    fn ds4_cache_layer_cardinality_refused() {
        let cfg = tiny_cfg(); // num_hidden_layers = 4
        let policy = single_policy();
        // Too few layers: resolve_layer would be out of range — must be a
        // typed error, never a panic, never a seed.
        let mut few = test_weights(&cfg);
        few.layers.truncate(cfg.num_hidden_layers - 1);
        let err = ds4_cached_moe_plans(std::slice::from_ref(&few), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::LayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers,
                got: cfg.num_hidden_layers - 1,
            }
        );
        assert!(few.moe_plan_cache.get().is_none());
        // Too many layers.
        let mut many = test_weights(&cfg);
        many.layers
            .push(crate::deepseek4::DeepseekV4LayerWeights::new_empty(0));
        let err = ds4_cached_moe_plans(std::slice::from_ref(&many), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::LayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers,
                got: cfg.num_hidden_layers + 1,
            }
        );
        assert!(many.moe_plan_cache.get().is_none());
        // Per-rank: rank 1 misshapen is reported with ITS rank.
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let mut ranks_bad = [test_weights(&cfg), test_weights(&cfg)];
        ranks_bad[1].layers.truncate(2);
        let err = ds4_cached_moe_plans(&ranks_bad, &cfg, &ep_policy(2)).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::LayerCountMismatch {
                rank: 1,
                expected: cfg.num_hidden_layers,
                got: 2,
            }
        );
        assert!(ranks[0].moe_plan_cache.get().is_none());
    }

    // ── 26. one full certification per aggregate acquisition ─────────────
    #[test]
    fn ds4_cache_full_certification_per_acquisition() {
        let cfg = tiny_cfg();
        let w = test_weights(&cfg);
        let policy = w.moe_policy.clone();
        let _seam = ds4_resolve_seam::SeamGuard::on();
        // Seed: exactly one resolution, one profile derivation, one key.
        let e1 = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap();
        assert_eq!(
            ds4_resolve_seam::count(),
            1,
            "exactly-once resolution at seed"
        );
        assert_eq!(ds4_resolve_seam::profiles(), 1, "one derivation at seed");
        assert_eq!(ds4_resolve_seam::keys(), 1, "one owned key at seed");
        // Every SAME-KEY acquisition re-certifies (derivation + key) — there
        // is no policy/config fast path — but never re-resolves and always
        // borrows the IDENTICAL entry.
        for _ in 0..5 {
            let e = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap();
            assert!(std::ptr::eq(e1, e), "same-key acquisitions share the entry");
        }
        assert_eq!(
            ds4_resolve_seam::count(),
            1,
            "same-key acquisitions must never re-resolve (exactly-once preserved)"
        );
        assert_eq!(
            ds4_resolve_seam::profiles(),
            6,
            "one full profile derivation per aggregate acquisition"
        );
        assert_eq!(
            ds4_resolve_seam::keys(),
            6,
            "one owned key construction per aggregate acquisition"
        );
        // A policy mismatch derives + builds the exact key for the mismatch
        // detail and still never re-resolves.
        let fresh = single_policy();
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &fresh).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );
        assert_eq!(ds4_resolve_seam::profiles(), 7, "mismatch derives once");
        assert_eq!(ds4_resolve_seam::keys(), 7, "mismatch builds one key");
        assert_eq!(ds4_resolve_seam::count(), 1, "mismatch never re-resolves");
    }

    // ── 27. canonical stable Single policy ────────────────────────────────
    #[test]
    fn ds4_canonical_single_policy_stable() {
        let cfg = tiny_cfg();
        let w = test_weights(&cfg);
        // The model-owned policy is ONE stable object: clones compare equal
        // (same mesh epoch) and a fresh `MoEExecutionPolicy::single()` is a
        // DIFFERENT exact policy (fresh epoch).
        assert_eq!(w.moe_policy, w.moe_policy.clone());
        assert_ne!(w.moe_policy, MoEExecutionPolicy::single());
        // Same canonical policy every call → same cached entry.
        let e1 = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &w.moe_policy).unwrap();
        let e2 = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &w.moe_policy).unwrap();
        assert!(std::ptr::eq(e1, e2));
        // A fresh per-call policy is refused as an exact-key mismatch — the
        // caller MUST use the canonical object (never per-call reconstruction).
        let err = ds4_cached_moe_plans(
            std::slice::from_ref(&w),
            &cfg,
            &MoEExecutionPolicy::single(),
        )
        .unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "{err:?}"
        );
    }

    // ── 28. unsupported MTP count refuses at the aggregate boundary ───────
    #[test]
    fn ds4_mtp_count_refused_before_empty_rank_checks() {
        let cfg = cfg_count2(); // num_nextn_predict_layers = 2
                                // Combined-invalid: count2 + EMPTY aggregate → MTP refusal first.
        let empty: [DeepseekV4Weights; 0] = [];
        let err = ds4_cached_moe_plans(&empty, &cfg, &single_policy()).unwrap_err();
        assert_eq!(err, Ds4PlanCacheError::MtpCountUnsupported { count: 2 });
        // Combined-invalid: count2 + wrong rank count → MTP refusal first.
        let one = [test_weights(&cfg)];
        let err = ds4_cached_moe_plans(&one, &cfg, &tp_policy(2)).unwrap_err();
        assert_eq!(err, Ds4PlanCacheError::MtpCountUnsupported { count: 2 });
        // Combined-invalid: count2 + complete weights → MTP refusal first.
        let err = ds4_cached_moe_plans(&one, &cfg, &single_policy()).unwrap_err();
        assert_eq!(err, Ds4PlanCacheError::MtpCountUnsupported { count: 2 });
    }

    // ── 29. authority stays truthful independent of the MoE runtime switch ─
    #[test]
    fn ds4_authority_truthful_without_moe_switch() {
        // There is NO moe_on anywhere in the residency/key/aggregate API:
        // complete weights always classify truthfully and seed — the runtime
        // enable switch is a forward-time concern (Lane B bypasses plan
        // lookup when MoE is disabled), never a residency lie.
        let cfg = tiny_cfg(); // count1
        let w = test_weights(&cfg);
        let profiles = ds4_resident_router_profiles(&cfg, &w).unwrap();
        assert_eq!(
            profiles,
            vec![
                Ds4RouterProfile::HashDevice,
                Ds4RouterProfile::HashDevice,
                Ds4RouterProfile::BiasAware,
                Ds4RouterProfile::BiasAware,
                Ds4RouterProfile::BiasAware, // MTP slot
            ]
        );
        let entry = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &w.moe_policy).unwrap();
        assert!(entry.resolution().is_ok());
        assert_eq!(entry.mtp_plan().unwrap().router_identity, ROUTER_BIAS_AWARE);
    }

    // ── 30. mutations after seed are NEVER stale Ok ──────────────────────
    // The aggregate re-certifies the full residency matrix on EVERY
    // acquisition: changed residency, cardinality, rank disagreement, and
    // partial bundles are caught on the next call — never a stale cached
    // entry.
    #[test]
    fn ds4_cache_changed_residency_not_stale_ok() {
        let cfg = tiny_cfg();
        let mut w = test_weights(&cfg);
        let policy = w.moe_policy.clone();
        let e1 = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap();
        assert_eq!(e1.plan(0).unwrap().router_identity, ROUTER_HASH);
        // (a) Hash → PrecomputedHost profile change on the SAME weights.
        w.layers[0].tid2eid_dev = None;
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy).unwrap_err();
        assert!(
            matches!(err, Ds4PlanCacheError::KeyMismatch { .. }),
            "changed residency must be an explicit key mismatch, never stale Ok: {err:?}"
        );
        // The ORIGINAL entry survives: the cell still holds the first key.
        let stored = w.moe_plan_cache.get().expect("seed entry survives");
        assert_eq!(
            stored.key().router_profiles[0][0],
            Ds4RouterProfile::HashDevice
        );

        // (b) Truncated layers after seed → typed cardinality refusal.
        let mut w2 = test_weights(&cfg);
        ds4_cached_moe_plans(std::slice::from_ref(&w2), &cfg, &policy).unwrap();
        w2.layers.truncate(cfg.num_hidden_layers - 1);
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w2), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::LayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers,
                got: cfg.num_hidden_layers - 1,
            }
        );
        // (c) Extra layers after seed → typed cardinality refusal.
        let mut w3 = test_weights(&cfg);
        ds4_cached_moe_plans(std::slice::from_ref(&w3), &cfg, &policy).unwrap();
        w3.layers
            .push(crate::deepseek4::DeepseekV4LayerWeights::new_empty(0));
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w3), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::LayerCountMismatch {
                rank: 0,
                expected: cfg.num_hidden_layers,
                got: cfg.num_hidden_layers + 1,
            }
        );
        // (d) Partial pointer bundle after seed → typed resource refusal.
        let mut w4 = test_weights(&cfg);
        ds4_cached_moe_plans(std::slice::from_ref(&w4), &cfg, &policy).unwrap();
        w4.layers[2].expert_w2_ptrs = None;
        let err = ds4_cached_moe_plans(std::slice::from_ref(&w4), &cfg, &policy).unwrap_err();
        assert_eq!(
            err,
            Ds4PlanCacheError::PartialExpertBundle {
                rank: 0,
                layer: 2,
                projection: "down",
            }
        );
        // (e) Rank disagreement after seed → typed agreement refusal.
        let cfg2 = tiny_cfg();
        let mut ranks = [test_weights(&cfg2), test_weights(&cfg2)];
        let ep = ep_policy(2);
        let seeded = ds4_cached_moe_plans(&ranks, &cfg2, &ep).unwrap();
        let seeded_ptr = seeded as *const Ds4PlanCacheEntry;
        let seeded_key = seeded.key().clone();
        ranks[1].layers[1].tid2eid_dev = None;
        let err = ds4_cached_moe_plans(&ranks, &cfg2, &ep).unwrap_err();
        assert!(
            matches!(
                err,
                Ds4PlanCacheError::RankRouterProfileDisagreement {
                    first_disagreeing_rank: 1
                }
            ),
            "{err:?}"
        );
        let stored = ranks[0]
            .moe_plan_cache
            .get()
            .expect("the original rank-zero entry remains seeded");
        assert_eq!(stored as *const Ds4PlanCacheEntry, seeded_ptr);
        assert_eq!(stored.key(), &seeded_key);
    }

    // ── 31. RED: pointer-only bundles are partial, not Unavailable ───────
    #[test]
    fn ds4_pointer_only_bundle_refused() {
        let cfg = tiny_cfg();
        // Only the gate-up pointer table survives — no blobs, no down pair.
        let mut w = test_weights(&cfg);
        w.layers[2].expert_gate_up_blob = None;
        w.layers[2].expert_w2_blob = None;
        w.layers[2].expert_w2_ptrs = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "gate_up",
            })
        );
        // Only the down pointer table survives.
        let mut w2 = test_weights(&cfg);
        w2.layers[2].expert_gate_up_blob = None;
        w2.layers[2].expert_gate_up_ptrs = None;
        w2.layers[2].expert_w2_blob = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w2),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "gate_up",
            })
        );
        // Both pointer tables, no blobs at all.
        let mut w3 = test_weights(&cfg);
        w3.layers[2].expert_gate_up_blob = None;
        w3.layers[2].expert_w2_blob = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w3),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 2,
                projection: "gate_up",
            })
        );
    }

    // ── 32. RED: hash layers require the gate weight (score GEMV) ────────
    #[test]
    fn ds4_hash_gate_required() {
        let cfg = tiny_cfg();
        let mut w = test_weights(&cfg);
        w.layers[0].gate_weight = None; // hash layer without gate scores
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 0,
                projection: "router_gate",
            })
        );
        // Host-completed hash fallback requires the gate too.
        let mut w2 = test_weights(&cfg);
        w2.layers[0].tid2eid_dev = None;
        w2.layers[0].gate_weight = None;
        assert_eq!(
            ds4_resident_router_profiles(&cfg, &w2),
            Err(Ds4BundleError::PartialExpertBundle {
                layer: 0,
                projection: "router_gate",
            })
        );
    }

    // ── 33. graph safety: PrecomputedHost refuses BEFORE capture ──────────
    #[test]
    fn ds4_graph_refuse_host_fallback_before_capture() {
        // Production path: acquire the authority for weights whose hash layer
        // has a host-only LUT → the reachable routed profile includes
        // PrecomputedHost → the graph gate refuses BEFORE any capture-capable
        // callback/seam runs.
        let cfg = tiny_cfg();
        let mut w = test_weights(&cfg);
        w.layers[0].tid2eid_dev = None;
        let entry = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &w.moe_policy).unwrap();
        let profiles = &entry.key().router_profiles[0];
        assert!(profiles.contains(&Ds4RouterProfile::PrecomputedHost));
        let err = ds4_graph_refuse_host_fallback(profiles).unwrap_err();
        assert!(err.contains("refused BEFORE capture"), "{err}");
        // HashDevice / BiasAware remain graph-admissible.
        let w2 = test_weights(&cfg);
        let e2 = ds4_cached_moe_plans(std::slice::from_ref(&w2), &cfg, &w2.moe_policy).unwrap();
        ds4_graph_refuse_host_fallback(&e2.key().router_profiles[0]).unwrap();
        // A fallback ANYWHERE in the reachable profile refuses.
        let mut mixed = e2.key().router_profiles[0].clone();
        mixed[1] = Ds4RouterProfile::PrecomputedHost;
        assert!(ds4_graph_refuse_host_fallback(&mixed).is_err());
    }

    // ── 36. MTP selection state machine: exact plan / refusals / disabled ─
    #[test]
    fn ds4_mtp_authority_before_pre_ffn() {
        use crate::forward::{select_mtp_authority_single, Ds4MtpSelection};
        // count1 + complete weights + enabled → Selected(Some) with the typed
        // MTP plan at EXACTLY layer N (num_hidden_layers), before any GPU work.
        let cfg = tiny_cfg();
        let w = test_weights(&cfg);
        let sel =
            select_mtp_authority_single(&cfg, &w, true).expect("count1 must select the MTP plan");
        match sel {
            Ds4MtpSelection::Selected(Some(plan)) => {
                assert_eq!(plan.layer, Some(cfg.num_hidden_layers));
                assert_eq!(plan.router_identity, ROUTER_BIAS_AWARE);
            }
            other => panic!("expected Selected(Some(plan)), got {other:?}"),
        }
        // Disabled count1 → Selected(None): PreFfn may run (shared/pre-FFN)
        // safely; the routed guard never fires.
        let sel = select_mtp_authority_single(&cfg, &w, false)
            .expect("disabled count1 must select Selected(None)");
        assert_eq!(sel, Ds4MtpSelection::Selected(None));
        assert_eq!(
            sel.plan_or_err("test")
                .expect("Selected(None) passes PreFfn"),
            None
        );
        // count0 → explicit Unconfigured refusal BEFORE any moe_on bypass:
        // enabled AND disabled count0 both refuse, so a disabled count0 never
        // reaches PreFfn/weight validation.
        let cfg0 = cfg_count0();
        let w0 = test_weights(&cfg0);
        let err = select_mtp_authority_single(&cfg0, &w0, true).unwrap_err();
        assert!(err.contains("Unconfigured"), "{err}");
        let err = select_mtp_authority_single(&cfg0, &w0, false).unwrap_err();
        assert!(
            err.contains("Unconfigured"),
            "disabled count0 must still refuse (no disable bypass): {err}"
        );
        // The mesh selector has the same count-state ordering.
        let ranks0 = [test_weights(&cfg0), test_weights(&cfg0)];
        let err = crate::forward::select_mtp_authority_mesh(&ranks0, &cfg0, &ep_policy(2), false)
            .unwrap_err();
        assert!(
            err.contains("Unconfigured"),
            "mesh disabled count0 must refuse: {err}"
        );
        // count>1 → refused EVEN WHEN DISABLED (the count check precedes the
        // MoE-disable bypass).
        let cfg2 = cfg_count2();
        let w2 = test_weights(&cfg2);
        let err = select_mtp_authority_single(&cfg2, &w2, true).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
        let err = select_mtp_authority_single(&cfg2, &w2, false).unwrap_err();
        assert!(
            err.contains("refused before the MoE-disable bypass"),
            "count>1 must refuse even when MoE is disabled: {err}"
        );
        // count1 + missing MTP layer + enabled → refused (MtpNotRouteable).
        let mut w3 = test_weights(&cfg);
        w3.mtp_layer = None;
        let err = select_mtp_authority_single(&cfg, &w3, true).unwrap_err();
        assert!(err.contains("MtpNotRouteable"), "{err}");
        // count1 + enabled + cached resolution failure → typed error
        // (Enabled state with a failed/missing plan must never route). The
        // mesh authority first binds the per-rank load layout to the Tp
        // policy, so the fixture records a matching TP slice load.
        let mut ranks2 = [test_weights(&cfg), test_weights(&cfg)];
        ranks2[0].moe_load_layout = crate::deepseek4::Ds4MoeLoadLayout::Tp { tp: 2, rank: 0 };
        ranks2[1].moe_load_layout = crate::deepseek4::Ds4MoeLoadLayout::Tp { tp: 2, rank: 1 };
        let err = crate::forward::select_mtp_authority_mesh(&ranks2, &cfg, &tp_policy(2), true)
            .unwrap_err();
        assert!(err.contains("ResolutionFailed"), "{err}");
        // Unselected → PreFfn refuses with a TYPED error (never a
        // debug-assert-only check).
        let err = Ds4MtpSelection::<'_>::Unselected
            .plan_or_err("test")
            .unwrap_err();
        assert!(
            err.contains("SelectAuthority must run before PreFfn"),
            "{err}"
        );
    }

    // ── 37. mesh policy binding refuses wrong kind / stale epoch / moe-off ─
    #[test]
    fn ds4_mesh_policy_binding_refusals() {
        use crate::forward::validate_mesh_policy_binding;
        use hipfire_runtime::multi_gpu::{DeviceMesh as Dm, DimKind as Dk};
        // Correct rank-one named mesh passes (rank-one Tp/Ep support).
        let mesh1 = Dm::rect(&[(Dk::Ep, 1)]);
        let p1 = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh1.clone()).unwrap();
        validate_mesh_policy_binding(&p1, MoEExecutionKind::Ep, Some(mesh1.epoch()), 1)
            .expect("rank-one named mesh must pass");
        // Wrong kind refuses.
        let mesh2 = Dm::rect(&[(Dk::Ep, 2)]);
        let p2 = MoEExecutionPolicy::new(MoEExecutionKind::Ep, mesh2.clone()).unwrap();
        let err = validate_mesh_policy_binding(&p2, MoEExecutionKind::Tp, Some(mesh2.epoch()), 2)
            .unwrap_err();
        assert!(err.contains("expected a Tp execution policy"), "{err}");
        // Stale/different epoch refuses.
        let stale = Dm::rect(&[(Dk::Ep, 2)]);
        let err = validate_mesh_policy_binding(&p2, MoEExecutionKind::Ep, Some(stale.epoch()), 2)
            .unwrap_err();
        assert!(err.contains("epoch differs"), "{err}");
        // The validation has NO moe_on concept: it runs at the entry start
        // regardless of the runtime switch — wrong policies refuse even when
        // MoE is disabled (the entries validate before acquisition).
        assert!(
            validate_mesh_policy_binding(&p2, MoEExecutionKind::Tp, Some(mesh2.epoch()), 2)
                .is_err()
        );
    }

    // ── 39. complete graph ACTION seam: eager / refuse / run with counts ──
    #[test]
    fn ds4_graph_action_eager_refuse_run_counts() {
        use crate::forward::{graph_forward_action, GraphAction};
        let cfg = tiny_cfg();
        let mut host = test_weights(&cfg);
        host.layers[0].tid2eid_dev = None; // PrecomputedHost reachable
        let _seam = ds4_resolve_seam::SeamGuard::on();
        // Graph OFF + host fallback → EagerDelegate with ZERO acquisitions:
        // the delegate (`decode_step`) performs its own single acquisition.
        let action = graph_forward_action(&cfg, &host, "gfx1151", Some(false), true);
        assert!(matches!(action, GraphAction::EagerDelegate), "{action:?}");
        assert_eq!(ds4_resolve_seam::profiles(), 0, "eager acquires nothing");
        assert_eq!(ds4_resolve_seam::count(), 0);
        // Graph ON + host fallback → GraphRefuse BEFORE any warmup/capture
        // callback; exactly ONE acquisition (the gate acquired then refused).
        let action = graph_forward_action(&cfg, &host, "gfx1151", Some(true), true);
        match action {
            GraphAction::GraphRefuse(e) => assert!(e.contains("refused BEFORE capture"), "{e}"),
            other => panic!("expected GraphRefuse, got {other:?}"),
        }
        assert_eq!(
            ds4_resolve_seam::profiles(),
            1,
            "refuse acquires exactly once"
        );
        assert_eq!(ds4_resolve_seam::count(), 1);
        // Graph ON + Hash/BiasAware → GraphRun carrying the acquired
        // authority (one acquisition/key/resolution; the same authority
        // feeds warmup AND capture).
        let w = test_weights(&cfg);
        let action = graph_forward_action(&cfg, &w, "gfx1151", Some(true), true);
        match action {
            GraphAction::GraphRun(authority) => assert!(authority.entry().is_some()),
            other => panic!("expected GraphRun, got {other:?}"),
        }
        assert_eq!(
            ds4_resolve_seam::profiles(),
            2,
            "graph run acquires exactly once"
        );
        assert_eq!(ds4_resolve_seam::keys(), 2);
        assert_eq!(ds4_resolve_seam::count(), 2);
    }

    // ── 40. EXACT typed TP1 + EP1 schedule evidence ───────────────────────
    // The sealed program is inspected ONLY through the approved shared
    // `LoweredMoeProgram` inspection API (executor_kind / rank_count /
    // step_count / collective / zero_before) — no family reconstruction,
    // no Debug parsing, no StepCollective derives.
    #[test]
    fn ds4_rank_one_production_authority_and_builder() {
        use crate::forward::build_ds4_parallel_program;
        use hipfire_dispatch::pipeline::StepCollective;
        use hipfire_runtime::moe_plan::MoeExecutorKind;
        use hipfire_runtime::multi_gpu::DimKind;
        // Both named rank-one kinds through the production path: authority
        // via the production aggregate, the REAL cached plan, the production
        // phase builder, and the SAME sealed program builder the mesh MoE
        // step consumes.
        for kind in [MoEExecutionKind::Tp, MoEExecutionKind::Ep] {
            let cfg = tp_cfg();
            let policy = match kind {
                MoEExecutionKind::Tp => tp_policy(1),
                MoEExecutionKind::Ep => ep_policy(1),
                _ => unreachable!(),
            };
            let w = test_weights(&cfg);
            let entry = ds4_cached_moe_plans(std::slice::from_ref(&w), &cfg, &policy)
                .expect("rank-one aggregate resolves");
            let plan = entry.plan(3).expect("real cached layer-3 plan");
            let hidden = cfg.hidden_size;
            let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
            let partial_i64 = synth_i64(hidden);
            let partial = synth_f32(hidden);
            let phases = || {
                ds4_decode_phases(
                    &tensors,
                    expert_ref(cfg.moe_intermediate_size, hidden),
                    Ds4DownTarget::I64 {
                        partial_i64,
                        partial,
                    },
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                    hidden,
                    cfg.swiglu_limit,
                    1,
                )
                .unwrap()
            };
            let router = ds4_router_plan(
                Ds4RouteSelection::BiasAware,
                synth_f32(cfg.n_routed_experts),
                Some(synth_f32(cfg.n_routed_experts)),
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap();
            let program = build_ds4_parallel_program(plan, &policy, router, vec![phases()])
                .expect("sealed parallel program for the rank-one group");
            // Approved executor + rank facts.
            assert_eq!(program.executor_kind(), MoeExecutorKind::Parallel);
            assert_eq!(program.rank_count(), 1);
            // Exact program length (4 steps: GateUp, Activation,
            // DownResidualI64, ConvertI64ToF32 — no extras); invalid ranks
            // yield None.
            assert_eq!(program.step_count(0), Some(4));
            assert_eq!(program.step_count(1), None);
            assert_eq!(program.step_count(usize::MAX), None);
            // Exact typed collective + zero-before vectors per step, via the
            // approved accessors.
            let expected_collectives: Vec<StepCollective> = match kind {
                MoEExecutionKind::Tp => vec![
                    StepCollective::None,
                    StepCollective::None,
                    StepCollective::AllReduceI64Tp { dim: hidden },
                    StepCollective::None,
                ],
                MoEExecutionKind::Ep => vec![
                    StepCollective::None,
                    StepCollective::None,
                    StepCollective::ZeroI64Only { dim: hidden },
                    StepCollective::AllReduce {
                        kind: DimKind::Ep,
                        dim: hidden,
                    },
                ],
                _ => unreachable!(),
            };
            for step in 0..4 {
                let got = program
                    .collective(0, step)
                    .expect("rank-0 step collective present");
                let want = &expected_collectives[step];
                assert!(
                    collective_eq(got, want),
                    "{kind:?} collective[{step}] mismatch: {got:?} vs {want:?}"
                );
                assert_eq!(
                    program.zero_before(0, step),
                    Some(step == 2),
                    "{kind:?} zero_before[{step}]"
                );
            }
            // Out-of-range steps and ranks yield None (never panic).
            assert!(program.collective(0, 4).is_none());
            assert!(program.collective(usize::MAX, 0).is_none());
            assert!(program.zero_before(0, 4).is_none());
            assert!(program.zero_before(usize::MAX, 0).is_none());
            // A wrong router identity on the SAME plan is a TYPED refusal
            // (structured identity check — never a silent fallback).
            let wrong = ds4_router_plan(
                Ds4RouteSelection::PrecomputedHost,
                synth_f32(cfg.n_routed_experts),
                None,
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap();
            let err = build_ds4_parallel_program(plan, &policy, wrong, vec![phases()]).unwrap_err();
            assert!(
                matches!(err, MoeLowerError::RouterIdentityMismatch { .. }),
                "{err}"
            );
        }
    }

    // ── 41. production tail ACTION sequence ───────────────────────────────
    #[test]
    fn ds4_tail_action_sequence() {
        use crate::forward::{ds4_tail_actions, Ds4TailAction as T};
        use crate::moe_lower::Ds4RouteSelection as R;
        // Every routed selection folds the partial BEFORE the HC mix; a
        // shared-only layer runs the mix alone. The mesh MoE step's Phase 3
        // dispatches EXACTLY this sequence — moving the mix before the add
        // would change both the sequence and the production behavior.
        assert_eq!(ds4_tail_actions(&R::BiasAware), &[T::AddRouted, T::HcMix]);
        assert_eq!(ds4_tail_actions(&R::Hash), &[T::AddRouted, T::HcMix]);
        assert_eq!(
            ds4_tail_actions(&R::PrecomputedHost),
            &[T::AddRouted, T::HcMix]
        );
        assert_eq!(ds4_tail_actions(&R::SharedOnly), &[T::HcMix]);
    }

    // ── 42. MTP public entries dispatch [SelectAuthority, PreFfn] ─────────
    #[test]
    fn ds4_mtp_entry_action_sequence() {
        use crate::forward::{ds4_mtp_entry_actions, Ds4MtpEntryAction as A};
        // The EXACT initial sequence EVERY MTP public entry (Single, EP, TP)
        // iterates: the authority + typed `entry.mtp_plan()` selection
        // happens exactly once BEFORE any pre-FFN GPU work — PreFfn cannot
        // run until a plan is present. Reversing the sequence would move
        // GPU work before the authority refusals (count0/count>1/failure).
        assert_eq!(ds4_mtp_entry_actions(), &[A::SelectAuthority, A::PreFfn]);
        assert_ne!(ds4_mtp_entry_actions(), &[A::PreFfn, A::SelectAuthority]);
    }

    // ── 43. deterministic MoE-disable snapshot (explicit bool, no env) ────
    #[test]
    fn ds4_deterministic_moe_off_snapshot() {
        use crate::forward::{graph_forward_action, select_mtp_authority_single, GraphAction};
        use crate::moe_lower::Ds4RouteSelection as R;
        let cfg = tiny_cfg();
        let w = test_weights(&cfg);
        // 1. The disabled snapshot yields a Disabled authority through the
        //    production graph seam (moe_on=false passed explicitly — no
        //    process-env mutation): routed enablement derives to false, so
        //    the batched pre-down's `do_routed` cannot fire (no routed
        //    kernels, no panic).
        let action = graph_forward_action(&cfg, &w, "gfx1151", Some(true), false);
        match action {
            GraphAction::GraphRun(authority) => {
                assert!(
                    authority.entry().is_none(),
                    "disabled snapshot must yield a Disabled authority"
                );
            }
            other => panic!("expected GraphRun(Disabled), got {other:?}"),
        }
        // 2. Disabled count1 MTP: Selected(None) reaches the PreFfn action
        //    (plan_or_err passes with None) — the routed guard never fires
        //    and count1 does not panic.
        let sel = select_mtp_authority_single(&cfg, &w, false).expect("disabled count1 selection");
        assert_eq!(
            sel.plan_or_err("test")
                .expect("PreFfn passes for Selected(None)"),
            None,
            "Selected(None) must never yield a routed plan"
        );
        assert_ne!(R::SharedOnly, R::BiasAware); // routed classification is per-layer
                                                 // 3. count>1 still refuses BEFORE the disable bypass.
        let cfg2 = cfg_count2();
        let w2 = test_weights(&cfg2);
        let err = select_mtp_authority_single(&cfg2, &w2, false).unwrap_err();
        assert!(
            err.contains("refused before the MoE-disable bypass"),
            "{err}"
        );
    }

    // ── 44. scalar EP i64 partial: byte-shaped owner exposed as [hidden] ──
    #[test]
    fn ds4_scalar_ep_i64_partial_logical_shape_view() {
        use crate::forward::{build_ds4_parallel_program, raw_view};
        let cfg = tiny_cfg();
        let policy = ep_policy(2);
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let entry = ds4_cached_moe_plans(&ranks, &cfg, &policy).unwrap();
        let plan = entry.plan(3).unwrap();
        let hidden = cfg.hidden_size;
        // The OWNER is byte-shaped Raw: shape `[hidden*8]`, byte capacity
        // `hidden*8` (production: [32768] / 32768 bytes for hidden=4096).
        let owner = synth_with_bytes(DType::Raw, hidden * 8, hidden * 8);
        let tensors = decode_tensors(cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let router = || {
            ds4_router_plan(
                Ds4RouteSelection::BiasAware,
                synth_f32(cfg.n_routed_experts),
                Some(synth_f32(cfg.n_routed_experts)),
                None,
                None,
                tensors.topk_indices,
                tensors.topk_weights,
                cfg.num_experts_per_tok,
                2.2,
            )
            .unwrap()
        };
        // Regression: binding the byte-shaped owner DIRECTLY (the pre-fix
        // scalar seam) fails the sealed lowering — ConvertI64ToF32 n=hidden
        // vs the down buffer's logical dim hidden*8.
        let phases_bytes = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, hidden),
                Ds4DownTarget::I64 {
                    partial_i64: owner,
                    partial: synth_f32(hidden),
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                hidden,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let err = build_ds4_parallel_program(
            plan,
            &policy,
            router(),
            vec![phases_bytes(), phases_bytes()],
        )
        .unwrap_err();
        assert!(
            matches!(err, MoeLowerError::I64DimensionMismatch { .. }),
            "{err}"
        );
        // The fix: the production scalar seam exposes the owner as a
        // NON-OWNING logical `[hidden]` view — same buffer pointer and full
        // byte capacity, new logical shape.
        let view = raw_view(owner, vec![hidden]);
        assert!(std::ptr::eq(view.buf.as_ptr(), owner.buf.as_ptr()));
        assert_eq!(view.buf.size(), owner.buf.size(), "byte capacity retained");
        assert_eq!(view.shape, vec![hidden]);
        let phases_view = || {
            ds4_decode_phases(
                &tensors,
                expert_ref(cfg.moe_intermediate_size, hidden),
                Ds4DownTarget::I64 {
                    partial_i64: &view,
                    partial: synth_f32(hidden),
                },
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                hidden,
                cfg.swiglu_limit,
                1,
            )
            .unwrap()
        };
        let program =
            build_ds4_parallel_program(plan, &policy, router(), vec![phases_view(), phases_view()])
                .unwrap();
        assert_eq!(program.step_count(0), Some(4));
    }
    // ── 45. load-layout binding refuses BEFORE the disable bypass ────────
    #[test]
    fn ds4_load_layout_binding_refuses_before_disable_bypass() {
        use crate::forward::{acquire_moe_authority_mesh, acquire_moe_authority_single};
        let cfg = tiny_cfg();
        // EP policy + Single-layout weights → typed load-layout refusal even
        // with moe_on=false: the layout binding is part of the entry
        // contract (like the policy-kind binding) — wrong layouts refuse
        // regardless of the runtime switch, BEFORE any cache lookup.
        let ranks = [test_weights(&cfg), test_weights(&cfg)];
        let err = acquire_moe_authority_mesh(&ranks, &cfg, &ep_policy(2), false).unwrap_err();
        assert!(err.contains("load-layout binding"), "{err}");
        // Tp policy + Single-layout weights → typed refusal (same contract).
        let err = acquire_moe_authority_mesh(&ranks, &cfg, &tp_policy(2), false).unwrap_err();
        assert!(err.contains("load-layout binding"), "{err}");
        // Matching EP layout passes the binding (still disabled).
        let mut ep = [test_weights(&cfg), test_weights(&cfg)];
        ep[0].moe_load_layout = crate::deepseek4::Ds4MoeLoadLayout::Ep {
            shard_tp: 2,
            rank: 0,
        };
        ep[1].moe_load_layout = crate::deepseek4::Ds4MoeLoadLayout::Ep {
            shard_tp: 2,
            rank: 1,
        };
        assert!(acquire_moe_authority_mesh(&ep, &cfg, &ep_policy(2), false).is_ok());
        // Single authority refuses a non-Single layout even when disabled.
        let mut w = test_weights(&cfg);
        w.moe_load_layout = crate::deepseek4::Ds4MoeLoadLayout::Tp { tp: 2, rank: 0 };
        let err = acquire_moe_authority_single(&cfg, &w, false).unwrap_err();
        assert!(err.contains("requires a Single load layout"), "{err}");
    }
}
