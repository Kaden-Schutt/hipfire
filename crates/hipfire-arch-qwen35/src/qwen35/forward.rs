// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 decode forward: MoE decode, `Qwen35Scratch`, the per-token layer
//! loop, the #397 Ship-6 lowered super-op pipeline, and GPU-logits entry points.

use super::batch::PrefillBatchScratch;
use super::config::LayerType;
use super::config::MropeCtx;
use super::config::Qwen35Config;
use super::prefill::dump_hidden_localize;
use super::prefill::routed_codebook_pair_batched_supported;
use super::prefill::trace_finite_if_enabled;
use super::prefill::PREFILL_MAX_BATCH;
use super::weights::per_expert_tier_tables;
use super::weights::DeltaNetState;
use super::weights::ExpertWeights;
use super::weights::LayerWeights;
use super::weights::MoeFfnWeights;
use super::weights::Qwen35Weights;
use super::weights::StateQuant;
use crate::speculative::HiddenStateRingBuffer;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::GivensRef;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::KvTierInputs;
use hipfire_dispatch::families::kv_tier::KvTierPlan;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::superop;
use hipfire_dispatch::pipeline::superop::ForwardBindings;
use hipfire_dispatch::pipeline::superop::LayerProgram;
use hipfire_dispatch::pipeline::superop::OpBinding;
use hipfire_dispatch::pipeline::superop::OpFlavor;
use hipfire_dispatch::pipeline::superop::SuperOp;
use hipfire_dispatch::pipeline::superop::SuperOpKind;
use hipfire_dispatch::pipeline::superop::WeightSlot;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_dispatch::types::DispatchError;
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama;
use hipfire_runtime::llama::fused_rmsnorm_rotate_for_mq;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::ParoRotation;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tp_shard::ShardConfig;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

// ─── MoE FFN (decode, batch=1) ──────────────────────────────────────────

/// Construct a non-owning `GpuTensor` view over `[offset_elems,
/// offset_elems + len_elems)` of `src`. Valid only for F32 (4 bytes/elem).

/// One-token MoE FFN: router → top-K → shared expert + top-K routed, added
/// into `x_residual` in place. `x_norm` is the already-RMSNormed FFN input.
///
/// Dense-compute decode reference implementation (Phase 1). Top-K selection
/// runs on CPU via a single D2H sync per layer on the router logits; the
/// shared-expert scalar gate is another D2H sync. Sparse-routing + batched
/// grouped-GEMM variants come in later phases — this version prioritizes
/// correctness and minimal surface area.
///
/// Matches HF `modeling_qwen3_5_moe.py`:
///   router_probs  = softmax(W_router · x_norm)            // [n_exp]
///   (idx, w)      = topk(router_probs, k)                  // [k]
///   if norm_topk:  w /= w.sum()
///   scalar        = sigmoid(W_shared_gate · x_norm)        // [1]
///   y_shared      = scalar * shared_expert(x_norm)         // [hidden]
///   y_moe         = sum_{k} w[k] * expert[idx[k]](x_norm)  // [hidden]
///   x_residual   += y_shared + y_moe
/// Non-owning borrow of the scratch buffers `moe_ffn_decode_impl` needs.
/// Callers construct one of these from either a `Qwen35Scratch` (preallocated,
/// hipGraph-capturable) or from tensors they own locally (heap path).
struct MoeScratchRef<'a> {
    router_logits: &'a GpuTensor,
    scalar_buf: &'a GpuTensor,
    x_rot_local: &'a GpuTensor,
    gate_up_buf: &'a GpuTensor,
    gate_buf: &'a GpuTensor,
    up_buf: &'a GpuTensor,
    ffn_hidden: &'a GpuTensor,
    ffn_out: &'a GpuTensor,
    gate_batch: &'a GpuTensor,
    up_batch: &'a GpuTensor,
    rot_batch: &'a GpuTensor,
    topk_indices: &'a GpuTensor,
    topk_weights: &'a GpuTensor,
    // [k_top × dim + ceil(dim/4)] f32 — per-(expert-rank) MoE down output
    // plus a zero-initialised u32 counter tail used only by the optional
    // expert-wave-preserving last-arriver combine experiment. The production
    // atomic-free expand+combine path ignores the tail. Mirrors the prefill
    // `pbs.moe_down_expanded_batch` layout with batch=1. Required so
    // the MoE FFN is byte-deterministic under hipGraph replay; see
    // task #100 root-cause notes in `forward_scratch`.
    down_expanded: &'a GpuTensor,
}

impl<'a> MoeScratchRef<'a> {
    /// View into a Qwen35Scratch's MoE fields. Panics if the caller didn't
    /// allocate MoE scratch (config.num_experts == 0).
    fn from_scratch(s: &'a Qwen35Scratch) -> Self {
        Self {
            router_logits: s
                .moe_router_logits
                .as_ref()
                .expect("MoE scratch not allocated"),
            scalar_buf: s.moe_scalar_buf.as_ref().expect("MoE scratch"),
            x_rot_local: s.moe_x_rot.as_ref().expect("MoE scratch"),
            gate_up_buf: s.moe_gate_up_buf.as_ref().expect("MoE scratch"),
            gate_buf: s.moe_gate_buf.as_ref().expect("MoE scratch"),
            up_buf: s.moe_up_buf.as_ref().expect("MoE scratch"),
            ffn_hidden: s.moe_ffn_hidden.as_ref().expect("MoE scratch"),
            ffn_out: s.moe_ffn_out.as_ref().expect("MoE scratch"),
            gate_batch: s.moe_gate_batch.as_ref().expect("MoE scratch"),
            up_batch: s.moe_up_batch.as_ref().expect("MoE scratch"),
            rot_batch: s.moe_rot_batch.as_ref().expect("MoE scratch"),
            topk_indices: s.moe_topk_indices.as_ref().expect("MoE scratch"),
            topk_weights: s.moe_topk_weights.as_ref().expect("MoE scratch"),
            down_expanded: s.moe_down_expanded.as_ref().expect("MoE scratch"),
        }
    }
}

/// Heap-allocating wrapper for callers without pre-allocated scratch (the
/// debug `forward()` path). Allocates 11 tensors, runs moe_ffn_decode_impl,
/// frees. NOT hipGraph-compatible. For hot-path decode, callers should go
/// through moe_ffn_decode_with_scratch which reuses pre-allocated buffers.
fn moe_ffn_decode(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let max_inter = mi.max(smi);

    let router_logits = gpu.alloc_tensor(&[n_exp], DType::F32)?;
    let scalar_buf = gpu.alloc_tensor(&[1], DType::F32)?;
    let x_rot_local = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_up_buf = gpu.alloc_tensor(&[2 * max_inter], DType::F32)?;
    let gate_buf = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let up_buf = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_hidden = gpu.alloc_tensor(&[max_inter], DType::F32)?;
    let ffn_out = gpu.alloc_tensor(&[hidden], DType::F32)?;
    let gate_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let up_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let rot_batch = gpu.alloc_tensor(&[k * mi], DType::F32)?;
    let topk_indices = gpu.alloc_tensor(&[k], DType::F32)?;
    let topk_weights = gpu.alloc_tensor(&[k], DType::F32)?;
    let down_expanded = gpu.zeros(&[k * hidden + hidden.div_ceil(4)], DType::F32)?;

    let refs = MoeScratchRef {
        router_logits: &router_logits,
        scalar_buf: &scalar_buf,
        x_rot_local: &x_rot_local,
        gate_up_buf: &gate_up_buf,
        gate_buf: &gate_buf,
        up_buf: &up_buf,
        ffn_hidden: &ffn_hidden,
        ffn_out: &ffn_out,
        gate_batch: &gate_batch,
        up_batch: &up_batch,
        rot_batch: &rot_batch,
        topk_indices: &topk_indices,
        topk_weights: &topk_weights,
        down_expanded: &down_expanded,
    };
    let result = moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, false, None, false, false,
    );

    for t in [
        router_logits,
        scalar_buf,
        x_rot_local,
        gate_up_buf,
        gate_buf,
        up_buf,
        ffn_hidden,
        ffn_out,
        gate_batch,
        up_batch,
        rot_batch,
        topk_indices,
        topk_weights,
        down_expanded,
    ] {
        gpu.free_tensor(t)?;
    }
    result
}

/// All gate-side + routed MoE weights are MQ4G256 — the precondition for
/// the prerotated fast path where the caller can fuse rmsnorm+FWHT via
/// `fused_rmsnorm_rotate_mq` and call `moe_ffn_decode_with_scratch_prerotated`.
pub(crate) fn ffn_all_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn_gate_side_mq4_for_moe(ffn)
        && ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
}

/// GATE-SIDE only MQ4 (router + shared expert), independent of the routed
/// experts. When true, the fused rmsnorm+FWHT path + prerotated MoE decode are
/// applicable even on a graded file: the MQ4 router routes on the rotated x via
/// the fused gate kernel (MoeResolution::gate_fusable), and the routed experts
/// (any FwhtG256 MQ-family) consume the same single rotated activation — saving
/// the un-fused rmsnorm + the per-component rotates the else-branch incurs.
/// The redline mq4r (MQ4 gate + graded experts) hits this; uniform MQ4 is a
/// superset (ffn_all_mq4_for_moe). Q8-router files (mq4p) correctly stay false.
pub(crate) fn ffn_gate_side_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
}

/// Detect any MQ3G256 / MQ3G256Lloyd weight inside a MoE FFN block (router,
/// shared expert gate/up/down, shared_expert_gate router-mix scalar, or any
/// routed expert's gate_up/down). The MoE batched FFN kernels assume HFQ4
/// layout (136 B/group); an MQ3 weight (104 B/group) or Lloyd-MQ3 weight
/// (112 B/group) would dispatch with the wrong stride. Used by the
/// captured-prefill and non-captured-prefill defense-in-depth checks.
///
/// Mirrors `is_mq3_any` in `forward_prefill_batch_single_chunk_captured`
/// (line 3325) so both cross-checks treat plain and Lloyd-MQ3 identically.
/// MQ3/MQ3-Lloyd in the STRUCTURAL parts of a MoE FFN (router, shared
/// expert gate/up/down). These are NOT served by the merged grouped kernel
/// (which only handles routed experts) → must still hard-error.
pub(crate) fn moe_ffn_has_mq3_structural(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    is_mq3_any(ffn.router.gpu_dtype)
        || is_mq3_any(ffn.shared_expert_gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.up.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.down.gpu_dtype)
}

/// MQ3/MQ3-Lloyd in ROUTED experts WITHOUT a tag table (uniform MQ3, not
/// the graded case the merged grouped kernel handles). When
/// `expert_dtype_tags` is `Some`, the merged grouped kernel carries the
/// correct per-expert stride and this returns false (pass).
pub(crate) fn moe_ffn_has_mq3_experts_uniform(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    ffn.expert_dtype_tags.is_none()
        && ffn
            .experts
            .iter()
            .any(|e| is_mq3_any(e.gate_up.gpu_dtype) || is_mq3_any(e.down.gpu_dtype))
}

/// Narrowed sibling of [`moe_ffn_has_mq3_experts_uniform`] for the NON-captured
/// batched-prefill entry point only.
///
/// The refusal that predicate drives exists because the batched MoE bodies used
/// to dispatch every routed weight through HFQ4-layout kernels (136 B/group), so
/// an MQ3/Lloyd routed expert (112 B/group) was read at the wrong stride. That is
/// no longer true for a UNIFORM-per-projection codebook pair: those now have real
/// grouped-GEMM arms keyed on their own dtype (`dispatch_grouped_gemm` ->
/// `gemm_mq{2,3}g256_lloyd_moe_grouped_wmma`), so the stride is correct by
/// construction and the layer is genuinely servable batched.
///
/// This MUST agree with [`moe_ffn_batched_admissible_for_dtypes`]'s codebook arm
/// in one direction: anything that arm ADMITS must return `false` here, or the
/// forward hard-errors on a model it just declared eligible. The shared helper
/// [`routed_codebook_pair_batched_supported`] is what keeps them in lockstep
/// (test: `mq3_refusal_never_fires_on_an_admitted_codebook_pair`). The reverse is
/// safe: returning `true` for a pair that is not admitted just keeps the
/// per-token fallback, which is today's behavior.
///
/// DELIBERATELY NOT used by `forward_prefill_batch_single_chunk_captured_opts`.
/// That entry point has no eligibility check and no per-token fallback: its
/// refusal is the SOLE guard there, and narrowing it would hand these dtypes to
/// a hipGraph-captured prefill that has never been validated (both the kernel
/// JIT and the `ensure_fp16_x` staging inside the grouped launchers happen on
/// the first call, i.e. mid-capture). Correct-and-slow beats fast-and-corrupt.
pub(crate) fn moe_ffn_has_unsupported_mq3_experts_uniform(
    ffn: &MoeFfnWeights,
    admit_codebook: bool,
) -> bool {
    unsupported_mq3_experts_uniform_from_dtypes(
        ffn.expert_dtype_tags.is_some(),
        ffn.experts
            .iter()
            .map(|e| (e.gate_up.gpu_dtype, e.down.gpu_dtype)),
        admit_codebook,
    )
}

/// Pure core of [`moe_ffn_has_unsupported_mq3_experts_uniform`], split out
/// because `MoeFfnWeights` needs live GPU tensors and cannot be built in a unit
/// test. `experts` yields `(gate_up_dtype, down_dtype)` per routed expert.
pub(crate) fn unsupported_mq3_experts_uniform_from_dtypes(
    has_tag_table: bool,
    experts: impl IntoIterator<Item = (DType, DType)>,
    admit_codebook: bool,
) -> bool {
    // A tag table means the merged grouped kernel carries the per-expert stride
    // — the graded case, never this refusal's concern.
    if has_tag_table {
        return false;
    }
    let dtypes: Vec<(DType, DType)> = experts.into_iter().collect();
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    if !dtypes
        .iter()
        .any(|&(gu, dn)| is_mq3_any(gu) || is_mq3_any(dn))
    {
        return false;
    }
    if !admit_codebook {
        return true;
    }
    let Some(&first) = dtypes.first() else {
        return true;
    };
    // Uniformity is re-derived rather than assumed: the admission arm only
    // reaches its codebook branch when `expert_gate_up_uniform &&
    // expert_down_uniform`, so a mixed-without-tags file must stay refused
    // (`dispatch_grouped_gemm` would apply experts[0]'s group stride to all).
    let uniform = dtypes.iter().all(|&d| d == first);
    !(uniform && routed_codebook_pair_batched_supported(first.0, first.1))
}

fn moe_ffn_has_mq6(ffn: &MoeFfnWeights) -> bool {
    let is_mq6 = |dt: DType| matches!(dt, DType::MQ6G256);
    is_mq6(ffn.router.gpu_dtype)
        || is_mq6(ffn.shared_expert_gate.gpu_dtype)
        || is_mq6(ffn.shared_expert.gate.gpu_dtype)
        || is_mq6(ffn.shared_expert.up.gpu_dtype)
        || is_mq6(ffn.shared_expert.down.gpu_dtype)
        || ffn
            .experts
            .iter()
            .any(|e| is_mq6(e.gate_up.gpu_dtype) || is_mq6(e.down.gpu_dtype))
}

pub(crate) fn layers_have_mq6_moe(layers: &[LayerWeights]) -> bool {
    layers.iter().any(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => moe_ffn_has_mq6(&l.ffn),
        LayerWeights::FullAttnMoe(l) => moe_ffn_has_mq6(&l.ffn),
        _ => false,
    })
}

/// Zero-alloc MoE decode for the scratch path. `scratch.moe_*` fields must
/// be populated (done automatically by `Qwen35Scratch::new` when config
/// indicates a MoE model). Safe to call under hipGraph stream capture.
pub(crate) fn moe_ffn_decode_with_scratch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, false, None, false, false,
    )
}

/// Same as `moe_ffn_decode_with_scratch` but expects the caller to have
/// already populated `scratch.moe_x_rot` with FWHT-rotated post-rmsnorm x
/// (e.g. via a fused `fused_rmsnorm_rotate_mq` launch at the call site).
/// For all-MQ4 MoE layers this saves one launch per layer by eliding the
/// internal `rotate_x_mq`. On non-MQ4 layers this flag is ignored.
pub(crate) fn moe_ffn_decode_with_scratch_prerotated(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, ffn, x_norm, x_residual, config, &refs, true, None, false, false,
    )
}

/// The actual MoE FFN implementation. Uses the caller-provided scratch
/// buffers, never allocates.
// ── REAP expert-importance capture (HIPFIRE_MOE_EXPERT_STATS=1) ────────────
// Per-(layer, expert) accumulators: routing count, Σ gate_weight,
// Σ ‖expert_output‖, Σ (gate × ‖output‖). The last is the true REAP
// contribution (gate-weighted output norm) — compared against raw frequency
// (count) to decide whether freq agrees with contribution before committing to
// any per-expert mixed-precision kernel. Dumped by `dump_expert_stats`.
static EXPERT_STATS: std::sync::Mutex<
    Option<std::collections::HashMap<(u16, u16), (u64, f64, f64, f64)>>,
> = std::sync::Mutex::new(None);
static EXPERT_STATS_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn expert_stats_enabled() -> bool {
    *EXPERT_STATS_ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_EXPERT_STATS")
            .ok()
            .as_deref()
            == Some("1")
    })
}
fn capture_expert_stats(
    gpu: &Gpu,
    layer_idx: u16,
    k: usize,
    hidden: usize,
    down_expanded: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
) {
    let dn = match gpu.download_f32(down_expanded) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ti = match gpu.download_f32(topk_indices) {
        Ok(v) => v,
        Err(_) => return,
    };
    let tw = match gpu.download_f32(topk_weights) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut guard = EXPERT_STATS.lock().unwrap();
    let m = guard.get_or_insert_with(std::collections::HashMap::new);
    for krank in 0..k {
        if krank >= ti.len() || krank >= tw.len() {
            break;
        }
        let e = (ti[krank].to_bits() as i32) as u16; // i32-in-F32 alias
        let w = tw[krank] as f64;
        let base = krank * hidden;
        if base + hidden > dn.len() {
            break;
        }
        let mut sq = 0.0f64;
        for j in 0..hidden {
            let x = dn[base + j] as f64;
            sq += x * x;
        }
        let norm = sq.sqrt();
        let ent = m.entry((layer_idx, e)).or_insert((0, 0.0, 0.0, 0.0));
        ent.0 += 1;
        ent.1 += w;
        ent.2 += norm;
        ent.3 += w * norm;
    }
}
/// Dump the accumulated per-(layer,expert) REAP stats to a TSV. Called from
/// eval harnesses when HIPFIRE_MOE_EXPERT_STATS_OUT is set.
pub fn dump_expert_stats(path: &str) {
    let guard = EXPERT_STATS.lock().unwrap();
    let m = match guard.as_ref() {
        Some(m) if !m.is_empty() => m,
        _ => {
            eprintln!("expert_stats: empty (capture not enabled?)");
            return;
        }
    };
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by_key(|((l, e), _)| (*l, *e));
    let mut out = String::from("layer\texpert\tcount\tsum_gate\tsum_norm\tsum_contrib\n");
    for ((l, e), (c, sg, sn, sc)) in rows {
        out.push_str(&format!("{l}\t{e}\t{c}\t{sg:.6}\t{sn:.6}\t{sc:.6}\n"));
    }
    match std::fs::write(path, out) {
        Ok(_) => eprintln!("expert_stats: wrote {path} ({} layer×expert rows)", m.len()),
        Err(e) => eprintln!("expert_stats: write failed {path}: {e}"),
    }
}

fn moe_ffn_decode_impl(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    s: &MoeScratchRef<'_>,
    x_rot_prerotated: bool,
    // EP (Ship 6 substrate-EP). `ep_routed_out = Some(partial)` redirects the
    // routed combine + shared-down into a zeroed partial (the EP executor
    // all-reduces it and adds into x_residual once); `None` = single-GPU into
    // x_residual (byte-identical). `ep_skip_shared` skips the shared-expert
    // down on rank>0 so the replicated shared expert is summed once.
    ep_routed_out: Option<&GpuTensor>,
    ep_skip_shared: bool,
    defer_routed_combine: bool,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    // SP2: if a layer's experts span >1 quant tier (e.g. a re-quant overlay
    // bumped some experts), expose the per-expert tier tables so dispatch
    // buckets by tier. The common case (a uniform layer — or paged mode where
    // `experts` is empty) yields None → unchanged uniform fast path.
    let (per_expert_gate_up, per_expert_down) = per_expert_tier_tables(ffn);
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        shared_expert_down: ffn.shared_expert.down.gpu_dtype,
        experts_all_gate_up_mq4: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global.iter().all(|(g, _)| *g == DType::MQ4G256)
        } else {
            ffn.experts
                .iter()
                .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
        },
        routed_gate_up: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global.first().map(|(g, _)| *g).unwrap_or(DType::F32)
        } else {
            ffn.experts
                .first()
                .map(|e| e.gate_up.gpu_dtype)
                .unwrap_or(DType::F32)
        },
        routed_down: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global.first().map(|(_, d)| *d).unwrap_or(DType::F32)
        } else {
            ffn.experts
                .first()
                .map(|e| e.down.gpu_dtype)
                .unwrap_or(DType::F32)
        },
        // Single source of truth: the tag table is built iff experts carry
        // mixed down dtypes, so its presence == the mixed-per-expert flag.
        routed_has_mixed_experts: ffn.expert_dtype_tags.is_some(),
        has_paro_shared: ffn.paro_shared.is_some(),
        per_expert_gate_up,
        per_expert_down,
    };
    // Resolution is owned by the MoeFamily (Ship 4.1). The model passes only
    // the dtype snapshot + k; the executor computes MoeResolution from MoeDtypes.

    // Per-expert (gate_up, down) refs for the generic CPU-top-K fallback in
    // `run_moe_decode` (k != 8 OR routed dtype not indexable). Empty in paged
    // mode (`ffn.experts` is empty — only the indexed GPU-top-K path runs
    // there), matching master's `ffn.experts[..]` indexing requirement.
    let routed_experts: Vec<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> = ffn
        .experts
        .iter()
        .map(|e| (e.gate_up.dispatch_ref(), e.down.dispatch_ref()))
        .collect();

    let moe_params = hipfire_dispatch::families::moe::MoeParams {
        dtypes: moe_dtypes,
        batch_size: 1,
        hidden,
        mi,
        smi,
        k,
        n_exp,
        norm_topk_prob: config.norm_topk_prob,
        x_rot_prerotated,
        defer_routed_combine,
        layer_idx: ffn.layer_idx,
        x_norm,
        x_residual,
        // EP (Ship 6 substrate-EP): threaded from moe_ffn_decode_impl params.
        // None/false (single-GPU) = byte-identical; Some(partial)/skip_shared
        // come from moe_ffn_dispatch_ep via run_layer_program_ep.
        routed_out: ep_routed_out,
        skip_shared: ep_skip_shared,
        router: ffn.router.dispatch_ref(),
        shared_expert_gate: ffn.shared_expert_gate.dispatch_ref(),
        shared_gate_w: ffn.shared_expert.gate.dispatch_ref(),
        shared_up_w: ffn.shared_expert.up.dispatch_ref(),
        shared_down_w: ffn.shared_expert.down.dispatch_ref(),
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        // Route A MoE-AWQ: `Some` only on per-expert-AWQ .hfq files (the
        // HIPFIRE_MOE_AWQ kill-switch is applied once at load in load_moe_ffn,
        // not per-token). `None` ⇒ plain silu+rotate (byte-identical).
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        // Per-expert mixed-precision decode: `Some` only on graded mixed
        // files; drives the merged dtype-tag-branched down kernel + forces
        // the shared combine. `None` ⇒ uniform path (byte-identical).
        expert_dtype_tags: ffn.expert_dtype_tags.as_ref(),
        routed_gate_up_k: ffn.experts.first().map_or(0, |e| e.gate_up.k),
        routed_down_m: ffn.experts.first().map_or(0, |e| e.down.m),
        routed_down_k: ffn.experts.first().map_or(0, |e| e.down.k),
        routed_experts: &routed_experts,
        routed_gate_up_paro: ffn.experts.first().and_then(|e| {
            e.gate_up
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        routed_down_paro: ffn.experts.first().and_then(|e| {
            e.down
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        router_logits: s.router_logits,
        scalar_buf: s.scalar_buf,
        x_rot_local: s.x_rot_local,
        gate_up_buf: s.gate_up_buf,
        gate_buf: s.gate_buf,
        up_buf: s.up_buf,
        ffn_hidden: s.ffn_hidden,
        ffn_out: s.ffn_out,
        gate_batch: s.gate_batch,
        up_batch: s.up_batch,
        rot_batch: s.rot_batch,
        topk_indices: s.topk_indices,
        topk_weights: s.topk_weights,
        down_expanded: s.down_expanded,
    };
    // Build one DispatchCtx per token (the family threads it through every
    // inner GEMV — no internal DispatchCtx::new reconstructions).
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    hipfire_runtime::llama::moe_family()
        .run(&ctx, gpu, &moe_params)
        .map_err(HipError::from)?;
    if expert_stats_enabled() {
        capture_expert_stats(
            gpu,
            ffn.layer_idx,
            k,
            hidden,
            s.down_expanded,
            s.topk_indices,
            s.topk_weights,
        );
    }
    Ok(())
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Shared forward pass — returns logits as CPU Vec<f32>.
fn forward_from_x(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let logits_gpu = forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)?;
    let logits_data = gpu.download_f32(&logits_gpu)?;
    gpu.free_tensor(logits_gpu)?;
    Ok(logits_data)
}

/// Shared forward pass — returns logits as GPU tensor (no download).
/// Shared forward pass — returns logits as GPU tensor (no download).
/// Caller must free the returned tensor.
///
/// Delegates to `forward_scratch_layers` via a temporary `Qwen35Scratch`,
/// ensuring test/demo paths exercise the same pipeline code as production.
/// NOT production-representative for benchmarking: allocates and frees a full
/// scratch bundle per call. Use `forward_scratch` with a persistent scratch
/// for perf measurement. Per-layer `DEBUG_LAYERS` trace and `trace_finite`
/// "qkvza" checkpoint are not emitted in this path — they are available
/// via `dump_hidden_localize` in the scratch path under HIPFIRE_DUMP_HIDDEN.
fn forward_from_x_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let required_tokens = checked_kv_end(pos, 1, "forward_from_x_gpu")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;

    // Allocate a temporary scratch bundle. repeat_window=1 (unused in this path).
    // kv_max_seq=8192 matches Qwen35Scratch::new default — sufficient for
    // test/demo single-token forward; these callers don't prefill.
    let scratch = Qwen35Scratch::new(gpu, config, 1)?;

    // Copy input embedding into scratch.x
    gpu.hip.memcpy_dtod(&scratch.x.buf, &x.buf, dim * 4)?;
    gpu.free_tensor(x)?;

    // Set position buffer
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // DEBUG_LAYERS: dump embedding + per-layer norms (same as old forward_from_x_gpu)
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Run the production pipeline
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, &scratch, None, None,
    )?;

    // DEBUG_LAYERS: dump per-layer residual norms
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "POST: first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Copy logits out of scratch before freeing — the returned tensor must
    // outlive the scratch bundle.
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    gpu.hip
        .memcpy_dtod(&logits.buf, &scratch.logits.buf, config.vocab_size * 4)?;

    // Free scratch (all pre-allocated buffers)
    scratch.free_gpu(gpu);

    Ok(logits)
}

/// Pre-allocated scratch buffers for zero-alloc qwen35 forward + GPU sampling.
pub struct Qwen35Scratch {
    // Persistent state
    pub x: GpuTensor,                      // [dim]
    pub tmp: GpuTensor,                    // [dim]
    pub pos_buf: hip_bridge::DeviceBuffer, // 4 bytes
    /// 3 i32 = (t, h, w) for the 3D mrope kernels. Written ONLY on the VL
    /// path (`MropeCtx` present); the 1D kernels never read it, so the
    /// text-only dispatch sequence is unaffected by its existence.
    pub pos_buf3: hip_bridge::DeviceBuffer, // 12 bytes

    // DeltaNet temporaries (reused across layers)
    pub dn_qkv: GpuTensor,      // [qkv_dim]
    pub dn_z: GpuTensor,        // [v_dim]
    pub dn_alpha: GpuTensor,    // [n_v_heads]
    pub dn_beta: GpuTensor,     // [n_v_heads]
    pub dn_conv_out: GpuTensor, // [qkv_dim]
    pub dn_q: GpuTensor,        // [v_dim] (after repeat-interleave)
    pub dn_k: GpuTensor,        // [v_dim]
    pub dn_v: GpuTensor,        // [v_dim]
    pub dn_q_raw: GpuTensor,    // [k_dim] (before repeat)
    pub dn_k_raw: GpuTensor,    // [k_dim]
    pub dn_attn_out: GpuTensor, // [v_dim]
    pub dn_normed: GpuTensor,   // [v_dim]

    // FullAttn temporaries (reused across layers)
    pub fa_q_full: GpuTensor,   // [n_heads * head_dim * 2]
    pub fa_q: GpuTensor,        // [n_heads * head_dim]
    pub fa_gate: GpuTensor,     // [n_heads * head_dim]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim]

    // Shared (used by both layer types)
    pub o: GpuTensor,          // [dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up: GpuTensor,         // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Sampling
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — token_id + rng
    pub repeat_buf: GpuTensor, // [repeat_window]

    // MagnumQuant rotation scratch: FWHT(x) shared across Q/K/V (or gate/up, etc).
    // DeltaNet's output projection consumes v_dim values, which can exceed both
    // dim and the unused dense hidden_dim on MoE checkpoints.
    pub x_rot: GpuTensor, // [max(dim, hidden_dim, v_dim)]

    // Flash attention partials buffer for tile+reduce 2-kernel path.
    // Size: n_heads * max_tiles * (2 + head_dim) floats.
    pub flash_partials: GpuTensor,
    // Flash attention tri-state (applies to Q8 path; asym modes are flash-only):
    //   0 = never      force non-flash at all contexts (except >15K sanity)
    //   1 = auto       (default) flash kicks in at ctx >= 2048
    //   2 = always     force flash at all contexts
    pub flash_mode: u8,

    // MoE scratch (allocated only when config.num_experts > 0). Pre-allocated
    // so moe_ffn_decode can be captured by hipGraph — the per-layer allocs
    // it used to do violated the "no allocator ops while capturing" rule.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1] shared-expert gate scalar
    pub moe_x_rot: Option<GpuTensor>,         // [dim]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2*max_inter]   fallback path
    pub moe_gate_buf: Option<GpuTensor>,      // [max_inter]     fallback path
    pub moe_up_buf: Option<GpuTensor>,        // [max_inter]     fallback path
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max_inter]     fallback path
    pub moe_ffn_out: Option<GpuTensor>,       // [dim]           fallback path
    pub moe_gate_batch: Option<GpuTensor>,    // [k × mi]
    pub moe_up_batch: Option<GpuTensor>,      // [k × mi]
    pub moe_rot_batch: Option<GpuTensor>,     // [k × mi]
    /// Phase 2b: GPU-side top-K outputs (kept on-device so moe_ffn_decode
    /// can stay in a graph-capturable stream).
    pub moe_topk_indices: Option<GpuTensor>, // [k] i32 stored as f32 alias
    pub moe_topk_weights: Option<GpuTensor>,  // [k] f32
    // Atomic-free MoE down expansion buffer for decode — payload [k × dim]
    // plus ceil(dim/4) f32-sized counter slots for the optional last-arriver
    // combine. The full allocation is zeroed once; each fused dispatch resets
    // the counters it consumes.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched` (batch_size=1) in `moe_ffn_decode_impl`'s
    // use_gpu_topk path. Replaces the K_TOP-way atomicAdd that introduced
    // non-deterministic wavefront-order-dependent FP rounding under hipGraph
    // replay (task #100).
    pub moe_down_expanded: Option<GpuTensor>,

    // Optional long-prefill scratch. Default is None to preserve VRAM
    // footprint; set HIPFIRE_PREFILL_REUSE_PBS=1 to allocate and reuse it.
    pub prefill_batch: Option<PrefillBatchScratch>,
}

fn qwen35_x_rot_len(dim: usize, hidden_dim: usize, v_dim: usize) -> usize {
    dim.max(hidden_dim).max(v_dim)
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        // Flash partials are sized for up to 8192 ctx. Override via new_with_kv_max.
        Self::new_with_kv_max(gpu, config, repeat_window, 8192)
    }

    pub fn new_with_kv_max(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            pos_buf3: gpu.hip.malloc(12)?,

            dn_qkv: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_z: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_alpha: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_beta: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_conv_out: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_q: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_k: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_v: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_q_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_k_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_attn_out: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_normed: gpu.alloc_tensor(&[v_dim], DType::F32)?,

            fa_q_full: gpu.alloc_tensor(&[q_dim * 2], DType::F32)?,
            fa_q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,

            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,

            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[repeat_window], DType::F32)?,
            x_rot: gpu.alloc_tensor(
                &[qwen35_x_rot_len(dim, config.hidden_dim, v_dim)],
                DType::F32,
            )?,

            // Flash attention partials: enough for the smallest tile used by
            // Q8 decode experiments and the fixed tile_size=128 paths.
            // n_heads * max_tiles * (2 + head_dim) floats per batched query
            // position; total buffer = batch_mult × per-position-bytes.
            //
            // batch_mult is the maximum query positions a single FA dispatch
            // can fit; the dispatcher (`launch_asym_flash_batched`) reads the
            // buffer's actual capacity at call time and auto-chunks larger
            // prefill batches into multiple sub-launches. So a lower
            // batch_mult here trades ~linear extra dispatch overhead on
            // prefill (PREFILL_MAX_BATCH=256 → ceil(256/batch_mult) calls per
            // FA layer) for ~linearly less VRAM at long context.
            //
            // The per-position size scales with kv_max_seq (= physical_cap
            // post-eviction), and that scaling is what made #85 visible: at
            // max_seq=170k, no CASK, 27B (n_heads=24, head_dim=256) the old
            // batch_mult=64 → 2.1 GB just for these partials, exceeding VRAM
            // headroom on 24 GB cards. Cutting batch_mult by 4× (16) keeps
            // the prefill chunking moderate while saving 1.6 GB at that
            // worst-case shape; CASK-on workloads (small physical_cap) are
            // unaffected because the buffer is already tiny there.
            //
            // Override with HIPFIRE_FLASH_PARTIALS_BATCH for tuning. Power of
            // two preferred (matches FA dispatcher chunking).
            flash_partials: {
                let tile_size = rdna_compute::attention::q8_flash_tile_size(
                    &gpu.arch,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_max_seq,
                )
                .min(128)
                // See llama.rs: also floor against the batched-attention tile,
                // since a smaller HIPFIRE_ATTN_TILE_SIZE raises max_tiles and
                // would undersize this same buffer.
                .min(gpu.attn_tile_size());
                let max_tiles = (kv_max_seq + tile_size - 1) / tile_size;
                let batch_mult = hipfire_runtime::config::get()
                    .flash_partials_batch
                    .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH)
                    .unwrap_or(16);
                gpu.alloc_tensor(
                    &[batch_mult * config.n_heads * max_tiles * (2 + config.head_dim)],
                    DType::F32,
                )?
            },
            // Flash attention tri-state for the Q8 path. Asym modes always
            // flash regardless.
            //   HIPFIRE_ATTN_FLASH=never|0|off    → non-flash at all contexts
            //   HIPFIRE_ATTN_FLASH=auto|1|on      → flash at ctx >= 2048
            //   HIPFIRE_ATTN_FLASH=always|2|force → flash at all contexts
            //
            // Default on gfx11/gfx12 (graph-capable archs): `2` (always
            // flash). On other archs: `1` (auto). The capture path at
            // qwen35.rs:8199 hard-wires `use_flash = capture_mode || ...`
            // because attention_q8_0_kv has variable block_size + variable
            // shared-mem (not capture-safe). Without an always-flash default
            // on capture-capable archs, direct mode at small ctx silently
            // uses attention_q8_0_kv while a captured-and-replayed forward
            // uses attention_flash_q8_0 — same math, different fp32
            // reduction order, observed as ~0.44 logit delta direct-vs-graph
            // on shisa-Qwen3.6-A3B-PARO (see
            // .scratch/hipgraph-moe-drift-audit.md Part A). Aligning the
            // default flips both paths to `attention_flash_q8_0` and makes
            // direct vs graph byte-identical at the cost of moving small-
            // context decode off the non-flash kernel (~few % attention
            // perf hit, small contribution to total MoE decode time).
            // Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override
            // for users who prefer the non-flash kernel and don't intend
            // to use graph capture.
            flash_mode: match hipfire_runtime::config::get().attention_flash_mode.as_str() {
                "never" | "0" | "off" => 0,
                "always" | "2" | "force" => 2,
                _ => {
                    let graph_capable_arch =
                        gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
                    if graph_capable_arch {
                        2
                    } else {
                        1
                    }
                }
            },

            moe_router_logits: None,
            moe_scalar_buf: None,
            moe_x_rot: None,
            moe_gate_up_buf: None,
            moe_gate_buf: None,
            moe_up_buf: None,
            moe_ffn_hidden: None,
            moe_ffn_out: None,
            moe_gate_batch: None,
            moe_up_batch: None,
            moe_rot_batch: None,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_down_expanded: None,
            prefill_batch: None,
        })
        .and_then(|mut s| {
            // Allocate MoE scratch only for MoE configs. Done after the
            // main struct init so these Options start as None for dense
            // models and never cost VRAM there.
            if config.num_experts > 0 {
                let hidden = config.dim;
                let n_exp = config.num_experts;
                let mi = config.moe_intermediate_size;
                let smi = config.shared_expert_intermediate_size;
                let max_inter = mi.max(smi);
                let k = config.num_experts_per_tok;
                s.moe_router_logits = Some(gpu.alloc_tensor(&[n_exp], DType::F32)?);
                s.moe_scalar_buf = Some(gpu.alloc_tensor(&[1], DType::F32)?);
                s.moe_x_rot = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_up_buf = Some(gpu.alloc_tensor(&[2 * max_inter], DType::F32)?);
                s.moe_gate_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_up_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_hidden = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_out = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_up_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_rot_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                // i32 topk_indices stored in an F32 tensor (same byte width).
                // The kernel that writes it casts the buffer to int*, and the
                // indexed MoE GEMV kernels read it as int*.
                s.moe_topk_indices = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                s.moe_topk_weights = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                // Atomic-free decode MoE down payload plus reusable counter tail.
                s.moe_down_expanded =
                    Some(gpu.zeros(&[k * hidden + hidden.div_ceil(4)], DType::F32)?);
                // Pre-warm MQ FWHT sign tables (otherwise the lazy init in
                // ensure_mq_signs fires during the first moe_ffn_decode and
                // blows up hipGraph capture with a hipMalloc-in-capture
                // error). Idempotent if already computed.
                gpu.ensure_mq_signs()?;
            }
            if hipfire_config::developer_var("HIPFIRE_PREFILL_REUSE_PBS")
                .ok()
                .as_deref()
                == Some("1")
            {
                let max_batch = hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&v| v >= 2)
                    .unwrap_or(PREFILL_MAX_BATCH);
                s.prefill_batch = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
            }
            Ok(s)
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM. Checked — reports first free failure.
    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        };
        note(gpu.free_tensor(self.x));
        note(gpu.free_tensor(self.tmp));
        let _ = gpu.bind_thread();
        note(gpu.hip.free(self.pos_buf).map(|_| ()));
        note(gpu.hip.free(self.pos_buf3).map(|_| ()));
        for t in [
            self.dn_qkv,
            self.dn_z,
            self.dn_alpha,
            self.dn_beta,
            self.dn_conv_out,
            self.dn_q,
            self.dn_k,
            self.dn_v,
            self.dn_q_raw,
            self.dn_k_raw,
            self.dn_attn_out,
            self.dn_normed,
            self.fa_q_full,
            self.fa_q,
            self.fa_gate,
            self.fa_k,
            self.fa_v,
            self.fa_attn_out,
            self.o,
            self.gate_ffn,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.x_rot,
            self.flash_partials,
        ] {
            note(gpu.free_tensor(t));
        }
        // MoE scratch — only present for MoE configs.
        for t in [
            self.moe_router_logits,
            self.moe_scalar_buf,
            self.moe_x_rot,
            self.moe_gate_up_buf,
            self.moe_gate_buf,
            self.moe_up_buf,
            self.moe_ffn_hidden,
            self.moe_ffn_out,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
        ] {
            if let Some(buf) = t {
                note(gpu.free_tensor(buf));
            }
        }
        if let Some(pbs) = self.prefill_batch {
            note(pbs.free_gpu(gpu));
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Per-device scratch bundle for the multi-GPU forward path. Each device gets
/// its own `Qwen35Scratch` because the residual stream `s.x` (and `s.logits`)
/// must live on the device executing the current band's layers — cross-band
/// boundaries copy `s.x` between devices via `Gpus::boundary_copy`. `s.logits`
/// is also allocated per-device for simplicity (~600 KB each at vocab=152K)
/// even though only the output device's `s.logits` is consumed post-loop.
pub struct Qwen35ScratchSet {
    pub per_device: Vec<Qwen35Scratch>,
}

impl Qwen35ScratchSet {
    pub fn new_with_kv_max_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let mut per_device = Vec::with_capacity(gpus.devices.len());
        for dev_idx in 0..gpus.devices.len() {
            let g = &mut gpus.devices[dev_idx];
            per_device.push(Qwen35Scratch::new_with_kv_max(
                g,
                config,
                repeat_window,
                kv_max_seq,
            )?);
        }
        Ok(Self { per_device })
    }

    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (dev_idx, scratch) in self.per_device.into_iter().enumerate() {
            scratch.free_gpu(&mut gpus.devices[dev_idx]);
        }
    }
}

pub(crate) fn checked_kv_end(start_pos: usize, token_count: usize, site: &str) -> HipResult<usize> {
    start_pos.checked_add(token_count).ok_or_else(|| {
        HipError::new(
            0,
            &format!("{site}: KV token range overflow ({start_pos} + {token_count})"),
        )
    })
}

#[inline]
fn ar_graph_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_AR_GRAPH_TRACE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[inline]
fn ar_graph_eligible_for_kv(requested: bool, compact_offset: usize) -> bool {
    // The captured single-token route is built while compact_offset is zero.
    // After eviction, Q/K RoPE must use physical_pos + compact_offset, and the
    // offset changes again at every later eviction. Neither hipGraph nor the
    // retained replay route currently has a dynamic offset input, so replaying
    // the old route silently rotates at the physical slot and corrupts decode.
    requested && compact_offset == 0
}

/// Zero-alloc forward pass using pre-allocated scratch buffers.
/// Logits stay on GPU in scratch.logits. Returns nothing — caller uses scratch.logits.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch")?;
    // Grow before any possible AR graph capture/replay. Stable virtual
    // addresses keep existing graph pointer arguments valid.
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    // hipGraph capture for MoE was previously gated off-by-default behind
    // HIPFIRE_GRAPH_MOE=1 because of a known drift bug (task #100): under
    // capture, A3B accumulated a per-step ~1-ULP delta that compounded
    // through the KV cache + GDN state and crossed the top-1 margin at
    // step ~7 (q8 KV) or ~114 (asym3 KV), producing visible token-loop
    // attractors by step 30-50 ("- **One**\n- **One**\n…").
    //
    // Root cause (fixed 2026-05-21): `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`
    // used K_TOP=8 concurrent `atomicAdd` writes per output row. FP32
    // addition is non-associative, so the final bits depend on wavefront
    // scheduling order. Under hipGraph replay that order differs from
    // direct execution (graph scheduling pipelines kernels differently),
    // introducing the systematic per-step delta. The kernel's own header
    // (`kernels/src/gemv_hfq4g256_moe_down.hip:14-19`) had already flagged
    // this non-determinism but rated it negligible based on the
    // direct-only smoke test — capture amplifies the effect.
    //
    // Fix: the MoE FFN decode path now uses the atomic-free expand+combine
    // pattern already used in prefill (`forward_prefill_batch_with_pbs`
    // L5217-5232): `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
    // writes one row per (expert-rank, m), then `moe_down_combine_k8_batched`
    // sums K_TOP slots into x_residual in a fixed iteration order. The
    // resulting MoE FFN output is byte-deterministic under both direct
    // execution and hipGraph replay.
    //
    // `experimental.graph.moe` remains the explicit policy control. The atomic
    // fix is necessary but not sufficient — the CPU-topK fallback path
    // (when not all gate-side MoE weights are MQ4G256, e.g. router=Q8 per
    // the post-2026-04 router-attractor fix) calls `download_f32(router_logits)`,
    // a sync D2H that fails under graph capture with hipError 906. Until
    // that D2H is migrated to a capture-safe equivalent, opting in only
    // works for models where the runtime takes the use_gpu_topk path.
    //
    // Reproducer used to characterize the fix:
    //   hipfire config set experimental.graph.forward true
    //   hipfire config set experimental.graph.moe true
    //   HIPFIRE_SMOKE_KV=q8 \
    //   HIPFIRE_SMOKE_MODE=chat HIPFIRE_SMOKE_STEPS=200 \
    //   HIPFIRE_SMOKE_PROMPT="Count from one to twenty in English." \
    //   ./target/release/examples/a3b_smoke_forward <uniform-mq4-a3b>
    //
    // Graph policy is resolved once from TOML before GPU initialization and is
    // carried directly by the GPU feature snapshot.
    // Default-ON (2026-06-16): cross-arch A/B validated — gfx12 +4.2% A3B mq4
    // decode, coherence-gate clean (fluent at decode pos 1-4 on current ROCm).
    // Opt out with `experimental.graph.moe = false`. Both root causes of the prior
    // crash/drift are fixed:
    //   1. atomicAdd drift (task #100, 2026-05-21): expand+combine pattern.
    //   2. CPU-topK fallback D2H: `download_f32(router_logits)` replaced by
    //      GPU `softmax_f32` + `moe_topk_renorm_k8` + small [k] D2H — fully
    //      capture-safe. Mixed-kmap A3B (Q8 router, post-PR #199) no longer
    //      crashes with hipError 906 under AR graph capture.
    // Validated + flipped to default-on 2026-06-16.
    let allow_moe = gpu.flags.graph_moe;
    // hipGraph per-forward-pass capture/replay default policy:
    //   - gfx12 (RDNA4): default-ON. +2.4-2.7% decode on 9B Qwen 3.5
    //     MFP4G32 (5-run mean, all positive, tight variance, 2026-05-11).
    //   - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
    //     0.8B HFP4G32 on 7900 XTX (5-run mean per model, all positive,
    //     variance 1.001-1.010×, 2026-05-11). Smaller win than gfx12 —
    //     gfx11 has less per-launch overhead to amortize — but real
    //     and consistent across model sizes.
    //   - other archs (RDNA1/2, CDNA): default-OFF (opt-in via
    //     `experimental.graph.forward = true`) since not yet A/B'd on those.
    //   - MoE configs follow `experimental.graph.moe`. The ~30-50-token
    //     attractor drift in the use_gpu_topk MoE down step was fixed
    //     2026-05-21 (task #100 — atomicAdd → expand+combine), but the
    //     CPU-topK fallback's `download_f32(router_logits)` D2H sync
    //     remains capture-incompatible, so mixed-kmap A3B (post-PR #199)
    //     can crash under graph capture even with the fix. Once that
    //     D2H is migrated to a capture-safe path, the MoE default can
    //     be flipped to follow the arch defaults.
    // An explicit `experimental.graph.forward = false` always wins.
    let graph_override = gpu.flags.graph_forward;
    let graph_arch_default = gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
    let graph_enabled = graph_override.unwrap_or(graph_arch_default);
    // AR-forward hipGraph RE-ENABLED (2026-06-16) — the kernarg-snapshot attractor
    // is re-verified GONE on current ROCm (HIP 7.x, coherence-gate clean, fluent at
    // decode pos 1-4 on gfx12 A3B mq4). Opt out with
    // `experimental.graph.ar = false`. The prior
    // 2026-05-15 disable rationale is retained below; it SUPERSEDED the
    // arch-default re-enable merged from master (`graph_enabled` above is kept
    // live so the graph policy and kill switch stay wired for when the
    // path is flipped back on). Empirically on ROCm 7.2.2 + gfx11 +
    // Qwen3.5-27B mq4, both replay AND capture+launch produce a token-0
    // attractor outside very narrow conditions:
    //   - Capture+launch at position 2 (after 1 direct warmup) → `!!!!!`
    //   - Capture+launch at position 4 (after 3 direct warmups) → correct
    //   - Replay of a working capture (any position) → `!!!!!` from pos+1 on
    // The kernarg-snapshot bug isn't fixable by warmup tuning OR caller-driven
    // commit gating (`end_decode_turn()`); both fail empirically. Master's
    // task-#100 fix targets MoE drift, NOT this AR-forward attractor, so the
    // merge does not clear the disable. Until the capture/replay attractor is
    // re-verified gone on current ROCm (7.13) via the coherence gate, AR
    // forward is direct-only. Policy infra (`ar_forward_kernel_dirty`,
    // `ar_forward_replay_enabled`, `end_decode_turn()`, `drop_captured_graph()`)
    // is preserved on Gpu so the path can be flipped on once the bug is fixed.
    // RE-VERIFY GATE (2026-06-12): the AR graph policy flips the 2026-05-15
    // disable back on to re-test the capture/replay attractor on current ROCm
    // (HIP 7.x). Default OFF preserves the direct-only behavior. When set, the
    // path still honors the HIPFIRE_GRAPH kill switch + arch default via
    // `graph_enabled`.
    let ar_graph_test = gpu.flags.graph_ar;
    // AR-forward hipGraph eligibility. Plain sequential single-token AR decode
    // is eligible BY DEFAULT (the consume below resets to true); spec-decode /
    // MTP re-seed and the verify/prefill batch path explicitly set this FALSE
    // right before their `forward_scratch` call so the plain-AR graph can never
    // capture or replay in a non-sequential context. An ineligible call also
    // INVALIDATES any captured graph (forces re-capture on the next plain call).
    let requested_graph_eligible = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    let graph_eligible =
        ar_graph_eligible_for_kv(requested_graph_eligible, kv_cache.compact_offset);
    // Redline's plain-AR capture/replay has the same eligibility contract as
    // the AR HipGraph. MTP/spec re-seed and verify calls must not contaminate
    // or consume the immutable single-token replay sequence.
    gpu.replay.set_forward_eligible(graph_eligible);
    gpu.replay
        .begin_auto_capture_if_armed()
        .map_err(|reason| HipError::new(0, reason))?;
    if ar_graph_test && !graph_eligible {
        gpu.graphs.ar_forward_replay_enabled = false;
        gpu.graphs.ar_forward_kernel_dirty = true;
    }
    // MoE models require `experimental.graph.moe` in addition to the
    // arch/kill-switch guards. Dense models (num_experts==0) are unaffected.
    let use_graph = ar_graph_test
        && graph_enabled
        && graph_eligible
        && !gpu.replay.is_enabled()
        && (config.num_experts == 0 || allow_moe);
    let _ = gpu.graphs.ar_forward_replay_enabled; // suppress unused warning

    // Embedding lookup into scratch.x (always direct, changes per token)
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    let pos_i32 = pos as i32;
    if gpu.replay.should_route_aql() {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        let replay = unsafe { gpu.replay.replay_linear_aql(pos) };
        return match replay {
            Ok(_) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared AQL replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if gpu.replay.should_route_pm4() {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        let replay = unsafe { gpu.replay.replay_pm4(pos) };
        return match replay {
            Ok(_) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared PM4 replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if use_graph && gpu.graphs.ar_forward_replay_enabled && gpu.graphs.graph_exec.is_some() {
        // ── Replay path: graph captured + kernels clean. Cheapest path: pos
        // memcpy + graph replay. The graph is position-agnostic (pos via
        // pos_buf), so replay is correct across positions and requests as long
        // as the buffers are the plain-AR continuation — which the spec markers
        // + verify invalidation guarantee. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] replay pos={pos}");
        }
    } else if use_graph && gpu.graphs.ar_forward_kernel_dirty {
        // ── Direct path (kernel-dirty): kernels are dirty (init or post-
        // model-load). Capture would trip "hipMalloc not permitted under
        // stream capture" on the first inline JIT. Mark clean after a
        // successful direct dispatch so subsequent calls can capture. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
        gpu.graphs.ar_forward_kernel_dirty = false;
    } else if use_graph {
        // ── Capture + launch: kernels are clean but caller has not committed
        // a replay yet (or graph_exec is None). Drop any prior captured graph,
        // record a fresh one, and launch it for this forward's output. After
        // the caller signals end_decode_turn(), the most recent capture is
        // promoted to the replay graph for the next decode turn. ──
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        gpu.graphs.begin_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
        gpu.graphs.end_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] capture pos={pos}");
        }
        // Intra-generate replay (2026-06-12): promote this fresh capture to the
        // replay graph immediately so the NEXT token replays (cheap: pos memcpy
        // + graph_launch) instead of re-capturing + re-instantiating every
        // token. The per-token instantiate is why the path "did nothing" — the
        // daemon never calls end_decode_turn() to enable replay.
        gpu.graphs.ar_forward_replay_enabled = true;
    } else {
        // ── Direct path (graph not eligible: arch / MoE config) ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
    }
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip.device_synchronize()?;
        gpu.replay
            .finish_capture()
            .map_err(|reason| HipError::new(0, reason))?;
        let prepare = if gpu.replay.uses_pm4_transport() {
            let launches = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|_| ())
        };
        if let Err(reason) = prepare {
            gpu.replay
                .poison(format!("Redline prepare after warmup failed: {reason}"));
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
    Ok(())
}

/// Populate the two inputs that intentionally stay outside a prepared decode
/// replay: the token embedding in `scratch.x` and the position scalar buffer.
/// Redline uses this exact boundary before its AQL packet batch.
pub fn prepare_scratch_inputs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        other => {
            return Err(HipError::new(
                0,
                &format!("unsupported embedding format for Redline: {other:?}"),
            ));
        }
    }
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
    Ok(())
}

/// Same as `forward_scratch` but also extracts hidden states from the
/// configured target layers into `hidden_rb`. Used by the DFlash draft path
/// during target verification. `hidden_rb.advance_head()` is called once
/// automatically at the end of the forward pass.
pub fn forward_scratch_with_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_with_hidden")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        Some(hidden_rb),
        None,
    )?;
    hidden_rb.advance_head();
    Ok(())
}

/// Zero-alloc forward from pre-computed embedding in scratch.x.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_embed")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Upload embedding directly into scratch.x
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
    )
}

/// Write `(t, h, w)` for sequence position `pos` into `scratch.pos_buf3`.
///
/// `compact_offset` mirrors what the 1D path does to `pos_buf` after a
/// TriAttention compaction: absolute rope phase = physical index + offset.
/// Note the daemon does NOT build an `MropeCtx` while eviction is armed
/// (physical indices are renumbered there, so `MropeCtx::positions` — which is
/// indexed by physical position — would be misaligned), so in practice this
/// arm sees `compact_offset == 0`. It is applied anyway so the mrope branch
/// never silently disagrees with its 1D twin.
fn upload_mrope_pos3(
    gpu: &mut Gpu,
    scratch: &Qwen35Scratch,
    mrope: &MropeCtx,
    pos: usize,
    compact_offset: usize,
) -> HipResult<()> {
    let mut p = mrope.pos3(pos);
    if compact_offset > 0 {
        let off = compact_offset as i32;
        for v in p.iter_mut() {
            *v += off;
        }
    }
    let mut bytes = [0u8; 12];
    for (i, v) in p.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }
    gpu.memcpy_htod_auto(&scratch.pos_buf3, &bytes)
}

/// A mrope forward is NOT plain sequential AR — it issues a different rope
/// kernel against a different position buffer. Mirror the contract that
/// MTP / spec re-seed / verify forwards follow before their `forward_scratch`
/// call: consume-and-reset the AR eligibility flag (so a caller-set `false`
/// cannot leak into the next text forward), tell Redline this forward is
/// ineligible, and invalidate any captured plain-AR hipGraph so the next text
/// forward re-captures instead of replaying one recorded around a VL step.
fn mark_mrope_forward_ineligible(gpu: &mut Gpu) {
    let _consumed = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    gpu.replay.set_forward_eligible(false);
    gpu.graphs.ar_forward_replay_enabled = false;
    gpu.graphs.ar_forward_kernel_dirty = true;
}

/// mrope-aware [`forward_scratch`]. `mrope == None` delegates verbatim to
/// [`forward_scratch`] — same kernels, same graph/replay routing, same
/// dispatch identity, so the certified retained-PM4 tape is untouched.
///
/// With `Some(ctx)` the call takes a deliberately plain, direct-dispatch path:
/// hipGraph capture/replay and the Redline AQL/PM4 replay routes are BYPASSED.
/// Those replay a *recorded* kernel sequence keyed only by `pos_buf`; a VL step
/// issues a different rope kernel reading a different buffer, so replaying a
/// text-captured graph (or capturing a VL step into one) would be silently
/// wrong. VL prefill/decode is per-token and image requests are rare, so the
/// lost launch amortization is not worth the aliasing risk.
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_mrope(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    let Some(mc) = mrope else {
        return forward_scratch(
            gpu, weights, config, token, pos, kv_cache, dn_state, scratch,
        );
    };
    mark_mrope_forward_ineligible(gpu);
    // Embedding lookup into scratch.x + the 1D pos scalar (still consumed by
    // the KV write and flash attention, which want the PHYSICAL slot).
    prepare_scratch_inputs(gpu, weights, config, token, pos, scratch)?;
    upload_mrope_pos3(gpu, scratch, mc, pos, kv_cache.compact_offset)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        Some(mc),
    )
}

/// mrope-aware [`forward_scratch_embed`] — the image-token prefill entry.
/// `mrope == None` delegates verbatim to [`forward_scratch_embed`].
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_embed_mrope(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    let Some(mc) = mrope else {
        return forward_scratch_embed(
            gpu,
            weights,
            config,
            embedding_data,
            pos,
            kv_cache,
            dn_state,
            scratch,
        );
    };
    mark_mrope_forward_ineligible(gpu);
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    upload_mrope_pos3(gpu, scratch, mc, pos, kv_cache.compact_offset)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        Some(mc),
    )
}

// ── Forward scratch layers (dispatch family version) ────────────────────

fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    // #397 Ship 6 — forward-as-pipeline. When HIPFIRE_FORWARD_LOWERED=1, route
    // single-GPU decode through the lowered super-op executor. Skipped when a
    // hidden-state ring buffer is active (spec-decode capture engages only the
    // hand path for now). Default off → the hand arms below run unchanged.
    //
    // Also skipped when a 3D mrope context is present: the lowered executor's
    // ATTEND_FULL binding (`Qwen35Bindings::run_attend`) still issues the 1D
    // `rope_partial_interleaved_f32` / `qwen35_fa_prep_gfx1100` pair and has no
    // channel for the (t,h,w) buffer, so routing a VL request through it would
    // silently reinstate sequential positions. VL therefore always takes the
    // hand arms below, which DO branch on `mrope`.
    if forward_lowered_enabled() && hidden_rb.is_none() && mrope.is_none() {
        return forward_scratch_layers_lowered(gpu, weights, config, pos, kv_cache, dn_state, s);
    }

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── DeltaNet QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                )?;

                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                )?;

                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;

                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} LinearAttention residual"),
                    &s.x,
                )?;
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;

                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, &s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // VL (image tokens present) → 3D mrope; everything else keeps
                // the original 1D kernel and its dispatch identity.
                match mrope {
                    Some(mc) => {
                        debug_assert_eq!(
                            mc.section, config.mrope_section,
                            "MropeCtx section disagrees with the loaded config \
                             (build it with MropeCtx::new)",
                        );
                        // `pos_buf3` is filled ONCE per token by
                        // `forward_scratch_mrope` / `..._embed_mrope`, exactly
                        // like `pos_buf` — the value is layer-invariant, and a
                        // per-layer re-upload would add a blocking 12-byte H2D
                        // per full-attention layer for no benefit.
                        gpu.rope_mrope_halfsplit_f32(
                            &s.fa_q,
                            &s.fa_k,
                            &s.pos_buf3,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            n_rot,
                            config.rope_theta,
                            mc.section,
                        )?
                    }
                    None => gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?,
                }
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                let fused_epilogue = kv_cache_attention_dispatch(
                    &ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos,
                )?;

                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                {
                    let wr = layer.wo.dispatch_ref();
                    let input = if fused_epilogue {
                        GemvInput::Prerotated(&s.fa_attn_out)
                    } else {
                        GemvInput::Raw(&s.fa_attn_out)
                    };
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input,
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttention residual"),
                    &s.x,
                )?;
                kv_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // ── DeltaNetMoe QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;

                // Find GDN call location by dumping after common operations
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                )?;
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                // DIAG: dump GDN inputs (per-token)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * config.linear_key_head_dim;
                    dump_hidden_localize(gpu, &s.dn_q, 1, pos, qk_dim, 0, "q_p");
                    dump_hidden_localize(gpu, &s.dn_k, 1, pos, qk_dim, 0, "k_p");
                    dump_hidden_localize(gpu, &s.dn_v, 1, pos, v_dim, 0, "v_p");
                    dump_hidden_localize(gpu, &s.dn_alpha, 1, pos, n_v_heads, 0, "alpha_p");
                    dump_hidden_localize(gpu, &s.dn_beta, 1, pos, n_v_heads, 0, "beta_p");
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }
                // DIAG: dump GDN attention output (per-token)
                if layer_idx == 0 {
                    dump_hidden_localize(
                        gpu,
                        &s.dn_attn_out,
                        1,
                        pos,
                        n_v_heads * config.linear_value_head_dim,
                        0,
                        "gdn_p",
                    );
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s, false)?;
                // DIAG: dump MoE router logits (per-token)
                if layer_idx == 0 {
                    if let Some(ref rl) = s.moe_router_logits {
                        dump_hidden_localize(gpu, rl, 1, pos, config.num_experts, 0, "router_p");
                    }
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;

                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // VL (image tokens present) → 3D mrope; everything else keeps
                // the original 1D kernel and its dispatch identity.
                match mrope {
                    Some(mc) => {
                        debug_assert_eq!(
                            mc.section, config.mrope_section,
                            "MropeCtx section disagrees with the loaded config \
                             (build it with MropeCtx::new)",
                        );
                        // `pos_buf3` is filled ONCE per token by
                        // `forward_scratch_mrope` / `..._embed_mrope`, exactly
                        // like `pos_buf` — the value is layer-invariant, and a
                        // per-layer re-upload would add a blocking 12-byte H2D
                        // per full-attention layer for no benefit.
                        gpu.rope_mrope_halfsplit_f32(
                            &s.fa_q,
                            &s.fa_k,
                            &s.pos_buf3,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            n_rot,
                            config.rope_theta,
                            mc.section,
                        )?
                    }
                    None => gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?,
                }
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                let fused_epilogue = kv_cache_attention_dispatch(
                    &ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos,
                )?;

                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                {
                    let wr = layer.wo.dispatch_ref();
                    let input = if fused_epilogue {
                        GemvInput::Prerotated(&s.fa_attn_out)
                    } else {
                        GemvInput::Raw(&s.fa_attn_out)
                    };
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input,
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s, false)?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                kv_layer_idx += 1;
            }

            // Mismatched layer weight / type combinations are unreachable
            // (the loader guarantees alignment).
            _ => unreachable!(),
        }
        dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
    }

    // Final norm + logits into scratch.logits
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    Ok(())
}

// ── Dispatch helpers ─────────────────────────────────────────────────────

/// Helper: convert `WeightTensor.paro` (if present) to `GivensRef`.
fn paro_to_givens(p: &ParoRotation) -> GivensRef<'_> {
    GivensRef {
        pairs: &p.pairs,
        theta: &p.theta,
        scales: &p.channel_scales,
        krot: p.krot as usize,
    }
}

/// Unified QKVZA (4-way) projection via execute_steps for DeltaNet layers.
/// Covers all dtypes — the interpreter selects fused QKVZA kernels for eligible
/// dtypes via FUSED_TABLE guards; everything else falls through to per-op
/// dispatch (including ParoQ4G128 which does individual Givens-rotated GEMV calls).
/// Replaces rmsnorm_rotate_dispatch + fused_qkvza_dispatch.
#[allow(clippy::too_many_arguments)]
fn qkvza_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    dn_qkv: &GpuTensor,
    dn_z: &GpuTensor,
    dn_beta: &GpuTensor,
    dn_alpha: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wqkv.gpu_dtype);
    if rotation == RotationPlan::Givens {
        // ParoQ4G128: plain rmsnorm, then per-weight Givens rotation inside run_auto.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: wqkv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: wz.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: w_beta.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: w_alpha.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Raw(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Raw(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Raw(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Raw(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        // FWHT-rotated (MQ family) or non-rotated (HFQ, Q8, etc.) dtypes.
        // RmsnormAutomatic handles FWHT when rotation != None;
        // downstream Gemv steps use Prerotated to avoid double-FWHT.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Prerotated(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Prerotated(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Prerotated(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Prerotated(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified QKV projection via execute_steps. Covers all dtypes — the interpreter
/// selects fused kernels for eligible dtypes via FUSED_TABLE guards; everything
/// else falls through to per-op dispatch. Replaces qkv_interpret_mq +
/// fused_qkv_dispatch + their preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn qkv_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    fa_v: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wq.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: wq.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: wk.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: wv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Raw(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Raw(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Raw(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Prerotated(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Prerotated(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Prerotated(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified gate+up (FFN) projection via execute_steps. Covers all dtypes.
/// Replaces fused_gate_up_dispatch + its preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn gate_up_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    ffn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,
    x_rot: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(w_gate.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: w_gate.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: w_up.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Raw(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Raw(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Prerotated(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Prerotated(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// MoE FFN dispatch — mirrors the two-path logic from the original.
fn moe_ffn_dispatch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    defer_routed_combine: bool,
) -> HipResult<()> {
    let r = if ffn_gate_side_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        let refs = MoeScratchRef::from_scratch(s);
        moe_ffn_decode_impl(
            gpu,
            ffn,
            x,
            x,
            config,
            &refs,
            true,
            None,
            false,
            defer_routed_combine,
        )
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_with_scratch(gpu, ffn, &s.tmp, x, config, s)
    };
    r?;
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(())
}

/// EP (Ship 6 substrate-EP) variant of `moe_ffn_dispatch`: same rmsnorm/rotate +
/// MoE decode, but the routed combine + shared-down accumulate into `routed_out`
/// (a zeroed per-rank partial the EP executor all-reduces), and `skip_shared`
/// gates the shared-expert down to rank 0. Calls `moe_ffn_decode_impl` directly
/// (the `with_scratch` wrappers don't carry EP params). The residual `x` is left
/// untouched — the executor adds the all-reduced partial into it afterward.
fn moe_ffn_dispatch_ep(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    routed_out: &GpuTensor,
    skip_shared: bool,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(s);
    if ffn_all_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        moe_ffn_decode_impl(
            gpu,
            ffn,
            x,
            x,
            config,
            &refs,
            true,
            Some(routed_out),
            skip_shared,
            false,
        )
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_impl(
            gpu,
            ffn,
            &s.tmp,
            x,
            config,
            &refs,
            false,
            Some(routed_out),
            skip_shared,
            false,
        )
    }
}

/// EP (Ship 6 substrate-EP, ported from tp-mtp-prototype Stage 3e): shard a MoE
/// layer's routed experts to `rank`. Frees the non-owned experts (the memory
/// win), compacts owned to the front of `ffn.experts` (so `experts[0]` stays a
/// valid shared-AWQ representative for the batched silu/rotate helpers), and
/// rebuilds the `[2·n_exp]` device pointer tables: owned global id → its
/// (compacted) buffer ptr; **non-owned → a shared ZEROED gate_up buffer**.
/// Zeroed quant bytes dequant to +0.0 → the non-owned expert's gate_up output
/// is 0 → silu·mul = 0 → rot = 0 → down output 0, so it contributes nothing
/// through `moe_down_combine` WITHOUT any masking kernel. (The non-owned down
/// ptr is irrelevant — its input rot is already 0 — so it reuses
/// `experts[0].down`.) Router / shared expert / attention stay full (replicated
/// in EP v1). The zero buffer is leaked for v1 (lives until teardown) to avoid
/// threading a lifetime field through `Qwen35Weights`.
pub fn shard_moe_experts(
    gpu: &mut Gpu,
    ffn: &mut MoeFfnWeights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
) -> HipResult<()> {
    if ffn.packed_expert_owners.is_some() {
        return Err(HipError::new(
            0,
            "shard_moe_experts cannot post-shard packed owners; use the streaming EP load path",
        ));
    }
    debug_assert_eq!(
        ffn.experts.len(),
        n_exp,
        "shard_moe_experts expects a full-loaded expert Vec (paged EP is unsupported in v1)",
    );
    // Free non-owned experts; compact owned to the front, recording global→local.
    let old = std::mem::take(&mut ffn.experts);
    let mut compacted: Vec<ExpertWeights> = Vec::with_capacity(shard.experts_per_rank(n_exp));
    let mut local_of_global = vec![usize::MAX; n_exp];
    for (e, ew) in old.into_iter().enumerate() {
        if shard.owns_expert(rank, e) {
            local_of_global[e] = compacted.len();
            compacted.push(ew);
        } else {
            let _ = gpu.free_tensor(ew.gate_up.buf);
            if let Some(s) = ew.gate_up.awq_scale {
                let _ = gpu.free_tensor(s);
            }
            let _ = gpu.free_tensor(ew.down.buf);
            if let Some(s) = ew.down.awq_scale {
                let _ = gpu.free_tensor(s);
            }
        }
    }
    assert!(
        !compacted.is_empty(),
        "shard_moe_experts: rank {rank} owns no experts (n_exp={n_exp}, tp={})",
        shard.tp_size,
    );

    // Shared zeroed gate_up buffer for non-owned slots (same byte size as a real
    // expert's gate_up). LEAKED (mem::forget) so the ptr stays valid for the
    // model's lifetime without a Qwen35Weights field — v1 TODO: own it properly.
    let gu_bytes = compacted[0].gate_up.buf.buf.size();
    let zero_gu = gpu.zeros(&[gu_bytes / 4], DType::F32)?;
    let dummy_gu = zero_gu.buf.as_ptr() as u64;
    let dummy_dn = compacted[0].down.buf.buf.as_ptr() as u64; // rot=0 ⇒ output 0 regardless
    std::mem::forget(zero_gu);

    // Rebuild the [2·n_exp] u64 pointer tables (8 B/ptr = 2 F32 slots).
    let mut gu = vec![0u64; n_exp];
    let mut dn = vec![0u64; n_exp];
    for e in 0..n_exp {
        if shard.owns_expert(rank, e) {
            let li = local_of_global[e];
            gu[e] = compacted[li].gate_up.buf.buf.as_ptr() as u64;
            dn[e] = compacted[li].down.buf.buf.as_ptr() as u64;
        } else {
            gu[e] = dummy_gu;
            dn[e] = dummy_dn;
        }
    }
    let gu_b: Vec<u8> = gu.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_b: Vec<u8> = dn.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&ffn.expert_gate_up_ptrs.buf, &gu_b)?;
    gpu.hip.memcpy_htod(&ffn.expert_down_ptrs.buf, &dn_b)?;

    // Route A MoE-AWQ under EP: rebuild the per-expert down.awq_scale pointer
    // table over the compacted set. Non-owned slots get a valid dummy pointer
    // (compacted[0]'s scale) — they read zeroed gate_up ⇒ silu output 0 ⇒
    // 0/scale = 0 regardless, so the all-reduced sum is unaffected.
    if let Some(awq_tbl) = ffn.expert_down_awq_ptrs.as_ref() {
        let dummy_aw = compacted[0]
            .down
            .awq_scale
            .as_ref()
            .map(|s| s.buf.as_ptr() as u64)
            .unwrap_or(0);
        let mut aw = vec![dummy_aw; n_exp];
        for (e, slot) in aw.iter_mut().enumerate() {
            if shard.owns_expert(rank, e) {
                let li = local_of_global[e];
                if let Some(s) = compacted[li].down.awq_scale.as_ref() {
                    *slot = s.buf.as_ptr() as u64;
                }
            }
        }
        let aw_b: Vec<u8> = aw.iter().flat_map(|p| p.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&awq_tbl.buf, &aw_b)?;
    }

    ffn.experts = compacted;
    Ok(())
}

/// Shard every MoE layer of a replicated `Qwen35Weights` to `rank`, calling
/// [`shard_moe_experts`] on each `DeltaNetMoe` / `FullAttnMoe` layer's FFN.
/// Dense / attention-only layers are untouched. Convenience wrapper for the EP
/// load path so callers (the `forward_ep` driver / examples) never reach into
/// `LayerWeights` internals. `n_exp` is the model's routed expert count
/// (`config.num_experts`).
///
/// `reap_active` MUST be `config.reap_keep.is_some()`. REAP expert-pruning and
/// EP sharding are mutually exclusive (ds4/minimax enforce the same at expert-
/// load time): under REAP `config.num_experts` is already overridden to the
/// KEPT count, so `shard_moe_experts`' `experts.len() == n_exp` precondition
/// would pass on a pruned model and the per-rank ownership math would re-remap
/// already-compacted expert ids → silent weight corruption. Refuse up front.
pub fn shard_all_moe_layers(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    reap_active: bool,
) -> HipResult<()> {
    if reap_active {
        return Err(HipError::new(
            0,
            "qwen35: REAP keep-map + EP sharding are mutually exclusive",
        ));
    }
    for layer in weights.layers.iter_mut() {
        match layer {
            LayerWeights::DeltaNetMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            LayerWeights::FullAttnMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            _ => {}
        }
    }
    Ok(())
}

/// TriAttention tap helper (inline from original forward).
fn triattn_tap(
    gpu: &mut Gpu,
    layer_idx: usize,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &s.fa_q.buf,
        1,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(&s.fa_q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(&s.fa_k)?;
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx,
                &q_cpu[..n_q],
                Some(&k_cpu[..n_k]),
            );
        } else {
            hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
        }
    }
    Ok(())
}

/// KV cache write + attention dispatch. Inline from original.
pub(crate) fn kv_cache_attention_dispatch(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    wo: &WeightTensor,
    layer_idx: usize,
    pos: usize,
) -> HipResult<bool> {
    let plan = KvTierPlan::derive(KvTierInputs {
        pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        ..kv_cache.tier_inputs()
    })
    .map_err(|e| HipError::new(0, &e.to_string()))?;
    let q8_route = plan.write_key == hipfire_dispatch::types::KernelKey::KvWriteQ8_0
        && plan.attend_key == hipfire_dispatch::types::KernelKey::AttnFlashQ8_0;
    let asym3_route = plan.write_key == hipfire_dispatch::types::KernelKey::KvWriteAsym3
        && plan.attend_key == hipfire_dispatch::types::KernelKey::AttnFlashAsym3;
    let fused_epilogue = qwen35_fa_epilogue_enabled(gpu, config, wo) && (q8_route || asym3_route);
    let io = AttnParams {
        q: &s.fa_q,
        k: &s.fa_k,
        v: &s.fa_v,
        k_cache: &kv_cache.k_gpu[layer_idx],
        v_cache: &kv_cache.v_gpu[layer_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &s.pos_buf,
        pos,
        positions: None,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        physical_cap: kv_cache.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&s.flash_partials),
        givens_cos: kv_cache.givens_cos.as_ref(),
        givens_sin: kv_cache.givens_sin.as_ref(),
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: fused_epilogue.then_some(&s.fa_gate),
        output: &s.fa_attn_out,
    };
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    Ok(fused_epilogue)
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: qwen35 DECODE lowered path (ADDITIVE).
//
// `HIPFIRE_FORWARD_LOWERED=1` routes the single-GPU decode layer loop through
// the dispatch substrate's `run_layer_program` executor (one pre-resolved
// `LayerProgram` of coarse super-ops per layer) instead of the hand-written
// arms in `forward_scratch_layers`. The hand arms are left UNTOUCHED, so the
// default (flag off) is byte-identical to master by construction; the lowered
// path is validated byte-identical via the external committed-token md5 gate
// (`FORWARD_LOWERED=0` vs `=1`, same prompt) on the fleet before the default is
// flipped per arch. See [[project_ship6_forward_pipeline_design_2026_06_07]].
//
// The super-op handlers call the SAME helper fns the hand path uses
// (`qkv/qkvza/gate_up_via_execute_steps`, `kv_cache_attention_dispatch`,
// `moe_ffn_dispatch`, `weight_gemv_swiglu_residual`) plus the inline attend/
// recurrent/gated-norm fragments. DIAG dumps / trace_finite / hidden_rb are
// output-neutral and omitted here (hidden_rb engages only the hand path).
// ─────────────────────────────────────────────────────────────────────────

/// qwen35-local super-op opcodes, encoded into `OpBinding.weights[0].0`. The
/// `SuperOpKind` routes to the `ForwardBindings` method; the opcode disambiguates
/// *which* op of that kind within the layer (qkv vs gate_up, wo vs down, …).
mod q35_op {
    // Proj
    pub const PROJ_QKV: u32 = 0;
    pub const PROJ_QKVZA: u32 = 1;
    pub const PROJ_GATE_UP: u32 = 2;
    // Attend
    pub const ATTEND_FULL: u32 = 0;
    pub const ATTEND_DN_PREP: u32 = 1;
    // ResidualGemv
    pub const RESID_WO: u32 = 0;
    pub const RESID_DOWN_SWIGLU: u32 = 1;
    // Norm
    pub const NORM_GATED: u32 = 0;
    // Recurrent
    pub const RECUR_GDN: u32 = 0;
    // Moe
    pub const MOE_FFN: u32 = 0;
}

/// The four qwen35 decoder-layer shapes. Derived from the `LayerWeights`
/// discriminant; kept as a plain enum so `lower_variant` is pure (no GpuTensor)
/// and unit-testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Q35Variant {
    DeltaNet,
    FullAttn,
    DeltaNetMoe,
    FullAttnMoe,
}

pub(crate) fn variant_of(layer: &LayerWeights) -> Q35Variant {
    match layer {
        LayerWeights::DeltaNet(_) => Q35Variant::DeltaNet,
        LayerWeights::FullAttn(_) => Q35Variant::FullAttn,
        LayerWeights::DeltaNetMoe(_) => Q35Variant::DeltaNetMoe,
        LayerWeights::FullAttnMoe(_) => Q35Variant::FullAttnMoe,
    }
}

#[inline]
fn q35_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one qwen35 decoder layer to a coarse-super-op `LayerProgram`. The op
/// SEQUENCE mirrors the matching hand arm in `forward_scratch_layers` exactly
/// (per the decode-forward variant map). Pure → unit-testable.
pub(crate) fn lower_variant(v: Q35Variant) -> LayerProgram {
    use q35_op::*;
    use SuperOpKind::{Attend, Moe, Norm, Proj, Recurrent, ResidualGemv};
    match v {
        Q35Variant::DeltaNet => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::FullAttn => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::DeltaNetMoe => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
        Q35Variant::FullAttnMoe => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn qkv_from_prerotated_mq(
    gpu: &mut Gpu,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    x_rot: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
) -> HipResult<()> {
    gpu.fused_qkv_hfq4g256(
        &wq.buf, &wk.buf, &wv.buf, x_rot, q, k, v, wq.m, wk.m, wv.m, wq.k,
    )
}

#[allow(clippy::too_many_arguments)]
fn qkvza_from_prerotated_mq(
    gpu: &mut Gpu,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    x_rot: &GpuTensor,
    qkv: &GpuTensor,
    z: &GpuTensor,
    beta: &GpuTensor,
    alpha: &GpuTensor,
) -> HipResult<()> {
    gpu.fused_qkvza_hfq4g256(
        &wqkv.buf,
        &wz.buf,
        &w_beta.buf,
        &w_alpha.buf,
        x_rot,
        qkv,
        z,
        beta,
        alpha,
        wqkv.m,
        wz.m,
        w_beta.m,
        w_alpha.m,
        wqkv.k,
    )
}

/// Per-layer execution context for the lowered decode path. Holds the current
/// layer's weights + shared scratch/state by reference; rebuilt each layer
/// iteration so the borrows stay scoped. `kv_cache` is the only `&mut` (DeltaNet
/// state is mutated through interior-mutable GpuTensor buffers via shared refs).
pub(crate) struct Qwen35Bindings<'a> {
    pub(crate) layer: &'a LayerWeights,
    pub(crate) s: &'a Qwen35Scratch,
    pub(crate) config: &'a Qwen35Config,
    pub(crate) kv_cache: &'a mut llama::KvCache,
    pub(crate) dn_state: &'a DeltaNetState,
    pub(crate) pos: usize,
    pub(crate) layer_idx: usize,
    pub(crate) delta_layer_idx: usize,
    pub(crate) k_dim: usize,
    pub(crate) v_dim: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) hd: usize,
    pub(crate) precomputed_attn_x_rot: bool,
    pub(crate) fa_output_prerotated: bool,
    pub(crate) defer_routed_combine: bool,
}

fn op_code(op: &OpBinding) -> u32 {
    op.weights.first().map(|w| w.0).unwrap_or(u32::MAX)
}

impl<'a> ForwardBindings for Qwen35Bindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = match op_code(op) {
            q35_op::PROJ_QKV => match self.layer {
                LayerWeights::FullAttn(l) => {
                    if self.precomputed_attn_x_rot {
                        qkv_from_prerotated_mq(
                            gpu,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                        )
                    } else {
                        qkv_via_execute_steps(
                            gpu,
                            ctx,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &l.attn_norm,
                            &s.x,
                            &s.tmp,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            config.norm_eps,
                        )
                    }
                }
                LayerWeights::FullAttnMoe(l) => {
                    if self.precomputed_attn_x_rot {
                        qkv_from_prerotated_mq(
                            gpu,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                        )
                    } else {
                        qkv_via_execute_steps(
                            gpu,
                            ctx,
                            &l.wq,
                            &l.wk,
                            &l.wv,
                            &l.attn_norm,
                            &s.x,
                            &s.tmp,
                            &s.x_rot,
                            &s.fa_q_full,
                            &s.fa_k,
                            &s.fa_v,
                            config.norm_eps,
                        )
                    }
                }
                _ => return Err(DispatchError::Hip("PROJ_QKV on non-FullAttn layer".into())),
            },
            q35_op::PROJ_QKVZA => {
                let (wqkv, wz, w_beta, w_alpha, attn_norm, dt_bias, a_log) = match self.layer {
                    LayerWeights::DeltaNet(l) => (
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                        &l.attn_norm,
                        &l.dt_bias,
                        &l.a_log,
                    ),
                    LayerWeights::DeltaNetMoe(l) => (
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                        &l.attn_norm,
                        &l.dt_bias,
                        &l.a_log,
                    ),
                    _ => {
                        return Err(DispatchError::Hip(
                            "PROJ_QKVZA on non-DeltaNet layer".into(),
                        ));
                    }
                };
                if self.precomputed_attn_x_rot {
                    qkvza_from_prerotated_mq(
                        gpu,
                        wqkv,
                        wz,
                        w_beta,
                        w_alpha,
                        &s.x_rot,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                    )
                } else if qkvza_scalar_prep_enabled(
                    gpu,
                    config,
                    self.n_v_heads,
                    self.dn_state.quant,
                    wqkv,
                    wz,
                    w_beta,
                    w_alpha,
                ) {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        wqkv,
                        &s.x,
                        attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    let eff_x = x_rot.unwrap_or(&s.tmp);
                    gpu.fused_qkvza_hfq4g256_scalar_prep_gfx1100(
                        &wqkv.buf,
                        &wz.buf,
                        &w_beta.buf,
                        &w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        dt_bias,
                        a_log,
                        wqkv.m,
                        wz.m,
                        w_beta.m,
                        w_alpha.m,
                        wqkv.k,
                    )
                } else {
                    qkvza_via_execute_steps(
                        gpu,
                        ctx,
                        wqkv,
                        wz,
                        w_beta,
                        w_alpha,
                        attn_norm,
                        &s.x,
                        &s.tmp,
                        &s.x_rot,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        config.norm_eps,
                    )
                }
            }
            q35_op::PROJ_GATE_UP => match self.layer {
                LayerWeights::DeltaNet(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                LayerWeights::FullAttn(l) => gate_up_via_execute_steps(
                    gpu,
                    ctx,
                    &l.w_gate,
                    &l.w_up,
                    &l.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                ),
                _ => {
                    return Err(DispatchError::Hip(
                        "PROJ_GATE_UP on MoE/unknown layer".into(),
                    ));
                }
            },
            other => return Err(DispatchError::Hip(format!("unknown PROJ opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::RESID_WO => {
                let (wo, input) = match self.layer {
                    LayerWeights::FullAttn(l) => {
                        let input = if self.fa_output_prerotated {
                            GemvInput::Prerotated(&s.fa_attn_out)
                        } else {
                            GemvInput::Raw(&s.fa_attn_out)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::FullAttnMoe(l) => {
                        let input = if self.fa_output_prerotated {
                            GemvInput::Prerotated(&s.fa_attn_out)
                        } else {
                            GemvInput::Raw(&s.fa_attn_out)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::DeltaNet(l) => {
                        let input = if gated_norm_mq_rotate_enabled(
                            gpu,
                            self.config,
                            self.n_v_heads,
                            &l.wo,
                        ) {
                            GemvInput::Prerotated(&s.x_rot)
                        } else {
                            GemvInput::Raw(&s.dn_normed)
                        };
                        (&l.wo, input)
                    }
                    LayerWeights::DeltaNetMoe(l) => {
                        let input = if gated_norm_mq_rotate_enabled(
                            gpu,
                            self.config,
                            self.n_v_heads,
                            &l.wo,
                        ) {
                            GemvInput::Prerotated(&s.x_rot)
                        } else {
                            GemvInput::Raw(&s.dn_normed)
                        };
                        (&l.wo, input)
                    }
                };
                let wr = wo.dispatch_ref();
                execute_steps(
                    gpu,
                    ctx,
                    &[Step::GemvResidual {
                        w: &wr,
                        input,
                        residual: &s.x,
                        out: &s.x,
                    }],
                )
                .map_err(|e| HipError::new(0, &e.to_string()))
            }
            q35_op::RESID_DOWN_SWIGLU => {
                let w_down = match self.layer {
                    LayerWeights::DeltaNet(l) => &l.w_down,
                    LayerWeights::FullAttn(l) => &l.w_down,
                    _ => return Err(HipError::new(0, "RESID_DOWN_SWIGLU on MoE layer")),
                };
                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu,
                    w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )
            }
            other => Err(HipError::new(0, &format!("unknown RESID opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (norm_weight, wo) = match self.layer {
            LayerWeights::DeltaNet(l) => (&l.norm_weight, &l.wo),
            LayerWeights::DeltaNetMoe(l) => (&l.norm_weight, &l.wo),
            _ => {
                return Err(DispatchError::Hip(
                    "NORM_GATED on non-DeltaNet layer".into(),
                ));
            }
        };
        if gated_norm_mq_rotate_enabled(gpu, config, self.n_v_heads, wo) {
            gpu.gated_norm_rotate_mq_gfx1100(
                &s.dn_attn_out,
                &s.dn_z,
                norm_weight,
                &s.x_rot,
                self.n_v_heads,
                config.linear_value_head_dim,
                config.norm_eps,
            )
        } else {
            gpu.gated_norm_f32(
                &s.dn_attn_out,
                &s.dn_z,
                norm_weight,
                &s.dn_normed,
                self.n_v_heads,
                config.linear_value_head_dim,
                config.norm_eps,
            )
        }
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let res: HipResult<()> = (|| match op_code(op) {
            q35_op::ATTEND_FULL => {
                let (q_norm, k_norm, wo) = match self.layer {
                    LayerWeights::FullAttn(l) => (&l.q_norm, &l.k_norm, &l.wo),
                    LayerWeights::FullAttnMoe(l) => (&l.q_norm, &l.k_norm, &l.wo),
                    _ => return Err(HipError::new(0, "ATTEND_FULL on non-FullAttn layer")),
                };
                let tap_enabled = hipfire_runtime::triattn::tap_enabled();
                let fused_prep = qwen35_fa_prep_enabled(gpu, config) && !tap_enabled;
                if !fused_prep {
                    gpu.deinterleave_f32(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                }
                if tap_enabled {
                    triattn_tap(gpu, self.layer_idx, s, config)?;
                }
                if self.kv_cache.compact_offset > 0 {
                    let abs = (self.pos + self.kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                if fused_prep {
                    gpu.qwen35_fa_prep_gfx1100(
                        &s.fa_q_full,
                        &s.fa_q,
                        &s.fa_gate,
                        &s.fa_k,
                        q_norm,
                        k_norm,
                        &s.pos_buf,
                        config.norm_eps,
                        config.rope_theta,
                        config.n_heads,
                        config.n_kv_heads,
                    )?;
                } else {
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                }
                if self.kv_cache.compact_offset > 0 {
                    let phys = self.pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }
                let fused_epilogue = kv_cache_attention_dispatch(
                    ctx,
                    gpu,
                    self.kv_cache,
                    s,
                    config,
                    wo,
                    self.layer_idx,
                    self.pos,
                )?;
                if !fused_epilogue {
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                }
                self.fa_output_prerotated = fused_epilogue;
                Ok(())
            }
            q35_op::ATTEND_DN_PREP => {
                let (dt_bias, a_log, conv_weight, wqkv, wz, w_beta, w_alpha) = match self.layer {
                    LayerWeights::DeltaNet(l) => (
                        &l.dt_bias,
                        &l.a_log,
                        &l.conv_weight,
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                    ),
                    LayerWeights::DeltaNetMoe(l) => (
                        &l.dt_bias,
                        &l.a_log,
                        &l.conv_weight,
                        &l.wqkv,
                        &l.wz,
                        &l.w_beta,
                        &l.w_alpha,
                    ),
                    _ => return Err(HipError::new(0, "ATTEND_DN_PREP on non-DeltaNet layer")),
                };
                let qkvza_scalar_prep = qkvza_scalar_prep_enabled(
                    gpu,
                    config,
                    self.n_v_heads,
                    self.dn_state.quant,
                    wqkv,
                    wz,
                    w_beta,
                    w_alpha,
                );
                let conv_scalar_prep = !qkvza_scalar_prep
                    && conv_scalar_prep_enabled(gpu, config, self.n_v_heads, self.dn_state.quant);
                if !qkvza_scalar_prep && !conv_scalar_prep {
                    gpu.fused_sigmoid_alpha_gate_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        dt_bias,
                        a_log,
                        self.n_v_heads,
                    )?;
                }
                if conv_qknorm_enabled(gpu, config, self.dn_state.quant) {
                    if conv_scalar_prep {
                        gpu.conv1d_silu_split_qknorm_scalar_prep_gfx1100(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_v,
                            &s.dn_qkv,
                            conv_weight,
                            &self.dn_state.conv_states[self.delta_layer_idx],
                            &s.dn_beta,
                            &s.dn_alpha,
                            dt_bias,
                            a_log,
                            self.k_dim,
                            self.v_dim,
                            config.linear_num_key_heads,
                            self.hd,
                            1.0 / (self.hd as f32).sqrt(),
                            config.norm_eps,
                            self.n_v_heads,
                        )?;
                    } else {
                        gpu.conv1d_silu_split_qknorm(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_v,
                            &s.dn_qkv,
                            conv_weight,
                            &self.dn_state.conv_states[self.delta_layer_idx],
                            self.k_dim,
                            self.v_dim,
                            config.linear_num_key_heads,
                            self.hd,
                            1.0 / (self.hd as f32).sqrt(),
                            config.norm_eps,
                        )?;
                    }
                } else {
                    gpu.conv1d_silu_split_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        conv_weight,
                        &self.dn_state.conv_states[self.delta_layer_idx],
                        self.k_dim,
                        self.v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        self.hd,
                        1.0 / (self.hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                }
                if gdn_compact_qk_div(gpu, config, self.n_v_heads, self.dn_state.quant).is_some() {
                    // The compact Q8 recurrence maps each state head directly
                    // to its shared Q/K head. Leave the normalized tensors
                    // compact and remove this materialization dispatch.
                } else if config.linear_num_key_heads < self.n_v_heads {
                    let ratio = self.n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        self.hd,
                    )?;
                } else {
                    // Keep the ratio-1 copy on the same kernel-dispatch
                    // surface as ratio>1. One deterministic Q+K kernel
                    // replaces two runtime memcpy nodes and makes the entire
                    // lowered decode body recordable by Redline AQL.
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        1,
                        self.hd,
                    )?;
                }
                Ok(())
            }
            other => Err(HipError::new(0, &format!("unknown ATTEND opcode {other}"))),
        })();
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        moe_ffn_dispatch(
            gpu,
            ffn,
            &s.x,
            ffn_norm,
            config,
            s,
            self.defer_routed_combine,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        skip_shared: bool,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Routed combine + shared-down (rank 0 only) accumulate into `routed_out`
        // (zeroed by the EP executor); s.x (the replicated attention residual) is
        // untouched until ep_add_into_residual after the all-reduce.
        moe_ffn_dispatch_ep(gpu, ffn, &s.x, ffn_norm, config, s, routed_out, skip_shared)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        // s.x += the all-reduced routed partial (the EP MoE output summed across
        // ranks). Mirrors the prototype's `tp_allreduce_add` residual step.
        let s = self.s;
        gpu.add_inplace_f32(&s.x, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_recurrent(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let dn = self.dn_state;
        let i = self.delta_layer_idx;
        let res: HipResult<()> = match dn.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
            StateQuant::Q8 => {
                if let Some(qk_head_div) = gdn_compact_qk_div(gpu, config, self.n_v_heads, dn.quant)
                {
                    gpu.gated_delta_net_q8_compact(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn.s_matrices[i],
                        &dn.s_scales[i],
                        &s.dn_attn_out,
                        1,
                        self.n_v_heads,
                        config.linear_value_head_dim,
                        qk_head_div,
                        dn.ef_residual(i),
                    )
                } else {
                    gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn.s_matrices[i],
                        &dn.s_scales[i],
                        &s.dn_attn_out,
                        1,
                        self.n_v_heads,
                        config.linear_value_head_dim,
                        dn.ef_residual(i),
                    )
                }
            }
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                &s.dn_q,
                &s.dn_k,
                &s.dn_v,
                &s.dn_alpha,
                &s.dn_beta,
                &dn.s_matrices[i],
                &dn.s_scales[i],
                &s.dn_attn_out,
                1,
                self.n_v_heads,
                config.linear_value_head_dim,
            ),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("qwen35 has no Conv super-op".into()))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "qwen35 has no Escape super-op ({kind:?})"
        )))
    }
}

/// Cached `HIPFIRE_FORWARD_LOWERED` toggle. #397 Ship 6: the qwen35 single-GPU
/// decode lowered path is **DEFAULT ON** as of 2026-06-07 — validated byte-
/// identical to the hand path via fleet decode byte-parity (RDNA3 k9lin / RDNA4
/// hiptrx / RDNA3.5 hipx, dense + MoE) and the full coherence battery (13 cases,
/// k9lin). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand arms
/// (still present in forward_scratch_layers); any other value (or unset) → lowered.
fn forward_lowered_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Exact gfx1151 admission gate for its certified Radiowave decode bundle.
/// Keep this separate from broad RDNA3 capability checks so no neighboring
/// architecture can inherit the gfx1151 schedules.
fn gfx1151_radiowave_fusions_enabled(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1151()
}

/// Decode path that keeps DeltaNet Q/K at their native head count and
/// lets each pair of value/state heads reuse one Q/K head. The architecture,
/// Q8-state, and 2:1-head gates keep every other configuration isolated; the
/// environment variable is a fail-closed escape hatch. The compact B2 schedule
/// is the certified default on gfx1100 and gfx1201; other architectures remain
/// isolated from the experiment.
fn gdn_compact2_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> bool {
    let mode = hipfire_config::developer_var("HIPFIRE_GDN_COMPACT2").ok();
    let arch_enabled = (gpu.arch_caps.is_gfx1201()
        || gpu.arch_caps.arch() == "gfx1100"
        || gfx1151_radiowave_fusions_enabled(gpu))
        && mode.as_deref() != Some("0");
    arch_enabled && quant == StateQuant::Q8 && config.linear_num_key_heads * 2 == n_v_heads
}

/// 3:1 compact-QK route for Qwen3.6-27B dense. A 512-token graph-on A/B/B/A
/// measured +0.45%, while profiling confirmed 853 -> 805 dispatches/token and
/// a net 64.35 us/token reduction in GPU compute. Set
/// `HIPFIRE_GDN_COMPACT3=0` to restore explicit Q/K materialization.
fn gdn_compact3_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_GDN_COMPACT3")
            .ok()
            .as_deref()
            != Some("0")
    });
    enabled
        && gpu.arch_caps.is_gfx1100()
        && quant == StateQuant::Q8
        && config.linear_num_key_heads * 3 == n_v_heads
        && super::config::qwen36_27b_dense_shape(config, n_v_heads)
}

fn gdn_compact_qk_div(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> Option<usize> {
    if gdn_compact2_enabled(gpu, config, n_v_heads, quant) {
        Some(2)
    } else if gdn_compact3_enabled(gpu, config, n_v_heads, quant) {
        Some(3)
    } else {
        None
    }
}

/// Keep DeltaNet's normalized output on chip and feed the exact MQ rotation
/// directly from LDS. Each pair of 128-value heads forms one 256-value MQ
/// group without changing either operation's arithmetic order. The 32-head
/// A3B route remains isolated from the gfx1100-only 48-head Qwen3.6-27B route;
/// the latter measured +0.45% over a 512-token A/B/B/A and removes 48
/// dispatches/token. Set `HIPFIRE_GATED_NORM_MQ_ROTATE=0` to restore both
/// explicit operations.
fn gated_norm_mq_rotate_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    wo: &WeightTensor,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_GATED_NORM_MQ_ROTATE")
            .ok()
            .as_deref()
            != Some("0")
    });
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100()
        || gfx1151_radiowave_fusions_enabled(gpu))
        && config.dim == 2_048
        && n_v_heads == 32)
        || (gpu.arch_caps.is_gfx1100() && super::config::qwen36_27b_dense_shape(config, n_v_heads));
    enabled
        && admitted_arch_shape
        && config.linear_value_head_dim == 128
        && wo.k == n_v_heads * config.linear_value_head_dim
        && wo.gpu_dtype == DType::MQ4G256
        && wo.awq_scale.is_none()
}

/// Collapse full-attention Q/gate deinterleave, Q/K RMS normalization, and
/// partial half-split RoPE into one head-local launch on the certified gfx1100
/// shapes. The Qwen3.6-27B q24k4 route measured +0.64% over a 512-token
/// A/B/B/A and reduced 757 -> 709 dispatches/token. Set
/// `HIPFIRE_QWEN35_FA_PREP_FUSE=0` to retain the legacy path.
/// The legacy interleaved-RoPE compatibility mode and diagnostic tap retain
/// the established multi-dispatch path.
fn qwen35_fa_prep_enabled(gpu: &Gpu, config: &Qwen35Config) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QWEN35_FA_PREP_FUSE")
            .ok()
            .as_deref()
            != Some("0")
    });
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100()
        || gfx1151_radiowave_fusions_enabled(gpu))
        && config.n_heads == 16
        && config.n_kv_heads == 2)
        || (gpu.arch_caps.is_gfx1100()
            && super::config::qwen36_27b_dense_shape(config, config.linear_num_value_heads)
            && config.n_heads == 24
            && config.n_kv_heads == 4);
    enabled
        && admitted_arch_shape
        && !gpu.flags.rope_interleaved_legacy
        && config.head_dim == 256
        && n_rot == 64
}

/// Fold the Qwen output gate plus MQ rotation into the flash-attention reduce
/// epilogue on certified gfx1100/MQ4 shapes. Extending the existing Q8 reducer
/// to Qwen3.6-27B's asym3 route measured +0.37% over a 512-token A/B/B/A and
/// reduced 709 -> 677 dispatches/token; a 1025-token replay remained exact. Set
/// `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE=0` to retain the legacy path.
fn qwen35_fa_epilogue_enabled(gpu: &Gpu, config: &Qwen35Config, wo: &WeightTensor) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QWEN35_FA_EPILOGUE_FUSE")
            .ok()
            .as_deref()
            != Some("0")
    });
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100()
        || gfx1151_radiowave_fusions_enabled(gpu))
        && config.n_heads == 16
        && config.n_kv_heads == 2)
        || (gpu.arch_caps.is_gfx1100()
            && super::config::qwen36_27b_dense_shape(config, config.linear_num_value_heads)
            && config.n_heads == 24
            && config.n_kv_heads == 4);
    enabled
        && admitted_arch_shape
        && config.head_dim == 256
        && wo.gpu_dtype == DType::MQ4G256
        && wo.awq_scale.is_none()
}

/// Exact dense Qwen3.6-27B shape shared by its gfx1100 decode selectors.
/// Defined in `super::config` as single source; this wrapper preserves
/// the PR's local symbol for selectors that historically lived in the same file.
fn qwen36_27b_dense_shape(config: &Qwen35Config, n_v_heads: usize) -> bool {
    super::config::qwen36_27b_dense_shape(config, n_v_heads)
}

/// Radiowave experiment: fold the DeltaNet beta/alpha scalar preparation into
/// the fixed-K QKVZA producer. Keep the gate deliberately narrow until exact
/// replay and stationary product certification justify a default flip.
#[allow(clippy::too_many_arguments)]
fn qkvza_scalar_prep_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QKVZA_SCALAR_PREP")
            .ok()
            .as_deref()
            == Some("1")
    });
    let dtype = wqkv.gpu_dtype;
    enabled
        && gpu.arch_caps.is_gfx1100()
        && gdn_compact2_enabled(gpu, config, n_v_heads, quant)
        && wqkv.k == 2_048
        && w_beta.m == n_v_heads
        && w_alpha.m == n_v_heads
        && wz.gpu_dtype == dtype
        && w_beta.gpu_dtype == dtype
        && w_alpha.gpu_dtype == dtype
        && matches!(dtype, DType::MQ4G256 | DType::HFQ4G256)
}

/// Schedule the independent beta/alpha transforms as one extra workgroup of
/// the following conv/QK-normalization dispatch. This keeps the hot QKVZA
/// projection unchanged while deleting the same boundary.
fn conv_scalar_prep_enabled(
    gpu: &Gpu,
    config: &Qwen35Config,
    n_v_heads: usize,
    quant: StateQuant,
) -> bool {
    static MODE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let mode = MODE.get_or_init(|| hipfire_config::developer_var("HIPFIRE_CONV_SCALAR_PREP").ok());
    let enabled = match mode.as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => {
            // Qwen3.6-27B dense on W7900/gfx1100: 512-token graph-on
            // A/B/B/A measured 35.376->35.657 tok/s (+0.79%) and
            // 28.105->27.891 ms p50 (-0.76%). The fused route deletes
            // 48 dispatches/token and stayed exact for 1,025 greedy tokens.
            super::config::qwen36_27b_dense_shape(config, n_v_heads)
        }
    };
    let shape = hipfire_config::developer_var("HIPFIRE_CONV_QKNORM_SHAPE").ok();
    enabled
        && gpu.arch_caps.is_gfx1100()
        && n_v_heads <= 256
        && shape.as_deref().is_none_or(|v| v == "b256")
        && conv_qknorm_enabled(gpu, config, quant)
}

fn conv_qknorm_enabled(gpu: &Gpu, config: &Qwen35Config, quant: StateQuant) -> bool {
    let mode = hipfire_config::developer_var("HIPFIRE_CONV_QKNORM").ok();
    let arch_enabled = (gpu.arch_caps.is_gfx1201()
        || gpu.arch_caps.arch() == "gfx1100"
        || gfx1151_radiowave_fusions_enabled(gpu))
        && mode.as_deref() != Some("0");
    arch_enabled && quant == StateQuant::Q8 && config.linear_key_head_dim == 128
}

/// Certified gfx1100 Radiowave schedule: retain each non-final routed-MoE
/// result in expanded scratch and let the next layer combine it while producing
/// that layer's normalized MQ activation. The whole-model admission check is
/// intentionally strict so every deferred producer has exactly one compatible
/// consumer and no fallback projection can observe an incomplete residual.
/// Enabled by default for the admitted mq4r shape; set
/// `HIPFIRE_MOE_COMBINE_NEXT_RMS=0` to restore the two-dispatch schedule.
fn moe_combine_next_rms_enabled(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config) -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let requested = *REQUESTED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS")
            .ok()
            .as_deref()
            != Some("0")
    });
    if !requested
        || !(gpu.arch_caps.is_gfx1100() || gfx1151_radiowave_fusions_enabled(gpu))
        || config.dim != 2_048
        || config.num_experts_per_tok != 8
        || config.n_layers < 2
        || config.n_layers != weights.layers.len()
        || weights.pager.is_some()
        || hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
        || hipfire_config::developer_var("HIPFIRE_QKVZA_SCALAR_PREP").as_deref() == Ok("1")
    {
        return false;
    }

    let mq4 =
        |w: &WeightTensor| w.gpu_dtype == DType::MQ4G256 && w.k == 2_048 && w.awq_scale.is_none();
    // MUST stay in lockstep with `routed_down_self_combines` in
    // hipfire-dispatch/src/pipeline/mod.rs — this asks the same question ("does
    // this layer's down GEMV write `down_expanded`?") and is the admission gate
    // for `defer_routed_combine`. Every dtype whose down GEMV self-combines via
    // atomicAdd into x_residual writes NO expanded buffer and must return false
    // here. That is the whole codebook family: MQ2/MQ3-Lloyd AND their
    // global-codebook siblings MQ2/MQ3-GL.
    //
    // Getting this wrong is SILENT. A GL layer that wrongly claims an expanded
    // buffer lets the next layer fold stale `down_expanded` contents into the
    // residual a second time, scaled by the wrong topk_weights — no error, no
    // crash. Uniform-GL models are masked by the `gpu.zeros` memset on
    // `s.moe_down_expanded` (folding 0.0 is a no-op), but a per-layer TIER_MAP
    // mixing MQ4 and MQ2GL/MQ3GL layers is NOT masked and double-counts on every
    // GL -> next-layer transition. This is the third copy of this invariant in
    // the tree and the second to drift; if a fourth appears, hoist it onto DType.
    let has_expanded_down = |ffn: &MoeFfnWeights| {
        ffn.expert_dtype_tags.is_some()
            || ffn.experts.first().is_some_and(|expert| {
                !matches!(
                    expert.down.gpu_dtype,
                    DType::MQ2G256Lloyd | DType::MQ3G256Lloyd | DType::MQ2G256GL | DType::MQ3G256GL
                )
            })
    };
    weights.layers.iter().all(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wqkv)
                && mq4(&l.wz)
                && mq4(&l.w_beta)
                && mq4(&l.w_alpha)
        }
        LayerWeights::FullAttnMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wq)
                && mq4(&l.wk)
                && mq4(&l.wv)
        }
        LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => false,
    })
}

/// Lowered (#397 Ship 6) single-GPU decode layer loop. Behaviorally equivalent
/// to `forward_scratch_layers`'s hand arms (validated byte-identical via the
/// external committed-token md5 gate). Builds a coarse-super-op `LayerProgram`
/// per layer and runs it through the dispatch substrate's executor.
fn forward_scratch_layers_lowered(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);
    let mut delta_layer_idx = 0usize;
    let combine_next_rms = moe_combine_next_rms_enabled(gpu, weights, config);
    let reuse_fused_rotation =
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM").as_deref() != Ok("1");

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let fused_transition = combine_next_rms && layer_idx > 0;
        if fused_transition {
            let attn_norm = match layer {
                LayerWeights::DeltaNetMoe(l) => &l.attn_norm,
                LayerWeights::FullAttnMoe(l) => &l.attn_norm,
                LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => {
                    unreachable!("moe_combine_next_rms_enabled admits only all-MoE models")
                }
            };
            gpu.moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100(
                s.moe_down_expanded.as_ref().expect("MoE scratch"),
                s.moe_topk_weights.as_ref().expect("MoE scratch"),
                &s.x,
                attn_norm,
                &s.x_rot,
                config.dim,
                config.num_experts_per_tok,
                config.norm_eps,
            )?;
            dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx - 1, "pertoken");
        }
        let program = lower_variant(variant_of(layer));
        {
            let mut bind = Qwen35Bindings {
                layer,
                s,
                config,
                kv_cache: &mut *kv_cache,
                dn_state,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
                precomputed_attn_x_rot: fused_transition && reuse_fused_rotation,
                fa_output_prerotated: false,
                defer_routed_combine: combine_next_rms && layer_idx + 1 < config.n_layers,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| HipError::new(0, &e.to_string()))?;
        }
        if matches!(
            layer,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
        if !combine_next_rms || layer_idx + 1 == config.n_layers {
            dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
        }
    }

    // Final norm + logits into scratch.logits (mirrors forward_scratch_layers).
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}

/// Forward pass returning logits ON GPU (no download). Caller must free the tensor.
/// Use with gpu.sample_top_p() after applying CPU-side n-gram blocking via download/modify/upload.
pub fn forward_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }
    forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Run one step with a pre-computed embedding vector (for VL visual token injection).
/// embedding_data: [dim] F32 values on CPU — uploaded to GPU as the initial hidden state.
pub fn forward_with_embedding(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let x = gpu.upload_f32(embedding_data, &[config.dim])?;
    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}
#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_dispatch::pipeline::superop::SuperOpKind;

    #[test]
    fn x_rot_covers_deltanet_value_width_for_moe_configs() {
        assert_eq!(qwen35_x_rot_len(2048, 0, 4096), 4096);
        assert_eq!(qwen35_x_rot_len(2048, 8192, 4096), 8192);
    }

    // ── #397 Ship 6 — lowered decode super-op program shapes ──────────────
    // The lowered LayerProgram per variant must mirror the hand-arm op sequence
    // in forward_scratch_layers exactly. These are CPU-pure (no GPU/GpuTensor).
    #[test]
    fn lowered_fullattn_program_shape() {
        use SuperOpKind::{Attend, Proj, ResidualGemv};
        let p = lower_variant(Q35Variant::FullAttn);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Proj, Attend, ResidualGemv, Proj, ResidualGemv]);
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKV);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_FULL);
        assert_eq!(p[2].binding.weights[0].0, q35_op::RESID_WO);
        assert_eq!(p[3].binding.weights[0].0, q35_op::PROJ_GATE_UP);
        assert_eq!(p[4].binding.weights[0].0, q35_op::RESID_DOWN_SWIGLU);
    }

    #[test]
    fn lowered_deltanet_program_shape() {
        use SuperOpKind::{Attend, Norm, Proj, Recurrent, ResidualGemv};
        let p = lower_variant(Q35Variant::DeltaNet);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Proj,
                Attend,
                Recurrent,
                Norm,
                ResidualGemv,
                Proj,
                ResidualGemv
            ]
        );
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKVZA);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_DN_PREP);
    }

    #[test]
    fn lowered_moe_variants_replace_dense_ffn_with_one_moe_op() {
        use SuperOpKind::Moe;
        let dn = lower_variant(Q35Variant::DeltaNetMoe);
        let fa = lower_variant(Q35Variant::FullAttnMoe);
        // MoE variants end in a single Moe super-op (no dense gate_up/down).
        assert_eq!(dn.last().unwrap().kind, Moe);
        assert_eq!(fa.last().unwrap().kind, Moe);
        assert!(
            dn.iter()
                .all(|o| o.binding.weights[0].0 != q35_op::PROJ_GATE_UP
                    || o.kind != SuperOpKind::Proj)
        );
        // FullAttnMoe is the shortest: Proj, Attend, ResidualGemv(wo), Moe.
        assert_eq!(fa.len(), 4);
        assert_eq!(dn.len(), 6);
    }

    #[test]
    fn lowered_variant_of_maps_layer_discriminant() {
        // variant_of is a thin discriminant map; assert the program lengths it
        // would produce per the documented layer shapes.
        assert_eq!(lower_variant(Q35Variant::FullAttn).len(), 5);
        assert_eq!(lower_variant(Q35Variant::DeltaNet).len(), 7);
        assert_eq!(lower_variant(Q35Variant::DeltaNetMoe).len(), 6);
        assert_eq!(lower_variant(Q35Variant::FullAttnMoe).len(), 4);
    }

    #[test]
    fn ar_graph_is_ineligible_after_kv_compaction() {
        assert!(ar_graph_eligible_for_kv(true, 0));
        assert!(!ar_graph_eligible_for_kv(true, 1));
        assert!(!ar_graph_eligible_for_kv(true, 128));
        assert!(!ar_graph_eligible_for_kv(false, 0));
    }
}
