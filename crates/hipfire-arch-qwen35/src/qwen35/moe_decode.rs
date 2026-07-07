// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 MoE FFN decode path (batch=1): expert routing, top-k selection,
//! paged-expert residency, and the `moe_ffn_decode` / `moe_ffn_decode_impl`
//! dispatch. On the decode hot path.

use super::*;

// ─── MoE FFN (decode, batch=1) ──────────────────────────────────────────

/// Construct a non-owning `GpuTensor` view over `[offset_elems,
/// offset_elems + len_elems)` of `src`. Valid only for F32 (4 bytes/elem).
/// The view MUST NOT outlive `src` — it shares the same GPU pointer.
#[inline]
fn slice_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    unsafe {
        let base = src.buf.as_ptr() as *mut u8;
        let ptr = base.add(offset_elems * 4);
        GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
            shape: vec![len_elems],
            dtype: DType::F32,
        }
    }
}

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
pub(crate) struct MoeScratchRef<'a> {
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
    // [k_top × dim] f32 — per-(expert-rank) MoE down output buffer for
    // the atomic-free expand+combine decode path. Mirrors the prefill
    // `pbs.moe_down_expanded_batch` layout with batch=1. Required so
    // the MoE FFN is byte-deterministic under hipGraph replay; see
    // task #100 root-cause notes in `forward_scratch`.
    down_expanded: &'a GpuTensor,
}

impl<'a> MoeScratchRef<'a> {
    /// View into a Qwen35Scratch's MoE fields. Panics if the caller didn't
    /// allocate MoE scratch (config.num_experts == 0).
    pub(crate) fn from_scratch(s: &'a Qwen35Scratch) -> Self {
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
#[allow(dead_code)]
pub(crate) fn moe_ffn_decode(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    layer_idx: usize,
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
    let down_expanded = gpu.alloc_tensor(&[k * hidden], DType::F32)?;

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
        gpu, pager, ffn, x_norm, x_residual, config, &refs, false, layer_idx, None, false,
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
    ffn.router.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
        && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
        && ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
}

/// Mixed Qwen3.5 A3B path where router/scalar still need plain RMSNorm(x)
/// but routed MQ2-Lloyd experts need FWHT(RMSNorm(x)). Use the plain+rotated
/// fused norm kernel, then feed `s.tmp` (plain) and `s.moe_x_rot` (rotated)
/// into `moe_ffn_decode_impl`.
pub(crate) fn ffn_routed_mq2_lloyd_plain_prerotate_for_moe(ffn: &MoeFfnWeights) -> bool {
    let Some(first) = ffn.experts.first() else {
        return false;
    };
    first.gate_up.gpu_dtype == DType::MQ2G256Lloyd
        && first.down.gpu_dtype == DType::MQ2G256Lloyd
        && first.gate_up.awq_scale.is_none()
        && first.down.awq_scale.is_none()
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
pub(crate) fn moe_ffn_has_mq3(ffn: &MoeFfnWeights) -> bool {
    if ffn.experts.is_empty() {
        return false;
    }
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    is_mq3_any(ffn.router.gpu_dtype)
        || is_mq3_any(ffn.shared_expert_gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.up.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.down.gpu_dtype)
        || ffn
            .experts
            .iter()
            .any(|e| is_mq3_any(e.gate_up.gpu_dtype) || is_mq3_any(e.down.gpu_dtype))
}

pub(crate) fn moe_ffn_has_mq3_lloyd(ffn: &MoeFfnWeights) -> bool {
    if ffn.experts.is_empty() {
        return false;
    }
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    is_lloyd(ffn.router.gpu_dtype)
        || is_lloyd(ffn.shared_expert_gate.gpu_dtype)
        || is_lloyd(ffn.shared_expert.gate.gpu_dtype)
        || is_lloyd(ffn.shared_expert.up.gpu_dtype)
        || is_lloyd(ffn.shared_expert.down.gpu_dtype)
        || ffn
            .experts
            .iter()
            .any(|e| is_lloyd(e.gate_up.gpu_dtype) || is_lloyd(e.down.gpu_dtype))
}

/// Zero-alloc MoE decode for the scratch path. `scratch.moe_*` fields must
/// be populated (done automatically by `Qwen35Scratch::new` when config
/// indicates a MoE model). Safe to call under hipGraph stream capture.
pub(crate) fn moe_ffn_decode_with_scratch(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    layer_idx: usize,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, pager, ffn, x_norm, x_residual, config, &refs, false, layer_idx, None, false,
    )
}

/// Same as `moe_ffn_decode_with_scratch` but expects the caller to have
/// already populated `scratch.moe_x_rot` with FWHT-rotated post-rmsnorm x
/// (e.g. via a fused `fused_rmsnorm_rotate_mq` launch at the call site).
/// For all-MQ4 MoE layers this saves one launch per layer by eliding the
/// internal `rotate_x_mq`. Mixed routed MQ2-Lloyd layers can also use
/// this path when the caller separately provides the plain RMSNorm output.
pub(crate) fn moe_ffn_decode_with_scratch_prerotated(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    layer_idx: usize,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    moe_ffn_decode_impl(
        gpu, pager, ffn, x_norm, x_residual, config, &refs, true, layer_idx, None, false,
    )
}

pub(crate) fn download_i32_tensor(
    gpu: &Gpu,
    tensor: &GpuTensor,
    len: usize,
) -> HipResult<Vec<i32>> {
    gpu.bind_thread()?;
    let mut data = vec![0i32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * 4) };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf)?;
    Ok(data)
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

pub(crate) fn upload_cpu_topk_to_device(
    gpu: &mut Gpu,
    topk_indices: &[usize],
    topk_weights: &[f32],
    topk_indices_tensor: &GpuTensor,
    topk_weights_tensor: &GpuTensor,
) -> HipResult<()> {
    if topk_indices.len() != topk_weights.len() {
        return Err(HipError::new(
            0,
            "CPU top-k upload received mismatched index/weight lengths",
        ));
    }
    let mut topk_i32 = Vec::with_capacity(topk_indices.len());
    for &idx in topk_indices {
        if idx > i32::MAX as usize {
            return Err(HipError::new(0, "CPU top-k expert index exceeds i32 range"));
        }
        topk_i32.push(idx as i32);
    }
    gpu.hip
        .memcpy_htod(&topk_indices_tensor.buf, i32_slice_as_bytes(&topk_i32))?;
    gpu.hip
        .memcpy_htod(&topk_weights_tensor.buf, f32_slice_as_bytes(topk_weights))?;
    Ok(())
}

pub(crate) fn cpu_topk_from_softmaxed_rows(
    probs: &[f32],
    n: usize,
    n_exp: usize,
    k_top: usize,
    norm_topk_prob: bool,
) -> HipResult<(Vec<usize>, Vec<f32>)> {
    let active_len = n.saturating_mul(n_exp);
    if probs.len() < active_len {
        return Err(HipError::new(
            0,
            "CPU top-k received probability buffer with unexpected length",
        ));
    }
    let mut all_indices = Vec::with_capacity(n * k_top);
    let mut all_weights = Vec::with_capacity(n * k_top);
    for token_idx in 0..n {
        let row = &probs[token_idx * n_exp..(token_idx + 1) * n_exp];
        let mut indices = (0..n_exp).collect::<Vec<_>>();
        indices.select_nth_unstable_by(k_top - 1, |&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut topk = indices.into_iter().take(k_top).collect::<Vec<_>>();
        topk.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut weights = topk.iter().map(|&idx| row[idx]).collect::<Vec<_>>();
        if norm_topk_prob {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
        }
        all_indices.extend(topk);
        all_weights.extend(weights);
    }
    Ok((all_indices, all_weights))
}

pub(crate) fn ensure_paged_experts_resident(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    indices: &[usize],
) -> HipResult<()> {
    if !ffn.experts.is_empty() {
        return Ok(());
    }
    let Some(pager) = pager else {
        return Err(HipError::new(
            0,
            "paged Qwen35-MoE expert layer has no weight pager",
        ));
    };
    let mut unique = indices
        .iter()
        .copied()
        .map(|expert| expert as u16)
        .collect::<Vec<_>>();
    unique.sort_unstable();
    unique.dedup();
    let mut pager = pager.borrow_mut();
    pager
        .would_fit_expert_module_set(ffn.layer_idx, &unique)
        .map_err(|e| HipError::new(0, &format!("page expert module set: {e}")))?;
    for &expert in &unique {
        pager
            .ensure_expert_module_resident(
                hipfire_runtime::weight_pager::ExpertModuleKey {
                    layer: ffn.layer_idx,
                    expert,
                },
                gpu,
            )
            .map_err(|e| HipError::new(0, &format!("page expert module: {e}")))?;
    }
    pager
        .patch_expert_module_ptr_table(
            ffn.layer_idx,
            &unique,
            &ffn.expert_gate_up_ptrs,
            &ffn.expert_down_ptrs,
            gpu,
        )
        .map_err(|e| HipError::new(0, &format!("patch expert module ptrs: {e}")))?;
    Ok(())
}

pub(crate) fn moe_expert_shape(
    ffn: &MoeFfnWeights,
) -> Option<hipfire_runtime::weight_pager::ExpertShape> {
    if let Some(shape) = ffn.expert_shape {
        Some(shape)
    } else {
        let first = ffn.experts.first()?;
        Some(hipfire_runtime::weight_pager::ExpertShape {
            gate_up_m: first.gate_up.m,
            gate_up_k: first.gate_up.k,
            down_m: first.down.m,
            down_k: first.down.k,
        })
    }
}

fn paged_moe_debug_sync(gpu: &Gpu, label: &str) -> HipResult<()> {
    if std::env::var("HIPFIRE_PAGED_MOE_DEBUG").ok().as_deref() == Some("1") {
        eprintln!("[paged_moe_debug] {label}");
        gpu.hip.device_synchronize()?;
    }
    Ok(())
}

pub(crate) fn ensure_qwen35_forward_capability(config: &Qwen35Config) -> HipResult<()> {
    if config.num_experts > 0
        && !config.has_shared_expert
        && std::env::var("HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD")
            .ok()
            .as_deref()
            != Some("1")
    {
        return Err(HipError::new(
            0,
            "routed-only Qwen3 MoE forward is not validated yet; load/catalog/probe are supported, but execution is gated to avoid GPU faults (set HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD=1 only for kernel debugging)",
        ));
    }
    Ok(())
}

/// The actual MoE FFN implementation. Uses the caller-provided scratch
/// buffers, never allocates.
pub(crate) fn moe_ffn_decode_impl(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    s: &MoeScratchRef<'_>,
    x_rot_prerotated: bool,
    layer_idx: usize,
    // EP (Ship 6 substrate-EP). `ep_routed_out = Some(partial)` redirects the
    // routed combine + shared-down into a zeroed partial (the EP executor
    // all-reduces it and adds into x_residual once); `None` = single-GPU into
    // x_residual (byte-identical). `ep_skip_shared` skips the shared-expert
    // down on rank>0 so the replicated shared expert is summed once.
    ep_routed_out: Option<&GpuTensor>,
    ep_skip_shared: bool,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let _ = hidden;

    let router_logits = s.router_logits;
    let scalar_buf = s.scalar_buf;
    let gate_up_buf = s.gate_up_buf;
    let gate_buf = s.gate_buf;
    let up_buf = s.up_buf;
    let ffn_hidden = s.ffn_hidden;
    let ffn_out = s.ffn_out;

    // Phase 2a-iii: rotate x_norm once per layer and share the rotated
    // buffer across every MQ4 GEMV that consumes it. Two independent users:
    //   1. The 4-way fused gate-side GEMV (gate_side_mq4) — requires router,
    //      shared_expert_gate, shared_expert.{gate,up} all MQ4G256.
    //   2. The indexed routed-expert gate_up GEMV (routed_gate_up_mq4) — fires
    //      whenever the routed gate_up family is MQ4G256, independent of the
    //      gate-side family's dtype.
    // We compute x_rot_local if EITHER user will fire. Models with a Q8
    // router (e.g. the post-PR-171 attractor rule for MoE) thus still get
    // the device-side top-K + indexed expert GEMV path — only the 4-way
    // fused GEMV falls back to four individual `weight_gemv` calls.
    let dispatch_flags = if let Some(dtypes) = MoePrefillDtypes::from_ffn(ffn) {
        moe_decode_dispatch_flags_for_dtypes(&dtypes, k, ffn.paro_shared.is_some())
    } else {
        let gate_side_mq4 = config.has_shared_expert
            && ffn.router.gpu_dtype == DType::MQ4G256
            && ffn.shared_expert_gate.gpu_dtype == DType::MQ4G256
            && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
            && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256
            && ffn
                .experts
                .iter()
                .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256);
        let shared_gate_up_mq4 = config.has_shared_expert
            && ffn.shared_expert.gate.gpu_dtype == DType::MQ4G256
            && ffn.shared_expert.up.gpu_dtype == DType::MQ4G256;
        MoeDecodeDispatchFlags {
            gate_side_mq4,
            shared_gate_up_mq4,
            routed_mq4: false,
            routed_mq6: false,
            routed_mq2_lloyd: false,
            routed_paro: false,
            routed_gate_up_mq4: false,
            routed_gate_up_mq6: false,
            routed_gate_up_mq2_lloyd: false,
            routed_gate_up_paro: false,
            routed_dtype_indexable_mq4: false,
            routed_dtype_indexable_mq6: false,
            routed_dtype_indexable_mq2_lloyd: false,
            routed_dtype_indexable_paro: false,
            routed_path: MoeDecodeIndexedRoutedPath::None,
            use_gpu_topk: false,
            needs_x_rot_local: gate_side_mq4,
        }
    };
    let gate_side_mq4 = dispatch_flags.gate_side_mq4;
    let shared_gate_up_mq4 = dispatch_flags.shared_gate_up_mq4;
    let routed_mq4 = dispatch_flags.routed_mq4;
    let routed_gate_up_mq4 = dispatch_flags.routed_gate_up_mq4;
    let routed_gate_up_paro = dispatch_flags.routed_gate_up_paro;
    let routed_dtype_indexable_mq4 = dispatch_flags.routed_dtype_indexable_mq4;
    let routed_dtype_indexable_mq6 = dispatch_flags.routed_dtype_indexable_mq6;
    let routed_dtype_indexable_mq2_lloyd = dispatch_flags.routed_dtype_indexable_mq2_lloyd;
    let routed_dtype_indexable_paro = dispatch_flags.routed_dtype_indexable_paro;
    // Detect Phase 2b+2c GPU-only fast path. When true, top-K runs on
    // device and the indexed MoE kernels consume topk_indices /
    // topk_weights directly — no D2H sync, hipGraph-capture-safe.
    // Note: this no longer requires `gate_side_mq4`. The device-side
    // `moe_topk_renorm_k8` kernel and the indexed gate_up/down GEMVs
    // consume router_logits/topk_indices/topk_weights/x_rot from device
    // buffers regardless of how router_logits was produced (fused-4 or
    // individual weight_gemv). Q8 routers (issue-#171 attractor rule)
    // are now first-class for graph capture. Mixed-kmap A3B layers
    // promoted to MQ6 dispatch through the HFQ6 indexed kernels instead
    // of the HFQ4 ones — same control flow, different kernel binary.
    let use_gpu_topk = dispatch_flags.use_gpu_topk;
    let needs_x_rot_local = dispatch_flags.needs_x_rot_local;
    let x_rot_local = if needs_x_rot_local {
        if !routed_gate_up_paro {
            // FWHT-rotated path needs the MQ sign LUT.
            gpu.ensure_mq_signs()?;
        }
        if !x_rot_prerotated {
            if routed_gate_up_paro {
                // ParoQuant routed experts: use the per-layer shared Givens
                // rotation (from `ffn.paro_shared.gate_up_*`). The loader
                // builds every expert's `paro` alias from the same sidecars,
                // so experts[0].gate_up.paro is the canonical handle.
                let paro = ffn.experts[0]
                    .gate_up
                    .paro
                    .as_ref()
                    .expect("routed_gate_up_paro implies experts[0].gate_up.paro.is_some()");
                hipfire_runtime::weights::rotate_x_paro_for(
                    gpu,
                    paro,
                    x_norm,
                    s.x_rot_local,
                    config.dim,
                )?;
            } else {
                // F2 / F1: AWQ-aware FWHT rotate. All MQ4 weights in this
                // layer consume the same post-rmsnorm x, so they share the
                // same input basis → identical imatrix → byte-identical
                // AWQ scales. When gate_side_mq4 is true the 4-way fused
                // GEMV expects rotation aligned with `ffn.router`'s AWQ
                // scale; otherwise pick `ffn.experts[0].gate_up` as the
                // routed-expert representative for the indexed kernel
                // path. When AWQ is disabled (no sidecar),
                // `rotate_x_mq_for` routes to the non-AWQ kernel —
                // byte-identical to pre-F2 either way.
                if ffn.experts.is_empty() && !gate_side_mq4 {
                    gpu.rotate_x_mq(x_norm, s.x_rot_local, config.dim)?;
                    paged_moe_debug_sync(gpu, "after paged rotate_x_mq")?;
                } else {
                    let next_lin = if gate_side_mq4 {
                        &ffn.router
                    } else {
                        &ffn.experts[0].gate_up
                    };
                    rotate_x_mq_for(gpu, next_lin, x_norm, s.x_rot_local, config.dim)?;
                }
            }
        }
        // else caller guarantees s.x_rot_local already holds the rotated x.
        Some(s.x_rot_local)
    } else {
        None
    };
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn
            .experts
            .first()
            .map(|e| e.gate_up.gpu_dtype)
            .unwrap_or(DType::F32),
        routed_down: ffn
            .experts
            .first()
            .map(|e| e.down.gpu_dtype)
            .unwrap_or(DType::F32),
        has_paro_shared: ffn.paro_shared.is_some(),
    };
    // Resolution is owned by the MoeFamily (Ship 4.1). The model passes only
    // the dtype snapshot + k; the executor computes MoeResolution from MoeDtypes.

    // ── 1+2b+3a. Fused 4-way GEMV (router + shared_expert_gate + shared.gate + shared.up) ──
    // All four read the SAME rotated x_rot_local with the SAME K. Fusing them
    // into `fused_qkvza_hfq4g256` saves 3 launch submits per MoE layer and
    // lets underused tails (shared_expert_gate_m=1, router_m=256) co-schedule
    // with the larger 512-row gate/up bodies. 40 layers × 3 saved launches
    // = 120 launches/fwd, ~8-12% cycle-time savings on 7900 XTX.
    let shared_gate = slice_f32_view(gate_buf, 0, smi);
    let shared_up = slice_f32_view(up_buf, 0, smi);
    if gate_side_mq4 {
        // All-MQ4 gate-side: use the 4-way fused prerotated GEMV. Router,
        // shared_expert_gate, shared_expert.gate, shared_expert.up — all
        // M×K matrices in HFQ4G256 storage (MQ4 weights are HFQ4 bytes pre-
        // rotated at quant time, so `gemv_hfq4g256` inner loop with the
        // FWHT-rotated input is mathematically equivalent to `gemv_mq4g256`).
        let xr = x_rot_local.expect("gate_side_mq4 implies x_rot_local is Some");
        gpu.fused_qkvza_hfq4g256(
            &ffn.router.buf,
            &ffn.shared_expert_gate.buf,
            &ffn.shared_expert.gate.buf,
            &ffn.shared_expert.up.buf,
            xr,
            router_logits,
            scalar_buf,
            &shared_gate,
            &shared_up,
            ffn.router.m,
            ffn.shared_expert_gate.m,
            ffn.shared_expert.gate.m,
            ffn.shared_expert.up.m,
            ffn.router.k,
        )?;
    } else {
        // Mixed-dtype fallback: four separate `weight_gemv` calls. Each
        // weight_gemv handles its own rotation for MQ4 weights internally
        // (via `gpu.mq_x_rot`, a distinct scratch from `s.x_rot_local`),
        // so the externally-computed `x_rot_local` is preserved for the
        // downstream indexed gate_up GEMV when routed_gate_up_mq4 is true.
        weight_gemv(gpu, &ffn.router, x_norm, router_logits)?;
        if ffn.experts.is_empty() {
            paged_moe_debug_sync(gpu, "after paged router gemv")?;
        }
        if config.has_shared_expert {
            weight_gemv(gpu, &ffn.shared_expert_gate, x_norm, scalar_buf)?;
            if shared_gate_up_mq4 {
                // Mixed router/scalar dtypes cannot use the 4-way fused gate-side
                // kernel, but shared gate+up can still share one MQ4 rotation.
                if x_rot_prerotated
                    && ffn.shared_expert.gate.awq_scale.is_none()
                    && ffn.shared_expert.up.awq_scale.is_none()
                {
                    gpu.fused_gate_up_hfq4g256(
                        &ffn.shared_expert.gate.buf,
                        &ffn.shared_expert.up.buf,
                        s.x_rot_local,
                        &shared_gate,
                        &shared_up,
                        ffn.shared_expert.gate.m,
                        ffn.shared_expert.up.m,
                        ffn.shared_expert.gate.k,
                    )?;
                } else {
                    gpu.ensure_mq_signs()?;
                    let shared_x_rot = GpuTensor {
                        buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                        shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                        dtype: DType::F32,
                    };
                    rotate_x_mq_for(
                        gpu,
                        &ffn.shared_expert.gate,
                        x_norm,
                        &shared_x_rot,
                        ffn.shared_expert.gate.k,
                    )?;
                    gpu.fused_gate_up_hfq4g256(
                        &ffn.shared_expert.gate.buf,
                        &ffn.shared_expert.up.buf,
                        &shared_x_rot,
                        &shared_gate,
                        &shared_up,
                        ffn.shared_expert.gate.m,
                        ffn.shared_expert.up.m,
                        ffn.shared_expert.gate.k,
                    )?;
                }
            } else {
                weight_gemv(gpu, &ffn.shared_expert.gate, x_norm, &shared_gate)?;
                weight_gemv(gpu, &ffn.shared_expert.up, x_norm, &shared_up)?;
            }
        }
    }

    // ── 2a. Top-K selection — GPU fast path or CPU fallback ──
    let (topk_indices_cpu, topk_weights_cpu): (Option<Vec<usize>>, Option<Vec<f32>>) =
        if use_gpu_topk {
            // GPU path: split softmax + top-K + renorm into two kernels so
            // the routing path uses identical softmax math to gpu.softmax_f32
            // (and thus to a CPU reference). The fused
            // moe_softmax_topk_renorm_k8 variant produced topk_weights that
            // differed from gpu.softmax_f32 + manual `*w /= sum` by exactly
            // 1 ULP per element, which compounds across 30+ MoE layers and
            // 8 experts/layer into a structural attractor on Qwen3.5-A3B
            // and 122B-A10B at MQ4. The new moe_topk_renorm_k8 takes
            // pre-softmaxed probs and uses direct division for renorm.
            gpu.softmax_f32(router_logits)?;
            gpu.moe_topk_renorm_k8(
                router_logits,
                s.topk_indices,
                s.topk_weights,
                n_exp,
                config.norm_topk_prob,
            )?;
            if ffn.experts.is_empty() {
                paged_moe_debug_sync(gpu, "after paged topk")?;
            }
            if moe_router_histogram_active() {
                let indices = download_i32_tensor(gpu, s.topk_indices, k)?
                    .into_iter()
                    .map(router_index_i32_to_usize)
                    .collect::<Vec<_>>();
                let weights = gpu.download_f32(s.topk_weights)?;
                record_moe_router_selection(layer_idx, &indices, &weights);
            }
            (None, None)
        } else {
            // Fallback: GPU softmax → CPU download → CPU top-K + renorm.
            gpu.softmax_f32(router_logits)?;
            let probs = gpu.download_f32(router_logits)?;
            let mut indices: Vec<usize> = (0..n_exp).collect();
            indices.select_nth_unstable_by(k - 1, |&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut topk_indices: Vec<usize> = indices.into_iter().take(k).collect();
            topk_indices.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut topk_weights: Vec<f32> = topk_indices.iter().map(|&i| probs[i]).collect();
            if config.norm_topk_prob {
                let sum: f32 = topk_weights.iter().sum();
                if sum > 0.0 {
                    for w in topk_weights.iter_mut() {
                        *w /= sum;
                    }
                }
            }
            record_moe_router_selection(layer_idx, &topk_indices, &topk_weights);
            (Some(topk_indices), Some(topk_weights))
        };
    if ffn.experts.is_empty() {
        if let (Some(indices), Some(weights)) =
            (topk_indices_cpu.as_ref(), topk_weights_cpu.as_ref())
        {
            upload_cpu_topk_to_device(gpu, indices, weights, s.topk_indices, s.topk_weights)?;
        }
    }
    if ffn.experts.is_empty() {
        let indices = if let Some(indices) = topk_indices_cpu.as_ref() {
            indices.clone()
        } else {
            download_i32_tensor(gpu, s.topk_indices, k)?
                .into_iter()
                .map(router_index_i32_to_usize)
                .collect::<Vec<_>>()
        };
        ensure_paged_experts_resident(gpu, pager, ffn, &indices)?;
        paged_moe_debug_sync(gpu, "after paged expert residency")?;
    }

    // The shared-expert gate scalar (in `scalar_buf`) is the RAW logit from
    // the 4-way fused GEMV — sigmoid is applied internally by
    // `gemv_hfq4g256_residual_sigmoid_scaled_gpu`, eliminating the separate
    // 1-elem `sigmoid_f32` launch (~40 saved per forward on A3B).
    if config.has_shared_expert {
        if ffn.shared_expert.down.gpu_dtype == DType::MQ4G256 {
            gpu.ensure_mq_signs()?;
            let x_rot_alias = GpuTensor {
                buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            };
            // F2: AWQ-aware silu_mul+rotate for the shared-expert down input.
            fused_silu_mul_rotate_mq_for(
                gpu,
                &ffn.shared_expert.down,
                &shared_gate,
                &shared_up,
                &x_rot_alias,
                smi,
            )?;
            gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
                &ffn.shared_expert.down.buf,
                &x_rot_alias,
                x_residual,
                scalar_buf,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
            )?;
        } else {
            // Non-MQ fallback path still needs the separate sigmoid + scaled-add.
            gpu.sigmoid_f32(scalar_buf)?;
            // Non-MQ fallback: pre-2a-ii path.
            let shared_hid = slice_f32_view(ffn_hidden, 0, smi);
            gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid)?;
            weight_gemv(gpu, &ffn.shared_expert.down, &shared_hid, ffn_out)?;
            gpu.scaled_add_inplace_gpu_scalar_f32(x_residual, ffn_out, scalar_buf)?;
        }
    }

    // ── 4. Top-K routed experts ──
    if routed_mq4 {
        gpu.ensure_mq_signs()?;
    }

    if use_gpu_topk || ffn.experts.is_empty() {
        // Phase 2b+2c GPU-only fast path: indexed MoE kernels read expert
        // IDs and weights from device buffers, zero D2H sync.
        //
        // Task #100 fix (2026-05-21): atomic-free expand+combine, mirroring
        // the prefill path (forward_prefill_batch_with_pbs L5217-5232). The
        // earlier single-launch `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`
        // used `atomicAdd` across K_TOP=8 blocks per row, which gives FP32
        // sums whose final bits depend on wavefront-scheduling order
        // (see gemv_hfq4g256_moe_down.hip:14-19 — the kernel's own comment
        // admits non-determinism). Under hipGraph capture the ordering
        // diverges from direct mode, so each forward step accumulates a
        // ~1-ULP delta that compounds through the KV cache + GDN state,
        // crossing the top-1 margin at step ~7 (q8 KV) or ~114 (asym3 KV).
        // Expanding into `s.down_expanded` (no atomics) then summing via
        // the fixed-order `moe_down_combine_k8_batched` makes the MoE FFN
        // output byte-deterministic, eliminating the cumulative drift.
        let xr = x_rot_local.expect(
            "use_gpu_topk implies routed_gate_up_{mq4,mq6,mq2_lloyd,paro} implies x_rot_local is Some",
        );
        let shape = moe_expert_shape(ffn).ok_or_else(|| {
            HipError::new(0, "missing MoE expert shape metadata for indexed decode")
        })?;
        let down_m = shape.down_m;
        let down_k = shape.down_k;
        let gate_up_k = shape.gate_up_k;
        // Dispatch the right indexed-GEMV layout for this layer's routed
        // dtype. Within a layer, gate_up and down share the same dtype
        // (kmap promotes whole expert tensor groups together; ParoQuant
        // loader builds gate_up + down at matching ParoQ4G128); the
        // `routed_dtype_indexable_*` checks above enforce this.
        if routed_dtype_indexable_mq4 {
            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                &ffn.expert_gate_up_ptrs,
                s.topk_indices,
                xr,
                s.gate_batch,
                s.up_batch,
                2 * mi,
                gate_up_k,
            )?;
        } else if routed_dtype_indexable_mq6 {
            // HFQ6 (200 B/group) indexed kernel.
            if ffn.experts.is_empty() {
                gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    s.topk_indices,
                    xr,
                    s.gate_batch,
                    s.up_batch,
                    2 * mi,
                    gate_up_k,
                    k,
                    1,
                )?;
            } else {
                gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
                    &ffn.expert_gate_up_ptrs,
                    s.topk_indices,
                    xr,
                    s.gate_batch,
                    s.up_batch,
                    2 * mi,
                    gate_up_k,
                )?;
            }
        } else if routed_dtype_indexable_mq2_lloyd {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                &ffn.expert_gate_up_ptrs,
                s.topk_indices,
                xr,
                s.gate_batch,
                s.up_batch,
                2 * mi,
                gate_up_k,
                k,
            )?;
        } else {
            // routed_dtype_indexable_paro — HFQ4G128 (72 B/group) indexed
            // kernel. xr is already Givens-rotated above by rotate_x_paro_for.
            gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
                &ffn.expert_gate_up_ptrs,
                s.topk_indices,
                xr,
                s.gate_batch,
                s.up_batch,
                2 * mi,
                gate_up_k,
            )?;
        }
        if ffn.experts.is_empty() {
            paged_moe_debug_sync(gpu, "after paged routed gate_up")?;
        }
        // Gate→down hop. MQ paths use a single fused silu+mul+FWHT kernel;
        // ParoQuant uses the structural mirror `fused_silu_mul_givens_rotate`
        // (silu+mul+per-channel-scale+krot Givens rounds in one launch).
        // The earlier 2-launch decomposition (silu_mul_f32 + givens_rotate)
        // produced a small but reproducible direct-vs-graph numerical
        // delta on gfx1151/HIP 7.13; fusing matches the MQ4 pattern that
        // hipGraph captures byte-identically.
        if routed_dtype_indexable_paro {
            let paro_down = ffn.experts[0]
                .down
                .paro
                .as_ref()
                .expect("routed_paro implies experts[0].down.paro.is_some()");
            gpu.fused_silu_mul_givens_rotate_f32(
                s.gate_batch,
                s.up_batch,
                s.rot_batch,
                &paro_down.pairs,
                &paro_down.theta,
                &paro_down.channel_scales,
                k,
                mi,
                paro_down.krot as usize,
            )?;
        } else {
            // F2: AWQ-aware silu_mul+FWHT-rotate. All experts in this MoE
            // layer share the same input residual basis → same imatrix
            // → byte-identical AWQ scales; experts[0].down is
            // representative. Helper dispatches on awq_scale presence,
            // not on weight bytes layout.
            if ffn.experts.is_empty() {
                gpu.fused_silu_mul_rotate_mq_batched(s.gate_batch, s.up_batch, s.rot_batch, mi, k)?;
            } else {
                fused_silu_mul_rotate_mq_batched_for(
                    gpu,
                    &ffn.experts[0].down,
                    s.gate_batch,
                    s.up_batch,
                    s.rot_batch,
                    mi,
                    k,
                )?;
            }
        }
        if ffn.experts.is_empty() {
            paged_moe_debug_sync(gpu, "after paged routed silu/rotate")?;
        }
        // Atomic-free expanded write: [k_top × down_m] f32, one block per
        // (m, krank) pair, no cross-block contention.
        if routed_dtype_indexable_mq4 {
            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                &ffn.expert_down_ptrs,
                s.topk_indices,
                s.rot_batch,
                s.down_expanded,
                down_m,
                down_k,
                k,
                1,
            )?;
        } else if routed_dtype_indexable_mq6 {
            gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                &ffn.expert_down_ptrs,
                s.topk_indices,
                s.rot_batch,
                s.down_expanded,
                down_m,
                down_k,
                k,
                1,
            )?;
        } else if routed_dtype_indexable_mq2_lloyd {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                &ffn.expert_down_ptrs,
                s.topk_indices,
                s.rot_batch,
                s.down_expanded,
                down_m,
                down_k,
                k,
                1,
            )?;
        } else {
            // routed_dtype_indexable_paro
            gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                &ffn.expert_down_ptrs,
                s.topk_indices,
                s.rot_batch,
                s.down_expanded,
                down_m,
                down_k,
                k,
                1,
            )?;
        }
        if ffn.experts.is_empty() {
            paged_moe_debug_sync(gpu, "after paged routed down")?;
        }
        // Deterministic combine: sums K_TOP slots into x_residual in a
        // fixed iteration order with `topk_weights` applied. This kernel
        // is dtype-independent — it operates on the f32 expanded buffer.
        gpu.moe_down_combine_k8_batched(s.down_expanded, s.topk_weights, x_residual, down_m, k, 1)?;
        if ffn.experts.is_empty() {
            paged_moe_debug_sync(gpu, "after paged routed combine")?;
        }
    } else {
        // CPU-top-K fallback path. Two sub-paths from here:
        //   (a) k==8 && all-MQ4 but gate_side wasn't all-MQ4 (e.g. router
        //       not MQ4): use the kernarg-pointer fused kernels with the
        //       CPU-selected indices.
        //   (b) Mixed-dtype or k != 8: per-expert loop.
        let topk_indices = topk_indices_cpu.expect("CPU-fallback path implies CPU top-K");
        let topk_weights = topk_weights_cpu.expect("CPU-fallback path implies CPU top-K");
        // `use_kernarg_fused` dispatches both gate_up and down through
        // HFQ4G256-layout kernels, so it needs routed.down MQ4 as well as
        // routed.gate_up. Previously this constraint was carried implicitly
        // by `x_rot_local.is_some()` (which required gate_side_mq4, which in
        // shipped configs implied routed_mq4); now that x_rot_local fires
        // for `routed_gate_up_mq4` alone, check `routed_mq4` explicitly.
        // With both checks the condition equals `use_gpu_topk`, so this
        // branch is effectively dead — kept for clarity until the cleanup.
        let use_kernarg_fused = k == 8 && routed_gate_up_mq4 && routed_mq4 && x_rot_local.is_some();
        if use_kernarg_fused {
            let xr = x_rot_local.unwrap();
            let e0 = &ffn.experts[topk_indices[0]];
            let e1 = &ffn.experts[topk_indices[1]];
            let e2 = &ffn.experts[topk_indices[2]];
            let e3 = &ffn.experts[topk_indices[3]];
            let e4 = &ffn.experts[topk_indices[4]];
            let e5 = &ffn.experts[topk_indices[5]];
            let e6 = &ffn.experts[topk_indices[6]];
            let e7 = &ffn.experts[topk_indices[7]];
            gpu.gemv_hfq4g256_moe_gate_up_k8(
                &e0.gate_up.buf,
                &e1.gate_up.buf,
                &e2.gate_up.buf,
                &e3.gate_up.buf,
                &e4.gate_up.buf,
                &e5.gate_up.buf,
                &e6.gate_up.buf,
                &e7.gate_up.buf,
                xr,
                s.gate_batch,
                s.up_batch,
                2 * mi,
                e0.gate_up.k,
            )?;
            // F2: AWQ-aware silu_mul+rotate; experts[0].down is representative
            // (all experts share imatrix at this layer's residual basis).
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &ffn.experts[0].down,
                s.gate_batch,
                s.up_batch,
                s.rot_batch,
                mi,
                k,
            )?;
            let scales = [
                topk_weights[0],
                topk_weights[1],
                topk_weights[2],
                topk_weights[3],
                topk_weights[4],
                topk_weights[5],
                topk_weights[6],
                topk_weights[7],
            ];
            gpu.gemv_hfq4g256_moe_down_residual_scaled_k8(
                &e0.down.buf,
                &e1.down.buf,
                &e2.down.buf,
                &e3.down.buf,
                &e4.down.buf,
                &e5.down.buf,
                &e6.down.buf,
                &e7.down.buf,
                s.rot_batch,
                x_residual,
                scales,
                e0.down.m,
                e0.down.k,
            )?;
        } else {
            // Per-expert fallback for layers that aren't all-MQ4 or have k != 8.
            for (&expert_idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
                let expert = &ffn.experts[expert_idx];
                // The MQ4 pre-rotated GEMV is an MQ4G256-ONLY dequant kernel
                // (136 B/group). `x_rot_local` is `Some` for *any* rotated
                // gate_up dtype (`needs_x_rot_local` includes MQ6/mq2-lloyd/paro),
                // so gating the pre-rotated call on `x_rot_local.is_some()` fed
                // MQ6G256 gate_up (200 B/group — what `--format mq4`/`mq6` tier
                // routed experts to) through the MQ4 kernel → misread groups →
                // NaN logits (the cross-arch qwen3.5-MoE mq4/mq6 NaN). Use the
                // pre-rotated fast path ONLY for genuine MQ4 gate_up; every other
                // dtype goes through `weight_gemv`, which dispatches the correct
                // per-dtype dequant (and applies its own rotation internally).
                if routed_gate_up_mq4 {
                    let xr = x_rot_local
                        .expect("routed_gate_up_mq4 ⇒ needs_x_rot_local ⇒ x_rot_local is Some");
                    gpu.gemv_mq4g256_prerotated(
                        &expert.gate_up.buf,
                        xr,
                        gate_up_buf,
                        expert.gate_up.m,
                        expert.gate_up.k,
                    )?;
                } else {
                    weight_gemv(gpu, &expert.gate_up, x_norm, gate_up_buf)?;
                }
                let gate_view = slice_f32_view(gate_up_buf, 0, mi);
                let up_view = slice_f32_view(gate_up_buf, mi, mi);
                if routed_mq4 {
                    let x_rot_alias = GpuTensor {
                        buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
                        shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                        dtype: DType::F32,
                    };
                    // F2: AWQ-aware silu_mul+rotate for this expert's down input.
                    fused_silu_mul_rotate_mq_for(
                        gpu,
                        &expert.down,
                        &gate_view,
                        &up_view,
                        &x_rot_alias,
                        mi,
                    )?;
                    gpu.gemv_hfq4g256_residual_scaled_cpu(
                        &expert.down.buf,
                        &x_rot_alias,
                        x_residual,
                        weight,
                        expert.down.m,
                        expert.down.k,
                    )?;
                } else {
                    let hid_view = slice_f32_view(ffn_hidden, 0, mi);
                    gpu.silu_mul_f32(&gate_view, &up_view, &hid_view)?;
                    weight_gemv(gpu, &expert.down, &hid_view, ffn_out)?;
                    gpu.scaled_add_inplace_cpu_scalar_f32(x_residual, ffn_out, weight)?;
                }
            }
        }
    }
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
    hipfire_runtime::dispatch::moe_family()
        .run(&ctx, gpu, &moe_params)
        .map_err(HipError::from)?;
    Ok(())
}
