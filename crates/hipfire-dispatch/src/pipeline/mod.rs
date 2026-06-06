// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;
#[allow(unused_imports)]
use hip_bridge;

pub(crate) mod steps;
pub use steps::{execute_steps, FusedPattern, GemvInput, Step};

pub struct Pipeline {
    pub ops: &'static [PipelineOp],
}

impl Pipeline {
    pub fn new(ops: &'static [PipelineOp]) -> Self { Self { ops } }

    pub fn can_satisfy(&self, requested: &[PipelineOp]) -> bool {
        if self.ops.len() > requested.len() { return false; }
        self.ops.iter().zip(requested.iter()).all(|(a, b)| a == b)
    }
}

pub struct LinearParams<'a> {
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub buf: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
}

pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),
    Moe(crate::families::moe::MoeParams<'a>),
}

pub fn execute_pipeline(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[PipelineOp],
    params: &PipelineParams,
    dtype: rdna_compute::DType,
    registry: &KernelRegistry,
) -> Result<(), DispatchError> {
    if let PipelineParams::Moe(p) = params {
        return run_moe_decode(gpu, p);
    }
    if let Some(key) = find_fused(registry, ctx, dtype, steps) {
        return dispatch_fused(gpu, key, params);
    }
    let params = match params {
        PipelineParams::Linear(p) => p,
        PipelineParams::Moe(_) => unreachable!(),
    };
    for &step in steps {
        match step {
            PipelineOp::RotateFwht => {
                use crate::families::rotation::{RotationFamily, RotationParams};
                let rot = RotationFamily::new();
                gpu.ensure_mq_signs().map_err(|e| DispatchError::Hip(e.to_string()))?;
                let x_rot = unsafe {
                    GpuTensor {
                        buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                        shape: vec![params.k],
                        dtype: rdna_compute::DType::F32,
                    }
                };
                rot.run(ctx, gpu, RotationParams {
                    x: params.x, x_up: None, w_norm: None,
                    x_plain: &x_rot, x_rot: &x_rot,
                    awq_scale: None, k: params.k,
                    eps: 1e-6, batch_size: 1,
                    variant: RotationVariant::Plain,
                    givens_pairs: None,
                    givens_theta: None,
                    givens_scales: None,
                    givens_krot: None,
                }).map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            PipelineOp::Gemv => {
                static GEMV_PIPELINE: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_PIPELINE.get_or_init(GemvFamily::new);
                let w = WeightRef {
                    buf: params.buf,
                    dtype,
                    m: params.m,
                    k: params.k,
                    row_stride: params.k,
                    rotation: None,
                    awq_scale: None,
                };
                gemv.run_auto(ctx, gpu, &w, params.x, params.y)?;
            }
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "pipeline", variant: "step",
                    arch: "", quant: "",
                });
            }
        }
    }
    Ok(())
}

fn find_fused(
    registry: &KernelRegistry,
    ctx: &DispatchCtx,
    dtype: rdna_compute::DType,
    requested: &[PipelineOp],
) -> Option<KernelKey> {
    use rdna_compute::DType;
    if dtype == DType::MFP4G32
        && requested.len() == 2
        && requested[0] == PipelineOp::RotateFwht
        && requested[1] == PipelineOp::Gemv
    {
        let key = KernelKey::GemvMfp4G32Fused;
        if registry.resolve(key, ctx, None).is_ok() { return Some(key); }
    }
    None
}

/// Slice a subrange of a flat F32 GpuTensor by element offset + length.
/// Mirrors qwen35::slice_f32_view — unsafe because it aliases device memory.
unsafe fn slice_moe_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    let base = src.buf.as_ptr() as *mut u8;
    let ptr = base.add(offset_elems * 4);
    GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
        shape: vec![len_elems],
        dtype: DType::F32,
    }
}

/// MoE decode executor. Ports the body of `moe_ffn_decode_impl` verbatim,
/// substituting `ffn.*`/`config.*`/`s.*` references with `MoeParams` fields.
/// Phase 1: GPU top-K path only (k=8, indexable routed dtype). The CPU
/// top-K fallback is not supported here — callers that need it retain the
/// original qwen35 path.
pub fn run_moe_decode(gpu: &mut Gpu, p: &crate::families::moe::MoeParams) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) };
    }
    let res = p.res;

    // ── Activation rotation (mirrors qwen35.rs x_rot_local block) ──────────
    let x_rot_local: Option<&GpuTensor> = if res.needs_x_rot_local {
        if !res.routed_indexable_paro {
            hip!(gpu.ensure_mq_signs())?;
        }
        if !p.x_rot_prerotated {
            if res.routed_indexable_paro {
                let paro = p.routed_gate_up_paro.as_ref()
                    .expect("routed_indexable_paro implies gate_up paro sidecar");
                hip!(gpu.givens_rotate_to(
                    p.x_norm, p.x_rot_local,
                    &paro.pairs, &paro.theta, &paro.scales,
                    1, p.hidden, paro.krot,
                ))?;
            } else if res.gate_side_mq4 {
                if let Some(awq) = p.router.awq_scale {
                    hip!(gpu.rotate_x_mq_awq(p.x_norm, awq, p.x_rot_local, p.hidden))?;
                } else {
                    hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
                }
            } else {
                // !gate_side_mq4 but routed MQ4/MQ6: no AWQ on MoE expert weights
                // in Phase 1 targets (A3B). Byte-identical for models without AWQ.
                hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
            }
        }
        Some(p.x_rot_local)
    } else {
        None
    };

    // ── Gate-side GEMV ───────────────────────────────────────────────────────
    // SAFETY: all slice views alias device memory owned by MoEParams' scratch tensors.
    let shared_gate = unsafe { slice_moe_f32_view(p.gate_buf, 0, p.smi) };
    let shared_up   = unsafe { slice_moe_f32_view(p.up_buf,   0, p.smi) };
    if res.gate_side_mq4 {
        let xr = x_rot_local.expect("gate_side_mq4 implies x_rot_local");
        hip!(gpu.fused_qkvza_hfq4g256(
            &p.router.buf, &p.shared_expert_gate.buf,
            &p.shared_gate_w.buf, &p.shared_up_w.buf,
            xr,
            p.router_logits, p.scalar_buf,
            &shared_gate, &shared_up,
            p.router.m, p.shared_expert_gate.m, p.shared_gate_w.m, p.shared_up_w.m,
            p.router.k,
        ))?;
    } else {
        static GEMV_GATE: OnceLock<GemvFamily> = OnceLock::new();
        let gemv = GEMV_GATE.get_or_init(GemvFamily::new);
        let ctx = DispatchCtx::new(gpu);
        gemv.run_auto(&ctx, gpu, &p.router,            p.x_norm, p.router_logits).map_err(|e| DispatchError::Hip(e.to_string()))?;
        gemv.run_auto(&ctx, gpu, &p.shared_expert_gate,p.x_norm, p.scalar_buf).map_err(|e| DispatchError::Hip(e.to_string()))?;
        gemv.run_auto(&ctx, gpu, &p.shared_gate_w,     p.x_norm, &shared_gate).map_err(|e| DispatchError::Hip(e.to_string()))?;
        gemv.run_auto(&ctx, gpu, &p.shared_up_w,       p.x_norm, &shared_up).map_err(|e| DispatchError::Hip(e.to_string()))?;
    }

    // ── Top-K + routed experts: CPU-top-K generic fallback ───────────────────
    // Fires when `!use_gpu_topk` (k != 8 OR routed dtype not indexable). This
    // ports master's `moe_ffn_decode_impl` CPU-fallback per-expert loop
    // (origin/master qwen35.rs, the `else` arm of `if use_gpu_topk`) so MoE
    // layers outside the {k=8, MQ4G256|MQ6G256|ParoQ4G128-routed} fast path
    // run instead of hard-panicking. #393 deleted this; restoring it keeps the
    // dispatch migration behavior-preserving.
    //
    // The fallback is self-contained: it does softmax → CPU top-K + renorm →
    // shared-expert down → generic per-expert routed loop, then returns. It
    // does NOT fall through to the indexed GPU-top-K path below (which assumes
    // k=8 + an indexable routed dtype).
    if !res.use_gpu_topk {
        return run_moe_decode_cpu_fallback(gpu, p, &shared_gate, &shared_up);
    }
    // DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN)
    if let Ok(dump_path) = std::env::var("HIPFIRE_DUMP_HIDDEN") {
        if gpu.hip.device_synchronize().is_ok() {
            if let Ok(all) = gpu.download_f32(p.router_logits) {
                use std::io::Write;
                let path = format!("{dump_path}.router_raw_p");
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = f.write_all(&(0u32).to_le_bytes());
                    for v in &all[..all.len().min(p.n_exp * 4 / 4)] {
                        let _ = f.write_all(&v.to_le_bytes());
                    }
                }
            }
        }
    }
    hip!(gpu.softmax_f32(p.router_logits))?;
    hip!(gpu.moe_topk_renorm_k8(p.router_logits, p.topk_indices, p.topk_weights, p.n_exp, p.norm_topk_prob))?;

    // ── Shared expert down ───────────────────────────────────────────────────
    if p.shared_down_w.dtype == DType::MQ4G256 {
        hip!(gpu.ensure_mq_signs())?;
        let x_rot_alias = unsafe { GpuTensor {
            buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
            shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        }};
        if let Some(awq) = p.shared_down_w.awq_scale {
            hip!(gpu.fused_silu_mul_rotate_mq_awq(&shared_gate, &shared_up, awq, &x_rot_alias, p.smi))?;
        } else {
            hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, p.smi))?;
        }
        hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            &p.shared_down_w.buf, &x_rot_alias, p.x_residual, p.scalar_buf,
            p.shared_down_w.m, p.shared_down_w.k,
        ))?;
    } else {
        // Non-MQ4 shared expert down: only reached when A3B shared expert
        // uses a non-MQ4 dtype. Requires deltanet feature for sigmoid_f32.
        // Returns UnsupportedVariant for builds without the feature to keep
        // hipfire-dispatch compilable without deltanet.
        #[cfg(feature = "deltanet")]
        {
            hip!(gpu.sigmoid_f32(p.scalar_buf))?;
            let shared_hid = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, p.smi) };
            hip!(gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid))?;
            static GEMV_DOWN: OnceLock<GemvFamily> = OnceLock::new();
            let gemv = GEMV_DOWN.get_or_init(GemvFamily::new);
            let ctx = DispatchCtx::new(gpu);
            gemv.run_auto(&ctx, gpu, &p.shared_down_w, &shared_hid, p.ffn_out).map_err(|e| DispatchError::Hip(e.to_string()))?;
            hip!(gpu.scaled_add_inplace_gpu_scalar_f32(p.x_residual, p.ffn_out, p.scalar_buf))?;
        }
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe", variant: "shared-down-non-mq4-requires-deltanet",
            arch: "", quant: "",
        });
    }

    // ── Indexed routed experts ────────────────────────────────────────────────
    if res.routed_indexable_mq4 {
        hip!(gpu.ensure_mq_signs())?;
    }
    let xr = x_rot_local.expect("use_gpu_topk implies x_rot_local is Some");
    let gate_up_k = p.routed_gate_up_k;
    let down_m    = p.routed_down_m;
    let down_k    = p.routed_down_k;

    if res.routed_indexable_mq4 {
        hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs, p.topk_indices, xr,
            p.gate_batch, p.up_batch, 2 * p.mi, gate_up_k,
        ))?;
    } else if res.routed_indexable_mq6 {
        hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs, p.topk_indices, xr,
            p.gate_batch, p.up_batch, 2 * p.mi, gate_up_k,
        ))?;
    } else {
        // routed_indexable_paro
        hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs, p.topk_indices, xr,
            p.gate_batch, p.up_batch, 2 * p.mi, gate_up_k,
        ))?;
    }

    // Gate→down: fused silu+mul+rotate
    if res.routed_indexable_paro {
        let paro_down = p.routed_down_paro.as_ref()
            .expect("routed_indexable_paro implies down paro sidecar");
        hip!(gpu.fused_silu_mul_givens_rotate_f32(
            p.gate_batch, p.up_batch, p.rot_batch,
            &paro_down.pairs, &paro_down.theta, &paro_down.scales,
            p.k, p.mi, paro_down.krot,
        ))?;
    } else {
        // MQ4/MQ6: no AWQ on expert down weights for Phase 1 targets (A3B)
        hip!(gpu.fused_silu_mul_rotate_mq_batched(p.gate_batch, p.up_batch, p.rot_batch, p.mi, p.k))?;
    }

    // Expanded write
    if res.routed_indexable_mq4 {
        hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs, p.topk_indices, p.rot_batch, p.down_expanded,
            down_m, down_k, p.k, 1,
        ))?;
    } else if res.routed_indexable_mq6 {
        hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs, p.topk_indices, p.rot_batch, p.down_expanded,
            down_m, down_k, p.k, 1,
        ))?;
    } else {
        // paro
        hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
            p.expert_down_ptrs, p.topk_indices, p.rot_batch, p.down_expanded,
            down_m, down_k, p.k, 1,
        ))?;
    }

    hip!(gpu.moe_down_combine_k8_batched(p.down_expanded, p.topk_weights, p.x_residual, down_m, p.k, 1))?;

    Ok(())
}

/// Generic CPU-top-K MoE decode fallback. Restores the per-expert loop #393
/// deleted from `moe_ffn_decode_impl` (origin/master qwen35.rs). Fires for any
/// MoE layer the GPU-top-K fast path can't serve: `k != 8`, or a routed expert
/// dtype outside `{MQ4G256, MQ6G256, ParoQ4G128}` (e.g. a Q8-routed MoE).
///
/// Sequence mirrors master exactly:
///   1. softmax(router_logits)
///   2. download probs → CPU top-K select + sort + renorm
///   3. shared-expert down (identical to the GPU-top-K path's shared-down block)
///   4. per-expert routed loop: gate_up GEMV → silu·mul → down GEMV → scaled add
///
/// Step 4 uses `GemvFamily::run_auto`, which is the dispatch-crate equivalent of
/// master's `weight_gemv`: it auto-rotates (FWHT for MQ family / Givens for Paro)
/// when the routed dtype requires it, and runs plain otherwise — so this single
/// loop covers every routed dtype, matching master's generic `weight_gemv` arm.
///
/// `shared_gate` / `shared_up` are the gate-side GEMV outputs computed by the
/// caller (`run_moe_decode`), passed through so the shared-expert math is shared.
fn run_moe_decode_cpu_fallback(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams,
    shared_gate: &GpuTensor,
    shared_up: &GpuTensor,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) };
    }

    // Per-expert weights are required to iterate (master indexed
    // `ffn.experts[expert_idx]`). They are empty under paged residency, where
    // only the indexed GPU-top-K path is supported — same invariant as master.
    if p.routed_experts.is_empty() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe", variant: "cpu-topk-fallback-needs-resident-experts",
            arch: "", quant: "",
        });
    }

    let k = p.k;
    let mi = p.mi;
    let n_exp = p.n_exp;

    // ── 1+2. softmax → CPU top-K + renorm (verbatim from master) ──────────────
    hip!(gpu.softmax_f32(p.router_logits))?;
    let probs = hip!(gpu.download_f32(p.router_logits))?;
    let mut indices: Vec<usize> = (0..n_exp).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_indices: Vec<usize> = indices.into_iter().take(k).collect();
    topk_indices.sort_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut topk_weights: Vec<f32> = topk_indices.iter().map(|&i| probs[i]).collect();
    if p.norm_topk_prob {
        let sum: f32 = topk_weights.iter().sum();
        if sum > 0.0 {
            for w in topk_weights.iter_mut() { *w /= sum; }
        }
    }

    // ── 3. Shared-expert down (identical to the GPU-top-K shared-down block) ──
    if p.shared_down_w.dtype == DType::MQ4G256 {
        hip!(gpu.ensure_mq_signs())?;
        let x_rot_alias = unsafe { GpuTensor {
            buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
            shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        }};
        if let Some(awq) = p.shared_down_w.awq_scale {
            hip!(gpu.fused_silu_mul_rotate_mq_awq(shared_gate, shared_up, awq, &x_rot_alias, p.smi))?;
        } else {
            hip!(gpu.fused_silu_mul_rotate_mq(shared_gate, shared_up, &x_rot_alias, p.smi))?;
        }
        hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            &p.shared_down_w.buf, &x_rot_alias, p.x_residual, p.scalar_buf,
            p.shared_down_w.m, p.shared_down_w.k,
        ))?;
    } else {
        #[cfg(feature = "deltanet")]
        {
            hip!(gpu.sigmoid_f32(p.scalar_buf))?;
            let shared_hid = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, p.smi) };
            hip!(gpu.silu_mul_f32(shared_gate, shared_up, &shared_hid))?;
            static GEMV_DOWN_FB: OnceLock<GemvFamily> = OnceLock::new();
            let gemv = GEMV_DOWN_FB.get_or_init(GemvFamily::new);
            let ctx = DispatchCtx::new(gpu);
            gemv.run_auto(&ctx, gpu, &p.shared_down_w, &shared_hid, p.ffn_out)?;
            hip!(gpu.scaled_add_inplace_gpu_scalar_f32(p.x_residual, p.ffn_out, p.scalar_buf))?;
        }
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe", variant: "shared-down-non-mq4-requires-deltanet",
            arch: "", quant: "",
        });
    }

    // ── 4. Per-expert routed loop (master's generic `weight_gemv` arm) ────────
    static GEMV_FB: OnceLock<GemvFamily> = OnceLock::new();
    let gemv = GEMV_FB.get_or_init(GemvFamily::new);

    for (&expert_idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
        let (gate_up_w, down_w) = &p.routed_experts[expert_idx];

        // gate_up: y = W·x  (run_auto auto-rotates for MQ/Paro dtypes).
        {
            let ctx = DispatchCtx::new(gpu);
            gemv.run_auto(&ctx, gpu, gate_up_w, p.x_norm, p.gate_up_buf)?;
        }
        let gate_view = unsafe { slice_moe_f32_view(p.gate_up_buf, 0, mi) };
        let up_view = unsafe { slice_moe_f32_view(p.gate_up_buf, mi, mi) };

        // silu(gate)·up → ffn_hidden, then down GEMV, then weighted residual add.
        let hid_view = unsafe { slice_moe_f32_view(p.ffn_hidden, 0, mi) };
        hip!(gpu.silu_mul_f32(&gate_view, &up_view, &hid_view))?;
        {
            let ctx = DispatchCtx::new(gpu);
            gemv.run_auto(&ctx, gpu, down_w, &hid_view, p.ffn_out)?;
        }
        hip!(gpu.scaled_add_inplace_cpu_scalar_f32(p.x_residual, p.ffn_out, weight))?;
    }

    Ok(())
}

/// DeepSeek-V4 bias-aware MoE decode executor. Transcribes the routed sub-graph
/// of `hipfire-arch-deepseek4::forward::ffn_routed` (the fused
/// `expert_gate_up_blob` branch): bias-aware top-k select → indexed MQ2-Lloyd
/// gate_up → batched silu·mul·clamp → batched FWHT rotate → indexed MQ2-Lloyd
/// down with route-scaled residual accumulation into `ffn_out`.
///
/// The router GEMV + `sqrt_softplus` (producing `p.scores`) and the shared
/// expert stay model-owned — the shared expert seeds `p.ffn_out` and this arm
/// accumulates into it, so the model must run it first. Decode only
/// (`batch_size == 1`); batched prefill is the grouped executor (Step 8).
pub fn run_moe_decode_bias_aware(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeBiasAwareParams,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) };
    }
    if p.batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe", variant: "bias-aware-decode-requires-batch-1",
            arch: "", quant: "",
        });
    }

    // 1. Bias-aware top-K: select on (scores + bias), weight on the unbiased
    //    scores, normalize, then fold in route_scale — all in one launch.
    hip!(gpu.deepseek4_moe_topk_bias_aware_f32(
        p.scores, p.gate_bias, p.topk_indices, p.topk_weights,
        p.n_exp as i32, p.k_top as i32, p.route_scale,
    ))?;

    // 2. Indexed MQ2-Lloyd gate_up: all k_top experts in one launch
    //    (M = 2*mi; the kernel splits rows r<mi → gate, r>=mi → up).
    hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        p.expert_gate_up_ptrs, p.topk_indices, p.x_rot,
        p.gate_batch, p.up_batch, 2 * p.mi, p.hidden, p.k_top,
    ))?;

    // 3. Batched silu·mul·clamp (in-place into gate_batch) then batched FWHT rotate.
    hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
        p.gate_batch, p.up_batch, p.gate_batch, p.mi, p.k_top, p.swiglu_limit,
    ))?;
    hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, p.mi, p.k_top))?;

    // 4. Indexed MQ2-Lloyd down: atomic-accumulate Σ_k w_k·(W_down[e_k]·rot)
    //    into ffn_out. route_scale is already baked into topk_weights.
    hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        p.expert_down_ptrs, p.topk_indices, p.topk_weights, p.rot_batch,
        p.ffn_out, p.hidden, p.mi, p.k_top,
    ))?;

    Ok(())
}

/// MQ2-Lloyd grouped-GEMM kernel variant (deepseek4 research levers; default
/// `Lloyd4w` on gfx11+, `Base` otherwise). Selected once per gate_up/down call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupedLloydVariant {
    N32,
    Cnd,
    EightW,
    Nosync,
    Mmqload,
    Lloyd4w,
    Base,
}

/// Mirror of `ffn_batched`'s grouped-GEMM if/else-if ladder (priority order:
/// n32 > cnd > 8w > nosync > mmqload > 4w > base). `n32`/`cnd`/`eightw` apply
/// only on the 4w path; `use_nosync` ⊂ `use_mmqload` ⊂ `use_lloyd_4w`.
fn select_grouped_lloyd_variant(
    use_lloyd_4w: bool,
    n32: bool,
    cnd: bool,
    eightw: bool,
    use_mmqload: bool,
    use_nosync: bool,
) -> GroupedLloydVariant {
    if use_lloyd_4w && n32 {
        GroupedLloydVariant::N32
    } else if use_lloyd_4w && cnd {
        GroupedLloydVariant::Cnd
    } else if use_lloyd_4w && eightw {
        GroupedLloydVariant::EightW
    } else if use_nosync {
        GroupedLloydVariant::Nosync
    } else if use_mmqload {
        GroupedLloydVariant::Mmqload
    } else if use_lloyd_4w {
        GroupedLloydVariant::Lloyd4w
    } else {
        GroupedLloydVariant::Base
    }
}

/// Dispatch one MQ2-Lloyd grouped GEMM. All seven variants share the signature
/// `(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows)`, so
/// this is called identically for gate_up (m=2*im, k=hidden, x_row_div=k_top,
/// rows=B) and down (m=hidden, k=im, x_row_div=1, rows=B*k_top).
#[allow(clippy::too_many_arguments)]
fn dispatch_grouped_lloyd(
    gpu: &mut Gpu,
    variant: GroupedLloydVariant,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total_max: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    use GroupedLloydVariant as V;
    let r = match variant {
        V::N32 => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::Cnd => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::EightW => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::Nosync => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::Mmqload => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::Lloyd4w => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
        V::Base => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows),
    };
    r.map_err(|e| DispatchError::Hip(e.to_string()))
}

/// DeepSeek-V4 batched/prefill MoE executor. Transcribes the routed block of
/// `hipfire-arch-deepseek4::forward::ffn_batched`: routing (hash or bias-aware)
/// → routed experts (grouped GEMM when `batch_size >= gate`, else scalar K4
/// indexed) → combine into `p.ffn_out` (the shared expert already seeded it).
/// Router GEMV + `sqrt_softplus` and the shared expert stay model-owned.
pub fn run_moe_prefill_bias_aware(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeBiasAwarePrefillParams,
) -> Result<(), DispatchError> {
    use crate::families::moe::MoePrefillRouting;
    macro_rules! hip {
        ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) };
    }
    let (hidden, im, n_exp, k_top, batch_size) = (p.hidden, p.mi, p.n_exp, p.k_top, p.batch_size);

    // ── Routing → topk_indices / topk_weights ────────────────────────────────
    match &p.routing {
        MoePrefillRouting::Hash { tid2eid, tokens } => {
            hip!(gpu.hash_router_normalize_f32_batched(
                tid2eid, p.scores, tokens,
                p.topk_indices, p.topk_weights,
                n_exp as i32, k_top as i32, p.route_scale, batch_size as i32,
            ))?;
        }
        MoePrefillRouting::BiasAware { gate_bias } => {
            hip!(gpu.deepseek4_moe_topk_bias_aware_batched_f32(
                p.scores, gate_bias,
                p.topk_indices, p.topk_weights,
                n_exp as i32, k_top as i32, p.route_scale, batch_size as i32,
            ))?;
        }
    }

    // DIAG: dump per-layer topk indices ([B, k_top] i32) — off by default.
    if let Ok(path) = std::env::var("HIPFIRE_DEEPSEEK4_DUMP_TOPK") {
        use std::io::Write;
        let raw = hip!(gpu.download_f32(p.topk_indices))?;
        let n = batch_size * k_top;
        let mut indices: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(raw[i].to_bits() as i32);
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .map_err(|e| DispatchError::Hip(format!("dump_topk open {path}: {e:?}")))?;
        let header = [p.layer_idx as i32, batch_size as i32, k_top as i32];
        let header_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 12) };
        f.write_all(header_bytes).map_err(|e| DispatchError::Hip(format!("dump_topk header: {e:?}")))?;
        let data_bytes = unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4) };
        f.write_all(data_bytes).map_err(|e| DispatchError::Hip(format!("dump_topk data: {e:?}")))?;
    }

    // ── Grouped vs scalar gate ────────────────────────────────────────────────
    let gate_threshold: usize = std::env::var("HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let use_grouped = batch_size >= gate_threshold
        && std::env::var("HIPFIRE_DEEPSEEK4_MOE_GROUPED").as_deref() != Ok("0");

    // Shared research levers (read once; default 4w on gfx11+).
    let lloyd_4w_base = match std::env::var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
        Ok("0") => Some(false),
        Ok("1") => Some(true),
        _ => None,
    };
    let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
    let n32 = std::env::var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
    let cnd = std::env::var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
    let eightw = std::env::var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
    let mmqload_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
    let nosync_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");

    if use_grouped {
        const BLOCK_M: usize = 16;
        let m_total_max = batch_size * k_top + n_exp * BLOCK_M;

        // Scatter: histogram + offsets + permute (single launch).
        hip!(gpu.moe_scatter_fused_k8(
            p.topk_indices, p.expert_token_counts, p.expert_offsets,
            p.sorted_slot_index, p.expert_tile_ids, p.inverse_perm,
            batch_size * k_top, n_exp, m_total_max, BLOCK_M,
        ))?;

        // Grouped gate_up GEMM (M=2*im, K=hidden, x_row_div=k_top, rows=B).
        let use_lloyd_4w_gu = lloyd_4w_base.unwrap_or(arch_4w) && (2 * im) % 64 == 0 && hidden % 256 == 0;
        let use_mmqload_gu = use_lloyd_4w_gu && mmqload_env;
        let use_nosync_gu = use_mmqload_gu && nosync_env;
        let v_gu = select_grouped_lloyd_variant(use_lloyd_4w_gu, n32, cnd, eightw, use_mmqload_gu, use_nosync_gu);
        dispatch_grouped_lloyd(
            gpu, v_gu, p.expert_gate_up_ptrs, p.expert_tile_ids, p.sorted_slot_index,
            p.x_rot, p.y_gate_up_grouped, 2 * im, hidden, k_top, m_total_max, batch_size,
        )?;

        // Unscatter + SwiGLU·clamp.
        let use_fused_unscatter_silu = std::env::var("HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU")
            .map(|s| s != "0")
            .unwrap_or(false);
        if use_fused_unscatter_silu {
            hip!(gpu.moe_unscatter_silu_clamp_k8(
                p.y_gate_up_grouped, p.sorted_slot_index, p.gate_batch,
                im, k_top, m_total_max, p.swiglu_limit,
            ))?;
        } else {
            hip!(gpu.moe_gate_up_unscatter_k8(
                p.y_gate_up_grouped, p.sorted_slot_index, p.gate_batch, p.up_batch,
                im, k_top, m_total_max,
            ))?;
            hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
                p.gate_batch, p.up_batch, p.gate_batch, im, batch_size * k_top, p.swiglu_limit,
            ))?;
        }

        // FWHT rotate.
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Grouped down GEMM (M=hidden, K=im, x_row_div=1, rows=B*k_top).
        let use_lloyd_4w_dn = lloyd_4w_base.unwrap_or(arch_4w) && hidden % 64 == 0 && im % 256 == 0;
        let use_mmqload_dn = use_lloyd_4w_dn && mmqload_env;
        let use_nosync_dn = use_mmqload_dn && nosync_env;
        let v_dn = select_grouped_lloyd_variant(use_lloyd_4w_dn, n32, cnd, eightw, use_mmqload_dn, use_nosync_dn);
        dispatch_grouped_lloyd(
            gpu, v_dn, p.expert_down_ptrs, p.expert_tile_ids, p.sorted_slot_index,
            p.rot_batch, p.y_down_grouped, hidden, im, 1, m_total_max, batch_size * k_top,
        )?;

        // Down-combine: weighted Σ over k_top slots, per (token, m), into ffn_out.
        hip!(gpu.moe_down_combine_grouped_k8(
            p.y_down_grouped, p.inverse_perm, p.topk_weights, p.ffn_out,
            hidden, k_top, batch_size,
        ))?;
    } else {
        // ── Scalar K4 path (batch_size < gate, or grouped opt-out) ──
        hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
            p.expert_gate_up_ptrs, p.topk_indices, p.x_rot,
            p.gate_batch, p.up_batch, 2 * im, hidden, k_top, batch_size,
        ))?;
        hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
            p.gate_batch, p.up_batch, p.gate_batch, im, batch_size * k_top, p.swiglu_limit,
        ))?;
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Down: deterministic expanded+combine (default; bit-reproducible for
        // spec-decode) vs non-deterministic atomic-accumulate.
        let deterministic =
            std::env::var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref() != Ok("0");
        if deterministic {
            hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                p.expert_down_ptrs, p.topk_indices, p.rot_batch, p.down_expert_outputs,
                hidden, im, k_top, batch_size,
            ))?;
            hip!(gpu.moe_down_combine_k8_batched(
                p.down_expert_outputs, p.topk_weights, p.ffn_out, hidden, k_top, batch_size,
            ))?;
        } else {
            hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                p.expert_down_ptrs, p.topk_indices, p.topk_weights, p.rot_batch, p.ffn_out,
                hidden, im, k_top, batch_size,
            ))?;
        }
    }

    Ok(())
}

pub fn dispatch_fused(
    gpu: &mut Gpu,
    key: KernelKey,
    params: &PipelineParams,
) -> Result<(), DispatchError> {
    let params = match params {
        PipelineParams::Linear(p) => p,
        PipelineParams::Moe(p) => return run_moe_decode(gpu, p),
    };
    macro_rules! hip {
        ($e:expr) => { $e.map_err(|e| DispatchError::Hip(e.to_string())) };
    }
    match key {
        KernelKey::GemvMfp4G32Fused => {
            gpu.ensure_mq_signs().map_err(|e| DispatchError::Hip(e.to_string()))?;
            let x_rot = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![params.k],
                    dtype: rdna_compute::DType::F32,
                }
            };
            hip!(gpu.gemv_mfp4g32_with_rotate(
                params.buf, params.x, params.y, &x_rot, params.m, params.k,
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "pipeline_fused", variant: "unknown",
            arch: "", quant: "",
        }),
    }
}
