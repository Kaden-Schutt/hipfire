// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::families::gemv::GemvFamily;
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;
#[allow(unused_imports)]
use hip_bridge;

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
                return Err(DispatchError::UnsupportedVariant {
                    family: "pipeline", variant: "gemv_in_pipeline",
                    arch: "", quant: "",
                });
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

    // ── Top-K (GPU path only in Phase 1) ─────────────────────────────────────
    if !res.use_gpu_topk {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe", variant: "cpu-topk-fallback",
            arch: "", quant: "",
        });
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
