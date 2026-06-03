// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::{Gpu, GpuTensor};

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
    // Moe(MoeParams<'a>) added in Task 4
}

pub fn execute_pipeline(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[PipelineOp],
    params: &PipelineParams,
    dtype: rdna_compute::DType,
    registry: &KernelRegistry,
) -> Result<(), DispatchError> {
    let params = match params {
        PipelineParams::Linear(p) => p,
    };
    if let Some(key) = find_fused(registry, ctx, dtype, steps) {
        return dispatch_fused(gpu, key, params);
    }

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

pub fn dispatch_fused(
    gpu: &mut Gpu,
    key: KernelKey,
    params: &PipelineParams,
) -> Result<(), DispatchError> {
    let params = match params {
        PipelineParams::Linear(p) => p,
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
