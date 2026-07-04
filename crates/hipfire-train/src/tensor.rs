// SPDX-License-Identifier: Apache-2.0
//! `TrainTensor`: an fp32 GPU tensor with an optional gradient buffer.
//!
//! The training graph is built from these. A frozen parameter (the quantized/
//! fp32 base) carries no grad; a trainable parameter (LoRA `A`/`B`) and every
//! activation that sits on a path to a trainable param carries `.grad`.

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub struct TrainTensor {
    /// Forward value, fp32, row-major, logical shape in `value.shape`.
    pub value: GpuTensor,
    /// Gradient w.r.t. this tensor (same shape), allocated lazily for nodes
    /// that need it. `None` for frozen leaves.
    pub grad: Option<GpuTensor>,
    /// Whether the optimizer should step this tensor (true only for trainable
    /// leaf parameters, e.g. LoRA adapters).
    pub trainable: bool,
}

impl TrainTensor {
    /// Wrap an existing fp32 GPU buffer as a frozen (no-grad) leaf.
    pub fn frozen(value: GpuTensor) -> Self {
        Self {
            value,
            grad: None,
            trainable: false,
        }
    }

    /// Wrap an fp32 GPU buffer as a trainable leaf and allocate its grad.
    pub fn trainable(gpu: &mut Gpu, value: GpuTensor) -> HipResult<Self> {
        let grad = gpu.zeros(&value.shape, DType::F32)?;
        Ok(Self {
            value,
            grad: Some(grad),
            trainable: true,
        })
    }

    /// Number of logical elements.
    pub fn numel(&self) -> usize {
        self.value.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.value.shape
    }

    /// Ensure a zero-initialized grad buffer exists (for activations whose grad
    /// is accumulated during backward). Idempotent.
    pub fn ensure_grad(&mut self, gpu: &mut Gpu) -> HipResult<&GpuTensor> {
        if self.grad.is_none() {
            self.grad = Some(gpu.zeros(&self.value.shape, DType::F32)?);
        }
        Ok(self.grad.as_ref().unwrap())
    }
}
