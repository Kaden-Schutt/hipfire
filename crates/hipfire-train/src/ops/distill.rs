// SPDX-License-Identifier: Apache-2.0
//! KL distillation loss (soft-target cross-entropy). One fused call: per-row
//! `loss = KL(teacher_p ‖ softmax(student))` and `d_logits = softmax(student) −
//! teacher_p`. `teacher_p` is a probability distribution (e.g. `softmax` of the
//! teacher's logits). Sum-reduction grad — divide by row count for a mean.

use hipfire_rdna::{Gpu, GpuTensor, HipResult};

pub fn distill_kl(
    gpu: &mut Gpu,
    student: &GpuTensor,
    teacher_p: &GpuTensor,
    loss: &GpuTensor,
    d_logits: &GpuTensor,
    rows: usize,
    v: usize,
) -> HipResult<()> {
    gpu.distill_kl_train(student, teacher_p, loss, d_logits, rows, v)
}
