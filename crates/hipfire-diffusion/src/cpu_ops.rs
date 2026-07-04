// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runtime-context matmul dispatch that stays with the pipeline.
//!
//! The pure CPU-reference tensor ops now live in the `hipfire-cpu` backend crate
//! (re-exported at this crate's root). This one dispatcher remains because it
//! consults the `DiffusionGenerationRuntimeContext` to pick the ROCm GPU linear
//! path when a device is active, falling back to the CPU-reference matmul.

use crate::{
    linear_optional_bias_with_runtime_context, matmul_vector, shape2, CpuTensor, DiffusionError,
    DiffusionGenerationRuntimeContext, DiffusionResult,
};

pub(crate) fn matmul_vector_with_runtime_context(
    vector: &[f32],
    matrix: &CpuTensor,
    runtime_context: &mut DiffusionGenerationRuntimeContext,
) -> DiffusionResult<Vec<f32>> {
    let [rows, cols] = shape2(matrix)?;
    if runtime_context.rocm_device_id().is_some() && vector.len() == cols {
        let input = CpuTensor {
            shape: vec![1, cols],
            data: vector.to_vec(),
        };
        let output =
            linear_optional_bias_with_runtime_context(&input, matrix, None, runtime_context)?;
        return Ok(output.data);
    }
    if runtime_context.rocm_device_id().is_some() && vector.len() != rows {
        return Err(DiffusionError::InvalidMetadata(format!(
            "vector length {} does not match projection matrix shape {:?}",
            vector.len(),
            matrix.shape
        )));
    }
    matmul_vector(vector, matrix).map_err(Into::into)
}
