// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4-specific GPU operations.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;

impl Gpu {
    /// Fused E-series PLE activation over a packed per-layer-input buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_ple_gelu_mul_strided_f32(
        &mut self,
        gate: &GpuTensor,
        projection_all: &GpuTensor,
        hidden: &GpuTensor,
        batch: usize,
        ple_dim: usize,
        packed_dim: usize,
        layer_idx: usize,
    ) -> HipResult<()> {
        if batch == 0 || ple_dim == 0 {
            return Ok(());
        }
        let ple_dim_i = i32::try_from(ple_dim)
            .map_err(|_| HipError::new(0, "Gemma 4 PLE dimension overflow"))?;
        let packed_dim_i = i32::try_from(packed_dim)
            .map_err(|_| HipError::new(0, "Gemma 4 packed PLE dimension overflow"))?;
        let layer_offset_i = layer_idx
            .checked_mul(ple_dim)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| HipError::new(0, "Gemma 4 PLE layer offset overflow"))?;
        let batch_i =
            i32::try_from(batch).map_err(|_| HipError::new(0, "Gemma 4 PLE batch overflow"))?;
        let batch_grid =
            u32::try_from(batch).map_err(|_| HipError::new(0, "Gemma 4 PLE grid overflow"))?;
        let ple_dim_grid = u32::try_from(ple_dim)
            .map_err(|_| HipError::new(0, "Gemma 4 PLE grid dimension overflow"))?;

        self.bind_thread()?;
        self.ensure_kernel(
            "gemma4_ple_activation",
            kernels::GEMMA4_PLE_ACTIVATION_SRC,
            "gemma4_ple_gelu_mul_strided_f32",
        )?;
        let gate_ptr = gate.buf.as_ptr();
        let projection_ptr = projection_all.buf.as_ptr();
        let hidden_ptr = hidden.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &gate_ptr as *const _ as *mut c_void,
            &projection_ptr as *const _ as *mut c_void,
            &hidden_ptr as *const _ as *mut c_void,
            &ple_dim_i as *const _ as *mut c_void,
            &packed_dim_i as *const _ as *mut c_void,
            &layer_offset_i as *const _ as *mut c_void,
            &batch_i as *const _ as *mut c_void,
        ];
        let block = 256u32;
        self.launch_maybe_blob(
            "gemma4_ple_gelu_mul_strided_f32",
            [ple_dim_grid.div_ceil(block), batch_grid, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gate_ptr);
                blob.push_ptr(projection_ptr);
                blob.push_ptr(hidden_ptr);
                blob.push_i32(ple_dim_i);
                blob.push_i32(packed_dim_i);
                blob.push_i32(layer_offset_i);
                blob.push_i32(batch_i);
                blob
            },
        )
    }
}
