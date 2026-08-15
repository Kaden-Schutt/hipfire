// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1201 operations.
//!
//! `Gfx1201Device` is the architecture proof. Its field and constructor are
//! private, it does not dereference to [`Gpu`], and it exposes only operations
//! implemented by gfx1201-owned sources.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::{Gpu, GpuTensor};

const MQ2_LLOYD_GATE_UP_EP_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_ep.gfx1201.hip");
const MQ2_LLOYD_GATE_UP_EP_KERNEL: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx1201_ep";
const MQ2_LLOYD_GATE_UP_COMPACT_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_compact_ep.gfx1201.hip"
);
const MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_compact_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_EP_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_ep.gfx1201.hip");
const MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_compact_ep.gfx1201.hip"
);
const MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_compact_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_ep.gfx1201.hip"
);
const MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_gfx1201_ep";

/// A mutable GPU borrow proven to target exact gfx1201.
///
/// The constructor is intentionally available only through
/// [`Gpu::try_gfx1201`]. There is no `Deref` or escape hatch to the generic
/// context.
pub struct Gfx1201Device<'gpu> {
    gpu: &'gpu mut Gpu,
}

impl Gpu {
    /// Borrow this context as an exact-gfx1201 device.
    pub fn try_gfx1201(&mut self) -> Option<Gfx1201Device<'_>> {
        self.arch_caps
            .is_gfx1201()
            .then_some(Gfx1201Device { gpu: self })
    }
}

impl Gfx1201Device<'_> {
    /// Micro-screen sister of [`Self::mq2_lloyd_moe_gate_up_ep`] which uses
    /// one workgroup per output row and loops only the rank-owned routed slots.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_gate_up_compact_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k.is_multiple_of(256),
            "gfx1201 MQ2 gate/up requires K%256=0"
        );
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
            MQ2_LLOYD_GATE_UP_COMPACT_EP_SRC,
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        )
    }

    /// DS4 TP/EP decode gate/up which keeps the fixed six-slot graph but skips
    /// MQ2 work for slots whose pointer resolves to the rank-local zero dummy.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_gate_up_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k.is_multiple_of(256),
            "gfx1201 MQ2 gate/up requires K%256=0"
        );
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
            MQ2_LLOYD_GATE_UP_EP_SRC,
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer =
            crate::profile::begin_timer(&self.gpu.hip, "gemv", MQ2_LLOYD_GATE_UP_EP_KERNEL, bytes);
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Deterministic DS4 TP/EP down projection. Ownership is derived from the
    /// gate/up pointer table because non-owned down slots deliberately reuse a
    /// valid compact weight pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Micro-screen sister of [`Self::mq2_lloyd_moe_down_expanded_ep`] which
    /// uses one workgroup per output row and loops rank-owned routed slots.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_compact_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        )
    }

    /// Micro-screen sister of [`Self::mq2_lloyd_moe_down_expanded_ep`] which
    /// cooperatively stages each four-entry MQ2 codebook in LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_lds_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }
}
