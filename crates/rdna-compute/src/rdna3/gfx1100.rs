// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1100 operations.
//!
//! `Gfx1100Device` is an architecture proof: its field and constructor are
//! private, it does not dereference to `Gpu`, and it exposes only operations
//! that have an exact gfx1100 contract. Candidate methods remain unreachable
//! from model dispatch until their micro and product gates pass.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::{Gpu, GpuTensor};

const MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_lds.gfx1100.hip");
const MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_gfx1100_candidate";
const HARMONIC_SPLIT_COMBINE_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/moe_down_combine_harmonic_split.gfx1100.hip");
const HARMONIC_SPLIT_COMBINE_CANDIDATE_KERNEL: &str =
    "moe_down_combine_harmonic_split_gfx1100_candidate";

/// A mutable GPU borrow proven to target exact gfx1100.
///
/// The constructor is intentionally available only through
/// `Gpu::try_gfx1100`. There is no escape hatch to the underlying generic
/// context.
pub struct Gfx1100Device<'gpu> {
    gpu: &'gpu mut Gpu,
}

impl Gpu {
    /// Borrow this context as an exact-gfx1100 device.
    ///
    /// gfx1101/gfx1102 and gfx1151 are deliberately rejected: their broad
    /// wave32/WMMA capability overlap is not an exact product proof.
    pub fn try_gfx1100(&mut self) -> Option<Gfx1100Device<'_>> {
        self.arch_caps
            .is_gfx1100()
            .then_some(Gfx1100Device { gpu: self })
    }
}

impl Gfx1100Device<'_> {
    /// Shipping K4+LDS gate/up implementation under an exact-gfx1100 proof.
    ///
    /// The shared kernel was explicitly ported from the gfx1100 MQ3 pattern;
    /// this wrapper prevents the harmonic route from reaching it through a
    /// generic architecture string.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_gate_up_k4_lds(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        assert_eq!(m, 4096, "gfx1100 DS4 MQ2 gate/up requires M=4096");
        assert_eq!(k, 4096, "gfx1100 DS4 MQ2 gate/up requires K=4096");
        assert!(
            (1..=6).contains(&k_top),
            "gfx1100 DS4 MQ2 gate/up requires top-k 1..=6"
        );
        self.gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            expert_ptrs,
            topk_indices,
            x_rot,
            y_gate,
            y_up,
            m,
            k,
            k_top,
        )
    }

    /// Exact routed-down candidate with one cooperative codebook load
    /// per K4 group. Arithmetic and output ownership match the deterministic
    /// expanded incumbent; only redundant scalar loads are removed.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_lds_candidate(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        assert_eq!(m, 4096, "gfx1100 DS4 MQ2 down requires M=4096");
        assert_eq!(k, 2048, "gfx1100 DS4 MQ2 down requires K=2048");
        assert!(
            (1..=6).contains(&k_top),
            "gfx1100 DS4 MQ2 down requires top-k 1..=6"
        );
        assert_eq!(
            batch_size, 1,
            "gfx1100 harmonic MQ2 candidate is decode-only"
        );
        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_batch_ptr = rot_batch.buf.as_ptr();
        let expert_outputs_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_batch_ptr as *const _ as *mut c_void,
            &expert_outputs_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_CANDIDATE_KERNEL,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_batch_ptr);
                blob.push_ptr(expert_outputs_ptr);
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

    /// Candidate exact-order combine for owner-packed harmonic expert rows.
    ///
    /// `slot_sources` contains six u32 values. Bit 31 selects `remote_outputs`;
    /// the remaining bits select the packed row within that owner. Both an
    /// ordinary VRAM remote buffer and a HIP-mapped host alias are legal inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn harmonic_moe_down_combine_split_candidate(
        &mut self,
        local_outputs: &GpuTensor,
        remote_outputs: &GpuTensor,
        slot_sources: &GpuTensor,
        topk_weights: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k_top: usize,
    ) -> HipResult<()> {
        assert_eq!(m, 4096, "gfx1100 harmonic combine requires M=4096");
        assert_eq!(k_top, 6, "gfx1100 harmonic combine requires top-k 6");
        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(
            HARMONIC_SPLIT_COMBINE_CANDIDATE_KERNEL,
            HARMONIC_SPLIT_COMBINE_CANDIDATE_SRC,
            HARMONIC_SPLIT_COMBINE_CANDIDATE_KERNEL,
        )?;
        let local_ptr = local_outputs.buf.as_ptr();
        let remote_ptr = remote_outputs.buf.as_ptr();
        let sources_ptr = slot_sources.buf.as_ptr();
        let weights_ptr = topk_weights.buf.as_ptr();
        let residual_ptr = x_residual.buf.as_ptr();
        let m_i32 = m as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &local_ptr as *const _ as *mut c_void,
            &remote_ptr as *const _ as *mut c_void,
            &sources_ptr as *const _ as *mut c_void,
            &weights_ptr as *const _ as *mut c_void,
            &residual_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            HARMONIC_SPLIT_COMBINE_CANDIDATE_KERNEL,
            [(m as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(local_ptr);
                blob.push_ptr(remote_ptr);
                blob.push_ptr(sources_ptr);
                blob.push_ptr(weights_ptr);
                blob.push_ptr(residual_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        )
    }
}
