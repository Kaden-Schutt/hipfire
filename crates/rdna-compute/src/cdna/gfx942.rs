// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx942 operations.
//!
//! Gfx942Device is the architecture proof: its field and constructor are
//! private, it does not dereference to Gpu, and it exposes only operations
//! implemented by gfx942-owned sources. This keeps model code from reaching a
//! CDNA kernel through a string/feature flag alone.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::{Gpu, GpuTensor};

const GROUPED_OLORA_E8_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx942.hip");
const GROUPED_OLORA_E8_KERNEL: &str = "gemv_mfp4g32_e8_soa_grouped_gfx942";
const GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_wave64x4.gfx942.hip");
const GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL: &str =
    "gemv_mfp4g32_e8_soa_grouped_wave64x4_gfx942_candidate";
const INDEXER_TOP_K_BUF_PARALLEL_SRC: &str =
    include_str!("../../../../kernels/src/indexer_top_k_buf_parallel.gfx942.hip");
const INDEXER_TOP_K_BUF_PARALLEL_KERNEL: &str = "indexer_top_k_buf_parallel_gfx942";
// Re-export of the frozen kernels.rs contract const (single include_str! site;
// all other gfx942 sources in this file predate the kernels.rs re-export
// convention and keep their local includes).
const INDEXER_TOP_K_BUF_BOUNDED_SRC: &str =
    crate::kernels::INDEXER_TOP_K_BUF_BOUNDED_GFX942_SRC;
const INDEXER_TOP_K_BUF_BOUNDED_KERNEL: &str = "indexer_top_k_buf_parallel_gfx942_bounded";
const MQ2_LLOYD_GATE_UP_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.gfx942.hip");
const MQ2_LLOYD_GATE_UP_WAVE64_KERNEL: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx942";
const MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_wave64x8.gfx942.hip");
const MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_gate_up_wave64x8_gfx942_candidate";
const MQ_ROTATE_X_WAVE64_SRC: &str = include_str!("../../../../kernels/src/mq_rotate_x.gfx942.hip");
const MQ_ROTATE_X_WAVE64_KERNEL: &str = "mq_rotate_x_gfx942";
const MQ2_LLOYD_DOWN_EXPANDED_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4.gfx942.hip");
const MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx942";
const MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.gfx942.hip");
const MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_gfx942";

/// A mutable GPU borrow proven to target exact gfx942.
///
/// The constructor is intentionally available only through Gpu::try_gfx942.
/// There is no Deref or escape hatch to the underlying generic context.
pub struct Gfx942Device<'gpu> {
    gpu: &'gpu mut Gpu,
}

impl Gpu {
    /// Borrow this context as an exact-gfx942 device.
    ///
    /// gfx940/gfx941 are deliberately rejected even though they are CDNA3.
    pub fn try_gfx942(&mut self) -> Option<Gfx942Device<'_>> {
        self.arch_caps
            .is_gfx942()
            .then_some(Gfx942Device { gpu: self })
    }
}

impl Gfx942Device<'_> {
    /// Native wave64 MQ2-Lloyd gate/up baseline for DS4 routed experts.
    pub fn mq2_lloyd_moe_gate_up_wave64(
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
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 gate/up requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            MQ2_LLOYD_GATE_UP_WAVE64_SRC,
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            [m as u32, k_top as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
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

    /// Exact-shape eight-wave gate/up candidate with workgroup-shared x tiles.
    pub fn mq2_lloyd_moe_gate_up_wave64x8_candidate(
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
        self.gpu.bind_thread()?;
        assert_eq!(m, 4096, "gfx942 wave64x8 MQ2 gate/up requires M=4096");
        assert_eq!(k, 4096, "gfx942 wave64x8 MQ2 gate/up requires K=4096");
        assert_eq!(k_top, 6, "gfx942 wave64x8 MQ2 gate/up requires top-k=6");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_SRC,
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
            [512, 6, 1],
            [512, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob
            },
        )
    }

    /// Exact-gfx942 wave64 MQ G256 rotation for a batch of routed experts.
    ///
    /// The source and symbol are private to this sealed device so the DS4
    /// backend cannot accidentally select them through a generic arch string.
    pub fn mq_rotate_x_wave64_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ rotate requires K%256=0");
        self.gpu.ensure_kernel(
            MQ_ROTATE_X_WAVE64_KERNEL,
            MQ_ROTATE_X_WAVE64_SRC,
            MQ_ROTATE_X_WAVE64_KERNEL,
        )?;
        self.gpu.ensure_mq_signs()?;
        let x_ptr = x.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let signs1_ptr = self
            .gpu
            .scratch
            .mq_signs1
            .as_ref()
            .expect("ensure_mq_signs populated signs1")
            .buf
            .as_ptr();
        let signs2_ptr = self
            .gpu
            .scratch
            .mq_signs2
            .as_ref()
            .expect("ensure_mq_signs populated signs2")
            .buf
            .as_ptr();
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &x_rot_ptr as *const _ as *mut c_void,
            &signs1_ptr as *const _ as *mut c_void,
            &signs2_ptr as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.gpu.hip, "fwht", MQ_ROTATE_X_WAVE64_KERNEL, bytes);
        let result = self.gpu.launch_maybe_blob(
            MQ_ROTATE_X_WAVE64_KERNEL,
            [((k / 256) * batch_size) as u32, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(x_rot_ptr);
                blob.push_ptr(signs1_ptr);
                blob.push_ptr(signs2_ptr);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        self.gpu.invalidate_x_caches_for(x_rot_ptr);
        result
    }

    /// Exact-gfx942 deterministic MQ2-Lloyd down projection.
    ///
    /// This preserves the incumbent expanded per-rank output and the caller's
    /// fixed-order combine. It never selects the atomic benchmark-only route.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_wave64(
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
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
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
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            [m.div_ceil(2) as u32, k_top as u32, batch_size as u32],
            [64, 1, 1],
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

    /// Exact-gfx942 atomic residual MQ2-Lloyd down projection used by the
    /// model's four static hash-routed layers. This restores the pre-backend
    /// wave64 route without exposing it through the generic `Gpu` surface.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_residual_wave64(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_SRC,
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let topk_weights_ptr = topk_weights.buf.as_ptr();
        let rot_batch_ptr = rot_batch.buf.as_ptr();
        let residual_ptr = residual.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &topk_weights_ptr as *const _ as *mut c_void,
            &rot_batch_ptr as *const _ as *mut c_void,
            &residual_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            [m as u32, k_top as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(topk_weights_ptr);
                blob.push_ptr(rot_batch_ptr);
                blob.push_ptr(residual_ptr);
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

    /// Exact-gfx942 deterministic indexer top-K.
    ///
    /// `bounded == false` selects the O(N^2) rank-count reference
    /// (`indexer_top_k_buf_parallel_gfx942`). `bounded == true` selects the
    /// opt-in O(N log^2 K) tile-merge bitonic port
    /// (`indexer_top_k_buf_parallel_gfx942_bounded`): a drop-in with
    /// identical dispatch shape and byte-identical output. The selected
    /// symbol keys both `ensure_kernel` and the launch, so the two modules
    /// never share a cache entry.
    ///
    /// This is intentionally not exposed through the generic `Gpu` surface:
    /// DeepSeek4 must hold a verified model-owned gfx942 backend before it can
    /// select the CDNA code object.
    pub fn indexer_top_k_buf_parallel(
        &mut self,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_k: i32,
        bounded: bool,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        let (src, kernel) = if bounded {
            (INDEXER_TOP_K_BUF_BOUNDED_SRC, INDEXER_TOP_K_BUF_BOUNDED_KERNEL)
        } else {
            (INDEXER_TOP_K_BUF_PARALLEL_SRC, INDEXER_TOP_K_BUF_PARALLEL_KERNEL)
        };
        self.gpu.ensure_kernel(kernel, src, kernel)?;
        let scores_ptr = scores.buf.as_ptr();
        let top_indices_ptr = top_indices.buf.as_ptr();
        let n_compressed_ptr = n_compressed_buf.buf.as_ptr();
        let k_ptr = k_buf.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &scores_ptr as *const _ as *mut c_void,
            &top_indices_ptr as *const _ as *mut c_void,
            &n_compressed_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &n_idx_heads as *const _ as *mut c_void,
            &max_k as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            kernel,
            [n_idx_heads as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(scores_ptr);
                blob.push_ptr(top_indices_ptr);
                blob.push_ptr(n_compressed_ptr);
                blob.push_ptr(k_ptr);
                blob.push_i32(n_idx_heads);
                blob.push_i32(max_k);
                blob
            },
        )
    }

    /// One-dispatch block-diagonal qt35 projection for DeepSeek4 O-LoRA:
    /// A[G,M,K] @ x[G,K] -> y[G,M].
    ///
    /// This is the retained baseline while a full-wave64 persistent projection
    /// engine is built. It is owned here so no RDNA/model route can select it.
    pub fn grouped_olora_e8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k % 256 == 0,
            "grouped E8 gfx942 kernel requires K%256==0, got K={k}"
        );
        self.gpu.ensure_kernel(
            GROUPED_OLORA_E8_KERNEL,
            GROUPED_OLORA_E8_SRC,
            GROUPED_OLORA_E8_KERNEL,
        )?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let groups_i32 = groups as i32;
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &groups_i32 as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            GROUPED_OLORA_E8_KERNEL,
            [m.div_ceil(2) as u32, groups as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(groups_i32);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Exact-shape native-wave64 candidate for the DS4 O-LoRA projection.
    ///
    /// Four wave64s compute four rows in one 256-thread workgroup. This is a
    /// channel-only candidate: no product dispatch calls it.
    pub fn grouped_olora_e8_wave64x4_candidate(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert_eq!(groups, 8, "gfx942 wave64x4 candidate requires G=8");
        assert_eq!(m, 1024, "gfx942 wave64x4 candidate requires M=1024");
        assert_eq!(k, 4096, "gfx942 wave64x4 candidate requires K=4096");
        self.gpu.ensure_kernel(
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_SRC,
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
        )?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
            [256, 8, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::arch_caps::ArchCaps;
    use crate::FeatureFlags;
    use std::sync::Arc;

    #[test]
    fn exact_gfx942_cap_is_not_a_cdna3_family_alias() {
        let caps = |arch| ArchCaps::new(arch, Arc::new(FeatureFlags::for_test(arch)));
        assert!(caps("gfx942").is_gfx942());
        for arch in ["gfx940", "gfx941", "gfx1151", "gfx1201"] {
            assert!(!caps(arch).is_gfx942(), "{arch} admitted as gfx942");
        }
    }
}
