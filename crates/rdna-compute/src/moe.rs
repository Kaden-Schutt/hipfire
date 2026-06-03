// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE scatter, permute, combine, and unscatter dispatch methods.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

impl Gpu {
    /// Combine pass for the atomic-free MoE down path. Sums K_TOP expert
    /// outputs per (token, m) weighted by topk_weights, accumulates into
    /// the residual stream. No cross-token contention — each token writes
    /// to its own M-column slice.
    pub fn moe_down_combine_k8_batched(
        &mut self,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        topk_weights: &GpuTensor,   // [batch_size × k_top] f32
        x_residual: &GpuTensor,     // [batch_size × m] f32 in-place +=
        m: usize, k_top: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_down_combine_k8_batched",
            kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC,
            "moe_down_combine_k8_batched",
        )?;
        let eop = expert_outputs.buf.as_ptr();
        let wp  = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &eop as *const _ as *mut c_void,
            &wp  as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // BW: expert_outputs read N*K_TOP*M, topk_weights N*K_TOP, x_residual r+w 2*N*M.
        let bytes = (batch_size * k_top * m + batch_size * k_top + 2 * batch_size * m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_down_combine_k8_batched", bytes,
        );
        let block_m: u32 = 256;
        let grid_x = (m as u32 + block_m - 1) / block_m;
        let result = self.launch_maybe_blob(
            "moe_down_combine_k8_batched",
            [grid_x, batch_size as u32, 1], [block_m, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(eop); b.push_ptr(wp); b.push_ptr(xrp);
                b.push_i32(m_val); b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 1: per-expert histogram.
    /// Single-CTA LDS-atomic histogram of `topk_indices[total_slots]`.
    /// Output `expert_token_counts[num_experts]` holds RAW counts; Phase 2
    /// rewrites them in place as padded counts.
    pub fn moe_scatter_histogram_k8(
        &mut self,
        topk_indices: &GpuTensor,        // [total_slots] i32
        expert_token_counts: &GpuTensor, // [num_experts] i32, written
        total_slots: usize,
        num_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_histogram_k8",
            kernels::MOE_SCATTER_HISTOGRAM_K8_SRC,
            "moe_scatter_histogram_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let cp = expert_token_counts.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (total_slots + num_experts) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_scatter_histogram_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_scatter_histogram_k8",
            [1, 1, 1], [256, 1, 1], lds_bytes, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip); b.push_ptr(cp);
                b.push_i32(ts_val); b.push_i32(ne_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 2: pad + exclusive scan.
    /// Rewrites `expert_token_counts` raw → padded (to a multiple of
    /// `block_m`) and writes `expert_offsets[num_experts + 1]` with the
    /// exclusive prefix sum. `expert_offsets[num_experts]` is M_total.
    pub fn moe_scatter_offsets_k8(
        &mut self,
        expert_token_counts: &GpuTensor, // [E] i32, in: raw, out: padded
        expert_offsets: &GpuTensor,      // [E+1] i32, written
        num_experts: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_offsets_k8",
            kernels::MOE_SCATTER_OFFSETS_K8_SRC,
            "moe_scatter_offsets_k8",
        )?;
        let cp = expert_token_counts.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let ne_val = num_experts as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (3 * num_experts + 1) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_scatter_offsets_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_scatter_offsets_k8",
            [1, 1, 1], [256, 1, 1], lds_bytes, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp); b.push_ptr(op);
                b.push_i32(ne_val); b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 3: scatter + tile ids.
    /// Writes `sorted_slot_index[m_total]` with each flat slot index at
    /// its bucket position (padding stays at the -1 sentinel) and
    /// `expert_tile_ids[m_total / block_m]` for the grouped-GEMM loop.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_permute_k8(
        &mut self,
        topk_indices: &GpuTensor,      // [total_slots] i32
        expert_offsets: &GpuTensor,    // [E+1] i32, exclusive padded scan
        sorted_slot_index: &GpuTensor, // [m_total] i32, written
        expert_tile_ids: &GpuTensor,   // [m_total / block_m] i32, written
        inverse_perm: &GpuTensor,      // [total_slots] i32, written: flat → sorted_pos
        total_slots: usize,
        num_experts: usize,
        m_total: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_permute_k8",
            kernels::MOE_SCATTER_PERMUTE_K8_SRC,
            "moe_scatter_permute_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let invp = inverse_perm.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mt_val = m_total as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &invp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        // BW: topk_indices + offsets + sorted_slot_index (init + writes)
        //     + expert_tile_ids (writes).
        let bytes = (total_slots + num_experts + 2 * m_total + m_total / block_m.max(1)) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_scatter_permute_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_scatter_permute_k8",
            [1, 1, 1], [256, 1, 1], lds_bytes, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip); b.push_ptr(op); b.push_ptr(sp); b.push_ptr(tp);
                b.push_ptr(invp);
                b.push_i32(ts_val); b.push_i32(ne_val);
                b.push_i32(mt_val); b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused single-CTA scatter pipeline. Replaces histogram + offsets +
    /// permute with one launch — saves ~2 launches × ~75µs per MoE layer
    /// (≈2-3ms across 40 A3B layers).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_fused_k8(
        &mut self,
        topk_indices: &GpuTensor,        // [total_slots] i32
        expert_token_counts: &GpuTensor, // [E] i32, out: padded
        expert_offsets: &GpuTensor,      // [E+1] i32, out: exclusive scan
        sorted_slot_index: &GpuTensor,   // [m_total_max] i32, out
        expert_tile_ids: &GpuTensor,     // [m_total / block_m] i32, out
        inverse_perm: &GpuTensor,        // [total_slots] i32, out
        total_slots: usize,
        num_experts: usize,
        m_total_max: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_fused_k8",
            kernels::MOE_SCATTER_FUSED_K8_SRC,
            "moe_scatter_fused_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let cp = expert_token_counts.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let invp = inverse_perm.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mtm_val = m_total_max as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &invp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &mtm_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (total_slots + 2 * num_experts + 2 * total_slots + num_experts) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_scatter_fused_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_scatter_fused_k8",
            [1, 1, 1], [256, 1, 1], lds_bytes, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip); b.push_ptr(cp); b.push_ptr(op);
                b.push_ptr(sp); b.push_ptr(tp); b.push_ptr(invp);
                b.push_i32(ts_val); b.push_i32(ne_val);
                b.push_i32(mtm_val); b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Path 2 down combine. Per (token, m) iterates K_TOP slots via
    /// `inverse_perm[token*K_TOP + k]`, applies topk_weights, and += into
    /// `x_residual`. No atomic contention (each (token, m) is owned by
    /// one thread).
    pub fn moe_down_combine_grouped_k8(
        &mut self,
        y_down_grouped: &GpuTensor, // [m_total × dim] f32
        inverse_perm: &GpuTensor,   // [N*K_TOP] i32
        topk_weights: &GpuTensor,   // [N × K_TOP] f32
        x_residual: &GpuTensor,     // [N × dim] f32 in-place +=
        dim: usize,
        k_top: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_down_combine_grouped_k8",
            kernels::MOE_DOWN_COMBINE_GROUPED_K8_SRC,
            "moe_down_combine_grouped_k8",
        )?;
        let yp = y_down_grouped.buf.as_ptr();
        let ip = inverse_perm.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let dim_val = dim as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (dim as u32 + block - 1) / block;
        let bytes = (n * dim * 4 * 2 + n * k_top * 4 + n * k_top * 4) as usize;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_down_combine_grouped_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_down_combine_grouped_k8",
            [grid_x, n as u32, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp); b.push_ptr(ip); b.push_ptr(wp); b.push_ptr(xrp);
                b.push_i32(dim_val); b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Path 2 unscatter combine for gate_up. Reads Y_grouped[m_total ×
    /// 2*mi] and writes the gate half (rows 0..mi) into `y_gate[token,
    /// k_rank, :]` and the up half (rows mi..2*mi) into `y_up[token,
    /// k_rank, :]`, where (token, k_rank) is recovered from
    /// `sorted_slot_index[slot]`. Padding slots are skipped.
    pub fn moe_gate_up_unscatter_k8(
        &mut self,
        y_grouped: &GpuTensor,         // [m_total × (2*mi)] f32
        sorted_slot_index: &GpuTensor, // [m_total] i32
        y_gate: &GpuTensor,            // [N × K_TOP × mi] f32, written
        y_up: &GpuTensor,              // [N × K_TOP × mi] f32, written
        mi: usize,
        k_top: usize,
        m_total: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_gate_up_unscatter_k8",
            kernels::MOE_GATE_UP_UNSCATTER_K8_SRC,
            "moe_gate_up_unscatter_k8",
        )?;
        let yp = y_grouped.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let gp = y_gate.buf.as_ptr();
        let up = y_up.buf.as_ptr();
        let mi_val = mi as i32;
        let kt_val = k_top as i32;
        let mt_val = m_total as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (mi as u32 + block - 1) / block;
        // BW: Y_grouped read (m_total*2*mi*4) + y_gate write (m_total*mi*4)
        //     + y_up write (m_total*mi*4) + sorted_slot_index (m_total*4).
        let bytes = (m_total * 2 * mi + m_total * 2 * mi + m_total) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_gate_up_unscatter_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_gate_up_unscatter_k8",
            [grid_x, m_total as u32, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp); b.push_ptr(sp); b.push_ptr(gp); b.push_ptr(up);
                b.push_i32(mi_val); b.push_i32(kt_val); b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    pub fn moe_unscatter_silu_clamp_k8(
        &mut self,
        y_grouped: &GpuTensor,         // [m_total × (2*mi)] f32
        sorted_slot_index: &GpuTensor, // [m_total] i32
        moe_gate_batch: &GpuTensor,    // [N × K_TOP × mi] f32, written
        mi: usize,
        k_top: usize,
        m_total: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_unscatter_silu_clamp_k8",
            kernels::MOE_UNSCATTER_SILU_CLAMP_K8_SRC,
            "moe_unscatter_silu_clamp_k8",
        )?;
        let yp = y_grouped.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let gp = moe_gate_batch.buf.as_ptr();
        let mi_val = mi as i32;
        let kt_val = k_top as i32;
        let mt_val = m_total as i32;
        let mut swiglu_lim = swiglu_limit;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &mut swiglu_lim as *mut _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (mi as u32 + block - 1) / block;
        // BW: Y_grouped read (m_total*2*mi*4) + moe_gate_batch write
        // (m_total*mi*4) + sorted_slot_index (m_total*4).  Half the
        // write traffic vs the unfused path (no y_up output).
        let bytes = (m_total * 2 * mi + m_total * mi + m_total) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip, "elementwise", "moe_unscatter_silu_clamp_k8", bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_unscatter_silu_clamp_k8",
            [grid_x, m_total as u32, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp); b.push_ptr(sp); b.push_ptr(gp);
                b.push_i32(mi_val); b.push_i32(kt_val); b.push_i32(mt_val);
                b.push_f32(swiglu_lim);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }
    pub fn hash_router_normalize_f32(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        token_id: i32,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32",
            kernels::HASH_ROUTER_NORMALIZE_SRC,
            "hash_router_normalize_f32",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut tid = token_id;
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp); b.push_ptr(sp); b.push_ptr(ip); b.push_ptr(wp);
            b.push_i32(tid); b.push_i32(ne); b.push_i32(kv); b.push_f32(rs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32",
            [1, 1, 1], [1, 1, 1], 0, &mut params, blob_builder,
        )
    }
    pub fn hash_router_normalize_f32_batched(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_ids: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_batched",
            kernels::HASH_ROUTER_NORMALIZE_BATCHED_SRC,
            "hash_router_normalize_f32_batched",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_ids.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tb as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp); b.push_ptr(sp); b.push_ptr(tb);
            b.push_ptr(ip); b.push_ptr(wp);
            b.push_i32(ne); b.push_i32(kv); b.push_f32(rs);
            b.push_i32(bs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32_batched",
            [batch_size as u32, 1, 1], [1, 1, 1], 0, &mut params, blob_builder,
        )
    }
    pub fn hash_router_normalize_f32_buf(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_id_buf: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_buf",
            kernels::HASH_ROUTER_NORMALIZE_BUF_SRC,
            "hash_router_normalize_f32_buf",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_id_buf.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tb as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp); b.push_ptr(sp); b.push_ptr(tb);
            b.push_ptr(ip); b.push_ptr(wp);
            b.push_i32(ne); b.push_i32(kv); b.push_f32(rs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32_buf",
            [1, 1, 1], [1, 1, 1], 0, &mut params, blob_builder,
        )
    }
    pub fn deepseek4_moe_topk_bias_aware_batched_f32(
        &mut self,
        scores: &GpuTensor,    // [B, n_exp]
        bias: &GpuTensor,      // [n_exp]
        indices: &GpuTensor,   // [B, k_top]
        weights: &GpuTensor,   // [B, k_top]
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware_batched",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_BATCHED_SRC,
            "deepseek4_moe_topk_bias_aware_batched_f32",
        )?;
        let func = &self.functions["deepseek4_moe_topk_bias_aware_batched_f32"];
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [n_exp as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_moe_topk_bias_aware_f32(
        &mut self,
        scores: &GpuTensor,    // [n_exp] fp32
        bias: &GpuTensor,      // [n_exp] fp32 (zero if hash-routed)
        indices: &GpuTensor,   // [k_top] i32 (typed as F32; raw bytes)
        weights: &GpuTensor,   // [k_top] fp32
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_SRC,
            "deepseek4_moe_topk_bias_aware_f32",
        )?;
        let func = &self.functions["deepseek4_moe_topk_bias_aware_f32"];
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [n_exp as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_topk_kv_gather_batched_f32(
        &mut self,
        kv_cache: &GpuTensor,    // [N_compressed, head_dim] shared
        topk_idx: &GpuTensor,    // [B, K] i32
        out: &GpuTensor,         // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_batched",
            kernels::V4F_TOPK_KV_GATHER_BATCHED_SRC,
            "deepseek4_topk_kv_gather_batched_f32",
        )?;
        let func = &self.functions["deepseek4_topk_kv_gather_batched_f32"];
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [k_active as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_topk_kv_gather_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        max_k: i32,         // upper bound on K — sets the captured grid size
        head_dim: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_BUF_SRC,
            "deepseek4_topk_kv_gather_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let ncp = n_compressed_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &ncp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp); b.push_ptr(ip); b.push_ptr(op);
            b.push_ptr(kbp); b.push_ptr(ncp);
            b.push_i32(hd); b.push_i32(os); b.push_i32(co); b.push_f32(sc);
            b
        };
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_f32_buf",
            [max_k as u32, 1, 1], [head_dim as u32, 1, 1], 0,
            &mut params, blob_builder,
        )
    }
    pub fn deepseek4_topk_kv_gather_identity_batched_f32(
        &mut self,
        kv_cache: &GpuTensor,    // [N_compressed, head_dim] shared
        out: &GpuTensor,         // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        out_stride: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_batched",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SRC,
            "deepseek4_topk_kv_gather_identity_batched_f32",
        )?;
        let func = &self.functions["deepseek4_topk_kv_gather_identity_batched_f32"];
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [k_active as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_topk_kv_gather_identity_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC,
            "deepseek4_topk_kv_gather_identity_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp); b.push_ptr(op); b.push_ptr(kbp);
            b.push_i32(hd); b.push_i32(os);
            b
        };
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            [max_k as u32, 1, 1], [head_dim as u32, 1, 1], 0,
            &mut params, blob_builder,
        )
    }

}
