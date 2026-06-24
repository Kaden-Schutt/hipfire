// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Conv1d dispatch (depthwise causal short-conv: decode, gated-decode, SiLU
//! split / routed / tree variants — the Mamba/LFM2/DeltaNet short-conv path).
//! Split out of `dispatch/mod.rs` (dispatch-refactor Phase 1, M1). Pure move.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// 1D causal conv (kernel_size=4) for decode. Updates ring buffer state.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_decode_f32(
        &mut self,
        output: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        n_channels: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_decode",
            kernels::CONV1D_DECODE_SRC,
            "conv1d_decode_f32",
        )?;
        let op = output.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = n_channels * 4 * 6;
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_decode_f32", bytes);
        let result = self.launch_maybe_blob(
            "conv1d_decode_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(op);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_i32(nc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Mamba-2 (nemotron_h) xBC short-conv decode step: depthwise causal conv1d
    /// (kernel_size=4) over `n_channels = conv_dim` channels, + per-channel bias
    /// (`use_conv_bias=true`), then SiLU over all channels. The caller splits the
    /// activated output into x / B / C afterwards. `state` is `[n_channels × 3]`
    /// (rolling K-1 history, updated in place). See
    /// `kernels/src/conv1d_bias_silu_decode.hip`.
    pub fn conv1d_bias_silu_decode_f32(
        &mut self,
        output: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        state: &GpuTensor,
        n_channels: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_bias_silu_decode",
            kernels::CONV1D_BIAS_SILU_DECODE_SRC,
            "conv1d_bias_silu_decode_f32",
        )?;
        let op = output.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (n_channels as u32).div_ceil(block);
        let func = &self.functions["conv1d_bias_silu_decode_f32"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// LFM2 LIV double-gated short-conv, single-token decode. Reads the in_proj
    /// output `bcx` [batch, 3*channels] (B | C_gate | x layout), applies the
    /// B*x pre-gate, runs the depthwise causal conv over the rolling `state`
    /// [batch, channels, K-1] history, applies the C_gate post-gate into
    /// `out_y` [batch, channels], and advances `state` in place. kernel_size K
    /// is a runtime arg (LFM2 K=3); conv_bias is always false.
    pub fn conv1d_gated_decode_f32(
        &mut self,
        bcx: &GpuTensor,
        state: &GpuTensor,
        weight: &GpuTensor,
        out_y: &GpuTensor,
        batch: usize,
        channels: usize,
        kernel_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_gated_decode",
            kernels::CONV1D_GATED_DECODE_SRC,
            "conv1d_gated_decode_f32",
        )?;
        let func = &self.functions["conv1d_gated_decode_f32"];
        let mut bp = bcx.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut oyp = out_y.buf.as_ptr();
        let mut bb = batch as i32;
        let mut cc = channels as i32;
        let mut kk = kernel_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut oyp as *mut _ as *mut c_void,
            &mut bb as *mut _ as *mut c_void,
            &mut cc as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (((batch * channels) as u32) + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Fused conv1d (kernel_size=4) + SiLU decode.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_f32(
        &mut self,
        output: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        n_channels: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("conv1d_silu", kernels::CONV1D_SILU_SRC, "conv1d_silu_f32")?;
        let func = &self.functions["conv1d_silu_f32"];
        let mut op = output.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused conv1d+SiLU that writes directly to Q/K/V buffers, replacing
    /// the conv1d_silu_f32 + three DtoD split copies in the DeltaNet path.
    /// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] — matches the
    /// wqkv projection output layout.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_f32(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.conv1d_silu_split_f32_n(q_out, k_out, v_out, input, weight, state, k_dim, v_dim, 1)
    }
    /// Batched conv1d + silu + Q/K/V split. Processes `n_tokens` tokens in
    /// order through the conv, advancing the ring-buffer state N times
    /// (identical state trajectory to calling the single-token variant N
    /// times). `input` / `q_out` / `k_out` / `v_out` are all [N × stride]
    /// row-major.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_f32_n(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_gfx1151() && n_tokens >= 64 {
            return self.conv1d_silu_split_f32_n_gfx1151(
                q_out, k_out, v_out, input, weight, state, k_dim, v_dim, n_tokens,
            );
        }
        self.ensure_kernel(
            "conv1d_silu_split",
            kernels::CONV1D_SILU_SPLIT_SRC,
            "conv1d_silu_split_f32",
        )?;
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nt = n_tokens as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer =
            crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_split_f32_n", bytes);
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_routed_f32_n(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state_ptrs: &GpuTensor,
        row_session_indices: &GpuTensor,
        ptr_layer_stride: usize,
        delta_layer_index: usize,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
        n_sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "conv1d_silu_split_routed",
            kernels::CONV1D_SILU_SPLIT_ROUTED_SRC,
            "conv1d_silu_split_routed_f32",
        )?;
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state_ptrs.buf.as_ptr();
        let rsp = row_session_indices.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = delta_layer_index as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nt = n_tokens as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &rsp as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "conv1d_silu_split_routed_f32_n",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_routed_f32",
            [grid, n_sessions as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_ptr(rsp);
                b.push_i32(ptr_stride);
                b.push_i32(layer);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Tree-aware variant of `conv1d_silu_split_f32_n`. `parent_indices[t]`
    /// is the linear slot index of token t's parent within the block, or
    /// a negative sentinel for pre-block ancestors: -1 selects conv_state[0]
    /// (most recent pre-block), -2 → state[1], -3 → state[2].
    ///
    /// Does NOT update conv_state — caller runs linear conv1d on the
    /// accepted spine post-acceptance to advance state.
    ///
    /// Port of SGLang's `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` branch in
    /// `causal_conv1d_update`. parent_indices supersedes retrieve_next_token
    /// / retrieve_next_sibling / retrieve_parent_token (the tree is already
    /// materialized host-side by `ddtree::linearize_tree`).
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_tree_f32_n(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        parent_indices: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1151 =
            self.arch_caps.is_gfx1151() && self.flags.conv1d_tree_gfx1151.unwrap_or(true);
        let (module_name, src, kernel_name, timer_name, grid_y) = if use_gfx1151 {
            (
                "conv1d_silu_split_tree_gfx1151",
                kernels::CONV1D_SILU_SPLIT_TREE_GFX1151_SRC,
                "conv1d_silu_split_tree_f32_gfx1151",
                "conv1d_silu_split_tree_f32_n_gfx1151",
                n_tokens as u32,
            )
        } else {
            (
                "conv1d_silu_split_tree",
                kernels::CONV1D_SILU_SPLIT_TREE_SRC,
                "conv1d_silu_split_tree_f32",
                "conv1d_silu_split_tree_f32_n",
                1,
            )
        };
        self.ensure_kernel(module_name, src, kernel_name)?;
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let pp = parent_indices.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nt = n_tokens as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", timer_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [grid, grid_y, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_ptr(pp);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(nt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
