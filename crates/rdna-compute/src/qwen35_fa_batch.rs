// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S6-fa-prep-q8-pair launchers: batched full-attention prep and paired Q8
//! K/V cache writes for gfx1100 DFlash verify.
//!
//! Both kernels are bit-exact folds of the launches they replace (see the
//! `.hip` headers); admission (gfx1100, `DflashFusionCtx::ChainVerify`,
//! exact 16Q/2K + HD256 + NROT64 shapes, kill switch) is enforced by the
//! `batch_chunk_full_attn_prepare` caller and the `KvWriteQ8_0Batched`
//! dispatch arm. The launchers only validate shapes and enqueue via
//! `launch_maybe_blob` with a retained `KernargBlob`, so they stay
//! hipGraph-capture safe.
//!
//! Kernel sources are `include_str!`'d here (not via `crate::kernels`) so
//! this slice never edits the shared kernel registry owned by the scaffold.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::HipResult;

const FA_PREP_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/qwen35_fa_prep_batched.gfx1100.hip");
const KV_PAIR_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_q8_0_pair_batched.gfx1100.hip");

/// Fixed prep geometry: 16 Q heads + 2 K heads, head_dim 256, n_rot 64.
const FA_NQ: u32 = 16;
const FA_NK: u32 = 2;

#[cfg(feature = "deltanet")]
impl Gpu {
    /// Batched gfx1100 full-attention prep. Folds deinterleave + Q/K rmsnorm
    /// + partial half-split RoPE (4 launches) into one `[18, batch_size]`
    /// grid of 256-thread blocks.
    ///
    /// `k` is read pre-norm and written post-norm+rope in place. `positions`
    /// carries the physical KV slots; `pos_offset` (`compact_offset`) shifts
    /// only the RoPE phase. Buffers are `[batch × heads × 256]` row-major
    /// F32; weights are `[256]` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_fa_prep_batched_gfx1100(
        &mut self,
        q_interleaved: &GpuTensor,
        q: &GpuTensor,
        gate: &GpuTensor,
        k: &GpuTensor,
        q_weight: &GpuTensor,
        k_weight: &GpuTensor,
        positions: &GpuTensor,
        eps: f32,
        freq_base: f32,
        pos_offset: i32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                1,
                "qwen35_fa_prep_batched_gfx1100 is certified only on gfx1100",
            ));
        }
        if batch_size == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "qwen35_fa_prep_batched_gfx1100 requires batch_size >= 1",
            ));
        }
        self.ensure_kernel(
            "qwen35_fa_prep_batched_gfx1100",
            FA_PREP_BATCHED_SRC,
            "qwen35_fa_prep_batched_gfx1100",
        )?;

        let qip = q_interleaved.buf.as_ptr();
        let qp = q.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let qwp = q_weight.buf.as_ptr();
        let kwp = k_weight.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let ep = eps;
        let fb = freq_base;
        let po = pos_offset;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qip as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &qwp as *const _ as *mut c_void,
            &kwp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
            &fb as *const _ as *mut c_void,
            &po as *const _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        // Per (head, token): interleaved read + norm read + q/gate/k writes.
        let bytes = batch_size * ((FA_NQ + FA_NK) as usize * 256 * 4 * 2);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "qwen35_fa_prep_batched_gfx1100",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "qwen35_fa_prep_batched_gfx1100",
            [FA_NQ + FA_NK, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qip);
                b.push_ptr(qp);
                b.push_ptr(gp);
                b.push_ptr(kp);
                b.push_ptr(qwp);
                b.push_ptr(kwp);
                b.push_ptr(pp);
                b.push_f32(ep);
                b.push_f32(fb);
                b.push_i32(po);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Paired gfx1100 Q8 K/V batched write. Folds the two
    /// `kv_cache_write_q8_0_batched` launches (K, then V) into one
    /// `[2 * total_blocks, batch_size]` grid. Legacy single-arena addressing
    /// only (`dst + pos * per_pos_bytes + gid * 34`), matching the dispatch
    /// arm it serves; slot/independent variants keep their own launchers.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache_write_q8_0_pair_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                1,
                "kv_cache_write_q8_0_pair_batched is certified only on gfx1100",
            ));
        }
        if batch_size == 0 || n_kv_heads == 0 || head_dim % 32 != 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "kv_cache_write_q8_0_pair_batched requires batch>=1, kv_heads>=1, head_dim%32==0",
            ));
        }
        self.ensure_kernel(
            "kv_cache_write_q8_0_pair_batched_gfx1100",
            KV_PAIR_BATCHED_SRC,
            "kv_cache_write_q8_0_pair_batched_gfx1100",
        )?;

        let mut kd = k_dst.buf.as_ptr();
        let mut vd = v_dst.buf.as_ptr();
        let mut ks = k_src.buf.as_ptr();
        let mut vs = v_src.buf.as_ptr();
        let mut p = positions.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kd as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = batch_size * n_kv_heads * head_dim * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "kv_write",
            "kv_cache_write_q8_0_pair_batched_gfx1100",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "kv_cache_write_q8_0_pair_batched_gfx1100",
            [total_blocks * 2, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kd);
                b.push_ptr(vd);
                b.push_ptr(ks);
                b.push_ptr(vs);
                b.push_ptr(p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
