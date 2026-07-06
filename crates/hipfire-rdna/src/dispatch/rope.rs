// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RoPE + MQ/FWHT rotation dispatch (the rotate_x_mq scratch-prep bridge used by gemv/gemm/fused). Pure move (Phase 1 M2).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;

impl Gpu {
    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    /// Exposed so callers can batch one rotation across multiple GEMVs that share x
    /// (e.g., Q/K/V projections all consume the same post-RMSNorm x).
    pub fn rotate_x_mq(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        // `mq_rotate_x` lives inside the `gemv_mq4g256` module — precompile
        // writes the .hsaco/.hash sidecar under that module name, so the
        // runtime cache key here MUST match or we silently JIT on first use.
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x", bytes);
        let result = self.launch_kernargs(
            "mq_rotate_x",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr s1_ptr, ptr s2_ptr, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        // Same cache-key contract as `rotate_x_mq` — see comment there.
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = self.launch_kernargs(
            "mq_rotate_x",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr s1, ptr s2, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// FWHT-128 standalone rotation for MQ4G128 activations.
    ///
    /// Mirrors `rotate_x_mq` but targets G128 groups (32 threads × 4 elems).
    /// Grid: [k/128, 1, 1]. Block: [32, 1, 1].
    pub fn rotate_x_mq_128(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_kernel("gemv_mq4g128", kernels::GEMV_MQ4G128_SRC, "mq_rotate_x_128")?;
        let s1_ptr = self.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 128) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_128", bytes);
        let result = self.launch_kernargs(
            "mq_rotate_x_128",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr s1_ptr, ptr s2_ptr, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Phase A Stage A — F2 AWQ-aware variant of `rotate_x_mq`.
    ///
    /// Divides each input element by `awq_scale[i]` BEFORE the FWHT,
    /// completing the AWQ math `(W·s) · (x/s) = W·x` for the
    /// post-projection input-rotate path (o_proj / out_proj). Use when
    /// the upcoming linear carries `awq_scale = Some(...)`; otherwise call
    /// the non-AWQ `rotate_x_mq`.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K.
    pub fn rotate_x_mq_awq(
        &mut self,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            "rotate_x_mq_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        // Bandwidth: read x + awq_scale, 2x256 signs, write x_rot.
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "rotate_x_mq_awq", bytes);
        let result = self.launch_kernargs(
            "rotate_x_mq_awq",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr awp, ptr s1_ptr, ptr s2_ptr, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Phase A Stage A — F2 batched AWQ variant of `rotate_x_mq`.
    /// Grid.y is the batch dim — processes [N × K] x/x_rot.
    pub fn rotate_x_mq_awq_batched(
        &mut self,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            "rotate_x_mq_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "fwht", "rotate_x_mq_awq_batched", bytes);
        let result = self.launch_kernargs(
            "rotate_x_mq_awq",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr awp, ptr s1, ptr s2, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Fused FWHT rotation + FP8 pack for the decode FP8 path.
    /// Writes both F32 (into `x_rot`) and FP8 (into `mq_x_rot_fp8`
    /// sibling scratch) in one kernel launch. Returns the FP8 buffer's
    /// device pointer for the caller to feed directly to the FP8 GEMV.
    /// gfx12-only — uses cvt_pk_fp8_f32.
    pub(crate) fn rotate_x_mq_dual_fp8(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<*mut c_void> {
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "mq_rotate_x_dual_fp8_gfx12",
            kernels::MQ_ROTATE_X_DUAL_FP8_GFX12_SRC,
            "mq_rotate_x_dual_fp8_gfx12",
        )?;
        // Lazily allocate the FP8 sibling scratch sized to match k bytes.
        if self.mq_x_rot_fp8_bytes < k {
            self.mq_x_rot_fp8 = Some(self.hip.malloc(k)?);
            self.mq_x_rot_fp8_bytes = k;
        }
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xfp = self.mq_x_rot_fp8.as_ref().unwrap().as_ptr();
        let n_groups = (k / 256) as u32;
        let kv = k as i32;
        let bytes = crate::profile::mq_rotate_bytes(k) + k; // +1 byte/elem fp8 write
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_dual_fp8", bytes);
        let result = self.launch_kernargs(
            "mq_rotate_x_dual_fp8_gfx12",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr xrp, ptr xfp, ptr s1_ptr, ptr s2_ptr, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // Same x_rot dst as the standalone rotation path → invalidate
        // any ensure_*_x caches that were keyed by this pointer.
        self.invalidate_x_caches_for(xrp);
        result?;
        Ok(xfp)
    }
    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(&mut self, x: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "mq8_rotate_quantize_x",
            kernels::GEMV_MQ8G256_SRC,
            "mq8_rotate_quantize_x",
        )?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;

        let rot_func = &self.functions["mq8_rotate_quantize_x"];
        let mut xp = x.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut xs = xs_ptr;
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                rot_func,
                [n_groups, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// GPU-side RoPE (rotary positional embedding) applied in-place to Q and K.
    /// pos_buf: GPU buffer containing a single i32 position value.
    pub fn rope_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope", kernels::ROPE_SRC, "rope_f32")?;

        let q_ptr = q.buf.as_ptr();
        let k_ptr = k.buf.as_ptr();
        let pos_ptr = pos_buf.as_ptr();
        let nhq = n_heads_q as i32;
        let nhk = n_heads_k as i32;
        let hd = head_dim as i32;
        let fb = freq_base;

        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid = (half + block - 1) / block;

        self.launch_kernargs(
            "rope_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr q_ptr, ptr k_ptr, ptr pos_ptr, i32 nhq, i32 nhk, i32 hd, f32 fb],
        )
    }
    /// Batched RoPE: apply to [batch_size] positions in one launch.
    /// q: [batch_size × q_dim], k: [batch_size × kv_dim].
    /// positions: GPU buffer of [batch_size] i32 position indices.
    pub fn rope_batched_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_batched",
            kernels::ROPE_BATCHED_SRC,
            "rope_batched_f32",
        )?;
        let func = &self.functions["rope_batched_f32"];
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid_x = (half + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Partial interleaved RoPE for Qwen3.5 full attention layers.
    #[cfg(feature = "deltanet")]
    /// Single-token RoPE. `pos_buf` is a device buffer holding one i32 position
    /// value (graph-capture-safe: the pointer is stable, content updated before replay).
    pub fn rope_partial_interleaved_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &hip_bridge::DeviceBuffer,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        n_rot: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // RoPE convention for Qwen3.5 partial rotary: HF
        // `transformers/models/qwen3_5/modeling_qwen3_5.py:573-579` uses
        // `rotate_half` — pairs are (i, i + n_rot/2), NOT (2i, 2i+1).
        // hipfire-quantize does NOT permute Q/K weights at quantize time, so
        // the half-split kernel below is the mathematically-correct match for
        // HF-converted weights and is the DEFAULT since 2026-05-12. The legacy
        // interleaved kernel produced a ~0.4 nat engine-drift floor on Qwen3.5
        // models (docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention
        // probe / halfsplit fix") and is retained behind
        // HIPFIRE_ROPE_INTERLEAVED_LEGACY=1 for any caller that needs
        // bit-for-bit reproduction of pre-flip outputs (legacy regression
        // probes, comparisons to historical benches).
        //
        // Function name kept as `rope_partial_interleaved_f32` to avoid a
        // workspace-wide rename in this commit; the dispatched kernel is now
        // `rope_partial_halfsplit_f32` by default.
        let legacy = self.flags.rope_interleaved_legacy;
        let (src, entry) = if legacy {
            (
                kernels::ROPE_PARTIAL_INTERLEAVED_SRC,
                "rope_partial_interleaved_f32",
            )
        } else {
            (
                kernels::ROPE_PARTIAL_HALFSPLIT_SRC,
                "rope_partial_halfsplit_f32",
            )
        };
        let cache_key = if legacy {
            "rope_partial_interleaved"
        } else {
            "rope_partial_halfsplit"
        };
        self.ensure_kernel(cache_key, src, entry)?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.as_ptr();
        let nhq = n_heads_q as i32;
        let nhk = n_heads_k as i32;
        let hd = head_dim as i32;
        let nr = n_rot as i32;
        let fb = freq_base;
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid = [(n_pairs + block - 1) / block, 1, 1];
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rope", entry, bytes);
        let result = self.launch_kernargs(
            entry,
            grid,
            [block, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr pp, i32 nhq, i32 nhk, i32 hd, i32 nr, f32 fb],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched partial-interleaved RoPE. Each batch row reads its physical
    /// position from positions[b], adds pos_offset for the RoPE angle only,
    /// and rotates the first n_rot dims of every Q and K head. Q/K are
    /// [batch_size x n_heads x head_dim] row-major.
    /// Byte-exact with rope_partial_interleaved_f32 at batch_size=1.
    #[cfg(feature = "deltanet")]
    pub fn rope_partial_interleaved_f32_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        n_rot: usize,
        freq_base: f32,
        batch_size: usize,
        pos_offset: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Halfsplit is the default since 2026-05-12; HIPFIRE_ROPE_INTERLEAVED_LEGACY=1
        // restores the pre-flip interleaved kernel for legacy reproducibility.
        // Function name retained for source-tree stability; the dispatched
        // kernel is halfsplit by default. See sibling
        // `rope_partial_interleaved_f32` for the rationale.
        let legacy = self.flags.rope_interleaved_legacy;
        let (cache_key, src, entry) = if legacy {
            (
                "rope_partial_interleaved_batched",
                kernels::ROPE_PARTIAL_INTERLEAVED_BATCHED_SRC,
                "rope_partial_interleaved_batched_f32",
            )
        } else {
            (
                "rope_partial_halfsplit_batched",
                kernels::ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC,
                "rope_partial_halfsplit_batched_f32",
            )
        };
        self.ensure_kernel(cache_key, src, entry)?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let nhq = n_heads_q as i32;
        let nhk = n_heads_k as i32;
        let hd = head_dim as i32;
        let nr = n_rot as i32;
        let fb = freq_base;
        let bs = batch_size as i32;
        let po = pos_offset;
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid_x = (n_pairs + block - 1) / block;
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rope", entry, bytes);
        let result = self.launch_kernargs(
            entry,
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr pp, i32 nhq, i32 nhk, i32 hd, i32 nr, f32 fb, i32 bs, i32 po
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// 2-D spatial RoPE with precomputed per-patch cos/sin tables.
    ///
    /// Used by the dots.ocr (Qwen2-VL family) vision tower. Applies a
    /// halfsplit rotation in-place to Q and K — pairs `(d, d + head_dim/2)`
    /// of each head are rotated by `cos[patch, d] / sin[patch, d]` from
    /// the precomputed tables.
    ///
    /// # Arguments
    ///
    /// - `q`: `[n_patches, n_heads_q, head_dim]` row-major, f32.
    /// - `k`: `[n_patches, n_heads_k, head_dim]` row-major, f32. For
    ///   vision attention `n_heads_q == n_heads_k` (no GQA in
    ///   `DotsVisionTransformer`).
    /// - `cos_table` / `sin_table`: `[n_patches, head_dim]` f32 each.
    ///   Built by `hipfire_arch_dots_ocr::rope::build_rope_2d_tables`
    ///   on the host and uploaded once per image. The second half of
    ///   each row is a copy of the first half (the quarter-repeat
    ///   invariant from `apply_rotary_pos_emb_vision`), but the kernel
    ///   reads `cos[patch, e]` / `sin[patch, e]` independently so the
    ///   same kernel works for any "halfsplit + per-position tables"
    ///   case.
    /// - `head_dim`: must be even (halfsplit requires `head_dim/2`
    ///   pairs).
    ///
    /// # See also
    ///
    /// - `kernels/src/rope_2d_halfsplit.hip` — kernel source.
    /// - `crates/hipfire-arch-dots-ocr/src/rope.rs::build_rope_2d_tables`
    ///   — host-side cos/sin builder.
    /// - docs/plans/dots-ocr-prd.md §1.6 — algorithm spec.
    pub fn rope_2d_halfsplit_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        cos_table: &GpuTensor,
        sin_table: &GpuTensor,
        n_patches: usize,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // The dots.ocr 2-D RoPE layout (`[hc, wc, hc, wc]` quarter-
        // repeat) requires head_dim to split into four equal quarters;
        // `head_dim % 4 == 0` is the load-bearing constraint, not just
        // evenness. Match the `rope::build_rope_2d_tables` panic.
        assert!(
            head_dim % 4 == 0,
            "rope_2d_halfsplit_f32: head_dim={head_dim} must be a multiple of 4 \
             (the dots.ocr quarter-repeat layout splits head_dim into [hc, wc, hc, wc])",
        );
        assert!(
            n_patches > 0,
            "rope_2d_halfsplit_f32: n_patches must be > 0"
        );
        assert!(
            n_heads_q > 0 || n_heads_k > 0,
            "rope_2d_halfsplit_f32: must rotate at least one of Q/K"
        );
        self.ensure_kernel(
            "rope_2d_halfsplit",
            kernels::ROPE_2D_HALFSPLIT_SRC,
            "rope_2d_halfsplit_f32",
        )?;

        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let cp = cos_table.buf.as_ptr();
        let sp = sin_table.buf.as_ptr();
        let np = n_patches as i32;
        let nhq = n_heads_q as i32;
        let nhk = n_heads_k as i32;
        let hd = head_dim as i32;

        let half = (head_dim / 2) as u32;
        let max_heads = n_heads_q.max(n_heads_k) as u32;
        // Grid: (n_patches, max_heads, 1), block: (head_dim/2, 1, 1).
        // For dots.ocr's 19520 patches × 12 heads × 64 threads per
        // block this is ~234k blocks of 64 threads — large but fine
        // on RDNA.
        let grid = [n_patches as u32, max_heads, 1];
        let block = [half, 1, 1];
        // Bytes-touched estimate for the profile timer: Q+K reads/writes
        // + cos/sin reads. Each thread touches 2 q/k entries and 2
        // cos/sin entries (cd, ce, sd, se).
        let max_heads_us = n_heads_q.max(n_heads_k);
        let bytes = (n_patches * max_heads_us * head_dim * 4 * 2)  // Q+K RMW
                  + (n_patches * head_dim * 4 * 2); // cos+sin reads
        let timer =
            crate::profile::begin_timer(&self.hip, "rope_2d", "rope_2d_halfsplit_f32", bytes);
        let result = self.launch_kernargs(
            "rope_2d_halfsplit_f32",
            grid,
            block,
            0,
            &kernargs![ptr qp, ptr kp, ptr cp, ptr sp, i32 np, i32 nhq, i32 nhk, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// 2-D spatial RoPE applied IN-PLACE to the Q and K slices of a
    /// fused interleaved `[n_patches, 3 * hidden]` QKV buffer. V is
    /// left untouched. Companion to [`Self::rope_2d_halfsplit_f32`].
    ///
    /// The fused-QKV variant matches the natural output layout of a
    /// single QKV GEMM (one row per patch, `[Q-all-heads, K-all-heads,
    /// V-all-heads]` along the second axis) — same layout
    /// `vit_attention_opt` expects — so the encoder block becomes:
    ///
    /// ```text
    /// single QKV GEMM  →  rope_2d_halfsplit_qkv_interleaved_f32  →  vit_attention_opt
    /// ```
    ///
    /// without intermediate split/merge copies.
    ///
    /// `cos_table` and `sin_table` are the precomputed per-patch tables
    /// of shape `[n_patches, head_dim]` produced by
    /// `hipfire_arch_dots_ocr::rope::build_rope_2d_tables`.
    pub fn rope_2d_halfsplit_qkv_interleaved_f32(
        &mut self,
        qkv: &GpuTensor,
        cos_table: &GpuTensor,
        sin_table: &GpuTensor,
        n_patches: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 4 == 0,
            "rope_2d_halfsplit_qkv_interleaved_f32: head_dim={head_dim} must be a multiple of 4 \
             (the dots.ocr quarter-repeat layout splits head_dim into [hc, wc, hc, wc])",
        );
        assert!(
            n_patches > 0,
            "rope_2d_halfsplit_qkv_interleaved_f32: n_patches must be > 0"
        );
        assert!(
            n_heads > 0,
            "rope_2d_halfsplit_qkv_interleaved_f32: n_heads must be > 0"
        );
        self.ensure_kernel(
            "rope_2d_halfsplit_qkv_interleaved",
            kernels::ROPE_2D_HALFSPLIT_QKV_INTERLEAVED_SRC,
            "rope_2d_halfsplit_qkv_interleaved_f32",
        )?;

        let qkvp = qkv.buf.as_ptr();
        let cp = cos_table.buf.as_ptr();
        let sp = sin_table.buf.as_ptr();
        let np = n_patches as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;

        let half = (head_dim / 2) as u32;
        let grid = [n_patches as u32, n_heads as u32, 1];
        let block = [half, 1, 1];
        // Bytes-touched estimate: per thread we RMW two Q entries + two
        // K entries (= 4 × 2 × 4 = 32 bytes) plus 4 cos/sin reads (= 16
        // bytes). Threads per kernel = n_patches * n_heads * head_dim/2.
        let bytes = (n_patches * n_heads * head_dim * 4 * 4)             // Q+K RMW (read+write each)
                  + (n_patches * head_dim * 4 * 2); // cos+sin reads
        let timer = crate::profile::begin_timer(
            &self.hip,
            "rope_2d",
            "rope_2d_halfsplit_qkv_interleaved_f32",
            bytes,
        );
        let result = self.launch_kernargs(
            "rope_2d_halfsplit_qkv_interleaved_f32",
            grid,
            block,
            0,
            &kernargs![ptr qkvp, ptr cp, ptr sp, i32 np, i32 nh, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Training RoPE forward (fp32), HF half-split. `x`,`out`: `[rows*d]`,
    /// rows = seq*n_heads; `pos`: `[seq]`.
    pub fn rope_train_fwd(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        pos: &GpuTensor,
        rows: usize,
        n_heads: usize,
        d: usize,
        base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_train_fwd", kernels::ROPE_TRAIN_SRC, "rope_train_fwd")?;
        let func = &self.functions["rope_train_fwd"];
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut nh = n_heads as i32;
        let mut di = d as i32;
        let mut basef = base;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut di as *mut _ as *mut c_void,
            &mut basef as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Training RoPE backward (fp32): rotation by −angle. `d_out`,`dx`: `[rows*d]`.
    pub fn rope_train_bwd(
        &mut self,
        d_out: &GpuTensor,
        dx: &GpuTensor,
        pos: &GpuTensor,
        rows: usize,
        n_heads: usize,
        d: usize,
        base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_train_bwd", kernels::ROPE_TRAIN_SRC, "rope_train_bwd")?;
        let func = &self.functions["rope_train_bwd"];
        let mut dop = d_out.buf.as_ptr();
        let mut dxp = dx.buf.as_ptr();
        let mut pp = pos.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut nh = n_heads as i32;
        let mut di = d as i32;
        let mut basef = base;
        let mut params: Vec<*mut c_void> = vec![
            &mut dop as *mut _ as *mut c_void,
            &mut dxp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut di as *mut _ as *mut c_void,
            &mut basef as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4-faithful tail RoPE in INTERLEAVED pair convention (pairs are
    /// (2i, 2i+1) within the tail region). Upstream DeepSeek V4's RoPE goes
    /// through `torch.view_as_complex`, which is the interleaved form.
    /// Distinct from `rope_tail_halfsplit` which uses HF rotate_half.
    pub fn rope_tail_interleaved(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_interleaved",
            kernels::ROPE_TAIL_INTERLEAVED_SRC,
            "rope_tail_interleaved_f32",
        )?;
        let func = &self.functions["rope_tail_interleaved_f32"];
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [(half + 31) / 32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Tail-only RoPE — BATCHED. Per batch row b reads positions[b] and
    /// rotates the LAST n_rot dims of each head. At batch_size == 1 with
    /// positions[0] == pos_buf[0] this is byte-identical to
    /// `rope_tail_interleaved`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail_interleaved_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_interleaved_batched",
            kernels::ROPE_TAIL_INTERLEAVED_BATCHED_SRC,
            "rope_tail_interleaved_batched_f32",
        )?;
        let func = &self.functions["rope_tail_interleaved_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [(half + 31) / 32, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// YaRN-aware tail-only RoPE (DeepSeek V4 compressed layers). Mirrors
    /// antirez/ds4 `rope_tail_ext_inplace`. Caller supplies:
    ///   freq_base    — 10000 (dense) or 160000 (compressed)
    ///   freq_scale   — 1.0 (dense) or 1/16 = 0.0625 (compressed)
    ///   ext_factor   — 0.0 (dense, no YaRN) or 1.0 (compressed)
    ///   attn_factor  — 1.0 net (cancels with the inner log correction)
    ///   corr_low/high — output of yarn_corr_dims (computed on host)
    ///   inverse      — 0 for forward, 1 for inverse rotation
    ///
    /// For ext_factor=0 the math collapses to plain rope_tail_interleaved
    /// at freq=freq_scale*freq_base.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail_yarn_interleaved(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
        inverse: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_yarn_interleaved",
            kernels::ROPE_TAIL_YARN_INTERLEAVED_SRC,
            "rope_tail_yarn_interleaved_f32",
        )?;
        let func = &self.functions["rope_tail_yarn_interleaved_f32"];
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut fs = freq_scale;
        let mut ef = ext_factor;
        let mut af = attn_factor;
        let mut cl = corr_low;
        let mut ch = corr_high;
        let mut inv = inverse;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut fs as *mut _ as *mut c_void,
            &mut ef as *mut _ as *mut c_void,
            &mut af as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
            &mut inv as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [(half + 31) / 32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HIP-graphs-safe in-place YaRN tail RoPE at `base + slot_buf[0] *
    /// head_dim`. -1 sentinel → no-op. Single-tensor (n_heads_q=1,
    /// n_heads_k=0). Set freq_scale=1.0, ext_factor=0.0 to recover
    /// plain rope_tail_interleaved.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn rope_tail_yarn_interleaved_at_slot_buf(
        &mut self,
        base: &GpuTensor,
        pos_buf: &GpuTensor,
        slot_buf: &GpuTensor,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_yarn_interleaved_at_slot_buf",
            kernels::ROPE_TAIL_YARN_INTERLEAVED_AT_SLOT_BUF_SRC,
            "rope_tail_yarn_interleaved_at_slot_buf_f32",
        )?;
        let bp = base.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let hd = head_dim;
        let nr = n_rot;
        let fb = freq_base;
        let fs = freq_scale;
        let ef = ext_factor;
        let af = attn_factor;
        let cl = corr_low;
        let ch = corr_high;
        let half = (n_rot / 2) as u32;
        let block = 32u32;
        let grid = (half + block - 1) / block;
        self.launch_kernargs(
            "rope_tail_yarn_interleaved_at_slot_buf_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![
                ptr bp, ptr pp, ptr sb, i32 hd, i32 nr, f32 fb, f32 fs, f32 ef, f32 af, f32 cl,
                f32 ch
            ],
        )
    }
    /// YaRN-aware tail RoPE — BATCHED. Per-batch positions array; same
    /// YaRN blend semantics as `rope_tail_yarn_interleaved`. Byte-identical
    /// to the sequential variant at batch_size == 1.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail_yarn_interleaved_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
        inverse: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_yarn_interleaved_batched",
            kernels::ROPE_TAIL_YARN_INTERLEAVED_BATCHED_SRC,
            "rope_tail_yarn_interleaved_batched_f32",
        )?;
        let func = &self.functions["rope_tail_yarn_interleaved_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut fs = freq_scale;
        let mut ef = ext_factor;
        let mut af = attn_factor;
        let mut cl = corr_low;
        let mut ch = corr_high;
        let mut inv = inverse;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut fs as *mut _ as *mut c_void,
            &mut ef as *mut _ as *mut c_void,
            &mut af as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
            &mut inv as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [(half + 31) / 32, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
