// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! S4-f16-residual-inputs: post-attention/down producers that emit the frozen
//! FP16 sidecars consumed by [`Gpu::gemm_mq4g256v2_residual_wmma_f16`].
//!
//! Three Gpu launch families (plain + AWQ each), all exact-gfx1100,
//! batched `[N x K]` row-major, `launch_maybe_blob` + `KernargBlob` only:
//!
//! * `gated_norm_rotate_mq_f16_batched` — LA post-GDN: gated RMSNorm + FWHT
//!   + F16 store. Replaces `gated_norm_f32_batched` + `rotate_x_mq_batched`
//!   + the GEMM `convert_f32_to_f16` prologue.
//! * `sigmoid_mul_rotate_mq_f16_batched` — FA post-attention:
//!   `sigmoid(gate)*attn` + FWHT + F16 store. Replaces `sigmoid_mul_f32` +
//!   `rotate_x_mq_batched` + convert. Does NOT mutate the attn input (the old
//!   in-place sigmoid write is skipped; nothing downstream reads it).
//! * `fused_silu_mul_rotate_mq_f16_batched` — FFN down: `silu(gate)*up` +
//!   FWHT + F16 store. Replaces `fused_silu_mul_mq_rotate_mq_batched` +
//!   convert 1:1.
//!
//! Bit-exactness: each F16 word must equal the old F32 pipeline's store
//! reloaded and cast by `convert_f32_to_f16` (`out[i] = (_Float16)in[i]`).
//! The F32 store/load round trip is exact, so the kernels compute the
//! identical F32 value in-register (same expression order as the sources)
//! and cast with the same cast. Any mismatch is a hard veto — see the
//! `test_mq_f16_residual_producers_gfx1100` example.
//!
//! Kernel sources are self-contained via `include_str!` (no shared-registry
//! edits). The `gemm_mq4g256v2_residual_wmma_f16` entry launches the SAME
//! kernel symbols as `gemm_mq4g256v2_residual_wmma` (same modules, same
//! grids) with the sidecar pointer wired directly as X, bypassing
//! `ensure_fp16_x`. Tier selection (ldsstage opt-in, split-K table, base
//! fallback, `residual_ksplit_off`) mirrors that function exactly; the hook
//! falls back to the old path wherever this entry returns Err.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

const GATED_NORM_F16_SRC: &str =
    include_str!("../../../kernels/src/gated_norm_mq_rotate_f16.gfx1100.hip");
const SIGMOID_MUL_F16_SRC: &str =
    include_str!("../../../kernels/src/sigmoid_mul_mq_rotate_f16.gfx1100.hip");
const FUSED_SILU_F16_SRC: &str =
    include_str!("../../../kernels/src/fused_silu_mul_mq_rotate_f16.gfx1100.hip");

/// Split-K width for the exact-gfx1100 DFlash verify tier (N<=16).
///
/// Copy of the frozen perf table owned by `gemm.rs::residual_ksplit_kw`
/// (the authority — keep in sync; do not retune here).
fn residual_ksplit_kw(k: usize) -> Option<usize> {
    if k % 256 != 0 || k == 0 {
        return None;
    }
    let g = k / 256;
    let want = if k <= 8192 { 4 } else { 8 };
    [want, 4, 2]
        .into_iter()
        .filter(|&kw| kw <= want)
        .find(|&kw| g >= kw && g % kw == 0)
}

fn check_f16_out(out: &GpuTensor, what: &str) -> HipResult<()> {
    if out.dtype != DType::F16 {
        return Err(hip_bridge::HipError::new(
            1,
            &format!("{what}: F16 sidecar required (got {:?})", out.dtype),
        ));
    }
    Ok(())
}

fn check_f32_in(x: &GpuTensor, what: &str) -> HipResult<()> {
    if x.dtype != DType::F32 {
        return Err(hip_bridge::HipError::new(
            1,
            &format!("{what}: F32 input required (got {:?})", x.dtype),
        ));
    }
    Ok(())
}

impl Gpu {
    /// LA post-GDN producer: gated RMSNorm + FWHT + direct F16 store.
    ///
    /// `x`, `z`: `[N x K]` F32 (`K = n_heads*head_dim`); `weight`:
    /// `[head_dim]` F32 norm weight; `out`: `[N x K]` F16 sidecar.
    /// Requires `head_dim == 128`, `K % 256 == 0`, exact gfx1100.
    /// After: `out == convert(old gated_norm+rotate F32)` byte-for-byte.
    pub fn gated_norm_rotate_mq_f16_batched(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(x, "gated_norm_rotate_mq_f16_batched")?;
        check_f32_in(z, "gated_norm_rotate_mq_f16_batched")?;
        check_f16_out(out, "gated_norm_rotate_mq_f16_batched")?;
        if head_dim != 128 {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_f16_batched: head_dim == 128 required",
            ));
        }
        let k = n_heads * head_dim;
        if k == 0 || k % 256 != 0 || batch_size == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_f16_batched: K % 256 == 0 and N >= 1 required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "gated_norm_mq_rotate_f16";
        const FUNC: &str = "gated_norm_mq_rotate_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, GATED_NORM_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        // Read x+z (+weight/signs), write half-size out.
        let bytes = crate::profile::gated_norm_bytes(k) * batch_size
            + crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// AWQ-aware sibling: `(gated_norm/scale)` before the FWHT. `awq_scale`:
    /// 1D F32 `[K]` in the unrotated basis. Dispatched only when the
    /// consuming wo carries an awq_scale.
    pub fn gated_norm_rotate_mq_awq_f16_batched(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(x, "gated_norm_rotate_mq_awq_f16_batched")?;
        check_f32_in(z, "gated_norm_rotate_mq_awq_f16_batched")?;
        check_f16_out(out, "gated_norm_rotate_mq_awq_f16_batched")?;
        if head_dim != 128 {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_awq_f16_batched: head_dim == 128 required",
            ));
        }
        let k = n_heads * head_dim;
        if k == 0 || k % 256 != 0 || batch_size == 0 || awq_scale.numel() < k {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_awq_f16_batched: K % 256 == 0, N >= 1, awq len >= K required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_awq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "gated_norm_mq_rotate_f16";
        const FUNC: &str = "gated_norm_mq_rotate_awq_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, GATED_NORM_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut ap = awq_scale.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(k) * batch_size
            + crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(ap);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// FA post-attention producer: `sigmoid(gate)*attn` + FWHT + direct F16
    /// store. `attn`, `gate`: `[N x K]` F32; `out`: `[N x K]` F16 sidecar.
    /// Requires `K % 256 == 0`, exact gfx1100. Does not mutate `attn`.
    pub fn sigmoid_mul_rotate_mq_f16_batched(
        &mut self,
        attn: &GpuTensor,
        gate: &GpuTensor,
        out: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(attn, "sigmoid_mul_rotate_mq_f16_batched")?;
        check_f32_in(gate, "sigmoid_mul_rotate_mq_f16_batched")?;
        check_f16_out(out, "sigmoid_mul_rotate_mq_f16_batched")?;
        if k == 0 || k % 256 != 0 || batch_size == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "sigmoid_mul_rotate_mq_f16_batched: K % 256 == 0 and N >= 1 required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "sigmoid_mul_rotate_mq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "sigmoid_mul_mq_rotate_f16";
        const FUNC: &str = "sigmoid_mul_mq_rotate_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, SIGMOID_MUL_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut ap = attn.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 2 + k * 2 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(gp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// AWQ-aware sibling of [`Gpu::sigmoid_mul_rotate_mq_f16_batched`].
    pub fn sigmoid_mul_rotate_mq_awq_f16_batched(
        &mut self,
        attn: &GpuTensor,
        gate: &GpuTensor,
        awq_scale: &GpuTensor,
        out: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(attn, "sigmoid_mul_rotate_mq_awq_f16_batched")?;
        check_f32_in(gate, "sigmoid_mul_rotate_mq_awq_f16_batched")?;
        check_f16_out(out, "sigmoid_mul_rotate_mq_awq_f16_batched")?;
        if k == 0 || k % 256 != 0 || batch_size == 0 || awq_scale.numel() < k {
            return Err(hip_bridge::HipError::new(
                1,
                "sigmoid_mul_rotate_mq_awq_f16_batched: K % 256 == 0, N >= 1, awq len >= K required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "sigmoid_mul_rotate_mq_awq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "sigmoid_mul_mq_rotate_f16";
        const FUNC: &str = "sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, SIGMOID_MUL_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut ap = attn.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + k * 2 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(gp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// FFN down producer: `silu(gate)*up` + FWHT + direct F16 store.
    /// `gate`, `up`: `[N x K]` F32; `out`: `[N x K]` F16 sidecar.
    /// Requires `K % 256 == 0`, exact gfx1100.
    pub fn fused_silu_mul_rotate_mq_f16_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(gate, "fused_silu_mul_rotate_mq_f16_batched")?;
        check_f32_in(up, "fused_silu_mul_rotate_mq_f16_batched")?;
        check_f16_out(out, "fused_silu_mul_rotate_mq_f16_batched")?;
        if k == 0 || k % 256 != 0 || batch_size == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "fused_silu_mul_rotate_mq_f16_batched: K % 256 == 0 and N >= 1 required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "fused_silu_mul_rotate_mq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "fused_silu_mul_mq_rotate_f16";
        const FUNC: &str = "fused_silu_mul_mq_rotate_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, FUSED_SILU_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 2 + k * 2 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// AWQ-aware sibling of [`Gpu::fused_silu_mul_rotate_mq_f16_batched`].
    pub fn fused_silu_mul_rotate_mq_awq_f16_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq_scale: &GpuTensor,
        out: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f32_in(gate, "fused_silu_mul_rotate_mq_awq_f16_batched")?;
        check_f32_in(up, "fused_silu_mul_rotate_mq_awq_f16_batched")?;
        check_f16_out(out, "fused_silu_mul_rotate_mq_awq_f16_batched")?;
        if k == 0 || k % 256 != 0 || batch_size == 0 || awq_scale.numel() < k {
            return Err(hip_bridge::HipError::new(
                1,
                "fused_silu_mul_rotate_mq_awq_f16_batched: K % 256 == 0, N >= 1, awq len >= K required",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "fused_silu_mul_rotate_mq_awq_f16_batched: exact gfx1100 required",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        const MODULE: &str = "fused_silu_mul_mq_rotate_f16";
        const FUNC: &str = "fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100";
        self.ensure_kernel(MODULE, FUSED_SILU_F16_SRC, FUNC)?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut op = out.buf.as_ptr();
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + k * 2 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(k / 256) as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(op);
        result
    }

    /// MQ4V2 residual GEMM consuming a pre-converted FP16 X directly.
    ///
    /// Same kernel symbols, modules, grids, and `Y += W@X` semantics as
    /// `gemm_mq4g256v2_residual_wmma`; the only difference is `x_f16`
    /// (DType::F16, e.g. an S4 sidecar) is wired straight in, bypassing
    /// `ensure_fp16_x` and its `convert_f32_to_f16` launch. Tier selection
    /// mirrors that function: ldsstage opt-in, split-K table, base
    /// fallback (`residual_ksplit_off` forces base). Any shape outside the
    /// routed verify domain (non-gfx1100, `batch_size > 16`, `K % 256 != 0`)
    /// returns Err so the caller keeps the old path.
    pub fn gemm_mq4g256v2_residual_wmma_f16(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        check_f16_out(x_f16, "gemm_mq4g256v2_residual_wmma_f16")?;
        if m == 0 || batch_size == 0 {
            return Ok(());
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                "gemm_mq4g256v2_residual_wmma_f16: exact gfx1100 required",
            ));
        }
        if batch_size > 16 {
            return Err(hip_bridge::HipError::new(
                1,
                "gemm_mq4g256v2_residual_wmma_f16: batch_size <= 16 (verify tier) required",
            ));
        }
        if k % 256 != 0 || k == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                "gemm_mq4g256v2_residual_wmma_f16: K must be a nonzero multiple of 256",
            ));
        }
        self.bind_thread()?;
        // ldsstage opt-in mirror (same predicate as the F32 entry).
        if self.flags.residual_ldsstage && k % 512 == 0 {
            return self.gemm_residual_f16_one(
                a_raw,
                x_f16,
                y,
                m,
                k,
                batch_size,
                "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage",
                kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX1100_LDSSTAGE_SRC,
                "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage",
                [256, 1, 1],
            );
        }
        // Split-K verify tier mirror (skipped under residual_ksplit_off,
        // exactly like the F32 entry).
        if !self.flags.residual_ksplit_off {
            if let Some(kw) = residual_ksplit_kw(k) {
                let func_name = match kw {
                    2 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds",
                    4 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds",
                    8 => "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds",
                    _ => {
                        return Err(hip_bridge::HipError::new(
                            1,
                            "gemm_mq4g256v2_residual_wmma_f16: bad split-K width",
                        ));
                    }
                };
                return self.gemm_residual_f16_one(
                    a_raw,
                    x_f16,
                    y,
                    m,
                    k,
                    batch_size,
                    "gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds",
                    kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX1100_KSPLIT_LDS_SRC,
                    func_name,
                    [(32 * kw) as u32, 1, 1],
                );
            }
        }
        // Base kernel mirror.
        self.gemm_residual_f16_one(
            a_raw,
            x_f16,
            y,
            m,
            k,
            batch_size,
            "gemm_mq4g256v2_residual_wmma",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_SRC,
            "gemm_mq4g256v2_residual_wmma",
            [32, 1, 1],
        )
    }

    /// Single-shot F16-X residual launch against one kernel symbol.
    /// ABI (arg order, grid math, byte accounting) mirrors the F32 entries.
    #[allow(clippy::too_many_arguments)]
    fn gemm_residual_f16_one(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        module: &'static str,
        src: &'static str,
        func_name: &'static str,
        block: [u32; 3],
    ) -> HipResult<()> {
        self.ensure_kernel(module, src, func_name)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
