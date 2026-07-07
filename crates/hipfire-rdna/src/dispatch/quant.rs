// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reference-layer activation quant kernels (W4A4 oq4, W4A8 oq8, per-token
//! int8). Split out of `dispatch/mod.rs` (Phase 1 of
//! docs/plans/2026-06-23-dispatch-refactor.md) — a pure behavior-preserving
//! move: the methods stay on `Gpu` via this child `impl` block and reach
//! `Gpu`'s module-private fields as a descendant of `dispatch`.

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Opus Quant W4A4: dynamic per-token/group INT4 activation quantizer.
    /// `x_f32` [B,K] → `xq_i4` [B,K/2] (packed signed int4) + `xs` [B,K/group]
    /// (f32 scales). gfx1103 wave32, zero LDS. `group % 32 == 0`, `k % group == 0`.
    pub fn quantize_act_oq4(
        &mut self,
        x_f32: &GpuTensor,
        xq_i4: &GpuTensor,
        xs: &GpuTensor,
        batch_size: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            group % 32,
            0,
            "quantize_act_oq4: group must be a multiple of 32"
        );
        assert_eq!(
            k % group,
            0,
            "quantize_act_oq4: K must be a multiple of group"
        );
        self.ensure_kernel(
            "quantize_act_oq4",
            kernels::QUANTIZE_ACT_OQ4_SRC,
            "quantize_act_oq4",
        )?;
        let xp = x_f32.buf.as_ptr();
        let xqp = xq_i4.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_g = (k / group) as u32;
        let grid_b = batch_size as u32;
        let func = &self.functions["quantize_act_oq4"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_g, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Opus Quant W8A8 grouped int8×int8 GEMM (the int8 generalization of
    /// [`Self::gemm_oq4_grouped_wmma`]). `w_i8`/`x_i8` are [M,K]/[B,K] signed int8
    /// rows; `w_scales`/`x_scales` are per-group f32; `y_f32` is [B,M].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq8_grouped_wmma(
        &mut self,
        w_i8: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_oq8_grouped_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "gemm_oq8_grouped_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "gemm_oq8_grouped_wmma",
            kernels::GEMM_OQ8_GROUPED_WMMA_SRC,
            "gemm_oq8_grouped_wmma",
        )?;
        let wp = w_i8.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = m.div_ceil(16) as u32;
        let grid_b = batch_size.div_ceil(16) as u32;
        let func = &self.functions["gemm_oq8_grouped_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Register-tiled Opus-Quant W8A8 GEMM — MB×NB 16×16 output subtiles per
    /// wave for ILP, the tiled counterpart of `gemm_oq8_grouped_wmma`. Same
    /// operands (W int8 [M,K], Ws [M,K/group], X int8 [B,K], Xs [B,K/group],
    /// Y f32 [B,M]). `(mb, nb)` ∈ {(2,2),(4,2),(2,4)} selects the entrypoint.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq8_tiled_wmma(
        &mut self,
        w_i8: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        mb: usize,
        nb: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % group, 0, "gemm_oq8_tiled_wmma: K must be a multiple of group");
        assert_eq!(group % 16, 0, "gemm_oq8_tiled_wmma: group must be a multiple of 16");
        let kname = match (mb, nb) {
            (2, 2) => "gemm_oq8_tiled_wmma_2x2",
            (4, 2) => "gemm_oq8_tiled_wmma_4x2",
            (2, 4) => "gemm_oq8_tiled_wmma_2x4",
            _ => panic!("gemm_oq8_tiled_wmma: unsupported tiling {mb}x{nb}"),
        };
        self.ensure_kernel(kname, kernels::GEMM_OQ8_TILED_WMMA_SRC, kname)?;
        let wp = w_i8.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = m.div_ceil(16 * mb) as u32;
        let grid_b = batch_size.div_ceil(16 * nb) as u32;
        let func = &self.functions[kname];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Register-tiled Opus-Quant W4A8 GEMM: packed signed-int4 weight (`W`
    /// [M,K/2], scales `Ws` [M,K/group]) unpacked to int8 × dynamic-int8
    /// activation (`X` [B,K] int8, `Xs` [B,K/group]) on iu8 WMMA → `Y` f32 [B,M].
    /// `(mb, nb)` ∈ {(2,2),(4,2),(2,4)}. Half the weight footprint of oq8.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq4a8_tiled_wmma(
        &mut self,
        w_i4: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        mb: usize,
        nb: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % group, 0, "gemm_oq4a8_tiled_wmma: K must be a multiple of group");
        assert_eq!(group % 16, 0, "gemm_oq4a8_tiled_wmma: group must be a multiple of 16");
        let kname = match (mb, nb) {
            (2, 2) => "gemm_oq4a8_tiled_wmma_2x2",
            (4, 2) => "gemm_oq4a8_tiled_wmma_4x2",
            (2, 4) => "gemm_oq4a8_tiled_wmma_2x4",
            _ => panic!("gemm_oq4a8_tiled_wmma: unsupported tiling {mb}x{nb}"),
        };
        self.ensure_kernel(kname, kernels::GEMM_OQ4A8_TILED_WMMA_SRC, kname)?;
        let wp = w_i4.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = m.div_ceil(16 * mb) as u32;
        let grid_b = batch_size.div_ceil(16 * nb) as u32;
        let func = &self.functions[kname];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Opus Quant W8A8 dynamic int8 activation quantizer (f32 → signed int8 +
    /// per-group f32 scales). `xq_i8` is [B,K] int8; `xs` is [B,K/group] f32.
    pub fn quantize_act_oq8(
        &mut self,
        x_f32: &GpuTensor,
        xq_i8: &GpuTensor,
        xs: &GpuTensor,
        batch_size: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            group % 32,
            0,
            "quantize_act_oq8: group must be a multiple of 32"
        );
        assert_eq!(
            k % group,
            0,
            "quantize_act_oq8: K must be a multiple of group"
        );
        self.ensure_kernel(
            "quantize_act_oq8",
            kernels::QUANTIZE_ACT_OQ8_SRC,
            "quantize_act_oq8",
        )?;
        let xp = x_f32.buf.as_ptr();
        let xqp = xq_i8.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_g = (k / group) as u32;
        let grid_b = batch_size as u32;
        let func = &self.functions["quantize_act_oq8"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_g, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched W4A4 (Opus oq4) GEMM for prefill: int4-quantize the FWHT-rotated
    /// activation `x_rot` [N×K] ONCE into the shared batched scratch, then a
    /// grouped WMMA GEMM into `y` [N×M]. `w_combined` is the loader's
    /// `[nibbles M*K/2 | f32 scales M*ng]` Raw buffer; the scale view is derived
    /// via `sub_offset`. group is fixed at 256 (oq4 codec). This is the batched
    /// counterpart of the decode `gemv_oq4_grouped` quantize+GEMV.
    pub fn gemm_oq4_grouped_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 256;
        self.ensure_oq4_scratch_batched(n, k, m)?;
        let ng = k / GROUP;
        let xq = GpuTensor {
            buf: unsafe { self.oq4_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * (k / 2)],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq4_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        self.quantize_act_oq4(x_rot, &xq, &xs, n, k, GROUP)?;
        let ws = w_combined.sub_offset(m * (k / 2), m * ng * 4);
        self.gemm_oq4_grouped_wmma(w_combined, &ws, &xq, &xs, y, m, k, n, GROUP)
    }

    /// Batched W4A4 oq4 GEMM with residual add: `residual[N×M] += W·x_rot`.
    /// GEMMs into the persistent batched f32 scratch (`oq4_ytmp_batch`, sized for
    /// M*N here) then a single elementwise add — there is no fused oq4 residual
    /// kernel, mirroring the decode `GemvResidual` Oq4 arm.
    pub fn gemm_oq4_grouped_residual_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.ensure_oq4_scratch_batched(n, k, m)?;
        let tmp = GpuTensor {
            buf: unsafe { self.oq4_ytmp_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * m],
            dtype: DType::F32,
        };
        self.gemm_oq4_grouped_act_batched(w_combined, x_rot, &tmp, m, k, n)?;
        let res_n = residual.sub_offset(0, n * m);
        self.add_inplace_f32(&res_n, &tmp)
    }

    /// Generic kernel library: WMMA GEMM, signed INT8 inputs → INT32 output.
    /// `a_i8` [M,K], `x_i8` [B,K] (int8), `y_i32` [B,M] (int32).
    /// gfx1103/RDNA3 wave32, zero LDS. Requires `k % 16 == 0` and wave32 WMMA.
    pub fn gemm_iu8_i32_wmma(
        &mut self,
        a_i8: &GpuTensor,
        x_i8: &GpuTensor,
        y_i32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_iu8_i32_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_iu8_i32_wmma",
            kernels::GEMM_IU8_I32_WMMA_SRC,
            "gemm_iu8_i32_wmma",
        )?;
        let ap = a_i8.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let yp = y_i32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_iu8_i32_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Reference kernel layer: per-token symmetric int8 activation quant.
    /// `x_f32` [B,K] → `xq_i8` [B,K] int8 + `xs_f32` [B] per-row scale. Zero LDS.
    pub fn quantize_act_int8_per_token(
        &mut self,
        x_f32: &GpuTensor,
        xq_i8: &GpuTensor,
        xs_f32: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "quantize_act_int8_per_token",
            kernels::QUANTIZE_ACT_INT8_PER_TOKEN_SRC,
            "quantize_act_int8_per_token",
        )?;
        let xp = x_f32.buf.as_ptr();
        let qp = xq_i8.buf.as_ptr();
        let sp = xs_f32.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let func = &self.functions["quantize_act_int8_per_token"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Reference kernel layer: int32 → f32 dequant by per-row × per-col scales.
    /// `y_i32` [B,M] · `x_scale` [B] · `w_scale` [M] → `y_f32` [B,M].
    pub fn dequant_i32_rowcol(
        &mut self,
        y_i32: &GpuTensor,
        x_scale: &GpuTensor,
        w_scale: &GpuTensor,
        y_f32: &GpuTensor,
        batch_size: usize,
        m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "dequant_i32_rowcol",
            kernels::DEQUANT_I32_ROWCOL_SRC,
            "dequant_i32_rowcol",
        )?;
        let yp = y_i32.buf.as_ptr();
        let xsp = x_scale.buf.as_ptr();
        let wsp = w_scale.buf.as_ptr();
        let op = y_f32.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        let n = batch_size * m;
        let grid = ((n + 255) / 256) as u32;
        let func = &self.functions["dequant_i32_rowcol"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
