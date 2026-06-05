// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
use crate::dispatch::{DType, Gpu, GpuTensor, FP8_GEMV_MIN_M};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.
    pub fn gemv_q4lut(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4lut", kernels::GEMV_Q4LUT_SRC, "gemv_q4lut")?;
        let func = &self.functions["gemv_q4lut"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // LDS: 8 codebooks × 16 entries × 2 bytes = 256 bytes
        let shared_mem = 256u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.
    pub fn gemv_q4wave(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4wave", kernels::GEMV_Q4WAVE_SRC, "gemv_q4wave")?;
        let func = &self.functions["gemv_q4wave"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.
    pub fn gemv_q4as8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4as8", kernels::GEMV_Q4AS8_SRC, "gemv_q4as8")?;
        let func = &self.functions["gemv_q4as8"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])
    pub fn gemv_f32(&mut self, a: &GpuTensor, x: &GpuTensor, y: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv", kernels::GEMV_SRC, "gemv_f32")?;
        let func = &self.functions["gemv_f32"];

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m;
        let mut k_val = k;
        let mut alpha_val = alpha;
        let mut beta_val = beta;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut alpha_val as *mut _ as *mut c_void,
            &mut beta_val as *mut _ as *mut c_void,
        ];

        // One block per row, 256 threads per block with shared memory reduction
        let block_size = 256u32.min(k as u32);
        let shared_mem = block_size * 4; // one float per thread
        let bytes = (m as usize) * (k as usize) * 4 + (k as usize) * 4 + (m as usize) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)
    pub fn gemv_q4k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q4k", kernels::GEMV_Q4K_SRC, "gemv_q4k")?;
        let func = &self.functions["gemv_q4k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory needed
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.
    pub fn gemv_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g128", kernels::GEMV_HFQ4G128_SRC, "gemv_hfq4g128")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq4g128_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g128", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g128",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// ParoQuant Givens rotation: apply learned pairwise rotations + channel
    /// scaling to activation vector x in-place. Called before GEMV on
    /// ParoQ4G128 weights.
    ///
    /// x: [seq_len, hidden_dim] F16 (modified in place)
    /// pairs: [krot, hidden_dim] I16
    /// theta: [krot, hidden_dim/2] F16
    /// channel_scales: [hidden_dim] F16
    pub fn givens_rotate(
        &mut self,
        x: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "givens_rotate_f32",
            kernels::GIVENS_ROTATE_SRC,
            "givens_rotate_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let x_ptr = x.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &pairs_ptr as *const _ as *mut c_void,
            &theta_ptr as *const _ as *mut c_void,
            &cs_ptr as *const _ as *mut c_void,
            &seq_val as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &krot_val as *const _ as *mut c_void,
        ];

        let smem = (cta_m * group_size * 4) as u32; // CTA_M * GROUP_SIZE * sizeof(float)

        // Bytes: read+write activation (2 × seq × dim × 4) + read pairs/theta/scales
        // (krot × dim × 2 for pairs+theta packed, dim × 2 for scales).
        let bytes = seq_len * hidden_dim * 4 * 2 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(&self.hip, "rotate", "givens_rotate_f32", bytes);
        let result = self.launch_maybe_blob(
            "givens_rotate_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(pairs_ptr);
                b.push_ptr(theta_ptr);
                b.push_ptr(cs_ptr);
                b.push_i32(seq_val);
                b.push_i32(dim_val);
                b.push_i32(krot_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Out-of-place Givens rotation. Reads `x_in`, writes rotated
    /// activations to `x_out`. Replaces the
    /// `copy_d2d + givens_rotate` pair used by `rotate_x_paro_for` —
    /// one graph node + one inter-node dependency removed.
    #[allow(clippy::too_many_arguments)]
    pub fn givens_rotate_to(
        &mut self,
        x_in: &GpuTensor,
        x_out: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "givens_rotate_to_f32",
            kernels::GIVENS_ROTATE_TO_SRC,
            "givens_rotate_to_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let in_ptr = x_in.buf.as_ptr();
        let out_ptr = x_out.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let mut params: Vec<*mut c_void> = vec![
            &in_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &pairs_ptr as *const _ as *mut c_void,
            &theta_ptr as *const _ as *mut c_void,
            &cs_ptr as *const _ as *mut c_void,
            &seq_val as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &krot_val as *const _ as *mut c_void,
        ];

        let smem = (cta_m * group_size * 4) as u32;

        // Bytes: read x_in (seq × dim × 4) + write x_out (seq × dim × 4)
        // + read pairs/theta/scales (krot × dim × 2 + dim × 2).
        let bytes = seq_len * hidden_dim * 4 * 2 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(&self.hip, "rotate", "givens_rotate_to_f32", bytes);
        let result = self.launch_maybe_blob(
            "givens_rotate_to_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(in_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pairs_ptr);
                b.push_ptr(theta_ptr);
                b.push_ptr(cs_ptr);
                b.push_i32(seq_val);
                b.push_i32(dim_val);
                b.push_i32(krot_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused silu(gate)*up + per-channel scale + krot rounds of Givens
    /// rotation. Single-launch replacement for the
    /// `silu_mul_f32 + givens_rotate` pair used by the ParoQuant routed
    /// gate→down hop. Same shared-memory + grid contract as
    /// `givens_rotate`, plus two additional input pointers (gate, up)
    /// and a separate output pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_silu_mul_givens_rotate_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_silu_mul_givens_rotate_f32",
            kernels::FUSED_SILU_MUL_GIVENS_ROTATE_SRC,
            "fused_silu_mul_givens_rotate_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let mut params: Vec<*mut c_void> = vec![
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &pairs_ptr as *const _ as *mut c_void,
            &theta_ptr as *const _ as *mut c_void,
            &cs_ptr as *const _ as *mut c_void,
            &seq_val as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &krot_val as *const _ as *mut c_void,
        ];

        let smem = (cta_m * group_size * 4) as u32;

        // Bytes: read gate (seq × dim × 4) + read up (seq × dim × 4) + write out
        // (seq × dim × 4) + read pairs/theta/scales (krot × dim × 2 + dim × 2).
        let bytes = seq_len * hidden_dim * 4 * 3 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_givens_rotate_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_silu_mul_givens_rotate_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gate_ptr);
                b.push_ptr(up_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pairs_ptr);
                b.push_ptr(theta_ptr);
                b.push_ptr(cs_ptr);
                b.push_i32(seq_val);
                b.push_i32(dim_val);
                b.push_i32(krot_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).
    pub fn ensure_paro_scratch(&mut self, dim: usize) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.scratch
            .ensure_paro_scratch(&self.hip, self.device_id, dim)
    }

    /// Ensure 4 rotation scratch buffers for Paro fused-kernel dispatch.
    /// Each buffer is sized [k] F32. Lazily allocated; grows on demand (never shrinks).
    pub fn ensure_paro_fused_scratch(&mut self, k: usize) -> HipResult<()> {
        self.scratch
            .ensure_paro_fused_scratch(&self.hip, self.device_id, k)
    }

    /// PARO4-G128T fused gate/up decode path. Gate and up have distinct
    /// Paro rotations, so this still rotates both, but batches the two
    /// rotations and the two pack4 GEMVs into two launches instead of four.
    pub fn fused_gate_up_paro4g128t(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        x_rot_gate: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T fused gate/up requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T fused gate/up requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot_gate.buf.size() / 4 >= k,
            "PARO4G128T fused gate/up gate scratch too small: {} floats for K={k}",
            x_rot_gate.buf.size() / 4
        );
        self.ensure_mq_signs()?;
        let x_rot_up = GpuTensor {
            buf: unsafe { self.scratch.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![self.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        assert!(
            x_rot_up.buf.size() / 4 >= k,
            "PARO4G128T fused gate/up up scratch too small: {} floats for K={k}",
            x_rot_up.buf.size() / 4
        );

        let rotate_kernel = "paro4g128t_dual_rotate";
        let gemv_kernel = "fused_gate_up_paro4g128t_pack4";
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, rotate_kernel)?;
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, gemv_kernel)?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrg = x_rot_gate.buf.as_ptr();
        let xru = x_rot_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let mut rotate_params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xrg as *const _ as *mut c_void,
            &xru as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let rotate_bytes = crate::profile::paro4g128t_rotate_bytes(m, k) * 2;
        let rotate_timer =
            crate::profile::begin_timer(&self.hip, "format", rotate_kernel, rotate_bytes);
        let rotate_result = self.launch_maybe_blob(
            rotate_kernel,
            [groups, 2, 1],
            [32, 1, 1],
            0,
            &mut rotate_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(xrg);
                b.push_ptr(xru);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = rotate_timer {
            t.finish(&self.hip);
        }
        rotate_result?;
        self.invalidate_x_caches_for(xrg);
        self.invalidate_x_caches_for(xru);

        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let mut gemv_params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xrg as *const _ as *mut c_void,
            &xru as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let gemv_bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * 4;
        let gemv_timer = crate::profile::begin_timer(&self.hip, "gemv", gemv_kernel, gemv_bytes);
        let gemv_result = self.launch_maybe_blob(
            gemv_kernel,
            [(m / 4) as u32, 2, 1],
            [32, 1, 1],
            0,
            &mut gemv_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xrg);
                b.push_ptr(xru);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = gemv_timer {
            t.finish(&self.hip);
        }
        gemv_result
    }

    /// PARO4-G128T fused LA projection path. The four Paro projections have
    /// distinct rotations, so this batches four rotates and four pack4 GEMVs
    /// into two launches.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_paro4g128t(
        &mut self,
        a0: &GpuTensor,
        a1: &GpuTensor,
        a2: &GpuTensor,
        a3: &GpuTensor,
        x: &GpuTensor,
        y0: &GpuTensor,
        y1: &GpuTensor,
        y2: &GpuTensor,
        y3: &GpuTensor,
        x_rot0: &GpuTensor,
        x_rot1: &GpuTensor,
        x_rot2: &GpuTensor,
        x_rot3: &GpuTensor,
        m0: usize,
        m1: usize,
        m2: usize,
        m3: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for (label, m) in [("m0", m0), ("m1", m1), ("m2", m2), ("m3", m3)] {
            assert_eq!(
                m % 8,
                0,
                "PARO4G128T fused LA {label} requires M multiple of 8, got {m}"
            );
        }
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T fused LA requires K multiple of 128, got {k}"
        );
        for (label, scratch) in [
            ("x_rot0", x_rot0),
            ("x_rot1", x_rot1),
            ("x_rot2", x_rot2),
            ("x_rot3", x_rot3),
        ] {
            assert!(
                scratch.buf.size() / 4 >= k,
                "PARO4G128T fused LA {label} scratch too small: {} floats for K={k}",
                scratch.buf.size() / 4
            );
        }
        let rotate_kernel = "paro4g128t_quad_rotate";
        let gemv_kernel = "fused_qkvza_paro4g128t_pack4";
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, rotate_kernel)?;
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, gemv_kernel)?;

        let a0p = a0.buf.as_ptr();
        let a1p = a1.buf.as_ptr();
        let a2p = a2.buf.as_ptr();
        let a3p = a3.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xr0p = x_rot0.buf.as_ptr();
        let xr1p = x_rot1.buf.as_ptr();
        let xr2p = x_rot2.buf.as_ptr();
        let xr3p = x_rot3.buf.as_ptr();
        let y0p = y0.buf.as_ptr();
        let y1p = y1.buf.as_ptr();
        let y2p = y2.buf.as_ptr();
        let y3p = y3.buf.as_ptr();
        let m0v = m0 as i32;
        let m1v = m1 as i32;
        let m2v = m2 as i32;
        let m3v = m3 as i32;
        let kv = k as i32;

        let mut rotate_params: Vec<*mut c_void> = vec![
            &a0p as *const _ as *mut c_void,
            &a1p as *const _ as *mut c_void,
            &a2p as *const _ as *mut c_void,
            &a3p as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xr0p as *const _ as *mut c_void,
            &xr1p as *const _ as *mut c_void,
            &xr2p as *const _ as *mut c_void,
            &xr3p as *const _ as *mut c_void,
            &m0v as *const _ as *mut c_void,
            &m1v as *const _ as *mut c_void,
            &m2v as *const _ as *mut c_void,
            &m3v as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let groups = (k / 128) as u32;
        let rotate_bytes = crate::profile::paro4g128t_rotate_bytes(m0, k)
            + crate::profile::paro4g128t_rotate_bytes(m1, k)
            + crate::profile::paro4g128t_rotate_bytes(m2, k)
            + crate::profile::paro4g128t_rotate_bytes(m3, k);
        let rotate_timer =
            crate::profile::begin_timer(&self.hip, "format", rotate_kernel, rotate_bytes);
        let rotate_result = self.launch_maybe_blob(
            rotate_kernel,
            [groups, 4, 1],
            [32, 1, 1],
            0,
            &mut rotate_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a0p);
                b.push_ptr(a1p);
                b.push_ptr(a2p);
                b.push_ptr(a3p);
                b.push_ptr(xp);
                b.push_ptr(xr0p);
                b.push_ptr(xr1p);
                b.push_ptr(xr2p);
                b.push_ptr(xr3p);
                b.push_i32(m0v);
                b.push_i32(m1v);
                b.push_i32(m2v);
                b.push_i32(m3v);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = rotate_timer {
            t.finish(&self.hip);
        }
        rotate_result?;
        for ptr in [xr0p, xr1p, xr2p, xr3p] {
            self.invalidate_x_caches_for(ptr);
        }

        let mut gemv_params: Vec<*mut c_void> = vec![
            &a0p as *const _ as *mut c_void,
            &a1p as *const _ as *mut c_void,
            &a2p as *const _ as *mut c_void,
            &a3p as *const _ as *mut c_void,
            &xr0p as *const _ as *mut c_void,
            &xr1p as *const _ as *mut c_void,
            &xr2p as *const _ as *mut c_void,
            &xr3p as *const _ as *mut c_void,
            &y0p as *const _ as *mut c_void,
            &y1p as *const _ as *mut c_void,
            &y2p as *const _ as *mut c_void,
            &y3p as *const _ as *mut c_void,
            &m0v as *const _ as *mut c_void,
            &m1v as *const _ as *mut c_void,
            &m2v as *const _ as *mut c_void,
            &m3v as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let max_m = m0.max(m1).max(m2).max(m3);
        let gemv_bytes = (crate::profile::gemv_paro4g128_prerotated_bytes(m0, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m1, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m2, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m3, k))
            * 2;
        let gemv_timer = crate::profile::begin_timer(&self.hip, "gemv", gemv_kernel, gemv_bytes);
        let gemv_result = self.launch_maybe_blob(
            gemv_kernel,
            [(max_m / 4) as u32, 4, 1],
            [32, 1, 1],
            0,
            &mut gemv_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a0p);
                b.push_ptr(a1p);
                b.push_ptr(a2p);
                b.push_ptr(a3p);
                b.push_ptr(xr0p);
                b.push_ptr(xr1p);
                b.push_ptr(xr2p);
                b.push_ptr(xr3p);
                b.push_ptr(y0p);
                b.push_ptr(y1p);
                b.push_ptr(y2p);
                b.push_ptr(y3p);
                b.push_i32(m0v);
                b.push_i32(m1v);
                b.push_i32(m2v);
                b.push_i32(m3v);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = gemv_timer {
            t.finish(&self.hip);
        }
        gemv_result
    }

    /// HFQ2-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq2g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g256", kernels::GEMV_HFQ2G256_SRC, "gemv_hfq2g256")?;
        let func = &self.functions["gemv_hfq2g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// MQ2-Lloyd GEMV (2-bit + per-block 4-entry fp16 codebook). K must be a
    /// multiple of 256. Same launch shape as gemv_hfq2g256 — header is the
    /// only layout difference.
    pub fn gemv_mq2g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd",
            kernels::GEMV_MQ2G256_LLOYD_SRC,
            "gemv_mq2g256_lloyd",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_mq2g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// MQ2-Lloyd GEMV with engine-side x rotation (matches `gemv_mq2g256_with_rotate`).
    pub fn gemv_mq2g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq2g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq2g256_lloyd(a_raw, x_rot, y, m, k)
    }

    /// MQ3-Lloyd GEMV (3-bit + per-block 8-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102 use the K4-unrolled + LDS-codebook
    /// variant; other archs fall back to the baseline switch-dispatch path.
    pub fn gemv_mq3g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) =
            kernels::gemv_mq3g256_lloyd_for_arch(&self.arch_caps, self.flags.lloyd_force_baseline);
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd, both of which bind.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd(a_raw, x_rot, y, m, k)
    }

    /// MQ4-Lloyd GEMV (4-bit + per-block 16-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102/1151 use the K4-unrolled + LDS-codebook
    /// variant (cooperative double-load for the 64-entry table). Other archs
    /// fall back to the chip-agnostic baseline switch-dispatch path.
    pub fn gemv_mq4g256_lloyd(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) =
            kernels::gemv_mq4g256_lloyd_for_arch(&self.arch_caps, self.flags.lloyd_force_baseline);
        self.ensure_kernel(module, src, "gemv_mq4g256_lloyd")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq4g256_lloyd",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4-Lloyd GEMV with engine-side x rotation.
    pub fn gemv_mq4g256_lloyd_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq4g256_lloyd.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq4g256_lloyd(a_raw, x_rot, y, m, k)
    }

    /// DIAGNOSTIC ONLY: K4 multi-accumulator MQ4-Lloyd GEMV. NOT for production.
    /// Used by examples/diag_mq4_lloyd_multiacc.rs to compare against the slow
    /// generic kernel on real model rows. See the kernel header for the
    /// open question this exists to investigate.
    pub fn gemv_mq4g256_lloyd_multiacc_diag(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq4g256_lloyd_multiacc_diag",
            kernels::GEMV_MQ4G256_LLOYD_MULTIACC_DIAG_GFX1100_SRC,
            "gemv_mq4g256_lloyd_multiacc_diag",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_mq4g256_lloyd_multiacc_diag",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// MQ4-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Mirrors
    /// gemv_mq3g256_lloyd_residual; same single-acc bug fix applies.
    pub fn gemv_mq4g256_lloyd_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq4g256_lloyd_residual_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemv_mq4g256_lloyd_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq4g256_lloyd_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq4g256_lloyd_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4-Lloyd residual GEMV with engine-side x rotation.
    pub fn gemv_mq4g256_lloyd_residual_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq4g256_lloyd_residual(a_raw, x_rot, y, m, k)
    }

    /// Fused Gate+Up MQ4-Lloyd: two GEMVs in one launch. Mirrors
    /// fused_gate_up_mq3g256_lloyd. Caller is responsible for pre-rotating x.
    pub fn fused_gate_up_mq4g256_lloyd(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_gate_up_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_gate_up_mq4g256_lloyd")?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let total = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(gate_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(up_m, k)
            - k * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_gate_up_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_gate_up_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm);
                b.push_i32(um);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused QKVZA MQ4-Lloyd: 4 LA-preamble GEMVs in one launch.
    pub fn fused_qkvza_mq4g256_lloyd(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkvza_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkvza_mq4g256_lloyd")?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(qkv_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(z_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(beta_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(alpha_m, k)
            - 3 * (k * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkvza_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_i);
                b.push_i32(z_m_i);
                b.push_i32(b_m_i);
                b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused QKV MQ4-Lloyd: 3 FA-preamble GEMVs in one launch.
    pub fn fused_qkv_mq4g256_lloyd(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq4g256_lloyd")?;
        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (q_m + k_m + v_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_i);
                b.push_i32(k_m_i);
                b.push_i32(v_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Used by
    /// `weight_gemv_residual` MQ3-Lloyd arm to eliminate the alloc + gemv +
    /// add_inplace_f32 + free fallback chain (saves ~4.4% of decode time on
    /// 9B Lloyd-MQ3, gfx1100, per the 2026-05-06 decode profile).
    pub fn gemv_mq3g256_lloyd_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_mq3g256_lloyd_residual_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemv_mq3g256_lloyd_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_mq3g256_lloyd_residual", bytes);
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd residual GEMV with engine-side x rotation.
    pub fn gemv_mq3g256_lloyd_residual_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to rotate_x_mq + gemv_mq3g256_lloyd_residual.
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_mq3g256_lloyd_residual(a_raw, x_rot, y, m, k)
    }

    /// Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch. Mirrors
    /// `fused_gate_up_hfq4g256` for the Lloyd-MQ3 dtype. Caller is
    /// responsible for pre-rotating x (FWHT) before invoking; the kernel
    /// itself only does the GEMV. Both `a_gate` and `a_up` must be MQ3-Lloyd
    /// matrices with the same K and codebook layout.
    pub fn fused_gate_up_mq3g256_lloyd(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_gate_up_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_gate_up_mq3g256_lloyd")?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let total = (gate_m + up_m) as u32;
        // Bandwidth: A_gate + A_up read, x read once, y_gate + y_up written.
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(gate_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(up_m, k)
            - k * 4; // x is shared, don't double-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_gate_up_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_gate_up_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm);
                b.push_i32(um);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused QKVZA MQ3-Lloyd: 4 LA-preamble GEMVs in one launch. Used by
    /// qwen35.rs DeltaNet decode when wqkv + wz + w_beta + w_alpha are
    /// all MQ3G256Lloyd. Mirrors `fused_qkvza_hfq4g256` — same routing
    /// (grid = qkv_m + z_m + beta_m + alpha_m, block picks A by gid),
    /// Lloyd K4+LDS body on gfx1100. Caller is responsible for
    /// pre-rotating x (FWHT); the kernel only does the GEMVs.
    pub fn fused_qkvza_mq3g256_lloyd(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkvza_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkvza_mq3g256_lloyd")?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        // Bandwidth: 4 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(qkv_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(z_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(beta_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(alpha_m, k)
            - 3 * (k * 4); // x is shared, don't quadruple-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkvza_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_i);
                b.push_i32(z_m_i);
                b.push_i32(b_m_i);
                b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused QKV MQ3-Lloyd: 3 FA-preamble GEMVs in one launch. Used by
    /// qwen35.rs FullAttention decode when wq + wk + wv are all
    /// MQ3G256Lloyd. Sibling of `fused_qkvza_mq3g256_lloyd` for the
    /// 3-projection FA case (vs LA's 4-projection QKVZA). Caller is
    /// responsible for pre-rotating x; the kernel only does the GEMVs.
    pub fn fused_qkv_mq3g256_lloyd(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq3g256_lloyd")?;
        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let total = (q_m + k_m + v_m) as u32;
        // Bandwidth: 3 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4); // x is shared, don't triple-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq3g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_i);
                b.push_i32(k_m_i);
                b.push_i32(v_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(&mut self) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.scratch
            .ensure_mq_signs(&self.hip, &mut self.pool, self.device_id)
    }

    /// Lazily initialize MagnumQuant FWHT sign tables for G128 (128 floats each, seeds 43 and 1043).
    /// Also allocates the shared `mq_x_rot` scratch if not already present — the G256 path
    /// (`ensure_mq_signs`) normally owns that allocation, but the G128 path must be
    /// self-sufficient so models that carry only MQ4G128 weights still get the scratch buffer.
    pub fn ensure_mq_signs_128(&mut self) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.scratch
            .ensure_mq_signs_128(&self.hip, &mut self.pool, self.device_id)
    }

    /// MagnumQuant GEMV: FWHT-rotated HFQ4-G256. Rotates x per group via ds_swizzle,
    /// then standard 4-bit dot product. signs1/signs2 are the FWHT sign tables (256 floats each).
    pub fn gemv_mq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "gemv_mq4g256")?;
        let func = &self.functions["gemv_mq4g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut s1_ptr = signs1.buf.as_ptr();
        let mut s2_ptr = signs2.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut s1_ptr as *mut _ as *mut c_void,
            &mut s2_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS for rotated x: 256 floats = 1024 bytes
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                1024,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFP4-G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
    ///
    /// v1 correctness anchor: no WMMA, no FP8, no rotation. K must be a multiple of 256
    /// (the kernel's 4-accumulator + tail-by-g%4 outer loop assumes the 256-element
    /// "iter window" stride; v2 will lift this to k%32==0). See `kernels/src/gemv_hfp4g32.hip`
    /// and `docs/quant-formats/hfp4.md`.
    pub fn gemv_hfp4g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32 requires K%256==0 in v1, got K={}",
            k
        );
        // Shape-gated: FP8 dot4 only when M is large enough that it
        // actually wins (FFN shapes). At M < 4096 the fallback wins or
        // ties; uniform-FP8 was net-negative in 9B Qwen 3.5 decode.
        if self.arch_caps.has_wmma_w32_gfx12() && self.flags.fp8_wmma && m >= FP8_GEMV_MIN_M {
            return self.gemv_hfp4g32_fp8_gfx12(a_raw, x, y, m, k);
        }
        // gfx11 (RDNA3) v_dot2_f32_f16 trickle-down: replaces the
        // fallback's F32 mul+fma chain with one fdot2 per 2 elements.
        // No new scratch (reuses ensure_fp16_x), no cross-kernel
        // context cost like the FP8 path had. Default-on for gfx11.
        // Kill switch HIPFIRE_DOT2_GEMV=0 for A/B benching.
        if self.arch_caps.has_wmma_w32() && self.flags.dot2_gemv {
            return self.gemv_hfp4g32_dot2_gfx11(a_raw, x, y, m, k);
        }
        self.gemv_hfp4g32_fallback(a_raw, x, y, m, k)
    }

    /// Direct fallback entry point (F32 mul+fma chain). Useful for
    /// A/B benchmarking against the dot2/fp8 variants.
    pub fn gemv_hfp4g32_fallback(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfp4g32_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfp4g32")?;
        let func = &self.functions["gemv_hfp4g32"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS: 16-entry FP16 LUT = 32 bytes.
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// gfx12 FP8-dot4 decode-path GEMV for HFP4G32. Uses
    /// `dot4_f32_fp8_fp8` to cut inner-loop ALU vs the dequant/FMA
    /// fallback. Activation X is consumed as FP8 (E4M3); when called
    /// via `gemv_hfp4g32` (env-gated routing for HFP4G32 weights, no
    /// rotation), this function calls `ensure_fp8_x` to pack F32 → FP8
    /// scratch. The MFP4G32 rotation path uses
    /// `rotate_x_mq_dual_fp8` + `gemv_hfp4g32_fp8_gfx12_with_fp8_ptr`
    /// instead so the FP8 pack is fused into the rotation kernel.
    pub fn gemv_hfp4g32_fp8_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32_fp8 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_fp8_gfx12",
            kernels::GEMV_HFP4G32_FP8_GFX12_SRC,
            "gemv_hfp4g32_fp8_gfx12",
        )?;
        let x_fp8_ptr = self.ensure_fp8_x(x, k)?;
        self.gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(a_raw, x_fp8_ptr, y, m, k)
    }

    /// Fused RMSNorm + MagnumQuant FWHT rotation. Replaces the
    /// `rmsnorm_f32` + `rotate_x_mq` sequence with a single kernel launch.
    /// Reads unnormalized `x` + rmsnorm `weight`, computes rmsnorm in LDS,
    /// applies the same per-256-element FWHT as `mq_rotate_x`, and writes
    /// the rotated normalized vector into `x_rot`.
    ///
    /// Preconditions:
    /// - `k` is a multiple of 256 (enforced by callers via `config.dim`)
    /// - `k` ≤ 16384 (LDS ceiling; 16K floats = 64KB minus reduce buffer)
    pub fn fused_rmsnorm_rotate_mq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x split: opt-in via HIPFIRE_GFX942_RMSNORM_SPLIT=1.
        // Two-kernel path (reduce + rotate) gives 5× more in-flight wave64s
        // on prefill scale; modest decode change. Math byte-identical.
        if self.flags.gfx942_rmsnorm_split {
            return self.fused_rmsnorm_rotate_mq_split_gfx942(x, weight, x_rot, k, eps, 1);
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        let block_size = 256u32;
        // Dynamic LDS: K floats for x_shared + 256 floats for reduce buffer.
        let shared_mem = ((k + 256) * 4) as u32;

        // Bandwidth: read x (K*4) + weight (K*4) + signs (2*256*4) + write x_rot (K*4)
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate", bytes);
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Phase A Stage A — AWQ-aware variant of fused_rmsnorm_rotate_mq.
    ///
    /// After computing the RMSNorm output, divides element-wise by
    /// `awq_scale[i]` BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (x/s) = W·x` where W·s is baked at quantize time.
    ///
    /// Use when the upcoming linear layer's WeightTensor carries
    /// `awq_scale = Some(...)`; otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).
    ///
    /// Backward-compatible: kernel is separate, no behavioral change for
    /// the standard fused_rmsnorm_rotate_mq path.
    pub fn fused_rmsnorm_rotate_mq_awq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            "fused_rmsnorm_mq_rotate_awq",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &awp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        // Bandwidth: read x + weight + awq_scale + signs + write x_rot.
        let bytes = k * 4 * 4 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_awq", bytes);
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_awq",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Batched `fused_rmsnorm_rotate_mq`. Grid.x is the batch dim — processes
    /// N tokens' [N × K] x into [N × K] x_rot in a single launch. Byte-exact
    /// against calling `fused_rmsnorm_rotate_mq` N times on separate x/x_rot
    /// buffers. Weight/signs are shared across the batch.
    /// Phase A Stage A — batched AWQ variant. Mirrors
    /// fused_rmsnorm_rotate_mq_batched but takes an additional
    /// `awq_scale: &GpuTensor` (length K, FP32) and dispatches the
    /// AWQ kernel. Caller selects based on the upcoming linear
    /// layer's WeightTensor.awq_scale being Some.
    pub fn fused_rmsnorm_rotate_mq_awq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            "fused_rmsnorm_mq_rotate_awq",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate_awq_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_awq",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// gfx942 two-kernel split: rmsnorm_reduce + rotate_with_rms.
    ///
    /// Replaces the single-WG-per-batch fused kernel with two kernels that
    /// each scale better on MI300X's 304 CUs. Kernel A computes rms per
    /// batch (1 WG/batch × 16 wave64s). Kernel B applies rmsnorm + FWHT
    /// per (group, batch) cell (K/256 × batch WGs × 1 wave64 each).
    ///
    /// For batch=256 K=5120: 20×256 = 5120 wave64s on Kernel B vs 1024 on
    /// the fused path → 5× more in-flight waves on prefill.
    ///
    /// Math byte-identical to fused_rmsnorm_mq_rotate.
    fn fused_rmsnorm_rotate_mq_split_gfx942(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "rmsnorm_reduce_gfx942",
            kernels::RMSNORM_REDUCE_GFX942_SRC,
            "rmsnorm_reduce_gfx942",
        )?;
        self.ensure_kernel(
            "rotate_with_rms_gfx942",
            kernels::ROTATE_WITH_RMS_GFX942_SRC,
            "rotate_with_rms_gfx942",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        // Allocate scratch tensor for rms_out (batch_size f32s).
        let rms_tensor = self.alloc_tensor(&[batch_size], DType::F32)?;
        let rms_ptr = rms_tensor.buf.as_ptr();

        // ─── Kernel A: rmsnorm_reduce ────────────────────────────────────
        let xp_a = x.buf.as_ptr();
        let kv_a = k as i32;
        let eps_a = eps;
        let mut params_a: Vec<*mut c_void> = vec![
            &xp_a as *const _ as *mut c_void,
            &rms_ptr as *const _ as *mut c_void,
            &kv_a as *const _ as *mut c_void,
            &eps_a as *const _ as *mut c_void,
        ];
        let bytes_a = batch_size * k * 4;
        let timer_a =
            crate::profile::begin_timer(&self.hip, "fused", "rmsnorm_reduce_gfx942", bytes_a);
        self.launch_maybe_blob(
            "rmsnorm_reduce_gfx942",
            [batch_size as u32, 1, 1],
            [1024, 1, 1],
            0,
            &mut params_a,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp_a);
                b.push_ptr(rms_ptr);
                b.push_i32(kv_a);
                b.push_f32(eps_a);
                b
            },
        )?;
        if let Some(t) = timer_a {
            t.finish(&self.hip);
        }

        // ─── Kernel B: rotate_with_rms ───────────────────────────────────
        let xp_b = x.buf.as_ptr();
        let wp_b = weight.buf.as_ptr();
        let xrp_b = x_rot.buf.as_ptr();
        let s1_b = s1_ptr;
        let s2_b = s2_ptr;
        let kv_b = k as i32;
        let mut params_b: Vec<*mut c_void> = vec![
            &xp_b as *const _ as *mut c_void,
            &wp_b as *const _ as *mut c_void,
            &s1_b as *const _ as *mut c_void,
            &s2_b as *const _ as *mut c_void,
            &rms_ptr as *const _ as *mut c_void,
            &xrp_b as *const _ as *mut c_void,
            &kv_b as *const _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        let bytes_b = batch_size * (k * 4 * 3 + 2 * 256 * 4);
        let timer_b =
            crate::profile::begin_timer(&self.hip, "fused", "rotate_with_rms_gfx942", bytes_b);
        let result = self.launch_maybe_blob(
            "rotate_with_rms_gfx942",
            [groups, batch_size as u32, 1],
            [64, 1, 1],
            0,
            &mut params_b,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp_b);
                b.push_ptr(wp_b);
                b.push_ptr(s1_b);
                b.push_ptr(s2_b);
                b.push_ptr(rms_ptr);
                b.push_ptr(xrp_b);
                b.push_i32(kv_b);
                b
            },
        );
        if let Some(t) = timer_b {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp_b);
        result
    }

    pub fn fused_rmsnorm_rotate_mq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x split — see fused_rmsnorm_rotate_mq docstring.
        if self.flags.gfx942_rmsnorm_split {
            return self.fused_rmsnorm_rotate_mq_split_gfx942(x, weight, x_rot, k, eps, batch_size);
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Fused SwiGLU + FWHT rotation. Reads gate/up, computes
    /// silu(gate[k])*up[k] on the fly, applies FWHT rotation, writes x_rot.
    /// Used as the w_down input stage for MQ4 — replaces the pair
    /// silu_mul_f32 + mq_rotate_x with one launch.
    pub fn fused_silu_mul_rotate_mq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        // Bandwidth: read gate + up, 2x256 signs, write x_rot.
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate", bytes);
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Batched `fused_silu_mul_rotate_mq`. Grid.y is the batch dim — processes
    /// N tokens' [N × K] gate/up/x_rot in a single launch.
    pub fn fused_silu_mul_rotate_mq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_mq_rotate_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Phase A Stage A — F2 AWQ-aware variant of `fused_silu_mul_rotate_mq`.
    ///
    /// After computing silu(gate)*up, divides element-wise by `awq_scale[i]`
    /// BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (silu(g)*u / s) = W·silu(g)*u` where W·s is baked at
    /// quantize time for the down_proj / w_down weights.
    ///
    /// Use when the down_proj `WeightTensor` carries `awq_scale = Some(...)`;
    /// otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).
    pub fn fused_silu_mul_rotate_mq_awq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            "fused_silu_mul_mq_rotate_awq",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &awp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        // Bandwidth: read gate + up + awq_scale, 2x256 signs, write x_rot.
        let bytes = k * 4 * 4 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate_awq", bytes);
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate_awq",
            [n_groups, 1, 1],
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
                b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Phase A Stage A — F2 batched AWQ variant of `fused_silu_mul_rotate_mq`.
    /// Grid.y is the batch dim — processes [N × K] gate/up/x_rot.
    pub fn fused_silu_mul_rotate_mq_awq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            "fused_silu_mul_mq_rotate_awq",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_mq_rotate_awq_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate_awq",
            [n_groups, batch_size as u32, 1],
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
                b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Invalidate any `ensure_*_x` caches whose source pointer matches
    /// `dst_ptr`. Must be called by any kernel that overwrites data at
    /// `dst_ptr` since the caches key on raw pointer equality and have
    /// no way to detect data changes otherwise.
    fn invalidate_x_caches_for(&mut self, dst_ptr: *mut c_void) {
        self.scratch.invalidate_x_caches_for(dst_ptr)
    }

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    pub fn rotate_x_mq(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        self.scratch.rotate_x_mq(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.pool,
            self.device_id,
            x,
            x_rot,
            k,
        )
    }

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        self.scratch.rotate_x_mq_batched(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.pool,
            self.device_id,
            x,
            x_rot,
            k,
            batch_size,
        )
    }

    /// FWHT-128 standalone rotation for MQ4G128 activations.
    ///
    /// Mirrors `rotate_x_mq` but targets G128 groups (32 threads × 4 elems).
    /// Grid: [k/128, 1, 1]. Block: [32, 1, 1].
    pub fn rotate_x_mq_128(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel("gemv_mq4g128", kernels::GEMV_MQ4G128_SRC, "mq_rotate_x_128")?;
        self.scratch.rotate_x_mq_128(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.pool,
            self.device_id,
            x,
            x_rot,
            k,
        )
    }

    /// Phase A Stage A — F2 AWQ-aware variant of `rotate_x_mq`.
    ///
    /// Divides each input element by `awq_scale[i]` BEFORE the FWHT.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K.
    pub fn rotate_x_mq_awq(
        &mut self,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            "rotate_x_mq_awq",
        )?;
        self.scratch.rotate_x_mq_awq(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.pool,
            self.device_id,
            x,
            awq_scale,
            x_rot,
            k,
        )
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
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            "rotate_x_mq_awq",
        )?;
        self.scratch.rotate_x_mq_awq_batched(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.pool,
            self.device_id,
            x,
            awq_scale,
            x_rot,
            k,
            batch_size,
        )
    }

    /// MagnumQuant MQ4: rotate x once, then GEMV against rotated x.
    /// MQ4 weights are stored in HFQ4-G256 format with FWHT pre-applied, so the GEMV
    /// inner loop is identical to standard HFQ4 — we reuse the arch-tuned HFQ4 kernel.
    pub fn gemv_mq4g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        // MQ4 = FWHT-rotated HFQ4-G256. dot(rot(W), rot(x)) = dot(W, x).
        // Route through the arch-specific HFQ4 kernel (4x unroll on gfx1100, etc).
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ4 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mq4g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ4-G128 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq_128` into `x_rot` first.
    pub fn gemv_mq4g128_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq4g128(a_raw, x_rot, y, m, k)
    }

    /// MFP4G32: rotate x once via FWHT, then HFP4G32 GEMV against rotated x.
    /// MFP4 weights are stored in HFP4G32 format (E2M1 + UE8M0 g32 + FP16 row scale)
    /// with the same 256-element FWHT pre-applied, so the GEMV inner loop is
    /// identical to standard HFP4 — we reuse `gemv_hfp4g32`.
    pub fn gemv_mfp4g32_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Shape-gated FP8 routing (Option α empirical embodiment): only
        // when M ≥ FP8_GEMV_MIN_M does FP8 dot4 win measurably on this
        // path. Below threshold (e.g. wo M=2048), the FP8 fused-rotation
        // costs more than the dot4 ALU savings — keep the F32 fallback.
        if self.arch_caps.has_wmma_w32_gfx12() && self.flags.fp8_wmma && m >= FP8_GEMV_MIN_M {
            let x_fp8_ptr = self.rotate_x_mq_dual_fp8(x, x_rot, k)?;
            return self.gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(a_raw, x_fp8_ptr, y, m, k);
        }
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfp4g32(a_raw, x_rot, y, m, k)
    }

    /// MFP4G32 with pre-rotated x. Skips the rotation step entirely — caller must
    /// have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mfp4g32_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfp4g32(a_raw, x_rot, y, m, k)
    }

    /// Fused FWHT rotation + FP8 pack for the decode FP8 path.
    /// Writes both F32 (into `x_rot`) and FP8 (into `mq_x_rot_fp8`
    /// sibling scratch) in one kernel launch. Returns the FP8 buffer's
    /// device pointer for the caller to feed directly to the FP8 GEMV.
    /// gfx12-only — uses cvt_pk_fp8_f32.
    fn rotate_x_mq_dual_fp8(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<*mut c_void> {
        self.ensure_kernel(
            "mq_rotate_x_dual_fp8_gfx12",
            kernels::MQ_ROTATE_X_DUAL_FP8_GFX12_SRC,
            "mq_rotate_x_dual_fp8_gfx12",
        )?;
        self.scratch.rotate_x_mq_dual_fp8(
            &self.hip,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            self.graphs.capture_mode,
            self.flags.force_blob_path,
            &mut self.compiler,
            &mut self.modules,
            &mut self.pool,
            self.device_id,
            x,
            x_rot,
            k,
        )
    }

    /// gfx11 (RDNA3) v_dot2_f32_f16 decode-path GEMV for HFP4G32.
    /// Takes F32 x and converts to FP16 INLINE in the inner loop;
    /// `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) does 2 FP16 muls +
    /// 1 FP32 add per VALU. Reduces inner-loop multiply count ~4×
    /// vs the fallback F32 mul+fma chain on ALU-bound shapes.
    /// Routed automatically from `gemv_hfp4g32` when on gfx11+ archs
    /// (gfx1100/1101/1102/1150/1151). NO ensure_fp16_x pre-pass —
    /// that's the v1 trap (eats the dot2 savings in production).
    pub fn gemv_hfp4g32_dot2_gfx11(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32_dot2 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_dot2_gfx11",
            kernels::GEMV_HFP4G32_DOT2_GFX11_SRC,
            "gemv_hfp4g32_dot2_gfx11",
        )?;
        let func = &self.functions["gemv_hfp4g32_dot2_gfx11"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FP8-dot4 GEMV variant that takes an FP8 device pointer directly
    /// (bypassing `ensure_fp8_x`). Used by `gemv_mfp4g32_with_rotate`
    /// after the fused rotation+pack kernel produces the FP8 buffer
    /// in-place.
    fn gemv_hfp4g32_fp8_gfx12_with_fp8_ptr(
        &mut self,
        a_raw: &GpuTensor,
        x_fp8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            k % 256 == 0,
            "gemv_hfp4g32_fp8 requires K%256==0, got K={}",
            k
        );
        self.ensure_kernel(
            "gemv_hfp4g32_fp8_gfx12",
            kernels::GEMV_HFP4G32_FP8_GFX12_SRC,
            "gemv_hfp4g32_fp8_gfx12",
        )?;
        let func = &self.functions["gemv_hfp4g32_fp8_gfx12"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_fp8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                32,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// MagnumQuant MQ3: rotate x once, then HFQ3-G256 GEMV against rotated x.
    /// MQ3 weights are stored in HFQ3-G256 format (104 B/group) with FWHT pre-applied,
    /// so the GEMV inner loop is identical to standard HFQ3.
    pub fn gemv_mq3g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ3 with pre-rotated x.
    pub fn gemv_mq3g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ2: rotate x once, then HFQ2-G256 GEMV against rotated x.
    /// MQ2 weights are stored in HFQ2-G256 format (72 B/group) with FWHT pre-applied.
    pub fn gemv_mq2g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ2 with pre-rotated x.
    pub fn gemv_mq2g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq2g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ6: rotate x via FWHT, then HFQ6 GEMV.
    pub fn gemv_mq6g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_x_mq(x, x_rot, k)?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ6 with pre-rotated x.
    pub fn gemv_mq6g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq6g256(a_raw, x_rot, y, m, k)
    }

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(&mut self, x: &GpuTensor, k: usize) -> HipResult<()> {
        // bind_thread: skip — delegated to scratch.rs
        self.ensure_kernel(
            "mq8_rotate_quantize_x",
            kernels::GEMV_MQ8G256_SRC,
            "mq8_rotate_quantize_x",
        )?;
        self.scratch.rotate_quantize_x_mq8(
            &self.hip,
            &self.functions,
            self.active_stream.as_ref(),
            &mut self.pool,
            self.device_id,
            x,
            k,
        )
    }

    /// MQ8 dp4a GEMV using pre-rotated+quantized x. Caller must have called
    /// `rotate_quantize_x_mq8(x, k)` first — results use the internal `mq_x_q8`/`mq_x_scales`.
    pub fn gemv_mq8g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_mq8g256", kernels::GEMV_MQ8G256_SRC, "gemv_mq8g256")?;

        let xq_ptr = self.scratch.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.scratch.mq_x_scales.as_ref().unwrap().as_ptr();

        let func = &self.functions["gemv_mq8g256"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut xs = xs_ptr;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// MagnumQuant MQ8: FWHT rotate + INT8 quantize x, then dp4a GEMV.
    pub fn gemv_mq8g256_with_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rotate_quantize_x_mq8(x, k)?;
        self.gemv_mq8g256_prerotated(a_raw, y, m, k)
    }

    /// HFQ3-G256 GEMV. K must be multiple of 256.
    /// Per-arch dispatch: gfx1100/1101/1102 uses the K4-unrolled
    /// 4-accumulator variant. The default kernel was re-ported to match
    /// the same ordering so non-RDNA3 archs (gfx1010, gfx1030, gfx12,
    /// gfx9xx) produce byte-exact results against the RDNA3 baseline.
    /// Uses `launch_maybe_blob` for HIPFIRE_GRAPH=1 capture safety.
    pub fn gemv_hfq3g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq3g256")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] dot x.
    /// Used by `weight_gemv_residual` MQ3 arm to eliminate the
    /// alloc+gemv+add+free fallback chain (saves ~3 launches per residual).
    /// gfx1100 selects the K4-unrolled chip-specific variant (commit 0003103,
    /// 9B MQ3 decode 114 to 141 tok/s); other archs use the K4-ported default
    /// (re-port in 9fdba4d keeps non-RDNA3 archs byte-exact with the prior
    /// gemv + add_inplace path). Uses launch_maybe_blob for HIPFIRE_GRAPH=1
    /// capture safety.
    pub fn gemv_hfq3g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq3g256_residual_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq3g256_residual")?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "gemv_hfq3g256_residual",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// MagnumQuant MQ3-G256 GEMV with fused residual add. The pre-rotation
    /// happens in a separate kernel via fused_silu_mul_mq_rotate or
    /// rotate_x_for_mq; this function just dispatches the underlying
    /// hfq3g256_residual against the already-rotated x.
    pub fn gemv_mq3g256_residual_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq3g256_residual(a_raw, x_rot, y, m, k)
    }

    /// HFQ3-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq3g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq3g128", kernels::GEMV_HFQ3G128_SRC, "gemv_hfq3g128")?;
        let func = &self.functions["gemv_hfq3g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ2-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq2g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq2g128", kernels::GEMV_HFQ2G128_SRC, "gemv_hfq2g128")?;
        let func = &self.functions["gemv_hfq2g128"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ6-G256 GEMV with fused residual add: y[row] += A[row] . x.
    /// Same shape as gemv_hfq6g256; only the final write differs (+= vs =).
    /// Used for wo and w_down in HFQ6 / MQ6 forward paths so the
    /// add_inplace_f32 follow-up launch can be elided.
    pub fn gemv_hfq6g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Wave64-native fast path (gfx906/908/94x): 2 rows per block, halves
        // grid.x. Mirrors the HFQ4 sibling at line ~5378. Plan §3.1.1 item 2
        // (gfx906-mq6-mq8-port.md v3.2.1 + v3.2.2). Byte-exact with the
        // wave32 base since each warp's 32-lane reduction stays in-warp.
        // ILP-prefetch variant gates on gemv_prefetch_enabled(arch) — default
        // on for gfx906 (Phase A.1b, mirror of HFQ4 +4.8% lever from `3ef127d`).
        if self.arch_caps.is_wave64_native() {
            let (kname, ksrc): (&str, &str) = if self.arch_caps.gemv_prefetch_enabled() {
                (
                    "gemv_hfq6g256_residual_wave64_prefetch",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                )
            } else {
                (
                    "gemv_hfq6g256_residual_wave64",
                    kernels::GEMV_HFQ6G256_RESIDUAL_WAVE64_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
            let func = &self.functions[kname];
            let grid = ((m as u32) + 1) / 2;
            return unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid, 1, 1],
                    [64, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        self.ensure_kernel(
            "gemv_hfq6g256_residual",
            kernels::GEMV_HFQ6G256_RESIDUAL_SRC,
            "gemv_hfq6g256_residual",
        )?;
        let func = &self.functions["gemv_hfq6g256_residual"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ6-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq6g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC, "gemv_hfq6g256")?;
        let func = &self.functions["gemv_hfq6g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ8-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq8g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq8g256", kernels::GEMV_HFQ8G256_SRC, "gemv_hfq8g256")?;
        let func = &self.functions["gemv_hfq8g256"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G512 GEMV. K must be multiple of 512.
    pub fn gemv_hfq4g512(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq4g512", kernels::GEMV_HFQ4G512_SRC, "gemv_hfq4g512")?;
        let func = &self.functions["gemv_hfq4g512"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G1024 GEMV. K must be multiple of 1024.
    pub fn gemv_hfq4g1024(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g1024",
            kernels::GEMV_HFQ4G1024_SRC,
            "gemv_hfq4g1024",
        )?;
        let func = &self.functions["gemv_hfq4g1024"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G256 GEMV: flat 4-bit with 256-weight groups. K must be multiple of 256.
    pub fn gemv_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (hfq4g256_src, hfq4g256_module) =
            kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
        self.ensure_kernel(hfq4g256_module, hfq4g256_src, "gemv_hfq4g256")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        // Multi-row GEMV: one warp computes R output rows, sharing x register
        // state across rows. Per-arch default picks R=1 on RDNA3 (negative)
        // and RDNA2 (has its own arch-specific narrow path), R=2 on the
        // default gfx1010-baseline path (gfx1010, gfx1013 Cyan Skillfish,
        // etc.). Override any arch with HIPFIRE_GEMV_ROWS ∈ {1, 2, 4, 8}.
        //
        // See gemv_rows_default() for the measurement data that motivates
        // the per-arch defaults.
        let rdna3 = self.arch_caps.is_rdna3_dgpu();
        let rows = self.arch_caps.gemv_rows_default();
        let use_multirow = rows > 1;

        // RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel.
        // Other non-RDNA3 archs: use wide kernel (2 rows/block) for large M.
        let use_wide = !use_multirow
            && m >= 64
            && !(self.arch_caps.is_rdna2() || self.arch_caps.is_rdna3_dgpu());

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256", bytes);
        let result = if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            let (mr_name, mr_src) = if rdna3 {
                (
                    "gemv_hfq4g256_multirow_rdna3",
                    kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC,
                )
            } else {
                (
                    "gemv_hfq4g256_multirow_default",
                    kernels::GEMV_HFQ4G256_MULTIROW_SRC,
                )
            };
            self.ensure_kernel(mr_name, mr_src, func_name)?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(
                func_name,
                [grid, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        } else if use_wide {
            self.ensure_kernel(
                "gemv_hfq4g256_wide",
                kernels::GEMV_HFQ4G256_WIDE_SRC,
                "gemv_hfq4g256_wide",
            )?;
            let grid = ((m + 1) / 2) as u32;
            self.launch_maybe_blob(
                "gemv_hfq4g256_wide",
                [grid, 1, 1],
                [64, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256",
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
    /// Same math as `gemv_hfq4g256` but the final write accumulates into `y`
    /// instead of overwriting. Used for wo / w_down projections where the
    /// following step would have been `x += gemv_out` via add_inplace_f32.
    pub fn gemv_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemv_hfq4g256_residual_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemv_hfq4g256_residual")?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        // CDNA3 wave64 fast path: 2 rows per block, halves grid.x. The base
        // kernel runs at half throughput on a wave64-native arch because
        // half the wave masks out per `__shfl_down`. Byte-exact with base.
        let cdna3 = self.arch_caps.is_wave64_native();

        // RDNA3 multi-row override path. Same selector as the non-residual
        // variant but there's currently no gfx1010-default multi-row residual
        // kernel, so non-RDNA3 archs still take the single-row residual path
        // regardless of HIPFIRE_GEMV_ROWS. (TODO: port the multi-row residual
        // kernel to the default path if/when the non-residual multi-row wins
        // scale to justify residual too.)
        let rdna3 = self.arch_caps.is_rdna3_dgpu();
        let rows = if rdna3 {
            self.flags.gemv_rows.unwrap_or(1)
        } else {
            1
        };
        let use_multirow = rdna3 && rows > 1;

        // Bandwidth: weight + x + y_read (for residual) + y_write.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_residual", bytes);
        let result = if cdna3 {
            // gfx94x (CDNA3 / MI300X) takes the LDS-cached 8-rows-per-WG path
            // when enabled; gfx906/908 (or env override) keep wave64 base.
            if self.flags.gfx942_gemv_v3 {
                let kname = "gemv_hfq4g256_residual_v3_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_V3_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 7) / 8;
                self.launch_maybe_blob(kname, [grid, 1, 1], [256, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            } else if self.arch_caps.is_cdna3() && self.flags.gfx942_gemv_v2.unwrap_or(true) {
                let kname = "gemv_hfq4g256_residual_v2_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_V2_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 3) / 4;
                self.launch_maybe_blob(kname, [grid, 1, 1], [128, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            } else if self.arch_caps.has_cdna3_lds_gemv()
                && !self.arch_caps.gemv_prefetch_enabled()
                && (k as u32) * 4 <= 32768
            {
                let kname = "gemv_hfq4g256_residual_gfx942";
                self.ensure_kernel(kname, kernels::GEMV_HFQ4G256_RESIDUAL_GFX942_SRC, kname)?;
                let grid = ((m as u32) + 7) / 8;
                let lds_bytes = (k as u32) * 4;
                self.launch_maybe_blob(
                    kname,
                    [grid, 1, 1],
                    [256, 1, 1],
                    lds_bytes,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(a_ptr);
                        b.push_ptr(x_ptr);
                        b.push_ptr(y_ptr);
                        b.push_i32(m_val);
                        b.push_i32(k_val);
                        b
                    },
                )
            } else {
                let (kname, ksrc): (&str, &str) = if self.arch_caps.gemv_prefetch_enabled() {
                    (
                        "gemv_hfq4g256_residual_wave64_prefetch",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                    )
                } else {
                    (
                        "gemv_hfq4g256_residual_wave64",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_SRC,
                    )
                };
                self.ensure_kernel(kname, ksrc, kname)?;
                let grid = ((m as u32) + 1) / 2;
                self.launch_maybe_blob(kname, [grid, 1, 1], [64, 1, 1], 0, &mut params, || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                })
            }
        } else if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_residual_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_residual_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_residual_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            self.ensure_kernel(
                "gemv_hfq4g256_residual_multirow_rdna3",
                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC,
                func_name,
            )?;
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            self.launch_maybe_blob(func_name, [grid, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            })
        } else {
            self.launch_maybe_blob(
                "gemv_hfq4g256_residual",
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                },
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4-G256 GEMV with fused SCALED residual add, CPU-scalar variant:
    ///   y[row] += scale * (A[row] · x)
    /// where `scale` is host-supplied by kernarg. Replaces the three-kernel
    /// tail of the MoE routed-expert epilogue (gemv → scale → add_inplace)
    /// with a single launch. Bit-exact with gemv_hfq4g256_residual followed
    /// by scaled_add_inplace_cpu_scalar when the inputs are identical —
    /// same accumulator layout, same pairwise combine.
    pub fn gemv_hfq4g256_residual_scaled_cpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        scale: f32,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_cpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let s_val = scale;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &s_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_scaled_cpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_cpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_f32(s_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4-G256 GEMV with fused SCALED residual add, GPU-scalar variant:
    ///   y[row] += c_buf[0] * (A[row] · x)
    /// Reads the scale from a 1-element device buffer. Used by the MoE
    /// shared-expert epilogue where `c_buf` holds sigmoid(gate · x) computed
    /// entirely on-device, avoiding a D2H sync.
    pub fn gemv_hfq4g256_residual_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_scaled_gpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_scaled_gpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Same as `gemv_hfq4g256_residual_scaled_gpu` but applies sigmoid to
    /// `c_buf[0]` before scaling — lets the caller skip a separate
    /// `sigmoid_f32` launch on the 1-elem shared-expert gate scalar.
    /// Used by the A3B MoE FFN shared-expert down path.
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        c_buf: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let c_ptr = c_buf.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// N-batched variant of `gemv_hfq4g256_residual_sigmoid_scaled_gpu`.
    /// `x_batch` is [N × K], `y_batch` is [N × M], `c_batch` is [N]. Each
    /// (row, token) block runs the HFQ4G256 GEMV body on its token's x
    /// row and atomicAdd's `sigmoid(c_batch[token]) * acc` into
    /// `y_batch[token × M + row]`. Used by the batched MoE FFN shared-
    /// expert down projection to eliminate N per-token launches.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4-G128 batched GEMV with fused per-token sigmoid-scaled residual.
    ///
    /// y_batch[token, row] += sigmoid(c_batch[token]) * (A[row] · x_batch[token])
    ///
    /// HFQ4-G128 layout: 72 bytes per 128-element group (vs HFQ4-G256's
    /// 136 B/256-element group). Used by the PARO shared-expert down
    /// dispatch in `prefill_moe_ffn_body_batched` (Phase 2 — admit gated
    /// behind HIPFIRE_PARO_BATCHED=1). Same grid/block contract as the
    /// HFQ4-G256 sister: grid=[M × batch_size × 1], block=[32 × 1 × 1].
    pub fn gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g128_residual_sigmoid_scaled",
            kernels::GEMV_HFQ4G128_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ6/MQ6 analogue of `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`.
    /// Same kernel shape (grid = `M × batch`, block = 32, one warp per
    /// `(row, token)`), but reads HFQ6's 200 B / group layout (4 B scale +
    /// 4 B zero + 192 B packed 6-bit nibbles). MQ6G256 shares storage with
    /// HFQ6G256 — caller applies the FWHT rotation upstream, same convention
    /// as MQ4 / HFQ4. Used by the batched MoE FFN shared-expert `down`
    /// projection in the AWQ-style mixed-precision path where shared.down
    /// is MQ6 (12 of 40 layers in AWQ A3B fall into this case).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
        &mut self,
        a_raw: &GpuTensor,
        x_batch: &GpuTensor,
        y_batch: &GpuTensor,
        c_batch: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_residual_sigmoid_scaled",
            kernels::GEMV_HFQ6G256_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x_batch.buf.as_ptr();
        let y_ptr = y_batch.buf.as_ptr();
        let c_ptr = c_batch.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // HFQ6 weight footprint: m * (k / 256) * 200 bytes per row + 4 B per
        // input/output cell. No dedicated profile helper yet (HFQ6 GEMV
        // currently doesn't appear in profile.rs); inlined here.
        let groups = k / 256;
        let weight_bytes = m * groups * 200;
        let bytes = batch_size * (weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_ptr(c_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in a
    /// single launch. Caller passes the 8 selected experts' weight
    /// tensors (in top-K order); the kernel's grid.y picks which expert
    /// each block uses. Outputs are SPLIT into `y_gate` (first mi rows of
    /// each expert) and `y_up` (second mi rows), both `[k_top × mi]`
    /// row-major, so the next-stage batched silu_mul_rotate can consume
    /// them as plain [batch × K] buffers without extra strided reads.
    ///
    /// Bit-exact with running `gemv_hfq4g256` 8 times (same accumulator
    /// layout and pairwise final combine). `k_top` is currently hardcoded
    /// to 8 to match A3B; a generic path can follow alongside Phase 2b.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8(
        &mut self,
        w0: &GpuTensor,
        w1: &GpuTensor,
        w2: &GpuTensor,
        w3: &GpuTensor,
        w4: &GpuTensor,
        w5: &GpuTensor,
        w6: &GpuTensor,
        w7: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, // [k_top × mi] — first half
        y_up: &GpuTensor,   // [k_top × mi] — second half
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_gate_up",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_SRC,
            "gemv_hfq4g256_moe_gate_up_k8",
        )?;
        let w0p = w0.buf.as_ptr();
        let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr();
        let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr();
        let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr();
        let w7p = w7.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void,
            &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void,
            &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void,
            &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void,
            &w7p as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // Bandwidth: 8× weight, x read 8× (cached in practice), 8×m writes.
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_moe_gate_up_k8", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_gate_up_k8",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p);
                b.push_ptr(w1p);
                b.push_ptr(w2p);
                b.push_ptr(w3p);
                b.push_ptr(w4p);
                b.push_ptr(w5p);
                b.push_ptr(w6p);
                b.push_ptr(w7p);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MoE fused down GEMV with scaled residual: accumulates 8 top-K
    /// experts' weighted contributions into `x_residual` in a single
    /// kernel launch. Grid.y selects the expert; each block atomicAdds
    /// `s_rank * (W_rank[row] · rot_batch[rank, :])` into `x_residual[row]`.
    /// Replaces 8 separate `gemv_hfq4g256_residual_scaled_cpu` calls.
    ///
    /// Atomic-add summation order is non-deterministic, so bit-exactness
    /// across runs isn't guaranteed (vs the sequential per-expert path).
    /// For A3B the MoE contribution is added on top of a non-trivial base,
    /// so the ordering-dependent FP noise is tiny in practice and the
    /// smoke-test decode still matches the Phase 2c step 2 output.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8(
        &mut self,
        w0: &GpuTensor,
        w1: &GpuTensor,
        w2: &GpuTensor,
        w3: &GpuTensor,
        w4: &GpuTensor,
        w5: &GpuTensor,
        w6: &GpuTensor,
        w7: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        scales: [f32; 8],
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down",
            kernels::GEMV_HFQ4G256_MOE_DOWN_SRC,
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
        )?;
        let w0p = w0.buf.as_ptr();
        let w1p = w1.buf.as_ptr();
        let w2p = w2.buf.as_ptr();
        let w3p = w3.buf.as_ptr();
        let w4p = w4.buf.as_ptr();
        let w5p = w5.buf.as_ptr();
        let w6p = w6.buf.as_ptr();
        let w7p = w7.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let [s0, s1, s2, s3, s4, s5, s6, s7] = scales;
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w0p as *const _ as *mut c_void,
            &w1p as *const _ as *mut c_void,
            &w2p as *const _ as *mut c_void,
            &w3p as *const _ as *mut c_void,
            &w4p as *const _ as *mut c_void,
            &w5p as *const _ as *mut c_void,
            &w6p as *const _ as *mut c_void,
            &w7p as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s0 as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &s3 as *const _ as *mut c_void,
            &s4 as *const _ as *mut c_void,
            &s5 as *const _ as *mut c_void,
            &s6 as *const _ as *mut c_void,
            &s7 as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_residual_scaled_k8",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(w0p);
                b.push_ptr(w1p);
                b.push_ptr(w2p);
                b.push_ptr(w3p);
                b.push_ptr(w4p);
                b.push_ptr(w5p);
                b.push_ptr(w6p);
                b.push_ptr(w7p);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_f32(s0);
                b.push_f32(s1);
                b.push_f32(s2);
                b.push_f32(s3);
                b.push_f32(s4);
                b.push_f32(s5);
                b.push_f32(s6);
                b.push_f32(s7);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MoE router GPU softmax + top-K + (optional) renormalize. One
    /// workgroup, no D2H sync. Writes [k_top] i32 indices and [k_top]
    /// f32 weights to device buffers. Hardcoded k_top=8 to match A3B.
    pub fn moe_softmax_topk_renorm_k8(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor, // i32 [k_top]
        topk_w: &GpuTensor,   // f32 [k_top]
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_softmax_topk_k8",
            kernels::MOE_SOFTMAX_TOPK_K8_SRC,
            "moe_softmax_topk_renorm_k8",
        )?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = n_exp * 4 + 8 * 8;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_softmax_topk_renorm_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_softmax_topk_renorm_k8",
            [1, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(n);
                b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MoE top-K + renorm given pre-softmaxed probs. Companion to the
    /// regular `softmax_f32`. The dispatch site runs `softmax_f32` first,
    /// then this kernel — same softmax math everywhere, no 1-ULP
    /// divergence between the routing path and a CPU reference.
    pub fn moe_topk_renorm_k8(
        &mut self,
        probs: &GpuTensor,    // [n_exp] f32, pre-softmaxed
        topk_idx: &GpuTensor, // i32 [k_top]
        topk_w: &GpuTensor,   // f32 [k_top]
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_topk_renorm_k8",
            kernels::MOE_TOPK_RENORM_K8_SRC,
            "moe_topk_renorm_k8",
        )?;
        let lp = probs.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = n_exp * 4 + 8 * 8;
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "moe_topk_renorm_k8", bytes);
        let result = self.launch_maybe_blob(
            "moe_topk_renorm_k8",
            [1, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(n);
                b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Index-aware MoE gate_up GEMV. Reads expert_ids from a device-side
    /// topk_indices buffer and weight bases from expert_ptrs[expert_id].
    /// hipGraph-capture-safe replacement for the kernarg-pointer variant.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
                [32u32, 1, 1],
                m as u32,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
            bytes,
        );
        let result =
            self.launch_maybe_blob(func_name, [grid_x, 8, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4G128 (ParoQuant) variant of the indexed MoE gate_up GEMV.
    /// wave32-only (gfx10/11/12) — no wave64 path yet because ParoQuant
    /// A3B is not currently validated on gfx94x.
    pub fn gemv_paro_q4g128_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_gate_up_indexed",
            kernels::GEMV_PARO_Q4G128_MOE_GATE_UP_INDEXED_SRC,
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed",
            [m as u32, 8, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Index-aware MoE down GEMV with scaled residual. Same pattern as
    /// the indexed gate_up; also reads scales from a device topk_weights
    /// buffer and atomicAdds the contribution into x_residual.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
                [32u32, 1, 1],
                m as u32,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed",
            bytes,
        );
        let result =
            self.launch_maybe_blob(func_name, [grid_x, 8, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// N-batched MoE softmax + top-K + renorm. Grid = (N, 1, 1); one
    /// workgroup per token. `logits` is [N × n_exp], `topk_idx` is
    /// [N × K_TOP] i32, `topk_w` is [N × K_TOP] f32.
    pub fn moe_softmax_topk_renorm_k8_batched(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_softmax_topk_k8_batched",
            kernels::MOE_SOFTMAX_TOPK_K8_BATCHED_SRC,
            "moe_softmax_topk_renorm_k8_batched",
        )?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = (n_exp * 4 + 8 * 8) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_softmax_topk_renorm_k8_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_softmax_topk_renorm_k8_batched",
            [batch_size as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(n);
                b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched companion of `moe_topk_renorm_k8` for the prefill path.
    /// Takes pre-softmaxed probs of shape `[batch_size × n_exp]` and writes
    /// `[batch_size × K_TOP]` indices and weights. Caller must run a batched
    /// softmax (`gpu.softmax_f32` on a [batch_size × n_exp] tensor) before
    /// calling this kernel.
    pub fn moe_topk_renorm_k8_batched(
        &mut self,
        probs: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_topk_renorm_k8_batched",
            kernels::MOE_TOPK_RENORM_K8_BATCHED_SRC,
            "moe_topk_renorm_k8_batched",
        )?;
        let lp = probs.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = if norm_topk { 1i32 } else { 0i32 };
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        let bytes = (n_exp * 4 + 8 * 8) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_topk_renorm_k8_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_topk_renorm_k8_batched",
            [batch_size as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(n);
                b.push_i32(nr);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// N-batched indexed MoE gate_up. Grid = (M, K_TOP, N). `x` is
    /// [N × K], `topk_indices` is [N × K_TOP] i32, `y_gate` and `y_up`
    /// are [N × K_TOP × MI] where MI = M / 2.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64",
                [64, 1, 1],
                2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
                [32, 1, 1],
                1,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// N-batched indexed MoE down + scaled residual. Grid = (M, K_TOP, N).
    /// `rot_batch` is [N × K_TOP × K], `x_residual` is [N × M]; the kernel
    /// atomicAdd's per-token slices. `topk_indices` / `topk_weights` are
    /// [N × K_TOP].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64",
                [64, 1, 1],
                2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_down_indexed_batched",
                kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_SRC,
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
            )?;
            (
                "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
                [32, 1, 1],
                1,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched",
            bytes,
        );
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Atomic-free counterpart to
    /// `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`. Writes
    /// each (token, krank) result to its own row of `expert_outputs`
    /// ([N × K_TOP × M], f32) instead of atomicAdd'ing the scaled sum into
    /// `x_residual`. Pair with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots back into the residual with topk_weights applied.
    ///
    /// Observed lift on R9700/gfx1201: 387 → ~900 GiB/s for the down GEMV
    /// (no K_TOP-way atomic contention per output cell). Wave32-only
    /// (RDNA) for now — the CDNA wave64 path stays on the residual_scaled
    /// kernel; atomicAdd on HBM is faster there and the contention pattern
    /// is different.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4G128 (ParoQuant) variant of the atomic-free batched indexed
    /// MoE down. Same expanded-output contract as the HFQ4G256 sibling;
    /// caller must follow with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots into x_residual with topk_weights applied. wave32-only.
    #[allow(clippy::too_many_arguments)]
    /// N-batched indexed MoE gate_up GEMV for HFQ4G128 (ParoQuant routed
    /// experts). Sister of `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
    /// with 72 B/group stride. The caller MUST pre-rotate x using the
    /// layer's shared `gate_up` Givens sidecar (givens_rotate_to into
    /// x_rot_batch) before calling — this kernel is rotation-agnostic and
    /// just reads HFQ4G128 nibbles. Grid: (M, K_TOP, N) wave32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            kernels::GEMV_PARO_Q4G128_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC,
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_gate_up_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemv_paro_q4g128_moe_down_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            kernels::GEMV_PARO_Q4G128_MOE_DOWN_K8_INDEXED_BATCHED_SRC,
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g128_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_paro_q4g128_moe_down_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Index-aware MoE gate_up GEMV for HFQ6G256-layout routed experts.
    /// Wave32 (RDNA) only — CDNA wave64 path stays on the residual_scaled
    /// kernel family. Used to keep mixed-kmap A3B (post-PR-199 alternating
    /// MQ4→MQ6 promotion) on the device-side top-K path under hipGraph
    /// capture.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_SRC,
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // HFQ6 uses 200 bytes/group vs HFQ4's 136. Bytes estimate scales
        // accordingly. Reuse the existing profile helper with a 200/136
        // ratio so timer estimates are roughly correct.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = 8 * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_gate_up_k8_indexed",
            [m as u32, 8, 1],
            [32u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ6G256 batched gate_up MoE GEMV. Same kernarg signature + grid
    /// (M, K_TOP, N) + gate/up output split as the HFQ4 batched gate_up
    /// kernel, only the per-group dequant differs (200 B/group, 6-bit).
    /// Pairs with the HFQ6 expanded down kernel for the batched LFM2.5-MoE
    /// decode path (MQ6-promoted expert layers). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_gate_up_indexed_batched",
            kernels::GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // 200 vs 136 B/group: scale the HFQ4 byte estimate by 200/136.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_gate_up_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ6G256 counterpart to `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.
    /// Atomic-free expand-then-combine for the MoE down step. Pairs with
    /// `moe_down_combine_k8_batched` (dtype-independent — operates on the
    /// f32 expanded buffer). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
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
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ6G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    pub fn gemv_q8_0(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        // Adaptive dispatch: wide kernel for small K (more threads per row),
        // narrow kernel for large K (more blocks, better occupancy).
        if k <= 1536 {
            self.ensure_kernel(
                "gemv_q8_0_wide",
                kernels::GEMV_Q8_0_WIDE_SRC,
                "gemv_q8_0_wide",
            )?;
            let block_size = 64u32; // 2 warps, each processes one row
            let grid = ((m + 1) / 2) as u32; // ceil(M/2)
            return self.launch_maybe_blob(
                "gemv_q8_0_wide",
                [grid, 1, 1],
                [block_size, 1, 1],
                0,
                &mut params,
                blob_builder,
            );
        }

        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let block_size = 32u32;
        self.launch_maybe_blob(
            "gemv_q8_0",
            [m as u32, 1, 1],
            [block_size, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// y = A_q8hfq * x (split-metadata Q8 GEMV, row_stride = padded row bytes)
    pub fn gemv_q8hfq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        row_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut rs_val = row_stride as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut rs_val as *mut _ as *mut c_void,
        ];

        if k <= 1536 {
            self.ensure_kernel(
                "gemv_q8hfq_wide",
                kernels::GEMV_Q8HFQ_WIDE_SRC,
                "gemv_q8hfq_wide",
            )?;
            let func = &self.functions["gemv_q8hfq_wide"];
            let block_size = 64u32;
            let grid = ((m + 1) / 2) as u32;
            return unsafe {
                self.hip
                    .launch_kernel(func, [grid, 1, 1], [block_size, 1, 1], 0, None, &mut params)
            };
        }

        self.ensure_kernel("gemv_q8hfq", kernels::GEMV_Q8HFQ_SRC, "gemv_q8hfq")?;
        let func = &self.functions["gemv_q8hfq"];
        unsafe {
            self.hip
                .launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q6k * x (quantized GEMV for Q6_K)
    pub fn gemv_q6k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_q6k", kernels::GEMV_Q6K_SRC, "gemv_q6k")?;
        let func = &self.functions["gemv_q6k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 64)
    /// a_raw: raw Q4_F16_G64 bytes on GPU, x: F32 input, y: F32 output
    /// Block: 36 bytes per 64 elements. K must be multiple of 64.
    /// Uses 128 threads (4 warps) with shared memory reduction for increased MLP.
    pub fn gemv_q4f16_g64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g64",
            kernels::GEMV_Q4F16_G64_SRC,
            "gemv_q4f16_g64",
        )?;
        let func = &self.functions["gemv_q4f16_g64"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (256-thread wide variant for occupancy testing)
    /// Element-strided access pattern matching F32 GEMV. Shared memory reduction.
    pub fn gemv_q4f16_g64_wide(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g64_wide",
            kernels::GEMV_Q4F16_G64_WIDE_SRC,
            "gemv_q4f16_g64_wide",
        )?;
        let func = &self.functions["gemv_q4f16_g64_wide"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 32)
    /// Block: 20 bytes per 32 elements. K must be multiple of 32.
    pub fn gemv_q4f16_g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q4f16_g32",
            kernels::GEMV_Q4F16_G32_SRC,
            "gemv_q4f16_g32",
        )?;
        let func = &self.functions["gemv_q4f16_g32"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn gemv_f16_xf32(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_f16_xf32", kernels::GEMV_F16_XF32_SRC, "gemv_f16_xf32")?;

        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(w_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        };

        self.launch_maybe_blob(
            "gemv_f16_xf32",
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,  // [k_top × K]
        x_residual: &GpuTensor, // [M]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_indexed",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // MQ2-Lloyd: 72 bytes / 256-weight group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_indexed_batched_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x_rot: &GpuTensor,        // [K] FWHT-rotated
        y_gate: &GpuTensor,       // [k_top × M/2]
        y_up: &GpuTensor,         // [k_top × M/2]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // MQ2-Lloyd: 72 bytes / 256-weight group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MiniMax-M2 (arch_id=10) MoE gate_up GEMV for MQ3-Lloyd experts
    /// (3-bit + 8-entry codebook, 112 B/group). Sibling of
    /// `deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed` — only the
    /// per-group byte stride differs (112 vs 72). X must be FWHT-pre-rotated.
    pub fn deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x_rot: &GpuTensor,        // [K] FWHT-rotated
        y_gate: &GpuTensor,       // [k_top × M/2]
        y_up: &GpuTensor,         // [k_top × M/2]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq3g256_lloyd_moe_gate_up_indexed",
            kernels::GEMV_MQ3G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // MQ3-Lloyd: 112 bytes / 256-weight group.
        let mq3_weight_bytes = m * (k / 256) * 112;
        let bytes = (k_top as usize) * (mq3_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MiniMax-M2 (arch_id=10) fused MoE down GEMV with scaled residual add
    /// for MQ3-Lloyd experts (3-bit + 8-entry codebook, 112 B/group). Sibling
    /// of `deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed` —
    /// only the per-group byte stride differs (112 vs 72).
    pub fn deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,  // [k_top × K]
        x_residual: &GpuTensor, // [M]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq3g256_lloyd_moe_down_indexed",
            kernels::GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        // MQ3-Lloyd: 112 bytes / 256-weight group.
        let mq3_weight_bytes = m * (k / 256) * 112;
        let bytes = (k_top as usize) * (mq3_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rbp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_BATCHED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn wo_per_group_batched_f32(
        &mut self,
        wo_a: &GpuTensor,  // [G, M, K] F32
        x_in: &GpuTensor,  // [B, G, K]
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_f32",
            kernels::WO_PER_GROUP_BATCHED_F32_SRC,
            "wo_per_group_batched_f32",
        )?;
        let func = &self.functions["wo_per_group_batched_f32"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn wo_per_group_batched_hfq4g256(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 256 * 136] bytes
        x_in: &GpuTensor,  // [B, G, K] FWHT-rotated
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_hfq4g256",
            kernels::WO_PER_GROUP_BATCHED_HFQ4G256_SRC,
            "wo_per_group_batched_hfq4g256",
        )?;
        let func = &self.functions["wo_per_group_batched_hfq4g256"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn wo_per_group_batched_q8_0(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 (no FWHT)
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        // DeepSeek V4 prefill shape on gfx1151 (G=8, M=1024, K=4096,
        // B=1024): strided WMMA is ~10x faster than the scalar per-row
        // kernel. Env keeps a one-command fallback for bisects.
        let default_wmma = self.arch == "gfx1151" && k % 32 == 0 && m >= 64 && batch_size >= 64;
        let use_wmma = std::env::var("HIPFIRE_DEEPSEEK4_WO_Q8_WMMA")
            .map(|s| s != "0")
            .unwrap_or(default_wmma);
        if use_wmma && k % 32 == 0 {
            return self.wo_per_group_batched_q8_0_wmma_4w(wo_a, x_in, y_out, g, m, k, batch_size);
        }
        self.wo_per_group_batched_q8_0_1w(wo_a, x_in, y_out, g, m, k, batch_size)
    }
    pub fn wo_per_group_batched_q8_0_1w(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 (no FWHT)
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_q8_0",
            kernels::WO_PER_GROUP_BATCHED_Q8_0_SRC,
            "wo_per_group_batched_q8_0",
        )?;
        let func = &self.functions["wo_per_group_batched_q8_0"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn wo_per_group_batched_q8_0_multirow(
        &mut self,
        wo_a: &GpuTensor,
        x_in: &GpuTensor,
        y_out: &GpuTensor,
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
        rows_per_block: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (name, grid_x) = match rows_per_block {
            2 => (
                "wo_per_group_batched_q8_0_multirow_r2",
                ((m as u32) + 1) / 2,
            ),
            4 => (
                "wo_per_group_batched_q8_0_multirow_r4",
                ((m as u32) + 3) / 4,
            ),
            _ => {
                return Err(hip_bridge::HipError::new(
                    1,
                    "wo_per_group_batched_q8_0_multirow: rows_per_block must be 2 or 4",
                ))
            }
        };
        self.ensure_kernel(name, kernels::WO_PER_GROUP_BATCHED_Q8_0_MULTIROW_SRC, name)?;
        let func = &self.functions[name];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn wo_per_group_batched_q8_0_wmma_4w(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 or F16
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "wo_per_group_batched_q8_0_wmma_4w: K must divide 32"
        );
        self.ensure_kernel(
            "wo_per_group_batched_q8_0_wmma_4w",
            kernels::WO_PER_GROUP_BATCHED_Q8_0_WMMA_4W_SRC,
            "wo_per_group_batched_q8_0_wmma_4w",
        )?;
        let xp_owned = x_in.buf.as_ptr();
        let mut xp = if matches!(x_in.dtype, DType::F16) {
            xp_owned
        } else {
            // Production prefill reuses the same x_in tensor pointer every
            // layer with new contents, so pointer-keyed conversion caching
            // would read stale FP16 here.
            self.convert_fp16_x_uncached(x_in, batch_size as usize * g as usize * k as usize)?
        };
        let func = &self.functions["wo_per_group_batched_q8_0_wmma_4w"];
        let mut wp = wo_a.buf.as_ptr();
        let mut yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [
                    ((m + 63) / 64) as u32,
                    ((batch_size + 63) / 64) as u32,
                    g as u32,
                ],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &mut self,
        expert_ptrs: &GpuTensor,    // [n_exp]
        topk_indices: &GpuTensor,   // [N × K_TOP]
        rot_batch: &GpuTensor,      // [N × K_TOP × K]
        expert_outputs: &GpuTensor, // [N × K_TOP × M] (written, no atomic)
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_EXPANDED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = rot_batch.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
