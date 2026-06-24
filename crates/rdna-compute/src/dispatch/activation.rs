// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Elementwise + activation kernels (silu/gelu/sigmoid/swiglu, scale/add/mul, softmax, layernorm). Pure move (Phase 1 M1).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// c = a + b (element-wise)
    pub fn add_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("add", kernels::ADD_SRC, "add_f32")?;
        let func = &self.functions["add_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params)
        }
    }
    /// c = a + b (element-wise), graph-capture-safe launch path. Keep the
    /// legacy `add_f32` path unchanged for non-captured callers; use this in
    /// subgraphs where stack-backed kernelParams would dangle on replay.
    pub fn add_f32_graph_safe(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("add", kernels::ADD_SRC, "add_f32")?;

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        self.launch_maybe_blob(
            "add_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(a_ptr);
                bb.push_ptr(b_ptr);
                bb.push_ptr(c_ptr);
                bb.push_i32(n_val);
                bb
            },
        )
    }
    /// a += b (in-place element-wise add)
    pub fn add_inplace_f32(&mut self, a: &GpuTensor, b: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("add_inplace", kernels::ADD_INPLACE_SRC, "add_inplace_f32")?;

        let n = a.numel() as i32;
        let a_ptr = a.buf.as_ptr();
        let b_ptr = b.buf.as_ptr();
        let n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "add_inplace_f32", bytes);
        let result = self.launch_maybe_blob(
            "add_inplace_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(a_ptr);
                bb.push_ptr(b_ptr);
                bb.push_i32(n_val);
                bb
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// c = a * b (element-wise)
    pub fn mul_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("mul", kernels::MUL_SRC, "mul_f32")?;
        let func = &self.functions["mul_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "mul_f32", bytes);
        let result = unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params)
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// out = silu(x)
    pub fn silu_f32(&mut self, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("silu", kernels::SILU_SRC, "silu_f32")?;
        let func = &self.functions["silu_f32"];

        let n = x.numel() as i32;
        let mut x_ptr = x.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params)
        }
    }
    /// ReLU-squared: `out = max(0, x)^2` (nemotron_h dense MLP activation).
    pub fn relu2_f32(&mut self, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("relu2", kernels::RELU2_SRC, "relu2_f32")?;
        let func = &self.functions["relu2_f32"];
        let n = x.numel() as i32;
        let mut out_ptr = out.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut n_val = n;
        let mut params: Vec<*mut c_void> = vec![
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params)
        }
    }
    /// out = silu(gate) * up — fused to avoid intermediate buffer
    /// Fused GeGLU: `out = gelu_tanh(gate) * up`. Gemma-family gated MLP
    /// (`gelu_pytorch_tanh`); same launch shape as [`Self::silu_mul_f32`].
    pub fn gelu_mul_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gelu_mul", kernels::GELU_MUL_SRC, "gelu_mul_f32")?;

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "gelu_mul_f32", bytes);
        let result = self.launch_maybe_blob(
            "gelu_mul_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gate_ptr);
                b.push_ptr(up_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn silu_mul_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("silu_mul", kernels::SILU_MUL_SRC, "silu_mul_f32")?;

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "silu_mul_f32", bytes);
        let result = self.launch_maybe_blob(
            "silu_mul_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gate_ptr);
                b.push_ptr(up_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// In-place softmax over last dimension
    pub fn softmax_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("softmax", kernels::SOFTMAX_SRC, "softmax_f32")?;

        let rows = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let x_ptr = x.buf.as_ptr();
        let n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];

        let block = 256u32.min(n as u32);
        let shared_mem = block * 4;

        // Graph-safe launch via launch_maybe_blob. Path B inserts this
        // call into the MoE forward path which gets captured under the
        // verify/HIPFIRE_GRAPH path; raw self.hip.launch_kernel would
        // capture stack-borne kernarg pointers that go dangling on replay.
        self.launch_maybe_blob(
            "softmax_f32",
            [rows as u32, 1, 1],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_i32(n_val);
                b
            },
        )
    }
    /// Sigmoid activation, in-place.
    #[cfg(feature = "deltanet")]
    /// Repeat-interleave Q and K key heads up to value heads count.
    /// Replaces the per-head memcpy loop in DeltaNet for ratio>1 configs:
    /// `dst[(kh*ratio+r)*hd + d] = src[kh*hd + d]`. Does Q and K together
    /// in one launch. For Qwen3.5 9B (24 layers × 64 D2D each), this saves
    /// ~1500 hipMemcpy calls per forward.
    pub fn repeat_interleave_qk_f32(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "repeat_interleave_qk",
            kernels::REPEAT_INTERLEAVE_QK_SRC,
            "repeat_interleave_qk_f32",
        )?;
        let qsp = q_src.buf.as_ptr();
        let ksp = k_src.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let nkh = n_key_heads as i32;
        let r = ratio as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qsp as *const _ as *mut c_void,
            &ksp as *const _ as *mut c_void,
            &qdp as *const _ as *mut c_void,
            &kdp as *const _ as *mut c_void,
            &nkh as *const _ as *mut c_void,
            &r as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total = (n_key_heads * ratio * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = (n_key_heads * head_dim * 4) * 2 // Q/K reads
                  + (n_key_heads * ratio * head_dim * 4) * 2; // Q/K writes
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "repeat_interleave_qk_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "repeat_interleave_qk_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qsp);
                b.push_ptr(ksp);
                b.push_ptr(qdp);
                b.push_ptr(kdp);
                b.push_i32(nkh);
                b.push_i32(r);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched repeat-interleave: repeat key heads across N batch elements in one launch.
    /// q_src/k_src: [N × n_key_heads × head_dim], q_dst/k_dst: [N × n_key_heads × ratio × head_dim].
    pub fn repeat_interleave_qk_f32_batched(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "repeat_interleave_qk_batched",
            kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC,
            "repeat_interleave_qk_f32_batched",
        )?;
        let mut qsp = q_src.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut qdp = q_dst.buf.as_ptr();
        let mut kdp = k_dst.buf.as_ptr();
        let mut nkh = n_key_heads as i32;
        let mut r = ratio as i32;
        let mut hd = head_dim as i32;
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qsp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut qdp as *mut _ as *mut c_void,
            &mut kdp as *mut _ as *mut c_void,
            &mut nkh as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let total = (n_key_heads * ratio * head_dim) as u32;
        let block = 256u32;
        let grid_x = (total + block - 1) / block;
        let bytes =
            n * ((n_key_heads * head_dim * 4) * 2 + (n_key_heads * ratio * head_dim * 4) * 2);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "repeat_interleave_qk_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "repeat_interleave_qk_f32_batched",
            [grid_x, n as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qsp);
                b.push_ptr(ksp);
                b.push_ptr(qdp);
                b.push_ptr(kdp);
                b.push_i32(nkh);
                b.push_i32(r);
                b.push_i32(hd);
                b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[cfg(feature = "deltanet")]
    pub fn sigmoid_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sigmoid", kernels::SIGMOID_SRC, "sigmoid_f32")?;
        let xp = x.buf.as_ptr();
        let n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "sigmoid_f32", bytes);
        let result = self.launch_maybe_blob(
            "sigmoid_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_i32(n);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused `out *= sigmoid(gate)`. Replaces the sigmoid_f32+mul_f32 pair
    /// in the FA attention epilogue (one launch per full-attention layer).
    pub fn sigmoid_mul_f32(&mut self, out: &GpuTensor, gate: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sigmoid_mul", kernels::SIGMOID_MUL_SRC, "sigmoid_mul_f32")?;
        let mut op = out.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut n = out.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize) * 3;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "sigmoid_mul_f32", bytes);
        let result = self.launch_maybe_blob(
            "sigmoid_mul_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(op);
                b.push_ptr(gp);
                b.push_i32(n);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Per-row temperature-scaled softmax probability gather. For each row
    /// `r` in `[0, n_rows)`, returns `probs_out[r] = softmax(logits[r] / temp)[indices[r]]`
    /// — i.e., the softmax probability of the specified token id in that
    /// row's temperature-scaled distribution.
    ///
    /// Used by MTP residual-acceptance sampling spec-decode:
    ///   - n_rows = 1: gather `p_draft(c_k)` after each draft sample
    ///   - n_rows = K: batched gather of `p_target(c_k)` over K verify
    ///     positions, avoiding the 6 MB D2H of full verify logits
    ///
    /// Launch: `n_rows` blocks × 256 threads. Numerically stable via
    /// max-subtraction inside the kernel. `temp` must be > 0.
    ///
    /// Output D2H: `n_rows × 4` bytes (typically ≤ 24 B for K ≤ 6).
    pub fn softmax_prob_gather_batched_f32(
        &mut self,
        logits: &GpuTensor,    // [n_rows × vocab] f32
        indices: &GpuTensor,   // [n_rows] i32 (we use F32 storage; caller reinterprets)
        probs_out: &GpuTensor, // [n_rows] f32
        vocab: usize,
        temperature: f32,
        n_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            temperature > 0.0,
            "softmax_prob_gather_batched: temperature must be > 0"
        );
        assert!(
            n_rows >= 1,
            "softmax_prob_gather_batched: n_rows must be >= 1"
        );
        self.ensure_kernel(
            "softmax_prob_gather_batched",
            kernels::SOFTMAX_PROB_GATHER_BATCHED_SRC,
            "softmax_prob_gather_batched",
        )?;
        let func = &self.functions["softmax_prob_gather_batched"];
        let mut lp = logits.buf.as_ptr();
        let mut ip = indices.buf.as_ptr();
        let mut pp = probs_out.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut tp = temperature;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
        ];
        let nth: u32 = 256;
        let lds: u32 = nth * 4 + 4; // scratch[256] + s_target slot
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_rows as u32, 1, 1],
                [nth, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Alpha gate compute: alpha[i] = softplus(alpha[i] + dt_bias[i]) * (-exp(a_log[i])).
    /// Replaces 85µs CPU roundtrip with ~3µs GPU kernel.
    #[cfg(feature = "deltanet")]
    pub fn alpha_gate_f32(
        &mut self,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("alpha_gate", kernels::ALPHA_GATE_SRC, "alpha_gate_f32")?;
        let func = &self.functions["alpha_gate_f32"];
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "alpha_gate_f32", bytes);
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
    /// Fused `y[i] += c * x[i]` with a CPU-supplied scalar. Merges the
    /// (scale_f32 + add_inplace_f32) pair used by the MoE routed-expert
    /// epilogue — one kernel launch instead of two.
    pub fn scaled_add_inplace_cpu_scalar_f32(
        &mut self,
        y: &GpuTensor,
        x: &GpuTensor,
        c: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            "scaled_add_inplace_cpu_scalar_f32",
        )?;
        let func = &self.functions["scaled_add_inplace_cpu_scalar_f32"];
        let n = y.numel();
        let mut yp = y.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut cv = c;
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut yp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "scaled_add_inplace_cpu_scalar_f32",
            bytes,
        );
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
    /// Fused `y[i] += c_buf[0] * x[i]` where `c_buf` is a 1-element GPU
    /// tensor. Used by the MoE shared-expert epilogue: the scalar gate
    /// is `sigmoid(W_shared_gate · x)` computed entirely on-device, so
    /// passing the result by device pointer saves the D2H sync that a
    /// plain `scale_f32(c_host)` would require.
    pub fn scaled_add_inplace_gpu_scalar_f32(
        &mut self,
        y: &GpuTensor,
        x: &GpuTensor,
        c_buf: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            "scaled_add_inplace_gpu_scalar_f32",
        )?;
        let n = y.numel();
        let yp = y.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let cp = c_buf.buf.as_ptr();
        let nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "scaled_add_inplace_gpu_scalar_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "scaled_add_inplace_gpu_scalar_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(xp);
                b.push_ptr(cp);
                b.push_i32(nv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused `y[row, col] += sigmoid(c_buf[row]) * x[row, col]`.
    /// Used by batched MoE shared experts where the scalar gate is one
    /// on-device logit per token.
    pub fn scaled_add_inplace_gpu_sigmoid_rows_f32(
        &mut self,
        y: &GpuTensor,
        x: &GpuTensor,
        c_buf: &GpuTensor,
        row_width: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            "scaled_add_inplace_gpu_sigmoid_rows_f32",
        )?;
        let n = row_width * batch_size;
        assert!(
            y.numel() >= n && x.numel() >= n,
            "scaled_add_inplace_gpu_sigmoid_rows_f32 expects y/x to cover batch rows"
        );
        assert!(
            c_buf.numel() >= batch_size,
            "scaled_add_inplace_gpu_sigmoid_rows_f32 expects one scalar per batch row"
        );
        let yp = y.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let cp = c_buf.buf.as_ptr();
        let row_width_v = row_width as i32;
        let n_v = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &row_width_v as *const _ as *mut c_void,
            &n_v as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "scaled_add_inplace_gpu_sigmoid_rows_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "scaled_add_inplace_gpu_sigmoid_rows_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(xp);
                b.push_ptr(cp);
                b.push_i32(row_width_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Training row-softmax forward (fp32). `s`,`y`: `[rows*n]`; writes p into y.
    pub fn softmax_train_fwd(
        &mut self,
        s: &GpuTensor,
        y: &GpuTensor,
        rows: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "softmax_train_fwd",
            kernels::SOFTMAX_TRAIN_SRC,
            "softmax_train_fwd",
        )?;
        let func = &self.functions["softmax_train_fwd"];
        let mut sp = s.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
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
    /// Training row-softmax backward (fp32). `dy`,`p`,`ds`: `[rows*n]`.
    pub fn softmax_train_bwd(
        &mut self,
        dy: &GpuTensor,
        p: &GpuTensor,
        ds: &GpuTensor,
        rows: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "softmax_train_bwd",
            kernels::SOFTMAX_TRAIN_SRC,
            "softmax_train_bwd",
        )?;
        let func = &self.functions["softmax_train_bwd"];
        let mut dyp = dy.buf.as_ptr();
        let mut pp = p.buf.as_ptr();
        let mut dsp = ds.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dyp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut dsp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
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
    /// Training SwiGLU forward (fp32): `out = silu(gate)*up`, all `[n]`.
    pub fn swiglu_train_fwd(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swiglu_train_fwd",
            kernels::SWIGLU_TRAIN_SRC,
            "swiglu_train_fwd",
        )?;
        let func = &self.functions["swiglu_train_fwd"];
        let mut gp = gate.buf.as_ptr();
        let mut up_ = up.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_ as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid = ((n as u32) + 255) / 256;
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
    /// Training SwiGLU backward (fp32). Produces `d_gate`,`d_up` `[n]`.
    pub fn swiglu_train_bwd(
        &mut self,
        d_out: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        d_gate: &GpuTensor,
        d_up: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swiglu_train_bwd",
            kernels::SWIGLU_TRAIN_SRC,
            "swiglu_train_bwd",
        )?;
        let func = &self.functions["swiglu_train_bwd"];
        let mut dop = d_out.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut up_ = up.buf.as_ptr();
        let mut dgp = d_gate.buf.as_ptr();
        let mut dup = d_up.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dop as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut up_ as *mut _ as *mut c_void,
            &mut dgp as *mut _ as *mut c_void,
            &mut dup as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid = ((n as u32) + 255) / 256;
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
    /// Elementwise sigmoid forward (fp32). `x`→`out` `[n]`.
    pub fn sigmoid_train_fwd(&mut self, x: &GpuTensor, out: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "sigmoid_train_fwd",
            kernels::SIGMOID_TRAIN_SRC,
            "sigmoid_train_fwd",
        )?;
        let func = &self.functions["sigmoid_train_fwd"];
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid = (n as u32).div_ceil(256);
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
    /// Elementwise sigmoid backward (fp32). `out` is the saved forward output;
    /// `d_x = d_out·out·(1-out)` `[n]`.
    pub fn sigmoid_train_bwd(
        &mut self,
        d_out: &GpuTensor,
        out: &GpuTensor,
        d_x: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "sigmoid_train_bwd",
            kernels::SIGMOID_TRAIN_SRC,
            "sigmoid_train_bwd",
        )?;
        let func = &self.functions["sigmoid_train_bwd"];
        let mut dop = d_out.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut dxp = d_x.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dop as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut dxp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid = (n as u32).div_ceil(256);
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
    /// LayerNorm with bias (batched): out = gamma * (x - mean) / sqrt(var + eps) + beta
    pub fn layernorm_batched(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        n: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("layernorm_f32", kernels::LAYERNORM_SRC, "layernorm_f32")?;
        let func = &self.functions["layernorm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut gp = gamma.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, n) as u32;
        // Round up to power of 2 for reduction
        let block_size = block_size.next_power_of_two();
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// GELU tanh approximation (in-place capable if x == out)
    pub fn gelu_tanh_f32(&mut self, x: &GpuTensor, out: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gelu_tanh_f32", kernels::GELU_TANH_SRC, "gelu_tanh_f32")?;
        let func = &self.functions["gelu_tanh_f32"];
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let blocks = ((n + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Apply 2D rotary positional embedding to the Q and K halves of a packed
    /// QKV buffer for the Qwen3.5-VL vision tower (V is left untouched).
    ///
    /// `cos_t` and `sin_t` are shaped `[N, head_dim/2]` and are looked up
    /// per-(token, d) pair; the kernel reuses the same scalar for both
    /// `d < head_dim/2` and `d + head_dim/2` halves (HF concatenates
    /// `(rotary_pos_emb, rotary_pos_emb)` along the last dim before the
    /// trig table, so the two halves see the same angle).
    ///
    /// Grid=[num_heads, N], Block=[head_dim/2].
    pub fn apply_rope_2d_vision_f32(
        &mut self,
        qkv: &GpuTensor,
        cos_t: &GpuTensor,
        sin_t: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "apply_rope_2d_vision",
            kernels::APPLY_ROPE_2D_VISION_SRC,
            "apply_rope_2d_vision_f32",
        )?;
        let func = &self.functions["apply_rope_2d_vision_f32"];
        let mut qp = qkv.buf.as_ptr();
        let mut cp = cos_t.buf.as_ptr();
        let mut sp = sin_t.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [half, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
