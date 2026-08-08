// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Element-wise norm, activation, arithmetic, cast, transpose, and
//! convolution dispatch methods.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};

/// Monotonic per-launch counter feeding the Q8 GatedDeltaNet state
/// stochastic-rounding dither. Supplies fresh, data-INDEPENDENT entropy each
/// requant so the rounding is genuinely unbiased across the recurrence — the
/// old seed used the state-derived `my_max` with no temporal term, which made
/// the dither a deterministic, data-correlated function and accumulated a
/// systematic bias that drifted the recurrent state on long generations.
static GDN_REQUANT_FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Reserve the same monotonically increasing stochastic-rounding frame IDs
/// used by ordinary HIP GatedDeltaNet launches. Direct AQL replay calls this
/// before submission so bypassing the Rust launch wrapper does not freeze the
/// captured frame scalar.
pub fn reserve_gdn_requant_frames(count: u32) -> u32 {
    GDN_REQUANT_FRAME.fetch_add(count, std::sync::atomic::Ordering::Relaxed)
}

/// Snapshot the next Q8 GatedDeltaNet frame for a single-threaded coherence
/// experiment. Production replay only reserves frames monotonically.
pub fn gdn_requant_frame_checkpoint() -> u32 {
    GDN_REQUANT_FRAME.load(std::sync::atomic::Ordering::Relaxed)
}

/// Restore a coherence-experiment checkpoint so two mutually exclusive arms
/// consume identical stochastic-rounding frame IDs.
pub fn restore_gdn_requant_frame_checkpoint(frame: u32) {
    GDN_REQUANT_FRAME.store(frame, std::sync::atomic::Ordering::Relaxed);
}

/// Q8 DeltaNet-state requant cadence for batched (n_tokens>1) launches.
/// `false` (DEFAULT) = single-end requant at the last token only (MQ4-fast path,
/// recovers the per-token-requant DFlash regression). `true` = per-token Q8
/// roundtrip (PARO drift-echo correctness, ~1.8× slower batched). Strictly OFF
/// for MQ4/HFQ; opt in via `HIPFIRE_DN_REQUANT_PER_TOKEN=1` for PARO checkpoints
/// (shisa-ai A3B). For n_tokens==1 (AR decode / DFlash draft) both are identical.
fn dn_requant_per_token() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DN_REQUANT_PER_TOKEN")
            .map(|v| {
                let v = v.trim();
                !v.is_empty() && v != "0"
            })
            .unwrap_or(false)
    })
}

/// Use the chunked (parallel) FP32 GDN kernel on the multi-token (n>1) linear
/// arm instead of the sequential batch_seq. DEFAULT OFF. Correctness-first
/// PoC: each chunk is a separate host-side launch (cross-chunk is serial).
/// Numerically EQUAL to batch_seq (oracle gdn_chunked_f32, 1.3e-15).
pub fn gdn_chunked() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_GDN_CHUNKED")
            .map(|v| {
                let v = v.trim();
                !v.is_empty() && v != "0"
            })
            .unwrap_or(false)
    })
}

/// Chunk size CS for the chunked FP32 GDN kernel. Default 16 (fits 64 KB LDS
/// with occupancy headroom). Clamped to [1, 32] (CS_MAX). CS=32 is opt-in and
/// occupancy-1; CS>32 is refused (LDS overflow).
pub fn gdn_chunk_size() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let cs = hipfire_config::developer_var("HIPFIRE_GDN_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16);
        cs.clamp(1, 32)
    })
}

impl Gpu {
    /// out = rmsnorm(x, weight, eps)
    pub fn rmsnorm_f32(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;

        let batch = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let x_ptr = x.buf.as_ptr();
        let w_ptr = weight.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n;
        let eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &eps_val as *const _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4; // float per thread

        let bytes = crate::profile::rmsnorm_bytes(batch * n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_f32", bytes);
        let result = self.launch_maybe_blob(
            "rmsnorm_f32",
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched RMSNorm: normalize `batch` vectors of length `n` independently.
    /// x and out can be the same buffer (in-place). Weight is [n], applied per vector.
    pub fn rmsnorm_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        n: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4;
        let bytes = crate::profile::rmsnorm_bytes(batch * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_batched", bytes);
        let result = self.launch_maybe_blob(
            "rmsnorm_f32",
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

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

    /// HIP-graphs-safe variant of `add_f32`. Uses `launch_maybe_blob` instead of
    /// raw `launch_kernel` so kernarg pointers survive stream capture.
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

    /// out = silu(gate) * up — fused to avoid intermediate buffer
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

    /// Batched temperature-scaled softmax, out-of-place. For each of `rows`
    /// rows of width `vocab`, writes `probs[r] = softmax(logits[r] / temp)`
    /// into `probs` and leaves `logits` untouched.
    ///
    /// Mirrors the host `softmax_temp_into` (apply temp to each logit, max over
    /// scaled logits, exp(scaled - max), normalize). The GPU reduction order
    /// (tree) differs from the host sequential sum, so the result is
    /// DISTRIBUTION-parity, NOT byte-identical — used only behind the
    /// HIPFIRE_DFLASH_FAST_SAMPLE opt-in in the DFlash sampled path.
    pub fn softmax_temp_batched_into_f32(
        &mut self,
        logits: &GpuTensor,
        probs: &GpuTensor,
        vocab: usize,
        rows: usize,
        temp: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "softmax_temp_batched",
            kernels::SOFTMAX_TEMP_BATCHED_SRC,
            "softmax_temp_batched_f32",
        )?;

        let logits_ptr = logits.buf.as_ptr();
        let probs_ptr = probs.buf.as_ptr();
        let n_val = vocab as i32;
        let inv_t = 1.0f32 / temp;

        let mut params: Vec<*mut c_void> = vec![
            &logits_ptr as *const _ as *mut c_void,
            &probs_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &inv_t as *const _ as *mut c_void,
        ];

        let block = 256u32.min(vocab as u32).max(1);
        let shared_mem = block * 4;

        self.launch_maybe_blob(
            "softmax_temp_batched_f32",
            [rows as u32, 1, 1],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(logits_ptr);
                b.push_ptr(probs_ptr);
                b.push_i32(n_val);
                b.push_f32(inv_t);
                b
            },
        )
    }

    /// Batched temperature-scaled softmax that ALSO emits, per row, the nucleus
    /// (top_p) threshold `tau_cut[r]` and kept mass `Z[r]` so the host can apply
    /// `p_i = (p_i >= tau_cut[r]) ? p_i / Z[r] : 0` (AR-equivalent nucleus
    /// truncation). `tau_cut` and `Z` must each be `[rows]`-shaped F32 tensors.
    ///
    /// `probs` is left as the FULL normalized softmax (the kernel does NOT
    /// truncate on-device; the host helper does, so the kernel stays a pure read
    /// plus two scalars). When `top_p >= 1.0` the nucleus phase is skipped and
    /// `tau_cut=0, Z=1` so host truncation is identity → byte-equivalent to
    /// `softmax_temp_batched_into_f32`.
    ///
    /// `tau_cut` is found by bisection over the mass predicate (no sort); see the
    /// kernel doc in `softmax_temp_batched.hip`. Distribution-parity (tree
    /// reduction), not byte-parity, to the host softmax — used behind
    /// HIPFIRE_DFLASH_FAST_SAMPLE.
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_temp_topp_batched_into_f32(
        &mut self,
        logits: &GpuTensor,
        probs: &GpuTensor,
        tau_cut: &GpuTensor,
        z: &GpuTensor,
        vocab: usize,
        rows: usize,
        temp: f32,
        top_p: f32,
        top_k: usize,
        min_p: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "softmax_temp_topp_batched",
            kernels::SOFTMAX_TEMP_BATCHED_SRC,
            "softmax_temp_topp_batched_f32",
        )?;

        let logits_ptr = logits.buf.as_ptr();
        let probs_ptr = probs.buf.as_ptr();
        let tau_ptr = tau_cut.buf.as_ptr();
        let z_ptr = z.buf.as_ptr();
        let n_val = vocab as i32;
        let inv_t = 1.0f32 / temp;
        let top_p_val = top_p;
        let top_k_val = top_k as i32;
        let min_p_val = min_p;

        let mut params: Vec<*mut c_void> = vec![
            &logits_ptr as *const _ as *mut c_void,
            &probs_ptr as *const _ as *mut c_void,
            &tau_ptr as *const _ as *mut c_void,
            &z_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &inv_t as *const _ as *mut c_void,
            &top_p_val as *const _ as *mut c_void,
            &top_k_val as *const _ as *mut c_void,
            &min_p_val as *const _ as *mut c_void,
        ];

        // Wider block than the plain softmax (256): the nucleus bisection
        // re-reads the row ~20×, and the grid is only `rows` blocks (~31), so
        // a wider block puts more threads on each active CU to hide memory
        // latency. shared_mem auto-scales (one float per thread for the reduce).
        let block = 1024u32.min(vocab as u32).max(1);
        let shared_mem = block * 4;

        self.launch_maybe_blob(
            "softmax_temp_topp_batched_f32",
            [rows as u32, 1, 1],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(logits_ptr);
                b.push_ptr(probs_ptr);
                b.push_ptr(tau_ptr);
                b.push_ptr(z_ptr);
                b.push_i32(n_val);
                b.push_f32(inv_t);
                b.push_f32(top_p_val);
                b.push_i32(top_k_val);
                b.push_f32(min_p_val);
                b
            },
        )
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
        let func = &self.functions["rope_f32"];

        let q_ptr = q.buf.as_ptr();
        let k_ptr = k.buf.as_ptr();
        let pos_ptr = pos_buf.as_ptr();
        let nhq = n_heads_q as i32;
        let nhk = n_heads_k as i32;
        let hd = head_dim as i32;
        let fb = freq_base;

        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &nhq as *const _ as *mut c_void,
            &nhk as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fb as *const _ as *mut c_void,
        ];

        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid = (half + block - 1) / block;

        self.launch_maybe_blob(
            "rope_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b.push_f32(fb);
                b
            },
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

    // ── DeltaNet ops (feature-gated) ─────────────────────────────────────

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
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nhq as *const _ as *mut c_void,
            &nhk as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &fb as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(entry, grid, [block, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(pp);
            b.push_i32(nhq);
            b.push_i32(nhk);
            b.push_i32(hd);
            b.push_i32(nr);
            b.push_f32(fb);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Exact gfx1100 Qwen3.6-35B full-attention preparation. Replaces the
    /// dependent deinterleave, Q RMS, K RMS, and partial half-split RoPE
    /// dispatches with one head-local launch. Admission is enforced by the
    /// architecture layer; this launcher intentionally exposes only the fixed
    /// 16Q/2K, head_dim=256, n_rot=64 shape.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_fa_prep_gfx1100(
        &mut self,
        q_interleaved: &GpuTensor,
        q: &GpuTensor,
        gate: &GpuTensor,
        k: &GpuTensor,
        q_weight: &GpuTensor,
        k_weight: &GpuTensor,
        pos_buf: &hip_bridge::DeviceBuffer,
        eps: f32,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (module, src, kernel) = if self.arch_caps.is_gfx1151() {
            (
                "qwen35_fa_prep_gfx1151",
                kernels::QWEN35_FA_PREP_GFX1151_SRC,
                "qwen35_fa_prep_gfx1151",
            )
        } else {
            (
                "qwen35_fa_prep_gfx1100",
                kernels::QWEN35_FA_PREP_GFX1100_SRC,
                "qwen35_fa_prep_gfx1100",
            )
        };
        self.ensure_kernel(module, src, kernel)?;

        let qip = q_interleaved.buf.as_ptr();
        let qp = q.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let qwp = q_weight.buf.as_ptr();
        let kwp = k_weight.buf.as_ptr();
        let pp = pos_buf.as_ptr();
        let ep = eps;
        let fb = freq_base;
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
        ];
        let bytes = (16 * 256 * 3 + 2 * 256 * 2 + 18 * 256) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [18, 1, 1],
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Full GPT-J **interleaved** RoPE — rotates pairs (2i, 2i+1) of the first
    /// `n_rot` dims. Matches HF Cohere2's `rotate_half` (`x1=x[..., ::2]`,
    /// `x2=x[..., 1::2]`, `rot=cat(-x2, x1)`), which is explicitly *different
    /// from Llama*. Unlike `rope_partial_interleaved_f32` — which despite its
    /// name dispatches the HALF-SPLIT kernel by default and only interleaves
    /// under `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1` — this ALWAYS dispatches the
    /// interleaved kernel with no global flag, so it is safe alongside
    /// half-split callers (Qwen3.5) in a shared daemon. Pass `n_rot = head_dim`
    /// for full rotary (Cohere2 has no partial_rotary_factor). arch_id 12.
    #[cfg(feature = "deltanet")]
    pub fn rope_interleaved_f32(
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
        self.ensure_kernel(
            "rope_partial_interleaved",
            kernels::ROPE_PARTIAL_INTERLEAVED_SRC,
            "rope_partial_interleaved_f32",
        )?;
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
        let timer =
            crate::profile::begin_timer(&self.hip, "rope", "rope_partial_interleaved_f32", bytes);
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &nhq as *const _ as *mut c_void,
            &nhk as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &fb as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            "rope_partial_interleaved_f32",
            grid,
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched partial-interleaved RoPE. Each batch row reads its absolute
    /// position from positions[b] and rotates the first n_rot dims of every
    /// Q and K head. Q/K are [batch_size × n_heads × head_dim] row-major.
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
        // Added to each positions[b] for the RoPE angle only (the caller's KV-write
        // keeps the raw physical positions). Pass kv_cache.compact_offset so batched
        // Q/K rotate at absolute phase after eviction/compaction; pass 0 when there
        // is no compaction (the common case) — it's a literal no-op offset then.
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
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut po = pos_offset;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
        ];
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid_x = (n_pairs + block - 1) / block;
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rope", entry, bytes);
        let result = self.launch_maybe_blob(
            entry,
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b.push_i32(bs);
                b.push_i32(po);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched GPT-J **interleaved** RoPE — always dispatches the interleaved
    /// batched kernel (no legacy flag), the batched twin of `rope_interleaved_f32`.
    /// For Cohere2 sliding layers in batched prefill; pos_offset = 0 (prefill
    /// does no KV compaction). arch_id 12.
    #[cfg(feature = "deltanet")]
    pub fn rope_interleaved_f32_batched(
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
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_partial_interleaved_batched",
            kernels::ROPE_PARTIAL_INTERLEAVED_BATCHED_SRC,
            "rope_partial_interleaved_batched_f32",
        )?;
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut po = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
        ];
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid_x = (n_pairs + block - 1) / block;
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "rope",
            "rope_partial_interleaved_batched_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "rope_partial_interleaved_batched_f32",
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b.push_i32(bs);
                b.push_i32(po);
                b
            },
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

        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &nhq as *const _ as *mut c_void,
            &nhk as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];

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
        let result =
            self.launch_maybe_blob("rope_2d_halfsplit_f32", grid, block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(cp);
                b.push_ptr(sp);
                b.push_i32(np);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b
            });
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

        let mut params: Vec<*mut c_void> = vec![
            &qkvp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];

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
        let result = self.launch_maybe_blob(
            "rope_2d_halfsplit_qkv_interleaved_f32",
            grid,
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qkvp);
                b.push_ptr(cp);
                b.push_ptr(sp);
                b.push_i32(np);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Split a fused interleaved `[n_patches, 3 * hidden]` QKV buffer
    /// into three separate `[n_patches, hidden]` Q, K, V buffers.
    /// Used by the dots.ocr vision encoder when feeding the
    /// non-causal `attention_dflash_f32` kernel (which expects Q/K/V
    /// as separate flat buffers).
    ///
    /// `hidden` here is `n_heads * head_dim` — the second axis of each
    /// of Q, K, V within the fused buffer.
    pub fn qkv_split_interleaved_f32(
        &mut self,
        qkv: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        n_patches: usize,
        hidden: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            n_patches > 0,
            "qkv_split_interleaved_f32: n_patches must be > 0"
        );
        assert!(hidden > 0, "qkv_split_interleaved_f32: hidden must be > 0");
        self.ensure_kernel(
            "qkv_split_interleaved",
            kernels::QKV_SPLIT_INTERLEAVED_SRC,
            "qkv_split_interleaved_f32",
        )?;

        let qkvp = qkv.buf.as_ptr();
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let np = n_patches as i32;
        let hd = hidden as i32;

        let mut params: Vec<*mut c_void> = vec![
            &qkvp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];

        let block_size = 256u32;
        let grid_y = ((hidden as u32) + block_size - 1) / block_size;
        let grid = [n_patches as u32, grid_y, 1];
        let block = [block_size, 1, 1];
        // Bytes-touched estimate: 3 reads + 3 writes per (patch, j) thread.
        let bytes = n_patches * hidden * 4 * 6;
        let timer =
            crate::profile::begin_timer(&self.hip, "qkv_split", "qkv_split_interleaved_f32", bytes);
        let result = self.launch_maybe_blob(
            "qkv_split_interleaved_f32",
            grid,
            block,
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qkvp);
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_i32(np);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// In-place F32 → bf16 → F32 round-trip on `x`. Used by the
    /// dots.ocr vision encoder for HF-bf16-precision emulation
    /// (see `kernels/src/bf16_round_trip.hip`).
    pub fn bf16_round_trip_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "bf16_round_trip",
            kernels::BF16_ROUND_TRIP_SRC,
            "bf16_round_trip_f32",
        )?;
        let xp = x.buf.as_ptr();
        let n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];
        let block_size = 256u32;
        let grid = (((n as u32) + block_size - 1) / block_size).max(1);
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer =
            crate::profile::begin_timer(&self.hip, "bf16_round_trip", "bf16_round_trip_f32", bytes);
        let result = self.launch_maybe_blob(
            "bf16_round_trip_f32",
            [grid, 1, 1],
            [block_size, 1, 1],
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

    /// Deinterleave: split [A_h0(hd), B_h0(hd), A_h1(hd), B_h1(hd), ...] into A and B.
    /// Replaces per-head memcpy loop (n_heads × 2 ioctls → 1 dispatch).
    pub fn deinterleave_f32(
        &mut self,
        interleaved: &GpuTensor,
        out_a: &GpuTensor,
        out_b: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deinterleave",
            kernels::DEINTERLEAVE_SRC,
            "deinterleave_f32",
        )?;
        let inp = interleaved.buf.as_ptr();
        let ap = out_a.buf.as_ptr();
        let bp = out_b.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &inp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = n_heads * head_dim * 4 * 3; // read interleaved, write both outputs
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "deinterleave_f32", bytes);
        let result = self.launch_maybe_blob(
            "deinterleave_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(inp);
                b.push_ptr(ap);
                b.push_ptr(bp);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched deinterleave: split [N × n_heads × head_dim × 2] interleaved
    /// Q+Gate into separate [N × n_heads × head_dim] Q and Gate tensors.
    /// Replaces the per-token gather/deinterleave/scatter loop in the FA
    /// batched prefill path.
    pub fn deinterleave_f32_batched(
        &mut self,
        interleaved: &GpuTensor,
        out_q: &GpuTensor,
        out_gate: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deinterleave_batched",
            kernels::DEINTERLEAVE_BATCHED_SRC,
            "deinterleave_f32_batched",
        )?;
        let mut inp = interleaved.buf.as_ptr();
        let mut qp = out_q.buf.as_ptr();
        let mut gp = out_gate.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut inp as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid_x = (total + block - 1) / block;
        let bytes = n * n_heads * head_dim * 4 * 3;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deinterleave_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "deinterleave_f32_batched",
            [grid_x, n as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(inp);
                b.push_ptr(qp);
                b.push_ptr(gp);
                b.push_i32(nh);
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

    /// Softplus activation, in-place.
    #[cfg(feature = "deltanet")]
    pub fn softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("softplus", kernels::SOFTPLUS_SRC, "softplus_f32")?;
        let func = &self.functions["softplus_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
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

    /// L2 normalization per head, in-place. One warp per head.
    #[cfg(feature = "deltanet")]
    pub fn l2_norm_f32(
        &mut self,
        x: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("l2_norm", kernels::L2_NORM_SRC, "l2_norm_f32")?;
        let func = &self.functions["l2_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "l2_norm_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [32, 1, 1],
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
        let func = &self.functions["conv1d_decode_f32"];
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

    /// Gated output norm: rmsnorm(x) * silu(z). Fused kernel.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let op = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32",
            [n_heads as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched `gated_norm_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32_batched(
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
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Gated Delta Net recurrence. S matrix in LDS. Processes all tokens sequentially.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net",
            kernels::GATED_DELTA_NET_SRC,
            "gated_delta_net_f32",
        )?;
        let func = &self.functions["gated_delta_net_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        // 32 threads, tiled S in LDS (4KB per tile). Grid: [n_heads, 128/8=16].
        let n_tiles = (128 / 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GDN recurrence with Q8-quantized S state — tiled LDS + warp-shuffle.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        ef_residual: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = s_q8.buf.as_ptr();
        let scp = s_scales.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = reserve_gdn_requant_frames(1) as i32;
        let efp: *mut c_void = ef_residual
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let n_tiles = (128 / 4) as u32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);

        // Use the lean "fast" kernel for the default path (no per-token requant).
        // The fast kernel keeps the requant OUTSIDE the per-token loop, avoiding
        // the VGPR spill-to-scratch on gfx906 wave64 that the full kernel suffers
        // when both code paths are compiled into one function (108→172 bytes scratch).
        // EF residual is supported in both paths; the split is only about cadence.
        let use_fast = !dn_requant_per_token();
        // Single-token entry point has no slot concept of its own; this call
        // site always launches in legacy (slot 0, zero-stride) mode, which
        // the kernel's `row_slot == nullptr` fast path makes bitwise
        // identical to the pre-slot addressing.
        let row_slot_null: *mut c_void = std::ptr::null_mut();
        let s_stride_zero: i32 = 0;

        let result = if use_fast {
            self.ensure_kernel(
                "gated_delta_net_q8_fast",
                kernels::GATED_DELTA_NET_Q8_FAST_SRC,
                "gated_delta_net_q8_fast",
            )?;
            let mut params: Vec<*mut c_void> = vec![
                &qp as *const _ as *mut c_void,
                &kp as *const _ as *mut c_void,
                &vp as *const _ as *mut c_void,
                &gp as *const _ as *mut c_void,
                &bp as *const _ as *mut c_void,
                &sp as *const _ as *mut c_void,
                &scp as *const _ as *mut c_void,
                &op as *const _ as *mut c_void,
                &nt as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &fr as *const _ as *mut c_void,
                &efp as *const _ as *mut c_void,
                &row_slot_null as *const _ as *mut c_void,
                &s_stride_zero as *const _ as *mut c_void,
            ];
            let timer = crate::profile::begin_timer(
                &self.hip,
                "deltanet",
                "gated_delta_net_q8_fast",
                bytes,
            );
            let r = self.launch_maybe_blob(
                "gated_delta_net_q8_fast",
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp);
                    b.push_ptr(kp);
                    b.push_ptr(vp);
                    b.push_ptr(gp);
                    b.push_ptr(bp);
                    b.push_ptr(sp);
                    b.push_ptr(scp);
                    b.push_ptr(op);
                    b.push_i32(nt);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_i32(fr);
                    b.push_ptr(efp);
                    b.push_ptr(row_slot_null);
                    b.push_i32(s_stride_zero);
                    b
                },
            );
            if let Some(t) = timer {
                t.finish(&self.hip);
            }
            r
        } else {
            self.ensure_kernel(
                "gated_delta_net_q8",
                kernels::GATED_DELTA_NET_Q8_SRC,
                "gated_delta_net_q8",
            )?;
            let rpt = 1i32; // per_token is always true in this path
            let mut params: Vec<*mut c_void> = vec![
                &qp as *const _ as *mut c_void,
                &kp as *const _ as *mut c_void,
                &vp as *const _ as *mut c_void,
                &gp as *const _ as *mut c_void,
                &bp as *const _ as *mut c_void,
                &sp as *const _ as *mut c_void,
                &scp as *const _ as *mut c_void,
                &op as *const _ as *mut c_void,
                &nt as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &fr as *const _ as *mut c_void,
                &efp as *const _ as *mut c_void,
                &rpt as *const _ as *mut c_void,
            ];
            let timer =
                crate::profile::begin_timer(&self.hip, "deltanet", "gated_delta_net_q8", bytes);
            let r = self.launch_maybe_blob(
                "gated_delta_net_q8",
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp);
                    b.push_ptr(kp);
                    b.push_ptr(vp);
                    b.push_ptr(gp);
                    b.push_ptr(bp);
                    b.push_ptr(sp);
                    b.push_ptr(scp);
                    b.push_ptr(op);
                    b.push_i32(nt);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_i32(fr);
                    b.push_ptr(efp);
                    b.push_i32(rpt);
                    b
                },
            );
            if let Some(t) = timer {
                t.finish(&self.hip);
            }
            r
        };
        result
    }

    /// Decode-only Q8 GDN path for a 2:1 value-head:QK-head layout.
    ///
    /// `q` and `k` remain compact (`n_heads / 2` heads). The kernel maps state
    /// head `h` to Q/K head `h / 2`, avoiding a separate repeat-interleave
    /// materialization. The launch ABI intentionally matches the regular fast
    /// kernel so replay's dynamic stochastic-rounding frame patch is shared.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_compact2(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        ef_residual: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let gfx1151_r8 = self.arch_caps.is_gfx1151()
            && hipfire_config::developer_var("HIPFIRE_GFX1151_GDN_R8").as_deref() == Ok("1");
        let gfx1151_r4x2 = self.arch_caps.is_gfx1151()
            && hipfire_config::developer_var("HIPFIRE_GFX1151_GDN_R4X2").as_deref() == Ok("1");
        let gfx1151_dpp = self.arch_caps.is_gfx1151()
            && hipfire_config::developer_var("HIPFIRE_GFX1151_GDN_DPP").as_deref() == Ok("1");
        let (kernel_name, kernel_src, n_tiles, block_size) = if gfx1151_dpp {
            (
                "gated_delta_net_q8_compact2_dpp_gfx1151",
                kernels::GATED_DELTA_NET_Q8_COMPACT2_DPP_GFX1151_SRC,
                128 / 4,
                32,
            )
        } else if gfx1151_r4x2 {
            (
                "gated_delta_net_q8_compact2_r4x2_gfx1151",
                kernels::GATED_DELTA_NET_Q8_COMPACT2_R4X2_GFX1151_SRC,
                128 / 8,
                64,
            )
        } else if gfx1151_r8 {
            (
                "gated_delta_net_q8_compact2_r8_gfx1151",
                kernels::GATED_DELTA_NET_Q8_COMPACT2_R8_GFX1151_SRC,
                128 / 8,
                32,
            )
        } else {
            let (kernel_name, kernel_src) =
                match hipfire_config::developer_var("HIPFIRE_GDN_COMPACT2_SHAPE").ok().as_deref() {
                    Some("b2") => (
                        "gated_delta_net_q8_compact2_b2",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC,
                    ),
                    Some("b4") => (
                        "gated_delta_net_q8_compact2_b4",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B4_SRC,
                    ),
                    Some("b8") => (
                        "gated_delta_net_q8_compact2_b8",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B8_SRC,
                    ),
                    Some("b12") => (
                        "gated_delta_net_q8_compact2_b12",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B12_SRC,
                    ),
                    Some("b16") => (
                        "gated_delta_net_q8_compact2_b16",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B16_SRC,
                    ),
                    _ => (
                        "gated_delta_net_q8_compact2_b2",
                        kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC,
                    ),
                };
            (kernel_name, kernel_src, 128 / 4, 32)
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;

        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = s_q8.buf.as_ptr();
        let scp = s_scales.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = reserve_gdn_requant_frames(1) as i32;
        let efp: *mut c_void = ef_residual
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &scp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
            &efp as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [n_heads as u32, n_tiles as u32, 1],
            [block_size, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sp);
                b.push_ptr(scp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(fr);
                b.push_ptr(efp);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched sequential `gated_delta_net_q8` for prefill.
    ///
    /// Launches the single-token kernel N times with offset pointers into
    /// [N × stride]-laid-out Q/K/V/gate/beta/output buffers. This preserves
    /// bit-exact semantics with N × `gated_delta_net_q8(n_tokens=1)` calls
    /// (i.e., dequant→update→requant per token, with stochastic rounding
    /// applied each step) — critical for byte-exact quality gate compliance.
    ///
    /// Why not just call the kernel once with `n_tokens=N`? The existing
    /// kernel dequants S_q8 once at start, runs N updates in FP32 inside
    /// LDS, and requants once at end. That collapses N rounding steps into
    /// one, producing numerically different output from sequential calls —
    /// diverges from the decode-path baseline.
    ///
    /// Q/K/V/output are [N × n_heads × head_dim] row-major.
    /// gate/beta are [N × n_heads] row-major.
    /// S_q8 / s_scales are the shared state (advanced N steps).
    ///
    /// `row_slot` / `s_stride_elems`: DeltaNet's S state is fixed-size and
    /// per-slot independent (unlike KV, no shared arena, no ragged length,
    /// no offset descriptor) — so the slot axis here is a stride, not a
    /// descriptor table. `row_slot[0]` selects the slot for the *entire*
    /// launch: the recurrence is sequential within a slot and never crosses
    /// slots, so one launch advances exactly one slot across its `n_tokens`
    /// tokens (a batch spanning several slots issues one launch per slot).
    /// `row_slot == None` (equivalently a null device pointer) with
    /// `s_stride_elems == 0` is legacy mode: the offset is 0 and addressing
    /// is bitwise unchanged. Only wired into the fast (`gated_delta_net_q8_fast`)
    /// kernel; the per-token-requant slow path (`gated_delta_net_q8`) does not
    /// support slots yet, so `row_slot.is_some()` with that path is rejected
    /// host-side rather than silently pinning every slot's writes to slot 0.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_q8_batch_seq_slots(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        // Optional f16 error-feedback residual; see gated_delta_net_q8. The
        // batched path requants per token in-launch, so EF carries token-to-token
        // (and chunk-boundary) error — consistent with the per-token decode/replay.
        ef_residual: Option<&GpuTensor>,
        row_slot: Option<&GpuTensor>,
        s_stride_elems: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_fast = !dn_requant_per_token();
        if row_slot.is_some() && !use_fast {
            return Err(hip_bridge::HipError::new(
                0,
                "gated_delta_net_q8_batch_seq_slots: slot mode requires the fast \
                 (single-end-requant) kernel; HIPFIRE_DN_REQUANT_PER_TOKEN=1 selects the \
                 per-token-requant slow kernel, which does not support the slot axis yet",
            ));
        }
        let kernel_name = if use_fast {
            "gated_delta_net_q8_fast"
        } else {
            "gated_delta_net_q8"
        };
        let kernel_src = if use_fast {
            kernels::GATED_DELTA_NET_Q8_FAST_SRC
        } else {
            kernels::GATED_DELTA_NET_Q8_SRC
        };
        let kernel_fn = if use_fast {
            "gated_delta_net_q8_fast"
        } else {
            "gated_delta_net_q8"
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_fn)?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sp = s_q8.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        // Reserve n_tokens sequential frame IDs so each token in the
        // single batched launch gets the same stochastic-rounding dither
        // seed it would have gotten from n_tokens sequential per-token
        // launches. The kernel indexes these as `frame + t` (t = 0..n-1).
        let mut fr = reserve_gdn_requant_frames(n_tokens as u32) as i32;
        let mut efp: *mut c_void = ef_residual
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let mut rpt = dn_requant_per_token() as i32;
        let mut rsp: *mut c_void = row_slot
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let mut sse = s_stride_elems as i32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_batch_seq",
            bytes,
        );

        let result = if use_fast {
            let mut params: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut gp as *mut _ as *mut c_void,
                &mut bp as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut scp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut nt as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut fr as *mut _ as *mut c_void,
                &mut efp as *mut _ as *mut c_void,
                &mut rsp as *mut _ as *mut c_void,
                &mut sse as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "gated_delta_net_q8_fast",
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp);
                    b.push_ptr(kp);
                    b.push_ptr(vp);
                    b.push_ptr(gp);
                    b.push_ptr(bp);
                    b.push_ptr(sp);
                    b.push_ptr(scp);
                    b.push_ptr(op);
                    b.push_i32(nt);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_i32(fr);
                    b.push_ptr(efp);
                    b.push_ptr(rsp);
                    b.push_i32(sse);
                    b
                },
            )
        } else {
            let mut params: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut gp as *mut _ as *mut c_void,
                &mut bp as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut scp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut nt as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut fr as *mut _ as *mut c_void,
                &mut efp as *mut _ as *mut c_void,
                &mut rpt as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "gated_delta_net_q8",
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp);
                    b.push_ptr(kp);
                    b.push_ptr(vp);
                    b.push_ptr(gp);
                    b.push_ptr(bp);
                    b.push_ptr(sp);
                    b.push_ptr(scp);
                    b.push_ptr(op);
                    b.push_i32(nt);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_i32(fr);
                    b.push_ptr(efp);
                    b.push_i32(rpt);
                    b
                },
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Legacy single-slot entry point. Preserved so existing call sites are
    /// untouched; passes `None, 0` (null row_slot, zero stride), which the
    /// kernel's `row_slot == nullptr` fast path treats as slot 0 with a
    /// zero offset — bitwise identical to the pre-slot addressing.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        ef_residual: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.gated_delta_net_q8_batch_seq_slots(
            q_batch,
            k_batch,
            v_batch,
            gate_batch,
            beta_batch,
            s_q8,
            s_scales,
            output_batch,
            n_tokens,
            n_heads,
            head_dim,
            ef_residual,
            None,
            0,
        )
    }

    /// Tree-aware variant of `gated_delta_net_q8_batch_seq`. Per-token
    /// S-tile persist-write so sibling tokens read the parent's post-update
    /// state via `s_tape_q8[parent_indices[t]]`. `parent_indices[t] < 0`
    /// means "read pre-block initial state from `s_q8_init`".
    ///
    /// Does NOT advance persistent `s_q8_init` / `s_scales_init` (those
    /// are the pre-block snapshot, read-only). Caller runs linear replay
    /// on the accepted spine post-acceptance to commit the trajectory.
    ///
    /// Tape layout (caller responsibility):
    /// - `s_tape_q8`:     `[n_tokens × n_heads × HD × HD]` i8 (scratch)
    /// - `s_tape_scales`: `[n_tokens × n_heads × HD]` f32 (scratch)
    /// - `parent_indices`: `[n_tokens]` i32 (host materialized by
    ///   `ddtree::linearize_tree`; spine topology is [-1, 0, 1, 2, ...])
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8_init: &GpuTensor,
        s_scales_init: &GpuTensor,
        s_tape_q8: &GpuTensor,
        s_tape_scales: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_q8_tree",
            kernels::GATED_DELTA_NET_Q8_TREE_SRC,
            "gated_delta_net_q8_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sip = s_q8_init.buf.as_ptr();
        let mut scip = s_scales_init.buf.as_ptr();
        let mut stp = s_tape_q8.buf.as_ptr();
        let mut stsp = s_tape_scales.buf.as_ptr();
        let mut pp = parent_indices.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sip as *mut _ as *mut c_void,
            &mut scip as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut stsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_tree_batch_seq",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sip);
                b.push_ptr(scip);
                b.push_ptr(stp);
                b.push_ptr(stsp);
                b.push_ptr(pp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// FP32 tree-aware GDN recurrence — full-precision counterpart of
    /// `gated_delta_net_q8_tree_batch_seq`. No scales tape and no per-token
    /// dequant/requant: `s_f32_init` (pre-block snapshot) and `s_tape_f32`
    /// (per-node tape) are plain f32. Used by the FP32 `StateQuant`
    /// spec-decode tree-verify path.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f32_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_f32_init: &GpuTensor,
        s_tape_f32: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_f32_tree",
            kernels::GATED_DELTA_NET_F32_TREE_SRC,
            "gated_delta_net_f32_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sip = s_f32_init.buf.as_ptr();
        let mut stp = s_tape_f32.buf.as_ptr();
        let mut pp = parent_indices.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sip as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_tree_batch_seq",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gated_delta_net_f32_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sip);
                b.push_ptr(stp);
                b.push_ptr(pp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched-sequential FP32 GDN recurrence — full-precision, same 32×32-tile
    /// parallelism as `gated_delta_net_q8_batch_seq`. Use on the FP32
    /// `StateQuant` batched prefill/verify path instead of the slow
    /// 128-thread single-token `gated_delta_net`. State advanced in place.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f32_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_f32: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_f32_batch_seq",
            kernels::GATED_DELTA_NET_F32_BATCH_SEQ_SRC,
            "gated_delta_net_f32_batch_seq",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sp = s_f32.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_batch_seq",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gated_delta_net_f32_batch_seq",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Chunked (parallel) FP32 GDN — numerically EQUAL to
    /// `gated_delta_net_f32_batch_seq` (oracle gdn_chunked_f32). Same buffer
    /// args plus a `chunk_size` scalar. The cross-chunk dependency (chunk ci's
    /// S_in = chunk ci-1's S_out) is serialized HERE by a host loop over
    /// chunks: one launch per chunk, grid `[n_heads, 1, 1]`, `chunk_index`
    /// passed as a scalar. S advances in place in `s_f32` (same as batch_seq).
    /// `chunk_size` is clamped to [1, 32]; the kernel's static LDS is sized for
    /// CS_MAX=32.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f32_chunked(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_f32: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        chunk_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_f32_chunked",
            kernels::GATED_DELTA_NET_F32_CHUNKED_SRC,
            "gated_delta_net_f32_chunked",
        )?;

        let cs = chunk_size.clamp(1, 32);
        let n_chunks = n_tokens.div_ceil(cs);
        // Threads per workgroup (BLK). The chunked kernel is parameterized over
        // blockDim.x: 32 = 1 wave/4-rows-per-lane, 128 = 4 waves (latency
        // hiding + 1 HD row/lane), 256 = 8 waves. Default 128; override for
        // perf sweeps via HIPFIRE_GDN_BLK.
        let blk: u32 = hipfire_config::developer_var("HIPFIRE_GDN_BLK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|b| (b.clamp(32, 256) / 32) * 32)
            .unwrap_or(128);

        let qp = q_batch.buf.as_ptr();
        let kp = k_batch.buf.as_ptr();
        let vp = v_batch.buf.as_ptr();
        let gp = gate_batch.buf.as_ptr();
        let bp = beta_batch.buf.as_ptr();
        let sp = s_f32.buf.as_ptr();
        let op = output_batch.buf.as_ptr();

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_chunked",
            bytes,
        );

        // Host loop over chunks — cross-chunk is serial (S_in = prev S_out).
        for ci in 0..n_chunks {
            let mut qp = qp;
            let mut kp = kp;
            let mut vp = vp;
            let mut gp = gp;
            let mut bp = bp;
            let mut sp = sp;
            let mut op = op;
            let mut nt = n_tokens as i32;
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut cs_i = cs as i32;
            let mut ci_i = ci as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut gp as *mut _ as *mut c_void,
                &mut bp as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut nt as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut cs_i as *mut _ as *mut c_void,
                &mut ci_i as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "gated_delta_net_f32_chunked",
                [n_heads as u32, 1, 1],
                [blk, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp);
                    b.push_ptr(kp);
                    b.push_ptr(vp);
                    b.push_ptr(gp);
                    b.push_ptr(bp);
                    b.push_ptr(sp);
                    b.push_ptr(op);
                    b.push_i32(nt);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_i32(cs_i);
                    b.push_i32(ci_i);
                    b
                },
            )?;
        }

        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        Ok(())
    }

    /// GDN recurrence with Q4-quantized S state.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q4(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q4: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_delta_net_q4",
            kernels::GATED_DELTA_NET_Q4_SRC,
            "gated_delta_net_q4",
        )?;
        let func = &self.functions["gated_delta_net_q4"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = s_q4.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [128, 1, 1],
                0,
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

    /// Scale vector by constant: x[i] *= scale. Replaces 48µs CPU roundtrip.
    #[cfg(feature = "deltanet")]
    pub fn scale_f32(&mut self, x: &GpuTensor, scale: f32) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("scale_f32", kernels::SCALE_F32_SRC, "scale_f32")?;
        let func = &self.functions["scale_f32"];
        let n = x.numel();
        let mut xp = x.buf.as_ptr();
        let mut nv = n as i32;
        let mut sv = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "scale_f32", bytes);
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

    /// Fused conv1d (kernel_size=4) + SiLU decode.
    /// Batched sigmoid-scaled residual add: `y[t,:] += sigmoid(scalars[t]) * x[t,:]`.
    /// `y` and `x` are `[n × dim]` f32; `scalars` is `[n]` pre-sigmoid logits.
    /// Batched analog of `sigmoid_f32` + `scaled_add_inplace_gpu_scalar_f32` used
    /// to fold the Q8 shared-expert down output into the residual stream in the
    /// E8 MoE batched-prefill body.
    pub fn sigmoid_scaled_residual_add_batched_f32(
        &mut self,
        y: &GpuTensor,
        x: &GpuTensor,
        scalars: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "sigmoid_scaled_residual_add_batched",
            kernels::SIGMOID_SCALED_RESIDUAL_ADD_BATCHED_SRC,
            "sigmoid_scaled_residual_add_batched_f32",
        )?;
        let yp = y.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let sp = scalars.buf.as_ptr();
        let nv = n as i32;
        let dv = dim as i32;
        let total = (n * dim) as u32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (total + block - 1) / block;
        self.launch_maybe_blob(
            "sigmoid_scaled_residual_add_batched_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(xp);
                b.push_ptr(sp);
                b.push_i32(nv);
                b.push_i32(dv);
                b
            },
        )
    }

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

    /// gfx1100/gfx1201 decode-only conv+SiLU+Q/K normalization fusion. The five
    /// compile-time block shapes are intentionally separate kernels so the
    /// screen changes cooperative work distribution rather than only metadata.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_qknorm(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, kernel_src, block) =
            match hipfire_config::developer_var("HIPFIRE_CONV_QKNORM_SHAPE").ok().as_deref() {
                Some("b32") => (
                    "conv1d_silu_split_qknorm_b32",
                    kernels::CONV1D_SILU_SPLIT_QKNORM_B32_SRC,
                    32u32,
                ),
                Some("b64") => (
                    "conv1d_silu_split_qknorm_b64",
                    kernels::CONV1D_SILU_SPLIT_QKNORM_B64_SRC,
                    64u32,
                ),
                Some("b128") => (
                    "conv1d_silu_split_qknorm_b128",
                    kernels::CONV1D_SILU_SPLIT_QKNORM_B128_SRC,
                    128u32,
                ),
                Some("b512") => (
                    "conv1d_silu_split_qknorm_b512",
                    kernels::CONV1D_SILU_SPLIT_QKNORM_B512_SRC,
                    512u32,
                ),
                _ => (
                    "conv1d_silu_split_qknorm_b256",
                    kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
                    256u32,
                ),
            };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;

        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let heads_per_wg = if block >= 256 { block / 256 } else { 1 };
        let qk_blocks = (n_heads as u32 + heads_per_wg - 1) / heads_per_wg;
        let v_blocks = (v_dim as u32 + block - 1) / block;
        let grid = qk_blocks + v_blocks;
        let bytes = crate::profile::conv1d_silu_bytes(2 * k_dim + v_dim);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
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
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1100 decode experiment: use one extra workgroup in the existing
    /// conv/QK-normalization launch to prepare the independent DeltaNet beta
    /// and alpha scalars. This removes the standalone scalar launch without
    /// enlarging the hot QKVZA projection kernel.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_qknorm_scalar_prep_gfx1100(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        n_v_heads: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() || head_dim != 128 || n_v_heads > 256 {
            return Err(hip_bridge::HipError::new(
                0,
                "conv scalar-prep fusion requires gfx1100, head_dim=128, n_v_heads<=256",
            ));
        }

        const KERNEL: &str = "conv1d_silu_split_qknorm_b256_scalar_prep";
        const BLOCK: u32 = 256;
        self.ensure_kernel(
            KERNEL,
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SCALAR_PREP_SRC,
            KERNEL,
        )?;

        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let nvh = n_v_heads as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
            &nvh as *const _ as *mut c_void,
        ];
        let heads_per_wg = BLOCK / 256;
        let qk_blocks = (n_heads as u32 + heads_per_wg - 1) / heads_per_wg;
        let v_blocks = (v_dim as u32 + BLOCK - 1) / BLOCK;
        let grid = qk_blocks + v_blocks + 1;
        let bytes = crate::profile::conv1d_silu_bytes(2 * k_dim + v_dim)
            + n_v_heads * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", KERNEL, bytes);
        let result = self.launch_maybe_blob(KERNEL, [grid, 1, 1], [BLOCK, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(vp);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_ptr(sp);
            b.push_ptr(bp);
            b.push_ptr(ap);
            b.push_ptr(dp);
            b.push_ptr(lp);
            b.push_i32(kd);
            b.push_i32(vd);
            b.push_i32(nh);
            b.push_i32(hd);
            b.push_f32(qs);
            b.push_f32(ep);
            b.push_i32(nvh);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
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
        self.ensure_kernel(
            "conv1d_silu_split_tree",
            kernels::CONV1D_SILU_SPLIT_TREE_SRC,
            "conv1d_silu_split_tree_f32",
        )?;
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
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "conv1d_silu_split_tree_f32_n",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "conv1d_silu_split_tree_f32",
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

    /// Compute cross-entropy loss for a single token on GPU.
    /// Returns -log(softmax(logits)[target]). Downloads 4 bytes instead of 600KB.
    pub fn cross_entropy_loss(
        &mut self,
        logits: &GpuTensor,
        target_buf: &DeviceBuffer,
        loss_buf: &GpuTensor,
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "cross_entropy_loss",
            kernels::CROSS_ENTROPY_LOSS_SRC,
            "cross_entropy_loss",
        )?;
        let func = &self.functions["cross_entropy_loss"];
        let mut lp = logits.buf.as_ptr();
        let mut tp = target_buf.as_ptr();
        let mut op = loss_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = (block_size * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ═══ Vision encoder dispatch (GEMM, LayerNorm, GELU, bias-add) ═══

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

    /// Bias-add: x[batch, n] += bias[n] (in-place, broadcast over batch dim)
    pub fn bias_add_f32(
        &mut self,
        x: &GpuTensor,
        bias: &GpuTensor,
        batch: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("bias_add_f32", kernels::BIAS_ADD_SRC, "bias_add_f32")?;
        let func = &self.functions["bias_add_f32"];
        let xp = x.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ni = n as i32;
        let total = (batch * n) as i32;
        let ti = total;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
        ];
        let blocks = ((total as usize + 255) / 256) as u32;
        self.launch_maybe_blob(
            "bias_add_f32",
            [blocks, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(bp);
                b.push_i32(ni);
                b.push_i32(ti);
                b
            },
        )
    }

    /// Transpose [rows, cols] → [cols, rows]
    pub fn transpose_f32(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        rows: usize,
        cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("transpose_f32", kernels::TRANSPOSE_SRC, "transpose_f32")?;
        let func = &self.functions["transpose_f32"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ri = rows as i32;
        let mut ci = cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ri as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
        ];
        let total = rows * cols;
        let blocks = ((total + 255) / 256) as u32;
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

    /// f32 → f16 elementwise cast. `src` must be `DType::F32`, `dst`
    /// must be `DType::F16`, both with the same logical length. Single
    /// pass over the buffer; block [256], grid `ceil(n / 256)`.
    pub fn cast_f32_to_f16(&mut self, src: &GpuTensor, dst: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(src.dtype, DType::F32, "cast_f32_to_f16: src must be F32");
        assert_eq!(dst.dtype, DType::F16, "cast_f32_to_f16: dst must be F16");
        let n_src: usize = src.shape.iter().product();
        let n_dst: usize = dst.shape.iter().product();
        assert_eq!(
            n_src, n_dst,
            "cast_f32_to_f16: src and dst element counts must match (src={n_src}, dst={n_dst})",
        );
        self.ensure_kernel(
            "cast_f32_to_f16",
            kernels::CAST_F32_TO_F16_SRC,
            "cast_f32_to_f16",
        )?;
        let func = &self.functions["cast_f32_to_f16"];
        let mut in_ptr = src.buf.as_ptr();
        let mut out_ptr = dst.buf.as_ptr();
        let mut n_val = n_src as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let grid = ((n_src + 255) / 256) as u32;
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

    pub fn deepseek4_convert_f32_to_f16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n: i64,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_convert_f32_to_f16",
            kernels::V4F_CONVERT_F32_TO_F16_SRC,
            "deepseek4_convert_f32_to_f16",
        )?;
        let func = &self.functions["deepseek4_convert_f32_to_f16"];
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let mut nn = n;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let n_wgs = ((n + 127) / 128) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_wgs, 1, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn fused_rmsnorm_rotate_mq_plain(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        x_plain: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_plain",
            kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC,
            "fused_rmsnorm_mq_rotate_plain",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xpp = x_plain.buf.as_ptr();
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
            &xpp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = k * 4 * 4 + 2 * 256 * 4; // +1 K*4 for x_plain write
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_plain", bytes);
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_plain",
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
                b.push_ptr(xpp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.scratch.invalidate_x_caches_for(xrp);
        self.scratch.invalidate_x_caches_for(xpp);
        result
    }
    pub fn fused_rmsnorm_rotate_mq_plain_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        x_plain: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_plain",
            kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC,
            "fused_rmsnorm_mq_rotate_plain",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut xpp = x_plain.buf.as_ptr();
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
            &mut xpp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate_plain_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_plain",
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
                b.push_ptr(xpp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.scratch.invalidate_x_caches_for(xrp);
        self.scratch.invalidate_x_caches_for(xpp);
        result
    }
    pub fn rmsnorm_f32_at_slot_buf(
        &mut self,
        base: &GpuTensor,
        weight: &GpuTensor,
        slot_buf: &GpuTensor,
        n: i32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rmsnorm_f32_at_slot_buf",
            kernels::RMSNORM_AT_SLOT_BUF_SRC,
            "rmsnorm_f32_at_slot_buf",
        )?;
        let bp = base.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut nv = n;
        let mut ev = eps;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut ev as *mut _ as *mut c_void,
        ];
        let block = 256u32.min(n as u32).next_power_of_two().max(32);
        let shared = block * 4;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(bp);
            b.push_ptr(wp);
            b.push_ptr(sb);
            b.push_i32(nv);
            b.push_f32(ev);
            b
        };
        self.launch_maybe_blob(
            "rmsnorm_f32_at_slot_buf",
            [1, 1, 1],
            [block, 1, 1],
            shared,
            &mut params,
            blob_builder,
        )
    }
    pub fn sqrt_softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "sqrt_softplus_f32",
            kernels::SQRT_SOFTPLUS_F32_SRC,
            "sqrt_softplus_f32",
        )?;
        let func = &self.functions["sqrt_softplus_f32"];
        let n = x.numel() as i32;
        let xp = x.buf.as_ptr();
        let mut nv = n;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let grid_x = ((n + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_fused_silu_mul_clamp_mq_rotate(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            kernels::V4F_FUSED_SILU_MUL_CLAMP_MQ_ROTATE_SRC,
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let lim = swiglu_limit;
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &lim as *const _ as *mut c_void,
        ];
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(lim);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.scratch.invalidate_x_caches_for(xrp);
        result
    }
    pub fn deepseek4_silu_mul_clamp_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_silu_mul_clamp",
            kernels::V4F_SILU_MUL_CLAMP_SRC,
            "deepseek4_silu_mul_clamp_f32",
        )?;

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;
        let mut limit_val = swiglu_limit;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut limit_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deepseek4_silu_mul_clamp_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "deepseek4_silu_mul_clamp_f32",
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
                b.push_f32(limit_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn deepseek4_silu_mul_clamp_f32_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        batch: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_silu_mul_clamp",
            kernels::V4F_SILU_MUL_CLAMP_SRC,
            "deepseek4_silu_mul_clamp_f32",
        )?;

        let n_i32 = n as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n_i32;
        let mut limit_val = swiglu_limit;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut limit_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n_i32 as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n) * batch;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deepseek4_silu_mul_clamp_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "deepseek4_silu_mul_clamp_f32",
            [grid, batch as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gate_ptr);
                b.push_ptr(up_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(limit_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
