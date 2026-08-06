// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
use crate::dispatch::{DType, Gpu, GpuTensor, FP8_GEMV_MIN_M};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

static GFX942_ROTATE_LIVE_VALIDATED: AtomicBool = AtomicBool::new(false);

fn gfx942_rotate_live_validation_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_GFX942_ROTATE_VALIDATE_LIVE")
        .ok()
        .as_deref()
        == Some("1")
}

fn validate_mq_rotate_live(input: &[f32], output: &[f32], k: usize, batch: usize) {
    let signs1 = crate::dispatch::gen_fwht_signs(42, 256);
    let signs2 = crate::dispatch::gen_fwht_signs(1042, 256);
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for row in 0..batch {
        for group in 0..k / 256 {
            let offset = row * k + group * 256;
            let mut values = [0.0f32; 256];
            for i in 0..256 {
                values[i] = input[offset + i] * signs1[i];
            }
            let mut stride = 1;
            while stride < 256 {
                for base in (0..256).step_by(stride * 2) {
                    for lane in 0..stride {
                        let a = values[base + lane];
                        let b = values[base + lane + stride];
                        values[base + lane] = a + b;
                        values[base + lane + stride] = a - b;
                    }
                }
                stride *= 2;
            }
            for i in 0..256 {
                let expected = values[i] * 0.0625 * signs2[i];
                let actual = output[offset + i];
                let abs = (actual - expected).abs();
                max_abs = max_abs.max(abs);
                if actual.to_bits() != expected.to_bits() {
                    mismatches += 1;
                    first.get_or_insert((offset + i, actual, expected));
                }
            }
        }
    }
    eprintln!(
        "[gfx942 rotate live oracle] k={k} batch={batch} mismatches={mismatches}/{} max_abs={max_abs:.8e} first={first:?}",
        k * batch
    );
}

/// DIAGNOSTIC: when HIPFIRE_E8_STRIP=1, gemv_mfp4g32_e8 (gfx1151) launches the
/// compute-stripped kernel instead of the real decode kernel — for measuring
/// the memory-vs-compute bound (output is garbage). Cached once.
fn e8_strip_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_STRIP")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// EXPERIMENT: HIPFIRE_E8_LDSX=1 routes gemv_mfp4g32_e8 (gfx1151) to the
/// LDS-staged-x + 4-rows/block variant (memory-level-parallelism lever).
fn e8_ldsx_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_LDSX")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// HIPFIRE_E8_DGPU_TWIN: on RDNA3 dGPU (gfx1100/1101/1102), route E8 MoE
/// GEMVs to the 4-way-unroll gfx11_dgpu twin rather than the gfx1151 kernel.
///
/// Default OFF. Measured on gfx1100 (RX 7900 XTX) 2026-06-13: WASH — twin 101.6
/// vs baseline 102.1 tok/s (-0.5%, within ±1-3% noise), coherent, 0 spill,
/// gfx1151 byte-untouched. This is a DIAGNOSTIC, not just a null result: dropping
/// occupancy 16→10 waves (62→94 VGPR) to buy 4-way ILP changed decode by ~0%, which
/// proves E8 decode is NOT memory-latency-bound on this dGPU — the 96MB Infinity
/// Cache keeps expert weights hot, so the baseline 16-wave occupancy already hides
/// the (cache-resident) latency. Therefore load-scheduling levers (unroll, prefetch,
/// multi-row) cannot move it. The real 102→150 gap vs uniform-mq4 (hfq4 *g256*) is
/// the E8 *g32* format: a UE8M0 block scale every 32 elements vs mq4's every 256 =
/// ~8x more scale bytes + ldexp per weight — extra memory traffic + dequant that is
/// the inherent PRICE of E8's MQ6-class quality, addressable only at the format level
/// (coarser groups / compacter scales), not by any decode kernel schedule.
/// Twins preserved for future experiments (larger models, >1 token/step prefill
/// batching) where the cache spills to GDDR6 and latency-hiding may pay off.
/// Set =1 to route through the 4-way-unroll twin for A/B.
fn e8_dgpu_twin_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_DGPU_TWIN")
            .map(|v| v == "1")
            .unwrap_or(false) // default OFF — delta <5% on gfx1100 (see comment)
    })
}

/// HIPFIRE_E8_SOA_EXPERTS: route E8 MoE gate_up decode to the SoA-coalesced kernel
/// (reads SoA-laid-out expert weights). MUST be consistent with the load path: when
/// set, routed E8 gate_up experts are transposed AoS->SoA at load, and the dispatch
/// uses the SoA kernel. Read once (cached) — never per launch.
///
/// Default OFF. Validated COHERENT on gfx1100 (q36a3b.mfp4e8-gptq-v2) — kernel +
/// transpose-on-load correct. But A3B decode = WASH: 102.0 (SoA) vs 102.0 (AoS) tok/s.
/// The standalone GEMV bench's +38-73% was on LARGE DENSE shapes (M=11008); A3B's
/// per-expert gate_up is small (M=2*moe_intermediate ~1536), which sits in the bench's
/// WASH regime (cf. qkv-kv M=512). So SoA coalescing helps big dense GEMVs, not the
/// small per-expert MoE GEMVs. Far below the +8% ship bar — kept opt-in + documented.
/// (Increment-1 yes/no per docs/plans/e8-soa-indexed-moe-decode.md: answer = no win.)
pub(crate) fn e8_soa_experts_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_SOA_EXPERTS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

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

    /// `fused_qkv_mq4g256_lloyd` with an optional Q/K/V bias folded into the
    /// kernel's lane-0 store (`HIPFIRE_FUSE_QKV_BIAS`). All-null = byte-identical
    /// to the unfused path (fp32 store→add). Both arch siblings (base + gfx1100)
    /// carry the 3 trailing bias params, so the kernarg ABI matches whichever
    /// `_for_arch` selects.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_mq4g256_lloyd_with_bias(
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
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq4g256_lloyd_qwen2_bias_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq4g256_lloyd_qwen2_bias")?;
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
        let bq = bias_q_ptr;
        let bk = bias_k_ptr;
        let bv = bias_v_ptr;
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
            &bq as *const _ as *mut c_void,
            &bk as *const _ as *mut c_void,
            &bv as *const _ as *mut c_void,
        ];
        let total = (q_m + k_m + v_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq4g256_lloyd", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_mq4g256_lloyd_qwen2_bias",
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
                b.push_ptr(bq);
                b.push_ptr(bk);
                b.push_ptr(bv);
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
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4);
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

    /// `fused_qkv_mq3g256_lloyd` with an optional Q/K/V bias folded into the
    /// kernel's lane-0 store (`HIPFIRE_FUSE_QKV_BIAS`). All-null = byte-identical
    /// to the unfused path. Both arch siblings (base + gfx1100) carry the 3
    /// trailing bias params.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_mq3g256_lloyd_with_bias(
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
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq3g256_lloyd_qwen2_bias_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq3g256_lloyd_qwen2_bias")?;
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
        let bq = bias_q_ptr;
        let bk = bias_k_ptr;
        let bv = bias_v_ptr;
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
            &bq as *const _ as *mut c_void,
            &bk as *const _ as *mut c_void,
            &bv as *const _ as *mut c_void,
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
            "fused_qkv_mq3g256_lloyd_qwen2_bias",
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
                b.push_ptr(bq);
                b.push_ptr(bk);
                b.push_ptr(bv);
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
        // on prefill scale; modest decode change. It is mathematically
        // equivalent, but its reduction order is not byte-identical.
        if self.flags.gfx942_rmsnorm_split {
            return self.fused_rmsnorm_rotate_mq_split_gfx942(x, weight, x_rot, k, eps, 1);
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_split && k == 2048 {
            return self.fused_rmsnorm_rotate_mq_split_gfx1100(x, weight, x_rot, k, eps);
        }
        if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_wavegrid && k == 2048 {
            return self.fused_rmsnorm_rotate_mq_wavegrid_gfx1100(x, weight, x_rot, k, eps);
        }
        let gfx1151_radiowave_fusions = self.arch_caps.is_gfx1151();
        let vecsum = k == 2048
            && ((self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_vecsum)
                || gfx1151_radiowave_fusions);
        let sign_lds = vecsum && self.flags.rdna3_rmsnorm_sign_lds;
        let sign_const = vecsum && self.flags.rdna3_rmsnorm_sign_const;
        let (kernel, source) = if sign_const {
            (
                "fused_rmsnorm_mq_rotate_vecsum_sign_const",
                kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_SIGN_CONST_GFX1100_SRC,
            )
        } else if sign_lds {
            (
                "fused_rmsnorm_mq_rotate_vecsum_sign_lds",
                kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_SIGN_LDS_GFX1100_SRC,
            )
        } else if vecsum {
            (
                "fused_rmsnorm_mq_rotate_vecsum",
                kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
            )
        } else {
            (
                "fused_rmsnorm_mq_rotate",
                kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            )
        };
        self.ensure_kernel(kernel, source, kernel)?;
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
        // The current kernel uses only reduce[256]. Keep the historical
        // K-float x_shared reservation as the A/B control; x_shared itself
        // disappeared when the first-group prefetch schedule landed.
        let shared_mem = if vecsum {
            0
        } else if self.flags.rmsnorm_mq_tight_lds {
            (256 * 4) as u32
        } else {
            ((k + 256) * 4) as u32
        };

        // Bandwidth: read x (K*4) + weight (K*4) + signs (2*256*4) + write x_rot (K*4)
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
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

    /// gfx1100 experiment: an exact one-block RMS reduction followed by eight
    /// independent wave32 FWHT workgroups. This preserves the fused kernel's
    /// arithmetic assignment without its cross-workgroup rendezvous.
    fn fused_rmsnorm_rotate_mq_split_gfx1100(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        const REDUCE: &str = "rmsnorm_reduce_gfx1100";
        const ROTATE: &str = "rotate_with_rms_gfx1100";
        self.scratch
            .ensure_mq_rmsnorm_wavegrid_scratch(&self.hip, self.device_id)?;
        self.ensure_kernel(REDUCE, kernels::RMSNORM_REDUCE_GFX1100_SRC, REDUCE)?;
        self.ensure_kernel(ROTATE, kernels::ROTATE_WITH_RMS_GFX1100_SRC, ROTATE)?;

        let xp = x.buf.as_ptr();
        let rms = self
            .scratch
            .mq_rmsnorm_wavegrid_scratch
            .as_ref()
            .unwrap()
            .as_ptr();
        let kv = k as i32;
        let eps_v = eps;
        let mut reduce_params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &rms as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];
        let reduce_timer = crate::profile::begin_timer(&self.hip, "fused", REDUCE, k * 4);
        self.launch_maybe_blob(
            REDUCE,
            [1, 1, 1],
            [256, 1, 1],
            0,
            &mut reduce_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(rms);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        )?;
        if let Some(t) = reduce_timer {
            t.finish(&self.hip);
        }

        let wp = weight.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let mut rotate_params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &rms as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let rotate_timer = crate::profile::begin_timer(&self.hip, "fused", ROTATE, bytes);
        let result = self.launch_maybe_blob(
            ROTATE,
            [(k / 256) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut rotate_params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(rms);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = rotate_timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// gfx1100 experiment: map the baseline block's eight independent wave32s
    /// onto eight workgroups so the reduction/FWHT can occupy eight CUs. The
    /// kernel preserves the baseline thread assignment and reduction tree.
    fn fused_rmsnorm_rotate_mq_wavegrid_gfx1100(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        const KERNEL: &str = "fused_rmsnorm_mq_rotate_wavegrid";
        self.scratch
            .ensure_mq_rmsnorm_wavegrid_scratch(&self.hip, self.device_id)?;
        self.ensure_kernel(
            KERNEL,
            kernels::FUSED_RMSNORM_MQ_ROTATE_WAVEGRID_GFX1100_SRC,
            KERNEL,
        )?;

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let scratch = self
            .scratch
            .mq_rmsnorm_wavegrid_scratch
            .as_ref()
            .unwrap()
            .as_ptr();
        let kv = k as i32;
        let eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &scratch as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", KERNEL, bytes);
        let result = self.launch_maybe_blob(KERNEL, [8, 1, 1], [32, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(xp);
            b.push_ptr(wp);
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_ptr(xrp);
            b.push_ptr(scratch);
            b.push_i32(kv);
            b.push_f32(eps_v);
            b
        });
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
            // kernel is __launch_bounds__(64) — 1 wave64 (lanes 0..31 work).
            // Launching 128 violates the bound → hipErrorLaunchFailure (719) on gfx942.
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

    /// Route A MoE-AWQ — per-routed-expert variant of
    /// `fused_silu_mul_rotate_mq_awq_batched`. Each batch row (routed-expert
    /// slot) selects its own down `awq_scale` from a device pointer table,
    /// indexed by `topk_indices[slot]`. Serves both decode (`batch_size` =
    /// k_top, `topk_indices` = `[k_top]`) and prefill (`batch_size` =
    /// total_slots = N·k_top, `topk_indices` = `[N·k_top]`) — the silu+rotate
    /// is elementwise per slot, so the same slot→expert mapping the indexed
    /// down GEMV uses applies here.
    ///
    /// `expert_down_awq_ptrs`: `[2·n_exp]` F32 = n_exp `u64` device pointers,
    /// each → that expert's `[K]` F32 down awq_scale (built by the loader, see
    /// qwen35.rs::load_moe_ffn). On `.hfq` files without per-expert sidecars
    /// the caller passes the plain `fused_silu_mul_rotate_mq_batched` instead.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_silu_mul_rotate_mq_awq_indexed_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        expert_down_awq_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate_awq_indexed",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_INDEXED_SRC,
            "fused_silu_mul_mq_rotate_awq_indexed",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let pp = expert_down_awq_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        // Read gate + up, 2×256 signs, write x_rot (awq_scale read is per-slot
        // strided — folded into the gate/up term for the bandwidth estimate).
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_mq_rotate_awq_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_silu_mul_mq_rotate_awq_indexed",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(up_p);
                b.push_ptr(pp);
                b.push_ptr(ip);
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
    pub(crate) fn invalidate_x_caches_for(&mut self, dst_ptr: *mut c_void) {
        self.scratch.invalidate_x_caches_for(dst_ptr)
    }

    /// gfx1100 DeltaNet decode specialization: reproduce two independent
    /// head-wise gated norms per workgroup, then consume their 256 normalized
    /// values directly from LDS for the MQ rotation.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_norm_rotate_mq_gfx1100(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let k = n_heads.checked_mul(head_dim).ok_or_else(|| {
            hip_bridge::HipError::new(1, "gated_norm_rotate_mq_gfx1100: size overflow")
        })?;
        if n_heads != 32 || head_dim != 128 {
            return Err(hip_bridge::HipError::new(
                1,
                "gated_norm_rotate_mq_gfx1100: expected 32 heads with head_dim=128",
            ));
        }
        if x.numel() < k || z.numel() < k || weight.numel() < head_dim || x_rot.numel() < k {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "gated_norm_rotate_mq_gfx1100: undersized tensor (x={}, z={}, weight={}, x_rot={}, required x/z/x_rot={}, weight={})",
                    x.numel(),
                    z.numel(),
                    weight.numel(),
                    x_rot.numel(),
                    k,
                    head_dim,
                ),
            ));
        }
        self.ensure_mq_signs()?;
        let (module, src, kernel) = if self.arch_caps.is_gfx1151() {
            (
                "gated_norm_mq_rotate_gfx1151",
                kernels::GATED_NORM_MQ_ROTATE_GFX1151_SRC,
                "gated_norm_mq_rotate_gfx1151",
            )
        } else {
            (
                "gated_norm_mq_rotate_gfx1100",
                kernels::GATED_NORM_MQ_ROTATE_GFX1100_SRC,
                "gated_norm_mq_rotate_gfx1100",
            )
        };
        self.ensure_kernel(module, src, kernel)?;

        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(k) + crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [(n_heads / 2) as u32, 1, 1],
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
                b.push_ptr(xrp);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    pub fn rotate_x_mq(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let validate_live = self.arch == "gfx942"
            && gfx942_rotate_live_validation_enabled()
            && !GFX942_ROTATE_LIVE_VALIDATED.swap(true, Ordering::Relaxed);
        let validation_input = if validate_live {
            self.hip.device_synchronize()?;
            Some(self.download_f32(x)?)
        } else {
            None
        };
        let (kernel, source) = if self.arch == "gfx942" {
            ("mq_rotate_x_gfx942", kernels::MQ_ROTATE_X_GFX942_SRC)
        } else {
            ("mq_rotate_x", kernels::GEMV_MQ4G256_SRC)
        };
        self.ensure_kernel(kernel, source, kernel)?;
        self.ensure_mq_signs()?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x", bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [(k / 256) as u32, 1, 1],
            [if self.arch == "gfx942" { 64 } else { 32 }, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        if let Some(input) = validation_input {
            self.hip.device_synchronize()?;
            let output = self.download_f32(x_rot)?;
            validate_mq_rotate_live(&input, &output, k, 1);
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
        let validate_live = self.arch == "gfx942"
            && gfx942_rotate_live_validation_enabled()
            && !GFX942_ROTATE_LIVE_VALIDATED.swap(true, Ordering::Relaxed);
        let validation_input = if validate_live {
            self.hip.device_synchronize()?;
            Some(self.download_f32(x)?)
        } else {
            None
        };
        let (kernel, source) = if self.arch == "gfx942" {
            ("mq_rotate_x_gfx942", kernels::MQ_ROTATE_X_GFX942_SRC)
        } else {
            ("mq_rotate_x", kernels::GEMV_MQ4G256_SRC)
        };
        self.ensure_kernel(kernel, source, kernel)?;
        self.ensure_mq_signs()?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [((k / 256) * batch_size) as u32, 1, 1],
            [if self.arch == "gfx942" { 64 } else { 32 }, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        if let Some(input) = validation_input {
            self.hip.device_synchronize()?;
            let output = self.download_f32(x_rot)?;
            validate_mq_rotate_live(&input, &output, k, batch_size);
        }
        self.invalidate_x_caches_for(xrp);
        result
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

    /// MFP4G32-Lloyd GEMV. Uses the per-tensor 16-entry Lloyd codebook stored
    /// as a 32-B prefix before row 0. x must already be FWHT-rotated.
    pub fn gemv_mfp4g32_lloyd(
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
            "gemv_mfp4g32_lloyd requires K%256==0, got K={k}"
        );
        self.ensure_kernel(
            "gemv_mfp4g32_lloyd",
            kernels::GEMV_MFP4G32_LLOYD_SRC,
            "gemv_mfp4g32_lloyd",
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
        unsafe {
            self.hip.launch_kernel(
                &self.functions["gemv_mfp4g32_lloyd"],
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// MFP4G32-Lloyd prerotated (x already FWHT-rotated by caller).
    pub fn gemv_mfp4g32_lloyd_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_mfp4g32_lloyd(a_raw, x_rot, y, m, k)
    }

    /// mfp4+P GEMV. mfp4 byte layout (NO prefix) but the per-32-block scale byte
    /// is E4M3 (FP8, non-power-of-2). x must already be FWHT-rotated. Uses the
    /// hard-coded E2M1 lattice (like gemv_hfp4g32), decoding the scale as E4M3.
    pub fn gemv_mfp4g32_p(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k % 256 == 0, "gemv_mfp4g32_p requires K%256==0, got K={k}");
        self.ensure_kernel(
            "gemv_mfp4g32_p",
            kernels::GEMV_MFP4G32_P_SRC,
            "gemv_mfp4g32_p",
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
        unsafe {
            self.hip.launch_kernel(
                &self.functions["gemv_mfp4g32_p"],
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// mfp4+P prerotated (x already FWHT-rotated by caller).
    pub fn gemv_mfp4g32_p_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_mfp4g32_p(a_raw, x_rot, y, m, k)
    }

    /// mfp4-E8 GEMV. mfp4+P byte layout (NO prefix) but the 16-B data region per block
    /// contains 4x32-bit E8 lattice codewords (8 weights/codeword), not E2M1 nibbles.
    /// x must already be FWHT-rotated. Decodes block scale as E4M3, then E8 coords * 0.88.
    pub fn gemv_mfp4g32_e8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k % 256 == 0, "gemv_mfp4g32_e8 requires K%256==0, got K={k}");

        // gfx1151 (Strix Halo): 2-rows-per-block variant for better occupancy.
        // 2 wave32 subgroups per block hide LPDDR5X latency better than 1 wave/block.
        // All other archs use the generic kernel (1 row/block, 32 threads) unchanged.
        if self.arch_caps.is_gfx1151() {
            // EXPERIMENT (HIPFIRE_E8_LDSX=1): LDS-staged x + 4 rows/block. Distinct
            // launch geometry (grid=ceil(M/4), block=128, dynamic LDS=K*4 bytes).
            if e8_ldsx_enabled() {
                self.ensure_kernel(
                    "gemv_mfp4g32_e8_ldsx_gfx1151",
                    kernels::GEMV_MFP4G32_E8_LDSX_GFX1151_SRC,
                    "gemv_mfp4g32_e8_ldsx_gfx1151",
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
                let grid_x = ((m + 3) / 4) as u32;
                let lds_bytes = (k * 4) as u32;
                return unsafe {
                    self.hip.launch_kernel(
                        &self.functions["gemv_mfp4g32_e8_ldsx_gfx1151"],
                        [grid_x, 1, 1],
                        [128, 1, 1],
                        lds_bytes,
                        self.stream_ref(),
                        &mut params,
                    )
                };
            }
            // DIAGNOSTIC (HIPFIRE_E8_STRIP=1): swap in the compute-stripped kernel
            // (same memory access, gutted decode) to measure the compute ceiling.
            // Output is garbage — perf-probe only.
            let (kname, ksrc) = if e8_strip_enabled() {
                (
                    "gemv_mfp4g32_e8_strip_gfx1151",
                    kernels::GEMV_MFP4G32_E8_STRIP_GFX1151_SRC,
                )
            } else {
                (
                    "gemv_mfp4g32_e8_gfx1151",
                    kernels::GEMV_MFP4G32_E8_GFX1151_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
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
            let grid_x = m as u32;
            return unsafe {
                self.hip.launch_kernel(
                    &self.functions[kname],
                    [grid_x, 1, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        self.ensure_kernel(
            "gemv_mfp4g32_e8",
            kernels::GEMV_MFP4G32_E8_SRC,
            "gemv_mfp4g32_e8",
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
        unsafe {
            self.hip.launch_kernel(
                &self.functions["gemv_mfp4g32_e8"],
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// mfp4-E8 prerotated (x already FWHT-rotated by caller).
    pub fn gemv_mfp4g32_e8_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_mfp4g32_e8(a_raw, x_rot, y, m, k)
    }

    /// mfp3-E8 GEMV. The caller supplies an already FWHT-rotated F32
    /// activation. Each row is a 16-byte header followed by K/32 13-byte
    /// blocks (one E4M3 scale and four 24-bit E8 codewords).
    pub fn gemv_mfp3g32_e8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k % 256 == 0, "gemv_mfp3g32_e8 requires K%256==0, got K={k}");
        const KERNEL: &str = "gemv_mfp3g32_e8";
        self.ensure_kernel(KERNEL, kernels::GEMV_MFP3G32_E8_SRC, KERNEL)?;
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
        self.launch_maybe_blob(KERNEL, [m as u32, 1, 1], [32, 1, 1], 0, &mut params, || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(a_ptr);
            blob.push_ptr(x_ptr);
            blob.push_ptr(y_ptr);
            blob.push_i32(m_val);
            blob.push_i32(k_val);
            blob
        })
    }

    /// mfp3-E8 prerotated entry point.
    pub fn gemv_mfp3g32_e8_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_mfp3g32_e8(a_raw, x_rot, y, m, k)
    }

    /// Fused gate+up mfp4-E8 decode GEMV — gfx1151 ONLY (Strix Halo).
    /// Two GEMVs (gate, up) in one launch over [gate_m + up_m] blocks. x must
    /// already be FWHT-rotated (the execute_steps RmsnormAutomatic producer does
    /// it). Bit-exact with two sequential gemv_mfp4g32_e8 calls — only the launch
    /// count shrinks (the gfx1151 launch-fusion lever). Reached ONLY via the
    /// gfx1151-gated guard in pipeline/steps.rs, so it never runs on other archs.
    pub fn fused_gate_up_mfp4g32_e8(
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
        assert!(
            k % 256 == 0,
            "fused_gate_up_mfp4g32_e8 requires K%256==0, got K={k}"
        );
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "fused_gate_up_mfp4g32_e8 is gfx1151-only (guard must firewall the arch)"
        );
        self.ensure_kernel(
            "fused_gate_up_mfp4g32_e8_gfx1151",
            kernels::FUSED_GATE_UP_MFP4G32_E8_GFX1151_SRC,
            "fused_gate_up_mfp4g32_e8_gfx1151",
        )?;
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
        unsafe {
            self.hip.launch_kernel(
                &self.functions["fused_gate_up_mfp4g32_e8_gfx1151"],
                [total, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused QKVZA mfp4-E8 decode GEMV — gfx1151 ONLY (Strix Halo).
    /// Four DeltaNet LA-preamble GEMVs (qkv, z, beta, alpha) in one launch. x
    /// must already be FWHT-rotated. Bit-exact with four sequential
    /// gemv_mfp4g32_e8 calls. gfx1151-gated via the steps.rs guard.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_mfp4g32_e8(
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
        assert!(
            k % 256 == 0,
            "fused_qkvza_mfp4g32_e8 requires K%256==0, got K={k}"
        );
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "fused_qkvza_mfp4g32_e8 is gfx1151-only (guard must firewall the arch)"
        );
        self.ensure_kernel(
            "fused_qkvza_mfp4g32_e8_gfx1151",
            kernels::FUSED_QKVZA_MFP4G32_E8_GFX1151_SRC,
            "fused_qkvza_mfp4g32_e8_gfx1151",
        )?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let mqkv = qkv_m as i32;
        let mz = z_m as i32;
        let mb = beta_m as i32;
        let ma = alpha_m as i32;
        let kv = k as i32;
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
            &mqkv as *const _ as *mut c_void,
            &mz as *const _ as *mut c_void,
            &mb as *const _ as *mut c_void,
            &ma as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        unsafe {
            self.hip.launch_kernel(
                &self.functions["fused_qkvza_mfp4g32_e8_gfx1151"],
                [total, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// mfp4-E8 SoA GEMV. Same E8 data as AoS but in structure-of-arrays layout
    /// (flag=0x06) for fully-coalesced codeword reads on gfx1151.
    /// x must already be FWHT-rotated. Output is bit-exact with gemv_mfp4g32_e8.
    pub fn gemv_mfp4g32_e8_soa(
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
            "gemv_mfp4g32_e8_soa requires K%256==0, got K={k}"
        );

        if self.arch_caps.is_gfx942() {
            const KERNEL: &str = "gemv_mfp4g32_e8_soa_gfx942";
            self.ensure_kernel(KERNEL, kernels::GEMV_MFP4G32_E8_SOA_GFX942_SRC, KERNEL)?;
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
            return self.launch_maybe_blob(
                KERNEL,
                [m.div_ceil(2) as u32, 1, 1],
                [64, 1, 1],
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
        }

        if self.arch_caps.is_gfx1151() {
            const KERNEL: &str = "gemv_mfp4g32_e8_soa_gfx1151";
            self.ensure_kernel(KERNEL, kernels::GEMV_MFP4G32_E8_SOA_GFX1151_SRC, KERNEL)?;
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
            return unsafe {
                self.hip.launch_kernel(
                    &self.functions[KERNEL],
                    [m as u32, 1, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        // Generic fallback for non-gfx1151
        self.ensure_kernel(
            "gemv_mfp4g32_e8_soa",
            kernels::GEMV_MFP4G32_E8_SOA_SRC,
            "gemv_mfp4g32_e8_soa",
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
        unsafe {
            self.hip.launch_kernel(
                &self.functions["gemv_mfp4g32_e8_soa"],
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Experimental gfx942 two-wave workgroup E8-SoA GEMV.
    ///
    /// This is an explicit micro-screen surface. Product dispatch continues
    /// to use `gemv_mfp4g32_e8_soa`; no architecture default selects w128.
    pub fn gemv_mfp4g32_e8_soa_w128_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx942(),
            "two-wave E8 kernel requires gfx942"
        );
        assert!(
            k % 256 == 0,
            "two-wave E8 kernel requires K%256==0, got K={k}"
        );
        const KERNEL: &str = "gemv_mfp4g32_e8_soa_w128_gfx942";
        self.ensure_kernel(KERNEL, kernels::GEMV_MFP4G32_E8_SOA_W128_GFX942_SRC, KERNEL)?;
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
            KERNEL,
            [m.div_ceil(4) as u32, 1, 1],
            [128, 1, 1],
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

    /// Experimental fixed two-job gfx942 E8-SoA GEMV.
    ///
    /// Both jobs consume the same prerotated activation but use independent
    /// weight and output allocations. This method is an explicit micro-screen
    /// surface; no product dispatch selects the pair kernel.
    pub fn gemv_mfp4g32_e8_soa_pair_gfx942(
        &mut self,
        a0: &GpuTensor,
        a1: &GpuTensor,
        x: &GpuTensor,
        y0: &GpuTensor,
        y1: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx942(),
            "two-job E8 kernel requires gfx942"
        );
        assert!(
            k % 256 == 0,
            "two-job E8 kernel requires K%256==0, got K={k}"
        );
        const KERNEL: &str = "gemv_mfp4g32_e8_soa_pair_gfx942";
        self.ensure_kernel(KERNEL, kernels::GEMV_MFP4G32_E8_SOA_PAIR_GFX942_SRC, KERNEL)?;
        let a0_ptr = a0.buf.as_ptr();
        let a1_ptr = a1.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y0_ptr = y0.buf.as_ptr();
        let y1_ptr = y1.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a0_ptr as *const _ as *mut c_void,
            &a1_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y0_ptr as *const _ as *mut c_void,
            &y1_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m.div_ceil(2) as u32, 2, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a0_ptr);
                blob.push_ptr(a1_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y0_ptr);
                blob.push_ptr(y1_ptr);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob
            },
        )
    }

    /// Generic SoA E8 GEMV variant launcher (bench experiments). Same grid/block/
    /// args as gemv_mfp4g32_e8_soa; only the kernel name + source differ.
    pub fn gemv_mfp4g32_e8_soa_variant(
        &mut self,
        kname: &'static str,
        src: &'static str,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k % 256 == 0, "{kname} requires K%256==0, got K={k}");
        self.ensure_kernel(kname, src, kname)?;
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
        self.launch_maybe_blob(kname, [m as u32, 1, 1], [32, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(x_ptr);
            b.push_ptr(y_ptr);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        })
    }

    /// Experimental gfx942 FP8-MFMA MFP4-E8 decode path. This is a micro-screen
    /// surface, not product dispatch: activation rounding changes arithmetic.
    pub fn gemv_mfp4g32_e8_soa_fp8_mfma_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx942(),
            "FP8 MFMA E8 kernel requires gfx942"
        );
        assert!(
            k % 256 == 0,
            "FP8 MFMA E8 kernel requires K%256==0, got K={k}"
        );
        const KERNEL: &str = "gemv_mfp4g32_e8_soa_fp8_mfma_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMV_MFP4G32_E8_SOA_FP8_MFMA_GFX942_SRC,
            KERNEL,
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m.div_ceil(16) as u32, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// One-launch gfx1151 E8-SoA block-diagonal GEMV:
    /// `A[G,M,K] @ x[G,K] -> y[G,M]`.
    /// Grouped E8 GEMV with buffer-SRD weight loads + gfx12 cache policy.
    /// Bit-exact with `gemv_mfp4g32_e8_soa_grouped_gfx1151`; only the weight
    /// fetch path differs. See
    /// `kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BUFFER_GFX1151_SRC`.
    pub fn gemv_mfp4g32_e8_soa_grouped_buffer_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_mfp4g32_e8_soa_grouped_impl(
            "gemv_mfp4g32_e8_soa_grouped_buffer_gfx1151",
            kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BUFFER_GFX1151_SRC,
            a,
            x,
            y,
            groups,
            m,
            k,
        )
    }

    pub fn gemv_mfp4g32_e8_soa_grouped_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_mfp4g32_e8_soa_grouped_impl(
            "gemv_mfp4g32_e8_soa_grouped_gfx1151",
            kernels::GEMV_MFP4G32_E8_SOA_GROUPED_GFX1151_SRC,
            a,
            x,
            y,
            groups,
            m,
            k,
        )
    }

    /// Exact-gfx1100 path for collapsing the eight O-LoRA E8 GEMVs into one
    /// 2-D launch. The included kernel preserves the incumbent width-32
    /// per-row arithmetic.
    pub fn gemv_mfp4g32_e8_soa_grouped_gfx1100(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx1100(),
            "grouped E8 gfx1100 requires exact gfx1100"
        );
        assert!(k % 256 == 0, "grouped E8 gfx1100 requires K%256=0");
        const KERNEL: &str = "gemv_mfp4g32_e8_soa_grouped_gfx1100";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMV_MFP4G32_E8_SOA_GROUPED_GFX1100_SRC,
            KERNEL,
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
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, groups as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
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

    /// Exact-gfx1100 launch surface for collapsing up to five
    /// differently-sized E8 projections which consume the same K-vector.
    pub fn gemv_mfp4g32_e8_soa_ragged_gfx1100(
        &mut self,
        weights: &[&GpuTensor],
        x: &GpuTensor,
        outputs: &[&GpuTensor],
        rows: &[usize],
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx1100(),
            "ragged E8 gfx1100 requires exact gfx1100"
        );
        assert!(
            (2..=5).contains(&weights.len()),
            "ragged E8 gfx1100 requires 2..=5 projections"
        );
        assert_eq!(weights.len(), outputs.len());
        assert_eq!(weights.len(), rows.len());
        assert!(k % 256 == 0, "ragged E8 gfx1100 requires K%256=0");

        const KERNEL: &str = "gemv_mfp4g32_e8_soa_ragged_gfx1100";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMV_MFP4G32_E8_SOA_RAGGED_GFX1100_SRC,
            KERNEL,
        )?;

        let first_a = weights[0].buf.as_ptr();
        let first_y = outputs[0].buf.as_ptr();
        let mut a_ptrs = [first_a; 5];
        let mut y_ptrs = [first_y; 5];
        let mut rows_i32 = [0_i32; 5];
        for i in 0..weights.len() {
            a_ptrs[i] = weights[i].buf.as_ptr();
            y_ptrs[i] = outputs[i].buf.as_ptr();
            rows_i32[i] = rows[i] as i32;
        }
        let x_ptr = x.buf.as_ptr();
        let k_i32 = k as i32;
        let total_rows: usize = rows.iter().sum();
        let mut params: Vec<*mut c_void> = Vec::with_capacity(17);
        for ptr in &a_ptrs {
            params.push(ptr as *const _ as *mut c_void);
        }
        params.push(&x_ptr as *const _ as *mut c_void);
        for ptr in &y_ptrs {
            params.push(ptr as *const _ as *mut c_void);
        }
        for row in &rows_i32 {
            params.push(row as *const _ as *mut c_void);
        }
        params.push(&k_i32 as *const _ as *mut c_void);

        self.launch_maybe_blob(
            KERNEL,
            [total_rows as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                for ptr in a_ptrs {
                    blob.push_ptr(ptr);
                }
                blob.push_ptr(x_ptr);
                for ptr in y_ptrs {
                    blob.push_ptr(ptr);
                }
                for row in rows_i32 {
                    blob.push_i32(row);
                }
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_mfp4g32_e8_soa_grouped_impl(
        &mut self,
        kname: &str,
        source: &str,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx1151(),
            "grouped E8 kernel is gfx1151-only"
        );
        assert!(
            k % 256 == 0,
            "grouped E8 kernel requires K%256==0, got K={k}"
        );
        let kname_owned = kname.to_owned();
        let kname: &str = &kname_owned;
        self.ensure_kernel(kname, source, kname)?;
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
        self.launch_maybe_blob(
            kname,
            [m as u32, groups as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
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

    /// SoA E8 GEMV, 4-way unroll. DeepSeek4 MQ2R route v3 uses the accepted
    /// gfx1151 temporal raw-buffer form; the tri-state override retains an
    /// exact portable control and allows isolated use on other gfx1151 routes.
    pub fn gemv_mfp4g32_e8_soa_u4(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let temporal_buffer = self.flags.gfx1151_e8_buffer.unwrap_or(false);
        if self.arch_caps.is_gfx1151() && temporal_buffer {
            return self.gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(0, a, x, y, m, k);
        }
        self.gemv_mfp4g32_e8_soa_variant(
            "gemv_mfp4g32_e8_soa_u4",
            kernels::GEMV_MFP4G32_E8_SOA_U4_SRC,
            a,
            x,
            y,
            m,
            k,
        )
    }

    pub fn gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(
        &mut self,
        cache_policy: u32,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "E8 cache-policy route is gfx1151-only"
        );
        let (name, source) = match cache_policy {
            0 => (
                "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL0_GFX1151_SRC,
            ),
            20 => (
                "gemv_mfp4g32_e8_soa_u4_buffer_cpol20_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL20_GFX1151_SRC,
            ),
            22 => (
                "gemv_mfp4g32_e8_soa_u4_buffer_cpol22_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL22_GFX1151_SRC,
            ),
            _ => panic!("unsupported gfx1151 E8 cache policy {cache_policy}"),
        };
        self.gemv_mfp4g32_e8_soa_variant(name, source, a, x, y, m, k)
    }

    /// gfx1151 E8-SoA U4 buffer GEMV with `sqrt(softplus(.))` fused into the
    /// store, for the DeepSeek V4 MoE router (`gate.weight @ x` then the routing
    /// affinity activation). Saves the standalone `sqrt_softplus_f32` launch and
    /// its dependent boundary; see
    /// `kernels::GEMV_MFP4G32_E8_SOA_U4_BUFFER_SQRT_SOFTPLUS_GFX1151_SRC`.
    ///
    /// Numerically identical to `gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(0, ..)`
    /// followed by `sqrt_softplus_f32`: the reduction is unchanged and the
    /// activation applies the same branched formulation to the same f32 value,
    /// which the unfused path would have round-tripped through memory exactly.
    pub fn gemv_mfp4g32_e8_soa_u4_buffer_sqrt_softplus_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "fused sqrt-softplus E8 route is gfx1151-only"
        );
        self.gemv_mfp4g32_e8_soa_variant(
            "gemv_mfp4g32_e8_soa_u4_buffer_sqrt_softplus_gfx1151",
            kernels::GEMV_MFP4G32_E8_SOA_U4_BUFFER_SQRT_SOFTPLUS_GFX1151_SRC,
            a,
            x,
            y,
            m,
            k,
        )
    }

    /// Batch sizes with a compiled batched-E8-GEMV instantiation.
    ///
    /// `E8_BATCHED_B` is a compile-time constant in the kernel (the per-token
    /// accumulators must live in registers, not scratch), so each supported
    /// `B` is a separate specialization and unlisted sizes have no kernel to
    /// launch. Callers must check membership before dispatching.
    pub const E8_BATCHED_GEMV_BATCHES: &'static [usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 16];

    /// Batched E8-SoA decode GEMV: `A[M,K] @ X[B,K]^T -> Y[B,M]`, one wave per
    /// output row with the weight row read once for all B tokens.
    ///
    /// Intended for speculative-verify batch sizes (B < 16), where the prefill
    /// WMMA GEMM computes a full 16-token tile regardless of B and launches
    /// only M/16 waves. B must be one of the compiled instantiations.
    pub fn gemv_mfp4g32_e8_soa_batched_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        batch: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "batched E8 decode GEMV is gfx1151-only"
        );
        assert!(k % 256 == 0, "batched E8 GEMV requires K%256==0, got K={k}");
        let (name, source) = match batch {
            1 => (
                "gemv_mfp4g32_e8_soa_batched_b1_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B1_GFX1151_SRC,
            ),
            2 => (
                "gemv_mfp4g32_e8_soa_batched_b2_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B2_GFX1151_SRC,
            ),
            3 => (
                "gemv_mfp4g32_e8_soa_batched_b3_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B3_GFX1151_SRC,
            ),
            4 => (
                "gemv_mfp4g32_e8_soa_batched_b4_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B4_GFX1151_SRC,
            ),
            5 => (
                "gemv_mfp4g32_e8_soa_batched_b5_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B5_GFX1151_SRC,
            ),
            6 => (
                "gemv_mfp4g32_e8_soa_batched_b6_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B6_GFX1151_SRC,
            ),
            7 => (
                "gemv_mfp4g32_e8_soa_batched_b7_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B7_GFX1151_SRC,
            ),
            8 => (
                "gemv_mfp4g32_e8_soa_batched_b8_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B8_GFX1151_SRC,
            ),
            16 => (
                "gemv_mfp4g32_e8_soa_batched_b16_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_BATCHED_B16_GFX1151_SRC,
            ),
            other => panic!("no batched E8 GEMV instantiation for B={other}"),
        };
        self.bind_thread()?;
        self.ensure_kernel(name, source, name)?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = a.byte_size() + batch * (k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", name, bytes);
        let result =
            self.launch_maybe_blob(name, [m as u32, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Two independent B=3 E8 projections with the same shape and input in a
    /// single dispatch. The matrix selector only widens the grid; each output
    /// row executes the exact scalar body and reduction order of the ordinary
    /// B3 kernel.
    pub fn gemv_mfp4g32_e8_soa_batched_pair_b3_gfx1151(
        &mut self,
        a0: &GpuTensor,
        a1: &GpuTensor,
        x: &GpuTensor,
        y0: &GpuTensor,
        y1: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "paired batched E8 decode GEMV is gfx1151-only"
        );
        assert!(
            k % 256 == 0,
            "paired batched E8 GEMV requires K%256==0, got K={k}"
        );
        let name = "gemv_mfp4g32_e8_soa_batched_pair_b3_gfx1151";
        self.bind_thread()?;
        self.ensure_kernel(
            name,
            kernels::GEMV_MFP4G32_E8_SOA_BATCHED_PAIR_B3_GFX1151_SRC,
            name,
        )?;
        let a0_ptr = a0.buf.as_ptr();
        let a1_ptr = a1.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y0_ptr = y0.buf.as_ptr();
        let y1_ptr = y1.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a0_ptr as *const _ as *mut c_void,
            &a1_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y0_ptr as *const _ as *mut c_void,
            &y1_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = a0.byte_size() + a1.byte_size() + 3 * (k * 4 + 2 * m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", name, bytes);
        let result = self.launch_maybe_blob(
            name,
            [(2 * m) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a0_ptr);
                blob.push_ptr(a1_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y0_ptr);
                blob.push_ptr(y1_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Up to seven independent B=3 E8 projections that share one input and K
    /// dimension. Each matrix contributes its own contiguous row range to the
    /// grid; workgroups still execute the ordinary B3 per-row arithmetic.
    pub fn gemv_mfp4g32_e8_soa_batched_pack_b3_gfx1151(
        &mut self,
        a: [&GpuTensor; 7],
        x: &GpuTensor,
        y: [&GpuTensor; 7],
        m: [usize; 7],
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "packed batched E8 decode GEMV is gfx1151-only"
        );
        assert!(
            k % 256 == 0,
            "packed batched E8 GEMV requires K%256==0, got K={k}"
        );
        let total_m: usize = m.iter().sum();
        assert!(total_m > 0, "packed batched E8 GEMV requires output rows");
        let name = "gemv_mfp4g32_e8_soa_batched_pack_b3_gfx1151";
        self.bind_thread()?;
        self.ensure_kernel(
            name,
            kernels::GEMV_MFP4G32_E8_SOA_BATCHED_PACK_B3_GFX1151_SRC,
            name,
        )?;

        let bytes = a
            .iter()
            .zip(m.iter())
            .filter(|(_, rows)| **rows > 0)
            .map(|(weight, _)| weight.byte_size())
            .sum::<usize>()
            + 3 * (k * 4 + total_m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", name, bytes);

        let [a0, a1, a2, a3, a4, a5, a6] = a;
        let [y0, y1, y2, y3, y4, y5, y6] = y;
        let [m0, m1, m2, m3, m4, m5, m6] = m.map(|value| value as i32);
        let a0_ptr = a0.buf.as_ptr();
        let a1_ptr = a1.buf.as_ptr();
        let a2_ptr = a2.buf.as_ptr();
        let a3_ptr = a3.buf.as_ptr();
        let a4_ptr = a4.buf.as_ptr();
        let a5_ptr = a5.buf.as_ptr();
        let a6_ptr = a6.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y0_ptr = y0.buf.as_ptr();
        let y1_ptr = y1.buf.as_ptr();
        let y2_ptr = y2.buf.as_ptr();
        let y3_ptr = y3.buf.as_ptr();
        let y4_ptr = y4.buf.as_ptr();
        let y5_ptr = y5.buf.as_ptr();
        let y6_ptr = y6.buf.as_ptr();
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a0_ptr as *const _ as *mut c_void,
            &a1_ptr as *const _ as *mut c_void,
            &a2_ptr as *const _ as *mut c_void,
            &a3_ptr as *const _ as *mut c_void,
            &a4_ptr as *const _ as *mut c_void,
            &a5_ptr as *const _ as *mut c_void,
            &a6_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y0_ptr as *const _ as *mut c_void,
            &y1_ptr as *const _ as *mut c_void,
            &y2_ptr as *const _ as *mut c_void,
            &y3_ptr as *const _ as *mut c_void,
            &y4_ptr as *const _ as *mut c_void,
            &y5_ptr as *const _ as *mut c_void,
            &y6_ptr as *const _ as *mut c_void,
            &m0 as *const _ as *mut c_void,
            &m1 as *const _ as *mut c_void,
            &m2 as *const _ as *mut c_void,
            &m3 as *const _ as *mut c_void,
            &m4 as *const _ as *mut c_void,
            &m5 as *const _ as *mut c_void,
            &m6 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            name,
            [total_m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                for ptr in [
                    a0_ptr, a1_ptr, a2_ptr, a3_ptr, a4_ptr, a5_ptr, a6_ptr, x_ptr, y0_ptr, y1_ptr,
                    y2_ptr, y3_ptr, y4_ptr, y5_ptr, y6_ptr,
                ] {
                    blob.push_ptr(ptr);
                }
                for rows in [m0, m1, m2, m3, m4, m5, m6, k_i32] {
                    blob.push_i32(rows);
                }
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched grouped E8-SoA decode GEMV: `A[G,M,K] @ X[B,G,K] -> Y[B,G,M]`,
    /// one wave per (group, row) with the weight row read once for all B
    /// tokens.
    ///
    /// Intended for speculative-verify batch sizes, where
    /// `gemm_mfp4g32_e8_soa_grouped_wmma` computes a full 16-token tile
    /// regardless of B. `batch` must be in [`Self::E8_BATCHED_GEMV_BATCHES`].
    pub fn gemv_mfp4g32_e8_soa_grouped_batched_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        batch: usize,
        g: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(
            self.arch_caps.is_gfx1151(),
            "batched grouped E8 GEMV is gfx1151-only"
        );
        assert!(
            k % 256 == 0,
            "batched grouped E8 GEMV requires K%256==0, got K={k}"
        );
        let (name, source) = match batch {
            1 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b1_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B1_GFX1151_SRC,
            ),
            2 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b2_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B2_GFX1151_SRC,
            ),
            3 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b3_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B3_GFX1151_SRC,
            ),
            4 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b4_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B4_GFX1151_SRC,
            ),
            5 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b5_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B5_GFX1151_SRC,
            ),
            6 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b6_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B6_GFX1151_SRC,
            ),
            7 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b7_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B7_GFX1151_SRC,
            ),
            8 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b8_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B8_GFX1151_SRC,
            ),
            16 => (
                "gemv_mfp4g32_e8_soa_grouped_batched_b16_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B16_GFX1151_SRC,
            ),
            other => panic!("no batched grouped E8 GEMV instantiation for B={other}"),
        };
        self.bind_thread()?;
        self.ensure_kernel(name, source, name)?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let g_i32 = g as i32;
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &g_i32 as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = a.byte_size() + batch * g * (k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", name, bytes);
        let result = self.launch_maybe_blob(
            name,
            [m as u32, g as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(g_i32);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// SoA E8 GEMV, 8-way unroll (bench experiment — cache-roofline MLP sweep).
    pub fn gemv_mfp4g32_e8_soa_u8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_mfp4g32_e8_soa_variant(
            "gemv_mfp4g32_e8_soa_u8",
            kernels::GEMV_MFP4G32_E8_SOA_U8_SRC,
            a,
            x,
            y,
            m,
            k,
        )
    }
    /// SoA E8 GEMV, lattice-decode STRIPPED (bench probe — compute ceiling, garbage out).
    pub fn gemv_mfp4g32_e8_soa_strip(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_mfp4g32_e8_soa_variant(
            "gemv_mfp4g32_e8_soa_strip",
            kernels::GEMV_MFP4G32_E8_SOA_STRIP_SRC,
            a,
            x,
            y,
            m,
            k,
        )
    }
    /// SoA E8 GEMV, LUT-based coordinate decode (bench experiment).
    pub fn gemv_mfp4g32_e8_soa_lut(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemv_mfp4g32_e8_soa_variant(
            "gemv_mfp4g32_e8_soa_lut",
            kernels::GEMV_MFP4G32_E8_SOA_LUT_SRC,
            a,
            x,
            y,
            m,
            k,
        )
    }

    /// mfp4-E8 SoA prerotated (x already FWHT-rotated by caller).
    pub fn gemv_mfp4g32_e8_soa_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_mfp4g32_e8_soa(a_raw, x_rot, y, m, k)
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

    /// MagnumQuant MQ5: rotate x via FWHT, then HFQ5 GEMV.
    pub fn gemv_mq5g256_with_rotate(
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
        self.gemv_hfq5g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ5 with pre-rotated x.
    pub fn gemv_mq5g256_prerotated(
        &mut self,
        a_raw: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.gemv_hfq5g256(a_raw, x_rot, y, m, k)
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
                    [32, 1, 1],
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
    /// HFQ5-G256 GEMV with fused residual add: y[row] += A[row] . x.
    /// Same shape as gemv_hfq5g256; only the final write differs (+= vs =).
    /// Used for wo and w_down in HFQ5 / MQ5 forward paths.
    pub fn gemv_hfq5g256_residual(
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
                    "gemv_hfq5g256_residual_wave64_prefetch",
                    kernels::GEMV_HFQ5G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                )
            } else {
                (
                    "gemv_hfq5g256_residual_wave64",
                    kernels::GEMV_HFQ5G256_RESIDUAL_WAVE64_SRC,
                )
            };
            self.ensure_kernel(kname, ksrc, kname)?;
            let func = &self.functions[kname];
            let grid = ((m as u32) + 1) / 2;
            return unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid, 1, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        self.ensure_kernel(
            "gemv_hfq5g256_residual",
            kernels::GEMV_HFQ5G256_RESIDUAL_SRC,
            "gemv_hfq5g256_residual",
        )?;
        let func = &self.functions["gemv_hfq5g256_residual"];
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
    /// HFQ5-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq5g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemv_hfq5g256", kernels::GEMV_HFQ5G256_SRC, "gemv_hfq5g256")?;
        let func = &self.functions["gemv_hfq5g256"];
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
        use std::sync::OnceLock;
        static GFX1151_LM_HEAD_DOT2: OnceLock<bool> = OnceLock::new();
        let gfx1151_lm_head_dot2 = self.arch_caps.is_gfx1151()
            && m == 248_320
            && k == 2_048
            && *GFX1151_LM_HEAD_DOT2.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_DOT2").as_deref() == Ok("1")
            });
        let gfx1151_lm_head_r1_hybrid_buffer =
            self.arch_caps.is_gfx1151() && m == 248_320 && k == 2_048;
        let use_lm_head_k2048 = self.arch_caps.is_gfx1100()
            && self.flags.rdna3_hfq4_lm_head_k2048
            && m == 248_320
            && k == 2_048;
        let func_name = if gfx1151_lm_head_dot2 {
            self.ensure_kernel(
                "gemv_hfq4g256_lm_head_dot2_gfx1151",
                kernels::GEMV_HFQ4G256_LM_HEAD_DOT2_GFX1151_SRC,
                "gemv_hfq4g256_lm_head_dot2_gfx1151",
            )?;
            "gemv_hfq4g256_lm_head_dot2_gfx1151"
        } else if gfx1151_lm_head_r1_hybrid_buffer {
            self.ensure_kernel(
                "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_LM_HEAD_R1_HYBRID_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151",
            )?;
            "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"
        } else if use_lm_head_k2048 {
            self.ensure_kernel(
                "gemv_hfq4g256_k2048_gfx1100",
                kernels::GEMV_HFQ4G256_K2048_GFX1100_SRC,
                "gemv_hfq4g256_k2048",
            )?;
            "gemv_hfq4g256_k2048"
        } else {
            let (hfq4g256_src, hfq4g256_module) =
                kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
            self.ensure_kernel(hfq4g256_module, hfq4g256_src, "gemv_hfq4g256")?;
            "gemv_hfq4g256"
        };

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
        let use_multirow = rows > 1 && !gfx1151_lm_head_dot2 && !gfx1151_lm_head_r1_hybrid_buffer;
        static GFX1151_LM_HEAD_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_lm_head_buffer = self.arch_caps.is_gfx1151()
            && rows == 2
            && m == 248_320
            && k == 2_048
            && *GFX1151_LM_HEAD_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_LM_HEAD_HYBRID_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_lm_head_hybrid_buffer = self.arch_caps.is_gfx1151()
            && rows == 2
            && m == 248_320
            && k == 2_048
            && *GFX1151_LM_HEAD_HYBRID_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_LM_HEAD_ALL_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_lm_head_all_buffer = self.arch_caps.is_gfx1151()
            && rows == 2
            && m == 248_320
            && k == 2_048
            && *GFX1151_LM_HEAD_ALL_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_ALL_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_LM_HEAD_CPOL: OnceLock<Option<String>> = OnceLock::new();
        let gfx1151_lm_head_cpol =
            if self.arch_caps.is_gfx1151() && rows == 2 && m == 248_320 && k == 2_048 {
                GFX1151_LM_HEAD_CPOL
                    .get_or_init(|| {
                        hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_CPOL").ok()
                    })
                    .as_deref()
            } else {
                None
            };
        static GFX1151_LM_HEAD_K2048: OnceLock<bool> = OnceLock::new();
        let gfx1151_lm_head_k2048 = self.arch_caps.is_gfx1151()
            && rows == 2
            && m == 248_320
            && k == 2_048
            && *GFX1151_LM_HEAD_K2048.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_LM_HEAD_K2048").as_deref() == Ok("1")
            });

        // RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel.
        // Other non-RDNA3 archs: use wide kernel (2 rows/block) for large M.
        let use_wide = !gfx1151_lm_head_dot2
            && !gfx1151_lm_head_r1_hybrid_buffer
            && !use_multirow
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
            let (mr_name, mr_src) = if gfx1151_lm_head_cpol == Some("glc") {
                (
                    "gemv_hfq4g256_multirow_gfx1151_hybrid_glc",
                    kernels::GEMV_HFQ4G256_MULTIROW_HYBRID_GLC_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_cpol == Some("slc") {
                (
                    "gemv_hfq4g256_multirow_gfx1151_hybrid_slc",
                    kernels::GEMV_HFQ4G256_MULTIROW_HYBRID_SLC_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_cpol == Some("dlc") {
                (
                    "gemv_hfq4g256_multirow_gfx1151_hybrid_dlc",
                    kernels::GEMV_HFQ4G256_MULTIROW_HYBRID_DLC_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_k2048 {
                (
                    "gemv_hfq4g256_multirow_gfx1151_k2048",
                    kernels::GEMV_HFQ4G256_MULTIROW_K2048_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_all_buffer {
                (
                    "gemv_hfq4g256_multirow_gfx1151_all_buffer",
                    kernels::GEMV_HFQ4G256_MULTIROW_ALL_BUFFER_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_hybrid_buffer {
                (
                    "gemv_hfq4g256_multirow_gfx1151_hybrid_buffer",
                    kernels::GEMV_HFQ4G256_MULTIROW_HYBRID_BUFFER_GFX1151_SRC,
                )
            } else if gfx1151_lm_head_buffer {
                (
                    "gemv_hfq4g256_multirow_gfx1151_buffer",
                    kernels::GEMV_HFQ4G256_MULTIROW_BUFFER_GFX1151_SRC,
                )
            } else if rdna3 {
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
                // The wide kernel maps warp_id 0/1 to the two rows owned by
                // each block. Launching one wave silently skipped every odd
                // row while preserving valid memory accesses.
                [64, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        } else {
            self.launch_maybe_blob(
                func_name,
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
        let use_k2048 = self.arch_caps.is_gfx1100()
            && self.flags.rdna3_hfq4_residual_stage_x32
            && self.flags.rdna3_hfq4_residual_k2048
            && k == 2_048;
        use std::sync::OnceLock;
        static GFX1151_WEIGHT_BUFFER_LOADS: OnceLock<bool> = OnceLock::new();
        let gfx1151_buffer = self.arch_caps.is_gfx1151()
            && *GFX1151_WEIGHT_BUFFER_LOADS.get_or_init(|| {
                // Keep residual separate from the combined gfx1151 probe:
                // LLVM spills this dual-row raw-buffer schedule (private=16),
                // which is not admissible for the scratch-free PM4 replay.
                hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_RESIDUAL").as_deref()
                    == Ok("1")
            });
        static GFX1151_RESIDUAL_HYBRID_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_hybrid_buffer = self.arch_caps.is_gfx1151()
            && *GFX1151_RESIDUAL_HYBRID_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER").as_deref()
                    == Ok("1")
            });
        let gfx1151_rt_low = self.arch_caps.is_gfx1151();
        static GFX1151_RESIDUAL_ROW1: OnceLock<bool> = OnceLock::new();
        let gfx1151_row1 = self.arch_caps.is_gfx1151()
            && *GFX1151_RESIDUAL_ROW1.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_ROW1").as_deref() == Ok("1")
            });
        static GFX1151_RESIDUAL_K4096: OnceLock<bool> = OnceLock::new();
        let gfx1151_k4096 = self.arch_caps.is_gfx1151()
            && m == 2_048
            && k == 4_096
            && *GFX1151_RESIDUAL_K4096.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_K4096").as_deref()
                    == Ok("1")
            });
        static RESIDUAL_CPOL: OnceLock<Option<String>> = OnceLock::new();
        let cpol = RESIDUAL_CPOL
            .get_or_init(|| hipfire_config::developer_var("HIPFIRE_RESIDUAL_CPOL").ok())
            .as_deref();
        let cpol = if self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_residual_stage_x32 {
            cpol
        } else {
            None
        };
        let (src, module, func_name) = if gfx1151_k4096 {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_K4096_GFX1151_SRC,
                "gemv_hfq4g256_residual_k4096_gfx1151",
                "gemv_hfq4g256_residual_k4096_gfx1151",
            )
        } else if gfx1151_hybrid_buffer {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_HYBRID_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_residual_hybrid_buffer_gfx1151",
                "gemv_hfq4g256_residual_hybrid_buffer_gfx1151",
            )
        } else if gfx1151_rt_low {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_RT_LOW_GFX1151_SRC,
                "gemv_hfq4g256_residual_rt_low_gfx1151",
                "gemv_hfq4g256_residual_rt_low_gfx1151",
            )
        } else if gfx1151_row1 {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_ROW1_GFX1151_SRC,
                "gemv_hfq4g256_residual_row1_gfx1151",
                "gemv_hfq4g256_residual_row1_gfx1151",
            )
        } else if cpol == Some("rt-low") {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_CPOL_RT_LOW_GFX1100_SRC,
                "gemv_hfq4g256_residual_cpol_rt_low_gfx1100",
                "gemv_hfq4g256_residual_cpol_rt_low",
            )
        } else if cpol == Some("rt") {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_CPOL_RT_GFX1100_SRC,
                "gemv_hfq4g256_residual_cpol_rt_gfx1100",
                "gemv_hfq4g256_residual_cpol_rt",
            )
        } else if cpol == Some("slc") {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_CPOL_SLC_GFX1100_SRC,
                "gemv_hfq4g256_residual_cpol_slc_gfx1100",
                "gemv_hfq4g256_residual_cpol_slc",
            )
        } else if use_k2048 {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_K2048_GFX1100_SRC,
                "gemv_hfq4g256_residual_k2048",
                "gemv_hfq4g256_residual_k2048",
            )
        } else if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_residual_stage_x32 {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_STAGE_X32_GFX1100_SRC,
                "gemv_hfq4g256_residual_stage_x32_gfx1100",
                "gemv_hfq4g256_residual",
            )
        } else if gfx1151_buffer {
            (
                kernels::GEMV_HFQ4G256_RESIDUAL_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_residual_buffer_gfx1151",
                "gemv_hfq4g256_residual",
            )
        } else {
            let (src, module) = kernels::gemv_hfq4g256_residual_for_arch(&self.arch_caps);
            (src, module, "gemv_hfq4g256_residual")
        };
        self.ensure_kernel(module, src, func_name)?;

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
        static GFX1151_RESIDUAL_WAVE64: OnceLock<bool> = OnceLock::new();
        let gfx1151_wave64 = self.arch_caps.is_gfx1151()
            && *GFX1151_RESIDUAL_WAVE64.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_WAVE64").as_deref()
                    == Ok("1")
            });
        static GFX1151_RESIDUAL_TIGHT_GRID: OnceLock<bool> = OnceLock::new();
        let gfx1151_tight_grid = self.arch_caps.is_gfx1151()
            && *GFX1151_RESIDUAL_TIGHT_GRID.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_TIGHT_GRID").as_deref()
                    == Ok("1")
            });
        static GFX1151_RESIDUAL_MULTIROW_R2: OnceLock<bool> = OnceLock::new();
        let gfx1151_multirow_r2 = self.arch_caps.is_gfx1151()
            && m == 2_048
            && k == 2_048
            && *GFX1151_RESIDUAL_MULTIROW_R2.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_RESIDUAL_MULTIROW_R2").as_deref()
                    == Ok("1")
            });
        let cdna3 = self.arch_caps.is_wave64_native() || gfx1151_wave64;

        // RDNA3 multi-row override path. Same selector as the non-residual
        // variant but there's currently no gfx1010-default multi-row residual
        // kernel, so non-RDNA3 archs still take the single-row residual path
        // regardless of HIPFIRE_GEMV_ROWS. (TODO: port the multi-row residual
        // kernel to the default path if/when the non-residual multi-row wins
        // scale to justify residual too.)
        let rdna3 = self.arch_caps.is_rdna3_dgpu();
        let rows = if gfx1151_multirow_r2 {
            2
        } else if rdna3 {
            self.flags.gemv_rows.unwrap_or(1)
        } else {
            1
        };
        let use_multirow = (rdna3 && rows > 1) || gfx1151_multirow_r2;

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
                let (module, ksrc, kname): (&str, &str, &str) = if gfx1151_wave64 {
                    (
                        "gemv_hfq4g256_residual_wave64_gfx1151",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_GFX1151_SRC,
                        "gemv_hfq4g256_residual_wave64",
                    )
                } else if self.arch_caps.gemv_prefetch_enabled() {
                    (
                        "gemv_hfq4g256_residual_wave64_prefetch",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_PREFETCH_SRC,
                        "gemv_hfq4g256_residual_wave64_prefetch",
                    )
                } else {
                    (
                        "gemv_hfq4g256_residual_wave64",
                        kernels::GEMV_HFQ4G256_RESIDUAL_WAVE64_SRC,
                        "gemv_hfq4g256_residual_wave64",
                    )
                };
                self.ensure_kernel(module, ksrc, kname)?;
                let grid = ((m as u32) + 1) / 2;
                let block = if gfx1151_wave64 { 64 } else { 32 };
                self.launch_maybe_blob(kname, [grid, 1, 1], [block, 1, 1], 0, &mut params, || {
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
            let (func_name, grid_div) = match (gfx1151_multirow_r2, rows) {
                (true, 2) => ("gemv_hfq4g256_residual_multirow_r2_gfx1151", 2u32),
                (false, 2) => ("gemv_hfq4g256_residual_multirow_r2", 2u32),
                (false, 4) => ("gemv_hfq4g256_residual_multirow_r4", 4u32),
                (false, 8) => ("gemv_hfq4g256_residual_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            let (module, source) = if gfx1151_multirow_r2 {
                (
                    "gemv_hfq4g256_residual_multirow_gfx1151",
                    kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1151_SRC,
                )
            } else {
                (
                    "gemv_hfq4g256_residual_multirow_rdna3",
                    kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC,
                )
            };
            self.ensure_kernel(module, source, func_name)?;
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
            // The generic residual kernel owns row0/row1 in one wave32 block.
            // Its legacy M-sized launch leaves the upper half of the grid to
            // exit at the row0 guard. Qualify that target-neutral contraction
            // independently on gfx1151 instead of coupling it to wave64.
            let grid = if gfx1151_tight_grid {
                (m as u32).div_ceil(2)
            } else {
                m as u32
            };
            self.launch_maybe_blob(func_name, [grid, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            })
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
        let rdna3_rows4 = self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_sigmoid_rows4;
        let rdna3_buffer = self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_sigmoid_buffer;
        use std::sync::OnceLock;
        static GFX1151_WEIGHT_BUFFER_LOADS: OnceLock<bool> = OnceLock::new();
        let gfx1151_k512_buffer = self.arch_caps.is_gfx1151()
            && m == 2_048
            && k == 512
            && *GFX1151_WEIGHT_BUFFER_LOADS.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS").as_deref()
                    == Ok("1")
                    || hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_SIGMOID")
                        .as_deref()
                        == Ok("1")
            });
        static SIGMOID_K512: OnceLock<bool> = OnceLock::new();
        static SIGMOID_HOIST_X16: OnceLock<bool> = OnceLock::new();
        let rdna3_hoist_x16 = self.arch_caps.is_gfx1100()
            && rdna3_buffer
            && !rdna3_rows4
            && m == 2_048
            && k == 512
            && *SIGMOID_HOIST_X16.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_RDNA3_HFQ4_SIGMOID_HOIST_X16").as_deref()
                    == Ok("1")
            });
        let rdna3_k512 = self.arch_caps.is_gfx1100()
            && rdna3_buffer
            && !rdna3_rows4
            && m == 2_048
            && k == 512
            && *SIGMOID_K512.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_RDNA3_HFQ4_SIGMOID_K512").as_deref()
                    == Ok("1")
            });
        let (module, source) = if rdna3_hoist_x16 {
            (
                "gemv_hfq4g256_residual_sigmoid_k512_hoist_x16_buffer_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_HOIST_X16_BUFFER_GFX1100_SRC,
            )
        } else if rdna3_k512 {
            (
                "gemv_hfq4g256_residual_sigmoid_k512_buffer_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_BUFFER_GFX1100_SRC,
            )
        } else if rdna3_rows4 && rdna3_buffer {
            (
                "gemv_hfq4g256_residual_sigmoid_rows4_buffer_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_ROWS4_BUFFER_GFX1100_SRC,
            )
        } else if rdna3_rows4 {
            (
                "gemv_hfq4g256_residual_sigmoid_rows4_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_ROWS4_GFX1100_SRC,
            )
        } else if rdna3_buffer {
            (
                "gemv_hfq4g256_residual_sigmoid_buffer_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_BUFFER_GFX1100_SRC,
            )
        } else if gfx1151_k512_buffer {
            (
                "gemv_hfq4g256_residual_sigmoid_k512_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_BUFFER_GFX1151_SRC,
            )
        } else {
            (
                "gemv_hfq4g256_residual_scaled",
                kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            )
        };
        self.ensure_kernel(module, source, "gemv_hfq4g256_residual_sigmoid_scaled_gpu")?;
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
        let grid_m = if rdna3_rows4 {
            m.div_ceil(4)
        } else if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_sigmoid_tight_grid {
            m.div_ceil(2)
        } else {
            m
        };
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            [grid_m as u32, 1, 1],
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
    /// HFQ5G256 batched residual GEMV with fused sigmoid-scaled add. Same
    /// shape as the HFQ6 sibling; reads the 168 B/group 5-bit layout. Used
    /// by the batched MoE-FFN shared-expert `down` projection where
    /// shared.down is MQ5.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq5g256_residual_sigmoid_scaled_gpu_batched(
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
            "gemv_hfq5g256_residual_sigmoid_scaled",
            kernels::GEMV_HFQ5G256_RESIDUAL_SIGMOID_SCALED_SRC,
            "gemv_hfq5g256_residual_sigmoid_scaled_gpu_batched",
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
        // HFQ5 weight footprint: m * (k / 256) * 168 bytes per row + 4 B per
        // input/output cell. No dedicated profile helper yet (HFQ5 GEMV
        // currently doesn't appear in profile.rs); inlined here.
        let groups = k / 256;
        let weight_bytes = m * groups * 168;
        let bytes = batch_size * (weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq5g256_residual_sigmoid_scaled_gpu_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq5g256_residual_sigmoid_scaled_gpu_batched",
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

    /// RDNA wave64 zero-LDS fused router. The source carries the required
    /// whole-translation-unit `-mwavefrontsize64` compiler flag.
    pub fn moe_router_softmax_topk_k8_wave64(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let name = "moe_router_softmax_topk_k8_wave64";
        self.ensure_kernel(name, kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC, name)?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = i32::from(norm_topk);
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(name, [1, 1, 1], [64, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(lp);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_i32(n);
            b.push_i32(nr);
            b
        })
    }

    /// gfx1100 wave64 fused router which emulates the production 256-thread
    /// softmax and wave32 top-k reduction order exactly. Restricted by the
    /// dispatch site to the 256-expert path it was designed to replace.
    pub fn moe_router_softmax_topk_k8_wave64_exact(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let name = "moe_router_softmax_topk_k8_wave64_exact";
        self.ensure_kernel(
            name,
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC,
            name,
        )?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = i32::from(norm_topk);
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(name, [1, 1, 1], [64, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(lp);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_i32(n);
            b.push_i32(nr);
            b
        })
    }

    /// gfx1100 heterogeneous dispatch for the two independent operations
    /// immediately following the fused gate-side projection: exact router
    /// selection and shared-expert SwiGLU + MQ rotation. Workgroup 0 runs the
    /// router while each later wave64 carries two original wave32 FWHT groups.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: usize,
        norm_topk: bool,
        shared_gate: &GpuTensor,
        shared_up: &GpuTensor,
        shared_x_rot: &GpuTensor,
        shared_k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        let name = "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate";
        self.ensure_kernel(
            "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate_gfx1100",
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SHARED_SILU_MQ_ROTATE_SRC,
            name,
        )?;

        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let n = n_exp as i32;
        let nr = i32::from(norm_topk);
        let gp = shared_gate.buf.as_ptr();
        let up = shared_up.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xr = shared_x_rot.buf.as_ptr();
        let k = shared_k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xr as *const _ as *mut c_void,
            &k as *const _ as *mut c_void,
        ];
        let activation_workgroups = shared_k.div_ceil(512) as u32;
        let bytes = n_exp * 4 + 8 * (4 + 4) + shared_k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", name, bytes);
        let result = self.launch_maybe_blob(
            name,
            [1 + activation_workgroups, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(n);
                b.push_i32(nr);
                b.push_ptr(gp);
                b.push_ptr(up);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xr);
                b.push_i32(k);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xr);
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

    /// mfp4-E8 grouped MoE gate_up (k8 indexed). Launches the batched kernel
    /// with N=1 (bid=0 collapses the K_TOP-stride terms), so the gate_batch/up_batch
    /// output matches the hfq4g256 k8-indexed contract.
    ///
    /// On RDNA3 dGPU (gfx1100/1101/1102) with HIPFIRE_E8_DGPU_TWIN enabled (default ON),
    /// dispatches the 4-way-unroll gfx11_dgpu twin for better latency-hiding on GDDR6.
    /// Falls back to the gfx1151 kernel on all other RDNA3 arches (gfx1151 iGPU).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mfp4g32_e8_moe_gate_up_k8_indexed(
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
        debug_assert!(
            self.arch_caps.has_wmma_w32() || self.arch_caps.is_rdna4(),
            "gemv_mfp4g32_e8_moe_gate_up_k8_indexed needs RDNA3 or RDNA4"
        );
        // SoA path (decode): when experts were transposed AoS->SoA at load. Same
        // batched kernel interface (this launcher runs it with batch=1).
        let use_soa = self.arch_caps.is_rdna3_dgpu() && e8_soa_experts_enabled();
        let use_dgpu_twin = !use_soa && self.arch_caps.is_rdna3_dgpu() && e8_dgpu_twin_enabled();
        let (kname, ksrc, kfn) = if use_soa {
            (
                "gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched",
                kernels::GEMV_MFP4G32_E8_SOA_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC,
                "gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched",
            )
        } else if use_dgpu_twin {
            (
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched_dgpu",
                kernels::GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX11_DGPU_SRC,
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched_dgpu",
            )
        } else {
            (
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched",
                kernels::GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX1151_SRC,
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched",
            )
        };
        self.ensure_kernel(kname, ksrc, kfn)?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = 8i32;
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
        self.launch_maybe_blob(kname, [m as u32, 8, 1], [32, 1, 1], 0, &mut params, || {
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
        })
    }

    /// Batched mfp4-E8 grouped MoE gate_up (k8 indexed). Same kernel as the decode
    /// wrapper but launched over `n` tokens (grid.z = n): reads
    /// `topk_indices[bid*k_top + krank]` and `x + bid*K`, writing
    /// `y_gate/y_up[bid*k_top*mi + krank*mi + ...]`. Used by `run_moe_prefill` Path 1
    /// for the batched verify / prefill of E8 A3B.
    /// RDNA3 dGPU (gfx1100/1101/1102): dispatches the 4-way-unroll twin when
    /// HIPFIRE_E8_DGPU_TWIN is enabled (default ON).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
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
        debug_assert!(
            self.arch_caps.has_wmma_w32() || self.arch_caps.is_rdna4(),
            "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched needs RDNA3 or RDNA4"
        );
        // SoA-coalesced path takes priority when experts were transposed AoS->SoA at
        // load (RDNA3 dGPU + HIPFIRE_E8_SOA_EXPERTS=1). Same params/grid as the AoS
        // kernel; only the in-weight addressing differs.
        let use_soa = self.arch_caps.is_rdna3_dgpu() && e8_soa_experts_enabled();
        let use_dgpu_twin = !use_soa && self.arch_caps.is_rdna3_dgpu() && e8_dgpu_twin_enabled();
        let (kname, ksrc, kfn) = if use_soa {
            (
                "gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched",
                kernels::GEMV_MFP4G32_E8_SOA_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC,
                "gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched",
            )
        } else if use_dgpu_twin {
            (
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched_dgpu",
                kernels::GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX11_DGPU_SRC,
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched_dgpu",
            )
        } else {
            (
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched",
                kernels::GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX1151_SRC,
                "gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched",
            )
        };
        self.ensure_kernel(kname, ksrc, kfn)?;
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
        self.launch_maybe_blob(
            kname,
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
        )
    }

    /// mfp4-E8 grouped MoE down (k8 indexed, atomic-free expanded).
    /// Writes expert_outputs[N × K_TOP × M]; caller folds via moe_down_combine_k8_batched.
    /// On RDNA3 dGPU (gfx1100/1101/1102) with HIPFIRE_E8_DGPU_TWIN (default ON),
    /// dispatches the 4-way-unroll gfx11_dgpu twin.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
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
        debug_assert!(
            self.arch_caps.has_wmma_w32() || self.arch_caps.is_rdna4(),
            "gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded needs RDNA3 or RDNA4"
        );
        let use_dgpu_twin = self.arch_caps.is_rdna3_dgpu() && e8_dgpu_twin_enabled();
        let (kname, ksrc, kfn) = if use_dgpu_twin {
            (
                "gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded_dgpu",
                kernels::GEMV_MFP4G32_E8_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_GFX11_DGPU_SRC,
                "gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded_dgpu",
            )
        } else {
            (
                "gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded",
                kernels::GEMV_MFP4G32_E8_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_GFX1151_SRC,
                "gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded",
            )
        };
        self.ensure_kernel(kname, ksrc, kfn)?;
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
        self.launch_maybe_blob(
            kname,
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
        )
    }

    /// Index-aware MoE gate_up GEMV. Reads expert_ids from a device-side
    /// topk_indices buffer and weight bases from expert_ptrs[expert_id].
    /// hipGraph-capture-safe replacement for the kernarg-pointer variant.
    #[allow(clippy::too_many_arguments)]
    /// `n_ranks` is the number of top-k ranks (grid.y workgroups) this launch
    /// drives. In the uniform decode path it is the full `k` (=8); in the
    /// mixed-tier path it is a per-tier bucket size (`n < 8`, possibly with a
    /// non-zero base offset baked into the sub-viewed tensors). The kernel reads
    /// `topk_indices[blockIdx.y]` and writes `y_gate/y_up[blockIdx.y*mi+…]`, so
    /// launching `n_ranks` workgroups runs exactly ranks `0..n_ranks` — never
    /// past the (sub-viewed) buffer.
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
        n_ranks: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        use std::sync::OnceLock;
        static GFX1151_GATE_UP_PERSISTENT_RANK8: OnceLock<bool> = OnceLock::new();
        let gfx1151_persistent_rank8 = self.arch_caps.is_gfx1151()
            && k == 2_048
            && n_ranks == 8
            && *GFX1151_GATE_UP_PERSISTENT_RANK8.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_PERSISTENT_RANK8").as_deref()
                    == Ok("1")
            });
        static GFX1151_GATE_UP_PAIRED_WAVES: OnceLock<bool> = OnceLock::new();
        let gfx1151_paired_waves = !gfx1151_persistent_rank8
            && self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_GATE_UP_PAIRED_WAVES.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_PAIRED_WAVES").as_deref()
                    == Ok("1")
            });
        static GFX1151_GATE_UP_SPLIT: OnceLock<bool> = OnceLock::new();
        let gfx1151_split = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_GATE_UP_SPLIT.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_SPLIT").as_deref() == Ok("1")
            });
        if !gfx1151_persistent_rank8 && !gfx1151_paired_waves && gfx1151_split {
            return self.gemv_hfq4g256_moe_gate_up_k8_indexed_split_gfx1151(
                expert_ptrs,
                topk_indices,
                x,
                y_gate,
                y_up,
                m,
                k,
                n_ranks,
            );
        }
        static GFX1151_GATE_UP_WAVE64: OnceLock<bool> = OnceLock::new();
        let gfx1151_wave64 = !gfx1151_persistent_rank8
            && !gfx1151_paired_waves
            && self.arch_caps.is_gfx1151()
            && *GFX1151_GATE_UP_WAVE64.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_WAVE64").as_deref()
                    == Ok("1")
            });
        let cdna_wave64 = self.arch_caps.is_wave64_native() || gfx1151_wave64;
        let (func_name, block, grid_x) = if cdna_wave64 {
            let (module, source) = if gfx1151_wave64 {
                (
                    "gemv_hfq4g256_moe_gate_up_indexed_wave64_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_GFX1151_SRC,
                )
            } else {
                (
                    "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
                )
            };
            self.ensure_kernel(
                module,
                source,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            use std::sync::OnceLock;
            static GATE_UP_WG2: OnceLock<bool> = OnceLock::new();
            let wg2 = self.arch_caps.is_gfx1100()
                && *GATE_UP_WG2.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_WG2").as_deref() == Ok("1")
                });
            // The unchanged base kernel maps blockIdx.x to one logical row in
            // each M/2 gate and up output. The legacy M-sized grid therefore
            // leaves its upper half to exit at the row >= M/2 guard. Default
            // on for gfx1100 after exact shadow and stationary tg128 checks;
            // HIPFIRE_MOE_GATE_UP_TIGHT_GRID=0 restores the legacy geometry.
            static GATE_UP_TIGHT_GRID: OnceLock<bool> = OnceLock::new();
            let tight_grid = if self.arch_caps.is_gfx1100() {
                *GATE_UP_TIGHT_GRID.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_TIGHT_GRID").as_deref()
                        != Ok("0")
                })
            } else if self.arch_caps.is_gfx1151() {
                // gfx1151 keeps its own performance-admission gate. The
                // kernel maps blockIdx.x to one row in each M/2 output, so
                // M/2 is semantically sufficient on every target; require an
                // explicit gfx1151 experiment before making it that arch's
                // default. Do not let the gfx1100 default bleed through a
                // shared RDNA3 capability predicate.
                hipfire_config::developer_var("HIPFIRE_GFX1151_MOE_TIGHT_GRID").as_deref()
                    == Ok("1")
                    || hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_TIGHT_GRID")
                        .as_deref()
                        == Ok("1")
            } else {
                false
            };
            // Two independent wave32 rows in one workgroup. Unlike the old
            // row-tile below, this does not duplicate per-wave accumulators or
            // change a row's reduction order.
            // Opt-in NUM_ROWS=2 register row-tile (HIPFIRE_MOE_GATE_UP_FUSED=1):
            // ceil(M/2) blocks, each owning 2 output rows sharing this expert's x.
            // Token-id exact vs base (per-row math unchanged). Grid divisor here
            // MUST match the kernel's MOE_GATE_UP_NUM_ROWS.
            static GATE_UP_FUSED: OnceLock<bool> = OnceLock::new();
            let fused = *GATE_UP_FUSED.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_FUSED").as_deref() == Ok("1")
            });
            static GATE_UP_RANK_INTERLEAVE: OnceLock<bool> = OnceLock::new();
            let rank_interleave = self.arch_caps.is_gfx1100()
                && n_ranks == 8
                && *GATE_UP_RANK_INTERLEAVE.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE").as_deref()
                        == Ok("1")
                });
            static GATE_UP_LOW_VGPR: OnceLock<bool> = OnceLock::new();
            let low_vgpr = self.arch_caps.is_gfx1100()
                && *GATE_UP_LOW_VGPR.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_LOW_VGPR").as_deref()
                        == Ok("1")
                });
            static GATE_UP_PAIR_VGPR: OnceLock<bool> = OnceLock::new();
            let pair_vgpr = self.arch_caps.is_gfx1100()
                && *GATE_UP_PAIR_VGPR.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_PAIR_VGPR").as_deref()
                        == Ok("1")
                });
            static GATE_UP_CPOL: OnceLock<String> = OnceLock::new();
            let cpol = if self.arch_caps.is_gfx1100() {
                GATE_UP_CPOL
                    .get_or_init(|| {
                        hipfire_config::developer_var("HIPFIRE_MOE_GATE_UP_CPOL")
                            // SLC is the gfx1100 product-certified policy:
                            // exact-shadow A-B-A and stationary tg128 A-B-A
                            // both beat the temporal-buffer control. Use
                            // `default` to restore cache-policy zero.
                            .unwrap_or_else(|_| "slc".to_owned())
                            .to_ascii_lowercase()
                    })
                    .as_str()
            } else {
                ""
            };
            let fixed_k2048 = self.arch_caps.is_gfx1100()
                && self.flags.rdna3_hfq4_moe_gate_up_k2048
                && k == 2_048;
            static GFX1151_GATE_UP_K2048: OnceLock<bool> = OnceLock::new();
            let gfx1151_k2048 = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_K2048.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_K2048").as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_LOW_VGPR: OnceLock<bool> = OnceLock::new();
            let gfx1151_low_vgpr = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_LOW_VGPR.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_LOW_VGPR").as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_PAIR_VGPR: OnceLock<bool> = OnceLock::new();
            let gfx1151_pair_vgpr = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_PAIR_VGPR.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_PAIR_VGPR").as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_PAIR_BUFFER: OnceLock<bool> = OnceLock::new();
            let gfx1151_pair_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_PAIR_BUFFER.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_PAIR_BUFFER").as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_HYBRID_BUFFER: OnceLock<bool> = OnceLock::new();
            let gfx1151_hybrid_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_HYBRID_BUFFER.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_HYBRID_BUFFER")
                        .as_deref()
                        == Ok("1")
                });
            static GFX1151_WEIGHT_BUFFER_LOADS: OnceLock<bool> = OnceLock::new();
            let gfx1151_k2048_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_WEIGHT_BUFFER_LOADS.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS").as_deref()
                        == Ok("1")
                        || hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_GATE_UP")
                            .as_deref()
                            == Ok("1")
                });
            static GFX1151_GATE_UP_ALL_BUFFER: OnceLock<bool> = OnceLock::new();
            let gfx1151_all_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_ALL_BUFFER.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_ALL_BUFFER").as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_ROUTE_ALL_BUFFER: OnceLock<bool> = OnceLock::new();
            let gfx1151_route_all_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_ROUTE_ALL_BUFFER.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_ROUTE_ALL_BUFFER")
                        .as_deref()
                        == Ok("1")
                });
            static GFX1151_GATE_UP_PAIR_ALL_BUFFER: OnceLock<bool> = OnceLock::new();
            let gfx1151_pair_all_buffer = self.arch_caps.is_gfx1151()
                && k == 2_048
                && *GFX1151_GATE_UP_PAIR_ALL_BUFFER.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_GATE_UP_PAIR_ALL_BUFFER")
                        .as_deref()
                        == Ok("1")
                });
            if gfx1151_persistent_rank8 {
                const PERSISTENT: &str =
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151";
                self.ensure_kernel(
                    PERSISTENT,
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_PERSISTENT_RANK8_GFX1151_SRC,
                    PERSISTENT,
                )?;
                (PERSISTENT, [256u32, 1, 1], ((m as u32) >> 1).min(160))
            } else if gfx1151_paired_waves {
                const PAIRED: &str =
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151";
                self.ensure_kernel(
                    PAIRED,
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_PAIRED_WAVES_K2048_GFX1151_SRC,
                    PAIRED,
                )?;
                (PAIRED, [64u32, 1, 1], (m as u32) >> 1)
            } else if gfx1151_hybrid_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_hybrid_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_HYBRID_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151",
                    [32u32, 1, 1],
                    (m as u32) >> 1,
                )
            } else if gfx1151_route_all_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_route_all_buffer_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_ROUTE_ALL_BUFFER_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_pair_all_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_pair_all_buffer_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_ALL_BUFFER_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_all_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_all_buffer_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_ALL_BUFFER_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_pair_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_pair_buffer_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_BUFFER_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_low_vgpr {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_low_vgpr_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_LOW_VGPR_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_pair_vgpr {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_pair_vgpr_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_VGPR_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_k2048 {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_k2048_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if pair_vgpr {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_pair_slc",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_PAIR_SLC_GFX1100_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if matches!(cpol, "glc" | "slc" | "dlc") {
                let (module, source, func) = match cpol {
                    "glc" => (
                        "gemv_hfq4g256_moe_gate_up_indexed_cpol_glc",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_GLC_GFX1100_SRC,
                        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
                    ),
                    "slc" => (
                        "gemv_hfq4g256_moe_gate_up_indexed_cpol_slc",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_SLC_GFX1100_SRC,
                        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
                    ),
                    "dlc" => (
                        "gemv_hfq4g256_moe_gate_up_indexed_cpol_dlc",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_DLC_GFX1100_SRC,
                        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
                    ),
                    _ => unreachable!(),
                };
                self.ensure_kernel(module, source, func)?;
                (
                    func,
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if low_vgpr {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_low_vgpr",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_LOW_VGPR_GFX1100_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if rank_interleave {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_rank_interleave",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_RANK_INTERLEAVE_GFX1100_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
                )?;
                let mi = (m as u32) >> 1;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
                    [32u32, 1, 1],
                    mi * 8,
                )
            } else if wg2 {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_wg2",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WG2_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
                )?;
                let mi = (m as u32) >> 1;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
                    [64u32, 1, 1],
                    mi.div_ceil(2),
                )
            } else if fused {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_indexed_rowtile",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_ROWTILE_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_rowtile",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_rowtile",
                    [32u32, 1, 1],
                    ((m as u32) + 1) / 2,
                )
            } else if fixed_k2048 {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_GFX1100_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            } else if gfx1151_k2048_buffer {
                self.ensure_kernel(
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151",
                    kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_BUFFER_GFX1151_SRC,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151",
                )?;
                (
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151",
                    [32u32, 1, 1],
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
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
                    if tight_grid {
                        (m as u32) >> 1
                    } else {
                        m as u32
                    },
                )
            }
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
        let result = self.launch_maybe_blob(
            func_name,
            [
                grid_x,
                if matches!(
                    func_name,
                    "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave"
                        | "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151"
                ) {
                    1
                } else {
                    n_ranks as u32
                },
                1,
            ],
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151-only structural gate/up decomposition for fixed-K=2048 MQ4R
    /// decode. The two launches are sequential on ordinary HIP, but their
    /// disjoint output contracts let retained Redline PM4 replay place them on
    /// separate queues and join once at the existing SiLU×multiply consumer.
    #[allow(clippy::too_many_arguments)]
    fn gemv_hfq4g256_moe_gate_up_k8_indexed_split_gfx1151(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        n_ranks: usize,
    ) -> HipResult<()> {
        debug_assert!(self.arch_caps.is_gfx1151());
        debug_assert_eq!(k, 2_048);

        const GATE: &str = "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151";
        const UP: &str = "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151";
        self.ensure_kernel(
            GATE,
            kernels::GEMV_HFQ4G256_MOE_GATE_INDEXED_K2048_GFX1151_SRC,
            GATE,
        )?;
        self.ensure_kernel(
            UP,
            kernels::GEMV_HFQ4G256_MOE_UP_INDEXED_K2048_GFX1151_SRC,
            UP,
        )?;

        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let grid = [(m as u32) >> 1, n_ranks as u32, 1];
        let block = [32, 1, 1];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_gate_up_split_gfx1151",
            bytes,
        );

        let mut gate_params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(GATE, grid, block, 0, &mut gate_params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(pp);
            b.push_ptr(ip);
            b.push_ptr(xp);
            b.push_ptr(ygp);
            b.push_i32(m_val);
            b.push_i32(k_val);
            b
        })?;

        let mut up_params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(UP, grid, block, 0, &mut up_params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(pp);
            b.push_ptr(ip);
            b.push_ptr(xp);
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
    /// `n_ranks` sets grid.y (the number of top-k ranks driven); see
    /// `gemv_hfq4g256_moe_gate_up_k8_indexed` for the uniform-vs-mixed contract.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_paro_q4g128_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] of u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        n_ranks: usize,
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
            [m as u32, n_ranks as u32, 1],
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
                [32, 1, 1],
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
                [32, 1, 1],
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
        use std::sync::OnceLock;
        static DOWN_CPOL_SLC: OnceLock<bool> = OnceLock::new();
        let cpol_slc = self.arch_caps.is_gfx1100()
            && *DOWN_CPOL_SLC.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_CPOL").as_deref() == Ok("slc")
            });
        static GFX1151_WEIGHT_BUFFER_LOADS: OnceLock<bool> = OnceLock::new();
        let gfx1151_buffer = self.arch_caps.is_gfx1151()
            && *GFX1151_WEIGHT_BUFFER_LOADS.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS").as_deref()
                    == Ok("1")
                    || hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_DOWN")
                        .as_deref()
                        == Ok("1")
            });
        static GFX1151_DOWN_HYBRID_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_hybrid_buffer = self.arch_caps.is_gfx1151()
            && k == 512
            && *GFX1151_DOWN_HYBRID_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_HYBRID_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_DOWN_ROW1_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_row1_buffer = self.arch_caps.is_gfx1151()
            && k == 512
            && *GFX1151_DOWN_ROW1_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_ROW1_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_DOWN_ROW2_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_row2_buffer = self.arch_caps.is_gfx1151()
            && k == 512
            && *GFX1151_DOWN_ROW2_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_ROW2_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_DOWN_ROW2_CLUSTERED: OnceLock<bool> = OnceLock::new();
        let gfx1151_row2_clustered = self.arch_caps.is_gfx1151()
            && k == 512
            && *GFX1151_DOWN_ROW2_CLUSTERED.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_ROW2_CLUSTERED").as_deref()
                    == Ok("1")
            });
        static GFX1151_DOWN_ROW8: OnceLock<bool> = OnceLock::new();
        let gfx1151_row8 = self.arch_caps.is_gfx1151()
            && k == 512
            && m % 8 == 0
            && *GFX1151_DOWN_ROW8.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_ROW8").as_deref() == Ok("1")
            });
        let (module_name, source, func_name) = if cpol_slc {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc_gfx1100",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_CPOL_SLC_GFX1100_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc",
            )
        } else if gfx1151_row8 {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW8_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151",
            )
        } else if gfx1151_hybrid_buffer {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_HYBRID_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151",
            )
        } else if gfx1151_row2_clustered {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW2_CLUSTERED_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151",
            )
        } else if gfx1151_row2_buffer {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW2_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151",
            )
        } else if gfx1151_row1_buffer {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW1_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151",
            )
        } else if gfx1151_buffer {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_BUFFER_GFX1151_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151",
            )
        } else {
            (
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
                kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
                "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            )
        };
        self.ensure_kernel(module_name, source, func_name)?;
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
        let timer = crate::profile::begin_timer(&self.hip, "gemv", func_name, bytes);
        // The expanded kernel owns four consecutive output rows per workgroup.
        // Keep this opt-in while it is qualified on gfx1100: the legacy launch
        // used `m` workgroups, leaving three quarters to exit at the row0 guard.
        static DOWN_TIGHT_GRID: OnceLock<bool> = OnceLock::new();
        let tight_grid = if self.arch_caps.is_gfx1100() {
            *DOWN_TIGHT_GRID.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_TIGHT_GRID").as_deref() == Ok("1")
            })
        } else if self.arch_caps.is_gfx1151() {
            // This kernel owns four rows per workgroup. Keep gfx1151's
            // measurement independent from gfx1100 even though the launch
            // contraction is semantically target-neutral.
            hipfire_config::developer_var("HIPFIRE_GFX1151_MOE_TIGHT_GRID").as_deref() == Ok("1")
                || hipfire_config::developer_var("HIPFIRE_GFX1151_DOWN_TIGHT_GRID").as_deref()
                    == Ok("1")
        } else {
            false
        };
        let grid_x = if gfx1151_row8 {
            (m as u32).div_ceil(8)
        } else if gfx1151_row1_buffer {
            m as u32
        } else if gfx1151_row2_buffer || gfx1151_row2_clustered {
            (m as u32).div_ceil(2)
        } else if tight_grid {
            (m as u32).div_ceil(4)
        } else {
            m as u32
        };
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, k_top as u32, batch_size as u32],
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

    /// Expert-wave-preserving down + deterministic combine experiment.
    ///
    /// The GEMV body and expanded stores are identical to the production
    /// kernel. A reusable counter tail appended to `expert_outputs` elects the
    /// last expert workgroup for each four-row tile; that wave folds the eight
    /// expanded slots in fixed rank order into `x_residual`. The caller must
    /// allocate and initially zero `batch_size * ceil(m/4)` f32-sized counter
    /// slots after the `[batch_size * k_top * m]` payload.
    /// Nine-path fused MoE gate_up (routed k=8, decode T=1): one CTA stages
    /// x into LDS once; 8 routed-expert warps share it. Replaces
    /// `gemv_hfq4g256_moe_gate_up_k8_indexed` — byte-exact per-row math.
    /// Grid: (mi/8, 1, 1); block 256 (8 warps × 32).
    pub fn gemv_hfq4g256_moe_ninepath_d3(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        mi: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_ninepath_d3",
            kernels::GEMV_HFQ4G256_MOE_NINEPATH_D3_SRC,
            "gemv_hfq4g256_moe_ninepath_d3",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let mi_val = mi as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * crate::profile::gemv_hfq4g256_bytes(2 * mi, k) + k * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_moe_ninepath_d3", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_ninepath_d3",
            [(mi as u32) / 8, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(mi_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Nine-path fused MoE down + weighted combine (routed k=8, decode T=1):
    /// folds the expanded [8 × down_m] intermediate in LDS and applies the
    /// k-ordered weighted fold in-kernel (no atomics). Replaces
    /// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    /// `moe_down_combine_k8_batched` — byte-exact at down_k=512.
    /// Grid: (down_m/16, 1, 1); block 256 (8 warps × 32).
    pub fn gemv_hfq4g256_moe_ninepath_d4(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        out: &GpuTensor,
        down_m: usize,
        down_k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_ninepath_d4",
            kernels::GEMV_HFQ4G256_MOE_NINEPATH_D4_SRC,
            "gemv_hfq4g256_moe_ninepath_d4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let xp = rot_batch.buf.as_ptr();
        let op = out.buf.as_ptr();
        let dm_val = down_m as i32;
        let dk_val = down_k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &dm_val as *const _ as *mut c_void,
            &dk_val as *const _ as *mut c_void,
        ];
        let bytes =
            8 * crate::profile::gemv_hfq4g256_bytes(down_m, down_k) + 8 * down_k * 4 + down_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_moe_ninepath_d4", bytes);
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_ninepath_d4",
            [(down_m as u32) / 16, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(xp);
                b.push_ptr(op);
                b.push_i32(dm_val);
                b.push_i32(dk_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_k8_indexed_last_combine(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        topk_weights: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
            kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_LAST_COMBINE_SRC,
            "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let twp = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &twp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
            bytes,
        );
        use std::sync::OnceLock;
        static DOWN_TIGHT_GRID: OnceLock<bool> = OnceLock::new();
        let grid_x = if self.arch_caps.is_gfx1100()
            && *DOWN_TIGHT_GRID.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_TIGHT_GRID").as_deref() == Ok("1")
            }) {
            (m as u32).div_ceil(4)
        } else {
            m as u32
        };
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
            [grid_x, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(rbp);
                b.push_ptr(eop);
                b.push_ptr(twp);
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

    /// Fused atomic-free MoE down: GEMV + K_TOP weighted-accumulate +
    /// residual add in ONE launch. Drop-in replacement for the two-kernel
    /// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` (writes
    /// `[N × K_TOP × M]`) FOLLOWED BY `moe_down_combine_k8_batched`
    /// (weighted-sums into `x_residual`). Each block owns one (token, row),
    /// loops all K_TOP experts internally, and does a single race-free
    /// `x_residual[token][row] += Σ_k weight[k]·down_k(row)` — no expanded
    /// intermediate, no atomicAdd. `x_residual` is accumulated in place
    /// (same `+=` contract as the combine it replaces). Wave32-only (RDNA);
    /// gated behind `HIPFIRE_MOE_DOWN_FUSED=1` at the dispatch layer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq4g256_moe_down_k8_indexed_fused_acc(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor, // [batch_size × m] f32, accumulated in place
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_hfq4g256_moe_down_k8_indexed_fused_acc",
            kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_FUSED_ACC_SRC,
            "gemv_hfq4g256_moe_down_k8_indexed_fused_acc",
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
        // Fused path skips the expanded [N×K_TOP×M] write + re-read; traffic
        // is the routed-expert weight reads plus the per-token residual write.
        let bytes =
            batch_size * k_top * crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq4g256_moe_down_k8_indexed_fused_acc",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq4g256_moe_down_k8_indexed_fused_acc",
            // NUM_ROWS=2 register row-tiling: each block owns 2 output rows (reuses the
            // per-expert X across them) -> ceil(M/2) blocks in x. Must match the kernel's
            // MOE_DOWN_FUSED_NUM_ROWS (2 — swept optimum for wave32 gfx1201, +3.5%).
            [((m + 1) / 2) as u32, batch_size as u32, 1],
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
    /// `n_ranks` sets grid.y (the number of top-k ranks driven); see
    /// `gemv_hfq4g256_moe_gate_up_k8_indexed` for the uniform-vs-mixed contract.
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
        n_ranks: usize,
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
            [m as u32, n_ranks as u32, 1],
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
    /// Index-aware MoE gate_up GEMV for HFQ5G256-layout routed experts.
    /// Wave32 (RDNA) only. Mirror of the HFQ6 sibling (168 B/group, 5-bit).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq5g256_moe_gate_up_k8_indexed(
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
            "gemv_hfq5g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ5G256_MOE_GATE_UP_INDEXED_SRC,
            "gemv_hfq5g256_moe_gate_up_k8_indexed",
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
        // HFQ5 uses 168 bytes/group vs HFQ4's 136. Bytes estimate scales
        // accordingly. Reuse the existing profile helper with a 168/136
        // ratio so timer estimates are roughly correct.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = 8 * (hfq4_bytes * 168 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq5g256_moe_gate_up_k8_indexed",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq5g256_moe_gate_up_k8_indexed",
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
    /// Merged per-expert MIXED-PRECISION MoE gate_up GEMV. Per-block branch on
    /// `dtype_tags[expert_id]` (0=MQ6 200B/group affine, 1=MQ2-Lloyd 72B/group
    /// codebook). Same kernarg signature + grid (M, K_TOP, N) + gate/up output
    /// split as the HFQ6 sibling, plus a leading `dtype_tags` u8 table.
    /// Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mixed_moe_gate_up_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor,
        dtype_tags: &GpuTensor,
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
            "gemv_mixed_moe_gate_up_indexed_batched",
            kernels::GEMV_MIXED_MOE_GATE_UP_INDEXED_BATCHED_SRC,
            "gemv_mixed_moe_gate_up_k8_indexed_batched",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let tp = dtype_tags.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // Mixed dtype: byte estimate uses the larger (MQ6 200 B/group) bound.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_mixed_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mixed_moe_gate_up_k8_indexed_batched",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(tp);
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
    /// HFQ5G256 batched gate_up MoE GEMV. Same kernarg signature + grid
    /// (M, K_TOP, N) + gate/up output split as the HFQ6 sibling; only the
    /// per-group dequant differs (168 B/group, 5-bit). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
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
            "gemv_hfq5g256_moe_gate_up_indexed_batched",
            kernels::GEMV_HFQ5G256_MOE_GATE_UP_INDEXED_BATCHED_SRC,
            "gemv_hfq5g256_moe_gate_up_k8_indexed_batched",
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
        // 168 vs 136 B/group: scale the HFQ4 byte estimate by 168/136.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 168 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq5g256_moe_gate_up_k8_indexed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq5g256_moe_gate_up_k8_indexed_batched",
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
    /// Merged per-expert MIXED-PRECISION expanded MoE down GEMV. Per-block branch
    /// on `dtype_tags[expert_id]` (0=MQ6 200B/group affine, 1=MQ2-Lloyd 72B/group
    /// codebook). Atomic-free expand-then-combine; pairs with
    /// `moe_down_combine_k8_batched` (dtype-independent f32 combine). Wave32
    /// (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mixed_moe_down_k8_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        dtype_tags: &GpuTensor,
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
            "gemv_mixed_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_MIXED_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            "gemv_mixed_moe_down_k8_indexed_batched_expanded",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let tp = dtype_tags.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let eop = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &eop as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // Mixed dtype: byte estimate uses the larger (MQ6 200 B/group) bound.
        let hfq4_bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let bytes = batch_size * k_top * (hfq4_bytes * 200 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_mixed_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mixed_moe_down_k8_indexed_batched_expanded",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(tp);
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
    /// HFQ5G256 counterpart to `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.
    /// Atomic-free expand-then-combine for the MoE down step. Pairs with
    /// `moe_down_combine_k8_batched` (dtype-independent). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
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
            "gemv_hfq5g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ5G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            "gemv_hfq5g256_moe_down_k8_indexed_batched_expanded",
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
        let bytes = batch_size * k_top * (hfq4_bytes * 168 / 136 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "gemv_hfq5g256_moe_down_k8_indexed_batched_expanded",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_hfq5g256_moe_down_k8_indexed_batched_expanded",
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
            let bytes = m * (k / 32) * 34 + k * 4 + m * 4;
            let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_q8_0_wide", bytes);
            let result = self.launch_maybe_blob(
                "gemv_q8_0_wide",
                [grid, 1, 1],
                [block_size, 1, 1],
                0,
                &mut params,
                blob_builder,
            );
            if let Some(t) = timer {
                t.finish(&self.hip);
            }
            return result;
        }

        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let block_size = 32u32;
        let bytes = m * (k / 32) * 34 + k * 4 + m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_q8_0", bytes);
        let result = self.launch_maybe_blob(
            "gemv_q8_0",
            [m as u32, 1, 1],
            [block_size, 1, 1],
            0,
            &mut params,
            blob_builder,
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
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

    /// BF16-weight × F32-input GEMV (native-bf16 reference / KLD oracle).
    /// Same shape contract as `gemv_f16_xf32`; the kernel widens bf16→f32
    /// losslessly (16-bit shift) per element. arch_id 12 (Cohere2-MoE).
    pub fn gemv_bf16_xf32(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_bf16_xf32",
            kernels::GEMV_BF16_XF32_SRC,
            "gemv_bf16_xf32",
        )?;

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
            "gemv_bf16_xf32",
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
        deepseek4_gfx1151_route: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // The accepted gfx1151 K8-all route is explicit model policy. Exact
        // gfx942 kernels live behind Gfx942Device; this generic method remains
        // portable on CDNA and for MiniMax/Qwen callers.
        let k8all = deepseek4_gfx1151_route && self.arch_caps.is_gfx1151() && k == 2048;
        let symbol = if k8all {
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
        } else {
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
        };
        self.ensure_kernel(
            symbol,
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            symbol,
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
            symbol,
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
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k8all(
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
        debug_assert_eq!(k, 2048);
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_indexed_batched_k8all",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K8ALL_SRC,
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed_batched",
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
        let bytes = batch_size * k_top * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k8all",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed_batched",
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
    /// Batched twin of `deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed`
    /// (MQ3-Lloyd, 3-bit + 8-entry codebook, 112 B/group). Enables minimax
    /// batched prefill / spec-verify on the mixed-tier `gate_up=MQ2 / down=MQ3`
    /// k-map. Mirrors the MQ2 `_batched_k4` wrapper; only the per-group stride
    /// (112 vs 72) and the kernel name differ.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
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
            "gemv_mq3g256_lloyd_moe_down_indexed_batched_k4",
            kernels::GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K4_SRC,
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
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
        let mq3_weight_bytes = m * (k / 256) * 112;
        let bytes = batch_size * k_top * (mq3_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed_batched_k4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
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
        // Exact gfx942 implementations are exposed only by Gfx942Device. This
        // shared API is the portable channel used by MiniMax and DS4 fallback.
        let logical_name = "gemv_mq2g256_lloyd_moe_gate_up_indexed";
        let symbol = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed";
        self.ensure_kernel(
            logical_name,
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            symbol,
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
            symbol,
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
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds(
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
        debug_assert_eq!(k, 4096);
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_BATCHED_K4096_LDS_SRC,
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds",
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
        let bytes = batch_size * k_top * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds",
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
        self.launch_maybe_blob(
            "wo_per_group_batched_hfq4g256",
            [m as u32, batch_size as u32, g as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(wp);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(g_i);
                blob.push_i32(m_i);
                blob.push_i32(k_i);
                blob.push_i32(bs);
                blob
            },
        )
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
        let use_wmma = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_WO_Q8_WMMA")
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
        // Untimed until now, like `gemm_q8_0_batched`: on CDNA3 the O-LoRA
        // Q8_0 tier lands here (the WMMA variant is gfx1151-gated), so
        // `HIPFIRE_PROFILE` was blind to one of the hottest weight paths.
        // Q8_0 packing is 34 B per 32-element group.
        let bytes = (g as usize)
            .saturating_mul(m as usize)
            .saturating_mul(k as usize)
            / 32
            * 34
            + (batch_size as usize)
                .saturating_mul(g as usize)
                .saturating_mul(k as usize)
                * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemv", "wo_per_group_batched_q8_0_1w", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
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
                ));
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
        // Exact gfx942 expansion lives behind Gfx942Device. The shared entry
        // point is intentionally portable for MiniMax and DS4 fallback.
        let logical_name = "gemv_mq2g256_lloyd_moe_down_expanded_k4";
        let source = kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_EXPANDED_K4_SRC;
        let symbol = "gemv_mq2g256_lloyd_moe_down_expanded_k4";
        self.ensure_kernel(logical_name, source, symbol)?;
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
        let timer = crate::profile::begin_timer(&self.hip, "gemv", logical_name, bytes);
        let result = self.launch_maybe_blob(
            symbol,
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
