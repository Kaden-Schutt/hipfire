// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 routed-expert decode executor (Task 10).
//!
//! This replaces step 4 (the per-expert routed loop) of
//! [`super::run_moe_decode_cpu_fallback`] for layers whose routed experts came
//! from an Escha-W2 checkpoint. Everything around it — the router GEMV, the
//! f16 logit rounding, top-k selection, and the shared expert — is unchanged
//! arch-6 code.
//!
//! # Why the routed loop needs replacing at all
//!
//! Escha's weights live in a ROTATED domain: the stored matrix is
//! `H·W·H` (up to the folded per-channel scales), so a matmul against it is
//! only the intended linear if the activation is Hadamard-transformed going
//! in and the result is Hadamard-transformed coming out. `escha_h128_in`
//! before the GEMV and `escha_h128_out` after are not a normalisation detail;
//! omitting them yields plausible-looking output that is wrong by ~1e-1
//! rather than ~1e-4.
//!
//! # Why it is phase-structured rather than a per-expert loop
//!
//! Task 8 measured the H128 kernels LAUNCH-bound: an empty kernel at the same
//! grid/block costs 1.74–1.78 us against a real launch's 2.4 us, and
//! overhead-subtracted time is nearly flat (0.59 → 0.69 us) from 16 to 136
//! blocks. A per-expert wiring costs `40 layers × 8 experts × 4 transforms =
//! 1280` launches/token = 3.07 ms = a **326 tok/s ceiling from the transforms
//! alone**, before any GEMV work. Running the token's `k` experts in phases —
//! all inputs transformed, then all GEMVs, then all outputs transformed — is
//! **4 H128-family launches per layer, 160 per token, 0.38 ms**.
//!
//! Phase order (per layer, decode, one token):
//!
//! | # | launches | what |
//! |---|----------|------|
//! | 1 | 1 | `escha_h128_in_batched` — gate_up input side, x broadcast |
//! | 2 | k | Q8_0 gate_up GEMV per selected expert |
//! | 3 | 1 | `escha_h128_out_batched` — gate_up output side |
//! | 4 | 1 | `escha_swiglu_batched` |
//! | 5 | 1 | `escha_h128_in_batched` — down input side, per-slot x |
//! | 6 | k | Q8_0 down GEMV per selected expert |
//! | 7 | 1 | `escha_h128_out_batched` — down output side |
//! | 8 | 1 | `moe_down_combine_k8_batched` with f16-rounded scores |
//!
//! The per-expert `rin_eff` / `rout_eff` rows are an extra index into the
//! already-resident `[E, IC]` / `[E, OC]` tensors — the batching is a kernel
//! indexing change plus a grid change, not new maths, and it is gated
//! bit-exactly against `escha_ref` by
//! `rdna-compute/examples/test_escha_h128_gpu_vs_cpu.rs`.

use rdna_compute::{Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::types::DispatchError;

/// Borrowed view of one layer's Escha-W2 transform tables plus the decode
/// scratch the phase structure needs. Built by the model
/// (`hipfire_arch_qwen35::qwen35::escha::EschaMoeTables::refs`); this crate
/// never owns or allocates any of it.
pub struct EschaRoutedRefs<'a> {
    /// `[n_exp, hidden]` f32 — folded `rin` for the gate_up projection.
    pub gate_up_rin: &'a GpuTensor,
    /// `[n_exp, 2*mi]` f32 — folded `rout` for gate_up. Carries the per-expert
    /// prune mask (zeros); see the zero contract in `escha_h128.hip`.
    pub gate_up_rout: &'a GpuTensor,
    /// `[n_exp, mi]` f32 — folded `rin` for the down projection.
    pub down_rin: &'a GpuTensor,
    /// `[n_exp, hidden]` f32 — folded `rout` for down.
    pub down_rout: &'a GpuTensor,
    /// `[k]` i32 — this token's selected expert ids (device).
    pub ids: &'a GpuTensor,
    /// `[k]` f32 — this token's combine weights, ALREADY f16-rounded.
    pub weights: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch.
    pub xh_gu: &'a GpuTensor,
    /// `[k, 2*mi]` f32 scratch.
    pub mid_gu: &'a GpuTensor,
    /// `[k, 2*mi]` f32 scratch.
    pub y_gu: &'a GpuTensor,
    /// `[k, mi]` f32 scratch.
    pub h: &'a GpuTensor,
    /// `[k, mi]` f32 scratch.
    pub xh_dn: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch.
    pub mid_dn: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch — the per-slot expert outputs the combine
    /// reduces.
    pub y_dn: &'a GpuTensor,
}

/// Number of H128-family launches this executor issues per call. Pinned as a
/// constant so the launch budget in the module docs is a checked claim rather
/// than a comment: `escha_launches_per_token(n_layers)` is what a decode step
/// actually costs, and `escha_routed_decode` asserts it in debug builds.
pub const ESCHA_H128_LAUNCHES_PER_LAYER: usize = 4;

/// H128 launches for a whole decode step. Independent of `k` — that is the
/// entire point of batching across experts.
pub fn escha_launches_per_token(n_layers: usize) -> usize {
    n_layers * ESCHA_H128_LAUNCHES_PER_LAYER
}

/// SAFETY: `src` is a device buffer of at least `offset_elems + len_elems`
/// f32; the returned view is non-owning and must not outlive `src`.
unsafe fn view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    let ptr = (src.buf.as_ptr() as *mut u8).add(offset_elems * 4);
    GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
        shape: vec![len_elems],
        dtype: rdna_compute::DType::F32,
    }
}

/// Run the routed half of one Escha-W2 MoE layer for one token.
///
/// `topk_ids` / `topk_weights` are host-side and already selected+renormalised
/// by the caller (production: `run_moe_decode_cpu_fallback`; the G4 gate
/// injects EschaLabs' shipped fixture instead, which is why this boundary is
/// public). `out` is accumulated into, never overwritten.
///
/// The combine multiplies by **`f16(score)`** — one of the three load-bearing
/// rounding points of the format. It is applied here, on the host copy, so the
/// caller's `topk_weights` device buffer is left untouched for any other
/// consumer (e.g. `capture_expert_stats`).
#[allow(clippy::too_many_arguments)]
pub fn escha_routed_decode(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    e: &EschaRoutedRefs<'_>,
    routed_experts: &[(WeightRef<'_>, WeightRef<'_>)],
    topk_ids: &[usize],
    topk_weights: &[f32],
    x_norm: &GpuTensor,
    out: &GpuTensor,
    hidden: usize,
    mi: usize,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($ex:expr) => {
            $ex.map_err(|err| DispatchError::Hip(err.to_string()))
        };
    }
    let k = topk_ids.len();
    // `moe_down_combine_k8_batched` unrolls to a hard 8 slots (`k < K_TOP`
    // guard inside a `for k in 0..8`), so it silently drops slots 8.. rather
    // than failing. Every escha SKU is k=8; reject anything else loudly here
    // instead of returning a quietly truncated sum.
    if k > 8 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-decode-supports-k<=8",
            arch: "",
            quant: "",
        });
    }
    if k == 0 || k != topk_weights.len() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-topk-id-weight-length-mismatch",
            arch: "",
            quant: "",
        });
    }
    for &id in topk_ids {
        if id >= routed_experts.len() {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "escha-topk-id-out-of-range",
                arch: "",
                quant: "",
            });
        }
    }

    // ids + f16-rounded combine weights -> device.
    let ids_i32: Vec<i32> = topk_ids.iter().map(|&i| i as i32).collect();
    let id_bytes: Vec<u8> = ids_i32.iter().flat_map(|v| v.to_le_bytes()).collect();
    hip!(gpu.hip.memcpy_htod(&e.ids.buf, &id_bytes))?;
    let w_bytes: Vec<u8> = topk_weights
        .iter()
        .map(|&w| f32::from(half::f16::from_f32(w)))
        .flat_map(|w| w.to_le_bytes())
        .collect();
    hip!(gpu.hip.memcpy_htod(&e.weights.buf, &w_bytes))?;

    // ── 1. gate_up input transform, all k slots, ONE launch ───────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        x_norm,
        e.gate_up_rin,
        e.ids,
        e.xh_gu,
        hidden,
        k,
        false,
    ))?;

    // ── 2. gate_up GEMV per selected expert ───────────────────────────────
    static GEMV: std::sync::OnceLock<GemvFamily> = std::sync::OnceLock::new();
    let gemv = GEMV.get_or_init(GemvFamily::new);
    for (s, &id) in topk_ids.iter().enumerate() {
        let x = unsafe { view(e.xh_gu, s * hidden, hidden) };
        let y = unsafe { view(e.mid_gu, s * 2 * mi, 2 * mi) };
        gemv.run_auto(ctx, gpu, &routed_experts[id].0, &x, &y)?;
    }

    // ── 3. gate_up output transform, ONE launch ───────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        e.mid_gu,
        e.gate_up_rout,
        e.ids,
        e.y_gu,
        2 * mi,
        k,
        true,
    ))?;

    // ── 4. SwiGLU on the f16-rounded merged output, gate = FIRST half ─────
    hip!(gpu.escha_swiglu_batched(e.y_gu, e.h, mi, k))?;

    // ── 5. down input transform, ONE launch (per-slot activation) ─────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        e.h,
        e.down_rin,
        e.ids,
        e.xh_dn,
        mi,
        k,
        true,
    ))?;

    // ── 6. down GEMV per selected expert ──────────────────────────────────
    for (s, &id) in topk_ids.iter().enumerate() {
        let x = unsafe { view(e.xh_dn, s * mi, mi) };
        let y = unsafe { view(e.mid_dn, s * hidden, hidden) };
        gemv.run_auto(ctx, gpu, &routed_experts[id].1, &x, &y)?;
    }

    // ── 7. down output transform, ONE launch ──────────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        e.mid_dn,
        e.down_rout,
        e.ids,
        e.y_dn,
        hidden,
        k,
        true,
    ))?;

    // ── 8. weighted combine into the residual, ONE launch ─────────────────
    hip!(gpu.moe_down_combine_k8_batched(e.y_dn, e.weights, out, hidden, k, 1))?;
    Ok(())
}
