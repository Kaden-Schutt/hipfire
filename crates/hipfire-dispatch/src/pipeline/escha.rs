// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 routed-expert decode executors (Tasks 10 and 11).
//!
//! Two entry points, one for each of the two routes into a routed MoE layer.
//! Both apply the SAME phase structure and the SAME H128 pair; they differ
//! only in where the routing lives.
//!
//! * [`escha_routed_decode`] — CPU-top-K route. `topk_ids` / `topk_weights`
//!   arrive on the HOST, already downloaded by
//!   [`super::run_moe_decode_cpu_fallback`]. Used when the layer is not
//!   admitted to the indexed path (k != 8, F32 control experts in the G4
//!   gate, …).
//! * [`escha_routed_decode_indexed`] — GPU-top-K route, the production
//!   decode/prefill path. Routing stays on the device end to end: the ids
//!   feed the transforms and the GEMVs as a device buffer, the combine
//!   weights are f16-rounded by a device kernel, and there is no D2H
//!   anywhere in the layer.
//!
//! Everything around them — the router GEMV, the f16 logit rounding, top-k
//! selection, and the shared expert — is unchanged arch-6 code.
//!
//! # Why the indexed route exists (Task 11, measured)
//!
//! `rocprofv3 --kernel-trace` over a warm 40-layer decode token on gfx1151:
//! 22.0 ms of GPU-busy against a 29.7 ms wall (profiled; 26.0 ms unprofiled),
//! of which the routed experts were 6.5 ms of kernel time, ~2.1 ms of
//! launch-gap across their 640 per-expert GEMV launches, and ~2.1 ms of
//! copy/stall around the per-layer `topk_indices` + `topk_weights` D2H and the
//! ids/weights H2D that followed it. The host round trip alone did NOT
//! dominate — but the round trip and the launch storm together were the whole
//! addressable overhead, and the round trip is also what made the layer
//! non-capturable under hipGraph. This route removes both: 640 routed GEMV
//! launches per token become 80, and the per-layer sync disappears.
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
//! Phase order (per layer, decode, one token). The `host` column is
//! [`escha_routed_decode`], `idx` is [`escha_routed_decode_indexed`]:
//!
//! | # | host | idx | what |
//! |---|------|-----|------|
//! | 0 | 0 (host-side rounding) | 1 | `escha_round_weights_f16_rne` |
//! | 1 | 1 | 1 | `escha_h128_in_batched` — gate_up input side, x broadcast |
//! | 2 | k | 1 | Q8_0 gate_up GEMV (per expert / all experts indexed) |
//! | 3 | 1 | 1 | `escha_h128_out_batched` — gate_up output side |
//! | 4 | 1 | 1 | `escha_swiglu_batched` |
//! | 5 | 1 | 1 | `escha_h128_in_batched` — down input side, per-slot x |
//! | 6 | k | 1 | Q8_0 down GEMV (per expert / all experts indexed) |
//! | 7 | 1 | 1 | `escha_h128_out_batched` — down output side |
//! | 8 | 1 | 1 | `moe_down_combine_k8_batched` with f16-rounded scores |
//!
//! At k=8 that is 22 launches + 2 H2D + (2 D2H upstream) per layer on the
//! host route against 9 launches and no host transfer at all on the indexed
//! one. The H128 budget — 4 per layer — is identical on both, which is why
//! `escha_launches_per_token` and the gates that read it are route-agnostic.
//!
//! The per-expert `rin_eff` / `rout_eff` rows are an extra index into the
//! already-resident `[E, IC]` / `[E, OC]` tensors — the batching is a kernel
//! indexing change plus a grid change, not new maths, and it is gated
//! bit-exactly against `escha_ref` by
//! `rdna-compute/examples/test_escha_h128_gpu_vs_cpu.rs`.

use rdna_compute::{EschaXGroup, Gpu, GpuTensor};

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
/// than a comment.
///
/// It is checked by the GATES, not by this function: `escha_routed_decode`
/// itself contains no assert (an earlier version of this doc claimed a
/// debug-build assert that never existed). The claim is enforced by reading
/// `rdna_compute::escha_h128_launches()` across a real forward pass —
/// `hipfire-arch-qwen35/examples/escha_moe_block_gate.rs` (one layer, per
/// (layer, token)) and `examples/escha_model_smoke.rs` (whole model, decode
/// AND prefill, against `escha_launches_per_token`).
///
/// This is an H128-TRANSFORM budget, not the layer's whole launch cost: a
/// decode layer at k=8 issues 22 launches in total (4 H128 + 1 SwiGLU +
/// 1 combine + 16 GEMV). See the phase table in the module docs.
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
        EschaXGroup::Broadcast,
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
        EschaXGroup::PerSlot,
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
        EschaXGroup::PerSlot,
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
        EschaXGroup::PerSlot,
    ))?;

    // ── 8. weighted combine into the residual, ONE launch ─────────────────
    hip!(gpu.moe_down_combine_k8_batched(e.y_dn, e.weights, out, hidden, k, 1))?;
    Ok(())
}

/// Device-resident routing for [`escha_routed_decode_indexed`].
///
/// These are the buffers the GPU top-K kernel wrote. Nothing here is ever
/// read by the host: that is the point of the route.
pub struct EschaIndexedRouting<'a> {
    /// `[n_exp]` u64 weight-base pointers for the Q8_0 gate_up slots, packed
    /// into an F32 tensor (2 f32 per pointer) — the same table the other
    /// indexed MoE GEMVs consume.
    pub expert_gate_up_ptrs: &'a GpuTensor,
    /// `[n_exp]` u64 weight-base pointers for the Q8_0 down slots.
    pub expert_down_ptrs: &'a GpuTensor,
    /// `[k]` selected expert ids as i32 (stored in an F32 tensor — same
    /// 4 B/elem — exactly as `moe_topk_renorm_k8` and friends write them).
    pub topk_indices: &'a GpuTensor,
    /// `[k]` combine weights, NOT yet f16-rounded. Left untouched; the
    /// rounded copy goes into the layer's own `weights` scratch.
    pub topk_weights: &'a GpuTensor,
    /// Rows of the gate_up weight matrix (`2 * mi`).
    pub gate_up_m: usize,
    /// Columns of the gate_up weight matrix (`hidden`).
    pub gate_up_k: usize,
    /// Rows of the down weight matrix (`hidden`).
    pub down_m: usize,
    /// Columns of the down weight matrix (`mi`).
    pub down_k: usize,
}

/// Run the routed half of one Escha-W2 MoE layer for one token, with the
/// routing left on the device.
///
/// Same eight phases as [`escha_routed_decode`] and the same H128 pair; the
/// only differences are that phases 2 and 6 are ONE indexed GEMV launch each
/// instead of `k`, and that the f16 rounding of the combine weights happens
/// in a kernel rather than on the host copy.
///
/// `out` is accumulated into, never overwritten.
///
/// # Why this is not just "the indexed path with escha bolted on"
///
/// The generic indexed routed body in [`super::run_moe_decode`] is not
/// reachable for escha and must never become reachable: it feeds the raw
/// activation straight into the expert GEMVs and combines the raw result, so
/// it would omit both Hadamard transforms and emit finite, fluent output
/// wrong by ~1e-1. `run_moe_decode` therefore branches to THIS function
/// before that body, and `check_moe_decode_supported` refuses any escha layer
/// that reaches the indexed path without the transform tables that make this
/// function callable at all.
#[allow(clippy::too_many_arguments)]
pub fn escha_routed_decode_indexed(
    gpu: &mut Gpu,
    e: &EschaRoutedRefs<'_>,
    r: &EschaIndexedRouting<'_>,
    out: &GpuTensor,
    x_norm: &GpuTensor,
    hidden: usize,
    mi: usize,
    k: usize,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($ex:expr) => {
            $ex.map_err(|err| DispatchError::Hip(err.to_string()))
        };
    }
    // Same hard k<=8 bound as the host route: `moe_down_combine_k8_batched`
    // unrolls to 8 slots and silently DROPS the rest rather than failing.
    if k == 0 || k > 8 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-decode-supports-k<=8",
            arch: "",
            quant: "",
        });
    }
    // The GEMVs read `expert_ptrs[topk_indices[slot]]` on device, so a
    // mis-sized weight table is an out-of-bounds READ (undefined behaviour),
    // not a wrong answer. Both projections' shapes must also agree with the
    // scratch the transforms were sized for — a mismatch there would have the
    // GEMV write past the end of `mid_gu` / `mid_dn`.
    if r.gate_up_m != 2 * mi || r.gate_up_k != hidden || r.down_m != hidden || r.down_k != mi {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-indexed-routed-shape-mismatch",
            arch: "",
            quant: "",
        });
    }

    // ── 0. f16-round the combine weights, ONE launch, out-of-place ────────
    // `f16(score)` is one of the three load-bearing rounding points of the
    // format. The host route does this on the downloaded copy; here it is a
    // kernel, writing the layer's own scratch so `topk_weights` stays intact
    // for any other consumer.
    hip!(gpu.escha_round_weights_f16_rne(r.topk_weights, e.weights, k))?;

    // ── 1. gate_up input transform, all k slots, ONE launch ───────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        x_norm,
        e.gate_up_rin,
        r.topk_indices,
        e.xh_gu,
        hidden,
        k,
        EschaXGroup::Broadcast,
    ))?;

    // ── 2. gate_up GEMV for ALL k experts, ONE launch ─────────────────────
    hip!(gpu.escha_gemv_q8_0_moe_k8_indexed_batched(
        r.expert_gate_up_ptrs,
        r.topk_indices,
        e.xh_gu,
        e.mid_gu,
        r.gate_up_m,
        r.gate_up_k,
        k,
    ))?;

    // ── 3. gate_up output transform, ONE launch ───────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        e.mid_gu,
        e.gate_up_rout,
        r.topk_indices,
        e.y_gu,
        2 * mi,
        k,
        EschaXGroup::PerSlot,
    ))?;

    // ── 4. SwiGLU on the f16-rounded merged output, gate = FIRST half ─────
    hip!(gpu.escha_swiglu_batched(e.y_gu, e.h, mi, k))?;

    // ── 5. down input transform, ONE launch (per-slot activation) ─────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        e.h,
        e.down_rin,
        r.topk_indices,
        e.xh_dn,
        mi,
        k,
        EschaXGroup::PerSlot,
    ))?;

    // ── 6. down GEMV for ALL k experts, ONE launch ────────────────────────
    hip!(gpu.escha_gemv_q8_0_moe_k8_indexed_batched(
        r.expert_down_ptrs,
        r.topk_indices,
        e.xh_dn,
        e.mid_dn,
        r.down_m,
        r.down_k,
        k,
    ))?;

    // ── 7. down output transform, ONE launch ──────────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        e.mid_dn,
        e.down_rout,
        r.topk_indices,
        e.y_dn,
        hidden,
        k,
        EschaXGroup::PerSlot,
    ))?;

    // ── 8. weighted combine into the residual, ONE launch ─────────────────
    hip!(gpu.moe_down_combine_k8_batched(e.y_dn, e.weights, out, hidden, k, 1))?;
    Ok(())
}

/// Run the routed half of one Escha-W2 MoE layer for a BATCH of `n_tokens`,
/// with the routing left on the device.
///
/// The batched-prefill twin of [`escha_routed_decode_indexed`]. Same eight
/// phases, same H128 pair, same kernels — the ONLY change is that `slots` is
/// `n_tokens * k` instead of `k`.
///
/// # Why that is enough, and why it is bit-identical per slot
///
/// Every kernel in the pipeline is already purely slot-parallel:
///
/// * `escha_gemv_q8_0_moe_k8_indexed_batched` (and its wide sibling) take
///   `krank = blockIdx.y`, read `x_batch + krank*K`, write `y_batch + krank*M`
///   and address the expert through `expert_ptrs[topk_indices[krank]]`. No
///   loop bound, accumulator, unroll or reduction depends on how many slots
///   there are, so slot `s` computes exactly the same sum in exactly the same
///   order whether the launch carried 8 slots or 2 048.
/// * `escha_h128_out_batched`, `escha_swiglu_batched` and
///   `moe_down_combine_k8_batched` are already `[slots, ...]` /
///   `[N, K_TOP, M]`-shaped and need nothing.
/// * `escha_h128_in_batched` needed the one change: its input-side activation
///   was either broadcast to every slot or one row per slot, and batched
///   prefill needs one row per TOKEN shared by that token's `k` slots. Hence
///   [`EschaXGroup::Grouped`].
///
/// That per-slot invariance is the reason the routed half of batched prefill
/// is asserted EQUAL to the per-token route rather than close to it — see the
/// gate in `hipfire-arch-qwen35/examples/escha_prefill_batch_gate.rs`. (The
/// DENSE half is not bit-identical: a batched WMMA GEMM does not accumulate
/// like a batch-1 GEMV.)
///
/// # Slot layout
///
/// Token-major: slot `s` is token `s / k`, rank `s % k`. This is the layout
/// `moe_topk_renorm_k8_batched` already writes `topk_indices` / `topk_weights`
/// in, and the layout `moe_down_combine_k8_batched` already reads
/// `expert_outputs` in, so nothing is permuted anywhere in this function.
///
/// `out` (`[n_tokens, hidden]`) is accumulated into, never overwritten.
#[allow(clippy::too_many_arguments)]
pub fn escha_routed_prefill_indexed(
    gpu: &mut Gpu,
    tables: &EschaRoutedRefs<'_>,
    scratch: &rdna_compute::scratch::EschaPrefillViews,
    r: &EschaIndexedRouting<'_>,
    out: &GpuTensor,
    x_norm_batch: &GpuTensor,
    hidden: usize,
    mi: usize,
    k: usize,
    n_tokens: usize,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($ex:expr) => {
            $ex.map_err(|err| DispatchError::Hip(err.to_string()))
        };
    }
    // `moe_down_combine_k8_batched` unrolls to a hard 8 slots per token
    // (`k < K_TOP` guard inside a `for k in 0..8`), so it silently DROPS ranks
    // 8.. rather than failing. Same bound as both decode routes.
    if k == 0 || k > 8 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-prefill-supports-k<=8",
            arch: "",
            quant: "",
        });
    }
    if n_tokens == 0 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-prefill-empty-batch",
            arch: "",
            quant: "",
        });
    }
    let slots = n_tokens
        .checked_mul(k)
        .ok_or(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-prefill-slot-overflow",
            arch: "",
            quant: "",
        })?;
    // The indexed GEMVs put `slots` on grid.y, which HIP caps at 65 535. Above
    // that the launch would be truncated, not rejected — tokens past the cap
    // would silently contribute nothing to the residual, which reads as a
    // mildly worse model rather than as a failure.
    if slots > 65_535 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-routed-prefill-slots-exceed-grid-y",
            arch: "",
            quant: "",
        });
    }
    // The scratch is model-global and grows on demand, and its views are cut
    // to a specific slot count; a caller that cut them for a different chunk
    // would have every kernel here read or write the wrong extent. The
    // wrappers also length-check exactly, so this is belt and braces — but it
    // names the mistake instead of reporting a numel mismatch six frames down.
    if scratch.slots != slots {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-prefill-scratch-slot-mismatch",
            arch: "",
            quant: "",
        });
    }
    // Same shape contract as the decode route: the GEMVs read
    // `expert_ptrs[topk_indices[slot]]` on device, so a mis-sized weight table
    // is an out-of-bounds READ, not a wrong answer, and a projection whose
    // shape disagrees with the scratch would have the GEMV write past the end
    // of `mid_gu` / `mid_dn`.
    if r.gate_up_m != 2 * mi || r.gate_up_k != hidden || r.down_m != hidden || r.down_k != mi {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "escha-indexed-routed-shape-mismatch",
            arch: "",
            quant: "",
        });
    }

    let weights = &scratch.weights;
    let xh_gu = &scratch.xh_gu;
    let mid_gu = &scratch.mid_gu;
    let y_gu = &scratch.y_gu;
    let h = &scratch.h;
    let xh_dn = &scratch.xh_dn;
    let mid_dn = &scratch.mid_dn;
    let y_dn = &scratch.y_dn;
    // `topk_indices` / `topk_weights` are the [max_batch x k] prefill scratch;
    // only the first `slots` entries are this chunk's.
    let ids = r.topk_indices.sub_offset(0, slots);
    let raw_weights = r.topk_weights.sub_offset(0, slots);

    // ── 0. f16-round the combine weights, ONE launch, out-of-place ────────
    hip!(gpu.escha_round_weights_f16_rne(&raw_weights, weights, slots))?;

    // ── 1. gate_up input transform, all slots, ONE launch ─────────────────
    // Grouped(k): slot s reads token s/k's row of `x_norm_batch`. This is the
    // one place batched prefill differs from decode, which broadcasts a single
    // row to all k slots.
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        x_norm_batch,
        tables.gate_up_rin,
        &ids,
        xh_gu,
        hidden,
        slots,
        EschaXGroup::Grouped(k),
    ))?;

    // ── 2. gate_up GEMV for ALL slots, ONE launch ─────────────────────────
    hip!(gpu.escha_gemv_q8_0_moe_k8_indexed_batched(
        r.expert_gate_up_ptrs,
        &ids,
        xh_gu,
        mid_gu,
        r.gate_up_m,
        r.gate_up_k,
        slots,
    ))?;

    // ── 3. gate_up output transform, ONE launch ───────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        mid_gu,
        tables.gate_up_rout,
        &ids,
        y_gu,
        2 * mi,
        slots,
        EschaXGroup::PerSlot,
    ))?;

    // ── 4. SwiGLU on the f16-rounded merged output, gate = FIRST half ─────
    hip!(gpu.escha_swiglu_batched(y_gu, h, mi, slots))?;

    // ── 5. down input transform, ONE launch (per-slot activation) ─────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_in_batched",
        h,
        tables.down_rin,
        &ids,
        xh_dn,
        mi,
        slots,
        EschaXGroup::PerSlot,
    ))?;

    // ── 6. down GEMV for ALL slots, ONE launch ────────────────────────────
    hip!(gpu.escha_gemv_q8_0_moe_k8_indexed_batched(
        r.expert_down_ptrs,
        &ids,
        xh_dn,
        mid_dn,
        r.down_m,
        r.down_k,
        slots,
    ))?;

    // ── 7. down output transform, ONE launch ──────────────────────────────
    hip!(gpu.escha_h128_batched(
        "escha_h128_out_batched",
        mid_dn,
        tables.down_rout,
        &ids,
        y_dn,
        hidden,
        slots,
        EschaXGroup::PerSlot,
    ))?;

    // ── 8. weighted combine into the residual, ONE launch ─────────────────
    // `[n_tokens, k, hidden]` folded into `[n_tokens, hidden]`; blockIdx.y is
    // already the token, so this is the same kernel decode uses at n = 1.
    hip!(gpu.moe_down_combine_k8_batched(y_dn, weights, out, hidden, k, n_tokens))?;
    Ok(())
}
