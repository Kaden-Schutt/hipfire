// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![cfg(feature = "new-dispatch")]

//! Demonstration: wiring the [`GemvFamily`] into the LLaMA forward pass.
//!
//! Replaces the inline `weight_gemv` / `weight_gemv_prerotated` calls
//! from `forward_scratch_layers` with dispatch-crate `GemvFamily::run`.
//!
//! ## Pattern
//!
//! Each [`LayerWeights`] field (wq, wk, wv, wo, w_gate, w_up, w_down)
//! is wrapped in a [`WeightRef`], then [`GemvFamily::run`] selects the
//! correct kernel by (DType, variant, arch caps).
//!
//! ## Coverage
//!
//! | Projection | Variant |
//! |---|---|
//! | Q / K / V  | [`GemvVariant::Plain`] |
//! | O          | [`GemvVariant::Plain`] |
//! | Gate / Up  | [`GemvVariant::Plain`] |
//! | Down       | [`GemvVariant::WithSwiGLUResidual`] |
//!
//! For MQ-family quants `Plain` returns `Err(DispatchError)` — those
//! formats need `Prerotated`. A production integration would add a
//! prerotated path (and the outer rotation call) for MQ weights. This
//! module establishes *the call pattern only*; the actual forward still
//! routes through `llama::forward_scratch_layers`.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, GemvParams, WeightRef};
use hipfire_dispatch::types::{DispatchError, GemvVariant};
use hipfire_runtime::llama::{ForwardScratch, LayerWeights};
use rdna_compute::Gpu;

/// Per-layer GEMV dispatch via [`GemvFamily`].
///
/// Mirrors the GEMV portion of one layer iteration in
/// `llama::forward_scratch_layers`. All non-residual projections
/// use [`GemvVariant::Plain`]; the FFN-down projection uses
/// [`GemvVariant::WithSwiGLUResidual`] with the caller-supplied
/// `silu(gate)·up` fused input.
///
/// The caller is responsible for:
/// - RMSNorm before this call (writes `scratch.tmp`)
/// - Attention + KV-cache between Q/K/V and O
/// - `silu_mul_f32` before the Down call (fills `scratch.ffn_hidden`)
/// - Logits head GEMV after the layer loop
pub fn forward_scratch_layer_gemv(
    ctx: &DispatchCtx,
    gemv: &GemvFamily,
    gpu: &mut Gpu,
    layer: &LayerWeights,
    scratch: &ForwardScratch,
) -> Result<(), DispatchError> {
    // ── Q attend:  q = Wq · tmp ───────────────────────
    {
        let w = WeightRef {
            buf: &layer.wq.buf,
            dtype: layer.wq.gpu_dtype,
            m: layer.wq.m,
            k: layer.wq.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.tmp,
            y: &scratch.q,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── K attend:  k = Wk · tmp ───────────────────────
    {
        let w = WeightRef {
            buf: &layer.wk.buf,
            dtype: layer.wk.gpu_dtype,
            m: layer.wk.m,
            k: layer.wk.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.tmp,
            y: &scratch.k,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── V attend:  v = Wv · tmp ───────────────────────
    {
        let w = WeightRef {
            buf: &layer.wv.buf,
            dtype: layer.wv.gpu_dtype,
            m: layer.wv.m,
            k: layer.wv.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.tmp,
            y: &scratch.v,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── O attend:  o = Wo · attn_out ─────────────────
    {
        let w = WeightRef {
            buf: &layer.wo.buf,
            dtype: layer.wo.gpu_dtype,
            m: layer.wo.m,
            k: layer.wo.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.attn_out,
            y: &scratch.o,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── Gate FFN:  gate = Wgate · tmp ─────────────────
    {
        let w = WeightRef {
            buf: &layer.w_gate.buf,
            dtype: layer.w_gate.gpu_dtype,
            m: layer.w_gate.m,
            k: layer.w_gate.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.tmp,
            y: &scratch.gate,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── Up FFN:    up = Wup · tmp ─────────────────────
    {
        let w = WeightRef {
            buf: &layer.w_up.buf,
            dtype: layer.w_up.gpu_dtype,
            m: layer.w_up.m,
            k: layer.w_up.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.tmp,
            y: &scratch.up,
            variant: GemvVariant::Plain,
            residual: None,
            gate: None,
            up: None,
        })?;
    }

    // ── Down FFN (fused SwiGLU + residual) ────────────
    //
    // WithSwiGLUResidual contract (HFQ dtypes):
    //   `gate` = pre-computed silu(gate)·up  (scratch.ffn_hidden)
    //   `residual` = stream to add into       (scratch.x)
    //   Kernel does:  residual += W · gate_input
    {
        let w = WeightRef {
            buf: &layer.w_down.buf,
            dtype: layer.w_down.gpu_dtype,
            m: layer.w_down.m,
            k: layer.w_down.k,
        };
        gemv.run(ctx, gpu, &GemvParams {
            w: &w,
            x: &scratch.ffn_hidden,
            y: &scratch.ffn_out,
            variant: GemvVariant::WithSwiGLUResidual,
            gate: Some(&scratch.ffn_hidden),
            residual: Some(&scratch.x),
            up: None,
        })?;
    }

    Ok(())
}
