// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dispatch coverage guardrail — catches the two recurring "missing dispatch arm"
//! defect classes at CI time, GPU-FREE (no kernels, no device, no GPU lock).
//!
//! Both defects that have already shipped on this branch reduce to a pure assertion
//! over the existing dispatch API:
//!
//!   1. KEY-CONSTRUCTOR GAP — a `KernelKey::for_*` constructor's `_ => UnsupportedVariant`
//!      arm is hit by a dtype a shipped model actually uses for that op, so the
//!      forward pass `.unwrap()`s an Err and HARD-PANICS on decode.
//!      Live example: `for_gemv_residual(Q8_0)` == Err  → qwen3.5-9b.q8f16 and
//!      qwen3.6-35b-a3b o_proj panic ("unsupported gemv.residual for /").
//!
//!   2. ARCH DEAD-GATE — a dtype's required `ArchPredicate` excludes an arch the
//!      model ships on, so `resolve()` returns MissingImpl / the path silently
//!      falls to a slow scalar kernel. Live example (fixed at 953ea648): MQ3/MQ6
//!      gated on a gfx11-only predicate, excluding gfx1201/RDNA4.
//!
//! Keep `FLEET` in sync with the model loaders' per-op weight dtypes: a new quant
//! format or shipped tier means new rows here. This is the structural fix #397
//! Phase-0.4 should adopt — a single coverage gate over (op × dtype × arch).

use crate::context::DispatchCtx;
use crate::types::*;
use rdna_compute::DType::{self, *};

/// The dispatch entry a forward pass reaches for a given weight role.
#[derive(Clone, Copy, Debug)]
enum Role {
    /// qkv / gate_up / lm_head — plain GEMV (rotation handled inside).
    Plain,
    /// o_proj — fused residual GEMV `y += W·x` (`Step::GemvResidual`).
    Residual,
    /// FFN down — fused `y += W·silu(gate·up)` (`weight_gemv_swiglu_residual`).
    SwigluResidual,
}

/// One (shipped model, weight role, dtype) the live forward pass exercises,
/// plus the archs that tier actually ships on.
struct OpUse {
    model: &'static str,
    role: Role,
    dtype: DType,
    archs: &'static [&'static str],
}

/// gfx that run wave32 WMMA-class quants — the interesting coverage surface.
const WAVE32: &[&str] = &[
    "gfx1100", "gfx1101", "gfx1102", // RDNA3 dGPU
    "gfx1150", "gfx1151", "gfx1152", // RDNA3.5 APU
    "gfx1200", "gfx1201",            // RDNA4
];
/// Everything incl. RDNA1/2 + CDNA, for dtypes whose arch gate is Always/dp4a.
const ALL: &[&str] = &[
    "gfx1010", "gfx1030", "gfx1031", "gfx1032",
    "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152",
    "gfx1200", "gfx1201", "gfx906", "gfx908", "gfx942",
];

/// PRODUCTION MATRIX — (model, role, dtype) the wired forward pass hits today.
/// The Q8_0 `Residual` rows are the ones the live gap panics on.
const FLEET: &[OpUse] = &[
    // ── q8f16: Q8 weights throughout. o_proj PANICS on every arch today ──
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::Residual, dtype: Q8_0, archs: ALL },
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::Plain,    dtype: Q8_0, archs: ALL },
    // FFN down is on the legacy path today, but hits swiglu_residual once migrated:
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::SwigluResidual, dtype: Q8_0, archs: ALL },

    // ── qwen3.6-35b-a3b MoE: Q8 attention o_proj (also PANICS today) ──
    OpUse { model: "qwen3.6-35b-a3b.mq4", role: Role::Residual, dtype: Q8_0, archs: WAVE32 },

    // ── dense MQ4/MQ3/Lloyd: o_proj IS supported — regression anchors (stay green) ──
    OpUse { model: "qwen3.5-27b.mq4",       role: Role::Residual, dtype: MQ4G256,      archs: WAVE32 },
    OpUse { model: "qwen3.5-27b.mq3",       role: Role::Residual, dtype: MQ3G256,      archs: WAVE32 },
    OpUse { model: "qwen3.6-27b.mq3-lloyd", role: Role::Residual, dtype: MQ3G256Lloyd, archs: WAVE32 },
    OpUse { model: "qwen3.6-35b-a3b.mq4",   role: Role::Plain,    dtype: MQ4G256,      archs: WAVE32 },
    // MQ6-promoted projections (A3B AWQ-attractor mitigation) — gate is HasMmq (must admit gfx12):
    OpUse { model: "qwen3.6-35b-a3b.mq4",   role: Role::Plain,    dtype: MQ6G256,      archs: WAVE32 },
];

/// Map (role, dtype) to the KernelKey constructor the forward pass calls.
fn construct_key(role: Role, dtype: DType) -> Result<KernelKey, DispatchError> {
    match role {
        Role::Plain          => KernelKey::for_gemv(dtype, GemvVariant::Plain, false),
        Role::Residual       => KernelKey::for_gemv_residual(dtype),
        Role::SwigluResidual => KernelKey::for_gemv_swiglu_residual(dtype),
    }
}

/// LAYER 1 — key-constructor coverage (catches the Q8-residual defect class).
/// Every (role, dtype) a shipped model uses MUST construct a key, never hit the
/// `_ => UnsupportedVariant` arm. FAILS TODAY on the q8f16 / A3B `Residual` Q8_0
/// rows — that failure IS the bug. Fix = add Q8_0 (+ any other shipped o_proj/down
/// dtype) to for_gemv_residual / for_gemv_swiglu_residual backed by a real
/// gemv_q8_0_residual kernel, OR keep o_proj on the legacy weight_gemv_residual path.
#[test]
fn fleet_key_constructors_have_no_gaps() {
    let mut failures = Vec::new();
    for u in FLEET {
        if let Err(e) = construct_key(u.role, u.dtype) {
            failures.push(format!(
                "  {} / {:?} / {:?}  →  {:?}  (forward .unwrap()s this Err → runtime panic)",
                u.model, u.role, u.dtype, e
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} shipped (model, role, dtype) combos hit an UnsupportedVariant key-constructor \
         arm and HARD-PANIC on decode:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER 2 — arch coverage (catches the gfx12-dead-gate defect class). For every
/// constructible shipped dtype × arch it ships on, the dtype's required arch
/// predicate MUST admit that arch (else resolve() → MissingImpl / scalar fallback).
/// PASSES today (953ea648 fix is in: HasWmmaW32/HasMmq admit gfx12); would have
/// FAILED before that fix.
#[test]
fn fleet_dtypes_resolve_on_every_target_arch() {
    let mut failures = Vec::new();
    for u in FLEET {
        if construct_key(u.role, u.dtype).is_err() {
            continue; // Layer 1 owns constructor gaps
        }
        let pred = KernelKey::dtype_arch_predicate(u.dtype);
        for &arch in u.archs {
            let ctx = DispatchCtx::for_test(arch);
            if !pred.eval_arch(&ctx) {
                failures.push(format!(
                    "  {} / {:?} ({:?}) dead-gated on {} (predicate {:?} → MissingImpl/scalar)",
                    u.model, u.dtype, u.role, arch, pred
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} shipped (model, dtype, arch) combos are arch-dead-gated:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER 1b — exhaustive o_proj/down dtype sweep (defense in depth). Reports every
/// plausible residual dtype the constructor currently rejects, and HARD-asserts the
/// one confirmed shipped (Q8_0) so the gap can't be reintroduced.
#[test]
fn residual_constructor_covers_confirmed_oproj_dtypes() {
    const OPROJ_DTYPES: &[DType] = &[
        Q8_0, MQ4G256, MQ3G256, MQ6G256, HFQ4G256, HFQ6G256,
        MQ3G256Lloyd, MQ4G256Lloyd, ParoQ4G128, MFP4G32, Q4K,
    ];
    let missing: Vec<_> = OPROJ_DTYPES
        .iter()
        .filter(|d| KernelKey::for_gemv_residual(**d).is_err())
        .collect();
    if !missing.is_empty() {
        eprintln!("for_gemv_residual currently MISSING arms for: {:?}", missing);
    }
    assert!(
        KernelKey::for_gemv_residual(Q8_0).is_ok(),
        "for_gemv_residual(Q8_0) is the confirmed o_proj panic (q8f16 + A3B) — must be supported"
    );
}
