// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Dispatch coverage guardrail — catches the two recurring "missing dispatch arm"
//! defect classes at CI time, GPU-FREE (no kernels, no device, no GPU lock).
//!
//! Both defects that have already shipped on this branch reduce to a pure assertion
//! over the existing dispatch API:
//!
//!   1. NO-DISPATCH-PLAN GAP — a forward op (e.g. `Step::GemvResidual` for o_proj)
//!      has neither a fused kernel nor a fallback path for a dtype a shipped model
//!      uses, so the lowering hits `_ => UnsupportedVariant` and the forward pass
//!      `.unwrap()`s an Err → HARD-PANIC on decode.
//!      Live example (now fixed): `for_gemv_residual(Q8_0)` == Err AND no fallback →
//!      qwen3.5-9b.q8f16 + qwen3.6-35b-a3b o_proj panicked. The fix routes
//!      no-fused-kernel residual dtypes through plain-GEMV-into-temp + add_inplace,
//!      so the invariant is: a residual dtype is dispatchable iff it has a fused
//!      residual kernel OR a plain GEMV (the fallback).
//!
//!   2. ARCH DEAD-GATE — a dtype's required `ArchPredicate` excludes an arch the
//!      model ships on, so `resolve()` returns MissingImpl / the path silently
//!      falls to a slow scalar kernel. Live example (fixed at 953ea648): MQ3/MQ6
//!      gated on a gfx11-only predicate, excluding gfx1201/RDNA4.
//!
//! Keep `FLEET` in sync with the model loaders' per-op weight dtypes: a new quant
//! format or shipped tier means new rows here. This is the structural guardrail
//! #397 Phase-0.4 should adopt — a single coverage gate over (op × dtype × arch).

use crate::context::DispatchCtx;
use crate::families::moe::{MoeDtypes, MoeResolution};
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
///
/// **WARNING:** This name is historical and misleading. These archs are WMMA-capable
/// (gfx11+), NOT merely wave32-capable. RDNA1 (`gfx1010`) and RDNA2 (`gfx1030+`)
/// are wave32 but do NOT have WMMA (`has_wmma = is_rdna3 || is_rdna4`). New tests
/// that need WMMA-specific arch lists should use `WMMA_ARCHS` below instead.
const WAVE32: &[&str] = &[
    "gfx1100", "gfx1101", "gfx1102", // RDNA3 dGPU
    "gfx1150", "gfx1151", "gfx1152", // RDNA3.5 APU
    "gfx1200", "gfx1201", // RDNA4
];

/// Archs with WMMA support (`has_wmma = is_rdna3 || is_rdna4`).
/// Distinct from wave32: RDNA1/2 are wave32 but lack WMMA.
const WMMA_ARCHS: &[&str] = WAVE32; // same set today, but semantically distinct

/// Everything incl. RDNA1/2 + CDNA, for dtypes whose arch gate is Always/dp4a.
const ALL: &[&str] = &[
    "gfx1010", "gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102", "gfx1150",
    "gfx1151", "gfx1152", "gfx1200", "gfx1201", "gfx906", "gfx908", "gfx942",
];

/// PRODUCTION MATRIX — (model, role, dtype) the wired forward pass hits today.
/// The Q8_0 `Residual` rows are the ones the live gap panicked on (now fixed via
/// the plain-GEMV fallback).
const FLEET: &[OpUse] = &[
    // ── q8f16: Q8 weights throughout. o_proj reaches Step::GemvResidual on every arch ──
    OpUse {
        model: "qwen3.5-9b.q8f16",
        role: Role::Residual,
        dtype: Q8_0,
        archs: ALL,
    },
    OpUse {
        model: "qwen3.5-9b.q8f16",
        role: Role::Plain,
        dtype: Q8_0,
        archs: ALL,
    },
    OpUse {
        model: "qwen3.5-9b.q8f16",
        role: Role::SwigluResidual,
        dtype: Q8_0,
        archs: ALL,
    },
    // ── qwen3.6-35b-a3b MoE: Q8 attention o_proj ──
    OpUse {
        model: "qwen3.6-35b-a3b.mq4",
        role: Role::Residual,
        dtype: Q8_0,
        archs: WAVE32,
    },
    // ── Paro o_proj (no fused residual kernel → uses the same fallback) ──
    OpUse {
        model: "qwen3.5-*.paro4g128",
        role: Role::Residual,
        dtype: ParoQ4G128,
        archs: WAVE32,
    },
    // ── dense MQ4/MQ3/Lloyd: o_proj has a fused residual kernel — anchors (stay green) ──
    // MQ4/MQ6 work on ALL archs (generic GEMV fallback for gfx906/RDNA1 + arch-tuned
    // variants for RDNA2/3/4). Previously WAVE32-only (excluded gfx906) because
    // dtype_arch_predicate returned HasDot2F32F16 (=has_dot2_f32_f16=RDNA1.1+), which
    // excludes gfx906. Fixed: dtype_arch_predicate now returns Always for MQ4G256.
    OpUse {
        model: "qwen3.5-9b.mq4",
        role: Role::Plain,
        dtype: MQ4G256,
        archs: ALL,
    },
    OpUse {
        model: "qwen3.5-27b.mq4",
        role: Role::Residual,
        dtype: MQ4G256,
        archs: ALL,
    },
    OpUse {
        model: "qwen3.5-27b.mq3",
        role: Role::Residual,
        dtype: MQ3G256,
        archs: WAVE32,
    },
    OpUse {
        model: "qwen3.6-27b.mq3-lloyd",
        role: Role::Residual,
        dtype: MQ3G256Lloyd,
        archs: WAVE32,
    },
    OpUse {
        model: "qwen3.6-35b-a3b.mq4",
        role: Role::Plain,
        dtype: MQ4G256,
        archs: ALL,
    },
    // MQ5-promoted projections — gate is HasMmq (gfx906 + RDNA3 + RDNA4), mirrors MQ6:
    OpUse {
        model: "qwen3.6-35b-a3b.mq5",
        role: Role::Plain,
        dtype: MQ5G256,
        archs: &[
            "gfx906", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201",
        ],
    },
    // MQ6-promoted projections (A3B AWQ-attractor mitigation) — gate is HasMmq (gfx906 + RDNA3 + RDNA4):
    OpUse {
        model: "qwen3.6-35b-a3b.mq4",
        role: Role::Plain,
        dtype: MQ6G256,
        archs: &[
            "gfx906", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201",
        ],
    },
    // ── RDNA4 coverage: same (role, dtype) combos explicitly anchored on gfx12 arch strings ──
    // Catches reintroduction of an RDNA4-only ArchPredicate that would dead-gate these.
    OpUse {
        model: "qwen3.5-27b.mq4-rdna4",
        role: Role::Plain,
        dtype: MQ4G256,
        archs: &["gfx1200", "gfx1201"],
    },
    OpUse {
        model: "qwen3.5-27b.mq3-rdna4",
        role: Role::Plain,
        dtype: MQ3G256,
        archs: &["gfx1200", "gfx1201"],
    },
    OpUse {
        model: "qwen3.6-27b.lloyd-rdna4",
        role: Role::Plain,
        dtype: MQ3G256Lloyd,
        archs: &["gfx1200", "gfx1201"],
    },
    OpUse {
        model: "a3b-moe-mq5-rdna4",
        role: Role::Plain,
        dtype: MQ5G256,
        archs: &["gfx1200", "gfx1201"],
    },
    OpUse {
        model: "a3b-moe-rdna4",
        role: Role::Plain,
        dtype: MQ6G256,
        archs: &["gfx1200", "gfx1201"],
    },
    // ── MQ4G256V2 / MQ6G256V2 (qt44/qt47 dual-half): dense residual + prerotated ──
    // Plain role is satisfied via prerotated (V2 has no KernelKey::for_gemv Plain arm;
    // rotation is external and run_auto selects Prerotated). Residual/Swiglu are fused.
    OpUse {
        model: "ornith1.5-35b-a3b.mq4v2",
        role: Role::Plain,
        dtype: MQ4G256V2,
        archs: WAVE32,
    },
    OpUse {
        model: "ornith1.5-35b-a3b.mq4v2",
        role: Role::Residual,
        dtype: MQ4G256V2,
        archs: WAVE32,
    },
    OpUse {
        model: "ornith1.5-35b-a3b.mq4v2",
        role: Role::SwigluResidual,
        dtype: MQ4G256V2,
        archs: WAVE32,
    },
    OpUse {
        model: "qwen3.5-moe.mq6v2",
        role: Role::Plain,
        dtype: MQ6G256V2,
        archs: WAVE32,
    },
    OpUse {
        model: "qwen3.5-moe.mq6v2",
        role: Role::Residual,
        dtype: MQ6G256V2,
        archs: WAVE32,
    },
    OpUse {
        model: "qwen3.5-moe.mq6v2",
        role: Role::SwigluResidual,
        dtype: MQ6G256V2,
        archs: WAVE32,
    },
    // RDNA4 anchors — V2 dual-half must not re-acquire a gfx11-only gate.
    OpUse {
        model: "ornith-mq4v2-rdna4",
        role: Role::Plain,
        dtype: MQ4G256V2,
        archs: &["gfx1200", "gfx1201"],
    },
    OpUse {
        model: "qwen-mq6v2-rdna4",
        role: Role::Plain,
        dtype: MQ6G256V2,
        archs: &["gfx1200", "gfx1201"],
    },
];

/// Does the forward lowering for (role, dtype) have ANY dispatch plan (so it
/// cannot hit an `UnsupportedVariant` panic)? Mirrors the real lowering:
/// - Plain          → plain GEMV OR prerotated (V2 has no Plain arm).
/// - Residual       → fused `gemv_*_residual` kernel OR plain/prerot fallback.
/// - SwigluResidual → fused swiglu-residual kernel OR plain/prerot fallback.
fn has_dispatch_plan(role: Role, dtype: DType) -> bool {
    let plain = KernelKey::for_gemv(dtype, GemvVariant::Plain, false).is_ok();
    // V2 MQ formats (and other prerotated-only dtypes) have no Plain GEMV key —
    // rotation is external and run_auto selects Prerotated. A Plain role is still
    // dispatchable when that path exists.
    let prerot = KernelKey::for_gemv_prerotated(dtype).is_ok();
    match role {
        Role::Plain => plain || prerot,
        Role::Residual => KernelKey::for_gemv_residual(dtype).is_ok() || plain || prerot,
        Role::SwigluResidual => {
            KernelKey::for_gemv_swiglu_residual(dtype).is_ok() || plain || prerot
        }
    }
}

/// LAYER 1 — dispatch-plan coverage (catches the missing-arm panic class). Every
/// (role, dtype) a shipped model uses MUST have a dispatch plan. Before the
/// Q8/Paro fix this FAILED on the q8f16/A3B/Paro `Residual` rows (the panic);
/// after the fix those resolve via the plain-GEMV fallback.
#[test]
fn fleet_ops_have_a_dispatch_plan() {
    let mut failures = Vec::new();
    for u in FLEET {
        if !has_dispatch_plan(u.role, u.dtype) {
            failures.push(format!(
                "  {} / {:?} / {:?}  →  no dispatch plan (no fused kernel, no plain fallback) → runtime panic",
                u.model, u.role, u.dtype
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} shipped (model, role, dtype) combos have NO dispatch plan and HARD-PANIC on decode:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER 2 — arch coverage (catches the gfx12-dead-gate defect class). For every
/// shipped dtype × arch it ships on, the dtype's required arch predicate MUST
/// admit that arch (else resolve() → MissingImpl / scalar fallback). Passes today
/// (953ea648 fix is in: HasWmma/HasMmq admit gfx12); would have failed before.
#[test]
fn fleet_dtypes_resolve_on_every_target_arch() {
    let mut failures = Vec::new();
    for u in FLEET {
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

/// LAYER 1b — o_proj/down dtype sweep (defense in depth). Reports which residual
/// dtypes still lack a FUSED kernel (use the slower fallback — informational), and
/// HARD-asserts the confirmed-shipped o_proj dtypes (Q8_0, ParoQ4G128) have a
/// dispatch plan so the panic can't be reintroduced.
#[test]
fn confirmed_oproj_dtypes_have_a_plan() {
    const OPROJ_DTYPES: &[DType] = &[
        Q8_0,
        MQ4G256,
        MQ3G256,
        MQ5G256,
        MQ6G256,
        HFQ4G256,
        HFQ6G256,
        MQ3G256Lloyd,
        MQ4G256Lloyd,
        ParoQ4G128,
        MFP4G32,
        Q4K,
    ];
    let no_fused: Vec<_> = OPROJ_DTYPES
        .iter()
        .filter(|d| KernelKey::for_gemv_residual(**d).is_err())
        .collect();
    if !no_fused.is_empty() {
        eprintln!(
            "residual dtypes with no FUSED kernel (use plain+add fallback): {:?}",
            no_fused
        );
    }
    for d in [Q8_0, ParoQ4G128] {
        assert!(
            has_dispatch_plan(Role::Residual, d),
            "Role::Residual / {:?} has no dispatch plan — the o_proj panic would return",
            d
        );
    }
}

/// LAYER 1d — MoE CPU-top-K fallback coverage (catches the #393 regression).
/// `run_moe_decode`'s GPU-top-K fast path only serves `k == 8` MoE layers whose
/// routed experts are indexable — today `{MQ4G256, MQ4G256V2, MQ6G256,
/// MQ6G256V2, ParoQ4G128, …}`. Every OTHER MoE layer (`k != 8`, or a routed
/// dtype like Q8_0) MUST take the generic CPU-top-K per-expert fallback —
/// #393 deleted that fallback so those layers hit
/// `UnsupportedVariant{cpu-topk-fallback}` and HARD-PANIC on decode.
///
/// GPU-free assertion in two parts, mirroring the runtime guarantees:
///   (a) The eligibility lattice routes these layers to the fallback, NOT the
///       k8 indexed path: `MoeResolution::resolve(..).use_gpu_topk == false`.
///   (b) The fallback's per-expert loop dispatches gate_up + down through
///       `GemvFamily::run_auto`, so the routed dtype MUST have a post-rotation
///       GEMV plan (Plain OR Prerotated — V2 uses Prerotated only).
#[test]
fn non_k8_and_q8_routed_moe_has_a_dispatch_plan() {
    // A representative non-indexable / non-k8 MoE matrix the fallback must serve.
    // (router/shared dtypes don't gate the fallback decision — only k + routed do.)
    struct MoeUse {
        name: &'static str,
        routed_gate_up: DType,
        routed_down: DType,
        k: usize,
    }
    /// run_auto needs Plain OR Prerotated (dtype_post_rotation_variant).
    fn has_run_auto_plan(dtype: DType) -> bool {
        KernelKey::for_gemv(dtype, GemvVariant::Plain, false).is_ok()
            || KernelKey::for_gemv_prerotated(dtype).is_ok()
    }
    let mut failures = Vec::new();
    for u in [
        // Q8-routed experts, k=8 → not indexable → CPU-top-K fallback.
        MoeUse {
            name: "q8-routed-moe (k=8)",
            routed_gate_up: Q8_0,
            routed_down: Q8_0,
            k: 8,
        },
        // MQ4 routed but k != 8 → k8 indexed kernels unusable → fallback.
        MoeUse {
            name: "mq4-routed-moe (k=4)",
            routed_gate_up: MQ4G256,
            routed_down: MQ4G256,
            k: 4,
        },
        // MQ4V2 / MQ6V2 uniform but k != 8 → indexable flag stays true, GPU-top-K off.
        MoeUse {
            name: "mq4v2-routed-moe (k=4)",
            routed_gate_up: MQ4G256V2,
            routed_down: MQ4G256V2,
            k: 4,
        },
        MoeUse {
            name: "mq6v2-routed-moe (k=4)",
            routed_gate_up: MQ6G256V2,
            routed_down: MQ6G256V2,
            k: 4,
        },
        // F32 routed experts, k=2 → fallback.
        MoeUse {
            name: "f32-routed-moe (k=2)",
            routed_gate_up: F32,
            routed_down: F32,
            k: 2,
        },
    ] {
        let d = MoeDtypes {
            router: Q8_0,
            shared_gate: Q8_0,
            shared_expert_gate: Q8_0,
            shared_expert_up: Q8_0,
            shared_expert_down: Q8_0,
            experts_all_gate_up_mq4: u.routed_gate_up == MQ4G256,
            routed_gate_up: u.routed_gate_up,
            routed_down: u.routed_down,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
            routed_escha_transforms: false,
        };
        let res = MoeResolution::resolve(&d, u.k);
        // (a) These layers MUST take the fallback, not the k8 indexed path.
        if res.use_gpu_topk {
            failures.push(format!(
                "  {}: resolved to GPU-top-K (use_gpu_topk=true) but routed dtype/k is non-indexable",
                u.name
            ));
        }
        // (b) The fallback's run_auto needs a GEMV plan for both halves.
        if !has_run_auto_plan(u.routed_gate_up) {
            failures.push(format!(
                "  {}: routed gate_up {:?} has no plain/prerot GEMV → fallback run_auto → UnsupportedVariant panic",
                u.name, u.routed_gate_up
            ));
        }
        if !has_run_auto_plan(u.routed_down) {
            failures.push(format!(
                "  {}: routed down {:?} has no plain/prerot GEMV → fallback run_auto → UnsupportedVariant panic",
                u.name, u.routed_down
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} non-k8 / non-indexable-routed MoE layers would HARD-PANIC \
         (the #393 cpu-topk-fallback regression):\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER 1e — MoE decode pre-guard (#397 Ship 4c). `run_moe_decode` now calls
/// `check_moe_decode_supported` BEFORE any GPU work, turning the two deep-fallback
/// failure modes into clean `DispatchError`s:
///   - `k` out of `[1, n_exp]` → the CPU fallback's `select_nth_unstable_by(k-1)`
///     panics; the pre-guard must reject it gracefully.
///   - a routed dtype on neither path (not GPU-top-K-indexable AND no resident
///     experts) → no kernel can run it; the pre-guard must reject it gracefully.
/// CRITICAL: the canonical fallback case `(op=moe, dtype=MQ4G256, k=4)` with
/// resident experts MUST pass the guard cleanly — `k != 8` is NOT an error.
#[test]
fn moe_decode_pre_guard_admits_fallback_and_rejects_invalid() {
    use crate::pipeline::check_moe_decode_supported;

    // The canonical MoE coverage row this task asks for:
    // (op=moe, dtype=MQ4G256, k=4) → resolves to the CPU fallback, NOT GPU-top-K.
    let mq4_k4 = MoeDtypes {
        router: Q8_0,
        shared_gate: Q8_0,
        shared_expert_gate: Q8_0,
        shared_expert_up: Q8_0,
        shared_expert_down: Q8_0,
        experts_all_gate_up_mq4: true,
        routed_gate_up: MQ4G256,
        routed_down: MQ4G256,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
        routed_escha_transforms: false,
    };
    let res_k4 = MoeResolution::resolve(&mq4_k4, 4);
    assert!(
        !res_k4.use_gpu_topk,
        "MQ4G256 k=4 must route to the CPU-top-K fallback (k != 8), not GPU-top-K"
    );
    // With resident experts the fallback can run it → guard MUST pass cleanly.
    assert!(
        check_moe_decode_supported(
            res_k4.use_gpu_topk,
            4,
            /*n_exp=*/ 64,
            /*resident=*/ true,
            /*has_escha=*/ false,
            /*escha_indexed_supported=*/ false
        )
        .is_ok(),
        "MQ4G256 k=4 with resident experts is a VALID fallback case — guard must not reject it"
    );

    // The GPU-top-K fast path (k=8 + indexable routed dtype) is also admitted.
    let mq4_k8 = MoeDtypes {
        routed_gate_up: MQ4G256,
        routed_down: MQ4G256,
        ..mq4_k4
    };
    let res_k8 = MoeResolution::resolve(&mq4_k8, 8);
    assert!(
        res_k8.use_gpu_topk,
        "MQ4G256 k=8 must be GPU-top-K-indexable"
    );
    assert!(
        check_moe_decode_supported(
            res_k8.use_gpu_topk,
            8,
            64,
            /*resident=*/ false,
            /*has_escha=*/ false,
            /*escha_indexed_supported=*/ false
        )
        .is_ok(),
        "GPU-top-K path is valid even under paged (non-resident) residency"
    );

    // (a) out-of-range k errors gracefully (no panic): k == 0 and k > n_exp.
    assert!(
        check_moe_decode_supported(false, 0, 64, true, false, false).is_err(),
        "k == 0 must be rejected (would panic select_nth_unstable_by(k-1))"
    );
    assert!(
        check_moe_decode_supported(false, 65, 64, true, false, false).is_err(),
        "k > n_exp must be rejected (would panic select_nth_unstable_by(k-1))"
    );

    // (b) routed dtype on NEITHER path: not GPU-top-K AND no resident experts.
    assert!(
        check_moe_decode_supported(
            /*use_gpu_topk=*/ false, 4, 64, /*resident=*/ false,
            /*has_escha=*/ false, /*escha_indexed_supported=*/ false
        )
        .is_err(),
        "non-fast-path dtype with no resident experts has no runnable path — reject gracefully"
    );
}

/// LAYER 1f — Escha-W2 on the indexed GPU-top-K path: SUPPORTED only in the
/// one shape that keeps the H128 pair, and failing closed in every other.
///
/// Escha-W2 weights live in a rotated domain. Only the executors in
/// `pipeline::escha` wrap the routed GEMVs in the H128 pair; an escha layer
/// that reached the GENERIC indexed routed body would skip both Hadamard
/// transforms and emit finite, fluent output wrong by ~1e-1. Nothing would
/// fire — not a NaN check, not a shape check, not a dtype check.
///
/// This test used to assert that escha could never be on the indexed path at
/// all, because the only thing keeping it off was a negative accident (the
/// loader materialises the experts as `Q8_0`, and no `routed_indexable_*` arm
/// admitted `Q8_0`). That accident has been replaced by real support:
/// `routed_indexable_escha_q8` admits Q8_0-on-both-projections experts that
/// carry the transform tables, and `run_moe_decode` routes them to
/// `escha::escha_routed_decode_indexed`, which applies the same H128 pair as
/// the CPU-top-K executor.
///
/// So the contract the guard now encodes is narrower and stronger — the
/// SPECIFIC combination is supported and everything adjacent to it is not:
///   1. escha + indexed + `escha_indexed_supported` → ADMITTED (production).
///   2. escha + indexed + tables MISSING → REJECTED (the executor could not
///      have been called; the generic body would run and drop the transforms).
///   3. escha + indexed via some OTHER arm (non-Q8_0 routed dtype) → REJECTED.
///      The escha indexed GEMV hard-codes the Q8_0 block layout, so this is
///      the mirror-image silent corruption of case 2.
///   4. escha + `use_gpu_topk == false` → admitted (the CPU-top-K route).
///   5. non-escha + indexed → admitted (every other MoE model).
///
/// Cases 2 and 3 are what go red if arm (c) is weakened to `true`; case 1 is
/// what goes red if it is left as the old blanket refusal.
#[test]
fn escha_layer_must_not_take_the_indexed_gpu_topk_path() {
    use crate::pipeline::check_moe_decode_supported;

    let base = MoeDtypes {
        router: Q8_0,
        shared_gate: Q8_0,
        shared_expert_gate: Q8_0,
        shared_expert_up: Q8_0,
        shared_expert_down: Q8_0,
        experts_all_gate_up_mq4: false,
        routed_gate_up: Q8_0,
        routed_down: Q8_0,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
        routed_escha_transforms: false,
    };

    // ── 1. Production shape: Q8_0 experts + resident transform tables ──────
    // `MoeResolution` is asked for the real answer rather than hand-set, so
    // this stays honest if the arms are reworked.
    let escha_q8 = MoeDtypes {
        routed_escha_transforms: true,
        per_expert_gate_up: None,
        per_expert_down: None,
        ..base
    };
    let res_escha = MoeResolution::resolve(&escha_q8, 8);
    assert!(
        res_escha.routed_indexable_escha_q8,
        "Q8_0-on-both-projections experts with resident H128 tables ARE the escha indexed arm"
    );
    assert!(
        res_escha.use_gpu_topk,
        "the escha arm must admit the layer to GPU-resident top-K — that is the whole point of \
         it (no per-layer topk D2H, no per-expert GEMV launch storm, hipGraph-capturable)"
    );
    assert!(
        check_moe_decode_supported(
            res_escha.use_gpu_topk,
            8,
            /*n_exp=*/ 64,
            /*resident=*/ true,
            /*has_escha=*/ true,
            /*escha_indexed_supported=*/ true,
        )
        .is_ok(),
        "escha on the indexed path WITH the escha executor behind it is the supported \
         production shape — the guard must admit it. If this is red the guard is still the old \
         blanket refusal and escha is stuck on the CPU-top-K route."
    );

    // ── 2. Tables missing: the executor could not run, so refuse ───────────
    let err = check_moe_decode_supported(
        res_escha.use_gpu_topk,
        8,
        64,
        /*resident=*/ true,
        /*has_escha=*/ true,
        /*escha_indexed_supported=*/ false,
    )
    .expect_err(
        "an escha layer on the indexed path WITHOUT its transform tables MUST be refused: the \
         escha executor cannot be called, so the generic indexed body would run the experts \
         with no H128 pair and emit finite, fluent, ~1e-1-wrong output with nothing to catch \
         it.",
    );
    match err {
        DispatchError::UnsupportedVariant {
            family, variant, ..
        } => {
            assert_eq!(family, "moe");
            assert_eq!(
                variant, "escha-routed-experts-on-indexed-gpu-topk-path",
                "the refusal must name escha and the unsupported path — a generic error here \
                 means the escha arm did not fire and something else rejected the case"
            );
        }
        other => panic!("expected UnsupportedVariant, got {other:?}"),
    }

    // ── 3. Indexable through a DIFFERENT arm while still escha ─────────────
    // A future graded/mixed escha file whose representative routed dtype is
    // not Q8_0 would resolve indexable via the MQ4 arm. The escha indexed
    // GEMV hard-codes the Q8_0 34 B/32-element block layout, so dispatching
    // it there is silent corruption; so is letting the generic body have it.
    let escha_via_mq4 = MoeDtypes {
        experts_all_gate_up_mq4: true,
        routed_gate_up: MQ4G256,
        routed_down: MQ4G256,
        routed_escha_transforms: true,
        per_expert_gate_up: None,
        per_expert_down: None,
        ..base
    };
    let res_mq4 = MoeResolution::resolve(&escha_via_mq4, 8);
    assert!(
        res_mq4.use_gpu_topk && !res_mq4.routed_indexable_escha_q8,
        "fixture precondition: indexable, but NOT through the escha arm"
    );
    assert!(
        check_moe_decode_supported(
            res_mq4.use_gpu_topk,
            8,
            64,
            true,
            /*has_escha=*/ true,
            // What `run_moe_decode` computes: the escha arm did not fire, so
            // the escha executor is not the one that would run.
            /*escha_indexed_supported=*/
            res_mq4.routed_indexable_escha_q8,
        )
        .is_err(),
        "an escha layer that reached the indexed path through a NON-escha arm must be refused — \
         the escha executor only decodes Q8_0, and the generic body applies no transforms"
    );

    // ── 4. CPU-top-K route stays admitted ──────────────────────────────────
    let escha_k4 = MoeDtypes {
        routed_escha_transforms: true,
        per_expert_gate_up: None,
        per_expert_down: None,
        ..base
    };
    let res_k4 = MoeResolution::resolve(&escha_k4, 4);
    assert!(
        !res_k4.use_gpu_topk,
        "k != 8 has no indexed kernel on any arm, escha included"
    );
    assert!(
        check_moe_decode_supported(res_k4.use_gpu_topk, 4, 64, true, true, false).is_ok(),
        "escha on the CPU-top-K fallback is still a supported shape — the guard must not \
         reject it"
    );

    // ── 5. Non-escha models on the indexed path are untouched ──────────────
    assert!(
        check_moe_decode_supported(
            res_mq4.use_gpu_topk,
            8,
            64,
            true,
            /*has_escha=*/ false,
            /*escha_indexed_supported=*/ false
        )
        .is_ok(),
        "the escha guard must not affect any non-escha MoE model"
    );
}

/// LAYER 1f (companion) — the Q8_0 indexed arm must stay ESCHA-SCOPED.
///
/// `routed_indexable_escha_q8` is the first arm in `MoeResolution` that
/// admits `Q8_0` routed experts, and `Q8_0` is the most common dtype in this
/// codebase. If the escha gate on it were ever dropped, every plain-Q8_0 MoE
/// model in the tree would silently move from the CPU-top-K fallback onto an
/// indexed path — either the generic body (whose kernels expect a different
/// container for the projections that arm implies) or, worse, the escha
/// executor, which would apply a Hadamard pair those weights were never
/// packed in. Both are fluent wrong output, not a fault.
#[test]
fn plain_q8_0_routed_experts_stay_off_the_indexed_path() {
    let plain_q8 = MoeDtypes {
        router: Q8_0,
        shared_gate: Q8_0,
        shared_expert_gate: Q8_0,
        shared_expert_up: Q8_0,
        shared_expert_down: Q8_0,
        experts_all_gate_up_mq4: false,
        routed_gate_up: Q8_0,
        routed_down: Q8_0,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
        // The ONLY difference from the escha production fixture above.
        routed_escha_transforms: false,
    };
    let res = MoeResolution::resolve(&plain_q8, 8);
    assert!(
        !res.routed_indexable_escha_q8,
        "the Q8_0 indexed arm must require the escha transform tables"
    );
    assert!(
        !res.use_gpu_topk,
        "a plain Q8_0 MoE model must keep taking the CPU-top-K fallback. If this is red, the \
         escha gate came off the Q8_0 arm and every Q8_0 MoE model in the tree just changed \
         execution path (and, on the escha executor, answer)."
    );
}

/// LAYER 1c — Q8/Paro were gapped in MULTIPLE GEMV variants: o_proj used Residual,
/// then the FFN/qkv used Prerotated (the second panic domino). Lock every variant
/// these dtypes are actually dispatched through.
#[test]
fn q8_and_paro_dispatchable_in_all_used_variants() {
    for d in [Q8_0, ParoQ4G128] {
        assert!(
            KernelKey::for_gemv(d, GemvVariant::Plain, false).is_ok(),
            "{:?}: plain GEMV missing (lm_head / direct GEMV panics)",
            d
        );
        assert!(
            KernelKey::for_gemv_prerotated(d).is_ok(),
            "{:?}: prerotated GEMV missing (FFN / qkv prerotated path panics)",
            d
        );
        assert!(
            has_dispatch_plan(Role::Residual, d),
            "{:?}: residual GEMV has no plan (o_proj panics)",
            d
        );
    }
}

/// LAYER 1d — for_gemv_prerotated must cover EVERY rotation-free dtype. The
/// run_fa_layer_body migration (and the already-migrated FullAttnMoe path) lower
/// the unfused QKV/gate_up fallback through GemvVariant::Prerotated; for a
/// rotation-free dtype "prerotated" == plain, so it MUST resolve (the legacy
/// run_auto path did). Before the fix HFQ6/HFQ3/F16/F32/Q4K/Q6K/HFP4 hard-errored.
#[test]
fn prerotated_covers_rotation_free_dtypes() {
    for d in [
        F16, F32, Q4K, Q6K, HFQ3G256, HFQ6G256, HFQ2G256, HFP4G32, Q8_0,
    ] {
        assert!(
            KernelKey::for_gemv_prerotated(d).is_ok(),
            "for_gemv_prerotated({:?}) errors — the unfused FA fallback panics where the \
             legacy run_auto->Plain path worked",
            d
        );
    }
    // A rotation-NEEDING dtype not explicitly handled must NOT fall through to plain
    // (the plain path would re-rotate already-rotated input): MQ4G128 stays an Err.
    assert!(
        KernelKey::for_gemv_prerotated(MQ4G128).is_err(),
        "MQ4G128 (FwhtG128) must stay an error — falling to plain would double-rotate"
    );
}

/// LAYER 2b — Attention family key coverage. Every attention key registered in
/// the attention table MUST resolve on every arch the fleet ships on. Catches:
///   - A new attention key that is accidentally gated to a narrow arch
///     (the gfx12 dead-gate pattern from 953ea648).
///   - The non-flash Q8 key (`AttnQ8_0Kv`) being missing from the table
///     (the B0 gap — short-context Q8 decode would silently reroute to flash).
#[test]
fn attention_keys_resolve_on_fleet_archs() {
    use crate::families::attention::AttentionFamily;

    /// Attention keys the qwen35 decode path exercises. `HasWmma` keys (GQA-fused)
    /// only resolve on WMMA-capable archs; all others must resolve everywhere.
    struct AttnKeyUse {
        key: KernelKey,
        /// Archs where this key MUST resolve. `Always`-gated keys use ALL;
        /// `HasWmma`-gated keys use WAVE32.
        archs: &'static [&'static str],
        /// Shape to pass. `None` bypasses shape gating. Batched keys need
        /// `batch_size > 1` to pass `BatchGt(1)` / `BatchEq(1)` gates.
        shape: Option<ShapeInfo>,
    }

    let attn_fleet: &[AttnKeyUse] = &[
        // KV write — single-token, Always-gated
        AttnKeyUse {
            key: KernelKey::KvWriteF32,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteQ8_0,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym4,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym4Fwht,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym3,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym3Fwht,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym2,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym2Fwht,
            archs: ALL,
            shape: None,
        },
        // Llama legacy KV write — single-token, Always-gated
        AttnKeyUse {
            key: KernelKey::KvWriteHfq4,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::KvWriteQ4,
            archs: ALL,
            shape: None,
        },
        // KV write — batched, Always-gated, BatchGt(1)
        AttnKeyUse {
            key: KernelKey::KvWriteAsym4Batched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym4FwhtBatched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym3Batched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym3FwhtBatched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym2Batched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteAsym2FwhtBatched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::KvWriteQ8_0Batched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        // Attention — single-token, Always-gated
        AttnKeyUse {
            key: KernelKey::AttnF32,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashQ8_0,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnQ8_0Kv,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym4,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym4Fwht,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym3,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym3Fwht,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym2,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym2Fwht,
            archs: ALL,
            shape: None,
        },
        // Llama legacy quant KV — single-token, Always-gated
        AttnKeyUse {
            key: KernelKey::AttnHfq4Kv,
            archs: ALL,
            shape: None,
        },
        AttnKeyUse {
            key: KernelKey::AttnQ4Kv,
            archs: ALL,
            shape: None,
        },
        // GQA-fused — HasWmma-gated
        AttnKeyUse {
            key: KernelKey::AttnGqaFused,
            archs: WMMA_ARCHS,
            shape: None,
        },
        // Attention — batched, Always-gated (scalar fallback), BatchGt(1)
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym4BatchedMasked,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym4FwhtBatchedMasked,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym3BatchedMasked,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym3FwhtBatchedMasked,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym2Batched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnFlashAsym2FwhtBatched,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
        AttnKeyUse {
            key: KernelKey::AttnQ8_0KvBatchedMasked,
            archs: ALL,
            shape: Some(ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 0,
                is_tree: false,
            }),
        },
    ];

    let family = AttentionFamily::new();
    let mut failures = Vec::new();
    for u in attn_fleet {
        for &arch in u.archs {
            let ctx = DispatchCtx::for_test(arch);
            if family.resolve(u.key, &ctx, u.shape.as_ref()).is_err() {
                failures.push(format!(
                    "  {:?} dead-gated on {} — resolve() returned Err",
                    u.key, arch
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} attention key × arch combos failed to resolve:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER 2c — Fused QKV/QKVZA/GateUp family key coverage. Every key registered
/// in the fused_qkv table MUST resolve on every arch the kernel actually runs on.
/// Catches the same dead-gate class as LAYER 2b but for the fused-kernel family.
///
/// Historical defect: `FusedQkvzaHfq4G256` was gated `HasWmma`, excluding gfx906
/// (dp4a path) and gfx1030/gfx1031 (wave32 generic path) even though the kernel
/// runs on all three. Found by A/B smoke on gfx906 (2026-06-06).
#[test]
fn fused_qkv_keys_resolve_on_fleet_archs() {
    use crate::families::fused_qkv::FusedQkvFamily;

    struct FusedKeyUse {
        key: KernelKey,
        /// Archs where this key MUST resolve.
        archs: &'static [&'static str],
    }

    let fused_fleet: &[FusedKeyUse] = &[
        // ── Cross-arch HFQ4G256 kernels (dp4a on gfx906, wave64 on CDNA, wave32 on RDNA) ──
        // QKV 3-way
        FusedKeyUse {
            key: KernelKey::FusedQkvHfq4G256,
            archs: ALL,
        },
        // QKVZA 4-way (DeltaNet linear attention)
        FusedKeyUse {
            key: KernelKey::FusedQkvzaHfq4G256,
            archs: ALL,
        },
        // Gate+Up 2-way
        FusedKeyUse {
            key: KernelKey::FusedGateUpHfq4G256,
            archs: ALL,
        },
        // Q4K (llama-format) — cross-arch
        FusedKeyUse {
            key: KernelKey::FusedQkvQ4K,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedGateUpQ4K,
            archs: ALL,
        },
        // Q8_0 — cross-arch
        FusedKeyUse {
            key: KernelKey::FusedGateUpQ8_0,
            archs: ALL,
        },
        // ── HFQ6G256 fused — cross-arch (batched gemm_*_hfq6g256 ladder:
        //    wmma_gfx12/wmma/dp4a/dot2/fp16/scalar). Was wrongly gfx906-only
        //    (HasDp4a), which dead-gated the AWQ A3B trunk on RDNA3/4. ──
        FusedKeyUse {
            key: KernelKey::FusedQkvHfq6G256,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvzaHfq6G256,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedGateUpHfq6G256,
            archs: ALL,
        },
        // ── MQ3/MQ4-Lloyd fused (W4: WMMA-free [32,1,1] wave32 scalar, run on
        //    every RDNA gen). Gate is HasWave32 (was a HasWmma dead-gate). Listed
        //    on WMMA_ARCHS here as a MUST-resolve floor; w4_* tests below assert
        //    the full RDNA1/2 admit + CDNA rejection on the GEMV-side siblings. ──
        FusedKeyUse {
            key: KernelKey::FusedQkvMq3G256Lloyd,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvMq4G256Lloyd,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvzaMq3G256Lloyd,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvzaMq4G256Lloyd,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedGateUpMq3G256Lloyd,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedGateUpMq4G256Lloyd,
            archs: WMMA_ARCHS,
        },
        // ── #397 Ship 5.2 slice 2: prefill gate+up dtypes ──
        // HFQ3G256: Always — base `gemm_gate_up_hfq3g256` carries a full
        // cross-arch internal ladder (MMQ→dp4a→dot2→fp16→scalar gfx1010), and
        // the run-arm picks WMMA vs base by arch, so the dtype runs everywhere.
        FusedKeyUse {
            key: KernelKey::FusedGateUpHfq3G256,
            archs: ALL,
        },
        // HFP4G32: WMMA-only — `gemm_gate_up_hfp4g32` dispatches ONLY to
        // gfx11/gfx12 WMMA siblings, no scalar fallback. Differs from the
        // sibling HFQ4 gate+up (ALL); must NOT resolve on RDNA1/2 or CDNA.
        FusedKeyUse {
            key: KernelKey::FusedGateUpHfp4G32,
            archs: WMMA_ARCHS,
        },
        // ── #397 Ship 5.2 slice 3: prefill QKV / QKVZA dtypes ──
        // Q8_0 fused QKV / QKVZA: WMMA-only — the run-arm calls
        // `gemm_qkv_q8_0_wmma` / `gemm_qkvza_q8_0_wmma` (gfx12 sibling on RDNA4
        // else gfx11 `_w32` WMMA), NO scalar/dp4a fallback and no decode method.
        // Differs from the gate+up Q8 row (ALL): that key ALSO has a non-WMMA
        // `fused_gate_up_q8_0` decode body; the QKV/QKVZA Q8 keys do not. Must
        // NOT resolve on RDNA1/2 or CDNA.
        FusedKeyUse {
            key: KernelKey::FusedQkvQ8_0,
            archs: WMMA_ARCHS,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvzaQ8_0,
            archs: WMMA_ARCHS,
        },
        // HFQ3G256 fused QKV / QKVZA: Always — base `gemm_qkv_hfq3g256` /
        // `gemm_qkvza_hfq3g256` carry a full cross-arch internal ladder
        // (MMQ→dp4a→dot2→fp16→scalar gfx1010), and the run-arm picks WMMA vs base
        // by arch, so the dtype runs everywhere (mirrors FusedGateUpHfq3G256).
        FusedKeyUse {
            key: KernelKey::FusedQkvHfq3G256,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvzaHfq3G256,
            archs: ALL,
        },
        // ── HasDp4a (gfx906 v_dot4_i32_i8) kernels ──
        // Paro fused: Always-gated (generic wave32 kernels, no ISA intrinsics)
        FusedKeyUse {
            key: KernelKey::FusedQkvzaParo4G128T,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedQkvParo4G128T,
            archs: ALL,
        },
        FusedKeyUse {
            key: KernelKey::FusedGateUpParo4G128T,
            archs: ALL,
        },
    ];

    let family = FusedQkvFamily::new();
    let mut failures = Vec::new();
    for u in fused_fleet {
        for &arch in u.archs {
            let ctx = DispatchCtx::for_test(arch);
            if family.resolve(u.key, &ctx, None).is_err() {
                failures.push(format!(
                    "  {:?} dead-gated on {} — resolve() returned Err",
                    u.key, arch
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} fused-QKV key × arch combos failed to resolve:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// C5 verification: full-attention keys resolve on their intended archs.
/// AttnFullF16 needs WMMA; AttnFullF32 is Always. Causal variants mirror.
#[test]
fn full_attention_keys_resolve_on_fleet_archs() {
    use crate::families::attention::AttentionFamily;

    struct FullAttnCase {
        key: KernelKey,
        archs: &'static [&'static str],
        shape: ShapeInfo,
    }

    let cases: &[FullAttnCase] = &[
        // AttnFullF16: needs HasWmma or HasWmmaGfx12, head_dim=128, batch>=64
        FullAttnCase {
            key: KernelKey::AttnFullF16,
            archs: WMMA_ARCHS,
            shape: ShapeInfo {
                batch_size: 64,
                head_dim: 128,
                m: 64,
                is_tree: false,
            },
        },
        // AttnFullF32: Always, scalar floor for any head_dim
        FullAttnCase {
            key: KernelKey::AttnFullF32,
            archs: ALL,
            shape: ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 16,
                is_tree: false,
            },
        },
        // AttnFullF16Causal: HasWmma or HasWmmaGfx12, head_dim=128
        FullAttnCase {
            key: KernelKey::AttnFullF16Causal,
            archs: WMMA_ARCHS,
            shape: ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 16,
                is_tree: false,
            },
        },
        // AttnFullF32Causal: Always, scalar floor
        FullAttnCase {
            key: KernelKey::AttnFullF32Causal,
            archs: ALL,
            shape: ShapeInfo {
                batch_size: 16,
                head_dim: 128,
                m: 16,
                is_tree: false,
            },
        },
    ];

    let family = AttentionFamily::new();
    let mut failures = Vec::new();
    for case in cases {
        for &arch in case.archs {
            let ctx = DispatchCtx::for_test(arch);
            if family.resolve(case.key, &ctx, Some(&case.shape)).is_err() {
                failures.push(format!(
                    "  {:?} dead-gated on {} — resolve() returned Err",
                    case.key, arch
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} full-attention key × arch combos failed to resolve:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// C5 verification: scalar floors resolve on non-WMMA archs (gfx906, gfx1030).
#[test]
fn scalar_floors_resolve_on_non_wmma_archs() {
    use crate::families::attention::AttentionFamily;
    let family = AttentionFamily::new();
    let non_wmma_archs: &[&str] = &["gfx906", "gfx1030"];
    let shape = ShapeInfo {
        batch_size: 16,
        head_dim: 128,
        m: 16,
        is_tree: false,
    };

    for &arch in non_wmma_archs {
        let ctx = DispatchCtx::for_test(arch);
        // DflashScalar (AttnFullF32)
        let r = family.resolve(KernelKey::AttnFullF32, &ctx, Some(&shape));
        assert!(
            r.is_ok(),
            "AttnFullF32 dead-gated on {} — should resolve to DflashScalar",
            arch
        );
        // CausalScalar (AttnFullF32Causal)
        let r = family.resolve(KernelKey::AttnFullF32Causal, &ctx, Some(&shape));
        assert!(
            r.is_ok(),
            "AttnFullF32Causal dead-gated on {} — should resolve to CausalScalar",
            arch
        );
    }
}

/// C5 verification: AttnFullF16 MUST NOT resolve on non-WMMA archs.
#[test]
fn f16_full_attention_rejected_on_non_wmma_archs() {
    use crate::families::attention::AttentionFamily;
    let family = AttentionFamily::new();
    let ctx = DispatchCtx::for_test("gfx906");
    let shape = ShapeInfo {
        batch_size: 64,
        head_dim: 128,
        m: 64,
        is_tree: false,
    };
    let r = family.resolve(KernelKey::AttnFullF16, &ctx, Some(&shape));
    assert!(
        r.is_err(),
        "AttnFullF16 should NOT resolve on gfx906 — no WMMA"
    );
}

// ── W4: un-brick RDNA2 (gfx1030/1031/1032) MQ3/Lloyd dispatch ─────────────────
//
// MQ3G256 + MQ2/MQ3/MQ4-Lloyd GEMVs are WMMA-free [32,1,1] wave32 scalar kernels
// that already run on RDNA1/2 via the qwen35 direct path, but resolve() dead-gated
// them as HasWmma → MissingImpl on RDNA1/2. W4 flips the gate to HasWave32. These
// tests pin the new behavior AND the hard non-regression constraint: because
// HasWave32 ⊇ HasWmma on RDNA3/4, the resolved variant must be BYTE-IDENTICAL on
// gfx1100/gfx1201 vs before (same KernelKey from the same single registered entry).

/// W4 dtypes whose GEMV gate is HasWave32 after the fix (the un-bricked set).
const W4_WAVE32_DTYPES: &[DType] = &[MQ3G256, MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256Lloyd];

/// W4: MQ3/Lloyd GEMVs resolve Ok on RDNA1 (gfx1010) + RDNA2 (gfx1030/1031/1032)
/// AFTER the HasWmma→HasWave32 fix. (Was MissingImpl before.)
#[test]
fn w4_mq3_lloyd_resolve_on_rdna1_and_rdna2() {
    use crate::families::gemv::GemvFamily;
    let fam = GemvFamily::new();
    let archs: &[&str] = &["gfx1010", "gfx1030", "gfx1031", "gfx1032"];
    let mut failures = Vec::new();
    for &d in W4_WAVE32_DTYPES {
        for &arch in archs {
            let ctx = DispatchCtx::for_test(arch);
            // Both the Plain and Prerotated GEMV variants must resolve (MQ-family
            // weights are always pre-rotated at the call site; Plain is the floor).
            for variant in [GemvVariant::Plain, GemvVariant::Prerotated] {
                if fam.resolve(d, variant, false, &ctx, None).is_err() {
                    failures.push(format!("  {:?} / {:?} dead-gated on {}", d, variant, arch));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\nW4 regression: MQ3/Lloyd GEMV still dead-gated on RDNA1/2:\n{}\n",
        failures.join("\n")
    );
}

/// W4 (FIX B): the MQ3/MQ4-Lloyd FUSED keys (QKV / QKVZA / GateUp) resolve on
/// RDNA1/2 after the fused_qkv_table HasWmma→HasWave32 flip, and STILL Err on
/// CDNA wave64. Covers the fused-table dead-gate distinct from the GEMV one above.
#[test]
fn w4_lloyd_fused_keys_un_bricked_on_rdna2_not_cdna() {
    use crate::families::fused_qkv::FusedQkvFamily;
    let family = FusedQkvFamily::new();
    let lloyd_fused: &[KernelKey] = &[
        KernelKey::FusedQkvMq3G256Lloyd,
        KernelKey::FusedQkvMq4G256Lloyd,
        KernelKey::FusedQkvzaMq3G256Lloyd,
        KernelKey::FusedQkvzaMq4G256Lloyd,
        KernelKey::FusedGateUpMq3G256Lloyd,
        KernelKey::FusedGateUpMq4G256Lloyd,
    ];
    let mut failures = Vec::new();
    for &key in lloyd_fused {
        // RDNA1/2 must now resolve.
        for arch in ["gfx1010", "gfx1030", "gfx1031", "gfx1032"] {
            if family
                .resolve(key, &DispatchCtx::for_test(arch), None)
                .is_err()
            {
                failures.push(format!("  {:?} dead-gated on {} (FIX B)", key, arch));
            }
        }
        // CDNA wave64 must still Err.
        for arch in ["gfx906", "gfx942"] {
            if family
                .resolve(key, &DispatchCtx::for_test(arch), None)
                .is_ok()
            {
                failures.push(format!("  {:?} wrongly admitted on CDNA {}", key, arch));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\nW4 FIX B regression on fused Lloyd keys:\n{}\n",
        failures.join("\n")
    );
}

/// W4 NON-REGRESSION: RDNA3 (gfx1100) and RDNA4 (gfx1201) resolve MQ3/Lloyd to the
/// SAME KernelKey as each other and as RDNA2 — byte-identical routing. A dispatch-
/// core superset change MUST NOT perturb the resolved variant on RDNA3/4.
#[test]
fn w4_mq3_lloyd_routing_byte_identical_on_rdna3_rdna4() {
    use crate::families::gemv::GemvFamily;
    let fam = GemvFamily::new();
    let mut failures = Vec::new();
    for &d in W4_WAVE32_DTYPES {
        for variant in [GemvVariant::Plain, GemvVariant::Prerotated] {
            let key_rdna2 = fam
                .resolve(d, variant, false, &DispatchCtx::for_test("gfx1030"), None)
                .ok()
                .map(|v| v.key);
            let key_rdna3 = fam
                .resolve(d, variant, false, &DispatchCtx::for_test("gfx1100"), None)
                .ok()
                .map(|v| v.key);
            let key_rdna4 = fam
                .resolve(d, variant, false, &DispatchCtx::for_test("gfx1201"), None)
                .ok()
                .map(|v| v.key);
            // All three must resolve to the same Some(KernelKey).
            if !(key_rdna3.is_some() && key_rdna3 == key_rdna4 && key_rdna3 == key_rdna2) {
                failures.push(format!(
                    "  {:?} / {:?}: rdna2={:?} rdna3={:?} rdna4={:?} (routing diverged)",
                    d, variant, key_rdna2, key_rdna3, key_rdna4
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\nW4 NON-REGRESSION FAIL — RDNA3/4 routing changed:\n{}\n",
        failures.join("\n")
    );
}

/// W4: CDNA (gfx906/gfx942 wave64) MUST STILL Err — HasWave32 excludes wave64;
/// a [32,1,1] kernel can't run on a wave64 lane layout. Guards against the lazy
/// "blanket Always" fix that would wrongly admit CDNA.
#[test]
fn w4_mq3_lloyd_still_rejected_on_cdna_wave64() {
    use crate::families::gemv::GemvFamily;
    let fam = GemvFamily::new();
    let cdna: &[&str] = &["gfx906", "gfx942"];
    let mut admitted = Vec::new();
    for &d in W4_WAVE32_DTYPES {
        for &arch in cdna {
            let ctx = DispatchCtx::for_test(arch);
            for variant in [GemvVariant::Plain, GemvVariant::Prerotated] {
                if fam.resolve(d, variant, false, &ctx, None).is_ok() {
                    admitted.push(format!(
                        "  {:?} / {:?} wrongly admitted on {}",
                        d, variant, arch
                    ));
                }
            }
        }
    }
    assert!(
        admitted.is_empty(),
        "\nW4: CDNA wave64 wrongly admitted to wave32 [32,1,1] kernel:\n{}\n",
        admitted.join("\n")
    );
}

// ── MQV2 completeness — every admitted uniform/mixed route has exact keys ─────
//
// Pins the structural contract for MQ4G256V2 (qt44) and MQ6G256V2 (qt47):
//   * dense KernelKey mappings never alias V1 HFQ4/HFQ6 siblings
//   * uniform MoE resolution claims the V2 indexable arm only when BOTH halves match
//   * pipeline route-kind helpers return exact "mq4v2"/"mq6v2" identities
//   * mixed tags 7..18 are complete and never collapse onto V1 tags 0..6
//   * fused/GEMM keys resolve on gfx11 + gfx12
// GPU-free; no product logic.

fn moe_dtypes_uniform(gate_up: DType, down: DType) -> MoeDtypes {
    MoeDtypes {
        router: Q8_0,
        shared_gate: Q8_0,
        shared_expert_gate: Q8_0,
        shared_expert_up: Q8_0,
        shared_expert_down: Q8_0,
        experts_all_gate_up_mq4: false,
        routed_gate_up: gate_up,
        routed_down: down,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
        routed_escha_transforms: false,
    }
}

/// LAYER MQV2-1 — exact dense KernelKey identity for every admitted V2 role.
/// Catches a missing residual/prerotated/swiglu arm AND a silent V1 alias.
#[test]
fn mqv2_uniform_dense_keys_are_exact_not_v1() {
    let cases: &[(DType, KernelKey, KernelKey, KernelKey)] = &[
        (
            MQ4G256V2,
            KernelKey::GemvMq4G256V2Prerotated,
            KernelKey::GemvMq4G256V2Residual,
            KernelKey::GemvMq4G256V2SwiGLUResidual,
        ),
        (
            MQ6G256V2,
            KernelKey::GemvMq6G256V2Prerotated,
            KernelKey::GemvMq6G256V2Residual,
            KernelKey::GemvMq6G256V2SwiGLUResidual,
        ),
    ];
    let mut failures = Vec::new();
    for &(dtype, want_prerot, want_res, want_swiglu) in cases {
        // No Plain arm — V2 is prerotated-only at the KernelKey layer.
        if KernelKey::for_gemv(dtype, GemvVariant::Plain, false).is_ok() {
            failures.push(format!(
                "  {:?}: unexpectedly has Plain GEMV key (must stay prerotated-only)",
                dtype
            ));
        }
        match KernelKey::for_gemv_prerotated(dtype) {
            Ok(k) if k == want_prerot => {}
            Ok(k) => failures.push(format!(
                "  {:?}: prerotated key {:?} ≠ exact {:?}",
                dtype, k, want_prerot
            )),
            Err(_) => failures.push(format!("  {:?}: missing prerotated key", dtype)),
        }
        match KernelKey::for_gemv_residual(dtype) {
            Ok(k) if k == want_res => {}
            Ok(k) => failures.push(format!(
                "  {:?}: residual key {:?} ≠ exact {:?}",
                dtype, k, want_res
            )),
            Err(_) => failures.push(format!("  {:?}: missing residual key", dtype)),
        }
        match KernelKey::for_gemv_swiglu_residual(dtype) {
            Ok(k) if k == want_swiglu => {}
            Ok(k) => failures.push(format!(
                "  {:?}: swiglu key {:?} ≠ exact {:?}",
                dtype, k, want_swiglu
            )),
            Err(_) => failures.push(format!("  {:?}: missing swiglu residual key", dtype)),
        }
        // V1 residual keys must NOT be returned for V2 dtypes.
        let v1_res = match dtype {
            MQ4G256V2 => KernelKey::GemvMq4G256Residual,
            MQ6G256V2 => KernelKey::GemvMq6G256Residual,
            _ => unreachable!(),
        };
        if matches!(KernelKey::for_gemv_residual(dtype), Ok(k) if k == v1_res) {
            failures.push(format!(
                "  {:?}: residual collapsed onto V1 key {:?}",
                dtype, v1_res
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\n{} MQV2 dense key identity failures:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER MQV2-2 — uniform MoE decode resolution + GPU-top-K positive controls.
/// k=8 both-sides-V2 → indexable + use_gpu_topk; split V1/V2 → neither arm.
#[test]
fn mqv2_uniform_moe_routes_are_indexable_and_exact() {
    let mut failures = Vec::new();

    // Positive: uniform MQ4V2 / MQ6V2 at k=8 take the native V2 indexed path.
    for (name, gu, dn, want_mq4v2, want_mq6v2) in [
        ("mq4v2-uniform-k8", MQ4G256V2, MQ4G256V2, true, false),
        ("mq6v2-uniform-k8", MQ6G256V2, MQ6G256V2, false, true),
    ] {
        let r = MoeResolution::resolve(&moe_dtypes_uniform(gu, dn), 8);
        if r.routed_indexable_mq4v2 != want_mq4v2 {
            failures.push(format!(
                "  {}: routed_indexable_mq4v2={} want {}",
                name, r.routed_indexable_mq4v2, want_mq4v2
            ));
        }
        if r.routed_indexable_mq6v2 != want_mq6v2 {
            failures.push(format!(
                "  {}: routed_indexable_mq6v2={} want {}",
                name, r.routed_indexable_mq6v2, want_mq6v2
            ));
        }
        // Must never claim the V1 sibling arm.
        if r.routed_indexable_mq4 {
            failures.push(format!("  {}: wrongly claimed routed_indexable_mq4", name));
        }
        if r.routed_indexable_mq6 {
            failures.push(format!("  {}: wrongly claimed routed_indexable_mq6", name));
        }
        if !r.use_gpu_topk {
            failures.push(format!(
                "  {}: use_gpu_topk=false — uniform V2 k=8 must take indexed path",
                name
            ));
        }
        if !r.needs_x_rot_local {
            failures.push(format!(
                "  {}: needs_x_rot_local=false — V2 kernels read ROTATED x",
                name
            ));
        }
        if !r.routed_indexable() {
            failures.push(format!("  {}: routed_indexable() false", name));
        }
    }

    // Negative: any V1/V2 or cross-V2 split is not indexable on either arm.
    for (name, gu, dn) in [
        ("mq4v2/mq4v1", MQ4G256V2, MQ4G256),
        ("mq4v1/mq4v2", MQ4G256, MQ4G256V2),
        ("mq6v2/mq6v1", MQ6G256V2, MQ6G256),
        ("mq6v1/mq6v2", MQ6G256, MQ6G256V2),
        ("mq4v2/mq6v2", MQ4G256V2, MQ6G256V2),
        ("mq6v2/mq4v2", MQ6G256V2, MQ4G256V2),
    ] {
        let r = MoeResolution::resolve(&moe_dtypes_uniform(gu, dn), 8);
        if r.routed_indexable_mq4v2
            || r.routed_indexable_mq6v2
            || r.routed_indexable_mq4
            || r.routed_indexable_mq6
        {
            failures.push(format!(
                "  {}: split pair claimed a uniform indexable arm (mq4v2={} mq6v2={} mq4={} mq6={})",
                name,
                r.routed_indexable_mq4v2,
                r.routed_indexable_mq6v2,
                r.routed_indexable_mq4,
                r.routed_indexable_mq6
            ));
        }
        if r.use_gpu_topk {
            failures.push(format!(
                "  {}: use_gpu_topk=true on split pair — must fall back, not guess layout",
                name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{} MQV2 uniform MoE resolution failures:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER MQV2-3 — pipeline route-kind helpers admit exact V2 identities.
/// Decode down / Path1 gate+down / grouped GEMM / ninepath D4 / shared dense.
/// Tag-aware Path1 mixed precedence is pinned here too.
#[test]
fn mqv2_admitted_route_kinds_are_exact() {
    use crate::pipeline::{
        decode_expanded_down_kind, decode_gate_uses_mixed, gate_up_varies, grouped_gemm_kind,
        ninepath_d4_family, prefill_path1_down_kind, prefill_path1_down_kind_tag_aware,
        prefill_path1_gate_up_kind, prefill_path1_gate_up_kind_tag_aware, shared_dense_down_kind,
    };

    let mut failures = Vec::new();

    // Uniform decode / prefill / grouped kinds.
    for (dtype, kind) in [(MQ4G256V2, "mq4v2"), (MQ6G256V2, "mq6v2")] {
        for (label, got) in [
            ("decode_expanded_down", decode_expanded_down_kind(dtype)),
            ("path1_gate_up", prefill_path1_gate_up_kind(dtype)),
            ("path1_down", prefill_path1_down_kind(dtype)),
            ("grouped_gemm", grouped_gemm_kind(dtype)),
        ] {
            if got != Some(kind) {
                failures.push(format!(
                    "  {}({:?}) = {:?} want Some({:?})",
                    label, dtype, got, kind
                ));
            }
        }
        // Must never share the V1 kind string.
        let v1_kind = match dtype {
            MQ4G256V2 => "hfq4",
            MQ6G256V2 => "hfq6",
            _ => unreachable!(),
        };
        if decode_expanded_down_kind(dtype) == Some(v1_kind) {
            failures.push(format!(
                "  decode_expanded_down({:?}) collapsed to V1 {:?}",
                dtype, v1_kind
            ));
        }
    }

    // Ninepath D4: only exact both-sides uniform pairs.
    if ninepath_d4_family(MQ4G256V2, MQ4G256V2) != Some("mq4v2") {
        failures.push("  ninepath_d4(MQ4V2,MQ4V2) ≠ Some(mq4v2)".into());
    }
    if ninepath_d4_family(MQ6G256V2, MQ6G256V2) != Some("mq6v2") {
        failures.push("  ninepath_d4(MQ6V2,MQ6V2) ≠ Some(mq6v2)".into());
    }
    for (gu, dn) in [
        (MQ4G256V2, MQ4G256),
        (MQ4G256, MQ4G256V2),
        (MQ6G256V2, MQ6G256),
        (MQ4G256V2, MQ6G256V2),
    ] {
        if ninepath_d4_family(gu, dn).is_some() {
            failures.push(format!(
                "  ninepath_d4({:?},{:?}) wrongly admitted {:?}",
                gu,
                dn,
                ninepath_d4_family(gu, dn)
            ));
        }
    }

    // Shared dense down: V2 prerotated, never V1 sigmoid_scaled.
    if shared_dense_down_kind(MQ4G256V2) != Some("mq4v2_prerotated") {
        failures.push(format!(
            "  shared_dense_down(MQ4V2) = {:?} want Some(mq4v2_prerotated)",
            shared_dense_down_kind(MQ4G256V2)
        ));
    }
    if shared_dense_down_kind(MQ6G256V2) != Some("mq6v2_prerotated") {
        failures.push(format!(
            "  shared_dense_down(MQ6V2) = {:?} want Some(mq6v2_prerotated)",
            shared_dense_down_kind(MQ6G256V2)
        ));
    }
    if shared_dense_down_kind(MQ4G256V2) == shared_dense_down_kind(MQ4G256) {
        failures.push("  shared_dense_down MQ4V2 collapsed onto MQ4V1".into());
    }

    // Mixed Path1: tags + varying gate_up → mixed gate; tags alone → mixed down;
    // uniform (no tags / no variation) keeps exact V2 kinds.
    if !decode_gate_uses_mixed(true, true) {
        failures.push("  decode_gate_uses_mixed(tags, varies) must be true".into());
    }
    if decode_gate_uses_mixed(true, false) {
        failures.push(
            "  decode_gate_uses_mixed(tags, !varies) must be false (uniform shortcut)".into(),
        );
    }
    if gate_up_varies(Some(&[MQ4G256V2, MQ4G256])) != true {
        failures.push("  gate_up_varies(MQ4V2,MQ4V1) must be true".into());
    }
    if gate_up_varies(Some(&[MQ4G256V2, MQ4G256V2])) {
        failures.push("  gate_up_varies(all MQ4V2) must be false".into());
    }
    if prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, true) != Some("mixed") {
        failures.push("  path1 gate tag-aware varies → mixed".into());
    }
    if prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, false) != Some("mq4v2") {
        failures.push("  path1 gate tag-aware uniform MQ4V2 → mq4v2".into());
    }
    if prefill_path1_gate_up_kind_tag_aware(MQ6G256V2, false, false) != Some("mq6v2") {
        failures.push("  path1 gate no-tags MQ6V2 → mq6v2".into());
    }
    if prefill_path1_down_kind_tag_aware(MQ4G256V2, true) != Some("mixed") {
        failures.push("  path1 down has_tags → mixed (no representative dispatch)".into());
    }
    if prefill_path1_down_kind_tag_aware(MQ6G256V2, false) != Some("mq6v2") {
        failures.push("  path1 down no-tags MQ6V2 → mq6v2".into());
    }

    assert!(
        failures.is_empty(),
        "\n{} MQV2 route-kind failures:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER MQV2-4 — mixed tag matrix completeness (frozen tags 7..18).
/// Every admitted pair has its exact tag; unknown pairs refuse; no V1 overlap.
#[test]
fn mqv2_mixed_routes_have_exact_tags_and_admission() {
    use crate::families::moe::MIXED_SUPPORTED_TIERS;
    use crate::pipeline::mixed_expert_dtype_tag;

    // Exact frozen pairs from mqv2-kernel-contracts.md.
    let admitted: &[(DType, DType, u8)] = &[
        (MQ4G256V2, MQ4G256V2, 7),
        (MQ6G256V2, MQ6G256V2, 8),
        (MQ4G256V2, MQ6G256, 9),
        (MQ4G256V2, MQ2G256Lloyd, 10),
        (MQ4G256V2, MQ4G256, 11),
        (MQ4G256, MQ4G256V2, 12),
        (MQ4G256V2, MQ3G256Lloyd, 13),
        (MQ4G256V2, MFP4G32E8, 14),
        (MQ4G256V2, MFP3G32E8, 15),
        (MQ4G256V2, MFP2G32E8, 16),
        (MQ4G256V2, MQ6G256V2, 17),
        (MQ4G256, MQ6G256V2, 18),
    ];
    let mut failures = Vec::new();
    let mut seen_tags = std::collections::HashSet::new();
    for &(gate, down, tag) in admitted {
        match mixed_expert_dtype_tag(gate, down) {
            Some(got) if got == tag => {
                seen_tags.insert(got);
            }
            Some(got) => failures.push(format!(
                "  tag({:?},{:?}) = {} want {}",
                gate, down, got, tag
            )),
            None => failures.push(format!(
                "  tag({:?},{:?}) = None want Some({})",
                gate, down, tag
            )),
        }
        // Tags 7..18 must never collide with V1 0..6.
        if tag <= 6 {
            failures.push(format!(
                "  admitted pair ({:?},{:?}) assigned V1-range tag {}",
                gate, down, tag
            ));
        }
    }
    // Completeness: every tag in 7..=18 appears exactly once.
    for t in 7u8..=18 {
        if !seen_tags.contains(&t) {
            failures.push(format!("  missing admitted tag {t} in frozen matrix"));
        }
    }

    // Loud reject: reverse MQ6V2/MQ4V2 and GL pairs.
    for (gate, down) in [
        (MQ6G256V2, MQ4G256V2),
        (MQ6G256V2, MQ4G256),
        (MQ6G256V2, MQ6G256),
        (MQ2G256GL, MQ4G256V2),
        (MQ4G256V2, MQ2G256GL),
    ] {
        if mixed_expert_dtype_tag(gate, down).is_some() {
            failures.push(format!(
                "  tag({:?},{:?}) should refuse (None), got {:?}",
                gate,
                down,
                mixed_expert_dtype_tag(gate, down)
            ));
        }
    }

    // MIXED_SUPPORTED_TIERS must list exactly the dtypes mixed kernels consume.
    for dt in [MQ4G256, MQ6G256, ParoQ4G128, MQ4G256V2, MQ6G256V2] {
        if !MIXED_SUPPORTED_TIERS.contains(&dt) {
            failures.push(format!(
                "  MIXED_SUPPORTED_TIERS missing {:?} (mixed branches consume it)",
                dt
            ));
        }
    }
    if MIXED_SUPPORTED_TIERS.len() != 5 {
        failures.push(format!(
            "  MIXED_SUPPORTED_TIERS len={} want 5 (no MQ2/3/5V2 widen)",
            MIXED_SUPPORTED_TIERS.len()
        ));
    }
    for dt in [MQ5G256V2, MQ2G256V2, MQ3G256V2] {
        if MIXED_SUPPORTED_TIERS.contains(&dt) {
            failures.push(format!("  MIXED_SUPPORTED_TIERS wrongly admits {:?}", dt));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{} MQV2 mixed-route completeness failures:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

/// LAYER MQV2-5 — fused QKV/QKVZA/GateUp + GEMM keys resolve on gfx11/gfx12.
/// Completeness over every admitted dense shared V2 dispatch key.
#[test]
fn mqv2_fused_and_gemm_keys_resolve_on_gfx11_gfx12() {
    use crate::families::fused_qkv::FusedQkvFamily;
    use crate::families::gemm::GemmFamily;

    let fused_keys: &[KernelKey] = &[
        KernelKey::FusedQkvMq4G256V2,
        KernelKey::FusedQkvzaMq4G256V2,
        KernelKey::FusedGateUpMq4G256V2,
        KernelKey::FusedQkvMq6G256V2,
        KernelKey::FusedQkvzaMq6G256V2,
        KernelKey::FusedGateUpMq6G256V2,
    ];
    let gemm_keys: &[KernelKey] = &[
        KernelKey::GemmMq4G256V2,
        KernelKey::GemmMq4G256V2Residual,
        KernelKey::GemmMq4G256V2BatchedLmhead,
        KernelKey::GemmMq6G256V2,
        KernelKey::GemmMq6G256V2Residual,
        KernelKey::GemmMq6G256V2BatchedLmhead,
    ];
    // Fused V2 table entries are Always-gated; GEMM V2 is HasWmma.
    let fused_archs: &[&str] = WAVE32;
    let gemm_archs: &[&str] = WMMA_ARCHS;

    let fused = FusedQkvFamily::new();
    let gemm = GemmFamily::new();
    let mut failures = Vec::new();

    for &key in fused_keys {
        for &arch in fused_archs {
            let ctx = DispatchCtx::for_test(arch);
            if fused.resolve(key, &ctx, None).is_err() {
                failures.push(format!("  fused {:?} dead-gated on {}", key, arch));
            }
        }
    }
    for &key in gemm_keys {
        for &arch in gemm_archs {
            let ctx = DispatchCtx::for_test(arch);
            // GemmFamily::resolve is dtype-keyed; residual/lmhead keys are
            // registered for explicit run_key and must resolve via the registry.
            if gemm.registry().resolve(key, &ctx, None).is_err() {
                failures.push(format!("  gemm {:?} dead-gated on {}", key, arch));
            }
        }
    }

    // Also pin GemmFamily dtype→key selection for uniform V2 prefill.
    for (dtype, want) in [
        (MQ4G256V2, KernelKey::GemmMq4G256V2),
        (MQ6G256V2, KernelKey::GemmMq6G256V2),
    ] {
        for arch in ["gfx1100", "gfx1201"] {
            let ctx = DispatchCtx::for_test(arch);
            match gemm.resolve(dtype, &ctx, None) {
                Ok(v) if v.key == want => {}
                Ok(v) => failures.push(format!(
                    "  GemmFamily::resolve({:?}, {}) key {:?} ≠ {:?}",
                    dtype, arch, v.key, want
                )),
                Err(e) => failures.push(format!(
                    "  GemmFamily::resolve({:?}, {}) Err({e:?})",
                    dtype, arch
                )),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "\n{} MQV2 fused/GEMM resolve failures:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}
