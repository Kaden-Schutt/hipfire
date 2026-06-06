// SPDX-License-Identifier: MIT OR Apache-2.0
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
    "gfx1200", "gfx1201",            // RDNA4
];

/// Archs with WMMA support (`has_wmma = is_rdna3 || is_rdna4`).
/// Distinct from wave32: RDNA1/2 are wave32 but lack WMMA.
const WMMA_ARCHS: &[&str] = WAVE32; // same set today, but semantically distinct

/// Everything incl. RDNA1/2 + CDNA, for dtypes whose arch gate is Always/dp4a.
const ALL: &[&str] = &[
    "gfx1010", "gfx1030", "gfx1031", "gfx1032",
    "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152",
    "gfx1200", "gfx1201", "gfx906", "gfx908", "gfx942",
];

/// PRODUCTION MATRIX — (model, role, dtype) the wired forward pass hits today.
/// The Q8_0 `Residual` rows are the ones the live gap panicked on (now fixed via
/// the plain-GEMV fallback).
const FLEET: &[OpUse] = &[
    // ── q8f16: Q8 weights throughout. o_proj reaches Step::GemvResidual on every arch ──
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::Residual, dtype: Q8_0, archs: ALL },
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::Plain,    dtype: Q8_0, archs: ALL },
    OpUse { model: "qwen3.5-9b.q8f16", role: Role::SwigluResidual, dtype: Q8_0, archs: ALL },

    // ── qwen3.6-35b-a3b MoE: Q8 attention o_proj ──
    OpUse { model: "qwen3.6-35b-a3b.mq4", role: Role::Residual, dtype: Q8_0, archs: WAVE32 },

    // ── Paro o_proj (no fused residual kernel → uses the same fallback) ──
    OpUse { model: "qwen3.5-*.paro4g128", role: Role::Residual, dtype: ParoQ4G128, archs: WAVE32 },

    // ── dense MQ4/MQ3/Lloyd: o_proj has a fused residual kernel — anchors (stay green) ──
    OpUse { model: "qwen3.5-27b.mq4",       role: Role::Residual, dtype: MQ4G256,      archs: WAVE32 },
    OpUse { model: "qwen3.5-27b.mq3",       role: Role::Residual, dtype: MQ3G256,      archs: WAVE32 },
    OpUse { model: "qwen3.6-27b.mq3-lloyd", role: Role::Residual, dtype: MQ3G256Lloyd, archs: WAVE32 },
    OpUse { model: "qwen3.6-35b-a3b.mq4",   role: Role::Plain,    dtype: MQ4G256,      archs: WAVE32 },
    // MQ6-promoted projections (A3B AWQ-attractor mitigation) — gate is HasMmq (must admit gfx12):
    OpUse { model: "qwen3.6-35b-a3b.mq4",   role: Role::Plain,    dtype: MQ6G256,      archs: WAVE32 },

    // ── RDNA4 coverage: same (role, dtype) combos explicitly anchored on gfx12 arch strings ──
    // Catches reintroduction of an RDNA4-only ArchPredicate that would dead-gate these.
    OpUse { model: "qwen3.5-27b.mq4-rdna4",   role: Role::Plain,    dtype: MQ4G256,      archs: &["gfx1200", "gfx1201"] },
    OpUse { model: "qwen3.5-27b.mq3-rdna4",   role: Role::Plain,    dtype: MQ3G256,      archs: &["gfx1200", "gfx1201"] },
    OpUse { model: "qwen3.6-27b.lloyd-rdna4", role: Role::Plain,    dtype: MQ3G256Lloyd, archs: &["gfx1200", "gfx1201"] },
    OpUse { model: "a3b-moe-rdna4",           role: Role::Plain,    dtype: MQ6G256,      archs: &["gfx1200", "gfx1201"] },
];

/// Does the forward lowering for (role, dtype) have ANY dispatch plan (so it
/// cannot hit an `UnsupportedVariant` panic)? Mirrors the real lowering:
/// - Plain          → needs a plain GEMV.
/// - Residual       → fused `gemv_*_residual` kernel OR plain-GEMV-into-temp + add.
/// - SwigluResidual → fused swiglu-residual kernel OR plain-GEMV fallback.
fn has_dispatch_plan(role: Role, dtype: DType) -> bool {
    let plain = KernelKey::for_gemv(dtype, GemvVariant::Plain, false).is_ok();
    match role {
        Role::Plain          => plain,
        Role::Residual       => KernelKey::for_gemv_residual(dtype).is_ok() || plain,
        Role::SwigluResidual => KernelKey::for_gemv_swiglu_residual(dtype).is_ok() || plain,
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
        Q8_0, MQ4G256, MQ3G256, MQ6G256, HFQ4G256, HFQ6G256,
        MQ3G256Lloyd, MQ4G256Lloyd, ParoQ4G128, MFP4G32, Q4K,
    ];
    let no_fused: Vec<_> = OPROJ_DTYPES
        .iter()
        .filter(|d| KernelKey::for_gemv_residual(**d).is_err())
        .collect();
    if !no_fused.is_empty() {
        eprintln!("residual dtypes with no FUSED kernel (use plain+add fallback): {:?}", no_fused);
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
/// routed experts are `{MQ4G256, MQ6G256, ParoQ4G128}`. Every OTHER MoE layer
/// (`k != 8`, or a routed dtype like Q8_0) MUST take the generic CPU-top-K
/// per-expert fallback — #393 deleted that fallback so those layers hit
/// `UnsupportedVariant{cpu-topk-fallback}` and HARD-PANIC on decode.
///
/// GPU-free assertion in two parts, mirroring the runtime guarantees:
///   (a) The eligibility lattice routes these layers to the fallback, NOT the
///       k8 indexed path: `MoeResolution::resolve(..).use_gpu_topk == false`.
///   (b) The fallback's per-expert loop dispatches gate_up + down through
///       `GemvFamily::run_auto`, so the routed dtype MUST have a plain-GEMV
///       dispatch plan (else `run_auto` → `UnsupportedVariant`).
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
    let mut failures = Vec::new();
    for u in [
        // Q8-routed experts, k=8 → not indexable → CPU-top-K fallback.
        MoeUse { name: "q8-routed-moe (k=8)", routed_gate_up: Q8_0, routed_down: Q8_0, k: 8 },
        // MQ4 routed but k != 8 → k8 indexed kernels unusable → fallback.
        MoeUse { name: "mq4-routed-moe (k=4)", routed_gate_up: MQ4G256, routed_down: MQ4G256, k: 4 },
        // F32 routed experts, k=2 → fallback.
        MoeUse { name: "f32-routed-moe (k=2)", routed_gate_up: F32, routed_down: F32, k: 2 },
    ] {
        let d = MoeDtypes {
            router: Q8_0,
            shared_gate: Q8_0,
            shared_expert_gate: Q8_0,
            shared_expert_up: Q8_0,
            experts_all_gate_up_mq4: u.routed_gate_up == MQ4G256,
            routed_gate_up: u.routed_gate_up,
            routed_down: u.routed_down,
            has_paro_shared: false,
        };
        let res = MoeResolution::resolve(&d, u.k);
        // (a) These layers MUST take the fallback, not the k8 indexed path.
        if res.use_gpu_topk {
            failures.push(format!(
                "  {}: resolved to GPU-top-K (use_gpu_topk=true) but routed dtype/k is non-indexable",
                u.name
            ));
        }
        // (b) The fallback's run_auto needs a plain-GEMV plan for both halves.
        if !KernelKey::for_gemv(u.routed_gate_up, GemvVariant::Plain, false).is_ok() {
            failures.push(format!(
                "  {}: routed gate_up {:?} has no plain GEMV → fallback run_auto → UnsupportedVariant panic",
                u.name, u.routed_gate_up
            ));
        }
        if !KernelKey::for_gemv(u.routed_down, GemvVariant::Plain, false).is_ok() {
            failures.push(format!(
                "  {}: routed down {:?} has no plain GEMV → fallback run_auto → UnsupportedVariant panic",
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

/// LAYER 1c — Q8/Paro were gapped in MULTIPLE GEMV variants: o_proj used Residual,
/// then the FFN/qkv used Prerotated (the second panic domino). Lock every variant
/// these dtypes are actually dispatched through.
#[test]
fn q8_and_paro_dispatchable_in_all_used_variants() {
    for d in [Q8_0, ParoQ4G128] {
        assert!(
            KernelKey::for_gemv(d, GemvVariant::Plain, false).is_ok(),
            "{:?}: plain GEMV missing (lm_head / direct GEMV panics)", d
        );
        assert!(
            KernelKey::for_gemv_prerotated(d).is_ok(),
            "{:?}: prerotated GEMV missing (FFN / qkv prerotated path panics)", d
        );
        assert!(
            has_dispatch_plan(Role::Residual, d),
            "{:?}: residual GEMV has no plan (o_proj panics)", d
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
    for d in [F16, F32, Q4K, Q6K, HFQ3G256, HFQ6G256, HFQ2G256, HFP4G32, Q8_0] {
        assert!(
            KernelKey::for_gemv_prerotated(d).is_ok(),
            "for_gemv_prerotated({:?}) errors — the unfused FA fallback panics where the \
             legacy run_auto->Plain path worked", d
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
        AttnKeyUse { key: KernelKey::KvWriteF32,            archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteQ8_0,           archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym4,          archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym4Fwht,      archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym3,          archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym3Fwht,      archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym2,          archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::KvWriteAsym2Fwht,      archs: ALL, shape: None },
        // KV write — batched, Always-gated, BatchGt(1)
        AttnKeyUse { key: KernelKey::KvWriteAsym4Batched,          archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteAsym4FwhtBatched,     archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteAsym3Batched,          archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteAsym3FwhtBatched,     archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteAsym2Batched,          archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteAsym2FwhtBatched,     archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::KvWriteQ8_0Batched,           archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        // Attention — single-token, Always-gated
        AttnKeyUse { key: KernelKey::AttnF32,               archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashQ8_0,         archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnQ8_0Kv,            archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym4,        archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym4Fwht,    archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym3,        archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym3Fwht,    archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym2,        archs: ALL, shape: None },
        AttnKeyUse { key: KernelKey::AttnFlashAsym2Fwht,    archs: ALL, shape: None },
        // GQA-fused — HasWmma-gated
        AttnKeyUse { key: KernelKey::AttnGqaFused,          archs: WMMA_ARCHS, shape: None },
        // Attention — batched, Always-gated (scalar fallback), BatchGt(1)
        AttnKeyUse { key: KernelKey::AttnFlashAsym4BatchedMasked,          archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnFlashAsym4FwhtBatchedMasked,     archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnFlashAsym3BatchedMasked,          archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnFlashAsym3FwhtBatchedMasked,     archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnFlashAsym2Batched,                archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnFlashAsym2FwhtBatched,           archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
        AttnKeyUse { key: KernelKey::AttnQ8_0KvBatchedMasked,             archs: ALL, shape: Some(ShapeInfo { batch_size: 16, head_dim: 128, m: 0, is_tree: false }) },
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
