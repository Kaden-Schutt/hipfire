//! DeepSeek V4 Flash model family dispatch tests.
//!
//! arch_id=9. Most specialized: hyper-connections (4 residual streams),
//! compressed-KV indexer, tail-only RoPE, Q/O-LoRA, FP4 experts,
//! sliding window attention (SWA), separate MTP spec-decode layer.
//!
//! CPU-only: no GPU hardware required. The DeepSeek4 arch crate is not a
//! dependency of this crate, so the per-layer `compress_ratio` / MTP wiring
//! is asserted through the shared dispatch + runtime surfaces the arch
//! consumes (typed router plans, the sealed MoE lowerer's identity gate, the
//! MoE resolution lattice, the KV-mode ladder, and the kernel registry).
//! `GpuTensor::null_for_test()` buffers carry no capacity, so full program
//! lowering stops at the lowerer's capacity validation — the identity gate
//! (which runs before it) is exactly what these tests exercise.

use rdna_compute::DType;

#[test]
fn deepseek4_prefill_batchable_formats() {
    use hipfire_runtime::llama::is_batchable_la;
    // DeepSeek V4 uses MQ4, Q8_0, and F16/F32 for its layers.
    for &arch in &["gfx1100", "gfx942"] {
        assert!(
            is_batchable_la(DType::MQ4G256, arch),
            "MQ4G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::HFQ4G256, arch),
            "HFQ4G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::Q8_0, arch),
            "Q8_0 batchable on {arch}"
        );
    }
}

// ── CPU test scaffolding ───────────────────────────────────────────────

fn null_f32(numel: usize) -> &'static rdna_compute::GpuTensor {
    let mut t = rdna_compute::GpuTensor::null_for_test();
    t.shape = vec![numel];
    t.dtype = DType::F32;
    Box::leak(Box::new(t))
}

fn null_i64(numel: usize) -> &'static rdna_compute::GpuTensor {
    let mut t = rdna_compute::GpuTensor::null_for_test();
    t.shape = vec![numel];
    t.dtype = DType::Raw;
    Box::leak(Box::new(t))
}

fn expert_ref() -> &'static hipfire_dispatch::families::moe::MoeExpertRef<'static> {
    Box::leak(Box::new(hipfire_dispatch::families::moe::MoeExpertRef {
        gate_up_ptrs: null_f32(4),
        down_ptrs: null_f32(4),
        dummy_gate_up: None,
        dtype: DType::MQ2G256Lloyd,
        n_experts: 8,
        expert_m: 48,
        expert_k: 64,
        owned: &[],
    }))
}

fn group(
    layer: Option<usize>,
    router_identity: &str,
    parallelism: hipfire_runtime::weight_manifest::ExpertParallelism,
    group_size: usize,
) -> hipfire_runtime::weight_manifest::ExpertGroupPlan {
    use hipfire_runtime::weight_manifest::*;
    let collective = match parallelism {
        ExpertParallelism::Single => None,
        ExpertParallelism::TensorParallel => Some(ExpertPostCombineAllReduce::TensorParallel),
        ExpertParallelism::ExpertParallel => Some(ExpertPostCombineAllReduce::ExpertParallel),
    };
    ExpertGroupPlan {
        group: "deepseek4.moe".into(),
        layer,
        n_experts: 8,
        group_size,
        parallelism,
        assignment: hipfire_runtime::tp_shard::ExpertAssign::Stride,
        experts: Vec::new(),
        source_layout: ExpertSourceLayout::PackedFused {
            gate_up: "experts_gate".into(),
            down: "experts_down".into(),
            sidecars: Vec::new(),
        },
        resources: ExpertResourceRequirements {
            bytes_per_expert: 1,
            alignment: 256,
        },
        router: "router_gate".into(),
        router_identity: router_identity.into(),
        allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
        collective,
    }
}

/// The DS4 routed program shape: [GateUp, MoeActivation, DownResidualI64,
/// ConvertI64ToF32] — the unified int64 path shared by main-layer decode,
/// MTP, and batched TP prefill. Null buffers: lowering stops at capacity
/// validation AFTER the identity gate, which is what the tests exercise.
fn i64_rank() -> hipfire_runtime::moe_plan::RoutedMoeStepPhases<'static> {
    use hipfire_dispatch::pipeline::{GemvInput, MoeActivationVariant, MoeProj, Step};
    use hipfire_runtime::moe_plan::RoutedMoePhases;
    let partial_i64 = null_i64(64);
    let down = Step::IndexedMoeGemv {
        experts: expert_ref(),
        which: MoeProj::DownResidualI64 {
            topk_weights: null_f32(2),
        },
        topk_indices: null_i64(2),
        input: GemvInput::Raw(null_f32(96)),
        out: partial_i64,
        k_top: 2,
        batch_size: 1,
    };
    RoutedMoePhases {
        router: Vec::new(),
        gate_up: vec![Step::IndexedMoeGemv {
            experts: expert_ref(),
            which: MoeProj::GateUp {
                up_out: null_f32(96),
            },
            topk_indices: null_i64(2),
            input: GemvInput::Raw(null_f32(64)),
            out: null_f32(96),
            k_top: 2,
            batch_size: 1,
        }],
        activation: vec![Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 7.0 },
            gate: null_f32(96),
            up: null_f32(96),
            rot_out: null_f32(96),
            inter: 48,
            k_top: 2,
        }],
        down: vec![down],
        combine: Vec::new(),
        finish: vec![Step::ConvertI64ToF32 {
            src: partial_i64,
            dst: null_f32(64),
            n: 64,
        }],
    }
}

fn tp_policy(ranks: usize) -> hipfire_runtime::moe_plan::MoEExecutionPolicy {
    use hipfire_runtime::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
    MoEExecutionPolicy::new(
        MoEExecutionKind::Tp,
        DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
    )
    .unwrap()
}

fn hash_plan() -> hipfire_dispatch::families::moe::RouterPlan<'static> {
    use hipfire_dispatch::families::moe::RouterPlan;
    RouterPlan::Hash {
        scores: null_f32(8),
        tokens: null_f32(1),
        tid2eid: null_f32(8),
        topk_indices: null_i64(2),
        topk_weights: null_f32(2),
        k_top: 2,
        normalize: true,
        route_scale: 2.2,
    }
}

fn bias_plan() -> hipfire_dispatch::families::moe::RouterPlan<'static> {
    use hipfire_dispatch::families::moe::RouterPlan;
    RouterPlan::BiasAwareTopK {
        scores: null_f32(8),
        gate_bias: null_f32(8),
        topk_indices: null_i64(2),
        topk_weights: null_f32(2),
        k_top: 2,
        normalize: true,
        route_scale: 2.2,
    }
}

fn precomputed_plan() -> hipfire_dispatch::families::moe::RouterPlan<'static> {
    use hipfire_dispatch::families::moe::RouterPlan;
    RouterPlan::Precomputed {
        topk_indices: null_i64(2),
        topk_weights: null_f32(2),
        k_top: 2,
        normalize: true,
        route_scale: 2.2,
    }
}

/// Lower the DS4 int64 program with null buffers. The lowerer validates
/// identity BEFORE protocol/capacity, so a matching declaration errors only
/// at capacity (never at identity), while a mismatched declaration errors at
/// `RouterIdentityMismatch`.
fn lower_err(
    group: &hipfire_runtime::weight_manifest::ExpertGroupPlan,
    policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
    router: hipfire_dispatch::families::moe::RouterPlan<'_>,
) -> hipfire_runtime::moe_plan::MoeLowerError {
    use hipfire_dispatch::families::moe::ExpertExecutionPlan;
    use hipfire_runtime::moe_plan::{lower_moe_steps, MoeProgramParts};
    lower_moe_steps(
        group,
        policy,
        MoeProgramParts {
            router,
            execution: ExpertExecutionPlan::IndexedQuantized,
            deferred_combine: false,

            ranks: vec![i64_rank(), i64_rank()],
        },
    )
    .expect_err("null buffers always fail capacity validation, not identity")
}

// ── MTP layer ─────────────────────────────────────────────────────────

#[test]
fn deepseek4_has_separate_mtp_layer() {
    // DeepSeek V4 has a separate MTP (Multi-Token Prediction) head loaded
    // from an addon HFQ file at layer index == num_hidden_layers. Its MoE is
    // dispatched by the SAME sealed indexed path as the main layers: the
    // MQ2-Lloyd int64 program (DownResidualI64 → ConvertI64ToF32) with the
    // same typed router identity contract — a separate layer identity, not
    // separate dispatch machinery.
    use hipfire_dispatch::families::moe::MoeResolution;
    use hipfire_runtime::weight_manifest::ExpertParallelism;

    // The MTP layer's routed experts resolve exactly like the main layers'
    // (uniform MQ2-Lloyd → indexed, GPU top-K).
    let dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: DType::MQ4G256,
        shared_gate: DType::MQ4G256,
        shared_expert_gate: DType::MQ4G256,
        shared_expert_up: DType::MQ4G256,
        shared_expert_down: DType::MQ4G256,
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ2G256Lloyd,
        routed_down: DType::MQ2G256Lloyd,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        gate_side_has_awq: false,
        routed_down_has_awq: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let res = MoeResolution::resolve(&dtypes, 8);
    assert!(res.use_gpu_topk, "MQ2-Lloyd routed experts use GPU top-K");
    assert!(
        res.routed_indexable(),
        "MQ2-Lloyd routed experts are indexable"
    );

    // The MTP layer carries its own layer identity (index == num_hidden_layers)
    // and its declared identity must match its typed router plan: a matching
    // declaration passes the identity gate (failing only at null-buffer
    // capacity), a mismatched declaration is rejected before any schedule.
    let mtp_group = group(
        Some(4),
        "bias_aware_topk",
        ExpertParallelism::TensorParallel,
        2,
    );
    let err = lower_err(&mtp_group, &tp_policy(2), bias_plan());
    assert!(
        !matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "matching MTP declaration passes the identity gate, got: {err}"
    );
    let mismatched = group(Some(4), "hash", ExpertParallelism::TensorParallel, 2);
    let err = lower_err(&mismatched, &tp_policy(2), bias_plan());
    assert!(
        matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "mismatched MTP declaration rejected at identity, got: {err}"
    );
}

// ── hash vs score routing ─────────────────────────────────────────────

#[test]
fn deepseek4_uses_hash_and_score_routed_moe() {
    // DeepSeek V4 MoE:
    // - Layers 0..num_hash_layers: hash-routed (tid2eid lookup table)
    // - Layers num_hash_layers..: score-routed (bias-aware top-K)
    // The typed router plans carry distinct selections, and the lowerer's
    // declared-identity contract rejects a "hash" declaration aliasing a
    // host-completed Precomputed plan (and vice versa).
    use hipfire_dispatch::families::moe::RouterSelection;
    use hipfire_runtime::weight_manifest::ExpertParallelism;

    assert_eq!(hash_plan().selection(), RouterSelection::Hash);
    assert_eq!(bias_plan().selection(), RouterSelection::BiasAwareTopK);
    assert_eq!(precomputed_plan().selection(), RouterSelection::Precomputed);

    // Positive identity matches pass the gate (failing only at null-buffer
    // capacity, never at identity).
    let group_hash = group(Some(1), "hash", ExpertParallelism::TensorParallel, 2);
    let err = lower_err(&group_hash, &tp_policy(2), hash_plan());
    assert!(
        !matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "declared hash + Hash plan passes the identity gate, got: {err}"
    );
    let group_precomputed = group(Some(1), "precomputed", ExpertParallelism::TensorParallel, 2);
    let err = lower_err(&group_precomputed, &tp_policy(2), precomputed_plan());
    assert!(
        !matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "declared precomputed + Precomputed plan passes the identity gate, got: {err}"
    );

    // No Hash alias in either direction: a host-completed Precomputed plan
    // must not lower under a "hash" declaration, and a device Hash plan must
    // not lower under a "precomputed" declaration.
    let err = lower_err(&group_hash, &tp_policy(2), precomputed_plan());
    assert!(
        matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "Precomputed plan under a hash declaration must be rejected, got: {err}"
    );
    let err = lower_err(&group_precomputed, &tp_policy(2), hash_plan());
    assert!(
        matches!(
            err,
            hipfire_runtime::moe_plan::MoeLowerError::RouterIdentityMismatch { .. }
        ),
        "Hash plan under a precomputed declaration must be rejected, got: {err}"
    );
}

// ── attention: two paths on one KV surface ────────────────────────────

#[test]
fn deepseek4_attention_dispatch_two_paths() {
    // Per-layer attention dispatch is the arch-side `compress_ratio == 0`
    // branch (SWA-only vs SWA + compressed-KV indexer); both paths consume
    // the same dispatch surface — the Q8_0 KV-cache quant resolved through
    // the runtime KV-mode ladder and batched on DS4's archs. The branch
    // itself is a per-layer config predicate in the arch crate (not a
    // dispatch-key difference), so this test pins the shared surface both
    // paths dispatch through.
    use hipfire_runtime::kv_mode::{resolve, KvMode, HFQ_Q8_ONLY_POLICY};
    use hipfire_runtime::llama::is_batchable_la;

    // DS4's KV mode resolves to Q8 without warning on the HFQ ladder
    // (head_dim = 512).
    let res = resolve("q8", &HFQ_Q8_ONLY_POLICY, 512);
    assert_eq!(res.mode, KvMode::Q8);
    assert!(res.warning.is_none());
    // An unrecognized string falls back to the same Q8 default.
    assert_eq!(resolve("asym3", &HFQ_Q8_ONLY_POLICY, 512).mode, KvMode::Q8);

    // Both paths' batched attention projections (Q8 KV gather + SWA ring)
    // are batchable on DS4's archs.
    for &arch in &["gfx1100", "gfx942"] {
        assert!(
            is_batchable_la(DType::Q8_0, arch),
            "Q8_0 attention batchable on {arch}"
        );
    }
}

// ── weight dtype → kernel family ──────────────────────────────────────

#[test]
fn deepseek4_weight_dtype_dispatch() {
    // Weight dtype → kernel family (DS4 non-expert + routed-expert surface):
    // - F16: gemv_f16_x_f16_wmma (WMMA GEMM family)
    // - Q8_0: gemv_q8_0
    // - Raw/MQ4: MQ4 prerotated path
    // - MQ2-Lloyd routed experts: MQ2-Lloyd prerotated path
    // - FP4 experts: FP4 codec + GEMV (prerotated Lloyd E8 family)
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::GemvFamily;
    use hipfire_dispatch::types::KernelKey;

    let family = GemvFamily::new();
    let ctx = DispatchCtx::for_test("gfx1100");
    for key in [
        KernelKey::GemvF16,
        KernelKey::GemvQ8_0,
        KernelKey::GemvMq4G256Prerotated,
        KernelKey::GemvMq2G256LloydPrerotated,
        KernelKey::GemvMfp4G32LloydPrerotated,
        KernelKey::GemvMfp4G32E8Prerotated,
    ] {
        assert!(
            family.registry().resolve(key, &ctx, None).is_ok(),
            "{key:?} resolves on gfx1100"
        );
    }
}
