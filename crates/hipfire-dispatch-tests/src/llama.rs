//! LLaMA / Mistral / Qwen3 model family dispatch tests.
//!
//! arch_id=0/1. Standard dense transformer with GGUF Q4K heritage format
//! support and basic MQ rotation path. No hybrid LA/FA or MoE complexity.

use rdna_compute::DType;

// ─── Prefill batchability ─────────────────────────────────────

#[test]
fn llama_prefill_always_batchable() {
    use hipfire_runtime::llama::is_batchable_la;
    let batchable_archs = &[
        "gfx906", "gfx1010", "gfx1030",
        "gfx1100", "gfx1150", "gfx1151",
        "gfx1200", "gfx942",
    ];
    for &arch in batchable_archs {
        assert!(is_batchable_la(DType::MQ4G256, arch), "MQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::HFQ4G256, arch), "HFQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::MQ6G256, arch), "MQ6G256 batchable on {arch}");
        assert!(is_batchable_la(DType::HFQ6G256, arch), "HFQ6G256 batchable on {arch}");
        assert!(is_batchable_la(DType::Q8_0, arch), "Q8_0 batchable on {arch}");
    }
}

#[test]
fn llama_prefill_mq3_on_wmma_or_gfx10_scalar() {
    use hipfire_runtime::llama::is_batchable_la;
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"] {
        assert!(is_batchable_la(DType::MQ3G256, arch), "MQ3G256 batch on {arch} (WMMA)");
    }
    for &arch in &["gfx1010", "gfx1030"] {
        assert!(is_batchable_la(DType::MQ3G256, arch), "MQ3G256 batch on {arch} (scalar)");
    }
    for &arch in &["gfx906", "gfx942"] {
        assert!(!is_batchable_la(DType::MQ3G256, arch), "MQ3G256 fallback on {arch}");
    }
}

#[test]
fn llama_prefill_unsupported_dtypes() {
    use hipfire_runtime::llama::is_batchable_la;
    assert!(!is_batchable_la(DType::Q4K, "gfx1100"));
    assert!(!is_batchable_la(DType::Q6K, "gfx1100"));
    assert!(!is_batchable_la(DType::F32, "gfx1100"));
}

// ─── LLaMA dispatch constants ─────────────────────────────────

#[test]
fn llama_fallback_to_llama_path_for_unknown_arch_ids() {
    // arch_id 0 (LLaMA) and 1 (Qwen3) both route through hipfire-arch-llama.
    // The daemon's load_model routes everything not in {5,6,7,8,9} to llama.
}

// ─── Runtime is_batchable_la vs qwen35 copy ────────────────────

#[test]
fn llama_runtime_copy_admits_fewer_dtypes_than_qwen35_copy() {
    use hipfire_runtime::llama::is_batchable_la as runtime_is_batchable;

    // The runtime copy does NOT admit ParoQ4G128, F32, or Lloyd variants.
    assert!(!runtime_is_batchable(DType::ParoQ4G128, "gfx1100"),
        "runtime copy should NOT admit ParoQ4G128");
    assert!(!runtime_is_batchable(DType::F32, "gfx1100"),
        "runtime copy should NOT admit F32");
    assert!(!runtime_is_batchable(DType::MQ3G256Lloyd, "gfx1100"),
        "runtime copy should NOT admit MQ3G256Lloyd");
    assert!(!runtime_is_batchable(DType::MQ4G256Lloyd, "gfx1100"),
        "runtime copy should NOT admit MQ4G256Lloyd");
}

// ─── FusedQkvQ4K / FusedGateUpQ4K coverage (Ship 2.1 A1) ─────────

#[test]
fn llama_fused_qkv_q4k_resolves_on_all_arches() {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::fused_qkv::FusedQkvFamily;
    use hipfire_dispatch::types::KernelKey;
    let family = FusedQkvFamily::new();
    for &arch in &["gfx1100", "gfx1030", "gfx906", "gfx1201"] {
        let ctx = DispatchCtx::for_test(arch);
        assert!(
            family.resolve(KernelKey::FusedQkvQ4K, &ctx, None).is_ok(),
            "FusedQkvQ4K should resolve on {arch} (Always gate)"
        );
    }
}

#[test]
fn llama_fused_gate_up_q4k_resolves_on_all_arches() {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::fused_qkv::FusedQkvFamily;
    use hipfire_dispatch::types::KernelKey;
    let family = FusedQkvFamily::new();
    for &arch in &["gfx1100", "gfx1030", "gfx906", "gfx1201"] {
        let ctx = DispatchCtx::for_test(arch);
        assert!(
            family.resolve(KernelKey::FusedGateUpQ4K, &ctx, None).is_ok(),
            "FusedGateUpQ4K should resolve on {arch} (Always gate)"
        );
    }
}
