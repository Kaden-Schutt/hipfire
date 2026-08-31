//! LLaMA / Mistral / Qwen3 model family dispatch tests.
//!
//! arch_id=0/1. Standard dense transformer with GGUF Q4K heritage format
//! support and basic MQ rotation path. No hybrid LA/FA or MoE complexity.

use rdna_compute::DType;

// ─── KV-tier dispatch sweep (Site A fix validation) ────────────
//
// These tests validate that each KvCache tier produces the correct
// KvTierPlan via KvTierPlan::derive — mirroring what arch.rs Site A
// (and the runtime forward) now do via kv_cache.tier_inputs().
// GPU-free: tests pure dispatch-resolution logic only.

fn tier_inputs_base() -> hipfire_dispatch::families::kv_tier::KvTierInputs {
    use hipfire_dispatch::families::kv_tier::{F32AttnPolicy, KvTierInputs};
    KvTierInputs {
        quant_asym4: false,
        quant_asym3: false,
        quant_asym2: false,
        quant_q8: false,
        quant_fwht: false,
        quant_hfq4: false,
        quant_q4: false,
        quant_int8: false,
        quant_hfq8: false,
        quant_bf16: false,
        f32_policy: F32AttnPolicy::Simple,
        v_mode_bits: 8,
        pos: 0,
        flash_mode: 0,
        capture_mode: false,
        batch_size: 1,
        is_tree: false,
        is_boundary: false,
        q8_windowed: false,
        window: 0,
    }
}

#[test]
fn llama_kv_tier_q8_uses_q8_kernels_no_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_q8()).unwrap();
    // Q8 never needs givens rotation buffers.
    assert!(!plan.uses_givens, "Q8 must not set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
    // Short-context Q8 defaults to non-flash single-token attend.
    assert_eq!(plan.attend_key, KernelKey::AttnQ8_0Kv);
}

#[test]
fn llama_kv_tier_asym4_uses_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_asym4()).unwrap();
    assert!(plan.uses_givens, "Asym4 must set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteAsym4);
    assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4);
}

#[test]
fn llama_kv_tier_asym3_uses_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_asym3()).unwrap();
    assert!(plan.uses_givens, "Asym3 must set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteAsym3);
    assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym3);
}

#[test]
fn llama_kv_tier_asym2_uses_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_asym2()).unwrap();
    assert!(plan.uses_givens, "Asym2 must set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteAsym2);
    assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym2);
}

#[test]
fn llama_kv_tier_hfq8_no_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_hfq8()).unwrap();
    assert!(!plan.uses_givens, "HFQ8 must not set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteHfq8);
    assert_eq!(plan.attend_key, KernelKey::AttnHfq8Kv);
}

#[test]
fn llama_kv_tier_q4_no_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_q4()).unwrap();
    assert!(!plan.uses_givens, "Q4 must not set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteQ4);
    assert_eq!(plan.attend_key, KernelKey::AttnQ4Kv);
}

#[test]
fn llama_kv_tier_hfq4_no_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base().with_hfq4()).unwrap();
    assert!(!plan.uses_givens, "HFQ4 must not set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteHfq4);
    assert_eq!(plan.attend_key, KernelKey::AttnHfq4Kv);
}

#[test]
fn llama_kv_tier_f32_no_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let plan = KvTierPlan::derive(tier_inputs_base()).unwrap();
    assert!(!plan.uses_givens, "F32 must not set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteF32);
    assert_eq!(plan.attend_key, KernelKey::AttnF32);
}

#[test]
fn llama_kv_tier_asym4_fwht_uses_givens() {
    use hipfire_dispatch::families::kv_tier::KvTierPlan;
    use hipfire_dispatch::types::KernelKey;
    let mut ti = tier_inputs_base().with_asym4();
    ti.quant_fwht = true;
    let plan = KvTierPlan::derive(ti).unwrap();
    assert!(plan.uses_givens, "Asym4+FWHT must set uses_givens");
    assert_eq!(plan.write_key, KernelKey::KvWriteAsym4Fwht);
    assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4Fwht);
}

// ─── Builder helpers (keep at module end) ─────────────────────

trait KvTierInputsExt: Sized {
    fn with_q8(self) -> Self;
    fn with_asym4(self) -> Self;
    fn with_asym3(self) -> Self;
    fn with_asym2(self) -> Self;
    fn with_hfq8(self) -> Self;
    fn with_q4(self) -> Self;
    fn with_hfq4(self) -> Self;
}

impl KvTierInputsExt for hipfire_dispatch::families::kv_tier::KvTierInputs {
    fn with_q8(mut self) -> Self {
        self.quant_q8 = true;
        self
    }
    fn with_asym4(mut self) -> Self {
        self.quant_asym4 = true;
        self
    }
    fn with_asym3(mut self) -> Self {
        self.quant_asym3 = true;
        self
    }
    fn with_asym2(mut self) -> Self {
        self.quant_asym2 = true;
        self
    }
    fn with_hfq8(mut self) -> Self {
        self.quant_hfq8 = true;
        self
    }
    fn with_q4(mut self) -> Self {
        self.quant_q4 = true;
        self
    }
    fn with_hfq4(mut self) -> Self {
        self.quant_hfq4 = true;
        self
    }
}

// ─── Prefill batchability ─────────────────────────────────────

#[test]
fn llama_prefill_always_batchable() {
    use hipfire_runtime::llama::is_batchable_la;
    let batchable_archs = &[
        "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx942",
    ];
    for &arch in batchable_archs {
        assert!(
            is_batchable_la(DType::MQ4G256, arch),
            "MQ4G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::HFQ4G256, arch),
            "HFQ4G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::MQ6G256, arch),
            "MQ6G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::HFQ6G256, arch),
            "HFQ6G256 batchable on {arch}"
        );
        assert!(
            is_batchable_la(DType::Q8_0, arch),
            "Q8_0 batchable on {arch}"
        );
    }
}

#[test]
fn llama_prefill_mq3_on_wmma_or_gfx10_scalar() {
    use hipfire_runtime::llama::is_batchable_la;
    for &arch in &[
        "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
    ] {
        assert!(
            is_batchable_la(DType::MQ3G256, arch),
            "MQ3G256 batch on {arch} (WMMA)"
        );
    }
    for &arch in &["gfx1010", "gfx1030"] {
        assert!(
            is_batchable_la(DType::MQ3G256, arch),
            "MQ3G256 batch on {arch} (scalar)"
        );
    }
    for &arch in &["gfx906", "gfx942"] {
        assert!(
            !is_batchable_la(DType::MQ3G256, arch),
            "MQ3G256 fallback on {arch}"
        );
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
    assert!(
        !runtime_is_batchable(DType::ParoQ4G128, "gfx1100"),
        "runtime copy should NOT admit ParoQ4G128"
    );
    assert!(
        !runtime_is_batchable(DType::F32, "gfx1100"),
        "runtime copy should NOT admit F32"
    );
    assert!(
        !runtime_is_batchable(DType::MQ3G256Lloyd, "gfx1100"),
        "runtime copy should NOT admit MQ3G256Lloyd"
    );
    assert!(
        !runtime_is_batchable(DType::MQ4G256Lloyd, "gfx1100"),
        "runtime copy should NOT admit MQ4G256Lloyd"
    );
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
            family
                .resolve(KernelKey::FusedGateUpQ4K, &ctx, None)
                .is_ok(),
            "FusedGateUpQ4K should resolve on {arch} (Always gate)"
        );
    }
}
