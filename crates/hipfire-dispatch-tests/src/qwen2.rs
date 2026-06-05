//! Qwen2 model family dispatch tests.
//!
//! arch_id=7. Simplest bring-up: F32-only KV cache, no MQ rotation path,
//! no fused kernels for bias. GQA-aware flash attention.

use rdna_compute::DType;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::fused_qkv::FusedQkvFamily;
use hipfire_dispatch::types::KernelKey;

#[test]
fn qwen2_prefill_batchable_formats() {
    use hipfire_runtime::llama::is_batchable_la;
    // Qwen2 uses standard quant formats.
    for &arch in &["gfx1100", "gfx1030", "gfx906"] {
        assert!(is_batchable_la(DType::MQ4G256, arch), "MQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::HFQ4G256, arch), "HFQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::Q8_0, arch), "Q8_0 batchable on {arch}");
    }
}

#[test]
fn qwen2_no_qk_norm_dispatch() {
    // Qwen2 does NOT have QK norm (unlike Qwen3).
    // QK norm dispatch: hipfire-runtime checks has_qk_norm config flag.
}

#[test]
fn qwen2_attention_bias_dispatch() {
    // Qwen2 has attention_bias=true on Q/K/V projections.
    // This adds bias_add_f32 calls after each GEMV.
    // The bias add is a separate kernel launch, not fused.
}

// ─── FusedGateUpQ8_0 coverage (Ship 2.1 A1) ─────────────────────

#[test]
fn qwen2_fused_gate_up_q8_0_resolves_on_all_arches() {
    let family = FusedQkvFamily::new();
    for &arch in &["gfx1100", "gfx1030", "gfx906", "gfx1201"] {
        let ctx = DispatchCtx::for_test(arch);
        assert!(
            family.resolve(KernelKey::FusedGateUpQ8_0, &ctx, None).is_ok(),
            "FusedGateUpQ8_0 should resolve on {arch} (Always gate)"
        );
    }
}

#[test]
fn qwen2_hfq4_qkv_resolves_on_all_arches() {
    // FusedQkvHfq4G256 predicate was widened to Always (A0).
    let family = FusedQkvFamily::new();
    for &arch in &["gfx1100", "gfx1030", "gfx906", "gfx1201"] {
        let ctx = DispatchCtx::for_test(arch);
        assert!(
            family.resolve(KernelKey::FusedQkvHfq4G256, &ctx, None).is_ok(),
            "FusedQkvHfq4G256 should resolve on {arch} (Always gate)"
        );
    }
}
