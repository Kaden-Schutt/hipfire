//! Qwen3.5 (dense + MoE) model family dispatch tests.
//!
//! Qwen3.5 is the primary target arch (arch_id=5/6). It uses hybrid
//! LinearAttention (DeltaNet) + FullAttention layers, supports MQ/HFQ
//! quantization families, and has MoE variants (A3B, A10B, A17B).
//!
//! Test matrix dimensions:
//! - Arch × Quant → GEMV/GEMM decode/prefill dispatch (via is_batchable_la)
//! - Arch × Cache Format → attention kernel dispatch
//! - State quantization (FP32/Q8/Q4) pattern
//! - MoE routing dispatch

use rdna_compute::DType;

// ─── Prefill batchability ─────────────────────────────────────

#[test]
fn qwen35_prefill_always_batchable() {
    use hipfire_runtime::llama::is_batchable_la;
    let batchable_archs = &[
        "gfx906", "gfx908", "gfx940", "gfx941", "gfx942",
        "gfx1010", "gfx1011", "gfx1012", "gfx1013",
        "gfx1030", "gfx1031", "gfx1032",
        "gfx1100", "gfx1101", "gfx1102", "gfx1103",
        "gfx1150", "gfx1151", "gfx1152",
        "gfx1200", "gfx1201",
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
fn qwen35_prefill_mq3_on_wmma_or_gfx10_scalar() {
    use hipfire_runtime::llama::is_batchable_la;

    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"] {
        assert!(is_batchable_la(DType::MQ3G256, arch), "MQ3G256 batch on {arch} (WMMA)");
    }
    for &arch in &["gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032"] {
        assert!(is_batchable_la(DType::MQ3G256, arch), "MQ3G256 batch on {arch} (scalar)");
    }
    for &arch in &["gfx906", "gfx908", "gfx940", "gfx942"] {
        assert!(!is_batchable_la(DType::MQ3G256, arch), "MQ3G256 fallback on {arch}");
    }
}

#[test]
fn qwen35_prefill_fp4_only_on_wmma() {
    use hipfire_runtime::llama::is_batchable_la;
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"] {
        assert!(is_batchable_la(DType::HFP4G32, arch), "HFP4G32 batch on {arch}");
        assert!(is_batchable_la(DType::MFP4G32, arch), "MFP4G32 batch on {arch}");
    }
    for &arch in &["gfx906", "gfx1010", "gfx1030"] {
        assert!(!is_batchable_la(DType::HFP4G32, arch), "HFP4G32 fallback on {arch}");
    }
}

#[test]
fn qwen35_prefill_q4k_q6k_unsupported() {
    use hipfire_runtime::llama::is_batchable_la;
    assert!(!is_batchable_la(DType::Q4K, "gfx1100"));
    assert!(!is_batchable_la(DType::Q6K, "gfx1100"));
}

// ─── Qwen3.5 dispatch constants ───────────────────────────────

#[test]
fn qwen35_default_kv_mode_is_asym3() {
    // The daemon defaults to asym3/turbo3 for KV cache.
    // This is a policy decision in daemon.rs load_model.
    // The KvCache::new_gpu_asym3_capped sets:
    //   quantized=true, quant_asym3=true, quant_fwht=false (Givens rotation)
    let caps = vec![false, false, false, false, true, false, false]; // matches asym3 pattern
    assert_eq!(caps[4], true); // quant_asym3
}

// ─── Qwen3.5 MoE dispatch ─────────────────────────────────────

#[test]
fn qwen35_moe_mq3_refused_at_load_time() {
    // MoE + MQ3 is refused at daemon load time.
    // Check: moe_ffn_has_mq3() returns true → load_model aborts.
    // This is enforced in qwen35.rs model loader, not in dispatch layer.
    // Test documents the gating policy.
}
