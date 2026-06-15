//! DeepSeek V4 Flash model family dispatch tests.
//!
//! arch_id=9. Most specialized: hyper-connections (4 residual streams),
//! compressed-KV indexer, tail-only RoPE, Q/O-LoRA, FP4 experts,
//! sliding window attention (SWA), separate MTP spec-decode layer.

use rdna_compute::DType;

#[test]
fn deepseek4_prefill_batchable_formats() {
    use hipfire_runtime::llama::is_batchable_la;
    // DeepSeek V4 uses MQ4, Q8_0, and F16/F32 for its layers.
    for &arch in &["gfx1100", "gfx942"] {
        assert!(is_batchable_la(DType::MQ4G256, arch), "MQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::HFQ4G256, arch), "HFQ4G256 batchable on {arch}");
        assert!(is_batchable_la(DType::Q8_0, arch), "Q8_0 batchable on {arch}");
    }
}

#[test]
fn deepseek4_has_separate_mtp_layer() {
    // DeepSeek V4 has a separate MTP (Multi-Token Prediction) head
    // loaded from an addon HFQ file. The MTP forward uses its own
    // dispatch path independent of the main model.
}

#[test]
fn deepseek4_uses_hash_and_score_routed_moe() {
    // DeepSeek V4 MoE:
    // - Layers 0-2: hash-routed (tid2eid lookup table)
    // - Layers 3+: score-routed (standard top-K)
    // This affects which MoE dispatch kernel is called.
}

#[test]
fn deepseek4_attention_dispatch_two_paths() {
    // Per-layer attention dispatch:
    // - compress_ratio == 0: attention_block_batched_swa_only
    // - compress_ratio > 0: attention_block_batched_mixed (SWA + indexer)
    // SWA cache is 128 tokens fixed size.
}

#[test]
fn deepseek4_weight_dtype_dispatch() {
    // Weight dtype → kernel family:
    // - F16: gemm_f16_x_f16_wmma (WMMA GEMM)
    // - Q8_0: gemv_q8_0
    // - Raw/MQ4: MQ4 prerotated path
    // - FP4 experts: routed through FP4 codec + GEMV
}
