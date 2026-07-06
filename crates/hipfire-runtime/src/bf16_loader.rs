//! GPTQ-target tensor-name predicate for the Tier-1 calibration path.
//!
//! The only live symbol here is [`is_gptq_target`], used by
//! `calibration.rs` (`HessianCollector`) and mirrored from
//! `scripts/collect_hessian.py` so the Tier-1 binary produces a
//! byte-compatible HFHS-v1 output with the Tier-2 Python path.
//!
//! History: this module formerly also held a `load_bf16_model`
//! safetensors-loader scaffold (`unimplemented!()`) plus its `Bf16Tensor`
//! / `TrunkBF16` metadata structs, sketched in the 2026-05-19 Tier-1
//! foundation series. They were never wired (the imatrix/hessian work
//! moved to its own pipeline) and were removed as dead scaffold on
//! 2026-06-15. Recover from git history if a BF16 calibration loader is
//! revived.

/// Returns true if a tensor name matches the GPTQ-target whitelist that
/// `collect_hessian` should accumulate a Hessian for. Mirrors
/// `scripts/collect_hessian.py::is_gptq_target` so the Tier 1 binary
/// produces a byte-compatible HFHS-v1 output with the Tier 2 Python
/// path.
///
/// Whitelist (suffixes matched against the last `.`-separated segment):
///
///   - Attention input projections: `q_proj`, `k_proj`, `v_proj`,
///     `qkv_proj`
///   - Attention output: `o_proj`, `out_proj`
///   - MLP: `gate_proj`, `up_proj`, `down_proj`, `gate_up_proj`
///   - Linear-attention (Gated DeltaNet):
///     `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`
///   - MoE router: `gate`
#[allow(dead_code)]
pub fn is_gptq_target(name: &str) -> bool {
    const TARGETS: &[&str] = &[
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "gate",
    ];
    // Strip a trailing `.weight` (HF safetensors stores Linear weights
    // as `<module>.weight`; the GPTQ targets are checked on the module
    // name, not the parameter name).
    let bare = name.strip_suffix(".weight").unwrap_or(name);
    let last = bare.rsplit('.').next().unwrap_or(bare);
    TARGETS.contains(&last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_target_recognizes_canonical_qwen35_names() {
        assert!(is_gptq_target("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.k_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.v_proj.weight"));
        assert!(is_gptq_target("model.layers.0.self_attn.o_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.gate_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.up_proj.weight"));
        assert!(is_gptq_target("model.layers.0.mlp.down_proj.weight"));
    }

    #[test]
    fn gptq_target_recognizes_moe_router() {
        // Qwen3.5-A3B MoE router lives at `model.layers.N.mlp.gate.weight`
        assert!(is_gptq_target("model.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn gptq_target_rejects_norms_and_embed() {
        assert!(!is_gptq_target("model.embed_tokens.weight"));
        assert!(!is_gptq_target("model.layers.0.input_layernorm.weight"));
        assert!(!is_gptq_target("model.norm.weight"));
        assert!(!is_gptq_target("lm_head.weight"));
    }

    #[test]
    fn gptq_target_recognizes_deltanet_projections() {
        assert!(is_gptq_target(
            "model.layers.0.linear_attn.in_proj_qkv.weight"
        ));
        assert!(is_gptq_target(
            "model.layers.0.linear_attn.in_proj_z.weight"
        ));
        assert!(is_gptq_target("model.layers.0.linear_attn.out_proj.weight"));
    }
}
