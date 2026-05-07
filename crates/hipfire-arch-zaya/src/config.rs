//! ZayaConfig: model-shape constants for ZAYA1, parsed from HFQ metadata.
//!
//! Field set mirrors `Zyphra/transformers@zaya1/configuration_zaya.py:23-122`
//! (`class ZayaConfig(PretrainedConfig)`) and the published
//! `Zyphra/ZAYA1-8B/config.json`. Defaults match the 8B config; arch_id 7
//! is reserved for the Zaya family in `crates/hipfire-runtime/src/arch.rs`
//! comment ladder.
//!
//! Port-status notes per field are inline. Anything tagged FREE maps onto
//! existing hipfire infrastructure with at most a kernel-parameter tweak.
//! Anything tagged NEW requires either a new kernel or a runtime-level
//! design decision (CCA recurrent state, MoD per-token routing, EDA).

use hipfire_runtime::hfq::HfqFile;
use serde::{Deserialize, Serialize};

/// Model-shape constants for a ZAYA1 family checkpoint.
///
/// All fields are u32/usize/f32/bool to keep the struct cheap to clone
/// across threads (the trait bound is `Config: Clone + Send + 'static`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZayaConfig {
    // -- Standard transformer shape -----------------------------------------
    /// Token vocab size. ZAYA1-8B = 262272 (Gemma-style 256k base + extras).
    /// Tokenizer support: FREE if Gemma4 tokenizer port is in tree.
    pub vocab_size: usize,
    /// Hidden / model dim. ZAYA1-8B = 2048.
    pub hidden_size: usize,
    /// FFN hidden dim (per-expert at `gated_linear_unit=true`). ZAYA1-8B = 4096.
    pub ffn_hidden_size: usize,
    /// Decoder block count. ZAYA1-8B = 80.
    pub num_hidden_layers: usize,
    /// Standard attention Q heads (downstream of CCA). ZAYA1-8B = 16.
    pub num_attention_heads: usize,
    /// Standard attention KV heads (GQA). ZAYA1-8B = 2 (8:1 GQA ratio).
    pub num_key_value_heads: usize,
    /// Per-head dim. ZAYA1-8B = 128 (= hidden_size / num_attention_heads).
    pub head_dim: usize,
    /// Max position. ZAYA1-8B = 131072 (128k context).
    pub max_position_embeddings: usize,

    // -- Norm / activation --------------------------------------------------
    /// FREE: RMSNorm epsilon. ZAYA1-8B = 1e-5.
    pub norm_epsilon: f32,
    /// FREE: SwiGLU (gated_linear_unit=true).
    pub activation_func: ActivationFunc,

    // -- RoPE ---------------------------------------------------------------
    /// FREE-ish: partial RoPE; rotates first `head_dim * partial_rotary_factor`
    /// dims, leaves the rest untouched. ZAYA1-8B = 0.5 (rotates 64 of 128).
    /// Requires either parameterizing existing RoPE kernel or adding a new
    /// entry point. See Phase 2 plan.
    pub partial_rotary_factor: f32,
    /// FREE: RoPE base frequency. ZAYA1-8B = 5_000_000 (long-context base).
    pub rope_theta: f32,

    // -- MoE ----------------------------------------------------------------
    /// NEW-ish: 16 experts. Existing qwen35 MoE handles 128/256/etc.
    pub num_experts: usize,
    /// FREE: top-1 (Switch-style). Simpler than qwen35's top-8; degenerate
    /// case of existing top-k.
    pub moe_router_topk: usize,
    /// NEW: MLP-based router (vs qwen35's linear). Small kernel addition.
    /// `zaya_mlp_expansion=256` sets router-MLP hidden dim. (TODO: confirm
    /// semantics from modeling_zaya.py:917 ZayaRouter; Phase 2.)
    pub zaya_mlp_expansion: usize,

    // -- CCA (RECURRENT, see 00-cca-disambiguation.md) ---------------------
    /// CCA enabled. ZAYA1-8B = true.
    pub cca: bool,
    /// CCA's own Q-head count, distinct from `num_attention_heads`.
    /// ZAYA1-8B = 8 (vs 16 standard attn). The CCA→Attention plumbing
    /// requires a head-rebalance step; details pending Phase 1 read of
    /// ZayaAttention (modeling_zaya.py:483).
    pub cca_num_q_heads: usize,
    /// CCA's KV head count (matches `num_query_groups`). ZAYA1-8B = 2.
    pub num_query_groups: usize,
    /// CCA depthwise-conv kernel size (first conv). ZAYA1-8B = 2.
    pub cca_time0: usize,
    /// CCA grouped-conv kernel size (second conv). ZAYA1-8B = 2.
    pub cca_time1: usize,

    // -- MoD (per-token layer skip; Phase 4 design doc) ---------------------
    /// NEW: enables per-token layer-skip routing. ZAYA1-8B = true.
    pub zaya_use_mod: bool,

    // -- EDA (identification pending Phase 5) -------------------------------
    /// NEW: enables an undocumented "EDA" component. ZAYA1-8B = true.
    /// Read modeling_zaya.py / modular_zaya.py to identify; see Phase 5.
    pub zaya_use_eda: bool,

    // -- Residual scaling ---------------------------------------------------
    /// NEW (trivial): per-block learnable residual scalar. ZAYA1-8B = true.
    /// Implementation: load per-block scalar tensor, multiply during
    /// residual add. See Phase 2.
    pub scale_residual_merge: bool,
    /// FREE: residual stream upcast to fp32. ZAYA1-8B config.json = true.
    /// Existing hipfire pattern parallels rmsnorm fp32 accumulator.
    pub residual_in_fp32: bool,

    // -- Misc / metadata ----------------------------------------------------
    /// FREE: tied input embed and lm_head. ZAYA1-8B inherits the default
    /// (true in `PretrainedConfig`).
    pub tie_word_embeddings: bool,
    /// Tokenizer special tokens (parsed from HFQ metadata, not config).
    /// ZAYA1-8B: pad=0, bos=2, eos=106 (Gemma `<end_of_turn>`).
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActivationFunc {
    Swiglu,
}

impl Default for ZayaConfig {
    /// Defaults match the published `Zyphra/ZAYA1-8B/config.json` so a
    /// `ZayaConfig::default()` corresponds to the 8B checkpoint. Smaller
    /// or larger ZAYA1 variants override via `config_from_hfq`.
    fn default() -> Self {
        Self {
            vocab_size: 262_272,
            hidden_size: 2048,
            ffn_hidden_size: 4096,
            num_hidden_layers: 80,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            max_position_embeddings: 131_072,
            norm_epsilon: 1e-5,
            activation_func: ActivationFunc::Swiglu,
            partial_rotary_factor: 0.5,
            rope_theta: 5_000_000.0,
            num_experts: 16,
            moe_router_topk: 1,
            zaya_mlp_expansion: 256,
            cca: true,
            cca_num_q_heads: 8,
            num_query_groups: 2,
            cca_time0: 2,
            cca_time1: 2,
            zaya_use_mod: true,
            zaya_use_eda: true,
            scale_residual_merge: true,
            residual_in_fp32: true,
            tie_word_embeddings: true,
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 106,
        }
    }
}

impl ZayaConfig {
    /// Parse a ZayaConfig out of an HFQ file's metadata blob.
    ///
    /// Phase 1 status: returns Err. No HFQ writer for ZAYA1 exists yet
    /// (the `hipfire-quantize` crate doesn't know the Zaya tensor
    /// naming convention). When the quantizer adds a Zaya path, this
    /// method walks the metadata JSON the way
    /// `hipfire_arch_qwen35::qwen35::config_from_hfq` does, branching
    /// on the arch_id stored in the HFQ header to handle ZAYA1
    /// variants.
    pub fn from_hfq(_hfq: &HfqFile) -> Result<Self, String> {
        Err(
            "ZayaConfig::from_hfq not implemented (Phase 1 scaffold). \
             ZAYA1 has no HFQ representation yet; quantizer support is \
             pending after Phase 6 recurrent-state design lands. See \
             docs/investigations/2026-05-07-zaya1-port-intake/."
                .to_string(),
        )
    }

    /// Validate that a parsed config matches the bf16 reference's shape.
    /// Used by Phase 1 / Phase 2 reference-dump validation; called by the
    /// `verify_against_torch` example before any forward attempt.
    pub fn assert_zaya1_8b_shape(&self) -> Result<(), String> {
        let want = Self::default();
        if self.vocab_size != want.vocab_size {
            return Err(format!("vocab_size {} != {}", self.vocab_size, want.vocab_size));
        }
        if self.hidden_size != want.hidden_size {
            return Err(format!("hidden_size {} != {}", self.hidden_size, want.hidden_size));
        }
        if self.num_hidden_layers != want.num_hidden_layers {
            return Err(format!(
                "num_hidden_layers {} != {}",
                self.num_hidden_layers, want.num_hidden_layers
            ));
        }
        if self.num_attention_heads != want.num_attention_heads
            || self.num_key_value_heads != want.num_key_value_heads
            || self.head_dim != want.head_dim
        {
            return Err(format!(
                "attn shape {}q/{}kv/{}d != {}q/{}kv/{}d",
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                want.num_attention_heads,
                want.num_key_value_heads,
                want.head_dim,
            ));
        }
        if self.partial_rotary_factor != want.partial_rotary_factor {
            return Err(format!(
                "partial_rotary_factor {} != {}",
                self.partial_rotary_factor, want.partial_rotary_factor
            ));
        }
        if self.num_experts != want.num_experts || self.moe_router_topk != want.moe_router_topk {
            return Err(format!(
                "MoE topology {}exp/top-{} != {}exp/top-{}",
                self.num_experts,
                self.moe_router_topk,
                want.num_experts,
                want.moe_router_topk
            ));
        }
        if !self.cca || !self.zaya_use_mod || !self.zaya_use_eda {
            return Err(format!(
                "feature flags off (cca={}, mod={}, eda={}); ZAYA1-8B has all on",
                self.cca, self.zaya_use_mod, self.zaya_use_eda
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_zaya1_8b_shape() {
        let cfg = ZayaConfig::default();
        cfg.assert_zaya1_8b_shape().expect("default must match published 8B config");
    }

    #[test]
    fn from_hfq_returns_err_in_phase1() {
        // `from_hfq` is a stub today; smoke-test that it returns Err so
        // anyone wiring it into the daemon fails loudly rather than
        // silently constructing a default config from a real HFQ file.
        // (HfqFile construction itself requires a real file, so this
        // test stays as a documentation marker rather than a runnable
        // assertion until quantizer support lands.)
    }
}
