// SPDX-License-Identifier: Apache-2.0
// hipfire — tiny random-init model fixtures for fast kernel/plumbing gating.
//
// Emits a HF-format (safetensors + config.json) random-init model in a
// hipfire-supported architecture's exact tensor layout, at "tiny" dims
// (<10M params) that still preserve the structural features gating needs
// (≥1 of each layer type, etc.). The output flows through the normal
// `--input` quantize path, so it exercises the arch-specific name-mapper too.
//
// The manifest here is the single source of truth re-used from what the
// ingest path expects; as new archs gain support, add a `tiny_*` builder.
// See TODO.md "Tiny random-init fixtures + golden-output tripwire".

// The fixture manifest vocabulary (TensorSpec/Init/Dt) lives in hipfire-arch-api so
// each family's `-spec` crate can DECLARE its ToyModel fixture with only that dep;
// this crate keeps the writer (seeded RNG → safetensors + shared tokenizer).
use hipfire_arch_api::{Dt, Init, TensorSpec};
use hipfire_primitives::conv::f32_to_bf16_bits as bf16_bits;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// Deterministic splitmix64 → reproducible fixtures across machines.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [-1, 1).
    fn next_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

/// Tiny DFlash draft sidecar fixture. This is not a quality model; it is a
/// runtime/training artifact shape that can flow through `dflash_convert`.
struct DflashTiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    target_layer_ids: Vec<usize>,
    num_target_layers: usize,
}

impl DflashTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            block_size: 16,
            target_layer_ids: vec![0, 1],
            num_target_layers: 4,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "dflash",
            "num_hidden_layers": self.layers,
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "block_size": self.block_size,
            "num_target_layers": self.num_target_layers,
            "dflash_config": {
                "mask_token_id": self.vocab - 1,
                "target_layer_ids": self.target_layer_ids,
            },
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let num_extract = self.target_layer_ids.len();
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "fc.weight",
            vec![h, num_extract * h],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            "hidden_norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16("norm.weight", vec![h], Init::NormOnes));
        for i in 0..self.layers {
            let p = format!("layers.{i}");
            let sa = format!("{p}.self_attn");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.03),
            ));
        }
        t
    }
}

/// Tiny Qwen3.5 (arch 5) dense text config. Mirrors the real text_config
/// fields the ingest/arch-detect path reads, at fixture dims.
struct Qwen35Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    full_attn_interval: usize,
    // linear-attn (DeltaNet)
    l_key_heads: usize,
    l_key_head_dim: usize,
    l_val_heads: usize,
    l_val_head_dim: usize,
    conv_kernel: usize,
    // MoE (arch 6). `experts == 0` ⇒ dense (arch 5).
    experts: usize,
    experts_per_tok: usize,
    moe_inter: usize,
    shared_inter: usize,
}

impl Qwen35Tiny {
    /// ~3.9M params: 4 layers (3 linear-attn + 1 full-attn), tiny vocab.
    /// head_dim is pinned to 128 — the gated DeltaNet kernels are specialized
    /// for HD=128 (and full-attn supports it), so smaller HDs hard-error.
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            full_attn_interval: 4,
            l_key_heads: 2,
            l_key_head_dim: 128,
            l_val_heads: 2,
            l_val_head_dim: 128,
            conv_kernel: 4,
            experts: 0,
            experts_per_tok: 0,
            moe_inter: 0,
            shared_inter: 0,
        }
    }

    /// ~6M params: arch-6 MoE. Same hybrid attention as the dense preset, but
    /// every layer's FFN is MoE (8 experts top-2 + an always-on shared expert),
    /// matching the A3B layout (all layers MoE; attention type still varies).
    fn moe_preset() -> Self {
        Self {
            experts: 8,
            experts_per_tok: 2,
            moe_inter: 128,
            shared_inter: 128,
            ..Self::preset()
        }
    }

    fn is_moe(&self) -> bool {
        self.experts > 0
    }

    /// `full_attention` every `full_attn_interval`-th layer (positions
    /// interval-1, 2*interval-1, …), else `linear_attention` — matches the
    /// real checkpoint's layer_types pattern.
    fn layer_types(&self) -> Vec<&'static str> {
        (0..self.layers)
            .map(|i| {
                if (i + 1) % self.full_attn_interval == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    }

    fn config_json(&self) -> serde_json::Value {
        let mut c = serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_output_gate": true,
            "full_attention_interval": self.full_attn_interval,
            "layer_types": self.layer_types(),
            "linear_num_key_heads": self.l_key_heads,
            "linear_key_head_dim": self.l_key_head_dim,
            "linear_num_value_heads": self.l_val_heads,
            "linear_value_head_dim": self.l_val_head_dim,
            "linear_conv_kernel_dim": self.conv_kernel,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        });
        if self.is_moe() {
            let o = c.as_object_mut().unwrap();
            o.insert("model_type".into(), "qwen3_5_moe_text".into());
            o.insert("num_experts".into(), self.experts.into());
            o.insert("num_experts_per_tok".into(), self.experts_per_tok.into());
            o.insert("moe_intermediate_size".into(), self.moe_inter.into());
            o.insert(
                "shared_expert_intermediate_size".into(),
                self.shared_inter.into(),
            );
            o.insert("norm_topk_prob".into(), true.into());
            o.insert("decoder_sparse_step".into(), 1.into());
            o.insert("mlp_only_layers".into(), serde_json::json!([]));
        }
        c
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let mut t = Vec::new();
        // Globals (tie_word_embeddings ⇒ no separate lm_head).
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::new(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));

        let qkv =
            self.l_key_heads * self.l_key_head_dim * 2 + self.l_val_heads * self.l_val_head_dim;
        let v_dim = self.l_val_heads * self.l_val_head_dim;
        let attn_q = self.n_heads * self.head_dim * 2; // attn_output_gate ⇒ 2× wide
        let kv_dim = self.n_kv_heads * self.head_dim;
        let o_in = self.n_heads * self.head_dim;

        for (i, kind) in self.layer_types().into_iter().enumerate() {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::new(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            if self.is_moe() {
                // MoE FFN: router + stacked-3D routed experts + always-on shared expert.
                let mi = self.moe_inter;
                let si = self.shared_inter;
                t.push(TensorSpec::new(
                    format!("{p}.mlp.gate.weight"),
                    vec![self.experts, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.gate_up_proj"),
                    vec![self.experts, 2 * mi, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.down_proj"),
                    vec![self.experts, h, mi],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.gate_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.up_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.down_proj.weight"),
                    vec![h, si],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    vec![1, h],
                    Init::Uniform(0.05),
                ));
            } else {
                // Dense MLP (SwiGLU).
                t.push(TensorSpec::new(
                    format!("{p}.mlp.gate_proj.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.up_proj.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.down_proj.weight"),
                    vec![h, self.inter],
                    Init::Uniform(0.05),
                ));
            }

            if kind == "linear_attention" {
                let la = format!("{p}.linear_attn");
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_qkv.weight"),
                    vec![qkv, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_z.weight"),
                    vec![v_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_a.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_b.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.A_log"),
                    vec![self.l_val_heads],
                    Init::ALog,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.dt_bias"),
                    vec![self.l_val_heads],
                    Init::Zeros,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.conv1d.weight"),
                    vec![qkv, 1, self.conv_kernel],
                    Init::Uniform(0.1),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.norm.weight"),
                    vec![self.l_val_head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.out_proj.weight"),
                    vec![h, v_dim],
                    Init::Uniform(0.05),
                ));
            } else {
                let sa = format!("{p}.self_attn");
                t.push(TensorSpec::new(
                    format!("{sa}.q_proj.weight"),
                    vec![attn_q, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.v_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.o_proj.weight"),
                    vec![h, o_in],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.q_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
            }
        }
        t
    }
}

/// Tiny Qwen2 (arch 7) dense text config. The distinguishing feature vs LLaMA is
/// Q/K/V **bias** (attention_bias=true) — routed through the dedicated
/// hipfire-arch-qwen2 crate, which the LLaMA-default arch_id=1 path silently
/// drops. The emit-time config carries `model_type:"qwen2"` (auto-detect →
/// arch_id 1); the quant step must pass `--arch-id 7` to reach the qwen2 loader.
struct Qwen2Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attention_bias": true,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        // tie_word_embeddings ⇒ no separate lm_head.
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.q_proj.bias"),
                vec![q_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.k_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.v_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.05),
            ));
        }
        t
    }
}

/// Tiny Gemma3 (arch 12) dense text config. Exercises the Gemma quirks the
/// ingest+forward special-case: per-head QK-norm, 4 norms/layer (the
/// pre/post feed-forward norms), GeGLU, head_dim independent of dim/n_heads,
/// dual-θ sliding-window interleave, and the (1+w) RMSNorm offset the quantizer
/// bakes at ingest (arch_id 12). `sliding_window_pattern:2` over 4 layers gives
/// both local-SWA and global layers.
struct Gemma3Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window_pattern: usize,
}

impl Gemma3Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128, // must be % 32 == 0 for the q8 KV path (forward.rs)
            sliding_window_pattern: 2,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "query_pre_attn_scalar": self.head_dim,
            "sliding_window": 64,
            "sliding_window_pattern": self.sliding_window_pattern,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        // tie_word_embeddings ⇒ no separate lm_head.
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            let sa = format!("{p}.self_attn");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.pre_feedforward_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_feedforward_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.05),
            ));
        }
        t
    }
}

/// Tiny MiniMax-M2 (arch 10) Mixtral-style MoE config. Distinct from the
/// Qwen3.5 MoE: per-expert pre-split `w1/w3/w2` tensors (no stacked-3D),
/// per-layer flat QK-norm, partial rotate_half RoPE, sigmoid routing with a
/// per-expert `e_score_correction_bias`, no shared expert, and **untied**
/// lm_head. Exercises the indexed-MoE GEMV kernel family. Expert input dim
/// (hidden, inter) must be a multiple of 256 for the mq4/mq6 expert path.
struct MiniMaxTiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    experts: usize,
    experts_per_tok: usize,
}

impl MiniMaxTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 256,
            vocab: 4096,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            rotary_dim: 32,
            experts: 8,
            experts_per_tok: 2,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "model_type": "minimax_m2",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "rotary_dim": self.rotary_dim,
            "num_local_experts": self.experts,
            "num_experts_per_tok": self.experts_per_tok,
            "use_qk_norm": true,
            "use_routing_bias": true,
            "scoring_func": "sigmoid",
            "rope_theta": 5_000_000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        // Untied: embed + separate lm_head.
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            let sa = format!("{p}.self_attn");
            let moe = format!("{p}.block_sparse_moe");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            // Per-layer QK-norm on the flat projection (q_dim / kv_dim wide).
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![q_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![kv_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            // Router + per-expert bias (loaded unconditionally by the minimax loader).
            t.push(TensorSpec::new(
                format!("{moe}.gate.weight"),
                vec![self.experts, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{moe}.e_score_correction_bias"),
                vec![self.experts],
                Init::Uniform(0.02),
            ));
            for e in 0..self.experts {
                let ep = format!("{moe}.experts.{e}");
                t.push(TensorSpec::new(
                    format!("{ep}.w1.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w3.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w2.weight"),
                    vec![h, self.inter],
                    Init::Uniform(0.05),
                ));
            }
        }
        t
    }
}

/// Tiny pure Mamba-2 (arch 15) config. Mirrors state-spaces tensor names:
/// `backbone.embedding.weight`, `backbone.layers.L.mixer.*`, `backbone.norm_f`.
struct Mamba2Tiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    expand: usize,
    head_dim: usize,
    d_state: usize,
    ngroups: usize,
    conv_kernel: usize,
    chunk_size: usize,
}

impl Mamba2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            layers: 2,
            expand: 2,
            head_dim: 64,
            d_state: 128,
            ngroups: 1,
            conv_kernel: 4,
            chunk_size: 64,
        }
    }

    fn d_inner(&self) -> usize {
        self.hidden * self.expand
    }

    fn num_heads(&self) -> usize {
        self.d_inner() / self.head_dim
    }

    fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.num_heads()
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Mamba2ForCausalLM"],
            "d_model": self.hidden,
            "d_intermediate": 0,
            "n_layer": self.layers,
            "vocab_size": self.vocab,
            "ssm_cfg": {
                "layer": "Mamba2",
                "d_state": self.d_state,
                "d_conv": self.conv_kernel,
                "expand": self.expand,
                "headdim": self.head_dim,
                "ngroups": self.ngroups,
                "chunk_size": self.chunk_size,
            },
            "attn_layer_idx": [],
            "attn_cfg": {},
            "rms_norm": true,
            "residual_in_fp32": true,
            "fused_add_norm": true,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": true,
            "rms_norm_eps": 1e-5,
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let d_inner = self.d_inner();
        let conv_dim = self.conv_dim();
        let projection_size = self.projection_size();
        let heads = self.num_heads();
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "backbone.embedding.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "backbone.norm_f.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("backbone.layers.{i}");
            let m = format!("{p}.mixer");
            t.push(TensorSpec::f16(
                format!("{p}.norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.in_proj.weight"),
                vec![projection_size, h],
                Init::Uniform(0.04),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.weight"),
                vec![conv_dim, 1, self.conv_kernel],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.bias"),
                vec![conv_dim],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.A_log"),
                vec![heads],
                Init::ALog,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.D"),
                vec![heads],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.dt_bias"),
                vec![heads],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.norm.weight"),
                vec![d_inner],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.out_proj.weight"),
                vec![h, d_inner],
                Init::Uniform(0.04),
            ));
        }
        t
    }
}

/// Generate little-endian bytes for one tensor at its declared dtype.
fn gen_bytes(spec: &TensorSpec, rng: &mut SplitMix64) -> Vec<u8> {
    let n: usize = spec.shape.iter().product();
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let v = match spec.init {
            Init::Uniform(s) => rng.next_unit() * s,
            Init::NormOnes => 1.0 + rng.next_unit() * 0.02,
            Init::ALog => -2.0 + rng.next_unit() * 0.5, // exp(A_log) small & positive
            Init::Zeros => 0.0,
        };
        let bits = match spec.dt {
            Dt::Bf16 => bf16_bits(v),
            Dt::F16 => crate::f32_to_f16(v),
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// Write a safetensors file: [u64 LE header len][JSON header][concatenated data].
fn write_safetensors(
    path: &Path,
    specs: &[TensorSpec],
    rng: &mut SplitMix64,
) -> Result<(), String> {
    let mut datas: Vec<Vec<u8>> = Vec::with_capacity(specs.len());
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for spec in specs {
        let bytes = gen_bytes(spec, rng);
        let end = offset + bytes.len();
        header.insert(
            spec.name.clone(),
            serde_json::json!({
                "dtype": spec.dt.st_name(),
                "shape": spec.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
        datas.push(bytes);
    }
    let header_json = serde_json::to_string(&header).map_err(|e| e.to_string())?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    f.write_all(&(header_json.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(header_json.as_bytes())
        .map_err(|e| e.to_string())?;
    for d in &datas {
        f.write_all(d).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Emit a tiny random-init fixture for `arch` into `out_dir` (created if absent).
/// Writes `config.json` + `model.safetensors`. Reproducible for a given `seed`.
/// GPT-2 byte→unicode mapping (the fixed table every byte-level BPE uses). MUST
/// match `hipfire_model::tokenizer::byte_to_gpt2_char` exactly, or the loader's
/// `build_byte_to_id` rejects the vocab. Printable bytes map to themselves; the
/// rest map to U+0100.. in byte order. Validated by `tests::tiny_tokenizer_loads`.
fn gpt2_byte_chars() -> [char; 256] {
    let mut out = ['?'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    out
}

/// A minimal, arch-agnostic byte-level BPE `tokenizer.json` for the tiny fixtures.
/// Every model's real tokenizer is fused to its trained weights and can't be
/// swapped — but a random-init fixture's tokenizer is arbitrary, so ALL fixtures
/// share this one: 256 single-byte tokens (no merges) + `<|endoftext|>`, with a
/// `ByteLevel` pre-tokenizer/decoder so hipfire detects it as byte-level BPE. This
/// makes each tiny `.hfq` a COMPLETE model the real `serving-core` loader accepts,
/// so quant-testing runs on the production load+forward path (no bespoke harness).
fn byte_level_tokenizer_json() -> serde_json::Value {
    let chars = gpt2_byte_chars();
    let mut vocab = serde_json::Map::new();
    for (i, c) in chars.iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::Value::from(i as u64));
    }
    let eot = 256u64;
    vocab.insert("<|endoftext|>".to_string(), serde_json::Value::from(eot));
    serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "added_tokens": [{
            "id": eot, "content": "<|endoftext|>", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true,
        }],
    })
}

/// Fetch a migrated family's fixture from the offline arch registry (its `-spec`
/// crate's `ToyModel`). Returns the same `(config, specs)` shape the local `*Tiny`
/// arms produce, so `emit_fixture` writes it through the identical path. The config
/// round-trips through a `Value` so the pretty-printed bytes stay byte-identical to
/// what the old in-crate arm wrote.
fn toy_fixture_from_registry(
    arch_id: u16,
    seed: u64,
) -> Result<(serde_json::Value, Vec<TensorSpec>), String> {
    use hipfire_arch_api::{ArchId, ArchRegistry};
    let f = ArchRegistry::build()
        .get(ArchId(arch_id))
        .and_then(|a| a.caps.toy_model)
        .ok_or_else(|| format!("--emit-fixture: arch_id {arch_id} declares no ToyModel"))?
        .fixture(seed);
    let config = serde_json::from_str(&f.config_json)
        .map_err(|e| format!("parse toy config for arch {arch_id}: {e}"))?;
    Ok((config, f.tensors))
}

pub fn emit_fixture(arch: &str, out_dir: &Path, seed: u64) -> Result<(), String> {
    let arch_norm = arch.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir:?}: {e}"))?;

    let (config, specs) = match arch_norm.as_str() {
        "qwen3_5" | "qwen35" | "qwen3_5_text" => {
            let m = Qwen35Tiny::preset();
            (m.config_json(), m.manifest())
        }
        "qwen3_5_moe" | "qwen35moe" | "qwen3_5_moe_text" => {
            let m = Qwen35Tiny::moe_preset();
            (m.config_json(), m.manifest())
        }
        "qwen2" => {
            let m = Qwen2Tiny::preset();
            (m.config_json(), m.manifest())
        }
        "gemma3" | "gemma3_text" => {
            let m = Gemma3Tiny::preset();
            (m.config_json(), m.manifest())
        }
        "minimax" | "minimax_m2" => {
            let m = MiniMaxTiny::preset();
            (m.config_json(), m.manifest())
        }
        "mamba2" | "mamba_2" => {
            let m = Mamba2Tiny::preset();
            (m.config_json(), m.manifest())
        }
        // llama (arch_id 0) migrated onto its `-spec` crate's ToyModel.
        "llama" | "mistral" => toy_fixture_from_registry(0, seed)?,
        "dflash" | "dflash_draft" | "tiny_dflash" => {
            let m = DflashTiny::preset();
            (m.config_json(), m.manifest())
        }
        other => {
            return Err(format!(
                "--emit-fixture: unsupported arch '{other}'. Supported: qwen3_5 \
                 (arch 5 dense), qwen3_5_moe (arch 6 MoE), qwen2 (arch 7, quantize \
                 with --arch-id 7), gemma3 (arch 12), minimax (arch 10), mamba2 \
                 (arch 15), llama (arch 0), dflash (draft sidecar). Add a tiny \
                 preset per arch as support lands."
            ));
        }
    };

    let mut rng = SplitMix64(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    write_safetensors(&out_dir.join("model.safetensors"), &specs, &mut rng)?;
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write config.json: {e}"))?;

    // Shared byte-level tokenizer → the quantizer embeds it into the .hfq metadata
    // ("tokenizer"), making the fixture a COMPLETE model the real serving-core
    // loader accepts (so quant tests run on the production path, not a bypass).
    std::fs::write(
        out_dir.join("tokenizer.json"),
        serde_json::to_string(&byte_level_tokenizer_json()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write tokenizer.json: {e}"))?;

    let n_params: usize = specs
        .iter()
        .map(|s| s.shape.iter().product::<usize>())
        .sum();
    eprintln!(
        "emit-fixture: wrote {arch_norm} fixture to {out_dir:?} \
         ({} tensors, {:.2}M params, seed {seed:#x})",
        specs.len(),
        n_params as f64 / 1e6,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_tokenizer_loads() {
        // Emit a fixture and load its tokenizer.json through hipfire's REAL
        // tokenizer. `from_hf_json` runs `build_byte_to_id`, which errors if any
        // byte 0..=255 is missing from the vocab — so this fails loudly if our
        // `gpt2_byte_chars` ever drifts from `byte_to_gpt2_char`.
        let dir = tempfile::tempdir().unwrap();
        emit_fixture("llama", dir.path(), 42).unwrap();
        let tok_path = dir.path().join("tokenizer.json");
        assert!(tok_path.exists(), "emit_fixture must write tokenizer.json");
        let tok = hipfire_model::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .expect("tokenizer.json parses")
            .expect("tokenizer present");
        // Every byte encodes to an id within the fixture's vocab (4096), so the
        // synthetic prompt used for KLD indexes valid embedding rows.
        let ids = tok.encode("hello, hipfire! 42");
        assert!(!ids.is_empty());
        assert!(
            ids.iter().all(|&id| id < 4096),
            "ids must fit vocab: {ids:?}"
        );
    }

    #[test]
    fn bf16_roundtrip_basic() {
        // 1.0 and 0.0 are exact in bf16.
        assert_eq!(bf16_bits(0.0), 0x0000);
        assert_eq!(bf16_bits(1.0), 0x3F80);
    }

    #[test]
    fn manifest_has_both_layer_types_and_is_tiny() {
        let m = Qwen35Tiny::preset();
        let lt = m.layer_types();
        assert!(lt.contains(&"linear_attention"));
        assert!(lt.contains(&"full_attention"));
        let specs = m.manifest();
        let n: usize = specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n < 10_000_000, "fixture must stay <10M params, got {n}");
        // in_proj_qkv = 2*key + value head dims.
        let qkv = specs
            .iter()
            .find(|s| s.name.ends_with("in_proj_qkv.weight"))
            .unwrap();
        assert_eq!(qkv.shape[0], 2 * 128 * 2 + 2 * 128);
    }

    #[test]
    fn moe_manifest_has_experts_router_shared_and_is_tiny() {
        let m = Qwen35Tiny::moe_preset();
        assert!(m.is_moe());
        let specs = m.manifest();
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("mlp.gate.weight"), "router");
        assert!(has("mlp.experts.gate_up_proj"), "stacked experts");
        assert!(has("mlp.experts.down_proj"));
        assert!(has("mlp.shared_expert.gate_proj.weight"), "shared expert");
        assert!(has("mlp.shared_expert_gate.weight"));
        // stacked-3D expert tensor: [num_experts, 2*moe_inter, hidden].
        let gu = specs
            .iter()
            .find(|s| s.name.ends_with("experts.gate_up_proj"))
            .unwrap();
        assert_eq!(gu.shape, vec![8, 2 * 128, 256]);
        let n: usize = specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n < 10_000_000, "moe fixture must stay <10M params, got {n}");
    }

    /// Total param count for a manifest, for the <10M tiny budget assert.
    fn n_params(specs: &[TensorSpec]) -> usize {
        specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum()
    }

    #[test]
    fn qwen2_manifest_has_qkv_bias_and_is_tiny() {
        let m = Qwen2Tiny::preset();
        let specs = m.manifest();
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("self_attn.q_proj.bias"), "qwen2 must carry q bias");
        assert!(has("self_attn.k_proj.bias"));
        assert!(has("self_attn.v_proj.bias"));
        assert!(has("mlp.gate_proj.weight"), "dense SwiGLU");
        assert!(!has("lm_head.weight"), "tied ⇒ no separate lm_head");
        assert!(
            n_params(&specs) < 10_000_000,
            "qwen2 fixture must stay <10M params"
        );
    }

    #[test]
    fn gemma3_manifest_has_four_norms_and_qk_norm() {
        let m = Gemma3Tiny::preset();
        let specs = m.manifest();
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("self_attn.q_norm.weight"), "per-head QK-norm");
        assert!(
            has("pre_feedforward_layernorm.weight"),
            "gemma 4-norm layout"
        );
        assert!(has("post_feedforward_layernorm.weight"));
        assert_eq!(
            m.head_dim % 32,
            0,
            "gemma3 head_dim must be %32==0 for q8 KV"
        );
        assert!(
            n_params(&specs) < 10_000_000,
            "gemma3 fixture must stay <10M params"
        );
    }

    #[test]
    fn minimax_manifest_has_split_experts_router_bias_and_untied_head() {
        let m = MiniMaxTiny::preset();
        let specs = m.manifest();
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("lm_head.weight"), "minimax is untied");
        assert!(has("block_sparse_moe.gate.weight"), "router");
        assert!(
            has("block_sparse_moe.e_score_correction_bias"),
            "routing bias"
        );
        assert!(has("block_sparse_moe.experts.0.w1.weight"), "split experts");
        assert!(has("block_sparse_moe.experts.0.w2.weight"));
        assert!(has("block_sparse_moe.experts.0.w3.weight"));
        // Expert input dims must be a multiple of 256 for the mq4/mq6 expert path.
        assert_eq!(m.hidden % 256, 0);
        assert_eq!(m.inter % 256, 0);
        // All experts identical shape (packed-layout uniform-stride requirement).
        let w1: Vec<_> = specs
            .iter()
            .filter(|s| s.name.ends_with(".w1.weight"))
            .collect();
        assert!(w1.windows(2).all(|w| w[0].shape == w[1].shape));
        assert!(
            n_params(&specs) < 10_000_000,
            "minimax fixture must stay <10M params"
        );
    }

    #[test]
    fn mamba2_manifest_has_state_spaces_names_and_is_tiny() {
        let m = Mamba2Tiny::preset();
        let specs = m.manifest();
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(
            has("backbone.embedding.weight"),
            "state-spaces embedding name"
        );
        assert!(has("lm_head.weight"), "saved tied head is accepted");
        assert!(has("mixer.in_proj.weight"));
        assert!(has("mixer.conv1d.weight"));
        assert!(has("mixer.A_log"));
        assert!(has("mixer.D"));
        assert!(has("mixer.dt_bias"));
        assert!(has("mixer.norm.weight"));
        assert!(has("mixer.out_proj.weight"));
        let in_proj = specs
            .iter()
            .find(|s| s.name.ends_with("mixer.in_proj.weight"))
            .unwrap();
        assert_eq!(in_proj.shape, vec![m.projection_size(), m.hidden]);
        assert!(
            n_params(&specs) < 10_000_000,
            "mamba2 fixture must stay <10M params"
        );
    }

    // llama's fixture invariants moved to hipfire-arch-llama-spec (co-located with the
    // ToyModel that now owns the manifest).

    #[test]
    fn emit_new_families_are_deterministic() {
        let base = std::env::temp_dir().join(format!("hipfire-fx-fam-{}", std::process::id()));
        for arch in ["qwen2", "gemma3", "minimax", "mamba2", "llama"] {
            let dir = base.join(arch);
            emit_fixture(arch, &dir, 7).unwrap();
            let a = std::fs::read(dir.join("model.safetensors")).unwrap();
            emit_fixture(arch, &dir, 7).unwrap();
            let b = std::fs::read(dir.join("model.safetensors")).unwrap();
            assert_eq!(
                a, b,
                "{arch}: same seed must produce byte-identical safetensors"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn emit_is_deterministic_for_seed() {
        let dir = std::env::temp_dir().join(format!("hipfire-fixture-test-{}", std::process::id()));
        emit_fixture("qwen3_5", &dir, 42).unwrap();
        let a = std::fs::read(dir.join("model.safetensors")).unwrap();
        emit_fixture("qwen3_5", &dir, 42).unwrap();
        let b = std::fs::read(dir.join("model.safetensors")).unwrap();
        assert_eq!(a, b, "same seed must produce byte-identical safetensors");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
