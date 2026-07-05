// SPDX-License-Identifier: Apache-2.0
//! Minimal LLaMA-family config for the training path.
//!
//! Parsed straight from a HuggingFace `config.json`. Only the fields the
//! un-fused fp32 forward needs — no quant config, no arch dispatch. Validated
//! against Supra-50M (`model_type: llama`, dense, no attention bias).

use std::path::Path;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

impl LlamaConfig {
    /// Read and validate `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let txt =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&txt).map_err(|e| format!("parse config.json: {e}"))?;
        Self::from_json_value(&v)
    }

    /// Parse from a `.hfq` HFQM metadata JSON string. The quantizer stores the
    /// HF config under `{"architecture":..,"config":{..}}` — this navigates to
    /// `config` and parses the same fields as `config.json`. Enables training to
    /// load its base from the exact served artifact (layer-1 runtime unification).
    pub fn from_hfq_metadata(metadata_json: &str) -> Result<Self, String> {
        let m: serde_json::Value =
            serde_json::from_str(metadata_json).map_err(|e| format!("parse hfq metadata: {e}"))?;
        let cfg = m.get("config").unwrap_or(&m);
        Self::from_json_value(cfg)
    }

    /// Core parser shared by `from_dir` and `from_hfq_metadata`.
    pub fn from_json_value(v: &serde_json::Value) -> Result<Self, String> {
        let model_type = v["model_type"].as_str().unwrap_or("");
        // Dense llama-family decoders share the same block geometry. qwen2/qwen3
        // add QK-norm, but consumers that only need embed/lm-head + geometry
        // (e.g. the DSpark drafter trainer, which never runs the target forward)
        // can load them as llama. Fused-inference correctness for qk-norm is a
        // separate concern handled by the arch crates, not this fp32 trainer.
        if !matches!(model_type, "llama" | "qwen2" | "qwen3") {
            return Err(format!(
                "hipfire-train supports dense llama-family model_type (llama/qwen2/qwen3), got {model_type:?}"
            ));
        }
        // Phase 0 is the plain dense path: reject biases we don't model yet.
        if v["attention_bias"].as_bool().unwrap_or(false) {
            return Err("attention_bias=true unsupported in Phase 0 (use a no-bias llama)".into());
        }

        let hidden_size = uget(v, "hidden_size")?;
        let num_attention_heads = uget(v, "num_attention_heads")?;
        let head_dim = v["head_dim"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let rope_theta = v["rope_theta"]
            .as_f64()
            .or_else(|| v["rope_parameters"]["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32;

        Ok(Self {
            hidden_size,
            intermediate_size: uget(v, "intermediate_size")?,
            num_hidden_layers: uget(v, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: v["num_key_value_heads"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(num_attention_heads),
            head_dim,
            vocab_size: uget(v, "vocab_size")?,
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            max_position_embeddings: v["max_position_embeddings"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2048),
        })
    }

    /// q/k/v projection output widths.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

fn uget(v: &serde_json::Value, key: &str) -> Result<usize, String> {
    v[key]
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("config.json missing/invalid usize field: {key}"))
}
