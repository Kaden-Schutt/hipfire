// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `nemotron_h` architecture support (NVIDIA Nemotron-3 family) — a **flat
//! sequence of residual blocks**, each one of: Mamba-2 SSM mixer (`M`),
//! GQA attention mixer (`*`), dense MLP / FFN (`-`), or routed MoE FFN (`E`),
//! selected per layer by the model's `hybrid_override_pattern`. Starting vehicle:
//! `NVIDIA-Nemotron-3-Nano-4B` (dense, no MoE) — see
//! `docs/plans/2026-06-24-nemotron-h-mamba2.md`.
//!
//! N0 (this module): the config + block taxonomy only — pure, GPU-free, parsed
//! from the HF `config.json`. The Mamba-2 SSD kernel, conv1d xBC variant, ReLU²
//! MLP, the per-block forward, weight loader, and serving impls land in later
//! loop iterations (N1+).

pub mod arch;
pub mod attn;
pub mod block;
pub mod block_gpu;
pub mod loader;
pub mod mlp;
pub mod model;
pub mod moe;
pub mod ssd;
pub mod weight;

use hipfire_mixer::{MixerKind, MixerProfile};
use serde::Deserialize;

/// One residual block in a nemotron_h stack. Unlike a standard transformer
/// layer (mixer **and** FFN per layer), nemotron_h interleaves these as
/// independent blocks via `hybrid_override_pattern`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// `M` — Mamba-2 selective-SSM mixer (carries conv + SSM recurrent state).
    Mamba2,
    /// `*` — multi-head (GQA) attention mixer (carries a KV cache).
    Attention,
    /// `-` — dense MLP / feed-forward (ReLU² for Nano); carries no state.
    Mlp,
    /// `E` — routed MoE feed-forward block; carries no recurrent state.
    Moe,
}

impl BlockKind {
    /// Parse one `hybrid_override_pattern` character.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'M' => Some(BlockKind::Mamba2),
            '*' => Some(BlockKind::Attention),
            '-' => Some(BlockKind::Mlp),
            'E' => Some(BlockKind::Moe),
            _ => None,
        }
    }

    /// Is this a token-mixer block (Mamba-2 or attention) vs. an FFN block?
    pub fn is_mixer(self) -> bool {
        matches!(self, BlockKind::Mamba2 | BlockKind::Attention)
    }
}

/// Parse a `hybrid_override_pattern` (e.g. `"M-M-M-MM-M-M*-..."`) into the
/// per-block kind list. Errors on any unrecognized character.
pub fn parse_block_pattern(pattern: &str) -> Result<Vec<BlockKind>, String> {
    pattern
        .chars()
        .map(|c| BlockKind::from_char(c).ok_or_else(|| format!("unknown block char {c:?}")))
        .collect()
}

/// Mamba-2 mixer shape (per the `mamba_*` / `ssm_*` config fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Config {
    pub num_heads: usize,
    pub head_dim: usize,
    /// Per-head SSM state width `N` (`ssm_state_size`).
    pub state_size: usize,
    /// B/C projection groups (`n_groups`).
    pub n_groups: usize,
    /// Depthwise causal short-conv kernel width (`conv_kernel`).
    pub conv_kernel: usize,
    /// Chunked-SSD prefill chunk length (`chunk_size`).
    pub chunk_size: usize,
    pub use_conv_bias: bool,
    pub proj_bias: bool,
    /// `dt` clamp bounds (`time_step_min` / `time_step_max`); Nano-4B = 0.001/0.1.
    pub dt_min: f32,
    pub dt_max: f32,
}

impl Mamba2Config {
    /// Inner SSM dim `d_inner = num_heads × head_dim` (NB: nemotron_h uses this,
    /// **not** `expand × hidden_size`).
    pub fn d_inner(&self) -> usize {
        self.num_heads * self.head_dim
    }
    /// Width of the conv'd `xBC = [x | B | C]` stream.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.n_groups * self.state_size
    }
    /// `in_proj` output width = `d_inner + conv_dim + num_heads` (`[z|xBC|dt]`,
    /// `d_mlp=0` for nemotron_h).
    pub fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.num_heads
    }
}

/// GQA attention mixer shape for the `*` blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub bias: bool,
}

/// Routed MoE FFN shape for `E` blocks (Nano-30B A3B).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeConfig {
    /// Number of routed experts (`n_routed_experts`).
    pub n_routed_experts: usize,
    /// Top-k experts selected per token (`num_experts_per_tok`).
    pub num_experts_per_tok: usize,
    /// Routed expert intermediate width (`moe_intermediate_size`).
    pub intermediate_size: usize,
    /// Shared expert count (`n_shared_experts`); Nano-30B uses 1.
    pub n_shared_experts: usize,
    /// Shared expert intermediate width (`moe_shared_expert_intermediate_size`).
    pub shared_expert_intermediate_size: usize,
    /// Router grouping (`n_group`).
    pub n_group: usize,
    /// Number of groups to keep before top-k (`topk_group`).
    pub topk_group: usize,
    /// Normalize selected sigmoid scores before applying `routed_scaling_factor`.
    pub norm_topk_prob: bool,
    /// Scale applied to selected routed-expert weights.
    pub routed_scaling_factor: f32,
}

/// Parsed nemotron_h model config.
#[derive(Clone, Debug, PartialEq)]
pub struct NemotronHConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// EOS / end-of-turn token id (`eos_token_id`); Nano-4B = 2.
    pub eos_token_id: u32,
    /// Per-block kinds, parsed from `hybrid_override_pattern` (length == num_layers).
    pub blocks: Vec<BlockKind>,
    pub mamba: Mamba2Config,
    pub attn: AttnConfig,
    /// Dense MLP intermediate width (`intermediate_size`).
    pub mlp_intermediate: usize,
    /// MLP activation tag (`mlp_hidden_act`, e.g. `"relu2"`).
    pub mlp_act: String,
    /// Routed MoE configuration. Present when the config has any `E` blocks.
    pub moe: Option<MoeConfig>,
}

impl NemotronHConfig {
    /// Parse from the HF `config.json` value.
    pub fn from_json(c: &serde_json::Value) -> Result<Self, String> {
        let raw: RawConfig =
            serde_json::from_value(c.clone()).map_err(|e| format!("nemotron_h config: {e}"))?;
        let blocks = parse_block_pattern(&raw.hybrid_override_pattern)?;
        // dt forward clamp = `time_step_limit` (a [min,max] pair), default
        // (0, +inf) when absent/null — i.e. effectively no clamp.
        let (dt_min, dt_max) = match raw.time_step_limit.as_ref() {
            Some(v) if v.len() == 2 => (v[0], v[1]),
            _ => (0.0f32, f32::INFINITY),
        };
        if blocks.len() != raw.num_hidden_layers {
            return Err(format!(
                "hybrid_override_pattern length {} != num_hidden_layers {}",
                blocks.len(),
                raw.num_hidden_layers
            ));
        }
        let has_moe = blocks.iter().any(|b| *b == BlockKind::Moe);
        let moe = if has_moe {
            let n_routed_experts = raw
                .n_routed_experts
                .ok_or_else(|| "nemotron_h MoE config missing n_routed_experts".to_string())?;
            let num_experts_per_tok = raw
                .num_experts_per_tok
                .ok_or_else(|| "nemotron_h MoE config missing num_experts_per_tok".to_string())?;
            let intermediate_size = raw
                .moe_intermediate_size
                .ok_or_else(|| "nemotron_h MoE config missing moe_intermediate_size".to_string())?;
            let n_shared_experts = raw.n_shared_experts.unwrap_or(0);
            let shared_expert_intermediate_size =
                raw.moe_shared_expert_intermediate_size.unwrap_or(0);
            if n_routed_experts == 0 || num_experts_per_tok == 0 || intermediate_size == 0 {
                return Err("nemotron_h MoE config has zero routed dimensions".to_string());
            }
            if n_shared_experts > 0 && shared_expert_intermediate_size == 0 {
                return Err(
                    "nemotron_h MoE config has shared experts but no shared intermediate size"
                        .to_string(),
                );
            }
            Some(MoeConfig {
                n_routed_experts,
                num_experts_per_tok,
                intermediate_size,
                n_shared_experts,
                shared_expert_intermediate_size,
                n_group: raw.n_group.unwrap_or(1),
                topk_group: raw.topk_group.unwrap_or(1),
                norm_topk_prob: raw.norm_topk_prob.unwrap_or(true),
                routed_scaling_factor: raw.routed_scaling_factor.unwrap_or(1.0),
            })
        } else {
            None
        };
        Ok(Self {
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            num_layers: raw.num_hidden_layers,
            rms_norm_eps: raw.rms_norm_eps,
            tie_word_embeddings: raw.tie_word_embeddings,
            eos_token_id: raw.eos_token_id,
            blocks,
            mamba: Mamba2Config {
                num_heads: raw.mamba_num_heads,
                head_dim: raw.mamba_head_dim,
                state_size: raw.ssm_state_size,
                n_groups: raw.n_groups,
                conv_kernel: raw.conv_kernel,
                chunk_size: raw.chunk_size,
                use_conv_bias: raw.use_conv_bias,
                proj_bias: raw.mamba_proj_bias,
                // The SSM forward clamps dt by `time_step_limit` (default
                // (0, inf) — a no-op). `time_step_min/max` are INIT-only (used
                // for dt_bias initialization), NOT the forward clamp — clamping
                // the forward to [0.001, 0.1] is wrong (verified vs HF dump).
                dt_min: dt_min,
                dt_max: dt_max,
            },
            attn: AttnConfig {
                num_heads: raw.num_attention_heads,
                num_kv_heads: raw.num_key_value_heads,
                head_dim: raw.head_dim,
                bias: raw.attention_bias,
            },
            mlp_intermediate: raw.intermediate_size,
            mlp_act: raw.mlp_hidden_act,
            moe,
        })
    }

    /// Bridge the Mamba-2 mixer shape to the GPU/CPU block dims
    /// ([`block::Mamba2Dims`]), folding in the top-level `hidden_size` and
    /// `rms_norm_eps`.
    pub fn mamba2_dims(&self) -> block::Mamba2Dims {
        block::Mamba2Dims {
            hidden_size: self.hidden_size,
            num_heads: self.mamba.num_heads,
            head_dim: self.mamba.head_dim,
            state_size: self.mamba.state_size,
            n_groups: self.mamba.n_groups,
            conv_kernel: self.mamba.conv_kernel,
            rms_norm_eps: self.rms_norm_eps,
            dt_min: self.mamba.dt_min,
            dt_max: self.mamba.dt_max,
        }
    }

    /// Runtime scale for Mamba `out_proj.weight`.
    ///
    /// Nano-4B's checkpoint/reference path needs the GPT-style residual rescale
    /// applied at load time. Nano-30B-A3B's MoE checkpoint already matches the
    /// HF reference with the stored `out_proj` bytes, so applying the same scale
    /// again collapses the first Mamba block output.
    pub fn mamba_out_proj_runtime_scale(&self) -> f32 {
        if self.moe.is_some() {
            1.0
        } else {
            1.0f32 / (self.num_layers as f32).sqrt()
        }
    }

    /// Number of blocks of each kind.
    pub fn count(&self, kind: BlockKind) -> usize {
        self.blocks.iter().filter(|&&b| b == kind).count()
    }

    /// The per-mixer-layer [`MixerProfile`] (the `M`/`*` blocks, in order) that
    /// keys the unified `SequenceState`: `Mamba2` → recurrent SSM state,
    /// `Attention` → KV. The `-` MLP blocks are FFN-only (no mixer state) and are
    /// excluded. `needs_kv_cache()` is true whenever the stack has an attention
    /// block.
    pub fn mixer_profile(&self) -> MixerProfile {
        MixerProfile::new(
            self.blocks
                .iter()
                .filter_map(|b| match b {
                    BlockKind::Mamba2 => Some(MixerKind::Mamba2),
                    BlockKind::Attention => Some(MixerKind::FullAttn),
                    BlockKind::Mlp | BlockKind::Moe => None,
                })
                .collect(),
        )
    }
}

/// Serde shape of the relevant `config.json` keys.
#[derive(Deserialize)]
struct RawConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_eos")]
    eos_token_id: u32,
    hybrid_override_pattern: String,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    ssm_state_size: usize,
    n_groups: usize,
    conv_kernel: usize,
    #[serde(default = "default_chunk")]
    chunk_size: usize,
    #[serde(default)]
    use_conv_bias: bool,
    #[serde(default)]
    mamba_proj_bias: bool,
    // INIT-only (dt_bias init); NOT the forward clamp. Kept for completeness.
    #[serde(default = "default_dt_min")]
    #[allow(dead_code)]
    time_step_min: f32,
    #[serde(default = "default_dt_max")]
    #[allow(dead_code)]
    time_step_max: f32,
    /// The forward dt clamp `[min, max]` (default (0, +inf) when null/absent).
    #[serde(default)]
    time_step_limit: Option<Vec<f32>>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default)]
    attention_bias: bool,
    intermediate_size: usize,
    #[serde(default = "default_act")]
    mlp_hidden_act: String,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    n_shared_experts: Option<usize>,
    #[serde(default)]
    moe_shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    n_group: Option<usize>,
    #[serde(default)]
    topk_group: Option<usize>,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    #[serde(default)]
    routed_scaling_factor: Option<f32>,
}

fn default_eps() -> f32 {
    1e-5
}
fn default_chunk() -> usize {
    256
}
fn default_eos() -> u32 {
    2
}
fn default_dt_min() -> f32 {
    0.001
}
fn default_dt_max() -> f32 {
    0.1
}
fn default_act() -> String {
    "relu2".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verified Nemotron-3-Nano-4B `hybrid_override_pattern`.
    const NANO_4B_PATTERN: &str = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-";
    /// The verified Nemotron-3-Nano-30B-A3B `hybrid_override_pattern`.
    const NANO_30B_PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

    #[test]
    fn parses_nano_4b_pattern() {
        let blocks = parse_block_pattern(NANO_4B_PATTERN).unwrap();
        assert_eq!(blocks.len(), 42);
        assert_eq!(
            blocks.iter().filter(|b| **b == BlockKind::Mamba2).count(),
            21
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|b| **b == BlockKind::Attention)
                .count(),
            4
        );
        assert_eq!(blocks.iter().filter(|b| **b == BlockKind::Mlp).count(), 17);
    }

    #[test]
    fn rejects_unknown_block_char() {
        assert!(parse_block_pattern("M-X-").is_err());
    }

    #[test]
    fn mamba2_derived_dims() {
        let m = Mamba2Config {
            num_heads: 96,
            head_dim: 80,
            state_size: 128,
            n_groups: 8,
            conv_kernel: 4,
            chunk_size: 256,
            use_conv_bias: true,
            proj_bias: false,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        assert_eq!(m.d_inner(), 7680); // heads*head_dim, NOT expand*hidden
        assert_eq!(m.conv_dim(), 7680 + 2 * 8 * 128); // x + B + C
        assert_eq!(m.projection_size(), 7680 + 9728 + 96);
    }

    #[test]
    fn full_config_from_json_nano_4b() {
        let json = serde_json::json!({
            "model_type": "nemotron_h",
            "hidden_size": 3136,
            "vocab_size": 131072,
            "num_hidden_layers": 42,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "hybrid_override_pattern": NANO_4B_PATTERN,
            "mamba_num_heads": 96,
            "mamba_head_dim": 80,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "chunk_size": 256,
            "use_conv_bias": true,
            "mamba_proj_bias": false,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "attention_bias": false,
            "intermediate_size": 12544,
            "mlp_hidden_act": "relu2",
            "time_step_min": 0.001,
            "time_step_max": 0.1,
        });
        let cfg = NemotronHConfig::from_json(&json).unwrap();
        assert_eq!(cfg.num_layers, 42);
        assert_eq!(cfg.blocks.len(), 42);
        assert_eq!(cfg.mamba.d_inner(), 7680);
        assert_eq!(cfg.count(BlockKind::Attention), 4);
        assert_eq!(cfg.mlp_act, "relu2");
        // No time_step_limit in the json → forward dt clamp is (0, +inf).
        assert_eq!(cfg.mamba.dt_min, 0.0);
        assert!(cfg.mamba.dt_max.is_infinite());
        // config → block dims bridge folds in hidden_size + eps.
        let dims = cfg.mamba2_dims();
        assert_eq!(dims.hidden_size, 3136);
        assert_eq!(dims.d_inner(), 7680);
        assert_eq!(dims.conv_dim(), 9728);
        assert_eq!(dims.norm_group_size(), 960);
        // MixerProfile excludes the MLP blocks (25 mixers: 21 Mamba2 + 4 attn).
        let prof = cfg.mixer_profile();
        assert_eq!(prof.n_layers(), 25);
        assert!(prof.needs_kv_cache()); // has attention blocks
        assert!(prof.has_recurrent_state()); // has Mamba2 blocks
        assert!(prof.is_hybrid());
        assert_eq!(
            cfg.mamba_out_proj_runtime_scale(),
            1.0f32 / (42.0f32).sqrt()
        );
    }

    #[test]
    fn parses_nano_30b_moe_config() {
        let json = serde_json::json!({
            "model_type": "nemotron_h",
            "hidden_size": 2688,
            "vocab_size": 131072,
            "num_hidden_layers": 52,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "hybrid_override_pattern": NANO_30B_PATTERN,
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "chunk_size": 128,
            "use_conv_bias": true,
            "mamba_proj_bias": false,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "attention_bias": false,
            "intermediate_size": 1856,
            "mlp_hidden_act": "relu2",
            "n_routed_experts": 128,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1856,
            "n_shared_experts": 1,
            "moe_shared_expert_intermediate_size": 3712,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
        });
        let cfg = NemotronHConfig::from_json(&json).unwrap();
        assert_eq!(cfg.num_layers, 52);
        assert_eq!(cfg.count(BlockKind::Mamba2), 23);
        assert_eq!(cfg.count(BlockKind::Attention), 6);
        assert_eq!(cfg.count(BlockKind::Moe), 23);
        assert_eq!(cfg.count(BlockKind::Mlp), 0);
        let moe = cfg.moe.unwrap();
        assert_eq!(moe.n_routed_experts, 128);
        assert_eq!(moe.num_experts_per_tok, 6);
        assert_eq!(moe.intermediate_size, 1856);
        assert_eq!(moe.n_shared_experts, 1);
        assert_eq!(moe.shared_expert_intermediate_size, 3712);
        assert!(moe.norm_topk_prob);
        assert_eq!(moe.routed_scaling_factor, 2.5);
        assert_eq!(cfg.mamba_out_proj_runtime_scale(), 1.0);
        // MoE blocks are FFN blocks, so the mixer profile still only tracks M/*.
        assert_eq!(cfg.mixer_profile().n_layers(), 29);
    }
}
