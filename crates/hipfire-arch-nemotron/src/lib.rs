// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `nemotron_h` architecture support (NVIDIA Nemotron-3 family) — a **flat
//! sequence of residual blocks**, each one of: Mamba-2 SSM mixer (`M`),
//! GQA attention mixer (`*`), or a dense MLP / FFN (`-`), selected per layer by
//! the model's `hybrid_override_pattern`. Starting vehicle:
//! `NVIDIA-Nemotron-3-Nano-4B` (dense, no MoE) — see
//! `docs/plans/2026-06-24-nemotron-h-mamba2.md`.
//!
//! N0 (this module): the config + block taxonomy only — pure, GPU-free, parsed
//! from the HF `config.json`. The Mamba-2 SSD kernel, conv1d xBC variant, ReLU²
//! MLP, the per-block forward, weight loader, and serving impls land in later
//! loop iterations (N1+).

pub mod block;
pub mod ssd;

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
}

impl BlockKind {
    /// Parse one `hybrid_override_pattern` character.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'M' => Some(BlockKind::Mamba2),
            '*' => Some(BlockKind::Attention),
            '-' => Some(BlockKind::Mlp),
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
}

/// GQA attention mixer shape for the `*` blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub bias: bool,
}

/// Parsed nemotron_h model config.
#[derive(Clone, Debug, PartialEq)]
pub struct NemotronHConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// Per-block kinds, parsed from `hybrid_override_pattern` (length == num_layers).
    pub blocks: Vec<BlockKind>,
    pub mamba: Mamba2Config,
    pub attn: AttnConfig,
    /// Dense MLP intermediate width (`intermediate_size`).
    pub mlp_intermediate: usize,
    /// MLP activation tag (`mlp_hidden_act`, e.g. `"relu2"`).
    pub mlp_act: String,
}

impl NemotronHConfig {
    /// Parse from the HF `config.json` value.
    pub fn from_json(c: &serde_json::Value) -> Result<Self, String> {
        let raw: RawConfig =
            serde_json::from_value(c.clone()).map_err(|e| format!("nemotron_h config: {e}"))?;
        let blocks = parse_block_pattern(&raw.hybrid_override_pattern)?;
        if blocks.len() != raw.num_hidden_layers {
            return Err(format!(
                "hybrid_override_pattern length {} != num_hidden_layers {}",
                blocks.len(),
                raw.num_hidden_layers
            ));
        }
        Ok(Self {
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            num_layers: raw.num_hidden_layers,
            rms_norm_eps: raw.rms_norm_eps,
            tie_word_embeddings: raw.tie_word_embeddings,
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
            },
            attn: AttnConfig {
                num_heads: raw.num_attention_heads,
                num_kv_heads: raw.num_key_value_heads,
                head_dim: raw.head_dim,
                bias: raw.attention_bias,
            },
            mlp_intermediate: raw.intermediate_size,
            mlp_act: raw.mlp_hidden_act,
        })
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
                    BlockKind::Mlp => None,
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
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default)]
    attention_bias: bool,
    intermediate_size: usize,
    #[serde(default = "default_act")]
    mlp_hidden_act: String,
}

fn default_eps() -> f32 {
    1e-5
}
fn default_chunk() -> usize {
    256
}
fn default_act() -> String {
    "relu2".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verified Nemotron-3-Nano-4B `hybrid_override_pattern`.
    const NANO_4B_PATTERN: &str = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-";

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
        assert!(parse_block_pattern("M-E-").is_err()); // 'E' (MoE) not in Nano
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
        };
        assert_eq!(m.d_inner(), 7680); // heads*head_dim, NOT expand*hidden
        assert_eq!(m.conv_dim(), 7680 + 2 * 8 * 128); // x + B + C
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
        });
        let cfg = NemotronHConfig::from_json(&json).unwrap();
        assert_eq!(cfg.num_layers, 42);
        assert_eq!(cfg.blocks.len(), 42);
        assert_eq!(cfg.mamba.d_inner(), 7680);
        assert_eq!(cfg.count(BlockKind::Attention), 4);
        assert_eq!(cfg.mlp_act, "relu2");
        // MixerProfile excludes the MLP blocks (25 mixers: 21 Mamba2 + 4 attn).
        let prof = cfg.mixer_profile();
        assert_eq!(prof.n_layers(), 25);
        assert!(prof.needs_kv_cache()); // has attention blocks
        assert!(prof.has_recurrent_state()); // has Mamba2 blocks
        assert!(prof.is_hybrid());
    }
}
