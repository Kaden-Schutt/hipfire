//! `Architecture` trait impl for the Qwen2 dense text decoder.
//!
//! The five required trait methods (`arch_id` / `name` / `config_from_hfq`
//! / `load_weights` / `new_state`) delegate to real implementations in
//! [`crate::qwen2`]. Optional overrides set Qwen2-specific defaults
//! where they diverge from the Qwen3.5 family conventions (mostly
//! `eos_filter_overrides.strip_think = Some(false)` — Qwen2 isn't a
//! thinking-mode model).
//!
//! Forward pass is intentionally NOT on this trait — see
//! `hipfire_runtime::arch` module docs for the rationale (static
//! dispatch in hot path, arch-specific forward signatures). Callers
//! reach the hot path via [`crate::qwen2::forward_step`] /
//! [`crate::qwen2::forward_step_greedy`] directly.

use crate::qwen2::{Qwen2Config, Qwen2State, Qwen2Weights};
use hipfire_runtime::arch::{
    Architecture, EosFilterOverrides, LoopGuardOverrides, PromptFrameOverrides, SamplerOverrides,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_manifest::{
    FusedQkvLayout, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::DType;
use rdna_compute::Gpu;

/// Zero-sized type marker for the Qwen2 arch.
pub struct Qwen2;

impl Architecture for Qwen2 {
    type Weights = Qwen2Weights;
    type State = Qwen2State;
    type Config = Qwen2Config;

    /// arch_id = 7 for the Qwen2 family.
    ///
    /// Note: `arch_id = 1` is nominally "plain Qwen3/Qwen2" per the trait
    /// doc, but in practice the LLaMA crate (`hipfire-arch-llama`) covers
    /// `arch_id = 0` AND `arch_id = 1` (Qwen3/Qwen2) via its
    /// `config_from_hfq` branch. The daemon dispatch at
    /// `daemon.rs:1494` routes everything `< 5` to the LLaMA path.
    /// Taking the next-free slot 7 avoids restructuring that.
    /// See `docs/architecture-ids.md` and `docs/plans/dots-ocr-prd.md`
    /// §3a.
    fn arch_id() -> u32 {
        7
    }

    fn name() -> &'static str {
        "qwen2"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Qwen2Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Qwen2Weights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Qwen2State::new(gpu, cfg)
    }

    /// Dense (GQA) weight manifest (device-mesh Phase 2). Qwen2 specifics vs
    /// vanilla llama: attention Q/K/V carry **biases** (`attention_bias`), and
    /// the lm_head may be **tied** to the embedding (`tie_word_embeddings`).
    fn weight_manifest(cfg: &Self::Config) -> Vec<WeightEntry> {
        use ShardPolicy::*;
        let (d, ff, hd) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let mut m = Vec::new();
        m.push(WeightEntry::model(
            "token_embd",
            vec![cfg.vocab_size, d],
            DType::F16,
            Pin(PinTarget::Embed),
        ));
        for l in 0..cfg.num_hidden_layers {
            m.push(WeightEntry::layer(
                "wq",
                l,
                vec![nh * hd, d],
                DType::F16,
                FusedQkv {
                    q_heads: nh,
                    kv_heads: nkv,
                    head_dim: hd,
                    layout: FusedQkvLayout::Qkv,
                },
            ));
            m.push(WeightEntry::layer(
                "wk",
                l,
                vec![nkv * hd, d],
                DType::F16,
                ColumnShard { axis: 0 },
            ));
            m.push(WeightEntry::layer(
                "wv",
                l,
                vec![nkv * hd, d],
                DType::F16,
                ColumnShard { axis: 0 },
            ));
            m.push(WeightEntry::layer(
                "wo",
                l,
                vec![d, nh * hd],
                DType::F16,
                RowShard { axis: 1 },
            ));
            if cfg.attention_bias {
                // Biases shard with their (column-parallel) projection's output.
                m.push(WeightEntry::layer(
                    "bq",
                    l,
                    vec![nh * hd],
                    DType::F32,
                    ColumnShard { axis: 0 },
                ));
                m.push(WeightEntry::layer(
                    "bk",
                    l,
                    vec![nkv * hd],
                    DType::F32,
                    ColumnShard { axis: 0 },
                ));
                m.push(WeightEntry::layer(
                    "bv",
                    l,
                    vec![nkv * hd],
                    DType::F32,
                    ColumnShard { axis: 0 },
                ));
            }
            m.push(WeightEntry::layer(
                "ffn_gate",
                l,
                vec![ff, d],
                DType::F16,
                ColumnShard { axis: 0 },
            ));
            m.push(WeightEntry::layer(
                "ffn_up",
                l,
                vec![ff, d],
                DType::F16,
                ColumnShard { axis: 0 },
            ));
            m.push(WeightEntry::layer(
                "ffn_down",
                l,
                vec![d, ff],
                DType::F16,
                RowShard { axis: 1 },
            ));
            m.push(WeightEntry::layer(
                "attn_norm",
                l,
                vec![d],
                DType::F32,
                Replicate,
            ));
            m.push(WeightEntry::layer(
                "ffn_norm",
                l,
                vec![d],
                DType::F32,
                Replicate,
            ));
        }
        m.push(WeightEntry::model(
            "output_norm",
            vec![d],
            DType::F32,
            Replicate,
        ));
        // Tied lm_head aliases the embedding; else a separate output-pinned weight.
        let lm_policy = if cfg.tie_word_embeddings {
            Tied {
                source: "token_embd".to_string(),
            }
        } else {
            Pin(PinTarget::Output)
        };
        m.push(WeightEntry::model(
            "lm_head",
            vec![cfg.vocab_size, d],
            DType::F16,
            lm_policy,
        ));
        m
    }

    /// Qwen2 is full-attention → one KV `StateEntry` per layer.
    fn state_manifest(cfg: &Self::Config) -> Vec<StateEntry> {
        (0..cfg.num_hidden_layers)
            .map(|l| {
                StateEntry::new(
                    StateKind::Kv {
                        quant: String::new(),
                    },
                    l,
                )
            })
            .collect()
    }

    // ── Optional overrides ────────────────────────────────────────────
    //
    // Qwen2-1.5B-Instruct uses standard ChatML framing (`<|im_start|>` /
    // `<|im_end|>`) and emits no `<think>` blocks. The qwen35 defaults
    // mostly fit; the one explicit override is to disable `<think>`
    // stripping since Qwen2-1.5B-Instruct doesn't emit thinking blocks.

    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        LoopGuardOverrides::default()
    }

    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides::default()
    }

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        // ChatML default applies to Qwen2-1.5B-Instruct.
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: vec![],
            holdback_prefixes: vec![],
            strip_think: Some(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_arch_id_and_name() {
        assert_eq!(Qwen2::arch_id(), 7);
        assert_eq!(Qwen2::name(), "qwen2");
    }
}
