// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for Qwen3.5.
//!
//! This is the canary arch implementation (PR 8 of
//! `docs/plans/engine-modularization.prd`). The trait surface here defines
//! what later arch crates (qwen35-vl in PR 9, llama in PR 11) must
//! implement.
//!
//! Forward-pass dispatch is INTENTIONALLY NOT routed through the trait.
//! `daemon.rs` and other consumers call `qwen35::forward_scratch`,
//! `qwen35::forward_prefill_batch`, etc. directly. Reasons:
//!   1. Forward signatures vary heavily across arches (number of buffers,
//!      KV layout, hybrid-vs-dense paths, vision conditioning, MoE expert
//!      management). Forcing a single trait shape would either bloat the
//!      contract or hide essential parameters behind opaque slots.
//!   2. Forward dispatch is hot-path. Static dispatch via concrete-type
//!      function calls keeps the call graph fully inlinable; dyn-trait
//!      dispatch in the inner loop costs measurable tok/s on small models.
//!   3. The trait's job is BRING-UP scaffolding (load → instantiate →
//!      generation-loop wiring), not runtime polymorphism. Once an arch
//!      is loaded, the daemon/CLI knows the concrete type at compile time.
//!
//! The trait gives:
//!   - one entry point per arch for config parsing + weight load + state
//!     init (the bring-up triple),
//!   - a place to register arch-specific overrides for loop_guard /
//!     sampler / prompt_frame / eos_filter without growing daemon's
//!     `match arch_id` ladder,
//!   - a discoverable contract for adding a new arch ("implement this trait
//!     and register your `arch_id`").

use crate::qwen35::{
    config_from_hfq as qwen35_config_from_hfq, DeltaNetState, LayerType, Qwen35Config,
    Qwen35Weights,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, PinTarget, PlacementHint, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::{DType, Gpu};

/// Type marker for Qwen3.5 architecture (dense Qwen3.5 0.8B/4B/9B/27B,
/// MoE Qwen3.5-A3B/A10B/A17B, dense Qwen3.6, MoE Qwen3.6-A3B). All share
/// the hybrid DeltaNet + FullAttention layer scheme.
pub struct Qwen35;

/// STEP-003: derive the compact DeltaNet LA-layer → device map from the
/// state manifest + mesh, defining the hand-built `la_to_device` sidecar
/// out of existence.
///
/// The manifest (`state_manifest`) emits one `StateKind::Recurrent` entry
/// per LinearAttention layer, keyed by GLOBAL layer index. Filtering for
/// Recurrent entries in manifest order yields exactly the compact DeltaNet
/// layer order, and the owning device is the mesh's
/// `device_for_layer(global_layer)` — the same placement the KV cache and
/// scratch sets use. The sidecar duplicated this computation in a second,
/// unkeyed `Vec<u8>` that could drift from the mesh; this helper is the
/// single source of truth for recurrent/conv state placement.
pub fn qwen35_la_devices(cfg: &Qwen35Config, gpus: &hipfire_hardware::Gpus) -> Vec<u8> {
    <Qwen35 as Architecture>::state_manifest(cfg)
        .iter()
        .filter(|e| matches!(e.kind, StateKind::Recurrent))
        .map(|e| gpus.device_for_layer(e.layer) as u8)
        .collect()
}

/// Qwen3.5-specific TP admission check. The current Qwen35 loader does not
/// materialize a TP manifest yet, so admission is deliberately pure config
/// policy: known unsafe packed layouts cannot be permitted by incomplete or
/// differently named source metadata.
pub fn qwen35_tp_preflight(cfg: &Qwen35Config, tp: usize) -> Result<(), String> {
    if tp <= 1 {
        return Ok(());
    }

    if cfg.n_layers > 0 && cfg.layer_types.len() != cfg.n_layers {
        return Err(format!(
            "qwen35: Tp={tp} requires layer_types for all {} layers (got {})",
            cfg.n_layers,
            cfg.layer_types.len()
        ));
    }

    // Every non-empty Qwen35 topology is refused for TP>1: the loader does
    // not materialize a TP manifest yet, so admission is structural policy
    // on the packed layouts, not a per-layer check. The first layer is
    // inspected only to pick the diagnostic reason naming that layout. An
    // empty `layer_types` (n_layers == 0) still admits, as before.
    let reason = match cfg.layer_types.first() {
        Some(LayerType::LinearAttention) => "DeltaNet wqkv",
        Some(LayerType::FullAttention) => "packed QGate wq",
        None => return Ok(()),
    };
    Err(format!(
        "qwen35: Tp={tp} is unsupported for {reason} (layer 0)"
    ))
}

impl Architecture for Qwen35 {
    type Weights = Qwen35Weights;
    type State = DeltaNetState;
    type Config = Qwen35Config;

    fn arch_id() -> u32 {
        // arch_id 5 = Qwen3.5 dense, arch_id 6 = Qwen3.5/3.6 MoE (A3B).
        // Returns the dense ID as the canonical "Qwen3.5 family" marker;
        // the actual id loaded at runtime is on `HfqFile::arch_id` and is
        // either 5 or 6.
        5
    }

    fn name() -> &'static str {
        "qwen35"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        // REAP is applied INSIDE `qwen35_config_from_hfq` (the public free fn)
        // so every caller — trait or direct — gets it; do NOT re-apply here,
        // or the keep-map would be applied twice (double-overriding num_experts
        // to kept-of-kept, which would then fail load_any's kept-count
        // validation against the original count).
        qwen35_config_from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        crate::store::load_qwen35_hfq_weights(hfq, cfg, gpu)
    }

    /// Logical Qwen3.5 weight inventory.  This deliberately describes the
    /// tensors owned by the current single-device `LayerWeights` variants;
    /// source-name resolution is format-specific, while fulfillment and typed
    /// assembly are shared by the production HFQ and Paro loaders.
    fn weight_manifest(cfg: &Self::Config) -> Vec<WeightEntry> {
        use ShardPolicy::*;

        let d = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let d_inner = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let is_moe = cfg.num_experts > 0;
        let mut m = Vec::new();
        let raw_f32 = || {
            DTypeConstraint::source_from_sources(vec![
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::Q8_0,
                DType::MQ4G256,
                DType::ParoQ4G128,
            ])
        };

        m.push(WeightEntry::model(
            "token_embd",
            vec![cfg.vocab_size, d],
            DType::F16,
            Pin(PinTarget::Embed),
        ));

        // HFQ's legacy source-read order: embedding, final norm, lm_head,
        // then layers.  Paro supplies a format-specific MoE order because its
        // scalar shared-expert gate is read before the quantized projections.
        m.push(WeightEntry::model(
            "output_norm",
            vec![d],
            DType::F32,
            Pin(PinTarget::Output),
        ));
        let lm_policy = if cfg.tie_word_embeddings {
            Tied {
                source: "token_embd".to_string(),
            }
        } else {
            Pin(PinTarget::Output)
        };
        m.push(
            WeightEntry::model("lm_head", vec![cfg.vocab_size, d], DType::F16, lm_policy)
                .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );

        for (layer, layer_type) in cfg.layer_types.iter().enumerate() {
            m.push(WeightEntry::layer(
                "attn_norm",
                layer,
                vec![d],
                DType::F32,
                Replicate,
            ));

            match layer_type {
                LayerType::FullAttention => {
                    // q_proj is Q|gate in Qwen3.5.  It is intentionally not
                    // advertised as a FusedQkv shard: Task 2 preflight owns
                    // the current unsafe packed-QGate TP admission rule.
                    m.push(WeightEntry::layer(
                        "wq",
                        layer,
                        vec![2 * q_dim, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wk",
                        layer,
                        vec![kv_dim, d],
                        DType::F16,
                        ColumnShard { axis: 0 },
                    ));
                    m.push(WeightEntry::layer(
                        "wv",
                        layer,
                        vec![kv_dim, d],
                        DType::F16,
                        ColumnShard { axis: 0 },
                    ));
                    m.push(WeightEntry::layer(
                        "wo",
                        layer,
                        vec![d, q_dim],
                        DType::F16,
                        RowShard { axis: 1 },
                    ));
                    m.push(WeightEntry::layer(
                        "q_norm",
                        layer,
                        vec![cfg.head_dim],
                        DType::F32,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "k_norm",
                        layer,
                        vec![cfg.head_dim],
                        DType::F32,
                        Replicate,
                    ));
                }
                LayerType::LinearAttention => {
                    // DeltaNet's packed QKV and its recurrent projections are
                    // replicated.  In particular, do not encode a guessed TP
                    // split in the manifest; qwen35_tp_preflight is the
                    // authoritative admission check until TP execution exists.
                    m.push(WeightEntry::layer(
                        "wqkv",
                        layer,
                        vec![qkv_dim, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wz",
                        layer,
                        vec![d_inner, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "w_alpha",
                        layer,
                        vec![cfg.linear_num_value_heads, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "w_beta",
                        layer,
                        vec![cfg.linear_num_value_heads, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer_with_dtype_constraint(
                        "a_log",
                        layer,
                        vec![cfg.linear_num_value_heads],
                        DType::F32,
                        raw_f32(),
                        Replicate,
                    ));
                    m.push(WeightEntry::layer_with_dtype_constraint(
                        "dt_bias",
                        layer,
                        vec![cfg.linear_num_value_heads],
                        DType::F32,
                        raw_f32(),
                        Replicate,
                    ));
                    m.push(WeightEntry::layer_with_dtype_constraint(
                        "conv",
                        layer,
                        vec![qkv_dim * cfg.conv_kernel_dim],
                        DType::F32,
                        raw_f32(),
                        Replicate,
                    ));
                    m.push(WeightEntry::layer_with_dtype_constraint(
                        "norm",
                        layer,
                        vec![cfg.linear_value_head_dim],
                        DType::F32,
                        raw_f32(),
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wo",
                        layer,
                        vec![d, d_inner],
                        DType::F16,
                        Replicate,
                    ));
                }
            }

            m.push(WeightEntry::layer(
                "ffn_norm",
                layer,
                vec![d],
                DType::F32,
                Replicate,
            ));

            if !is_moe {
                m.push(WeightEntry::layer(
                    "ffn_gate",
                    layer,
                    vec![cfg.hidden_dim, d],
                    DType::F16,
                    ColumnShard { axis: 0 },
                ));
                m.push(WeightEntry::layer(
                    "ffn_up",
                    layer,
                    vec![cfg.hidden_dim, d],
                    DType::F16,
                    ColumnShard { axis: 0 },
                ));
                m.push(WeightEntry::layer(
                    "ffn_down",
                    layer,
                    vec![d, cfg.hidden_dim],
                    DType::F16,
                    RowShard { axis: 1 },
                ));
            } else {
                m.push(WeightEntry::layer(
                    "router",
                    layer,
                    vec![cfg.num_experts, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "shared_gate",
                    layer,
                    vec![cfg.shared_expert_intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "shared_up",
                    layer,
                    vec![cfg.shared_expert_intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "shared_down",
                    layer,
                    vec![d, cfg.shared_expert_intermediate_size],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "shared_expert_gate",
                    layer,
                    vec![1, d],
                    DType::F16,
                    Replicate,
                ));
                // Keep gate_up and down separate because each
                // `ExpertWeights` owns exactly those two tensors.  There is
                // intentionally no packed 3-D surrogate here.
                for expert_idx in 0..cfg.num_experts {
                    m.push(WeightEntry::layer(
                        format!("expert.{expert_idx}.gate_up"),
                        layer,
                        vec![2 * cfg.moe_intermediate_size, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        format!("expert.{expert_idx}.down"),
                        layer,
                        vec![d, cfg.moe_intermediate_size],
                        DType::F16,
                        Replicate,
                    ));
                }
            }
        }

        m
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DeltaNetState::new(gpu, cfg)
            .map_err(|e| format!("qwen35: DeltaNetState::new failed: {e:?}"))
    }

    /// State manifest (device-mesh Phase 2): Qwen3.5/3.6 is a hybrid, so the
    /// per-layer state depends on the layer's [`LayerType`] (the LA-vs-full-attn
    /// knowledge the plan §4 wants living in manifest construction, so the
    /// `StateStore` can be keyed by *global* layer index and the DeltaNet
    /// `la_to_device` sidecar is defined out of existence):
    /// - `FullAttention` → a KV cache ([`StateKind::Kv`]).
    /// - `LinearAttention` (DeltaNet) → the recurrent S-matrix
    ///   ([`StateKind::Recurrent`]) **and** the short-conv1d state
    ///   ([`StateKind::Conv`]) — a DeltaNet layer holds both (see
    ///   `DeltaNetState::{recurrent…, conv_states}`).
    ///
    /// Keyed by global layer index; the quant mode is a load-time choice
    /// resolved at fulfillment (empty here).
    fn state_manifest(cfg: &Self::Config) -> Vec<StateEntry> {
        let mut m = Vec::with_capacity(cfg.layer_types.len());
        for (l, lt) in cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::FullAttention => m.push(StateEntry::new(
                    StateKind::Kv {
                        quant: String::new(),
                    },
                    l,
                )),
                LayerType::LinearAttention => {
                    m.push(StateEntry::new(StateKind::Recurrent, l));
                    m.push(StateEntry::new(StateKind::Conv, l));
                }
            }
        }
        m
    }

    /// MoE expert-group manifest (STEP-002 Task 9): one layer-local Single
    /// group per MoE layer with stable semantic identities matching the
    /// actual dispatch builder plans — router identity `softmax_topk` (the
    /// builders' softmax routing) and exact allowed execution membership
    /// `[indexed_quantized, grouped_quantized]`: indexed decode and prefill
    /// Path 0/1 build indexed steps, grouped prefill Path 2 builds the
    /// grouped scatter → gate-up → gate-up deinterleave → activation → down →
    /// inverse-permuting combine program. Both declared identities are
    /// admitted; the shared lowerer's concrete-steps guard refuses any
    /// program whose typed label mismatches its concrete protocol (a grouped
    /// chain claimed as indexed is an `ExecutionIdentityMismatch` even with
    /// both declared). Qwen35 has NO parallel expert-group admission: a
    /// TP/EP policy resolves zero groups here (the runtime then refuses any
    /// parallel MoE program construction), and the production decode entry
    /// refuses before constructing program parts.
    fn expert_group_manifest(
        cfg: &Self::Config,
        policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
    ) -> Vec<hipfire_runtime::weight_manifest::ExpertGroupSpec> {
        crate::qwen35::qwen35_moe_expert_group_specs(cfg, policy)
    }

    // Optional overrides default to the trait scaffold's Qwen3.5-flavored
    // baseline. Qwen3.5 IS the canonical arch the trait was designed
    // around, so no overrides needed here. Future arches (gemma4, llama)
    // will exercise the override surface.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(layer_types: &[&str]) -> Qwen35Config {
        config_with_tie(layer_types, true)
    }

    fn config_with_tie(layer_types: &[&str], tie_word_embeddings: bool) -> Qwen35Config {
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": layer_types.len(),
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "layer_types": layer_types,
            "tie_word_embeddings": tie_word_embeddings,
        });
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": inner}).to_string())
            .unwrap()
    }

    fn moe_config(layer_types: &[&str]) -> Qwen35Config {
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": layer_types.len(),
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "layer_types": layer_types,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 384,
        });
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": inner}).to_string())
            .unwrap()
    }

    fn entry<'a>(manifest: &'a [WeightEntry], name: &str, layer: usize) -> &'a WeightEntry {
        manifest
            .iter()
            .find(|e| e.name == name && e.layer == Some(layer))
            .unwrap_or_else(|| panic!("missing {name} at layer {layer}"))
    }

    fn entry_model<'a>(manifest: &'a [WeightEntry], name: &str) -> &'a WeightEntry {
        manifest
            .iter()
            .find(|e| e.name == name && e.layer.is_none())
            .unwrap_or_else(|| panic!("missing model entry {name}"))
    }

    fn metadata_config(outer_tie: Option<bool>, nested_tie: Option<bool>) -> Qwen35Config {
        let mut config = serde_json::json!({
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000,
        });
        if let Some(tie) = outer_tie {
            config["tie_word_embeddings"] = serde_json::json!(tie);
        }
        if nested_tie.is_some() {
            let mut text_config = config.clone();
            text_config
                .as_object_mut()
                .unwrap()
                .remove("tie_word_embeddings");
            if let Some(tie) = nested_tie {
                text_config["tie_word_embeddings"] = serde_json::json!(tie);
            }
            config["text_config"] = text_config;
        }
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": config}).to_string())
            .unwrap()
    }

    fn assert_entry(
        manifest: &[WeightEntry],
        name: &str,
        layer: usize,
        shape: Vec<usize>,
        dtype: DType,
        dtype_constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) {
        let e = entry(manifest, name, layer);
        assert_eq!(e.logical_shape, shape, "{name} shape");
        assert_eq!(e.dtype, dtype, "{name} dtype");
        assert_eq!(e.dtype_constraint, dtype_constraint, "{name} constraint");
        assert_eq!(e.policy, policy, "{name} policy");
    }

    fn assert_model_entry(
        manifest: &[WeightEntry],
        name: &str,
        shape: Vec<usize>,
        dtype: DType,
        dtype_constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) {
        let e = entry_model(manifest, name);
        assert_eq!(e.logical_shape, shape, "{name} shape");
        assert_eq!(e.dtype, dtype, "{name} dtype");
        assert_eq!(e.dtype_constraint, dtype_constraint, "{name} constraint");
        assert_eq!(e.policy, policy, "{name} policy");
    }

    fn expected_names(full_attention: bool, moe: bool) -> std::collections::BTreeSet<String> {
        let mut names = [
            "token_embd",
            "attn_norm",
            "ffn_norm",
            "output_norm",
            "lm_head",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
        names.extend(
            if full_attention {
                ["wq", "wk", "wv", "wo", "q_norm", "k_norm"]
            } else {
                ["wqkv", "wz", "w_alpha", "w_beta", "a_log", "dt_bias"]
            }
            .into_iter()
            .map(str::to_owned),
        );
        if full_attention {
            // DeltaNet-only raw names are intentionally absent from FA.
        } else {
            names.extend(["conv", "norm", "wo"].into_iter().map(str::to_owned));
        }
        if moe {
            names.extend(
                [
                    "router",
                    "shared_expert_gate",
                    "shared_gate",
                    "shared_up",
                    "shared_down",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            for expert in 0..3 {
                names.insert(format!("expert.{expert}.gate_up"));
                names.insert(format!("expert.{expert}.down"));
            }
        } else {
            names.extend(
                ["ffn_gate", "ffn_up", "ffn_down"]
                    .into_iter()
                    .map(str::to_owned),
            );
        }
        names
    }

    #[test]
    fn tp_preflight_allows_single_device() {
        qwen35_tp_preflight(&config(&["linear_attention"]), 1).unwrap();
    }

    #[test]
    fn tp_preflight_rejects_deltanet_from_config_alone() {
        let err = qwen35_tp_preflight(&config(&["linear_attention"]), 2).unwrap_err();
        assert_eq!(
            err,
            "qwen35: Tp=2 is unsupported for DeltaNet wqkv (layer 0)"
        );
    }

    #[test]
    fn tp_preflight_rejects_packed_qgate_from_config_alone() {
        let err = qwen35_tp_preflight(&config(&["full_attention"]), 2).unwrap_err();
        assert_eq!(
            err,
            "qwen35: Tp=2 is unsupported for packed QGate wq (layer 0)"
        );
    }

    #[test]
    fn tp_preflight_allows_empty_topology() {
        qwen35_tp_preflight(&config(&[]), 2).unwrap();
    }

    #[test]
    fn config_rejects_empty_layer_types_for_declared_layers() {
        let inner = serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 1000,
            "layer_types": [],
        });
        let err = crate::qwen35::config_from_metadata_json(
            &serde_json::json!({"config": inner}).to_string(),
        )
        .unwrap_err();
        assert!(err.contains("layer_types length 0 does not match num_hidden_layers 2"));
    }

    #[test]
    fn weight_manifest_uses_config_tie_policy_for_lm_head() {
        let tied = <Qwen35 as Architecture>::weight_manifest(&config(&["full_attention"]));
        assert!(matches!(
            entry_model(&tied, "lm_head").policy,
            ShardPolicy::Tied { ref source } if source == "token_embd"
        ));

        let untied_cfg = config_with_tie(&["full_attention"], false);
        let untied = <Qwen35 as Architecture>::weight_manifest(&untied_cfg);
        let lm_head = entry_model(&untied, "lm_head");
        assert_eq!(lm_head.policy, ShardPolicy::Pin(PinTarget::Output));
        assert_eq!(lm_head.logical_shape, vec![1000, 1024]);
    }

    #[test]
    fn manifest_payload_order_matches_legacy_orchestrator() {
        fn expected(layer_types: &[LayerType], experts: usize) -> Vec<(String, Option<usize>)> {
            let mut out = vec![
                ("token_embd".into(), None),
                ("output_norm".into(), None),
                ("lm_head".into(), None),
            ];
            for (layer, layer_type) in layer_types.iter().enumerate() {
                out.push(("attn_norm".into(), Some(layer)));
                match layer_type {
                    LayerType::LinearAttention => {
                        out.extend(
                            [
                                "wqkv", "wz", "w_alpha", "w_beta", "a_log", "dt_bias", "conv",
                                "norm", "wo",
                            ]
                            .into_iter()
                            .map(|name| (name.into(), Some(layer))),
                        );
                    }
                    LayerType::FullAttention => {
                        out.extend(
                            ["wq", "wk", "wv", "wo", "q_norm", "k_norm"]
                                .into_iter()
                                .map(|name| (name.into(), Some(layer))),
                        );
                    }
                }
                out.push(("ffn_norm".into(), Some(layer)));
                if experts == 0 {
                    out.extend(
                        ["ffn_gate", "ffn_up", "ffn_down"]
                            .into_iter()
                            .map(|name| (name.into(), Some(layer))),
                    );
                } else {
                    out.push(("router".into(), Some(layer)));
                    out.extend(
                        [
                            "shared_gate",
                            "shared_up",
                            "shared_down",
                            "shared_expert_gate",
                        ]
                        .into_iter()
                        .map(|name| (name.into(), Some(layer))),
                    );
                    for expert in 0..experts {
                        out.push((format!("expert.{expert}.gate_up"), Some(layer)));
                        out.push((format!("expert.{expert}.down"), Some(layer)));
                    }
                }
            }
            out
        }

        for (cfg, experts) in [
            (config(&["linear_attention", "full_attention"]), 0),
            (moe_config(&["linear_attention", "full_attention"]), 3),
        ] {
            let actual: Vec<_> = Qwen35::weight_manifest(&cfg)
                .into_iter()
                .map(|entry| (entry.name, entry.layer))
                .collect();
            assert_eq!(actual, expected(&cfg.layer_types, experts));
        }
    }

    #[test]
    fn raw_f32_constraint_lists_allowed_source_dtypes() {
        let manifest = <Qwen35 as Architecture>::weight_manifest(&config(&["linear_attention"]));
        let expected_sources = vec![
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::Q8_0,
            DType::MQ4G256,
            DType::ParoQ4G128,
        ];
        for name in ["a_log", "dt_bias", "conv", "norm"] {
            assert_eq!(
                entry(&manifest, name, 0).dtype_constraint,
                DTypeConstraint::source_from_sources(expected_sources.clone()),
                "{name}"
            );
        }
    }

    #[test]
    fn tied_lm_head_is_reuploaded_at_pp_output_placement() {
        let manifest = <Qwen35 as Architecture>::weight_manifest(&config(&["full_attention"]));
        let mesh = hipfire_hardware::DeviceMesh::rect(&[(hipfire_hardware::DimKind::Pp, 2)]);
        assert_eq!(
            hipfire_runtime::weight_manifest::placement_devices(
                entry_model(&manifest, "token_embd"),
                &mesh,
                1
            ),
            vec![0]
        );
        assert_eq!(
            hipfire_runtime::weight_manifest::placement_devices(
                entry_model(&manifest, "lm_head"),
                &mesh,
                1
            ),
            vec![1]
        );
    }

    #[test]
    fn tie_word_embeddings_follow_outer_then_nested_then_default_precedence() {
        assert!(metadata_config(Some(true), None).tie_word_embeddings);
        assert!(!metadata_config(Some(false), None).tie_word_embeddings);
        assert!(metadata_config(None, Some(true)).tie_word_embeddings);
        assert!(!metadata_config(None, Some(false)).tie_word_embeddings);
        assert!(!metadata_config(Some(false), Some(true)).tie_word_embeddings);
        assert!(metadata_config(None, None).tie_word_embeddings);
    }

    #[test]
    fn malformed_layer_types_are_rejected_and_mixed_topology_is_preserved() {
        let malformed = serde_json::json!({
            "config": {
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "vocab_size": 1000,
                "layer_types": ["full_attention"],
            }
        });
        let err = crate::qwen35::config_from_metadata_json(&malformed.to_string()).unwrap_err();
        assert!(err.contains("layer_types length 1 does not match num_hidden_layers 2"));

        let mixed = config(&["full_attention", "linear_attention", "full_attention"]);
        assert_eq!(mixed.layer_types.len(), mixed.n_layers);
        assert_eq!(mixed.layer_types[0], LayerType::FullAttention);
        assert_eq!(mixed.layer_types[1], LayerType::LinearAttention);
        assert_eq!(mixed.layer_types[2], LayerType::FullAttention);
    }

    #[test]
    fn weight_manifest_shapes_policies_and_constraints_are_exact_for_each_topology() {
        let cases = [
            (config(&["full_attention"]), true),
            (config(&["linear_attention"]), false),
            (moe_config(&["full_attention"]), true),
            (moe_config(&["linear_attention"]), false),
        ];
        for (cfg, full_attention) in cases {
            let manifest = <Qwen35 as Architecture>::weight_manifest(&cfg);
            let actual_names: std::collections::BTreeSet<_> =
                manifest.iter().map(|e| e.name.clone()).collect();
            assert_eq!(
                actual_names,
                expected_names(full_attention, cfg.num_experts > 0)
            );
            assert_model_entry(
                &manifest,
                "token_embd",
                vec![cfg.vocab_size, cfg.dim],
                DType::F16,
                DTypeConstraint::any_source(),
                ShardPolicy::Pin(PinTarget::Embed),
            );
            assert_model_entry(
                &manifest,
                "output_norm",
                vec![cfg.dim],
                DType::F32,
                DTypeConstraint::any_source(),
                ShardPolicy::Pin(PinTarget::Output),
            );
            assert_model_entry(
                &manifest,
                "lm_head",
                vec![cfg.vocab_size, cfg.dim],
                DType::F16,
                DTypeConstraint::any_source(),
                ShardPolicy::Tied {
                    source: "token_embd".into(),
                },
            );
            assert_entry(
                &manifest,
                "attn_norm",
                0,
                vec![cfg.dim],
                DType::F32,
                DTypeConstraint::any_source(),
                ShardPolicy::Replicate,
            );
            assert_entry(
                &manifest,
                "ffn_norm",
                0,
                vec![cfg.dim],
                DType::F32,
                DTypeConstraint::any_source(),
                ShardPolicy::Replicate,
            );

            if full_attention {
                let q_dim = cfg.n_heads * cfg.head_dim;
                let kv_dim = cfg.n_kv_heads * cfg.head_dim;
                assert_entry(
                    &manifest,
                    "wq",
                    0,
                    vec![2 * q_dim, cfg.dim],
                    DType::F16,
                    DTypeConstraint::any_source(),
                    ShardPolicy::Replicate,
                );
                for (name, shape) in [("wk", vec![kv_dim, cfg.dim]), ("wv", vec![kv_dim, cfg.dim])]
                {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        shape,
                        DType::F16,
                        DTypeConstraint::any_source(),
                        ShardPolicy::ColumnShard { axis: 0 },
                    );
                }
                assert_entry(
                    &manifest,
                    "wo",
                    0,
                    vec![cfg.dim, q_dim],
                    DType::F16,
                    DTypeConstraint::any_source(),
                    ShardPolicy::RowShard { axis: 1 },
                );
                for name in ["q_norm", "k_norm"] {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        vec![cfg.head_dim],
                        DType::F32,
                        DTypeConstraint::any_source(),
                        ShardPolicy::Replicate,
                    );
                }
            } else {
                let qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                let d_inner = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                assert_entry(
                    &manifest,
                    "wqkv",
                    0,
                    vec![qkv_dim, cfg.dim],
                    DType::F16,
                    DTypeConstraint::any_source(),
                    ShardPolicy::Replicate,
                );
                for (name, shape) in [
                    ("wz", vec![d_inner, cfg.dim]),
                    ("w_alpha", vec![cfg.linear_num_value_heads, cfg.dim]),
                    ("w_beta", vec![cfg.linear_num_value_heads, cfg.dim]),
                    ("wo", vec![cfg.dim, d_inner]),
                ] {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        shape,
                        DType::F16,
                        DTypeConstraint::any_source(),
                        ShardPolicy::Replicate,
                    );
                }
                for (name, shape) in [
                    ("a_log", vec![cfg.linear_num_value_heads]),
                    ("dt_bias", vec![cfg.linear_num_value_heads]),
                    ("conv", vec![qkv_dim * cfg.conv_kernel_dim]),
                    ("norm", vec![cfg.linear_value_head_dim]),
                ] {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        shape,
                        DType::F32,
                        DTypeConstraint::source_from_sources(vec![
                            DType::F16,
                            DType::BF16,
                            DType::F32,
                            DType::Q8_0,
                            DType::MQ4G256,
                            DType::ParoQ4G128,
                        ]),
                        ShardPolicy::Replicate,
                    );
                }
            }

            if cfg.num_experts == 0 {
                for (name, shape, policy) in [
                    (
                        "ffn_gate",
                        vec![cfg.hidden_dim, cfg.dim],
                        ShardPolicy::ColumnShard { axis: 0 },
                    ),
                    (
                        "ffn_up",
                        vec![cfg.hidden_dim, cfg.dim],
                        ShardPolicy::ColumnShard { axis: 0 },
                    ),
                    (
                        "ffn_down",
                        vec![cfg.dim, cfg.hidden_dim],
                        ShardPolicy::RowShard { axis: 1 },
                    ),
                ] {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        shape,
                        DType::F16,
                        DTypeConstraint::any_source(),
                        policy,
                    );
                }
                assert!(manifest.iter().all(|e| !e.name.starts_with("expert.")));
                assert!(manifest.iter().all(|e| e.name != "router"));
            } else {
                for (name, shape) in [
                    ("router", vec![cfg.num_experts, cfg.dim]),
                    ("shared_expert_gate", vec![1, cfg.dim]),
                    (
                        "shared_gate",
                        vec![cfg.shared_expert_intermediate_size, cfg.dim],
                    ),
                    (
                        "shared_up",
                        vec![cfg.shared_expert_intermediate_size, cfg.dim],
                    ),
                    (
                        "shared_down",
                        vec![cfg.dim, cfg.shared_expert_intermediate_size],
                    ),
                ] {
                    assert_entry(
                        &manifest,
                        name,
                        0,
                        shape,
                        DType::F16,
                        DTypeConstraint::any_source(),
                        ShardPolicy::Replicate,
                    );
                }
                for expert in 0..cfg.num_experts {
                    assert_entry(
                        &manifest,
                        &format!("expert.{expert}.gate_up"),
                        0,
                        vec![2 * cfg.moe_intermediate_size, cfg.dim],
                        DType::F16,
                        DTypeConstraint::any_source(),
                        ShardPolicy::Replicate,
                    );
                    assert_entry(
                        &manifest,
                        &format!("expert.{expert}.down"),
                        0,
                        vec![cfg.dim, cfg.moe_intermediate_size],
                        DType::F16,
                        DTypeConstraint::any_source(),
                        ShardPolicy::Replicate,
                    );
                }
                for name in ["ffn_gate", "ffn_up", "ffn_down"] {
                    assert!(manifest.iter().all(|e| e.name != name), "{name}");
                }
            }
        }
    }

    #[test]
    fn qwen35_weight_manifest_does_not_change_global_state_indices() {
        let cfg = config(&[
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]);
        let state = <Qwen35 as Architecture>::state_manifest(&cfg);
        assert_eq!(state.len(), 6);
        assert_eq!(
            state[0],
            StateEntry::new(
                StateKind::Kv {
                    quant: String::new()
                },
                0
            )
        );
        assert_eq!(state[1], StateEntry::new(StateKind::Recurrent, 1));
        assert_eq!(state[2], StateEntry::new(StateKind::Conv, 1));
        assert_eq!(
            state[3],
            StateEntry::new(
                StateKind::Kv {
                    quant: String::new()
                },
                2
            )
        );
        assert_eq!(state[4], StateEntry::new(StateKind::Recurrent, 3));
        assert_eq!(state[5], StateEntry::new(StateKind::Conv, 3));
    }

    #[test]
    #[ignore] // GPU: HIPFIRE_EMULATE_GPUS=2 (or 2 real devices); see pp_parity.rs
    fn qwen35_la_devices_matches_mesh_placement() {
        use hipfire_runtime::multi_gpu::Gpus;
        // 5 layers, 2 devices → uniform split [3, 2]:
        //   layers 0-2 → dev 0, layers 3-4 → dev 1.
        let cfg = config(&[
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
        ]);
        let mut gpus = Gpus::init_uniform(2, cfg.n_layers).expect("init_uniform");
        let map = qwen35_la_devices(&cfg, &gpus);
        // LA layers (global 1, 3, 4) → compact 0, 1, 2.
        let expected: Vec<u8> = [1usize, 3, 4]
            .iter()
            .map(|&l| gpus.device_for_layer(l) as u8)
            .collect();
        assert_eq!(
            map, expected,
            "manifest-derived LA map must equal mesh placement"
        );
        assert_eq!(map.len(), 3, "one entry per LinearAttention layer");
        // Sanity: the band split really is [3, 2] on a 5-layer/2-device mesh.
        assert_eq!(gpus.layer_to_device[0], 0);
        assert_eq!(gpus.layer_to_device[2], 0);
        assert_eq!(gpus.layer_to_device[3], 1);
        assert_eq!(gpus.layer_to_device[4], 1);
        // The derived map is what allocation uses: rebuild the state and check
        // the LA layer S matrices landed on the manifest-derived device.
        // Under HIPFIRE_EMULATE_GPUS the logical ranks alias one physical
        // device, so pointer_get_attributes cannot distinguish them; assert
        // placement only when the physical devices are distinct.
        let distinct_physical = (0..gpus.devices.len()).all(|a| {
            (0..gpus.devices.len())
                .all(|b| a == b || gpus.devices[a].device_id != gpus.devices[b].device_id)
        });
        let (dn, _la_to_device) = crate::qwen35::DeltaNetState::new_with_quant_multi(
            &mut gpus,
            &cfg,
            crate::qwen35::StateQuant::Q8,
        )
        .expect("dn multi");
        if distinct_physical {
            for (i, &dev) in map.iter().enumerate() {
                let attr = gpus.devices[dev as usize]
                    .hip
                    .pointer_get_attributes(&dn.s_matrices[i].buf)
                    .expect("s_matrix attrs");
                assert_eq!(
                    attr.device, dev as i32,
                    "DeltaNetState s_matrices[{i}] must live on its manifest-derived device"
                );
            }
        }
        dn.free_gpu_multi(&mut gpus, &_la_to_device);
    }

    // ── STEP-002 Task 9: Single expert-group manifest ─────────────────────

    fn single_policy() -> hipfire_runtime::moe_plan::MoEExecutionPolicy {
        hipfire_runtime::moe_plan::MoEExecutionPolicy::single()
    }

    fn tp_policy(ranks: usize) -> hipfire_runtime::moe_plan::MoEExecutionPolicy {
        hipfire_runtime::moe_plan::MoEExecutionPolicy::new(
            hipfire_runtime::moe_plan::MoEExecutionKind::Tp,
            hipfire_hardware::DeviceMesh::rect(&[(hipfire_hardware::DimKind::Tp, ranks)]),
        )
        .unwrap()
    }

    fn ep_policy(ranks: usize) -> hipfire_runtime::moe_plan::MoEExecutionPolicy {
        hipfire_runtime::moe_plan::MoEExecutionPolicy::new(
            hipfire_runtime::moe_plan::MoEExecutionKind::Ep,
            hipfire_hardware::DeviceMesh::rect(&[(hipfire_hardware::DimKind::Ep, ranks)]),
        )
        .unwrap()
    }

    /// The production Qwen manifest resolves exactly one Single group per MoE
    /// layer, and only under the Single policy.
    #[test]
    fn qwen35_expert_group_manifest_resolves_only_with_single_policy() {
        let cfg = moe_config(&["linear_attention", "full_attention"]);
        let specs = <Qwen35 as Architecture>::expert_group_manifest(&cfg, &single_policy());
        assert_eq!(specs.len(), 2, "one group per MoE layer");
        for (layer, spec) in specs.iter().enumerate() {
            assert_eq!(spec.layer, Some(layer));
            assert_eq!(spec.n_experts, cfg.num_experts);
            assert_eq!(
                spec.parallelism,
                hipfire_runtime::weight_manifest::ExpertParallelism::Single
            );
            assert_eq!(spec.group, format!("qwen35_moe_layer_{layer}"));
        }
    }

    /// TP/EP policies must resolve no Qwen expert groups at all — the refusal
    /// happens before any program construction.
    #[test]
    fn qwen35_expert_group_manifest_refuses_tp_ep_before_program_construction() {
        let cfg = moe_config(&["full_attention"]);
        assert!(
            <Qwen35 as Architecture>::expert_group_manifest(&cfg, &tp_policy(2)).is_empty(),
            "TP policy must resolve no Qwen expert groups"
        );
        assert!(
            <Qwen35 as Architecture>::expert_group_manifest(&cfg, &ep_policy(2)).is_empty(),
            "EP policy must resolve no Qwen expert groups"
        );
    }

    /// The manifest identities must exactly match the plans the dispatch
    /// decode builder actually builds: softmax top-K routing + indexed
    /// quantized execution, with the per-expert manifest sources claimed by
    /// each group.
    #[test]
    fn qwen35_expert_group_manifest_identities_match_dispatch_builder_plans() {
        let cfg = moe_config(&["full_attention"]);
        let specs = <Qwen35 as Architecture>::expert_group_manifest(&cfg, &single_policy());
        assert_eq!(specs.len(), 1);
        let spec = &specs[0];
        assert_eq!(spec.router, "router");
        assert_eq!(spec.router_identity, "softmax_topk");
        // Qwen declares exactly indexed+grouped execution membership (the
        // CPU fallback is never declared — it lives outside lowering).
        assert_eq!(
            spec.allowed_executions,
            vec![
                hipfire_runtime::weight_manifest::ExpertExecutionIdentity::IndexedQuantized,
                hipfire_runtime::weight_manifest::ExpertExecutionIdentity::GroupedQuantized,
            ]
        );
        assert_eq!(
            spec.source_layout,
            hipfire_runtime::weight_manifest::ExpertSourceLayout::PerExpertFused {
                gate_up: (0..cfg.num_experts)
                    .map(|e| format!("expert.{e}.gate_up"))
                    .collect(),
                down: (0..cfg.num_experts)
                    .map(|e| format!("expert.{e}.down"))
                    .collect(),
                sidecars: Vec::new(),
            }
        );
    }

    /// Router-contract pin (re-opened by Lane R): the runtime's expert-group
    /// reference validation now requires the router entry's FIRST dimension
    /// to equal n_experts, and the Qwen35 router is physically
    /// `[n_experts, dim]` (declared in the weight manifest) — so resolution
    /// succeeds. A transposed `[dim, n_experts]` router stays refused by the
    /// read-only contract; this test pins both directions so the lane cannot
    /// drift silently.
    #[test]
    fn qwen35_expert_group_manifest_resolves_with_experts_first_router() {
        let cfg = moe_config(&["full_attention"]);
        let specs = <Qwen35 as Architecture>::expert_group_manifest(&cfg, &single_policy());
        assert_eq!(specs.len(), 1);
        let manifest = <Qwen35 as Architecture>::weight_manifest(&cfg);
        let plan =
            hipfire_runtime::weight_manifest::resolve_expert_group_plan(&specs[0], &manifest, 1)
                .expect("experts-first router must resolve");
        assert_eq!(plan.n_experts, cfg.num_experts);
        assert_eq!(plan.group_size, 1);

        // Transposed `[dim, n_experts]` must be refused by the first-axis
        // contract with a deterministic router-shape error.
        let mut transposed = manifest;
        transposed
            .iter_mut()
            .find(|entry| entry.name == "router")
            .unwrap()
            .logical_shape = vec![cfg.dim, cfg.num_experts];
        let err =
            hipfire_runtime::weight_manifest::resolve_expert_group_plan(&specs[0], &transposed, 1)
                .unwrap_err();
        assert!(
            err.contains("router reference 'router'"),
            "refusal must name the router reference, got: {err}"
        );
        assert!(
            err.contains("first dimension must equal n_experts"),
            "refusal must be the router-shape contract, got: {err}"
        );
    }

    /// Dense (non-MoE) Qwen configs declare no expert groups at all.
    #[test]
    fn qwen35_dense_config_declares_no_expert_groups() {
        let cfg = config(&["full_attention"]);
        assert!(<Qwen35 as Architecture>::expert_group_manifest(&cfg, &single_policy()).is_empty());
    }
}
