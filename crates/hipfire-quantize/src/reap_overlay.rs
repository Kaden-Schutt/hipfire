//! SP4: selective re-quant overlay builder. See
//! docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md §3.
//!
//! Two pure-ish units shared by the overlay (and, later, bake) flow:
//!   * `quantize_to_format` — tier-name → existing `quantize_*` encoder dispatch.
//!   * `reap_override_for`   — arch-aware tensor-name → override-tier resolver.

use crate::{
    gen_fwht_signs, quantize_hfq4g256, quantize_hfq6g256, quantize_mq2g256_lloyd,
    quantize_mq3g256_lloyd, quantize_mq4g256, quantize_mq4g256_lloyd, quantize_mq6g256,
    quantize_q8f16, quantize_rwq4g256, HfqTensor, QuantType,
};
use hipfire_reap::plan::{QuantOverride, ReapPlan, Role};

/// Quantize one tensor's f32 data to the named tier, returning the HFQ tensor.
/// Covers the self-calibrating tiers usable in an overlay without an imatrix.
/// `shape` is the row-major tensor shape (e.g. `[rows, cols]`).
///
/// The FWHT-rotated tiers (mq*/lloyd) use the canonical sign seeds 42 / 1042 —
/// the same seeds the monolithic quantize loop uses (`main.rs` `gen_fwht_signs(42|1042, 256)`),
/// so the bytes produced here are byte-identical to `--format <tier>`.
pub fn quantize_to_format(
    name: &str,
    fmt: &str,
    f32_data: &[f32],
    shape: &[usize],
) -> Result<HfqTensor, String> {
    let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
    // Canonical FWHT sign tables (only built for the rotated tiers).
    let signs = || (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
    let (qt, gs, data) = match fmt {
        "q8" | "q8f16" => (QuantType::Q8F16, 32u32, quantize_q8f16(f32_data)),
        "hfq4" | "hfq4g256" => (QuantType::HFQ4G256, 256, quantize_hfq4g256(f32_data)),
        "hfq6" | "hfq6g256" => (QuantType::HFQ6G256, 256, quantize_hfq6g256(f32_data)),
        "mq4" | "mq4g256" => {
            let (s1, s2) = signs();
            (
                QuantType::MQ4G256,
                256,
                quantize_mq4g256(f32_data, &s1, &s2),
            )
        }
        "rwq4" | "rwq4g256" | "radiowave" => {
            let (s1, s2) = signs();
            (
                QuantType::RWQ4G256,
                256,
                quantize_rwq4g256(f32_data, &s1, &s2),
            )
        }
        "mq6" | "mq6g256" => {
            let (s1, s2) = signs();
            (
                QuantType::MQ6G256,
                256,
                quantize_mq6g256(f32_data, &s1, &s2),
            )
        }
        "mq2lloyd" | "mq2g256lloyd" => {
            let (s1, s2) = signs();
            (
                QuantType::MQ2G256Lloyd,
                256,
                quantize_mq2g256_lloyd(f32_data, &s1, &s2),
            )
        }
        "mq3lloyd" | "mq3g256lloyd" => {
            let (s1, s2) = signs();
            (
                QuantType::MQ3G256Lloyd,
                256,
                quantize_mq3g256_lloyd(f32_data, &s1, &s2),
            )
        }
        "mq4lloyd" | "mq4g256lloyd" => {
            let (s1, s2) = signs();
            (
                QuantType::MQ4G256Lloyd,
                256,
                quantize_mq4g256_lloyd(f32_data, &s1, &s2),
            )
        }
        other => {
            return Err(format!(
                "reap: unsupported overlay tier '{other}' for {name}"
            ))
        }
    };
    Ok(HfqTensor {
        name: name.to_string(),
        quant_type: qt,
        shape: shape_u32,
        group_size: gs,
        data,
        spilled_len: 0,
    })
}

/// Detected arch family for tensor-name matching (the quantizer already knows
/// the arch_id; pass the matching variant in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReapArch {
    Deepseek4,
    Qwen35,
    Lfm2Moe,
    Minimax,
}

impl ReapArch {
    /// Parse the `--reap-arch <name>` flag value.
    pub fn from_flag(s: &str) -> Result<ReapArch, String> {
        match s {
            "deepseek4" | "deepseek_v4" | "ds4" => Ok(ReapArch::Deepseek4),
            "qwen35" | "qwen3_5_moe" | "qwen3.5" => Ok(ReapArch::Qwen35),
            "lfm2moe" | "lfm2_moe" => Ok(ReapArch::Lfm2Moe),
            "minimax" => Ok(ReapArch::Minimax),
            other => Err(format!(
                "reap: unknown --reap-arch '{other}' (expected deepseek4|qwen35|lfm2moe|minimax)"
            )),
        }
    }

    /// Best-effort auto-detection from the quantizer's resolved `arch_id`.
    /// Minimax has no quantizer arch_id, so it is only reachable via the
    /// explicit `--reap-arch minimax` flag.
    pub fn from_arch_id(arch_id: u32) -> Option<ReapArch> {
        match arch_id {
            9 => Some(ReapArch::Deepseek4),
            6 => Some(ReapArch::Qwen35),
            11 => Some(ReapArch::Lfm2Moe),
            _ => None,
        }
    }
}

/// Overlay driver: quantize ONLY the tensors the plan overrides.
///
/// `tensors` yields `(name, shape, f32_data)` for the model's tensors. The
/// caller is responsible for skipping the (expensive) f32 decode of tensors
/// that don't match the plan — `build_overlay` only quantizes what it's
/// handed (and re-checks the match so a hand-built iterator can over-supply).
/// Returns the subset of `HfqTensor`s to write into `overlay.hfq`.
///
/// `main.rs`'s overlay branch inlines this logic (to skip the f32 decode of
/// non-matched tensors); this standalone driver is exercised by the
/// `tests/reap_overlay_integ.rs` integration target, hence the allow.
#[allow(dead_code)]
pub fn build_overlay(
    arch: ReapArch,
    plan: &ReapPlan,
    tensors: impl Iterator<Item = (String, Vec<usize>, Vec<f32>)>,
) -> Result<Vec<HfqTensor>, String> {
    let mut out = Vec::new();
    for (name, shape, f32) in tensors {
        if let Some(fmt) = reap_override_for(&name, arch, plan) {
            out.push(quantize_to_format(&name, fmt, &f32, &shape)?);
        }
    }
    Ok(out)
}

/// Bake driver: quantize EVERY tensor, applying the plan's per-tensor override
/// tier where one matches and falling back to `base_fmt` otherwise.
///
/// `tensors` yields `(name, shape, f32_data)` for the model's tensors. Unlike
/// `build_overlay` (subset only), bake emits the whole model. Pruning is Task 2
/// — this version emits every tensor handed to it.
///
/// `main.rs`'s bake path inlines the override hook into the existing whole-model
/// quantize loop (so non-overridden tensors keep their arch-specific default
/// quant); this standalone helper is exercised by the in-module bake tests,
/// hence the allow.
#[allow(dead_code)]
pub fn bake_tensors(
    arch: ReapArch,
    plan: &ReapPlan,
    base_fmt: &str,
    tensors: impl Iterator<Item = (String, Vec<usize>, Vec<f32>)>,
) -> Result<Vec<HfqTensor>, String> {
    let mut out = Vec::new();
    for (name, shape, f32) in tensors {
        let fmt = reap_override_for(&name, arch, plan).unwrap_or(base_fmt);
        out.push(quantize_to_format(&name, fmt, &f32, &shape)?);
    }
    Ok(out)
}

/// Resolve a tensor name to its override tier under `plan`, or None.
/// Matches by (layer, role, [expert]) using the arch's tensor naming.
pub fn reap_override_for<'a>(name: &str, arch: ReapArch, plan: &'a ReapPlan) -> Option<&'a str> {
    for ov in &plan.quant_overrides {
        if tensor_matches(name, arch, ov) {
            return Some(ov.tier.as_str());
        }
    }
    None
}

/// The per-arch routed-expert name segment and the weight-suffix predicate that
/// together identify a routed-expert tensor. Shared by `tensor_matches` and
/// `expert_index_of` (DRY).
fn routed_expert_seg(arch: ReapArch) -> (&'static str, fn(&str) -> bool) {
    match arch {
        ReapArch::Deepseek4 => (".ffn.experts.", |n| {
            n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")
        }),
        ReapArch::Qwen35 => (".mlp.experts.", |n| {
            n.ends_with(".gate_up_proj.weight") || n.ends_with(".down_proj.weight")
        }),
        ReapArch::Lfm2Moe => (".feed_forward.experts.", |n| {
            n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")
        }),
        ReapArch::Minimax => (".block_sparse_moe.experts.", |n| {
            n.ends_with(".w1.weight") || n.ends_with(".w2.weight") || n.ends_with(".w3.weight")
        }),
    }
}

/// Parse the routed-expert index `E` out of a per-expert tensor name for `arch`,
/// or `None` if the name is not a routed-expert weight tensor. The index is the
/// first dotted token after the arch's expert segment (e.g.
/// `layers.0.ffn.experts.7.w1.weight` → `7`). Shared by `tensor_matches` (override
/// resolution) and `bake_expert_rename` (prune/renumber) so both agree on layout.
///
/// NOTE: Qwen3.5-MoE ships its routed experts as a single 3-D stacked tensor
/// (`...mlp.experts.gate_up_proj`, no per-expert index in the *source* name) that
/// the quantizer splits into per-expert names at quant time. This parser matches
/// the *split* form (`...mlp.experts.{E}.gate_up_proj.weight`); the bake's prune
/// of the stacked source happens in the split loop, not via this fn.
pub fn expert_index_of(name: &str, arch: ReapArch) -> Option<u32> {
    let (seg, w_ok) = routed_expert_seg(arch);
    if !name.contains(seg) || !w_ok(name) {
        return None;
    }
    let after = &name[name.find(seg).unwrap() + seg.len()..];
    after.split('.').next().and_then(|s| s.parse().ok())
}

/// For a routed-expert tensor under an active keep, decide: drop it, or emit it
/// renamed to its compact slot. Returns `None` to drop (the expert was pruned),
/// `Some(new_name)` to keep (the original expert-index token replaced by its
/// compact slot index). `keep_l` is `keep[layer]` — original expert indices in
/// compact-slot order. Non-routed-expert names (no parseable index) return `None`;
/// callers gate this fn on routed-expert tensors only.
pub fn bake_expert_rename(
    name: &str,
    arch: ReapArch,
    _layer: usize,
    keep_l: &[u32],
) -> Option<String> {
    let e = expert_index_of(name, arch)?;
    let slot = keep_l.iter().position(|&k| k == e)?;
    // Replace the expert-index token `.{seg}{E}.` with `.{seg}{slot}.`. The
    // segment ends with `.`, and the token is immediately followed by `.`, so a
    // bounded replace of `{seg}{E}.` → `{seg}{slot}.` is unambiguous.
    let (seg, _) = routed_expert_seg(arch);
    let from = format!("{seg}{e}.");
    let to = format!("{seg}{slot}.");
    Some(name.replacen(&from, &to, 1))
}

/// Does `name` name the MoE router weight (`[orig_experts, hidden]`) for `arch`?
/// Used by the bake prune to row-gather the router to the kept expert set. The
/// caller additionally checks the tensor's shape[0] == original_experts so a
/// shared-expert gate (`[1, hidden]`) can never be mistaken for the router.
pub fn is_reap_router_weight(name: &str, arch: ReapArch) -> bool {
    match arch {
        ReapArch::Deepseek4 => name.ends_with(".ffn.gate.weight"),
        ReapArch::Qwen35 => name.ends_with(".mlp.gate.weight"),
        ReapArch::Lfm2Moe => name.ends_with(".feed_forward.gate.weight"),
        ReapArch::Minimax => name.ends_with(".block_sparse_moe.gate.weight"),
    }
}

/// Does `name` name the per-expert routing bias (`[orig_experts]`) for `arch`?
/// Row-gathered alongside the router under a keep so the bias aligns with the
/// gathered logits in compact-slot order.
pub fn is_reap_expert_bias(name: &str, arch: ReapArch) -> bool {
    match arch {
        ReapArch::Deepseek4 => name.ends_with(".ffn.gate.bias"),
        // Qwen3.5-MoE routing has no per-expert correction bias.
        ReapArch::Qwen35 => false,
        ReapArch::Lfm2Moe => name.ends_with(".feed_forward.expert_bias"),
        ReapArch::Minimax => name.ends_with(".block_sparse_moe.e_score_correction_bias"),
    }
}

/// Parse the layer index `L` from a tensor name's `layers.{L}.` segment, or
/// `None` if absent. All four arches embed the layer this way (with or without a
/// `model.`/`model.language_model.` prefix). Uses `rfind` so a `.layers.` deeper
/// in the path (none expected) prefers the last occurrence.
pub fn bake_layer_of(name: &str) -> Option<usize> {
    let marker = "layers.";
    let i = name.rfind(marker)?;
    let rest = &name[i + marker.len()..];
    rest.split('.').next().and_then(|s| s.parse().ok())
}

fn tensor_matches(name: &str, arch: ReapArch, ov: &QuantOverride) -> bool {
    // Layer gate: the name must reference `ov.layer`. All four arches embed the
    // layer index as `.layers.{L}.` or `layers.{L}.`.
    let layer_tok = format!("layers.{}.", ov.layer);
    if !name.contains(&layer_tok) {
        return false;
    }
    match ov.role {
        Role::RoutedExperts => {
            let (seg, w_ok) = routed_expert_seg(arch);
            if !name.contains(seg) || !w_ok(name) {
                return false;
            }
            if ov.experts.is_empty() {
                return true; // whole role at this layer
            }
            // expert index via the shared parser (verified against
            // main.rs — `layers.L.ffn.experts.E.{w1,w2,w3}.weight`).
            let eidx = match expert_index_of(name, arch) {
                Some(e) => e,
                None => return false,
            };
            ov.experts.contains(&eidx)
        }
        Role::Attention => {
            name.contains(".self_attn.") || name.contains(".attn.") || name.contains(".attention.")
        }
        Role::Router => {
            name.contains(".gate.weight")
                || name.contains(".router")
                || name.contains(".gate.tid2eid")
        }
        Role::SharedExpert => name.contains(".shared_expert") || name.contains(".shared_experts"),
        Role::LmHead => name.contains("lm_head") || name.contains("output.weight"),
        Role::Embed => name.contains("embed_tokens") || name.contains("tok_embeddings"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_underlying_encoder_byte_for_byte() {
        // 512 f32 (two 256-groups) of varied values.
        let f32: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let direct = quantize_hfq4g256(&f32);
        let t = quantize_to_format("x", "hfq4g256", &f32, &[2, 256]).unwrap();
        assert_eq!(t.data, direct, "overlay encode must equal direct encode");
        assert_eq!(t.shape, vec![2u32, 256]);
        assert_eq!(t.quant_type, QuantType::HFQ4G256);
    }

    #[test]
    fn mq4_matches_with_canonical_signs() {
        let f32: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let direct = quantize_mq4g256(&f32, &s1, &s2);
        let t = quantize_to_format("x", "mq4", &f32, &[1, 256]).unwrap();
        assert_eq!(t.data, direct);
    }

    #[test]
    fn mq3lloyd_matches_with_canonical_signs() {
        let f32: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.21).cos() * 7.0).collect();
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let direct = quantize_mq3g256_lloyd(&f32, &s1, &s2);
        let t = quantize_to_format("x", "mq3lloyd", &f32, &[1, 256]).unwrap();
        assert_eq!(t.data, direct);
    }

    #[test]
    fn rejects_unknown_tier() {
        let err = quantize_to_format("x", "bogus", &[0.0; 256], &[1, 256]).unwrap_err();
        assert!(
            err.contains("unsupported overlay tier 'bogus'"),
            "got: {err}"
        );
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    fn plan_with(json: &str) -> ReapPlan {
        // write to a tempdir & load_unchecked
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("reap_plan.json"), json).unwrap();
        ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn ds4_specific_experts() {
        let p = plan_with(
            r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7],"tier":"mq3lloyd"}]}"#,
        );
        assert_eq!(
            reap_override_for("layers.20.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p),
            Some("mq3lloyd")
        );
        assert_eq!(
            reap_override_for("layers.20.ffn.experts.8.w1.weight", ReapArch::Deepseek4, &p),
            None
        ); // wrong expert
        assert_eq!(
            reap_override_for("layers.21.ffn.experts.7.w1.weight", ReapArch::Deepseek4, &p),
            None
        ); // wrong layer
    }

    #[test]
    fn qwen35_whole_role() {
        let p = plan_with(
            r#"{"original_experts":128,"num_layers":48,
            "quant_overrides":[{"layer":5,"role":"routed_experts","tier":"hfq6"}]}"#,
        );
        assert_eq!(
            reap_override_for(
                "model.layers.5.mlp.experts.99.gate_up_proj.weight",
                ReapArch::Qwen35,
                &p
            ),
            Some("hfq6")
        );
        assert_eq!(
            reap_override_for(
                "model.layers.5.mlp.experts.99.down_proj.weight",
                ReapArch::Qwen35,
                &p
            ),
            Some("hfq6")
        );
        assert_eq!(
            reap_override_for(
                "model.layers.5.self_attn.q_proj.weight",
                ReapArch::Qwen35,
                &p
            ),
            None
        );
    }

    #[test]
    fn layer_token_no_false_positive_2_vs_20() {
        let p = plan_with(
            r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":2,"role":"routed_experts","experts":[0],"tier":"q8"}]}"#,
        );
        // layer-20 tensor must NOT match a layer-2 override
        assert_eq!(
            reap_override_for("layers.20.ffn.experts.0.w1.weight", ReapArch::Deepseek4, &p),
            None
        );
        // layer-2 does match
        assert_eq!(
            reap_override_for("layers.2.ffn.experts.0.w1.weight", ReapArch::Deepseek4, &p),
            Some("q8")
        );
    }

    #[test]
    fn attention_role() {
        let p = plan_with(
            r#"{"original_experts":256,"num_layers":43,
            "quant_overrides":[{"layer":41,"role":"attention","tier":"q8"}]}"#,
        );
        assert_eq!(
            reap_override_for(
                "model.layers.41.self_attn.q_proj.weight",
                ReapArch::Qwen35,
                &p
            ),
            Some("q8")
        );
        assert_eq!(
            reap_override_for(
                "model.layers.40.self_attn.q_proj.weight",
                ReapArch::Qwen35,
                &p
            ),
            None
        );
    }
}

#[cfg(test)]
mod bake_tests {
    use super::*;

    fn plan_with(json: &str) -> ReapPlan {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("reap_plan.json"), json).unwrap();
        ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn no_override_bakes_all_at_base_fmt() {
        // Empty quant_overrides → every tensor falls back to base_fmt (q8).
        let p = plan_with(r#"{"original_experts":128,"num_layers":48,"quant_overrides":[]}"#);
        let data = vec![0.3f32; 256];
        let tensors = vec![(
            "layers.0.self_attn.q_proj.weight".to_string(),
            vec![1usize, 256],
            data.clone(),
        )];
        let out = bake_tensors(ReapArch::Qwen35, &p, "q8", tensors.into_iter()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].quant_type, QuantType::Q8F16);
        // Bytes equal a direct q8 of the same data (anchor: no-override bake == normal quantize).
        let direct = quantize_to_format("x", "q8", &data, &[1, 256]).unwrap();
        assert_eq!(out[0].data, direct.data);
    }

    #[test]
    fn override_wins_over_base_fmt() {
        // Layer-0 attention override → hfq6, beating the q8 base_fmt.
        let p = plan_with(
            r#"{"original_experts":128,"num_layers":48,
            "quant_overrides":[{"layer":0,"role":"attention","tier":"hfq6"}]}"#,
        );
        let tensors = vec![(
            "layers.0.self_attn.q_proj.weight".to_string(),
            vec![2usize, 256],
            vec![0.1f32; 512],
        )];
        let out = bake_tensors(ReapArch::Qwen35, &p, "q8", tensors.into_iter()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].quant_type, QuantType::HFQ6G256); // override, not base q8
    }
}

#[cfg(test)]
mod prune_tests {
    use super::*;

    #[test]
    fn renames_kept_expert_to_compact_slot() {
        // keep[0] = [0,2,3]; original expert 2 → compact slot 1.
        let n = "layers.0.ffn.experts.2.w1.weight";
        assert_eq!(
            bake_expert_rename(n, ReapArch::Deepseek4, 0, &[0, 2, 3]),
            Some("layers.0.ffn.experts.1.w1.weight".to_string())
        );
    }

    #[test]
    fn drops_pruned_expert() {
        let n = "layers.0.ffn.experts.5.w1.weight";
        assert_eq!(
            bake_expert_rename(n, ReapArch::Deepseek4, 0, &[0, 2, 3]),
            None
        );
    }

    #[test]
    fn rename_keeps_slot0_identity() {
        // expert 0 stays slot 0 (identity rename, still Some).
        let n = "layers.7.ffn.experts.0.w2.weight";
        assert_eq!(
            bake_expert_rename(n, ReapArch::Deepseek4, 7, &[0, 4, 9]),
            Some("layers.7.ffn.experts.0.w2.weight".to_string())
        );
    }

    #[test]
    fn expert_index_parses_per_arch() {
        assert_eq!(
            expert_index_of("layers.3.ffn.experts.12.w1.weight", ReapArch::Deepseek4),
            Some(12)
        );
        assert_eq!(
            expert_index_of(
                "model.layers.3.mlp.experts.4.gate_up_proj.weight",
                ReapArch::Qwen35
            ),
            Some(4)
        );
        assert_eq!(
            expert_index_of(
                "layers.1.feed_forward.experts.7.w3.weight",
                ReapArch::Lfm2Moe
            ),
            Some(7)
        );
        assert_eq!(
            expert_index_of(
                "model.layers.2.block_sparse_moe.experts.31.w2.weight",
                ReapArch::Minimax
            ),
            Some(31)
        );
        // Non-routed-expert tensors → None.
        assert_eq!(
            expert_index_of("layers.0.self_attn.q_proj.weight", ReapArch::Deepseek4),
            None
        );
        assert_eq!(
            expert_index_of("layers.0.ffn.gate.weight", ReapArch::Deepseek4),
            None
        );
    }

    #[test]
    fn minimax_rename_renumbers() {
        let n = "model.layers.2.block_sparse_moe.experts.9.w1.weight";
        assert_eq!(
            bake_expert_rename(n, ReapArch::Minimax, 2, &[1, 9, 13]),
            Some("model.layers.2.block_sparse_moe.experts.1.w1.weight".to_string())
        );
    }

    #[test]
    fn layer_parse() {
        assert_eq!(bake_layer_of("layers.0.ffn.experts.2.w1.weight"), Some(0));
        assert_eq!(
            bake_layer_of("model.language_model.layers.41.mlp.gate.weight"),
            Some(41)
        );
        assert_eq!(bake_layer_of("model.embed_tokens.weight"), None);
    }
}

/// SP4b Task 2 prune finalize: router/bias row-gather + metadata expert-count
/// patch. These reach crate-internal `patch_expert_count_metadata` (in main.rs),
/// hence the in-module placement (binary crate has no lib target).
#[cfg(test)]
mod bake_finalize_tests {
    use super::*;
    use crate::patch_expert_count_metadata;
    use hipfire_reap::gather::gather_rows;

    #[test]
    fn router_gather_keeps_only_kept_rows() {
        // Router weight [orig_experts=4, dim=2] of f32 (8 bytes/row). Keep {2,0}.
        let dim = 2usize;
        let orig = 4usize;
        let mut bytes = Vec::new();
        for e in 0..orig {
            for d in 0..dim {
                bytes.extend_from_slice(&((e * 10 + d) as f32).to_le_bytes());
            }
        }
        let keep_l: Vec<u32> = vec![2, 0];
        let (new_shape, gathered) = gather_rows(&[orig, dim], &bytes, &keep_l).unwrap();
        // Row count == keep_l.len(); inner dim preserved.
        assert_eq!(new_shape, vec![keep_l.len(), dim]);
        assert_eq!(gathered.len(), keep_l.len() * dim * 4);
        // Compact slot 0 = original expert 2's first element (2*10+0 = 20.0).
        let slot0_e0 = f32::from_le_bytes(gathered[0..4].try_into().unwrap());
        assert_eq!(slot0_e0, 20.0);
        // Compact slot 1 = original expert 0's first element (0.0).
        let slot1_e0 = f32::from_le_bytes(gathered[dim * 4..dim * 4 + 4].try_into().unwrap());
        assert_eq!(slot1_e0, 0.0);
    }

    #[test]
    fn router_bias_classification_per_arch() {
        // Router weight names.
        assert!(is_reap_router_weight(
            "layers.5.ffn.gate.weight",
            ReapArch::Deepseek4
        ));
        assert!(is_reap_router_weight(
            "model.layers.5.mlp.gate.weight",
            ReapArch::Qwen35
        ));
        assert!(is_reap_router_weight(
            "layers.5.feed_forward.gate.weight",
            ReapArch::Lfm2Moe
        ));
        assert!(is_reap_router_weight(
            "model.layers.5.block_sparse_moe.gate.weight",
            ReapArch::Minimax
        ));
        // Shared-expert gate must NOT be classed as the router.
        assert!(!is_reap_router_weight(
            "model.layers.5.mlp.shared_expert_gate.weight",
            ReapArch::Qwen35
        ));
        // Per-expert bias names.
        assert!(is_reap_expert_bias(
            "layers.5.ffn.gate.bias",
            ReapArch::Deepseek4
        ));
        assert!(is_reap_expert_bias(
            "layers.5.feed_forward.expert_bias",
            ReapArch::Lfm2Moe
        ));
        assert!(is_reap_expert_bias(
            "model.layers.5.block_sparse_moe.e_score_correction_bias",
            ReapArch::Minimax
        ));
        assert!(!is_reap_expert_bias(
            "layers.5.ffn.gate.bias",
            ReapArch::Qwen35
        ));
    }

    fn meta(config: &str) -> String {
        format!(r#"{{"architecture":"x","config":{config},"tokenizer":"{{}}"}}"#)
    }

    #[test]
    fn patches_ds4_n_routed_experts() {
        let m = meta(r#"{"n_routed_experts":256,"hidden_size":7168}"#);
        let out = patch_expert_count_metadata(&m, ReapArch::Deepseek4, 32).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["config"]["n_routed_experts"], 32);
        assert_eq!(v["config"]["hidden_size"], 7168); // untouched
    }

    #[test]
    fn patches_qwen35_num_experts_top_level() {
        let m = meta(r#"{"num_experts":128}"#);
        let out = patch_expert_count_metadata(&m, ReapArch::Qwen35, 64).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["config"]["num_experts"], 64);
    }

    #[test]
    fn patches_qwen35_num_experts_in_text_config() {
        // VL-style nested config: num_experts lives under text_config.
        let m = meta(r#"{"text_config":{"num_experts":128}}"#);
        let out = patch_expert_count_metadata(&m, ReapArch::Qwen35, 64).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["config"]["text_config"]["num_experts"], 64);
    }

    #[test]
    fn patches_minimax_num_local_experts() {
        let m = meta(r#"{"num_local_experts":64}"#);
        let out = patch_expert_count_metadata(&m, ReapArch::Minimax, 16).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["config"]["num_local_experts"], 16);
    }

    #[test]
    fn patches_lfm2_num_experts() {
        let m = meta(r#"{"num_experts":32}"#);
        let out = patch_expert_count_metadata(&m, ReapArch::Lfm2Moe, 8).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["config"]["num_experts"], 8);
    }

    #[test]
    fn errors_when_field_absent() {
        // ds4 expects n_routed_experts; a config without it must error (never
        // silently ship the original count).
        let m = meta(r#"{"hidden_size":7168}"#);
        let err = patch_expert_count_metadata(&m, ReapArch::Deepseek4, 32).unwrap_err();
        assert!(err.contains("n_routed_experts"), "got: {err}");
    }
}

/// SP4 Task 4 integration test: `build_overlay` selection + byte-correctness,
/// then a real `write_hfq` → `HfqFile` container round-trip.
///
/// This lives in-module (not under `tests/`) because `hipfire-quantize` is a
/// binary crate with no library target — a `tests/` integration target can't
/// reach the crate-internal `write_hfq` / `quantize_hfq4g256` / `HfqTensor`.
/// `hipfire-runtime` (CPU-only HFQ container reader) is a dev-dependency.
#[cfg(test)]
mod integ {
    use super::*;
    use crate::{quantize_hfq4g256, write_hfq};
    use hipfire_reap::plan::ReapPlan;
    use hipfire_runtime::hfq::HfqFile;
    use std::sync::Mutex;

    // `HfqFile::open` READS the process-global `HIPFIRE_REAP_PLAN` env var on
    // every call. The end-to-end test below sets it before `open`, so it must
    // serialize against any other test in this module that calls `open` (none
    // today, but keep the guard to mirror SP3 T2's hipfire-runtime test).
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn plan_layer0_expert0_hfq4(dir: &std::path::Path) -> ReapPlan {
        std::fs::write(
            dir.join("reap_plan.json"),
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"routed_experts","experts":[0],"tier":"hfq4"}]}"#,
        )
        .unwrap();
        ReapPlan::load_unchecked(dir.to_str().unwrap()).unwrap()
    }

    /// A 4x256 tensor's worth of varied f32 (matches the plan's matched tensor).
    fn matched_f32() -> Vec<f32> {
        (0..4 * 256)
            .map(|i| ((i as f32) * 0.017).sin() * 3.0)
            .collect()
    }

    #[test]
    fn build_overlay_selects_and_byte_matches_then_round_trips() {
        // This test calls `HfqFile::open`, which reads the process-global
        // `HIPFIRE_REAP_PLAN`. Serialize against `sp4_overlay_to_sp3_load_end_to_end`
        // (which sets/removes that env var) so a concurrent set can't leak a stray
        // overlay into this `open` (was a latent flake before SP4b T2 added tests
        // that perturbed scheduling).
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let arch = ReapArch::Deepseek4;
        let d = tempfile::tempdir().unwrap();
        let plan = plan_layer0_expert0_hfq4(d.path());

        // 3 hand-built tensors: one matches (layer 0 routed expert 0), two don't
        // (wrong expert, and an attention tensor not in the plan).
        let matched_name = "layers.0.ffn.experts.0.w1.weight".to_string();
        let matched = matched_f32();
        let tensors = vec![
            (matched_name.clone(), vec![4usize, 256], matched.clone()),
            (
                "layers.0.ffn.experts.1.w1.weight".to_string(),
                vec![4usize, 256],
                vec![0.5f32; 4 * 256],
            ),
            (
                "layers.0.self_attn.q_proj.weight".to_string(),
                vec![4usize, 256],
                vec![1.0f32; 4 * 256],
            ),
        ];

        let out = build_overlay(arch, &plan, tensors.into_iter()).unwrap();

        // (a) exactly the 1 matched tensor is emitted.
        assert_eq!(out.len(), 1, "only the matched tensor should be emitted");
        assert_eq!(out[0].name, matched_name);
        assert_eq!(out[0].quant_type, QuantType::HFQ4G256);
        assert_eq!(out[0].shape, vec![4u32, 256]);

        // (b) bytes equal a direct quantize_hfq4g256 of the matched f32.
        let direct = quantize_hfq4g256(&matched);
        assert_eq!(
            out[0].data, direct,
            "overlay bytes must equal direct encode"
        );

        // (c) write_hfq → HfqFile round-trip: retrievable by name, right qt.
        let overlay_path = d.path().join("overlay.hfq");
        // Minimal metadata object — write_hfq's container + HfqFile's JSON
        // brace-scan both accept `{}`.
        write_hfq(&overlay_path, 9, "{}", &out, None).unwrap();

        let hf = HfqFile::open(&overlay_path).unwrap();
        assert_eq!(hf.arch_id, 9);
        let info = hf
            .find_tensor_info(&matched_name)
            .expect("matched tensor retrievable by name from overlay.hfq");
        assert_eq!(info.quant_type, QuantType::HFQ4G256 as u8);
        assert_eq!(info.shape, vec![4u32, 256]);
        // The non-matched tensors must be absent from the overlay.
        assert!(hf
            .find_tensor_info("layers.0.self_attn.q_proj.weight")
            .is_none());
        assert!(hf
            .find_tensor_info("layers.0.ffn.experts.1.w1.weight")
            .is_none());

        // And the stored data bytes match what we wrote.
        let (_, data) = hf
            .tensor_data(&matched_name)
            .expect("tensor data readable back");
        assert_eq!(
            data,
            &direct[..],
            "round-tripped data must equal direct encode"
        );
    }

    /// SP3 Task 3: full SP4-overlay → SP3-load round trip on REAL HFQ
    /// containers. Build a base (expert + attention tensors, both Q8F16),
    /// build an overlay that re-quantizes ONLY the expert tensor to HFQ4G256,
    /// point `HIPFIRE_REAP_PLAN` at the overlay dir, then `HfqFile::open` the
    /// base and assert: overlay attached, expert resolves to the overlay's
    /// HFQ4G256 bytes, attention falls through to base Q8F16.
    #[test]
    fn sp4_overlay_to_sp3_load_end_to_end() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        let exp_name = "layers.0.ffn.experts.0.w1.weight";
        let attn_name = "layers.0.self_attn.q_proj.weight";
        // [2, 256] = 512 f32 (multiple of the 256 group for the HFQ4G256 tier).
        let shape = [2usize, 256];
        let exp_f32: Vec<f32> = (0..2 * 256)
            .map(|i| ((i as f32) * 0.019).sin() * 5.0)
            .collect();
        let attn_f32: Vec<f32> = (0..2 * 256)
            .map(|i| ((i as f32) * 0.011).cos() * 2.0)
            .collect();

        // --- (1) base model: both tensors → Q8F16, arch 9, written to base.hfq.
        let base_dir = tempfile::tempdir().unwrap();
        let base_path = base_dir.path().join("base.hfq");
        let exp_q8 = quantize_to_format(exp_name, "q8", &exp_f32, &shape).unwrap();
        let attn_q8 = quantize_to_format(attn_name, "q8", &attn_f32, &shape).unwrap();
        assert_eq!(exp_q8.quant_type, QuantType::Q8F16);
        write_hfq(&base_path, 9, "{}", &[exp_q8, attn_q8], None).unwrap();

        // --- (2) overlay: re-quantize ONLY the expert tensor → HFQ4G256.
        let plan_dir = tempfile::tempdir().unwrap();
        let overlay_path = plan_dir.path().join("overlay.hfq");
        let exp_overlay = quantize_to_format(exp_name, "hfq4g256", &exp_f32, &shape).unwrap();
        assert_eq!(exp_overlay.quant_type, QuantType::HFQ4G256);
        write_hfq(&overlay_path, 9, "{}", &[exp_overlay], None).unwrap();

        // The byte-exact reference the overlay must serve for the expert tensor.
        let direct_hfq4 = quantize_to_format(exp_name, "hfq4g256", &exp_f32, &shape)
            .unwrap()
            .data;

        // --- (3) point HIPFIRE_REAP_PLAN at the overlay dir, then open the base.
        // Capture everything we need while the guard is held; remove the env var
        // immediately after `open` (before any assert can unwind) so a failure
        // can't leak the var to a concurrent `open`.
        std::env::set_var("HIPFIRE_REAP_PLAN", plan_dir.path());
        let f = HfqFile::open(&base_path).unwrap();
        std::env::remove_var("HIPFIRE_REAP_PLAN");

        // Overlay auto-attached (arch_id 9 matches; expert name is a base subset).
        assert!(
            f.has_overlay(),
            "overlay.hfq should auto-attach from HIPFIRE_REAP_PLAN"
        );

        // Expert tensor resolves to the OVERLAY (HFQ4G256), and its bytes equal
        // a direct hfq4g256 encode of the same f32.
        let exp_info = f
            .find_tensor_info(exp_name)
            .expect("expert tensor resolvable");
        assert_eq!(
            exp_info.quant_type,
            QuantType::HFQ4G256 as u8,
            "expert tensor must resolve to the overlay tier"
        );
        let (exp_info2, exp_bytes) = f.tensor_data_vec(exp_name).expect("expert bytes");
        assert_eq!(exp_info2.quant_type, QuantType::HFQ4G256 as u8);
        assert_eq!(
            exp_bytes, direct_hfq4,
            "expert overlay bytes must equal a direct hfq4g256 encode"
        );

        // Attention tensor falls through to BASE (Q8F16).
        let attn_info = f
            .find_tensor_info(attn_name)
            .expect("attention tensor resolvable from base");
        assert_eq!(
            attn_info.quant_type,
            QuantType::Q8F16 as u8,
            "attention tensor must fall through to base Q8F16"
        );
    }
}
