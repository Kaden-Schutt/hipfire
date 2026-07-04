// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The **ingest** capability: an arch's quant-policy knowledge, expressed as NEEDS
//! (importance + requirements) — never as formats. Plus the codec-capability
//! vocabulary and a reference allocator the deployment uses to turn needs into a
//! concrete codec under a bit budget.
//!
//! This is what dissolves the `is_q8_tensor` / `is_deepseek4_keep_f16` smell. The
//! arch says "this tensor is important and gather-indexed"; the deployment DERIVES
//! a random-access high-bit codec. No format name and no arch name appears in the
//! arch's `impl Ingest`. Three layers, none naming another's concern:
//!
//!  - ARCH declares needs:      [`Ingest::importance`] + [`Ingest::requires`]
//!  - CODEC declares abilities: [`CodecCaps`]
//!  - DEPLOYMENT matches them:  [`allocate`] under [`target_bits`]

/// What a tensor *is* — a mechanical name→role classification. Structural and
/// format-free; `importance`/`requires` are usually computed from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    AttnProj,
    Mlp,
    /// Writes into the residual stream (out_proj / down_proj) — accumulates error.
    ResidualWriter,
    Embed,
    LmHead,
    Router,
    Expert,
    Conv1d,
    Norm,
    Other,
}

/// Ordered precision NEED tier for a tensor — a finer, discrete companion to the
/// scalar [`Ingest::importance`]. It is a *need*, never a format: the deployment
/// maps a class to a concrete codec (no on-disk format token appears here). The
/// ordering is by fidelity (`Aggressive` cheapest → `SourcePrecision` richest), so
/// consumers can compare with `>=`.
///
/// Where scalar importance collapses distinct needs onto one value (a numerically
/// critical MLA compressor and ordinary attention both read as ~255), this splits
/// them: the compressor is [`SourcePrecision`](PrecisionClass::SourcePrecision), the
/// attention is [`High`](PrecisionClass::High). That is what lets the quantizer drop
/// the arch-name keep-lists (`is_deepseek4_keep_f16`, `is_nemotron_h_*`) — each arch
/// declares the class for its special tensors in its own `-spec` crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionClass {
    /// Bulk that tolerates the most aggressive (2-bit-class) compression.
    Aggressive,
    /// The compressible bulk (4-bit-class) — FFN / experts under a normal budget.
    Compressed,
    /// Keep at high precision — the structurally protected set.
    High,
    /// Pinned high precision: do NOT compress below high even under a tight budget
    /// (e.g. an aggressive low-bit target). SSM ingress / residual writers that corrupt state
    /// when lossy sit here — above ordinary `High`, which a tight budget may spend down.
    Pinned,
    /// Keep at source fidelity — never quantized (the deployment lands it at bf16/f16).
    /// Numerically critical stream generators (MLA compressor / indexer) sit here.
    SourcePrecision,
}

/// Format-AGNOSTIC representability requirements a tensor places on its codec. A
/// requirement is a *need* ("I must be randomly accessible"), never a solution
/// ("store me with some specific codec").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapReq {
    /// The tensor is gather / random-indexed (embeddings, lm_head): the codec must
    /// support O(1) access to a single row. This rules out sequential trellis
    /// codecs — but the arch never says "trellis", only "I need random access".
    pub random_access: bool,
}

impl CapReq {
    pub const NONE: CapReq = CapReq {
        random_access: false,
    };
    pub const RANDOM_ACCESS: CapReq = CapReq {
        random_access: true,
    };
}

/// The ingest capability: pure, format-free quant-policy priors for an arch's
/// tensors. Every method is a function of the tensor name only, so the offline
/// quantizer can consult it without any runtime/GPU state.
pub trait Ingest: Sync + 'static {
    /// Mechanical name→role classification.
    fn role(&self, tensor: &str) -> TensorRole;
    /// Structural saliency prior in `0..=255` (higher → protect harder). A
    /// numerically-critical tensor is simply `255`; "keep it high precision" is not
    /// a separate concept and never names a format. The deployment may fuse this
    /// with measured (GuidedQuant/Fisher) saliency when calibration exists.
    fn importance(&self, tensor: &str) -> u8;
    /// Hard, format-agnostic representability requirements.
    fn requires(&self, tensor: &str) -> CapReq;
    /// Discrete precision NEED tier. Defaults to a role-derived class; an arch
    /// overrides this to pin specific tensors ABOVE the structural default — e.g.
    /// its MLA compressors to [`PrecisionClass::SourcePrecision`], its SSM ingress to
    /// [`PrecisionClass::Pinned`]. Because the quantizer consults it keyed by
    /// `ArchId`, a class an arch does not declare can never leak onto another family.
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        default_precision_class(self.role(tensor))
    }
}

/// A codec's self-declared capabilities. NO arch name appears here; `name` is for
/// reporting only and is supplied by the codec registry / deployment, never by an
/// arch.
#[derive(Debug, Clone, Copy)]
pub struct CodecCaps {
    /// Human tag for reporting (e.g. the deployment's codec id). Not matched on.
    pub name: &'static str,
    /// Average bits per weight (mixed-precision codecs use a fractional value).
    pub bits_per_weight: f32,
    /// Group size the codec quantizes in; `0` means ungrouped (no divisibility
    /// constraint on the tensor's inner dimension).
    pub group_size: usize,
    /// Supports O(1) access to a single row (embeddings / lm_head gather).
    pub random_access: bool,
}

/// Reference importance→target-bits curve — a DEPLOYMENT policy default. Bare
/// numbers only (no format names), so this crate stays format-token-free.
pub fn target_bits(importance: u8) -> f32 {
    match importance {
        0..=49 => 2.0,
        50..=250 => 4.0,
        _ => 8.0,
    }
}

/// Pick a codec for one tensor of inner dimension `k`:
///  1. keep only codecs that satisfy `req` AND whose group size divides `k`;
///  2. meet the importance bit-floor — the *smallest* valid codec with
///     `bits_per_weight >= target_bits(importance)`;
///  3. if the budget can't reach the floor, best-effort to the *largest* valid
///     codec.
///
/// Returns `None` only when no codec satisfies the requirements at all (e.g. a
/// random-access tensor but only sequential codecs on offer).
pub fn allocate<'a>(
    importance: u8,
    req: CapReq,
    k: usize,
    codecs: &'a [CodecCaps],
) -> Option<&'a CodecCaps> {
    let target = target_bits(importance);
    let valid = |c: &CodecCaps| {
        (!req.random_access || c.random_access) && (c.group_size == 0 || k % c.group_size == 0)
    };
    // Smallest codec meeting the importance floor.
    let meet = codecs
        .iter()
        .filter(|&c| valid(c) && c.bits_per_weight >= target)
        .min_by(|a, b| a.bits_per_weight.total_cmp(&b.bits_per_weight));
    if meet.is_some() {
        return meet;
    }
    // Budget can't meet the floor: take the largest codec that's still valid.
    codecs
        .iter()
        .filter(|&c| valid(c))
        .max_by(|a, b| a.bits_per_weight.total_cmp(&b.bits_per_weight))
}

// ---------------------------------------------------------------------------
// Shared policy helpers. Most families' quant policy is the same "protect the
// gather tables + attention + norms, compress the FFN/expert bulk" prior; these
// let a family's `-spec` crate be a ~6-line delegation. Families override where
// they genuinely differ (MLA compressors, SSM protections, …).
// ---------------------------------------------------------------------------

/// Name→role classifier covering the transformer families (dense, MoE, and the
/// SSM/hybrid mixers). Checked most-specific first.
pub fn transformer_role(name: &str) -> TensorRole {
    if name.contains("embed") {
        TensorRole::Embed
    } else if name.contains("lm_head") {
        TensorRole::LmHead
    // MoE routers: small but routing-sensitive (must resolve before generic mlp).
    } else if name.ends_with("mlp.gate.weight")
        || name.ends_with("shared_expert_gate.weight")
        || name.ends_with(".mixer.gate.weight")
        || name.ends_with(".router.weight")
        || name.ends_with("block_sparse_moe.gate.weight")
    {
        TensorRole::Router
    // MoE experts: the bulk of a sparse model.
    } else if name.contains(".experts.") || name.contains(".expert.") {
        TensorRole::Expert
    // Short convolution (Mamba / LFM2 / DeltaNet): tiny, runs every token.
    } else if name.contains("conv1d") {
        TensorRole::Conv1d
    // Attention / SSM ingress projections (incl. linear/delta attn, Mamba in_proj).
    } else if name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("qkv")
        || name.contains("self_attn")
        || name.contains("linear_attn")
        || name.contains("in_proj")
    {
        TensorRole::AttnProj
    // Residual writers: attention/SSM output projections that accumulate into the
    // residual stream.
    } else if name.contains("o_proj") || name.contains("out_proj") {
        TensorRole::ResidualWriter
    } else if name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains("down_proj")
        || name.contains("mlp.")
        || name.contains("ffn")
    {
        TensorRole::Mlp
    } else if name.contains("norm") {
        TensorRole::Norm
    } else {
        TensorRole::Other
    }
}

/// Default importance prior for a role: protect the gather tables, attention,
/// residual writers, norms and routers at high precision; compress the FFN/expert
/// bulk. A structural prior only — the quantizer refines the actual bits later.
pub fn default_importance(role: TensorRole) -> u8 {
    match role {
        TensorRole::Embed
        | TensorRole::LmHead
        | TensorRole::Norm
        | TensorRole::AttnProj
        | TensorRole::ResidualWriter
        | TensorRole::Router
        | TensorRole::Conv1d => 255,
        TensorRole::Mlp | TensorRole::Expert => 128,
        TensorRole::Other => 160,
    }
}

/// Default precision class for a role — the discrete companion to
/// [`default_importance`], kept consistent with it: the protected set is `High`, the
/// FFN/expert bulk is `Compressed`. No role defaults to `Pinned`/`SourcePrecision`;
/// those are only ever reached by an arch's explicit override, so the quantizer's
/// pinned/source-precision paths fire only for tensors a family deliberately declares.
pub fn default_precision_class(role: TensorRole) -> PrecisionClass {
    match role {
        TensorRole::Embed
        | TensorRole::LmHead
        | TensorRole::Norm
        | TensorRole::AttnProj
        | TensorRole::ResidualWriter
        | TensorRole::Router
        | TensorRole::Conv1d => PrecisionClass::High,
        TensorRole::Mlp | TensorRole::Expert | TensorRole::Other => PrecisionClass::Compressed,
    }
}

/// Default requirements for a role: only the gather-indexed tables need random
/// access.
pub fn default_requires(role: TensorRole) -> CapReq {
    match role {
        TensorRole::Embed | TensorRole::LmHead => CapReq::RANDOM_ACCESS,
        _ => CapReq::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_class_is_ordered_and_role_default_is_sane() {
        // Fidelity ordering the quantizer relies on for `>=` comparisons.
        assert!(PrecisionClass::SourcePrecision > PrecisionClass::Pinned);
        assert!(PrecisionClass::Pinned > PrecisionClass::High);
        assert!(PrecisionClass::High > PrecisionClass::Compressed);
        assert!(PrecisionClass::Compressed > PrecisionClass::Aggressive);
        // No role defaults into the pinned/source tiers — those need an explicit
        // per-arch override, so a family can't accidentally pin another's tensors.
        for role in [
            TensorRole::Embed,
            TensorRole::LmHead,
            TensorRole::Norm,
            TensorRole::AttnProj,
            TensorRole::ResidualWriter,
            TensorRole::Router,
            TensorRole::Conv1d,
            TensorRole::Mlp,
            TensorRole::Expert,
            TensorRole::Other,
        ] {
            assert!(default_precision_class(role) <= PrecisionClass::High);
        }
        // Consistent with default_importance: protected set high, bulk compressed.
        assert_eq!(
            default_precision_class(TensorRole::AttnProj),
            PrecisionClass::High
        );
        assert_eq!(
            default_precision_class(TensorRole::Mlp),
            PrecisionClass::Compressed
        );
    }

    #[test]
    fn transformer_role_classifies_families() {
        assert_eq!(
            transformer_role("model.embed_tokens.weight"),
            TensorRole::Embed
        );
        assert_eq!(transformer_role("lm_head.weight"), TensorRole::LmHead);
        assert_eq!(
            transformer_role("model.layers.0.mlp.gate.weight"),
            TensorRole::Router
        );
        assert_eq!(
            transformer_role("model.layers.0.mlp.experts.3.up_proj.weight"),
            TensorRole::Expert
        );
        assert_eq!(
            transformer_role("model.layers.0.self_attn.q_proj.weight"),
            TensorRole::AttnProj
        );
        assert_eq!(
            transformer_role("backbone.layers.0.mixer.out_proj.weight"),
            TensorRole::ResidualWriter
        );
        assert_eq!(
            transformer_role("backbone.layers.0.mixer.conv1d.weight"),
            TensorRole::Conv1d
        );
        assert_eq!(
            transformer_role("model.layers.0.mlp.up_proj.weight"),
            TensorRole::Mlp
        );
        assert_eq!(transformer_role("model.norm.weight"), TensorRole::Norm);
        // The protected set is high-importance; the FFN/expert bulk is compressible.
        assert!(default_importance(TensorRole::AttnProj) > default_importance(TensorRole::Mlp));
        assert_eq!(default_requires(TensorRole::Embed), CapReq::RANDOM_ACCESS);
        assert_eq!(default_requires(TensorRole::Mlp), CapReq::NONE);
    }

    // Generic codec descriptors. NOTE: deliberately NOT real format names — this
    // crate must stay format-token-free (the purity gate greps it). The allocator
    // matches on capabilities, never on names, so generic tags exercise it fully.
    fn reg() -> Vec<CodecCaps> {
        vec![
            CodecCaps {
                name: "seq-2b",
                bits_per_weight: 2.0,
                group_size: 256,
                random_access: false,
            },
            CodecCaps {
                name: "seq-4b",
                bits_per_weight: 4.0,
                group_size: 256,
                random_access: false,
            },
            CodecCaps {
                name: "ra-4b",
                bits_per_weight: 4.0,
                group_size: 0,
                random_access: true,
            },
            CodecCaps {
                name: "ra-8b",
                bits_per_weight: 8.0,
                group_size: 0,
                random_access: true,
            },
        ]
    }

    #[test]
    fn random_access_need_derives_a_random_access_codec() {
        // A max-importance gather-indexed tensor (embeddings): importance floor 8b
        // + random-access requirement → the random-access 8b codec. This is exactly
        // is_q8_tensor's embed behaviour, DERIVED from needs — no format named here.
        let r = reg();
        let sel = allocate(255, CapReq::RANDOM_ACCESS, 4096, &r).unwrap();
        assert_eq!(sel.name, "ra-8b");
        assert!(sel.random_access);
    }

    #[test]
    fn importance_drives_bits() {
        let r = reg();
        // Low importance → 2b floor → cheapest codec.
        assert_eq!(allocate(10, CapReq::NONE, 4096, &r).unwrap().name, "seq-2b");
        // Medium importance → 4b floor → the (first) 4b codec.
        assert_eq!(
            allocate(128, CapReq::NONE, 4096, &r).unwrap().name,
            "seq-4b"
        );
    }

    #[test]
    fn group_divisibility_excludes_grouped_codecs() {
        // k=300 is not divisible by 256 → grouped codecs are excluded, only
        // ungrouped ones remain; medium importance → the ungrouped 4b codec.
        let r = reg();
        let sel = allocate(128, CapReq::NONE, 300, &r).unwrap();
        assert_eq!(sel.group_size, 0);
        assert_eq!(sel.name, "ra-4b");
    }

    #[test]
    fn none_when_requirement_unsatisfiable() {
        // Need random access, but only sequential codecs on offer.
        let seq = [CodecCaps {
            name: "seq-2b",
            bits_per_weight: 2.0,
            group_size: 256,
            random_access: false,
        }];
        assert!(allocate(255, CapReq::RANDOM_ACCESS, 4096, &seq).is_none());
    }

    #[test]
    fn best_effort_when_budget_below_floor() {
        // Max importance wants 8b, but the best available is 4b → best-effort to
        // the largest valid codec rather than failing.
        let small = [
            CodecCaps {
                name: "seq-2b",
                bits_per_weight: 2.0,
                group_size: 256,
                random_access: false,
            },
            CodecCaps {
                name: "seq-4b",
                bits_per_weight: 4.0,
                group_size: 256,
                random_access: false,
            },
        ];
        assert_eq!(
            allocate(255, CapReq::NONE, 4096, &small).unwrap().name,
            "seq-4b"
        );
    }
}
