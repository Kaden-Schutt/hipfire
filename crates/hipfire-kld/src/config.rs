// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! `KldConfig` — the single env contract for KLD reference build *and* candidate
//! scoring.
//!
//! The self-inconsistency this refactor fixes came from two binaries reading a
//! *different subset* of the determinism-affecting env vars (e.g. one set
//! `HIPFIRE_KLD_FP32_GQA4_ATTN` and the other never read it). Centralizing every
//! such flag in one struct — populated once and applied identically to ref-build
//! and score — makes that class of drift unrepresentable.
//!
//! `from_env` reads the current environment; [`KldConfig::to_env_pairs`] renders
//! the exact `(key, value)` set the forward path expects, so the daemon applies
//! one consistent environment for both phases.

use serde::{Deserialize, Serialize};

/// Scoring traversal of the captured hidden states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScoringMode {
    /// Batched prefill, lm-head fan-out per scored position (canonical).
    Prefill,
    /// Per-position decode loop (historical baseline).
    PerToken,
    /// Single batched pass over the whole context, no prefix/scored split.
    SingleShot,
}

impl ScoringMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ScoringMode::Prefill => "prefill",
            ScoringMode::PerToken => "per-token",
            ScoringMode::SingleShot => "single-shot",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "prefill" => Some(ScoringMode::Prefill),
            "per-token" | "per_token" | "pertoken" => Some(ScoringMode::PerToken),
            "single-shot" | "single_shot" | "singleshot" => Some(ScoringMode::SingleShot),
            _ => None,
        }
    }
}

/// Every flag that can change scored logits. Defaults encode the canonical eval
/// contract (prompt-normalization OFF, graph OFF, fp32 KV, GQA4 attention ON,
/// direct-f16kv attention OFF) so ref-build and score start from the same point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KldConfig {
    pub scoring_mode: ScoringMode,
    pub top_k: usize,
    pub kv_mode: String,
    /// Collapse `\n{3,}` before tokenizing. Eval forces OFF for determinism.
    pub normalize_prompt: bool,
    /// hipGraph capture of the prefill stack.
    pub graph: bool,
    /// Direct fp32 GQA4 attention prefill variant.
    pub fp32_gqa4_attn: bool,
    /// Direct f16-KV WMMA attention prefill variant.
    pub direct_f16kv_attn: bool,
    /// Reuse a resident `PrefillBatchScratch`.
    pub reuse_pbs: bool,
    /// Max prefill batch (tokens per `forward_prefill_chunk`). `None` → engine default.
    pub prefill_max_batch: Option<usize>,
}

impl Default for KldConfig {
    fn default() -> Self {
        KldConfig {
            scoring_mode: ScoringMode::Prefill,
            top_k: 256,
            kv_mode: "fp32".to_string(),
            normalize_prompt: false,
            graph: false,
            fp32_gqa4_attn: true,
            direct_f16kv_attn: false,
            reuse_pbs: true,
            prefill_max_batch: None,
        }
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key).ok().as_deref() {
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES") => true,
        Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO") => false,
        _ => default,
    }
}

impl KldConfig {
    /// Populate from the current environment, falling back to the canonical
    /// defaults for anything unset.
    pub fn from_env() -> Self {
        let d = KldConfig::default();
        KldConfig {
            scoring_mode: std::env::var("HIPFIRE_KLD_SCORING_MODE")
                .ok()
                .and_then(|s| ScoringMode::parse(&s))
                .unwrap_or(d.scoring_mode),
            top_k: std::env::var("HIPFIRE_KLD_TOP_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d.top_k),
            kv_mode: std::env::var("HIPFIRE_KV_MODE").unwrap_or(d.kv_mode),
            normalize_prompt: env_bool("HIPFIRE_NORMALIZE_PROMPT", d.normalize_prompt),
            graph: env_bool("HIPFIRE_KLD_GRAPH", d.graph),
            fp32_gqa4_attn: env_bool("HIPFIRE_KLD_FP32_GQA4_ATTN", d.fp32_gqa4_attn),
            direct_f16kv_attn: env_bool("HIPFIRE_KLD_DIRECT_F16KV_ATTN", d.direct_f16kv_attn),
            reuse_pbs: env_bool("HIPFIRE_PREFILL_REUSE_PBS", d.reuse_pbs),
            prefill_max_batch: std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| v >= 2)
                .or(d.prefill_max_batch),
        }
    }

    /// The exact `(key, value)` environment the forward path expects for this
    /// config. The daemon applies these once so ref-build and score share one
    /// environment (eliminating the read-mismatch drift). `n_ctx` seeds the
    /// prefill-max-batch default when the config leaves it unset.
    pub fn to_env_pairs(&self, n_ctx: usize) -> Vec<(&'static str, String)> {
        let b = |v: bool| if v { "1" } else { "0" }.to_string();
        vec![
            ("HIPFIRE_NORMALIZE_PROMPT", b(self.normalize_prompt)),
            ("HIPFIRE_GRAPH", b(self.graph)),
            ("HIPFIRE_KLD_GRAPH", b(self.graph)),
            ("HIPFIRE_KLD_FP32_GQA4_ATTN", b(self.fp32_gqa4_attn)),
            ("HIPFIRE_KLD_DIRECT_F16KV_ATTN", b(self.direct_f16kv_attn)),
            ("HIPFIRE_KLD_DIRECT_WMMA_ATTN", b(self.direct_f16kv_attn)),
            ("HIPFIRE_PREFILL_REUSE_PBS", b(self.reuse_pbs)),
            ("HIPFIRE_KV_MODE", self.kv_mode.clone()),
            (
                "HIPFIRE_PREFILL_MAX_BATCH",
                self.prefill_max_batch.unwrap_or(n_ctx).to_string(),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_canonical() {
        let c = KldConfig::default();
        assert_eq!(c.scoring_mode, ScoringMode::Prefill);
        assert_eq!(c.top_k, 256);
        assert!(!c.normalize_prompt && !c.graph);
        assert!(c.fp32_gqa4_attn && !c.direct_f16kv_attn);
    }

    #[test]
    fn env_pairs_cover_the_drift_flags() {
        // The exact flags that caused the eval/ref divergence must be emitted.
        let pairs = KldConfig::default().to_env_pairs(2048);
        let keys: Vec<_> = pairs.iter().map(|(k, _)| *k).collect();
        for need in [
            "HIPFIRE_KLD_FP32_GQA4_ATTN",
            "HIPFIRE_KLD_DIRECT_F16KV_ATTN",
            "HIPFIRE_GRAPH",
            "HIPFIRE_PREFILL_MAX_BATCH",
        ] {
            assert!(keys.contains(&need), "missing {need}");
        }
        // max_batch defaults to n_ctx when unset.
        let mb = pairs
            .iter()
            .find(|(k, _)| *k == "HIPFIRE_PREFILL_MAX_BATCH")
            .unwrap();
        assert_eq!(mb.1, "2048");
    }

    #[test]
    fn scoring_mode_round_trips() {
        for m in [
            ScoringMode::Prefill,
            ScoringMode::PerToken,
            ScoringMode::SingleShot,
        ] {
            assert_eq!(ScoringMode::parse(m.as_str()), Some(m));
        }
    }
}
