// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Self-describing reference metadata + a compatibility check.
//!
//! The multi-day gfx1103 bisection that motivated this refactor came down to a
//! reference being scored by *diverged code* with a *different config* on a
//! *different arch* — and nothing flagged it; the eval just emitted a
//! plausible-but-meaningless 2.85. The fix is to make the reference fully
//! self-describe the `(code, config, arch, tokenizer)` it was produced under,
//! and to have the consumer compare its own run context and **refuse on hard
//! mismatches / warn on soft ones** before trusting a number.
//!
//! [`RefMeta`] is that self-description; [`RunEnv`] is the consumer's context;
//! [`compat`] is the gate.

use crate::config::KldConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Code/host provenance of the producer that built a reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProducerInfo {
    pub hipfire_version: String,
    /// Git commit the reference was built from. **Was null historically — that
    /// omission is exactly what let a cross-version comparison go unnoticed.**
    pub git_commit: Option<String>,
    pub git_describe: Option<String>,
    pub git_dirty: Option<bool>,
    pub gpu_arch: String,
    pub producer_cmd: Option<String>,
}

/// The full self-describing header for a `.kldref` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefMeta {
    pub schema: u32,
    pub base_model_id: String,
    /// Source (reference-precision) model weight hash.
    pub source_model_sha256: String,
    /// Tokenizer identity — guards against a candidate that tokenizes differently.
    pub tokenizer_sha256: Option<String>,
    pub arch_id: u32,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_chunk: usize,
    pub scored_per_chunk: usize,
    pub scoring_start: usize,
    pub top_k: usize,
    pub total_scored: usize,
    pub slice_path: String,
    pub slice_md5: String,
    /// The COMPLETE determinism contract the reference was built under. The
    /// consumer applies the same and diffs it field-by-field.
    pub config: KldConfig,
    pub producer: ProducerInfo,
    /// Per-payload-blob codec tag (e.g. `"top_indices" -> "bitpacked-idx:18"`).
    pub payload_codecs: BTreeMap<String, String>,
    /// Hash over the decoded payloads — integrity + daemon-side resident cache key.
    pub content_sha256: Option<String>,
}

/// The consumer's current run context, compared against a [`RefMeta`].
#[derive(Debug, Clone, PartialEq)]
pub struct RunEnv {
    pub git_commit: Option<String>,
    pub gpu_arch: String,
    pub arch_id: u32,
    pub n_vocab: usize,
    pub tokenizer_sha256: Option<String>,
    pub config: KldConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The number would be meaningless — refuse to score.
    Error,
    /// The number is suspect — surface loudly but allow.
    Warn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub field: &'static str,
    pub severity: Severity,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompatReport {
    pub mismatches: Vec<Mismatch>,
}

impl CompatReport {
    pub fn is_clean(&self) -> bool {
        self.mismatches.is_empty()
    }
    pub fn has_errors(&self) -> bool {
        self.mismatches
            .iter()
            .any(|m| m.severity == Severity::Error)
    }
    pub fn errors(&self) -> impl Iterator<Item = &Mismatch> {
        self.mismatches
            .iter()
            .filter(|m| m.severity == Severity::Error)
    }
    pub fn warnings(&self) -> impl Iterator<Item = &Mismatch> {
        self.mismatches
            .iter()
            .filter(|m| m.severity == Severity::Warn)
    }
}

/// Compare a consumer's run context against a reference's recorded provenance.
///
/// Hard (`Error`) mismatches make the score meaningless: different model
/// architecture, vocabulary, or tokenizer. Soft (`Warn`) mismatches make it
/// suspect: different code commit (the historical silent-divergence cause),
/// different GPU arch (cross-arch numerics), or any differing determinism flag
/// in [`KldConfig`]. Equal-or-unknown fields don't fire (we don't warn on
/// absent provenance, only on present-and-different).
pub fn compat(meta: &RefMeta, run: &RunEnv) -> CompatReport {
    let mut m = Vec::new();

    if meta.arch_id != run.arch_id {
        m.push(Mismatch {
            field: "arch_id",
            severity: Severity::Error,
            detail: format!("ref {} != run {}", meta.arch_id, run.arch_id),
        });
    }
    if meta.n_vocab != run.n_vocab {
        m.push(Mismatch {
            field: "n_vocab",
            severity: Severity::Error,
            detail: format!("ref {} != run {}", meta.n_vocab, run.n_vocab),
        });
    }
    if let (Some(a), Some(b)) = (&meta.tokenizer_sha256, &run.tokenizer_sha256) {
        if a != b {
            m.push(Mismatch {
                field: "tokenizer_sha256",
                severity: Severity::Error,
                detail: "reference and candidate tokenizers differ".to_string(),
            });
        }
    }

    if let (Some(a), Some(b)) = (&meta.producer.git_commit, &run.git_commit) {
        if a != b {
            m.push(Mismatch {
                field: "git_commit",
                severity: Severity::Warn,
                detail: format!("ref {a} != run {b} (forward code may differ)"),
            });
        }
    }
    if meta.producer.gpu_arch != run.gpu_arch {
        m.push(Mismatch {
            field: "gpu_arch",
            severity: Severity::Warn,
            detail: format!(
                "ref {} != run {} (cross-arch numerics)",
                meta.producer.gpu_arch, run.gpu_arch
            ),
        });
    }

    config_mismatches(&meta.config, &run.config, &mut m);
    CompatReport { mismatches: m }
}

fn config_mismatches(r: &KldConfig, c: &KldConfig, out: &mut Vec<Mismatch>) {
    macro_rules! diff {
        ($field:ident, $name:literal) => {
            if r.$field != c.$field {
                out.push(Mismatch {
                    field: $name,
                    severity: Severity::Warn,
                    detail: format!("ref {:?} != run {:?}", r.$field, c.$field),
                });
            }
        };
    }
    diff!(scoring_mode, "config.scoring_mode");
    diff!(top_k, "config.top_k");
    diff!(kv_mode, "config.kv_mode");
    diff!(normalize_prompt, "config.normalize_prompt");
    diff!(graph, "config.graph");
    diff!(fp32_gqa4_attn, "config.fp32_gqa4_attn");
    diff!(direct_f16kv_attn, "config.direct_f16kv_attn");
    diff!(reuse_pbs, "config.reuse_pbs");
    diff!(prefill_max_batch, "config.prefill_max_batch");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> RefMeta {
        RefMeta {
            schema: 2,
            base_model_id: "qwen3.5-0.8b-bf16".into(),
            source_model_sha256: "abc".into(),
            tokenizer_sha256: Some("tok".into()),
            arch_id: 5,
            n_vocab: 248_320,
            n_ctx: 2048,
            n_chunk: 1175,
            scored_per_chunk: 1023,
            scoring_start: 1024,
            top_k: 256,
            total_scored: 1_202_025,
            slice_path: "slice.txt".into(),
            slice_md5: "md5".into(),
            config: KldConfig::default(),
            producer: ProducerInfo {
                hipfire_version: "0.2.0".into(),
                git_commit: Some("aaaa".into()),
                gpu_arch: "gfx1151".into(),
                ..Default::default()
            },
            payload_codecs: BTreeMap::new(),
            content_sha256: None,
        }
    }

    fn run() -> RunEnv {
        RunEnv {
            git_commit: Some("aaaa".into()),
            gpu_arch: "gfx1151".into(),
            arch_id: 5,
            n_vocab: 248_320,
            tokenizer_sha256: Some("tok".into()),
            config: KldConfig::default(),
        }
    }

    #[test]
    fn identical_context_is_clean() {
        assert!(compat(&meta(), &run()).is_clean());
    }

    #[test]
    fn diverged_commit_warns_not_errors() {
        // The exact failure mode: same model/arch/tokenizer, different code.
        let mut r = run();
        r.git_commit = Some("bbbb".into());
        let rep = compat(&meta(), &r);
        assert!(!rep.has_errors());
        assert!(rep.warnings().any(|m| m.field == "git_commit"));
    }

    #[test]
    fn arch_and_tokenizer_mismatch_are_errors() {
        let mut r = run();
        r.arch_id = 7;
        r.tokenizer_sha256 = Some("other".into());
        let rep = compat(&meta(), &r);
        assert!(rep.has_errors());
        assert!(rep.errors().any(|m| m.field == "arch_id"));
        assert!(rep.errors().any(|m| m.field == "tokenizer_sha256"));
    }

    #[test]
    fn differing_config_flag_warns() {
        // The GQA4/direct-attn read-mismatch class: same code, different flag.
        let mut r = run();
        r.config.fp32_gqa4_attn = !r.config.fp32_gqa4_attn;
        let rep = compat(&meta(), &r);
        assert!(rep.warnings().any(|m| m.field == "config.fp32_gqa4_attn"));
    }

    #[test]
    fn cross_arch_warns() {
        let mut r = run();
        r.gpu_arch = "gfx1103".into();
        assert!(compat(&meta(), &r)
            .warnings()
            .any(|m| m.field == "gpu_arch"));
    }

    #[test]
    fn absent_provenance_does_not_warn() {
        // A ref with no git_commit recorded shouldn't spuriously warn.
        let mut me = meta();
        me.producer.git_commit = None;
        let mut r = run();
        r.git_commit = Some("zzzz".into());
        assert!(!compat(&me, &r)
            .mismatches
            .iter()
            .any(|m| m.field == "git_commit"));
    }
}
