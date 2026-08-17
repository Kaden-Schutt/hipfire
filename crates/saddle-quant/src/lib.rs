//! saddle-quant — quantization-quality toolkit.
//!
//! # Why this crate exists
//!
//! The artifact formats (HFQ/HFQM containers, HFHS Hessians, GGUF imatrices,
//! KLD references, score sequences) had no owner. `hipfire-quantize` is
//! primarily a binary; its `lib.rs` exports `hfqm`/`hessian_io`/`hfhs_diag`
//! but not `gguf_input`, and no crate outside it takes the dependency. The
//! result, measured 2026-08-16: **29 files** across `crates/` and `scripts/`
//! independently parse the HFQ header, and the tree has already had to delete
//! redundant parsers by hand (`3dfd1b3f5`, "third redundant HFQ parser").
//!
//! This crate owns each format exactly once.
//!
//! # Methodology defects this crate exists to fix
//!
//! The previous KLD reference pipeline (`build_kld_ref_native` +
//! `eval_hipfire`) produced numbers that could not support the claims made
//! from them. Enumerated so the replacement is written against a spec:
//!
//! 1. **Truncated estimator, unvalidated.** `top_k=256` over a 248,320-token
//!    vocabulary retains 0.103% of the distribution, with residual mass folded
//!    in by approximation. Never validated against full-vocab, and therefore
//!    not comparable to llama.cpp's `--kl-divergence`.
//! 2. **Arbitrary scoring window.** Only positions `[n_ctx/2, n_ctx-1)` were
//!    scored; the first half of every chunk was discarded by construction.
//! 3. **No context depth.** The KV cache was cleared per chunk, so 24
//!    independent 2048-token windows were measured and nothing beyond 2K —
//!    while `kv-mode q8`, the term most likely to degrade with depth, was on.
//! 4. **Unvalidated teacher.** The oracle is hipfire's own forward. A gemma4
//!    teacher scored PPL 1392 on WikiText-2 (a lossless BF16 passthrough of
//!    the upstream parent) and still produced a structurally valid reference.
//!    A broken teacher silently yields a broken reference.
//! 5. **Provenance not stored.** Oracle NLL/PPL were `eprintln`'d at build
//!    time, never written to the header, so `muse-glimmer`'s oracle is
//!    unrecoverable and its KLD cannot be normalized against any other model.
//! 6. **No integrity gate.** Every scoring run logged
//!    `manifest.json missing; skipping ref sha256 check`.
//! 7. **Underpowered.** At `n_chunk=24` the bootstrap 95% CIs of five arms
//!    spanning 0.044–0.067 overlapped; only one pair separated.
//!
//! Accordingly: [`Estimator`] records how a reference was estimated,
//! [`WindowSpec`] makes the scored range explicit rather than implied,
//! [`OracleStats`] is stored in the header, [`ArtifactId`] carries a digest
//! for every input, and reduction reports intervals rather than point
//! estimates.

pub mod corpus;
pub mod eval;
pub mod format;
pub mod stats;

/// Content-addressed identity for any input or output artifact.
///
/// Every reference records the digest of its teacher and its corpus so a
/// score can be refused when either drifts, instead of silently comparing
/// against the wrong distribution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactId {
    /// Path as supplied at build time. Informational only — never trusted for
    /// identity, since local filenames drift across boxes.
    pub path: String,
    /// Lowercase hex SHA-256 of the full byte stream. This is the identity.
    pub sha256: String,
    pub bytes: u64,
}

/// How a reference distribution was estimated.
///
/// Recorded in the header because a top-k reference and a full-vocab
/// reference are different measurements, and a number from one must never be
/// compared against a number from the other.
// No `Eq`: `bias_vs_full` is an `Option<f64>` and float equality is not an
// equivalence relation. `PartialEq` is enough for the mismatch check.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Estimator {
    /// Exact: every vocabulary entry retained.
    FullVocab,
    /// Top-`k` retained plus a single aggregated residual-mass term.
    ///
    /// `bias_vs_full` is the measured mean-KLD delta against a `FullVocab`
    /// reference over the same tokens, or `None` when never calibrated. An
    /// uncalibrated truncated estimator is reportable but not comparable.
    TopK { k: u32, bias_vs_full: Option<f64> },
}

/// The scored token range within each chunk, stated explicitly.
///
/// The previous pipeline hardcoded `score_from = n_ctx / 2` with no way to
/// express or record any other choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WindowSpec {
    /// Leading positions excluded from scoring so the model has context.
    pub warmup: usize,
    /// First scored position (inclusive), counted from chunk start.
    pub score_from: usize,
    /// Last scored position (exclusive).
    pub score_to: usize,
    /// Whether KV state carries across chunks. `false` reproduces the legacy
    /// per-chunk reset and measures no depth beyond `n_ctx`.
    pub carry_kv: bool,
}

impl WindowSpec {
    /// Scored positions per chunk.
    pub fn scored_per_chunk(&self) -> usize {
        self.score_to.saturating_sub(self.score_from)
    }

    /// The legacy back-half window, for reproducing historical numbers only.
    pub fn legacy_half(n_ctx: usize) -> Self {
        Self { warmup: 0, score_from: n_ctx / 2, score_to: n_ctx - 1, carry_kv: false }
    }
}

/// Teacher-oracle statistics over its own reference tokens.
///
/// Stored in the header. This is what makes cross-model comparison possible:
/// a candidate's KLD is only interpretable next to the divergence its own
/// teacher achieves, and PPL degradation needs the teacher's PPL as a
/// denominator.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OracleStats {
    pub mean_nll: f64,
    pub ppl: f64,
    pub n_scored: usize,
}

/// Verdict from the teacher plausibility gate.
///
/// Exists because a BF16 passthrough of an upstream parent once scored PPL
/// 1392 and produced a reference that looked structurally perfect.
#[derive(Debug, Clone, PartialEq)]
pub enum TeacherVerdict {
    Plausible(OracleStats),
    Implausible { stats: OracleStats, reason: String },
}

/// One arm's reduced score with an uncertainty interval.
///
/// Point estimates are not a reportable result: at `n_chunk=24` five arms
/// spanning 0.044–0.067 mean KLD had overlapping 95% intervals.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArmScore {
    pub label: String,
    pub mean_kld: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub p99_kld: f64,
    pub mean_nll: f64,
    pub ppl: f64,
    pub n_chunks: usize,
}

impl ArmScore {
    /// Whether two arms are distinguishable at the reported interval, i.e.
    /// their confidence intervals do not overlap. A ranking between
    /// non-separated arms is not supported by the measurement.
    pub fn separated_from(&self, other: &ArmScore) -> bool {
        self.ci_hi < other.ci_lo || other.ci_hi < self.ci_lo
    }
}

/// Errors surfaced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{artifact}: bad magic (expected {expected:?}, found {found:?})")]
    BadMagic { artifact: &'static str, expected: &'static str, found: String },
    #[error("{artifact}: unsupported version {found} (supported: {supported})")]
    UnsupportedVersion { artifact: &'static str, found: u32, supported: &'static str },
    #[error("{artifact}: truncated at {context} (need {need} bytes, have {have})")]
    Truncated { artifact: &'static str, context: &'static str, need: usize, have: usize },
    #[error("digest mismatch for {what}: reference recorded {expected}, found {found}")]
    DigestMismatch { what: String, expected: String, found: String },
    #[error("estimator mismatch: reference is {reference:?}, candidate scored as {candidate:?}")]
    EstimatorMismatch { reference: Estimator, candidate: Estimator },
    #[error("malformed: {0}")]
    Malformed(String),
}

pub type Result<T, E = QuantError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_scored_count_matches_legacy_pipeline() {
        // The historical pipeline reported scored/chunk=1023 at n_ctx=2048.
        let w = WindowSpec::legacy_half(2048);
        assert_eq!(w.scored_per_chunk(), 1023);
        assert_eq!(w.score_from, 1024);
        assert!(!w.carry_kv, "legacy window cleared KV per chunk");
    }

    #[test]
    fn separation_is_symmetric_and_rejects_overlap() {
        let arm = |label: &str, lo: f64, hi: f64| ArmScore {
            label: label.into(),
            mean_kld: (lo + hi) / 2.0,
            ci_lo: lo,
            ci_hi: hi,
            p99_kld: 0.0,
            mean_nll: 0.0,
            ppl: 0.0,
            n_chunks: 24,
        };
        // Measured 2026-08-16: best arm vs uncalibrated DID separate.
        let best = arm("barto-a55-q8head", 0.0360, 0.0535);
        let uncal = arm("uncalibrated", 0.0575, 0.0786);
        assert!(best.separated_from(&uncal));
        assert!(uncal.separated_from(&best), "separation must be symmetric");

        // ...but best vs the shipped trunk did NOT, and must not be ranked.
        let trunk = arm("shipped-trunk", 0.0488, 0.0705);
        assert!(!best.separated_from(&trunk));
        assert!(!trunk.separated_from(&best));
    }
}
