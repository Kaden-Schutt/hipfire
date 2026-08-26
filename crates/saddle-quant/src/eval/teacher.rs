//! Teacher plausibility gate.
//!
//! During the 2026-08-16 campaign a **gemma4-12b BF16 teacher** — a lossless
//! passthrough of the upstream parent — scored **PPL 1392.33** on WikiText-2
//! (NLL 7.238736) and **PPL 230.91** on the AG slice, while the qwen3.8-27b
//! teacher on the identical harness scored **PPL 6.2385** (NLL 1.830742).
//! The gemma4 reference was structurally valid (correct magic, geometry, block
//! count) but silently useless, and every artifact calibrated against it was
//! unvalidated.
//!
//! The diagnostic fingerprint: NLL 7.24 sits between a correct value (~2.3)
//! and uniform-over-vocabulary (`ln(262144) = 12.48`). That is a partially
//! broken forward pass — some pathway carrying signal while another is
//! broken — as opposed to a totally broken one that would sit at uniform.
//!
//! This module is the CPU oracle that a future GPU variant will be validated
//! against, and the real fallback for overnight runs. Precision and contract
//! clarity matter more than speed.

use crate::{Estimator, OracleStats, QuantError, Result, TeacherVerdict, WindowSpec};

// ---------------------------------------------------------------------------
// TeacherGate
// ---------------------------------------------------------------------------

/// Gate that decides whether a teacher's self-reported statistics are
/// plausible.
///
/// All thresholds are in natural-log / perplexity space (`ppl = exp(nll)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeacherGate {
    /// Vocabulary size of the teacher. Used to compute `uniform_nll = ln(n_vocab)`.
    pub n_vocab: usize,
    /// Perplexity at or above which the teacher is implausible.
    ///
    /// A healthy teacher on WikiText-2 is typically single-digit PPL; even a
    /// weak teacher is well below 50. A PPL of hundreds (gemma4 measured
    /// 1392) is a certain failure.
    pub max_plausible_ppl: f64,
    /// Perplexity at or below which the teacher is implausibly good.
    ///
    /// `ppl = exp(nll) >= 1`. A PPL near 1.0 means the scored tokens leaked
    /// into the context or the window is wrong. No real teacher is that good
    /// on held-out text.
    pub min_plausible_ppl: f64,
    /// How close to `ln(n_vocab)` is fatal.
    ///
    /// If `|mean_nll - ln(n_vocab)| < uniform_nll_margin` the model is
    /// emitting near-uniform predictions — a broken forward pass that is still
    /// numerically finite and may even have plausible-looking PPL on some
    /// corpora.
    pub uniform_nll_margin: f64,
}

impl Default for TeacherGate {
    fn default() -> Self {
        Self {
            // 0 signals "no vocab configured" — callers should use
            // `for_vocab`. Uniform checks are skipped when 0.
            n_vocab: 0,
            // PPL >= 100 is already absurd for a competent teacher on
            // WikiText-2 (qwen measured 6.24, gemma-broken measured 1392).
            // 100 is above the inconsistency test's 99.0 so that case
            // reaches the mismatch diagnostic instead of this one.
            max_plausible_ppl: 100.0,
            // PPL <= 1.5 is suspiciously good; PPL == 1.0 is perfect
            // prediction and cannot occur without leakage. 1.5 leaves room
            // for a very strong teacher (PPL ~2–3) while catching leakage.
            min_plausible_ppl: 1.5,
            // Within 0.5 nats of uniform is indistinguishable from a broken
            // path that is merely adding noise on top of a uniform base.
            uniform_nll_margin: 0.5,
        }
    }
}

impl TeacherGate {
    /// Construct a gate for a specific vocabulary size with otherwise
    /// default thresholds.
    pub fn for_vocab(n_vocab: usize) -> Self {
        Self {
            n_vocab,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// uniform_nll
// ---------------------------------------------------------------------------

/// `ln(n_vocab)`, the NLL of a uniform distribution over the vocabulary.
///
/// Returns `Err(QuantError::Malformed)` when `n_vocab == 0`.
pub fn uniform_nll(n_vocab: usize) -> Result<f64> {
    if n_vocab == 0 {
        return Err(QuantError::Malformed(
            "uniform_nll: n_vocab is 0".to_string(),
        ));
    }
    Ok((n_vocab as f64).ln())
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

/// Evaluate teacher oracle statistics against the gate.
///
/// Checks are ordered so the most specific / diagnostic reason wins:
///
/// 1. non-finite `mean_nll` or `ppl` (the `SCORE_FAIL` class)
/// 2. `n_scored == 0`
/// 3. `ppl >= max_plausible_ppl` (too bad — gemma4 class)
/// 4. `ppl <= min_plausible_ppl` (too good — leakage / window bug)
/// 5. `mean_nll` within `uniform_nll_margin` of `ln(n_vocab)` (near-uniform)
/// 6. `mean_nll` / `ppl` mismatch (`|ppl - exp(mean_nll)|` beyond tolerance)
pub fn evaluate(gate: &TeacherGate, stats: OracleStats) -> TeacherVerdict {
    // 1. non-finite — SCORE_FAIL class: a harness grep for `[0-9.]+` silently
    // dropped a NaN arm and it was reported as missing rather than broken.
    if !stats.mean_nll.is_finite() || !stats.ppl.is_finite() {
        return TeacherVerdict::Implausible {
            stats,
            reason: format!(
                "non-finite statistic: mean_nll={}, ppl={} (SCORE_FAIL)",
                stats.mean_nll, stats.ppl
            ),
        };
    }

    // 2. no scored positions
    if stats.n_scored == 0 {
        return TeacherVerdict::Implausible {
            stats,
            reason: format!("n_scored == 0: no tokens were scored (n_scored=0)"),
        };
    }

    // 3. PPL absurdly high
    if stats.ppl >= gate.max_plausible_ppl {
        return TeacherVerdict::Implausible {
            stats,
            reason: format!(
                "perplexity too high: ppl={:.4} >= max_plausible_ppl={:.4}",
                stats.ppl, gate.max_plausible_ppl
            ),
        };
    }

    // 4. PPL suspiciously low (leakage / window error)
    if stats.ppl <= gate.min_plausible_ppl {
        return TeacherVerdict::Implausible {
            stats,
            reason: format!(
                "perplexity too low: ppl={:.4} <= min_plausible_ppl={:.4} (possible context leakage or window error)",
                stats.ppl, gate.min_plausible_ppl
            ),
        };
    }

    // 5. near-uniform predictions
    if gate.n_vocab != 0 {
        if let Ok(uniform) = uniform_nll(gate.n_vocab) {
            let distance = (stats.mean_nll - uniform).abs();
            if distance < gate.uniform_nll_margin {
                return TeacherVerdict::Implausible {
                    stats,
                    reason: format!(
                        "near-uniform predictions: mean_nll={:.4} within {:.4} of uniform_nll={:.4} (ln(n_vocab={}))",
                        stats.mean_nll, gate.uniform_nll_margin, uniform, gate.n_vocab
                    ),
                };
            }
        }
    }

    // 6. mean_nll / ppl inconsistency — catches mis-derived or hand-edited
    // statistics. Relative tolerance 1e-3 (0.1%) comfortably exceeds rounding
    // of the measured pairs (error ~2e-6) but flags the 99.0 vs 6.24 case.
    let expected_ppl = stats.mean_nll.exp();
    // expected_ppl is finite because mean_nll is finite (checked above)
    if expected_ppl.is_finite() {
        let diff = (stats.ppl - expected_ppl).abs();
        // relative to expected; expected >= 1 for non-negative NLL but may be
        // <1 if NLL negative (should not happen for valid NLL), still handled.
        let denom = expected_ppl.abs().max(1e-12);
        let rel = diff / denom;
        const REL_TOL: f64 = 1e-3;
        // also guard absolute diff for very small expected: but expected is
        // at least ~1, so relative suffices. Include an absolute floor.
        if rel > REL_TOL && diff > 1e-9 {
            return TeacherVerdict::Implausible {
                stats,
                reason: format!(
                    "mean_nll/ppl mismatch: ppl={:.4} but exp(mean_nll)={:.4} (mean_nll={:.4}, rel_diff={:.4})",
                    stats.ppl, expected_ppl, stats.mean_nll, rel
                ),
            };
        }
    }

    TeacherVerdict::Plausible(stats)
}

// ---------------------------------------------------------------------------
// Degradation
// ---------------------------------------------------------------------------

/// Report of how a quantized candidate's PPL compares to its teacher.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegradationReport {
    /// Teacher (oracle) PPL.
    pub teacher_ppl: f64,
    /// Candidate (quantized) PPL.
    pub candidate_ppl: f64,
    /// `(candidate / teacher - 1) * 100` in percent. Negative means the
    /// candidate scored *below* its teacher.
    pub delta_pct: f64,
    /// Whether the candidate scored below its teacher.
    ///
    /// **This is NOT a quality win.** It was measured (AG slice, alpha=0.05
    /// arm at PPL 7.4737 against a 7.7852 oracle) and is within the range
    /// where quantization noise sharpens one realized sequence. Callers must
    /// report KLD, which is strictly positive, as the honest divergence
    /// figure.
    pub candidate_better: bool,
}

/// Compute PPL degradation of a candidate against its teacher.
///
/// `delta_pct = (candidate_ppl / teacher_ppl - 1) * 100`.
///
/// Returns `Err(Malformed)` if either PPL is non-finite or `<= 0`.
pub fn degradation(teacher: OracleStats, candidate_ppl: f64) -> Result<DegradationReport> {
    if !teacher.ppl.is_finite() || teacher.ppl <= 0.0 {
        return Err(QuantError::Malformed(format!(
            "degradation: teacher ppl must be finite and >0, got {}",
            teacher.ppl
        )));
    }
    if !candidate_ppl.is_finite() || candidate_ppl <= 0.0 {
        return Err(QuantError::Malformed(format!(
            "degradation: candidate ppl must be finite and >0, got {}",
            candidate_ppl
        )));
    }
    let delta_pct = (candidate_ppl / teacher.ppl - 1.0) * 100.0;
    let candidate_better = candidate_ppl < teacher.ppl;
    Ok(DegradationReport {
        teacher_ppl: teacher.ppl,
        candidate_ppl,
        delta_pct,
        candidate_better,
    })
}

// ---------------------------------------------------------------------------
// Cross-model comparability
// ---------------------------------------------------------------------------

/// Identity of a reference needed to decide whether two KLD numbers are
/// comparable.
///
/// KLD values are only comparable when they share vocabulary (normaliser),
/// estimator (truncation), window (scored positions), chunk count and corpus.
/// The `model` field is informational only and does not affect comparability.
// No `Eq`: `Estimator` carries an `Option<f64>` (`bias_vs_full`), so float
// equality rules it out. `PartialEq` is all comparability needs.
#[derive(Debug, Clone, PartialEq)]
pub struct RefIdentity {
    pub model: String,
    pub n_vocab: usize,
    pub estimator: Estimator,
    pub window: WindowSpec,
    pub n_chunk: usize,
    pub corpus_sha256: String,
}

/// Whether two references are comparable (same vocab, estimator, window,
/// chunk count and corpus digest).
///
/// This encodes a real error from the campaign: a KLD of 0.032464 on
/// muse-glimmer (n_vocab 202,048) was compared against 0.043776 on
/// qwen3.8-27b (n_vocab 248,320) as though one were better. Different
/// teacher, different vocabulary, no shared normaliser — not comparable.
pub fn cross_model_comparable(a: &RefIdentity, b: &RefIdentity) -> bool {
    a.n_vocab == b.n_vocab
        && a.estimator == b.estimator
        && a.window == b.window
        && a.n_chunk == b.n_chunk
        && a.corpus_sha256 == b.corpus_sha256
}

/// Name each mismatching field between two identities.
///
/// Only fields that participate in [`cross_model_comparable`] are reported;
/// `model` differences are ignored for comparability but are not listed here.
pub fn incomparability_reasons(a: &RefIdentity, b: &RefIdentity) -> Vec<String> {
    let mut reasons = Vec::new();
    if a.n_vocab != b.n_vocab {
        reasons.push(format!("n_vocab mismatch: {} vs {}", a.n_vocab, b.n_vocab));
    }
    if a.estimator != b.estimator {
        reasons.push(format!(
            "estimator mismatch: {:?} vs {:?}",
            a.estimator, b.estimator
        ));
    }
    if a.window != b.window {
        reasons.push(format!("window mismatch: {:?} vs {:?}", a.window, b.window));
    }
    if a.n_chunk != b.n_chunk {
        reasons.push(format!("n_chunk mismatch: {} vs {}", a.n_chunk, b.n_chunk));
    }
    if a.corpus_sha256 != b.corpus_sha256 {
        reasons.push(format!(
            "corpus_sha256 mismatch: {} vs {}",
            a.corpus_sha256, b.corpus_sha256
        ));
    }
    reasons
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gate_262144() -> TeacherGate {
        TeacherGate::for_vocab(262_144)
    }
    fn gate_248320() -> TeacherGate {
        TeacherGate::for_vocab(248_320)
    }

    #[test]
    fn uniform_nll_correct() {
        let u = uniform_nll(262_144).unwrap();
        assert!((u - 12.476649250079015).abs() < 1e-12, "u={u}");
        let u2 = uniform_nll(248_320).unwrap();
        assert!((u2 - 12.422473515976991).abs() < 1e-12, "u2={u2}");
    }

    #[test]
    fn uniform_nll_zero_is_malformed() {
        let err = uniform_nll(0).unwrap_err();
        match err {
            QuantError::Malformed(msg) => assert!(!msg.is_empty()),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn gemma_regression_implausible_mentions_perplexity() {
        let stats = OracleStats {
            mean_nll: 7.238736,
            ppl: 1392.33,
            n_scored: 24_552,
        };
        let verdict = evaluate(&gate_262144(), stats);
        match verdict {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("perplexity") || lower.contains("ppl"),
                    "reason must mention perplexity, got: {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("gemma4 must be Implausible"),
        }
    }

    #[test]
    fn qwen_real_plausible() {
        let stats = OracleStats {
            mean_nll: 1.830742,
            ppl: 6.2385,
            n_scored: 24_552,
        };
        let verdict = evaluate(&gate_248320(), stats);
        match verdict {
            TeacherVerdict::Plausible(s) => assert_eq!(s, stats),
            TeacherVerdict::Implausible { reason, .. } => {
                panic!("qwen must be Plausible, got Implausible: {reason}")
            }
        }
    }

    #[test]
    fn near_uniform_implausible_cites_uniformity() {
        let n_vocab = 262_144;
        let uniform = uniform_nll(n_vocab).unwrap();
        let stats = OracleStats {
            mean_nll: uniform - 0.1,
            ppl: (uniform - 0.1).exp(),
            n_scored: 1000,
        };
        // Use a gate where max is high enough not to trigger before uniform.
        let gate = TeacherGate {
            n_vocab,
            max_plausible_ppl: 1_000_000.0,
            min_plausible_ppl: 1.5,
            uniform_nll_margin: 0.5,
        };
        let verdict = evaluate(&gate, stats);
        match verdict {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("uniform"),
                    "reason must cite uniformity, got: {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("near-uniform must be Implausible"),
        }
    }

    #[test]
    fn near_uniform_far_not_triggered() {
        // 5.2 away from uniform (gemma distance) should NOT be considered uniform
        // when margin is 0.5 and ppl is plausible small.
        let n_vocab = 262_144;
        let gate = TeacherGate {
            n_vocab,
            max_plausible_ppl: 100.0,
            min_plausible_ppl: 1.5,
            uniform_nll_margin: 0.5,
        };
        // Construct a plausible small ppl but mean far from uniform: choose mean 2.0 ppl 7.389
        let stats = OracleStats {
            mean_nll: 2.0,
            ppl: 2.0_f64.exp(),
            n_scored: 1000,
        };
        // Should be plausible (not near uniform, ppl ok, consistent)
        match evaluate(&gate, stats) {
            TeacherVerdict::Plausible(_) => {}
            TeacherVerdict::Implausible { reason, .. } => {
                panic!("ppl 7.38 with mean 2.0 should be plausible, got {reason}")
            }
        }
    }

    #[test]
    fn nan_implausible_cites_non_finite() {
        let stats = OracleStats {
            mean_nll: f64::NAN,
            ppl: 6.2385,
            n_scored: 1000,
        };
        let verdict = evaluate(&gate_248320(), stats);
        match verdict {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("non-finite")
                        || lower.contains("non finite")
                        || lower.contains("finite"),
                    "reason must cite non-finite, got: {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("NaN mean_nll must be Implausible"),
        }

        // ppl NaN
        let stats2 = OracleStats {
            mean_nll: 1.8,
            ppl: f64::NAN,
            n_scored: 1000,
        };
        match evaluate(&gate_248320(), stats2) {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("finite") || lower.contains("non-finite"),
                    "got {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("NaN ppl must be Implausible"),
        }

        // inf
        let stats3 = OracleStats {
            mean_nll: f64::INFINITY,
            ppl: 6.0,
            n_scored: 1000,
        };
        match evaluate(&gate_248320(), stats3) {
            TeacherVerdict::Implausible { reason, .. } => {
                assert!(reason.to_lowercase().contains("finite"));
            }
            TeacherVerdict::Plausible(_) => panic!("inf must be Implausible"),
        }
    }

    #[test]
    fn inconsistency_implausible_cites_mismatch() {
        let stats = OracleStats {
            mean_nll: 1.830742,
            ppl: 99.0,
            n_scored: 1000,
        };
        // gate with max >99 so it reaches mismatch check
        let gate = gate_248320();
        assert!(gate.max_plausible_ppl > 99.0);
        let verdict = evaluate(&gate, stats);
        match verdict {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("mismatch") || lower.contains("inconsistent"),
                    "reason must cite mismatch, got: {reason}"
                );
                // should mention both quantities
                assert!(
                    lower.contains("mean_nll") || lower.contains("mean"),
                    "reason should mention mean_nll, got {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("mismatch must be Implausible"),
        }
    }

    #[test]
    fn zero_scored_implausible() {
        let stats = OracleStats {
            mean_nll: 1.830742,
            ppl: 6.2385,
            n_scored: 0,
        };
        match evaluate(&gate_248320(), stats) {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("n_scored") || lower.contains("0"),
                    "got {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("n_scored 0 must be Implausible"),
        }
    }

    #[test]
    fn low_ppl_implausible() {
        let stats = OracleStats {
            mean_nll: 0.1,
            ppl: 1.1051701859880925, // exp(0.1)
            n_scored: 1000,
        };
        // 1.105 < 1.5 min
        let verdict = evaluate(&gate_248320(), stats);
        match verdict {
            TeacherVerdict::Implausible { reason, .. } => {
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("perplexity") || lower.contains("ppl") || lower.contains("low"),
                    "got {reason}"
                );
            }
            TeacherVerdict::Plausible(_) => panic!("low ppl must be Implausible"),
        }
    }

    #[test]
    fn high_ppl_boundary() {
        // exactly at max should be implausible (at or above)
        let gate = gate_248320();
        let stats = OracleStats {
            mean_nll: gate.max_plausible_ppl.ln(),
            ppl: gate.max_plausible_ppl,
            n_scored: 1000,
        };
        match evaluate(&gate, stats) {
            TeacherVerdict::Implausible { .. } => {}
            TeacherVerdict::Plausible(_) => panic!("ppl == max must be Implausible"),
        }
        // just below max should be plausible if consistent
        let just_below = gate.max_plausible_ppl - 1.0;
        let stats2 = OracleStats {
            mean_nll: just_below.ln(),
            ppl: just_below,
            n_scored: 1000,
        };
        match evaluate(&gate, stats2) {
            TeacherVerdict::Plausible(_) => {}
            TeacherVerdict::Implausible { reason, .. } => {
                panic!("just below max should be plausible, got {reason}")
            }
        }
    }

    #[test]
    fn degradation_negative_and_positive() {
        let teacher = OracleStats {
            mean_nll: 2.052, // approx ln 7.7852
            ppl: 7.7852,
            n_scored: 1000,
        };
        let report = degradation(teacher, 7.4737).unwrap();
        assert!(
            report.candidate_better,
            "candidate 7.4737 < teacher 7.7852 should be better"
        );
        assert!(
            report.delta_pct < 0.0,
            "delta_pct negative, got {}",
            report.delta_pct
        );
        // compute expected delta ~ -4%
        let expected = (7.4737 / 7.7852 - 1.0) * 100.0;
        assert!((report.delta_pct - expected).abs() < 1e-9);

        let teacher2 = OracleStats {
            mean_nll: 1.830742,
            ppl: 6.2385,
            n_scored: 1000,
        };
        let report2 = degradation(teacher2, 6.4088).unwrap();
        assert!(!report2.candidate_better);
        // ~+2.73%
        assert!(
            (report2.delta_pct - 2.73).abs() < 0.01,
            "got {}",
            report2.delta_pct
        );
        let expected2 = (6.4088 / 6.2385 - 1.0) * 100.0;
        assert!((report2.delta_pct - expected2).abs() < 1e-12);
    }

    #[test]
    fn degradation_errors() {
        let good = OracleStats {
            mean_nll: 1.5,
            ppl: 4.48,
            n_scored: 10,
        };
        // candidate non-finite
        assert!(matches!(
            degradation(good, f64::NAN).unwrap_err(),
            QuantError::Malformed(_)
        ));
        assert!(matches!(
            degradation(good, f64::INFINITY).unwrap_err(),
            QuantError::Malformed(_)
        ));
        assert!(matches!(
            degradation(good, 0.0).unwrap_err(),
            QuantError::Malformed(_)
        ));
        assert!(matches!(
            degradation(good, -1.0).unwrap_err(),
            QuantError::Malformed(_)
        ));
        // teacher bad
        let bad_teacher = OracleStats {
            mean_nll: 1.0,
            ppl: f64::NAN,
            n_scored: 10,
        };
        assert!(matches!(
            degradation(bad_teacher, 6.0).unwrap_err(),
            QuantError::Malformed(_)
        ));
        let zero_teacher = OracleStats {
            mean_nll: 0.0,
            ppl: 0.0,
            n_scored: 10,
        };
        assert!(matches!(
            degradation(zero_teacher, 6.0).unwrap_err(),
            QuantError::Malformed(_)
        ));
    }

    #[test]
    fn cross_comparable_vocab_mismatch() {
        let base = RefIdentity {
            model: "qwen3-27b".to_string(),
            n_vocab: 248_320,
            estimator: Estimator::FullVocab,
            window: WindowSpec {
                warmup: 0,
                score_from: 1024,
                score_to: 2048,
                carry_kv: false,
            },
            n_chunk: 24,
            corpus_sha256: "abc123".to_string(),
        };
        let mut other = base.clone();
        other.n_vocab = 202_048;
        assert!(!cross_model_comparable(&base, &other));
        let reasons = incomparability_reasons(&base, &other);
        assert!(
            reasons.iter().any(|r| r.to_lowercase().contains("vocab")),
            "reasons must name vocab, got {reasons:?}"
        );
        // identical returns true
        assert!(cross_model_comparable(&base, &base));
        assert!(incomparability_reasons(&base, &base).is_empty());
    }

    #[test]
    fn cross_comparable_all_fields() {
        let base = RefIdentity {
            model: "model-a".to_string(),
            n_vocab: 248_320,
            estimator: Estimator::FullVocab,
            window: WindowSpec {
                warmup: 0,
                score_from: 0,
                score_to: 1024,
                carry_kv: true,
            },
            n_chunk: 24,
            corpus_sha256: "deadbeef".to_string(),
        };
        // different estimator
        let mut e = base.clone();
        e.estimator = Estimator::TopK {
            k: 256,
            bias_vs_full: None,
        };
        assert!(!cross_model_comparable(&base, &e));
        assert!(incomparability_reasons(&base, &e)
            .iter()
            .any(|r| r.to_lowercase().contains("estimator")));

        // different window
        let mut w = base.clone();
        w.window = WindowSpec {
            warmup: 0,
            score_from: 512,
            score_to: 1024,
            carry_kv: true,
        };
        assert!(!cross_model_comparable(&base, &w));
        assert!(incomparability_reasons(&base, &w)
            .iter()
            .any(|r| r.to_lowercase().contains("window")));

        // different n_chunk
        let mut c = base.clone();
        c.n_chunk = 48;
        assert!(!cross_model_comparable(&base, &c));
        assert!(incomparability_reasons(&base, &c)
            .iter()
            .any(|r| r.to_lowercase().contains("n_chunk")));

        // different corpus
        let mut d = base.clone();
        d.corpus_sha256 = "different".to_string();
        assert!(!cross_model_comparable(&base, &d));
        assert!(incomparability_reasons(&base, &d)
            .iter()
            .any(|r| r.to_lowercase().contains("corpus")));

        // model name alone does NOT make incomparable (informational only)
        let mut m = base.clone();
        m.model = "different-model".to_string();
        assert!(cross_model_comparable(&base, &m));
        assert!(incomparability_reasons(&base, &m).is_empty());
    }

    #[test]
    fn gate_defaults_and_for_vocab() {
        let d = TeacherGate::default();
        assert!(d.max_plausible_ppl > d.min_plausible_ppl);
        assert!(d.max_plausible_ppl > 10.0);
        assert!(d.min_plausible_ppl >= 1.0 && d.min_plausible_ppl < 5.0);
        assert!(d.uniform_nll_margin > 0.0 && d.uniform_nll_margin < 5.0);

        let g = TeacherGate::for_vocab(42_000);
        assert_eq!(g.n_vocab, 42_000);
        assert_eq!(g.max_plausible_ppl, d.max_plausible_ppl);
        assert_eq!(g.min_plausible_ppl, d.min_plausible_ppl);
    }

    #[test]
    fn check_order_non_finite_wins_over_ppl() {
        // NaN ppl that would also be "too high" should still cite non-finite
        let stats = OracleStats {
            mean_nll: f64::NAN,
            ppl: 9999.0,
            n_scored: 1000,
        };
        match evaluate(&gate_248320(), stats) {
            TeacherVerdict::Implausible { reason, .. } => {
                assert!(reason.to_lowercase().contains("finite"));
            }
            _ => panic!("must be Implausible"),
        }
    }

    #[test]
    fn check_order_n_scored_wins_over_ppl() {
        let stats = OracleStats {
            mean_nll: 1.8,
            ppl: 6.0,
            n_scored: 0,
        };
        match evaluate(&gate_248320(), stats) {
            TeacherVerdict::Implausible { reason, .. } => {
                assert!(reason.to_lowercase().contains("n_scored"));
            }
            _ => panic!("must be Implausible"),
        }
    }
}
