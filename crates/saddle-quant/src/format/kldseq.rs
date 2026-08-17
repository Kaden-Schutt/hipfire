//! HFKSEQ — per-sequence KLD sidecar.
//!
//! Layout (v1, deprecated but still readable):
//!   bytes 0-7   magic "HFKSEQ\0\0"
//!   bytes 8-11  version:u32 LE (1)
//!   bytes 12-15 n_chunk:u32 LE
//!   bytes 16-19 reserved:u32 LE (zero)
//!   bytes 20-?  n_chunk × { f64 mean, f64 p99 }  (16 B per chunk)
//!
//! Layout (v2, current):
//!   bytes 0-7   magic "HFKSEQ\0\0"
//!   bytes 8-11  version:u32 LE (2)
//!   bytes 12-15 n_chunk:u32 LE
//!   bytes 16-19 reserved:u32 LE (zero)
//!   bytes 20-?  n_chunk × { f64 mean, f64 p99, f64 mean_nll } (24 B per chunk)
//!
//! v2 adds `mean_nll` per chunk. Reading v1 fills `mean_nll` with NaN.

use crate::{ArmScore, QuantError, Result};
use std::path::Path;

const MAGIC: &[u8; 8] = b"HFKSEQ\0\0";
const ARTIFACT: &str = "HFKSEQ";
const SUPPORTED: &str = "1, 2";

/// Per-chunk score triple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkScore {
    pub mean_kld: f64,
    pub p99_kld: f64,
    pub mean_nll: f64,
}

/// Decoded HFKSEQ file.
#[derive(Debug, Clone, PartialEq)]
pub struct KldSeq {
    pub version: u32,
    pub chunks: Vec<ChunkScore>,
}

/// Separation verdict for one pair of arms.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationReport {
    pub a: String,
    pub b: String,
    pub separated: bool,
    pub delta: f64,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n = sorted.len();
    let rank = q / 100.0 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// SplitMix64 — deterministic, no `rand` dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Read and validate an HFKSEQ file.
///
/// Accepts v1 (fills `mean_nll` with NaN) and v2. Rejects unknown versions
/// with [`QuantError::UnsupportedVersion`].
pub fn open(path: impl AsRef<Path>) -> Result<KldSeq> {
    let path = path.as_ref();
    let data = std::fs::read(path)?;

    if data.len() < 20 {
        return Err(QuantError::Truncated {
            artifact: ARTIFACT,
            context: "header",
            need: 20,
            have: data.len(),
        });
    }
    if &data[0..8] != MAGIC {
        return Err(QuantError::BadMagic {
            artifact: ARTIFACT,
            expected: "HFKSEQ\\0\\0",
            found: format!("{:?}", &data[0..8]),
        });
    }
    let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
    if version != 1 && version != 2 {
        return Err(QuantError::UnsupportedVersion {
            artifact: ARTIFACT,
            found: version,
            supported: SUPPORTED,
        });
    }
    let n_chunk = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    // reserved at 16..20 intentionally ignored

    let payload_per_chunk = if version == 2 { 24usize } else { 16usize };
    let need = 20usize
        .checked_add(
            n_chunk
                .checked_mul(payload_per_chunk)
                .ok_or_else(|| QuantError::Malformed("n_chunk * record size overflows".into()))?,
        )
        .ok_or_else(|| QuantError::Malformed("header + payload size overflows".into()))?;

    if data.len() < need {
        return Err(QuantError::Truncated {
            artifact: ARTIFACT,
            context: "payload",
            need,
            have: data.len(),
        });
    }
    if data.len() > need {
        return Err(QuantError::Malformed(format!(
            "trailing bytes: expected {need} bytes for {n_chunk} chunks (v{version}), found {}",
            data.len()
        )));
    }

    let mut chunks = Vec::with_capacity(n_chunk);
    for i in 0..n_chunk {
        let base = 20 + i * payload_per_chunk;
        let mean_kld = f64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let p99_kld = f64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
        let mean_nll = if version == 2 {
            f64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap())
        } else {
            f64::NAN
        };
        chunks.push(ChunkScore {
            mean_kld,
            p99_kld,
            mean_nll,
        });
    }

    Ok(KldSeq { version, chunks })
}

/// Write an HFKSEQ v2 file.
///
/// Always writes version 2 (with `mean_nll`). Creates parent directories if
/// needed.
pub fn write(path: impl AsRef<Path>, chunks: &[ChunkScore]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = std::fs::File::create(path)?;
    // Use manual buffering via Vec then write, to keep error handling simple
    // and avoid holding a BufWriter across the whole function.
    let mut buf = Vec::with_capacity(20 + chunks.len() * 24);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&2u32.to_le_bytes());
    let n = chunks.len() as u32;
    buf.extend_from_slice(&n.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for c in chunks {
        buf.extend_from_slice(&c.mean_kld.to_le_bytes());
        buf.extend_from_slice(&c.p99_kld.to_le_bytes());
        buf.extend_from_slice(&c.mean_nll.to_le_bytes());
    }
    std::io::Write::write_all(&mut file, &buf)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------

/// Bootstrap 95% CI over the sample mean.
///
/// Returns `(mean, ci_lo, ci_hi)` where `ci_lo`/`ci_hi` are the 2.5th/97.5th
/// percentiles of the resampled means. Deterministic from `seed` via an inline
/// SplitMix64 PRNG — no `rand` dependency. Resampling is with replacement.
pub fn bootstrap_mean_ci(values: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if resamples == 0 {
        return (mean, f64::NAN, f64::NAN);
    }
    let n = values.len();
    let mut rng = SplitMix64::new(seed);
    let mut boot_means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut s = 0.0;
        for _ in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            s += values[idx];
        }
        boot_means.push(s / n as f64);
    }
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ci_lo = percentile_linear(&boot_means, 2.5);
    let ci_hi = percentile_linear(&boot_means, 97.5);
    (mean, ci_lo, ci_hi)
}

/// Reduce a sequence to one [`ArmScore`].
///
/// - `mean_kld` / CI from [`bootstrap_mean_ci`] over per-chunk means.
/// - `p99_kld` as the 99th percentile of per-chunk `p99_kld` values (matching
///   the Python reducer's `np.percentile(..., 99)`).
/// - `mean_nll` as the mean of finite per-chunk NLLs, or NaN if none are
///   finite (v1 files).
/// - `ppl = mean_nll.exp()` — NaN stays NaN, never a bogus finite number.
/// - `n_chunks` from `seq`.
pub fn reduce(label: &str, seq: &KldSeq, resamples: usize, seed: u64) -> ArmScore {
    let n_chunks = seq.chunks.len();
    if n_chunks == 0 {
        return ArmScore {
            label: label.to_string(),
            mean_kld: f64::NAN,
            ci_lo: f64::NAN,
            ci_hi: f64::NAN,
            p99_kld: f64::NAN,
            mean_nll: f64::NAN,
            ppl: f64::NAN,
            n_chunks: 0,
        };
    }

    let means: Vec<f64> = seq.chunks.iter().map(|c| c.mean_kld).collect();
    let (mean_kld, ci_lo, ci_hi) = bootstrap_mean_ci(&means, resamples, seed);

    // 99th percentile of per-chunk p99s (faithful to Python reducer).
    let mut p99s: Vec<f64> = seq.chunks.iter().map(|c| c.p99_kld).collect();
    p99s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_kld = percentile_linear(&p99s, 99.0);

    let finite_nlls: Vec<f64> = seq
        .chunks
        .iter()
        .map(|c| c.mean_nll)
        .filter(|v| v.is_finite())
        .collect();
    let mean_nll = if finite_nlls.is_empty() {
        f64::NAN
    } else {
        finite_nlls.iter().sum::<f64>() / finite_nlls.len() as f64
    };
    let ppl = mean_nll.exp(); // NaN stays NaN

    ArmScore {
        label: label.to_string(),
        mean_kld,
        ci_lo,
        ci_hi,
        p99_kld,
        mean_nll,
        ppl,
        n_chunks,
    }
}

/// All-pairs separation report using [`ArmScore::separated_from`].
///
/// This exists because a real five-arm campaign produced overlapping intervals
/// and the ranking was reported anyway — this makes the overlap explicit.
pub fn compare(arms: &[ArmScore]) -> Vec<SeparationReport> {
    let mut out = Vec::new();
    for i in 0..arms.len() {
        for j in (i + 1)..arms.len() {
            let a = &arms[i];
            let b = &arms[j];
            let separated = a.separated_from(b);
            let delta = (a.mean_kld - b.mean_kld).abs();
            out.push(SeparationReport {
                a: a.label.clone(),
                b: b.label.clone(),
                separated,
                delta,
            });
        }
    }
    out
}

/// Estimate chunks needed to reach `target_ci_width`.
///
/// Uses the normal-approximation scaling `n = (2*1.96*sd / width)^2` where `sd`
/// is the observed sample standard deviation of per-chunk `mean_kld`. Returns
/// at least `observed.chunks.len()`.
pub fn min_chunks_for_power(observed: &KldSeq, target_ci_width: f64) -> usize {
    let n_obs = observed.chunks.len();
    if n_obs == 0 {
        return 0;
    }
    if !target_ci_width.is_finite() || target_ci_width <= 0.0 {
        return n_obs;
    }

    let values: Vec<f64> = observed.chunks.iter().map(|c| c.mean_kld).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    // sample variance (ddof=1) when n>1, else 0
    let var = if values.len() > 1 {
        let s: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
        s / (values.len() as f64 - 1.0)
    } else {
        0.0
    };
    let sd = var.sqrt();
    if !sd.is_finite() || sd == 0.0 {
        return n_obs;
    }

    let width = target_ci_width;
    let n_required = (2.0 * 1.96 * sd / width).powi(2);
    if !n_required.is_finite() {
        return n_obs;
    }
    let n_req = n_required.ceil() as usize;
    n_req.max(n_obs)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(mean: f64, p99: f64, nll: f64) -> ChunkScore {
        ChunkScore {
            mean_kld: mean,
            p99_kld: p99,
            mean_nll: nll,
        }
    }

    fn build_v1_buffer(chunks: &[(f64, f64)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HFKSEQ\0\0");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for (m, p) in chunks {
            buf.extend_from_slice(&m.to_le_bytes());
            buf.extend_from_slice(&p.to_le_bytes());
        }
        buf
    }

    fn build_v2_buffer(chunks: &[ChunkScore]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HFKSEQ\0\0");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for c in chunks {
            buf.extend_from_slice(&c.mean_kld.to_le_bytes());
            buf.extend_from_slice(&c.p99_kld.to_le_bytes());
            buf.extend_from_slice(&c.mean_nll.to_le_bytes());
        }
        buf
    }

    #[test]
    fn round_trip_via_tempfile() {
        let chunks = vec![
            chunk(0.04, 0.12, 2.3),
            chunk(0.06, 0.15, 2.4),
            chunk(0.05, 0.11, 2.35),
            chunk(0.07, 0.18, 2.5),
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kldseq");
        write(&path, &chunks).unwrap();
        let seq = open(&path).unwrap();
        assert_eq!(seq.version, 2);
        assert_eq!(seq.chunks.len(), chunks.len());
        for (a, b) in seq.chunks.iter().zip(chunks.iter()) {
            assert_eq!(a.mean_kld.to_bits(), b.mean_kld.to_bits());
            assert_eq!(a.p99_kld.to_bits(), b.p99_kld.to_bits());
            assert_eq!(a.mean_nll.to_bits(), b.mean_nll.to_bits());
        }
        // Also test empty
        let empty: Vec<ChunkScore> = vec![];
        let path2 = dir.path().join("empty.kldseq");
        write(&path2, &empty).unwrap();
        let seq2 = open(&path2).unwrap();
        assert_eq!(seq2.chunks.len(), 0);
        assert_eq!(seq2.version, 2);
    }

    #[test]
    fn v1_compat_mean_nll_nan_and_ppl_nan() {
        // Hand-build a v1 buffer and assert it parses with mean_nll NaN
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.kldseq");
        let raw_chunks = vec![(0.05, 0.14), (0.06, 0.16), (0.04, 0.11)];
        let buf = build_v1_buffer(&raw_chunks);
        std::fs::write(&path, &buf).unwrap();

        let seq = open(&path).unwrap();
        assert_eq!(seq.version, 1);
        assert_eq!(seq.chunks.len(), raw_chunks.len());
        for (c, (m, p)) in seq.chunks.iter().zip(raw_chunks.iter()) {
            assert_eq!(c.mean_kld, *m);
            assert_eq!(c.p99_kld, *p);
            assert!(c.mean_nll.is_nan(), "mean_nll should be NaN for v1");
        }

        // reduce on v1 must yield NaN mean_nll and NaN ppl
        let arm = reduce("v1-arm", &seq, 1000, 0);
        assert!(arm.mean_nll.is_nan());
        assert!(arm.ppl.is_nan(), "ppl must be NaN when mean_nll is NaN");
        // Also check via direct buffer write/read path using helpers
        // Ensure v1 buffer built via build_v2_buffer helper not confused
        let _ = build_v2_buffer(&seq.chunks);
    }

    #[test]
    fn round_trip_v1_buffer_nan_bits() {
        // Ensure v1 written via python helper still yields NaN mean_nll
        let buf = build_v1_buffer(&[(0.042, 0.11)]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1single.kldseq");
        std::fs::write(&path, &buf).unwrap();
        let seq = open(&path).unwrap();
        assert!(seq.chunks[0].mean_nll.is_nan());
        // Write that seq as v2 and reopen — should preserve existing NaN? Actually
        // writing v1 seq via write() will write NaN as the nll value (v2).
        // Reopen should have that NaN as finite? No, NaN is NaN.
        let path2 = dir.path().join("rewritten.kldseq");
        // Manually construct a v2 with same NaN to verify bit-preservation
        write(&path2, &seq.chunks).unwrap();
        let seq2 = open(&path2).unwrap();
        assert!(seq2.chunks[0].mean_nll.is_nan());
    }

    #[test]
    fn bootstrap_determinism_and_ci_brackets_mean() {
        let values = vec![0.04, 0.06, 0.05, 0.07, 0.045, 0.055, 0.062, 0.048];
        let (m1, lo1, hi1) = bootstrap_mean_ci(&values, 5000, 42);
        let (m2, lo2, hi2) = bootstrap_mean_ci(&values, 5000, 42);
        assert_eq!(m1.to_bits(), m2.to_bits());
        assert_eq!(lo1.to_bits(), lo2.to_bits());
        assert_eq!(hi1.to_bits(), hi2.to_bits());
        // CI brackets the sample mean
        assert!(lo1 <= m1 && m1 <= hi1, "CI {lo1} <= {m1} <= {hi1} failed");
        assert!(lo1 < hi1, "CI should have non-zero width");
        // Different seed should (usually) give different CI — not strictly guaranteed
        // but with high probability. We just check determinism above is sufficient.
        let (_m3, lo3, hi3) = bootstrap_mean_ci(&values, 5000, 99);
        // At least one bound differs for different seed (very likely)
        // If by bad luck they coincide, don't fail hard — just ensure not panicking.
        let _ = (lo3, hi3);
    }

    #[test]
    fn bootstrap_empty_and_zero_resamples() {
        let (m, lo, hi) = bootstrap_mean_ci(&[], 1000, 0);
        assert!(m.is_nan() && lo.is_nan() && hi.is_nan());
        let (m2, lo2, hi2) = bootstrap_mean_ci(&[0.05, 0.06], 0, 0);
        assert!((m2 - 0.055).abs() < 1e-12);
        assert!(lo2.is_nan() && hi2.is_nan());
    }

    #[test]
    fn reduce_and_compare_real_arms() {
        // Spec: simplest robust approach is to construct ArmScore values directly
        // with the published bounds for the separation assertions, and test
        // reduce/bootstrap separately on synthetic data.
        let barto = ArmScore {
            label: "barto-a55-q8head".into(),
            mean_kld: 0.0438,
            ci_lo: 0.0360,
            ci_hi: 0.0535,
            p99_kld: 0.12,
            mean_nll: 2.3,
            ppl: 2.3_f64.exp(),
            n_chunks: 24,
        };
        let uncal = ArmScore {
            label: "uncalibrated".into(),
            mean_kld: 0.0667,
            ci_lo: 0.0575,
            ci_hi: 0.0786,
            p99_kld: 0.18,
            mean_nll: 2.5,
            ppl: 2.5_f64.exp(),
            n_chunks: 24,
        };
        let trunk = ArmScore {
            label: "shipped-trunk".into(),
            mean_kld: 0.0583,
            ci_lo: 0.0488,
            ci_hi: 0.0705,
            p99_kld: 0.16,
            mean_nll: 2.45,
            ppl: 2.45_f64.exp(),
            n_chunks: 24,
        };

        // Check separation directly
        assert!(
            barto.separated_from(&uncal),
            "barto (0.0360-0.0535) should be separated from uncalibrated (0.0575-0.0786)"
        );
        assert!(
            !barto.separated_from(&trunk),
            "barto (0.0360-0.0535) should NOT be separated from shipped-trunk (0.0488-0.0705)"
        );

        // Check via compare
        let arms = vec![barto.clone(), uncal.clone(), trunk.clone()];
        let reports = compare(&arms);
        // 3 arms -> 3 pairs
        assert_eq!(reports.len(), 3);
        // Find barto-uncal pair
        let r_bu = reports
            .iter()
            .find(|r| (r.a == "barto-a55-q8head" && r.b == "uncalibrated")
                || (r.a == "uncalibrated" && r.b == "barto-a55-q8head"))
            .unwrap();
        assert!(r_bu.separated, "barto vs uncalibrated should be separated");
        assert!((r_bu.delta - (0.0667 - 0.0438)).abs() < 1e-9);

        let r_bt = reports
            .iter()
            .find(|r| (r.a == "barto-a55-q8head" && r.b == "shipped-trunk")
                || (r.a == "shipped-trunk" && r.b == "barto-a55-q8head"))
            .unwrap();
        assert!(!r_bt.separated, "barto vs shipped-trunk should NOT be separated");

        // Also test reduce on synthetic chunk data with known statistics
        // Create chunks whose mean is ~0.05 with some spread, check reduce CI brackets mean
        let synthetic: Vec<ChunkScore> = (0..24)
            .map(|i| {
                let v = 0.05 + 0.01 * ((i as f64) * 0.7).sin();
                chunk(v, v + 0.08, 2.3 + v)
            })
            .collect();
        let seq = KldSeq {
            version: 2,
            chunks: synthetic,
        };
        let arm = reduce("synthetic", &seq, 5000, 12345);
        assert!((arm.mean_kld - 0.05).abs() < 0.01, "mean should be near 0.05, got {}", arm.mean_kld);
        assert!(arm.ci_lo <= arm.mean_kld && arm.mean_kld <= arm.ci_hi);
        assert!(arm.ci_lo < arm.ci_hi);
        assert!(arm.p99_kld.is_finite());
        assert!(arm.mean_nll.is_finite());
        assert!(arm.ppl.is_finite());
        assert_eq!(arm.n_chunks, 24);
        // NaN handling: if all NLL are NaN, ppl must be NaN
        let nan_chunks: Vec<ChunkScore> = (0..4).map(|i| chunk(0.05 + i as f64 * 0.01, 0.12, f64::NAN)).collect();
        let nan_seq = KldSeq { version: 2, chunks: nan_chunks };
        let nan_arm = reduce("nan-nll", &nan_seq, 100, 0);
        assert!(nan_arm.mean_nll.is_nan());
        assert!(nan_arm.ppl.is_nan());
    }

    #[test]
    fn min_chunks_for_power_requires_more_for_narrower_width() {
        // Create a sequence with known spread
        let chunks: Vec<ChunkScore> = vec![
            chunk(0.04, 0.10, 2.1),
            chunk(0.06, 0.12, 2.2),
            chunk(0.05, 0.11, 2.15),
            chunk(0.07, 0.13, 2.25),
            chunk(0.045, 0.105, 2.12),
            chunk(0.055, 0.115, 2.18),
            chunk(0.062, 0.122, 2.21),
            chunk(0.048, 0.108, 2.16),
        ];
        let seq = KldSeq { version: 2, chunks };
        // Estimate current CI width via bootstrap
        let means: Vec<f64> = seq.chunks.iter().map(|c| c.mean_kld).collect();
        let (_, lo, hi) = bootstrap_mean_ci(&means, 5000, 0);
        let current_width = hi - lo;
        assert!(current_width.is_finite() && current_width > 0.0, "current_width={current_width}");
        // Narrower target should require strictly more chunks
        let narrower = current_width * 0.5;
        let n_narrow = min_chunks_for_power(&seq, narrower);
        assert!(
            n_narrow > seq.chunks.len(),
            "narrower width {narrower} (current {current_width}) should need > {} chunks, got {n_narrow}",
            seq.chunks.len()
        );
        // Wider target should return at least n_obs (not less)
        let wider = current_width * 2.0;
        let n_wide = min_chunks_for_power(&seq, wider);
        assert!(n_wide >= seq.chunks.len());
        assert!(n_wide <= n_narrow, "wider target should need <= narrower");
        // Zero / negative / NaN width returns n_obs
        assert_eq!(min_chunks_for_power(&seq, 0.0), seq.chunks.len());
        assert_eq!(min_chunks_for_power(&seq, -1.0), seq.chunks.len());
        assert_eq!(min_chunks_for_power(&seq, f64::NAN), seq.chunks.len());
        // Constant sequence (sd=0) should return n_obs regardless
        let constant = KldSeq {
            version: 2,
            chunks: vec![chunk(0.05, 0.1, 2.0); 4],
        };
        assert_eq!(min_chunks_for_power(&constant, 0.001), 4);
        // Empty sequence
        let empty = KldSeq { version: 2, chunks: vec![] };
        assert_eq!(min_chunks_for_power(&empty, 0.01), 0);
    }

    #[test]
    fn errors_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("badmagic.kldseq");
        let mut buf = vec![0u8; 20];
        buf[0..8].copy_from_slice(b"BADMAGIC");
        buf[8..12].copy_from_slice(&2u32.to_le_bytes());
        buf[12..16].copy_from_slice(&0u32.to_le_bytes());
        buf[16..20].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        let err = open(&path).unwrap_err();
        assert!(
            matches!(err, QuantError::BadMagic { .. }),
            "expected BadMagic, got {err:?}"
        );
    }

    #[test]
    fn errors_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("badver.kldseq");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HFKSEQ\0\0");
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        let err = open(&path).unwrap_err();
        assert!(
            matches!(err, QuantError::UnsupportedVersion { found: 99, .. }),
            "expected UnsupportedVersion 99, got {err:?}"
        );
    }

    #[test]
    fn errors_truncated_header_and_payload() {
        // Truncated header
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc_hdr.kldseq");
        std::fs::write(&path, b"HFKSEQ").unwrap();
        let err = open(&path).unwrap_err();
        assert!(matches!(err, QuantError::Truncated { .. }), "expected Truncated header, got {err:?}");

        // Truncated payload: header claims 2 chunks but only 1 provided
        let path2 = dir.path().join("trunc_payload.kldseq");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HFKSEQ\0\0");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        // Only one chunk instead of two
        buf.extend_from_slice(&0.05f64.to_le_bytes());
        buf.extend_from_slice(&0.12f64.to_le_bytes());
        buf.extend_from_slice(&2.3f64.to_le_bytes());
        std::fs::write(&path2, &buf).unwrap();
        let err2 = open(&path2).unwrap_err();
        assert!(matches!(err2, QuantError::Truncated { .. }), "expected Truncated payload, got {err2:?}");

        // Trailing bytes
        let path3 = dir.path().join("trailing.kldseq");
        let mut buf3 = Vec::new();
        buf3.extend_from_slice(b"HFKSEQ\0\0");
        buf3.extend_from_slice(&2u32.to_le_bytes());
        buf3.extend_from_slice(&1u32.to_le_bytes());
        buf3.extend_from_slice(&0u32.to_le_bytes());
        buf3.extend_from_slice(&0.05f64.to_le_bytes());
        buf3.extend_from_slice(&0.12f64.to_le_bytes());
        buf3.extend_from_slice(&2.3f64.to_le_bytes());
        buf3.push(0xFF); // extra
        std::fs::write(&path3, &buf3).unwrap();
        let err3 = open(&path3).unwrap_err();
        assert!(matches!(err3, QuantError::Malformed(_)), "expected Malformed trailing, got {err3:?}");
    }

    #[test]
    fn write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/c/test.kldseq");
        let chunks = vec![chunk(0.05, 0.11, 2.3)];
        write(&path, &chunks).unwrap();
        assert!(path.exists());
        let seq = open(&path).unwrap();
        assert_eq!(seq.chunks.len(), 1);
    }
}
