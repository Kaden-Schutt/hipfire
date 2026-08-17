// SPDX-License-Identifier: Apache-2.0
//! Activation-statistics diagnostics — diffing `Σx²` (imatrix `in_sum2`) and
//! `diag(H)` (Hessian diagonal) vectors.
//!
//! Both are per-input-channel sums of squared activations, length `K` per
//! tensor, and should agree up to a global scale for the same corpus and tap
//! point. Where they do not, the corpus or tap is mis-specified.
//!
//! AWQ consumes only the diagonal via `s[j] = sqrt(in_sum2[j])^alpha`,
//! geo-mean normalized to 1, so rank ordering (Spearman) matters more than
//! magnitude.

use crate::format::imatrix::{safetensors_to_ggml_name, Imatrix};
use crate::{QuantError, Result};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// VectorStats
// ---------------------------------------------------------------------------

/// Summary statistics for one `in_sum2` / `diag(H)` vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorStats {
    pub k: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub zeros: usize,
    pub nonfinite: usize,
}

/// Describe a single vector.
///
/// * `zeros` counts exact `0.0` (including `-0.0`).
/// * `nonfinite` counts `NaN` / `±inf`.
/// * `sum` / `mean` / `min` / `max` are computed over finite entries only;
///   if there are no finite entries they are `0.0`.
pub fn describe(v: &[f32]) -> VectorStats {
    let k = v.len();
    let mut zeros = 0usize;
    let mut nonfinite = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut finite_count = 0usize;

    for &x in v {
        if !x.is_finite() {
            nonfinite += 1;
            continue;
        }
        if x == 0.0 {
            zeros += 1;
        }
        let xd = x as f64;
        sum += xd;
        if xd < min {
            min = xd;
        }
        if xd > max {
            max = xd;
        }
        finite_count += 1;
    }

    if finite_count == 0 {
        VectorStats {
            k,
            sum: 0.0,
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            zeros,
            nonfinite,
        }
    } else {
        VectorStats {
            k,
            sum,
            mean: sum / finite_count as f64,
            min,
            max,
            zeros,
            nonfinite,
        }
    }
}

// ---------------------------------------------------------------------------
// VectorComparison
// ---------------------------------------------------------------------------

/// Pairwise comparison of two length-`K` vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorComparison {
    pub k: usize,
    pub pearson: f64,
    pub spearman: f64,
    pub scale_ratio: f64,
    pub cosine: f64,
    pub max_rel_delta: f64,
    pub disagreeing_channels: usize,
}

fn pearson_corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den == 0.0 {
        if den_a == 0.0 && den_b == 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    let r = num / den;
    r.clamp(-1.0, 1.0)
}

fn ranks(v: &[f32]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        // Use total ordering via bits for determinism; NaN will sort last
        let ai = v[i];
        let aj = v[j];
        match ai.partial_cmp(&aj) {
            Some(o) => o,
            None => {
                // One or both NaN — order by is_nan then bits
                let a_nan = ai.is_nan();
                let b_nan = aj.is_nan();
                match (a_nan, b_nan) {
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    _ => ai.to_bits().cmp(&aj.to_bits()),
                }
            }
        }
    });
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Group ties: exact f32 equality (includes -0.0 == 0.0)
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Average rank of positions i+1 ..= j (1-indexed)
        let avg = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

fn spearman_corr(a: &[f32], b: &[f32]) -> f64 {
    if a.len() <= 1 {
        return 1.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let n = ra.len() as f64;
    let mean_ra = ra.iter().sum::<f64>() / n;
    let mean_rb = rb.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..ra.len() {
        let da = ra[i] - mean_ra;
        let db = rb[i] - mean_rb;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den == 0.0 {
        if den_a == 0.0 && den_b == 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    (num / den).clamp(-1.0, 1.0)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        if na == 0.0 && nb == 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Compare two equal-length vectors.
///
/// Returns `Err(QuantError::Malformed)` on length mismatch.
/// `rel_tol` controls `disagreeing_channels`: a channel `j` disagrees when
/// `|a[j]-b[j]| / max(|a[j]|,|b[j]|) > rel_tol`.  Zero/zero gives delta 0;
/// any non-finite mismatch gives `inf`.
pub fn compare_vectors(a: &[f32], b: &[f32], rel_tol: f64) -> Result<VectorComparison> {
    if a.len() != b.len() {
        return Err(QuantError::Malformed(format!(
            "compare_vectors: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let k = a.len();
    if k == 0 {
        return Err(QuantError::Malformed(
            "compare_vectors: empty vectors".to_string(),
        ));
    }

    let pearson = pearson_corr(a, b);
    let spearman = spearman_corr(a, b);
    let cosine = cosine_sim(a, b);

    // scale_ratio = mean(b)/mean(a)
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / k as f64;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / k as f64;
    let scale_ratio = if mean_a == 0.0 {
        if mean_b == 0.0 {
            1.0
        } else if mean_b.is_finite() && mean_a.is_finite() {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        mean_b / mean_a
    };

    let mut max_rel_delta = 0.0f64;
    let mut disagreeing = 0usize;
    for i in 0..k {
        let av = a[i];
        let bv = b[i];
        let rel = if !av.is_finite() || !bv.is_finite() {
            if av.to_bits() == bv.to_bits() {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            let denom = (av.abs() as f64).max(bv.abs() as f64);
            if denom == 0.0 {
                0.0
            } else {
                ((av - bv).abs() as f64) / denom
            }
        };
        if rel > max_rel_delta {
            max_rel_delta = rel;
        }
        if rel > rel_tol {
            disagreeing += 1;
        }
        // handle INFINITY propagation for max
        if rel.is_infinite() {
            max_rel_delta = f64::INFINITY;
        }
    }

    Ok(VectorComparison {
        k,
        pearson,
        spearman,
        scale_ratio,
        cosine,
        max_rel_delta,
        disagreeing_channels: disagreeing,
    })
}

// ---------------------------------------------------------------------------
// ImatrixDiff
// ---------------------------------------------------------------------------

/// Diff of two imatrices keyed by tensor name.
#[derive(Debug, Clone)]
pub struct ImatrixDiff {
    pub matched: Vec<(String, VectorComparison)>,
    pub only_in_a: Vec<String>,
    pub only_in_b: Vec<String>,
}

fn canonical_name(name: &str) -> String {
    if let Some(mapped) = safetensors_to_ggml_name(name) {
        mapped
    } else {
        name.to_string()
    }
}

/// Diff two imatrices, matching entries by name.
///
/// Tries each key directly and via `safetensors_to_ggml_name`, so a
/// safetensors-keyed entry on one side matches a `blk.*`-keyed entry on the
/// other.
pub fn diff_imatrix(a: &Imatrix, b: &Imatrix, rel_tol: f64) -> ImatrixDiff {
    // Build canonical -> original name maps
    let mut a_canon: BTreeMap<String, String> = BTreeMap::new();
    for name in a.names() {
        let can = canonical_name(name);
        // Keep first occurrence if duplicate canonical (should not happen)
        a_canon.entry(can).or_insert_with(|| name.to_string());
    }
    let mut b_canon: BTreeMap<String, String> = BTreeMap::new();
    for name in b.names() {
        let can = canonical_name(name);
        b_canon.entry(can).or_insert_with(|| name.to_string());
    }

    let all_keys: BTreeSet<String> = a_canon
        .keys()
        .chain(b_canon.keys())
        .cloned()
        .collect();

    let mut matched = Vec::new();
    let mut only_in_a = Vec::new();
    let mut only_in_b = Vec::new();

    for can in all_keys {
        match (a_canon.get(&can), b_canon.get(&can)) {
            (Some(a_name), Some(b_name)) => {
                let av = a.get(a_name).expect("a entry must exist");
                let bv = b.get(b_name).expect("b entry must exist");
                // If lengths differ, treat as maximal disagreement rather than
                // dropping the entry. compare_vectors would Err; synthesize a
                // comparison marking full disagreement.
                let cmp = match compare_vectors(av, bv, rel_tol) {
                    Ok(c) => c,
                    Err(_) => VectorComparison {
                        k: av.len().max(bv.len()),
                        pearson: 0.0,
                        spearman: 0.0,
                        scale_ratio: f64::NAN,
                        cosine: 0.0,
                        max_rel_delta: f64::INFINITY,
                        disagreeing_channels: av.len().max(bv.len()),
                    },
                };
                // Use the canonical display name; keep the a-side original if
                // the canonical is derived from safetensors, but either is
                // fine — we use the a-side name for stability.
                matched.push((a_name.clone(), cmp));
            }
            (Some(a_name), None) => only_in_a.push(a_name.clone()),
            (None, Some(b_name)) => only_in_b.push(b_name.clone()),
            (None, None) => unreachable!(),
        }
    }

    // Sort matched for deterministic output
    matched.sort_by(|(a, _), (b, _)| a.cmp(b));
    only_in_a.sort();
    only_in_b.sort();

    ImatrixDiff {
        matched,
        only_in_a,
        only_in_b,
    }
}

impl ImatrixDiff {
    /// Return the `n` worst entries by ascending `spearman` (most
    /// anti-correlated first).
    pub fn worst_by_spearman(&self, n: usize) -> Vec<&(String, VectorComparison)> {
        let mut v: Vec<&(String, VectorComparison)> = self.matched.iter().collect();
        v.sort_by(|a, b| {
            a.1.spearman
                .partial_cmp(&b.1.spearman)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.truncate(n);
        v
    }

    /// Median `spearman` across matched entries, or `None` if empty.
    pub fn median_spearman(&self) -> Option<f64> {
        if self.matched.is_empty() {
            return None;
        }
        let mut vals: Vec<f64> = self.matched.iter().map(|(_, c)| c.spearman).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = vals.len() / 2;
        if vals.len() % 2 == 1 {
            Some(vals[mid])
        } else {
            Some((vals[mid - 1] + vals[mid]) / 2.0)
        }
    }

    /// Short human-readable summary for CLI printing.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "matched: {}, only_in_a: {}, only_in_b: {}",
            self.matched.len(),
            self.only_in_a.len(),
            self.only_in_b.len()
        ));
        if let Some(med) = self.median_spearman() {
            s.push_str(&format!(", median spearman: {med:.4}"));
        }
        if !self.matched.is_empty() {
            let worst = self.worst_by_spearman(1);
            if let Some((name, cmp)) = worst.first() {
                s.push_str(&format!(", worst: {name} (spearman={:.4})", cmp.spearman));
            }
        }
        if !self.only_in_a.is_empty() {
            let preview: Vec<&str> = self.only_in_a.iter().take(3).map(|x| x.as_str()).collect();
            s.push_str(&format!(", only_in_a e.g.: {}", preview.join(", ")));
        }
        if !self.only_in_b.is_empty() {
            let preview: Vec<&str> = self.only_in_b.iter().take(3).map(|x| x.as_str()).collect();
            s.push_str(&format!(", only_in_b e.g.: {}", preview.join(", ")));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// AWQ scales
// ---------------------------------------------------------------------------

/// Compute AWQ per-channel scales `s[j]` for one linear-layer weight tensor.
///
/// Ported exactly from `crates/hipfire-quantize/src/main.rs::compute_awq_scales`
/// (lines ~6189-6217):
/// `s_raw[j] = sqrt(max(in_sum2[j], 1e-12))^alpha`, then divide by
/// `exp(mean(ln s_raw))` so the geometric mean is 1.
/// Clamped to `[1e-2, 1e2]` for f16 safety.
pub fn awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    if k == 0 {
        return Vec::new();
    }
    let half_alpha = alpha as f64 * 0.5;
    let mut log_s_raw: Vec<f64> = Vec::with_capacity(k);
    let mut sum_log: f64 = 0.0;
    for &v in in_sum2 {
        let v_clamped = (v as f64).max(1e-12).min(1e30);
        let log_s = half_alpha * v_clamped.ln();
        log_s_raw.push(log_s);
        sum_log += log_s;
    }
    let mean_log = sum_log / k as f64;
    const AWQ_SCALE_MIN: f32 = 1e-2;
    const AWQ_SCALE_MAX: f32 = 1e2;
    log_s_raw
        .into_iter()
        .map(|l| ((l - mean_log).exp() as f32).clamp(AWQ_SCALE_MIN, AWQ_SCALE_MAX))
        .collect()
}

/// Apply `awq_scales` to both vectors and compare the resulting scales.
///
/// This answers the end-to-end question: do two `Σx²` vectors produce the
/// *same AWQ scales*?  Scale-invariant differences in the raw imatrix cancel
/// out and should yield `max_rel_delta ≈ 0`.
pub fn scales_agree(a: &[f32], b: &[f32], alpha: f32, rel_tol: f64) -> Result<VectorComparison> {
    if a.len() != b.len() {
        return Err(QuantError::Malformed(format!(
            "scales_agree: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(QuantError::Malformed(
            "scales_agree: empty vectors".to_string(),
        ));
    }
    let sa = awq_scales(a, alpha);
    let sb = awq_scales(b, alpha);
    compare_vectors(&sa, &sb, rel_tol)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ---- helpers: minimal GGUF builder (mirrors format::imatrix tests) ----

    struct TestTensorSpec {
        name: String,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
    }

    fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_gguf(tensors: Vec<TestTensorSpec>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut offset: u64 = 0;
        for t in &tensors {
            write_string(&mut buf, &t.name);
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &d in &t.shape {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            let numel: usize = t.shape.iter().map(|&d| d as usize).product();
            let bsize = if t.dtype == 0 { numel * 4 } else { 0 };
            let adv = if bsize > 0 { bsize } else { t.data.len() };
            offset += adv as u64;
        }
        let pos = buf.len();
        let alignment = 32usize;
        let data_offset = (pos + alignment - 1) / alignment * alignment;
        buf.resize(data_offset, 0);
        for t in tensors {
            buf.extend_from_slice(&t.data);
        }
        buf
    }

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    fn make_imatrix(entries: Vec<(&str, Vec<f32>)>) -> tempfile::NamedTempFile {
        let mut tensors = Vec::new();
        for (name, vals) in entries {
            tensors.push(TestTensorSpec {
                name: format!("{name}.in_sum2"),
                shape: vec![vals.len() as u64],
                dtype: 0,
                data: f32s_to_bytes(&vals),
            });
            tensors.push(TestTensorSpec {
                name: format!("{name}.counts"),
                shape: vec![1],
                dtype: 0,
                data: f32s_to_bytes(&[10.0]),
            });
        }
        let bytes = build_gguf(tensors);
        write_temp_gguf(&bytes)
    }

    // ---- describe ----

    #[test]
    fn describe_with_zeros_and_nan() {
        let v = vec![0.0f32, 1.0, 0.0, f32::NAN, 2.0, -0.0];
        let s = describe(&v);
        assert_eq!(s.k, 6);
        // -0.0 == 0.0 counts as zero, so 0.0, 0.0, -0.0 = 3
        assert_eq!(s.zeros, 3);
        assert_eq!(s.nonfinite, 1);
        // finite values are [0,1,0,2,-0] => sum=3, mean=0.6, min=0, max=2
        assert!((s.sum - 3.0).abs() < 1e-9);
        assert!((s.mean - 0.6).abs() < 1e-9);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 2.0);
    }

    #[test]
    fn describe_all_finite_no_zeros() {
        let v = vec![1.0f32, 2.0, 3.0];
        let s = describe(&v);
        assert_eq!(s.zeros, 0);
        assert_eq!(s.nonfinite, 0);
        assert!((s.sum - 6.0).abs() < 1e-9);
        assert!((s.mean - 2.0).abs() < 1e-9);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 3.0);
    }

    // ---- compare_vectors ----

    #[test]
    fn compare_identical_vectors() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        let c = compare_vectors(&a, &b, 1e-6).unwrap();
        assert!((c.pearson - 1.0).abs() < 1e-12, "pearson {}", c.pearson);
        assert!((c.spearman - 1.0).abs() < 1e-12, "spearman {}", c.spearman);
        assert!((c.cosine - 1.0).abs() < 1e-12, "cosine {}", c.cosine);
        assert!((c.scale_ratio - 1.0).abs() < 1e-12, "scale {}", c.scale_ratio);
        assert_eq!(c.disagreeing_channels, 0);
        assert!(c.max_rel_delta < 1e-9);
    }

    #[test]
    fn compare_scaled_vector_spearman_still_one_and_ratio_1000() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b: Vec<f32> = a.iter().map(|x| x * 1000.0).collect();
        let c = compare_vectors(&a, &b, 1e-6).unwrap();
        assert!((c.spearman - 1.0).abs() < 1e-12, "spearman {}", c.spearman);
        assert!((c.pearson - 1.0).abs() < 1e-12, "pearson {}", c.pearson);
        assert!(
            (c.scale_ratio - 1000.0).abs() < 1e-9,
            "scale_ratio {}",
            c.scale_ratio
        );
        // cosine should also be 1 for pure scale
        assert!((c.cosine - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compare_reversed_spearman_minus_one() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let c = compare_vectors(&a, &b, 1e-6).unwrap();
        assert!(
            (c.spearman - (-1.0)).abs() < 1e-12,
            "spearman {}",
            c.spearman
        );
        assert!(
            (c.pearson - (-1.0)).abs() < 1e-12,
            "pearson {}",
            c.pearson
        );
    }

    #[test]
    fn compare_length_mismatch_err() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        let err = compare_vectors(&a, &b, 1e-6).unwrap_err();
        match err {
            QuantError::Malformed(_) => {}
            _ => panic!("expected Malformed, got {err:?}"),
        }
    }

    #[test]
    fn compare_disagreeing_channels_and_max_rel() {
        let a = vec![1.0f32, 1.0, 1.0];
        let b = vec![1.0f32, 2.0, 1.0];
        // rel for channel 1: |1-2|/2=0.5
        let c = compare_vectors(&a, &b, 0.1).unwrap();
        assert_eq!(c.disagreeing_channels, 1);
        assert!((c.max_rel_delta - 0.5).abs() < 1e-12);
        // with larger tol, no disagreeing
        let c2 = compare_vectors(&a, &b, 0.6).unwrap();
        assert_eq!(c2.disagreeing_channels, 0);
    }

    // ---- awq_scales ----

    #[test]
    fn awq_geometric_mean_is_one() {
        let v = vec![1.0f32, 4.0, 9.0, 16.0, 25.0];
        let s = awq_scales(&v, 0.5);
        assert_eq!(s.len(), v.len());
        let geo_mean = (s.iter().map(|&x| (x as f64).ln()).sum::<f64>() / s.len() as f64).exp();
        assert!(
            (geo_mean - 1.0).abs() < 1e-6,
            "geo_mean {geo_mean}"
        );
    }

    #[test]
    fn awq_alpha_zero_all_ones() {
        let v = vec![1.0f32, 2.0, 3.0, 100.0];
        let s = awq_scales(&v, 0.0);
        for &x in &s {
            assert!((x - 1.0).abs() < 1e-6, "x {x}");
        }
    }

    #[test]
    fn awq_zero_channel_not_nan() {
        let v = vec![0.0f32, 1.0, 2.0, 3.0];
        let s = awq_scales(&v, 0.5);
        for &x in &s {
            assert!(x.is_finite(), "non-finite scale {x}");
            assert!(!x.is_nan());
        }
        // zero channel should get the smallest scale (floored)
        assert!(s[0] < s[1]);
    }

    #[test]
    fn awq_clamping() {
        // Pathologically large value should be clamped to [1e-2, 1e2]
        let v = vec![1e30f32, 1.0, 1.0, 1.0];
        let s = awq_scales(&v, 0.5);
        for &x in &s {
            assert!(x >= 1e-2 && x <= 1e2, "out of clamp {x}");
            assert!(x.is_finite());
        }
    }

    // ---- scales_agree ----

    #[test]
    fn scales_agree_global_scale_identical() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|x| x * 42.0).collect();
        let cmp = scales_agree(&a, &b, 0.5, 1e-6).unwrap();
        // AWQ is scale-invariant: max_rel_delta ≈ 0
        assert!(
            cmp.max_rel_delta < 1e-6,
            "max_rel_delta {}",
            cmp.max_rel_delta
        );
        assert!((cmp.pearson - 1.0).abs() < 1e-12);
        assert!((cmp.spearman - 1.0).abs() < 1e-12);
        assert_eq!(cmp.disagreeing_channels, 0);
    }

    #[test]
    fn scales_agree_detects_structural_difference() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let cmp = scales_agree(&a, &b, 0.5, 1e-6).unwrap();
        // Reversed ordering should produce different scales
        assert!(cmp.spearman < 0.0, "spearman {}", cmp.spearman);
        assert!(cmp.max_rel_delta > 0.1);
    }

    // ---- diff_imatrix ----

    #[test]
    fn diff_imatrix_partitioning() {
        let vals_a = vec![1.0f32, 2.0, 3.0];
        let vals_b = vec![1.0f32, 2.0, 3.0];
        let vals_shared = vec![5.0f32, 6.0, 7.0];

        // a has: blk.0.ffn_up + blk.0.ffn_gate
        let fa = make_imatrix(vec![
            ("blk.0.ffn_up.weight", vals_a.clone()),
            ("blk.0.ffn_gate.weight", vals_shared.clone()),
        ]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();

        // b has: blk.0.ffn_gate + blk.0.ffn_down
        let fb = make_imatrix(vec![
            ("blk.0.ffn_gate.weight", vals_shared.clone()),
            ("blk.0.ffn_down.weight", vals_b.clone()),
        ]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();

        let diff = diff_imatrix(&a, &b, 1e-6);
        assert_eq!(diff.matched.len(), 1);
        assert_eq!(diff.only_in_a.len(), 1);
        assert_eq!(diff.only_in_b.len(), 1);
        assert!(diff.only_in_a.iter().any(|n| n.contains("ffn_up")));
        assert!(diff.only_in_b.iter().any(|n| n.contains("ffn_down")));
    }

    #[test]
    fn diff_imatrix_cross_keying() {
        // a is ggml-keyed, b is safetensors-keyed for same logical tensor
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let fa = make_imatrix(vec![("blk.0.ffn_up.weight", vals.clone())]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();

        let fb = make_imatrix(vec![(
            "model.language_model.layers.0.mlp.up_proj.weight",
            vals.clone(),
        )]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();

        let diff = diff_imatrix(&a, &b, 1e-6);
        assert_eq!(diff.matched.len(), 1, "cross-keyed should match");
        assert!(diff.only_in_a.is_empty());
        assert!(diff.only_in_b.is_empty());
        // matched comparison should be perfect
        let (_, cmp) = &diff.matched[0];
        assert!((cmp.spearman - 1.0).abs() < 1e-12);
    }

    #[test]
    fn diff_imatrix_cross_keying_reverse_direction() {
        // a safetensors, b ggml — opposite direction
        let vals = vec![2.0f32, 4.0, 6.0];
        let fa = make_imatrix(vec![(
            "model.language_model.layers.1.mlp.gate_proj.weight",
            vals.clone(),
        )]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();
        let fb = make_imatrix(vec![("blk.1.ffn_gate.weight", vals.clone())]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();

        let diff = diff_imatrix(&a, &b, 1e-6);
        assert_eq!(diff.matched.len(), 1);
        assert!(diff.only_in_a.is_empty());
        assert!(diff.only_in_b.is_empty());
    }

    #[test]
    fn worst_by_spearman_ascending() {
        // Build a with 3 entries, b with permuted values to create varying spearman
        // Entry 0: identical => spearman 1
        // Entry 1: reversed => spearman -1 (worst)
        // Entry 2: slightly perturbed => spearman ~ high but <1
        let fa = make_imatrix(vec![
            ("blk.0.ffn_up.weight", vec![1.0, 2.0, 3.0, 4.0]),
            ("blk.1.ffn_up.weight", vec![1.0, 2.0, 3.0, 4.0]),
            ("blk.2.ffn_up.weight", vec![1.0, 2.0, 3.0, 4.0]),
        ]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();
        let fb = make_imatrix(vec![
            ("blk.0.ffn_up.weight", vec![1.0, 2.0, 3.0, 4.0]), // identical
            ("blk.1.ffn_up.weight", vec![4.0, 3.0, 2.0, 1.0]), // reversed
            ("blk.2.ffn_up.weight", vec![1.0, 2.0, 3.0, 3.5]), // slightly off
        ]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();

        let diff = diff_imatrix(&a, &b, 1e-6);
        assert_eq!(diff.matched.len(), 3);
        let worst = diff.worst_by_spearman(3);
        assert_eq!(worst.len(), 3);
        // Should be ascending: most negative first
        for w in worst.windows(2) {
            assert!(
                w[0].1.spearman <= w[1].1.spearman + 1e-12,
                "not ascending: {} > {}",
                w[0].1.spearman,
                w[1].1.spearman
            );
        }
        // The worst should be the reversed one (blk.1)
        assert!(
            worst[0].0.contains("blk.1"),
            "worst should be blk.1, got {}",
            worst[0].0
        );
        assert!((worst[0].1.spearman - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn median_spearman() {
        let fa = make_imatrix(vec![
            ("blk.0.ffn_up.weight", vec![1.0, 2.0, 3.0]),
            ("blk.1.ffn_up.weight", vec![1.0, 2.0, 3.0]),
            ("blk.2.ffn_up.weight", vec![1.0, 2.0, 3.0]),
        ]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();
        let fb = make_imatrix(vec![
            ("blk.0.ffn_up.weight", vec![1.0, 2.0, 3.0]),
            ("blk.1.ffn_up.weight", vec![1.0, 2.0, 3.0]),
            ("blk.2.ffn_up.weight", vec![3.0, 2.0, 1.0]),
        ]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();
        let diff = diff_imatrix(&a, &b, 1e-6);
        // spearman values: 1, 1, -1 => sorted -1,1,1 => median 1
        let med = diff.median_spearman().unwrap();
        assert!((med - 1.0).abs() < 1e-12, "median {med}");
    }

    #[test]
    fn median_spearman_empty_none() {
        let fa = make_imatrix(vec![("blk.0.ffn_up.weight", vec![1.0, 2.0])]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();
        let fb = make_imatrix(vec![("blk.1.ffn_up.weight", vec![1.0, 2.0])]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();
        let diff = diff_imatrix(&a, &b, 1e-6);
        assert!(diff.matched.is_empty());
        assert!(diff.median_spearman().is_none());
    }

    #[test]
    fn summary_contains_counts() {
        let fa = make_imatrix(vec![("blk.0.ffn_up.weight", vec![1.0, 2.0])]);
        let a = crate::format::imatrix::open(fa.path()).unwrap();
        let fb = make_imatrix(vec![("blk.0.ffn_up.weight", vec![1.0, 2.0])]);
        let b = crate::format::imatrix::open(fb.path()).unwrap();
        let diff = diff_imatrix(&a, &b, 1e-6);
        let s = diff.summary();
        assert!(s.contains("matched: 1"), "summary {s}");
        assert!(s.contains("only_in_a: 0"), "summary {s}");
        assert!(s.contains("only_in_b: 0"), "summary {s}");
    }
}
