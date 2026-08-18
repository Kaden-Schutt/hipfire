// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent-logits (`.plog`) container and KLD/PPL comparator.
//!
//! File layout (§7 of the parent-checkpoint contract):
//!
//! ```text
//! magic    : b"HFPLOG01"   (8 bytes)
//! n_tokens : u32 LE
//! vocab    : u32 LE
//! reserved : u64 LE (zero)
//! logits   : n_tokens * vocab * f32 LE, row-major
//!            row t = logits AFTER position t
//! ```
//!
//! Writers stream one row at a time so a 32K-token × 129280-vocab capture
//! (≈16 GiB) never has to live on the heap as one tensor. Readers mmap the
//! body so `row(t)` is an O(1) slice into the mapping — peak RSS is bounded
//! by touched pages, not file size.

use memmap2::Mmap;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Magic bytes identifying a hipfire parent-logits file.
pub const PLOG_MAGIC: &[u8; 8] = b"HFPLOG01";

const HEADER_BYTES: usize = 8 + 4 + 4 + 8; // magic + n_tokens + vocab + reserved

/// Streaming writer for `.plog` files. Never buffers the full logit tensor.
pub struct PlogWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    n_tokens: usize,
    vocab: usize,
    rows_written: usize,
}

impl PlogWriter {
    /// Create a new `.plog` at `path` with a fixed shape.
    ///
    /// The header is written immediately. Call [`push_row`] exactly
    /// `n_tokens` times, then [`finish`].
    pub fn create(path: &Path, n_tokens: usize, vocab: usize) -> Result<Self, String> {
        if n_tokens > u32::MAX as usize {
            return Err(format!(
                "deepseek4 parent: plog n_tokens {n_tokens} exceeds u32"
            ));
        }
        if vocab > u32::MAX as usize {
            return Err(format!(
                "deepseek4 parent: plog vocab {vocab} exceeds u32"
            ));
        }
        if vocab == 0 {
            return Err("deepseek4 parent: plog vocab must be > 0".into());
        }
        let file = File::create(path).map_err(|e| {
            format!(
                "deepseek4 parent: create plog {}: {e}",
                path.display()
            )
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(PLOG_MAGIC)
            .map_err(|e| format!("deepseek4 parent: write plog magic: {e}"))?;
        writer
            .write_all(&(n_tokens as u32).to_le_bytes())
            .map_err(|e| format!("deepseek4 parent: write plog n_tokens: {e}"))?;
        writer
            .write_all(&(vocab as u32).to_le_bytes())
            .map_err(|e| format!("deepseek4 parent: write plog vocab: {e}"))?;
        writer
            .write_all(&0u64.to_le_bytes())
            .map_err(|e| format!("deepseek4 parent: write plog reserved: {e}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            n_tokens,
            vocab,
            rows_written: 0,
        })
    }

    /// Append one logit row. `logits.len()` must equal the declared vocab.
    pub fn push_row(&mut self, logits: &[f32]) -> Result<(), String> {
        if logits.len() != self.vocab {
            return Err(format!(
                "deepseek4 parent: plog push_row len {} != vocab {}",
                logits.len(),
                self.vocab
            ));
        }
        if self.rows_written >= self.n_tokens {
            return Err(format!(
                "deepseek4 parent: plog push_row exceeds n_tokens {}",
                self.n_tokens
            ));
        }
        let bytes = f32_slice_as_le_bytes(logits);
        self.writer.write_all(bytes).map_err(|e| {
            format!(
                "deepseek4 parent: write plog row {} to {}: {e}",
                self.rows_written,
                self.path.display()
            )
        })?;
        self.rows_written += 1;
        Ok(())
    }

    /// Flush and close. Errors unless exactly `n_tokens` rows were pushed.
    pub fn finish(mut self) -> Result<(), String> {
        if self.rows_written != self.n_tokens {
            return Err(format!(
                "deepseek4 parent: plog finish row count {} != n_tokens {} ({})",
                self.rows_written,
                self.n_tokens,
                self.path.display()
            ));
        }
        self.writer
            .flush()
            .map_err(|e| format!("deepseek4 parent: flush plog {}: {e}", self.path.display()))?;
        Ok(())
    }
}

/// Memory-mapped `.plog` reader. Peak RSS tracks touched pages, not file size.
pub struct PlogReader {
    /// Kept alive for the lifetime of `mmap`.
    _file: File,
    mmap: Mmap,
    n_tokens: usize,
    vocab: usize,
}

impl PlogReader {
    /// Open and validate a `.plog`. Rejects bad magic and size/header mismatch.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| {
            format!(
                "deepseek4 parent: open plog {}: {e}",
                path.display()
            )
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            format!(
                "deepseek4 parent: mmap plog {}: {e}",
                path.display()
            )
        })?;
        if mmap.len() < HEADER_BYTES {
            return Err(format!(
                "deepseek4 parent: plog {} truncated header ({} bytes)",
                path.display(),
                mmap.len()
            ));
        }
        if &mmap[0..8] != PLOG_MAGIC.as_slice() {
            return Err(format!(
                "deepseek4 parent: plog {} bad magic (expected HFPLOG01)",
                path.display()
            ));
        }
        let n_tokens = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let vocab = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let _reserved = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
        if vocab == 0 {
            return Err(format!(
                "deepseek4 parent: plog {} vocab is zero",
                path.display()
            ));
        }
        let body = n_tokens
            .checked_mul(vocab)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                format!(
                    "deepseek4 parent: plog {} shape overflow n_tokens={n_tokens} vocab={vocab}",
                    path.display()
                )
            })?;
        let expected = HEADER_BYTES + body;
        if mmap.len() != expected {
            return Err(format!(
                "deepseek4 parent: plog {} size {} disagrees with header (expected {expected} for n_tokens={n_tokens} vocab={vocab})",
                path.display(),
                mmap.len()
            ));
        }
        // Alignment: header is 24 bytes; body of f32 starts at offset 24.
        // memmap base is page-aligned; offset 24 is 8-byte aligned → ok for f32.
        debug_assert_eq!((mmap.as_ptr() as usize + HEADER_BYTES) % std::mem::align_of::<f32>(), 0);
        Ok(Self {
            _file: file,
            mmap,
            n_tokens,
            vocab,
        })
    }

    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Borrow logit row `t` (0-based). Errors on out-of-range index.
    pub fn row(&self, t: usize) -> Result<&[f32], String> {
        if t >= self.n_tokens {
            return Err(format!(
                "deepseek4 parent: plog row index {t} out of range (n_tokens={})",
                self.n_tokens
            ));
        }
        let start = HEADER_BYTES + t * self.vocab * 4;
        let end = start + self.vocab * 4;
        Ok(bytes_as_f32_slice(&self.mmap[start..end]))
    }
}

/// Summary of distributional divergence between a reference and a candidate
/// parent-logits file, scored on the same token IDs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DivergenceReport {
    pub n_tokens: usize,
    pub vocab: usize,
    pub kld_mean: f64,
    pub kld_p50: f64,
    pub kld_p95: f64,
    pub kld_max: f64,
    /// Fraction of positions whose argmax matches.
    pub top1_agreement: f64,
    /// Mean `|top5(p) ∩ top5(q)| / 5`.
    pub top5_overlap_mean: f64,
    /// `exp(mean NLL)` of `token_ids[t + 1]` under the reference, over the
    /// first `n_tokens - 1` rows. See [`compare`] for why the shift is here.
    pub ppl_reference: f64,
    /// `exp(mean NLL)` of `token_ids[t + 1]` under the candidate.
    pub ppl_candidate: f64,
    pub max_abs_logit_delta: f32,
}

/// Compare two `.plog` files on the same shape against `token_ids`.
///
/// KLD is `D_KL(P_reference || Q_candidate)` over softmaxed logits, computed
/// per position in f64 with max-subtraction log-sum-exp, across **all** rows.
/// KLD needs no targets, so it is unaffected by the shift below.
///
/// # The target shift lives here, on purpose
///
/// `.plog` row `t` holds the logits **after** position `t`, i.e. the
/// prediction for token `t + 1`. PPL therefore scores row `t` against
/// `token_ids[t + 1]`, over rows `0..n-1`; the final row predicts a token
/// past the end of the sequence and is not scored.
///
/// This shift used to be the caller's job, and the caller got it wrong: the
/// first real Gate 6 run scored row `t` against `token_ids[t]` and reported
/// perplexities of 5.7e6 (parent) and 1.8e6 (MQ2R) — *worse than the uniform
/// 129,280* — while both models were independently emitting coherent text and
/// hitting 30-54% next-token top-1. An empirical shift scan confirmed
/// `argmax == token_ids[t]` in 0 of 48 sampled rows against
/// `argmax == token_ids[t + 1]` in 13-26 of 48.
///
/// Doing the shift internally makes that misuse impossible, which is the same
/// lesson as the BF16-vs-f32 `amax` bug: an interface's input convention is
/// part of its contract, and leaving it implicit means it will eventually be
/// violated by someone reading only the signature.
///
/// `token_ids` is the full sequence the logits were produced from and must
/// have exactly `n_tokens` entries.
pub fn compare(
    reference: &PlogReader,
    candidate: &PlogReader,
    token_ids: &[u32],
) -> Result<DivergenceReport, String> {
    if reference.n_tokens() != candidate.n_tokens() || reference.vocab() != candidate.vocab() {
        return Err(format!(
            "deepseek4 parent: plog shape mismatch ref=({}×{}) cand=({}×{})",
            reference.n_tokens(),
            reference.vocab(),
            candidate.n_tokens(),
            candidate.vocab()
        ));
    }
    let n = reference.n_tokens();
    let v = reference.vocab();
    if token_ids.len() != n {
        return Err(format!(
            "deepseek4 parent: token_ids len {} != n_tokens {n}",
            token_ids.len()
        ));
    }
    if n == 0 {
        return Err("deepseek4 parent: plog compare on empty file".into());
    }

    let mut klds = Vec::with_capacity(n);
    let mut top1_hits = 0usize;
    let mut top5_overlap_sum = 0.0f64;
    let mut nll_ref_sum = 0.0f64;
    let mut nll_cand_sum = 0.0f64;
    let mut max_abs_logit_delta = 0.0f32;

    // Row t predicts token t+1, so PPL is scored over rows 0..n-1 only. KLD
    // needs no target and is accumulated across every row.
    let mut scored = 0usize;
    for t in 0..n {
        let pref = reference.row(t)?;
        let pcand = candidate.row(t)?;

        for i in 0..v {
            let d = (pref[i] - pcand[i]).abs();
            if d > max_abs_logit_delta {
                max_abs_logit_delta = d;
            }
        }

        let target = if t + 1 < n {
            let tgt = token_ids[t + 1] as usize;
            if tgt >= v {
                return Err(format!(
                    "deepseek4 parent: token_ids[{}]={tgt} out of vocab {v}",
                    t + 1
                ));
            }
            Some(tgt)
        } else {
            None
        };

        let (log_p, nll_p) = log_softmax_and_nll(pref, target.unwrap_or(0));
        let (log_q, nll_q) = log_softmax_and_nll(pcand, target.unwrap_or(0));
        if target.is_some() {
            nll_ref_sum += nll_p;
            nll_cand_sum += nll_q;
            scored += 1;
        }

        // D_KL(P||Q) = Σ_i exp(log_p_i) * (log_p_i - log_q_i)
        let mut kld = 0.0f64;
        for i in 0..v {
            let lp = log_p[i];
            let lq = log_q[i];
            kld += lp.exp() * (lp - lq);
        }
        klds.push(kld);

        let top_p = top_k_indices(pref, 5);
        let top_q = top_k_indices(pcand, 5);
        if top_p[0] == top_q[0] {
            top1_hits += 1;
        }
        let mut overlap = 0usize;
        for &a in &top_p {
            if top_q.contains(&a) {
                overlap += 1;
            }
        }
        top5_overlap_sum += overlap as f64 / 5.0;
    }

    let kld_mean = klds.iter().sum::<f64>() / n as f64;
    let kld_max = klds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = klds;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let kld_p50 = percentile_sorted(&sorted, 0.50);
    let kld_p95 = percentile_sorted(&sorted, 0.95);

    Ok(DivergenceReport {
        n_tokens: n,
        vocab: v,
        kld_mean,
        kld_p50,
        kld_p95,
        kld_max,
        top1_agreement: top1_hits as f64 / n as f64,
        top5_overlap_mean: top5_overlap_sum / n as f64,
        // Divide by the number of *scored* rows, not n: the final row has no
        // in-sequence target. At n = 1 nothing is scored and PPL is NaN, which
        // is the honest answer rather than exp(0) = 1.
        ppl_reference: if scored == 0 {
            f64::NAN
        } else {
            (nll_ref_sum / scored as f64).exp()
        },
        ppl_candidate: if scored == 0 {
            f64::NAN
        } else {
            (nll_cand_sum / scored as f64).exp()
        },
        max_abs_logit_delta,
    })
}

/// Log-softmax in f64 with max-subtraction; also returns NLL of `target`.
fn log_softmax_and_nll(logits: &[f32], target: usize) -> (Vec<f64>, f64) {
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if x > max {
            max = x;
        }
    }
    let max64 = max as f64;
    let mut sum = 0.0f64;
    for &x in logits {
        sum += ((x as f64) - max64).exp();
    }
    let log_sum = max64 + sum.ln();
    let mut log_p = Vec::with_capacity(logits.len());
    for &x in logits {
        log_p.push((x as f64) - log_sum);
    }
    let nll = -log_p[target];
    (log_p, nll)
}

/// Indices of the `k` largest values (descending). Stable on ties by lower index.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn f32_slice_as_le_bytes(v: &[f32]) -> &[u8] {
    // Host is LE (Linux x86_64 / aarch64 targets we ship). Format is LE.
    debug_assert!(cfg!(target_endian = "little"));
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_as_f32_slice(bytes: &[u8]) -> &[f32] {
    debug_assert_eq!(bytes.len() % 4, 0);
    debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "hipfire-plog-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn write_read_round_trip() {
        let path = temp_path("roundtrip");
        let n_tokens = 7usize;
        let vocab = 11usize;
        let mut rows = Vec::new();
        {
            let mut w = PlogWriter::create(&path, n_tokens, vocab).unwrap();
            for t in 0..n_tokens {
                let row: Vec<f32> = (0..vocab)
                    .map(|i| (t * 100 + i) as f32 * 0.125 - 3.0)
                    .collect();
                w.push_row(&row).unwrap();
                rows.push(row);
            }
            w.finish().unwrap();
        }
        let r = PlogReader::open(&path).unwrap();
        assert_eq!(r.n_tokens(), n_tokens);
        assert_eq!(r.vocab(), vocab);
        for t in 0..n_tokens {
            let got = r.row(t).unwrap();
            assert_eq!(got.len(), vocab);
            for i in 0..vocab {
                assert_eq!(
                    got[i].to_bits(),
                    rows[t][i].to_bits(),
                    "mismatch at row {t} col {i}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn finish_requires_exact_row_count_and_push_checks_len() {
        let path = temp_path("finish");
        let mut w = PlogWriter::create(&path, 3, 4).unwrap();
        let bad = w.push_row(&[1.0, 2.0, 3.0]); // len 3 != vocab 4
        assert!(
            bad.unwrap_err().contains("push_row len"),
            "wrong-length row must error"
        );
        w.push_row(&[0.0; 4]).unwrap();
        w.push_row(&[1.0; 4]).unwrap();
        // only 2 of 3 rows
        let err = w.finish().unwrap_err();
        assert!(
            err.contains("finish row count"),
            "truncated finish must error, got: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_bad_magic_and_truncated_body() {
        let path = temp_path("badmagic");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"NOPELOG0").unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap();
            f.write_all(&0f32.to_le_bytes()).unwrap();
        }
        let err = match PlogReader::open(&path) {
            Ok(_) => panic!("expected bad magic to error"),
            Err(e) => e,
        };
        assert!(err.contains("bad magic"), "got: {err}");
        let _ = std::fs::remove_file(&path);

        let path2 = temp_path("trunc");
        {
            let mut w = PlogWriter::create(&path2, 2, 3).unwrap();
            w.push_row(&[0.0; 3]).unwrap();
            w.push_row(&[1.0; 3]).unwrap();
            w.finish().unwrap();
            // Truncate body by one f32.
            let f = File::options().write(true).open(&path2).unwrap();
            f.set_len((HEADER_BYTES + 2 * 3 * 4 - 4) as u64).unwrap();
        }
        let err = match PlogReader::open(&path2) {
            Ok(_) => panic!("expected truncated body to error"),
            Err(e) => e,
        };
        assert!(
            err.contains("disagrees with header") || err.contains("truncated"),
            "got: {err}"
        );
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn compare_identical_is_zero_kld() {
        let path = temp_path("ident");
        let n = 5usize;
        let v = 8usize;
        let ids: Vec<u32> = (0..n as u32).map(|t| t % v as u32).collect();
        {
            let mut w = PlogWriter::create(&path, n, v).unwrap();
            for t in 0..n {
                let row: Vec<f32> = (0..v).map(|i| ((t + i) as f32) * 0.3 - 1.0).collect();
                w.push_row(&row).unwrap();
            }
            w.finish().unwrap();
        }
        let r = PlogReader::open(&path).unwrap();
        let report = compare(&r, &r, &ids).unwrap();
        assert_eq!(report.kld_mean, 0.0);
        assert_eq!(report.kld_max, 0.0);
        assert_eq!(report.top1_agreement, 1.0);
        assert_eq!(report.top5_overlap_mean, 1.0);
        assert_eq!(report.ppl_reference, report.ppl_candidate);
        assert!(report.ppl_reference.is_finite());
        assert_eq!(report.max_abs_logit_delta, 0.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hand_computed_kld_one_position() {
        // Construct known discrete distributions via log-prob logits stored as f32.
        //   P = [0.5, 0.25, 0.25]
        //   Q = [0.25, 0.5, 0.25]
        // Ideal D_KL(P||Q) = 0.5*ln(2) + 0.25*(-ln(2)) + 0 = 0.25 * ln(2).
        // We feed f32-rounded ln(p) as logits, then recompute the expected KLD
        // from those exact f32 values (widened to f64 + log-sum-exp) so the
        // assertion catches base-e/base-2 or P/Q-swap bugs without fighting
        // f32 quantization of ln(0.25).
        let p_logits_f32 = [0.5f64.ln() as f32, 0.25f64.ln() as f32, 0.25f64.ln() as f32];
        let q_logits_f32 = [0.25f64.ln() as f32, 0.5f64.ln() as f32, 0.25f64.ln() as f32];

        fn log_softmax_f64(logits: &[f32]) -> [f64; 3] {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|&x| ((x as f64) - max).exp()).sum();
            let log_sum = max + sum.ln();
            [
                logits[0] as f64 - log_sum,
                logits[1] as f64 - log_sum,
                logits[2] as f64 - log_sum,
            ]
        }
        let lp = log_softmax_f64(&p_logits_f32);
        let lq = log_softmax_f64(&q_logits_f32);
        let expected: f64 = (0..3).map(|i| lp[i].exp() * (lp[i] - lq[i])).sum();
        // Sanity: expected must be within a few ulps of the analytic 0.25*ln2.
        let analytic = 0.25 * std::f64::consts::LN_2;
        assert!(
            (expected - analytic).abs() < 1e-6,
            "fixture drifted too far from analytic: {expected} vs {analytic}"
        );

        let path_p = temp_path("kld-p");
        let path_q = temp_path("kld-q");
        {
            let mut w = PlogWriter::create(&path_p, 1, 3).unwrap();
            w.push_row(&p_logits_f32).unwrap();
            w.finish().unwrap();
        }
        {
            let mut w = PlogWriter::create(&path_q, 1, 3).unwrap();
            w.push_row(&q_logits_f32).unwrap();
            w.finish().unwrap();
        }
        let pref = PlogReader::open(&path_p).unwrap();
        let pcand = PlogReader::open(&path_q).unwrap();
        let report = compare(&pref, &pcand, &[0u32]).unwrap();
        assert!(
            (report.kld_mean - expected).abs() < 1e-12,
            "kld_mean={} expected={expected} (base-e / P||Q regression)",
            report.kld_mean
        );
        // Also pin against the closed form so a P/Q swap still fails hard.
        assert!(
            (report.kld_mean - analytic).abs() < 1e-6,
            "kld_mean={} analytic={analytic}",
            report.kld_mean
        );
        let _ = std::fs::remove_file(&path_p);
        let _ = std::fs::remove_file(&path_q);
    }

    #[test]

    fn log_sum_exp_stable_on_huge_logits() {
        let path_a = temp_path("huge-a");
        let path_b = temp_path("huge-b");
        // Two rows, not one: row t is scored against token_ids[t + 1], so a
        // single-row file has nothing to score and PPL is legitimately NaN.
        // Two rows keeps this a real stability check on both KLD and PPL.
        {
            let mut w = PlogWriter::create(&path_a, 2, 4).unwrap();
            w.push_row(&[800.0, -800.0, 799.0, 100.0]).unwrap();
            w.push_row(&[-800.0, 800.0, 100.0, 799.0]).unwrap();
            w.finish().unwrap();
        }
        {
            let mut w = PlogWriter::create(&path_b, 2, 4).unwrap();
            w.push_row(&[790.0, -790.0, 800.0, 50.0]).unwrap();
            w.push_row(&[-790.0, 790.0, 50.0, 800.0]).unwrap();
            w.finish().unwrap();
        }
        let a = PlogReader::open(&path_a).unwrap();
        let b = PlogReader::open(&path_b).unwrap();
        // Score row 0 against token 0, which BOTH files rank first. With
        // logits spanning ±800 any other target gives NLL in the hundreds and
        // exp() legitimately overflows to +inf — that would be arithmetic
        // working correctly, not the stability property under test here.
        let report = compare(&a, &b, &[3u32, 0u32]).unwrap();
        assert!(
            report.kld_mean.is_finite() && !report.kld_mean.is_nan(),
            "kld_mean not finite: {}",
            report.kld_mean
        );
        assert!(report.ppl_reference.is_finite() && !report.ppl_reference.is_nan());
        assert!(report.ppl_candidate.is_finite() && !report.ppl_candidate.is_nan());
        assert!(report.kld_max.is_finite());
        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
    }

    /// Pins the target shift that the first real Gate 6 run got wrong.
    ///
    /// Row `t` predicts token `t + 1`. Build a reference whose row `t` puts
    /// all its mass on `token_ids[t + 1]`: PPL must be ~1. Scoring against
    /// `token_ids[t]` instead — the old behaviour — would put the mass on a
    /// token this construction makes deliberately unlikely, and PPL would
    /// explode. That is exactly the 5.7e6 the parent reported.
    #[test]
    fn ppl_scores_row_t_against_token_t_plus_one() {
        let path = temp_path("shift");
        let vocab = 8usize;
        let ids: Vec<u32> = vec![7, 1, 2, 3, 4];
        let n = ids.len();
        {
            let mut w = PlogWriter::create(&path, n, vocab).unwrap();
            for t in 0..n {
                let mut row = vec![-30.0f32; vocab];
                // Confident about the NEXT token; the current token is the
                // least likely thing to come next in this construction.
                let next = if t + 1 < n { ids[t + 1] } else { 0 } as usize;
                row[next] = 30.0;
                w.push_row(&row).unwrap();
            }
            w.finish().unwrap();
        }
        let r = PlogReader::open(&path).unwrap();
        let report = compare(&r, &r, &ids).unwrap();

        assert_eq!(report.kld_mean, 0.0, "self-comparison must be zero KLD");
        assert!(
            (report.ppl_reference - 1.0).abs() < 1e-6,
            "row t must be scored against token_ids[t+1]; got PPL {}",
            report.ppl_reference
        );
        // Guard the other direction: had the shift been dropped, row 0 would
        // be scored against ids[0] = 7, which row 0 assigns -30.0. Confirm the
        // construction really would have blown up, so this test cannot pass
        // vacuously if someone reverts the shift.
        let mut row0 = vec![-30.0f32; vocab];
        row0[ids[1] as usize] = 30.0;
        assert!(
            row0[ids[0] as usize] < 0.0,
            "fixture must make the unshifted target improbable"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn row_out_of_range_errors() {
        let path = temp_path("oor");
        {
            let mut w = PlogWriter::create(&path, 2, 3).unwrap();
            w.push_row(&[0.0; 3]).unwrap();
            w.push_row(&[1.0; 3]).unwrap();
            w.finish().unwrap();
        }
        let r = PlogReader::open(&path).unwrap();
        assert!(r.row(2).unwrap_err().contains("out of range"));
        let _ = std::fs::remove_file(&path);
    }
}
