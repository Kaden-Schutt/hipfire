// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Unified speculative-decode metrics accumulator.
//!
//! One per-request accumulator that EVERY strategy (dflash, ddtree, mtp, dspark,
//! deepseek4-mtp) feeds through the `Speculator::step -> SpecStep` seam, replacing
//! the per-path hand-rolled `u64` counters and the three divergent `done`-event
//! schemas. It records PRIMITIVES (`proposed`/`accepted`/`committed`) rather than
//! any strategy step type, so this core crate takes no dependency on a strategy
//! crate (`SpecStep` lives in `hipfire-specdecode-dspark`, `SpecStepResult` in the
//! arch crates — both depend on this crate, not the other way around).
//!
//! Strategy-SPECIALIZED metrics (dspark confidence histogram, ddtree tree size,
//! seed-oracle, mtp compressed-vocab, ...) are contributed separately via
//! `Speculator::drain_extra_metrics` and live in their own crates — the core
//! metric never names them.

use serde_json::{json, Value};

/// Arch-agnostic per-request spec-decode metrics. Fed one window at a time from
/// the daemon spec loop; serialized once into the `done` event.
#[derive(Debug, Clone, Default)]
pub struct SpecMetrics {
    /// Speculative windows (verify cycles) run.
    pub windows: usize,
    /// Total drafts proposed across windows (after any per-strategy truncation,
    /// e.g. dspark confidence cut-off).
    pub proposed_total: usize,
    /// Total drafts accepted — the τ numerator.
    pub accepted_total: usize,
    /// Total tokens committed (accepted prefix + bonus token per window).
    pub committed_total: usize,
    /// Per-window accepted-count histogram, indexed by accepted count
    /// (`0..=block_size`). `acceptance_hist[i]` = windows with exactly `i`
    /// drafts accepted. Empty if constructed with `default()`.
    pub acceptance_hist: Vec<usize>,
}

impl SpecMetrics {
    /// `block_size` sizes the acceptance histogram (`0..=block_size`).
    pub fn new(block_size: usize) -> Self {
        Self {
            windows: 0,
            proposed_total: 0,
            accepted_total: 0,
            committed_total: 0,
            acceptance_hist: vec![0; block_size + 1],
        }
    }

    /// Record one speculative window. `proposed` = drafts offered this window,
    /// `accepted` = drafts accepted, `committed` = tokens emitted this window
    /// (accepted prefix + bonus). Every strategy already produces these three
    /// counts on its step result, so a single call unifies them.
    pub fn record_window(&mut self, proposed: usize, accepted: usize, committed: usize) {
        self.windows += 1;
        self.proposed_total += proposed;
        self.accepted_total += accepted;
        self.committed_total += committed;
        if accepted < self.acceptance_hist.len() {
            self.acceptance_hist[accepted] += 1;
        }
    }

    /// Mean accepted drafts per window — τ (Leviathan). Excludes the bonus token.
    pub fn tau(&self) -> f32 {
        if self.windows == 0 {
            0.0
        } else {
            self.accepted_total as f32 / self.windows as f32
        }
    }

    /// Mean committed tokens per window (≈ τ + 1); the effective per-cycle decode
    /// advance vs plain AR.
    pub fn mean_committed(&self) -> f32 {
        if self.windows == 0 {
            0.0
        } else {
            self.committed_total as f32 / self.windows as f32
        }
    }

    /// Fraction of proposed drafts that were accepted (`0..=1`).
    pub fn accept_rate(&self) -> f32 {
        if self.proposed_total == 0 {
            0.0
        } else {
            self.accepted_total as f32 / self.proposed_total as f32
        }
    }

    /// Mean drafts proposed per window — the effective draft-block length after
    /// any per-strategy truncation. (A collapse to ~1.0 means the drafter/gate is
    /// never proposing more than one token, i.e. no block speculation.)
    pub fn mean_draft_len(&self) -> f32 {
        if self.windows == 0 {
            0.0
        } else {
            self.proposed_total as f32 / self.windows as f32
        }
    }

    /// Canonical `done`-event spec block. Field names are the eval-parseable set
    /// (`tau`, `accept_rate`, `windows`, ...) so every strategy reports uniformly.
    /// Rates rounded to 3 dp for stable JSON.
    pub fn to_json(&self) -> Value {
        let r3 = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
        json!({
            "windows": self.windows,
            "proposed": self.proposed_total,
            "accepted": self.accepted_total,
            "committed": self.committed_total,
            "tau": r3(self.tau()),
            "accept_rate": r3(self.accept_rate()),
            "mean_draft_len": r3(self.mean_draft_len()),
            "mean_committed": r3(self.mean_committed()),
            "acceptance_hist": self.acceptance_hist,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_derive() {
        let mut m = SpecMetrics::new(7);
        // window A: proposed 7, accepted 3, committed 4 (3 + bonus).
        m.record_window(7, 3, 4);
        // window B: proposed 5, accepted 0, committed 1 (bonus only).
        m.record_window(5, 0, 1);
        assert_eq!(m.windows, 2);
        assert_eq!(m.proposed_total, 12);
        assert_eq!(m.accepted_total, 3);
        assert_eq!(m.committed_total, 5);
        assert!((m.tau() - 1.5).abs() < 1e-6); // 3 accepted / 2 windows
        assert!((m.accept_rate() - 0.25).abs() < 1e-6); // 3 / 12
        assert!((m.mean_draft_len() - 6.0).abs() < 1e-6); // 12 / 2
        assert!((m.mean_committed() - 2.5).abs() < 1e-6); // 5 / 2
        assert_eq!(m.acceptance_hist[3], 1);
        assert_eq!(m.acceptance_hist[0], 1);
    }

    #[test]
    fn empty_is_zero() {
        let m = SpecMetrics::new(4);
        assert_eq!(m.tau(), 0.0);
        assert_eq!(m.accept_rate(), 0.0);
        assert_eq!(m.mean_draft_len(), 0.0);
        assert_eq!(m.to_json()["windows"], 0);
    }

    #[test]
    fn hist_clamps_out_of_range() {
        let mut m = SpecMetrics::new(2); // hist width 3 (0..=2)
        m.record_window(5, 5, 6); // accepted 5 > 2 → not recorded in hist, but totals count
        assert_eq!(m.accepted_total, 5);
        assert_eq!(m.acceptance_hist.iter().sum::<usize>(), 0);
    }
}
