// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode telemetry counters (arch-agnostic).
//!
//! DDTree meta-verifier pruner telemetry. Previously process-global
//! `AtomicU64` statics (P6 anti-pattern: cross-request accumulation, only
//! surfaced via `eprintln!`). Now a **thread-local per-request accumulator**:
//! the daemon spec loop is synchronous (one request per worker thread at a
//! time), so `reset` at request start + `read`/`to_json` at the `done` event
//! gives correct per-request tree-size stats, drained into the unified
//! spec-decode `done` block alongside [`crate::SpecMetrics`]. The record API
//! (`record_ddtree_meta_nodes`) is unchanged, so the deep qwen35 call sites and
//! the `dflash_spec_demo` reader need no signature churn.

use serde_json::{json, Value};
use std::cell::Cell;

thread_local! {
    /// Per-request DDTree meta accumulator: (cycles, total_nodes, max_nodes, min_nodes).
    /// `min_nodes` starts at `u64::MAX` and is normalized to 0 on read when no
    /// cycle was observed.
    static DDTREE_META: Cell<(u64, u64, u64, u64)> = const { Cell::new((0, 0, 0, u64::MAX)) };
}

/// Record one meta-verifier cycle's tree size (`tree.num_nodes()`), into the
/// current thread's per-request accumulator.
pub fn record_ddtree_meta_nodes(n: usize) {
    let n64 = n as u64;
    DDTREE_META.with(|c| {
        let (cycles, total, max, min) = c.get();
        c.set((cycles + 1, total + n64, max.max(n64), min.min(n64)));
    });
}

/// DDTree meta-verifier pruner telemetry: per-request tree-size histogram.
/// `cycles` = cycles observed; `total_nodes` = sum of `tree.num_nodes()` across
/// cycles; `max_nodes` / `min_nodes` = range observed.
#[derive(Debug, Clone, Copy, Default)]
pub struct DdtreeMetaStats {
    pub cycles: u64,
    pub total_nodes: u64,
    pub max_nodes: u64,
    pub min_nodes: u64,
}

impl DdtreeMetaStats {
    /// Mean nodes per cycle (0 when no cycle observed).
    pub fn mean_nodes(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.total_nodes as f32 / self.cycles as f32
        }
    }

    /// Specialized ext block for the unified spec-decode `done` event. `None`
    /// when this request drove no meta-verifier cycle (nothing to report).
    pub fn to_json(&self) -> Option<Value> {
        if self.cycles == 0 {
            return None;
        }
        let r3 = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
        Some(json!({
            "cycles": self.cycles,
            "total_nodes": self.total_nodes,
            "max_nodes": self.max_nodes,
            "min_nodes": self.min_nodes,
            "mean_nodes": r3(self.mean_nodes()),
        }))
    }
}

/// Snapshot the current thread's per-request DDTree meta accumulator.
pub fn read_ddtree_meta_stats() -> DdtreeMetaStats {
    let (cycles, total_nodes, max_nodes, min_nodes) = DDTREE_META.with(|c| c.get());
    DdtreeMetaStats {
        cycles,
        total_nodes,
        max_nodes,
        min_nodes: if cycles == 0 { 0 } else { min_nodes },
    }
}

/// Reset the current thread's per-request DDTree meta accumulator. Call at the
/// start of each spec-decode request.
pub fn reset_ddtree_meta_stats() {
    DDTREE_META.with(|c| c.set((0, 0, 0, u64::MAX)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddtree_meta_accumulates_and_resets_per_thread() {
        // Runs on its own test thread → isolated thread-local.
        reset_ddtree_meta_stats();
        assert_eq!(read_ddtree_meta_stats().cycles, 0);
        assert!(read_ddtree_meta_stats().to_json().is_none()); // no cycle → no block

        record_ddtree_meta_nodes(5);
        record_ddtree_meta_nodes(9);
        record_ddtree_meta_nodes(3);
        let s = read_ddtree_meta_stats();
        assert_eq!(s.cycles, 3);
        assert_eq!(s.total_nodes, 17);
        assert_eq!(s.max_nodes, 9);
        assert_eq!(s.min_nodes, 3);
        assert!((s.mean_nodes() - 17.0 / 3.0).abs() < 1e-6);
        let j = s.to_json().expect("cycles>0 → Some");
        assert_eq!(j["cycles"], 3);
        assert_eq!(j["max_nodes"], 9);
        assert_eq!(j["min_nodes"], 3);

        reset_ddtree_meta_stats();
        assert_eq!(read_ddtree_meta_stats().cycles, 0);
    }
}
