// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode telemetry counters (arch-agnostic).
//!
//! Relocated from `hipfire-arch-qwen35::speculative` (P1): process-global
//! atomic counters + their snapshot struct.

use std::sync::atomic::{AtomicU64, Ordering};

/// DDTree meta-verifier pruner telemetry: per-cycle tree-size histogram.
/// `cycle_count` = cycles observed; `total_nodes` = sum of tree.num_nodes()
/// across cycles; `max_nodes` / `min_nodes` = range observed.
static DDTREE_META_CYCLES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_TOTAL_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MAX_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MIN_NODES: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn record_ddtree_meta_nodes(n: usize) {
    let n64 = n as u64;
    DDTREE_META_CYCLES.fetch_add(1, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.fetch_add(n64, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.fetch_max(n64, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.fetch_min(n64, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DdtreeMetaStats {
    pub cycles: u64,
    pub total_nodes: u64,
    pub max_nodes: u64,
    pub min_nodes: u64,
}

pub fn read_ddtree_meta_stats() -> DdtreeMetaStats {
    let c = DDTREE_META_CYCLES.load(Ordering::Relaxed);
    DdtreeMetaStats {
        cycles: c,
        total_nodes: DDTREE_META_TOTAL_NODES.load(Ordering::Relaxed),
        max_nodes: DDTREE_META_MAX_NODES.load(Ordering::Relaxed),
        min_nodes: if c == 0 {
            0
        } else {
            DDTREE_META_MIN_NODES.load(Ordering::Relaxed)
        },
    }
}

pub fn reset_ddtree_meta_stats() {
    DDTREE_META_CYCLES.store(0, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.store(u64::MAX, Ordering::Relaxed);
}
