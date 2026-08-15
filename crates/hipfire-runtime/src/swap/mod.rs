// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// KV swap: an idle session's GPU state survives losing its slot.
//
// The governing invariant, from which everything here follows: **the tokens
// are authoritative and KV is only a cache**. Every failure path in this
// module degrades to "re-prefill from the session's tokens" — slow, never
// wrong. There is deliberately no path that yields silently incorrect output,
// because that failure surfaces as a subtly worse agent rather than an error.

pub mod snapshot;
pub mod store;

/// Why a swap operation could not be completed.
///
/// Every variant means the same thing to a caller: drop the snapshot, mark the
/// session cold, and re-prefill. They are distinguished only so the log says
/// something useful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapError {
    /// The snapshot was taken against a different model, dtype or layout.
    Stamp(String),
    /// Length or checksum mismatch — the bytes are not what was written.
    Corrupt(String),
    /// Filesystem failure.
    Io(String),
    /// A device copy failed.
    Gpu(String),
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwapError::Stamp(m) => write!(f, "swap stamp mismatch: {m}"),
            SwapError::Corrupt(m) => write!(f, "swap payload corrupt: {m}"),
            SwapError::Io(m) => write!(f, "swap io error: {m}"),
            SwapError::Gpu(m) => write!(f, "swap gpu error: {m}"),
        }
    }
}

impl std::error::Error for SwapError {}

/// Default host-tier budget. A budget, not a reservation: snapshots are
/// allocated on demand and freed on take. 30 sessions averaging 8K tokens is
/// ~4 GB, so this absorbs a realistic pool while leaving a 125 GiB box room
/// for the model and everything else. Beyond it, snapshots go to disk.
pub const DEFAULT_HOST_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Ties the store to the session table: capture on eviction, restore on
/// resume, and mark the session `Cold` on any failure.
///
/// Deliberately thin. The policy it enforces is the whole of the design:
/// evict only idle sessions, LRU, and never let a swap failure become
/// anything worse than a re-prefill.
pub struct SwapManager {
    store: store::SwapStore,
    evictions: usize,
    restores: usize,
    failures: usize,
}

impl SwapManager {
    pub fn new(dir: std::path::PathBuf, host_budget_bytes: u64) -> Result<Self, SwapError> {
        Ok(SwapManager {
            store: store::SwapStore::new(dir, host_budget_bytes)?,
            evictions: 0,
            restores: 0,
            failures: 0,
        })
    }

    pub fn store(&self) -> &store::SwapStore {
        &self.store
    }

    /// `(evictions, restores, failures)`.
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.evictions, self.restores, self.failures)
    }

    /// Park a snapshot under `key`. A failure here is not fatal: the caller
    /// marks the session cold and it re-prefills next turn.
    pub fn park(&mut self, key: u64, snap: snapshot::SlotSnapshot) -> Result<(), SwapError> {
        match self.store.put(key, snap) {
            Ok(()) => {
                self.evictions += 1;
                Ok(())
            }
            Err(e) => {
                self.failures += 1;
                Err(e)
            }
        }
    }

    /// Retrieve a parked snapshot. On any error the key is dropped, because a
    /// snapshot that failed to load once will not load later.
    pub fn unpark(&mut self, key: u64) -> Result<snapshot::SlotSnapshot, SwapError> {
        match self.store.take(key) {
            Ok(s) => {
                self.restores += 1;
                Ok(s)
            }
            Err(e) => {
                self.failures += 1;
                self.store.drop_key(key);
                Err(e)
            }
        }
    }

    pub fn forget(&mut self, key: u64) {
        self.store.drop_key(key);
    }
}

#[cfg(test)]
mod manager_tests {
    use super::*;
    use crate::swap::snapshot::{checksum_of, SlotSnapshot, SnapshotStamp};

    fn snap(n: usize) -> SlotSnapshot {
        let payload = vec![1u8; n];
        let checksum = checksum_of(&payload);
        SlotSnapshot {
            stamp: SnapshotStamp {
                model_hash: 1,
                kv_dtype_tag: 1,
                per_pos_bytes: 4,
                n_fa_layers: 1,
                dn_layout_version: 1,
                cap: 64,
                dn_bytes: (n - 8) as u64,
            },
            seq_len: 1,
            tokens: vec![1],
            payload,
            checksum,
        }
    }

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hipfire-swapmgr-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn park_then_unpark_returns_the_snapshot_and_counts_both() {
        let mut m = SwapManager::new(dir("rt"), 1 << 20).unwrap();
        m.park(7, snap(64)).unwrap();
        assert_eq!(m.stats().0, 1);
        let back = m.unpark(7).unwrap();
        assert_eq!(back.payload.len(), 64);
        assert_eq!(m.stats().1, 1);
    }

    #[test]
    fn unparking_an_unknown_key_fails_and_is_counted_not_panicked() {
        let mut m = SwapManager::new(dir("miss"), 1 << 20).unwrap();
        assert!(m.unpark(99).is_err());
        assert_eq!(m.stats().2, 1, "the failure must be visible in stats");
    }

    #[test]
    fn a_failed_unpark_drops_the_key_so_it_is_not_retried_forever() {
        let mut m = SwapManager::new(dir("drop"), 1 << 20).unwrap();
        m.park(1, snap(64)).unwrap();
        let _ = m.unpark(1);
        assert!(m.unpark(1).is_err());
        assert_eq!(m.store().tier_of(1), None);
    }
}
