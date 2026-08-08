// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SwapStore — where a swapped-out session's snapshot lives.
//
// Two tiers: host memory up to a budget, then scratch files. Disk is not
// needed to reach a large session pool on a 125 GiB box — host RAM already
// gets there — so the disk tier exists for the tail, not the common case.
//
// Spill files are SCRATCH: the directory is cleared on construction and each
// file is removed when taken or dropped. They still carry a full stamp, so
// making them durable later is a policy change rather than a format change.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::swap::snapshot::SlotSnapshot;
use crate::swap::SwapError;

enum Entry {
    Host(SlotSnapshot),
    Disk { path: PathBuf, bytes: u64 },
}

pub struct SwapStore {
    dir: PathBuf,
    host_budget: u64,
    host_used: u64,
    entries: HashMap<u64, Entry>,
}

impl SwapStore {
    /// Open a store rooted at `dir`, clearing anything already there.
    ///
    /// Clearing is the point: a leftover file from a previous run describes a
    /// session this process knows nothing about, and restoring one would be
    /// the worst failure class available — plausible bytes, wrong conversation.
    pub fn new(dir: PathBuf, host_budget_bytes: u64) -> Result<Self, SwapError> {
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| SwapError::Io(e.to_string()))?;
        }
        std::fs::create_dir_all(&dir).map_err(|e| SwapError::Io(e.to_string()))?;
        Ok(SwapStore {
            dir,
            host_budget: host_budget_bytes,
            host_used: 0,
            entries: HashMap::new(),
        })
    }

    pub fn host_used(&self) -> u64 {
        self.host_used
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `"host"`, `"disk"`, or `None` when the key is not stored.
    pub fn tier_of(&self, key: u64) -> Option<&'static str> {
        self.entries.get(&key).map(|e| match e {
            Entry::Host(_) => "host",
            Entry::Disk { .. } => "disk",
        })
    }

    fn path_for(&self, key: u64) -> PathBuf {
        self.dir.join(format!("{key}.snap"))
    }

    /// Store a snapshot, in host memory if it fits the budget, else on disk.
    pub fn put(&mut self, key: u64, snap: SlotSnapshot) -> Result<(), SwapError> {
        self.drop_key(key);
        let bytes = snap.payload.len() as u64;
        if self.host_used + bytes <= self.host_budget {
            self.host_used += bytes;
            self.entries.insert(key, Entry::Host(snap));
            return Ok(());
        }
        let path = self.path_for(key);
        let encoded = snap.to_bytes();
        let n = encoded.len() as u64;
        std::fs::write(&path, &encoded).map_err(|e| SwapError::Io(e.to_string()))?;
        self.entries.insert(key, Entry::Disk { path, bytes: n });
        Ok(())
    }

    /// Remove and return a snapshot. The key does not survive a successful
    /// take: a session is either resident or swapped, never both.
    pub fn take(&mut self, key: u64) -> Result<SlotSnapshot, SwapError> {
        match self.entries.remove(&key) {
            None => Err(SwapError::Io(format!("no snapshot for key {key}"))),
            Some(Entry::Host(snap)) => {
                self.host_used = self.host_used.saturating_sub(snap.payload.len() as u64);
                Ok(snap)
            }
            Some(Entry::Disk { path, .. }) => {
                let bytes = std::fs::read(&path).map_err(|e| SwapError::Io(e.to_string()))?;
                // Remove regardless of whether the parse succeeds: a file that
                // will not parse is never going to, and leaving it behind means
                // the scratch directory grows without bound.
                let _ = std::fs::remove_file(&path);
                SlotSnapshot::from_bytes(&bytes)
            }
        }
    }

    /// Forget a key, freeing its budget or deleting its file. Idempotent.
    pub fn drop_key(&mut self, key: u64) {
        match self.entries.remove(&key) {
            Some(Entry::Host(snap)) => {
                self.host_used = self.host_used.saturating_sub(snap.payload.len() as u64);
            }
            Some(Entry::Disk { path, .. }) => {
                let _ = std::fs::remove_file(&path);
            }
            None => {}
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for SwapStore {
    fn drop(&mut self) {
        // Scratch means scratch.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap::snapshot::{checksum_of, SnapshotStamp};

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hipfire-swap-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn small_snap() -> SlotSnapshot {
        let payload = vec![3u8; 128];
        let checksum = checksum_of(&payload);
        SlotSnapshot {
            stamp: SnapshotStamp {
                model_hash: 1,
                kv_dtype_tag: 1,
                per_pos_bytes: 4,
                n_fa_layers: 2,
                dn_layout_version: 1,
                cap: 128,
                dn_bytes: 64,
            },
            seq_len: 8,
            tokens: vec![4, 5, 6],
            payload,
            checksum,
        }
    }

    #[test]
    fn a_snapshot_under_budget_stays_in_host_memory() {
        let mut s = SwapStore::new(tmpdir("host"), 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), Some("host"));
        assert!(s.host_used() > 0);
    }

    #[test]
    fn exceeding_the_budget_spills_to_disk() {
        let mut s = SwapStore::new(tmpdir("disk"), 64).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), Some("disk"));
        assert_eq!(s.host_used(), 0, "a spilled snapshot holds no host budget");
    }

    #[test]
    fn take_returns_the_same_bytes_from_either_tier() {
        for (i, budget) in [1u64 << 20, 64].into_iter().enumerate() {
            let mut s = SwapStore::new(tmpdir(&format!("rt{i}")), budget).unwrap();
            let original = small_snap();
            s.put(1, original.clone()).unwrap();
            let back = s.take(1).unwrap();
            assert_eq!(back.payload, original.payload);
            assert_eq!(back.tokens, original.tokens);
            assert_eq!(back.seq_len, original.seq_len);
            assert_eq!(back.stamp, original.stamp);
        }
    }

    #[test]
    fn take_frees_the_budget_and_the_key() {
        let mut s = SwapStore::new(tmpdir("free"), 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        let _ = s.take(1).unwrap();
        assert_eq!(s.host_used(), 0);
        assert!(s.take(1).is_err(), "a taken key must not resolve twice");
    }

    #[test]
    fn a_truncated_spill_file_is_an_error_not_a_panic() {
        let dir = tmpdir("trunc");
        let mut s = SwapStore::new(dir.clone(), 64).unwrap();
        s.put(1, small_snap()).unwrap();
        let f = std::fs::read_dir(&dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        std::fs::write(&f, b"short").unwrap();
        assert!(s.take(1).is_err());
    }

    #[test]
    fn new_clears_a_stale_directory() {
        let dir = tmpdir("stale");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("leftover.bin"), b"junk").unwrap();
        let s = SwapStore::new(dir.clone(), 64).unwrap();
        assert_eq!(
            std::fs::read_dir(s.dir()).unwrap().count(),
            0,
            "a leftover file describes a session this process knows nothing about"
        );
    }

    #[test]
    fn drop_key_frees_budget_and_is_idempotent() {
        let mut s = SwapStore::new(tmpdir("dropk"), 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        s.drop_key(1);
        assert_eq!(s.host_used(), 0);
        s.drop_key(1);
        assert_eq!(s.tier_of(1), None);
    }
}
