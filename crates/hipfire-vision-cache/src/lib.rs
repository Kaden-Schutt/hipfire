// SPDX-License-Identifier: Apache-2.0
// hipfire — vision embedding cache (content-addressed on-disk LRU). See LICENSE / NOTICE.

//! Content-addressed, on-disk LRU cache for **projected vision embeddings**.
//!
//! The SigLIP encode is the dominant per-request cost of a multimodal request,
//! and video makes it `K×` (`K` frames = `K` encodes). The same image/frame
//! recurs constantly — re-runs, repeated frames across a clip, multi-turn — so
//! caching the *post-projector* rows lets a repeat submission skip the tower
//! (SigLIP + projector + the host download) entirely. That beats any further
//! encode-kernel shaving (see `docs/plans/2026-06-20-medgemma-vision-pipeline-goals.md`).
//!
//! This crate is **GPU-free and inference-hot-path-free**: pure `std` + an
//! `xxh64` content hash. It knows nothing about how the rows were produced; the
//! daemon hashes the submitted image bytes, probes here, and on a miss encodes
//! and inserts.
//!
//! ## Key (128-bit composite)
//! - `ns_hash = xxh64(namespace)` — the namespace is the vision-config / arch
//!   identity (arch id + tower/projector config). Folding it into the key means
//!   embeddings from different models/towers can **never alias**.
//! - `img_hash = xxh64(submitted image bytes)` — the pre-decode bytes, so the
//!   hash is computed at submission, before any preprocessing.
//!
//! ## Value
//! [`CachedEmbedding`] — the `mm_tokens × text_hidden` post-projector rows
//! (e.g. `256×2560`), `f32`. Cache values stay `f32`: each is small relative to
//! the budget and exactness keeps a hit byte-identical to a miss.
//!
//! ## On-disk layout
//! A directory store, one payload file per entry plus a small manifest:
//! - `<dir>/<nshex>-<imghex>.vrow` — a [`VROW_MAGIC`] header (hashes, shape, an
//!   `xxh64` payload checksum) followed by the `f32`-LE payload. Written
//!   atomically (tmp + rename). The checksum makes a truncated/corrupt entry
//!   detectable, so it is treated as a **miss**, never a panic.
//! - `<dir>/manifest` — the LRU index (access counter, budget, per-entry size +
//!   last-access). Atomically replaced; rebuilt by scanning `.vrow` headers if
//!   it is missing or corrupt, so the cache survives a lost manifest.
//!
//! LRU is approximate across restarts: a `get` bumps last-access **in memory**
//! only (no write per hit); the manifest is flushed on `insert`/evict — the
//! rare, heavy ops — which captures the current in-memory access order.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::hash::Hasher;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use twox_hash::XxHash64;

/// Magic for a per-entry payload file: "HipFire Vision Cache".
const VROW_MAGIC: &[u8; 4] = b"HFVC";
/// Magic for the manifest file: "HipFire Vision Manifest".
const MANIFEST_MAGIC: &[u8; 4] = b"HFVM";
const FORMAT_VERSION: u32 = 1;
/// `magic(4) + version(4) + ns(8) + img(8) + rows(4) + cols(4) + payload_hash(8) + payload_len(8)`.
const VROW_HEADER_LEN: usize = 48;

/// `xxh64` of `bytes` with the given seed.
fn xxh64_seeded(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = XxHash64::with_seed(seed);
    h.write(bytes);
    h.finish()
}

/// `xxh64` of `bytes` with the default seed.
fn xxh64(bytes: &[u8]) -> u64 {
    xxh64_seeded(0, bytes)
}

/// A 128-bit content-addressed cache key: `(namespace hash, image-bytes hash)`.
///
/// The namespace component is what prevents embeddings from different vision
/// towers / model arches from aliasing each other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub ns_hash: u64,
    pub img_hash: u64,
}

impl CacheKey {
    /// Build a key from the namespace (vision-config / arch identity) and the
    /// submitted image bytes (pre-decode).
    pub fn new(namespace: &str, image_bytes: &[u8]) -> Self {
        CacheKey {
            ns_hash: xxh64(namespace.as_bytes()),
            img_hash: xxh64(image_bytes),
        }
    }

    fn file_stem(&self) -> String {
        format!("{:016x}-{:016x}", self.ns_hash, self.img_hash)
    }

    fn file_name(&self) -> String {
        format!("{}.vrow", self.file_stem())
    }
}

/// A cached projected-embedding block: `n_rows × n_cols` row-major `f32`.
#[derive(Clone, PartialEq, Debug)]
pub struct CachedEmbedding {
    pub n_rows: usize,
    pub n_cols: usize,
    /// Row-major, length `n_rows * n_cols`.
    pub data: Vec<f32>,
}

impl CachedEmbedding {
    pub fn new(n_rows: usize, n_cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            n_rows * n_cols,
            "CachedEmbedding data length {} != n_rows*n_cols {}",
            data.len(),
            n_rows * n_cols
        );
        CachedEmbedding {
            n_rows,
            n_cols,
            data,
        }
    }

    fn payload_len(&self) -> u64 {
        (self.data.len() * 4) as u64
    }
}

/// Hit/miss counters (cheap, lock-free), for the cache-hit observability the
/// pipeline goal calls for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Entries skipped/dropped because their on-disk payload was missing or
    /// failed its checksum (counted as misses too).
    pub corrupt: u64,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    size_bytes: u64,
    last_access: u64,
    n_rows: u32,
    n_cols: u32,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<CacheKey, Entry>,
    access_counter: u64,
    total_bytes: u64,
}

/// A content-addressed, on-disk LRU cache for projected vision embeddings.
pub struct VisionCache {
    dir: PathBuf,
    max_bytes: u64,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    corrupt: AtomicU64,
}

impl VisionCache {
    /// Open (creating if needed) the cache rooted at `dir`, capped at
    /// `max_bytes` of payload. Loads the manifest if present, else rebuilds the
    /// index by scanning `.vrow` headers.
    pub fn open(dir: impl AsRef<Path>, max_bytes: u64) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let inner = match read_manifest(&dir.join("manifest")) {
            Some(inner) => inner,
            None => rebuild_index(&dir)?,
        };
        let cache = VisionCache {
            dir,
            max_bytes,
            inner: Mutex::new(inner),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            corrupt: AtomicU64::new(0),
        };
        // A smaller cap than what is on disk may require an immediate trim.
        {
            let mut guard = cache.inner.lock().unwrap();
            cache.evict_to_budget(&mut guard, None)?;
            cache.persist(&guard)?;
        }
        Ok(cache)
    }

    /// Look up by key. Returns the cached rows on a hit, `None` on a miss.
    pub fn get(&self, key: &CacheKey) -> io::Result<Option<CachedEmbedding>> {
        let mut guard = self.inner.lock().unwrap();
        let present = guard.entries.contains_key(key);
        if !present {
            drop(guard);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        // Bump recency in memory (flushed on the next insert/evict).
        guard.access_counter += 1;
        let access = guard.access_counter;
        match self.read_payload(key)? {
            Some(embedding) => {
                if let Some(e) = guard.entries.get_mut(key) {
                    e.last_access = access;
                }
                drop(guard);
                self.hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(embedding))
            }
            None => {
                // Payload vanished or failed its checksum: drop the entry and
                // report a miss so the caller re-encodes.
                if let Some(e) = guard.entries.remove(key) {
                    guard.total_bytes = guard.total_bytes.saturating_sub(e.size_bytes);
                }
                let _ = fs::remove_file(self.dir.join(key.file_name()));
                self.persist(&guard)?;
                drop(guard);
                self.corrupt.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    /// Convenience: hash `namespace` + `image_bytes` and look up.
    pub fn get_with(
        &self,
        namespace: &str,
        image_bytes: &[u8],
    ) -> io::Result<Option<CachedEmbedding>> {
        self.get(&CacheKey::new(namespace, image_bytes))
    }

    /// Insert (or replace) the rows for `key`, then evict LRU entries until the
    /// store is within budget. The just-inserted entry is never evicted, and at
    /// least one entry is always retained.
    pub fn insert(&self, key: &CacheKey, embedding: &CachedEmbedding) -> io::Result<()> {
        let size_bytes = embedding.payload_len();
        self.write_payload(key, embedding)?;

        let mut guard = self.inner.lock().unwrap();
        if let Some(old) = guard.entries.remove(key) {
            guard.total_bytes = guard.total_bytes.saturating_sub(old.size_bytes);
        }
        guard.access_counter += 1;
        let access = guard.access_counter;
        guard.total_bytes += size_bytes;
        guard.entries.insert(
            *key,
            Entry {
                size_bytes,
                last_access: access,
                n_rows: embedding.n_rows as u32,
                n_cols: embedding.n_cols as u32,
            },
        );
        self.evict_to_budget(&mut guard, Some(*key))?;
        self.persist(&guard)?;
        Ok(())
    }

    /// Convenience: hash `namespace` + `image_bytes` and insert.
    pub fn insert_with(
        &self,
        namespace: &str,
        image_bytes: &[u8],
        embedding: &CachedEmbedding,
    ) -> io::Result<()> {
        self.insert(&CacheKey::new(namespace, image_bytes), embedding)
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total payload bytes currently held.
    pub fn total_bytes(&self) -> u64 {
        self.inner.lock().unwrap().total_bytes
    }

    /// Configured budget.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Snapshot of the hit/miss counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            corrupt: self.corrupt.load(Ordering::Relaxed),
        }
    }

    /// Drop every entry and the manifest.
    pub fn clear(&self) -> io::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        for key in guard.entries.keys().cloned().collect::<Vec<_>>() {
            let _ = fs::remove_file(self.dir.join(key.file_name()));
        }
        guard.entries.clear();
        guard.total_bytes = 0;
        let _ = fs::remove_file(self.dir.join("manifest"));
        Ok(())
    }

    /// Evict the least-recently-accessed entries until within budget, never
    /// removing `keep` and never dropping below one entry.
    fn evict_to_budget(&self, inner: &mut Inner, keep: Option<CacheKey>) -> io::Result<()> {
        while inner.total_bytes > self.max_bytes && inner.entries.len() > 1 {
            // Pick the lowest last_access entry that isn't `keep`.
            let victim = inner
                .entries
                .iter()
                .filter(|(k, _)| Some(**k) != keep)
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| *k);
            let Some(victim) = victim else { break };
            if let Some(e) = inner.entries.remove(&victim) {
                inner.total_bytes = inner.total_bytes.saturating_sub(e.size_bytes);
            }
            let _ = fs::remove_file(self.dir.join(victim.file_name()));
        }
        Ok(())
    }

    /// Write a payload file atomically (tmp + rename).
    fn write_payload(&self, key: &CacheKey, embedding: &CachedEmbedding) -> io::Result<()> {
        let mut payload = Vec::with_capacity(embedding.data.len() * 4);
        for v in &embedding.data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let payload_hash = xxh64(&payload);

        let mut buf = Vec::with_capacity(VROW_HEADER_LEN + payload.len());
        buf.extend_from_slice(VROW_MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&key.ns_hash.to_le_bytes());
        buf.extend_from_slice(&key.img_hash.to_le_bytes());
        buf.extend_from_slice(&(embedding.n_rows as u32).to_le_bytes());
        buf.extend_from_slice(&(embedding.n_cols as u32).to_le_bytes());
        buf.extend_from_slice(&payload_hash.to_le_bytes());
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&payload);

        let final_path = self.dir.join(key.file_name());
        let tmp_path = self.dir.join(format!("{}.tmp", key.file_stem()));
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Read + validate a payload file. `Ok(None)` on a missing / corrupt entry.
    fn read_payload(&self, key: &CacheKey) -> io::Result<Option<CachedEmbedding>> {
        let path = self.dir.join(key.file_name());
        let mut f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut header = [0u8; VROW_HEADER_LEN];
        if f.read_exact(&mut header).is_err() {
            return Ok(None);
        }
        if &header[0..4] != VROW_MAGIC {
            return Ok(None);
        }
        let n_rows = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
        let n_cols = u32::from_le_bytes(header[28..32].try_into().unwrap()) as usize;
        let payload_hash = u64::from_le_bytes(header[32..40].try_into().unwrap());
        let payload_len = u64::from_le_bytes(header[40..48].try_into().unwrap()) as usize;
        if payload_len != n_rows * n_cols * 4 {
            return Ok(None);
        }
        let mut payload = vec![0u8; payload_len];
        if f.read_exact(&mut payload).is_err() {
            return Ok(None);
        }
        if xxh64(&payload) != payload_hash {
            return Ok(None);
        }
        let data: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Some(CachedEmbedding {
            n_rows,
            n_cols,
            data,
        }))
    }

    /// Atomically rewrite the manifest from the in-memory index.
    fn persist(&self, inner: &Inner) -> io::Result<()> {
        let mut buf = Vec::with_capacity(32 + inner.entries.len() * 40);
        buf.extend_from_slice(MANIFEST_MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&inner.access_counter.to_le_bytes());
        buf.extend_from_slice(&self.max_bytes.to_le_bytes());
        buf.extend_from_slice(&(inner.entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // pad
        for (key, e) in &inner.entries {
            buf.extend_from_slice(&key.ns_hash.to_le_bytes());
            buf.extend_from_slice(&key.img_hash.to_le_bytes());
            buf.extend_from_slice(&e.size_bytes.to_le_bytes());
            buf.extend_from_slice(&e.last_access.to_le_bytes());
            buf.extend_from_slice(&e.n_rows.to_le_bytes());
            buf.extend_from_slice(&e.n_cols.to_le_bytes());
        }
        let final_path = self.dir.join("manifest");
        let tmp_path = self.dir.join("manifest.tmp");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }
}

/// Parse a manifest file into an in-memory index. `None` if absent or corrupt
/// (the caller then rebuilds by scanning payload headers).
fn read_manifest(path: &Path) -> Option<Inner> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 32 || &bytes[0..4] != MANIFEST_MAGIC {
        return None;
    }
    let access_counter = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let entry_count = u32::from_le_bytes(bytes[24..28].try_into().ok()?) as usize;
    let mut inner = Inner {
        access_counter,
        ..Default::default()
    };
    let mut off = 32;
    for _ in 0..entry_count {
        if off + 40 > bytes.len() {
            return None;
        }
        let ns_hash = u64::from_le_bytes(bytes[off..off + 8].try_into().ok()?);
        let img_hash = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().ok()?);
        let size_bytes = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().ok()?);
        let last_access = u64::from_le_bytes(bytes[off + 24..off + 32].try_into().ok()?);
        let n_rows = u32::from_le_bytes(bytes[off + 32..off + 36].try_into().ok()?);
        let n_cols = u32::from_le_bytes(bytes[off + 36..off + 40].try_into().ok()?);
        inner.total_bytes += size_bytes;
        inner.entries.insert(
            CacheKey { ns_hash, img_hash },
            Entry {
                size_bytes,
                last_access,
                n_rows,
                n_cols,
            },
        );
        off += 40;
    }
    Some(inner)
}

/// Rebuild the index by scanning `.vrow` headers in `dir` (manifest lost).
fn rebuild_index(dir: &Path) -> io::Result<Inner> {
    let mut inner = Inner::default();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("vrow") {
            continue;
        }
        let mut f = match File::open(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut header = [0u8; VROW_HEADER_LEN];
        if f.read_exact(&mut header).is_err() || &header[0..4] != VROW_MAGIC {
            continue;
        }
        let ns_hash = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let img_hash = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let n_rows = u32::from_le_bytes(header[24..28].try_into().unwrap());
        let n_cols = u32::from_le_bytes(header[28..32].try_into().unwrap());
        let payload_len = u64::from_le_bytes(header[40..48].try_into().unwrap());
        inner.access_counter += 1;
        let access = inner.access_counter;
        inner.total_bytes += payload_len;
        inner.entries.insert(
            CacheKey { ns_hash, img_hash },
            Entry {
                size_bytes: payload_len,
                last_access: access,
                n_rows,
                n_cols,
            },
        );
    }
    Ok(inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn emb(n_rows: usize, n_cols: usize, fill: f32) -> CachedEmbedding {
        CachedEmbedding::new(n_rows, n_cols, vec![fill; n_rows * n_cols])
    }

    #[test]
    fn key_is_deterministic_and_namespace_scoped() {
        let a = CacheKey::new("gemma3-vl/256x2560", b"image-bytes");
        let b = CacheKey::new("gemma3-vl/256x2560", b"image-bytes");
        assert_eq!(a, b, "same ns + bytes must hash identically");

        let diff_ns = CacheKey::new("qwen35-vl/256x2560", b"image-bytes");
        assert_ne!(a, diff_ns, "different namespace must not alias");

        let diff_bytes = CacheKey::new("gemma3-vl/256x2560", b"other-bytes");
        assert_ne!(a, diff_bytes, "different image bytes must not alias");
    }

    #[test]
    fn roundtrip_is_byte_exact() {
        let dir = tempdir().unwrap();
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        let key = CacheKey::new("ns", b"img");
        let mut e = emb(4, 3, 0.0);
        for (i, v) in e.data.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 1.25;
        }
        cache.insert(&key, &e).unwrap();
        let got = cache.get(&key).unwrap().expect("hit");
        assert_eq!(got, e, "round-trip must be byte-exact f32");
    }

    /// Goal-4 evidence for Goal 1: a hit returns exactly what a miss would have
    /// encoded. We stand in for "encode" with a deterministic producer and
    /// assert the cached path is identical.
    #[test]
    fn hit_equals_miss() {
        let dir = tempdir().unwrap();
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        let ns = "gemma3-vl/arch13";
        let image = b"\x89PNG-pretend-image-bytes";

        // Miss path: encode, then insert.
        assert!(cache.get_with(ns, image).unwrap().is_none());
        let encoded = {
            let mut e = emb(256, 8, 0.0);
            for (i, v) in e.data.iter_mut().enumerate() {
                *v = ((i % 97) as f32) / 97.0;
            }
            e
        };
        cache.insert_with(ns, image, &encoded).unwrap();

        // Hit path: must equal what the miss produced.
        let hit = cache.get_with(ns, image).unwrap().expect("hit");
        assert_eq!(hit, encoded, "cache hit must equal the encoded miss output");

        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn miss_on_unknown_key() {
        let dir = tempdir().unwrap();
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        assert!(cache
            .get(&CacheKey::new("ns", b"never-inserted"))
            .unwrap()
            .is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn eviction_holds_budget_and_keeps_newest() {
        let dir = tempdir().unwrap();
        // Each entry is 10 rows * 10 cols * 4 = 400 payload bytes. Budget = 900
        // bytes → at most 2 entries fit.
        let cache = VisionCache::open(dir.path(), 900).unwrap();
        for i in 0..5u32 {
            let key = CacheKey::new("ns", &i.to_le_bytes());
            cache.insert(&key, &emb(10, 10, i as f32)).unwrap();
        }
        assert!(cache.total_bytes() <= 900, "budget must hold after inserts");
        assert!(cache.len() <= 2);
        // The most-recently inserted (i=4) must survive.
        let newest = CacheKey::new("ns", &4u32.to_le_bytes());
        assert!(
            cache.get(&newest).unwrap().is_some(),
            "newest must be retained"
        );
        // The oldest (i=0) must have been evicted.
        let oldest = CacheKey::new("ns", &0u32.to_le_bytes());
        assert!(
            cache.get(&oldest).unwrap().is_none(),
            "oldest must be evicted"
        );
    }

    #[test]
    fn lru_recency_protects_recently_read() {
        let dir = tempdir().unwrap();
        // Budget for exactly 2 entries (400 bytes each, cap 900).
        let cache = VisionCache::open(dir.path(), 900).unwrap();
        let k0 = CacheKey::new("ns", b"0");
        let k1 = CacheKey::new("ns", b"1");
        cache.insert(&k0, &emb(10, 10, 0.0)).unwrap();
        cache.insert(&k1, &emb(10, 10, 1.0)).unwrap();
        // Touch k0 so it becomes most-recently-used.
        assert!(cache.get(&k0).unwrap().is_some());
        // Insert a third → must evict k1 (untouched), not k0 (just read).
        let k2 = CacheKey::new("ns", b"2");
        cache.insert(&k2, &emb(10, 10, 2.0)).unwrap();
        assert!(
            cache.get(&k0).unwrap().is_some(),
            "recently-read entry must survive"
        );
        assert!(
            cache.get(&k1).unwrap().is_none(),
            "untouched LRU entry must be evicted"
        );
    }

    #[test]
    fn persists_across_reopen() {
        let dir = tempdir().unwrap();
        let key = CacheKey::new("ns", b"persistent");
        let e = emb(16, 16, std::f32::consts::PI);
        {
            let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
            cache.insert(&key, &e).unwrap();
            assert_eq!(cache.total_bytes(), e.payload_len());
        }
        // Reopen the same dir: entry + budget accounting survive.
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), e.payload_len());
        assert_eq!(cache.get(&key).unwrap().expect("hit"), e);
    }

    #[test]
    fn reopen_without_manifest_rebuilds_from_payloads() {
        let dir = tempdir().unwrap();
        let key = CacheKey::new("ns", b"rebuild");
        let e = emb(8, 8, 1.0);
        {
            let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
            cache.insert(&key, &e).unwrap();
        }
        fs::remove_file(dir.path().join("manifest")).unwrap();
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        assert_eq!(cache.len(), 1, "index rebuilt from .vrow headers");
        assert_eq!(cache.get(&key).unwrap().expect("hit"), e);
    }

    #[test]
    fn truncated_payload_is_a_miss_not_a_panic() {
        let dir = tempdir().unwrap();
        let key = CacheKey::new("ns", b"corrupt");
        let e = emb(8, 8, 2.0);
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        cache.insert(&key, &e).unwrap();
        // Truncate the payload file mid-data.
        let path = dir.path().join(key.file_name());
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len((VROW_HEADER_LEN + 8) as u64).unwrap();
        // A corrupt entry reads as a miss and is dropped, no panic.
        assert!(cache.get(&key).unwrap().is_none());
        assert_eq!(cache.len(), 0, "corrupt entry dropped");
        assert_eq!(cache.stats().corrupt, 1);
    }

    #[test]
    fn insert_replaces_in_place() {
        let dir = tempdir().unwrap();
        let cache = VisionCache::open(dir.path(), 1 << 30).unwrap();
        let key = CacheKey::new("ns", b"replace");
        cache.insert(&key, &emb(4, 4, 1.0)).unwrap();
        cache.insert(&key, &emb(4, 4, 9.0)).unwrap();
        assert_eq!(cache.len(), 1, "re-insert must not duplicate");
        assert_eq!(cache.total_bytes(), emb(4, 4, 0.0).payload_len());
        let got = cache.get(&key).unwrap().expect("hit");
        assert!(got.data.iter().all(|&v| v == 9.0), "value replaced");
    }
}
