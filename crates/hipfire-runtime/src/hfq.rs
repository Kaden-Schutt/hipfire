// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ (.hfq) file loader for hipfire-native Q4_F16 quantized models.

use crate::llama::{
    f16_to_f32, EmbeddingFormat, LayerWeights, LlamaConfig, LlamaWeights, ModelArch, WeightTensor,
};
use crate::model_load::{load_weights as rt_load_weights, LoadedWeights, WeightSource};
use crate::weight_backend::{
    decode_raw_codec, flat_name_candidates, load_embedding, raw_codec, resolve_lm_head,
    reupload_f16_as_f32, HfqBackend, WeightBackend,
};
use hip_bridge::{HipError, HipResult};
use memmap2::Mmap;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Drop page cache for a file byte range via posix_fadvise(FADV_DONTNEED).
/// On unified-memory APUs (e.g. Strix Halo), mmap'd model data and
/// hipMalloc'd GPU copies share physical RAM — without this, loading
/// a 65 GB model consumes ~130 GB (mmap cache + GPU copy).
/// Note: madvise(MADV_DONTNEED) does NOT work on MAP_SHARED file-backed
/// mappings (memmap2 default). posix_fadvise on the fd does.
#[cfg(unix)]
fn fadvise_dontneed(fd: std::os::unix::io::RawFd, offset: usize, len: usize) {
    unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_fd: i32, _offset: usize, _len: usize) {}

/// The only tensor a head overlay may carry.
const HEAD_TENSOR_NAME: &str = "lm_head.weight";

impl HfqFile {
    /// Start a background parallel cache warmer: N worker threads pread the
    /// data region chunk-sequentially into the page cache while the loader
    /// uploads tensors. Rationale (measured 2026-08-22, gfx1100 + btrfs md0):
    /// a single `FADV_WILLNEED` does not saturate the array (~1.7 GB/s
    /// effective serial reads), while 6 concurrent readers reach ~4.5 GB/s;
    /// overlapping those reads with H2D uploads hides most of the disk time.
    /// No-op when page-cache eviction is enabled (UMA — warming would double
    /// RAM pressure). The returned guard stops the workers on drop.
    #[cfg(unix)]
    pub fn start_cache_warmup(&self) -> CacheWarmerGuard {
        if self.evict_page_cache || self.mostly_page_cached() {
            return CacheWarmerGuard::empty();
        }
        let end = self
            .tensors
            .last()
            .map(|t| t.data_offset + t.data_size)
            .unwrap_or(0);
        if end == 0 {
            return CacheWarmerGuard::empty();
        }
        const CHUNK: usize = 16 << 20;
        // Measured 2026-08-23 on gfx1100 + btrfs md0, time-to-99%-resident for
        // a 5.3 GB mq4 after drop_caches: 4 threads 3.69 GB/s, 8 threads
        // 4.58 GB/s, 12 threads 5.00 GB/s (plateau = array sequential
        // ceiling). 8 captures most of the win without a small army of
        // threads competing with the upload thread.
        const THREADS: usize = 8;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let path = self.path.clone();
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let stop = stop.clone();
            let next = next.clone();
            let path = path.clone();
            // Fallible spawn: under thread exhaustion (RLIMIT_NPROC, cgroup
            // pids.max) `thread::spawn` would panic mid-load and kill the
            // daemon. Degrade instead — fewer warmers only costs disk
            // bandwidth; zero means the upload proceeds unwarmed.
            match std::thread::Builder::new()
                .name("hfq-cache-warmer".into())
                .spawn(move || {
                    let Ok(file) = std::fs::File::open(&path) else {
                        return;
                    };
                    use std::os::unix::io::AsRawFd;
                    let fd = file.as_raw_fd();
                    let mut buf = vec![0u8; CHUNK];
                    loop {
                        if stop.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let off = i * CHUNK;
                        if off >= end {
                            return;
                        }
                        let len = CHUNK.min(end - off);
                        let mut got = 0usize;
                        while got < len {
                            let n = unsafe {
                                libc::pread(
                                    fd,
                                    buf[got..].as_mut_ptr() as *mut libc::c_void,
                                    len - got,
                                    (off + got) as libc::off_t,
                                )
                            };
                            if n <= 0 {
                                return;
                            }
                            got += n as usize;
                        }
                    }
                }) {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        if handles.is_empty() {
            return CacheWarmerGuard::empty();
        }
        CacheWarmerGuard {
            stop,
            handles: Some(handles),
        }
    }

    /// Fraction of data pages already resident in the page cache, via
    /// `mincore` over an untouched shared mapping (mapping does not fault
    /// pages in, so this is free). Used to skip the warmer when a repeat
    /// load would only re-copy an already-resident file.
    #[cfg(unix)]
    pub fn mostly_page_cached(&self) -> bool {
        let end = self
            .tensors
            .last()
            .map(|t| t.data_offset + t.data_size)
            .unwrap_or(0);
        if end == 0 {
            return false;
        }
        let Ok(file) = std::fs::File::open(&self.path) else {
            return false;
        };
        let Ok(mmap) = (unsafe { memmap2::Mmap::map(&file) }) else {
            return false;
        };
        let page = 4096usize;
        let n_pages = mmap.len().div_ceil(page);
        let mut vec = vec![0u8; n_pages];
        let rc = unsafe {
            libc::mincore(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                vec.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return false;
        }
        let resident = vec.iter().filter(|b| *b & 1 != 0).count();
        resident * 100 >= n_pages * 90
    }

    /// [`Self::mostly_page_cached`] memoized per file instance. The probe
    /// maps + mincores the whole multi-GB file (~0.4 s on an 18.8 GB model),
    /// so callers inside a layer sweep must not re-run it per layer — but the
    /// answer belongs to THIS file: a process-global memo goes stale when the
    /// daemon swaps models (a warm model A would make a cold model B take the
    /// zero-copy path and soft-fault serially at ~0.25 GB/s).
    #[cfg(unix)]
    pub fn mostly_page_cached_memo(&self) -> bool {
        match self.pages_resident_memo.get() {
            1 => true,
            2 => false,
            _ => {
                let resident = self.mostly_page_cached();
                self.pages_resident_memo.set(if resident { 1 } else { 2 });
                resident
            }
        }
    }

    /// Non-unix fallback: no mincore probe available; never pre-detected as
    /// cached (callers fall back to the pread path).
    #[cfg(not(unix))]
    pub fn mostly_page_cached_memo(&self) -> bool {
        false
    }

    /// Non-unix fallback: never pre-detected as cached.
    #[cfg(not(unix))]
    fn mostly_page_cached(&self) -> bool {
        false
    }

    /// Non-unix fallback: nothing to warm.
    #[cfg(not(unix))]
    pub fn start_cache_warmup(&self) -> CacheWarmerGuard {
        CacheWarmerGuard::empty()
    }
}

/// Stops the cache-warmup workers when dropped (load finished or abandoned).
pub struct CacheWarmerGuard {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(unix)]
    handles: Option<Vec<std::thread::JoinHandle<()>>>,
    #[cfg(not(unix))]
    handles: Option<()>,
}

impl CacheWarmerGuard {
    fn empty() -> Self {
        Self {
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            handles: None,
        }
    }
}

#[cfg(unix)]
impl Drop for CacheWarmerGuard {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handles) = self.handles.take() {
            for h in handles {
                let _ = h.join();
            }
        }
    }
}

#[derive(Clone)]
pub struct HfqTensorInfo {
    pub name: String,
    pub quant_type: u8, // serialized QuantType byte (e.g. 13=MQ4G256, 15=MQ6G256, 31=MQ5G256); see hipfire-quantize QuantType enum
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}
/// Ordered absolute tensor manifest entry for source identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqTensorManifestEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}

/// Exact immutable source seal for an HFQ file.
///
/// Equality is exact byte/field equality over canonical path, platform file
/// identity (dev/ino), file length, mtime, arch, metadata_json, and the
/// ordered absolute tensor manifest (name, quant_type, shape, group_size,
/// data_offset, data_size). The opened base offset is captured through the
/// absolute data_offset values. This is an immutable value; callers store it
/// behind an `Arc` and never mutate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqSourceIdentity {
    pub canonical_path: std::path::PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    pub manifest: Vec<HfqTensorManifestEntry>,
}

/// Author-recommended sampling defaults baked into a .hfq's
/// `generation_config` metadata, surfaced by [`HfqFile::recommended_sampling`].
/// Each field is independently `Option` so the consumer can fall through to its
/// own default per-knob. `repeat_penalty` is HF's `repetition_penalty` under
/// the daemon's field name.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecommendedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repeat_penalty: Option<f32>,
}

pub struct HfqFile {
    _file: File,
    /// Path used to open the file. Exposed via [`Self::path`] so the
    /// weight pager can open its own file handle for paged reads without
    /// going through this struct (cleanly separates HfqFile's mmap-based
    /// tensor lookup from the pager's pread/io_uring transport).
    path: std::path::PathBuf,
    /// mmap for tensor data access on discrete-GPU systems where GPU VRAM
    /// is separate from system RAM (no double-buffering cost).
    /// `None` on unified-memory APUs (Strix Halo etc.) where mmap pages
    /// and hipMalloc share physical RAM — keeping the mmap alive doubles
    /// memory consumption. Dropped after header/index parsing via
    /// `drop_mmap()`. When `None`, all tensor reads go through `pread`.
    mmap: Option<Mmap>,
    pub arch_id: u32,
    pub metadata_json: String,
    tensors: Vec<HfqTensorInfo>,
    tensor_map: HashMap<String, usize>,
    /// Reusable read buffer for pread-based tensor reads.
    /// Avoids page cache buildup on unified-memory APUs where mmap pages
    /// can't be evicted while the mapping exists (FADV_DONTNEED is ignored
    /// for mmap'd regions per Linux kernel docs).
    pread_buf: std::cell::RefCell<Vec<u8>>,
    /// Whether tensor reads evict their pages after reading
    /// (`posix_fadvise(DONTNEED)`). Defaults to `true` — the historical
    /// behavior, required on unified-memory APUs where cached model pages
    /// compete 1:1 with hipMalloc'd VRAM staging in system RAM. Discrete-GPU
    /// loaders should disable this via [`Self::set_evict_page_cache`] so
    /// repeat loads hit the page cache and kernel readahead works; measured
    /// 2026-08-22 on gfx1100: fadvise-per-tensor forces a full disk re-read
    /// of the model on every load (~1.3 GB/s effective vs multi-GB/s cache).
    evict_page_cache: bool,
    /// Memoized result of [`Self::mostly_page_cached`] for THIS file:
    /// 0 = not probed, 1 = resident, 2 = not resident. Per-instance (not a
    /// process global) so loading a second model in the same daemon probes
    /// its own file instead of inheriting the previous model's answer.
    pages_resident_memo: std::cell::Cell<u8>,
    /// Optional overlay HFQ whose tensors shadow this file's by name (the
    /// REAP load-time splice, SP3). When `Some`, every tensor read method
    /// consults the overlay first and falls back to the base. When `None`
    /// (the common case), every read path is byte-identical to pre-overlay
    /// behavior — each overlay check is gated on `self.overlay.is_some()`.
    overlay: Option<Box<HfqFile>>,
}

impl HfqFile {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let reap_plan = hipfire_config::developer_var("HIPFIRE_REAP_PLAN")
            .ok()
            .map(std::path::PathBuf::from);
        Self::open_with_reap_plan(path, reap_plan.as_deref())
    }

    /// `open` with the REAP plan injected instead of taken from process config.
    ///
    /// Public because the process config is a START-TIME SNAPSHOT
    /// (`hipfire_config::active_or_local_process_config` is a `OnceLock`), so a
    /// test that does `set_var("HIPFIRE_REAP_PLAN")` then `open()` only works if
    /// it happens to be the first thing in the process to read config — an
    /// order-dependent coin flip. Inject the plan here instead.
    pub fn open_with_reap_plan(path: &Path, reap_plan: Option<&Path>) -> std::io::Result<Self> {
        let mut f = Self::open_at_offset(path, 0)?;
        // REAP load-time overlay splice (SP3): when the process policy points
        // at a dir containing `overlay.hfq`, attach it so its re-quantized
        // tensors shadow the base by name. Opened via `open_at_offset` (NOT
        // `open`) so the overlay does NOT recursively auto-attach. A mismatched
        // arch_id is logged and we proceed base-only — the safe default for
        // unrelated model opens that happen to share the process policy.
        if let Some(dir) = reap_plan {
            let ov_path = dir.join("overlay.hfq");
            if ov_path.exists() {
                // NOTE: a failure to attach (unreadable overlay, arch mismatch,
                // missing tensor, or shape mismatch) is logged and we proceed
                // base-only — the safe default for an unrelated model open that
                // merely shares the env var. The tradeoff (tracked on PR #445):
                // if the overlay WAS meant for this model but is broken, the
                // model silently loads UNPRUNED — watch stderr for this WARNING.
                match Self::open_at_offset(&ov_path, 0)
                    .map_err(|e| e.to_string())
                    .and_then(|ov| {
                        let n = ov.tensors.len();
                        f.attach_overlay(ov).map(|_| n)
                    }) {
                    Ok(n) => eprintln!(
                        "reap: overlay ACTIVE — {n} tensor(s) from {ov_path:?} shadow the base"
                    ),
                    Err(e) => eprintln!(
                        "reap: WARNING overlay at {ov_path:?} not attached, \
                         loading UNPRUNED base: {e}"
                    ),
                }
            }
        }
        Ok(f)
    }

    /// Attach a HEAD overlay: a single-tensor `.hfq` (built by
    /// `hipfire-quantize --head-only`) whose `lm_head.weight` shadows the
    /// base's, so one body can serve several head carriers.
    ///
    /// Same guards as [`Self::attach_overlay`] — matching arch_id, the name
    /// must already exist in the base, and the logical shape must match; only
    /// the quant tier may differ. Failure is an ERROR, not a warning: unlike
    /// the REAP path (where proceeding unpruned is a safe default for an
    /// unrelated model that merely shares an env var), a head overlay is
    /// requested explicitly, so silently serving the base's head would hand
    /// back a model the operator did not ask for.
    pub fn attach_head_overlay(&mut self, head_path: &Path) -> Result<(), String> {
        let ov = Self::open_at_offset(head_path, 0)
            .map_err(|e| format!("head overlay {head_path:?}: {e}"))?;
        // A head overlay must contain ONLY head tensors. Without this, passing
        // a full model to --head "succeeds": every name exists in the base with
        // a matching shape, so attach_overlay's guards all pass and the entire
        // model silently shadows itself. Caught by a negative control that
        // passed the base as its own overlay.
        let foreign: Vec<&str> = ov
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| *n != HEAD_TENSOR_NAME)
            .collect();
        if !foreign.is_empty() {
            return Err(format!(
                "head overlay {head_path:?}: expected only `{HEAD_TENSOR_NAME}`, found {} \
                 tensor(s) including `{}` — this looks like a full model, not a \
                 `hipfire-quantize --head-only` build",
                ov.tensors.len(),
                foreign[0],
            ));
        }
        if ov.tensors.is_empty() {
            return Err(format!("head overlay {head_path:?}: contains no tensors"));
        }
        let n = ov.tensors.len();
        self.attach_overlay(ov)
            .map_err(|e| format!("head overlay {head_path:?}: {e}"))?;
        eprintln!("  head overlay: {n} tensor(s) from {head_path:?} shadow the base");
        Ok(())
    }

    /// Attach an overlay whose tensors shadow this file's by name. Used by the
    /// REAP load-time splice (SP3). Errors if arch_id differs (wrong model).
    pub fn attach_overlay(&mut self, overlay: HfqFile) -> Result<(), String> {
        if overlay.arch_id != self.arch_id {
            return Err(format!(
                "reap overlay: arch_id {} != base arch_id {}",
                overlay.arch_id, self.arch_id
            ));
        }
        // A pure-shadow overlay only RE-quantizes existing tensors; it must not
        // introduce names absent from the base, and each shadow must have the
        // SAME logical shape as the base tensor it replaces (only the quant tier
        // — quant_type/group_size/data_size — may differ). The name check catches
        // an overlay built for a DIFFERENT model; the shape check catches two
        // same-arch checkpoints (e.g. a 0.8B vs 9B at the same arch_id) whose
        // tensors collide by name but differ in dimensions — splicing those would
        // silently corrupt the weights. Called BEFORE self.overlay is set, so
        // find_tensor_info searches only the base — correct.
        for ti in &overlay.tensors {
            match self.find_tensor_info(&ti.name) {
                None => {
                    return Err(format!(
                        "reap overlay: tensor '{}' not present in base — overlay likely built for a different model",
                        ti.name
                    ));
                }
                Some(base_ti) if base_ti.shape != ti.shape => {
                    return Err(format!(
                        "reap overlay: tensor '{}' shape {:?} != base shape {:?} — overlay built for a different model",
                        ti.name, ti.shape, base_ti.shape
                    ));
                }
                Some(_) => {}
            }
        }
        self.overlay = Some(Box::new(overlay));
        Ok(())
    }

    /// True when an overlay is attached (its tensors shadow the base).
    pub fn has_overlay(&self) -> bool {
        self.overlay.is_some()
    }

    /// Open an HFQM container that lives inside a larger file, starting at
    /// `base_offset`. Used by the bundled `.mq4-mtp` loader to parse the
    /// MTP section embedded after the trunk's tensor data.
    ///
    /// The whole file is mmap'd, and the HFQM header is read starting at
    /// `base_offset`. All stored offsets inside the HFQM container are
    /// rebased to absolute file offsets (`base_offset + stored_offset`).
    ///
    /// Callers passing `base_offset = 0` go through the canonical [`Self::open`]
    /// entry point.
    pub fn open_at_offset(path: &Path, base_offset: u64) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Sequential access hint: helps the kernel readahead and drop pages sooner.
        #[cfg(unix)]
        {
            mmap.advise(memmap2::Advice::Sequential).ok();
            // Also advise the file descriptor for the data region.
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
            }
        }

        let base = base_offset as usize;
        let file_len = mmap.len();
        // A truncated or corrupt container must surface as an error instead of
        // panicking on an out-of-bounds slice: open_at_offset advertises a
        // fallible signature, so malformed input is an `Err`, not a panic (#578).
        let need = |end: usize, what: &str| -> std::io::Result<()> {
            if end > file_len {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "HfqFile: truncated or corrupt HFQ container: reading {what} \
                         needs {end} bytes but the file is only {file_len}"
                    ),
                ))
            } else {
                Ok(())
            }
        };
        need(base + 32, "the 32-byte header")?;

        // Parse header (32 bytes) at base offset.
        let magic = &mmap[base..base + 4];
        if magic != b"HFQM" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HfqFile: not an HFQ container at offset {base}"),
            ));
        }
        let _version = u32::from_le_bytes(mmap[base + 4..base + 8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[base + 8..base + 12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(mmap[base + 12..base + 16].try_into().unwrap()) as usize;
        // Stored offsets are relative to the container start; rebase to absolute
        // file offsets so all the existing mmap slicing below works unchanged.
        let metadata_offset =
            u64::from_le_bytes(mmap[base + 16..base + 24].try_into().unwrap()) as usize + base;
        let data_offset =
            u64::from_le_bytes(mmap[base + 24..base + 32].try_into().unwrap()) as usize + base;
        // Guard the metadata slice below: a truncated/corrupt container can hold
        // offsets that overrun the file or cross over each other (#578).
        if metadata_offset > data_offset || data_offset > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HfqFile: invalid metadata/data offsets ({metadata_offset}, {data_offset}) \
                     for a {file_len}-byte file"
                ),
            ));
        }

        // Read metadata JSON
        // Metadata ends at the tensor index, which starts right after metadata
        // The tensor index is at metadata_offset + metadata_len
        // We need to find where the index starts - it's right after metadata
        // The index starts with a u32 tensor count
        // Let's scan for it by reading from metadata_offset until we find the tensor count
        let index_start = metadata_offset;
        // First, find the metadata end by looking for the tensor count in the index
        // The metadata is a JSON blob. The index follows immediately.
        // We know data_offset, so index is between metadata_offset and data_offset.
        // The index format starts with n_tensors u32. We need to find where metadata ends.
        // Since we wrote metadata then index, and metadata_offset = 32 (header size),
        // we need the metadata length. Let's parse the JSON to find its end.
        let meta_bytes = &mmap[metadata_offset..data_offset];
        // Find end of JSON by scanning for matching braces
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let mut json_end = 0;
        for (i, &b) in meta_bytes.iter().enumerate() {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' && in_string {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_string = !in_string;
                continue;
            }
            if !in_string {
                if b == b'{' {
                    brace_depth += 1;
                }
                if b == b'}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        json_end = i + 1;
                        break;
                    }
                }
            }
        }
        // A truncated/corrupt container whose metadata JSON never closes its
        // top-level brace leaves `json_end == 0`. Without this guard we'd slice
        // an empty metadata string and read the tensor-index count from the
        // metadata start (the wrong offset) → a bogus `idx_n` that panics the
        // `assert_eq!` below or silently mis-parses the index. Now that overlays
        // (`overlay.hfq`) are user-built artifacts fed through this same parser,
        // surface it as a clean error instead. (`{}` ⇒ json_end == 2, fine.)
        if json_end == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HfqFile: metadata JSON at offset {metadata_offset} is not brace-terminated \
                     (truncated or corrupt HFQ container)"
                ),
            ));
        }
        let metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).to_string();

        // Parse tensor index (follows metadata JSON)
        let mut pos = metadata_offset + json_end;
        need(pos + 4, "the tensor-index count")?;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        if idx_n != n_tensors {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HfqFile: tensor-index count {idx_n} does not match header count {n_tensors}"
                ),
            ));
        }
        pos += 4;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_map = HashMap::new();
        let mut cumulative_offset = data_offset;

        for i in 0..n_tensors {
            need(pos + 2, "a tensor name length")?;
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            need(pos + name_len, "a tensor name")?;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
            pos += name_len;
            need(pos + 2, "a tensor quant type and dimension count")?;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            need(pos + n_dims * 4, "a tensor shape")?;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            need(pos + 12, "a tensor group and data size")?;
            let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            tensor_map.insert(name.clone(), i);
            tensors.push(HfqTensorInfo {
                name,
                quant_type,
                shape,
                group_size,
                data_offset: cumulative_offset,
                data_size,
            });
            cumulative_offset += data_size;
        }
        let me = Self {
            _file: file,
            path: path.to_path_buf(),
            mmap: Some(mmap),
            arch_id,
            metadata_json,
            tensors,
            tensor_map,
            pread_buf: std::cell::RefCell::new(Vec::new()),
            evict_page_cache: true,
            pages_resident_memo: std::cell::Cell::new(0),
            overlay: None,
        };

        // Structural-pillar tripwire: a qwen3.5/3.6 checkpoint MUST carry
        // arch_id 5 (dense) or 6 (MoE) so the daemon's chat-template resolver
        // routes it through the froggeric pillar and its forward dispatches to
        // the qwen35 crate. If the retained source provenance says qwen3* but
        // the stamped arch_id is anything else, the pillar cannot fire — hard-
        // fail at load rather than silently serving a mis-framed model.
        if !matches!(me.arch_id, 5 | 6) {
            if let Some(prov) = me.qwen3_provenance() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "qwen3* model ({prov}) stamped arch_id={}; cannot guarantee froggeric pillar — re-quantize with correct arch_id (5=dense, 6=MoE)",
                        me.arch_id
                    ),
                ));
            }
        }

        Ok(me)
    }

    /// Drop the mmap to free the virtual address mapping. After this call,
    /// `tensor_data()` returns `None` and all reads go through `tensor_data_pread()`.
    ///
    /// On unified-memory APUs (Strix Halo, Steam Deck, etc.), GPU and CPU
    /// share physical RAM. Keeping the mmap alive while hipMalloc copies
    /// tensor data into GPU buffers doubles memory consumption (mmap pages
    /// + GPU copy both resident). Dropping the mmap after header/index
    /// parsing lets the kernel reclaim those pages.
    ///
    /// On discrete-GPU systems this is unnecessary (GPU VRAM is separate),
    /// so callers should only invoke this when UMA is detected.
    pub fn drop_mmap(&mut self) {
        self.mmap = None;
    }

    /// Enable/disable post-read page-cache eviction for this file. See the
    /// `evict_page_cache` field docs. Discrete-GPU loaders call this with
    /// `false` before loading; UMA paths keep the default (`true`).
    pub fn set_evict_page_cache(&mut self, evict: bool) {
        self.evict_page_cache = evict;
    }

    /// Whether reads on this file evict pages after reading.
    pub fn evicts_page_cache(&self) -> bool {
        self.evict_page_cache
    }

    /// Release the pread reuse buffer back to the allocator. After a
    /// large tensor is read (e.g. DeepSeek V4's `head.weight` at ~560 MB on a Q8F16
    /// build), `pread_buf` keeps that capacity for the rest of the
    /// `HfqFile`'s lifetime. On UMA systems where GPU and CPU share
    /// physical RAM, that competes with the routed-expert upload pass —
    /// the difference between fitting and OOM-at-layer-42 on the 88 GB
    /// deepseek4-q8-mtp build.
    ///
    /// Call between load phases when you know subsequent reads will be
    /// much smaller than the peak. Safe at any time; the next pread
    /// auto-grows the buffer as needed.
    pub fn shrink_pread_buf(&self) {
        let mut buf = self.pread_buf.borrow_mut();
        buf.clear();
        buf.shrink_to_fit();
    }

    /// Path the HFQ file was opened from. The weight pager uses this to
    /// open its own file handle for paged reads — keeping the pager's
    /// transport independent of this struct's lifetime / mmap.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The upstream HuggingFace Jinja `chat_template` baked into this
    /// .hfq's `tokenizer_config` metadata. `None` when the source model
    /// did not ship a chat_template (rare for instruct models, common
    /// for base models). The runtime renders this when present so prompt
    /// framing matches the model's training-time expectation; absent or
    /// failing renders fall back to the hand-rolled `prompt_frame` path.
    pub fn chat_template(&self) -> Option<String> {
        let meta: serde_json::Value = serde_json::from_str(&self.metadata_json).ok()?;
        let ct = meta.get("tokenizer_config")?.get("chat_template")?;
        // Plain-string form: return it directly.
        if let Some(s) = ct.as_str() {
            return Some(s.to_string());
        }
        // List form (Cohere-style {name, template}): fall back to the "default"
        // entry (else the first) so we don't return None on the list shape.
        self.chat_template_named("default")
    }

    /// Select a named chat template. Cohere-style models ship `chat_template`
    /// as a LIST of {name, template}; older models ship a plain string. Returns
    /// the requested name, else the "default" entry, else the first; for a plain
    /// string, returns it directly.
    pub fn chat_template_named(&self, name: &str) -> Option<String> {
        let meta: serde_json::Value = serde_json::from_str(&self.metadata_json).ok()?;
        let ct = meta.get("tokenizer_config")?.get("chat_template")?;
        if let Some(s) = ct.as_str() {
            return Some(s.to_string());
        }
        let arr = ct.as_array()?;
        let pick = |want: &str| {
            arr.iter()
                .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(want))
                .and_then(|t| t.get("template"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        };
        pick(name).or_else(|| pick("default")).or_else(|| {
            arr.first()
                .and_then(|t| t.get("template"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
    }

    /// Author-recommended sampling defaults baked into this .hfq's
    /// `generation_config` metadata. The quantizer copies the source model's
    /// `generation_config.json` verbatim into `metadata.generation_config`
    /// (hipfire-quantize/src/main.rs); HF stores the model author's preferred
    /// inference defaults there (`temperature`, `top_p`, `top_k`,
    /// `repetition_penalty`). This is the lowest-priority, *baked* layer of the
    /// sampling-inheritance ladder — the daemon prefers an explicit per-request
    /// value or the curated registry `recommended_settings` over it, and uses
    /// it only as a fallback ahead of the hardcoded arch ladder. Every field is
    /// independently `Option`: absent keys stay `None` so the daemon falls
    /// through to its own default for just that one knob.
    ///
    /// Note the key remap: HF's `repetition_penalty` is surfaced here as
    /// `repeat_penalty`, matching the daemon's sampler field name (the daemon
    /// reads the `repeat_penalty` request key, not `repetition_penalty`).
    /// Returns `None` only when the metadata is unparseable or carries no
    /// `generation_config` block at all.
    pub fn recommended_sampling(&self) -> Option<RecommendedSampling> {
        let meta: serde_json::Value = serde_json::from_str(&self.metadata_json).ok()?;
        // `generation_config` is `null` when the source model shipped no
        // generation_config.json (the quantizer writes `Option` → JSON null).
        let gc = meta.get("generation_config")?;
        if gc.is_null() {
            return None;
        }
        let f32_at = |k: &str| gc.get(k).and_then(|v| v.as_f64()).map(|x| x as f32);
        let u32_at = |k: &str| gc.get(k).and_then(|v| v.as_u64()).map(|x| x as u32);
        Some(RecommendedSampling {
            temperature: f32_at("temperature"),
            top_p: f32_at("top_p"),
            top_k: u32_at("top_k"),
            // HF spells it `repetition_penalty`; map to the daemon's name.
            repeat_penalty: f32_at("repetition_penalty"),
        })
    }

    /// Detect whether this .hfq's retained source provenance indicates a
    /// qwen3.5 / qwen3.6 model. The quantizer writes the original family
    /// identity into the .hfq metadata blob: the safetensors path stores
    /// `metadata.architecture` (= `config.model_type`) and the full
    /// `metadata.config` (which carries `architectures[]` and `model_type`);
    /// the GGUF path stores `metadata.architecture` (e.g. "qwen3moe") and
    /// `metadata.config.model_type`. This scans all of those for the same
    /// qwen3_5/qwen3.5/qwen3_6/qwen3.6 substrings used by
    /// `safetensors_source::detect_arch_id` so the load-time pillar tripwire
    /// can flag a qwen3* checkpoint whose `arch_id` header was mis-stamped
    /// (i.e. not 5/6). Returns the matched provenance string for the error
    /// message, or `None` when no qwen3* marker is present (or metadata is
    /// unparseable / absent).
    pub fn qwen3_provenance(&self) -> Option<String> {
        let meta: serde_json::Value = serde_json::from_str(&self.metadata_json).ok()?;
        let is_qwen35 = |s: &str| {
            let l = s.to_lowercase();
            l.contains("qwen3_5")
                || l.contains("qwen3.5")
                || l.contains("qwen3_6")
                || l.contains("qwen3.6")
        };
        // metadata.architecture (model_type str for safetensors; GGUF arch str).
        if let Some(a) = meta.get("architecture").and_then(|v| v.as_str()) {
            if is_qwen35(a) {
                return Some(format!("architecture={a}"));
            }
        }
        // metadata.config.model_type and metadata.config.architectures[].
        let cfg = meta.get("config");
        if let Some(mt) = cfg
            .and_then(|c| c.get("model_type"))
            .and_then(|v| v.as_str())
        {
            if is_qwen35(mt) {
                return Some(format!("model_type={mt}"));
            }
        }
        if let Some(archs) = cfg
            .and_then(|c| c.get("architectures"))
            .and_then(|v| v.as_array())
        {
            for v in archs {
                if let Some(a) = v.as_str() {
                    if is_qwen35(a) {
                        return Some(format!("architectures={a}"));
                    }
                }
            }
        }
        None
    }

    /// Resolve a tensor name, trying common prefix variants.
    ///
    /// Qwen3.5 safetensors-converted files store tensors under
    /// `model.language_model.layers.N.` while the canonical GGUF-derived
    /// hipfire-quantize path produces `model.layers.N.`. Callers consistently
    /// pass one prefix style; this helper tries the exact name first, then
    /// strips or adds the `model.language_model.` prefix so a model file
    /// from either pipeline loads cleanly. Returns `None` only when no
    /// variant matches — the per-callsite `?` early-return is preserved.
    fn resolve_idx(&self, name: &str) -> Option<usize> {
        if let Some(&idx) = self.tensor_map.get(name) {
            return Some(idx);
        }
        // Strip "model.language_model." → "model."
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let short = format!("model.{rest}");
            if let Some(&idx) = self.tensor_map.get(&short) {
                return Some(idx);
            }
        }
        // Add "model.language_model." prefix: "model.X" → "model.language_model.X"
        if let Some(rest) = name.strip_prefix("model.") {
            let long = format!("model.language_model.{rest}");
            if let Some(&idx) = self.tensor_map.get(&long) {
                return Some(idx);
            }
        }
        // Try with `model.` / `model.language_model.` added when name has no
        // `model.` prefix at all (e.g. `lm_head.weight`).
        if !name.starts_with("model.") {
            let with_model = format!("model.{name}");
            if let Some(&idx) = self.tensor_map.get(&with_model) {
                return Some(idx);
            }
            let with_lm = format!("model.language_model.{name}");
            if let Some(&idx) = self.tensor_map.get(&with_lm) {
                return Some(idx);
            }
        }
        None
    }

    /// Look up a tensor's metadata (name, quant_type, shape, byte offset/size)
    /// without copying its data. The weight pager calls this at load time to
    /// register byte ranges without forcing eager VRAM allocation.
    pub fn find_tensor_info(&self, name: &str) -> Option<&HfqTensorInfo> {
        // Overlay-first resolution (SP3): an attached overlay shadows the base
        // by name. Gated on `is_some()` so the no-overlay path is unchanged.
        if self.overlay.is_some() {
            if let Some(ov) = &self.overlay {
                if let Some(info) = ov.find_tensor_info(name) {
                    return Some(info);
                }
            }
        }
        let idx = self.resolve_idx(name)?;
        Some(&self.tensors[idx])
    }

    pub fn tensor_data(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        // Overlay-first resolution (SP3): if the overlay has this tensor,
        // return its data; otherwise fall through to the base.
        if self.overlay.is_some() {
            if let Some(ov) = &self.overlay {
                if ov.find_tensor_info(name).is_some() {
                    return ov.tensor_data(name);
                }
            }
        }
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];
        // NOTE: `self.mmap` may legitimately be None here — UMA loads drop the
        // mapping in prepare() and callers probe tensor_data() first, falling
        // back to tensor_data_vec()/tensor_data_pread() on None. Do NOT
        // assert-fail debug builds on that expected path.
        let mmap = self.mmap.as_ref()?;
        Some((
            info,
            &mmap[info.data_offset..info.data_offset + info.data_size],
        ))
    }

    /// Read tensor data via pread into a reusable buffer, then FADV_DONTNEED
    /// the file range. On unified-memory APUs (Strix Halo etc.), mmap pages
    /// can't be evicted while the mapping exists, so pread + fadvise is the
    /// only way to prevent page cache from starving hipMalloc.
    ///
    /// Returns (info, guard) where guard derefs to `&[u8]`. The buffer is
    /// reused across calls — the previous data is overwritten.
    #[cfg(unix)]
    pub fn tensor_data_pread(
        &self,
        name: &str,
    ) -> Option<(&HfqTensorInfo, std::cell::Ref<'_, Vec<u8>>)> {
        use std::os::unix::io::AsRawFd;
        // Overlay-first resolution (SP3): the overlay reads from its own fd /
        // pread_buf, so the returned guard borrows the overlay, not the base.
        if self.overlay.is_some() {
            if let Some(ov) = &self.overlay {
                if ov.find_tensor_info(name).is_some() {
                    return ov.tensor_data_pread(name);
                }
            }
        }
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];
        let fd = self._file.as_raw_fd();
        {
            let mut buf = self.pread_buf.borrow_mut();
            buf.resize(info.data_size, 0);
            let mut total_read = 0usize;
            while total_read < info.data_size {
                let n = unsafe {
                    libc::pread(
                        fd,
                        buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                        info.data_size - total_read,
                        (info.data_offset + total_read) as libc::off_t,
                    )
                };
                if n <= 0 {
                    break;
                }
                total_read += n as usize;
            }
            // Evict these pages from cache — works because pread doesn't hold a
            // mapping. Skipped when the loader disabled eviction (discrete-GPU
            // loads want the page cache warm for repeat loads).
            if self.evict_page_cache {
                fadvise_dontneed(fd, info.data_offset, info.data_size);
            }
        }
        Some((info, self.pread_buf.borrow()))
    }

    /// Non-unix fallback: just delegates to mmap-based tensor_data.
    #[cfg(not(unix))]
    pub fn tensor_data_pread(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        // Overlay-first resolution (SP3). `tensor_data` already consults the
        // overlay, so delegating preserves it; the explicit guard keeps this
        // symmetric with the unix path.
        if self.overlay.is_some() {
            if let Some(ov) = &self.overlay {
                if ov.find_tensor_info(name).is_some() {
                    return ov.tensor_data_pread(name);
                }
            }
        }
        self.tensor_data(name)
    }

    /// Read tensor data using the best available path:
    /// - Unix with pread support: pread + fadvise_dontneed (avoids page cache buildup)
    /// - Fallback: mmap slice (returns None if mmap was dropped)
    ///
    /// Returns owned Vec<u8> to avoid lifetime issues with the pread RefCell.
    pub fn tensor_data_vec(&self, name: &str) -> Option<(&HfqTensorInfo, Vec<u8>)> {
        // Overlay-first resolution (SP3): the returned info reference borrows
        // the overlay (which lives as long as `&self`), and the Vec is owned.
        if self.overlay.is_some() {
            if let Some(ov) = &self.overlay {
                if ov.find_tensor_info(name).is_some() {
                    return ov.tensor_data_vec(name);
                }
            }
        }
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self._file.as_raw_fd();
            let mut buf = vec![0u8; info.data_size];
            let mut total_read = 0usize;
            while total_read < info.data_size {
                let n = unsafe {
                    libc::pread(
                        fd,
                        buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                        info.data_size - total_read,
                        (info.data_offset + total_read) as libc::off_t,
                    )
                };
                if n <= 0 {
                    break;
                }
                total_read += n as usize;
            }
            if self.evict_page_cache {
                fadvise_dontneed(fd, info.data_offset, info.data_size);
            }
            return Some((info, buf));
        }

        #[cfg(not(unix))]
        {
            let mmap = self.mmap.as_ref()?;
            Some((
                info,
                mmap[info.data_offset..info.data_offset + info.data_size].to_vec(),
            ))
        }
    }

    /// Borrowed mmap byte range `[offset, offset+len)`. Zero-copy accessor for
    /// callers that verified a set of tensors is contiguous in-file (via
    /// `find_tensor_info` offsets) and want one slice covering all of them —
    /// e.g. packed-expert blob uploads straight from the page cache.
    ///
    /// Returns `None` when the mapping was dropped (UMA / post-`drop_mmap`) or
    /// when an overlay is attached (a range may span two files). Bounds-checked.
    pub fn data_range(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if self.overlay.is_some() {
            return None;
        }
        let mmap = self.mmap.as_ref()?;
        let end = offset.checked_add(len)?;
        if end > mmap.len() {
            return None;
        }
        Some(&mmap[offset..end])
    }

    /// Release page cache for a byte range. Only works if the range is NOT mmap'd.
    #[allow(dead_code)]
    pub fn drop_pages_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if self.evict_page_cache {
                fadvise_dontneed(self._file.as_raw_fd(), offset, len);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
        }
    }

    /// Return the (start_offset, end_offset) byte range covering all tensors
    /// whose name contains `prefix.` (e.g. "layers.5.").
    #[allow(dead_code)]
    pub fn layer_data_range(&self, prefix: &str) -> Option<(usize, usize)> {
        let needle = format!("{prefix}.");
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for t in &self.tensors {
            if t.name.contains(&needle) {
                lo = lo.min(t.data_offset);
                hi = hi.max(t.data_offset + t.data_size);
            }
        }
        if lo < hi {
            Some((lo, hi))
        } else {
            None
        }
    }

    fn find_tensor(&self, name: &str) -> Option<&HfqTensorInfo> {
        self.resolve_idx(name).map(|i| &self.tensors[i])
    }

    /// Returns the name of the first tensor whose `quant_type` matches `qt`,
    /// or `None` if none match. Used by the daemon's DFlash-refusal guard to
    /// detect MQ3/MQ2 body weights without iterating the index outside this
    /// module.
    pub fn first_tensor_with_quant_type(&self, qt: u8) -> Option<&str> {
        self.tensors
            .iter()
            .find(|t| t.quant_type == qt)
            .map(|t| t.name.as_str())
    }

    /// All tensors in index order. For tools that scan the file (e.g.
    /// dump_norms, quant_quality_mse, compare_hfq) — the engine itself
    /// looks tensors up by name via `find_tensor_info` /
    /// `tensor_data_vec`.
    pub fn tensors(&self) -> &[HfqTensorInfo] {
        &self.tensors
    }

    /// Compute the exact immutable source identity for this opened HFQ file.
    ///
    /// Equality is over canonical path, platform file identity (dev/ino on
    /// Unix), file length, mtime, arch_id, exact metadata_json, and the
    /// ordered absolute tensor manifest (name, quant_type, shape, group_size,
    /// data_offset, data_size). The base offset is captured via the absolute
    /// data_offset values. This must be called before any EP GPU allocation
    /// so the admitted seal binds the exact bytes that will be loaded.
    pub fn load_identity(&self) -> HipResult<HfqSourceIdentity> {
        let canonical_path =
            std::fs::canonicalize(&self.path).unwrap_or_else(|_| self.path.clone());
        let meta = std::fs::metadata(&self.path).map_err(|e| {
            HipError::new(
                0,
                &format!("HfqFile::load_identity: metadata({:?}): {}", self.path, e),
            )
        })?;
        let len = meta.len();
        #[cfg(unix)]
        let (dev, ino) = {
            use std::os::unix::fs::MetadataExt;
            (meta.dev(), meta.ino())
        };
        #[cfg(not(unix))]
        let (dev, ino) = (0u64, 0u64);
        let (mtime_secs, mtime_nanos) = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| (d.as_secs() as i64, d.subsec_nanos()))
            .unwrap_or((0, 0));
        let manifest = self
            .tensors
            .iter()
            .map(|t| HfqTensorManifestEntry {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_offset: t.data_offset,
                data_size: t.data_size,
            })
            .collect();
        Ok(HfqSourceIdentity {
            canonical_path,
            dev,
            ino,
            len,
            mtime_secs,
            mtime_nanos,
            arch_id: self.arch_id,
            metadata_json: self.metadata_json.clone(),
            manifest,
        })
    }

    /// Convenience: load the source identity wrapped in an immutable `Arc`.
    pub fn load_identity_arc(&self) -> HipResult<std::sync::Arc<HfqSourceIdentity>> {
        self.load_identity().map(std::sync::Arc::new)
    }
}

// ─── ModelSource impl for HfqFile ───────────────────────────────────────────

impl crate::model_source::ModelSource for HfqFile {
    fn metadata_json(&self) -> &str {
        &self.metadata_json
    }

    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn quant_config(&self) -> Option<&crate::model_source::QuantConfig> {
        None // HFQ files encode quant_type per-tensor, not via a global config
    }

    fn tensor_data(&self, name: &str) -> Option<(&crate::model_source::TensorInfo, &[u8])> {
        // HfqFile's tensor_data returns (&HfqTensorInfo, &[u8]).
        // We can't return a reference to a TensorInfo we don't own,
        // so this method is not directly usable for HFQ. The HFQ path
        // continues to use HfqFile's native methods; ModelSource is
        // primarily for safetensors. See the blanket adapter below.
        None
    }

    fn tensor_info(&self, name: &str) -> Option<&crate::model_source::TensorInfo> {
        None // HFQ uses its own HfqTensorInfo type
    }

    fn tensor_names(&self) -> Vec<&str> {
        // Overlay-first resolution (SP3): the union of base ∪ overlay names,
        // deduped. Deterministic order = base index order, then any
        // overlay-only names sorted, so callers that sort get a stable result.
        if let Some(ov) = &self.overlay {
            let mut names: Vec<&str> = self.tensors.iter().map(|t| t.name.as_str()).collect();
            let base: std::collections::HashSet<&str> = names.iter().copied().collect();
            let mut extra: Vec<&str> = ov
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .filter(|n| !base.contains(n))
                .collect();
            extra.sort();
            extra.dedup();
            names.extend(extra);
            return names;
        }
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn chat_template(&self) -> Option<String> {
        // Delegate to HfqFile's own chat_template method
        HfqFile::chat_template(self)
    }
}

// ─── Config from HFQ / safetensors metadata ─────────────────────────────────

#[derive(Deserialize)]
struct RawLlamaConfig {
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_llama_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_llama_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_llama_rope")]
    rope_theta: f32,
    // Real safetensors configs ship `bos_token_id`/`eos_token_id` as either a
    // scalar (most LLaMA/Mistral) or an array (Llama-3.1: `[128001, 128009]`).
    // Keep them as raw Values and resolve to the FIRST element in finalize
    // (uniform with qwen2's `eos_token_id = eos_token_ids[0]`).
    #[serde(default)]
    bos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

fn default_llama_eps() -> f32 {
    1e-5
}
fn default_llama_max_pos() -> usize {
    2048
}
fn default_llama_rope() -> f32 {
    10000.0
}
/// Resolve a scalar-or-array `bos_token_id`/`eos_token_id` to a single token,
/// using the first element of an array (uniform with qwen2). Absent/null/
/// unexpected → default.
fn first_token_or(v: Option<&serde_json::Value>, default: u32) -> u32 {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32).unwrap_or(default),
        Some(serde_json::Value::Array(a)) => a
            .first()
            .and_then(|e| e.as_u64())
            .map(|x| x as u32)
            .unwrap_or(default),
        _ => default,
    }
}

/// Shared finalize for the LLaMA-family config parsers. `has_qk_norm` is a
/// tensor-presence probe that can't go through a `&str`-only deserializer, so
/// the caller supplies it.
fn llama_config_from_value(
    config: &serde_json::Value,
    has_qk_norm: bool,
) -> Result<LlamaConfig, String> {
    let raw: RawLlamaConfig = serde_json::from_value(config.clone())
        .map_err(|e| format!("llama: parsing config failed: {e}"))?;
    let arch = match raw.model_type.as_str() {
        "llama" | "mistral" => ModelArch::Llama,
        "qwen3" | "qwen2" => ModelArch::Qwen3,
        _ => ModelArch::Llama,
    };
    let n_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
    let head_dim = raw
        .head_dim
        .unwrap_or(raw.hidden_size / raw.num_attention_heads);
    Ok(LlamaConfig {
        arch,
        dim: raw.hidden_size,
        hidden_dim: raw.intermediate_size,
        n_layers: raw.num_hidden_layers,
        n_heads: raw.num_attention_heads,
        n_kv_heads,
        vocab_size: raw.vocab_size,
        head_dim,
        norm_eps: raw.rms_norm_eps,
        max_seq_len: raw.max_position_embeddings,
        rope_freq_base: raw.rope_theta,
        bos_token: first_token_or(raw.bos_token_id.as_ref(), 1),
        eos_token: first_token_or(raw.eos_token_id.as_ref(), 2),
        has_qk_norm,
    })
}

pub fn config_from_hfq(hfq: &HfqFile) -> Result<LlamaConfig, String> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|e| format!("llama: metadata_json invalid: {e}"))?;
    let config = meta
        .get("config")
        .ok_or_else(|| "llama: metadata_json missing `config`".to_string())?;
    let has_qk_norm = hfq
        .find_tensor("model.layers.0.self_attn.q_norm.weight")
        .is_some();
    llama_config_from_value(config, has_qk_norm)
}

// ─── Weight Loading ─────────────────────────────────────────────────────────

/// Load a tensor as F32 on GPU (for norms, embeddings).
fn load_f16_tensor(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    st_name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => {
            // F16
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        2 => {
            // F32
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => panic!(
            "expected F16/F32 tensor for {st_name}, got quant_type={}",
            info.quant_type
        ),
    };

    gpu.upload_f32(&f32_data, shape)
}

/// Load an AWQ scale sidecar tensor from an HFQ file onto GPU.
///
/// Phase A Stage A — AWQ sidecar lookup. The quantizer emits per-tensor
/// sidecars named `<weight_name>.awq_scale.weight` (1D F16, length K)
/// alongside MQ4-quantized weights. The forward path uses these to apply
/// `x /= awq_scale` before the rotation kernel, completing the AWQ
/// math `(W·s) · (x/s) = W·x`. Backward-compatible: when no sidecar
/// exists (the common case for pre-Stage-A .hfq files), this returns
/// None and the runtime behaves identically to before.
///
/// Naming convention: replace trailing `.weight` with `.awq_scale.weight`.
/// Matches hipfire-quantize's emit pattern.
///
/// Internally uses `tensor_data_vec`, which on Unix routes through pread
/// + fadvise_dontneed (avoids page cache buildup on unified-memory APUs)
/// and on non-Unix falls back to mmap. Sidecars are small (K ≤ ~12288
/// elements, ~48 KB peak), so the owned-Vec copy is negligible.
pub fn load_awq_scale(hfq: &HfqFile, gpu: &Gpu, weight_name: &str, k: usize) -> Option<GpuTensor> {
    let sidecar_name = match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    };
    let (sc_info, sc_data) = hfq.tensor_data_vec(&sidecar_name)?;
    // Must be 1D F16, length K. quant_type 1 = F16 per the existing
    // load_f16_tensor path.
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return None;
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return None;
    }
    // Convert F16 → F32 on host before upload, so the kernel receives
    // a `const float*` and doesn't need <hip/hip_fp16.h>. The 2× VRAM
    // cost vs raw F16 is negligible at these sizes.
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_bytes.len()]).ok()
}

/// Load a weight tensor (quantized or F16) onto GPU.
pub(crate) fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    let st_name = candidates(name)
        .into_iter()
        .find(|c| hfq.find_tensor_info(c).is_some())
        .ok_or_else(|| HipError::new(0, &format!("tensor not found: {name}")))?;
    let (info, data) = hfq
        .tensor_data(&st_name)
        .ok_or_else(|| HipError::new(0, &format!("tensor not found: {st_name}")))?;

    let mut wt = match info.quant_type {
        1 => {
            // F16 — the HFQ path host-decodes to F32 for the F32 GEMV (NOT a
            // verbatim upload, so not a RAW_CODECS row; dequant_weight_raw keeps F16).
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok::<WeightTensor, HipError>(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        other => match raw_codec(other) {
            Some(c) => decode_raw_codec(gpu, c, data, m, k, &st_name),
            None => Err(HipError::new(
                0,
                &format!("unsupported quant_type {other} for weight {st_name}"),
            )),
        },
    }?;
    // Centralized AWQ sidecar attachment. Replaces the prior per-arm
    // inline `load_awq_scale()` calls at the qt=13 / qt=17 arms — those
    // were the only loaders touching `awq_scale` and missing arms (qt=15
    // MQ6, qt=18 MQ2, qt=19/20 Lloyd, qt=24 MFP4) would silently drop
    // sidecars if added later. Routed through `DType::supports_awq_sidecar`
    // so future widening is a single helper edit, not a scattered
    // per-loader hunt. See dispatch.rs for the allow-list rationale.
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, &st_name, k);
    }
    Ok(wt)
}

/// Pread-safe variant of [`load_weight_tensor`]: uses `tensor_data_vec` so it
/// works after `drop_mmap()`. Signature matches the `HfqBackend::read_proj`
/// function-pointer type. Used by sidecar loaders that call `drop_mmap()`
/// before loading (e.g. the DSpark qwen3 sidecar loader on UMA).
pub fn load_weight_tensor_pread(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    let st_name = candidates(name)
        .into_iter()
        .find(|c| hfq.find_tensor_info(c).is_some())
        .ok_or_else(|| HipError::new(0, &format!("tensor not found: {name}")))?;
    let (info, data) = hfq
        .tensor_data_vec(&st_name)
        .ok_or_else(|| HipError::new(0, &format!("tensor not found: {st_name}")))?;

    let mut wt = match info.quant_type {
        1 => {
            // F16 — host-decode to F32 for the F32 GEMV path.
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok::<WeightTensor, HipError>(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        other => match raw_codec(other) {
            Some(c) => decode_raw_codec(gpu, c, &data, m, k, &st_name),
            None => Err(HipError::new(
                0,
                &format!("unsupported quant_type {other} for weight {st_name}"),
            )),
        },
    }?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, &st_name, k);
    }
    Ok(wt)
}

/// Llama-family HFQ weight source. Single-GPU only in practice; preserves the
/// pre-refactor behaviour exactly — no mmap drop, no tied-lm_head alias
/// (always reupload), flat name layout, `norm_bias = 0.0`.
struct LlamaHfqSource<'a> {
    hfq: &'a HfqFile,
    cfg: &'a LlamaConfig,
}
impl WeightSource for LlamaHfqSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    fn prepare(&mut self, _n_devices: usize) -> HipResult<()> {
        Ok(()) // llama keeps the mmap (read-only `&HfqFile`); matches pre-refactor.
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        load_embedding_llama(self.hfq, gpu, self.cfg)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        load_f16_tensor(self.hfq, gpu, "model.norm.weight", &[self.cfg.dim])
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let cfg = self.cfg;
        let hfq = self.hfq;
        let has_separate = hfq.find_tensor("lm_head.weight").is_some();
        resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            cfg.vocab_size,
            cfg.dim,
            |gpu| {
                load_weight_tensor(
                    hfq,
                    gpu,
                    "lm_head.weight",
                    cfg.vocab_size,
                    cfg.dim,
                    flat_name_candidates,
                )
            },
            |gpu| {
                let data = hfq.tensor_data("model.embed_tokens.weight").unwrap().1;
                reupload_f16_as_f32(gpu, &data, cfg.vocab_size, cfg.dim)
            },
        )
    }

    fn read_layer(&mut self, gpu: &mut Gpu, i: usize) -> HipResult<LayerWeights> {
        let cfg = self.cfg;
        let q_out_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        eprintln!("  loading layer {i}/{} ...", cfg.n_layers);
        let mut b = HfqBackend {
            hfq: self.hfq,
            gpu,
            norm_bias: 0.0,
            candidates: flat_name_candidates,
            read_proj: load_weight_tensor,
            layer: i,
        };
        load_layer(&mut b, cfg, q_out_dim, kv_dim, i)
    }

    fn free_layer(&mut self, gpu: &mut Gpu, layer: LayerWeights) {
        layer.free_gpu(gpu);
    }
}

/// Load llama-family `model.embed_tokens.weight` and classify its embedding
/// format. Verbatim extraction of the former inline block in `load_weights_hfq`.
fn load_embedding_llama(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    config: &LlamaConfig,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    eprintln!("  loading token_embd...");
    let (info, data) = hfq
        .tensor_data("model.embed_tokens.weight")
        .ok_or_else(|| HipError::new(0, "llama: embed_tokens not found"))?;
    // Q4K embeddings are llama-family-only (GGUF-derived). qwen2/qwen35 have no
    // Q4K embedding-lookup kernel — that is why the shared `load_embedding` /
    // `embed_classify` deliberately rejects qt 4 (rejecting at load gives a clean
    // error instead of an "unsupported embedding format" panic deep in the qwen
    // forward pass). So Q4K stays an explicit llama-only branch here; everything
    // else delegates to the shared loader, which also handles the bf16 (qt 16)
    // case that the old `load_f16_tensor` path silently panicked on.
    if info.quant_type == 4 {
        eprintln!("    (Q4K raw, {} MB)", data.len() / 1_000_000);
        let buf = gpu.upload_raw(data, &[data.len()])?;
        return Ok((buf, EmbeddingFormat::Q4K));
    }
    load_embedding(gpu, info.quant_type, data, config.vocab_size, config.dim)
}

/// Load LLaMA weights from an HFQ file onto GPU.
pub fn load_weights_hfq(
    hfq: &HfqFile,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    // R2 guard: the LLaMA-family loader does NOT read Q/K/V proj bias —
    // `LayerWeights` has no `wq_bias` / `wk_bias` / `wv_bias` fields and
    // the per-layer load below only names `*.q_proj.weight`. Qwen2
    // requires those biases (`attention_bias=true` is the modeling
    // default). The quantiser used to auto-tag every Qwen2 model as
    // `arch_id=1`, which the daemon dispatches to this loader; the
    // result was silently-wrong outputs with no warning. As of the
    // `--arch-id` flag (see `hipfire-quantize`), Qwen2 models should be
    // tagged `arch_id=7` and dispatched to `hipfire-arch-qwen2`.
    //
    // If we see `q_proj.bias` while loading as the LLaMA family, the
    // input is a mis-tagged Qwen2 HFQ. Refuse hard with a pointer at
    // the correct path. (Detection by manifest is robust to either the
    // model_type tag or the model family — both LLaMA and Qwen3 lack
    // these bias tensors, so any HFQ with `model.layers.0.self_attn.q_proj.bias`
    // is by definition a Qwen2-family input.)
    if hfq
        .find_tensor_info("model.layers.0.self_attn.q_proj.bias")
        .is_some()
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "refusing to load Qwen2 HFQ through the LLaMA family path: \
                 tensor `model.layers.0.self_attn.q_proj.bias` is present, \
                 which means this is a Qwen2 (attention_bias=true) model. \
                 The LLaMA loader drops Q/K/V proj bias and would produce \
                 wrong outputs. \
                 Current HFQ arch_id = {}. Re-quantise with \
                 `hipfire-quantize --arch-id 7 ...` so the daemon \
                 dispatches arch_id=7 to hipfire-arch-qwen2 (once that \
                 crate is wired in), or — for inspection only — load \
                 directly via `cargo run --example inspect_hfq -p \
                 hipfire-arch-qwen2 -- <path>`. \
                 See docs/plans/dots-ocr-devlog.md §7.",
                hfq.arch_id
            ),
        ));
    }

    let mut source = LlamaHfqSource { hfq, cfg: config };
    let layout = crate::model_load::Layout::single(config.n_layers);
    let LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    } = rt_load_weights(&mut source, std::slice::from_mut(gpu), &layout)?;
    Ok(LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    })
}

struct LayerLoadStaging {
    attn_norm: Option<GpuTensor>,
    wq: Option<WeightTensor>,
    wk: Option<WeightTensor>,
    wv: Option<WeightTensor>,
    wo: Option<WeightTensor>,
    q_norm: Option<GpuTensor>,
    k_norm: Option<GpuTensor>,
    ffn_norm: Option<GpuTensor>,
    w_gate: Option<WeightTensor>,
    w_up: Option<WeightTensor>,
    w_down: Option<WeightTensor>,
}

impl LayerLoadStaging {
    fn free_gpu(&mut self, gpu: &mut Gpu) {
        if let Some(tensor) = self.attn_norm.take() {
            let _ = gpu.free_tensor(tensor);
        }
        if let Some(weight) = self.wq.take() {
            weight.free_all(gpu);
        }
        if let Some(weight) = self.wk.take() {
            weight.free_all(gpu);
        }
        if let Some(weight) = self.wv.take() {
            weight.free_all(gpu);
        }
        if let Some(weight) = self.wo.take() {
            weight.free_all(gpu);
        }
        if let Some(tensor) = self.q_norm.take() {
            let _ = gpu.free_tensor(tensor);
        }
        if let Some(tensor) = self.k_norm.take() {
            let _ = gpu.free_tensor(tensor);
        }
        if let Some(tensor) = self.ffn_norm.take() {
            let _ = gpu.free_tensor(tensor);
        }
        if let Some(weight) = self.w_gate.take() {
            weight.free_all(gpu);
        }
        if let Some(weight) = self.w_up.take() {
            weight.free_all(gpu);
        }
        if let Some(weight) = self.w_down.take() {
            weight.free_all(gpu);
        }
    }

    fn into_layer(&mut self) -> LayerWeights {
        LayerWeights {
            attn_norm: self.attn_norm.take().expect("staged layer attn_norm"),
            wq: self.wq.take().expect("staged layer wq"),
            wk: self.wk.take().expect("staged layer wk"),
            wv: self.wv.take().expect("staged layer wv"),
            wo: self.wo.take().expect("staged layer wo"),
            q_norm: self.q_norm.take(),
            k_norm: self.k_norm.take(),
            ffn_norm: self.ffn_norm.take().expect("staged layer ffn_norm"),
            w_gate: self.w_gate.take().expect("staged layer w_gate"),
            w_up: self.w_up.take().expect("staged layer w_up"),
            w_down: self.w_down.take().expect("staged layer w_down"),
        }
    }
}

/// Single llama per-layer walk over a `WeightBackend`. Dense-only (no MoE,
/// no DeltaNet). `q_out_dim`/`kv_dim` are passed in so the caller reuses the
/// exact dims it already computes.
pub fn load_layer<B: WeightBackend>(
    b: &mut B,
    config: &crate::llama::LlamaConfig,
    q_out_dim: usize,
    kv_dim: usize,
    i: usize,
) -> HipResult<LayerWeights> {
    let mut staged = LayerLoadStaging {
        attn_norm: None,
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        q_norm: None,
        k_norm: None,
        ffn_norm: None,
        w_gate: None,
        w_up: None,
        w_down: None,
    };
    let result = (|| -> HipResult<LayerWeights> {
        b.set_layer(i);
        staged.attn_norm = Some(b.norm("input_layernorm.weight", &[config.dim])?);
        staged.wq = Some(b.proj("self_attn.q_proj", q_out_dim, config.dim)?);
        staged.wk = Some(b.proj("self_attn.k_proj", kv_dim, config.dim)?);
        staged.wv = Some(b.proj("self_attn.v_proj", kv_dim, config.dim)?);
        staged.wo = Some(b.proj("self_attn.o_proj", config.dim, q_out_dim)?);
        if config.has_qk_norm {
            staged.q_norm = Some(b.norm("self_attn.q_norm.weight", &[config.head_dim])?);
            staged.k_norm = Some(b.norm("self_attn.k_norm.weight", &[config.head_dim])?);
        }
        staged.ffn_norm = Some(b.norm("post_attention_layernorm.weight", &[config.dim])?);
        staged.w_gate = Some(b.proj("mlp.gate_proj", config.hidden_dim, config.dim)?);
        staged.w_up = Some(b.proj("mlp.up_proj", config.hidden_dim, config.dim)?);
        staged.w_down = Some(b.proj("mlp.down_proj", config.dim, config.hidden_dim)?);
        Ok(staged.into_layer())
    })();
    if result.is_err() {
        staged.free_gpu(b.gpu_mut());
    }
    result
}

// ─── ParoQuant safetensors loading (LLaMA / Qwen3 arch) ────────────────────

/// Parse a LlamaConfig from a SafetensorsSource's metadata JSON.
/// The metadata JSON has structure: `{ "config": { ...config.json... } }`.
pub fn config_from_safetensors_llama(
    source: &dyn crate::model_source::ModelSource,
) -> Result<LlamaConfig, String> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json())
        .map_err(|e| format!("llama: metadata_json invalid: {e}"))?;
    let config = meta
        .get("config")
        .ok_or_else(|| "llama: metadata_json missing `config`".to_string())?;
    let has_qk_norm = source
        .tensor_info("model.layers.0.self_attn.q_norm.weight")
        .is_some();
    llama_config_from_value(config, has_qk_norm)
}

/// Load a ParoQuant-quantized weight tensor from a safetensors source.
/// Repacks AWQ INT4 data to HFQ4G128 and uploads ParoQuant rotation metadata.
fn load_paroquant_weight_from_source(
    source: &dyn crate::model_source::ModelSource,
    gpu: &Gpu,
    tensor_prefix: &str, // e.g. "model.layers.0.mlp.gate_proj"
    out_dim: usize,      // M
    in_dim: usize,       // K
    group_size: u32,
    krot: u8,
) -> HipResult<WeightTensor> {
    use crate::llama::ParoRotation;

    let qw_name = format!("{tensor_prefix}.qweight");
    let qz_name = format!("{tensor_prefix}.qzeros");
    let sc_name = format!("{tensor_prefix}.scales");
    let pairs_name = format!("{tensor_prefix}.pairs");
    let theta_name = format!("{tensor_prefix}.theta");
    let cs_name = format!("{tensor_prefix}.channel_scales");

    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {sc_name}")))?;

    let hfq_data = crate::paro::repack_awq_to_hfq4g128(
        qw_data,
        qz_data,
        sc_data,
        out_dim,
        in_dim,
        group_size as usize,
    );
    let buf = gpu.upload_raw(&hfq_data, &[hfq_data.len()])?;

    let (_, pairs_data) = source
        .tensor_data(&pairs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}")))?;
    let (_, theta_data) = source
        .tensor_data(&theta_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}")))?;
    let (_, cs_data) = source
        .tensor_data(&cs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;

    let pairs = gpu.upload_raw(pairs_data, &[pairs_data.len()])?;
    let theta = gpu.upload_raw(theta_data, &[theta_data.len()])?;
    let channel_scales = gpu.upload_raw(cs_data, &[cs_data.len()])?;

    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::ParoQ4G128,
        m: out_dim,
        k: in_dim,
        row_stride: 0,
        paro: Some(ParoRotation {
            pairs,
            theta,
            channel_scales,
            krot: krot as u32,
            group_size,
            is_alias: false,
        }),
        awq_scale: None,
    })
}

/// Load an FP16 weight tensor from safetensors as F32 on GPU.
fn load_fp16_weight_tensor_from_source(
    source: &dyn crate::model_source::ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    // Handles F16/BF16/F32 (raw HF checkpoints — incl. lm_head — are commonly BF16).
    let f32_data = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load an F16 norm weight as F32 on GPU (raw, no +1.0 bias — HF convention).
fn paro_load_llama_norm_raw(
    source: &dyn crate::model_source::ModelSource,
    gpu: &mut Gpu,
    name: &str, // e.g. "layers.0.input_layernorm.weight"
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let full = format!("model.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    // Handles F16/BF16/F32 (raw HF checkpoints — incl. the final norm — are commonly BF16).
    let v = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
    gpu.upload_f32(&v, shape)
}

/// LLaMA-family ParoQuant source used by the shared whole-model transaction.
struct LlamaParoSource<'a> {
    source: &'a dyn crate::model_source::ModelSource,
    cfg: &'a LlamaConfig,
    group_size: u32,
    krot: u8,
}

impl WeightSource for LlamaParoSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices > 1 {
            return Err(HipError::new(
                0,
                "ParoQuant LLaMA loading is single-GPU only",
            ));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd (LLaMA/Qwen3 safetensors)...");
        let (info, data) = self
            .source
            .tensor_data("model.embed_tokens.weight")
            .ok_or_else(|| HipError::new(0, "PARO tensor not found: embed_tokens not found"))?;
        let f32_data = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
        let token_embd = gpu.upload_f32(&f32_data, &[self.cfg.vocab_size, self.cfg.dim])?;
        Ok((token_embd, EmbeddingFormat::F32))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        paro_load_llama_norm_raw(self.source, gpu, "norm.weight", &[self.cfg.dim])
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let source = self.source;
        let cfg = self.cfg;
        let group_size = self.group_size;
        let krot = self.krot;
        let has_separate = source.tensor_info("lm_head.weight").is_some();
        resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            cfg.vocab_size,
            cfg.dim,
            |gpu| {
                if source.tensor_info("lm_head.qweight").is_some() {
                    load_paroquant_weight_from_source(
                        source,
                        gpu,
                        "lm_head",
                        cfg.vocab_size,
                        cfg.dim,
                        group_size,
                        krot,
                    )
                } else {
                    load_fp16_weight_tensor_from_source(
                        source,
                        gpu,
                        "lm_head.weight",
                        cfg.vocab_size,
                        cfg.dim,
                    )
                }
            },
            |gpu| {
                let (info, data) =
                    source
                        .tensor_data("model.embed_tokens.weight")
                        .ok_or_else(|| {
                            HipError::new(0, "PARO tensor not found: embed_tokens for lm_head")
                        })?;
                let f32_data =
                    crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[cfg.vocab_size, cfg.dim])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m: cfg.vocab_size,
                    k: cfg.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            },
        )
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let cfg = self.cfg;
        let q_out_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        eprintln!(
            "  loading layer {layer_idx}/{} (ParoQuant LLaMA/Qwen3)...",
            cfg.n_layers
        );
        let mut b = crate::weight_backend::ParoBackend {
            source: self.source,
            gpu,
            mp: "model",
            layer: layer_idx,
            norm_bias: 0.0,
        };
        load_layer(&mut b, cfg, q_out_dim, kv_dim, layer_idx)
    }

    fn free_layer(&mut self, gpu: &mut Gpu, layer: LayerWeights) {
        layer.free_gpu(gpu);
    }
}

/// Load LLaMA/Qwen3 weights from a safetensors model — ParoQuant/AWQ *or* raw
/// unquantized FP. ParoQuant is a transparent layer: [`ParoBackend::proj`] tries
/// the `.qweight` augmentors first and falls back to the raw `.weight`
/// (F16/BF16/F32) when no quant sidecar is present, so a checkpoint with no
/// `quantization_config` loads as plain FP. The `group_size`/`krot` below are
/// only consumed by the quant path; raw checkpoints never read them.
///
pub fn load_weights_paroquant_llama(
    source: &dyn crate::model_source::ModelSource,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    let (group_size, krot) = source
        .quant_config()
        .map(|qc| (qc.group_size, qc.krot))
        .unwrap_or((128, 0));
    let mut source = LlamaParoSource {
        source,
        cfg: config,
        group_size,
        krot,
    };
    let layout = crate::model_load::Layout::single(config.n_layers);
    let LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    } = rt_load_weights(&mut source, std::slice::from_mut(gpu), &layout)?;
    Ok(LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    })
}
// ─── HFQM streaming writer (calibration collector) ──────────────────────────

/// One in-memory tensor for [`write_hfqm_package_mem`].
pub struct HfqMemTensor {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data: Vec<u8>,
}

/// Descriptor for one tensor in a streaming HFQM write. `data_len` is the exact
/// payload byte count (deterministic from `shape` + `quant_type`), so the index
/// can be written before any payload is materialized — that is what lets the
/// collector stream multi-GB Hessians one tensor at a time instead of holding
/// them all in RAM.
pub struct HfqStreamEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_len: u64,
}

/// Streaming HFQM writer: write the header + metadata + index up front (payload
/// sizes come from `entries`), then call `write_nth(i, w)` once per entry, in
/// index order, to stream that tensor's `data_len` bytes directly to the file.
/// Only one tensor's payload need exist in memory at a time. `write_nth` MUST
/// write exactly `entries[i].data_len` bytes. This is the canonical HFQM layout
/// impl (the in-memory [`write_hfqm_package_mem`] is a thin wrapper over it).
pub fn write_hfqm_package_streaming(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    entries: &[HfqStreamEntry],
    mut write_nth: impl FnMut(usize, &mut dyn Write) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let meta = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + meta.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        let nb = e.name.as_bytes();
        if nb.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry name too long: {}", e.name),
            ));
        }
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(e.quant_type);
        index.push(e.shape.len() as u8);
        for &d in &e.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&e.group_size.to_le_bytes());
        index.extend_from_slice(&e.data_len.to_le_bytes());
    }
    let data_start = index_offset + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(b"HFQM")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&arch_id.to_le_bytes())?;
    f.write_all(&(entries.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(meta)?;
    f.write_all(&index)?;
    f.write_all(&vec![0u8; (data_offset - data_start) as usize])?;
    for (i, e) in entries.iter().enumerate() {
        let mut counter = CountingWriter {
            inner: &mut f,
            written: 0,
        };
        write_nth(i, &mut counter)?;
        if counter.written != e.data_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQM entry {}: wrote {} bytes, index declared {}",
                    e.name, counter.written, e.data_len
                ),
            ));
        }
    }
    f.flush()?;
    Ok(())
}

struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    written: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write an HFQM container from in-memory tensors. Thin wrapper over
/// [`write_hfqm_package_streaming`] for callers that already hold every payload
/// in RAM (e.g. small sidecars, tests). Large producers (the calibration
/// collector) should stream instead.
pub fn write_hfqm_package_mem(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    tensors: &[HfqMemTensor],
) -> std::io::Result<()> {
    let entries: Vec<HfqStreamEntry> = tensors
        .iter()
        .map(|t| HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape: t.shape.clone(),
            group_size: t.group_size,
            data_len: t.data.len() as u64,
        })
        .collect();
    write_hfqm_package_streaming(path, arch_id, metadata_json, &entries, |i, w| {
        w.write_all(&tensors[i].data)
    })
}

/// HFQM package reader for calibration part files. Thin wrapper over
/// [`HfqFile`] that exposes the collector's `combine_calib_parts` API
/// (`entries()` + `blob_data(name)`) without duplicating index parsing.
pub struct HfqPackage {
    inner: HfqFile,
}

impl HfqPackage {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            inner: HfqFile::open(path)?,
        })
    }
    pub fn entries(&self) -> Vec<HfqPackageEntry> {
        self.inner
            .tensors()
            .iter()
            .map(|t| HfqPackageEntry {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_offset: t.data_offset,
                data_size: t.data_size,
            })
            .collect()
    }
    pub fn blob_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.inner.tensors().iter().find(|t| t.name == name)?;
        self.inner
            .mmap
            .as_ref()
            .map(|m| &m[info.data_offset..info.data_offset + info.data_size] as &[u8])
    }
}
#[derive(Debug, Clone)]
pub struct HfqPackageEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}

#[cfg(test)]
mod llama_config_tests {
    use super::*;

    fn config_value(model_type: &str) -> serde_json::Value {
        serde_json::json!({
            "architecture": "llama",
            "config": {
                "model_type": model_type,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 8192,
                "rope_theta": 500000.0,
                "bos_token_id": 128000,
                "eos_token_id": 128001
            }
        })
    }

    #[test]
    fn full_envelope_every_field() {
        let envelope = config_value("llama");
        let config = envelope.get("config").unwrap();
        let c = llama_config_from_value(config, false).unwrap();
        assert_eq!(c.arch, ModelArch::Llama);
        assert_eq!(c.dim, 4096);
        assert_eq!(c.hidden_dim, 11008);
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.n_heads, 32);
        assert_eq!(c.n_kv_heads, 8);
        assert_eq!(c.vocab_size, 32000);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.norm_eps, 1e-6);
        assert_eq!(c.max_seq_len, 8192);
        assert_eq!(c.rope_freq_base, 500000.0);
        assert_eq!(c.bos_token, 128000);
        assert_eq!(c.eos_token, 128001);
        assert!(!c.has_qk_norm);
    }

    #[test]
    fn arch_enum_mapping() {
        let cases = [
            ("llama", ModelArch::Llama),
            ("mistral", ModelArch::Llama),
            ("qwen3", ModelArch::Qwen3),
            ("qwen2", ModelArch::Qwen3),
            ("something_else", ModelArch::Llama),
        ];
        for (mt, expect) in cases {
            let envelope = config_value(mt);
            let config = envelope.get("config").unwrap();
            let c = llama_config_from_value(config, false).unwrap();
            assert_eq!(c.arch, expect, "model_type {mt:?}");
        }
    }

    #[test]
    fn defaults_when_optional_absent() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000
        });
        let c = llama_config_from_value(&config, false).unwrap();
        assert_eq!(c.norm_eps, 1e-5);
        assert_eq!(c.max_seq_len, 2048);
        assert_eq!(c.rope_freq_base, 10000.0);
        assert_eq!(c.bos_token, 1);
        assert_eq!(c.eos_token, 2);
        assert_eq!(c.n_kv_heads, 32); // defaults to n_heads
        assert_eq!(c.head_dim, 128); // dim / n_heads = 4096 / 32
    }

    #[test]
    fn array_eos_token_id_uses_first_element() {
        // Llama-3.1 ships `eos_token_id` as an array `[128001, 128009]`. The OLD
        // hand-walked parser silently fell back to the default on an array; the
        // serde port (scalar u32) would HARD-ERROR. We now take the first
        // element (uniform with qwen2's `eos_token_id = eos_token_ids[0]`).
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128009]
        });
        let c = llama_config_from_value(&config, false).unwrap();
        assert_eq!(c.eos_token, 128001);
        assert_eq!(c.bos_token, 128000);
    }

    #[test]
    fn missing_required_is_err() {
        let config = serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000
        });
        assert!(llama_config_from_value(&config, false).is_err());
    }

    #[test]
    fn has_qk_norm_passthrough() {
        let envelope = config_value("qwen3");
        let config = envelope.get("config").unwrap();
        let c = llama_config_from_value(config, true).unwrap();
        assert!(c.has_qk_norm);
    }
}

// ─── Overlay resolution tests (SP3) ─────────────────────────────────────────

#[cfg(test)]
mod overlay_tests {
    use super::*;
    use crate::model_source::ModelSource; // for `tensor_names`
    use std::io::Write;

    /// Minimal HFQ writer mirroring `hipfire-quantize`'s `write_hfq`
    /// (`crates/hipfire-quantize/src/main.rs:3398`) byte-for-byte for the
    /// layout `HfqFile::open` parses:
    ///   - 32B header: magic "HFQM", u32 version, u32 arch_id, u32 n_tensors,
    ///     u64 metadata_offset, u64 data_offset.
    ///   - metadata JSON (we emit `{}`; the parser scans matching braces to
    ///     find its end, so it must be valid balanced JSON).
    ///   - index: u32 n; per tensor: u16 name_len, name bytes, u8 quant_type,
    ///     u8 n_dims, n_dims×u32 shape, u32 group_size, u64 data_size.
    ///   - zero padding so the data region starts 4096-aligned.
    ///   - tensor data, concatenated in index order (offsets are derived at
    ///     read time cumulatively from `data_offset`).
    fn write_min_hfq(path: &Path, arch_id: u32, tensors: &[(&str, u8, &[u32], &[u8])]) {
        let metadata = b"{}"; // balanced JSON; brace-scan parser stops at the close brace
        let header_size: u64 = 32;
        let metadata_offset = header_size;
        let index_offset = metadata_offset + metadata.len() as u64;

        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, qt, shape, data) in tensors {
            let nb = name.as_bytes();
            index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            index.extend_from_slice(nb);
            index.push(*qt);
            index.push(shape.len() as u8);
            for &d in *shape {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes()); // group_size
            index.extend_from_slice(&(data.len() as u64).to_le_bytes());
        }

        let data_start_unaligned = index_offset + index.len() as u64;
        let data_offset = (data_start_unaligned + 4095) & !4095;

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // version
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(&index).unwrap();
        let pad = (data_offset - data_start_unaligned) as usize;
        f.write_all(&vec![0u8; pad]).unwrap();
        for (_, _, _, data) in tensors {
            f.write_all(data).unwrap();
        }
        f.flush().unwrap();
    }

    #[test]
    fn truncated_container_errors_instead_of_panicking() {
        // #578: open_at_offset advertises `io::Result` but used to panic on an
        // out-of-bounds slice for a truncated/corrupt container. Every prefix of
        // a valid container that lands in the header/metadata/index region (i.e.
        // below the 4096-aligned data region, which open does not read) must now
        // return Err rather than panic.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.hfq");
        write_min_hfq(&path, 9, &[("A", 3, &[2, 4], &vec![1u8; 8])]);

        // Sanity: the full, valid container opens.
        assert!(HfqFile::open_at_offset(&path, 0).is_ok());

        let full = std::fs::read(&path).unwrap();
        for len in [0usize, 4, 16, 31, 32, 34, 40, 100, 2048, 4095] {
            let p = dir.path().join(format!("trunc_{len}.hfq"));
            std::fs::write(&p, &full[..len]).unwrap();
            assert!(
                HfqFile::open_at_offset(&p, 0).is_err(),
                "truncating to {len} bytes should error, not panic"
            );
        }

        // A header-valid container whose tensor-index name length runs past the
        // end of the file must also error (exercises the index-loop bounds check
        // rather than the header/offset guards).
        let mut crafted = Vec::new();
        crafted.extend_from_slice(b"HFQM");
        crafted.extend_from_slice(&1u32.to_le_bytes()); // version
        crafted.extend_from_slice(&9u32.to_le_bytes()); // arch_id
        crafted.extend_from_slice(&1u32.to_le_bytes()); // n_tensors
        crafted.extend_from_slice(&32u64.to_le_bytes()); // metadata_offset
        crafted.extend_from_slice(&40u64.to_le_bytes()); // data_offset (== file len)
        crafted.extend_from_slice(b"{}"); // metadata JSON (offsets 32..34)
        crafted.extend_from_slice(&1u32.to_le_bytes()); // index tensor count (34..38)
        crafted.extend_from_slice(&9999u16.to_le_bytes()); // name_len past EOF (38..40)
        assert_eq!(crafted.len(), 40);
        let cpath = dir.path().join("bad_namelen.hfq");
        std::fs::write(&cpath, &crafted).unwrap();
        assert!(HfqFile::open_at_offset(&cpath, 0).is_err());
    }

    #[test]
    fn overlay_tensor_shadows_base() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.hfq");
        let ov = dir.path().join("overlay.hfq");
        // base has tensorA (qt=3) + tensorB (qt=3); overlay re-quantizes A to qt=8.
        write_min_hfq(
            &base,
            9,
            &[
                ("A", 3, &[2, 4], &vec![1u8; 2 * 4]),
                ("B", 3, &[2, 4], &vec![2u8; 2 * 4]),
            ],
        );
        write_min_hfq(&ov, 9, &[("A", 8, &[2, 4], &vec![9u8; 2 * 4])]);
        let mut f = HfqFile::open(&base).unwrap();
        f.attach_overlay(HfqFile::open(&ov).unwrap()).unwrap();
        // A resolves to overlay (qt 8, bytes 9); B falls through to base (qt 3, bytes 2).
        let (ia, da) = f.tensor_data_vec("A").unwrap();
        assert_eq!(ia.quant_type, 8);
        assert!(da.iter().all(|&b| b == 9));
        let (ib, db) = f.tensor_data_vec("B").unwrap();
        assert_eq!(ib.quant_type, 3);
        assert!(db.iter().all(|&b| b == 2));
        assert_eq!(f.find_tensor_info("A").unwrap().quant_type, 8);
        // tensor_names is the union (base ∪ overlay), no dup.
        let mut names = f.tensor_names();
        names.sort();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn overlay_arch_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("b.hfq");
        let ov = dir.path().join("o.hfq");
        write_min_hfq(&base, 9, &[("A", 3, &[1, 4], &vec![0u8; 4])]);
        write_min_hfq(&ov, 6, &[("A", 3, &[1, 4], &vec![0u8; 4])]);
        let mut f = HfqFile::open(&base).unwrap();
        let err = f.attach_overlay(HfqFile::open(&ov).unwrap()).unwrap_err();
        assert!(err.contains("arch_id 6 != base arch_id 9"), "got: {err}");
    }

    #[test]
    fn open_auto_attaches_overlay_from_process_policy() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.hfq");
        write_min_hfq(&base, 9, &[("A", 3, &[1, 4], &vec![1u8; 4])]);
        // overlay.hfq lives in the plan dir.
        let plan = tempfile::tempdir().unwrap();
        write_min_hfq(
            &plan.path().join("overlay.hfq"),
            9,
            &[("A", 8, &[1, 4], &vec![7u8; 4])],
        );
        let f = HfqFile::open_with_reap_plan(&base, Some(plan.path())).unwrap();
        assert!(f.has_overlay());
        assert_eq!(f.find_tensor_info("A").unwrap().quant_type, 8); // overlay won
    }

    #[test]
    fn overlay_with_foreign_tensor_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("b.hfq");
        let ov = dir.path().join("o.hfq");
        write_min_hfq(&base, 9, &[("A", 3, &[1, 4], &vec![0u8; 4])]);
        // same arch_id 9, but overlay has a tensor "Z" the base lacks
        write_min_hfq(
            &ov,
            9,
            &[
                ("A", 8, &[1, 4], &vec![1u8; 4]),
                ("Z", 8, &[1, 4], &vec![1u8; 4]),
            ],
        );
        let mut f = HfqFile::open(&base).unwrap();
        let err = f.attach_overlay(HfqFile::open(&ov).unwrap()).unwrap_err();
        assert!(err.contains("'Z' not present in base"), "got: {err}");
    }
}
