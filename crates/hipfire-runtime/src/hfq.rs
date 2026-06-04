// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFQ (.hfq) file loader for hipfire-native Q4_F16 quantized models.

use crate::llama::{
    f16_to_f32, EmbeddingFormat, LayerWeights, LlamaConfig, LlamaWeights, ModelArch, WeightTensor,
};
use hip_bridge::{HipError, HipResult};
use memmap2::Mmap;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const HFQM_MAGIC: &[u8; 4] = b"HFQM";
pub const HFQM_VERSION: u32 = 1;
/// Reserved `arch_id` for HFQM containers that are not direct model-weight
/// packages. KLD references, imatrix captures, CASK/TriAttention centers, and
/// other sidecar-style artifacts should use this value and describe their
/// semantics through metadata `artifact_kind` plus named package entries.
pub const HFQM_ARCH_NON_WEIGHT_PACKAGE: u32 = 0;

#[derive(Debug, Clone)]
pub struct HfqPackageEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
}

pub struct HfqPackage {
    _file: File,
    mmap: Mmap,
    pub version: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    entries: Vec<HfqPackageEntry>,
    entry_map: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct HfqPackageWriteEntry {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub source_path: std::path::PathBuf,
    pub data_size: u64,
}

fn json_blob_end(bytes: &[u8]) -> Option<usize> {
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
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
            } else if b == b'}' {
                brace_depth -= 1;
                if brace_depth == 0 {
                    return Some(i + 1);
                }
            }
        }
    }
    None
}

fn parse_hfqm_index(
    mmap: &[u8],
    base: usize,
    metadata_offset: usize,
    data_offset: usize,
    n_entries: usize,
) -> std::io::Result<(String, Vec<HfqPackageEntry>, HashMap<String, usize>)> {
    if metadata_offset > data_offset || data_offset > mmap.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid HFQM offsets metadata={metadata_offset} data={data_offset}"),
        ));
    }
    let meta_bytes = &mmap[metadata_offset..data_offset];
    let json_end = json_blob_end(meta_bytes).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HFQM metadata JSON did not end",
        )
    })?;
    let metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).to_string();
    let mut pos = metadata_offset + json_end;
    if pos + 4 > data_offset {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "HFQM index missing tensor count",
        ));
    }
    let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
    if idx_n != n_entries {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HFQM index count {idx_n} != header count {n_entries}"),
        ));
    }
    pos += 4;

    let mut entries = Vec::with_capacity(n_entries);
    let mut entry_map = HashMap::new();
    let mut cumulative_offset = data_offset;
    for i in 0..n_entries {
        if pos + 2 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name length",
            ));
        }
        let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + name_len + 2 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at name/shape header",
            ));
        }
        let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
        pos += name_len;
        let quant_type = mmap[pos];
        pos += 1;
        let n_dims = mmap[pos] as usize;
        pos += 1;
        if pos + n_dims * 4 + 12 > data_offset {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HFQM index truncated at shape/data_size",
            ));
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if cumulative_offset + data_size > mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "HFQM entry {name} data range {}..{} exceeds file size {}",
                    cumulative_offset,
                    cumulative_offset + data_size,
                    mmap.len()
                ),
            ));
        }
        entry_map.insert(name.clone(), i);
        entries.push(HfqPackageEntry {
            name,
            quant_type,
            shape,
            group_size,
            data_offset: cumulative_offset - base,
            data_size,
        });
        cumulative_offset += data_size;
    }
    Ok((metadata_json, entries, entry_map))
}

impl HfqPackage {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 32 || &mmap[0..4] != HFQM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an HFQM package",
            ));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let n_entries = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let (metadata_json, entries, entry_map) =
            parse_hfqm_index(&mmap, 0, metadata_offset, data_offset, n_entries)?;
        Ok(Self {
            _file: file,
            mmap,
            version,
            arch_id,
            metadata_json,
            entries,
            entry_map,
        })
    }

    pub fn entries(&self) -> &[HfqPackageEntry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&HfqPackageEntry> {
        self.entry_map.get(name).map(|&idx| &self.entries[idx])
    }

    pub fn blob_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.entry(name)?;
        Some(&self.mmap[entry.data_offset..entry.data_offset + entry.data_size])
    }
}

pub fn write_hfqm_package_from_files(
    path: &Path,
    arch_id: u32,
    metadata_json: &str,
    entries: &[HfqPackageWriteEntry],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + metadata_bytes.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        let name_bytes = entry.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry name too long: {}", entry.name),
            ));
        }
        if entry.shape.len() > u8::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("HFQM entry has too many dims: {}", entry.name),
            ));
        }
        index.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index.extend_from_slice(name_bytes);
        index.push(entry.quant_type);
        index.push(entry.shape.len() as u8);
        for &dim in &entry.shape {
            index.extend_from_slice(&dim.to_le_bytes());
        }
        index.extend_from_slice(&entry.group_size.to_le_bytes());
        index.extend_from_slice(&entry.data_size.to_le_bytes());
    }
    let data_start_unaligned = index_offset + index.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQM_MAGIC)?;
    f.write_all(&HFQM_VERSION.to_le_bytes())?;
    f.write_all(&arch_id.to_le_bytes())?;
    f.write_all(&(entries.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(metadata_bytes)?;
    f.write_all(&index)?;
    let pad_size = (data_offset - data_start_unaligned) as usize;
    if pad_size > 0 {
        f.write_all(&vec![0u8; pad_size])?;
    }
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    for entry in entries {
        let expected = std::fs::metadata(&entry.source_path)?.len();
        if expected != entry.data_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "HFQM entry {} source size {} != declared {}",
                    entry.name, expected, entry.data_size
                ),
            ));
        }
        let mut src = File::open(&entry.source_path)?;
        src.seek(SeekFrom::Start(0))?;
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
        }
    }
    f.flush()?;
    Ok(())
}

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

pub struct HfqTensorInfo {
    pub name: String,
    pub quant_type: u8, // 0=Q4F16G64, 1=F16, 2=F32
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data_offset: usize,
    pub data_size: usize,
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
}

impl HfqFile {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Self::open_at_offset(path, 0)
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
        assert!(
            base + 32 <= mmap.len(),
            "HfqFile::open_at_offset: base ({base}) + 32 > file size ({}); not enough bytes for header",
            mmap.len(),
        );

        // Parse header (32 bytes) at base offset.
        let magic = &mmap[base..base + 4];
        assert_eq!(magic, b"HFQM", "Not an HFQ container at offset {base}");
        let _version = u32::from_le_bytes(mmap[base + 4..base + 8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[base + 8..base + 12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(mmap[base + 12..base + 16].try_into().unwrap()) as usize;
        // Stored offsets are relative to the container start; rebase to absolute
        // file offsets so all the existing mmap slicing below works unchanged.
        let metadata_offset =
            u64::from_le_bytes(mmap[base + 16..base + 24].try_into().unwrap()) as usize + base;
        let data_offset =
            u64::from_le_bytes(mmap[base + 24..base + 32].try_into().unwrap()) as usize + base;

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
        let metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).to_string();

        // Parse tensor index (follows metadata JSON)
        let mut pos = metadata_offset + json_end;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n_tensors);
        pos += 4;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_map = HashMap::new();
        let mut cumulative_offset = data_offset;

        for i in 0..n_tensors {
            let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
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

        Ok(Self {
            _file: file,
            path: path.to_path_buf(),
            mmap: Some(mmap),
            arch_id,
            metadata_json,
            tensors,
            tensor_map,
            pread_buf: std::cell::RefCell::new(Vec::new()),
        })
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
        meta.get("tokenizer_config")?
            .get("chat_template")?
            .as_str()
            .map(|s| s.to_string())
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
        let idx = self.resolve_idx(name)?;
        Some(&self.tensors[idx])
    }

    pub fn tensor_data(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        let idx = self.resolve_idx(name)?;
        let info = &self.tensors[idx];
        debug_assert!(
            self.mmap.is_some(),
            "tensor_data() called after drop_mmap() — use tensor_data_vec() or tensor_data_pread() instead (tensor: {name})"
        );
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
            // Evict these pages from cache — works because pread doesn't hold a mapping.
            fadvise_dontneed(fd, info.data_offset, info.data_size);
        }
        Some((info, self.pread_buf.borrow()))
    }

    /// Non-unix fallback: just delegates to mmap-based tensor_data.
    #[cfg(not(unix))]
    pub fn tensor_data_pread(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        self.tensor_data(name)
    }

    /// Read tensor data using the best available path:
    /// - Unix with pread support: pread + fadvise_dontneed (avoids page cache buildup)
    /// - Fallback: mmap slice (returns None if mmap was dropped)
    ///
    /// Returns owned Vec<u8> to avoid lifetime issues with the pread RefCell.
    pub fn tensor_data_vec(&self, name: &str) -> Option<(&HfqTensorInfo, Vec<u8>)> {
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
            fadvise_dontneed(fd, info.data_offset, info.data_size);
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

    /// Release page cache for a byte range. Only works if the range is NOT mmap'd.
    #[allow(dead_code)]
    pub fn drop_pages_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            fadvise_dontneed(self._file.as_raw_fd(), offset, len);
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

// ─── Config from HFQ metadata ───────────────────────────────────────────────

pub fn config_from_hfq(hfq: &HfqFile) -> Option<LlamaConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;

    let arch_str = config.get("model_type")?.as_str()?;
    let arch = match arch_str {
        "llama" => ModelArch::Llama,
        "qwen3" | "qwen2" => ModelArch::Qwen3,
        _ => ModelArch::Llama,
    };

    let dim = config.get("hidden_size")?.as_u64()? as usize;
    let n_layers = config.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = config.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let hidden_dim = config.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = config.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let max_seq_len = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rope_freq_base = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let has_qk_norm = hfq
        .find_tensor("model.layers.0.self_attn.q_norm.weight")
        .is_some();

    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);

    let bos_token = config
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let eos_token = config
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    Some(LlamaConfig {
        arch,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps,
        max_seq_len,
        rope_freq_base,
        bos_token,
        eos_token,
        has_qk_norm,
    })
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
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    st_name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));

    let mut wt = match info.quant_type {
        0 => {
            // Q4F16G64
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q4F16G64,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            // Q8F16 — same block format as GGML Q8_0 (34 bytes per 32 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        4 => {
            // Q4_K — GGML-compatible Q4_K blocks (144 bytes per 256 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q4K,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        5 => {
            // Q8HFQ — split-metadata layout (scales then values, 128B-aligned rows)
            let n_groups = k / 32;
            let raw_row = n_groups * 2 + k;
            let row_stride = (raw_row + 127) & !127;
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8HFQ,
                m,
                k,
                row_stride,
                paro: None,
                awq_scale: None,
            })
        }
        6 => {
            // HFQ4-G256 — flat 4-bit, 136 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            // HFQ4-G128 — flat 4-bit, 72 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        28 => {
            // PARO4-G128 — ParoQuant rotated-activation W4 probe format
            assert!(
                k % 128 == 0,
                "PARO4G128 weight {st_name} has K={k} but kernel requires K%128==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::PARO4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        29 => {
            // PARO4-G128T — same metadata, qweight retiled as [M/8, K] for GEMV
            assert!(
                k % 128 == 0,
                "PARO4G128T weight {st_name} has K={k} but kernel requires K%128==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::PARO4G128T,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            // HFQ6-G256 — 6-bit, 200 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        9 => {
            // HFQ2-G256 — flat 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        10 => {
            // HFQ2-G128 — flat 2-bit, 40 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ2G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            // HFQ3-G256 — flat 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            // HFQ3-G128 — flat 3-bit, 56 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256 — MagnumQuant FWHT-rotated 4-bit
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            // MQ8-G256 — MagnumQuant FWHT-rotated symmetric INT8, dp4a
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256 — MagnumQuant FWHT-rotated 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            // MQ2-G256 — MagnumQuant FWHT-rotated 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook, 112 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale.
            // Per-row hdr 16 B + (K/32) blocks × 17 B. See docs/quant-formats/hfp4.md.
            // K%256 — kernel constraint (gemv_hfp4g32 in dispatch.rs);
            // refuse here so a stale or externally-quantized file fails at
            // load instead of panicking on first dispatch.
            assert!(
                k % 256 == 0,
                "HFP4G32 v1 weight {st_name} has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement).
            // Same byte layout as qtype 21; format_flags=0x05 in row hdr.
            // See docs/quant-formats/hfp4.md.
            assert!(
                k % 256 == 0,
                "MFP4G32 weight {st_name} has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            // MQ4-G256-Lloyd — 4-bit + 16-entry fp16 codebook, 160 bytes per 256 elements.
            // Renumbered from qtype 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
            // Models quantized pre-renumber MUST be re-quantized.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => {
            // F16 — dequant to F32 for F32 GEMV
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
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
        _ => panic!(
            "unsupported quant_type {} for weight {st_name}",
            info.quant_type
        ),
    }?;
    // Centralized AWQ sidecar attachment. Replaces the prior per-arm
    // inline `load_awq_scale()` calls at the qt=13 / qt=17 arms — those
    // were the only loaders touching `awq_scale` and missing arms (qt=15
    // MQ6, qt=18 MQ2, qt=19/20 Lloyd, qt=24 MFP4) would silently drop
    // sidecars if added later. Routed through `DType::supports_awq_sidecar`
    // so future widening is a single helper edit, not a scattered
    // per-loader hunt. See dispatch.rs for the allow-list rationale.
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, st_name, k);
    }
    Ok(wt)
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

    eprintln!("  loading token_embd...");
    let embd_info = hfq
        .tensor_data("model.embed_tokens.weight")
        .expect("embed_tokens not found");
    let (token_embd, embd_fmt) = if embd_info.0.quant_type == 4 {
        // Q4_K: upload raw, use Q4K embedding lookup at inference
        eprintln!("    (Q4K raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::Q4K,
        )
    } else if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G256,
        )
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G128,
        )
    } else if embd_info.0.quant_type == 3 {
        // Q8F16: upload raw, use Q8 embedding lookup at inference
        eprintln!("    (Q8 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::Q8_0,
        )
    } else {
        (
            load_f16_tensor(
                hfq,
                gpu,
                "model.embed_tokens.weight",
                &[config.vocab_size, config.dim],
            )?,
            EmbeddingFormat::F32,
        )
    };

    eprintln!("  loading output_norm...");
    let output_norm = load_f16_tensor(hfq, gpu, "model.norm.weight", &[config.dim])?;

    eprintln!("  loading output...");
    let output = if hfq.find_tensor("lm_head.weight").is_some() {
        load_weight_tensor(hfq, gpu, "lm_head.weight", config.vocab_size, config.dim)?
    } else {
        // Tied embeddings — reuse token_embd as output weights (F32 for GEMV)
        let data = hfq.tensor_data("model.embed_tokens.weight").unwrap().1;
        let f32_data: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ...", config.n_layers);
        let p = format!("model.layers.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;
        let q_out_dim = config.n_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: load_f16_tensor(
                hfq,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                &[config.dim],
            )?,
            wq: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_out_dim,
                config.dim,
            )?,
            wk: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                config.dim,
            )?,
            wv: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                config.dim,
            )?,
            wo: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                config.dim,
                q_out_dim,
            )?,
            q_norm: if config.has_qk_norm {
                Some(load_f16_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(load_f16_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                )?)
            } else {
                None
            },
            ffn_norm: load_f16_tensor(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[config.dim],
            )?,
            w_gate: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.gate_proj.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_up: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.up_proj.weight"),
                config.hidden_dim,
                config.dim,
            )?,
            w_down: load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.down_proj.weight"),
                config.dim,
                config.hidden_dim,
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-hfqm-package-test-{}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn writes_and_reads_non_weight_hfqm_package() {
        let payload_a = temp_path("tokens.bin");
        let payload_b = temp_path("values.bin");
        let package_path = temp_path("ref.kldref.hfq");
        std::fs::write(&payload_a, [1u8, 2, 3, 4]).unwrap();
        std::fs::write(&payload_b, [5u8, 6, 7, 8, 9, 10, 11, 12]).unwrap();

        let metadata = serde_json::json!({
            "artifact_kind": "hipfire.kldref",
            "package_schema": "hipfire.kldref.v1",
            "n_ctx": 2,
            "n_chunk": 1,
            "top_k": 1
        })
        .to_string();
        let entries = vec![
            HfqPackageWriteEntry {
                name: "kldref.tokens".to_string(),
                quant_type: 0,
                shape: vec![1, 2],
                group_size: 0,
                source_path: payload_a.clone(),
                data_size: 4,
            },
            HfqPackageWriteEntry {
                name: "kldref.top_log_probs".to_string(),
                quant_type: 0,
                shape: vec![1, 1, 2],
                group_size: 0,
                source_path: payload_b.clone(),
                data_size: 8,
            },
        ];
        write_hfqm_package_from_files(
            &package_path,
            HFQM_ARCH_NON_WEIGHT_PACKAGE,
            &metadata,
            &entries,
        )
        .unwrap();

        let package = HfqPackage::open(&package_path).unwrap();
        assert_eq!(package.version, HFQM_VERSION);
        assert_eq!(package.arch_id, HFQM_ARCH_NON_WEIGHT_PACKAGE);
        assert!(package
            .metadata_json
            .contains("\"artifact_kind\":\"hipfire.kldref\""));
        assert_eq!(package.entries().len(), 2);
        assert_eq!(package.entry("kldref.tokens").unwrap().shape, vec![1, 2]);
        assert_eq!(package.blob_data("kldref.tokens").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(
            package.blob_data("kldref.top_log_probs").unwrap(),
            &[5, 6, 7, 8, 9, 10, 11, 12]
        );

        let _ = std::fs::remove_file(payload_a);
        let _ = std::fs::remove_file(payload_b);
        let _ = std::fs::remove_file(package_path);
    }
}

// ─── ParoQuant safetensors loading (LLaMA / Qwen3 arch) ────────────────────

/// Parse a LlamaConfig from a SafetensorsSource's metadata JSON.
/// The metadata JSON has structure: `{ "config": { ...config.json... } }`.
pub fn config_from_safetensors_llama(
    source: &dyn crate::model_source::ModelSource,
) -> Option<LlamaConfig> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json()).ok()?;
    let config = meta.get("config")?;

    let arch_str = config.get("model_type")?.as_str()?;
    let arch = match arch_str {
        "llama" | "mistral" => ModelArch::Llama,
        "qwen3" | "qwen2" => ModelArch::Qwen3,
        _ => ModelArch::Llama,
    };

    let dim = config.get("hidden_size")?.as_u64()? as usize;
    let n_layers = config.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = config.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let hidden_dim = config.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = config.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let max_seq_len = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rope_freq_base = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);

    let bos_token = config
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let eos_token = config
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    // Detect QK norm from tensor names
    let has_qk_norm = source
        .tensor_info("model.layers.0.self_attn.q_norm.weight")
        .is_some();

    Some(LlamaConfig {
        arch,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps,
        max_seq_len,
        rope_freq_base,
        bos_token,
        eos_token,
        has_qk_norm,
    })
}

/// Repack AWQ-format INT4 weights into HFQ4G128 layout (ParoQuant uses AWQ packing).
///
/// SYNC: must match `repack_awq_to_hfq4g128` in
/// `crates/hipfire-arch-qwen35/src/qwen35.rs`. Duplicated to avoid a
/// cross-crate dependency cycle (the qwen35 crate already depends on this
/// one); keep the two bodies byte-identical when editing.
fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_row = groups_per_row * 72;
    let mut out = vec![0u8; out_dim * bytes_per_row];

    debug_assert_eq!(
        qweight.as_ptr() as usize % 4,
        0,
        "AWQ qweight not 4-byte aligned"
    );
    let qw: &[u32] =
        unsafe { std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4) };
    let qw_cols = out_dim / 8;

    debug_assert_eq!(
        qzeros.as_ptr() as usize % 4,
        0,
        "AWQ qzeros not 4-byte aligned"
    );
    let qz: &[u32] =
        unsafe { std::slice::from_raw_parts(qzeros.as_ptr() as *const u32, qzeros.len() / 4) };
    let qz_cols = out_dim / 8;

    debug_assert_eq!(
        scales.as_ptr() as usize % 2,
        0,
        "AWQ scales not 2-byte aligned"
    );
    let sc: &[u16] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, scales.len() / 2) };

    for m in 0..out_dim {
        for g in 0..groups_per_row {
            let row_off = m * bytes_per_row + g * 72;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            const AWQ_DEQUANT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..64 {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
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

    let hfq_data = repack_awq_to_hfq4g128(
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
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
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

/// Load a ParoQuant weight (quantized or FP16 fallback) using `model.` tensor prefix.
fn paro_load_llama_wt(
    source: &dyn crate::model_source::ModelSource,
    gpu: &Gpu,
    prefix: &str, // e.g. "layers.0.self_attn.q_proj"
    m: usize,
    k: usize,
    gs: u32,
    kr: u8,
) -> HipResult<WeightTensor> {
    let fp = format!("model.{prefix}");
    if source.tensor_info(&format!("{fp}.qweight")).is_some() {
        load_paroquant_weight_from_source(source, gpu, &fp, m, k, gs, kr)
    } else {
        load_fp16_weight_tensor_from_source(source, gpu, &format!("{fp}.weight"), m, k)
    }
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
    let v: Vec<f32> = if info.dtype == "F16" {
        data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    gpu.upload_f32(&v, shape)
}

/// Load LLaMA/Qwen3 weights from a ParoQuant safetensors model.
///
/// Tensor naming convention: `model.layers.{i}.self_attn.q_proj.{qweight,...}`
/// (no `model.language_model.` prefix — that's Qwen3.5-specific).
pub fn load_weights_paroquant_llama(
    source: &dyn crate::model_source::ModelSource,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    // Embedding
    eprintln!("  loading token_embd (ParoQuant LLaMA/Qwen3)...");
    let embd_name = "model.embed_tokens.weight";
    let (_, embd_data) = source
        .tensor_data(embd_name)
        .ok_or_else(|| HipError::new(0, "PARO tensor not found: embed_tokens not found"))?;
    let f32_embd: Vec<f32> = embd_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let token_embd = gpu.upload_f32(&f32_embd, &[config.vocab_size, config.dim])?;
    let embd_fmt = EmbeddingFormat::F32;

    // Output norm
    eprintln!("  loading output_norm...");
    let output_norm = paro_load_llama_norm_raw(source, gpu, "norm.weight", &[config.dim])?;

    // Output / lm_head (tied or separate)
    let output = if source.tensor_info("lm_head.weight").is_some() {
        eprintln!("  loading output (separate lm_head)...");
        let lm_prefix = "lm_head";
        if source
            .tensor_info(&format!("{lm_prefix}.qweight"))
            .is_some()
        {
            load_paroquant_weight_from_source(
                source,
                gpu,
                lm_prefix,
                config.vocab_size,
                config.dim,
                gs,
                kr,
            )?
        } else {
            load_fp16_weight_tensor_from_source(
                source,
                gpu,
                &format!("{lm_prefix}.weight"),
                config.vocab_size,
                config.dim,
            )?
        }
    } else {
        eprintln!("  loading output (tied embeddings)...");
        let (_, td) = source
            .tensor_data(embd_name)
            .ok_or_else(|| HipError::new(0, "PARO tensor not found: embed_tokens for lm_head"))?;
        let f: Vec<f32> = td
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    };

    // Layers
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!(
            "  loading layer {i}/{} (ParoQuant LLaMA/Qwen3)...",
            config.n_layers
        );
        let p = format!("layers.{i}");
        let q_out_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let q_norm = if config.has_qk_norm {
            Some(paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.self_attn.q_norm.weight"),
                &[config.head_dim],
            )?)
        } else {
            None
        };
        let k_norm = if config.has_qk_norm {
            Some(paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.self_attn.k_norm.weight"),
                &[config.head_dim],
            )?)
        } else {
            None
        };

        let layer = LayerWeights {
            attn_norm: paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                &[config.dim],
            )?,
            wq: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.q_proj"),
                q_out_dim,
                config.dim,
                gs,
                kr,
            )?,
            wk: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.k_proj"),
                kv_dim,
                config.dim,
                gs,
                kr,
            )?,
            wv: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.v_proj"),
                kv_dim,
                config.dim,
                gs,
                kr,
            )?,
            wo: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.self_attn.o_proj"),
                config.dim,
                q_out_dim,
                gs,
                kr,
            )?,
            q_norm,
            k_norm,
            ffn_norm: paro_load_llama_norm_raw(
                source,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[config.dim],
            )?,
            w_gate: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.gate_proj"),
                config.hidden_dim,
                config.dim,
                gs,
                kr,
            )?,
            w_up: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.up_proj"),
                config.hidden_dim,
                config.dim,
                gs,
                kr,
            )?,
            w_down: paro_load_llama_wt(
                source,
                gpu,
                &format!("{p}.mlp.down_proj"),
                config.dim,
                config.hidden_dim,
                gs,
                kr,
            )?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
    })
}
