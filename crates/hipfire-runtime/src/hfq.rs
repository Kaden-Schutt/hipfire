//! HFQ (.hfq) file loader for hipfire-native Q4_F16 quantized models.

use crate::llama::{
    f16_to_f32, EmbeddingFormat, LayerWeights, LlamaConfig, LlamaWeights, ModelArch, WeightTensor,
};
use hip_bridge::HipResult;
use memmap2::Mmap;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::fs::File;
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
        libc::posix_fadvise(fd, offset as libc::off_t, len as libc::off_t, libc::POSIX_FADV_DONTNEED);
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

        // Parse header (32 bytes)
        let magic = &mmap[0..4];
        assert_eq!(magic, b"HFQM", "Not an HFQ file");
        let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

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
                if b == b'{' { brace_depth += 1; }
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
            mmap: Some(mmap), arch_id, metadata_json, tensors, tensor_map,
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

    /// Look up a tensor's metadata (name, quant_type, shape, byte offset/size)
    /// without copying its data. The weight pager calls this at load time to
    /// register byte ranges without forcing eager VRAM allocation.
    pub fn find_tensor_info(&self, name: &str) -> Option<&HfqTensorInfo> {
        let idx = *self.tensor_map.get(name)?;
        Some(&self.tensors[idx])
    }

    pub fn tensor_data(&self, name: &str) -> Option<(&HfqTensorInfo, &[u8])> {
        let idx = *self.tensor_map.get(name)?;
        let info = &self.tensors[idx];
        debug_assert!(
            self.mmap.is_some(),
            "tensor_data() called after drop_mmap() — use tensor_data_vec() or tensor_data_pread() instead (tensor: {name})"
        );
        let mmap = self.mmap.as_ref()?;
        Some((info, &mmap[info.data_offset..info.data_offset + info.data_size]))
    }

    /// Read tensor data via pread into a reusable buffer, then FADV_DONTNEED
    /// the file range. On unified-memory APUs (Strix Halo etc.), mmap pages
    /// can't be evicted while the mapping exists, so pread + fadvise is the
    /// only way to prevent page cache from starving hipMalloc.
    ///
    /// Returns (info, guard) where guard derefs to `&[u8]`. The buffer is
    /// reused across calls — the previous data is overwritten.
    #[cfg(unix)]
    pub fn tensor_data_pread(&self, name: &str) -> Option<(&HfqTensorInfo, std::cell::Ref<'_, Vec<u8>>)> {
        use std::os::unix::io::AsRawFd;
        let idx = *self.tensor_map.get(name)?;
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
                if n <= 0 { break; }
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
        let idx = *self.tensor_map.get(name)?;
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
                if n <= 0 { break; }
                total_read += n as usize;
            }
            fadvise_dontneed(fd, info.data_offset, info.data_size);
            return Some((info, buf));
        }

        #[cfg(not(unix))]
        {
            let mmap = self.mmap.as_ref()?;
            Some((info, mmap[info.data_offset..info.data_offset + info.data_size].to_vec()))
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
        { let _ = (offset, len); }
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
        if lo < hi { Some((lo, hi)) } else { None }
    }

    fn find_tensor(&self, name: &str) -> Option<&HfqTensorInfo> {
        self.tensor_map.get(name).map(|&i| &self.tensors[i])
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
    let n_kv_heads = config.get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let hidden_dim = config.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = config.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = config.get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let max_seq_len = config.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let rope_freq_base = config.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    let has_qk_norm = hfq.find_tensor("model.layers.0.self_attn.q_norm.weight").is_some();

    let head_dim = config.get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);

    let bos_token = config.get("bos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let eos_token = config.get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    Some(LlamaConfig {
        arch, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
        head_dim, norm_eps, max_seq_len, rope_freq_base,
        bos_token, eos_token,
        has_qk_norm,
    })
}

// ─── Weight Loading ─────────────────────────────────────────────────────────

/// Load a tensor as F32 on GPU (for norms, embeddings).
fn load_f16_tensor(hfq: &HfqFile, gpu: &mut Gpu, st_name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => { // F16
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        2 => { // F32
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        _ => panic!("expected F16/F32 tensor for {st_name}, got quant_type={}", info.quant_type),
    };

    gpu.upload_f32(&f32_data, shape)
}

/// Load a weight tensor (quantized or F16) onto GPU.
fn load_weight_tensor(hfq: &HfqFile, gpu: &Gpu, st_name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
    let (info, data) = hfq.tensor_data(st_name)
        .unwrap_or_else(|| panic!("tensor not found: {st_name}"));

    match info.quant_type {
        0 => { // Q4F16G64
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q4F16G64, m, k, row_stride: 0 })
        }
        3 => { // Q8F16 — same block format as GGML Q8_0 (34 bytes per 32 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0 })
        }
        4 => { // Q4_K — GGML-compatible Q4_K blocks (144 bytes per 256 elements)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q4K, m, k, row_stride: 0 })
        }
        5 => { // Q8HFQ — split-metadata layout (scales then values, 128B-aligned rows)
            let n_groups = k / 32;
            let raw_row = n_groups * 2 + k;
            let row_stride = (raw_row + 127) & !127;
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8HFQ, m, k, row_stride })
        }
        6 => { // HFQ4-G256 — flat 4-bit, 136 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0 })
        }
        7 => { // HFQ4-G128 — flat 4-bit, 72 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0 })
        }
        8 => { // HFQ6-G256 — 6-bit, 200 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ6G256, m, k, row_stride: 0 })
        }
        9 => { // HFQ2-G256 — flat 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ2G256, m, k, row_stride: 0 })
        }
        10 => { // HFQ2-G128 — flat 2-bit, 40 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ2G128, m, k, row_stride: 0 })
        }
        11 => { // HFQ3-G256 — flat 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G256, m, k, row_stride: 0 })
        }
        12 => { // HFQ3-G128 — flat 3-bit, 56 bytes per 128 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ3G128, m, k, row_stride: 0 })
        }
        13 => { // MQ4-G256 — MagnumQuant FWHT-rotated 4-bit
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        14 => { // MQ8-G256 — MagnumQuant FWHT-rotated symmetric INT8, dp4a
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ8G256, m, k, row_stride: 0 })
        }
        17 => { // MQ3-G256 — MagnumQuant FWHT-rotated 3-bit, 104 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ3G256, m, k, row_stride: 0 })
        }
        18 => { // MQ2-G256 — MagnumQuant FWHT-rotated 2-bit, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ2G256, m, k, row_stride: 0 })
        }
        19 => { // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook, 72 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ2G256Lloyd, m, k, row_stride: 0 })
        }
        20 => { // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook, 112 bytes per 256 elements
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ3G256Lloyd, m, k, row_stride: 0 })
        }
        21 => { // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale.
                // Per-row hdr 16 B + (K/32) blocks × 17 B. See docs/quant-formats/hfp4.md.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFP4G32, m, k, row_stride: 0 })
        }
        1 => { // F16 — dequant to F32 for F32 GEMV
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        _ => panic!("unsupported quant_type {} for weight {st_name}", info.quant_type),
    }
}

/// Load LLaMA weights from an HFQ file onto GPU.
pub fn load_weights_hfq(
    hfq: &HfqFile,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    eprintln!("  loading token_embd...");
    let embd_info = hfq.tensor_data("model.embed_tokens.weight")
        .expect("embed_tokens not found");
    let (token_embd, embd_fmt) = if embd_info.0.quant_type == 4 {
        // Q4_K: upload raw, use Q4K embedding lookup at inference
        eprintln!("    (Q4K raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::Q4K)
    } else if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G256)
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G128)
    } else if embd_info.0.quant_type == 3 {
        // Q8F16: upload raw, use Q8 embedding lookup at inference
        eprintln!("    (Q8 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::Q8_0)
    } else {
        (load_f16_tensor(hfq, gpu, "model.embed_tokens.weight",
            &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
    };

    eprintln!("  loading output_norm...");
    let output_norm = load_f16_tensor(hfq, gpu, "model.norm.weight", &[config.dim])?;

    eprintln!("  loading output...");
    let output = if hfq.find_tensor("lm_head.weight").is_some() {
        load_weight_tensor(hfq, gpu, "lm_head.weight", config.vocab_size, config.dim)?
    } else {
        // Tied embeddings — reuse token_embd as output weights (F32 for GEMV)
        let data = hfq.tensor_data("model.embed_tokens.weight").unwrap().1;
        let f32_data: Vec<f32> = data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim, row_stride: 0 }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ...", config.n_layers);
        let p = format!("model.layers.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;
        let q_out_dim = config.n_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: load_f16_tensor(hfq, gpu,
                &format!("{p}.input_layernorm.weight"), &[config.dim])?,
            wq: load_weight_tensor(hfq, gpu,
                &format!("{p}.self_attn.q_proj.weight"), q_out_dim, config.dim)?,
            wk: load_weight_tensor(hfq, gpu,
                &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
            wv: load_weight_tensor(hfq, gpu,
                &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
            wo: load_weight_tensor(hfq, gpu,
                &format!("{p}.self_attn.o_proj.weight"), config.dim, q_out_dim)?,
            q_norm: if config.has_qk_norm {
                Some(load_f16_tensor(hfq, gpu,
                    &format!("{p}.self_attn.q_norm.weight"), &[config.head_dim])?)
            } else { None },
            k_norm: if config.has_qk_norm {
                Some(load_f16_tensor(hfq, gpu,
                    &format!("{p}.self_attn.k_norm.weight"), &[config.head_dim])?)
            } else { None },
            ffn_norm: load_f16_tensor(hfq, gpu,
                &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
            w_gate: load_weight_tensor(hfq, gpu,
                &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
            w_up: load_weight_tensor(hfq, gpu,
                &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
            w_down: load_weight_tensor(hfq, gpu,
                &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights { token_embd, embd_format: embd_fmt, output_norm, output, layers })
}
