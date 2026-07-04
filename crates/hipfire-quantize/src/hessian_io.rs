//! HFQM calibration-package Hessian reader.
//!
//! Reads per-tensor Hessians out of a unified `.calib.hfq` (HFQM) package
//! produced by the native single-load collector
//! (`hipfire_arch_qwen35::qwen35::collect_calibration_artifacts` →
//! `write_calib_artifacts`, or the `hipfire collect-artifacts` CLI / daemon
//! `Collect` op). Each dense projection is stored as a `<name>.hessian`
//! `[K,K]` tensor alongside its `<name>.imatrix`. Legacy packages store full
//! row-major F32; compact packages store exact F32 diagonal plus BF16 lower
//! strict triangle. MoE routed experts are imatrix-only (their full Hessians
//! don't fit), so they carry no `.hessian` entry and `get` returns `None` for
//! them (the quantizer then skips LDLQ for that tensor — exactly the prior
//! behavior).
//!
//! This replaced the standalone HFHS `.hessian.bin` sidecar format: the engine
//! emits one container, the quantizer reads it directly, and there is no second
//! Hessian format to keep in sync.
//!
//! Design choices (unchanged from the sidecar era):
//! - **mmap-based.** A 9B Hessian package is multi-GB; mmap with sequential
//!   advice lets the kernel page tensors in as the per-tensor Cholesky walk
//!   progresses, then evict.
//! - **Zero-copy.** `HessianRef` borrows from the mmap; the caller promotes
//!   FP32 → FP64 only at Cholesky time.
//! - **Index built at open.** A `HashMap<name, entry>` over the `.hessian`
//!   tensors gives O(1) lookup for the quantizer's per-tensor query, and a
//!   parallel vector index exposes `.imatrix` entries for AWQ.
//!
//! Consumer integration: the quantizer (`HIPFIRE_QTIP_HESSIAN` →
//! `HessianSidecar::open`) queries `get(tensor_name_without_dot_weight_suffix,
//! 0)` per LDLQ-target tensor. AWQ/import code also iterates `imatrices()` so
//! imatrix-only routed experts still feed activation-aware quantization even
//! when LDLQ is unavailable.

use byteorder::{ByteOrder, LittleEndian};
use hipfire_primitives::conv::bf16_bits_to_f32 as bf16_to_f32;
use memmap2::{Advice, Mmap};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const HFQM_MAGIC: &[u8; 4] = b"HFQM";
const HFQM_VERSION_SUPPORTED: u32 = 1;
const HEADER_SIZE: usize = 32;
/// HFQM `quant_type` byte for dense F32 tensors.
const QUANT_TYPE_F32: u8 = 2;
/// Calibration-only HFQM `quant_type` for compact Hessians:
/// exact F32 diagonal followed by BF16 lower strict triangle.
const QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32: u8 = 130;
/// Suffix the collector appends to a tensor's canonical name for its Hessian.
const HESSIAN_SUFFIX: &str = ".hessian";
/// Suffix the collector appends to a tensor's canonical name for its imatrix.
const IMATRIX_SUFFIX: &str = ".imatrix";

#[derive(Debug)]
pub enum HessianError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    TruncatedFile {
        needed: usize,
        have: usize,
    },
    InvalidData(String),
    NegativeDiagonal {
        tensor: String,
        index: usize,
        value: f32,
    },
}

impl std::fmt::Display for HessianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HessianError::Io(e) => write!(f, "I/O error: {e}"),
            HessianError::InvalidMagic(m) => {
                write!(f, "invalid HFQM magic: got {m:?}, expected {HFQM_MAGIC:?}")
            }
            HessianError::UnsupportedVersion(v) => {
                write!(f, "unsupported HFQM version {v}, this build understands v{HFQM_VERSION_SUPPORTED}")
            }
            HessianError::TruncatedFile { needed, have } => {
                write!(f, "HFQM truncated: needed {needed} bytes, file is {have}")
            }
            HessianError::InvalidData(m) => write!(f, "invalid HFQM package: {m}"),
            HessianError::NegativeDiagonal {
                tensor,
                index,
                value,
            } => write!(
                f,
                "Hessian for tensor {tensor:?} has negative diagonal H[{index},{index}] = {value} \
                 (should be ≥0 by PSD construction; likely FP corruption — fall back to plain MQ4)"
            ),
        }
    }
}

impl std::error::Error for HessianError {}

impl From<std::io::Error> for HessianError {
    fn from(e: std::io::Error) -> Self {
        HessianError::Io(e)
    }
}

/// FP precision of a stored Hessian. Determines stride per element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianDtype {
    F32,
    F64,
    Bf16TrilDiagF32,
}

impl HessianDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            HessianDtype::F32 => 4,
            HessianDtype::F64 => 8,
            HessianDtype::Bf16TrilDiagF32 => 0,
        }
    }
}

fn compact_hessian_bytes(k: usize) -> usize {
    k * 4 + k * (k - 1)
}

fn lower_strict_index(i: usize, j: usize) -> usize {
    debug_assert!(i > j);
    i * (i - 1) / 2 + j
}

/// Zero-copy view into one Hessian record in the mmap.
pub struct HessianRef<'a> {
    pub name: &'a str,
    #[allow(dead_code)]
    pub expert_idx: u32,
    pub k: usize,
    pub dtype: HessianDtype,
    /// Row-major Hessian payload, `K * K * size_bytes()` bytes.
    pub bytes: &'a [u8],
}

impl<'a> HessianRef<'a> {
    /// Iterate the Hessian as `f64` values, promoting from FP32 if needed.
    /// The quantizer's Cholesky path uses this — never reads FP32 directly.
    pub fn iter_f64(&self) -> impl Iterator<Item = f64> + '_ {
        let n = self.k * self.k;
        (0..n).map(move |idx| self.at(idx / self.k, idx % self.k))
    }

    /// Read the `[i, j]` entry as f64. O(1).
    pub fn at(&self, i: usize, j: usize) -> f64 {
        debug_assert!(
            i < self.k && j < self.k,
            "out of bounds: H[{i},{j}] K={}",
            self.k
        );
        let off = (i * self.k + j) * self.dtype.size_bytes();
        match self.dtype {
            HessianDtype::F32 => LittleEndian::read_f32(&self.bytes[off..off + 4]) as f64,
            HessianDtype::F64 => LittleEndian::read_f64(&self.bytes[off..off + 8]),
            HessianDtype::Bf16TrilDiagF32 => {
                if i == j {
                    let off = i * 4;
                    LittleEndian::read_f32(&self.bytes[off..off + 4]) as f64
                } else {
                    let (r, c) = if i > j { (i, j) } else { (j, i) };
                    let off = self.k * 4 + lower_strict_index(r, c) * 2;
                    bf16_to_f32(LittleEndian::read_u16(&self.bytes[off..off + 2])) as f64
                }
            }
        }
    }
}

/// Zero-copy view into one imatrix record in the mmap.
pub struct ImatrixRef<'a> {
    pub name: &'a str,
    pub k: usize,
    /// F32 vector payload, `K * 4` bytes.
    pub bytes: &'a [u8],
}

impl<'a> ImatrixRef<'a> {
    pub fn iter_f32(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.k).map(move |idx| {
            let off = idx * 4;
            LittleEndian::read_f32(&self.bytes[off..off + 4])
        })
    }
}

/// Per-Hessian record (computed at open, points into the mmap). `name` is the
/// canonical tensor name with the `.hessian` suffix stripped — the key the
/// quantizer queries.
struct TensorEntry {
    name: String,
    k: usize,
    dtype: HessianDtype,
    payload_offset: usize,
    payload_bytes: usize,
}

/// Per-imatrix record (computed at open, points into the mmap). `name` is the
/// canonical tensor name with the `.imatrix` suffix stripped.
struct ImatrixEntry {
    name: String,
    k: usize,
    payload_offset: usize,
    payload_bytes: usize,
}

pub struct HessianSidecar {
    // Mmap kept alive for the sidecar's lifetime; all `HessianRef` views
    // borrow from this. `_file` keeps the fd alive on Unix.
    mmap: Mmap,
    _file: File,
    index: HashMap<String, TensorEntry>,
    imatrix_index: HashMap<String, ImatrixEntry>,
}

impl std::fmt::Debug for HessianSidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HessianSidecar")
            .field("mmap_len", &self.mmap.len())
            .field("n_tensors", &self.index.len())
            .field("n_imatrix_tensors", &self.imatrix_index.len())
            .finish()
    }
}

/// Find the byte index just past the first complete top-level JSON object in
/// `bytes` (the HFQM metadata blob is immediately followed by the tensor
/// index). Mirrors the engine-side `hfq::json_blob_end`.
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

impl HessianSidecar {
    pub fn open(path: &Path) -> Result<Self, HessianError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Hint sequential access: the quantizer walks tensor-by-tensor.
        #[cfg(unix)]
        {
            mmap.advise(Advice::Sequential).ok();
        }
        let _ = Advice::Sequential; // silence unused on non-unix

        if mmap.len() < HEADER_SIZE {
            return Err(HessianError::TruncatedFile {
                needed: HEADER_SIZE,
                have: mmap.len(),
            });
        }
        let magic: [u8; 4] = mmap[0..4].try_into().unwrap();
        if &magic != HFQM_MAGIC {
            return Err(HessianError::InvalidMagic(magic));
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != HFQM_VERSION_SUPPORTED {
            return Err(HessianError::UnsupportedVersion(version));
        }
        // mmap[8..12] = arch_id (unused for a calibration package)
        let n_entries = LittleEndian::read_u32(&mmap[12..16]) as usize;
        let metadata_offset = LittleEndian::read_u64(&mmap[16..24]) as usize;
        let data_offset = LittleEndian::read_u64(&mmap[24..32]) as usize;
        if metadata_offset > data_offset || data_offset > mmap.len() {
            return Err(HessianError::InvalidData(format!(
                "offsets metadata={metadata_offset} data={data_offset} len={}",
                mmap.len()
            )));
        }

        // Metadata JSON is self-delimited; the tensor index follows it.
        let meta_bytes = &mmap[metadata_offset..data_offset];
        let json_end = json_blob_end(meta_bytes)
            .ok_or_else(|| HessianError::InvalidData("metadata JSON did not end".into()))?;
        let mut pos = metadata_offset + json_end;
        if pos + 4 > data_offset {
            return Err(HessianError::InvalidData(
                "index missing tensor count".into(),
            ));
        }
        let idx_n = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
        if idx_n != n_entries {
            return Err(HessianError::InvalidData(format!(
                "index count {idx_n} != header count {n_entries}"
            )));
        }
        pos += 4;

        // Walk the index. Payloads are laid out contiguously from `data_offset`
        // in index order; retain `.hessian` tensors for LDLQ and `.imatrix`
        // vectors for AWQ / activation-aware quantization.
        let mut index = HashMap::new();
        let mut imatrix_index = HashMap::new();
        let mut cumulative_offset = data_offset;
        for _ in 0..n_entries {
            if pos + 2 > data_offset {
                return Err(HessianError::InvalidData(
                    "index truncated at name length".into(),
                ));
            }
            let name_len = LittleEndian::read_u16(&mmap[pos..pos + 2]) as usize;
            pos += 2;
            if pos + name_len + 2 > data_offset {
                return Err(HessianError::InvalidData(
                    "index truncated at name/header".into(),
                ));
            }
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
            pos += name_len;
            let quant_type = mmap[pos];
            pos += 1;
            let n_dims = mmap[pos] as usize;
            pos += 1;
            if pos + n_dims * 4 + 12 > data_offset {
                return Err(HessianError::InvalidData(
                    "index truncated at shape/data_size".into(),
                ));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize);
                pos += 4;
            }
            // group_size (u32) then data_len (u64)
            pos += 4;
            let data_size = LittleEndian::read_u64(&mmap[pos..pos + 8]) as usize;
            pos += 8;
            let payload_offset = cumulative_offset;
            cumulative_offset += data_size;
            if cumulative_offset > mmap.len() {
                return Err(HessianError::TruncatedFile {
                    needed: cumulative_offset,
                    have: mmap.len(),
                });
            }

            if let Some(base) = name.strip_suffix(HESSIAN_SUFFIX) {
                // Retain dense `.hessian` tensors in either legacy F32 or compact
                // BF16-triangle storage. Both expose the same logical [K,K] API.
                if shape.len() != 2 || shape[0] != shape[1] {
                    continue;
                }
                let k = shape[0];
                let dtype = match quant_type {
                    QUANT_TYPE_F32 => {
                        if k * k * 4 != data_size {
                            return Err(HessianError::InvalidData(format!(
                                "{name}: dense F32 K={k} implies {} bytes but data_size={data_size}",
                                k * k * 4
                            )));
                        }
                        HessianDtype::F32
                    }
                    QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32 => {
                        let expected = compact_hessian_bytes(k);
                        if expected != data_size {
                            return Err(HessianError::InvalidData(format!(
                                "{name}: compact BF16-tril K={k} implies {expected} bytes but data_size={data_size}"
                            )));
                        }
                        HessianDtype::Bf16TrilDiagF32
                    }
                    _ => continue,
                };
                index.insert(
                    base.to_string(),
                    TensorEntry {
                        name: base.to_string(),
                        k,
                        dtype,
                        payload_offset,
                        payload_bytes: data_size,
                    },
                );
                continue;
            }

            if let Some(base) = name.strip_suffix(IMATRIX_SUFFIX) {
                if quant_type != QUANT_TYPE_F32 || shape.len() != 1 {
                    continue;
                }
                let k = shape[0];
                if k * 4 != data_size {
                    return Err(HessianError::InvalidData(format!(
                        "{name}: F32 imatrix K={k} implies {} bytes but data_size={data_size}",
                        k * 4
                    )));
                }
                imatrix_index.insert(
                    base.to_string(),
                    ImatrixEntry {
                        name: base.to_string(),
                        k,
                        payload_offset,
                        payload_bytes: data_size,
                    },
                );
            }
        }

        Ok(Self {
            mmap,
            _file: file,
            index,
            imatrix_index,
        })
    }

    /// Look up a Hessian by tensor name (the `.hfq` weight name with the
    /// trailing `.weight` stripped). `expert_idx` is retained for signature
    /// compatibility but ignored: MoE experts are encoded in the tensor name
    /// itself (`...experts.{x}...`) and are imatrix-only, so they have no
    /// `.hessian` entry and resolve to `None`. Returns `None` when the tensor
    /// has no Hessian — the quantizer treats that as "skip LDLQ for this
    /// tensor".
    pub fn get(&self, name: &str, _expert_idx: u32) -> Option<HessianRef<'_>> {
        let entry = self.index.get(name)?;
        Some(HessianRef {
            name: &entry.name,
            expert_idx: 0,
            k: entry.k,
            dtype: entry.dtype,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    /// Iterate all stored Hessians. Used for bulk validation passes (e.g.
    /// symmetry / PSD check at start of quantize) and debug dumps.
    #[allow(dead_code)]
    pub fn tensors(&self) -> impl Iterator<Item = HessianRef<'_>> + '_ {
        self.index.values().map(|entry| HessianRef {
            name: &entry.name,
            expert_idx: 0,
            k: entry.k,
            dtype: entry.dtype,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    /// Look up an imatrix by canonical tensor name (without `.imatrix` suffix).
    pub fn imatrix(&self, name: &str) -> Option<ImatrixRef<'_>> {
        let entry = self.imatrix_index.get(name)?;
        Some(ImatrixRef {
            name: &entry.name,
            k: entry.k,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    /// Iterate all stored imatrix vectors, including routed experts that do not
    /// have a full Hessian.
    pub fn imatrices(&self) -> impl Iterator<Item = ImatrixRef<'_>> + '_ {
        self.imatrix_index.values().map(|entry| ImatrixRef {
            name: &entry.name,
            k: entry.k,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.index.len()
    }

    pub fn n_imatrix_tensors(&self) -> usize {
        self.imatrix_index.len()
    }

    /// Cheap symmetry sanity check on a per-tensor basis. Samples 32 random
    /// off-diagonal pairs; verifies `|H[i,j] - H[j,i]| / max(|H[i,i]|, |H[j,j]|) < tol`.
    /// Returns `Ok(())` if OK, `Err` describing the first violating pair.
    ///
    /// Use a fixed RNG seed for determinism — debugging a regressed model
    /// shouldn't change the validation outcome between runs.
    pub fn check_symmetry(href: &HessianRef<'_>, tol: f64) -> Result<(), String> {
        use std::num::Wrapping;
        let k = href.k;
        if k < 2 {
            return Ok(());
        }
        let mut rng = Wrapping(0xdeadbeefu64);
        let mut next = || {
            rng = rng * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
            (rng.0 >> 32) as usize
        };
        for _ in 0..32 {
            let i = next() % k;
            let j = next() % k;
            if i == j {
                continue;
            }
            let a = href.at(i, j);
            let b = href.at(j, i);
            let diag = href.at(i, i).abs().max(href.at(j, j).abs()).max(1e-30);
            if ((a - b).abs() / diag) > tol {
                return Err(format!(
                    "{}: asymmetric at H[{i},{j}]={a:.6e} vs H[{j},{i}]={b:.6e} (diag={diag:.6e})",
                    href.name
                ));
            }
        }
        Ok(())
    }

    /// PSD diagnostic: scan all diagonals for negativity. PSD-by-construction
    /// guarantees `H[i,i] >= 0`; FP corruption (e.g. from a partial download)
    /// can produce negatives. Returns the first negative diagonal, if any.
    pub fn check_positive_diagonal(href: &HessianRef<'_>) -> Result<(), HessianError> {
        for i in 0..href.k {
            let v = href.at(i, i);
            if v < 0.0 {
                return Err(HessianError::NegativeDiagonal {
                    tensor: href.name.to_string(),
                    index: i,
                    value: v as f32,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal HFQM `.calib.hfq` with one `.hessian` tensor (`tA`, K=2)
    /// plus its sibling `.imatrix` vector. Mirrors
    /// `hipfire_runtime::hfq::write_hfqm_package_mem`.
    fn make_test_package() -> NamedTempFile {
        struct Entry {
            name: &'static str,
            quant_type: u8,
            shape: Vec<u32>,
            data: Vec<u8>,
        }
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let entries = vec![
            Entry {
                name: "tA.hessian",
                quant_type: 2,
                shape: vec![2, 2],
                data: f32_bytes(&[1.0, 0.5, 0.5, 2.0]),
            },
            Entry {
                name: "tA.imatrix",
                quant_type: 2,
                shape: vec![2],
                data: f32_bytes(&[1.0, 2.0]),
            },
        ];

        let metadata = b"{\"artifact_kind\":\"calibration\"}";
        let metadata_offset = 32u64;
        let index_offset = metadata_offset + metadata.len() as u64;
        let mut index = Vec::new();
        index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in &entries {
            index.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
            index.extend_from_slice(e.name.as_bytes());
            index.push(e.quant_type);
            index.push(e.shape.len() as u8);
            for &d in &e.shape {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes()); // group_size
            index.extend_from_slice(&(e.data.len() as u64).to_le_bytes());
        }
        let data_start = index_offset + index.len() as u64;
        let data_offset = (data_start + 4095) & !4095;

        let mut tf = NamedTempFile::new().unwrap();
        let f = tf.as_file_mut();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u32.to_le_bytes()).unwrap(); // arch_id
        f.write_all(&(entries.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(&index).unwrap();
        f.write_all(&vec![0u8; (data_offset - data_start) as usize])
            .unwrap();
        for e in &entries {
            f.write_all(&e.data).unwrap();
        }
        tf.flush().unwrap();
        tf
    }

    fn make_compact_test_package() -> NamedTempFile {
        use hipfire_primitives::conv::f32_to_bf16_bits;

        struct Entry {
            name: &'static str,
            quant_type: u8,
            shape: Vec<u32>,
            data: Vec<u8>,
        }
        let mut compact = Vec::new();
        for v in [1.0f32, 2.0, 4.0] {
            compact.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.5f32, -0.25, 0.75] {
            compact.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        let entries = vec![
            Entry {
                name: "tB.hessian",
                quant_type: QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32,
                shape: vec![3, 3],
                data: compact,
            },
            Entry {
                name: "tB.imatrix",
                quant_type: 2,
                shape: vec![3],
                data: [1.0f32, 2.0, 4.0]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            },
        ];

        let metadata = b"{\"artifact_kind\":\"calibration\"}";
        let metadata_offset = 32u64;
        let index_offset = metadata_offset + metadata.len() as u64;
        let mut index = Vec::new();
        index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in &entries {
            index.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
            index.extend_from_slice(e.name.as_bytes());
            index.push(e.quant_type);
            index.push(e.shape.len() as u8);
            for &d in &e.shape {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes());
            index.extend_from_slice(&(e.data.len() as u64).to_le_bytes());
        }
        let data_start = index_offset + index.len() as u64;
        let data_offset = (data_start + 4095) & !4095;

        let mut tf = NamedTempFile::new().unwrap();
        let f = tf.as_file_mut();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(entries.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(&index).unwrap();
        f.write_all(&vec![0u8; (data_offset - data_start) as usize])
            .unwrap();
        for e in &entries {
            f.write_all(&e.data).unwrap();
        }
        tf.flush().unwrap();
        tf
    }

    #[test]
    fn open_and_lookup_roundtrip() {
        let tf = make_test_package();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        assert_eq!(sc.n_tensors(), 1);
        assert_eq!(sc.n_imatrix_tensors(), 1);

        let ta = sc.get("tA", 0).expect("tA missing");
        assert_eq!(ta.k, 2);
        assert_eq!(ta.dtype, HessianDtype::F32);
        assert_eq!(ta.at(0, 0), 1.0);
        assert_eq!(ta.at(0, 1), 0.5);
        assert_eq!(ta.at(1, 0), 0.5);
        assert_eq!(ta.at(1, 1), 2.0);

        // The query name is the canonical name WITHOUT the `.hessian` suffix.
        assert!(sc.get("tA.hessian", 0).is_none());
        assert!(sc.get("not_there", 0).is_none());

        let ia = sc.imatrix("tA").expect("tA imatrix missing");
        assert_eq!(ia.k, 2);
        assert_eq!(ia.iter_f32().collect::<Vec<_>>(), vec![1.0, 2.0]);
        assert!(sc.imatrix("tA.imatrix").is_none());
    }

    #[test]
    fn open_and_lookup_compact_bf16_tril_roundtrip() {
        let tf = make_compact_test_package();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        assert_eq!(sc.n_tensors(), 1);
        assert_eq!(sc.n_imatrix_tensors(), 1);

        let tb = sc.get("tB", 0).expect("tB missing");
        assert_eq!(tb.k, 3);
        assert_eq!(tb.dtype, HessianDtype::Bf16TrilDiagF32);
        assert_eq!(tb.at(0, 0), 1.0);
        assert_eq!(tb.at(1, 1), 2.0);
        assert_eq!(tb.at(2, 2), 4.0);
        assert_eq!(tb.at(1, 0), 0.5);
        assert_eq!(tb.at(0, 1), 0.5);
        assert_eq!(tb.at(2, 0), -0.25);
        assert_eq!(tb.at(0, 2), -0.25);
        assert_eq!(tb.at(2, 1), 0.75);
        assert_eq!(tb.at(1, 2), 0.75);
        assert_eq!(
            tb.iter_f64().collect::<Vec<_>>(),
            vec![1.0, 0.5, -0.25, 0.5, 2.0, 0.75, -0.25, 0.75, 4.0]
        );
        HessianSidecar::check_symmetry(&tb, 0.0).unwrap();
        HessianSidecar::check_positive_diagonal(&tb).unwrap();

        let ib = sc.imatrix("tB").expect("tB imatrix missing");
        assert_eq!(ib.k, 3);
        assert_eq!(ib.iter_f32().collect::<Vec<_>>(), vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"XXXX").unwrap();
        tf.write_all(&[0u8; 28]).unwrap();
        tf.flush().unwrap();
        match HessianSidecar::open(tf.path()) {
            Err(HessianError::InvalidMagic(m)) => assert_eq!(&m, b"XXXX"),
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn rejects_future_version() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"HFQM").unwrap();
        tf.write_all(&99u32.to_le_bytes()).unwrap();
        tf.write_all(&[0u8; 24]).unwrap();
        tf.flush().unwrap();
        match HessianSidecar::open(tf.path()) {
            Err(HessianError::UnsupportedVersion(v)) => assert_eq!(v, 99),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"HFQM").unwrap();
        tf.flush().unwrap();
        assert!(matches!(
            HessianSidecar::open(tf.path()),
            Err(HessianError::TruncatedFile { .. })
        ));
    }

    #[test]
    fn symmetry_check_passes_on_symmetric_h() {
        let tf = make_test_package();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        let ta = sc.get("tA", 0).unwrap();
        HessianSidecar::check_symmetry(&ta, 1e-6).expect("tA is symmetric");
    }

    #[test]
    fn psd_diagonal_check_passes_on_positive_h() {
        let tf = make_test_package();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        let ta = sc.get("tA", 0).unwrap();
        HessianSidecar::check_positive_diagonal(&ta).expect("tA has positive diagonal");
    }
}
