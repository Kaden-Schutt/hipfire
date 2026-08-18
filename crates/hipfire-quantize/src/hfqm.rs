// SPDX-License-Identifier: Apache-2.0
//! HFQM calibration-package reader — unified `.calib.hfq` container.
//!
//! Reads per-tensor Hessians (`<name>.hessian` [K,K]) and imatrix vectors
//! (`<name>.imatrix` [K]) from a HFQM package produced by the native
//! single-load collector (`collect_calibration_artifacts`). Both dense F32
//! and compact qt=130 (BF16 lower-triangle + F32 diagonal) Hessian storage
//! are supported. Consumer (`gptq_pipeline_mq4g256`) always receives a dense
//! full K×K f32 row-major buffer.
//!
//! HFQM layout (mirrors hipfire-runtime `hfq` writer):
//!   header 32B: magic "HFQM" | version u32=1 | arch_id u32 | n_entries u32
//!               | metadata_offset u64 | data_offset u64
//!   metadata: JSON blob `{...}` (self-delimited)
//!   index: n_entries u32 | per-entry { name_len u16 | name | quant_type u8 | n_dims u8
//!           | shape[n_dims×u32] | group_size u32 | data_size u64 }
//!   payloads: concatenated tensor bytes in index order from data_offset.

use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Advice, Mmap};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const HFQM_MAGIC: &[u8; 4] = b"HFQM";
const HFQM_VERSION_SUPPORTED: u32 = 1;
const HEADER_SIZE: usize = 32;
const QUANT_TYPE_F32: u8 = 2;
const QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32: u8 = 130;
const HESSIAN_SUFFIX: &str = ".hessian";
const IMATRIX_SUFFIX: &str = ".imatrix";

#[derive(Debug)]
pub enum HfqmError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    TruncatedFile { needed: usize, have: usize },
    InvalidData(String),
}

impl std::fmt::Display for HfqmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfqmError::Io(e) => write!(f, "I/O error: {e}"),
            HfqmError::InvalidMagic(m) => {
                write!(f, "invalid HFQM magic: got {m:?}, expected {HFQM_MAGIC:?}")
            }
            HfqmError::UnsupportedVersion(v) => write!(
                f,
                "unsupported HFQM version {v}, expected {HFQM_VERSION_SUPPORTED}"
            ),
            HfqmError::TruncatedFile { needed, have } => {
                write!(f, "HFQM truncated: needed {needed} bytes, file is {have}")
            }
            HfqmError::InvalidData(m) => write!(f, "invalid HFQM: {m}"),
        }
    }
}
impl std::error::Error for HfqmError {}
impl From<std::io::Error> for HfqmError {
    fn from(e: std::io::Error) -> Self {
        HfqmError::Io(e)
    }
}

// BF16 -> F32: bf16 keeps top 16 bits of f32 (sign+exponent+7 mantissa bits).
// Reconstruction is bits <<16 as f32. No hipfire-primitives dep needed.
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfqmHessianDtype {
    F32,
    Bf16TrilDiagF32,
}
impl HfqmHessianDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            HfqmHessianDtype::F32 => 4,
            HfqmHessianDtype::Bf16TrilDiagF32 => 0,
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

/// Zero-copy view into one Hessian in the mmap.
pub struct HfqmHessianRef<'a> {
    pub name: &'a str,
    pub k: usize,
    pub dtype: HfqmHessianDtype,
    pub bytes: &'a [u8],
}

impl<'a> HfqmHessianRef<'a> {
    pub fn at(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.k && j < self.k);
        match self.dtype {
            HfqmHessianDtype::F32 => {
                let off = (i * self.k + j) * 4;
                LittleEndian::read_f32(&self.bytes[off..off + 4]) as f64
            }
            HfqmHessianDtype::Bf16TrilDiagF32 => {
                if i == j {
                    let off = i * 4;
                    LittleEndian::read_f32(&self.bytes[off..off + 4]) as f64
                } else {
                    let (r, c) = if i > j { (i, j) } else { (j, i) };
                    let off = self.k * 4 + lower_strict_index(r, c) * 2;
                    bf16_bits_to_f32(LittleEndian::read_u16(&self.bytes[off..off + 2])) as f64
                }
            }
        }
    }

    /// Materialize dense full K×K f32 row-major. This is what `gptq_pipeline_mq4g256` asserts.
    pub fn to_dense_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.k * self.k];
        match self.dtype {
            HfqmHessianDtype::F32 => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = LittleEndian::read_f32(&self.bytes[i * 4..i * 4 + 4]);
                }
            }
            HfqmHessianDtype::Bf16TrilDiagF32 => {
                // Fill: diagonal exact f32, off-diag via bf16 triangle mirrored.
                for i in 0..self.k {
                    for j in 0..self.k {
                        out[i * self.k + j] = self.at(i, j) as f32;
                    }
                }
            }
        }
        out
    }

    pub fn iter_f64(&self) -> impl Iterator<Item = f64> + '_ {
        let k = self.k;
        (0..k * k).map(move |idx| self.at(idx / k, idx % k))
    }
}

/// Zero-copy view into one imatrix vector.
pub struct HfqmImatrixRef<'a> {
    pub name: &'a str,
    pub k: usize,
    pub bytes: &'a [u8],
}
impl<'a> HfqmImatrixRef<'a> {
    pub fn iter_f32(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.k).map(move |idx| LittleEndian::read_f32(&self.bytes[idx * 4..idx * 4 + 4]))
    }
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.iter_f32().collect()
    }
}

struct TensorEntry {
    name: String,
    k: usize,
    dtype: HfqmHessianDtype,
    payload_offset: usize,
    payload_bytes: usize,
}
struct ImatrixEntry {
    name: String,
    k: usize,
    payload_offset: usize,
    payload_bytes: usize,
}

pub struct HfqmPackage {
    mmap: Mmap,
    _file: File,
    index: HashMap<String, TensorEntry>,
    imatrix_index: HashMap<String, ImatrixEntry>,
}

impl std::fmt::Debug for HfqmPackage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfqmPackage")
            .field("mmap_len", &self.mmap.len())
            .field("n_hessians", &self.index.len())
            .field("n_imatrix", &self.imatrix_index.len())
            .finish()
    }
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

impl HfqmPackage {
    pub fn open(path: &Path) -> Result<Self, HfqmError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        {
            mmap.advise(Advice::Sequential).ok();
        }
        let _ = Advice::Sequential;

        if mmap.len() < HEADER_SIZE {
            return Err(HfqmError::TruncatedFile {
                needed: HEADER_SIZE,
                have: mmap.len(),
            });
        }
        let magic: [u8; 4] = mmap[0..4].try_into().unwrap();
        if &magic != HFQM_MAGIC {
            return Err(HfqmError::InvalidMagic(magic));
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != HFQM_VERSION_SUPPORTED {
            return Err(HfqmError::UnsupportedVersion(version));
        }
        let n_entries = LittleEndian::read_u32(&mmap[12..16]) as usize;
        let metadata_offset = LittleEndian::read_u64(&mmap[16..24]) as usize;
        let data_offset = LittleEndian::read_u64(&mmap[24..32]) as usize;
        if metadata_offset > data_offset || data_offset > mmap.len() {
            return Err(HfqmError::InvalidData(format!(
                "offsets metadata={metadata_offset} data={data_offset} len={}",
                mmap.len()
            )));
        }
        let meta_bytes = &mmap[metadata_offset..data_offset];
        let json_end = json_blob_end(meta_bytes)
            .ok_or_else(|| HfqmError::InvalidData("metadata JSON did not end".into()))?;
        let mut pos = metadata_offset + json_end;
        if pos + 4 > data_offset {
            return Err(HfqmError::InvalidData("index missing tensor count".into()));
        }
        let idx_n = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
        if idx_n != n_entries {
            return Err(HfqmError::InvalidData(format!(
                "index count {idx_n} != header {n_entries}"
            )));
        }
        pos += 4;
        let mut index = HashMap::new();
        let mut imatrix_index = HashMap::new();
        let mut cumulative_offset = data_offset;
        for _ in 0..n_entries {
            if pos + 2 > data_offset {
                return Err(HfqmError::InvalidData("index truncated at name len".into()));
            }
            let name_len = LittleEndian::read_u16(&mmap[pos..pos + 2]) as usize;
            pos += 2;
            if pos + name_len + 2 > data_offset {
                return Err(HfqmError::InvalidData(
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
                return Err(HfqmError::InvalidData(
                    "index truncated at shape/data_size".into(),
                ));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize);
                pos += 4;
            }
            pos += 4; // group_size
            let data_size = LittleEndian::read_u64(&mmap[pos..pos + 8]) as usize;
            pos += 8;
            let payload_offset = cumulative_offset;
            cumulative_offset += data_size;
            if cumulative_offset > mmap.len() {
                return Err(HfqmError::TruncatedFile {
                    needed: cumulative_offset,
                    have: mmap.len(),
                });
            }

            if let Some(base) = name.strip_suffix(HESSIAN_SUFFIX) {
                if shape.len() != 2 || shape[0] != shape[1] {
                    continue;
                }
                let k = shape[0];
                let dtype = match quant_type {
                    QUANT_TYPE_F32 => {
                        if k * k * 4 != data_size {
                            return Err(HfqmError::InvalidData(format!(
                                "{name} dense F32 K={k} bytes mismatch"
                            )));
                        }
                        HfqmHessianDtype::F32
                    }
                    QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32 => {
                        let exp = compact_hessian_bytes(k);
                        if exp != data_size {
                            return Err(HfqmError::InvalidData(format!(
                                "{name} compact K={k} exp {exp} != {data_size}"
                            )));
                        }
                        HfqmHessianDtype::Bf16TrilDiagF32
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
                    return Err(HfqmError::InvalidData(format!(
                        "{name} imatrix bytes mismatch"
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

    pub fn get(&self, name: &str) -> Option<HfqmHessianRef<'_>> {
        let e = self.index.get(name)?;
        Some(HfqmHessianRef {
            name: &e.name,
            k: e.k,
            dtype: e.dtype,
            bytes: &self.mmap[e.payload_offset..e.payload_offset + e.payload_bytes],
        })
    }
    /// Alias used by calibration dispatch (expert_idx ignored).
    pub fn get_with_expert(&self, name: &str, _expert_idx: u32) -> Option<HfqmHessianRef<'_>> {
        self.get(name)
    }

    pub fn get_dense_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.get(name).map(|r| r.to_dense_f32())
    }

    pub fn imatrix(&self, name: &str) -> Option<HfqmImatrixRef<'_>> {
        let e = self.imatrix_index.get(name)?;
        Some(HfqmImatrixRef {
            name: &e.name,
            k: e.k,
            bytes: &self.mmap[e.payload_offset..e.payload_offset + e.payload_bytes],
        })
    }
    pub fn tensors(&self) -> impl Iterator<Item = HfqmHessianRef<'_>> + '_ {
        self.index.values().map(|e| HfqmHessianRef {
            name: &e.name,
            k: e.k,
            dtype: e.dtype,
            bytes: &self.mmap[e.payload_offset..e.payload_offset + e.payload_bytes],
        })
    }
    pub fn imatrices(&self) -> impl Iterator<Item = HfqmImatrixRef<'_>> + '_ {
        self.imatrix_index.values().map(|e| HfqmImatrixRef {
            name: &e.name,
            k: e.k,
            bytes: &self.mmap[e.payload_offset..e.payload_offset + e.payload_bytes],
        })
    }
    pub fn n_tensors(&self) -> usize {
        self.index.len()
    }
    pub fn n_imatrix_tensors(&self) -> usize {
        self.imatrix_index.len()
    }
}

// --- Helpers for writing test fixtures (qt=130 encoding) ---

/// Encode a symmetric K×K f32 matrix into compact qt=130 bytes (F32 diagonal + BF16 lower triangle).
/// Used only in tests to build round-trip fixtures.
pub fn encode_compact_hessian(matrix: &[f32], k: usize) -> Vec<u8> {
    assert_eq!(matrix.len(), k * k);
    let mut out = vec![0u8; compact_hessian_bytes(k)];
    // diagonal f32 LE
    for i in 0..k {
        out[i * 4..i * 4 + 4].copy_from_slice(&matrix[i * k + i].to_le_bytes());
    }
    // lower triangle bf16
    for i in 0..k {
        for j in 0..i {
            let idx = lower_strict_index(i, j);
            let off = k * 4 + idx * 2;
            let bf = f32_to_bf16_bits(matrix[i * k + j]);
            out[off..off + 2].copy_from_slice(&bf.to_le_bytes());
        }
    }
    out
}

/// Expand compact qt=130 bytes back to dense full symmetric K×K f32.
pub fn expand_compact_hessian(compact: &[u8], k: usize) -> Vec<f32> {
    assert_eq!(compact.len(), compact_hessian_bytes(k));
    let mut out = vec![0.0f32; k * k];
    for i in 0..k {
        out[i * k + i] = LittleEndian::read_f32(&compact[i * 4..i * 4 + 4]);
    }
    for i in 0..k {
        for j in 0..i {
            let idx = lower_strict_index(i, j);
            let off = k * 4 + idx * 2;
            let v = bf16_bits_to_f32(LittleEndian::read_u16(&compact[off..off + 2]));
            out[i * k + j] = v;
            out[j * k + i] = v;
        }
    }
    out
}

// --- Container detection helper ---
pub fn is_hfqm(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
        return false;
    };
    mmap.len() >= 4 && &mmap[0..4] == HFQM_MAGIC
}
pub fn is_hfhs(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
        return false;
    };
    mmap.len() >= 4 && &mmap[0..4] == b"HFHS"
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_hfqm_package(entries: Vec<(String, u8, Vec<u32>, Vec<u8>)>) -> NamedTempFile {
        // entries: (name, quant_type, shape, payload)
        let metadata = b"{}";
        let n = entries.len() as u32;
        // header 32
        let mut buf: Vec<u8> = Vec::new();
        // Will fill header later after we know offsets; easier to build index first.
        let metadata_offset = 32usize;
        let json_end = metadata.len();
        let index_start = metadata_offset + json_end;
        // index: n_entries u32 + each entry
        let mut index_bytes = Vec::new();
        index_bytes.write_u32::<LittleEndian>(n).unwrap();
        for (name, qt, shape, payload) in &entries {
            index_bytes
                .write_u16::<LittleEndian>(name.len() as u16)
                .unwrap();
            index_bytes.extend_from_slice(name.as_bytes());
            index_bytes.push(*qt);
            index_bytes.push(shape.len() as u8);
            for &d in shape {
                index_bytes.write_u32::<LittleEndian>(d).unwrap();
            }
            index_bytes.write_u32::<LittleEndian>(256).unwrap(); // group_size dummy
            index_bytes
                .write_u64::<LittleEndian>(payload.len() as u64)
                .unwrap();
        }
        let data_offset = index_start + index_bytes.len();
        let total = data_offset + entries.iter().map(|(_, _, _, p)| p.len()).sum::<usize>();
        buf.resize(total, 0);
        // header
        buf[0..4].copy_from_slice(b"HFQM");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&5u32.to_le_bytes()); // arch_id dense
        buf[12..16].copy_from_slice(&n.to_le_bytes());
        buf[16..24].copy_from_slice(&(metadata_offset as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(data_offset as u64).to_le_bytes());
        buf[metadata_offset..metadata_offset + json_end].copy_from_slice(metadata);
        buf[index_start..index_start + index_bytes.len()].copy_from_slice(&index_bytes);
        let mut off = data_offset;
        for (_, _, _, payload) in entries {
            buf[off..off + payload.len()].copy_from_slice(&payload);
            off += payload.len();
        }
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(&buf).unwrap();
        tf.flush().unwrap();
        tf
    }

    #[test]
    fn qt130_expansion_roundtrip_known_symmetric() {
        // Build a known symmetric PSD matrix K=4: H = X^T X style with known values.
        // Use small values to keep bf16 error bounded.
        let k = 4usize;
        // symmetric matrix:
        // [[ 4.0, 1.5, 0.7, 0.2],
        //  [ 1.5, 3.0, 1.0, 0.4],
        //  [ 0.7, 1.0, 2.0, 0.9],
        //  [ 0.2, 0.4, 0.9, 1.5]]
        let mat: Vec<f32> = vec![
            4.0, 1.5, 0.7, 0.2, 1.5, 3.0, 1.0, 0.4, 0.7, 1.0, 2.0, 0.9, 0.2, 0.4, 0.9, 1.5,
        ];
        let compact = encode_compact_hessian(&mat, k);
        assert_eq!(compact.len(), compact_hessian_bytes(k));
        let expanded = expand_compact_hessian(&compact, k);
        assert_eq!(expanded.len(), k * k);
        // diagonal exact (stored f32)
        for i in 0..k {
            assert_eq!(
                expanded[i * k + i],
                mat[i * k + i],
                "diagonal must be exact at {i}"
            );
        }
        // off-diagonal within bf16 tolerance: bf16 has ~7 mantissa bits vs f32 23,
        // so step ~ 2^(exp-7). For values ~1.0, ulp ~0.0078. Check relative.
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                let a = mat[i * k + j];
                let b = expanded[i * k + j];
                let diff = (a - b).abs();
                // bf16 absolute error for |a|~1 is <0.01, for larger ~4 maybe 0.03.
                // Use loose tolerance 0.02 or 1% relative.
                let tol = (a.abs() * 0.01).max(0.02);
                assert!(
                    diff <= tol,
                    "off-diag [{i},{j}] diff {diff} exceeds {tol}: {a} vs {b}"
                );
                // symmetry
                assert_eq!(
                    expanded[i * k + j],
                    expanded[j * k + i],
                    "symmetry fail [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn qt130_ref_expansion_via_package() {
        // End-to-end through HfqmPackage reading a file with one compact hessian + one imatrix.
        let k = 3usize;
        let h: Vec<f32> = vec![2.0, 0.5, 0.3, 0.5, 1.5, 0.2, 0.3, 0.2, 1.0];
        let h_compact = encode_compact_hessian(&h, k);
        let imat: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut imat_bytes = Vec::new();
        for &v in &imat {
            imat_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let tf = write_hfqm_package(vec![
            (
                "model.layers.0.mlp.down_proj".to_string() + ".hessian",
                QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32,
                vec![k as u32, k as u32],
                h_compact,
            ),
            (
                "model.layers.0.mlp.down_proj".to_string() + ".imatrix",
                QUANT_TYPE_F32,
                vec![k as u32],
                imat_bytes,
            ),
        ]);
        let pkg = HfqmPackage::open(tf.path()).expect("open");
        assert_eq!(pkg.n_tensors(), 1);
        assert_eq!(pkg.n_imatrix_tensors(), 1);
        let href = pkg.get("model.layers.0.mlp.down_proj").unwrap();
        assert_eq!(href.k, k);
        assert_eq!(href.dtype, HfqmHessianDtype::Bf16TrilDiagF32);
        // diagonal exact
        for i in 0..k {
            assert_eq!(href.at(i, i) as f32, h[i * k + i]);
        }
        let dense = href.to_dense_f32();
        // dense symmetric
        for i in 0..k {
            for j in 0..k {
                assert_eq!(dense[i * k + j], dense[j * k + i]);
            }
        }
        let iref = pkg.imatrix("model.layers.0.mlp.down_proj").unwrap();
        assert_eq!(iref.to_vec_f32(), imat);
    }

    #[test]
    fn dense_f32_and_compact_both_open() {
        let k = 2usize;
        let h_dense: Vec<f32> = vec![1.0, 0.25, 0.25, 2.0];
        let mut h_bytes = Vec::new();
        for &v in &h_dense {
            h_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let tf = write_hfqm_package(vec![(
            "a.hessian".to_string(),
            QUANT_TYPE_F32,
            vec![k as u32, k as u32],
            h_bytes,
        )]);
        let pkg = HfqmPackage::open(tf.path()).unwrap();
        let href = pkg.get("a").unwrap();
        assert_eq!(href.dtype, HfqmHessianDtype::F32);
        assert_eq!(href.to_dense_f32(), h_dense);
    }
}
