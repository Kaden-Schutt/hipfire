//! HFHS Hipfire Hessian Sidecar reader.
//!
//! Reads the per-tensor Hessian binary file produced by
//! `scripts/collect_hessian.py` (the Python calibration collector for Stage B
//! GPTQ). Format specification: `docs/plans/gptq-hessian-format.md`.
//!
//! Design choices:
//! - **mmap-based.** A 9B Hessian sidecar is ~6 GB; mmap with sequential
//!   POSIX advice lets the kernel page in tensor-by-tensor as the
//!   quantizer's per-tensor Cholesky walk progresses, then evicts.
//! - **Zero-copy.** `HessianRef` borrows from the mmap; the caller copies /
//!   promotes only when needed (e.g. Cholesky's FP32 → FP64 promotion at
//!   quantize time).
//! - **Index built at open time.** A `HashMap<(name, expert_idx), offset>`
//!   gives O(1) lookup for the quantizer's per-tensor query. With 200
//!   tensors per 9B model, the index is < 32 KB.
//!
//! Consumer integration (Phase 2): `crates/hipfire-quantize/src/gptq.rs`
//! calls `HessianSidecar::open(path)` once per model, then queries
//! `get(tensor_name_without_dot_weight_suffix, 0)` per MQ4G256 tensor.

use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Advice, Mmap};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const HFHS_MAGIC: &[u8; 4] = b"HFHS";
const HFHS_VERSION_SUPPORTED: u32 = 1;
const HEADER_SIZE: usize = 24;
const DTYPE_F32: u32 = 1;
const DTYPE_F64: u32 = 2;

#[derive(Debug)]
pub enum HessianError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    TruncatedFile { needed: usize, have: usize },
    NegativeDiagonal { tensor: String, index: usize, value: f32 },
    UnknownDtype(u32),
}

impl std::fmt::Display for HessianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HessianError::Io(e) => write!(f, "I/O error: {e}"),
            HessianError::InvalidMagic(m) => {
                write!(f, "invalid HFHS magic: got {m:?}, expected {:?}", HFHS_MAGIC)
            }
            HessianError::UnsupportedVersion(v) => {
                write!(f, "unsupported HFHS version {v}, this build understands v{HFHS_VERSION_SUPPORTED}")
            }
            HessianError::TruncatedFile { needed, have } => {
                write!(f, "HFHS truncated: needed {needed} bytes, file is {have}")
            }
            HessianError::NegativeDiagonal { tensor, index, value } => write!(
                f,
                "Hessian for tensor {tensor:?} has negative diagonal H[{index},{index}] = {value} \
                 (should be ≥0 by PSD construction; likely FP corruption — fall back to plain MQ4)"
            ),
            HessianError::UnknownDtype(d) => write!(f, "unknown HFHS dtype flag {d}"),
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
}

impl HessianDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            HessianDtype::F32 => 4,
            HessianDtype::F64 => 8,
        }
    }
}

/// Zero-copy view into one Hessian record in the mmap.
pub struct HessianRef<'a> {
    pub name: &'a str,
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
        let bytes = self.bytes;
        let n = self.k * self.k;
        (0..n).map(move |i| match self.dtype {
            HessianDtype::F32 => LittleEndian::read_f32(&bytes[i * 4..i * 4 + 4]) as f64,
            HessianDtype::F64 => LittleEndian::read_f64(&bytes[i * 8..i * 8 + 8]),
        })
    }

    /// Read the `[i, j]` entry as f64. O(1).
    pub fn at(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.k && j < self.k, "out of bounds: H[{i},{j}] K={}", self.k);
        let off = (i * self.k + j) * self.dtype.size_bytes();
        match self.dtype {
            HessianDtype::F32 => LittleEndian::read_f32(&self.bytes[off..off + 4]) as f64,
            HessianDtype::F64 => LittleEndian::read_f64(&self.bytes[off..off + 8]),
        }
    }
}

/// Per-tensor record layout (computed at open, points into the mmap).
struct TensorEntry {
    name_offset: usize,    // byte offset of the name string in mmap
    name_len: usize,
    expert_idx: u32,
    k: usize,
    dtype: HessianDtype,
    payload_offset: usize, // byte offset of K*K float payload in mmap
    payload_bytes: usize,
}

pub struct HessianSidecar {
    // Mmap kept alive for the sidecar's lifetime; all `HessianRef` views
    // borrow from this. `_file` keeps the fd alive on Unix.
    mmap: Mmap,
    _file: File,
    index: HashMap<(String, u32), TensorEntry>,
}

impl std::fmt::Debug for HessianSidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HessianSidecar")
            .field("mmap_len", &self.mmap.len())
            .field("n_tensors", &self.index.len())
            .finish()
    }
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
        // Header
        let magic: [u8; 4] = mmap[0..4].try_into().unwrap();
        if &magic != HFHS_MAGIC {
            return Err(HessianError::InvalidMagic(magic));
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != HFHS_VERSION_SUPPORTED {
            return Err(HessianError::UnsupportedVersion(version));
        }
        let n_tensors = LittleEndian::read_u64(&mmap[8..16]) as usize;
        let _reserved = LittleEndian::read_u64(&mmap[16..24]);

        // Walk records, build index.
        let mut index = HashMap::with_capacity(n_tensors);
        let mut pos = HEADER_SIZE;
        for _ in 0..n_tensors {
            if pos + 4 > mmap.len() {
                return Err(HessianError::TruncatedFile { needed: pos + 4, have: mmap.len() });
            }
            let name_len = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            if pos + name_len + 12 > mmap.len() {
                return Err(HessianError::TruncatedFile {
                    needed: pos + name_len + 12,
                    have: mmap.len(),
                });
            }
            let name_offset = pos;
            let name = std::str::from_utf8(&mmap[pos..pos + name_len])
                .map_err(|_| HessianError::InvalidMagic([0; 4]))?  // reuse for UTF-8 failure
                .to_string();
            pos += name_len;
            let expert_idx = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let k = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            let dtype_flag = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let dtype = match dtype_flag {
                DTYPE_F32 => HessianDtype::F32,
                DTYPE_F64 => HessianDtype::F64,
                d => return Err(HessianError::UnknownDtype(d)),
            };
            let payload_bytes = k * k * dtype.size_bytes();
            if pos + payload_bytes > mmap.len() {
                return Err(HessianError::TruncatedFile {
                    needed: pos + payload_bytes,
                    have: mmap.len(),
                });
            }
            index.insert(
                (name, expert_idx),
                TensorEntry {
                    name_offset,
                    name_len,
                    expert_idx,
                    k,
                    dtype,
                    payload_offset: pos,
                    payload_bytes,
                },
            );
            pos += payload_bytes;
        }

        Ok(Self {
            mmap,
            _file: file,
            index,
        })
    }

    /// Look up a Hessian by (`tensor_name`, `expert_idx`). The name SHOULD
    /// be the `.hfq` tensor name with the trailing `.weight` stripped (see
    /// the format spec §3.1). Returns `None` if the tensor isn't in the
    /// sidecar — the quantizer treats this as "skip GPTQ for this tensor".
    pub fn get(&self, name: &str, expert_idx: u32) -> Option<HessianRef<'_>> {
        // Allocate-free lookup: HashMap key is (&str, u32) won't work
        // because the map owns the String. Use a per-call key tuple via
        // `get_key_value` requires Borrow<(String,u32)> — clone is cheaper
        // than alternative gymnastics for this rare-call path.
        let entry = self.index.get(&(name.to_string(), expert_idx))?;
        Some(HessianRef {
            name: std::str::from_utf8(&self.mmap[entry.name_offset..entry.name_offset + entry.name_len])
                .ok()?,
            expert_idx: entry.expert_idx,
            k: entry.k,
            dtype: entry.dtype,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    /// Iterate all stored Hessians. Used for bulk validation passes (e.g.
    /// symmetry / PSD check at start of quantize) and debug dumps.
    pub fn tensors(&self) -> impl Iterator<Item = HessianRef<'_>> + '_ {
        self.index.values().map(|entry| HessianRef {
            name: std::str::from_utf8(&self.mmap[entry.name_offset..entry.name_offset + entry.name_len])
                .unwrap_or(""),
            expert_idx: entry.expert_idx,
            k: entry.k,
            dtype: entry.dtype,
            bytes: &self.mmap[entry.payload_offset..entry.payload_offset + entry.payload_bytes],
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.index.len()
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
    /// guarantees `H[i,i] >= 0`; FP corruption (e.g. from a partial sidecar
    /// download) can produce negatives. Returns the first negative diagonal,
    /// if any.
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

    /// Build a minimal v1 HFHS file with two tiny tensors for round-trip
    /// testing.
    fn make_test_sidecar() -> NamedTempFile {
        let mut tf = NamedTempFile::new().unwrap();
        let f = tf.as_file_mut();

        // Header
        f.write_all(b"HFHS").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();      // version
        f.write_all(&2u64.to_le_bytes()).unwrap();      // n_tensors
        f.write_all(&0u64.to_le_bytes()).unwrap();      // reserved

        // Tensor 1: "tA", expert_idx=0, K=2, FP32, H = [[1.0, 0.5], [0.5, 2.0]]
        let name1 = b"tA";
        f.write_all(&(name1.len() as u32).to_le_bytes()).unwrap();
        f.write_all(name1).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();      // expert_idx
        f.write_all(&2u32.to_le_bytes()).unwrap();      // K
        f.write_all(&1u32.to_le_bytes()).unwrap();      // dtype = F32
        for v in [1.0_f32, 0.5_f32, 0.5_f32, 2.0_f32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }

        // Tensor 2: "tB", expert_idx=3, K=2, FP64, H = [[3.0, 1.0], [1.0, 4.0]]
        let name2 = b"tB";
        f.write_all(&(name2.len() as u32).to_le_bytes()).unwrap();
        f.write_all(name2).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();      // expert_idx
        f.write_all(&2u32.to_le_bytes()).unwrap();      // K
        f.write_all(&2u32.to_le_bytes()).unwrap();      // dtype = F64
        for v in [3.0_f64, 1.0_f64, 1.0_f64, 4.0_f64] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }

        tf.flush().unwrap();
        tf
    }

    #[test]
    fn open_and_lookup_roundtrip() {
        let tf = make_test_sidecar();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        assert_eq!(sc.n_tensors(), 2);

        let ta = sc.get("tA", 0).expect("tA missing");
        assert_eq!(ta.k, 2);
        assert_eq!(ta.dtype, HessianDtype::F32);
        assert_eq!(ta.at(0, 0), 1.0);
        assert_eq!(ta.at(0, 1), 0.5);
        assert_eq!(ta.at(1, 0), 0.5);
        assert_eq!(ta.at(1, 1), 2.0);

        let tb = sc.get("tB", 3).expect("tB missing");
        assert_eq!(tb.k, 2);
        assert_eq!(tb.dtype, HessianDtype::F64);
        assert_eq!(tb.at(0, 0), 3.0);
        assert_eq!(tb.at(1, 1), 4.0);

        // Wrong expert_idx → None
        assert!(sc.get("tB", 0).is_none());
        assert!(sc.get("not_there", 0).is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"XXXX\x01\x00\x00\x00").unwrap();
        tf.write_all(&[0u8; 16]).unwrap();
        tf.flush().unwrap();
        match HessianSidecar::open(tf.path()) {
            Err(HessianError::InvalidMagic(m)) => assert_eq!(&m, b"XXXX"),
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn rejects_future_version() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"HFHS").unwrap();
        tf.write_all(&99u32.to_le_bytes()).unwrap();
        tf.write_all(&[0u8; 16]).unwrap();
        tf.flush().unwrap();
        match HessianSidecar::open(tf.path()) {
            Err(HessianError::UnsupportedVersion(v)) => assert_eq!(v, 99),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"HFHS").unwrap();
        // Only 4 of 24 header bytes
        tf.flush().unwrap();
        assert!(matches!(
            HessianSidecar::open(tf.path()),
            Err(HessianError::TruncatedFile { .. })
        ));
    }

    #[test]
    fn symmetry_check_passes_on_symmetric_h() {
        let tf = make_test_sidecar();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        let ta = sc.get("tA", 0).unwrap();
        HessianSidecar::check_symmetry(&ta, 1e-6).expect("tA is symmetric");
    }

    #[test]
    fn psd_diagonal_check_passes_on_positive_h() {
        let tf = make_test_sidecar();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        let ta = sc.get("tA", 0).unwrap();
        HessianSidecar::check_positive_diagonal(&ta).expect("tA has positive diagonal");
    }
}
