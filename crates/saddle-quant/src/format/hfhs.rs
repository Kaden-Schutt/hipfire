// SPDX-License-Identifier: Apache-2.0
//! HFHS v1 and E8H1 `.hblk` readers — single owner for hipfire Hessian formats.
//!
//! HFHS v1 (full K×K per tensor):
//!   header 24 B LE: magic b"HFHS" | version:u32=1 | n_tensors:u64 | reserved:u64=0
//!   record: name_len:u32 | name:utf8 | expert_idx:u32 | K:u32 | dtype_flag:u32
//!           payload: K*K floats row-major (flag 1=f32, 2=f64)
//!
//! E8H1 .hblk (block-diagonal 256):
//!   header 12 B LE: magic:u32=0x45384831 ("E8H1") | n_blocks:u32 | K:u32
//!   payload: n_blocks * 256*256 f32 LE row-major per block; K MUST equal n_blocks*256
//!   values are raw sum_t x_b x_b^T (NOT normalized)

use crate::{QuantError, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

const HFHS_MAGIC: &[u8; 4] = b"HFHS";
const HFHS_VERSION: u32 = 1;
const E8H1_MAGIC: u32 = 0x45384831;
const HESSIAN_BLOCK: u32 = 256;
const HBLK_BLOCK_ELEMS: usize = 256 * 256;
const HBLK_BLOCK_BYTES: usize = HBLK_BLOCK_ELEMS * 4;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One record in an HFHS v1 file — index entry only, payload stays on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfhsEntry {
    pub name: String,
    pub expert_idx: u32,
    pub k: u32,
    pub dtype_flag: u32,
    /// Byte offset of the K×K payload from the start of the file.
    pub data_offset: u64,
}

/// Index of an HFHS v1 file. No payloads are resident.
#[derive(Debug, Clone)]
pub struct HfhsFile {
    pub version: u32,
    pub entries: Vec<HfhsEntry>,
}

/// Header of an E8H1 `.hblk` file (block-diagonal Hessians).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct E8h1File {
    pub n_blocks: u32,
    pub k: u32,
}

// ---------------------------------------------------------------------------
// Helpers — LE decodes without an extra crate.
// ---------------------------------------------------------------------------

#[inline]
fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

#[inline]
fn u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

// ---------------------------------------------------------------------------
// HFHS
// ---------------------------------------------------------------------------

/// Open and index an HFHS v1 file. Payloads are NOT read; `data_offset` is
/// recorded for later [`HfhsFile::diagonal`] via the mmap.
pub fn open(path: impl AsRef<Path>) -> Result<HfhsFile> {
    let file = File::open(path.as_ref())?;
    // SAFETY: read-only mmap of the file on disk; the file stays open via
    // the mmap's lifetime in the caller for diagonal/hblk_block. Here we just
    // need a temporary view to index; the returned HfhsFile holds offsets.
    let mmap = unsafe { Mmap::map(&file)? };
    parse_hfhs(&mmap)
}

fn parse_hfhs(mmap: &[u8]) -> Result<HfhsFile> {
    if mmap.len() < 24 {
        return Err(QuantError::Truncated {
            artifact: "HFHS",
            context: "header",
            need: 24,
            have: mmap.len(),
        });
    }
    if &mmap[0..4] != HFHS_MAGIC {
        let found = String::from_utf8_lossy(&mmap[0..4]).to_string();
        return Err(QuantError::BadMagic {
            artifact: "HFHS",
            expected: "HFHS",
            found,
        });
    }
    let version = u32_le(mmap, 4);
    if version != HFHS_VERSION {
        return Err(QuantError::UnsupportedVersion {
            artifact: "HFHS",
            found: version,
            supported: "1",
        });
    }
    let n_tensors = u64_le(mmap, 8) as usize;
    // reserved at 16..24 ignored

    let mut entries = Vec::with_capacity(n_tensors);
    let mut pos: usize = 24;
    for _ in 0..n_tensors {
        if pos + 4 > mmap.len() {
            return Err(QuantError::Truncated {
                artifact: "HFHS",
                context: "name_len",
                need: pos + 4,
                have: mmap.len(),
            });
        }
        let name_len = u32_le(mmap, pos) as usize;
        pos += 4;
        if pos + name_len + 12 > mmap.len() {
            return Err(QuantError::Truncated {
                artifact: "HFHS",
                context: "record header",
                need: pos + name_len + 12,
                have: mmap.len(),
            });
        }
        let name_bytes = &mmap[pos..pos + name_len];
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| QuantError::Malformed(format!("invalid utf8 in HFHS name: {e}")))?
            .to_string();
        pos += name_len;
        let expert_idx = u32_le(mmap, pos);
        pos += 4;
        let k = u32_le(mmap, pos);
        pos += 4;
        let dtype_flag = u32_le(mmap, pos);
        pos += 4;
        let esz: usize = match dtype_flag {
            1 => 4,
            2 => 8,
            _ => {
                return Err(QuantError::Malformed(format!(
                    "unknown HFHS dtype_flag {dtype_flag}"
                )))
            }
        };
        let k_u64 = k as u64;
        let payload_bytes = k_u64
            .checked_mul(k_u64)
            .and_then(|v| v.checked_mul(esz as u64))
            .ok_or_else(|| QuantError::Malformed(format!("K={k} payload size overflow")))?
            as usize;
        if pos + payload_bytes > mmap.len() {
            return Err(QuantError::Truncated {
                artifact: "HFHS",
                context: "payload",
                need: pos + payload_bytes,
                have: mmap.len(),
            });
        }
        let data_offset = pos as u64;
        entries.push(HfhsEntry {
            name,
            expert_idx,
            k,
            dtype_flag,
            data_offset,
        });
        pos += payload_bytes;
    }
    Ok(HfhsFile { version, entries })
}

impl HfhsFile {
    /// Convenience associated function mirroring the free [`open`].
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        open(path)
    }

    /// Extract only the diagonal `H[j][j]` for `j in 0..K`, striding the
    /// row-major K×K payload. Handles both `dtype_flag` 1 (f32) and 2 (f64).
    ///
    /// `mmap` must be the mmap of the same file that was indexed to produce
    /// `self`/`entry`. This keeps the reader index-only: a file that is tens
    /// of GB is never materialised.
    pub fn diagonal(&self, mmap: &Mmap, entry: &HfhsEntry) -> Result<Vec<f64>> {
        let k = entry.k as usize;
        let esz: usize = match entry.dtype_flag {
            1 => 4,
            2 => 8,
            _ => {
                return Err(QuantError::Malformed(format!(
                    "unknown HFHS dtype_flag {}",
                    entry.dtype_flag
                )))
            }
        };
        let payload_bytes = k
            .checked_mul(k)
            .and_then(|v| v.checked_mul(esz))
            .ok_or_else(|| QuantError::Malformed(format!("K={} payload size overflow", entry.k)))?;
        let off = entry.data_offset as usize;
        if off + payload_bytes > mmap.len() {
            return Err(QuantError::Truncated {
                artifact: "HFHS",
                context: "payload",
                need: off + payload_bytes,
                have: mmap.len(),
            });
        }
        let mut diag = Vec::with_capacity(k);
        for j in 0..k {
            let idx = j * k + j;
            let b_off = off + idx * esz;
            // b_off + esz <= off + payload_bytes <= mmap.len() by check above
            let v = match entry.dtype_flag {
                1 => {
                    let bytes: [u8; 4] = mmap[b_off..b_off + 4].try_into().unwrap();
                    f32::from_le_bytes(bytes) as f64
                }
                2 => {
                    let bytes: [u8; 8] = mmap[b_off..b_off + 8].try_into().unwrap();
                    f64::from_le_bytes(bytes)
                }
                _ => unreachable!(),
            };
            diag.push(v);
        }
        Ok(diag)
    }
}

// ---------------------------------------------------------------------------
// E8H1
// ---------------------------------------------------------------------------

/// Open and validate an E8H1 `.hblk` file.
pub fn open_hblk(path: impl AsRef<Path>) -> Result<E8h1File> {
    let file = File::open(path.as_ref())?;
    let mmap = unsafe { Mmap::map(&file)? };
    parse_hblk(&mmap)
}

fn parse_hblk(mmap: &[u8]) -> Result<E8h1File> {
    if mmap.len() < 12 {
        return Err(QuantError::Truncated {
            artifact: "E8H1",
            context: "header",
            need: 12,
            have: mmap.len(),
        });
    }
    let magic = u32_le(mmap, 0);
    if magic != E8H1_MAGIC {
        return Err(QuantError::BadMagic {
            artifact: "E8H1",
            expected: "E8H1",
            found: format!("0x{magic:08x}"),
        });
    }
    let n_blocks = u32_le(mmap, 4);
    let k = u32_le(mmap, 8);
    let expected_k = n_blocks
        .checked_mul(HESSIAN_BLOCK)
        .ok_or_else(|| QuantError::Malformed(format!("n_blocks={n_blocks} overflow")))?;
    if k != expected_k {
        return Err(QuantError::Malformed(format!(
            "K={k} != n_blocks={n_blocks}*256 (expected {expected_k})"
        )));
    }
    let total_payload = (n_blocks as usize)
        .checked_mul(HBLK_BLOCK_BYTES)
        .ok_or_else(|| QuantError::Malformed(format!("n_blocks={n_blocks} payload overflow")))?;
    let want = 12usize
        .checked_add(total_payload)
        .ok_or_else(|| QuantError::Malformed("E8H1 payload size overflow".to_string()))?;
    if mmap.len() < want {
        return Err(QuantError::Truncated {
            artifact: "E8H1",
            context: "payload",
            need: want,
            have: mmap.len(),
        });
    }
    Ok(E8h1File { n_blocks, k })
}

impl E8h1File {
    /// Convenience associated function mirroring the free [`open_hblk`].
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        open_hblk(path)
    }

    /// Return one 256×256 block (row-major, `n_blocks` blocks total).
    pub fn hblk_block(&self, mmap: &Mmap, i: u32) -> Result<Vec<f32>> {
        if i >= self.n_blocks {
            return Err(QuantError::Malformed(format!(
                "block index {i} out of range (n_blocks={})",
                self.n_blocks
            )));
        }
        let total_payload = (self.n_blocks as usize) * HBLK_BLOCK_BYTES;
        let want = 12 + total_payload;
        if mmap.len() < want {
            return Err(QuantError::Truncated {
                artifact: "E8H1",
                context: "payload",
                need: want,
                have: mmap.len(),
            });
        }
        let offset = 12 + (i as usize) * HBLK_BLOCK_BYTES;
        if offset + HBLK_BLOCK_BYTES > mmap.len() {
            return Err(QuantError::Truncated {
                artifact: "E8H1",
                context: "block payload",
                need: offset + HBLK_BLOCK_BYTES,
                have: mmap.len(),
            });
        }
        let mut out = Vec::with_capacity(HBLK_BLOCK_ELEMS);
        for idx in 0..HBLK_BLOCK_ELEMS {
            let o = offset + idx * 4;
            let bytes: [u8; 4] = mmap[o..o + 4].try_into().unwrap();
            out.push(f32::from_le_bytes(bytes));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared contract
// ---------------------------------------------------------------------------

/// Filesystem key for a Hessian tensor, replacing `/` and `\` with `_` and
/// `..` with `_`. Contract shared by all writers — ported exactly from
/// `reference_gptq/formats.py:hessian_key`.
pub fn hessian_key(tensor_name: &str) -> String {
    tensor_name
        .replace('/', "_")
        .replace('\\', "_")
        .replace("..", "_")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn mmap_file(path: &Path) -> Mmap {
        let f = File::open(path).unwrap();
        unsafe { Mmap::map(&f).unwrap() }
    }

    fn write_hfhs_header(f: &mut File, version: u32, n_tensors: u64) {
        f.write_all(b"HFHS").unwrap();
        f.write_all(&version.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
    }

    fn write_hfhs_record_f32(f: &mut File, name: &str, expert_idx: u32, k: u32, payload: &[f32]) {
        let nb = name.as_bytes();
        f.write_all(&(nb.len() as u32).to_le_bytes()).unwrap();
        f.write_all(nb).unwrap();
        f.write_all(&expert_idx.to_le_bytes()).unwrap();
        f.write_all(&k.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // f32
        for v in payload {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    fn write_hfhs_record_f64(f: &mut File, name: &str, expert_idx: u32, k: u32, payload: &[f64]) {
        let nb = name.as_bytes();
        f.write_all(&(nb.len() as u32).to_le_bytes()).unwrap();
        f.write_all(nb).unwrap();
        f.write_all(&expert_idx.to_le_bytes()).unwrap();
        f.write_all(&k.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap(); // f64
        for v in payload {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn hfhs_roundtrip_two_entries_mixed_dtype() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            write_hfhs_header(&mut f, 1, 2);
            // Entry 0: f32, K=3
            // row-major 3x3: diag [1.0,2.0,3.0], off-diag 0.5
            let m_f32: Vec<f32> = vec![
                1.0, 0.5, 0.5, //
                0.5, 2.0, 0.5, //
                0.5, 0.5, 3.0,
            ];
            write_hfhs_record_f32(&mut f, "layer.attn.q_proj", 0, 3, &m_f32);
            // Entry 1: f64, K=2
            let m_f64: Vec<f64> = vec![
                10.0, 1.0, //
                1.0, 20.0,
            ];
            write_hfhs_record_f64(&mut f, "mlp.down_proj", 1, 2, &m_f64);
            f.flush().unwrap();
        }
        let hfhs = open(&path).unwrap();
        assert_eq!(hfhs.version, 1);
        assert_eq!(hfhs.entries.len(), 2);
        assert_eq!(hfhs.entries[0].name, "layer.attn.q_proj");
        assert_eq!(hfhs.entries[0].k, 3);
        assert_eq!(hfhs.entries[0].expert_idx, 0);
        assert_eq!(hfhs.entries[0].dtype_flag, 1);
        assert_eq!(hfhs.entries[1].name, "mlp.down_proj");
        assert_eq!(hfhs.entries[1].k, 2);
        assert_eq!(hfhs.entries[1].expert_idx, 1);
        assert_eq!(hfhs.entries[1].dtype_flag, 2);
        // offsets: first payload immediately after its header
        let name0_len = "layer.attn.q_proj".len();
        let expected_off0 = (24 + 4 + name0_len + 12) as u64;
        assert_eq!(hfhs.entries[0].data_offset, expected_off0);
        let expected_off1 = expected_off0 + 9 * 4 + 4 + "mlp.down_proj".len() as u64 + 12;
        assert_eq!(hfhs.entries[1].data_offset, expected_off1);

        let mmap = mmap_file(&path);
        let d0 = hfhs.diagonal(&mmap, &hfhs.entries[0]).unwrap();
        assert_eq!(d0.len(), 3);
        assert_eq!(d0, vec![1.0, 2.0, 3.0]);
        let d1 = hfhs.diagonal(&mmap, &hfhs.entries[1]).unwrap();
        assert_eq!(d1.len(), 2);
        assert_eq!(d1, vec![10.0, 20.0]);
    }

    #[test]
    fn hfhs_diagonal_does_not_materialise_full_matrix() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let k: u32 = 64;
        let diag_vals: Vec<f64> = (0..k).map(|j| j as f64 * 1.5 + 0.25).collect();
        // Build payload where diag = diag_vals, off-diag = 999.0 (f32)
        let mut payload = vec![999.0f32; (k as usize) * (k as usize)];
        for j in 0..k as usize {
            payload[j * k as usize + j] = diag_vals[j] as f32;
        }
        {
            let mut f = File::create(&path).unwrap();
            write_hfhs_header(&mut f, 1, 1);
            write_hfhs_record_f32(&mut f, "big.tensor", 0, k, &payload);
            f.flush().unwrap();
        }
        let hfhs = open(&path).unwrap();
        assert_eq!(hfhs.entries.len(), 1);
        assert_eq!(hfhs.entries[0].k, 64);
        let mmap = mmap_file(&path);
        let d = hfhs.diagonal(&mmap, &hfhs.entries[0]).unwrap();
        assert_eq!(d.len(), 64, "diagonal must be K, not K*K");
        assert_ne!(d.len(), 4096);
        // Compare with f32→f64 conversion (payload was f32)
        let expected: Vec<f64> = diag_vals.iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(d, expected);
        // Ensure we did not mistake off-diag (999) for diag
        for v in &d {
            assert!(*v != 999.0);
        }
    }

    #[test]
    fn e8h1_roundtrip_two_blocks() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let n_blocks: u32 = 2;
        let k: u32 = 512;
        // block0: 0..65535 as f32, block1: 65536..131071
        let block0: Vec<f32> = (0..HBLK_BLOCK_ELEMS).map(|i| i as f32).collect();
        let block1: Vec<f32> = (0..HBLK_BLOCK_ELEMS).map(|i| (i + HBLK_BLOCK_ELEMS) as f32).collect();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&E8H1_MAGIC.to_le_bytes()).unwrap();
            f.write_all(&n_blocks.to_le_bytes()).unwrap();
            f.write_all(&k.to_le_bytes()).unwrap();
            for v in &block0 {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            for v in &block1 {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.flush().unwrap();
        }
        let e8 = open_hblk(&path).unwrap();
        assert_eq!(e8.n_blocks, 2);
        assert_eq!(e8.k, 512);
        let mmap = mmap_file(&path);
        let got1 = e8.hblk_block(&mmap, 1).unwrap();
        assert_eq!(got1.len(), HBLK_BLOCK_ELEMS);
        assert_eq!(got1, block1);
        // also check block 0 for completeness
        let got0 = e8.hblk_block(&mmap, 0).unwrap();
        assert_eq!(got0, block0);
    }

    #[test]
    fn hfhs_bad_magic() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"BAD!").unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }
        let err = open(&path).unwrap_err();
        match err {
            QuantError::BadMagic { artifact, .. } => assert_eq!(artifact, "HFHS"),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn hfhs_unsupported_version() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            write_hfhs_header(&mut f, 2, 0); // version 2 unsupported
            f.flush().unwrap();
        }
        let err = open(&path).unwrap_err();
        match err {
            QuantError::UnsupportedVersion { artifact, found, .. } => {
                assert_eq!(artifact, "HFHS");
                assert_eq!(found, 2);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn hfhs_truncated_payload() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            write_hfhs_header(&mut f, 1, 1);
            // Claim K=4 f32 => need 64 bytes payload, provide only 10
            let name = "t";
            f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap(); // expert_idx
            f.write_all(&4u32.to_le_bytes()).unwrap(); // K
            f.write_all(&1u32.to_le_bytes()).unwrap(); // f32
            f.write_all(&[0u8; 10]).unwrap(); // truncated payload
            f.flush().unwrap();
        }
        let err = open(&path).unwrap_err();
        match err {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "HFHS"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn e8h1_malformed_k_mismatch() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&E8H1_MAGIC.to_le_bytes()).unwrap();
            f.write_all(&2u32.to_le_bytes()).unwrap(); // n_blocks=2
            f.write_all(&500u32.to_le_bytes()).unwrap(); // K=500 != 512
            // pad payload anyway
            f.write_all(&vec![0u8; 2 * HBLK_BLOCK_BYTES]).unwrap();
            f.flush().unwrap();
        }
        let err = open_hblk(&path).unwrap_err();
        match err {
            QuantError::Malformed(msg) => assert!(msg.contains("K="), "msg: {msg}"),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn e8h1_bad_magic() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&0xdeadbeefu32.to_le_bytes()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&256u32.to_le_bytes()).unwrap();
            f.write_all(&vec![0u8; HBLK_BLOCK_BYTES]).unwrap();
            f.flush().unwrap();
        }
        let err = open_hblk(&path).unwrap_err();
        match err {
            QuantError::BadMagic { artifact, .. } => assert_eq!(artifact, "E8H1"),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn e8h1_truncated() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&E8H1_MAGIC.to_le_bytes()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&256u32.to_le_bytes()).unwrap();
            // provide only half a block
            f.write_all(&vec![0u8; HBLK_BLOCK_BYTES / 2]).unwrap();
            f.flush().unwrap();
        }
        let err = open_hblk(&path).unwrap_err();
        match err {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "E8H1"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn hessian_key_replaces() {
        assert_eq!(hessian_key("a/b"), "a_b");
        assert_eq!(hessian_key("a\\b"), "a_b");
        assert_eq!(hessian_key("a..b"), "a_b");
        assert_eq!(hessian_key("a/b\\c..d"), "a_b_c_d");
        assert_eq!(hessian_key("model/layers/0/attn"), "model_layers_0_attn");
        assert_eq!(hessian_key("..hidden"), "_hidden");
        assert_eq!(hessian_key("a..b..c"), "a_b_c");
    }

    #[test]
    fn diagonal_truncated_mmap() {
        // Entry claims payload beyond mmap length
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            write_hfhs_header(&mut f, 1, 1);
            let m: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            write_hfhs_record_f32(&mut f, "t", 0, 2, &m);
            f.flush().unwrap();
        }
        let hfhs = open(&path).unwrap();
        // Corrupt the entry's offset to point beyond file
        let mut bad_entry = hfhs.entries[0].clone();
        bad_entry.data_offset = 99999;
        let mmap = mmap_file(&path);
        let err = hfhs.diagonal(&mmap, &bad_entry).unwrap_err();
        match err {
            QuantError::Truncated { artifact, .. } => assert_eq!(artifact, "HFHS"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn hblk_block_out_of_range() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&E8H1_MAGIC.to_le_bytes()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&256u32.to_le_bytes()).unwrap();
            f.write_all(&vec![0u8; HBLK_BLOCK_BYTES]).unwrap();
            f.flush().unwrap();
        }
        let e8 = open_hblk(&path).unwrap();
        let mmap = mmap_file(&path);
        let err = e8.hblk_block(&mmap, 1).unwrap_err();
        match err {
            QuantError::Malformed(_) => {},
            other => panic!("expected Malformed, got {other:?}"),
        }
    }
}
