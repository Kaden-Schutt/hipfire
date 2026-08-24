// SPDX-License-Identifier: Apache-2.0
//! Minimal reader for the retired **HFHS-v1** standalone Hessian sidecar,
//! recovered just to extract per-tensor **diagonals**. The diagonal of a
//! per-input-channel Hessian `H[j,j] = Σ_token x[j]²` is exactly AWQ's
//! `in_sum2[j]`, so this bridges an existing `*.hessian.bin` to the AWQ /
//! SmoothQuant per-channel activation statistic without re-running calibration.
//!
//! The dense Hessian package itself was migrated to HFQM (`.calib.hfq`,
//! `hessian_io.rs` / `hfqm.rs`), which retired the HFHS reader; this module reads only what
//! AWQ needs (the diagonal), not the full `[K,K]` payloads.
//!
//! HFHS-v1 layout:
//!   header 24B: magic "HFHS" | version u32(=1) | n_tensors u64 | reserved u64
//!   per record: name_len u32 | name | expert_idx u32 | k u32 | dtype u32
//!               | payload [k*k * (4 if F32 else 8)]   (dtype 1=F32, 2=F64)

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const HFHS_MAGIC: &[u8; 4] = b"HFHS";
const DTYPE_F32: u32 = 1;
const DTYPE_F64: u32 = 2;

/// Read every tensor's Hessian diagonal from an HFHS-v1 sidecar.
/// Returns `tensor_name → in_sum2[0..k]` (the diagonal, f32). Expert-indexed
/// records (`expert_idx > 0`) are skipped — AWQ keys by dense tensor name.
pub fn read_diagonals(path: &Path) -> std::io::Result<HashMap<String, Vec<f32>>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let inval = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());

    if mmap.len() < 24 || &mmap[0..4] != HFHS_MAGIC {
        return Err(inval("not an HFHS-v1 file (bad magic)"));
    }
    let version = LittleEndian::read_u32(&mmap[4..8]);
    if version != 1 {
        return Err(inval(&format!("unsupported HFHS version {version}")));
    }
    let n_tensors = LittleEndian::read_u64(&mmap[8..16]) as usize;

    let mut out: HashMap<String, Vec<f32>> = HashMap::with_capacity(n_tensors);
    let mut pos = 24usize;
    for _ in 0..n_tensors {
        if pos + 4 > mmap.len() {
            return Err(inval("truncated at name_len"));
        }
        let name_len = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
        pos += 4;
        if pos + name_len + 12 > mmap.len() {
            return Err(inval("truncated at name/header"));
        }
        let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
        pos += name_len;
        let expert_idx = LittleEndian::read_u32(&mmap[pos..pos + 4]);
        pos += 4;
        let k = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
        pos += 4;
        let dtype = LittleEndian::read_u32(&mmap[pos..pos + 4]);
        pos += 4;
        let esz = match dtype {
            DTYPE_F32 => 4usize,
            DTYPE_F64 => 8usize,
            d => return Err(inval(&format!("unknown HFHS dtype {d}"))),
        };
        let payload_bytes = k * k * esz;
        if pos + payload_bytes > mmap.len() {
            return Err(inval("truncated payload"));
        }
        // Read only the diagonal H[j,j].
        if expert_idx == 0 {
            let mut diag = vec![0.0f32; k];
            for j in 0..k {
                let off = pos + (j * k + j) * esz;
                diag[j] = match dtype {
                    DTYPE_F32 => LittleEndian::read_f32(&mmap[off..off + 4]),
                    _ => LittleEndian::read_f64(&mmap[off..off + 8]) as f32,
                };
            }
            out.insert(name, diag);
        }
        pos += payload_bytes;
    }
    Ok(out)
}

/// Lazy full-`[K,K]` reader for the HFHS-v1 sidecar — the off-diagonal payload
/// the LDLQ (GPTQ/OBS) error-feedback weight quant needs (the diagonal alone is
/// only enough for AWQ). Builds a name→(offset,k,dtype) index over the mmap at
/// open; `get_full(name)` materializes one tensor's `k*k` f32 matrix on demand
/// (the whole file is ~GBs, so we never hold all of them at once). Expert-indexed
/// records (`expert_idx > 0`) are indexed too but keyed by name only (last wins);
/// LDLQ is dense-only so callers look up dense tensor names.
pub struct HfhsFull {
    _mmap: Mmap,
    _file: File,
    index: HashMap<String, (usize, usize, u32)>, // name → (payload_offset, k, dtype)
}

impl HfhsFull {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let inval = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if mmap.len() < 24 || &mmap[0..4] != HFHS_MAGIC {
            return Err(inval("not an HFHS-v1 file (bad magic)"));
        }
        if LittleEndian::read_u32(&mmap[4..8]) != 1 {
            return Err(inval("unsupported HFHS version"));
        }
        let n_tensors = LittleEndian::read_u64(&mmap[8..16]) as usize;
        let mut index = HashMap::with_capacity(n_tensors);
        let mut pos = 24usize;
        for _ in 0..n_tensors {
            if pos + 4 > mmap.len() {
                return Err(inval("truncated at name_len"));
            }
            let name_len = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            if pos + name_len + 12 > mmap.len() {
                return Err(inval("truncated at name/header"));
            }
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
            pos += name_len;
            let expert_idx = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let k = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            let dtype = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let esz = match dtype {
                DTYPE_F32 => 4usize,
                DTYPE_F64 => 8usize,
                d => return Err(inval(&format!("unknown HFHS dtype {d}"))),
            };
            let payload_bytes = k * k * esz;
            if pos + payload_bytes > mmap.len() {
                return Err(inval("truncated payload"));
            }
            if expert_idx == 0 {
                index.insert(name, (pos, k, dtype));
            }
            pos += payload_bytes;
        }
        // keep file alive via _file
        Ok(Self {
            _mmap: mmap,
            _file: file,
            index,
        })
    }

    pub fn k_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).map(|&(_, k, _)| k)
    }

    /// Materialize the full row-major `k*k` Hessian (f32) for `name`, or `None`.
    pub fn get_full(&self, name: &str) -> Option<Vec<f32>> {
        let &(off, k, dtype) = self.index.get(name)?;
        let mut h = vec![0.0f32; k * k];
        let bytes = &self._mmap;
        match dtype {
            DTYPE_F32 => {
                for (i, hv) in h.iter_mut().enumerate() {
                    let o = off + i * 4;
                    *hv = LittleEndian::read_f32(&bytes[o..o + 4]);
                }
            }
            _ => {
                for (i, hv) in h.iter_mut().enumerate() {
                    let o = off + i * 8;
                    *hv = LittleEndian::read_f64(&bytes[o..o + 8]) as f32;
                }
            }
        }
        Some(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    fn write_record(buf: &mut Vec<u8>, name: &str, expert: u32, k: usize, diag_plus: &[f32]) {
        buf.write_u32::<LittleEndian>(name.len() as u32).unwrap();
        buf.extend_from_slice(name.as_bytes());
        buf.write_u32::<LittleEndian>(expert).unwrap();
        buf.write_u32::<LittleEndian>(k as u32).unwrap();
        buf.write_u32::<LittleEndian>(DTYPE_F32).unwrap();
        // full K*K payload, diagonal = diag_plus[j], off-diagonal = 0
        for i in 0..k {
            for j in 0..k {
                let v = if i == j { diag_plus[i] } else { 0.0f32 };
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
        }
    }

    #[test]
    fn synthetic_roundtrip_extracts_diagonal_and_skips_experts() {
        let mut buf = Vec::new();
        buf.extend_from_slice(HFHS_MAGIC);
        buf.write_u32::<LittleEndian>(1).unwrap(); // version
        buf.write_u64::<LittleEndian>(2).unwrap(); // n_tensors
        buf.write_u64::<LittleEndian>(0).unwrap(); // reserved
        write_record(&mut buf, "blk.0.attn_q", 0, 3, &[1.0, 4.0, 9.0]);
        write_record(&mut buf, "blk.0.ffn_gate", 1, 2, &[2.0, 5.0]); // expert>0 → skipped

        let mut tf = tempfile::NamedTempFile::new().unwrap();
        tf.write_all(&buf).unwrap();
        tf.flush().unwrap();

        let m = read_diagonals(tf.path()).unwrap();
        assert_eq!(m.len(), 1, "expert-indexed record should be skipped");
        assert_eq!(m.get("blk.0.attn_q").unwrap(), &vec![1.0, 4.0, 9.0]);
        assert!(!m.contains_key("blk.0.ffn_gate"));
    }
}
