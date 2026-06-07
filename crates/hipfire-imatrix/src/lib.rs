// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HFIM — HipFire IMatrix sidecar: native (non-GGUF) per-tensor activation
//! importance for AWQ scale computation. The last calibration artifact that
//! was still llama.cpp/GGUF-derived; HFIM makes it hipfire-native, removing the
//! cross-engine confound (llama.cpp tokenizer disagrees with hipfire on ~46% of
//! token positions, plus the DeltaNet-port forward gap).
//!
//! This is the **single source of truth** for the HFIM byte format. It is
//! depended on by both sides:
//!   - the native collector (writer) — `hipfire-runtime`'s
//!     `collect_imatrix_native` example runs hipfire's own f32 oracle forward
//!     and tokenizer over the calibration corpus, accumulating `Σ act²` at the
//!     input of every linear, then calls [`Imatrix::write_to_file`].
//!   - the quantizer (reader) — `hipfire-quantize` calls [`Imatrix::open`] and
//!     feeds [`ImatrixEntry::rms_act`] into the AWQ scale path.
//!
//! Sibling of `hipfire-quantize/src/hessian_io.rs` (HFHS, the GPTQ Hessian).
//! Mirrors the `HF**` container convention used by `.hfq` (HFQM) and `.hfhs`:
//!
//! ```text
//! Header (20 bytes, little-endian):
//!   [0..4]   magic   = b"HFIM"
//!   [4..8]   version = u32 (=1)
//!   [8..12]  n_tensors = u32
//!   [12..20] n_tokens  = u64   (total calib tokens accumulated)
//! Per-tensor record (repeated n_tensors times):
//!   u16  name_len
//!   utf8 name           (HIPFIRE-native tensor name — no GGUF↔safetensors remap)
//!   u32  in_dim
//!   u64  count          (tokens that contributed to THIS tensor)
//!   f32  sum_sq[in_dim]  (Σ act² per input channel)
//! ```
//!
//! Consumed by `hipfire-quantize` AWQ: `rms_act[j] = sqrt(sum_sq[j] / count)`,
//! `s[j] = rms_act[j]^α`.

use byteorder::{ByteOrder, LittleEndian};
use std::collections::HashMap;
use std::path::Path;

pub const HFIM_MAGIC: &[u8; 4] = b"HFIM";
pub const HFIM_VERSION: u32 = 1;
const HEADER_SIZE: usize = 20;

#[derive(Debug)]
pub enum ImatrixError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    Truncated { needed: usize, have: usize },
}

impl std::fmt::Display for ImatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImatrixError::Io(e) => write!(f, "I/O error: {e}"),
            ImatrixError::InvalidMagic(m) => {
                write!(f, "invalid HFIM magic: got {m:?}, expected {HFIM_MAGIC:?}")
            }
            ImatrixError::UnsupportedVersion(v) => {
                write!(f, "unsupported HFIM version {v}, this build understands v{HFIM_VERSION}")
            }
            ImatrixError::Truncated { needed, have } => {
                write!(f, "HFIM truncated: needed {needed} bytes, file is {have}")
            }
        }
    }
}

impl std::error::Error for ImatrixError {}

impl From<std::io::Error> for ImatrixError {
    fn from(e: std::io::Error) -> Self {
        ImatrixError::Io(e)
    }
}

/// Per-tensor activation importance.
#[derive(Debug, Clone)]
pub struct ImatrixEntry {
    pub in_dim: u32,
    pub count: u64,
    pub sum_sq: Vec<f32>,
}

impl ImatrixEntry {
    /// `rms_act[j] = sqrt(Σ act²[j] / count)` — the per-input-channel RMS the
    /// AWQ scale `s[j] = rms_act[j]^α` is computed from. Returns f64 for the
    /// downstream pow. `count == 0` yields all-zero (caller treats as unit).
    pub fn rms_act(&self) -> Vec<f64> {
        let n = self.count.max(1) as f64;
        self.sum_sq.iter().map(|&s| ((s as f64) / n).max(0.0).sqrt()).collect()
    }
}

/// A loaded HFIM sidecar, keyed by hipfire-native tensor name.
#[derive(Debug, Default)]
pub struct Imatrix {
    pub n_tokens: u64,
    pub tensors: HashMap<String, ImatrixEntry>,
}

impl Imatrix {
    /// Read an entire HFIM file (small — per-channel vectors, ~tens of MB for a
    /// 27B). Not mmap'd; the whole thing fits comfortably in RAM.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ImatrixError> {
        let buf = std::fs::read(path)?;
        Self::from_bytes(&buf)
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, ImatrixError> {
        if buf.len() < HEADER_SIZE {
            return Err(ImatrixError::Truncated { needed: HEADER_SIZE, have: buf.len() });
        }
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if &magic != HFIM_MAGIC {
            return Err(ImatrixError::InvalidMagic(magic));
        }
        let version = LittleEndian::read_u32(&buf[4..8]);
        if version != HFIM_VERSION {
            return Err(ImatrixError::UnsupportedVersion(version));
        }
        let n_tensors = LittleEndian::read_u32(&buf[8..12]) as usize;
        let n_tokens = LittleEndian::read_u64(&buf[12..20]);

        let mut tensors = HashMap::with_capacity(n_tensors);
        let mut pos = HEADER_SIZE;
        let need = |pos: usize, extra: usize, have: usize| -> Result<(), ImatrixError> {
            if pos + extra > have {
                Err(ImatrixError::Truncated { needed: pos + extra, have })
            } else {
                Ok(())
            }
        };
        for _ in 0..n_tensors {
            need(pos, 2, buf.len())?;
            let name_len = LittleEndian::read_u16(&buf[pos..pos + 2]) as usize;
            pos += 2;
            need(pos, name_len, buf.len())?;
            let name = String::from_utf8_lossy(&buf[pos..pos + name_len]).to_string();
            pos += name_len;
            need(pos, 12, buf.len())?;
            let in_dim = LittleEndian::read_u32(&buf[pos..pos + 4]) as usize;
            pos += 4;
            let count = LittleEndian::read_u64(&buf[pos..pos + 8]);
            pos += 8;
            need(pos, in_dim * 4, buf.len())?;
            let mut sum_sq = vec![0.0f32; in_dim];
            LittleEndian::read_f32_into(&buf[pos..pos + in_dim * 4], &mut sum_sq);
            pos += in_dim * 4;
            tensors.insert(
                name,
                ImatrixEntry { in_dim: in_dim as u32, count, sum_sq },
            );
        }
        Ok(Imatrix { n_tokens, tensors })
    }

    /// Serialize to the HFIM byte layout (used by the native collector's writer
    /// and the round-trip test).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(HFIM_MAGIC);
        out.extend_from_slice(&HFIM_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.n_tokens.to_le_bytes());
        // Deterministic order for reproducible files.
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        for name in names {
            let e = &self.tensors[name];
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&e.in_dim.to_le_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            for &s in &e.sum_sq {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    /// Write the HFIM to `path` (collector convenience).
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    pub fn get(&self, name: &str) -> Option<&ImatrixEntry> {
        self.tensors.get(name)
    }

    /// Build an `Imatrix` from a raw accumulator: a map of
    /// `tensor_name -> (Σ act² as f64 per channel, token count)`. The collector
    /// keeps f64 sums on the host for accumulation precision; HFIM stores f32
    /// (the AWQ pow is well-conditioned at f32). `n_tokens` is the total number
    /// of calibration tokens processed (distinct from per-tensor `count`, which
    /// equals the number of forward applications of that specific tensor).
    pub fn from_accum(
        accum: &HashMap<String, (Vec<f64>, u64)>,
        n_tokens: u64,
    ) -> Self {
        let mut tensors = HashMap::with_capacity(accum.len());
        for (name, (sum_sq_f64, count)) in accum {
            let sum_sq: Vec<f32> = sum_sq_f64.iter().map(|&v| v as f32).collect();
            tensors.insert(
                name.clone(),
                ImatrixEntry { in_dim: sum_sq.len() as u32, count: *count, sum_sq },
            );
        }
        Imatrix { n_tokens, tensors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hfim_round_trip() {
        let mut im = Imatrix { n_tokens: 1000, tensors: HashMap::new() };
        im.tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            ImatrixEntry { in_dim: 4, count: 1000, sum_sq: vec![1.0, 4.0, 9.0, 16.0] },
        );
        let bytes = im.to_bytes();
        let back = Imatrix::from_bytes(&bytes).unwrap();
        assert_eq!(back.n_tokens, 1000);
        let e = back.get("model.layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(e.in_dim, 4);
        assert_eq!(e.sum_sq, vec![1.0, 4.0, 9.0, 16.0]);
        // rms_act[j] = sqrt(sumsq/count): sqrt(1/1000)=0.0316, sqrt(16/1000)=0.1265
        let rms = e.rms_act();
        assert!((rms[0] - (1.0f64 / 1000.0).sqrt()).abs() < 1e-9);
        assert!((rms[3] - (16.0f64 / 1000.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_magic() {
        let bad = vec![0u8; 20];
        assert!(matches!(Imatrix::from_bytes(&bad), Err(ImatrixError::InvalidMagic(_))));
    }

    #[test]
    fn from_accum_builds_entries() {
        let mut accum: HashMap<String, (Vec<f64>, u64)> = HashMap::new();
        accum.insert("w.a".to_string(), (vec![2.0, 8.0], 4));
        let im = Imatrix::from_accum(&accum, 4);
        let e = im.get("w.a").unwrap();
        assert_eq!(e.count, 4);
        assert_eq!(e.sum_sq, vec![2.0f32, 8.0]);
        // rms = sqrt(8/4) = sqrt(2)
        assert!((e.rms_act()[1] - 2.0f64.sqrt()).abs() < 1e-9);
    }
}
