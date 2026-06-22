// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! `HFKSEQ` — the per-sequence KLD result file consumed by `kld_reduce.py`.
//!
//! Layout (version 2):
//! ```text
//! magic  "HFKSEQ\0\0"   (8 bytes)
//! u32    version = 2
//! u32    n_chunk
//! u32    reserved = 0
//! per chunk: f64 mean_kld, f64 p99_kld, f64 mean_nll   (LE)
//! ```

const MAGIC: &[u8; 8] = b"HFKSEQ\0\0";
const VERSION: u32 = 2;

/// One scored chunk's aggregate KLD statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkResult {
    pub mean_kld: f64,
    pub p99_kld: f64,
    pub mean_nll: f64,
}

/// Serialize chunk results into the HFKSEQ v2 byte layout.
pub fn encode(chunks: &[ChunkResult]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + chunks.len() * 24);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for c in chunks {
        out.extend_from_slice(&c.mean_kld.to_le_bytes());
        out.extend_from_slice(&c.p99_kld.to_le_bytes());
        out.extend_from_slice(&c.mean_nll.to_le_bytes());
    }
    out
}

/// Parse the HFKSEQ v2 byte layout.
pub fn decode(buf: &[u8]) -> Result<Vec<ChunkResult>, String> {
    if buf.len() < 20 {
        return Err(format!("HFKSEQ too short: {}", buf.len()));
    }
    if &buf[0..8] != MAGIC {
        return Err("HFKSEQ bad magic".to_string());
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(format!("HFKSEQ unsupported version {version}"));
    }
    let n_chunk = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    // bytes[16..20] reserved
    let need = 20 + n_chunk * 24;
    if buf.len() < need {
        return Err(format!("HFKSEQ truncated: {} < {need}", buf.len()));
    }
    let mut out = Vec::with_capacity(n_chunk);
    for i in 0..n_chunk {
        let o = 20 + i * 24;
        out.push(ChunkResult {
            mean_kld: f64::from_le_bytes(buf[o..o + 8].try_into().unwrap()),
            p99_kld: f64::from_le_bytes(buf[o + 8..o + 16].try_into().unwrap()),
            mean_nll: f64::from_le_bytes(buf[o + 16..o + 24].try_into().unwrap()),
        });
    }
    Ok(out)
}

/// Write HFKSEQ to a path (creates parent dirs).
pub fn write_file(path: &std::path::Path, chunks: &[ChunkResult]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, encode(chunks))
}

/// Read HFKSEQ from a path.
pub fn read_file(path: &std::path::Path) -> std::io::Result<Vec<ChunkResult>> {
    let bytes = std::fs::read(path)?;
    decode(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let chunks = vec![
            ChunkResult {
                mean_kld: 0.0173,
                p99_kld: 1.767,
                mean_nll: 2.21,
            },
            ChunkResult {
                mean_kld: 2.853151,
                p99_kld: 9.4,
                mean_nll: 3.108899,
            },
        ];
        let bytes = encode(&chunks);
        assert_eq!(&bytes[0..8], MAGIC);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, chunks);
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(decode(b"NOPE\0\0\0\0\0\0\0\0\0\0\0\0").is_err());
        let mut bytes = encode(&[ChunkResult {
            mean_kld: 1.0,
            p99_kld: 2.0,
            mean_nll: 3.0,
        }]);
        bytes.truncate(25);
        assert!(decode(&bytes).is_err());
    }
}
