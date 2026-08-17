//! Artifact formats — one canonical parser each.
//!
//! Every reader here replaces a family of ad-hoc parsers scattered across
//! `crates/` and `scripts/`. Before this module, 29 files independently
//! walked the HFQ header; the tree had already deleted redundant parsers by
//! hand (`3dfd1b3f5`). The rule for this module is simple: if a format is
//! read anywhere in the repo, it is read *here*, and everything else calls in.
//!
//! The whole module is GPU-free and CPU-only by construction so that CI, the
//! Python bindings, and offline tooling can link it without a ROCm toolchain.

use crate::{ArtifactId, Result};
use std::path::Path;

pub mod hfhs;
pub mod hfq;
pub mod imatrix;
pub mod kldref;
pub mod kldseq;

/// Quantization type tags as written into the HFQ tensor index.
///
/// Values are wire constants — they are persisted in every `.hfq` on disk and
/// must never be renumbered. Mirrors the encoder's `QuantType` discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    Hfq4G256 = 6,
    Hfq4G128 = 7,
    Hfq6G256 = 8,
    Hfq2G256 = 9,
    Hfq2G128 = 10,
    Hfq3G256 = 11,
    Hfq3G128 = 12,
    Mq4G256 = 13,
    Mq8G256 = 14,
    Mq6G256 = 15,
    Bf16 = 16,
    Mfp4G32 = 17,
    Mfp4E8 = 18,
    Mq2G256Lloyd = 19,
    Mq3G256Lloyd = 20,
}

impl QuantType {
    /// Wire tag → enum. `None` for tags this build does not know, which is a
    /// forward-compatibility case (newer artifact, older reader), not a bug.
    pub fn from_tag(tag: u8) -> Option<Self> {
        use QuantType::*;
        Some(match tag {
            0 => Q4F16G64, 1 => F16, 2 => F32, 3 => Q8F16, 4 => Q4K, 5 => Q8HFQ,
            6 => Hfq4G256, 7 => Hfq4G128, 8 => Hfq6G256, 9 => Hfq2G256,
            10 => Hfq2G128, 11 => Hfq3G256, 12 => Hfq3G128, 13 => Mq4G256,
            14 => Mq8G256, 15 => Mq6G256, 16 => Bf16, 17 => Mfp4G32,
            18 => Mfp4E8, 19 => Mq2G256Lloyd, 20 => Mq3G256Lloyd,
            _ => return None,
        })
    }
}

/// One entry of an HFQ tensor index.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEntry {
    pub name: String,
    /// Raw wire tag, retained even when [`QuantType::from_tag`] does not know
    /// it, so an unknown dtype is reported rather than silently dropped.
    pub quant_tag: u8,
    pub quant_type: Option<QuantType>,
    pub shape: Vec<u32>,
    pub group_size: u32,
    /// Byte offset of this tensor's payload from the start of the file.
    pub data_offset: u64,
    pub data_size: u64,
}

impl TensorEntry {
    pub fn elements(&self) -> u64 {
        self.shape.iter().map(|&d| d as u64).product()
    }

    /// Effective bits per weight for this tensor's payload.
    ///
    /// This is the number that makes cross-quantizer comparison honest: a
    /// 4.25 bpw artifact and a 5.29 bpw artifact are not the same operating
    /// point regardless of what they are named.
    pub fn bits_per_weight(&self) -> f64 {
        let n = self.elements();
        if n == 0 { return 0.0; }
        (self.data_size as f64 * 8.0) / n as f64
    }
}

/// SHA-256 an artifact and capture its identity.
///
/// Streamed so a 53 GB teacher does not have to be resident.
pub fn identify(path: impl AsRef<Path>) -> Result<ArtifactId> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let path = path.as_ref();
    let mut file = std::fs::File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(ArtifactId {
        path: path.display().to_string(),
        sha256: format!("{:x}", hasher.finalize()),
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpw_matches_measured_artifacts() {
        // qwen3.8-27b lm_head [248320, 5120] at MQ4G256 occupied 675,430,400 B
        // in the .mq4r arm and 1,350,860,800 B at Q8F16 in the .mq4 trunk.
        let mq4 = TensorEntry {
            name: "lm_head.weight".into(),
            quant_tag: 13,
            quant_type: Some(QuantType::Mq4G256),
            shape: vec![248_320, 5_120],
            group_size: 256,
            data_offset: 0,
            data_size: 675_430_400,
        };
        assert_eq!(mq4.elements(), 1_271_398_400);
        assert!((mq4.bits_per_weight() - 4.25).abs() < 1e-9);

        let q8 = TensorEntry { data_size: 1_350_860_800, ..mq4.clone() };
        assert!((q8.bits_per_weight() - 8.5).abs() < 1e-9);
    }

    #[test]
    fn unknown_quant_tag_is_reported_not_dropped() {
        assert_eq!(QuantType::from_tag(13), Some(QuantType::Mq4G256));
        assert_eq!(QuantType::from_tag(200), None);
    }
}
