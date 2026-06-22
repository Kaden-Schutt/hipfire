// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! `HFKREF` — the persistable KLD reference archive.
//!
//! A self-describing container: the [`RefMeta`] provenance header (so a consumer
//! can run [`crate::compat`] before trusting a score) followed by four
//! codec-encoded payload sections. The token stream is embedded, so the archive
//! is functionally portable without the original slice file.
//!
//! Layout (version 1):
//! ```text
//! magic   "HFKREF\0\0"   (8)
//! u32     version = 1
//! u32     meta_len ; RefMeta JSON (meta_len bytes)
//! repeat per section:
//!   u32   name_len ; name (utf8)
//!   u32   codec_len ; BlobCodec JSON
//!   u64   data_len ; data
//! ```
//! Sections, in order: `tokens` [n_chunk·n_ctx] u32, `top_indices`
//! [n_chunk·scored·top_k] u32 (bit-packed), `top_log_probs` same shape f32,
//! `residual_mass` [n_chunk·scored] f32.

use crate::codec::{self, BlobCodec};
use crate::meta::RefMeta;
use crate::refblock::RefBlock;

const MAGIC: &[u8; 8] = b"HFKREF\0\0";
const VERSION: u32 = 1;

/// In-memory KLD reference: provenance + the flattened payload arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct RefArchive {
    pub meta: RefMeta,
    /// [n_chunk · n_ctx] token ids (the embedded tokenized slice).
    pub tokens: Vec<u32>,
    /// [n_chunk · scored_per_chunk · top_k] reference top-K token ids.
    pub top_indices: Vec<u32>,
    /// [n_chunk · scored_per_chunk · top_k] matching log-probabilities.
    pub top_log_probs: Vec<f32>,
    /// [n_chunk · scored_per_chunk] residual mass per scored position.
    pub residual_mass: Vec<f32>,
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}
fn bytes_to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_section(out: &mut Vec<u8>, name: &str, codec: &BlobCodec, data: &[u8]) {
    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    let codec_json = serde_json::to_vec(codec).expect("codec serialize");
    out.extend_from_slice(&(codec_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&codec_json);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
}

fn read_u32(b: &[u8], off: &mut usize) -> Result<u32, String> {
    if *off + 4 > b.len() {
        return Err("HFKREF truncated (u32)".into());
    }
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}
fn read_u64(b: &[u8], off: &mut usize) -> Result<u64, String> {
    if *off + 8 > b.len() {
        return Err("HFKREF truncated (u64)".into());
    }
    let v = u64::from_le_bytes(b[*off..*off + 8].try_into().unwrap());
    *off += 8;
    Ok(v)
}
fn read_bytes<'a>(b: &'a [u8], off: &mut usize, len: usize) -> Result<&'a [u8], String> {
    if *off + len > b.len() {
        return Err("HFKREF truncated (bytes)".into());
    }
    let s = &b[*off..*off + len];
    *off += len;
    Ok(s)
}

impl RefArchive {
    /// Scored positions per chunk (derived from the metadata).
    pub fn scored_per_chunk(&self) -> usize {
        self.meta.scored_per_chunk
    }

    /// View of the reference block for chunk `c`, scored index `j`.
    pub fn block(&self, c: usize, j: usize) -> RefBlock<'_> {
        let k = self.meta.top_k;
        let flat = (c * self.meta.scored_per_chunk + j) * k;
        let resid = c * self.meta.scored_per_chunk + j;
        RefBlock {
            top_indices: &self.top_indices[flat..flat + k],
            top_log_probs: &self.top_log_probs[flat..flat + k],
            residual_mass: self.residual_mass[resid],
        }
    }

    /// Serialize to the HFKREF byte layout. Token ids and `top_indices` are
    /// bit-packed at `ceil(log2(n_vocab))` bits (lossless); log-probs/residual
    /// are raw f32. The chosen codecs are recorded in `meta.payload_codecs`.
    pub fn encode(&self) -> Vec<u8> {
        let bits = codec::bits_for_vocab(self.meta.n_vocab);
        let idx_codec = BlobCodec::BitpackedIdx { bits };

        let mut meta = self.meta.clone();
        meta.payload_codecs.clear();
        meta.payload_codecs
            .insert("tokens".into(), format!("bitpacked-idx:{bits}"));
        meta.payload_codecs
            .insert("top_indices".into(), format!("bitpacked-idx:{bits}"));
        meta.payload_codecs
            .insert("top_log_probs".into(), "raw-f32".into());
        meta.payload_codecs
            .insert("residual_mass".into(), "raw-f32".into());

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let meta_json = serde_json::to_vec(&meta).expect("meta serialize");
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta_json);

        let tok_packed = codec::bitpack(&self.tokens, bits).expect("token ids fit n_vocab bits");
        write_section(&mut out, "tokens", &idx_codec, &tok_packed);
        let idx_packed = codec::bitpack(&self.top_indices, bits).expect("top ids fit n_vocab bits");
        write_section(&mut out, "top_indices", &idx_codec, &idx_packed);
        write_section(
            &mut out,
            "top_log_probs",
            &BlobCodec::RawF32,
            &f32s_to_bytes(&self.top_log_probs),
        );
        write_section(
            &mut out,
            "residual_mass",
            &BlobCodec::RawF32,
            &f32s_to_bytes(&self.residual_mass),
        );
        out
    }

    /// Parse the HFKREF byte layout.
    pub fn decode(buf: &[u8]) -> Result<RefArchive, String> {
        let mut off = 0usize;
        let magic = read_bytes(buf, &mut off, 8)?;
        if magic != MAGIC {
            return Err("HFKREF bad magic".into());
        }
        let version = read_u32(buf, &mut off)?;
        if version != VERSION {
            return Err(format!("HFKREF unsupported version {version}"));
        }
        let meta_len = read_u32(buf, &mut off)? as usize;
        let meta_bytes = read_bytes(buf, &mut off, meta_len)?;
        let meta: RefMeta =
            serde_json::from_slice(meta_bytes).map_err(|e| format!("HFKREF meta json: {e}"))?;

        let mut tokens = None;
        let mut top_indices = None;
        let mut top_log_probs = None;
        let mut residual_mass = None;

        while off < buf.len() {
            let name_len = read_u32(buf, &mut off)? as usize;
            let name = String::from_utf8(read_bytes(buf, &mut off, name_len)?.to_vec())
                .map_err(|_| "HFKREF section name utf8".to_string())?;
            let codec_len = read_u32(buf, &mut off)? as usize;
            let blob_codec: BlobCodec =
                serde_json::from_slice(read_bytes(buf, &mut off, codec_len)?)
                    .map_err(|e| format!("HFKREF codec json: {e}"))?;
            let data_len = read_u64(buf, &mut off)? as usize;
            let data = read_bytes(buf, &mut off, data_len)?;

            let decode_idx = |count: usize| -> Result<Vec<u32>, String> {
                match &blob_codec {
                    BlobCodec::BitpackedIdx { bits } => codec::bitunpack(data, count, *bits),
                    BlobCodec::RawU32 => Ok(bytes_to_u32s(data)),
                    other => Err(format!("HFKREF {name}: unexpected codec {other:?}")),
                }
            };
            match name.as_str() {
                "tokens" => {
                    let count = meta.n_chunk * meta.n_ctx;
                    tokens = Some(decode_idx(count)?);
                }
                "top_indices" => {
                    let count = meta.n_chunk * meta.scored_per_chunk * meta.top_k;
                    top_indices = Some(decode_idx(count)?);
                }
                "top_log_probs" => top_log_probs = Some(bytes_to_f32s(data)),
                "residual_mass" => residual_mass = Some(bytes_to_f32s(data)),
                _ => {} // forward-compat: ignore unknown sections
            }
        }

        Ok(RefArchive {
            meta,
            tokens: tokens.ok_or("HFKREF missing tokens")?,
            top_indices: top_indices.ok_or("HFKREF missing top_indices")?,
            top_log_probs: top_log_probs.ok_or("HFKREF missing top_log_probs")?,
            residual_mass: residual_mass.ok_or("HFKREF missing residual_mass")?,
        })
    }

    pub fn write_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        std::fs::write(path, self.encode())
    }

    pub fn read_file(path: &std::path::Path) -> std::io::Result<RefArchive> {
        let bytes = std::fs::read(path)?;
        RefArchive::decode(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KldConfig;
    use crate::math::{score_position, top_k_log_softmax};
    use std::collections::BTreeMap;

    fn tiny_archive() -> RefArchive {
        // 2 chunks, n_ctx=8 → scoring_start=4, scored_per_chunk=3, top_k=4, vocab=64.
        let (n_chunk, n_ctx, scored, top_k, vocab) = (2usize, 8usize, 3usize, 4usize, 64usize);
        let tokens: Vec<u32> = (0..(n_chunk * n_ctx) as u32)
            .map(|i| (i * 7) % vocab as u32)
            .collect();
        // Build a deterministic per-position reduction from synthetic logits.
        let mut top_indices = Vec::new();
        let mut top_log_probs = Vec::new();
        let mut residual_mass = Vec::new();
        for c in 0..n_chunk {
            for j in 0..scored {
                let logits: Vec<f32> = (0..vocab)
                    .map(|v| ((v + c * 3 + j) % 17) as f32 * 0.2)
                    .collect();
                let r = top_k_log_softmax(&logits, top_k);
                top_indices.extend_from_slice(&r.indices);
                top_log_probs.extend_from_slice(&r.log_probs);
                residual_mass.push(r.residual_mass);
            }
        }
        let meta = RefMeta {
            schema: 2,
            base_model_id: "tiny".into(),
            source_model_sha256: "x".into(),
            tokenizer_sha256: None,
            arch_id: 5,
            n_vocab: vocab,
            n_ctx,
            n_chunk,
            scored_per_chunk: scored,
            scoring_start: n_ctx / 2,
            top_k,
            total_scored: n_chunk * scored,
            slice_path: "s".into(),
            slice_md5: "m".into(),
            config: KldConfig::default(),
            producer: Default::default(),
            payload_codecs: BTreeMap::new(),
            content_sha256: None,
        };
        RefArchive {
            meta,
            tokens,
            top_indices,
            top_log_probs,
            residual_mass,
        }
    }

    #[test]
    fn archive_round_trips_lossless() {
        let a = tiny_archive();
        let bytes = a.encode();
        let b = RefArchive::decode(&bytes).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.top_indices, b.top_indices);
        assert_eq!(a.top_log_probs, b.top_log_probs);
        assert_eq!(a.residual_mass, b.residual_mass);
        assert_eq!(a.meta.n_vocab, b.meta.n_vocab);
        // codec tags recorded
        assert!(b
            .meta
            .payload_codecs
            .get("top_indices")
            .unwrap()
            .starts_with("bitpacked-idx"));
    }

    #[test]
    fn block_accessor_scores_self_to_zero() {
        // A block read back from the archive, scored against the SAME logits it
        // was built from, gives ~0 — the persisted-ref self-consistency at the
        // container level.
        let vocab = 64usize;
        let logits: Vec<f32> = (0..vocab).map(|v| (v % 17) as f32 * 0.2).collect();
        let r = top_k_log_softmax(&logits, 4);
        let mut a = tiny_archive();
        // overwrite chunk0/j0 with this known reduction
        a.top_indices[0..4].copy_from_slice(&r.indices);
        a.top_log_probs[0..4].copy_from_slice(&r.log_probs);
        a.residual_mass[0] = r.residual_mass;
        let bytes = a.encode();
        let b = RefArchive::decode(&bytes).unwrap();
        let s = score_position(&b.block(0, 0), &logits, 1);
        assert!(s.kld < 1e-6, "persisted-ref self KLD ~0, got {}", s.kld);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(RefArchive::decode(b"NOPE____").is_err());
    }
}
