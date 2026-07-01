// SPDX-License-Identifier: Apache-2.0
//! Binary `.lora.hfq` container for [`hipfire_steer::lora::LoraAdapter`].
//!
//! The adapter's rank-1 residual deltas serialize as a canonical HFQM package:
//! two f32 tensors `a` / `b` (each `[n_deltas, hidden]`, one row per delta) plus a
//! JSON provenance blob in the container metadata. Much smaller than the JSON
//! container (raw f32 vs stringified) and readable by the standard hfq tooling.
//!
//! This lives in its own crate rather than in `hipfire-steer` because the hfq
//! writer is in `hipfire-runtime`, which depends on the arch crates that depend on
//! `hipfire-steer` — so `steer -> runtime` would be a cycle. See
//! `docs/plans/2026-06-30-abliteration-lora.md`.

use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

use hipfire_runtime::hfq::{
    write_hfqm_package_mem, HfqFile, HfqMemTensor, HFQM_ARCH_NON_WEIGHT_PACKAGE,
};
use hipfire_steer::lora::{self, LoraAdapter, LoraDelta, LoraMeta, LoraTarget};

/// hfq `quant_type` byte for a plain f32 tensor.
const HFQ_QT_F32: u8 = 2;
const CONTAINER: &str = "hipfire-lora";
const CONTAINER_VERSION: u32 = 1;

/// Provenance stored in the HFQM metadata blob. The per-delta directions live in
/// the `a` / `b` tensors; `layers[i]` is delta `i`'s target block.
#[derive(Serialize, Deserialize)]
struct LoraHfqMeta {
    container: String,
    version: u32,
    id: String,
    scale: f32,
    meta: LoraMeta,
    layers: Vec<usize>,
}

fn delta_layer(d: &LoraDelta) -> usize {
    match d.target {
        LoraTarget::Residual { layer } => layer,
    }
}

/// Write `adapter` as a binary `.lora.hfq` container. Rank-1 residual deltas only
/// (the ablate form); richer adapters should use the JSON container.
pub fn write_lora_hfq(path: &Path, adapter: &LoraAdapter) -> Result<(), String> {
    let hidden = adapter.meta.hidden;
    let n = adapter.deltas.len();
    let mut layers = Vec::with_capacity(n);
    let mut a_data = Vec::with_capacity(n * hidden * 4);
    let mut b_data = Vec::with_capacity(n * hidden * 4);
    for d in &adapter.deltas {
        if d.a.len() != 1 || d.b.len() != 1 {
            return Err(format!(
                "lora-hfq: layer {} delta is rank {} — the binary container is rank-1 only",
                delta_layer(d),
                d.a.len()
            ));
        }
        if d.a[0].len() != hidden || d.b[0].len() != hidden {
            return Err(format!(
                "lora-hfq: layer {} delta width != meta.hidden ({hidden})",
                delta_layer(d)
            ));
        }
        layers.push(delta_layer(d));
        a_data.extend(d.a[0].iter().flat_map(|x| x.to_le_bytes()));
        b_data.extend(d.b[0].iter().flat_map(|x| x.to_le_bytes()));
    }

    let meta = LoraHfqMeta {
        container: CONTAINER.to_string(),
        version: CONTAINER_VERSION,
        id: adapter.id.clone(),
        scale: adapter.scale,
        meta: adapter.meta.clone(),
        layers,
    };
    let metadata_json =
        serde_json::to_string(&meta).map_err(|e| format!("lora-hfq: serialize meta: {e}"))?;
    let shape = vec![n as u32, hidden as u32];
    let tensors = vec![
        HfqMemTensor {
            name: "a".to_string(),
            quant_type: HFQ_QT_F32,
            shape: shape.clone(),
            group_size: 0,
            data: a_data,
        },
        HfqMemTensor {
            name: "b".to_string(),
            quant_type: HFQ_QT_F32,
            shape,
            group_size: 0,
            data: b_data,
        },
    ];
    write_hfqm_package_mem(path, HFQM_ARCH_NON_WEIGHT_PACKAGE, &metadata_json, &tensors)
        .map_err(|e| format!("lora-hfq: write {}: {e}", path.display()))
}

/// Read a `.lora.hfq` container written by [`write_lora_hfq`].
pub fn read_lora_hfq(path: &Path) -> Result<LoraAdapter, String> {
    let f = HfqFile::open(path).map_err(|e| format!("lora-hfq: open {}: {e}", path.display()))?;
    adapter_from_hfq(&f, &path.display().to_string())
}

/// Reconstruct a [`LoraAdapter`] from an already-open HFQM container (top-level or
/// an embedded section via [`HfqFile::open_at_offset`]). `src` labels errors.
fn adapter_from_hfq(f: &HfqFile, src: &str) -> Result<LoraAdapter, String> {
    let meta: LoraHfqMeta = serde_json::from_str(&f.metadata_json)
        .map_err(|e| format!("lora-hfq: parse metadata of {src}: {e}"))?;
    if meta.container != CONTAINER {
        return Err(format!(
            "lora-hfq: {src} is not a lora container (container {:?})",
            meta.container
        ));
    }
    if meta.version != CONTAINER_VERSION {
        return Err(format!(
            "lora-hfq: {src} version {} unsupported (expected {CONTAINER_VERSION})",
            meta.version
        ));
    }
    let hidden = meta.meta.hidden;
    let a = rows(f, "a", hidden)?;
    let b = rows(f, "b", hidden)?;
    if a.len() != meta.layers.len() || b.len() != meta.layers.len() {
        return Err(format!(
            "lora-hfq: {src} row count ({}/{}) != {} layers",
            a.len(),
            b.len(),
            meta.layers.len()
        ));
    }
    let deltas = meta
        .layers
        .iter()
        .enumerate()
        .map(|(i, &layer)| LoraDelta {
            target: LoraTarget::Residual { layer },
            a: vec![a[i].clone()],
            b: vec![b[i].clone()],
        })
        .collect();
    Ok(LoraAdapter {
        id: meta.id,
        scale: meta.scale,
        deltas,
        meta: meta.meta,
    })
}

/// Read a LoRA adapter from either the binary (`HFQM`) or JSON container,
/// detected by the file's magic.
pub fn read_lora_any(path: &Path) -> Result<LoraAdapter, String> {
    let mut head = [0u8; 4];
    let n = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|e| format!("lora: read {}: {e}", path.display()))?;
    if n == 4 && &head == b"HFQM" {
        read_lora_hfq(path)
    } else {
        lora::read_adapter(path)
    }
}

// ── Bundle-into-one-hfq (the `--merge-lora` artifact) ────────────────────────

/// Trailer magic for a LoRA adapter bundled onto the end of a model `.hfq`.
/// Mirrors the MTP bundle (`HFBNDMTP`): the last 16 bytes are this 8-byte magic
/// followed by a little-endian `u64` byte offset of the embedded adapter HFQM
/// section within the file.
pub const LORA_BUNDLE_MAGIC: &[u8; 8] = b"HFBNDLRA";
const LORA_BUNDLE_TRAILER_LEN: u64 = 16;

/// Produce a self-contained model at `out`: the base model `.hfq` with `adapter`
/// appended as a second HFQM section + a bundle trailer. The daemon auto-applies
/// the adapter when such a model loads (see [`read_bundled_lora`]). The base is
/// copied byte-for-byte, so the trunk's own load path is unchanged.
pub fn merge_lora_into_model(base: &Path, adapter: &LoraAdapter, out: &Path) -> Result<(), String> {
    use std::io::{Seek, SeekFrom, Write};

    // Serialize the adapter as a standalone HFQM into a temp file, then read bytes.
    let tmp = std::env::temp_dir().join(format!(
        "hipfire-lora-merge-{}-{}.lora.hfq",
        std::process::id(),
        adapter.deltas.len()
    ));
    write_lora_hfq(&tmp, adapter)?;
    let adapter_bytes =
        std::fs::read(&tmp).map_err(|e| format!("lora-hfq: read temp section: {e}"))?;
    let _ = std::fs::remove_file(&tmp);

    std::fs::copy(base, out).map_err(|e| {
        format!(
            "lora-hfq: copy base {} -> {}: {e}",
            base.display(),
            out.display()
        )
    })?;
    let base_len = std::fs::metadata(out)
        .map_err(|e| format!("lora-hfq: stat {}: {e}", out.display()))?
        .len();

    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(out)
        .map_err(|e| format!("lora-hfq: open {} for append: {e}", out.display()))?;
    f.seek(SeekFrom::End(0))
        .map_err(|e| format!("lora-hfq: seek {}: {e}", out.display()))?;
    f.write_all(&adapter_bytes)
        .map_err(|e| format!("lora-hfq: append adapter section: {e}"))?;
    let mut trailer = [0u8; LORA_BUNDLE_TRAILER_LEN as usize];
    trailer[..8].copy_from_slice(LORA_BUNDLE_MAGIC);
    trailer[8..].copy_from_slice(&base_len.to_le_bytes());
    f.write_all(&trailer)
        .map_err(|e| format!("lora-hfq: write bundle trailer: {e}"))?;
    Ok(())
}

/// If `path` ends in a LoRA bundle trailer, return the embedded adapter section's
/// byte offset. `Ok(None)` for a plain model with no bundled adapter.
pub fn detect_bundled_lora_offset(path: &Path) -> std::io::Result<Option<u64>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let size = f.seek(SeekFrom::End(0))?;
    if size < LORA_BUNDLE_TRAILER_LEN {
        return Ok(None);
    }
    f.seek(SeekFrom::End(-(LORA_BUNDLE_TRAILER_LEN as i64)))?;
    let mut trailer = [0u8; LORA_BUNDLE_TRAILER_LEN as usize];
    f.read_exact(&mut trailer)?;
    if &trailer[..8] != LORA_BUNDLE_MAGIC {
        return Ok(None);
    }
    let off = u64::from_le_bytes(trailer[8..].try_into().unwrap());
    if off >= size - LORA_BUNDLE_TRAILER_LEN {
        return Ok(None);
    }
    Ok(Some(off))
}

/// Read the adapter bundled into a merged model `.hfq`, or `Ok(None)` if `path`
/// carries no bundle trailer.
pub fn read_bundled_lora(path: &Path) -> Result<Option<LoraAdapter>, String> {
    let off = detect_bundled_lora_offset(path)
        .map_err(|e| format!("lora-hfq: probe bundle trailer of {}: {e}", path.display()))?;
    let Some(off) = off else {
        return Ok(None);
    };
    let f = HfqFile::open_at_offset(path, off)
        .map_err(|e| format!("lora-hfq: open embedded section of {}: {e}", path.display()))?;
    adapter_from_hfq(&f, &format!("{}@{off}", path.display())).map(Some)
}

fn rows(f: &HfqFile, name: &str, hidden: usize) -> Result<Vec<Vec<f32>>, String> {
    let (_, bytes) = f
        .tensor_data_vec(name)
        .ok_or_else(|| format!("lora-hfq: missing '{name}' tensor"))?;
    if hidden == 0 || bytes.len() % (hidden * 4) != 0 {
        return Err(format!(
            "lora-hfq: '{name}' byte length {} not a multiple of hidden*4 ({})",
            bytes.len(),
            hidden * 4
        ));
    }
    Ok(bytes
        .chunks_exact(hidden * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_steer::lora::abliteration_adapter;
    use hipfire_steer::SteerMode;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn hfq_container_round_trips() {
        let dirs = vec![unit(&[0.3, -0.4, 0.5, 0.2]), unit(&[0.1, 0.9, -0.2, 0.3])];
        let ad = abliteration_adapter("rt", &dirs, SteerMode::Ablate, 0.6, 0..2).unwrap();
        let path =
            std::env::temp_dir().join(format!("hipfire-lorahfq-{}.lora.hfq", std::process::id()));
        write_lora_hfq(&path, &ad).unwrap();

        // Magic-detected read path.
        let back = read_lora_any(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ad, back);
        assert_eq!(back.deltas.len(), 2);
    }

    #[test]
    fn bundle_round_trips_and_leaves_base_openable() {
        // A stand-in "base model" hfq (one small f32 tensor).
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("hipfire-base-{pid}.hfq"));
        let dummy = hipfire_runtime::hfq::HfqMemTensor {
            name: "x".to_string(),
            quant_type: 2,
            shape: vec![2],
            group_size: 0,
            data: vec![0u8; 8],
        };
        hipfire_runtime::hfq::write_hfqm_package_mem(&base, 0, "{}", std::slice::from_ref(&dummy))
            .unwrap();
        assert!(detect_bundled_lora_offset(&base).unwrap().is_none());

        let dirs = vec![unit(&[1.0, 2.0, 3.0, 4.0])];
        let ad = abliteration_adapter("b", &dirs, SteerMode::Ablate, 0.3, 0..1).unwrap();
        let out = std::env::temp_dir().join(format!("hipfire-merged-{pid}.hfq"));
        merge_lora_into_model(&base, &ad, &out).unwrap();

        // The trunk still opens (adapter section is past the base's payload)...
        let trunk = hipfire_runtime::hfq::HfqFile::open(&out).unwrap();
        assert!(trunk.find_tensor_info("x").is_some());
        // ...and the bundled adapter round-trips.
        let back = read_bundled_lora(&out).unwrap().unwrap();
        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(&out);
        assert_eq!(ad, back);
    }

    #[test]
    fn read_lora_any_falls_back_to_json() {
        let dirs = vec![unit(&[0.5, 0.5, 0.5, 0.5])];
        let ad = abliteration_adapter("j", &dirs, SteerMode::Ablate, 0.9, 0..1).unwrap();
        let path =
            std::env::temp_dir().join(format!("hipfire-lorajson-{}.lora.json", std::process::id()));
        lora::write_adapter(&path, &ad).unwrap();
        let back = read_lora_any(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ad, back);
    }
}
