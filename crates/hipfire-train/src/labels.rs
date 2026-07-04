// SPDX-License-Identifier: Apache-2.0
//! PFlash drafter label IO — shared by the standalone `ssm_drafter_train` example
//! and the daemon `train_drafter` op (see docs/plans/2026-06-19-train-as-daemon-op.md).
//!
//! Daemon `pflash_labels` emits one JSONL line per chunk
//! (`{tokens, mid_scores, shallow_scores}`) plus a `<path>.embed.bin` `QEMB`
//! sidecar (the target's fp32 token embedding). This loads both and (optionally)
//! shuffles chunks before the train/eval split.

use hipfire_rdna::{Gpu, GpuTensor};

/// A loaded label set: per-chunk tokens + mid/shallow block scores + the shared
/// (frozen) target embedding, plus its geometry.
pub struct LabelSet {
    pub chunks: Vec<Vec<u32>>,
    pub label_mid: Vec<Vec<f32>>,
    pub base_shallow: Vec<Vec<f32>>,
    pub embed: GpuTensor, // [vocab, h_t], frozen shared target embedding
    pub h_t: usize,
    pub vocab: usize,
}

/// Load daemon `pflash_labels` JSONL + its `<jsonl>.embed.bin` (`QEMB`) sidecar.
/// Each chunk's token length must equal `seq`.
pub fn load_daemon_labels(
    gpu: &mut Gpu,
    jsonl: &str,
    seq: usize,
) -> Result<LabelSet, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(jsonl)?;
    let (mut chunks, mut label_mid, mut base_shallow) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let arr_u32 = |k: &str| -> Vec<u32> {
            v[k].as_array()
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect())
                .unwrap_or_default()
        };
        let arr_f32 = |k: &str| -> Vec<f32> {
            v[k].as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default()
        };
        let toks = arr_u32("tokens");
        if toks.len() != seq {
            return Err(format!("daemon label chunk len {} != seq {seq}", toks.len()).into());
        }
        chunks.push(toks);
        label_mid.push(arr_f32("mid_scores"));
        base_shallow.push(arr_f32("shallow_scores"));
    }
    if chunks.is_empty() {
        return Err(format!("no label chunks in {jsonl}").into());
    }

    // embed sidecar: QEMB | u32 vocab | u32 dim | vocab*dim f32
    let bytes = std::fs::read(format!("{jsonl}.embed.bin"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"QEMB" {
        return Err("daemon embed sidecar: bad magic".into());
    }
    let vocab = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    let data: Vec<f32> = bytes[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if data.len() != vocab * dim {
        return Err(format!(
            "embed sidecar size mismatch: {} != {}",
            data.len(),
            vocab * dim
        )
        .into());
    }
    let embed = gpu.upload_f32(&data, &[vocab, dim])?;
    Ok(LabelSet {
        chunks,
        label_mid,
        base_shallow,
        embed,
        h_t: dim,
        vocab,
    })
}

/// Deterministic Fisher–Yates shuffle of the chunk/label arrays in lockstep,
/// BEFORE the train/eval split. A content-ordered corpus (docs → crates → …)
/// would otherwise put a different-domain eval tail against the train set.
pub fn shuffle_in_place(
    chunks: &mut Vec<Vec<u32>>,
    label_mid: &mut Vec<Vec<f32>>,
    base_shallow: &mut Vec<Vec<f32>>,
    seed: u64,
) {
    let mut s = seed;
    let mut perm: Vec<usize> = (0..chunks.len()).collect();
    for i in (1..perm.len()).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (s >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    let take_u32 = |src: &mut Vec<Vec<u32>>, p: &[usize]| -> Vec<Vec<u32>> {
        p.iter().map(|&k| std::mem::take(&mut src[k])).collect()
    };
    let take_f32 = |src: &mut Vec<Vec<f32>>, p: &[usize]| -> Vec<Vec<f32>> {
        p.iter().map(|&k| std::mem::take(&mut src[k])).collect()
    };
    *chunks = take_u32(chunks, &perm);
    *label_mid = take_f32(label_mid, &perm);
    *base_shallow = take_f32(base_shallow, &perm);
}

/// Save SSM-drafter weights (best-eval snapshot) to a flat container. Minimal:
/// magic `SDFT` | u32 ver | u32 epoch | u32 n_tensors | (u32 len, f32[len])*.
/// Resume/AdamW-state persistence is deferred to a later step.
pub fn save_ssm_drafter_weights(
    path: &str,
    weights: &[Vec<f32>],
    epoch: u32,
) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = format!("{path}.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(b"SDFT")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&epoch.to_le_bytes())?;
    f.write_all(&(weights.len() as u32).to_le_bytes())?;
    for w in weights {
        f.write_all(&(w.len() as u32).to_le_bytes())?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
        f.write_all(bytes)?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, path)
}
