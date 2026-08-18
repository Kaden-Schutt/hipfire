// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Expert-prediction adapter — forecasts layer L+1's routing from layer L.
//!
//! Expert paging is I/O bound: at an 8 GiB budget, expert transfer is 41.9% of
//! decode wall time. Hiding that behind compute needs the expert set BEFORE the
//! pass arrives at the layer, and ds4's own router cannot supply it — the top-k
//! depends on the activations at that layer.
//!
//! Three training-free predictors were measured and rejected (share of the miss
//! stream each could name in advance): recency 0.1%, token-conditioned 13.0%,
//! and running the real `gate_{L+1}` on `h_L` 4.2% (27.6% recall@6). The last
//! fails because mHC replaces the residual stream with 4 Sinkhorn-mixed
//! streams, so there is no slowly-evolving residual to read through.
//!
//! A TRAINED linear map does work. Fitted offline by closed-form ridge and
//! truncated to rank r:
//!
//! | rank | params/layer |  total | recall@6 |
//! |------|--------------|--------|----------|
//! |   32 |      139,264 |   5.6M |    54.9% |
//! |  128 |      557,056 |  22.3M |    75.6% |
//! | full |    1,048,576 |  41.9M |    78.7% |
//!
//! r=128 is the operating point: 45 MB at f16 (0.05% of an 86 GB model) and
//! ~22M MACs/token, under 1% of per-token compute.
//!
//! The prediction only RANKS prefetch candidates. The frozen native router
//! still makes the real selection, so a wrong ranking costs a prefetch miss —
//! never a wrong answer.

use rdna_compute::{DType, Gpu, GpuTensor};

const MAGIC: &[u8; 8] = b"HFADPT\0\0";

/// One layer's factors: `z_{L+1} ~= B (A h_L) + gate_bias_{L+1}`.
pub struct AdapterLayer {
    /// Source layer; predicts `src_layer + 1`.
    pub src_layer: usize,
    /// `[rank, d_model]`, f16.
    pub a: GpuTensor,
    /// `[n_exp, rank]`, f16.
    pub b: GpuTensor,
}

pub struct ExpertAdapter {
    pub layers: Vec<AdapterLayer>,
    pub d_model: usize,
    pub n_exp: usize,
    pub rank: usize,
    /// Scratch `[rank]` for the intermediate `A h`.
    hidden: GpuTensor,
    /// Scratch `[n_exp]` for the predicted scores.
    scores: GpuTensor,
    /// Host-side `gate_bias` per predicted layer.
    ///
    /// The bias is a per-layer CONSTANT, so fetching it from the device on
    /// every layer of every token costs one synchronous D2H per layer -- ~40
    /// pipeline drains per token, which measured as a large part of a 27%
    /// regression. Cache on first use.
    bias_cache: std::collections::HashMap<usize, Vec<f32>>,
}

impl ExpertAdapter {
    /// Parse and upload an adapter produced by `export_adapter.py`.
    pub fn load(path: &str, gpu: &mut Gpu) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("adapter read {path}: {e}"))?;
        if raw.len() < 28 || &raw[..8] != MAGIC {
            return Err(format!("adapter {path}: bad magic"));
        }
        let u32at = |o: usize| -> usize {
            u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]) as usize
        };
        let (version, n_entries, d_model, n_exp, rank) =
            (u32at(8), u32at(12), u32at(16), u32at(20), u32at(24));
        if version != 1 {
            return Err(format!("adapter {path}: version {version} unsupported"));
        }
        let a_bytes = rank * d_model * 2;
        let b_bytes = n_exp * rank * 2;
        let mut off = 28;
        let mut layers = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            if off + 4 + a_bytes + b_bytes > raw.len() {
                return Err(format!(
                    "adapter {path}: truncated at layer {}",
                    layers.len()
                ));
            }
            let src_layer = u32at(off);
            off += 4;
            let a = gpu
                .upload_raw(&raw[off..off + a_bytes], &[rank, d_model])
                .map_err(|e| format!("adapter upload A l{src_layer}: {e:?}"))?;
            off += a_bytes;
            let b = gpu
                .upload_raw(&raw[off..off + b_bytes], &[n_exp, rank])
                .map_err(|e| format!("adapter upload B l{src_layer}: {e:?}"))?;
            off += b_bytes;
            layers.push(AdapterLayer { src_layer, a, b });
        }
        let hidden = gpu
            .alloc_tensor(&[rank], DType::F32)
            .map_err(|e| format!("adapter scratch: {e:?}"))?;
        let scores = gpu
            .alloc_tensor(&[n_exp], DType::F32)
            .map_err(|e| format!("adapter scores: {e:?}"))?;
        eprintln!(
            "deepseek4: expert adapter loaded — {} layers, rank {rank}, {:.1} MB",
            layers.len(),
            raw.len() as f64 / 1e6
        );
        Ok(Self {
            layers,
            d_model,
            n_exp,
            rank,
            hidden,
            scores,
            bias_cache: std::collections::HashMap::new(),
        })
    }

    /// `gate_bias` for `layer`, downloaded once and cached thereafter.
    pub fn cached_bias(
        &mut self,
        layer: usize,
        tensor: Option<&GpuTensor>,
        gpu: &mut Gpu,
    ) -> Option<&[f32]> {
        if !self.bias_cache.contains_key(&layer) {
            let v = gpu.download_f32(tensor?).ok()?;
            self.bias_cache.insert(layer, v);
        }
        self.bias_cache.get(&layer).map(|v| v.as_slice())
    }

    /// Index of the entry predicting from `src_layer`, if present.
    pub fn entry_for(&self, src_layer: usize) -> Option<usize> {
        self.layers.iter().position(|l| l.src_layer == src_layer)
    }

    /// Predict layer `src_layer + 1`'s scores from `h` and return the top-`m`
    /// expert ids, best first.
    ///
    /// `bias` is the NEXT layer's `gate_bias`; the live router selects over
    /// (scores + bias), so ranking without it would predict a different top-k
    /// than the one actually taken.
    pub fn predict_topm(
        &mut self,
        idx: usize,
        h: &GpuTensor,
        bias: Option<&[f32]>,
        m: usize,
        gpu: &mut Gpu,
    ) -> Result<Vec<u16>, String> {
        let (rank, n_exp) = (self.rank, self.n_exp);
        let (a, b) = {
            let l = &self.layers[idx];
            (&l.a, &l.b)
        };
        gpu.gemv_f16_xf32(a, h, &self.hidden, rank, self.d_model)
            .map_err(|e| format!("adapter A gemv: {e:?}"))?;
        gpu.gemv_f16_xf32(b, &self.hidden, &self.scores, n_exp, rank)
            .map_err(|e| format!("adapter B gemv: {e:?}"))?;
        let mut s = gpu
            .download_f32(&self.scores)
            .map_err(|e| format!("adapter download: {e:?}"))?;
        if let Some(bs) = bias {
            for (v, bv) in s.iter_mut().zip(bs.iter()) {
                *v += *bv;
            }
        }
        let mut order: Vec<u16> = (0..n_exp.min(s.len()) as u16).collect();
        order.sort_unstable_by(|&x, &y| {
            s[y as usize]
                .partial_cmp(&s[x as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(m);
        Ok(order)
    }
}
