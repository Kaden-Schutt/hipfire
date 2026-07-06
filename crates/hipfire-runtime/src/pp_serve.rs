// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense pipeline-parallel served model (P-C). The llama-family analog of
//! [`crate::tp_serve::TpModel`], but for the `Pp` axis: layers are banded across
//! stages and the residual hidden is handed across each stage seam via
//! `Gpus::boundary_copy` — the driver-owned loop the pivot's "three homes" locks
//! (PP lives ABOVE `execute_steps`, at the driver). No executor change.
//!
//! Each token: embed on stage 0 → per-stage `forward_scratch_band(band_range)`
//! with a `boundary_copy` of the residual between stages → `forward_scratch_head`
//! (final norm + lm_head) on the last stage. Capacity-PP (one token in flight,
//! sequential); the win is VRAM to fit a bigger model, not throughput.
//!
//! **PP is EXACT** (a plain F32 residual byte-copy, no collective/reorder), so the
//! oracle bar is max|Δ|=0 vs single-device (proven in `examples/llama_store_pp.rs`;
//! `s_ef_residual` is DeltaNet-only, N/A to dense llama).
//!
//! **Stream regime (correctness):** stages keep `active_stream = None` — the sync
//! `boundary_copy` + sync memset path — matching the single-device reference, so
//! the =0 bar holds. Do NOT `ensure_rank_streams` here (the None→Some memset
//! sync→async trap).
//!
//! Scope: llama-family (arch_id 0/1), Q8 KV, per-token forward (no batched
//! prefill yet), stateless per request (pos 0). Emulated Pp-N proves the BANDING
//! logic; per-device weight residency + cross-device transport need real HW (under
//! `HIPFIRE_EMULATE_GPUS` all stages alias device 0). Whole weights live on the
//! output stage; real-HW per-stage weight banding (VRAM win) is a follow-up.

use crate::hfq::HfqFile;
use crate::llama::{self, ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use crate::multi_gpu::Gpus;
use std::ops::Range;

/// A dense llama model loaded pipeline-parallel across `pp` stages.
pub struct PpModel {
    gpus: Gpus,
    pp: usize,
    config: LlamaConfig,
    /// Whole model, resident on the output (last) stage device. Under emulation
    /// every stage aliases device 0, so the banded forward reads it on any stage;
    /// real-HW per-stage weight banding is a follow-up (see module doc).
    weights: LlamaWeights,
    /// Per-stage decode scratch (the residual `x` + transient buffers).
    scratch: Vec<ForwardScratch>,
    /// Per-stage KV cache (sized to all layers; each stage writes only its band).
    kv: Vec<KvCache>,
    /// Layer range owned by each stage (`bands[s]`).
    bands: Vec<Range<usize>>,
    dim: usize,
    max_seq: usize,
}

impl PpModel {
    pub fn eos_token(&self) -> u32 {
        self.config.eos_token
    }
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
    pub fn pp(&self) -> usize {
        self.pp
    }

    /// Load a dense llama-family HFQ pipeline-parallel across `pp` stages.
    pub fn load(path: &str, pp: usize, max_seq: usize) -> Result<Self, String> {
        if pp < 2 {
            return Err(format!("PpModel::load needs pp>=2 (got {pp})"));
        }
        let hfq = HfqFile::open(std::path::Path::new(path)).map_err(|e| format!("{e}"))?;
        if !matches!(hfq.arch_id, 0 | 1) {
            return Err(format!(
                "dense PP serve is llama-family only (arch_id 0/1); got arch_id={}",
                hfq.arch_id
            ));
        }
        let config = crate::hfq::config_from_hfq(&hfq)?;
        let n_layers = config.n_layers;
        let dim = config.dim;

        // `init_uniform` bands layers across stages (layer_to_device via
        // uniform_split_counts — the same split `mesh.stage_for_layer` uses).
        let mut gpus =
            Gpus::init_uniform(pp, n_layers).map_err(|e| format!("init_uniform: {e:?}"))?;
        // Peer access for the cross-stage boundary copy. NB: NO ensure_rank_streams
        // / active_stream — stages stay on the sync memset + sync boundary path.
        gpus.enable_peer_all()
            .map_err(|e| format!("enable_peer_all: {e:?}"))?;

        // Layer bands from the device mapping (stage s = contiguous run of layers).
        let out_dev = gpus.output_device;
        let mut bands: Vec<Range<usize>> = Vec::with_capacity(pp);
        {
            let mut s = 0usize;
            let mut start = 0usize;
            for l in 0..n_layers {
                let d = gpus.device_for_layer(l);
                if d != s {
                    bands.push(start..l);
                    s = d;
                    start = l;
                }
            }
            bands.push(start..n_layers);
        }
        if bands.len() != pp {
            return Err(format!(
                "PpModel: {} bands for pp={pp} (layer_to_device not contiguous?)",
                bands.len()
            ));
        }

        // Whole weights on the output stage device (see struct doc). Under
        // emulation this is device 0 = every stage.
        let weights = {
            let g = &mut gpus.devices[out_dev];
            g.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
            crate::hfq::load_weights_hfq(&hfq, &config, g)
                .map_err(|e| format!("load_weights: {e:?}"))?
        };

        // Per-stage scratch + KV on each stage's device.
        let mut scratch = Vec::with_capacity(pp);
        let mut kv = Vec::with_capacity(pp);
        for s in 0..pp {
            let g = &mut gpus.devices[s];
            g.bind_thread().map_err(|e| format!("bind{s}: {e:?}"))?;
            scratch.push(
                ForwardScratch::new_with_max_seq(g, &config, max_seq)
                    .map_err(|e| format!("scratch{s}: {e:?}"))?,
            );
            kv.push(
                KvCache::new_gpu_q8(g, n_layers, config.n_kv_heads, config.head_dim, max_seq)
                    .map_err(|e| format!("kv{s}: {e:?}"))?,
            );
        }

        Ok(PpModel {
            gpus,
            pp,
            config,
            weights,
            scratch,
            kv,
            bands,
            dim,
            max_seq,
        })
    }

    /// Run one pipeline-parallel token forward at `pos`: embed on stage 0 → per-
    /// stage `forward_scratch_band` with a residual `boundary_copy` between stages.
    /// Leaves the last stage's residual in `scratch[pp-1].x` for [`logits`].
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String> {
        if pos >= self.max_seq {
            return Err(format!("pos {pos} >= max_seq {}", self.max_seq));
        }
        // Stage 0: embed + its band.
        {
            let g = &mut self.gpus.devices[0];
            g.bind_thread().map_err(herr)?;
            llama::forward_scratch_embed(
                g,
                &self.weights,
                &self.config,
                token,
                pos,
                &self.scratch[0],
            )
            .map_err(herr)?;
            llama::forward_scratch_band(
                g,
                &self.weights,
                &self.config,
                self.bands[0].clone(),
                pos,
                &mut self.kv[0],
                &self.scratch[0],
            )
            .map_err(herr)?;
        }
        // Stages 1..pp: boundary-copy the residual from the previous stage, run band.
        for s in 1..self.pp {
            let evt = self
                .gpus
                .boundary_copy(
                    s - 1,
                    s,
                    &self.scratch[s - 1].x.buf,
                    &self.scratch[s].x.buf,
                    self.dim * 4,
                )
                .map_err(herr)?;
            self.gpus.wait_boundary(evt).map_err(herr)?;
            let g = &mut self.gpus.devices[s];
            g.bind_thread().map_err(herr)?;
            // `forward_scratch_band` reads `scratch.pos_buf` for RoPE + attention,
            // but only `forward_scratch_embed` (stage 0) sets it — so every
            // downstream stage must set its own pos_buf, else it RoPEs/attends at a
            // stale position (0). This is invisible at pos 0 (the init value), which
            // is why the single-position oracle missed it.
            g.hip
                .memcpy_htod(&self.scratch[s].pos_buf, &(pos as i32).to_ne_bytes())
                .map_err(herr)?;
            llama::forward_scratch_band(
                g,
                &self.weights,
                &self.config,
                self.bands[s].clone(),
                pos,
                &mut self.kv[s],
                &self.scratch[s],
            )
            .map_err(herr)?;
        }
        Ok(())
    }

    /// Final norm + lm_head on the last stage → the vocab logits for the token
    /// after the last [`forward_token`].
    pub fn logits(&mut self) -> Result<Vec<f32>, String> {
        let last = self.pp - 1;
        let g = &mut self.gpus.devices[last];
        g.bind_thread().map_err(herr)?;
        llama::forward_scratch_head(g, &self.weights, &self.config, &self.scratch[last])
            .map_err(herr)?;
        g.download_f32(&self.scratch[last].logits).map_err(herr)
    }
}

fn herr(e: hip_bridge::HipError) -> String {
    e.to_string()
}
