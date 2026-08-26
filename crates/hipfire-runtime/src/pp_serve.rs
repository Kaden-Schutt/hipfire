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
//! logic + per-stage weight/KV residency (each layer's weights + KV load on its
//! band's device via `Layout::from_gpus`; embed on stage 0, final-norm/lm_head on
//! the last stage). Under `HIPFIRE_EMULATE_GPUS` all stages still alias device 0,
//! so cross-device transport is only exercised on real multi-GPU HW; the per-
//! device VRAM split is real there.

use crate::hfq::HfqFile;
use crate::llama::{
    self, ForwardScratch, KvCache, KvCacheExt, LlamaConfig, LlamaWeights, PrefillScratch,
};
use crate::multi_gpu::Gpus;
use hipfire_hardware::{DeviceMesh, DimKind};
use rdna_compute::{DType, GpuTensor};
use std::ops::Range;

/// A dense llama model loaded pipeline-parallel across `pp` stages.
pub struct PpModel {
    gpus: Gpus,
    pp: usize,
    config: LlamaConfig,
    /// Model weights distributed across stages: embed on stage 0, final-norm +
    /// lm_head on the output stage, each layer on its band's device. Full-length
    /// `layers` Vec; each stage dereferences only its band's (locally resident)
    /// entries.
    weights: LlamaWeights,
    /// Per-stage decode scratch (the residual `x` + transient buffers).
    scratch: Vec<ForwardScratch>,
    /// One KV cache distributed across stages: layer l's k/v resides on
    /// `device_for_layer(l)` (global index), so each stage writes its band's
    /// entries on its own device. Replaces the prior `pp` full-length caches.
    kv: KvCache,
    /// Layer range owned by each stage (`bands[s]`).
    bands: Vec<Range<usize>>,
    dim: usize,
    max_seq: usize,
}

/// Rollback ownership for `PpModel` construction (Oracle gate-1 blocker 5):
/// every GPU resource `load_from_hfq_inner` acquires after `Gpus::from_mesh`
/// becomes owned by this staging guard the moment it exists, so a failure at
/// any later step frees every allocation on its owning device instead of
/// leaking. Disarmed on success — the fields are moved into the final
/// `PpModel`, so no resource is ever both guard-owned and model-owned.
/// The rollback path and `PpModel::free` share one free-order helper.
struct PpStaging {
    gpus: Option<Gpus>,
    weights: Option<LlamaWeights>,
    /// Completed per-stage scratches; `Some` once the first stage is done
    /// (each scratch is pushed the moment it is fully built, so a mid-loop
    /// failure still frees every completed stage on its device).
    scratch: Option<Vec<ForwardScratch>>,
    kv: Option<KvCache>,
}

impl PpStaging {
    fn gpus(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staged gpus")
    }
    fn gpus_ref(&self) -> &Gpus {
        self.gpus.as_ref().expect("staged gpus")
    }
}

/// Free PP-owned resources in `PpModel::free` order: the per-stage
/// distributed weights, then the distributed KV cache, then each stage's
/// scratch on its stage device. `Option` fields let a partially-built
/// construction (rollback) and a complete model (free) share one path.
fn free_pp_resources(
    gpus: &mut Gpus,
    weights: Option<LlamaWeights>,
    kv: Option<KvCache>,
    scratch: Option<Vec<ForwardScratch>>,
) {
    if let Some(weights) = weights {
        weights.free_gpu_multi(gpus);
    }
    if let Some(kv) = kv {
        kv.free_gpu_multi(gpus);
    }
    if let Some(scratch) = scratch {
        for (s, sc) in scratch.into_iter().enumerate() {
            let g = &mut gpus.devices[s];
            let _ = g.bind_thread();
            sc.free_gpu(g);
        }
    }
}

/// Per-device PP teardown shared by `PpModel::free` and construction
/// rollback: invalidate weight caches + graph state and drain the pool so
/// the VRAM returns to the system. Stages keep `active_stream = None` (the
/// sync boundary path), so there is no stream to destroy.
fn teardown_pp_devices(gpus: &mut Gpus) {
    for dev in gpus.devices.iter_mut() {
        let _ = dev.bind_thread();
        dev.invalidate_weight_caches();
        dev.invalidate_graph_state();
        dev.drain_pool();
    }
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

    /// Load a dense llama-family HFQ for pipeline-parallel serving,
    /// consuming an already-opened HFQ handle.  Gated behind the
    /// `loader-internal` feature — only hipfire-loader should call
    /// this; external consumers use `hipfire_loader::load_admitted`.
    #[doc(hidden)]
    #[cfg(feature = "loader-internal")]
    pub fn load_from_hfq(hfq: &HfqFile, mesh: &DeviceMesh, max_seq: usize) -> Result<Self, String> {
        Self::load_from_hfq_inner(hfq, mesh, max_seq)
    }

    /// Open an HFQ and load a dense PP model.
    ///
    /// **Deprecated** — use `hipfire_loader::load_model_pp` instead.
    /// The loader guarantees admitted-source consistency; this direct
    /// path-opening convenience bypasses the admission guard and will
    /// be removed before 1.0.
    #[deprecated(
        since = "0.2.0",
        note = "use hipfire_loader::load_model_pp for admitted-source consistency"
    )]
    pub fn load(path: &str, mesh: &DeviceMesh, max_seq: usize) -> Result<Self, String> {
        let hfq = HfqFile::open(std::path::Path::new(path)).map_err(|e| format!("{e}"))?;
        Self::load_from_hfq_inner(&hfq, mesh, max_seq)
    }

    fn load_from_hfq_inner(
        hfq: &HfqFile,
        mesh: &DeviceMesh,
        max_seq: usize,
    ) -> Result<Self, String> {
        let pp = mesh.size_of(DimKind::Pp);
        if pp < 2 {
            return Err(format!(
                "PpModel::load needs a Pp axis with size>=2 (got {pp})"
            ));
        }
        if !matches!(hfq.arch_id, 0 | 1) {
            return Err(format!(
                "dense PP serve is llama-family only (arch_id 0/1); got arch_id={}",
                hfq.arch_id
            ));
        }
        let config = crate::hfq::config_from_hfq(&hfq)?;
        let n_layers = config.n_layers;
        let dim = config.dim;

        // ── Construction rollback ownership (Oracle gate-1 blocker 5) ──
        // After `Gpus::from_mesh`, every acquired GPU resource is owned by
        // `staged` the moment it exists; a failure anywhere in the fallible
        // construction below frees every staged resource on its owning
        // device (free order shared with `PpModel::free`). On success the
        // fields are moved out (disarm) into the final model — no resource
        // is ever both guard-owned and model-owned.
        let mut staged = PpStaging {
            gpus: None,
            weights: None,
            scratch: None,
            kv: None,
        };
        let construction = (|| -> Result<Vec<Range<usize>>, String> {
            // `init_uniform` bands layers across stages (layer_to_device via
            // uniform_split_counts — the same split `mesh.stage_for_layer` uses).
            let gpus = Gpus::from_mesh(mesh, n_layers).map_err(|e| format!("from_mesh: {e:?}"))?;
            staged.gpus = Some(gpus);

            // Layer bands from the device mapping (stage s = contiguous run of
            // layers).
            let mut bands: Vec<Range<usize>> = Vec::with_capacity(pp);
            {
                let gpus = staged.gpus_ref();
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

            // Per-stage-resident weights: embed on stage 0, final-norm/lm_head on
            // the output stage, each layer on its band's device (Layout::from_gpus
            // uses the same device_for_layer mapping as `bands`). The forward
            // reads by global layer index; each stage only touches its band's
            // (locally resident) entries.
            let layout = crate::model_load::Layout::from_gpus(staged.gpus_ref(), n_layers);
            let weights = crate::hfq::load_weights_hfq_distributed(
                &hfq,
                &config,
                &mut staged.gpus().devices,
                &layout,
            )
            .map_err(|e| format!("load_weights: {e:?}"))?;
            staged.weights = Some(weights);

            // Per-stage decode scratch (small residual buffers, one set per stage
            // device). Each completed scratch is guard-owned the moment it exists.
            for s in 0..pp {
                let g = &mut staged.gpus().devices[s];
                g.bind_thread().map_err(|e| format!("bind{s}: {e:?}"))?;
                let sc = ForwardScratch::new_with_max_seq(g, &config, max_seq)
                    .map_err(|e| format!("scratch{s}: {e:?}"))?;
                staged
                    .scratch
                    .get_or_insert_with(|| Vec::with_capacity(pp))
                    .push(sc);
            }
            // One distributed KV cache: layer l's k/v on device_for_layer(l),
            // global index. Replaces the prior `pp` full-length caches (pp× → 1×
            // KV).
            let kv = KvCache::new_gpu_q8_multi(
                staged.gpus(),
                n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
            )
            .map_err(|e| format!("kv: {e:?}"))?;
            staged.kv = Some(kv);

            // Test-only fault seam (Oracle gate-1 blocker 5): a deterministic
            // mid-construction failure AFTER every owned GPU resource is live
            // (the per-stage distributed weights, every stage's decode scratch,
            // and the distributed KV cache). The one-shot arm is consumed here,
            // so at most one construction attempt fails. Compiled out of
            // production builds;
            // `tests::failed_construction_reclaims_every_owned_gpu_allocation`
            // proves every earlier allocation is reclaimed on its owning device.
            #[cfg(test)]
            if fault_seam::consume() {
                return Err("pp_serve fault seam: forced failure after all GPU allocations".into());
            }

            // Peer access for the cross-stage boundary copy — enabled AFTER every
            // stage's weights + scratch + KV are live. `enable_peer_all` does not
            // retroactively map allocations made after the enable call (its
            // documented contract), so calling it earlier would let post-alloc peer
            // copies silently write nothing on real multi-GPU HW. NB: NO
            // ensure_rank_streams / active_stream — stages stay on the sync memset +
            // sync boundary path.
            staged
                .gpus()
                .enable_peer_all()
                .map_err(|e| format!("enable_peer_all: {e:?}"))?;

            Ok(bands)
        })();
        let bands = match construction {
            Ok(bands) => bands,
            Err(e) => {
                // Rollback: free every staged resource in `PpModel::free`
                // order on its owning device, invalidate caches/graphs and
                // drain pools — the GPU is back at the pre-construction state
                // before the error is returned.
                if let Some(gpus) = staged.gpus.as_mut() {
                    free_pp_resources(
                        gpus,
                        staged.weights.take(),
                        staged.kv.take(),
                        staged.scratch.take(),
                    );
                    teardown_pp_devices(gpus);
                }
                return Err(e);
            }
        };

        // Disarm: move every staged resource into the final model — nothing
        // is both guard-owned and model-owned after this point.
        let gpus = staged.gpus.take().expect("staged gpus");
        let weights = staged.weights.take().expect("staged weights");
        let scratch = staged.scratch.take().expect("staged scratch");
        let kv = staged.kv.take().expect("staged kv");
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
                &mut self.kv,
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
                &mut self.kv,
                &self.scratch[s],
            )
            .map_err(herr)?;
        }
        Ok(())
    }

    /// Batched pipeline-parallel prefill: run the whole prompt in ONE batched
    /// forward per stage (banded over layers), handing the `[n×dim]` residual
    /// across each stage seam via `boundary_copy` — the batched analog of
    /// [`forward_token`]. Fills every stage's KV for positions `0..n` and leaves
    /// the last-position residual in `scratch[pp-1].x` so [`logits`] (unchanged)
    /// returns the last-position logits. Decode resumes at `pos = n`.
    ///
    /// Single-batch only: `n > PREFILL_MAX_BATCH` (256) falls back to the
    /// per-token loop (no cross-chunk prefill in this cut).
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if n > crate::llama::PREFILL_MAX_BATCH {
            // >256: keep today's per-token behaviour (no cross-chunk in this cut).
            for (pos, &t) in tokens.iter().enumerate() {
                self.forward_token(t, pos)?;
            }
            return Ok(());
        }
        if n > self.max_seq {
            return Err(format!("prefill n {n} > max_seq {}", self.max_seq));
        }
        let dim = self.dim;

        // positions = [0, 1, ..., n-1] i32. Same for EVERY stage (PP bands layers,
        // not positions); each stage gets its own device copy. i32 packed into an
        // f32-typed tensor (same byte width), mirroring `prefill_forward`.
        let pos_bytes: Vec<u8> = (0..n as i32).flat_map(|p| p.to_ne_bytes()).collect();

        // Keep each stage's scratch + residual batch alive across the stage loop
        // (the seam `boundary_copy` reads the previous stage's `x_batch`); freed
        // after the last stage, alloc-per-call like `prefill_forward`.
        let mut scratches: Vec<PrefillScratch> = Vec::with_capacity(self.pp);
        let mut x_batches: Vec<GpuTensor> = Vec::with_capacity(self.pp);

        // ── Stage 0: embed the batch on device 0, then run its band. ──
        {
            let g = &mut self.gpus.devices[0];
            g.bind_thread().map_err(herr)?;
            let scratch0 = PrefillScratch::alloc(g, &self.config, n).map_err(herr)?;
            let x_batch0 = g.alloc_tensor(&[n, dim], DType::F32).map_err(herr)?;
            // Embedding: lookup each token into the batch buffer (mirror prefill_forward).
            let x_single = g.alloc_tensor(&[dim], DType::F32).map_err(herr)?;
            for (i, &token) in tokens.iter().enumerate() {
                llama::embedding_lookup_dispatch(
                    g,
                    self.weights.embd_format,
                    &self.weights.token_embd,
                    &x_single,
                    token,
                    dim,
                )
                .map_err(herr)?;
                g.hip
                    .memcpy_dtod_at(&x_batch0.buf, i * dim * 4, &x_single.buf, 0, dim * 4)
                    .map_err(herr)?;
            }
            g.free_tensor(x_single).map_err(herr)?;
            let positions = g.alloc_tensor(&[n], DType::F32).map_err(herr)?;
            g.hip
                .memcpy_htod(&positions.buf, &pos_bytes)
                .map_err(herr)?;
            llama::prefill_forward_band(
                g,
                &self.weights,
                &self.config,
                &x_batch0,
                self.bands[0].clone(),
                &mut self.kv,
                &positions,
                &scratch0,
                n,
            )
            .map_err(herr)?;
            g.free_tensor(positions).map_err(herr)?;
            scratches.push(scratch0);
            x_batches.push(x_batch0);
        }

        // ── Stages 1..pp: boundary-copy the residual batch from the previous
        //    stage, then run this stage's band. ──
        for s in 1..self.pp {
            // Allocate this stage's buffers first (needs the device mutably), then
            // drop that borrow before the `&self.gpus` boundary_copy.
            let (scratch_s, x_batch_s, positions) = {
                let g = &mut self.gpus.devices[s];
                g.bind_thread().map_err(herr)?;
                let scratch_s = PrefillScratch::alloc(g, &self.config, n).map_err(herr)?;
                let x_batch_s = g.alloc_tensor(&[n, dim], DType::F32).map_err(herr)?;
                let positions = g.alloc_tensor(&[n], DType::F32).map_err(herr)?;
                g.hip
                    .memcpy_htod(&positions.buf, &pos_bytes)
                    .map_err(herr)?;
                (scratch_s, x_batch_s, positions)
            };
            let evt = self
                .gpus
                .boundary_copy(s - 1, s, &x_batches[s - 1].buf, &x_batch_s.buf, n * dim * 4)
                .map_err(herr)?;
            self.gpus.wait_boundary(evt).map_err(herr)?;
            {
                let g = &mut self.gpus.devices[s];
                g.bind_thread().map_err(herr)?;
                llama::prefill_forward_band(
                    g,
                    &self.weights,
                    &self.config,
                    &x_batch_s,
                    self.bands[s].clone(),
                    &mut self.kv,
                    &positions,
                    &scratch_s,
                    n,
                )
                .map_err(herr)?;
                g.free_tensor(positions).map_err(herr)?;
            }
            scratches.push(scratch_s);
            x_batches.push(x_batch_s);
        }

        // ── Logits handoff: copy the LAST-position row of the last stage's
        //    residual into `scratch[last].x` — the buffer `logits()` (final norm
        //    + lm_head) reads. ──
        let last = self.pp - 1;
        {
            let g = &mut self.gpus.devices[last];
            g.bind_thread().map_err(herr)?;
            g.hip
                .memcpy_dtod_at(
                    &self.scratch[last].x.buf,
                    0,
                    &x_batches[last].buf,
                    (n - 1) * dim * 4,
                    dim * 4,
                )
                .map_err(herr)?;
        }

        // ── Free per-stage scratch + residual batch. ──
        for (s, (sc, xb)) in scratches.into_iter().zip(x_batches).enumerate() {
            let g = &mut self.gpus.devices[s];
            g.bind_thread().map_err(herr)?;
            sc.free(g).map_err(herr)?;
            g.free_tensor(xb).map_err(herr)?;
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

    /// Free every GPU allocation this PP model owns (per-stage-distributed
    /// weights + KV + per-stage scratch), then drain each device pool. Same
    /// rationale as [`crate::tp_serve::TpModel::free`]: a bare `drop(PpModel)`
    /// reclaimed nothing (no freeing `Drop` on `GpuTensor` / `DeviceBuffer` /
    /// `GpuPool`, and `Gpu::drop` only re-binds), so a load/unload cycle leaked
    /// the model.
    pub fn free(self) {
        // Same free order the construction rollback uses: distributed weights,
        // then distributed KV, then per-stage scratch, then the per-device
        // teardown.
        let PpModel {
            gpus,
            weights,
            kv,
            scratch,
            ..
        } = self;
        let mut gpus = gpus;
        free_pp_resources(&mut gpus, Some(weights), Some(kv), Some(scratch));
        teardown_pp_devices(&mut gpus);
        // `gpus` drops here → tears down stage device contexts.
    }
}

fn herr(e: hip_bridge::HipError) -> String {
    e.to_string()
}

/// Test-only one-shot fault hook for the PP constructor (compiled out of
/// production builds). When armed, the next `load_from_hfq_inner` fails
/// deterministically at the post-allocation seam in the constructor — after
/// every owned GPU resource is live — and the arm is consumed, so at most one
/// construction attempt can observe it. Atomic, so full-suite isolation needs
/// no global lock beyond the tests' own `GPU_TEST_LOCK`.
#[cfg(test)]
mod fault_seam {
    use std::sync::atomic::{AtomicBool, Ordering};

    static FAIL_NEXT_LOAD: AtomicBool = AtomicBool::new(false);

    /// Arm the next PP construction to fail at the post-allocation seam.
    pub fn arm_fail_next_load() {
        FAIL_NEXT_LOAD.store(true, Ordering::SeqCst);
    }

    /// One-shot consume: returns true exactly once per arm.
    pub fn consume() -> bool {
        FAIL_NEXT_LOAD.swap(false, Ordering::SeqCst)
    }

    /// True while an arm is pending (tests assert the arm is consumed).
    pub fn is_armed() -> bool {
        FAIL_NEXT_LOAD.load(Ordering::SeqCst)
    }

    /// Clear any pending arm (suite-isolation backstop).
    pub fn reset() {
        FAIL_NEXT_LOAD.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_gpu::{DeviceMesh, DimKind};
    use std::io::Write;

    /// Write a minimal HFQ fixture for host-only preflight tests.
    fn write_minimal_hfq(
        dir: &tempfile::TempDir,
        name: &str,
        arch_id: u32,
        metadata_json: &str,
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let meta_bytes = metadata_json.as_bytes();
        let metadata_offset: u64 = 32;
        let n_tensors: u32 = 0;
        let index_offset = metadata_offset + meta_bytes.len() as u64;
        let data_offset = index_offset + 4;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        path
    }

    #[test]
    fn load_from_hfq_rejects_non_llama_arch_id_before_gpu() {
        let dir = tempfile::tempdir().unwrap();
        // DeepSeek4 (arch_id=9) is not llama-family
        let path = write_minimal_hfq(
            &dir,
            "ds4.hfq",
            9,
            r#"{"config":{"model_type":"deepseek_v4","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}}"#,
        );
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]);
        let err = match PpModel::load_from_hfq_inner(&hfq, &mesh, 64) {
            Err(e) => e,
            Ok(_) => panic!("expected error for non-llama arch_id, got Ok"),
        };
        assert!(
            err.contains("arch_id=9"),
            "expected arch_id=9 error, got: {err}"
        );
        assert!(
            err.contains("llama-family only"),
            "expected llama-family error, got: {err}"
        );
    }

    #[test]
    fn load_from_hfq_rejects_pp_below_two() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_minimal_hfq(
            &dir,
            "llama.hfq",
            0,
            r#"{"config":{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}}"#,
        );
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 1)]);
        let err = match PpModel::load_from_hfq_inner(&hfq, &mesh, 64) {
            Err(e) => e,
            Ok(_) => panic!("expected error for pp<2, got Ok"),
        };
        assert!(
            err.contains("Pp axis with size>=2"),
            "expected pp>=2 error, got: {err}"
        );
    }

    /// Oracle gate-1 blocker 5 (PP): construction lacks rollback ownership —
    /// a fallible step after GPU allocations (here: the post-allocation seam
    /// before `enable_peer_all`) leaks every earlier owned resource. The arm
    /// fires only after the per-stage distributed weights, every stage's
    /// decode scratch and the distributed KV cache are live, so a correct
    /// rollback must free them all on their owning devices. Fails today:
    /// those allocations stay hipMalloc'd and each device's free VRAM stays
    /// below the warm baseline. Uses the shared dense-llama fixture + GPU
    /// helpers from `crate::tp_serve::test_support`.
    #[test]
    fn failed_construction_reclaims_every_owned_gpu_allocation() {
        use crate::tp_serve::test_support;
        use rdna_compute::Gpu;

        // Same gate as the other GPU tests: skip cleanly on GPU-less CI.
        if Gpu::init().is_err() {
            eprintln!("skip: no GPU");
            return;
        }
        let _lock = test_support::GPU_TEST_LOCK.lock().unwrap();
        let _emulate = test_support::EnvGuard::set("HIPFIRE_EMULATE_GPUS", "2");

        let dir = tempfile::tempdir().unwrap();
        let path = test_support::write_dense_llama_hfq(&dir);
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]);

        // Probe handles for per-device VRAM observation — same mesh/degree as
        // the constructor's `Gpus`, kept alive across baseline and post-failure
        // reads so its own context overhead cancels out.
        let mut probe =
            Gpus::from_mesh(&mesh, test_support::N_LAYERS).expect("probe Gpus must bind");

        // Warm-up cycle: one successful load + free absorbs one-time driver /
        // kernel residency, so the measured baseline is stable.
        {
            let model = PpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ)
                .expect("warm-up PP load must succeed");
            model.free();
        }
        for dev in probe.devices.iter_mut() {
            dev.drain_pool();
        }
        let baseline: Vec<usize> = probe.devices.iter().map(test_support::vram_free).collect();

        // Forced mid-construction failure at the post-allocation seam.
        fault_seam::reset();
        fault_seam::arm_fail_next_load();
        let err = match PpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ) {
            Ok(_) => panic!("armed PP fault must fail the construction"),
            Err(e) => e,
        };
        assert!(
            err.contains("fault seam"),
            "expected the post-allocation seam to fail, got: {err}"
        );
        assert!(
            !fault_seam::is_armed(),
            "fault arm must be one-shot (consumed by the failed construction)"
        );

        // Every owning device must have reclaimed all earlier allocations.
        for (d, &base) in baseline.iter().enumerate() {
            let after = test_support::vram_free(&probe.devices[d]);
            assert!(
                base.abs_diff(after) < test_support::VRAM_TOLERANCE,
                "device {d}: VRAM not reclaimed after failed PP construction \
                 (Oracle gate-1 blocker 5): baseline={base} after={after} \
                 delta={} — every owned allocation must be freed on its device",
                base.saturating_sub(after)
            );
        }

        // Success cycle after the failure: the GPU must remain usable and a
        // full load + free must return VRAM to the same baseline.
        {
            let model = PpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ)
                .expect("PP load must succeed after the forced failure");
            model.free();
        }
        for (d, &base) in baseline.iter().enumerate() {
            let after = test_support::vram_free(&probe.devices[d]);
            assert!(
                base.abs_diff(after) < test_support::VRAM_TOLERANCE,
                "device {d}: VRAM not recovered after the post-failure success cycle: \
                 baseline={base} after={after}"
            );
        }
    }
}
