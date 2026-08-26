// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense tensor-parallel served model (PB-TP5). The reusable production form of
//! the `tp_decode_parity` example: a whole dense llama-family HFQ loaded
//! tensor-parallel over a `Gpus` mesh, exposing a per-token forward + logits so
//! the daemon's `generate_tp` can prefill + decode.
//!
//! Each token: embed on rank 0 (broadcast the replicated hidden) → all layers as
//! per-rank `Step` lists through `execute_steps_tp` (KV write at the token's pos)
//! → final norm + lm_head on rank 0. The residual hidden `x` stays replicated
//! across ranks (every all-reduce + replicated ResidualAdd/Rmsnorm keeps it in
//! sync). Validated argmax-exact vs single-GPU `forward_scratch` (see
//! `examples/tp_decode_parity.rs`).
//!
//! Scope: llama-family dense (arch_id 0/1), Q8 KV, MQ4G256 weights, single-axis
//! `Tp` mesh. Prefill is batched (`prefill`, single-batch ≤256; >256 falls back
//! to the per-token loop); each request starts at pos 0 (stateless — no
//! multi-turn KV reuse).

use crate::hfq::HfqFile;
use crate::llama::{self, KvCache, KvCacheExt, LlamaConfig, LlamaWeights};
use crate::multi_gpu::Gpus;
use crate::weight_manifest::{ShardPolicy, WeightEntry};
use crate::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_hardware::{DeviceMesh, DimKind};
use rdna_compute::{DType, Gpu, GpuTensor};

const MQ4G256_QT: u8 = 13;

/// Per-rank persistent decode buffers + KV cache + replicated per-layer norms.
struct TpRank {
    x: GpuTensor,
    tmp: GpuTensor,
    x_rot: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attn: GpuTensor,
    o: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    hidden: GpuTensor,
    fo: GpuTensor,
    partials: GpuTensor,
    pos_buf: DeviceBuffer,
    kv: KvCache,
    /// Per layer: (attn_norm, ffn_norm, q_norm, k_norm), all F32, replicated.
    norms: Vec<(GpuTensor, GpuTensor, GpuTensor, GpuTensor)>,
}

/// Rollback ownership for `TpModel` construction (Oracle gate-1 blocker 5):
/// every GPU resource `load_from_hfq_inner` acquires after `Gpus::from_mesh`
/// becomes owned by this staging guard the moment it exists, so a failure at
/// any later step frees every allocation on its owning device instead of
/// leaking. Disarmed on success — the fields are moved into the final
/// `TpModel`, so no resource is ever both guard-owned and model-owned.
/// The rollback path and `TpModel::free` share one free-order helper.
struct TpStaging {
    /// Rank→device map (set before any rank resource can exist, so a rollback
    /// with completed ranks or rank-0 weights always has a valid group).
    group: Option<Vec<usize>>,
    gpus: Option<Gpus>,
    weights: Option<LlamaWeights>,
    store: Option<WeightStore>,
    /// Completed ranks, each pushed the moment it is fully built.
    ranks: Option<Vec<TpRank>>,
    /// The rank currently being built — owned from its first allocation, so a
    /// mid-rank failure frees the partial set on the rank's device.
    cur: RankBuild,
}

/// Partial per-rank construction buffer. Every tensor / norm / `pos_buf` /
/// KV becomes guard-owned the moment its allocation succeeds; `finish_rank`
/// moves them into a `TpRank` (disarm) only after the whole rank is built.
#[derive(Default)]
struct RankBuild {
    dev: Option<usize>,
    /// x, tmp, x_rot, q, k, v, attn, o, gate, up, hidden, fo, partials — in
    /// `TpRank` field order.
    tensors: Vec<GpuTensor>,
    /// 4 per layer: (attn_norm, ffn_norm, q_norm, k_norm).
    norms: Vec<GpuTensor>,
    pos_buf: Option<DeviceBuffer>,
    kv: Option<KvCache>,
}

impl TpStaging {
    fn gpus(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staged gpus")
    }
    fn gpus_ref(&self) -> &Gpus {
        self.gpus.as_ref().expect("staged gpus")
    }
    fn begin_rank(&mut self, dev: usize) {
        self.cur = RankBuild {
            dev: Some(dev),
            ..RankBuild::default()
        };
    }
    /// Adopt a completed per-rank KV cache into the in-flight rank.
    fn rank_kv(&mut self, kv: KvCache) {
        self.cur.kv = Some(kv);
    }
    /// Upload one replicated norm weight and adopt it into the in-flight rank.
    fn rank_norm(&mut self, dev: usize, vals: &[f32]) -> Result<(), String> {
        let g = &self.gpus_ref().devices[dev];
        let t = up_f32(g, vals)?;
        self.cur.norms.push(t);
        Ok(())
    }
    /// Allocate a rank scratch tensor and adopt it into the in-flight rank.
    fn rank_alloc(
        &mut self,
        dev: usize,
        shape: &[usize],
        dtype: DType,
        what: &str,
    ) -> Result<(), String> {
        let g = &mut self.gpus().devices[dev];
        let t = g
            .alloc_tensor(shape, dtype)
            .map_err(|e| format!("{what}: {e:?}"))?;
        self.cur.tensors.push(t);
        Ok(())
    }
    /// Allocate the rank's `pos_buf` and adopt it into the in-flight rank.
    fn rank_pos(&mut self, dev: usize) -> Result<(), String> {
        let g = &mut self.gpus().devices[dev];
        let b = g.hip.malloc(4).map_err(|e| format!("pos_buf: {e:?}"))?;
        self.cur.pos_buf = Some(b);
        Ok(())
    }
    /// Disarm the in-flight rank: move every adopted piece into a `TpRank`.
    fn finish_rank(&mut self) -> TpRank {
        let cur = std::mem::take(&mut self.cur);
        let mut tensors = cur.tensors.into_iter();
        let mut next = || tensors.next().expect("rank tensor order");
        let mut norms: Vec<(GpuTensor, GpuTensor, GpuTensor, GpuTensor)> =
            Vec::with_capacity(cur.norms.len() / 4);
        let mut n = cur.norms.into_iter();
        while let (Some(a), Some(f), Some(q), Some(k)) = (n.next(), n.next(), n.next(), n.next()) {
            norms.push((a, f, q, k));
        }
        TpRank {
            x: next(),
            tmp: next(),
            x_rot: next(),
            q: next(),
            k: next(),
            v: next(),
            attn: next(),
            o: next(),
            gate: next(),
            up: next(),
            hidden: next(),
            fo: next(),
            partials: next(),
            pos_buf: cur.pos_buf.expect("rank pos_buf"),
            kv: cur.kv.expect("rank kv"),
            norms,
        }
    }
}

/// Upload F32 weights as a raw-byte GPU tensor (replicated norms).
fn up_f32(g: &Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
    g.upload_raw(&b, &[v.len()])
        .map_err(|e| format!("norm upload: {e:?}"))
}

/// Free TP-owned resources in `TpModel::free` order: the in-flight rank (if
/// any) on its recorded device, then each completed rank's buffers + norms +
/// pos_buf + KV on its rank's device, then the sharded store across devices,
/// then rank-0's full weights. `Option` fields let a partially-built
/// construction (rollback) and a complete model (free) share one path.
fn free_tp_resources(
    gpus: &mut Gpus,
    group: &[usize],
    ranks: Vec<TpRank>,
    cur: RankBuild,
    store: Option<WeightStore>,
    weights: Option<LlamaWeights>,
) {
    // In-flight rank: partial tensors/norms/pos_buf/KV on its own device.
    if let Some(dev) = cur.dev {
        let g = &mut gpus.devices[dev];
        let _ = g.bind_thread();
        for t in cur.tensors {
            let _ = g.free_tensor(t);
        }
        for t in cur.norms {
            let _ = g.free_tensor(t);
        }
        if let Some(b) = cur.pos_buf {
            let _ = g.hip.free(b);
        }
        if let Some(kv) = cur.kv {
            let _ = kv.free_gpu(g);
        }
    }
    // Completed ranks, each on its rank's device.
    for (r, rank) in ranks.into_iter().enumerate() {
        let g = &mut gpus.devices[group[r]];
        let _ = g.bind_thread();
        for t in [
            rank.x,
            rank.tmp,
            rank.x_rot,
            rank.q,
            rank.k,
            rank.v,
            rank.attn,
            rank.o,
            rank.gate,
            rank.up,
            rank.hidden,
            rank.fo,
            rank.partials,
        ] {
            let _ = g.free_tensor(t);
        }
        for (a, f, q, k) in rank.norms {
            let _ = g.free_tensor(a);
            let _ = g.free_tensor(f);
            let _ = g.free_tensor(q);
            let _ = g.free_tensor(k);
        }
        let _ = g.hip.free(rank.pos_buf);
        let _ = rank.kv.free_gpu(g);
    }
    if let Some(store) = store {
        store.free_all(gpus);
    }
    if let Some(weights) = weights {
        let g = &mut gpus.devices[group[0]];
        let _ = g.bind_thread();
        weights.free_gpu(g);
    }
}

/// Per-device TP teardown shared by `TpModel::free` and construction
/// rollback: invalidate weight caches + graph state, drain the pool so the
/// VRAM returns to the system, and destroy the per-device stream `load`
/// created (`active_stream = Some`). Nothing else tears that stream down, so
/// skipping it leaks a HIP stream (and its driver-side command buffer).
fn teardown_tp_devices(gpus: &mut Gpus) {
    if let Err(error) = gpus.free_peer_reduce_scratch() {
        eprintln!("TP teardown: failed to free peer-reduce scratch: {error}");
    }
    for dev in gpus.devices.iter_mut() {
        let _ = dev.bind_thread();
        dev.invalidate_weight_caches();
        dev.invalidate_graph_state();
        dev.drain_pool();
        if let Some(s) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(s);
        }
    }
}

/// A dense llama model loaded tensor-parallel and ready to serve.
pub struct TpModel {
    gpus: Gpus,
    mesh: DeviceMesh,
    group: Vec<usize>,
    tp: usize,
    config: LlamaConfig,
    /// Held on rank 0 for embed / final-norm / lm_head (and as the F32 norm source).
    weights: LlamaWeights,
    /// Sharded quant weights keyed by (logical_name, layer, device).
    store: WeightStore,
    ranks: Vec<TpRank>,
    collectives: Vec<TpCollective>,
    phys_cap: usize,
    // cached dims
    d: usize,
    hpr: usize,
    kvpr: usize,
    q_dim_r: usize,
    kv_dim_r: usize,
    inter_r: usize,
    qkv_rot: RotationPlan,
    ffn_rot: RotationPlan,
}

impl TpModel {
    pub fn eos_token(&self) -> u32 {
        self.config.eos_token
    }
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
    pub fn tp(&self) -> usize {
        self.tp
    }
    pub fn max_seq(&self) -> usize {
        self.config.max_seq_len
    }

    /// Load a dense llama-family HFQ tensor-parallel across `tp` ranks,
    /// consuming an already-opened HFQ handle.  Gated behind the
    /// `loader-internal` feature — only hipfire-loader should call
    /// this; external consumers use `hipfire_loader::load_admitted`.
    #[doc(hidden)]
    #[cfg(feature = "loader-internal")]
    pub fn load_from_hfq(hfq: &HfqFile, mesh: &DeviceMesh, max_seq: usize) -> Result<Self, String> {
        Self::load_from_hfq_inner(hfq, mesh, max_seq)
    }

    /// Open an HFQ and load a dense TP model.
    ///
    /// **Deprecated** — use `hipfire_loader::load_model_tp` instead.
    /// The loader guarantees admitted-source consistency; this direct
    /// path-opening convenience bypasses the admission guard and will
    /// be removed before 1.0.
    #[deprecated(
        since = "0.2.0",
        note = "use hipfire_loader::load_model_tp for admitted-source consistency"
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
        let tp = mesh.size_of(DimKind::Tp);
        if tp < 2 {
            return Err(format!(
                "TpModel::load needs a Tp axis with size>=2 (got {tp})"
            ));
        }
        if !matches!(hfq.arch_id, 0 | 1) {
            return Err(format!(
                "dense TP serve is llama-family only (arch_id 0/1); got arch_id={} — use ep for MoE",
                hfq.arch_id
            ));
        }
        let config = crate::hfq::config_from_hfq(&hfq)?;
        if !config.has_qk_norm {
            // The per-rank Step list emits Step::QkNorm unconditionally; a
            // non-qk-norm llama would need a variant. Qwen3-family (has_qk_norm)
            // is the validated case.
            return Err("dense TP serve currently requires a qk-norm (Qwen3-family) llama".into());
        }
        let (d, ff, nh, nkv, hd, n_layers) = (
            config.dim,
            config.hidden_dim,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.n_layers,
        );
        let q_dim = nh * hd;
        if nh % tp != 0 || nkv % tp != 0 || ff % tp != 0 {
            return Err(format!(
                "TP shard requires nh({nh}) nkv({nkv}) inter({ff}) all divisible by tp({tp})"
            ));
        }
        // Group-alignment for the FWHT-G256 row shards (wo k=q_dim, down k=ff).
        for (name, kdim) in [("wo", q_dim), ("ffn_down", ff)] {
            if (kdim / tp) % 256 != 0 {
                return Err(format!(
                    "{name} row-shard k/tp = {}/{} = {} not %256==0 (MQ4G256 group-alignment)",
                    kdim,
                    tp,
                    kdim / tp
                ));
            }
        }

        // ── Construction rollback ownership (Oracle gate-1 blocker 5) ──
        // After `Gpus::from_mesh`, every acquired GPU resource is owned by
        // `staged` the moment it exists; a failure anywhere in the fallible
        // construction below frees every staged resource on its owning
        // device (free order shared with `TpModel::free`). On success the
        // fields are moved out (disarm) into the final model — no resource
        // is ever both guard-owned and model-owned.
        let (hpr, kvpr) = (nh / tp, nkv / tp);
        let (q_dim_r, kv_dim_r, inter_r) = (hpr * hd, kvpr * hd, ff / tp);
        let x_rot_cap = d.max(ff);
        let phys_cap = max_seq;
        let chunks = phys_cap.div_ceil(128);
        let mut staged = TpStaging {
            group: None,
            gpus: None,
            weights: None,
            store: None,
            ranks: None,
            cur: RankBuild::default(),
        };
        let construction = (|| -> Result<(RotationPlan, RotationPlan), String> {
            // `Gpus` is staged BEFORE the stream loop so a mid-stream failure
            // leaves the already-created streams owned by the guard (rollback
            // destroys them via `teardown_tp_devices`).
            let gpus = Gpus::from_mesh(mesh, n_layers).map_err(|e| format!("from_mesh: {e:?}"))?;
            staged.gpus = Some(gpus);
            for dev in staged.gpus().devices.iter_mut() {
                dev.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
                let s = dev
                    .hip
                    .stream_create()
                    .map_err(|e| format!("stream_create: {e:?}"))?;
                dev.active_stream = Some(s);
            }

            // Rank→device map (pure mesh lookup; staged before any rank
            // resource can exist, so a rollback with completed ranks or
            // rank-0 weights always knows each rank's device).
            let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
            staged.group = Some(group.clone());

            // Whole weights on rank 0 (embed / final-norm / lm_head + F32 norm source).
            let weights = {
                let g = &mut staged.gpus().devices[0];
                g.bind_thread().map_err(|e| format!("bind0: {e:?}"))?;
                crate::hfq::load_weights_hfq(&hfq, &config, g)
                    .map_err(|e| format!("load_weights: {e:?}"))?
            };
            let qkv_rot = dtype_rotation_plan(weights.layers[0].wq.gpu_dtype);
            let ffn_rot = dtype_rotation_plan(weights.layers[0].w_gate.gpu_dtype);
            staged.weights = Some(weights);

            // Store→forward bridge: shard every layer's quant weights (uses the
            // caller-provided mesh — no internal reconstruction).
            let store = build_store(&hfq, &config, mesh, staged.gpus_ref())?;
            staged.store = Some(store);

            // Per-layer replicated norm CPU copies (download once from rank 0).
            let norms_cpu: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_layers)
                .map(
                    |l| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
                        let g = &staged.gpus_ref().devices[0];
                        let lw = &staged.weights.as_ref().expect("staged weights").layers[l];
                        Ok((
                            g.download_f32(&lw.attn_norm).map_err(herr)?,
                            g.download_f32(&lw.ffn_norm).map_err(herr)?,
                            g.download_f32(
                                lw.q_norm
                                    .as_ref()
                                    .ok_or_else(|| format!("layer {l}: q_norm missing"))?,
                            )
                            .map_err(herr)?,
                            g.download_f32(
                                lw.k_norm
                                    .as_ref()
                                    .ok_or_else(|| format!("layer {l}: k_norm missing"))?,
                            )
                            .map_err(herr)?,
                        ))
                    },
                )
                .collect::<Result<_, _>>()?;

            for &dev in &group {
                staged.begin_rank(dev);
                {
                    let g = &mut staged.gpus().devices[dev];
                    g.bind_thread().map_err(|e| format!("bind{dev}: {e:?}"))?;
                    let kv = KvCache::new_gpu_q8(g, n_layers, kvpr, hd, phys_cap)
                        .map_err(|e| format!("kv: {e:?}"))?;
                    staged.rank_kv(kv);
                }
                for (a, f, q, k) in &norms_cpu {
                    staged.rank_norm(dev, a)?;
                    staged.rank_norm(dev, f)?;
                    staged.rank_norm(dev, q)?;
                    staged.rank_norm(dev, k)?;
                }
                staged.rank_alloc(dev, &[d], DType::F32, "x")?;
                staged.rank_alloc(dev, &[d], DType::F32, "tmp")?;
                staged.rank_alloc(dev, &[x_rot_cap], DType::F32, "x_rot")?;
                staged.rank_alloc(dev, &[q_dim_r], DType::F32, "q")?;
                staged.rank_alloc(dev, &[kv_dim_r], DType::F32, "k")?;
                staged.rank_alloc(dev, &[kv_dim_r], DType::F32, "v")?;
                staged.rank_alloc(dev, &[q_dim_r], DType::F32, "attn")?;
                staged.rank_alloc(dev, &[d], DType::F32, "o")?;
                staged.rank_alloc(dev, &[inter_r], DType::F32, "gate")?;
                staged.rank_alloc(dev, &[inter_r], DType::F32, "up")?;
                staged.rank_alloc(dev, &[inter_r], DType::F32, "hidden")?;
                staged.rank_alloc(dev, &[d], DType::F32, "fo")?;
                staged.rank_alloc(dev, &[hpr * chunks * (2 + hd)], DType::F32, "partials")?;
                staged.rank_pos(dev)?;
                let rank = staged.finish_rank();
                staged
                    .ranks
                    .get_or_insert_with(|| Vec::with_capacity(tp))
                    .push(rank);
            }

            // Test-only fault seam (Oracle gate-1 blocker 5): a deterministic
            // mid-construction failure AFTER every owned GPU resource is live
            // (per-device streams, rank-0 full weights, the sharded store, and all
            // per-rank KV + scratch). The one-shot arm is consumed here, so at most
            // one construction attempt fails. Compiled out of production builds;
            // `tests::failed_construction_reclaims_every_owned_gpu_allocation`
            // proves every earlier allocation is reclaimed on its owning device.
            #[cfg(test)]
            if fault_seam::consume() {
                return Err("tp_serve fault seam: forced failure after all GPU allocations".into());
            }

            // Peer access for the cross-rank all-reduce peer copies — enabled AFTER
            // all weights (rank-0 full + sharded store) + KV + per-rank scratch are
            // live. `enable_peer_all` does not retroactively map allocations made
            // after the enable call (its documented contract), so calling it earlier
            // would let the all-reduce peer copies silently write nothing on real
            // multi-GPU HW.
            staged
                .gpus()
                .enable_peer_all()
                .map_err(|e| format!("enable_peer_all: {e:?}"))?;

            Ok((qkv_rot, ffn_rot))
        })();
        let (qkv_rot, ffn_rot) = match construction {
            Ok(parts) => parts,
            Err(e) => {
                // Rollback: free every staged resource in `TpModel::free`
                // order on its owning device, destroy the per-device streams,
                // invalidate caches/graphs and drain pools — the GPU is back
                // at the pre-construction state before the error is returned.
                if let Some(gpus) = staged.gpus.as_mut() {
                    free_tp_resources(
                        gpus,
                        staged.group.as_deref().unwrap_or(&[]),
                        staged.ranks.take().unwrap_or_default(),
                        std::mem::take(&mut staged.cur),
                        staged.store.take(),
                        staged.weights.take(),
                    );
                    teardown_tp_devices(gpus);
                }
                return Err(e);
            }
        };

        let mut collectives: Vec<TpCollective> = (0..16).map(|_| TpCollective::None).collect();
        collectives[8] = TpCollective::AllReduceOut { dim: d };
        collectives[14] = TpCollective::AllReduceOut { dim: d };

        // Disarm: move every staged resource into the final model — nothing
        // is both guard-owned and model-owned after this point.
        let gpus = staged.gpus.take().expect("staged gpus");
        let group = staged.group.take().expect("staged group");
        let weights = staged.weights.take().expect("staged weights");
        let store = staged.store.take().expect("staged store");
        let ranks = staged.ranks.take().expect("staged ranks");
        Ok(TpModel {
            gpus,
            mesh: mesh.clone(),
            group,
            tp,
            config,
            weights,
            store,
            ranks,
            collectives,
            phys_cap,
            d,
            hpr,
            kvpr,
            q_dim_r,
            kv_dim_r,
            inter_r,
            qkv_rot,
            ffn_rot,
        })
    }

    /// Run one tensor-parallel token forward at `pos`: embed (rank 0) + broadcast
    /// → all layers via `execute_steps_tp` (KV write at `pos`). Mutates the
    /// replicated hidden + per-rank KV.
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String> {
        let n_layers = self.config.n_layers;
        let (d, hd) = (self.d, self.config.head_dim);
        let dev0 = self.group[0];

        // Embed on rank 0.
        {
            let g = &mut self.gpus.devices[dev0];
            g.bind_thread().map_err(herr)?;
            llama::embedding_lookup_dispatch(
                g,
                self.weights.embd_format,
                &self.weights.token_embd,
                &self.ranks[0].x,
                token,
                d,
            )
            .map_err(herr)?;
            g.hip
                .stream_synchronize(g.active_stream.as_ref().unwrap())
                .map_err(herr)?;
        }
        // Broadcast the replicated hidden + set pos on every rank.
        let x0 = self.gpus.devices[dev0]
            .download_f32(&self.ranks[0].x)
            .map_err(herr)?;
        let x0b: Vec<u8> = x0.iter().flat_map(|f| f.to_ne_bytes()).collect();
        for (r, &dev) in self.group.iter().enumerate() {
            let g = &mut self.gpus.devices[dev];
            g.bind_thread().map_err(herr)?;
            g.hip
                .memcpy_htod(&self.ranks[r].pos_buf, &(pos as i32).to_ne_bytes())
                .map_err(herr)?;
            if r != 0 {
                g.hip
                    .memcpy_htod(&self.ranks[r].x.buf, &x0b)
                    .map_err(herr)?;
            }
        }

        // Per-layer TP forward. Edition-2021 disjoint closure captures + disjoint
        // field borrows let the executor mutate `self.gpus` while the Step lists
        // borrow `self.ranks`/`self.store`/`self.mesh`/`self.collectives`.
        let mq = DType::MQ4G256;
        let (hpr, kvpr, q_dim_r, kv_dim_r, inter_r) = (
            self.hpr,
            self.kvpr,
            self.q_dim_r,
            self.kv_dim_r,
            self.inter_r,
        );
        let (eps, theta, qkv_rot, ffn_rot, phys_cap) = (
            self.config.norm_eps,
            self.config.rope_freq_base,
            self.qkv_rot,
            self.ffn_rot,
            self.phys_cap,
        );
        for l in 0..n_layers {
            let store = &self.store;
            let ranks = &self.ranks;
            let group = &self.group;
            let per_rank_steps: Vec<Vec<Step>> = (0..self.tp)
                .map(|r| {
                    let dv = group[r];
                    let s = &ranks[r];
                    let (an, fnrm, qn, kn) = &s.norms[l];
                    build_layer_steps(LayerIo {
                        x: &s.x,
                        tmp: &s.tmp,
                        x_rot: &s.x_rot,
                        q: &s.q,
                        k: &s.k,
                        v: &s.v,
                        attn: &s.attn,
                        o: &s.o,
                        gate: &s.gate,
                        up: &s.up,
                        hidden: &s.hidden,
                        fo: &s.fo,
                        pos_buf: &s.pos_buf,
                        attn_norm: an,
                        ffn_norm: fnrm,
                        q_norm: qn,
                        k_norm: kn,
                        wq: wref(resident_l(store, "wq", l, dv), mq, q_dim_r, d),
                        wk: wref(resident_l(store, "wk", l, dv), mq, kv_dim_r, d),
                        wv: wref(resident_l(store, "wv", l, dv), mq, kv_dim_r, d),
                        wo: wref(resident_l(store, "wo", l, dv), mq, d, q_dim_r),
                        w_gate: wref(resident_l(store, "ffn_gate", l, dv), mq, inter_r, d),
                        w_up: wref(resident_l(store, "ffn_up", l, dv), mq, inter_r, d),
                        w_down: wref(resident_l(store, "ffn_down", l, dv), mq, d, inter_r),
                        plan: KvTierPlan::derive(KvTierInputs {
                            pos,
                            ..s.kv.tier_inputs()
                        })
                        .map_err(|e| e.to_string())
                        .unwrap(),
                        k_cache: &s.kv.k_gpu[l],
                        v_cache: &s.kv.v_gpu[l],
                        partials: &s.partials,
                        nh: hpr,
                        nkv: kvpr,
                        hd,
                        d,
                        eps,
                        theta,
                        qkv_rot,
                        ffn_rot,
                        pos,
                        physical_cap: phys_cap,
                    })
                })
                .collect();
            execute_steps_tp(
                &self.mesh,
                &mut self.gpus,
                &per_rank_steps,
                &self.collectives,
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Batched tensor-parallel prefill: run the whole prompt in ONE batched
    /// forward, sharded head/inter-parallel across the `Tp` group, so every
    /// rank's KV is filled for positions `0..n` and the last-position residual
    /// lands in `ranks[0].x` for [`logits`] (unchanged). Decode resumes at
    /// `pos = n`. The batched analog of [`forward_token`].
    ///
    /// The GEMMs (`Step::Gemm`), attention (batched `Step::Attend`, which writes
    /// its `n` keys to the Q8 KV cache internally), SiLU-mul and residual adds run
    /// through [`execute_steps_tp`] — the two row-parallel projections (`wo`,
    /// `w_down`) carry an `AllReduceOut{n*d}` collective. The three ops that have
    /// no batched `Step` form (the *plain* pre-GEMM rmsnorm — `Step::Gemm` rotates
    /// its input itself, unlike decode's pre-rotating `RmsnormAutomatic` — plus
    /// per-head qk-norm and RoPE) run as direct per-rank batched-kernel calls
    /// between the executor segments, identical to `prefill_forward_band`. The
    /// residual `x` stays replicated across ranks (embed→broadcast is the only
    /// cross-rank move; all-reduces + replicated norms/residuals keep it in sync).
    ///
    /// Single-batch only: `n > PREFILL_MAX_BATCH` (256) falls back to the
    /// per-token loop (no cross-chunk prefill in this cut).
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if n > crate::llama::PREFILL_MAX_BATCH {
            for (pos, &t) in tokens.iter().enumerate() {
                self.forward_token(t, pos)?;
            }
            return Ok(());
        }
        if n > self.phys_cap {
            return Err(format!("prefill n {n} > max_seq {}", self.phys_cap));
        }
        let (d, hd) = (self.d, self.config.head_dim);
        let (hpr, kvpr, q_dim_r, kv_dim_r, inter_r) = (
            self.hpr,
            self.kvpr,
            self.q_dim_r,
            self.kv_dim_r,
            self.inter_r,
        );
        let (eps, theta, phys_cap) = (
            self.config.norm_eps,
            self.config.rope_freq_base,
            self.phys_cap,
        );
        let n_layers = self.config.n_layers;
        let mq = DType::MQ4G256;
        // Largest INPUT dim any batched MQ4G256 Gemm sees on a rank (col ops k=d;
        // row wo k=q_dim/tp; row down k=inter/tp). The Step::Gemm MQ4G256 arm
        // aliases the persistent `gpu.scratch.mq_x_rot` for the FWHT rotation and
        // does NOT grow it (gemv.rs:416), so for a batched prefill it writes
        // `n × k` F32 into a buffer sized for B=1 decode → grow it first.
        let max_k = d.max(q_dim_r).max(inter_r);

        // Per-rank batched buffers, allocated fresh for this prefill (freed at end),
        // mirroring `prefill_forward`'s alloc-per-call. `x` is the replicated
        // residual [n×d]; the rest are on-rank shards.
        struct PfBuf {
            x: GpuTensor,
            tmp: GpuTensor,
            q: GpuTensor,
            k: GpuTensor,
            v: GpuTensor,
            attn: GpuTensor,
            o: GpuTensor,
            gate: GpuTensor,
            up: GpuTensor,
            hidden: GpuTensor,
            fo: GpuTensor,
            positions: GpuTensor,
        }
        let pos_bytes: Vec<u8> = (0..n as i32).flat_map(|p| p.to_ne_bytes()).collect();
        let mut pbs: Vec<PfBuf> = Vec::with_capacity(self.tp);
        for &dev in &self.group {
            let g = &mut self.gpus.devices[dev];
            g.bind_thread().map_err(herr)?;
            // Grow mq_x_rot to hold `n × max_k` F32 (finding #2). ensure_mq_signs
            // first so the sign tables + (default 32768) buffer exist, then replace
            // the buffer if it is too small (bigger is fine for later decode).
            g.ensure_mq_signs().map_err(herr)?;
            let need = n * max_k;
            let have = g
                .scratch
                .mq_x_rot
                .as_ref()
                .map(|t| t.buf.size() / 4)
                .unwrap_or(0);
            if have < need {
                let newbuf = g.alloc_tensor(&[need], DType::F32).map_err(herr)?;
                if let Some(old) = g.scratch.mq_x_rot.replace(newbuf) {
                    g.free_tensor(old).map_err(herr)?;
                }
            }
            let positions = g.alloc_tensor(&[n], DType::F32).map_err(herr)?;
            g.hip
                .memcpy_htod(&positions.buf, &pos_bytes)
                .map_err(herr)?;
            pbs.push(PfBuf {
                x: g.alloc_tensor(&[n, d], DType::F32).map_err(herr)?,
                tmp: g.alloc_tensor(&[n, d], DType::F32).map_err(herr)?,
                q: g.alloc_tensor(&[n, q_dim_r], DType::F32).map_err(herr)?,
                k: g.alloc_tensor(&[n, kv_dim_r], DType::F32).map_err(herr)?,
                v: g.alloc_tensor(&[n, kv_dim_r], DType::F32).map_err(herr)?,
                attn: g.alloc_tensor(&[n, q_dim_r], DType::F32).map_err(herr)?,
                o: g.alloc_tensor(&[n, d], DType::F32).map_err(herr)?,
                gate: g.alloc_tensor(&[n, inter_r], DType::F32).map_err(herr)?,
                up: g.alloc_tensor(&[n, inter_r], DType::F32).map_err(herr)?,
                hidden: g.alloc_tensor(&[n, inter_r], DType::F32).map_err(herr)?,
                fo: g.alloc_tensor(&[n, d], DType::F32).map_err(herr)?,
                positions,
            });
        }

        // Batch-embed the n tokens on rank 0 into [n×d], then broadcast the
        // replicated hidden to every rank (mirror forward_token's embed+broadcast
        // but [n×d] instead of [d]).
        {
            let dev0 = self.group[0];
            let g = &mut self.gpus.devices[dev0];
            g.bind_thread().map_err(herr)?;
            let x_single = g.alloc_tensor(&[d], DType::F32).map_err(herr)?;
            for (i, &token) in tokens.iter().enumerate() {
                llama::embedding_lookup_dispatch(
                    g,
                    self.weights.embd_format,
                    &self.weights.token_embd,
                    &x_single,
                    token,
                    d,
                )
                .map_err(herr)?;
                g.hip
                    .memcpy_dtod_at(&pbs[0].x.buf, i * d * 4, &x_single.buf, 0, d * 4)
                    .map_err(herr)?;
            }
            g.free_tensor(x_single).map_err(herr)?;
            g.hip
                .stream_synchronize(g.active_stream.as_ref().unwrap())
                .map_err(herr)?;
        }
        let x0 = self.gpus.devices[self.group[0]]
            .download_f32(&pbs[0].x)
            .map_err(herr)?;
        let x0b: Vec<u8> = x0.iter().flat_map(|f| f.to_ne_bytes()).collect();
        for (r, &dev) in self.group.iter().enumerate() {
            if r == 0 {
                continue;
            }
            let g = &mut self.gpus.devices[dev];
            g.bind_thread().map_err(herr)?;
            g.hip.memcpy_htod(&pbs[r].x.buf, &x0b).map_err(herr)?;
        }

        // Per-layer TP forward. Manual per-rank rmsnorm/qknorm/rope calls run on
        // each device's active_stream (same stream as the executor's GEMMs → no
        // extra sync needed); execute_steps_tp handles the two all-reduces.
        for l in 0..n_layers {
            // (1) attn rmsnorm (replicated, plain — feeds Step::Gemm which rotates).
            for (r, &dev) in self.group.iter().enumerate() {
                let g = &mut self.gpus.devices[dev];
                g.bind_thread().map_err(herr)?;
                g.rmsnorm_batched(&pbs[r].x, &self.ranks[r].norms[l].0, &pbs[r].tmp, n, d, eps)
                    .map_err(herr)?;
            }
            // (2) column QKV projections via execute_steps_tp.
            {
                let store = &self.store;
                let group = &self.group;
                let w: Vec<[WeightRef; 3]> = (0..self.tp)
                    .map(|r| {
                        let dv = group[r];
                        [
                            wref(resident_l(store, "wq", l, dv), mq, q_dim_r, d),
                            wref(resident_l(store, "wk", l, dv), mq, kv_dim_r, d),
                            wref(resident_l(store, "wv", l, dv), mq, kv_dim_r, d),
                        ]
                    })
                    .collect();
                let steps: Vec<Vec<Step>> = (0..self.tp)
                    .map(|r| {
                        let p = &pbs[r];
                        vec![
                            Step::Gemm {
                                w: &w[r][0],
                                x: &p.tmp,
                                y: &p.q,
                                batch: n,
                            },
                            Step::Gemm {
                                w: &w[r][1],
                                x: &p.tmp,
                                y: &p.k,
                                batch: n,
                            },
                            Step::Gemm {
                                w: &w[r][2],
                                x: &p.tmp,
                                y: &p.v,
                                batch: n,
                            },
                        ]
                    })
                    .collect();
                let coll = [TpCollective::None, TpCollective::None, TpCollective::None];
                execute_steps_tp(&self.mesh, &mut self.gpus, &steps, &coll)
                    .map_err(|e| e.to_string())?;
            }
            // (3) qk-norm + RoPE on this rank's owned heads (per-rank, batched).
            for (r, &dev) in self.group.iter().enumerate() {
                let g = &mut self.gpus.devices[dev];
                g.bind_thread().map_err(herr)?;
                let (_, _, qn, kn) = &self.ranks[r].norms[l];
                g.rmsnorm_batched(&pbs[r].q, qn, &pbs[r].q, n * hpr, hd, eps)
                    .map_err(herr)?;
                g.rmsnorm_batched(&pbs[r].k, kn, &pbs[r].k, n * kvpr, hd, eps)
                    .map_err(herr)?;
                g.rope_batched_f32(
                    &pbs[r].q,
                    &pbs[r].k,
                    &pbs[r].positions,
                    hpr,
                    kvpr,
                    hd,
                    theta,
                    n,
                )
                .map_err(herr)?;
            }
            // (4) batched attention (writes n keys to Q8 KV internally) + row wo
            //     (partial [n×d]) → AllReduceOut → replicated residual add.
            {
                let store = &self.store;
                let group = &self.group;
                let ranks = &self.ranks;
                let wo: Vec<WeightRef> = (0..self.tp)
                    .map(|r| wref(resident_l(store, "wo", l, group[r]), mq, d, q_dim_r))
                    .collect();
                let steps: Vec<Vec<Step>> = (0..self.tp)
                    .map(|r| {
                        let p = &pbs[r];
                        let s = &ranks[r];
                        let plan = KvTierPlan::derive(KvTierInputs {
                            pos: 0,
                            batch_size: n,
                            ..s.kv.tier_inputs()
                        })
                        .expect("batched Q8 KV plan");
                        vec![
                            Step::Attend {
                                plan,
                                io: AttnParams {
                                    q: &p.q,
                                    k: &p.k,
                                    v: &p.v,
                                    k_cache: &s.kv.k_gpu[l],
                                    v_cache: &s.kv.v_gpu[l],
                                    k_scales: None,
                                    v_scales: None,
                                    pos_buf: &s.pos_buf,
                                    pos: 0,
                                    positions: Some(&p.positions),
                                    n_heads: hpr,
                                    n_kv_heads: kvpr,
                                    head_dim: hd,
                                    physical_cap: phys_cap,
                                    output_gate: None,
                                    batch_size: n,
                                    max_ctx_len: n,
                                    flash_partials: None,
                                    givens_cos: None,
                                    givens_sin: None,
                                    tree_bias: None,
                                    block_start: 0,
                                    block_cols: 0,
                                    output: &p.attn,
                                },
                            },
                            Step::Gemm {
                                w: &wo[r],
                                x: &p.attn,
                                y: &p.o,
                                batch: n,
                            },
                            Step::ResidualAdd {
                                x: &p.x,
                                y: &p.o,
                                dim: d,
                            },
                        ]
                    })
                    .collect();
                let coll = [
                    TpCollective::None,
                    TpCollective::AllReduceOut { dim: n * d },
                    TpCollective::None,
                ];
                execute_steps_tp(&self.mesh, &mut self.gpus, &steps, &coll)
                    .map_err(|e| e.to_string())?;
            }
            // (5) ffn rmsnorm (replicated, plain).
            for (r, &dev) in self.group.iter().enumerate() {
                let g = &mut self.gpus.devices[dev];
                g.bind_thread().map_err(herr)?;
                g.rmsnorm_batched(&pbs[r].x, &self.ranks[r].norms[l].1, &pbs[r].tmp, n, d, eps)
                    .map_err(herr)?;
            }
            // (6) column gate/up + SiLU-mul + row down (partial) → AllReduceOut →
            //     replicated residual add.
            {
                let store = &self.store;
                let group = &self.group;
                let w: Vec<[WeightRef; 3]> = (0..self.tp)
                    .map(|r| {
                        let dv = group[r];
                        [
                            wref(resident_l(store, "ffn_gate", l, dv), mq, inter_r, d),
                            wref(resident_l(store, "ffn_up", l, dv), mq, inter_r, d),
                            wref(resident_l(store, "ffn_down", l, dv), mq, d, inter_r),
                        ]
                    })
                    .collect();
                let steps: Vec<Vec<Step>> = (0..self.tp)
                    .map(|r| {
                        let p = &pbs[r];
                        vec![
                            Step::Gemm {
                                w: &w[r][0],
                                x: &p.tmp,
                                y: &p.gate,
                                batch: n,
                            },
                            Step::Gemm {
                                w: &w[r][1],
                                x: &p.tmp,
                                y: &p.up,
                                batch: n,
                            },
                            Step::SiluMul {
                                gate: &p.gate,
                                up: &p.up,
                                out: &p.hidden,
                            },
                            Step::Gemm {
                                w: &w[r][2],
                                x: &p.hidden,
                                y: &p.fo,
                                batch: n,
                            },
                            Step::ResidualAdd {
                                x: &p.x,
                                y: &p.fo,
                                dim: d,
                            },
                        ]
                    })
                    .collect();
                let coll = [
                    TpCollective::None,
                    TpCollective::None,
                    TpCollective::None,
                    TpCollective::AllReduceOut { dim: n * d },
                    TpCollective::None,
                ];
                execute_steps_tp(&self.mesh, &mut self.gpus, &steps, &coll)
                    .map_err(|e| e.to_string())?;
            }
        }

        // Logits handoff: copy the LAST-position row of rank 0's [n×d] residual
        // into rank 0's decode `x` (the buffer `logits()` reads). Then free the
        // per-rank batched buffers.
        {
            let dev0 = self.group[0];
            let g = &mut self.gpus.devices[dev0];
            g.bind_thread().map_err(herr)?;
            g.hip
                .memcpy_dtod_at(
                    &self.ranks[0].x.buf,
                    0,
                    &pbs[0].x.buf,
                    (n - 1) * d * 4,
                    d * 4,
                )
                .map_err(herr)?;
        }
        for (r, b) in pbs.into_iter().enumerate() {
            let g = &mut self.gpus.devices[self.group[r]];
            g.bind_thread().map_err(herr)?;
            for t in [
                b.x,
                b.tmp,
                b.q,
                b.k,
                b.v,
                b.attn,
                b.o,
                b.gate,
                b.up,
                b.hidden,
                b.fo,
                b.positions,
            ] {
                g.free_tensor(t).map_err(herr)?;
            }
        }
        Ok(())
    }

    /// Final norm + lm_head on rank 0 → the vocab logits for predicting the token
    /// after the last `forward_token`.
    pub fn logits(&mut self) -> Result<Vec<f32>, String> {
        let dev0 = self.group[0];
        let g = &mut self.gpus.devices[dev0];
        g.bind_thread().map_err(herr)?;
        let tmp = g.alloc_tensor(&[self.d], DType::F32).map_err(herr)?;
        let logits = g
            .alloc_tensor(&[self.config.vocab_size], DType::F32)
            .map_err(herr)?;
        g.rmsnorm_f32(
            &self.ranks[0].x,
            &self.weights.output_norm,
            &tmp,
            self.config.norm_eps,
        )
        .map_err(herr)?;
        llama::weight_gemv(g, &self.weights.output, &tmp, &logits).map_err(herr)?;
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .map_err(herr)?;
        let out = g.download_f32(&logits).map_err(herr)?;
        let _ = g.free_tensor(tmp);
        let _ = g.free_tensor(logits);
        Ok(out)
    }

    /// Free every GPU allocation this TP model owns (per-rank buffers + KV +
    /// replicated norms + `pos_buf`, the sharded quant store, and rank-0's full
    /// weights), then drain each device pool so the VRAM returns to the system.
    ///
    /// `unload_model`'s prior bare `drop(TpModel)` reclaimed *nothing*: none of
    /// `GpuTensor` / `hip_bridge::DeviceBuffer` / `GpuPool` has a freeing `Drop`,
    /// and `Gpu::drop` only re-binds the device — so every load/unload cycle
    /// leaked the whole model. This mirrors the EP unload path in the loader:
    /// typed frees → `drain_pool` → drop `Gpus`.
    pub fn free(self) {
        // Same free order the construction rollback uses: per-rank buffers +
        // norms + pos_buf + KV on each rank's device, then the sharded store,
        // then rank-0's full weights, then the per-device teardown.
        let TpModel {
            gpus,
            group,
            ranks,
            store,
            weights,
            ..
        } = self;
        let mut gpus = gpus;
        free_tp_resources(
            &mut gpus,
            &group,
            ranks,
            RankBuild::default(),
            Some(store),
            Some(weights),
        );
        teardown_tp_devices(&mut gpus);
        // `gpus` drops here → tears down device contexts.
    }
}

fn herr(e: hip_bridge::HipError) -> String {
    e.to_string()
}

/// Fulfill the dense llama per-layer quant manifest from the HFQ (the bridge).
fn build_store(
    hfq: &HfqFile,
    config: &LlamaConfig,
    mesh: &DeviceMesh,
    gpus: &Gpus,
) -> Result<WeightStore, String> {
    let (d, ff, nh, nkv, hd, n_layers) = (
        config.dim,
        config.hidden_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.n_layers,
    );
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let p_col = ShardPolicy::ColumnShard { axis: 0 };
    let p_row = ShardPolicy::RowShard { axis: 1 };
    let mq = DType::MQ4G256;
    let mut manifest = Vec::with_capacity(n_layers * 7);
    for l in 0..n_layers {
        manifest.push(WeightEntry::layer(
            "wq",
            l,
            vec![q_dim, d],
            mq,
            p_col.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "wk",
            l,
            vec![kv_dim, d],
            mq,
            p_col.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "wv",
            l,
            vec![kv_dim, d],
            mq,
            p_col.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "wo",
            l,
            vec![d, q_dim],
            mq,
            p_row.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "ffn_gate",
            l,
            vec![ff, d],
            mq,
            p_col.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "ffn_up",
            l,
            vec![ff, d],
            mq,
            p_col.clone(),
        ));
        manifest.push(WeightEntry::layer(
            "ffn_down",
            l,
            vec![d, ff],
            mq,
            p_row.clone(),
        ));
    }
    fulfill_manifest(&manifest, mesh, n_layers, gpus, |e| {
        let suffix = match e.name.as_str() {
            "wq" => "self_attn.q_proj",
            "wk" => "self_attn.k_proj",
            "wv" => "self_attn.v_proj",
            "wo" => "self_attn.o_proj",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            _ => "mlp.down_proj",
        };
        let name = format!("model.layers.{}.{suffix}.weight", e.layer.unwrap());
        let (info, bytes) = hfq
            .tensor_data(&name)
            .ok_or_else(|| format!("missing {name}"))?;
        if info.quant_type != MQ4G256_QT {
            return Err(format!(
                "{name} quant_type {} != MQ4G256; dense TP serve requires an mq4 model",
                info.quant_type
            ));
        }
        Ok((bytes.to_vec(), DType::MQ4G256))
    })
    .map_err(|e| format!("fulfill_manifest: {e:?}"))
}

fn wref<'a>(t: &'a GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf: t,
        dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    }
}
fn resident_l<'a>(store: &'a WeightStore, name: &str, layer: usize, dev: usize) -> &'a GpuTensor {
    match store.get(name, Some(layer), dev) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("{name} L{layer} not resident on device {dev}"),
    }
}
fn leak<'a>(w: WeightRef<'a>) -> &'a WeightRef<'a> {
    Box::leak(Box::new(w))
}

/// Test-only one-shot fault hook for the TP constructor (compiled out of
/// production builds). When armed, the next `load_from_hfq_inner` fails
/// deterministically at the post-allocation seam in the constructor — after
/// every owned GPU resource is live — and the arm is consumed, so at most one
/// construction attempt can observe it. Atomic, so full-suite isolation needs
/// no global lock beyond the tests' own `GPU_TEST_LOCK`.
#[cfg(test)]
mod fault_seam {
    use std::sync::atomic::{AtomicBool, Ordering};

    static FAIL_NEXT_LOAD: AtomicBool = AtomicBool::new(false);

    /// Arm the next TP construction to fail at the post-allocation seam.
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

/// Shared GPU observation helpers for the TP/PP construction-rollback tests
/// (`pp_serve.rs` reaches this via `crate::tp_serve::test_support`). Compiled
/// out of production builds.
#[cfg(test)]
pub(crate) mod test_support {
    use rdna_compute::Gpu;
    use std::io::Write;

    /// Serializes the TP/PP rollback tests: device binding, VRAM state and the
    /// `HIPFIRE_EMULATE_GPUS` env var are process-global, so the two tests
    /// (and any future sibling) must not interleave (same pattern as the
    /// per-module `GPU_TEST_LOCK`s elsewhere in the workspace).
    pub static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// VRAM recovery tolerance in bytes — matches the qwen35 rollback tests
    /// (64 MiB) so driver/allocator noise cannot flip the assertion.
    pub const VRAM_TOLERANCE: usize = 64 * 1024 * 1024;

    /// Restores an env var to its prior state on drop (the same guard pattern
    /// as `hipfire-hardware`'s `EnvVarGuard`).
    pub struct EnvGuard {
        key: &'static str,
        prior: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        pub fn set(key: &'static str, value: &str) -> Self {
            let prior = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prior }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(prior) => std::env::set_var(self.key, prior),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// Free VRAM in bytes on `gpu`'s physical device.
    pub fn vram_free(gpu: &Gpu) -> usize {
        gpu.hip.get_vram_info().expect("hipMemGetInfo").0
    }

    /// Dense llama fixture dims. Chosen so a mid-construction leak exceeds
    /// `VRAM_TOLERANCE` on EVERY owning device for both TP(2) and PP(2)
    /// meshes: each layer's weights are ~23 MiB, so PP(2) leaks ~92 MiB per
    /// stage device and TP(2) leaks rank-0 weights + both sharded stores
    /// (~184 MiB per emulated/physical device under `HIPFIRE_EMULATE_GPUS`).
    /// Divisibility: q_dim=2048, kv_dim=512, ff=2048 are all %2==0 and their
    /// tp-shards are %256==0 (MQ4G256 row-shard group alignment).
    pub const DIM: usize = 512;
    pub const FF: usize = 2048;
    pub const N_HEADS: usize = 32;
    pub const N_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 64;
    pub const N_LAYERS: usize = 8;
    pub const VOCAB: usize = 64;
    pub const MAX_SEQ: usize = 64;

    /// Write a minimal dense llama-family HFQ (arch_id 0, qk-norm'd config)
    /// with real-sized zero payloads for every tensor the TP/PP constructors
    /// read: MQ4G256 (qt 13, 4 B/element passthrough) projections and F16
    /// (qt 1) norms/embedding/lm_head. Payload bytes are never inspected
    /// before forward, so zero data loads identically to a real model.
    pub fn write_dense_llama_hfq(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let config_json = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": DIM,
            "intermediate_size": FF,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1_000_000.0,
        });
        let metadata = serde_json::json!({ "config": config_json }).to_string();

        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let mut tensors: Vec<(String, u8, Vec<u32>, usize)> = Vec::new();
        let mut push = |name: String, qt: u8, shape: Vec<u32>| {
            let elems = shape.iter().map(|&d| d as usize).product::<usize>();
            let elem_bytes = if qt == 1 { 2 } else { 4 };
            tensors.push((name, qt, shape, elems * elem_bytes));
        };
        // Model-level tensors (F16 host-decoded to F32 by the loader).
        push(
            "model.embed_tokens.weight".into(),
            1,
            vec![VOCAB as u32, DIM as u32],
        );
        push("model.norm.weight".into(), 1, vec![DIM as u32]);
        push("lm_head.weight".into(), 1, vec![VOCAB as u32, DIM as u32]);
        for l in 0..N_LAYERS {
            push(
                format!("model.layers.{l}.input_layernorm.weight"),
                1,
                vec![DIM as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.q_proj.weight"),
                13,
                vec![q_dim as u32, DIM as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.k_proj.weight"),
                13,
                vec![kv_dim as u32, DIM as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.v_proj.weight"),
                13,
                vec![kv_dim as u32, DIM as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.o_proj.weight"),
                13,
                vec![DIM as u32, q_dim as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.q_norm.weight"),
                1,
                vec![HEAD_DIM as u32],
            );
            push(
                format!("model.layers.{l}.self_attn.k_norm.weight"),
                1,
                vec![HEAD_DIM as u32],
            );
            push(
                format!("model.layers.{l}.post_attention_layernorm.weight"),
                1,
                vec![DIM as u32],
            );
            push(
                format!("model.layers.{l}.mlp.gate_proj.weight"),
                13,
                vec![FF as u32, DIM as u32],
            );
            push(
                format!("model.layers.{l}.mlp.up_proj.weight"),
                13,
                vec![FF as u32, DIM as u32],
            );
            push(
                format!("model.layers.{l}.mlp.down_proj.weight"),
                13,
                vec![DIM as u32, FF as u32],
            );
        }

        // Index: u32 count, then per tensor (name_len u16, name, qt u8, ndim
        // u8, shape u32×ndim, group_size u32, data_size u64). The header's
        // `data_offset` MUST be the first payload's start — `HfqFile` derives
        // every payload offset cumulatively from it — so it is
        // 32 (header) + metadata + index, with payloads written immediately
        // after the index (no padding; the parser does not require any).
        let mut idx = Vec::new();
        idx.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, qt, shape, data_size) in &tensors {
            idx.extend_from_slice(&(name.len() as u16).to_le_bytes());
            idx.extend_from_slice(name.as_bytes());
            idx.push(*qt);
            idx.push(shape.len() as u8);
            for d in shape {
                idx.extend_from_slice(&d.to_le_bytes());
            }
            idx.extend_from_slice(&0u32.to_le_bytes()); // group_size
            idx.extend_from_slice(&(*data_size as u64).to_le_bytes());
        }
        let data_offset = 32u64 + metadata.len() as u64 + idx.len() as u64;

        let path = dir.path().join("dense-llama-tp-pp.hfq");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u32.to_le_bytes()).unwrap(); // arch_id 0 (llama-family)
        f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&32u64.to_le_bytes()).unwrap(); // metadata_offset
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(metadata.as_bytes()).unwrap();
        f.write_all(&idx).unwrap();
        for (_, _, _, data_size) in &tensors {
            f.write_all(&vec![0u8; *data_size]).unwrap();
        }
        f.flush().unwrap();
        path
    }
}

#[allow(clippy::too_many_arguments)]
struct LayerIo<'a> {
    x: &'a GpuTensor,
    tmp: &'a GpuTensor,
    x_rot: &'a GpuTensor,
    q: &'a GpuTensor,
    k: &'a GpuTensor,
    v: &'a GpuTensor,
    attn: &'a GpuTensor,
    o: &'a GpuTensor,
    gate: &'a GpuTensor,
    up: &'a GpuTensor,
    hidden: &'a GpuTensor,
    fo: &'a GpuTensor,
    pos_buf: &'a DeviceBuffer,
    attn_norm: &'a GpuTensor,
    ffn_norm: &'a GpuTensor,
    q_norm: &'a GpuTensor,
    k_norm: &'a GpuTensor,
    wq: WeightRef<'a>,
    wk: WeightRef<'a>,
    wv: WeightRef<'a>,
    wo: WeightRef<'a>,
    w_gate: WeightRef<'a>,
    w_up: WeightRef<'a>,
    w_down: WeightRef<'a>,
    plan: KvTierPlan,
    k_cache: &'a GpuTensor,
    v_cache: &'a GpuTensor,
    partials: &'a GpuTensor,
    nh: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    eps: f32,
    theta: f32,
    qkv_rot: RotationPlan,
    ffn_rot: RotationPlan,
    pos: usize,
    physical_cap: usize,
}

/// The dense-layer 16-op per-rank Step list (mirrors `arch_spec::dense_forward`;
/// the two row-parallel projections wo/down are split into
/// `Gemv → AllReduceOut(idx 8,14) → ResidualAdd`).
fn build_layer_steps(r: LayerIo<'_>) -> Vec<Step<'_>> {
    vec![
        Step::RmsnormAutomatic {
            x: r.x,
            norm_weight: r.attn_norm,
            x_plain: r.tmp,
            out: r.x_rot,
            awq_scale: None,
            k: r.d,
            eps: r.eps,
            rotation: r.qkv_rot,
        },
        Step::Gemv {
            w: leak(r.wq),
            input: GemvInput::Prerotated(r.x_rot),
            out: r.q,
        },
        Step::Gemv {
            w: leak(r.wk),
            input: GemvInput::Prerotated(r.x_rot),
            out: r.k,
        },
        Step::Gemv {
            w: leak(r.wv),
            input: GemvInput::Prerotated(r.x_rot),
            out: r.v,
        },
        Step::QkNorm {
            x: r.q,
            weight: r.q_norm,
            n_groups: r.nh,
            head_dim: r.hd,
            eps: r.eps,
        },
        Step::QkNorm {
            x: r.k,
            weight: r.k_norm,
            n_groups: r.nkv,
            head_dim: r.hd,
            eps: r.eps,
        },
        Step::Rope {
            q: r.q,
            k: r.k,
            pos_buf: r.pos_buf,
            n_heads: r.nh,
            n_kv_heads: r.nkv,
            head_dim: r.hd,
            theta: r.theta,
        },
        Step::Attend {
            plan: r.plan,
            io: AttnParams {
                q: r.q,
                k: r.k,
                v: r.v,
                k_cache: r.k_cache,
                v_cache: r.v_cache,
                k_scales: None,
                v_scales: None,
                pos_buf: r.pos_buf,
                pos: r.pos,
                positions: None,
                n_heads: r.nh,
                n_kv_heads: r.nkv,
                head_dim: r.hd,
                physical_cap: r.physical_cap,
                output_gate: None,
                batch_size: 1,
                max_ctx_len: 0,
                flash_partials: Some(r.partials),
                givens_cos: None,
                givens_sin: None,
                tree_bias: None,
                block_start: 0,
                block_cols: 0,
                output: r.attn,
            },
        },
        Step::Gemv {
            w: leak(r.wo),
            input: GemvInput::Raw(r.attn),
            out: r.o,
        },
        Step::ResidualAdd {
            x: r.x,
            y: r.o,
            dim: r.d,
        },
        Step::RmsnormAutomatic {
            x: r.x,
            norm_weight: r.ffn_norm,
            x_plain: r.tmp,
            out: r.x_rot,
            awq_scale: None,
            k: r.d,
            eps: r.eps,
            rotation: r.ffn_rot,
        },
        Step::Gemv {
            w: leak(r.w_gate),
            input: GemvInput::Prerotated(r.x_rot),
            out: r.gate,
        },
        Step::Gemv {
            w: leak(r.w_up),
            input: GemvInput::Prerotated(r.x_rot),
            out: r.up,
        },
        Step::SiluMul {
            gate: r.gate,
            up: r.up,
            out: r.hidden,
        },
        Step::Gemv {
            w: leak(r.w_down),
            input: GemvInput::Raw(r.hidden),
            out: r.fo,
        },
        Step::ResidualAdd {
            x: r.x,
            y: r.fo,
            dim: r.d,
        },
    ]
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
        let data_offset = index_offset + 4; // just n_tensors field
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_tensors=0
        f.flush().unwrap();
        path
    }

    #[test]
    fn load_from_hfq_rejects_non_llama_arch_id_before_gpu() {
        let dir = tempfile::tempdir().unwrap();
        // DeepSeek4 (arch_id=9) is not llama-family (0|1 expected)
        let path = write_minimal_hfq(
            &dir,
            "ds4.hfq",
            9,
            r#"{"config":{"model_type":"deepseek_v4","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}}"#,
        );
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let err = match TpModel::load_from_hfq_inner(&hfq, &mesh, 64) {
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
    fn load_from_hfq_rejects_tp_below_two() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_minimal_hfq(
            &dir,
            "llama.hfq",
            0,
            r#"{"config":{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"intermediate_size":11008,"vocab_size":32000}}"#,
        );
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 1)]);
        let err = match TpModel::load_from_hfq_inner(&hfq, &mesh, 64) {
            Err(e) => e,
            Ok(_) => panic!("expected error for tp<2, got Ok"),
        };
        assert!(
            err.contains("Tp axis with size>=2"),
            "expected tp>=2 error, got: {err}"
        );
    }

    #[test]
    fn load_from_hfq_rejects_missing_qk_norm() {
        let dir = tempfile::tempdir().unwrap();
        // Llama with arch_id=0, no q_norm tensor → passes arch_id check
        // but fails has_qk_norm. Use a proper llama config.
        let path = write_minimal_hfq(
            &dir,
            "llama_nq.hfq",
            1, // PlainQwen3 — has arch_id in 0|1
            r#"{"config":{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"intermediate_size":8192,"vocab_size":32000}}"#,
        );
        let hfq = HfqFile::open(&path).unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        let err = match TpModel::load_from_hfq_inner(&hfq, &mesh, 64) {
            Err(e) => e,
            Ok(_) => panic!("expected qk-norm error for no-q_norm model, got Ok"),
        };
        // arch_id=1 is in [0,1], but no q_norm tensor → qk-norm error
        assert!(
            err.contains("qk-norm"),
            "expected qk-norm error for no-q_norm model, got: {err}"
        );
    }

    /// Oracle gate-1 blocker 5 (TP): construction lacks rollback ownership —
    /// a fallible step after GPU allocations (here: the post-allocation seam
    /// before `enable_peer_all`) leaks every earlier owned resource. The arm
    /// fires only after all of rank-0's full weights, the sharded store and
    /// every rank's KV + scratch are live, so a correct rollback must free
    /// them all on their owning devices. Fails today: those allocations stay
    /// hipMalloc'd and each device's free VRAM stays below the warm baseline.
    #[test]
    fn failed_construction_reclaims_every_owned_gpu_allocation() {
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
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]);

        // Probe handles for per-device VRAM observation — same mesh/degree as
        // the constructor's `Gpus`, kept alive across baseline and post-failure
        // reads so its own context overhead cancels out.
        let mut probe =
            Gpus::from_mesh(&mesh, test_support::N_LAYERS).expect("probe Gpus must bind");

        // Warm-up cycle: one successful load + free absorbs one-time driver /
        // kernel residency, so the measured baseline is stable.
        {
            let model = TpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ)
                .expect("warm-up TP load must succeed");
            model.free();
        }
        for dev in probe.devices.iter_mut() {
            dev.drain_pool();
        }
        let baseline: Vec<usize> = probe.devices.iter().map(test_support::vram_free).collect();

        // Forced mid-construction failure at the post-allocation seam.
        fault_seam::reset();
        fault_seam::arm_fail_next_load();
        let err = match TpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ) {
            Ok(_) => panic!("armed TP fault must fail the construction"),
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
                "device {d}: VRAM not reclaimed after failed TP construction \
                 (Oracle gate-1 blocker 5): baseline={base} after={after} \
                 delta={} — every owned allocation must be freed on its device",
                base.saturating_sub(after)
            );
        }

        // Success cycle after the failure: the GPU must remain usable and a
        // full load + free must return VRAM to the same baseline.
        {
            let model = TpModel::load_from_hfq_inner(&hfq, &mesh, test_support::MAX_SEQ)
                .expect("TP load must succeed after the forced failure");
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
