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
//! `Tp` mesh. Prefill is per-token (no batched prefill yet); each request starts
//! at pos 0 (stateless — no multi-turn KV reuse).

use crate::hfq::HfqFile;
use crate::llama::{self, KvCache, LlamaConfig, LlamaWeights};
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

    /// Load a dense llama-family HFQ tensor-parallel across `tp` ranks.
    pub fn load(path: &str, tp: usize, max_seq: usize) -> Result<Self, String> {
        if tp < 2 {
            return Err(format!("TpModel::load needs tp>=2 (got {tp})"));
        }
        let hfq = HfqFile::open(std::path::Path::new(path)).map_err(|e| format!("{e}"))?;
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

        let mut gpus =
            Gpus::init_uniform(tp, n_layers).map_err(|e| format!("init_uniform: {e:?}"))?;
        gpus.enable_peer_all()
            .map_err(|e| format!("enable_peer_all: {e:?}"))?;
        for dev in gpus.devices.iter_mut() {
            dev.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
            let s = dev
                .hip
                .stream_create()
                .map_err(|e| format!("stream_create: {e:?}"))?;
            dev.active_stream = Some(s);
        }

        // Whole weights on rank 0 (embed / final-norm / lm_head + F32 norm source).
        let weights = {
            let g = &mut gpus.devices[0];
            g.bind_thread().map_err(|e| format!("bind0: {e:?}"))?;
            crate::hfq::load_weights_hfq(&hfq, &config, g)
                .map_err(|e| format!("load_weights: {e:?}"))?
        };
        let qkv_rot = dtype_rotation_plan(weights.layers[0].wq.gpu_dtype);
        let ffn_rot = dtype_rotation_plan(weights.layers[0].w_gate.gpu_dtype);

        // Store→forward bridge: shard every layer's quant weights.
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, tp)]);
        let store = build_store(&hfq, &config, &mesh, &gpus)?;

        // Per-layer replicated norm CPU copies (download once from rank 0).
        let norms_cpu: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|l| {
                let g = &gpus.devices[0];
                let lw = &weights.layers[l];
                (
                    g.download_f32(&lw.attn_norm).unwrap(),
                    g.download_f32(&lw.ffn_norm).unwrap(),
                    g.download_f32(lw.q_norm.as_ref().unwrap()).unwrap(),
                    g.download_f32(lw.k_norm.as_ref().unwrap()).unwrap(),
                )
            })
            .collect();

        let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
        let (hpr, kvpr) = (nh / tp, nkv / tp);
        let (q_dim_r, kv_dim_r, inter_r) = (hpr * hd, kvpr * hd, ff / tp);
        let x_rot_cap = d.max(ff);
        let phys_cap = max_seq;
        let chunks = phys_cap.div_ceil(128);

        let up_f32 = |g: &Gpu, v: &[f32]| {
            let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
            g.upload_raw(&b, &[v.len()]).unwrap()
        };
        let mut ranks = Vec::with_capacity(tp);
        for &dev in &group {
            let g = &mut gpus.devices[dev];
            g.bind_thread().map_err(|e| format!("bind{dev}: {e:?}"))?;
            let kv = KvCache::new_gpu_q8(g, n_layers, kvpr, hd, phys_cap)
                .map_err(|e| format!("kv: {e:?}"))?;
            let norms = norms_cpu
                .iter()
                .map(|(a, f, q, k)| (up_f32(g, a), up_f32(g, f), up_f32(g, q), up_f32(g, k)))
                .collect();
            ranks.push(TpRank {
                x: g.alloc_tensor(&[d], DType::F32).unwrap(),
                tmp: g.alloc_tensor(&[d], DType::F32).unwrap(),
                x_rot: g.alloc_tensor(&[x_rot_cap], DType::F32).unwrap(),
                q: g.alloc_tensor(&[q_dim_r], DType::F32).unwrap(),
                k: g.alloc_tensor(&[kv_dim_r], DType::F32).unwrap(),
                v: g.alloc_tensor(&[kv_dim_r], DType::F32).unwrap(),
                attn: g.alloc_tensor(&[q_dim_r], DType::F32).unwrap(),
                o: g.alloc_tensor(&[d], DType::F32).unwrap(),
                gate: g.alloc_tensor(&[inter_r], DType::F32).unwrap(),
                up: g.alloc_tensor(&[inter_r], DType::F32).unwrap(),
                hidden: g.alloc_tensor(&[inter_r], DType::F32).unwrap(),
                fo: g.alloc_tensor(&[d], DType::F32).unwrap(),
                partials: g
                    .alloc_tensor(&[hpr * chunks * (2 + hd)], DType::F32)
                    .unwrap(),
                pos_buf: g.hip.malloc(4).unwrap(),
                kv,
                norms,
            });
        }

        let mut collectives: Vec<TpCollective> = (0..16).map(|_| TpCollective::None).collect();
        collectives[8] = TpCollective::AllReduceOut { dim: d };
        collectives[14] = TpCollective::AllReduceOut { dim: d };

        Ok(TpModel {
            gpus,
            mesh,
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
