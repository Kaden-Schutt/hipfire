// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP4c increment A: a **whole REAL quant transformer layer (attn + FFN)**
//! through the store→forward bridge + `execute_steps_tp` == single-device. This
//! is the core of `dense_forward_tp`: it composes every validated primitive on a
//! real layer-0 of `qwen3-0.6b-llama.mq4` —
//!   - MQ4G256 QKV / O / gate / up / down sharded by the llama `weight_manifest`
//!     (`fulfill_manifest`, the real bridge: Column qkv+gate+up, Row wo+down),
//!   - Qwen3 per-head Q/K RMSNorm (`Step::QkNorm`) on owned heads,
//!   - RoPE + a first-class `Step::Attend` (Q8 KV tier) on per-rank owned heads,
//!   - row-parallel wo/down → `AllReduceOut` → `Step::ResidualAdd` (PB-TP4a).
//!
//! The per-rank layer Step list (16 ops) mirrors `dense_forward` (arch_spec.rs)
//! exactly, with the two `GemvResidual` row ops (wo, down) split into
//! `Gemv → AllReduceOut → ResidualAdd`. The single-device reference runs the SAME
//! Step list one op at a time (`execute_steps(&[step])` never fuses a lone step)
//! on the WHOLE weights → op-for-op identical kernels, so any delta is sharding,
//! not fusion.
//!
//! HIST=0 (decode the first token; `Step::Attend` writes it, attends over pos 0):
//! this validates the bridge + the full per-layer TP construction on real
//! weights. Multi-key Q8 attention under TP is exercised by the full-model
//! prefill (increment B); the attention MATH head-parallel was validated
//! multi-key in PB-TP4b. Validated on emulated Tp-2 (gfx1151).
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_quant_layer_parity [model.mq4]

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::llama::{KvCache, LlamaConfig, LlamaWeights};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};

const TP: usize = 2;
const MAX_SEQ: usize = 64;
const TOL: f32 = 1e-3;
const POS: usize = 0;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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
fn resident<'a>(store: &'a WeightStore, name: &str, dev: usize) -> &'a GpuTensor {
    match store.get(name, Some(0), dev) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("{name} not resident on device {dev}"),
    }
}
/// Q8 KV tier inputs for a decode at `pos` (mirrors llama's attend_plan).
fn q8_tier(kv: &KvCache, pos: usize) -> KvTierInputs {
    KvTierInputs {
        pos,
        ..kv.tier_inputs()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_execute_steps_quant_layer_parity: SKIPPED (no {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Load the whole model onto device 0 (reference weights + HFQ for the bridge).
    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(model_path)).expect("open model");
    let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let weights: LlamaWeights = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, g).expect("load_weights_hfq")
    };
    let (d, ff, nh, nkv, hd) = (
        config.dim,
        config.hidden_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
    );
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let eps = config.norm_eps;
    let theta = config.rope_freq_base;
    let qkv_rot = dtype_rotation_plan(weights.layers[0].wq.gpu_dtype);
    let ffn_rot = dtype_rotation_plan(weights.layers[0].w_gate.gpu_dtype);
    let x_rot_cap = d.max(ff);
    eprintln!(
        "layer-0: d={d} ff={ff} nh={nh} nkv={nkv} hd={hd} qkv_rot={qkv_rot:?} qk_norm={}",
        config.has_qk_norm
    );

    // Deterministic hidden (replicated).
    let x0: Vec<f32> = (0..d).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let x_bytes: Vec<u8> = x0.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let chunks = MAX_SEQ.div_ceil(128);

    // ── Single-device reference: SAME Step list, one op at a time (no fusion) ──
    let x_ref = {
        let l0 = &weights.layers[0];
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let ctx = DispatchCtx::new(g);
        let x = g.upload_raw(&x_bytes, &[d]).unwrap();
        let tmp = g.alloc_tensor(&[d], DType::F32).unwrap();
        let x_rot = g.alloc_tensor(&[x_rot_cap], DType::F32).unwrap();
        let q = g.alloc_tensor(&[q_dim], DType::F32).unwrap();
        let k = g.alloc_tensor(&[kv_dim], DType::F32).unwrap();
        let v = g.alloc_tensor(&[kv_dim], DType::F32).unwrap();
        let attn = g.alloc_tensor(&[q_dim], DType::F32).unwrap();
        let o = g.alloc_tensor(&[d], DType::F32).unwrap();
        let gate = g.alloc_tensor(&[ff], DType::F32).unwrap();
        let up = g.alloc_tensor(&[ff], DType::F32).unwrap();
        let hidden = g.alloc_tensor(&[ff], DType::F32).unwrap();
        let fo = g.alloc_tensor(&[d], DType::F32).unwrap();
        let partials = g
            .alloc_tensor(&[nh * chunks * (2 + hd)], DType::F32)
            .unwrap();
        let pos_buf = g.hip.malloc(4).unwrap();
        g.hip
            .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
            .unwrap();
        let mut kv = KvCache::new_gpu_q8(g, 1, nkv, hd, MAX_SEQ).unwrap();

        let steps = build_layer_steps(LayerRefs {
            x: &x,
            tmp: &tmp,
            x_rot: &x_rot,
            q: &q,
            k: &k,
            v: &v,
            attn: &attn,
            o: &o,
            gate: &gate,
            up: &up,
            hidden: &hidden,
            fo: &fo,
            pos_buf: &pos_buf,
            attn_norm: &l0.attn_norm,
            ffn_norm: &l0.ffn_norm,
            q_norm: l0.q_norm.as_ref(),
            k_norm: l0.k_norm.as_ref(),
            wq: l0.wq.dispatch_ref(),
            wk: l0.wk.dispatch_ref(),
            wv: l0.wv.dispatch_ref(),
            wo: l0.wo.dispatch_ref(),
            w_gate: l0.w_gate.dispatch_ref(),
            w_up: l0.w_up.dispatch_ref(),
            w_down: l0.w_down.dispatch_ref(),
            plan: KvTierPlan::derive(q8_tier(&kv, POS)).unwrap(),
            k_cache: &kv.k_gpu[0],
            v_cache: &kv.v_gpu[0],
            partials: &partials,
            nh,
            nkv,
            hd,
            d,
            eps,
            theta,
            qkv_rot,
            ffn_rot,
        });
        // Run each step alone → launch_op path, no cross-step fusion.
        for s in &steps {
            execute_steps(g, &ctx, std::slice::from_ref(s)).expect("ref step");
        }
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let _ = &mut kv;
        g.download_f32(&x).unwrap()
    };

    // ── Store→forward bridge: shard layer-0 quant weights (Tp-2) ──
    let p_col = ShardPolicy::ColumnShard { axis: 0 };
    let p_row = ShardPolicy::RowShard { axis: 1 };
    let mq = DType::MQ4G256;
    let manifest = vec![
        WeightEntry::layer("wq", 0, vec![q_dim, d], mq, p_col.clone()),
        WeightEntry::layer("wk", 0, vec![kv_dim, d], mq, p_col.clone()),
        WeightEntry::layer("wv", 0, vec![kv_dim, d], mq, p_col.clone()),
        WeightEntry::layer("wo", 0, vec![d, q_dim], mq, p_row.clone()),
        WeightEntry::layer("ffn_gate", 0, vec![ff, d], mq, p_col.clone()),
        WeightEntry::layer("ffn_up", 0, vec![ff, d], mq, p_col.clone()),
        WeightEntry::layer("ffn_down", 0, vec![d, ff], mq, p_row.clone()),
    ];
    let name_on_disk = |n: &str| -> &'static str {
        match n {
            "wq" => "model.layers.0.self_attn.q_proj.weight",
            "wk" => "model.layers.0.self_attn.k_proj.weight",
            "wv" => "model.layers.0.self_attn.v_proj.weight",
            "wo" => "model.layers.0.self_attn.o_proj.weight",
            "ffn_gate" => "model.layers.0.mlp.gate_proj.weight",
            "ffn_up" => "model.layers.0.mlp.up_proj.weight",
            _ => "model.layers.0.mlp.down_proj.weight",
        }
    };
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let (info, bytes) = hfq.tensor_data(name_on_disk(&e.name)).unwrap();
        assert_eq!(info.quant_type, 13, "{} not MQ4G256", e.name);
        Ok((bytes.to_vec(), DType::MQ4G256))
    })
    .expect("shard weights");

    // Replicated norms: download the loaded F32 norms, upload to every rank.
    let an0 = gpus.devices[0]
        .download_f32(&weights.layers[0].attn_norm)
        .unwrap();
    let fn0 = gpus.devices[0]
        .download_f32(&weights.layers[0].ffn_norm)
        .unwrap();
    let qn0 = weights.layers[0]
        .q_norm
        .as_ref()
        .map(|t| gpus.devices[0].download_f32(t).unwrap());
    let kn0 = weights.layers[0]
        .k_norm
        .as_ref()
        .map(|t| gpus.devices[0].download_f32(t).unwrap());

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let hpr = nh / TP;
    let kvpr = nkv / TP;
    let q_dim_r = hpr * hd;
    let kv_dim_r = kvpr * hd;
    let inter_r = ff / TP;

    struct RankBufs {
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
        pos_buf: hip_bridge::DeviceBuffer,
        attn_norm: GpuTensor,
        ffn_norm: GpuTensor,
        q_norm: Option<GpuTensor>,
        k_norm: Option<GpuTensor>,
        kv: KvCache,
    }
    let up_f32 = |g: &Gpu, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
        g.upload_raw(&b, &[v.len()]).unwrap()
    };
    let mut bufs: Vec<RankBufs> = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        let pos_buf = g.hip.malloc(4).unwrap();
        g.hip
            .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
            .unwrap();
        let kv = KvCache::new_gpu_q8(g, 1, kvpr, hd, MAX_SEQ).unwrap();
        bufs.push(RankBufs {
            x: g.upload_raw(&x_bytes, &[d]).unwrap(),
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
            pos_buf,
            attn_norm: up_f32(g, &an0),
            ffn_norm: up_f32(g, &fn0),
            q_norm: qn0.as_ref().map(|v| up_f32(g, v)),
            k_norm: kn0.as_ref().map(|v| up_f32(g, v)),
            kv,
        });
    }

    // Per-rank sharded WeightRefs from the store (built inline; WeightRef isn't Clone).
    let per_rank_steps: Vec<Vec<Step>> = (0..TP)
        .map(|r| {
            let dv = group[r];
            build_layer_steps(LayerRefs {
                x: &bufs[r].x,
                tmp: &bufs[r].tmp,
                x_rot: &bufs[r].x_rot,
                q: &bufs[r].q,
                k: &bufs[r].k,
                v: &bufs[r].v,
                attn: &bufs[r].attn,
                o: &bufs[r].o,
                gate: &bufs[r].gate,
                up: &bufs[r].up,
                hidden: &bufs[r].hidden,
                fo: &bufs[r].fo,
                pos_buf: &bufs[r].pos_buf,
                attn_norm: &bufs[r].attn_norm,
                ffn_norm: &bufs[r].ffn_norm,
                q_norm: bufs[r].q_norm.as_ref(),
                k_norm: bufs[r].k_norm.as_ref(),
                wq: wref(resident(&store, "wq", dv), mq, q_dim_r, d),
                wk: wref(resident(&store, "wk", dv), mq, kv_dim_r, d),
                wv: wref(resident(&store, "wv", dv), mq, kv_dim_r, d),
                wo: wref(resident(&store, "wo", dv), mq, d, q_dim_r),
                w_gate: wref(resident(&store, "ffn_gate", dv), mq, inter_r, d),
                w_up: wref(resident(&store, "ffn_up", dv), mq, inter_r, d),
                w_down: wref(resident(&store, "ffn_down", dv), mq, d, inter_r),
                plan: KvTierPlan::derive(q8_tier(&bufs[r].kv, POS)).unwrap(),
                k_cache: &bufs[r].kv.k_gpu[0],
                v_cache: &bufs[r].kv.v_gpu[0],
                partials: &bufs[r].partials,
                nh: hpr,
                nkv: kvpr,
                hd,
                d,
                eps,
                theta,
                qkv_rot,
                ffn_rot,
            })
        })
        .collect();

    // Collectives: only the two row-parallel projections (wo @ idx 8, down @ idx 14).
    let mut collectives: Vec<TpCollective> = (0..16).map(|_| TpCollective::None).collect();
    collectives[8] = TpCollective::AllReduceOut { dim: d };
    collectives[14] = TpCollective::AllReduceOut { dim: d };
    assert_eq!(
        per_rank_steps[0].len(),
        16,
        "layer step count changed — update collective indices"
    );

    execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");

    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let x_tp = gpus.devices[dev0].download_f32(&bufs[0].x).unwrap();

    let diff = max_abs_diff(&x_tp, &x_ref);
    let mag = x_ref.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!(
        "[tp-quant-layer] real MQ4G256 layer (attn+FFN) through store->bridge + execute_steps_tp \
         vs single-device: max|Δ|={diff:.3e} (ref max|x|={mag:.3e})"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    assert!(
        diff < TOL,
        "TP quant layer diverges from single-device: max|Δ|={diff}"
    );
    println!(
        "tp_execute_steps_quant_layer_parity: real quant transformer layer through the \
         store->forward bridge + execute_steps_tp == single-device — PB-TP4c increment A validated"
    );
}

/// Everything one layer's Step list borrows (whole or per-rank).
struct LayerRefs<'a> {
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
    pos_buf: &'a hip_bridge::DeviceBuffer,
    attn_norm: &'a GpuTensor,
    ffn_norm: &'a GpuTensor,
    q_norm: Option<&'a GpuTensor>,
    k_norm: Option<&'a GpuTensor>,
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
}

/// The dense-layer Step list (mirrors `dense_forward`), with the two row-parallel
/// projections (wo, down) split into `Gemv → [AllReduceOut] → ResidualAdd` per
/// PB-TP4a. Owns `r.plan`/`r.wq`.. by move; the caller builds one `LayerRefs`
/// per rank. Fixed length 16 (indices 8 = wo, 14 = down are the row ops).
fn build_layer_steps(r: LayerRefs<'_>) -> Vec<Step<'_>> {
    let mut steps = Vec::with_capacity(16);
    // Attention block.
    steps.push(Step::RmsnormAutomatic {
        x: r.x,
        norm_weight: r.attn_norm,
        x_plain: r.tmp,
        out: r.x_rot,
        awq_scale: None,
        k: r.d,
        eps: r.eps,
        rotation: r.qkv_rot,
    });
    steps.push(Step::Gemv {
        w: leak(r.wq),
        input: GemvInput::Prerotated(r.x_rot),
        out: r.q,
    });
    steps.push(Step::Gemv {
        w: leak(r.wk),
        input: GemvInput::Prerotated(r.x_rot),
        out: r.k,
    });
    steps.push(Step::Gemv {
        w: leak(r.wv),
        input: GemvInput::Prerotated(r.x_rot),
        out: r.v,
    });
    steps.push(Step::QkNorm {
        x: r.q,
        weight: r.q_norm.expect("qwen3 q_norm"),
        n_groups: r.nh,
        head_dim: r.hd,
        eps: r.eps,
    });
    steps.push(Step::QkNorm {
        x: r.k,
        weight: r.k_norm.expect("qwen3 k_norm"),
        n_groups: r.nkv,
        head_dim: r.hd,
        eps: r.eps,
    });
    steps.push(Step::Rope {
        q: r.q,
        k: r.k,
        pos_buf: r.pos_buf,
        n_heads: r.nh,
        n_kv_heads: r.nkv,
        head_dim: r.hd,
        theta: r.theta,
    });
    steps.push(Step::Attend {
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
            pos: POS,
            positions: None,
            n_heads: r.nh,
            n_kv_heads: r.nkv,
            head_dim: r.hd,
            physical_cap: MAX_SEQ,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(r.partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: r.attn,
        },
    });
    // Row-parallel o_proj → all-reduce (idx 8) → residual.
    steps.push(Step::Gemv {
        w: leak(r.wo),
        input: GemvInput::Raw(r.attn),
        out: r.o,
    });
    steps.push(Step::ResidualAdd {
        x: r.x,
        y: r.o,
        dim: r.d,
    });
    // FFN block.
    steps.push(Step::RmsnormAutomatic {
        x: r.x,
        norm_weight: r.ffn_norm,
        x_plain: r.tmp,
        out: r.x_rot,
        awq_scale: None,
        k: r.d,
        eps: r.eps,
        rotation: r.ffn_rot,
    });
    steps.push(Step::Gemv {
        w: leak(r.w_gate),
        input: GemvInput::Prerotated(r.x_rot),
        out: r.gate,
    });
    steps.push(Step::Gemv {
        w: leak(r.w_up),
        input: GemvInput::Prerotated(r.x_rot),
        out: r.up,
    });
    steps.push(Step::SiluMul {
        gate: r.gate,
        up: r.up,
        out: r.hidden,
    });
    // Row-parallel down → all-reduce (idx 14) → residual.
    steps.push(Step::Gemv {
        w: leak(r.w_down),
        input: GemvInput::Raw(r.hidden),
        out: r.fo,
    });
    steps.push(Step::ResidualAdd {
        x: r.x,
        y: r.fo,
        dim: r.d,
    });
    steps
}

/// `Step::Gemv` borrows `&WeightRef`; the LayerRefs owns the WeightRefs by value.
/// Leak them for the lifetime of this example (one layer, one forward) so the
/// Step list can hold `&'_ WeightRef`. (A real `dense_forward_tp` keeps the
/// per-rank WeightRefs in a Vec that outlives the Step list; here leaking keeps
/// the example flat.)
fn leak<'a>(w: WeightRef<'a>) -> &'a WeightRef<'a> {
    Box::leak(Box::new(w))
}
