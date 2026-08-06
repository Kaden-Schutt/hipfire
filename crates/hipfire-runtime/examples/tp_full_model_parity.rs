// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP4c increment B: **full-model tensor-parallel forward == single-GPU**.
//! Embed → all 28 sharded transformer layers (via `execute_steps_tp`) → final
//! norm → lm_head → logits, on a real `qwen3-0.6b-llama.mq4`, compared to the
//! production `llama::forward_scratch` single-GPU path. This is the capstone of
//! the store→forward bridge + `dense_forward_tp`: the whole model runs
//! tensor-parallel over an emulated Tp-2 mesh.
//!
//! - The store→forward bridge (`fulfill_manifest`) shards EVERY layer's MQ4G256
//!   QKV/O/gate/up/down (Column qkv+gate+up, Row wo+down); replicated norms are
//!   uploaded to each rank from the loaded F32 weights.
//! - Each layer is the validated 16-op per-rank Step list (increment A); the
//!   residual hidden `x` is kept replicated across ranks (every all-reduce +
//!   replicated ResidualAdd/Rmsnorm keeps it in sync), so embed (rank 0, then
//!   broadcast once) and the final norm + lm_head (rank 0) need no sharding.
//!
//! Single decode position (pos 0): validates the full embed→layers→lm_head stack
//! end-to-end. Multi-key attention under TP was validated in PB-TP4b; per-layer
//! q/k/o/ffn sharding in increment A. The reference (`forward_scratch`) is the
//! FUSED production path, so logits match to a fusion tolerance; the invariant
//! that matters — the greedy argmax token — must be identical. Emulated Tp-2, gfx1151.
//!
//! Run: HIP_VISIBLE_DEVICES=0 HIPFIRE_DETERMINISTIC=1 \
//!      cargo run -p hipfire-runtime --release --example tp_full_model_parity [model.mq4] [token]

use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::llama::{self, ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};

const TP: usize = 2;
const MAX_SEQ: usize = 64;
const POS: usize = 0;

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
fn q8_tier(kv: &KvCache, pos: usize) -> KvTierInputs {
    KvTierInputs {
        pos,
        ..kv.tier_inputs()
    }
}
fn leak<'a>(w: WeightRef<'a>) -> &'a WeightRef<'a> {
    Box::leak(Box::new(w))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let token: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9707); // "Hello"

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_full_model_parity: SKIPPED (no {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(model_path)).expect("open model");
    let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let weights: LlamaWeights = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, g).expect("load_weights_hfq")
    };
    let (d, ff, nh, nkv, hd, n_layers) = (
        config.dim,
        config.hidden_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.n_layers,
    );
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let eps = config.norm_eps;
    let theta = config.rope_freq_base;
    let qkv_rot = dtype_rotation_plan(weights.layers[0].wq.gpu_dtype);
    let ffn_rot = dtype_rotation_plan(weights.layers[0].w_gate.gpu_dtype);
    eprintln!("model: d={d} ff={ff} nh={nh} nkv={nkv} hd={hd} layers={n_layers} token={token}");

    // ── Reference: production single-GPU forward_scratch (pos 0) ──
    let (logits_ref, argmax_ref) = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let mut kv = KvCache::new_gpu_q8(g, n_layers, nkv, hd, MAX_SEQ).unwrap();
        let scratch = ForwardScratch::new(g, &config).unwrap();
        llama::forward_scratch(
            g, &weights, &config, token, POS, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
        )
        .expect("forward_scratch");
        let l = g.download_f32(&scratch.logits).unwrap();
        let am = llama::argmax(&l);
        (l, am)
    };

    // ── Store→forward bridge: shard EVERY layer's quant weights (Tp-2) ──
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
    let disk = |name: &str, l: usize| -> String {
        let suffix = match name {
            "wq" => "self_attn.q_proj",
            "wk" => "self_attn.k_proj",
            "wv" => "self_attn.v_proj",
            "wo" => "self_attn.o_proj",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            _ => "mlp.down_proj",
        };
        format!("model.layers.{l}.{suffix}.weight")
    };
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, n_layers, &gpus, |e| {
        let (info, bytes) = hfq.tensor_data(&disk(&e.name, e.layer.unwrap())).unwrap();
        assert_eq!(info.quant_type, 13, "{} not MQ4G256", e.name);
        Ok((bytes.to_vec(), DType::MQ4G256))
    })
    .expect("shard weights");

    // Replicated per-layer norms: download loaded F32, upload to every rank.
    let up_f32 = |g: &Gpu, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
        g.upload_raw(&b, &[v.len()]).unwrap()
    };
    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let hpr = nh / TP;
    let kvpr = nkv / TP;
    let q_dim_r = hpr * hd;
    let kv_dim_r = kvpr * hd;
    let inter_r = ff / TP;
    let x_rot_cap = d.max(ff);
    let chunks = MAX_SEQ.div_ceil(128);

    // Per-layer norm CPU copies (download once from rank 0).
    struct LayerNormCpu {
        attn: Vec<f32>,
        ffn: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
    }
    let norms_cpu: Vec<LayerNormCpu> = (0..n_layers)
        .map(|l| {
            let g = &gpus.devices[0];
            LayerNormCpu {
                attn: g.download_f32(&weights.layers[l].attn_norm).unwrap(),
                ffn: g.download_f32(&weights.layers[l].ffn_norm).unwrap(),
                q: g.download_f32(weights.layers[l].q_norm.as_ref().unwrap())
                    .unwrap(),
                k: g.download_f32(weights.layers[l].k_norm.as_ref().unwrap())
                    .unwrap(),
            }
        })
        .collect();

    // Per-rank persistent buffers + KV + per-layer norm tensors.
    struct RankState {
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
        kv: KvCache,
        norms: Vec<(GpuTensor, GpuTensor, GpuTensor, GpuTensor)>, // per layer: attn,ffn,q,k
    }
    let mut st: Vec<RankState> = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        let pos_buf = g.hip.malloc(4).unwrap();
        g.hip
            .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
            .unwrap();
        let kv = KvCache::new_gpu_q8(g, n_layers, kvpr, hd, MAX_SEQ).unwrap();
        let norms = norms_cpu
            .iter()
            .map(|n| {
                (
                    up_f32(g, &n.attn),
                    up_f32(g, &n.ffn),
                    up_f32(g, &n.q),
                    up_f32(g, &n.k),
                )
            })
            .collect();
        st.push(RankState {
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
            pos_buf,
            kv,
            norms,
        });
    }

    // ── Embed on rank 0, broadcast the replicated hidden to every rank ──
    {
        let g = &mut gpus.devices[group[0]];
        g.bind_thread().unwrap();
        llama::embedding_lookup_dispatch(
            g,
            weights.embd_format,
            &weights.token_embd,
            &st[0].x,
            token,
            d,
        )
        .expect("embed");
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
    }
    let x0_cpu = gpus.devices[group[0]].download_f32(&st[0].x).unwrap();
    let x0_bytes: Vec<u8> = x0_cpu.iter().flat_map(|f| f.to_ne_bytes()).collect();
    for (r, &dev) in group.iter().enumerate() {
        if r == 0 {
            continue;
        }
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        g.hip.memcpy_htod(&st[r].x.buf, &x0_bytes).unwrap();
    }

    // ── Per-layer TP forward: build the 16-op Step list per rank, run it ──
    let mut collectives: Vec<TpCollective> = (0..16).map(|_| TpCollective::None).collect();
    collectives[8] = TpCollective::AllReduceOut { dim: d };
    collectives[14] = TpCollective::AllReduceOut { dim: d };

    for l in 0..n_layers {
        let per_rank_steps: Vec<Vec<Step>> = (0..TP)
            .map(|r| {
                let dv = group[r];
                let s = &st[r];
                let (an, fnrm, qn, kn) = &s.norms[l];
                build_layer_steps(
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.q,
                    &s.k,
                    &s.v,
                    &s.attn,
                    &s.o,
                    &s.gate,
                    &s.up,
                    &s.hidden,
                    &s.fo,
                    &s.pos_buf,
                    an,
                    fnrm,
                    qn,
                    kn,
                    wref(resident_l(&store, "wq", l, dv), mq, q_dim_r, d),
                    wref(resident_l(&store, "wk", l, dv), mq, kv_dim_r, d),
                    wref(resident_l(&store, "wv", l, dv), mq, kv_dim_r, d),
                    wref(resident_l(&store, "wo", l, dv), mq, d, q_dim_r),
                    wref(resident_l(&store, "ffn_gate", l, dv), mq, inter_r, d),
                    wref(resident_l(&store, "ffn_up", l, dv), mq, inter_r, d),
                    wref(resident_l(&store, "ffn_down", l, dv), mq, d, inter_r),
                    KvTierPlan::derive(q8_tier(&s.kv, POS)).unwrap(),
                    &s.kv.k_gpu[l],
                    &s.kv.v_gpu[l],
                    &s.partials,
                    hpr,
                    kvpr,
                    hd,
                    d,
                    eps,
                    theta,
                    qkv_rot,
                    ffn_rot,
                )
            })
            .collect();
        execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives)
            .expect("execute_steps_tp");
    }

    // ── Final norm + lm_head on rank 0 → logits ──
    let logits_tp = {
        let g = &mut gpus.devices[group[0]];
        g.bind_thread().unwrap();
        let tmp = g.alloc_tensor(&[d], DType::F32).unwrap();
        let logits = g.alloc_tensor(&[config.vocab_size], DType::F32).unwrap();
        g.rmsnorm_f32(&st[0].x, &weights.output_norm, &tmp, eps)
            .unwrap();
        llama::weight_gemv(g, &weights.output, &tmp, &logits).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        g.download_f32(&logits).unwrap()
    };
    let argmax_tp = llama::argmax(&logits_tp);

    let diff = logits_tp
        .iter()
        .zip(&logits_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let mag = logits_ref.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!(
        "[tp-full-model] Tp-2 forward vs single-GPU forward_scratch: \
         argmax_tp={argmax_tp} argmax_ref={argmax_ref} logit max|Δ|={diff:.3e} (ref max|logit|={mag:.3e})"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    assert_eq!(
        argmax_tp, argmax_ref,
        "TP full-model argmax {argmax_tp} != single-GPU argmax {argmax_ref}"
    );
    println!(
        "tp_full_model_parity: full-model Tp-2 forward argmax == single-GPU forward_scratch \
         — PB-TP4c increment B validated (dense_forward_tp end-to-end)"
    );
}

/// The dense-layer 16-op Step list (mirrors `dense_forward`; row ops wo/down split
/// into Gemv → AllReduceOut(idx 8,14) → ResidualAdd per PB-TP4a).
#[allow(clippy::too_many_arguments)]
fn build_layer_steps<'a>(
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
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x,
            norm_weight: attn_norm,
            x_plain: tmp,
            out: x_rot,
            awq_scale: None,
            k: d,
            eps,
            rotation: qkv_rot,
        },
        Step::Gemv {
            w: leak(wq),
            input: GemvInput::Prerotated(x_rot),
            out: q,
        },
        Step::Gemv {
            w: leak(wk),
            input: GemvInput::Prerotated(x_rot),
            out: k,
        },
        Step::Gemv {
            w: leak(wv),
            input: GemvInput::Prerotated(x_rot),
            out: v,
        },
        Step::QkNorm {
            x: q,
            weight: q_norm,
            n_groups: nh,
            head_dim: hd,
            eps,
        },
        Step::QkNorm {
            x: k,
            weight: k_norm,
            n_groups: nkv,
            head_dim: hd,
            eps,
        },
        Step::Rope {
            q,
            k,
            pos_buf,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            theta,
        },
        Step::Attend {
            plan,
            io: AttnParams {
                q,
                k,
                v,
                k_cache,
                v_cache,
                k_scales: None,
                v_scales: None,
                pos_buf,
                pos: POS,
                positions: None,
                n_heads: nh,
                n_kv_heads: nkv,
                head_dim: hd,
                physical_cap: MAX_SEQ,
                batch_size: 1,
                max_ctx_len: 0,
                flash_partials: Some(partials),
                givens_cos: None,
                givens_sin: None,
                tree_bias: None,
                block_start: 0,
                block_cols: 0,
                output_gate: None,
                output: attn,
            },
        },
        Step::Gemv {
            w: leak(wo),
            input: GemvInput::Raw(attn),
            out: o,
        },
        Step::ResidualAdd { x, y: o, dim: d },
        Step::RmsnormAutomatic {
            x,
            norm_weight: ffn_norm,
            x_plain: tmp,
            out: x_rot,
            awq_scale: None,
            k: d,
            eps,
            rotation: ffn_rot,
        },
        Step::Gemv {
            w: leak(w_gate),
            input: GemvInput::Prerotated(x_rot),
            out: gate,
        },
        Step::Gemv {
            w: leak(w_up),
            input: GemvInput::Prerotated(x_rot),
            out: up,
        },
        Step::SiluMul {
            gate,
            up,
            out: hidden,
        },
        Step::Gemv {
            w: leak(w_down),
            input: GemvInput::Raw(hidden),
            out: fo,
        },
        Step::ResidualAdd { x, y: fo, dim: d },
    ]
}
