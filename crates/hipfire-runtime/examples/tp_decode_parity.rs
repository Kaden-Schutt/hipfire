// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP5 validation: **dense tensor-parallel DECODE parity** — prefill a prompt
//! and greedy-decode N tokens fully tensor-parallel (Tp-2), and assert the token
//! stream is IDENTICAL to the production single-GPU `llama::forward_scratch`.
//! This is the serve-loop analog of `ep_decode_parity`: where
//! `tp_full_model_parity` proved a single-position forward, this proves the real
//! generation loop — prefill + multi-token decode with a GROWING per-rank KV
//! cache, so multi-key attention runs head-parallel under TP for real.
//!
//! Each token: embed (rank 0, broadcast the replicated hidden) → all 28 sharded
//! layers via `execute_steps_tp` (KV write at the token's pos) → final norm +
//! lm_head (rank 0) → argmax → feed back. The reference runs `forward_scratch`
//! (fused production path) over the same prompt; both must pick the same token
//! at every step (argmax-exact; logits match to a fusion tolerance).
//!
//! Two gates (mirrors ep_decode_parity):
//!   1. per-step argmax parity (hard assert) over prefill-last + N decode steps,
//!   2. an FNV-1a hash of the full generated id stream (printed + asserted equal).
//!
//! Emulated Tp-2 (gfx1151). Run: HIP_VISIBLE_DEVICES=0 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example tp_decode_parity [model.mq4] [steps] [prompt]

use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::llama::{
    self, ForwardScratch, KvCache, KvCacheExt, LlamaConfig, LlamaWeights,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};

const TP: usize = 2;
const MAX_SEQ: usize = 512;

fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let prompt = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("The capital of France is");

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_decode_parity: SKIPPED (no {TP}-rank Gpus: {e})");
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
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
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

    let prompt_tokens = tokenizer.encode(prompt);
    assert!(!prompt_tokens.is_empty(), "empty prompt");
    eprintln!(
        "model: d={d} layers={n_layers} | prompt={} toks, decode {steps} steps",
        prompt_tokens.len()
    );

    // ── Reference: single-GPU forward_scratch (prefill + greedy decode) ──
    let ref_ids: Vec<u32> = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let mut kv = KvCache::new_gpu_q8(g, n_layers, nkv, hd, MAX_SEQ).unwrap();
        let scratch = ForwardScratch::new(g, &config).unwrap();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            llama::forward_scratch(
                g, &weights, &config, tok, pos, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
            )
            .expect("ref prefill");
        }
        let mut ids = Vec::with_capacity(steps + 1);
        let mut next = llama::argmax(&g.download_f32(&scratch.logits).unwrap());
        ids.push(next);
        for step in 0..steps {
            let pos = prompt_tokens.len() + step;
            llama::forward_scratch(
                g, &weights, &config, next, pos, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
            )
            .expect("ref decode");
            next = llama::argmax(&g.download_f32(&scratch.logits).unwrap());
            ids.push(next);
        }
        ids
    };

    // ── Store→forward bridge: shard every layer's quant weights (Tp-2) ──
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

    let up_f32 = |g: &Gpu, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
        g.upload_raw(&b, &[v.len()]).unwrap()
    };
    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let hpr = nh / TP;
    let kvpr = nkv / TP;
    let (q_dim_r, kv_dim_r, inter_r) = (hpr * hd, kvpr * hd, ff / TP);
    let x_rot_cap = d.max(ff);
    let chunks = MAX_SEQ.div_ceil(128);

    // Per-layer replicated norm CPU copies (download once from rank 0).
    let norms_cpu: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_layers)
        .map(|l| {
            let g = &gpus.devices[0];
            (
                g.download_f32(&weights.layers[l].attn_norm).unwrap(),
                g.download_f32(&weights.layers[l].ffn_norm).unwrap(),
                g.download_f32(weights.layers[l].q_norm.as_ref().unwrap())
                    .unwrap(),
                g.download_f32(weights.layers[l].k_norm.as_ref().unwrap())
                    .unwrap(),
            )
        })
        .collect();

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
        norms: Vec<(GpuTensor, GpuTensor, GpuTensor, GpuTensor)>,
    }
    let mut st: Vec<RankState> = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        let kv = KvCache::new_gpu_q8(g, n_layers, kvpr, hd, MAX_SEQ).unwrap();
        let norms = norms_cpu
            .iter()
            .map(|(a, f, q, k)| (up_f32(g, a), up_f32(g, f), up_f32(g, q), up_f32(g, k)))
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
            pos_buf: g.hip.malloc(4).unwrap(),
            kv,
            norms,
        });
    }

    let mut collectives: Vec<TpCollective> = (0..16).map(|_| TpCollective::None).collect();
    collectives[8] = TpCollective::AllReduceOut { dim: d };
    collectives[14] = TpCollective::AllReduceOut { dim: d };

    // One tensor-parallel token forward at `pos`: embed(rank0)+broadcast → 28
    // sharded layers (KV write at pos). Returns nothing; mutates x + KV.
    let tp_token = |gpus: &mut Gpus, st: &mut [RankState], token: u32, pos: usize| {
        // Embed on rank 0 + broadcast the replicated hidden.
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
        let x0 = gpus.devices[group[0]].download_f32(&st[0].x).unwrap();
        let x0b: Vec<u8> = x0.iter().flat_map(|f| f.to_ne_bytes()).collect();
        for (r, &dev) in group.iter().enumerate() {
            let g = &mut gpus.devices[dev];
            g.bind_thread().unwrap();
            g.hip
                .memcpy_htod(&st[r].pos_buf, &(pos as i32).to_ne_bytes())
                .unwrap();
            if r != 0 {
                g.hip.memcpy_htod(&st[r].x.buf, &x0b).unwrap();
            }
        }
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
                        KvTierPlan::derive(KvTierInputs {
                            pos,
                            ..s.kv.tier_inputs()
                        })
                        .unwrap(),
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
                        pos,
                    )
                })
                .collect();
            execute_steps_tp(&mesh, gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");
        }
    };

    // Final norm + lm_head on rank 0 → greedy token.
    let tp_logits_argmax = |gpus: &mut Gpus, st: &[RankState]| -> u32 {
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
        let am = llama::argmax(&g.download_f32(&logits).unwrap());
        let _ = g.free_tensor(tmp);
        let _ = g.free_tensor(logits);
        am
    };

    // ── TP prefill + greedy decode ──
    let mut tp_ids = Vec::with_capacity(steps + 1);
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        tp_token(&mut gpus, &mut st, tok, pos);
    }
    let mut next = tp_logits_argmax(&mut gpus, &st);
    tp_ids.push(next);
    for step in 0..steps {
        let pos = prompt_tokens.len() + step;
        tp_token(&mut gpus, &mut st, next, pos);
        next = tp_logits_argmax(&mut gpus, &st);
        tp_ids.push(next);
    }

    // ── Compare ──
    let mut first_div = None;
    for (i, (a, b)) in tp_ids.iter().zip(&ref_ids).enumerate() {
        if a != b {
            first_div = Some(i);
            break;
        }
    }
    let ref_fnv = fnv1a(&ref_ids);
    let tp_fnv = fnv1a(&tp_ids);
    eprintln!("ref text: {:?}", tokenizer.decode(&ref_ids));
    eprintln!(" tp text: {:?}", tokenizer.decode(&tp_ids));
    println!(
        "[tp-decode] steps={} ref_fnv={ref_fnv:016x} tp_fnv={tp_fnv:016x} first_div={:?}",
        tp_ids.len(),
        first_div
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    assert_eq!(
        first_div, None,
        "TP decode diverged from single-GPU at step {first_div:?}: tp={tp_ids:?} ref={ref_ids:?}"
    );
    assert_eq!(tp_fnv, ref_fnv, "token-stream FNV mismatch");
    println!(
        "tp_decode_parity: dense TP prefill+decode token stream == single-GPU forward_scratch \
         ({} tokens, argmax-exact) — PB-TP5 serve loop validated",
        tp_ids.len()
    );
}

/// The dense-layer 16-op Step list at token position `pos` (mirrors dense_forward;
/// row ops wo/down split into Gemv → AllReduceOut(idx 8,14) → ResidualAdd).
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
    pos: usize,
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
                pos,
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
