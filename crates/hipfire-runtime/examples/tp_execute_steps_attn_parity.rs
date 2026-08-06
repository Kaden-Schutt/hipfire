// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP4b proof: a **head-parallel attention block driven through
//! `execute_steps_tp` via a first-class `Step::Attend`** == single-device. PB-3
//! (`tp_attn_parity`) validated the head-parallel attention MATH on *raw* ops
//! (`rope_f32` + `kv_cache_write` + `attention_f32`). This closes the executor
//! gap: the whole attention block is now one per-rank `Step` list the TP executor
//! shards —
//! ```text
//!   xn      = rmsnorm(x, attn_norm)         [Replicate → None]
//!   q_r     = Wq_r · xn                      [ColumnShard axis0 → None]  (rank owns q heads)
//!   k_r,v_r = Wk_r·xn , Wv_r·xn              [ColumnShard axis0 → None]  (rank owns kv heads)
//!   rope(q_r, k_r, POS)                      [Step::Rope — per-head, exact on the local slice]
//!   attn_r  = Attend{plan_r, io_r}           [Step::Attend — KV-write + attend on owned heads]
//!   o       = all_reduce_r(Wo_r · attn_r)    [RowShard axis1 → AllReduceOut{D}]
//!   x       = x + o                          [Step::ResidualAdd — after the collective, replicated]
//! ```
//! `Step::Attend` carries a REAL `KvTierPlan` + `AttnParams` (the shape llama's
//! `attend_plan` builds), NOT a hand-wired kernel call: the F32/`Simple` tier
//! (all quant flags off) selects the same `attention_f32` PB-3 used raw, so this
//! is a pure "route the proven math through the executor" step. Each rank gets a
//! KV cache sized to its OWN kv heads (clean GQA split preserves the
//! n_heads/n_kv_heads ratio per rank), seeded with its column-slice of the F32
//! history; `Step::Attend` writes the current token into each rank's cache.
//!
//! The single-device reference runs the identical per-op kernels (same
//! `run_attention`) on the whole heads + full cache. Validated on emulated Tp-2
//! (`HIPFIRE_EMULATE_GPUS=2`, gfx1151). `q_dim/tp` kept 64-aligned.
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_attn_parity

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::{AttentionFamily, AttnParams};
use hipfire_dispatch::families::gemv::{GemvFamily, WeightRef};
use hipfire_dispatch::families::kv_tier::{F32AttnPolicy, KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::RotationPlan;
use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};

const D: usize = 128; // hidden dim
const NH: usize = 4; // query heads
const NKV: usize = 2; // kv heads (GQA group = NH/NKV = 2)
const HD: usize = 32; // head dim
const TP: usize = 2;
const HIST: usize = 5; // cached tokens of history
const POS: usize = HIST; // decode this token at slot HIST
const MAX_SEQ: usize = 64;
const EPS: f32 = 1e-5;
const FREQ_BASE: f32 = 10000.0;
const TOL: f32 = 2e-3;

const Q_DIM: usize = NH * HD; // 128
const KV_DIM: usize = NKV * HD; // 64

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}
fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
fn attn_norm() -> Vec<f32> {
    (0..D).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect()
}
fn wq() -> Vec<f32> {
    (0..Q_DIM * D)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect()
}
fn wk() -> Vec<f32> {
    (0..KV_DIM * D)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.021)
        .collect()
}
fn wv() -> Vec<f32> {
    (0..KV_DIM * D)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.023)
        .collect()
}
fn wo() -> Vec<f32> {
    (0..D * Q_DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.025)
        .collect()
}
// F32 KV history (slots [0,HIST) filled, slot POS left for Step::Attend to write).
fn kcache_full() -> Vec<f32> {
    let mut c = vec![0f32; MAX_SEQ * KV_DIM];
    for t in 0..HIST {
        for j in 0..KV_DIM {
            c[t * KV_DIM + j] = (((t * 7 + j * 3) % 17) as f32 - 8.0) * 0.03;
        }
    }
    c
}
fn vcache_full() -> Vec<f32> {
    let mut c = vec![0f32; MAX_SEQ * KV_DIM];
    for t in 0..HIST {
        for j in 0..KV_DIM {
            c[t * KV_DIM + j] = (((t * 5 + j * 2) % 13) as f32 - 6.0) * 0.035;
        }
    }
    c
}

fn resident<'a>(store: &'a WeightStore, name: &str, dev: usize) -> &'a GpuTensor {
    match store.get(name, Some(0), dev) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("{name} not resident on device {dev}"),
    }
}
fn wref<'a>(t: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf: t,
        dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    }
}
/// The F32/`Simple` KV tier — all quant flags off → `KernelKey::AttnF32`, i.e.
/// the plain `attention_f32` PB-3 exercised raw. This is the shape llama's
/// `attend_plan` builds (`KvTierPlan::derive(kv.tier_inputs())`); here the cache
/// is a plain F32 buffer so the inputs are hand-built rather than read off a
/// quantised `KvCache`.
fn f32_tier(pos: usize) -> KvTierInputs {
    KvTierInputs {
        quant_asym4: false,
        quant_asym3: false,
        quant_asym2: false,
        quant_q8: false,
        quant_fwht: false,
        quant_hfq4: false,
        quant_q4: false,
        quant_int8: false,
        quant_hfq8: false,
        f32_policy: F32AttnPolicy::Simple,
        v_mode_bits: 0,
        pos,
        flash_mode: 0,
        capture_mode: false,
        batch_size: 1,
        is_tree: false,
        is_boundary: false,
        q8_windowed: false,
        window: 0,
    }
}

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_execute_steps_attn_parity: SKIPPED (no {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    let x0: Vec<f32> = (0..D).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let x_bytes = f32_to_bytes(&x0);

    // ── Single-device reference: whole heads, full cache, same run_attention ──
    let gemv = GemvFamily::new();
    let attn_fam = AttentionFamily::new();
    let x_ref = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let ctx = DispatchCtx::new(g);
        let up = |g: &Gpu, v: &[f32], s: &[usize]| g.upload_raw(&f32_to_bytes(v), s).unwrap();
        let x = g.upload_raw(&x_bytes, &[D]).unwrap();
        let xn = g.alloc_tensor(&[D], DType::F32).unwrap();
        let q = g.alloc_tensor(&[Q_DIM], DType::F32).unwrap();
        let k = g.alloc_tensor(&[KV_DIM], DType::F32).unwrap();
        let v = g.alloc_tensor(&[KV_DIM], DType::F32).unwrap();
        let attn = g.alloc_tensor(&[Q_DIM], DType::F32).unwrap();
        let o = g.alloc_tensor(&[D], DType::F32).unwrap();
        let w_an = up(g, &attn_norm(), &[D]);
        let w_q = up(g, &wq(), &[Q_DIM, D]);
        let w_k = up(g, &wk(), &[KV_DIM, D]);
        let w_v = up(g, &wv(), &[KV_DIM, D]);
        let w_o = up(g, &wo(), &[D, Q_DIM]);
        let kcache = up(g, &kcache_full(), &[MAX_SEQ * KV_DIM]);
        let vcache = up(g, &vcache_full(), &[MAX_SEQ * KV_DIM]);
        let chunks = MAX_SEQ.div_ceil(128);
        let partials = g
            .alloc_tensor(&[NH * chunks * (2 + HD)], DType::F32)
            .unwrap();
        let pos_buf = g.hip.malloc(4).unwrap();
        g.hip
            .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
            .unwrap();

        g.rmsnorm_f32(&x, &w_an, &xn, EPS).unwrap();
        gemv.run_auto(&ctx, g, &wref(&w_q, Q_DIM, D), &xn, &q)
            .unwrap();
        gemv.run_auto(&ctx, g, &wref(&w_k, KV_DIM, D), &xn, &k)
            .unwrap();
        gemv.run_auto(&ctx, g, &wref(&w_v, KV_DIM, D), &xn, &v)
            .unwrap();
        g.rope_f32(&q, &k, &pos_buf, NH, NKV, HD, FREQ_BASE)
            .unwrap();
        let plan = KvTierPlan::derive(f32_tier(POS)).unwrap();
        let io = AttnParams {
            q: &q,
            k: &k,
            v: &v,
            k_cache: &kcache,
            v_cache: &vcache,
            k_scales: None,
            v_scales: None,
            pos_buf: &pos_buf,
            pos: POS,
            positions: None,
            n_heads: NH,
            n_kv_heads: NKV,
            head_dim: HD,
            physical_cap: MAX_SEQ,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &attn,
        };
        attn_fam.run_attention(&ctx, g, &plan, &io).unwrap();
        gemv.run_auto(&ctx, g, &wref(&w_o, D, Q_DIM), &attn, &o)
            .unwrap();
        g.add_f32(&x, &o, &x).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let mut out = vec![0u8; D * 4];
        g.hip.memcpy_dtoh(&mut out, &x.buf).unwrap();
        bytes_to_f32(&out)
    };

    // ── TP path: shard the layer weights, run the attention block via the executor ──
    let p_rep = ShardPolicy::Replicate;
    let p_col = ShardPolicy::ColumnShard { axis: 0 };
    let p_row = ShardPolicy::RowShard { axis: 1 };
    let manifest = vec![
        WeightEntry::layer("attn_norm", 0, vec![D], DType::F32, p_rep.clone()),
        WeightEntry::layer("wq", 0, vec![Q_DIM, D], DType::F32, p_col.clone()),
        WeightEntry::layer("wk", 0, vec![KV_DIM, D], DType::F32, p_col.clone()),
        WeightEntry::layer("wv", 0, vec![KV_DIM, D], DType::F32, p_col.clone()),
        WeightEntry::layer("wo", 0, vec![D, Q_DIM], DType::F32, p_row.clone()),
    ];
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "attn_norm" => f32_to_bytes(&attn_norm()),
            "wq" => f32_to_bytes(&wq()),
            "wk" => f32_to_bytes(&wk()),
            "wv" => f32_to_bytes(&wv()),
            _ => f32_to_bytes(&wo()),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let hpr = NH / TP; // q heads per rank
    let kvpr = NKV / TP; // kv heads per rank
    let q_dim_r = hpr * HD;
    let kv_dim_r = kvpr * HD;

    // Per-rank scratch/caches (owned by the example, NOT inside `gpus` → no alias
    // with the &mut gpus the executor takes).
    struct RankBufs {
        x: GpuTensor,
        xn: GpuTensor,
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        attn: GpuTensor,
        o: GpuTensor,
        kcache: GpuTensor,
        vcache: GpuTensor,
        partials: GpuTensor,
        pos_buf: hip_bridge::DeviceBuffer,
    }
    let kfull = kcache_full();
    let vfull = vcache_full();
    let chunks = MAX_SEQ.div_ceil(128);
    let mut bufs: Vec<RankBufs> = Vec::with_capacity(TP);
    for (rank, &dev) in group.iter().enumerate() {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        // rank r owns kv heads [r*kvpr,(r+1)*kvpr) → cache columns [r*kv_dim_r,..).
        let mut kr = vec![0f32; MAX_SEQ * kv_dim_r];
        let mut vr = vec![0f32; MAX_SEQ * kv_dim_r];
        for t in 0..MAX_SEQ {
            for j in 0..kv_dim_r {
                kr[t * kv_dim_r + j] = kfull[t * KV_DIM + rank * kv_dim_r + j];
                vr[t * kv_dim_r + j] = vfull[t * KV_DIM + rank * kv_dim_r + j];
            }
        }
        let pos_buf = g.hip.malloc(4).unwrap();
        g.hip
            .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
            .unwrap();
        bufs.push(RankBufs {
            x: g.upload_raw(&x_bytes, &[D]).unwrap(),
            xn: g.alloc_tensor(&[D], DType::F32).unwrap(),
            q: g.alloc_tensor(&[q_dim_r], DType::F32).unwrap(),
            k: g.alloc_tensor(&[kv_dim_r], DType::F32).unwrap(),
            v: g.alloc_tensor(&[kv_dim_r], DType::F32).unwrap(),
            attn: g.alloc_tensor(&[q_dim_r], DType::F32).unwrap(),
            o: g.alloc_tensor(&[D], DType::F32).unwrap(),
            kcache: g
                .upload_raw(&f32_to_bytes(&kr), &[MAX_SEQ * kv_dim_r])
                .unwrap(),
            vcache: g
                .upload_raw(&f32_to_bytes(&vr), &[MAX_SEQ * kv_dim_r])
                .unwrap(),
            partials: g
                .alloc_tensor(&[hpr * chunks * (2 + HD)], DType::F32)
                .unwrap(),
            pos_buf,
        });
    }

    // Per-rank WeightRefs + the Attend plan/io that the Step list borrows.
    let an_refs: Vec<&GpuTensor> = group
        .iter()
        .map(|&d| resident(&store, "attn_norm", d))
        .collect();
    let wq_refs: Vec<WeightRef> = group
        .iter()
        .map(|&d| wref(resident(&store, "wq", d), q_dim_r, D))
        .collect();
    let wk_refs: Vec<WeightRef> = group
        .iter()
        .map(|&d| wref(resident(&store, "wk", d), kv_dim_r, D))
        .collect();
    let wv_refs: Vec<WeightRef> = group
        .iter()
        .map(|&d| wref(resident(&store, "wv", d), kv_dim_r, D))
        .collect();
    let wo_refs: Vec<WeightRef> = group
        .iter()
        .map(|&d| wref(resident(&store, "wo", d), D, q_dim_r))
        .collect();
    // AttnParams is not Clone (borrow struct), so it is built inline per rank in
    // the step list below; the KvTierPlan (Clone) is cheap to re-derive inline.
    let per_rank_steps: Vec<Vec<Step>> = (0..TP)
        .map(|r| {
            vec![
                Step::RmsnormAutomatic {
                    x: &bufs[r].x,
                    norm_weight: an_refs[r],
                    x_plain: &bufs[r].xn,
                    out: &bufs[r].xn,
                    awq_scale: None,
                    k: D,
                    eps: EPS,
                    rotation: RotationPlan::None,
                },
                Step::Gemv {
                    w: &wq_refs[r],
                    input: GemvInput::Raw(&bufs[r].xn),
                    out: &bufs[r].q,
                },
                Step::Gemv {
                    w: &wk_refs[r],
                    input: GemvInput::Raw(&bufs[r].xn),
                    out: &bufs[r].k,
                },
                Step::Gemv {
                    w: &wv_refs[r],
                    input: GemvInput::Raw(&bufs[r].xn),
                    out: &bufs[r].v,
                },
                Step::Rope {
                    q: &bufs[r].q,
                    k: &bufs[r].k,
                    pos_buf: &bufs[r].pos_buf,
                    n_heads: hpr,
                    n_kv_heads: kvpr,
                    head_dim: HD,
                    theta: FREQ_BASE,
                },
                Step::Attend {
                    plan: KvTierPlan::derive(f32_tier(POS)).unwrap(),
                    io: AttnParams {
                        q: &bufs[r].q,
                        k: &bufs[r].k,
                        v: &bufs[r].v,
                        k_cache: &bufs[r].kcache,
                        v_cache: &bufs[r].vcache,
                        k_scales: None,
                        v_scales: None,
                        pos_buf: &bufs[r].pos_buf,
                        pos: POS,
                        positions: None,
                        n_heads: hpr,
                        n_kv_heads: kvpr,
                        head_dim: HD,
                        physical_cap: MAX_SEQ,
                        batch_size: 1,
                        max_ctx_len: 0,
                        flash_partials: Some(&bufs[r].partials),
                        givens_cos: None,
                        givens_sin: None,
                        tree_bias: None,
                        block_start: 0,
                        block_cols: 0,
                        output_gate: None,
                        output: &bufs[r].attn,
                    },
                },
                Step::Gemv {
                    w: &wo_refs[r],
                    input: GemvInput::Raw(&bufs[r].attn),
                    out: &bufs[r].o,
                },
                Step::ResidualAdd {
                    x: &bufs[r].x,
                    y: &bufs[r].o,
                    dim: D,
                },
            ]
        })
        .collect();
    let collectives = vec![
        TpCollective::None,                    // rmsnorm
        TpCollective::None,                    // wq (col)
        TpCollective::None,                    // wk (col)
        TpCollective::None,                    // wv (col)
        TpCollective::None,                    // rope
        TpCollective::None,                    // attend (on-rank heads)
        TpCollective::AllReduceOut { dim: D }, // wo (row) → sum partials
        TpCollective::None,                    // residual add (after collective, replicated)
    ];

    execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");

    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let mut out = vec![0u8; D * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &bufs[0].x.buf)
        .unwrap();
    let x_tp = bytes_to_f32(&out);

    let diff = max_abs_diff(&x_tp, &x_ref);
    println!(
        "[tp-attn-executor] head-parallel attention block (Step::Attend) through execute_steps_tp \
         vs single-device: max|Δ|={diff:.3e}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    assert!(
        diff < TOL,
        "TP attention block diverges from single-device: max|Δ|={diff}"
    );
    println!(
        "tp_execute_steps_attn_parity: head-parallel Step::Attend through execute_steps_tp == \
         single-device — PB-TP4b validated"
    );
}
