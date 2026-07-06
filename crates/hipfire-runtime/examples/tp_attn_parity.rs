// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-3 functional proof: an **attention-head-parallel** transformer layer that
//! matches the single-device forward, on real hardware. Where `tp_forward_parity`
//! proves the FFN half (column→row GEMV + all-reduce), this adds the genuinely
//! new mechanism — the **attention block** — and validates a WHOLE TP transformer
//! layer (attention + FFN) against a single-device reference computed with the
//! *same* GPU kernels (so RoPE / softmax / GQA conventions are identical by
//! construction, not reimplemented on the host).
//!
//! Per-layer TP dataflow (Megatron), decode of one token at position `POS` with
//! `HIST` cached tokens of history:
//! ```text
//!   xn   = rmsnorm(x, attn_norm)              [replicated]
//!   q_r  = Wq_r · xn                          [Wq column-shard → rank owns q_head_range]
//!   k_r  = Wk_r · xn ; v_r = Wv_r · xn        [Wk/Wv column-shard → rank owns kv_head_range]
//!   rope(q_r, k_r, POS)                       [per-head → the local head slice is exact]
//!   kv_write(cache_r, k_r/v_r, POS)           [each rank owns a KV cache sized to local kv_dim]
//!   a_r  = attention(q_r, cache_r, POS)       [GQA on owned heads; ratio preserved per rank]
//!   o    = all_reduce_r( Wo_r · a_r )         [Wo row-shard → partial per rank, summed]
//!   x    = x + o                              [attention residual, x stays replicated]
//!   ... then the proven FFN block (rmsnorm → col W1 → silu → row W2 → all_reduce → residual)
//! ```
//!
//! Head-parallel is *exact*, not approximate: RoPE is per-head, and clean GQA
//! splitting keeps each rank's Q heads mapped entirely onto its own KV heads
//! (`n_heads/n_kv_heads` ratio preserved), so a rank's `attention_f32` over its
//! local `[max_seq, kv_dim/tp]` cache equals the corresponding head-slice of the
//! full-cache attention. Only `Wo`'s output crosses ranks (one all-reduce), same
//! as the FFN. Validated on an emulated Tp-2 mesh (`HIPFIRE_EMULATE_GPUS=2`);
//! `q_dim/tp` and `inter/tp` are kept 64-aligned (the `gemv_f32` reduction-dim
//! constraint from `tp_gemv_parity`).
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_attn_parity

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, GpuTensor};

const D: usize = 128; // hidden dim
const NH: usize = 4; // query heads
const NKV: usize = 2; // kv heads (GQA group = NH/NKV = 2)
const HD: usize = 32; // head dim
const INTER: usize = 128; // FFN intermediate
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

// ── Deterministic host ground-truth tensors ──────────────────────────────
fn attn_norm() -> Vec<f32> {
    (0..D).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect()
}
fn ffn_norm() -> Vec<f32> {
    (0..D).map(|i| 1.0 + (i % 7) as f32 * 0.013).collect()
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
fn w1() -> Vec<f32> {
    (0..INTER * D)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect()
}
fn w2() -> Vec<f32> {
    (0..D * INTER)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect()
}
// Synthetic already-processed KV history: cache[t][kv_head][d], row-major
// [MAX_SEQ, KV_DIM]; slots [0,HIST) filled, the rest zero (unread: seq_len=POS+1).
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

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_attn_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Decode-token hidden (replicated on every rank).
    let x0: Vec<f32> = (0..D).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

    // ── Single-device reference: the same op chain, whole (unsharded) weights ──
    let y_ref = single_device_forward(&mut gpus, &x0);

    // ── TP path: shard the layer weights, run the head-parallel + FFN executor ──
    let mut manifest = Vec::new();
    manifest.push(WeightEntry::layer(
        "attn_norm",
        0,
        vec![D],
        DType::F32,
        ShardPolicy::Replicate,
    ));
    manifest.push(WeightEntry::layer(
        "wq",
        0,
        vec![Q_DIM, D],
        DType::F32,
        ShardPolicy::ColumnShard { axis: 0 },
    ));
    manifest.push(WeightEntry::layer(
        "wk",
        0,
        vec![KV_DIM, D],
        DType::F32,
        ShardPolicy::ColumnShard { axis: 0 },
    ));
    manifest.push(WeightEntry::layer(
        "wv",
        0,
        vec![KV_DIM, D],
        DType::F32,
        ShardPolicy::ColumnShard { axis: 0 },
    ));
    manifest.push(WeightEntry::layer(
        "wo",
        0,
        vec![D, Q_DIM],
        DType::F32,
        ShardPolicy::RowShard { axis: 1 },
    ));
    manifest.push(WeightEntry::layer(
        "ffn_norm",
        0,
        vec![D],
        DType::F32,
        ShardPolicy::Replicate,
    ));
    manifest.push(WeightEntry::layer(
        "w1",
        0,
        vec![INTER, D],
        DType::F32,
        ShardPolicy::ColumnShard { axis: 0 },
    ));
    manifest.push(WeightEntry::layer(
        "w2",
        0,
        vec![D, INTER],
        DType::F32,
        ShardPolicy::RowShard { axis: 1 },
    ));
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "attn_norm" => f32_to_bytes(&attn_norm()),
            "wq" => f32_to_bytes(&wq()),
            "wk" => f32_to_bytes(&wk()),
            "wv" => f32_to_bytes(&wv()),
            "wo" => f32_to_bytes(&wo()),
            "ffn_norm" => f32_to_bytes(&ffn_norm()),
            "w1" => f32_to_bytes(&w1()),
            _ => f32_to_bytes(&w2()),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    let y_tp = tp_forward(&mut gpus, &mesh, &store, &x0);

    let d_ref = max_abs_diff(&y_tp, &y_ref);
    println!("[tp-attn] full TP transformer layer (attn+FFN) vs single-device: max|Δ|={d_ref:.2e}");
    assert!(
        d_ref < TOL,
        "TP attention layer diverges from single-device: max|Δ|={d_ref}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().expect("bind");
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!(
        "tp_attn_parity: attention-head-parallel + FFN TP layer == single-device — PB-3 validated"
    );
}

/// Reference: run the identical decode-layer op chain on device 0 with the whole
/// (unsharded) weights and a full KV cache. Returns the post-layer hidden `[D]`.
fn single_device_forward(gpus: &mut Gpus, x0: &[f32]) -> Vec<f32> {
    let g = &mut gpus.devices[0];
    g.bind_thread().expect("bind");
    let up = |g: &rdna_compute::Gpu, v: &[f32], shape: &[usize]| {
        g.upload_raw(&f32_to_bytes(v), shape).expect("upload")
    };

    let x = up(g, x0, &[D]);
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

    let pos_buf = g.hip.malloc(4).unwrap();
    g.hip
        .memcpy_htod(&pos_buf, &(POS as i32).to_ne_bytes())
        .unwrap();

    // Attention block.
    g.rmsnorm_f32(&x, &w_an, &xn, EPS).unwrap();
    g.gemv_f32(&w_q, &xn, &q).unwrap();
    g.gemv_f32(&w_k, &xn, &k).unwrap();
    g.gemv_f32(&w_v, &xn, &v).unwrap();
    g.rope_f32(&q, &k, &pos_buf, NH, NKV, HD, FREQ_BASE)
        .unwrap();
    g.kv_cache_write(&kcache, &k, &pos_buf, KV_DIM).unwrap();
    g.kv_cache_write(&vcache, &v, &pos_buf, KV_DIM).unwrap();
    g.attention_f32(
        &q,
        &kcache,
        &vcache,
        &attn,
        &pos_buf,
        POS + 1,
        NH,
        NKV,
        HD,
        MAX_SEQ,
    )
    .unwrap();
    g.gemv_f32(&w_o, &attn, &o).unwrap();
    g.add_f32(&x, &o, &x).unwrap();

    // FFN block.
    let w_fn = up(g, &ffn_norm(), &[D]);
    let w_1 = up(g, &w1(), &[INTER, D]);
    let w_2 = up(g, &w2(), &[D, INTER]);
    let xn2 = g.alloc_tensor(&[D], DType::F32).unwrap();
    let gt = g.alloc_tensor(&[INTER], DType::F32).unwrap();
    let ft = g.alloc_tensor(&[D], DType::F32).unwrap();
    g.rmsnorm_f32(&x, &w_fn, &xn2, EPS).unwrap();
    g.gemv_f32(&w_1, &xn2, &gt).unwrap();
    g.silu_f32(&gt, &gt).unwrap();
    g.gemv_f32(&w_2, &gt, &ft).unwrap();
    g.add_f32(&x, &ft, &x).unwrap();

    g.hip
        .stream_synchronize(g.active_stream.as_ref().unwrap())
        .unwrap();
    let mut out = vec![0u8; D * 4];
    g.hip.memcpy_dtoh(&mut out, &x.buf).unwrap();
    bytes_to_f32(&out)
}

/// TP path: per-rank sharded weights + one all-reduce after each row-parallel op.
fn tp_forward(gpus: &mut Gpus, mesh: &DeviceMesh, store: &WeightStore, x0: &[f32]) -> Vec<f32> {
    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let tp = group.len();
    let hpr = NH / tp; // q heads per rank
    let kvpr = NKV / tp; // kv heads per rank (clean GQA split)
    let q_dim_r = hpr * HD; // local attention out width
    let kv_dim_r = kvpr * HD; // local kv cache width
    let inter_r = INTER / tp;

    struct RankScratch {
        x: GpuTensor,
        xn: GpuTensor,
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        attn: GpuTensor,
        o: GpuTensor, // row-parallel partial (all-reduced in place)
        kcache: GpuTensor,
        vcache: GpuTensor,
        pos_buf: hip_bridge::DeviceBuffer,
        // FFN scratch
        xn2: GpuTensor,
        g: GpuTensor,
        f: GpuTensor, // row-parallel partial
    }

    let x_bytes = f32_to_bytes(x0);
    let kfull = kcache_full();
    let vfull = vcache_full();
    let mut scratch: Vec<RankScratch> = Vec::with_capacity(tp);
    for (rank, &dev) in group.iter().enumerate() {
        let g = &mut gpus.devices[dev];
        g.bind_thread().expect("bind");
        // Seed this rank's KV cache with its own kv-head slice of the history:
        // rank r owns kv heads [r*kvpr, (r+1)*kvpr) → columns [r*kv_dim_r, ...).
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
        scratch.push(RankScratch {
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
            pos_buf,
            xn2: g.alloc_tensor(&[D], DType::F32).unwrap(),
            g: g.alloc_tensor(&[inter_r], DType::F32).unwrap(),
            f: g.alloc_tensor(&[D], DType::F32).unwrap(),
        });
    }

    // ── Attention block (head-parallel) ──
    for (rank, &dev) in group.iter().enumerate() {
        let w_an = resident(store, "attn_norm", dev);
        let w_q = resident(store, "wq", dev);
        let w_k = resident(store, "wk", dev);
        let w_v = resident(store, "wv", dev);
        let w_o = resident(store, "wo", dev);
        let s = &scratch[rank];
        let g = &mut gpus.devices[dev];
        g.bind_thread().expect("bind");
        g.rmsnorm_f32(&s.x, w_an, &s.xn, EPS).unwrap();
        g.gemv_f32(w_q, &s.xn, &s.q).unwrap();
        g.gemv_f32(w_k, &s.xn, &s.k).unwrap();
        g.gemv_f32(w_v, &s.xn, &s.v).unwrap();
        // RoPE on owned heads (per-head → exact on the local slice).
        g.rope_f32(&s.q, &s.k, &s.pos_buf, hpr, kvpr, HD, FREQ_BASE)
            .unwrap();
        g.kv_cache_write(&s.kcache, &s.k, &s.pos_buf, kv_dim_r)
            .unwrap();
        g.kv_cache_write(&s.vcache, &s.v, &s.pos_buf, kv_dim_r)
            .unwrap();
        g.attention_f32(
            &s.q,
            &s.kcache,
            &s.vcache,
            &s.attn,
            &s.pos_buf,
            POS + 1,
            hpr,
            kvpr,
            HD,
            MAX_SEQ,
        )
        .unwrap();
        // Row-parallel O-proj → partial residual contribution.
        g.gemv_f32(w_o, &s.attn, &s.o).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
    }
    // All-reduce the O-proj partials, then attention residual.
    let refs: Vec<&_> = scratch.iter().map(|s| &s.o.buf).collect();
    gpus.all_reduce_sum_f32_peer(&group, &refs, D).unwrap();
    for (rank, &dev) in group.iter().enumerate() {
        let s = &scratch[rank];
        let g = &mut gpus.devices[dev];
        g.bind_thread().expect("bind");
        g.add_f32(&s.x, &s.o, &s.x).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
    }

    // ── FFN block (the proven column→row pattern) ──
    for (rank, &dev) in group.iter().enumerate() {
        let w_fn = resident(store, "ffn_norm", dev);
        let w_1 = resident(store, "w1", dev);
        let w_2 = resident(store, "w2", dev);
        let s = &scratch[rank];
        let g = &mut gpus.devices[dev];
        g.bind_thread().expect("bind");
        g.rmsnorm_f32(&s.x, w_fn, &s.xn2, EPS).unwrap();
        g.gemv_f32(w_1, &s.xn2, &s.g).unwrap();
        g.silu_f32(&s.g, &s.g).unwrap();
        g.gemv_f32(w_2, &s.g, &s.f).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
    }
    let refs: Vec<&_> = scratch.iter().map(|s| &s.f.buf).collect();
    gpus.all_reduce_sum_f32_peer(&group, &refs, D).unwrap();
    for (rank, &dev) in group.iter().enumerate() {
        let s = &scratch[rank];
        let g = &mut gpus.devices[dev];
        g.bind_thread().expect("bind");
        g.add_f32(&s.x, &s.f, &s.x).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
    }

    // Read the final replicated hidden off rank 0.
    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let mut out = vec![0u8; D * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &scratch[0].x.buf)
        .unwrap();
    bytes_to_f32(&out)
}
