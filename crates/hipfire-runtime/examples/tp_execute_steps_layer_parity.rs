// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP2 proof: a **replicated `RmsnormAutomatic` + manifest-DERIVED collective
//! plan** through `execute_steps_tp`. PB-TP1 proved the executor shards a
//! column→row GEMV pair with a *hand-written* collective list; PB-TP2 adds two
//! things the real forward needs:
//!
//!  1. A **non-GEMV replicated step** (`RmsnormAutomatic`, `RotationPlan::None` →
//!     plain F32 rmsnorm) flows through the TP executor unchanged — every rank
//!     computes the same norm on the same replicated hidden, no collective.
//!  2. The `collectives` list is **derived from each step's weight `ShardPolicy`**
//!     via `collective_for_policy` (the single-source-of-truth partitioner:
//!     `RowShard → AllReduce{Tp}`, everything else → `None`), instead of being
//!     hand-authored — so a forgotten reduce is structurally impossible.
//!
//! Per rank (the FFN block's linear skeleton — silu is still deferred, the Step
//! IR has no activation op; PB-TP3):
//! ```text
//!   xn  = rmsnorm(x, norm)     [Replicate → derived TpCollective::None]
//!   g_r = W1_r · xn            [ColumnShard → derived None]
//!   o   = all_reduce_r(W2_r · g_r)  [RowShard → derived AllReduceOut{D}]
//! ```
//! Validated vs a single-device reference (same rmsnorm + GEMV kernels) on
//! emulated Tp-2 (`HIPFIRE_EMULATE_GPUS=2`, gfx1151). INTER/TP kept 64-aligned.
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_layer_parity

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_dispatch::types::RotationPlan;
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{collective_for_policy, ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, GpuTensor};

const D: usize = 128;
const INTER: usize = 128; // INTER/TP = 64, gemv-k-aligned
const TP: usize = 2;
const EPS: f32 = 1e-5;
const TOL: f32 = 2e-3;

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

fn norm_w() -> Vec<f32> {
    (0..D).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect()
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

/// The single-source-of-truth derivation PB-TP2 demonstrates: a step's
/// `ShardPolicy` decides its output collective. Row-sharded → all-reduce the
/// step's `out` (of length `out_dim`) over the `Tp` group; everything else → none.
fn collective_for(policy: &ShardPolicy, out_dim: usize) -> TpCollective {
    match collective_for_policy(policy) {
        Some(CollectiveHint::AllReduce { kind: DimKind::Tp }) => {
            TpCollective::AllReduceOut { dim: out_dim }
        }
        _ => TpCollective::None,
    }
}

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "tp_execute_steps_layer_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})"
            );
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

    // ── Single-device reference: xn = rmsnorm(x); g = W1·xn; o = W2·g ──
    let y_ref = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let x = g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap();
        let nw = g.upload_raw(&f32_to_bytes(&norm_w()), &[D]).unwrap();
        let w_1 = g.upload_raw(&f32_to_bytes(&w1()), &[INTER, D]).unwrap();
        let w_2 = g.upload_raw(&f32_to_bytes(&w2()), &[D, INTER]).unwrap();
        let xn = g.alloc_tensor(&[D], DType::F32).unwrap();
        let gt = g.alloc_tensor(&[INTER], DType::F32).unwrap();
        let ot = g.alloc_tensor(&[D], DType::F32).unwrap();
        g.rmsnorm_f32(&x, &nw, &xn, EPS).unwrap();
        g.gemv_f32(&w_1, &xn, &gt).unwrap();
        g.gemv_f32(&w_2, &gt, &ot).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let mut out = vec![0u8; D * 4];
        g.hip.memcpy_dtoh(&mut out, &ot.buf).unwrap();
        bytes_to_f32(&out)
    };

    // ── TP path: manifest (norm Replicate, W1 Column, W2 Row) → sharded store ──
    let p_norm = ShardPolicy::Replicate;
    let p_w1 = ShardPolicy::ColumnShard { axis: 0 };
    let p_w2 = ShardPolicy::RowShard { axis: 1 };
    let manifest = vec![
        WeightEntry::layer("norm", 0, vec![D], DType::F32, p_norm.clone()),
        WeightEntry::layer("w1", 0, vec![INTER, D], DType::F32, p_w1.clone()),
        WeightEntry::layer("w2", 0, vec![D, INTER], DType::F32, p_w2.clone()),
    ];
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "norm" => f32_to_bytes(&norm_w()),
            "w1" => f32_to_bytes(&w1()),
            _ => f32_to_bytes(&w2()),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let inter_r = INTER / TP;

    // Per-rank buffers.
    let mut x_r = Vec::with_capacity(TP);
    let mut xplain_r = Vec::with_capacity(TP);
    let mut xn_r = Vec::with_capacity(TP);
    let mut g_r = Vec::with_capacity(TP);
    let mut o_r = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        x_r.push(g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap());
        xplain_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
        xn_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
        g_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        o_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
    }
    // Per-rank sharded weight refs.
    let norm_refs: Vec<&GpuTensor> = group
        .iter()
        .map(|&dev| resident(&store, "norm", dev))
        .collect();
    let w1_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w1", dev), inter_r, D))
        .collect();
    let w2_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w2", dev), D, inter_r))
        .collect();

    // Per-rank Step lists: rmsnorm → column gemv → row gemv.
    let per_rank_steps: Vec<Vec<Step>> = (0..TP)
        .map(|r| {
            vec![
                Step::RmsnormAutomatic {
                    x: &x_r[r],
                    norm_weight: norm_refs[r],
                    x_plain: &xplain_r[r],
                    out: &xn_r[r],
                    awq_scale: None,
                    k: D,
                    eps: EPS,
                    rotation: RotationPlan::None,
                },
                Step::Gemv {
                    w: &w1_refs[r],
                    input: GemvInput::Raw(&xn_r[r]),
                    out: &g_r[r],
                },
                Step::Gemv {
                    w: &w2_refs[r],
                    input: GemvInput::Raw(&g_r[r]),
                    out: &o_r[r],
                },
            ]
        })
        .collect();

    // Collectives DERIVED from each step's ShardPolicy (single source of truth).
    let collectives = vec![
        collective_for(&p_norm, D),
        collective_for(&p_w1, inter_r),
        collective_for(&p_w2, D),
    ];
    assert!(
        matches!(collectives[0], TpCollective::None)
            && matches!(collectives[1], TpCollective::None)
            && matches!(collectives[2], TpCollective::AllReduceOut { dim } if dim == D),
        "derived collective plan must be [None, None, AllReduceOut(D)]"
    );

    execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");

    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let mut out = vec![0u8; D * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &o_r[0].buf)
        .unwrap();
    let y_tp = bytes_to_f32(&out);

    let d_ref = max_abs_diff(&y_tp, &y_ref);
    println!("[tp-execute-steps] rmsnorm→col→row (derived collectives) vs single-device: max|Δ|={d_ref:.2e}");
    assert!(
        d_ref < TOL,
        "execute_steps_tp layer diverges from single-device: max|Δ|={d_ref}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!("tp_execute_steps_layer_parity: rmsnorm + manifest-derived collectives through execute_steps_tp == single-device — PB-TP2 validated");
}
