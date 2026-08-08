// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP1 proof: tensor-parallelism *inside the executor*. Where
//! `tp_forward_parity`/`tp_attn_parity` hand-roll the per-rank op loop, this
//! routes a column→row GEMV pair **through `execute_steps_tp`** — the Step-IR
//! executor itself shards across the mesh's `Tp` group and injects the Tp
//! all-reduce after the row-parallel step. Same weights, same collective, but
//! now the *executor* owns the parallelism (the pivot's grand-unify thesis).
//!
//! Per rank (a pure-linear column→row pair — no silu; the Step IR has no
//! activation op yet, so PB-TP1 validates the executor's shard + collective on
//! GEMV alone):
//! ```text
//!   g_r = W1_r · x        [W1 column-parallel → rank owns inter/tp rows]   (TpCollective::None)
//!   o   = all_reduce_r( W2_r · g_r )   [W2 row-parallel → partial per rank, summed]  (AllReduceOut)
//! ```
//! After the executor returns, every rank holds `o == W2·W1·x`. Validated vs a
//! single-device reference computed with the same GEMV kernel on emulated Tp-2
//! (`HIPFIRE_EMULATE_GPUS=2`, gfx1151). INTER/TP is kept 64-aligned (the
//! `gemv_f32` reduction-dim constraint from `tp_gemv_parity`).
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_parity

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{execute_steps_tp, GemvInput, Step, TpCollective};
use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, GpuTensor};

const D: usize = 128; // input/output dim
const INTER: usize = 128; // intermediate (INTER/TP = 64, gemv-k-aligned)
const TP: usize = 2;
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

/// f32 WeightRef over a resident tensor (no rotation, no AWQ).
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

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_execute_steps_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})");
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

    // ── Single-device reference: g = W1·x ; o = W2·g (whole weights, one GPU) ──
    let y_ref = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let x = g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap();
        let w_1 = g.upload_raw(&f32_to_bytes(&w1()), &[INTER, D]).unwrap();
        let w_2 = g.upload_raw(&f32_to_bytes(&w2()), &[D, INTER]).unwrap();
        let gt = g.alloc_tensor(&[INTER], DType::F32).unwrap();
        let ot = g.alloc_tensor(&[D], DType::F32).unwrap();
        g.gemv_f32(&w_1, &x, &gt).unwrap();
        g.gemv_f32(&w_2, &gt, &ot).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let mut out = vec![0u8; D * 4];
        g.hip.memcpy_dtoh(&mut out, &ot.buf).unwrap();
        bytes_to_f32(&out)
    };

    // ── TP path: shard W1 (Column) + W2 (Row), run through execute_steps_tp ──
    let manifest = vec![
        WeightEntry::layer(
            "w1",
            0,
            vec![INTER, D],
            DType::F32,
            ShardPolicy::ColumnShard { axis: 0 },
        ),
        WeightEntry::layer(
            "w2",
            0,
            vec![D, INTER],
            DType::F32,
            ShardPolicy::RowShard { axis: 1 },
        ),
    ];
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "w1" => f32_to_bytes(&w1()),
            _ => f32_to_bytes(&w2()),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let inter_r = INTER / TP;

    // Per-rank buffers (owned here; the Steps borrow them).
    let mut x_r = Vec::with_capacity(TP);
    let mut g_r = Vec::with_capacity(TP);
    let mut o_r = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        x_r.push(g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap());
        g_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        o_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
    }
    // Per-rank sharded WeightRefs (W1_r: [inter/tp, D] column; W2_r: [D, inter/tp] row).
    let w1_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w1", dev), inter_r, D))
        .collect();
    let w2_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w2", dev), D, inter_r))
        .collect();

    // Per-rank Step lists (lock-step): column gemv (no collective) → row gemv (all-reduce).
    let per_rank_steps: Vec<Vec<Step>> = (0..TP)
        .map(|r| {
            vec![
                Step::Gemv {
                    w: &w1_refs[r],
                    input: GemvInput::Raw(&x_r[r]),
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
    let collectives = vec![TpCollective::None, TpCollective::AllReduceOut { dim: D }];

    execute_steps_tp(&mesh, &mut gpus, &per_rank_steps, &collectives).expect("execute_steps_tp");

    // Read the (now-replicated) result off rank 0.
    let dev0 = group[0];
    gpus.devices[dev0].bind_thread().unwrap();
    let mut out = vec![0u8; D * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &o_r[0].buf)
        .unwrap();
    let y_tp = bytes_to_f32(&out);

    let d_ref = max_abs_diff(&y_tp, &y_ref);
    println!("[tp-execute-steps] column→row GEMV through execute_steps_tp vs single-device: max|Δ|={d_ref:.2e}");
    assert!(
        d_ref < TOL,
        "execute_steps_tp diverges from single-device: max|Δ|={d_ref}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!(
        "tp_execute_steps_parity: TP inside execute_steps == single-device — PB-TP1 validated"
    );
}
