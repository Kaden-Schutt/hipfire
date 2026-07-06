// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-TP3 proof: a **whole SwiGLU FFN block** through `execute_steps_tp`. PB-TP2
//! did rmsnorm + a column→row linear pair but stopped short of the FFN's silu —
//! the Step IR had no activation op (silu was fused into gate-up kernels). PB-TP3
//! adds `Step::SiluMul { gate, up, out }` (→ `gpu.silu_mul_f32`) so the FFN is one
//! contiguous per-rank step list the executor shards:
//! ```text
//!   xn   = rmsnorm(x, norm)        [Replicate → None]
//!   g_r  = W_gate_r · xn           [ColumnShard → None]
//!   u_r  = W_up_r   · xn           [ColumnShard → None]
//!   h_r  = silu(g_r) * u_r         [SiluMul — elementwise on the on-rank inter/tp slice → None]
//!   o    = all_reduce_r(W_down_r · h_r)   [RowShard → AllReduceOut{D}]
//! ```
//! The gate/up column shards leave the intermediate `inter/tp` on-rank; `SiluMul`
//! runs elementwise on that slice (no cross-rank dependency); only `W_down`'s
//! output crosses ranks (one all-reduce). Collectives are DERIVED from each
//! weight's `ShardPolicy` (PB-TP2); the silu step carries no weight → always None.
//! Validated vs a single-device reference (same kernels) on emulated Tp-2
//! (`HIPFIRE_EMULATE_GPUS=2`, gfx1151). INTER/TP kept 64-aligned.
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_execute_steps_ffn_parity

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
fn w_gate() -> Vec<f32> {
    (0..INTER * D)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect()
}
fn w_up() -> Vec<f32> {
    (0..INTER * D)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.017)
        .collect()
}
fn w_down() -> Vec<f32> {
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
fn collective_for(policy: &ShardPolicy, out_dim: usize) -> TpCollective {
    match collective_for_policy(policy) {
        Some(CollectiveHint::AllReduce { kind: DimKind::Tp }) => {
            TpCollective::AllReduceOut { dim: out_dim }
        }
        _ => TpCollective::None,
    }
}
fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, TP) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "tp_execute_steps_ffn_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})"
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

    // ── Single-device reference: full SwiGLU FFN block (GPU kernels) ──
    let y_ref = {
        let g = &mut gpus.devices[0];
        g.bind_thread().unwrap();
        let x = g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap();
        let nw = g.upload_raw(&f32_to_bytes(&norm_w()), &[D]).unwrap();
        let wg = g.upload_raw(&f32_to_bytes(&w_gate()), &[INTER, D]).unwrap();
        let wu = g.upload_raw(&f32_to_bytes(&w_up()), &[INTER, D]).unwrap();
        let wd = g.upload_raw(&f32_to_bytes(&w_down()), &[D, INTER]).unwrap();
        let xn = g.alloc_tensor(&[D], DType::F32).unwrap();
        let gt = g.alloc_tensor(&[INTER], DType::F32).unwrap();
        let ut = g.alloc_tensor(&[INTER], DType::F32).unwrap();
        let ht = g.alloc_tensor(&[INTER], DType::F32).unwrap();
        let ot = g.alloc_tensor(&[D], DType::F32).unwrap();
        g.rmsnorm_f32(&x, &nw, &xn, EPS).unwrap();
        g.gemv_f32(&wg, &xn, &gt).unwrap();
        g.gemv_f32(&wu, &xn, &ut).unwrap();
        g.silu_mul_f32(&gt, &ut, &ht).unwrap();
        g.gemv_f32(&wd, &ht, &ot).unwrap();
        g.hip
            .stream_synchronize(g.active_stream.as_ref().unwrap())
            .unwrap();
        let mut out = vec![0u8; D * 4];
        g.hip.memcpy_dtoh(&mut out, &ot.buf).unwrap();
        bytes_to_f32(&out)
    };

    // A tiny host-math cross-check on the reference (silu path sanity).
    {
        let ms = x0.iter().map(|v| v * v).sum::<f32>() / D as f32;
        let rms = (ms + EPS).sqrt();
        let nw = norm_w();
        let xn: Vec<f32> = (0..D).map(|i| x0[i] / rms * nw[i]).collect();
        let (wg, wu, wd) = (w_gate(), w_up(), w_down());
        let mut h = vec![0f32; INTER];
        for j in 0..INTER {
            let mut gs = 0.0;
            let mut us = 0.0;
            for i in 0..D {
                gs += wg[j * D + i] * xn[i];
                us += wu[j * D + i] * xn[i];
            }
            h[j] = silu(gs) * us;
        }
        let mut o = vec![0f32; D];
        for i in 0..D {
            let mut s = 0.0;
            for j in 0..INTER {
                s += wd[i * INTER + j] * h[j];
            }
            o[i] = s;
        }
        let d = max_abs_diff(&o, &y_ref);
        assert!(
            d < TOL,
            "GPU reference disagrees with host SwiGLU: max|Δ|={d}"
        );
    }

    // ── TP path ──
    let p_norm = ShardPolicy::Replicate;
    let p_col = ShardPolicy::ColumnShard { axis: 0 };
    let p_row = ShardPolicy::RowShard { axis: 1 };
    let manifest = vec![
        WeightEntry::layer("norm", 0, vec![D], DType::F32, p_norm.clone()),
        WeightEntry::layer("w_gate", 0, vec![INTER, D], DType::F32, p_col.clone()),
        WeightEntry::layer("w_up", 0, vec![INTER, D], DType::F32, p_col.clone()),
        WeightEntry::layer("w_down", 0, vec![D, INTER], DType::F32, p_row.clone()),
    ];
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, 1, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "norm" => f32_to_bytes(&norm_w()),
            "w_gate" => f32_to_bytes(&w_gate()),
            "w_up" => f32_to_bytes(&w_up()),
            _ => f32_to_bytes(&w_down()),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let inter_r = INTER / TP;

    let mut x_r = Vec::with_capacity(TP);
    let mut xplain_r = Vec::with_capacity(TP);
    let mut xn_r = Vec::with_capacity(TP);
    let mut g_r = Vec::with_capacity(TP);
    let mut u_r = Vec::with_capacity(TP);
    let mut h_r = Vec::with_capacity(TP);
    let mut o_r = Vec::with_capacity(TP);
    for &dev in &group {
        let g = &mut gpus.devices[dev];
        g.bind_thread().unwrap();
        x_r.push(g.upload_raw(&f32_to_bytes(&x0), &[D]).unwrap());
        xplain_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
        xn_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
        g_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        u_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        h_r.push(g.alloc_tensor(&[inter_r], DType::F32).unwrap());
        o_r.push(g.alloc_tensor(&[D], DType::F32).unwrap());
    }
    let norm_refs: Vec<&GpuTensor> = group
        .iter()
        .map(|&dev| resident(&store, "norm", dev))
        .collect();
    let wg_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_gate", dev), inter_r, D))
        .collect();
    let wu_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_up", dev), inter_r, D))
        .collect();
    let wd_refs: Vec<WeightRef> = group
        .iter()
        .map(|&dev| wref(resident(&store, "w_down", dev), D, inter_r))
        .collect();

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
                    w: &wg_refs[r],
                    input: GemvInput::Raw(&xn_r[r]),
                    out: &g_r[r],
                },
                Step::Gemv {
                    w: &wu_refs[r],
                    input: GemvInput::Raw(&xn_r[r]),
                    out: &u_r[r],
                },
                Step::SiluMul {
                    gate: &g_r[r],
                    up: &u_r[r],
                    out: &h_r[r],
                },
                Step::Gemv {
                    w: &wd_refs[r],
                    input: GemvInput::Raw(&h_r[r]),
                    out: &o_r[r],
                },
            ]
        })
        .collect();

    // Collectives: weight-bearing steps derive from ShardPolicy; SiluMul (no
    // weight, elementwise on the on-rank slice) is always None.
    let collectives = vec![
        collective_for(&p_norm, D),
        collective_for(&p_col, inter_r),
        collective_for(&p_col, inter_r),
        TpCollective::None,
        collective_for(&p_row, D),
    ];

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
    println!("[tp-execute-steps] full SwiGLU FFN block through execute_steps_tp vs single-device: max|Δ|={d_ref:.2e}");
    assert!(
        d_ref < TOL,
        "execute_steps_tp FFN diverges from single-device: max|Δ|={d_ref}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().unwrap();
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!("tp_execute_steps_ffn_parity: full SwiGLU FFN (Step::SiluMul) through execute_steps_tp == single-device — PB-TP3 validated");
}
