// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-4 functional proof: an **end-to-end multi-layer tensor-parallel forward**
//! that matches the single-device forward, on real hardware. Where
//! `tp_gemv_parity` proves the isolated primitives, this runs a full L-layer
//! FFN-residual stack through the actual TP executor pattern the production
//! `dense_forward` needs — per-rank sharded weights, on-device rank loop, one
//! all-reduce per row-parallel op, and cross-layer residual — with NO host
//! round-trip between layers.
//!
//! Per layer (SwiGLU-free FFN, the TP-load-bearing subset of a transformer block):
//!   xn = rmsnorm(x, norm)           [replicated — every rank computes it]
//!   g  = W1 · xn                     [W1 column-parallel → rank r owns inter/tp rows]
//!   h  = silu(g)                     [elementwise on the on-rank intermediate slice]
//!   y  = all_reduce_r( W2_r · h_r )  [W2 row-parallel → partial per rank, summed]
//!   x  = x + y                       [residual; x stays replicated across ranks]
//!
//! The whole per-layer chain stays on each rank's device; only the FFN output
//! crosses ranks (one all-reduce). Validated vs a host F32 reference on an
//! emulated Tp-2 mesh (`HIPFIRE_EMULATE_GPUS=2`). INTER/TP is kept 64-aligned
//! (the `gemv_f32` reduction-dim constraint documented in `tp_gemv_parity`).
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_forward_parity

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::fulfill_manifest;
use rdna_compute::DType;

const D: usize = 128; // hidden dim
const INTER: usize = 128; // FFN intermediate (INTER/TP = 64, gemv-k-aligned)
const L: usize = 4; // layers
const TP: usize = 2;
const EPS: f32 = 1e-5;
const TOL: f32 = 2e-3;

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// Deterministic per-layer weights (host-side ground truth).
fn norm_w(l: usize) -> Vec<f32> {
    (0..D).map(|i| 1.0 + ((i + l) % 5) as f32 * 0.01).collect()
}
fn w1(l: usize) -> Vec<f32> {
    (0..INTER * D)
        .map(|i| (((i + 7 * l) % 13) as f32 - 6.0) * 0.02)
        .collect()
}
fn w2(l: usize) -> Vec<f32> {
    (0..D * INTER)
        .map(|i| (((i + 5 * l) % 11) as f32 - 5.0) * 0.02)
        .collect()
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

/// Host F32 reference: the exact same L-layer FFN-residual forward, whole weights.
fn host_forward(x0: &[f32]) -> Vec<f32> {
    let mut x = x0.to_vec();
    for l in 0..L {
        let nw = norm_w(l);
        let (w1l, w2l) = (w1(l), w2(l));
        let ms = x.iter().map(|v| v * v).sum::<f32>() / D as f32;
        let rms = (ms + EPS).sqrt();
        let xn: Vec<f32> = (0..D).map(|i| x[i] / rms * nw[i]).collect();
        let mut h = vec![0f32; INTER];
        for j in 0..INTER {
            let mut s = 0.0;
            for i in 0..D {
                s += w1l[j * D + i] * xn[i];
            }
            h[j] = silu(s);
        }
        for i in 0..D {
            let mut s = 0.0;
            for j in 0..INTER {
                s += w2l[i * INTER + j] * h[j];
            }
            x[i] += s;
        }
    }
    x
}

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, L) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_forward_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Input hidden (replicated on every rank).
    let x0: Vec<f32> = (0..D).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let y_ref = host_forward(&x0);

    // Shard every layer's weights: norm Replicate, W1 ColumnShard, W2 RowShard.
    let mut manifest = Vec::new();
    for l in 0..L {
        manifest.push(WeightEntry::layer(
            "norm",
            l,
            vec![D],
            DType::F32,
            ShardPolicy::Replicate,
        ));
        manifest.push(WeightEntry::layer(
            "w1",
            l,
            vec![INTER, D],
            DType::F32,
            ShardPolicy::ColumnShard { axis: 0 },
        ));
        manifest.push(WeightEntry::layer(
            "w2",
            l,
            vec![D, INTER],
            DType::F32,
            ShardPolicy::RowShard { axis: 1 },
        ));
    }
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let store = fulfill_manifest(&manifest, &mesh, L, &gpus, |e| {
        let bytes = match e.name.as_str() {
            "norm" => f32_to_bytes(&norm_w(e.layer.unwrap())),
            "w1" => f32_to_bytes(&w1(e.layer.unwrap())),
            _ => f32_to_bytes(&w2(e.layer.unwrap())),
        };
        Ok((bytes, DType::F32))
    })
    .expect("shard weights");

    // ── The TP forward: the reusable library executor (hipfire_runtime::tp_forward) ──
    let got = hipfire_runtime::tp_forward::tp_ffn_forward(
        &mut gpus, &mesh, &store, &x0, L, D, INTER, EPS,
    )
    .expect("tp_ffn_forward");
    let d_ref = max_abs_diff(&got, &y_ref);
    println!("[tp-forward] {L}-layer FFN-residual TP vs host: max|Δ|={d_ref:.2e}");
    assert!(
        d_ref < TOL,
        "TP forward diverges from host reference: max|Δ|={d_ref}"
    );

    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().expect("bind");
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!("tp_forward_parity: end-to-end multi-layer TP forward == single-device — FUNCTIONAL TP validated");
}
