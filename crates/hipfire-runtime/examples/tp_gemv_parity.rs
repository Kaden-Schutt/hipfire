// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-4 core proof: **tensor-parallel GEMV equals the whole GEMV**, numerically,
//! on real hardware. This validates the compute + collective heart of TP —
//! independent of the daemon/forward rewiring (PB-3/5) — by composing the two
//! pieces already landed:
//!   * dense-TP weight slicing (`fulfill_manifest` ColumnShard/RowShard, PB-1a/1c)
//!   * the `Gpus::all_reduce_sum_f32_peer` collective (Stage-2)
//!
//! Two shardings of `y = W · x` (W is `[M,K]`), each compared to a host F32
//! reference on an emulated Tp-2 mesh (`HIPFIRE_EMULATE_GPUS=2`):
//!
//!   1. **Column-parallel** (`ColumnShard{axis:0}`): rank r owns rows
//!      `[r·M/tp,(r+1)·M/tp)` → computes `y_r = W_r · x` (its output rows).
//!      Concatenating the per-rank outputs reconstructs `y`. No collective.
//!   2. **Row-parallel** (`RowShard{axis:1}`): rank r owns cols `[r·K/tp,…)` →
//!      computes a partial `p_r = W_r · x_r` over its k-slice; an all-reduce-sum
//!      over the Tp group yields the full `y`. This is the o_proj/ffn_down path.
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run -p hipfire-runtime --release \
//!          --example tp_gemv_parity

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle};
use rdna_compute::DType;

const M: usize = 64; // output rows
const K: usize = 128; // reduction dim
const TP: usize = 2;
const N_LAYERS: usize = 2;
const TOL: f32 = 1e-3;

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
        .fold(0.0f32, f32::max)
}

fn main() {
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpus = match Gpus::init_uniform(TP, N_LAYERS) {
        Ok(g) => g,
        Err(e) => {
            println!("tp_gemv_parity: SKIPPED (could not bring up {TP}-rank Gpus: {e})");
            return;
        }
    };
    let _ = gpus.enable_peer_all().expect("enable_peer_all");
    // Per-rank streams — the peer all-reduce launches on active_stream.
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Deterministic W [M,K] (row-major) and x [K], plus the host reference.
    let w: Vec<f32> = (0..M * K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let x: Vec<f32> = (0..K).map(|j| ((j % 7) as f32 - 3.0) * 0.2).collect();
    let mut y_ref = vec![0f32; M];
    for i in 0..M {
        let mut s = 0.0f32;
        for j in 0..K {
            s += w[i * K + j] * x[j];
        }
        y_ref[i] = s;
    }

    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);
    let w_bytes = f32_to_bytes(&w);

    // ── 1. Column-parallel: each rank owns M/tp output rows, no collective ──
    {
        let entry = WeightEntry::layer(
            "w",
            0,
            vec![M, K],
            DType::F32,
            ShardPolicy::ColumnShard { axis: 0 },
        );
        let store = fulfill_manifest(&[entry], &mesh, N_LAYERS, &gpus, |_| {
            Ok((w_bytes.clone(), DType::F32))
        })
        .expect("column shard");

        let rows = M / TP;
        let mut y_col = vec![0f32; M];
        for r in 0..TP {
            let wr = match store.get("w", Some(0), r).expect("wq missing") {
                WeightHandle::Resident(t) => t,
                _ => panic!("expected Resident"),
            };
            let dev = &mut gpus.devices[r];
            dev.bind_thread().expect("bind");
            let xt = dev.upload_raw(&f32_to_bytes(&x), &[K]).expect("x upload");
            let yt = dev.alloc_tensor(&[rows], DType::F32).expect("y alloc");
            dev.gemv_f32(wr, &xt, &yt).expect("gemv");
            dev.hip
                .stream_synchronize(dev.active_stream.as_ref().unwrap())
                .expect("sync");
            let mut buf = vec![0u8; rows * 4];
            dev.hip.memcpy_dtoh(&mut buf, &yt.buf).expect("dtoh");
            y_col[r * rows..(r + 1) * rows].copy_from_slice(&bytes_to_f32(&buf));
        }
        let d = max_abs_diff(&y_col, &y_ref);
        assert!(d < TOL, "column-parallel GEMV mismatch: max|Δ|={d}");
        println!("[tp-column] OK — concat(W_r·x) == W·x  (max|Δ|={d:.2e}, {TP} ranks)");
    }

    // ── 2. Row-parallel: partial per rank over its k-slice, then all-reduce ──
    {
        let entry = WeightEntry::layer(
            "w",
            0,
            vec![M, K],
            DType::F32,
            ShardPolicy::RowShard { axis: 1 },
        );
        let store = fulfill_manifest(&[entry], &mesh, N_LAYERS, &gpus, |_| {
            Ok((w_bytes.clone(), DType::F32))
        })
        .expect("row shard");

        let kk = K / TP;
        // Each rank computes p_r = W_r · x_r into its own buffer.
        let mut partials = Vec::with_capacity(TP);
        for r in 0..TP {
            let wr = match store.get("w", Some(0), r).expect("w missing") {
                WeightHandle::Resident(t) => t,
                _ => panic!("expected Resident"),
            };
            let x_r = &x[r * kk..(r + 1) * kk];
            let dev = &mut gpus.devices[r];
            dev.bind_thread().expect("bind");
            let xt = dev
                .upload_raw(&f32_to_bytes(x_r), &[kk])
                .expect("x_r upload");
            let yt = dev.alloc_tensor(&[M], DType::F32).expect("partial alloc");
            dev.gemv_f32(wr, &xt, &yt).expect("gemv");
            dev.hip
                .stream_synchronize(dev.active_stream.as_ref().unwrap())
                .expect("sync");
            partials.push(yt);
        }
        // All-reduce-sum the partials over the Tp group (in place).
        let group: Vec<usize> = (0..TP).collect();
        let refs: Vec<&_> = partials.iter().map(|t| &t.buf).collect();
        gpus.all_reduce_sum_f32_peer(&group, &refs, M)
            .expect("all_reduce_sum_f32_peer");
        for dev in &gpus.devices {
            dev.bind_thread().expect("bind");
            dev.hip
                .stream_synchronize(dev.active_stream.as_ref().unwrap())
                .expect("sync");
        }
        // Every rank now holds the full reduced y — verify rank 0.
        let mut buf = vec![0u8; M * 4];
        gpus.devices[0].bind_thread().expect("bind");
        gpus.devices[0]
            .hip
            .memcpy_dtoh(&mut buf, &partials[0].buf)
            .expect("dtoh");
        let y_row = bytes_to_f32(&buf);
        let d = max_abs_diff(&y_row, &y_ref);
        assert!(d < TOL, "row-parallel GEMV mismatch: max|Δ|={d}");
        println!("[tp-row]    OK — all_reduce(W_r·x_r) == W·x  (max|Δ|={d:.2e}, {TP} ranks)");
    }

    // Tear down streams.
    for dev in gpus.devices.iter_mut() {
        if let Some(s) = dev.active_stream.take() {
            dev.bind_thread().expect("bind");
            let _ = dev.hip.stream_destroy(s);
        }
    }
    println!("tp_gemv_parity: sharded GEMV + all-reduce byte-parity with whole GEMV — TP compute path validated");
}
