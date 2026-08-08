// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-parallel forward executor (device-mesh P-B).
//!
//! The reusable TP execution pattern the production dense forward needs, lifted
//! out of the `tp_forward_parity` validation example into callable library code:
//! run a per-layer op sequence on **every rank of a `Tp` group**, each rank
//! computing with its **own sharded weights** (from a [`WeightStore`], placed by
//! [`crate::weight_store::fulfill_manifest`]'s Column/Row slicing), and inject a
//! single all-reduce after each **row-parallel** op — the Megatron dataflow:
//!
//! ```text
//!   xn = rmsnorm(x, norm)             (replicated: every rank computes it)
//!   g  = W1 · xn                      (W1 column-parallel → rank owns inter/tp rows)
//!   h  = silu(g)                      (elementwise on the on-rank slice)
//!   y  = all_reduce_r( W2_r · h_r )   (W2 row-parallel → partial per rank, summed)
//!   x  = x + y                        (residual; x stays replicated)
//! ```
//!
//! Only the row-parallel output crosses ranks (one collective per layer). The
//! hidden state stays replicated on each rank, so no broadcast is needed between
//! layers. This is the FFN half of a transformer block; attention adds the same
//! shape (column-parallel head-split QKV + row-parallel O-proj all-reduce) and
//! is layered on top when the arch forward adopts this executor.
//!
//! Validated end-to-end vs a single-device reference (`tp_forward_parity`,
//! gfx1151 emulated Tp-2): a 4-layer FFN-residual stack matches to max|Δ|=1.2e-7
//! with the ranks staying bit-identical.

use crate::multi_gpu::Gpus;
use crate::weight_store::{WeightHandle, WeightStore};
use hipfire_hardware::{DeviceMesh, DimKind};
use rdna_compute::{DType, GpuTensor};

/// Per-rank scratch for the TP FFN executor — allocated once, reused per layer.
struct RankScratch {
    /// Replicated hidden state on this rank (updated in place each layer).
    x: GpuTensor,
    /// rmsnorm output (replicated).
    xn: GpuTensor,
    /// Column-parallel intermediate `g`/`h` slice (`inter/tp`).
    g: GpuTensor,
    /// Row-parallel FFN output partial (`d`), all-reduced in place.
    p: GpuTensor,
}

fn resident<'a>(
    store: &'a WeightStore,
    name: &str,
    layer: usize,
    dev: usize,
) -> Result<&'a GpuTensor, String> {
    match store.get(name, Some(layer), dev) {
        Some(WeightHandle::Resident(t)) => Ok(t),
        Some(WeightHandle::Alias(_)) => Err(format!(
            "{name}[{layer}] on {dev} is an alias, not resident"
        )),
        None => Err(format!("{name}[{layer}] missing on device {dev}")),
    }
}

/// Run an `n_layers` FFN-residual stack tensor-parallel over the mesh's `Tp`
/// group, returning the final hidden state (read off rank 0; every rank holds
/// the same replicated result). `store` must hold, per layer, `"norm"`
/// (Replicate), `"w1"` (`ColumnShard{axis:0}`, shape `[inter, d]`) and `"w2"`
/// (`RowShard{axis:1}`, shape `[d, inter]`), placed on the Tp group.
///
/// `inter / tp` must be aligned to the F32 gemv reduction-dim tile (see
/// `tp_gemv_parity`); callers keep this via `validate_manifest`.
///
/// Preconditions the caller owns: `gpus` has per-device `active_stream` set and
/// peer access enabled (the all-reduce launches on the active stream).
#[allow(clippy::too_many_arguments)]
pub fn tp_ffn_forward(
    gpus: &mut Gpus,
    mesh: &DeviceMesh,
    store: &WeightStore,
    x0: &[f32],
    n_layers: usize,
    d: usize,
    inter: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let group = mesh.group_along(DimKind::Tp, &mesh.coord_of(0));
    let tp = group.len();
    if tp <= 1 {
        return Err(format!(
            "tp_ffn_forward: Tp group size {tp} — needs a Tp>1 mesh"
        ));
    }
    if inter % tp != 0 {
        return Err(format!(
            "tp_ffn_forward: inter {inter} not divisible by Tp {tp}"
        ));
    }
    let inter_r = inter / tp;

    // Per-rank scratch (x seeded with the replicated input).
    let x_bytes: Vec<u8> = x0.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let mut scratch: Vec<RankScratch> = Vec::with_capacity(tp);
    for &dev in &group {
        let g = gpus
            .devices
            .get_mut(dev)
            .ok_or_else(|| format!("device {dev} out of range"))?;
        g.bind_thread().map_err(|e| e.to_string())?;
        scratch.push(RankScratch {
            x: g.upload_raw(&x_bytes, &[d]).map_err(|e| e.to_string())?,
            xn: g
                .alloc_tensor(&[d], DType::F32)
                .map_err(|e| e.to_string())?,
            g: g.alloc_tensor(&[inter_r], DType::F32)
                .map_err(|e| e.to_string())?,
            p: g.alloc_tensor(&[d], DType::F32)
                .map_err(|e| e.to_string())?,
        });
    }

    for layer in 0..n_layers {
        // Per-rank op chain: rmsnorm → column gemv → silu → row gemv → partial.
        for (rank, &dev) in group.iter().enumerate() {
            let nw = resident(store, "norm", layer, dev)?;
            let w1 = resident(store, "w1", layer, dev)?;
            let w2 = resident(store, "w2", layer, dev)?;
            let s = &scratch[rank];
            let g = gpus
                .devices
                .get_mut(dev)
                .ok_or_else(|| format!("device {dev} out of range"))?;
            g.bind_thread().map_err(|e| e.to_string())?;
            g.rmsnorm_f32(&s.x, nw, &s.xn, eps)
                .map_err(|e| e.to_string())?;
            g.gemv_f32(w1, &s.xn, &s.g).map_err(|e| e.to_string())?;
            g.silu_f32(&s.g, &s.g).map_err(|e| e.to_string())?;
            g.gemv_f32(w2, &s.g, &s.p).map_err(|e| e.to_string())?;
            g.hip
                .stream_synchronize(g.active_stream.as_ref().unwrap())
                .map_err(|e| e.to_string())?;
        }
        // All-reduce the row-parallel FFN output over the Tp group.
        let refs: Vec<&_> = scratch.iter().map(|s| &s.p.buf).collect();
        gpus.all_reduce_sum_f32_peer(&group, &refs, d)
            .map_err(|e| e.to_string())?;
        // Residual add on each rank (x stays replicated: same x, same reduced y).
        for (rank, &dev) in group.iter().enumerate() {
            let s = &scratch[rank];
            let g = gpus
                .devices
                .get_mut(dev)
                .ok_or_else(|| format!("device {dev} out of range"))?;
            g.bind_thread().map_err(|e| e.to_string())?;
            g.add_f32(&s.x, &s.p, &s.x).map_err(|e| e.to_string())?;
            g.hip
                .stream_synchronize(g.active_stream.as_ref().unwrap())
                .map_err(|e| e.to_string())?;
        }
    }

    // Read the final replicated hidden off rank 0.
    let dev0 = group[0];
    gpus.devices[dev0]
        .bind_thread()
        .map_err(|e| e.to_string())?;
    let mut out = vec![0u8; d * 4];
    gpus.devices[dev0]
        .hip
        .memcpy_dtoh(&mut out, &scratch[0].x.buf)
        .map_err(|e| e.to_string())?;
    Ok(out
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
