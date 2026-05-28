// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stage 3a smoke: validate the tensor-parallel **row-parallel `wo`**
//! decomposition on the *real* quantized `wo` of `qwen3.5-0.8b.mq4`.
//!
//! Claim under test (TP plan §3.2, the load-bearing 3a mechanism):
//! row-sharding `wo` = each rank consumes only its local Q-heads' slice of
//! the attention output and produces a *partial* residual contribution;
//! summing the partials across ranks via `Gpus::all_reduce_sum_f32`
//! reconstructs the full `wo @ attn_out` within fp tolerance.
//!
//! This isolates the all-reduce/decomposition correctness (the genuinely
//! novel TP integration) from the `forward_scratch_tp` layer-loop plumbing
//! and the `wo` quant-column-slice-load problem (deferred to milestone 3b).
//! It does NOT yet shard the attention compute itself — it feeds the real
//! `wo` GEMV a masked full-width input, which is mathematically identical to
//! a column slice but needs no quant slicer.
//!
//! Each rank `r` zeroes `attn_out` outside `ShardConfig::wo_col_range(r)`,
//! runs the full `wo` GEMV (zeros contribute exactly 0), then the partials
//! are all-reduce-summed. Result must match a single-GPU `wo @ attn_out`.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 cargo run --release -p hipfire-arch-qwen35 \
//!       --example tp_wo_allreduce_smoke -- ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{self, LayerType, LayerWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, WeightTensor};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::GpuTensor;
use std::path::Path;

const TP: usize = 2;
const REL_TOL: f32 = 1e-4;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3.5-0.8b.mq4",
            std::env::var("HOME").unwrap_or_else(|_| "/home/kaden".into())
        )
    });

    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    println!("=== TP wo-partial + all-reduce smoke (Stage 3a) ===");
    println!(
        "model: {path}\nconfig: {} layers, dim={}, n_heads={}, n_kv_heads={}, head_dim={}, num_experts={}",
        config.n_layers, config.dim, config.n_heads, config.n_kv_heads, config.head_dim, config.num_experts
    );

    // Shard config — clean TP=2 GQA split (n_kv_heads must divide tp).
    let shard = ShardConfig::new(TP, false, config.num_experts, ExpertAssign::Stride)
        .expect("ShardConfig::new");
    shard
        .validate(config.n_heads, config.n_kv_heads)
        .expect("ShardConfig::validate head geometry");
    println!(
        "shard: tp={}, q_heads/rank={}, wo_col_range(0)={:?}, wo_col_range(1)={:?}",
        TP,
        shard.q_heads_per_rank(config.n_heads),
        shard.wo_col_range(0, config.n_heads, config.head_dim),
        shard.wo_col_range(1, config.n_heads, config.head_dim),
    );

    let mut gpus = Gpus::init_tp(TP, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let s = dev.hip.stream_create().expect("stream_create");
        dev.active_stream = Some(s);
    }

    // Replicated load: full model on every rank (Stage 3a uses full wo).
    let mut weights = Vec::with_capacity(TP);
    for r in 0..TP {
        gpus.devices[r].bind_thread().expect("bind");
        let w = qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).expect("load_weights");
        weights.push(w);
    }

    let attn_dim = config.n_heads * config.head_dim; // wo input dim (k)
    let dim = config.dim; // wo output dim (m)

    // qwen3.5-0.8b is a HYBRID: FullAttention layers are periodic (every
    // 4th), DeltaNet (LinearAttention) elsewhere. Pick the first FullAttn
    // layer for the wo decomposition test.
    let fa_layer = config
        .layer_types
        .iter()
        .position(|&t| t == LayerType::FullAttention)
        .expect("model has at least one FullAttention layer");
    println!("using FullAttn layer {fa_layer} (model is hybrid DeltaNet+FullAttn)");

    let wo: Vec<&WeightTensor> = (0..TP)
        .map(|r| match &weights[r].layers[fa_layer] {
            LayerWeights::FullAttn(l) => &l.wo,
            other => panic!(
                "layer {fa_layer} is not FullAttn ({:?})",
                std::mem::discriminant(other)
            ),
        })
        .collect();
    assert_eq!(wo[0].m, dim, "wo.m (output) should equal dim");
    assert_eq!(wo[0].k, attn_dim, "wo.k (input) should equal n_heads*head_dim");

    // Deterministic attention output, identical across ranks.
    let attn_out: Vec<f32> = (0..attn_dim)
        .map(|i| ((i % 97) as f32) * 0.013 - 0.6)
        .collect();

    // ── Reference: single-GPU full wo @ attn_out on rank 0. ───────────
    gpus.devices[0].bind_thread().expect("bind");
    let x_full = gpus.devices[0]
        .upload_f32(&attn_out, &[attn_dim])
        .expect("upload x_full");
    let y_full = gpus.devices[0]
        .upload_f32(&vec![0.0f32; dim], &[dim])
        .expect("alloc y_full");
    llama::weight_gemv(&mut gpus.devices[0], wo[0], &x_full, &y_full).expect("reference gemv");
    gpus.devices[0].hip.device_synchronize().expect("sync ref");
    let reference = gpus.devices[0].download_f32(&y_full).expect("download ref");

    // ── TP: per-rank masked partial wo, then all-reduce-sum. ──────────
    let mut keep_x: Vec<GpuTensor> = Vec::with_capacity(TP);
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(TP);
    for r in 0..TP {
        let range = shard.wo_col_range(r, config.n_heads, config.head_dim);
        let mut masked = vec![0.0f32; attn_dim];
        masked[range.clone()].copy_from_slice(&attn_out[range]);

        gpus.devices[r].bind_thread().expect("bind");
        let x_r = gpus.devices[r]
            .upload_f32(&masked, &[attn_dim])
            .expect("upload masked");
        let y_r = gpus.devices[r]
            .upload_f32(&vec![0.0f32; dim], &[dim])
            .expect("alloc partial");
        llama::weight_gemv(&mut gpus.devices[r], wo[r], &x_r, &y_r).expect("partial gemv");
        keep_x.push(x_r);
        partials.push(y_r);
    }
    // Ensure all per-rank GEMVs (default stream) complete before the
    // all-reduce (which runs on each device's active_stream).
    for dev in &gpus.devices {
        dev.bind_thread().expect("bind");
        dev.hip.device_synchronize().expect("sync gemv");
    }

    let refs: Vec<&_> = partials.iter().map(|p| &p.buf).collect();
    gpus.all_reduce_sum_f32(&refs, dim).expect("all_reduce_sum_f32");
    for dev in &gpus.devices {
        dev.bind_thread().expect("bind");
        dev.hip
            .stream_synchronize(dev.active_stream.as_ref().unwrap())
            .expect("sync all_reduce");
    }

    // ── Compare every rank's reduced buffer to the reference. ─────────
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    let ref_max = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut all_ok = true;
    for r in 0..TP {
        gpus.devices[r].bind_thread().expect("bind");
        let got = gpus.devices[r].download_f32(&partials[r]).expect("download partial");
        let (mut rmax_abs, mut rmax_rel) = (0.0f32, 0.0f32);
        for i in 0..dim {
            let abs = (got[i] - reference[i]).abs();
            let rel = abs / (ref_max + 1e-12);
            rmax_abs = rmax_abs.max(abs);
            rmax_rel = rmax_rel.max(rel);
        }
        let ok = rmax_rel < REL_TOL;
        println!(
            "  rank {r}: max|Δ|={rmax_abs:.3e}, max rel={rmax_rel:.3e}  (ref max |y|={ref_max:.3e})  {}",
            if ok { "OK" } else { "FAIL" }
        );
        worst_abs = worst_abs.max(rmax_abs);
        worst_rel = worst_rel.max(rmax_rel);
        all_ok &= ok;
    }

    // Cleanup streams.
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        if let Some(s) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(s);
        }
    }

    println!(
        "\nworst across ranks: max|Δ|={worst_abs:.3e}, max rel={worst_rel:.3e}, tol={REL_TOL:.0e}"
    );
    assert!(
        all_ok,
        "TP wo-partial + all-reduce did not reconstruct single-GPU wo @ attn_out within {REL_TOL:.0e}"
    );
    println!("tp_wo_allreduce_smoke: PASS");
}
