// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stage 3a: end-to-end parity of the FULL FullAttn TP machinery against
//! the single-GPU path, on one real FullAttn layer of qwen3.5-0.8b.mq4.
//!
//! Validates the complete `FaPhase` TP flow — attention (replicated) →
//! `sigmoid_mul` → mask to local Q-heads (`mul_f32`) → PARTIAL `wo` into
//! `s.o` → `all_reduce_sum_f32` across ranks → `add_f32` into `s.x` → FFN —
//! produces the same layer output as single-GPU `run_fa_layer_body(Full)`
//! (attention → residual `wo` → FFN). This is the novel/risky TP piece;
//! DeltaNet layers (run replicated in the full `forward_scratch_tp`) reuse
//! existing code unchanged and are not exercised here.
//!
//! Reference (rank 0, fresh scratch/kv): inject input → s.x, run
//! `FaPhase::Full`. TP (per rank): inject the same input → s.x, run
//! `FaPhase::TpAttn{mask_r}` (partial in s.o), all-reduce s.o, add into
//! s.x, run `FaPhase::TpFfn`. Compare rank-0 s.x to the reference.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 cargo run --release -p hipfire-arch-qwen35 \
//!       --features deltanet --example tp_fa_layer_parity -- \
//!       ~/.hipfire/models/qwen3.5-0.8b.mq4

use hipfire_arch_qwen35::qwen35::{self, FaPhase, LayerType, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use std::path::Path;

const TP: usize = 2;
const REL_TOL: f32 = 1e-4;

fn set_input(gpu: &rdna_compute::Gpu, s: &Qwen35Scratch, x: &[f32], pos: i32) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x))
    };
    gpu.hip.memcpy_htod(&s.x.buf, bytes).expect("htod s.x");
    gpu.hip
        .memcpy_htod(&s.pos_buf, &pos.to_ne_bytes())
        .expect("htod pos_buf");
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3.5-0.8b.mq4",
            std::env::var("HOME").unwrap_or_else(|_| "/home/kaden".into())
        )
    });

    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    let dim = config.dim;
    let attn_dim = config.n_heads * config.head_dim;
    println!("=== TP FullAttn-layer parity (Stage 3a) ===");
    println!(
        "dim={dim}, n_heads={}, n_kv_heads={}, head_dim={}, attn_dim={attn_dim}",
        config.n_heads, config.n_kv_heads, config.head_dim
    );

    let shard = ShardConfig::new(TP, false, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();

    let fa_layer = config
        .layer_types
        .iter()
        .position(|&t| t == LayerType::FullAttention)
        .expect("a FullAttn layer");
    println!("FullAttn layer under test: {fa_layer}");

    let mut gpus = Gpus::init_tp(TP, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().expect("bind");
        let st = dev.hip.stream_create().expect("stream");
        dev.active_stream = Some(st);
    }

    // Replicated weights on every rank.
    let mut weights = Vec::with_capacity(TP);
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap());
    }

    // Per-rank scratch + kv (fresh, pos 0).
    let mut scratches = Vec::with_capacity(TP);
    let mut kvs = Vec::with_capacity(TP);
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &config, 128).unwrap());
        kvs.push(
            KvCache::new_gpu_q8(&mut gpus.devices[r], config.n_layers, config.n_kv_heads, config.head_dim, 256)
                .unwrap(),
        );
    }

    // Per-rank head mask over the attention output [attn_dim]: 1.0 on this
    // rank's local Q-heads (wo_col_range), 0.0 elsewhere.
    let mut masks = Vec::with_capacity(TP);
    for r in 0..TP {
        let range = shard.wo_col_range(r, config.n_heads, config.head_dim);
        let mut m = vec![0.0f32; attn_dim];
        m[range].iter_mut().for_each(|v| *v = 1.0);
        gpus.devices[r].bind_thread().unwrap();
        masks.push(gpus.devices[r].upload_f32(&m, &[attn_dim]).unwrap());
    }

    // Deterministic layer input, identical on ref + every rank.
    let input: Vec<f32> = (0..dim).map(|i| ((i % 31) as f32) * 0.03 - 0.45).collect();

    // ── Reference: single-GPU FaPhase::Full on a fresh ref scratch/kv. ─
    gpus.devices[0].bind_thread().unwrap();
    let ref_scratch = Qwen35Scratch::new(&mut gpus.devices[0], &config, 128).unwrap();
    let mut ref_kv =
        KvCache::new_gpu_q8(&mut gpus.devices[0], config.n_layers, config.n_kv_heads, config.head_dim, 256).unwrap();
    set_input(&gpus.devices[0], &ref_scratch, &input, 0);
    qwen35::run_fa_layer_body(
        &mut gpus.devices[0], &weights[0], &config, fa_layer, 0, 0, &mut ref_kv, &ref_scratch, FaPhase::Full,
    )
    .expect("ref Full");
    gpus.devices[0].hip.device_synchronize().unwrap();
    let reference = gpus.devices[0].download_f32(&ref_scratch.x).unwrap();

    // ── TP: TpAttn (partial wo → s.o) per rank, all-reduce, add, TpFfn. ─
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        set_input(&gpus.devices[r], &scratches[r], &input, 0);
        qwen35::run_fa_layer_body(
            &mut gpus.devices[r], &weights[r], &config, fa_layer, 0, 0, &mut kvs[r], &scratches[r],
            FaPhase::TpAttn { mask: Some(&masks[r]) },
        )
        .expect("TpAttn");
    }
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        gpus.devices[r].hip.device_synchronize().unwrap();
    }
    // all-reduce the partial wo contributions (s.o) across ranks.
    let refs: Vec<&_> = scratches.iter().map(|s| &s.o.buf).collect();
    gpus.all_reduce_sum_f32(&refs, dim).expect("all_reduce s.o");
    for r in 0..TP {
        gpus.devices[r].bind_thread().unwrap();
        gpus.devices[r]
            .hip
            .stream_synchronize(gpus.devices[r].active_stream.as_ref().unwrap())
            .unwrap();
        // s.x += s.o   (residual update with the full attention contribution)
        let (x, o) = (&scratches[r].x, &scratches[r].o);
        gpus.devices[r].add_f32(x, o, x).expect("add residual");
        // FFN on the synced residual.
        qwen35::run_fa_layer_body(
            &mut gpus.devices[r], &weights[r], &config, fa_layer, 0, 0, &mut kvs[r], &scratches[r],
            FaPhase::TpFfn,
        )
        .expect("TpFfn");
        gpus.devices[r].hip.device_synchronize().unwrap();
    }

    let tp_out = gpus.devices[0].download_f32(&scratches[0].x).unwrap();

    // ── Compare. ──────────────────────────────────────────────────────
    let ref_max = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..dim {
        let abs = (tp_out[i] - reference[i]).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / (ref_max + 1e-12));
    }
    // cleanup streams
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().unwrap();
        if let Some(st) = dev.active_stream.take() {
            let _ = dev.hip.stream_destroy(st);
        }
    }
    println!("ref max|y|={ref_max:.4e}  max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}  tol={REL_TOL:.0e}");
    assert!(
        max_rel < REL_TOL,
        "FA TP layer output diverged from single-GPU Full by {max_rel:.3e} > {REL_TOL:.0e}"
    );
    println!("tp_fa_layer_parity: PASS");
}
