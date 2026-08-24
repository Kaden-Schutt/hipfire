// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 parent-layer canary: load layer 0 with experts and run
//! `parent_layer_forward_traced` over 16 rows at `start_pos = 0`.
//!
//! Must run on gfx942 (mi300x).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_layer_smoke \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```

use hipfire_ds4_parent::attention::{
    all_finite, l2_norm, PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_SWA_WINDOW,
};
use hipfire_ds4_parent::codec::round_to_bf16;
use hipfire_ds4_parent::forward::{
    parent_layer_forward_traced, ParentForwardScratch, ParentLayerTrace, PARENT_HC_DIM,
    PARENT_HC_MULT,
};
use hipfire_ds4_parent::inventory::ParentInventory;
use hipfire_ds4_parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_ds4_parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const ROWS: usize = 16;
const LAYER: usize = 0;
const START_POS: usize = 0;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut model = DEFAULT_MODEL.to_owned();
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--model") {
        if let Some(p) = args.get(i + 1) {
            model = p.clone();
        }
    } else if let Some(p) = args.iter().skip(1).find(|a| !a.starts_with('-')) {
        model = p.clone();
    }
    let model_path = Path::new(&model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }

    println!("=== ds4_parent_layer_smoke (Gate 4 canary) ===");
    println!("model: {}", model_path.display());
    println!("layer: {LAYER}  rows: {ROWS}  start_pos: {START_POS}");
    println!("hc_mult: {PARENT_HC_MULT}  dim: {PARENT_DIM}  hc_dim: {PARENT_HC_DIM}");

    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;

    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err("deepseek4 parent: gfx942 required".to_owned());
    }
    println!("gpu: gfx942");

    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK: layers={} hash_layers={} compress_ratios[0]={} n_routed={} topk={}",
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.compress_ratio(0),
        cfg.n_routed_experts,
        cfg.num_experts_per_tok
    );
    if cfg.compress_ratio(0) != 0 {
        return Err(format!(
            "deepseek4 parent: expected layer 0 compress_ratio=0, got {}",
            cfg.compress_ratio(0)
        ));
    }
    // Layer 0 is hash-routed — input_ids required.
    if LAYER >= cfg.num_hash_layers {
        return Err(format!(
            "deepseek4 parent: expected layer 0 hash-routed (num_hash_layers={})",
            cfg.num_hash_layers
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: 0..1,
        load_experts: true,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    println!(
        "loaded layers={:?} experts={} (n={}) in {load_s:.3}s  resident={:.3} GiB",
        weights.layer_range,
        weights.experts_loaded,
        weights.layers[0].experts.len(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let layer = &weights.layers[0];
    assert_eq!(layer.layer_idx, LAYER);
    assert_eq!(layer.compress_ratio, 0);
    assert!(layer.tid2eid.is_some(), "layer 0 must carry tid2eid");

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, ROWS)?;
    let scratch_bytes = scratch.bytes();
    println!(
        "ParentForwardScratch::bytes() = {scratch_bytes} ({:.3} MiB)  max_rows={}",
        scratch_bytes as f64 / (1024.0 * 1024.0),
        scratch.max_rows()
    );

    // Persistent SWA ring (zero-init = no pre-chunk history).
    let kv_ring = gpu
        .zeros(
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
            DType::F32,
        )
        .map_err(|e| format!("deepseek4 parent: kv_ring alloc: {e:?}"))?;

    // Multi-stream HC residual input [rows, hc_mult, dim] F32.
    // Stream 0 carries a deterministic BF16-ish grid; other streams start at 0
    // (matches the embed → residual-streams init pattern of the reference).
    let mut x_host = vec![0.0f32; ROWS * PARENT_HC_DIM];
    for r in 0..ROWS {
        for d in 0..PARENT_DIM {
            let v = (((r * 131 + d * 17) % 200) as f32 - 100.0) * 0.01;
            x_host[r * PARENT_HC_DIM + d] = round_to_bf16(v);
        }
    }
    let x = upload_f32(&mut gpu, &x_host, &[ROWS, PARENT_HC_MULT, PARENT_DIM])?;
    let out = gpu
        .zeros(&[ROWS, PARENT_HC_MULT, PARENT_DIM], DType::F32)
        .map_err(|e| format!("deepseek4 parent: out alloc: {e:?}"))?;

    // Hash-routing input_ids (layer 0). Deterministic small ids.
    let input_ids: Vec<u32> = (0..ROWS as u32).map(|i| 1000 + i * 17).collect();
    println!("input_ids (hash route) = {:?}", input_ids);

    // Warmup (also proves the path runs before we time).
    let mut trace = ParentLayerTrace::default();
    parent_layer_forward_traced(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &mut scratch,
        LAYER,
        &x,
        ROWS,
        START_POS,
        Some(&input_ids),
        &kv_ring,
        &out,
        &mut trace,
    )?;
    // Reset ring + out for a clean timed run.
    zero_f32(
        &gpu,
        &kv_ring,
        PARENT_N_KV_HEADS * PARENT_HEAD_DIM * PARENT_SWA_WINDOW,
    )?;
    zero_f32(&gpu, &out, ROWS * PARENT_HC_DIM)?;

    let mut trace = ParentLayerTrace::default();
    let t0 = Instant::now();
    parent_layer_forward_traced(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &mut scratch,
        LAYER,
        &x,
        ROWS,
        START_POS,
        Some(&input_ids),
        &kv_ring,
        &out,
        &mut trace,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out_host = download_f32(&gpu, &out, ROWS * PARENT_HC_DIM)?;
    let finite = all_finite(&out_host);
    let out_norm = l2_norm(&out_host);

    println!();
    println!("── stage L2 norms (ParentLayerTrace) ──");
    println!("  hc_pre_attn  = {:.6}", trace.hc_pre_attn);
    println!("  attn_norm    = {:.6}", trace.attn_norm);
    println!("  attn_out     = {:.6}", trace.attn_out);
    println!("  hc_post_attn = {:.6}", trace.hc_post_attn);
    println!("  hc_pre_ffn   = {:.6}", trace.hc_pre_ffn);
    println!("  ffn_norm     = {:.6}", trace.ffn_norm);
    println!("  moe_out      = {:.6}", trace.moe_out);
    println!("  hc_post_ffn  = {:.6}", trace.hc_post_ffn);
    println!();
    println!("output finite        = {finite}");
    println!("output L2 norm       = {out_norm:.6}");
    println!("wall-clock (timed)   = {wall_ms:.2} ms");
    println!(
        "ParentForwardScratch = {scratch_bytes} bytes ({:.3} MiB)",
        scratch_bytes as f64 / (1024.0 * 1024.0)
    );

    // Confirm FFN half rewrote post/comb: after full forward, post buffer
    // must equal a standalone FFN-only hc_pre on the post-attn residual.
    // We check a cheaper proxy: hc_pre_ffn L2 is nonzero and distinct from
    // hc_pre_attn (different HC weights + different residual). Degenerate
    // equality would mean the FFN half was skipped or reused attn state.
    if (trace.hc_pre_attn - trace.hc_pre_ffn).abs() < 1e-6
        && trace.hc_pre_attn > 0.0
        && trace.hc_pre_ffn > 0.0
    {
        println!(
            "NOTE: hc_pre_attn L2 ≈ hc_pre_ffn L2 ({:.6}); not proof of reuse \
             (different weights can still yield similar norms) — see source: \
             FFN hc_pre overwrites the single post/comb pair after attn hc_post \
             consumed them.",
            trace.hc_pre_attn
        );
    } else {
        println!(
            "FFN half post/comb: hc_pre_attn L2={:.6} vs hc_pre_ffn L2={:.6} \
             (distinct; FFN half ran its own hc_pre into the shared post/comb buffers)",
            trace.hc_pre_attn, trace.hc_pre_ffn
        );
    }

    // Allocation contract: this binary only allocs scratch once + x/out/kv_ring.
    // The layer forward itself must not alloc_tensor — verified by code audit
    // (parent_layer_forward_inner has no alloc_tensor call).
    println!(
        "alloc inside layer forward = none (code audit: parent_layer_forward_inner \
         has zero Gpu::alloc_tensor calls; only pre-sized scratch + host staging)"
    );

    // Degeneracy / non-finite checks — report, don't paper over.
    let stages = [
        ("hc_pre_attn", trace.hc_pre_attn),
        ("attn_norm", trace.attn_norm),
        ("attn_out", trace.attn_out),
        ("hc_post_attn", trace.hc_post_attn),
        ("hc_pre_ffn", trace.hc_pre_ffn),
        ("ffn_norm", trace.ffn_norm),
        ("moe_out", trace.moe_out),
        ("hc_post_ffn", trace.hc_post_ffn),
    ];
    let mut bad = false;
    if !finite {
        println!("FINDING: output contains non-finite values");
        bad = true;
    }
    for (name, v) in stages {
        if !v.is_finite() {
            println!("FINDING: stage {name} L2 is non-finite ({v})");
            bad = true;
        } else if v == 0.0 {
            println!("FINDING: stage {name} L2 is exactly zero (degenerate)");
            bad = true;
        }
    }
    // Order-of-magnitude check against neighbours.
    for w in stages.windows(2) {
        let (n0, v0) = w[0];
        let (n1, v1) = w[1];
        if v0.is_finite() && v1.is_finite() && v0 > 0.0 && v1 > 0.0 {
            let ratio = v0.max(v1) / v0.min(v1);
            if ratio > 1e6 {
                println!(
                    "FINDING: stage norms {n0}={v0:.6} vs {n1}={v1:.6} differ by {ratio:.3e}x"
                );
                bad = true;
            }
        }
    }

    if bad {
        return Err("deepseek4 parent: layer canary found non-finite or degenerate stage".into());
    }
    println!("PASS: layer-0 16-token canary finite with non-degenerate stage norms");
    Ok(())
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: alloc: {e:?}"))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: htod: {e:?}"))?;
    Ok(t)
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let mut host = vec![0.0f32; nelems];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nelems * 4)
    };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: dtoh: {e:?}"))?;
    Ok(host)
}

fn zero_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<(), String> {
    let zeros = vec![0.0f32; nelems];
    let bytes = unsafe {
        std::slice::from_raw_parts(zeros.as_ptr() as *const u8, nelems * 4)
    };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: zero htod: {e:?}"))?;
    Ok(())
}
