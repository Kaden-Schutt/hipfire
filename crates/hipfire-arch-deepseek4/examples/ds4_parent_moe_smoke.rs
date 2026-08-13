// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 parent-MoE smoke: load one real layer with experts and run
//! `parent_moe_forward` over 16 rows of plausible activations.
//!
//! Must run on gfx942 (mi300x).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_moe_smoke \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```

use hipfire_arch_deepseek4::parent::codec::round_to_bf16;
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::gate_ref;
use hipfire_arch_deepseek4::parent::moe::{
    parent_moe_forward_counted, parent_route, ParentMoeScratch, PARENT_DIM, PARENT_ROUTE_SCALE,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const ROWS: usize = 16;
/// Score-routed layer (num_hash_layers = 3 → layer 3 is the first score layer).
const LAYER: usize = 3;

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
    let model = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    // Also accept --model PATH
    let mut model = model;
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--model") {
        if let Some(p) = args.get(i + 1) {
            model = p.clone();
        }
    }
    let model_path = Path::new(&model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }

    println!("=== ds4_parent_moe_smoke ===");
    println!("model: {}", model_path.display());
    println!("layer: {LAYER}  rows: {ROWS}");

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
        "admit OK: layers={} n_routed={} topk={} hash_layers={}",
        cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.num_hash_layers
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: LAYER..(LAYER + 1),
        load_experts: true,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    println!(
        "loaded layer {LAYER} with {} experts in {load_s:.3}s  resident={:.3} GiB",
        weights.layers[0].experts.len(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let layer = &weights.layers[0];
    assert_eq!(layer.layer_idx, LAYER);

    // Scratch
    let mut scratch = ParentMoeScratch::new(&mut gpu, &cfg, ROWS)?;
    let scratch_bytes = scratch.bytes();
    println!(
        "ParentMoeScratch::bytes() = {scratch_bytes} ({:.3} MiB)  max_rows={}",
        scratch_bytes as f64 / (1024.0 * 1024.0),
        scratch.max_rows()
    );
    // Must be small constant, not proportional to 256 experts.
    if scratch_bytes > 64 * 1024 * 1024 {
        return Err(format!(
            "deepseek4 parent: scratch unexpectedly large ({scratch_bytes} bytes)"
        ));
    }

    // Plausible BF16 activations: deterministic grid of small magnitudes.
    let mut x_f32 = vec![0.0f32; ROWS * PARENT_DIM];
    for r in 0..ROWS {
        for k in 0..PARENT_DIM {
            // Mix of magnitudes, BF16-representable after round.
            let v = (((r * 131 + k * 17) % 200) as f32 - 100.0) * 0.01;
            x_f32[r * PARENT_DIM + k] = round_to_bf16(v);
        }
    }
    let x_bytes = pack_f32_to_bf16_bytes(&x_f32);
    let x = {
        let t = gpu
            .alloc_tensor(&[ROWS, PARENT_DIM], DType::BF16)
            .map_err(|e| format!("deepseek4 parent: alloc x: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&t.buf, &x_bytes)
            .map_err(|e| format!("deepseek4 parent: upload x: {e:?}"))?;
        t
    };
    let out = gpu
        .zeros(&[ROWS, PARENT_DIM], DType::F32)
        .map_err(|e| format!("deepseek4 parent: alloc out: {e:?}"))?;

    // Route (host f32 path — see parent_route docs).
    let route_t0 = Instant::now();
    let routing = parent_route(&mut gpu, backend, layer, &cfg, &x, ROWS, None)?;
    let route_ms = route_t0.elapsed().as_secs_f64() * 1000.0;
    let distinct = routing.distinct_experts();
    println!(
        "routing: topk={} distinct_experts={distinct}  wall={route_ms:.2} ms",
        routing.topk
    );
    println!(
        "  first-row indices={:?} weights={:?}",
        &routing.indices[..routing.topk],
        &routing.weights[..routing.topk]
    );

    // Cross-check routing against layer_ref::gate_ref (f64 oracle).
    {
        let w_bytes = {
            let n = cfg.n_routed_experts * PARENT_DIM * 2;
            let mut b = vec![0u8; n];
            gpu.hip
                .memcpy_dtoh(&mut b, &layer.gate_weight.buf)
                .map_err(|e| format!("deepseek4 parent: gate_w dtoh: {e:?}"))?;
            b
        };
        let mut w_f32 = vec![0.0f32; cfg.n_routed_experts * PARENT_DIM];
        for i in 0..w_f32.len() {
            let bits = u16::from_le_bytes([w_bytes[i * 2], w_bytes[i * 2 + 1]]);
            w_f32[i] = f32::from_bits((bits as u32) << 16);
        }
        let bias_f32 = if let Some(b) = layer.gate_bias.as_ref() {
            let mut data = vec![0.0f32; cfg.n_routed_experts];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
            };
            gpu.hip
                .memcpy_dtoh(bytes, &b.buf)
                .map_err(|e| format!("deepseek4 parent: gate_bias dtoh: {e:?}"))?;
            Some(data)
        } else {
            None
        };
        let oracle = gate_ref(
            &x_f32,
            &w_f32,
            bias_f32.as_deref(),
            ROWS,
            PARENT_DIM,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            PARENT_ROUTE_SCALE as f64,
            true,
        )?;
        let mut idx_mismatch = 0usize;
        let mut max_w_abs = 0.0f32;
        for i in 0..routing.indices.len() {
            if routing.indices[i] != oracle.indices[i] {
                idx_mismatch += 1;
            }
            max_w_abs = max_w_abs.max((routing.weights[i] - oracle.weights[i]).abs());
        }
        println!(
            "gate_ref cross-check: idx_mismatch={idx_mismatch}/{}  max|w_gpu-w_ref|={max_w_abs:.3e}  route_scale={}",
            routing.indices.len(),
            PARENT_ROUTE_SCALE
        );
        if idx_mismatch != 0 {
            return Err(format!(
                "deepseek4 parent: routing indices disagree with gate_ref ({idx_mismatch} slots)"
            ));
        }
        if max_w_abs > 1e-5 {
            return Err(format!(
                "deepseek4 parent: routing weights disagree with gate_ref (max abs {max_w_abs})"
            ));
        }
    }


    // MoE forward
    let fwd_t0 = Instant::now();
    let decode_calls = parent_moe_forward_counted(
        &mut gpu,
        backend,
        layer,
        &cfg,
        &mut scratch,
        &x,
        ROWS,
        &routing,
        &out,
    )?;
    // Sync before timing end.
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;
    let fwd_s = fwd_t0.elapsed().as_secs_f64();

    // Download output
    let y = {
        let mut data = vec![0.0f32; ROWS * PARENT_DIM];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &out.buf)
            .map_err(|e| format!("deepseek4 parent: download out: {e:?}"))?;
        data
    };

    let mut finite = true;
    let mut sum_sq = 0.0f64;
    let mut abs_max = 0.0f32;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    for &v in &y {
        if v.is_nan() {
            finite = false;
            n_nan += 1;
        } else if v.is_infinite() {
            finite = false;
            n_inf += 1;
        } else {
            sum_sq += (v as f64) * (v as f64);
            abs_max = abs_max.max(v.abs());
        }
    }
    let norm = sum_sq.sqrt();
    let expected_decodes = distinct * 3; // w1, w3, w2 per selected expert

    println!();
    println!("=== results ===");
    println!("finite: {finite}  nan={n_nan} inf={n_inf}");
    println!("||out||_2 = {norm:.6}  max|out| = {abs_max:.6}");
    println!("distinct experts selected: {distinct}");
    println!("decode calls issued: {decode_calls}  (expected ~ {expected_decodes} = distinct*3)");
    println!(
        "grouping proof: decode_calls/3 = {} vs rows*topk = {}",
        decode_calls / 3,
        ROWS * routing.topk
    );
    println!("forward wall-clock: {fwd_s:.3} s");
    println!("ParentMoeScratch::bytes() = {scratch_bytes}");
    println!("routing path: host f32 (gate GEMM + sqrtsoftplus + topk on CPU)");
    println!(
        "  reason: reference Gate uses linear(x.float(), weight.float()) with NO act-quant;\
         parent_linear_* always act-quants, so GPU linear is wrong for the gate.\
         rows×256 is tiny; host matches reference exactly."
    );

    // Sample a few output values
    print!("out[0,:8] = [");
    for j in 0..8.min(PARENT_DIM) {
        if j > 0 {
            print!(", ");
        }
        print!("{:.5}", y[j]);
    }
    println!("]");

    let _ = gpu.free_tensor(x);
    let _ = gpu.free_tensor(out);

    if !finite {
        return Err("deepseek4 parent: MoE output is not finite".to_owned());
    }
    if decode_calls != expected_decodes {
        return Err(format!(
            "deepseek4 parent: decode_calls={decode_calls} != distinct*3={expected_decodes} \
             (grouping broken?)"
        ));
    }
    if decode_calls >= ROWS * routing.topk * 3 {
        return Err(format!(
            "deepseek4 parent: decode_calls={decode_calls} looks like per-token decode \
             (rows*topk*3={}) — grouping failed",
            ROWS * routing.topk * 3
        ));
    }

    println!();
    println!("PASS: parent MoE smoke");
    Ok(())
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = v.to_bits();
        // v is already BF16-representable via round_to_bf16; just take top 16.
        let b = (bits >> 16) as u16;
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}
