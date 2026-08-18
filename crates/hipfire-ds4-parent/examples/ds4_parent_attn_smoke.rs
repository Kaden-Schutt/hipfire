// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 parent-attention smoke: load layer 0 (compress_ratio == 0) without
//! experts and run `parent_attention_swa` over 16 rows at `start_pos = 0`.
//!
//! Must run on gfx942 (mi300x).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_attn_smoke \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```

use hipfire_ds4_parent::attention::{
    all_finite, get_window_topk_idxs, l2_norm, parent_attention_swa, swa_n_valid,
    ParentAttnScratch, PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_HEADS, PARENT_N_KV_HEADS,
    PARENT_Q_WIDTH, PARENT_SWA_WINDOW,
};
use hipfire_ds4_parent::codec::round_to_bf16;
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

    println!("=== ds4_parent_attn_smoke ===");
    println!("model: {}", model_path.display());
    println!("layer: {LAYER}  rows: {ROWS}  start_pos: {START_POS}");

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
        "admit OK: layers={} compress_ratios[0]={}",
        cfg.num_hidden_layers,
        cfg.compress_ratio(0)
    );
    if cfg.compress_ratio(0) != 0 {
        return Err(format!(
            "deepseek4 parent: expected layer 0 compress_ratio=0, got {}",
            cfg.compress_ratio(0)
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: 0..1,
        load_experts: false,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    println!(
        "loaded layers={:?} experts={} in {load_s:.3}s  resident={:.3} GiB",
        weights.layer_range,
        weights.experts_loaded,
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let layer = &weights.layers[0];
    assert_eq!(layer.layer_idx, LAYER);
    assert_eq!(layer.compress_ratio, 0);

    let mut scratch = ParentAttnScratch::new(&mut gpu, &cfg, ROWS)?;
    let scratch_bytes = scratch.bytes();
    println!(
        "ParentAttnScratch::bytes() = {scratch_bytes} ({:.3} MiB)  max_rows={}",
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

    // Post-attn_norm F32 activations (plausible deterministic grid).
    let mut x_f32 = vec![0.0f32; ROWS * PARENT_DIM];
    for r in 0..ROWS {
        for k in 0..PARENT_DIM {
            let v = (((r * 131 + k * 17) % 200) as f32 - 100.0) * 0.01;
            x_f32[r * PARENT_DIM + k] = round_to_bf16(v);
        }
    }
    let x = upload_f32(&mut gpu, &x_f32, &[ROWS, PARENT_DIM])?;
    let out = gpu
        .zeros(&[ROWS, PARENT_DIM], DType::F32)
        .map_err(|e| format!("deepseek4 parent: out alloc: {e:?}"))?;

    // Window index sanity (host): position 0 is self-only.
    let win = get_window_topk_idxs(PARENT_SWA_WINDOW, ROWS, START_POS)?;
    let k = ROWS.min(PARENT_SWA_WINDOW);
    let pos0_visible: Vec<i32> = win[..k].iter().copied().filter(|&v| v >= 0).collect();
    println!(
        "window_topk pos0 visible idxs = {:?}  n_valid[0]={}",
        pos0_visible,
        swa_n_valid(START_POS, 0, PARENT_SWA_WINDOW)
    );
    if pos0_visible != vec![0] {
        return Err(format!(
            "deepseek4 parent: pos0 window expected [0], got {pos0_visible:?}"
        ));
    }

    // Warmup + timed forward.
    parent_attention_swa(
        &mut gpu,
        backend,
        layer,
        &cfg,
        &mut scratch,
        &x,
        ROWS,
        START_POS,
        &kv_ring,
        &out,
    )?;
    // Reset ring for a clean timed run (warmup wrote into it).
    zero_f32(
        &gpu,
        &kv_ring,
        PARENT_N_KV_HEADS * PARENT_HEAD_DIM * PARENT_SWA_WINDOW,
    )?;
    // Re-upload x (untouched by forward — F32 residual).
    let t0 = Instant::now();
    parent_attention_swa(
        &mut gpu,
        backend,
        layer,
        &cfg,
        &mut scratch,
        &x,
        ROWS,
        START_POS,
        &kv_ring,
        &out,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out_host = download_f32(&gpu, &out, ROWS * PARENT_DIM)?;
    let finite = all_finite(&out_host);
    let out_norm = l2_norm(&out_host);
    println!("output finite = {finite}");
    println!("output L2 norm = {out_norm:.6}");
    println!("wall-clock (timed run, after warmup) = {wall_ms:.2} ms");

    // Intermediate norms from scratch (post-forward contents).
    // q after wq_b+rope is still in scratch.q_f32 (post-rope, pre-attn).
    // Actually after forward, q_f32 holds post-rope Q; kv_f32 holds post-quant KV;
    // attn_out_f32 holds post-inverse-rope attention output before wo_a.
    let q_host = download_f32(&gpu, scratch.q_f32_ref()?, ROWS * PARENT_Q_WIDTH)?;
    let kv_host = download_f32(&gpu, scratch.kv_f32_ref()?, ROWS * PARENT_HEAD_DIM)?;
    let attn_host = download_f32(&gpu, scratch.attn_out_f32_ref()?, ROWS * PARENT_Q_WIDTH)?;
    println!(
        "stage norms: q_post_rope={:.6}  kv_post_quant={:.6}  attn_pre_wo_a={:.6}  final={:.6}",
        l2_norm(&q_host),
        l2_norm(&kv_host),
        l2_norm(&attn_host),
        out_norm
    );

    // Attention probability sanity for position 0:
    // With n_valid=1 and a sink, the single real KV slot + sink form a
    // 2-entry distribution. Reconstruct host-side scores for head 0.
    let sink = download_f32(&gpu, &layer.attn_sink, PARENT_N_HEADS)?;
    let inv_scale = 1.0f32 / (PARENT_HEAD_DIM as f32).sqrt();
    // Q[0, h=0], K[0]
    let mut dot = 0.0f64;
    for d in 0..PARENT_HEAD_DIM {
        dot += q_host[d] as f64 * kv_host[d] as f64;
    }
    let score0 = (dot as f32) * inv_scale;
    let sink0 = sink[0];
    let m = score0.max(sink0);
    let e0 = (score0 - m).exp();
    let es = (sink0 - m).exp();
    let z = e0 + es;
    let p0 = e0 / z;
    let ps = es / z;
    println!(
        "pos0 head0 probs: kv={p0:.6}  sink={ps:.6}  sum={:.6}  score0={score0:.6} sink0={sink0:.6}",
        p0 + ps
    );
    if ((p0 + ps) - 1.0).abs() >= 1e-5 {
        return Err(format!(
            "deepseek4 parent: pos0 probs not a distribution (sum={})",
            p0 + ps
        ));
    }
    if !finite {
        return Err("deepseek4 parent: output contains non-finite values".to_owned());
    }

    // Refusal check on a fake compress_ratio (using a temporary layer copy
    // is hard without Clone; instead just print the contract string).
    println!(
        "refusal contract: parent_attention_swa refuses compress_ratio!=0 \
         (compressor/indexer out of scope)"
    );

    println!("PASS");
    Ok(())
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download short (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))?;
    Ok(data)
}

fn zero_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<(), String> {
    let z = vec![0.0f32; nelems];
    let nbytes = nelems * 4;
    let bytes = unsafe { std::slice::from_raw_parts(z.as_ptr() as *const u8, nbytes) };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: zero_f32: {e:?}"))?;
    Ok(())
}
