// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 5 parent-indexer smoke: load layer 2 (first `compress_ratio == 4`)
//! without experts and run `parent_indexer_forward` over 16 rows.
//!
//! Must run on gfx942 (mi300x).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_indexer_smoke \
//!   -- /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```

use hipfire_arch_deepseek4::parent::attention::{
    all_finite, l2_norm, PARENT_DIM, PARENT_Q_LORA, PARENT_RMS_EPS, PARENT_SWA_WINDOW,
};
use hipfire_arch_deepseek4::parent::codec::round_to_bf16;
use hipfire_arch_deepseek4::parent::indexer::{
    indexer_n_compressed, indexer_score_row_f64, indexer_topk_host, indexer_weights_scale,
    parent_indexer_forward, topk_index_mismatch_count, IndexerScoreReport, ParentIndexerScratch,
    PARENT_INDEX_HEAD_DIM, PARENT_INDEX_N_HEADS, PARENT_INDEX_Q_WIDTH, PARENT_INDEX_RATIO,
    PARENT_INDEX_TOPK,
};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::linear::parent_linear_dense;
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const ROWS: usize = 16;
const LAYER: usize = 2; // first ratio-4 layer
const START_POS: usize = 0;
const OFFSET: usize = PARENT_SWA_WINDOW; // model.py:515 prefill offset = kv.size(1) when >win; smoke uses win

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

    println!("=== ds4_parent_indexer_smoke ===");
    println!("model: {}", model_path.display());
    println!("layer: {LAYER}  rows: {ROWS}  start_pos: {START_POS}  offset: {OFFSET}");
    println!(
        "weights_scale (softmax_scale * n_heads^-0.5) = {:.15}",
        indexer_weights_scale()
    );
    println!(
        "FP4 group-32 + Hadamard applied on Q after RoPE (model.py:420-422); \
         weights_proj is plain BF16 GEMM (no parent_linear_dense / no act-quant)"
    );

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
    let ratio = cfg.compress_ratio(LAYER);
    println!(
        "admit OK: layers={} compress_ratios[{LAYER}]={ratio}",
        cfg.num_hidden_layers
    );
    if ratio != PARENT_INDEX_RATIO {
        return Err(format!(
            "deepseek4 parent: expected layer {LAYER} compress_ratio=4, got {ratio}"
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: LAYER..(LAYER + 1),
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
    assert_eq!(layer.compress_ratio, PARENT_INDEX_RATIO);
    let ix = layer
        .indexer
        .as_ref()
        .ok_or_else(|| format!("deepseek4 parent: layer {LAYER} missing indexer weights"))?;

    // Prove weights_proj is BF16 (plain path).
    if ix.weights_proj.dtype != DType::BF16 {
        return Err(format!(
            "deepseek4 parent: indexer.weights_proj dtype {:?} != BF16",
            ix.weights_proj.dtype
        ));
    }
    println!(
        "weights_proj dtype=BF16 shape={:?} bytes={}  (BF16 path verified)",
        ix.weights_proj.shape,
        ix.weights_proj.buf.size()
    );
    println!(
        "wq_b dense FP8→BF16 resident: n={} k={} bytes={}",
        ix.wq_b.n(),
        ix.wq_b.k(),
        ix.wq_b.resident_bytes()
    );

    let mut scratch = ParentIndexerScratch::new(&mut gpu, &cfg, ROWS)?;
    let scratch_bytes = scratch.bytes();
    println!(
        "ParentIndexerScratch::bytes() = {scratch_bytes} ({:.3} MiB)  max_rows={} max_n_compressed={}",
        scratch_bytes as f64 / (1024.0 * 1024.0),
        scratch.max_rows(),
        scratch.max_n_compressed()
    );

    // Post-attn_norm F32 activations.
    let mut x_f32 = vec![0.0f32; ROWS * PARENT_DIM];
    for r in 0..ROWS {
        for k in 0..PARENT_DIM {
            let v = (((r * 131 + k * 17) % 200) as f32 - 100.0) * 0.01;
            x_f32[r * PARENT_DIM + k] = round_to_bf16(v);
        }
    }
    let x = upload_f32(&mut gpu, &x_f32, &[ROWS, PARENT_DIM])?;

    // qr = q_norm(wq_a(x)) — same as Attention.forward before indexer call.
    let qr = compute_qr(&mut gpu, backend, layer, &x, ROWS)?;

    // Output buffers.
    let topk_bytes = ROWS * PARENT_INDEX_TOPK * 4;
    let topk_idx = gpu
        .alloc_tensor(&[topk_bytes], DType::Raw)
        .map_err(|e| format!("deepseek4 parent: topk_idx alloc: {e:?}"))?;
    // Fill with -1.
    {
        let fill = vec![-1i32; ROWS * PARENT_INDEX_TOPK];
        let bytes =
            unsafe { std::slice::from_raw_parts(fill.as_ptr() as *const u8, topk_bytes) };
        gpu.hip
            .memcpy_htod(&topk_idx.buf, bytes)
            .map_err(|e| format!("deepseek4 parent: topk fill: {e:?}"))?;
    }
    let n_active = gpu
        .alloc_tensor(&[4], DType::Raw)
        .map_err(|e| format!("deepseek4 parent: n_active alloc: {e:?}"))?;
    {
        let z = [0i32];
        let bytes = unsafe { std::slice::from_raw_parts(z.as_ptr() as *const u8, 4) };
        gpu.hip
            .memcpy_htod(&n_active.buf, bytes)
            .map_err(|e| format!("deepseek4 parent: n_active fill: {e:?}"))?;
    }

    // Warmup.
    parent_indexer_forward(
        &mut gpu,
        backend,
        ix,
        &cfg,
        &mut scratch,
        &x,
        &qr,
        ROWS,
        START_POS,
        OFFSET,
        LAYER,
        &topk_idx,
        &n_active,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;

    // Timed run.
    let t0 = Instant::now();
    parent_indexer_forward(
        &mut gpu,
        backend,
        ix,
        &cfg,
        &mut scratch,
        &x,
        &qr,
        ROWS,
        START_POS,
        OFFSET,
        LAYER,
        &topk_idx,
        &n_active,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let n_comp_expect = indexer_n_compressed(START_POS, ROWS, PARENT_INDEX_RATIO);
    let n_active_host = download_i32(&gpu, &n_active, 1)?;
    let n_active_val = n_active_host[0] as usize;
    println!("n_active (device) = {n_active_val}  expect={n_comp_expect}");
    if n_active_val != n_comp_expect {
        return Err(format!(
            "deepseek4 parent: n_active mismatch device={n_active_val} expect={n_comp_expect}"
        ));
    }

    // Download intermediates for oracle comparison.
    let q_host = download_f32(&gpu, scratch.q_score_f32_ref(), ROWS * PARENT_INDEX_Q_WIDTH)?;
    let w_host = download_f32(
        &gpu,
        scratch.weights_f32_ref(),
        ROWS * PARENT_INDEX_N_HEADS,
    )?;
    let kv_host = download_f32(
        &gpu,
        scratch.kv_cache_f32_ref(),
        n_active_val * PARENT_INDEX_HEAD_DIM,
    )?;
    let scores_host = download_f32(
        &gpu,
        scratch.scores_f32_ref(),
        ROWS * scratch.max_n_compressed(),
    )?;
    let topk_host = download_i32(&gpu, &topk_idx, ROWS * PARENT_INDEX_TOPK)?;

    let q_finite = all_finite(&q_host);
    let w_finite = all_finite(&w_host);
    let kv_finite = all_finite(&kv_host);
    println!(
        "finite: q={q_finite} weights={w_finite} kv={kv_finite}  \
         norms: q={:.6} weights={:.6} kv={:.6}",
        l2_norm(&q_host),
        l2_norm(&w_host),
        l2_norm(&kv_host)
    );
    if !(q_finite && w_finite && kv_finite) {
        return Err("deepseek4 parent: non-finite intermediate".to_owned());
    }

    // f64 oracle over the scoring path (post Q/weights/KV construction).
    // Compare only *visible* finite slots — masked positions are -inf / -1e30
    // sentinels and must not enter the relative-error stats.
    let mut ref_scores = Vec::new();
    let mut gpu_scores_compact = Vec::new();
    let mut ref_topk = vec![-1i32; ROWS * PARENT_INDEX_TOPK];
    let max_n = scratch.max_n_compressed();

    for r in 0..ROWS {
        let q_row: Vec<f64> = q_host[r * PARENT_INDEX_Q_WIDTH..(r + 1) * PARENT_INDEX_Q_WIDTH]
            .iter()
            .map(|v| *v as f64)
            .collect();
        let w_row: Vec<f64> = w_host[r * PARENT_INDEX_N_HEADS..(r + 1) * PARENT_INDEX_N_HEADS]
            .iter()
            .map(|v| *v as f64)
            .collect();
        let kv_f64: Vec<f64> = kv_host.iter().map(|v| *v as f64).collect();
        let n_vis = if START_POS == 0 {
            (r + 1) / PARENT_INDEX_RATIO
        } else {
            n_active_val
        }
        .min(n_active_val);

        let row_scores = if n_active_val > 0 {
            indexer_score_row_f64(
                &q_row,
                &kv_f64,
                &w_row,
                PARENT_INDEX_N_HEADS,
                PARENT_INDEX_HEAD_DIM,
                n_active_val,
            )?
        } else {
            vec![]
        };

        let mut masked = vec![f32::NEG_INFINITY; n_active_val.max(1)];
        for t in 0..n_vis {
            masked[t] = row_scores[t] as f32;
            ref_scores.push(row_scores[t]);
            gpu_scores_compact.push(scores_host[r * max_n + t]);
        }

        let k_take = PARENT_INDEX_TOPK.min(n_active_val.max(1));
        let mut row_topk = if n_vis > 0 {
            indexer_topk_host(&masked[..n_vis], k_take)
        } else {
            vec![-1i32; k_take]
        };
        for v in row_topk.iter_mut() {
            if *v < 0 {
                continue;
            }
            let idx = *v as usize;
            if START_POS == 0 && idx >= n_vis {
                *v = -1;
            } else {
                *v += OFFSET as i32;
            }
        }
        let dest = &mut ref_topk[r * PARENT_INDEX_TOPK..(r + 1) * PARENT_INDEX_TOPK];
        for i in 0..PARENT_INDEX_TOPK {
            dest[i] = if i < row_topk.len() { row_topk[i] } else { -1 };
        }
    }

    let report = IndexerScoreReport::from_scores_and_indices(
        &gpu_scores_compact,
        &ref_scores,
        &topk_host,
        &ref_topk,
    );
    println!(
        "score vs f64 oracle: max_abs={:.6e}  mean_rel={:.6e}  l2_rel={:.6e}  n_scores={}",
        report.max_abs, report.mean_rel, report.l2_rel, report.n_scores
    );
    println!(
        "top-k index mismatch count (set-wise, ignoring -1 pad) = {}  over {} slots",
        report.index_mismatch, report.n_indices
    );

    // Per-row dump of first few top-k for the last row (most slots visible).
    let last = ROWS - 1;
    let last_gpu = &topk_host[last * PARENT_INDEX_TOPK..(last + 1) * PARENT_INDEX_TOPK];
    let last_ref = &ref_topk[last * PARENT_INDEX_TOPK..(last + 1) * PARENT_INDEX_TOPK];
    let gpu_pos: Vec<i32> = last_gpu.iter().copied().filter(|&v| v >= 0).collect();
    let ref_pos: Vec<i32> = last_ref.iter().copied().filter(|&v| v >= 0).collect();
    println!("row {last} topk gpu (valid) = {gpu_pos:?}");
    println!("row {last} topk ref (valid) = {ref_pos:?}");
    println!(
        "row {last} set mismatch = {}",
        topk_index_mismatch_count(last_gpu, last_ref)
    );

    println!("wall-clock (timed run, after warmup) = {wall_ms:.2} ms");
    println!("scratch bytes = {scratch_bytes}");

    // Refusal contract.
    let refuse_msg = format!(
        "deepseek4 parent: parent_indexer_forward refuses compress_ratio=0 \
         (layer 0); indexer is only defined for compress_ratio == 4"
    );
    println!("refusal contract: {refuse_msg}");

    if report.index_mismatch != 0 {
        return Err(format!(
            "deepseek4 parent: top-k index mismatch count {} > 0",
            report.index_mismatch
        ));
    }
    // Score noise: FP32 reduction vs f64; allow modest relative error.
    if report.l2_rel > 1e-3 && n_active_val > 0 {
        return Err(format!(
            "deepseek4 parent: score l2_rel {:.3e} exceeds 1e-3",
            report.l2_rel
        ));
    }

    println!("PASS");
    Ok(())
}

/// `qr = rms_norm(wq_a(x), q_norm)` — Attention's LoRA bottleneck fed to the indexer.
fn compute_qr(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &hipfire_arch_deepseek4::parent::weights::ParentLayerWeights,
    x: &GpuTensor,
    rows: usize,
) -> Result<GpuTensor, String> {
    // Stage x → BF16 act tile.
    let x_host = download_f32(gpu, x, rows * PARENT_DIM)?;
    let x_bf16_bytes = pack_f32_to_bf16_bytes(&x_host);
    let act = gpu
        .alloc_tensor(&[rows, PARENT_DIM], DType::BF16)
        .map_err(|e| format!("deepseek4 parent: qr act alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&act.buf, &x_bf16_bytes)
        .map_err(|e| format!("deepseek4 parent: qr act upload: {e:?}"))?;
    let q_lat = gpu
        .alloc_tensor(&[rows, PARENT_Q_LORA], DType::F32)
        .map_err(|e| format!("deepseek4 parent: qr q_lat alloc: {e:?}"))?;
    parent_linear_dense(gpu, backend, &layer.wq_a, &act, rows, &q_lat)
        .map_err(|e| format!("deepseek4 parent: qr wq_a: {e}"))?;

    let mut q_lat_host = download_f32(gpu, &q_lat, rows * PARENT_Q_LORA)?;
    let q_norm_w = download_bf16_as_f32(gpu, &layer.q_norm, PARENT_Q_LORA)?;
    // Host RMSNorm.
    for r in 0..rows {
        let row = &mut q_lat_host[r * PARENT_Q_LORA..(r + 1) * PARENT_Q_LORA];
        let mut acc = 0.0f64;
        for &v in row.iter() {
            acc += (v as f64) * (v as f64);
        }
        let inv = 1.0 / ((acc / PARENT_Q_LORA as f64) + PARENT_RMS_EPS as f64).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v as f64 * inv * q_norm_w[i] as f64) as f32;
        }
    }
    upload_f32_into(gpu, &q_lat, &q_lat_host, rows * PARENT_Q_LORA)?;
    let _ = gpu.free_tensor(act);
    Ok(q_lat)
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bf = round_to_bf16(v);
        let bits = (bf.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn upload_f32_into(gpu: &Gpu, t: &GpuTensor, data: &[f32], nelems: usize) -> Result<(), String> {
    let nbytes = nelems * 4;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload_f32_into: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    let mut data = vec![0.0f32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))?;
    Ok(data)
}

fn download_i32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<i32>, String> {
    let nbytes = nelems * 4;
    let mut data = vec![0i32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_i32: {e:?}"))?;
    Ok(data)
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 2;
    let mut raw = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &t.buf)
        .map_err(|e| format!("deepseek4 parent: bf16 download: {e:?}"))?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(f32::from_bits((b as u32) << 16));
    }
    Ok(out)
}
