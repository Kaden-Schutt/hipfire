// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 5 attention lift smoke: run parent attention on one layer of each
//! compress_ratio class against the real checkpoint.
//!
//! - layer 0  ratio 0   pure SWA (regression lock)
//! - layer 2  ratio 4   compressor + indexer
//! - layer 3  ratio 128 compressor only
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_attn_mixed_smoke \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```
//!
//! Must run on gfx942 (mi300x).

use hipfire_ds4_parent::attention::{
    all_finite, get_compress_topk_idxs, get_window_topk_idxs, l2_norm, parent_attention_swa,
    swa_n_valid, ParentAttnScratch, PARENT_ATTN_INDEX_TOPK, PARENT_DIM, PARENT_HEAD_DIM,
    PARENT_N_HEADS, PARENT_N_KV_HEADS, PARENT_Q_WIDTH, PARENT_SWA_WINDOW,
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
const START_POS: usize = 0;
const LAYERS: &[(usize, usize)] = &[(0, 0), (2, 4), (3, 128)];
/// Prior Gate-4 layer-0 attention-only output L2 (synthetic post-norm x).
const LAYER0_ATTN_ONLY_L2_REF: f32 = 314.98;
const LAYER0_ATTN_ONLY_L2_TOL: f32 = 2.0;
/// Prior pos0 head0 probs (kv + sink) on the same synthetic input.
const LAYER0_P_KV_REF: f32 = 0.233705;
const LAYER0_P_SINK_REF: f32 = 0.766295;
const LAYER0_P_TOL: f32 = 1e-4;


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

    println!("=== ds4_parent_attn_mixed_smoke ===");
    println!("model: {}", model_path.display());
    println!("rows: {ROWS}  start_pos: {START_POS}");
    println!("layers: {:?}", LAYERS);

    host_index_checks()?;

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
    for &(li, er) in LAYERS {
        let got = cfg.compress_ratio(li);
        if got != er {
            return Err(format!(
                "deepseek4 parent: expected layer {li} compress_ratio={er}, got {got}"
            ));
        }
    }
    println!(
        "admit OK: layers={} ratios[0,2,3]=[{},{},{}]",
        cfg.num_hidden_layers,
        cfg.compress_ratio(0),
        cfg.compress_ratio(2),
        cfg.compress_ratio(3)
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: 0..4,
        load_experts: true,
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

    let mut attn_scratch = ParentAttnScratch::new(&mut gpu, &cfg, ROWS)?;
    let attn_bytes = attn_scratch.bytes();
    println!(
        "ParentAttnScratch::bytes() = {attn_bytes} ({:.3} MiB)  max_rows={} max_n_compressed={}",
        attn_bytes as f64 / (1024.0 * 1024.0),
        attn_scratch.max_rows(),
        attn_scratch.max_n_compressed()
    );
    // Historical pure-SWA own-tile footprint was ~13.6 MiB at 16 rows; the
    // lift adds topk_staged (~16 MiB) + main_kv + compressor + indexer.
    if attn_bytes < 20 * 1024 * 1024 {
        return Err(format!(
            "deepseek4 parent: ParentAttnScratch bytes {attn_bytes} too small — \
             compressor/indexer tiles missing?"
        ));
    }
    // Deterministic post-attn_norm F32 activations (same grid as Gate-4 attn smoke).
    // Used for attention-only path + pos0 probability regression lock.
    let mut x_attn_host = vec![0.0f32; ROWS * PARENT_DIM];
    for r in 0..ROWS {
        for k in 0..PARENT_DIM {
            let v = (((r * 131 + k * 17) % 200) as f32 - 100.0) * 0.01;
            x_attn_host[r * PARENT_DIM + k] = round_to_bf16(v);
        }
    }
    let x_attn = upload_f32(&mut gpu, &x_attn_host, &[ROWS, PARENT_DIM])?;

    // Layer-forward residual: real embed rows expanded across hc streams
    // (matches Gate-4 layer gate input so attn_out L2 314.98 is comparable).
    const VOCAB: usize = 129_280;
    const TOKEN_SEED: u64 = 0xD5_46_A7_E4_04_6A_7E_u64;
    let token_ids = select_token_ids(TOKEN_SEED, ROWS);
    println!("token_ids = {:?}", token_ids);
    let embed_host = download_bf16_as_f32(&gpu, &weights.embed, VOCAB * PARENT_DIM)?;
    let mut x_single = vec![0.0f32; ROWS * PARENT_DIM];
    for (r, &tid) in token_ids.iter().enumerate() {
        let src = (tid as usize) * PARENT_DIM;
        x_single[r * PARENT_DIM..(r + 1) * PARENT_DIM]
            .copy_from_slice(&embed_host[src..src + PARENT_DIM]);
    }
    let mut x_hc = vec![0.0f32; ROWS * PARENT_HC_DIM];
    for r in 0..ROWS {
        let row = &x_single[r * PARENT_DIM..(r + 1) * PARENT_DIM];
        for h in 0..PARENT_HC_MULT {
            let dst = (r * PARENT_HC_MULT + h) * PARENT_DIM;
            x_hc[dst..dst + PARENT_DIM].copy_from_slice(row);
        }
    }
    let x_layer = upload_f32(&mut gpu, &x_hc, &[ROWS, PARENT_HC_MULT, PARENT_DIM])?;

    // Report fwd scratch after construction (before any forward).
    let mut fwd_scratch = ParentForwardScratch::new(&mut gpu, &cfg, ROWS)?;
    let fwd_bytes_before = fwd_scratch.bytes();
    println!(
        "ParentForwardScratch::bytes() = {fwd_bytes_before} ({:.3} MiB)",
        fwd_bytes_before as f64 / (1024.0 * 1024.0)
    );


    for &(layer_idx, expect_ratio) in LAYERS {
        let local = layer_idx - weights.layer_range.start;
        let layer = &weights.layers[local];
        if layer.compress_ratio != expect_ratio {
            return Err(format!(
                "deepseek4 parent: layer {layer_idx} compress_ratio={} != {expect_ratio}",
                layer.compress_ratio
            ));
        }
        println!();
        println!(
            "── layer {layer_idx}  compress_ratio={expect_ratio} ──────────────────────"
        );

        let kv_ring = zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )?;
        let out_attn = zeros_f32(&mut gpu, &[ROWS, PARENT_DIM])?;

        let t0 = Instant::now();
        parent_attention_swa(
            &mut gpu,
            backend,
            layer,
            &cfg,
            &mut attn_scratch,
            &x_attn,
            ROWS,
            START_POS,
            &kv_ring,
            &out_attn,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("deepseek4 parent: sync attn l{layer_idx}: {e:?}"))?;
        let attn_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let out_host = download_f32(&gpu, &out_attn, ROWS * PARENT_DIM)?;
        let finite = all_finite(&out_host);
        let out_norm = l2_norm(&out_host);
        let q_host = download_f32(&gpu, attn_scratch.q_f32_ref()?, ROWS * PARENT_Q_WIDTH)?;
        let kv_host = download_f32(&gpu, attn_scratch.kv_f32_ref()?, ROWS * PARENT_HEAD_DIM)?;
        let attn_pre = download_f32(
            &gpu,
            attn_scratch.attn_out_f32_ref()?,
            ROWS * PARENT_Q_WIDTH,
        )?;
        println!(
            "  attn-only: finite={finite}  out_L2={out_norm:.6}  wall={attn_ms:.2} ms"
        );
        println!(
            "  stage: q_post_rope={:.6}  kv_post_quant={:.6}  attn_pre_wo_a={:.6}",
            l2_norm(&q_host),
            l2_norm(&kv_host),
            l2_norm(&attn_pre)
        );
        if !finite {
            return Err(format!(
                "deepseek4 parent: layer {layer_idx} attn output non-finite"
            ));
        }

        let sink = download_f32(&gpu, &layer.attn_sink, PARENT_N_HEADS)?;
        if expect_ratio == 0 {
            let (p_kv, p_sink) = pos0_swa_probs(&q_host, &kv_host, sink[0])?;
            println!(
                "  pos0 head0 probs: kv={p_kv:.6}  sink={p_sink:.6}  sum={:.6}",
                p_kv + p_sink
            );
            if ((p_kv + p_sink) - 1.0).abs() >= 1e-5 {
                return Err(format!(
                    "deepseek4 parent: layer 0 pos0 probs sum {}",
                    p_kv + p_sink
                ));
            }
            if (p_kv - LAYER0_P_KV_REF).abs() > LAYER0_P_TOL
                || (p_sink - LAYER0_P_SINK_REF).abs() > LAYER0_P_TOL
            {
                return Err(format!(
                    "deepseek4 parent: layer 0 pos0 probs REGRESSED \
                     (kv={p_kv:.6} expect≈{LAYER0_P_KV_REF}, \
                      sink={p_sink:.6} expect≈{LAYER0_P_SINK_REF})"
                ));
            }
            println!(
                "  ratio-0 regression LOCK: pos0 probs match prior \
                 ({LAYER0_P_KV_REF}+{LAYER0_P_SINK_REF})"
            );
            let d = (out_norm - LAYER0_ATTN_ONLY_L2_REF).abs();
            println!(
                "  ratio-0 regression: attn-only out_L2={out_norm:.4}  \
                 prior={LAYER0_ATTN_ONLY_L2_REF}  |Δ|={d:.4}"
            );
            if d > LAYER0_ATTN_ONLY_L2_TOL {
                return Err(format!(
                    "deepseek4 parent: layer 0 attn-only out_L2 REGRESSED \
                     (got {out_norm:.4}, prior {LAYER0_ATTN_ONLY_L2_REF}, \
                      tol {LAYER0_ATTN_ONLY_L2_TOL})"
                ));
            }
            println!("  ratio-0 regression LOCK: attn-only out_L2 within tol");
        } else if expect_ratio == 4 {
            let n_active = download_i32(&gpu, attn_scratch.n_active_topk_ref(), ROWS)?;
            let n_swa0 = swa_n_valid(START_POS, 0, PARENT_SWA_WINDOW);
            let n_tk0 = n_active[0].max(0) as usize;
            let n_swa15 = swa_n_valid(START_POS, 15, PARENT_SWA_WINDOW);
            let n_tk15 = n_active[15].max(0) as usize;
            println!(
                "  pos0  n_valid_swa={n_swa0}  n_active_topk={n_tk0}  joint_slots={}",
                n_swa0 + n_tk0 + 1
            );
            println!(
                "  pos15 n_valid_swa={n_swa15} n_active_topk={n_tk15} joint_slots={}",
                n_swa15 + n_tk15 + 1
            );
            if n_tk15 == 0 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} pos15 expected n_active_topk>0 \
                     (rows={ROWS} ratio=4), got 0"
                ));
            }
            let topk_idx = download_i32(
                &gpu,
                attn_scratch.topk_idx_ref(),
                ROWS * PARENT_ATTN_INDEX_TOPK,
            )?;
            let row15 = &topk_idx[15 * PARENT_ATTN_INDEX_TOPK..16 * PARENT_ATTN_INDEX_TOPK];
            let mut seen = std::collections::BTreeSet::new();
            let mut n_valid_idx = 0usize;
            let mut dup = 0usize;
            for &v in row15 {
                if v < 0 {
                    continue;
                }
                n_valid_idx += 1;
                if !seen.insert(v) {
                    dup += 1;
                }
            }
            println!(
                "  pos15 topk: valid_idx={n_valid_idx}  dups={dup}  \
                 (indices address main_kv_cache, disjoint from SWA ring)"
            );
            if dup > 0 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} pos15 topk has {dup} internal duplicates"
                ));
            }
            let p0 = joint_prob_sum_row_head0(
                &gpu,
                attn_scratch.swa_staged_ref(),
                attn_scratch.topk_staged_ref(),
                sink[0],
                /*row=*/ 0,
                n_swa0,
                n_tk0,
                &q_host,
            )?;
            let p15 = joint_prob_sum_row_head0(
                &gpu,
                attn_scratch.swa_staged_ref(),
                attn_scratch.topk_staged_ref(),
                sink[0],
                /*row=*/ 15,
                n_swa15,
                n_tk15,
                &q_host,
            )?;
            println!("  pos0  head0 joint prob sum = {p0:.6}");
            println!("  pos15 head0 joint prob sum = {p15:.6}");
            if (p0 - 1.0).abs() >= 1e-4 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} pos0 joint probs sum {p0}"
                ));
            }
            if (p15 - 1.0).abs() >= 1e-4 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} pos15 joint probs sum {p15}"
                ));
            }

        } else {
            // ratio 128 at rows=16: no compress event yet (16/128=0).
            let n_active = download_i32(&gpu, attn_scratch.n_active_topk_ref(), ROWS)?;
            let max_a = n_active.iter().copied().max().unwrap_or(0);
            println!(
                "  ratio-128 short prefill: max n_active_topk={max_a} \
                 (expect 0 when rows < ratio)"
            );
            if max_a != 0 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} expected n_active=0 at \
                     rows={ROWS} ratio=128, got max={max_a}"
                ));
            }
            let (p_kv, p_sink) = pos0_swa_probs(&q_host, &kv_host, sink[0])?;
            println!(
                "  pos0 head0 (SWA-only fallback) probs: kv={p_kv:.6} sink={p_sink:.6} sum={:.6}",
                p_kv + p_sink
            );
            if ((p_kv + p_sink) - 1.0).abs() >= 1e-5 {
                return Err(format!(
                    "deepseek4 parent: layer {layer_idx} pos0 probs sum {}",
                    p_kv + p_sink
                ));
            }
        }

        // Full layer forward (traced).
        let out_layer = zeros_f32(&mut gpu, &[ROWS, PARENT_HC_MULT, PARENT_DIM])?;
        let kv_ring2 = zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )?;
        upload_f32_into(&gpu, &x_layer, &x_hc)?;
        let mut trace = ParentLayerTrace::default();
        let t1 = Instant::now();
        parent_layer_forward_traced(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut fwd_scratch,
            layer_idx,
            &x_layer,
            ROWS,
            START_POS,
            Some(&token_ids),
            &kv_ring2,
            &out_layer,
            &mut trace,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("deepseek4 parent: sync layer l{layer_idx}: {e:?}"))?;
        let layer_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let out_hc = download_f32(&gpu, &out_layer, ROWS * PARENT_HC_DIM)?;
        let layer_finite = all_finite(&out_hc);
        println!(
            "  layer-fwd: finite={layer_finite}  out_L2={:.6}  wall={layer_ms:.2} ms",
            l2_norm(&out_hc)
        );
        println!(
            "  stages: hc_pre_attn={:.4} attn_norm={:.4} attn_out={:.4} \
             hc_post_attn={:.4} hc_pre_ffn={:.4} ffn_norm={:.4} moe_out={:.4} \
             hc_post_ffn={:.4}",
            trace.hc_pre_attn,
            trace.attn_norm,
            trace.attn_out,
            trace.hc_post_attn,
            trace.hc_pre_ffn,
            trace.ffn_norm,
            trace.moe_out,
            trace.hc_post_ffn
        );
        if !layer_finite {
            return Err(format!(
                "deepseek4 parent: layer {layer_idx} full forward non-finite"
            ));
        }
        if layer_idx == 0 {
            // Layer-forward residual uses embed rows (Gate-4 style); attn_out
            // L2 here is not the same quantity as the synthetic attn-only
            // baseline. Report it for the gate log; hard lock is attn-only.
            println!(
                "  layer-fwd attn_out L2={:.4}  (embed residual; attn-only lock above)",
                trace.attn_out
            );
        }
    }



    let bytes_after = fwd_scratch.bytes();
    println!();
    println!(
        "ParentForwardScratch::bytes() before={fwd_bytes_before} after={bytes_after} \
         (capacity fixed; no per-call device alloc by contract)"
    );
    println!("ParentAttnScratch::bytes() = {attn_bytes}");
    println!("PASS");
    Ok(())
}

fn host_index_checks() -> Result<(), String> {
    let w = get_window_topk_idxs(128, 16, 0)?;
    let k = 16;
    if w[0] != 0 || w[1..k].iter().any(|&v| v != -1) {
        return Err("deepseek4 parent: ratio-0 window pos0 broken".into());
    }
    let c4 = get_compress_topk_idxs(4, 16, 0, 0)?;
    if c4.len() != 16 * 4 {
        return Err(format!(
            "deepseek4 parent: compress topk ratio4 prefill len {}",
            c4.len()
        ));
    }
    if &c4[15 * 4..16 * 4] != [0, 1, 2, 3] {
        return Err(format!(
            "deepseek4 parent: compress topk ratio4 row15 {:?}",
            &c4[15 * 4..16 * 4]
        ));
    }
    let d4 = get_compress_topk_idxs(4, 1, 15, 0)?;
    if d4 != [0, 1, 2, 3] {
        return Err(format!(
            "deepseek4 parent: compress topk ratio4 decode {:?}",
            d4
        ));
    }
    let c128 = get_compress_topk_idxs(128, 16, 0, 0)?;
    if !c128.is_empty() {
        return Err("deepseek4 parent: ratio128 short prefill should be empty".into());
    }
    let d128 = get_compress_topk_idxs(128, 1, 255, 0)?;
    if d128 != [0, 1] {
        return Err(format!(
            "deepseek4 parent: compress topk ratio128 decode {:?}",
            d128
        ));
    }
    println!("host index construction: OK (ratio 0/4/128 prefill+decode)");
    Ok(())
}

fn pos0_swa_probs(q_host: &[f32], kv_host: &[f32], sink0: f32) -> Result<(f32, f32), String> {
    let inv_scale = 1.0f32 / (PARENT_HEAD_DIM as f32).sqrt();
    let mut dot = 0.0f64;
    for d in 0..PARENT_HEAD_DIM {
        dot += q_host[d] as f64 * kv_host[d] as f64;
    }
    let score0 = (dot as f32) * inv_scale;
    let m = score0.max(sink0);
    let e0 = (score0 - m).exp();
    let es = (sink0 - m).exp();
    let z = e0 + es;
    Ok((e0 / z, es / z))
}

/// Host recompute of row `r` head0 joint softmax over SWA staged + topk staged + sink.
/// Returns the sum of normalized probabilities (must be 1).
fn joint_prob_sum_row_head0(
    gpu: &Gpu,
    swa_staged: &GpuTensor,
    topk_staged: &GpuTensor,
    sink0: f32,
    row: usize,
    n_swa: usize,
    n_topk: usize,
    q_host: &[f32],
) -> Result<f32, String> {
    let inv_scale = 1.0f32 / (PARENT_HEAD_DIM as f32).sqrt();
    // Download full staged buffers (small: 16*512*128 and 16*512*512).
    let swa_all = download_f32(
        gpu,
        swa_staged,
        ROWS * PARENT_HEAD_DIM * PARENT_SWA_WINDOW,
    )?;
    let topk_all = download_f32(
        gpu,
        topk_staged,
        ROWS * PARENT_HEAD_DIM * PARENT_ATTN_INDEX_TOPK,
    )?;
    let swa_base = row * PARENT_HEAD_DIM * PARENT_SWA_WINDOW;
    let topk_base = row * PARENT_HEAD_DIM * PARENT_ATTN_INDEX_TOPK;
    let q_base = row * PARENT_Q_WIDTH; // head 0 starts here
    let mut scores = Vec::with_capacity(n_swa + n_topk + 1);
    for col in 0..n_swa {
        let mut dot = 0.0f64;
        for d in 0..PARENT_HEAD_DIM {
            // layout: [row, d, col] with col stride = window
            dot += q_host[q_base + d] as f64
                * swa_all[swa_base + d * PARENT_SWA_WINDOW + col] as f64;
        }
        scores.push((dot as f32) * inv_scale);
    }
    for col in 0..n_topk {
        let mut dot = 0.0f64;
        for d in 0..PARENT_HEAD_DIM {
            dot += q_host[q_base + d] as f64
                * topk_all[topk_base + d * PARENT_ATTN_INDEX_TOPK + col] as f64;
        }
        scores.push((dot as f32) * inv_scale);
    }
    scores.push(sink0);
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for &s in &scores {
        z += (s - m).exp();
    }
    if z <= 0.0 || !z.is_finite() {
        return Err(format!(
            "deepseek4 parent: joint softmax partition non-finite/zero z={z} row={row}"
        ));
    }
    let mut sum_p = 0.0f32;
    for &s in &scores {
        sum_p += (s - m).exp() / z;
    }
    Ok(sum_p)
}


fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn upload_f32_into(gpu: &Gpu, t: &GpuTensor, data: &[f32]) -> Result<(), String> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload_f32_into: {e:?}"))
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: zeros: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; nelems * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))?;
    let mut out = vec![0.0f32; nelems];
    for i in 0..nelems {
        out[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    Ok(out)
}

fn download_i32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<i32>, String> {
    let mut bytes = vec![0u8; nelems * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_i32: {e:?}"))?;
    let mut out = vec![0i32; nelems];
    for i in 0..nelems {
        out[i] = i32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    Ok(out)
}

fn select_token_ids(seed: u64, n: usize) -> Vec<u32> {
    const VOCAB: u64 = 129_280;
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    let fixed = [0u32, 1, 2, 7, 42, 256, 1000, 50256];
    for &t in fixed.iter().take(n.min(fixed.len())) {
        out.push((t as u64 % VOCAB) as u32);
    }
    while out.len() < n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push((z % VOCAB) as u32);
    }
    out
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 2;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download_bf16 short (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_bf16: {e:?}"))?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    Ok(out)
}

