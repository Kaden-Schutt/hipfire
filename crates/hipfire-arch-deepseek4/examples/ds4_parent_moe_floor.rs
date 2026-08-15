// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Floor calibration for the assembled MoE host oracle.
//!
//! MoeRow0 certified L5 row0 expert-35 full path at max_abs=5.72e-6 /
//! mean_rel=1.78e-7 against a domain-matched (act-quant + f64) host oracle.
//! This harness asks the single question Main needs:
//!
//!   what does the *assembled* MoE block comparison report on that same L5
//!   row0 case, and how does it compare to L0?
//!
//! If assembled L5 row0 is ~5e-3, the bisect L0 figure is bf16-class floor and
//! not a defect. If L5 is ~1e-7 and L0 is 6e-3, L0 is real.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_moe_floor \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!      --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!      --rows 128
//! ```

use hipfire_arch_deepseek4::parent::attention::{
    PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_RMS_EPS, PARENT_SWA_WINDOW,
};
use hipfire_arch_deepseek4::parent::codec::{act_quant_fp8_inplace_ref, round_to_bf16};
use hipfire_arch_deepseek4::parent::forward::{
    parent_layer_forward, ParentForwardScratch, PARENT_HC_DIM, PARENT_HC_EPS, PARENT_HC_MULT,
    PARENT_HC_SINKHORN_ITERS,
};
use hipfire_arch_deepseek4::parent::head::parent_embed;
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::{
    expert_swiglu_ref, gate_hash_ref, gate_ref, hc_pre_ref, rms_norm_ref, RoutingResult,
};
use hipfire_arch_deepseek4::parent::moe::{
    parent_route, PARENT_MOE_INTER, PARENT_ROUTE_SCALE, PARENT_SWIGLU_LIMIT,
};
use hipfire_arch_deepseek4::parent::weights::{
    ParentLayerWeights, ParentLoadPlan, ParentWeights,
};
use hipfire_arch_deepseek4::parent::{Ds4ParentBackend, ParentQuantConfig};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_TOKEN_IDS: &str =
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin";
const DEFAULT_ROWS: usize = 128;
const VOCAB: usize = 129_280;
/// Layers to calibrate. L5 is MoeRow0's certified case; L0 is the bisect first-hit.
const CALIB_LAYERS: &[usize] = &[0, 5];

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
    let args = parse_args()?;
    let model_path = Path::new(&args.model);
    let mut token_ids = read_token_ids(&args.token_ids)?;
    if args.rows < token_ids.len() {
        token_ids.truncate(args.rows);
    } else if args.rows > token_ids.len() {
        return Err(format!(
            "deepseek4 parent: --rows {} exceeds token-ids length {}",
            args.rows,
            token_ids.len()
        ));
    }
    let rows = token_ids.len();
    let start_pos = 0usize;
    let max_layer = *CALIB_LAYERS.iter().max().unwrap() + 1;

    println!("=== ds4_parent_moe_floor ===");
    println!("model: {}", model_path.display());
    println!("token_ids: {} (n={rows})", args.token_ids.display());
    println!("calibrate layers: {CALIB_LAYERS:?}");
    println!("question: assembled MoE floor on L5 row0 (MoeRow0 case) vs L0");

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

    let admit_t0 = Instant::now();
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK ({:.1} ms): layers={} hash_layers={} n_routed={} topk={}",
        admit_t0.elapsed().as_secs_f64() * 1000.0,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..max_layer,
        load_experts: true,
    };
    println!("load plan: layers={:?} experts=true", plan.layers);
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!(
        "loaded layers={:?} experts={} in {:.3}s  resident={:.3} GiB",
        weights.layer_range,
        weights.experts_loaded,
        load_t0.elapsed().as_secs_f64(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, rows)?;
    let hc_a = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let mut kv_rings = Vec::with_capacity(max_layer);
    for i in 0..max_layer {
        let ring = zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )
        .map_err(|e| format!("kv_ring[{i}]: {e}"))?;
        kv_rings.push(ring);
    }

    parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc_a)?;

    let w_decode = gpu
        .alloc_tensor(&[PARENT_MOE_INTER, PARENT_DIM], DType::BF16)
        .map_err(|e| format!("deepseek4 parent: w_decode alloc: {e:?}"))?;

    // Host caches for calibrated layers only (filled on first visit).
    let mut host: Vec<Option<HostLayer>> = (0..max_layer).map(|_| None).collect();
    let mut use_a = true;

    for layer_i in 0..max_layer {
        let layer = &weights.layers[layer_i];
        let (x, out) = if use_a {
            (&hc_a, &hc_b)
        } else {
            (&hc_b, &hc_a)
        };
        let input_ids = if layer_i < cfg.num_hash_layers {
            Some(token_ids.as_slice())
        } else {
            None
        };

        parent_layer_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut scratch,
            layer_i,
            x,
            rows,
            start_pos,
            input_ids,
            &kv_rings[layer_i],
            out,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync L{layer_i}: {e:?}"))?;

        if CALIB_LAYERS.contains(&layer_i) {
            if host[layer_i].is_none() {
                host[layer_i] = Some(HostLayer::download(&gpu, layer, &cfg)?);
            }
            let hl = host[layer_i].as_ref().unwrap();
            calibrate_layer(
                &mut gpu,
                backend,
                layer,
                hl,
                &cfg,
                &scratch,
                &token_ids,
                rows,
                layer_i,
                &w_decode,
            )?;
        }
        use_a = !use_a;
    }

    let _ = gpu.free_tensor(w_decode);
    for r in kv_rings {
        let _ = gpu.free_tensor(r);
    }
    let _ = gpu.free_tensor(hc_a);
    let _ = gpu.free_tensor(hc_b);
    println!("RESULT floor_calibration_done");
    Ok(())
}

fn calibrate_layer(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    layer: &ParentLayerWeights,
    hl: &HostLayer,
    cfg: &ParentQuantConfig,
    scratch: &ParentForwardScratch,
    token_ids: &[u32],
    rows: usize,
    layer_i: usize,
    w_decode: &GpuTensor,
) -> Result<(), String> {
    let dim = PARENT_DIM;

    // GPU MoE I/O as actually executed by the just-finished forward.
    let moe_gpu = download_f32(gpu, scratch.stream_block(), rows * dim)?;
    let ffn_norm_f32 = download_f32(gpu, scratch.stream_normed(), rows * dim)?;
    let moe_x_bf16 = download_bf16_as_f32(gpu, scratch.moe_x_bf16(), rows * dim)?;

    let (r0_gpu_l2, r0_x_l2) = (row_l2(&moe_gpu, 0, dim), row_l2(&moe_x_bf16, 0, dim));
    let mut rest_gpu = 0.0f64;
    let mut rest_x = 0.0f64;
    let mut nrest = 0usize;
    for r in 1..rows.min(4) {
        rest_gpu += row_l2(&moe_gpu, r, dim);
        rest_x += row_l2(&moe_x_bf16, r, dim);
        nrest += 1;
    }
    if nrest > 0 {
        rest_gpu /= nrest as f64;
        rest_x /= nrest as f64;
    }

    // GPU routing (source of truth for what the block ran).
    let routing = parent_route(
        gpu,
        backend,
        layer,
        cfg,
        scratch.moe_x_bf16(),
        rows,
        if layer_i < cfg.num_hash_layers {
            Some(token_ids)
        } else {
            None
        },
    )?;
    let topk = routing.topk;
    let wsum: f64 = routing.weights[..topk].iter().map(|w| *w as f64).sum();
    println!();
    println!(
        "=== L{layer_i} ratio={} hash={} ===",
        layer.compress_ratio,
        layer_i < cfg.num_hash_layers
    );
    println!(
        "  row0 moe_x_L2={r0_x_l2:.4} moe_out_L2={r0_gpu_l2:.4}  rows1..3 mean x_L2={rest_x:.4} out_L2={rest_gpu:.4}"
    );
    println!(
        "  row0 indices={:?} weights={:?} sum={wsum:.6}",
        &routing.indices[..topk],
        &routing.weights[..topk]
    );

    // Host gate cross-check on BF16-widened x (matches parent_route input domain).
    {
        let is_hash = layer_i < cfg.num_hash_layers;
        let host_rt = if is_hash {
            let hash = gate_hash_ref(
                token_ids,
                hl.tid2eid.as_ref().unwrap(),
                cfg.n_routed_experts,
                topk,
            )?;
            route_hash_weights(
                &moe_x_bf16,
                &hl.gate_weight,
                &hash.indices,
                rows,
                cfg.n_routed_experts,
                topk,
            )?
        } else {
            gate_ref(
                &moe_x_bf16,
                &hl.gate_weight,
                hl.gate_bias.as_deref(),
                rows,
                dim,
                cfg.n_routed_experts,
                topk,
                PARENT_ROUTE_SCALE as f64,
                true,
            )?
        };
        let mut idx_mis = 0usize;
        let mut max_dw = 0.0f32;
        for i in 0..rows * topk {
            if routing.indices[i] != host_rt.indices[i] {
                idx_mis += 1;
            }
            max_dw = max_dw.max((routing.weights[i] - host_rt.weights[i]).abs());
        }
        println!(
            "  route vs host: idx_mismatch={idx_mis}/{} max|dw|={max_dw:.3e}",
            rows * topk
        );
    }

    // ── Primary: assembled MoE oracle with GPU routing + GPU BF16 x ────────
    let rt = RoutingResult {
        weights: routing.weights.clone(),
        indices: routing.indices.clone(),
    };
    let y_a = moe_ref_assembled(gpu, layer, hl, &moe_x_bf16, &rt, rows, w_decode)?;
    let m_a_all = metrics(&moe_gpu, &y_a);
    let m_a_r0 = metrics_row(&moe_gpu, &y_a, 0, dim);
    let m_a_r1 = if rows > 1 {
        metrics_row(&moe_gpu, &y_a, 1, dim)
    } else {
        (0.0, 0.0, 0.0)
    };
    println!(
        "  ASSEMBLED(gpu_route,gpu_bf16_x)  ALL  max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e}  gpu_L2={:.4} ref_L2={:.4}",
        m_a_all.0,
        m_a_all.1,
        m_a_all.2,
        l2(&moe_gpu),
        l2(&y_a)
    );
    println!(
        "  ASSEMBLED(gpu_route,gpu_bf16_x)  ROW0 max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e}  gpu_L2={r0_gpu_l2:.4} ref_L2={:.4}",
        m_a_r0.0,
        m_a_r0.1,
        m_a_r0.2,
        row_l2(&y_a, 0, dim)
    );
    println!(
        "  ASSEMBLED(gpu_route,gpu_bf16_x)  ROW1 max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e}",
        m_a_r1.0, m_a_r1.1, m_a_r1.2
    );

    // Shared-only host vs (assembled − routed) is hard without GPU split; instead
    // report shared-only host magnitude on row0 for context.
    {
        let gate = dense_linear_bf16_host(&moe_x_bf16[..dim], &hl.shared_w1, 1, PARENT_MOE_INTER, dim)?;
        let up = dense_linear_bf16_host(&moe_x_bf16[..dim], &hl.shared_w3, 1, PARENT_MOE_INTER, dim)?;
        let hid = expert_swiglu_ref(
            &gate,
            &up,
            1,
            PARENT_MOE_INTER,
            PARENT_SWIGLU_LIMIT as f64,
            None,
        );
        let shared0 = dense_linear_bf16_host(&hid, &hl.shared_w2, 1, dim, PARENT_MOE_INTER)?;
        println!(
            "  SHARED-only host row0 L2={:.4}  (context; MoeRow0 had ~95 on L5)",
            l2(&shared0)
        );
    }

    // Single-expert host path on row0 first expert (MoeRow0-style numbers).
    {
        let eid = routing.indices[0] as usize;
        let w = routing.weights[0];
        let y_host = one_expert_host(gpu, layer, eid, &moe_x_bf16[..dim], w, w_decode)?;
        println!(
            "  SINGLE expert eid={eid} w={w:.5} row0 host_L2={:.4}  (MoeRow0 eid35 was 215.5968)",
            l2(&y_host)
        );
    }

    // Bisect-style path: f32 ffn_norm_ref + host route (what produced 6e-3).
    {
        let residual_hc = download_f32(gpu, scratch.residual_hc(), rows * PARENT_HC_DIM)?;
        let (y_ref, _, _) = hc_pre_ref(
            &residual_hc,
            &hl.hc_ffn_fn,
            &hl.hc_ffn_scale,
            &hl.hc_ffn_base,
            rows,
            PARENT_HC_MULT,
            dim,
            PARENT_RMS_EPS as f64,
            PARENT_HC_SINKHORN_ITERS as usize,
            PARENT_HC_EPS as f64,
        )?;
        let ffn_norm_ref = rms_norm_ref(&y_ref, &hl.ffn_norm, PARENT_RMS_EPS as f64, dim);
        let norm_m = metrics(&ffn_norm_f32, &ffn_norm_ref);
        let x_m = metrics(&moe_x_bf16, &ffn_norm_ref);
        let host_rt = route_host(
            &ffn_norm_ref,
            hl,
            cfg,
            layer_i,
            rows,
            Some(token_ids),
        )?;
        let y_b = moe_ref_assembled(gpu, layer, hl, &ffn_norm_ref, &host_rt, rows, w_decode)?;
        let m_b_all = metrics(&moe_gpu, &y_b);
        let m_b_r0 = metrics_row(&moe_gpu, &y_b, 0, dim);
        println!(
            "  BISECT-style (f32_norm_ref+host_route) ALL  max_abs={:.6e} l2_rel={:.6e}  (ffn_norm gpu-vs-ref max_abs={:.3e}; bf16_x-vs-ref max_abs={:.3e})",
            m_b_all.0, m_b_all.2, norm_m.0, x_m.0
        );
        println!(
            "  BISECT-style ROW0 max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e}",
            m_b_r0.0, m_b_r0.1, m_b_r0.2
        );
    }

    let floor_tag = if m_a_r0.0 < 1e-4 {
        "SUB_1e-4 (f32-class; NOT bf16 floor)"
    } else if m_a_r0.0 < 1e-2 {
        "1e-4..1e-2 (bf16-class floor candidate)"
    } else {
        ">=1e-2 (above bf16 unit roundoff; real gap)"
    };
    println!(
        "  FLOOR_CALL L{layer_i} assembled_row0_max_abs={:.6e} → {floor_tag}",
        m_a_r0.0
    );
    println!(
        "RESULT layer={layer_i} assembled_row0_max_abs={:.6e} assembled_row0_l2_rel={:.6e} assembled_all_max_abs={:.6e} assembled_all_l2_rel={:.6e}",
        m_a_r0.0, m_a_r0.2, m_a_all.0, m_a_all.2
    );
    Ok(())
}

// ── MoE assembled oracle (domain-matched: act-quant + f64 GEMM, BF16 w) ─────

fn moe_ref_assembled(
    gpu: &mut Gpu,
    layer: &ParentLayerWeights,
    hl: &HostLayer,
    x_f32: &[f32],
    routing: &RoutingResult,
    rows: usize,
    w_decode: &GpuTensor,
) -> Result<Vec<f32>, String> {
    let dim = PARENT_DIM;
    let inter = PARENT_MOE_INTER;
    let topk = routing.indices.len() / rows;
    let n_experts = layer.experts.len();
    let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    for r in 0..rows {
        for t in 0..topk {
            let eid = routing.indices[r * topk + t] as usize;
            let w = routing.weights[r * topk + t];
            if eid >= n_experts {
                return Err(format!("expert id {eid} oob"));
            }
            groups[eid].push((r, w));
        }
    }
    let mut y = vec![0.0f32; rows * dim];
    for (eid, members) in groups.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let n_tok = members.len();
        let expert = &layer.experts[eid];
        let mut xg = vec![0.0f32; n_tok * dim];
        let mut rw = vec![0.0f32; n_tok];
        for (i, &(row, w)) in members.iter().enumerate() {
            xg[i * dim..(i + 1) * dim].copy_from_slice(&x_f32[row * dim..(row + 1) * dim]);
            rw[i] = w;
        }
        expert
            .w1
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("w1 decode: {e}"))?;
        let w1 = download_bf16_as_f32(gpu, w_decode, inter * dim)?;
        let gate = dense_linear_bf16_host(&xg, &w1, n_tok, inter, dim)?;
        expert
            .w3
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("w3 decode: {e}"))?;
        let w3 = download_bf16_as_f32(gpu, w_decode, inter * dim)?;
        let up = dense_linear_bf16_host(&xg, &w3, n_tok, inter, dim)?;
        let hid = expert_swiglu_ref(
            &gate,
            &up,
            n_tok,
            inter,
            PARENT_SWIGLU_LIMIT as f64,
            Some(&rw),
        );
        expert
            .w2
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("w2 decode: {e}"))?;
        let w2 = download_bf16_as_f32(gpu, w_decode, dim * inter)?;
        let eout = dense_linear_bf16_host(&hid, &w2, n_tok, dim, inter)?;
        for (i, &(row, _)) in members.iter().enumerate() {
            let src = i * dim;
            let dst = row * dim;
            for j in 0..dim {
                y[dst + j] += eout[src + j];
            }
        }
    }
    // Shared.
    let gate = dense_linear_bf16_host(x_f32, &hl.shared_w1, rows, inter, dim)?;
    let up = dense_linear_bf16_host(x_f32, &hl.shared_w3, rows, inter, dim)?;
    let hid = expert_swiglu_ref(
        &gate,
        &up,
        rows,
        inter,
        PARENT_SWIGLU_LIMIT as f64,
        None,
    );
    let shared = dense_linear_bf16_host(&hid, &hl.shared_w2, rows, dim, inter)?;
    for i in 0..rows * dim {
        y[i] += shared[i];
    }
    Ok(y)
}

fn one_expert_host(
    gpu: &mut Gpu,
    layer: &ParentLayerWeights,
    eid: usize,
    x0: &[f32],
    w: f32,
    w_decode: &GpuTensor,
) -> Result<Vec<f32>, String> {
    let dim = PARENT_DIM;
    let inter = PARENT_MOE_INTER;
    let expert = &layer.experts[eid];
    expert.w1.decode_into(gpu, w_decode)?;
    let w1 = download_bf16_as_f32(gpu, w_decode, inter * dim)?;
    expert.w3.decode_into(gpu, w_decode)?;
    let w3 = download_bf16_as_f32(gpu, w_decode, inter * dim)?;
    expert.w2.decode_into(gpu, w_decode)?;
    let w2 = download_bf16_as_f32(gpu, w_decode, dim * inter)?;
    let gate = dense_linear_bf16_host(x0, &w1, 1, inter, dim)?;
    let up = dense_linear_bf16_host(x0, &w3, 1, inter, dim)?;
    let hid = expert_swiglu_ref(&gate, &up, 1, inter, PARENT_SWIGLU_LIMIT as f64, Some(&[w]));
    dense_linear_bf16_host(&hid, &w2, 1, dim, inter)
}

fn dense_linear_bf16_host(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * k {
        return Err(format!("x len {} != rows*k {}", x.len(), rows * k));
    }
    if w.len() != n * k {
        return Err(format!("w len {} != n*k {}", w.len(), n * k));
    }
    // Domain match: BF16 lattice then act-quant (act_quant itself BF16-rounds first).
    let mut xq: Vec<f32> = x.iter().copied().map(round_to_bf16).collect();
    act_quant_fp8_inplace_ref(&mut xq, k, 128)?;
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xb = r * k;
        for o in 0..n {
            let mut s = 0.0f64;
            let wb = o * k;
            for i in 0..k {
                s += (xq[xb + i] as f64) * (w[wb + i] as f64);
            }
            out[r * n + o] = s as f32;
        }
    }
    Ok(out)
}

fn route_hash_weights(
    x: &[f32],
    gate_w: &[f32],
    indices: &[u32],
    rows: usize,
    n_experts: usize,
    topk: usize,
) -> Result<RoutingResult, String> {
    let dim = PARENT_DIM;
    let mut scores = vec![0.0f32; rows * n_experts];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        for e in 0..n_experts {
            let wr = &gate_w[e * dim..(e + 1) * dim];
            let mut acc = 0.0f64;
            for k in 0..dim {
                acc += xr[k] as f64 * wr[k] as f64;
            }
            let sp = if acc > 0.0 {
                acc + (-acc).exp().ln_1p()
            } else {
                acc.exp().ln_1p()
            };
            scores[r * n_experts + e] = sp.sqrt() as f32;
        }
    }
    let mut weights = vec![0.0f32; rows * topk];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for t in 0..topk {
            let eid = indices[r * topk + t] as usize;
            let s = scores[r * n_experts + eid];
            weights[r * topk + t] = s;
            sum += s;
        }
        if sum > 0.0 {
            for t in 0..topk {
                weights[r * topk + t] = weights[r * topk + t] / sum * PARENT_ROUTE_SCALE;
            }
        }
    }
    Ok(RoutingResult {
        weights,
        indices: indices.to_vec(),
    })
}

fn route_host(
    x: &[f32],
    hl: &HostLayer,
    cfg: &ParentQuantConfig,
    layer_idx: usize,
    rows: usize,
    input_ids: Option<&[u32]>,
) -> Result<RoutingResult, String> {
    let dim = PARENT_DIM;
    let n_experts = cfg.n_routed_experts;
    let topk = cfg.num_experts_per_tok;
    if layer_idx < cfg.num_hash_layers {
        let ids = input_ids.ok_or("hash needs ids")?;
        let tid2eid = hl.tid2eid.as_ref().ok_or("missing tid2eid")?;
        let hash = gate_hash_ref(ids, tid2eid, n_experts, topk)?;
        route_hash_weights(x, &hl.gate_weight, &hash.indices, rows, n_experts, topk)
    } else {
        gate_ref(
            x,
            &hl.gate_weight,
            hl.gate_bias.as_deref(),
            rows,
            dim,
            n_experts,
            topk,
            PARENT_ROUTE_SCALE as f64,
            true,
        )
    }
}

// ── Host layer cache ────────────────────────────────────────────────────────

struct HostLayer {
    ffn_norm: Vec<f32>,
    hc_ffn_fn: Vec<f32>,
    hc_ffn_base: Vec<f32>,
    hc_ffn_scale: Vec<f32>,
    gate_weight: Vec<f32>,
    gate_bias: Option<Vec<f32>>,
    tid2eid: Option<Vec<i64>>,
    shared_w1: Vec<f32>,
    shared_w2: Vec<f32>,
    shared_w3: Vec<f32>,
}

impl HostLayer {
    fn download(
        gpu: &Gpu,
        layer: &ParentLayerWeights,
        cfg: &ParentQuantConfig,
    ) -> Result<Self, String> {
        let dim = PARENT_DIM;
        let inter = PARENT_MOE_INTER;
        let mix_hc = (2 + PARENT_HC_MULT) * PARENT_HC_MULT;
        let hc_flat = PARENT_HC_DIM;
        let gate_bias = layer
            .gate_bias
            .as_ref()
            .map(|b| download_f32(gpu, b, cfg.n_routed_experts))
            .transpose()?;
        let tid2eid = layer
            .tid2eid
            .as_ref()
            .map(|t| download_i64(gpu, t, VOCAB * cfg.num_experts_per_tok))
            .transpose()?;
        Ok(Self {
            ffn_norm: download_bf16_as_f32(gpu, &layer.ffn_norm, dim)?,
            hc_ffn_fn: download_f32(gpu, &layer.hc_ffn_fn, mix_hc * hc_flat)?,
            hc_ffn_base: download_f32(gpu, &layer.hc_ffn_base, mix_hc)?,
            hc_ffn_scale: download_f32(gpu, &layer.hc_ffn_scale, 3)?,
            gate_weight: download_bf16_as_f32(
                gpu,
                &layer.gate_weight,
                cfg.n_routed_experts * dim,
            )?,
            gate_bias,
            tid2eid,
            shared_w1: download_bf16_as_f32(gpu, layer.shared_w1.tensor(), inter * dim)?,
            shared_w2: download_bf16_as_f32(gpu, layer.shared_w2.tensor(), dim * inter)?,
            shared_w3: download_bf16_as_f32(gpu, layer.shared_w3.tensor(), inter * dim)?,
        })
    }
}

// ── Metrics / IO ────────────────────────────────────────────────────────────

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut n_rel = 0usize;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as f64 - y as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        let ay = (y as f64).abs();
        if ay > 1e-8 {
            sum_rel += d / ay;
            n_rel += 1;
        }
        let dd = x as f64 - y as f64;
        num += dd * dd;
        den += (y as f64) * (y as f64);
    }
    let mean_rel = if n_rel > 0 {
        sum_rel / n_rel as f64
    } else {
        0.0
    };
    let l2_rel = if den > 0.0 {
        num.sqrt() / den.sqrt()
    } else {
        num.sqrt()
    };
    (max_abs, mean_rel, l2_rel)
}

fn metrics_row(a: &[f32], b: &[f32], row: usize, dim: usize) -> (f64, f64, f64) {
    metrics(
        &a[row * dim..(row + 1) * dim],
        &b[row * dim..(row + 1) * dim],
    )
}

fn l2(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn row_l2(v: &[f32], row: usize, dim: usize) -> f64 {
    l2(&v[row * dim..(row + 1) * dim])
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "f32 download too small have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("f32 dtoh: {e:?}"))?;
    Ok(data)
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 2;
    if t.buf.size() < nbytes {
        return Err(format!(
            "bf16 download too small have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("bf16 dtoh: {e:?}"))?;
    let mut out = vec![0.0f32; nelems];
    for i in 0..nelems {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out[i] = f32::from_bits((bits as u32) << 16);
    }
    Ok(out)
}

fn download_i64(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<i64>, String> {
    let nbytes = nelems * 8;
    if t.buf.size() < nbytes {
        return Err(format!(
            "i64 download too small have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut data = vec![0i64; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("i64 dtoh: {e:?}"))?;
    Ok(data)
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: zeros_f32: {e:?}"))
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read tokens: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("tokens.bin len {} not multiple of 4", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(out)
}

struct Args {
    model: String,
    token_ids: PathBuf,
    rows: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model = DEFAULT_MODEL.to_owned();
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut rows = DEFAULT_ROWS;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                model = it.next().ok_or("--model needs value")?;
            }
            "--token-ids" => {
                token_ids = PathBuf::from(it.next().ok_or("--token-ids needs value")?);
            }
            "--rows" => {
                rows = it
                    .next()
                    .ok_or("--rows needs value")?
                    .parse()
                    .map_err(|e| format!("--rows: {e}"))?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "ds4_parent_moe_floor [--model DIR] [--token-ids PATH] [--rows N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        model,
        token_ids,
        rows,
    })
}
