// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Hypothesis-free per-layer f64 bisect of the DS4 parent forward.
//!
//! Runs a 128-token parent forward and, for each layer, feeds that layer's
//! *actual GPU residual* into a composed f64 reference of the FFN half
//! (`hc_pre_ffn` → `ffn_norm` → route/MoE → `hc_post_ffn`), comparing against
//! the GPU layer output. Layers with `compress_ratio == 0` also get a full
//! layer reference that includes `attention_swa_ref` (ratio>0 attention is
//! still owned by the RatioNAttn sibling).
//!
//! Feeding the real GPU intermediate at each layer makes this a per-layer
//! localization rather than an accumulating end-to-end drift.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_layer_bisect \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!      --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!      --rows 128
//! ```
//!
//! Must run on gfx942 (mi300x).

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
    attention_swa_ref, expert_swiglu_ref, gate_hash_ref, gate_ref, hc_post_ref, hc_pre_ref,
    rms_norm_ref, AttnSwARefWeights, RoutingResult,
};
use hipfire_arch_deepseek4::parent::moe::{
    group_tokens_by_expert, PARENT_MOE_INTER, PARENT_ROUTE_SCALE, PARENT_SWIGLU_LIMIT,
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
/// Position buckets matching `/root/plog_pos_scan.py` (clamped to available rows).
const BUCKETS: &[(usize, usize)] = &[(0, 1), (1, 32), (32, 64), (64, 128)];
/// Error floor consistent with ratio-0 attention oracle (~1e-6).
const CLEAN_MAX_ABS: f64 = 5e-5;
const CLEAN_L2_REL: f64 = 1e-5;

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
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }

    let mut token_ids = read_token_ids(&args.token_ids)?;
    if token_ids.is_empty() {
        return Err("deepseek4 parent: token-ids file is empty".into());
    }
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

    println!("=== ds4_parent_layer_bisect ===");
    println!("model: {}", model_path.display());
    println!("token_ids: {} (n={rows})", args.token_ids.display());
    println!("start_pos: {start_pos}");
    println!("scope: FFN-half for all layers; full-layer (attn+ffn) for compress_ratio==0");
    println!();

    let wall0 = Instant::now();

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
    if cfg.num_hidden_layers != 43 {
        return Err(format!(
            "deepseek4 parent: expected 43 layers, got {}",
            cfg.num_hidden_layers
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..cfg.num_hidden_layers,
        load_experts: true,
    };
    println!(
        "load plan: layers={:?} experts=true  (expect ~150.8 GiB)",
        plan.layers
    );
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    println!(
        "loaded layers={:?} experts={} in {load_s:.3} s  resident={:.3} GiB",
        weights.layer_range,
        weights.experts_loaded,
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, rows)?;
    let hc_a = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let mut kv_rings = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let ring = zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )
        .map_err(|e| format!("kv_ring[{i}]: {e}"))?;
        kv_rings.push(ring);
    }

    // Embed → HC residual.
    parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc_a)?;

    // Cache HC weights / norms once (host) for every layer.
    let mut host_layers = Vec::with_capacity(cfg.num_hidden_layers);
    let cache_t0 = Instant::now();
    for layer in &weights.layers {
        host_layers.push(HostLayerWeights::download(&gpu, layer, &cfg)?);
    }
    println!(
        "host weight cache: {} layers in {:.2} s",
        host_layers.len(),
        cache_t0.elapsed().as_secs_f64()
    );

    // Decode scratch for MoE oracle (BF16 weight tile).
    let w_decode = gpu
        .alloc_tensor(&[PARENT_MOE_INTER, PARENT_DIM], DType::BF16)
        .map_err(|e| format!("deepseek4 parent: w_decode alloc: {e:?}"))?;

    println!();
    println!(
        "{:>5} {:>5} {:>10} {:>12} {:>12} {:>12}  buckets(max_abs)",
        "L", "ratio", "scope", "max_abs", "mean_rel", "l2_rel"
    );
    println!("{}", "-".repeat(110));

    let mut first_divergent: Option<Divergent> = None;
    let mut rows_out: Vec<LayerReport> = Vec::with_capacity(cfg.num_hidden_layers);
    let mut use_a_as_input = true;
    let fwd_t0 = Instant::now();

    for layer_i in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_i];
        let hl = &host_layers[layer_i];
        let ratio = layer.compress_ratio;
        let (x, out) = if use_a_as_input {
            (&hc_a, &hc_b)
        } else {
            (&hc_b, &hc_a)
        };
        let kv_ring = &kv_rings[layer_i];
        let input_ids = if layer_i < cfg.num_hash_layers {
            Some(token_ids.as_slice())
        } else {
            None
        };

        // Capture GPU input HC.
        let x_host = download_f32(&gpu, x, rows * PARENT_HC_DIM)?;

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
            kv_ring,
            out,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("deepseek4 parent: sync layer {layer_i}: {e:?}"))?;

        let out_gpu = download_f32(&gpu, out, rows * PARENT_HC_DIM)?;
        let residual_hc = download_f32(&gpu, scratch.residual_hc(), rows * PARENT_HC_DIM)?;
        let moe_gpu = download_f32(&gpu, scratch.stream_block(), rows * PARENT_DIM)?;
        let post_gpu = download_f32(&gpu, scratch.post(), rows * PARENT_HC_MULT)?;
        let comb_gpu = download_f32(
            &gpu,
            scratch.comb(),
            rows * PARENT_HC_MULT * PARENT_HC_MULT,
        )?;
        let ffn_y_gpu = download_f32(&gpu, scratch.stream_y(), rows * PARENT_DIM)?;
        let ffn_norm_gpu = download_f32(&gpu, scratch.stream_normed(), rows * PARENT_DIM)?;

        // ── FFN-half oracle from GPU residual_hc ────────────────────────
        let (y_ref, post_ref, comb_ref) = hc_pre_ref(
            &residual_hc,
            &hl.hc_ffn_fn,
            &hl.hc_ffn_scale,
            &hl.hc_ffn_base,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
            PARENT_RMS_EPS as f64,
            PARENT_HC_SINKHORN_ITERS as usize,
            PARENT_HC_EPS as f64,
        )?;
        let ffn_norm_ref =
            rms_norm_ref(&y_ref, &hl.ffn_norm, PARENT_RMS_EPS as f64, PARENT_DIM);
        let routing = route_ref(
            &ffn_norm_ref,
            hl,
            &cfg,
            layer_i,
            rows,
            input_ids,
        )?;
        let moe_ref = moe_ref_host(
            &mut gpu,
            layer,
            hl,
            &ffn_norm_ref,
            &routing,
            rows,
            &w_decode,
        )?;
        let out_ffn_ref = hc_post_ref(
            &moe_ref,
            &residual_hc,
            &post_ref,
            &comb_ref,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
        );

        let ffn_metrics = metrics(&out_gpu, &out_ffn_ref);
        let ffn_buckets = bucket_metrics(&out_gpu, &out_ffn_ref, rows, PARENT_HC_DIM);

        // Intermediate FFN checks (diagnostic).
        let y_m = metrics(&ffn_y_gpu, &y_ref);
        let post_m = metrics(&post_gpu, &post_ref);
        let comb_m = metrics(&comb_gpu, &comb_ref);
        let norm_m = metrics(&ffn_norm_gpu, &ffn_norm_ref);
        let moe_m = metrics(&moe_gpu, &moe_ref);

        // ── Full-layer oracle for ratio-0 ───────────────────────────────
        let (full_metrics, full_buckets, scope) = if ratio == 0 {
            let out_full = full_layer_ref_ratio0(
                &x_host,
                hl,
                rows,
                start_pos,
                layer_i,
                &cfg,
                input_ids,
                &mut gpu,
                layer,
                &w_decode,
            )?;
            let m = metrics(&out_gpu, &out_full);
            let b = bucket_metrics(&out_gpu, &out_full, rows, PARENT_HC_DIM);
            (Some(m), Some(b), "full")
        } else {
            (None, None, "ffn")
        };

        let (rep_max, rep_mean, rep_l2, rep_buckets) = if let Some(m) = full_metrics {
            (m.0, m.1, m.2, full_buckets.unwrap())
        } else {
            (ffn_metrics.0, ffn_metrics.1, ffn_metrics.2, ffn_buckets.clone())
        };

        let dirty = rep_max > CLEAN_MAX_ABS || rep_l2 > CLEAN_L2_REL;
        if dirty && first_divergent.is_none() {
            first_divergent = Some(Divergent {
                layer: layer_i,
                ratio,
                scope: scope.to_owned(),
                max_abs: rep_max,
                mean_rel: rep_mean,
                l2_rel: rep_l2,
            });
        }

        let bucket_s = format_buckets(&rep_buckets);
        println!(
            "{layer_i:>5} {ratio:>5} {scope:>10} {rep_max:>12.4e} {rep_mean:>12.4e} {rep_l2:>12.4e}  {bucket_s}"
        );
        // Always print FFN-half line when full was primary, for completeness.
        if full_metrics.is_some() {
            let fb = format_buckets(&ffn_buckets);
            println!(
                "{:>5} {:>5} {:>10} {:>12.4e} {:>12.4e} {:>12.4e}  {fb}",
                "",
                "",
                "ffn",
                ffn_metrics.0,
                ffn_metrics.1,
                ffn_metrics.2
            );
        }
        // Stage split when FFN is dirty — pin the sub-block.
        if ffn_metrics.0 > CLEAN_MAX_ABS || ffn_metrics.2 > CLEAN_L2_REL {
            println!(
                "      stages: hc_pre.y={:.3e} post={:.3e} comb={:.3e} ffn_norm={:.3e} moe={:.3e}",
                y_m.0, post_m.0, comb_m.0, norm_m.0, moe_m.0
            );
        }

        rows_out.push(LayerReport {
            layer: layer_i,
            ratio,
            scope: scope.to_owned(),
            max_abs: rep_max,
            mean_rel: rep_mean,
            l2_rel: rep_l2,
            buckets: rep_buckets,
            ffn_max_abs: ffn_metrics.0,
            ffn_mean_rel: ffn_metrics.1,
            ffn_l2_rel: ffn_metrics.2,
            ffn_buckets,
            stage_max_abs: [y_m.0, post_m.0, comb_m.0, norm_m.0, moe_m.0],
        });

        use_a_as_input = !use_a_as_input;

        // Early signal to sibling as soon as we have a first hit.
        if let Some(d) = first_divergent.as_ref() {
            if d.layer == layer_i {
                println!(
                    ">> FIRST DIVERGENT layer={} ratio={} scope={} max_abs={:.4e} l2_rel={:.4e}",
                    d.layer, d.ratio, d.scope, d.max_abs, d.l2_rel
                );
            }
        }
    }

    let fwd_s = fwd_t0.elapsed().as_secs_f64();
    let wall_s = wall0.elapsed().as_secs_f64();

    // Free decode tile + rings.
    let _ = gpu.free_tensor(w_decode);
    for r in kv_rings {
        let _ = gpu.free_tensor(r);
    }
    let _ = gpu.free_tensor(hc_a);
    let _ = gpu.free_tensor(hc_b);

    println!();
    println!("=== summary ===");
    println!("forward+oracle wall (post-load): {fwd_s:.2} s");
    println!("total wall (incl load):          {wall_s:.2} s");
    println!("load wall:                       {load_s:.2} s");
    println!("rows: {rows}");
    println!();
    println!("Per-layer table (primary scope):");
    println!(
        "{:>5} {:>5} {:>10} {:>12} {:>12} {:>12}  {}",
        "L", "ratio", "scope", "max_abs", "mean_rel", "l2_rel", "buckets max_abs"
    );
    for r in &rows_out {
        println!(
            "{:>5} {:>5} {:>10} {:>12.4e} {:>12.4e} {:>12.4e}  {}",
            r.layer,
            r.ratio,
            r.scope,
            r.max_abs,
            r.mean_rel,
            r.l2_rel,
            format_buckets(&r.buckets)
        );
    }
    println!();
    println!("FFN-half only (all layers, from GPU residual_hc):");
    println!(
        "{:>5} {:>5} {:>12} {:>12} {:>12}  {}",
        "L", "ratio", "max_abs", "mean_rel", "l2_rel", "buckets max_abs"
    );
    for r in &rows_out {
        println!(
            "{:>5} {:>5} {:>12.4e} {:>12.4e} {:>12.4e}  {}",
            r.layer,
            r.ratio,
            r.ffn_max_abs,
            r.ffn_mean_rel,
            r.ffn_l2_rel,
            format_buckets(&r.ffn_buckets)
        );
    }

    match &first_divergent {
        Some(d) => {
            println!();
            println!(
                "FIRST DIVERGING LAYER: L{} compress_ratio={} scope={} \
                 max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e}",
                d.layer, d.ratio, d.scope, d.max_abs, d.mean_rel, d.l2_rel
            );
            if let Some(rep) = rows_out.iter().find(|r| r.layer == d.layer) {
                println!(
                    "  FFN stages max_abs: hc_pre.y={:.3e} post={:.3e} comb={:.3e} \
                     ffn_norm={:.3e} moe={:.3e}",
                    rep.stage_max_abs[0],
                    rep.stage_max_abs[1],
                    rep.stage_max_abs[2],
                    rep.stage_max_abs[3],
                    rep.stage_max_abs[4]
                );
            }
        }
        None => {
            println!();
            println!(
                "NO LAYER DIVERGES above floor (max_abs<{CLEAN_MAX_ABS:.1e}, \
                 l2_rel<{CLEAN_L2_REL:.1e}). Defect is outside any single layer \
                 (inter-layer plumbing / KV rings / head / embed) OR in the \
                 ratio>0 attention half (not covered by this FFN-scoped bisect)."
            );
            let any_ratio0_full = rows_out.iter().any(|r| r.scope == "full" && r.max_abs < CLEAN_MAX_ABS);
            let ffn_all_clean = rows_out
                .iter()
                .all(|r| r.ffn_max_abs < CLEAN_MAX_ABS && r.ffn_l2_rel < CLEAN_L2_REL);
            if ffn_all_clean && any_ratio0_full {
                println!(
                    "  FFN-half clean on all 43 layers; ratio-0 full layer clean. \
                     Points at ratio>0 attention path (layers 2-42)."
                );
            }
        }
    }

    // Machine-readable one-liner for hub.
    if let Some(d) = &first_divergent {
        println!(
            "RESULT first_divergent_layer={} ratio={} scope={} max_abs={:.6e} l2_rel={:.6e} wall_s={wall_s:.2}",
            d.layer, d.ratio, d.scope, d.max_abs, d.l2_rel
        );
    } else {
        println!("RESULT first_divergent_layer=none wall_s={wall_s:.2}");
    }

    Ok(())
}

// ── Full ratio-0 layer reference ────────────────────────────────────────────

fn full_layer_ref_ratio0(
    x_hc: &[f32],
    hl: &HostLayerWeights,
    rows: usize,
    start_pos: usize,
    layer_idx: usize,
    cfg: &ParentQuantConfig,
    input_ids: Option<&[u32]>,
    gpu: &mut Gpu,
    layer: &ParentLayerWeights,
    w_decode: &GpuTensor,
) -> Result<Vec<f32>, String> {
    // Attn half.
    let (y, post, comb) = hc_pre_ref(
        x_hc,
        &hl.hc_attn_fn,
        &hl.hc_attn_scale,
        &hl.hc_attn_base,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
        PARENT_RMS_EPS as f64,
        PARENT_HC_SINKHORN_ITERS as usize,
        PARENT_HC_EPS as f64,
    )?;
    let attn_in = rms_norm_ref(&y, &hl.attn_norm, PARENT_RMS_EPS as f64, PARENT_DIM);
    let aw = AttnSwARefWeights {
        wq_a: &hl.wq_a,
        wq_b: &hl.wq_b,
        wkv: &hl.wkv,
        wo_a: &hl.wo_a,
        wo_b: &hl.wo_b,
        q_norm: &hl.q_norm,
        kv_norm: &hl.kv_norm,
        attn_sink: &hl.attn_sink,
    };
    let attn = attention_swa_ref(&attn_in, &aw, rows, start_pos, 0)?;
    let residual_hc = hc_post_ref(
        &attn.o,
        x_hc,
        &post,
        &comb,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
    );

    // FFN half.
    let (y2, post2, comb2) = hc_pre_ref(
        &residual_hc,
        &hl.hc_ffn_fn,
        &hl.hc_ffn_scale,
        &hl.hc_ffn_base,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
        PARENT_RMS_EPS as f64,
        PARENT_HC_SINKHORN_ITERS as usize,
        PARENT_HC_EPS as f64,
    )?;
    let ffn_in = rms_norm_ref(&y2, &hl.ffn_norm, PARENT_RMS_EPS as f64, PARENT_DIM);
    let routing = route_ref(&ffn_in, hl, cfg, layer_idx, rows, input_ids)?;
    let moe = moe_ref_host(gpu, layer, hl, &ffn_in, &routing, rows, w_decode)?;
    Ok(hc_post_ref(
        &moe,
        &residual_hc,
        &post2,
        &comb2,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
    ))
}

// ── MoE host oracle (decode selected experts via GPU, matmul in f64) ────────

fn moe_ref_host(
    gpu: &mut Gpu,
    layer: &ParentLayerWeights,
    hl: &HostLayerWeights,
    x_f32: &[f32],
    routing: &RoutingResult,
    rows: usize,
    w_decode: &GpuTensor,
) -> Result<Vec<f32>, String> {
    let dim = PARENT_DIM;
    let inter = PARENT_MOE_INTER;
    let topk = routing.indices.len() / rows;
    // Build ParentRouting-shaped grouping via local indices/weights.
    let mut pr_indices = vec![0u32; rows * topk];
    let mut pr_weights = vec![0.0f32; rows * topk];
    for i in 0..rows * topk {
        pr_indices[i] = routing.indices[i];
        pr_weights[i] = routing.weights[i];
    }
    // group_tokens_by_expert wants ParentRouting — reimplement grouping inline.
    let n_experts = layer.experts.len();
    let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    for r in 0..rows {
        for t in 0..topk {
            let eid = pr_indices[r * topk + t] as usize;
            let w = pr_weights[r * topk + t];
            if eid >= n_experts {
                return Err(format!(
                    "deepseek4 parent: moe_ref expert id {eid} out of range ({n_experts})"
                ));
            }
            groups[eid].push((r, w));
        }
    }
    let _ = group_tokens_by_expert; // keep import intentional for parity docs

    let mut y = vec![0.0f32; rows * dim];

    for (eid, members) in groups.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let n_tok = members.len();
        let expert = &layer.experts[eid];
        // Gather x rows.
        let mut xg = vec![0.0f32; n_tok * dim];
        let mut rw = vec![0.0f32; n_tok];
        for (i, &(row, w)) in members.iter().enumerate() {
            xg[i * dim..(i + 1) * dim].copy_from_slice(&x_f32[row * dim..(row + 1) * dim]);
            rw[i] = w;
        }
        // w1
        expert
            .w1
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("moe_ref w1 decode eid={eid}: {e}"))?;
        let w1 = download_bf16_as_f32(gpu, w_decode, inter * dim)?;
        let gate = dense_linear_bf16_host(&xg, &w1, n_tok, inter, dim)?;
        // w3
        expert
            .w3
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("moe_ref w3 decode eid={eid}: {e}"))?;
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
        // w2 — input K = inter, N = dim
        // decode tile is [inter, dim] BF16 = inter*dim elems; w2 is [dim, inter].
        expert
            .w2
            .decode_into(gpu, w_decode)
            .map_err(|e| format!("moe_ref w2 decode eid={eid}: {e}"))?;
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

    // Shared expert (no route weight).
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

fn dense_linear_bf16_host(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * k {
        return Err(format!(
            "dense_linear_bf16_host: x len {} != rows*k {}",
            x.len(),
            rows * k
        ));
    }
    if w.len() != n * k {
        return Err(format!(
            "dense_linear_bf16_host: w len {} != n*k {}",
            w.len(),
            n * k
        ));
    }
    // Round x to BF16 lattice then act-quant (matches GPU linear boundary).
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

fn route_ref(
    x: &[f32],
    hl: &HostLayerWeights,
    cfg: &ParentQuantConfig,
    layer_idx: usize,
    rows: usize,
    input_ids: Option<&[u32]>,
) -> Result<RoutingResult, String> {
    let dim = PARENT_DIM;
    let n_experts = cfg.n_routed_experts;
    let topk = cfg.num_experts_per_tok;
    let is_hash = layer_idx < cfg.num_hash_layers;
    if is_hash {
        let ids = input_ids.ok_or_else(|| {
            format!("deepseek4 parent: hash layer {layer_idx} needs input_ids")
        })?;
        let tid2eid = hl.tid2eid.as_ref().ok_or_else(|| {
            format!("deepseek4 parent: hash layer {layer_idx} missing tid2eid")
        })?;
        // Indices from hash table; weights from uncorrected scores (same as parent_route).
        let hash = gate_hash_ref(ids, tid2eid, n_experts, topk)?;
        // Score path for weights: gate_ref gives score-topk; we need gather-by-hash-idx.
        // Replicate parent_route: scores = sqrtsoftplus(x @ W^T), gather at hash indices,
        // L1-norm * route_scale.
        let full = gate_ref(
            x,
            &hl.gate_weight,
            None, // bias unused for weight values on hash path
            rows,
            dim,
            n_experts,
            topk,
            PARENT_ROUTE_SCALE as f64,
            true,
        )?;
        let _ = full; // scores path below
        // Direct score gather matching parent_route / hash_route_weights.
        let mut scores = vec![0.0f32; rows * n_experts];
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            for e in 0..n_experts {
                let wr = &hl.gate_weight[e * dim..(e + 1) * dim];
                let mut acc = 0.0f64;
                for k in 0..dim {
                    acc += xr[k] as f64 * wr[k] as f64;
                }
                // sqrtsoftplus
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
                let eid = hash.indices[r * topk + t] as usize;
                let s = scores[r * n_experts + eid];
                weights[r * topk + t] = s;
                sum += s;
            }
            if sum > 0.0 {
                for t in 0..topk {
                    weights[r * topk + t] =
                        weights[r * topk + t] / sum * PARENT_ROUTE_SCALE;
                }
            }
        }
        Ok(RoutingResult {
            weights,
            indices: hash.indices,
        })
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

// ── Host weight cache ───────────────────────────────────────────────────────

struct HostLayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    attn_sink: Vec<f32>,
    wq_a: Vec<f32>,
    wq_b: Vec<f32>,
    wkv: Vec<f32>,
    wo_a: Vec<f32>,
    wo_b: Vec<f32>,
    hc_attn_fn: Vec<f32>,
    hc_attn_base: Vec<f32>,
    hc_attn_scale: Vec<f32>,
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

impl HostLayerWeights {
    fn download(
        gpu: &Gpu,
        layer: &ParentLayerWeights,
        cfg: &ParentQuantConfig,
    ) -> Result<Self, String> {
        let dim = PARENT_DIM;
        let inter = PARENT_MOE_INTER;
        let mix_hc = (2 + PARENT_HC_MULT) * PARENT_HC_MULT;
        let hc_flat = PARENT_HC_DIM;

        // Dense BF16 attention projections — shapes from ParentDenseWeight.
        let wq_a_n = layer.wq_a.n();
        let wq_a_k = layer.wq_a.k();
        let wq_b_n = layer.wq_b.n();
        let wq_b_k = layer.wq_b.k();
        let wkv_n = layer.wkv.n();
        let wkv_k = layer.wkv.k();
        let wo_a_n = layer.wo_a.n();
        let wo_a_k = layer.wo_a.k();
        let wo_b_n = layer.wo_b.n();
        let wo_b_k = layer.wo_b.k();

        let gate_bias = if let Some(b) = layer.gate_bias.as_ref() {
            Some(download_f32(gpu, b, cfg.n_routed_experts)?)
        } else {
            None
        };
        let tid2eid = if let Some(t) = layer.tid2eid.as_ref() {
            Some(download_i64(gpu, t, VOCAB * cfg.num_experts_per_tok)?)
        } else {
            None
        };

        Ok(Self {
            attn_norm: download_bf16_as_f32(gpu, &layer.attn_norm, dim)?,
            ffn_norm: download_bf16_as_f32(gpu, &layer.ffn_norm, dim)?,
            q_norm: download_bf16_as_f32(gpu, &layer.q_norm, layer.q_norm.numel())?,
            kv_norm: download_bf16_as_f32(gpu, &layer.kv_norm, layer.kv_norm.numel())?,
            attn_sink: download_f32(gpu, &layer.attn_sink, layer.attn_sink.numel())?,
            wq_a: download_bf16_as_f32(gpu, layer.wq_a.tensor(), wq_a_n * wq_a_k)?,
            wq_b: download_bf16_as_f32(gpu, layer.wq_b.tensor(), wq_b_n * wq_b_k)?,
            wkv: download_bf16_as_f32(gpu, layer.wkv.tensor(), wkv_n * wkv_k)?,
            wo_a: download_bf16_as_f32(gpu, layer.wo_a.tensor(), wo_a_n * wo_a_k)?,
            wo_b: download_bf16_as_f32(gpu, layer.wo_b.tensor(), wo_b_n * wo_b_k)?,
            hc_attn_fn: download_f32(gpu, &layer.hc_attn_fn, mix_hc * hc_flat)?,
            hc_attn_base: download_f32(gpu, &layer.hc_attn_base, mix_hc)?,
            hc_attn_scale: download_f32(gpu, &layer.hc_attn_scale, 3)?,
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
            shared_w1: download_bf16_as_f32(
                gpu,
                layer.shared_w1.tensor(),
                inter * dim,
            )?,
            shared_w2: download_bf16_as_f32(
                gpu,
                layer.shared_w2.tensor(),
                dim * inter,
            )?,
            shared_w3: download_bf16_as_f32(
                gpu,
                layer.shared_w3.tensor(),
                inter * dim,
            )?,
        })
    }
}

// ── Metrics / buckets ───────────────────────────────────────────────────────

struct LayerReport {
    layer: usize,
    ratio: usize,
    scope: String,
    max_abs: f64,
    mean_rel: f64,
    l2_rel: f64,
    buckets: Vec<(usize, usize, f64, f64, f64)>,
    ffn_max_abs: f64,
    ffn_mean_rel: f64,
    ffn_l2_rel: f64,
    ffn_buckets: Vec<(usize, usize, f64, f64, f64)>,
    stage_max_abs: [f64; 5],
}

struct Divergent {
    layer: usize,
    ratio: usize,
    scope: String,
    max_abs: f64,
    mean_rel: f64,
    l2_rel: f64,
}

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len(), "metrics length mismatch");
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

fn bucket_metrics(
    a: &[f32],
    b: &[f32],
    rows: usize,
    width: usize,
) -> Vec<(usize, usize, f64, f64, f64)> {
    let mut out = Vec::new();
    for &(lo, hi) in BUCKETS {
        let lo = lo.min(rows);
        let hi = hi.min(rows);
        if lo >= hi {
            continue;
        }
        let mut max_abs = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut n_rel = 0usize;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in lo..hi {
            let aa = &a[r * width..(r + 1) * width];
            let bb = &b[r * width..(r + 1) * width];
            for (&x, &y) in aa.iter().zip(bb.iter()) {
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
        out.push((lo, hi, max_abs, mean_rel, l2_rel));
    }
    out
}

fn format_buckets(b: &[(usize, usize, f64, f64, f64)]) -> String {
    b.iter()
        .map(|(lo, hi, mx, _, l2)| format!("[{lo},{hi})={mx:.2e}/{l2:.2e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── IO helpers ──────────────────────────────────────────────────────────────

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems
        .checked_mul(4)
        .ok_or_else(|| "deepseek4 parent: f32 download size overflow".to_owned())?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: f32 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: f32 download: {e:?}"))?;
    Ok(data)
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems
        .checked_mul(2)
        .ok_or_else(|| "deepseek4 parent: bf16 download size overflow".to_owned())?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: bf16 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
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

fn download_i64(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<i64>, String> {
    let nbytes = nelems
        .checked_mul(8)
        .ok_or_else(|| "deepseek4 parent: i64 download size overflow".to_owned())?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: i64 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut raw = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &t.buf)
        .map_err(|e| format!("deepseek4 parent: i64 download: {e:?}"))?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let mut le = [0u8; 8];
        le.copy_from_slice(&raw[i * 8..i * 8 + 8]);
        out.push(i64::from_le_bytes(le));
    }
    Ok(out)
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: zeros_f32: {e:?}"))
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "deepseek4 parent: read token-ids {}: {e}",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "deepseek4 parent: token-ids {} size {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut le = [0u8; 4];
        le.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        out.push(u32::from_le_bytes(le));
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
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args
                    .get(i + 1)
                    .ok_or("--model needs a value")?
                    .clone();
                i += 2;
            }
            "--token-ids" => {
                token_ids = PathBuf::from(args.get(i + 1).ok_or("--token-ids needs a value")?);
                i += 2;
            }
            "--rows" => {
                rows = args
                    .get(i + 1)
                    .ok_or("--rows needs a value")?
                    .parse()
                    .map_err(|e| format!("--rows: {e}"))?;
                i += 2;
            }
            s if !s.starts_with('-') => {
                model = s.to_owned();
                i += 1;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if rows == 0 {
        return Err("--rows must be > 0".into());
    }
    Ok(Args {
        model,
        token_ids,
        rows,
    })
}
