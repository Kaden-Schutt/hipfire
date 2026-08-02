// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Hypothesis-free per-layer f64 bisect of the DS4 parent forward.
//!
//! Runs a 128-token parent forward and, for each layer, feeds that layer's
//! *actual GPU residual* into composed f64 references:
//!   - FFN-half from GPU `residual_hc` (post-attn), MoE on BF16-staged `moe_x`
//!   - Attn-half residual: `hc_pre` → BF16-round(`attn_norm`) → joint
//!     `attention_swa_ref` (SWA + compressor + indexer when ratio>0) → `hc_post`,
//!     compared to GPU `residual_hc`
//!   - Full-layer composition of the two
//!
//! The attention oracle MUST carry compressor/indexer weights on ratio>0
//! layers. Building `AttnSwARefWeights { compressor: None, indexer: None }`
//! against a GPU that attends the joint window+compressed key set was artifact
//! #6 of this investigation (fake "23x ratio-4 vs ratio-128 gap").
//!
//! Domain rule (bitten many times): GPU linears stage F32→BF16 before GEMM.
//! Host oracles must consume the same BF16 lattice, not f32 rms_norm tails.
//!
//! Floor calibration: layer 0 (ratio 0, no compressor/indexer) must land at
//! ~1e-6 before any ratio-4/128 number is interpreted.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_layer_bisect \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!      --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!      --rows 128 [--max-layers N]
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
    rms_norm_ref, AttnCompRefWeights, AttnIndexerRefWeights, AttnSwARefWeights, RoutingResult,
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
/// Position buckets matching `/root/plog_pos_scan.py` (clamped to available rows).
const BUCKETS: &[(usize, usize)] = &[(0, 1), (1, 32), (32, 64), (64, 128)];
/// Error floor consistent with ratio-0 attention oracle (~1e-6 abs; ~5e-6 K=8192).
/// Absolute threshold is deliberately loose vs the true floor so residual growth
/// deep in the stack does not flag every layer; relative l2 is the primary gate.
const CLEAN_MAX_ABS: f64 = 5e-5;
const CLEAN_L2_REL: f64 = 1e-5;
/// Soft floor check for layer 0 calibration (must reproduce ~1e-6).
const LAYER0_FLOOR_MAX_ABS: f64 = 5e-5;
const LAYER0_FLOOR_L2_REL: f64 = 1e-5;

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
    println!("scope: full-layer + residual_hc(attn-half JOINT) + FFN-half; BF16 domain; abs+rel+in_amax+row_dist");

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
    let n_layers = args
        .max_layers
        .unwrap_or(cfg.num_hidden_layers)
        .min(cfg.num_hidden_layers);
    if n_layers == 0 {
        return Err("--max-layers must be > 0".into());
    }
    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..n_layers,
        load_experts: true,
    };
    println!(
        "load plan: layers={:?} experts=true  (n_layers={n_layers})",
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
    let mut kv_rings = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
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
    let mut host_layers = Vec::with_capacity(n_layers);
    let cache_t0 = Instant::now();
    for layer in weights.layers.iter().take(n_layers) {
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
        "{:>5} {:>5} {:>10} {:>12} {:>12} {:>12} {:>12}  {}",
        "L", "ratio", "scope", "max_abs", "l2_rel", "in_amax", "row_max", "row_dist"
    );
    println!("{}", "-".repeat(120));

    let mut first_divergent: Option<Divergent> = None;
    let mut rows_out: Vec<LayerReport> = Vec::with_capacity(cfg.num_hidden_layers);
    let mut use_a_as_input = true;
    let fwd_t0 = Instant::now();

    for layer_i in 0..n_layers {
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
        // HC / ffn_norm stay on the f32 residual path (matches GPU hc_pre /
        // rms). MoE must consume the *BF16-staged* activation the GPU block
        // actually saw (`moe_x_bf16`), plus the GPU's own routing — feeding
        // f32 `ffn_norm_ref` here manufactures a ~6e-3 max_abs that looks
        // like a defect but is pure input-domain mismatch (floor calib L0/L5
        // row0 assembled max_abs 3.6e-7 / 5.7e-6 with BF16 x).
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
        let moe_x_bf16 = download_bf16_as_f32(&gpu, scratch.moe_x_bf16(), rows * PARENT_DIM)?;
        let gpu_routing = parent_route(
            &mut gpu,
            backend,
            layer,
            &cfg,
            scratch.moe_x_bf16(),
            rows,
            input_ids,
        )?;
        let routing = RoutingResult {
            weights: gpu_routing.weights,
            indices: gpu_routing.indices,
        };
        let moe_ref = moe_ref_host(
            &mut gpu,
            layer,
            hl,
            &moe_x_bf16,
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

        let ffn_metrics = metrics(&out_gpu, &out_ffn_ref, rows, PARENT_HC_DIM);
        let ffn_buckets = bucket_metrics(&out_gpu, &out_ffn_ref, rows, PARENT_HC_DIM);

        // Intermediate FFN checks (diagnostic).
        let y_m = metrics(&ffn_y_gpu, &y_ref, rows, PARENT_DIM);
        let post_m = metrics(&post_gpu, &post_ref, rows, PARENT_HC_MULT);
        let comb_m = metrics(
            &comb_gpu,
            &comb_ref,
            rows,
            PARENT_HC_MULT * PARENT_HC_MULT,
        );
        let norm_m = metrics(&ffn_norm_gpu, &ffn_norm_ref, rows, PARENT_DIM);
        let moe_m = metrics(&moe_gpu, &moe_ref, rows, PARENT_DIM);
        // ── Attn-half residual oracle (GPU residual_hc is post-attn) ─────
        // Same domain trap as MoE: GPU stages F32→BF16 before every linear.
        // `dense_linear_bf16_ref` BF16-rounds inside act_quant, but we still
        // explicitly BF16-round the attn input to match AttnOracle / GPU
        // `stage_f32_to_act_bf16` and kill any non-quant path divergence.
        let (attn_y, attn_post, attn_comb) = hc_pre_ref(
            &x_host,
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
        let attn_norm_f32 =
            rms_norm_ref(&attn_y, &hl.attn_norm, PARENT_RMS_EPS as f64, PARENT_DIM);
        let attn_in_bf16: Vec<f32> = attn_norm_f32.iter().copied().map(round_to_bf16).collect();
        let aw = hl.attn_ref_weights();
        // Sanity: ratio>0 must have compressor; ratio==4 must have indexer.
        if ratio > 0 && aw.compressor.is_none() {
            return Err(format!(
                "layer {layer_i} ratio={ratio}: missing main compressor weights in host cache"
            ));
        }
        if ratio == 4 && aw.indexer.is_none() {
            return Err(format!(
                "layer {layer_i} ratio=4: missing indexer weights in host cache"
            ));
        }
        let attn_ref = attention_swa_ref(&attn_in_bf16, &aw, rows, start_pos, ratio)?;
        let residual_hc_ref = hc_post_ref(
            &attn_ref.o,
            &x_host,
            &attn_post,
            &attn_comb,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
        );
        let rhc_metrics = metrics(&residual_hc, &residual_hc_ref, rows, PARENT_HC_DIM);
        let rhc_buckets = bucket_metrics(&residual_hc, &residual_hc_ref, rows, PARENT_HC_DIM);

        // ── Full-layer oracle (BF16 domain on attn + MoE inputs) ─────────
        let out_full = full_layer_ref(
            &x_host,
            hl,
            rows,
            start_pos,
            layer_i,
            ratio,
            &cfg,
            input_ids,
            &mut gpu,
            layer,
            &w_decode,
        )?;
        let full_metrics = metrics(&out_gpu, &out_full, rows, PARENT_HC_DIM);
        let full_buckets = bucket_metrics(&out_gpu, &out_full, rows, PARENT_HC_DIM);
        let scope = "full";

        // Layer-0 floor calibration — must land ~1e-6 before interpreting anything else.
        if layer_i == 0 {
            println!();
            println!("=== LAYER 0 FLOOR CALIBRATION (ratio=0, no compressor/indexer) ===");
            println!(
                "  full:   max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
                full_metrics.max_abs,
                full_metrics.l2_rel,
                full_metrics.in_amax,
                format_row_dist(&full_metrics.row_max_abs)
            );
            println!(
                "  res_hc: max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
                rhc_metrics.max_abs,
                rhc_metrics.l2_rel,
                rhc_metrics.in_amax,
                format_row_dist(&rhc_metrics.row_max_abs)
            );
            println!(
                "  ffn:    max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
                ffn_metrics.max_abs,
                ffn_metrics.l2_rel,
                ffn_metrics.in_amax,
                format_row_dist(&ffn_metrics.row_max_abs)
            );
            let l0_ok = full_metrics.max_abs <= LAYER0_FLOOR_MAX_ABS
                && full_metrics.l2_rel <= LAYER0_FLOOR_L2_REL
                && rhc_metrics.max_abs <= LAYER0_FLOOR_MAX_ABS
                && rhc_metrics.l2_rel <= LAYER0_FLOOR_L2_REL;
            if l0_ok {
                println!(
                    "  FLOOR OK: layer 0 reproduces ~1e-6 class floor under joint oracle. \
                     Proceeding to interpret ratio-4/128 layers."
                );
            } else {
                println!(
                    "  FLOOR FAIL: layer 0 does NOT reproduce the known ~1e-6 floor. \
                     Harness is still wrong — DO NOT interpret downstream numbers."
                );
            }
            println!();
        }

        let dirty = full_metrics.max_abs > CLEAN_MAX_ABS || full_metrics.l2_rel > CLEAN_L2_REL;
        // Prefer relative gate once residual has grown (in_amax >> 1).
        let dirty_rel = full_metrics.l2_rel > CLEAN_L2_REL
            || (full_metrics.in_amax > 0.0
                && full_metrics.max_abs / full_metrics.in_amax > CLEAN_L2_REL * 10.0
                && full_metrics.max_abs > CLEAN_MAX_ABS);
        if dirty_rel && first_divergent.is_none() {
            first_divergent = Some(Divergent {
                layer: layer_i,
                ratio,
                scope: scope.to_owned(),
                max_abs: full_metrics.max_abs,
                mean_rel: full_metrics.mean_rel,
                l2_rel: full_metrics.l2_rel,
                in_amax: full_metrics.in_amax,
            });
        }
        let _ = dirty; // kept for future absolute-only diagnostics

        let full_row_max = full_metrics
            .row_max_abs
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        println!(
            "{layer_i:>5} {ratio:>5} {scope:>10} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}  {}",
            full_metrics.max_abs,
            full_metrics.l2_rel,
            full_metrics.in_amax,
            full_row_max,
            format_row_dist(&full_metrics.row_max_abs)
        );
        {
            let rmax = rhc_metrics
                .row_max_abs
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
            println!(
                "{:>5} {:>5} {:>10} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}  {}",
                "",
                "",
                "res_hc",
                rhc_metrics.max_abs,
                rhc_metrics.l2_rel,
                rhc_metrics.in_amax,
                rmax,
                format_row_dist(&rhc_metrics.row_max_abs)
            );
            println!(
                "      res_hc buckets: {}",
                format_buckets(&rhc_buckets)
            );
        }
        {
            let fmax = ffn_metrics
                .row_max_abs
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
            println!(
                "{:>5} {:>5} {:>10} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}  {}",
                "",
                "",
                "ffn",
                ffn_metrics.max_abs,
                ffn_metrics.l2_rel,
                ffn_metrics.in_amax,
                fmax,
                format_row_dist(&ffn_metrics.row_max_abs)
            );
            println!(
                "      ffn buckets: {}",
                format_buckets(&ffn_buckets)
            );
        }
        // Stage split when FFN or residual_hc is dirty.
        if ffn_metrics.max_abs > CLEAN_MAX_ABS
            || ffn_metrics.l2_rel > CLEAN_L2_REL
            || rhc_metrics.max_abs > CLEAN_MAX_ABS
            || rhc_metrics.l2_rel > CLEAN_L2_REL
        {
            println!(
                "      stages: hc_pre.y={:.3e} post={:.3e} comb={:.3e} ffn_norm={:.3e} moe={:.3e} res_hc={:.3e}",
                y_m.max_abs,
                post_m.max_abs,
                comb_m.max_abs,
                norm_m.max_abs,
                moe_m.max_abs,
                rhc_metrics.max_abs
            );
        }

        rows_out.push(LayerReport {
            layer: layer_i,
            ratio,
            scope: scope.to_owned(),
            max_abs: full_metrics.max_abs,
            mean_rel: full_metrics.mean_rel,
            l2_rel: full_metrics.l2_rel,
            in_amax: full_metrics.in_amax,
            gpu_amax: full_metrics.gpu_amax,
            row_max_abs: full_metrics.row_max_abs,
            buckets: full_buckets,
            rhc_max_abs: rhc_metrics.max_abs,
            rhc_mean_rel: rhc_metrics.mean_rel,
            rhc_l2_rel: rhc_metrics.l2_rel,
            rhc_in_amax: rhc_metrics.in_amax,
            rhc_row_max_abs: rhc_metrics.row_max_abs,
            ffn_max_abs: ffn_metrics.max_abs,
            ffn_mean_rel: ffn_metrics.mean_rel,
            ffn_l2_rel: ffn_metrics.l2_rel,
            ffn_in_amax: ffn_metrics.in_amax,
            ffn_row_max_abs: ffn_metrics.row_max_abs,
            ffn_buckets,
            stage_max_abs: [
                y_m.max_abs,
                post_m.max_abs,
                comb_m.max_abs,
                norm_m.max_abs,
                moe_m.max_abs,
            ],
        });

        use_a_as_input = !use_a_as_input;

        // Early signal to sibling as soon as we have a first hit.
        if let Some(d) = first_divergent.as_ref() {
            if d.layer == layer_i {
                println!(
                    ">> FIRST DIVERGENT layer={} ratio={} scope={} max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}",
                    d.layer, d.ratio, d.scope, d.max_abs, d.l2_rel, d.in_amax
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
    println!("oracle: JOINT attention_swa_ref (SWA + compressor + indexer)");
    println!();

    // Layer-0 floor first — gate every other number on this.
    if let Some(l0) = rows_out.iter().find(|r| r.layer == 0) {
        let l0_ok = l0.max_abs <= LAYER0_FLOOR_MAX_ABS
            && l0.l2_rel <= LAYER0_FLOOR_L2_REL
            && l0.rhc_max_abs <= LAYER0_FLOOR_MAX_ABS
            && l0.rhc_l2_rel <= LAYER0_FLOOR_L2_REL;
        println!("=== LAYER 0 FLOOR (must be ~1e-6 before interpreting anything) ===");
        println!(
            "  full   max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
            l0.max_abs,
            l0.l2_rel,
            l0.in_amax,
            format_row_dist(&l0.row_max_abs)
        );
        println!(
            "  res_hc max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
            l0.rhc_max_abs,
            l0.rhc_l2_rel,
            l0.rhc_in_amax,
            format_row_dist(&l0.rhc_row_max_abs)
        );
        println!(
            "  ffn    max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
            l0.ffn_max_abs,
            l0.ffn_l2_rel,
            l0.ffn_in_amax,
            format_row_dist(&l0.ffn_row_max_abs)
        );
        if l0_ok {
            println!("  VERDICT floor: OK — harness reproduces known layer-0 floor.");
        } else {
            println!(
                "  VERDICT floor: FAIL — harness still wrong; ratio-4/128 numbers are inconclusive."
            );
        }
        println!();
    }

    println!("Per-layer table (full | attn-half | ffn-half):");
    println!(
        "{:>5} {:>5} {:>12} {:>12} {:>12}  {:>12} {:>12} {:>12}  {:>12} {:>12} {:>12}  {}",
        "L",
        "ratio",
        "full_abs",
        "full_l2",
        "full_inamax",
        "attn_abs",
        "attn_l2",
        "attn_inamax",
        "ffn_abs",
        "ffn_l2",
        "ffn_inamax",
        "full_row_dist"
    );
    for r in &rows_out {
        println!(
            "{:>5} {:>5} {:>12.4e} {:>12.4e} {:>12.4e}  {:>12.4e} {:>12.4e} {:>12.4e}  {:>12.4e} {:>12.4e} {:>12.4e}  {}",
            r.layer,
            r.ratio,
            r.max_abs,
            r.l2_rel,
            r.in_amax,
            r.rhc_max_abs,
            r.rhc_l2_rel,
            r.rhc_in_amax,
            r.ffn_max_abs,
            r.ffn_l2_rel,
            r.ffn_in_amax,
            format_row_dist(&r.row_max_abs)
        );
    }

    // Explicit full per-row max dump for each layer (global max already in dist).
    println!();
    println!("Per-layer GLOBAL max abs error (full / attn / ffn) + argmax row:");
    for r in &rows_out {
        let (fmax, farg) = argmax_row(&r.row_max_abs);
        let (amax, aarg) = argmax_row(&r.rhc_row_max_abs);
        let (mmax, marg) = argmax_row(&r.ffn_row_max_abs);
        println!(
            "  L{:>2} r={:>3}: full={:.4e}@r{farg}  attn={:.4e}@r{aarg}  ffn={:.4e}@r{marg}  in_amax={:.4e}",
            r.layer, r.ratio, fmax, amax, mmax, r.in_amax
        );
    }

    match &first_divergent {
        Some(d) => {
            println!();
            println!(
                "FIRST DIVERGING LAYER: L{} compress_ratio={} scope={} \
                 max_abs={:.6e} mean_rel={:.6e} l2_rel={:.6e} in_amax={:.6e} \
                 (abs/in_amax={:.3e})",
                d.layer,
                d.ratio,
                d.scope,
                d.max_abs,
                d.mean_rel,
                d.l2_rel,
                d.in_amax,
                if d.in_amax > 0.0 {
                    d.max_abs / d.in_amax
                } else {
                    0.0
                }
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
                println!(
                    "  attn-half: max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
                    rep.rhc_max_abs,
                    rep.rhc_l2_rel,
                    rep.rhc_in_amax,
                    format_row_dist(&rep.rhc_row_max_abs)
                );
                println!(
                    "  ffn-half:  max_abs={:.4e} l2_rel={:.4e} in_amax={:.4e}  {}",
                    rep.ffn_max_abs,
                    rep.ffn_l2_rel,
                    rep.ffn_in_amax,
                    format_row_dist(&rep.ffn_row_max_abs)
                );
            }
        }
        None => {
            println!();
            println!(
                "NO LAYER DIVERGES above floor (max_abs<{CLEAN_MAX_ABS:.1e} OR \
                 l2_rel<{CLEAN_L2_REL:.1e} with abs/in_amax gate). \
                 Every layer is correct given its GPU input under the JOINT \
                 attention oracle. Combined with a clean PlumbingProbe wiring \
                 result this forces the defect into something neither bisect \
                 nor plumbing has modelled."
            );
        }
    }

    // Machine-readable one-liner for hub.
    if let Some(d) = &first_divergent {
        println!(
            "RESULT first_divergent_layer={} ratio={} scope={} max_abs={:.6e} l2_rel={:.6e} in_amax={:.6e} wall_s={wall_s:.2}",
            d.layer, d.ratio, d.scope, d.max_abs, d.l2_rel, d.in_amax
        );
    } else {
        println!("RESULT first_divergent_layer=none wall_s={wall_s:.2}");
    }

    Ok(())
}

// ── Full layer reference (joint attn SWA+compress+indexer + FFN) ────────────

fn full_layer_ref(
    x_hc: &[f32],
    hl: &HostLayerWeights,
    rows: usize,
    start_pos: usize,
    layer_idx: usize,
    compress_ratio: usize,
    cfg: &ParentQuantConfig,
    input_ids: Option<&[u32]>,
    gpu: &mut Gpu,
    layer: &ParentLayerWeights,
    w_decode: &GpuTensor,
) -> Result<Vec<f32>, String> {
    // Attn half — joint oracle (compressor/indexer wired when ratio > 0).
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
    // Match GPU stage_f32_to_act_bf16 before first attn linear.
    let attn_in_bf16: Vec<f32> = attn_in.iter().copied().map(round_to_bf16).collect();
    let aw = hl.attn_ref_weights();
    let attn = attention_swa_ref(&attn_in_bf16, &aw, rows, start_pos, compress_ratio)?;
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
    // Match GPU `stage_f32_to_bf16` before MoE (same domain as floor calib).
    let ffn_in_bf16: Vec<f32> = ffn_in.iter().copied().map(round_to_bf16).collect();
    let routing = route_ref(&ffn_in_bf16, hl, cfg, layer_idx, rows, input_ids)?;
    let moe = moe_ref_host(gpu, layer, hl, &ffn_in_bf16, &routing, rows, w_decode)?;
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
    // Group (row, weight) by expert id (same contract as group_tokens_by_expert).
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
    /// Main compressor (ratio 4 / 128). All four present or all absent.
    comp_wkv: Option<Vec<f32>>,
    comp_wgate: Option<Vec<f32>>,
    comp_norm: Option<Vec<f32>>,
    comp_ape: Option<Vec<f32>>,
    /// Indexer (ratio 4 only). All six present or all absent.
    ix_wq_b: Option<Vec<f32>>,
    ix_weights_proj: Option<Vec<f32>>,
    ix_comp_wkv: Option<Vec<f32>>,
    ix_comp_wgate: Option<Vec<f32>>,
    ix_comp_norm: Option<Vec<f32>>,
    ix_comp_ape: Option<Vec<f32>>,
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

        // Main compressor (ratio > 0).
        let (comp_wkv, comp_wgate, comp_norm, comp_ape) =
            if let Some(c) = layer.compressor.as_ref() {
                let proj = c.wkv.shape.get(0).copied().unwrap_or(0);
                let dim_k = c.wkv.shape.get(1).copied().unwrap_or(PARENT_DIM);
                (
                    Some(download_bf16_as_f32(gpu, &c.wkv, proj * dim_k)?),
                    Some(download_bf16_as_f32(gpu, &c.wgate, proj * dim_k)?),
                    Some(download_bf16_as_f32(gpu, &c.norm, PARENT_HEAD_DIM)?),
                    Some(download_f32(
                        gpu,
                        &c.ape,
                        c.ape.shape.iter().product::<usize>().max(1),
                    )?),
                )
            } else {
                (None, None, None, None)
            };

        // Indexer (ratio == 4).
        let (ix_wq_b, ix_weights_proj, ix_comp_wkv, ix_comp_wgate, ix_comp_norm, ix_comp_ape) =
            if let Some(ix) = layer.indexer.as_ref() {
                let wq = download_bf16_as_f32(gpu, ix.wq_b.tensor(), ix.wq_b.n() * ix.wq_b.k())?;
                let wp_n = ix.weights_proj.shape.get(0).copied().unwrap_or(0);
                let wp_k = ix.weights_proj.shape.get(1).copied().unwrap_or(PARENT_DIM);
                let wp = download_bf16_as_f32(gpu, &ix.weights_proj, wp_n * wp_k)?;
                let cproj = ix.compressor_wkv.shape.get(0).copied().unwrap_or(0);
                let cdim = ix.compressor_wkv.shape.get(1).copied().unwrap_or(PARENT_DIM);
                (
                    Some(wq),
                    Some(wp),
                    Some(download_bf16_as_f32(gpu, &ix.compressor_wkv, cproj * cdim)?),
                    Some(download_bf16_as_f32(gpu, &ix.compressor_wgate, cproj * cdim)?),
                    // index head_dim = 128
                    Some(download_bf16_as_f32(gpu, &ix.compressor_norm, 128)?),
                    Some(download_f32(
                        gpu,
                        &ix.compressor_ape,
                        ix.compressor_ape.shape.iter().product::<usize>().max(1),
                    )?),
                )
            } else {
                (None, None, None, None, None, None)
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
            comp_wkv,
            comp_wgate,
            comp_norm,
            comp_ape,
            ix_wq_b,
            ix_weights_proj,
            ix_comp_wkv,
            ix_comp_wgate,
            ix_comp_norm,
            ix_comp_ape,
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

    /// Build joint-oracle weight views (SWA + compressor + indexer when present).
    fn attn_ref_weights(&self) -> AttnSwARefWeights<'_> {
        let compressor = match (
            self.comp_wkv.as_deref(),
            self.comp_wgate.as_deref(),
            self.comp_norm.as_deref(),
            self.comp_ape.as_deref(),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => Some(AttnCompRefWeights {
                wkv: a,
                wgate: b,
                norm: c,
                ape: d,
            }),
            _ => None,
        };
        let indexer = match (
            self.ix_wq_b.as_deref(),
            self.ix_weights_proj.as_deref(),
            self.ix_comp_wkv.as_deref(),
            self.ix_comp_wgate.as_deref(),
            self.ix_comp_norm.as_deref(),
            self.ix_comp_ape.as_deref(),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => {
                Some(AttnIndexerRefWeights {
                    wq_b: a,
                    weights_proj: b,
                    compressor_wkv: c,
                    compressor_wgate: d,
                    compressor_norm: e,
                    compressor_ape: f,
                })
            }
            _ => None,
        };
        AttnSwARefWeights {
            wq_a: &self.wq_a,
            wq_b: &self.wq_b,
            wkv: &self.wkv,
            wo_a: &self.wo_a,
            wo_b: &self.wo_b,
            q_norm: &self.q_norm,
            kv_norm: &self.kv_norm,
            attn_sink: &self.attn_sink,
            compressor,
            indexer,
        }
    }
}


// ── Metrics / buckets ───────────────────────────────────────────────────────

/// Comparison metrics between GPU tensor `a` and reference `b`.
///
/// - `max_abs` / `mean_rel` / `l2_rel`: global elementwise error
/// - `in_amax`: max |b_i| (reference magnitude — the scale absolute error sits on)
/// - `gpu_amax`: max |a_i|
/// - `row_max_abs`: per-row max |a-b| length = rows (full distribution, not sampled)
#[derive(Clone, Debug)]
struct CmpMetrics {
    max_abs: f64,
    mean_rel: f64,
    l2_rel: f64,
    in_amax: f64,
    gpu_amax: f64,
    /// Per-row max abs error (length = rows).
    row_max_abs: Vec<f64>,
}

struct LayerReport {
    layer: usize,
    ratio: usize,
    scope: String,
    // full-layer
    max_abs: f64,
    mean_rel: f64,
    l2_rel: f64,
    in_amax: f64,
    gpu_amax: f64,
    row_max_abs: Vec<f64>,
    buckets: Vec<(usize, usize, f64, f64, f64)>,
    // attn-half (residual_hc)
    rhc_max_abs: f64,
    rhc_mean_rel: f64,
    rhc_l2_rel: f64,
    rhc_in_amax: f64,
    rhc_row_max_abs: Vec<f64>,
    // ffn-half
    ffn_max_abs: f64,
    ffn_mean_rel: f64,
    ffn_l2_rel: f64,
    ffn_in_amax: f64,
    ffn_row_max_abs: Vec<f64>,
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
    in_amax: f64,
}

fn metrics(a: &[f32], b: &[f32], rows: usize, width: usize) -> CmpMetrics {
    assert_eq!(a.len(), b.len(), "metrics length mismatch");
    assert_eq!(a.len(), rows * width, "metrics geometry mismatch");
    let mut max_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut n_rel = 0usize;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut in_amax = 0.0f64;
    let mut gpu_amax = 0.0f64;
    let mut row_max_abs = vec![0.0f64; rows];
    for r in 0..rows {
        let mut rmax = 0.0f64;
        for c in 0..width {
            let i = r * width + c;
            let x = a[i] as f64;
            let y = b[i] as f64;
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
            if d > rmax {
                rmax = d;
            }
            let ay = y.abs();
            if ay > in_amax {
                in_amax = ay;
            }
            let ax = x.abs();
            if ax > gpu_amax {
                gpu_amax = ax;
            }
            if ay > 1e-8 {
                sum_rel += d / ay;
                n_rel += 1;
            }
            num += (x - y) * (x - y);
            den += y * y;
        }
        row_max_abs[r] = rmax;
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
    CmpMetrics {
        max_abs,
        mean_rel,
        l2_rel,
        in_amax,
        gpu_amax,
        row_max_abs,
    }
}

fn argmax_row(row_max: &[f64]) -> (f64, usize) {
    let mut best = 0.0f64;
    let mut arg = 0usize;
    for (i, &v) in row_max.iter().enumerate() {
        if v > best {
            best = v;
            arg = i;
        }
    }
    (best, arg)
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

/// Full per-row max_abs distribution summary (not a sampled subset).
fn format_row_dist(row_max: &[f64]) -> String {
    if row_max.is_empty() {
        return "empty".into();
    }
    let mut sorted = row_max.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let p50 = sorted[n / 2];
    let p90 = sorted[(n * 9 / 10).min(n - 1)];
    let p99 = sorted[(n * 99 / 100).min(n - 1)];
    let global = sorted[n - 1];
    let mut argmax = 0usize;
    for (i, &v) in row_max.iter().enumerate() {
        if v == global {
            argmax = i;
            break;
        }
    }
    format!("p50={p50:.3e} p90={p90:.3e} p99={p99:.3e} max={global:.3e}@r{argmax}")
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
    max_layers: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = DEFAULT_MODEL.to_owned();
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut rows = DEFAULT_ROWS;
    let mut max_layers = None;
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
            "--max-layers" => {
                max_layers = Some(
                    args.get(i + 1)
                        .ok_or("--max-layers needs a value")?
                        .parse()
                        .map_err(|e| format!("--max-layers: {e}"))?,
                );
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
        max_layers,
    })
}
