// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 of the DS4 parent-checkpoint calibration path: 16-token one-layer
//! canary of the ORIGINAL mixed-precision parent checkpoint.
//!
//! Composes `parent_layer_forward_traced` over real layer 0 (compress_ratio==0,
//! hash-routed) and cross-checks every stage that has an f64 CPU oracle in
//! `parent::layer_ref`. Finiteness alone is not the gate.
//!
//! Usage:
//! ```text
//! ds4_parent_layer_gate --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!                       [--layer 0] [--rows 16] [--manifest PATH] \
//!                       [--skip-shard-hashes]
//! ```
//!
//! Must run on gfx942 (mi300x).

use hipfire_arch_deepseek4::parent::attention::{
    PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_RMS_EPS, PARENT_SWA_WINDOW,
};
use hipfire_arch_deepseek4::parent::forward::{
    parent_layer_forward_traced, ParentForwardScratch, ParentLayerTrace, PARENT_HC_DIM,
    PARENT_HC_EPS, PARENT_HC_MULT, PARENT_HC_SINKHORN_ITERS,
};
use hipfire_arch_deepseek4::parent::hc::{parent_hc_pre, parent_rms_norm, ParentHcParams};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::{
    expert_swiglu_ref, gate_hash_ref, hc_post_ref, hc_pre_ref, rms_norm_ref,
};
use hipfire_arch_deepseek4::parent::linear::{parent_linear_dense, parent_linear_expert};
use hipfire_arch_deepseek4::parent::manifest::{
    sha256_file, CaptureBoundary, CaptureInfo, ModelInfo, ModelQuantInfo, ParentManifest,
    ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_arch_deepseek4::parent::moe::{
    parent_route, PARENT_MOE_INTER, PARENT_ROUTE_SCALE, PARENT_SWIGLU_LIMIT,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_LAYER: usize = 0;
const DEFAULT_ROWS: usize = 16;
/// Deterministic PRNG seed for token-id selection (printed).
const TOKEN_SEED: u64 = 0xD5_46_A7_E4_04_6A_7E_u64;
/// Embed table rows (checkpoint `embed.weight` shape[0]).
const VOCAB: usize = 129_280;
/// Mean relative-error ceiling consistent with f32 round-off (Gate 4).
const MEAN_REL_TOL: f64 = 1e-5;
/// Absolute ceiling for near-zero reference elements.
const ABS_TOL: f64 = 1e-4;
/// Stage-to-stage L2 ratio outside this band is flagged degenerate.
///
/// Note: RMSNorm with the checkpoint's small direct-multiply weights
/// (attn_norm mean ≈ 0.03) legitimately drops L2 by ~17×. That is NOT
/// collapse — the closed-form check below catches real offset bugs.
/// Band is therefore wide; the RMSNorm predicted-vs-measured assertion
/// is the precise check for the norm stages.
const NORM_RATIO_LO: f64 = 1e-3;
const NORM_RATIO_HI: f64 = 1e3;
/// RMSNorm predicted L2 must match measured within this relative tolerance.
/// `L2 ≈ sqrt(rows*dim) * mean(|w|)` assumes roughly uniform |w|; a few
/// percent covers the real non-uniform weight distribution.
const RMSNORM_PRED_TOL: f64 = 0.05;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

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
            "deepseek4 parent: --model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }

    let layer_idx = args.layer;
    let rows = args.rows;
    let start_pos = 0usize;

    println!("=== ds4_parent_layer_gate (Gate 4) ===");
    println!("model: {}", model_path.display());
    println!("layer: {layer_idx}  rows: {rows}  start_pos: {start_pos}");
    println!("token_seed: {TOKEN_SEED:#x}");
    println!("skip_shard_hashes: {}", args.skip_shard_hashes);
    if let Some(m) = args.manifest.as_ref() {
        println!("manifest: {}", m.display());
    }
    println!();

    // ── 1. Admit + inventory + load ─────────────────────────────────────
    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;

    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err(
            "deepseek4 parent: gfx942 required (parent calibration is fail-closed)"
                .to_owned(),
        );
    }
    println!("gpu: gfx942");

    let admit_t0 = Instant::now();
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    let admit_ms = admit_t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "admit OK ({admit_ms:.1} ms): layers={} hash_layers={} n_routed={} topk={} \
         compress_ratios[{layer_idx}]={}",
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.compress_ratio(layer_idx)
    );
    if cfg.compress_ratio(layer_idx) != 0 {
        return Err(format!(
            "deepseek4 parent: Gate 4 targets compress_ratio==0; layer {layer_idx} has {}",
            cfg.compress_ratio(layer_idx)
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: layer_idx..(layer_idx + 1),
        load_experts: true,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    let res = weights.residency();
    println!(
        "loaded layers={:?} experts={} in {load_s:.3} s",
        weights.layer_range, weights.experts_loaded
    );
    println!(
        "residency: total={:.3} GiB  dense_bf16={:.3} expert={:.3} bf16={:.3} f32={:.3} i64={:.3}",
        res.total_bytes() as f64 / GIB,
        res.dense_bf16_bytes as f64 / GIB,
        res.expert_compressed_bytes as f64 / GIB,
        res.bf16_bytes as f64 / GIB,
        res.f32_bytes as f64 / GIB,
        res.i64_bytes as f64 / GIB,
    );

    let layer = &weights.layers[0];
    if layer.layer_idx != layer_idx {
        return Err(format!(
            "deepseek4 parent: loaded layer_idx {} != requested {layer_idx}",
            layer.layer_idx
        ));
    }
    if layer.experts.len() != cfg.n_routed_experts {
        return Err(format!(
            "deepseek4 parent: experts.len() {} != n_routed_experts {}",
            layer.experts.len(),
            cfg.n_routed_experts
        ));
    }

    // ── 2. Deterministic in-distribution activations from embed rows ────
    let token_ids = select_token_ids(TOKEN_SEED, rows);
    print!("token_ids[{rows}] = [");
    for (i, &t) in token_ids.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{t}");
    }
    println!("]");

    let embed_host = download_bf16_as_f32(&gpu, &weights.embed, VOCAB * PARENT_DIM)?;
    // Single-stream embed rows, then expand to hc_mult copies (model.py:914-916).
    let mut x_single = vec![0.0f32; rows * PARENT_DIM];
    for (r, &tid) in token_ids.iter().enumerate() {
        let src = (tid as usize) * PARENT_DIM;
        x_single[r * PARENT_DIM..(r + 1) * PARENT_DIM]
            .copy_from_slice(&embed_host[src..src + PARENT_DIM]);
    }
    let mut x_hc = vec![0.0f32; rows * PARENT_HC_DIM];
    for r in 0..rows {
        let row = &x_single[r * PARENT_DIM..(r + 1) * PARENT_DIM];
        for h in 0..PARENT_HC_MULT {
            let dst = (r * PARENT_HC_MULT + h) * PARENT_DIM;
            x_hc[dst..dst + PARENT_DIM].copy_from_slice(row);
        }
    }
    let x_l2 = l2_norm(&x_hc);
    let x_abs_max = x_hc.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    println!(
        "input x [rows,hc,dim]=[{rows},{},{}]  L2={x_l2:.6}  max|x|={x_abs_max:.6}  \
         (embed rows repeated across hc streams)",
        PARENT_HC_MULT, PARENT_DIM
    );

    let x = upload_f32(&mut gpu, &x_hc, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let out = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let kv_ring = zeros_f32(
        &mut gpu,
        &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
    )?;

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, rows)?;
    println!(
        "ParentForwardScratch::bytes() = {} ({:.3} MiB)  max_rows={}",
        scratch.bytes(),
        scratch.bytes() as f64 / (1024.0 * 1024.0),
        scratch.max_rows()
    );

    // ── 3. Traced full-layer forward ────────────────────────────────────
    let mut trace = ParentLayerTrace::default();
    let fwd_t0 = Instant::now();
    parent_layer_forward_traced(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &mut scratch,
        layer_idx,
        &x,
        rows,
        start_pos,
        Some(&token_ids),
        &kv_ring,
        &out,
        &mut trace,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync after layer: {e:?}"))?;
    let fwd_ms = fwd_t0.elapsed().as_secs_f64() * 1000.0;
    println!("parent_layer_forward_traced wall = {fwd_ms:.2} ms");

    let out_host = download_f32(&gpu, &out, rows * PARENT_HC_DIM)?;
    let (n_nan, n_inf, n_zero, out_l2, out_abs_max) = finite_stats(&out_host);
    let zero_frac = n_zero as f64 / out_host.len() as f64;
    println!();
    println!("=== stage norms (from ParentLayerTrace) ===");
    let stage_names = [
        "hc_pre_attn",
        "attn_norm",
        "attn_out",
        "hc_post_attn",
        "hc_pre_ffn",
        "ffn_norm",
        "moe_out",
        "hc_post_ffn",
    ];
    let stage_norms = [
        trace.hc_pre_attn,
        trace.attn_norm,
        trace.attn_out,
        trace.hc_post_attn,
        trace.hc_pre_ffn,
        trace.ffn_norm,
        trace.moe_out,
        trace.hc_post_ffn,
    ];
    for (name, n) in stage_names.iter().zip(stage_norms.iter()) {
        println!("  {name:<14} L2 = {n:.6}");
    }
    println!(
        "  {:<14} L2 = {out_l2:.6}  (downloaded out; should match hc_post_ffn)",
        "out"
    );
    println!();
    println!("=== finiteness ===");
    println!(
        "out: nan={n_nan} inf={n_inf} exact_zero={n_zero}/{} ({zero_frac:.4})  \
         max|out|={out_abs_max:.6}",
        out_host.len()
    );

    // ── 4. Degeneracy: stage-to-stage norm ratios ───────────────────────
    //
    // RMSNorm with direct-multiply weights (model.py:197-202, NO +1 offset)
    // forces per-row RMS≈1, so post-norm L2 ≈ sqrt(rows*dim)*mean(|w|).
    // Real layer-0 attn_norm mean|w|≈0.0295 → predicted L2≈7.55; the drop
    // from hc_pre_attn≈132 to attn_norm≈7.6 is therefore CORRECT, not a
    // collapse. Flag only true zeros / explosions / non-finite.
    println!();
    println!("=== degeneracy (stage-to-stage L2 ratios) ===");
    let chain: Vec<(&str, f64)> = vec![
        ("input_hc", x_l2 as f64),
        ("hc_pre_attn", stage_norms[0] as f64),
        ("attn_norm", stage_norms[1] as f64),
        ("attn_out", stage_norms[2] as f64),
        ("hc_post_attn", stage_norms[3] as f64),
        ("hc_pre_ffn", stage_norms[4] as f64),
        ("ffn_norm", stage_norms[5] as f64),
        ("moe_out", stage_norms[6] as f64),
        ("hc_post_ffn", stage_norms[7] as f64),
    ];
    let mut degen_flags: Vec<String> = Vec::new();
    for w in chain.windows(2) {
        let (a_name, a) = w[0];
        let (b_name, b) = w[1];
        let ratio = if a > 0.0 { b / a } else { f64::INFINITY };
        let flag = if !a.is_finite() || !b.is_finite() {
            "NONFINITE"
        } else if a < 1e-12 {
            "PRIOR_COLLAPSE"
        } else if b < 1e-6 {
            // Absolute floor: a stage that lands near zero is broken even if
            // the prior was also small. RMSNorm stages land O(1..100).
            "COLLAPSE"
        } else if ratio < NORM_RATIO_LO {
            "COLLAPSE"
        } else if ratio > NORM_RATIO_HI {
            "EXPLODE"
        } else {
            "ok"
        };
        println!("  {a_name} → {b_name}: ratio={ratio:.6e}  [{flag}]");
        if flag != "ok" {
            degen_flags.push(format!("{a_name}->{b_name}:{flag}"));
        }
    }
    println!(
        "  zero_frac(out) = {zero_frac:.6}  {}",
        if zero_frac > 0.5 {
            "[COLLAPSE-ish: >50% exact zeros]"
        } else {
            "[ok]"
        }
    );
    if zero_frac > 0.5 {
        degen_flags.push(format!("zero_frac={zero_frac:.4}"));
    }

    // ── 4b. Closed-form RMSNorm L2 prediction ───────────────────────────
    // Reference RMSNorm is `weight * (x * rsqrt(var+eps))` with NO +1 offset
    // (model.py:197-202). Confirmed on real checkpoint: attn_norm weights
    // are tightly clustered positive around ~0.03 (an offset representation
    // would center near 0 and include negatives). With per-row RMS forced
    // to 1, ||out||_2 ≈ sqrt(rows*dim) * mean(|w|).
    println!();
    println!("=== RMSNorm closed-form L2 (predicted vs measured) ===");
    let attn_norm_w = download_bf16_as_f32(&gpu, &layer.attn_norm, PARENT_DIM)?;
    let ffn_norm_w = download_bf16_as_f32(&gpu, &layer.ffn_norm, PARENT_DIM)?;
    let rmsnorm_scale = ((rows * PARENT_DIM) as f64).sqrt(); // = 256 at 16×4096
    let attn_w_mean = mean_abs(&attn_norm_w);
    let ffn_w_mean = mean_abs(&ffn_norm_w);
    let attn_pred = rmsnorm_scale * attn_w_mean;
    let ffn_pred = rmsnorm_scale * ffn_w_mean;
    let attn_meas = stage_norms[1] as f64;
    let ffn_meas = stage_norms[5] as f64;
    let attn_rel = (attn_meas - attn_pred).abs() / attn_pred.max(1e-12);
    let ffn_rel = (ffn_meas - ffn_pred).abs() / ffn_pred.max(1e-12);
    // Weight sign / range sanity: direct-multiply trained weights are all > 0.
    let attn_neg = attn_norm_w.iter().filter(|&&v| v < 0.0).count();
    let ffn_neg = ffn_norm_w.iter().filter(|&&v| v < 0.0).count();
    let attn_w_min = attn_norm_w.iter().cloned().fold(f32::INFINITY, f32::min);
    let attn_w_max = attn_norm_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ffn_w_min = ffn_norm_w.iter().cloned().fold(f32::INFINITY, f32::min);
    let ffn_w_max = ffn_norm_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  sqrt(rows*dim) = {rmsnorm_scale:.4}  (rows={rows} dim={})",
        PARENT_DIM
    );
    println!(
        "  attn_norm.weight: mean|w|={attn_w_mean:.6}  range=[{attn_w_min:.5}, {attn_w_max:.5}]  \
         n_neg={attn_neg}"
    );
    println!(
        "  ffn_norm.weight:  mean|w|={ffn_w_mean:.6}  range=[{ffn_w_min:.5}, {ffn_w_max:.5}]  \
         n_neg={ffn_neg}"
    );
    println!(
        "  attn_norm L2: predicted={attn_pred:.4}  measured={attn_meas:.4}  \
         rel_err={attn_rel:.4e}  (tol={RMSNORM_PRED_TOL})"
    );
    println!(
        "  ffn_norm  L2: predicted={ffn_pred:.4}  measured={ffn_meas:.4}  \
         rel_err={ffn_rel:.4e}  (tol={RMSNORM_PRED_TOL})"
    );
    // Note the expected ~17× drop across attn_norm is weight-driven, not collapse.
    if stage_norms[0] > 0.0 {
        let drop = stage_norms[1] / stage_norms[0];
        println!(
            "  attn_norm / hc_pre_attn = {drop:.6}  \
             (≈ mean|w_attn|={attn_w_mean:.4}; expected scale-down, not collapse)"
        );
    }

    // ── 5. Oracle cross-checks ──────────────────────────────────────────
    // After a full forward the FFN half's intermediates survive in scratch
    // (stream_y / stream_normed / stream_block / post / comb / residual_hc /
    // out). The attention half's post/comb were overwritten — re-run attn
    // hc_pre + rms_norm against the same x for those oracles.
    println!();
    println!("=== oracle cross-checks (f64 layer_ref) ===");
    let mut checks: Vec<CheckRow> = Vec::new();

    // Closed-form RMSNorm assertions enter the check table.
    {
        let attn_pass = attn_rel <= RMSNORM_PRED_TOL
            && attn_neg == 0
            && attn_w_min > 0.0
            && attn_meas.is_finite();
        checks.push(CheckRow {
            name: "attn_norm_closed_form".into(),
            max_abs: (attn_meas - attn_pred).abs(),
            mean_rel: attn_rel,
            l2_rel: attn_rel,
            pass: attn_pass,
            detail: format!(
                "pred={attn_pred:.4} meas={attn_meas:.4} mean|w|={attn_w_mean:.5} n_neg={attn_neg}"
            ),
        });
        let ffn_pass = ffn_rel <= RMSNORM_PRED_TOL
            && ffn_neg == 0
            && ffn_w_min > 0.0
            && ffn_meas.is_finite();
        checks.push(CheckRow {
            name: "ffn_norm_closed_form".into(),
            max_abs: (ffn_meas - ffn_pred).abs(),
            mean_rel: ffn_rel,
            l2_rel: ffn_rel,
            pass: ffn_pass,
            detail: format!(
                "pred={ffn_pred:.4} meas={ffn_meas:.4} mean|w|={ffn_w_mean:.5} n_neg={ffn_neg}"
            ),
        });
        println!(
            "  attn_norm_closed_form: {}",
            if attn_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  ffn_norm_closed_form:  {}",
            if ffn_pass { "PASS" } else { "FAIL" }
        );
    }

    let mix_hc = (2 + PARENT_HC_MULT) * PARENT_HC_MULT;
    let hc_attn_fn = download_f32(&gpu, &layer.hc_attn_fn, mix_hc * PARENT_HC_DIM)?;
    let hc_attn_base = download_f32(&gpu, &layer.hc_attn_base, mix_hc)?;
    let hc_attn_scale = download_f32(&gpu, &layer.hc_attn_scale, 3)?;
    let hc_ffn_fn = download_f32(&gpu, &layer.hc_ffn_fn, mix_hc * PARENT_HC_DIM)?;
    let hc_ffn_base = download_f32(&gpu, &layer.hc_ffn_base, mix_hc)?;
    let hc_ffn_scale = download_f32(&gpu, &layer.hc_ffn_scale, 3)?;

    // 5a. Attn-half hc_pre + rms_norm — re-run GPU path into fresh tiles so
    // we still have the intermediates (full forward overwrote them).
    {
        let y = zeros_f32(&mut gpu, &[rows, PARENT_DIM])?;
        let post = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT])?;
        let comb = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_HC_MULT])?;
        let y_norm = zeros_f32(&mut gpu, &[rows, PARENT_DIM])?;
        let p = ParentHcParams {
            fn_mat: &layer.hc_attn_fn,
            base: &layer.hc_attn_base,
            scale: &layer.hc_attn_scale,
        };
        parent_hc_pre(
            &mut gpu,
            backend,
            &x,
            p,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
            PARENT_RMS_EPS,
            PARENT_HC_SINKHORN_ITERS,
            PARENT_HC_EPS,
            &y,
            &post,
            &comb,
        )?;
        let y_gpu = download_f32(&gpu, &y, rows * PARENT_DIM)?;
        let post_gpu = download_f32(&gpu, &post, rows * PARENT_HC_MULT)?;
        let comb_gpu = download_f32(&gpu, &comb, rows * PARENT_HC_MULT * PARENT_HC_MULT)?;
        let (y_ref, post_ref, comb_ref) = hc_pre_ref(
            &x_hc,
            &hc_attn_fn,
            &hc_attn_scale,
            &hc_attn_base,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
            PARENT_RMS_EPS as f64,
            PARENT_HC_SINKHORN_ITERS as usize,
            PARENT_HC_EPS as f64,
        )?;
        checks.push(check_stage("hc_pre_attn.y", &y_gpu, &y_ref));
        checks.push(check_stage("hc_pre_attn.post", &post_gpu, &post_ref));
        checks.push(check_stage("hc_pre_attn.comb", &comb_gpu, &comb_ref));

        parent_rms_norm(
            &mut gpu,
            backend,
            &y,
            &layer.attn_norm,
            &y_norm,
            rows,
            PARENT_DIM,
            PARENT_RMS_EPS,
        )?;
        let norm_gpu = download_f32(&gpu, &y_norm, rows * PARENT_DIM)?;
        let norm_ref = rms_norm_ref(&y_gpu, &attn_norm_w, PARENT_RMS_EPS as f64, PARENT_DIM);
        checks.push(check_stage("attn_norm", &norm_gpu, &norm_ref));

        for t in [y, post, comb, y_norm] {
            let _ = gpu.free_tensor(t);
        }
    }

    // 5b. FFN-half intermediates still resident after the traced forward.
    {
        let residual_hc = download_f32(&gpu, scratch.residual_hc(), rows * PARENT_HC_DIM)?;
        let stream_y = download_f32(&gpu, scratch.stream_y(), rows * PARENT_DIM)?;
        let stream_normed = download_f32(&gpu, scratch.stream_normed(), rows * PARENT_DIM)?;
        let stream_block = download_f32(&gpu, scratch.stream_block(), rows * PARENT_DIM)?;
        let post = download_f32(&gpu, scratch.post(), rows * PARENT_HC_MULT)?;
        let comb = download_f32(
            &gpu,
            scratch.comb(),
            rows * PARENT_HC_MULT * PARENT_HC_MULT,
        )?;

        // residual_hc is the attn-half hc_post output = FFN hc_pre input.
        let (y_ref, post_ref, comb_ref) = hc_pre_ref(
            &residual_hc,
            &hc_ffn_fn,
            &hc_ffn_scale,
            &hc_ffn_base,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
            PARENT_RMS_EPS as f64,
            PARENT_HC_SINKHORN_ITERS as usize,
            PARENT_HC_EPS as f64,
        )?;
        checks.push(check_stage("hc_pre_ffn.y", &stream_y, &y_ref));
        checks.push(check_stage("hc_pre_ffn.post", &post, &post_ref));
        checks.push(check_stage("hc_pre_ffn.comb", &comb, &comb_ref));

        let norm_ref = rms_norm_ref(&stream_y, &ffn_norm_w, PARENT_RMS_EPS as f64, PARENT_DIM);
        checks.push(check_stage("ffn_norm", &stream_normed, &norm_ref));

        // hc_post_ffn: out = post * moe_out + comb @ residual_hc
        let out_ref = hc_post_ref(
            &stream_block,
            &residual_hc,
            &post,
            &comb,
            rows,
            PARENT_HC_MULT,
            PARENT_DIM,
        );
        checks.push(check_stage("hc_post_ffn", &out_host, &out_ref));

        let (rn, ri, rz, rl2, rmax) = finite_stats(&residual_hc);
        println!(
            "  residual_hc (post-attn HC state): nan={rn} inf={ri} zero={rz} \
             L2={rl2:.6} max|v|={rmax:.6}"
        );
        if rn > 0 || ri > 0 {
            checks.push(CheckRow {
                name: "residual_hc_finite".into(),
                max_abs: f64::NAN,
                mean_rel: f64::NAN,
                l2_rel: f64::NAN,
                pass: false,
                detail: format!("nan={rn} inf={ri}"),
            });
        }
    }

    // 5c. Routing — layer 0 is hash-routed. Indices from tid2eid; weights
    // still come from unbiased sqrtsoftplus scores (parent_route).
    {
        let moe_x = scratch.moe_x_bf16();
        let routing = parent_route(
            &mut gpu,
            backend,
            layer,
            &cfg,
            moe_x,
            rows,
            Some(&token_ids),
        )?;
        let tid2eid = layer.tid2eid.as_ref().ok_or_else(|| {
            format!("deepseek4 parent: layer {layer_idx} is hash-routed but tid2eid missing")
        })?;
        let topk = cfg.num_experts_per_tok;
        let tid_bytes = {
            let n = VOCAB * topk * 8;
            let mut b = vec![0u8; n];
            if tid2eid.buf.size() < n {
                return Err(format!(
                    "deepseek4 parent: tid2eid short (have {} need {n})",
                    tid2eid.buf.size()
                ));
            }
            gpu.hip
                .memcpy_dtoh(&mut b, &tid2eid.buf)
                .map_err(|e| format!("deepseek4 parent: tid2eid dtoh: {e:?}"))?;
            b
        };
        let mut tid_i64 = vec![0i64; VOCAB * topk];
        for i in 0..tid_i64.len() {
            let mut le = [0u8; 8];
            le.copy_from_slice(&tid_bytes[i * 8..i * 8 + 8]);
            tid_i64[i] = i64::from_le_bytes(le);
        }
        let hash_oracle = gate_hash_ref(&token_ids, &tid_i64, cfg.n_routed_experts, topk)?;
        let mut idx_mismatch = 0usize;
        for i in 0..routing.indices.len() {
            if routing.indices[i] != hash_oracle.indices[i] {
                idx_mismatch += 1;
            }
        }
        // gate_hash_ref returns uniform 1/topk; live path gathers real scores
        // then L1-norms * route_scale. Indices vs hash oracle; weight sum per
        // row must equal route_scale (1.5).
        let mut weight_sum_bad = 0usize;
        let mut max_wsum_err = 0.0f32;
        for r in 0..rows {
            let mut s = 0.0f32;
            for t in 0..topk {
                s += routing.weights[r * topk + t];
            }
            let err = (s - PARENT_ROUTE_SCALE).abs();
            max_wsum_err = max_wsum_err.max(err);
            if err > 1e-4 {
                weight_sum_bad += 1;
            }
        }
        let route_pass = idx_mismatch == 0 && weight_sum_bad == 0;
        println!(
            "  route/hash: idx_mismatch={idx_mismatch}/{}  weight_sum_err_max={max_wsum_err:.3e}  \
             distinct_experts={}  route_scale={}",
            routing.indices.len(),
            routing.distinct_experts(),
            PARENT_ROUTE_SCALE
        );
        println!(
            "    first-row indices={:?} weights={:?}",
            &routing.indices[..topk],
            &routing.weights[..topk]
        );
        checks.push(CheckRow {
            name: "route_hash_indices".into(),
            max_abs: idx_mismatch as f64,
            mean_rel: max_wsum_err as f64,
            l2_rel: 0.0,
            pass: route_pass,
            detail: format!(
                "idx_mismatch={idx_mismatch} wsum_bad_rows={weight_sum_bad} \
                 distinct={}",
                routing.distinct_experts()
            ),
        });

        // 5d. expert_swiglu on the shared expert (gate/up via public dense linears).
        {
            let gate_t = zeros_f32(&mut gpu, &[rows, PARENT_MOE_INTER])?;
            let up_t = zeros_f32(&mut gpu, &[rows, PARENT_MOE_INTER])?;
            let x_bf_bytes = download_bytes(&gpu, moe_x, rows * PARENT_DIM * 2)?;
            let act1 = {
                let t = gpu
                    .alloc_tensor(&[rows, PARENT_DIM], DType::BF16)
                    .map_err(|e| format!("deepseek4 parent: swiglu act1: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&t.buf, &x_bf_bytes)
                    .map_err(|e| format!("deepseek4 parent: swiglu act1 htod: {e:?}"))?;
                t
            };
            parent_linear_dense(
                &mut gpu,
                backend,
                &layer.shared_w1,
                &act1,
                rows,
                &gate_t,
            )?;
            let act2 = {
                let t = gpu
                    .alloc_tensor(&[rows, PARENT_DIM], DType::BF16)
                    .map_err(|e| format!("deepseek4 parent: swiglu act2: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&t.buf, &x_bf_bytes)
                    .map_err(|e| format!("deepseek4 parent: swiglu act2 htod: {e:?}"))?;
                t
            };
            parent_linear_dense(
                &mut gpu,
                backend,
                &layer.shared_w3,
                &act2,
                rows,
                &up_t,
            )?;
            let gate = download_f32(&gpu, &gate_t, rows * PARENT_MOE_INTER)?;
            let up = download_f32(&gpu, &up_t, rows * PARENT_MOE_INTER)?;
            let swiglu = expert_swiglu_ref(
                &gate,
                &up,
                rows,
                PARENT_MOE_INTER,
                PARENT_SWIGLU_LIMIT as f64,
                None,
            );
            let mut clamp_gate = 0usize;
            let mut clamp_up = 0usize;
            for i in 0..gate.len() {
                if gate[i] > PARENT_SWIGLU_LIMIT {
                    clamp_gate += 1;
                }
                if up[i] > PARENT_SWIGLU_LIMIT || up[i] < -PARENT_SWIGLU_LIMIT {
                    clamp_up += 1;
                }
            }
            let sw_l2 = l2_norm(&swiglu);
            let sw_finite = swiglu.iter().all(|v| v.is_finite());
            // Contrast against a WRONG symmetric clamp: when any gate value is
            // < -limit the asymmetric path must differ.
            let mut naive = vec![0.0f32; gate.len()];
            for i in 0..gate.len() {
                let mut g = gate[i] as f64;
                let mut u = up[i] as f64;
                let lim = PARENT_SWIGLU_LIMIT as f64;
                if u > lim {
                    u = lim;
                } else if u < -lim {
                    u = -lim;
                }
                if g > lim {
                    g = lim;
                } else if g < -lim {
                    g = -lim;
                }
                let sig = 1.0 / (1.0 + (-g).exp());
                naive[i] = (g * sig * u) as f32;
            }
            let mut asym_diff = 0.0f64;
            let mut asym_count = 0usize;
            for i in 0..gate.len() {
                let d = (swiglu[i] as f64 - naive[i] as f64).abs();
                if d > 1e-8 {
                    asym_count += 1;
                    asym_diff = asym_diff.max(d);
                }
            }
            let neg_gate = gate.iter().filter(|&&g| g < -PARENT_SWIGLU_LIMIT).count();
            let pass = sw_finite && sw_l2 > 1e-6 && (neg_gate == 0 || asym_count > 0);
            println!(
                "  expert_swiglu (shared): L2={sw_l2:.6} finite={sw_finite} \
                 clamp_hits gate_hi={clamp_gate} up_both={clamp_up} \
                 neg_gate_below_limit={neg_gate} asym_diff_elems={asym_count} \
                 max|asym-sym|={asym_diff:.3e}"
            );
            checks.push(CheckRow {
                name: "expert_swiglu_shared".into(),
                max_abs: asym_diff,
                mean_rel: 0.0,
                l2_rel: 0.0,
                pass,
                detail: format!(
                    "L2={sw_l2:.4} clamp_g={clamp_gate} clamp_u={clamp_up} \
                     neg_gate={neg_gate} asym_elems={asym_count}"
                ),
            });

            // One routed expert, row 0, with its route weight applied inside swiglu.
            if let Some(&eid) = routing.indices.first() {
                let eid = eid as usize;
                let expert = &layer.experts[eid];
                let w_elems = PARENT_MOE_INTER * PARENT_DIM;
                let w_bf16 = gpu
                    .alloc_tensor(&[w_elems], DType::BF16)
                    .map_err(|e| format!("deepseek4 parent: expert w scratch: {e:?}"))?;
                expert
                    .w1
                    .decode_into(&mut gpu, &w_bf16)
                    .map_err(|e| format!("deepseek4 parent: expert w1 decode: {e}"))?;
                let mut tok0 = vec![0u8; PARENT_DIM * 2];
                tok0.copy_from_slice(&x_bf_bytes[..PARENT_DIM * 2]);
                let act = {
                    let t = gpu
                        .alloc_tensor(&[1, PARENT_DIM], DType::BF16)
                        .map_err(|e| format!("deepseek4 parent: expert act: {e:?}"))?;
                    gpu.hip
                        .memcpy_htod(&t.buf, &tok0)
                        .map_err(|e| format!("deepseek4 parent: expert act htod: {e:?}"))?;
                    t
                };
                let g1 = zeros_f32(&mut gpu, &[1, PARENT_MOE_INTER])?;
                parent_linear_expert(
                    &mut gpu,
                    backend,
                    &w_bf16,
                    expert.w1.n(),
                    expert.w1.k(),
                    &act,
                    1,
                    &g1,
                )?;
                expert
                    .w3
                    .decode_into(&mut gpu, &w_bf16)
                    .map_err(|e| format!("deepseek4 parent: expert w3 decode: {e}"))?;
                let act3 = {
                    let t = gpu
                        .alloc_tensor(&[1, PARENT_DIM], DType::BF16)
                        .map_err(|e| format!("deepseek4 parent: expert act3: {e:?}"))?;
                    gpu.hip
                        .memcpy_htod(&t.buf, &tok0)
                        .map_err(|e| format!("deepseek4 parent: expert act3 htod: {e:?}"))?;
                    t
                };
                let u1 = zeros_f32(&mut gpu, &[1, PARENT_MOE_INTER])?;
                parent_linear_expert(
                    &mut gpu,
                    backend,
                    &w_bf16,
                    expert.w3.n(),
                    expert.w3.k(),
                    &act3,
                    1,
                    &u1,
                )?;
                let g = download_f32(&gpu, &g1, PARENT_MOE_INTER)?;
                let u = download_f32(&gpu, &u1, PARENT_MOE_INTER)?;
                let rw = [routing.weights[0]];
                let sw = expert_swiglu_ref(
                    &g,
                    &u,
                    1,
                    PARENT_MOE_INTER,
                    PARENT_SWIGLU_LIMIT as f64,
                    Some(&rw),
                );
                let sw_l2 = l2_norm(&sw);
                let sw_fin = sw.iter().all(|v| v.is_finite());
                let sw_u = expert_swiglu_ref(
                    &g,
                    &u,
                    1,
                    PARENT_MOE_INTER,
                    PARENT_SWIGLU_LIMIT as f64,
                    None,
                );
                let mut max_scale_err = 0.0f64;
                for i in 0..sw.len() {
                    let expect = sw_u[i] as f64 * rw[0] as f64;
                    max_scale_err = max_scale_err.max((sw[i] as f64 - expect).abs());
                }
                let pass = sw_fin && sw_l2 > 0.0 && max_scale_err < 1e-5;
                println!(
                    "  expert_swiglu (routed eid={eid} row0 w={:.5}): L2={sw_l2:.6} \
                     finite={sw_fin} max|w*unweighted - weighted|={max_scale_err:.3e}",
                    rw[0]
                );
                checks.push(CheckRow {
                    name: format!("expert_swiglu_routed_{eid}"),
                    max_abs: max_scale_err,
                    mean_rel: 0.0,
                    l2_rel: 0.0,
                    pass,
                    detail: format!("L2={sw_l2:.4} scale_err={max_scale_err:.3e}"),
                });
                for t in [w_bf16, act, act3, g1, u1] {
                    let _ = gpu.free_tensor(t);
                }
            }

            for t in [gate_t, up_t, act1, act2] {
                let _ = gpu.free_tensor(t);
            }
        }
    }

    // ── 6. PASS/FAIL table ──────────────────────────────────────────────
    println!();
    println!("=== Gate 4 check table ===");
    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>6}  detail",
        "stage", "max_abs", "mean_rel", "l2_rel", "verdict"
    );
    let mut all_pass = true;
    let fin_pass = n_nan == 0 && n_inf == 0;
    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>6}  nan={n_nan} inf={n_inf}",
        "output_finite",
        "-",
        "-",
        "-",
        if fin_pass { "PASS" } else { "FAIL" }
    );
    if !fin_pass {
        all_pass = false;
    }
    let degen_pass = degen_flags.is_empty();
    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>6}  {}",
        "stage_norm_ratios",
        "-",
        "-",
        "-",
        if degen_pass { "PASS" } else { "FAIL" },
        if degen_flags.is_empty() {
            "all ratios in band".into()
        } else {
            degen_flags.join("; ")
        }
    );
    if !degen_pass {
        all_pass = false;
    }
    let trace_out_err = ((trace.hc_post_ffn as f64) - (out_l2 as f64)).abs()
        / (out_l2 as f64).max(1e-12);
    let trace_pass = trace_out_err < 1e-5 && stage_norms.iter().all(|n| n.is_finite());
    println!(
        "{:<28} {:>12.3e} {:>12} {:>12} {:>6}  |trace.hc_post_ffn - ||out|||/||out||",
        "trace_norms_finite",
        trace_out_err,
        "-",
        "-",
        if trace_pass { "PASS" } else { "FAIL" }
    );
    if !trace_pass {
        all_pass = false;
    }

    for c in &checks {
        let verdict = if c.pass { "PASS" } else { "FAIL" };
        if !c.pass {
            all_pass = false;
        }
        println!(
            "{:<28} {:>12.3e} {:>12.3e} {:>12.3e} {:>6}  {}",
            c.name, c.max_abs, c.mean_rel, c.l2_rel, verdict, c.detail
        );
    }

    // ── 7. Manifest (optional) ──────────────────────────────────────────
    if let Some(path) = args.manifest.as_ref() {
        println!();
        println!("=== manifest ===");
        let (producer, engine) = ParentManifest::probe_environment("gfx942")?;
        let source_info = build_source_info(model_path, args.skip_shard_hashes)?;
        let manifest = ParentManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            produced_utc: utc_now_rfc3339(),
            producer,
            engine,
            source: source_info,
            model: ModelInfo {
                model_type: cfg.model_type.clone(),
                num_hidden_layers: cfg.num_hidden_layers,
                mtp_loaded: false,
                rope_convention: "yarn".to_string(),
                quant: ModelQuantInfo {
                    quant_method: cfg.quant_method.clone(),
                    fmt: cfg.fmt.clone(),
                    scale_fmt: cfg.scale_fmt.clone(),
                    expert_dtype: cfg.expert_dtype.clone(),
                    weight_block_size: cfg.weight_block_size,
                },
            },
            // No corpus consumed — canary drives synthetic embed rows. null
            // corpus is allowed only when there are no outputs and no captured
            // activations (both true here).
            corpus: None,
            capture: CaptureInfo {
                boundary: CaptureBoundary::PostDynamicFp8,
                tensors: Vec::new(),
            },
            outputs: Vec::new(),
        };
        manifest
            .validate()
            .map_err(|e| format!("deepseek4 parent: manifest.validate failed: {e}"))?;
        println!("manifest.validate(): OK (null corpus, no outputs)");
        manifest.write_to(path)?;
        println!("wrote {}", path.display());
    }

    let _ = gpu.free_tensor(x);
    let _ = gpu.free_tensor(out);
    let _ = gpu.free_tensor(kv_ring);

    println!();
    println!("=== Gate 4 summary ===");
    println!(
        "load:            {load_s:.3} s  resident={:.3} GiB",
        res.total_bytes() as f64 / GIB
    );
    println!("forward:         {fwd_ms:.2} ms");
    println!(
        "output:          L2={out_l2:.6}  finite={}  zero_frac={zero_frac:.4}",
        fin_pass
    );
    println!(
        "oracle checks:   {}/{} PASS",
        checks.iter().filter(|c| c.pass).count(),
        checks.len()
    );
    println!(
        "degeneracy:      {}",
        if degen_pass { "PASS" } else { "FAIL" }
    );
    let sane = fin_pass
        && degen_pass
        && out_l2 > 1.0
        && out_l2 < 1.0e6
        && stage_norms[2] > 1.0 // attn_out
        && stage_norms[6] > 1.0; // moe_out
    println!(
        "interpretation:  {}",
        if sane {
            "layer behaves sanely — finite, non-collapsed norms, attn/moe produce \
             O(1..1e3) energy; RMSNorm L2 matches closed-form mean|w|*sqrt(N)"
        } else if !fin_pass {
            "NON-FINITE output — defect in a composed sub-block"
        } else if !degen_pass {
            "DEGENERATE stage-norm trajectory — a sub-block is zeroing or exploding"
        } else {
            "numeric/oracle disagreement — see FAIL rows above"
        }
    );
    if all_pass {
        println!("GATE 4: PASS");
        Ok(())
    } else {
        Err("deepseek4 parent: GATE 4 FAIL — see check table".to_owned())
    }
}

// ── Check helpers ───────────────────────────────────────────────────────────

struct CheckRow {
    name: String,
    max_abs: f64,
    mean_rel: f64,
    l2_rel: f64,
    pass: bool,
    detail: String,
}

fn check_stage(name: &str, gpu: &[f32], refer: &[f32]) -> CheckRow {
    assert_eq!(
        gpu.len(),
        refer.len(),
        "check_stage {name}: len mismatch {} vs {}",
        gpu.len(),
        refer.len()
    );
    let (max_abs, _max_rel, _mean_abs, mean_rel, l2_rel) = rel_stats(gpu, refer);
    // Pass criterion: mean relative error at f32 noise, absolute error bounded.
    // max_rel alone is not decisive (near-zero ref elements inflate it).
    let pass = max_abs.is_finite()
        && mean_rel.is_finite()
        && l2_rel.is_finite()
        && max_abs <= ABS_TOL
        && mean_rel <= MEAN_REL_TOL
        && l2_rel <= MEAN_REL_TOL * 10.0;
    let detail = format!("n={}", gpu.len());
    println!(
        "  {name}: max_abs={max_abs:.6e} mean_rel={mean_rel:.6e} l2_rel={l2_rel:.6e}  {}",
        if pass { "PASS" } else { "FAIL" }
    );
    CheckRow {
        name: name.to_owned(),
        max_abs,
        mean_rel,
        l2_rel,
        pass,
        detail,
    }
}

fn rel_stats(gpu: &[f32], refer: &[f32]) -> (f64, f64, f64, f64, f64) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let n = gpu.len() as f64;
    for (&g, &r) in gpu.iter().zip(refer.iter()) {
        let g = g as f64;
        let r = r as f64;
        let abs = (g - r).abs();
        let denom = r.abs().max(1e-3);
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        sum_abs += abs;
        sum_rel += rel;
        sum_sq_err += abs * abs;
        sum_sq_ref += r * r;
    }
    let l2_rel = sum_sq_err.sqrt() / sum_sq_ref.sqrt().max(1e-12);
    (max_abs, max_rel, sum_abs / n, sum_rel / n, l2_rel)
}

fn finite_stats(v: &[f32]) -> (usize, usize, usize, f32, f32) {
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_zero = 0usize;
    let mut sum_sq = 0.0f64;
    let mut abs_max = 0.0f32;
    for &x in v {
        if x.is_nan() {
            n_nan += 1;
        } else if x.is_infinite() {
            n_inf += 1;
        } else {
            if x == 0.0 {
                n_zero += 1;
            }
            sum_sq += (x as f64) * (x as f64);
            abs_max = abs_max.max(x.abs());
        }
    }
    (n_nan, n_inf, n_zero, sum_sq.sqrt() as f32, abs_max)
}

fn l2_norm(v: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for &x in v {
        if x.is_finite() {
            s += (x as f64) * (x as f64);
        }
    }
    s.sqrt() as f32
}

fn mean_abs(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for &x in v {
        s += (x as f64).abs();
    }
    s / v.len() as f64
}

// ── Token selection ─────────────────────────────────────────────────────────

/// SplitMix64-derived deterministic token ids in `[0, VOCAB)`.
fn select_token_ids(seed: u64, n: usize) -> Vec<u32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    // Mix in a few "real" low ids so the hash table and embed rows are
    // exercised on both common and mid-vocab tokens.
    let fixed = [0u32, 1, 2, 7, 42, 256, 1000, 50256];
    for &t in fixed.iter().take(n.min(fixed.len())) {
        out.push(t % VOCAB as u32);
    }
    while out.len() < n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push((z % VOCAB as u64) as u32);
    }
    out
}

// ── IO helpers ──────────────────────────────────────────────────────────────

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: zeros: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download_f32 short (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))?;
    Ok(data)
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

fn download_bytes(gpu: &Gpu, t: &GpuTensor, nbytes: usize) -> Result<Vec<u8>, String> {
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download_bytes short (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut b = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut b, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_bytes: {e:?}"))?;
    Ok(b)
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    layer: usize,
    rows: usize,
    manifest: Option<PathBuf>,
    skip_shard_hashes: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut layer = DEFAULT_LAYER;
    let mut rows = DEFAULT_ROWS;
    let mut manifest: Option<PathBuf> = None;
    let mut skip_shard_hashes = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "flag --model missing value".to_string())?,
                );
            }
            "--layer" => {
                let v = args
                    .next()
                    .ok_or_else(|| "flag --layer missing value".to_string())?;
                layer = v.parse().map_err(|e| format!("--layer: {e}"))?;
            }
            "--rows" => {
                let v = args
                    .next()
                    .ok_or_else(|| "flag --rows missing value".to_string())?;
                rows = v.parse().map_err(|e| format!("--rows: {e}"))?;
                if rows == 0 {
                    return Err("--rows must be > 0".into());
                }
            }
            "--manifest" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --manifest missing value".to_string())?;
                manifest = Some(PathBuf::from(p));
            }
            "--skip-shard-hashes" => skip_shard_hashes = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_layer_gate --model <dir> \
                     [--layer 0] [--rows 16] [--manifest out/manifest.json] \
                     [--skip-shard-hashes]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    Ok(Args {
        model,
        layer,
        rows,
        manifest,
        skip_shard_hashes,
    })
}

// ── Manifest source pinning (mirrors inventory gate) ────────────────────────

fn build_source_info(root: &Path, skip_shard_hashes: bool) -> Result<SourceInfo, String> {
    let root_str = root
        .to_str()
        .ok_or_else(|| "deepseek4 parent: model root is not valid UTF-8".to_string())?
        .to_string();

    let config_path = root.join("config.json");
    let index_path = root.join("model.safetensors.index.json");
    let tokenizer_path = root.join("tokenizer.json");

    let config_sha256 = sha256_file(&config_path)?;
    let index_sha256 = if index_path.is_file() {
        sha256_file(&index_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing index file {}",
            index_path.display()
        ));
    };
    let tokenizer_sha256 = if tokenizer_path.is_file() {
        sha256_file(&tokenizer_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing tokenizer.json at {}",
            tokenizer_path.display()
        ));
    };

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("deepseek4 parent: read_dir {}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err("deepseek4 parent: no .safetensors shards found".into());
    }

    let hash_t0 = Instant::now();
    let mut shards = Vec::with_capacity(shard_paths.len());
    for (i, p) in shard_paths.iter().enumerate() {
        let meta = std::fs::metadata(p)
            .map_err(|e| format!("deepseek4 parent: metadata {}: {e}", p.display()))?;
        let bytes = meta.len();
        let file = p
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("deepseek4 parent: bad shard name {}", p.display()))?
            .to_string();
        let sha256 = if skip_shard_hashes {
            format!("SKIPPED_SHARD_HASH_{i:02}")
        } else {
            let t0 = Instant::now();
            let h = sha256_file(p)?;
            println!(
                "  hashed {file} ({bytes} bytes) in {:.1} s → {}",
                t0.elapsed().as_secs_f64(),
                &h[..16]
            );
            h
        };
        shards.push(ShardInfo {
            file,
            sha256,
            bytes,
        });
    }
    if !skip_shard_hashes {
        println!(
            "hashed {} shards in {:.1} s",
            shards.len(),
            hash_t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "SKIPPED hashing {} shards (--skip-shard-hashes); placeholders are not a pin",
            shards.len()
        );
    }

    Ok(SourceInfo {
        root: root_str,
        index_sha256,
        shards,
        config_sha256,
        tokenizer_sha256,
    })
}

fn utc_now_rfc3339() -> String {
    if let Ok(out) = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}
