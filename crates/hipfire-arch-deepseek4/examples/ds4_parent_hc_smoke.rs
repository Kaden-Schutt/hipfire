// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 4 HC / RMSNorm smoke: GPU parent path vs f64 `layer_ref` oracle.
//!
//! Runs on gfx942 (mi300x). Compares at `rows=16, hc_mult=4, dim=4096` and
//! also checks Sinkhorn doubly-stochastic property on real layer-0 HC params.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_hc_smoke \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```

use hipfire_arch_deepseek4::parent::hc::{
    parent_hc_head, parent_hc_post, parent_hc_pre, parent_rms_norm, ParentHcParams,
};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::{
    hc_head_ref, hc_post_ref, hc_pre_ref, hc_split_sinkhorn_ref, rms_norm_ref,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const ROWS: usize = 16;
const HC_MULT: usize = 4;
const DIM: usize = 4096;
const NORM_EPS: f32 = 1e-6;
const HC_EPS: f32 = 1e-6;
const SINKHORN_ITERS: i32 = 20;
/// Mean relative-error ceiling consistent with f32 round-off.
const REL_TOL: f64 = 1e-5;
/// Absolute ceiling: K=16384 f32 GEMM noise is ~sqrt(K)*2^-24 ≈ 1e-5 on O(1).
const ABS_TOL: f64 = 1e-4;
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
    let model = parse_model_arg();
    let model_path = Path::new(&model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }

    println!("=== ds4_parent_hc_smoke ===");
    println!("model: {}", model_path.display());
    println!("shape: rows={ROWS} hc_mult={HC_MULT} dim={DIM}");
    println!("norm_eps={NORM_EPS} hc_eps={HC_EPS} sinkhorn_iters={SINKHORN_ITERS}");

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
        "admit OK: layers={} (loading layer 0 only, experts=false)",
        cfg.num_hidden_layers
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..1,
        load_experts: false,
    };
    let t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!(
        "loaded layer 0 (no experts) in {:.3}s  resident={:.3} GiB",
        t0.elapsed().as_secs_f64(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let layer = &weights.layers[0];
    assert_eq!(layer.layer_idx, 0);

    // ── Synthetic multi-stream activations (deterministic, BF16-ish) ──
    let hc_dim = HC_MULT * DIM;
    let mut x_host = vec![0.0f32; ROWS * hc_dim];
    for r in 0..ROWS {
        for h in 0..HC_MULT {
            for d in 0..DIM {
                let v = (((r * 131 + h * 97 + d * 17) % 400) as f32 - 200.0) * 0.01;
                x_host[r * hc_dim + h * DIM + d] = v;
            }
        }
    }

    // Download real HC params from layer 0 (F32 resident).
    let hc_fn = gpu
        .download_f32(&layer.hc_attn_fn)
        .map_err(|e| format!("download hc_attn_fn: {e:?}"))?;
    let hc_base = gpu
        .download_f32(&layer.hc_attn_base)
        .map_err(|e| format!("download hc_attn_base: {e:?}"))?;
    let hc_scale = gpu
        .download_f32(&layer.hc_attn_scale)
        .map_err(|e| format!("download hc_attn_scale: {e:?}"))?;
    let hc_head_fn = gpu
        .download_f32(&weights.hc_head_fn)
        .map_err(|e| format!("download hc_head_fn: {e:?}"))?;
    let hc_head_base = gpu
        .download_f32(&weights.hc_head_base)
        .map_err(|e| format!("download hc_head_base: {e:?}"))?;
    let hc_head_scale = gpu
        .download_f32(&weights.hc_head_scale)
        .map_err(|e| format!("download hc_head_scale: {e:?}"))?;

    println!(
        "hc_attn_scale = {:?}\nhc_attn_base[0..8] = {:?}",
        hc_scale,
        &hc_base[..8.min(hc_base.len())]
    );
    println!(
        "hc_head_scale = {:?}  hc_head_base = {:?}",
        hc_head_scale, hc_head_base
    );

    // ── Sinkhorn doubly-stochastic check on real params ──────────────
    {
        // Build mixes the same way hc_pre does for row 0 only, then run ref sinkhorn.
        let mix_hc = (2 + HC_MULT) * HC_MULT;
        let mut mixes0 = vec![0.0f32; mix_hc];
        let x0 = &x_host[..hc_dim];
        let mut acc = 0.0f64;
        for &v in x0 {
            acc += (v as f64) * (v as f64);
        }
        let rsqrt = (acc / hc_dim as f64 + NORM_EPS as f64).sqrt().recip();
        for o in 0..mix_hc {
            let mut s = 0.0f64;
            let wbase = o * hc_dim;
            for k in 0..hc_dim {
                s += (x0[k] as f64) * (hc_fn[wbase + k] as f64);
            }
            mixes0[o] = (s * rsqrt) as f32;
        }
        let (_pre, _post, comb) = hc_split_sinkhorn_ref(
            &mixes0,
            &hc_scale,
            &hc_base,
            1,
            HC_MULT,
            SINKHORN_ITERS as usize,
            HC_EPS as f64,
        )?;
        let (row_max_dev, col_max_dev) = doubly_stochastic_devs(&comb, HC_MULT);
        println!(
            "sinkhorn doubly-stochastic (layer0 attn, row0): max|row_sum-1|={row_max_dev:.3e}  max|col_sum-1|={col_max_dev:.3e}"
        );
        // Last op is column-normalize, so cols ≈ 1; rows stay slightly off
        // because every row pass divides by (sum + eps). With hc_eps=1e-6 and
        // hc=4 this is typically O(1e-2..1e-1) on real layer-0 comb logits —
        // report, don't fail-closed unless pathologically broken.
        if col_max_dev > 1e-3 || row_max_dev > 0.5 {
            return Err(format!(
                "deepseek4 parent: sinkhorn far from doubly-stochastic (row_dev={row_max_dev} col_dev={col_max_dev})"
            ));
        }
    }

    // ── Upload activations ───────────────────────────────────────────
    let x = upload_f32(&mut gpu, &x_host, &[ROWS, HC_MULT, DIM])?;
    let y = zeros(&mut gpu, &[ROWS, DIM])?;
    let post = zeros(&mut gpu, &[ROWS, HC_MULT])?;
    let comb = zeros(&mut gpu, &[ROWS, HC_MULT, HC_MULT])?;
    let out_post = zeros(&mut gpu, &[ROWS, HC_MULT, DIM])?;
    let y_head = zeros(&mut gpu, &[ROWS, DIM])?;
    let y_norm = zeros(&mut gpu, &[ROWS, DIM])?;

    let p = ParentHcParams {
        fn_mat: &layer.hc_attn_fn,
        base: &layer.hc_attn_base,
        scale: &layer.hc_attn_scale,
    };

    // ── hc_pre ───────────────────────────────────────────────────────
    let t_pre = Instant::now();
    parent_hc_pre(
        &mut gpu,
        backend,
        &x,
        p,
        ROWS,
        HC_MULT,
        DIM,
        NORM_EPS,
        SINKHORN_ITERS,
        HC_EPS,
        &y,
        &post,
        &comb,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync after hc_pre: {e:?}"))?;
    let pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
    let y_gpu = download(&gpu, &y)?;
    let post_gpu = download(&gpu, &post)?;
    let comb_gpu = download(&gpu, &comb)?;

    let t_ref = Instant::now();
    let (y_ref, post_ref, comb_ref) = hc_pre_ref(
        &x_host,
        &hc_fn,
        &hc_scale,
        &hc_base,
        ROWS,
        HC_MULT,
        DIM,
        NORM_EPS as f64,
        SINKHORN_ITERS as usize,
        HC_EPS as f64,
    )?;
    let ref_pre_ms = t_ref.elapsed().as_secs_f64() * 1000.0;
    report("hc_pre.y", &y_gpu, &y_ref)?;
    report("hc_pre.post", &post_gpu, &post_ref)?;
    report("hc_pre.comb", &comb_gpu, &comb_ref)?;
    println!("  wall: gpu={pre_ms:.2} ms  cpu_ref={ref_pre_ms:.1} ms");

    // Comb doubly-stochastic on GPU output too.
    {
        let (rd, cd) = max_doubly_stochastic_devs_batched(&comb_gpu, ROWS, HC_MULT);
        println!("  gpu comb doubly-stochastic: max|row-1|={rd:.3e} max|col-1|={cd:.3e}");
    }

    // ── hc_post ──────────────────────────────────────────────────────
    // Use a synthetic transform output (what attn/ffn would produce).
    let mut x_trans = vec![0.0f32; ROWS * DIM];
    for r in 0..ROWS {
        for d in 0..DIM {
            x_trans[r * DIM + d] = (((r * 41 + d * 3) % 100) as f32 - 50.0) * 0.02;
        }
    }
    let x_t = upload_f32(&mut gpu, &x_trans, &[ROWS, DIM])?;
    let t_post = Instant::now();
    parent_hc_post(
        &mut gpu,
        backend,
        &x_t,
        &x, // residual = original multi-stream x
        &post,
        &comb,
        ROWS,
        HC_MULT,
        DIM,
        &out_post,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync after hc_post: {e:?}"))?;
    let post_ms = t_post.elapsed().as_secs_f64() * 1000.0;
    let out_gpu = download(&gpu, &out_post)?;
    let out_ref = hc_post_ref(&x_trans, &x_host, &post_gpu, &comb_gpu, ROWS, HC_MULT, DIM);
    report("hc_post.out", &out_gpu, &out_ref)?;
    println!("  wall: gpu={post_ms:.2} ms");

    // ── hc_head ──────────────────────────────────────────────────────
    let p_head = ParentHcParams {
        fn_mat: &weights.hc_head_fn,
        base: &weights.hc_head_base,
        scale: &weights.hc_head_scale,
    };
    let t_head = Instant::now();
    parent_hc_head(
        &mut gpu,
        backend,
        &x,
        p_head,
        ROWS,
        HC_MULT,
        DIM,
        NORM_EPS,
        HC_EPS,
        &y_head,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync after hc_head: {e:?}"))?;
    let head_ms = t_head.elapsed().as_secs_f64() * 1000.0;
    let head_gpu = download(&gpu, &y_head)?;
    let head_ref = hc_head_ref(
        &x_host,
        &hc_head_fn,
        &hc_head_scale,
        &hc_head_base,
        ROWS,
        HC_MULT,
        DIM,
        NORM_EPS as f64,
        HC_EPS as f64,
    )?;
    report("hc_head.y", &head_gpu, &head_ref)?;
    println!("  wall: gpu={head_ms:.2} ms");

    // Head must differ from pre (different path / weights).
    {
        let mut max_diff = 0.0f64;
        for (&a, &b) in head_gpu.iter().zip(y_gpu.iter()) {
            max_diff = max_diff.max((a as f64 - b as f64).abs());
        }
        println!("  |hc_head.y - hc_pre.y|_max = {max_diff:.6} (must be > 0)");
        if max_diff < 1e-8 {
            return Err(
                "deepseek4 parent: hc_head and hc_pre produced identical y — paths collapsed"
                    .to_owned(),
            );
        }
    }

    // ── rms_norm with real BF16 attn_norm weight ─────────────────────
    // Use hc_pre.y as the norm input (single-stream [rows, dim]).
    let w_bf16_bytes = {
        let n = DIM * 2;
        let mut b = vec![0u8; n];
        gpu.hip
            .memcpy_dtoh(&mut b, &layer.attn_norm.buf)
            .map_err(|e| format!("download attn_norm: {e:?}"))?;
        b
    };
    let mut w_f32 = vec![0.0f32; DIM];
    for i in 0..DIM {
        let bits = u16::from_le_bytes([w_bf16_bytes[2 * i], w_bf16_bytes[2 * i + 1]]);
        w_f32[i] = f32::from_bits((bits as u32) << 16);
    }
    // y already holds the gpu buffer used as rms_norm input.
    let t_norm = Instant::now();
    parent_rms_norm(
        &mut gpu,
        backend,
        &y,
        &layer.attn_norm,
        &y_norm,
        ROWS,
        DIM,
        NORM_EPS,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync after rms_norm: {e:?}"))?;
    let norm_ms = t_norm.elapsed().as_secs_f64() * 1000.0;
    let norm_gpu = download(&gpu, &y_norm)?;
    let norm_ref = rms_norm_ref(&y_gpu, &w_f32, NORM_EPS as f64, DIM);
    report("rms_norm.out", &norm_gpu, &norm_ref)?;
    println!("  wall: gpu={norm_ms:.2} ms");

    // Cleanup
    for t in [x, y, post, comb, out_post, y_head, y_norm, x_t] {
        let _ = gpu.free_tensor(t);
    }

    println!();
    println!("ALL HC CHECKS PASSED (mean_rel<={REL_TOL:.0e}, max_abs<={ABS_TOL:.0e})");
    Ok(())
}

fn parse_model_arg() -> String {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--model") {
        if let Some(p) = args.get(i + 1) {
            return p.clone();
        }
    }
    args.into_iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned())
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}

fn zeros(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: zeros: {e:?}"))
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>, String> {
    gpu.download_f32(t)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))
}

/// Returns (max_abs, max_rel, mean_abs, mean_rel, l2_rel).
fn rel_stats(gpu: &[f32], refer: &[f32]) -> (f64, f64, f64, f64, f64) {
    assert_eq!(gpu.len(), refer.len());
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
        // Floor the denom so near-zero reference values don't explode max_rel.
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

fn report(name: &str, gpu: &[f32], refer: &[f32]) -> Result<(), String> {
    let (max_abs, max_rel, mean_abs, mean_rel, l2_rel) = rel_stats(gpu, refer);
    println!(
        "{name}: max_abs={max_abs:.6e} max_rel={max_rel:.6e} mean_abs={mean_abs:.6e} mean_rel={mean_rel:.6e} l2_rel={l2_rel:.6e}  n={}",
        gpu.len()
    );
    // Pass when both absolute and mean/L2 relative errors are at f32 noise.
    // max_rel alone is not decisive: near-zero reference elements inflate it
    // even when abs error is ~1e-6 (observed on hc_pre.y).
    let pass = max_abs <= ABS_TOL && mean_rel <= REL_TOL && l2_rel <= REL_TOL * 10.0;
    if !pass {
        let mut worst: Vec<(usize, f64, f32, f32)> = Vec::new();
        for (i, (&g, &r)) in gpu.iter().zip(refer.iter()).enumerate() {
            let abs = ((g as f64) - (r as f64)).abs();
            if worst.len() < 5 {
                worst.push((i, abs, g, r));
                worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            } else if abs > worst[4].1 {
                worst[4] = (i, abs, g, r);
                worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        }
        eprintln!("  worst absolute offenders for {name}:");
        for (i, abs, g, r) in &worst {
            eprintln!("    [{i}] gpu={g} ref={r} abs={abs:.3e}");
        }
        return Err(format!(
            "deepseek4 parent: {name} failed agreement \
             (max_abs={max_abs:.3e} mean_rel={mean_rel:.3e} l2_rel={l2_rel:.3e}; \
             tol abs={ABS_TOL:.0e} mean_rel={REL_TOL:.0e})"
        ));
    }
    Ok(())
}

fn doubly_stochastic_devs(comb: &[f32], hc: usize) -> (f64, f64) {
    let mut row_max = 0.0f64;
    let mut col_max = 0.0f64;
    for j in 0..hc {
        let mut rs = 0.0f64;
        let mut cs = 0.0f64;
        for k in 0..hc {
            rs += comb[j * hc + k] as f64;
            cs += comb[k * hc + j] as f64;
        }
        row_max = row_max.max((rs - 1.0).abs());
        col_max = col_max.max((cs - 1.0).abs());
    }
    (row_max, col_max)
}

fn max_doubly_stochastic_devs_batched(comb: &[f32], rows: usize, hc: usize) -> (f64, f64) {
    let mut row_max = 0.0f64;
    let mut col_max = 0.0f64;
    for r in 0..rows {
        let base = r * hc * hc;
        let (rd, cd) = doubly_stochastic_devs(&comb[base..base + hc * hc], hc);
        row_max = row_max.max(rd);
        col_max = col_max.max(cd);
    }
    (row_max, col_max)
}
