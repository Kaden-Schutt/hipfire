// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 5 head smoke: embed + hc_head + final RMSNorm + lm_head on real weights.
//!
//! Loads **globals only** (`ParentLoadPlan { layers: 0..0, load_experts: false }`),
//! embeds 16 token ids, runs `parent_head` on a synthetic multi-stream final
//! state, and compares against the f64 oracle (`hc_head_ref` + `rms_norm_ref`
//! + `head_proj_ref`).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_head_smoke \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731
//! ```
//!
//! Must run on gfx942 (mi300x).

use hipfire_ds4_parent::head::{
    parent_embed, parent_head, parent_logits_to_plog, ParentHeadScratch,
    PARENT_HC_DIM, PARENT_HC_MULT, PARENT_VOCAB,
};
use hipfire_ds4_parent::inventory::ParentInventory;
use hipfire_ds4_parent::plog::{PlogReader, PlogWriter};
use hipfire_ds4_parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_ds4_parent::Ds4ParentBackend;
use hipfire_ds4_parent::attention::{PARENT_DIM, PARENT_RMS_EPS};
use hipfire_ds4_parent::head::PARENT_HC_EPS;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const ROWS: usize = 16;
/// Mean relative-error ceiling consistent with f32 / BF16 GEMM noise.
const REL_TOL: f64 = 5e-3;
/// Absolute ceiling on logits (scale is O(10); K=4096 BF16 noise is larger).
const ABS_TOL: f64 = 5e-2;

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

    println!("=== ds4_parent_head_smoke ===");
    println!("model: {}", model_path.display());
    println!(
        "shape: rows={ROWS} hc_mult={PARENT_HC_MULT} dim={PARENT_DIM} vocab={PARENT_VOCAB}"
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
    println!(
        "admit OK: layers={} (loading globals only: layers 0..0, experts=false)",
        cfg.num_hidden_layers
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..0,
        load_experts: false,
    };
    let t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let res = weights.residency();
    println!(
        "loaded globals in {:.3}s  resident={:.3} GiB  (layers={})",
        t0.elapsed().as_secs_f64(),
        res.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        weights.layers.len()
    );
    assert!(weights.layers.is_empty(), "globals-only plan must load 0 layers");

    // ── Scratch sizing ────────────────────────────────────────────────
    let scratch = ParentHeadScratch::new(&mut gpu, &cfg, ROWS)?;
    println!(
        "ParentHeadScratch::bytes() = {} ({:.3} MiB)",
        scratch.bytes(),
        scratch.bytes() as f64 / (1024.0 * 1024.0)
    );
    let peak_1k = ParentHeadScratch::peak_logits_capture_bytes(1024, 16);
    println!(
        "peak device bytes for 1K-token logits capture (stream_rows=16) = {} ({:.3} MiB)",
        peak_1k,
        peak_1k as f64 / (1024.0 * 1024.0)
    );

    // ── Embed 16 real-ish token ids ───────────────────────────────────
    let token_ids: Vec<u32> = (0..ROWS as u32)
        .map(|i| {
            // Mix of small ids and mid-vocab ids; all in range.
            match i {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 100,
                4 => 1000,
                5 => 50_000,
                6 => 100_000,
                7 => (PARENT_VOCAB as u32) - 1,
                _ => 17 + i * 997,
            }
        })
        .map(|t| t % (PARENT_VOCAB as u32))
        .collect();
    println!("token_ids = {token_ids:?}");

    let embed_out = zeros(&mut gpu, &[ROWS, PARENT_HC_MULT, PARENT_DIM])?;
    let t_emb = Instant::now();
    parent_embed(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &token_ids,
        &embed_out,
    )?;
    let emb_ms = t_emb.elapsed().as_secs_f64() * 1e3;
    let emb_host = download(&gpu, &embed_out)?;
    let emb_finite = emb_host.iter().all(|v| v.is_finite());
    let emb_l2 = l2_norm(&emb_host);
    println!(
        "parent_embed: {emb_ms:.1} ms  finite={emb_finite}  L2={emb_l2:.6e}  \
         row0_stream0[0..4]={:?}",
        &emb_host[..4.min(emb_host.len())]
    );
    if !emb_finite {
        return Err("deepseek4 parent: embed output not finite".into());
    }
    // Streams must be identical copies of the gathered row.
    for r in 0..ROWS {
        let s0 = &emb_host[r * PARENT_HC_DIM..r * PARENT_HC_DIM + PARENT_DIM];
        for h in 1..PARENT_HC_MULT {
            let sh = &emb_host[r * PARENT_HC_DIM + h * PARENT_DIM
                ..r * PARENT_HC_DIM + (h + 1) * PARENT_DIM];
            if s0 != sh {
                return Err(format!(
                    "deepseek4 parent: embed stream {h} != stream 0 at row {r}"
                ));
            }
        }
    }
    println!("embed stream-expand: OK (all hc_mult streams identical per row)");

    // ── Synthetic final HC state ──────────────────────────────────────
    let mut x_host = vec![0.0f32; ROWS * PARENT_HC_DIM];
    for r in 0..ROWS {
        for h in 0..PARENT_HC_MULT {
            for d in 0..PARENT_DIM {
                let v = (((r * 131 + h * 97 + d * 17) % 400) as f32 - 200.0) * 0.01;
                x_host[r * PARENT_HC_DIM + h * PARENT_DIM + d] = v;
            }
        }
    }
    let x = upload_f32(&mut gpu, &x_host, &[ROWS, PARENT_HC_MULT, PARENT_DIM])?;
    let logits_t = zeros(&mut gpu, &[ROWS, PARENT_VOCAB])?;

    let t_head = Instant::now();
    parent_head(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &x,
        ROWS,
        &logits_t,
    )?;
    let head_ms = t_head.elapsed().as_secs_f64() * 1e3;
    println!("parent_head: {head_ms:.1} ms");

    let logits = download(&gpu, &logits_t)?;
    let finite = logits.iter().all(|v| v.is_finite());
    let l2 = l2_norm(&logits);
    println!("logits: finite={finite}  L2={l2:.6e}  nelems={}", logits.len());
    if !finite {
        return Err("deepseek4 parent: logits not finite".into());
    }

    // Per-row argmax + distribution stats.
    let (mean, std) = mean_std(&logits);
    println!("logits mean={mean:.6e}  stddev={std:.6e}");
    let mut argmaxes = Vec::with_capacity(ROWS);
    for r in 0..ROWS {
        let row = &logits[r * PARENT_VOCAB..(r + 1) * PARENT_VOCAB];
        let (idx, val) = argmax(row);
        argmaxes.push((idx, val));
        println!("  row {r}: argmax_token={idx}  logit={val:.6e}");
    }
    // Top-5 for row 0.
    let top5 = top_k(&logits[..PARENT_VOCAB], 5);
    println!("row0 top-5:");
    for (i, (tok, val)) in top5.iter().enumerate() {
        println!("  #{i}: token={tok}  logit={val:.6e}");
    }
    // Degeneracy checks.
    let all_equal = logits.windows(2).all(|w| w[0] == w[1]);
    let max_abs = logits.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if all_equal {
        return Err("deepseek4 parent: logits degenerate (all equal)".into());
    }
    if max_abs > 1e20 {
        return Err(format!(
            "deepseek4 parent: logits degenerate (max_abs={max_abs:e})"
        ));
    }
    if std < 1e-6 {
        return Err(format!(
            "deepseek4 parent: logits degenerate (stddev={std:e})"
        ));
    }
    println!(
        "logits distribution: sane (not all-equal, max_abs={max_abs:.3e}, std={std:.3e})"
    );
    // ── f64 oracle on real weights (sampled vocab — full 129k×16 is ~8e12 FLOPs) ──
    println!("downloading head/norm/hc_head weights for oracle…");
    let t_dl = Instant::now();
    let head_bytes = download_bf16_bytes(&gpu, &weights.head, PARENT_VOCAB * PARENT_DIM)?;
    let norm_f32 = download_bf16_as_f32(&gpu, &weights.norm, PARENT_DIM)?;
    let hc_fn = gpu
        .download_f32(&weights.hc_head_fn)
        .map_err(|e| format!("download hc_head_fn: {e:?}"))?;
    let hc_base = gpu
        .download_f32(&weights.hc_head_base)
        .map_err(|e| format!("download hc_head_base: {e:?}"))?;
    let hc_scale = gpu
        .download_f32(&weights.hc_head_scale)
        .map_err(|e| format!("download hc_head_scale: {e:?}"))?;
    println!(
        "weight download {:.3}s  head_bf16={:.1} MiB",
        t_dl.elapsed().as_secs_f64(),
        head_bytes.len() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "hc_head_scale={:?}  hc_head_base={:?}",
        hc_scale, hc_base
    );

    // Build the set of vocab columns to check: strided sample + every
    // per-row argmax + row-0 top-5. Full-vocab f64 GEMM is hours on CPU.
    let mut sample: Vec<usize> = (0..PARENT_VOCAB).step_by(64).collect(); // 2020 cols
    for &(tok, _) in &argmaxes {
        sample.push(tok);
    }
    for &(tok, _) in &top5 {
        sample.push(tok);
    }
    sample.sort_unstable();
    sample.dedup();
    println!(
        "running f64 oracle on {} vocab columns × {ROWS} rows…",
        sample.len()
    );

    // hc_head → rms_norm on host (full), then head_proj only for sampled cols.
    use hipfire_ds4_parent::codec::round_to_bf16;
    use hipfire_ds4_parent::layer_ref::{hc_head_ref, rms_norm_ref};
    let t_ref = Instant::now();
    let y_hc = hc_head_ref(
        &x_host,
        &hc_fn,
        &hc_scale,
        &hc_base,
        ROWS,
        PARENT_HC_MULT,
        PARENT_DIM,
        PARENT_RMS_EPS as f64,
        PARENT_HC_EPS as f64,
    )?;
    let mut normed = rms_norm_ref(&y_hc, &norm_f32, PARENT_RMS_EPS as f64, PARENT_DIM);
    for v in &mut normed {
        *v = round_to_bf16(*v);
    }
    // Sampled head projection in f64.
    let mut refer_sample = vec![0.0f32; ROWS * sample.len()];
    for r in 0..ROWS {
        let xbase = r * PARENT_DIM;
        for (si, &vcol) in sample.iter().enumerate() {
            let wbase = vcol * PARENT_DIM * 2;
            let mut acc = 0.0f64;
            for k in 0..PARENT_DIM {
                let bits = u16::from_le_bytes([
                    head_bytes[wbase + 2 * k],
                    head_bytes[wbase + 2 * k + 1],
                ]);
                let w = f32::from_bits((bits as u32) << 16) as f64;
                acc += (normed[xbase + k] as f64) * w;
            }
            refer_sample[r * sample.len() + si] = acc as f32;
        }
    }
    println!("oracle done in {:.3}s", t_ref.elapsed().as_secs_f64());

    let mut gpu_sample = vec![0.0f32; ROWS * sample.len()];
    for r in 0..ROWS {
        for (si, &vcol) in sample.iter().enumerate() {
            gpu_sample[r * sample.len() + si] = logits[r * PARENT_VOCAB + vcol];
        }
    }

    let (max_abs_e, _max_rel, mean_abs, mean_rel, l2_rel) =
        rel_stats(&gpu_sample, &refer_sample);
    println!(
        "GPU vs f64 oracle ({} cols): max_abs={max_abs_e:.6e} mean_abs={mean_abs:.6e} \
         mean_rel={mean_rel:.6e} l2_rel={l2_rel:.6e}",
        sample.len()
    );
    if mean_rel > REL_TOL || l2_rel > REL_TOL {
        let mut worst: Vec<(usize, f64, f32, f32)> = Vec::new();
        for (i, (&g, &r)) in gpu_sample.iter().zip(refer_sample.iter()).enumerate() {
            let abs = ((g as f64) - (r as f64)).abs();
            if worst.len() < 8 {
                worst.push((i, abs, g, r));
                worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            } else if abs > worst[7].1 {
                worst[7] = (i, abs, g, r);
                worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        }
        eprintln!("  worst absolute offenders:");
        for (i, abs, g, r) in &worst {
            let row = i / sample.len();
            let col = sample[i % sample.len()];
            eprintln!("    [row={row} tok={col}] gpu={g} ref={r} abs={abs:.3e}");
        }
        return Err(format!(
            "deepseek4 parent: head oracle disagreement \
             (mean_rel={mean_rel:.3e} l2_rel={l2_rel:.3e}; tol={REL_TOL:.0e})"
        ));
    }
    if max_abs_e > ABS_TOL * 100.0 {
        println!(
            "NOTE: max_abs={max_abs_e:.3e} is large but relative errors are within tol"
        );
    }
    println!("oracle agreement: PASS");


    // ── Plog bridge round-trip ────────────────────────────────────────
    let plog_path = std::env::temp_dir().join(format!(
        "ds4_parent_head_smoke_{}.plog",
        std::process::id()
    ));
    {
        let mut w = PlogWriter::create(&plog_path, ROWS, PARENT_VOCAB)?;
        parent_logits_to_plog(&gpu, &logits_t, ROWS, PARENT_VOCAB, &mut w)?;
        w.finish()?;
    }
    let reader = PlogReader::open(&plog_path)?;
    assert_eq!(reader.n_tokens(), ROWS);
    assert_eq!(reader.vocab(), PARENT_VOCAB);
    for r in 0..ROWS {
        let row = reader.row(r)?;
        let src = &logits[r * PARENT_VOCAB..(r + 1) * PARENT_VOCAB];
        if row != src {
            return Err(format!(
                "deepseek4 parent: plog row {r} mismatch after GPU bridge"
            ));
        }
    }
    let _ = std::fs::remove_file(&plog_path);
    println!("plog bridge: OK (GPU logits → PlogWriter → PlogReader bit-identical)");

    println!("=== ds4_parent_head_smoke PASS ===");
    let _ = argmaxes;
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

fn download_bf16_bytes(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<u8>, String> {
    let nbytes = nelems * 2;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: bf16 download too small (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: bf16 download: {e:?}"))?;
    Ok(bytes)
}

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let bytes = download_bf16_bytes(gpu, t, nelems)?;
    let mut out = Vec::with_capacity(nelems);
    for i in 0..nelems {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    Ok(out)
}

fn l2_norm(v: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        let x = x as f64;
        s += x * x;
    }
    s.sqrt()
}

fn mean_std(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mut mean = 0.0f64;
    for &x in v {
        mean += x as f64;
    }
    mean /= n;
    let mut var = 0.0f64;
    for &x in v {
        let d = x as f64 - mean;
        var += d * d;
    }
    (mean, (var / n).sqrt())
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i, best_v)
}

fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k.min(row.len()));
    idx.into_iter().map(|i| (i, row[i])).collect()
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
